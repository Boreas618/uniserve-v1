//! Replicas and their roles, and the handoff of a request between two of them
//! (`specs/kv-cache.md` §2.1, §4.2).
//!
//! Every scheduler is a replica of the deployment, built to serve both phases; its role
//! (`replicas.<id>.role`, the step values' [`Tuning::role`]) decides which requests it takes
//! from the next submission on, and nothing else: a request it took serves to its end.
//!
//! - **Role.** A mixed replica takes every request. A prefill replica takes a request that
//!   hands off ([`GenerationRequest::hand_off`]), a decode replica a continuation: one that
//!   imports what a prefill replica lent ([`GenerationRequest::lend`]), or one another migration
//!   moved with the tokens it generated; a request of any other kind is rejected with
//!   [`RejectionKind::Role`], which the leader routes again.
//! - **The prefill side.** A request that hands off runs as any other until its prompt is
//!   computed and its first token sampled. Once no call of it is in flight it lends its owner's
//!   blocks (`Lend`), which the workers stamp as they stamp a `Detach`, and once the lend took
//!   effect it ends as migratable with the lend ([`MigrationCause::Handoff`]): the leader submits
//!   its continuation to the replica its route names for its tokens. The lend releases the owner
//!   when that replica's import returns it or its lease passes. A request that holds no blocks
//!   ends with no lend, and its continuation computes its prompt again.
//! - **The decode side.** A request that imports asks the deployment's KV program, through the
//!   engine's Storage Server, how its full pages arrive ([`Path`]), stating what this replica's
//!   tiers hold of them by its view of the cache ([`Output::Handoff`]), and its admission imports
//!   the pages as the path calls for. Which pages those are, and what the import asks of the
//!   buffer pool, are the KV decisions' (`kv/handoff.rs`).

use proto::request::{
    Event, GenerationRequest, Lend, MigrationCause, RejectionKind, Request as Input, RequestId,
};
use scheduling::kv::{Path, Tier};
use scheduling::routing::Role;
use storage::runtime::{self, Lender, OwnerId};

use super::kv::{self, Step};
use super::request::RequestStatus;
use super::{Output, Scheduler};

/// The reason a replica's role refuses a request, by what the request needs: a prefill replica
/// takes the requests that hand off, a decode replica the continuations, of a handoff (they name
/// a lend) or of another migration (they carry the tokens generated before).
fn refusal(role: Role, request: &GenerationRequest) -> Option<&'static str> {
    let continues = request.lend.is_some() || !request.generated_token_ids.is_empty();
    match role {
        Role::Mixed => None,
        Role::Prefill if request.hand_off => None,
        Role::Decode if !request.hand_off && continues => None,
        Role::Prefill => Some("a prefill replica computes prompts for other replicas alone"),
        Role::Decode => {
            Some("a decode replica continues requests whose prompts other replicas computed")
        }
    }
}

impl Scheduler {
    /// The role this scheduler serves under at this step.
    pub(super) fn role(&self) -> Role {
        self.tuning.load().role(self.index as usize)
    }

    /// Whether this scheduler's role takes `input`: `Err` with the rejection to answer it with.
    pub(super) fn takes(&self, input: &Input) -> Result<(), (RejectionKind, String)> {
        let Input::Generation(request) = input else {
            return Ok(());
        };
        match refusal(self.role(), request) {
            None => Ok(()),
            Some(reason) => Err((RejectionKind::Role, reason.to_owned())),
        }
    }

    /// Whether running request `id` lends its KV now: it hands off, its prompt is computed and a
    /// token sampled after it, it has not ended, and no call of it is in flight.
    pub(super) fn hands_off(&self, id: RequestId) -> bool {
        let Some(request) = self.requests.get(&id) else {
            return false;
        };
        request.status == RequestStatus::Running
            && request.terminal.is_none()
            && request.generation().is_some_and(|input| input.hand_off)
            && request.num_computed_tokens >= request.num_prompt_tokens
            && request.token_ids.len() as u32 > request.num_prompt_tokens
            && request.idle()
    }

    /// Lends the KV of every running request that hands off now ([`Self::hands_off`]): called
    /// once per pass of the loop.
    pub(super) fn start_handoffs(&mut self) {
        let due: Vec<RequestId> = self
            .running
            .iter()
            .copied()
            .filter(|&id| self.hands_off(id))
            .collect();
        for id in due {
            self.hand_off(id);
        }
    }

    /// Lends request `id`'s KV: its owner's `Lend`, stamped by the groups whose device block
    /// tables hold its entries. The request waits out of every batch until the lend's answer.
    pub(super) fn hand_off(&mut self, id: RequestId) {
        let Some(request) = self.requests.get(&id) else {
            return;
        };
        let Some(owner) = kv::kv_owner(request) else {
            return self.finish(id, handed(None));
        };
        let lease = u64::try_from(self.config.lease.as_nanos()).unwrap_or(u64::MAX);
        if let Some(request) = self.requests.get_mut(&id) {
            request.status = RequestStatus::WaitingForRemoteKvs;
            request.pending = 1;
        }
        self.submit(runtime::Request::Lend {
            owner,
            lease,
            after: Vec::new(),
        });
        self.mark(id);
    }

    /// The answer to request `id`'s `Lend` of `owner`, which took effect (a refusal names the
    /// scheduler's own defect and stops it before this): the request ends as migratable with the
    /// lend, which the owner leaves the request for.
    pub(super) fn lend_answered(&mut self, id: RequestId, owner: OwnerId) {
        let Some(request) = self.requests.get_mut(&id) else {
            return;
        };
        request.status = RequestStatus::Running;
        request.pending = 0;

        // The owner is the lend's: it is detached when the lend ends, not with the request.
        request
            .attachments
            .retain(|attachment| attachment.1 != owner);
        let tokens = request.num_prompt_tokens;
        let stored = self.stored_pages(id);
        self.owners.remove(&owner);
        self.drop_unmapped(owner);
        let lend = Lend {
            endpoint: self.config.storage.clone(),
            session: self.session,
            owner: owner.0,
            tokens,
            stored,
        };
        self.finish(id, handed(Some(lend)));
    }

    /// The leading full pages of request `id`'s prompt that `cache` holds in the remote store:
    /// what a decode replica may pull from there ([`kv::Kv::stored_pages`]).
    fn stored_pages(&self, id: RequestId) -> u32 {
        self.requests
            .get(&id)
            .map_or(0, |request| self.kv.stored_pages(request, &self.groups))
    }

    /// Asks the KV program how request `id`'s lent pages arrive, from what this replica's tiers
    /// hold of them as `cache` sees it ([`kv::Kv::handoff`]); the engine answers with
    /// [`Self::routed`]. A request that imports nothing (a model that keeps a recurrent state)
    /// takes a pull of what the tiers hold as its path.
    pub(super) fn ask_path(&mut self, id: RequestId) {
        let Some(request) = self.requests.get(&id) else {
            return;
        };
        let Some(lend) = request.generation().and_then(|input| input.lend.as_ref()) else {
            return;
        };
        let local = self.local(lend);
        match self.kv.handoff(request, lend, local, &self.groups) {
            Some(handoff) => self.emit(Output::Handoff(id, handoff)),
            None => self.routed(id, Path::Pull(Tier::Host)),
        }
    }

    /// Whether `lend` lies in this host's buffer pool: its Storage Server is this host's.
    fn local(&self, lend: &Lend) -> bool {
        lend.endpoint.is_none() || lend.endpoint == self.config.storage
    }

    /// The KV program chose `path` for request `id`'s lent pages: its admission goes on with the
    /// round it calls for.
    pub(super) fn routed(&mut self, id: RequestId, path: Path) {
        let Some(request) = self.requests.get_mut(&id) else {
            return;
        };
        if request.path.is_some() {
            return;
        }
        request.path = Some(path);
        // An admission already placed and attached waits for the path alone.
        let waiting = request.status == RequestStatus::WaitingForRemoteKvs && request.queued;
        if waiting {
            for ask in self.handoff_requests(id) {
                self.ask(id, ask);
            }
            self.answer_arrived(id);
        }
        self.mark(id);
    }

    /// The first round of the admission of request `id`, which imports a lend once its path is
    /// known ([`kv::Kv::handoff_requests`]), from the lender its lend names: this host's buffer
    /// pool, or the Storage Server of another.
    pub(super) fn handoff_requests(&mut self, id: RequestId) -> Vec<runtime::Request> {
        let Some(request) = self.requests.get(&id) else {
            return Vec::new();
        };
        let lender = request
            .generation()
            .and_then(|input| input.lend.as_ref())
            .map(|lend| Lender {
                endpoint: (!self.local(lend)).then(|| lend.endpoint.clone()).flatten(),
                session: lend.session,
                owner: OwnerId(lend.owner),
            });
        let step = Step {
            groups: &self.groups,
            config: &self.config,
            budget: &self.budget,
            prefetch_min_tokens: self.policy.limits.prefetch_min_tokens,
        };
        let Some(request) = self.requests.get_mut(&id) else {
            return Vec::new();
        };
        self.kv.handoff_requests(request, lender, &step)
    }
}

/// The terminal of a request that hands off, with the lend its continuation imports.
fn handed(lend: Option<Lend>) -> Event {
    Event::Migratable {
        cause: MigrationCause::Handoff,
        denoised_steps: 0,
        stored_layout: Vec::new(),
        lend,
    }
}
