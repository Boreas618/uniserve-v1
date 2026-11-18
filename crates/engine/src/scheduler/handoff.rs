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
//!   tiers hold of them by its view of the cache ([`Output::Handoff`]). A push or a read imports
//!   every page of the lend into blocks of its own (`Import`), which its full pages are
//!   committed under their keys once filled, and it runs from the lend's tokens; a pull reads the
//!   full pages from the tier as any admission reads a prefix, ends the lend with an import of
//!   no page and computes the tokens past its hit, the prompt's partial last page among them. An
//!   import that fails leaves the request to an admission of its own, which computes its prompt.
//!   A model that keeps a recurrent state ends the lend at once and computes its prompt: its live
//!   state lies in the prefill replica's slot, which no import reaches.

use proto::request::{
    Event, GenerationRequest, Lend, MigrationCause, RejectionKind, Request as Input, RequestId,
};
use scheduling::kv::{Found, Handoff, Path, Tier, Transport};
use scheduling::routing::Role;
use storage::runtime::{self, Completion, Lender, OwnerId};

use super::cache::{DEVICE, HOST, REMOTE};
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
        let Some(owner) = self.kv_owner(request) else {
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
    /// what a decode replica may pull from there.
    fn stored_pages(&self, id: RequestId) -> u32 {
        let Some(request) = self.requests.get(&id) else {
            return 0;
        };
        let Some(class) = self.attention_class(request) else {
            return 0;
        };
        let keys = request.keys.get(class).map_or(&[][..], Vec::as_slice);
        let leading = self
            .expected_hit(
                keys,
                self.kv_group(request)
                    .unwrap_or(storage::runtime::GroupId(0)),
            )
            .iter()
            .take_while(|tiers| **tiers & REMOTE != 0)
            .count();
        u32::try_from(leading).unwrap_or(u32::MAX)
    }

    /// The first cache class of request `id`'s model that every request reads: the class whose
    /// pages a handoff counts.
    fn attention_class(&self, request: &super::Request) -> Option<usize> {
        self.shapes(request)
            .iter()
            .position(|(kind, _)| kind.required_rows())
    }

    /// Asks the KV program how request `id`'s lent pages arrive, from what this replica's tiers
    /// hold of them as `cache` sees it; the engine answers with [`Self::routed`]. A model that
    /// keeps a recurrent state imports nothing: its path is a pull of what the tiers hold.
    pub(super) fn ask_path(&mut self, id: RequestId) {
        let Some(request) = self.requests.get(&id) else {
            return;
        };
        let Some(lend) = request.generation().and_then(|input| input.lend.clone()) else {
            return;
        };
        if self.state_shape(request).is_some() {
            return self.routed(id, Path::Pull(Tier::Host));
        }
        let (Some(class), Some(group)) = (self.attention_class(request), self.kv_group(request))
        else {
            return self.routed(id, Path::Pull(Tier::Host));
        };
        let block = self
            .shapes(request)
            .get(class)
            .map_or(1, |&(_, block)| block.max(1));
        let pages = lend.tokens / block;
        let keys = request.keys.get(class).map_or(&[][..], Vec::as_slice);
        let keys = keys.get(..pages as usize).unwrap_or(keys);
        let tiers = self.expected_hit(keys, group);
        let leading = |bit: u8| {
            let held = tiers.iter().take_while(|tiers| **tiers & bit != 0).count();
            u32::try_from(held).unwrap_or(u32::MAX)
        };
        let handoff = Handoff {
            pages,
            tail: lend.tokens % block,
            found: vec![
                Found {
                    tier: Tier::Device,
                    pages: leading(DEVICE),
                },
                Found {
                    tier: Tier::Host,
                    pages: leading(HOST),
                },
                Found {
                    tier: Tier::Store,
                    pages: leading(REMOTE).max(lend.stored.min(pages)),
                },
            ],
            local: self.local(&lend),
            transports: vec![Transport::Nixl],
        };
        self.emit(Output::Handoff(id, handoff));
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
    /// known: for a push or a read, the import of every lent page of each class every request
    /// reads; for a pull, the admission's own round and an import of no page, which ends the
    /// lend. A model that keeps a recurrent state imports nothing either.
    pub(super) fn handoff_requests(&mut self, id: RequestId) -> Vec<runtime::Request> {
        let Some(request) = self.requests.get(&id) else {
            return Vec::new();
        };
        let (Some(lend), Some(path), Some(owner)) = (
            request.generation().and_then(|input| input.lend.clone()),
            request.path,
            self.kv_owner(request),
        ) else {
            return self.admission_requests(id);
        };
        let lender = Lender {
            endpoint: (!self.local(&lend))
                .then(|| lend.endpoint.clone())
                .flatten(),
            session: lend.session,
            owner: OwnerId(lend.owner),
        };
        // The Storage Server's span of the handoff names the request.
        let trace = request.input.trace();
        let push = match path {
            Path::Push(_) => true,
            Path::Read(_) => false,
            Path::Pull(_) => {
                let mut asks = self.admission_requests(id);
                asks.push(runtime::Request::Import {
                    owner,
                    tables: Vec::new(),
                    lender,
                    push: false,
                    trace,
                });
                return asks;
            }
        };

        // Every page of the lend, of every class every request reads.
        let shapes = self.shapes(request).to_vec();
        let tables: Vec<(u8, u32, u32)> = shapes
            .iter()
            .enumerate()
            .filter(|(_, (kind, _))| kind.required_rows())
            .filter_map(|(class, &(_, block))| {
                let pages = lend.tokens.div_ceil(block.max(1));
                Some((u8::try_from(class).ok()?, 0, pages))
            })
            .filter(|(.., pages)| *pages > 0)
            .collect();
        if let Some(request) = self.requests.get_mut(&id) {
            request.blocks = vec![super::Blocks::default(); shapes.len()];
            request.num_cached_tokens = 0;
            request.num_host_tokens = 0;
            request.num_peer_tokens = 0;
        }
        vec![runtime::Request::Import {
            owner,
            tables,
            lender,
            push,
            trace,
        }]
    }

    /// Applies the answer to request `id`'s `Import` of `tables` and returns the requests it
    /// calls for. Blocks imported for a lend of `tokens` tokens hold its prompt up to them: each
    /// class's full pages are committed under their keys, read-only from now on, its partial
    /// last page is the request's to write on, and the blocks of its first chunk and the low
    /// watermark follow, so that it runs from the lend's tokens. A failed import leaves the
    /// request to the admission's own round, which computes what no tier holds. An import of no
    /// page holds nothing.
    pub(super) fn imported(
        &mut self,
        id: RequestId,
        (owner, tables): (OwnerId, &[(u8, u32, u32)]),
        completion: &Completion,
    ) -> Vec<runtime::Request> {
        if tables.is_empty() {
            return Vec::new();
        }
        let Some(request) = self.requests.get(&id) else {
            return Vec::new();
        };
        let Some(tokens) = request
            .generation()
            .and_then(|input| input.lend.as_ref())
            .map(|lend| lend.tokens)
        else {
            return Vec::new();
        };
        let Completion::Imported { .. } = completion else {
            return self.admission_requests(id);
        };

        let shapes = self.shapes(request).to_vec();
        let mut follow = Vec::new();
        let mut blocks = request.blocks.clone();
        for &(class, _, pages) in tables {
            let Some(&(_, block)) = shapes.get(usize::from(class)) else {
                continue;
            };
            let full = tokens / block.max(1);
            let keys = request.keys.get(usize::from(class));
            for at in 0..full {
                let Some(key) = keys.and_then(|keys| keys.get(at as usize)) else {
                    continue;
                };
                let tokens = u16::try_from(block).unwrap_or(u16::MAX);
                follow.push(super::memory::commit(owner, class, at, *key, tokens));
            }
            if let Some(table) = blocks.get_mut(usize::from(class)) {
                *table = super::Blocks {
                    first: 0,
                    shared: full,
                    end: pages,
                    asked: pages,
                };
            }
        }
        let Some(request) = self.requests.get_mut(&id) else {
            return follow;
        };
        request.blocks = blocks;
        request.num_cached_tokens = tokens;

        // The first chunk's blocks, from the lend's tokens on.
        let request = request.clone();
        let mut blocks = request.blocks.clone();
        self.first_call_asks(&request, owner, tokens, &mut blocks, &mut follow);
        if let Some(row) = self.requests.get_mut(&id) {
            row.blocks = blocks;
        }
        follow
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
