//! Request residency and capability-based worker dispatch.
//!
//! The scheduler asks `Dispatch` which loaded worker replica executes a call.
//! Candidates are the workers whose reported `supported_calls` and stage
//! bindings cover the call; once a request is placed on a replica for a
//! stage, its later calls for that stage stay there. A request's
//! decode calls stay on the replica that took its first decode-stage call,
//! which is its prefill replica unless that replica only prefills
//! (`WorkerGroupRole::Prefill`): then its cache is transferred to a decode
//! replica (`crate::transfer`).

use super::*;
use ipc::ForwardMode;

#[derive(Default)]
pub(super) struct Dispatch {
    /// Replica that holds each request's state for one bound stage name.
    ///
    /// Admission records the prefill worker of a token request and every route
    /// of a media request; `select_worker` records an entry each time it places
    /// a call, keyed by the name the worker binds (which may be
    /// `DEFAULT_BLOCK` rather than the requested name).
    /// `Scheduler::schedule` prunes entries of requests that are no
    /// longer running or retiring.
    pub(super) affinity: HashMap<(RequestKey, String), crate::WorkerGroupId>,
    /// Replica that decodes each request for one bound stage name: the
    /// replica of its decode-stage calls (`decodes`), recorded as `affinity`
    /// is.
    pub(super) decode_affinity: HashMap<(RequestKey, String), crate::WorkerGroupId>,
    /// The groups that prefill only (`WorkerGroupRole::Prefill`): their
    /// requests decode on the scheduler's decode groups.
    pub(super) prefill_groups: HashSet<crate::WorkerGroupId>,
    /// The groups that decode only (`WorkerGroupRole::Decode`): a request
    /// reaches them by its transfer, never with its prompt.
    pub(super) decode_groups: HashSet<crate::WorkerGroupId>,
}

/// Whether a call of `kind` belongs to a request's decode stage, which a
/// decode replica may hold apart from its prefill replica.
fn decodes(kind: CallKind) -> bool {
    matches!(
        kind,
        CallKind::Forward(ForwardMode::Decode | ForwardMode::Verify)
    )
}

/// Who chooses the replica a call runs on.
pub(super) enum Choice {
    /// The mechanism's own rule: the replica holding the request's state, the
    /// one already serving another of its stages, or the one with the
    /// fewest running requests.
    Existing,
    /// The policy's response for a request that has not run there yet: the worker
    /// it named among the candidates `views::replicas` showed it.
    Target(crate::WorkerGroupId),
}

impl Dispatch {
    /// Enumerates loaded stages that can execute the requested call.
    ///
    /// Which stage serves a media call is the model's state, not the
    /// engine's: a worker resolves it from the capabilities its stages
    /// implement and reports it as `media_stages`. That report is the
    /// only authority here. Calls without an explicit media route use the
    /// protocol's default stage for an undivided model.
    pub(super) fn worker_candidates<'a>(
        &'a self,
        epoch: &'a super::submission::Submissions,
        info: &'a MainProcessInfo,
        kind: CallKind,
    ) -> impl Iterator<Item = (&'a crate::WorkerGroupId, &'a str, &'a MainProcessInfo)> {
        let stage = match kind {
            CallKind::Media(media_call) => info
                .media_stages
                .get(&media_call)
                .map_or(DEFAULT_BLOCK, String::as_str),
            _ => DEFAULT_BLOCK,
        };
        self.stage_candidates(epoch, kind, stage)
    }

    /// Enumerates workers that support `kind` and can serve `stage`, with
    /// the stage name each one binds.
    ///
    /// A worker binding `stage` by name serves it under that name. A
    /// worker with no stage bindings, or with a `DEFAULT_BLOCK`
    /// binding, serves any requested stage under `DEFAULT_BLOCK`.
    /// Other workers are excluded.
    pub(super) fn stage_candidates<'a>(
        &'a self,
        epoch: &'a super::submission::Submissions,
        kind: CallKind,
        stage: &'a str,
    ) -> impl Iterator<Item = (&'a crate::WorkerGroupId, &'a str, &'a MainProcessInfo)> {
        epoch.info().workers.iter().filter_map(move |(id, info)| {
            if !info.supported_calls.contains(&kind) {
                return None;
            }
            let bound_stage = if info.stages.iter().any(|binding| binding.name == stage) {
                stage
            } else if info.stages.is_empty()
                || info
                    .stages
                    .iter()
                    .any(|binding| binding.name == DEFAULT_BLOCK)
            {
                DEFAULT_BLOCK
            } else {
                return None;
            };
            Some((id, bound_stage, info))
        })
    }

    /// Whether a request already runs somewhere: a replica holds its mutable
    /// state, so its next call goes there and the policy is not asked.
    pub(super) fn placed(&self, request: RequestKey) -> bool {
        self.affinity.keys().any(|(key, _)| *key == request)
            || self.decode_affinity.keys().any(|(key, _)| *key == request)
    }

    /// Selects one replica while preserving request residency.
    ///
    /// Returns `None` when no worker can serve the stage, when the replica
    /// that already holds this request's stage state is no longer a
    /// candidate or has no submission capacity, when a replica already holding
    /// another of the request's stages has no capacity, or when no other
    /// candidate has capacity. Records nothing; `select_worker` records the
    /// choice.
    pub(super) fn stage_target<'a>(
        &'a self,
        epoch: &'a super::submission::Submissions,
        request: RequestKey,
        kind: CallKind,
        stage: &'a str,
        choice: &Choice,
    ) -> Option<(&'a crate::WorkerGroupId, &'a str)> {
        // A decode-stage call runs on a replica that decodes, and every other
        // call (a prompt's prefill) on one that is not a decode group: a
        // request reaches a decode group by its transfer.
        let candidates = self
            .stage_candidates(epoch, kind, stage)
            .filter(|(id, _, _)| {
                if decodes(kind) {
                    !self.prefill_groups.contains(*id)
                } else {
                    !self.decode_groups.contains(*id)
                }
            })
            .collect::<Vec<_>>();
        let bound = candidates.first()?.1;

        // Once a request has used this stage, its mutable request state and
        // graph slot live on that exact replica. Queue pressure may delay the
        // next call but must never migrate it.
        let pinned = if decodes(kind) {
            &self.decode_affinity
        } else {
            &self.affinity
        };
        if let Some(owner) = pinned.get(&(request, bound.to_owned())) {
            let (id, stage, _) = candidates.into_iter().find(|(id, _, _)| *id == owner)?;
            return epoch.has_capacity(id).then_some((id, stage));
        }

        // A request that has not run here yet runs where the policy says
        // (design §4.4). A named worker that is not a candidate, or whose queue
        // has no room, leaves the call to a later step: the mechanism never
        // substitutes a replica the policy did not choose.
        if let Choice::Target(named) = choice {
            let (id, stage, _) = candidates.into_iter().find(|(id, _, _)| *id == named)?;
            return epoch.has_capacity(id).then_some((id, stage));
        }

        // Prefer a worker already selected for another stage of this
        // request. This keeps a replicated denoiser and its media decoders in
        // one process and avoids publishing their large latent products.
        let placed = self
            .affinity
            .iter()
            .filter_map(|((key, _), worker)| (*key == request).then_some(worker))
            .collect::<HashSet<_>>();
        if let Some((id, stage, _)) = candidates
            .iter()
            .copied()
            .find(|(id, _, _)| placed.contains(id))
        {
            return epoch.has_capacity(id).then_some((id, stage));
        }

        // Among replicas with queue capacity, a new request goes to the one
        // holding the fewest distinct running requests, so a burst fans out
        // instead of filling the first configured worker. Queue capacity is a
        // hard condition; ties go to the earliest worker in
        // `WorkerInfo::workers`.
        candidates
            .into_iter()
            .filter(|(id, _, _)| epoch.has_capacity(id))
            .min_by_key(|(id, _, _)| {
                self.affinity
                    .iter()
                    .filter_map(|((key, _), worker)| (worker == *id).then_some(*key))
                    .collect::<HashSet<_>>()
                    .len()
            })
            .map(|(id, stage, _)| (id, stage))
    }

    /// Selects a replica for `kind` using the aggregate media routing in
    /// `info`, as `worker_candidates` does, while preserving residency.
    pub(super) fn worker_target<'a>(
        &'a self,
        epoch: &'a super::submission::Submissions,
        info: &'a MainProcessInfo,
        request: RequestKey,
        kind: CallKind,
        choice: &Choice,
    ) -> Option<(&'a crate::WorkerGroupId, &'a str)> {
        let stage = match kind {
            CallKind::Media(media_call) => info
                .media_stages
                .get(&media_call)
                .map_or("model", String::as_str),
            _ => "model",
        };
        self.stage_target(epoch, request, kind, stage, choice)
    }

    /// Places `request`'s calls of `kind` on the replica that will execute
    /// them, as `select_worker` places such a call of the protocol's default
    /// stage: the replica the request is placed on for the stage
    /// that serves them, or one chosen and recorded now, so that the call
    /// prepared later lands there. `None` when no eligible replica has
    /// submission capacity.
    pub(super) fn place(
        &mut self,
        epoch: &super::submission::Submissions,
        info: &MainProcessInfo,
        request: RequestKey,
        kind: CallKind,
    ) -> Option<crate::WorkerGroupId> {
        let (id, bound_stage) =
            self.worker_target(epoch, info, request, kind, &Choice::Existing)?;
        let target = (id.clone(), bound_stage.to_owned());
        let pinned = if decodes(kind) {
            &mut self.decode_affinity
        } else {
            &mut self.affinity
        };
        pinned.insert((request, target.1), target.0.clone());
        Some(target.0)
    }

    /// Binds prepared work to its configured stage and records request residency.
    ///
    /// Returns `None`, recording nothing, when `stage_target` finds no
    /// eligible replica with submission capacity.
    pub(super) fn select_worker(
        &mut self,
        epoch: &super::submission::Submissions,
        info: &MainProcessInfo,
        call: &Call,
        choice: &Choice,
    ) -> Option<(crate::WorkerGroupId, String)> {
        let (id, bound_stage) = if call.stage == DEFAULT_BLOCK {
            self.worker_target(epoch, info, call.request_key, call.kind, choice)
        } else {
            self.stage_target(epoch, call.request_key, call.kind, &call.stage, choice)
        }?;
        let target = (id.clone(), bound_stage.to_owned());
        let pinned = if decodes(call.kind) {
            &mut self.decode_affinity
        } else {
            &mut self.affinity
        };
        pinned.insert((call.request_key, target.1.clone()), target.0.clone());
        Some(target)
    }

    /// Includes lifecycle commands in destination ordering, even when a batch has no compute.
    ///
    /// Returns each call's placed worker plus every worker holding residency
    /// for a command's request. `Scheduler::release` uses this
    /// set to hold back later batches that share a blocked destination.
    pub(super) fn batch_workers(&self, batch: &ExecutionBatch) -> HashSet<crate::WorkerGroupId> {
        let mut targets = batch
            .requests
            .iter()
            .map(|(_, dispatch)| dispatch.worker.clone())
            .collect::<HashSet<_>>();
        for command in &batch.commands {
            targets.extend(
                self.affinity
                    .iter()
                    .chain(&self.decode_affinity)
                    .filter_map(|((request, _), worker)| {
                        (*request == command.request_key()).then_some(worker.clone())
                    }),
            );
        }
        targets
    }
}
