//! Preempting token requests that cannot get the KV cache units their next
//! call needs.
//!
//! A running request that could not get the units of its next call is
//! starved ([`Scheduler::note_starved`]). The step's preemption relieves it:
//! the policy names the victims of the shortage, and the mechanism frees
//! exactly those ([`Scheduler::preempt_shortage`]): a request gives its units
//! up and is recomputed, an idle page goes to the host pool, an encoder
//! entry leaves the encoder cache. Two states remain that the preemption
//! does not reach, and this module ends them: a request whose sequence
//! outgrew the whole cache ends with the cache-capacity finish
//! ([`Scheduler::finish_outgrown`]), and when every running request starves
//! and the policy names none, the one admitted last is recomputed
//! ([`Scheduler::recompute_starved`]). Video and image requests are
//! preempted at denoising-step boundaries instead (`super::media`).

use crate::*;

impl Scheduler {
    /// Notes that the running request `id` could not get the `needed` units
    /// its next call needs, for the next step's preemption
    /// (`Scheduler::preempt_shortage`), [`Self::recompute_starved`] and,
    /// while the tail is given back, [`Self::note_stuck_holder`].
    pub(crate) fn note_starved(&mut self, id: RequestId, needed: usize) {
        match self.starved.iter_mut().find(|(starved, _)| *starved == id) {
            Some((_, units)) => *units = (*units).max(needed),
            None => self.starved.push((id, needed)),
        }
        self.note_stuck_holder(id);
    }

    /// Asks the policy for the victims of the shortage the mechanism reports,
    /// validates them, and frees exactly those (design §4.4).
    ///
    /// The mechanism decides that there is a shortage and owns every cache; the
    /// policy decides what to give up. A victim list that does not validate is
    /// a policy failure, like any other invalid decision.
    pub(crate) fn preempt_shortage(&mut self) {
        // A request no victim could let grow is no shortage a policy can
        // relieve: it ends at the length the cache holds.
        self.finish_outgrown();
        let Some(shortage) = views::shortage_now(self) else {
            return;
        };
        let Some(mut policy) = self.policy.take() else {
            return;
        };
        let victims = match policy.preempt(&shortage) {
            Ok(victims) => victims,
            Err(error) => {
                let _ = self.policy_failed(policy, error);
                return;
            }
        };
        let live = self.live();
        match mechanism::validate_victims(&shortage, &victims, &live) {
            Ok(()) => {}
            Err(mechanism::VictimError::Invalid(error)) => {
                let _ = self.policy_failed(policy, error);
                return;
            }
            Err(mechanism::VictimError::Vanished(message)) => {
                // Nothing of this step can free a page or end a request between
                // the observation the policy answered from and this call, so a
                // victim that is gone means the mechanism let go of it without
                // recording that it did: an invariant of its own, never the
                // policy's decision (§4.4's taxonomy).
                self.policy = Some(policy);
                self.invariant_broken(&message);
                return;
            }
        }
        self.policy = Some(policy);
        for victim in &victims {
            self.free_victim(victim);
        }
        if !victims.is_empty() {
            tracing::debug!(
                kind = ?shortage.kind,
                need = shortage.need,
                victims = victims.len(),
                "the policy freed what it named"
            );
        }
    }

    /// Frees the one thing a victim names.
    ///
    /// An encoder entry leaves the encoder cache and its worker buffer goes with
    /// it; a device page goes to the host KV cache, exactly as a withdrawal's
    /// pages do, and the units it held return when the copy completes (or at
    /// once when the host cache has nowhere to put it).
    fn free_victim(&mut self, victim: &scheduling::guest::Victim) {
        match victim {
            scheduling::guest::Victim::Encoding(id) => {
                let key = self
                    .storage
                    .encoder_cache
                    .entries()
                    .map(|(key, _)| key)
                    .find(|key| views::encoding_id(key) == *id);
                let freed = key.and_then(|key| self.storage.encoder_cache.evict(&key));
                if let Some(product) = freed {
                    self.free_buffers([product.buffer_id()]);
                }
            }
            scheduling::guest::Victim::Page(page) => {
                if let Some(unit) = views::unit_of(*page) {
                    self.offload_units(&[unit]);
                }
            }
            // Today's capacity preemption, which the design keeps: a request
            // that must give its units away is marked migratable with `Drain`,
            // as `recompute_starved` does when it withdraws the units its
            // requests hold. `Drain` asks the engine to hold the epoch the
            // stream is in, and there is no epoch change here, so the daemon
            // resubmits the request as a continuation carrying the state it
            // reached. Only a migratable request is marked: one the daemon
            // could not resubmit would be lost. The request retires at its next
            // call boundary, and what it holds returns then.
            scheduling::guest::Victim::Request(request) => {
                self.migrate_running(MigrationCause::Drain, |id| id.0 == request.0);
            }
        }
    }

    /// Ends the starved requests no preemption can let grow: the units one
    /// holds and the units its next call needs are more than the whole cache
    /// gives, so it could not go on with every other request gone. Its
    /// sequence is as long as the cache holds, and it ends there with the
    /// cache-capacity finish (`FinishReason::CacheCapacity`), once its calls
    /// in flight drain, rather than starving until the watchdog fails it.
    pub(crate) fn finish_outgrown(&mut self) {
        let capacity = self.storage.capacity_units();
        let outgrown: Vec<(RequestId, usize)> = self
            .starved
            .iter()
            .filter_map(|&(id, needed)| {
                let state = self.requests.tokens(&id)?;
                if self.requests.finished(&id) {
                    return None;
                }
                let held: usize = state
                    .block_tables()?
                    .iter()
                    .map(BlockTable::held_units)
                    .sum();
                (held.saturating_add(needed) > capacity).then_some((id, held))
            })
            .collect();
        for (id, held) in outgrown {
            self.starved.retain(|(starved, _)| *starved != id);
            tracing::warn!(
                request = id.0,
                held,
                capacity,
                "a request needs more KV cache units than the cache holds: it ends at the length \
                 the cache holds"
            );
            if let Some(status) = self.requests.status_mut(&id) {
                status.finish(Terminal::finish(FinishReason::CacheCapacity));
            }
        }
    }

    /// Recomputes a running request when none can go on and the policy gave
    /// none up: every running token request could not get the cache units
    /// its next call needs ([`Self::note_starved`]) as the worker was
    /// refilled, which `submitted` nothing, no batch is in flight and no
    /// request is leaving, so none would end and let units go. The one of
    /// the lowest priority (the highest value) admitted last ends as
    /// migratable, and the daemon submits it again with its generated tokens,
    /// to wait for units like any new request; the units it holds return to
    /// allocation for the others. A request running alone gains nothing from
    /// giving up its own units, and is not recomputed: one the cache cannot
    /// hold ends at its length ([`Self::finish_outgrown`]).
    ///
    /// The step's preemption is where a starved request is relieved: the policy
    /// names the victim (`Scheduler::preempt_shortage`). This breaks the one
    /// state that preemption would not reach, a policy that names no request.
    pub(crate) fn recompute_starved(&mut self, submitted: bool) {
        self.finish_outgrown();
        if self.starved.is_empty()
            || submitted
            || !matches!(self.lifecycle, crate::Lifecycle::Serving)
            || !self.inflight.pending_batches.is_empty()
            || !self.inflight.pending_submissions.is_empty()
            || self
                .requests
                .running_tokens()
                .any(|(_, status, _)| status.is_finished())
        {
            return;
        }
        let starved = |id: &RequestId| self.starved.iter().any(|(starved, _)| starved == id);
        // Each live token request with its priority and the time it arrived.
        let live: Vec<(RequestId, i32, u64)> = self
            .requests
            .running_tokens()
            .filter(|(_, status, _)| !status.is_finished())
            .map(|(id, _, state)| {
                let queued_us = self
                    .requests
                    .get(&id)
                    .map_or(0, |request| request.queued_us);
                (id, state.req.priority, queued_us)
            })
            .collect();
        if live.len() < 2 || live.iter().any(|(id, _, _)| !starved(id)) {
            return;
        }
        let Some(victim) = live
            .iter()
            .max_by_key(|(_, priority, queued_us)| (*priority, *queued_us))
            .map(|(id, _, _)| *id)
        else {
            return;
        };
        tracing::info!(
            request = victim.0,
            running = live.len(),
            "no running request can get the cache units it needs: the one admitted last is \
             recomputed"
        );
        self.migrate_running(MigrationCause::Drain, |id| id == victim);
    }
}
