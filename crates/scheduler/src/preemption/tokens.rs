//! Preempting token requests that cannot get the KV cache units their next
//! call needs.
//!
//! A running request that could not get the units of its next call is
//! starved ([`Scheduler::note_starved`]). The step's reclaim relieves it:
//! the policy names a victim that gives its units up and is recomputed
//! (`Scheduler::reclaim_shortage`). Two states remain that the reclaim
//! does not reach, and this module ends them: a request whose sequence
//! outgrew the whole cache ends with the cache-capacity finish
//! ([`Scheduler::finish_outgrown`]), and when every running request starves
//! and the policy names none, the one admitted last is recomputed
//! ([`Scheduler::recompute_starved`]). Video and image requests are
//! preempted at denoising-step boundaries instead (`super::media`).

use crate::*;

impl Scheduler {
    /// Notes that the running request `id` could not get the `needed` units
    /// its next call needs, for the next step's reclaim
    /// (`Scheduler::reclaim_shortage`), [`Self::recompute_starved`] and,
    /// while the tail is given back, [`Self::note_stuck_holder`].
    pub(crate) fn note_starved(&mut self, id: RequestId, needed: usize) {
        match self.starved.iter_mut().find(|(starved, _)| *starved == id) {
            Some((_, units)) => *units = (*units).max(needed),
            None => self.starved.push((id, needed)),
        }
        self.note_stuck_holder(id);
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
                let state = self.running.get(&id)?;
                if state.terminal_intent.is_terminal() {
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
            if let Some(state) = self.running.get_mut(&id) {
                state.terminal_intent.finish(FinishReason::CacheCapacity);
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
    /// The step's reclaim is where a starved request is relieved: the policy
    /// names the victim (`Scheduler::reclaim_shortage`). This breaks the one
    /// state that reclaim would not reach, a policy that names no request.
    pub(crate) fn recompute_starved(&mut self, submitted: bool) {
        self.finish_outgrown();
        if self.starved.is_empty()
            || submitted
            || !matches!(self.lifecycle, crate::Lifecycle::Serving)
            || !self.inflight.pending_batches.is_empty()
            || !self.inflight.pending_submissions.is_empty()
            || self
                .running
                .values()
                .any(|state| state.terminal_intent.is_terminal())
        {
            return;
        }
        let starved = |id: &RequestId| self.starved.iter().any(|(starved, _)| starved == id);
        let live: Vec<(&RequestId, &RequestState)> = self
            .running
            .iter()
            .filter(|(_, state)| !state.terminal_intent.is_terminal())
            .collect();
        if live.len() < 2 || live.iter().any(|(id, _)| !starved(id)) {
            return;
        }
        let Some(victim) = live
            .iter()
            .max_by(|(_, a), (_, b)| {
                (a.req.priority, a.queued_us)
                    .partial_cmp(&(b.req.priority, b.queued_us))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| **id)
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
