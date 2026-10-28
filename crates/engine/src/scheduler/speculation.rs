//! Speculative decoding: each running request's draft cache, the drafts its
//! verifications carry, and the refill of a draft cache that is missing.
//!
//! In a deployment that speculates (`SchedulerConfig::draft_tokens` above
//! zero) the workers run the model's drafter after every forward. The
//! drafter's entry at position `i` reads the target's hidden state at `i`
//! and the token at `i + 1`; a forward writes the entries of the positions
//! it evaluates whose next token it knows, into the request's draft cache
//! pages (the KV cache's optional draft group), and the call that asks for
//! them returns the drafts the drafter proposes after its last token.
//!
//! A request's draft cache is complete when its entries are valid for every
//! position its cache holds ([`DraftCache::valid`]). Only then does its next
//! verification carry drafts, as many as the scheduler chooses each step
//! (zero to the deployment's limit and the request's remaining tokens);
//! otherwise it verifies none, a verification row of one token, and decodes
//! correctly. A draft cache goes missing when the request reuses a prefix
//! whose draft pages the pool lacks, when a world change moves it from a
//! world that did not speculate, or when its draft pages were released to
//! make room for another request's required pages. Its forwards keep
//! writing the entries of their own positions ([`DraftCache::tail`]); a
//! refill evaluates the target model again over the positions before them,
//! in chunks of the prefill budget a step leaves idle, when the refill
//! policy finds it worth its cost ([`crate::scheduler::RefillPolicy`]).
//! Once the refill reaches the entries the request wrote itself, its draft
//! cache is complete and it speculates.

use super::*;
use crate::kv_cache::Checkpoint;
use ipc::{CallOutput, ForwardMode};

/// A running request's draft cache, in a deployment that speculates.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct DraftCache {
    /// Leading positions whose draft cache entries are valid: every entry
    /// before it was written from the target's hidden state at its position
    /// and the request's token after it.
    pub(crate) valid: u32,
    /// Positions `[start, end)` after a gap whose entries are valid: those
    /// the request's forwards wrote since its draft cache went missing. A
    /// refill that reaches `start` makes them part of the valid prefix.
    pub(crate) tail: Option<(u32, u32)>,
    /// The drafts the request's last forward proposed after its last
    /// committed token, for its next verification.
    pub(crate) proposals: Vec<u32>,
}

impl DraftCache {
    /// Records that the entries of positions `[start, end)` are valid.
    pub(crate) fn record(&mut self, start: u32, end: u32) {
        if start >= end {
            return;
        }
        if start <= self.valid {
            self.valid = self.valid.max(end);
        } else if let Some((first, last)) = self.tail.as_mut()
            && *first <= start
            && start <= *last
        {
            *last = (*last).max(end);
        } else {
            // A gap separates these entries from every valid one.
            self.tail = Some((start, end));
        }
        // A valid prefix that reaches the tail takes it.
        if let Some((first, last)) = self.tail
            && first <= self.valid
        {
            self.valid = self.valid.max(last);
            self.tail = None;
        }
    }

    /// Forgets every entry: the request's draft cache pages released their
    /// units.
    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    /// Whether the entries of every position before `visible` are valid.
    pub(crate) fn complete(&self, visible: u32) -> bool {
        self.valid >= visible
    }

    /// The position a refill evaluates up to: the first the request wrote
    /// itself after the gap, or `visible`.
    fn refill_end(&self, visible: u32) -> u32 {
        self.tail.map_or(visible, |(start, _)| start.min(visible))
    }
}

/// The tokens a request's cache holds, in position order: its prompt (a
/// migrated request's ending with the tokens it generated before) and the
/// tokens it generated since its admission.
fn cached_token(state: &RequestState, position: u32) -> Option<u32> {
    let prompt = &state.req.prompt_token_ids;
    let position = position as usize;
    if let Some(&token) = prompt.get(position) {
        return Some(token);
    }
    let own = state
        .generated_token_ids
        .get(state.req.generated_token_ids.len()..)
        .unwrap_or_default();
    own.get(position - prompt.len()).copied()
}

impl Scheduler {
    /// Whether the deployment speculates and its pool keeps the draft cache:
    /// the world's workers run the drafter.
    pub(in crate::scheduler) fn speculates(&self) -> bool {
        self.config.draft_tokens > 0
            && self
                .storage
                .cache
                .as_ref()
                .is_some_and(|cache| cache.coordinator.draft_group().is_some())
    }

    /// The drafts `id`'s next verification carries, from token position
    /// `visible`: its last forward's proposals when its draft cache is
    /// complete, at most the deployment's limit and one fewer than the
    /// tokens it may still generate, none otherwise.
    pub(in crate::scheduler) fn verify_drafts(&self, id: RequestId, visible: u32) -> Vec<u32> {
        let Some(state) = self.running.get(&id) else {
            return Vec::new();
        };
        if !state.draft.complete(visible) {
            return Vec::new();
        }
        let remaining = state
            .req
            .max_und_tokens
            .saturating_sub(state.num_generated_tokens)
            .saturating_sub(1);
        let count = state
            .draft
            .proposals
            .len()
            .min(self.config.draft_tokens)
            .min(remaining);
        state.draft.proposals[..count].to_vec()
    }

    /// The drafts a forward of `id` whose first position is `start` asks
    /// for: the deployment's limit when the request's draft cache is valid
    /// through `start`, so that the forward completes it, none otherwise.
    pub(in crate::scheduler) fn proposals_for(&self, id: RequestId, start: u32) -> u32 {
        self.running
            .get(&id)
            .filter(|state| state.draft.complete(start))
            .map_or(0, |_| self.config.draft_tokens as u32)
    }

    /// Records the draft cache entries `call`'s accepted result wrote, and
    /// keeps the drafts it proposed for the request's next verification.
    ///
    /// A text prompt chunk writes the entries of its positions whose next
    /// token it knows (every one with a sampled token or a lookahead, else
    /// all but its last); a verification those of its current token and
    /// accepted drafts, which its committed tokens follow. Entries count
    /// only where the request's draft pages hold units.
    pub(in crate::scheduler) fn record_draft(
        &mut self,
        id: RequestId,
        call: &Call,
        record: &CallOutput,
    ) {
        if !self.speculates() || record.status != CallStatus::Ok {
            return;
        }
        let start = call.coordinates.cache_visible_len;
        let end = match call.kind {
            CallKind::Forward(ForwardMode::Prefill)
                if is_prompt_extend(call) && call.vision_inputs.is_empty() =>
            {
                let count = call.input_token_ids.len() as u32;
                let known = call.token_output.is_some() || call.draft.lookahead.is_some();
                start + count - u32::from(!known && count > 0)
            }
            CallKind::Forward(ForwardMode::Verify | ForwardMode::Decode) => {
                start + record.committed_tokens.len().max(1) as u32
            }
            _ => start,
        };
        if call.kind == CallKind::Forward(ForwardMode::Verify) {
            self.count_verification(id, call, record);
        }
        let held = self.holds_draft(id, start, end);
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        if held {
            state.draft.record(start, end);
        }
        state.draft.proposals = record.draft_token_ids.clone();
    }

    /// Counts an accepted verification's drafts: those it accepted, the
    /// accepted prefix of its committed tokens (all of them when a finish
    /// token among the drafts ended the commit), and those it rejected.
    fn count_verification(&self, id: RequestId, call: &Call, record: &CallOutput) {
        let drafts = generation::verify_drafts(call);
        let committed = &record.committed_tokens;
        let matching = committed
            .iter()
            .zip(drafts)
            .take_while(|(token, draft)| token == draft)
            .count();
        let terminal = matching == committed.len()
            && committed.last().is_some_and(|token| {
                self.running
                    .get(&id)
                    .is_some_and(|state| state.finish_token_ids.contains(token))
            });
        let accepted = if terminal {
            matching
        } else {
            matching.min(committed.len().saturating_sub(1))
        };
        let stats = &self.stats.worker;
        stats
            .spec_verify_accepted_tokens
            .fetch_add(accepted as u64, Ordering::Relaxed);
        stats
            .spec_verify_rejected_tokens
            .fetch_add((drafts.len() - accepted) as u64, Ordering::Relaxed);
        stats
            .spec_verify_committed_tokens
            .fetch_add(committed.len() as u64, Ordering::Relaxed);
    }

    /// Whether `id`'s draft cache pages covering positions `[start, end)`
    /// hold units.
    fn holds_draft(&self, id: RequestId, start: u32, end: u32) -> bool {
        let (Some(state), Some(cache)) = (self.running.get(&id), self.storage.cache.as_ref())
        else {
            return false;
        };
        state.allocations().is_some_and(|allocations| {
            cache
                .coordinator
                .holds_draft(&allocations.prefix_cache, start as usize, end as usize)
        })
    }

    /// The token boundary whose state checkpoint `id`'s refill continues
    /// from while its draft cache is missing, which its state tables keep.
    pub(in crate::scheduler) fn refill_cursor(&self, id: RequestId) -> Option<usize> {
        let state = self.running.get(&id)?;
        (self.speculates() && !state.draft.complete(state.cache_visible_len))
            .then_some(state.draft.valid as usize)
    }

    /// Whether `id` is due a refill of its missing draft cache: it decodes,
    /// has no call in flight, and the refill policy finds evaluating the
    /// positions before the entries it wrote itself worth the decode steps
    /// speculation saves over the tokens it may still generate.
    pub(in crate::scheduler) fn refill_due(&self, id: RequestId) -> bool {
        if !self.speculates() || self.inflight.has_pending_calls(id) {
            return false;
        }
        let Some(state) = self.running.get(&id) else {
            return false;
        };
        let visible = state.cache_visible_len;
        if state.phase != Phase::DecodeUnd
            || !state.worker_registered
            || state.terminal_intent.is_terminal()
            || state.round_closing
            || state.draft.complete(visible)
            || state.num_computed_prompt_tokens < state.req.prompt_token_ids.len() as u32
        {
            return false;
        }
        let tokens = state
            .draft
            .refill_end(visible)
            .saturating_sub(state.draft.valid);
        let remaining = state
            .req
            .max_und_tokens
            .saturating_sub(state.num_generated_tokens);
        tokens > 0 && self.config.refill.worth(remaining, tokens as usize)
    }

    /// Plans the next chunk of `id`'s refill within `budget` tokens.
    ///
    /// The chunk evaluates the request's tokens from the end of its valid
    /// draft cache, where its state tables keep a checkpoint (or from the
    /// beginning, where they keep none), towards the first entry it wrote
    /// itself. A model with layers kept per request slot ends a chunk short
    /// of that at a checkpoint boundary, from which the next continues. The
    /// chunk's tables grow first: the state checkpoint it writes and its
    /// draft pages, without which no refill is planned. Returns `None` when
    /// no chunk fits or the tables cannot grow.
    pub(in crate::scheduler) fn plan_refill(
        &mut self,
        id: RequestId,
        budget: usize,
    ) -> Option<Call> {
        let cache = self.storage.cache.as_ref()?;
        let state = self.running.get(&id)?;
        let visible = state.cache_visible_len;
        let target = state.draft.refill_end(visible);
        let allocation = &state.allocations()?.prefix_cache;
        let start = if cache
            .coordinator
            .resumes_at(allocation, state.draft.valid as usize)
        {
            state.draft.valid
        } else {
            0
        };
        let mut end = (start as usize + budget).min(target as usize) as u32;
        if end < target
            && let Some(block) = cache.coordinator.state_block_size()
        {
            end = end / block as u32 * block as u32;
        }
        if end <= start {
            return None;
        }
        let tokens = (start..end)
            .map(|position| cached_token(state, position))
            .collect::<Option<Vec<_>>>()?;
        // The token after the chunk: a cached one, or the committed token
        // the request's next forward evaluates.
        let lookahead = if end < visible {
            cached_token(state, end)?
        } else {
            state.next_token
        };

        let state = self.running.get_mut(&id)?;
        let allocations = state.allocations_mut()?;
        let cache = self.storage.cache.as_ref()?;
        cache
            .coordinator
            .hold_states_from(&mut allocations.prefix_cache, start as usize);
        cache
            .grow(
                &mut allocations.prefix_cache,
                start,
                end,
                Checkpoint::Boundary,
            )
            .ok()?;
        if !cache
            .coordinator
            .holds_draft(&allocations.prefix_cache, start as usize, end as usize)
        {
            return None;
        }
        self.plan_call(id, |_scheduler, request| {
            generation::plan_refill(request, start, tokens, lookahead)
        })
    }

    /// Applies an accepted refill chunk: the entries of its positions are
    /// valid, the request's state tables move their kept checkpoint to the
    /// chunk's end, and its draft cache pages that became complete publish.
    pub(in crate::scheduler) fn accept_refill(
        &mut self,
        id: RequestId,
        call: &Call,
        record: &CallOutput,
    ) {
        if record.status != CallStatus::Ok {
            return;
        }
        let Some(start) = call.draft.refill else {
            return;
        };
        let end = start + call.input_token_ids.len() as u32;
        if self.holds_draft(id, start, end)
            && let Some(state) = self.running.get_mut(&id)
        {
            state.draft.record(start, end);
            if state.draft.complete(state.cache_visible_len) {
                tracing::info!(
                    request_id = id.0,
                    tokens = state.draft.valid,
                    "refilled a request's draft cache: it speculates from now on"
                );
            }
        }
        self.advance_request_cache(id);
    }

    /// Releases the draft cache of running requests with no call in flight,
    /// one at a time, the request with the fewest tokens left to generate
    /// first, until the pool has `units` free units; returns whether it has
    /// them. A request whose draft cache is released verifies no drafts
    /// until a refill restores it.
    pub(in crate::scheduler) fn release_drafts(&mut self, units: usize) -> bool {
        let Some(cache) = self.storage.cache.as_ref() else {
            return false;
        };
        if cache.block_pool.free_units() >= units {
            return true;
        }
        if cache.coordinator.draft_group().is_none() {
            return false;
        }
        let mut candidates: Vec<(usize, RequestId)> = self
            .running
            .iter()
            .filter(|(id, state)| {
                !self.inflight.has_pending_calls(**id)
                    && (state.draft.valid > 0 || state.draft.tail.is_some())
            })
            .map(|(id, state)| {
                (
                    state
                        .req
                        .max_und_tokens
                        .saturating_sub(state.num_generated_tokens),
                    *id,
                )
            })
            .collect();
        candidates.sort();
        for (_, id) in candidates {
            let Some(cache) = self.storage.cache.as_ref() else {
                return false;
            };
            if cache.block_pool.free_units() >= units {
                break;
            }
            let Some(state) = self.running.get_mut(&id) else {
                continue;
            };
            if let Some(allocations) = state.allocations_mut() {
                cache
                    .coordinator
                    .release_draft(&mut allocations.prefix_cache);
            }
            state.draft.clear();
            tracing::debug!(
                request_id = id.0,
                "released a request's draft cache for another request's pages"
            );
        }
        self.storage
            .cache
            .as_ref()
            .is_some_and(|cache| cache.block_pool.free_units() >= units)
    }
}

#[cfg(test)]
mod tests {
    use super::DraftCache;

    #[test]
    fn entries_after_a_gap_join_the_valid_prefix_once_a_refill_reaches_them() {
        let mut draft = DraftCache::default();
        // A migrated request reused 512 tokens without their draft cache
        // and computed the rest of its prompt.
        draft.record(512, 700);
        assert_eq!((draft.valid, draft.tail), (0, Some((512, 700))));
        assert!(!draft.complete(700));
        // Its decoding extends the entries it writes.
        draft.record(700, 703);
        assert_eq!(draft.tail, Some((512, 703)));
        // A refill chunk short of the tail leaves the gap.
        draft.record(0, 256);
        assert_eq!((draft.valid, draft.tail), (256, Some((512, 703))));
        // The chunk reaching it completes the draft cache.
        draft.record(256, 512);
        assert_eq!((draft.valid, draft.tail), (703, None));
        assert!(draft.complete(703));
    }

    #[test]
    fn entries_after_a_second_gap_replace_the_tail() {
        let mut draft = DraftCache {
            valid: 64,
            tail: Some((128, 192)),
            proposals: Vec::new(),
        };
        // Pages without units left a gap after the tail.
        draft.record(256, 300);
        assert_eq!((draft.valid, draft.tail), (64, Some((256, 300))));
    }
}
