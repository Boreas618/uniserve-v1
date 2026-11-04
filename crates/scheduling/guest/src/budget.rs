//! Filling one scheduling step's work under the step's room.
//!
//! Design §4.4: the mechanism states each step's room and rejects a `Plan`
//! that exceeds its token, call, graph-tier or slot limits as a policy
//! failure, so a policy charges itself as it fills the step. This is the
//! structure that does the charging, shared by every policy.
//!
//! Two budgets apply, as today:
//!
//! - the step's tokens and calls (`max_num_batched_tokens`, `max_batch`);
//! - the tokens the step may spend on work that writes the prefix cache,
//!   which is the whole step except while an epoch warms and bounds new
//!   prefix writes (`Room::prefill_tokens`, today's `Warming::prefill_tokens`).
//!
//! The budgets are the step's own: the mechanism starts every step with them
//! (today `assemble_batch` starts `budget = max_num_batched_tokens` on every
//! pass), and the work of earlier steps was charged to those steps. So nothing
//! is subtracted for work already in flight; [`Room::inflight`] says what those
//! steps sent, which the prefill window reads and the budget does not.
//!
//! A call is charged what it costs: a prefill chunk its tokens, a decode or
//! verification step its query tokens, a denoising step its latent units.

use crate::decision::{Room, Work};
use crate::ids::{PhaseId, RequestId};
use crate::view::PhaseKind;

/// The tokens and calls one step may still spend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shares {
    /// Tokens the step may still spend.
    tokens: u32,
    /// Calls the step may still add.
    calls: usize,
    /// Tokens the step may still spend on work that writes the prefix cache;
    /// `None` when nothing limits them beyond `tokens`.
    prefill: Option<u32>,
}

impl Shares {
    /// The shares of one step.
    pub fn new(tokens: u32, calls: usize, prefill: Option<u32>) -> Self {
        Self {
            tokens,
            calls,
            prefill,
        }
    }

    /// The shares a room states: the step's tokens and calls, and the
    /// warm-up's bound on prefix writes.
    pub fn of(room: &Room) -> Self {
        Self {
            tokens: room.token_budget,
            calls: room.call_budget,
            prefill: room.prefill_tokens,
        }
    }

    /// Tokens the step may still spend.
    pub fn tokens(&self) -> u32 {
        self.tokens
    }

    /// Calls the step may still add.
    pub fn calls(&self) -> usize {
        self.calls
    }

    /// Tokens the step may still spend on work that writes the prefix cache.
    pub fn prefill(&self) -> u32 {
        self.prefill.unwrap_or(self.tokens)
    }

    /// Whether no further call fits.
    pub fn full(&self) -> bool {
        self.calls == 0 || self.tokens == 0
    }

    /// Takes one call costing `tokens` out of the step, returning whether it
    /// fitted.
    ///
    /// A call that does not fit changes nothing, so a policy can leave the
    /// room to a later request, as the current scheduler does when a call
    /// cannot reserve its storage.
    pub fn charge(&mut self, tokens: u32) -> bool {
        if self.calls == 0 || tokens > self.tokens {
            return false;
        }
        self.calls -= 1;
        self.tokens -= tokens;
        true
    }

    /// Takes one call costing `tokens` that also writes the prefix cache out
    /// of both budgets, returning whether it fitted in both.
    pub fn charge_prefill(&mut self, tokens: u32) -> bool {
        if self.calls == 0 || tokens > self.tokens || tokens > self.prefill() {
            return false;
        }
        self.calls -= 1;
        self.tokens -= tokens;
        if let Some(prefill) = self.prefill.as_mut() {
            *prefill -= tokens;
        }
        true
    }
}

/// Returns the tokens of the next prompt chunk of a request whose next prompt
/// token is `start` and whose prompt is `prompt_tokens` long.
///
/// Design §4.4: the chunk is the least of the tokens left, the policy's
/// `chunked_prefill_size` and the tokens the step has left, and then ends at
/// the last KV page boundary at or before that end, so that a prefix hit can
/// reuse whole pages. A chunk that reaches the prompt end may end there, and a
/// chunk is never shorter than one token.
///
/// The start is `cached_prefix_tokens + prefilled_tokens`
/// ([`crate::RequestView`]); `page_tokens` is the KV page size
/// ([`crate::Room::page_tokens`]). A zero here means the prompt is done.
pub fn prefill_chunk(
    start: u32,
    prompt_tokens: u32,
    chunked_prefill_size: u32,
    share: u32,
    page_tokens: u32,
) -> u32 {
    let left = prompt_tokens.saturating_sub(start);
    if left == 0 {
        return 0;
    }
    // The three bounds, and at least one token so that a prompt always
    // advances: the step's share may be zero while the prompt is unfinished.
    let want = left
        .min(chunked_prefill_size.max(1))
        .min(share)
        .max(1)
        .min(left);
    // A chunk that reaches the prompt end may end there.
    if want >= left {
        return left;
    }
    let end = start.saturating_add(want);
    let aligned = end
        .checked_div(page_tokens)
        .map_or(end, |pages| pages * page_tokens);
    aligned.saturating_sub(start).max(1)
}

/// The token work one call of `kind` states, as the current scheduler charges
/// it (`call_token_cost` in `batching.rs`).
///
/// A prefill or refill chunk is the chunk the policy sizes; every other phase
/// states the charge the mechanism's view already states for the request's
/// next call ([`crate::RequestView::next_tokens`]: a verification's `1 +
/// drafts`, a denoising step's `latent units x guidance branches`, one for
/// decode).
pub fn work_tokens(kind: PhaseKind, chunk: u32, next_tokens: u32) -> u32 {
    match kind {
        PhaseKind::Prefill | PhaseKind::Refill => chunk,
        _ => next_tokens.max(1),
    }
}

/// The step being filled: its remaining shares and the work put in it so far.
///
/// A policy adds work in its own order and stops when nothing more fits; the
/// order is the decision, the charging is not.
#[derive(Clone, Debug)]
pub struct Step {
    shares: Shares,
    work: Vec<Work>,
}

impl Step {
    /// A step with the given shares and no work.
    pub fn new(shares: Shares) -> Self {
        Self {
            shares,
            work: Vec::new(),
        }
    }

    /// The shares the step has left.
    pub fn shares(&self) -> Shares {
        self.shares
    }

    /// Adds one call, returning whether it fitted. A call that does not fit is
    /// not added and leaves the room to later requests.
    ///
    /// `writes_prefix` says the call's phase writes the prefix cache, which
    /// draws on the warm-up bound as well as on the step's tokens
    /// ([`PhaseKind::writes_prefix_cache`]).
    pub fn add(
        &mut self,
        request: RequestId,
        phase: PhaseId,
        tokens: u32,
        writes_prefix: bool,
    ) -> bool {
        let charged = if writes_prefix {
            self.shares.charge_prefill(tokens)
        } else {
            self.shares.charge(tokens)
        };
        if !charged {
            return false;
        }
        self.work.push(Work {
            request,
            phase,
            tokens,
        });
        true
    }

    /// The work of the step, in the order it was added.
    pub fn into_work(self) -> Vec<Work> {
        self.work
    }

    /// How many calls the step holds.
    pub fn len(&self) -> usize {
        self.work.len()
    }

    /// Whether the step holds no work.
    pub fn is_empty(&self) -> bool {
        self.work.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_call_is_charged_its_tokens_and_one_call() {
        let mut shares = Shares::new(10, 2, None);
        assert!(shares.charge(4));
        assert_eq!((shares.tokens(), shares.calls()), (6, 1));
        assert!(shares.charge(6));
        assert!(shares.full());
        assert!(!shares.charge(1));
    }

    #[test]
    fn a_call_that_does_not_fit_charges_nothing() {
        let mut shares = Shares::new(5, 4, None);
        assert!(!shares.charge(6));
        assert_eq!((shares.tokens(), shares.calls()), (5, 4));
    }

    #[test]
    fn prefix_writing_work_draws_on_both_budgets() {
        let mut shares = Shares::new(10, 4, Some(3));
        assert!(!shares.charge_prefill(4));
        assert_eq!((shares.tokens(), shares.prefill()), (10, 3));
        assert!(shares.charge_prefill(3));
        assert_eq!((shares.tokens(), shares.prefill()), (7, 0));
        assert!(!shares.charge_prefill(1));
        assert!(shares.charge(1));
        assert_eq!(shares.tokens(), 6);
    }

    #[test]
    fn without_a_warm_up_bound_the_whole_step_is_the_bound() {
        let mut shares = Shares::new(8, 2, None);
        assert_eq!(shares.prefill(), 8);
        assert!(shares.charge_prefill(5));
        assert_eq!((shares.tokens(), shares.prefill()), (3, 3));
    }

    #[test]
    fn the_room_of_a_step_is_its_own_budgets() {
        let room = Room {
            token_budget: 100,
            call_budget: 8,
            prefill_tokens: Some(40),
            ..Room::default()
        };
        let shares = Shares::of(&room);
        assert_eq!(shares.tokens(), 100);
        assert_eq!(shares.calls(), 8);
        assert_eq!(shares.prefill(), 40);
    }

    #[test]
    fn the_chunk_is_clipped_to_every_limit() {
        // The prompt end clips it; a chunk that reaches it may end there.
        assert_eq!(prefill_chunk(0, 100, 512, 1024, 16), 100);
        // `chunked_prefill_size` clips it, then the page boundary.
        assert_eq!(prefill_chunk(0, 2000, 512, 4096, 64), 512);
        // The step's share clips it, then the page boundary.
        assert_eq!(prefill_chunk(0, 2000, 4096, 100, 16), 96);
        // A finished prompt has no chunk.
        assert_eq!(prefill_chunk(100, 100, 512, 1024, 16), 0);
    }

    #[test]
    fn a_chunk_that_reaches_the_prompt_end_may_end_there() {
        // Four tokens left, off a page boundary: the chunk is those four.
        assert_eq!(prefill_chunk(96, 100, 512, 1024, 16), 4);
        // A prompt shorter than one page.
        assert_eq!(prefill_chunk(0, 7, 512, 1024, 16), 7);
    }

    #[test]
    fn a_chunk_is_never_shorter_than_one_token() {
        assert_eq!(prefill_chunk(15, 100, 8, 8, 16), 1);
        assert_eq!(prefill_chunk(0, 100, 512, 0, 16), 1);
    }

    #[test]
    fn an_aligned_start_leaves_a_chunk_of_whole_pages() {
        assert_eq!(prefill_chunk(32, 2000, 100, 1000, 16), 96);
        // 64 + 4096 = 4160 is itself a page boundary (65 × 64), so the whole
        // share fits and nothing is clipped.
        assert_eq!(prefill_chunk(64, 5000, 4096, 4096, 64), 4096);
    }

    #[test]
    fn the_step_keeps_the_order_work_was_added_in() {
        let mut step = Step::new(Shares::new(8, 3, None));
        assert!(step.add(RequestId(1), PhaseId(0), 2, false));
        assert!(step.add(RequestId(2), PhaseId(1), 2, false));
        assert!(!step.add(RequestId(3), PhaseId(0), 9, true));
        assert!(step.add(RequestId(4), PhaseId(0), 4, true));
        assert!(step.shares().full());
        let work = step.into_work();
        let order: Vec<u64> = work.iter().map(|work| work.request.0).collect();
        assert_eq!(order, vec![1, 2, 4]);
    }

    #[test]
    fn a_chunk_states_its_tokens_and_another_phase_its_charge() {
        assert_eq!(work_tokens(PhaseKind::Prefill, 7, 5), 7);
        assert_eq!(work_tokens(PhaseKind::Refill, 7, 5), 7);
        assert_eq!(work_tokens(PhaseKind::Decode, 7, 1), 1);
        assert_eq!(work_tokens(PhaseKind::Verify, 7, 9), 9);
        assert_eq!(work_tokens(PhaseKind::Denoise, 7, 32), 32);
    }
}
