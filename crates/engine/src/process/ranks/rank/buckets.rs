//! The buckets a rank captures and the phases they serve, as the scheduler names them in a
//! batch.

use crate::batch::Phase;
use crate::batch::capacity::variants;
use crate::runner::buckets::Buckets;

/// The refusal a stated graph tier earns, if any.
///
/// The two bounds are not the same kind of thing, and this is where that is decided:
///
/// - **The token bound is a limit a policy is checked against.** The scheduler holds its
///   policy's `max_num_batched_tokens` to the largest token count its groups' buckets hold
///   (`crate::scheduler::policy::ceilings`, from `Capacity::buckets`), and a policy that declares
///   more is refused. A token tier capture drops therefore changes what a deployment's policy may
///   declare, and the deployment would find that out at its precopy - after weights, from a
///   sentence about two numbers its own file does not contain. It is refused here instead.
/// - **The row bound only sizes a batch.** A decode tier above the request bound changes no
///   policy's declaration: the scheduler forms batches of at most the group's slots
///   (`Capacity::holds`), whatever its buckets hold, which is what a deployment whose limits move
///   in place relies on - its tiers outlive one setting of the bound. It refuses nothing.
///
/// A zero states no size at all and is dropped, as it always was.
pub(super) fn tier_refusal(
    decode: Option<&[u32]>,
    rows: u32,
    prefill: &[u32],
    longest: u32,
) -> Option<String> {
    // Named for the reader, decided as below: rows within the bound or not, this is not a refusal.
    let _ = (decode, rows);
    let stated = prefill
        .iter()
        .copied()
        .find(|size| *size != 0 && *size > longest)?;
    Some(format!(
        "cuda_graph_config.prefill_token_sizes states {stated}, above the {longest} that \
         max_model_len admits: the bucket would be captured and never filled, and the catalogue a \
         policy's token budget is checked against is the one capture keeps - so this rank would \
         refuse a policy against a bound the deployment never named"
    ))
}

/// The phases a runner over `buckets` serves: every phase a batch of which may be launched as the
/// variant of one of the buckets (`crate::batch::capacity::variants`, the scheduler's own rule), in
/// the order of [`Phase::ALL`].
///
/// The rule is read from the scheduler's side so the two cannot disagree: a prefill bucket
/// serves `Prefill` and `Refill` (a refill chunk is a prefill-class chunk of a decoding request),
/// a decode bucket `Decode`, a verification bucket `Verify` (a speculating deployment's decode
/// steps are verifications, [`Buckets::with_drafts`]), and a host bucket the host phase of its
/// variant. The drafter's bucket and an expert join serve no phase: the worker issues the first
/// behind the batch it follows, and a union member's join batch states the phase of the member
/// that asks for it.
pub(super) fn served_phases(buckets: &Buckets) -> Vec<Phase> {
    Phase::ALL
        .into_iter()
        .filter(|phase| {
            buckets
                .iter()
                .any(|bucket| variants(*phase).contains(&bucket.variant))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rank serves the phases its buckets are launched as, and no phase it does not run.
    #[test]
    fn a_model_runner_serves_the_phases_its_buckets_are_planned_as() {
        let buckets = Buckets::causal_lm(&[8], &[64], true);

        assert_eq!(
            served_phases(&buckets),
            vec![Phase::Prefill, Phase::Decode, Phase::Refill],
            "a prefill and a decode batch route to the rank, and a refill chunk runs as a prefill"
        );
    }

    /// A speculating rank's decode steps are verifications: it serves `Verify` and no `Decode`,
    /// and the drafter's buckets, which no batch states, serve no phase.
    #[test]
    fn a_speculating_rank_verifies_and_its_drafter_serves_no_phase() {
        let buckets = Buckets::causal_lm(&[8], &[64], true).with_drafts(3);

        assert_eq!(
            served_phases(&buckets),
            vec![Phase::Prefill, Phase::Verify, Phase::Refill]
        );
    }

    /// A deployment whose stated token tier is longer than its model is refused before weights,
    /// and the refusal names all three figures: the option, the size it states and the bound.
    #[test]
    fn a_stated_prefill_tier_above_the_model_is_refused_naming_all_three_figures() {
        let refusal = tier_refusal(None, 8, &[256, 512, 1024], 512)
            .expect("a stated prefill tier above the model was accepted");

        assert!(
            refusal.contains("cuda_graph_config.prefill_token_sizes"),
            "the option is not named: {refusal}"
        );
        assert!(
            refusal.contains("states 1024"),
            "the stated figure is not named: {refusal}"
        );
        assert!(
            refusal.contains("above the 512 that max_model_len admits"),
            "the captured figure or its key is not named: {refusal}"
        );
    }

    /// A deployment whose stated decode tiers exceed its request bound still starts: that bound
    /// sizes a batch, nothing checks a policy against it, and the limit moves in place.
    #[test]
    fn a_stated_decode_tier_above_the_request_bound_is_not_refused() {
        assert_eq!(
            tier_refusal(Some(&[1, 2, 4, 8, 16]), 4, &[], 4096),
            None,
            "a decode tier above the request bound was refused, which the live-update deployment \
             states deliberately"
        );
        // And a zero in a stated catalogue is a size of nothing, not a refusal.
        assert_eq!(
            tier_refusal(None, 8, &[0, 4, 512], 512),
            None,
            "a zero in a stated prefill catalogue was refused"
        );
    }
}
