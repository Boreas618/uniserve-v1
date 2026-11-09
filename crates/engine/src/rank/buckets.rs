//! The buckets a rank captures and the phases they serve, in the numbering the scheduler plans
//! with.

use ipc::PhaseId;
use worker::plan::{Buckets, Variant};

/// The buckets of a member of an expert union: each causal bucket, and the join, captured for the
/// step sizes it meets.
///
/// A union's ranks agree a step's rows per rank before they launch, and the expert work of a graph
/// is shaped for the rows it holds. Every rank decoding agrees at most the largest decode bucket's
/// rows, which the smallest graph holds; a decode beside a peer's prefill of a few prompts takes a
/// middle one, and a step a peer fills with a large prefill takes the graph of the whole capacity.
/// One graph for every step would shape each decode's expert work for the largest prefill any rank
/// could send.
pub(super) fn expert_classes(buckets: &Buckets, capacity: u64) -> Buckets {
    let capacity = u32::try_from(capacity).unwrap_or(u32::MAX);
    let decode = buckets
        .iter()
        .filter(|bucket| bucket.variant.is_decode())
        .map(|bucket| bucket.rows)
        .max()
        .unwrap_or(1);
    Buckets::new(
        buckets
            .iter()
            .copied()
            .chain(std::iter::once(worker::plan::Bucket::expert_join()))
            .flat_map(|bucket| {
                // A step this graph would serve is within a quarter of the whole capacity, where the
                // smaller shape saves little and the capture still costs its warmup's time.
                let small = bucket.tokens.max(decode).min(capacity);
                let mut classes = vec![bucket.at_capacity(capacity)];
                if small.saturating_mul(4) <= capacity {
                    classes.push(bucket.at_capacity(small));
                }
                // A decode beside a peer's prefill of a few prompts: the step is that prefill's
                // rows, which a graph of the whole capacity would shape every expert layer for.
                let beside = bucket.variant.is_decode()
                    || bucket.variant == worker::plan::Variant::ExpertJoin;
                if beside && small < PREFILL_BESIDE && PREFILL_BESIDE.saturating_mul(4) <= capacity
                {
                    classes.push(bucket.at_capacity(PREFILL_BESIDE));
                }
                classes.into_iter()
            }),
    )
}

/// The rows per rank of a step in which a peer prefills a few prompts while this rank decodes.
const PREFILL_BESIDE: u32 = 1024;

/// One epoch's build config, resolved against the node the main process runs on.
/// The catalogue of prefill buckets a rank captures, and where its bound came from.
///
/// The bound is the largest token count any captured bucket holds, which is also the ceiling a
/// policy's `max_num_batched_tokens` is checked against: a policy that wants a wider step
/// than the rank's graphs can serve is refused, and the refusal is only actionable when it says
/// which figure came from which place. `stated` separates the deployment's own
/// `cuda_graph.prefill_token_sizes` from the build's default catalogue, because a deployment that
/// configured none cannot find its bound in its own launch file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefillBound {
    /// The largest prefill token count any captured bucket holds.
    pub(crate) bound: u32,
    /// Whether the deployment stated the sizes rather than taking the build's default catalogue.
    pub(crate) stated: bool,
}

/// The refusal a stated graph tier earns, if any.
///
/// The two bounds are not the same kind of thing, and this is where that is decided:
///
/// - **The token bound is a contract a policy is checked against.** `ceilings_of` gives the
///   scheduler's policy `max_num_batched_tokens` from the catalogue's captured token bound
///   (`crates/scheduler/src/scheduler/policy.rs`), and a policy that declares more is refused by the
///   mechanism. A token tier capture drops therefore changes what a deployment's policy may
///   declare, and the deployment would find that out at its precopy - after weights, from a
///   sentence about two numbers its own file does not contain. It is refused here instead.
/// - **The row bound only sizes a batch.** A decode tier above the request bound changes no
///   policy's declaration: `Scheduler::new` lowers the limit, names both figures and keeps serving
///   (`crates/scheduler/src/scheduler/run.rs`), which is what a deployment whose limits move in place
///   relies on - its tiers outlive one setting of the bound. It refuses nothing.
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

/// The phases this node's runner serves, in the numbering the scheduler plans with: one id per phase
/// kind this rank's own buckets are launched as.
///
/// The kinds come from the buckets - what this rank's model runner can launch - and the ids from
/// `scheduler::phase_id`, the table the scheduler's `Plan` numbers a request's work with. Never from a
/// literal: a model runner that stated `PhaseId(0)` for its device (text encoding) served no phase the
/// scheduler ever planned, and the worker refused the deployment's first completion with `no
/// device of this rank serves phase PhaseId(3)`.
/// The phases the artifact's three host calls are planned as.
///
/// They serve no graph, so no bucket states them: the mechanism plans each call by its phase, and a
/// device that did not state these three would refuse the first encode of every request.
/// The one host call every rank serves: the encode of the units it decoded.
pub(super) fn unit_phases() -> Vec<PhaseId> {
    vec![PhaseId(
        u16::try_from(scheduler::phase_id(scheduler::PhaseKind::VideoEncode).0)
            .expect("a phase id fits"),
    )]
}

/// The phase id `kind` is planned as.
pub(super) fn phase(kind: scheduler::PhaseKind) -> PhaseId {
    PhaseId(u16::try_from(scheduler::phase_id(kind).0).expect("a phase id fits"))
}

pub(super) fn host_phases() -> Vec<PhaseId> {
    [
        scheduler::PhaseKind::VideoEncode,
        scheduler::PhaseKind::AudioEncode,
        scheduler::PhaseKind::Mux,
        scheduler::PhaseKind::MediaRead,
        scheduler::PhaseKind::VisionEncoding,
        scheduler::PhaseKind::LatentEncoding,
    ]
    .into_iter()
    .map(|kind| PhaseId(u16::try_from(scheduler::phase_id(kind).0).expect("a phase id fits")))
    .collect()
}

pub(super) fn served_phases(buckets: &Buckets) -> Vec<PhaseId> {
    let mut kinds: Vec<scheduler::PhaseKind> = buckets
        .iter()
        .map(|bucket| match bucket.variant {
            Variant::PrefillWrite | Variant::PrefillOutput | Variant::PrefillGreedy => {
                scheduler::PhaseKind::Prefill
            }
            Variant::DecodeGreedy | Variant::DecodeSampled => scheduler::PhaseKind::Decode,
            Variant::Encode => scheduler::PhaseKind::TextEncoding,
            Variant::Denoise => scheduler::PhaseKind::Denoise,
            // The scheduler plans a clip's decode as `VideoDecode` and a track's as `AudioDecode`
            // (`call_phase`), so a device that advertised `ImageDecode` for either served no phase
            // the scheduler ever planned - the same defect this function's doc records.
            Variant::Decode => scheduler::PhaseKind::VideoDecode,
            Variant::AudioDecode => scheduler::PhaseKind::AudioDecode,
            // The artifact's three host calls are calls the mechanism submits and waits on, so the
            // device states the phases they run in even though no graph serves them.
            Variant::VideoEncode => scheduler::PhaseKind::VideoEncode,
            Variant::AudioEncode => scheduler::PhaseKind::AudioEncode,
            Variant::Mux => scheduler::PhaseKind::Mux,
            Variant::LatentPrepare => scheduler::PhaseKind::LatentPrepare,
            Variant::MediaRead => scheduler::PhaseKind::MediaRead,
            // The two encoders whose work a later call computes are answered as they arrive, under
            // the phase of either: a bucket of this variant names no one of them.
            Variant::ConditionEncode => scheduler::PhaseKind::VisionEncoding,
            // A join is no phase a scheduler plans: a peer opens the step, and the union wakes
            // this rank for it.
            Variant::ExpertJoin => scheduler::PhaseKind::Decode,
        })
        .collect();
    // A refill chunk is a prefill-class chunk of a decoding request and runs wherever a prefill
    // chunk does, so the device that serves prefill states it as well: the scheduler may name a
    // refill and the batch then routes here.
    if kinds.contains(&scheduler::PhaseKind::Prefill) {
        kinds.push(scheduler::PhaseKind::Refill);
    }
    kinds.sort_by_key(|kind| scheduler::phase_id(*kind).0);
    kinds.dedup();
    kinds
        .into_iter()
        // The scheduler numbers a phase with the epoch's `u32` and the worker names one with the
        // wire's `u16`: the plan is where the two meet, and a kind that did not fit would be a table
        // this process cannot state rather than a phase to serve quietly.
        .map(|kind| PhaseId(u16::try_from(scheduler::phase_id(kind).0).expect("a phase id fits")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rank serves the phases its buckets are launched as, in the scheduler's numbering, and no
    /// phase it does not run.
    #[test]
    fn a_model_runner_serves_the_phases_its_buckets_are_planned_as() {
        let buckets = Buckets::causal_lm(&[8], &[64], true);
        let phases = served_phases(&buckets);
        let id =
            |kind| PhaseId(u16::try_from(scheduler::phase_id(kind).0).expect("a phase id fits"));
        assert!(
            phases.contains(&id(scheduler::PhaseKind::Prefill)),
            "a prefill batch routes to this node's device: {phases:?}"
        );
        assert!(
            phases.contains(&id(scheduler::PhaseKind::Decode)),
            "a decode batch routes to this node's device: {phases:?}"
        );
        assert!(
            !phases.contains(&id(scheduler::PhaseKind::TextEncoding)),
            "this node encodes no request itself: {phases:?}"
        );
        assert!(
            !phases.contains(&id(scheduler::PhaseKind::Denoise)),
            "a batch this plan has no bucket for is not served: {phases:?}"
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
