//! Scheduling policy and configured queue and batch limits.
//!
//! These are requested limits: `Scheduler::new` clamps the token
//! and sequence limits, and the batch limit when the executor advertises a
//! nonzero call limit, to the executor's capacity.

use proto::request::SchedulingPolicy;

/// Default maximum calls selected by one generation scheduling pass.
pub const DEFAULT_MAX_BATCH: usize = 128;
/// Default token budget for one generation scheduling pass.
pub const DEFAULT_MAX_NUM_BATCHED_TOKENS: usize = 8192;
/// Default maximum number of concurrently running requests.
pub const DEFAULT_MAX_RUNNING_REQUESTS: usize = 128;
/// Default token threshold above which a prefill is chunked.
pub const DEFAULT_CHUNKED_PREFILL_SIZE: usize = DEFAULT_MAX_NUM_BATCHED_TOKENS;
/// When a request whose draft cache is missing has it refilled.
///
/// A refill evaluates the target model again over the request's context
/// before the first position whose draft cache entry is missing, the cost
/// of a prefill of those tokens; once refilled, the request speculates.
/// With `g` the speedup speculation gives a request (the decode steps' time
/// one of its verification steps takes the place of) and `k` the time of
/// evaluating one context token, in decode steps, a refill of `tokens`
/// tokens of a request that may still generate `remaining` tokens is worth
/// it when `remaining * (1 - 1 / g) > tokens * k`.
///
/// The scheduler measures both from what its world executes
/// (`SpeculationMeter`); a value stated here takes the place of the
/// measured one. Until the world has measured a value it needs and none is
/// stated, a request is refilled: its refill and its speculation measure
/// it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RefillPolicy {
    /// The speedup `g` stated in place of the measured one; at least one.
    pub expected_speedup: Option<f64>,
    /// The cost `k` of one refilled token, in decode steps, stated in place
    /// of the measured one; not negative.
    pub token_cost: Option<f64>,
}

impl RefillPolicy {
    /// Whether refilling `tokens` context tokens of a request that may
    /// still generate `remaining` tokens is worth its cost, with the world's
    /// measured speedup and token cost where none is stated.
    pub fn worth(
        &self,
        measured_speedup: Option<f64>,
        measured_token_cost: Option<f64>,
        remaining: usize,
        tokens: usize,
    ) -> bool {
        let (Some(speedup), Some(cost)) = (
            self.expected_speedup.or(measured_speedup),
            self.token_cost.or(measured_token_cost),
        ) else {
            return true;
        };
        let saved = remaining as f64 * (1.0 - 1.0 / speedup.max(1.0));
        saved > tokens as f64 * cost.max(0.0)
    }
}

#[derive(Clone, Copy, Debug)]
/// Sequence, token, and queue limits for scheduling.
pub struct SchedulerConfig {
    /// Waiting-queue ordering policy.
    pub policy: SchedulingPolicy,
    /// Maximum calls selected by one generation scheduling pass, across the
    /// per-call-kind batches the pass emits.
    pub max_batch: usize,
    /// Token budget of one generation scheduling pass. Each selected call is
    /// charged its token work, which for a denoising call covers its latent
    /// units and guidance branches.
    pub max_num_batched_tokens: usize,
    /// Maximum concurrently running requests, token and media together.
    pub max_running_requests: usize,
    /// Per-request token ceiling for one prefill chunk.
    pub chunked_prefill_size: usize,
    /// Admission limit on waiting requests, or `None` to queue without
    /// bound. Submission counts waiting token and media requests together
    /// with finished requests whose output events are still retained.
    pub max_queued_requests: Option<usize>,
    /// Block-diffusion sampling of every canvas the deployment generates, as
    /// the ranks are launched with. A block-diffusion runtime has no
    /// autoregressive decode, so each request it does not read out runs
    /// under this sampling; without it the runtime serves readouts only.
    pub canvas: Option<ipc::CanvasSampling>,
    /// The most draft tokens a verification carries per request, as the
    /// world's workers were launched with (`EngineSpec::speculation`); zero
    /// when the deployment does not speculate.
    pub draft_tokens: usize,
    /// When a request whose draft cache is missing has it refilled.
    pub refill: RefillPolicy,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            policy: SchedulingPolicy::Fcfs,
            max_batch: DEFAULT_MAX_BATCH,
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_running_requests: DEFAULT_MAX_RUNNING_REQUESTS,
            chunked_prefill_size: DEFAULT_CHUNKED_PREFILL_SIZE,
            max_queued_requests: None,
            canvas: None,
            draft_tokens: 0,
            refill: RefillPolicy::default(),
        }
    }
}
