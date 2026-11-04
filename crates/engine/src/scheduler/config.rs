//! Configured queue, token, and batch limits for scheduling.
//!
//! These are requested limits: `Scheduler::new` clamps the token
//! and sequence limits, and the batch limit when the executor advertises a
//! nonzero call limit, to the executor's capacity.

/// Default maximum calls selected by one generation scheduling pass.
pub const DEFAULT_MAX_BATCH: usize = 128;
/// Default token budget for one generation scheduling pass.
pub const DEFAULT_MAX_NUM_BATCHED_TOKENS: usize = 8192;
/// Default maximum number of concurrently running requests.
pub const DEFAULT_MAX_RUNNING_REQUESTS: usize = 128;
/// Default token threshold above which a prefill is chunked.
pub const DEFAULT_CHUNKED_PREFILL_SIZE: usize = DEFAULT_MAX_NUM_BATCHED_TOKENS;

#[derive(Clone, Debug)]
/// Sequence, token, and queue limits for scheduling.
pub struct SchedulerConfig {
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
    /// The captured graph sizes the deployment states, in tokens (the spec's
    /// `cuda_graphs` decode batch sizes and prefill token sizes), ascending and
    /// without duplicates: `Room.graph_tiers`, the bound a plan's batch must
    /// fit.
    ///
    /// Empty when the spec states none, and on a reconfigure, which carries no
    /// graph catalog: a worker then runs its own captured catalog or eagerly,
    /// and the tier bound is honest but vacuous. The worker echoing the graph
    /// sizes it actually launched is the open item that closes it.
    pub graph_tiers: Vec<u32>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch: DEFAULT_MAX_BATCH,
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_running_requests: DEFAULT_MAX_RUNNING_REQUESTS,
            chunked_prefill_size: DEFAULT_CHUNKED_PREFILL_SIZE,
            max_queued_requests: None,
            canvas: None,
            draft_tokens: 0,
            graph_tiers: Vec::new(),
        }
    }
}
