//! The policy's parameters and the limits it declares.
//!
//! A scheduling policy is a program with arguments (design §5.1). The
//! arguments that shape scheduling are [`Parameters`]: the process hands them
//! to the policy at instantiation, and the policy returns its [`Limits`] once
//! at instantiation. The mechanism checks the limits against the build
//! config's ceilings; a limit above a ceiling, or zero where the mechanism
//! requires a positive count, is a policy failure (design §4.4).

use serde::{Deserialize, Serialize};

use crate::time::Nanos;

/// The scheduling policy's arguments, as the deployment config states them
/// (design §7.7: `scheduler_config` and the `speculative_config` decision
/// values). `None` means the policy's own default.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Parameters {
    /// Tokens of one prefill chunk.
    #[serde(default)]
    pub chunked_prefill_size: Option<u32>,
    /// The waiting queue's bound; `None` queues without bound.
    #[serde(default)]
    pub max_queued_requests: Option<usize>,
    /// Calls per step.
    #[serde(default)]
    pub max_batch: Option<usize>,
    /// Tokens per step.
    #[serde(default)]
    pub max_num_batched_tokens: Option<u32>,
    /// The smallest prefix-hit deepening the policy asks for.
    #[serde(default)]
    pub prefetch_min_tokens: Option<u32>,
    /// How long a resident request may go without progressing before the
    /// policy has starved it (design §4.4; the mechanism caps it at 30 s, as
    /// sched_ext caps `timeout_ms`).
    #[serde(default)]
    pub timeout: Option<Nanos>,
    /// The speedup speculation gives a request, stated in place of the
    /// measured one (`speculative_config.expected_speedup`).
    #[serde(default)]
    pub expected_speedup: Option<f64>,
    /// The cost of one refilled token, in decode steps
    /// (`speculative_config.refill_token_cost`).
    #[serde(default)]
    pub refill_token_cost: Option<f64>,
}

/// The limits a policy declares, read once at instantiation.
///
/// The mechanism clamps its own hard constraints by them and rejects a
/// declaration that exceeds a build config ceiling; they are decisions of the
/// policy, not the mechanism's.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Limits {
    /// Tokens of one prefill chunk.
    pub chunked_prefill_size: u32,
    /// The waiting queue's bound; `None` queues without bound.
    pub max_queued_requests: Option<usize>,
    /// Calls per step.
    pub max_batch: usize,
    /// Tokens per step.
    pub max_num_batched_tokens: u32,
    /// The smallest prefix-hit deepening the policy asks for.
    pub prefetch_min_tokens: u32,
    /// How long a resident request may go without progressing before the
    /// policy has starved it.
    pub timeout: Nanos,
}

impl Default for Limits {
    /// The values today's scheduler defaults to (`SchedulerConfig::default`),
    /// so a policy that states nothing reproduces them.
    fn default() -> Self {
        Self {
            chunked_prefill_size: 8192,
            max_queued_requests: None,
            max_batch: 128,
            max_num_batched_tokens: 8192,
            prefetch_min_tokens: 256,
            timeout: Nanos(30_000_000_000),
        }
    }
}

impl Limits {
    /// The limits a policy's parameters state, on today's defaults where the
    /// parameters are silent.
    pub fn from_parameters(parameters: &Parameters) -> Self {
        let defaults = Self::default();
        Self {
            chunked_prefill_size: parameters
                .chunked_prefill_size
                .unwrap_or(defaults.chunked_prefill_size),
            max_queued_requests: parameters.max_queued_requests,
            max_batch: parameters.max_batch.unwrap_or(defaults.max_batch),
            max_num_batched_tokens: parameters
                .max_num_batched_tokens
                .unwrap_or(defaults.max_num_batched_tokens),
            prefetch_min_tokens: parameters
                .prefetch_min_tokens
                .unwrap_or(defaults.prefetch_min_tokens),
            timeout: parameters.timeout.unwrap_or(defaults.timeout),
        }
    }
}
