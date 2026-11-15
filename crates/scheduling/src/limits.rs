//! A policy's parameters and the limits it declares.
//!
//! A scheduling policy is a package with arguments (design §5.1): its manifest declares each
//! parameter it takes with its default (`[parameters]`), a deployment states the arguments it
//! changes, and the Build Server hands the package revision every declared parameter, the
//! deployment's value where it states one and the declared default otherwise. The process hands
//! those arguments to the policy at instantiation as [`Parameters`], and the policy returns its
//! [`Limits`] once. The mechanism checks the limits against the build config's ceilings; a limit
//! above a ceiling, or zero where the mechanism requires a positive count, is a policy failure
//! (design §4.4).

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::time::Nanos;

/// The order a waiting queue admits by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueOrder {
    /// The instant each request reached the main process.
    Arrival,
    /// The request's own priority, lower first, then arrival.
    Priority,
}

/// A scheduling policy package's arguments, as its revision states them: every parameter its
/// manifest declares, at the deployment's value or the declared default, and the arguments a
/// deployment states beyond those (design §7.7: `scheduler_config` and the `speculative_config`
/// decision values). `None` is a parameter the revision does not state; which of them a policy
/// requires is the policy's, and a revision that leaves out a required one fails the policy's
/// instantiation ([`Unstated`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Parameters {
    /// The order the waiting queue admits by.
    pub order: Option<QueueOrder>,
    /// Tokens of one prefill chunk.
    pub chunked_prefill_size: Option<u32>,
    /// The waiting queue's bound; unbounded when unstated.
    pub max_queued_requests: Option<usize>,
    /// Calls per step.
    pub max_batch: Option<usize>,
    /// Tokens per step. A package declares no default: the mechanism states the scheduler's own
    /// step budget where the deployment states none.
    pub max_num_batched_tokens: Option<u32>,
    /// The smallest prefix-hit deepening the policy asks for, in tokens.
    pub prefetch_min_tokens: Option<u32>,
    /// How long a running request that waits for no transfer may go without progressing before
    /// the mechanism ends it as starved, in nanoseconds (design §4.4).
    pub timeout: Option<Nanos>,
    /// Batches carrying prefill work that may be in flight before ready token progress takes the
    /// step instead.
    pub prefill_window: Option<u32>,
    /// The speedup speculation gives a request, stated in place of the
    /// measured one (`speculative_config.expected_speedup`).
    pub expected_speedup: Option<f64>,
    /// The cost of one refilled token, in decode steps
    /// (`speculative_config.refill_token_cost`).
    pub refill_token_cost: Option<f64>,
}

/// A parameter a policy requires that its package revision does not state: the package declares
/// no default for it and the deployment states none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unstated(pub &'static str);

impl fmt::Display for Unstated {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "the policy's package revision states no `{}`",
            self.0
        )
    }
}

impl std::error::Error for Unstated {}

/// `value`, the argument of the required parameter `name`, or [`Unstated`] naming it.
///
/// # Errors
///
/// [`Unstated`] when the revision does not state the parameter.
pub fn stated<T>(value: Option<T>, name: &'static str) -> Result<T, Unstated> {
    value.ok_or(Unstated(name))
}

/// The limits a policy declares, read once at instantiation.
///
/// The mechanism clamps its own hard constraints by them and rejects a
/// declaration that exceeds a build config ceiling; they are decisions of the
/// policy, not the mechanism's.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
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
    /// How long a running request may go without progressing before the
    /// policy has starved it.
    pub timeout: Nanos,
}
