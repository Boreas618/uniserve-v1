//! Why a policy failed.
//!
//! Every failure here is a policy failure in the sense of §4.4: the mechanism
//! stops executing the decisions of this policy and reports it, so the daemon
//! reverts the update during probation, or, after a commit, the mechanism
//! instantiates the previous policy on the state it holds and reports
//! `PolicyReverted` (§7.3).
//!
//! The failures §4.4 names split in two. Those of the module itself are found
//! here: a trap, running past the fuel of one call, limits outside the build
//! config's ceilings, and an answer that breaks the call protocol. Those over a
//! decision are found by the mechanism and reported through the same type: a
//! decision naming an object that does not exist, does not fit or cannot be
//! chosen, work that breaks a hard limit, an expert order that is not a legal
//! placement, and a resident request starved past `limits().timeout`.
//!
//! Nothing here is recoverable inside the instance: a policy that trapped is
//! not called again.

use scheduling_guest::ids::RequestId;
use scheduling_guest::wire::Call;

/// A policy failure (§4.4).
#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    /// The module bytes were rejected: not a wasm32 module, a component, an
    /// artifact precompiled for another engine, or a package that does not
    /// compile.
    #[error("the policy module was rejected: {message}")]
    Load { message: String },
    /// The module compiled but cannot be instantiated: an export the interface
    /// needs is missing or has the wrong signature, or instantiation itself
    /// failed.
    #[error("the policy module cannot be instantiated: {message}")]
    Instantiate { message: String },
    /// The policy trapped in a call: `unreachable`, an out-of-bounds access, a
    /// failed allocation, a panic in a Rust policy.
    #[error("the policy trapped in `{}`: {trap}", call.method())]
    Trap { call: Call, trap: wasmtime::Trap },
    /// The policy ran past the fuel its instance allows one call.
    #[error("the policy ran past its fuel in `{}`", call.method())]
    FuelExhausted { call: Call },
    /// `limits()` returned values the build config does not allow: above a
    /// ceiling the mechanism gave, zero where a positive count is required, or
    /// a timeout above the 30 seconds §4.4's watchdog allows.
    #[error("the policy's limits are outside the build config: {message}")]
    BadLimits { message: String },
    /// The policy's answer breaks the call protocol: a buffer the module cannot
    /// hold, bytes that do not decode into the interface's type, or a decision
    /// that names an object which does not exist or cannot be chosen.
    #[error("the policy's answer in `{}` is malformed: {message}", call.method())]
    Malformed { call: Call, message: String },
    /// The mechanism's watchdog found a resident request that did not progress
    /// past `limits().timeout` (§4.4). This is the failure sched_ext's watchdog
    /// makes: a runnable task that waited longer than the scheduler declared is
    /// the scheduler's failure, not the task's.
    #[error("the policy starved request {} past its timeout", request.0)]
    Starved { request: RequestId },
    /// A call on an instance whose earlier call failed. An instance that
    /// trapped or ran out of fuel keeps nothing a later call could use, so the
    /// mechanism drops it and instantiates the policy it keeps from the module
    /// (§7.3).
    #[error("the policy instance failed in `{}` earlier and is not called again", call.method())]
    Poisoned { call: Call },
}

impl PolicyError {
    /// The call this failure happened in, when it belongs to one.
    pub const fn call(&self) -> Option<Call> {
        match self {
            Self::Load { .. }
            | Self::Instantiate { .. }
            | Self::BadLimits { .. }
            | Self::Starved { .. } => None,
            Self::Trap { call, .. }
            | Self::FuelExhausted { call }
            | Self::Malformed { call, .. }
            | Self::Poisoned { call } => Some(*call),
        }
    }

    /// A decision the mechanism found invalid, reported as a policy failure
    /// (§4.4: a decision naming an object that does not exist, does not fit or
    /// cannot be chosen is the policy's failure, not the mechanism's).
    pub fn invalid(call: Call, message: impl Into<String>) -> Self {
        Self::Malformed {
            call,
            message: message.into(),
        }
    }
}
