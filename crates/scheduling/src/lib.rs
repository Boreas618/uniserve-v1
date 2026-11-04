//! The scheduling interface between the scheduler's mechanism and its policy
//! (§4.4).
//!
//! The scheduler is split the way `sched_ext` splits the Linux scheduler: the
//! mechanism (`crates/engine`) owns request state, storage, the hard limits and
//! execution, and the policy decides. The mechanism hands the policy its events
//! when state changes and asks it for one decision at each decision point; the
//! policy holds its own soft state, and the mechanism validates and executes
//! each decision.
//!
//! This crate is the host side of that boundary: it loads the policy's wasm32
//! module and runs it in the scheduler thread under wasmtime, with a fuel budget
//! on every call. The interface itself — [`Policy`], `Plan`, `Room`, `Event` and
//! the views — lives in `crates/scheduling/guest` together with the wire
//! encoding, because the policy crates are compiled against the same types. The
//! default policies are `fcfs` and `priority` of `crates/scheduling/policies`.
//!
//! The three failures of §4.4 that belong to the module itself are reported as
//! [`PolicyError`]: a trap, running past the fuel, and limits outside the build
//! config. A decision that names something that does not exist or cannot be
//! chosen, work that breaks a hard limit, an expert order that is not a
//! permutation, and a resident request starved past `limits().timeout` are
//! found by the mechanism, which fails the policy through the same type. Either
//! way the policy has failed: during probation the update is reverted, and after
//! a commit the mechanism instantiates the previous policy on the state it holds
//! (§7.3).

mod engine;
mod error;
mod instance;
mod limits;
mod module;

pub use engine::Engine;
pub use error::PolicyError;
pub use instance::{DEFAULT_CALL_FUEL, PolicyInstance};
pub use limits::MAX_STARVATION_TIMEOUT;
pub use module::PolicyModule;

/// The interface both sides are built from: the events, decisions, snapshot and
/// views of §4.4, the `Policy` trait a policy implements, and the wire format.
///
/// The mechanism and the policy modules are compiled from this one crate, so
/// there is no second set of types to keep in step.
pub use scheduling_guest as guest;
/// The `Policy` trait, which a policy implements inside its module.
pub use scheduling_guest::Policy;
/// The calls of the interface, as the mechanism names them in a failure.
pub use scheduling_guest::wire::Call;
