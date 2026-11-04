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
pub use instance::{PolicyInstance, DEFAULT_CALL_FUEL};
pub use limits::MAX_STARVATION_TIMEOUT;
pub use module::PolicyModule;

/// The interface both sides are built from: the events, decisions, snapshot and
/// views of §4.4, the `Policy` trait a policy implements, and the wire format.
///
/// The mechanism and the policy modules are compiled from this one crate, so
/// there is no second set of types to keep in step.
pub use scheduling_guest as guest;
/// The calls of the interface, as the mechanism names them in a failure.
pub use scheduling_guest::wire::Call;
/// The `Policy` trait, which a policy implements inside its module.
pub use scheduling_guest::Policy;

/// The wasmtime version this crate is built against, which a policy package
/// declares in its `[toolchain]` and which the Program Server's
/// `scheduling-precompile` reports (§5.2, §5.3).
///
/// A precompiled module carries wasmtime's own artifact format and the
/// configuration it was compiled with, so a module precompiled by another
/// wasmtime version, or with another `Config`, is rejected at load. This
/// constant and the `wasmtime` version pinned in `Cargo.toml` are therefore
/// one fact, and a test below fails if they drift.
pub const WASMTIME_VERSION: &str = "50.0.1";

#[cfg(test)]
mod tests {
    use super::WASMTIME_VERSION;

    /// The declared wasmtime version is the one `Cargo.toml` pins: a package's
    /// `[toolchain]` names it, and a module precompiled against another version
    /// would not load.
    #[test]
    fn the_declared_wasmtime_version_is_the_one_cargo_pins() {
        let manifest = include_str!("../Cargo.toml");
        let pin = format!("version = \"{WASMTIME_VERSION}\"");
        assert!(
            manifest.contains(&pin),
            "Cargo.toml does not pin {pin}: the declared wasmtime version and the dependency drifted"
        );
    }

    /// The default policies' packages declare the same wasmtime version: the
    /// Program Server precompiles a package with the toolchain its manifest
    /// names, and a module precompiled by another version does not load.
    #[test]
    fn the_default_policies_declare_the_same_wasmtime_version() {
        let decl = format!("version = \"{WASMTIME_VERSION}\"");
        for (backend, manifest) in [
            ("fcfs", include_str!("../policies/fcfs/package.toml")),
            (
                "priority",
                include_str!("../policies/priority/package.toml"),
            ),
        ] {
            assert!(
                manifest.contains(&decl),
                "the {backend} package does not declare {decl}"
            );
        }
    }
}
