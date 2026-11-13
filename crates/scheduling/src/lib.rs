//! The host side of the scheduling policy: the wasmtime engine a policy module is compiled for
//! and loaded with, and the wasmtime version a policy package declares.
//!
//! A policy is a `wasm32` module of `crates/scheduling/policies`, built against the interface of
//! `crates/scheduling/guest`. The Build Server precompiles it with the `scheduling-precompile`
//! binary of this crate, and the engine's schedulers load it with the same [`Engine`] and run it
//! under fuel (`engine::scheduler::policy`).

mod engine;
mod error;

pub use engine::Engine;
pub use error::PolicyError;

/// The wasmtime version this crate is built against, which a policy package
/// declares in its `[toolchain]` and which the Build Server's
/// `scheduling-precompile` reports (§5.2, §5.3).
///
/// A precompiled module carries wasmtime's own artifact format and the
/// configuration it was compiled with, so a module precompiled by another
/// wasmtime version, or with another `Config`, is rejected at load. This
/// constant and the `wasmtime` version the workspace's `Cargo.toml` pins are
/// therefore one fact, and a test below fails if they drift.
pub const WASMTIME_VERSION: &str = "50.0.1";

#[cfg(test)]
mod tests {
    use super::WASMTIME_VERSION;

    /// The declared wasmtime version is the one the workspace's `Cargo.toml`
    /// pins for this crate and the engine: a package's `[toolchain]` names it,
    /// and a module precompiled against another version would not load.
    #[test]
    fn the_declared_wasmtime_version_is_the_one_cargo_pins() {
        let manifest = include_str!("../../../Cargo.toml");
        let pin = format!("version = \"{WASMTIME_VERSION}\"");
        assert!(
            manifest.contains(&pin),
            "Cargo.toml does not pin {pin}: the declared wasmtime version and the dependency drifted"
        );
    }

    /// The default policies' packages declare the same wasmtime version: the
    /// Build Server precompiles a package with the toolchain its manifest
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
