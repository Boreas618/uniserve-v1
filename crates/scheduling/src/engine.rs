//! The wasmtime engine policy modules are compiled for and executed in.
//!
//! A policy module is a `wasm32` Rust crate of `crates/scheduling/policies`
//! (§3, §4.4). The Program Server compiles it with `WasmSource` (§5.2),
//! precompiles it for the host's CPU with [`Engine::precompile`] and stores the
//! artifact in its module store (§5.3); the scheduler threads of the main
//! process load that artifact with [`Engine::module`].
//!
//! One engine serves a process: a module precompiled by one engine is rejected
//! by another whose configuration differs, so the process that precompiles and
//! the process that loads must agree, and the Program Server and the main
//! process build their engines the one way this type builds them.

use wasmtime::{Config, OptLevel};

use crate::error::PolicyError;

/// The engine every policy module of one process is compiled for and executed
/// in.
///
/// Cloning shares the engine; wasmtime's engine is the compilation context, not
/// a running policy.
#[derive(Clone)]
pub struct Engine {
    inner: wasmtime::Engine,
}

impl Engine {
    /// The target a policy module is precompiled for: the host architecture's
    /// baseline Linux triple.
    ///
    /// Naming the architecture and nothing finer is what makes the Program
    /// Server's artifact portable within it (§5.3): the module store keeps one
    /// artifact per architecture, and any host of that architecture loads it
    /// whatever its CPU model. A host whose architecture has no policy target
    /// fails here rather than compiling a module no other host can load.
    pub fn target() -> Option<&'static str> {
        match std::env::consts::ARCH {
            "x86_64" => Some("x86_64-unknown-linux-gnu"),
            "aarch64" => Some("aarch64-unknown-linux-gnu"),
            _ => None,
        }
    }

    /// Builds the process's engine.
    ///
    /// Fuel metering is enabled for the whole engine (§4.4: every call into a
    /// policy has a budget), backtraces are off because a failure is reported
    /// as its trap code alone, and the compiler runs at `Speed` because a
    /// policy compiled for a different optimisation level is not the artifact
    /// the Program Server stored. The target is [`Self::target`], so an
    /// artifact the Program Server precompiled for this architecture loads
    /// here even though the two engines are different objects.
    pub fn new() -> Result<Self, PolicyError> {
        let target = Self::target().ok_or_else(|| PolicyError::Load {
            message: format!(
                "the host architecture {} has no policy target",
                std::env::consts::ARCH
            ),
        })?;
        let mut config = Config::new();
        config.consume_fuel(true);
        config.cranelift_opt_level(OptLevel::Speed);
        // A failure is reported as its trap code alone, so no backtrace is
        // collected (`None`): the mechanism reverts the policy rather than
        // debugging it in the scheduler thread.
        config.wasm_backtrace_max_frames(None);
        config.target(target).map_err(|error| PolicyError::Load {
            message: format!("{target} is not a policy target: {error}"),
        })?;
        let inner = wasmtime::Engine::new(&config).map_err(|error| PolicyError::Load {
            message: format!("this host has no engine for wasm32 policy modules: {error}"),
        })?;
        Ok(Self { inner })
    }

    /// Precompiles a policy module for this host, which is what the Program
    /// Server stores (`WasmSource`, §5.3). `wasm` is the wasm32 binary the
    /// policy crate compiled to; text is accepted only where the `wat` feature
    /// is on, which is the tests.
    pub fn precompile(&self, wasm: &[u8]) -> Result<Vec<u8>, PolicyError> {
        self.inner
            .precompile_module(wasm)
            .map_err(|error| PolicyError::Load {
                message: format!("cannot precompile the policy module: {error}"),
            })
    }

    /// Loads a policy module from the bytes the Program Server serves: the
    /// precompiled artifact when the bytes are one, the wasm32 binary
    /// otherwise.
    pub fn module(&self, bytes: &[u8]) -> Result<crate::PolicyModule, PolicyError> {
        crate::PolicyModule::load(self, bytes)
    }

    /// The wasmtime engine itself, for the module and instance code of this
    /// crate.
    pub(crate) const fn inner(&self) -> &wasmtime::Engine {
        &self.inner
    }
}
