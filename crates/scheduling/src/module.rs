//! A policy module as the scheduler thread keeps it: loaded once, instantiated
//! as often as the current state requires.
//!
//! §4.4 keeps an instance per scheduler thread, and the switch keeps the
//! previous policy's *module* — not its instance — so that a revert
//! instantiates it on the state the mechanism holds at that moment (§7.3:
//! "调度器在下一步以当前的快照实例化上一个调度策略并换上"). That is why a
//! module and an instance are separate values here: a module is immutable and
//! cheap to clone and to instantiate again, and an instance holds all the
//! state.

use std::sync::Arc;

use scheduling_guest::limits::{Limits, Parameters};
use scheduling_guest::view::Snapshot;
use wasmtime::{Module, Precompiled};

use crate::error::PolicyError;
use crate::instance::PolicyInstance;
use crate::Engine;

/// A loaded policy module: the compiled wasm32 module of one policy backend and
/// revision (§5.1), ready to be instantiated.
///
/// Cloning shares the compiled module, so keeping the previous policy costs one
/// `Arc` and nothing of the compiler.
#[derive(Clone)]
pub struct PolicyModule {
    engine: Engine,
    module: Arc<Module>,
}

impl PolicyModule {
    /// Loads the bytes the Program Server serves for a policy program: the
    /// precompiled artifact when the bytes are one (§5.3, "按本机 CPU 预编译"),
    /// the wasm32 binary otherwise.
    pub fn load(engine: &Engine, bytes: &[u8]) -> Result<Self, PolicyError> {
        match wasmtime::Engine::detect_precompiled(bytes) {
            Some(Precompiled::Module) => Self::from_precompiled(engine, bytes),
            Some(Precompiled::Component) => Err(PolicyError::Load {
                message: "a policy is a core wasm module, not a component".into(),
            }),
            None => Self::from_wasm(engine, bytes),
        }
    }

    /// Loads a wasm32 binary module and compiles it.
    ///
    /// The bytes are the package's compiled crate; a deployment serves the
    /// artifact [`Engine::precompile`] made instead, so this is the path a
    /// policy takes when the store holds no precompiled artifact for this host.
    pub fn from_wasm(engine: &Engine, bytes: &[u8]) -> Result<Self, PolicyError> {
        let module = Module::new(engine.inner(), bytes).map_err(|error| PolicyError::Load {
            message: format!("cannot compile the policy module: {error}"),
        })?;
        Ok(Self {
            engine: engine.clone(),
            module: Arc::new(module),
        })
    }

    /// Loads an artifact [`Engine::precompile`] produced.
    ///
    /// `Module::deserialize` is unsafe because wasmtime trusts the artifact's
    /// code and its metadata; these bytes come from the Program Server's module
    /// store, whose paths are named by the program's backend, revision and
    /// target (§5.3) and whose content an `install` put there (§6.3), so this
    /// code trusts the store exactly as the mechanism trusts the kernels it
    /// loads from it.
    pub fn from_precompiled(engine: &Engine, bytes: &[u8]) -> Result<Self, PolicyError> {
        // Safety: the artifact was precompiled by an engine configured as this
        // one, and it comes from the Program Server's store (§5.3); an artifact
        // of another engine or another format is rejected by `deserialize`.
        let module = unsafe { Module::deserialize(engine.inner(), bytes) }.map_err(|error| {
            PolicyError::Load {
                message: format!("cannot load the precompiled policy module: {error}"),
            }
        })?;
        Ok(Self {
            engine: engine.clone(),
            module: Arc::new(module),
        })
    }

    /// Makes one instance of this module from `init`, and reads its limits.
    ///
    /// `init` carries the program's arguments and the whole state at this
    /// moment, so a policy installed in place of another knows every request,
    /// cache page, encoding and replica that already exists (§4.4); its own
    /// soft state starts empty. `ceilings` are the build config's, which the
    /// policy's `limits()` must stay within.
    ///
    /// A module is instantiated again whenever the mechanism needs a fresh
    /// policy: for the previous policy after a revert, and for the route
    /// instance of the control thread.
    pub fn instantiate(
        &self,
        state: &Snapshot,
        parameters: &Parameters,
        ceilings: &Limits,
    ) -> Result<PolicyInstance, PolicyError> {
        PolicyInstance::new(&self.engine, &self.module, state, parameters, ceilings)
    }
}
