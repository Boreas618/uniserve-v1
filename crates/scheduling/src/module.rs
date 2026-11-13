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
use wasmtime::Module;

use crate::Engine;
use crate::error::PolicyError;
use crate::instance::PolicyInstance;

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
    /// Loads the bytes the Build Server serves for a policy package: the
    /// precompiled artifact when the bytes are one (§5.3, "按本机 CPU 预编译"),
    /// the wasm32 binary otherwise ([`Engine::load`]).
    pub fn load(engine: &Engine, bytes: &[u8]) -> Result<Self, PolicyError> {
        Ok(Self {
            engine: engine.clone(),
            module: Arc::new(engine.load(bytes)?),
        })
    }

    /// Makes one instance of this module from `init`, and reads its limits.
    ///
    /// `init` carries the package's arguments and the whole state at this
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
