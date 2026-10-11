//! The numerical interface table Python hands the worker at startup.
//!
//! `uniserve_models.interface.table()` maps `<area>.<verb>` names to
//! functions. [`Interface`] resolves the entries startup needs into typed
//! fields once, failing startup with the name of any missing entry, and
//! keeps every entry so capability runtimes can resolve their own
//! (`component.bind`, ...) by name.

use std::collections::BTreeMap;

use tvm_ffi::Map;

use crate::ffi::Function;
use crate::{Error, Result};

/// Startup and lifecycle entries of the interface table.
pub struct Interface {
    /// Declare a component call set's workspace requirements.
    pub prepare: Function,
    /// Bind Rust-allocated workspaces to a prepared component.
    pub bind: Function,
    /// Join the torch process world for numerical collectives.
    pub initialize: Function,
    /// Create torch groups on the Rust plan.
    pub create_groups: Function,
    /// Build a `DeviceMesh` bound to created groups.
    pub mesh: Function,
    /// Destroy the process world's groups in reverse creation order.
    pub destroy: Function,
    /// A counter that changes whenever a kernel provider choice changes.
    pub kernel_choices: Function,
    /// Start a profiler window.
    pub profiler_start: Function,
    /// Stop a profiler window and export its trace.
    pub profiler_stop: Function,
    /// Collect, freeze and disable the Python garbage collector.
    pub freeze_gc: Function,
    entries: BTreeMap<String, Function>,
}

impl Interface {
    /// Resolve the typed entries of `table`.
    pub fn from_table(table: Map<tvm_ffi::String, tvm_ffi::Function>) -> Result<Self> {
        let entries: BTreeMap<String, Function> = table
            .iter()
            .map(|(name, function)| {
                let name = name.as_str().to_owned();
                (name.clone(), Function::new(name, function))
            })
            .collect();
        let require = |name: &str| {
            entries.get(name).cloned().ok_or_else(|| {
                Error::Interface(format!(
                    "the numerical interface table has no `{name}` entry"
                ))
            })
        };
        Ok(Self {
            prepare: require("component.prepare")?,
            bind: require("component.bind")?,
            initialize: require("distributed.initialize")?,
            create_groups: require("distributed.create_groups")?,
            mesh: require("distributed.mesh")?,
            destroy: require("distributed.destroy")?,
            kernel_choices: require("kernels.choices")?,
            profiler_start: require("profiling.start")?,
            profiler_stop: require("profiling.stop")?,
            freeze_gc: require("runtime.freeze_gc")?,
            entries,
        })
    }

    /// A capability runtime's entry. Fails naming the missing entry.
    pub fn function(&self, name: &str) -> Result<Function> {
        self.optional(name).ok_or_else(|| {
            Error::Interface(format!(
                "the numerical interface table has no `{name}` entry"
            ))
        })
    }

    /// The Python callable at a dotted import path, for example
    /// `uniserve_models.runner.causal_lm.CausalLMRunner.decode`: a function,
    /// a class (called to construct it) or a method (called with the object
    /// first).
    pub fn import(&self, path: &str) -> Result<Function> {
        let imported = self
            .function("runtime.import")?
            .call(&[(&tvm_ffi::Any::from(tvm_ffi::String::from(path))).into()])?;
        let function = tvm_ffi::Function::try_from(imported)
            .map_err(|_| Error::Interface(format!("`{path}` is not callable")))?;
        Ok(Function::new(path, function))
    }

    pub fn optional(&self, name: &str) -> Option<Function> {
        self.entries.get(name).cloned()
    }

    /// Every entry name, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }
}
