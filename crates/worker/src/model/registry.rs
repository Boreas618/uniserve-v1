//! The components this rank holds and their bindings, by capability.
//!
//! Capability runtimes (text, canvas, image, video, media, experts) look up
//! the components serving a call kind or runner family here, create their
//! bindings through [`crate::model::bind`] and register them, so startup
//! closes every binding in one place and kernel reports cover all of them.

use std::collections::BTreeSet;

use uniserve_core::ComponentConfig;
use uniserve_ipc::CallKind;

use crate::Result;
use crate::config::Device;
use crate::ffi::Handle;

use super::bind::ComponentBinding;
use super::components::{CallDecl, Capability, Declarations, RunnerKind, host_kinds};
use crate::parallel::topology::is_host_component;

/// One component placed on this rank.
pub struct Placed {
    pub name: String,
    pub config: ComponentConfig,
    /// Declared calls; empty for a host component.
    pub calls: Vec<CallDecl>,
    pub kinds: BTreeSet<CallKind>,
    /// The device the component's bindings compute on: the rank's primary
    /// device. Modules a tower placement puts on a generation device run
    /// there through the bindings' transfers.
    pub device: Device,
    /// The component's numerical mesh (a `DeviceMesh` handle), if bound.
    pub mesh: Option<Handle>,
    /// The temporal unit ring of a distributed component.
    pub units: Option<Handle>,
}

impl Placed {
    pub fn host(&self) -> bool {
        is_host_component(&self.name)
    }

    /// The runner families serving this component's calls.
    pub fn runners(&self) -> BTreeSet<RunnerKind> {
        self.calls.iter().filter_map(CallDecl::runner).collect()
    }

    pub fn has(&self, capability: Capability) -> bool {
        self.calls.iter().any(|call| call.is(capability))
    }
}

/// Placed components and their bindings.
#[derive(Default)]
pub struct Components {
    placed: Vec<Placed>,
    bindings: Vec<ComponentBinding>,
}

impl Components {
    /// Record the components of `declarations` placed on this rank.
    /// `meshes` supplies each member component's mesh and unit ring.
    pub fn new(
        declarations: &Declarations,
        placements: &std::collections::BTreeMap<String, ComponentConfig>,
        rank: usize,
        primary: Device,
        mut meshes: impl FnMut(&str) -> (Option<Handle>, Option<Handle>),
    ) -> Self {
        let placed = placements
            .iter()
            .filter(|(_, config)| config.ranks.contains(&rank))
            .map(|(name, config)| {
                let kinds = declarations.component_kinds(name);
                let (mesh, units) = meshes(name);
                Placed {
                    name: name.clone(),
                    config: config.clone(),
                    calls: declarations.calls(name).to_vec(),
                    kinds,
                    device: primary,
                    mesh,
                    units,
                }
            })
            .collect();
        Self {
            placed,
            bindings: Vec::new(),
        }
    }

    pub fn placed(&self) -> &[Placed] {
        &self.placed
    }

    pub fn get(&self, name: &str) -> Option<&Placed> {
        self.placed.iter().find(|placed| placed.name == name)
    }

    /// Components serving `kind` on this rank.
    pub fn serving(&self, kind: CallKind) -> impl Iterator<Item = &Placed> {
        self.placed
            .iter()
            .filter(move |placed| placed.kinds.contains(&kind))
    }

    /// Components a runner family serves on this rank.
    pub fn for_runner(&self, runner: RunnerKind) -> impl Iterator<Item = &Placed> {
        self.placed
            .iter()
            .filter(move |placed| placed.runners().contains(&runner))
    }

    /// Whether this rank holds a host component (media reading, video
    /// encoding or muxing).
    pub fn holds_host(&self) -> bool {
        self.placed
            .iter()
            .any(|placed| !host_kinds(&placed.name).is_empty())
    }

    /// Whether every held component is a host component or has no calls,
    /// so the rank needs no numerical weights.
    pub fn host_only(&self) -> bool {
        self.placed
            .iter()
            .all(|placed| placed.host() || placed.calls.is_empty())
    }

    /// Register bindings a capability runtime created.
    pub fn add_bindings(&mut self, bindings: Vec<ComponentBinding>) {
        self.bindings.extend(bindings);
    }

    pub fn bindings(&self) -> &[ComponentBinding] {
        &self.bindings
    }

    /// Close every binding, newest first, attempting all before returning
    /// the first failure.
    pub fn close(&mut self, aborted: bool) -> Result<()> {
        let mut failure = None;
        while let Some(mut binding) = self.bindings.pop() {
            if let Err(error) = binding.close(aborted) {
                failure.get_or_insert(error);
            }
        }
        failure.map_or(Ok(()), Err)
    }
}
