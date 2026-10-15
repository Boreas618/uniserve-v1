//! The components this rank holds, by capability.
//!
//! Capability runtimes (text, diffusion LM, image, video, media, experts)
//! look up the components serving a call kind or runner family here and
//! construct their runners for them.

use std::collections::BTreeSet;

use uniserve_core::ComponentConfig;
use uniserve_ipc::CallKind;

use crate::config::Device;
use crate::ffi::Handle;

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

/// The components placed on this rank.
#[derive(Default)]
pub struct Components {
    placed: Vec<Placed>,
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
        Self { placed }
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
}
