//! What a module is compiled for: an architecture, or a GPU model.
//!
//! A module's directory in the store has one more name after the revision,
//! the target (design section 5.3). Almost always that is the architecture the
//! cubin or native library targets, such as `sm_100a`. A binary that compiles
//! the GPU's SM count into itself instead is compiled per GPU model: the
//! FlashAttention-4 package reads the SM count while it traces its kernel
//! (`flash_attn/cute/tile_scheduler.py:409-413`), so its modules are stored
//! under the GPU model's name.
//!
//! Which targets this host builds for is a property of its devices, so the
//! Program Server is told them: it launches no kernel and allocates no device
//! memory (design section 6.1), and the executable gives it the device list
//! through [`Devices`] — in the `uniserve program-server` command, from the
//! platform crate's device query (design section 6.2's crate dependency).

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// How a package's target is named: by architecture, or by GPU model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetKind {
    /// The instruction set, such as `sm_100a`.
    #[default]
    Arch,
    /// The GPU model, for a binary whose code depends on the device.
    Gpu,
}

/// A target a module is built and stored for, such as `sm_100a`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Target(String);

impl Target {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Target {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// One device of the host, as far as compiling for it is concerned: the
/// architecture its code is built for, and its model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    /// The architecture name, such as `sm_100a`.
    pub arch: Target,
    /// The GPU model, such as the name its vendor gives it.
    pub model: Target,
}

/// The host's devices, which the Program Server builds for.
///
/// The Program Server does not call CUDA or NVML itself (design section 4.1:
/// only the platform crate does); the executable hands it the device list
/// with an implementation of this trait.
pub trait Devices: Send + Sync {
    fn devices(&self) -> Result<Vec<Device>>;
}

/// The targets of a set of devices for a package's kind of target, without
/// repeats and in a stable order.
pub fn targets(devices: &[Device], kind: TargetKind) -> Vec<Target> {
    let mut targets: Vec<Target> = devices
        .iter()
        .map(|device| match kind {
            TargetKind::Arch => device.arch.clone(),
            TargetKind::Gpu => device.model.clone(),
        })
        .collect();
    targets.sort();
    targets.dedup();
    targets
}

/// Fails with the targets a request named that this host does not have.
pub fn check(devices: &[Device], kind: TargetKind, requested: &[Target]) -> Result<()> {
    let available = targets(devices, kind);
    for target in requested {
        if !available.contains(target) {
            return Err(Error::Build {
                op: String::new(),
                backend: String::new(),
                revision: 0,
                target: target.to_string(),
                descriptor: String::new(),
                message: format!(
                    "this host builds for {available:?} and not for {target}",
                    available = available
                        .iter()
                        .map(|target| target.as_str())
                        .collect::<Vec<_>>()
                ),
            });
        }
    }
    Ok(())
}
