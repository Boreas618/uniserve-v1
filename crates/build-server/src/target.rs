//! What a module is compiled for, and the hardware facts of each GPU target.
//!
//! A module's directory in the store has one more name after the revision, the
//! target (design section 5.3): the GPU architecture a kernel's cubin or native
//! library is compiled for, such as `sm_100a`; `wasm32-unknown-unknown` for a
//! scheduling policy's module; and the Rust target triple of this host for the
//! engine's executable.
//!
//! The GPU targets and their hardware facts are described once, in the Build
//! Server's Python package (`python/src/uniserve_build_server/targets.toml`),
//! which this crate compiles in: the targets a package may build for, the
//! target a device of a compute capability is built for, and the facts every
//! compilation is handed in its request (warp size, architecture families,
//! shared and tensor memory, dependent launch, FP8), so a package states what
//! it requires of a target instead of restating hardware facts.
//!
//! Which targets this host builds for is a property of its devices: the
//! `serve` command is given them as `--target`s or by its caller, which reads
//! their compute capabilities from the driver ([`Target::arch`]). The Build
//! Server itself links no device driver, launches no kernel and allocates no
//! device memory (design section 6.1).

use std::collections::BTreeMap;
use std::fmt;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// The description of the GPU targets, as the Build Server's Python package
/// states it for its compilations too.
const DESCRIPTION: &str = include_str!("../python/src/uniserve_build_server/targets.toml");

/// The target a scheduling policy's module is built for.
pub const WASM: &str = "wasm32-unknown-unknown";

/// The hardware facts of one GPU target's architecture, as `targets.toml`
/// states them; every compilation for the target is handed them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Facts {
    /// The code generator's backend, `cuda`.
    pub backend: String,
    /// The compute capability as one number, 100 for 10.0.
    pub capability: u32,
    /// Whether the target is the architecture-specific instruction set of
    /// exactly its capability, which a device of that capability is built
    /// for, rather than a family target every device of the family runs.
    pub specific: bool,
    /// The targets whose code runs on this one, itself first.
    pub families: Vec<String>,
    /// The threads of a warp.
    pub warp_size: u32,
    /// The largest shared memory one block may opt into, in bytes.
    pub shared_memory: u64,
    /// Whether the architecture has tensor memory.
    pub tensor_memory: bool,
    /// Whether a kernel may launch dependent grids.
    pub dependent_launch: bool,
    /// Whether the tensor cores take FP8 operands.
    pub fp8: bool,
}

/// Every GPU target by name, parsed once from [`DESCRIPTION`].
fn described() -> &'static BTreeMap<String, Facts> {
    static DESCRIBED: OnceLock<BTreeMap<String, Facts>> = OnceLock::new();
    DESCRIBED.get_or_init(|| {
        #[expect(
            clippy::expect_used,
            reason = "the description is a file of this crate, compiled in and parsed by its tests"
        )]
        toml::from_str(DESCRIPTION).expect("targets.toml describes the GPU targets")
    })
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

    /// This host's Rust target triple, such as `aarch64-unknown-linux-gnu`:
    /// the target of a module that runs on the host's CPU as its own process,
    /// the engine's. It is the triple this crate was compiled for, which its
    /// build script records.
    pub fn host() -> Self {
        Self(env!("BUILD_SERVER_HOST_TARGET").to_owned())
    }

    /// The target a scheduling policy's module is built for, [`WASM`].
    pub fn wasm() -> Self {
        Self(WASM.to_owned())
    }

    /// The target a device of compute capability `(major, minor)` is built
    /// for: the architecture-specific target of that capability, such as
    /// `sm_100a` for 10.0. The specific instruction set is the one kernels are
    /// built for: a family target omits the architecture's own instructions.
    ///
    /// # Errors
    ///
    /// Fails when no target of the description is the specific one of that
    /// capability: the Build Server builds nothing such a device runs.
    pub fn arch((major, minor): (i32, i32)) -> Result<Self> {
        let capability = u32::try_from(major * 10 + minor).ok();
        described()
            .iter()
            .find(|(_, facts)| facts.specific && Some(facts.capability) == capability)
            .map(|(name, _)| Self(name.clone()))
            .ok_or_else(|| Error::Build {
                op: String::new(),
                backend: String::new(),
                revision: 0,
                target: format!("compute capability {major}.{minor}"),
                descriptor: String::new(),
                message: format!(
                    "no GPU target is the specific instruction set of compute capability \
                     {major}.{minor}; the targets are {:?}",
                    described().keys().collect::<Vec<_>>()
                ),
            })
    }

    /// The hardware facts of this target, when it is a GPU target of the
    /// description.
    pub fn facts(&self) -> Option<&'static Facts> {
        described().get(&self.0)
    }
}

impl fmt::Display for Target {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The description parses, every family a target names is a target or a
    /// family of the description's own naming, and each capability a device
    /// reports has exactly one specific target.
    #[test]
    fn the_description_names_one_specific_target_per_capability() {
        let targets = described();
        assert!(!targets.is_empty());
        for (name, facts) in targets {
            assert_eq!(
                facts.families.first(),
                Some(name),
                "{name} runs its own code first"
            );
            let specific = targets
                .values()
                .filter(|other| other.specific && other.capability == facts.capability)
                .count();
            assert_eq!(
                specific, 1,
                "capability {} has one specific target",
                facts.capability
            );
        }
        assert_eq!(Target::arch((10, 0)).unwrap(), Target::new("sm_100a"));
        assert_eq!(Target::arch((9, 0)).unwrap(), Target::new("sm_90a"));
        assert!(Target::arch((7, 5)).is_err());
    }
}
