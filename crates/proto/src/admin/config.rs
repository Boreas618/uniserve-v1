//! What a deployment executes and owns, as its observation rows state it: the module sets its
//! epochs run, the GPUs and storage of its nodes, its epochs' records and the processes its
//! daemons manage (`process` events).

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::module::{ModuleSlot, PackageRevision};

/// How the requests in flight at an epoch's switch continue when the new
/// epoch computes other numerics than the one they ran on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InFlight {
    /// Each continues with the state it reached, which the new epoch reads
    /// as its own; the pages it writes afterwards reach no shared tier.
    #[default]
    Keep,
    /// Each computes its state again from its tokens, exactly.
    Recompute,
}

/// A module set: the engine package, the kernel package of each slot and the scheduling
/// policy package.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Modules {
    pub engine: PackageRevision,
    pub kernels: Vec<KernelPackage>,
    pub policy: PackageRevision,
}

/// The kernel package that fills a slot: an operator, or a phase of one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelPackage {
    #[serde(flatten)]
    pub slot: ModuleSlot,
    pub package: PackageRevision,
}

/// A GPU of a deployment's hosts: the host it is on, as rank layouts name it, and its UUID. A
/// manager of several deployments reads which GPUs they share by it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Gpu {
    pub host: String,
    pub gpu: GpuUuid,
}

impl std::fmt::Display for Gpu {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} of {}", self.gpu, self.host)
    }
}

/// A GPU's UUID, as the device reports it. It reads and writes as `GPU-` and its 16 bytes in
/// lowercase hexadecimal groups of 8, 4, 4, 4 and 12 digits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuUuid(pub [u8; 16]);

impl std::fmt::Display for GpuUuid {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&storage::protocol::gpu_name(&self.0))
    }
}

impl std::str::FromStr for GpuUuid {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        let malformed = || format!("`{text}` is not a GPU's UUID (`GPU-` and 32 hex digits)");
        let digits: Vec<u8> = text
            .strip_prefix("GPU-")
            .ok_or_else(malformed)?
            .bytes()
            .filter(|digit| *digit != b'-')
            .collect();
        if digits.len() != 32 {
            return Err(malformed());
        }
        let mut uuid = [0_u8; 16];
        let (pairs, _) = digits.as_chunks::<2>();
        for (byte, pair) in uuid.iter_mut().zip(pairs) {
            let pair = std::str::from_utf8(pair).map_err(|_| malformed())?;
            *byte = u8::from_str_radix(pair, 16).map_err(|_| malformed())?;
        }
        Ok(Self(uuid))
    }
}

impl Serialize for GpuUuid {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for GpuUuid {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

/// What one node's Storage Server keeps for the deployment (`quotas`): the node's host
/// identity, its quota of every GPU the deployment uses, what the host admits of new host
/// storage, and what the deployment's tasks copied there by priority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeUsage {
    pub host: String,
    pub quotas: Vec<storage::protocol::Quota>,
    pub host_storage: storage::protocol::HostStorage,
    pub tasks: Vec<storage::protocol::TaskTotal>,
}

/// One epoch's record (`uniserve_epoch_state` names its state): its number, the build
/// configuration, the modules by slot, the modules a generation's trial keeps, the generation
/// its engines numbered, and where the epoch is in its life.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochStatus {
    pub epoch: u64,
    pub engine_config: crate::config::EngineConfig,
    pub engine: PackageRevision,
    pub kernels: Vec<KernelPackage>,
    pub policy: PackageRevision,
    /// The kernel packages a generation's trial keeps for a rollback.
    pub previous_kernels: Vec<KernelPackage>,
    /// The scheduling policy package a generation's trial keeps for a rollback.
    pub previous_policy: Option<PackageRevision>,
    /// The generation the epoch's own engines numbered the modules they execute with.
    pub generation: Option<u64>,
    pub status: EpochState,
}

/// Where an epoch is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpochState {
    /// Its engines start beside the serving epoch's, map external state
    /// read-only and park.
    Preparing,
    /// Its engines serve requests.
    Serving,
    /// Its engines serve while what they replaced is kept for a rollback:
    /// the previous epoch, or the previous generation of this one.
    Trial,
    /// Its engines exited and its storage was retired.
    Released,
}

/// One engine process a node's daemon manages, as its `process` events state it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManagedProcess {
    /// The worker group it runs in (`engine`, `worker`, ...): the group a
    /// launch generation is allocated for.
    pub group: String,
    /// Its rank inside that group.
    pub rank: u64,
    /// Its process id on its node, once started.
    pub pid: Option<u32>,
    /// The epoch it serves, once its own report named it.
    pub epoch: Option<u64>,
    /// The build identity it was started from.
    pub build: String,
    /// The launch generation: the identity of this launch of the process,
    /// which the deployment's engine control allocates and the process names
    /// in its launch file and its reports. It is not the model generation
    /// ([`EpochStatus::generation`]), which the engine numbers for the modules
    /// it executes; 0 until launched.
    pub generation: u64,
    /// Where it is (`state`, and `reason` for a failed one).
    #[serde(flatten)]
    pub state: ProcessState,
}

/// The state of a process a node's daemon manages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "reason", rename_all = "snake_case")]
pub enum ProcessState {
    /// The daemon started it and it has not reported for service yet.
    Starting,
    /// It serves.
    Serving,
    /// It is parked: it executes nothing and keeps its storage until it is
    /// promoted or ends.
    Parked,
    /// It ended.
    Stopped,
    /// It failed, and why.
    Failed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The configuration's spelling of the two continuation modes; `keep` is the default.
    #[test]
    fn in_flight_is_snake_case_and_defaults_to_keep() {
        assert_eq!(InFlight::default(), InFlight::Keep);
        assert_eq!(serde_json::to_value(InFlight::Keep).unwrap(), "keep");
        assert_eq!(
            serde_json::from_str::<InFlight>("\"recompute\"").unwrap(),
            InFlight::Recompute
        );
    }

    /// A GPU's UUID reads back from the form it writes, and nothing else reads as one.
    #[test]
    fn a_gpu_uuid_reads_back_from_its_written_form() {
        let uuid = GpuUuid([
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
            0xcd, 0xef,
        ]);
        let written = uuid.to_string();
        assert_eq!(written, "GPU-12345678-9abc-def0-0123-456789abcdef");
        assert_eq!(written.parse::<GpuUuid>().unwrap(), uuid);
        for malformed in ["12345678-9abc-def0-0123-456789abcdef", "GPU-1234", "GPU-zz"] {
            assert!(malformed.parse::<GpuUuid>().is_err(), "{malformed}");
        }
    }

    /// A process reads back with its state, a failed one's reason among it.
    #[test]
    fn a_process_reads_back_with_its_state() {
        for state in [
            ProcessState::Serving,
            ProcessState::Failed("it exited".to_owned()),
        ] {
            let process = ManagedProcess {
                group: "engine".to_owned(),
                rank: 0,
                pid: Some(42),
                epoch: Some(1),
                build: String::new(),
                generation: 3,
                state,
            };
            let encoded = serde_json::to_value(&process).unwrap();
            assert_eq!(
                serde_json::from_value::<ManagedProcess>(encoded).unwrap(),
                process
            );
        }
    }
}
