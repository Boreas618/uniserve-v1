//! A deployment's configuration as the admin API reads and changes it: `GET /admin/config`'s
//! status and `PUT /admin/config`'s request and result.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::update::{Unavailable, UpdateReport};
use crate::module::{ModuleSlot, PackageRevision};

/// The body of `PUT /admin/config`: a change of the deployment's configuration, as `uniserve
/// serve --config FILE --set PATH=VALUE` states one, and whether the answer waits for it. On an
/// idle node it brings the deployment of the configuration up.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ApplyRequest {
    /// Change this independent replica in place. Without an index, the change covers the
    /// deployment.
    pub replica: Option<usize>,
    /// The whole target configuration as TOML, as `--config` reads it; `None` takes the
    /// current one.
    pub config: Option<String>,
    /// `PATH=VALUE` assignments over that configuration, as `--set` states them.
    pub set: Vec<String>,
    pub in_flight: InFlight,
    /// Report what the change would do and change nothing.
    pub dry_run: bool,
    /// Apply a configuration a rollback marked unavailable.
    pub force: bool,
    /// Seconds from the apply by which the update must be committed: one that is not is
    /// rolled back by the daemon. Unset leaves the trial to the submitter's and the operator's
    /// decision.
    pub deadline_s: Option<f64>,
    /// Answer once the change applied, its update reached its trial or ended before one, or
    /// the deployment it brings up serves; otherwise at once (`202`).
    pub wait: bool,
}

impl Default for ApplyRequest {
    /// The current configuration, applied to the deployment, answered once applied.
    fn default() -> Self {
        Self {
            replica: None,
            config: None,
            set: Vec::new(),
            in_flight: InFlight::default(),
            dry_run: false,
            force: false,
            deadline_s: None,
            wait: true,
        }
    }
}

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

/// What `PUT /admin/config` did (`change`), the result of its operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "change", rename_all = "snake_case")]
pub enum ConfigChange {
    /// The target is the current configuration.
    None,
    /// The replicas took what changed in place.
    InPlace { in_place: Vec<InPlaceChange> },
    /// What the change would do; nothing changed.
    DryRun { dry_run: DryRun },
    /// An update started in its generation form: the update as it stands.
    Generation { update: UpdateReport },
    /// An update started in its epoch form: the update as it stands.
    Epoch { update: UpdateReport },
    /// The daemon's own options changed: the daemon restarted itself in place, and `pid` is
    /// its successor, which applies the rest of the change in an operation of its own when
    /// `then` states it.
    Daemon { pid: u32, then: Option<Following> },
    /// An idle node brought the deployment of the configuration up, and it serves.
    Deployment { deployment: String },
}

/// What a daemon's hot restart's successor does after it serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Following {
    /// It applies the rest of the change, an update, in an operation of its own (`config`).
    Update,
}

/// What one replica changed in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InPlaceChange {
    /// The replica's index.
    pub replica: usize,
    pub switched: Switched,
}

/// What a replica switches in place, at its engine's next step boundary or from the daemon's
/// next request on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Switched {
    /// The fence probes' switch (`profiling_config.fence_probes`, `fence_probes_off`).
    FenceProbes,
    /// The scheduling limits the policy package already runs with.
    SchedulingLimits,
}

/// What a dry run found a change would do (`change`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "change", rename_all = "snake_case")]
pub enum DryRun {
    /// No update: what each replica the change reaches would change.
    InPlace { replicas: Vec<ReplicaDryRun> },
    /// An update in its generation form, inside the serving engines of `epoch`, to the module
    /// set the apply would fix.
    Generation { epoch: u64, modules: Modules },
    /// An update in its epoch form: the epoch it would create, its engine package, its
    /// replicas, each replica's engines, and whether the serving epoch's layout differs
    /// (`shrink`).
    Epoch {
        epoch: u64,
        engine: PackageRevision,
        replicas: usize,
        engines: Vec<Vec<EngineLaunch>>,
        shrink: bool,
    },
    /// An idle node would bring the deployment `deployment` up.
    Deployment { deployment: String },
}

/// What one replica's dry run found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicaDryRun {
    /// The replica's index.
    pub replica: usize,
    /// Whether the daemon's own options change: a hot restart of the daemon.
    pub daemon: bool,
    /// Whether the scheduling limits change in place.
    pub scheduling: bool,
    /// Whether the fence probes' switch alone changes, in place.
    pub fence_probes: bool,
}

/// One engine a dry run's epoch would start: its host and the launch generation it would take.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineLaunch {
    pub host: String,
    pub launch_generation: u64,
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

/// `GET /admin/config`: the deployment's configuration and resource ownership: the first
/// replica's status at the top level, every replica's, the GPUs and quotas of the deployment's
/// nodes, the update in flight or the last one, the targets a rollback marked unavailable,
/// every epoch's record and the engine processes the daemon manages. An idle node answers it
/// with nothing in it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfigStatus {
    #[serde(flatten)]
    pub first: Option<ReplicaStatus>,
    pub replicas: Vec<ReplicaStatus>,
    /// The GPUs the deployment's epochs hold storage on; `None` while the quotas are not read.
    pub gpus: Option<Vec<GpuShare>>,
    pub quotas: Option<Quotas>,
    pub update: Option<UpdateReport>,
    pub unavailable: Vec<Unavailable>,
    pub epochs: Vec<EpochStatus>,
    pub processes: Vec<ManagedProcess>,
}

/// One replica's configuration and what it executes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicaStatus {
    /// Its configuration's non-default options, as reloadable TOML.
    pub config: String,
    /// The digest that names the configuration.
    pub digest: String,
    pub executing: Executing,
}

/// What one replica executes, as its status states it: the epoch, and the engine's state there
/// or why the epoch is unavailable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Executing {
    pub epoch: u64,
    pub state: ExecutingState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Where a replica's engine is, as its status names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutingState {
    Starting,
    Parked,
    Serving,
    Draining,
    Unavailable,
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

/// A GPU the deployment's epochs hold storage on (`GET /admin/config`'s `gpus`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuShare {
    #[serde(flatten)]
    pub gpu: Gpu,
    /// Its storage in bytes.
    pub total: u64,
    /// The deployment's quota there in bytes, when it states one.
    pub quota: Option<u64>,
    /// The bytes the quota reserves now.
    pub used: u64,
}

/// The nodes' storage as `GET /admin/config` states it: each node's usage, or why it was not
/// read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Quotas {
    Read(Vec<NodeUsage>),
    Unread { error: String },
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

/// One epoch's record as `GET /admin/config` reports it (`epochs`): its number, the build
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

/// One engine process a node's daemon manages (`GET /admin/config`'s `processes`).
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

    /// The API's spelling of the two continuation modes is the wire the operator sends; `keep`
    /// is the default when the field is absent.
    #[test]
    fn in_flight_is_snake_case_and_defaults_to_keep() {
        assert_eq!(serde_json::to_value(InFlight::Keep).unwrap(), "keep");
        assert_eq!(
            serde_json::from_str::<InFlight>("\"recompute\"").unwrap(),
            InFlight::Recompute
        );
        let request: ApplyRequest = serde_json::from_str("{}").unwrap();
        assert_eq!(request, ApplyRequest::default());
        assert_eq!((request.in_flight, request.wait), (InFlight::Keep, true));
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
