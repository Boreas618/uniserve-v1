//! Epochs: what a deployment executes, what fills it, and how the daemon
//! moves it.
//!
//! An epoch is one configuration executed: its main processes on every node,
//! their communicators, graphs and the storage they create
//! (`specs/worker.md` §0). The leader's daemon keeps one [`EpochRecord`] per
//! epoch: the [`BuildConfig`] it was built from, the programs that fill its
//! operators and its scheduling interface, the programs a probation keeps as
//! the previous generation, and its [`Status`].
//!
//! A configuration change is routed mechanically by comparing the records
//! (§7.2): with the build config unchanged and only the filled programs
//! different, the programs move inside the serving main process (§7.3); with
//! a different build config, a new epoch is built (§7.4); a change of the
//! daemon's or the frontend's own options, or of their executables, is a
//! hot restart (§7.5). Nothing about a change is declared by the change
//! itself.
//!
//! The values are the daemon's own, not the wire's: [`Program`] is
//! `crates/program-server`'s type, which the programs swarm defines, and the
//! daemon records the wire form ([`crate::proto`]'s `process::Program`)
//! until that crate lands.

use std::collections::BTreeMap;
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::PathBuf;
use std::sync::Mutex;

pub use program_server::{PhaseKind, Program, ProgramSlot};
use proto::request::ModelDtype;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::options::{
    CudaGraphOptions, KvCacheOptions, LoadOptions, ServeConfig, VideoOptions, WorkerOptions,
};

/// The operator names the epoch record keys its kernel programs by (§5.1,
/// §5.2), of the operators whose backend a configuration selects. They are
/// `crates/ops`'s `Op::NAME` of the same operator, so the daemon, the
/// Program Server and the main process cannot disagree about a slot's name.
///
/// The complete slot set is `crates/ops`'s operator names. The rest are
/// filled by the epoch's programs like any other slot and need no entry
/// here: their backend is not a configuration field, so this module names
/// only the operators the configuration selects a backend for. The
/// scheduling interface is not an operator at all: its program is the epoch
/// record's `policy` field.
pub mod operators {
    use ops::Op as _;

    /// Paged attention, the operator `attention_backend` and the per-phase
    /// attention fields select a backend for (§4.2, §9).
    pub const PAGED_ATTENTION: &str = ops::attention::PagedAttention::NAME;
    /// The grouped expert GEMM's gather and compute, the operator
    /// `moe_runner_backend` selects a backend for (§5.1, §9). The grouped
    /// experts are two operators, and one program backend fills both.
    pub const GATHER_GEMM: &str = ops::moe::GatherGemm::NAME;
    /// The grouped expert GEMM's route and combine, filled by the same
    /// program backend as [`GATHER_GEMM`].
    pub const ROUTE_GEMM: &str = ops::moe::RouteGemm::NAME;
    /// The expert exchange around the grouped expert GEMM, the operator
    /// `moe_config.exchange` selects a backend for (§9).
    pub const EXPERT_EXCHANGE: &str = ops::moe::ExpertExchange::NAME;
}

/// The key an epoch's programs are stored under: the operator, and the
/// phase when the configuration fills it per phase (§5.2). Programs and
/// slots are `crates/program-server`'s types, which the daemon, the main
/// process and the Program Server all name programs with (§5.1).
pub type KernelKey = ProgramSlot;

/// One program as the process interface's wire form: its parameters as the
/// TOML text of each value, so a main process of another build reads each
/// back as the value it was (§5.1).
pub fn program_to_wire(program: &Program) -> proto::process::Program {
    proto::process::Program {
        backend: program.backend.clone(),
        revision: program.revision,
        arguments: program
            .arguments
            .iter()
            .map(|(key, value)| (key.clone(), value.to_string()))
            .collect(),
    }
}

/// The program `wire` states, its parameters read back as TOML values.
pub fn program_from_wire(wire: proto::process::Program) -> Program {
    Program {
        backend: wire.backend,
        revision: wire.revision,
        arguments: wire
            .arguments
            .into_iter()
            .map(|(key, value)| {
                let value = value
                    .parse::<toml::Value>()
                    .unwrap_or(toml::Value::String(value));
                (key, value)
            })
            .collect(),
    }
}

/// One operator's program as the `Precopy` command states it (§7.3 step 2).
pub fn kernel_to_wire(slot: &KernelKey, program: &Program) -> proto::process::KernelProgram {
    proto::process::KernelProgram {
        op: slot.op.clone(),
        phase: slot.phase.map(|phase| {
            let phase = match phase {
                PhaseKind::Prefill => proto::process::PhaseKind::Prefill,
                PhaseKind::Decode => proto::process::PhaseKind::Decode,
                PhaseKind::Encode => proto::process::PhaseKind::Encode,
                PhaseKind::Denoise => proto::process::PhaseKind::Denoise,
            };
            phase as i32
        }),
        program: Some(program_to_wire(program)),
    }
}

/// The values that build a main process: every one of them is a value a new
/// epoch must be built for, and no other change is (§7.1, §7.7).
///
/// The projection keeps exactly what §7.7 assigns to the build config: what
/// fixes the executable, the parallelism and block assignment, the numerical
/// representation, the KV layout, the captured graph buckets, the request
/// slots and the loading options. Values that only inform a decision are
/// parameters of the programs that decide, and values that name an
/// implementation select a program's backend; neither appears here, so two
/// configurations that differ only in those compare equal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildConfig {
    /// The build of the main process executable this epoch runs (§7.1): a
    /// replaced executable is a new epoch even at the same configuration.
    pub executable: String,
    /// Longest sequence a request may reach, in tokens.
    pub max_model_len: Option<NonZeroU32>,
    /// Concurrently resident requests.
    pub max_running_requests: NonZeroUsize,
    /// Each rank's share of its device's storage.
    pub mem_fraction_static: f64,
    /// Element type of parameters and activations.
    pub dtype: ModelDtype,
    /// The checkpoint's quantization (§1). The configuration field is
    /// `quantization_config` until §7.7 replaces it with `quantization`.
    pub quantization: Map<String, Value>,
    /// Tensor-parallel workers of each scheduler.
    pub tensor_parallel_size: NonZeroUsize,
    /// Schedulers of the epoch.
    pub data_parallel_size: NonZeroUsize,
    /// The deployment file that places worker groups and their blocks; its
    /// resolution is part of the epoch's parallelism and block assignment.
    pub worker_groups: Option<PathBuf>,
    /// The host whose main processes serve requests.
    pub engine_host: Option<String>,
    /// How the checkpoint is read and mapped.
    pub load: LoadOptions,
    /// The KV cache's layout and capacity.
    pub kv_cache: KvCacheOptions,
    /// Whether routed experts are sharded across ranks.
    pub expert_parallel: bool,
    /// Microbatches overlapping attention with dedicated expert workers.
    pub moe_microbatches: NonZeroU32,
    /// The captured graph buckets.
    pub cuda_graph: CudaGraphOptions,
    /// The worker configuration each rank runs under.
    pub worker: WorkerOptions,
    /// The draft tokens a verification carries; zero does not speculate.
    pub num_draft_tokens: u32,
    /// The video configuration the epoch's media work takes.
    pub video: VideoOptions,
    /// The components whose module instances carry fencing probes, which a
    /// graph captures.
    pub fence_components: Vec<String>,
    /// Whether the workers sample PCs of graph kernels.
    pub graph_pc_sampling: bool,
}

impl BuildConfig {
    /// The build config `config` builds a main process from, as this
    /// daemon's own executable constitutes it.
    ///
    /// The executable's identity is this crate's [`proto::BUILD`]: today the
    /// daemon and the main process are one binary, so a deployment cannot
    /// run them at different builds. A daemon that starts a main process of
    /// another build compares that executable's identity instead (§7.4).
    pub fn of(config: &ServeConfig) -> Self {
        Self {
            executable: proto::BUILD.to_owned(),
            max_model_len: config.max_model_len,
            max_running_requests: config.max_running_requests,
            mem_fraction_static: config.mem_fraction_static,
            dtype: config.dtype,
            quantization: config.quantization_config.clone(),
            tensor_parallel_size: config.tensor_parallel_size,
            data_parallel_size: config.data_parallel_size,
            worker_groups: config.worker_groups.clone(),
            engine_host: config.engine_host.clone(),
            load: config.load_config.clone(),
            kv_cache: config.kv_cache_config.clone(),
            expert_parallel: config.moe_config.expert_parallel,
            moe_microbatches: config.moe_config.microbatches,
            cuda_graph: config.cuda_graph_config.clone(),
            worker: config.worker_config.clone(),
            num_draft_tokens: config.speculative_config.num_draft_tokens,
            video: config.video_config.clone(),
            fence_components: config.profiling_config.fence_components.clone(),
            graph_pc_sampling: config.profiling_config.graph_pc_sampling,
        }
    }
}

/// Where an epoch is in its life (§7.3, §7.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Its main processes prepare the next generation of programs while the
    /// serving epoch runs (the program layer's precopy), or they start as
    /// standbys that map storage read-only and park (the epoch layer's
    /// precopy).
    Precopy,
    /// Its main processes serve requests.
    Serving,
    /// Its main processes serve while the previous generation is kept for a
    /// rollback.
    Probation,
    /// Its main processes exited and its storage was retired.
    Purged,
}

/// What the leader's daemon records of one epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochRecord {
    /// What built the epoch.
    pub build_config: BuildConfig,
    /// The kernel program that fills each operator, or each phase of one;
    /// a phase of `None` covers every phase of its operator (§5.2).
    pub kernels: BTreeMap<KernelKey, Program>,
    /// The scheduling program.
    pub policy: Program,
    /// The programs a probation keeps as the previous generation: a revert
    /// and a restart during the probation use them, and a commit empties
    /// them (§7.3).
    pub previous_kernels: BTreeMap<KernelKey, Program>,
    /// The previous scheduling program, kept until the next commit.
    pub previous_policy: Option<Program>,
    /// Where the epoch is in its life.
    pub status: Status,
    /// The MODEL generation: the number the epoch's own main processes gave
    /// the programs they execute, read from their `Parked`, `Promoted` or
    /// launch report and never invented here (§2.5, §7.4). It is not a
    /// process's identity: that is [`ManagedProcess::generation`], the
    /// launch generation, allocated by [`Epochs::next_generation`].
    pub generation: Option<u64>,
    /// The deployment's world allocation this epoch executes (§7.4), which
    /// the epoch's launch file and its Storage Server claim carry.
    pub world: u64,
}

/// The role of a process a node's daemon manages (§2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The deployment's application process on the leader.
    Frontend,
    /// The node's main process for one epoch.
    MainProcess,
}

/// The state of a process a node's daemon manages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessState {
    /// The daemon started it and it has not reported for service yet.
    Starting,
    /// It serves.
    Serving,
    /// It is parked: it executes nothing and keeps its storage until it is
    /// promoted or ends (§2.5).
    Parked,
    /// It ended.
    Stopped,
    /// It failed, and why.
    Failed(String),
}

/// One process a node's daemon manages, as `GET /admin/config` reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedProcess {
    /// What it is.
    pub role: Role,
    /// The worker group it runs in (`engine`, `worker`, ...): the group a
    /// launch generation is allocated for (§2.5).
    pub group: String,
    /// Its rank inside that group.
    pub rank: u64,
    /// Its process id on its node, once started.
    pub pid: Option<u32>,
    /// The epoch a main process belongs to; `None` for the frontend.
    pub epoch: Option<u64>,
    /// The build identity it was started from.
    pub build: String,
    /// The LAUNCH generation: the identity of this launch of the process,
    /// which the daemon allocates ([`Epochs::next_generation`]) and the
    /// process names in its launch file and its reports. It is not the
    /// model generation ([`EpochRecord::generation`]), which the main
    /// process numbers for the programs it executes; 0 until launched.
    pub generation: u64,
    /// Where it is.
    pub state: ProcessState,
}

/// One configuration change in flight or just applied: one operation with
/// one number, one probation and one commit or revert (§7.2).
///
/// The first failure reported while the update runs is kept, not the last
/// (§7.3 step 4): if the update reverts, it is marked unavailable with that
/// first error, and applying the same configuration again takes `force`.
#[derive(Debug, Clone, PartialEq)]
pub struct Update {
    /// The update's number, from the daemon's sequence of operations.
    pub number: u64,
    /// What the update moves the deployment to.
    pub target: Deployment,
    /// Where the update is.
    pub state: UpdateState,
    /// The first failure reported while the update ran.
    pub first_failure: Option<Failure>,
}

/// Where an update is (§7.2, §7.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateState {
    /// The new generation serves while the previous one is kept: the
    /// probation, from which a commit or a revert follows (§7.3 step 4).
    Probation,
    /// The probation ended in favour of the new generation.
    Committed,
    /// The probation ended by returning to the previous generation.
    Reverted,
}

/// One failure that failed an update: a fault the main process observed, a
/// stalled job, non-finite logits, or a scheduling policy that had to be
/// reverted (§7.3 step 4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    /// A stable short name: `sticky`, `illegal_address`, `launch_failed`,
    /// `out_of_memory`, `process_exit`, `stalled`, `non_finite_logits` or
    /// `policy_reverted`.
    pub kind: String,
    /// The account the main process gave, for the daemon's record.
    pub reason: String,
    /// The CUDA ordinal the failure names, when it names one.
    pub device: Option<u32>,
}

impl Failure {
    /// The failure a main process's report states, or `None` for a report
    /// that states none: an acknowledgement (`Precopied`, `Switched`,
    /// `Committed`, `Reverted`, `Purged`) or a failed precopy, which drops
    /// the new generation and leaves the serving one alone (§7.3 step 2).
    pub fn of_report(report: &proto::process::Report) -> Option<Self> {
        use proto::process::report::Kind;
        let kind = report.kind.as_ref()?;
        match kind {
            Kind::Fault(fault) => Some(Self {
                kind: if fault.kind.is_empty() {
                    "fault".to_owned()
                } else {
                    fault.kind.clone()
                },
                reason: fault.reason.clone(),
                device: Some(fault.device),
            }),
            Kind::Stalled(stalled) => Some(Self {
                kind: "stalled".to_owned(),
                reason: format!(
                    "the {} job of batch {} ran {} ms",
                    stalled.job, stalled.batch, stalled.elapsed_ms
                ),
                device: None,
            }),
            Kind::NonFiniteLogits(logits) => Some(Self {
                kind: "non_finite_logits".to_owned(),
                reason: format!("non-finite logits for request {}", logits.request),
                device: None,
            }),
            Kind::PolicyReverted(reverted) => Some(Self {
                kind: "policy_reverted".to_owned(),
                reason: {
                    // Presence is the contract (§4.4): absent is the route
                    // instance the engine's control thread runs, a value one
                    // scheduler thread's policy.
                    let instance = match reverted.scheduler {
                        Some(scheduler) => format!("scheduler {scheduler}"),
                        None => "the engine's control thread".to_owned(),
                    };
                    let fatal = if reverted.fatal {
                        ", and no previous policy module is left"
                    } else {
                        ""
                    };
                    format!(
                        "the policy of {instance} (backend {} revision {}) failed: {}{fatal}",
                        reverted.backend, reverted.revision, reverted.reason
                    )
                },
                device: None,
            }),
            Kind::Parked(_)
            | Kind::Promoted(_)
            | Kind::Precopied(_)
            | Kind::Switched(_)
            | Kind::Committed(_)
            | Kind::Reverted(_)
            | Kind::Purged(_)
            | Kind::Drained(_)
            | Kind::PrecopyFailed(_) => None,
        }
    }
}

impl Update {
    /// The update `number` that moves a deployment to `target`; its
    /// probation has not ended.
    pub fn new(number: u64, target: Deployment) -> Self {
        Self {
            number,
            target,
            state: UpdateState::Probation,
            first_failure: None,
        }
    }

    /// Records `failure` unless a failure was recorded already: the first
    /// error an update meets is the one it is marked unavailable with.
    pub fn fail(&mut self, failure: Failure) {
        if self.first_failure.is_none() {
            self.first_failure = Some(failure);
        }
    }

    /// Ends the probation in favour of the new generation (§7.3 step 5).
    pub fn commit(&mut self) {
        self.state = UpdateState::Committed;
    }

    /// Ends the probation by returning to the previous deployment (§7.3
    /// step 5): the update stays unavailable with its first failure, and
    /// applying it again takes `force`.
    ///
    /// A deliberate rollback is not a failure: it states none, so the update
    /// it ends stays available and applying it again needs no `force`.
    /// Only a revert after a failure is marked with one.
    pub fn revert(&mut self) {
        self.state = UpdateState::Reverted;
    }

    /// Why this update may not be applied again without `force`, when it
    /// reverted after a failure (§7.3 step 4).
    pub fn unavailable(&self) -> Option<&Failure> {
        match self.state {
            UpdateState::Reverted => self.first_failure.as_ref(),
            UpdateState::Probation | UpdateState::Committed => None,
        }
    }
}

/// One phase of an update, as the daemon's event log names it (§7.2, §7.3):
/// the program layer's steps and the probation's end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// The new generation's programs are being prepared in the serving main
    /// processes (§7.3 step 2).
    Precopy,
    /// Every main process prepared them.
    Precopied,
    /// The switchover to the new generation is under way (§7.3 step 3).
    Switchover,
    /// Every main process executes the new generation.
    Switched,
    /// The new generation serves while the previous one is kept (§7.3
    /// step 4).
    Probation,
    /// The probation ended in favour of the new generation (§7.3 step 5).
    Committed,
    /// The probation ended by returning to the previous generation, with
    /// the reason it did.
    Reverted,
    /// The generation the probation kept is gone (§7.3 step 6).
    Purged,
    /// A main process could not prepare the new generation's programs; the
    /// serving one is untouched (§7.3 step 2).
    PrecopyFailed,
    /// A scheduling policy failed and the main process reinstantiated the
    /// previous one (§4.4).
    PolicyReverted,
}

impl Phase {
    /// The name the event log and the update's readers use.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Precopy => "precopy",
            Self::Precopied => "precopied",
            Self::Switchover => "switchover",
            Self::Switched => "switched",
            Self::Probation => "probation",
            Self::Committed => "committed",
            Self::Reverted => "reverted",
            Self::Purged => "purged",
            Self::PrecopyFailed => "precopy_failed",
            Self::PolicyReverted => "policy_reverted",
        }
    }
}

/// A program-layer update in flight, as the daemon keeps it between the
/// request that started it and the one that ends its probation (§7.3 steps
/// 2-6): the update, the epoch it moves, and every main process with the
/// generation it prepared.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramUpdate {
    /// The update, in its probation.
    pub update: Update,
    /// The epoch whose programs it moves.
    pub epoch: u64,
    /// The main processes that prepared a generation: the node they run on,
    /// the engine number their reports carry, and the generation each of
    /// them numbered.
    pub prepared: Vec<PreparedProcess>,
}

/// One main process that prepared a generation of a program update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedProcess {
    pub host: String,
    pub engine: u64,
    pub generation: u64,
}

/// The epochs a deployment's daemon keeps and the processes it manages on
/// its node: the state `GET /admin/config` reports and a change is routed
/// by (§7.2).
#[derive(Debug, Default)]
pub struct Epochs {
    records: Mutex<BTreeMap<u64, EpochRecord>>,
    processes: Mutex<Vec<ManagedProcess>>,
    /// The daemon's process table, once the daemon that supervises this
    /// node's processes attached it: the processes are the supervisor's to
    /// report, and this record keeps what each of them is (§2.3).
    supervised: std::sync::OnceLock<std::sync::Arc<Mutex<Vec<ManagedProcess>>>>,
    update: Mutex<Option<Update>>,
    /// The last launch generation allocated per worker group and rank.
    generations: Mutex<BTreeMap<(String, u64), u64>>,
    /// The daemon's command channel to its main processes (`supervision::
    /// ProcessCommands`), which the program layer commands through.
    commands: Mutex<Option<crate::supervision::ProcessCommands>>,
    /// The program-layer update in flight: its probation's record (§7.3).
    program: Mutex<Option<ProgramUpdate>>,
}

impl Epochs {
    /// No epoch was built and no process started yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records `record` as epoch `epoch`'s, replacing what was recorded of
    /// that epoch before.
    pub fn record(&self, epoch: u64, record: EpochRecord) {
        locked(&self.records).insert(epoch, record);
    }

    /// The launch generation to start `group`'s rank `rank` in (§2.5,
    /// §7.4): one more than the highest launch generation this daemon knows,
    /// which is the one it allocated last for that group and rank or the one
    /// the process table records — for any group and rank. A standby of a
    /// serving process is `serving + 1`, and a rank or group without a process
    /// yet still allocates above every number the deployment has used, so one
    /// number names one launch across the deployment.
    ///
    /// This is the one place a generation is allocated: a caller that
    /// computed one itself would launch two processes under one number.
    pub fn next_generation(&self, group: &str, rank: u64) -> u64 {
        let runnning = self
            .processes()
            .iter()
            .map(|process| process.generation)
            .max()
            .unwrap_or(0);
        let mut allocated = locked(&self.generations);
        let key = (group.to_owned(), rank);
        let last = allocated.get(&key).copied().unwrap_or(0).max(runnning);
        let next = last + 1;
        allocated.insert(key, next);
        next
    }

    /// Records `update` as the deployment's update: the one in flight, or
    /// the last one that ended.
    pub fn set_update(&self, update: Update) {
        *locked(&self.update) = Some(update);
    }

    /// The deployment's update: the one in flight, or the last one that
    /// ended.
    pub fn update(&self) -> Option<Update> {
        locked(&self.update).clone()
    }

    /// Why `target` may not be applied without `force` (§7.2, §7.3 step 4):
    /// the last update reverted from the same programs after a failure.
    ///
    /// The comparison is the programs', revisions included, because those
    /// are what a reverted program-layer update was: applying the same
    /// configuration again selects the same revisions, so the refusal is
    /// about the change and not about how it was asked for.
    pub fn unavailable(&self, target: &Deployment) -> Option<Failure> {
        let update = locked(&self.update);
        let update = update.as_ref()?;
        let failure = update.unavailable()?;
        (update.target.programs == target.programs).then(|| failure.clone())
    }

    /// Records that epoch `epoch` now executes `programs`, keeping what it
    /// executed before as the probation's previous generation (§7.3 steps
    /// 2-4). The record's status stays what it was: the programs are in the
    /// main process, the probation is the update's.
    pub fn precopied(&self, epoch: u64, programs: FilledPrograms) {
        let mut records = locked(&self.records);
        let Some(record) = records.get_mut(&epoch) else {
            return;
        };
        record.previous_kernels = std::mem::replace(&mut record.kernels, programs.kernels);
        record.previous_policy = Some(std::mem::replace(&mut record.policy, programs.policy));
        // The new programs serve while the previous generation is kept as
        // the last resort (§7.3 step 4).
        if record.status == Status::Serving {
            record.status = Status::Probation;
        }
    }

    /// Records that epoch `epoch`'s new programs are what serves and the
    /// previous generation is no longer kept (§7.3 step 5).
    pub fn committed(&self, epoch: u64) {
        let mut records = locked(&self.records);
        let Some(record) = records.get_mut(&epoch) else {
            return;
        };
        record.previous_kernels.clear();
        record.previous_policy = None;
        record.status = Status::Serving;
    }

    /// Records that everything older than the generation epoch `epoch`
    /// executes is gone (§7.3 step 6).
    pub fn purged(&self, epoch: u64) {
        let mut records = locked(&self.records);
        if let Some(record) = records.get_mut(&epoch) {
            record.status = Status::Purged;
        }
    }

    /// Ends a probation by putting the previous generation back: what the
    /// epoch executed before the update is what it executes again (§7.3
    /// step 5), and the generation the update prepared is dropped with it.
    pub fn reverted(&self, epoch: u64) {
        let mut records = locked(&self.records);
        let Some(record) = records.get_mut(&epoch) else {
            return;
        };
        if !record.previous_kernels.is_empty() || record.previous_policy.is_some() {
            record.kernels = std::mem::take(&mut record.previous_kernels);
            record.policy = record
                .previous_policy
                .take()
                .unwrap_or_else(|| record.policy.clone());
        }
        record.status = Status::Serving;
    }

    /// Records epoch `epoch` as serving the `deployment` this node installed
    /// at startup: its build config and programs, the generation its main
    /// processes numbered, and the world it executes (§7.4). The daemon
    /// records nothing it did not install, so this is called with what the
    /// main processes answered, not with what the configuration asked for.
    ///
    /// Answers the epoch's number.
    pub fn install(
        &self,
        epoch: u64,
        deployment: &Deployment,
        world: u64,
        generations: &[u64],
    ) -> u64 {
        self.record(
            epoch,
            EpochRecord {
                build_config: deployment.build_config.clone(),
                kernels: deployment.programs.kernels.clone(),
                policy: deployment.programs.policy.clone(),
                previous_kernels: BTreeMap::new(),
                previous_policy: None,
                status: Status::Serving,
                generation: generations.first().copied(),
                world,
            },
        );
        epoch
    }

    /// Gives the epochs the daemon's command channel to its main processes,
    /// once, when the daemon starts (§2.3): the program layer commands
    /// through it and reads its reports.
    pub fn supervise(&self, commands: crate::supervision::ProcessCommands) {
        *locked(&self.commands) = Some(commands);
    }

    /// Hands the serving epoch's programs to a main process that joined it
    /// after the epoch was installed — the process a relaunch starts — and
    /// switches it to them (§2.5, §7.5).
    ///
    /// A main process starts with no programs: it resolves no kernel itself,
    /// and its first request fails every operator it was never given. The
    /// first start installs them through the same switch (`Leader::install_programs`);
    /// a relaunched process has to be given the very programs its predecessor
    /// ran, because the epoch they belong to is already serving and nothing
    /// else sends them.
    ///
    /// # Errors
    ///
    /// Fails when no epoch serves, when the daemon has no channel to its main
    /// processes, and when the process refuses the programs or does not answer.
    pub fn hand_to(
        &self,
        host: &str,
        engine: u64,
        config: &crate::options::ServeConfig,
    ) -> Result<(), Failure> {
        let commands = locked(&self.commands).clone().ok_or_else(|| Failure {
            kind: "no_command_channel".to_owned(),
            reason: "the daemon has no channel to its main processes".to_owned(),
            device: None,
        })?;
        let (_, record) = self.serving().ok_or_else(|| Failure {
            kind: "no_epoch".to_owned(),
            reason: "the daemon executes no epoch".to_owned(),
            device: None,
        })?;
        let programs = FilledPrograms {
            kernels: record.kernels.clone(),
            policy: record.policy.clone(),
        };
        let deployment = Deployment {
            config: config.clone(),
            build_config: crate::epoch::BuildConfig::of(config),
            programs,
        };
        let process = crate::engine::control::EpochProcess {
            host: host.to_owned(),
            engine,
        };
        let mut switch = crate::engine::control::ProgramSwitch::new(
            &commands,
            vec![process],
            Update::new(0, deployment),
        );
        switch.precopy(&record.kernels, &record.policy)?;
        switch.switchover()?;
        switch.commit()?;
        tracing::info!(
            engine,
            kernels = record.kernels.len(),
            policy = %record.policy.backend,
            "the joining main process took the epoch's programs"
        );
        Ok(())
    }

    /// The program-layer update in flight, if there is one.
    pub fn program_update(&self) -> Option<ProgramUpdate> {
        locked(&self.program).clone()
    }

    /// Whether a program-layer update is on probation: the main processes
    /// execute its generation, and the epoch keeps the generation that served
    /// before it as the last resort until it is committed or reverted (§7.3
    /// step 4).
    pub fn program_in_flight(&self) -> bool {
        self.program_update()
            .is_some_and(|program| program.update.state == UpdateState::Probation)
    }

    /// Applies `update`, a program-layer change a routing decision prepared,
    /// to the main processes `processes`: their programs are prepared
    /// (`Precopy`), they switch to the generation each numbered, and the
    /// epoch's record takes the new programs as the probation's (§7.3 steps
    /// 2-4). The probation lasts until [`Self::commit_programs`] or
    /// [`Self::revert_programs`].
    ///
    /// Refuses an update a revert marked unavailable unless `force` asks for
    /// it anyway (§7.2).
    ///
    /// # Errors
    ///
    /// Fails with the first failure and leaves what serves untouched: a
    /// refused precopy changes nothing, and a fault, a stalled job,
    /// non-finite logits or a policy revert fails the update with its own
    /// account.
    pub fn apply_programs(
        &self,
        update: Update,
        processes: Vec<crate::engine::control::EpochProcess>,
        force: bool,
    ) -> Result<(), Failure> {
        // §7.2: one update is one operation with one probation, committed or
        // reverted as a whole. A second change applied now would replace the
        // epoch's last resort with the generation the first update has not
        // proven yet, so it waits for the commit or the revert.
        if self.program_in_flight() {
            return Err(Failure {
                kind: "in_flight".to_owned(),
                reason: "a program-layer update is on probation: it is committed or reverted \
                         before another change is applied"
                    .to_owned(),
                device: None,
            });
        }
        let commands = locked(&self.commands).clone().ok_or_else(|| Failure {
            kind: "no_command_channel".to_owned(),
            reason: "the daemon has no channel to its main processes".to_owned(),
            device: None,
        })?;
        let (epoch, _) = self.serving().ok_or_else(|| Failure {
            kind: "no_epoch".to_owned(),
            reason: "the daemon executes no epoch".to_owned(),
            device: None,
        })?;
        // §7.2: an update that a revert marked unavailable is refused until
        // it is applied with `force`, whatever it says about itself.
        if !force {
            if let Some(failure) = self.unavailable(&update.target) {
                return Err(Failure {
                    kind: "unavailable".to_owned(),
                    reason: format!(
                        "this update was reverted and is unavailable after a {} failure: {}",
                        failure.kind, failure.reason
                    ),
                    device: failure.device,
                });
            }
        }
        let programs = update.target.programs.clone();
        let mut switch =
            crate::engine::control::ProgramSwitch::new(&commands, processes, update.clone());
        switch.precopy(&programs.kernels, &programs.policy)?;
        switch.switchover()?;
        // The generation the main processes execute is the epoch's now; the
        // probation keeps what it executed before as its last resort.
        self.precopied(epoch, programs);
        let prepared = switch
            .processes()
            .iter()
            .zip(switch.generations().iter().copied())
            .map(|(process, generation)| PreparedProcess {
                host: process.host.clone(),
                engine: process.engine,
                generation,
            })
            .collect();
        let mut update = switch.update().clone();
        update.state = UpdateState::Probation;
        *locked(&self.program) = Some(ProgramUpdate {
            update: update.clone(),
            epoch,
            prepared,
        });
        *locked(&self.update) = Some(update);
        Ok(())
    }

    /// Ends the program-layer probation in favour of the generation that
    /// executes: every main process commits it and purges what it no longer
    /// runs (§7.3 steps 5-6), and the epoch's record keeps no previous
    /// generation.
    ///
    /// Answers `Ok(false)` when the update in flight is not a program-layer
    /// one, so a caller that also has epoch-layer updates asks and falls
    /// through.
    ///
    /// # Errors
    ///
    /// Fails with the first failure of the commit, leaving the probation
    /// standing for [`Self::revert_programs`].
    pub fn commit_programs(&self) -> anyhow::Result<bool> {
        let Some(program) = self.program_update() else {
            return Ok(false);
        };
        let commands = locked(&self.commands)
            .clone()
            .ok_or_else(|| anyhow::anyhow!("the daemon has no channel to its main processes"))?;
        let processes: Vec<_> = program
            .prepared
            .iter()
            .map(|prepared| crate::engine::control::EpochProcess {
                host: prepared.host.clone(),
                engine: prepared.engine,
            })
            .collect();
        let generations: Vec<u64> = program
            .prepared
            .iter()
            .map(|prepared| prepared.generation)
            .collect();
        let mut switch = crate::engine::control::ProgramSwitch::resuming(
            &commands,
            processes,
            program.update.clone(),
            generations,
        );
        if let Err(failure) = switch.commit().and_then(|()| switch.purge()) {
            anyhow::bail!("committing the program update: {}", failure.reason);
        }
        // The epoch keeps serving: what the commit purges is the previous
        // *generation* of programs in every main process, not the epoch.
        self.committed(program.epoch);
        *locked(&self.update) = Some(switch.update().clone());
        *locked(&self.program) = None;
        Ok(true)
    }

    /// Returns every main process to the generation it served before the
    /// program-layer update, with `reason` (§7.3 step 5), so the update is
    /// unavailable with its first failure until it is applied with `force`
    /// (§7.2).
    ///
    /// Answers `Ok(false)` when the update in flight is not a program-layer
    /// one.
    ///
    /// # Errors
    ///
    /// Fails when a main process does not answer the revert. The epochs'
    /// records are the previous generation's either way.
    pub fn revert_programs(&self, reason: &str) -> anyhow::Result<bool> {
        self.revert_programs_because(reason, None)
    }

    /// Reverts the program-layer update on probation because a main process
    /// failed, and records that failure as the update's first one (§7.3
    /// step 4): a main process that exits, an executor it can no longer use, a
    /// stuck job, a non-finite logit or a policy error all end the probation in
    /// favour of the generation that served before it, and the update stays
    /// unavailable with this failure until it is applied with `force` (§7.2).
    ///
    /// Answers whether an update was on probation to revert. A process that
    /// cannot be told is logged, not returned: what serves is the previous
    /// generation either way, and the report moves nothing.
    pub fn program_failure(&self, failure: Failure) -> bool {
        if !self.program_in_flight() {
            return false;
        }
        let reason = format!("{}: {}", failure.kind, failure.reason);
        if let Err(error) = self.revert_programs_because(&reason, Some(failure)) {
            tracing::warn!(%error, "reverting the program update after a failure");
        }
        true
    }

    /// [`Self::revert_programs`], with the failure that caused an automatic
    /// revert recorded as the update's first one when the processes did not
    /// produce one themselves.
    fn revert_programs_because(
        &self,
        reason: &str,
        cause: Option<Failure>,
    ) -> anyhow::Result<bool> {
        let Some(program) = self.program_update() else {
            return Ok(false);
        };
        let commands = locked(&self.commands)
            .clone()
            .ok_or_else(|| anyhow::anyhow!("the daemon has no channel to its main processes"))?;
        let processes: Vec<_> = program
            .prepared
            .iter()
            .map(|prepared| crate::engine::control::EpochProcess {
                host: prepared.host.clone(),
                engine: prepared.engine,
            })
            .collect();
        let generations: Vec<u64> = program
            .prepared
            .iter()
            .map(|prepared| prepared.generation)
            .collect();
        let mut switch = crate::engine::control::ProgramSwitch::resuming(
            &commands,
            processes,
            program.update.clone(),
            generations,
        );
        let outcome = switch.revert(reason);
        self.reverted(program.epoch);
        let mut update = switch.update().clone();
        // The failure that made the revert automatic is the update's first one
        // unless a process reported one of its own while it was being told.
        if let Some(cause) = cause {
            update.fail(cause);
        }
        *locked(&self.update) = Some(update);
        *locked(&self.program) = None;
        if let Err(failure) = outcome {
            anyhow::bail!("reverting the program update: {}", failure.reason);
        }
        Ok(true)
    }

    /// Epoch `epoch`'s record, if the daemon keeps one.
    pub fn get(&self, epoch: u64) -> Option<EpochRecord> {
        locked(&self.records).get(&epoch).cloned()
    }

    /// Every epoch the daemon keeps, by number.
    pub fn records(&self) -> BTreeMap<u64, EpochRecord> {
        locked(&self.records).clone()
    }

    /// The epoch that serves requests: the highest numbered one that is
    /// serving or on probation.
    pub fn serving(&self) -> Option<(u64, EpochRecord)> {
        locked(&self.records)
            .iter()
            .rev()
            .find(|(_, record)| matches!(record.status, Status::Serving | Status::Probation))
            .map(|(epoch, record)| (*epoch, record.clone()))
    }

    /// Replaces the node's process table with `processes`.
    pub fn set_processes(&self, processes: Vec<ManagedProcess>) {
        match self.supervised.get() {
            Some(table) => *locked(table) = processes,
            None => *locked(&self.processes) = processes,
        }
    }

    /// The node's process table.
    pub fn processes(&self) -> Vec<ManagedProcess> {
        match self.supervised.get() {
            Some(table) => locked(table).clone(),
            None => locked(&self.processes).clone(),
        }
    }

    /// Reports the node's processes from `table` from now on: the daemon that
    /// supervises them writes them there as they start and end, so the record
    /// names every process of the deployment and never a stale one (§2.3).
    /// The first table attached is the one kept.
    pub fn attach_processes(&self, table: std::sync::Arc<Mutex<Vec<ManagedProcess>>>) {
        let _ = self.supervised.set(table);
    }

    /// The epochs and the process table as `GET /admin/config` reports
    /// them: one array entry per epoch, ascending by number, with the
    /// programs in the form the daemon records, and one entry per process
    /// the daemon manages.
    pub fn report(&self) -> Value {
        let epochs = locked(&self.records)
            .iter()
            .map(|(epoch, record)| record_report(*epoch, record))
            .collect::<Vec<_>>();
        // The rows the daemon supervises now, not the ones this record was
        // given before it attached its table: a process that started or ended
        // is reported from the table's state (§2.3).
        let processes = self
            .processes()
            .iter()
            .map(process_report)
            .collect::<Vec<_>>();
        serde_json::json!({ "epochs": epochs, "processes": processes })
    }
}

/// One epoch record as JSON: its number, the build config, the programs by
/// operator and phase, the previous generation, and the status, as the
/// design names them.
fn record_report(epoch: u64, record: &EpochRecord) -> Value {
    let kernels = record
        .kernels
        .iter()
        .map(|(slot, program)| {
            serde_json::json!({
                "op": slot.op,
                "phase": slot.phase.map(PhaseKind::as_str),
                "program": program_report(program),
            })
        })
        .collect::<Vec<_>>();
    let previous = record
        .previous_kernels
        .iter()
        .map(|(slot, program)| {
            serde_json::json!({
                "op": slot.op,
                "phase": slot.phase.map(PhaseKind::as_str),
                "program": program_report(program),
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "epoch": epoch,
        "build_config": serde_json::to_value(&record.build_config).unwrap_or(Value::Null),
        "kernels": kernels,
        "policy": program_report(&record.policy),
        "previous_kernels": previous,
        "previous_policy": record.previous_policy.as_ref().map(program_report),
        "generation": record.generation,
        "world": record.world,
        "status": match record.status {
            Status::Precopy => "precopy",
            Status::Serving => "serving",
            Status::Probation => "probation",
            Status::Purged => "purged",
        },
    })
}

/// One program as JSON: its backend, revision and parameters.
fn program_report(program: &Program) -> Value {
    serde_json::json!({
        "backend": program.backend,
        "revision": program.revision,
        "arguments": program.arguments,
    })
}

/// One managed process as JSON.
fn process_report(process: &ManagedProcess) -> Value {
    serde_json::json!({
        "role": match process.role {
            Role::Frontend => "frontend",
            Role::MainProcess => "main_process",
        },
        "pid": process.pid,
        "epoch": process.epoch,
        "group": process.group,
        "rank": process.rank,
        "build": process.build,
        "generation": process.generation,
        "state": match &process.state {
            ProcessState::Starting => "starting".to_owned(),
            ProcessState::Serving => "serving".to_owned(),
            ProcessState::Parked => "parked".to_owned(),
            ProcessState::Stopped => "stopped".to_owned(),
            ProcessState::Failed(reason) => format!("failed: {reason}"),
        },
    })
}

fn locked<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The operator names the epoch record keys its kernel programs by (§5.1,
/// §5.2), of the operators whose backend a configuration selects. They must
/// equal `crates/ops`'s `Op::NAME` of the same operator, which the model and
/// sched swarms write, so a mismatch is a wrong Program Server lookup and
/// shows up at prepare; they are stated once here for that reason.
///
/// The complete slot set is `crates/ops`'s operator names. The rest are
/// filled by the epoch's programs like any other slot and need no entry
/// here: their backend is not a configuration field, so this module names
/// only the operators the configuration selects a backend for. The
/// scheduling interface is not an operator at all: its program is the epoch
/// record's `policy` field.
/// The programs a configuration fills, by operator and phase, and the
/// scheduling interface's (§5.1, §5.2).
///
/// The backends and parameters come from the configuration; the revisions
/// come from `prepare`, which selects each package's latest revision at the
/// upstream Program Server (§5.1). A program's `revision` is zero until it
/// was prepared. An operator whose backend the configuration does not state
/// is absent: `prepare` resolves it by the GPU's architecture and the model
/// and fills it in.
#[derive(Debug, Clone, PartialEq)]
pub struct FilledPrograms {
    /// The kernel program that fills each operator, or each phase of one.
    pub kernels: BTreeMap<KernelKey, Program>,
    /// The scheduling program.
    pub policy: Program,
}

impl FilledPrograms {
    /// The programs `config` fills, without the revisions `prepare` selects.
    pub fn of(config: &ServeConfig) -> Self {
        let mut kernels = BTreeMap::new();
        let attention = attention_programs(config);
        kernels.extend(attention);
        // The exchange slot is filled only when the deployment spreads its
        // experts over ranks. At one rank the model's MoE block takes its local
        // arm and launches no exchange at all (`nn::Moe::forward` routes
        // through `forward_expert_parallel` only when an expert lives on
        // another rank), so a slot filled here would pin a program no call
        // reaches — and, worse, the Program Server would refuse the epoch for a
        // package nothing runs.
        if config.moe_config.expert_parallel {
            kernels.insert(
                KernelKey::all(operators::EXPERT_EXCHANGE),
                Program {
                    backend: word(&config.moe_config.exchange),
                    revision: 0,
                    arguments: toml::Table::new(),
                },
            );
        }
        if let Some(backend) = &config.moe_runner_backend {
            // The grouped expert GEMM is two operators, gather and route,
            // and one program backend fills both (§9).
            for operator in [operators::GATHER_GEMM, operators::ROUTE_GEMM] {
                kernels.insert(
                    KernelKey::all(operator),
                    Program {
                        backend: backend.clone(),
                        revision: 0,
                        arguments: toml::Table::new(),
                    },
                );
            }
        }
        Self {
            kernels,
            policy: Program {
                backend: word(&config.scheduler_config.policy),
                revision: 0,
                arguments: policy_arguments(config),
            },
        }
    }

    /// Refuses a program the deployment left at `auto`, naming the option that
    /// would state it.
    ///
    /// A program is selected by its backend's name: a Program Server holds its
    /// packages under `(operator, backend, revision)` and answers `Latest` by
    /// that name alone, so `auto` resolves at no server — it is a
    /// configuration literal, not a package. A deployment that leaves one of
    /// these at `auto` has not said which program it means, and the refusal
    /// says so here rather than as a missing package later.
    ///
    /// # Errors
    ///
    /// Fails with the option to state for the first program left at `auto`.
    pub fn stated(&self) -> Result<(), String> {
        let option = |key: &KernelKey| match key.op.as_str() {
            operators::PAGED_ATTENTION => "attention_backend",
            operators::EXPERT_EXCHANGE => "moe_config.exchange",
            operators::GATHER_GEMM | operators::ROUTE_GEMM => "moe_runner_backend",
            _ => "the option that names its backend",
        };
        for (key, program) in &self.kernels {
            if program.backend == AUTO_BACKEND {
                return Err(format!(
                    "the deployment leaves {} at `{AUTO_BACKEND}`, and a program is selected by \
                     its backend's name: state the backend this deployment means (§5.1) — the \
                     operator `{}` is filled by the package of that name at the Program Server",
                    option(key),
                    key.op
                ));
            }
        }
        if self.policy.backend == AUTO_BACKEND {
            return Err(format!(
                "the deployment leaves scheduler_config.policy at `{AUTO_BACKEND}`, and a \
                 program is selected by its backend's name: state the scheduling policy this \
                 deployment serves (§5.1, §7.7)"
            ));
        }
        Ok(())
    }
}

/// The backend name `auto`: a choice the deployment has not made yet, which no
/// Program Server holds a package under.
const AUTO_BACKEND: &str = "auto";

/// The attention operator's programs: one entry for every phase when the
/// per-phase backends differ, or a single entry whose phase is `None` when
/// they agree (§5.2).
fn attention_programs(config: &ServeConfig) -> Vec<(KernelKey, Program)> {
    let prefill = config
        .prefill_attention_backend
        .clone()
        .unwrap_or_else(|| config.attention_backend.clone());
    let decode = config
        .decode_attention_backend
        .clone()
        .unwrap_or_else(|| config.attention_backend.clone());
    let arguments = || toml_arguments(&config.flashinfer_config);
    if prefill == decode {
        return vec![(
            KernelKey::all(operators::PAGED_ATTENTION),
            Program {
                backend: prefill,
                revision: 0,
                arguments: arguments(),
            },
        )];
    }
    vec![
        (
            KernelKey::phase(operators::PAGED_ATTENTION, PhaseKind::Prefill),
            Program {
                backend: prefill,
                revision: 0,
                arguments: arguments(),
            },
        ),
        (
            KernelKey::phase(operators::PAGED_ATTENTION, PhaseKind::Decode),
            Program {
                backend: decode,
                revision: 0,
                arguments: arguments(),
            },
        ),
    ]
}

/// The scheduling program's parameters: the values §7.7 assigns to the
/// scheduling policy, which decides with them inside the build config's
/// bounds (§4.4).
fn policy_arguments(config: &ServeConfig) -> toml::Table {
    #[derive(Serialize)]
    struct Arguments {
        #[serde(skip_serializing_if = "Option::is_none")]
        chunked_prefill_size: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_queued_requests: Option<u32>,
        max_batch: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_num_batched_tokens: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        expected_speedup: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        refill_token_cost: Option<f64>,
    }
    let scheduler = &config.scheduler_config;
    // The token bound this launch gives its workers, which is the same
    // `max_num_batched_tokens` its `EngineSpec` states (§2.5): the policy's
    // declared limits are these bounds, not a second opinion about them. A
    // prefill chunk that the deployment states larger than its token bound is
    // refused by the mechanism (`PolicyError::BadLimits`), never clamped here.
    // The token bound is the deployment's stated budget, and when it states
    // none the largest shape its captured graphs bound a batch to — the figure
    // the engine admits a batch under (`proto::catalog::largest_tier`, read by
    // both ends). Without this fallback a deployment that captures a 512-token
    // prefill graph and states no budget hands its policy the guest's 8192-token
    // default, and the mechanism refuses the epoch's limits against the graph's
    // ceiling after the weights are loaded. A stated bound above the catalog is
    // refused at load (`ServeConfig::validate`), never clamped here.
    let token_bound = config
        .max_num_batched_tokens
        .map(|size| u32::try_from(size.get()).unwrap_or(u32::MAX))
        .or_else(|| config.cuda_graph_config.tiers().tokens);
    // A batch of requests is bounded by the largest captured decode graph, and
    // the deployment's batch is the smaller of its own and that: a deployment
    // that leaves the batch at its default means the catalog's figure, and one
    // that states more than its graphs capture is refused at load
    // (`ServeConfig::validate`), never clamped here.
    let max_batch = u32::try_from(scheduler.max_batch.get()).unwrap_or(u32::MAX);
    let max_batch = config
        .cuda_graph_config
        .tiers()
        .rows
        .map_or(max_batch, |rows| max_batch.min(rows));
    let arguments = Arguments {
        chunked_prefill_size: scheduler
            .chunked_prefill_size
            .map(|size| u32::try_from(size.get()).unwrap_or(u32::MAX))
            .or(token_bound),
        max_queued_requests: scheduler
            .max_queued_requests
            .map(|size| u32::try_from(size.get()).unwrap_or(u32::MAX)),
        max_batch,
        max_num_batched_tokens: token_bound,
        expected_speedup: config.speculative_config.expected_speedup,
        refill_token_cost: config.speculative_config.refill_token_cost,
    };
    toml_arguments(&arguments)
}

/// `value`'s fields as the TOML table a program's arguments carry (§5.1).
/// Values the field's serialization leaves out, such as an unset option, are
/// absent.
fn toml_arguments<T: Serialize>(value: &T) -> toml::Table {
    match toml::Value::try_from(value) {
        Ok(toml::Value::Table(table)) => table,
        _ => toml::Table::new(),
    }
}

/// The word a vocabulary value is named by, as the configuration spells it
/// (`fcfs`, `deepep`): its serialized form, which is also the program
/// backend's name (§5.1).
fn word<T: Serialize>(value: &T) -> String {
    match toml::Value::try_from(value) {
        Ok(toml::Value::String(word)) => word,
        Ok(other) => other.to_string(),
        Err(_) => String::new(),
    }
}

/// What a configuration change moves (§7.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// Nothing changed: the update does nothing.
    None,
    /// Only the filled programs changed: they move inside the serving main
    /// process (§7.3).
    Program,
    /// The build config changed: a new epoch is built (§7.4).
    Epoch,
    /// The daemon's or the frontend's own options changed, or their
    /// executables: the process is handed over to a successor (§7.5), whose
    /// own routing applies what else the change states.
    Host,
}

/// A deployment as the daemon routes a change between: its configuration,
/// the build config it builds a main process from, and the programs it
/// fills.
#[derive(Debug, Clone, PartialEq)]
pub struct Deployment {
    /// The configuration.
    pub config: ServeConfig,
    /// The projection that builds a main process.
    pub build_config: BuildConfig,
    /// The filled programs; the deployment a change is compared against
    /// carries the revisions `prepare` selected, and the target's are
    /// prepared before the comparison.
    pub programs: FilledPrograms,
}

impl Deployment {
    /// The deployment `config` describes, with the programs it fills and
    /// without the revisions `prepare` selects.
    pub fn of(config: ServeConfig) -> Self {
        Self {
            build_config: BuildConfig::of(&config),
            programs: FilledPrograms::of(&config),
            config,
        }
    }
}

/// Which layer `next` moves `previous` on (§7.2), mechanically: the
/// host layer when the options the daemon or the frontend read alone
/// differ, the epoch layer when the build config differs, the program layer
/// when only the filled programs differ, and nothing otherwise. Nothing
/// about a change is declared by the change itself.
pub fn layer(previous: &Deployment, next: &Deployment) -> Layer {
    if host_layer_differ(&previous.config, &next.config) {
        Layer::Host
    } else if previous.build_config != next.build_config {
        Layer::Epoch
    } else if previous.programs != next.programs {
        Layer::Program
    } else {
        Layer::None
    }
}

/// Whether `current` and `target` differ in the options assigned to the host
/// layer (§7.7): the ones the frontend reads alone (its HTTP server, the
/// request and output processing, the media and video jobs, migration and
/// the model's processor options) and the ones the daemon reads alone
/// (logging, the update's own policy and the admin API's address). A change
/// of them is a hot restart of the process that reads them (§7.5).
///
/// `readout_config` and `diffusion_generation_config` are the daemon's own
/// today and belong to neither list; §7.7 retires them, and until then a
/// change of them is the host layer's rather than silently ignored.
pub fn host_layer_differ(current: &ServeConfig, target: &ServeConfig) -> bool {
    crate::options::daemon_options_differ(current, target)
        || current.update_config != target.update_config
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::engine::stand_in::Seen;
    use crate::options::SchedulerOptions;

    /// A program of backend `backend` at revision 1 with no parameters.
    fn program(backend: &str) -> Program {
        Program {
            backend: backend.to_owned(),
            revision: 1,
            arguments: toml::Table::new(),
        }
    }

    /// A serving epoch's record at the default configuration.
    fn record() -> EpochRecord {
        let config = ServeConfig::default();
        EpochRecord {
            build_config: BuildConfig::of(&config),
            kernels: BTreeMap::from([(KernelKey::all("rms_norm"), program("triton"))]),
            policy: program("fcfs"),
            previous_kernels: BTreeMap::new(),
            previous_policy: None,
            status: Status::Serving,
            generation: None,
            world: 3,
        }
    }

    /// A main process that joins the serving epoch after it was installed — the
    /// process a relaunch starts — is handed the epoch's own programs and
    /// switched to them. A process that starts with none waits for a
    /// program-layer command nothing sends, and answers 503 to every request,
    /// so the epoch layer is what gives it them (§2.5, §7.5).
    #[test]
    fn a_joining_main_process_takes_the_epochs_own_programs() {
        use crate::engine::stand_in::{Answer, Seen, main_processes_recording};

        let (commands, seen, carried) = main_processes_recording(vec![(9, Answer::Serves(12))]);
        let epochs = Epochs::new();
        let config = ServeConfig::default();
        epochs.record(0, record());
        epochs.supervise(commands);

        epochs
            .hand_to("node-0", 9, &config)
            .expect("the epoch's programs reached the joining process");
        assert_eq!(
            *seen.lock().unwrap(),
            vec![
                (9, Seen::Precopy),
                (9, Seen::Switchover(12)),
                (9, Seen::Commit)
            ],
            "the joining process prepares, switches to and commits the epoch's programs"
        );
        let carried = carried.lock().unwrap();
        assert_eq!(carried.len(), 1);
        assert_eq!(
            carried[0].1.kernels, 1,
            "the very operator the epoch's record names, not an empty set"
        );
        assert_eq!(carried[0].1.policy, "fcfs");

        // A daemon that executes no epoch has nothing to hand over, and says so
        // rather than leaving the process waiting.
        let (commands, _seen, _carried) = main_processes_recording(vec![(9, Answer::Serves(12))]);
        let empty = Epochs::new();
        empty.supervise(commands);
        let failure = empty
            .hand_to("node-0", 9, &config)
            .expect_err("a daemon with no epoch hands nothing over");
        assert_eq!(failure.kind, "no_epoch");
    }

    /// A program-layer update in flight is the epoch's probation: the record
    /// names the programs the main processes took, the previous generation is
    /// kept, and a commit ends the probation without the epoch stopping
    /// (§7.3 steps 4-6).
    #[test]
    fn a_program_update_serves_from_the_generation_it_switched_to_until_it_is_committed() {
        use crate::engine::stand_in::{Answer, Seen, main_processes, process};

        let (commands, seen) =
            main_processes(vec![(7, Answer::Serves(11)), (8, Answer::Serves(12))]);
        let epochs = Epochs::new();
        let config = ServeConfig::default();
        let serving = Deployment {
            config: config.clone(),
            build_config: BuildConfig::of(&config),
            programs: FilledPrograms::of(&config),
        };
        epochs.install(0, &serving, 3, &[10]);
        epochs.supervise(commands);

        // A change that moves only the programs: one operator's backend.
        let mut target = serving.clone();
        let mut programs = FilledPrograms::of(&config);
        programs
            .policy
            .arguments
            .insert("max_batch".to_owned(), toml::Value::Integer(8));
        target.programs = programs.clone();
        let update = Update::new(1, target);
        epochs
            .apply_programs(
                update,
                vec![process("rank-0", 7), process("rank-1", 8)],
                false,
            )
            .expect("both main processes took the programs");

        // The epoch is on probation from the generation the update switched
        // to, and the record names what executes.
        let program = epochs.program_update().expect("the update is in flight");
        assert_eq!(program.update.number, 1);
        assert_eq!(program.update.state, UpdateState::Probation);
        assert_eq!(program.epoch, 0);
        let prepared: Vec<u64> = program.prepared.iter().map(|p| p.generation).collect();
        assert_eq!(prepared, vec![11, 12]);
        let serving_epoch = epochs.get(0).expect("the epoch is recorded");
        assert_eq!(serving_epoch.status, Status::Probation);
        assert_eq!(serving_epoch.policy, programs.policy);
        assert!(serving_epoch.previous_policy.is_some());

        assert!(
            epochs
                .commit_programs()
                .expect("both main processes committed"),
            "a program-layer update is committed here"
        );
        let serving_epoch = epochs.get(0).expect("the epoch is recorded");
        assert_eq!(serving_epoch.status, Status::Serving);
        assert!(serving_epoch.previous_policy.is_none());
        assert!(
            serving_epoch.previous_kernels.is_empty(),
            "a commit keeps no previous generation"
        );
        assert!(epochs.program_update().is_none());

        // Each process was commanded in the protocol's order, and the
        // generation each switched to is the one it numbered itself.
        let seen = seen.lock().unwrap().clone();
        assert_eq!(
            seen,
            vec![
                (7, Seen::Precopy),
                (8, Seen::Precopy),
                (7, Seen::Switchover(11)),
                (8, Seen::Switchover(12)),
                (7, Seen::Commit),
                (8, Seen::Commit),
                (7, Seen::Purge(11)),
                (8, Seen::Purge(12)),
            ]
        );
    }

    /// A main process that fails during the probation ends it in favour of the
    /// generation that served before, and the update stays unavailable with the
    /// failure that caused it until it is applied with `force` (§7.2, §7.3
    /// step 4).
    #[test]
    fn a_main_process_failure_during_probation_reverts_the_program_update() {
        use crate::engine::stand_in::{Answer, Seen, main_processes, process};

        let (commands, seen) = main_processes(vec![(7, Answer::Serves(11))]);
        let epochs = Epochs::new();
        let config = ServeConfig::default();
        let serving = Deployment {
            config: config.clone(),
            build_config: BuildConfig::of(&config),
            programs: FilledPrograms::of(&config),
        };
        epochs.install(0, &serving, 3, &[10]);
        epochs.supervise(commands);

        let mut target = serving.clone();
        let mut programs = FilledPrograms::of(&config);
        programs
            .policy
            .arguments
            .insert("max_batch".to_owned(), toml::Value::Integer(8));
        target.programs = programs.clone();
        epochs
            .apply_programs(
                Update::new(1, target.clone()),
                vec![process("rank-0", 7)],
                false,
            )
            .expect("the update switches over");
        assert!(epochs.program_in_flight());

        let failure = Failure {
            kind: "illegal_address".to_owned(),
            reason: "the executor can no longer run".to_owned(),
            device: Some(0),
        };
        assert!(
            epochs.program_failure(failure.clone()),
            "a failure during probation reverts the update"
        );

        // The previous generation serves again, nothing is in flight, and the
        // update is unavailable for the failure that caused the revert.
        let record = epochs.get(0).expect("the epoch is recorded");
        assert_eq!(record.status, Status::Serving);
        assert_eq!(record.policy, serving.programs.policy);
        assert!(record.previous_policy.is_none());
        assert!(epochs.program_update().is_none());
        assert_eq!(
            epochs
                .update()
                .expect("the update is recorded")
                .first_failure,
            Some(failure.clone())
        );
        assert_eq!(epochs.unavailable(&target), Some(failure));
        assert!(
            !epochs.program_failure(Failure {
                kind: "policy_error".to_owned(),
                reason: "no probation to end".to_owned(),
                device: None,
            }),
            "a failure after the revert moves nothing"
        );

        let seen = seen.lock().unwrap().clone();
        assert!(
            matches!(seen.last(), Some((7, Seen::Revert(_)))),
            "the main process was told to revert: {seen:?}"
        );
    }

    /// One update is one operation with one probation (§7.2): a second
    /// program-layer change waits for the first to be committed or reverted,
    /// so the epoch's last resort stays the generation that served before the
    /// first began (§7.3 steps 4 and 5).
    #[test]
    fn a_second_program_update_waits_for_the_first_ones_probation() {
        use crate::engine::stand_in::{Answer, main_processes, process};

        let (commands, _seen) = main_processes(vec![(7, Answer::Serves(11))]);
        let epochs = Epochs::new();
        let config = ServeConfig::default();
        let serving = Deployment {
            config: config.clone(),
            build_config: BuildConfig::of(&config),
            programs: FilledPrograms::of(&config),
        };
        epochs.install(0, &serving, 3, &[10]);
        epochs.supervise(commands);

        let mut first = serving.clone();
        let mut programs = FilledPrograms::of(&config);
        programs
            .policy
            .arguments
            .insert("max_batch".to_owned(), toml::Value::Integer(8));
        first.programs = programs;
        epochs
            .apply_programs(Update::new(1, first), vec![process("rank-0", 7)], false)
            .expect("the first update switches over");
        let last_resort = epochs
            .get(0)
            .expect("the epoch is recorded")
            .previous_policy
            .clone();
        assert!(
            last_resort.is_some(),
            "the first update keeps the policy it replaced"
        );

        let mut second = serving.clone();
        let mut programs = FilledPrograms::of(&config);
        programs
            .policy
            .arguments
            .insert("max_batch".to_owned(), toml::Value::Integer(4));
        second.programs = programs;
        let failure = epochs
            .apply_programs(Update::new(2, second), vec![process("rank-0", 7)], false)
            .expect_err("an update on probation is not replaced");
        assert_eq!(failure.kind, "in_flight");

        // Nothing moved: the first update is still the one in flight, and the
        // last resort is still the generation that served before it.
        assert_eq!(
            epochs
                .program_update()
                .expect("the first update is in flight")
                .update
                .number,
            1
        );
        assert_eq!(
            epochs
                .get(0)
                .expect("the epoch is recorded")
                .previous_policy,
            last_resort
        );
    }

    /// A revert puts the previous generation back in every main process and
    /// in the epoch's record, and the update stays unavailable with its first
    /// failure until it is applied with `force` (§7.2, §7.3 step 5).
    #[test]
    fn a_reverted_program_update_leaves_the_previous_programs_serving() {
        use crate::engine::stand_in::{Answer, main_processes, process};

        let (commands, seen) = main_processes(vec![(7, Answer::Serves(11))]);
        let epochs = Epochs::new();
        let config = ServeConfig::default();
        let serving = Deployment {
            config: config.clone(),
            build_config: BuildConfig::of(&config),
            programs: FilledPrograms::of(&config),
        };
        epochs.install(0, &serving, 3, &[10]);
        epochs.supervise(commands);

        let mut target = serving.clone();
        let mut programs = FilledPrograms::of(&config);
        programs
            .policy
            .arguments
            .insert("max_batch".to_owned(), toml::Value::Integer(8));
        target.programs = programs;
        epochs
            .apply_programs(Update::new(2, target), vec![process("rank-0", 7)], false)
            .expect("the main process took the programs");
        assert!(
            epochs
                .revert_programs("a trial refused it")
                .expect("the revert was answered")
        );

        let serving_epoch = epochs.get(0).expect("the epoch is recorded");
        assert_eq!(serving_epoch.status, Status::Serving);
        assert_eq!(
            serving_epoch.policy, serving.programs.policy,
            "the previous programs serve again"
        );
        let reverted = seen.lock().unwrap().clone();
        assert!(
            reverted.iter().any(|(engine, what)| *engine == 7
                && *what == Seen::Revert("a trial refused it".to_owned())),
            "{reverted:?}"
        );
        let update = epochs.update().expect("the update is recorded");
        assert_eq!(update.state, UpdateState::Reverted);
    }

    /// An update a revert marked unavailable is refused until it is applied
    /// with `force`, which is the whole of §7.2's refusal: the same target,
    /// applied again, is not taken twice by accident.
    #[test]
    fn a_reverted_update_is_refused_until_it_is_forced() {
        use crate::engine::stand_in::{Answer, main_processes, process};

        let (commands, _) = main_processes(vec![(7, Answer::Serves(11)), (8, Answer::Serves(12))]);
        let epochs = Epochs::new();
        let config = ServeConfig::default();
        let serving = Deployment {
            config: config.clone(),
            build_config: BuildConfig::of(&config),
            programs: FilledPrograms::of(&config),
        };
        epochs.install(0, &serving, 3, &[10]);
        epochs.supervise(commands);

        let mut target = serving.clone();
        let mut programs = FilledPrograms::of(&config);
        programs
            .policy
            .arguments
            .insert("max_batch".to_owned(), toml::Value::Integer(8));
        target.programs = programs;

        let processes = || vec![process("rank-0", 7), process("rank-1", 8)];
        epochs
            .apply_programs(Update::new(1, target.clone()), processes(), false)
            .expect("the first application is taken");
        // The revert a failure causes, which is the one §7.2 makes the update
        // unavailable with (§7.3 step 4): the failure is recorded as the
        // update's first one, and the reason the refusal quotes is its
        // account.
        assert!(
            epochs.program_failure(Failure {
                kind: "policy_reverted".to_owned(),
                reason: "the policy refused a decision".to_owned(),
                device: None,
            }),
            "the update on probation is reverted"
        );

        let refused = epochs
            .apply_programs(Update::new(1, target.clone()), processes(), false)
            .expect_err("a reverted update is unavailable");
        assert_eq!(refused.kind, "unavailable");
        assert!(
            refused.reason.contains("the policy refused a decision"),
            "{}",
            refused.reason
        );
        assert!(epochs.program_update().is_none());

        epochs
            .apply_programs(Update::new(1, target), processes(), true)
            .expect("force applies it anyway");
        assert!(epochs.program_update().is_some());
    }

    /// A refused precopy changes nothing that serves, and an update in flight
    /// is what a commit finds: with none, a commit answers that it is not a
    /// program-layer update, so a caller falls through to the world path.
    #[test]
    fn a_refused_precopy_records_no_probation_and_a_commit_falls_through() {
        use crate::engine::stand_in::{Answer, main_processes, process};

        let (commands, _) = main_processes(vec![(
            7,
            Answer::Refuses("the policy module did not build".to_owned()),
        )]);
        let epochs = Epochs::new();
        let config = ServeConfig::default();
        let serving = Deployment {
            config: config.clone(),
            build_config: BuildConfig::of(&config),
            programs: FilledPrograms::of(&config),
        };
        epochs.install(0, &serving, 3, &[10]);
        epochs.supervise(commands);
        assert!(!epochs.commit_programs().expect("nothing to commit"));

        let mut target = serving.clone();
        let mut programs = FilledPrograms::of(&config);
        programs
            .policy
            .arguments
            .insert("max_batch".to_owned(), toml::Value::Integer(8));
        target.programs = programs;
        let failure = epochs
            .apply_programs(Update::new(3, target), vec![process("rank-0", 7)], false)
            .expect_err("a refused precopy fails the update");
        assert_eq!(failure.kind, "precopy_failed");
        assert!(epochs.program_update().is_none());
        let serving_epoch = epochs.get(0).expect("the epoch is recorded");
        assert_eq!(serving_epoch.status, Status::Serving);
        assert_eq!(serving_epoch.policy, serving.programs.policy);
    }

    /// The process table reports what the daemon attached: every process it
    /// supervises, with the fields the daemon's own service wrote, and no
    /// row of a process that ended (§2.3).
    #[test]
    fn the_process_table_reports_the_processes_the_daemon_supervises() {
        let epochs = Epochs::new();
        epochs.set_processes(vec![ManagedProcess {
            role: Role::MainProcess,
            group: "engine".to_owned(),
            rank: 0,
            pid: Some(41),
            epoch: Some(0),
            build: "uniserve".to_owned(),
            generation: 3,
            state: ProcessState::Serving,
        }]);
        let table = std::sync::Arc::new(Mutex::new(Vec::new()));
        epochs.attach_processes(std::sync::Arc::clone(&table));

        // The daemon's service owns the rows from now on: an empty table
        // reports no process, and the record's own rows are not reported
        // beside it.
        assert!(epochs.processes().is_empty());
        *locked(&table) = vec![ManagedProcess {
            role: Role::MainProcess,
            group: "engine".to_owned(),
            rank: 0,
            pid: Some(42),
            epoch: Some(0),
            build: "uniserve".to_owned(),
            generation: 4,
            state: ProcessState::Serving,
        }];
        let reported = epochs.report();
        assert_eq!(reported["processes"].as_array().map(Vec::len), Some(1));
        assert_eq!(reported["processes"][0]["pid"], serde_json::json!(42));
        assert_eq!(reported["processes"][0]["generation"], serde_json::json!(4));
        assert_eq!(reported["processes"][0]["state"], "serving");
    }

    /// The policy's declared limits are the launch's own bounds: the token
    /// bound the deployment gives its workers is the policy's
    /// `max_num_batched_tokens`, or the largest shape its captured graphs bound
    /// a batch to when it states no budget; an unstated prefill chunk is that
    /// same bound, so the policy's limits cannot disagree with the workers'
    /// `BatchCapacity` (§4.4). A deployment that states a chunk larger than
    /// its bound keeps its own value here — the mechanism refuses it
    /// (`PolicyError::BadLimits`), and `ServeConfig::validate` refuses it at
    /// load by the option's name, rather than this code clamping it.
    #[test]
    fn the_policys_limits_are_the_launchs_own_token_bound() {
        let mut config = ServeConfig::default();
        config.max_num_batched_tokens = NonZeroUsize::new(8192);
        config.scheduler_config.chunked_prefill_size = None;
        let programs = FilledPrograms::of(&config);
        assert_eq!(
            programs.policy.arguments["max_num_batched_tokens"],
            toml::Value::Integer(8192)
        );
        assert_eq!(
            programs.policy.arguments["chunked_prefill_size"],
            toml::Value::Integer(8192),
            "an unstated chunk is the deployment's own token bound"
        );

        // A stated chunk above the bound is stated, not clamped: the
        // mechanism's `BadLimits` is the refusal.
        config.scheduler_config.chunked_prefill_size = NonZeroUsize::new(16384);
        let programs = FilledPrograms::of(&config);
        assert_eq!(
            programs.policy.arguments["chunked_prefill_size"],
            toml::Value::Integer(16384)
        );

        // A deployment that states a graph catalog and no budget gives its
        // policy the largest shape the catalog bounds a batch to: the same
        // figure the engine admits a batch under, so a policy that states no
        // bound of its own never declares more than the engine built for.
        let mut config = ServeConfig::default();
        config.cuda_graph_config.prefill_token_sizes = Some(vec![
            NonZeroU32::new(128).unwrap(),
            NonZeroU32::new(512).unwrap(),
        ]);
        config.cuda_graph_config.decode_batch_sizes = Some(vec![
            NonZeroU32::new(1).unwrap(),
            NonZeroU32::new(8).unwrap(),
        ]);
        let programs = FilledPrograms::of(&config);
        assert_eq!(
            programs.policy.arguments["max_num_batched_tokens"],
            toml::Value::Integer(512),
            "an unstated budget is the largest shape the captured graphs bound a batch to"
        );
        assert_eq!(
            programs.policy.arguments["chunked_prefill_size"],
            toml::Value::Integer(512),
            "and an unstated chunk is that same bound"
        );

        // The batch the policy may hold follows the same catalog: the
        // deployment's own batch, or the largest decode graph when it states
        // none, and a catalog that grows with a new world moves the derived
        // figure with it.
        let mut config = ServeConfig::default();
        config.cuda_graph_config.decode_batch_sizes = Some(vec![
            NonZeroU32::new(1).unwrap(),
            NonZeroU32::new(2).unwrap(),
            NonZeroU32::new(4).unwrap(),
            NonZeroU32::new(8).unwrap(),
        ]);
        let programs = FilledPrograms::of(&config);
        assert_eq!(
            programs.policy.arguments["max_batch"],
            toml::Value::Integer(8),
            "an unstated batch is the largest decode graph the deployment captured"
        );
        config
            .cuda_graph_config
            .decode_batch_sizes
            .as_mut()
            .unwrap()
            .push(NonZeroU32::new(16).unwrap());
        let programs = FilledPrograms::of(&config);
        assert_eq!(
            programs.policy.arguments["max_batch"],
            toml::Value::Integer(16),
            "and it moves with the catalog a new world captures"
        );
        // A batch the deployment states below the catalog keeps its own value.
        config.scheduler_config.max_batch = NonZeroUsize::new(4).unwrap();
        let programs = FilledPrograms::of(&config);
        assert_eq!(
            programs.policy.arguments["max_batch"],
            toml::Value::Integer(4)
        );

        // A deployment that states no token bound and no catalog states no
        // limit, and the mechanism's ceilings are the workers' own.
        let config = ServeConfig::default();
        let programs = FilledPrograms::of(&config);
        assert!(
            !programs
                .policy
                .arguments
                .contains_key("max_num_batched_tokens")
        );
        assert!(
            !programs
                .policy
                .arguments
                .contains_key("chunked_prefill_size")
        );
    }

    /// Which layer a change moves on is decided by what it moves, never by
    /// the change itself (§7.2, §7.7): the host layer for the options the
    /// daemon or the frontend read alone, the epoch layer for the values
    /// that build a main process, the program layer for the backends and
    /// parameters of the programs, nothing for no change.
    ///
    /// This test is also the rule's enforcement: a value that leaks from the
    /// program layer's lists into [`BuildConfig::of`], or out of it, fails
    /// here by name.
    #[test]
    fn a_change_is_routed_by_what_it_moves() {
        /// How a case changes the default configuration.
        fn set(change: fn(&mut ServeConfig)) -> ServeConfig {
            let mut config = ServeConfig::default();
            change(&mut config);
            config
        }

        // The frontend's own options (§7.7 host layer, frontend row).
        let host_frontend: Vec<(&str, fn(&mut ServeConfig))> = vec![
            ("server_config.port", |config: &mut ServeConfig| {
                config.server_config.port = 8001
            }),
            ("chat_config", |config: &mut ServeConfig| {
                config.chat_config.template = Some("{{messages}}".to_owned())
            }),
            ("mm_processor_kwargs", |config: &mut ServeConfig| {
                config
                    .mm_processor_kwargs
                    .insert("min_pixels".to_owned(), Value::from(1024));
            }),
        ];
        // The daemon's own options (§7.7 host layer, daemon row).
        let host_daemon: Vec<(&str, fn(&mut ServeConfig))> = vec![
            ("logging_config.log_requests", |config: &mut ServeConfig| {
                config.logging_config.log_requests = true
            }),
            ("update_config", |config: &mut ServeConfig| {
                config.update_config.eager_prepare = !config.update_config.eager_prepare
            }),
        ];
        // The values that build a main process (§7.7 build-config row).
        let epoch: Vec<(&str, fn(&mut ServeConfig))> = vec![
            ("max_running_requests", |config: &mut ServeConfig| {
                config.max_running_requests = NonZeroUsize::new(4).unwrap()
            }),
            ("max_model_len", |config: &mut ServeConfig| {
                config.max_model_len = Some(NonZeroU32::new(2048).unwrap())
            }),
            ("mem_fraction_static", |config: &mut ServeConfig| {
                config.mem_fraction_static = 0.5
            }),
            (
                "speculative_config.num_draft_tokens",
                |config: &mut ServeConfig| config.speculative_config.num_draft_tokens = 4,
            ),
            ("moe_config.microbatches", |config: &mut ServeConfig| {
                config.moe_config.microbatches = NonZeroU32::new(2).unwrap()
            }),
            (
                "profiling_config.graph_pc_sampling",
                |config: &mut ServeConfig| config.profiling_config.graph_pc_sampling = true,
            ),
        ];
        // The program backends (§7.7 backend row) and the program parameters
        // (§7.7 parameter row).
        let program: Vec<(&str, fn(&mut ServeConfig))> = vec![
            ("scheduler_config.policy", |config: &mut ServeConfig| {
                config.scheduler_config.policy = proto::request::SchedulingPolicy::Priority
            }),
            (
                "scheduler_config.chunked_prefill_size",
                |config: &mut ServeConfig| {
                    config.scheduler_config.chunked_prefill_size = NonZeroUsize::new(2048)
                },
            ),
            ("scheduler_config.max_batch", |config: &mut ServeConfig| {
                config.scheduler_config.max_batch = NonZeroUsize::new(8).unwrap()
            }),
            ("max_num_batched_tokens", |config: &mut ServeConfig| {
                config.max_num_batched_tokens = NonZeroUsize::new(4096)
            }),
            (
                "speculative_config.expected_speedup",
                |config: &mut ServeConfig| config.speculative_config.expected_speedup = Some(2.5),
            ),
            (
                "speculative_config.refill_token_cost",
                |config: &mut ServeConfig| config.speculative_config.refill_token_cost = Some(0.5),
            ),
            ("attention_backend", |config: &mut ServeConfig| {
                config.attention_backend = "flashinfer".to_owned()
            }),
            ("prefill_attention_backend", |config: &mut ServeConfig| {
                config.prefill_attention_backend = Some("fa4".to_owned())
            }),
            ("decode_attention_backend", |config: &mut ServeConfig| {
                config.decode_attention_backend = Some("trtllm".to_owned())
            }),
            ("moe_config.exchange", |config: &mut ServeConfig| {
                config.moe_config.exchange = proto::request::ExpertExchange::DeepEp
            }),
            ("moe_runner_backend", |config: &mut ServeConfig| {
                config.moe_runner_backend = Some("cutlass".to_owned())
            }),
            (
                "flashinfer_config.workspace_size",
                |config: &mut ServeConfig| config.flashinfer_config.workspace_size = 1 << 30,
            ),
        ];

        let base = Deployment::of(ServeConfig::default());
        assert_eq!(layer(&base, &base), Layer::None, "no change");

        for (what, change) in host_frontend.iter().chain(host_daemon.iter()) {
            let next = Deployment::of(set(*change));
            assert_eq!(
                layer(&base, &next),
                Layer::Host,
                "{what} reads the host layer"
            );
        }
        for (what, change) in &epoch {
            let next = Deployment::of(set(*change));
            assert_eq!(
                layer(&base, &next),
                Layer::Epoch,
                "{what} builds a new epoch"
            );
        }
        for (what, change) in &program {
            let next = Deployment::of(set(*change));
            assert_eq!(
                layer(&base, &next),
                Layer::Program,
                "{what} moves the programs"
            );
        }

        // Only the epoch layer's values build the main process: the host
        // layer's and the program layer's leave the build config equal
        // (§7.7).
        for (what, change) in host_frontend
            .iter()
            .chain(host_daemon.iter())
            .chain(program.iter())
        {
            assert_eq!(
                BuildConfig::of(&set(*change)),
                base.build_config,
                "{what} is not a build-config value"
            );
        }
        for (what, change) in &epoch {
            assert_ne!(
                BuildConfig::of(&set(*change)),
                base.build_config,
                "{what} builds the main process"
            );
        }

        // A program's revision is a program-layer change: the same backend
        // and parameters at a newly installed revision move the programs.
        let mut installed = base.clone();
        for program in installed.programs.kernels.values_mut() {
            program.revision = 3;
        }
        assert_eq!(layer(&base, &installed), Layer::Program, "a new revision");
    }

    /// The programs a configuration fills are keyed by the operators the
    /// design names, with the attention operator split by phase only when
    /// its per-phase fields state different backends (§5.2), and the
    /// scheduling program carries the values that only inform its decisions
    /// (§7.7).
    #[test]
    fn the_filled_programs_name_the_operators_and_phases() {
        let config = ServeConfig::default();
        let programs = FilledPrograms::of(&config);

        // One attention program for every phase, and no expert exchange for a
        // deployment whose experts all live on one rank: the model's MoE block
        // takes its local arm and launches no exchange, so the slot is absent
        // rather than filled with a program nothing calls (§5.1, §9). No
        // grouped expert GEMM either, until the configuration names one.
        let attention = &programs.kernels[&KernelKey::all(operators::PAGED_ATTENTION)];
        assert_eq!(attention.backend, "auto");
        assert_eq!(attention.revision, 0, "no revision before prepare");
        for operator in [
            operators::EXPERT_EXCHANGE,
            operators::GATHER_GEMM,
            operators::ROUTE_GEMM,
        ] {
            assert!(
                !programs.kernels.contains_key(&KernelKey::all(operator)),
                "an operator the configuration does not fill is absent"
            );
        }

        // A deployment that spreads its experts over ranks does route through
        // the exchange, and the slot states the backend its configuration
        // names.
        let mut parallel = ServeConfig::default();
        parallel.moe_config.expert_parallel = true;
        let programs = FilledPrograms::of(&parallel);
        assert_eq!(
            programs.kernels[&KernelKey::all(operators::EXPERT_EXCHANGE)].backend,
            "alltoall"
        );

        // The scheduling program's backend is the policy's name, and its
        // parameters are the values §7.7 assigns to it, as TOML strings.
        assert_eq!(programs.policy.backend, "fcfs");
        assert_eq!(
            programs.policy.arguments["max_batch"],
            toml::Value::Integer(128)
        );
        let scheduler = SchedulerOptions {
            max_queued_requests: NonZeroUsize::new(16),
            chunked_prefill_size: NonZeroUsize::new(2048),
            ..SchedulerOptions::default()
        };
        let mut stated = ServeConfig::default();
        stated.scheduler_config = scheduler;
        stated.max_num_batched_tokens = NonZeroUsize::new(4096);
        stated.speculative_config.expected_speedup = Some(2.5);
        let programs = FilledPrograms::of(&stated);
        assert_eq!(
            programs.policy.arguments["chunked_prefill_size"],
            toml::Value::Integer(2048)
        );
        assert_eq!(
            programs.policy.arguments["max_queued_requests"],
            toml::Value::Integer(16)
        );
        assert_eq!(
            programs.policy.arguments["max_num_batched_tokens"],
            toml::Value::Integer(4096)
        );
        assert_eq!(
            programs.policy.arguments["expected_speedup"],
            toml::Value::Float(2.5)
        );

        // Per-phase attention: two entries, each naming its phase, once the
        // fields state different backends.
        let mut split = ServeConfig::default();
        split.prefill_attention_backend = Some("fa4".to_owned());
        split.decode_attention_backend = Some("trtllm".to_owned());
        let programs = FilledPrograms::of(&split);
        let prefill =
            &programs.kernels[&KernelKey::phase(operators::PAGED_ATTENTION, PhaseKind::Prefill)];
        let decode =
            &programs.kernels[&KernelKey::phase(operators::PAGED_ATTENTION, PhaseKind::Decode)];
        assert_eq!(
            (prefill.backend.as_str(), decode.backend.as_str()),
            ("fa4", "trtllm")
        );
        assert!(
            !programs
                .kernels
                .contains_key(&KernelKey::all(operators::PAGED_ATTENTION)),
            "a per-phase fill covers no whole operator"
        );

        // One backend for both phases fills the operator once, whether the
        // operator states it or both phases state the same one; an unstated
        // phase falls back to `attention_backend`, which the split above
        // keeps as a phase of its own.
        let mut one = ServeConfig::default();
        one.attention_backend = "fa4".to_owned();
        let programs = FilledPrograms::of(&one);
        assert_eq!(
            programs.kernels[&KernelKey::all(operators::PAGED_ATTENTION)].backend,
            "fa4"
        );
        assert!(
            !programs.kernels.contains_key(&KernelKey::phase(
                operators::PAGED_ATTENTION,
                PhaseKind::Prefill
            )),
            "one backend covers every phase of the operator"
        );
        let mut agreed = ServeConfig::default();
        agreed.prefill_attention_backend = Some("fa4".to_owned());
        agreed.decode_attention_backend = Some("fa4".to_owned());
        let programs = FilledPrograms::of(&agreed);
        assert_eq!(
            programs.kernels[&KernelKey::all(operators::PAGED_ATTENTION)].backend,
            "fa4",
            "phases that agree collapse to the operator's slot"
        );

        // A stated grouped expert GEMM program fills both of its operators.
        let mut moe = ServeConfig::default();
        moe.moe_runner_backend = Some("cutlass".to_owned());
        let programs = FilledPrograms::of(&moe);
        for operator in [operators::GATHER_GEMM, operators::ROUTE_GEMM] {
            assert_eq!(
                programs.kernels[&KernelKey::all(operator)].backend,
                "cutlass"
            );
        }
    }

    /// A program is selected by its backend's name, and `auto` names no
    /// package: a deployment that leaves a program at `auto` is refused here,
    /// with the option that states it, rather than at a Program Server that
    /// has no such package (§5.1).
    #[test]
    fn a_program_left_at_auto_is_refused_by_the_option_that_states_it() {
        let refused = FilledPrograms::of(&ServeConfig::default())
            .stated()
            .expect_err("the default attention backend is a choice, not a package");
        assert!(
            refused.contains("attention_backend") && refused.contains("auto"),
            "{refused}"
        );

        let mut stated = ServeConfig::default();
        stated.attention_backend = "trtllm".to_owned();
        stated.moe_config.expert_parallel = true;
        stated.moe_config.exchange = proto::request::ExpertExchange::AllToAll;
        FilledPrograms::of(&stated)
            .stated()
            .expect("every program this deployment fills has a backend's name");
    }

    /// The update keeps the first failure it meets, not the last, and a
    /// revert marks it unavailable with that first error (§7.3 step 4): the
    /// daemon reports the reason with the update and refuses it without
    /// `force` afterwards.
    #[test]
    fn an_update_keeps_the_first_failure_and_is_unavailable_with_it() {
        let mut update = Update::new(1, Deployment::of(ServeConfig::default()));
        update.commit();
        assert_eq!(
            update.unavailable(),
            None,
            "a committed update is available"
        );

        let mut update = Update::new(2, Deployment::of(ServeConfig::default()));
        let first = Failure {
            kind: "illegal_address".to_owned(),
            reason: "an illegal memory access was encountered".to_owned(),
            device: Some(3),
        };
        update.fail(first.clone());
        update.fail(Failure {
            kind: "stalled".to_owned(),
            reason: "the forward job of batch 7 ran 400 ms".to_owned(),
            device: None,
        });
        assert_eq!(update.first_failure.as_ref(), Some(&first));
        assert_eq!(update.unavailable(), None, "the probation has not ended");

        update.revert();
        assert_eq!(update.state, UpdateState::Reverted);
        assert_eq!(update.unavailable(), Some(&first));
    }

    /// The reports that fail an update become its failure, with the kind
    /// the main process named; the acknowledgements do not.
    #[test]
    fn the_main_processs_reports_become_the_updates_failure() {
        use proto::process::report::Kind;

        let fault = proto::process::Report {
            kind: Some(Kind::Fault(proto::process::Fault {
                kind: "launch_failed".to_owned(),
                reason: "too many resources requested".to_owned(),
                device: 2,
            })),
        };
        assert_eq!(
            Failure::of_report(&fault),
            Some(Failure {
                kind: "launch_failed".to_owned(),
                reason: "too many resources requested".to_owned(),
                device: Some(2),
            })
        );

        let stalled = proto::process::Report {
            kind: Some(Kind::Stalled(proto::process::Stalled {
                generation: 4,
                batch: "batch-9".to_owned(),
                job: "copy".to_owned(),
                elapsed_ms: 400,
            })),
        };
        assert_eq!(
            Failure::of_report(&stalled).map(|failure| failure.kind),
            Some("stalled".to_owned())
        );

        // The policy's failure names the instance, the backend and the
        // revision, and whether no previous module is left (§4.4).
        let reverted = proto::process::Report {
            kind: Some(Kind::PolicyReverted(proto::process::PolicyReverted {
                scheduler: Some(2),
                backend: "trtllm".to_owned(),
                revision: 7,
                reason: "the fuel budget is exhausted".to_owned(),
                fatal: false,
            })),
        };
        let failure = Failure::of_report(&reverted).expect("a policy revert fails the update");
        assert_eq!(failure.kind, "policy_reverted");
        assert!(failure.reason.contains("scheduler 2"), "{}", failure.reason);
        assert!(failure.reason.contains("trtllm"), "{}", failure.reason);
        assert!(failure.reason.contains("revision 7"), "{}", failure.reason);
        assert!(
            failure.reason.contains("the fuel budget is exhausted"),
            "{}",
            failure.reason
        );
        let route = proto::process::Report {
            kind: Some(Kind::PolicyReverted(proto::process::PolicyReverted {
                scheduler: None,
                backend: "fcfs".to_owned(),
                revision: 1,
                reason: "a decision was refused".to_owned(),
                fatal: true,
            })),
        };
        let failure = Failure::of_report(&route).expect("a fatal policy revert fails the update");
        assert!(
            failure.reason.contains("the engine's control thread"),
            "{}",
            failure.reason
        );
        assert!(
            failure.reason.contains("no previous policy module is left"),
            "{}",
            failure.reason
        );

        for acknowledged in [
            proto::process::Report {
                kind: Some(Kind::Precopied(proto::process::Precopied {
                    generation: 4,
                    uncompared: Vec::new(),
                })),
            },
            proto::process::Report {
                kind: Some(Kind::Switched(proto::process::Switched { generation: 4 })),
            },
            proto::process::Report {
                kind: Some(Kind::Committed(proto::process::Committed {})),
            },
            proto::process::Report {
                kind: Some(Kind::Purged(proto::process::Purged { generation: 4 })),
            },
            // A failed precopy drops the new generation and leaves the
            // serving one alone: it fails no update (§7.3 step 2).
            proto::process::Report {
                kind: Some(Kind::PrecopyFailed(proto::process::PrecopyFailed {
                    reason: "the package has no module for this descriptor".to_owned(),
                })),
            },
        ] {
            assert_eq!(Failure::of_report(&acknowledged), None);
        }
    }

    /// Programs are recorded per operator and, where the configuration fills
    /// one per phase, per phase; the epoch that serves is the one of the
    /// highest number that is serving or on probation.
    #[test]
    fn epochs_are_reported_by_operator_phase_and_status() {
        let epochs = Epochs::new();
        let mut first = record();
        first.status = Status::Probation;
        epochs.record(1, first);
        let mut second = record();
        second.status = Status::Precopy;
        second.kernels.insert(
            KernelKey::phase(operators::PAGED_ATTENTION, PhaseKind::Decode),
            program("trtllm"),
        );
        epochs.record(2, second);
        assert_eq!(
            epochs
                .serving()
                .map(|(epoch, record)| (epoch, record.status)),
            Some((1, Status::Probation))
        );

        // The epochs are one array, ascending, each naming its number.
        let report = epochs.report();
        let entries = report["epochs"].as_array().expect("epochs is an array");
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry["epoch"].as_u64())
                .collect::<Vec<_>>(),
            vec![Some(1), Some(2)]
        );
        assert_eq!(entries[0]["status"], Value::from("probation"));
        assert_eq!(entries[1]["status"], Value::from("precopy"));
        assert_eq!(entries[1]["kernels"][0]["phase"], Value::from("decode"));
        assert_eq!(
            entries[1]["kernels"][0]["program"]["backend"],
            Value::from("trtllm")
        );
    }

    /// A launch generation is allocated in one place: one more than the
    /// highest generation the daemon knows for that group and rank, which is
    /// the serving epoch's (§2.5, §7.4), so a standby of a serving epoch is
    /// `serving + 1`, and two callers never get one number.
    #[test]
    fn a_launch_generation_is_one_more_than_the_serving_epochs() {
        let epochs = Epochs::new();
        assert_eq!(epochs.next_generation("engine", 0), 1, "the first launch");

        // A serving process's launch generation is the base: its standby is
        // one more, and so is the next allocation after that.
        epochs.set_processes(vec![ManagedProcess {
            role: Role::MainProcess,
            group: "engine".to_owned(),
            rank: 0,
            pid: Some(42),
            epoch: Some(1),
            build: "2.0.0+0123456789abcdef".to_owned(),
            generation: 6,
            state: ProcessState::Serving,
        }]);
        assert_eq!(
            epochs.next_generation("engine", 0),
            7,
            "the standby of a serving process is serving + 1"
        );
        assert_eq!(
            epochs.next_generation("engine", 0),
            8,
            "an allocation is one more than the last one"
        );
        assert_eq!(
            epochs.next_generation("engine", 1),
            7,
            "a rank allocates for itself"
        );
        assert_eq!(
            epochs.next_generation("worker", 0),
            7,
            "a group allocates for itself"
        );
    }

    /// The process table names each process's role, epoch and state.
    #[test]
    fn the_process_table_reports_what_the_daemon_manages() {
        let epochs = Epochs::new();
        epochs.set_processes(vec![
            ManagedProcess {
                role: Role::Frontend,
                group: "frontend".to_owned(),
                rank: 0,
                pid: Some(41),
                epoch: None,
                build: "2.0.0+0123456789abcdef".to_owned(),
                generation: 0,
                state: ProcessState::Serving,
            },
            ManagedProcess {
                role: Role::MainProcess,
                group: "engine".to_owned(),
                rank: 0,
                pid: Some(42),
                epoch: Some(7),
                build: "2.0.0+0123456789abcdef".to_owned(),
                generation: 4,
                state: ProcessState::Parked,
            },
        ]);
        let report = epochs.report();
        assert_eq!(report["processes"][0]["role"], Value::from("frontend"));
        assert_eq!(report["processes"][1]["role"], Value::from("main_process"));
        assert_eq!(report["processes"][1]["epoch"], Value::from(7));
        assert_eq!(report["processes"][1]["state"], Value::from("parked"));
    }

    /// The epoch's build config is written by the daemon and read by the main
    /// process through this same type, so what the writer serializes must read
    /// back: an unset list is written as `null` and must be read as `None`,
    /// not refused.
    #[test]
    fn a_build_config_reads_back_what_it_wrote() {
        let config = ServeConfig::default();
        let build = BuildConfig::of(&config);
        assert!(build.cuda_graph.flow_shapes.is_none());
        let text = serde_json::to_string(&build).expect("a build config is JSON");
        let read: BuildConfig =
            serde_json::from_str(&text).expect("the build config reads back through its writer");
        assert_eq!(read, build);
    }
}
