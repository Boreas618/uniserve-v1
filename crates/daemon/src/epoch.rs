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

use std::collections::{BTreeMap, HashMap};
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::PathBuf;
use std::sync::Mutex;

use proto::request::ModelDtype;
use program_server::{PhaseKind, Program, ProgramSlot};
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
                let value = value.parse::<toml::Value>().unwrap_or(toml::Value::String(value));
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
    /// Its process id on its node, once started.
    pub pid: Option<u32>,
    /// The epoch a main process belongs to; `None` for the frontend.
    pub epoch: Option<u64>,
    /// The build identity it was started from.
    pub build: String,
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
            Kind::PolicyReverted(_) => Some(Self {
                kind: "policy_reverted".to_owned(),
                reason: "the scheduling policy failed and the previous one serves".to_owned(),
                device: None,
            }),
            Kind::Precopied(_)
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

/// The epochs a deployment's daemon keeps and the processes it manages on
/// its node: the state `GET /admin/config` reports and a change is routed
/// by (§7.2).
#[derive(Debug, Default)]
pub struct Epochs {
    records: Mutex<BTreeMap<u64, EpochRecord>>,
    processes: Mutex<Vec<ManagedProcess>>,
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
        *locked(&self.processes) = processes;
    }

    /// The node's process table.
    pub fn processes(&self) -> Vec<ManagedProcess> {
        locked(&self.processes).clone()
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
        let processes = locked(&self.processes)
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
        "build": process.build,
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
    mutex.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
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
        kernels.insert(
            KernelKey::all(operators::EXPERT_EXCHANGE),
            Program {
                backend: word(&config.moe_config.exchange),
                revision: 0,
                arguments: toml::Table::new(),
            },
        );
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
}

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
    let arguments = Arguments {
        chunked_prefill_size: scheduler
            .chunked_prefill_size
            .map(|size| u32::try_from(size.get()).unwrap_or(u32::MAX)),
        max_queued_requests: scheduler
            .max_queued_requests
            .map(|size| u32::try_from(size.get()).unwrap_or(u32::MAX)),
        max_batch: u32::try_from(scheduler.max_batch.get()).unwrap_or(u32::MAX),
        max_num_batched_tokens: config
            .max_num_batched_tokens
            .map(|size| u32::try_from(size.get()).unwrap_or(u32::MAX)),
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
    use crate::options::{SchedulerOptions, SpeculativeOptions};

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
        }
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
            ("speculative_config.num_draft_tokens", |config: &mut ServeConfig| {
                config.speculative_config.num_draft_tokens = 4
            }),
            ("moe_config.microbatches", |config: &mut ServeConfig| {
                config.moe_config.microbatches = NonZeroU32::new(2).unwrap()
            }),
            ("profiling_config.graph_pc_sampling", |config: &mut ServeConfig| {
                config.profiling_config.graph_pc_sampling = true
            }),
        ];
        // The program backends (§7.7 backend row) and the program parameters
        // (§7.7 parameter row).
        let program: Vec<(&str, fn(&mut ServeConfig))> = vec![
            ("scheduler_config.policy", |config: &mut ServeConfig| {
                config.scheduler_config.policy = proto::request::SchedulingPolicy::Priority
            }),
            ("scheduler_config.chunked_prefill_size", |config: &mut ServeConfig| {
                config.scheduler_config.chunked_prefill_size = NonZeroUsize::new(2048)
            }),
            ("scheduler_config.max_batch", |config: &mut ServeConfig| {
                config.scheduler_config.max_batch = NonZeroUsize::new(8).unwrap()
            }),
            ("max_num_batched_tokens", |config: &mut ServeConfig| {
                config.max_num_batched_tokens = NonZeroUsize::new(4096)
            }),
            ("speculative_config.expected_speedup", |config: &mut ServeConfig| {
                config.speculative_config.expected_speedup = Some(2.5)
            }),
            ("speculative_config.refill_token_cost", |config: &mut ServeConfig| {
                config.speculative_config.refill_token_cost = Some(0.5)
            }),
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
            ("flashinfer_config.workspace_size", |config: &mut ServeConfig| {
                config.flashinfer_config.workspace_size = 1 << 30
            }),
        ];

        let base = Deployment::of(ServeConfig::default());
        assert_eq!(layer(&base, &base), Layer::None, "no change");

        for (what, change) in host_frontend.iter().chain(host_daemon.iter()) {
            let next = Deployment::of(set(*change));
            assert_eq!(layer(&base, &next), Layer::Host, "{what} reads the host layer");
        }
        for (what, change) in &epoch {
            let next = Deployment::of(set(*change));
            assert_eq!(layer(&base, &next), Layer::Epoch, "{what} builds a new epoch");
        }
        for (what, change) in &program {
            let next = Deployment::of(set(*change));
            assert_eq!(layer(&base, &next), Layer::Program, "{what} moves the programs");
        }

        // Only the epoch layer's values build the main process: the host
        // layer's and the program layer's leave the build config equal
        // (§7.7).
        for (what, change) in host_frontend.iter().chain(host_daemon.iter()).chain(program.iter()) {
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

        // One attention program for every phase, the expert exchange's
        // backend, and no grouped expert GEMM until the configuration names
        // one: prepare resolves it (§5.1).
        let attention = &programs.kernels[&KernelKey::all(operators::PAGED_ATTENTION)];
        assert_eq!(attention.backend, "auto");
        assert_eq!(attention.revision, 0, "no revision before prepare");
        assert_eq!(
            programs.kernels[&KernelKey::all(operators::EXPERT_EXCHANGE)].backend,
            "alltoall"
        );
        for operator in [operators::GATHER_GEMM, operators::ROUTE_GEMM] {
            assert!(
                !programs.kernels.contains_key(&KernelKey::all(operator)),
                "an operator the configuration does not fill is absent"
            );
        }

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
        assert_eq!((prefill.backend.as_str(), decode.backend.as_str()), ("fa4", "trtllm"));
        assert!(
            !programs
                .kernels
                .contains_key(&KernelKey::all(operators::PAGED_ATTENTION)),
            "a per-phase fill covers no whole operator"
        );

        // One stated backend for both phases fills the operator once.
        let mut one = ServeConfig::default();
        one.prefill_attention_backend = Some("fa4".to_owned());
        let programs = FilledPrograms::of(&one);
        assert_eq!(
            programs.kernels[&KernelKey::all(operators::PAGED_ATTENTION)].backend,
            "fa4"
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

    /// The update keeps the first failure it meets, not the last, and a
    /// revert marks it unavailable with that first error (§7.3 step 4): the
    /// daemon reports the reason with the update and refuses it without
    /// `force` afterwards.
    #[test]
    fn an_update_keeps_the_first_failure_and_is_unavailable_with_it() {
        let mut update = Update::new(1, Deployment::of(ServeConfig::default()));
        update.commit();
        assert_eq!(update.unavailable(), None, "a committed update is available");

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

        for acknowledged in [
            proto::process::Report {
                kind: Some(Kind::Precopied(proto::process::Precopied { generation: 4 })),
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
            epochs.serving().map(|(epoch, record)| (epoch, record.status)),
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

    /// The process table names each process's role, epoch and state.
    #[test]
    fn the_process_table_reports_what_the_daemon_manages() {
        let epochs = Epochs::new();
        epochs.set_processes(vec![
            ManagedProcess {
                role: Role::Frontend,
                pid: Some(41),
                epoch: None,
                build: "2.0.0+0123456789abcdef".to_owned(),
                state: ProcessState::Serving,
            },
            ManagedProcess {
                role: Role::MainProcess,
                pid: Some(42),
                epoch: Some(7),
                build: "2.0.0+0123456789abcdef".to_owned(),
                state: ProcessState::Parked,
            },
        ]);
        let report = epochs.report();
        assert_eq!(report["processes"][0]["role"], Value::from("frontend"));
        assert_eq!(report["processes"][1]["role"], Value::from("main_process"));
        assert_eq!(report["processes"][1]["epoch"], Value::from(7));
        assert_eq!(report["processes"][1]["state"], Value::from("parked"));
    }
}
