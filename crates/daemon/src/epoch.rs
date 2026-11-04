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

use proto::request::ModelDtype;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::options::{
    CudaGraphOptions, KvCacheOptions, LoadOptions, ServeConfig, VideoOptions, WorkerOptions,
};

/// A program the epoch fills an operator or the scheduling interface with:
/// the backend's name, the revision of its package the upstream Program
/// Server numbered, and its parameters (§5.1).
///
/// The Rust type is `crates/program-server`'s `Program`; the programs swarm
/// owns it, and the daemon uses its wire form until that crate lands. The
/// wire form is the same three values, so nothing but the type name changes
/// then.
pub type Program = proto::process::Program;

/// The phase of an operator's calls a kernel program fills. An operator is
/// filled once for all its phases unless the configuration fills it per
/// phase, which attention allows (§5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PhaseKind {
    /// Prompt processing.
    Prefill,
    /// One token of a request in progress.
    Decode,
}

impl PhaseKind {
    /// The wire form of this phase.
    pub fn to_wire(self) -> proto::process::PhaseKind {
        match self {
            Self::Prefill => proto::process::PhaseKind::Prefill,
            Self::Decode => proto::process::PhaseKind::Decode,
        }
    }

    /// The phase `wire` names; `None` for the unspecified value, which a
    /// `KernelProgram` leaves absent for an operator filled for every face.
    pub fn from_wire(wire: proto::process::PhaseKind) -> Option<Self> {
        match wire {
            proto::process::PhaseKind::Prefill => Some(Self::Prefill),
            proto::process::PhaseKind::Decode => Some(Self::Decode),
            proto::process::PhaseKind::Unspecified => None,
        }
    }
}

/// What one operator is filled with, or which phases of it.
pub type KernelKey = (String, Option<PhaseKind>);

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
    /// them: the records of §5.1, with the programs in the wire form the
    /// daemon records, and one entry per process the daemon manages.
    pub fn report(&self) -> Value {
        let epochs = locked(&self.records)
            .iter()
            .map(|(epoch, record)| (epoch.to_string(), record_report(record)))
            .collect::<Map<String, Value>>();
        let processes = locked(&self.processes)
            .iter()
            .map(process_report)
            .collect::<Vec<_>>();
        serde_json::json!({ "epochs": epochs, "processes": processes })
    }
}

/// One epoch record as JSON: the programs by operator and phase, the
/// previous generation, and the status, as the design names them.
fn record_report(record: &EpochRecord) -> Value {
    let kernels = record
        .kernels
        .iter()
        .map(|((op, phase), program)| {
            serde_json::json!({
                "op": op,
                "phase": phase.map(|phase| match phase {
                    PhaseKind::Prefill => "prefill",
                    PhaseKind::Decode => "decode",
                }),
                "program": program_report(program),
            })
        })
        .collect::<Vec<_>>();
    let previous = record
        .previous_kernels
        .iter()
        .map(|((op, phase), program)| {
            serde_json::json!({
                "op": op,
                "phase": phase.map(|phase| match phase {
                    PhaseKind::Prefill => "prefill",
                    PhaseKind::Decode => "decode",
                }),
                "program": program_report(program),
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
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
            arguments: std::collections::HashMap::new(),
        }
    }

    /// A serving epoch's record at the default configuration.
    fn record() -> EpochRecord {
        let config = ServeConfig::default();
        EpochRecord {
            build_config: BuildConfig::of(&config),
            kernels: BTreeMap::from([(("rms_norm".to_owned(), None), program("triton"))]),
            policy: program("fcfs"),
            previous_kernels: BTreeMap::new(),
            previous_policy: None,
            status: Status::Serving,
        }
    }

    /// A change of the values that only inform a decision, or that select a
    /// program's backend, leaves the build config equal: the update is the
    /// program layer's. A change of anything that builds the main process
    /// makes it differ, and the update is the epoch layer's (§7.2, §7.7).
    #[test]
    fn the_build_config_differs_only_on_what_builds_the_main_process() {
        let base = BuildConfig::of(&ServeConfig::default());

        // The scheduling program's backend and parameters, the attention
        // and expert-exchange backends, and the refill policy: program layer.
        let mut program_layer = ServeConfig::default();
        program_layer.scheduler_config = SchedulerOptions {
            policy: crate::options::SchedulingPolicy::Priority,
            max_queued_requests: NonZeroUsize::new(16),
            chunked_prefill_size: NonZeroUsize::new(2048),
            ..SchedulerOptions::default()
        };
        program_layer.attention_backend = "flashinfer".to_owned();
        program_layer.moe_config.exchange = crate::options::ExpertExchange::DeepEp;
        program_layer.speculative_config = SpeculativeOptions {
            expected_speedup: Some(2.5),
            refill_token_cost: Some(0.5),
            ..SpeculativeOptions::default()
        };
        assert_eq!(BuildConfig::of(&program_layer), base);

        // The workers, the slots, the KV layout, the graph buckets, the
        // draft tokens and the probed components: epoch layer.
        for changed in [
            {
                let mut config = ServeConfig::default();
                config.max_running_requests = NonZeroUsize::new(4).unwrap();
                config
            },
            {
                let mut config = ServeConfig::default();
                config.max_model_len = Some(NonZeroU32::new(2048).unwrap());
                config
            },
            {
                let mut config = ServeConfig::default();
                config.mem_fraction_static = 0.5;
                config
            },
            {
                let mut config = ServeConfig::default();
                config.speculative_config.num_draft_tokens = 4;
                config
            },
            {
                let mut config = ServeConfig::default();
                config.moe_config.microbatches = NonZeroU32::new(2).unwrap();
                config
            },
            {
                let mut config = ServeConfig::default();
                config.profiling_config.graph_pc_sampling = true;
                config
            },
        ] {
            assert_ne!(
                BuildConfig::of(&changed),
                base,
                "a change of the values that build the main process makes a new epoch"
            );
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
            ("paged_attention".to_owned(), Some(PhaseKind::Decode)),
            program("trtllm"),
        );
        epochs.record(2, second);
        assert_eq!(
            epochs.serving().map(|(epoch, record)| (epoch, record.status)),
            Some((1, Status::Probation))
        );

        let report = epochs.report();
        assert_eq!(report["epochs"]["1"]["status"], Value::from("probation"));
        assert_eq!(report["epochs"]["2"]["status"], Value::from("precopy"));
        assert_eq!(
            report["epochs"]["2"]["kernels"][0]["phase"],
            Value::from("decode")
        );
        assert_eq!(
            report["epochs"]["2"]["kernels"][0]["program"]["backend"],
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
