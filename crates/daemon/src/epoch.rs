//! Epochs: what a deployment executes and which modules it runs.
//!
//! An epoch is one engine build with its build configuration, executed: its
//! engines on every node, their graphs and the storage they create. The
//! leader's daemon keeps one [`EpochRecord`] per epoch: the
//! [`proto::config::EngineConfig`] it was built from, the module set it runs,
//! the modules a trial keeps as the previous generation, and its [`EpochState`].
//!
//! A configuration change is routed by comparing what it moves ([`change`]):
//! with the build configuration unchanged and only the kernel or policy
//! modules different, the serving engines take them as a generation; with a
//! different build configuration or engine package, a new epoch is started
//! beside the serving one; a change of the daemon's own options is a hot
//! restart. Nothing about a change is declared by the change itself. The
//! update that moves the deployment is `crate::updates::update::Update`; this
//! module holds the records it writes.

use std::collections::BTreeMap;
use std::sync::Mutex;

use proto::admin::{EpochState, EpochStatus, KernelPackage, ManagedProcess};
pub use proto::module::{ModuleSet, ModuleSlot, PackageRevision, Phase};
// The operators the configuration selects a backend for are named by the
// module vocabulary (`proto::module::operator`), whose names each
// operator's `Op::NAME` is: the daemon, the Build Server and the engine
// cannot disagree about a slot's name.
use proto::module::operator;
use serde::Serialize;

use crate::options::ServerArgs;

/// The key an epoch's modules are stored under: the operator, and the
/// phase when the configuration fills it per phase. Modules and
/// slots are `proto::module`'s types, which the daemon, the engine
/// and the Build Server all name modules with.
pub type KernelKey = ModuleSlot;

/// What the leader's daemon records of one epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochRecord {
    /// What built the epoch.
    pub config: proto::config::EngineConfig,
    /// The engine package the epoch's engines run as, at the revision
    /// fixed when the epoch's configuration was applied.
    pub engine: PackageRevision,
    /// The kernel package that fills each operator, or each phase of one;
    /// a phase of `None` covers every phase of its operator.
    pub kernels: BTreeMap<KernelKey, PackageRevision>,
    /// The scheduling policy package.
    pub policy: PackageRevision,
    /// The modules a generation's trial keeps as the previous generation: a
    /// rollback returns to them, and a commit drops them.
    pub previous: Option<Generation>,
    /// Where the epoch is in its life.
    pub status: EpochState,
    /// The generation: the number the epoch's own engines gave the modules
    /// they execute, read from their `Parked` or `Serving` report and never
    /// invented here. It is not a process's identity: that is
    /// [`ManagedProcess::generation`], the launch generation, allocated by
    /// the deployment's engine control (`EngineControl::launch_number`).
    pub generation: Option<u64>,
}

/// The modules one generation of an epoch executes, which a generation update's trial keeps for
/// a rollback.
#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    /// The kernel package that fills each operator, or each phase of one.
    pub kernels: BTreeMap<KernelKey, PackageRevision>,
    /// The scheduling policy package.
    pub policy: PackageRevision,
}

/// An epoch the daemon keeps a record of. Records are never removed, so an operation on a
/// `Recorded` epoch's record always finds it: only [`Epochs`] makes one, when it records the
/// epoch or reads its record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Recorded(u64);

impl Recorded {
    /// The epoch's number.
    pub fn number(self) -> u64 {
        self.0
    }
}

/// The epochs a deployment's daemon keeps and the processes it manages on
/// its node: the state `GET /admin/config` reports and a change is routed
/// by.
#[derive(Debug, Default)]
pub struct Epochs {
    records: Mutex<BTreeMap<u64, EpochRecord>>,
    /// The node's process table, which the daemon that supervises the node's processes keeps
    /// current as they start and end; these records add what each launch is.
    processes: ProcessTable,
    /// The epoch requests were last moved to: the request path of this daemon follows it by
    /// moving its hold onto that epoch's engine.
    served: tokio::sync::watch::Sender<u64>,
}

/// A node's process table: the daemon's supervision writes it, and the epoch records report it.
pub type ProcessTable = std::sync::Arc<Mutex<Vec<ManagedProcess>>>;

impl Epochs {
    /// No epoch built yet, over the node's process table `processes`.
    pub fn new(processes: ProcessTable) -> Self {
        Self {
            processes,
            ..Self::default()
        }
    }

    /// States that `epoch` serves requests from now on.
    pub fn announce_serve(&self, epoch: u64) {
        self.served.send_replace(epoch);
    }

    /// The epoch that was last stated to serve, as it changes.
    pub fn served(&self) -> tokio::sync::watch::Receiver<u64> {
        self.served.subscribe()
    }

    /// Records `record` as epoch `epoch`'s, replacing what was recorded of
    /// that epoch before.
    pub fn record(&self, epoch: u64, record: EpochRecord) -> Recorded {
        service::lock(&self.records).insert(epoch, record);
        Recorded(epoch)
    }

    /// Epoch `epoch` as recorded, if the daemon keeps its record.
    pub fn recorded(&self, epoch: u64) -> Option<Recorded> {
        service::lock(&self.records)
            .contains_key(&epoch)
            .then_some(Recorded(epoch))
    }

    /// Runs `change` on the record of `epoch`, which is kept since it is recorded.
    #[expect(
        clippy::expect_used,
        reason = "a `Recorded` epoch's record exists: only `Epochs` makes one, and no record is \
                  ever removed"
    )]
    fn change_record(&self, epoch: Recorded, change: impl FnOnce(&mut EpochRecord)) {
        let mut records = service::lock(&self.records);
        let record = records
            .get_mut(&epoch.0)
            .expect("a recorded epoch's record is never removed");
        change(record);
    }

    /// Records that epoch `epoch` switched to `modules` on trial, keeping
    /// what it executed before as the previous generation.
    pub fn switched(&self, epoch: Recorded, modules: ModuleSet) {
        self.change_record(epoch, |record| {
            record.previous = Some(Generation {
                kernels: std::mem::replace(&mut record.kernels, modules.kernels),
                policy: std::mem::replace(&mut record.policy, modules.policy),
            });
            if record.status == EpochState::Serving {
                record.status = EpochState::Trial;
            }
        });
    }

    /// Records that epoch `epoch`'s new modules are what serves and the
    /// previous generation is no longer kept.
    pub fn committed(&self, epoch: Recorded) {
        self.change_record(epoch, |record| {
            record.previous = None;
            record.status = EpochState::Serving;
        });
    }

    /// Ends a generation's trial by putting the previous generation back:
    /// what the epoch executed before the update is what it executes again.
    /// An epoch that kept none executes what it executed.
    pub fn rolled_back(&self, epoch: Recorded) {
        self.change_record(epoch, |record| {
            if let Some(previous) = record.previous.take() {
                record.kernels = previous.kernels;
                record.policy = previous.policy;
            }
            record.status = EpochState::Serving;
        });
    }

    /// Records epoch `epoch` as serving the `deployment` this node started:
    /// its build configuration, the module set its launch files stated and
    /// the generation its engines numbered for it. The epoch is recorded
    /// under the number its launch gave it, as an update's target and an
    /// adopted epoch are.
    pub fn install(&self, epoch: u64, deployment: &Epoch, generations: &[u64]) -> Recorded {
        self.record(
            epoch,
            EpochRecord {
                config: deployment.engine.clone(),
                engine: deployment.modules.engine.clone(),
                kernels: deployment.modules.kernels.clone(),
                policy: deployment.modules.policy.clone(),
                previous: None,
                status: EpochState::Serving,
                generation: generations.first().copied(),
            },
        )
    }

    /// Epoch `epoch`'s record, if the daemon keeps one.
    pub fn get(&self, epoch: u64) -> Option<EpochRecord> {
        service::lock(&self.records).get(&epoch).cloned()
    }

    /// Every epoch the daemon keeps, by number.
    pub fn records(&self) -> BTreeMap<u64, EpochRecord> {
        service::lock(&self.records).clone()
    }

    /// The epoch that serves requests: the highest numbered one that is
    /// serving or on trial.
    pub fn serving(&self) -> Option<(Recorded, EpochRecord)> {
        service::lock(&self.records)
            .iter()
            .rev()
            .find(|(_, record)| matches!(record.status, EpochState::Serving | EpochState::Trial))
            .map(|(epoch, record)| (Recorded(*epoch), record.clone()))
    }

    /// Records `adopted`, a process a hot restart's successor took over: the
    /// row its daemon's service already lists for the same launch (group, rank
    /// and launch generation) takes the epoch, build and state it serves under
    /// and keeps the pid the service read; a row is added when none lists it.
    pub fn adopt_process(&self, adopted: ManagedProcess) {
        let mut processes = self.processes();
        let listed = processes.iter_mut().find(|row| {
            row.group == adopted.group
                && row.rank == adopted.rank
                && row.generation == adopted.generation
        });
        match listed {
            Some(row) => {
                row.epoch = adopted.epoch;
                row.build = adopted.build;
                row.state = adopted.state;
            }
            None => processes.push(adopted),
        }
        self.set_processes(processes);
    }

    /// Replaces the node's process table with `processes`.
    pub fn set_processes(&self, processes: Vec<ManagedProcess>) {
        *service::lock(&self.processes) = processes;
    }

    /// The node's process table.
    pub fn processes(&self) -> Vec<ManagedProcess> {
        service::lock(&self.processes).clone()
    }

    /// Every epoch's record as `GET /admin/config` reports it (`epochs`), ascending by number.
    pub fn statuses(&self) -> Vec<EpochStatus> {
        service::lock(&self.records)
            .iter()
            .map(|(epoch, record)| EpochStatus {
                epoch: *epoch,
                engine_config: record.config.clone(),
                engine: record.engine.clone(),
                kernels: kernel_packages(&record.kernels),
                policy: record.policy.clone(),
                previous_kernels: record
                    .previous
                    .as_ref()
                    .map(|previous| kernel_packages(&previous.kernels))
                    .unwrap_or_default(),
                previous_policy: record
                    .previous
                    .as_ref()
                    .map(|previous| previous.policy.clone()),
                generation: record.generation,
                status: record.status,
            })
            .collect()
    }
}

/// The kernel packages of `kernels` by slot, in the slots' order, as a report states them.
pub fn kernel_packages(kernels: &BTreeMap<KernelKey, PackageRevision>) -> Vec<KernelPackage> {
    kernels
        .iter()
        .map(|(slot, package)| KernelPackage {
            slot: slot.clone(),
            package: package.clone(),
        })
        .collect()
}

/// The module set `config` names: the engine package, the scheduling policy's
/// and the kernel package of each operator, or phase of one, it selects a
/// backend for.
///
/// The backends and parameters come from the configuration; the revisions
/// are fixed when the configuration is applied, at the Build Server.
/// A package's `revision` is zero until then, unless the
/// configuration states one. An operator whose backend the configuration
/// does not state is absent: the deployment names
/// nothing for it, and the Build Server selects its package for each of its
/// descriptor keys among the packages that cover them, by manifest priority.
pub fn modules_of(config: &ServerArgs) -> ModuleSet {
    let mut kernels = BTreeMap::new();
    let attention = attention_packages(config);
    kernels.extend(attention);
    // The exchange slot is filled only when the deployment spreads its
    // experts over ranks. At one rank the model's MoE block takes its local
    // arm and launches no exchange at all (`nn::Moe::forward` routes
    // through `forward_expert_parallel` only when an expert lives on
    // another rank), so a slot filled here would pin a package no call
    // reaches — and, worse, the Build Server would refuse the epoch for a
    // package nothing runs.
    if config.moe_config.expert_parallel {
        kernels.insert(
            KernelKey::all(operator::EXPERT_EXCHANGE),
            PackageRevision {
                backend: config
                    .moe_config
                    .exchange
                    .clone()
                    .unwrap_or_else(|| proto::config::DEFAULT_EXPERT_EXCHANGE.to_owned()),
                revision: 0,
                arguments: toml::Table::new(),
            },
        );
    }
    if let Some(backend) = &config.moe_runner_backend {
        // The grouped expert GEMM is two operators, gather and route,
        // and one package backend fills both.
        for operator in [operator::GATHER_GEMM, operator::ROUTE_GEMM] {
            kernels.insert(
                KernelKey::all(operator),
                PackageRevision {
                    backend: backend.clone(),
                    revision: 0,
                    arguments: toml::Table::new(),
                },
            );
        }
    }
    ModuleSet {
        engine: PackageRevision {
            backend: config.worker_config.engine.clone(),
            revision: config.worker_config.engine_revision.unwrap_or(0),
            arguments: toml::Table::new(),
        },
        kernels,
        policy: PackageRevision {
            backend: config.scheduler_config.policy.clone(),
            revision: 0,
            arguments: policy_arguments(config),
        },
    }
}

/// The attention operator's modules: one entry for every phase when the
/// per-phase backends agree, or one entry per phase when they differ.
/// A phase whose backend is unset names no package: the Build Server selects its
/// keys' packages.
fn attention_packages(config: &ServerArgs) -> Vec<(KernelKey, PackageRevision)> {
    let prefill = config
        .prefill_attention_backend
        .clone()
        .or_else(|| config.attention_backend.clone());
    let decode = config
        .decode_attention_backend
        .clone()
        .or_else(|| config.attention_backend.clone());
    // No attention package declares parameters: its arguments are none.
    let package = |backend: String| PackageRevision {
        backend,
        revision: 0,
        arguments: toml::Table::new(),
    };
    if prefill == decode {
        return prefill
            .map(|backend| (KernelKey::all(operator::PAGED_ATTENTION), package(backend)))
            .into_iter()
            .collect();
    }
    [(Phase::Prefill, prefill), (Phase::Decode, decode)]
        .into_iter()
        .filter_map(|(phase, backend)| Some((phase, backend?)))
        .map(|(phase, backend)| {
            (
                KernelKey::phase(operator::PAGED_ATTENTION, phase),
                package(backend),
            )
        })
        .collect()
}

/// The scheduling policy package's parameters: the configuration's values the
/// scheduling policy decides with, inside the build config's bounds.
fn policy_arguments(config: &ServerArgs) -> toml::Table {
    #[derive(Serialize)]
    struct Arguments {
        #[serde(skip_serializing_if = "Option::is_none")]
        order: Option<proto::config::QueueOrder>,
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
    // `max_num_batched_tokens` its engine configuration states: the policy's
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
    // refused at load (`ServerArgs::validate`), never clamped here.
    let token_bound = config
        .max_num_batched_tokens
        .map(|size| u32::try_from(size.get()).unwrap_or(u32::MAX))
        .or_else(|| config.cuda_graph_config.tiers().tokens)
        // A media deployment captures no token graph, and its prompt is the most one batch holds.
        .or_else(|| config.max_model_len.map(|length| length.get()));
    // A batch of requests is bounded by the largest captured decode graph, and
    // the deployment's batch is the smaller of its own and that: a deployment
    // that leaves the batch at its default means the catalog's figure, and one
    // that states more than its graphs capture is refused at load
    // (`ServerArgs::validate`), never clamped here.
    let max_batch = u32::try_from(scheduler.max_batch.get()).unwrap_or(u32::MAX);
    let max_batch = config
        .cuda_graph_config
        .tiers()
        .rows
        .map_or(max_batch, |rows| max_batch.min(rows));
    // A batch holds running requests, so the deployment's bound on those bounds it as well: a media
    // deployment captures no decode graph, and its bound is the only one it states.
    let max_batch =
        max_batch.min(u32::try_from(config.max_running_requests.get()).unwrap_or(u32::MAX));
    let arguments = Arguments {
        order: scheduler.order,
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

/// `value`'s fields as the TOML table a package's arguments carry.
/// Values the field's serialization leaves out, such as an unset option, are
/// absent.
fn toml_arguments<T: Serialize>(value: &T) -> toml::Table {
    match toml::Value::try_from(value) {
        Ok(toml::Value::Table(table)) => table,
        _ => toml::Table::new(),
    }
}

/// What a configuration change moves, and so the least it needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Change {
    /// Nothing an engine executes changed.
    None,
    /// Only kernel or policy modules changed: the serving engines take them
    /// as a generation.
    Generation,
    /// The build configuration or the engine package changed: a new epoch
    /// starts beside the serving one.
    Epoch,
    /// The daemon's own options changed: the daemon is handed over to a
    /// successor, whose own routing applies what else the change states.
    Host,
}

/// A deployment as the daemon routes a change between: its configuration,
/// the build config it builds an engine from, and the modules it
/// fills.
#[derive(Debug, Clone, PartialEq)]
pub struct Epoch {
    /// The configuration.
    pub config: ServerArgs,
    /// What it builds the epoch's engines from, resolved once from `config`.
    pub engine: proto::config::EngineConfig,
    /// The filled modules; the deployment a change is compared against
    /// carries the revisions `prepare` selected, and the target's are
    /// prepared before the comparison.
    pub modules: ModuleSet,
}

impl Epoch {
    /// The deployment `config` describes, with the modules it fills and
    /// without the revisions `prepare` selects.
    ///
    /// # Errors
    ///
    /// Fails when the configuration does not resolve into an engine configuration
    /// ([`ServerArgs::engine_config`]): its deployment file does not read or place.
    pub fn of(config: ServerArgs) -> anyhow::Result<Self> {
        let modules = crate::epoch::modules_of(&config);
        Self::with_modules(config, modules)
    }

    /// The deployment `config` describes, with the modules `prepare` selected for it.
    ///
    /// # Errors
    ///
    /// Fails as [`Self::of`] does.
    pub fn with_modules(config: ServerArgs, modules: ModuleSet) -> anyhow::Result<Self> {
        Ok(Self {
            engine: config.engine_config()?,
            config,
            modules,
        })
    }
}

/// What `next` changes of `previous`, mechanically: the daemon's own options
/// first, then the build configuration or the engine package, then the
/// kernel and policy modules. Nothing about a change is declared by the
/// change itself.
pub fn change(previous: &Epoch, next: &Epoch) -> Change {
    if host_options_differ(&previous.config, &next.config) {
        Change::Host
    } else if previous.engine.build_part() != next.engine.build_part()
        || previous.modules.engine != next.modules.engine
    {
        // Another engine package, or another revision of it, is another
        // executable: its engines are started beside the serving ones.
        Change::Epoch
    } else if previous.modules != next.modules {
        Change::Generation
    } else {
        Change::None
    }
}

/// Whether `current` and `target` differ in the options the daemon reads
/// alone ([`crate::options::DaemonPart`]). A change of them is a hot restart
/// of the daemon.
pub fn host_options_differ(current: &ServerArgs, target: &ServerArgs) -> bool {
    current.daemon_part() != target.daemon_part()
}

#[cfg(test)]
mod tests {
    use std::num::{NonZeroU32, NonZeroUsize};

    use super::*;
    use crate::options::SchedulerConfig;

    /// A package of backend `backend` at revision 1 with no parameters.
    fn package(backend: &str) -> PackageRevision {
        PackageRevision {
            backend: backend.to_owned(),
            revision: 1,
            arguments: toml::Table::new(),
        }
    }

    /// A serving epoch's record at the default configuration.
    fn record() -> EpochRecord {
        let config = ServerArgs::default();
        EpochRecord {
            config: config.engine_config().expect("the configuration resolves"),
            engine: package("uniserve"),
            kernels: BTreeMap::from([(KernelKey::all("rms_norm"), package("base"))]),
            policy: package("fcfs"),
            previous: None,
            status: EpochState::Serving,
            generation: None,
        }
    }

    /// Another checkpoint's weights are other keys, so only a new epoch reads them: a change of
    /// the model's files that fell to a generation update would report an update that moved
    /// nothing.
    #[test]
    fn a_change_of_the_models_files_is_an_epoch_change() {
        let serving = Epoch::of(ServerArgs::default()).expect("the configuration resolves");
        let other = ServerArgs {
            model: Some("org/other-model".to_owned()),
            ..Default::default()
        };
        assert_eq!(
            change(
                &serving,
                &Epoch::of(other).expect("the configuration resolves")
            ),
            Change::Epoch
        );
    }

    /// The records report the node's process table as the daemon's supervision keeps it: every
    /// process it supervises, with the fields it wrote, and no row of a process that ended.
    #[test]
    fn the_process_table_reports_the_processes_the_daemon_supervises() {
        let table = ProcessTable::default();
        let epochs = Epochs::new(std::sync::Arc::clone(&table));
        assert!(epochs.processes().is_empty());
        *service::lock(&table) = vec![ManagedProcess {
            group: "engine".to_owned(),
            rank: 0,
            pid: Some(42),
            epoch: Some(0),
            build: "uniserve".to_owned(),
            generation: 4,
            state: proto::admin::ProcessState::Serving,
        }];
        assert_eq!(epochs.processes(), *service::lock(&table));
        service::lock(&table).clear();
        assert!(epochs.processes().is_empty());
    }

    /// A process a hot restart's successor adopted is one row: the row the
    /// successor's service lists for it takes the epoch it serves, with the
    /// pid the service read.
    #[test]
    fn an_adopted_process_is_one_row_of_the_table() {
        let table = std::sync::Arc::new(Mutex::new(vec![ManagedProcess {
            group: "engine".to_owned(),
            rank: 0,
            pid: Some(42),
            epoch: None,
            build: "uniserve".to_owned(),
            generation: 1,
            state: proto::admin::ProcessState::Serving,
        }]));
        let epochs = Epochs::new(std::sync::Arc::clone(&table));

        epochs.adopt_process(ManagedProcess {
            group: "engine".to_owned(),
            rank: 0,
            pid: None,
            epoch: Some(3),
            build: "engine".to_owned(),
            generation: 1,
            state: proto::admin::ProcessState::Serving,
        });
        let processes = epochs.processes();
        assert_eq!(processes.len(), 1, "{processes:?}");
        assert_eq!((processes[0].pid, processes[0].epoch), (Some(42), Some(3)));
    }

    /// The policy's declared limits are the launch's own bounds: the token
    /// bound the deployment gives its workers is the policy's
    /// `max_num_batched_tokens`, or the largest shape its captured graphs bound
    /// a batch to when it states no budget; an unstated prefill chunk is that
    /// same bound, so the policy's limits cannot disagree with the workers'
    /// `BatchCapacity`. A deployment that states a chunk larger than
    /// its bound keeps its own value here — the mechanism refuses it
    /// (`PolicyError::BadLimits`), and `ServerArgs::validate` refuses it at
    /// load by the option's name, rather than this code clamping it.
    #[test]
    fn the_policys_limits_are_the_launchs_own_token_bound() {
        let mut config = ServerArgs {
            max_num_batched_tokens: NonZeroUsize::new(8192),
            ..Default::default()
        };
        config.scheduler_config.chunked_prefill_size = None;
        let modules = crate::epoch::modules_of(&config);
        assert_eq!(
            modules.policy.arguments["max_num_batched_tokens"],
            toml::Value::Integer(8192)
        );
        assert_eq!(
            modules.policy.arguments["chunked_prefill_size"],
            toml::Value::Integer(8192),
            "an unstated chunk is the deployment's own token bound"
        );

        // A stated chunk above the bound is stated, not clamped: the
        // mechanism's `BadLimits` is the refusal.
        config.scheduler_config.chunked_prefill_size = NonZeroUsize::new(16384);
        let modules = crate::epoch::modules_of(&config);
        assert_eq!(
            modules.policy.arguments["chunked_prefill_size"],
            toml::Value::Integer(16384)
        );

        // A deployment that states a graph catalog and no budget gives its
        // policy the largest shape the catalog bounds a batch to: the same
        // figure the engine admits a batch under, so a policy that states no
        // bound of its own never declares more than the engine built for.
        let mut config = ServerArgs::default();
        config.cuda_graph_config.prefill_token_sizes = Some(vec![
            NonZeroU32::new(128).unwrap(),
            NonZeroU32::new(512).unwrap(),
        ]);
        config.cuda_graph_config.decode_batch_sizes = Some(vec![
            NonZeroU32::new(1).unwrap(),
            NonZeroU32::new(8).unwrap(),
        ]);
        let modules = crate::epoch::modules_of(&config);
        assert_eq!(
            modules.policy.arguments["max_num_batched_tokens"],
            toml::Value::Integer(512),
            "an unstated budget is the largest shape the captured graphs bound a batch to"
        );
        assert_eq!(
            modules.policy.arguments["chunked_prefill_size"],
            toml::Value::Integer(512),
            "and an unstated chunk is that same bound"
        );

        // The batch the policy may hold follows the same catalog: the
        // deployment's own batch, or the largest decode graph when it states
        // none, and a catalog that grows with a new epoch moves the derived
        // figure with it.
        let mut config = ServerArgs::default();
        config.cuda_graph_config.decode_batch_sizes = Some(vec![
            NonZeroU32::new(1).unwrap(),
            NonZeroU32::new(2).unwrap(),
            NonZeroU32::new(4).unwrap(),
            NonZeroU32::new(8).unwrap(),
        ]);
        let modules = crate::epoch::modules_of(&config);
        assert_eq!(
            modules.policy.arguments["max_batch"],
            toml::Value::Integer(8),
            "an unstated batch is the largest decode graph the deployment captured"
        );
        config
            .cuda_graph_config
            .decode_batch_sizes
            .as_mut()
            .unwrap()
            .push(NonZeroU32::new(16).unwrap());
        let modules = crate::epoch::modules_of(&config);
        assert_eq!(
            modules.policy.arguments["max_batch"],
            toml::Value::Integer(16),
            "and it moves with the catalog a new epoch captures"
        );
        // A batch the deployment states below the catalog keeps its own value.
        config.scheduler_config.max_batch = NonZeroUsize::new(4).unwrap();
        let modules = crate::epoch::modules_of(&config);
        assert_eq!(
            modules.policy.arguments["max_batch"],
            toml::Value::Integer(4)
        );

        // A deployment that states no token bound and no catalog states no
        // limit, and the mechanism's ceilings are the workers' own.
        let config = ServerArgs::default();
        let modules = crate::epoch::modules_of(&config);
        assert!(
            !modules
                .policy
                .arguments
                .contains_key("max_num_batched_tokens")
        );
        assert!(
            !modules
                .policy
                .arguments
                .contains_key("chunked_prefill_size")
        );
    }

    /// The order a deployment states for its waiting queue is the policy's `order` argument;
    /// one that states none leaves the package's declared default.
    #[test]
    fn the_queue_order_is_the_policys_order_argument() {
        let mut config = ServerArgs::default();
        assert!(
            !crate::epoch::modules_of(&config)
                .policy
                .arguments
                .contains_key("order")
        );

        config.scheduler_config.order = Some(proto::config::QueueOrder::Priority);
        assert_eq!(
            crate::epoch::modules_of(&config).policy.arguments["order"],
            toml::Value::String("priority".into())
        );
    }

    /// Which [`Change`] a change takes is decided by what it moves, never by
    /// the change itself: `Host` for the options the
    /// daemon or the frontend read alone, `Epoch` for the values
    /// that build an engine, `Generation` for the backends and
    /// parameters of the modules, `None` for no change.
    ///
    /// This test is also the rule's enforcement: a value that leaks from a
    /// generation update's lists into [`proto::config::EngineConfig::of`], or out of it, fails
    /// here by name.
    #[test]
    fn a_change_is_routed_by_what_it_moves() {
        /// A named change of the default configuration.
        type Case = (&'static str, fn(&mut ServerArgs));

        /// How a case changes the default configuration.
        fn set(change: fn(&mut ServerArgs)) -> ServerArgs {
            let mut config = ServerArgs::default();
            change(&mut config);
            config
        }

        // The frontend's own options: a `Host` change.
        let host_frontend: Vec<Case> = vec![
            (
                "server_config.request_id_headers",
                |config: &mut ServerArgs| config.server_config.request_id_headers = true,
            ),
            ("chat_config", |config: &mut ServerArgs| {
                config.chat_config.template = Some(proto::config::ChatTemplateSource::Inline(
                    "{{messages}}".to_owned(),
                ))
            }),
            ("mm_processor_kwargs", |config: &mut ServerArgs| {
                config
                    .mm_processor_kwargs
                    .insert("min_pixels".to_owned(), serde_json::Value::from(1024));
            }),
        ];
        // The daemon's own options: a `Host` change.
        let host_daemon: Vec<Case> = vec![
            ("logging_config.log_requests", |config: &mut ServerArgs| {
                config.logging_config.log_requests = true
            }),
            ("update_config", |config: &mut ServerArgs| {
                config.update_config.transfer_slowdown = Some(0.25)
            }),
        ];
        // The values that build an engine: an `Epoch` change.
        let epoch: Vec<Case> = vec![
            ("max_running_requests", |config: &mut ServerArgs| {
                config.max_running_requests = NonZeroUsize::new(4).unwrap()
            }),
            ("max_model_len", |config: &mut ServerArgs| {
                config.max_model_len = Some(NonZeroU32::new(2048).unwrap())
            }),
            ("mem_fraction_static", |config: &mut ServerArgs| {
                config.mem_fraction_static = 0.5
            }),
            ("lora", |config: &mut ServerArgs| {
                config.lora = Some("org/adapter".to_owned())
            }),
            (
                "speculative_config.num_draft_tokens",
                |config: &mut ServerArgs| config.speculative_config.num_draft_tokens = 4,
            ),
            ("moe_config.microbatches", |config: &mut ServerArgs| {
                config.moe_config.microbatches = NonZeroU32::new(2).unwrap()
            }),
            (
                "profiling_config.graph_pc_sampling",
                |config: &mut ServerArgs| config.profiling_config.graph_pc_sampling = true,
            ),
        ];
        // The package backends and the package parameters: a `Generation` change.
        let package: Vec<Case> = vec![
            ("scheduler_config.policy", |config: &mut ServerArgs| {
                config.scheduler_config.policy = "another".to_owned()
            }),
            ("scheduler_config.order", |config: &mut ServerArgs| {
                config.scheduler_config.order = Some(proto::config::QueueOrder::Priority)
            }),
            (
                "scheduler_config.chunked_prefill_size",
                |config: &mut ServerArgs| {
                    config.scheduler_config.chunked_prefill_size = NonZeroUsize::new(2048)
                },
            ),
            ("scheduler_config.max_batch", |config: &mut ServerArgs| {
                config.scheduler_config.max_batch = NonZeroUsize::new(8).unwrap()
            }),
            ("max_num_batched_tokens", |config: &mut ServerArgs| {
                config.max_num_batched_tokens = NonZeroUsize::new(4096)
            }),
            (
                "speculative_config.expected_speedup",
                |config: &mut ServerArgs| config.speculative_config.expected_speedup = Some(2.5),
            ),
            (
                "speculative_config.refill_token_cost",
                |config: &mut ServerArgs| config.speculative_config.refill_token_cost = Some(0.5),
            ),
            ("attention_backend", |config: &mut ServerArgs| {
                config.attention_backend = Some("attention-a".to_owned())
            }),
            ("prefill_attention_backend", |config: &mut ServerArgs| {
                config.prefill_attention_backend = Some("attention-b".to_owned())
            }),
            ("decode_attention_backend", |config: &mut ServerArgs| {
                config.decode_attention_backend = Some("attention-c".to_owned())
            }),
            ("moe_runner_backend", |config: &mut ServerArgs| {
                config.moe_runner_backend = Some("grouped".to_owned())
            }),
        ];

        let base = Epoch::of(ServerArgs::default()).expect("the configuration resolves");
        assert_eq!(change(&base, &base), Change::None, "no change");

        for (what, edit) in host_frontend.iter().chain(host_daemon.iter()) {
            let next = Epoch::of(set(*edit)).expect("the configuration resolves");
            assert_eq!(
                change(&base, &next),
                Change::Host,
                "{what} is the daemon's own option"
            );
            // The hot restart's successor serves the daemon's options the change states, so
            // the change it applies again (`"then": "update"`) restarts nothing.
            let successor = base.config.with_daemon_part(next.config.daemon_part());
            assert!(
                !host_options_differ(&successor, &next.config),
                "the successor of a change of {what} serves it"
            );
        }
        for (what, edit) in &epoch {
            let next = Epoch::of(set(*edit)).expect("the configuration resolves");
            assert_eq!(
                change(&base, &next),
                Change::Epoch,
                "{what} builds a new epoch"
            );
        }
        for (what, edit) in &package {
            let next = Epoch::of(set(*edit)).expect("the configuration resolves");
            assert_eq!(
                change(&base, &next),
                Change::Generation,
                "{what} moves the modules"
            );
        }

        // The exchange is a package only where experts are spread over ranks:
        // a single-rank deployment's MoE block takes its local arm and launches
        // no exchange at all (`nn::Moe::forward` routes through
        // `forward_expert_parallel` only when an expert lives on another rank),
        // so the setting moves nothing there — and filling the slot anyway
        // would pin a package no call reaches.
        let exchange_only =
            set(|config: &mut ServerArgs| config.moe_config.exchange = Some("deepep".to_owned()));
        assert_eq!(
            change(
                &base,
                &Epoch::of(exchange_only.clone()).expect("the configuration resolves")
            ),
            Change::None,
            "at one rank the exchange setting moves nothing"
        );
        let mut spread = ServerArgs::default();
        spread.moe_config.expert_parallel = true;
        let mut spread_exchanged = spread.clone();
        spread_exchanged.moe_config.exchange = Some("deepep".to_owned());
        assert_eq!(
            change(
                &Epoch::of(spread).expect("the configuration resolves"),
                &Epoch::of(spread_exchanged).expect("the configuration resolves")
            ),
            Change::Generation,
            "in an expert-parallel deployment the exchange moves the modules"
        );

        // Only an epoch update's values build the engine: the daemon's own
        // options and a generation update's leave the build config equal.
        for (what, change) in host_frontend
            .iter()
            .chain(host_daemon.iter())
            .chain(package.iter())
        {
            assert_eq!(
                set(*change)
                    .engine_config()
                    .expect("the configuration resolves")
                    .build_part()
                    .identity(),
                base.engine.build_part().identity(),
                "{what} is not a build-config value"
            );
        }
        for (what, change) in &epoch {
            assert_ne!(
                set(*change)
                    .engine_config()
                    .expect("the configuration resolves")
                    .build_part()
                    .identity(),
                base.engine.build_part().identity(),
                "{what} builds the engine"
            );
        }

        // A package's revision is a generation change: the same backend
        // and parameters at a newly installed revision move the modules.
        let named_base = Epoch::of(set(|config: &mut ServerArgs| {
            config.attention_backend = Some("attention-c".to_owned());
        }))
        .expect("the configuration resolves");
        let mut installed = named_base.clone();
        for package in installed.modules.kernels.values_mut() {
            package.revision = 3;
        }
        assert_eq!(
            change(&named_base, &installed),
            Change::Generation,
            "a new revision"
        );
    }

    /// The modules a configuration fills are keyed by the operators the
    /// design names, with the attention operator split by phase only when
    /// its per-phase fields state different backends, and the
    /// scheduling policy package carries the values that only inform its decisions.
    #[test]
    fn the_filled_modules_name_the_operators_and_phases() {
        let config = ServerArgs::default();
        let modules = crate::epoch::modules_of(&config);

        // One attention package for every phase, and no expert exchange for a
        // deployment whose experts all live on one rank: the model's MoE block
        // takes its local arm and launches no exchange, so the slot is absent
        // rather than filled with a package nothing calls. No
        // grouped expert GEMM either, until the configuration names one.
        // The default attention backend is `auto`: the deployment names no
        // attention package, and the Build Server selects its packages.
        for operator in [
            operator::PAGED_ATTENTION,
            operator::EXPERT_EXCHANGE,
            operator::GATHER_GEMM,
            operator::ROUTE_GEMM,
        ] {
            assert!(
                !modules.kernels.contains_key(&KernelKey::all(operator)),
                "an operator the configuration does not fill is absent"
            );
        }

        // A deployment that spreads its experts over ranks does route through
        // the exchange, and the slot states the backend its configuration
        // names.
        let mut parallel = ServerArgs::default();
        parallel.moe_config.expert_parallel = true;
        let modules = crate::epoch::modules_of(&parallel);
        assert_eq!(
            modules.kernels[&KernelKey::all(operator::EXPERT_EXCHANGE)].backend,
            "alltoall"
        );

        // The scheduling policy package's backend is the policy's name, and its
        // parameters are the configuration's values it reads, as TOML strings.
        assert_eq!(modules.policy.backend, "arrival");
        assert_eq!(
            modules.policy.arguments["max_batch"],
            toml::Value::Integer(128)
        );
        let scheduler = SchedulerConfig {
            max_queued_requests: NonZeroUsize::new(16),
            chunked_prefill_size: NonZeroUsize::new(2048),
            ..SchedulerConfig::default()
        };
        let mut stated = ServerArgs {
            scheduler_config: scheduler,
            max_num_batched_tokens: NonZeroUsize::new(4096),
            ..Default::default()
        };
        stated.speculative_config.expected_speedup = Some(2.5);
        let modules = crate::epoch::modules_of(&stated);
        assert_eq!(
            modules.policy.arguments["chunked_prefill_size"],
            toml::Value::Integer(2048)
        );
        assert_eq!(
            modules.policy.arguments["max_queued_requests"],
            toml::Value::Integer(16)
        );
        assert_eq!(
            modules.policy.arguments["max_num_batched_tokens"],
            toml::Value::Integer(4096)
        );
        assert_eq!(
            modules.policy.arguments["expected_speedup"],
            toml::Value::Float(2.5)
        );

        // Per-phase attention: two entries, each naming its phase, once the
        // fields state different backends.
        let split = ServerArgs {
            prefill_attention_backend: Some("attention-b".to_owned()),
            decode_attention_backend: Some("attention-c".to_owned()),
            ..Default::default()
        };
        let modules = crate::epoch::modules_of(&split);
        let prefill =
            &modules.kernels[&KernelKey::phase(operator::PAGED_ATTENTION, Phase::Prefill)];
        let decode = &modules.kernels[&KernelKey::phase(operator::PAGED_ATTENTION, Phase::Decode)];
        assert_eq!(
            (prefill.backend.as_str(), decode.backend.as_str()),
            ("attention-b", "attention-c")
        );
        assert!(
            !modules
                .kernels
                .contains_key(&KernelKey::all(operator::PAGED_ATTENTION)),
            "a per-phase fill covers no whole operator"
        );

        // One backend for both phases fills the operator once, whether the
        // operator states it or both phases state the same one; an unstated
        // phase falls back to `attention_backend`, which the split above
        // keeps as a phase of its own.
        let one = ServerArgs {
            attention_backend: Some("attention-b".to_owned()),
            ..Default::default()
        };
        let modules = crate::epoch::modules_of(&one);
        assert_eq!(
            modules.kernels[&KernelKey::all(operator::PAGED_ATTENTION)].backend,
            "attention-b"
        );
        assert!(
            !modules
                .kernels
                .contains_key(&KernelKey::phase(operator::PAGED_ATTENTION, Phase::Prefill)),
            "one backend covers every phase of the operator"
        );
        let agreed = ServerArgs {
            prefill_attention_backend: Some("attention-b".to_owned()),
            decode_attention_backend: Some("attention-b".to_owned()),
            ..Default::default()
        };
        let modules = crate::epoch::modules_of(&agreed);
        assert_eq!(
            modules.kernels[&KernelKey::all(operator::PAGED_ATTENTION)].backend,
            "attention-b",
            "phases that agree collapse to the operator's slot"
        );

        // A stated grouped expert GEMM package fills both of its operators.
        let moe = ServerArgs {
            moe_runner_backend: Some("grouped".to_owned()),
            ..Default::default()
        };
        let modules = crate::epoch::modules_of(&moe);
        for operator in [operator::GATHER_GEMM, operator::ROUTE_GEMM] {
            assert_eq!(
                modules.kernels[&KernelKey::all(operator)].backend,
                "grouped"
            );
        }
    }

    /// A backend option left unset names no slot, whichever phase it is
    /// for: the Build Server selects that operator's packages, and a phase
    /// the deployment does name keeps its package.
    #[test]
    fn an_unset_backend_names_no_package() {
        let mut config = ServerArgs {
            decode_attention_backend: Some("base".to_owned()),
            ..Default::default()
        };
        let modules = crate::epoch::modules_of(&config);
        assert_eq!(
            modules.kernels.keys().cloned().collect::<Vec<_>>(),
            vec![KernelKey::phase(operator::PAGED_ATTENTION, Phase::Decode)]
        );

        config.attention_backend = Some("attention-c".to_owned());
        config.decode_attention_backend = None;
        let modules = crate::epoch::modules_of(&config);
        assert_eq!(
            modules.kernels[&KernelKey::all(operator::PAGED_ATTENTION)].backend,
            "attention-c"
        );
    }

    /// Modules are recorded per operator and, where the configuration fills
    /// one per phase, per phase; the epoch that serves is the one of the
    /// highest number that is serving or on trial.
    #[test]
    fn epochs_are_reported_by_operator_phase_and_status() {
        let epochs = Epochs::default();
        let mut first = record();
        first.status = EpochState::Trial;
        epochs.record(1, first);
        let mut second = record();
        second.status = EpochState::Preparing;
        second.kernels.insert(
            KernelKey::phase(operator::PAGED_ATTENTION, Phase::Decode),
            package("attention-c"),
        );
        epochs.record(2, second);
        assert_eq!(
            epochs
                .serving()
                .map(|(epoch, record)| (epoch.number(), record.status)),
            Some((1, EpochState::Trial))
        );

        // The epochs are reported ascending, each naming its number.
        let statuses = epochs.statuses();
        assert_eq!(
            statuses
                .iter()
                .map(|status| (status.epoch, status.status))
                .collect::<Vec<_>>(),
            vec![(1, EpochState::Trial), (2, EpochState::Preparing)]
        );
        let decode = &statuses[1].kernels[0];
        assert_eq!(decode.slot.phase, Some(Phase::Decode));
        assert_eq!(decode.package.backend, "attention-c");
    }

    /// The epoch's engine configuration is written by the daemon into each
    /// launch file and read by the engine through this same type, so
    /// what the writer serializes must read
    /// back: an unset list is written as `null` and must be read as `None`,
    /// not refused.
    #[test]
    fn an_engine_config_reads_back_what_it_wrote() {
        let config = ServerArgs::default();
        let build = config.engine_config().expect("the configuration resolves");
        assert!(build.cuda_graph.decode_batch_sizes.is_none());
        let text = serde_json::to_string(&build).expect("an engine config is JSON");
        let read: proto::config::EngineConfig =
            serde_json::from_str(&text).expect("the engine config reads back through its writer");
        assert_eq!(read, build);
    }
}
