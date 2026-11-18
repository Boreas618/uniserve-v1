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
pub use proto::module::{ModuleSet, ModuleSlot, PackageRevision};
use serde::Serialize;

use crate::config::Config;

/// The key an epoch's modules are stored under: the operator, and the
/// descriptor key the configuration's `[kernels]` table pins its package
/// for. Modules and slots are `proto::module`'s types, which the daemon, the
/// engine and the Build Server all name modules with.
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
/// its node: the state its sampled gauges (`uniserve_epoch_state`) and `process` events
/// report and a change is routed by.
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

    /// Every epoch's record, ascending by number, whose state `uniserve_epoch_state` samples.
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
/// and the kernel package its `[kernels]` table pins for each descriptor key it lists.
///
/// The revisions are fixed when the configuration is applied, at the Build Server: a package's
/// `revision` is zero until then, unless the configuration states one, and a pin of `latest`
/// becomes the latest installed revision. A key the table does not list is absent: the
/// deployment names nothing for it, and the Build Server selects its package among the packages
/// that cover it, by manifest priority. An operator's names are those of the module vocabulary
/// (`proto::module::operator`), which [`Config::validate`] holds the table to, so the daemon, the
/// Build Server and the engine cannot disagree about a slot's name.
pub fn modules_of(config: &Config) -> ModuleSet {
    let kernels = config
        .kernels
        .iter()
        .flat_map(|(op, pins)| {
            pins.iter().map(move |(key, pin)| {
                let package = PackageRevision {
                    backend: pin.backend.clone(),
                    revision: pin.revision.unwrap_or(0),
                    arguments: toml::Table::new(),
                };
                (KernelKey::key(op.as_str(), key.as_str()), package)
            })
        })
        .collect();
    ModuleSet {
        engine: PackageRevision {
            backend: config.worker_config.engine.clone(),
            revision: config.worker_config.engine_revision.unwrap_or(0),
            arguments: toml::Table::new(),
        },
        kernels,
        // The installed default until the package's programs are fixed ([`with_programs`]).
        policy: PackageRevision {
            backend: proto::config::DEFAULT_SCHEDULING.to_owned(),
            revision: 0,
            arguments: policy_arguments(config),
        },
        programs: Default::default(),
    }
}

/// `named`, the module set of `config` ([`modules_of`]), with the package's programs as their
/// apply fixed them: every program recorded, and the scheduling program, when the package lists
/// one, the engines' policy, its `args` over the configuration's policy arguments. A package
/// that lists none keeps the installed default.
pub fn with_programs(
    mut named: ModuleSet,
    config: &Config,
    programs: BTreeMap<String, proto::module::Program>,
) -> ModuleSet {
    let scheduling = programs
        .iter()
        .find(|(_, program)| program.kind == scheduling::program::Kind::Scheduling);
    if let Some((name, program)) = scheduling {
        let mut arguments = policy_arguments(config);
        if let Some(entry) = config.programs.get(name) {
            arguments.extend(entry.args.clone());
        }
        named.policy = PackageRevision {
            backend: program.backend.clone(),
            revision: program.revision,
            arguments,
        };
    }
    named.programs = programs;
    named
}

/// The scheduling policy's arguments the configuration states beside its program's `args`: the
/// speculation's decision values. The step's budget, its prefill chunk and the waiting queue's
/// bound are no argument of the policy: the schedulers read them at every step
/// (`proto::config::Tuning`).
fn policy_arguments(config: &Config) -> toml::Table {
    #[derive(Serialize)]
    struct Arguments {
        #[serde(skip_serializing_if = "Option::is_none")]
        expected_speedup: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        refill_token_cost: Option<f64>,
    }
    let arguments = Arguments {
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
    pub config: Config,
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
    /// ([`Config::engine_config`]): its deployment file does not read or place.
    pub fn of(config: Config) -> anyhow::Result<Self> {
        let modules = crate::epoch::modules_of(&config);
        Self::with_modules(config, modules)
    }

    /// The deployment `config` describes, with the modules `prepare` selected for it.
    ///
    /// # Errors
    ///
    /// Fails as [`Self::of`] does.
    pub fn with_modules(config: Config, modules: ModuleSet) -> anyhow::Result<Self> {
        Ok(Self {
            engine: config.engine_config()?,
            config,
            modules,
        })
    }
}

/// What `next` changes of `previous`, by the effects its options declare
/// ([`proto::reload`]) and the module sets both fix: the daemon's own options
/// first, then an epoch (an option that builds the engine, a replaced Storage
/// Server, or another engine package or revision), then the kernel and
/// program modules and the kernel probes' placement. A change of options that
/// take effect at the next step alone is `None`: no update moves it.
pub fn change(previous: &Epoch, next: &Epoch) -> Change {
    let effects = proto::reload::Effects::of(&proto::reload::changes(
        Some(&previous.config),
        &next.config,
    ));
    if effects.daemon {
        Change::Host
    } else if effects.has(proto::reload::Effect::Epoch)
        || previous.modules.engine != next.modules.engine
    {
        // Another engine package, or another revision of it, is another
        // executable: its engines are started beside the serving ones.
        Change::Epoch
    } else if effects.has(proto::reload::Effect::Generation)
        || previous.modules != next.modules
        || previous.engine.profiling.kernel_instance != next.engine.profiling.kernel_instance
    {
        // Another kernel package, or the kernel probes placed elsewhere: the graphs are
        // captured again.
        Change::Generation
    } else {
        Change::None
    }
}

#[cfg(test)]
mod tests {
    use std::num::{NonZeroU32, NonZeroUsize};

    use super::*;
    use crate::config::SchedulerConfig;

    /// The pin `text` states (`backend@revision`).
    fn pin(text: &str) -> proto::module::Pin {
        text.parse().expect("a pin reads")
    }

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
        let config = Config::default();
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
        let serving = Epoch::of(Config::default()).expect("the configuration resolves");
        let other = Config {
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

    /// The step's budget, its prefill chunk and the waiting queue's bound are step values, no
    /// argument of the scheduling policy: the policy's package is the same whatever they are,
    /// and the launch's step values state them as the deployment does.
    #[test]
    fn the_steps_budget_is_a_step_value_not_a_policy_argument() {
        let mut config = Config {
            max_num_batched_tokens: NonZeroUsize::new(4096),
            ..Default::default()
        };
        config.scheduler_config.chunked_prefill_size = NonZeroUsize::new(1024);
        config.scheduler_config.max_batch = NonZeroUsize::new(16).unwrap();
        config.scheduler_config.max_queued_requests = NonZeroUsize::new(8);
        let modules = crate::epoch::modules_of(&config);
        assert_eq!(
            modules.policy,
            crate::epoch::modules_of(&Config::default()).policy
        );

        let tuning = proto::config::Tuning::of(
            &config.engine_config().expect("the configuration resolves"),
            5,
        );
        assert_eq!(tuning.revision, 5);
        assert_eq!(tuning.max_num_batched_tokens, NonZeroUsize::new(4096));
        assert_eq!(tuning.chunked_prefill_size, NonZeroUsize::new(1024));
        assert_eq!(tuning.max_batch.get(), 16);
        assert_eq!(tuning.max_queued_requests, NonZeroUsize::new(8));
    }

    /// The package's scheduling program is the engines' policy, its `args` the policy's
    /// arguments over the configuration's; a package that lists none runs the installed default,
    /// and its other programs are recorded without changing the policy.
    #[test]
    fn the_scheduling_program_is_the_engines_policy() {
        let program = |kind, backend: &str| proto::module::Program {
            source: format!("uniserve:{backend}"),
            backend: backend.to_owned(),
            revision: 3,
            kind,
            digest: "ab".repeat(32),
        };
        let mut config = Config::default();
        config.speculative_config.expected_speedup = Some(2.0);
        let chunk = BTreeMap::from([(
            "chunk".to_owned(),
            program(scheduling::program::Kind::Control, "chunk"),
        )]);
        let named = with_programs(modules_of(&config), &config, chunk.clone());
        assert_eq!(named.policy.backend, proto::config::DEFAULT_SCHEDULING);
        assert_eq!(named.programs, chunk);

        config.programs.insert(
            "fair".to_owned(),
            toml::from_str("source = \"uniserve:fair\"\nargs = { order = \"priority\" }\n")
                .unwrap(),
        );
        let mut programs = chunk;
        programs.insert(
            "fair".to_owned(),
            program(scheduling::program::Kind::Scheduling, "fair"),
        );
        let named = with_programs(modules_of(&config), &config, programs);
        assert_eq!(
            (named.policy.backend.as_str(), named.policy.revision),
            ("fair", 3)
        );
        assert_eq!(named.policy.arguments["order"].as_str(), Some("priority"));
        assert_eq!(
            named.policy.arguments["expected_speedup"].as_float(),
            Some(2.0)
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
        type Case = (&'static str, fn(&mut Config));

        /// How a case changes the default configuration.
        fn set(change: fn(&mut Config)) -> Config {
            let mut config = Config::default();
            change(&mut config);
            config
        }

        // The frontend's own options: a `Host` change.
        let host_frontend: Vec<Case> = vec![
            ("server_config.request_id_headers", |config: &mut Config| {
                config.server_config.request_id_headers = true
            }),
            ("chat_config", |config: &mut Config| {
                config.chat_config.template = Some(proto::config::ChatTemplateSource::Inline(
                    "{{messages}}".to_owned(),
                ))
            }),
            ("mm_processor_kwargs", |config: &mut Config| {
                config
                    .mm_processor_kwargs
                    .insert("min_pixels".to_owned(), serde_json::Value::from(1024));
            }),
        ];
        // The daemon's own options: a `Host` change.
        let host_daemon: Vec<Case> = vec![
            ("logging_config.log_requests", |config: &mut Config| {
                config.logging_config.log_requests = true
            }),
            ("update_config", |config: &mut Config| {
                config.update_config.transfer_slowdown = Some(0.25)
            }),
        ];
        // The values that build an engine: an `Epoch` change.
        let epoch: Vec<Case> = vec![
            ("max_running_requests", |config: &mut Config| {
                config.max_running_requests = NonZeroUsize::new(4).unwrap()
            }),
            ("max_model_len", |config: &mut Config| {
                config.max_model_len = Some(NonZeroU32::new(2048).unwrap())
            }),
            // A larger share is divided by a new epoch's ranks.
            ("mem_fraction_static", |config: &mut Config| {
                config.mem_fraction_static = 0.9
            }),
            ("lora", |config: &mut Config| {
                config.lora = Some("org/adapter".to_owned())
            }),
            (
                "speculative_config.num_draft_tokens",
                |config: &mut Config| config.speculative_config.num_draft_tokens = 4,
            ),
            (
                "profiling_config.graph_pc_sampling",
                |config: &mut Config| config.profiling_config.graph_pc_sampling = true,
            ),
        ];
        // The package backends and the package parameters: a `Generation` change.
        let package: Vec<Case> = vec![
            (
                "speculative_config.expected_speedup",
                |config: &mut Config| config.speculative_config.expected_speedup = Some(2.5),
            ),
            (
                "speculative_config.refill_token_cost",
                |config: &mut Config| config.speculative_config.refill_token_cost = Some(0.5),
            ),
            // The options the graphs are captured with.
            ("moe_config.microbatches", |config: &mut Config| {
                config.moe_config.microbatches = NonZeroU32::new(2).unwrap()
            }),
            ("profiling_config.fence_modules", |config: &mut Config| {
                config.profiling_config.fence_modules = Some(vec!["*.Attention".to_owned()])
            }),
            ("kernels", |config: &mut Config| {
                config.kernels.insert(
                    "paged_attention".to_owned(),
                    [(
                        "phase_decode_head_128".to_owned(),
                        pin("attention-a@latest"),
                    )]
                    .into(),
                );
            }),
        ];

        // The step values: in place, at the schedulers' next step.
        let step: Vec<Case> = vec![
            (
                "scheduler_config.chunked_prefill_size",
                |config: &mut Config| {
                    config.scheduler_config.chunked_prefill_size = NonZeroUsize::new(2048)
                },
            ),
            ("scheduler_config.max_batch", |config: &mut Config| {
                config.scheduler_config.max_batch = NonZeroUsize::new(8).unwrap()
            }),
            (
                "scheduler_config.max_queued_requests",
                |config: &mut Config| {
                    config.scheduler_config.max_queued_requests = NonZeroUsize::new(8)
                },
            ),
            ("max_num_batched_tokens", |config: &mut Config| {
                config.max_num_batched_tokens = NonZeroUsize::new(4096)
            }),
            // What every main process reads beside its batches, and what the next start of a
            // process reads: no option of these sizes a storage or shapes a graph.
            ("worker_config.watchdog_timeout", |config: &mut Config| {
                config.worker_config.watchdog_timeout = std::time::Duration::from_secs(60)
            }),
            ("worker_config.join_timeout", |config: &mut Config| {
                config.worker_config.join_timeout = std::time::Duration::from_secs(60)
            }),
            ("worker_config.compare_timeout", |config: &mut Config| {
                config.worker_config.compare_timeout = std::time::Duration::from_secs(60)
            }),
            ("worker_config.report_period", |config: &mut Config| {
                config.worker_config.report_period = std::time::Duration::from_millis(250)
            }),
            ("worker_config.copy_budget", |config: &mut Config| {
                config.worker_config.copy_budget = 64 << 20
            }),
            ("device_config.queue_depth", |config: &mut Config| {
                config.device_config.queue_depth = NonZeroUsize::new(4).unwrap()
            }),
            ("profiling_config.nvtx_ranges", |config: &mut Config| {
                config.profiling_config.nvtx_ranges = true
            }),
            // Switched off, a capability the epoch laid out storage for goes unused.
            ("kv_cache_config.prefix_caching", |config: &mut Config| {
                config.kv_cache_config.prefix_caching = false
            }),
            ("kv_cache_config.host_tier", |config: &mut Config| {
                config.kv_cache_config.host_tier = false
            }),
            // A smaller share of the GPUs is given back by the serving engines' caches.
            ("mem_fraction_static", |config: &mut Config| {
                config.mem_fraction_static = 0.5
            }),
        ];

        let base = Epoch::of(Config::default()).expect("the configuration resolves");
        assert_eq!(change(&base, &base), Change::None, "no change");
        for (what, edit) in &step {
            let next = Epoch::of(set(*edit)).expect("the configuration resolves");
            assert_eq!(change(&base, &next), Change::None, "{what} moves no update");
            let changes = proto::reload::changes(Some(&base.config), &next.config);
            assert_eq!(
                proto::reload::Effects::of(&changes).heaviest(),
                Some(proto::reload::Effect::Step),
                "{what} takes effect at the next step"
            );
        }

        for (what, edit) in host_frontend.iter().chain(host_daemon.iter()) {
            let next = Epoch::of(set(*edit)).expect("the configuration resolves");
            assert_eq!(
                change(&base, &next),
                Change::Host,
                "{what} is the daemon's own option"
            );
            // The hot restart's successor serves the daemon's options the change states, so
            // the change it applies again restarts nothing.
            let mut successor = base.config.clone();
            proto::reload::Reload::take(
                &mut successor,
                &next.config,
                proto::reload::Effect::Daemon,
            );
            assert_eq!(
                successor, next.config,
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
        // Switched on, the capability needs the storage a new epoch lays out.
        let mut uncached = Config::default();
        uncached.kv_cache_config.prefix_caching = false;
        uncached.kv_cache_config.host_tier = false;
        let uncached = Epoch::of(uncached).expect("the configuration resolves");
        assert_eq!(change(&uncached, &base), Change::Epoch, "switching on");
        for (what, edit) in &package {
            let next = Epoch::of(set(*edit)).expect("the configuration resolves");
            assert_eq!(
                change(&base, &next),
                Change::Generation,
                "{what} moves the modules"
            );
        }

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
        let named_base = Epoch::of(set(|config: &mut Config| {
            config.kernels.insert(
                "paged_attention".to_owned(),
                [(
                    "phase_decode_head_128".to_owned(),
                    pin("attention-c@latest"),
                )]
                .into(),
            );
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

    /// The modules a configuration fills are the pins of its `[kernels]` table, one slot per
    /// pinned descriptor key at the revision the pin states (zero for `latest`, which the apply
    /// fixes), and none for a key the table leaves to the Build Server; the scheduling policy
    /// package carries the values that only inform its decisions.
    #[test]
    fn the_filled_modules_are_the_pinned_keys() {
        let modules = crate::epoch::modules_of(&Config::default());
        assert!(
            modules.kernels.is_empty(),
            "a deployment that pins nothing leaves every key to the Build Server"
        );

        let prefill = "phase_prefill_head_128";
        let decode = "phase_decode_head_128";
        let mut pinned = Config::default();
        pinned.kernels.insert(
            "paged_attention".to_owned(),
            [
                (prefill.to_owned(), pin("trtllm@7")),
                (decode.to_owned(), pin("triton@latest")),
            ]
            .into(),
        );
        pinned.kernels.insert(
            "gather_gemm".to_owned(),
            [("m_64".to_owned(), pin("grouped@2"))].into(),
        );
        let modules = crate::epoch::modules_of(&pinned);
        let held: Vec<(KernelKey, &str, u32)> = modules
            .kernels
            .iter()
            .map(|(slot, package)| (slot.clone(), package.backend.as_str(), package.revision))
            .collect();
        assert_eq!(
            held,
            vec![
                (KernelKey::key("gather_gemm", "m_64"), "grouped", 2),
                (KernelKey::key("paged_attention", decode), "triton", 0),
                (KernelKey::key("paged_attention", prefill), "trtllm", 7),
            ]
        );

        // The scheduling policy package's backend is the policy's name, and its
        // parameters are the configuration's values it reads, as TOML values.
        assert_eq!(modules.policy.backend, "arrival");
        let scheduler = SchedulerConfig {
            max_queued_requests: NonZeroUsize::new(16),
            chunked_prefill_size: NonZeroUsize::new(2048),
            ..SchedulerConfig::default()
        };
        let mut stated = Config {
            scheduler_config: scheduler,
            max_num_batched_tokens: NonZeroUsize::new(4096),
            ..Default::default()
        };
        stated.speculative_config.expected_speedup = Some(2.5);
        let modules = crate::epoch::modules_of(&stated);
        assert_eq!(
            modules.policy.arguments["expected_speedup"],
            toml::Value::Float(2.5)
        );
    }

    /// A table that names an operator this build lacks, an empty key, or one backend of an
    /// operator at two revisions is refused before anything is selected.
    #[test]
    fn a_table_the_build_server_could_not_select_by_is_refused() {
        let stated = crate::config::Stated::from_paths(Vec::new());
        let refused = |op: &str, pins: &[(&str, &str)]| {
            let mut config = Config {
                model: Some("org/model".to_owned()),
                ..Config::default()
            };
            config.kernels.insert(
                op.to_owned(),
                pins.iter()
                    .map(|(key, text)| ((*key).to_owned(), pin(text)))
                    .collect(),
            );
            config
                .validate(&stated)
                .err()
                .map(|error| error.to_string())
        };
        assert_eq!(
            refused("paged_attention", &[("a", "trtllm@7"), ("b", "triton@2")]),
            None
        );
        for (op, pins, named) in [
            ("no_such_operator", vec![("a", "trtllm@7")], "no operator"),
            ("paged_attention", vec![("", "trtllm@7")], "empty"),
            (
                "paged_attention",
                vec![("a", "trtllm@7"), ("b", "trtllm@latest")],
                "one revision",
            ),
        ] {
            let error = refused(op, &pins).expect("the table is refused");
            assert!(error.contains(named), "{error}");
        }
    }

    /// Modules are recorded per operator and, where the configuration pins
    /// a descriptor key, per key; the epoch that serves is the one of the
    /// highest number that is serving or on trial.
    #[test]
    fn epochs_are_reported_by_operator_key_and_status() {
        let epochs = Epochs::default();
        let mut first = record();
        first.status = EpochState::Trial;
        epochs.record(1, first);
        let mut second = record();
        second.status = EpochState::Preparing;
        second.kernels.insert(
            KernelKey::key("paged_attention", "phase_decode_head_128"),
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
        assert_eq!(decode.slot.key.as_deref(), Some("phase_decode_head_128"));
        assert_eq!(decode.package.backend, "attention-c");
    }

    /// The epoch's engine configuration is written by the daemon into each
    /// launch file and read by the engine through this same type, so
    /// what the writer serializes must read
    /// back: an unset list is written as `null` and must be read as `None`,
    /// not refused.
    #[test]
    fn an_engine_config_reads_back_what_it_wrote() {
        let config = Config::default();
        let build = config.engine_config().expect("the configuration resolves");
        assert!(build.cuda_graph.decode_batch_sizes.is_none());
        let text = serde_json::to_string(&build).expect("an engine config is JSON");
        let read: proto::config::EngineConfig =
            serde_json::from_str(&text).expect("the engine config reads back through its writer");
        assert_eq!(read, build);
    }
}
