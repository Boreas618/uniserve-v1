//! The scheduling policy a scheduler thread runs, and the generation switch
//! that replaces it in place (§4.4, §7.3).
//!
//! The mechanism asks the policy for one decision at each decision point and
//! validates and executes the answer; this module owns the wasmtime instance
//! that answers, the module kept for a revert, the instance a precopy prepared,
//! the stages that change them, and the control thread's own instance for
//! `route`.
//!
//! The switch follows §7.3's stages. A precopy instantiates the new policy on
//! the state that exists now and asks it every decision once, executing
//! nothing, so an illegal answer or an invalid decision fails the update while
//! the serving policy is untouched. A switchover puts the prepared instance in
//! at a batch boundary and keeps the module it replaced. A commit leaves that
//! module in place until the next precopy, which is where §7.3's purge of it
//! happens. A revert instantiates it on the state of that moment and puts it
//! in. A policy that fails is reverted the same way and reported as
//! [`PolicyReverted`]; the control thread forwards the report to the daemons,
//! which record the previous policy as the epoch's current one.
//!
//! Both instances — each scheduler thread's and the control thread's
//! [`Router`] — come from one package and switch together (§4.4: "两者是同一个
//! package，一起换代"). The router's instance answers `route` from the arriving
//! request's view and the ranks' views alone; it is told no events.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use scheduling::guest::decision::{Plan, Room, Shortage, Victim};
use scheduling::guest::event::Event;
pub use scheduling::guest::limits::{Limits, Parameters};
use scheduling::guest::policy::Instantiation;
use scheduling::guest::view::{
    CallView, RankView, ReplicaView, RequestState, RequestView, Snapshot,
};
use scheduling::guest::wire::Call;
use scheduling::{Engine, PolicyError, PolicyInstance, PolicyModule};
use serde::{Deserialize, Serialize};

/// A policy package's identity and arguments (§5.1): the backend the
/// configuration names, the revision the Build Server numbered the package
/// under, and the arguments that shape its decisions.
///
/// The arguments are part of the package — a policy's parameters are not the
/// mechanism's configuration (§4.4) — so a change of either is a new package
/// and a new generation generation (§7.2).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicyPackage {
    /// The backend's name, as `scheduler_config.policy` spells it.
    pub backend: String,
    /// The revision the Build Server numbered the package's install (§5.1).
    pub revision: u32,
    /// The policy's arguments, from the deployment configuration (§7.7).
    pub arguments: Parameters,
    /// The arguments as the daemon stated them, one TOML value each, which is
    /// what the Build Server resolves the module with: the server fills the
    /// package's defaults for the ones the deployment leaves out.
    pub stated: std::collections::BTreeMap<String, String>,
}

impl PolicyPackage {
    /// The package a daemon's `Precopy` states (§7.3 step 2, §5.1): the
    /// backend, the revision, and the policy's arguments as the TOML text of
    /// each configured value.
    ///
    /// The parameter names are the policy's own (§4.4, §7.7): this build knows
    /// the ones [`Parameters`] declares, and an argument it does not know is
    /// refused rather than ignored, because the package's arguments are the
    /// package and serving under fewer of them is not serving under it.
    pub fn from_arguments(
        backend: impl Into<String>,
        revision: u32,
        arguments: &std::collections::BTreeMap<String, String>,
    ) -> Result<Self, PolicyError> {
        let backend = backend.into();
        let mut parameters = Parameters::default();
        for (name, value) in arguments {
            match name.as_str() {
                "chunked_prefill_size" => {
                    parameters.chunked_prefill_size = Some(argument(value, name)?)
                }
                "max_queued_requests" => {
                    parameters.max_queued_requests = Some(argument(value, name)?)
                }
                "max_batch" => parameters.max_batch = Some(argument(value, name)?),
                "max_num_batched_tokens" => {
                    parameters.max_num_batched_tokens = Some(argument(value, name)?)
                }
                "prefetch_min_tokens" => {
                    parameters.prefetch_min_tokens = Some(argument(value, name)?)
                }
                "timeout" => parameters.timeout = Some(argument(value, name)?),
                "expected_speedup" => parameters.expected_speedup = Some(argument(value, name)?),
                "refill_token_cost" => parameters.refill_token_cost = Some(argument(value, name)?),
                other => {
                    return Err(PolicyError::Load {
                        message: format!(
                            "the {backend} policy package states `{other}`, which this build's \
                             policy arguments do not name"
                        ),
                    });
                }
            }
        }
        Ok(Self {
            backend,
            revision,
            arguments: parameters,
            stated: arguments.clone(),
        })
    }

    /// The package as the Build Server names it (§5.1): the backend, the
    /// revision, and the stated arguments as TOML.
    pub fn to_package_revision(&self) -> Result<proto::module::PackageRevision, PolicyError> {
        let mut arguments = toml::Table::new();
        for (name, value) in &self.stated {
            arguments.insert(name.clone(), argument::<toml::Value>(value, name)?);
        }
        Ok(proto::module::PackageRevision {
            backend: self.backend.clone(),
            revision: self.revision,
            arguments,
        })
    }

    /// The package a daemon's `Precopy` states for the scheduling policy
    /// (§7.3 step 2): its `package` field, as the process interface carries it.
    ///
    /// The wire form is the Build Server's own `PackageRevision` — a backend, a
    /// revision and the arguments as TOML text — so nothing about a package is
    /// decided here: a name the interface does not have is refused, as it is by
    /// [`Self::from_arguments`].
    pub fn from_package_revision(
        package: &proto::process::PackageRevision,
    ) -> Result<Self, PolicyError> {
        let arguments: std::collections::BTreeMap<String, String> = package
            .arguments
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        Self::from_arguments(package.backend.clone(), package.revision, &arguments)
    }
}

/// One configured argument of a package, parsed as the type the policy
/// parameter has.
///
/// The package states each argument as TOML text of the value itself — `2048`,
/// `2.5` — so it is parsed as a TOML value and turned into the parameter's type,
/// not read as a document.
fn argument<T: serde::de::DeserializeOwned>(value: &str, name: &str) -> Result<T, PolicyError> {
    let failure = |error: String| PolicyError::Load {
        message: format!("the policy argument `{name} = {value}` does not parse: {error}"),
    };
    let parsed: toml::Value = value
        .trim()
        .parse()
        .map_err(|error: toml::de::Error| failure(error.to_string()))?;
    parsed
        .try_into()
        .map_err(|error: toml::de::Error| failure(error.to_string()))
}

/// The build config's hard bounds on what one step may be given to run (§4.4):
/// the calls and tokens the epoch's workers were started for.
///
/// These are the ceilings a policy's `limits()` is checked against, so they
/// come from the build config — the captured graph tiers' call bound and token
/// size — and never from the policy's own arguments, which would make the check
/// vacuous.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BatchCapacity {
    /// Calls one step may carry, from the workers' batch call bound.
    pub max_batch: u32,
    /// Tokens one step may carry, from the workers' batch token bound.
    pub max_num_batched_tokens: u32,
}

impl BatchCapacity {
    /// The capacity the loaded workers were started for.
    pub fn of(info: &ipc::MainProcessInfo) -> Self {
        Self {
            max_batch: info.max_batch_calls,
            max_num_batched_tokens: info.max_batch_tokens,
        }
    }
}

/// The build config's ceilings for a policy's limits (§4.4): the batch and token
/// bounds the epoch was built for, the deepening floor of a prefix hit, and the
/// longest starvation deadline.
///
/// A policy declares its limits once at instantiation and they are checked
/// against these; a limit above its ceiling is a policy failure
/// (`PolicyError::BadLimits`), never a clamp. The queue bound is the policy's
/// own: the deployment states none the epoch was built for.
pub fn ceilings_of(capacity: BatchCapacity) -> Limits {
    Limits {
        // A prefill chunk is bounded by the step's token budget whatever the
        // policy's own argument says.
        chunked_prefill_size: capacity.max_num_batched_tokens,
        max_queued_requests: None,
        max_batch: capacity.max_batch as usize,
        max_num_batched_tokens: capacity.max_num_batched_tokens,
        // §4.4 fixes the deepening floor of a prefix hit and the longest
        // starvation deadline (30 s, as sched_ext caps `timeout_ms`).
        prefetch_min_tokens: PREFETCH_FLOOR_TOKENS,
        timeout: scheduling::MAX_STARVATION_TIMEOUT,
    }
}

/// The process's WASM engine: one engine for every policy instance the main
/// process runs, the scheduler threads' and the control thread's.
///
/// An engine holds the compiler and its configuration, not a module's state; a
/// failure to build it is the deployment's, and every precopy reports it the
/// same way.
pub(crate) fn process_engine() -> Result<&'static Engine, PolicyError> {
    static ENGINE: OnceLock<Result<Engine, String>> = OnceLock::new();
    match ENGINE.get_or_init(|| Engine::new().map_err(|error| error.to_string())) {
        Ok(engine) => Ok(engine),
        Err(message) => Err(PolicyError::Load {
            message: message.clone(),
        }),
    }
}

/// The build config's smallest prefix-hit deepening (§4.4: a host hit that
/// deepens the device hit by fewer tokens is not prefetched).
///
/// It is a build-config value, so a policy's `prefetch_min_tokens` may be above
/// it but not below it (`scheduling`'s check of the policy's limits).
pub const PREFETCH_FLOOR_TOKENS: u32 = 256;

/// Where the switch resolves a policy package's module (§5.1, §5.3).
///
/// The daemon's commands and the epoch record name a package by backend,
/// revision and arguments, never by content: the content comes from the
/// Build Server, through this source.
pub trait PolicySource: Send + Sync {
    /// The module bytes of `package`: the artifact the Build Server
    /// precompiled for this host, or the wasm32 binary when it has none.
    fn bytes(&self, package: &PolicyPackage) -> Result<Vec<u8>, PolicyError>;
}

/// The Build Server's client, as the source of a policy's module (§5.3,
/// §7.3 step 2).
///
/// The daemon names a package by backend, revision and arguments, never by
/// content: this asks the host's Build Server to `resolve` it — the store's
/// copy, or a build of it when the store misses it — under the `scheduling`
/// operator's single descriptor ([`build_server::manifest::SINGLE_KEY`]).
///
/// The key it resolves under is the store's wasm target, not a CPU triple: a
/// policy is one module per revision, built from its snapshot for
/// [`build_server::manifest::WASM_TARGET`], and it belongs to the whole
/// architecture rather than to one CPU (§5.3). The host's precompile triple is
/// wasmtime's business and lives in [`Engine::new`].
///
/// One connection is kept and reused: a precopy is the only caller, and it
/// runs on a thread of its own, so the lock is never held across a step.
#[derive(Debug)]
pub struct ModuleSource {
    socket: PathBuf,
    target: build_server::Target,
    client: std::sync::Mutex<Option<build_server::Client>>,
}

/// The operator a scheduling policy's package fills (§5.2).
const POLICY_OP: &str = "scheduling";

impl ModuleSource {
    /// A source that resolves packages through the Build Server listening on
    /// `socket`, under the store's key for a scheduling policy.
    ///
    /// The key is [`build_server::manifest::WASM_TARGET`] and nothing finer.
    /// Asking for the host's precompile triple instead named a path no install
    /// writes and no policy package declares, so the resolve could not read the
    /// artifact the install built; a host that cannot precompile a module still
    /// fails, but in [`Engine::new`], which is where that target is chosen.
    pub fn connect(socket: impl Into<PathBuf>) -> Self {
        Self {
            socket: socket.into(),
            target: build_server::Target::new(build_server::manifest::WASM_TARGET),
            client: std::sync::Mutex::new(None),
        }
    }

    /// The socket the source asks.
    pub fn socket(&self) -> &Path {
        &self.socket
    }

    /// The store key every resolve of this source asks for.
    pub fn target(&self) -> &build_server::Target {
        &self.target
    }

    /// With `self`'s connection open, for the caller to resolve through.
    fn with_client<T>(
        &self,
        ask: impl FnOnce(&mut build_server::Client) -> Result<T, PolicyError>,
    ) -> Result<T, PolicyError> {
        let mut client = crate::lock(&self.client);
        if client.is_none() {
            *client = Some(
                build_server::Client::connect(&self.socket).map_err(|error| PolicyError::Load {
                    message: format!(
                        "cannot reach the Build Server at {}: {error}",
                        self.socket.display()
                    ),
                })?,
            );
        }
        let client = client.as_mut().ok_or_else(|| PolicyError::Load {
            message: "the Build Server connection was closed".to_owned(),
        })?;
        let answer = ask(client);
        if answer.is_err() {
            // A failed exchange may leave the stream out of step; the next
            // precopy opens a new one.
            *crate::lock(&self.client) = None;
        }
        answer
    }
}

impl PolicySource for ModuleSource {
    fn bytes(&self, package: &PolicyPackage) -> Result<Vec<u8>, PolicyError> {
        let package_revision = package.to_package_revision()?;
        // The Build Server builds a policy module its store does not hold, and it streams what it
        // is doing to the callback below. Discarding that account left a 300-second wait that named
        // no step; a resolve is the one step of a precopy that compiles, so it is printed with the
        // target it asked for - the daemon's prepare and this engine's resolve must name the same
        // one, or the store the daemon filled is not the store this resolve reads.
        let started = Instant::now();
        tracing::info!(
            backend = %package.backend,
            revision = package.revision,
            target = %self.target,
            "resolving the policy package for this host's target"
        );
        let module = self.with_client(|client| {
            client
                .resolve(
                    POLICY_OP,
                    &package_revision,
                    &self.target,
                    build_server::manifest::SINGLE_KEY,
                    &mut |progress| {
                        tracing::info!(
                            target = %self.target,
                            path = %progress.path,
                            state = ?progress.state,
                            "the Build Server is building this policy's module"
                        );
                    },
                )
                .map_err(|error| PolicyError::Load {
                    message: format!(
                        "the Build Server did not resolve the {}/{} policy for {}: {error}",
                        package.backend, package.revision, self.target
                    ),
                })
        })?;
        tracing::info!(
            backend = %package.backend,
            revision = package.revision,
            target = %self.target,
            ms = started.elapsed().as_millis() as u64,
            files = module.files.len(),
            "the policy package resolved"
        );
        // A module built for another version of the policy interface would be asked decisions
        // through calls whose messages mean something else to it.
        if module.interface_version != scheduling::guest::INTERFACE_VERSION {
            return Err(PolicyError::Load {
                message: format!(
                    "the {}/{} policy implements version {} of the scheduling policy interface \
                     and this engine instantiates version {}",
                    package.backend,
                    package.revision,
                    module.interface_version,
                    scheduling::guest::INTERFACE_VERSION
                ),
            });
        }
        let mut file = module
            .files
            .into_iter()
            .find(|file| file.name == build_server::source::MODULE_WASM)
            .ok_or_else(|| PolicyError::Load {
                message: format!(
                    "the {}/{} policy's module carries no {}",
                    package.backend,
                    package.revision,
                    build_server::source::MODULE_WASM
                ),
            })?
            .file;
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut bytes).map_err(|error| PolicyError::Load {
            message: format!(
                "cannot read the {}/{} policy module: {error}",
                package.backend, package.revision
            ),
        })?;
        Ok(bytes)
    }
}

/// A policy module a caller already holds in memory: a store's answer, read
/// once, that keeps handing out the same bytes.
#[derive(Clone, Debug, Default)]
pub struct PolicyBytes {
    bytes: Vec<u8>,
}

impl PolicyBytes {
    /// The source of the module `bytes`.
    pub fn new(bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            bytes: bytes.into(),
        }
    }
}

impl PolicySource for PolicyBytes {
    fn bytes(&self, _package: &PolicyPackage) -> Result<Vec<u8>, PolicyError> {
        Ok(self.bytes.clone())
    }
}

/// Everything a policy is instantiated with, and every state argument of its
/// decisions, as the mechanism holds it at one moment (§4.4).
///
/// It is owned so it can be built on the scheduler thread at a step boundary
/// and handed to a blocking thread for a precopy, and so a revert instantiates
/// the previous policy on the state it is handed rather than on a capture taken
/// when the failing policy was installed. The mechanism builds it; the switch
/// only reads it.
#[derive(Clone, Debug, Default)]
pub struct TrialState {
    /// Every object that exists: the `Snapshot` of `Policy::init`.
    pub snapshot: Snapshot,
    /// The step's room, for `dispatch`.
    pub room: Room,
    /// The shortage the mechanism faces, for `reclaim`; `None` skips the call,
    /// as the mechanism calls `reclaim` only when something is short.
    pub shortage: Option<Shortage>,
    /// A queued request and the ranks, for `route`; `None` skips the call.
    pub arrival: Option<(RequestView, Vec<RankView>)>,
    /// A request's first call on a block and the replicas, for `place`; `None`
    /// skips the call.
    pub call: Option<(CallView, Vec<ReplicaView>)>,
}

/// The mechanism's validation of a decision's answer, without executing it
/// (§4.4, §7.3 step 2).
///
/// The validators are the mechanism's own — the same ones that gate a live
/// decision — so a candidate that passes a precopy is judged by exactly what
/// will judge it once it serves. They are grouped here so the switch can run
/// them on a candidate; the mechanism implements this for its own validators.
pub trait Validators {
    /// Whether `plan` names only objects that exist and are selectable, and
    /// fits the step's hard limits.
    fn plan(&self, plan: &Plan) -> Result<(), PolicyError>;

    /// Whether every victim is evictable for `shortage`, and legal for it.
    fn victims(&self, shortage: &Shortage, victims: &[Victim]) -> Result<(), PolicyError>;

    /// Whether `index` is a rank that may serve `request`.
    fn route(
        &self,
        request: &RequestView,
        ranks: &[RankView],
        index: usize,
    ) -> Result<(), PolicyError>;

    /// Whether `index` is a replica that may run `call`.
    fn place(
        &self,
        call: &CallView,
        replicas: &[ReplicaView],
        index: usize,
    ) -> Result<(), PolicyError>;
}

/// A stage of §7.3's switch, asked for by the daemon and applied by the
/// scheduler at a batch boundary where no batch and no submission is in flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// §7.3 step 3: the instance a precopy prepared becomes the serving one; the
    /// module it replaces is kept to revert to.
    Switchover,
    /// §7.3 step 4: the previous policy is instantiated on the current state
    /// and becomes the serving one.
    Revert,
    /// §7.3 step 6: the previous module is dropped, once no batch uses it.
    Purge,
    /// §7.3 step 5: the update is committed. The previous module stays until
    /// the next precopy, which replaces it.
    Commit,
}

/// A policy instantiated on the state that existed when its precopy ran,
/// waiting for the switchover that puts it in (§7.3 step 2).
///
/// Owned, so the precopy can run on a blocking thread and hand the result back:
/// the scheduler thread keeps the serving policy throughout, and neither thread
/// borrows the other's slot.
pub struct Candidate {
    package: PolicyPackage,
    module: PolicyModule,
    instance: PolicyInstance,
    /// The guest calls the trial made, which the counters carry once the
    /// candidate is adopted. A trial leaves the package as it found it, so the
    /// candidate serves on a second instance and the trialled one's counters
    /// are read here before it is dropped.
    trial: TrialCost,
    /// The snapshot the instance was instantiated on: the state its own
    /// knowledge starts from, which the mechanism seeds the recorder with when
    /// the candidate starts serving (§4.4).
    state: Snapshot,
}

impl Candidate {}

/// What an epoch's policy has cost so far, in the two counters `Metrics`
/// carries for it (design §12's deterministic proxy for per-step cost).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PolicyUsage {
    /// Guest calls made since the epoch's policy first served, a precopy's
    /// trial included.
    pub calls: u64,
    /// Fuel those calls consumed, summed over every instance the epoch's policy
    /// has had: a switchover carries its predecessor's total forward, so the
    /// counters only ever grow.
    pub fuel: u64,
    /// Those calls by decision, the instance serving now and every one it
    /// replaced (§4.4).
    ///
    /// This is the epoch's call discipline as a caller can see it: one `observe`
    /// and one `dispatch` per step whatever they answer, each optional decision
    /// only where the state names it, and the trial a precopy ran on the
    /// instance a switchover adopts. A policy that replaced another carries the
    /// replaced instance's counts forward, as `calls` and `fuel` do.
    ///
    /// The one difference from `calls`: a precopy whose candidate a later
    /// precopy displaced is in `calls` and not here, since no instance of the
    /// epoch's policy ever served under it. Its decision set is still the
    /// mechanism's - it is a trial - but it is not a step the epoch took.
    pub decisions: scheduling::PolicyCost,
}

/// What one scheduler tells the control thread's counters on a statistics tick:
/// the step it reached, and what its policy has cost (§12).
///
/// The step is the epoch's own count — the scheduler takes one per policy step
/// (`Scheduler::policy_step`) — so the engine's largest is the epoch's step
/// count, while each batch a step submits is counted separately by the batch
/// counters.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    /// The step this scheduler reached; zero before its first policy step.
    pub step: u64,
    /// The calls and fuel of the scheduler's policy, or `None` before its first
    /// switchover.
    pub usage: Option<PolicyUsage>,
}

/// The running cost of the instance a slot serves, and of the ones it replaced.
#[derive(Debug, Default)]
struct Cost {
    calls: u64,
    /// Fuel of the instances that stopped serving, and of the trials whose
    /// candidates were adopted.
    spent: u64,
    /// The decisions of the instances that stopped serving; the serving
    /// instance's own are read from it, as its fuel is.
    decisions: scheduling::PolicyCost,
}

/// What a trial cost: the instance's decisions and the fuel it burned.
///
/// The instance that made them is dropped once the candidate exists, so these
/// are kept beside it and added to the counters when it is adopted (§7.3).
#[derive(Clone, Copy, Debug, Default)]
struct TrialCost {
    decisions: scheduling::PolicyCost,
    fuel: u64,
}

impl Cost {
    /// Counts one call into `instance` around `call`.
    fn call<T>(
        &mut self,
        instance: &mut PolicyInstance,
        call: impl FnOnce(&mut PolicyInstance) -> Result<T, PolicyError>,
    ) -> Result<T, PolicyError> {
        self.calls += 1;
        call(instance)
    }

    /// The counters as they stand, the serving instance's fuel included.
    fn usage(&self, instance: &PolicyInstance) -> PolicyUsage {
        PolicyUsage {
            calls: self.calls,
            fuel: self.spent + instance.fuel_consumed(),
            decisions: self.decisions + instance.cost(),
        }
    }

    /// An instance stops serving: its fuel stays in the total.
    fn replace(&mut self, instance: &PolicyInstance) {
        self.spent += instance.fuel_consumed();
        self.decisions = self.decisions + instance.cost();
    }

    /// A trial's calls are counted when its candidate is adopted: they are
    /// calls the epoch's policy made, whether or not it goes on serving under
    /// that package.
    ///
    /// The trialled instance is the record, not a count taken beside it: its
    /// instantiation, the limits it read and every decision the trial asked
    /// are all in its cost, which [`Candidate`] carries because the trial's
    /// instance is dropped. Adding it to the decisions as well as to the call
    /// count keeps `calls` and the sum of the instances' decisions the same
    /// number by construction.
    fn trial(&mut self, trial: TrialCost, serving: &PolicyInstance) {
        // The serving instance's own calls before it serves - its instantiation
        // and the limits it read - are counted here once: `usage` reads the
        // instance's cost live into the decisions, and `calls` has no other
        // sight of them, so without this the two totals differ by exactly them.
        self.calls += trial.decisions.each().map(|(_, calls)| calls).sum::<u64>()
            + serving.cost().each().map(|(_, calls)| calls).sum::<u64>();
        self.decisions = self.decisions + trial.decisions;
        self.spent += trial.fuel;
    }
}

impl Candidate {
    /// The package the candidate was loaded from.
    pub fn package(&self) -> &PolicyPackage {
        &self.package
    }
}

/// A policy that failed, and the policy that serves in its place (§7.3).
///
/// An engine-level report, not a request event: the control thread forwards it
/// to every daemon, which records the previous policy as the epoch's current
/// one, so a later restart loads that one. The report does not say the epoch is
/// fatal; the thread forwarding it knows that from [`PolicySlot::fail`] and
/// [`RoutePolicy::fail`] returning `Err`, which means no previous policy exists
/// to serve under.
#[derive(Clone, Debug)]
pub struct PolicyReverted {
    /// The scheduler whose instance failed; `None` while the scheduler thread
    /// reports it, since a scheduler does not know its index, and `None` for the
    /// control thread's route instance. The control thread names the scheduler
    /// it took a report from.
    pub scheduler: Option<usize>,
    /// The package that failed.
    pub package: PolicyPackage,
    /// Why it failed (§4.4's list: a trap, fuel, bad limits, an invalid
    /// decision, or starvation).
    pub reason: String,
    /// Whether the epoch cannot serve on: the failed policy had no previous
    /// policy to revert to, or the previous policy did not instantiate. It is
    /// set by the thread that failed the epoch, from [`PolicySlot::fail`] or
    /// [`RoutePolicy::fail`] returning `Err`; a report at all therefore says the
    /// policy failed, and `fatal` says whether the epoch did.
    pub fatal: bool,
}

impl PolicyReverted {}

/// A request that has just reached the main process, as the control thread
/// sees it before any scheduler does: what the `route` decision orders by
/// (§4.4).
///
/// The features are read from the submitted request and the routing table, not
/// from a scheduler's state: the request is not on a scheduler yet.
#[derive(Clone, Debug)]
pub struct Arrival {
    /// The request's identity.
    pub id: ipc::RequestId,
    /// The priority the request carries.
    pub priority: i32,
    /// Whether it is token generation or media work.
    pub class: scheduling::guest::view::RequestClass,
    /// Tokens of its prompt.
    pub prompt_tokens: u32,
    /// Tokens it already generated on the epochs that served it before.
    pub generated_tokens: u32,
    /// Tokens it may still generate.
    pub remaining_tokens: u32,
    /// Whether it resumes from its generated tokens.
    pub migratable: bool,
    /// The rank its stored cache was resumed from, when it resumes one; the
    /// policy returns it and the mechanism skips routing the request.
    pub cache_rank: Option<usize>,
    /// When it reached the main process, on the mechanism's clock.
    pub arrived_at: scheduling::guest::time::Nanos,
}

impl Arrival {
    /// The features of a submitted request, arriving now; `cache_rank` is the
    /// rank its stored cache was resumed from, when it resumes one.
    pub fn of(
        id: ipc::RequestId,
        request: &proto::request::Request,
        cache_rank: Option<usize>,
    ) -> Self {
        use proto::request::Request;
        use scheduling::guest::view::RequestClass;

        let prompt_tokens = match request {
            Request::Generation(request) => request.prompt_token_ids.len(),
            Request::Diffusion(request) => request.prompt_token_ids.len(),
        };
        let (class, generated_tokens, remaining_tokens, migratable) = match request {
            Request::Generation(request) => (
                RequestClass::Token,
                request.generated_token_ids.len(),
                u32::try_from(request.max_und_tokens).unwrap_or(u32::MAX),
                !request.generated_token_ids.is_empty(),
            ),
            Request::Diffusion(request) => (
                RequestClass::Media,
                0,
                0,
                request.trajectory.denoised_steps > 0,
            ),
        };
        Self {
            id,
            priority: request.priority(),
            class,
            prompt_tokens: u32::try_from(prompt_tokens).unwrap_or(u32::MAX),
            generated_tokens: u32::try_from(generated_tokens).unwrap_or(u32::MAX),
            remaining_tokens,
            migratable,
            cache_rank,
            arrived_at: scheduling::guest::Nanos(
                crate::clock::now_monotonic_us().saturating_mul(1_000),
            ),
        }
    }
}

/// The view of a request that has just arrived (`Event::Arrived`): nothing has
/// happened to it yet, so every feature but the submitted ones is empty.
pub fn arrival_view(arrival: &Arrival) -> RequestView {
    RequestView {
        id: scheduling::guest::ids::RequestId(arrival.id.0),
        state: RequestState::Queued,
        phase: None,
        priority: arrival.priority,
        arrived_at: arrival.arrived_at,
        admitted_at: None,
        prompt_tokens: arrival.prompt_tokens,
        prefill_tokens: arrival.prompt_tokens,
        generated_tokens: arrival.generated_tokens,
        cached_prefix_tokens: 0,
        prefilled_tokens: 0,
        cached_pages: 0,
        last_emitted_at: None,
        class: arrival.class,
        waiting_transfer: false,
        next_tokens: 0,
        reservation_units: 0,
        prefetch_tokens: 0,
        refill_tokens: None,
        remaining_tokens: arrival.remaining_tokens,
        migratable: arrival.migratable,
        cache_rank: arrival.cache_rank,
    }
}

/// One scheduler thread's policy instance and the switch that replaces it
/// (§4.4, §7.3).
///
/// The module is held beside the instance because a switchover keeps the module
/// it replaces for a revert; only a module, never a stale instance, is kept
/// (§4.4's `previous: Option<PolicyModule>`).
pub struct PolicySlot {
    engine: Engine,
    ceilings: Limits,
    /// The package the serving instance was instantiated from.
    package: PolicyPackage,
    /// The serving instance's module, kept so the next switchover can retain it.
    module: PolicyModule,
    /// The serving instance.
    instance: PolicyInstance,
    /// The module and package of the policy a revert puts back. `None` before
    /// the first switchover and after a revert: no older policy exists then.
    previous: Option<(PolicyPackage, PolicyModule)>,
    /// The instance, module and state a precopy prepared, put in by the
    /// switchover.
    candidate: Option<Candidate>,
    /// The stage asked for and not yet applied, applied at the next boundary.
    pending: Option<Stage>,
    /// The snapshot the serving instance was instantiated on: the state its own
    /// knowledge starts from (§4.4).
    state: Snapshot,
    /// What this scheduler's policy has cost (§12).
    cost: Cost,
}

impl PolicySlot {
    /// The slot a first switchover puts in place from the candidate its precopy
    /// prepared: §7.3's policy for a fresh epoch, whose precopy ran before any
    /// policy served.
    pub fn from_candidate(engine: &Engine, ceilings: Limits, candidate: Candidate) -> Self {
        // The epoch's first policy serves under the trial its precopy ran, so
        // those calls are the epoch's policy's, counted as an adopted
        // candidate's are (§12): a counter that dropped them would report an
        // epoch whose first policy never called the guest.
        let mut cost = Cost::default();
        cost.trial(candidate.trial, &candidate.instance);
        Self {
            engine: engine.clone(),
            ceilings,
            package: candidate.package,
            module: candidate.module,
            instance: candidate.instance,
            previous: None,
            candidate: None,
            pending: None,
            state: candidate.state,
            cost,
        }
    }

    /// Loads `package`'s module from `source` and instantiates it on `state`
    /// (§7.3: a policy put in by a switch from another policy).
    ///
    /// The policy's limits are read once here and checked against `ceilings`; a
    /// limit outside them is a policy failure and the epoch does not serve under
    /// it.
    pub fn install(
        engine: &Engine,
        source: &dyn PolicySource,
        package: PolicyPackage,
        state: &TrialState,
        ceilings: &Limits,
    ) -> Result<Self, PolicyError> {
        let init = init_of(&package, state);
        let module = PolicyModule::load(engine, &source.bytes(&package)?)?;
        let instance = module.instantiate(&init.state, &init.parameters, ceilings)?;
        Ok(Self {
            engine: engine.clone(),
            ceilings: *ceilings,
            package,
            module,
            instance,
            previous: None,
            candidate: None,
            pending: None,
            state: init.state,
            cost: Cost::default(),
        })
    }

    /// The snapshot the serving instance was instantiated on: the state the
    /// policy knows, which the mechanism's recorder is seeded from when this
    /// instance starts serving (§4.4).
    pub fn instantiated_on(&self) -> &Snapshot {
        &self.state
    }

    /// The package the slot serves now.
    pub fn package(&self) -> &PolicyPackage {
        &self.package
    }

    /// The policy's limits, as declared at instantiation and checked against
    /// the build config.
    pub fn limits(&self) -> Limits {
        self.instance.limits()
    }

    /// `observe`: every event since the previous step, in order (§4.4).
    pub fn observe(&mut self, events: &[Event]) -> Result<(), PolicyError> {
        let (instance, cost) = (&mut self.instance, &mut self.cost);
        cost.call(instance, |instance| instance.observe(events))
    }

    /// `dispatch`: this step's plan (§4.4).
    pub fn dispatch(&mut self, room: &Room) -> Result<Plan, PolicyError> {
        let (instance, cost) = (&mut self.instance, &mut self.cost);
        cost.call(instance, |instance| instance.dispatch(room))
    }

    /// `reclaim`: which objects to free, in order, for a shortage (§4.4).
    pub fn reclaim(&mut self, shortage: &Shortage) -> Result<Vec<Victim>, PolicyError> {
        let (instance, cost) = (&mut self.instance, &mut self.cost);
        cost.call(instance, |instance| instance.reclaim(shortage))
    }

    /// `route`: the rank that serves a request that has just arrived (§4.4).
    /// Only the [`Router`]'s slot is asked.
    pub fn route(
        &mut self,
        request: &RequestView,
        ranks: &[RankView],
    ) -> Result<usize, PolicyError> {
        let (instance, cost) = (&mut self.instance, &mut self.cost);
        cost.call(instance, |instance| instance.route(request, ranks))
    }

    /// `place`: which replica runs a request's first call on a block (§4.4).
    pub fn place(
        &mut self,
        call: &CallView,
        replicas: &[ReplicaView],
    ) -> Result<usize, PolicyError> {
        let (instance, cost) = (&mut self.instance, &mut self.cost);
        cost.call(instance, |instance| instance.place(call, replicas))
    }

    /// The calls this scheduler's policy has made and the fuel they consumed
    /// (§12's cost counters, for `Metrics.policy_fuel`/`policy_calls`).
    pub fn usage(&self) -> PolicyUsage {
        self.cost.usage(&self.instance)
    }

    /// §7.3 step 2: instantiate `package` on `state` and ask it every decision
    /// of §4.4 once, executing nothing.
    ///
    /// An answer that does not decode, a trap, fuel exhaustion, limits outside
    /// the build config, a decision that names something that does not exist or
    /// cannot be chosen all fail the precopy: no candidate is returned and the serving policy is
    /// untouched. `validators` are the mechanism's own, so the candidate is
    /// judged by what will judge it live.
    ///
    /// A static function so it runs on a blocking thread and its [`Candidate`]
    /// is handed back to the scheduler thread, which keeps the serving slot
    /// throughout.
    pub fn prepare(
        engine: &Engine,
        source: &dyn PolicySource,
        package: PolicyPackage,
        state: TrialState,
        ceilings: &Limits,
        validators: &dyn Validators,
    ) -> Result<Candidate, PolicyError> {
        let TrialState {
            snapshot,
            room,
            shortage,
            arrival,
            call,
        } = state;
        let init = Instantiation {
            parameters: package.arguments.clone(),
            state: snapshot,
        };
        // Three blocking steps and one trial, timed separately: a wait that names none of them
        // cannot be told from a hang, and the resolve is the only one that compiles.
        let resolving = Instant::now();
        let bytes = source.bytes(&package)?;
        let resolve = resolving.elapsed();
        let loading = Instant::now();
        let module = PolicyModule::load(engine, &bytes)?;
        let load = loading.elapsed();
        let instantiating = Instant::now();
        let mut instance = module.instantiate(&init.state, &init.parameters, ceilings)?;
        let instantiate = instantiating.elapsed();
        let trying = Instant::now();

        // Every decision of §4.4's table once, on the state that exists now and
        // without executing any of it. The calls are the trial's cost, counted
        // once the candidate is adopted; `TrialCost` records them below.
        let plan = instance.dispatch(&room)?;
        validators.plan(&plan)?;
        if let Some(shortage) = &shortage {
            let victims = instance.reclaim(shortage)?;
            validators.victims(shortage, &victims)?;
        }
        if let Some((request, ranks)) = &arrival {
            let index = instance.route(request, ranks)?;
            validators.route(request, ranks, index)?;
        }
        if let Some((call, replicas)) = &call {
            let index = instance.place(call, replicas)?;
            validators.place(call, replicas, index)?;
        }

        // A trial must leave a stateful package as it found it. The calls the
        // trial made are its own, and a package that decides from its own call
        // history has spent them: serving must start from the package's `init`
        // on this boundary's state, not from what the trial left behind. The
        // candidate is therefore a second instance of the same module on that
        // state, and the trialled one is dropped with the cost it counted.
        let trial = TrialCost {
            decisions: instance.cost(),
            fuel: instance.fuel_consumed(),
        };
        tracing::info!(
            backend = %package.backend,
            revision = package.revision,
            resolve_ms = resolve.as_millis() as u64,
            load_ms = load.as_millis() as u64,
            instantiate_ms = instantiate.as_millis() as u64,
            trial_ms = trying.elapsed().as_millis() as u64,
            calls = trial.decisions.each().map(|(_, calls)| calls).sum::<u64>(),
            "the spans of this precopy's policy preparation"
        );
        let instance = module.instantiate(&init.state, &init.parameters, ceilings)?;
        Ok(Candidate {
            package,
            module,
            instance,
            trial,
            state: init.state,
        })
    }

    /// Keeps a candidate a precopy prepared, for the switchover to put in.
    ///
    /// A newer candidate replaces an older one that never reached its
    /// switchover: only a later precopy's result can be the one the daemon's
    /// switchover means.
    pub fn adopt(&mut self, candidate: Candidate) {
        self.cost.trial(candidate.trial, &candidate.instance);
        self.candidate = Some(candidate);
    }

    /// A precopy on the calling thread: [`Self::prepare`] and [`Self::adopt`].
    pub fn precopy(
        &mut self,
        source: &dyn PolicySource,
        package: PolicyPackage,
        state: TrialState,
        validators: &dyn Validators,
    ) -> Result<(), PolicyError> {
        let candidate = Self::prepare(
            &self.engine,
            source,
            package,
            state,
            &self.ceilings,
            validators,
        )?;
        self.adopt(candidate);
        Ok(())
    }

    /// Drops an update the precopy prepared, leaving the serving policy as it
    /// is (§7.3: a failed or abandoned precopy discards the new generation).
    pub fn discard(&mut self) {
        self.candidate = None;
    }

    /// Whether a precopy prepared an instance for the switchover.
    pub fn has_candidate(&self) -> bool {
        self.candidate.is_some()
    }

    /// Asks for a stage of §7.3 to be applied at the next batch boundary.
    ///
    /// A newer request replaces an older one that was not applied yet: the
    /// daemon's stages arrive in order, and only the latest can be the one the
    /// mechanism has not reached.
    pub fn request(&mut self, stage: Stage) {
        self.pending = Some(stage);
    }

    /// Applies the stage a daemon asked for at a batch boundary, where no batch
    /// and no submission is in flight (§7.3 steps 3 to 6).
    ///
    /// `Some` is a policy failure found while applying the stage — a revert
    /// whose previous policy could not be instantiated is not one, it is an
    /// error — so a caller that only changes stages gets `None`; the failure
    /// path of a serving policy is [`Self::fail`].
    pub fn boundary(&mut self, state: &TrialState) -> Result<Option<PolicyReverted>, PolicyError> {
        match self.pending.take() {
            None | Some(Stage::Commit) => Ok(None),
            Some(Stage::Switchover) => {
                self.switch_over(state)?;
                Ok(None)
            }
            Some(Stage::Revert) => {
                self.revert(state)?;
                Ok(None)
            }
            Some(Stage::Purge) => {
                self.previous = None;
                Ok(None)
            }
        }
    }

    /// §7.3 step 3: the prepared module serves from this boundary, instantiated
    /// on `state`, and the module it replaces is kept to revert to.
    ///
    /// The precopy proved the module on the state of its own moment, and the
    /// instance it left has seen no event since: under traffic the pages and
    /// requests moved on while the update ran its shadow step. So the module is
    /// instantiated again here, as a revert instantiates the previous one, and
    /// the instance and the mechanism start from the same state.
    ///
    /// A switchover without a prepared candidate fails: it would install no
    /// policy while the daemon believes the update cut over.
    fn switch_over(&mut self, state: &TrialState) -> Result<(), PolicyError> {
        let Some(candidate) = self.candidate.take() else {
            return Err(PolicyError::invalid(
                Call::Init,
                "a switchover arrived without a policy prepared by a precopy",
            ));
        };
        let init = init_of(&candidate.package, state);
        let instance =
            candidate
                .module
                .instantiate(&init.state, &init.parameters, &self.ceilings)?;
        let replaced_package = std::mem::replace(&mut self.package, candidate.package);
        let replaced_module = std::mem::replace(&mut self.module, candidate.module);
        let replaced = std::mem::replace(&mut self.instance, instance);
        self.state = init.state;
        self.cost.replace(&replaced);
        // §7.3 step 5: the replaced module stays until the next precopy, which
        // replaces it; committing therefore keeps it, and a revert finds it.
        self.previous = Some((replaced_package, replaced_module));
        Ok(())
    }

    /// §7.3 step 4: the previous policy is instantiated on `state` and becomes
    /// the serving one; a candidate a precopy prepared is dropped.
    ///
    /// Re-instantiated rather than kept running: the instance that served
    /// before the switchover did not see the events since, so its state is
    /// stale, and §7.3's revert is a reverse switchover on the current state.
    ///
    /// `Err` names why the revert could not be made: no previous module is
    /// kept, or it does not instantiate on this state. The epoch then has no
    /// policy to serve under.
    fn revert(&mut self, state: &TrialState) -> Result<(), PolicyError> {
        self.candidate = None;
        let Some((package, module)) = self.previous.take() else {
            return Err(PolicyError::invalid(
                Call::Init,
                "the failed policy has no previous policy module to revert to",
            ));
        };
        let init = init_of(&package, state);
        let instance = module.instantiate(&init.state, &init.parameters, &self.ceilings)?;
        let _ = std::mem::replace(&mut self.package, package);
        let _ = std::mem::replace(&mut self.module, module);
        let replaced = std::mem::replace(&mut self.instance, instance);
        self.state = init.state;
        self.cost.replace(&replaced);
        Ok(())
    }

    /// A decision failed: serve on under the previous policy and answer the
    /// report the control thread hands to the daemons (§7.3, "提交之后调度策略
    /// 出错").
    ///
    /// The previous policy is instantiated on `state`, the state at the moment
    /// of the failure, so a policy that fails long after the commit takes over
    /// what the mechanism holds then. `Err` means there is no previous policy
    /// to serve under: the epoch cannot go on and the mechanism stops it.
    pub fn fail(
        &mut self,
        error: &PolicyError,
        state: &TrialState,
    ) -> Result<PolicyReverted, PolicyError> {
        let report = PolicyReverted {
            scheduler: None,
            package: self.package.clone(),
            reason: error.to_string(),
            fatal: false,
        };
        self.revert(state)?;
        Ok(report)
    }
}

/// The instantiation message of `package` on `state`.
fn init_of(package: &PolicyPackage, state: &TrialState) -> Instantiation {
    Instantiation {
        parameters: package.arguments.clone(),
        state: state.snapshot.clone(),
    }
}

/// The mechanism's validation of the control thread's only decision.
///
/// The route instance answers `route` alone: a plan, victims and a placement
/// are a scheduler thread's decisions, judged there. A route decision must name
/// a rank that was offered and is alive.
pub struct RouteTrial;

impl Validators for RouteTrial {
    fn plan(&self, _plan: &Plan) -> Result<(), PolicyError> {
        Ok(())
    }

    fn victims(&self, _shortage: &Shortage, _victims: &[Victim]) -> Result<(), PolicyError> {
        Ok(())
    }

    fn route(
        &self,
        request: &RequestView,
        ranks: &[RankView],
        index: usize,
    ) -> Result<(), PolicyError> {
        crate::mechanism::validate_route(request, ranks, index)
    }

    fn place(
        &self,
        _call: &CallView,
        _replicas: &[ReplicaView],
        _index: usize,
    ) -> Result<(), PolicyError> {
        Ok(())
    }
}

/// The policy instance that answers `route` for every data-parallel rank of a
/// main process (§4.4).
///
/// # Which instance routes
///
/// Routing is decided on the engine's control thread, before any scheduler
/// holds the request, so the instance that answers it is the control thread's
/// own: an instance of the epoch's policy package, prepared, switched over,
/// committed and reverted with the schedulers' instances, so all of them run
/// one package (§4.4: "两者是同一个 package，一起换代"). It is the only
/// instance asked `route`; a scheduler's instance never is.
///
/// The route instance is instantiated on the ranks alone and told no events:
/// it decides from the arriving request's view and each rank's view (the
/// requests the rank holds and whether its scheduler lives), which the control
/// thread counts from the requests it routed. A decision that names a rank
/// outside the ranks offered, or one whose scheduler stopped, is refused as the
/// policy's failure (`mechanism::validate_route`); the previous package then
/// routes, and the request is refused when none can.
pub struct Router {
    /// The serving instance, `None` until the epoch's first switchover.
    slot: Option<PolicySlot>,
    /// The instance a precopy prepared for the next switchover.
    candidate: Option<Candidate>,
    /// The build config's ceilings the instances are instantiated under.
    ceilings: Limits,
}

/// Why `route` named no rank for a request.
#[derive(Debug)]
pub struct RouteRefusal {
    /// Why no rank was named.
    pub reason: String,
    /// The policy failure the refusal reverted, when a serving policy failed;
    /// `fatal` when no previous package could take over.
    pub reverted: Option<Box<PolicyReverted>>,
}

impl Router {
    /// A router with no policy, which routes nothing until a switchover puts
    /// in the instance a precopy prepared.
    pub fn new(ceilings: Limits) -> Self {
        Self {
            slot: None,
            candidate: None,
            ceilings,
        }
    }

    /// The state a route instance is instantiated on: the ranks it routes
    /// among, and nothing else (the type documentation says why).
    fn state(ranks: &[RankView]) -> TrialState {
        TrialState {
            snapshot: Snapshot {
                ranks: ranks.to_vec(),
                ..Snapshot::default()
            },
            ..TrialState::default()
        }
    }

    /// §7.3 step 2 for the route instance: loads `package` and instantiates
    /// it on `ranks`, executing nothing.
    ///
    /// A static function, so the module's compilation runs on a thread of
    /// the caller's choosing; [`Self::adopt`] keeps the answer.
    pub fn prepare(
        source: &dyn PolicySource,
        package: PolicyPackage,
        ranks: &[RankView],
        ceilings: &Limits,
    ) -> Result<Candidate, PolicyError> {
        let engine = process_engine()?;
        PolicySlot::prepare(
            engine,
            source,
            package,
            Self::state(ranks),
            ceilings,
            &RouteTrial,
        )
    }

    /// The ceilings this router instantiates under.
    pub fn ceilings(&self) -> Limits {
        self.ceilings
    }

    /// Keeps the candidate a precopy prepared for the next switchover.
    pub fn adopt(&mut self, candidate: Candidate) {
        self.candidate = Some(candidate);
    }

    /// Drops the candidate a failed or abandoned precopy prepared.
    pub fn discard(&mut self) {
        self.candidate = None;
    }

    /// §7.3 step 3: the prepared instance routes from now on. A switchover
    /// with no prepared instance fails rather than leaving the epoch without
    /// a router.
    pub fn switchover(&mut self, ranks: &[RankView]) -> Result<(), PolicyError> {
        let Some(candidate) = self.candidate.take() else {
            return Err(PolicyError::invalid(
                Call::Init,
                "a route policy switchover arrived without a policy prepared by a precopy",
            ));
        };
        match self.slot.as_mut() {
            Some(slot) => {
                slot.adopt(candidate);
                slot.request(Stage::Switchover);
                slot.boundary(&Self::state(ranks)).map(|_| ())
            }
            None => {
                self.slot = Some(PolicySlot::from_candidate(
                    process_engine()?,
                    self.ceilings,
                    candidate,
                ));
                Ok(())
            }
        }
    }

    /// §7.3 step 5: the update is committed.
    pub fn commit(&mut self) {
        if let Some(slot) = self.slot.as_mut() {
            slot.request(Stage::Commit);
            let _ = slot.boundary(&TrialState::default());
        }
    }

    /// §7.3 step 4: the previous package routes again, instantiated on
    /// `ranks`; a prepared candidate is dropped.
    pub fn revert(&mut self, ranks: &[RankView]) -> Result<(), PolicyError> {
        self.candidate = None;
        let Some(slot) = self.slot.as_mut() else {
            return Ok(());
        };
        slot.request(Stage::Revert);
        slot.boundary(&Self::state(ranks)).map(|_| ())
    }

    /// The calls the serving instance made and the fuel they consumed (§12).
    pub fn usage(&self) -> PolicyUsage {
        self.slot
            .as_ref()
            .map_or_else(PolicyUsage::default, PolicySlot::usage)
    }

    /// The package the serving instance runs, `None` before the first
    /// switchover.
    pub fn package(&self) -> Option<&PolicyPackage> {
        self.slot.as_ref().map(PolicySlot::package)
    }

    /// `route`: the index of the rank in `ranks` that serves `arrival`.
    ///
    /// A node with one scheduler serves every request on that rank and asks
    /// no policy. A request that resumes a stored cache is routed to the rank
    /// recorded for it by the policy itself (`RequestView::cache_rank`), and
    /// the answer is checked like any other. An answer that names no live rank
    /// offered fails the policy: the previous package is instantiated on
    /// `ranks` and routes this request and later ones. When no previous
    /// package can take over, the request is refused and the refusal's
    /// `reverted` is fatal: the update failed.
    pub fn route(&mut self, arrival: &Arrival, ranks: &[RankView]) -> Result<usize, RouteRefusal> {
        if ranks.len() <= 1 {
            return Ok(0);
        }
        let refusal = match self.decide(arrival, ranks) {
            Ok(rank) => return Ok(rank),
            Err(refusal) => refusal,
        };
        match refusal.reverted {
            // The previous package serves again: it routes this request.
            Some(reverted) if !reverted.fatal => {
                tracing::warn!(
                    backend = %reverted.package.backend,
                    revision = reverted.package.revision,
                    reason = %reverted.reason,
                    "the routing policy failed and the previous policy routes again"
                );
                self.decide(arrival, ranks).map_err(|refusal| RouteRefusal {
                    reason: refusal.reason,
                    reverted: None,
                })
            }
            reverted => Err(RouteRefusal {
                reason: refusal.reason,
                reverted,
            }),
        }
    }

    /// One `route` answer of the serving instance, checked; a failure reverts
    /// the instance ([`Self::route`]).
    fn decide(&mut self, arrival: &Arrival, ranks: &[RankView]) -> Result<usize, RouteRefusal> {
        let Some(slot) = self.slot.as_mut() else {
            return Err(RouteRefusal {
                reason: "no routing policy serves: the epoch's first switchover has not put \
                         one in"
                    .to_owned(),
                reverted: None,
            });
        };
        let request = arrival_view(arrival);
        let answer = slot
            .route(&request, ranks)
            .and_then(|index| RouteTrial.route(&request, ranks, index).map(|()| index));
        let error = match answer {
            Ok(index) => return Ok(index),
            Err(error) => error,
        };
        let reverted = match slot.fail(&error, &Self::state(ranks)) {
            Ok(report) => report,
            Err(fatal) => {
                // No previous package can route: the epoch has no router.
                let package = slot.package().clone();
                self.slot = None;
                PolicyReverted {
                    scheduler: None,
                    package,
                    reason: format!("{error}; no previous policy can take over: {fatal}"),
                    fatal: true,
                }
            }
        };
        Err(RouteRefusal {
            reason: format!("the routing policy failed: {error}"),
            reverted: Some(Box::new(reverted)),
        })
    }
}

#[cfg(test)]
mod tests {
    //! The switch of §7.3 over toy policy modules written as WebAssembly text.
    //!
    //! The toys speak the committed call protocol (`scheduling::guest::wire`)
    //! and answer with the interface's own JSON, so the tests exercise the
    //! switch rather than a mock of it: a trap, a fuel run-out, a limit outside
    //! the build config, an invalid decision, and the stages of §7.3.

    use scheduling::guest::decision::{Plan, Shortage, Storage, Victim};
    use scheduling::guest::ids::RequestId;
    use scheduling::guest::time::Nanos;
    use scheduling::guest::view::{RankView, RequestClass, RequestState};

    use super::*;

    use crate::toy::{Toy, fitting_limits, toy_module};

    /// A `PolicySource` of one toy module.
    fn source(toy: &Toy) -> PolicyBytes {
        PolicyBytes::new(toy_module(toy))
    }

    /// The package the tests install.
    fn package(backend: &str, revision: u32) -> PolicyPackage {
        PolicyPackage {
            backend: backend.to_owned(),
            revision,
            arguments: Parameters::default(),
            stated: Default::default(),
        }
    }

    /// A plan that names one request, so two packages' answers differ.
    fn plan_of(request: u64) -> Plan {
        Plan {
            refill: vec![RequestId(request)],
            ..Plan::default()
        }
    }

    /// The ceilings a test's build config states.
    fn ceilings() -> Limits {
        Limits {
            max_batch: 8,
            max_num_batched_tokens: 1024,
            chunked_prefill_size: 512,
            ..Limits::default()
        }
    }

    /// A state whose every optional decision has something to decide about.
    fn state() -> TrialState {
        TrialState {
            snapshot: Snapshot::default(),
            room: Room::default(),
            shortage: Some(Shortage {
                kind: Storage::KvUnits,
                need: 1,
            }),
            arrival: Some((
                RequestView {
                    id: RequestId(7),
                    state: RequestState::Queued,
                    phase: None,
                    priority: 0,
                    arrived_at: Nanos(0),
                    admitted_at: None,
                    prompt_tokens: 8,
                    prefill_tokens: 8,
                    generated_tokens: 0,
                    cached_prefix_tokens: 0,
                    prefilled_tokens: 0,
                    cached_pages: 0,
                    last_emitted_at: None,
                    class: RequestClass::Token,
                    waiting_transfer: false,
                    next_tokens: 0,
                    reservation_units: 0,
                    prefetch_tokens: 0,
                    refill_tokens: None,
                    remaining_tokens: 0,
                    migratable: false,
                    cache_rank: None,
                },
                vec![RankView {
                    index: 0,
                    in_flight: 0,
                    alive: true,
                }],
            )),
            call: None,
        }
    }

    #[test]
    fn the_route_trial_refuses_an_index_that_names_no_live_rank() {
        let ranks = vec![
            RankView {
                index: 0,
                in_flight: 2,
                alive: true,
            },
            RankView {
                index: 1,
                in_flight: 0,
                alive: false,
            },
        ];
        let (request, _) = state().arrival.unwrap();
        assert!(RouteTrial.route(&request, &ranks, 0).is_ok());
        let dead = RouteTrial.route(&request, &ranks, 1).unwrap_err();
        assert!(
            matches!(
                dead,
                PolicyError::Malformed {
                    call: Call::Route,
                    ..
                }
            ),
            "a dead rank is a bad decision, got {dead}"
        );
        let outside = RouteTrial.route(&request, &ranks, 2).unwrap_err();
        assert!(
            matches!(
                outside,
                PolicyError::Malformed {
                    call: Call::Route,
                    ..
                }
            ),
            "an index outside the ranks is a bad decision, got {outside}"
        );
    }

    /// A `Validators` that accepts every answer unless told otherwise.
    #[derive(Default)]
    struct Checks {
        /// Refuse every route answer.
        refuse_route: bool,
        /// Refuse the plan.
        refuse_plan: bool,
    }

    impl Validators for Checks {
        fn plan(&self, _plan: &Plan) -> Result<(), PolicyError> {
            if self.refuse_plan {
                return Err(PolicyError::invalid(Call::Dispatch, "refused"));
            }
            Ok(())
        }

        fn victims(&self, _shortage: &Shortage, _victims: &[Victim]) -> Result<(), PolicyError> {
            Ok(())
        }

        fn route(
            &self,
            _request: &RequestView,
            _ranks: &[RankView],
            _index: usize,
        ) -> Result<(), PolicyError> {
            if self.refuse_route {
                return Err(PolicyError::invalid(Call::Route, "refused"));
            }
            Ok(())
        }

        fn place(
            &self,
            _call: &CallView,
            _replicas: &[ReplicaView],
            _index: usize,
        ) -> Result<(), PolicyError> {
            Ok(())
        }
    }

    /// A slot serving `toy` as package `a`.
    fn slot(toy: &Toy) -> PolicySlot {
        PolicySlot::install(
            &Engine::new().expect("an engine"),
            &source(toy),
            package("a", 1),
            &state(),
            &ceilings(),
        )
        .expect("the toy policy installs")
    }

    /// Switches `slot` to `toy` at a boundary, as a daemon's precopy,
    /// switchover and commit do.
    fn switch_to(slot: &mut PolicySlot, toy: &Toy) {
        slot.precopy(&source(toy), package("b", 2), state(), &Checks::default())
            .expect("the candidate passes its precopy");
        slot.request(Stage::Switchover);
        slot.boundary(&state()).expect("the switchover applies");
        slot.request(Stage::Commit);
        slot.boundary(&state()).expect("the commit applies");
    }

    #[test]
    fn a_slot_serves_the_limits_its_package_declares() {
        let declared = Limits {
            max_batch: 3,
            ..fitting_limits()
        };
        let slot = slot(&Toy {
            limits: declared,
            ..Toy::default()
        });
        assert_eq!(slot.limits().max_batch, 3);
        assert_eq!(slot.package().backend, "a");
    }

    #[test]
    fn a_policy_whose_limits_exceed_the_build_config_does_not_install() {
        let too_large = Limits {
            max_num_batched_tokens: ceilings().max_num_batched_tokens + 1,
            ..fitting_limits()
        };
        let installed = PolicySlot::install(
            &Engine::new().expect("an engine"),
            &source(&Toy {
                limits: too_large,
                ..Toy::default()
            }),
            package("a", 1),
            &state(),
            &ceilings(),
        );
        assert!(matches!(installed, Err(PolicyError::BadLimits { .. })));
    }

    #[test]
    fn a_precopy_prepares_a_candidate_apart_from_the_serving_policy_and_adopting_it_switches() {
        let engine = Engine::new().expect("an engine");
        let mut slot = slot(&Toy {
            plan: plan_of(1),
            ..Toy::default()
        });

        // The precopy runs where the serving slot is not: the serving policy
        // answers throughout, and the candidate is only a value.
        let candidate = PolicySlot::prepare(
            &engine,
            &source(&Toy {
                plan: plan_of(2),
                ..Toy::default()
            }),
            package("b", 2),
            state(),
            &ceilings(),
            &Checks::default(),
        )
        .expect("the candidate passes its precopy");
        assert_eq!(candidate.package().backend, "b");
        assert_eq!(slot.package().backend, "a");
        assert_eq!(
            slot.dispatch(&Room::default()).expect("an answer").refill,
            plan_of(1).refill
        );

        slot.adopt(candidate);
        slot.request(Stage::Switchover);
        assert!(
            slot.boundary(&state())
                .expect("the boundary applies")
                .is_none()
        );
        assert_eq!(slot.package().backend, "b");
        assert_eq!(
            slot.dispatch(&Room::default()).expect("an answer").refill,
            plan_of(2).refill
        );
    }

    #[test]
    fn a_precopy_asks_every_decision_the_state_names() {
        // A trap in each decision's export fails the precopy, which proves the
        // decision was asked: reclaim and route only when the state names
        // them.
        for call in [Call::Dispatch, Call::Reclaim, Call::Route] {
            let mut slot = slot(&Toy::default());
            let candidate = Toy {
                trap: Some(call),
                ..Toy::default()
            };
            let prepared = slot.precopy(
                &source(&candidate),
                package("b", 2),
                state(),
                &Checks::default(),
            );
            assert!(
                matches!(prepared, Err(PolicyError::Trap { call: failed, .. }) if failed == call),
                "{call:?} was not asked: {prepared:?}"
            );
            assert!(!slot.has_candidate());
        }
    }

    #[test]
    fn a_candidate_that_runs_past_its_fuel_fails_the_precopy() {
        let mut slot = slot(&Toy::default());
        let candidate = Toy {
            spin: Some(Call::Dispatch),
            ..Toy::default()
        };
        let prepared = slot.precopy(
            &source(&candidate),
            package("b", 2),
            state(),
            &Checks::default(),
        );
        assert!(matches!(
            prepared,
            Err(PolicyError::FuelExhausted {
                call: Call::Dispatch
            })
        ));
        assert!(!slot.has_candidate());
    }

    #[test]
    fn an_invalid_decision_fails_the_precopy() {
        let mut slot = slot(&Toy::default());
        let refused_plan = slot.precopy(
            &source(&Toy::default()),
            package("b", 2),
            state(),
            &Checks {
                refuse_plan: true,
                ..Checks::default()
            },
        );
        assert!(matches!(
            refused_plan,
            Err(PolicyError::Malformed {
                call: Call::Dispatch,
                ..
            })
        ));
        let refused_route = slot.precopy(
            &source(&Toy::default()),
            package("b", 2),
            state(),
            &Checks {
                refuse_route: true,
                ..Checks::default()
            },
        );
        assert!(matches!(
            refused_route,
            Err(PolicyError::Malformed {
                call: Call::Route,
                ..
            })
        ));
        assert!(!slot.has_candidate());
    }

    #[test]
    fn a_switchover_serves_the_prepared_policy_and_a_commit_keeps_the_one_it_replaced() {
        let mut slot = slot(&Toy {
            plan: plan_of(1),
            ..Toy::default()
        });
        assert_eq!(
            slot.dispatch(&Room::default()).expect("an answer").refill,
            plan_of(1).refill
        );

        slot.precopy(
            &source(&Toy {
                plan: plan_of(2),
                ..Toy::default()
            }),
            package("b", 2),
            state(),
            &Checks::default(),
        )
        .expect("the candidate passes");
        // Until the switchover the serving policy answers.
        assert_eq!(
            slot.dispatch(&Room::default()).expect("an answer").refill,
            plan_of(1).refill
        );
        assert_eq!(slot.package().backend, "a");

        slot.request(Stage::Switchover);
        assert!(
            slot.boundary(&state())
                .expect("the boundary applies")
                .is_none()
        );
        assert_eq!(
            slot.dispatch(&Room::default()).expect("an answer").refill,
            plan_of(2).refill
        );
        assert_eq!(slot.package().backend, "b");

        // The commit keeps the replaced policy, so a later failure reverts.
        slot.request(Stage::Commit);
        assert!(
            slot.boundary(&state())
                .expect("the boundary applies")
                .is_none()
        );
        let report = slot
            .fail(
                &PolicyError::invalid(Call::Dispatch, "the policy answered nothing"),
                &state(),
            )
            .expect("the previous policy is kept");
        assert_eq!(report.package.backend, "b");
        assert_eq!(
            slot.dispatch(&Room::default()).expect("an answer").refill,
            plan_of(1).refill
        );
        assert_eq!(slot.package().backend, "a");
    }

    #[test]
    fn a_failure_with_no_previous_policy_cannot_be_reverted() {
        let mut slot = slot(&Toy::default());
        let failed = slot.fail(&PolicyError::invalid(Call::Dispatch, "refused"), &state());
        assert!(matches!(failed, Err(PolicyError::Malformed { .. })));
    }

    #[test]
    fn a_purge_drops_the_previous_policy_so_a_later_failure_cannot_be_reverted() {
        let mut slot = slot(&Toy::default());
        switch_to(&mut slot, &Toy::default());
        slot.request(Stage::Purge);
        assert!(
            slot.boundary(&state())
                .expect("the boundary applies")
                .is_none()
        );
        let failed = slot.fail(&PolicyError::invalid(Call::Dispatch, "refused"), &state());
        assert!(matches!(failed, Err(PolicyError::Malformed { .. })));
    }

    #[test]
    fn a_daemon_revert_instantiates_the_previous_policy_on_the_current_state() {
        let mut slot = slot(&Toy {
            plan: plan_of(1),
            ..Toy::default()
        });
        switch_to(
            &mut slot,
            &Toy {
                plan: plan_of(2),
                ..Toy::default()
            },
        );
        assert_eq!(
            slot.dispatch(&Room::default()).expect("an answer").refill,
            plan_of(2).refill
        );

        // The state a revert is instantiated on is this one, not the one the
        // precopy saw.
        let mut later = state();
        later.room.step = 9;
        slot.request(Stage::Revert);
        assert!(
            slot.boundary(&later)
                .expect("the boundary applies")
                .is_none()
        );
        assert_eq!(slot.package().backend, "a");
        assert_eq!(
            slot.dispatch(&Room::default()).expect("an answer").refill,
            plan_of(1).refill
        );
    }

    #[test]
    fn a_package_is_built_from_the_arguments_the_daemon_states() {
        let arguments = [
            ("chunked_prefill_size".to_owned(), "2048".to_owned()),
            ("max_batch".to_owned(), "16".to_owned()),
            ("expected_speedup".to_owned(), "2.5".to_owned()),
        ]
        .into_iter()
        .collect();
        let package =
            PolicyPackage::from_arguments("fcfs", 3, &arguments).expect("the arguments parse");
        assert_eq!(package.backend, "fcfs");
        assert_eq!(package.revision, 3);
        assert_eq!(package.arguments.chunked_prefill_size, Some(2048));
        assert_eq!(package.arguments.max_batch, Some(16));
        assert_eq!(package.arguments.expected_speedup, Some(2.5));
    }

    #[test]
    fn the_package_a_precopy_states_is_the_package_the_policy_runs() {
        // The wire form is the Build Server's `PackageRevision`: a name, a revision,
        // and each argument as its TOML text. Nothing about the package is
        // decided by the mechanism on the way.
        let stated = proto::process::PackageRevision {
            backend: "priority".to_owned(),
            revision: 7,
            arguments: [
                ("max_batch".to_owned(), "32".to_owned()),
                ("timeout".to_owned(), "5".to_owned()),
            ]
            .into_iter()
            .collect(),
        };
        let package = PolicyPackage::from_package_revision(&stated).expect("the package parses");
        assert_eq!(package.backend, "priority");
        assert_eq!(package.revision, 7);
        assert_eq!(package.arguments.max_batch, Some(32));
        assert_eq!(
            package.arguments.timeout,
            Some(scheduling::guest::time::Nanos(5))
        );
    }

    #[test]
    fn a_package_argument_this_build_does_not_name_is_refused() {
        let arguments = [("chunked_prefill_size_v2".to_owned(), "2048".to_owned())]
            .into_iter()
            .collect();
        let built = PolicyPackage::from_arguments("fcfs", 3, &arguments);
        assert!(matches!(built, Err(PolicyError::Load { .. })));
    }

    #[test]
    fn the_router_routes_switches_reverts_and_refuses_a_rank_it_was_not_offered() {
        let ranks = vec![
            RankView {
                index: 0,
                in_flight: 0,
                alive: true,
            },
            RankView {
                index: 1,
                in_flight: 0,
                alive: true,
            },
        ];
        let arrival = Arrival {
            id: ipc::RequestId(7),
            priority: 0,
            class: RequestClass::Token,
            prompt_tokens: 8,
            generated_tokens: 0,
            remaining_tokens: 8,
            migratable: false,
            cache_rank: None,
            arrived_at: Nanos(0),
        };
        let mut router = Router::new(ceilings());
        assert!(
            router.route(&arrival, &ranks).is_err(),
            "no policy routes yet"
        );
        assert!(router.switchover(&ranks).is_err(), "nothing was prepared");

        let toy = |route: u32| Toy {
            route,
            ..Toy::default()
        };
        let prepared = Router::prepare(&source(&toy(1)), package("a", 1), &ranks, &ceilings())
            .expect("the candidate prepares");
        router.adopt(prepared);
        router
            .switchover(&ranks)
            .expect("the switchover installs it");
        assert_eq!(router.route(&arrival, &ranks).expect("a rank"), 1);

        // A package that names a rank it was not offered fails, and the
        // package before it routes the request and those after it.
        let prepared = Router::prepare(&source(&toy(5)), package("b", 2), &ranks, &ceilings())
            .expect("the candidate prepares");
        router.adopt(prepared);
        router
            .switchover(&ranks)
            .expect("the switchover installs it");
        assert_eq!(
            router
                .route(&arrival, &ranks)
                .expect("the previous package routes"),
            1
        );
        assert_eq!(router.package().map(|p| p.backend.as_str()), Some("a"));
        assert_eq!(router.route(&arrival, &ranks).expect("a rank"), 1);

        // A node of one scheduler routes every request there and asks no policy.
        assert_eq!(router.route(&arrival, &ranks[..1]).expect("rank 0"), 0);
    }

    #[test]
    fn a_package_resolves_with_the_arguments_the_deployment_stated() {
        // The Build Server resolves a package by backend, revision and the
        // arguments that were stated; it fills the package's defaults for the
        // rest, so the source carries exactly what the daemon named.
        let arguments = std::collections::BTreeMap::from([
            ("max_batch".to_owned(), "16".to_owned()),
            ("timeout".to_owned(), "5000000000".to_owned()),
        ]);
        let built = PolicyPackage::from_arguments("fcfs", 7, &arguments)
            .expect("the arguments are the policy's");
        let server = built
            .to_package_revision()
            .expect("the arguments are TOML values");
        assert_eq!(server.backend, "fcfs");
        assert_eq!(server.revision, 7);
        assert_eq!(server.arguments["max_batch"].as_integer(), Some(16));
        assert_eq!(
            server.arguments["timeout"].as_integer(),
            Some(5_000_000_000)
        );
        assert_eq!(server.arguments.len(), 2);
    }

    #[test]
    fn a_policy_is_resolved_under_the_stores_wasm_key() {
        // A policy is one module per revision, built from its snapshot for the
        // store's wasm target (`Manifest::builds_for` answers that key and no
        // other for a package with no kernel), so a source that named a CPU
        // triple asked for a path no install writes and no package declares.
        let source = ModuleSource::connect("/tmp/none.sock");
        assert_eq!(
            source.target().as_str(),
            build_server::manifest::WASM_TARGET
        );
        assert_eq!(source.socket(), std::path::Path::new("/tmp/none.sock"));
    }
}

/// Each package the tree ships is a deployment's policy module, and its
/// `[parameters]` table is the argument set `Precopy` delivers as TOML text
/// (§5.1, §7.7). A parameter whose spelling this build cannot read fails every
/// deployment that names that package, and it fails at the first `Precopy`
/// rather than anywhere the deployment's configuration could be blamed.
#[cfg(test)]
mod shipped_packages {
    use super::PolicyPackage;

    #[test]
    fn every_package_states_arguments_this_build_reads() {
        let root = std::path::Path::new(proto::WORKSPACE);
        for package in [
            "crates/scheduling/policies/fcfs/package.toml",
            "crates/scheduling/policies/priority/package.toml",
            "crates/scheduling/policies/fixture/package.toml",
        ] {
            let path = root.join(package);
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            // `toml::from_str` is the document parser: `Value`'s own `FromStr` parses one value,
            // so a package - comments, tables and all - fails there at 1:1 with "expected
            // nothing", which reads like an empty file and is not.
            let parsed: toml::Value = toml::from_str(&text).unwrap_or_else(|error| {
                // A parse error at 1:1 on a package that is valid in the tree
                // means the run read an empty file: name the path and its size
                // rather than leave the reader to guess which copy it was.
                panic!(
                    "{}: {error} ({} bytes, read from {})",
                    package,
                    text.len(),
                    path.display()
                )
            });
            let backend = parsed
                .get("backend")
                .and_then(toml::Value::as_str)
                .unwrap_or_else(|| panic!("{package} names no backend"));
            let mut arguments = std::collections::BTreeMap::new();
            for (name, value) in parsed
                .get("parameters")
                .and_then(toml::Value::as_table)
                .into_iter()
                .flatten()
            {
                arguments.insert(name.clone(), value.to_string());
            }
            PolicyPackage::from_arguments(backend, 0, &arguments)
                .unwrap_or_else(|error| panic!("{package}: {error}"));
        }
    }
}
