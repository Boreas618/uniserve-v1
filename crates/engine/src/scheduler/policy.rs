//! The scheduling policy a scheduler thread runs, and the program-layer switch
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
//! Both instances — each scheduler thread's and the control thread's — come
//! from one program and switch together (§4.4: "两者是同一个 program，一起换代").
//! The control thread's instance is told every rank's `Arrived`, `Emitted` and
//! `Finished` as the control thread sees them, so a policy that tracks
//! per-tenant usage routes on what each rank actually served.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use scheduling::guest::decision::{Plan, Room, Shortage, Victim};
use scheduling::guest::event::Event;
use scheduling::guest::ids::PageId;
use scheduling::guest::limits::{Limits, Parameters};
use scheduling::guest::policy::Instantiation;
use scheduling::guest::view::{
    CallView, ExpertLoad, ExpertOrder, RankView, ReplicaView, RequestState, RequestView, Snapshot,
};
use scheduling::guest::wire::Call;
use scheduling::{Engine, PolicyError, PolicyInstance, PolicyModule};
use serde::{Deserialize, Serialize};

/// A policy program's identity and arguments (§5.1): the backend the
/// configuration names, the revision the Program Server numbered the package
/// under, and the arguments that shape its decisions.
///
/// The arguments are part of the program — a policy's parameters are not the
/// mechanism's configuration (§4.4) — so a change of either is a new program
/// and a new program-layer generation (§7.2).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicyProgram {
    /// The backend's name, as `scheduler_config.policy` spells it.
    pub backend: String,
    /// The revision the Program Server numbered the package's install (§5.1).
    pub revision: u32,
    /// The policy's arguments, from the deployment configuration (§7.7).
    pub arguments: Parameters,
    /// The arguments as the daemon stated them, one TOML value each, which is
    /// what the Program Server resolves the module with: the server fills the
    /// package's defaults for the ones the deployment leaves out.
    pub stated: std::collections::BTreeMap<String, String>,
}

impl PolicyProgram {
    /// The program a daemon's `Precopy` states (§7.3 step 2, §5.1): the
    /// backend, the revision, and the policy's arguments as the TOML text of
    /// each configured value.
    ///
    /// The parameter names are the policy's own (§4.4, §7.7): this build knows
    /// the ones [`Parameters`] declares, and an argument it does not know is
    /// refused rather than ignored, because the program's arguments are the
    /// program and serving under fewer of them is not serving under it.
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
                            "the {backend} policy program states `{other}`, which this build's \
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

    /// The program as the Program Server names it (§5.1): the backend, the
    /// revision, and the stated arguments as TOML.
    pub fn to_server_program(&self) -> Result<program_server::Program, PolicyError> {
        let mut arguments = toml::Table::new();
        for (name, value) in &self.stated {
            arguments.insert(name.clone(), argument::<toml::Value>(value, name)?);
        }
        Ok(program_server::Program {
            backend: self.backend.clone(),
            revision: self.revision,
            arguments,
        })
    }

    /// The program a daemon's `Precopy` states for the scheduling policy
    /// (§7.3 step 2): its `program` field, as the process interface carries it.
    ///
    /// The wire form is the Program Server's own `Program` — a backend, a
    /// revision and the arguments as TOML text — so nothing about a program is
    /// decided here: a name the interface does not have is refused, as it is by
    /// [`Self::from_arguments`].
    pub fn from_program(program: &proto::process::Program) -> Result<Self, PolicyError> {
        let arguments: std::collections::BTreeMap<String, String> = program
            .arguments
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        Self::from_arguments(program.backend.clone(), program.revision, &arguments)
    }
}

/// One configured argument of a program, parsed as the type the policy
/// parameter has.
fn argument<T: serde::de::DeserializeOwned>(value: &str, name: &str) -> Result<T, PolicyError> {
    toml::from_str(&format!("value = {value}")).map_err(|error| PolicyError::Load {
        message: format!("the policy argument `{name} = {value}` does not parse: {error}"),
    })
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
    pub fn of(info: &ipc::WorkerInfo) -> Self {
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

/// Where the switch resolves a policy program's module (§5.1, §5.3).
///
/// The daemon's commands and the epoch record name a program by backend,
/// revision and arguments, never by content: the content comes from the
/// Program Server, through this source.
pub trait PolicySource: Send + Sync {
    /// The module bytes of `program`: the artifact the Program Server
    /// precompiled for this host, or the wasm32 binary when it has none.
    fn bytes(&self, program: &PolicyProgram) -> Result<Vec<u8>, PolicyError>;
}

/// The Program Server's client, as the source of a policy's module (§5.3,
/// §7.3 step 2).
///
/// The daemon names a program by backend, revision and arguments, never by
/// content: this asks the host's Program Server to `resolve` it — the store's
/// copy, or a build of it when the store misses it — for the baseline target
/// this engine pinned at start, under the `scheduling` operator's single
/// descriptor ([`program_server::manifest::POLICY_DESCRIPTOR`]).
///
/// One connection is kept and reused: a precopy is the only caller, and it
/// runs on a thread of its own, so the lock is never held across a step.
#[derive(Debug)]
pub struct ProgramSource {
    socket: PathBuf,
    target: program_server::Target,
    client: std::sync::Mutex<Option<program_server::Client>>,
}

/// The operator a scheduling policy's program fills (§5.2).
const POLICY_OP: &str = "scheduling";

impl ProgramSource {
    /// A source that resolves programs through the Program Server listening on
    /// `socket`, for the target this host's engine precompiles for.
    ///
    /// A host without a baseline target has no Program Server artifacts to
    /// load, which is a precopy failure like any other.
    pub fn connect(socket: impl Into<PathBuf>) -> Result<Self, PolicyError> {
        let name = Engine::target().ok_or_else(|| PolicyError::Load {
            message: format!(
                "no baseline wasm target for this host (architecture {}): its Program Server \
                 precompiles none",
                std::env::consts::ARCH
            ),
        })?;
        Ok(Self {
            socket: socket.into(),
            target: program_server::Target::new(name),
            client: std::sync::Mutex::new(None),
        })
    }

    /// The socket the source asks.
    pub fn socket(&self) -> &Path {
        &self.socket
    }

    /// With `self`'s connection open, for the caller to resolve through.
    fn with_client<T>(
        &self,
        ask: impl FnOnce(&mut program_server::Client) -> Result<T, PolicyError>,
    ) -> Result<T, PolicyError> {
        let mut client = crate::lock(&self.client);
        if client.is_none() {
            *client = Some(
                program_server::Client::connect(&self.socket).map_err(|error| {
                    PolicyError::Load {
                        message: format!(
                            "cannot reach the Program Server at {}: {error}",
                            self.socket.display()
                        ),
                    }
                })?,
            );
        }
        let client = client.as_mut().ok_or_else(|| PolicyError::Load {
            message: "the Program Server connection was closed".to_owned(),
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

impl PolicySource for ProgramSource {
    fn bytes(&self, program: &PolicyProgram) -> Result<Vec<u8>, PolicyError> {
        let server_program = program.to_server_program()?;
        let module = self.with_client(|client| {
            client
                .resolve(
                    POLICY_OP,
                    &server_program,
                    &self.target,
                    program_server::manifest::POLICY_DESCRIPTOR,
                    &mut |_| {},
                )
                .map_err(|error| PolicyError::Load {
                    message: format!(
                        "the Program Server did not resolve the {}/{} policy for {}: {error}",
                        program.backend, program.revision, self.target
                    ),
                })
        })?;
        let mut file = module
            .files
            .into_iter()
            .find(|file| file.name == program_server::source::MODULE_WASM)
            .ok_or_else(|| PolicyError::Load {
                message: format!(
                    "the {}/{} policy's module carries no {}",
                    program.backend,
                    program.revision,
                    program_server::source::MODULE_WASM
                ),
            })?
            .file;
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut bytes).map_err(|error| PolicyError::Load {
            message: format!(
                "cannot read the {}/{} policy module: {error}",
                program.backend, program.revision
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
    fn bytes(&self, _program: &PolicyProgram) -> Result<Vec<u8>, PolicyError> {
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
    /// The step's room, for `dispatch` and `warm`.
    pub room: Room,
    /// The shortage the mechanism faces, for `reclaim`; `None` skips the call,
    /// as the mechanism calls `reclaim` only when something is short.
    pub shortage: Option<Shortage>,
    /// A queued request and the ranks, for `route`; `None` skips the call.
    pub arrival: Option<(RequestView, Vec<RankView>)>,
    /// A request's first call on a block and the replicas, for `place`; `None`
    /// skips the call.
    pub call: Option<(CallView, Vec<ReplicaView>)>,
    /// The load the experts are balanced against, for `experts`; `None` skips
    /// the call.
    pub expert_load: Option<ExpertLoad>,
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

    /// Whether every page named may be warmed.
    fn warm(&self, pages: &[PageId]) -> Result<(), PolicyError>;

    /// Whether `order` is a legal placement of the experts of `load`.
    fn experts(&self, load: &ExpertLoad, order: &ExpertOrder) -> Result<(), PolicyError>;
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
    program: PolicyProgram,
    module: PolicyModule,
    instance: PolicyInstance,
    /// The guest calls the trial made, which the counters carry once the
    /// candidate is adopted.
    calls: u64,
}

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
}

/// The running cost of the instance a slot serves, and of the ones it replaced.
#[derive(Debug, Default)]
struct Cost {
    calls: u64,
    /// Fuel of the instances that stopped serving, and of the trials whose
    /// candidates were adopted.
    spent: u64,
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
        }
    }

    /// An instance stops serving: its fuel stays in the total.
    fn replace(&mut self, instance: &PolicyInstance) {
        self.spent += instance.fuel_consumed();
    }

    /// A trial's calls are counted when its candidate is adopted: they are
    /// calls the epoch's policy made, whether or not it goes on serving under
    /// that program. Its fuel is already inside the instance's own total.
    fn trial(&mut self, calls: u64) {
        self.calls += calls;
    }
}

impl Candidate {
    /// The program the candidate was loaded from.
    pub fn program(&self) -> &PolicyProgram {
        &self.program
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
    /// The program that failed.
    pub program: PolicyProgram,
    /// Why it failed (§4.4's list: a trap, fuel, bad limits, an invalid
    /// decision, an illegal expert order, or starvation).
    pub reason: String,
    /// Whether the epoch cannot serve on: the failed policy had no previous
    /// policy to revert to, or the previous policy did not instantiate. It is
    /// set by the thread that failed the epoch, from [`PolicySlot::fail`] or
    /// [`RoutePolicy::fail`] returning `Err`; a report at all therefore says the
    /// policy failed, and `fatal` says whether the epoch did.
    pub fatal: bool,
}

impl PolicyReverted {
    /// The report as the process interface carries it (§7.3).
    ///
    /// A scheduler thread does not know its index, so it reports `None` and the
    /// control thread names it; the control thread's own instance is absent
    /// (`None`) on the wire as well (§7.3).
    pub fn to_process(&self) -> proto::process::PolicyReverted {
        proto::process::PolicyReverted {
            scheduler: self.scheduler.map(|index| index as u32),
            backend: self.program.backend.clone(),
            revision: self.program.revision,
            reason: self.reason.clone(),
            fatal: self.fatal,
        }
    }
}

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
    /// The program the serving instance was instantiated from.
    program: PolicyProgram,
    /// The serving instance's module, kept so the next switchover can retain it.
    module: PolicyModule,
    /// The serving instance.
    instance: PolicyInstance,
    /// The module and program of the policy a revert puts back. `None` before
    /// the first switchover and after a revert: no older policy exists then.
    previous: Option<(PolicyProgram, PolicyModule)>,
    /// The instance and module a precopy prepared, put in by the switchover.
    candidate: Option<(PolicyProgram, PolicyModule, PolicyInstance)>,
    /// The stage asked for and not yet applied, applied at the next boundary.
    pending: Option<Stage>,
    /// What this scheduler's policy has cost (§12).
    cost: Cost,
}

impl PolicySlot {
    /// The slot a first switchover puts in place from the candidate its precopy
    /// prepared: §7.3's policy for a fresh epoch, whose precopy ran before any
    /// policy served.
    pub fn from_candidate(engine: &Engine, ceilings: Limits, candidate: Candidate) -> Self {
        Self {
            engine: engine.clone(),
            ceilings,
            program: candidate.program,
            module: candidate.module,
            instance: candidate.instance,
            previous: None,
            candidate: None,
            pending: None,
            cost: Cost::default(),
        }
    }

    /// Loads `program`'s module from `source` and instantiates it on `state`
    /// (§7.3: a policy put in by a switch from another policy).
    ///
    /// The policy's limits are read once here and checked against `ceilings`; a
    /// limit outside them is a policy failure and the epoch does not serve under
    /// it.
    pub fn install(
        engine: &Engine,
        source: &dyn PolicySource,
        program: PolicyProgram,
        state: &TrialState,
        ceilings: &Limits,
    ) -> Result<Self, PolicyError> {
        let init = init_of(&program, state);
        let module = PolicyModule::load(engine, &source.bytes(&program)?)?;
        let instance = module.instantiate(&init.state, &init.parameters, ceilings)?;
        Ok(Self {
            engine: engine.clone(),
            ceilings: *ceilings,
            program,
            module,
            instance,
            previous: None,
            candidate: None,
            pending: None,
            cost: Cost::default(),
        })
    }

    /// The program the slot serves now.
    pub fn program(&self) -> &PolicyProgram {
        &self.program
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

    /// `place`: which replica runs a request's first call on a block (§4.4).
    pub fn place(
        &mut self,
        call: &CallView,
        replicas: &[ReplicaView],
    ) -> Result<usize, PolicyError> {
        let (instance, cost) = (&mut self.instance, &mut self.cost);
        cost.call(instance, |instance| instance.place(call, replicas))
    }

    /// `warm`: which pages to warm, in order, during an epoch's precopy (§4.4).
    pub fn warm(&mut self, room: &Room) -> Result<Vec<PageId>, PolicyError> {
        let (instance, cost) = (&mut self.instance, &mut self.cost);
        cost.call(instance, |instance| instance.warm(room))
    }

    /// `experts`: the placement of experts on ranks (§4.4).
    pub fn experts(&mut self, load: &ExpertLoad) -> Result<ExpertOrder, PolicyError> {
        let (instance, cost) = (&mut self.instance, &mut self.cost);
        cost.call(instance, |instance| instance.experts(load))
    }

    /// The calls this scheduler's policy has made and the fuel they consumed
    /// (§12's cost counters, for `Metrics.policy_fuel`/`policy_calls`).
    pub fn usage(&self) -> PolicyUsage {
        self.cost.usage(&self.instance)
    }

    /// §7.3 step 2: instantiate `program` on `state` and ask it every decision
    /// of §4.4 once, executing nothing.
    ///
    /// An answer that does not decode, a trap, fuel exhaustion, limits outside
    /// the build config, a decision that names something that does not exist or
    /// cannot be chosen, and an expert order that is not a legal placement all
    /// fail the precopy: no candidate is returned and the serving policy is
    /// untouched. `validators` are the mechanism's own, so the candidate is
    /// judged by what will judge it live.
    ///
    /// A static function so it runs on a blocking thread and its [`Candidate`]
    /// is handed back to the scheduler thread, which keeps the serving slot
    /// throughout.
    pub fn prepare(
        engine: &Engine,
        source: &dyn PolicySource,
        program: PolicyProgram,
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
            expert_load,
        } = state;
        let init = Instantiation {
            parameters: program.arguments.clone(),
            state: snapshot,
        };
        let module = PolicyModule::load(engine, &source.bytes(&program)?)?;
        let mut instance = module.instantiate(&init.state, &init.parameters, ceilings)?;

        // Every decision of §4.4's table once, on the state that exists now and
        // without executing any of it. The calls are the trial's cost, counted
        // once the candidate is adopted.
        let mut calls = 0;
        let plan = instance.dispatch(&room)?;
        calls += 1;
        validators.plan(&plan)?;
        if let Some(shortage) = &shortage {
            let victims = instance.reclaim(shortage)?;
            calls += 1;
            validators.victims(shortage, &victims)?;
        }
        if let Some((request, ranks)) = &arrival {
            let index = instance.route(request, ranks)?;
            calls += 1;
            validators.route(request, ranks, index)?;
        }
        if let Some((call, replicas)) = &call {
            let index = instance.place(call, replicas)?;
            calls += 1;
            validators.place(call, replicas, index)?;
        }
        let pages = instance.warm(&room)?;
        calls += 1;
        validators.warm(&pages)?;
        if let Some(load) = &expert_load {
            let order = instance.experts(load)?;
            calls += 1;
            validators.experts(load, &order)?;
        }

        Ok(Candidate {
            program,
            module,
            instance,
            calls,
        })
    }

    /// Keeps a candidate a precopy prepared, for the switchover to put in.
    ///
    /// A newer candidate replaces an older one that never reached its
    /// switchover: only a later precopy's result can be the one the daemon's
    /// switchover means.
    pub fn adopt(&mut self, candidate: Candidate) {
        self.cost.trial(candidate.calls);
        self.candidate = Some((candidate.program, candidate.module, candidate.instance));
    }

    /// A precopy on the calling thread: [`Self::prepare`] and [`Self::adopt`].
    pub fn precopy(
        &mut self,
        source: &dyn PolicySource,
        program: PolicyProgram,
        state: TrialState,
        validators: &dyn Validators,
    ) -> Result<(), PolicyError> {
        let candidate = Self::prepare(
            &self.engine,
            source,
            program,
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

    /// The stage asked for and not applied yet, if any.
    pub fn pending(&self) -> Option<Stage> {
        self.pending
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
                self.switch_over();
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

    /// §7.3 step 3: the prepared instance becomes the serving one, and the
    /// module it replaces is kept to revert to.
    ///
    /// A switchover without a prepared candidate leaves the serving policy in
    /// place: the daemon's precopy failed and the update is discarded.
    fn switch_over(&mut self) {
        let Some((program, module, instance)) = self.candidate.take() else {
            tracing::warn!("a policy switchover arrived without a policy prepared by a precopy");
            return;
        };
        let replaced_program = std::mem::replace(&mut self.program, program);
        let replaced_module = std::mem::replace(&mut self.module, module);
        let replaced = std::mem::replace(&mut self.instance, instance);
        self.cost.replace(&replaced);
        // §7.3 step 5: the replaced module stays until the next precopy, which
        // replaces it; committing therefore keeps it, and a revert finds it.
        self.previous = Some((replaced_program, replaced_module));
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
        let Some((program, module)) = self.previous.take() else {
            return Err(PolicyError::invalid(
                Call::Init,
                "the failed policy has no previous policy module to revert to",
            ));
        };
        let init = init_of(&program, state);
        let instance = module.instantiate(&init.state, &init.parameters, &self.ceilings)?;
        let _ = std::mem::replace(&mut self.program, program);
        let _ = std::mem::replace(&mut self.module, module);
        let replaced = std::mem::replace(&mut self.instance, instance);
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
            program: self.program.clone(),
            reason: error.to_string(),
            fatal: false,
        };
        self.revert(state)?;
        Ok(report)
    }
}

/// The instantiation message of `program` on `state`.
fn init_of(program: &PolicyProgram, state: &TrialState) -> Instantiation {
    Instantiation {
        parameters: program.arguments.clone(),
        state: state.snapshot.clone(),
    }
}

/// The mechanism's validation of the control thread's only decision.
///
/// The route instance answers `route` alone: a plan, victims, a placement, a
/// warm list and an expert order are a scheduler thread's decisions, judged
/// there. A route decision must name a rank that is alive.
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
        _request: &RequestView,
        ranks: &[RankView],
        index: usize,
    ) -> Result<(), PolicyError> {
        match ranks.get(index) {
            Some(rank) if rank.alive => Ok(()),
            Some(_) => Err(PolicyError::invalid(
                Call::Route,
                format!("route chose rank {index}, which is not alive"),
            )),
            None => Err(PolicyError::invalid(
                Call::Route,
                format!("route chose rank {index} of {} ranks", ranks.len()),
            )),
        }
    }

    fn place(
        &self,
        _call: &CallView,
        _replicas: &[ReplicaView],
        _index: usize,
    ) -> Result<(), PolicyError> {
        Ok(())
    }

    fn warm(&self, _pages: &[PageId]) -> Result<(), PolicyError> {
        Ok(())
    }

    fn experts(&self, _load: &ExpertLoad, _order: &ExpertOrder) -> Result<(), PolicyError> {
        Ok(())
    }
}

/// The control thread's own policy instance, which answers `route` (§4.4).
///
/// It is the same program as the scheduler threads' instances and changes with
/// them: a precopy prepares it on the same state, a switchover puts it in, and a
/// revert instantiates the previous program again.
pub struct RoutePolicy {
    engine: Engine,
    ceilings: Limits,
    program: PolicyProgram,
    module: PolicyModule,
    instance: PolicyInstance,
    previous: Option<(PolicyProgram, PolicyModule)>,
    candidate: Option<(PolicyProgram, PolicyModule, PolicyInstance)>,
    /// What the control thread's policy has cost (§12).
    cost: Cost,
}

impl RoutePolicy {
    /// Instantiates `program` on `state`, which is the state routing is decided
    /// on: every rank, and every request the epoch already holds.
    pub fn install(
        engine: &Engine,
        source: &dyn PolicySource,
        program: PolicyProgram,
        state: &TrialState,
        ceilings: &Limits,
    ) -> Result<Self, PolicyError> {
        let init = init_of(&program, state);
        let module = PolicyModule::load(engine, &source.bytes(&program)?)?;
        let instance = module.instantiate(&init.state, &init.parameters, ceilings)?;
        Ok(Self {
            engine: engine.clone(),
            ceilings: *ceilings,
            program,
            module,
            instance,
            previous: None,
            candidate: None,
            cost: Cost::default(),
        })
    }

    /// The route instance an epoch's first switchover installs from the
    /// candidate its precopy prepared.
    pub fn from_candidate(engine: &Engine, ceilings: Limits, candidate: Candidate) -> Self {
        Self {
            engine: engine.clone(),
            ceilings,
            program: candidate.program,
            module: candidate.module,
            instance: candidate.instance,
            previous: None,
            candidate: None,
            cost: Cost::default(),
        }
    }

    /// The program the route instance runs.
    pub fn program(&self) -> &PolicyProgram {
        &self.program
    }

    /// `observe`: every event the control thread saw since the last call, in
    /// the order it happened.
    pub fn observe(&mut self, events: &[Event]) -> Result<(), PolicyError> {
        if events.is_empty() {
            return Ok(());
        }
        let (instance, cost) = (&mut self.instance, &mut self.cost);
        cost.call(instance, |instance| instance.observe(events))
    }

    /// `route`: the index in `ranks` of the rank that serves a new request.
    pub fn route(
        &mut self,
        request: &RequestView,
        ranks: &[RankView],
    ) -> Result<usize, PolicyError> {
        let (instance, cost) = (&mut self.instance, &mut self.cost);
        cost.call(instance, |instance| instance.route(request, ranks))
    }

    /// The calls this instance has made and the fuel they consumed (§12's cost
    /// counters, for `Metrics.policy_fuel`/`policy_calls`).
    pub fn usage(&self) -> PolicyUsage {
        self.cost.usage(&self.instance)
    }

    /// §7.3 step 2 for the control thread's instance: instantiate `program` on
    /// `state` and, when the state names a request arriving, ask it one `route`
    /// decision, executing nothing.
    ///
    /// A precopy at a batch boundary may have no request arriving — routing is
    /// decided when one does — so the trial skips `route` then rather than fail:
    /// the decision is tried by the next precopy that has an arrival, and the
    /// instance is installed either way. `validators` are the mechanism's own
    /// validation of a route decision.
    pub fn prepare(
        engine: &Engine,
        source: &dyn PolicySource,
        program: PolicyProgram,
        state: &TrialState,
        ceilings: &Limits,
        validators: &dyn Validators,
    ) -> Result<Candidate, PolicyError> {
        let init = init_of(&program, state);
        let module = PolicyModule::load(engine, &source.bytes(&program)?)?;
        let mut instance = module.instantiate(&init.state, &init.parameters, ceilings)?;
        let mut calls = 0;
        if let Some((request, ranks)) = &state.arrival {
            let index = instance.route(request, ranks)?;
            calls += 1;
            validators.route(request, ranks, index)?;
        }
        Ok(Candidate {
            program,
            module,
            instance,
            calls,
        })
    }

    /// Keeps a candidate a precopy prepared, for the switchover to put in.
    pub fn adopt(&mut self, candidate: Candidate) {
        self.cost.trial(candidate.calls);
        self.candidate = Some((candidate.program, candidate.module, candidate.instance));
    }

    /// A precopy on the calling thread: [`Self::prepare`] and [`Self::adopt`].
    pub fn precopy(
        &mut self,
        source: &dyn PolicySource,
        program: PolicyProgram,
        state: &TrialState,
        validators: &dyn Validators,
    ) -> Result<(), PolicyError> {
        let candidate = Self::prepare(
            &self.engine,
            source,
            program,
            state,
            &self.ceilings,
            validators,
        )?;
        self.adopt(candidate);
        Ok(())
    }

    /// §7.3 step 3 for the control thread's instance.
    pub fn switchover(&mut self) {
        let Some((program, module, instance)) = self.candidate.take() else {
            tracing::warn!("a route policy switchover arrived without a prepared instance");
            return;
        };
        let replaced_program = std::mem::replace(&mut self.program, program);
        let replaced_module = std::mem::replace(&mut self.module, module);
        let replaced = std::mem::replace(&mut self.instance, instance);
        self.cost.replace(&replaced);
        self.previous = Some((replaced_program, replaced_module));
    }

    /// §7.3 step 6 for the control thread's instance.
    pub fn purge(&mut self) {
        self.previous = None;
    }

    /// §7.3 step 4 for the control thread's instance, on `state`.
    pub fn revert(&mut self, state: &TrialState) -> Result<(), PolicyError> {
        self.candidate = None;
        let Some((program, module)) = self.previous.take() else {
            return Err(PolicyError::invalid(
                Call::Init,
                "the failed route policy has no previous policy module to revert to",
            ));
        };
        let init = init_of(&program, state);
        let instance = module.instantiate(&init.state, &init.parameters, &self.ceilings)?;
        let _ = std::mem::replace(&mut self.program, program);
        let _ = std::mem::replace(&mut self.module, module);
        let replaced = std::mem::replace(&mut self.instance, instance);
        self.cost.replace(&replaced);
        Ok(())
    }

    /// A `route` failure: revert on `state` and answer the report the control
    /// thread forwards like a scheduler's.
    pub fn fail(
        &mut self,
        error: &PolicyError,
        state: &TrialState,
    ) -> Result<PolicyReverted, PolicyError> {
        let report = PolicyReverted {
            scheduler: None,
            program: self.program.clone(),
            reason: error.to_string(),
            fatal: false,
        };
        self.revert(state)?;
        Ok(report)
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

    use serde::Serialize;

    use scheduling::guest::decision::{Plan, Shortage, Storage, Victim};
    use scheduling::guest::ids::{PageId, RequestId};
    use scheduling::guest::time::Nanos;
    use scheduling::guest::view::{ExpertLoad, ExpertOrder, RankView, RequestClass, RequestState};

    use super::*;

    /// Where the toy keeps its message buffer and its replies.
    const MESSAGE: u32 = 4096;
    const PLAN: u32 = 8192;
    const LIMITS: u32 = 12288;
    const VICTIMS: u32 = 16384;
    const PAGES: u32 = 20480;
    const EXPERTS: u32 = 24576;

    /// The answer a toy policy module gives, and the one call it fails in.
    struct Toy {
        plan: Plan,
        limits: Limits,
        victims: Vec<Victim>,
        pages: Vec<PageId>,
        experts: ExpertOrder,
        /// The call whose export traps (`unreachable`).
        trap: Option<Call>,
        /// The call whose export never returns.
        spin: Option<Call>,
    }

    impl Default for Toy {
        fn default() -> Self {
            Self {
                plan: Plan::default(),
                limits: fitting_limits(),
                victims: Vec::new(),
                pages: Vec::new(),
                experts: ExpertOrder::default(),
                trap: None,
                spin: None,
            }
        }
    }

    /// The module bytes of `toy`.
    fn toy_module(toy: &Toy) -> Vec<u8> {
        let bodies = [
            (Call::Dispatch, PLAN, json(&toy.plan)),
            (Call::Reclaim, VICTIMS, json(&toy.victims)),
            (Call::Warm, PAGES, json(&toy.pages)),
            (Call::Experts, EXPERTS, json(&toy.experts)),
        ];
        let mut functions = vec![
            format!("(func (export \"sched_alloc\") (param i32) (result i32) i32.const {MESSAGE})"),
            "(func (export \"sched_free\") (param i32 i32))".to_owned(),
            "(func (export \"sched_init\") (param i32 i32) (result i64) i64.const 0)".to_owned(),
            "(func (export \"sched_observe\") (param i32 i32) (result i64) i64.const 0)".to_owned(),
        ];
        let mut data = vec![format!(
            "(data (i32.const {PLAN}) \"{}\")",
            escape(&json(&toy.plan))
        )];
        for (call, offset, answer) in bodies {
            data.push(format!(
                "(data (i32.const {offset}) \"{}\")",
                escape(&answer)
            ));
            functions.push(buffered(call, offset, answer.len(), toy));
        }
        functions.push(buffered(Call::Limits, LIMITS, json(&toy.limits).len(), toy));
        data.push(format!(
            "(data (i32.const {LIMITS}) \"{}\")",
            escape(&json(&toy.limits))
        ));
        for (call, index) in [(Call::Route, 0), (Call::Place, 0)] {
            functions.push(indexed(call, index, toy));
        }
        let source = format!(
            "(module (memory (export \"memory\") 1) {} {})",
            data.join(" "),
            functions.join(" ")
        );
        wat::parse_str(&source).expect("the toy policy module is valid WebAssembly text")
    }

    /// The export of a call that answers with a reply buffer.
    fn buffered(call: Call, offset: u32, len: usize, toy: &Toy) -> String {
        let params = if call == Call::Limits {
            String::new()
        } else {
            "(param i32 i32)".to_owned()
        };
        let packed = (u64::from(offset) << 32) | u64::try_from(len).expect("a reply fits u64");
        if toy.trap == Some(call) {
            format!(
                "(func (export \"{}\") {params} (result i64) unreachable)",
                call.export()
            )
        } else if toy.spin == Some(call) {
            format!(
                "(func (export \"{}\") {params} (result i64) (loop br 0) unreachable)",
                call.export()
            )
        } else {
            format!(
                "(func (export \"{}\") {params} (result i64) i64.const {packed})",
                call.export()
            )
        }
    }

    /// The export of a call that answers with the chosen index.
    fn indexed(call: Call, index: u32, toy: &Toy) -> String {
        if toy.trap == Some(call) {
            format!(
                "(func (export \"{}\") (param i32 i32) (result i32) unreachable)",
                call.export()
            )
        } else {
            format!(
                "(func (export \"{}\") (param i32 i32) (result i32) i32.const {index})",
                call.export()
            )
        }
    }

    /// `value` as the protocol's JSON.
    fn json<T: Serialize>(value: &T) -> String {
        serde_json::to_string(value).expect("an interface value is JSON")
    }

    /// `text` escaped for a WebAssembly text data segment.
    fn escape(text: &str) -> String {
        text.replace('\\', "\\\\").replace('"', "\\\"")
    }

    /// A `PolicySource` of one toy module.
    fn source(toy: &Toy) -> PolicyBytes {
        PolicyBytes::new(toy_module(toy))
    }

    /// The program the tests install.
    fn program(backend: &str, revision: u32) -> PolicyProgram {
        PolicyProgram {
            backend: backend.to_owned(),
            revision,
            arguments: Parameters::default(),
            stated: Default::default(),
        }
    }

    /// A plan that names one request, so two programs' answers differ.
    fn plan_of(request: u64) -> Plan {
        Plan {
            refill: vec![RequestId(request)],
            ..Plan::default()
        }
    }

    /// A policy's limits that fit `ceilings`.
    fn fitting_limits() -> Limits {
        Limits {
            chunked_prefill_size: 256,
            max_batch: 4,
            max_num_batched_tokens: 512,
            ..Limits::default()
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
            expert_load: Some(ExpertLoad {
                ranks: 1,
                experts: 1,
                tokens: vec![0],
            }),
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

        fn warm(&self, _pages: &[PageId]) -> Result<(), PolicyError> {
            Ok(())
        }

        fn experts(&self, _load: &ExpertLoad, _order: &ExpertOrder) -> Result<(), PolicyError> {
            Ok(())
        }
    }

    /// A slot serving `toy` as program `a`.
    fn slot(toy: &Toy) -> PolicySlot {
        PolicySlot::install(
            &Engine::new().expect("an engine"),
            &source(toy),
            program("a", 1),
            &state(),
            &ceilings(),
        )
        .expect("the toy policy installs")
    }

    /// Switches `slot` to `toy` at a boundary, as a daemon's precopy,
    /// switchover and commit do.
    fn switch_to(slot: &mut PolicySlot, toy: &Toy) {
        slot.precopy(&source(toy), program("b", 2), state(), &Checks::default())
            .expect("the candidate passes its precopy");
        slot.request(Stage::Switchover);
        slot.boundary(&state()).expect("the switchover applies");
        slot.request(Stage::Commit);
        slot.boundary(&state()).expect("the commit applies");
    }

    #[test]
    fn a_slot_serves_the_limits_its_program_declares() {
        let declared = Limits {
            max_batch: 3,
            ..fitting_limits()
        };
        let slot = slot(&Toy {
            limits: declared,
            ..Toy::default()
        });
        assert_eq!(slot.limits().max_batch, 3);
        assert_eq!(slot.program().backend, "a");
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
            program("a", 1),
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
            program("b", 2),
            state(),
            &ceilings(),
            &Checks::default(),
        )
        .expect("the candidate passes its precopy");
        assert_eq!(candidate.program().backend, "b");
        assert_eq!(slot.program().backend, "a");
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
        assert_eq!(slot.program().backend, "b");
        assert_eq!(
            slot.dispatch(&Room::default()).expect("an answer").refill,
            plan_of(2).refill
        );
    }

    #[test]
    fn a_precopy_asks_every_decision_the_state_names() {
        // A trap in each decision's export fails the precopy, which proves the
        // decision was asked: reclaim, route and experts only when the state
        // names them, warm always.
        for call in [
            Call::Dispatch,
            Call::Reclaim,
            Call::Route,
            Call::Warm,
            Call::Experts,
        ] {
            let mut slot = slot(&Toy::default());
            let candidate = Toy {
                trap: Some(call),
                ..Toy::default()
            };
            let prepared = slot.precopy(
                &source(&candidate),
                program("b", 2),
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
            program("b", 2),
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
            program("b", 2),
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
            program("b", 2),
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
            program("b", 2),
            state(),
            &Checks::default(),
        )
        .expect("the candidate passes");
        // Until the switchover the serving policy answers.
        assert_eq!(
            slot.dispatch(&Room::default()).expect("an answer").refill,
            plan_of(1).refill
        );
        assert_eq!(slot.program().backend, "a");

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
        assert_eq!(slot.program().backend, "b");

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
        assert_eq!(report.program.backend, "b");
        assert_eq!(
            slot.dispatch(&Room::default()).expect("an answer").refill,
            plan_of(1).refill
        );
        assert_eq!(slot.program().backend, "a");
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
        assert_eq!(slot.program().backend, "a");
        assert_eq!(
            slot.dispatch(&Room::default()).expect("an answer").refill,
            plan_of(1).refill
        );
    }

    #[test]
    fn a_program_is_built_from_the_arguments_the_daemon_states() {
        let arguments = [
            ("chunked_prefill_size".to_owned(), "2048".to_owned()),
            ("max_batch".to_owned(), "16".to_owned()),
            ("expected_speedup".to_owned(), "2.5".to_owned()),
        ]
        .into_iter()
        .collect();
        let program =
            PolicyProgram::from_arguments("fcfs", 3, &arguments).expect("the arguments parse");
        assert_eq!(program.backend, "fcfs");
        assert_eq!(program.revision, 3);
        assert_eq!(program.arguments.chunked_prefill_size, Some(2048));
        assert_eq!(program.arguments.max_batch, Some(16));
        assert_eq!(program.arguments.expected_speedup, Some(2.5));
    }

    #[test]
    fn the_program_a_precopy_states_is_the_program_the_policy_runs() {
        // The wire form is the Program Server's `Program`: a name, a revision,
        // and each argument as its TOML text. Nothing about the program is
        // decided by the mechanism on the way.
        let stated = proto::process::Program {
            backend: "priority".to_owned(),
            revision: 7,
            arguments: [
                ("max_batch".to_owned(), "32".to_owned()),
                ("timeout".to_owned(), "5".to_owned()),
            ]
            .into_iter()
            .collect(),
        };
        let program = PolicyProgram::from_program(&stated).expect("the program parses");
        assert_eq!(program.backend, "priority");
        assert_eq!(program.revision, 7);
        assert_eq!(program.arguments.max_batch, Some(32));
        assert_eq!(
            program.arguments.timeout,
            Some(scheduling::guest::time::Nanos(5))
        );
    }

    #[test]
    fn a_program_argument_this_build_does_not_name_is_refused() {
        let arguments = [("chunked_prefill_size_v2".to_owned(), "2048".to_owned())]
            .into_iter()
            .collect();
        let built = PolicyProgram::from_arguments("fcfs", 3, &arguments);
        assert!(matches!(built, Err(PolicyError::Load { .. })));
    }

    #[test]
    fn the_route_instance_prepares_switches_and_reverts_like_a_scheduler_slot() {
        let engine = Engine::new().expect("an engine");
        let mut route = RoutePolicy::install(
            &engine,
            &source(&Toy::default()),
            program("a", 1),
            &state(),
            &ceilings(),
        )
        .expect("the toy policy installs");
        let (request, ranks) = state().arrival.expect("a request");
        assert_eq!(route.route(&request, &ranks).expect("a route"), 0);

        route
            .precopy(
                &source(&Toy::default()),
                program("b", 2),
                &state(),
                &Checks::default(),
            )
            .expect("the candidate passes");
        route.switchover();
        assert_eq!(route.program().backend, "b");
        let report = route
            .fail(&PolicyError::invalid(Call::Route, "refused"), &state())
            .expect("the previous policy is kept");
        assert!(report.scheduler.is_none());
        assert_eq!(route.program().backend, "a");
    }

    #[test]
    fn a_program_resolves_with_the_arguments_the_deployment_stated() {
        // The Program Server resolves a program by backend, revision and the
        // arguments that were stated; it fills the package's defaults for the
        // rest, so the source carries exactly what the daemon named.
        let arguments = std::collections::BTreeMap::from([
            ("max_batch".to_owned(), "16".to_owned()),
            ("timeout".to_owned(), "\"5s\"".to_owned()),
        ]);
        let built = PolicyProgram::from_arguments("fcfs", 7, &arguments)
            .expect("the arguments are the policy's");
        let server = built
            .to_server_program()
            .expect("the arguments are TOML values");
        assert_eq!(server.backend, "fcfs");
        assert_eq!(server.revision, 7);
        assert_eq!(server.arguments["max_batch"].as_integer(), Some(16));
        assert_eq!(server.arguments["timeout"].as_str(), Some("5s"));
        assert_eq!(server.arguments.len(), 2);
    }

    #[test]
    fn a_host_without_a_baseline_target_has_no_program_server_artifacts() {
        // `ProgramSource::connect` names the target the engine precompiles for;
        // a host that has none cannot resolve a policy at all, and says so
        // instead of loading the module for another host.
        match ProgramSource::connect("/tmp/none.sock") {
            Ok(source) => assert_eq!(source.socket(), std::path::Path::new("/tmp/none.sock")),
            Err(error) => assert!(error.to_string().contains("architecture"), "{error}"),
        }
    }
}
