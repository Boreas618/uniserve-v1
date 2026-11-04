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
//! The control thread's instance sees every rank's `Arrived`, `Emitted` and
//! `Finished` through a [`RouteFeed`], so a policy that tracks per-tenant usage
//! routes on what each rank actually served.

use std::path::{Path, PathBuf};

use crossbeam_channel::{Receiver, Sender};
use scheduling::guest::decision::{Plan, Room, Shortage, Victim};
use scheduling::guest::event::Event;
use scheduling::guest::ids::PageId;
use scheduling::guest::limits::{Limits, Parameters};
use scheduling::guest::policy::Init;
use scheduling::guest::view::{
    CallView, ExpertLoad, ExpertOrder, RankView, ReplicaView, RequestView, Snapshot,
};
use scheduling::guest::wire::Call;
use scheduling::{Engine, PolicyError, PolicyInstance, PolicyModule};
use serde::{Deserialize, Serialize};

use crate::scheduler::SchedulerConfig;

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
}

/// The policy a scheduler configuration states, before the daemon names the
/// revision it installed (§5.1).
///
/// The arguments are the configuration's scheduling values, so the policy sees
/// the deployment's own limits and applies its own defaults only where the
/// configuration states none.
pub fn program_of(config: &SchedulerConfig) -> PolicyProgram {
    PolicyProgram {
        backend: backend_name(config.policy).to_owned(),
        revision: 0,
        arguments: parameters_of(config),
    }
}

/// The backend name of a scheduling policy word, as `scheduler_config.policy`
/// spells it and the Program Server's store names it.
pub fn backend_name(policy: proto::request::SchedulingPolicy) -> &'static str {
    match policy {
        proto::request::SchedulingPolicy::Fcfs => "fcfs",
        proto::request::SchedulingPolicy::Priority => "priority",
    }
}

/// The policy's arguments from the scheduler configuration (§7.7: the
/// `scheduler_config` fields and the `speculative_config` decision values).
pub fn parameters_of(config: &SchedulerConfig) -> Parameters {
    Parameters {
        chunked_prefill_size: Some(token_count(config.chunked_prefill_size)),
        max_queued_requests: config.max_queued_requests,
        max_batch: Some(config.max_batch),
        max_num_batched_tokens: Some(token_count(config.max_num_batched_tokens)),
        // The policy states its own deepening floor and starvation deadline;
        // the build config only bounds them (`ceilings_of`).
        prefetch_min_tokens: None,
        timeout: None,
        expected_speedup: config.refill.expected_speedup,
        refill_token_cost: config.refill.token_cost,
    }
}

/// The build config's ceilings for a policy's limits (§4.4): the batch, token,
/// queue, deepening and starvation bounds the epoch was built for.
///
/// A policy declares its limits once at instantiation and they are checked
/// against these; a limit above its ceiling is a policy failure
/// (`PolicyError::BadLimits`), never a clamp.
pub fn ceilings_of(config: &SchedulerConfig) -> Limits {
    Limits {
        chunked_prefill_size: token_count(config.chunked_prefill_size),
        max_queued_requests: config.max_queued_requests,
        max_batch: config.max_batch,
        max_num_batched_tokens: token_count(config.max_num_batched_tokens),
        // §4.4 fixes the deepening floor of a prefix hit and the longest
        // starvation deadline (30 s, as sched_ext caps `timeout_ms`).
        prefetch_min_tokens: PREFETCH_FLOOR_TOKENS,
        timeout: scheduling::MAX_STARVATION_TIMEOUT,
    }
}

/// The build config's smallest prefix-hit deepening (§4.4: a host hit that
/// deepens the device hit by fewer tokens is not prefetched).
///
/// It is a build-config value, so a policy's `prefetch_min_tokens` may be above
/// it but not below it (`scheduling`'s check of the policy's limits).
pub const PREFETCH_FLOOR_TOKENS: u32 = 256;

/// `tokens` as the interface's token field, saturating instead of wrapping.
fn token_count(tokens: usize) -> u32 {
    u32::try_from(tokens).unwrap_or(u32::MAX)
}

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

/// The Program Server's module store, read by path (§5.3).
///
/// A policy's module path is
/// `scheduling/<backend>/<revision>/<target>/<descriptor>` below the store's
/// root, the layout every program's modules use.
#[derive(Clone, Debug)]
pub struct PolicyStore {
    root: PathBuf,
}

impl PolicyStore {
    /// A store rooted at `root`, the directory the Program Server shares.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The store's root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The path of the module `program` resolves to.
    ///
    /// A policy package compiles to one module per revision rather than one per
    /// descriptor, so the descriptor segment is the package's single `default`
    /// one.
    pub fn module_path(&self, program: &PolicyProgram) -> PathBuf {
        self.root
            .join("scheduling")
            .join(&program.backend)
            .join(program.revision.to_string())
            .join("wasm32")
            .join("default")
            .join("module")
    }
}

impl PolicySource for PolicyStore {
    fn bytes(&self, program: &PolicyProgram) -> Result<Vec<u8>, PolicyError> {
        let path = self.module_path(program);
        std::fs::read(&path).map_err(|error| PolicyError::Load {
            message: format!(
                "cannot read the {}/{} policy module at {}: {error}",
                program.backend,
                program.revision,
                path.display()
            ),
        })
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

/// A policy that failed, and the policy that serves in its place (§7.3).
///
/// An engine-level report, not a request event: the control thread forwards it
/// to every daemon, which records the previous policy as the epoch's current
/// one, so a later restart loads that one.
#[derive(Clone, Debug)]
pub struct PolicyReverted {
    /// The scheduler whose instance failed; `None` for the control thread's
    /// route instance.
    pub scheduler: Option<usize>,
    /// The program that failed.
    pub program: PolicyProgram,
    /// Why it failed (§4.4's list: a trap, fuel, bad limits, an invalid
    /// decision, an illegal expert order, or starvation).
    pub reason: String,
}

/// What a scheduler thread's policy slot hears from outside its own rank.
#[derive(Clone, Default)]
pub struct PolicyPorts {
    /// The scheduler index this slot serves, named in a revert report.
    pub scheduler: Option<usize>,
    /// The control thread's route instance, fed this rank's arrivals,
    /// emissions and completions.
    pub route: Option<RouteFeed>,
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
    /// The last failure's report, for a mechanism that hands it over by
    /// polling rather than through `fail`'s answer.
    report: Option<PolicyReverted>,
    ports: PolicyPorts,
}

impl PolicySlot {
    /// Loads `program`'s module from `source` and instantiates it on `state`
    /// (§7.3: the epoch's first policy).
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
        ports: PolicyPorts,
    ) -> Result<Self, PolicyError> {
        let init = init_of(&program, state);
        let module = PolicyModule::load(engine, &source.bytes(&program)?)?;
        let instance = module.instantiate(&init, ceilings)?;
        Ok(Self {
            engine: engine.clone(),
            ceilings: *ceilings,
            program,
            module,
            instance,
            previous: None,
            candidate: None,
            pending: None,
            report: None,
            ports,
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

    /// Fuel every call into the serving instance has consumed, which is §12's
    /// per-step policy cost when the mechanism takes it around a step.
    pub fn fuel_consumed(&self) -> u64 {
        self.instance.fuel_consumed()
    }

    /// `observe`: every event since the previous step, in order (§4.4).
    ///
    /// The rank's arrivals, emissions and completions also feed the control
    /// thread's route instance, so routing sees the usage the policy sees.
    pub fn observe(&mut self, events: &[Event]) -> Result<(), PolicyError> {
        if let Some(route) = &self.ports.route {
            route.feed(events);
        }
        self.instance.observe(events)
    }

    /// `dispatch`: this step's plan (§4.4).
    pub fn dispatch(&mut self, room: &Room) -> Result<Plan, PolicyError> {
        self.instance.dispatch(room)
    }

    /// `reclaim`: which objects to free, in order, for a shortage (§4.4).
    pub fn reclaim(&mut self, shortage: &Shortage) -> Result<Vec<Victim>, PolicyError> {
        self.instance.reclaim(shortage)
    }

    /// `place`: which replica runs a request's first call on a block (§4.4).
    pub fn place(
        &mut self,
        call: &CallView,
        replicas: &[ReplicaView],
    ) -> Result<usize, PolicyError> {
        self.instance.place(call, replicas)
    }

    /// `warm`: which pages to warm, in order, during an epoch's precopy (§4.4).
    pub fn warm(&mut self, room: &Room) -> Result<Vec<PageId>, PolicyError> {
        self.instance.warm(room)
    }

    /// `experts`: the placement of experts on ranks (§4.4).
    pub fn experts(&mut self, load: &ExpertLoad) -> Result<ExpertOrder, PolicyError> {
        self.instance.experts(load)
    }

    /// §7.3 step 2, on a blocking thread: instantiate `program` on `state` and
    /// ask it every decision of §4.4 once, executing nothing.
    ///
    /// An answer that does not decode, a trap, fuel exhaustion, limits outside
    /// the build config, a decision that names something that does not exist or
    /// cannot be chosen, and an expert order that is not a legal placement all
    /// fail the precopy: the candidate is not kept and the serving policy is
    /// untouched. `validators` are the mechanism's own, so the candidate is
    /// judged by what will judge it live.
    pub fn precopy(
        &mut self,
        source: &dyn PolicySource,
        program: PolicyProgram,
        state: TrialState,
        validators: &dyn Validators,
    ) -> Result<(), PolicyError> {
        let TrialState {
            snapshot,
            room,
            shortage,
            arrival,
            call,
            expert_load,
        } = state;
        let init = Init {
            parameters: program.arguments.clone(),
            state: snapshot,
        };
        let module = PolicyModule::load(&self.engine, &source.bytes(&program)?)?;
        let mut instance = module.instantiate(&init, &self.ceilings)?;

        // Every decision of §4.4's table once, on the state that exists now and
        // without executing any of it.
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
        let pages = instance.warm(&room)?;
        validators.warm(&pages)?;
        if let Some(load) = &expert_load {
            let order = instance.experts(load)?;
            validators.experts(load, &order)?;
        }

        self.candidate = Some((program, module, instance));
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
                Ok(self.take_report())
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
            tracing::warn!(
                scheduler = ?self.ports.scheduler,
                "a policy switchover arrived without a policy prepared by a precopy"
            );
            return;
        };
        let replaced_program = std::mem::replace(&mut self.program, program);
        let replaced_module = std::mem::replace(&mut self.module, module);
        let _ = std::mem::replace(&mut self.instance, instance);
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
        let instance = module.instantiate(&init, &self.ceilings)?;
        let _ = std::mem::replace(&mut self.program, program);
        let _ = std::mem::replace(&mut self.module, module);
        let _ = std::mem::replace(&mut self.instance, instance);
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
            scheduler: self.ports.scheduler,
            program: self.program.clone(),
            reason: error.to_string(),
        };
        self.revert(state)?;
        self.report = Some(report.clone());
        Ok(report)
    }

    /// The report of a failure the mechanism did not take from [`Self::fail`]'s
    /// answer.
    pub fn take_report(&mut self) -> Option<PolicyReverted> {
        self.report.take()
    }
}

/// The instantiation message of `program` on `state`.
fn init_of(program: &PolicyProgram, state: &TrialState) -> Init {
    Init {
        parameters: program.arguments.clone(),
        state: state.snapshot.clone(),
    }
}

/// One scheduler thread's feed into the control thread's route instance.
///
/// §4.4: the route instance "也收到各 rank 的 `Arrived`、`Emitted` 与 `Finished`",
/// so a policy that tracks per-tenant usage routes on the same usage it would
/// have seen had it judged every step itself.
#[derive(Clone)]
pub struct RouteFeed(Sender<Event>);

impl RouteFeed {
    /// A feed and the control thread's end of it.
    ///
    /// Unbounded because dropping an event would leave the route instance with
    /// a view of usage that never happened; the events are proportional to
    /// state changes, and the control thread drains them whenever it routes a
    /// request.
    pub fn channel() -> (Self, Receiver<Event>) {
        let (sender, receiver) = crossbeam_channel::unbounded();
        (Self(sender), receiver)
    }

    /// Forwards the three event kinds routing sees; every other kind is the
    /// policy's own state, not a rank's usage.
    fn feed(&self, events: &[Event]) {
        for event in events {
            if matches!(
                event,
                Event::Arrived { .. } | Event::Emitted { .. } | Event::Finished { .. }
            ) && self.0.send(event.clone()).is_err()
            {
                tracing::error!("the route policy's control thread stopped taking ranking events");
                return;
            }
        }
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
    feed: Receiver<Event>,
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
        feed: Receiver<Event>,
    ) -> Result<Self, PolicyError> {
        let init = init_of(&program, state);
        let module = PolicyModule::load(engine, &source.bytes(&program)?)?;
        let instance = module.instantiate(&init, ceilings)?;
        Ok(Self {
            engine: engine.clone(),
            ceilings: *ceilings,
            program,
            module,
            instance,
            previous: None,
            candidate: None,
            feed,
        })
    }

    /// The program the route instance runs.
    pub fn program(&self) -> &PolicyProgram {
        &self.program
    }

    /// Applies every event the rank schedulers reported since the last call, in
    /// the order they happened.
    pub fn observe(&mut self) -> Result<(), PolicyError> {
        let mut events = Vec::new();
        while let Ok(event) = self.feed.try_recv() {
            events.push(event);
        }
        if events.is_empty() {
            return Ok(());
        }
        self.instance.observe(&events)
    }

    /// `route`: the index in `ranks` of the rank that serves a new request.
    pub fn route(
        &mut self,
        request: &RequestView,
        ranks: &[RankView],
    ) -> Result<usize, PolicyError> {
        self.instance.route(request, ranks)
    }

    /// §7.3 step 2 for the control thread's instance: instantiate `program` on
    /// `state` and ask it one `route` decision, executing nothing.
    ///
    /// `check` validates the answer with the mechanism's own validation of a
    /// route decision.
    pub fn precopy(
        &mut self,
        source: &dyn PolicySource,
        program: PolicyProgram,
        state: &TrialState,
        validators: &dyn Validators,
    ) -> Result<(), PolicyError> {
        let (request, ranks) = state.arrival.as_ref().ok_or_else(|| {
            PolicyError::invalid(
                Call::Route,
                "the precopy state names no arriving request to route",
            )
        })?;
        let init = init_of(&program, state);
        let module = PolicyModule::load(&self.engine, &source.bytes(&program)?)?;
        let mut instance = module.instantiate(&init, &self.ceilings)?;
        let index = instance.route(request, ranks)?;
        validators.route(request, ranks, index)?;
        self.candidate = Some((program, module, instance));
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
        let _ = std::mem::replace(&mut self.instance, instance);
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
        let instance = module.instantiate(&init, &self.ceilings)?;
        let _ = std::mem::replace(&mut self.program, program);
        let _ = std::mem::replace(&mut self.module, module);
        let _ = std::mem::replace(&mut self.instance, instance);
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
    use scheduling::guest::event::{End, Event};
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
            PolicyPorts::default(),
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
            PolicyPorts::default(),
        );
        assert!(matches!(installed, Err(PolicyError::BadLimits { .. })));
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
    fn the_route_feed_carries_arrivals_emissions_and_completions_only() {
        let (feed, received) = RouteFeed::channel();
        let mut slot = PolicySlot::install(
            &Engine::new().expect("an engine"),
            &source(&Toy::default()),
            program("a", 1),
            &state(),
            &ceilings(),
            PolicyPorts {
                scheduler: Some(0),
                route: Some(feed),
            },
        )
        .expect("the toy policy installs");
        let arrived = Event::Arrived {
            request: RequestView {
                id: RequestId(3),
                ..state().arrival.expect("a request").0
            },
        };
        let events = vec![
            arrived.clone(),
            Event::Stepped {
                step: 1,
                duration: scheduling::guest::time::Duration(1),
            },
            Event::Emitted {
                request: RequestId(3),
                tokens: 1,
                at: Nanos(2),
            },
            Event::Finished {
                request: RequestId(3),
                end: End::Completed,
            },
        ];
        slot.observe(&events).expect("the policy observes");
        let forwarded: Vec<Event> = received.try_iter().collect();
        assert_eq!(forwarded.len(), 3);
        assert!(matches!(forwarded[0], Event::Arrived { .. }));
        assert!(matches!(forwarded[1], Event::Emitted { .. }));
        assert!(matches!(forwarded[2], Event::Finished { .. }));
    }

    #[test]
    fn the_route_instance_prepares_switches_and_reverts_like_a_scheduler_slot() {
        let engine = Engine::new().expect("an engine");
        let (_feed, received) = RouteFeed::channel();
        let mut route = RoutePolicy::install(
            &engine,
            &source(&Toy::default()),
            program("a", 1),
            &state(),
            &ceilings(),
            received,
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
    fn the_store_names_a_policy_module_by_backend_revision_target_and_descriptor() {
        let store = PolicyStore::new("/programs");
        let path = store.module_path(&program("fcfs", 7));
        assert_eq!(
            path,
            std::path::Path::new("/programs/scheduling/fcfs/7/wasm32/default/module")
        );
    }
}
