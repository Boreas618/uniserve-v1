//! The scheduling policy's host (`specs/scheduler-design.md` §5): [`Policy`], a WASM module's
//! instance with the seven calls of the policy interface, its switch and rollback (§3.9), the
//! check of the limits it declares, and [`PolicyError`], the faults of a policy.
//!
//! The interface is the `scheduling` crate's: every message is JSON of one of its types
//! written into the module's memory through `sched_alloc`, and a reply is a buffer the module
//! allocated, packed as `ptr << 32 | len` (`0` for none), or for `route` and `place` the chosen
//! index itself (`scheduling::wire`). Every export the host enters, the two buffer exports
//! included, starts with a fuel budget of its own, so a policy that loops ends its call instead
//! of the step; a call's budget grows with the message it decodes.
//!
//! The policy holds the view, and the scheduler tells it what changed as events (§5.2); the host
//! keeps the view of each request the policy holds as those events leave it, so that a request
//! whose step moved nothing the events do not carry is not restated. A decision that names
//! something that cannot be done is cut or
//! skipped where it is used (§3.4, §3.6), never a fault; a fault is a trap, exhausted fuel, a reply
//! that does not decode, or limits outside the ceilings.
//!
//! A policy module is the wasm32 binary the Build Server builds a policy package to, compiled
//! here under the one wasmtime configuration of this module ([`runtime`]), which meters fuel,
//! when a scheduler loads it ([`load`]).

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::batch::Shape;
use proto::config::SchedulerConfig;
use proto::request::RequestId;
use scheduling::decision::{DispatchOutput, Room, Shortage, Victim};
use scheduling::event::Event;
use scheduling::ids::RequestId as PolicyRequestId;
use scheduling::limits::{Limits, Parameters};
use scheduling::policy::Instantiation;
use scheduling::time::Nanos;
use scheduling::view::{
    CallView, RankView, ReplicaView, RequestStatus as Standing, RequestView, Snapshot,
};
use scheduling::wire::{self, Call};
use wasmtime::{Config, Memory, Module, OptLevel, Store, Trap, TypedFunc, WasmParams, WasmResults};

use super::view::view;
use super::{Capacity, Error, Output, Request, RequestStatus, Scheduler, WorkerGroup};

/// The target a policy module is compiled for: the host architecture's baseline Linux triple, so
/// the code is the same on every host of the architecture whatever its CPU model. `None` on an
/// architecture with no policy target, where [`runtime`] fails.
pub(crate) fn target() -> Option<&'static str> {
    match std::env::consts::ARCH {
        "x86_64" => Some("x86_64-unknown-linux-gnu"),
        "aarch64" => Some("aarch64-unknown-linux-gnu"),
        _ => None,
    }
}

/// The wasmtime engine policy modules are compiled for and run in.
///
/// Fuel metering is on for the whole engine (every call into a policy has a budget), backtraces
/// are off because a failure is reported as its trap code alone, and the compiler runs at
/// `Speed`. The target is [`target`]. Each call builds a new engine; a module keeps the engine it
/// was compiled with, and its instances run in that one.
///
/// # Errors
///
/// [`PolicyError::Load`] when the host's architecture has no policy target or wasmtime refuses
/// the configuration.
fn runtime() -> Result<wasmtime::Engine, PolicyError> {
    let target = target().ok_or_else(|| {
        PolicyError::Load(format!(
            "the host architecture {} has no policy target",
            std::env::consts::ARCH
        ))
    })?;

    let mut config = Config::new();
    config.consume_fuel(true);
    config.cranelift_opt_level(OptLevel::Speed);
    // A fault is reported as its trap code alone (`None`: no frames are collected): the
    // scheduler reverts the policy rather than debugging it in its own thread.
    config.wasm_backtrace_max_frames(None);
    config
        .target(target)
        .map_err(|error| PolicyError::Load(format!("{target} is not a policy target: {error}")))?;

    wasmtime::Engine::new(&config).map_err(|error| {
        PolicyError::Load(format!(
            "this host has no engine for wasm32 policy modules: {error}"
        ))
    })
}

/// The module a scheduler instantiates ([`Policy::new`]) from the bytes the Build Server serves
/// for a policy package: the wasm32 binary its crate compiled to, compiled here. Text is accepted
/// where wasmtime's `wat` feature is on, which is the tests. The module carries the fuel-metering
/// configuration of [`runtime`], so every instance of it is metered per call.
///
/// # Errors
///
/// [`PolicyError::Load`] when the host has no policy engine, or the bytes are not a core wasm
/// module.
pub fn load(bytes: &[u8]) -> Result<Module, PolicyError> {
    Module::new(&runtime()?, bytes)
        .map_err(|error| PolicyError::Load(format!("cannot compile the policy module: {error}")))
}

/// The fuel one call into a policy module may consume, and each export the host enters for that
/// call (`H/instance.rs:32`): far above a step's `observe` and `dispatch` of a real policy, and a
/// bound on one that never returns.
pub(super) const DEFAULT_CALL_FUEL: u64 = 10_000_000;

/// The fuel a call is given beyond [`DEFAULT_CALL_FUEL`] for every byte of its message.
///
/// A message can state every cached block: `init`'s snapshot states every request and every
/// cached block, and the events of the step after the scheduler's startup `Listing` state each
/// block it lists as a `Cached` page, which a scheduler that starts beside a warm cache (an
/// epoch's target of the same layout maps the serving epoch's regions) learns after its policy
/// was instantiated. Decoding such a message grows with the cache: typed JSON decoding in a
/// wasm32 module costs 50 to 100 fuel a byte, and the packaged `arrival` spends about 70 a byte of
/// cached blocks on its whole `init`. Twice the decoding's dearest leaves a policy as much again
/// for its own state, so a policy instantiates on, and observes, a warm cache of any size, and
/// the bound still ends a call that never returns. A step's other events cost what the step
/// changed, and its other messages are small, so their budget stays near [`DEFAULT_CALL_FUEL`].
const MESSAGE_FUEL_PER_BYTE: u64 = 200;

/// The longest starvation bound a policy may declare (`limits().timeout`), as `sched_ext` caps
/// `timeout_ms` at 30 seconds.
const MAX_TIMEOUT: Nanos = Nanos(30_000_000_000);

/// The smallest prefix-hit deepening a policy may declare (`prefetch_min_tokens`): a shorter hit is
/// computed faster than it is fetched.
const PREFETCH_FLOOR_TOKENS: u32 = 256;

/// The serving scheduling policy, and the module it replaced, kept for rollback (SD §2.1, §5.2).
///
/// Invariants: every state change a view names is in `events` (or its request in `dirty`)
/// before the next `dispatch`; `told` holds what the serving instance holds of each request once
/// it has observed `events`; `limits` is within the ceilings the policy was instantiated under.
/// The policy's own memory is never carried: every instantiation starts from `init` with the
/// snapshot of that moment.
pub(crate) struct Policy {
    /// The module's instance with its exports.
    instance: Instance,
    /// The module and the package arguments the instance was made from.
    module: Arc<Module>,
    parameters: Parameters,
    /// The ceilings the serving policy's limits were checked against, which a rollback checks the
    /// previous policy's against again.
    ceilings: Limits,
    /// The module and arguments the last switch replaced, which a rollback instantiates; `None`
    /// once the switch is committed.
    pub previous: Option<(Arc<Module>, Parameters)>,
    /// Whether the serving instance was made by [`Policy::recover`] and has not completed a
    /// decision since (a scheduler's `dispatch`, the routing instance's `route`): a fault of it
    /// then ends the scheduler, or the engine (SD §5.2 "Faults").
    recovering: bool,
    /// The limits the serving policy declared: the step's token and call budgets, the chunk limit,
    /// the queue bound, the prefetch threshold and the starvation bound the scheduler enforces.
    pub limits: Limits,
    /// The state changes since the policy was last told, in the order they happened.
    pub events: Vec<Event>,
    /// The requests whose view may have changed since the policy was last told, each once
    /// (`Scheduler::mark`); `dispatch` tells each whose view did change as one `Updated`.
    pub dirty: Vec<RequestId>,
    /// The view of each request the serving instance holds as the events it was told leave it
    /// (SD §5.2): the view its snapshot or the last `Arrived` or `Updated` stated, running once
    /// `Admitted`, and advanced by every `Emitted` as the policy advances it.
    told: HashMap<RequestId, RequestView>,
    /// The calls of the interface that reached a module, over every instance of this policy.
    pub calls: u64,
    /// The fuel those calls consumed.
    pub fuel: u64,
}

/// Why a policy module could not be loaded, or a fault of a policy (SD §5.2): what ends its
/// instance.
#[derive(Debug)]
pub enum PolicyError {
    /// The module's bytes were rejected: not a core wasm32 module; or this host has no engine
    /// to compile them with.
    Load(String),
    /// The module does not instantiate as a policy: it does not link, lacks the memory or an
    /// export the interface calls, implements another version of the interface, or its engine
    /// meters no fuel.
    Instantiate(String),
    /// The policy trapped in a call: `unreachable`, an access outside its memory, a panic.
    Trap { call: Call, trap: Trap },
    /// The policy ran past the fuel of one call.
    Fuel { call: Call },
    /// A message or a reply the protocol cannot carry: a reply missing where one is due, bytes
    /// that do not decode into the call's type, or a buffer outside the module's memory.
    Malformed { call: Call, message: String },
    /// The limits the policy declared are outside the ceilings.
    Limits(String),
}

impl fmt::Display for PolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(message) => write!(formatter, "it was not loaded: {message}"),
            Self::Instantiate(message) => write!(formatter, "it does not instantiate: {message}"),
            Self::Trap { call, trap } => {
                write!(formatter, "it trapped in `{}`: {trap}", call.method())
            }
            Self::Fuel { call } => write!(formatter, "it ran past its fuel in `{}`", call.method()),
            Self::Malformed { call, message } => {
                write!(
                    formatter,
                    "its reply in `{}` is malformed: {message}",
                    call.method()
                )
            }
            Self::Limits(message) => {
                write!(formatter, "its limits are outside the ceilings: {message}")
            }
        }
    }
}

impl std::error::Error for PolicyError {}

impl Policy {
    /// Instantiates `module` as a scheduler's policy (or the engine's routing policy): constructs
    /// it from the package's `parameters`, initializes it with `snapshot`, reads its limits once
    /// and checks them against `ceilings`.
    ///
    /// # Errors
    ///
    /// The fault of `init` or `limits`, or [`PolicyError::Limits`] for limits outside the
    /// ceilings.
    pub(crate) fn new(
        module: Arc<Module>,
        parameters: Parameters,
        snapshot: Snapshot,
        ceilings: &Limits,
    ) -> Result<Self, PolicyError> {
        let (mut calls, mut fuel) = (0, 0);
        let told = told_by(&snapshot);
        let (instance, limits) = start(
            &module,
            &parameters,
            snapshot,
            ceilings,
            &mut calls,
            &mut fuel,
        )?;

        Ok(Self {
            instance,
            module,
            parameters,
            ceilings: *ceilings,
            previous: None,
            recovering: false,
            limits,
            events: Vec::new(),
            dirty: Vec::new(),
            told,
            calls,
            fuel,
        })
    }

    /// A new generation's policy serves from the next call on (SD §3.9): `module` is
    /// instantiated on `snapshot` and its limits are checked against `ceilings`; the replaced
    /// module and its arguments become `previous`.
    ///
    /// # Errors
    ///
    /// [`Error::Policy`] with the new module's fault; the serving policy is then untouched.
    pub(crate) fn switch(
        &mut self,
        module: Arc<Module>,
        parameters: Parameters,
        snapshot: Snapshot,
        ceilings: &Limits,
    ) -> Result<(), Error> {
        let told = told_by(&snapshot);
        let (instance, limits) = start(
            &module,
            &parameters,
            snapshot,
            ceilings,
            &mut self.calls,
            &mut self.fuel,
        )
        .map_err(Error::Policy)?;

        let module = std::mem::replace(&mut self.module, module);
        let parameters = std::mem::replace(&mut self.parameters, parameters);
        self.previous = Some((module, parameters));
        self.install(instance, limits, *ceilings, told);
        Ok(())
    }

    /// The policy the last switch replaced serves again (SD §3.9): `previous` is instantiated on
    /// `snapshot` under the serving ceilings and serves in place of the serving policy, which is
    /// dropped; nothing is kept to roll back to after it.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRequest`] when no switch is left to roll back, and [`Error::Policy`] with
    /// the previous module's fault; either way the serving policy and `previous` are untouched.
    pub(crate) fn rollback(&mut self, snapshot: Snapshot) -> Result<(), Error> {
        let Some((module, parameters)) = self.previous.clone() else {
            return Err(Error::InvalidRequest(
                "no replaced policy is kept to roll back to".into(),
            ));
        };

        self.serve(module, parameters, snapshot)
            .map_err(Error::Policy)?;
        self.previous = None;
        Ok(())
    }

    /// Whether a fault of the serving instance can be recovered from by [`Policy::recover`]:
    /// unless that instance was made by a recovery and has not completed a decision since (a
    /// `dispatch`, or the routing instance's `route`), since a fault that a fresh instance meets
    /// again before it decides is not one a recovery cures.
    pub(crate) fn recoverable(&self) -> bool {
        !self.recovering
    }

    /// After a fault of the serving instance (SD §5.2 "Faults"): the policy the last switch
    /// replaced serves again, or with none kept a fresh instance of the serving module, either
    /// instantiated on `snapshot` under the serving ceilings. The faulted instance's memory and
    /// the events gathered for it are dropped, since the snapshot states what they told. Nothing
    /// is kept to roll back to after it.
    ///
    /// # Errors
    ///
    /// The fault of the instantiation; the faulted instance then stays, and the scheduler fails.
    pub(crate) fn recover(&mut self, snapshot: Snapshot) -> Result<(), PolicyError> {
        let (module, parameters) = match &self.previous {
            Some((module, parameters)) => (Arc::clone(module), parameters.clone()),
            None => (Arc::clone(&self.module), self.parameters.clone()),
        };

        self.serve(module, parameters, snapshot)?;
        self.previous = None;
        self.recovering = true;
        Ok(())
    }

    /// The trial of the last switch ends in its favour (SD §3.9): the module and arguments it
    /// replaced are dropped, and nothing is left to roll back to.
    pub(crate) fn commit(&mut self) {
        self.previous = None;
    }

    /// One step's decision (SD §5.2): every request marked since the last step whose view now
    /// differs from the one the policy holds once it has observed the step's events is told as
    /// `Updated` with its view now (and every mark is cleared), the step's events go to
    /// `observe`, and `dispatch` is asked with `room`. `groups` are the scheduler's, which a view
    /// reads.
    ///
    /// # Errors
    ///
    /// The fault of `observe` or `dispatch`; an empty `dispatch` reply is
    /// [`PolicyError::Malformed`].
    pub(crate) fn dispatch(
        &mut self,
        room: &Room,
        requests: &mut HashMap<RequestId, Request>,
        groups: &[WorkerGroup],
    ) -> Result<DispatchOutput, PolicyError> {
        // What the policy will hold of each request once it observed the step's events.
        for event in &self.events {
            follow(&mut self.told, event);
        }

        // A request whose fields changed since the policy was told is told again, once, when its
        // view differs from what the policy holds: a decode that moved only its generated tokens
        // is stated by its `Emitted`. One that ended meanwhile was told by its `Finished`, after
        // which the policy forgets it.
        for id in self.dirty.drain(..) {
            let Some(request) = requests.get_mut(&id) else {
                self.told.remove(&id);
                continue;
            };
            request.dirty = false;
            if request.status == RequestStatus::Finished {
                continue;
            }

            // A refill is offered as a step of the policy's own token limit sizes it.
            let tokens = Some(self.limits.max_num_batched_tokens);
            let view = view(request, groups, tokens);
            if self.told.get(&id) == Some(&view) {
                continue;
            }
            self.told.insert(id, view.clone());
            self.events.push(Event::Updated { request: view });
        }

        // `observe` replies nothing a host reads.
        let events = std::mem::take(&mut self.events);
        self.call(Call::Observe, wire::encode(&events))?;

        let reply = self
            .call(Call::Dispatch, wire::encode(room))?
            .ok_or_else(|| malformed(Call::Dispatch, "the policy replied nothing"))?;
        let output = decoded(Call::Dispatch, wire::decode(&reply))?;
        self.recovering = false;
        Ok(output)
    }

    /// The victims the policy names for a shortage, in its order; an empty reply names none.
    ///
    /// # Errors
    ///
    /// The fault of `preempt`.
    pub(crate) fn preempt(&mut self, shortage: &Shortage) -> Result<Vec<Victim>, PolicyError> {
        match self.call(Call::Preempt, wire::encode(shortage))? {
            Some(reply) => decoded(Call::Preempt, wire::decode(&reply)),
            None => Ok(Vec::new()),
        }
    }

    /// The index in `ranks` of the data-parallel rank that serves a new request: the engine's
    /// routing policy's call, never a scheduler's. A route the policy completes is the routing
    /// instance's decision, as a `dispatch` is a scheduler's: an instance [`Policy::recover`] made
    /// is recoverable again once it has routed a request.
    ///
    /// # Errors
    ///
    /// The fault of `route`.
    pub(crate) fn route(
        &mut self,
        request: &RequestView,
        ranks: &[RankView],
    ) -> Result<usize, PolicyError> {
        let index = self.index(Call::Route, wire::encode(&(request, ranks)))?;
        self.recovering = false;
        Ok(index)
    }

    /// The index in `replicas` of the group a request's first call of a phase runs on. The
    /// caller replaces an index that names no candidate (SD §3.5 `place`).
    ///
    /// # Errors
    ///
    /// The fault of `place`.
    pub(crate) fn place(
        &mut self,
        call: &CallView,
        replicas: &[ReplicaView],
    ) -> Result<usize, PolicyError> {
        self.index(Call::Place, wire::encode(&(call, replicas)))
    }

    /// `module` with `parameters`, instantiated on `snapshot` under the serving ceilings, serves
    /// in place of the serving instance; on a fault the serving instance is untouched.
    fn serve(
        &mut self,
        module: Arc<Module>,
        parameters: Parameters,
        snapshot: Snapshot,
    ) -> Result<(), PolicyError> {
        let ceilings = self.ceilings;
        let told = told_by(&snapshot);
        let (instance, limits) = start(
            &module,
            &parameters,
            snapshot,
            &ceilings,
            &mut self.calls,
            &mut self.fuel,
        )?;

        self.module = module;
        self.parameters = parameters;
        self.install(instance, limits, ceilings, told);
        Ok(())
    }

    /// Makes `instance` the serving one, with its limits and the ceilings they were checked
    /// against, and `told`, the views of the snapshot it started from, as what it holds. The
    /// events gathered for the instance it replaces are dropped, since the snapshot states what
    /// they told; marked requests stay marked, and their views are told to it at its first
    /// `dispatch` where they differ from the snapshot's.
    fn install(
        &mut self,
        instance: Instance,
        limits: Limits,
        ceilings: Limits,
        told: HashMap<RequestId, RequestView>,
    ) {
        self.instance = instance;
        self.limits = limits;
        self.ceilings = ceilings;
        self.recovering = false;
        self.events.clear();
        self.told = told;
    }

    /// One call that takes an encoded message and replies with a buffer: the reply's bytes, or
    /// `None` for the empty reply.
    fn call(
        &mut self,
        call: Call,
        message: Result<Vec<u8>, serde_json::Error>,
    ) -> Result<Option<Vec<u8>>, PolicyError> {
        let message = encoded(call, message)?;
        self.calls += 1;

        let packed = self.instance.exchange(call, &message, &mut self.fuel)?;
        self.instance.take(call, packed, &mut self.fuel)
    }

    /// One call that takes an encoded message and replies with an index.
    fn index(
        &mut self,
        call: Call,
        message: Result<Vec<u8>, serde_json::Error>,
    ) -> Result<usize, PolicyError> {
        let message = encoded(call, message)?;
        self.calls += 1;

        let index = self.instance.exchange(call, &message, &mut self.fuel)?;
        usize::try_from(index).map_err(|_| malformed(call, "the index does not fit this host"))
    }
}

/// A policy module's instance: its store, its memory and the exports the interface calls.
struct Instance {
    store: Store<()>,
    memory: Memory,
    alloc: TypedFunc<u32, u32>,
    free: TypedFunc<(u32, u32), ()>,
    init: TypedFunc<(u32, u32), u64>,
    observe: TypedFunc<(u32, u32), u64>,
    dispatch: TypedFunc<(u32, u32), u64>,
    preempt: TypedFunc<(u32, u32), u64>,
    route: TypedFunc<(u32, u32), u32>,
    place: TypedFunc<(u32, u32), u32>,
    limits: TypedFunc<(), u64>,
}

impl Instance {
    /// Instantiates `module` and binds every export the interface calls, naming the first that
    /// is missing or has another signature.
    fn new(module: &Module) -> Result<Self, PolicyError> {
        let mut store = Store::new(module.engine(), ());
        // An engine that meters no fuel refuses a budget, and a policy that loops would then
        // hold the scheduler's thread.
        store.set_fuel(DEFAULT_CALL_FUEL).map_err(|error| {
            PolicyError::Instantiate(format!(
                "its engine meters no fuel (load it with `policy::load`): {error}"
            ))
        })?;

        let instance = wasmtime::Instance::new(&mut store, module, &[])
            .map_err(|error| PolicyError::Instantiate(error.to_string()))?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| PolicyError::Instantiate("it exports no `memory`".into()))?;

        // A module of another version of the interface would be asked decisions through calls
        // whose messages mean something else to it: it states the version it was compiled
        // against before any call is made.
        let version: TypedFunc<(), u32> =
            bind(&mut store, &instance, wire::EXPORT_INTERFACE_VERSION)?;
        let implemented = version.call(&mut store, ()).map_err(|error| {
            PolicyError::Instantiate(format!("it states no interface version: {error}"))
        })?;
        if implemented != scheduling::INTERFACE_VERSION {
            return Err(PolicyError::Instantiate(format!(
                "it implements version {implemented} of the scheduling policy interface and this \
                 engine instantiates version {}",
                scheduling::INTERFACE_VERSION
            )));
        }

        Ok(Self {
            alloc: bind(&mut store, &instance, "sched_alloc")?,
            free: bind(&mut store, &instance, "sched_free")?,
            init: bind(&mut store, &instance, Call::Init.export())?,
            observe: bind(&mut store, &instance, Call::Observe.export())?,
            dispatch: bind(&mut store, &instance, Call::Dispatch.export())?,
            preempt: bind(&mut store, &instance, Call::Preempt.export())?,
            route: bind(&mut store, &instance, Call::Route.export())?,
            place: bind(&mut store, &instance, Call::Place.export())?,
            limits: bind(&mut store, &instance, Call::Limits.export())?,
            store,
            memory,
        })
    }

    /// Writes `message` into a buffer of the module's memory, enters the export of `call` with
    /// it and frees it. Returns the export's result: a packed reply, or for `route` and `place`
    /// the index. `fuel` gathers what every entry consumed.
    fn exchange(&mut self, call: Call, message: &[u8], fuel: &mut u64) -> Result<u64, PolicyError> {
        let len = u32::try_from(message.len())
            .map_err(|_| malformed(call, "the message is larger than wasm32 addresses"))?;
        let ptr = enter(
            &mut self.store,
            call,
            &self.alloc,
            len,
            DEFAULT_CALL_FUEL,
            fuel,
        )?;
        if ptr == 0 && len != 0 {
            return Err(malformed(call, "the module could not hold the message"));
        }
        self.memory
            .write(&mut self.store, ptr as usize, message)
            .map_err(|error| malformed(call, format!("cannot write the message: {error}")))?;

        let buffer = (ptr, len);
        let budget = budget(len);
        let store = &mut self.store;
        let result = match call {
            Call::Init => enter(store, call, &self.init, buffer, budget, fuel)?,
            Call::Observe => enter(store, call, &self.observe, buffer, budget, fuel)?,
            Call::Dispatch => enter(store, call, &self.dispatch, buffer, budget, fuel)?,
            Call::Preempt => enter(store, call, &self.preempt, buffer, budget, fuel)?,
            Call::Route => u64::from(enter(store, call, &self.route, buffer, budget, fuel)?),
            Call::Place => u64::from(enter(store, call, &self.place, buffer, budget, fuel)?),
            Call::Limits => return Err(malformed(call, "this call takes no message")),
        };

        enter(store, call, &self.free, buffer, DEFAULT_CALL_FUEL, fuel)?;
        Ok(result)
    }

    /// The limits the policy declares: `sched_limits`, which takes no message.
    fn read_limits(&mut self, fuel: &mut u64) -> Result<Option<Vec<u8>>, PolicyError> {
        let packed = enter(
            &mut self.store,
            Call::Limits,
            &self.limits,
            (),
            DEFAULT_CALL_FUEL,
            fuel,
        )?;
        self.take(Call::Limits, packed, fuel)
    }

    /// Copies a packed reply out of the module's memory and frees it; `None` for the empty reply.
    fn take(
        &mut self,
        call: Call,
        packed: u64,
        fuel: &mut u64,
    ) -> Result<Option<Vec<u8>>, PolicyError> {
        let Some((ptr, len)) = wire::unpack(packed) else {
            return Ok(None);
        };

        let mut reply = vec![0; len as usize];
        self.memory
            .read(&self.store, ptr as usize, &mut reply)
            .map_err(|error| {
                malformed(
                    call,
                    format!("the reply {ptr}+{len} lies outside the module's memory: {error}"),
                )
            })?;
        enter(
            &mut self.store,
            call,
            &self.free,
            (ptr, len),
            DEFAULT_CALL_FUEL,
            fuel,
        )?;
        Ok(Some(reply))
    }
}

/// The views an instance started from `snapshot` holds: those of the requests it states.
fn told_by(snapshot: &Snapshot) -> HashMap<RequestId, RequestView> {
    snapshot
        .requests
        .iter()
        .map(|view| (RequestId(view.id.0), view.clone()))
        .collect()
}

/// Advances `told`, the views a policy holds, by one event it observes, as the policy does
/// (`scheduling::event::Event`): an arrival or a restatement states the view, an admission makes
/// it running, an emission adds its tokens to those generated and takes them from those that
/// remain, and an end forgets it. The view's `last_emitted_at` is never the host's to state.
fn follow(told: &mut HashMap<RequestId, RequestView>, event: &Event) {
    let id = |request: &PolicyRequestId| RequestId(request.0);
    match event {
        Event::Arrived { request } | Event::Updated { request } => {
            told.insert(RequestId(request.id.0), request.clone());
        }
        Event::Admitted { request } => {
            if let Some(view) = told.get_mut(&id(request)) {
                view.state = Standing::Running;
            }
        }
        Event::Emitted {
            request, tokens, ..
        } => {
            if let Some(view) = told.get_mut(&id(request)) {
                view.generated_tokens = view.generated_tokens.saturating_add(*tokens);
                view.remaining_tokens = view.remaining_tokens.saturating_sub(*tokens);
            }
        }
        Event::Finished { request, .. } => {
            told.remove(&id(request));
        }
        _ => {}
    }
}

/// Instantiates `module` from `parameters` and `snapshot` (`init`), reads its limits once
/// (`limits`) and checks them against `ceilings`. `calls` and `fuel` gather the two calls.
fn start(
    module: &Module,
    parameters: &Parameters,
    snapshot: Snapshot,
    ceilings: &Limits,
    calls: &mut u64,
    fuel: &mut u64,
) -> Result<(Instance, Limits), PolicyError> {
    let mut instance = Instance::new(module)?;

    let instantiation = Instantiation {
        parameters: parameters.clone(),
        state: snapshot,
    };
    let message = encoded(Call::Init, wire::encode(&instantiation))?;
    *calls += 1;
    let packed = instance.exchange(Call::Init, &message, fuel)?;
    instance.take(Call::Init, packed, fuel)?;

    *calls += 1;
    let reply = instance
        .read_limits(fuel)?
        .ok_or_else(|| malformed(Call::Limits, "the policy replied nothing"))?;
    let limits: Limits = decoded(Call::Limits, wire::decode(&reply))?;
    check(&limits, ceilings)?;

    Ok((instance, limits))
}

/// The fuel an export is entered with for a message of `len` bytes: [`DEFAULT_CALL_FUEL`], and
/// [`MESSAGE_FUEL_PER_BYTE`] more for each byte of the message.
fn budget(len: u32) -> u64 {
    DEFAULT_CALL_FUEL.saturating_add(MESSAGE_FUEL_PER_BYTE.saturating_mul(u64::from(len)))
}

/// Enters one export with a fresh fuel budget of `budget`, adds what it consumed to `fuel`, and
/// reads a failure as the fault it is.
fn enter<P: WasmParams, R: WasmResults>(
    store: &mut Store<()>,
    call: Call,
    export: &TypedFunc<P, R>,
    params: P,
    budget: u64,
    fuel: &mut u64,
) -> Result<R, PolicyError> {
    store
        .set_fuel(budget)
        .map_err(|error| malformed(call, format!("cannot set the call's fuel: {error}")))?;
    let result = export.call(&mut *store, params);
    let left = store.get_fuel().unwrap_or(0);
    *fuel = fuel.saturating_add(budget.saturating_sub(left));

    result.map_err(|error| match error.downcast_ref::<Trap>() {
        Some(Trap::OutOfFuel) => PolicyError::Fuel { call },
        Some(&trap) => PolicyError::Trap { call, trap },
        None => malformed(call, format!("the call failed outside the module: {error}")),
    })
}

/// The export `name` with the signature the interface gives it.
fn bind<P: WasmParams, R: WasmResults>(
    store: &mut Store<()>,
    instance: &wasmtime::Instance,
    name: &str,
) -> Result<TypedFunc<P, R>, PolicyError> {
    instance.get_typed_func(&mut *store, name).map_err(|error| {
        PolicyError::Instantiate(format!("`{name}` is not the interface's: {error}"))
    })
}

/// The bytes of a message for `call`, as `wire::encode` encoded it.
fn encoded(
    call: Call,
    message: Result<Vec<u8>, serde_json::Error>,
) -> Result<Vec<u8>, PolicyError> {
    message.map_err(|error| malformed(call, format!("cannot encode the message: {error}")))
}

/// A reply of `call`, decoded (`wire::decode`).
fn decoded<T>(call: Call, reply: Result<T, serde_json::Error>) -> Result<T, PolicyError> {
    reply.map_err(|error| malformed(call, format!("the reply does not decode: {error}")))
}

fn malformed(call: Call, message: impl Into<String>) -> PolicyError {
    PolicyError::Malformed {
        call,
        message: message.into(),
    }
}

/// Refuses limits outside `ceilings` (`H/limits.rs:30-93`): a count that is zero or above its
/// ceiling, an unbounded queue against a bounded ceiling or a bound above it, a prefetch threshold
/// below the floor, and a starvation bound that is zero or above 30 s or the ceiling's. A limit
/// is refused rather than clamped, since the policy's own decisions are sized to the limits it
/// declared.
fn check(limits: &Limits, ceilings: &Limits) -> Result<(), PolicyError> {
    let counts = [
        (
            "chunked_prefill_size",
            u64::from(limits.chunked_prefill_size),
            u64::from(ceilings.chunked_prefill_size),
        ),
        (
            "max_batch",
            limits.max_batch as u64,
            ceilings.max_batch as u64,
        ),
        (
            "max_num_batched_tokens",
            u64::from(limits.max_num_batched_tokens),
            u64::from(ceilings.max_num_batched_tokens),
        ),
    ];
    for (name, limit, ceiling) in counts {
        if limit == 0 {
            return Err(PolicyError::Limits(format!("{name} is zero")));
        }
        if limit > ceiling {
            return Err(PolicyError::Limits(format!(
                "{name} is {limit}, above the ceiling of {ceiling}"
            )));
        }
    }

    // An unbounded queue exceeds any bound; a policy may bound its queue tighter than the ceiling.
    match (limits.max_queued_requests, ceilings.max_queued_requests) {
        (Some(queued), Some(ceiling)) if queued > ceiling => {
            return Err(PolicyError::Limits(format!(
                "max_queued_requests is {queued}, above the ceiling of {ceiling}"
            )));
        }
        (None, Some(ceiling)) => {
            return Err(PolicyError::Limits(format!(
                "max_queued_requests is unbounded, above the ceiling of {ceiling}"
            )));
        }
        _ => {}
    }

    // A policy may prefetch more eagerly than the floor, never more sparingly.
    if limits.prefetch_min_tokens < ceilings.prefetch_min_tokens {
        return Err(PolicyError::Limits(format!(
            "prefetch_min_tokens is {}, below the floor of {}",
            limits.prefetch_min_tokens, ceilings.prefetch_min_tokens
        )));
    }

    let bound = ceilings.timeout.min(MAX_TIMEOUT);
    if limits.timeout.0 == 0 || limits.timeout > bound {
        return Err(PolicyError::Limits(format!(
            "timeout is {} ns, outside (0, {}] ns",
            limits.timeout.0, bound.0
        )));
    }
    Ok(())
}

/// The parameters a scheduler instantiates its policy with (SD §5.2, §7.1): the package's
/// arguments, with a step's token budget, where they state none, the scheduler's own `config`
/// resolved (`SchedulerConfig::max_num_batched_tokens`: the deployment's, else the widest prefill
/// bucket its workers captured). The policy's declared limits budget every step, so a package
/// that leaves the budget to the deployment is given the deployment's, not the guest's default.
pub(super) fn budgeted(mut parameters: Parameters, config: &SchedulerConfig) -> Parameters {
    if parameters.max_num_batched_tokens.is_none() {
        parameters.max_num_batched_tokens = u32::try_from(config.max_num_batched_tokens).ok();
    }
    parameters
}

/// The ceilings a scheduler's policy declares its limits within (SD §5.2), from what the
/// workers of its groups captured: the step's tokens and a prefill chunk at most the widest
/// prefill bucket's tokens, its calls at most the largest bucket's rows, a prefetch threshold of
/// at least 256 tokens and a starvation bound of at most 30 s. A bound no captured bucket states
/// (no group captured a prefill, or any, bucket: no such batch is formed) does not bind. The queue
/// bound is the policy's own.
pub fn ceilings<'a>(capacities: impl IntoIterator<Item = &'a Capacity>) -> Limits {
    let buckets: Vec<&Shape> = capacities
        .into_iter()
        .flat_map(|capacity| &capacity.buckets)
        .collect();

    let tokens = buckets
        .iter()
        .filter(|shape| shape.variant.stage() == "prefill")
        .map(|shape| shape.tokens)
        .max()
        .unwrap_or(u32::MAX);
    let rows = buckets
        .iter()
        .map(|shape| shape.rows as usize)
        .max()
        .unwrap_or(usize::MAX);

    Limits {
        chunked_prefill_size: tokens,
        max_queued_requests: None,
        max_batch: rows,
        max_num_batched_tokens: tokens,
        prefetch_min_tokens: PREFETCH_FLOOR_TOKENS,
        timeout: MAX_TIMEOUT,
    }
}

/// The policy generation (SD §3.9).
impl Scheduler {
    /// A new generation's policy serves from this step on, instantiated on the scheduler's state
    /// now with its package's parameters ([`budgeted`]); the one it replaces is kept for rollback.
    /// On failure the serving policy is untouched and the error is the reply to `SetPolicy`.
    pub(super) fn set_policy(
        &mut self,
        module: Arc<Module>,
        parameters: Parameters,
    ) -> Result<(), Error> {
        let snapshot = self.snapshot();
        let ceilings = ceilings(self.groups.iter().map(|group| &group.capacity));
        let parameters = budgeted(parameters, &self.config);
        self.policy.switch(module, parameters, snapshot, &ceilings)
    }

    /// The serving policy faulted during a step (SD §5.2 "Faults"): the policy it replaced, or
    /// with none a fresh instance of it, serves again from a fresh snapshot and the engine is told
    /// with `Output::PolicyRolledBack`. The scheduler fails with the fault when that instantiation
    /// faults too, or when the faulted instance was itself made so and had not yet decided a step.
    pub(super) fn policy_failed(&mut self, fault: PolicyError) {
        if self.policy.recoverable() {
            let snapshot = self.snapshot();
            if self.policy.recover(snapshot).is_ok() {
                self.emit(Output::PolicyRolledBack(fault.to_string()));
                return;
            }
        }
        self.fail(Error::Policy(fault));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::OnceLock;

    use scheduling::End;
    use scheduling::Nanos;
    use scheduling::ids::{PageId, RequestId as GuestId, StageId};
    use scheduling::time::Duration;
    use scheduling::view::{
        PageView, Phase as GuestPhase, PhaseView, RequestClass, RequestStatus as GuestStatus,
    };

    use super::super::view::tests::waiting;
    use super::super::view::view_of;
    use super::*;

    /// The packaged policy, `arrival`, built for `wasm32-unknown-unknown` from this tree once per
    /// test process, into a target directory of its own (`it-policy`, as `testkit` builds it), and
    /// compiled on first use by [`load`].
    pub(crate) fn packaged() -> Arc<Module> {
        static MODULE: OnceLock<Arc<Module>> = OnceLock::new();

        let module = MODULE.get_or_init(|| {
            let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
            let target = std::env::var_os("CARGO_TARGET_DIR")
                .map_or_else(|| workspace.join("target"), PathBuf::from)
                .join("it-policy");
            let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());

            let mut build = Command::new(cargo);
            build
                .current_dir(&workspace)
                .args(["build", "-q", "--release"]);
            build.args(["--target", "wasm32-unknown-unknown", "--target-dir"]);
            build.arg(&target);
            build.args(["-p", "scheduling-policy-arrival"]);
            let output = build.output().unwrap();
            assert!(
                output.status.success(),
                "building the packaged policy: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let wasm = target.join("wasm32-unknown-unknown/release/scheduling_policy_arrival.wasm");
            Arc::new(load(&std::fs::read(wasm).unwrap()).unwrap())
        });
        Arc::clone(module)
    }

    /// The arguments a revision of the packaged policy runs with when its deployment states
    /// `stated` (TOML): the package's declared defaults, replaced by `stated`.
    pub(crate) fn packaged_arguments(stated: &str) -> Parameters {
        crate::modules::load::parameters(&testkit::serving::policy_revision(stated)).unwrap()
    }

    /// The limits a [`toy`] module declares: chunks and steps of 8192 tokens, 128 calls a step,
    /// a 256-token prefetch threshold, a 30 s starvation bound and an unbounded queue, which the
    /// ceilings of workers that captured no bucket ([`ceilings`] of none) hold.
    pub(crate) fn toy_limits() -> Limits {
        Limits {
            chunked_prefill_size: 8192,
            max_queued_requests: None,
            max_batch: 128,
            max_num_batched_tokens: 8192,
            prefetch_min_tokens: 256,
            timeout: Nanos(30_000_000_000),
        }
    }

    /// A module that speaks the interface's exports, whose `sched_init` runs `body` (a `wat`
    /// expression of type `i64`), whose `sched_limits` replies [`toy_limits`], whose
    /// `sched_dispatch` replies a decision that names nothing, and whose other exports reply
    /// nothing, or for `sched_route` the first rank.
    pub(crate) fn toy(body: &str) -> Arc<Module> {
        exports(body, "(i32.const 0)")
    }

    /// A routing instance of the exports of [`toy`] whose `sched_init` initializes and whose
    /// `sched_route` runs `route`, a `wat` expression of type `i32` over the message's address
    /// `$at` and length `$len`.
    pub(crate) fn router(route: &str) -> Arc<Module> {
        exports("(i64.const 0)", route)
    }

    /// The module of [`toy`] and [`router`]: `init` is `sched_init`'s body, `route` is
    /// `sched_route`'s.
    fn exports(init: &str, route: &str) -> Arc<Module> {
        Arc::new(load(toy_text(init, route, scheduling::INTERFACE_VERSION).as_bytes()).unwrap())
    }

    /// The text of the module of [`exports`], which states `version` as the version of the
    /// interface it implements.
    fn toy_text(init: &str, route: &str, version: u32) -> String {
        // Every message is written at 1024, and the replies lie at the end of the one page, past
        // any snapshot a test's scheduler sends: the limits at 64768 and the decision at 65024,
        // each returned as `ptr << 32 | len`.
        let reply = |at: u64, bytes: &[u8]| {
            let escaped: String = bytes.iter().map(|byte| format!("\\{byte:02x}")).collect();
            (
                (at << 32) | bytes.len() as u64,
                format!(r#"(data (i32.const {at}) "{escaped}")"#),
            )
        };
        let (limits, limits_data) = reply(64768, &wire::encode(&toy_limits()).unwrap());
        let (empty, empty_data) = reply(65024, &wire::encode(&DispatchOutput::default()).unwrap());
        format!(
            r#"(module
                (memory (export "memory") 1)
                {limits_data}
                {empty_data}
                (func (export "sched_interface_version") (result i32) (i32.const {version}))
                (func (export "sched_alloc") (param i32) (result i32) (i32.const 1024))
                (func (export "sched_free") (param i32 i32))
                (func (export "sched_init") (param i32 i32) (result i64) {init})
                (func (export "sched_observe") (param i32 i32) (result i64) (i64.const 0))
                (func (export "sched_dispatch") (param i32 i32) (result i64) (i64.const {empty}))
                (func (export "sched_preempt") (param i32 i32) (result i64) (i64.const 0))
                (func (export "sched_route") (param $at i32) (param $len i32) (result i32) {route})
                (func (export "sched_place") (param i32 i32) (result i32) (i32.const 0))
                (func (export "sched_limits") (result i64) (i64.const {limits})))"#
        )
    }

    /// Arguments that state a deployment's limits, and the ceilings of workers that captured
    /// exactly that much.
    fn limited(tokens: u32, calls: usize) -> (Parameters, Limits) {
        let parameters = packaged_arguments(&format!(
            "chunked_prefill_size = {tokens}\nmax_batch = {calls}\nmax_num_batched_tokens = {tokens}"
        ));
        let ceilings = Limits {
            chunked_prefill_size: tokens,
            max_batch: calls,
            max_num_batched_tokens: tokens,
            ..ceilings([])
        };
        (parameters, ceilings)
    }

    /// `parameters` with the waiting queue admitting by the request's priority, then arrival.
    fn by_priority(parameters: Parameters) -> Parameters {
        Parameters {
            order: Some(scheduling::QueueOrder::Priority),
            ..parameters
        }
    }

    fn room() -> Room {
        Room {
            token_budget: 512,
            call_budget: 8,
            page_tokens: 16,
            ..Room::default()
        }
    }

    /// The packaged policy declares the limits its arguments state, admits what arrived in the
    /// order its arguments name, and costs `init` and `limits` once and one `observe` and one
    /// `dispatch` per step.
    #[test]
    fn a_packaged_policy_admits_what_arrived_in_its_order() {
        let (parameters, ceilings) = limited(512, 8);
        // Request 1 arrives first with the less urgent priority (a lower value is more urgent).
        let first = waiting(1, 5, &[1, 2, 3]);
        let second = waiting(2, 1, &[4, 5]);
        let mut requests = HashMap::from([(first.id, first), (second.id, second)]);

        for (name, parameters, admitted) in [
            ("arrival", parameters.clone(), [1, 2]),
            ("priority", by_priority(parameters), [2, 1]),
        ] {
            let mut policy =
                Policy::new(packaged(), parameters, Snapshot::default(), &ceilings).unwrap();
            assert_eq!(policy.limits.max_batch, 8, "{name}");
            assert_eq!(policy.limits.max_num_batched_tokens, 512, "{name}");
            assert_eq!(policy.calls, 2, "{name}: init and limits");

            for id in [1, 2] {
                let request = &requests[&RequestId(id)];
                policy.events.push(Event::Arrived {
                    request: view_of(request, &[]),
                });
            }
            let output = policy.dispatch(&room(), &mut requests, &[]).unwrap();
            assert_eq!(output.admit, admitted.map(GuestId).to_vec(), "{name}");
            assert!(output.reject.is_empty(), "{name}");
            assert_eq!(policy.calls, 4, "{name}: one observe and one dispatch");
            assert!(policy.events.is_empty(), "{name}: the events were told");

            let fuel = policy.fuel;
            policy.dispatch(&room(), &mut requests, &[]).unwrap();
            assert!(policy.fuel > fuel, "{name}: every call is metered");
        }
    }

    /// Limits above a ceiling refuse the instantiation; a switch they refuse leaves the serving
    /// policy as it was.
    #[test]
    fn limits_above_a_ceiling_refuse_the_policy() {
        let (parameters, ceilings) = limited(512, 8);
        let (wide, _) = limited(1024, 8);
        let refused = Policy::new(packaged(), wide.clone(), Snapshot::default(), &ceilings);
        assert!(matches!(refused, Err(PolicyError::Limits(_))));

        let mut policy =
            Policy::new(packaged(), parameters, Snapshot::default(), &ceilings).unwrap();
        let refused = policy.switch(
            packaged(),
            by_priority(wide),
            Snapshot::default(),
            &ceilings,
        );
        assert!(matches!(
            refused,
            Err(Error::Policy(PolicyError::Limits(_)))
        ));
        assert_eq!(policy.limits.max_num_batched_tokens, 512);
        assert!(policy.previous.is_none());
    }

    /// A switch serves the new policy and keeps the replaced one for exactly one rollback; a
    /// committed switch has nothing to roll back to.
    #[test]
    fn a_switch_keeps_the_replaced_policy_for_one_rollback() {
        let (serving, ceilings) = limited(512, 8);
        let (next, _) = limited(256, 4);
        let mut policy = Policy::new(packaged(), serving, Snapshot::default(), &ceilings).unwrap();

        policy
            .switch(
                packaged(),
                by_priority(next),
                Snapshot::default(),
                &ceilings,
            )
            .unwrap();
        assert_eq!(policy.limits.max_batch, 4);
        assert!(policy.previous.is_some());

        policy.rollback(Snapshot::default()).unwrap();
        assert_eq!(policy.limits.max_batch, 8);
        assert!(matches!(
            policy.rollback(Snapshot::default()),
            Err(Error::InvalidRequest(_))
        ));
    }

    /// A trap, exhausted fuel and a reply outside the module's memory are faults of the module,
    /// each named by the call it happened in; a switch to such a module leaves the serving policy dispatching.
    #[test]
    fn a_faulting_module_is_refused_by_name() {
        let ceilings = toy_limits();
        let fault = |body: &str| {
            Policy::new(
                toy(body),
                Parameters::default(),
                Snapshot::default(),
                &ceilings,
            )
            .err()
        };

        assert!(matches!(
            fault("(unreachable)"),
            Some(PolicyError::Trap {
                call: Call::Init,
                ..
            })
        ));
        assert!(matches!(
            fault("(loop $spin (br $spin)) (i64.const 0)"),
            Some(PolicyError::Fuel { call: Call::Init })
        ));
        // A reply at an address past the module's one page of memory.
        assert!(matches!(
            fault("(i64.const 0x0010000000000010)"),
            Some(PolicyError::Malformed {
                call: Call::Init,
                ..
            })
        ));

        let mut policy = Policy::new(
            packaged(),
            packaged_arguments("max_num_batched_tokens = 8192"),
            Snapshot::default(),
            &ceilings,
        )
        .unwrap();
        let refused = policy.switch(
            toy("(unreachable)"),
            Parameters::default(),
            Snapshot::default(),
            &ceilings,
        );
        assert!(refused.is_err());
        assert!(policy.previous.is_none());
        let mut requests = HashMap::new();
        policy.dispatch(&room(), &mut requests, &[]).unwrap();
    }

    /// The ceilings are the widest prefill bucket's tokens and the largest bucket's rows; a
    /// bound no bucket states does not bind.
    #[test]
    fn the_ceilings_are_what_the_workers_captured() {
        use crate::batch::Variant;

        let mut capacity = super::super::tests::capacity();
        capacity.buckets = vec![
            Shape::prefill(4, 2048, Variant::PrefillOutput),
            Shape::prefill(1, 4096, Variant::PrefillWrite),
            Shape::decode(64, Variant::DecodeSampled),
        ];
        let limits = ceilings([&capacity]);
        assert_eq!(limits.max_num_batched_tokens, 4096);
        assert_eq!(limits.chunked_prefill_size, 4096);
        assert_eq!(limits.max_batch, 64);
        assert_eq!(limits.prefetch_min_tokens, 256);
        assert_eq!(limits.timeout, Nanos(30_000_000_000));

        let unbound = ceilings([&super::super::tests::capacity()]);
        assert_eq!(unbound.max_num_batched_tokens, u32::MAX);
        assert_eq!(unbound.max_batch, usize::MAX);
    }

    /// A module states the version of the policy interface it was compiled against, and one of
    /// another version is refused before any call; the packaged policy states this engine's.
    #[test]
    fn a_module_of_another_interface_version_is_refused() {
        let ceilings = toy_limits();
        assert!(
            Policy::new(
                packaged(),
                limited(512, 8).0,
                Snapshot::default(),
                &ceilings
            )
            .is_ok()
        );

        let stale = toy_text(
            "(i64.const 0)",
            "(i32.const 0)",
            scheduling::INTERFACE_VERSION - 1,
        );
        let refused = Policy::new(
            Arc::new(load(stale.as_bytes()).unwrap()),
            Parameters::default(),
            Snapshot::default(),
            &ceilings,
        );
        assert!(
            matches!(&refused, Err(PolicyError::Instantiate(message)) if message.contains("version")),
            "{:?}",
            refused.err()
        );
    }

    /// A page id as the leading eight bytes of a block's key give them: spread over every bit.
    fn spread(number: u64) -> PageId {
        let mut mixed = number.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        PageId(mixed ^ (mixed >> 31))
    }

    /// An instant on the host's monotonic clock some months after it started.
    const NOW: u64 = 12_300_000_000_000_000;

    /// The view a scheduler states of request `id` of a decode load: a 512-token prompt and
    /// 1024 tokens to generate, of which it has generated `generated`, holding `pages` blocks.
    fn decoding(id: u64, state: GuestStatus, generated: u32, pages: u32) -> RequestView {
        let running = state == GuestStatus::Running;
        RequestView {
            id: GuestId(id),
            state,
            phase: running.then_some(GuestPhase::Decode.id()),
            priority: 0,
            arrived_at: Nanos(NOW + id * 1_000_000),
            admitted_at: running.then_some(Nanos(NOW + id * 1_000_000)),
            prompt_tokens: 512,
            prefill_tokens: 512,
            generated_tokens: generated,
            cached_prefix_tokens: 0,
            prefilled_tokens: 512,
            cached_pages: pages,
            last_emitted_at: None,
            class: RequestClass::Token,
            waiting_transfer: false,
            next_tokens: 1,
            reservation_units: 0,
            prefetch_tokens: 0,
            refill_tokens: None,
            remaining_tokens: 1024 - generated,
            migratable: true,
            cache_rank: None,
        }
    }

    /// A 64-token block of a decode load's cache, last used at `used`.
    fn block(page: PageId, used: u64, evictable: bool) -> PageView {
        PageView {
            id: page,
            tokens: 64,
            last_used: Nanos(NOW + used),
            on_host: false,
            evictable,
        }
    }

    /// The phases a text deployment's snapshot states.
    fn text_phases() -> Vec<PhaseView> {
        [GuestPhase::Prefill, GuestPhase::Decode]
            .map(|kind| PhaseView {
                id: kind.id(),
                kind,
                block: StageId(0),
            })
            .to_vec()
    }

    /// The fuel one call of the policy consumed, the call's message and reply included, and its
    /// reply.
    fn metered(policy: &mut Policy, call: Call, message: Vec<u8>) -> (u64, Option<Vec<u8>>) {
        let before = policy.fuel;
        let reply = policy.call(call, Ok(message)).unwrap();
        (policy.fuel - before, reply)
    }

    /// A step of a decode load in which most requests end together: 64 requests decode, and 60
    /// of them end and give up the 16 blocks each held, which the step tells in 249 events
    /// (an advance and an emission per request, an end and one release per ending request, the
    /// step's end; the views of the 4 that run on moved by nothing their emissions do not state).
    /// The step's `observe` and `dispatch`, and those of a step in which all 64 decode on, each
    /// consume at most half of [`DEFAULT_CALL_FUEL`]: half leaves room for a step twice as heavy
    /// (twice the requests ending, or twice the blocks each) before a call runs out, while
    /// telling each block as a restated page, or decoding each event through a buffered copy,
    /// exceeds it. The policy saw the release: it offloads every block given up, once.
    #[test]
    fn a_step_in_which_most_requests_end_stays_well_within_a_calls_fuel() {
        const RUNNING: u64 = 64;
        const ENDING: u64 = 60;
        const BLOCKS: u64 = 16;
        const LIMIT: u64 = DEFAULT_CALL_FUEL / 2;
        let held = |id: u64| (0..BLOCKS).map(move |index| spread(id * BLOCKS + index));

        let (parameters, ceilings) = limited(8192, 128);
        let snapshot = Snapshot {
            phases: text_phases(),
            ..Snapshot::default()
        };
        let mut policy = Policy::new(packaged(), parameters, snapshot, &ceilings).unwrap();
        let observe = |policy: &mut Policy, events: &[Event]| {
            metered(policy, Call::Observe, wire::encode(&events).unwrap()).0
        };

        // The requests arrive, are admitted and fill their blocks over earlier steps.
        let arrivals: Vec<Event> = (1..=RUNNING)
            .map(|id| Event::Arrived {
                request: decoding(id, GuestStatus::Waiting, 0, 0),
            })
            .chain((1..=RUNNING).map(|id| Event::Admitted {
                request: GuestId(id),
            }))
            .collect();
        observe(&mut policy, &arrivals);
        for id in 1..=RUNNING {
            let filled: Vec<Event> = held(id)
                .enumerate()
                .map(|(used, page)| Event::Cached {
                    page: block(page, id * BLOCKS + used as u64, false),
                })
                .collect();
            observe(&mut policy, &filled);
        }

        // A step in which every request decodes on.
        let progress = |id: u64| {
            [
                Event::Advanced {
                    request: GuestId(id),
                    phase: GuestPhase::Decode.id(),
                    tokens: 1,
                },
                Event::Emitted {
                    request: GuestId(id),
                    tokens: 1,
                    at: Nanos(NOW + id),
                },
            ]
        };
        let mut steady: Vec<Event> = (1..=RUNNING).flat_map(progress).collect();
        steady.push(Event::Stepped {
            step: 1000,
            duration: Duration(9_500_000),
        });

        // The step in which 60 of them end.
        let mut ending: Vec<Event> = (1..=RUNNING).flat_map(progress).collect();
        for id in 1..=ENDING {
            ending.push(Event::Finished {
                request: GuestId(id),
                end: End::Completed,
            });
            ending.push(Event::Released {
                pages: held(id).collect(),
            });
        }
        ending.push(Event::Stepped {
            step: 1001,
            duration: Duration(9_500_000),
        });
        assert_eq!(ending.len(), 249);

        let room = Room {
            token_budget: 8192,
            call_budget: 128,
            page_tokens: 64,
            ..Room::default()
        };
        for (step, events) in [
            ("decoding", &steady),
            ("ending", &ending),
            ("after", &steady),
        ] {
            let observed = observe(&mut policy, events);
            let (dispatched, reply) =
                metered(&mut policy, Call::Dispatch, wire::encode(&room).unwrap());
            assert!(observed <= LIMIT, "{step}: observe took {observed} fuel");
            assert!(
                dispatched <= LIMIT,
                "{step}: dispatch took {dispatched} fuel"
            );

            let output: DispatchOutput = wire::decode(&reply.unwrap()).unwrap();
            let released = if step == "ending" { ENDING * BLOCKS } else { 0 };
            assert_eq!(output.offload.len() as u64, released, "{step}");
        }
    }

    /// A steady step of a decode load of 256 running requests (a `max_running_requests` in the
    /// hundreds): each request's decode was scheduled and applied, so the scheduler marked it
    /// and told its advance and its emission, and nothing else of its view moved. The policy is
    /// not told the views again, so the step's `observe` and `dispatch` together consume at most
    /// half of [`DEFAULT_CALL_FUEL`], where restating each view costs about as much as a call
    /// has; and its view stays exact: every request decodes, and in the next step a request
    /// with as many decodes in flight as its group's call window allows is told it has no call
    /// ready, and is not named.
    #[test]
    fn a_steady_step_of_many_decoding_requests_stays_well_within_a_calls_fuel() {
        use crate::batch::Phase;

        const RUNNING: u64 = 256;
        const LIMIT: u64 = DEFAULT_CALL_FUEL / 2;
        let mut bench = super::super::tests::bench(4096);
        let scheduler = &mut bench.scheduler;
        let group = scheduler.groups[0].id();
        scheduler.groups[0].capacity.phases =
            (1 << Phase::Prefill.index()) | (1 << Phase::Decode.index());

        // Requests of a 512-token prompt that generated 8 tokens of their 16 and decode on group
        // A, with nothing in flight.
        for id in 1..=RUNNING {
            let mut request = waiting(id, 0, &[1; 512]);
            request.status = RequestStatus::Running;
            request.queued = false;
            request.placement[Phase::Prefill.index()] = Some(group);
            request.placement[Phase::Decode.index()] = Some(group);
            request.token_ids.extend([7; 8]);
            request.num_computed_tokens = 520;
            scheduler.requests.insert(request.id, request);
            scheduler.running.push(RequestId(id));
        }
        let (parameters, ceilings) = limited(8192, 512);
        let snapshot = scheduler.snapshot();
        scheduler.policy = Policy::new(packaged(), parameters, snapshot, &ceilings).unwrap();

        // One decode of each request ran and was applied, as `schedule` and `update` do.
        let step = |scheduler: &mut Scheduler, number: u64| {
            for id in (1..=RUNNING).map(RequestId) {
                if let Some(request) = scheduler.requests.get_mut(&id) {
                    request.scheduled[Phase::Decode.index()] += 1;
                    request.completed[Phase::Decode.index()] += 1;
                    request.num_computed_tokens += 1;
                    request.token_ids.push(7);
                }
                scheduler.mark(id);
                scheduler.policy.events.push(Event::Advanced {
                    request: GuestId(id.0),
                    phase: Phase::Decode.id(),
                    tokens: 1,
                });
                scheduler.policy.events.push(Event::Emitted {
                    request: GuestId(id.0),
                    tokens: 1,
                    at: Nanos(NOW + number),
                });
            }
            scheduler.policy.events.push(Event::Stepped {
                step: number,
                duration: Duration(9_500_000),
            });
        };
        let room = Room {
            token_budget: 8192,
            call_budget: 512,
            page_tokens: 64,
            ..Room::default()
        };
        let decoded = |output: &DispatchOutput| -> Vec<u64> {
            output.work.iter().map(|work| work.request.0).collect()
        };

        step(scheduler, 1);
        let fuel = scheduler.policy.fuel;
        let output = scheduler
            .policy
            .dispatch(&room, &mut scheduler.requests, &scheduler.groups)
            .unwrap();
        let spent = scheduler.policy.fuel - fuel;
        assert!(spent <= LIMIT, "observe and dispatch took {spent} fuel");
        assert_eq!(decoded(&output), (1..=RUNNING).collect::<Vec<_>>());

        // Request 1 now has two decodes in flight, its group's call window.
        step(scheduler, 2);
        if let Some(request) = scheduler.requests.get_mut(&RequestId(1)) {
            request.scheduled[Phase::Decode.index()] += 2;
        }
        let output = scheduler
            .policy
            .dispatch(&room, &mut scheduler.requests, &scheduler.groups)
            .unwrap();
        assert_eq!(decoded(&output), (2..=RUNNING).collect::<Vec<_>>());
    }

    /// A policy instantiates on the snapshot of a warm cache: 128 running requests and the 4096
    /// blocks of 64 tokens a cache of 262144 tokens holds, none of them held. Its `init` decodes
    /// far more than [`DEFAULT_CALL_FUEL`] allows a step's call, and is given fuel for every byte
    /// of the snapshot. A step costs `dispatch` what the step changed, not what the cache holds:
    /// the blocks the snapshot states were the instance before's to offload, and a block given
    /// up since is offloaded once.
    #[test]
    fn a_policy_instantiates_on_the_snapshot_of_a_warm_cache() {
        const LIMIT: u64 = DEFAULT_CALL_FUEL / 2;
        let (parameters, ceilings) = limited(8192, 128);
        let snapshot = Snapshot {
            requests: (1..=128)
                .map(|id| decoding(id, GuestStatus::Running, 500, 16))
                .collect(),
            pages: (0..4096)
                .map(|number| block(spread(number), number, true))
                .collect(),
            phases: text_phases(),
            ..Snapshot::default()
        };

        let mut policy = Policy::new(packaged(), parameters, snapshot, &ceilings).unwrap();
        assert!(
            policy.fuel > DEFAULT_CALL_FUEL,
            "the snapshot is one a step's budget cannot decode"
        );
        let room = Room {
            token_budget: 8192,
            call_budget: 128,
            page_tokens: 64,
            ..Room::default()
        };
        for released in [Vec::new(), vec![spread(5000)], Vec::new()] {
            // A block cached since the snapshot, which a request held and gave up.
            for (number, &page) in (5000..).zip(&released) {
                policy.events.push(Event::Cached {
                    page: block(page, number, false),
                });
            }
            policy.events.push(Event::Released {
                pages: released.clone(),
            });
            let fuel = policy.fuel;
            let output = policy.dispatch(&room, &mut HashMap::new(), &[]).unwrap();
            let spent = policy.fuel - fuel;
            assert!(spent <= LIMIT, "observe and dispatch took {spent} fuel");
            assert_eq!(output.offload, released);
        }
    }
}
