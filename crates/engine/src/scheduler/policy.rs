//! The scheduling policy's host (`specs/scheduler-design.md` §5): [`Policy`], a WASM module's
//! instance with the seven calls of the policy interface, its switch and rollback (§3.9), the
//! check of the limits it declares, and [`PolicyError`], the faults of a policy.
//!
//! The interface is the guest crate's (`scheduling_guest`): every message is JSON of a guest type
//! written into the module's memory through `sched_alloc`, and a reply is a buffer the module
//! allocated, packed as `ptr << 32 | len` (`0` for none), or for `route` and `place` the chosen
//! index itself (`scheduling_guest::wire`). Every export the host enters, the two buffer exports
//! included, starts with a fuel budget of its own, so a policy that loops ends its call instead
//! of the step.
//!
//! The host keeps no view of its own: the policy holds the view, and the scheduler tells it what
//! changed as events (§5.2). A decision that names something that cannot be done is cut or
//! skipped where it is used (§3.4, §3.6), never a fault; a fault is a trap, exhausted fuel, a reply
//! that does not decode, or limits outside the ceilings.
//!
//! Policy modules are compiled by the engine `scheduling::Engine` configures (`Engine::load`),
//! which meters fuel; a module of another engine fails to instantiate here.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use ipc::batch::Shape;
use proto::request::RequestId;
use scheduling_guest::decision::{DispatchOutput, Room, Shortage, Victim};
use scheduling_guest::event::Event;
use scheduling_guest::limits::{Limits, Parameters};
use scheduling_guest::policy::Instantiation;
use scheduling_guest::time::Nanos;
use scheduling_guest::view::{CallView, RankView, ReplicaView, RequestView, Snapshot};
use scheduling_guest::wire::{self, Call};
use wasmtime::{Memory, Module, Store, Trap, TypedFunc, WasmParams, WasmResults};

use super::view::view_of;
use super::{Capacity, Error, Output, Request, RequestStatus, Scheduler, WorkerGroup};

/// The fuel one call into a policy module may consume, and each export the host enters for that
/// call (`H/instance.rs:32`): far above a step's `observe` and `dispatch` of a real policy, and a
/// bound on one that never returns.
const DEFAULT_CALL_FUEL: u64 = 10_000_000;

/// The longest starvation bound a policy may declare (`limits().timeout`), as `sched_ext` caps
/// `timeout_ms` at 30 seconds.
const MAX_TIMEOUT: Nanos = Nanos(30_000_000_000);

/// The smallest prefix-hit deepening a policy may declare (`prefetch_min_tokens`): a shorter hit is
/// computed faster than it is fetched.
const PREFETCH_FLOOR_TOKENS: u32 = 256;

/// The serving scheduling policy, and the module it replaced, kept for rollback (SD §2.1, §5.2).
///
/// Invariants: every state change a view names is in `events` (or its request in `dirty`)
/// before the next `dispatch`; `limits` is within the ceilings the policy was instantiated
/// under. The policy's own memory is never carried: every instantiation starts from `init` with
/// the snapshot of that moment.
pub struct Policy {
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
    /// The limits the serving policy declared: the step's token and call budgets, the chunk limit,
    /// the queue bound, the prefetch threshold and the starvation bound the scheduler enforces.
    pub limits: Limits,
    /// The state changes since the policy was last told, in the order they happened.
    pub events: Vec<Event>,
    /// The requests whose view changed since the policy was last told, each once
    /// (`Scheduler::mark`); `dispatch` tells each as one `Updated`.
    pub dirty: Vec<RequestId>,
    /// The calls of the interface that reached a module, over every instance of this policy.
    pub calls: u64,
    /// The fuel those calls consumed.
    pub fuel: u64,
}

/// A fault of a policy (SD §5.2): what ends its instance.
#[derive(Debug)]
pub enum PolicyError {
    /// The module does not instantiate as a policy: it does not link, lacks the memory or an
    /// export the interface calls, or its engine meters no fuel.
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
    pub fn new(
        module: Arc<Module>,
        parameters: Parameters,
        snapshot: Snapshot,
        ceilings: &Limits,
    ) -> Result<Self, PolicyError> {
        let (mut calls, mut fuel) = (0, 0);
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
            limits,
            events: Vec::new(),
            dirty: Vec::new(),
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
    pub fn switch(
        &mut self,
        module: Arc<Module>,
        parameters: Parameters,
        snapshot: Snapshot,
        ceilings: &Limits,
    ) -> Result<(), Error> {
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
        self.install(instance, limits, *ceilings);
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
    pub fn rollback(&mut self, snapshot: Snapshot) -> Result<(), Error> {
        let Some((module, parameters)) = self.previous.take() else {
            return Err(Error::InvalidRequest(
                "no replaced policy is kept to roll back to".into(),
            ));
        };

        let ceilings = self.ceilings;
        match start(
            &module,
            &parameters,
            snapshot,
            &ceilings,
            &mut self.calls,
            &mut self.fuel,
        ) {
            Ok((instance, limits)) => {
                self.module = module;
                self.parameters = parameters;
                self.install(instance, limits, ceilings);
                Ok(())
            }
            Err(fault) => {
                self.previous = Some((module, parameters));
                Err(Error::Policy(fault))
            }
        }
    }

    /// The trial of the last switch ends in its favour (SD §3.9): the module and arguments it
    /// replaced are dropped, and nothing is left to roll back to.
    pub fn commit(&mut self) {
        self.previous = None;
    }

    /// One step's decision (SD §5.2): every request marked since the last step is told once as
    /// `Updated` with its view now (and its mark cleared), the step's events go to `observe`, and
    /// `dispatch` is asked with `room`. `groups` are the scheduler's, which a view reads.
    ///
    /// # Errors
    ///
    /// The fault of `observe` or `dispatch`; an empty `dispatch` reply is
    /// [`PolicyError::Malformed`].
    pub fn dispatch(
        &mut self,
        room: &Room,
        requests: &mut HashMap<RequestId, Request>,
        groups: &[WorkerGroup],
    ) -> Result<DispatchOutput, PolicyError> {
        // A request whose fields changed since the policy was told is told again, once, whatever
        // changed. One that ended meanwhile was told by its `Finished`, after which the policy
        // forgets it.
        for id in self.dirty.drain(..) {
            let Some(request) = requests.get_mut(&id) else {
                continue;
            };
            request.dirty = false;
            if request.status != RequestStatus::Finished {
                self.events.push(Event::Updated {
                    request: view_of(request, groups),
                });
            }
        }

        // `observe` replies nothing a host reads.
        let events = std::mem::take(&mut self.events);
        self.call(Call::Observe, wire::encode(&events))?;

        let reply = self
            .call(Call::Dispatch, wire::encode(room))?
            .ok_or_else(|| malformed(Call::Dispatch, "the policy replied nothing"))?;
        decoded(Call::Dispatch, wire::decode(&reply))
    }

    /// The victims the policy names for a shortage, in its order; an empty reply names none.
    ///
    /// # Errors
    ///
    /// The fault of `preempt`.
    pub fn preempt(&mut self, shortage: &Shortage) -> Result<Vec<Victim>, PolicyError> {
        match self.call(Call::Preempt, wire::encode(shortage))? {
            Some(reply) => decoded(Call::Preempt, wire::decode(&reply)),
            None => Ok(Vec::new()),
        }
    }

    /// The index in `ranks` of the data-parallel rank that serves a new request: the engine's
    /// routing policy's call, never a scheduler's.
    ///
    /// # Errors
    ///
    /// The fault of `route`.
    pub fn route(
        &mut self,
        request: &RequestView,
        ranks: &[RankView],
    ) -> Result<usize, PolicyError> {
        self.index(Call::Route, wire::encode(&(request, ranks)))
    }

    /// The index in `replicas` of the group a request's first call of a phase runs on. The
    /// caller replaces an index that names no candidate (SD §3.5 `place`).
    ///
    /// # Errors
    ///
    /// The fault of `place`.
    pub fn place(
        &mut self,
        call: &CallView,
        replicas: &[ReplicaView],
    ) -> Result<usize, PolicyError> {
        self.index(Call::Place, wire::encode(&(call, replicas)))
    }

    /// Makes `instance` the serving one, with its limits and the ceilings they were checked
    /// against. The events gathered for the instance it replaces are dropped, since the snapshot
    /// `instance` started from states what they told; marked requests stay marked, and their views
    /// are told to it at its first `dispatch`.
    fn install(&mut self, instance: Instance, limits: Limits, ceilings: Limits) {
        self.instance = instance;
        self.limits = limits;
        self.ceilings = ceilings;
        self.events.clear();
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
                "its engine meters no fuel (compile it with `scheduling::Engine`): {error}"
            ))
        })?;

        let instance = wasmtime::Instance::new(&mut store, module, &[])
            .map_err(|error| PolicyError::Instantiate(error.to_string()))?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| PolicyError::Instantiate("it exports no `memory`".into()))?;

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
        let ptr = enter(&mut self.store, call, &self.alloc, len, fuel)?;
        if ptr == 0 && len != 0 {
            return Err(malformed(call, "the module could not hold the message"));
        }
        self.memory
            .write(&mut self.store, ptr as usize, message)
            .map_err(|error| malformed(call, format!("cannot write the message: {error}")))?;

        let buffer = (ptr, len);
        let result = match call {
            Call::Init => enter(&mut self.store, call, &self.init, buffer, fuel)?,
            Call::Observe => enter(&mut self.store, call, &self.observe, buffer, fuel)?,
            Call::Dispatch => enter(&mut self.store, call, &self.dispatch, buffer, fuel)?,
            Call::Preempt => enter(&mut self.store, call, &self.preempt, buffer, fuel)?,
            Call::Route => u64::from(enter(&mut self.store, call, &self.route, buffer, fuel)?),
            Call::Place => u64::from(enter(&mut self.store, call, &self.place, buffer, fuel)?),
            Call::Limits => return Err(malformed(call, "this call takes no message")),
        };

        enter(&mut self.store, call, &self.free, buffer, fuel)?;
        Ok(result)
    }

    /// The limits the policy declares: `sched_limits`, which takes no message.
    fn read_limits(&mut self, fuel: &mut u64) -> Result<Option<Vec<u8>>, PolicyError> {
        let packed = enter(&mut self.store, Call::Limits, &self.limits, (), fuel)?;
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
        enter(&mut self.store, call, &self.free, (ptr, len), fuel)?;
        Ok(Some(reply))
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

/// Enters one export with a fresh fuel budget, adds what it consumed to `fuel`, and reads a
/// failure as the fault it is.
fn enter<P: WasmParams, R: WasmResults>(
    store: &mut Store<()>,
    call: Call,
    export: &TypedFunc<P, R>,
    params: P,
    fuel: &mut u64,
) -> Result<R, PolicyError> {
    store
        .set_fuel(DEFAULT_CALL_FUEL)
        .map_err(|error| malformed(call, format!("cannot set the call's fuel: {error}")))?;
    let result = export.call(&mut *store, params);
    let left = store.get_fuel().unwrap_or(0);
    *fuel = fuel.saturating_add(DEFAULT_CALL_FUEL.saturating_sub(left));

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

/// The ceilings a scheduler's policy declares its limits within (SD §5.2), from what the
/// executors of its groups captured: the step's tokens and a prefill chunk at most the widest
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
    /// now; the one it replaces is kept for rollback. On failure the serving policy is untouched
    /// and the error is the reply to `SetPolicy`.
    pub(super) fn set_policy(
        &mut self,
        module: Arc<Module>,
        parameters: Parameters,
    ) -> Result<(), Error> {
        let snapshot = self.snapshot();
        let ceilings = ceilings(self.groups.iter().map(|group| &group.capacity));
        self.policy.switch(module, parameters, snapshot, &ceilings)
    }

    /// The serving policy faulted during a step: the policy it replaced serves again from a fresh
    /// snapshot and the engine is told with `Output::PolicyRolledBack`; with none to roll back to,
    /// or one that faults too, the scheduler fails with the fault.
    pub(super) fn policy_failed(&mut self, fault: PolicyError) {
        let snapshot = self.snapshot();
        if self.policy.rollback(snapshot).is_ok() {
            // An engine that no longer listens has stopped this scheduler's epoch.
            let _ = self
                .output
                .send(Output::PolicyRolledBack(fault.to_string()));
            return;
        }
        self.fail(Error::Policy(fault));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::OnceLock;

    use scheduling_guest::ids::RequestId as GuestId;

    use super::super::view::tests::waiting;
    use super::*;

    /// The packaged policies, by name: `fcfs`, `priority` and `fixture`, built for
    /// `wasm32-unknown-unknown` from this tree once per test process, into a target directory of
    /// their own (`it-policy`, as `testkit` builds `fcfs`), and each compiled on first use by the
    /// engine `scheduling::Engine` configures.
    pub(crate) fn packaged(name: &str) -> Arc<Module> {
        const NAMES: [&str; 3] = ["fcfs", "priority", "fixture"];
        static BUILT: OnceLock<PathBuf> = OnceLock::new();
        static MODULES: [OnceLock<Arc<Module>>; 3] = [const { OnceLock::new() }; 3];

        let built = BUILT.get_or_init(|| {
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
            for name in NAMES {
                build.arg("-p").arg(format!("scheduling-policy-{name}"));
            }
            let output = build.output().unwrap();
            assert!(
                output.status.success(),
                "building the packaged policies: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            target.join("wasm32-unknown-unknown/release")
        });

        let index = NAMES.iter().position(|packaged| *packaged == name).unwrap();
        let module = MODULES[index].get_or_init(|| {
            let wasm = std::fs::read(built.join(format!("scheduling_policy_{name}.wasm"))).unwrap();
            Arc::new(scheduling::Engine::new().unwrap().load(&wasm).unwrap())
        });
        Arc::clone(module)
    }

    /// A module that speaks the interface's exports, whose `sched_init` runs `body` (a `wat`
    /// expression of type `i64`), whose `sched_limits` and `sched_dispatch` reply `{}` (every limit
    /// at its default; a decision that names nothing) and whose other exports reply nothing.
    pub(crate) fn toy(body: &str) -> Arc<Module> {
        // The `{}` reply lies at 2048: `ptr << 32 | len`.
        let empty = (2048_u64 << 32) | 2;
        let text = format!(
            r#"(module
                (memory (export "memory") 1)
                (data (i32.const 2048) "{{}}")
                (func (export "sched_alloc") (param i32) (result i32) (i32.const 1024))
                (func (export "sched_free") (param i32 i32))
                (func (export "sched_init") (param i32 i32) (result i64) {body})
                (func (export "sched_observe") (param i32 i32) (result i64) (i64.const 0))
                (func (export "sched_dispatch") (param i32 i32) (result i64) (i64.const {empty}))
                (func (export "sched_preempt") (param i32 i32) (result i64) (i64.const 0))
                (func (export "sched_route") (param i32 i32) (result i32) (i32.const 0))
                (func (export "sched_place") (param i32 i32) (result i32) (i32.const 0))
                (func (export "sched_limits") (result i64) (i64.const {empty})))"#
        );
        let engine = scheduling::Engine::new().unwrap();
        Arc::new(engine.load(text.as_bytes()).unwrap())
    }

    /// Arguments that state a deployment's limits, and the ceilings of executors that captured
    /// exactly that much.
    fn limited(tokens: u32, calls: usize) -> (Parameters, Limits) {
        let parameters = Parameters {
            chunked_prefill_size: Some(tokens),
            max_batch: Some(calls),
            max_num_batched_tokens: Some(tokens),
            ..Parameters::default()
        };
        let ceilings = Limits {
            chunked_prefill_size: tokens,
            max_batch: calls,
            max_num_batched_tokens: tokens,
            ..Limits::default()
        };
        (parameters, ceilings)
    }

    fn room() -> Room {
        Room {
            token_budget: 512,
            call_budget: 8,
            page_tokens: 16,
            ..Room::default()
        }
    }

    /// Each packaged policy declares the limits its arguments state, admits what arrived in its
    /// own order, and costs `init` and `limits` once and one `observe` and one `dispatch` per
    /// step.
    #[test]
    fn a_packaged_policy_admits_what_arrived_in_its_order() {
        let (parameters, ceilings) = limited(512, 8);
        // Request 1 arrives first with the less urgent priority (a lower value is more urgent).
        let first = waiting(1, 5, &[1, 2, 3]);
        let second = waiting(2, 1, &[4, 5]);
        let mut requests = HashMap::from([(first.id, first), (second.id, second)]);

        for (name, admitted) in [("fcfs", [1, 2]), ("fixture", [1, 2]), ("priority", [2, 1])] {
            let mut policy = Policy::new(
                packaged(name),
                parameters.clone(),
                Snapshot::default(),
                &ceilings,
            )
            .unwrap();
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
        let refused = Policy::new(
            packaged("fcfs"),
            wide.clone(),
            Snapshot::default(),
            &ceilings,
        );
        assert!(matches!(refused, Err(PolicyError::Limits(_))));

        let mut policy =
            Policy::new(packaged("fcfs"), parameters, Snapshot::default(), &ceilings).unwrap();
        let refused = policy.switch(packaged("priority"), wide, Snapshot::default(), &ceilings);
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
        let mut policy =
            Policy::new(packaged("fcfs"), serving, Snapshot::default(), &ceilings).unwrap();

        policy
            .switch(packaged("priority"), next, Snapshot::default(), &ceilings)
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
        let ceilings = Limits::default();
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
            packaged("fcfs"),
            Parameters::default(),
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
    fn the_ceilings_are_what_the_executors_captured() {
        use ipc::batch::Variant;

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
}
