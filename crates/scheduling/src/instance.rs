//! One running policy: a wasmtime instance the scheduler thread calls.
//!
//! The call protocol is the ABI of `crates/scheduling/guest` (`scheduling_guest
//! ::wire`): the host reserves a buffer in the module's linear memory through
//! `sched_alloc`, writes the message of the call into it as JSON, calls the
//! call's export with the buffer's pointer and length, reads the reply the
//! export returns — as `ptr << 32 | len` of a buffer the host then frees, or,
//! for `route` and `place`, as the chosen index itself — and decodes it.
//!
//! Every call into the module, the two buffer calls included, starts with a
//! fresh fuel budget (§4.4: "每次调用有执行上限（fuel）"). A policy that runs past
//! it, or traps, fails the call: the instance is not used again, and the
//! mechanism instantiates the policy it keeps from the module (§7.3).

use scheduling_guest::decision::{Plan, Room, Shortage, Victim};
use scheduling_guest::event::Event;
use scheduling_guest::ids::PageId;
use scheduling_guest::limits::Limits;
use scheduling_guest::policy::Init;
use scheduling_guest::view::{
    CallView, ExpertLoad, ExpertOrder, RankView, ReplicaView, RequestView,
};
use scheduling_guest::wire::{self, Call};
use wasmtime::{Instance, Memory, Store, Trap, TypedFunc};

use crate::Engine;
use crate::error::PolicyError;

/// Fuel one call may consume unless the mechanism gives another budget.
///
/// §12 leaves the figure to be measured: `observe` and `dispatch` run once per
/// step, and their cost must stay far below a step's decode. Until that
/// measurement exists this is a generous ceiling that still stops a policy that
/// never returns; the measurement sets the budget the deployment runs with.
pub const DEFAULT_CALL_FUEL: u64 = 10_000_000;

/// A policy module instantiated in the scheduler thread.
///
/// One instance belongs to one thread: wasmtime's store is not shared, and
/// §4.4 gives each scheduler thread its own instance, the control thread its
/// own for `route`, and a precopy its own for evaluating the new policy's
/// decisions on the current state.
pub struct PolicyInstance {
    store: Store<()>,
    memory: Memory,
    exports: Exports,
    limits: Limits,
    call_fuel: u64,
    /// Fuel every call into the module has consumed since instantiation, which
    /// is how §12's per-step policy cost is measured: the mechanism takes the
    /// difference around a step's `observe` and `dispatch`.
    fuel_consumed: u64,
    /// The call whose failure ended this instance, if one did.
    failed: Option<Call>,
}

/// The module's exports the call protocol needs.
struct Exports {
    allocate: TypedFunc<u32, u32>,
    release: TypedFunc<(u32, u32), ()>,
    init: TypedFunc<(u32, u32), u64>,
    observe: TypedFunc<(u32, u32), u64>,
    dispatch: TypedFunc<(u32, u32), u64>,
    reclaim: TypedFunc<(u32, u32), u64>,
    route: TypedFunc<(u32, u32), u32>,
    place: TypedFunc<(u32, u32), u32>,
    warm: TypedFunc<(u32, u32), u64>,
    experts: TypedFunc<(u32, u32), u64>,
    limits: TypedFunc<(), u64>,
}

impl Exports {
    /// Binds every export the protocol needs, naming the first one that is
    /// missing or has another signature.
    fn bind(store: &mut Store<()>, instance: &Instance) -> Result<Self, PolicyError> {
        Ok(Self {
            allocate: bind(store, instance, "sched_alloc")?,
            release: bind(store, instance, "sched_free")?,
            init: bind(store, instance, Call::Init.export())?,
            observe: bind(store, instance, Call::Observe.export())?,
            dispatch: bind(store, instance, Call::Dispatch.export())?,
            reclaim: bind(store, instance, Call::Reclaim.export())?,
            route: bind(store, instance, Call::Route.export())?,
            place: bind(store, instance, Call::Place.export())?,
            warm: bind(store, instance, Call::Warm.export())?,
            experts: bind(store, instance, Call::Experts.export())?,
            limits: bind(store, instance, Call::Limits.export())?,
        })
    }

    /// The export of a call that answers with a reply buffer.
    fn replying(&self, call: Call) -> Option<&TypedFunc<(u32, u32), u64>> {
        match call {
            Call::Init => Some(&self.init),
            Call::Observe => Some(&self.observe),
            Call::Dispatch => Some(&self.dispatch),
            Call::Reclaim => Some(&self.reclaim),
            Call::Warm => Some(&self.warm),
            Call::Experts => Some(&self.experts),
            Call::Route | Call::Place | Call::Limits => None,
        }
    }

    /// The export of a call that answers with the chosen index.
    fn indexing(&self, call: Call) -> Option<&TypedFunc<(u32, u32), u32>> {
        match call {
            Call::Route => Some(&self.route),
            Call::Place => Some(&self.place),
            _ => None,
        }
    }
}

/// Binds one export of a given signature, reporting the export by name.
fn bind<Params, Results>(
    store: &mut Store<()>,
    instance: &Instance,
    name: &str,
) -> Result<TypedFunc<Params, Results>, PolicyError>
where
    Params: wasmtime::WasmParams,
    Results: wasmtime::WasmResults,
{
    instance
        .get_typed_func::<Params, Results>(store, name)
        .map_err(|error| PolicyError::Instantiate {
            message: format!(
                "the module does not export `{name}` as the interface needs it: {error}"
            ),
        })
}

impl PolicyInstance {
    /// Instantiates `module`, drives `init` with `init`, reads the policy's
    /// limits once and checks them against the build config's `ceilings`.
    ///
    /// A policy put in place of another starts here: `init` carries the program
    /// arguments and the whole state, so it knows every request, cache page,
    /// encoding and replica that already exists (§4.4). Its own soft state
    /// starts empty, which is what makes a mid-flight swap well defined.
    pub(crate) fn new(
        engine: &Engine,
        module: &wasmtime::Module,
        init: &Init,
        ceilings: &Limits,
    ) -> Result<Self, PolicyError> {
        let mut store = Store::new(engine.inner(), ());
        let instance =
            Instance::new(&mut store, module, &[]).map_err(|error| PolicyError::Instantiate {
                message: format!("the policy module does not instantiate: {error}"),
            })?;
        let memory =
            instance
                .get_memory(&mut store, "memory")
                .ok_or_else(|| PolicyError::Instantiate {
                    message: "the module exports no `memory`".into(),
                })?;
        let exports = Exports::bind(&mut store, &instance)?;
        let mut policy = Self {
            store,
            memory,
            exports,
            limits: Limits::default(),
            call_fuel: DEFAULT_CALL_FUEL,
            fuel_consumed: 0,
            failed: None,
        };
        policy.initialize(init)?;
        policy.limits = policy.read_limits()?;
        crate::limits::check(&policy.limits, ceilings)?;
        Ok(policy)
    }

    /// The fuel one call may consume.
    pub const fn call_fuel(&self) -> u64 {
        self.call_fuel
    }

    /// Sets the fuel one call may consume, from the mechanism's configuration.
    pub const fn set_call_fuel(&mut self, fuel: u64) {
        self.call_fuel = fuel;
    }

    /// Fuel every call into the module has consumed since instantiation.
    ///
    /// Fuel is a deterministic count of the work the module did, so this is
    /// the measurement §12 asks for: the mechanism records it before and after
    /// a step's `observe` and `dispatch` and reports the difference beside the
    /// step's duration.
    pub const fn fuel_consumed(&self) -> u64 {
        self.fuel_consumed
    }

    /// The limits the policy declared at instantiation, already checked against
    /// the build config's ceilings.
    pub const fn limits(&self) -> Limits {
        self.limits
    }

    /// The call whose failure ended this instance, if one did (§7.3: the
    /// mechanism reverts the policy and drops the instance).
    pub const fn failure(&self) -> Option<Call> {
        self.failed
    }

    // -- the decisions of §4.4, in the order a policy's life makes them ------

    /// `init`, called once at instantiation. Answers nothing.
    fn initialize(&mut self, init: &Init) -> Result<(), PolicyError> {
        self.run(Call::Init, |policy| {
            let message = encode(Call::Init, init)?;
            let _ = policy.send(Call::Init, &message)?;
            Ok(())
        })
    }

    /// `observe`: every event since the previous step, in order. Answers
    /// nothing.
    pub fn observe(&mut self, events: &[Event]) -> Result<(), PolicyError> {
        self.run(Call::Observe, |policy| {
            let message = encode(Call::Observe, &events)?;
            let _ = policy.send(Call::Observe, &message)?;
            Ok(())
        })
    }

    /// `dispatch`: this step's plan.
    pub fn dispatch(&mut self, room: &Room) -> Result<Plan, PolicyError> {
        self.run(Call::Dispatch, |policy| {
            let message = encode(Call::Dispatch, room)?;
            let reply = policy.send(Call::Dispatch, &message)?;
            decode(Call::Dispatch, reply)
        })
    }

    /// `reclaim`: which objects to free, in order, for a shortage. An empty
    /// answer frees nothing.
    pub fn reclaim(&mut self, shortage: &Shortage) -> Result<Vec<Victim>, PolicyError> {
        self.run(Call::Reclaim, |policy| {
            let message = encode(Call::Reclaim, shortage)?;
            match policy.send(Call::Reclaim, &message)? {
                Some(reply) => decode(Call::Reclaim, Some(reply)),
                None => Ok(Vec::new()),
            }
        })
    }

    /// `route`: the index in `ranks` of the rank that serves a new request.
    pub fn route(
        &mut self,
        request: &RequestView,
        ranks: &[RankView],
    ) -> Result<usize, PolicyError> {
        self.run(Call::Route, |policy| {
            let message = encode(Call::Route, &(request, ranks))?;
            let index = policy.send_index(Call::Route, &message)?;
            Ok(index as usize)
        })
    }

    /// `place`: the index in `replicas` of the replica a request's first call
    /// on a block runs on. Later calls of the same block use the same replica,
    /// which the mechanism remembers.
    pub fn place(
        &mut self,
        call: &CallView,
        replicas: &[ReplicaView],
    ) -> Result<usize, PolicyError> {
        self.run(Call::Place, |policy| {
            let message = encode(Call::Place, &(call, replicas))?;
            let index = policy.send_index(Call::Place, &message)?;
            Ok(index as usize)
        })
    }

    /// `warm`: which pages to warm during an epoch's precopy, in order.
    pub fn warm(&mut self, room: &Room) -> Result<Vec<PageId>, PolicyError> {
        self.run(Call::Warm, |policy| {
            let message = encode(Call::Warm, room)?;
            match policy.send(Call::Warm, &message)? {
                Some(reply) => decode(Call::Warm, Some(reply)),
                None => Ok(Vec::new()),
            }
        })
    }

    /// `experts`: the placement of experts on ranks.
    pub fn experts(&mut self, load: &ExpertLoad) -> Result<ExpertOrder, PolicyError> {
        self.run(Call::Experts, |policy| {
            let message = encode(Call::Experts, load)?;
            let reply = policy.send(Call::Experts, &message)?;
            decode(Call::Experts, reply)
        })
    }

    /// `limits`, read once here at instantiation and not called again.
    fn read_limits(&mut self) -> Result<Limits, PolicyError> {
        self.run(Call::Limits, |policy| {
            policy.charge(Call::Limits)?;
            let packed = policy
                .exports
                .limits
                .call(&mut policy.store, ())
                .map_err(|error| classify(Call::Limits, error))?;
            policy.spent();
            let reply = policy.take_reply(Call::Limits, packed)?;
            decode(Call::Limits, reply)
        })
    }

    // -- the call protocol ---------------------------------------------------

    /// Runs one call's body, ending the instance if the call fails.
    fn run<T>(
        &mut self,
        call: Call,
        body: impl FnOnce(&mut Self) -> Result<T, PolicyError>,
    ) -> Result<T, PolicyError> {
        if let Some(failed) = self.failed {
            return Err(PolicyError::Poisoned { call: failed });
        }
        match body(self) {
            Ok(value) => Ok(value),
            Err(error) => {
                self.failed = Some(call);
                Err(error)
            }
        }
    }

    /// Sends a message to a call that answers with a reply buffer, and returns
    /// the reply's bytes (`None` for the empty reply).
    fn send(&mut self, call: Call, message: &[u8]) -> Result<Option<Vec<u8>>, PolicyError> {
        let len = u32::try_from(message.len())
            .map_err(|_| malformed(call, "the message is larger than wasm32 can address"))?;
        let ptr = self.allocate(call, message.len())?;
        self.write(call, ptr, message)?;
        self.charge(call)?;
        let packed = {
            let export = self
                .exports
                .replying(call)
                .ok_or_else(|| malformed(call, "this call does not answer with a buffer"))?;
            let reply = export
                .call(&mut self.store, (ptr, len))
                .map_err(|error| classify(call, error))?;
            self.spent();
            reply
        };
        self.release(call, ptr, len)?;
        self.take_reply(call, packed)
    }

    /// Sends a message to a call that answers with the chosen index.
    fn send_index(&mut self, call: Call, message: &[u8]) -> Result<u32, PolicyError> {
        let len = u32::try_from(message.len())
            .map_err(|_| malformed(call, "the message is larger than wasm32 can address"))?;
        let ptr = self.allocate(call, message.len())?;
        self.write(call, ptr, message)?;
        self.charge(call)?;
        let index = {
            let export = self
                .exports
                .indexing(call)
                .ok_or_else(|| malformed(call, "this call does not answer with an index"))?;
            let index = export
                .call(&mut self.store, (ptr, len))
                .map_err(|error| classify(call, error))?;
            self.spent();
            index
        };
        self.release(call, ptr, len)?;
        Ok(index)
    }

    /// Reserves `len` bytes of the module's memory for the message.
    fn allocate(&mut self, call: Call, message_len: usize) -> Result<u32, PolicyError> {
        let len = u32::try_from(message_len)
            .map_err(|_| malformed(call, "the message is larger than wasm32 can address"))?;
        self.charge(call)?;
        let ptr = self
            .exports
            .allocate
            .call(&mut self.store, len)
            .map_err(|error| classify(call, error))?;
        self.spent();
        if ptr == 0 && len != 0 {
            return Err(malformed(call, "the module could not reserve the message"));
        }
        Ok(ptr)
    }

    /// Writes the message into the buffer `allocate` reserved.
    fn write(&mut self, call: Call, ptr: u32, message: &[u8]) -> Result<(), PolicyError> {
        if message.is_empty() {
            return Ok(());
        }
        self.memory
            .write(&mut self.store, ptr as usize, message)
            .map_err(|error| malformed(call, format!("cannot write the message: {error}")))
    }

    /// Releases a buffer of the module's memory.
    fn release(&mut self, call: Call, ptr: u32, len: u32) -> Result<(), PolicyError> {
        self.charge(call)?;
        let released = self
            .exports
            .release
            .call(&mut self.store, (ptr, len))
            .map_err(|error| classify(call, error));
        self.spent();
        released
    }

    /// Copies the reply an export returned out of the module's memory and
    /// releases it.
    fn take_reply(&mut self, call: Call, packed: u64) -> Result<Option<Vec<u8>>, PolicyError> {
        let Some((ptr, len)) = wire::unpack(packed) else {
            return Ok(None);
        };
        let reply = self.read(call, ptr, len)?;
        self.release(call, ptr, len)?;
        Ok(Some(reply))
    }

    /// Copies `len` bytes at `ptr` out of the module's memory.
    fn read(&self, call: Call, ptr: u32, len: u32) -> Result<Vec<u8>, PolicyError> {
        let start = ptr as usize;
        let len = len as usize;
        let size = self.memory.data_size(&self.store);
        let end = start
            .checked_add(len)
            .ok_or_else(|| malformed(call, "the reply's buffer wraps the address space"))?;
        if end > size {
            return Err(malformed(
                call,
                format!("the reply's buffer {start}..{end} lies outside the module's {size} bytes"),
            ));
        }
        let mut reply = vec![0u8; len];
        self.memory
            .read(&self.store, start, &mut reply)
            .map_err(|error| malformed(call, format!("cannot read the reply: {error}")))?;
        Ok(reply)
    }

    /// Records the fuel the call that just returned consumed.
    fn spent(&mut self) {
        let remaining = self.store.get_fuel().unwrap_or(0);
        self.fuel_consumed = self
            .fuel_consumed
            .saturating_add(self.call_fuel.saturating_sub(remaining));
    }

    /// Gives the next call into the module its whole fuel budget.
    fn charge(&mut self, call: Call) -> Result<(), PolicyError> {
        self.store
            .set_fuel(self.call_fuel)
            .map_err(|error| malformed(call, format!("cannot set the call's fuel: {error}")))
    }
}

/// Encodes a message for the module's memory.
fn encode<T: serde::Serialize>(call: Call, message: &T) -> Result<Vec<u8>, PolicyError> {
    wire::encode(message).map_err(|error| PolicyError::Malformed {
        call,
        message: format!("cannot encode the message: {error}"),
    })
}

/// Decodes a reply of the call's type; `None` is the empty reply.
///
/// A decision whose answer carries something (`dispatch`, `experts`, `limits`)
/// must answer; `reclaim` and `warm` may answer nothing, which frees nothing.
fn decode<T: serde::de::DeserializeOwned>(
    call: Call,
    reply: Option<Vec<u8>>,
) -> Result<T, PolicyError> {
    let Some(reply) = reply else {
        return Err(malformed(call, "the policy answered nothing"));
    };
    wire::decode(&reply)
        .map_err(|error| malformed(call, format!("the answer does not decode: {error}")))
}

/// A protocol failure of the call's own making.
fn malformed(call: Call, message: impl Into<String>) -> PolicyError {
    PolicyError::Malformed {
        call,
        message: message.into(),
    }
}

/// Reads a failed call as the policy failure it is: running past the fuel
/// budget is its own failure (§4.4), and every other trap is the policy's.
fn classify(call: Call, error: wasmtime::Error) -> PolicyError {
    match error.downcast_ref::<Trap>() {
        Some(&Trap::OutOfFuel) => PolicyError::FuelExhausted { call },
        Some(&trap) => PolicyError::Trap { call, trap },
        None => malformed(call, format!("the call failed outside the module: {error}")),
    }
}
