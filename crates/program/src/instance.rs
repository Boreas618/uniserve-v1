//! A program's instance: its store, its exports, the fuel of each call, and how a call ends.

use std::fmt;

use scheduling::program::{EXPORTS, name};
use scheduling::wire::{decode, encode, unpack};
use serde::Serialize;
use serde::de::DeserializeOwned;
use wasmtime::{
    ExternType, Memory, ResourceLimiter, Store, Trap, TypedFunc, WasmParams, WasmResults,
};

use crate::{Host, Module};

/// What each call into a program runs within.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// The fuel of one call: roughly the wasm instructions it may execute.
    pub fuel: u64,
    /// The fuel a call is given beyond `fuel` for every byte of a message it decodes (its input,
    /// and the replies of the host functions it calls), so that decoding a large input does not
    /// exhaust the budget of the decision.
    pub per_byte: u64,
    /// The bytes the program's linear memory may grow to.
    pub memory: usize,
}

impl Budget {
    /// What a control program's call runs within on the leader daemon: a second's worth of
    /// instructions, ample for a decision over a few series, with the fuel of decoding the
    /// configuration on top of it; and the memory of a program that keeps moving averages over
    /// its own history.
    pub const CONTROL: Budget = Budget {
        fuel: 1_000_000_000,
        per_byte: 200,
        memory: 256 << 20,
    };

    /// What a routing program's call runs within on the leader daemon: it is on every request's
    /// path, so its budget is a small fraction of a request's admission, beside decoding the
    /// replicas' states.
    pub const ROUTING: Budget = Budget {
        fuel: 20_000_000,
        per_byte: 200,
        memory: 64 << 20,
    };
}

/// One call into a program, by the export it enters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Call {
    New,
    Step,
    Save,
    Restore,
}

impl Call {
    /// The export the call enters.
    pub fn name(self) -> &'static str {
        match self {
            Call::New => name::NEW,
            Call::Step => name::STEP,
            Call::Save => name::SAVE,
            Call::Restore => name::RESTORE,
        }
    }
}

/// Why an instance ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The program panicked, with the panic's text.
    Panic(String),
    /// The call ran past its fuel.
    Fuel,
    /// The program's memory would have grown past its budget, and the program trapped.
    Memory,
    /// The program trapped otherwise: an access outside its memory, an unreachable instruction.
    Trap(Trap),
    /// The program broke the call convention: a buffer outside its memory, a reply missing or
    /// not of the call's type, a message to a host function that is not one.
    Malformed(String),
    /// The embedder disabled the program ([`Instance::disable`]).
    Disabled,
}

impl fmt::Display for Reason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reason::Panic(text) => write!(formatter, "it panicked: {text}"),
            Reason::Fuel => write!(formatter, "it ran past its fuel"),
            Reason::Memory => write!(formatter, "it ran past its memory"),
            Reason::Trap(trap) => write!(formatter, "it trapped: {trap}"),
            Reason::Malformed(text) => write!(formatter, "it broke the call convention: {text}"),
            Reason::Disabled => write!(formatter, "it was disabled"),
        }
    }
}

/// What a program did or failed to do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The module was not loaded or not instantiated: no engine on this host, bytes that are not
    /// a module, imports its host does not offer, an export of the interface missing or of
    /// another signature, an initial memory above the budget.
    Load(String),
    /// The package's `args` table is not the program's `Args`, with why.
    Args(String),
    /// The saved state is not the program's current type, with why; the embedder creates the
    /// program anew ([`Instance::new`]).
    State(String),
    /// The host's message for the call does not encode as JSON (a map whose keys are not
    /// strings); the instance is untouched.
    Message(String),
    /// The instance ended in `call`, or had ended in it before.
    Ended { call: Call, reason: Reason },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Load(text) => write!(formatter, "the program was not loaded: {text}"),
            Error::Args(text) => write!(formatter, "the program's args do not decode: {text}"),
            Error::State(text) => write!(formatter, "the saved state does not decode: {text}"),
            Error::Message(text) => write!(formatter, "the message does not encode: {text}"),
            Error::Ended { call, reason } => {
                write!(
                    formatter,
                    "the program ended in `{}`: {reason}",
                    call.name()
                )
            }
        }
    }
}

impl std::error::Error for Error {}

/// The calls made into an instance and the fuel they consumed, the buffer calls included.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub calls: u64,
    pub fuel: u64,
}

/// A host function's refusal of what the module handed it, which ends the instance as
/// [`Reason::Malformed`].
#[derive(Debug)]
pub(crate) struct Fault(String);

impl Fault {
    pub(crate) fn error(text: impl Into<String>) -> wasmtime::Error {
        wasmtime::Error::new(Fault(text.into()))
    }
}

impl fmt::Display for Fault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Fault {}

/// The data of an instance's store: the embedder, and what the host functions and the memory
/// limit record for the call in progress.
pub(crate) struct Context<H> {
    pub(crate) host: H,
    /// The text of the panic the module handed over before it traps.
    pub(crate) panic: Option<String>,
    /// The budget's fuel per byte, which a host function's reply adds to the call.
    pub(crate) per_byte: u64,
    /// The bytes the memory may grow to.
    memory: usize,
    /// Whether a growth past `memory` was refused, which explains the trap that follows.
    exceeded: bool,
}

/// The elements a module's function table may grow to: a Rust module's table holds its
/// indirectly called functions and does not grow.
const TABLE_ELEMENTS: usize = 1 << 16;

impl<H: Send> ResourceLimiter for Context<H> {
    fn memory_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> wasmtime::Result<bool> {
        if desired > self.memory {
            self.exceeded = true;
            return Ok(false);
        }
        Ok(true)
    }

    fn table_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> wasmtime::Result<bool> {
        Ok(desired <= TABLE_ELEMENTS)
    }
}

/// Checks that `module` has the memory and every export of the program interface; their
/// signatures are checked when an instance binds them.
pub(crate) fn check(module: &wasmtime::Module) -> Result<(), crate::Error> {
    if !matches!(module.get_export(name::MEMORY), Some(ExternType::Memory(_))) {
        return Err(Error::Load("the module exports no memory".into()));
    }
    for export in EXPORTS {
        if !matches!(module.get_export(export), Some(ExternType::Func(_))) {
            return Err(Error::Load(format!(
                "the module exports no function `{export}`"
            )));
        }
    }
    Ok(())
}

/// A program's instance: one program, its own linear memory, called one call at a time.
///
/// Once a call ends it ([`Error::Ended`]), every later call returns the same reason.
pub struct Instance<H: 'static> {
    store: Store<Context<H>>,
    memory: Memory,
    alloc: TypedFunc<u32, u32>,
    free: TypedFunc<(u32, u32), ()>,
    new: TypedFunc<(u32, u32), u64>,
    step: TypedFunc<(u32, u32), u64>,
    save: TypedFunc<(), u64>,
    restore: TypedFunc<(u32, u32), u64>,
    budget: Budget,
    ended: Option<(Call, Reason)>,
    usage: Usage,
}

impl<H: Host> Instance<H> {
    /// Instantiates `module` with `host` answering its host functions and creates the program
    /// from the package's `args` table (the `new` export).
    ///
    /// # Errors
    ///
    /// [`Error::Load`] when the module does not instantiate, [`Error::Args`] when `args` is not
    /// the program's `Args`, [`Error::Ended`] when `new` fails.
    pub fn new(
        module: &Module<H>,
        host: H,
        budget: &Budget,
        args: &impl Serialize,
    ) -> Result<Self, Error> {
        let mut instance = Self::start(module, host, budget)?;

        let args = encode(args).map_err(|error| Error::Message(error.to_string()))?;
        match instance.exchange(Call::New, &args)? {
            None => Ok(instance),
            Some(text) => Err(Error::Args(String::from_utf8_lossy(&text).into_owned())),
        }
    }

    /// Instantiates `module` with `host` answering its host functions and restores the program
    /// from `state`, which [`Instance::save`] returned, perhaps for an earlier version of the
    /// program (the `restore` export).
    ///
    /// # Errors
    ///
    /// [`Error::State`] when `state` is not the program's current type, so the embedder creates
    /// it anew; otherwise as [`Instance::new`].
    pub fn restore(
        module: &Module<H>,
        host: H,
        budget: &Budget,
        state: &[u8],
    ) -> Result<Self, Error> {
        let mut instance = Self::start(module, host, budget)?;

        match instance.exchange(Call::Restore, state)? {
            None => Ok(instance),
            Some(text) => Err(Error::State(String::from_utf8_lossy(&text).into_owned())),
        }
    }

    /// One step: `input` in, the program's output out. `I` encodes as the program's `Input`
    /// and `O` is its `Output`, as the SDK's types for the program's kind give them.
    ///
    /// # Errors
    ///
    /// [`Error::Ended`] when the step fails, its reply missing or not an `O` included;
    /// [`Error::Message`] when `input` does not encode.
    pub fn step<I: Serialize, O: DeserializeOwned>(&mut self, input: &I) -> Result<O, Error> {
        let input = encode(input).map_err(|error| Error::Message(error.to_string()))?;
        let reply = self.exchange(Call::Step, &input)?;
        self.decode(Call::Step, reply)
    }

    /// The program's state, which [`Instance::restore`] takes back.
    ///
    /// # Errors
    ///
    /// [`Error::Ended`] when `save` fails.
    pub fn save(&mut self) -> Result<Vec<u8>, Error> {
        self.alive()?;

        let save = self.save.clone();
        let packed = self.enter(Call::Save, save, (), self.budget.fuel)?;
        self.take(Call::Save, packed)?
            .ok_or_else(|| self.end(Call::Save, malformed("it replied no state")))
    }

    /// Ends the instance: every later call returns [`Reason::Disabled`].
    pub fn disable(&mut self) {
        if self.ended.is_none() {
            self.ended = Some((Call::Step, Reason::Disabled));
        }
    }

    /// The call that ended the instance and why; `None` while it runs.
    pub fn ended(&self) -> Option<&(Call, Reason)> {
        self.ended.as_ref()
    }

    /// The calls made into the instance and the fuel they consumed.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// The embedder.
    pub fn host(&self) -> &H {
        &self.store.data().host
    }

    /// The embedder, to change between calls.
    pub fn host_mut(&mut self) -> &mut H {
        &mut self.store.data_mut().host
    }

    /// Instantiates `module` in a store of its own and binds the exports.
    fn start(module: &Module<H>, host: H, budget: &Budget) -> Result<Self, Error> {
        let context = Context {
            host,
            panic: None,
            per_byte: budget.per_byte,
            memory: budget.memory,
            exceeded: false,
        };
        let mut store = Store::new(module.pre.module().engine(), context);
        store.limiter(|context| context as &mut dyn ResourceLimiter);
        store
            .set_fuel(budget.fuel)
            .map_err(|error| Error::Load(format!("the engine meters no fuel: {error}")))?;

        let instance = module
            .pre
            .instantiate(&mut store)
            .map_err(|error| Error::Load(format!("the module does not instantiate: {error}")))?;
        let memory = instance
            .get_memory(&mut store, name::MEMORY)
            .ok_or_else(|| Error::Load("the module exports no memory".into()))?;

        Ok(Self {
            alloc: bind(&mut store, &instance, name::ALLOC)?,
            free: bind(&mut store, &instance, name::FREE)?,
            new: bind(&mut store, &instance, name::NEW)?,
            step: bind(&mut store, &instance, name::STEP)?,
            save: bind(&mut store, &instance, name::SAVE)?,
            restore: bind(&mut store, &instance, name::RESTORE)?,
            store,
            memory,
            budget: *budget,
            ended: None,
            usage: Usage::default(),
        })
    }

    /// Fails with the call that ended the instance and why, if one did.
    fn alive(&self) -> Result<(), Error> {
        match &self.ended {
            Some((call, reason)) => Err(Error::Ended {
                call: *call,
                reason: reason.clone(),
            }),
            None => Ok(()),
        }
    }

    /// Writes `message` into a buffer of the module's memory, enters the export of `call` with
    /// it, frees it, and returns the reply's bytes (`None` for the empty reply).
    fn exchange(&mut self, call: Call, message: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        self.alive()?;

        let len = u32::try_from(message.len())
            .map_err(|_| Error::Message("the message is larger than wasm32 addresses".into()))?;
        // A call decodes its message, so its budget, and the budget of the allocation that holds
        // the message, grow with it.
        let fuel = self
            .budget
            .fuel
            .saturating_add(self.budget.per_byte.saturating_mul(u64::from(len)));

        let alloc = self.alloc.clone();
        let ptr = self.enter(call, alloc, len, fuel)?;
        if ptr == 0 && len != 0 {
            return Err(self.end(call, malformed("it could not hold the message")));
        }
        if self
            .memory
            .write(&mut self.store, ptr as usize, message)
            .is_err()
        {
            return Err(self.end(
                call,
                malformed("its message buffer lies outside its memory"),
            ));
        }

        let export = match call {
            Call::New => self.new.clone(),
            Call::Step => self.step.clone(),
            Call::Restore => self.restore.clone(),
            Call::Save => return Err(Error::Message("`save` takes no message".into())),
        };
        let packed = self.enter(call, export, (ptr, len), fuel)?;

        let free = self.free.clone();
        self.enter(call, free, (ptr, len), self.budget.fuel)?;
        self.take(call, packed)
    }

    /// Copies the packed reply out of the module's memory and frees it.
    fn take(&mut self, call: Call, packed: u64) -> Result<Option<Vec<u8>>, Error> {
        let Some((ptr, len)) = unpack(packed) else {
            return Ok(None);
        };

        let mut reply = vec![0; len as usize];
        if self
            .memory
            .read(&self.store, ptr as usize, &mut reply)
            .is_err()
        {
            let text = format!("its reply {ptr}+{len} lies outside its memory");
            return Err(self.end(call, malformed(text)));
        }
        let free = self.free.clone();
        self.enter(call, free, (ptr, len), self.budget.fuel)?;
        Ok(Some(reply))
    }

    /// The reply of `call` decoded as an `O`, ending the instance when it is missing or not one.
    fn decode<O: DeserializeOwned>(
        &mut self,
        call: Call,
        reply: Option<Vec<u8>>,
    ) -> Result<O, Error> {
        let Some(reply) = reply else {
            return Err(self.end(call, malformed("it replied nothing")));
        };
        decode(&reply).map_err(|error| {
            self.end(
                call,
                malformed(format!("its reply does not decode: {error}")),
            )
        })
    }

    /// Enters one export with a fresh fuel budget of `fuel` and adds what it consumed to the
    /// usage; a failure ends the instance with its reason.
    fn enter<P: WasmParams, R: WasmResults>(
        &mut self,
        call: Call,
        export: TypedFunc<P, R>,
        params: P,
        fuel: u64,
    ) -> Result<R, Error> {
        if let Err(error) = self.store.set_fuel(fuel) {
            return Err(self.end(call, malformed(format!("cannot set its fuel: {error}"))));
        }
        self.store.data_mut().panic = None;
        self.store.data_mut().exceeded = false;

        let result = export.call(&mut self.store, params);

        // The host functions add the fuel of their replies, so the budget may have grown.
        let left = self.store.get_fuel().unwrap_or(0);
        self.usage.calls += 1;
        self.usage.fuel = self.usage.fuel.saturating_add(fuel.saturating_sub(left));

        result.map_err(|error| {
            let reason = self.reason(&error);
            self.end(call, reason)
        })
    }

    /// The reason a call that failed with `error` ends the instance.
    fn reason(&mut self, error: &wasmtime::Error) -> Reason {
        let context = self.store.data_mut();
        if let Some(fault) = error.downcast_ref::<Fault>() {
            return Reason::Malformed(fault.0.clone());
        }
        match error.downcast_ref::<Trap>() {
            Some(Trap::OutOfFuel) => Reason::Fuel,
            Some(&trap) => match context.panic.take() {
                Some(text) => Reason::Panic(text),
                None if context.exceeded => Reason::Memory,
                None => Reason::Trap(trap),
            },
            None => Reason::Malformed(format!("the call failed outside the module: {error}")),
        }
    }

    /// Ends the instance in `call` for `reason`, and the error that reports it.
    fn end(&mut self, call: Call, reason: Reason) -> Error {
        self.ended = Some((call, reason.clone()));
        Error::Ended { call, reason }
    }
}

fn malformed(text: impl Into<String>) -> Reason {
    Reason::Malformed(text.into())
}

/// The export `name` with the signature the interface gives it.
fn bind<H, P: WasmParams, R: WasmResults>(
    store: &mut Store<Context<H>>,
    instance: &wasmtime::Instance,
    name: &str,
) -> Result<TypedFunc<P, R>, Error> {
    instance
        .get_typed_func(&mut *store, name)
        .map_err(|error| Error::Load(format!("`{name}` is not the interface's: {error}")))
}
