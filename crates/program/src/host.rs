//! The host functions: what the embedder answers ([`Host`], [`Control`]) and how a module's
//! calls of `uniserve.*` reach it.

use scheduling::control::{Config, Filter, Query, Record, Report};
use scheduling::program::{IMPORTS, Level, PANIC, name};
use scheduling::wire::{decode, encode, pack};
use wasmtime::{Caller, Extern, Linker};

use crate::Error;
use crate::instance::{Context, Fault};

/// The embedder of a program: what answers `uniserve.log`, which every program may call.
///
/// The host calls it on the thread that calls the program, during the call.
pub trait Host: Send + 'static {
    /// A record the program wrote with the `log` crate; the embedder writes it to its `events`
    /// table as a `log` record of the program.
    fn log(&mut self, level: Level, text: &str);
}

/// The embedder of a control program: what answers the host functions behind its
/// `State` (`uniserve.stats`, `uniserve.events`, `uniserve.validate`).
///
/// Each is called only when the program reads the view or validates, during its step: a step
/// that reads no series computes none.
pub trait Control: Host {
    /// The value of one aggregate of the `stats` table; NaN when the query selects no value.
    /// The query's window ends at the step's instant.
    fn stats(&mut self, query: &Query) -> f64;

    /// The records of the `events` table the filter selects, oldest first. The filter's window
    /// ends at the step's instant.
    fn events(&mut self, filter: &Filter) -> Vec<Record>;

    /// What applying `config` would do, without applying it; or why `apply` would reject it.
    fn validate(&mut self, config: &Config) -> Result<Report, String>;
}

/// Links `uniserve.log`.
pub(crate) fn log<H: Host>(linker: &mut Linker<Context<H>>) -> Result<(), Error> {
    linker
        .func_wrap(
            IMPORTS,
            name::LOG,
            |mut caller: Caller<'_, Context<H>>, level: u32, ptr: u32, len: u32| {
                let text = read(&mut caller, ptr, len)?;
                let text = String::from_utf8_lossy(&text);

                // A panic's text is the reason the trap that follows ends the instance.
                if level == PANIC {
                    caller.data_mut().panic = Some(text.into_owned());
                    return Ok(());
                }

                let level = Level::from_wire(level)
                    .ok_or_else(|| Fault::error(format!("{level} is not a log level")))?;
                caller.data_mut().host.log(level, &text);
                Ok(())
            },
        )
        .map_err(link)?;
    Ok(())
}

/// Links `uniserve.stats`, `uniserve.events` and `uniserve.validate`.
pub(crate) fn control<H: Control>(linker: &mut Linker<Context<H>>) -> Result<(), Error> {
    linker
        .func_wrap(
            IMPORTS,
            name::STATS,
            |mut caller: Caller<'_, Context<H>>, ptr: u32, len: u32| {
                let query: Query = message(&mut caller, ptr, len, "stats query")?;
                Ok(caller.data_mut().host.stats(&query))
            },
        )
        .map_err(link)?;

    linker
        .func_wrap(
            IMPORTS,
            name::EVENTS,
            |mut caller: Caller<'_, Context<H>>, ptr: u32, len: u32| {
                let filter: Filter = message(&mut caller, ptr, len, "events filter")?;
                let records = caller.data_mut().host.events(&filter);
                reply(&mut caller, &records)
            },
        )
        .map_err(link)?;

    linker
        .func_wrap(
            IMPORTS,
            name::VALIDATE,
            |mut caller: Caller<'_, Context<H>>, ptr: u32, len: u32| {
                let config: Config = message(&mut caller, ptr, len, "configuration")?;
                let report = caller.data_mut().host.validate(&config);
                reply(&mut caller, &report)
            },
        )
        .map_err(link)?;
    Ok(())
}

fn link(error: wasmtime::Error) -> Error {
    Error::Load(format!("cannot link the host functions: {error}"))
}

/// The module's memory, a fault when it exports none.
fn memory<H>(caller: &mut Caller<'_, Context<H>>) -> wasmtime::Result<wasmtime::Memory> {
    caller
        .get_export(name::MEMORY)
        .and_then(Extern::into_memory)
        .ok_or_else(|| Fault::error("the module exports no memory"))
}

/// The `len` bytes at `ptr` of the module's memory, a fault when they lie outside it.
fn read<H>(caller: &mut Caller<'_, Context<H>>, ptr: u32, len: u32) -> wasmtime::Result<Vec<u8>> {
    let memory = memory(caller)?;
    let mut bytes = vec![0; len as usize];
    memory
        .read(&*caller, ptr as usize, &mut bytes)
        .map_err(|_| Fault::error(format!("the message {ptr}+{len} lies outside the memory")))?;
    Ok(bytes)
}

/// The message at `ptr`, decoded; a fault when it is not a `T`.
fn message<H, T: serde::de::DeserializeOwned>(
    caller: &mut Caller<'_, Context<H>>,
    ptr: u32,
    len: u32,
    what: &str,
) -> wasmtime::Result<T> {
    let bytes = read(caller, ptr, len)?;
    decode(&bytes).map_err(|error| Fault::error(format!("the {what} does not decode: {error}")))
}

/// Writes `value` into a buffer of the module's own `alloc` export and packs it, giving the call
/// the fuel the module spends decoding it: the budget's fuel per byte of a message.
fn reply<H>(
    caller: &mut Caller<'_, Context<H>>,
    value: &impl serde::Serialize,
) -> wasmtime::Result<u64> {
    let bytes = encode(value).map_err(|error| Fault::error(format!("cannot encode: {error}")))?;
    let len = u32::try_from(bytes.len())
        .map_err(|_| Fault::error("the reply is larger than wasm32 addresses"))?;

    let fuel = caller.get_fuel()?;
    let extra = caller.data().per_byte.saturating_mul(u64::from(len));
    caller.set_fuel(fuel.saturating_add(extra))?;

    let alloc = caller
        .get_export(name::ALLOC)
        .and_then(Extern::into_func)
        .ok_or_else(|| Fault::error("the module exports no `alloc`"))?
        .typed::<u32, u32>(&*caller)
        .map_err(|error| Fault::error(format!("`alloc` is not the interface's: {error}")))?;
    let ptr = alloc.call(&mut *caller, len)?;

    let memory = memory(caller)?;
    memory
        .write(&mut *caller, ptr as usize, &bytes)
        .map_err(|_| Fault::error(format!("the buffer {ptr}+{len} lies outside the memory")))?;
    Ok(pack(ptr, len))
}
