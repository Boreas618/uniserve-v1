//! The program host: runs a policy package's programs, each a wasm32 module built from a crate
//! that implements `scheduling::program::Program`.
//!
//! The leader daemon hosts the control and routing programs, every Storage Server the KV
//! program, and every scheduler the scheduling policy; this library is the one mechanism they
//! share. The interface is the `scheduling` crate's: the exports a module offers, the host
//! functions it imports, and the messages, JSON of the SDK's types in buffers of the module's
//! linear memory (`scheduling::program`).
//!
//! # Use
//!
//! ```ignore
//! let module = Module::<Leader>::control(&bytes)?;          // compiled once, instantiated often
//! let mut chunk = match Instance::restore(&module, leader, &budget, &saved) {
//!     Err(Error::State(_)) => Instance::new(&module, leader, &budget, &args)?,
//!     instance => instance?,
//! };
//! let config: Option<Config> = chunk.step(&State::new("chunk", now, config))?;
//! let saved = chunk.save()?;
//! ```
//!
//! - One wasmtime engine per process ([`engine`]) compiles every module; a [`Module`] is linked
//!   for its embedder's [`Host`] once and instantiated any number of times.
//! - Every program is an [`Instance`] of its own, with its own store and linear memory and no
//!   WASI: it reaches nothing but its host functions. Its calls run one at a time (`&mut self`).
//! - Every call is given a fuel budget ([`Budget`]), the instructions it may execute, and its
//!   memory is bounded.
//! - The host functions are answered by the embedder: `uniserve.log` by [`Host`], which every
//!   program may call, and `uniserve.stats`, `uniserve.group`, `uniserve.events` and
//!   `uniserve.validate` by [`Control`], which only a control program's module is linked with.
//! - A program that panics, runs past its fuel, traps or breaks the protocol ends its instance
//!   ([`Error::Ended`] with its [`Reason`]); the embedder records it. The host never panics on
//!   a program's behaviour.
//! - A program's state is saved ([`Instance::save`]) and restored ([`Instance::restore`]); a
//!   state that does not decode as the program's current type is [`Error::State`], and the
//!   embedder creates the program anew.

mod host;
mod instance;

use std::sync::OnceLock;

pub use host::{Control, Host};
pub use instance::{Budget, Call, Error, Instance, Reason, Usage};
pub use scheduling::control::{
    Config, Filter, Function, Group, Query, Record, Report, Room, State,
};
pub use scheduling::program::{Kind, Level};

/// The process's one wasmtime engine, which compiles and runs every program module.
///
/// Fuel metering is on (every call into a program has a budget), the compiler optimizes for
/// speed, and backtraces are off: a failure is reported by its reason, with a panic's own text.
///
/// # Errors
///
/// [`Error::Load`] when wasmtime has no engine for this host; the error is the same for every
/// later call.
pub fn engine() -> Result<&'static wasmtime::Engine, Error> {
    static ENGINE: OnceLock<Result<wasmtime::Engine, String>> = OnceLock::new();

    let engine = ENGINE.get_or_init(|| {
        let mut config = wasmtime::Config::new();
        config.consume_fuel(true);
        config.cranelift_opt_level(wasmtime::OptLevel::Speed);
        config.wasm_backtrace_max_frames(None);

        wasmtime::Engine::new(&config)
            .map_err(|error| format!("this host has no engine for program modules: {error}"))
    });
    engine.as_ref().map_err(|error| Error::Load(error.clone()))
}

/// The kind of program the module `bytes` is and the version of the interface it was compiled
/// against, as its section states them (`scheduling::program::SECTION`): what decides which host
/// runs it and with which host functions.
///
/// # Errors
///
/// [`Error::Load`] when the bytes are no module, or it states no kind or states it twice.
pub fn kind(bytes: &[u8]) -> Result<(Kind, u32), Error> {
    let mut stated = None;
    for payload in wasmparser::Parser::new(0).parse_all(bytes) {
        let payload =
            payload.map_err(|error| Error::Load(format!("the bytes are not a module: {error}")))?;
        let wasmparser::Payload::CustomSection(section) = payload else {
            continue;
        };
        if section.name() != scheduling::program::SECTION {
            continue;
        }
        if stated.is_some() {
            return Err(Error::Load(format!(
                "the module states its kind twice (`{}`)",
                scheduling::program::SECTION
            )));
        }
        stated = Some(scheduling::program::stated(section.data()).ok_or_else(|| {
            Error::Load(format!(
                "the module's `{}` section states no kind of program",
                scheduling::program::SECTION
            ))
        })?);
    }
    stated.ok_or_else(|| {
        Error::Load(format!(
            "the module states no kind (`{}`): it was not built with the program SDK's export \
             macros",
            scheduling::program::SECTION
        ))
    })
}

/// A program module, compiled by [`engine`] and linked with the host functions its embedder
/// `H` answers; instantiated by [`Instance::new`] and [`Instance::restore`].
///
/// Cloning shares the compiled code.
pub struct Module<H: 'static> {
    pre: wasmtime::InstancePre<instance::Context<H>>,
    kind: Kind,
}

impl<H: 'static> Clone for Module<H> {
    fn clone(&self) -> Self {
        Self {
            pre: self.pre.clone(),
            kind: self.kind,
        }
    }
}

impl<H: 'static> Module<H> {
    /// The kind of program the module states it is.
    pub fn kind(&self) -> Kind {
        self.kind
    }
}

impl<H: Host> Module<H> {
    /// Compiles `bytes`, a wasm32 module of a routing or KV program: linked with `uniserve.log`
    /// only.
    ///
    /// # Errors
    ///
    /// [`Error::Load`] when the bytes are not a module, it states another kind ([`kind`]) or
    /// another version of the program interface, it imports anything else, or it lacks the
    /// memory or an export of the program interface, or has one of another signature.
    pub fn new(bytes: &[u8]) -> Result<Self, Error> {
        let mut linker = wasmtime::Linker::new(engine()?);
        host::log(&mut linker)?;
        Self::link(bytes, &linker, &[Kind::Routing, Kind::Kv])
    }

    fn link(
        bytes: &[u8],
        linker: &wasmtime::Linker<instance::Context<H>>,
        kinds: &[Kind],
    ) -> Result<Self, Error> {
        let (kind, version) = kind(bytes)?;
        if !kinds.contains(&kind) {
            return Err(Error::Load(format!(
                "the module is a {} program, which this host does not run",
                kind.name()
            )));
        }
        if version != kind.interface_version() {
            return Err(Error::Load(format!(
                "the module speaks version {version} of the {} program interface and this host \
                 version {}",
                kind.name(),
                kind.interface_version()
            )));
        }

        let module = wasmtime::Module::new(engine()?, bytes)
            .map_err(|error| Error::Load(format!("the bytes are not a program module: {error}")))?;
        instance::check(&module)?;

        let pre = linker
            .instantiate_pre(&module)
            .map_err(|error| Error::Load(format!("the module does not link: {error}")))?;
        Ok(Self { pre, kind })
    }
}

impl<H: Control> Module<H> {
    /// Compiles `bytes`, a wasm32 module, as a control program: linked with `uniserve.log`,
    /// `uniserve.stats`, `uniserve.group`, `uniserve.events` and `uniserve.validate`.
    ///
    /// # Errors
    ///
    /// As [`Module::new`].
    pub fn control(bytes: &[u8]) -> Result<Self, Error> {
        let mut linker = wasmtime::Linker::new(engine()?);
        host::log(&mut linker)?;
        host::control(&mut linker)?;
        Self::link(bytes, &linker, &[Kind::Control])
    }
}
