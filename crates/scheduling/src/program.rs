//! The program interface: what every program of a policy package implements, the exports a
//! program module offers its host, and the host functions it imports.
//!
//! A program reads one input and gives one output; its own fields are its state, which the host
//! saves and restores across restarts and package updates. The four programs differ only in
//! their input and output and in when the host steps them:
//!
//! | Program | Host | Input | Output | Stepped |
//! | --- | --- | --- | --- | --- |
//! | control | leader daemon | [`crate::control::State`] | `Option<Config>` | every `period` |
//! | routing | leader daemon | [`crate::routing::Routing`] | [`crate::routing::Route`] | every request |
//! | KV | every Storage Server | [`crate::kv::Hook`] | [`crate::kv::Action`] | every handoff; every `period` on the leader |
//! | scheduling | every scheduler | the scheduling policy interface ([`crate::policy::Policy`]) | | every step |
//!
//! # The module
//!
//! A program crate implements [`Program`] for its type and invokes [`export!`](crate::export)
//! once, which generates the module's exports:
//!
//! | Export | Signature | Message | Reply |
//! | --- | --- | --- | --- |
//! | `alloc` | `(len: u32) -> u32` | | a buffer of `len` bytes the host writes a message into |
//! | `free` | `(ptr: u32, len: u32)` | | releases a message buffer or a reply buffer |
//! | `new` | `(ptr: u32, len: u32) -> u64` | the package's `args` table | `0`, or the text of why the table is not the program's `Args` |
//! | `step` | `(ptr: u32, len: u32) -> u64` | the input | the output |
//! | `save` | `() -> u64` | | the program's state |
//! | `restore` | `(ptr: u32, len: u32) -> u64` | a state `save` replied | `0`, or the text of why the state is not the program's type |
//!
//! The module also holds the custom section [`SECTION`], which states the program's [`Kind`]
//! (from its `Input`) and the version of the interface it was compiled against: a host reads it
//! before it links the module, so a package names a program's source and never its kind.
//!
//! Every message and reply is JSON of its type ([`crate::wire::encode`]) in a buffer of the
//! module's linear memory; a reply is packed as `ptr << 32 | len` ([`crate::wire::pack`]), and
//! the host copies it out and frees it. A program fails only by trapping: a panic, a message
//! that does not decode, an exhausted fuel budget. Before it traps on a panic, the module hands
//! the panic's text to the host ([`PANIC`]), which reports it as the reason the instance ended.
//!
//! # The host functions
//!
//! A module imports from the module [`IMPORTS`] only what it uses; its host links what the
//! program's kind may call, and a module that imports anything else does not load.
//!
//! | Import | Signature | Programs | Message | Reply |
//! | --- | --- | --- | --- | --- |
//! | `log` | `(level: u32, ptr: u32, len: u32)` | all | the record's text | |
//! | `stats` | `(ptr: u32, len: u32) -> f64` | control | [`crate::control::Query`] | the value |
//! | `group` | `(ptr: u32, len: u32) -> u64` | control | [`crate::control::Group`] | `BTreeMap<String, f64>` |
//! | `events` | `(ptr: u32, len: u32) -> u64` | control | [`crate::control::Filter`] | `Vec<Record>` |
//! | `validate` | `(ptr: u32, len: u32) -> u64` | control | `Config` | `Result<Report, String>` |
//!
//! A reply of a host function is a buffer the host allocates with the module's `alloc` export,
//! packed as an export's reply is, which the module frees once it has decoded it.

use serde::Serialize;
use serde::de::DeserializeOwned;

/// A program: reads one input, gives one output. Its own fields are its hidden state, which the
/// host saves ([`Serialize`]) and restores ([`DeserializeOwned`]); a field added to a program
/// takes `#[serde(default)]` so that a state saved before decodes.
pub trait Program: Serialize + DeserializeOwned {
    /// What the package's `args` table for the program deserializes to.
    type Args: DeserializeOwned;
    /// What one step reads, which makes the program the [`Kind`] its input is read by.
    type Input: Input;
    /// What one step gives.
    type Output: Serialize;

    /// Creates the program when its package is applied.
    fn new(args: Self::Args) -> Self;

    /// Reads one input and gives one output.
    fn step(&mut self, input: &Self::Input) -> Self::Output;
}

/// The kinds of program, by where they run and what they read: a module states its kind
/// ([`SECTION`]), so a package's `config.toml` names a program's source and never its kind.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// The leader daemon's control program: [`crate::control::State`] in, a configuration out.
    Control,
    /// The leader daemon's routing program: [`crate::routing::Routing`] in,
    /// [`crate::routing::Route`] out.
    Routing,
    /// Every Storage Server's KV program: [`crate::kv::Hook`] in, [`crate::kv::Action`] out.
    Kv,
    /// Every scheduler's scheduling policy ([`crate::policy::Policy`]).
    Scheduling,
}

impl Kind {
    /// Every kind.
    pub const ALL: [Kind; 4] = [Kind::Control, Kind::Routing, Kind::Kv, Kind::Scheduling];

    /// The kind's name, as `events` and the configuration's reports name it.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Control => "control",
            Kind::Routing => "routing",
            Kind::Kv => "kv",
            Kind::Scheduling => "scheduling",
        }
    }

    /// The kind's number in a module's [`SECTION`].
    pub const fn code(self) -> u8 {
        match self {
            Kind::Control => 1,
            Kind::Routing => 2,
            Kind::Kv => 3,
            Kind::Scheduling => 4,
        }
    }

    /// The kind numbered `code` in a module's [`SECTION`]; `None` for a number of no kind.
    pub fn of_code(code: u8) -> Option<Kind> {
        Kind::ALL.into_iter().find(|kind| kind.code() == code)
    }

    /// The version of the interface a program of this kind speaks: [`INTERFACE_VERSION`], or the
    /// scheduling policy interface's own ([`crate::INTERFACE_VERSION`]).
    pub fn interface_version(self) -> u32 {
        match self {
            Kind::Control | Kind::Routing | Kind::Kv => INTERFACE_VERSION,
            Kind::Scheduling => crate::INTERFACE_VERSION,
        }
    }
}

/// What a program reads in a step: the input of one kind of program.
pub trait Input: DeserializeOwned {
    /// The kind of program that reads it.
    const KIND: Kind;
}

impl Input for crate::control::State {
    const KIND: Kind = Kind::Control;
}

impl Input for crate::routing::Routing {
    const KIND: Kind = Kind::Routing;
}

impl Input for crate::kv::Hook {
    const KIND: Kind = Kind::Kv;
}

/// A KV program's handoff hook is a program of its own inside the KV program ([`crate::kv::Kv`]).
impl Input for crate::kv::Handoff {
    const KIND: Kind = Kind::Kv;
}

/// A KV program's tiering hook is a program of its own inside the KV program ([`crate::kv::Kv`]).
impl Input for crate::kv::Tiering {
    const KIND: Kind = Kind::Kv;
}

/// The version of the program interface of the control, routing and KV programs: the exports,
/// the host functions and the messages of this module and of [`crate::control`],
/// [`crate::routing`] and [`crate::kv`]. Every change that is not an addition raises it.
pub const INTERFACE_VERSION: u32 = 1;

/// The custom section of a program module that states what it is: its [`Kind`]'s code, then the
/// version of the interface it was compiled against as a little-endian `u32` ([`section`]). The
/// export macros write it ([`export!`](crate::export), [`export_policy!`](crate::export_policy));
/// a host reads it before it links the module, and a Build Server records it.
pub const SECTION: &str = "uniserve.program";

/// The bytes of a module's [`SECTION`] for a program of `kind` compiled against `version`.
pub const fn section(kind: Kind, version: u32) -> [u8; 5] {
    let version = version.to_le_bytes();
    [kind.code(), version[0], version[1], version[2], version[3]]
}

/// The kind and interface version a module's [`SECTION`] states; `None` when the bytes are not
/// such a statement.
pub fn stated(bytes: &[u8]) -> Option<(Kind, u32)> {
    let [code, a, b, c, d] = *bytes else {
        return None;
    };
    Some((Kind::of_code(code)?, u32::from_le_bytes([a, b, c, d])))
}

/// The import module of the host functions.
pub const IMPORTS: &str = "uniserve";

/// The names of the exports ([`EXPORTS`]) and of the host functions.
pub mod name {
    pub const MEMORY: &str = "memory";
    pub const ALLOC: &str = "alloc";
    pub const FREE: &str = "free";
    pub const NEW: &str = "new";
    pub const STEP: &str = "step";
    pub const SAVE: &str = "save";
    pub const RESTORE: &str = "restore";

    pub const LOG: &str = "log";
    pub const STATS: &str = "stats";
    pub const GROUP: &str = "group";
    pub const EVENTS: &str = "events";
    pub const VALIDATE: &str = "validate";
}

/// The functions a program module exports beside its `memory`.
pub const EXPORTS: [&str; 6] = [
    name::ALLOC,
    name::FREE,
    name::NEW,
    name::STEP,
    name::SAVE,
    name::RESTORE,
];

/// The level `log` is called with to hand the host the text of the panic that is about to trap.
pub const PANIC: u32 = 0;

/// The level of a log record, as `uniserve.log` carries it (the `log` crate's levels).
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

impl Level {
    /// The level `uniserve.log` was called with, `None` for a number that is no level ([`PANIC`]
    /// included).
    pub fn from_wire(level: u32) -> Option<Level> {
        match level {
            1 => Some(Level::Error),
            2 => Some(Level::Warn),
            3 => Some(Level::Info),
            4 => Some(Level::Debug),
            5 => Some(Level::Trace),
            _ => None,
        }
    }

    fn of(level: log::Level) -> Level {
        match level {
            log::Level::Error => Level::Error,
            log::Level::Warn => Level::Warn,
            log::Level::Info => Level::Info,
            log::Level::Debug => Level::Debug,
            log::Level::Trace => Level::Trace,
        }
    }
}

/// Generates the exports of the [`Program`] type `$program` (see [`crate::program`]).
///
/// A program crate invokes it once, at its root:
///
/// ```ignore
/// export!(Chunk);
/// ```
///
/// The exports exist only in the wasm32 module; a native build of the crate (its own tests, the
/// workspace's checks) checks that `$program` is a [`Program`] and exports nothing, since the
/// export names are those of the C library's allocator.
#[macro_export]
macro_rules! export {
    ($program:ty) => {
        const _: fn() = || {
            fn program<P: $crate::program::Program>() {}
            program::<$program>();
        };

        // The module's statement of its kind and interface version, which its host reads before
        // it links it.
        #[cfg(target_arch = "wasm32")]
        #[used]
        #[unsafe(link_section = "uniserve.program")]
        static PROGRAM_SECTION: [u8; 5] = $crate::program::section(
            <<$program as $crate::program::Program>::Input as $crate::program::Input>::KIND,
            $crate::program::INTERFACE_VERSION,
        );

        #[cfg(target_arch = "wasm32")]
        const _: () = {
            // The program of this module, which `new` or `restore` fills. wasm32 without threads
            // runs one export at a time and the host calls `new` or `restore` before any other
            // call of the program, which makes the one mutable slot sound.
            static mut PROGRAM: ::core::option::Option<$program> = ::core::option::Option::None;

            #[unsafe(no_mangle)]
            extern "C" fn alloc(len: u32) -> u32 {
                $crate::abi::allocate(len)
            }

            #[unsafe(no_mangle)]
            unsafe extern "C" fn free(ptr: u32, len: u32) {
                // SAFETY: the host hands back a buffer `alloc` or a reply made.
                unsafe { $crate::abi::release(ptr, len) }
            }

            #[unsafe(no_mangle)]
            extern "C" fn new(ptr: u32, len: u32) -> u64 {
                // SAFETY: the host's message lives for this call, and the slot is this module's.
                unsafe { $crate::program::guest::new::<$program>(&raw mut PROGRAM, ptr, len) }
            }

            #[unsafe(no_mangle)]
            extern "C" fn step(ptr: u32, len: u32) -> u64 {
                // SAFETY: as in `new`.
                unsafe { $crate::program::guest::step::<$program>(&raw mut PROGRAM, ptr, len) }
            }

            #[unsafe(no_mangle)]
            extern "C" fn save() -> u64 {
                // SAFETY: as in `new`.
                unsafe { $crate::program::guest::save::<$program>(&raw mut PROGRAM) }
            }

            #[unsafe(no_mangle)]
            extern "C" fn restore(ptr: u32, len: u32) -> u64 {
                // SAFETY: as in `new`.
                unsafe { $crate::program::guest::restore::<$program>(&raw mut PROGRAM, ptr, len) }
            }
        };
    };
}

/// The module side of the exports and the host functions, which [`export!`](crate::export)
/// expands into calls of, and through which the views of [`crate::control`] call the host.
pub mod guest {
    use std::sync::Once;

    use serde::Serialize;
    use serde::de::DeserializeOwned;

    use super::{Level, PANIC, Program};
    use crate::abi::{hand, message};
    use crate::wire::{decode, encode, unpack};

    /// Installs, once per module, the panic hook that hands the host a panic's text before the
    /// module traps, and the logger that forwards the `log` crate's records to `uniserve.log`.
    fn install() {
        static INSTALL: Once = Once::new();

        INSTALL.call_once(|| {
            std::panic::set_hook(Box::new(|panic| imports::log(PANIC, &panic.to_string())));
            if log::set_logger(&LOGGER).is_ok() {
                log::set_max_level(log::LevelFilter::Trace);
            }
        });
    }

    /// The `log` crate's logger of a program module.
    struct Logger;

    static LOGGER: Logger = Logger;

    impl log::Log for Logger {
        fn enabled(&self, _: &log::Metadata<'_>) -> bool {
            true
        }

        fn log(&self, record: &log::Record<'_>) {
            let text = record.args().to_string();
            imports::log(Level::of(record.level()) as u32, &text);
        }

        fn flush(&self) {}
    }

    /// The program in `slot`, a trap before `new` or `restore` filled it.
    ///
    /// # Safety
    ///
    /// `slot` is the module's one program slot, and no other reference to it is live.
    unsafe fn program<'a, P>(slot: *mut Option<P>) -> &'a mut P {
        // SAFETY: the caller's obligation.
        let slot = unsafe { &mut *slot };
        slot.as_mut()
            .unwrap_or_else(|| panic!("the host calls `new` or `restore` before any other call"))
    }

    /// Encodes `value` into a reply buffer, a trap when it does not encode.
    fn reply<T: Serialize>(value: &T, what: &str) -> u64 {
        let bytes =
            encode(value).unwrap_or_else(|error| panic!("the {what} does not encode: {error}"));
        hand(&bytes)
    }

    /// The `new` export: creates the program from the package's `args` table; replies the text
    /// of why the table is not `P::Args` instead.
    ///
    /// # Safety
    ///
    /// `ptr` and `len` name the host's message for this call; `slot` is the module's one slot.
    pub unsafe fn new<P: Program>(slot: *mut Option<P>, ptr: u32, len: u32) -> u64 {
        install();
        // SAFETY: the caller's obligation.
        let bytes = unsafe { message(ptr, len) };

        match decode::<P::Args>(bytes) {
            Ok(args) => {
                // SAFETY: the caller's obligation.
                unsafe { *slot = Some(P::new(args)) };
                0
            }
            Err(error) => hand(error.to_string().as_bytes()),
        }
    }

    /// The `step` export: one input, one output.
    ///
    /// # Safety
    ///
    /// As [`new`].
    pub unsafe fn step<P: Program>(slot: *mut Option<P>, ptr: u32, len: u32) -> u64 {
        // SAFETY: the caller's obligation.
        let bytes = unsafe { message(ptr, len) };
        let input: P::Input =
            decode(bytes).unwrap_or_else(|error| panic!("the input does not decode: {error}"));

        // SAFETY: the caller's obligation.
        let output = unsafe { program(slot) }.step(&input);
        reply(&output, "output")
    }

    /// The `save` export: the program's state.
    ///
    /// # Safety
    ///
    /// `slot` is the module's one slot.
    pub unsafe fn save<P: Program>(slot: *mut Option<P>) -> u64 {
        // SAFETY: the caller's obligation.
        let program = unsafe { program(slot) };
        reply(program, "state")
    }

    /// The `restore` export: the program from a state `save` replied, perhaps by an earlier
    /// version of the program; replies the text of why it is not `P` instead.
    ///
    /// # Safety
    ///
    /// As [`new`].
    pub unsafe fn restore<P: Program>(slot: *mut Option<P>, ptr: u32, len: u32) -> u64 {
        install();
        // SAFETY: the caller's obligation.
        let bytes = unsafe { message(ptr, len) };

        match decode::<P>(bytes) {
            Ok(program) => {
                // SAFETY: the caller's obligation.
                unsafe { *slot = Some(program) };
                0
            }
            Err(error) => hand(error.to_string().as_bytes()),
        }
    }

    /// Calls the host function `function` with `request` encoded and decodes the buffer it
    /// replies; a trap when either does not code, since host and program are built from the
    /// same types.
    fn exchange<T: DeserializeOwned>(
        function: fn(u32, u32) -> u64,
        request: &impl Serialize,
        what: &str,
    ) -> T {
        let bytes =
            encode(request).unwrap_or_else(|error| panic!("the {what} does not encode: {error}"));
        let Ok(len) = u32::try_from(bytes.len()) else {
            panic!("a {what} fits in a u32 length");
        };

        let packed = function(bytes.as_ptr() as usize as u32, len);
        let (ptr, len) =
            unpack(packed).unwrap_or_else(|| panic!("the host replied nothing to the {what}"));
        // SAFETY: the host wrote the reply into a buffer of this module's `alloc` export.
        let reply = unsafe { message(ptr, len) };
        let value = decode(reply)
            .unwrap_or_else(|error| panic!("the reply to the {what} does not decode: {error}"));
        // SAFETY: the buffer came from `alloc` with this length and is no longer read.
        unsafe { crate::abi::release(ptr, len) };
        value
    }

    /// `uniserve.stats`: the value of one aggregate over one window of a series.
    pub fn stats(query: &crate::control::Query) -> f64 {
        let bytes =
            encode(query).unwrap_or_else(|error| panic!("the query does not encode: {error}"));
        let Ok(len) = u32::try_from(bytes.len()) else {
            panic!("a query fits in a u32 length");
        };
        imports::stats(bytes.as_ptr() as usize as u32, len)
    }

    /// `uniserve.group`: one aggregate of each value of a label among a selection's series.
    pub fn group(group: &crate::control::Group) -> std::collections::BTreeMap<String, f64> {
        exchange(imports::group, group, "grouping")
    }

    /// `uniserve.events`: the records of one kind a filter selects.
    pub fn events(filter: &crate::control::Filter) -> Vec<crate::control::Record> {
        exchange(imports::events, filter, "filter")
    }

    /// `uniserve.validate`: what applying `config` would do, or why it is rejected.
    pub fn validate(config: &crate::control::Config) -> Result<crate::control::Report, String> {
        exchange(imports::validate, config, "configuration")
    }

    /// The host functions. They exist only in a wasm32 module, where the host links them; a
    /// native build of a program has no host to call.
    #[cfg(target_arch = "wasm32")]
    mod imports {
        #[link(wasm_import_module = "uniserve")]
        unsafe extern "C" {
            #[link_name = "log"]
            fn host_log(level: u32, ptr: u32, len: u32);
            #[link_name = "stats"]
            fn host_stats(ptr: u32, len: u32) -> f64;
            #[link_name = "group"]
            fn host_group(ptr: u32, len: u32) -> u64;
            #[link_name = "events"]
            fn host_events(ptr: u32, len: u32) -> u64;
            #[link_name = "validate"]
            fn host_validate(ptr: u32, len: u32) -> u64;
        }

        pub(super) fn log(level: u32, text: &str) {
            // SAFETY: the host reads `len` bytes at `ptr` during the call; `text` outlives it.
            unsafe { host_log(level, text.as_ptr() as u32, text.len() as u32) }
        }

        pub(super) fn stats(ptr: u32, len: u32) -> f64 {
            // SAFETY: the host reads the message during the call; the caller keeps it alive.
            unsafe { host_stats(ptr, len) }
        }

        pub(super) fn group(ptr: u32, len: u32) -> u64 {
            // SAFETY: as in `stats`.
            unsafe { host_group(ptr, len) }
        }

        pub(super) fn events(ptr: u32, len: u32) -> u64 {
            // SAFETY: as in `stats`.
            unsafe { host_events(ptr, len) }
        }

        pub(super) fn validate(ptr: u32, len: u32) -> u64 {
            // SAFETY: as in `stats`.
            unsafe { host_validate(ptr, len) }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod imports {
        const HOSTLESS: &str = "a program calls its host only as a wasm32 module in the program \
                                host; a native build has none";

        pub(super) fn log(_: u32, _: &str) {
            panic!("{HOSTLESS}")
        }

        pub(super) fn stats(_: u32, _: u32) -> f64 {
            panic!("{HOSTLESS}")
        }

        pub(super) fn group(_: u32, _: u32) -> u64 {
            panic!("{HOSTLESS}")
        }

        pub(super) fn events(_: u32, _: u32) -> u64 {
            panic!("{HOSTLESS}")
        }

        pub(super) fn validate(_: u32, _: u32) -> u64 {
            panic!("{HOSTLESS}")
        }
    }
}
