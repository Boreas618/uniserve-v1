//! The host of a processing program (`scheduling::processing`): its module, linked with the
//! vocabulary host functions ([`Module::processing`]), and its instances ([`Processor`]), whose
//! calls run under a budget of their own per export ([`Budgets`]).

use scheduling::processing::video::{InputError, Process, VideoInput, VideoSpec};
use scheduling::processing::{Bind, Bound, Event, Family, Open, Prompt, Refusal, Render, name};
use scheduling::program::{IMPORTS, Kind};
use scheduling::wire::encode;
use serde::Serialize;
use serde::de::DeserializeOwned;
use wasmtime::{Caller, Linker, TypedFunc};

use crate::instance::{Context, Core, Fault};
use crate::{Call, Error, Host, Module, Reason, Usage};

/// The embedder of a processing program: the vocabulary of the checkpoint its instance is bound
/// to, which answers `uniserve.token` and `uniserve.encode`, beside `uniserve.log`.
///
/// The host calls it on the thread that calls the program, during the call.
pub trait Vocabulary: Host {
    /// The id of the token whose text is exactly `text`, if the vocabulary has one.
    fn token(&mut self, text: &str) -> Option<u32>;

    /// `text` encoded without special tokens.
    ///
    /// # Errors
    ///
    /// The tokenizer's failure to encode it.
    fn encode(&mut self, text: &str) -> Result<Vec<u32>, String>;
}

/// The fuel one call of an export runs within.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fuel {
    /// The fuel of the call: roughly the wasm instructions it may execute.
    pub call: u64,
    /// The fuel the call is given beyond `call` for every byte of its message (and of the
    /// replies of the host functions it calls), whose decoding and processing scale with it.
    pub per_byte: u64,
}

/// What each export of a processing program runs within, and the memory of an instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budgets {
    /// `families` and `bind`, once per instance: reading the checkpoint's files and compiling
    /// its template.
    pub bind: Fuel,
    /// `render`, once per request, which scales with the conversation's text.
    pub render: Fuel,
    /// The parser's calls, on every engine event of every streaming request.
    pub parse: Fuel,
    /// `video_check` and `video_process`.
    pub video: Fuel,
    /// The bytes an instance's linear memory may grow to: the most any of its calls needs.
    pub memory: usize,
}

impl Budgets {
    /// What a processing program's calls run within on the leader daemon.
    ///
    /// Fuel is roughly the instructions a call may execute, and each budget is a bound against
    /// a program that never ends, far above what its call takes: rendering decodes and walks
    /// the conversation's JSON and the template's output once each, a few hundred instructions
    /// a byte, and a conversation of hundreds of thousands of tokens is a few megabytes; a parse
    /// call reads the text one engine event decoded. The memory holds such a conversation's
    /// message, its template values and its prompt at once, with room for the compiled
    /// template.
    pub const LEADER: Budgets = Budgets {
        bind: Fuel {
            call: 20_000_000_000,
            per_byte: 2_000,
        },
        render: Fuel {
            call: 2_000_000_000,
            per_byte: 20_000,
        },
        parse: Fuel {
            call: 200_000_000,
            per_byte: 20_000,
        },
        video: Fuel {
            call: 2_000_000_000,
            per_byte: 2_000,
        },
        memory: 2 << 30,
    };
}

impl<H: Vocabulary> Module<H> {
    /// Compiles `bytes`, a wasm32 module of a processing program: linked with `uniserve.log`,
    /// `uniserve.token` and `uniserve.encode`.
    ///
    /// # Errors
    ///
    /// [`Error::Load`] when the bytes are not a module, it states another kind or another
    /// version of the processing interface, it imports anything else, or it lacks the memory
    /// or an export of the processing interface.
    pub fn processing(bytes: &[u8]) -> Result<Self, Error> {
        let mut linker = Linker::new(crate::engine()?);
        crate::host::log(&mut linker)?;
        vocabulary(&mut linker)?;
        Self::link(bytes, &linker, &[Kind::Processing])
    }
}

/// Links `uniserve.token` and `uniserve.encode`.
fn vocabulary<H: Vocabulary>(linker: &mut Linker<Context<H>>) -> Result<(), Error> {
    let link =
        |error: wasmtime::Error| Error::Load(format!("cannot link the host functions: {error}"));
    linker
        .func_wrap(
            IMPORTS,
            name::TOKEN,
            |mut caller: Caller<'_, Context<H>>, ptr: u32, len: u32| {
                let bytes = crate::host::read(&mut caller, ptr, len)?;
                let text = std::str::from_utf8(&bytes)
                    .map_err(|_| Fault::error("the token's text is not UTF-8"))?;
                let id = caller.data_mut().host.token(text);
                // No token of the text is -1, which no id of a u32 vocabulary is.
                Ok(id.map_or(-1, i64::from))
            },
        )
        .map_err(link)?;
    linker
        .func_wrap(
            IMPORTS,
            name::ENCODE,
            |mut caller: Caller<'_, Context<H>>, ptr: u32, len: u32| {
                let text: String = crate::host::message(&mut caller, ptr, len, "text to encode")?;
                let encoded = caller.data_mut().host.encode(&text);
                crate::host::reply(&mut caller, &encoded)
            },
        )
        .map_err(link)?;
    Ok(())
}

/// One instance of a processing program: one bound family and at most one open parser, called
/// one call at a time.
///
/// An instance is created unbound ([`Processor::new`]); every call but [`Processor::families`]
/// needs [`Processor::bind`] first. Once a call ends it ([`Error::Ended`]), every later call
/// returns the same reason: a trap in the family's code ends this instance, and the request it
/// served, alone.
pub struct Processor<H: 'static> {
    core: Core<H>,
    families: TypedFunc<(), u64>,
    bind: TypedFunc<(u32, u32), u64>,
    render: TypedFunc<(u32, u32), u64>,
    open: TypedFunc<(u32, u32), u64>,
    push: TypedFunc<(u32, u32), u64>,
    finish: TypedFunc<(), u64>,
    save: TypedFunc<(), u64>,
    restore: TypedFunc<(u32, u32), u64>,
    check: TypedFunc<(u32, u32), u64>,
    process: TypedFunc<(u32, u32), u64>,
    /// The module's message buffer, as its `inbox` states it: its place and length.
    inbox: (u32, u32),
    budgets: Budgets,
}

impl<H: Vocabulary> Processor<H> {
    /// Instantiates `module`, a processing program's ([`Module::processing`]), with `host`
    /// answering its host functions, within `budgets`.
    ///
    /// # Errors
    ///
    /// [`Error::Load`] when the module does not instantiate or an export is not the
    /// interface's.
    pub fn new(module: &Module<H>, host: H, budgets: &Budgets) -> Result<Self, Error> {
        let (mut core, instance) = Core::start(
            module,
            host,
            budgets.bind.call,
            budgets.bind.per_byte,
            budgets.memory,
        )?;
        let inbox_export = core.export(&instance, name::INBOX)?;
        let inbox = core.buffer(Call::Inbox, inbox_export, budgets.bind.call)?;
        Ok(Self {
            inbox,
            families: core.export(&instance, name::FAMILIES)?,
            bind: core.export(&instance, name::BIND)?,
            render: core.export(&instance, name::RENDER)?,
            open: core.export(&instance, name::PARSE_OPEN)?,
            push: core.export(&instance, name::PARSE_PUSH)?,
            finish: core.export(&instance, name::PARSE_FINISH)?,
            save: core.export(&instance, name::PARSE_SAVE)?,
            restore: core.export(&instance, name::PARSE_RESTORE)?,
            check: core.export(&instance, name::VIDEO_CHECK)?,
            process: core.export(&instance, name::VIDEO_PROCESS)?,
            core,
            budgets: *budgets,
        })
    }

    /// The families the program serves.
    ///
    /// # Errors
    ///
    /// [`Error::Ended`] when the call fails.
    pub fn families(&mut self) -> Result<Vec<Family>, Error> {
        let export = self.families.clone();
        self.call_empty(Call::Families, export, self.budgets.bind)
    }

    /// Binds the instance's family to the checkpoint `bind` states, and the facts it serves
    /// with; `Ok(Err(..))` is the family's refusal of the checkpoint or the options.
    ///
    /// # Errors
    ///
    /// [`Error::Ended`] when the call fails.
    pub fn bind(&mut self, bind: &Bind) -> Result<Result<Bound, String>, Error> {
        let export = self.bind.clone();
        self.call(Call::Bind, export, bind, self.budgets.bind)
    }

    /// The prompt of one chat request, or the family's refusal of it.
    ///
    /// # Errors
    ///
    /// [`Error::Ended`] when the call fails.
    pub fn render(&mut self, render: &Render) -> Result<Result<Prompt, Refusal>, Error> {
        let export = self.render.clone();
        self.call(Call::Render, export, render, self.budgets.render)
    }

    /// Opens the parser of one response.
    ///
    /// # Errors
    ///
    /// [`Error::Ended`] when the call fails.
    pub fn open(&mut self, open: &Open) -> Result<Result<(), String>, Error> {
        let export = self.open.clone();
        self.call(Call::ParseOpen, export, open, self.budgets.parse)
    }

    /// The events one decoded update parses into, or the malformed output that fails the
    /// response.
    ///
    /// # Errors
    ///
    /// [`Error::Ended`] when the call fails.
    pub fn push(&mut self, text: &str) -> Result<Result<Vec<Event>, String>, Error> {
        let export = self.push.clone();
        self.call(Call::ParsePush, export, &text, self.budgets.parse)
    }

    /// The events the parser held back, at the end of generation.
    ///
    /// # Errors
    ///
    /// [`Error::Ended`] when the call fails.
    pub fn finish(&mut self) -> Result<Result<Vec<Event>, String>, Error> {
        let export = self.finish.clone();
        self.call_empty(Call::ParseFinish, export, self.budgets.parse)
    }

    /// The open parser's state, which [`Self::restore`] takes back.
    ///
    /// # Errors
    ///
    /// [`Error::Ended`] when the call fails.
    pub fn save(&mut self) -> Result<Vec<u8>, Error> {
        let export = self.save.clone();
        self.call_empty(Call::ParseSave, export, self.budgets.parse)
    }

    /// Restores a parser from `state`, which [`Self::save`] returned; `Ok(Err(..))` when the
    /// state is not one this program saved.
    ///
    /// # Errors
    ///
    /// [`Error::Ended`] when the call fails.
    pub fn restore(&mut self, state: &[u8]) -> Result<Result<(), String>, Error> {
        let export = self.restore.clone();
        self.call(Call::ParseRestore, export, &state, self.budgets.parse)
    }

    /// Applies the bound video family's rules that need no media.
    ///
    /// # Errors
    ///
    /// [`Error::Ended`] when the call fails.
    pub fn check_video(&mut self, spec: &VideoSpec) -> Result<Result<(), InputError>, Error> {
        let export = self.check.clone();
        self.call(Call::VideoCheck, export, spec, self.budgets.video)
    }

    /// Processes a video request whose references were probed.
    ///
    /// # Errors
    ///
    /// [`Error::Ended`] when the call fails.
    pub fn process_video(
        &mut self,
        process: &Process,
    ) -> Result<Result<VideoInput, InputError>, Error> {
        let export = self.process.clone();
        self.call(Call::VideoProcess, export, process, self.budgets.video)
    }

    /// The call that ended the instance and why; `None` while it runs.
    pub fn ended(&self) -> Option<&(Call, Reason)> {
        self.core.ended()
    }

    /// The calls made into the instance and the fuel they consumed.
    pub fn usage(&self) -> Usage {
        self.core.usage()
    }

    /// The bytes the instance's linear memory holds now, which only grows.
    pub fn memory(&self) -> usize {
        self.core.memory_size()
    }

    /// The embedder, to change between calls: a request's instance answers the vocabulary of
    /// the checkpoint it was bound to.
    pub fn host_mut(&mut self) -> &mut H {
        &mut self.core.store.data_mut().host
    }

    /// One call of `export` with `message` under `budget`, its reply decoded as an `O`.
    fn call<I: Serialize, O: DeserializeOwned>(
        &mut self,
        call: Call,
        export: TypedFunc<(u32, u32), u64>,
        message: &I,
        budget: Fuel,
    ) -> Result<O, Error> {
        let message = encode(message).map_err(|error| Error::Message(error.to_string()))?;
        self.core.consume(
            call,
            export,
            &message,
            self.inbox,
            budget.call,
            budget.per_byte,
        )
    }

    /// One call of `export`, which takes no message, under `budget`.
    fn call_empty<O: DeserializeOwned>(
        &mut self,
        call: Call,
        export: TypedFunc<(), u64>,
        budget: Fuel,
    ) -> Result<O, Error> {
        self.core.consume_empty(call, export, budget.call)
    }
}

impl<H> std::fmt::Debug for Processor<H> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Processor").finish_non_exhaustive()
    }
}
