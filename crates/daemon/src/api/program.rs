//! An epoch's processing program on the leader daemon: the request side of its model families,
//! a wasm32 module its module set fixes (`ModuleSet::processing`) and the daemon loads per epoch.
//!
//! - [`Program`] is one loaded module: compiled once, with the families it declares, which a
//!   checkpoint resolves by ([`Program::resolve`]).
//! - [`Binding`] is one family of it bound to one checkpoint: the binding's facts, and the pool
//!   of instances bound to it. Every instance is bound with the same files and options, so each
//!   states the facts the first stated.
//! - [`Session`] is one request's instance, checked out of its binding's pool when the request
//!   arrives and returned when its response ends: the request's render, its parser and its video
//!   rules run in it, one call at a time, so a parser needs no identity and a fault in the
//!   family's code ends that request's instance alone. An instance whose call failed, or whose
//!   memory grew past [`IDLE_MEMORY`], is dropped instead of returned.
//!
//! The program reaches the checkpoint's tokenizer through its host functions ([`Vocabulary`]),
//! which the daemon answers from the tokenizer it holds: the program states text, and the
//! daemon tokenizes it.

use std::sync::{Arc, Mutex};

use program::{Budgets, Error as ProgramError, Host, Level, Module, Processor};
use proto::module::PackageRevision;
use scheduling::processing::video::{InputError, Process, VideoInput, VideoSpec};
use scheduling::processing::{
    Bind, Bound, Event, Family, Open, Prompt, Refusal, Render, ResolveError, RootFile,
};
use sha2::{Digest as _, Sha256};
use tracing::{debug, warn};

use crate::api::tokenizer::DynTokenizer;

/// The most linear memory an idle instance keeps: one that grew past it, rendering a long
/// conversation, is dropped when its request ends rather than held by the pool.
pub const IDLE_MEMORY: usize = 256 << 20;

/// The most idle instances a binding keeps: what serves the requests that arrive at once without
/// binding another instance, beyond which an ended request's instance is dropped.
pub const IDLE_INSTANCES: usize = 256;

/// Why a processing program does not load, bind, or answer a call.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The module does not load, its instance ended, or its call failed otherwise.
    #[error("the processing program {program} failed: {source}")]
    Program {
        /// The program's revision, as `backend@revision`.
        program: String,
        /// The host's report.
        #[source]
        source: ProgramError,
    },
    /// The family refused the checkpoint or the options it was bound with.
    #[error("the {family} family refuses the checkpoint: {message}")]
    Bind {
        /// The family.
        family: String,
        /// Its refusal.
        message: String,
    },
    /// An instance bound later stated other facts than the binding's first.
    #[error("an instance of the {family} family stated other facts than its binding's first")]
    Facts {
        /// The family.
        family: String,
    },
}

/// The daemon's answer to a processing program's host functions: the bound checkpoint's
/// tokenizer, and its log, which the daemon's own carries.
pub struct Vocabulary {
    /// The tokenizer; `None` for an instance that only declares its families.
    tokenizer: Option<DynTokenizer>,
    /// The program's revision, which its log records name.
    program: Arc<str>,
}

impl Host for Vocabulary {
    fn log(&mut self, level: Level, text: &str) {
        match level {
            Level::Error | Level::Warn => warn!(program = %self.program, "{text}"),
            Level::Info | Level::Debug | Level::Trace => debug!(program = %self.program, "{text}"),
        }
    }
}

impl program::Vocabulary for Vocabulary {
    fn token(&mut self, text: &str) -> Option<u32> {
        self.tokenizer.as_ref()?.token_to_id(text)
    }

    fn encode(&mut self, text: &str) -> Result<Vec<u32>, String> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or("the instance is bound to no tokenizer")?;
        tokenizer
            .encode(text, false)
            .map_err(|error| error.to_string())
    }
}

/// One loaded processing program.
pub struct Program {
    module: Module<Vocabulary>,
    /// The package revision it was built from.
    revision: PackageRevision,
    /// The SHA-256 of its module, in hex: what tells two programs apart.
    digest: String,
    /// `backend@revision`, as reports name it.
    name: Arc<str>,
    families: Vec<Family>,
    budgets: Budgets,
}

impl std::fmt::Debug for Program {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Program")
            .field("name", &self.name)
            .field("digest", &self.digest)
            .finish_non_exhaustive()
    }
}

impl Program {
    /// Compiles the module `bytes` of the processing program `revision` and reads the families
    /// it declares, under the leader's budgets ([`Budgets::LEADER`]).
    ///
    /// # Errors
    ///
    /// Fails when the bytes are no processing program of this daemon's interface version, or
    /// its `families` call fails.
    pub fn load(bytes: &[u8], revision: PackageRevision) -> Result<Arc<Self>, Error> {
        Self::load_within(bytes, revision, Budgets::LEADER)
    }

    /// [`Self::load`] under `budgets`.
    ///
    /// # Errors
    ///
    /// As [`Self::load`].
    pub fn load_within(
        bytes: &[u8],
        revision: PackageRevision,
        budgets: Budgets,
    ) -> Result<Arc<Self>, Error> {
        let name: Arc<str> = format!("{}@{}", revision.backend, revision.revision).into();
        let failed = |source| Error::Program {
            program: name.to_string(),
            source,
        };
        let module = Module::processing(bytes).map_err(failed)?;
        let host = Vocabulary {
            tokenizer: None,
            program: Arc::clone(&name),
        };
        let families = Processor::new(&module, host, &budgets)
            .and_then(|mut instance| instance.families())
            .map_err(failed)?;
        Ok(Arc::new(Self {
            module,
            revision,
            digest: format!("{:x}", Sha256::digest(bytes)),
            name,
            families,
            budgets,
        }))
    }

    /// The package revision the program was built from.
    pub fn revision(&self) -> &PackageRevision {
        &self.revision
    }

    /// The SHA-256 of its module, in hex.
    pub fn digest(&self) -> &str {
        &self.digest
    }

    /// The families it declares.
    pub fn families(&self) -> &[Family] {
        &self.families
    }

    /// The root files its families declare their classes in, each once.
    pub fn root_files(&self) -> Vec<RootFile> {
        scheduling::processing::root_files(&self.families)
    }

    /// The family a checkpoint declares in its root files: `files` holds the text of each of
    /// [`Self::root_files`] the checkpoint publishes.
    ///
    /// # Errors
    ///
    /// As `scheduling::processing::resolve`: a checkpoint no family claims is refused, naming
    /// what the program serves.
    pub fn resolve<'t>(
        &self,
        files: impl IntoIterator<Item = (RootFile, &'t str)>,
    ) -> Result<&Family, ResolveError> {
        scheduling::processing::resolve(&self.families, files)
    }

    /// Binds `bind.family` to the checkpoint `bind` states, whose tokenizer is `tokenizer`:
    /// the binding new requests check their instances out of, and its facts.
    ///
    /// # Errors
    ///
    /// [`Error::Bind`] for the family's refusal, [`Error::Program`] when the call fails.
    pub fn bind(
        self: &Arc<Self>,
        bind: Bind,
        tokenizer: DynTokenizer,
    ) -> Result<Arc<Binding>, Error> {
        let mut instance = self.instance(&tokenizer)?;
        let facts = self.bound(&mut instance, &bind)?;
        Ok(Arc::new(Binding {
            program: Arc::clone(self),
            bind,
            tokenizer,
            facts,
            idle: Mutex::new(vec![instance]),
        }))
    }

    /// A fresh instance answering `tokenizer`'s vocabulary.
    fn instance(&self, tokenizer: &DynTokenizer) -> Result<Processor<Vocabulary>, Error> {
        let host = Vocabulary {
            tokenizer: Some(Arc::clone(tokenizer)),
            program: Arc::clone(&self.name),
        };
        Processor::new(&self.module, host, &self.budgets).map_err(|source| self.failed(source))
    }

    /// Binds `instance` with `bind`, and the facts it states.
    fn bound(&self, instance: &mut Processor<Vocabulary>, bind: &Bind) -> Result<Bound, Error> {
        instance
            .bind(bind)
            .map_err(|source| self.failed(source))?
            .map_err(|message| Error::Bind {
                family: bind.family.clone(),
                message,
            })
    }

    /// The failure of a call into the program.
    fn failed(&self, source: ProgramError) -> Error {
        Error::Program {
            program: self.name.to_string(),
            source,
        }
    }
}

/// One family of a program bound to one checkpoint, and the pool of its instances.
pub struct Binding {
    program: Arc<Program>,
    /// What every instance is bound with.
    bind: Bind,
    tokenizer: DynTokenizer,
    /// The facts the first instance stated.
    facts: Bound,
    /// The instances no request holds, bound and ready.
    idle: Mutex<Vec<Processor<Vocabulary>>>,
}

impl std::fmt::Debug for Binding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Binding")
            .field("program", &self.program)
            .field("family", &self.bind.family)
            .finish_non_exhaustive()
    }
}

impl Binding {
    /// The program it is a family of.
    pub fn program(&self) -> &Arc<Program> {
        &self.program
    }

    /// The bound family's declaration.
    pub fn family(&self) -> Option<&Family> {
        self.program
            .families
            .iter()
            .find(|family| family.id == self.bind.family)
    }

    /// The facts the binding states.
    pub fn facts(&self) -> &Bound {
        &self.facts
    }

    /// An instance for one request: an idle one, or one bound now.
    ///
    /// # Errors
    ///
    /// Fails when a new instance does not bind as the binding's first did.
    pub fn checkout(self: &Arc<Self>) -> Result<Session, Error> {
        let idle = service::lock(&self.idle).pop();
        let instance = match idle {
            Some(instance) => instance,
            None => {
                let mut instance = self.program.instance(&self.tokenizer)?;
                if self.program.bound(&mut instance, &self.bind)? != self.facts {
                    return Err(Error::Facts {
                        family: self.bind.family.clone(),
                    });
                }
                instance
            }
        };
        Ok(Session {
            instance: Some(instance),
            binding: Arc::clone(self),
        })
    }

    /// Takes `instance` back from the request that ends with it, unless it ended, grew past
    /// [`IDLE_MEMORY`], or the pool is full.
    fn release(&self, instance: Processor<Vocabulary>) {
        if instance.ended().is_some() || instance.memory() > IDLE_MEMORY {
            return;
        }
        let mut idle = service::lock(&self.idle);
        if idle.len() < IDLE_INSTANCES {
            idle.push(instance);
        }
    }
}

/// One request's instance of its binding, returned to the pool when the session drops.
pub struct Session {
    /// The instance, `None` only while it is dropped.
    instance: Option<Processor<Vocabulary>>,
    binding: Arc<Binding>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Session")
            .field("binding", &self.binding)
            .finish_non_exhaustive()
    }
}

impl Session {
    /// The binding the session's instance is of.
    pub fn binding(&self) -> &Arc<Binding> {
        &self.binding
    }

    /// Calls the session's instance.
    fn call<T>(
        &mut self,
        call: impl FnOnce(&mut Processor<Vocabulary>) -> Result<T, ProgramError>,
    ) -> Result<T, Error> {
        // The instance is taken only as the session drops.
        let Some(instance) = self.instance.as_mut() else {
            let gone = ProgramError::Message("the session's instance is gone".to_owned());
            return Err(self.binding.program.failed(gone));
        };
        call(instance).map_err(|source| self.binding.program.failed(source))
    }

    /// The prompt of one chat request, or the family's refusal of it.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] when the call fails.
    pub fn render(&mut self, render: &Render) -> Result<Result<Prompt, Refusal>, Error> {
        self.call(|instance| instance.render(render))
    }

    /// Opens the response's parser; `Ok(Err(..))` when the request's tools cannot be parsed
    /// against.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] when the call fails.
    pub fn open(&mut self, open: &Open) -> Result<Result<(), String>, Error> {
        self.call(|instance| instance.open(open))
    }

    /// The events one decoded update parses into, or the malformed output that fails the
    /// response.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] when the call fails.
    pub fn push(&mut self, text: &str) -> Result<Result<Vec<Event>, String>, Error> {
        self.call(|instance| instance.push(text))
    }

    /// The events the parser held back, at the end of generation.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] when the call fails.
    pub fn finish(&mut self) -> Result<Result<Vec<Event>, String>, Error> {
        self.call(Processor::finish)
    }

    /// The parser's state, which [`Self::restore`] takes back in this program's instances.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] when the call fails.
    pub fn save(&mut self) -> Result<Vec<u8>, Error> {
        self.call(Processor::save)
    }

    /// Restores the parser `state` states; `Ok(Err(..))` when it is not this program's.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] when the call fails.
    pub fn restore(&mut self, state: &[u8]) -> Result<Result<(), String>, Error> {
        self.call(|instance| instance.restore(state))
    }

    /// Applies the bound video family's rules that need no media.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] when the call fails.
    pub fn check_video(&mut self, spec: &VideoSpec) -> Result<Result<(), InputError>, Error> {
        self.call(|instance| instance.check_video(spec))
    }

    /// Processes a video request whose references were probed.
    ///
    /// # Errors
    ///
    /// [`Error::Program`] when the call fails.
    pub fn process_video(
        &mut self,
        process: &Process,
    ) -> Result<Result<VideoInput, InputError>, Error> {
        self.call(|instance| instance.process_video(process))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(instance) = self.instance.take() {
            self.binding.release(instance);
        }
    }
}

/// The module of the processing program `revision`, which the Build Server at `socket` builds
/// first when its store does not hold it, loaded. Blocks on the Build Server.
///
/// # Errors
///
/// Fails when the Build Server cannot be reached or holds no such revision, the module does not
/// build, or it does not load.
pub fn fetch(socket: &std::path::Path, revision: &PackageRevision) -> anyhow::Result<Arc<Program>> {
    use std::io::Read as _;

    use anyhow::Context as _;

    let mut client = build_server::Client::connect(socket)
        .with_context(|| format!("connecting to the Build Server at {}", socket.display()))?;
    let module = client
        .resolve(
            proto::module::PROGRAM,
            revision,
            &build_server::Target::wasm(),
            build_server::manifest::SINGLE_KEY,
            &mut |_| {},
        )
        .with_context(|| {
            format!(
                "resolving the processing program {}@{}",
                revision.backend, revision.revision
            )
        })?;
    let file = module
        .files
        .into_iter()
        .find(|file| file.name == build_server::source::MODULE_WASM)
        .with_context(|| {
            format!(
                "the processing program's module carries no {}",
                build_server::source::MODULE_WASM
            )
        })?;
    let mut bytes = Vec::with_capacity(usize::try_from(file.size).unwrap_or(0));
    (&file.file)
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading {}", file.name))?;
    Ok(Program::load(&bytes, revision.clone())?)
}
