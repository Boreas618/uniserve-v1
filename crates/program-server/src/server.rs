//! The Program Server's object model (design section 6.2) and its four
//! operations (design section 6.3).
//!
//! A [`ProgramServer`] keeps what is installed — one [`Package`] per
//! `(operator, backend, revision)`, each holding its manifest, its directory
//! and its [`Source`] — the module [`Store`] of section 5.3, the host's
//! [`Toolchains`] and the compile [`Queue`] of section 6.2. It serves
//! `install`, `prepare`, `resolve` and `list` on a Unix socket to any number
//! of clients: one thread per connection, one driver thread for the queue.
//!
//! The server compiles, keeps and provides. It launches no kernel and
//! allocates no device memory (section 6.1), and it never starts an update of
//! a deployment (section 7.2): a package installed here changes nothing until
//! a daemon asks for its revision.
//!
//! `prepare` is the daemon's call. It selects the program that fills each
//! operator — the latest installed revision of a backend, or the exact
//! program the daemon names — checks that each program covers every
//! descriptor the epoch needs, and commits every missing module: those builds
//! are `Background`, ahead-of-time work, and the call returns once they are
//! committed so that the daemon's later `resolve` calls are all store hits
//! (section 7.3, step 1). `resolve` is the main process's call: it returns a
//! module's files as descriptors, and a module that is not in the store yet
//! is built first as `Waiting` work, ahead of every `Background` build.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::Duration;

use crate::builder::{Build, BuildState, Builder, Limits, Urgency};
use crate::manifest::Manifest;
use crate::program::{Program, ProgramSlot};
use crate::protocol::{
    self, Chosen, Connection, Descriptors, Module, ModuleFile, PrepareRequest, Progress, Selector,
    wire,
};
use crate::source::Source;
use crate::store::{ModulePath, Root, Store};
use crate::target::{self, Device, Target};
use crate::toolchain::{ToolchainConfig, Toolchains};
use crate::{Error, Result};

pub use crate::protocol::{Installed, Listing};

/// How long the queue's driver waits before it looks at the host's memory
/// again while builds it cannot start are queued.
const DRIVER_TICK: Duration = Duration::from_millis(250);

/// One installed program package: its manifest, its directory and how its
/// sources become modules (design sections 5.2 and 6.2).
pub struct Package {
    pub manifest: Manifest,
    /// The package's directory: a copy of the installed package under the
    /// server's package root.
    pub root: PathBuf,
    pub source: Box<dyn Source>,
}

impl Package {
    /// The toolchain this package's sources compile with.
    pub fn toolchain(&self) -> crate::manifest::ToolchainKind {
        self.source.toolchain()
    }

    /// The directory an installed revision of a package lives in.
    fn revision_directory(package_root: &Path, op: &str, backend: &str, revision: u32) -> PathBuf {
        package_root
            .join(op)
            .join(backend)
            .join(revision.to_string())
    }
}

/// How to serve: what `uniserve program-server serve` takes.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// The socket to answer on.
    pub socket: PathBuf,
    /// The package root. It is writable on the upstream Program Server, which
    /// accepts `install`, and a read-only mount of the upstream's root on
    /// every other one (design section 6.2).
    pub package_root: PathBuf,
    /// The module store's roots: the first is writable, the others are
    /// shared read-only roots (design section 5.3).
    pub store_roots: Vec<Root>,
    /// The upstream Program Server's socket, on a server that is not the
    /// upstream: an `install` here is refused and names it (design section
    /// 6.3). `None` on the upstream itself.
    pub upstream: Option<PathBuf>,
    /// The queue's limits (design section 6.2).
    pub limits: Limits,
    /// Where this host's toolchains are.
    pub toolchains: ToolchainConfig,
    /// The devices this host builds for: their architectures and models
    /// (design section 5.3).
    pub devices: Vec<Device>,
}

/// A Program Server: what is installed, the module store, the toolchains and
/// the compile queue.
pub struct ProgramServer {
    socket: PathBuf,
    package_root: PathBuf,
    /// Whether this server accepts `install`, and where the upstream is.
    upstream: Option<PathBuf>,
    packages: RwLock<BTreeMap<(String, String, u32), Arc<Package>>>,
    store: Arc<Store>,
    toolchains: Arc<Toolchains>,
    devices: Vec<Device>,
    queue: Arc<Queue>,
}

impl ProgramServer {
    /// Opens a server over its configuration: loads the installed packages
    /// from the package root, discovers the toolchains and opens the store.
    /// The compile queue's driver starts with [`Self::serve`].
    pub fn open(config: &ServerConfig) -> Result<ProgramServer> {
        let store = Arc::new(Store::open(config.store_roots.clone())?);
        let toolchains = Arc::new(Toolchains::discover(&config.toolchains)?);
        let packages = load_packages(&config.package_root)?;
        let queue = Queue::new(Arc::clone(&store), Arc::clone(&toolchains), config.limits);
        Ok(ProgramServer {
            socket: config.socket.clone(),
            package_root: config.package_root.clone(),
            upstream: config.upstream.clone(),
            packages: RwLock::new(packages),
            store,
            toolchains,
            devices: config.devices.clone(),
            queue,
        })
    }

    /// The host's toolchains (design section 6.2).
    pub fn toolchains(&self) -> &Toolchains {
        &self.toolchains
    }

    /// Serves until the process ends: binds the socket, answers every client
    /// on its own thread, and drives the compile queue.
    ///
    /// # Errors
    ///
    /// Fails when the socket cannot be bound or a listener cannot be made.
    pub fn serve(self: Arc<Self>) -> Result<()> {
        Queue::drive(Arc::clone(&self.queue));
        let listener = bind(&self.socket)?;
        tracing::info!(
            "program server listening on {} with {} package(s)",
            self.socket.display(),
            self.packages
                .read()
                .map(|packages| packages.len())
                .unwrap_or(0)
        );
        for stream in listener.incoming() {
            let stream = match stream {
                Ok(stream) => stream,
                Err(error) => {
                    tracing::warn!("accepting a program server connection: {error}");
                    continue;
                }
            };
            let server = Arc::clone(&self);
            std::thread::spawn(move || {
                if let Err(error) = server.serve_connection(stream) {
                    tracing::debug!("a program server connection ended: {error}");
                }
            });
        }
        Ok(())
    }

    /// Answers one client's connection until it closes.
    fn serve_connection(&self, stream: std::os::unix::net::UnixStream) -> Result<()> {
        let mut connection = Connection::new(stream);
        loop {
            let (request, _) = connection.receive::<wire::Request>()?;
            let Some(request) = request.request else {
                return Err(Error::Protocol("a request carries no request".into()));
            };
            let answered = self.answer(&mut connection, request);
            if let Err(error) = answered {
                // The client's request failed; it is told why in the frame
                // that would have answered it.
                let refused = refusal(&error);
                connection.send(
                    &wire::ServerMessage {
                        message: Some(wire::server_message::Message::Refused(refused)),
                    },
                    &[],
                )?;
            }
        }
    }

    /// Answers one request, sending its answer on `connection`.
    fn answer(&self, connection: &mut Connection, request: wire::request::Request) -> Result<()> {
        match request {
            wire::request::Request::Install(install) => {
                let installed = self.install(Path::new(&install.package))?;
                connection.send(
                    &wire::ServerMessage {
                        message: Some(wire::server_message::Message::Installed(wire::Installed {
                            package: Some(wire::Package {
                                op: installed.package.op.clone(),
                                backend: installed.package.backend.clone(),
                                revision: installed.package.revision,
                            }),
                        })),
                    },
                    &[],
                )
            }
            wire::request::Request::Prepare(prepare) => {
                let request = prepare_request(prepare)?;
                let mut replies = Replies::new(connection);
                let prepared = self.prepare(&request, &mut |progress| {
                    replies.progress(progress);
                })?;
                replies.check()?;
                let prepared = wire::Prepared {
                    programs: prepared
                        .programs
                        .into_iter()
                        .map(|chosen| wire::Chosen {
                            op: chosen.slot.op.clone(),
                            phase: chosen
                                .slot
                                .phase
                                .map_or(wire::ProgramPhase::All, wire::ProgramPhase::from)
                                as i32,
                            program: Some((&chosen.program).into()),
                        })
                        .collect(),
                };
                connection.send(
                    &wire::ServerMessage {
                        message: Some(wire::server_message::Message::Prepared(prepared)),
                    },
                    &[],
                )
            }
            wire::request::Request::Resolve(resolve) => {
                let program = Program::try_from(
                    resolve
                        .program
                        .ok_or_else(|| Error::Protocol("a resolve names no program".into()))?,
                )?;
                let target = Target::new(resolve.target);
                let mut replies = Replies::new(connection);
                let module = self.resolve(
                    &resolve.op,
                    &program,
                    &target,
                    &resolve.descriptor,
                    &mut |progress| replies.progress(progress),
                )?;
                replies.check()?;
                let mut files = Vec::with_capacity(module.files.len());
                let mut descriptors = Vec::with_capacity(module.files.len());
                for file in &module.files {
                    files.push(wire::ModuleFile {
                        name: file.name.clone(),
                        size: file.size,
                    });
                }
                for file in module.files {
                    descriptors.push(std::os::fd::OwnedFd::from(file.file));
                }
                connection.send(
                    &wire::ServerMessage {
                        message: Some(wire::server_message::Message::Resolved(wire::Resolved {
                            form: wire::ModuleForm::from(module.form) as i32,
                            files,
                            // Field 3 is superseded by `entries` and left empty.
                            entry: String::new(),
                            entries: module.entries.into_iter().collect(),
                            metadata: module.metadata.unwrap_or_default(),
                        })),
                    },
                    &descriptors,
                )
            }
            wire::request::Request::List(_) => {
                let listing = self.list()?;
                let listing = wire::Listing {
                    packages: listing
                        .packages
                        .into_iter()
                        .map(|package| wire::ListedPackage {
                            op: package.op,
                            backend: package.backend,
                            revision: package.revision,
                            modules: package
                                .modules
                                .into_iter()
                                .map(|module| wire::ListedModule {
                                    target: module.target.to_string(),
                                    descriptor: module.descriptor,
                                    form: wire::ModuleForm::from(module.form) as i32,
                                })
                                .collect(),
                        })
                        .collect(),
                };
                connection.send(
                    &wire::ServerMessage {
                        message: Some(wire::server_message::Message::Listing(listing)),
                    },
                    &[],
                )
            }
        }
    }

    /// Installs a program package and answers its revision (design section
    /// 6.3).
    ///
    /// Only the upstream Program Server installs; every other one refuses and
    /// names the upstream. A revision fixes the package's sources, tuning data
    /// and toolchain versions, so an installed revision is never overwritten:
    /// each installation of the same operator and backend is the next
    /// revision (design section 5.1).
    pub fn install(&self, directory: &Path) -> Result<protocol::Installed> {
        if let Some(upstream) = &self.upstream {
            return Err(Error::Refused {
                reason: format!(
                    "only the upstream Program Server installs packages, and this is not it: \
                     the package was not installed from {}",
                    directory.display()
                ),
                upstream: Some(upstream.display().to_string()),
            });
        }
        let manifest = Manifest::load(directory).map_err(|source| Error::Manifest {
            package: directory.to_path_buf(),
            source: Box::new(source),
        })?;
        let source = manifest.source(directory)?;
        let mut packages = self
            .packages
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let revision = packages
            .keys()
            .filter(|(op, backend, _)| op == &manifest.op && backend == &manifest.backend)
            .map(|(_, _, revision)| *revision)
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| Error::Refused {
                reason: format!("{} {} has no revision left", manifest.op, manifest.backend),
                upstream: None,
            })?;
        let root = Package::revision_directory(
            &self.package_root,
            &manifest.op,
            &manifest.backend,
            revision,
        );
        install_package(directory, &root)?;
        let package = Arc::new(Package {
            manifest,
            root,
            source,
        });
        let (op, backend) = (
            package.manifest.op.clone(),
            package.manifest.backend.clone(),
        );
        packages.insert((op.clone(), backend.clone(), revision), package);
        Ok(protocol::Installed {
            package: protocol::Package {
                op,
                backend,
                revision,
            },
        })
    }

    /// Selects the programs of an epoch's operators and commits every module
    /// their descriptors need that the store misses (design section 6.3).
    ///
    /// Each build is reported through `progress` as it is queued and as it
    /// ends; the call returns the selected programs once every module is
    /// committed, and fails with the first build that failed.
    pub fn prepare(
        &self,
        request: &PrepareRequest,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<protocol::Prepared> {
        let mut chosen = Vec::with_capacity(request.programs.len());
        for entry in &request.programs {
            let program = self.select(&entry.slot.op, &entry.selector)?;
            chosen.push(Chosen {
                slot: entry.slot.clone(),
                program,
            });
        }

        // Asking for the descriptors of an operator no program fills would
        // silently leave the epoch without its modules; it is the request's
        // inconsistency, not a missing build.
        for descriptors in &request.descriptors {
            if !chosen.iter().any(|chosen| chosen.slot.op == descriptors.op) {
                return Err(Error::Protocol(format!(
                    "prepare asks for the descriptors of {} and no program fills it",
                    descriptors.op
                )));
            }
        }

        let mut builds: Vec<(ModulePath, Receiver<BuildState>)> = Vec::new();
        let mut failures: Vec<Error> = Vec::new();
        for chosen in &chosen {
            let package = self.package(&chosen.slot.op, &chosen.program)?;
            let mut keys = Vec::new();
            for descriptors in request
                .descriptors
                .iter()
                .filter(|descriptors| descriptors.op == chosen.slot.op)
            {
                for key in &descriptors.keys {
                    if !keys.contains(key) {
                        keys.push(key.clone());
                    }
                }
            }
            for key in keys {
                if !package.manifest.covers(&key) {
                    return Err(Error::UnsupportedDescriptor {
                        op: chosen.slot.op.clone(),
                        backend: chosen.program.backend.clone(),
                        revision: chosen.program.revision,
                        descriptor: key,
                    });
                }
                for target in self.targets(&package, &request.targets)? {
                    let path = ModulePath::new(
                        chosen.slot.op.clone(),
                        chosen.program.backend.clone(),
                        chosen.program.revision,
                        target.to_string(),
                        key.clone(),
                    )?;
                    if self.store.contains(&path) {
                        progress(Progress {
                            path,
                            state: BuildState::Committed,
                            detail: String::new(),
                        });
                        continue;
                    }
                    if let Some(BuildState::Failed(detail)) = self.queue.state(&path) {
                        // A build that failed stays failed: `prepare` reports
                        // it and does not retry it, so a broken package cannot
                        // spin.
                        progress(Progress {
                            path: path.clone(),
                            state: BuildState::Failed(detail.clone()),
                            detail: detail.clone(),
                        });
                        failures.push(Error::Build {
                            op: path.op.clone(),
                            backend: path.backend.clone(),
                            revision: path.revision,
                            target: path.target.clone(),
                            descriptor: path.descriptor.clone(),
                            message: detail,
                        });
                        continue;
                    }
                    let receiver =
                        self.queue
                            .submit(Arc::clone(&package), &path, Urgency::Background);
                    builds.push((path, receiver));
                }
            }
        }

        for (path, receiver) in builds {
            let state = receiver.recv().unwrap_or(BuildState::Failed(
                "the build's queue ended without an answer".into(),
            ));
            progress(Progress {
                path: path.clone(),
                state: state.clone(),
                detail: match &state {
                    BuildState::Failed(detail) => detail.clone(),
                    _ => String::new(),
                },
            });
            if let BuildState::Failed(detail) = state {
                failures.push(Error::Build {
                    op: path.op.clone(),
                    backend: path.backend.clone(),
                    revision: path.revision,
                    target: path.target.clone(),
                    descriptor: path.descriptor.clone(),
                    message: detail,
                });
            }
        }
        if let Some(failure) = failures.into_iter().next() {
            return Err(failure);
        }
        Ok(protocol::Prepared { programs: chosen })
    }

    /// Resolves one module, building it first when the store misses it
    /// (design section 6.3).
    ///
    /// A missing module is `Waiting` work, ahead of every `Background` build
    /// queued for `prepare`, and the call returns after it is committed.
    pub fn resolve(
        &self,
        op: &str,
        program: &Program,
        target: &Target,
        descriptor: &str,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Module> {
        let package = self.package(op, program)?;
        if !package.manifest.builds_for(target.as_str()) {
            return Err(Error::Build {
                op: op.to_owned(),
                backend: program.backend.clone(),
                revision: program.revision,
                target: target.to_string(),
                descriptor: descriptor.to_owned(),
                message: format!(
                    "the package builds for {}",
                    if package.manifest.kernel.is_some() {
                        format!(
                            "{:?}",
                            package
                                .manifest
                                .kernel
                                .as_ref()
                                .map(|kernel| &kernel.architectures)
                        )
                    } else {
                        crate::manifest::WASM_TARGET.to_owned()
                    }
                ),
            });
        }
        if !package.manifest.covers(descriptor) {
            return Err(Error::UnsupportedDescriptor {
                op: op.to_owned(),
                backend: program.backend.clone(),
                revision: program.revision,
                descriptor: descriptor.to_owned(),
            });
        }
        let path = ModulePath::new(
            op,
            program.backend.clone(),
            program.revision,
            target.to_string(),
            descriptor,
        )?;
        if !self.store.contains(&path) {
            // A failed build is retried by the resolve that needs it: a
            // transient failure (host memory, a lost file) is worth another
            // attempt, and the package's revision is what a real change
            // would replace.
            let receiver = self
                .queue
                .submit(Arc::clone(&package), &path, Urgency::Waiting);
            let state = receiver.recv().unwrap_or(BuildState::Failed(
                "the build's queue ended without an answer".into(),
            ));
            progress(Progress {
                path: path.clone(),
                state: state.clone(),
                detail: match &state {
                    BuildState::Failed(detail) => detail.clone(),
                    _ => String::new(),
                },
            });
            if let BuildState::Failed(detail) = state {
                return Err(Error::Build {
                    op: path.op.clone(),
                    backend: path.backend.clone(),
                    revision: path.revision,
                    target: path.target.clone(),
                    descriptor: path.descriptor.clone(),
                    message: detail,
                });
            }
        }
        let files = self.store.files(&path)?.ok_or_else(|| Error::Build {
            op: path.op.clone(),
            backend: path.backend.clone(),
            revision: path.revision,
            target: path.target.clone(),
            descriptor: path.descriptor.clone(),
            message: "the module is not in the store after its build was committed".into(),
        })?;
        let mut module_files = Vec::with_capacity(files.len());
        for file in files {
            let handle = fs::File::open(&file.path)?;
            let size = handle.metadata()?.len();
            module_files.push(ModuleFile {
                name: file.name,
                size,
                file: handle,
            });
        }
        Ok(Module {
            form: package.manifest.module.form,
            files: module_files,
            metadata: package.manifest.module.metadata.clone(),
            entries: package.manifest.module.entries.clone(),
        })
    }

    /// Lists the installed packages, their revisions and their modules.
    pub fn list(&self) -> Result<protocol::Listing> {
        let packages = self
            .packages
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut listed: BTreeMap<(String, String, u32), protocol::ListedPackage> = packages
            .iter()
            .map(|((op, backend, revision), _package)| {
                (
                    (op.clone(), backend.clone(), *revision),
                    protocol::ListedPackage {
                        op: op.clone(),
                        backend: backend.clone(),
                        revision: *revision,
                        modules: Vec::new(),
                    },
                )
            })
            .collect();
        for path in self.store.list()? {
            let Some(entry) =
                listed.get_mut(&(path.op.clone(), path.backend.clone(), path.revision))
            else {
                continue;
            };
            let form = packages
                .get(&(path.op.clone(), path.backend.clone(), path.revision))
                .map_or(crate::manifest::ModuleForm::Cubin, |package| {
                    package.manifest.module.form
                });
            entry.modules.push(protocol::ListedModule {
                target: Target::new(path.target),
                descriptor: path.descriptor,
                form,
            });
        }
        Ok(protocol::Listing {
            packages: listed.into_values().collect(),
        })
    }

    /// The program a `prepare` names for an operator, with the package's
    /// parameter defaults filled in for every argument the request leaves out
    /// (design section 5.2): what the daemon records is the program the
    /// package will run, not a partial one.
    fn select(&self, op: &str, selector: &Selector) -> Result<Program> {
        let program = match selector {
            Selector::Fixed(program) => {
                self.package(op, program)?;
                program.clone()
            }
            Selector::Latest { backend, arguments } => {
                let packages = self
                    .packages
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let revision = packages
                    .keys()
                    .filter(|(package_op, package_backend, _)| {
                        package_op == op && package_backend == backend
                    })
                    .map(|(_, _, revision)| *revision)
                    .max()
                    .ok_or_else(|| Error::NotInstalled {
                        op: op.to_owned(),
                        backend: backend.clone(),
                    })?;
                Program {
                    backend: backend.clone(),
                    revision,
                    arguments: arguments.clone(),
                }
            }
        };
        let package = self.package(op, &program)?;
        let mut arguments = toml::Table::new();
        for (name, default) in &package.manifest.parameters {
            arguments.insert(name.clone(), default.clone());
        }
        for (name, value) in &program.arguments {
            arguments.insert(name.clone(), value.clone());
        }
        Ok(Program {
            arguments,
            ..program
        })
    }

    /// The installed package a program names.
    fn package(&self, op: &str, program: &Program) -> Result<Arc<Package>> {
        self.packages
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&(op.to_owned(), program.backend.clone(), program.revision))
            .cloned()
            .ok_or_else(|| Error::UnknownRevision {
                op: op.to_owned(),
                backend: program.backend.clone(),
                revision: program.revision,
            })
    }

    /// The targets a package is built for: those the request names, checked
    /// against this host, or every target of the host that the package
    /// supports (design section 5.3). A scheduling policy is built once, for
    /// wasm32, whatever the host's GPUs are.
    fn targets(&self, package: &Package, requested: &[Target]) -> Result<Vec<Target>> {
        let wasm = Target::new(crate::manifest::WASM_TARGET);
        if package.manifest.kernel.is_none() {
            if !requested.is_empty()
                && !requested
                    .iter()
                    .any(|target| target.as_str() == crate::manifest::WASM_TARGET)
            {
                return Err(Error::Build {
                    op: package.manifest.op.clone(),
                    backend: package.manifest.backend.clone(),
                    revision: 0,
                    target: requested
                        .iter()
                        .map(Target::to_string)
                        .collect::<Vec<_>>()
                        .join(", "),
                    descriptor: String::new(),
                    message: "a scheduling policy is built for wasm32 alone".into(),
                });
            }
            return Ok(vec![wasm]);
        }
        let kind = package.manifest.target_kind();
        let host = target::targets(&self.devices, kind);
        let wanted = if requested.is_empty() {
            host.clone()
        } else {
            target::check(&self.devices, kind, requested)?;
            requested.to_vec()
        };
        let mut selected = Vec::new();
        for wanted in wanted {
            if package.manifest.builds_for(wanted.as_str()) {
                selected.push(wanted);
            } else if !requested.is_empty() {
                // A target the epoch names and the package cannot build for
                // is a missing module the daemon must be told about, not a
                // target to skip.
                return Err(Error::Build {
                    op: package.manifest.op.clone(),
                    backend: package.manifest.backend.clone(),
                    revision: 0,
                    target: wanted.to_string(),
                    descriptor: String::new(),
                    message: format!(
                        "the package builds for {:?}",
                        package
                            .manifest
                            .kernel
                            .as_ref()
                            .map(|kernel| &kernel.architectures)
                    ),
                });
            }
        }
        if selected.is_empty() {
            return Err(Error::Build {
                op: package.manifest.op.clone(),
                backend: package.manifest.backend.clone(),
                revision: 0,
                target: String::new(),
                descriptor: String::new(),
                message: format!(
                    "the package builds for {:?} and this host builds for {:?}",
                    package
                        .manifest
                        .kernel
                        .as_ref()
                        .map(|kernel| &kernel.architectures),
                    host
                ),
            });
        }
        Ok(selected)
    }
}

/// The queue's side of a build: what is being built, and who waits for it.
struct Queue {
    store: Arc<Store>,
    toolchains: Arc<Toolchains>,
    state: Mutex<QueueState>,
    wake: Condvar,
}

struct QueueState {
    builder: Builder,
    /// The build of each path that is queued or running: what to compile.
    jobs: HashMap<ModulePath, Arc<BuildJob>>,
    /// One sender per waiting request, by the waiter's identity.
    waiters: HashMap<u64, mpsc::Sender<BuildState>>,
    next_waiter: u64,
    stopping: bool,
}

/// One queued build: the package it builds and the module path.
struct BuildJob {
    package: Arc<Package>,
    path: ModulePath,
}

impl Queue {
    fn new(store: Arc<Store>, toolchains: Arc<Toolchains>, limits: Limits) -> Arc<Queue> {
        Arc::new(Queue {
            store,
            toolchains,
            state: Mutex::new(QueueState {
                builder: Builder::new(limits),
                jobs: HashMap::new(),
                waiters: HashMap::new(),
                next_waiter: 0,
                stopping: false,
            }),
            wake: Condvar::new(),
        })
    }

    /// Submits a build for a module path, or joins the one queued or running
    /// for it and raises its urgency. Answers when the build ends.
    fn submit(
        &self,
        package: Arc<Package>,
        path: &ModulePath,
        urgency: Urgency,
    ) -> Receiver<BuildState> {
        let (sender, receiver) = mpsc::channel();
        let mut state = self.lock();
        if self.store.contains(path)
            || matches!(state.builder.state(path), Some(BuildState::Committed))
        {
            let _ = sender.send(BuildState::Committed);
            return receiver;
        }
        let waiter = state.next_waiter;
        state.next_waiter += 1;
        state.builder.submit(path, urgency, waiter);
        state.waiters.insert(waiter, sender);
        state.jobs.entry(path.clone()).or_insert_with(|| {
            Arc::new(BuildJob {
                package,
                path: path.clone(),
            })
        });
        drop(state);
        self.wake.notify_all();
        receiver
    }

    /// What is known about a path: committed by the store, or the state of
    /// its build.
    fn state(&self, path: &ModulePath) -> Option<BuildState> {
        if self.store.contains(path) {
            return Some(BuildState::Committed);
        }
        self.lock().builder.state(path)
    }

    /// Drives the queue until the server ends: starts every build the limits
    /// allow, and waits for memory or for a build to end.
    fn drive(self: Arc<Self>) {
        loop {
            let mut state = self.lock();
            if state.stopping {
                return;
            }
            // Without `MemAvailable` the host cannot be gated on memory; the
            // job limit still applies.
            let framed = available_host_memory().unwrap_or(u64::MAX);
            let QueueState { builder, jobs, .. } = &mut *state;
            let started = builder.start(framed, |build| {
                jobs.get(&build.path)
                    .map_or(0, |job| peak(&self.toolchains, job))
            });
            if started.is_empty() {
                let timeout = if builder.pending() == 0 {
                    // Nothing queued: wait until something is submitted.
                    DRIVER_TICK * 8
                } else {
                    DRIVER_TICK
                };
                let _ = self
                    .wake
                    .wait_timeout(state, timeout)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                continue;
            }
            let mut spawned = Vec::with_capacity(started.len());
            for path in started {
                if let Some(job) = jobs.get(&path).cloned() {
                    spawned.push(job);
                }
            }
            drop(state);
            for job in spawned {
                let queue = Arc::clone(&self);
                std::thread::spawn(move || queue.build(job));
            }
        }
    }

    /// Runs one build and tells its waiters, and the driver, how it ended.
    ///
    /// A build that panics fails like one that returned an error: the
    /// failure is the build's log, and the queue keeps its slot.
    fn build(self: Arc<Self>, job: Arc<BuildJob>) {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.compile(&job)))
            .unwrap_or_else(|panic| {
                let message = panic
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_owned())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "the build panicked".to_owned());
                Err(Error::Build {
                    op: job.path.op.clone(),
                    backend: job.path.backend.clone(),
                    revision: job.path.revision,
                    target: job.path.target.clone(),
                    descriptor: job.path.descriptor.clone(),
                    message: format!("the build panicked: {message}"),
                })
            })
            .map_err(|error| error.to_string());
        let mut state = self.lock();
        let ended: Option<Build> = state.builder.finish(&job.path, outcome);
        state.jobs.remove(&job.path);
        if let Some(build) = ended {
            for waiter in &build.waiters {
                if let Some(sender) = state.waiters.remove(waiter) {
                    let _ = sender.send(build.state.clone());
                }
            }
        }
        drop(state);
        self.wake.notify_all();
    }

    /// Compiles one module into the store.
    fn compile(&self, job: &BuildJob) -> Result<()> {
        let package = &job.package;
        let kind = package.toolchain();
        let toolchain = self.toolchains.get(kind)?;
        if toolchain.version() != package.manifest.toolchain.version {
            return Err(Error::Toolchain {
                kind,
                message: format!(
                    "the package {} {} revision {} is written for {kind} {} and this host has {}",
                    package.manifest.op,
                    package.manifest.backend,
                    job.path.revision,
                    package.manifest.toolchain.version,
                    toolchain.version()
                ),
            });
        }
        let workspace = crate::source::Workspace::new()?;
        let target = Target::new(job.path.target.clone());
        let built = package.source.build(
            &job.path.op,
            &job.path.descriptor,
            &target,
            toolchain,
            &workspace,
        )?;
        if built.files.is_empty() {
            return Err(Error::Build {
                op: job.path.op.clone(),
                backend: job.path.backend.clone(),
                revision: job.path.revision,
                target: job.path.target.clone(),
                descriptor: job.path.descriptor.clone(),
                message: "the build wrote no file".into(),
            });
        }
        self.store
            .publish(&job.path, workspace.path(), &built.files)?;
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, QueueState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// The peak host memory one build reserves, as its toolchain reports it.
fn peak(toolchains: &Toolchains, job: &BuildJob) -> u64 {
    let kind = job.package.toolchain();
    toolchains
        .get(kind)
        .map(|toolchain| toolchain.peak_host_bytes(&job.path.op, &job.path.descriptor))
        .unwrap_or(0)
}

/// The answers a request sends on its connection while it runs.
struct Replies<'a> {
    connection: &'a mut Connection,
    failure: Option<Error>,
}

impl<'a> Replies<'a> {
    fn new(connection: &'a mut Connection) -> Self {
        Self {
            connection,
            failure: None,
        }
    }

    fn progress(&mut self, progress: Progress) {
        if self.failure.is_some() {
            return;
        }
        let message = wire::ServerMessage {
            message: Some(wire::server_message::Message::Progress(wire::Progress {
                op: progress.path.op.clone(),
                backend: progress.path.backend.clone(),
                revision: progress.path.revision,
                target: progress.path.target.clone(),
                descriptor: progress.path.descriptor.clone(),
                state: wire::BuildState::from(&progress.state) as i32,
                detail: progress.detail.clone(),
            })),
        };
        if let Err(error) = self.connection.send(&message, &[]) {
            self.failure = Some(error);
        }
    }

    fn check(&mut self) -> Result<()> {
        match self.failure.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// The error a client is told about a failed request.
fn refusal(error: &Error) -> wire::Refused {
    match error {
        Error::Refused { reason, upstream } => wire::Refused {
            reason: reason.clone(),
            upstream: upstream.clone().unwrap_or_default(),
        },
        other => wire::Refused {
            reason: other.to_string(),
            upstream: String::new(),
        },
    }
}

/// Loads every installed package under a package root.
fn load_packages(root: &Path) -> Result<BTreeMap<(String, String, u32), Arc<Package>>> {
    let mut packages = BTreeMap::new();
    for op in directories(root)? {
        let op_name = file_name(&op);
        for backend in directories(&op)? {
            let backend_name = file_name(&backend);
            for revision in directories(&backend)? {
                let Some(number) = revision
                    .file_name()
                    .and_then(|name| name.to_str())
                    .and_then(|name| name.parse::<u32>().ok())
                else {
                    continue;
                };
                if !revision.join("package.toml").is_file() {
                    continue;
                }
                let manifest = Manifest::load(&revision).map_err(|source| Error::Manifest {
                    package: revision.clone(),
                    source: Box::new(source),
                })?;
                if manifest.op != op_name || manifest.backend != backend_name {
                    return Err(Error::Manifest {
                        package: revision.clone(),
                        source: Box::new(crate::manifest::ManifestError::Invalid(format!(
                            "the package is installed as {op_name}/{backend_name} and its manifest \
                             says {}/{}",
                            manifest.op, manifest.backend
                        ))),
                    });
                }
                let source = manifest.source(&revision)?;
                packages.insert(
                    (op_name.clone(), backend_name.clone(), number),
                    Arc::new(Package {
                        manifest,
                        root: revision.clone(),
                        source,
                    }),
                );
            }
        }
    }
    Ok(packages)
}

/// Copies a package into its revision's directory, atomically: a crash leaves
/// no directory a later scan would mistake for an installed revision.
fn install_package(directory: &Path, destination: &Path) -> Result<()> {
    let parent = destination.parent().ok_or_else(|| {
        Error::Io(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} has no parent directory", destination.display()),
        ))
    })?;
    fs::create_dir_all(parent)?;
    let staged = parent.join(format!(
        ".{}.staged-{}",
        destination
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default(),
        std::process::id()
    ));
    if staged.exists() {
        fs::remove_dir_all(&staged)?;
    }
    fs::create_dir_all(&staged)?;
    copy_tree(directory, &staged)?;
    if let Ok(handle) = fs::File::open(&staged) {
        handle.sync_all()?;
    }
    fs::rename(&staged, destination)?;
    if let Ok(handle) = fs::File::open(parent) {
        handle.sync_all()?;
    }
    Ok(())
}

/// Copies a directory tree, keeping relative paths and file modes.
fn copy_tree(from: &Path, to: &Path) -> Result<()> {
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let source = entry.path();
        let destination = to.join(entry.file_name());
        let kind = entry.file_type()?;
        if kind.is_dir() {
            fs::create_dir_all(&destination)?;
            copy_tree(&source, &destination)?;
        } else if kind.is_symlink() {
            let target = fs::read_link(&source)?;
            std::os::unix::fs::symlink(target, &destination)?;
        } else {
            fs::copy(&source, &destination)?;
            fs::File::open(&destination)?.sync_all()?;
        }
    }
    Ok(())
}

/// Binds the server's socket. A socket no process answers on is left from a
/// server that ended, and is replaced.
fn bind(socket: &Path) -> Result<std::os::unix::net::UnixListener> {
    if let Some(parent) = socket.parent() {
        fs::create_dir_all(parent)?;
    }
    match std::os::unix::net::UnixListener::bind(socket) {
        Ok(listener) => Ok(listener),
        Err(error) if error.kind() == io::ErrorKind::AddrInUse => {
            // A Program Server that still answers is not replaced.
            if std::os::unix::net::UnixStream::connect(socket).is_ok() {
                return Err(Error::Io(io::Error::new(
                    io::ErrorKind::AddrInUse,
                    format!("a program server answers on {}", socket.display()),
                )));
            }
            fs::remove_file(socket)?;
            Ok(std::os::unix::net::UnixListener::bind(socket)?)
        }
        Err(error) => Err(Error::Io(error)),
    }
}

/// The descriptors a `prepare` asks for, as the server's request.
fn prepare_request(message: wire::Prepare) -> Result<PrepareRequest> {
    let mut programs = Vec::with_capacity(message.programs.len());
    for program in message.programs {
        let slot = ProgramSlot {
            op: program.op,
            phase: Option::<crate::program::PhaseKind>::try_from(
                wire::ProgramPhase::try_from(program.phase)
                    .map_err(|_| Error::Protocol("unknown program phase".into()))?,
            )?,
        };
        let selector = match program.selector {
            Some(wire::program_request::Selector::Backend(backend)) => Selector::Latest {
                backend: backend.backend,
                arguments: arguments(&backend.arguments)?,
            },
            Some(wire::program_request::Selector::Program(program)) => {
                Selector::Fixed(Program::try_from(program)?)
            }
            None => {
                return Err(Error::Protocol(
                    "a program request names no backend or program".into(),
                ));
            }
        };
        programs.push(crate::protocol::ProgramRequest { slot, selector });
    }
    let descriptors = message
        .descriptors
        .into_iter()
        .map(|descriptors| Descriptors {
            op: descriptors.op,
            keys: descriptors.keys,
        })
        .collect();
    Ok(PrepareRequest {
        targets: message.targets.into_iter().map(Target::new).collect(),
        programs,
        descriptors,
    })
}

/// A program's arguments, which travel as a TOML table.
fn arguments(text: &str) -> Result<toml::Table> {
    if text.is_empty() {
        return Ok(toml::Table::new());
    }
    toml::from_str(text)
        .map_err(|error| Error::Protocol(format!("a program's arguments are not TOML: {error}")))
}

/// The host storage available to new work without swapping, in bytes:
/// `MemAvailable` of `/proc/meminfo`. `None` when the host states none, in
/// which case builds are not gated on memory.
fn available_host_memory() -> Option<u64> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|value| value.trim().strip_suffix("kB"))
        .and_then(|kilobytes| kilobytes.trim().parse::<u64>().ok())
        .map(|kilobytes| kilobytes << 10)
}

/// The subdirectories of a directory, in name order.
fn directories(path: &Path) -> Result<Vec<PathBuf>> {
    let mut directories = Vec::new();
    if !path.is_dir() {
        return Ok(directories);
    }
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            directories.push(entry.path());
        }
    }
    directories.sort();
    Ok(directories)
}

/// A path's last name.
fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}
