//! The Build Server's object model (design section 6.2) and its
//! operations (design section 6.3).
//!
//! A [`BuildServer`] keeps what is installed — one [`Package`] per
//! `(operator, backend, revision)`, each holding its manifest, its directory
//! and its [`Source`] — the module [`Store`] of section 5.3, the host's
//! [`Toolchains`] and the compile [`Queue`] of section 6.2. It serves
//! `install`, `prepare`, `resolve`, `list` and `cover` on a Unix socket to any
//! number of clients: one thread per connection, one driver thread for the
//! queue.
//!
//! The server compiles, keeps and provides. It launches no kernel and
//! allocates no device memory (section 6.1), and it never starts an update of
//! a deployment (section 7.2): a package installed here changes nothing until
//! a daemon asks for its revision.
//!
//! `prepare` is the daemon's call. It selects the package revision that fills each
//! operator — the latest installed revision of a backend, or the exact
//! package revision the daemon names — checks that each package revision covers every
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
use crate::inputs::Inputs;
use crate::manifest::Manifest;
use service::Connection;

use crate::protocol::{
    self, Descriptors, Module, ModuleFile, PrepareRequest, Progress, Selector, wire,
};
use crate::source::Source;
use crate::store::{ModulePath, Root, Store};
use crate::target::{self, Device, Target};
use crate::toolchain::{ToolchainConfig, Toolchains};
use crate::{Error, Result};
use proto::module::{ModuleSlot, PackageRevision};

/// How long the queue's driver waits before it looks at the host's memory
/// again while builds it cannot start are queued.
const DRIVER_TICK: Duration = Duration::from_millis(250);

/// One installed package: its manifest, its directory and how its
/// sources become modules (design sections 5.2 and 6.2).
pub struct Package {
    pub manifest: Manifest,
    /// The package's directory: a copy of the installed package under the
    /// server's package root.
    pub root: PathBuf,
    pub source: Box<dyn Source>,
    /// What the revision's modules are built from, as its installation
    /// recorded it; its digest is in every module path of the revision.
    pub inputs: Inputs,
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

/// How to serve: what `uniserve build-server serve` takes.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// The socket to serve on.
    pub socket: PathBuf,
    /// The package root. It is writable on the upstream Build Server, which
    /// accepts `install`, and a read-only mount of the upstream's root on
    /// every other one (design section 6.2).
    pub package_root: PathBuf,
    /// The module store's roots: the first is writable, the others are
    /// shared read-only roots (design section 5.3).
    pub store_roots: Vec<Root>,
    /// The upstream Build Server's socket, on a server that is not the
    /// upstream: an `install` here is refused and names it (design section
    /// 6.3). `None` on the upstream itself.
    pub upstream: Option<PathBuf>,
    /// Stores this server serves and never writes, each a directory a store
    /// build made (`packages` and `modules`, [`crate::build`]): the one a
    /// release ships with the modules of its own packages. Their revisions
    /// are served beside the package root's and their modules are found
    /// beside the store roots', so a host serves them with no toolchain and
    /// builds nothing.
    pub read_only_stores: Vec<PathBuf>,
    /// The queue's limits (design section 6.2).
    pub limits: Limits,
    /// Where this host's toolchains are.
    pub toolchains: ToolchainConfig,
    /// The devices this host builds for: their architectures and models
    /// (design section 5.3).
    pub devices: Vec<Device>,
}

/// A Build Server: what is installed, the module store, the toolchains and
/// the compile queue.
pub struct BuildServer {
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

impl BuildServer {
    /// Opens a server over its configuration: loads the installed packages
    /// from the package root, discovers the toolchains and opens the store.
    /// The compile queue's driver starts with [`Self::serve`].
    pub fn open(config: &ServerConfig) -> Result<BuildServer> {
        let mut roots = config.store_roots.clone();
        let mut packages = load_packages(&config.package_root)?;
        for store in &config.read_only_stores {
            roots.push(Root {
                path: store.join(crate::build::MODULES),
                writable: false,
            });
            for (revision, package) in load_packages(&store.join(crate::build::PACKAGES))? {
                // A revision this host installed holds its number: installs
                // number after every revision the server serves, so the two
                // meet only when a store is added under revisions installed
                // before it, and the host's own is the one its records name.
                if packages.contains_key(&revision) {
                    tracing::warn!(
                        store = %store.display(),
                        op = %revision.0,
                        backend = %revision.1,
                        revision = revision.2,
                        "a read-only store's revision is hidden by one of the package root"
                    );
                    continue;
                }
                packages.insert(revision, package);
            }
        }
        let store = Arc::new(Store::open(roots)?);
        let toolchains = Arc::new(Toolchains::discover(&config.toolchains)?);
        let queue = Queue::new(Arc::clone(&store), Arc::clone(&toolchains), config.limits);
        Ok(BuildServer {
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

    /// Serves until the process ends: binds the socket, serves every client
    /// on its own thread, and drives the compile queue.
    ///
    /// # Errors
    ///
    /// Fails when the socket cannot be bound or a listener cannot be made.
    pub fn serve(self: Arc<Self>) -> Result<()> {
        self.serve_until(Arc::new(std::sync::atomic::AtomicBool::new(false)))
    }

    /// Serves until `stop` is set: what [`Self::serve`] does, for a supervisor
    /// that must shut the server down. The socket is removed on the way out.
    ///
    /// # Errors
    ///
    /// Fails when another server owns the socket, the socket cannot be bound
    /// or a listener cannot be made.
    pub fn serve_until(self: Arc<Self>, stop: Arc<std::sync::atomic::AtomicBool>) -> Result<()> {
        let Some((listener, lock)) = service::bind(&self.socket)? else {
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::AddrInUse,
                format!("a build server serves on {}", self.socket.display()),
            )));
        };
        self.serve_on(listener, lock, stop)
    }

    /// Serves on `listener`, the socket `lock` makes this server the owner of
    /// ([`service::bind`]), until `stop` is set; the lock removes the socket
    /// on the way out.
    ///
    /// # Errors
    ///
    /// Fails when the listener cannot be made non-blocking.
    pub fn serve_on(
        self: Arc<Self>,
        listener: std::os::unix::net::UnixListener,
        lock: service::SocketLock,
        stop: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<()> {
        // The queue's driver runs beside the listener: it starts the builds
        // the request queue holds and wakes the requests waiting on them.
        std::thread::spawn({
            let queue = Arc::clone(&self.queue);
            move || Queue::drive(queue)
        });
        listener.set_nonblocking(true)?;
        tracing::info!(
            "build server listening on {} with {} package(s)",
            self.socket.display(),
            self.packages
                .read()
                .map(|packages| packages.len())
                .unwrap_or(0)
        );
        while !stop.load(std::sync::atomic::Ordering::SeqCst) {
            let stream = match listener.accept() {
                Ok((stream, _)) => stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    // A supervisor can shut the server down between two
                    // accept attempts; a client's connection is still handled
                    // by its own thread, so this wait costs it nothing.
                    std::thread::sleep(ACCEPT_TICK);
                    continue;
                }
                Err(error) => {
                    tracing::warn!("accepting a build server connection: {error}");
                    std::thread::sleep(ACCEPT_TICK);
                    continue;
                }
            };
            if stream.set_nonblocking(false).is_err() {
                continue;
            }
            let server = Arc::clone(&self);
            std::thread::spawn(move || {
                if let Err(error) = server.serve_connection(stream) {
                    tracing::debug!("a build server connection ended: {error}");
                }
            });
        }
        self.queue.stop();
        drop(lock);
        Ok(())
    }

    /// Serves one client's connection until it closes.
    fn serve_connection(&self, stream: std::os::unix::net::UnixStream) -> Result<()> {
        let mut connection = Connection::new(stream)?;
        loop {
            let request = connection
                .recv::<wire::Request>(std::time::Duration::MAX)?
                .ok_or_else(|| Error::Protocol("the client sent no request".into()))?;
            let Some(request) = request.request else {
                return Err(Error::Protocol("a request carries no request".into()));
            };
            let responded = self.respond(&mut connection, request);
            if let Err(error) = responded {
                // The client's request failed; it is told why in the frame
                // that would have returned it.
                let refused = refusal(&error);
                connection.deliver(&wire::ServerMessage {
                    message: Some(wire::server_message::Message::Refused(refused)),
                })?;
            }
        }
    }

    /// Responds to one request, sending its response on `connection`.
    fn respond(&self, connection: &mut Connection, request: wire::request::Request) -> Result<()> {
        match request {
            wire::request::Request::Install(install) => {
                let installed = self.install(Path::new(&install.package))?;
                Ok(connection.deliver(&wire::ServerMessage {
                    message: Some(wire::server_message::Message::Installed(
                        wire::InstallResponse {
                            package: Some(wire::Package {
                                op: installed.op,
                                backend: installed.backend,
                                revision: installed.revision,
                            }),
                        },
                    )),
                })?)
            }
            wire::request::Request::Prepare(prepare) => {
                let request = prepare_request(prepare)?;
                let mut sender = ProgressSender::new(connection);
                let prepared = self.prepare(&request, &mut |progress| {
                    sender.progress(progress);
                })?;
                sender.check()?;
                let prepared = wire::PrepareResponse {
                    packages: prepared
                        .into_iter()
                        .map(|(slot, package)| wire::SlotPackage {
                            op: slot.op,
                            phase: protocol::phase_to_wire(slot.phase),
                            package: Some((&package).into()),
                            part: slot.part.unwrap_or_default(),
                        })
                        .collect(),
                };
                Ok(connection.deliver(&wire::ServerMessage {
                    message: Some(wire::server_message::Message::Prepared(prepared)),
                })?)
            }
            wire::request::Request::Resolve(resolve) => {
                let revision = PackageRevision::try_from(
                    resolve
                        .package
                        .ok_or_else(|| Error::Protocol("a resolve names no package".into()))?,
                )?;
                let target = Target::new(resolve.target);
                let mut sender = ProgressSender::new(connection);
                let module = self.resolve(
                    &resolve.op,
                    &revision,
                    &target,
                    &resolve.descriptor,
                    &mut |progress| sender.progress(progress),
                )?;
                sender.check()?;
                let mut files = Vec::with_capacity(module.files.len());
                let mut descriptors = Vec::with_capacity(module.files.len());
                for file in &module.files {
                    files.push(wire::ModuleFile {
                        name: file.name.clone(),
                        size: file.size,
                        path: file.path.to_string_lossy().into_owned(),
                    });
                }
                for file in module.files {
                    descriptors.push(std::os::fd::OwnedFd::from(file.file));
                }
                let response = wire::ServerMessage {
                    message: Some(wire::server_message::Message::Resolved(
                        wire::ResolveResponse {
                            form: wire::ModuleForm::from(module.form) as i32,
                            files,
                            // Field 3 is superseded by `entries` and left empty.
                            entry: String::new(),
                            entries: module.entries.into_iter().collect(),
                            metadata: module.metadata.unwrap_or_default(),
                            interface_version: module.interface_version,
                        },
                    )),
                };
                // The module's files ride on the frame that names them, in its order.
                let borrowed: Vec<std::os::fd::BorrowedFd<'_>> =
                    descriptors.iter().map(std::os::fd::AsFd::as_fd).collect();
                connection.send_with(&response, &borrowed)?;
                Ok(connection.flush()?)
            }
            wire::request::Request::Cover(cover) => {
                let mut preferred = Vec::with_capacity(cover.preferred.len());
                for entry in cover.preferred {
                    preferred.push(protocol::Preference {
                        phase: wire::Phase::try_from(entry.phase)
                            .map_err(|_| Error::Protocol("unknown package phase".into()))?
                            .try_into()?,
                        backend: entry.backend,
                        revision: (entry.revision != 0).then_some(entry.revision),
                    });
                }
                let coverage = self.cover(&protocol::CoverRequest {
                    op: cover.op,
                    targets: cover.targets.into_iter().map(Target::new).collect(),
                    keys: cover.keys,
                    preferred,
                    fixed: cover
                        .fixed
                        .into_iter()
                        .map(|fixed| (fixed.backend, fixed.revision))
                        .collect(),
                })?;
                let coverage = wire::Coverage {
                    covered: coverage
                        .covered
                        .into_iter()
                        .map(|(package, keys)| wire::PackageKeys {
                            package: Some(wire::Package {
                                op: package.op,
                                backend: package.backend,
                                revision: package.revision,
                            }),
                            keys,
                        })
                        .collect(),
                    uncovered: coverage.uncovered,
                    candidates: coverage
                        .candidates
                        .into_iter()
                        .map(|package| wire::Package {
                            op: package.op,
                            backend: package.backend,
                            revision: package.revision,
                        })
                        .collect(),
                };
                Ok(connection.deliver(&wire::ServerMessage {
                    message: Some(wire::server_message::Message::Coverage(coverage)),
                })?)
            }
            wire::request::Request::List(_) => {
                let listing = wire::ListResponse {
                    packages: self
                        .list()?
                        .into_iter()
                        .map(|(package, modules)| wire::PackageEntry {
                            op: package.op,
                            backend: package.backend,
                            revision: package.revision,
                            modules: modules
                                .into_iter()
                                .map(|module| wire::ModuleEntry {
                                    target: module.target.to_string(),
                                    descriptor: module.descriptor,
                                    form: wire::ModuleForm::from(module.form) as i32,
                                    interface_version: module.interface_version,
                                })
                                .collect(),
                        })
                        .collect(),
                };
                Ok(connection.deliver(&wire::ServerMessage {
                    message: Some(wire::server_message::Message::Listing(listing)),
                })?)
            }
        }
    }

    /// Installs a package and returns its revision (design section
    /// 6.3).
    ///
    /// Only the upstream Build Server installs; every other one refuses and
    /// names the upstream. A revision fixes the package's sources, tuning data
    /// and toolchain versions, so an installed revision is never overwritten:
    /// each installation of the same operator and backend is the next
    /// revision (design section 5.1), numbered after every revision directory
    /// the package root holds.
    ///
    /// The revision records its build inputs ([`Inputs`]): the digest of the
    /// package's files and this host's identity of the toolchain that builds
    /// them. A host without that toolchain records why it has none, and no host
    /// builds the revision until it is installed where the toolchain is.
    pub fn install(&self, directory: &Path) -> Result<protocol::Package> {
        if let Some(upstream) = &self.upstream {
            return Err(Error::Refused {
                reason: format!(
                    "only the upstream Build Server installs packages, and this is not it: \
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
        let mut packages = self
            .packages
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // A revision directory the server does not serve (one installed with
        // no record of its inputs) still holds its number.
        let on_disk = directories(&self.package_root.join(&manifest.op).join(&manifest.backend))
            .unwrap_or_default()
            .iter()
            .filter_map(|revision| revision.file_name()?.to_str()?.parse::<u32>().ok())
            .max();
        let revision = packages
            .keys()
            .filter(|(op, backend, _)| op == &manifest.op && backend == &manifest.backend)
            .map(|(_, _, revision)| *revision)
            .chain(on_disk)
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
        let kind = manifest.source(directory)?.toolchain();
        let toolchain = self
            .toolchains
            .identity(kind)
            .unwrap_or_else(|error| format!("{kind} unavailable: {error}"));
        let (files, generated) = crate::snapshot::package_files(directory)?;
        let inputs = Inputs {
            package: crate::snapshot::files_digest(&files, generated.as_deref())?,
            toolchain,
        };
        install_package(&files, generated.as_deref(), &root, &inputs, |staged| {
            // A crate's revision is built with its lockfile, locked: the
            // lockfile the snapshot took is its workspace's, so it is reduced
            // here, once, to the crates the snapshot holds. A host without
            // cargo keeps the revision as it is and builds nothing of it.
            match (
                &generated,
                self.toolchains.get(crate::manifest::ToolchainKind::Cargo),
            ) {
                (Some(_), Ok(cargo)) => crate::source::lock_snapshot(cargo, staged),
                _ => Ok(()),
            }
        })?;
        // The source compiles the installed copy, not the tree the install
        // named: a downstream server's package root is a read-only mount of the
        // upstream's, and the publisher's directory need not exist there.
        let source = manifest.source(&root)?;
        let package = Arc::new(Package {
            manifest,
            root,
            source,
            inputs,
        });
        let (op, backend) = (
            package.manifest.op.clone(),
            package.manifest.backend.clone(),
        );
        packages.insert((op.clone(), backend.clone(), revision), package);
        Ok(protocol::Package {
            op,
            backend,
            revision,
        })
    }

    /// The latest installed revision of the package in `directory` when that
    /// revision was installed from the same build inputs, and `None` otherwise.
    ///
    /// A revision fixes what its modules are built from (design section 5.1):
    /// the package's files, and the toolchain with its compile driver and pins.
    /// Its record of them ([`Inputs`]) is compared with the directory's files
    /// and this host's toolchain as they are now, so a changed kernel source,
    /// driver, pin or toolchain is a new revision. Only the latest revision is
    /// compared, because it is the one `prepare` selects by backend: modules
    /// built under an older revision with the same inputs are not the ones a
    /// later `prepare` would reach.
    ///
    /// # Errors
    ///
    /// Fails when the directory holds no valid manifest or either tree cannot
    /// be read.
    pub fn installed(&self, directory: &Path) -> Result<Option<protocol::Package>> {
        let manifest = Manifest::load(directory).map_err(|source| Error::Manifest {
            package: directory.to_path_buf(),
            source: Box::new(source),
        })?;
        let latest = {
            let packages = self
                .packages
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            packages
                .iter()
                .filter(|((op, backend, _), _)| op == &manifest.op && backend == &manifest.backend)
                .max_by_key(|((_, _, revision), _)| *revision)
                .map(|((_, _, revision), package)| (*revision, Arc::clone(package)))
        };
        let Some((revision, package)) = latest else {
            return Ok(None);
        };
        let Ok(toolchain) = self.toolchains.identity(package.toolchain()) else {
            return Ok(None);
        };
        if Inputs::of(directory, toolchain)? != package.inputs {
            return Ok(None);
        }
        Ok(Some(protocol::Package {
            op: manifest.op,
            backend: manifest.backend,
            revision,
        }))
    }

    /// Selects the package revisions of an epoch's operators and commits every module
    /// their descriptors need that the store misses (design section 6.3).
    ///
    /// Each build is reported through `progress` as it is queued and as it
    /// ends; the call returns each requested slot with the package revision
    /// selected for it once every module is committed, and fails with the
    /// first build that failed.
    pub fn prepare(
        &self,
        request: &PrepareRequest,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<(ModuleSlot, PackageRevision)>> {
        let mut chosen = Vec::with_capacity(request.packages.len());
        for entry in &request.packages {
            let revision = self.select(&entry.slot.op, &entry.selector)?;
            chosen.push((entry.slot.clone(), revision));
        }

        // Asking for the descriptors of a slot no package revision fills would
        // silently leave the epoch without its modules; it is the request's
        // inconsistency, not a missing build.
        for descriptors in &request.descriptors {
            if !chosen.iter().any(|(slot, _)| *slot == descriptors.slot) {
                return Err(Error::Protocol(format!(
                    "prepare asks for the descriptors of {:?} and no package fills that slot",
                    descriptors.slot
                )));
            }
        }

        let mut builds: Vec<(ModulePath, Receiver<BuildState>)> = Vec::new();
        let mut failures: Vec<Error> = Vec::new();
        for (slot, revision) in &chosen {
            let package = self.package(&slot.op, revision)?;
            // A slot's package revision builds the keys named for that slot alone: two
            // slots of one operator, one per phase or per part, hold disjoint
            // keys, and a package is never asked for another slot's.
            let mut keys = Vec::new();
            for descriptors in request
                .descriptors
                .iter()
                .filter(|descriptors| descriptors.slot == *slot)
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
                        op: slot.op.clone(),
                        backend: revision.backend.clone(),
                        revision: revision.revision,
                        descriptor: key,
                    });
                }
                for target in self.targets(&package, &request.targets)? {
                    let path = ModulePath::new(
                        slot.op.clone(),
                        revision.backend.clone(),
                        revision.revision,
                        package.inputs.digest(),
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
                "the build's queue ended without a response".into(),
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
        Ok(chosen)
    }

    /// Resolves one module, building it first when the store misses it
    /// (design section 6.3).
    ///
    /// A missing module is `Waiting` work, ahead of every `Background` build
    /// queued for `prepare`, and the call returns after it is committed.
    pub fn resolve(
        &self,
        op: &str,
        revision: &PackageRevision,
        target: &Target,
        descriptor: &str,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Module> {
        let package = self.package(op, revision)?;
        if !package.manifest.builds_for(target.as_str()) {
            return Err(Error::Build {
                op: op.to_owned(),
                backend: revision.backend.clone(),
                revision: revision.revision,
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
                        package.manifest.single_target().to_string()
                    }
                ),
            });
        }
        if !package.manifest.covers(descriptor) {
            return Err(Error::UnsupportedDescriptor {
                op: op.to_owned(),
                backend: revision.backend.clone(),
                revision: revision.revision,
                descriptor: descriptor.to_owned(),
            });
        }
        let path = ModulePath::new(
            op,
            revision.backend.clone(),
            revision.revision,
            package.inputs.digest(),
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
                "the build's queue ended without a response".into(),
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
        for (name, path) in files {
            let handle = fs::File::open(&path)?;
            let size = handle.metadata()?.len();
            module_files.push(ModuleFile {
                name,
                size,
                file: handle,
                path,
            });
        }
        let entries = entries_of(&package, &self.store.directory(&path))?;
        Ok(Module {
            form: package.manifest.module.form,
            files: module_files,
            metadata: package.manifest.module.metadata_name(&path.descriptor),
            entries,
            // The record written with the module when it was built; the
            // revision's manifest fixed it.
            interface_version: self
                .store
                .interface_version(&path)
                .unwrap_or(package.manifest.interface_version),
        })
    }

    /// Lists every installed package revision with its modules.
    pub fn list(&self) -> Result<Vec<(protocol::Package, Vec<protocol::ModuleEntry>)>> {
        let packages = self
            .packages
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut listed: BTreeMap<(String, String, u32), Vec<protocol::ModuleEntry>> = packages
            .keys()
            .map(|key| (key.clone(), Vec::new()))
            .collect();
        for path in self.store.list()? {
            let key = (path.op.clone(), path.backend.clone(), path.revision);
            // A module under the revision's number and other inputs was built
            // by another installation, and is not this revision's.
            let Some(package) = packages
                .get(&key)
                .filter(|package| package.inputs.digest() == path.inputs)
            else {
                continue;
            };
            let form = package.manifest.module.form;
            let Some(modules) = listed.get_mut(&key) else {
                continue;
            };
            let interface_version = self
                .store
                .interface_version(&path)
                .unwrap_or(package.manifest.interface_version);
            modules.push(protocol::ModuleEntry {
                target: Target::new(path.target),
                descriptor: path.descriptor,
                form,
                interface_version,
            });
        }
        Ok(listed
            .into_iter()
            .map(|((op, backend, revision), modules)| {
                (
                    protocol::Package {
                        op,
                        backend,
                        revision,
                    },
                    modules,
                )
            })
            .collect())
    }

    /// Selects the installed package of each of an operator's descriptor keys:
    /// the one rule by which a key's package revision is chosen (design section 6.3).
    ///
    /// The candidates are the packages a module set fixed for the operator,
    /// when the request states them, and otherwise each backend's latest
    /// installed revision - or the revision a named backend states; of those,
    /// the ones that build for every requested target. The response lists them
    /// in selection order, which is what the daemon fixes a module set from. A key goes to the first candidate whose manifest
    /// covers it, in the selection order: a backend the request names for the
    /// key's phase ([`proto::module::PhaseKind::of_key`]) first, then the manifest's
    /// `priority`, highest first, then the backend whose name sorts first. A
    /// named backend that does not cover a key leaves it to the next candidate,
    /// so a deployment that names its attention backend still serves the keys
    /// only another package declares. A key no candidate covers is returned in
    /// `uncovered`. Nothing is built, and the response does not depend on what
    /// the store has built: every host selects alike.
    ///
    /// # Errors
    ///
    /// Fails when the request names or fixes a revision of a backend that is
    /// not installed.
    pub fn cover(&self, request: &protocol::CoverRequest) -> Result<protocol::Coverage> {
        let packages = self
            .packages
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // Each backend's revision: the one a module set fixed, when the
        // request states them, and otherwise the one a named backend states,
        // else the latest installed.
        let mut revisions: BTreeMap<&str, (u32, &Arc<Package>)> = BTreeMap::new();
        for ((op, backend, revision), package) in packages.iter() {
            if *op != request.op {
                continue;
            }
            let stated = if request.fixed.is_empty() {
                request
                    .preferred
                    .iter()
                    .find(|preference| preference.backend == *backend)
                    .and_then(|preference| preference.revision)
            } else {
                // A package the set does not hold is no candidate at all.
                match request.fixed.iter().find(|(fixed, _)| fixed == backend) {
                    Some((_, fixed)) => Some(*fixed),
                    None => continue,
                }
            };
            if stated.is_some_and(|stated| stated != *revision) {
                continue;
            }
            let held = revisions
                .entry(backend.as_str())
                .or_insert((*revision, package));
            if *revision > held.0 {
                *held = (*revision, package);
            }
        }
        let stated = request
            .preferred
            .iter()
            .filter(|_| request.fixed.is_empty())
            .filter_map(|preference| Some((&preference.backend, preference.revision?)))
            .chain(
                request
                    .fixed
                    .iter()
                    .map(|(backend, revision)| (backend, *revision)),
            );
        for (backend, revision) in stated {
            if !revisions.contains_key(backend.as_str()) {
                return Err(Error::UnknownRevision {
                    op: request.op.clone(),
                    backend: backend.clone(),
                    revision,
                });
            }
        }
        let mut candidates: Vec<(&str, u32, &Arc<Package>)> = revisions
            .into_iter()
            .filter(|(_, (_, package))| {
                request
                    .targets
                    .iter()
                    .all(|target| package.manifest.builds_for(target.as_str()))
            })
            .map(|(backend, (revision, package))| (backend, revision, package))
            .collect();
        candidates.sort_by(|left, right| {
            right
                .2
                .manifest
                .priority
                .cmp(&left.2.manifest.priority)
                .then_with(|| left.0.cmp(right.0))
        });

        // The keys selected for each candidate, index for index.
        let mut chosen: Vec<Vec<String>> = vec![Vec::new(); candidates.len()];
        let mut uncovered: Vec<String> = Vec::new();
        let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for key in &request.keys {
            if !seen.insert(key.as_str()) {
                continue;
            }
            let phase = proto::module::PhaseKind::of_key(key);
            let preferred = |backend: &str| {
                request.preferred.iter().any(|preference| {
                    preference.backend == backend
                        && (preference.phase.is_none() || preference.phase == phase)
                })
            };
            let covering = |index: &usize| candidates[*index].2.manifest.covers(key);
            let selected = (0..candidates.len())
                .filter(|index| preferred(candidates[*index].0))
                .find(covering)
                .or_else(|| (0..candidates.len()).find(covering));
            match selected {
                Some(index) => chosen[index].push(key.clone()),
                None => uncovered.push(key.clone()),
            }
        }

        let selected = candidates
            .iter()
            .map(|(backend, revision, _)| protocol::Package {
                op: request.op.clone(),
                backend: (*backend).to_owned(),
                revision: *revision,
            })
            .collect();
        let covered = candidates
            .iter()
            .zip(chosen)
            .filter(|(_, keys)| !keys.is_empty())
            .map(|((backend, revision, _), keys)| {
                (
                    protocol::Package {
                        op: request.op.clone(),
                        backend: (*backend).to_owned(),
                        revision: *revision,
                    },
                    keys,
                )
            })
            .collect();
        Ok(protocol::Coverage {
            covered,
            uncovered,
            candidates: selected,
        })
    }

    /// The package revision a `prepare` names for an operator, with the package's
    /// parameter defaults filled in for every argument the request leaves out
    /// (design section 5.2): what the daemon records is the package revision the
    /// package will run, not a partial one.
    fn select(&self, op: &str, selector: &Selector) -> Result<PackageRevision> {
        let revision = match selector {
            Selector::Fixed(revision) => {
                self.package(op, revision)?;
                revision.clone()
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
                PackageRevision {
                    backend: backend.clone(),
                    revision,
                    arguments: arguments.clone(),
                }
            }
        };
        let package = self.package(op, &revision)?;
        let mut arguments = toml::Table::new();
        for (name, default) in &package.manifest.parameters {
            arguments.insert(name.clone(), default.clone());
        }
        for (name, value) in &revision.arguments {
            arguments.insert(name.clone(), value.clone());
        }
        Ok(PackageRevision {
            arguments,
            ..revision
        })
    }

    /// The installed package a package revision names.
    fn package(&self, op: &str, revision: &PackageRevision) -> Result<Arc<Package>> {
        self.packages
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&(op.to_owned(), revision.backend.clone(), revision.revision))
            .cloned()
            .ok_or_else(|| Error::UnknownRevision {
                op: op.to_owned(),
                backend: revision.backend.clone(),
                revision: revision.revision,
            })
    }

    /// The targets a package is built for: those the request names, checked
    /// against this host, or every target of the host that the package
    /// supports (design section 5.3). A scheduling policy is built once, for
    /// wasm32, whatever the host's GPUs are.
    fn targets(&self, package: &Package, requested: &[Target]) -> Result<Vec<Target>> {
        if package.manifest.kernel.is_none() {
            // A package of one module has one target, whatever the host's GPUs
            // are: wasm32 for a policy, the host's platform for an executable.
            // The targets a request names are its kernel packages', so they
            // do not select among this package's.
            return Ok(vec![package.manifest.single_target()]);
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

/// The C entries a module is launched through: its package's
/// `[module.entries]`, overridden by each role the module's entries file names
/// (design section 4.2). The file is part of the module, written by the build
/// for this descriptor, so a package whose modules each launch through their
/// own C entry returns the entries of the module it resolved.
fn entries_of(package: &Package, directory: &Path) -> Result<BTreeMap<String, String>> {
    let mut entries = package.manifest.module.entries.clone();
    let Some(name) = package.manifest.module.entries_file.as_deref() else {
        return Ok(entries);
    };
    let file = directory.join(name);
    let text = fs::read_to_string(&file).map_err(|error| {
        Error::Protocol(format!(
            "the module names its entries in {name}, which it does not hold: {error}"
        ))
    })?;
    let written: BTreeMap<String, String> = serde_json::from_str(&text).map_err(|error| {
        Error::Protocol(format!(
            "{name} is not a JSON object of C entry roles to symbols: {error}"
        ))
    })?;
    entries.extend(written);
    Ok(entries)
}

/// How long the listener waits between accept attempts, so a supervisor can
/// stop the server.
const ACCEPT_TICK: Duration = Duration::from_millis(10);

/// The queue's side of a build: what is being built, and who waits for it.
struct Queue {
    store: Arc<Store>,
    toolchains: Arc<Toolchains>,
    state: Mutex<QueueState>,
    wake: Condvar,
}

struct QueueState {
    builder: Builder,
    /// The package of each path whose build is queued or running: what
    /// compiles it.
    packages: HashMap<ModulePath, Arc<Package>>,
    /// One sender per waiting request, by the waiter's identity.
    waiters: HashMap<u64, mpsc::Sender<BuildState>>,
    next_waiter: u64,
    stopping: bool,
}

impl Queue {
    fn new(store: Arc<Store>, toolchains: Arc<Toolchains>, limits: Limits) -> Arc<Queue> {
        Arc::new(Queue {
            store,
            toolchains,
            state: Mutex::new(QueueState {
                builder: Builder::new(limits),
                packages: HashMap::new(),
                waiters: HashMap::new(),
                next_waiter: 0,
                stopping: false,
            }),
            wake: Condvar::new(),
        })
    }

    /// Submits a build for a module path, or joins the one queued or running
    /// for it and raises its urgency. Returns when the build ends.
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
        state.packages.entry(path.clone()).or_insert(package);
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
            let QueueState {
                builder, packages, ..
            } = &mut *state;
            let started = builder.start(framed, |build| {
                packages
                    .get(&build.path)
                    .map_or(0, |package| peak(&self.toolchains, package, &build.path))
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
                if let Some(package) = packages.get(&path).cloned() {
                    spawned.push((package, path));
                }
            }
            drop(state);
            for (package, path) in spawned {
                let queue = Arc::clone(&self);
                std::thread::spawn(move || queue.build(&package, &path));
            }
        }
    }

    /// Runs one build and tells its waiters, and the driver, how it ended.
    ///
    /// A build that panics fails like one that returned an error: the
    /// failure is the build's log, and the queue keeps its slot.
    fn build(self: Arc<Self>, package: &Package, path: &ModulePath) {
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.compile(package, path)))
                .unwrap_or_else(|panic| {
                    let message = panic
                        .downcast_ref::<&str>()
                        .map(|message| (*message).to_owned())
                        .or_else(|| panic.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "the build panicked".to_owned());
                    Err(Error::Build {
                        op: path.op.clone(),
                        backend: path.backend.clone(),
                        revision: path.revision,
                        target: path.target.clone(),
                        descriptor: path.descriptor.clone(),
                        message: format!("the build panicked: {message}"),
                    })
                })
                .map_err(|error| error.to_string());
        let mut state = self.lock();
        let ended: Option<Build> = state.builder.finish(path, result);
        state.packages.remove(path);
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

    /// Stops the queue's driver once the build it was starting has ended.
    fn stop(&self) {
        self.lock().stopping = true;
        self.wake.notify_all();
    }

    /// Compiles one module into the store.
    fn compile(&self, package: &Package, path: &ModulePath) -> Result<()> {
        let kind = package.toolchain();
        let workspace = crate::source::Workspace::new()?;
        let target = Target::new(path.target.clone());
        let toolchain = self.toolchains.get(kind)?;
        // A module is stored under the inputs its revision recorded, so
        // only the toolchain and compile driver the revision was
        // installed with may build it.
        let identity = self.toolchains.identity(kind)?;
        if identity != package.inputs.toolchain {
            return Err(Error::Toolchain {
                kind,
                message: format!(
                    "the package {} {} revision {} was installed for {} and this host \
                     builds with {identity}: install the package again",
                    package.manifest.op,
                    package.manifest.backend,
                    path.revision,
                    package.inputs.toolchain,
                ),
            });
        }
        // A package that compiles nothing has no compiler version to
        // match, and a package that declares no version is built by
        // whatever this host has: both leave `[toolchain] version`
        // empty, and neither is compared.
        if kind != crate::manifest::ToolchainKind::None
            && !package.manifest.toolchain.version.is_empty()
            && toolchain.version() != package.manifest.toolchain.version
        {
            return Err(Error::Toolchain {
                kind,
                message: format!(
                    "the package {} {} revision {} is written for {kind} {} and this host \
                     has {}",
                    package.manifest.op,
                    package.manifest.backend,
                    path.revision,
                    package.manifest.toolchain.version,
                    toolchain.version()
                ),
            });
        }
        self.toolchains.check_packages(
            &package.manifest.op,
            &package.manifest.backend,
            &package.manifest.toolchain,
        )?;
        let built =
            package
                .source
                .build(&path.op, &path.descriptor, &target, toolchain, &workspace)?;
        if built.is_empty() {
            return Err(Error::Build {
                op: path.op.clone(),
                backend: path.backend.clone(),
                revision: path.revision,
                target: path.target.clone(),
                descriptor: path.descriptor.clone(),
                message: "the build wrote no file".into(),
            });
        }
        self.store.publish(
            path,
            workspace.path(),
            &built,
            package.manifest.interface_version,
        )?;
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, QueueState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// The peak host memory one build reserves, as its toolchain reports it.
fn peak(toolchains: &Toolchains, package: &Package, path: &ModulePath) -> u64 {
    let kind = package.toolchain();
    toolchains
        .get(kind)
        .map(|toolchain| toolchain.peak_host_bytes(&path.op, &path.descriptor))
        .unwrap_or(0)
}

/// Sends the progress of a request on its connection while it runs.
///
/// A send that fails is kept and ends the sending: the builds go on, nothing
/// more is written to a connection that is gone, and [`Self::check`] returns
/// the failure once the request's work has ended.
struct ProgressSender<'a> {
    connection: &'a mut Connection,
    failure: Option<Error>,
}

impl<'a> ProgressSender<'a> {
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
                inputs: progress.path.inputs.clone(),
                target: progress.path.target.clone(),
                descriptor: progress.path.descriptor.clone(),
                state: wire::BuildState::from(&progress.state) as i32,
                detail: progress.detail.clone(),
            })),
        };
        if let Err(error) = self.connection.deliver(&message) {
            self.failure = Some(error.into());
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
fn refusal(error: &Error) -> wire::Refusal {
    match error {
        Error::Refused { reason, upstream } => wire::Refusal {
            reason: reason.clone(),
            upstream: upstream.clone().unwrap_or_default(),
        },
        other => wire::Refusal {
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
                // A revision installed with no record of its inputs cannot name
                // the modules it would build, and modules found under its number
                // may come from other inputs: it is not served, and its package
                // is installed again.
                let Some(inputs) = Inputs::read(&revision)? else {
                    tracing::warn!(
                        "{} records no build inputs and is not served: install the package again",
                        revision.display()
                    );
                    continue;
                };
                let source = manifest.source(&revision)?;
                packages.insert(
                    (op_name.clone(), backend_name.clone(), number),
                    Arc::new(Package {
                        manifest,
                        root: revision.clone(),
                        source,
                        inputs,
                    }),
                );
            }
        }
    }
    Ok(packages)
}

/// Writes a package's snapshot into its revision's directory with the record
/// of its build inputs, atomically: a crash leaves no directory a later scan
/// would mistake for an installed revision, and no revision without its
/// record. `finish` completes the staged revision before it is published.
///
/// A symbolic link is copied as what it names: an installed revision holds
/// every byte it builds from, so a source several packages share in the tree
/// (an operator's ABI header, one convolution source behind three operators)
/// is one file there and a file of each package's revision.
fn install_package(
    files: &[(String, PathBuf)],
    generated: Option<&str>,
    destination: &Path,
    inputs: &Inputs,
    finish: impl FnOnce(&Path) -> Result<()>,
) -> Result<()> {
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
    crate::snapshot::write_files(files, generated, &staged)?;
    finish(&staged)?;
    inputs.write(&staged)?;
    if let Ok(handle) = fs::File::open(&staged) {
        handle.sync_all()?;
    }
    fs::rename(&staged, destination)?;
    if let Ok(handle) = fs::File::open(parent) {
        handle.sync_all()?;
    }
    Ok(())
}

/// The descriptors a `prepare` asks for, as the server's request.
fn prepare_request(message: wire::Prepare) -> Result<PrepareRequest> {
    let mut packages = Vec::with_capacity(message.packages.len());
    for revision in message.packages {
        let slot = protocol::slot_of(revision.op, revision.phase, revision.part)?;
        let selector = match revision.selector {
            Some(wire::package_request::Selector::Backend(backend)) => Selector::Latest {
                backend: backend.backend,
                arguments: arguments(&backend.arguments)?,
            },
            Some(wire::package_request::Selector::Package(revision)) => {
                Selector::Fixed(PackageRevision::try_from(revision)?)
            }
            None => {
                return Err(Error::Protocol(
                    "a package request names no backend or package".into(),
                ));
            }
        };
        packages.push(crate::protocol::PackageRequest { slot, selector });
    }
    let mut descriptors = Vec::with_capacity(message.descriptors.len());
    for entry in message.descriptors {
        descriptors.push(Descriptors {
            slot: protocol::slot_of(entry.op, entry.phase, entry.part)?,
            keys: entry.keys,
        });
    }
    Ok(PrepareRequest {
        targets: message.targets.into_iter().map(Target::new).collect(),
        packages,
        descriptors,
    })
}

/// A package revision's arguments, which travel as a TOML table.
fn arguments(text: &str) -> Result<toml::Table> {
    if text.is_empty() {
        return Ok(toml::Table::new());
    }
    toml::from_str(text)
        .map_err(|error| Error::Protocol(format!("a package's arguments are not TOML: {error}")))
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
