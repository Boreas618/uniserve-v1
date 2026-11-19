//! The Build Server's object model (design section 6.2) and its
//! operations (design section 6.3).
//!
//! A [`BuildServer`] keeps what is installed — one [`Package`] per
//! `(operator, backend, revision)`, each holding its manifest, its directory
//! and its [`Source`] — the module [`Store`] of section 5.3, the host's
//! [`Toolchains`] and the compile [`Queue`] of section 6.2, whose driver
//! thread runs for the server's life. A store build calls its operations in
//! process; `serve_on` serves `install`, `prepare`, `resolve`, `list` and
//! `cover` on a Unix socket to any number of clients, one thread per
//! connection.
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
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::Duration;

use crate::builder::{Build, BuildState, Builder, Limits, Urgency};
use crate::inputs::{Inputs, Origin};
use crate::manifest::{Manifest, ToolchainKind};
use service::Connection;

use crate::protocol::{
    self, Descriptors, Module, ModuleFile, PrepareRequest, Progress, Selector, wire,
};
use crate::snapshot::WORKSPACE;
use crate::source::{Cache, Source, Workspace};
use crate::store::{ModulePath, Root, Staging, Store, directories, file_name};
use crate::target::Target;
use crate::toolchain::Toolchains;
use crate::{Error, Result};
use proto::module::{ModuleSlot, PROGRAM, PackageRevision};

/// How long the queue's driver waits before it reads the host's memory again
/// while builds wait for memory and none of its own runs. The host states no
/// event when other processes release memory, so this is the one wait of the
/// driver that a notification cannot end; a submitted or finished build ends
/// it at once.
const MEMORY_RECHECK: Duration = Duration::from_secs(1);

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
    /// Where the revision's sources came from: the commit of the tree it was
    /// installed from and whether that tree was clean.
    pub origin: Origin,
    /// The operators whose kernel packages the revision launches, as the vocabulary its
    /// snapshot holds names them ([`Manifest::operators`]): an engine revision's; empty for
    /// every other package.
    pub operators: Vec<String>,
}

impl Package {
    /// The toolchain this package's sources compile with, as its manifest
    /// declares it.
    pub fn toolchain(&self) -> ToolchainKind {
        self.manifest.toolchain.kind
    }

    /// Where the package's toolchain runs: the workspace of a cargo package's
    /// snapshot, whose `rust-toolchain.toml` selects cargo's release, and the
    /// package's root otherwise.
    fn toolchain_directory(&self) -> PathBuf {
        if self.manifest.source.kind.builds_with_cargo() {
            self.root.join(crate::snapshot::WORKSPACE)
        } else {
            self.root.clone()
        }
    }

    /// The directory an installed revision of a package lives in.
    fn revision_directory(package_root: &Path, op: &str, backend: &str, revision: u32) -> PathBuf {
        package_root
            .join(op)
            .join(backend)
            .join(revision.to_string())
    }
}

/// How a Build Server works: its roots, its upstream, its limits, its targets
/// and its cache.
#[derive(Clone, Debug)]
pub struct ServerConfig {
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
    /// The GPU targets this host builds for (design section 5.3).
    pub targets: Vec<Target>,
    /// The root of what builds keep between them ([`Cache`]): pinned sources,
    /// cargo's target directory, vendored registry sources and workspaces.
    pub cache: PathBuf,
    /// The source tree whose Python environment compiles Triton, CuTe DSL and
    /// CUDA packages ([`crate::tree`]); `None` for a server with no such
    /// toolchain, which serves what is installed and the read-only stores.
    pub source: Option<PathBuf>,
}

/// A Build Server: what is installed, the module store, the toolchains and
/// the compile queue.
pub struct BuildServer {
    package_root: PathBuf,
    /// Whether this server accepts `install`, and where the upstream is.
    upstream: Option<PathBuf>,
    packages: RwLock<BTreeMap<(String, String, u32), Arc<Package>>>,
    store: Arc<Store>,
    toolchains: Arc<Toolchains>,
    targets: Vec<Target>,
    queue: Arc<Queue>,
    /// The queue's driver, which runs until the server is dropped.
    driver: Option<std::thread::JoinHandle<()>>,
    /// The source tree the server was started with, whose workspace a policy
    /// package's program crates are built in ([`Self::install_program`]).
    source: Option<PathBuf>,
    /// What builds and installs keep between them: the program SDK's snapshot
    /// and the workspaces an installed program crate is composed in.
    cache: Cache,
    /// The snapshot of the source tree's workspace a program crate is added to,
    /// taken at the first program installation ([`Sdk`]).
    sdk: Mutex<Option<Arc<Sdk>>>,
    /// The upstream server's hold of its package root, which makes it the one
    /// writer of the revisions there; a downstream server writes none.
    _package_root: Option<fs::File>,
}

/// The program SDK a policy package's program crates are built against: a copy,
/// in the server's cache, of the snapshot of its source tree's workspace, which
/// holds the SDK crate (`scheduling`) and the workspace dependencies a program
/// crate states with `workspace = true`.
///
/// The snapshot is taken once per server, as the tree's own packages are
/// installed once when the server starts, so every program crate a server
/// installs is built against one SDK, and the copy does not change under a
/// build when the tree is edited.
struct Sdk {
    /// The copy's root, the workspace's root directory.
    root: PathBuf,
    /// The copy's files, each by its path below `root`.
    files: Vec<String>,
}

impl BuildServer {
    /// Opens a server over its configuration: loads the installed packages
    /// from the package root, reads the toolchains, opens the store and starts
    /// the compile queue's driver, which runs until the server is dropped.
    pub fn open(config: &ServerConfig) -> Result<BuildServer> {
        let package_root = match config.upstream {
            None => Some(crate::store::hold(&config.package_root)?),
            Some(_) => None,
        };
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
        let toolchains = Arc::new(Toolchains::discover(config.source.as_deref()));
        let queue = Queue::new(
            Arc::clone(&store),
            Arc::clone(&toolchains),
            config.limits,
            Cache::new(config.cache.clone()),
        );
        let driver = std::thread::spawn({
            let queue = Arc::clone(&queue);
            move || queue.drive()
        });
        Ok(BuildServer {
            package_root: config.package_root.clone(),
            upstream: config.upstream.clone(),
            packages: RwLock::new(packages),
            store,
            toolchains,
            targets: config.targets.clone(),
            queue,
            driver: Some(driver),
            source: config.source.clone(),
            cache: Cache::new(config.cache.clone()),
            sdk: Mutex::new(None),
            _package_root: package_root,
        })
    }

    /// Serves on `listener`, the socket `lock` makes this server the owner of
    /// ([`service::host::bind`], which the caller ran, so the socket accepts
    /// connections before this is called), until `stop` becomes readable or is
    /// closed; the lock removes the socket on the way out.
    ///
    /// Each client's connection is served on its own thread.
    ///
    /// # Errors
    ///
    /// Fails when the wait for a connection or for `stop` fails.
    pub fn serve_on(
        self: Arc<Self>,
        listener: std::os::unix::net::UnixListener,
        lock: service::host::SocketLock,
        stop: &impl AsRawFd,
    ) -> Result<()> {
        listener.set_nonblocking(true)?;
        tracing::info!(
            "build server listening with {} package(s)",
            self.packages
                .read()
                .map(|packages| packages.len())
                .unwrap_or(0)
        );
        loop {
            let mut ready = [
                libc::pollfd {
                    fd: listener.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                },
                libc::pollfd {
                    fd: stop.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                },
            ];
            // SAFETY: poll borrows this live array only for the call.
            if unsafe { libc::poll(ready.as_mut_ptr(), ready.len() as libc::nfds_t, -1) } < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(Error::Io(error));
            }
            if ready[1].revents != 0 {
                break;
            }
            let stream = match listener.accept() {
                Ok((stream, _)) => stream,
                // A client that connected and left before the accept leaves
                // nothing to accept; the next poll waits for the next one.
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => continue,
                Err(error) => {
                    tracing::warn!("accepting a build server connection: {error}");
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
            wire::request::Request::InstallProgram(install) => {
                let files = install
                    .files
                    .into_iter()
                    .map(|file| (PathBuf::from(file.path), file.contents))
                    .collect();
                let installed = self.install_program(&install.directory, &files)?;
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
                            key: slot.key,
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
                            entries: module.entries.into_iter().collect(),
                            metadata: module.metadata.unwrap_or_default(),
                            interface_version: module.interface_version,
                            executable: module.executable.unwrap_or_default(),
                            commit: module.origin.commit,
                            clean: module.origin.clean,
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
                let pins = cover
                    .pins
                    .into_iter()
                    .map(|pin| {
                        let pinned = proto::module::Pin {
                            backend: pin.backend,
                            revision: (pin.revision != 0).then_some(pin.revision),
                        };
                        (pin.key, pinned)
                    })
                    .collect();
                let coverage = self.cover(&protocol::CoverRequest {
                    op: cover.op,
                    targets: cover.targets.into_iter().map(Target::new).collect(),
                    keys: cover.keys,
                    pins,
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
                        .map(|entry| wire::PackageEntry {
                            op: entry.package.op,
                            backend: entry.package.backend,
                            revision: entry.package.revision,
                            commit: entry.origin.commit,
                            clean: entry.origin.clean,
                            operators: entry.operators,
                            modules: entry
                                .modules
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
    /// them. A host without that toolchain refuses the install with the
    /// reason, because it cannot state what the revision's modules are built
    /// by.
    pub fn install(&self, directory: &Path) -> Result<protocol::Package> {
        self.upstream_only(&directory.display().to_string())?;
        let manifest = Manifest::load(directory).map_err(|source| Error::Manifest {
            package: directory.to_path_buf(),
            source: Box::new(source),
        })?;
        // A cargo package's toolchain is the one its workspace's
        // `rust-toolchain.toml` selects, which cargo finds from the package's
        // own directory.
        let toolchain = self
            .toolchains
            .identity(manifest.toolchain.kind, directory)?;
        let files = crate::snapshot::package_files(directory, &manifest)?;
        let inputs = Inputs {
            package: crate::snapshot::files_digest(&files)?,
            toolchain,
        };
        let origin = Origin::of(&crate::snapshot::source_root(directory, &manifest)?);
        self.install_files(manifest, &files, inputs, origin)
    }

    /// Installs the program crate `programs/<directory>` of a policy package,
    /// whose files are `files` (each by its path below that directory), as a
    /// revision of the [`proto::module::PROGRAM`] package `programs.<directory>`
    /// ([`program_backend`]), unless the latest revision of it holds the same
    /// build inputs, which is returned instead. `prepare` and `resolve` then
    /// build its wasm32 module as any program package's.
    ///
    /// The crate is built against the program SDK: it is added as a member to
    /// the snapshot of the workspace of the source tree this server was
    /// started with ([`Sdk`]), so it states its dependencies on the SDK
    /// (`scheduling`) and on the workspace's other dependencies with
    /// `workspace = true`, and cargo locks it among the workspace's packages
    /// offline, from the sources vendored for the workspace's lockfile. The
    /// revision's `package.toml` is the server's: a wasm source of the
    /// operator and backend above, whose module states its own kind. Its
    /// origin has no commit: the crate came with a package, from no tree.
    ///
    /// # Errors
    ///
    /// Refused on a server that is not the upstream or was started with no
    /// source tree, for a directory name that is not one plain path component,
    /// a file path that leaves the crate, or a crate with no `Cargo.toml`;
    /// fails when the workspace cannot be composed or cargo cannot lock the
    /// crate among it.
    pub fn install_program(
        &self,
        directory: &str,
        files: &BTreeMap<PathBuf, Vec<u8>>,
    ) -> Result<protocol::Package> {
        let named = format!("the program crate programs/{directory}");
        self.upstream_only(&named)?;
        let refused = |reason: String| Error::Refused {
            reason,
            upstream: None,
        };
        if directory.is_empty()
            || !directory
                .chars()
                .all(|char| char.is_ascii_alphanumeric() || char == '-' || char == '_')
        {
            return Err(refused(format!(
                "{named} is not installed: a program crate's directory is named by letters, \
                 digits, `-` and `_`"
            )));
        }
        for path in files.keys() {
            if path.as_os_str().is_empty()
                || !path
                    .components()
                    .all(|component| matches!(component, std::path::Component::Normal(_)))
            {
                return Err(refused(format!(
                    "{named} is not installed: its file {} leaves the crate's directory",
                    path.display()
                )));
            }
        }
        if !files.contains_key(Path::new("Cargo.toml")) {
            return Err(refused(format!(
                "{named} is not installed: it holds no Cargo.toml"
            )));
        }
        let sdk = self.sdk(&named)?;

        // The workspace the crate is locked in: the SDK's files linked from
        // its copy, with a manifest that lists the crate among the members and
        // a lockfile of its own, which cargo rewrites.
        let work = self.cache.work();
        fs::create_dir_all(&work)?;
        let staging = Staging::new(&work)?;
        let workspace = staging.path();
        for name in &sdk.files {
            let destination = workspace.join(name);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            if fs::hard_link(sdk.root.join(name), &destination).is_err() {
                fs::copy(sdk.root.join(name), &destination)?;
            }
        }
        let member = Path::new(PROGRAMS).join(directory);
        let crate_root = workspace.join(&member);
        if crate_root.exists() {
            return Err(refused(format!(
                "{named} is not installed: the SDK's workspace holds {} itself",
                member.display()
            )));
        }
        for (path, contents) in files {
            let destination = crate_root.join(path);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&destination, contents)?;
        }
        let backend = program_backend(directory);
        fs::write(crate_root.join(Manifest::FILE), program_manifest(&backend))?;
        let mut manifest: toml::Table =
            toml::from_str(&fs::read_to_string(sdk.root.join("Cargo.toml"))?)
                .map_err(|error| refused(format!("the SDK's Cargo.toml does not read: {error}")))?;
        manifest
            .get_mut("workspace")
            .and_then(|workspace| workspace.get_mut("members"))
            .and_then(toml::Value::as_array_mut)
            .ok_or_else(|| refused("the SDK's Cargo.toml lists no workspace members".into()))?
            .push(toml::Value::String(member.to_string_lossy().into_owned()));
        for (name, contents) in [
            (
                "Cargo.toml",
                toml::to_string(&manifest)
                    .map_err(|error| refused(error.to_string()))?
                    .into_bytes(),
            ),
            ("Cargo.lock", fs::read(sdk.root.join("Cargo.lock"))?),
        ] {
            // A linked file is the SDK copy's own: it is replaced, never written.
            fs::remove_file(workspace.join(name))?;
            fs::write(workspace.join(name), contents)?;
        }
        let vendor = crate::snapshot::vendored(&sdk.root, &self.cache)?;
        let locked = crate::toolchain::cargo(&crate_root)
            .args(["metadata", "--format-version", "1", "--offline", "--config"])
            .arg(&vendor)
            .output()?;
        if !locked.status.success() {
            return Err(refused(format!(
                "{named} is not installed: cargo does not lock it among the SDK's workspace: {}",
                String::from_utf8_lossy(&locked.stderr).trim()
            )));
        }

        // The revision's snapshot: its manifest, the workspace as it now is,
        // and the crate's files.
        let manifest = Manifest::load(&crate_root).map_err(|source| Error::Manifest {
            package: crate_root.clone(),
            source: Box::new(source),
        })?;
        let mut snapshot: BTreeMap<String, PathBuf> = sdk
            .files
            .iter()
            .map(|name| (format!("{WORKSPACE}/{name}"), workspace.join(name)))
            .collect();
        for path in files
            .keys()
            .map(PathBuf::as_path)
            .chain([Path::new(Manifest::FILE)])
        {
            let name = Path::new(WORKSPACE).join(&member).join(path);
            snapshot.insert(name.to_string_lossy().into_owned(), crate_root.join(path));
        }
        let mut snapshot: Vec<(String, PathBuf)> = snapshot.into_iter().collect();
        snapshot.insert(
            0,
            (Manifest::FILE.to_owned(), crate_root.join(Manifest::FILE)),
        );
        let inputs = Inputs {
            package: crate::snapshot::files_digest(&snapshot)?,
            toolchain: self
                .toolchains
                .identity(manifest.toolchain.kind, &crate_root)?,
        };

        let latest = {
            let packages = self
                .packages
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            packages
                .iter()
                .filter(|((op, name, _), _)| op == PROGRAM && *name == backend)
                .max_by_key(|((_, _, revision), _)| *revision)
                .map(|((_, _, revision), package)| (*revision, package.inputs.clone()))
        };
        if let Some((revision, installed)) = latest
            && installed == inputs
        {
            return Ok(protocol::Package {
                op: PROGRAM.to_owned(),
                backend,
                revision,
            });
        }
        self.install_files(manifest, &snapshot, inputs, Origin::default())
    }

    /// The program SDK's snapshot, taken from the source tree at the first
    /// call and kept for the server's life; `installed` names what needs it
    /// in the refusal of a server with no source tree.
    fn sdk(&self, installed: &str) -> Result<Arc<Sdk>> {
        let mut sdk = self
            .sdk
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(sdk) = &*sdk {
            return Ok(Arc::clone(sdk));
        }
        let source = self.source.as_deref().ok_or_else(|| Error::Refused {
            reason: format!(
                "{installed} is not installed: this Build Server was started with no source \
                 tree (`--source`), whose workspace holds the program SDK it is built against"
            ),
            upstream: None,
        })?;

        // The snapshot's names are below `workspace/`; the copy holds the
        // workspace's files at its root.
        let files: Vec<(String, PathBuf)> = crate::snapshot::workspace_files(source)?
            .into_iter()
            .map(|(name, path)| {
                let name = name
                    .strip_prefix(&format!("{WORKSPACE}/"))
                    .unwrap_or(&name)
                    .to_owned();
                (name, path)
            })
            .collect();
        let root = self.cache.sdk(&crate::snapshot::files_digest(&files)?);
        if !root.join("Cargo.lock").is_file() {
            let parent = root.parent().ok_or_else(|| {
                Error::Io(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("{} has no parent directory", root.display()),
                ))
            })?;
            fs::create_dir_all(parent)?;
            let staging = Staging::new(parent)?;
            crate::snapshot::write_files(&files, staging.path())?;
            match staging.publish(&root) {
                Ok(()) => {}
                // Another server of this cache copied the same snapshot first.
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::AlreadyExists | io::ErrorKind::DirectoryNotEmpty
                    ) && root.join("Cargo.lock").is_file() => {}
                Err(error) => return Err(Error::Io(error)),
            }
        }
        let copied = Arc::new(Sdk {
            root,
            files: files.into_iter().map(|(name, _)| name).collect(),
        });
        *sdk = Some(Arc::clone(&copied));
        Ok(copied)
    }

    /// Refuses an installation on a server that is not the upstream, naming
    /// what was to be installed and where the upstream is.
    fn upstream_only(&self, installed: &str) -> Result<()> {
        match &self.upstream {
            Some(upstream) => Err(Error::Refused {
                reason: format!(
                    "only the upstream Build Server installs packages, and this is not it: \
                     {installed} was not installed"
                ),
                upstream: Some(upstream.display().to_string()),
            }),
            None => Ok(()),
        }
    }

    /// Installs the snapshot `files` of a package whose manifest is `manifest`
    /// as the next revision of its operator and backend, with its build inputs
    /// and origin recorded beside it.
    ///
    /// A package that names an operator vocabulary is refused unless the
    /// snapshot holds it and it reads, before anything is written: every
    /// installed revision lists the operators its own files name.
    fn install_files(
        &self,
        manifest: Manifest,
        files: &[(String, PathBuf)],
        inputs: Inputs,
        origin: Origin,
    ) -> Result<protocol::Package> {
        // The package's directory, which a refusal names: the one its manifest was taken from.
        let directory = files
            .iter()
            .find(|(name, _)| name == Manifest::FILE)
            .and_then(|(_, path)| path.parent())
            .unwrap_or(Path::new(""));
        let operators = operators(&manifest, directory, |name| {
            files
                .iter()
                .find(|(file, _)| file == name)
                .map(|(_, path)| path.clone())
        })?;

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
        install_package(files, &root, &inputs, &origin)?;
        // The source compiles the installed copy, not the tree the install
        // named: a downstream server's package root is a read-only mount of the
        // upstream's, and the publisher's directory need not exist there.
        let source = manifest.source(&root);
        let package = Arc::new(Package {
            manifest,
            root,
            source,
            inputs,
            origin,
            operators,
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

    /// Installs the package in `directory` unless its latest revision was
    /// installed from the same build inputs ([`Self::installed`]), and returns
    /// that revision or the new one. Modules built under an unchanged revision
    /// stay the ones `prepare` reaches.
    ///
    /// # Errors
    ///
    /// Fails as [`Self::installed`] and [`Self::install`] fail.
    ///
    /// An unchanged revision installed from a tree that was not clean takes the
    /// origin of a clean tree that holds the same files: the revision is then
    /// that commit's.
    pub fn install_if_changed(&self, directory: &Path) -> Result<protocol::Package> {
        let Some(installed) = self.installed(directory)? else {
            return self.install(directory);
        };
        let manifest = Manifest::load(directory).map_err(|source| Error::Manifest {
            package: directory.to_path_buf(),
            source: Box::new(source),
        })?;
        let origin = Origin::of(&crate::snapshot::source_root(directory, &manifest)?);
        if origin.clean {
            self.adopt_origin(&installed, origin)?;
        }
        Ok(installed)
    }

    /// Records `origin` as the origin of the installed revision `installed`,
    /// when its own is not clean.
    fn adopt_origin(&self, installed: &protocol::Package, origin: Origin) -> Result<()> {
        let mut packages = self
            .packages
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let key = (
            installed.op.clone(),
            installed.backend.clone(),
            installed.revision,
        );
        let Some(package) = packages.get(&key) else {
            return Ok(());
        };
        if package.origin.clean {
            return Ok(());
        }
        origin.write(&package.root)?;
        let adopted = Arc::new(Package {
            manifest: package.manifest.clone(),
            root: package.root.clone(),
            source: package.manifest.source(&package.root),
            inputs: package.inputs.clone(),
            origin,
            operators: package.operators.clone(),
        });
        packages.insert(key, adopted);
        Ok(())
    }

    /// Installs the serving packages of the source tree `source`
    /// ([`crate::tree::packages`]), each unless its latest revision was
    /// installed from the same build inputs ([`Self::install_if_changed`]), so
    /// a deployment selects from the tree this server was started with.
    ///
    /// A package whose toolchain this host lacks is not installed, and the
    /// reason is logged: as for a missing toolchain anywhere, only a
    /// deployment that needs the package fails, naming the package it lacks.
    ///
    /// # Errors
    ///
    /// Fails when the tree cannot be read or a package cannot be installed for
    /// any other reason, such as an invalid manifest.
    pub fn install_tree(&self, source: &Path) -> Result<()> {
        for directory in crate::tree::packages(source)? {
            match self.install_if_changed(&directory) {
                Ok(package) => tracing::info!(
                    op = %package.op,
                    backend = %package.backend,
                    revision = package.revision,
                    "a package of the source tree is installed"
                ),
                Err(Error::Toolchain { kind, message }) => tracing::warn!(
                    package = %directory.display(),
                    toolchain = %kind,
                    "a package of the source tree is not installed, for this host lacks its \
                     toolchain: {message}"
                ),
                Err(error) => return Err(error),
            }
        }
        Ok(())
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
        let Ok(toolchain) = self.toolchains.identity(package.toolchain(), directory) else {
            return Ok(None);
        };
        if Inputs::of(directory, &manifest, toolchain)? != package.inputs {
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
            // slots of one operator, one per pinned key or per part, hold disjoint
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
                message: match &package.manifest.kernel {
                    Some(kernel) => format!(
                        "the package requires {:?} of a target, which {target} does not meet",
                        crate::target::named(&kernel.requires)
                    ),
                    None => format!(
                        "the package builds for {} alone",
                        package.manifest.single_target()
                    ),
                },
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
            // A build that failed stays failed, for `resolve` as for
            // `prepare`: the queue starts a build only when the host has its
            // memory, so a failure is the package's or the toolchain's, and a
            // new revision or toolchain is what changes it.
            if let Some(BuildState::Failed(detail)) = self.queue.state(&path) {
                return Err(Error::Build {
                    op: path.op.clone(),
                    backend: path.backend.clone(),
                    revision: path.revision,
                    target: path.target.clone(),
                    descriptor: path.descriptor.clone(),
                    message: detail,
                });
            }
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
        let directory = self.store.find(&path).ok_or_else(|| Error::Build {
            op: path.op.clone(),
            backend: path.backend.clone(),
            revision: path.revision,
            target: path.target.clone(),
            descriptor: path.descriptor.clone(),
            message: "the module left the store while it was resolved".into(),
        })?;
        let entries = crate::source::entries(&package.manifest, &directory)?;
        Ok(Module {
            origin: package.origin.clone(),
            form: package.manifest.module.form,
            files: module_files,
            metadata: package.manifest.module.metadata_name(&path.descriptor),
            entries,
            executable: package.manifest.executable().map(str::to_owned),
            // The record written with the module when it was built: every
            // committed module states the interface version it was built for.
            interface_version: self
                .store
                .interface_version(&path)
                .ok_or_else(|| Error::Build {
                    op: path.op.clone(),
                    backend: path.backend.clone(),
                    revision: path.revision,
                    target: path.target.clone(),
                    descriptor: path.descriptor.clone(),
                    message: "the module's record states no interface version".into(),
                })?,
        })
    }

    /// Lists every installed package revision with its origin, the operators it launches and its
    /// modules.
    pub fn list(&self) -> Result<Vec<protocol::PackageEntry>> {
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
            // A listed module is committed, and its mark states its version.
            let Some(interface_version) = self.store.interface_version(&path) else {
                continue;
            };
            modules.push(protocol::ModuleEntry {
                target: Target::new(path.target),
                descriptor: path.descriptor,
                form,
                interface_version,
            });
        }
        Ok(listed
            .into_iter()
            .map(|(key, modules)| {
                let (origin, operators) = packages
                    .get(&key)
                    .map(|package| (package.origin.clone(), package.operators.clone()))
                    .unwrap_or_default();
                let (op, backend, revision) = key;
                protocol::PackageEntry {
                    package: protocol::Package {
                        op,
                        backend,
                        revision,
                    },
                    origin,
                    operators,
                    modules,
                }
            })
            .collect())
    }

    /// Selects the installed package of each of an operator's descriptor keys:
    /// the one rule by which a key's package revision is chosen (design section 6.3).
    ///
    /// The candidates are the packages a module set fixed for the operator,
    /// when the request states them, and otherwise each backend's latest
    /// installed revision, or the revision a pin states; of those, the ones
    /// that build for every requested target. The response lists them in
    /// selection order, which is what the daemon fixes a module set from: the
    /// manifest's `priority`, highest first, then the backend whose name sorts
    /// first. A key the request pins (the deployment's `[kernels]` table, by the
    /// key's exact text) goes to its pinned backend; every other key goes to the
    /// first candidate in selection order whose manifest covers it. A key no
    /// candidate covers is returned in `uncovered`. Nothing is built, and the
    /// response does not depend on what the store has built: every host
    /// selects alike.
    ///
    /// # Errors
    ///
    /// Fails when the request pins or fixes a revision of a backend that is
    /// not installed, or pins a key to a backend that is no candidate for the
    /// targets or does not cover the key.
    pub fn cover(&self, request: &protocol::CoverRequest) -> Result<protocol::Coverage> {
        let packages = self
            .packages
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // The revision a pin states of each backend. One backend serves an operator at one
        // revision, so its pins state one, and its unpinned keys are selected at it too.
        let mut pinned: BTreeMap<&str, Option<u32>> = BTreeMap::new();
        for (key, pin) in &request.pins {
            let held = *pinned.entry(pin.backend.as_str()).or_insert(pin.revision);
            if held != pin.revision {
                return Err(Error::Refused {
                    reason: format!(
                        "{} key {key} pins {pin}, and another key pins {} at another revision: \
                         one backend serves an operator at one revision",
                        request.op, pin.backend
                    ),
                    upstream: None,
                });
            }
        }

        // Each backend's revision: the one a module set fixed, when the
        // request states them, and otherwise the one a pin states, else the
        // latest installed.
        let mut revisions: BTreeMap<&str, (u32, &Arc<Package>)> = BTreeMap::new();
        for ((op, backend, revision), package) in packages.iter() {
            if *op != request.op {
                continue;
            }
            let stated = if request.fixed.is_empty() {
                pinned.get(backend.as_str()).copied().flatten()
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
        let stated = pinned
            .iter()
            .filter(|_| request.fixed.is_empty())
            .filter_map(|(backend, revision)| Some(((*backend).to_owned(), (*revision)?)))
            .chain(request.fixed.iter().cloned());
        for (backend, revision) in stated {
            if !revisions.contains_key(backend.as_str()) {
                return Err(Error::UnknownRevision {
                    op: request.op.clone(),
                    backend,
                    revision,
                });
            }
        }
        for backend in pinned.keys() {
            if !revisions.contains_key(backend) {
                return Err(Error::NotInstalled {
                    op: request.op.clone(),
                    backend: (*backend).to_owned(),
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
            let covering = |index: &usize| candidates[*index].2.manifest.covers(key);
            if let Some(pin) = request.pins.get(key) {
                // A pinned key is the pin's alone: a backend that cannot build it for these
                // targets refuses the selection rather than handing the key to another.
                let index = candidates
                    .iter()
                    .position(|(backend, _, _)| *backend == pin.backend)
                    .ok_or_else(|| Error::Refused {
                        reason: format!(
                            "{} key {key} is pinned to {}, which builds for none of the targets \
                             {:?}",
                            request.op, pin.backend, request.targets
                        ),
                        upstream: None,
                    })?;
                if !covering(&index) {
                    return Err(Error::UnsupportedDescriptor {
                        op: request.op.clone(),
                        backend: pin.backend.clone(),
                        revision: candidates[index].1,
                        descriptor: key.clone(),
                    });
                }
                chosen[index].push(key.clone());
                continue;
            }
            match (0..candidates.len()).find(covering) {
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
        let Some(kernel) = &package.manifest.kernel else {
            // A package of one module has one target, whatever the host's GPUs
            // are: wasm32 for a policy, the host's platform for an executable.
            // The targets a request names are its kernel packages', so they
            // do not select among this package's.
            return Ok(vec![package.manifest.single_target()]);
        };
        let refused = |target: String, message: String| Error::Build {
            op: package.manifest.op.clone(),
            backend: package.manifest.backend.clone(),
            revision: 0,
            target,
            descriptor: String::new(),
            message,
        };
        for target in requested {
            if !self.targets.contains(target) {
                return Err(refused(
                    target.to_string(),
                    format!(
                        "this host builds for {:?} and not for {target}",
                        self.targets
                    ),
                ));
            }
        }
        let wanted = if requested.is_empty() {
            &self.targets
        } else {
            requested
        };
        let mut selected = Vec::new();
        for target in wanted {
            if package.manifest.builds_for(target.as_str()) {
                selected.push(target.clone());
            } else if !requested.is_empty() {
                // A target the epoch names and the package cannot build for
                // is a missing module the daemon must be told about, not a
                // target to skip.
                return Err(refused(
                    target.to_string(),
                    format!(
                        "the package requires {:?} of a target",
                        crate::target::named(&kernel.requires)
                    ),
                ));
            }
        }
        if selected.is_empty() {
            return Err(refused(
                String::new(),
                format!(
                    "the package requires {:?} of a target, and this host builds for {:?}",
                    crate::target::named(&kernel.requires),
                    self.targets
                ),
            ));
        }
        Ok(selected)
    }
}

impl Drop for BuildServer {
    /// Stops the queue's driver once the build it was starting has begun;
    /// builds already running end on their own threads.
    fn drop(&mut self) {
        self.queue.stop();
        if let Some(driver) = self.driver.take() {
            let _ = driver.join();
        }
    }
}

/// The queue's side of a build: what is being built, and who waits for it.
struct Queue {
    store: Arc<Store>,
    toolchains: Arc<Toolchains>,
    cache: Cache,
    /// The compiler processes each build may run, the queue's job limit.
    jobs: usize,
    peaks: crate::toolchain::PeakMemory,
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
    fn new(
        store: Arc<Store>,
        toolchains: Arc<Toolchains>,
        limits: Limits,
        cache: Cache,
    ) -> Arc<Queue> {
        Arc::new(Queue {
            store,
            toolchains,
            cache,
            jobs: limits.max_jobs,
            peaks: limits.peaks,
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
    /// allow, and waits for a build to be submitted or to end.
    ///
    /// A build that cannot read the host's memory fails rather than start
    /// ungated: the memory reservation is what keeps the host from running out.
    fn drive(self: Arc<Self>) {
        loop {
            let mut state = self.lock();
            if state.stopping {
                return;
            }
            let pending = state.builder.pending();
            if pending == 0 {
                drop(
                    self.wake
                        .wait(state)
                        .unwrap_or_else(|poisoned| poisoned.into_inner()),
                );
                continue;
            }
            let available = match platform::host::available_memory() {
                Ok(available) => available,
                Err(error) => {
                    let message = format!("the host's available memory cannot be read: {error}");
                    let refused = state.builder.fail_queued(&message);
                    self.notify(&mut state, refused);
                    continue;
                }
            };
            let QueueState {
                builder, packages, ..
            } = &mut *state;
            let started = builder.start(available, |build| {
                packages
                    .get(&build.path)
                    .map_or(0, |package| self.peaks.of(package.toolchain()))
            });
            if started.is_empty() {
                // A running build's end frees what it reserved and wakes the
                // driver; with none running, only the host's other processes
                // can free memory, which they do without an event.
                let running = builder.running();
                drop(if running > 0 {
                    self.wake
                        .wait(state)
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                } else {
                    self.wake
                        .wait_timeout(state, MEMORY_RECHECK)
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .0
                });
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

    /// Tells the waiters of `ended` builds how they ended.
    fn notify(&self, state: &mut QueueState, ended: Vec<Build>) {
        for build in ended {
            state.packages.remove(&build.path);
            for waiter in &build.waiters {
                if let Some(sender) = state.waiters.remove(waiter) {
                    let _ = sender.send(build.state.clone());
                }
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
        let ended = state.builder.finish(path, result);
        self.notify(&mut state, ended.into_iter().collect());
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
        let workspace = Workspace::new(&self.cache)?;
        let target = Target::new(path.target.clone());
        let directory = package.toolchain_directory();
        // A module is stored under the inputs its revision recorded, so only
        // the toolchain and compile driver the revision was installed with may
        // build it.
        let identity = self.toolchains.identity(kind, &directory)?;
        if identity != package.inputs.toolchain {
            return Err(Error::Toolchain {
                kind,
                message: format!(
                    "the package {} {} revision {} was installed for {} and this host builds \
                     with {identity}: install the package again",
                    package.manifest.op,
                    package.manifest.backend,
                    path.revision,
                    package.inputs.toolchain,
                ),
            });
        }
        self.toolchains.check_packages(
            &package.manifest.op,
            &package.manifest.backend,
            &package.manifest.toolchain,
        )?;
        let built = package.source.build(&crate::source::Build {
            op: &path.op,
            key: &path.descriptor,
            target: &target,
            toolchains: &self.toolchains,
            cache: &self.cache,
            jobs: self.jobs,
            workspace: &workspace,
        })?;
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
        let interface_version =
            crate::source::stated_interface(&package.manifest, workspace.path()).map_err(
                |message| Error::Build {
                    op: path.op.clone(),
                    backend: path.backend.clone(),
                    revision: path.revision,
                    target: path.target.clone(),
                    descriptor: path.descriptor.clone(),
                    message,
                },
            )?;
        self.store
            .publish(path, workspace.path(), &built, interface_version)?;
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, QueueState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
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

/// The directory of a policy package that holds its program crates, and of
/// the workspace a program crate is composed into.
const PROGRAMS: &str = "programs";

/// The backend a policy package's program crate `programs/<directory>` is
/// installed under: the crate's path with a dot, so it names no program of the
/// release, whose backends are plain names.
pub fn program_backend(directory: &str) -> String {
    format!("{PROGRAMS}.{directory}")
}

/// The `package.toml` of a policy package's program crate installed as
/// `backend`: a wasm source of the program interface, whose module states its
/// kind and interface version itself.
fn program_manifest(backend: &str) -> String {
    format!(
        "# A policy package's program crate, as the Build Server installed it.\n\
         op = \"{PROGRAM}\"\nbackend = \"{backend}\"\n\n[source]\nkind = \"wasm\"\n"
    )
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
                if !revision.join(Manifest::FILE).is_file() {
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
                // Every installed revision records its build inputs, which name
                // the modules it builds.
                let inputs = Inputs::read(&revision)?;
                let origin = Origin::read(&revision)?;
                let operators = operators(&manifest, &revision, |name| Some(revision.join(name)))?;
                let source = manifest.source(&revision);
                packages.insert(
                    (op_name.clone(), backend_name.clone(), number),
                    Arc::new(Package {
                        manifest,
                        root: revision.clone(),
                        source,
                        inputs,
                        origin,
                        operators,
                    }),
                );
            }
        }
    }
    Ok(packages)
}

/// The operators the package of `manifest` launches ([`Manifest::operators`]): the vocabulary
/// `locate` finds by its name inside the package's revision ([`Manifest::operators_file`]),
/// read; none for a package that names none.
///
/// # Errors
///
/// [`Error::Manifest`], naming `package`, when `locate` finds no such file or the vocabulary does
/// not read ([`crate::manifest::read_operators`]).
fn operators(
    manifest: &Manifest,
    package: &Path,
    locate: impl FnOnce(&str) -> Option<PathBuf>,
) -> Result<Vec<String>> {
    let Some(name) = manifest.operators_file() else {
        return Ok(Vec::new());
    };
    let refused = |source| Error::Manifest {
        package: package.to_path_buf(),
        source: Box::new(source),
    };
    let path = locate(&name).ok_or_else(|| {
        refused(crate::manifest::ManifestError::Invalid(format!(
            "the package's snapshot holds no {name}, the vocabulary its `operators` names: no \
             crate of its workspace packages that file"
        )))
    })?;
    crate::manifest::read_operators(&path).map_err(refused)
}

/// Writes a package's snapshot into its revision's directory with the record
/// of its build inputs, atomically: a crash leaves no directory a later scan
/// would mistake for an installed revision, and no revision without its
/// record.
///
/// A symbolic link is copied as what it names: an installed revision holds
/// every byte it builds from, so a source several packages share in the tree
/// (an operator's ABI header, one convolution source behind three operators)
/// is one file there and a file of each package's revision.
fn install_package(
    files: &[(String, PathBuf)],
    destination: &Path,
    inputs: &Inputs,
    origin: &Origin,
) -> Result<()> {
    let parent = destination.parent().ok_or_else(|| {
        Error::Io(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} has no parent directory", destination.display()),
        ))
    })?;
    let staging = Staging::new(parent)?;
    crate::snapshot::write_files(files, staging.path())?;
    inputs.write(staging.path())?;
    origin.write(staging.path())?;
    fs::File::open(staging.path())?.sync_all()?;
    staging.publish(destination)?;
    Ok(())
}

/// The descriptors a `prepare` asks for, as the server's request.
fn prepare_request(message: wire::Prepare) -> Result<PrepareRequest> {
    let mut packages = Vec::with_capacity(message.packages.len());
    for revision in message.packages {
        let slot = protocol::slot_of(revision.op, revision.key, revision.part);
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
            slot: protocol::slot_of(entry.op, entry.key, entry.part),
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
