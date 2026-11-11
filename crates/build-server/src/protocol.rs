//! The Build Server's protocol and its client (design section 6.3).
//!
//! A connection to the Build Server speaks length-delimited Protobuf
//! frames: the client sends one [`wire::Request`], the server responds with
//! [`wire::ServerMessage`] frames until the one that responds to the request.
//! AOT builds report themselves as [`Progress`] frames while `prepare` runs;
//! `resolve` reports JIT builds the same way. A `resolve` response's module
//! files travel as descriptors with `SCM_RIGHTS`, attached to the frame that
//! names them, in the order it lists them.
//!
//! The schema is append-only and lives in `proto/build_server.proto`: a
//! field is never renumbered, retyped or removed, and a new request or response
//! is a new field of its oneof. The daemon and the main process therefore
//! keep speaking a Build Server of a later build.
//!
//! [`Client`] is the half the daemon and the main process use: the daemon
//! calls [`Client::prepare`] before a deployment starts and before an update,
//! and the main process calls [`Client::resolve`] when it starts and when a
//! package revision changes, and [`Client::cover`] to learn which installed package
//! builds each descriptor key of an operator the deployment does not name.

use std::collections::VecDeque;
use std::io;
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::path::Path;

use prost::Message;
use service::frame;

use crate::builder::BuildState;
use crate::manifest::ModuleForm;
use crate::store::ModulePath;
use crate::target::Target;
use crate::{Error, Result};
use proto::module::{ModuleSlot, PackageRevision, PhaseKind};

/// The messages of the protocol, as the build script compiled them.
pub mod wire {
    include!(concat!(env!("OUT_DIR"), "/uniserve.build_server.rs"));
}

/// Bytes one receive reads at most.
const READ_CHUNK: usize = 64 * 1024;

/// An installed package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Package {
    pub op: String,
    pub backend: String,
    pub revision: u32,
}

/// The descriptors one slot's package revision must cover: `prepare` hands the package revision
/// that fills `slot` exactly these keys, and no slot another's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Descriptors {
    pub slot: ModuleSlot,
    /// Descriptor keys, in the canonical form the operator's `Display` gives.
    pub keys: Vec<String>,
}

/// How a `prepare` names the package revision that fills an operator.
#[derive(Clone, Debug, PartialEq)]
pub enum Selector {
    /// The backend's latest installed revision.
    Latest {
        backend: String,
        arguments: toml::Table,
    },
    /// Exactly this package revision, revision and all.
    Fixed(PackageRevision),
}

/// One operator of an epoch and the package revision that fills it.
#[derive(Clone, Debug, PartialEq)]
pub struct PackageRequest {
    pub slot: ModuleSlot,
    pub selector: Selector,
}

/// What the daemon asks `prepare` for: the package revisions of a set of operators,
/// and the descriptors their modules must cover.
#[derive(Clone, Debug, PartialEq)]
pub struct PrepareRequest {
    /// The targets to build for (design section 5.3).
    pub targets: Vec<Target>,
    pub packages: Vec<PackageRequest>,
    pub descriptors: Vec<Descriptors>,
}

/// One module's build, as it progresses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Progress {
    pub path: ModulePath,
    pub state: BuildState,
    /// On a failure, the build's log.
    pub detail: String,
}

// A `prepare` that waited through a Build Server restart (design section 7.5)
// may report the same step again: the steps of an attempt the restart
// abandoned are not undone, and the attempt that follows finds the modules it
// committed. A consumer shows progress monotonically and never counts frames.

/// One file of a module, over the descriptor it arrived on.
#[derive(Debug)]
pub struct ModuleFile {
    /// The file's name inside the module's directory.
    pub name: String,
    pub size: u64,
    pub file: std::fs::File,
    /// The file in the server's store. A module is never replaced, so the
    /// path names these bytes for as long as the store holds them: a process
    /// of the server's host that runs the file as an executable starts it
    /// from here. A module that is loaded is read from [`Self::file`].
    pub path: std::path::PathBuf,
}

/// A module the Build Server resolved: its form, its files, and what the
/// main process needs to launch it (design sections 4.2 and 5.3).
#[derive(Debug)]
pub struct Module {
    pub form: ModuleForm,
    pub files: Vec<ModuleFile>,
    /// A cubin module's launch metadata, as the metadata JSON's file name
    /// among [`Self::files`].
    pub metadata: Option<String>,
    /// A native module's C entry symbols by role, which its package declares
    /// (design section 4.2): [`LAUNCH_ENTRY`](crate::manifest::LAUNCH_ENTRY)
    /// and any lifecycle entries around it.
    pub entries: std::collections::BTreeMap<String, String>,
    /// The version of the interface the module's package implements, as the
    /// store recorded it when the module was built. Its loader compares it
    /// with the version it was built for and refuses another.
    pub interface_version: u32,
}

/// One module of an installed package, as `list` names it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleEntry {
    pub target: Target,
    pub descriptor: String,
    pub form: ModuleForm,
    /// The interface version the module was built for.
    pub interface_version: u32,
}

/// What `cover` asks: which installed package the Build Server selects for
/// each of an operator's descriptor keys when its modules are built for
/// `targets`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoverRequest {
    pub op: String,
    pub targets: Vec<Target>,
    /// Descriptor keys, in the canonical form the operator's `Display` gives.
    pub keys: Vec<String>,
    /// The backends the deployment names for the operator, preferred for the
    /// keys of their phase; empty leaves every key to the manifests'
    /// priorities.
    pub preferred: Vec<Preference>,
    /// The revisions a module set fixed for the operator's packages, each a
    /// backend and its revision. When any is stated, the candidates are
    /// exactly these: a package or a revision installed since is none. Empty
    /// takes each backend's latest installed revision, which is what the
    /// daemon asks when it fixes a module set.
    pub fixed: Vec<(String, u32)>,
}

/// A backend a deployment names for an operator: preferred over every other
/// package for the keys of `phase` (every key when `None`) that it covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Preference {
    pub phase: Option<PhaseKind>,
    pub backend: String,
    /// The revision to use; `None` is the backend's latest installed one.
    pub revision: Option<u32>,
}

/// The response to `cover`: each requested key once, either under the package
/// selected for it or in `uncovered`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Coverage {
    /// Each selected package with the keys it was selected for: the packages
    /// in the server's selection order, and each one's keys in the order the
    /// request named them.
    pub covered: Vec<(Package, Vec<String>)>,
    /// The keys no installed package of the operator covers for the targets.
    pub uncovered: Vec<String>,
    /// Every candidate the selection chose among, in its selection order.
    pub candidates: Vec<Package>,
}

impl From<&PackageRevision> for wire::PackageRevision {
    fn from(revision: &PackageRevision) -> Self {
        wire::PackageRevision {
            backend: revision.backend.clone(),
            revision: revision.revision,
            arguments: toml::to_string(&revision.arguments).unwrap_or_default(),
        }
    }
}

impl TryFrom<wire::PackageRevision> for PackageRevision {
    type Error = Error;

    fn try_from(revision: wire::PackageRevision) -> Result<Self> {
        Ok(PackageRevision {
            backend: revision.backend,
            revision: revision.revision,
            arguments: arguments(&revision.arguments)?,
        })
    }
}

/// Parses a package revision's arguments, which travel as a TOML table.
fn arguments(text: &str) -> Result<toml::Table> {
    if text.is_empty() {
        return Ok(toml::Table::new());
    }
    toml::from_str(text)
        .map_err(|error| Error::Protocol(format!("a package's arguments are not TOML: {error}")))
}

/// A slot's phase as the wire states it: `ALL` for a slot of every phase.
pub(crate) fn phase_to_wire(phase: Option<PhaseKind>) -> i32 {
    phase.map_or(wire::Phase::All, wire::Phase::from) as i32
}

/// The slot an operator, a wire phase and a wire part name: an empty part is
/// a slot that holds the operator's every descriptor.
pub(crate) fn slot_of(op: String, phase: i32, part: String) -> Result<ModuleSlot> {
    Ok(ModuleSlot {
        op,
        phase: wire::Phase::try_from(phase)
            .map_err(|_| Error::Protocol("unknown package phase".into()))?
            .try_into()?,
        part: (!part.is_empty()).then_some(part),
    })
}

impl From<PhaseKind> for wire::Phase {
    fn from(phase: PhaseKind) -> Self {
        match phase {
            PhaseKind::Prefill => wire::Phase::Prefill,
            PhaseKind::Decode => wire::Phase::Decode,
            PhaseKind::Encode => wire::Phase::Encode,
            PhaseKind::Denoise => wire::Phase::Denoise,
        }
    }
}

impl TryFrom<wire::Phase> for Option<PhaseKind> {
    type Error = Error;

    fn try_from(phase: wire::Phase) -> Result<Self> {
        Ok(match phase {
            wire::Phase::All => None,
            wire::Phase::Prefill => Some(PhaseKind::Prefill),
            wire::Phase::Decode => Some(PhaseKind::Decode),
            wire::Phase::Encode => Some(PhaseKind::Encode),
            wire::Phase::Denoise => Some(PhaseKind::Denoise),
        })
    }
}

impl From<ModuleForm> for wire::ModuleForm {
    fn from(form: ModuleForm) -> Self {
        match form {
            ModuleForm::Cubin => wire::ModuleForm::Cubin,
            ModuleForm::Native => wire::ModuleForm::Native,
            ModuleForm::Wasm => wire::ModuleForm::Wasm,
            ModuleForm::Executable => wire::ModuleForm::Executable,
        }
    }
}

impl TryFrom<wire::ModuleForm> for ModuleForm {
    type Error = Error;

    fn try_from(form: wire::ModuleForm) -> Result<Self> {
        match form {
            wire::ModuleForm::Cubin => Ok(ModuleForm::Cubin),
            wire::ModuleForm::Native => Ok(ModuleForm::Native),
            wire::ModuleForm::Wasm => Ok(ModuleForm::Wasm),
            wire::ModuleForm::Executable => Ok(ModuleForm::Executable),
            wire::ModuleForm::Unspecified => Err(Error::Protocol("a module has no form".into())),
        }
    }
}

impl From<&BuildState> for wire::BuildState {
    fn from(state: &BuildState) -> Self {
        match state {
            BuildState::Queued => wire::BuildState::Queued,
            BuildState::Building => wire::BuildState::Building,
            BuildState::Committed => wire::BuildState::Committed,
            BuildState::Failed(_) => wire::BuildState::Failed,
        }
    }
}

impl From<wire::BuildState> for BuildState {
    fn from(state: wire::BuildState) -> Self {
        match state {
            wire::BuildState::Queued => BuildState::Queued,
            wire::BuildState::Building => BuildState::Building,
            wire::BuildState::Committed => BuildState::Committed,
            wire::BuildState::Failed => BuildState::Failed(String::new()),
            wire::BuildState::Unspecified => BuildState::Queued,
        }
    }
}

impl From<wire::Progress> for Progress {
    fn from(progress: wire::Progress) -> Self {
        let state = progress.state().into();
        Progress {
            path: ModulePath {
                op: progress.op,
                backend: progress.backend,
                revision: progress.revision,
                inputs: progress.inputs,
                target: progress.target,
                descriptor: progress.descriptor,
            },
            state,
            detail: progress.detail,
        }
    }
}

/// The request a `prepare` sends, as the wire message.
fn prepare_message(request: &PrepareRequest) -> Result<wire::Prepare> {
    let mut packages = Vec::with_capacity(request.packages.len());
    for entry in &request.packages {
        let selector = match &entry.selector {
            Selector::Latest { backend, arguments } => {
                wire::package_request::Selector::Backend(wire::Backend {
                    backend: backend.clone(),
                    arguments: toml::to_string(arguments).map_err(|error| {
                        Error::Protocol(format!("a package's arguments are not TOML: {error}"))
                    })?,
                })
            }
            Selector::Fixed(revision) => {
                wire::package_request::Selector::Package((revision).into())
            }
        };
        packages.push(wire::PackageRequest {
            op: entry.slot.op.clone(),
            phase: phase_to_wire(entry.slot.phase),
            selector: Some(selector),
            part: entry.slot.part.clone().unwrap_or_default(),
        });
    }
    let descriptors = request
        .descriptors
        .iter()
        .map(|descriptors| wire::Descriptors {
            op: descriptors.slot.op.clone(),
            keys: descriptors.keys.clone(),
            phase: phase_to_wire(descriptors.slot.phase),
            part: descriptors.slot.part.clone().unwrap_or_default(),
        })
        .collect();
    Ok(wire::Prepare {
        targets: request
            .targets
            .iter()
            .map(|target| target.to_string())
            .collect(),
        packages,
        descriptors,
    })
}

/// A connection to the Build Server: framed messages, and the descriptors
/// that travel with them.
#[derive(Debug)]
pub struct Connection {
    stream: UnixStream,
    inbox: Vec<u8>,
    descriptors: VecDeque<OwnedFd>,
}

/// How long a `prepare` or a `resolve` waits through a restart, and how often
/// it looks for the server again (design section 7.5 bounds the wait).
///
/// The wait is a wall-clock budget: an attempt costs its own time, and the
/// pauses are what is left of the budget, so a slow attempt shortens the pause
/// rather than the budget. A caller needing a hard deadline of its own wraps
/// the call.
const WAIT: std::time::Duration = std::time::Duration::from_secs(30);
const WAIT_BETWEEN: std::time::Duration = std::time::Duration::from_millis(100);

/// Pauses before the next attempt, and returns whether there is time left.
fn pause_until(deadline: std::time::Instant) -> bool {
    let now = std::time::Instant::now();
    if now >= deadline {
        return false;
    }
    std::thread::sleep(WAIT_BETWEEN.min(deadline - now));
    true
}

/// Reads one response frame on `connection`.
fn response_on(connection: &mut Connection) -> Result<wire::server_message::Message> {
    let (message, descriptors) = connection.receive::<wire::ServerMessage>()?;
    let response = message.message.ok_or_else(|| unexpected(None))?;
    connection.hold(descriptors);
    Ok(response)
}

impl Connection {
    /// Opens a blocking connection to a Build Server's socket.
    pub fn connect(socket: &Path) -> Result<Connection> {
        Ok(Connection::new(UnixStream::connect(socket)?))
    }

    /// Wraps a stream, as the server does for an accepted connection.
    pub fn new(stream: UnixStream) -> Connection {
        Connection {
            stream,
            inbox: Vec::new(),
            descriptors: VecDeque::new(),
        }
    }

    /// Sends one frame with the descriptors that belong to it, as a descriptor frame of the
    /// workspace's framing module (`service::frame`): they ride on the frame's first byte, so a
    /// receiver that reads a frame exactly never gets another frame's descriptors.
    pub fn send<M: Message>(&mut self, message: &M, descriptors: &[OwnedFd]) -> Result<()> {
        let owned = descriptors
            .iter()
            .map(OwnedFd::try_clone)
            .collect::<io::Result<Vec<_>>>()?;
        for mut segment in frame::descriptor_frame(&message.encode_to_vec(), owned)? {
            while !segment.sent() {
                segment.send(self.stream.as_fd())?;
            }
        }
        Ok(())
    }

    /// Receives one frame and the descriptors that came with it.
    pub fn receive<M: Message + Default>(&mut self) -> Result<(M, Vec<OwnedFd>)> {
        loop {
            if let Some((payload, descriptors)) =
                frame::take_descriptor_frame(&mut self.inbox, &mut self.descriptors)
            {
                let message = M::decode(payload.as_slice())
                    .map_err(|error| Error::Protocol(format!("undecodable frame: {error}")))?;
                return Ok((message, descriptors));
            }
            self.fill()?;
        }
    }

    /// Reads the bytes the current frame still needs, and the descriptors that ride on them.
    fn fill(&mut self) -> Result<()> {
        let frame_bytes = frame::descriptor_frame_bytes(&self.inbox).unwrap_or(frame::HEADER_BYTES);
        let needed = frame_bytes.saturating_sub(self.inbox.len()).max(1);
        let mut buffer = vec![0_u8; needed.min(READ_CHUNK)];
        let received = frame::receive_with_descriptors(
            self.stream.as_fd(),
            &mut buffer,
            &mut self.descriptors,
        )?;
        if received == 0 {
            return Err(Error::Io(io::Error::from(io::ErrorKind::UnexpectedEof)));
        }
        self.inbox.extend_from_slice(&buffer[..received]);
        Ok(())
    }
}

/// A connection to a Build Server, as its clients speak it: the daemon
/// prepares package revisions with it, and the main process resolves them.
#[derive(Debug)]
pub struct Client {
    connection: Connection,
    /// The socket this client was opened on, when it was opened by its path.
    /// A client that wraps an existing connection has no path and does not
    /// wait through a restart.
    socket: Option<std::path::PathBuf>,
}

impl Client {
    /// Opens a connection to the Build Server at `socket`.
    pub fn connect(socket: &Path) -> Result<Client> {
        Ok(Client {
            connection: Connection::connect(socket)?,
            socket: Some(socket.to_path_buf()),
        })
    }

    /// Wraps an open connection.
    pub fn new(connection: Connection) -> Client {
        Client {
            connection,
            socket: None,
        }
    }

    /// Installs a package and returns its revision.
    ///
    /// Only the upstream Build Server installs; another one refuses and
    /// names the upstream, which the error carries.
    pub fn install(&mut self, package: &Path) -> Result<Package> {
        let package = package
            .to_str()
            .ok_or_else(|| {
                Error::Protocol(format!(
                    "the package path {} is not UTF-8",
                    package.display()
                ))
            })?
            .to_owned();
        self.connection.send(
            &wire::Request {
                request: Some(wire::request::Request::Install(wire::Install { package })),
            },
            &[],
        )?;
        match self.response()? {
            wire::server_message::Message::Installed(installed) => installed
                .package
                .map(|package| Package {
                    op: package.op,
                    backend: package.backend,
                    revision: package.revision,
                })
                .ok_or_else(|| Error::Protocol("installed without a package".into())),
            wire::server_message::Message::Refused(refused) => Err(refused_error(refused)),
            other => Err(unexpected(Some(other))),
        }
    }

    /// Selects an epoch's package revisions and has every module of their descriptors
    /// built, reporting each build through `progress` (design section 6.3).
    /// Returns each requested slot with the package revision selected for it.
    ///
    /// This waits until every missing module is committed; a module that
    /// cannot be built fails the call with the build's log.
    pub fn prepare(
        &mut self,
        request: &PrepareRequest,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<(ModuleSlot, PackageRevision)>> {
        self.call(|connection| Self::prepare_on(connection, request, &mut *progress))
    }

    /// One attempt at a `prepare`: the whole exchange on one connection.
    fn prepare_on(
        connection: &mut Connection,
        request: &PrepareRequest,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<(ModuleSlot, PackageRevision)>> {
        connection.send(
            &wire::Request {
                request: Some(wire::request::Request::Prepare(prepare_message(request)?)),
            },
            &[],
        )?;
        loop {
            match response_on(connection)? {
                wire::server_message::Message::Progress(step) => progress(step.into()),
                wire::server_message::Message::Prepared(prepared) => {
                    let mut chosen = Vec::with_capacity(prepared.packages.len());
                    for revision in prepared.packages {
                        let slot = slot_of(revision.op, revision.phase, revision.part)?;
                        let revision = revision.package.ok_or_else(|| {
                            Error::Protocol("a chosen package carries no package".into())
                        })?;
                        chosen.push((slot, revision.try_into()?));
                    }
                    return Ok(chosen);
                }
                wire::server_message::Message::Refused(refused) => {
                    return Err(refused_error(refused));
                }
                other => return Err(unexpected(Some(other))),
            }
        }
    }

    /// Resolves one module, building it first when the store misses it
    /// (design section 6.3).
    pub fn resolve(
        &mut self,
        op: &str,
        revision: &PackageRevision,
        target: &Target,
        descriptor: &str,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Module> {
        self.call(|connection| {
            Self::resolve_on(connection, op, revision, target, descriptor, &mut *progress)
        })
    }

    /// One attempt at a `resolve`: the whole exchange on one connection.
    fn resolve_on(
        connection: &mut Connection,
        op: &str,
        revision: &PackageRevision,
        target: &Target,
        descriptor: &str,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Module> {
        connection.send(
            &wire::Request {
                request: Some(wire::request::Request::Resolve(wire::Resolve {
                    op: op.to_owned(),
                    package: Some(revision.into()),
                    target: target.to_string(),
                    descriptor: descriptor.to_owned(),
                })),
            },
            &[],
        )?;
        loop {
            match response_on(connection)? {
                wire::server_message::Message::Progress(step) => progress(step.into()),
                wire::server_message::Message::Resolved(resolved) => {
                    let form = wire::ModuleForm::try_from(resolved.form)
                        .map_err(|_| Error::Protocol("unknown module form".into()))?
                        .try_into()?;
                    let descriptors = connection.take_descriptors();
                    if descriptors.len() != resolved.files.len() {
                        return Err(Error::Protocol(format!(
                            "a module names {} files and came with {}",
                            resolved.files.len(),
                            descriptors.len()
                        )));
                    }
                    let files = resolved
                        .files
                        .into_iter()
                        .zip(descriptors)
                        .map(|(file, descriptor)| {
                            Ok(ModuleFile {
                                name: file.name,
                                size: file.size,
                                file: std::fs::File::from(descriptor),
                                path: file.path.into(),
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    return Ok(Module {
                        form,
                        files,
                        metadata: (!resolved.metadata.is_empty()).then_some(resolved.metadata),
                        entries: resolved.entries.into_iter().collect(),
                        interface_version: resolved.interface_version,
                    });
                }
                wire::server_message::Message::Refused(refused) => {
                    return Err(refused_error(refused));
                }
                other => return Err(unexpected(Some(other))),
            }
        }
    }

    /// Lists every installed package revision with its modules.
    pub fn list(&mut self) -> Result<Vec<(Package, Vec<ModuleEntry>)>> {
        self.connection.send(
            &wire::Request {
                request: Some(wire::request::Request::List(wire::List {})),
            },
            &[],
        )?;
        match self.response()? {
            wire::server_message::Message::Listing(listing) => {
                let mut packages = Vec::with_capacity(listing.packages.len());
                for package in listing.packages {
                    let mut modules = Vec::with_capacity(package.modules.len());
                    for module in package.modules {
                        let form = wire::ModuleForm::try_from(module.form)
                            .map_err(|_| {
                                Error::Protocol(format!(
                                    "a listed module has an unknown form {}",
                                    module.form
                                ))
                            })?
                            .try_into()?;
                        modules.push(ModuleEntry {
                            target: Target::new(module.target),
                            descriptor: module.descriptor,
                            form,
                            interface_version: module.interface_version,
                        });
                    }
                    packages.push((
                        Package {
                            op: package.op,
                            backend: package.backend,
                            revision: package.revision,
                        },
                        modules,
                    ));
                }
                Ok(packages)
            }
            wire::server_message::Message::Refused(refused) => Err(refused_error(refused)),
            other => Err(unexpected(Some(other))),
        }
    }

    /// Returns which installed package covers each of an operator's
    /// descriptor keys for `request.targets`; nothing is built.
    pub fn cover(&mut self, request: &CoverRequest) -> Result<Coverage> {
        self.connection.send(
            &wire::Request {
                request: Some(wire::request::Request::Cover(wire::Cover {
                    op: request.op.clone(),
                    targets: request.targets.iter().map(Target::to_string).collect(),
                    keys: request.keys.clone(),
                    preferred: request
                        .preferred
                        .iter()
                        .map(|preference| wire::Preference {
                            phase: phase_to_wire(preference.phase),
                            backend: preference.backend.clone(),
                            revision: preference.revision.unwrap_or(0),
                        })
                        .collect(),
                    fixed: request
                        .fixed
                        .iter()
                        .map(|(backend, revision)| wire::FixedRevision {
                            backend: backend.clone(),
                            revision: *revision,
                        })
                        .collect(),
                })),
            },
            &[],
        )?;
        match self.response()? {
            wire::server_message::Message::Coverage(coverage) => {
                let mut covered = Vec::with_capacity(coverage.covered.len());
                for entry in coverage.covered {
                    let package = entry.package.ok_or_else(|| {
                        Error::Protocol("a covering entry names no package".into())
                    })?;
                    covered.push((
                        Package {
                            op: package.op,
                            backend: package.backend,
                            revision: package.revision,
                        },
                        entry.keys,
                    ));
                }
                Ok(Coverage {
                    covered,
                    uncovered: coverage.uncovered,
                    candidates: coverage
                        .candidates
                        .into_iter()
                        .map(|package| Package {
                            op: package.op,
                            backend: package.backend,
                            revision: package.revision,
                        })
                        .collect(),
                })
            }
            wire::server_message::Message::Refused(refused) => Err(refused_error(refused)),
            other => Err(unexpected(Some(other))),
        }
    }

    /// The next frame, with the descriptors that rode on it held for the
    /// response it belongs to.
    ///
    /// Progress reports are frames like any other: the requests that report
    /// builds read them in their own loop, so dropping them here would lose
    /// every report a caller asked for.
    /// Runs one call, waiting through a Build Server restart (design section
    /// 7.5).
    ///
    /// The calls that wait are `prepare` and `resolve`: the Build Server is
    /// not on the serving path, so a restart may only make an epoch's
    /// preparation wait. An attempt that ends because the server was away is
    /// retried on a fresh connection, because a restart ends the old one;
    /// anything the server itself decided — a refusal, a build failure, an
    /// unknown revision, an unsupported descriptor — is returned at once,
    /// since retrying would repeat the decision. A call that timed out or read
    /// something out of protocol is not a restart either: the server was there.
    ///
    /// A retry re-asks for the same content: package revision revisions name their
    /// content and are never overwritten, so a second attempt selects the same
    /// package revisions and finds the modules an abandoned attempt committed. Steps
    /// may therefore be reported twice.
    fn call<T>(&mut self, mut attempt: impl FnMut(&mut Connection) -> Result<T>) -> Result<T> {
        let Some(socket) = self.socket.clone() else {
            return attempt(&mut self.connection);
        };
        let deadline = std::time::Instant::now() + WAIT;
        let restarted = || Error::Restarted {
            socket: socket.display().to_string(),
            waited: WAIT,
        };
        // Until the wait is over: run the attempt; when it ended because the
        // server was away, take a fresh connection after a pause and run it
        // again. Each attempt costs its own time, so the pause is what is left
        // of the budget rather than a fixed step.
        loop {
            match attempt(&mut self.connection) {
                Ok(response) => return Ok(response),
                Err(error) if error.away() => {
                    if !pause_until(deadline) {
                        return Err(restarted());
                    }
                }
                Err(error) => return Err(error),
            }
            loop {
                match Connection::connect(&socket) {
                    Ok(connection) => {
                        self.connection = connection;
                        break;
                    }
                    Err(error) if error.away() => {
                        if !pause_until(deadline) {
                            return Err(restarted());
                        }
                    }
                    Err(error) => return Err(error),
                }
            }
        }
    }

    fn response(&mut self) -> Result<wire::server_message::Message> {
        response_on(&mut self.connection)
    }
}

impl Connection {
    /// Holds the descriptors of a frame just read, for the response that names
    /// them.
    fn hold(&mut self, descriptors: Vec<OwnedFd>) {
        self.descriptors.extend(descriptors);
    }

    /// The descriptors the response just read came with.
    ///
    /// A `resolve` response's files ride on its frame, so they are taken after
    /// the frame is read; the client pairs them with the names in the response.
    pub fn take_descriptors(&mut self) -> Vec<OwnedFd> {
        std::mem::take(&mut self.descriptors).into()
    }
}

fn refused_error(refused: wire::Refusal) -> Error {
    Error::Refused {
        reason: refused.reason,
        upstream: (!refused.upstream.is_empty()).then_some(refused.upstream),
    }
}

fn unexpected(message: Option<wire::server_message::Message>) -> Error {
    Error::Protocol(format!("unexpected response {message:?}"))
}
