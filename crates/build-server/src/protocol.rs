//! The Build Server's protocol and its client (design section 6.3).
//!
//! A connection to the Build Server is the workspace's one connection
//! (`service::Connection`) carrying Protobuf messages, one frame each:
//! the client sends one [`wire::Request`], the server responds with
//! [`wire::ServerMessage`] frames until the one that responds to the request.
//! AOT builds report themselves as [`Progress`] frames while `prepare` runs;
//! `resolve` reports JIT builds the same way. A `resolve` response's module
//! files travel as descriptors with `SCM_RIGHTS`, attached to the frame that
//! names them, in the order it lists them.
//!
//! The schema lives in `proto/build_server.proto` and is one release's: the
//! Build Server and its clients of one release speak it.
//!
//! [`Client`] is the half the daemon and the main process use: the daemon
//! calls [`Client::prepare`] before a deployment starts and before an update,
//! and the main process calls [`Client::resolve`] when it starts and when a
//! package revision changes, and [`Client::cover`] to learn which installed package
//! builds each descriptor key of an operator the deployment does not name.

use std::io;
use std::os::fd::OwnedFd;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

use prost::Message;
use service::Connection;

use crate::builder::BuildState;
pub use crate::inputs::Origin;
use crate::manifest::ModuleForm;
use crate::store::ModulePath;
use crate::target::Target;
use crate::{Error, Result};
use proto::module::{ModuleSlot, PackageRevision, Phase, phase_of};

/// The messages of the protocol, as the build script compiled them.
pub mod wire {
    include!(concat!(env!("OUT_DIR"), "/uniserve.build_server.rs"));
}

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
    /// An executable module's entry: the name among [`Self::files`] of the
    /// executable its process runs as, which the package's manifest names
    /// (`[source] binary`); `None` for every other form.
    pub executable: Option<String>,
    /// Where the module's package revision came from: the commit of the tree
    /// it was installed from and whether that tree was clean.
    pub origin: Origin,
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
    pub phase: Option<Phase>,
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

/// A slot's phase as the wire states it: the phase's id, absent for a slot of every phase.
pub(crate) fn phase_to_wire(phase: Option<Phase>) -> Option<u32> {
    proto::module::phase_to_wire(phase)
}

/// The phase a wire id states ([`phase_to_wire`]).
///
/// # Errors
///
/// `Protocol` for an id no phase has.
pub(crate) fn phase_from_wire(phase: Option<u32>) -> Result<Option<Phase>> {
    phase_of(phase).map_err(|error| Error::Protocol(error.to_string()))
}

/// The slot an operator, a wire phase and a wire part name: an empty part is
/// a slot that holds the operator's every descriptor.
pub(crate) fn slot_of(op: String, phase: Option<u32>, part: String) -> Result<ModuleSlot> {
    Ok(ModuleSlot {
        op,
        phase: phase_from_wire(phase)?,
        part: (!part.is_empty()).then_some(part),
    })
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

/// Reads one response frame on `connection`, with the descriptors that rode on it: a `resolve`
/// response's files, which the client pairs with the names the response lists.
fn response_on(
    connection: &mut Connection,
) -> Result<(wire::server_message::Message, Vec<OwnedFd>)> {
    let (message, descriptors) = connection
        .recv_with::<wire::ServerMessage>(Duration::MAX)?
        .ok_or_else(|| unexpected(None))?;
    let response = message.message.ok_or_else(|| unexpected(None))?;
    Ok((response, descriptors))
}

/// Connects to the Build Server listening at `socket`.
///
/// # Errors
///
/// Fails when nothing listens there.
pub fn connect(socket: &Path) -> Result<Connection> {
    Ok(Connection::new(UnixStream::connect(socket)?)?)
}

/// The messages of the protocol on the workspace's one connection: each a frame
/// (`service::frame`) whose payload is the message's Protobuf encoding, with the descriptors the
/// message names riding on it, so a receiver that reads a frame never gets another frame's
/// descriptors.
macro_rules! messages {
    ($($message:ty),* $(,)?) => {
        $(
            impl service::Message for $message {

                fn encode(&self) -> io::Result<Vec<u8>> {
                    Ok(Message::encode_to_vec(self))
                }

                fn decode(payload: &[u8]) -> io::Result<Self> {
                    <Self as Message>::decode(payload).map_err(|error| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("undecodable frame: {error}"),
                        )
                    })
                }
            }
        )*
    };
}

messages!(wire::Request, wire::ServerMessage);

/// A connection to a Build Server, as its clients speak it: the daemon
/// prepares package revisions with it, and the main process resolves them.
///
/// A Build Server that stops while a request is in flight ends the request with
/// the connection's error: the Build Server is not on the serving path, and the
/// daemon starts the host's Build Server again when it next prepares an epoch.
pub struct Client {
    connection: Connection,
}

impl Client {
    /// Opens a connection to the Build Server at `socket`.
    pub fn connect(socket: &Path) -> Result<Client> {
        Ok(Client {
            connection: connect(socket)?,
        })
    }

    /// Wraps an open connection.
    pub fn new(connection: Connection) -> Client {
        Client { connection }
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
        self.connection.deliver(&wire::Request {
            request: Some(wire::request::Request::Install(wire::Install { package })),
        })?;
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

    /// Installs a policy package's program crate `programs/<directory>`, whose
    /// files are `files` (each by its path below that directory), and returns
    /// its revision: the latest one when it holds the same build inputs
    /// (`crate::BuildServer::install_program`).
    ///
    /// Only the upstream Build Server installs; another one refuses and names
    /// the upstream, which the error carries.
    pub fn install_program(
        &mut self,
        directory: &str,
        files: &std::collections::BTreeMap<std::path::PathBuf, Vec<u8>>,
    ) -> Result<Package> {
        let files = files
            .iter()
            .map(|(path, contents)| {
                let path = path.to_str().ok_or_else(|| {
                    Error::Protocol(format!(
                        "the program file path {} is not UTF-8",
                        path.display()
                    ))
                })?;
                Ok(wire::ProgramFile {
                    path: path.to_owned(),
                    contents: contents.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.connection.deliver(&wire::Request {
            request: Some(wire::request::Request::InstallProgram(
                wire::InstallProgram {
                    directory: directory.to_owned(),
                    files,
                },
            )),
        })?;
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
        let connection = &mut self.connection;
        connection.deliver(&wire::Request {
            request: Some(wire::request::Request::Prepare(prepare_message(request)?)),
        })?;
        loop {
            match response_on(connection)?.0 {
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
        let connection = &mut self.connection;
        connection.deliver(&wire::Request {
            request: Some(wire::request::Request::Resolve(wire::Resolve {
                op: op.to_owned(),
                package: Some(revision.into()),
                target: target.to_string(),
                descriptor: descriptor.to_owned(),
            })),
        })?;
        loop {
            let (response, descriptors) = response_on(connection)?;
            match response {
                wire::server_message::Message::Progress(step) => progress(step.into()),
                wire::server_message::Message::Resolved(resolved) => {
                    let form = wire::ModuleForm::try_from(resolved.form)
                        .map_err(|_| Error::Protocol("unknown module form".into()))?
                        .try_into()?;
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
                        executable: (!resolved.executable.is_empty())
                            .then_some(resolved.executable),
                        origin: Origin {
                            commit: resolved.commit,
                            clean: resolved.clean,
                        },
                    });
                }
                wire::server_message::Message::Refused(refused) => {
                    return Err(refused_error(refused));
                }
                other => return Err(unexpected(Some(other))),
            }
        }
    }

    /// Lists every installed package revision with its origin and its modules.
    pub fn list(&mut self) -> Result<Vec<(Package, Origin, Vec<ModuleEntry>)>> {
        self.connection.deliver(&wire::Request {
            request: Some(wire::request::Request::List(wire::List {})),
        })?;
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
                        Origin {
                            commit: package.commit,
                            clean: package.clean,
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
        self.connection.deliver(&wire::Request {
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
        })?;
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

    /// The next frame of the response to the request this client sent.
    fn response(&mut self) -> Result<wire::server_message::Message> {
        response_on(&mut self.connection).map(|(response, _)| response)
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
