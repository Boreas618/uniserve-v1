//! The Program Server's protocol and its client (design section 6.3).
//!
//! A connection to the Program Server speaks length-delimited Protobuf
//! frames: the client sends one [`wire::Request`], the server answers with
//! [`wire::ServerMessage`] frames until the one that answers the request.
//! AOT builds report themselves as [`Progress`] frames while `prepare` runs;
//! `resolve` reports JIT builds the same way. A `resolve` answer's module
//! files travel as descriptors with `SCM_RIGHTS`, attached to the frame that
//! names them, in the order it lists them.
//!
//! The schema is append-only and lives in `proto/program_server.proto`: a
//! field is never renumbered, retyped or removed, and a new request or answer
//! is a new field of its oneof. The daemon and the main process therefore
//! keep speaking a Program Server of a later build.
//!
//! [`Client`] is the half the daemon and the main process use: the daemon
//! calls [`Client::prepare`] before a deployment starts and before an update,
//! and the main process calls [`Client::resolve`] when it starts and when a
//! program changes.

use std::io::{self, IoSlice, IoSliceMut};
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::path::Path;

use prost::Message;
use rustix::net::{RecvAncillaryBuffer, SendAncillaryBuffer, SendAncillaryMessage};

use crate::builder::BuildState;
use crate::manifest::ModuleForm;
use crate::program::{PhaseKind, Program, ProgramSlot};
use crate::store::ModulePath;
use crate::target::Target;
use crate::{Error, Result};

/// The messages of the protocol, as the build script compiled them.
pub mod wire {
    include!(concat!(env!("OUT_DIR"), "/uniserve.program_server.rs"));
}

/// Descriptors one `sendmsg` carries: the kernel's `SCM_MAX_FD`.
pub const MAX_DESCRIPTORS: usize = 253;

/// Bytes of a frame header: the payload's length and its descriptor count.
const HEADER: usize = 8;

/// Bytes one receive reads at most.
const READ_CHUNK: usize = 64 * 1024;

/// An installed program package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Package {
    pub op: String,
    pub backend: String,
    pub revision: u32,
}

/// The answer to `install`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Installed {
    pub package: Package,
}

/// The descriptors one operator's program must cover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Descriptors {
    pub op: String,
    /// Descriptor keys, in the canonical form the operator's `Display` gives.
    pub keys: Vec<String>,
}

/// How a `prepare` names the program that fills an operator.
#[derive(Clone, Debug, PartialEq)]
pub enum Selector {
    /// The backend's latest installed revision.
    Latest {
        backend: String,
        arguments: toml::Table,
    },
    /// Exactly this program, revision and all.
    Fixed(Program),
}

/// One operator of an epoch and the program that fills it.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramRequest {
    pub slot: ProgramSlot,
    pub selector: Selector,
}

/// What the daemon asks `prepare` for: the programs of a set of operators,
/// and the descriptors their modules must cover.
#[derive(Clone, Debug, PartialEq)]
pub struct PrepareRequest {
    /// The targets to build for (design section 5.3).
    pub targets: Vec<Target>,
    pub programs: Vec<ProgramRequest>,
    pub descriptors: Vec<Descriptors>,
}

/// A program `prepare` selected.
#[derive(Clone, Debug, PartialEq)]
pub struct Chosen {
    pub slot: ProgramSlot,
    pub program: Program,
}

/// The answer to `prepare`: the selected programs, with their revisions.
#[derive(Clone, Debug, PartialEq)]
pub struct Prepared {
    pub programs: Vec<Chosen>,
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
}

/// A module the Program Server resolved: its form, its files, and what the
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
}

/// One module a listing names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListedModule {
    pub target: Target,
    pub descriptor: String,
    pub form: ModuleForm,
}

/// One installed package a listing names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListedPackage {
    pub op: String,
    pub backend: String,
    pub revision: u32,
    pub modules: Vec<ListedModule>,
}

/// The answer to `list`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Listing {
    pub packages: Vec<ListedPackage>,
}

impl From<&Program> for wire::ProgramRef {
    fn from(program: &Program) -> Self {
        wire::ProgramRef {
            backend: program.backend.clone(),
            revision: program.revision,
            arguments: toml::to_string(&program.arguments).unwrap_or_default(),
        }
    }
}

impl TryFrom<wire::ProgramRef> for Program {
    type Error = Error;

    fn try_from(program: wire::ProgramRef) -> Result<Self> {
        Ok(Program {
            backend: program.backend,
            revision: program.revision,
            arguments: arguments(&program.arguments)?,
        })
    }
}

/// Parses a program's arguments, which travel as a TOML table.
fn arguments(text: &str) -> Result<toml::Table> {
    if text.is_empty() {
        return Ok(toml::Table::new());
    }
    toml::from_str(text)
        .map_err(|error| Error::Protocol(format!("a program's arguments are not TOML: {error}")))
}

impl From<PhaseKind> for wire::ProgramPhase {
    fn from(phase: PhaseKind) -> Self {
        match phase {
            PhaseKind::Prefill => wire::ProgramPhase::Prefill,
            PhaseKind::Decode => wire::ProgramPhase::Decode,
            PhaseKind::Encode => wire::ProgramPhase::Encode,
            PhaseKind::Denoise => wire::ProgramPhase::Denoise,
        }
    }
}

impl TryFrom<wire::ProgramPhase> for Option<PhaseKind> {
    type Error = Error;

    fn try_from(phase: wire::ProgramPhase) -> Result<Self> {
        Ok(match phase {
            wire::ProgramPhase::All => None,
            wire::ProgramPhase::Prefill => Some(PhaseKind::Prefill),
            wire::ProgramPhase::Decode => Some(PhaseKind::Decode),
            wire::ProgramPhase::Encode => Some(PhaseKind::Encode),
            wire::ProgramPhase::Denoise => Some(PhaseKind::Denoise),
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
    let mut programs = Vec::with_capacity(request.programs.len());
    for entry in &request.programs {
        let selector = match &entry.selector {
            Selector::Latest { backend, arguments } => {
                wire::program_request::Selector::Backend(wire::Backend {
                    backend: backend.clone(),
                    arguments: toml::to_string(arguments).map_err(|error| {
                        Error::Protocol(format!("a program's arguments are not TOML: {error}"))
                    })?,
                })
            }
            Selector::Fixed(program) => wire::program_request::Selector::Program((program).into()),
        };
        programs.push(wire::ProgramRequest {
            op: entry.slot.op.clone(),
            phase: entry
                .slot
                .phase
                .map_or(wire::ProgramPhase::All, wire::ProgramPhase::from)
                as i32,
            selector: Some(selector),
        });
    }
    let descriptors = request
        .descriptors
        .iter()
        .map(|descriptors| wire::Descriptors {
            op: descriptors.op.clone(),
            keys: descriptors.keys.clone(),
        })
        .collect();
    Ok(wire::Prepare {
        targets: request
            .targets
            .iter()
            .map(|target| target.to_string())
            .collect(),
        programs,
        descriptors,
    })
}

/// A connection to the Program Server: framed messages, and the descriptors
/// that travel with them.
#[derive(Debug)]
pub struct Connection {
    stream: UnixStream,
    inbox: Vec<u8>,
    descriptors: Vec<OwnedFd>,
}

impl Connection {
    /// Opens a blocking connection to a Program Server's socket.
    pub fn connect(socket: &Path) -> Result<Connection> {
        Ok(Connection::new(UnixStream::connect(socket)?))
    }

    /// Wraps a stream, as the server does for an accepted connection.
    pub fn new(stream: UnixStream) -> Connection {
        Connection {
            stream,
            inbox: Vec::new(),
            descriptors: Vec::new(),
        }
    }

    /// The stream, for a server that hands connections between threads.
    pub fn stream(&self) -> &UnixStream {
        &self.stream
    }

    /// Sends one frame with the descriptors that belong to it.
    ///
    /// Descriptors ride on the frame's first byte, so a receiver that reads a
    /// frame exactly never gets another frame's descriptors.
    pub fn send<M: Message>(&mut self, message: &M, descriptors: &[OwnedFd]) -> Result<()> {
        if descriptors.len() > MAX_DESCRIPTORS {
            return Err(Error::Protocol(format!(
                "a frame carries at most {MAX_DESCRIPTORS} descriptors, not {}",
                descriptors.len()
            )));
        }
        let payload = message.encode_to_vec();
        let length = u32::try_from(payload.len())
            .map_err(|_| Error::Protocol("a frame exceeds 4 GiB".into()))?;
        let count = u32::try_from(descriptors.len())
            .map_err(|_| Error::Protocol("too many descriptors".into()))?;
        let mut bytes = Vec::with_capacity(HEADER + payload.len());
        bytes.extend_from_slice(&length.to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
        bytes.extend_from_slice(&payload);

        let mut sent = 0;
        while sent < bytes.len() {
            let borrowed: Vec<BorrowedFd<'_>> = descriptors.iter().map(AsFd::as_fd).collect();
            let mut space =
                [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_DESCRIPTORS))];
            let mut control = SendAncillaryBuffer::new(&mut space);
            if sent == 0 && !borrowed.is_empty() {
                control.push(SendAncillaryMessage::ScmRights(&borrowed));
            }
            let written = loop {
                match rustix::net::sendmsg(
                    &self.stream,
                    &[IoSlice::new(&bytes[sent..])],
                    &mut control,
                    rustix::net::SendFlags::NOSIGNAL,
                ) {
                    Ok(written) => break written,
                    Err(rustix::io::Errno::INTR) => continue,
                    Err(error) => return Err(Error::Io(error.into())),
                }
            };
            drop(borrowed);
            if written == 0 {
                return Err(Error::Io(io::Error::from(io::ErrorKind::WriteZero)));
            }
            sent += written;
        }
        Ok(())
    }

    /// Receives one frame and the descriptors that came with it.
    pub fn receive<M: Message + Default>(&mut self) -> Result<(M, Vec<OwnedFd>)> {
        loop {
            if let Some(frame) = self.next_frame()? {
                return Ok(frame);
            }
            self.fill()?;
        }
    }

    /// The whole frame once its bytes and descriptors have arrived.
    fn next_frame<M: Message + Default>(&mut self) -> Result<Option<(M, Vec<OwnedFd>)>> {
        if self.inbox.len() < HEADER {
            return Ok(None);
        }
        let length = u32::from_le_bytes(self.inbox[..4].try_into().unwrap_or([0; 4])) as usize;
        let count = u32::from_le_bytes(self.inbox[4..HEADER].try_into().unwrap_or([0; 4])) as usize;
        if self.inbox.len() < HEADER + length || self.descriptors.len() < count {
            return Ok(None);
        }
        let message = M::decode(&self.inbox[HEADER..HEADER + length])
            .map_err(|error| Error::Protocol(format!("undecodable frame: {error}")))?;
        self.inbox.drain(..HEADER + length);
        let descriptors = self.descriptors.drain(..count).collect();
        Ok(Some((message, descriptors)))
    }

    /// Reads the bytes the current frame still needs, and the descriptors
    /// that ride on its first byte.
    fn fill(&mut self) -> Result<()> {
        let needed = self.frame_bytes().saturating_sub(self.inbox.len()).max(1);
        let mut buffer = vec![0_u8; needed.min(READ_CHUNK)];
        let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_DESCRIPTORS))];
        let mut control = RecvAncillaryBuffer::new(&mut space);
        let message = loop {
            match rustix::net::recvmsg(
                &self.stream,
                &mut [IoSliceMut::new(&mut buffer)],
                &mut control,
                rustix::net::RecvFlags::CMSG_CLOEXEC,
            ) {
                Ok(message) => break message,
                Err(rustix::io::Errno::INTR) => continue,
                Err(error) => return Err(Error::Io(error.into())),
            }
        };
        if message.flags.contains(rustix::net::ReturnFlags::CTRUNC) {
            return Err(Error::Protocol(
                "a frame carried more descriptors than one receive takes".into(),
            ));
        }
        for received in control.drain() {
            if let rustix::net::RecvAncillaryMessage::ScmRights(descriptors) = received {
                self.descriptors.extend(descriptors);
            }
        }
        if message.bytes == 0 {
            return Err(Error::Io(io::Error::from(io::ErrorKind::UnexpectedEof)));
        }
        let received = buffer[..message.bytes].to_vec();
        self.inbox.extend_from_slice(&received);
        Ok(())
    }

    /// The bytes of the frame being received: the header, or the whole frame
    /// once the header is here.
    fn frame_bytes(&self) -> usize {
        if self.inbox.len() < HEADER {
            return HEADER;
        }
        let length = u32::from_le_bytes(self.inbox[..4].try_into().unwrap_or([0; 4])) as usize;
        HEADER + length
    }
}

/// A connection to a Program Server, as its clients speak it: the daemon
/// prepares programs with it, and the main process resolves them.
#[derive(Debug)]
pub struct Client {
    connection: Connection,
}

impl Client {
    /// Opens a connection to the Program Server at `socket`.
    pub fn connect(socket: &Path) -> Result<Client> {
        Ok(Client {
            connection: Connection::connect(socket)?,
        })
    }

    /// Wraps an open connection.
    pub fn new(connection: Connection) -> Client {
        Client { connection }
    }

    /// Installs a program package and answers its revision.
    ///
    /// Only the upstream Program Server installs; another one refuses and
    /// names the upstream, which the error carries.
    pub fn install(&mut self, package: &Path) -> Result<Installed> {
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
        match self.answer()? {
            wire::server_message::Message::Installed(installed) => Ok(Installed {
                package: installed
                    .package
                    .map(|package| Package {
                        op: package.op,
                        backend: package.backend,
                        revision: package.revision,
                    })
                    .ok_or_else(|| Error::Protocol("installed without a package".into()))?,
            }),
            wire::server_message::Message::Refused(refused) => Err(refused_error(refused)),
            other => Err(unexpected(Some(other))),
        }
    }

    /// Selects an epoch's programs and has every module of their descriptors
    /// built, reporting each build through `progress` (design section 6.3).
    ///
    /// This waits until every missing module is committed; a module that
    /// cannot be built fails the call with the build's log.
    pub fn prepare(
        &mut self,
        request: &PrepareRequest,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Prepared> {
        self.connection.send(
            &wire::Request {
                request: Some(wire::request::Request::Prepare(prepare_message(request)?)),
            },
            &[],
        )?;
        loop {
            match self.answer()? {
                wire::server_message::Message::Progress(step) => progress(step.into()),
                wire::server_message::Message::Prepared(prepared) => {
                    let mut chosen = Vec::with_capacity(prepared.programs.len());
                    for program in prepared.programs {
                        let slot = ProgramSlot {
                            op: program.op,
                            phase: wire::ProgramPhase::try_from(program.phase)
                                .map_err(|_| Error::Protocol("unknown phase".into()))?
                                .try_into()?,
                        };
                        let program = program.program.ok_or_else(|| {
                            Error::Protocol("a chosen program carries no program".into())
                        })?;
                        chosen.push(Chosen {
                            slot,
                            program: program.try_into()?,
                        });
                    }
                    return Ok(Prepared { programs: chosen });
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
        program: &Program,
        target: &Target,
        descriptor: &str,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Module> {
        self.connection.send(
            &wire::Request {
                request: Some(wire::request::Request::Resolve(wire::Resolve {
                    op: op.to_owned(),
                    program: Some(program.into()),
                    target: target.to_string(),
                    descriptor: descriptor.to_owned(),
                })),
            },
            &[],
        )?;
        loop {
            match self.answer()? {
                wire::server_message::Message::Progress(step) => progress(step.into()),
                wire::server_message::Message::Resolved(resolved) => {
                    let form = wire::ModuleForm::try_from(resolved.form)
                        .map_err(|_| Error::Protocol("unknown module form".into()))?
                        .try_into()?;
                    let descriptors = self.connection.take_descriptors();
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
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    return Ok(Module {
                        form,
                        files,
                        metadata: (!resolved.metadata.is_empty()).then_some(resolved.metadata),
                        entries: resolved.entries.into_iter().collect(),
                    });
                }
                wire::server_message::Message::Refused(refused) => {
                    return Err(refused_error(refused));
                }
                other => return Err(unexpected(Some(other))),
            }
        }
    }

    /// Lists the installed packages, their revisions and their modules.
    pub fn list(&mut self) -> Result<Listing> {
        self.connection.send(
            &wire::Request {
                request: Some(wire::request::Request::List(wire::List {})),
            },
            &[],
        )?;
        match self.answer()? {
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
                        modules.push(ListedModule {
                            target: Target::new(module.target),
                            descriptor: module.descriptor,
                            form,
                        });
                    }
                    packages.push(ListedPackage {
                        op: package.op,
                        backend: package.backend,
                        revision: package.revision,
                        modules,
                    });
                }
                Ok(Listing { packages })
            }
            wire::server_message::Message::Refused(refused) => Err(refused_error(refused)),
            other => Err(unexpected(Some(other))),
        }
    }

    /// The next frame, with the descriptors that rode on it held for the
    /// answer it belongs to.
    ///
    /// Progress reports are frames like any other: the requests that report
    /// builds read them in their own loop, so dropping them here would lose
    /// every report a caller asked for.
    fn answer(&mut self) -> Result<wire::server_message::Message> {
        let (message, descriptors) = self.connection.receive::<wire::ServerMessage>()?;
        let answer = message.message.ok_or_else(|| unexpected(None))?;
        self.connection.hold(descriptors);
        Ok(answer)
    }
}

impl Connection {
    /// Holds the descriptors of a frame just read, for the answer that names
    /// them.
    fn hold(&mut self, descriptors: Vec<OwnedFd>) {
        self.descriptors.extend(descriptors);
    }

    /// The descriptors the answer just read came with.
    ///
    /// A `resolve` answer's files ride on its frame, so they are taken after
    /// the frame is read; the client pairs them with the names in the answer.
    pub fn take_descriptors(&mut self) -> Vec<OwnedFd> {
        std::mem::take(&mut self.descriptors)
    }
}

fn refused_error(refused: wire::Refused) -> Error {
    Error::Refused {
        reason: refused.reason,
        upstream: (!refused.upstream.is_empty()).then_some(refused.upstream),
    }
}

fn unexpected(message: Option<wire::server_message::Message>) -> Error {
    Error::Protocol(format!("unexpected answer {message:?}"))
}
