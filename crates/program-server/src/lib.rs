//! The Program Server (design section 6): it compiles programs, keeps them,
//! and hands them out.
//!
//! One process per host, beside the Storage Server, started as
//! `uniserve program-server`. It does three things: compile a program package
//! into the modules of its descriptors, keep those modules, and answer
//! `install`, `prepare`, `resolve` and `list` on its Unix socket. It launches
//! no kernel and allocates no device memory, and it is not on the serving
//! path: a main process asks it for modules when it starts and when a program
//! changes, so a Program Server restart delays only those two moments.
//!
//! `python/` holds the Program Server's own Python environment and the
//! compile drivers of [`source`]: Triton and CuTe DSL compilation are the only
//! Python in the system, and they run here, never in the main process.
//!
//! The pieces of the design:
//!
//! - [`server`]: the object model of section 6.2, and `install`, `prepare`,
//!   `resolve`, `list` (section 6.3).
//! - [`store`]: the module store of section 5.3, whose paths are made of the
//!   operator's and backend's names, the revision, the target and the
//!   descriptor key, and whose every directory ends in a `.committed` mark.
//! - [`builder`]: the compile queue of section 6.2, with its urgencies, its
//!   deduplication by module path and its host-memory limit.
//! - [`source`] and [`toolchain`]: how a package's sources become a module,
//!   by nvcc, Triton, CuTe DSL or a wasm32 build.
//! - [`manifest`]: the `package.toml` of section 5.2.
//! - [`protocol`]: the append-only Protobuf protocol and its client, which
//!   the daemon and the main process use.
//! - [`program`]: [`Program`], the backend name, revision and arguments an
//!   epoch fills each operator with (section 5.1), shared by the daemon and
//!   the main process.

pub mod builder;
pub mod command;
pub mod manifest;
pub mod program;
pub mod protocol;
pub mod server;
pub mod source;
pub mod store;
pub mod target;
pub mod toolchain;

pub use builder::{BuildState, Limits, Urgency};
pub use manifest::{Manifest, ManifestError, ModuleForm, SourceKind, TargetKind, ToolchainKind};
pub use program::{PhaseKind, Program, ProgramSlot};
pub use protocol::{
    Client, Descriptors, Module, ModuleFile, PrepareRequest, Prepared, ProgramRequest, Progress,
    Selector,
};
pub use server::{Installed, Listing, ProgramServer, ServerConfig};
pub use store::{ModulePath, Root, Store};
pub use target::{Device, Devices, Target};
pub use toolchain::{ToolJob, Toolchain, Toolchains};

use std::path::PathBuf;

/// Why an operation of the Program Server failed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("program server I/O: {0}")]
    Io(#[from] std::io::Error),
    /// The package's files do not say what [`manifest`] requires.
    #[error("program package {package}: {source}")]
    Manifest {
        /// The package's directory.
        package: PathBuf,
        #[source]
        source: Box<ManifestError>,
    },
    /// The request needs the upstream Program Server, and this one is not it.
    #[error("the program server refused: {reason}")]
    Refused {
        reason: String,
        /// Where the upstream Program Server is, when the request needed it.
        upstream: Option<String>,
    },
    /// The peer answered something this build does not expect.
    #[error("the program server answered out of protocol: {0}")]
    Protocol(String),
    /// The Program Server was away for the whole of the wait (design section
    /// 7.5): the client reconnected `attempts` times without a server
    /// answering, so preparation is told a restart outlasted it.
    #[error("the program server at {socket} was away for {attempts} attempts")]
    Restarted {
        /// The socket that stayed unanswered.
        socket: String,
        /// How many attempts were made before the wait ended.
        attempts: usize,
    },
    /// No such program was installed.
    #[error("no {op} program of backend {backend} at revision {revision}")]
    UnknownRevision {
        op: String,
        backend: String,
        revision: u32,
    },
    /// No revision of a backend is installed for an operator.
    #[error("no {op} program of backend {backend} is installed")]
    NotInstalled { op: String, backend: String },
    /// The program does not cover a descriptor the caller needs.
    #[error("{op} {backend} revision {revision} covers no descriptor {descriptor}")]
    UnsupportedDescriptor {
        op: String,
        backend: String,
        revision: u32,
        descriptor: String,
    },
    /// A module could not be built or found.
    #[error("building {op} {backend} revision {revision} for {target} {descriptor}: {message}")]
    Build {
        op: String,
        backend: String,
        revision: u32,
        target: String,
        descriptor: String,
        message: String,
    },
    /// A toolchain is missing or is not the version the package declares.
    #[error("toolchain {kind}: {message}")]
    Toolchain {
        kind: ToolchainKind,
        message: String,
    },
}

/// What an operation of the Program Server answers.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Whether this error says the Program Server was away: the transport
    /// ended or nothing was listening, which is a restart rather than a
    /// decision (design section 7.5).
    ///
    /// Only the transport's own failures count. Everything the server decided
    /// — a refusal, a build failure, an unknown revision — is the server
    /// having answered, and retrying it would repeat the decision.
    pub fn away(&self) -> bool {
        matches!(
            self,
            Error::Io(source)
                if matches!(
                    source.kind(),
                    std::io::ErrorKind::NotFound
                        | std::io::ErrorKind::ConnectionRefused
                        | std::io::ErrorKind::ConnectionReset
                        | std::io::ErrorKind::BrokenPipe
                        | std::io::ErrorKind::UnexpectedEof
                        | std::io::ErrorKind::NotConnected
                )
        )
    }
}
