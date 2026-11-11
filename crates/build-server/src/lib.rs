//! The Build Server (design section 6): it compiles package revisions, keeps them,
//! and hands them out.
//!
//! One process per host and serving user, beside the Storage Server: a host
//! service (the `service` crate) on [`default_socket`], started as
//! `uniserve build-server serve`, and by a daemon whose host has none. It
//! does three things: compile a package
//! into the modules of its descriptors, keep those modules, and response
//! `install`, `prepare`, `resolve`, `list` and `cover` on its Unix socket. It launches
//! no kernel and allocates no device memory, and it is not on the serving
//! path: a main process asks it for modules when it starts and when a package revision
//! changes, so a Build Server restart delays only those two moments.
//!
//! `python/` holds the Build Server's own Python environment and the
//! compile drivers of [`source`]: Triton and CuTe DSL compilation are the only
//! Python in the system, and they run here, never in the main process.
//!
//! The pieces of the design:
//!
//! - [`server`]: the object model of section 6.2, and `install`, `prepare`,
//!   `resolve`, `list` (section 6.3) and `cover`, which returns the package
//!   whose manifest covers each of an operator's descriptor keys.
//! - [`store`]: the module store of section 5.3, whose paths are made of the
//!   operator's and backend's names, the revision, the digest of its build
//!   inputs, the target and the descriptor key, and whose every directory ends
//!   in a `.committed` mark.
//! - [`inputs`]: what a module is built from (the package's files, the
//!   toolchain, the compile driver and its pins) and the digest that names it.
//! - [`snapshot`]: the files an installation takes of a package, which for a
//!   package cargo builds are its crate and the workspace crates it depends on.
//! - [`build`]: `uniserve build-server build`, which fills a store from
//!   descriptor key lists.
//! - [`builder`]: the compile queue of section 6.2, with its urgencies, its
//!   deduplication by module path and its host-memory limit.
//! - [`source`] and [`toolchain`]: how a package's sources become a module,
//!   by nvcc, Triton, CuTe DSL or a wasm32 build.
//! - [`manifest`]: the `package.toml` of section 5.2.
//! - [`protocol`]: the append-only Protobuf protocol and its client, which
//!   the daemon and the main process use.
//!
//! The package revisions it serves, the slots they fill and their phases are
//! `proto::module`'s (section 5.1), which the daemon and the main process
//! name them with too.

pub mod build;
pub mod builder;
pub mod command;
pub mod inputs;
pub mod manifest;
pub mod protocol;
pub mod server;
pub mod snapshot;
pub mod source;
pub mod store;
pub mod target;
pub mod toolchain;

pub use builder::{BuildState, Limits, Urgency};
pub use manifest::{Manifest, ManifestError, ModuleForm, SourceKind, TargetKind, ToolchainKind};
pub use protocol::{
    Client, CoverRequest, Coverage, Descriptors, Module, ModuleEntry, ModuleFile, PackageRequest,
    Preference, PrepareRequest, Progress, Selector,
};
pub use server::{BuildServer, ServerConfig};
pub use store::{ModulePath, Root, Store};
pub use target::{Device, Target};

/// The socket of this host's Build Server for the calling user:
/// `build-server.sock` in the user's service directory
/// ([`service::runtime_directory`]).
pub fn default_socket() -> std::path::PathBuf {
    service::runtime_directory().join("build-server.sock")
}
pub use toolchain::{Toolchain, Toolchains};

use std::path::PathBuf;

/// Why an operation of the Build Server failed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("build server I/O: {0}")]
    Io(#[from] std::io::Error),
    /// The package's files do not say what [`manifest`] requires.
    #[error("package {package}: {source}")]
    Manifest {
        /// The package's directory.
        package: PathBuf,
        #[source]
        source: Box<ManifestError>,
    },
    /// The request needs the upstream Build Server, and this one is not it.
    #[error("the build server refused: {reason}")]
    Refused {
        reason: String,
        /// Where the upstream Build Server is, when the request needed it.
        upstream: Option<String>,
    },
    /// The peer sent something this build does not expect.
    #[error("the build server responded out of protocol: {0}")]
    Protocol(String),
    /// The Build Server was away for the whole of the wait (design section
    /// 7.5): the client reconnected `attempts` times without a server
    /// responding, so preparation is told a restart outlasted it.
    #[error("the build server at {socket} was away for the whole {waited:?} wait")]
    Restarted {
        /// The socket no server served.
        socket: String,
        /// How long the client waited before it gave up.
        waited: std::time::Duration,
    },
    /// No such package revision was installed.
    #[error("no {op} package of backend {backend} at revision {revision}")]
    UnknownRevision {
        op: String,
        backend: String,
        revision: u32,
    },
    /// No revision of a backend is installed for an operator.
    #[error("no {op} package of backend {backend} is installed")]
    NotInstalled { op: String, backend: String },
    /// The package revision does not cover a descriptor the caller needs.
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
    /// A store build failed somewhere: the keys no installed backend built,
    /// and how many prepares or resolves failed in all. A key a second
    /// backend built still counts its first backend's failure.
    #[error(
        "{failures} build failure(s), and {} of {total} modules did not build: [{}]",
        keys.len(),
        keys.join(", ")
    )]
    Unbuilt {
        /// Each key no backend built, as `operator key`.
        keys: Vec<String>,
        /// How many keys the build was given.
        total: usize,
        /// How many failures the build reported.
        failures: usize,
    },
    /// A toolchain is missing or is not the version the package declares.
    #[error("toolchain {kind}: {message}")]
    Toolchain {
        kind: ToolchainKind,
        message: String,
    },
}

/// What an operation of the Build Server returns.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Whether this error says the Build Server was away: the transport
    /// ended or nothing was listening, which is a restart rather than a
    /// decision (design section 7.5).
    ///
    /// Only the transport's own failures count. Everything the server decided
    /// — a refusal, a build failure, an unknown revision — is the server
    /// having responded, and retrying it would repeat the decision.
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
