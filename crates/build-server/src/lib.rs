//! The Build Server (design section 6): it compiles package revisions, keeps them,
//! and hands them out.
//!
//! One process per host and serving user, beside the Storage Server: a host
//! service (the `service` crate) on [`default_socket`], started as
//! `uniserve build-server serve`, and by a daemon whose host has none. It
//! does three things: compile a package into the modules of its descriptors,
//! keep those modules, and answer `install`, `prepare`, `resolve`, `list` and
//! `cover` on its Unix socket. It launches no kernel and allocates no device
//! memory, and it is not on the serving path: a main process asks it for
//! modules when it starts and when a package revision changes.
//!
//! `python/` holds the Build Server's own Python environment: its toolchain
//! description and the compile drivers of [`source`]. Triton, CuTe DSL and
//! CUDA compilation are the only Python in the system, and they run there,
//! never in the main process.
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
//!   package cargo builds is its whole workspace, and the vendored sources of
//!   its lockfile.
//! - [`build`]: `uniserve build-server build`, which fills a store from
//!   descriptor key lists.
//! - [`builder`]: the compile queue of section 6.2, with its urgencies, its
//!   deduplication by module path and its host-memory limit.
//! - [`source`] and [`toolchain`]: how a package's sources become a module,
//!   through a build module's typed compile request or a cargo build, and the
//!   declared toolchains that compile them.
//! - [`target`]: what a module is compiled for, and each GPU target's
//!   hardware facts.
//! - [`manifest`]: the `package.toml` of section 5.2.
//! - [`protocol`]: the Protobuf protocol and its client, which the daemon and
//!   the main process use.
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
pub use manifest::{Manifest, ManifestError, ModuleForm, SourceKind, ToolchainKind};
pub use protocol::{
    Client, CoverRequest, Coverage, Descriptors, Module, ModuleEntry, ModuleFile, PackageRequest,
    Preference, PrepareRequest, Progress, Selector,
};
pub use server::{BuildServer, ServerConfig};
pub use store::{ModulePath, Root, Store};
pub use target::Target;
pub use toolchain::Toolchains;

/// The socket of this host's Build Server for the calling user: `build-server.sock` in the
/// user's runtime directory ([`service::RuntimeDirectory`]), where the running server also
/// publishes its readiness record (`service::host::running`).
///
/// # Errors
///
/// Fails when the user has no runtime directory.
pub fn default_socket() -> std::io::Result<std::path::PathBuf> {
    Ok(service::RuntimeDirectory::of_user()?.socket("build-server"))
}

/// A Build Server's readiness record: its socket (`build`).
pub type Readiness = service::Readiness;

/// The readiness record of a Build Server of this process serving on `socket`.
pub fn readiness(socket: &std::path::Path) -> Readiness {
    service::Readiness::of_process(
        service::State::Serving,
        None,
        service::Endpoints {
            build: Some(service::Endpoint::unix(socket)),
            ..service::Endpoints::default()
        },
        (),
    )
}

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
