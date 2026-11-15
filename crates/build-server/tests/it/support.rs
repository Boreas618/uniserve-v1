//! What the tests share: the GPU targets fixtures are written for, a server
//! configuration over fresh roots, vendor cubin packages, which build on any
//! host because nothing compiles them, and a server serving on a socket of its
//! own.

use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;

use build_server::builder::Limits;
use build_server::protocol::{
    Client, Descriptors, Module, PackageRequest, PrepareRequest, Selector,
};
use build_server::server::{BuildServer, ServerConfig};
use build_server::store::Root;
use build_server::target::Target;
use proto::module::{ModuleSlot, PackageRevision};

/// The target a device of compute capability 10.0 is built for: the target
/// the fixtures' kernel packages build for, and the one the test servers'
/// host builds for.
pub(crate) fn arch() -> Target {
    Target::arch((10, 0)).unwrap()
}

/// The target of compute capability 9.0: one the test servers' host does not
/// build for.
pub(crate) fn other_arch() -> Target {
    Target::arch((9, 0)).unwrap()
}

/// `manifest` with every `{arch}` replaced by [`arch`]'s name, so a fixture
/// names the target its test host builds for without spelling a GPU
/// architecture.
pub(crate) fn for_arch(manifest: &str) -> String {
    manifest.replace("{arch}", arch().as_str())
}

/// The right to open Build Servers in this test process, held until the
/// guard is dropped.
///
/// An upstream server owns its package root through an `flock`, which belongs
/// to the open file description, and opening a server spawns the program
/// that describes the toolchains. A child process holds a duplicate of every
/// descriptor of the process that spawned it until it runs its program, so a
/// lock one test has just released can still be held, by a child another
/// test's thread is spawning, at the moment the first test opens the same
/// root again, and that open is refused. A test that opens a server holds
/// this for its whole run, so those tests run one at a time.
pub(crate) fn exclusive() -> MutexGuard<'static, ()> {
    static SERVERS: Mutex<()> = Mutex::new(());
    // A test that failed while holding the guard leaves nothing to repair.
    SERVERS.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The configuration of an upstream Build Server over fresh directories in
/// `directory`, on a host that builds for [`arch`].
///
/// The package root and the writable store root are named as a store build
/// names them (`build_server::build::PACKAGES` and `MODULES`), so the
/// configuration over a store directory is the one that serves the store.
pub(crate) fn config(directory: &Path) -> ServerConfig {
    ServerConfig {
        package_root: directory.join(build_server::build::PACKAGES),
        store_roots: vec![Root {
            path: directory.join(build_server::build::MODULES),
            writable: true,
        }],
        upstream: None,
        read_only_stores: Vec::new(),
        limits: Limits::default(),
        targets: vec![arch()],
        cache: directory.join("cache"),
    }
}

/// Writes each file under `root`, creating its directories.
pub(crate) fn write(root: &Path, files: &[(&str, &[u8])]) {
    for (name, bytes) in files {
        let path = root.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, bytes).unwrap();
    }
}

/// The bytes of a vendor cubin as a package carries one: the ELF magic and
/// the descriptor key, so the modules of two keys differ. The Build Server
/// stores a cubin as it is, and nothing here loads one.
pub(crate) fn cubin(key: &str) -> Vec<u8> {
    [b"\x7fELF".as_slice(), key.as_bytes()].concat()
}

/// The manifest of a vendor cubin package of `op` and `backend`: one cubin and
/// one launch metadata file per descriptor, which nothing compiles, so any
/// host builds it (design section 5.3). `dimensions` is the inline table of
/// its one descriptor shape, such as `{ dtype = ["bf16"] }`.
pub(crate) fn vendor_manifest(
    op: &str,
    backend: &str,
    priority: u32,
    dimensions: &str,
    architecture: &Target,
) -> String {
    // `{descriptor}` is the manifest's own placeholder for a module's key,
    // which every build replaces; the other names are this fixture's.
    r#"
op = "{op}"
backend = "{backend}"
interface_version = 1
priority = {priority}

[module]
metadata = "metadata_{descriptor}.json"

[source]
kind = "cubin"
cubin = "artifact/module_{descriptor}.cubin"
metadata = "artifact/metadata_{descriptor}.json"

[kernel]
requires = ["{architecture}"]
descriptors = [{ dimensions = {dimensions} }]
"#
    .replace("{op}", op)
    .replace("{backend}", backend)
    .replace("{priority}", &priority.to_string())
    .replace("{architecture}", architecture.as_str())
    .replace("{dimensions}", dimensions)
}

/// The launch metadata a vendor package carries beside each cubin: the kernel
/// symbol the main process launches.
pub(crate) const METADATA: &[u8] = br#"{"name": "uniserve_vendor_kernel"}"#;

/// Writes the vendor cubin package `manifest` into `directory`, carrying the
/// cubin and the launch metadata of each of `keys`, and returns the directory.
pub(crate) fn vendor_package(directory: &Path, manifest: &str, keys: &[&str]) -> PathBuf {
    write(directory, &[("package.toml", manifest.as_bytes())]);
    for key in keys {
        let cubin = cubin(key);
        write(
            directory,
            &[
                (&format!("artifact/module_{key}.cubin"), &cubin),
                (&format!("artifact/metadata_{key}.json"), METADATA),
            ],
        );
    }
    directory.to_path_buf()
}

/// The bytes of a resolved module's file `name`, read through the descriptor
/// it came with.
pub(crate) fn file_of(module: &Module, name: &str) -> Vec<u8> {
    let names: Vec<&str> = module.files.iter().map(|file| file.name.as_str()).collect();
    let mut file = &module
        .files
        .iter()
        .find(|file| file.name == name)
        .unwrap_or_else(|| panic!("the module has no {name}, only {names:?}"))
        .file;
    let mut bytes = Vec::new();
    file.seek(SeekFrom::Start(0)).unwrap();
    file.read_to_end(&mut bytes).unwrap();
    bytes
}

/// A `prepare` of `keys` from the latest revision of `op`'s `backend`, for
/// `targets`.
pub(crate) fn latest(
    op: &str,
    backend: &str,
    keys: &[&str],
    targets: Vec<Target>,
) -> PrepareRequest {
    PrepareRequest {
        targets,
        packages: vec![PackageRequest {
            slot: ModuleSlot::all(op),
            selector: Selector::Latest {
                backend: backend.to_owned(),
                arguments: toml::Table::new(),
            },
        }],
        descriptors: vec![Descriptors {
            slot: ModuleSlot::all(op),
            keys: keys.iter().map(|key| (*key).to_owned()).collect(),
        }],
    }
}

/// Revision `revision` of `backend`, with no arguments.
pub(crate) fn revision(backend: &str, revision: u32) -> PackageRevision {
    PackageRevision {
        backend: backend.to_owned(),
        revision,
        arguments: toml::Table::new(),
    }
}

/// A server serving on its socket, on a thread of its own, until it is
/// stopped.
pub(crate) struct Serving {
    pub(crate) socket: PathBuf,
    /// The other end of the descriptor the server stops on.
    stop: UnixStream,
    /// The server's thread, which returns why serving failed, if it did.
    thread: JoinHandle<Result<(), String>>,
}

/// Binds `socket` and serves `server` on it.
///
/// The socket is bound before the server's thread starts, so it accepts
/// connections as soon as this returns: a client connects without waiting
/// for the thread.
pub(crate) fn serve(server: BuildServer, socket: &Path) -> Serving {
    let (listener, starting) = service::host::bind::<build_server::command::Settings>(socket)
        .unwrap()
        .owned(socket)
        .expect("no other server owns the socket");
    let lock = starting
        .finish(&build_server::readiness(
            socket,
            &build_server::command::Settings::default(),
        ))
        .unwrap();
    let (stop, stopped) = UnixStream::pair().unwrap();
    let server = Arc::new(server);
    let thread = std::thread::spawn(move || {
        server
            .serve_on(listener, lock, &stopped)
            .map_err(|error| error.to_string())
    });
    Serving {
        socket: socket.to_path_buf(),
        stop,
        thread,
    }
}

impl Serving {
    /// A client connected to the server.
    pub(crate) fn client(&self) -> Client {
        Client::connect(&self.socket).unwrap()
    }

    /// Makes the stop descriptor readable and returns what the server
    /// returned.
    pub(crate) fn stop(self) -> Result<(), String> {
        (&self.stop).write_all(&[1]).unwrap();
        self.thread.join().unwrap()
    }

    /// Closes the stop descriptor's other end and returns what the server
    /// returned.
    pub(crate) fn close(self) -> Result<(), String> {
        drop(self.stop);
        self.thread.join().unwrap()
    }
}
