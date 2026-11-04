//! Serving on the Unix socket (design section 6.3): a server answers a client
//! that installs a package and lists what is installed, a socket no process
//! answers on is replaced, and a socket a server answers on is not taken from
//! it.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use program_server::Error;
use program_server::builder::Limits;
use program_server::protocol::Client;
use program_server::server::{ProgramServer, ServerConfig};
use program_server::store::Root;
use program_server::toolchain::ToolchainConfig;

/// The manifest of a Triton `rms_norm` package.
const MANIFEST: &str = r#"
op = "rms_norm"
backend = "triton"

[toolchain]
kind = "python"
version = "3.7.1"

[module]
form = "cubin"

[source]
kind = "triton"
files = ["src/rms.py"]
script = "src/compile.py"

[kernel]
architectures = ["sm_100a"]
descriptors = { dimensions = { dtype = ["bf16"] } }
"#;

/// Writes a package into `directory` and answers it.
fn package(directory: &Path) -> std::path::PathBuf {
    std::fs::create_dir_all(directory.join("src")).unwrap();
    std::fs::write(directory.join("package.toml"), MANIFEST).unwrap();
    std::fs::write(directory.join("src/rms.py"), b"# the kernel\n").unwrap();
    std::fs::write(directory.join("src/compile.py"), b"# the driver\n").unwrap();
    directory.to_path_buf()
}

fn config(temporary: &Path) -> ServerConfig {
    ServerConfig {
        socket: temporary.join("program-server.sock"),
        package_root: temporary.join("packages"),
        store_roots: vec![Root {
            path: temporary.join("store"),
            writable: true,
        }],
        upstream: None,
        limits: Limits::default(),
        toolchains: ToolchainConfig {
            nvcc: None,
            python: std::path::PathBuf::from("/python/bin/python"),
            wasmtime: None,
            max_jobs: 1,
        },
        devices: Vec::new(),
    }
}

/// Connects to a socket being served, waiting for it to listen.
fn connect(socket: &Path) -> Client {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match Client::connect(socket) {
            Ok(client) => return client,
            Err(_) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!(
                "the program server never listened on {}: {error}",
                socket.display()
            ),
        }
    }
}

#[test]
fn a_server_answers_an_installation_and_a_listing_over_its_socket() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path());
    // A socket file that no server answers on is what a server that ended
    // leaves; it is replaced.
    std::fs::write(&configuration.socket, b"stale").unwrap();

    let server = Arc::new(ProgramServer::open(&configuration).unwrap());
    std::thread::spawn(move || server.serve());
    let mut client = connect(&configuration.socket);

    let directory = package(&temporary.path().join("package"));
    let installed = client.install(&directory).unwrap();
    assert_eq!(installed.package.op, "rms_norm");
    assert_eq!(installed.package.revision, 1);

    let listing = client.list().unwrap();
    assert_eq!(listing.packages.len(), 1);
    assert_eq!(listing.packages[0].revision, 1);
}

#[test]
fn a_server_does_not_take_a_socket_another_server_answers_on() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path());
    let server = Arc::new(ProgramServer::open(&configuration).unwrap());
    std::thread::spawn(move || server.serve());
    // Wait until it listens.
    drop(connect(&configuration.socket));

    let second = Arc::new(ProgramServer::open(&configuration).unwrap());
    match second.serve() {
        Err(Error::Io(error)) => {
            assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse);
        }
        other => panic!("the second server answered {other:?}"),
    }
}
