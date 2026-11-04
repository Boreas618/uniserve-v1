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
kind = "triton"
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
            wasm: None,
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
    std::thread::spawn(move || {
        if let Err(error) = server.serve() {
            eprintln!("the server stopped: {error}");
        }
    });
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
    std::thread::spawn(move || {
        if let Err(error) = server.serve() {
            eprintln!("the server stopped: {error}");
        }
    });
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

/// Copies a package, files and directories, the way `install` leaves one.
fn copy_tree(source: &Path, destination: &Path) {
    std::fs::create_dir_all(destination).unwrap();
    for entry in std::fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let target = destination.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// Asks for one program of `backend` with no descriptors, which commits no
/// modules: the call is the wire round trip alone.
fn request(backend: &str) -> program_server::protocol::PrepareRequest {
    program_server::protocol::PrepareRequest {
        targets: Vec::new(),
        programs: vec![program_server::protocol::ProgramRequest {
            slot: program_server::ProgramSlot::all("rms_norm"),
            selector: program_server::protocol::Selector::Latest {
                backend: backend.to_owned(),
                arguments: toml::Table::new(),
            },
        }],
        descriptors: Vec::new(),
    }
}

/// Serves `configuration` on a thread until the flag it answers is raised.
fn serving(
    configuration: &ServerConfig,
) -> (Arc<std::sync::atomic::AtomicBool>, std::thread::JoinHandle<()>) {
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let server = Arc::new(ProgramServer::open(configuration).unwrap());
    let thread = std::thread::spawn({
        let stop = Arc::clone(&stop);
        move || {
            if let Err(error) = server.serve_until(stop) {
                eprintln!("the server stopped: {error}");
            }
        }
    });
    (stop, thread)
}

#[test]
fn a_prepare_waits_through_a_server_restart_and_answers_from_the_new_one() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path());

    // A first server accepts one connection and goes away with it: the client's
    // connection dies, nothing listens on the path, and the next server is what
    // the call has to wait for.
    let first = std::os::unix::net::UnixListener::bind(&configuration.socket).unwrap();
    let mut client = Client::connect(&configuration.socket).unwrap();
    drop(first.accept().unwrap());
    drop(first);
    let _ = std::fs::remove_file(&configuration.socket);

    // The package the restarted server will find, installed in its root the way
    // `install` leaves it.
    let source = package(&temporary.path().join("package"));
    let installed = configuration.package_root.join("rms_norm/triton/1");
    copy_tree(&source, &installed);

    let restart = {
        let socket_directory = temporary.path().to_path_buf();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            serving(&config(&socket_directory))
        })
    };

    let started = Instant::now();
    let prepared = client
        .prepare(&request("triton"), &mut |_| {})
        .expect("the prepare waits for the restarted server");
    assert_eq!(prepared.programs.len(), 1);
    assert!(
        started.elapsed() >= Duration::from_millis(250),
        "the call answered after {:?}, so it did not wait",
        started.elapsed()
    );

    let (second_stop, second_thread) = restart.join().unwrap();
    second_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = second_thread.join();
}

#[test]
fn a_refusal_is_not_retried_through_a_restart() {
    let temporary = tempfile::tempdir().unwrap();
    let configuration = config(temporary.path());
    let (stop, thread) = serving(&configuration);
    let mut client = connect(&configuration.socket);
    let directory = package(&temporary.path().join("package"));
    client.install(&directory).unwrap();

    // The server answers that this backend is not installed: it decided, so the
    // client returns the decision rather than waiting for a restart.
    let started = Instant::now();
    let refused = client
        .prepare(&request("cuda"), &mut |_| {})
        .expect_err("no cuda backend was installed");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "the refusal took {:?}, so it was retried",
        started.elapsed()
    );
    assert!(
        matches!(
            refused,
            Error::UnknownRevision { .. } | Error::Refused { .. } | Error::Protocol(_)
        ),
        "the refusal was {refused:?}"
    );
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    thread.join().unwrap();
}
