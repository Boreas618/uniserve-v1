//! Servers bind their listeners when they start, accept port 0, and report what they bound in
//! their readiness record on the descriptor their caller passes (`--ready-fd`); a host service
//! answers `status` and `stop` over its socket.

use std::io::BufRead as _;
use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
use std::os::unix::process::CommandExt as _;
use std::path::Path;
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

use service::{Endpoint, Readiness, ReadyReader, State};

/// How long a server of this test takes to bind and serve: no model, no device.
const START: Duration = Duration::from_secs(60);

/// The descriptor number the readiness pipe's write end takes in the child.
const READY_FD: i32 = 3;

/// A pipe whose ends are close-on-exec.
fn pipe() -> (OwnedFd, OwnedFd) {
    let mut ends = [0; 2];
    // SAFETY: pipe2 fills the two descriptors it creates.
    let created = unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_CLOEXEC) };
    assert_eq!(created, 0);
    // SAFETY: pipe2 returned two new descriptors this test owns.
    unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) }
}

/// A policy package in `directory` whose `config.toml` is `text`; returns its directory.
fn package(directory: &Path, text: &str) -> std::path::PathBuf {
    let package = directory.join("model");
    std::fs::create_dir_all(&package).unwrap();
    std::fs::write(package.join("config.toml"), text).unwrap();
    package
}

/// `uniserve ARGUMENTS --ready-fd 3`, the readiness pipe's write end at descriptor 3, under the
/// runtime and state directory `runtime`, with no device visible and its output to `stdout`;
/// returns the child and the pipe's read end.
fn spawn_ready(runtime: &Path, arguments: &[&str], stdout: Stdio) -> (Child, ReadyReader) {
    let (reader, writer) = pipe();
    let write_end = writer.as_raw_fd();
    let mut command = Command::new(env!("CARGO_BIN_EXE_uniserve"));
    command
        .args(arguments)
        .args(["--ready-fd", &READY_FD.to_string()])
        .env("XDG_RUNTIME_DIR", runtime)
        .env("XDG_STATE_HOME", runtime)
        .env("CUDA_VISIBLE_DEVICES", "")
        .stdout(stdout);
    // SAFETY: dup2 is async-signal-safe; the duplicate is not close-on-exec, so the child
    // inherits it at its number.
    unsafe {
        command.pre_exec(move || {
            if libc::dup2(write_end, READY_FD) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let child = command.spawn().unwrap();
    drop(writer);
    (child, ReadyReader::new(reader))
}

/// `uniserve ARGUMENTS` under the runtime directory `runtime`, run to its end.
fn run(runtime: &Path, arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_uniserve"))
        .args(arguments)
        .env("XDG_RUNTIME_DIR", runtime)
        .output()
        .unwrap()
}

/// A TCP endpoint's port, which a listener bound on port 0 states as the one it took.
fn bound_port(endpoint: Option<&Endpoint>) -> u16 {
    endpoint.and_then(Endpoint::port).expect("a TCP endpoint")
}

/// A Storage Server whose settings ask for TCP port 0 reports the socket and the TCP endpoint it
/// bound once it serves; `status` answers the same record over its socket, and `stop` returns
/// once the server ended, leaving no server owning the socket.
#[test]
fn a_storage_server_reports_what_it_bound_and_stops_on_request() {
    let runtime = tempfile::tempdir().unwrap();
    let socket = runtime.path().join("storage-server.sock");
    let settings = runtime.path().join("storage-server.toml");
    std::fs::write(&settings, "[server]\ntcp = \"127.0.0.1:0\"\n").unwrap();
    let socket_text = socket.to_str().unwrap();

    let (mut server, mut ready) = spawn_ready(
        runtime.path(),
        &[
            "storage-server",
            "serve",
            "--socket",
            socket_text,
            "--settings",
            settings.to_str().unwrap(),
        ],
        Stdio::inherit(),
    );
    let record: Readiness<serde_json::Value> = ready.wait(State::Serving, START).unwrap();
    assert_eq!(record.pid, server.id());
    assert_eq!(record.endpoints.storage, Some(Endpoint::unix(&socket)));
    assert_ne!(bound_port(record.endpoints.peer.as_ref()), 0);
    assert_eq!(record.settings["server"]["tcp"], "127.0.0.1:0");

    let status = run(
        runtime.path(),
        &["storage-server", "status", "--socket", socket_text],
    );
    assert!(status.status.success(), "{status:?}");
    let answered: Readiness<serde_json::Value> = Readiness::parse(
        String::from_utf8(status.stdout)
            .unwrap()
            .trim_end()
            .as_bytes(),
    )
    .unwrap();
    assert_eq!(answered, record);

    let stopped = run(
        runtime.path(),
        &["storage-server", "stop", "--socket", socket_text],
    );
    assert!(stopped.status.success(), "{stopped:?}");
    assert!(server.wait().unwrap().success());
    assert!(!socket.exists());
    assert!(
        service::host::running::<serde_json::Value>(&socket)
            .unwrap()
            .is_none()
    );
}

/// A daemon whose package asks for port 0 binds its client listener and its admin sockets, in
/// an instance directory of its own inside the runtime directory, before its deployment starts,
/// and states them in its readiness record. Its admin socket answers a query of its observation
/// store while the deployment comes up, and its instance directory goes with it.
#[test]
fn a_daemon_reports_the_listeners_it_bound_on_port_0() {
    let runtime = tempfile::tempdir().unwrap();
    let package = package(
        runtime.path(),
        "model = \"org/model\"\n[admin_config]\nport = 0\n",
    );
    let (mut daemon, mut ready) = spawn_ready(
        runtime.path(),
        &["serve", package.to_str().unwrap()],
        Stdio::inherit(),
    );

    let listening: Readiness = ready.wait(State::Listening, START).unwrap();
    let instance = listening.instance.clone().expect("an instance directory");
    assert_eq!(
        instance.parent(),
        Some(runtime.path().join("uniserve").as_path())
    );
    assert_ne!(bound_port(listening.endpoints.client.as_ref()), 0);
    let admin = listening.endpoints.admin.clone().expect("an admin socket");
    assert_eq!(admin, Endpoint::unix(instance.join("admin.sock")));

    let mut stream = std::os::unix::net::UnixStream::connect(admin.path().unwrap()).unwrap();
    std::io::Write::write_all(
        &mut stream,
        b"GET /admin/query?q=SELECT%201%20AS%20one HTTP/1.1\r\nHost: admin\r\n\
          Connection: close\r\n\r\n",
    )
    .unwrap();
    let mut response = String::new();
    std::io::Read::read_to_string(&mut stream, &mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");

    // SAFETY: a signal to this test's own child.
    let signalled = unsafe { libc::kill(daemon.id() as libc::pid_t, libc::SIGTERM) };
    assert_eq!(signalled, 0);
    assert!(daemon.wait().unwrap().success());
    assert!(!instance.exists());
}

/// A package that names no model is refused before the daemon binds anything, naming what it
/// lacks.
#[test]
fn a_package_without_a_model_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let package = package(directory.path(), "[admin_config]\nport = 0\n");
    let refused = Command::new(env!("CARGO_BIN_EXE_uniserve"))
        .arg("serve")
        .arg(&package)
        .env("XDG_STATE_HOME", directory.path())
        .output()
        .unwrap();
    assert!(!refused.status.success());
    let message =
        String::from_utf8_lossy(&refused.stderr) + String::from_utf8_lossy(&refused.stdout);
    assert!(message.contains("names no model"), "{message}");
}

/// How long a daemon that serves no deployment yet takes to end once it was sent SIGTERM: far
/// less than the join timeout (`daemon::JOIN_TIMEOUT`) a leader awaits its other nodes for.
const STOP: Duration = Duration::from_secs(30);

/// A leader of a deployment over two hosts that is sent SIGTERM while it awaits the other host's
/// daemon ends at once, and cleanly, rather than waiting out the join timeout: no other node
/// joins, and no deployment starts.
#[test]
fn a_leader_stopped_while_it_awaits_its_other_nodes_ends() {
    let runtime = tempfile::tempdir().unwrap();
    let host = rustix::system::uname()
        .nodename()
        .to_string_lossy()
        .into_owned();
    // A deployment of several nodes names a fixed join port, which the other nodes dial: one
    // no listener of this host holds now, which the kernel picks.
    let join = std::net::TcpListener::bind("0.0.0.0:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    // The second host is one no node finds itself by, so the leader awaits it for as long as
    // the join timeout lasts.
    let package = package(
        runtime.path(),
        &format!(
            "model = \"org/model\"\nhosts = [\"{host}\", \"member.invalid\"]\n\
             [admin_config]\nport = 0\njoin = {join}\n"
        ),
    );
    let (mut daemon, mut ready) = spawn_ready(
        runtime.path(),
        &["serve", package.to_str().unwrap()],
        Stdio::piped(),
    );
    let listening: Readiness = ready.wait(State::Listening, START).unwrap();
    assert_eq!(bound_port(listening.endpoints.join.as_ref()), join);

    // The daemon's log states when the leader awaits its other nodes; the reader goes on
    // draining the log to its end, so the daemon never blocks on it.
    let (lines, logged) = std::sync::mpsc::channel();
    let output = daemon.stdout.take().unwrap();
    std::thread::spawn(move || {
        for line in std::io::BufReader::new(output).lines() {
            let Ok(line) = line else { break };
            let _ = lines.send(line);
        }
    });
    let deadline = Instant::now() + START;
    loop {
        let line = logged
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .expect("the leader awaits its other nodes");
        if line.contains("awaiting the deployment's other nodes") {
            break;
        }
    }

    // SAFETY: a signal to this test's own child.
    let signalled = unsafe { libc::kill(daemon.id() as libc::pid_t, libc::SIGTERM) };
    assert_eq!(signalled, 0);
    let stopped = Instant::now();
    let status = loop {
        if let Some(status) = daemon.try_wait().unwrap() {
            break status;
        }
        if stopped.elapsed() > STOP {
            daemon.kill().unwrap();
            daemon.wait().unwrap();
            panic!("the leader did not end within {STOP:?} of SIGTERM");
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    assert!(status.success(), "{status:?}");
    let logged: Vec<String> = logged.try_iter().collect();
    assert!(
        !logged
            .iter()
            .any(|line| line.contains("the node serves the deployment")),
        "{logged:#?}"
    );
}
