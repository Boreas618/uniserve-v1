//! Servers bind their listeners when they start, accept port 0, and report what they bound in
//! their readiness record on the descriptor their caller passes (`--ready-fd`); a host service
//! answers `status` and `stop` over its socket.

use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
use std::os::unix::process::CommandExt as _;
use std::path::Path;
use std::process::{Child, Command, Output};
use std::time::Duration;

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

/// `uniserve ARGUMENTS --ready-fd 3`, the readiness pipe's write end at descriptor 3, under the
/// runtime directory `runtime` and with no device visible; returns the child and the pipe's read
/// end.
fn spawn_ready(runtime: &Path, arguments: &[&str]) -> (Child, ReadyReader) {
    let (reader, writer) = pipe();
    let write_end = writer.as_raw_fd();
    let mut command = Command::new(env!("CARGO_BIN_EXE_uniserve"));
    command
        .args(arguments)
        .args(["--ready-fd", &READY_FD.to_string()])
        .env("XDG_RUNTIME_DIR", runtime)
        .env("CUDA_VISIBLE_DEVICES", "");
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

/// A daemon asked for port 0 binds its client listener and its admin sockets, in an instance
/// directory of its own inside the runtime directory, before it serves, and states them in its
/// readiness record: first listening, then serving. Its admin socket answers once it serves, and
/// its instance directory goes with it.
#[test]
fn a_daemon_reports_the_listeners_it_bound_on_port_0() {
    let runtime = tempfile::tempdir().unwrap();
    let (mut daemon, mut ready) = spawn_ready(runtime.path(), &["serve", "--port", "0"]);

    let listening: Readiness = ready.wait(State::Listening, START).unwrap();
    let instance = listening.instance.clone().expect("an instance directory");
    assert_eq!(
        instance.parent(),
        Some(runtime.path().join("uniserve").as_path())
    );
    assert_ne!(bound_port(listening.endpoints.client.as_ref()), 0);
    let admin = listening.endpoints.admin.clone().expect("an admin socket");
    assert_eq!(admin, Endpoint::unix(instance.join("admin.sock")));

    let serving: Readiness = ready.wait(State::Serving, START).unwrap();
    assert_eq!(serving.endpoints, listening.endpoints);
    let mut stream = std::os::unix::net::UnixStream::connect(admin.path().unwrap()).unwrap();
    std::io::Write::write_all(
        &mut stream,
        b"GET /admin/node HTTP/1.1\r\nHost: admin\r\nConnection: close\r\n\r\n",
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

/// A daemon run without a runtime directory has nowhere to put its sockets and is refused.
#[test]
fn a_daemon_without_a_runtime_directory_is_refused() {
    let refused = Command::new(env!("CARGO_BIN_EXE_uniserve"))
        .args(["serve", "--port", "0"])
        .env_remove("XDG_RUNTIME_DIR")
        .output()
        .unwrap();
    assert!(!refused.status.success());
    let message =
        String::from_utf8_lossy(&refused.stderr) + String::from_utf8_lossy(&refused.stdout);
    assert!(message.contains("XDG_RUNTIME_DIR"), "{message}");
}
