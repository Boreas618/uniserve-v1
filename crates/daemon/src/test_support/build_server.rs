//! A deployment's Build Server on a node as a daemon finds it serving there: a server another
//! daemon of the deployment started, whose process and settings its readiness record states.
//!
//! The stand-in owns the deployment's socket as a Build Server does (`service::host::bind`) and
//! publishes a record of the settings it was given. The process the record names is a child of
//! the test that runs until the stand-in drops: it is what a stop of the server signals
//! (`service::host::stop`), so a daemon that replaces the server ends that process and leaves a
//! record that no longer names a running one. The stand-in answers each `prepare` with the
//! package revisions the request fixes, as a server that holds every module answers, and closes
//! the connection of every other request unanswered, so resolving a module there fails.

use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use build_server::protocol::wire;
use service::Connection;

/// A Build Server of one deployment serving on a node's runtime directory (the module
/// documentation).
pub(crate) struct BuildServerStandIn {
    socket: PathBuf,
    settings: build_server::command::Settings,
    /// The server's process, which its record names.
    process: std::process::Child,
    /// Set when the stand-in drops: the thread that owns the socket ends at its next connection.
    ending: Arc<AtomicBool>,
    serving: Option<std::thread::JoinHandle<()>>,
}

impl BuildServerStandIn {
    /// Serves as the Build Server of `deployment` in the runtime directory `runtime` with
    /// `settings`.
    pub(crate) fn serve(
        runtime: &service::RuntimeDirectory,
        deployment: &str,
        settings: build_server::command::Settings,
    ) -> Self {
        let socket = build_server::socket(runtime, deployment).unwrap();
        let process = std::process::Command::new("sleep")
            .arg("3600")
            .spawn()
            .expect("starting the stand-in's process");
        let (listener, starting) = service::host::bind::<build_server::command::Settings>(&socket)
            .unwrap()
            .owned(&socket)
            .expect("no server owns the stand-in's socket");
        let mut record = build_server::readiness(&socket, &settings);
        record.pid = process.id();
        let lock = starting.finish(&record).unwrap();
        let ending = Arc::new(AtomicBool::new(false));
        let ended = Arc::clone(&ending);
        let serving = std::thread::spawn(move || {
            // The socket is the stand-in's until this thread ends and drops the lock, which
            // removes the socket's file.
            let _lock = lock;
            for stream in listener.incoming() {
                if ended.load(Ordering::Acquire) {
                    return;
                }
                if let Ok(stream) = stream {
                    answer(Connection::new(stream).unwrap());
                }
            }
        });
        Self {
            socket,
            settings,
            process,
            ending,
            serving: Some(serving),
        }
    }

    /// Whether the deployment's Build Server is still this stand-in as it started: its process
    /// runs, and the server that owns the socket publishes the record of that process with the
    /// settings the stand-in serves with.
    pub(crate) fn serves_as_started(&mut self) -> bool {
        let running = self.process.try_wait().unwrap().is_none();
        let record =
            service::host::running::<build_server::command::Settings>(&self.socket).unwrap();
        running
            && record.is_some_and(|record| {
                record.pid == self.process.id() && record.settings == self.settings
            })
    }
}

impl Drop for BuildServerStandIn {
    fn drop(&mut self) {
        self.ending.store(true, Ordering::Release);
        // A connection wakes the thread's accept, which then ends.
        let _ = UnixStream::connect(&self.socket);
        if let Some(serving) = self.serving.take() {
            let _ = serving.join();
        }
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

/// Answers the one request a client sends on `connection`: a `prepare` with the revisions it
/// fixes, anything else with the connection's end.
fn answer(mut connection: Connection) {
    let Ok(request) = connection.recv_next::<wire::Request>() else {
        return;
    };
    if let Some(wire::request::Request::Prepare(prepare)) = request.request {
        let packages = prepare
            .packages
            .into_iter()
            .map(|requested| wire::SlotPackage {
                op: requested.op,
                key: requested.key,
                part: requested.part,
                package: match requested.selector {
                    Some(wire::package_request::Selector::Package(revision)) => Some(revision),
                    _ => None,
                },
            })
            .collect();
        let _ = connection.deliver(&wire::ServerMessage {
            message: Some(wire::server_message::Message::Prepared(
                wire::PrepareResponse { packages },
            )),
        });
    }
}
