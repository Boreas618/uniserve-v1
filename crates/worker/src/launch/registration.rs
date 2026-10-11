//! The rank's registration with the head, and the head's liveness while the
//! rank starts.
//!
//! A rank connects to the head before it loads anything. Its first frame is
//! its [`Registration`]; the head answers with the rank's
//! [`LaunchDescriptor`]. The same connection then carries requests and
//! responses, and its close is the rank's signal to exit: the service
//! observes it while the rank serves, and [`ExitOnClose`] while it starts.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use uniserve_ipc::channel::Hangup;
use uniserve_ipc::{Channel, LaunchDescriptor, Registration, VERSION};

use super::Arguments;
use crate::{Error, Result};

/// Bounds a TCP connect to the head. It does not bound the wait for the
/// launch descriptor.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

/// Longest one wait for the launch descriptor blocks before it is repeated;
/// the wait as a whole has no deadline.
const DESCRIPTOR_WAIT: Duration = Duration::from_secs(60);

/// How often the startup watch re-checks whether it has been stopped.
const WATCH_INTERVAL: Duration = Duration::from_millis(200);

/// Connect to the head, register, and receive the launch descriptor.
///
/// The head answers once every rank of the launch has registered, which may
/// take as long as its slowest peer needs to start, so the wait for the
/// descriptor has no deadline of its own. It ends when the head closes the
/// connection: the head refuses the registration, gives up the launch, or is
/// gone.
///
/// The function uses nothing but its arguments, so any thread may call it.
/// The returned channel still has its initial capacity; the caller sets the
/// one the descriptor states.
pub fn register(arguments: &Arguments) -> Result<(Channel, LaunchDescriptor)> {
    let failed = |error| Error::Transport(format!("registering with {}: {error}", arguments.head));
    let mut channel = Channel::connect(&arguments.head, CONNECT_TIMEOUT).map_err(failed)?;
    channel
        .send(&Registration::Rank {
            worker_id: arguments.worker_id.clone(),
            rank: arguments.rank,
            generation: arguments.generation,
            version: VERSION.to_owned(),
        })
        .map_err(failed)?;

    loop {
        if let Some(descriptor) = channel.try_recv().map_err(failed)? {
            return Ok((channel, descriptor));
        }
        channel.wait(DESCRIPTOR_WAIT).map_err(failed)?;
    }
}

/// Ends the process when the head closes the connection, for as long as the
/// rank is starting.
///
/// A starting rank loads its model and may wait in a collective for its
/// peers, and calls its channel in neither, so it would not notice that the
/// head is gone. This watches the connection from its own thread meanwhile.
/// The rank cannot unwind from wherever it is blocked, so the exit is
/// immediate and runs no teardown, as a termination signal before serving
/// does. Dropping the watch stops it; the rank does so when it begins to
/// serve, from where its service observes the connection.
pub struct ExitOnClose {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl ExitOnClose {
    /// Start watching the connection `hangup` observes.
    pub fn watch(hangup: Hangup) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("uniserve-head-watch".into())
            .spawn(move || {
                while !stopped.load(Ordering::Acquire) {
                    // A failed poll is treated as a close: the connection can
                    // no longer be observed, so the rank cannot be left to
                    // run unwatched.
                    if hangup.wait(WATCH_INTERVAL).unwrap_or(true)
                        && !stopped.load(Ordering::Acquire)
                    {
                        tracing::error!(
                            target: "uniserve_worker.launch",
                            "the head closed the connection while the rank was starting"
                        );
                        // SAFETY: `_exit` ends the process without running
                        // any handler, which is safe from any thread.
                        unsafe { libc::_exit(1) };
                    }
                }
            })
            .map_err(|error| Error::Transport(format!("starting the head watch: {error}")))?;
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }
}

impl Drop for ExitOnClose {
    /// Stops the watch and waits for its thread, so no exit follows.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
