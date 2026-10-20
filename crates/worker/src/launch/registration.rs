//! The rank's registration with the head, and the head's liveness while the
//! rank starts.
//!
//! A rank connects to the head before it loads anything. Its first frame is
//! its [`Registration`], which names the rank, the launch of its group it was
//! started for and its build; the head answers with the rank's
//! [`LaunchDescriptor`]. The same connection then carries requests and
//! responses, and its close is the rank's signal to exit: the service
//! observes it while the rank serves, and [`HeadWatch`] while it starts.

use std::os::unix::net::UnixStream;
use std::thread::JoinHandle;
use std::time::Duration;

use uniserve_ipc::channel::Hangup;
use uniserve_ipc::{Channel, LaunchDescriptor, Registration, VERSION};

use super::Arguments;
use crate::{Error, Result};

/// Connect to the head, register, and receive the launch descriptor.
///
/// The head answers once every rank of the launch has registered, which may
/// take as long as its slowest peer needs to start, so the wait has no
/// deadline. It ends when the head closes the connection: the head refuses
/// the registration (a rank of a launch its group has since replaced, or of
/// another build), gives up the launch, or is gone.
///
/// The function uses nothing but its arguments, so any thread may call it.
pub fn register(arguments: &Arguments) -> Result<(Channel, LaunchDescriptor)> {
    let failed =
        |error| Error::transport_source(format!("registering with {}", arguments.head), error);
    let mut channel = Channel::connect(&arguments.head).map_err(failed)?;
    channel
        .send(&Registration::Worker {
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
        // Nothing but the descriptor or the close can end this wait, so its
        // length is immaterial.
        channel.wait(Duration::MAX).map_err(failed)?;
    }
}

/// Ends the process when the head closes the connection, for as long as the
/// rank is starting.
///
/// A starting rank loads its model and may wait in a collective for its
/// peers, and calls its channel in neither. Without this watch a rank whose
/// head died meanwhile would keep its devices until someone killed it. The
/// rank cannot unwind from wherever it is blocked, so the exit is immediate
/// and runs no teardown, as a termination signal before serving does.
/// Dropping the watch stops it; the rank does so when it begins to serve,
/// from where its service observes the connection.
pub(crate) struct HeadWatch {
    /// Closing this end cancels the watch thread's wait.
    cancel: Option<UnixStream>,
    thread: Option<JoinHandle<()>>,
}

impl HeadWatch {
    /// Start watching the connection `hangup` observes.
    pub(crate) fn start(hangup: Hangup) -> Result<Self> {
        let failed = |error| Error::transport_source("starting the head watch", error);
        let (cancel, cancelled) = UnixStream::pair().map_err(failed)?;
        let thread = std::thread::Builder::new()
            .name("uniserve-head-watch".into())
            .spawn(move || {
                // A failed wait is treated as a close: the connection can no
                // longer be observed, so the rank cannot be left unwatched.
                if hangup.wait(&cancelled).unwrap_or(true) {
                    tracing::error!("the head closed the connection while the rank was starting");
                    super::end(1);
                }
            })
            .map_err(failed)?;
        Ok(Self {
            cancel: Some(cancel),
            thread: Some(thread),
        })
    }
}

impl Drop for HeadWatch {
    /// Cancels the watch and waits for its thread, so no exit follows.
    fn drop(&mut self) {
        self.cancel.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
