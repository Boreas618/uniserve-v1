//! Interrupt and termination disposition of a rank process.
//!
//! Before a rank serves, SIGINT ends it with status 130 and SIGTERM with the
//! default signal death, so an operator or a failed launch can stop a
//! starting rank. Once serving, both are ignored with a log line: a
//! terminal's Ctrl-C or a service manager's stop reaches every process of the
//! job, and only the head may close a rank, collectively with its group.
//!
//! The handler itself only writes the signal number to a pipe (an
//! async-signal-safe operation); a dedicated thread applies the policy.
//! SIGQUIT and SIGUSR1 stay with Python's faulthandler.

use std::io::Read;
use std::os::fd::{AsRawFd, OwnedFd};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{Error, Result};

/// The installed policy. Dropping it leaves the handlers installed: a rank
/// keeps its disposition until it exits.
pub(crate) struct Signals {
    serving: Arc<AtomicBool>,
}

impl Signals {
    /// Install the startup disposition for SIGINT and SIGTERM.
    pub(crate) fn install() -> Result<Self> {
        let (reader, writer) = rustix::pipe::pipe_with(rustix::pipe::PipeFlags::CLOEXEC)
            .map_err(|error| Error::transport_source("signal pipe", error))?;
        // The write end lives for the process: handlers may run until exit.
        let writer: &'static OwnedFd = Box::leak(Box::new(writer));
        let fd = writer.as_raw_fd();
        for signal in [libc::SIGINT, libc::SIGTERM] {
            let byte = signal as u8;
            // SAFETY: the action only calls write(2) on a descriptor that is
            // never closed, which is async-signal-safe.
            unsafe {
                signal_hook_registry::register(signal, move || {
                    let _ = libc::write(fd, (&raw const byte).cast(), 1);
                })
            }
            .map_err(|error| Error::transport_source("installing a signal handler", error))?;
        }

        let serving = Arc::new(AtomicBool::new(false));
        let policy = Arc::clone(&serving);
        std::thread::Builder::new()
            .name("worker-signals".into())
            .spawn(move || dispatch(std::fs::File::from(reader), &policy))
            .map_err(|error| Error::transport_source("signal thread", error))?;
        Ok(Self { serving })
    }

    /// Switch to the serving disposition: signals are logged and ignored.
    pub(crate) fn serving(&self) {
        self.serving.store(true, Ordering::Release);
    }
}

fn dispatch(mut signals: std::fs::File, serving: &AtomicBool) {
    let mut byte = [0u8; 1];
    while signals.read_exact(&mut byte).is_ok() {
        let signal = i32::from(byte[0]);
        let name = if signal == libc::SIGINT {
            "SIGINT"
        } else {
            "SIGTERM"
        };
        if serving.load(Ordering::Acquire) {
            tracing::info!("ignoring {name}; the head shuts this rank down with its worker group");
            continue;
        }
        if signal == libc::SIGINT {
            tracing::warn!("worker interrupted; shutting down");
            // Ending at once runs no destructor, which could wait on device
            // work or peers a failed startup abandoned.
            super::end(130);
        }
        // Restore the default disposition and deliver the signal again, so
        // the daemon observes a signal death.
        // SAFETY: plain libc calls with valid arguments.
        unsafe {
            libc::signal(libc::SIGTERM, libc::SIG_DFL);
            libc::kill(libc::getpid(), libc::SIGTERM);
        }
    }
}
