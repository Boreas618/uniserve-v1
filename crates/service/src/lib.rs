//! What every UniServe process shares: its runtime files, the endpoints it binds and the
//! readiness record that reports them, a deployment's servers on a node, the one connection every
//! process interface runs on, logging and the stop signals.
//!
//! Runtime files live in the user's runtime directory ([`RuntimeDirectory`]); a server that is
//! not one of a deployment's servers owns an instance directory there ([`Instance`]). Every
//! server binds its listeners when it starts, port 0 allowed, and reports the [`Endpoint`]s it bound in its
//! [`Readiness`] record, written to the descriptor its caller passed ([`ReadyFd`]) and served by
//! its status route; callers read the record instead of choosing ports or deriving paths.
//!
//! The Storage Server and the Build Server are a deployment's servers on each of its nodes
//! ([`host`]): one of each per deployment and node, in the deployment's runtime directory
//! ([`RuntimeDirectory::deployment`]), owning its socket by a lock, publishing its readiness
//! record beside its socket, started detached from its starter when absent ([`host::start`]) and
//! stopped by its `stop` request or SIGTERM ([`host::stop`]). A server logs to standard error in the one convention of every process ([`logging`])
//! and stops on SIGTERM or SIGINT ([`stop_signals`]), the one stop mechanism every process waits
//! on.
//!
//! Every process interface runs on one [`Connection`] over a Unix or TCP
//! socket, framed by [`frame`], whose messages each protocol encodes
//! ([`Message`]). [`lock`] is the one way a thread takes a mutex, and [`wait`],
//! [`wait_while`] and [`wait_timeout`] the one way it waits on a condition variable.

mod connection;
mod directories;
mod endpoint;
pub mod frame;
pub mod host;
pub mod logging;
pub mod msgpack;
mod readiness;
mod runtime;

pub use connection::{
    Connection, Handed, Hangup, Listener, Liveness, Message, Socket, Wake, Waker,
    receive_buffer_bytes, wait_ready,
};
pub use directories::{cache_directory, data_directory, deployment};
pub use endpoint::{Endpoint, EndpointError};
pub use readiness::{Endpoints, Readiness, ReadyFd, ReadyReader, State};
pub use runtime::{Instance, RuntimeDirectory};

use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

/// What a thread that finds a lock poisoned states as it panics too.
const POISONED: &str = "a thread panicked while it held the lock";

/// Locks `mutex`.
///
/// Every build profile sets `panic = "abort"`, so a holder's panic ends the
/// process and no thread ever observes a poisoned lock. A test build unwinds
/// instead; a thread that finds the lock poisoned there panics as well, so
/// the state a panicking holder left is never used.
#[expect(
    clippy::expect_used,
    reason = "a poisoned lock is unreachable under `panic = \"abort\"`"
)]
pub fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().expect(POISONED)
}

/// Waits on `condvar`, releasing `guard` until it is notified, as [`lock`]
/// takes the lock.
#[expect(
    clippy::expect_used,
    reason = "a poisoned lock is unreachable under `panic = \"abort\"`"
)]
pub fn wait<'a, T>(condvar: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    condvar.wait(guard).expect(POISONED)
}

/// Waits on `condvar` while `condition` holds, as [`wait`] does.
#[expect(
    clippy::expect_used,
    reason = "a poisoned lock is unreachable under `panic = \"abort\"`"
)]
pub fn wait_while<'a, T>(
    condvar: &Condvar,
    guard: MutexGuard<'a, T>,
    condition: impl FnMut(&mut T) -> bool,
) -> MutexGuard<'a, T> {
    condvar.wait_while(guard, condition).expect(POISONED)
}

/// Waits on `condvar` for at most `timeout`, as [`wait`] does; whether the
/// wait timed out is the second value.
#[expect(
    clippy::expect_used,
    reason = "a poisoned lock is unreachable under `panic = \"abort\"`"
)]
pub fn wait_timeout<'a, T>(
    condvar: &Condvar,
    guard: MutexGuard<'a, T>,
    timeout: Duration,
) -> (MutexGuard<'a, T>, bool) {
    let (guard, waited) = condvar.wait_timeout(guard, timeout).expect(POISONED);
    (guard, waited.timed_out())
}

/// Raises the soft limit on open descriptors to the hard limit. Descriptors cross processes in
/// bulk ([`frame`]): a Storage Server holds one per retained storage of every worker on its
/// host, and a rank receives one per part of a KV region's segments it maps, thousands of
/// them for a large cache, which a soft limit of a thousand would refuse.
///
/// # Errors
///
/// What `getrlimit` or `setrlimit` fails with.
pub fn raise_descriptor_limit() -> io::Result<()> {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit fills the struct it is given.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } != 0 {
        return Err(io::Error::last_os_error());
    }

    limit.rlim_cur = limit.rlim_max;
    // SAFETY: setrlimit reads the struct it is given.
    if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// A descriptor that becomes readable, and a flag that is set, when SIGTERM
/// or SIGINT arrives: a server that polls waits on the descriptor, and one
/// that loops reads the flag.
///
/// # Errors
///
/// Fails when the pipe or a handler cannot be made.
pub fn stop_signals() -> io::Result<(OwnedFd, Arc<AtomicBool>)> {
    let mut ends = [0; 2];
    // SAFETY: pipe2 fills the two descriptors it creates.
    if unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_CLOEXEC | libc::O_NONBLOCK) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: pipe2 returned two new descriptors this function owns.
    let (reader, writer) =
        unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) };

    // The write end lives for the process: a handler may run until exit.
    let writer = Box::leak(Box::new(writer)).as_raw_fd();
    let stop = Arc::new(AtomicBool::new(false));
    for signal in [libc::SIGTERM, libc::SIGINT] {
        let flag = Arc::clone(&stop);
        // SAFETY: the action stores to an atomic and writes to a descriptor
        // that is never closed, both async-signal-safe.
        unsafe {
            signal_hook_registry::register(signal, move || {
                flag.store(true, Ordering::Release);
                let _ = libc::write(writer, b"x".as_ptr().cast(), 1);
            })
        }?;
    }
    Ok((reader, stop))
}
