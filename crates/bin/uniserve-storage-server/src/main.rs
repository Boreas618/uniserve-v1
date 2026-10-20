//! `uniserve-storage-server`: the Storage Server of one host.
//!
//! `serve` holds this host's retained memory for the calling user until it
//! is stopped with SIGTERM or SIGINT. The head and each launcher start it
//! with `--daemon` when their host has none: the process then leaves its
//! starter's session and process tree, writes its log beside its socket and
//! signals its starter once it listens, so restarting a head or a launcher
//! never ends it.

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context;
use clap::{Parser, Subcommand};
use uniserve_storage::{Server, socket};

#[derive(Parser)]
#[command(
    name = "uniserve-storage-server",
    about = "The Storage Server of one host"
)]
struct Arguments {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Serve this host's retained memory until stopped.
    Serve {
        /// The socket to listen on; this user's default socket when unset.
        #[arg(long)]
        socket: Option<PathBuf>,
        /// Leave the starter's session, log beside the socket, and return to
        /// the starter once listening.
        #[arg(long)]
        daemon: bool,
    },
}

fn main() -> ExitCode {
    let Commands::Serve { socket, daemon } = Arguments::parse().command;
    let socket = socket.unwrap_or_else(socket::default_socket);
    // Forked before any thread exists.
    let ready = if daemon {
        match detach(&socket) {
            Ok(ready) => Some(ready),
            Err(error) => {
                eprintln!("uniserve-storage-server: error: {error:#}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("UNISERVE_STORAGE_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();
    match serve(&socket, ready) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!("storage server failed: {error:#}");
            ExitCode::FAILURE
        }
    }
}

/// Bind the socket and serve until stopped. `ready` is the starter's pipe,
/// written once the server listens or found another one listening.
fn serve(socket: &Path, ready: Option<OwnedFd>) -> anyhow::Result<()> {
    let signal = |ready: Option<OwnedFd>| {
        if let Some(ready) = ready {
            let _ = std::fs::File::from(ready).write_all(b"1");
        }
    };
    let Some((listener, _lock)) =
        socket::bind(socket).with_context(|| format!("binding {}", socket.display()))?
    else {
        tracing::info!(socket = %socket.display(), "another storage server serves this socket");
        signal(ready);
        return Ok(());
    };
    let (wake, stop) = stop_signals()?;
    let mut server = Server::new(listener)?;
    tracing::info!(socket = %socket.display(), "storage server listening");
    signal(ready);
    server.run(wake.as_fd(), &|| stop.load(Ordering::Acquire))?;
    tracing::info!("storage server stopped");
    Ok(())
}

/// A descriptor that becomes readable, and a flag that is set, when SIGTERM
/// or SIGINT arrives.
fn stop_signals() -> anyhow::Result<(OwnedFd, Arc<AtomicBool>)> {
    let mut ends = [0; 2];
    // SAFETY: pipe2 fills the two descriptors it creates.
    if unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_CLOEXEC | libc::O_NONBLOCK) } != 0 {
        return Err(std::io::Error::last_os_error()).context("creating the signal pipe");
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
        }
        .context("installing a signal handler")?;
    }
    Ok((reader, stop))
}

/// Leave the starter: fork, let the parent wait for the child's readiness
/// and exit, and make the child the leader of a session of its own with its
/// standard streams on `storage-server.log` beside the socket. Returns the
/// child's end of the readiness pipe.
fn detach(socket: &Path) -> anyhow::Result<OwnedFd> {
    let directory = socket.parent().context("the socket has no directory")?;
    std::fs::create_dir_all(directory)
        .with_context(|| format!("creating {}", directory.display()))?;
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(directory.join("storage-server.log"))
        .context("opening the storage server's log")?;

    let mut ends = [0; 2];
    // SAFETY: pipe2 fills the two descriptors it creates.
    if unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        return Err(std::io::Error::last_os_error()).context("creating the readiness pipe");
    }
    // SAFETY: pipe2 returned two new descriptors this function owns.
    let (reader, writer) =
        unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) };

    // SAFETY: the process has one thread, so the child may run anything.
    match unsafe { libc::fork() } {
        -1 => Err(std::io::Error::last_os_error()).context("forking the storage server"),
        0 => {
            drop(reader);
            // SAFETY: setsid and dup2 on descriptors this process owns.
            unsafe {
                libc::setsid();
                let null = libc::open(c"/dev/null".as_ptr(), libc::O_RDONLY);
                if null >= 0 {
                    libc::dup2(null, 0);
                    libc::close(null);
                }
                libc::dup2(log.as_raw_fd(), 1);
                libc::dup2(log.as_raw_fd(), 2);
            }
            Ok(writer)
        }
        _ => {
            drop(writer);
            // The child writes once it listens; a child that dies first
            // closes the pipe, which reads as end of file.
            let mut byte = [0_u8; 1];
            let started = std::fs::File::from(reader).read(&mut byte).unwrap_or(0) == 1;
            // SAFETY: `_exit` ends the starter without running handlers.
            unsafe { libc::_exit(if started { 0 } else { 1 }) }
        }
    }
}
