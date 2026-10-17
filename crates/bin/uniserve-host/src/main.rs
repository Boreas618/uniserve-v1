//! The per-host launcher.
//!
//! Each host of a deployment runs one of these, started by the cluster; exactly
//! one host runs the head. The engine does not start processes on other hosts,
//! so a launcher connects out to the head, registers its host identity, receives
//! the launches of the ranks placed on its host, of every worker group the
//! deployment places there, spawns them, reports their exits by
//! worker, generation and rank, stops one group's ranks on instruction ahead
//! of that group's relaunch, and terminates them all when the head's
//! connection closes. For a group whose first rank it runs, it also holds the
//! bound socket that rank serves the group's collective store on, from the
//! head's reservation until the rank inherits it at spawn.
//!
//! Its ranks never outlive it. An orderly end of supervision terminates them,
//! and on Linux the kernel kills each rank when the launcher dies by any other
//! means. A rank is its child (`RankChild`), so each exit it reaps, reported
//! or terminated, first removes the CUDA driver memory a rank that did not
//! end in order left behind. Its log output is diagnostics only: a launcher
//! is often started over an SSH session whose pipe closes before the
//! deployment stops, and a failed log write must not end supervision.
//!
//! It does nothing else. The head derives every launch value once, so a
//! launcher's command line is the head's address and its own host identity.
//!
//! The connection carries the messages of `uniserve_ipc::launcher` as frames,
//! after the launcher's `Registration`; the head's side is the engine's
//! `LauncherRegistry`. The launcher accepts no inbound connection.

use std::collections::HashMap;
use std::net::{Ipv4Addr, TcpListener};
use std::os::fd::AsRawFd;
use std::process::Command;
use std::time::Duration;

use anyhow::Context;
use clap::Parser;
use uniserve_ipc::launch::RankChild;
use uniserve_ipc::launcher::{Instruction, RankExit, RankSpawn, Report, Reservation};
use uniserve_ipc::{Channel, IpcError, Registration, VERSION};

/// Bounds how long a rank's exit waits to be reported: exit reporting must
/// not depend on another instruction arriving.
const EXIT_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// How often a running rank's NCCL names are noted
/// (`RankChild::note_nccl_segments`). A rank reports that it is ready to the
/// head, not to its launcher, so the launcher notes them while the rank
/// lives; its communicators exist from its startup on.
const SEGMENT_NOTE_INTERVAL: Duration = Duration::from_secs(5);

/// Spawns and supervises the ranks an instance places on this host.
#[derive(Parser, Debug)]
#[command(
    name = "uniserve-host",
    about = "Launch and supervise one host's ranks"
)]
struct Args {
    /// Address of the head: the TCP address its log names for launchers.
    #[arg(long)]
    head: String,
    /// This host's identity, as the placement names it.
    #[arg(long = "host-identity")]
    host_identity: String,
}

/// A rank's identity on this host: its worker group and its rank within it.
type RankKey = (String, u32);

/// One rank this launcher owns.
struct Rank {
    child: RankChild,
    /// The group generation the rank's spawn named.
    generation: u64,
    /// When the rank's NCCL names were last noted.
    noted: std::time::Instant,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        // tracing-subscriber reports a failed write with `eprintln!`, which
        // panics, and so aborts this process, when stderr is the same closed
        // pipe; dropping the event keeps supervision independent of it.
        .log_internal_errors(false)
        .init();
    let args = Args::parse();

    let mut head = Channel::connect(&args.head)
        .with_context(|| format!("connecting to the head at {}", args.head))?;
    head.send(&Registration::Launcher {
        host: args.host_identity.clone(),
        version: VERSION.to_owned(),
    })
    .context("registering this host with the head")?;
    tracing::info!(host = %args.host_identity, head = %args.head, "presented this host");

    let mut ranks: HashMap<RankKey, Rank> = HashMap::new();
    let outcome = supervise(&mut head, &args, &mut ranks);

    // However supervision ends (the head closing the connection because the
    // instance stopped or the head was lost, a `Terminate`, or an error), this
    // host's ranks are terminated before the launcher exits.
    terminate(&mut ranks);
    outcome
}

/// Follows the head's instructions until it closes the connection or sends
/// `Terminate`, reporting rank exits as they happen.
///
/// Fails on an undecodable instruction, a failed reservation or rank start,
/// or a failed report to the head. The caller terminates the remaining ranks
/// in every case.
fn supervise(
    head: &mut Channel,
    args: &Args,
    ranks: &mut HashMap<RankKey, Rank>,
) -> anyhow::Result<()> {
    // Bound store sockets by worker group, each held until that group's first
    // rank is spawned here and inherits it.
    let mut rendezvous: HashMap<String, TcpListener> = HashMap::new();
    loop {
        report_exits(head, ranks)?;

        // A closed connection is the head's instruction to stop: the instance
        // stopped, the head was lost, or it refused this launcher's build.
        let instruction = match head.try_recv::<Instruction>() {
            Ok(Some(instruction)) => instruction,
            Ok(None) => {
                // A close observed by the wait is reported by the receive
                // that follows, which is where the launcher acts on it.
                match head.wait(EXIT_POLL_INTERVAL) {
                    Ok(()) | Err(IpcError::Transport(_)) => continue,
                    Err(error) => {
                        return Err(error).context("waiting for an instruction from the head");
                    }
                }
            }
            Err(IpcError::Transport(reason)) => {
                // The reason's cause, such as a reset connection, is its source.
                let reason = anyhow::Error::new(reason);
                tracing::info!(
                    reason = format!("{reason:#}"),
                    "head connection closed; terminating this host's ranks"
                );
                return Ok(());
            }
            Err(error) => return Err(error).context("decoding an instruction from the head"),
        };
        match instruction {
            Instruction::Reserve { worker_id } => {
                // This launcher holds the bound socket until it spawns the
                // group's first rank, which inherits it, so no other process
                // on this host, and no other group's reservation, can take the
                // port meanwhile.
                let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, 0))
                    .context("reserving a collective store address")?;
                let port = listener.local_addr()?.port();
                let reservation = Report::Reservation(Reservation {
                    worker_id: worker_id.clone(),
                    port,
                    // The first rank inherits the socket at this number.
                    listen_fd: listener.as_raw_fd(),
                });
                rendezvous.insert(worker_id, listener);
                head.send(&reservation)
                    .context("reporting a reservation to the head")?;
            }
            Instruction::Spawn(spawn) => {
                // The group's first rank serves its store on the reserved
                // socket; the others connect to it and bind nothing.
                let store_listener = if spawn.rank == 0 {
                    rendezvous.remove(&spawn.worker_id)
                } else {
                    None
                };
                let key = (spawn.worker_id.clone(), spawn.rank);
                let started = start_rank(args, *spawn, store_listener).with_context(|| {
                    format!("starting rank {} of worker {} on this host", key.1, key.0)
                })?;
                ranks.insert(key.clone(), started);
                tracing::info!(worker = %key.0, rank = key.1, "started a rank");
            }
            Instruction::Stop { worker_id } => {
                // The group's ranks leave supervision before they are killed,
                // so their exits are not reported. An unspawned reservation of
                // the group is closed.
                let stopped: Vec<RankKey> = ranks
                    .keys()
                    .filter(|(worker, _)| *worker == worker_id)
                    .cloned()
                    .collect();
                let mut group: HashMap<RankKey, Rank> = stopped
                    .into_iter()
                    .filter_map(|key| ranks.remove(&key).map(|rank| (key, rank)))
                    .collect();
                terminate(&mut group);
                rendezvous.remove(&worker_id);
                tracing::info!(worker = %worker_id, "head asked this host to stop a worker's ranks");
            }
            Instruction::Terminate => {
                tracing::info!("head asked this host to stop");
                return Ok(());
            }
        }
    }
}

/// Starts one rank, which connects to the head, registers as a member of
/// the launch `spawn.generation` names and receives its launch descriptor
/// there.
///
/// `store_listener` is the group's reserved collective store socket when this
/// rank serves the store. The rank inherits it at the descriptor number this
/// launcher reported with the reservation, which the head names in the rank's
/// launch descriptor. This process keeps no copy once the rank is started.
fn start_rank(
    args: &Args,
    spawn: RankSpawn,
    store_listener: Option<TcpListener>,
) -> anyhow::Result<Rank> {
    let mut command = uniserve_ipc::launch::rank_command(
        &spawn.python,
        &spawn.head,
        &spawn.worker_id,
        spawn.rank,
        spawn.generation,
    );
    for (name, value) in &spawn.environment {
        command.env(name, value);
    }
    die_with_launcher(&mut command);
    if let Some(listener) = store_listener {
        uniserve_ipc::launch::inherit_listener(&mut command, listener);
    }

    let child = RankChild::spawn(&mut command)
        .with_context(|| format!("spawning rank {} with {}", spawn.rank, spawn.python))?;
    tracing::debug!(rank = spawn.rank, host = %args.host_identity, "spawned");
    Ok(Rank {
        child,
        generation: spawn.generation,
        noted: std::time::Instant::now(),
    })
}

/// Makes the rank `command` starts die with this launcher.
///
/// The kernel sends the rank SIGKILL when the thread that spawned it exits,
/// which covers a launcher killed by a signal or aborted before `terminate`
/// runs. Every rank is spawned from the main thread, whose exit is the
/// process's. A rank whose launcher exited between fork and the request is
/// already orphaned, so it fails to start instead.
#[cfg(target_os = "linux")]
fn die_with_launcher(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    let launcher = std::process::id() as libc::pid_t;
    // SAFETY: the closure runs in the forked child between fork and exec and
    // calls only `prctl` and `getppid`, which are async-signal-safe, and
    // builds its errors without allocating.
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::getppid() != launcher {
                return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
            }
            Ok(())
        });
    }
}

/// Elsewhere only an orderly end of supervision terminates the ranks.
#[cfg(not(target_os = "linux"))]
fn die_with_launcher(_command: &mut Command) {}

/// Notes the NCCL names of each running rank every
/// [`SEGMENT_NOTE_INTERVAL`], and reports every rank that has exited since
/// the last report and stops supervising it. A rank whose exit status cannot
/// be read is reported as exited.
fn report_exits(head: &mut Channel, ranks: &mut HashMap<RankKey, Rank>) -> anyhow::Result<()> {
    let mut exited = Vec::new();
    for (key, owned) in ranks.iter_mut() {
        if owned.noted.elapsed() >= SEGMENT_NOTE_INTERVAL {
            owned.noted = std::time::Instant::now();
            let _ = owned.child.note_nccl_segments();
        }
        let status = match owned.child.try_wait() {
            Ok(Some(status)) => status.to_string(),
            Ok(None) => continue,
            Err(error) => format!("exit status unreadable: {error}"),
        };
        exited.push((key.clone(), owned.generation, status));
    }
    for ((worker_id, rank), generation, status) in exited {
        ranks.remove(&(worker_id.clone(), rank));
        let exit = Report::Exit(RankExit {
            worker_id: worker_id.clone(),
            generation,
            rank,
            status: status.clone(),
        });
        head.send(&exit)
            .context("reporting a rank exit to the head")?;
        tracing::warn!(worker = %worker_id, generation, rank, status = %status, "a rank exited");
    }
    Ok(())
}

/// Kills every rank in `ranks`, then reaps each one, leaving `ranks` empty.
fn terminate(ranks: &mut HashMap<RankKey, Rank>) {
    for ((worker_id, rank), owned) in ranks.iter_mut() {
        if let Err(error) = owned.child.kill() {
            tracing::warn!(worker = %worker_id, rank, %error, "a rank could not be stopped");
        }
    }
    for (_, mut owned) in ranks.drain() {
        let _ = owned.child.wait();
    }
}
