//! The per-host launcher.
//!
//! Each host of a deployment runs one of these, started by the cluster; exactly
//! one host runs the head. The engine does not start processes on other hosts,
//! so a launcher connects out to the head, registers its host identity, receives
//! the launches of the workers placed on its host, of every worker group the
//! deployment places there, spawns them, reports their exits by
//! worker group, generation and rank, stops one group's workers on instruction ahead
//! of that group's relaunch, and terminates them all when the head's
//! connection closes. For a group whose first worker it runs, it also holds the
//! bound socket that worker serves the group's collective store on, from the
//! head's reservation until the worker inherits it at spawn.
//!
//! Its workers never outlive it. An orderly end of supervision terminates them,
//! and on Linux the kernel kills each worker when the launcher dies by any other
//! means. A worker is its child (`WorkerChild`), so each exit it reaps, reported
//! or terminated, first removes the CUDA driver memory a worker that did not
//! end in order left behind. Its log output is diagnostics only: a launcher
//! is often started over an SSH session whose pipe closes before the
//! deployment stops, and a failed log write must not end supervision.
//!
//! On the head's instruction it also claims the deployment at its host's
//! Storage Server, starting the server when the host has none, and holds
//! that claim for the head while their connection lasts.
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
use uniserve_ipc::launch::WorkerChild;
use uniserve_ipc::launcher::{
    Instruction, Report, Reservation, StorageClaim, WorkerExit, WorkerSpawn,
};
use uniserve_ipc::{Channel, IpcError, Registration, VERSION};

/// Bounds how long a worker's exit waits to be reported: exit reporting must
/// not depend on another instruction arriving.
const EXIT_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Longest a Storage Server this launcher starts takes to listen.
const STORAGE_START_TIMEOUT: Duration = Duration::from_secs(30);

/// How often a running worker's NCCL names are noted
/// (`WorkerChild::note_nccl_segments`). A worker reports that it is ready to the
/// head, not to its launcher, so the launcher notes them while the worker
/// lives; its communicators exist from its startup on.
const SEGMENT_NOTE_INTERVAL: Duration = Duration::from_secs(5);

/// Spawns and supervises the workers a head places on this host.
#[derive(Parser, Debug)]
#[command(
    name = "uniserve-launcher",
    about = "Launch and supervise one host's workers"
)]
struct Args {
    /// Address of the head: the TCP address its log names for launchers.
    #[arg(long)]
    head: String,
    /// This host's identity, as the placement names it.
    #[arg(long = "host-identity")]
    host_identity: String,
}

/// A worker's identity on this host: its worker group and its rank within it.
type WorkerKey = (String, u32);

/// One worker this launcher owns.
struct Worker {
    child: WorkerChild,
    /// The group generation the worker's spawn named.
    generation: u64,
    /// When the worker's NCCL names were last noted.
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

    let mut workers: HashMap<WorkerKey, Worker> = HashMap::new();
    let outcome = supervise(&mut head, &args, &mut workers);

    // However supervision ends (the head closing the connection because the
    // head stopped or was lost, a `Terminate`, or an error), this
    // host's workers are terminated before the launcher exits.
    terminate(&mut workers);
    outcome
}

/// Follows the head's instructions until it closes the connection or sends
/// `Terminate`, reporting worker exits as they happen.
///
/// Fails on an undecodable instruction, a failed reservation or worker
/// start, or a failed report to the head. The caller terminates the
/// remaining workers in every case.
fn supervise(
    head: &mut Channel,
    args: &Args,
    workers: &mut HashMap<WorkerKey, Worker>,
) -> anyhow::Result<()> {
    // Bound store sockets by worker group, each held until that group's first
    // worker is spawned here and inherits it.
    let mut rendezvous: HashMap<String, TcpListener> = HashMap::new();
    // The connection that holds the deployment's claim at this host's
    // Storage Server for the head; the claim ends with it.
    let mut storage: Option<uniserve_storage::Client> = None;
    loop {
        report_exits(head, workers)?;

        // A closed connection is the head's instruction to stop: the head
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
                    "head connection closed; terminating this host's workers"
                );
                return Ok(());
            }
            Err(error) => return Err(error).context("decoding an instruction from the head"),
        };
        match instruction {
            Instruction::Reserve { worker_id } => {
                // This launcher holds the bound socket until it spawns the
                // group's first worker, which inherits it, so no other process
                // on this host, and no other group's reservation, can take the
                // port meanwhile.
                let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, 0))
                    .context("reserving a collective store address")?;
                let port = listener.local_addr()?.port();
                let reservation = Report::Reservation(Reservation {
                    port,
                    // The first worker inherits the socket at this number.
                    listen_fd: listener.as_raw_fd(),
                });
                rendezvous.insert(worker_id, listener);
                head.send(&reservation)
                    .context("reporting a reservation to the head")?;
            }
            Instruction::Spawn(spawn) => {
                // The group's first worker serves its store on the reserved
                // socket; the others connect to it and bind nothing.
                let store_listener = if spawn.rank == 0 {
                    rendezvous.remove(&spawn.worker_id)
                } else {
                    None
                };
                let key = (spawn.worker_id.clone(), spawn.rank);
                let started = start_worker(args, *spawn, store_listener).with_context(|| {
                    format!(
                        "starting rank {} of worker group {} on this host",
                        key.1, key.0
                    )
                })?;
                workers.insert(key.clone(), started);
                tracing::info!(worker = %key.0, rank = key.1, "started a worker");
            }
            Instruction::Stop { worker_id } => {
                // The group's workers leave supervision before they are killed,
                // so their exits are not reported. An unspawned reservation of
                // the group is closed.
                let stopped: Vec<WorkerKey> = workers
                    .keys()
                    .filter(|(worker, _)| *worker == worker_id)
                    .cloned()
                    .collect();
                let mut group: HashMap<WorkerKey, Worker> = stopped
                    .into_iter()
                    .filter_map(|key| workers.remove(&key).map(|worker| (key, worker)))
                    .collect();
                terminate(&mut group);
                rendezvous.remove(&worker_id);
                tracing::info!(worker = %worker_id, "head asked this host to stop a worker group's workers");
            }
            Instruction::Claim(claim) => {
                let deployment = claim.deployment.clone();
                let claimed =
                    claim_storage(claim, &mut storage).map_err(|error| format!("{error:#}"));
                if let Err(message) = &claimed {
                    tracing::warn!(deployment, message, "claiming this host's storage failed");
                } else {
                    tracing::info!(deployment, "claimed this host's storage for the head");
                }
                head.send(&Report::Claim(claimed))
                    .context("reporting a storage claim to the head")?;
            }
            Instruction::Terminate => {
                tracing::info!("head asked this host to stop");
                return Ok(());
            }
        }
    }
}

/// Claims the deployment at this host's Storage Server, starting the server
/// when the host has none, and keeps the claiming connection in `storage`.
fn claim_storage(
    claim: StorageClaim,
    storage: &mut Option<uniserve_storage::Client>,
) -> anyhow::Result<uniserve_ipc::StorageLaunch> {
    let socket = uniserve_storage::socket::default_socket();
    uniserve_storage::socket::ensure(
        std::path::Path::new(&claim.server),
        &socket,
        STORAGE_START_TIMEOUT,
    )
    .context("starting this host's storage server")?;
    let mut client = uniserve_storage::Client::connect(
        &socket,
        uniserve_storage::Role::Head {
            deployment: claim.deployment.clone(),
        },
    )?;
    let token = client.claim(
        claim
            .scopes
            .into_iter()
            .map(|(group, rank)| uniserve_storage::Scope {
                deployment: claim.deployment.clone(),
                group,
                rank,
            })
            .collect(),
    )?;
    *storage = Some(client);
    Ok(uniserve_ipc::StorageLaunch {
        socket: socket.to_string_lossy().into_owned(),
        deployment: claim.deployment,
        claim: token.0,
    })
}

/// Starts one worker, which connects to the head, registers as a member of
/// the launch `spawn.generation` names and receives its launch descriptor
/// there.
///
/// `store_listener` is the group's reserved collective store socket when this
/// worker serves the store. The worker inherits it at the descriptor number
/// this launcher reported with the reservation, which the head names in the
/// worker's launch descriptor. This process keeps no copy once the worker is
/// started.
fn start_worker(
    args: &Args,
    spawn: WorkerSpawn,
    store_listener: Option<TcpListener>,
) -> anyhow::Result<Worker> {
    let mut command = uniserve_ipc::launch::worker_command(
        &spawn.executable,
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

    let child = WorkerChild::spawn(&mut command)
        .with_context(|| format!("spawning rank {} with {}", spawn.rank, spawn.executable))?;
    tracing::debug!(rank = spawn.rank, host = %args.host_identity, "spawned");
    Ok(Worker {
        child,
        generation: spawn.generation,
        noted: std::time::Instant::now(),
    })
}

/// Makes the worker `command` starts die with this launcher.
///
/// The kernel sends the worker SIGKILL when the thread that spawned it exits,
/// which covers a launcher killed by a signal or aborted before `terminate`
/// runs. Every worker is spawned from the main thread, whose exit is the
/// process's. A worker whose launcher exited between fork and the request is
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

/// Elsewhere only an orderly end of supervision terminates the workers.
#[cfg(not(target_os = "linux"))]
fn die_with_launcher(_command: &mut Command) {}

/// Notes the NCCL names of each running worker every
/// [`SEGMENT_NOTE_INTERVAL`], and reports every worker that has exited since
/// the last report and stops supervising it. A worker whose exit status
/// cannot be read is reported as exited.
fn report_exits(
    head: &mut Channel,
    workers: &mut HashMap<WorkerKey, Worker>,
) -> anyhow::Result<()> {
    let mut exited = Vec::new();
    for (key, owned) in workers.iter_mut() {
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
        workers.remove(&(worker_id.clone(), rank));
        let exit = Report::Exit(WorkerExit {
            worker_id: worker_id.clone(),
            generation,
            rank,
            status: status.clone(),
        });
        head.send(&exit)
            .context("reporting a worker exit to the head")?;
        tracing::warn!(worker = %worker_id, generation, rank, status = %status, "a worker exited");
    }
    Ok(())
}

/// Kills every worker in `workers`, then reaps each one, leaving `workers`
/// empty.
fn terminate(workers: &mut HashMap<WorkerKey, Worker>) {
    for ((worker_id, rank), owned) in workers.iter_mut() {
        if let Err(error) = owned.child.kill() {
            tracing::warn!(worker = %worker_id, rank, %error, "a worker could not be stopped");
        }
    }
    for (_, mut owned) in workers.drain() {
        let _ = owned.child.wait();
    }
}
