//! Hot restart of a leader daemon: a successor process, of the same build or
//! another, takes the deployment over while the daemon it replaces finishes
//! the requests it took, as Envoy's hot restart does.
//!
//! `POST /admin/restart` starts the successor (`uniserve serve --predecessor
//! FD`) with one end of a socket pair at that descriptor. They talk over it:
//!
//! 1. The daemon sends its options (`Prepare`). The successor loads the
//!    model's files and binds a request socket of its own, which it answers
//!    with (`Prepared`).
//! 2. The daemon stops its service thread between two rounds and sends what
//!    the thread served, its HTTP listener and the connection its engine's
//!    process lives under, with the arguments the engine serves with, the admin's log
//!    and the video jobs (`Takeover`), and commands the engine to attach to
//!    the successor's request socket (`EngineCommand::Attach`).
//! 3. The successor goes on with the service, adopts the engine once it
//!    attaches, and accepts on the HTTP listener (`Serving`).
//! 4. The daemon stops accepting and finishes the requests and video jobs
//!    it took, relaying each later change of a job to the successor
//!    (`Job`). The successor owns recovery and supplies engine connections
//!    for continuations (`Resume`, `Resumed`); further handoffs forward
//!    those requests to the current owner. Once its requests end, the daemon
//!    closes its engine connections without shutting the engine down.
//!
//! No connection closes: the listener, the processes' connections and the
//! claim cross as descriptors, and each request ends on the daemon that took
//! it. A connection that arrives meanwhile waits in the listener's queue
//! until one of the two accepts it. Until the successor serves, a failure
//! leaves the daemon serving as before, except after the service thread
//! stopped, which then fails the restart and the daemon with it.

use std::os::fd::{AsFd as _, AsRawFd as _, BorrowedFd, FromRawFd as _, OwnedFd};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use proto::{Connection, Handed};
use serde::{Deserialize, Serialize};
use storage::transport::Connection as Channel;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::engine::control::EngineControl;
use crate::http::admin::{Admin, HandedAdmin};
use crate::http::listener::{HandedListener, Listener};
use crate::leader::{Leader, stats_sink};
use crate::openai::video_jobs::{VideoJob, VideoJobs};
use crate::options::Stated;
use crate::supervision::{Daemon, HandedService};

pub(crate) mod requests;
pub(crate) mod streams;

/// The descriptor a successor finds the handoff socket at.
pub const PREDECESSOR_DESCRIPTOR: i32 = 3;

/// How long each side waits for the other's next step.
const STEP_TIMEOUT: Duration = Duration::from_secs(600);

/// How often a wait rechecks the channel.
const POLL: Duration = Duration::from_millis(20);

/// The messages of the handoff socket.
#[derive(Debug, Serialize, Deserialize)]
enum Message {
    /// Daemon to successor: the options it serves with.
    Prepare(Box<Prepare>),
    /// Successor to daemon: its request socket, which the engine attaches
    /// to.
    Prepared { requests: String },
    /// Daemon to successor: everything it serves, with the descriptors of
    /// [`Takeover`].
    Takeover(Box<Takeover>),
    /// Successor to daemon: the responses in flight it took over, which the
    /// predecessor answers: a response named here is the successor's from
    /// then on, and one the successor does not name resumes in the
    /// predecessor (design §7.5.1 step 4).
    Adopted {
        pid: u32,
        streams: Vec<streams::Taken>,
    },
    /// Successor to daemon: it serves; its process id.
    Serving { pid: u32 },
    /// A predecessor asks the admission owner for its next engine front.
    Resume {
        id: u64,
        replica: usize,
        requests: String,
        known_engine: Option<u64>,
        previous_epoch: Option<u64>,
    },
    Resumed {
        id: u64,
        attachment: Result<requests::Attachment, String>,
    },
    /// Daemon to successor: a video job's record changed, with its
    /// artifact's descriptor once it completed.
    Job {
        record: Box<VideoJob>,
        artifact: Option<u64>,
    },
    /// Either side: it gives up, and why.
    Failed { reason: String },
}

/// The options a daemon serves with.
#[derive(Debug, Serialize, Deserialize)]
struct Prepare {
    /// The configuration as TOML, as `GET /admin/config` prints
    /// it.
    config: String,
    stated: Vec<String>,
    /// The API key, which printing leaves out.
    api_key: Option<String>,
    /// The daemon's process socket, which the successor serves from the
    /// handoff on and its engines and workers connect to.
    processes: String,
    /// Whether the configuration listens on another address than the
    /// daemon's listener: the successor binds it before the handover.
    rebind: bool,
}

/// What a daemon hands its successor. The descriptors travel in this
/// order: the HTTP listener, the engine's control connection, the service's
/// descriptors, then each completed job's artifact.
#[derive(Debug, Serialize, Deserialize)]
struct Takeover {
    /// The responses in flight, which the successor takes over before it
    /// serves; their sockets follow the other descriptors, in this order.
    streams: Vec<streams::StreamHandoff>,
    listener: HandedListener,
    replicas: Vec<ReplicaHandoff>,
    adapters: std::collections::BTreeMap<String, crate::leader::Adapter>,
    service: HandedService,
    service_descriptors: usize,
    admin: HandedAdmin,
    jobs: Vec<(VideoJob, Option<u64>)>,
    then: Option<crate::updates::change::ApplyRequest>,
    /// Whether new requests waited when the predecessor handed over. The
    /// successor's gate starts as the predecessor's ended: a hot restart must
    /// not reopen admissions (§7.5.1), and the in-process hold alone would
    /// reopen them once the frontend serves.
    frontend_paused: bool,
    /// The programs the serving epoch runs. A configuration names a part of
    /// them and the store fills the rest, so the successor cannot derive them.
    programs: Option<HandedPrograms>,
}

/// The programs of the epoch a takeover continues.
#[derive(Debug, Serialize, Deserialize)]
struct HandedPrograms {
    kernels: Vec<(crate::epoch::KernelKey, program_server::program::Program)>,
    policy: program_server::program::Program,
}

/// One replica's control connection and executing epoch. Descriptors follow
/// replica order after the HTTP listener and before the service descriptors.
#[derive(Debug, Serialize, Deserialize)]
struct ReplicaHandoff {
    /// The connection the engine's process lives under. A process an epoch switch started has
    /// none: it lives under the node's supervision, which the service hands over.
    control: Option<(bool, Vec<u8>)>,
    engine: u64,
    next_engine: u64,
    /// The arguments the replica's engine process serves with.
    served: proto::config::EngineArgs,
    first_request: u64,
    config: String,
    paused: bool,
    asleep: bool,
    sleep_paused: bool,
    unavailable: Vec<crate::updates::epoch::Unavailable>,
}

/// What `POST /admin/restart` needs of the daemon it restarts.
pub(crate) struct Restarter {
    /// The daemon's service, which `serve` stops once the restart handed it
    /// over.
    pub(crate) daemon: Arc<std::sync::Mutex<Option<Daemon>>>,
    /// A duplicate of the HTTP listener, and what the successor needs of it.
    listener: std::sync::Mutex<Option<(HandedListener, OwnedFd)>>,
    stated: Stated,
    /// Cancelled once the successor serves: this daemon stops accepting.
    pub(crate) accepting: CancellationToken,
    /// Set once the deployment was handed over: this daemon detaches from
    /// the engine instead of shutting it down, and leaves the service and
    /// the storage to the successor.
    pub(crate) handed_over: Arc<AtomicBool>,
    /// Whether the restart in flight has concluded, and how: `None` while one
    /// runs, `Some(true)` once a successor adopted the deployment, `Some(false)`
    /// when it gave up.
    ///
    /// The process must not exit while a successor is still adopting: the
    /// handoff channel is a socket pair held by the restarting task, so exiting
    /// closes it under the successor, which then reports `the peer closed the
    /// connection` (run `bdf651928`).
    decided: tokio::sync::watch::Sender<Option<bool>>,
    /// Whether a restart is running now.
    ///
    /// `decided` starts at `None`, which is also its state before any restart
    /// was asked for: a waiter that read `None` alone would wait out the whole
    /// grace on every ordinary shutdown, so what says "wait" is this flag, set
    /// when a restart starts and cleared when it concludes.
    in_flight: Arc<AtomicBool>,
    /// The responses in flight, which a restart pauses at an engine event
    /// boundary and hands to the successor (design §7.5.1), attached once
    /// the HTTP server serves.
    handoff: std::sync::Mutex<Option<Arc<crate::http::handoff::Handoff>>>,
}

impl Restarter {
    /// The restarter of the daemon serving with `stated` on `listener`,
    /// whose service `daemon` holds.
    ///
    /// # Errors
    ///
    /// Fails when the listener cannot be duplicated.
    pub(crate) fn new(
        daemon: Arc<std::sync::Mutex<Option<Daemon>>>,
        listener: &mut Listener,
        stated: Stated,
        accepting: CancellationToken,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            daemon,
            listener: std::sync::Mutex::new(Some(
                listener
                    .hand_over()
                    .context("duplicating the HTTP listener")?,
            )),
            stated,
            accepting,
            handed_over: Arc::new(AtomicBool::new(false)),
            decided: tokio::sync::watch::channel(None).0,
            in_flight: Arc::new(AtomicBool::new(false)),
            handoff: std::sync::Mutex::new(None),
        })
    }

    /// The registry of the responses in flight, which the HTTP server
    /// registers every handoffable response with.
    pub(crate) fn attach_handoff(&self, handoff: Arc<crate::http::handoff::Handoff>) {
        let mut slot = crate::leader::locked(&self.handoff);
        *slot = Some(handoff);
    }

    /// The registry of the responses in flight, once the HTTP server serves.
    pub(crate) fn handoff(&self) -> Option<Arc<crate::http::handoff::Handoff>> {
        crate::leader::locked(&self.handoff).clone()
    }

    /// Restarts the daemon as `executable` (this daemon's own when `None`),
    /// as the module describes, up to the successor serving; returns its
    /// process id. Video job changes go on to the successor until this
    /// daemon exits.
    ///
    /// # Errors
    ///
    /// Fails when the successor does not start or gives up; before the
    /// service was handed over, this daemon serves on.
    pub(crate) async fn restart(
        &self,
        executable: Option<String>,
        config: Option<crate::options::ServerArgs>,
        then: Option<Box<crate::updates::change::ApplyRequest>>,
        leader: &Arc<Leader>,
        admin: &Arc<Admin>,
        jobs: &Arc<VideoJobs>,
    ) -> anyhow::Result<u32> {
        self.in_flight.store(true, Ordering::Release);
        let outcome = self
            .hand_over(executable, config, then, leader, admin, jobs)
            .await;
        // The restart is concluded however it ended: a daemon waiting to exit
        // stops waiting here, and a successor that adopted the deployment is the
        // only case in which this one may let go of it. The decision is stated
        // before the flag clears, so a waiter that sees the flag still set also
        // sees the decision.
        let _ = self.decided.send(Some(outcome.is_ok()));
        self.in_flight.store(false, Ordering::Release);
        outcome
    }

    /// The restarter's channel receiver: whoever must wait for the decision
    /// subscribes before the restart starts.
    pub(crate) fn decide(&self) -> tokio::sync::watch::Receiver<Option<bool>> {
        self.decided.subscribe()
    }

    /// Whether a restart runs now: a daemon ending its service waits for one
    /// that is in flight, and for nothing else.
    pub(crate) fn restarting(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.in_flight)
    }

    /// The restart itself, up to the successor serving.
    async fn hand_over(
        &self,
        executable: Option<String>,
        config: Option<crate::options::ServerArgs>,
        then: Option<Box<crate::updates::change::ApplyRequest>>,
        leader: &Arc<Leader>,
        admin: &Arc<Admin>,
        jobs: &Arc<VideoJobs>,
    ) -> anyhow::Result<u32> {
        let executable = match executable {
            Some(executable) => std::path::PathBuf::from(executable),
            None => std::env::current_exe().context("finding this daemon's executable")?,
        };
        // No epoch change, rollback or relaunch replaces the engine meanwhile.
        let _quiet = leader.quiesce().await;
        // The parked standbys are this daemon's processes (§2.5): the successor
        // numbers and starts its own once it has adopted the engines, so ending
        // them here keeps a second engine's memory off the device across the
        // handover instead of leaving the old daemon's copies parked behind it.
        leader.end_standbys().await;
        let handoff = leader.engine_handoff()?;
        let (ours, theirs) = UnixStream::pair().context("creating the handoff socket")?;
        // The persistent reply reader must yield to the continuations it
        // wakes, including when the handoff channel has no further traffic.
        ours.set_nonblocking(true)?;
        let successor = spawn(&executable, theirs)?;
        info!(pid = successor, executable = %executable.display(), "started the successor daemon");
        let mut channel = Channel::new(ours);

        // The successor serves with the daemon's options the change
        // states, over the configuration the engine serves.
        let options = config.unwrap_or_else(|| handoff[0].config.clone());
        // The listener binds the daemon's own address (`admin_config`): a
        // change of it is the rebind this successor makes before the
        // handover, and the previous daemon keeps serving if the bind fails.
        let address = |admin: &crate::options::AdminConfig| {
            (admin.host.clone(), admin.port, admin.uds.clone())
        };
        let rebind = address(&options.admin_config) != address(&handoff[0].config.admin_config);
        let config = crate::options::printed(&options)?;
        let processes = self
            .daemon
            .lock()
            .map_err(|_| anyhow::anyhow!("the daemon's lock is poisoned"))?
            .as_ref()
            .and_then(Daemon::address)
            .context("this daemon serves no process socket")?
            .to_owned();
        send(
            &mut channel,
            &Message::Prepare(Box::new(Prepare {
                config,
                stated: self.stated.paths(),
                api_key: options.server_config.api_key.clone(),
                processes,
                rebind,
            })),
            &[],
        )?;
        let requests = match recv(&mut channel).await? {
            (Message::Prepared { requests }, _) => requests,
            (Message::Failed { reason }, _) => anyhow::bail!("the successor gave up: {reason}"),
            (other, _) => anyhow::bail!("the successor answered {other:?}"),
        };

        // From here the service is the successor's.
        let (service, service_descriptors) = {
            let mut daemon = self
                .daemon
                .lock()
                .map_err(|_| anyhow::anyhow!("the daemon's lock is poisoned"))?;
            daemon
                .as_mut()
                .context("the daemon's service is not running")?
                .hand_over()?
        };
        let (listener, listener_descriptor) = self
            .listener
            .lock()
            .map_err(|_| anyhow::anyhow!("the listener's lock is poisoned"))?
            .take()
            .context("the HTTP listener was handed over already")?;
        let mut descriptors = vec![listener_descriptor];
        let mut replicas = Vec::new();
        for replica in &handoff {
            let control = replica.client.take_control().map(|control| {
                let control = control.hand_over();
                descriptors.push(control.descriptor);
                (control.unix, control.received)
            });
            replicas.push(ReplicaHandoff {
                control,
                engine: replica.engine,
                next_engine: replica.next_engine,
                served: replica.served.clone(),
                first_request: replica.client.next_request_id() + (1 << 32),
                config: crate::options::printed(&replica.config)?,
                paused: replica.paused,
                asleep: replica.asleep,
                sleep_paused: replica.sleep_paused,
                unavailable: replica.unavailable.clone(),
            });
        }
        // Stop every response in flight at its next engine event boundary, so
        // its state and its socket travel with the takeover (design §7.5.1).
        let mut paused = self.pause_streams().await;
        let mut stream_descriptors = std::mem::take(&mut paused.descriptors);
        let stream_records = std::mem::take(&mut paused.records);
        let writer = requests::writer(&channel)?;
        // Serialize the snapshot before any later job update uses the same
        // stream. Continuation requests share this writer after handoff.
        {
            let mut sending = crate::leader::locked(&writer);
            let jobs_handed = jobs.hand_over(job_relay(Arc::clone(&writer)));
            let service_count = service_descriptors.len();
            descriptors.extend(service_descriptors);
            let mut job_records = Vec::with_capacity(jobs_handed.len());
            for (record, media) in jobs_handed {
                let artifact = match media {
                    Some(media) => {
                        descriptors.push(artifact_descriptor(&media)?);
                        Some(media.len() as u64)
                    }
                    None => None,
                };
                job_records.push((record, artifact));
            }
            // The responses' sockets go last, so the successor reads them
            // after every other descriptor.
            descriptors.append(&mut stream_descriptors);
            tracing::info!(
                descriptors = descriptors.len(),
                service = service_count,
                engines = handoff.len(),
                "the daemon hands its service, its engines and the responses in flight over"
            );
            let borrowed: Vec<BorrowedFd<'_>> = descriptors.iter().map(|fd| fd.as_fd()).collect();
            send(
                &mut sending,
                &Message::Takeover(Box::new(Takeover {
                    streams: stream_records,
                    listener,
                    replicas,
                    adapters: crate::leader::locked(&leader.replicas.adapters).clone(),
                    service,
                    service_descriptors: service_count,
                    admin: admin.hand_over(),
                    jobs: job_records,
                    then: then.map(|request| *request),
                    frontend_paused: leader.status().paused,
                    programs: leader.epochs().serving().map(|(_, record)| HandedPrograms {
                        kernels: record.kernels.into_iter().collect(),
                        policy: record.policy,
                    }),
                })),
                &borrowed,
            )?;
        }
        // The predecessor's copies of what it handed over stay open until the successor
        // answers: closing them at the send was the earliest moment a peer could see its own
        // interface go away, and the successor had by then neither resumed it nor claimed the
        // deployment.
        self.handed_over.store(true, Ordering::Release);
        // From here the successor owns these engines: this daemon's watch sees their clients
        // go away with the handover and must not read that as a loss.
        for replica in leader.replicas() {
            replica
                .handed_over
                .store(true, std::sync::atomic::Ordering::Release);
        }
        for replica in handoff {
            replica.client.attach(&requests);
        }

        // The successor answers once: it either took the responses in flight
        // over, or serves without them, or gave up.
        let (pid, taken) = match recv(&mut channel).await? {
            (Message::Adopted { pid, streams }, _) => {
                info!(
                    count = streams.len(),
                    "the successor took the responses in flight over"
                );
                (pid, Some((pid, streams)))
            }
            (Message::Serving { pid }, _) => (pid, None),
            (Message::Failed { reason }, _) => {
                anyhow::bail!("the successor took the service and gave up: {reason}")
            }
            (other, _) => anyhow::bail!("the successor answered {other:?}"),
        };
        // The successor holds the service now, and this daemon is done with its copies.
        drop(descriptors);
        // The successor's own id either way: it is the process the `handoff ...
        // adopted` record names, and the one every response it did not take
        // stays paused behind here.
        match taken {
            Some((successor_pid, streams)) => paused.decide(&streams, successor_pid),
            // A successor that serves without taking any response over: every
            // response in flight stays here.
            None => paused.decide(&[], pid),
        }
        info!(
            pid,
            "the successor serves; this daemon finishes its requests"
        );
        let continuations = requests::Requests::new(writer);
        for (index, replica) in leader.replicas().iter().enumerate() {
            let client = replica
                .hold
                .client()
                .context("the handed-over replica has no client")?;
            let control = Arc::clone(
                replica
                    .control
                    .get()
                    .context("the handed-over replica has no request socket")?,
            );
            replica.hold.hand_over(Arc::new(requests::RemoteHold::new(
                Arc::clone(&continuations),
                index,
                control,
                client,
            )));
        }
        self.accepting.cancel();
        tokio::spawn(async move {
            loop {
                match receive(&mut channel, None).await {
                    Ok((Message::Resumed { id, attachment }, _)) => {
                        continuations.answer(id, attachment)
                    }
                    Ok((message, _)) => warn!("the successor daemon sent {message:?}"),
                    Err(error) => {
                        continuations.close(format!(
                            "the successor daemon closed its continuation relay: {error:#}"
                        ));
                        return;
                    }
                }
            }
        });
        Ok(pid)
    }
}

/// Starts `executable serve --predecessor` with `socket` at
/// [`PREDECESSOR_DESCRIPTOR`]; returns its process id. The successor is not
/// this daemon's to stop: it outlives it.
fn spawn(executable: &std::path::Path, socket: UnixStream) -> anyhow::Result<u32> {
    let descriptor = socket.as_raw_fd();
    let mut command = std::process::Command::new(executable);
    command.args([
        "serve",
        "--predecessor",
        &PREDECESSOR_DESCRIPTOR.to_string(),
    ]);
    // SAFETY: the closure runs in the forked child before exec and calls
    // only `dup2`, which is async-signal-safe; `socket` stays open in the
    // parent until the spawn returned.
    unsafe {
        command.pre_exec(move || {
            if libc::dup2(descriptor, PREDECESSOR_DESCRIPTOR) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let child = command.spawn().context("starting the successor daemon")?;
    drop(socket);
    Ok(child.id())
}

/// The relay of video job changes over the handoff socket `stream`: each
/// change is one `Job` message, with the artifact's descriptor once the job
/// completed. A change that cannot be sent is logged: the successor holds
/// the job's last record it received.
fn job_relay(writer: requests::Writer) -> crate::openai::video_jobs::Relay {
    Box::new(move |record, media| {
        let mut channel = crate::leader::locked(&writer);
        let artifact = media.map(|media| (media.len() as u64, artifact_descriptor(media)));
        let sent = match artifact {
            Some((length, Ok(descriptor))) => send(
                &mut channel,
                &Message::Job {
                    record: Box::new(record.clone()),
                    artifact: Some(length),
                },
                &[descriptor.as_fd()],
            ),
            Some((_, Err(error))) => Err(error),
            None => send(
                &mut channel,
                &Message::Job {
                    record: Box::new(record.clone()),
                    artifact: None,
                },
                &[],
            ),
        };
        if let Err(error) = sent {
            warn!(job = %record.id, "relaying a video job to the successor failed: {error:#}");
        }
    })
}

/// A descriptor of shared memory holding `media`'s bytes: the artifact's own
/// when it has one, else a new one they are copied into.
fn artifact_descriptor(
    media: &Arc<crate::openai::video_jobs::RetainedMedia>,
) -> anyhow::Result<OwnedFd> {
    if let Some(descriptor) = media.descriptor() {
        return Ok(descriptor.try_clone_to_owned()?);
    }
    let (object, mut file) = proto::SharedMemory::create(media.len())
        .context("creating shared memory for a video artifact")?;
    std::io::Write::write_all(&mut file, media)?;
    // The descriptor keeps the storage; the object's name is not needed.
    drop(object);
    Ok(OwnedFd::from(file))
}

/// Sends `message` with `descriptors` and waits until it left.
fn send(
    channel: &mut Channel,
    message: &Message,
    descriptors: &[BorrowedFd<'_>],
) -> anyhow::Result<()> {
    channel
        .send(message, descriptors)
        .context("writing to the handoff socket")?;
    while !channel.flush().context("writing to the handoff socket")? {
        std::thread::sleep(POLL);
    }
    Ok(())
}

/// The next message and its descriptors, waiting up to [`STEP_TIMEOUT`].
async fn recv(channel: &mut Channel) -> anyhow::Result<(Message, Vec<OwnedFd>)> {
    receive(channel, Some(STEP_TIMEOUT)).await
}

async fn receive(
    channel: &mut Channel,
    timeout: Option<Duration>,
) -> anyhow::Result<(Message, Vec<OwnedFd>)> {
    let deadline = timeout.map(|timeout| Instant::now() + timeout);
    loop {
        if let Some(frame) = channel
            .next_frame::<Message>()
            .context("reading the handoff socket")?
        {
            return Ok(frame);
        }
        anyhow::ensure!(
            !channel.is_closed(),
            "the other daemon closed the handoff socket"
        );
        anyhow::ensure!(
            deadline.is_none_or(|deadline| Instant::now() < deadline),
            "the other daemon did not answer in time"
        );
        channel.receive().context("reading the handoff socket")?;
        tokio::time::sleep(POLL).await;
    }
}

/// Serves as the successor of the daemon at the other end of the handoff
/// socket at `descriptor`, until `shutdown` is cancelled, as
/// [`crate::http::serve`] serves a deployment it started.
///
/// # Errors
///
/// Fails when the handoff fails or the deployment's serving fails.
pub async fn take_over(descriptor: i32, shutdown: CancellationToken) -> anyhow::Result<()> {
    // SAFETY: the daemon that started this process placed the handoff
    // socket at this descriptor, which nothing else in this process owns.
    let stream = UnixStream::from(unsafe { OwnedFd::from_raw_fd(descriptor) });
    stream.set_nonblocking(true)?;
    let mut channel = Channel::new(stream);
    let prepare = match recv(&mut channel).await? {
        (Message::Prepare(prepare), _) => prepare,
        (other, _) => anyhow::bail!("the daemon sent {other:?} first"),
    };
    match prepared_successor(&mut channel, *prepare, shutdown).await {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = send(
                &mut channel,
                &Message::Failed {
                    reason: format!("{error:#}"),
                },
                &[],
            );
            Err(error)
        }
    }
}

/// The successor's side once it has the options.
async fn prepared_successor(
    channel: &mut Channel,
    prepare: Prepare,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let (mut config, _) = crate::options::resolve_text(&prepare.config, &[])?;
    config.server_config.api_key = prepare.api_key;
    let stated = Stated::from_paths(prepare.stated);
    let prepared = Leader::prepare(&config, &stated).await?;
    // Another listening address is bound before the handover, so that a
    // refused bind leaves the previous daemon serving; the listener it hands
    // over then closes once the previous daemon stops accepting.
    let rebound = if prepare.rebind {
        let listener = Listener::bind(&config.admin_config)
            .await
            .context("binding the configuration's listening address")?;
        info!(address = %listener.local_addr()?, "listening on the configuration's address");
        Some(listener)
    } else {
        None
    };
    let control = EngineControl::new(
        &config.request_socket(),
        prepare.processes.clone(),
        config.host_identity.clone(),
        config.served_model_name().to_owned(),
        config.program_server.clone(),
    )?;
    send(
        channel,
        &Message::Prepared {
            requests: control.request_address().to_owned(),
        },
        &[],
    )?;

    let (takeover, descriptors) = match recv(channel).await? {
        (Message::Takeover(takeover), descriptors) => {
            tracing::info!(
                descriptors = descriptors.len(),
                service = takeover.service_descriptors,
                engines = takeover.replicas.len(),
                "the successor received the service, the engines and the responses in flight"
            );
            (takeover, descriptors)
        }
        (Message::Failed { reason }, _) => anyhow::bail!("the daemon gave up: {reason}"),
        (other, _) => anyhow::bail!("the daemon sent {other:?}"),
    };
    let Takeover {
        streams,
        listener,
        replicas,
        adapters,
        service,
        service_descriptors,
        admin,
        jobs,
        then,
        frontend_paused,
        programs,
    } = *takeover;
    let mut descriptors = descriptors.into_iter();
    let mut next = || {
        descriptors
            .next()
            .context("the takeover lacks a descriptor")
    };
    // Each step of the successor names itself in the reason it gives the predecessor: a
    // failure here is reported as `the successor took the service and gave up: <reason>`,
    // and a transport error alone says what happened but not where.
    let handed =
        Listener::resume(listener, next()?).context("resuming the handed-over HTTP listener")?;
    let listener = match rebound {
        Some(rebound) => {
            drop(handed);
            rebound
        }
        None => handed,
    };
    let replicas = replicas
        .into_iter()
        .map(|replica| {
            let connection = replica
                .control
                .as_ref()
                .map(|(unix, received)| {
                    Connection::resume(Handed {
                        descriptor: next()?,
                        unix: *unix,
                        received: received.clone(),
                    })
                    .context("resuming a replica's engine control connection")
                })
                .transpose()?;
            Ok((replica, connection))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let service_fds = (0..service_descriptors)
        .map(|_| next())
        .collect::<anyhow::Result<Vec<_>>>()?;
    let socket = prepare
        .processes
        .strip_prefix("unix:")
        .map(std::path::PathBuf::from);
    let mut daemon = Daemon::resume(
        config.host_identity.clone(),
        Some((prepare.processes.clone(), socket)),
        service,
        service_fds,
    )
    .context("resuming the node's daemon service")?;
    let epoch_records = Arc::new(crate::epoch::Epochs::new());
    // Every later update commands the same main processes through this daemon's
    // channel, as the first daemon's did after its start.
    if let Some(commands) = daemon.commands() {
        epoch_records.supervise(commands);
    }
    let (updates, awake) = daemon
        .updates(
            Arc::clone(&epoch_records),
            config.served_model_name().to_owned(),
        )
        .context("the leader's daemon serves no epoch layer")?;
    Daemon::advance_epochs(Arc::clone(&updates), awake)
        .context("starting the epoch layer's thread")?;
    let control = Arc::new(control);
    let adopting = Arc::clone(&control);
    let daemon_options = config.clone();
    let adopted = tokio::task::spawn_blocking(move || {
        replicas
            .into_iter()
            .enumerate()
            .map(|(index, (replica, connection))| {
                let (options, _) = crate::options::resolve_text(&replica.config, &[])?;
                let options = crate::options::with_daemon_options(&options, &daemon_options);
                let served = replica.served.clone();
                let observer = Arc::new(crate::observe::Observer::default());
                let sink = stats_sink(options.model().to_owned(), Arc::clone(&observer), index);
                let client = adopting.adopt(
                    replica.engine,
                    replica.next_engine,
                    served.epoch,
                    connection,
                    sink,
                )?;
                client.number_requests_from(replica.first_request);
                Ok((replica, options, served, observer, client))
            })
            .collect::<anyhow::Result<Vec<_>>>()
    })
    .await
    .context("the engines' adoption did not complete")??;
    let mut adopted = adopted.into_iter();
    let (handed, options, served, observer, client) =
        adopted.next().context("the takeover names no replicas")?;
    let served_epoch = served.epoch;
    let leader = Arc::new(
        Leader::finish(
            &config,
            prepared,
            Arc::clone(&control),
            client,
            served,
            observer,
            updates,
        )
        .context("finishing the deployment the takeover names")?,
    );
    // The successor adopts the epoch that serves (§2.5, §7.4, §7.5).
    {
        let (record, process) = adopted_epoch(served_epoch, handed.engine, &options, programs)?;
        epoch_records.record(served_epoch, record);
        let mut processes = epoch_records.processes();
        processes.push(process);
        epoch_records.set_processes(processes);
    }
    if frontend_paused {
        // The predecessor was gated when it handed over, so the successor's
        // gate starts closed: the in-process hold is re-asserted here (its
        // `adopt_pause` already carried the per-replica state, and this is the
        // one place that states the deployment's), and the frontend's gate,
        // once a frontend process holds one, is closed at the same point.
        leader.pause();
        info!("the successor continues gated: admissions stay closed across the handoff");
    }
    let first = Arc::clone(&leader.replicas()[0]);
    *crate::leader::locked(&first.config) = options;
    first.adopt_pause(handed.paused, handed.asleep, handed.sleep_paused);
    first.adopt_unavailable(handed.unavailable);
    // §2.5 continues across the handoff: the successor adopts the engines and
    // then keeps a standby of its own, or the deployment would serve without one
    // until the next restart. The epoch's programs serve already, so it prepares
    // from them as soon as it parks.
    first.prepare_standby();
    first.start_standby();
    for (index, (handed, options, served, observer, client)) in adopted.enumerate() {
        let replica = crate::leader::Replica::new(
            &options,
            Arc::new(client),
            observer,
            index + 1,
            Arc::downgrade(&leader.replicas),
            Arc::clone(&leader.replicas.adapters),
        );
        let _ = replica.control.set(Arc::clone(&control));
        *crate::leader::locked(&replica.served) = Some(served);
        replica.adopt_pause(handed.paused, handed.asleep, handed.sleep_paused);
        replica.adopt_unavailable(handed.unavailable);
        replica.monitor();
        replica.prepare_standby();
        replica.start_standby();
        leader
            .replicas
            .serving
            .send_modify(|replicas| replicas.push(replica));
    }

    *crate::leader::locked(&leader.replicas.adapters) = adapters;
    let retire = Arc::new(AtomicBool::new(false));
    let admin = Admin::resume(shutdown.clone(), Arc::clone(&retire), admin);
    admin.record(
        "deployment",
        format!("serving {} after a hot restart", leader.served_model_name()),
    );
    admin.watch(&leader);
    if let Some(request) = then {
        crate::http::admin::apply_after_restart(&admin, &leader, request);
    }
    let video_jobs = Arc::new(VideoJobs::new(
        config.video_job_config.clone(),
        shutdown.child_token(),
    ));
    for (record, artifact) in jobs {
        let media = match artifact {
            Some(length) => Some(proto::MappedMedia::map(next()?, length)?),
            None => None,
        };
        video_jobs.adopt(
            record,
            media.map(|media| Arc::new(media) as Arc<dyn proto::request::ArtifactMedia>),
        );
    }
    // The responses in flight go over once their engine streams do: a
    // response only moves when the engine serves it, and the predecessor
    // answers these before it stops writing anything.
    if !streams.is_empty() {
        let adopting = Arc::clone(&leader);
        let mut sockets = std::iter::from_fn(|| next().ok());
        let adopted = streams::adopt(&adopting, streams, &video_jobs, &mut || {
            sockets
                .next()
                .context("the takeover lacks a response's descriptor")
        });
        send(
            channel,
            &Message::Adopted {
                pid: std::process::id(),
                streams: adopted.streams,
            },
            &[],
        )?;
    }
    send(
        channel,
        &Message::Serving {
            pid: std::process::id(),
        },
        &[],
    )?;
    info!("serving the deployment the previous daemon handed over");

    // The predecessor owns its original HTTP streams; admission and engine
    // recovery belong here. The same channel carries job updates and asks
    // for the next front when one of those streams needs to continue.
    let relayed = Arc::clone(&video_jobs);
    let holds = leader
        .replicas()
        .iter()
        .map(|replica| Arc::clone(&replica.hold))
        .collect::<Vec<_>>();
    let writer = requests::writer(channel)?;
    let mut relay = std::mem::replace(channel, Channel::new(UnixStream::pair()?.0));
    let relay = tokio::spawn(async move {
        loop {
            match receive(&mut relay, None).await {
                Ok((Message::Job { record, artifact }, descriptors)) => {
                    let media = match (artifact, descriptors.into_iter().next()) {
                        (Some(length), Some(descriptor)) => {
                            proto::MappedMedia::map(descriptor, length)
                                .ok()
                                .map(|media| {
                                    Arc::new(media) as Arc<dyn proto::request::ArtifactMedia>
                                })
                        }
                        _ => None,
                    };
                    relayed.adopt(*record, media);
                }
                Ok((
                    Message::Resume {
                        id,
                        replica,
                        requests,
                        known_engine,
                        previous_epoch,
                    },
                    _,
                )) => {
                    let hold = holds.get(replica).cloned();
                    let writer = Arc::clone(&writer);
                    tokio::spawn(async move {
                        let attachment = match hold {
                            Some(hold) => hold
                                .attach(requests, known_engine, previous_epoch)
                                .await
                                .map_err(|error| error.to_string()),
                            None => Err(format!("the handoff has no replica {replica}")),
                        };
                        if let Err(error) = send(
                            &mut crate::leader::locked(&writer),
                            &Message::Resumed { id, attachment },
                            &[],
                        ) {
                            warn!("answering a predecessor continuation failed: {error:#}");
                        }
                    });
                }
                Ok((other, _)) => warn!("the replaced daemon sent {other:?}"),
                // The replaced daemon exited.
                Err(_) => return,
            }
        }
    });

    let daemon = Arc::new(std::sync::Mutex::new(Some(daemon)));
    let served = crate::http::serve_leader(
        &config,
        &stated,
        Arc::clone(&leader),
        admin,
        listener,
        video_jobs,
        Arc::clone(&daemon),
        &shutdown,
    )
    .await;
    let handed_over = served.as_ref().is_ok_and(|handed| *handed);
    if handed_over {
        // A second hot restart may finish this daemon's HTTP requests while
        // its predecessor still has streams. Keep forwarding their admission
        // to the latest owner until that predecessor drains and disconnects.
        let _ = relay.await;
    } else {
        relay.abort();
    }
    let served = served.map(drop);
    crate::http::finish(&config, &leader, daemon, retire, handed_over, served).await
}

/// The epoch record and the process row of an epoch a restarting daemon
/// adopts: the build config and the programs the adopted configuration
/// fills, serving, at the launch generation the child reported (which is the
/// identity it runs under, §7.4). The record's own generation stays `None`:
/// that field is the MODEL generation the adopted child numbers for its
/// programs, and a daemon that adopted it never saw that number — the launch
/// generation belongs to the process row alone.
///
/// Without the record the epoch layer says no epoch serves and refuses the
/// next switch; without the row the next launch would be numbered 1 again,
/// under a live child's generation (§2.5).
fn adopted_epoch(
    epoch: u64,
    engine: u64,
    options: &crate::options::ServerArgs,
    programs: Option<HandedPrograms>,
) -> anyhow::Result<(crate::epoch::EpochRecord, crate::epoch::ManagedProcess)> {
    let deployment = crate::epoch::Deployment::of(options.clone())?;
    let (kernels, policy) = match programs {
        Some(programs) => (programs.kernels.into_iter().collect(), programs.policy),
        None => (
            deployment.programs.kernels.clone(),
            deployment.programs.policy.clone(),
        ),
    };
    let record = crate::epoch::EpochRecord {
        config: deployment.engine.clone(),
        kernels,
        policy,
        previous_kernels: std::collections::BTreeMap::new(),
        previous_policy: None,
        status: crate::epoch::Status::Serving,
        generation: None,
    };
    let process = crate::epoch::ManagedProcess {
        role: crate::epoch::Role::MainProcess,
        group: crate::engine::control::ENGINE_GROUP.to_owned(),
        rank: 0,
        pid: None,
        epoch: Some(epoch),
        build: deployment.engine.executable.clone(),
        generation: engine,
        state: crate::epoch::ProcessState::Serving,
    };
    Ok((record, process))
}

/// How long a daemon that stopped serving waits for a restart in flight to
/// conclude before it exits anyway: the successor's own bounds are its connect
/// timeout and the prepare deadline, both shorter than this.
pub(crate) const RESTART_GRACE: std::time::Duration = std::time::Duration::from_secs(120);

#[cfg(test)]
mod tests {
    use super::*;

    /// The adopted epoch states the configuration it serves, and its process
    /// row the launch generation its child reported, so a later switch
    /// compares against it and numbers its children above it. The record's
    /// generation is the model generation, which the adopting daemon did not
    /// see and does not invent.
    #[test]
    fn an_adopted_epoch_records_the_build_it_serves_and_the_generation_it_runs_at() {
        let options = crate::options::ServerArgs::default();
        let (record, process) =
            adopted_epoch(7, 12, &options, None).expect("the configuration resolves");
        assert_eq!(record.status, crate::epoch::Status::Serving);
        assert_eq!(
            record.generation, None,
            "the model generation is not the launch generation the child reported"
        );
        assert_eq!(
            record.config,
            options.engine_config().expect("the configuration resolves")
        );
        assert_eq!(process.generation, 12);
        assert_eq!(process.epoch, Some(7));
        assert_eq!(process.state, crate::epoch::ProcessState::Serving);
        assert_eq!(process.role, crate::epoch::Role::MainProcess);
    }
}
