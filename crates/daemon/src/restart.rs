//! Hot restart of a leader daemon: a successor process, of the same build or
//! another, takes the deployment over while the daemon it replaces finishes
//! the requests it took, as Envoy's hot restart does.
//!
//! `POST /admin/restart` starts the successor (`uniserve serve --takeover
//! FD`) with one end of a socket pair at that descriptor. They talk over it:
//!
//! 1. The daemon sends its options (`Prepare`). The successor loads the
//!    model's files and binds a request socket of its own, which it answers
//!    with (`Prepared`).
//! 2. The daemon stops its service thread between two rounds and sends what
//!    the thread served, its HTTP listener and the connection its engine's
//!    process lives under, with the spec the engine serves, the admin's log
//!    and the video jobs (`Takeover`), and commands the engine to attach to
//!    the successor's request socket (`EngineCommand::Attach`).
//! 3. The successor goes on with the service, adopts the engine once it
//!    attaches, and accepts on the HTTP listener (`Serving`).
//! 4. The daemon stops accepting and finishes the requests and video jobs
//!    it took, relaying each later change of a job to the successor
//!    (`Job`); then it closes its connection to the engine without shutting
//!    it down and exits.
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
use prost::Message as _;
use proto::request::wire;
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

/// The descriptor a successor finds the handoff socket at.
pub const TAKEOVER_DESCRIPTOR: i32 = 3;

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
    /// Successor to daemon: it serves; its process id.
    Serving { pid: u32 },
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
    /// The configuration as TOML, as `uniserve serve --print-config` prints
    /// it.
    config: String,
    stated: Vec<String>,
    /// The API key, which printing leaves out.
    api_key: Option<String>,
    /// The daemon's process socket, which the successor serves from the
    /// handoff on and its engines and workers connect to.
    processes: String,
}

/// What a daemon hands its successor. The descriptors travel in this
/// order: the HTTP listener, the engine's control connection, the service's
/// descriptors, then each completed job's artifact.
#[derive(Debug, Serialize, Deserialize)]
struct Takeover {
    listener: HandedListener,
    /// The control connection's state, its descriptor travelling apart.
    control: (bool, Vec<u8>),
    service: HandedService,
    /// How many descriptors the service sends.
    service_descriptors: usize,
    engine: u64,
    next_engine: u64,
    /// The spec the engine serves (`wire::EngineSpec`, encoded).
    served: Vec<u8>,
    /// The identity the successor's first request takes.
    first_request: u64,
    admin: HandedAdmin,
    /// Each job's record, and its artifact's length once it completed.
    jobs: Vec<(VideoJob, Option<u64>)>,
    /// New requests wait; the engine sleeps, and whether its sleep paused
    /// them; the LoRA adapters the workers hold (`Leader::adopt_pause`).
    paused: bool,
    asleep: bool,
    sleep_paused: bool,
    adapters: std::collections::BTreeMap<String, crate::leader::Adapter>,
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
    /// the memory to the successor.
    pub(crate) handed_over: Arc<AtomicBool>,
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
        })
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
        leader: &Arc<Leader>,
        admin: &Arc<Admin>,
        jobs: &Arc<VideoJobs>,
    ) -> anyhow::Result<u32> {
        let executable = match executable {
            Some(executable) => std::path::PathBuf::from(executable),
            None => std::env::current_exe().context("finding this daemon's executable")?,
        };
        // No hot update or relaunch replaces the engine meanwhile.
        let _quiet = leader.quiesce().await;
        let handoff = leader.engine_handoff()?;
        let (ours, theirs) = UnixStream::pair().context("creating the handoff socket")?;
        let successor = spawn(&executable, theirs)?;
        info!(pid = successor, executable = %executable.display(), "started the successor daemon");
        let mut channel = Channel::new(ours);

        let config = crate::options::printed(&handoff.config)?;
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
                api_key: handoff.config.server_config.api_key.clone(),
                processes,
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
        let control = handoff
            .client
            .take_control()
            .context("the engine's control connection is not this daemon's")?;
        let control = control.hand_over();
        let first_request = handoff.client.next_request_id() + (1 << 32);
        let jobs_handed = jobs.hand_over(job_relay(channel.stream()));
        let mut descriptors: Vec<OwnedFd> = vec![listener_descriptor, control.descriptor];
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
        let borrowed: Vec<BorrowedFd<'_>> = descriptors.iter().map(|fd| fd.as_fd()).collect();
        send(
            &mut channel,
            &Message::Takeover(Box::new(Takeover {
                listener,
                control: (control.unix, control.received),
                service,
                service_descriptors: service_count,
                engine: handoff.engine,
                next_engine: handoff.next_engine,
                served: handoff.served.encode_to_vec(),
                first_request,
                admin: admin.hand_over(),
                jobs: job_records,
                paused: handoff.paused,
                asleep: handoff.asleep,
                sleep_paused: handoff.sleep_paused,
                adapters: handoff.adapters,
            })),
            &borrowed,
        )?;
        drop(descriptors);
        self.handed_over.store(true, Ordering::Release);
        handoff.client.attach(&requests);

        let pid = match recv(&mut channel).await? {
            (Message::Serving { pid }, _) => pid,
            (Message::Failed { reason }, _) => {
                anyhow::bail!("the successor took the service and gave up: {reason}")
            }
            (other, _) => anyhow::bail!("the successor answered {other:?}"),
        };
        info!(
            pid,
            "the successor serves; this daemon finishes its requests"
        );
        self.accepting.cancel();
        // The job relay writes on its own handle of the socket, which stays
        // open until this process exits.
        drop(channel);
        Ok(pid)
    }
}

/// Starts `executable serve --takeover` with `socket` at
/// [`TAKEOVER_DESCRIPTOR`]; returns its process id. The successor is not
/// this daemon's to stop: it outlives it.
fn spawn(executable: &std::path::Path, socket: UnixStream) -> anyhow::Result<u32> {
    let descriptor = socket.as_raw_fd();
    let mut command = std::process::Command::new(executable);
    command.args(["serve", "--takeover", &TAKEOVER_DESCRIPTOR.to_string()]);
    // SAFETY: the closure runs in the forked child before exec and calls
    // only `dup2`, which is async-signal-safe; `socket` stays open in the
    // parent until the spawn returned.
    unsafe {
        command.pre_exec(move || {
            if libc::dup2(descriptor, TAKEOVER_DESCRIPTOR) == -1 {
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
fn job_relay(stream: &storage::transport::Socket) -> crate::openai::video_jobs::Relay {
    let stream = match stream {
        storage::transport::Socket::Unix(stream) => stream.try_clone().ok(),
        storage::transport::Socket::Tcp(_) => None,
    };
    let channel = std::sync::Mutex::new(stream.map(Channel::new));
    Box::new(move |record, media| {
        let Ok(mut channel) = channel.lock() else {
            return;
        };
        let Some(channel) = channel.as_mut() else {
            return;
        };
        let artifact = media.map(|media| (media.len() as u64, artifact_descriptor(media)));
        let sent = match artifact {
            Some((length, Ok(descriptor))) => send(
                channel,
                &Message::Job {
                    record: Box::new(record.clone()),
                    artifact: Some(length),
                },
                &[descriptor.as_fd()],
            ),
            Some((_, Err(error))) => Err(error),
            None => send(
                channel,
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
    // The descriptor keeps the memory; the object's name is not needed.
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
    let deadline = Instant::now() + STEP_TIMEOUT;
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
            Instant::now() < deadline,
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
    let control = EngineControl::new(
        &config.request_socket(),
        prepare.processes.clone(),
        config.host_identity.clone(),
        config.served_model_name().to_owned(),
    )?;
    send(
        channel,
        &Message::Prepared {
            requests: control.request_address().to_owned(),
        },
        &[],
    )?;

    let (takeover, descriptors) = match recv(channel).await? {
        (Message::Takeover(takeover), descriptors) => (takeover, descriptors),
        (Message::Failed { reason }, _) => anyhow::bail!("the daemon gave up: {reason}"),
        (other, _) => anyhow::bail!("the daemon sent {other:?}"),
    };
    let Takeover {
        listener,
        control: (control_unix, control_received),
        service,
        service_descriptors,
        engine,
        next_engine,
        served,
        first_request,
        admin,
        jobs,
        paused,
        asleep,
        sleep_paused,
        adapters,
    } = *takeover;
    let mut descriptors = descriptors.into_iter();
    let mut next = || {
        descriptors
            .next()
            .context("the takeover lacks a descriptor")
    };
    let listener = Listener::resume(listener, next()?)?;
    let control_connection = Connection::resume(Handed {
        descriptor: next()?,
        unix: control_unix,
        received: control_received,
    })?;
    let service_fds = (0..service_descriptors)
        .map(|_| next())
        .collect::<anyhow::Result<Vec<_>>>()?;
    let socket = prepare
        .processes
        .strip_prefix("unix:")
        .map(std::path::PathBuf::from)
        .context("the daemon's process socket is not a Unix socket")?;
    let daemon = Daemon::resume(
        config.host_identity.clone(),
        Some((prepare.processes.clone(), socket)),
        service,
        service_fds,
    )?;
    let served = wire::EngineSpec::decode(served.as_slice())?;
    let sink = stats_sink(config.model().to_owned());
    let spec = served.clone();
    let (client, control) = tokio::task::spawn_blocking(move || {
        control
            .adopt(engine, next_engine, spec, control_connection, sink)
            .map(|client| (client, control))
    })
    .await
    .context("the engine's adoption did not complete")??;
    client.number_requests_from(first_request);
    let leader = Arc::new(Leader::finish(&config, prepared, control, client, served)?);
    leader.adopt_pause(paused, asleep, sleep_paused, adapters);

    let retire = Arc::new(AtomicBool::new(false));
    let admin = Admin::resume(shutdown.clone(), Arc::clone(&retire), admin);
    admin.record(
        "deployment",
        format!("serving {} after a hot restart", leader.served_model_name()),
    );
    admin.watch(&leader);
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
    send(
        channel,
        &Message::Serving {
            pid: std::process::id(),
        },
        &[],
    )?;
    info!("serving the deployment the previous daemon handed over");

    // The replaced daemon relays its jobs' changes until it exits.
    let relayed = Arc::clone(&video_jobs);
    let mut relay = std::mem::replace(channel, Channel::new(UnixStream::pair()?.0));
    tokio::spawn(async move {
        loop {
            match recv(&mut relay).await {
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
    let served = served.map(drop);
    crate::http::finish(&config, &leader, daemon, retire, handed_over, served).await
}
