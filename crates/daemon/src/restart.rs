//! Hot restart of a leader daemon: a successor process takes the deployment over while the
//! daemon it replaces finishes the requests it took, as Envoy's hot restart does. The successor
//! may run another build than the daemon it replaces.
//!
//! A change of the daemon's own options (`proto::reload::Effect::Daemon`) starts the successor
//! from the base executable the deployment names (`executable`), or by re-executing the daemon's
//! own image when it names none (`IMAGE serve --predecessor FD`), which inherits one end of a
//! socket pair at the descriptor the argument names. They talk over it:
//!
//! 1. The daemon sends its options, the window of versions of the engine interface it speaks,
//!    the versions of the engine interface its connections agreed on (its engines' links,
//!    their workers', its members' and its own clients') and the Storage Server protocol
//!    version of its claim (`Prepare`). The successor goes on with those connections as they
//!    are, so it gives up unless its windows hold every one of those versions
//!    (`proto::ENGINE_INTERFACE`, `storage::protocol::VERSIONS`) and the host's Storage Server
//!    serves it; it also agrees with the daemon on the version the handoff's messages are
//!    written in, the newest both speak. The options carry the records of every replica the
//!    takeover hands over (the arguments its engine serves with and its configuration), which
//!    the successor reads and resolves here, so a record of another build it cannot read ends
//!    the restart before anything crossed. The successor loads the model's files, with the
//!    serving epoch's processing program from the Build Server that serves the deployment, and
//!    binds a request socket of its own, which it responds with, with the agreed version
//!    (`Prepared`). Whatever it gives up on, the options it cannot read included, it states
//!    (`Failed`).
//! 2. The daemon stops its service thread between two rounds and sends what
//!    the thread served, its HTTP listener and the connection its engine's
//!    process lives under, with the arguments the engine serves with and the video jobs
//!    (`Takeover`), and commands the engine to attach to
//!    the successor's request socket (`EngineCommand::Attach`). The engine says hello there in
//!    the one version it speaks.
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
//! it. The engines are the same processes throughout and speak the version they agreed on with
//! the daemon that started them; an epoch update the successor runs afterwards starts engines of
//! its own build, which agree on the newest version both speak. The handoff's messages are part
//! of the engine interface: they change only by additions (a field with a default, a variant),
//! each written only when it holds another value than its default, so a successor of another
//! build within the window reads them; the configuration records they carry refuse a field
//! they do not know, which is why the successor reads them before the handover. A test reads
//! the handoff of the window's oldest version as that version's build wrote it
//! (`tests/fixtures/engine-interface-14/`). A connection that arrives
//! meanwhile waits in the listener's queue until one of the two accepts it. Until the successor
//! serves, a failure leaves the daemon serving as before, except after the service thread
//! stopped, which then fails the restart and the daemon with it: every check a successor of
//! another build may fail runs before it.
//!
//! The deployment's Build Server is part of what the daemon serves: until the handover, it is
//! the server the predecessor serves with, which the successor only reads from and never
//! starts, stops or replaces, so a restart the successor gives up leaves it as it was, the same
//! process with the same settings and no revision of another source tree installed. The
//! successor's own settings (`build_config`) take effect with the handover: its service starts
//! the node's processes at a server of those settings, and once it serves, its node has the
//! server serve with them before it converges the rest of the change (`Node::resume`).

use std::os::fd::{AsFd as _, AsRawFd as _, BorrowedFd, FromRawFd as _, OwnedFd};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use service::{Connection, Handed};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::api::openai::video_jobs::{VideoJob, VideoJobs};
use crate::config::Stated;
use crate::engine::control::EngineControl;
use crate::http::listener::{Listener, ListenerHandoff};
use crate::leader::{Leader, stats_sink};
use crate::supervision::{Daemon, ServiceHandoff};

pub(crate) mod requests;
pub(crate) mod streams;

/// The messages of the handoff socket.
#[derive(Debug, Serialize, Deserialize)]
enum Message {
    /// Daemon to successor: the options it serves with.
    Prepare(Box<Prepare>),
    /// Successor to daemon: its request socket, which the engine attaches
    /// to, and the version of the engine interface the handoff's messages are written in, the
    /// newest both daemons speak.
    Prepared { requests: String, version: u32 },
    /// Daemon to successor: everything it serves, with the descriptors of
    /// [`Takeover`].
    Takeover(Box<Takeover>),
    /// Successor to daemon: the responses in flight it took over, which the
    /// predecessor decides by: a response named here is the successor's from
    /// then on, and one the successor does not name resumes in the
    /// predecessor.
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

/// A change a hot restart hands its successor to converge: the revision it converges to, and
/// whether its writer forced a target a rollback marked unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Then {
    pub(crate) revision: u64,
    pub(crate) force: bool,
}

/// The options a daemon serves with.
#[derive(Debug, Serialize, Deserialize)]
struct Prepare {
    /// The configuration as TOML, as the revision's `config.toml` prints it.
    config: String,
    stated: Vec<String>,
    /// The API key, which printing leaves out.
    api_key: Option<String>,
    /// The daemon's process socket, which the successor serves from the
    /// handoff on and its engines and workers connect to.
    processes: String,
    /// The daemon's instance directory, whose sockets the successor binds over and which it
    /// owns once the handover completed.
    instance: std::path::PathBuf,
    /// The deployment's name, which names its persistent state the successor goes on with.
    deployment: String,
    /// Whether the configuration listens on another address than the
    /// daemon's listener: the successor binds it before the handover.
    rebind: bool,
    /// The window of versions of the engine interface the daemon speaks
    /// ([`proto::ENGINE_INTERFACE`]), over which the two agree on the version of the handoff's
    /// messages.
    versions: service::Versions,
    /// The versions of the engine interface the connections the daemon hands over agreed on:
    /// its engines' links, their workers', its members' and its own clients'. A successor whose
    /// window misses one gives up before anything is handed over.
    spoken: Vec<u32>,
    /// The version of the Storage Server protocol the daemon's claim agreed on, once an engine
    /// claimed the node: the claim's connection crosses as it is.
    storage: Option<u32>,
    /// The processing program the serving epoch processes requests through, which the
    /// successor loads from the deployment's Build Server: it restores every response's parser
    /// in it and processes new requests with it.
    processing: proto::module::PackageRevision,
    /// The records of the replicas the takeover hands over, in its order. The successor reads
    /// them, and resolves each configuration, before it answers [`Message::Prepared`], so a
    /// record of another build that this one cannot read ends the restart while the daemon
    /// serves on, instead of after the service crossed. The takeover carries the same records
    /// again, where a successor of a build before this field reads them; a predecessor of such
    /// a build states none, and its records are read in the takeover alone.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    replicas: Vec<ReplicaRecord>,
}

/// What a replica's engine serves with, as the predecessor hands it over: the records of the
/// handoff that the configuration's types read, which refuse a field they do not know.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReplicaRecord {
    /// The arguments the replica's engine process serves with.
    served: proto::config::EngineArgs,
    /// The replica's configuration, as the revision's `config.toml` prints it.
    config: String,
}

/// What a daemon hands its successor. The descriptors travel in this
/// order: the HTTP listener, the engine's control connection, the service's
/// descriptors, then each completed job's artifact.
#[derive(Debug, Serialize, Deserialize)]
struct Takeover {
    /// The responses in flight, which the successor takes over before it
    /// serves; their sockets follow the other descriptors, in this order.
    streams: Vec<streams::StreamHandoff>,
    listener: ListenerHandoff,
    replicas: Vec<ReplicaHandoff>,
    adapters: std::collections::BTreeMap<String, crate::leader::Adapter>,
    service: ServiceHandoff,
    service_descriptors: usize,
    jobs: Vec<(VideoJob, Option<u64>)>,
    /// The change that restarted the daemon: the successor converges the rest of it.
    then: Option<Then>,
    /// Whether the leader's admissions were paused (new requests waited) when the predecessor
    /// handed over: the successor's admissions start as the predecessor's ended, so a hot
    /// restart never reopens them.
    admissions_paused: bool,
    /// The modules the serving epoch runs. A configuration names a part of
    /// them and the store fills the rest, so the successor cannot derive them.
    modules: Option<proto::module::ModuleSet>,
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
    config: String,
    paused: bool,
    asleep: bool,
    sleep_paused: bool,
    unavailable: Vec<proto::admin::Unavailable>,
}

/// What a hot restart needs of the daemon it restarts.
pub(crate) struct Restarter {
    /// The daemon's service, which `serve` stops once the restart handed it
    /// over.
    pub(crate) daemon: Arc<std::sync::Mutex<Option<Daemon>>>,
    /// A duplicate of the HTTP listener, and what the successor needs of it:
    /// its descriptors, in the order [`Listener::resume`] takes them.
    listener: std::sync::Mutex<Option<(ListenerHandoff, Vec<OwnedFd>)>>,
    /// The daemon's instance directory, which the successor goes on with.
    instance: std::path::PathBuf,
    /// The deployment's name, which names its persistent state.
    deployment: String,
    stated: Stated,
    /// Cancelled once the successor serves: this daemon stops accepting.
    pub(crate) accepting: CancellationToken,
    /// Set once the deployment was handed over: this daemon detaches from
    /// the engine instead of shutting it down, and leaves the service and
    /// the storage to the successor.
    pub(crate) handoff_sent: Arc<AtomicBool>,
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
    /// boundary and hands to the successor, attached once
    /// the HTTP server serves.
    handoff: std::sync::Mutex<Option<Arc<crate::http::handoff::Handoff>>>,
}

impl Restarter {
    /// The restarter of the deployment `deployment` serving with `stated` on the client
    /// listener of which `listener` is a duplicate, whose daemon service `daemon` holds, on a
    /// node whose instance directory is `instance`; `accepting` stops the node accepting once a
    /// successor serves.
    pub(crate) fn new(
        daemon: Arc<std::sync::Mutex<Option<Daemon>>>,
        listener: (ListenerHandoff, Vec<OwnedFd>),
        instance: &std::path::Path,
        deployment: String,
        stated: Stated,
        accepting: CancellationToken,
    ) -> Self {
        Self {
            daemon,
            listener: std::sync::Mutex::new(Some(listener)),
            instance: instance.to_owned(),
            deployment,
            stated,
            accepting,
            handoff_sent: Arc::new(AtomicBool::new(false)),
            decided: tokio::sync::watch::channel(None).0,
            in_flight: Arc::new(AtomicBool::new(false)),
            handoff: std::sync::Mutex::new(None),
        }
    }

    /// The registry of the responses in flight, which the HTTP server
    /// registers every handoffable response with.
    pub(crate) fn attach_handoff(&self, handoff: Arc<crate::http::handoff::Handoff>) {
        let mut slot = service::lock(&self.handoff);
        *slot = Some(handoff);
    }

    /// The registry of the responses in flight, once the HTTP server serves.
    pub(crate) fn handoff(&self) -> Option<Arc<crate::http::handoff::Handoff>> {
        service::lock(&self.handoff).clone()
    }

    /// Restarts the daemon as a successor, as the module describes, up to the successor serving;
    /// returns its process id. The successor runs the base executable `config` names
    /// (`executable`), or this daemon's own image. Video job changes go on to the successor until
    /// this daemon exits.
    ///
    /// # Errors
    ///
    /// Fails when the successor does not start or gives up; before the
    /// service was handed over, this daemon serves on.
    pub(crate) async fn restart(
        &self,
        config: Option<crate::config::Config>,
        then: Option<Then>,
        leader: &Arc<Leader>,
        jobs: &Arc<VideoJobs>,
    ) -> anyhow::Result<u32> {
        self.in_flight.store(true, Ordering::Release);
        let result = self.hand_over(config, then, leader, jobs).await;
        // The restart is concluded however it ended: a daemon waiting to exit
        // stops waiting here, and a successor that adopted the deployment is the
        // only case in which this one may let go of it. The decision is stated
        // before the flag clears, so a waiter that sees the flag still set also
        // sees the decision.
        let _ = self.decided.send(Some(result.is_ok()));
        self.in_flight.store(false, Ordering::Release);
        result
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
        config: Option<crate::config::Config>,
        then: Option<Then>,
        leader: &Arc<Leader>,
        jobs: &Arc<VideoJobs>,
    ) -> anyhow::Result<u32> {
        // No epoch change, rollback or relaunch replaces the engine meanwhile.
        let _quiet = leader.quiesce().await;
        // The parked standbys are this daemon's processes: the successor
        // numbers and starts its own once it has adopted the engines, so ending
        // them here keeps a second engine's memory off the device across the
        // handover instead of leaving the old daemon's copies parked behind it.
        leader.end_standbys().await;
        let handoff = leader.engine_handoff()?;
        // The successor serves with the daemon's options the change
        // states, over the configuration the engine serves.
        let options = config.unwrap_or_else(|| handoff[0].config.clone());
        let image = options
            .executable
            .clone()
            .unwrap_or_else(|| std::path::PathBuf::from(RUNNING_IMAGE));
        let (ours, theirs) = UnixStream::pair().context("creating the handoff socket")?;
        // The persistent reply reader must yield to the continuations it
        // wakes, including when the handoff channel has no further traffic.
        ours.set_nonblocking(true)?;
        let successor = spawn(theirs, &image)?;
        info!(pid = successor, image = %image.display(), "started the successor daemon");
        let mut channel = Connection::new(ours)?;

        // The listener binds the daemon's own address (`admin_config`): a
        // change of it is the rebind this successor makes before the
        // handover, and the previous daemon keeps serving if the bind fails.
        let address = |admin: &crate::config::AdminConfig| {
            (admin.host.clone(), admin.port, admin.uds.clone())
        };
        let rebind = address(&options.admin_config) != address(&handoff[0].config.admin_config);
        let config = crate::config::printed(&options)?;
        let processes = ::service::lock(&self.daemon)
            .as_ref()
            .and_then(Daemon::address)
            .context("this daemon serves no process socket")?
            .to_owned();
        // No epoch change runs meanwhile (`quiesce`), so the serving processor's program is the
        // one every response in flight parses in.
        let processing = leader.processor().program().revision().clone();
        // The versions of what crosses: no epoch change, relaunch or standby starts an engine
        // meanwhile, so they hold until the handover.
        let spoken = ::service::lock(&self.daemon)
            .as_ref()
            .map(Daemon::spoken)
            .unwrap_or_default();
        let engines = handoff
            .iter()
            .flat_map(|replica| replica.client.interfaces().collect::<Vec<_>>());
        let versions: std::collections::BTreeSet<u32> =
            spoken.engine.iter().copied().chain(engines).collect();
        // Each replica's records, which the successor reads before the handover and the takeover
        // carries again: no epoch change runs meanwhile, so they hold until it.
        let records = handoff
            .iter()
            .map(|replica| {
                Ok(ReplicaRecord {
                    served: replica.served.clone(),
                    config: crate::config::printed(&replica.config)?,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        send(
            &mut channel,
            &Message::Prepare(Box::new(Prepare {
                config,
                stated: self.stated.paths(),
                api_key: options.server_config.api_key.clone(),
                processes,
                instance: self.instance.clone(),
                deployment: self.deployment.clone(),
                rebind,
                versions: proto::ENGINE_INTERFACE,
                spoken: versions.into_iter().collect(),
                storage: spoken.storage,
                processing,
                replicas: records.clone(),
            })),
            &[],
        )?;
        let requests = match recv(&mut channel).await? {
            (Message::Prepared { requests, version }, _) => {
                // The successor agreed on the newest version both windows hold, which this
                // daemon writes the handoff in.
                anyhow::ensure!(
                    proto::ENGINE_INTERFACE.speaks(version),
                    "the successor agreed on version {version} of the engine interface, which \
                     this daemon (versions {}) does not speak",
                    proto::ENGINE_INTERFACE
                );
                info!(version, "the successor agreed on the handoff's version");
                requests
            }
            (Message::Failed { reason }, _) => anyhow::bail!("the successor gave up: {reason}"),
            (other, _) => anyhow::bail!("the successor responded {other:?}"),
        };

        // From here the service is the successor's.
        let (service, service_descriptors) = {
            let mut daemon = ::service::lock(&self.daemon);
            daemon
                .as_mut()
                .context("the daemon's service is not running")?
                .hand_over()?
        };
        let (listener, mut descriptors) = ::service::lock(&self.listener)
            .take()
            .context("the HTTP listener was handed over already")?;
        let mut replicas = Vec::new();
        for (replica, record) in handoff.iter().zip(records) {
            let control = match replica.client.take_control() {
                Some(control) => {
                    let control = control
                        .hand_over()
                        .context("handing over an engine's control connection")?;
                    descriptors.push(control.descriptor);
                    Some((control.unix, control.received))
                }
                None => None,
            };
            replicas.push(ReplicaHandoff {
                control,
                engine: replica.engine,
                next_engine: replica.next_engine,
                served: record.served,
                config: record.config,
                paused: replica.paused,
                asleep: replica.asleep,
                sleep_paused: replica.sleep_paused,
                unavailable: replica.unavailable.clone(),
            });
        }
        // Stop every response in flight at its next engine event boundary, so
        // its state and its socket travel with the takeover.
        let mut paused = self.pause_streams().await;
        let mut stream_descriptors = std::mem::take(&mut paused.descriptors);
        let stream_records = std::mem::take(&mut paused.records);
        let writer = requests::writer(&channel)?;
        // Serialize the snapshot before any later job update uses the same
        // stream. Continuation requests share this writer after handoff.
        {
            let mut sending = service::lock(&writer);
            let jobs_handoff = jobs.hand_over(job_relay(Arc::clone(&writer)));
            let service_count = service_descriptors.len();
            descriptors.extend(service_descriptors);
            let mut job_records = Vec::with_capacity(jobs_handoff.len());
            for (record, media) in jobs_handoff {
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
                    adapters: service::lock(&leader.replicas.adapters).clone(),
                    service,
                    service_descriptors: service_count,
                    jobs: job_records,
                    then,
                    admissions_paused: leader.paused(),
                    modules: leader.epochs().serving().map(|(_, record)| {
                        proto::module::ModuleSet {
                            engine: record.engine,
                            kernels: record.kernels,
                            policy: record.policy,
                            programs: Default::default(),
                            processing: record.processing,
                        }
                    }),
                })),
                &borrowed,
            )?;
        }
        // The predecessor's copies of what it handed over stay open until the successor
        // responds: closing them at the send was the earliest moment a peer could see its own
        // interface go away, and the successor had by then neither resumed it nor claimed the
        // deployment.
        self.handoff_sent.store(true, Ordering::Release);
        // From here the successor owns these engines: this daemon's watch sees their clients
        // go away with the handover and must not read that as a loss.
        for replica in leader.replicas() {
            replica
                .handoff_sent
                .store(true, std::sync::atomic::Ordering::Release);
        }
        for replica in handoff {
            replica.client.attach(&requests);
        }

        // The successor responds once: it either took the responses in flight
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
            (other, _) => anyhow::bail!("the successor responded {other:?}"),
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
                .context("the replica of the handoff has no client")?;
            let control = Arc::clone(
                replica
                    .control
                    .get()
                    .context("the replica of the handoff has no request socket")?,
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
                match recv(&mut channel).await {
                    Ok((Message::Resumed { id, attachment }, _)) => {
                        continuations.respond(id, attachment)
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

/// The running process's own executable image, as the kernel holds it: what the successor of a
/// hot restart executes when the deployment names no `executable`, so it is the running build
/// even when the file it was started from was replaced since.
const RUNNING_IMAGE: &str = "/proc/self/exe";

/// Starts the successor, `serve --predecessor FD` of the base executable `image`, inheriting
/// `socket` at the descriptor `FD` names; returns its process id. The successor is not this
/// daemon's to stop: it outlives it.
fn spawn(socket: UnixStream, image: &std::path::Path) -> anyhow::Result<u32> {
    let descriptor = socket.as_raw_fd();
    let mut command = std::process::Command::new(image);
    command.args(["serve", "--predecessor", &descriptor.to_string()]);
    // SAFETY: the closure runs in the forked child before exec and calls only `fcntl`, which is
    // async-signal-safe; `socket` stays open in the parent until the spawn returned. Only the
    // child's copy of the descriptor loses close-on-exec, so no other child inherits it.
    unsafe {
        command.pre_exec(move || {
            if libc::fcntl(descriptor, libc::F_SETFD, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let child = command
        .spawn()
        .with_context(|| format!("starting the successor daemon from {}", image.display()))?;
    drop(socket);
    Ok(child.id())
}

/// The relay of video job changes over the handoff socket `stream`: each
/// change is one `Job` message, with the artifact's descriptor once the job
/// completed. A change that cannot be sent is logged: the successor holds
/// the job's last record it received.
fn job_relay(writer: requests::Writer) -> crate::api::openai::video_jobs::Relay {
    Box::new(move |record, media| {
        let mut channel = service::lock(&writer);
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
    media: &Arc<crate::api::openai::video_jobs::RetainedMedia>,
) -> anyhow::Result<OwnedFd> {
    if let Some(descriptor) = media.descriptor() {
        return Ok(descriptor.try_clone_to_owned()?);
    }
    let mut file = proto::shared_memory::anonymous(media.len())
        .context("creating shared memory for a video artifact")?;
    std::io::Write::write_all(&mut file, media)?;
    // The successor maps it as a medium, which is sealed against writes and size changes.
    proto::shared_memory::seal(&file).context("sealing a video artifact")?;
    Ok(OwnedFd::from(file))
}

/// Sends `message` with `descriptors` and waits until it left.
fn send(
    channel: &mut Connection,
    message: &Message,
    descriptors: &[BorrowedFd<'_>],
) -> anyhow::Result<()> {
    channel
        .send_with(message, descriptors)
        .context("writing to the handoff socket")?;
    channel.flush().context("writing to the handoff socket")
}

/// The other daemon's next message and its descriptors, read as the socket becomes readable.
/// The wait ends with the message, or with the socket's end when the other daemon gave up or
/// exited.
async fn recv(channel: &mut Connection) -> anyhow::Result<(Message, Vec<OwnedFd>)> {
    loop {
        match channel.try_recv_with::<Message>() {
            Ok(Some(frame)) => return Ok(frame),
            Ok(None) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                anyhow::bail!("the other daemon closed the handoff socket")
            }
            Err(error) => return Err(error).context("reading the handoff socket"),
        }
        // Readiness of a descriptor this task borrows for the wait alone: the connection keeps
        // owning it, and the registration ends before the next read.
        let readable =
            tokio::io::unix::AsyncFd::with_interest(channel.as_fd(), tokio::io::Interest::READABLE)
                .context("waiting on the handoff socket")?;
        readable
            .readable()
            .await
            .context("waiting on the handoff socket")?
            .retain_ready();
    }
}

/// The handoff socket's messages are MessagePack with named fields in the workspace's frames
/// (`service::msgpack`).
impl service::Message for Message {
    fn encode(&self) -> std::io::Result<Vec<u8>> {
        service::msgpack::encode(self)
    }

    fn decode(payload: &[u8]) -> std::io::Result<Self> {
        service::msgpack::decode(payload)
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
    let mut channel = Connection::new(stream)?;
    // Every failure is the predecessor's reason to serve on, the options it cannot read
    // included.
    match successor(&mut channel, shutdown).await {
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

/// The successor's side: the options first, then everything after them.
async fn successor(channel: &mut Connection, shutdown: CancellationToken) -> anyhow::Result<()> {
    let prepare = match recv(channel).await? {
        (Message::Prepare(prepare), _) => prepare,
        (other, _) => anyhow::bail!("the daemon sent {other:?} first"),
    };
    // The successor's whole bring-up is one future, which lives on the heap rather than in the
    // caller's frame.
    Box::pin(prepared_successor(channel, *prepare, shutdown)).await
}

/// The successor's side once it has the options.
async fn prepared_successor(
    channel: &mut Connection,
    prepare: Prepare,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    // The engines and every other connection go on through this daemon in the versions they
    // agreed on, so a version this build does not speak ends the restart here, while the
    // predecessor serves as it was; so does a Storage Server this build cannot speak to, since
    // the claims cross on its connections.
    let version = takes_over(
        &prepare,
        proto::ENGINE_INTERFACE,
        ::storage::protocol::VERSIONS,
    )?;
    info!(
        version,
        engines = ?prepare.spoken,
        storage = ?prepare.storage,
        "this daemon speaks every version the deployment's connections agreed on"
    );
    let mut config = handed_over(&prepare.config)?;
    config.server_config.api_key = prepare.api_key;
    let stated = Stated::from_paths(prepare.stated);
    // The records the takeover hands over were read with the options; each configuration is
    // resolved as the adoption resolves it after the handover, so a record of another build
    // this one refuses ends the restart while the predecessor serves.
    for (index, record) in prepare.replicas.iter().enumerate() {
        let options = replica_options(&record.config, &config)
            .with_context(|| format!("reading the configuration of replica {index}"))?;
        crate::epoch::Epoch::of(options)
            .with_context(|| format!("resolving the configuration of replica {index}"))?;
    }
    let deployment = prepare.deployment.clone();
    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        storage_server_speaks(&storage::socket(
            &service::RuntimeDirectory::of_user()?,
            &deployment,
        )?)
    })
    .await
    .context("the check of the Storage Server's protocol did not complete")??;
    // Another listening address is bound before the handover, so that a
    // refused bind leaves the previous daemon serving; the listener it hands
    // over then closes once the previous daemon stops accepting.
    let rebound = if prepare.rebind {
        let listener = Listener::bind(&config.admin_config)
            .await
            .context("binding the configuration's listening address")?;
        info!(address = %listener.endpoint()?, "listening on the configuration's address");
        Some(listener)
    } else {
        None
    };
    // The deployment's Build Server is the predecessor's until the handover: the successor
    // reaches it at its socket as it serves, whatever settings it serves with, and starts,
    // stops or replaces none, so giving up from here on leaves it as it was. This daemon's
    // settings take effect once the handover committed.
    let control = EngineControl::new(
        &config.request_socket(&prepare.instance),
        prepare.processes.clone(),
        config.leader_host(),
        prepare.deployment.clone(),
        build_server::socket(&service::RuntimeDirectory::of_user()?, &prepare.deployment)
            .with_context(|| {
                format!("finding the Build Server socket of {}", prepare.deployment)
            })?,
    )?;
    // The successor processes requests through the serving epoch's processing program, which
    // the deployment's Build Server holds at the revision the predecessor names; a deployment
    // whose Build Server does not serve has none to read it from, and the restart ends here.
    let socket = control.build_server().to_owned();
    let revision = prepare.processing.clone();
    let program =
        tokio::task::spawn_blocking(move || crate::api::program::fetch(&socket, &revision))
            .await
            .context("loading the processing program did not complete")??;
    let prepared = Leader::prepare(&program, &config, &stated).await?;
    send(
        channel,
        &Message::Prepared {
            requests: control.request_address().to_owned(),
            version,
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
        jobs,
        then,
        admissions_paused,
        modules,
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
    let resumed =
        Listener::resume(listener, &mut next).context("resuming the handoff's HTTP listener")?;
    let listener = match rebound {
        Some(rebound) => {
            drop(resumed);
            rebound
        }
        None => resumed,
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
    // The process id of each engine the successor adopts, which its `process` event states: the
    // service hands every process it supervises over under its launch generation.
    let adopted_pids: std::collections::BTreeMap<u64, u32> = replicas
        .iter()
        .filter_map(|(replica, _)| {
            let pid = service.pid(crate::engine::control::ENGINE_GROUP, 0, replica.engine)?;
            Some((replica.engine, pid))
        })
        .collect();
    // The node's service is this daemon's from here, and starts the node's processes at a Build
    // Server of this daemon's settings, not of the predecessor's it handed over.
    let daemon = Daemon::resume(
        config.leader_host(),
        Some(prepare.processes.clone()),
        config.build_config.clone(),
        service,
        service_fds,
    )
    .context("resuming the node's daemon service")?;
    // The deployment's revisions and its observation store, which the adopted engines' processes
    // and reports write from the first one on.
    let revisions = Arc::new(crate::policy::revision::Revisions::open(
        crate::policy::revision::directory(&prepare.deployment)?,
    )?);
    let recorder = crate::observation::record::Recorder::new(
        crate::observation::Store::new(
            Arc::new(crate::observation::System),
            crate::observation::Retention::default(),
        )
        .context("creating the observation store")?,
        revisions.current()?.unwrap_or(0),
    );
    let (updates, awake) = daemon
        .updater(prepare.deployment.clone(), recorder.clone())
        .context("the leader's daemon commands no engine")?;
    let epoch_records = Arc::clone(updates.epochs());
    Daemon::advance_updates(Arc::clone(&updates), awake)
        .context("starting the updater's thread")?;
    let control = Arc::new(control);
    let adopting = Arc::clone(&control);
    let daemon_options = config.clone();
    let reporting = recorder.clone();
    let adopted = tokio::task::spawn_blocking(move || {
        replicas
            .into_iter()
            .enumerate()
            .map(|(index, (replica, connection))| {
                let options = replica_options(&replica.config, &daemon_options)?;
                let served = replica.served.clone();
                let observer = Arc::new(crate::observe::Observer::new(crate::observe::span(
                    &options.update_config,
                )));
                let sink = stats_sink(Arc::clone(&observer), reporting.clone(), index);
                let client = adopting.adopt(
                    replica.engine,
                    replica.next_engine,
                    served.epoch,
                    connection,
                    sink,
                )?;
                Ok((replica, options, served, observer, client))
            })
            .collect::<anyhow::Result<Vec<_>>>()
    })
    .await
    .context("the engines' adoption did not complete")??;
    let mut adopted = adopted.into_iter();
    let (handoff, options, served, observer, client) =
        adopted.next().context("the takeover names no replicas")?;
    let served_epoch = served.epoch;
    let interface = client.interface();
    let leader = Arc::new(
        Leader::finish(
            &config,
            prepared,
            Arc::clone(&control),
            client,
            served,
            observer,
            updates,
            recorder.clone(),
        )
        .context("finishing the deployment the takeover names")?,
    );
    // The successor adopts the epoch that serves, and below the epoch each other replica serves.
    {
        let (record, process) = adopted_epoch(
            served_epoch,
            (handoff.engine, interface),
            adopted_pids.get(&handoff.engine).copied(),
            &options,
            modules.clone(),
        )?;
        epoch_records.record(served_epoch, record);
        // The successor's observation store starts empty: the process it adopted is its first
        // `process` event.
        recorder.process(&options.leader_host(), &process);
        epoch_records.adopt_process(process);
    }
    if admissions_paused {
        // The predecessor was gated when it handed over, so the successor's
        // gate starts closed: the in-process hold is re-asserted here (its
        // `adopt_pause` already carried the per-replica state, and this is the
        // one place that states the deployment's), and the frontend's gate,
        // once a frontend process holds one, is closed at the same point.
        for replica in leader.replicas() {
            replica.pause();
        }
        info!("the successor continues gated: admissions stay closed across the handoff");
    }
    let first = Arc::clone(&leader.replicas()[0]);
    *service::lock(&first.config) = options;
    first.adopt_pause(handoff.paused, handoff.asleep, handoff.sleep_paused);
    leader.updates.adopt_unavailable(handoff.unavailable);
    // The successor adopts the engines and then keeps a standby of its own, or the deployment
    // would serve without one until the next restart. Its engine resolves at the deployment's
    // Build Server, which serves with this daemon's settings only once the node had it serve
    // with them, after the handover: the node starts the standby then, before it converges the
    // change that restarted the daemon (`Node::resume`). Without such a change the options are
    // the predecessor's own, whose Build Server serves already.
    if then.is_none() {
        first.start_standby();
    }
    for (index, (handoff, options, served, observer, client)) in adopted.enumerate() {
        let interface = client.interface();
        let replica = crate::leader::Replica::new(
            &options,
            Arc::new(client),
            observer,
            recorder.clone(),
            index + 1,
            Arc::downgrade(&leader.replicas),
            Arc::clone(&leader.replicas.adapters),
        );
        let _ = replica.control.set(Arc::clone(&control));
        // A replica an epoch update moved apart from the first serves an epoch of its own.
        if epoch_records.recorded(served.epoch).is_none() {
            let (record, process) = adopted_epoch(
                served.epoch,
                (handoff.engine, interface),
                adopted_pids.get(&handoff.engine).copied(),
                &options,
                modules.clone(),
            )?;
            epoch_records.record(served.epoch, record);
            recorder.process(&options.leader_host(), &process);
            epoch_records.adopt_process(process);
        }
        *service::lock(&replica.served) = Some(served);
        replica.adopt_pause(handoff.paused, handoff.asleep, handoff.sleep_paused);
        leader.updates.adopt_unavailable(handoff.unavailable);
        replica.monitor();
        leader
            .replicas
            .serving
            .send_modify(|replicas| replicas.push(replica));
    }

    *service::lock(&leader.replicas.adapters) = adapters;
    // The deployment's end is its node's as the predecessor's node lived.
    let token = shutdown.clone();
    // The handover completed: the predecessor's instance directory is this daemon's now, and
    // its admin socket is bound over in it.
    let instance = service::Instance::resume(prepare.instance.clone());
    let admin = crate::http::admin::Listener::bind(instance.path())?;
    let mut ready = crate::http::Ready::new(None, instance.path());
    {
        let endpoints = ready.endpoints();
        endpoints.client = Some(listener.endpoint()?);
        endpoints.admin = Some(admin.endpoint());
    }
    let video_jobs = Arc::new(VideoJobs::new(
        config.video_job_config.clone(),
        token.child_token(),
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
    // decides these before it stops writing anything.
    let mut adopted = Vec::new();
    if !streams.is_empty() {
        let adopting = Arc::clone(&leader);
        let mut sockets = std::iter::from_fn(|| next().ok());
        let keep_alive = config.server_config.stream_keep_alive;
        let adoption = streams::adopt(&adopting, streams, &video_jobs, keep_alive, &mut || {
            sockets
                .next()
                .context("the takeover lacks a response's descriptor")
        })?;
        send(
            channel,
            &Message::Adopted {
                pid: std::process::id(),
                streams: adoption.streams,
            },
            &[],
        )?;
        adopted = adoption.adopted;
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
    let mut relay = std::mem::replace(channel, Connection::new(UnixStream::pair()?.0)?);
    let relay = tokio::spawn(async move {
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
                            &mut service::lock(&writer),
                            &Message::Resumed { id, attachment },
                            &[],
                        ) {
                            warn!("responding to a predecessor continuation failed: {error:#}");
                        }
                    });
                }
                Ok((other, _)) => warn!("the replaced daemon sent {other:?}"),
                // The replaced daemon exited.
                Err(_) => return,
            }
        }
    });

    // From here the successor's node serves the adopted deployment as any node serves one.
    let start = crate::http::node::Start {
        options: config.clone(),
        deployment: prepare.deployment.clone(),
        revisions,
        recorder,
        instance,
        ready,
        shutdown,
    };
    let brought = crate::http::node::Brought {
        config,
        stated,
        leader,
        jobs: video_jobs,
        adopted,
        daemon: Some(daemon),
        token,
        then,
    };
    let ended = crate::http::node::run(
        start,
        listener,
        admin,
        crate::http::node::First::Serving(Box::new(brought)),
    )
    .await;
    if ended.as_ref().is_ok_and(|ended| ended.handed_over) {
        // A second hot restart may finish this daemon's HTTP requests while
        // its predecessor still has streams. Keep forwarding their admission
        // to the latest owner until that predecessor drains and disconnects.
        let _ = relay.await;
    } else {
        relay.abort();
    }
    ended.map(drop)
}

/// The epoch record and the process row of an epoch a restarting daemon
/// adopts: the build config and the modules the adopted configuration
/// fills, serving, at the launch generation the engine reported (which is the
/// identity it runs under) with the version of the engine interface it attached in, `(engine,
/// interface)`, with the process id `pid` the service handed over for that launch on this
/// node. The record's own generation stays `None`:
/// that field is the MODEL generation the adopted engine numbers for its
/// modules, and a daemon that adopted it never saw that number — the launch
/// generation belongs to the process row alone.
///
/// Without the record the updater says no epoch serves and refuses the
/// next switch; without the row the next launch would be numbered 1 again,
/// under a live engine's generation.
fn adopted_epoch(
    epoch: u64,
    (engine, interface): (u64, Option<u32>),
    pid: Option<u32>,
    options: &crate::config::Config,
    modules: Option<proto::module::ModuleSet>,
) -> anyhow::Result<(crate::epoch::EpochRecord, proto::admin::ManagedProcess)> {
    let deployment = crate::epoch::Epoch::of(options.clone())?;
    let modules = modules.unwrap_or_else(|| deployment.modules.clone());
    let record = crate::epoch::EpochRecord {
        config: deployment.engine.clone(),
        engine: modules.engine,
        kernels: modules.kernels,
        policy: modules.policy,
        processing: modules.processing,
        previous: None,
        status: proto::admin::EpochState::Serving,
        generation: None,
    };
    let process = proto::admin::ManagedProcess {
        group: crate::engine::control::ENGINE_GROUP.to_owned(),
        rank: 0,
        pid,
        epoch: Some(epoch),
        build: deployment.engine.executable.clone(),
        generation: engine,
        state: proto::admin::ProcessState::Serving,
        interface,
    };
    Ok((record, process))
}

/// The options a replica serves with in the successor: its configuration `printed` as the
/// predecessor handed it over, with the successor's own daemon options `daemon` in place of the
/// predecessor's.
///
/// # Errors
///
/// When this build cannot read the configuration: an option it does not know, or a value its
/// option does not take.
fn replica_options(
    printed: &str,
    daemon: &crate::config::Config,
) -> anyhow::Result<crate::config::Config> {
    let mut options = handed_over(printed)?;
    proto::reload::Reload::take(&mut options, daemon, proto::reload::Effect::Daemon);
    Ok(options)
}

/// A configuration a predecessor handed over, `printed` as it prints its options and each
/// replica's configuration, read as a package's `config.toml` is, with its checkpoint stated
/// once. The leader of an earlier build names every replica's model in `model`, which for a
/// deployment that states `model_dir` is that directory again: such a `model` is the leader's
/// name of the directory, not a statement of the package, and is dropped, so the options are
/// validated as a start validates the package. A `model` that names anything else beside
/// `model_dir` is kept, and refused as a start refuses it.
///
/// # Errors
///
/// As [`crate::config::parse`].
fn handed_over(printed: &str) -> anyhow::Result<crate::config::Config> {
    let (mut options, _) = crate::config::parse(printed)?;
    let directory = options
        .model_dir
        .as_deref()
        .and_then(std::path::Path::to_str);
    if directory.is_some() && options.model.as_deref() == directory {
        options.model = None;
    }
    Ok(options)
}

/// The version of the engine interface a successor of the windows `engine` and `storage` agrees
/// on with the predecessor that sent `prepare`, the newest both speak, when the successor speaks
/// every version the predecessor's connections agreed on: those of the engine interface on its
/// engines', members' and clients' connections, and the Storage Server protocol's on its claim.
///
/// # Errors
///
/// Why the successor cannot take the deployment over: the windows of the engine interface share
/// no version, or a connection's version lies outside the successor's windows.
fn takes_over(
    prepare: &Prepare,
    engine: service::Versions,
    storage: service::Versions,
) -> anyhow::Result<u32> {
    let version = engine.agree(prepare.versions).with_context(|| {
        format!(
            "the daemon speaks versions {} of the engine interface and this successor {engine}, \
             which share none",
            prepare.versions
        )
    })?;
    if let Some(unspoken) = prepare
        .spoken
        .iter()
        .find(|version| !engine.speaks(**version))
    {
        anyhow::bail!(
            "the deployment's connections speak version {unspoken} of the engine interface, which \
             this successor (versions {engine}) does not speak"
        );
    }
    if let Some(claim) = prepare.storage
        && !storage.speaks(claim)
    {
        anyhow::bail!(
            "the claim at the host's Storage Server speaks version {claim} of its protocol, which \
             this successor (versions {storage}) does not speak"
        );
    }
    Ok(version)
}

/// Whether the host's Storage Server at `socket` serves this build: the successor's own
/// connections to it agree on a version of its protocol ([`storage::protocol::VERSIONS`]),
/// and a server whose window shares none refuses this build's hello with the typed version
/// refusal, which names both windows. A host where no server listens has none to speak to yet.
///
/// # Errors
///
/// [`storage::Error::Version`] when the server speaks no version of this build's window, and
/// any other failure to reach a server that listens.
fn storage_server_speaks(socket: &std::path::Path) -> anyhow::Result<()> {
    match storage::Client::connect(socket, storage::Role::Operator) {
        Ok(_) => Ok(()),
        Err(storage::Error::Io(error))
            if matches!(
                error.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
            ) =>
        {
            Ok(())
        }
        Err(error) => Err(anyhow::Error::new(error).context(format!(
            "the host's Storage Server at {} does not serve this daemon",
            socket.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use storage::protocol::{
        ClientMessage, Refusal, ServerMessage, VERSIONS, VersionRefusal, Welcome,
    };

    use super::*;

    /// An update's report, which the takeover carries to the successor, crosses the handoff's
    /// MessagePack whole once its switch ended: the codec has no 128-bit integers, so a field of
    /// one would fail every hot restart taken after an update.
    #[test]
    fn an_update_report_crosses_the_handoff() {
        use proto::admin::{ChangeForm, Phase, UpdateReport};

        let report = UpdateReport {
            change: ChangeForm::Epoch,
            update: 3,
            phase: Phase::Trial,
            epoch: 2,
            rollback: false,
            switch_ms: Some(1_234),
            trial_s: Some(4.5),
            deadline_s: None,
            failure: None,
            uncompared: Vec::new(),
            claim_failed: None,
            generation: None,
            epoch_form: None,
            trial: None,
            transfers: None,
        };
        let encoded = service::msgpack::encode(&report).expect("the report encodes");
        let decoded: UpdateReport = service::msgpack::decode(&encoded).expect("the report decodes");
        assert_eq!(decoded, report);
    }

    /// The adopted epoch states the configuration it serves, and its process
    /// row the launch generation its engine reported, so a later switch
    /// compares against it and numbers its engines above it. The record's
    /// generation is the model generation, which the adopting daemon did not
    /// see and does not invent.
    #[test]
    fn an_adopted_epoch_records_the_build_it_serves_and_the_generation_it_runs_at() {
        let options = crate::config::Config::default();
        let (record, process) = adopted_epoch(7, (12, Some(14)), Some(4321), &options, None)
            .expect("the configuration resolves");
        assert_eq!(record.status, proto::admin::EpochState::Serving);
        assert_eq!(
            record.generation, None,
            "the model generation is not the launch generation the engine reported"
        );
        assert_eq!(
            record.config,
            options.engine_config().expect("the configuration resolves")
        );
        assert_eq!(process.generation, 12);
        assert_eq!(process.interface, Some(14));
        assert_eq!(process.pid, Some(4321));
        assert_eq!(process.epoch, Some(7));
        assert_eq!(process.state, proto::admin::ProcessState::Serving);
    }

    /// A host's Storage Server of versions no later than `spoken`, which this build does not
    /// speak, as a successor daemon reaches it at its socket: it refuses every hello with the
    /// typed version refusal, naming both windows.
    fn storage_server_of_version(spoken: u32) -> (tempfile::TempDir, std::path::PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("storage.sock");
        let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else {
                    return;
                };
                let mut connection = Connection::new(stream).unwrap();
                if let Ok(Some(ClientMessage::Hello(hello))) =
                    connection.recv::<ClientMessage>(std::time::Duration::from_secs(30))
                {
                    let refused = ServerMessage::Welcome(Welcome::Refused {
                        refusal: Refusal::Version(VersionRefusal {
                            offered: hello.oldest,
                            spoken,
                            offered_newest: hello.newest,
                            spoken_oldest: Some(spoken),
                        }),
                    });
                    let _ = connection.deliver(&refused);
                }
            }
        });
        (directory, socket)
    }

    /// A successor daemon does not take a deployment over when the host's Storage Server
    /// speaks no protocol version of its build's window: the claims cross on that server's
    /// connections. A host with no server listening has none to speak to.
    #[test]
    fn a_successor_needs_the_storage_server_to_speak_its_protocol() {
        let (_directory, socket) = storage_server_of_version(VERSIONS.oldest() - 1);
        let error = storage_server_speaks(&socket).unwrap_err();
        assert!(
            matches!(
                error.downcast_ref::<storage::Error>(),
                Some(storage::Error::Version(refusal))
                    if refusal.offered == VERSIONS.oldest()
                        && refusal.spoken == VERSIONS.oldest() - 1
            ),
            "{error:#}"
        );

        let absent = tempfile::tempdir().unwrap();
        storage_server_speaks(&absent.path().join("storage.sock")).unwrap();
    }

    /// The options a predecessor of the window `versions` sends, whose connections agreed on
    /// `spoken` and whose claim on `storage`.
    fn prepare(versions: service::Versions, spoken: &[u32], storage: Option<u32>) -> Prepare {
        Prepare {
            config: String::new(),
            stated: Vec::new(),
            api_key: None,
            processes: String::new(),
            instance: std::path::PathBuf::new(),
            deployment: "model".to_owned(),
            rebind: false,
            versions,
            spoken: spoken.to_vec(),
            storage,
            processing: proto::module::PackageRevision {
                backend: "uniserve".to_owned(),
                revision: 1,
                arguments: Default::default(),
            },
            replicas: Vec::new(),
        }
    }

    /// A successor of another build takes the deployment over when its windows hold every
    /// version the predecessor's connections agreed on, and writes the handoff in the newest
    /// version both daemons speak; it gives up, naming the version, when a connection's version
    /// or the claim's lies outside its windows, or when the daemons' windows share none.
    #[test]
    fn a_successor_takes_over_the_versions_its_windows_hold() {
        let engine = service::Versions::new(14, 15);
        let storage = service::Versions::one(33);

        let earlier = prepare(service::Versions::one(14), &[14], Some(33));
        assert_eq!(takes_over(&earlier, engine, storage).unwrap(), 14);
        let later = prepare(service::Versions::new(15, 16), &[15], Some(33));
        assert_eq!(takes_over(&later, engine, storage).unwrap(), 15);

        let error = takes_over(
            &prepare(service::Versions::new(14, 16), &[14, 16], Some(33)),
            engine,
            storage,
        )
        .unwrap_err();
        assert!(error.to_string().contains("version 16"), "{error:#}");

        let error = takes_over(
            &prepare(service::Versions::one(14), &[14], Some(32)),
            engine,
            storage,
        )
        .unwrap_err();
        assert!(error.to_string().contains("version 32"), "{error:#}");

        let error = takes_over(
            &prepare(service::Versions::new(16, 17), &[16], None),
            engine,
            storage,
        )
        .unwrap_err();
        assert!(error.to_string().contains("share none"), "{error:#}");
    }

    /// The options a predecessor sends cross the handoff socket's encoding whole, with the
    /// versions a successor checks.
    #[test]
    fn the_options_cross_the_handoff_with_their_versions() {
        let sent = Message::Prepare(Box::new(prepare(
            service::Versions::new(14, 15),
            &[14],
            Some(33),
        )));
        let encoded = service::Message::encode(&sent).unwrap();
        let Message::Prepare(received) = <Message as service::Message>::decode(&encoded).unwrap()
        else {
            panic!("the options decode as another message");
        };
        assert_eq!(received.versions, service::Versions::new(14, 15));
        assert_eq!(received.spoken, [14]);
        assert_eq!(received.storage, Some(33));
    }

    /// The hot restart of a predecessor of the engine interface's oldest version
    /// ([`oldest`](crate::test_support::oldest)): this build reads its options and its takeover
    /// and resolves every configuration they carry, as a successor does, and writes each message
    /// back in no field a successor of that build does not read.
    #[test]
    fn the_handoff_of_the_oldest_version_reads_both_ways() {
        use crate::test_support::oldest;

        for bytes in [oldest::PREPARE, oldest::TAKEOVER] {
            let message = <Message as service::Message>::decode(bytes)
                .expect("this build reads the handoff of the oldest version");
            match &message {
                Message::Prepare(prepare) => {
                    crate::config::parse(&prepare.config).expect("the options' configuration");
                }
                Message::Takeover(takeover) => {
                    assert!(
                        !takeover.replicas.is_empty(),
                        "the takeover hands a replica over"
                    );
                    for replica in &takeover.replicas {
                        let options =
                            replica_options(&replica.config, &crate::config::Config::default())
                                .expect("the replica's configuration");
                        crate::epoch::Epoch::of(options).expect("the replica's configuration");
                    }
                }
                other => panic!("the fixture holds {other:?}"),
            }

            let written: serde_json::Value =
                service::msgpack::decode(&service::Message::encode(&message).unwrap()).unwrap();
            let unread = oldest::unread(&written, &service::msgpack::decode(bytes).unwrap());
            assert!(
                unread.is_empty(),
                "a build of the oldest version does not read {unread:?}"
            );
        }
    }

    /// The checkpoint directory every case below states.
    const MODEL_DIR: &str = "/models/checkpoint";

    /// `printed`, a configuration of the handoff, as the oldest version's leader prints it for a
    /// deployment whose package states `model_dir`: that leader names the replica's model in
    /// `model`, which is then the directory again.
    fn with_directory(printed: &str, model: &str) -> String {
        let (mut options, _) = crate::config::parse(printed).expect("the fixture's configuration");
        options.model_dir = Some(MODEL_DIR.into());
        options.model = Some(model.to_owned());
        crate::config::printed(&options).expect("the configuration prints")
    }

    /// A successor reads the hot restart of a predecessor of the oldest version that serves a
    /// checkpoint directory: in the options (`Prepare`) and in every replica's record of the
    /// takeover, the `model` the predecessor's leader wrote beside `model_dir` is the directory,
    /// and the configuration reads as the directory stated once, which validates as a start of
    /// the package does. A `model` that names another checkpoint beside the directory is the
    /// package's own statement of two checkpoints, and is refused as a start refuses it.
    #[test]
    fn the_handoff_of_a_checkpoint_directory_states_it_once() {
        use crate::test_support::oldest;

        let checks = |printed: &str, read: &dyn Fn(&str) -> crate::config::Config| {
            let options = read(&with_directory(printed, MODEL_DIR));
            assert_eq!(options.model, None);
            assert_eq!(
                options.model_dir.as_deref(),
                Some(std::path::Path::new(MODEL_DIR))
            );
            assert_eq!(options.model(), MODEL_DIR);
            options
                .validate(&Stated::from_paths(vec!["model_dir".to_owned()]))
                .expect("the options validate as a start of the package");
            crate::epoch::Epoch::of(options).expect("the configuration resolves");

            let other = read(&with_directory(printed, "org/other-model"));
            let refused = other
                .validate(&Stated::from_paths(vec![
                    "model".to_owned(),
                    "model_dir".to_owned(),
                ]))
                .expect_err("two checkpoints are refused");
            assert!(
                refused.to_string().contains("both state one checkpoint"),
                "{refused:#}"
            );
        };

        let Message::Prepare(prepare) = <Message as service::Message>::decode(oldest::PREPARE)
            .expect("this build reads the options of the oldest version")
        else {
            panic!("the fixture holds no options");
        };
        checks(&prepare.config, &|printed| {
            handed_over(printed).expect("the options' configuration")
        });

        let Message::Takeover(takeover) = <Message as service::Message>::decode(oldest::TAKEOVER)
            .expect("this build reads the takeover of the oldest version")
        else {
            panic!("the fixture holds no takeover");
        };
        assert!(
            !takeover.replicas.is_empty(),
            "the takeover hands a replica over"
        );
        for replica in &takeover.replicas {
            checks(&replica.config, &|printed| {
                replica_options(printed, &crate::config::Config::default())
                    .expect("the replica's configuration")
            });
        }
    }

    /// What a leader of this build hands over for a deployment whose package states a
    /// checkpoint directory: each replica's configuration, as the takeover's record prints it,
    /// states the directory alone, so a successor of any build validates it as a start of the
    /// package.
    #[tokio::test]
    async fn a_leader_hands_a_checkpoint_directory_over_as_the_package_states_it() {
        use crate::test_support::{
            configured_tokenizer, fake_client, leader_with, test_model, text_fake,
        };

        let options = crate::config::Config {
            model_dir: Some(MODEL_DIR.into()),
            ..crate::config::Config::default()
        };
        let client = Arc::new(fake_client(
            testkit::args(crate::test_support::SERVED_MODEL),
            testkit::Fixture::new(text_fake),
        ));
        let leader = leader_with(
            options,
            test_model(crate::test_support::CHAT),
            configured_tokenizer(),
            vec![client],
        );
        for replica in leader.replicas() {
            let printed =
                crate::config::printed(&service::lock(&replica.config)).expect("the record prints");
            let (read, stated) = crate::config::parse(&printed).expect("the record reads");
            assert_eq!(read.model, None, "{printed}");
            assert_eq!(
                read.model_dir.as_deref(),
                Some(std::path::Path::new(MODEL_DIR))
            );
            read.validate(&stated)
                .expect("the record validates as a start of the package");
        }
    }

    /// The payload of one frame as it is, as a predecessor of another build writes it.
    struct Frame(Vec<u8>);

    impl service::Message for Frame {
        fn encode(&self) -> std::io::Result<Vec<u8>> {
            Ok(self.0.clone())
        }

        fn decode(payload: &[u8]) -> std::io::Result<Self> {
            Ok(Self(payload.to_vec()))
        }
    }

    /// What a successor answers to options it receives as `payload` at the other end of its
    /// handoff socket, and how its takeover ended.
    async fn answer(payload: Vec<u8>) -> (Message, anyhow::Result<()>) {
        use std::os::fd::IntoRawFd as _;

        let (ours, theirs) = UnixStream::pair().unwrap();
        let mut channel = Connection::new(ours).unwrap();
        channel.send_with(&Frame(payload), &[]).unwrap();
        channel.flush().unwrap();
        let (taken, answer) = tokio::join!(
            take_over(theirs.into_raw_fd(), CancellationToken::new()),
            recv(&mut channel)
        );
        (answer.expect("the successor answers").0, taken)
    }

    /// A successor gives up, stating why, on a record of a replica it cannot read, before it
    /// answers the options and so before anything is handed over: the predecessor serves on. A
    /// record of a later build that states a field this build does not know is one, in the
    /// arguments a replica's engine serves with as in its configuration.
    #[tokio::test]
    async fn a_successor_gives_up_on_a_record_it_cannot_read_before_the_handover() {
        use crate::test_support::oldest;

        /// The options of a later build, whose replica records state a field this build does not
        /// know: what the handoff socket carries for them, as that build's types write it.
        #[derive(Serialize)]
        enum Later<'a> {
            Prepare(LaterOptions<'a>),
        }
        #[derive(Serialize)]
        struct LaterOptions<'a> {
            #[serde(flatten)]
            options: &'a Prepare,
            replicas: Vec<LaterRecord<'a>>,
        }
        #[derive(Serialize)]
        struct LaterRecord<'a> {
            served: LaterArguments<'a>,
            config: String,
        }
        #[derive(Serialize)]
        struct LaterArguments<'a> {
            #[serde(flatten)]
            served: &'a proto::config::EngineArgs,
            #[serde(skip_serializing_if = "Option::is_none")]
            later_option: Option<u32>,
        }

        let Message::Takeover(takeover) =
            <Message as service::Message>::decode(oldest::TAKEOVER).unwrap()
        else {
            panic!("the fixture holds a takeover");
        };
        let Message::Prepare(mut options) =
            <Message as service::Message>::decode(oldest::PREPARE).unwrap()
        else {
            panic!("the fixture holds options");
        };
        // A deployment no Storage Server of this host serves.
        options.deployment = "restart-records".to_owned();
        let replica = &takeover.replicas[0];
        let payload = |later_option: Option<u32>, config: String| {
            service::msgpack::encode(&Later::Prepare(LaterOptions {
                options: &options,
                replicas: vec![LaterRecord {
                    served: LaterArguments {
                        served: &replica.served,
                        later_option,
                    },
                    config,
                }],
            }))
            .unwrap()
        };
        let served = payload(Some(1), replica.config.clone());
        let configured = payload(None, format!("later_option = 1\n{}", replica.config));

        for (later, field) in [(served, "later_option"), (configured, "replica 0")] {
            let (answer, taken) = answer(later).await;
            let Message::Failed { reason } = answer else {
                panic!("the successor answered {answer:?}");
            };
            assert!(
                reason.contains(field) && reason.contains("later_option"),
                "{reason}"
            );
            assert!(taken.is_err());
        }
    }

    /// A successor that gives up before the handover leaves the deployment's Build Server as
    /// it was, although its options state other Build Server settings (another source tree and
    /// other build limits, as a change of `build_config` does): the same process serves on the
    /// same socket with the predecessor's settings. Here the successor gives up after it
    /// reached that server, on the serving epoch's processing program, which the server does
    /// not hold.
    #[tokio::test]
    async fn a_successor_that_gives_up_leaves_the_build_server_as_it_was() {
        use crate::test_support::build_server::BuildServerStandIn;
        use crate::test_support::oldest;

        // A deployment of this process's own in the user's runtime directory, where a successor
        // finds the deployment's servers; no Storage Server serves it.
        let deployment = format!("restart-build-server-{}", std::process::id());
        let runtime = service::RuntimeDirectory::of_user().unwrap();
        let serving = build_server::command::Settings {
            max_jobs: Some(3),
            ..build_server::command::Settings::default()
        };
        let mut server = BuildServerStandIn::serve(&runtime, &deployment, serving);

        let Message::Prepare(mut options) =
            <Message as service::Message>::decode(oldest::PREPARE).unwrap()
        else {
            panic!("the fixture holds options");
        };
        let instance = tempfile::tempdir().unwrap();
        options.deployment.clone_from(&deployment);
        options.instance = instance.path().to_owned();
        options.rebind = false;
        let (mut config, _) = crate::config::parse(&options.config).unwrap();
        config.build_config = build_server::command::Settings {
            source: Some("/another/source/tree".into()),
            max_jobs: Some(24),
            ..build_server::command::Settings::default()
        };
        options.config = crate::config::printed(&config).unwrap();

        let (answer, taken) =
            answer(service::Message::encode(&Message::Prepare(options)).unwrap()).await;
        let Message::Failed { reason } = answer else {
            panic!("the successor answered {answer:?}");
        };
        assert!(reason.contains("processing program"), "{reason}");
        assert!(taken.is_err());
        assert!(
            server.serves_as_started(),
            "the successor that gave up changed the deployment's Build Server"
        );

        drop(server);
        let _ = std::fs::remove_dir_all(runtime.deployment(&deployment).unwrap().path());
    }
}
