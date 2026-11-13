//! The peer channel: a worker group's jobs that a scheduler node's engine process hands to the
//! other engine processes of its epoch, and what they return.
//!
//! An epoch whose worker group has ranks on several hosts runs one engine process per host. The
//! scheduler node's process holds the schedulers; every other process (a peer) holds only ranks.
//! A peer dials the scheduler node's peer listener when it starts and registers
//! ([`ipc::PeerRegistration`]); the scheduler node accepts every peer its launch counts before
//! its ranks load ([`accept`]), so a peer that never arrives fails the start by name.
//!
//! From then on each connection carries the jobs of the groups whose ranks the peer holds, as
//! [`Executors`] hands them to every rank: [`ipc::MainProcessRequest::Submit`] and `Shadow` with
//! a batch and the number the scheduler node gave it, `Map` with a completion of the buffer pool
//! for the ranks' device block tables, and `Stamp` with requests to apply and stamp. The peer
//! hands each to its own ranks of the group and responds with
//! [`ipc::MainProcessResponse::Result`] (the batch's output and the requests its ranks returned,
//! each stamped with the largest stamp among them) or with `Error`, under the request's
//! `message_id`. Every `Result` states the last number the peer's ranks of the group have ended,
//! which the scheduler node takes into the group's [`Sequences`] before it completes the
//! submission, so the group's `completed` is the smallest over all of its ranks. The scheduler
//! node asks [`ipc::MainProcessRequest::Info`] once the ranks loaded, and the peer responds with
//! its first rank's report, which the scheduler sizes a worker group that starts on the peer's
//! host by ([`Remote::info`]).
//!
//! Each end's connection is owned by one link thread, which sends what other threads queued
//! (they fire the channel's wake), reads what arrived and settles it; nothing else touches the
//! socket. A connection's close is the only death signal of the process at its other end: the
//! link marks the process's [`Liveness`] lost and wakes its control thread, which ends the
//! process, and the daemon then replaces the epoch's failure domain.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use ipc::{
    Channel, Listener, MainProcessRequest, MainProcessResponse, MainProcessResponseError,
    PeerAccepted, PeerRegistration,
};
use storage::runtime::{Completion, GroupId, Request};

use super::{Executors, Sequences};
use crate::completion::{Fault, Outcome, Submission, SubmissionId, SubmissionState};
use crate::{Error, Result, lock};

/// The longest a link thread waits for its connection before it looks at its queue unasked.
/// Every queue fires the channel's wake, so this bounds only a wake that was lost.
const LINK_WAIT: Duration = Duration::from_millis(500);

/// How long a registration may take to arrive once a peer connected: its first frame.
const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(10);

/// How long an engine process that lost a peer link has to end before it is ended, as the
/// engine gives it to [`Liveness::new`].
pub const LOST_EXIT: Duration = Duration::from_secs(20);

/// Whether the processes at the far ends of an engine process's peer links still serve, and the
/// first reason one stopped.
///
/// A link that closes sets it and wakes the control thread, which reads it on its next pass and
/// ends the process: a group missing a rank cannot run a batch, and the daemon replaces the
/// whole failure domain, not one process of it.
pub struct Liveness {
    lost: AtomicBool,
    reason: Mutex<Option<String>>,
    wake: ipc::Wake,
    /// How long the process has to end once a link is lost.
    exit_after: Duration,
}

impl Liveness {
    /// Liveness that wakes the control thread through `wake` when a link is lost, and ends the
    /// process `exit_after` later if the control thread has not ([`LOST_EXIT`]).
    pub fn new(wake: ipc::Wake, exit_after: Duration) -> Arc<Self> {
        Arc::new(Self {
            lost: AtomicBool::new(false),
            reason: Mutex::new(None),
            wake,
            exit_after,
        })
    }

    /// The first reason a link was lost, once one was.
    pub fn lost(&self) -> Option<String> {
        if !self.lost.load(Ordering::Acquire) {
            return None;
        }
        lock(&self.reason).clone()
    }

    /// Records that a link was lost for `reason`, keeping the first reason, and wakes the
    /// control thread.
    fn lose(&self, reason: String) {
        let first = {
            let mut slot = lock(&self.reason);
            let first = slot.is_none();
            if first {
                tracing::error!(%reason, "a peer link of this engine process was lost");
                *slot = Some(reason);
            }
            first
        };
        self.lost.store(true, Ordering::Release);
        self.wake.wake();

        // The control thread ends the process on its next pass. A pass that never comes (a call
        // of the control thread that waits on a device whose collective waits for the lost
        // process's ranks) does not keep the process: it ends at the deadline all the same,
        // without its orderly close, which a hung collective could not finish either. Its
        // daemon removes the driver and NCCL files it leaves when it reaps it.
        if first {
            let exit_after = self.exit_after;
            let spawned = std::thread::Builder::new()
                .name("peer-lost".into())
                .spawn(move || {
                    std::thread::sleep(exit_after);
                    tracing::error!(
                        "the control thread did not end the process after a peer was lost"
                    );

                    // SAFETY: `_exit` ends the process at once; nothing of this process runs
                    // after it.
                    unsafe { libc::_exit(1) };
                });
            if let Err(error) = spawned {
                tracing::warn!(%error, "the lost peer's exit deadline did not start");
            }
        }
    }
}

/// Accepts the registrations of `expected` peers on `listener`, each of the epoch whose
/// scheduler node runs under engine number `engine`, and returns each channel with the first
/// rank its process serves, in rank order.
///
/// # Errors
///
/// When the peers have not all registered within `timeout`, when a registration names another
/// epoch or another build (the two ends would decode each other's frames into other meanings),
/// or when two peers register for one rank.
pub fn accept(
    listener: &Listener,
    expected: usize,
    engine: u64,
    timeout: Duration,
) -> Result<Vec<(u32, Channel)>> {
    let deadline = Instant::now() + timeout;
    let mut accepted: Vec<(u32, Channel)> = Vec::with_capacity(expected);

    while accepted.len() < expected {
        let Some(mut channel) = listener
            .try_accept()
            .map_err(|error| Error::transport_source("accepting a peer engine process", error))?
        else {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(Error::transport(format!(
                    "{} of the epoch's {expected} peer engine processes registered within {} s",
                    accepted.len(),
                    timeout.as_secs()
                )));
            }

            // The listener's sockets become readable when a peer connects.
            let mut sockets = listener.poll_fds();
            let millis = i32::try_from(left.as_millis().max(1)).unwrap_or(i32::MAX);
            // SAFETY: the call borrows this live array for its own duration only.
            unsafe { libc::poll(sockets.as_mut_ptr(), sockets.len() as libc::nfds_t, millis) };
            continue;
        };

        channel
            .wait(REGISTRATION_TIMEOUT)
            .map_err(|error| Error::transport_source("waiting for a peer's registration", error))?;
        let registration: PeerRegistration = channel
            .try_recv()
            .map_err(|error| Error::transport_source("reading a peer's registration", error))?
            .ok_or_else(|| Error::transport("a peer connected without a registration"))?;

        if !registration.accepted_by(engine) {
            return Err(Error::Invalid(format!(
                "rank {} registered for engine {} of build {} with engine {engine} of build {}",
                registration.rank,
                registration.engine,
                registration.version,
                ipc::VERSION
            )));
        }
        if accepted.iter().any(|(rank, _)| *rank == registration.rank) {
            return Err(Error::Invalid(format!(
                "two peer engine processes registered for rank {}",
                registration.rank
            )));
        }

        let rank = registration.rank;
        channel
            .send(&PeerAccepted { rank })
            .map_err(|error| Error::transport_source("accepting a peer's registration", error))?;
        tracing::info!(rank, "a peer engine process registered");
        accepted.push((rank, channel));
    }

    accepted.sort_by_key(|(rank, _)| *rank);
    Ok(accepted)
}

/// Dials the scheduler node's peer listener at `address` and registers this process as the one
/// whose ranks start at `rank`, in the epoch whose scheduler node runs under engine `engine`.
///
/// The scheduler node may still be starting and has no socket to wait on until it listens, so
/// the dial is repeated until `timeout`.
///
/// # Errors
///
/// When no connection is made in time, or the scheduler node does not accept the registration.
pub fn register(address: &str, rank: u32, engine: u64, timeout: Duration) -> Result<Channel> {
    let deadline = Instant::now() + timeout;
    let mut channel = loop {
        match Channel::connect(address) {
            Ok(channel) => break channel,
            Err(error) if Instant::now() >= deadline => {
                return Err(Error::transport_source(
                    format!("dialling the scheduler node's peer listener at {address}"),
                    error,
                ));
            }
            Err(_) => std::thread::sleep(Duration::from_millis(100)),
        }
    };

    channel
        .send(&PeerRegistration {
            rank,
            engine,
            version: ipc::VERSION.to_owned(),
        })
        .map_err(|error| Error::transport_source("sending this process's registration", error))?;

    loop {
        if let Some(accepted) = channel.try_recv::<PeerAccepted>().map_err(|error| {
            Error::transport_source("reading the registration's response", error)
        })? {
            if accepted.rank != rank {
                return Err(Error::Invalid(format!(
                    "the scheduler node accepted rank {} for this process's rank {rank}",
                    accepted.rank
                )));
            }
            return Ok(channel);
        }

        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(Error::transport(format!(
                "the scheduler node at {address} did not accept this process's registration"
            )));
        }
        channel.wait(left).map_err(|error| {
            Error::transport_source("awaiting the registration's response", error)
        })?;
    }
}

/// The submissions a peer has not completed yet, by the `message_id` they were sent under: each
/// with the sequences of its group and the rank the peer reports there as.
type Pending = Mutex<HashMap<u64, (Arc<SubmissionState>, Arc<Sequences>, usize)>>;

/// The `Info` requests a peer has not responded to yet, by their `message_id`: each takes the
/// peer's report, or its refusal.
type Infos = Mutex<HashMap<u64, Sender<std::result::Result<ipc::MainProcessInfo, String>>>>;

/// The scheduler node's link to one peer engine process: where a group's jobs for the peer's
/// ranks go, and how what they return comes back.
pub struct Remote {
    /// The first rank the peer's process serves, which its registration stated.
    rank: u32,
    outbox: Sender<MainProcessRequest>,
    wake: ipc::Wake,
    pending: Arc<Pending>,
    infos: Arc<Infos>,
    /// The `message_id` of the next request.
    next: AtomicU64,
    liveness: Arc<Liveness>,
    closing: Arc<AtomicBool>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Remote {
    /// Starts the link thread over `channel`, the registered connection of the peer whose ranks
    /// start at `rank`; a close of the connection marks `liveness` lost.
    ///
    /// # Errors
    ///
    /// When the thread cannot start.
    pub fn spawn(rank: u32, channel: Channel, liveness: Arc<Liveness>) -> Result<Arc<Self>> {
        let (outbox, queued) = crossbeam_channel::unbounded();
        let pending: Arc<Pending> = Arc::default();
        let infos: Arc<Infos> = Arc::default();
        let closing = Arc::new(AtomicBool::new(false));
        let wake = channel.wake();

        let thread = {
            let pending = Arc::clone(&pending);
            let infos = Arc::clone(&infos);
            let liveness = Arc::clone(&liveness);
            let closing = Arc::clone(&closing);

            std::thread::Builder::new()
                .name(format!("peer-{rank}"))
                .spawn(move || {
                    remote_link(
                        rank, channel, &queued, &pending, &infos, &liveness, &closing,
                    );
                })
                .map_err(|error| Error::transport_source("starting a peer link", error))?
        };

        Ok(Arc::new(Self {
            rank,
            outbox,
            wake,
            pending,
            infos,
            next: AtomicU64::new(1),
            liveness,
            closing,
            thread: Mutex::new(Some(thread)),
        }))
    }

    /// The first rank the peer's process serves.
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// What the peer's first rank reports of itself, which the scheduler sizes a worker group
    /// by: the peer responds once its ranks loaded, within `timeout`.
    ///
    /// # Errors
    ///
    /// Once the link is lost, when the peer refuses, or when no response arrives in time.
    pub fn info(&self, timeout: Duration) -> Result<ipc::MainProcessInfo> {
        self.live()?;

        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (response, responded) = crossbeam_channel::bounded(1);
        lock(&self.infos).insert(id, response);
        self.outbox
            .send(MainProcessRequest::Info {
                message_id: Some(id),
            })
            .map_err(|_| Error::State("the peer link's thread has ended"))?;
        self.wake.wake();

        match responded.recv_timeout(timeout) {
            Ok(Ok(info)) => Ok(info),
            Ok(Err(reason)) => Err(Error::transport(format!(
                "rank {}'s engine process did not report itself: {reason}",
                self.rank
            ))),
            Err(_) => {
                lock(&self.infos).remove(&id);
                Err(Error::transport(format!(
                    "rank {}'s engine process did not report itself within {} s",
                    self.rank,
                    timeout.as_secs()
                )))
            }
        }
    }

    /// Refuses once the link is lost.
    fn live(&self) -> Result<()> {
        match self.liveness.lost() {
            Some(reason) => Err(Error::transport(format!(
                "the peer link of rank {} is lost: {reason}",
                self.rank
            ))),
            None => Ok(()),
        }
    }

    /// Sends `request`, a job of the group whose sequences are `sequences`, to the peer's ranks,
    /// which report there as rank `rank`; the submission completes with what they return, once
    /// their ended sequence is stated.
    ///
    /// # Errors
    ///
    /// Once the link is lost: the group cannot run its jobs without the peer's ranks.
    pub(super) fn send(
        &self,
        mut request: MainProcessRequest,
        sequences: &Arc<Sequences>,
        rank: usize,
    ) -> Result<Submission> {
        self.live()?;

        let id = self.next.fetch_add(1, Ordering::Relaxed);
        request.set_message_id(Some(id));
        let (submission, state) = Submission::pending(SubmissionId(id));
        lock(&self.pending).insert(id, (state, Arc::clone(sequences), rank));

        self.outbox
            .send(request)
            .map_err(|_| Error::State("the peer link's thread has ended"))?;
        self.wake.wake();
        Ok(submission)
    }

    /// Closes the link: the peer reads the close and ends, as the scheduler node does.
    pub fn close(&self) {
        self.closing.store(true, Ordering::Release);
        let _ = self.outbox.send(MainProcessRequest::close());
        self.wake.wake();

        if let Some(thread) = lock(&self.thread).take()
            && thread.join().is_err()
        {
            tracing::warn!(rank = self.rank, "a peer link's thread panicked");
        }
    }
}

impl Drop for Remote {
    fn drop(&mut self) {
        self.close();
    }
}

/// The scheduler node's side of the connection to the peer whose ranks start at `rank`: sends
/// what the scheduler queued, settles each response by its `message_id`, and on a close fails
/// every submission and report request the peer left open.
fn remote_link(
    rank: u32,
    mut channel: Channel,
    queued: &Receiver<MainProcessRequest>,
    pending: &Pending,
    infos: &Infos,
    liveness: &Liveness,
    closing: &AtomicBool,
) {
    let failure = loop {
        let mut failed = None;

        while let Ok(request) = queued.try_recv() {
            let close = matches!(request, MainProcessRequest::Close { .. });
            if let Err(error) = channel.send(&request) {
                failed = Some(format!("sending to rank {rank}'s engine process: {error}"));
                break;
            }
            if close {
                // An orderly close: the frame goes out before the connection does.
                let _ = channel.flush();
                channel.shutdown();
                fail_open(
                    pending,
                    infos,
                    "the epoch's scheduler node closed its peer links",
                );
                return;
            }
        }
        if let Some(failed) = failed {
            break failed;
        }

        loop {
            match channel.try_recv::<MainProcessResponse>() {
                Ok(Some(response)) => settle(rank, pending, infos, response),
                Ok(None) => break,
                Err(error) => {
                    failed = Some(format!("rank {rank}'s engine process: {error}"));
                    break;
                }
            }
        }
        if let Some(failed) = failed {
            break failed;
        }

        if let Err(error) = channel.wait(LINK_WAIT) {
            break format!("rank {rank}'s engine process: {error}");
        }
    };

    fail_open(pending, infos, &failure);

    // A close this side asked for is not a loss of the peer.
    if !closing.load(Ordering::Acquire) {
        liveness.lose(failure);
    }
}

/// Settles the submission or report request `response` belongs to.
fn settle(rank: u32, pending: &Pending, infos: &Infos, response: MainProcessResponse) {
    // A report request's response, or its refusal, settles that request.
    let asked = match &response {
        MainProcessResponse::Info { message_id, .. }
        | MainProcessResponse::Error { message_id, .. } => {
            message_id.and_then(|id| lock(infos).remove(&id))
        }
        _ => None,
    };
    if let Some(asked) = asked {
        let _ = asked.send(match response {
            MainProcessResponse::Info { info, .. } => Ok(info),
            MainProcessResponse::Error { error, .. } => Err(error.message),
            _ => Err("the peer responded to a report request with another kind".to_owned()),
        });
        return;
    }

    let (id, status, ended) = match response {
        MainProcessResponse::Result {
            message_id,
            result,
            requests,
            ended,
        } => match ipc::decode_requests(&requests) {
            Ok(requests) => (
                message_id,
                Outcome::Success((result, requests)),
                Some(ended),
            ),
            Err(error) => (
                message_id,
                Outcome::Failed(fault(format!(
                    "rank {rank}'s engine process returned a request that is not one: {error}"
                ))),
                None,
            ),
        },
        MainProcessResponse::Error { message_id, error } => (
            message_id,
            Outcome::Failed(fault(format!(
                "rank {rank}'s engine process failed the job: {}",
                error.message
            ))),
            None,
        ),
        MainProcessResponse::Stalled { stall } => {
            // The peer's own watchdog fails its rank on a stall; its close follows.
            tracing::warn!(
                rank,
                batch = ?stall.batch_id,
                running_ms = stall.running_ms,
                "a peer engine process reports a stalled job"
            );
            return;
        }
        other => {
            tracing::warn!(rank, kind = ?other.kind(), "a peer sent what nothing asked for");
            return;
        }
    };

    let state = id.and_then(|id| lock(pending).remove(&id));
    match state {
        Some((state, sequences, reporting)) => {
            // The group's word covers what the peer's ranks ended before anyone reads the job's
            // output.
            if let Some(ended) = ended {
                sequences.ended(reporting, ended);
            }
            state.finish(status);
        }
        None => tracing::warn!(
            rank,
            ?id,
            "a peer completed a submission this link did not send"
        ),
    }
}

/// Fails every submission and report request the peer left open with `reason`.
fn fail_open(pending: &Pending, infos: &Infos, reason: &str) {
    for (_, (state, ..)) in lock(pending).drain() {
        state.finish(Outcome::Failed(fault(reason.to_owned())));
    }
    for (_, asked) in lock(infos).drain() {
        let _ = asked.send(Err(reason.to_owned()));
    }
}

/// A failure of the peer channel, as a failed submission carries it.
fn fault(reason: String) -> Arc<Fault> {
    Arc::new(Error::transport(reason))
}

/// A peer's side of its connection to the scheduler node: hands every job the scheduler node
/// sends to this process's ranks, and responds to each once its ranks completed it.
pub struct Upstream {
    closing: Arc<AtomicBool>,
    wake: ipc::Wake,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl Upstream {
    /// Serves the scheduler node's requests arriving on `channel` with `executors`, this
    /// process's ranks, responding to a report request with `info`, its first rank's report; a
    /// close of the connection marks `liveness` lost.
    ///
    /// Two threads: the link, which owns the connection, and the responder, which waits for each
    /// submission in the order it was made (each executor takes its jobs in that order) and gives
    /// its response to the link, with the group's ended sequence as this process knows it then.
    ///
    /// # Errors
    ///
    /// When a thread cannot start.
    pub fn serve(
        channel: Channel,
        executors: Arc<Executors>,
        info: ipc::MainProcessInfo,
        liveness: Arc<Liveness>,
    ) -> Result<Self> {
        let (responses, responded) = crossbeam_channel::unbounded::<MainProcessResponse>();
        let (waiting, awaited) = crossbeam_channel::unbounded::<(u64, GroupId, Submission)>();
        let closing = Arc::new(AtomicBool::new(false));
        let wake = channel.wake();

        let responder = {
            let wake = wake.clone();
            let executors = Arc::clone(&executors);

            std::thread::Builder::new()
                .name("peer-responder".into())
                .spawn(move || {
                    for (id, group, submission) in awaited {
                        // Each rank stated its ended sequence before its part completed.
                        let response = match (submission.wait(), executors.completed(group)) {
                            (Outcome::Success((output, requests)), Ok(ended)) => {
                                let mut response =
                                    MainProcessResponse::result(output, &requests, ended);
                                response.set_message_id(Some(id));
                                response
                            }
                            (Outcome::Success(_), Err(error)) => refusal(id, error.to_string()),
                            (Outcome::Failed(error), _) => refusal(id, error.to_string()),
                            (Outcome::Cancelled, _) => refusal(id, "the job was cancelled".into()),
                        };
                        if responses.send(response).is_err() {
                            return;
                        }
                        wake.wake();
                    }
                })
                .map_err(|error| Error::transport_source("starting the peer responder", error))?
        };

        let link = {
            let closing = Arc::clone(&closing);

            std::thread::Builder::new()
                .name("peer-upstream".into())
                .spawn(move || {
                    upstream_link(
                        channel,
                        executors.as_ref(),
                        &info,
                        &waiting,
                        &responded,
                        &liveness,
                        &closing,
                    );
                })
                .map_err(|error| Error::transport_source("starting the peer link", error))?
        };

        Ok(Self {
            closing,
            wake,
            threads: Mutex::new(vec![link, responder]),
        })
    }

    /// Ends the link: the scheduler node reads the close.
    pub fn close(&self) {
        self.closing.store(true, Ordering::Release);
        self.wake.wake();

        for thread in std::mem::take(&mut *lock(&self.threads)) {
            if thread.join().is_err() {
                tracing::warn!("a peer link's thread panicked");
            }
        }
    }
}

impl Drop for Upstream {
    fn drop(&mut self) {
        self.close();
    }
}

/// A peer's side of its connection: hands each job the scheduler node sends to this process's
/// ranks, and sends each response the responder queued.
fn upstream_link(
    mut channel: Channel,
    executors: &Executors,
    info: &ipc::MainProcessInfo,
    waiting: &Sender<(u64, GroupId, Submission)>,
    responded: &Receiver<MainProcessResponse>,
    liveness: &Liveness,
    closing: &AtomicBool,
) {
    let failure = loop {
        if closing.load(Ordering::Acquire) {
            let _ = channel.flush();
            channel.shutdown();
            return;
        }

        let mut failed = None;
        while let Ok(response) = responded.try_recv() {
            if let Err(error) = channel.send(&response) {
                failed = Some(format!("responding to the scheduler node: {error}"));
                break;
            }
        }
        if let Some(failed) = failed {
            break failed;
        }

        loop {
            let request = match channel.try_recv::<MainProcessRequest>() {
                Ok(Some(request)) => request,
                Ok(None) => break,
                Err(error) => {
                    failed = Some(format!("the scheduler node's engine process: {error}"));
                    break;
                }
            };

            let (id, group, submitted) = match request {
                MainProcessRequest::Submit {
                    message_id,
                    sequence,
                    pass,
                    batch,
                } => (
                    message_id.unwrap_or(0),
                    batch.group,
                    executors.run(sequence, *batch, pass, false),
                ),
                MainProcessRequest::Shadow {
                    message_id,
                    sequence,
                    batch,
                } => (
                    message_id.unwrap_or(0),
                    batch.group,
                    executors.run(sequence, *batch, false, true),
                ),
                MainProcessRequest::Map {
                    message_id,
                    group,
                    slot,
                    request,
                    completion,
                } => (
                    message_id.unwrap_or(0),
                    group,
                    mapped(executors, group, slot, &request, &completion),
                ),
                MainProcessRequest::Stamp {
                    message_id,
                    group,
                    requests,
                } => (
                    message_id.unwrap_or(0),
                    group,
                    ipc::decode_requests(&requests)
                        .map_err(|error| {
                            Error::transport_source("reading the requests of a stamp", error)
                        })
                        .and_then(|requests| executors.stamp(group, requests)),
                ),
                MainProcessRequest::Close { .. } => {
                    failed = Some("the scheduler node closed the epoch's peer links".into());
                    break;
                }
                // The scheduler node sizes a worker group whose first rank is this process's by
                // its report.
                MainProcessRequest::Info { message_id } => {
                    let response = MainProcessResponse::Info {
                        message_id,
                        info: info.clone(),
                    };
                    if let Err(error) = channel.send(&response) {
                        failed = Some(format!("responding to the scheduler node: {error}"));
                        break;
                    }
                    continue;
                }
                other => {
                    // The daemon commands this process over its own process interface (its
                    // modules, a promotion, a shutdown); the peer channel carries a group's jobs.
                    let id = match &other {
                        MainProcessRequest::Promote { message_id }
                        | MainProcessRequest::Hook { message_id, .. } => message_id.unwrap_or(0),
                        _ => 0,
                    };
                    let refused = refusal(
                        id,
                        format!(
                            "the peer channel serves a group's jobs, not {:?}",
                            other.kind()
                        ),
                    );
                    if let Err(error) = channel.send(&refused) {
                        failed = Some(format!("responding to the scheduler node: {error}"));
                        break;
                    }
                    continue;
                }
            };

            match submitted {
                Ok(submission) => {
                    if waiting.send((id, group, submission)).is_err() {
                        failed = Some("the peer responder ended".into());
                        break;
                    }
                }
                Err(error) => {
                    if let Err(error) = channel.send(&refusal(id, error.to_string())) {
                        failed = Some(format!("responding to the scheduler node: {error}"));
                        break;
                    }
                }
            }
        }
        if let Some(failed) = failed {
            break failed;
        }

        if let Err(error) = channel.wait(LINK_WAIT) {
            break format!("the scheduler node's engine process: {error}");
        }
    };

    if !closing.load(Ordering::Acquire) {
        liveness.lose(failure);
    }
}

/// Maps the completion of a `Map` frame, with the request it answers, into this process's
/// ranks of `group` under `slot`.
fn mapped(
    executors: &Executors,
    group: GroupId,
    slot: u32,
    request: &[u8],
    completion: &[u8],
) -> Result<Submission> {
    let read = |error| Error::transport_source("reading the records of a map", error);
    let (id, request) = Request::decode(request).map_err(read)?;
    let (answered, completion) = Completion::decode(completion).map_err(read)?;
    if answered != id {
        return Err(Error::transport(format!(
            "a map carries request {} with the completion of request {}",
            id.0, answered.0
        )));
    }

    executors.map(group, slot, id, request, completion)
}

/// The response to request `id` that failed for `message`.
fn refusal(id: u64, message: String) -> MainProcessResponse {
    MainProcessResponse::Error {
        message_id: Some(id),
        error: MainProcessResponseError {
            message,
            code: None,
            fatal: false,
            phase: None,
            route: None,
            calls: Vec::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bound on the tests' waits that only a defect reaches.
    const LONG: Duration = Duration::from_secs(30);

    /// Liveness of a test process, which a lost link must not end, with the waker it fires.
    fn liveness() -> (Arc<Liveness>, ipc::Waker) {
        let waker = ipc::Waker::new().expect("a waker");
        (
            Liveness::new(waker.wake(), Duration::from_secs(3600)),
            waker,
        )
    }

    /// The submission of a decode batch of one row of group 0, numbered 1, which the tests
    /// send.
    fn submit() -> MainProcessRequest {
        let batch = ipc::batch::Batch {
            id: ipc::batch::BatchId(1),
            group: GroupId(0),
            phase: ipc::batch::Phase::Decode,
            generation: 0,
            shape: ipc::batch::Shape::decode(1, ipc::batch::Variant::DecodeGreedy),
            requests: Vec::new(),
            columns: ipc::batch::Columns::default(),
            media: Vec::new(),
            new_slots: Vec::new(),
        };

        MainProcessRequest::submit(1, false, batch)
    }

    /// The next request the scheduler node sends on `channel`.
    fn next_request(channel: &mut Channel) -> MainProcessRequest {
        loop {
            if let Some(request) = channel.try_recv::<MainProcessRequest>().expect("a request") {
                return request;
            }
            channel.wait(LONG).expect("the request arrives");
        }
    }

    /// The scheduler node's link to one peer that registers as rank `rank` of engine `engine`
    /// and then runs `peer` over its channel, with the peer's thread.
    fn linked(
        rank: u32,
        engine: u64,
        liveness: Arc<Liveness>,
        peer: impl FnOnce(Channel) + Send + 'static,
    ) -> (Arc<Remote>, JoinHandle<()>) {
        let listener = Listener::bind(&["127.0.0.1:0"]).expect("a listener");
        let address = listener.addresses()[0].clone();
        let peer = std::thread::spawn(move || {
            peer(register(&address, rank, engine, LONG).expect("the peer registers"));
        });

        let (accepted, channel) = accept(&listener, 1, engine, LONG)
            .expect("the peer registers")
            .pop()
            .expect("one peer");
        assert_eq!(accepted, rank);

        let remote = Remote::spawn(rank, channel, liveness).expect("the link");
        (remote, peer)
    }

    #[test]
    fn a_peer_registers_with_the_scheduler_node_of_its_epoch() {
        let listener = Listener::bind(&["127.0.0.1:0"]).expect("a listener");

        let address = listener.addresses()[0].clone();
        let dialling = std::thread::spawn(move || register(&address, 4, 7, LONG).map(|_| ()));
        let accepted = accept(&listener, 1, 7, LONG).expect("the peer registers");
        assert_eq!(
            accepted.iter().map(|(rank, _)| *rank).collect::<Vec<_>>(),
            vec![4]
        );
        dialling
            .join()
            .expect("the dial")
            .expect("the registration is accepted");

        // A registration of another epoch is refused, and the peer learns it from the close.
        let address = listener.addresses()[0].clone();
        let stranger = std::thread::spawn(move || {
            register(&address, 4, 8, Duration::from_secs(2)).map(|_| ())
        });
        assert!(accept(&listener, 1, 7, LONG).is_err());
        assert!(stranger.join().expect("the stranger's dial").is_err());
    }

    #[test]
    fn peers_that_do_not_arrive_fail_the_start_by_count() {
        let listener = Listener::bind(&["127.0.0.1:0"]).expect("a listener");

        let refused = accept(&listener, 2, 7, Duration::from_millis(50))
            .err()
            .expect("no peer dials");
        assert!(
            refused.to_string().contains("0 of the epoch's 2"),
            "{refused}"
        );
    }

    #[test]
    fn a_peer_reports_its_first_rank_and_a_refusal_is_the_request_s_error() {
        let stated = ipc::MainProcessInfo {
            request_slots: 7,
            ..ipc::MainProcessInfo::default()
        };
        let report = stated.clone();
        let (liveness, _waker) = liveness();

        let (remote, peer) = linked(2, 5, liveness, move |mut channel| {
            // The first request gets the report and the second a refusal.
            for responded in 0..2 {
                let request = next_request(&mut channel);
                let MainProcessRequest::Info { message_id } = request else {
                    panic!("the scheduler node asked for a report: {request:?}");
                };
                let response = if responded == 0 {
                    MainProcessResponse::Info {
                        message_id,
                        info: report.clone(),
                    }
                } else {
                    refusal(message_id.unwrap_or(0), "not loaded".to_owned())
                };
                channel.send(&response).expect("the response is sent");
                channel.flush().expect("the response leaves");
            }

            // The link's close ends the peer.
            assert!(matches!(
                next_request(&mut channel),
                MainProcessRequest::Close { .. }
            ));
        });

        assert_eq!(remote.info(LONG).expect("the peer reports"), stated);
        let refused = remote
            .info(LONG)
            .expect_err("a refusal is the request's error");
        assert!(refused.to_string().contains("not loaded"), "{refused}");

        remote.close();
        peer.join().expect("the peer");
    }

    #[test]
    fn a_closed_peer_fails_its_submissions_and_is_lost() {
        let (liveness, waker) = liveness();

        let (remote, peer) = linked(1, 3, Arc::clone(&liveness), |mut channel| {
            // Read the one submission, then go away without responding.
            next_request(&mut channel);
            channel.shutdown();
        });

        let sequences = Sequences::new(GroupId(0), 1, Vec::new());
        let submission = remote
            .send(submit(), &sequences, 0)
            .expect("the batch is sent");
        assert!(matches!(submission.wait(), Outcome::Failed(_)));
        peer.join().expect("the peer");

        // The close is the peer's death signal: the link fires the control thread's wake.
        let mut wake = [waker.poll_fd()];
        // SAFETY: the call borrows this live array for its own duration only.
        let woken = unsafe { libc::poll(wake.as_mut_ptr(), 1, 30_000) };
        assert_eq!(woken, 1, "the lost link did not wake the control thread");
        assert!(liveness.lost().is_some());
        assert!(remote.send(submit(), &sequences, 0).is_err());

        // Nothing the peer ran is counted as ended.
        assert_eq!(sequences.completed(), 0);
    }
}
