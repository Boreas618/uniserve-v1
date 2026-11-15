//! The peer channel: what a scheduler node's engine process exchanges with the other engine
//! processes of its epoch.
//!
//! An epoch whose worker group has ranks on several hosts runs one engine process per host. The
//! scheduler node's process holds the schedulers; every other process (a peer) holds only ranks.
//! A peer dials the scheduler node's peer listener when it starts and registers
//! ([`PeerRegistration`]); the scheduler node accepts every peer its launch counts before
//! its ranks load ([`accept`]), so a peer that never arrives fails the start by name.
//!
//! **The start.** Once its session with its host's buffer pool is open, the scheduler node
//! relays it to every peer ([`Remote::session`]), which reads it on the registered connection
//! before any thread owns it ([`session`]); the peer's ranks open over it, and the peer answers
//! with what the scheduler node grants, places and schedules them by
//! ([`MainProcessResponse::Placed`]).
//!
//! **The jobs.** From then on each connection carries the jobs of the groups whose ranks the
//! peer holds, as [`Route`] hands them to every rank: [`MainProcessRequest::Submit`] and
//! `Shadow` with a batch and the number the scheduler node gave it, `Map` with a completion of
//! the buffer pool for the ranks' device block tables, `Stamp` with requests to apply and stamp,
//! and a generation's `Switch`, `Commit` and `Rollback`, which reach every rank of the peer in
//! the order of every group's jobs. The peer hands each to its own ranks and responds with
//! [`MainProcessResponse::Result`] (the batch's output and the requests its ranks returned, each
//! stamped with the largest stamp among them) or with `Error`, under the request's `message_id`.
//! Every response to a group's job states the last number the peer's ranks of the group have
//! ended, which the scheduler node takes into the group's [`Sequences`] before it completes the
//! submission, so the group's `completed` is the smallest over all of its ranks. A batch that
//! failed on the peer after copies started with it returns their `Finish { ok: false }`, which
//! the scheduler node's submission fails with as a local rank's would ([`Error::Copies`]).
//!
//! **The data plane.** A `Data` frame runs one operation of the data plane on the peer's ranks
//! ([`DataOperation`]); the peer runs them one at a time in the order they arrived, on a thread
//! of its own, and answers each with the JSON of its value
//! ([`MainProcessResponse::Operated`]).
//!
//! Each end's connection is owned by one link thread, which sends what other threads queued
//! (they fire the channel's wake), reads what arrived and settles it; nothing else touches the
//! socket. A connection's close is the only death signal of the process at its other end: the
//! link marks the process's [`Liveness`] lost and calls the control thread's callback, which ends
//! the process, and the daemon then replaces the epoch's failure domain. The scheduler node's
//! orderly end closes its links with a `Close` frame first, which a peer reads as the end of the
//! epoch: its control thread ends in order too.

pub mod link;
pub mod messages;
pub(crate) mod registration;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use service::Connection;

use self::link::Listener;
use self::messages::{DataOperation, MainProcessRequest, MainProcessResponse};
use self::registration::{PeerAccepted, PeerRegistration};
use storage::runtime::{Completion, GroupId, Open, Opened, Request};

use super::route::Route;
use crate::worker::Sequences;
use crate::worker::completion::{Fault, Outcome, Submission, SubmissionId, SubmissionState, Waker};
use crate::worker::{Error, Result};
use service::lock;

/// The longest a link thread waits for its connection before it looks at its queue unasked.
/// Every queue fires the channel's wake, so this bounds only a wake that was lost.
const LINK_WAIT: Duration = Duration::from_millis(500);

/// How long a registration may take to arrive once a peer connected: its first frame.
const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(10);

/// How long an engine process that lost a peer link has to end before it is ended, as the
/// engine gives it to [`Liveness::new`].
pub(crate) const LOST_EXIT: Duration = Duration::from_secs(20);

/// Whether the processes at the far ends of an engine process's peer links still serve, and the
/// first reason one stopped.
///
/// A link that closes sets it and calls the control thread's callback, which tells the control
/// thread, and the control thread ends the process: a group missing a rank cannot run a batch,
/// and the daemon replaces the whole failure domain, not one process of it. A peer's link that
/// the scheduler node closed in order ([`MainProcessRequest::Close`]) is no loss: it records the
/// epoch's end and calls the same callback, and the peer's control thread ends in order.
pub struct Liveness {
    lost: AtomicBool,
    reason: Mutex<Option<String>>,
    /// The scheduler node closed this peer's link in order.
    closed: AtomicBool,
    /// Called once, on the link thread that lost or closed its link, after the reason is
    /// recorded: it only signals the control thread.
    callback: Waker,
    /// How long the process has to end once a link is lost.
    exit_after: Duration,
}

impl Liveness {
    /// Liveness that calls `callback` when a link is first lost, and ends the process
    /// `exit_after` later if the control thread has not ([`LOST_EXIT`]).
    pub fn new(callback: Waker, exit_after: Duration) -> Arc<Self> {
        Arc::new(Self {
            lost: AtomicBool::new(false),
            reason: Mutex::new(None),
            closed: AtomicBool::new(false),
            callback,
            exit_after,
        })
    }

    /// The first reason a link was lost, once one was.
    pub(crate) fn lost(&self) -> Option<String> {
        if !self.lost.load(Ordering::Acquire) {
            return None;
        }
        lock(&self.reason).clone()
    }

    /// Whether the scheduler node closed this peer's link in order: the epoch ends, and the peer
    /// with it.
    pub(crate) fn closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Records that the scheduler node closed this peer's link in order, and calls the control
    /// thread's callback the first time.
    fn close(&self) {
        if !self.closed.swap(true, Ordering::AcqRel) {
            tracing::info!("the scheduler node closed this peer's link: the epoch ends");
            (self.callback)();
        }
    }

    /// Records that a link was lost for `reason`, keeping the first reason, and calls the
    /// control thread's callback the first time.
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
        if first {
            (self.callback)();
        }

        // The control thread ends the process on its next pass. A pass that never comes (a call
        // of the control thread that waits on a device whose collective waits for the lost
        // process's ranks) does not keep the process: it ends at the deadline all the same,
        // without its orderly close, which a hung collective could not finish either.
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
) -> Result<Vec<(u32, Connection)>> {
    let deadline = Instant::now() + timeout;
    let mut accepted: Vec<(u32, Connection)> = Vec::with_capacity(expected);

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
                registration::VERSION
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
pub fn register(address: &str, rank: u32, engine: u64, timeout: Duration) -> Result<Connection> {
    let endpoint: service::Endpoint = address.parse().map_err(|error| {
        Error::transport_source(
            format!("the scheduler node's peer listener {address}"),
            std::io::Error::from(error),
        )
    })?;
    let deadline = Instant::now() + timeout;
    let mut channel = loop {
        match Connection::connect(&endpoint) {
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
            version: registration::VERSION.to_owned(),
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

/// Reads the session the scheduler node relays to this peer ([`Remote::session`]) on `channel`,
/// its registered connection, before any thread owns it, within `timeout`: the request's
/// `message_id`, which the answer ([`MainProcessResponse::Placed`]) carries, the layout the
/// session was opened with, and the buffer pool's answer.
///
/// # Errors
///
/// When no frame arrives in time, the connection fails, or its first frame is not the session.
pub(crate) fn session(channel: &mut Connection, timeout: Duration) -> Result<(u64, Open, Opened)> {
    let deadline = Instant::now() + timeout;
    loop {
        let request = channel.try_recv::<MainProcessRequest>().map_err(|error| {
            Error::transport_source("reading the scheduler node's session", error)
        })?;
        match request {
            Some(MainProcessRequest::Session {
                message_id,
                open,
                opened,
            }) => return Ok((message_id.unwrap_or(0), *open, *opened)),
            Some(other) => {
                return Err(Error::transport(format!(
                    "the scheduler node's first request after the registration is no session: \
                     {other:?}"
                )));
            }
            None => {}
        }

        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(Error::transport(format!(
                "the scheduler node relayed no session within {} s",
                timeout.as_secs()
            )));
        }
        channel.wait(left).map_err(|error| {
            Error::transport_source("awaiting the scheduler node's session", error)
        })?;
    }
}

/// The submissions a peer has not completed yet, by the `message_id` they were sent under: each
/// with, for a job of a group, the sequences of its group and the rank the peer reports there
/// as; none for a generation's switch, commit or rollback, which addresses no group.
type Pending = Mutex<HashMap<u64, (Arc<SubmissionState>, Option<(Arc<Sequences>, usize)>)>>;

/// What a peer answers a request that is no job of a group (the relayed session, a data plane's
/// operation): its response, or why there is none.
pub(crate) type Answer = Receiver<std::result::Result<MainProcessResponse, String>>;

/// The requests that are no job of a group a peer has not answered yet, by their `message_id`.
type Asked = Mutex<HashMap<u64, Sender<std::result::Result<MainProcessResponse, String>>>>;

/// The scheduler node's link to one peer engine process: where a group's jobs for the peer's
/// ranks go, and how what they return comes back.
pub struct Remote {
    /// The first rank the peer's process serves, which its registration stated.
    rank: u32,
    outbox: Sender<MainProcessRequest>,
    wake: service::Wake,
    pending: Arc<Pending>,
    asked: Arc<Asked>,
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
    pub fn spawn(rank: u32, channel: Connection, liveness: Arc<Liveness>) -> Result<Arc<Self>> {
        let (outbox, queued) = crossbeam_channel::unbounded();
        let pending: Arc<Pending> = Arc::default();
        let asked: Arc<Asked> = Arc::default();
        let closing = Arc::new(AtomicBool::new(false));
        let wake = channel.wake();

        let thread = {
            let pending = Arc::clone(&pending);
            let asked = Arc::clone(&asked);
            let liveness = Arc::clone(&liveness);
            let closing = Arc::clone(&closing);

            std::thread::Builder::new()
                .name(format!("peer-{rank}"))
                .spawn(move || {
                    remote_link(
                        rank, channel, &queued, &pending, &asked, &liveness, &closing,
                    );
                })
                .map_err(|error| Error::transport_source("starting a peer link", error))?
        };

        Ok(Arc::new(Self {
            rank,
            outbox,
            wake,
            pending,
            asked,
            next: AtomicU64::new(1),
            liveness,
            closing,
            thread: Mutex::new(Some(thread)),
        }))
    }

    /// The first rank the peer's process serves.
    pub(crate) fn rank(&self) -> u32 {
        self.rank
    }

    /// Relays the session `open` opened and `opened` answered to the peer, whose ranks open
    /// over it ([`MainProcessRequest::Session`]). Its answer arrives once the peer's ranks are
    /// open, bound, spawned and warm: [`MainProcessResponse::Placed`], or the peer's failure.
    ///
    /// # Errors
    ///
    /// Once the link is lost.
    pub(crate) fn session(&self, open: &Open, opened: &Opened) -> Result<Answer> {
        self.ask(MainProcessRequest::session(open, opened))
    }

    /// Runs `operation` of the data plane on the peer's ranks
    /// ([`MainProcessRequest::Data`]). Its answer is [`MainProcessResponse::Operated`] with the
    /// JSON of the operation's value, or the peer's failure; see [`operated`].
    ///
    /// # Errors
    ///
    /// Once the link is lost.
    pub(crate) fn operate(&self, operation: DataOperation) -> Result<Answer> {
        self.ask(MainProcessRequest::data(operation))
    }

    /// Sends `request`, which is no job of a group, and returns where its answer arrives.
    fn ask(&self, mut request: MainProcessRequest) -> Result<Answer> {
        self.live()?;

        let id = self.next.fetch_add(1, Ordering::Relaxed);
        request.set_message_id(Some(id));
        let (answer, answered) = crossbeam_channel::bounded(1);
        lock(&self.asked).insert(id, answer);

        if self.outbox.send(request).is_err() {
            lock(&self.asked).remove(&id);
            return Err(Error::State("the peer link's thread has ended"));
        }
        self.wake.wake();
        Ok(answered)
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

    /// Sends `request`, a job of the peer's ranks; the submission completes with what they
    /// return. For a job of a group, `reporting` is the group's sequences and the rank the peer
    /// reports there as, which take the peer's ended sequence before the submission completes;
    /// a generation's switch, commit or rollback addresses no group and states none.
    ///
    /// # Errors
    ///
    /// Once the link is lost: the group cannot run its jobs without the peer's ranks.
    pub(super) fn send(
        &self,
        mut request: MainProcessRequest,
        reporting: Option<(&Arc<Sequences>, usize)>,
    ) -> Result<Submission> {
        self.live()?;

        let id = self.next.fetch_add(1, Ordering::Relaxed);
        request.set_message_id(Some(id));
        let (submission, state) = Submission::pending(SubmissionId(id));
        let reporting = reporting.map(|(sequences, rank)| (Arc::clone(sequences), rank));
        lock(&self.pending).insert(id, (state, reporting));

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

/// The value a peer's data plane operation returned, in the link's encoding, once its answer
/// arrived.
///
/// # Errors
///
/// The peer's failure of the operation, an answer of another kind, or a link lost before the
/// answer.
pub(crate) fn operated(rank: u32, answer: &Answer) -> Result<Vec<u8>> {
    let answered = answer
        .recv()
        .map_err(|_| Error::State("the peer link ended before it answered"))?;
    match answered {
        Ok(MainProcessResponse::Operated { reply, .. }) => Ok(reply),
        Ok(other) => Err(Error::transport(format!(
            "rank {rank}'s engine process answered an operation with {:?}",
            other.kind()
        ))),
        Err(reason) => Err(Error::transport(format!(
            "rank {rank}'s engine process failed the operation: {reason}"
        ))),
    }
}

/// The scheduler node's side of the connection to the peer whose ranks start at `rank`: sends
/// what the scheduler queued, settles each response by its `message_id`, and on a close fails
/// every submission and request the peer left open.
fn remote_link(
    rank: u32,
    mut channel: Connection,
    queued: &Receiver<MainProcessRequest>,
    pending: &Pending,
    asked: &Asked,
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
                    asked,
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
                Ok(Some(response)) => settle(rank, pending, asked, response),
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

    fail_open(pending, asked, &failure);

    // A close this side asked for is not a loss of the peer.
    if !closing.load(Ordering::Acquire) {
        liveness.lose(failure);
    }
}

/// Settles the submission or the request `response` belongs to.
fn settle(rank: u32, pending: &Pending, asked: &Asked, response: MainProcessResponse) {
    // An answer to a request that is no job of a group, or its refusal, settles that request.
    let question = match &response {
        MainProcessResponse::Placed { message_id, .. }
        | MainProcessResponse::Operated { message_id, .. }
        | MainProcessResponse::Error { message_id, .. } => {
            message_id.and_then(|id| lock(asked).remove(&id))
        }
        _ => None,
    };
    if let Some(question) = question {
        let _ = question.send(match response {
            MainProcessResponse::Error { error, .. } => Err(error.message),
            answer => Ok(answer),
        });
        return;
    }

    let (id, status, ended) = match response {
        MainProcessResponse::Result {
            message_id,
            result,
            requests,
            ended,
        } => match messages::decode_requests(&requests) {
            Ok(requests) => (
                message_id,
                Outcome::Success((result, requests)),
                Some(ended),
            ),
            Err(error) => (
                message_id,
                Outcome::Failed(fault(Error::transport(format!(
                    "rank {rank}'s engine process returned a request that is not one: {error}"
                )))),
                None,
            ),
        },
        MainProcessResponse::Error {
            message_id,
            error,
            requests,
            ended,
        } => {
            let failure = Error::transport(format!(
                "rank {rank}'s engine process failed the job: {}",
                error.message
            ));
            // The copies that started with a failed batch end with it, on every rank: the
            // scheduler gives each up with the `Finish` the peer's ranks returned.
            let failure = match messages::decode_requests(&requests) {
                Ok(finished) if finished.is_empty() => failure,
                Ok(finished) => Error::Copies {
                    error: Box::new(failure),
                    finished,
                },
                Err(error) => Error::transport(format!(
                    "rank {rank}'s engine process returned a request that is not one: {error}"
                )),
            };
            (message_id, Outcome::Failed(fault(failure)), Some(ended))
        }
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
        Some((state, reporting)) => {
            // The group's word covers what the peer's ranks ended before anyone reads the job's
            // outcome.
            if let (Some((sequences, reporting)), Some(ended)) = (reporting, ended) {
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

/// Fails every submission and request the peer left open with `reason`.
fn fail_open(pending: &Pending, asked: &Asked, reason: &str) {
    for (_, (state, _)) in lock(pending).drain() {
        state.finish(Outcome::Failed(fault(Error::transport(reason.to_owned()))));
    }
    for (_, question) in lock(asked).drain() {
        let _ = question.send(Err(reason.to_owned()));
    }
}

/// `error`, as a failed submission carries it.
fn fault(error: Error) -> Arc<Fault> {
    Arc::new(error)
}

/// What a peer runs a data plane's operation with: the operation's value as JSON, or its
/// failure in words. It runs on the peer's operations thread, one operation at a time.
pub type Operate = Box<dyn FnMut(DataOperation) -> std::result::Result<Vec<u8>, String> + Send>;

/// One job of this process's ranks the responder waits for, in the order it was handed: the
/// request's `message_id`, the group whose ended sequence its response states (none for a
/// generation's switch, commit or rollback), and its submission.
type Awaited = (u64, Option<GroupId>, Submission);

/// A peer's side of its connection to the scheduler node: hands every job the scheduler node
/// sends to this process's ranks, and responds to each once its ranks completed it.
pub struct Upstream {
    closing: Arc<AtomicBool>,
    wake: service::Wake,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl Upstream {
    /// Serves the scheduler node's requests arriving on `channel` with `route`, this process's
    /// ranks, running each data plane's operation with `operate`; a close of the connection
    /// marks `liveness` lost, and the scheduler node's orderly close marks it closed.
    ///
    /// Three threads: the link, which owns the connection and hands every job to the ranks in
    /// the order it arrived; the responder, which waits for each job in that order (each worker
    /// takes its jobs in that order) and gives its response to the link, with the group's ended
    /// sequence as this process knows it then; and the operations thread, which runs the data
    /// plane's operations one at a time in the order they arrived.
    ///
    /// # Errors
    ///
    /// When a thread cannot start.
    pub fn serve(
        channel: Connection,
        route: Arc<Route>,
        liveness: Arc<Liveness>,
        mut operate: Operate,
    ) -> Result<Self> {
        let (responses, responded) = crossbeam_channel::unbounded::<MainProcessResponse>();
        let (waiting, awaited) = crossbeam_channel::unbounded::<Awaited>();
        let (operations, operating) = crossbeam_channel::unbounded::<(u64, DataOperation)>();
        let closing = Arc::new(AtomicBool::new(false));
        let wake = channel.wake();

        let responder = {
            let wake = wake.clone();
            let responses = responses.clone();
            let route = Arc::clone(&route);

            std::thread::Builder::new()
                .name("peer-responder".into())
                .spawn(move || {
                    for (id, group, submission) in awaited {
                        // Each rank stated its ended sequence before its part completed, so the
                        // group's is read once the job completed.
                        let outcome = submission.wait();
                        let ended = group.map_or(Ok(0), |group| route.completed(group));
                        let response = match (outcome, ended) {
                            (Outcome::Success((output, requests)), Ok(ended)) => {
                                let mut response =
                                    MainProcessResponse::result(output, &requests, ended);
                                response.set_message_id(Some(id));
                                response
                            }
                            (Outcome::Success(_), Err(error)) => {
                                MainProcessResponse::failed(id, error.to_string(), &[], 0)
                            }
                            (Outcome::Failed(error), ended) => {
                                // A failed batch returns the `Finish` of each copy it started.
                                let finished = error
                                    .downcast_ref::<Error>()
                                    .map_or(&[][..], Error::finished);
                                let ended = ended.unwrap_or(0);
                                MainProcessResponse::failed(id, error.to_string(), finished, ended)
                            }
                            (Outcome::Cancelled, _) => MainProcessResponse::failed(
                                id,
                                "the job was cancelled".into(),
                                &[],
                                0,
                            ),
                        };
                        if responses.send(response).is_err() {
                            return;
                        }
                        wake.wake();
                    }
                })
                .map_err(|error| Error::transport_source("starting the peer responder", error))?
        };

        let operator = {
            let wake = wake.clone();

            std::thread::Builder::new()
                .name("peer-operations".into())
                .spawn(move || {
                    for (id, operation) in operating {
                        let response = match operate(operation) {
                            Ok(reply) => MainProcessResponse::Operated {
                                message_id: Some(id),
                                reply,
                            },
                            Err(reason) => MainProcessResponse::failed(id, reason, &[], 0),
                        };
                        if responses.send(response).is_err() {
                            return;
                        }
                        wake.wake();
                    }
                })
                .map_err(|error| {
                    Error::transport_source("starting the peer's operations thread", error)
                })?
        };

        let link = {
            let closing = Arc::clone(&closing);

            std::thread::Builder::new()
                .name("peer-upstream".into())
                .spawn(move || {
                    upstream_link(
                        channel,
                        route.as_ref(),
                        (&waiting, &operations),
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
            threads: Mutex::new(vec![link, responder, operator]),
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
/// ranks and each data plane's operation to the operations thread, in the order they arrived,
/// and sends each response the responder and the operations thread queued.
fn upstream_link(
    mut channel: Connection,
    route: &Route,
    (waiting, operations): (&Sender<Awaited>, &Sender<(u64, DataOperation)>),
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
                    Some(batch.group),
                    route.run(sequence, *batch, pass, false),
                ),
                MainProcessRequest::Shadow {
                    message_id,
                    sequence,
                    batch,
                } => (
                    message_id.unwrap_or(0),
                    Some(batch.group),
                    route.run(sequence, *batch, false, true),
                ),
                MainProcessRequest::Map {
                    message_id,
                    group,
                    slot,
                    request,
                    completion,
                } => (
                    message_id.unwrap_or(0),
                    Some(group),
                    mapped(route, group, slot, &request, &completion),
                ),
                MainProcessRequest::Stamp {
                    message_id,
                    group,
                    requests,
                } => (
                    message_id.unwrap_or(0),
                    Some(group),
                    messages::decode_requests(&requests)
                        .map_err(|error| {
                            Error::transport_source("reading the requests of a stamp", error)
                        })
                        .and_then(|requests| route.stamp(group, requests)),
                ),
                MainProcessRequest::Switch {
                    message_id,
                    generation,
                } => (message_id.unwrap_or(0), None, route.switch(generation)),
                MainProcessRequest::Commit {
                    message_id,
                    generation,
                } => (message_id.unwrap_or(0), None, route.commit(generation)),
                MainProcessRequest::Rollback {
                    message_id,
                    generation,
                } => (message_id.unwrap_or(0), None, route.rollback(generation)),
                MainProcessRequest::Data {
                    message_id,
                    operation,
                } => {
                    if operations
                        .send((message_id.unwrap_or(0), operation))
                        .is_err()
                    {
                        failed = Some("the peer's operations thread ended".into());
                        break;
                    }
                    continue;
                }
                // The session was read before this thread owned the connection.
                MainProcessRequest::Session { message_id, .. } => (
                    message_id.unwrap_or(0),
                    None,
                    Err(Error::Invalid(
                        "the session was relayed once, at this process's start".to_owned(),
                    )),
                ),
                MainProcessRequest::Close { .. } => {
                    // The scheduler node's orderly end: this process ends in order with it.
                    let _ = channel.flush();
                    channel.shutdown();
                    liveness.close();
                    return;
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
                    let ended = group.map_or(Ok(0), |group| route.completed(group));
                    let refusal = MainProcessResponse::failed(
                        id,
                        error.to_string(),
                        error.finished(),
                        ended.unwrap_or(0),
                    );
                    if let Err(error) = channel.send(&refusal) {
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
    route: &Route,
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

    route.map(group, slot, id, request, completion)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::{Batch, BatchId, BatchOutput, Capacity, Columns, Phase, Shape, Variant};
    use crate::executor::tests::host_batch;
    use storage::protocol::{Location, Segment, UnitRun};
    use storage::runtime::{Region, RequestId, Tier, VERSION};

    /// A bound on the tests' waits that only a defect reaches.
    const LONG: Duration = Duration::from_secs(30);

    /// The worker group of the tests' batches.
    const GROUP: GroupId = GroupId(0);

    /// Liveness of a test process, which a lost link must not end, with what its callback
    /// receives once a link is lost or closed.
    fn liveness() -> (Arc<Liveness>, Receiver<()>) {
        let (lost, called) = crossbeam_channel::unbounded();
        let callback: Waker = Arc::new(move || {
            let _ = lost.send(());
        });
        (Liveness::new(callback, Duration::from_secs(3600)), called)
    }

    /// Batch `id` of [`GROUP`]: a greedy decode of no row.
    fn batch(id: u64) -> Batch {
        Batch {
            id: BatchId(id),
            group: GROUP,
            phase: Phase::Decode,
            generation: 0,
            shape: Shape::decode(1, Variant::DecodeGreedy),
            requests: Vec::new(),
            columns: Columns::default(),
            media: Vec::new(),
            new_slots: Vec::new(),
        }
    }

    /// The next request the scheduler node sends on `channel`.
    fn next_request(channel: &mut Connection) -> MainProcessRequest {
        loop {
            if let Some(request) = channel.try_recv::<MainProcessRequest>().expect("a request") {
                return request;
            }
            channel.wait(LONG).expect("the request arrives");
        }
    }

    /// Sends `response` as the answer to the request sent under `id`.
    fn answer(channel: &mut Connection, id: Option<u64>, mut response: MainProcessResponse) {
        response.set_message_id(id);
        channel.deliver(&response).expect("the response is sent");
    }

    /// The scheduler node's link to one peer that registers as rank `rank` of engine `engine`
    /// and then runs `peer` over its channel, with the peer's thread.
    fn linked(
        rank: u32,
        engine: u64,
        liveness: Arc<Liveness>,
        peer: impl FnOnce(Connection) + Send + 'static,
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

    /// The route of a node that holds no rank of [`GROUP`], whose one rank `remote` holds.
    fn routed(remote: &Arc<Remote>) -> (Route, Arc<Sequences>) {
        let sequences = Sequences::new(GROUP, 1, Vec::new());
        let route = Route::new(
            vec![Arc::clone(&sequences)],
            Vec::new(),
            vec![(vec![GROUP], Arc::clone(remote))],
        )
        .expect("the route");
        (route, sequences)
    }

    /// A text group's capacity: decode on two slots.
    fn capacity() -> Capacity {
        let mut rows = [0; Phase::COUNT];
        rows[Phase::Decode.index()] = 2;
        Capacity {
            phases: 1 << Phase::Decode.index(),
            slots: 2,
            kv_cache: None,
            cache_groups: Vec::new(),
            seeds: vec![storage::Key([4; 32])],
            buckets: vec![Shape::decode(2, Variant::DecodeGreedy)],
            rows,
            max_model_len: 4096,
            denoise_steps: 0,
            bos_token_id: 1,
            eos_token_ids: vec![2],
            feature_bytes: 0,
            encoder_identity: String::new(),
            latent_seed: 0,
            media_bytes: Vec::new(),
            call_window: 2,
        }
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
    fn a_relayed_session_is_answered_with_what_the_peer_s_ranks_hold() {
        let open = Open {
            version: VERSION,
            epoch: 3,
            trial: false,
            regions: vec![Region {
                name: "kv 0 ab".to_owned(),
                tier: Tier::Device,
                space: "kv 0 ab".to_owned(),
                unit_bytes: 128,
                part_bytes: 64,
                reserved: 1,
                units: 0,
                limit: 65,
                segments: Vec::new(),
                memory: String::new(),
            }],
            classes: Vec::new(),
            trajectory: 0,
            groups: Vec::new(),
            schedulers: Vec::new(),
            peers: Vec::new(),
        };
        let opened = Opened {
            regions: vec![(4, 0)],
            segments: vec![vec![(
                Segment {
                    epoch: 2,
                    ordinal: 0,
                },
                0,
                32,
            )]],
            limits: vec![65],
            registers: Vec::new(),
        };
        let created = vec![(
            "kv 0 ab".to_owned(),
            vec![(
                Segment {
                    epoch: 3,
                    ordinal: 1,
                },
                32,
                33,
            )],
        )];
        let places = vec![(
            "kv 0 ab".to_owned(),
            "a/0,b/0".to_owned(),
            1,
            vec![UnitRun {
                field: "prefix_cache.key.values/0".to_owned(),
                key: storage::Key([9; 32]),
                location: Location::Device { gpu: [7; 16] },
                first: 0,
                count: 65,
                offset: 0,
                bytes: 32,
            }],
        )];
        let capacities = vec![(GROUP, capacity())];

        let (liveness, _called) = liveness();
        let relayed = (open.clone(), opened.clone());
        let held = (created.clone(), places.clone(), capacities.clone());
        let (remote, peer) = linked(1, 5, liveness, move |mut channel| {
            // The session is the first request after the registration, read before any thread
            // owns the connection.
            let (id, open, opened) = session(&mut channel, LONG).expect("the relayed session");
            assert_eq!((open, opened), relayed);

            let (created, places, capacities) = held;
            let placed = MainProcessResponse::Placed {
                message_id: None,
                endpoint: "10.0.0.2:7000".to_owned(),
                created,
                places,
                capacities,
            };
            answer(&mut channel, Some(id), placed);

            assert!(matches!(
                next_request(&mut channel),
                MainProcessRequest::Close { .. }
            ));
        });

        let answered = remote
            .session(&open, &opened)
            .expect("the session is relayed")
            .recv_timeout(LONG)
            .expect("the peer answers")
            .expect("the peer placed its ranks");
        let MainProcessResponse::Placed {
            endpoint,
            created: created_read,
            places: places_read,
            capacities: capacities_read,
            ..
        } = answered
        else {
            panic!("the peer answered the session with {answered:?}");
        };
        assert_eq!(endpoint, "10.0.0.2:7000");
        assert_eq!(created_read, created);
        assert_eq!(places_read, places);
        assert_eq!(capacities_read, capacities);

        remote.close();
        peer.join().expect("the peer");
    }

    #[test]
    fn a_batch_that_failed_on_a_peer_gives_up_the_copies_it_started() {
        let finish = Request::Finish {
            copy: RequestId(31),
            ok: false,
            after: vec![(GROUP, 1)],
        };
        let (liveness, _called) = liveness();

        let returned = finish.clone();
        let (remote, peer) = linked(1, 3, liveness, move |mut channel| {
            let MainProcessRequest::Submit {
                message_id,
                sequence,
                ..
            } = next_request(&mut channel)
            else {
                panic!("the scheduler node sent no batch");
            };
            assert_eq!(sequence, 1);

            // The peer's ranks failed the batch after its copy started, and ended it.
            let id = message_id.unwrap_or(0);
            let failed = MainProcessResponse::failed(id, "a device fault".into(), &[returned], 1);
            channel.deliver(&failed).expect("the failure is sent");

            assert!(matches!(
                next_request(&mut channel),
                MainProcessRequest::Close { .. }
            ));
        });
        let (route, sequences) = routed(&remote);

        let submission = route.submit(batch(1)).expect("the batch is handed");
        let Outcome::Failed(fault) = submission.wait() else {
            panic!("a batch the peer failed succeeded");
        };
        assert!(fault.to_string().contains("a device fault"), "{fault}");
        // The copy is given up as a local rank's would be, stamped with the batch.
        let error = fault
            .downcast_ref::<Error>()
            .expect("the worker's own error");
        assert_eq!(error.finished(), [finish]);
        // What the peer's ranks ended counts towards the group's `completed`.
        assert_eq!(sequences.completed(), 1);

        remote.close();
        peer.join().expect("the peer");
    }

    /// A video encode round is the work of every process that holds ranks of its group, each
    /// encoding the units its ranks decoded: a node that holds none of the group's ranks hands
    /// it to both peers to run, its rows naming no rank, and a row one peer failed fails. Any
    /// other host batch runs once, on the first peer, and the second passes it.
    #[test]
    fn a_video_encode_runs_in_every_process_and_another_host_batch_in_the_first_alone() {
        let (liveness, _called) = liveness();

        // Each peer runs or passes each batch as the node hands it; the second fails its part of
        // every row it runs.
        let peer = |failing: bool, expected: [(u64, bool); 2]| {
            move |mut channel: Connection| {
                for (sequence, pass) in expected {
                    let MainProcessRequest::Submit {
                        message_id,
                        sequence: handed,
                        pass: passed,
                        batch,
                    } = next_request(&mut channel)
                    else {
                        panic!("the scheduler node sent no batch");
                    };
                    assert_eq!((handed, passed), (sequence, pass));
                    assert!(batch.media.iter().all(|row| row.rank.is_none()));

                    let mut call = crate::runner::performed();
                    if failing {
                        call.status = crate::batch::CallStatus::Error;
                        call.error_code = Some(crate::batch::ErrorCode::ComputeError);
                    }
                    let calls = if pass { Vec::new() } else { vec![call] };
                    let output = BatchOutput {
                        calls,
                        ..BatchOutput::default()
                    };
                    let ended = MainProcessResponse::result(output, &[], sequence);
                    answer(&mut channel, message_id, ended);
                }
                assert!(matches!(
                    next_request(&mut channel),
                    MainProcessRequest::Close { .. }
                ));
            }
        };
        let (first, first_peer) = linked(
            1,
            3,
            Arc::clone(&liveness),
            peer(false, [(1, false), (2, false)]),
        );
        let (second, second_peer) = linked(2, 3, liveness, peer(true, [(1, false), (2, true)]));
        let sequences = Sequences::new(GROUP, 2, Vec::new());
        let links = vec![
            (vec![GROUP], Arc::clone(&first)),
            (vec![GROUP], Arc::clone(&second)),
        ];
        let route = Route::new(vec![Arc::clone(&sequences)], Vec::new(), links).expect("the route");

        let encode = host_batch(1, GROUP, Phase::VideoEncode, Variant::VideoEncode);
        let Outcome::Success((encoded, _)) = route.submit(encode).expect("handed").wait() else {
            panic!("the encode failed");
        };
        assert_eq!(encoded.calls.len(), 1);
        assert_eq!(encoded.calls[0].status, crate::batch::CallStatus::Error);

        let track = host_batch(2, GROUP, Phase::AudioEncode, Variant::AudioEncode);
        let Outcome::Success((muxed, _)) = route.submit(track).expect("handed").wait() else {
            panic!("the audio encode failed");
        };
        assert_eq!(muxed.calls.len(), 1);
        assert_eq!(muxed.calls[0].status, crate::batch::CallStatus::Ok);
        assert_eq!(sequences.completed(), 2);

        first.close();
        second.close();
        first_peer.join().expect("the first peer");
        second_peer.join().expect("the second peer");
    }

    #[test]
    fn a_switch_reaches_a_peer_between_the_batches_handed_before_and_after_it() {
        let (liveness, _called) = liveness();

        let (remote, peer) = linked(1, 3, liveness, |mut channel| {
            let mut seen = Vec::new();
            for _ in 0..3 {
                let (id, response) = match next_request(&mut channel) {
                    MainProcessRequest::Submit {
                        message_id,
                        sequence,
                        ..
                    } => {
                        seen.push(format!("batch {sequence}"));
                        let ended =
                            MainProcessResponse::result(BatchOutput::default(), &[], sequence);
                        (message_id, ended)
                    }
                    MainProcessRequest::Switch {
                        message_id,
                        generation,
                    } => {
                        seen.push(format!("switch {generation}"));
                        let ended = MainProcessResponse::result(BatchOutput::default(), &[], 0);
                        (message_id, ended)
                    }
                    other => panic!("the scheduler node sent {other:?}"),
                };
                answer(&mut channel, id, response);
            }
            assert_eq!(seen, ["batch 1", "switch 2", "batch 2"]);

            assert!(matches!(
                next_request(&mut channel),
                MainProcessRequest::Close { .. }
            ));
        });
        let (route, sequences) = routed(&remote);

        let before = route.submit(batch(1)).expect("the first batch is handed");
        let switch = route.switch(2).expect("the switch is handed");
        let after = route.submit(batch(2)).expect("the second batch is handed");
        for submission in [before, switch, after] {
            assert!(matches!(submission.wait(), Outcome::Success(_)));
        }
        // The switch takes no number: the batch after it is the group's second.
        assert_eq!(sequences.completed(), 2);

        remote.close();
        peer.join().expect("the peer");
    }

    #[test]
    fn a_peer_runs_the_data_plane_s_operations_in_the_order_they_arrived() {
        let (liveness, _called) = liveness();
        let (peer_liveness, peer_called) = self::liveness();
        let operated_log: Arc<Mutex<Vec<DataOperation>>> = Arc::default();

        let log = Arc::clone(&operated_log);
        let (remote, peer) = linked(1, 3, liveness, move |channel| {
            let route = Route::new(Vec::new(), Vec::new(), Vec::new()).expect("a route");
            let operate: Operate = Box::new(move |operation| {
                lock(&log).push(operation.clone());
                match operation {
                    DataOperation::Prepared { generation } => Ok(generation.to_le_bytes().to_vec()),
                    _ => Err("this peer refuses the operation".to_owned()),
                }
            });
            let upstream = Upstream::serve(
                channel,
                Arc::new(route),
                Arc::clone(&peer_liveness),
                operate,
            )
            .expect("the peer serves");

            // The scheduler node's orderly close ends the epoch here too: it is no loss.
            peer_called.recv_timeout(LONG).expect("the close is told");
            assert!(peer_liveness.closed());
            assert_eq!(peer_liveness.lost(), None);
            upstream.close();
        });

        let sent = [
            DataOperation::Prepared { generation: 3 },
            DataOperation::Sleep,
            DataOperation::Prepared { generation: 1 },
            DataOperation::Prepared { generation: 2 },
        ];
        let answers: Vec<Answer> = sent
            .iter()
            .map(|operation| {
                remote
                    .operate(operation.clone())
                    .expect("the operation is sent")
            })
            .collect();

        // Each reply comes back under its own request, a refusal as the operation's error.
        let replies: Vec<std::result::Result<Vec<u8>, String>> = answers
            .iter()
            .map(|answer| operated(1, answer).map_err(|error| error.to_string()))
            .collect();
        assert_eq!(replies[0], Ok(3_u64.to_le_bytes().to_vec()));
        assert!(
            replies[1]
                .as_ref()
                .is_err_and(|error| error.contains("refuses")),
            "{:?}",
            replies[1]
        );
        assert_eq!(replies[2], Ok(1_u64.to_le_bytes().to_vec()));
        assert_eq!(replies[3], Ok(2_u64.to_le_bytes().to_vec()));
        assert_eq!(*lock(&operated_log), sent);

        remote.close();
        peer.join().expect("the peer");
    }

    #[test]
    fn a_closed_peer_fails_its_submissions_and_is_lost() {
        let (liveness, called) = liveness();

        let (remote, peer) = linked(1, 3, Arc::clone(&liveness), |mut channel| {
            // Read the one submission, then go away without responding.
            next_request(&mut channel);
            channel.shutdown();
        });

        let sequences = Sequences::new(GROUP, 1, Vec::new());
        let submit = || MainProcessRequest::submit(1, false, batch(1));
        let submission = remote
            .send(submit(), Some((&sequences, 0)))
            .expect("the batch is sent");
        assert!(matches!(submission.wait(), Outcome::Failed(_)));
        peer.join().expect("the peer");

        // The close is the peer's death signal: the link calls the control thread's callback,
        // once, with the reason recorded.
        assert!(
            called.recv_timeout(LONG).is_ok(),
            "the lost link did not call the control thread's callback"
        );
        assert!(liveness.lost().is_some());
        assert!(!liveness.closed());
        assert!(remote.send(submit(), Some((&sequences, 0))).is_err());

        // Nothing the peer ran is counted as ended.
        assert_eq!(sequences.completed(), 0);
    }
}
