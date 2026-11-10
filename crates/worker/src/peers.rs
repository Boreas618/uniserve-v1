//! The peer channel: the batches a scheduler node's main process submits to the other main
//! processes of its epoch, and their answers.
//!
//! An epoch whose worker group has ranks on several hosts runs one main process per host. The
//! scheduler node's process holds the schedulers; every other process (a peer) holds only ranks.
//! A peer dials the scheduler node's peer listener when it starts and registers
//! ([`ipc::PeerRegistration`]); the scheduler node accepts every peer its launch counts before
//! its ranks load ([`accept`]), so a peer that never arrives fails the start by name.
//!
//! From then on each connection carries one batch contract: the scheduler node sends
//! [`ipc::MainProcessRequest::Submit`] (and `Shadow`) with the [`ipc::execution::Batch`] its
//! scheduler formed, and the peer answers [`ipc::MainProcessResponse::Result`] with the
//! [`ipc::execution::BatchOutput`] its ranks produced, or `Error`, under the request's
//! `message_id`. The ranks of one group run every batch together, so the scheduler's submission
//! of a batch is joined with the peer's ([`ipc::execution::Submission::joined`]).
//!
//! Each end's connection is owned by one link thread, which sends what other threads queued
//! (firing the channel's local wake), reads what arrived and settles it; nothing else touches
//! the socket. A connection's close is the only death signal of the process at its other end:
//! the link marks the process's [`Liveness`] lost and wakes its control thread, which ends the
//! process, and the daemon's standby rules then replace the epoch's failure domain.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use ipc::execution::{Batch, Outcome, Submission, SubmissionId, SubmissionState};
use ipc::{
    Channel, Listener, MainProcessRequest, MainProcessResponse, MainProcessResponseError,
    PeerAccepted, PeerRegistration,
};

use crate::batch::Workers;
use crate::{Error, Result};

/// How long a link thread sleeps when nothing arrives: the bound on how late it notices a queue
/// it was not woken for. Every queue fires the channel's wake, so this is a backstop.
const LINK_POLL: Duration = Duration::from_millis(500);

/// How long a registration may take to arrive once a peer connected: its first frame.
const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(10);

/// Whether the processes at the far ends of a main process's peer links still serve, and the
/// first reason one stopped.
///
/// A link that closes sets it and wakes the control thread, which reads it on its next pass and
/// ends the process: a group missing a rank cannot run a batch, and the daemon replaces the
/// whole failure domain rather than one process of it.
pub struct Liveness {
    lost: AtomicBool,
    reason: Mutex<Option<String>>,
    wake: ipc::Wake,
}

impl Liveness {
    /// Liveness that wakes the control thread through `wake` when a link is lost.
    pub fn new(wake: ipc::Wake) -> Arc<Self> {
        Arc::new(Self {
            lost: AtomicBool::new(false),
            reason: Mutex::new(None),
            wake,
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
                tracing::error!(%reason, "a peer link of this main process was lost");
                *slot = Some(reason);
            }
            first
        };
        self.lost.store(true, Ordering::Release);
        self.wake.wake();
        // The control thread ends the process on its next pass. A pass that never comes - a call
        // of the control thread that waits on a device whose collective waits for the lost
        // process's ranks - does not keep the process: it ends after `LOST_EXIT` all the same,
        // without its orderly close, which a hung collective could not finish either. Its daemon
        // removes the driver and NCCL files it leaves when it reaps it.
        if first {
            let spawned = std::thread::Builder::new()
                .name("peer-lost".into())
                .spawn(|| {
                    std::thread::sleep(LOST_EXIT);
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

/// How long a main process that lost a peer link has to end before it is ended.
const LOST_EXIT: Duration = Duration::from_secs(20);

/// Accepts the registrations of `expected` peers on `listener`, each of the epoch whose
/// scheduler node runs under engine number `engine`, and answers each channel with the first
/// rank its process serves.
///
/// # Errors
///
/// Fails when the peers have not all registered within `timeout`, when a registration names
/// another epoch or another build (the two ends would decode each other's frames into other
/// meanings), or when two peers claim one rank.
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
            .map_err(|error| Error::transport_source("accepting a peer main process", error))?
        else {
            if Instant::now() >= deadline {
                return Err(Error::transport(format!(
                    "{} of the epoch's {expected} peer main processes registered within {} s",
                    accepted.len(),
                    timeout.as_secs()
                )));
            }
            std::thread::sleep(Duration::from_millis(20));
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
                "two peer main processes registered for rank {}",
                registration.rank
            )));
        }
        let rank = registration.rank;
        channel
            .send(&PeerAccepted { rank })
            .map_err(|error| Error::transport_source("answering a peer's registration", error))?;
        tracing::info!(rank, "a peer main process registered");
        accepted.push((rank, channel));
    }
    accepted.sort_by_key(|(rank, _)| *rank);
    Ok(accepted)
}

/// Dials the scheduler node's peer listener at `address` and registers this process as the one
/// whose ranks start at `rank`, in the epoch whose scheduler node runs under engine `engine`.
///
/// The scheduler node may still be starting, so the dial is repeated until `timeout`.
///
/// # Errors
///
/// Fails when no connection is made in time, or the scheduler node does not accept the
/// registration.
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
        if let Some(accepted) = channel
            .try_recv::<PeerAccepted>()
            .map_err(|error| Error::transport_source("reading the registration's answer", error))?
        {
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
                "the scheduler node at {address} did not answer this process's registration"
            )));
        }
        channel.wait(left).map_err(|error| {
            Error::transport_source("awaiting the registration's answer", error)
        })?;
    }
}

/// The scheduler node's link to one peer main process: where its scheduler's batches for the
/// peer's ranks go, and how their outcomes come back.
pub struct Remote {
    /// The first rank the peer's process serves, which its registration stated.
    rank: u32,
    outbox: Sender<MainProcessRequest>,
    wake: ipc::Wake,
    /// The submissions the peer has not answered yet, by the `message_id` they were sent under.
    pending: Arc<Mutex<HashMap<u64, Arc<SubmissionState>>>>,
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
    /// Fails when the thread cannot start.
    pub fn spawn(rank: u32, channel: Channel, liveness: Arc<Liveness>) -> Result<Arc<Self>> {
        let (outbox, queued) = crossbeam_channel::unbounded();
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let closing = Arc::new(AtomicBool::new(false));
        let wake = channel.wake();
        let thread = {
            let pending = Arc::clone(&pending);
            let liveness = Arc::clone(&liveness);
            let closing = Arc::clone(&closing);
            std::thread::Builder::new()
                .name(format!("peer-{rank}"))
                .spawn(move || remote_link(rank, channel, &queued, &pending, &liveness, &closing))
                .map_err(|error| Error::transport_source("starting a peer link", error))?
        };
        Ok(Arc::new(Self {
            rank,
            outbox,
            wake,
            pending,
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

    /// Sends `batch` to the peer's ranks; the submission completes with their answer.
    ///
    /// # Errors
    ///
    /// Fails once the link is lost: the group cannot run the batch without the peer's ranks.
    pub fn submit(&self, batch: Batch) -> Result<Submission> {
        self.send(MainProcessRequest::submit(batch))
    }

    /// Sends the shadow `batch` to the peer's ranks, as [`Self::submit`] sends a batch.
    ///
    /// # Errors
    ///
    /// As [`Self::submit`].
    pub fn shadow(&self, batch: Batch) -> Result<Submission> {
        self.send(MainProcessRequest::shadow(batch))
    }

    fn send(&self, mut request: MainProcessRequest) -> Result<Submission> {
        if let Some(reason) = self.liveness.lost() {
            return Err(Error::transport(format!(
                "the peer link of rank {} is lost: {reason}",
                self.rank
            )));
        }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        request.set_message_id(Some(id));
        let (submission, state) = Submission::pending(SubmissionId(id));
        lock(&self.pending).insert(id, state);
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

/// The scheduler node's side of one peer connection: sends what the scheduler queued, settles
/// each answer by its `message_id`, and on a close fails every unanswered submission.
fn remote_link(
    rank: u32,
    mut channel: Channel,
    queued: &Receiver<MainProcessRequest>,
    pending: &Mutex<HashMap<u64, Arc<SubmissionState>>>,
    liveness: &Liveness,
    closing: &AtomicBool,
) {
    let failure = loop {
        let mut failed = None;
        while let Ok(request) = queued.try_recv() {
            let close = matches!(request, MainProcessRequest::Close { .. });
            if let Err(error) = channel.send(&request) {
                failed = Some(format!("sending to rank {rank}'s main process: {error}"));
                break;
            }
            if close {
                // An orderly close: the frame goes out before the connection does.
                let _ = channel.flush();
                channel.shutdown();
                fail_pending(pending, "the epoch's scheduler node closed its peer links");
                return;
            }
        }
        if let Some(failed) = failed {
            break failed;
        }
        loop {
            match channel.try_recv::<MainProcessResponse>() {
                Ok(Some(response)) => settle(rank, pending, response),
                Ok(None) => break,
                Err(error) => {
                    failed = Some(format!("rank {rank}'s main process: {error}"));
                    break;
                }
            }
        }
        if let Some(failed) = failed {
            break failed;
        }
        if let Err(error) = channel.wait(LINK_POLL) {
            break format!("rank {rank}'s main process: {error}");
        }
    };
    fail_pending(pending, &failure);
    // A close this side asked for is not a loss of the peer.
    if !closing.load(Ordering::Acquire) {
        liveness.lose(failure);
    }
}

/// Settles the submission `response` answers.
fn settle(
    rank: u32,
    pending: &Mutex<HashMap<u64, Arc<SubmissionState>>>,
    response: MainProcessResponse,
) {
    let (id, outcome) = match response {
        MainProcessResponse::Result { message_id, result } => {
            (message_id, Outcome::Success(result))
        }
        MainProcessResponse::Error { message_id, error } => (
            message_id,
            Outcome::Failed(fault(format!(
                "rank {rank}'s main process failed the batch: {}",
                error.message
            ))),
        ),
        MainProcessResponse::Stalled { stall } => {
            // The peer's own watchdog fails its rank on a stall; its close follows.
            tracing::warn!(
                rank,
                batch = ?stall.batch_id,
                running_ms = stall.running_ms,
                "a peer main process reports a stalled job"
            );
            return;
        }
        other => {
            tracing::warn!(rank, kind = ?other.kind(), "a peer answered what nothing asked");
            return;
        }
    };
    let state = id.and_then(|id| lock(pending).remove(&id));
    match state {
        Some(state) => state.finish(outcome),
        None => tracing::warn!(
            rank,
            ?id,
            "a peer answered a submission this link did not send"
        ),
    }
}

/// Fails every unanswered submission with `reason`.
fn fail_pending(pending: &Mutex<HashMap<u64, Arc<SubmissionState>>>, reason: &str) {
    for (_, state) in lock(pending).drain() {
        state.finish(Outcome::Failed(fault(reason.to_owned())));
    }
}

/// A failure of the peer channel, as a submission's outcome carries it.
fn fault(reason: String) -> Arc<ipc::execution::Fault> {
    Arc::new(Error::transport(reason))
}

/// A peer's side of its connection to the scheduler node: runs every batch the scheduler node
/// sends on this process's ranks, and answers each once its ranks completed it.
pub struct Upstream {
    closing: Arc<AtomicBool>,
    wake: ipc::Wake,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl Upstream {
    /// Serves the scheduler node's requests arriving on `channel` with `workers`, this process's
    /// ranks; a close of the connection marks `liveness` lost.
    ///
    /// Two threads: the link, which owns the connection, and the responder, which waits for each
    /// submission in the order it was made (each device runs its batches in that order) and hands
    /// its answer back to the link.
    ///
    /// # Errors
    ///
    /// Fails when a thread cannot start.
    pub fn serve(channel: Channel, workers: Arc<Workers>, liveness: Arc<Liveness>) -> Result<Self> {
        let (answers, answered) = crossbeam_channel::unbounded::<MainProcessResponse>();
        let (waiting, awaited) = crossbeam_channel::unbounded::<(u64, Submission)>();
        let closing = Arc::new(AtomicBool::new(false));
        let wake = channel.wake();
        let responder = {
            let wake = wake.clone();
            std::thread::Builder::new()
                .name("peer-responder".into())
                .spawn(move || {
                    for (id, submission) in awaited {
                        let response = match submission.wait() {
                            Outcome::Success(output) => MainProcessResponse::Result {
                                message_id: Some(id),
                                result: output,
                            },
                            Outcome::Failed(error) => refusal(id, error.to_string()),
                            Outcome::Cancelled => refusal(id, "the batch was cancelled".into()),
                        };
                        if answers.send(response).is_err() {
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
                    upstream_link(channel, &workers, &waiting, &answered, &liveness, &closing);
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

/// A peer's side of its connection: submits each batch the scheduler node sends, and sends each
/// answer the responder queued.
fn upstream_link(
    mut channel: Channel,
    workers: &Workers,
    waiting: &Sender<(u64, Submission)>,
    answered: &Receiver<MainProcessResponse>,
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
        while let Ok(response) = answered.try_recv() {
            if let Err(error) = channel.send(&response) {
                failed = Some(format!("answering the scheduler node: {error}"));
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
                    failed = Some(format!("the scheduler node's main process: {error}"));
                    break;
                }
            };
            let (id, submitted) = match request {
                MainProcessRequest::Submit { message_id, batch } => {
                    (message_id.unwrap_or(0), workers.submit(*batch))
                }
                MainProcessRequest::Shadow { message_id, batch } => {
                    (message_id.unwrap_or(0), workers.shadow(*batch))
                }
                MainProcessRequest::Close { .. } => {
                    failed = Some("the scheduler node closed the epoch's peer links".into());
                    break;
                }
                other => {
                    // The scheduler node's daemon commands this process itself (its programs, a
                    // promotion, a shutdown) over its own process interface; the peer channel
                    // carries batches.
                    let id = match &other {
                        MainProcessRequest::Info { message_id }
                        | MainProcessRequest::Promote { message_id }
                        | MainProcessRequest::Hook { message_id, .. } => message_id.unwrap_or(0),
                        _ => 0,
                    };
                    let refused = refusal(
                        id,
                        format!("the peer channel serves batches, not {:?}", other.kind()),
                    );
                    if let Err(error) = channel.send(&refused) {
                        failed = Some(format!("answering the scheduler node: {error}"));
                        break;
                    }
                    continue;
                }
            };
            match submitted {
                Ok(submission) => {
                    if waiting.send((id, submission)).is_err() {
                        failed = Some("the peer responder ended".into());
                        break;
                    }
                }
                Err(error) => {
                    if let Err(error) = channel.send(&refusal(id, error.to_string())) {
                        failed = Some(format!("answering the scheduler node: {error}"));
                        break;
                    }
                }
            }
        }
        if let Some(failed) = failed {
            break failed;
        }
        if let Err(error) = channel.wait(LINK_POLL) {
            break format!("the scheduler node's main process: {error}");
        }
    };
    if !closing.load(Ordering::Acquire) {
        liveness.lose(failure);
    }
}

/// The answer to request `id` that failed for `message`.
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

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A decode batch of nothing, which the tests send and no rank runs.
    fn batch() -> Batch {
        Batch {
            id: ipc::execution::BatchId(1),
            phase: ipc::PhaseId(0),
            calls: Vec::new(),
            shape: ipc::execution::Shape::decode(1, ipc::execution::Variant::DecodeGreedy),
            columns: ipc::execution::BatchColumns::default(),
            generation: 0,
            cache_units: Vec::new(),
            block_tables: Vec::new(),
            page_loads: Vec::new(),
            page_offloads: Vec::new(),
            latent_restores: Vec::new(),
            host_outputs: Vec::new(),
            images: Vec::new(),
            frees: Vec::new(),
            media: None,
        }
    }

    /// A peer that registers with the scheduler node's listener is accepted under its rank; one
    /// of another epoch is refused by name.
    #[test]
    fn a_peer_registers_with_the_scheduler_node_of_its_epoch() {
        let listener = Listener::bind(&["127.0.0.1:0"]).expect("a listener");
        let address = listener.addresses()[0].clone();
        let dialling = std::thread::spawn(move || {
            register(&address, 4, 7, Duration::from_secs(10)).map(|_| ())
        });
        let accepted =
            accept(&listener, 1, 7, Duration::from_secs(10)).expect("the peer registers");
        assert_eq!(
            accepted.iter().map(|(rank, _)| *rank).collect::<Vec<_>>(),
            vec![4]
        );
        dialling
            .join()
            .expect("the dial")
            .expect("the registration is answered");

        let address = listener.addresses()[0].clone();
        let stranger = std::thread::spawn(move || {
            let _ = register(&address, 4, 8, Duration::from_secs(2));
        });
        let refused = accept(&listener, 1, 7, Duration::from_secs(10));
        assert!(
            refused.is_err(),
            "a registration of another epoch is refused"
        );
        stranger.join().expect("the stranger's dial");
    }

    /// A submission to a peer whose connection closes fails, and the close marks the scheduler
    /// node's peers lost.
    #[test]
    fn a_closed_peer_fails_its_submissions_and_is_lost() {
        let listener = Listener::bind(&["127.0.0.1:0"]).expect("a listener");
        let address = listener.addresses()[0].clone();
        let peer = std::thread::spawn(move || {
            let mut channel = register(&address, 1, 3, Duration::from_secs(10)).expect("registers");
            // Read the one submission, then go away without answering.
            loop {
                if channel
                    .try_recv::<MainProcessRequest>()
                    .expect("a request")
                    .is_some()
                {
                    break;
                }
                channel
                    .wait(Duration::from_secs(5))
                    .expect("the request arrives");
            }
            channel.shutdown();
        });
        let (rank, channel) = accept(&listener, 1, 3, Duration::from_secs(10))
            .expect("the peer registers")
            .pop()
            .expect("one peer");
        let waker = ipc::Waker::new().expect("a waker");
        let liveness = Liveness::new(waker.wake());
        let remote = Remote::spawn(rank, channel, Arc::clone(&liveness)).expect("the link");
        let submission = remote.submit(batch()).expect("the batch is sent");
        assert!(matches!(submission.wait(), Outcome::Failed(_)));
        peer.join().expect("the peer");
        let deadline = Instant::now() + Duration::from_secs(10);
        while liveness.lost().is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            liveness.lost().is_some(),
            "the close is the peer's death signal"
        );
        assert!(remote.submit(batch()).is_err());
    }
}
