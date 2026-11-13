//! How an operation of the worker ended, the completion its waiters share, and the submission of
//! a job to the executors.
//!
//! A host task of a [`ThreadPool`](crate::thread_pool::ThreadPool) ends with an [`Outcome`] and
//! is waited for through a [`Completion`]. A job submitted to an executor, or to every rank of a
//! worker group, is waited for through a [`Submission`], which completes with what the job
//! returns: the batch's output, and the requests the executors hand back for the buffer pool's
//! submission queue (the requests a stamp applied, each with the group's stamp, and the `Finish`
//! of each copy a batch ran).
//!
//! Nothing here waits by polling: a waiter sleeps on a condition variable, and a [`Waker`] is
//! called once an outcome lands.

use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ipc::batch::BatchOutput;
use storage::runtime::{GroupId, Request};

use crate::{Error, lock};

/// A failure a [`Submission`] reports: the worker's own [`Error`] in every submission this crate
/// completes, which a reader may downcast to; a peer's failure arrives as one too.
pub type Fault = dyn std::error::Error + Send + Sync;

/// How an operation of the worker ended.
///
/// Storage a device addressed is released only on a success: a failure or a cancellation does
/// not establish that the device stopped addressing it. `E` is the failure's type: the worker's
/// own error for a host task, a [`Fault`] for a submission.
#[derive(Debug)]
pub enum Outcome<T = (), E: ?Sized = Error> {
    /// The operation completed with its value.
    Success(T),
    /// The operation failed.
    Failed(Arc<E>),
    /// The operation was cancelled before it completed.
    Cancelled,
}

impl<T: Clone, E: ?Sized> Clone for Outcome<T, E> {
    fn clone(&self) -> Self {
        match self {
            Self::Success(value) => Self::Success(value.clone()),
            Self::Failed(error) => Self::Failed(Arc::clone(error)),
            Self::Cancelled => Self::Cancelled,
        }
    }
}

/// One completion shared by an operation and its waiters: its outcome, recorded once, the
/// waiters it wakes and the callbacks it returns for dispatch.
///
/// `C` is the callback type (such as a [`Waker`]), `T` the operation's value and `E` its failure.
/// Completing an operation returns its callbacks, so its owner dispatches them after it released
/// its own lock.
pub struct Completion<C, T = (), E: ?Sized = Error> {
    state: Mutex<CompletionState<C, T, E>>,
    /// Signalled when the outcome lands, so a waiter sleeps instead of polling.
    ready: Condvar,
}

struct CompletionState<C, T, E: ?Sized> {
    outcome: Option<Outcome<T, E>>,
    callbacks: Vec<C>,
}

/// A completion that already has its outcome: the second outcome is not recorded.
#[derive(Debug, thiserror::Error)]
#[error("operation has already completed")]
pub struct Settled;

impl<C, T, E: ?Sized> Default for Completion<C, T, E> {
    fn default() -> Self {
        Self {
            state: Mutex::new(CompletionState {
                outcome: None,
                callbacks: Vec::new(),
            }),
            ready: Condvar::new(),
        }
    }
}

impl<C, T: Clone, E: ?Sized> Completion<C, T, E> {
    fn state(&self) -> MutexGuard<'_, CompletionState<C, T, E>> {
        // No code mutates the state partially under the lock, so a panicking holder cannot leave
        // an operation half completed.
        lock(&self.state)
    }

    /// Whether the operation ended.
    pub fn done(&self) -> bool {
        self.state().outcome.is_some()
    }

    /// Whether the operation ended successfully.
    pub fn succeeded(&self) -> bool {
        matches!(self.state().outcome, Some(Outcome::Success(_)))
    }

    /// The outcome, once the operation ended.
    pub fn outcome(&self) -> Option<Outcome<T, E>> {
        self.state().outcome.clone()
    }

    /// Waits for the operation's outcome.
    pub fn wait(&self) -> Outcome<T, E> {
        let mut state = self.state();
        loop {
            if let Some(outcome) = &state.outcome {
                return outcome.clone();
            }
            state = self
                .ready
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Waits at most `timeout` for the outcome; `Duration::ZERO` is a look.
    pub fn wait_timeout(&self, timeout: Duration) -> Option<Outcome<T, E>> {
        let deadline = Instant::now() + timeout;
        let mut state = self.state();
        while state.outcome.is_none() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            state = self
                .ready
                .wait_timeout(state, left)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        state.outcome.clone()
    }

    /// Registers a callback, or returns it for immediate dispatch when the operation ended.
    pub fn subscribe(&self, callback: C) -> Option<C> {
        let mut state = self.state();
        if state.outcome.is_some() {
            return Some(callback);
        }
        state.callbacks.push(callback);
        None
    }

    /// Records `outcome` when the operation has none yet, wakes every waiter, and returns the
    /// callbacks to run; `None` when an outcome was already recorded, which this one does not
    /// replace.
    pub fn settle(&self, outcome: Outcome<T, E>) -> Option<Vec<C>> {
        let mut state = self.state();
        if state.outcome.is_some() {
            return None;
        }
        state.outcome = Some(outcome);
        let callbacks = std::mem::take(&mut state.callbacks);
        drop(state);
        self.ready.notify_all();
        Some(callbacks)
    }

    /// Cancels an unresolved completion, returning whether it is now cancelled and the callbacks
    /// to run. This does not stop its producer.
    pub fn cancel(&self) -> (bool, Vec<C>) {
        let mut state = self.state();
        match state.outcome {
            Some(Outcome::Cancelled) => return (true, Vec::new()),
            Some(_) => return (false, Vec::new()),
            None => {}
        }
        state.outcome = Some(Outcome::Cancelled);
        let callbacks = std::mem::take(&mut state.callbacks);
        drop(state);
        self.ready.notify_all();
        (true, callbacks)
    }
}

impl<C, T: Clone, E> Completion<C, T, E> {
    /// Records the operation's result, as [`Completion::settle`] does.
    ///
    /// # Errors
    ///
    /// [`Settled`] when the operation already has an outcome.
    pub fn complete(&self, result: Result<T, E>) -> Result<Vec<C>, Settled> {
        let outcome = match result {
            Ok(value) => Outcome::Success(value),
            Err(error) => Outcome::Failed(Arc::new(error)),
        };
        self.settle(outcome).ok_or(Settled)
    }
}

/// What a completion calls once its outcome lands ([`Submission::on_settled`]).
///
/// It runs on the thread that completed the operation (a device's executor thread, a peer
/// link's thread), so it must only signal: wake a waiter, never do the waiter's work.
pub type Waker = Arc<dyn Fn() + Send + Sync>;

/// A submission's number within the queue or the peer link that made it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SubmissionId(pub u64);

/// What the caller of a job holds until the job ended, on one rank or on every rank of a worker
/// group.
///
/// A job's submission completes once its executor has performed the job's work and the device
/// has completed it (the executor waits on its stream's work before it posts a job), with what
/// the job returns: a batch's [`BatchOutput`] and the `Finish` of each copy it ran; a stamp's
/// requests, each stamped; nothing for any other job. A group's submission joins the submissions
/// of its ranks ([`Submission::joined`]): it completes when every rank's did.
pub struct Submission {
    id: SubmissionId,
    state: Arc<SubmissionState>,
    /// The same job on the other ranks of the group: the submission completes when they all did.
    peers: Vec<Submission>,
}

/// The completion state of one submission, which the executor's queue or a peer link completes.
pub struct SubmissionState {
    /// The submission's own outcome, which `Completion` records once and waits on, with the
    /// wakers [`Submission::on_settled`] registered.
    completion: Completion<Waker, (BatchOutput, Vec<Request>), Fault>,
    /// What the job was doing when it failed, written where the failure happened: the caller
    /// cannot know it, and an error alone does not say which bucket or operator broke.
    failure: Mutex<Option<String>>,
}

impl Default for SubmissionState {
    fn default() -> Self {
        Self {
            completion: Completion::default(),
            failure: Mutex::new(None),
        }
    }
}

impl Submission {
    /// A submission waiting for its job's completion, with the state the executor's queue or a
    /// peer link completes.
    pub fn pending(id: SubmissionId) -> (Self, Arc<SubmissionState>) {
        let state = Arc::new(SubmissionState::default());
        (
            Self {
                id,
                state: Arc::clone(&state),
                peers: Vec::new(),
            },
            state,
        )
    }

    /// This submission and the same job's on `peers`, the other ranks of its group, as one.
    ///
    /// The ranks of a group run one job together, so it has ended when every one of them ended.
    /// The joined outcome is this submission's, with two exceptions: a peer's failure replaces it,
    /// because a rank that failed leaves the others' output without its part; and the requests
    /// the ranks return are joined, one per request, each with its stamps joined (the largest
    /// stamp each group states, the group's stamp being the largest over its ranks) and, for a
    /// `Finish`, the copy failed when it failed on any rank. Ranks that return different
    /// requests fail the joined outcome.
    #[must_use]
    pub fn joined(mut self, peers: Vec<Submission>) -> Self {
        self.peers = peers;
        self
    }

    /// The submission's number.
    pub fn id(&self) -> SubmissionId {
        self.id
    }

    /// Waits until the submission completed and returns its outcome.
    pub fn wait(&self) -> Outcome<(BatchOutput, Vec<Request>), Fault> {
        let own = self.state.completion.wait();
        let peers = self.peers.iter().map(Self::wait).collect();
        joined_outcome(own, peers)
    }

    /// Waits at most `timeout` for the submission's completion.
    pub fn wait_timeout(
        &self,
        timeout: Duration,
    ) -> Option<Outcome<(BatchOutput, Vec<Request>), Fault>> {
        let deadline = Instant::now() + timeout;
        let own = self
            .state
            .completion
            .wait_timeout(deadline.saturating_duration_since(Instant::now()))?;

        let mut peers = Vec::with_capacity(self.peers.len());
        for peer in &self.peers {
            peers.push(peer.wait_timeout(deadline.saturating_duration_since(Instant::now()))?);
        }
        Some(joined_outcome(own, peers))
    }

    /// Calls `wake` when this rank's part or any peer's part settles, at once for a part that
    /// already has: a caller holding several submissions waits on one signal of its own and reads
    /// [`Submission::outcome`] of each when it fires, instead of polling them.
    ///
    /// `wake` may be called once per part, so a waiter checks the outcome again rather than
    /// counting calls.
    pub fn on_settled(&self, wake: &Waker) {
        if let Some(wake) = self.state.completion.subscribe(Arc::clone(wake)) {
            wake();
        }
        for peer in &self.peers {
            peer.on_settled(wake);
        }
    }

    /// The submission's outcome, once every part completed.
    pub fn outcome(&self) -> Option<Outcome<(BatchOutput, Vec<Request>), Fault>> {
        let own = self.state.completion.outcome()?;
        let mut peers = Vec::with_capacity(self.peers.len());
        for peer in &self.peers {
            peers.push(peer.outcome()?);
        }
        Some(joined_outcome(own, peers))
    }

    /// What the submission was doing when it failed, when it did.
    ///
    /// The executor thread writes it where the failure happens (the job it was performing and
    /// the bucket, generation or operator that named it), because the caller holds only the
    /// bare error.
    pub fn failure_label(&self) -> Option<String> {
        lock(&self.state.failure)
            .clone()
            .or_else(|| self.peers.iter().find_map(Self::failure_label))
    }
}

impl SubmissionState {
    /// Records what this submission was doing when it failed, for the report that reads it.
    pub fn note_failure(&self, label: String) {
        let mut slot = lock(&self.failure);
        if slot.is_none() {
            *slot = Some(label);
        }
    }

    /// Completes the submission with `outcome`; the first completion stands, and a later one (a
    /// cancellation racing a completion) is dropped.
    pub fn finish(self: Arc<Self>, outcome: Outcome<(BatchOutput, Vec<Request>), Fault>) {
        for wake in self.completion.settle(outcome).unwrap_or_default() {
            wake();
        }
    }
}

/// The outcome of a job on every rank of its group, from this rank's (`own`) and the peers'
/// ([`Submission::joined`]).
fn joined_outcome(
    own: Outcome<(BatchOutput, Vec<Request>), Fault>,
    peers: Vec<Outcome<(BatchOutput, Vec<Request>), Fault>>,
) -> Outcome<(BatchOutput, Vec<Request>), Fault> {
    let Outcome::Success((output, mut requests)) = own else {
        return own;
    };

    for peer in peers {
        match peer {
            Outcome::Success((_, returned)) => {
                if let Err(error) = join_requests(&mut requests, returned) {
                    let fault: Arc<Fault> = Arc::new(error);
                    return Outcome::Failed(fault);
                }
            }
            failed => return failed,
        }
    }
    Outcome::Success((output, requests))
}

/// Joins `other`, the requests another rank of the group returned for the same job, into
/// `requests`: the same requests in the same order, of which a stamp is the largest either states
/// for its group, and a copy's `Finish` failed when either failed.
fn join_requests(requests: &mut [Request], other: Vec<Request>) -> crate::Result<()> {
    if requests.len() != other.len() {
        return Err(Error::Invariant(format!(
            "the ranks of a group returned {} and {} requests for one job",
            requests.len(),
            other.len()
        )));
    }

    for (request, other) in requests.iter_mut().zip(other) {
        if unstamped(request) != unstamped(&other) {
            return Err(Error::Invariant(format!(
                "the ranks of a group returned {request:?} and {other:?} for one job"
            )));
        }

        if let (Request::Finish { ok, .. }, Request::Finish { ok: theirs, .. }) =
            (&mut *request, &other)
        {
            *ok &= *theirs;
        }
        for &(group, sequence) in stamps(&other).into_iter().flatten() {
            add_stamp(request, group, sequence);
        }
    }
    Ok(())
}

/// `request` without its stamps and, for a `Finish`, its result: what two ranks return alike for
/// one request.
fn unstamped(request: &Request) -> Request {
    let mut request = request.clone();
    if let Some(after) = stamps_mut(&mut request) {
        after.clear();
    }
    if let Request::Finish { ok, .. } = &mut request {
        *ok = true;
    }
    request
}

/// The stamps of a request the executors stamp.
fn stamps(request: &Request) -> Option<&storage::runtime::Stamps> {
    match request {
        Request::Detach { after, .. }
        | Request::Release { after, .. }
        | Request::Commit { after, .. }
        | Request::Transfer { after, .. }
        | Request::Finish { after, .. } => Some(after),
        _ => None,
    }
}

/// Adds `sequence` to the stamps of `request`, a request the executors stamp, as the stamp of
/// `group`: the larger of it and the stamp the request states for the group already. A request
/// the executors do not stamp is left as it is.
pub(crate) fn add_stamp(request: &mut Request, group: GroupId, sequence: u64) {
    let Some(after) = stamps_mut(request) else {
        return;
    };

    match after.iter_mut().find(|(stamped, _)| *stamped == group) {
        Some((_, stamp)) => *stamp = (*stamp).max(sequence),
        None => after.push((group, sequence)),
    }
}

/// The stamps of a request the executors stamp, to add to.
fn stamps_mut(request: &mut Request) -> Option<&mut storage::runtime::Stamps> {
    match request {
        Request::Detach { after, .. }
        | Request::Release { after, .. }
        | Request::Commit { after, .. }
        | Request::Transfer { after, .. }
        | Request::Finish { after, .. } => Some(after),
        _ => None,
    }
}

impl std::fmt::Debug for Submission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Submission")
            .field("id", &self.id)
            .field("completed", &self.outcome().is_some())
            .finish()
    }
}

impl std::fmt::Debug for SubmissionState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SubmissionState")
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use storage::runtime::{OwnerId, RequestId};

    use super::*;
    use crate::Result;

    #[test]
    fn a_subscriber_racing_completion_is_notified_exactly_once() -> Result<()> {
        for _ in 0..64 {
            let completion = Completion::<Waker>::default();
            let calls = Arc::new(AtomicUsize::new(0));
            let observer: Waker = {
                let calls = Arc::clone(&calls);
                Arc::new(move || {
                    calls.fetch_add(1, Ordering::Relaxed);
                })
            };

            thread::scope(|scope| -> Result<()> {
                let subscriber = scope.spawn(|| {
                    if let Some(observer) = completion.subscribe(observer) {
                        observer();
                    }
                });
                for observer in completion.complete(Ok(()))? {
                    observer();
                }
                subscriber
                    .join()
                    .map_err(|_| Error::State("subscriber panicked"))
            })?;

            assert_eq!(calls.load(Ordering::Relaxed), 1);
            assert!(completion.succeeded());
            assert!(!completion.cancel().0);
            assert!(completion.complete(Ok(())).is_err());
        }
        Ok(())
    }

    #[test]
    fn a_failure_and_a_cancellation_are_done_and_not_succeeded() -> Result<()> {
        let failed = Completion::<Waker>::default();
        assert!(!failed.done());
        failed.complete(Err(Error::transport("device completion unknown")))?;
        assert!(failed.done());
        assert!(!failed.succeeded());
        let Some(Outcome::Failed(error)) = failed.outcome() else {
            return Err(Error::State("the producer error was lost"));
        };
        assert!(error.to_string().contains("device completion unknown"));

        let cancelled = Completion::<Waker>::default();
        assert!(cancelled.cancel().0);
        assert!(cancelled.done());
        assert!(!cancelled.succeeded());
        assert!(matches!(cancelled.outcome(), Some(Outcome::Cancelled)));
        assert!(cancelled.cancel().0);
        assert!(cancelled.complete(Ok(())).is_err());
        Ok(())
    }

    /// A submission of one rank that completed with `requests`.
    fn completed(requests: Vec<Request>) -> Submission {
        let (submission, state) = Submission::pending(SubmissionId(0));
        state.finish(Outcome::Success((BatchOutput::default(), requests)));
        submission
    }

    #[test]
    fn a_group_s_requests_carry_the_largest_stamp_of_its_ranks_and_a_copy_failed_on_any() {
        let release = |stamp| Request::Release {
            owner: OwnerId(1),
            class: 0,
            at: 2,
            count: 1,
            after: vec![(GroupId(0), 9), (GroupId(3), stamp)],
        };
        let finish = |ok, stamp| Request::Finish {
            copy: RequestId(4),
            ok,
            after: vec![(GroupId(3), stamp)],
        };

        let joined = completed(vec![release(5), finish(true, 5)]).joined(vec![
            completed(vec![release(7), finish(false, 7)]),
            completed(vec![release(6), finish(true, 6)]),
        ]);
        let Some(Outcome::Success((_, requests))) = joined.outcome() else {
            panic!("the joined submission completed");
        };
        assert_eq!(requests, [release(7), finish(false, 7)]);

        // Ranks that return other requests for one job leave the group's answer unknown.
        let diverged = completed(vec![release(5)]).joined(vec![completed(vec![])]);
        assert!(matches!(diverged.outcome(), Some(Outcome::Failed(_))));
    }
}
