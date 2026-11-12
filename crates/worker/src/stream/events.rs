//! Shared CUDA event leases and deferred storage release.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use platform::{DeviceId, Event};

use crate::{Error, Result, lock};

/// The identifier of an executor job: here, the job that waits for the events of deferred
/// releases, whose completion ([`EventPool::complete`]) runs them. The executor numbers its jobs
/// in submission order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(pub u64);

/// How long a host wait for device work may last where storage is given up outside the
/// executor's jobs: at shutdown, when no job is left to route a completion, and where a buffer
/// closes before it was announced ready.
pub const DRAIN_LIMIT: Duration = Duration::from_secs(60);

/// Waits on the host until `event`'s latest record completed, at most until `deadline`.
///
/// An event whose record is still queued in a job counts as incomplete, so a job that is never
/// performed ends this wait at the deadline instead of holding its thread. The driver has no
/// bounded wait on an event, so the bound is kept by querying the event every millisecond.
///
/// # Errors
///
/// [`Error::Resource`] when the deadline passes first, and the driver's error when the query
/// fails.
pub fn drain_event(event: &Event, deadline: Instant) -> Result<()> {
    while !event.query()? {
        if Instant::now() >= deadline {
            return Err(Error::Resource(
                "device work did not complete within the drain limit",
            ));
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}

/// A returned event is reused by a lease of the same device, timing flag and interprocess flag.
type PoolKey = (DeviceId, bool, bool);

/// The release of storage that device work addressed, with the notification of whoever waits for
/// it. It runs once every event it was deferred behind completed.
pub type Release = Box<dyn FnOnce() -> Result<()> + Send>;

/// Starts the executor job that completes once the events it is given completed, and returns the
/// job's identifier. The device is the one whose stream waits for the events.
///
/// It is called with the pool locked, so it does not use the pool.
pub type Submitter = Box<dyn Fn(DeviceId, Vec<Arc<Event>>) -> JobId + Send + Sync>;

/// A release deferred behind events.
struct Deferral {
    /// The events the release waits for: one retained reference of each.
    events: Vec<Arc<Event>>,
    release: Release,
}

/// An event out on lease.
struct ActiveEvent {
    /// Held while the lease is active, so its key, the event's address, names no other event.
    _event: Arc<Event>,
    device: DeviceId,
    /// The references storage owners hold on the lease; the event returns to the pool when the
    /// last one is released.
    references: usize,
}

struct State {
    /// Returned events, ready for another lease.
    available: HashMap<PoolKey, Vec<Arc<Event>>>,
    /// Leased events by address.
    active: HashMap<usize, ActiveEvent>,
    /// Deferred releases by the job whose completion runs them.
    deferred: HashMap<JobId, Vec<Deferral>>,
    /// Deferred releases whose job is still to be started: each was deferred on another thread
    /// than the submitter's, or before a submitter was installed.
    unsubmitted: Vec<Deferral>,
    /// What starts the jobs deferred releases wait for, and the thread that installed it: the
    /// one thread that submits jobs.
    submitter: Option<(ThreadId, Submitter)>,
}

/// Owns pooled events until their references and device accesses retire.
///
/// A storage owner records an event after the last device access to its storage, holds a
/// reference to it, and either releases the storage once it knows the event completed or defers
/// the release until it has ([`Self::defer`]).
///
/// While a rank serves, the pool asks the driver nothing. A deferred release waits for an
/// executor job that holds its events: the executor orders the job's completion behind the
/// events, and whoever takes the jobs' completions passes the pool the ones of its jobs
/// ([`Self::complete`]). The release runs there, on that thread, after the pool's lock is
/// released, so it may use the pool again. The job is always submitted behind the job that
/// records the events: a wait on an event that has no record yet passes.
///
/// A deferred release waits on events whose records were made or are queued in jobs already
/// submitted. On the thread that installed the [`Submitter`] its job is started at once; from
/// any other thread it is kept until shutdown.
///
/// At shutdown nobody takes completions: [`Self::drain`] waits for the events on the host, with
/// a deadline, and runs the releases kept.
///
/// The lock guards the pool's tables only, and no driver call is made under it: a driver call
/// that does not return stops only the thread that made it, while one that hung under this lock
/// would also stop the control thread, which takes the lock for every completion it passes on.
pub struct EventPool {
    state: Mutex<State>,
}

impl Default for EventPool {
    fn default() -> Self {
        Self {
            state: Mutex::new(State {
                available: HashMap::new(),
                active: HashMap::new(),
                deferred: HashMap::new(),
                unsubmitted: Vec::new(),
                submitter: None,
            }),
        }
    }
}

impl EventPool {
    /// An empty pool with no submitter.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // No owner's release runs under the lock: deferred releases run after unlock.
        lock(&self.state)
    }

    /// Installs or clears what starts the jobs deferred releases wait for. The calling thread
    /// becomes the one that starts them: the thread that submits the rank's jobs and takes their
    /// completions.
    pub fn set_submitter(&self, submitter: Option<Submitter>) {
        let submitter = submitter.map(|submitter| (std::thread::current().id(), submitter));
        let previous = std::mem::replace(&mut self.lock().submitter, submitter);
        drop(previous);
    }

    /// Defers `release` until every event of `events` completed, taking over one retained
    /// reference of each. The events' records were made, or are queued in jobs already
    /// submitted. With no events `release` runs now.
    ///
    /// # Errors
    ///
    /// [`Error::Invariant`] when an event is not a lease of this pool holding the references
    /// passed, or when the submitter's thread defers behind an event of no active lease; the
    /// failure of `release` when it ran now.
    pub fn defer(&self, events: Vec<Arc<Event>>, release: Release) -> Result<()> {
        let mut state = self.lock();
        if events.is_empty() {
            drop(state);
            return release();
        }
        require_references(&state, &events)?;

        let deferral = Deferral { events, release };

        let here = std::thread::current().id();
        if matches!(&state.submitter, Some((thread, _)) if *thread == here) {
            return submit_waiting(&mut state, deferral);
        }
        state.unsubmitted.push(deferral);
        Ok(())
    }

    /// Runs the deferred releases that waited for `job`, whose completion says that their events
    /// completed. Independent releases still run after one failed.
    ///
    /// # Errors
    ///
    /// [`Error::State`] when the job did not succeed: a failed job established nothing, and its
    /// releases stay deferred until shutdown. Otherwise the first failure of a release.
    pub fn complete(&self, job: JobId, succeeded: bool) -> Result<()> {
        if !succeeded {
            return Err(Error::State(
                "a failed job did not establish the completion of deferred work",
            ));
        }

        let releases: Vec<Release> = {
            let mut state = self.lock();
            let Some(deferred) = state.deferred.remove(&job) else {
                return Ok(());
            };
            for deferral in &deferred {
                if let Err(error) = require_references(&state, &deferral.events) {
                    state.deferred.insert(job, deferred);
                    return Err(error);
                }
            }
            for event in deferred.iter().flat_map(|deferral| &deferral.events) {
                release_reference(&mut state, event);
            }
            deferred
                .into_iter()
                .map(|deferral| deferral.release)
                .collect()
        };
        run(releases)
    }

    /// Waits on the host for the events of every deferred release, at most [`DRAIN_LIMIT`], and
    /// runs the releases, until none is left: at shutdown, when nobody takes the completions of
    /// the jobs they wait for.
    ///
    /// # Errors
    ///
    /// The first failure of a wait or of a release. A release whose events did not complete in
    /// time stays deferred.
    pub fn drain(&self) -> Result<()> {
        let deadline = Instant::now() + DRAIN_LIMIT;
        loop {
            let deferred = take_deferred(&mut self.lock());
            if deferred.is_empty() {
                return Ok(());
            }

            let mut failure = Ok(());
            let mut completed = Vec::new();
            for deferral in deferred {
                let waited = deferral
                    .events
                    .iter()
                    .try_for_each(|event| drain_event(event, deadline));

                let mut state = self.lock();
                match waited.and_then(|()| require_references(&state, &deferral.events)) {
                    Ok(()) => {
                        for event in &deferral.events {
                            release_reference(&mut state, event);
                        }
                        completed.push(deferral.release);
                    }
                    Err(error) => {
                        state.unsubmitted.push(deferral);
                        failure = failure.and(Err(error));
                    }
                }
            }
            failure.and(run(completed))?;
        }
    }
}

impl Drop for EventPool {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        let deferred = take_deferred(state);
        if deferred.is_empty() {
            return;
        }

        // A pool dropped undrained still owns storage whose release waits on device work.
        // Without a successful drain, bounded like every host wait for device work, the storage
        // cannot be returned.
        let deadline = Instant::now() + DRAIN_LIMIT;
        let drained = deferred.iter().all(|deferral| {
            deferral
                .events
                .iter()
                .all(|event| drain_event(event, deadline).is_ok())
        });
        if drained {
            for deferral in deferred {
                let _ = (deferral.release)();
            }
        } else {
            crate::leak(deferred);
        }
    }
}

/// Takes every deferred release out of `state`, started or not.
fn take_deferred(state: &mut State) -> Vec<Deferral> {
    let mut deferred = std::mem::take(&mut state.unsubmitted);
    deferred.extend(state.deferred.drain().flat_map(|(_, deferred)| deferred));
    deferred
}

/// The device whose stream waits for `events`: the first event's (a stream may wait for another
/// device's event).
fn device(state: &State, events: &[Arc<Event>]) -> Option<DeviceId> {
    events
        .first()
        .and_then(|event| state.active.get(&key(event)))
        .map(|active| active.device)
}

/// Starts, through the installed submitter, the job `deferral` waits for; its release runs once
/// that job completed. A deferral whose job could not be started is kept until shutdown.
fn submit_waiting(state: &mut State, deferral: Deferral) -> Result<()> {
    let Some((_, submitter)) = &state.submitter else {
        state.unsubmitted.push(deferral);
        return Err(invariant("deferred work has no submitter"));
    };

    match device(state, &deferral.events) {
        Some(device) => {
            let job = submitter(device, deferral.events.clone());
            state.deferred.entry(job).or_default().push(deferral);
            Ok(())
        }
        None => {
            state.unsubmitted.push(deferral);
            Err(invariant("deferred work waits on no active device event"))
        }
    }
}

/// Runs every release and returns the first failure.
fn run(releases: impl IntoIterator<Item = Release>) -> Result<()> {
    let mut result = Ok(());
    for release in releases {
        if let Err(error) = release()
            && result.is_ok()
        {
            result = Err(error);
        }
    }
    result
}

/// Checks that every event of `events` is an active lease holding at least as many references
/// as `events` names it.
fn require_references(state: &State, events: &[Arc<Event>]) -> Result<()> {
    let mut counts = HashMap::new();
    for event in events {
        *counts.entry(key(event)).or_insert(0_usize) += 1;
    }

    for (key, count) in counts {
        if state
            .active
            .get(&key)
            .is_none_or(|active| active.references < count)
        {
            return Err(invariant("deferred device event has invalid ownership"));
        }
    }
    Ok(())
}

/// Releases one reference of `event`'s lease, and returns the event to the pool with the last.
fn release_reference(state: &mut State, event: &Arc<Event>) {
    if let Some(active) = state.active.get_mut(&key(event)) {
        active.references -= 1;
        if active.references == 0 {
            let device = active.device;
            let flags = event.flags();
            state.active.remove(&key(event));
            state
                .available
                .entry((device, flags.timing, flags.interprocess))
                .or_default()
                .push(Arc::clone(event));
        }
    }
}

/// The key of an event's lease: its address.
fn key(event: &Arc<Event>) -> usize {
    Arc::as_ptr(event) as usize
}

fn invariant(message: &str) -> Error {
    Error::Invariant(message.into())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    use super::*;

    /// A pool whose waiting jobs are started by the calling thread: the submitter numbers them
    /// as the executor would.
    fn pool() -> Arc<EventPool> {
        let events = EventPool::new();
        let next = AtomicU64::new(0);
        events.set_submitter(Some(Box::new(move |_device, _events| {
            JobId(next.fetch_add(1, Ordering::Relaxed))
        })));
        events
    }

    /// A release that counts its runs.
    fn counting(runs: &Arc<AtomicUsize>) -> Release {
        let runs = Arc::clone(runs);
        Box::new(move || {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
    }

    #[test]
    fn a_release_with_no_event_to_wait_for_runs_at_once() -> Result<()> {
        let events = pool();
        let runs = Arc::new(AtomicUsize::new(0));
        events.defer(Vec::new(), counting(&runs))?;
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn a_job_no_release_waits_for_completes_with_nothing_to_run() -> Result<()> {
        let events = pool();
        events.complete(JobId(7), true)?;
        assert!(events.complete(JobId(7), false).is_err());
        events.drain()
    }
}
