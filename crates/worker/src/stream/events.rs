//! Shared CUDA event leases and deferred storage release.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use platform::{DeviceId, Event};

use crate::{Error, Result, lock};

/// The worker's name for the job that waits for deferred work's events, and
/// whose completion (`complete`) hands the work over. The worker thread
/// allocates them in submission order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(pub u64);

/// How long a host wait for device work may last where storage is given up
/// outside the worker's jobs: at shutdown, when no job is left to route a
/// completion, and where a buffer closes before it was announced ready.
pub const DRAIN_LIMIT: Duration = Duration::from_secs(60);

/// Wait on the host until `event`'s latest record completed, at most until
/// `deadline`. An event whose record is still queued in a job counts as
/// incomplete, so a job that is never performed ends this wait at the deadline
/// instead of holding its thread.
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

type PoolKey = (DeviceId, bool, bool);

/// Work that runs once every event it waits on has completed: the release
/// of storage that device work addressed, and its completion observers.
pub type Deferred = Box<dyn FnOnce() -> Result<()> + Send>;

/// Deferred work on its way to the pool: a job carries it from the call
/// that recorded its events to the job's submission ([`Issue::defer`]).
pub struct Deferral {
    /// The events `owner` waits on, one retained reference of each.
    pub(crate) events: Vec<Arc<Event>>,
    pub(crate) owner: Deferred,
}

/// Starts the job that completes once the events it is given completed
/// ([`crate::job::JobQueue::after`]) and names it. The device is the one whose
/// notification stream waits for the events. It is called with the pool locked,
/// so it does not use the pool.
pub type Submitter = Box<dyn Fn(DeviceId, Vec<Arc<Event>>) -> JobId + Send + Sync>;

#[derive(PartialEq)]
enum Status {
    Open,
    Closed,
}

struct ActiveEvent {
    /// Held while the lease is active, so its key, the event's address, names no other event.
    _event: Arc<Event>,
    device: DeviceId,
    references: usize,
}

struct State {
    available: HashMap<PoolKey, Vec<Arc<Event>>>,
    active: HashMap<usize, ActiveEvent>,
    /// Deferred work by the job whose completion runs it.
    deferred: HashMap<JobId, Vec<Deferral>>,
    /// Deferred work whose job is still to be started: it was deferred on
    /// another thread than the submitter's, or before a submitter was
    /// installed.
    unsubmitted: Vec<Deferral>,
    status: Status,
    /// What starts the jobs of deferred work, and the thread that installed
    /// it: the one thread that submits jobs.
    submitter: Option<(ThreadId, Submitter)>,
}

/// Owns pooled events until their references and device accesses retire.
///
/// Storage owners record a fence after the last device access to their
/// storage, retain a reference to it, and either release it once they
/// know it completed or defer the storage's release until it has
/// ([`Issue::defer`], `defer`).
///
/// The pool asks the driver nothing. Deferred work waits for a job that holds
/// its events (`crate::job::JobQueue::after`): the builder thread orders the
/// job's completion behind the events, and whoever takes the jobs' completions
/// hands the pool the ones of its jobs (`owns`, `complete`). The work runs
/// there, on that thread, after the pool's lock is released, so it may use the
/// pool again. The job is always submitted behind the job that records the
/// events: a wait on an event that has no record yet passes.
///
/// Deferred work waits on events whose records were made or are queued in
/// jobs already submitted. On the thread that installed the `Submitter` its
/// job is started at once; from any other thread it is kept until shutdown.
///
/// At shutdown nobody takes completions: [`drain`](Self::drain) waits for the
/// events on the host, with a deadline, and runs the work kept.
///
/// The lock guards bookkeeping only. A record, which calls the driver or
/// appends to a job, is booked under the lock and made outside it. This is
/// the design's failure handling (section 10), not a matter of speed: a
/// driver call that does not return stops only the thread that made it,
/// and a `cuEventRecord` that hung under this lock would also stop the
/// control thread, which takes it for every fence and every completion it
/// hands over.
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
                status: Status::Open,
                submitter: None,
            }),
        }
    }
}

impl EventPool {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // No owner's work runs under the lock: deferred work runs after
        // unlock.
        lock(&self.state)
    }

    /// Install or clear what starts the jobs of deferred work. The calling
    /// thread becomes the one that starts them: the thread that submits the
    /// rank's jobs and takes their completions.
    pub(crate) fn set_submitter(&self, submitter: Option<Submitter>) {
        let submitter = submitter.map(|submitter| (std::thread::current().id(), submitter));
        let previous = std::mem::replace(&mut self.lock().submitter, submitter);
        drop(previous);
    }

    /// Transfers one retained reference of each event, and the release work
    /// `owner`, to deferred release, outside the recording of any job: the
    /// events' records were made, or are queued in jobs already submitted.
    /// `owner` runs once every event completed. With no events, or after
    /// shutdown drained the pool, it runs now.
    ///
    /// Work deferred while a job is recorded goes through the job's
    /// recorder instead ([`Issue::defer`]).
    pub(crate) fn defer(&self, events: Vec<Arc<Event>>, owner: Deferred) -> Result<()> {
        let mut state = self.lock();
        if state.status == Status::Closed || events.is_empty() {
            drop(state);
            return owner();
        }
        require_references(&state, &events)?;
        let deferral = Deferral { events, owner };

        let here = std::thread::current().id();
        if matches!(&state.submitter, Some((thread, _)) if *thread == here) {
            return submit_waiting(&mut state, deferral);
        }
        state.unsubmitted.push(deferral);
        Ok(())
    }

    /// Whether `job` is one that deferred work waits for.
    #[cfg(test)]
    pub(crate) fn owns(&self, job: JobId) -> bool {
        self.lock().deferred.contains_key(&job)
    }

    /// Runs the deferred work that waited for `job`, whose completion says
    /// that the work's events completed, and returns the first failure;
    /// independent owners still run after one failed. A job that failed
    /// established nothing: its work stays deferred until shutdown.
    pub(crate) fn complete(&self, job: JobId, succeeded: bool) -> Result<()> {
        if !succeeded {
            return Err(Error::State(
                "a failed job did not establish the completion of deferred work",
            ));
        }
        let owners: Vec<Deferred> = {
            let mut state = self.lock();
            let Some(deferrals) = state.deferred.remove(&job) else {
                return Ok(());
            };
            for deferral in &deferrals {
                if let Err(error) = require_references(&state, &deferral.events) {
                    state.deferred.insert(job, deferrals);
                    return Err(error);
                }
            }
            for event in deferrals.iter().flat_map(|deferral| &deferral.events) {
                release_ready(&mut state, event, 1);
            }
            deferrals
                .into_iter()
                .map(|deferral| deferral.owner)
                .collect()
        };
        run(owners)
    }

    /// Waits on the host for the events of all deferred work, at most
    /// `DRAIN_LIMIT`, and runs that work, until none is left: at
    /// shutdown, when nobody takes the completions of the jobs it waits
    /// for. Work whose events did not complete in time stays deferred.
    pub fn drain(&self) -> Result<()> {
        let deadline = Instant::now() + DRAIN_LIMIT;
        loop {
            let deferrals: Vec<Deferral> = {
                let mut state = self.lock();
                let mut deferrals = std::mem::take(&mut state.unsubmitted);
                deferrals.extend(state.deferred.drain().flat_map(|(_, deferrals)| deferrals));
                deferrals
            };
            if deferrals.is_empty() {
                return Ok(());
            }

            let mut failure = Ok(());
            let mut completed = Vec::new();
            for deferral in deferrals {
                let waited = deferral
                    .events
                    .iter()
                    .try_for_each(|event| drain_event(event, deadline));
                let mut state = self.lock();
                match waited.and_then(|()| require_references(&state, &deferral.events)) {
                    Ok(()) => {
                        for event in &deferral.events {
                            release_ready(&mut state, event, 1);
                        }
                        completed.push(deferral.owner);
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
        let mut deferrals = std::mem::take(&mut state.unsubmitted);
        deferrals.extend(state.deferred.drain().flat_map(|(_, deferrals)| deferrals));
        if deferrals.is_empty() {
            return;
        }
        // An unclosed pool still owns storage whose release waits on device
        // work. Without a successful drain, bounded like every host wait
        // for device work, it cannot be returned.
        let deadline = Instant::now() + DRAIN_LIMIT;
        let drained = deferrals.iter().all(|deferral| {
            deferral
                .events
                .iter()
                .all(|event| drain_event(event, deadline).is_ok())
        });
        if drained {
            for deferral in deferrals {
                let _ = (deferral.owner)();
            }
        } else {
            crate::leak(deferrals);
        }
    }
}

/// The device whose notification stream waits for `events`: the first
/// event's (a stream may wait for another device's event).
fn device(state: &State, events: &[Arc<Event>]) -> Option<DeviceId> {
    events
        .first()
        .and_then(|event| state.active.get(&key(event)))
        .map(|active| active.device)
}

/// Start, through the installed submitter, the job one deferral made
/// outside a recording waits for; its work runs once that job completed.
fn submit_waiting(state: &mut State, deferral: Deferral) -> Result<()> {
    let Some((_, submitter)) = &state.submitter else {
        state.unsubmitted.push(deferral);
        return Err(invariant("deferred work has no submitter"));
    };
    let job = match device(state, &deferral.events) {
        Some(device) => Ok(submitter(device, deferral.events.clone())),
        None => Err(invariant("deferred work waits on no active device event")),
    };
    match job {
        Ok(job) => {
            state.deferred.entry(job).or_default().push(deferral);
            Ok(())
        }
        Err(error) => {
            state.unsubmitted.push(deferral);
            Err(error)
        }
    }
}

/// Run every owner and return the first failure.
fn run(owners: impl IntoIterator<Item = Deferred>) -> Result<()> {
    let mut result = Ok(());
    for owner in owners {
        if let Err(error) = owner()
            && result.is_ok()
        {
            result = Err(error);
        }
    }
    result
}

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

fn release_ready(state: &mut State, event: &Arc<Event>, count: usize) {
    if let Some(active) = state.active.get_mut(&key(event)) {
        active.references -= count;
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

    /// A pool whose waiting jobs are started by the calling thread: the
    /// submitter numbers them and keeps each job's id, so a test hands a
    /// completion over exactly as the worker thread would.
    fn served() -> (Arc<EventPool>, Arc<AtomicU64>) {
        let events = EventPool::new();
        let next = Arc::new(AtomicU64::new(0));
        let counter = Arc::clone(&next);
        events.set_submitter(Some(Box::new(move |_device, _events| {
            JobId(counter.fetch_add(1, Ordering::Relaxed))
        })));
        (events, next)
    }

    /// Deferred work that counts its runs.
    fn counting(runs: &Arc<AtomicUsize>) -> Deferred {
        let runs = Arc::clone(runs);
        Box::new(move || {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
    }

    #[test]
    fn work_with_no_event_to_wait_for_runs_at_once() -> Result<()> {
        let (events, _) = served();
        let runs = Arc::new(AtomicUsize::new(0));
        events.defer(Vec::new(), counting(&runs))?;
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn a_job_that_waits_for_no_deferred_work_is_not_owned() -> Result<()> {
        let (events, _) = served();
        assert!(!events.owns(JobId(7)));
        events.complete(JobId(7), true)?;
        Ok(())
    }
}
