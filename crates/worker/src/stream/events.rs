//! Shared CUDA event leases and deferred storage release.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;
use std::time::Instant;

use uniserve_cuda::{DeviceId, Event, EventFlags};
use uniserve_ipc::Wake;

use super::Issue;
use crate::lane::{Completion, JobId};
use crate::transfer::{DRAIN_LIMIT, drain_event};
use crate::{Error, Result};

type PoolKey = (i32, bool, bool);

/// Work that runs once every event it waits on has completed: the release
/// of storage that device work addressed, and its completion observers.
pub type Deferred = Box<dyn FnOnce() -> Result<()> + Send>;

/// Deferred work on its way to the pool: a job carries it from the call
/// that recorded its events to the job's submission ([`Issue::defer`]).
pub(crate) struct Deferral {
    /// The events `owner` waits on, one retained reference of each.
    pub(crate) events: Vec<Arc<Event>>,
    pub(crate) owner: Deferred,
}

/// Starts the job that completes once the events it is given completed
/// ([`crate::lane::JobQueues::after`]) and names it: on `lane` when the
/// events' records are queued there, on any lane otherwise. The device is
/// the one whose notification stream waits for the events. It is called
/// with the pool locked, so it does not use the pool.
pub type Submitter =
    Box<dyn Fn(DeviceId, Option<usize>, Vec<Arc<Event>>) -> Result<JobId> + Send + Sync>;

/// What starts the job of deferred work a submitted job carried.
type Start<'a> = dyn Fn(DeviceId, Vec<Arc<Event>>) -> Result<JobId> + 'a;

#[derive(PartialEq)]
enum Status {
    Open,
    Closing,
    Closed,
}

struct ActiveEvent {
    event: Arc<Event>,
    device: i32,
    references: usize,
    /// Whether the lease was recorded, at once or into a job.
    recorded: bool,
    /// The lane and job whose queued record of the lease is not made yet.
    queued: Option<(usize, JobId)>,
}

struct State {
    available: HashMap<PoolKey, Vec<Arc<Event>>>,
    active: HashMap<usize, ActiveEvent>,
    /// Deferred work by the job whose completion runs it.
    deferred: HashMap<JobId, Vec<Deferral>>,
    /// The leases whose records each submitted job queued, until it
    /// completed.
    queued: HashMap<JobId, Vec<usize>>,
    /// Deferred work whose job is still to be started: it was deferred on
    /// another thread than the submitter's, or before a submitter was
    /// installed.
    unsubmitted: Vec<Deferral>,
    status: Status,
    wake: Option<Wake>,
    /// What starts the jobs of deferred work, and the thread that installed
    /// it: the one thread that submits jobs.
    submitter: Option<(ThreadId, Submitter)>,
}

/// Owns pooled events until their references and device accesses retire.
///
/// Storage owners record a fence after the last device access to their
/// storage, retain a reference to it, and either release it once they
/// know it completed or defer the storage's release until it has
/// ([`Issue::defer`], [`defer`](Self::defer)).
///
/// The pool asks the driver nothing. Deferred work waits for a job that
/// holds its events ([`crate::lane::JobQueues::after`]): a lane orders the
/// job's completion behind the events, and whoever takes the lanes'
/// completions hands the pool the ones of its jobs
/// ([`owns`](Self::owns), [`complete`](Self::complete)). The work runs
/// there, on that thread, after the pool's lock is released, so it may use
/// the pool again. The job is always submitted behind the job that records
/// the events, on the same lane: a lane passes a wait on an event that has
/// no record yet.
///
/// - Work deferred while a job is recorded rides with that job, and the
///   job's submitter starts the waiting job right behind it
///   ([`defer_submitted`](Self::defer_submitted)).
/// - Work deferred outside any recording waits on events whose records
///   were made or are queued in jobs already submitted. On the thread that
///   installed the [`Submitter`] its job is started at once; from any other
///   thread it is kept, the installed wake is fired, and the submitter's
///   thread starts it ([`submit_deferred`](Self::submit_deferred)).
///
/// At shutdown nobody takes completions: [`drain`](Self::drain) and
/// [`close`](Self::close) wait for the events on the host, with a deadline.
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
                queued: HashMap::new(),
                unsubmitted: Vec::new(),
                status: Status::Open,
                wake: None,
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
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Install or clear the wake fired when work is deferred on another
    /// thread than the submitter's, which then starts the work's job.
    pub fn set_wake(&self, wake: Option<Wake>) {
        let previous = std::mem::replace(&mut self.lock().wake, wake);
        drop(previous);
    }

    /// Install or clear what starts the jobs of deferred work. The calling
    /// thread becomes the one that starts them: the thread that submits the
    /// lanes' jobs and takes their completions.
    pub fn set_submitter(&self, submitter: Option<Submitter>) {
        let submitter = submitter.map(|submitter| (std::thread::current().id(), submitter));
        let previous = std::mem::replace(&mut self.lock().submitter, submitter);
        drop(previous);
    }

    /// Acquires an unrecorded lease with no retained references. Storage
    /// owners retain their references before transferring them to release or
    /// deferral.
    pub fn acquire(&self, device: i32, timing: bool, interprocess: bool) -> Result<Arc<Event>> {
        let mut state = self.lock();
        require_open(&state)?;

        let available = state
            .available
            .entry((device, timing, interprocess))
            .or_default();
        // A caller may retain an event object after releasing its lease.
        // Reuse only when the pool is its last owner, preserving that observation.
        let reusable = available
            .iter()
            .position(|event| Arc::strong_count(event) == 1);
        let event = reusable
            .map(|index| available.swap_remove(index))
            .unwrap_or_else(|| {
                Arc::new(Event::new(EventFlags {
                    timing,
                    interprocess,
                    external: false,
                }))
            });
        state.active.insert(
            key(&event),
            ActiveEvent {
                event: Arc::clone(&event),
                device,
                references: 0,
                recorded: false,
                queued: None,
            },
        );
        Ok(event)
    }

    /// Records this lease once on `stream`.
    pub fn record(&self, event: &Arc<Event>, stream: impl Issue) -> Result<()> {
        let mut state = self.lock();
        require_open(&state)?;
        let device = producer(&stream)?;
        let active = require(&mut state, event, device)?;
        if active.recorded {
            return Err(invariant("device event was recorded more than once"));
        }
        stream.record(event)?;
        active.recorded = true;
        Ok(())
    }

    pub fn retain(&self, event: &Arc<Event>, device: i32, count: usize) -> Result<()> {
        let mut state = self.lock();
        require_open(&state)?;
        if count == 0 {
            return Err(invariant("device event retain count must be positive"));
        }
        require(&mut state, event, device)?.references += count;
        Ok(())
    }

    /// Acquire, record on `stream` and retain one reference: the fence of
    /// work just submitted to `stream`.
    pub fn fence(&self, stream: impl Issue, interprocess: bool) -> Result<Arc<Event>> {
        let device = producer(&stream)?;
        let event = self.acquire(device, false, interprocess)?;
        self.record(&event, stream)?;
        self.retain(&event, device, 1)?;
        Ok(event)
    }

    /// Returns references after their last device access completed, which
    /// the caller established: the pool does not ask the driver. It knows
    /// only whether the lease was recorded, and rejects the final release
    /// of one that was not, preserving its references.
    pub fn release(&self, event: &Arc<Event>, count: usize) -> Result<()> {
        let mut state = self.lock();
        // Shutdown already drained and released every lease. A completed
        // deferred owner may still release another retained event.
        if state.status == Status::Closed {
            return Ok(());
        }
        let active = state
            .active
            .get_mut(&key(event))
            .ok_or_else(|| invariant("device event is not owned by its declared device"))?;
        if count == 0 || active.references < count {
            return Err(invariant("device event reference accounting is invalid"));
        }
        if active.references == count && !active.recorded {
            return Err(invariant(
                "device event was released before it was recorded",
            ));
        }

        release_ready(&mut state, event, count);
        Ok(())
    }

    /// Transfers one retained reference of each event, and the release work
    /// `owner`, to deferred release, outside the recording of any job: the
    /// events' records were made, or are queued in jobs already submitted.
    /// `owner` runs once every event completed. With no events, or after
    /// shutdown drained the pool, it runs now.
    ///
    /// Work deferred while a job is recorded goes through the job's
    /// recorder instead ([`Issue::defer`]).
    pub fn defer(&self, events: Vec<Arc<Event>>, owner: Deferred) -> Result<()> {
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
        let wake = state.wake.clone();
        drop(state);
        if let Some(wake) = wake {
            wake.wake();
        }
        Ok(())
    }

    /// Takes the deferred work a job carried when that job was submitted.
    /// `start` starts the job that waits for the work's events on the lane
    /// the carrying job was submitted to, behind it. After shutdown drained
    /// the pool the work runs now.
    pub(crate) fn defer_submitted(
        &self,
        deferrals: Vec<Deferral>,
        start: impl Fn(DeviceId, Vec<Arc<Event>>) -> Result<JobId>,
    ) -> Result<()> {
        if deferrals.is_empty() {
            return Ok(());
        }
        let mut state = self.lock();
        if state.status == Status::Closed {
            drop(state);
            return run(deferrals.into_iter().map(|deferral| deferral.owner));
        }
        for deferral in &deferrals {
            require_references(&state, &deferral.events)?;
        }
        submit(&mut state, deferrals, Some(&start))
    }

    /// Starts the jobs of the work that was deferred on other threads, or
    /// before the submitter was installed. Does nothing on a thread that is
    /// not the submitter's.
    pub fn submit_deferred(&self) -> Result<()> {
        let mut state = self.lock();
        let here = std::thread::current().id();
        if state.unsubmitted.is_empty()
            || !matches!(&state.submitter, Some((thread, _)) if *thread == here)
        {
            return Ok(());
        }
        let deferrals = std::mem::take(&mut state.unsubmitted);
        submit(&mut state, deferrals, None)
    }

    /// Note that `job`, submitted to `lane`, queued the records of
    /// `events`: deferred work waiting on them waits on that lane until the
    /// job completed ([`Self::performed`]).
    pub(crate) fn queued(&self, job: JobId, lane: usize, events: &[Arc<Event>]) {
        let mut state = self.lock();
        let mut keys = Vec::with_capacity(events.len());
        for event in events {
            if let Some(active) = state.active.get_mut(&key(event)) {
                active.queued = Some((lane, job));
                keys.push(key(event));
            }
        }
        if !keys.is_empty() {
            state.queued.insert(job, keys);
        }
    }

    /// Note that `job` completed: the records it queued are made.
    pub(crate) fn performed(&self, job: JobId) {
        let mut state = self.lock();
        for key in state.queued.remove(&job).unwrap_or_default() {
            if let Some(active) = state.active.get_mut(&key)
                && active.queued.is_some_and(|(_, queued)| queued == job)
            {
                active.queued = None;
            }
        }
    }

    /// Whether `job` is one that deferred work waits for.
    pub fn owns(&self, job: JobId) -> bool {
        self.lock().deferred.contains_key(&job)
    }

    /// Runs the deferred work that waited for `job`, whose completion says
    /// that the work's events completed, and returns the first failure;
    /// independent owners still run after one failed. A job that failed
    /// established nothing: its work stays deferred until shutdown and the
    /// job's failure is returned.
    pub fn complete(&self, job: JobId, completion: Completion) -> Result<()> {
        completion.check().map_err(|failure| failure.error)?;
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

    /// The device of an active lease.
    pub fn device(&self, event: &Arc<Event>) -> Result<i32> {
        self.lock()
            .active
            .get(&key(event))
            .map(|active| active.device)
            .ok_or_else(|| invariant("device event is not an active lease"))
    }

    /// Waits on the host for the events of all deferred work, at most
    /// [`DRAIN_LIMIT`], and runs that work, until none is left: at
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

    /// Stops new work, waits for every recorded lease on the host, at most
    /// [`DRAIN_LIMIT`], and runs all deferred work. Existing leases may
    /// still release afterwards. A lease that did not complete in time
    /// fails the close and keeps the pool's storage.
    pub fn close(&self) -> Result<()> {
        let events: Vec<_> = {
            let mut state = self.lock();
            if state.status == Status::Open {
                state.status = Status::Closing;
            }
            state
                .active
                .values()
                .filter(|active| active.recorded)
                .map(|active| Arc::clone(&active.event))
                .collect()
        };
        let deadline = Instant::now() + DRAIN_LIMIT;
        for event in events {
            drain_event(&event, deadline)?;
        }

        let owners: Vec<_> = {
            let mut state = self.lock();
            state.status = Status::Closed;
            state.active.clear();
            state.available.clear();
            state.wake = None;
            state.submitter = None;
            let mut deferrals = std::mem::take(&mut state.unsubmitted);
            deferrals.extend(state.deferred.drain().flat_map(|(_, deferrals)| deferrals));
            deferrals
                .into_iter()
                .map(|deferral| deferral.owner)
                .collect()
        };
        run(owners)
    }

    /// Whether deferred work still waits for device completion.
    pub fn pending(&self) -> bool {
        let state = self.lock();
        !state.deferred.is_empty() || !state.unsubmitted.is_empty()
    }
}

impl Drop for EventPool {
    #[expect(
        clippy::mem_forget,
        reason = "unknown physical completion must retain deferred storage"
    )]
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
            std::mem::forget(deferrals);
        }
    }
}

/// Start the jobs that wait for the events of `deferrals` and keep the work
/// under their names. `start` starts one job behind the job that carried
/// them, on its lane; without one the installed submitter starts one job
/// per lane whose queued records the events wait for (one on any lane when
/// every record is made), and a deferral whose events span several lanes
/// runs once all of its jobs completed. Work whose job could not be
/// started stays unsubmitted, so shutdown still runs it.
fn submit(state: &mut State, deferrals: Vec<Deferral>, start: Option<&Start<'_>>) -> Result<()> {
    let Some(start) = start else {
        let mut result = Ok(());
        for deferral in deferrals {
            result = result.and(submit_waiting(state, deferral));
        }
        return result;
    };
    let events: Vec<Arc<Event>> = deferrals
        .iter()
        .flat_map(|deferral| deferral.events.iter().cloned())
        .collect();
    let started = match device(state, &events) {
        Some(device) => start(device, events),
        None => Err(invariant("deferred work waits on no active device event")),
    };
    match started {
        Ok(job) => {
            state.deferred.entry(job).or_default().extend(deferrals);
            Ok(())
        }
        Err(error) => {
            state.unsubmitted.extend(deferrals);
            Err(error)
        }
    }
}

/// The device whose notification stream waits for `events`: the first
/// event's (a stream may wait for another device's event).
fn device(state: &State, events: &[Arc<Event>]) -> Option<DeviceId> {
    events
        .first()
        .and_then(|event| state.active.get(&key(event)))
        .map(|active| DeviceId(active.device))
}

/// Start, through the installed submitter, the jobs one deferral made
/// outside a recording waits for: one per lane whose queued records its
/// events wait for, each waiting for that lane's events. The deferral's
/// work runs once every one of them completed.
fn submit_waiting(state: &mut State, deferral: Deferral) -> Result<()> {
    let mut lanes: Vec<(Option<usize>, Vec<Arc<Event>>)> = Vec::new();
    for event in &deferral.events {
        let lane = state
            .active
            .get(&key(event))
            .and_then(|active| active.queued.map(|(lane, _)| lane));
        match lanes.iter_mut().find(|(queued, _)| *queued == lane) {
            Some((_, events)) => events.push(Arc::clone(event)),
            None => lanes.push((lane, vec![Arc::clone(event)])),
        }
    }
    // Events whose records are made join any lane's job.
    if lanes.len() > 1
        && let Some(position) = lanes.iter().position(|(lane, _)| lane.is_none())
    {
        let (_, made) = lanes.remove(position);
        lanes[0].1.extend(made);
    }
    let Some((_, submitter)) = &state.submitter else {
        state.unsubmitted.push(deferral);
        return Err(invariant("deferred work has no submitter"));
    };

    let Deferral { events, owner } = deferral;
    let mut started = Vec::with_capacity(lanes.len());
    let mut failure = None;
    for (lane, waited) in &lanes {
        let job = match device(state, waited) {
            Some(device) => submitter(device, *lane, waited.clone()),
            None => Err(invariant("deferred work waits on no active device event")),
        };
        match job {
            Ok(job) => started.push(job),
            Err(error) => {
                failure = Some(error);
                break;
            }
        }
    }
    if let Some(error) = failure {
        // Jobs already started wait for nothing that runs; the work is kept
        // whole for shutdown.
        state.unsubmitted.push(Deferral { events, owner });
        return Err(error);
    }

    // Each job releases its own events' references at its completion; the
    // last to complete runs the work.
    let owner = Arc::new(Mutex::new((started.len(), Some(owner))));
    for ((_, waited), job) in lanes.into_iter().zip(started) {
        let owner = Arc::clone(&owner);
        let part: Deferred = Box::new(move || {
            let mut joined = owner.lock().unwrap_or_else(PoisonError::into_inner);
            joined.0 -= 1;
            match (joined.0, joined.1.take()) {
                (0, Some(owner)) => {
                    drop(joined);
                    owner()
                }
                (_, owner) => {
                    joined.1 = owner;
                    Ok(())
                }
            }
        });
        state.deferred.entry(job).or_default().push(Deferral {
            events: waited,
            owner: part,
        });
    }
    let _ = events;
    Ok(())
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

/// The device of the stream `stream` records on.
fn producer(stream: &impl Issue) -> Result<i32> {
    let stream = stream
        .stream()
        .ok_or_else(|| invariant("a device event is recorded on a stream"))?;
    Ok(stream.device().0)
}

fn require_open(state: &State) -> Result<()> {
    if state.status != Status::Open {
        return Err(invariant("device event pool is closed"));
    }
    Ok(())
}

fn require<'a>(
    state: &'a mut State,
    event: &Arc<Event>,
    device: i32,
) -> Result<&'a mut ActiveEvent> {
    state
        .active
        .get_mut(&key(event))
        .filter(|active| active.device == device)
        .ok_or_else(|| invariant("device event is not owned by its declared device"))
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
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use uniserve_ipc::{CallKind, Channel, ForwardMode, IpcError, Listener};

    use super::*;
    use crate::ffi::{DeferredCall, Function};
    use crate::lane::{Job, JobQueues};

    const DECODE: CallKind = CallKind::Forward(ForwardMode::Decode);

    /// A pool whose deferred work gets its jobs from `jobs`, started by
    /// the calling thread. The queues have no lane, so a job completes
    /// when it is started; what is under test is when the pool runs the
    /// work.
    fn served() -> (Arc<EventPool>, JobQueues) {
        let events = EventPool::new();
        let jobs = JobQueues::default();
        let lane = jobs.clone();
        events.set_submitter(Some(Box::new(move |device, _, events| {
            lane.after(DECODE, device, events)
        })));
        (events, jobs)
    }

    fn transport(error: IpcError) -> Error {
        Error::Transport(error.to_string())
    }

    /// A retained lease of device 0.
    fn fence(events: &EventPool) -> Result<Arc<Event>> {
        let event = events.acquire(0, false, false)?;
        events.retain(&event, 0, 1)?;
        Ok(event)
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
    fn deferred_work_runs_when_its_jobs_completion_is_handed_over() -> Result<()> {
        let (events, jobs) = served();
        let runs = Arc::new(AtomicUsize::new(0));
        let event = fence(&events)?;

        // Without an event there is nothing to wait for.
        events.defer(Vec::new(), counting(&runs))?;
        assert_eq!(runs.load(Ordering::Relaxed), 1);

        // With one, the work waits for the job the pool started, however
        // long ago that job completed.
        events.defer(vec![Arc::clone(&event)], counting(&runs))?;
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert!(events.pending());
        let (job, completion) = jobs.take().ok_or(Error::State("no job was started"))?;
        assert!(events.owns(job));

        events.complete(job, completion)?;
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        assert!(!events.pending() && !events.owns(job));
        // The work's reference went with it: the lease is returned.
        assert!(events.device(&event).is_err());
        Ok(())
    }

    #[test]
    fn deferred_work_waits_on_the_lanes_that_queued_its_records() -> Result<()> {
        let events = EventPool::new();
        let jobs = JobQueues::default();
        // The lane each waiting job was asked for, in order.
        let lanes = Arc::new(Mutex::new(Vec::new()));
        let asked = Arc::clone(&lanes);
        let queues = jobs.clone();
        events.set_submitter(Some(Box::new(move |device, lane, events| {
            asked
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(lane);
            queues.after(DECODE, device, events)
        })));
        let asked = || std::mem::take(&mut *lanes.lock().unwrap_or_else(PoisonError::into_inner));
        let runs = Arc::new(AtomicUsize::new(0));
        let first = fence(&events)?;
        let second = fence(&events)?;
        // Two submitted jobs that queued records; their own completions
        // are not what the work waits for.
        let recording = jobs.call(Job::new(), false);
        let other = jobs.call(Job::new(), false);
        while jobs.take().is_some() {}

        // A record queued on lane 1: the work waits there.
        events.queued(recording, 1, std::slice::from_ref(&first));
        events.defer(vec![Arc::clone(&first)], counting(&runs))?;
        assert_eq!(asked(), vec![Some(1)]);
        let (job, completion) = jobs.take().ok_or(Error::State("no job was started"))?;
        events.complete(job, completion)?;
        assert_eq!(runs.load(Ordering::Relaxed), 1);

        // Records queued on two lanes: one waiting job each, and the work
        // runs once both completed.
        let first = fence(&events)?;
        events.queued(recording, 1, std::slice::from_ref(&first));
        events.queued(other, 0, std::slice::from_ref(&second));
        events.defer(
            vec![Arc::clone(&first), Arc::clone(&second)],
            counting(&runs),
        )?;
        let mut waited = asked();
        waited.sort();
        assert_eq!(waited, vec![Some(0), Some(1)]);
        let (job, completion) = jobs.take().ok_or(Error::State("no job was started"))?;
        events.complete(job, completion)?;
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        let (job, completion) = jobs.take().ok_or(Error::State("no job was started"))?;
        events.complete(job, completion)?;
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // Once the recording jobs completed, their records are made: any
        // lane waits.
        events.performed(recording);
        events.performed(other);
        let third = fence(&events)?;
        events.queued(recording, 1, std::slice::from_ref(&third));
        events.performed(recording);
        events.defer(vec![third], counting(&runs))?;
        assert_eq!(asked(), vec![None]);
        Ok(())
    }

    #[test]
    fn work_a_job_carried_waits_for_the_job_started_behind_it() -> Result<()> {
        let (events, jobs) = served();
        let runs = Arc::new(AtomicUsize::new(0));
        let event = fence(&events)?;

        // Recording starts no job: the work rides with the job that
        // records its event.
        let carrying = Job::new();
        carrying
            .on(None)
            .defer(&events, vec![event], counting(&runs))?;
        assert!(jobs.take().is_none() && !events.pending());

        // Its submitter names the job that waits behind it.
        let carried = carrying.deferrals();
        events.defer_submitted(carried, |device, fences| jobs.after(DECODE, device, fences))?;
        assert_eq!(runs.load(Ordering::Relaxed), 0);
        let (job, completion) = jobs.take().ok_or(Error::State("no job was started"))?;
        events.complete(job, completion)?;
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn work_deferred_on_another_thread_gets_its_job_from_the_submitters() -> Result<()> {
        let (events, jobs) = served();
        let runs = Arc::new(AtomicUsize::new(0));
        // The wake of a connected channel: a fired wake leaves its local
        // descriptor readable.
        let listener = Listener::bind(&["127.0.0.1:0"]).map_err(transport)?;
        let owner = Channel::connect(&listener.addresses()[0]).map_err(transport)?;
        events.set_wake(Some(owner.wake()));
        let event = fence(&events)?;

        // The other thread starts no job; it wakes the submitter's.
        std::thread::scope(|scope| {
            scope
                .spawn(|| events.defer(vec![event], counting(&runs)))
                .join()
        })
        .map_err(|_| Error::State("the deferring thread panicked"))??;
        assert!(jobs.take().is_none());
        let mut descriptors = owner.poll_fds();
        // SAFETY: `descriptors` is a valid array of pollfd of the length given.
        let ready = unsafe { libc::poll(descriptors.as_mut_ptr(), 2, 0) };
        assert!(ready > 0 && descriptors[1].revents & libc::POLLIN != 0);
        assert!(events.pending());

        events.submit_deferred()?;
        let (job, completion) = jobs.take().ok_or(Error::State("no job was started"))?;
        events.complete(job, completion)?;
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn a_failed_job_leaves_its_work_deferred_until_the_pool_drains() -> Result<()> {
        let (events, jobs) = served();
        let runs = Arc::new(AtomicUsize::new(0));
        let event = fence(&events)?;

        // The job the work waits for fails at a call.
        let refused = tvm_ffi::Function::from_packed(|_| {
            Err(tvm_ffi::Error::new(tvm_ffi::RUNTIME_ERROR, "refused", ""))
        });
        let failing = Job::new();
        failing.on(None).call(DeferredCall::new(
            Function::new("test.failing", refused),
            Vec::new(),
        ))?;
        let failing = RefCell::new(Some(failing));
        let carrying = Job::new();
        carrying
            .on(None)
            .defer(&events, vec![event], counting(&runs))?;
        events.defer_submitted(carrying.deferrals(), |_, _| {
            let job = failing.take().ok_or(Error::State("started twice"))?;
            jobs.start(DECODE, job, true)
        })?;

        // Its failure says nothing about the event: the work does not run.
        let (job, completion) = jobs.take().ok_or(Error::State("no job was started"))?;
        assert!(events.complete(job, completion).is_err());
        assert_eq!(runs.load(Ordering::Relaxed), 0);
        assert!(events.pending());

        // At shutdown the pool waits for the event itself.
        events.drain()?;
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert!(!events.pending());
        Ok(())
    }

    #[test]
    fn work_without_a_submitter_waits_for_one_or_for_shutdown() -> Result<()> {
        let events = EventPool::new();
        let runs = Arc::new(AtomicUsize::new(0));
        let event = fence(&events)?;
        events.defer(vec![event], counting(&runs))?;
        events.submit_deferred()?;
        assert_eq!(runs.load(Ordering::Relaxed), 0);

        events.close()?;
        assert_eq!(runs.load(Ordering::Relaxed), 1);

        // A closed pool has nothing left to wait for.
        events.defer(Vec::new(), counting(&runs))?;
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        Ok(())
    }
}
