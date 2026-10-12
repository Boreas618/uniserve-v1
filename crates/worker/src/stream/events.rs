//! Shared CUDA event leases and deferred storage release.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use uniserve_cuda::{DeviceGuard, DeviceId, Event, EventFlags, Stream};
use uniserve_ipc::Wake;

use super::Issue;
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
    /// The stream the lease was recorded on, which selects its wake stream.
    stream: Option<usize>,
}

struct DeferredRelease {
    events: Vec<Arc<Event>>,
    owner: Deferred,
}

struct State {
    available: HashMap<PoolKey, Vec<Arc<Event>>>,
    active: HashMap<usize, ActiveEvent>,
    deferred: Vec<DeferredRelease>,
    wake_streams: HashMap<(i32, usize), Arc<Stream>>,
    status: Status,
    wake: Option<Wake>,
}

/// Owns pooled events until their references and device accesses retire.
///
/// Storage owners record a fence after the last device access to their
/// storage, retain a reference to it, and either release it once it is
/// query-ready or [`defer`](Self::defer) the storage's release until it is.
/// Deferred work runs on the thread that calls [`reap`](Self::reap) (or
/// another pool method that reaps), after the pool's lock is released, so it
/// may use the pool again.
///
/// When a completion wake is installed, [`wake_after`](Self::wake_after)
/// orders a host function after an event that fires the wake, so the service
/// loop reaps promptly instead of polling.
pub struct EventPool {
    state: Mutex<State>,
}

impl Default for EventPool {
    fn default() -> Self {
        Self {
            state: Mutex::new(State {
                available: HashMap::new(),
                active: HashMap::new(),
                deferred: Vec::new(),
                wake_streams: HashMap::new(),
                status: Status::Open,
                wake: None,
            }),
        }
    }
}

impl EventPool {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // No user code runs under the lock: deferred work runs after unlock.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Install or clear the wake fired after events passed to
    /// [`wake_after`](Self::wake_after).
    pub fn set_wake(&self, wake: Option<Wake>) {
        let previous = std::mem::replace(&mut self.lock().wake, wake);
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
                stream: None,
            },
        );
        Ok(event)
    }

    /// Records this lease once on `stream`.
    pub fn record(&self, event: &Arc<Event>, stream: impl Issue) -> Result<()> {
        let mut state = self.lock();
        require_open(&state)?;
        let (device, raw) = producer(&stream)?;
        let active = require(&mut state, event, device)?;
        if active.stream.is_some() {
            return Err(invariant("device event was recorded more than once"));
        }
        stream.record(event)?;
        active.stream = Some(raw);
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
        let (device, _) = producer(&stream)?;
        let event = self.acquire(device, false, interprocess)?;
        self.record(&event, stream)?;
        self.retain(&event, device, 1)?;
        Ok(event)
    }

    /// Returns references after their last device access. A rejected final
    /// release preserves its references so the caller can defer them instead.
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
        if active.references == count && (active.stream.is_none() || !event.query()?) {
            return Err(invariant(
                "device event was released before it became query-ready",
            ));
        }

        release_ready(&mut state, event, count);
        Ok(())
    }

    /// Transfers one retained reference of each event, and the release work
    /// `owner`, to deferred release. `owner` runs once every event completed.
    /// With no events, or after shutdown drained the pool, it runs now.
    pub fn defer(&self, events: Vec<Arc<Event>>, owner: Deferred) -> Result<()> {
        let immediate = {
            let mut state = self.lock();
            if state.status == Status::Closed || events.is_empty() {
                Some(owner)
            } else {
                require_references(&state, &events)?;
                state.deferred.push(DeferredRelease { events, owner });
                None
            }
        };
        match immediate {
            Some(owner) => owner(),
            None => Ok(()),
        }
    }

    /// Runs deferred work whose events completed and returns the first
    /// failure. Failed queries retain their owners; independent completed
    /// owners still run after a failure.
    pub fn reap(&self) -> Result<()> {
        let (completed, mut failure) = {
            let mut state = self.lock();
            collect_completed(&mut state)
        };
        for owner in completed {
            if let Err(error) = owner()
                && failure.is_ok()
            {
                failure = Err(error);
            }
        }
        failure
    }

    /// The device of an active lease.
    pub fn device(&self, event: &Arc<Event>) -> Result<i32> {
        self.lock()
            .active
            .get(&key(event))
            .map(|active| active.device)
            .ok_or_else(|| invariant("device event is not an active lease"))
    }

    /// Orders the installed completion wake after `event`, whose record
    /// was made, on an independent notification stream per producer.
    /// Without a wake this does nothing.
    pub fn wake_after(&self, event: &Arc<Event>) -> Result<()> {
        None::<&Stream>.wake(self, event)
    }

    /// The installed completion wake, for an issuer that orders it behind
    /// an event on a notification stream of its own (a lane's job).
    pub(crate) fn wake(&self) -> Option<Wake> {
        self.lock().wake.clone()
    }

    /// The stream and wake that notify of `event`'s completion for an
    /// issuer without a lane ([`Self::wake_after`]): one notification
    /// stream per producer stream, created on the calling thread when
    /// first asked for. `None` without an installed wake.
    pub(crate) fn notification(&self, event: &Arc<Event>) -> Result<Option<(Arc<Stream>, Wake)>> {
        let mut state = self.lock();
        let Some(wake) = state.wake.clone() else {
            return Ok(None);
        };
        require_open(&state)?;
        let (device, producer) = state
            .active
            .get(&key(event))
            .and_then(|active| Some((active.device, active.stream?)))
            .ok_or_else(|| {
                invariant("completion notification requires a recorded producer event")
            })?;
        let stream = match state.wake_streams.get(&(device, producer)) {
            Some(stream) => Arc::clone(stream),
            None => {
                let _device = DeviceGuard::new(DeviceId(device))?;
                let stream = Arc::new(Stream::new(DeviceId(device))?);
                state
                    .wake_streams
                    .insert((device, producer), Arc::clone(&stream));
                stream
            }
        };
        Ok(Some((stream, wake)))
    }

    /// Stops new work, waits for every recorded lease on the host and runs
    /// all deferred work. Existing leases may still release afterwards.
    pub fn close(&self) -> Result<()> {
        let (streams, events) = {
            let mut state = self.lock();
            if state.status == Status::Open {
                state.status = Status::Closing;
            }
            let streams: Vec<_> = state.wake_streams.values().cloned().collect();
            let events: Vec<_> = state
                .active
                .values()
                .filter(|active| active.stream.is_some())
                .map(|active| Arc::clone(&active.event))
                .collect();
            (streams, events)
        };
        for stream in streams {
            stream.synchronize()?;
        }
        for event in events {
            event.synchronize()?;
        }

        let owners: Vec<_> = {
            let mut state = self.lock();
            state.status = Status::Closed;
            state.wake_streams.clear();
            state.active.clear();
            state.available.clear();
            state.wake = None;
            state
                .deferred
                .drain(..)
                .map(|release| release.owner)
                .collect()
        };
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

    /// Whether deferred work still waits for device completion.
    pub fn pending(&self) -> bool {
        !self.lock().deferred.is_empty()
    }
}

impl Drop for EventPool {
    #[expect(
        clippy::mem_forget,
        reason = "unknown physical completion must retain deferred storage"
    )]
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        if state.deferred.is_empty() {
            return;
        }
        // An unclosed pool still owns storage whose release waits on device
        // work. Without a successful drain it cannot be returned.
        let drained = state.deferred.iter().all(|release| {
            release
                .events
                .iter()
                .all(|event| event.synchronize().is_ok())
        });
        let deferred = std::mem::take(&mut state.deferred);
        if drained {
            for release in deferred {
                let _ = (release.owner)();
            }
        } else {
            std::mem::forget(deferred);
        }
    }
}

fn collect_completed(state: &mut State) -> (Vec<Deferred>, Result<()>) {
    let mut completed = Vec::new();
    let mut failure = Ok(());
    let mut index = 0;

    while index < state.deferred.len() {
        let release = &state.deferred[index];
        let ready = release.events.iter().try_fold(true, |ready, event| {
            event.query().map(|done| ready && done).map_err(Error::Cuda)
        });
        let result = ready.and_then(|ready| {
            if ready {
                require_references(state, &release.events)?;
            }
            Ok(ready)
        });

        match result {
            Ok(true) => {
                let release = state.deferred.swap_remove(index);
                for event in &release.events {
                    release_ready(state, event, 1);
                }
                completed.push(release.owner);
            }
            result => {
                if let Err(error) = result
                    && failure.is_ok()
                {
                    failure = Err(error);
                }
                index += 1;
            }
        }
    }
    (completed, failure)
}

/// The device and raw handle of the stream `stream` records on.
fn producer(stream: &impl Issue) -> Result<(i32, usize)> {
    let stream = stream
        .stream()
        .ok_or_else(|| invariant("a device event is recorded on a stream"))?;
    Ok((stream.device().0, stream.raw()))
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
