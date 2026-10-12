//! Cooperative host turns for numerical microbatches.

use std::cell::RefCell;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

use uniserve_cuda::Stream;

use super::expert_union::ExpertUnion;
use crate::ffi::{Handle, cooperative};
use crate::stream::Issue;
use crate::{Error, Result};

thread_local! {
    static CURRENT: RefCell<Option<(Arc<Microbatches>, usize)>> = const { RefCell::new(None) };
}

struct State {
    active: Vec<bool>,
    turn: usize,
    running: bool,
    closed: bool,
    aborted: bool,
    failure: Option<usize>,
}

/// Rotate ordinary numerical calls at expert dispatch points.
///
/// Each index runs on its own persistent host lane. Calls start in index order,
/// yield to the next active index and leave the rotation when they return.
/// The caller joins all host tasks before ending an invocation or closing.
pub struct Microbatches {
    state: Mutex<State>,
    ready: Condvar,
}

impl Microbatches {
    pub fn new(count: usize) -> Result<Self> {
        if count == 0 {
            return Err(Error::Invalid(
                "microbatches require at least one context".into(),
            ));
        }

        Ok(Self {
            state: Mutex::new(State {
                active: vec![false; count],
                turn: 0,
                running: false,
                closed: false,
                aborted: false,
                failure: None,
            }),
            ready: Condvar::new(),
        })
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn begin(&self, count: usize) -> Result<()> {
        let mut state = self.state();
        if count != state.active.len() {
            return Err(Error::Invalid(
                "each microbatch context needs one numerical call".into(),
            ));
        }
        if state.running {
            return Err(Error::State("microbatch invocation is already running"));
        }
        if state.closed {
            return Err(Error::State("microbatch execution is closed"));
        }

        state.active.fill(true);
        state.turn = 0;
        state.running = true;
        state.aborted = false;
        state.failure = None;
        Ok(())
    }

    /// Execute one admitted index, once per invocation. The caller starts
    /// every index before joining any of them. Waiting and yielding require
    /// no language runtime lock; only the numerical callback acquires it.
    pub fn run<T, E>(
        self: &Arc<Self>,
        index: usize,
        call: impl FnOnce() -> std::result::Result<T, E>,
    ) -> Result<std::result::Result<T, E>> {
        let previous = CURRENT.replace(Some((Arc::clone(self), index)));
        let mut turn = Turn {
            owner: Arc::clone(self),
            index,
            previous,
            failed: true,
        };
        self.wait(index, self.state())?;
        let result = call();
        turn.failed = result.is_err();
        Ok(result)
    }

    fn wait(&self, index: usize, state: MutexGuard<'_, State>) -> Result<()> {
        let state = self
            .ready
            .wait_while(state, |state| state.turn != index && !state.aborted)
            .unwrap_or_else(PoisonError::into_inner);
        if state.aborted {
            return Err(Error::State("microbatch execution aborted"));
        }
        Ok(())
    }

    fn resume(&self, index: usize) -> Result<()> {
        let mut state = self.state();
        next(&mut state, index);
        self.ready.notify_all();
        self.wait(index, state)
    }

    /// Interrupt suspended calls. True means this is the first failure;
    /// otherwise the failing task remains the source of the invocation error.
    pub fn abort(&self) -> bool {
        let mut state = self.state();
        let first = !state.aborted;
        state.aborted = true;
        self.ready.notify_all();
        first
    }

    pub fn failure(&self) -> Option<usize> {
        self.state().failure
    }

    /// Complete an invocation after joining its host turns and device streams.
    pub fn end(&self) {
        let mut state = self.state();
        state.running = false;
        state.active.fill(false);
        state.failure = None;
    }

    pub fn close(&self) -> Result<()> {
        let mut state = self.state();
        if state.running {
            return Err(Error::State("cannot close running microbatches"));
        }
        state.closed = true;
        Ok(())
    }
}

/// Yield after submitting an expert dispatch. Outside a microbatch call this
/// is a no-op, so numerical layers use the same operation in ordinary forwards.
pub fn yield_microbatch() -> Result<()> {
    let current = CURRENT.with_borrow(Clone::clone);
    if let Some((owner, index)) = current {
        owner.resume(index)?;
    }
    Ok(())
}

fn next(state: &mut State, index: usize) {
    let count = state.active.len();
    if let Some(next) = (1..=count)
        .map(|offset| (index + offset) % count)
        .find(|&next| state.active[next])
    {
        state.turn = next;
    }
}

/// Retire a turn on ordinary return, callback error or unwinding. Restoring
/// the thread-local owner also permits nested independent numerical invocations.
struct Turn {
    owner: Arc<Microbatches>,
    index: usize,
    previous: Option<(Arc<Microbatches>, usize)>,
    failed: bool,
}

impl Drop for Turn {
    fn drop(&mut self) {
        CURRENT.set(self.previous.take());
        let mut state = self.owner.state();
        if self.failed && !state.aborted {
            state.failure = Some(self.index);
            state.aborted = true;
        }
        state.active[self.index] = false;
        next(&mut state, self.index);
        self.owner.ready.notify_all();
    }
}

/// One microbatch's numerical call, run on its lane thread inside the
/// rotation.
pub type MicrobatchCall<T> = Box<dyn FnOnce() -> Result<T> + Send>;

type Job = Box<dyn FnOnce() + Send>;

/// One persistent host thread executing a microbatch's calls in order.
struct LaneThread {
    jobs: Option<crossbeam_channel::Sender<Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// Persistent host threads and the cooperative rotation of numerical
/// microbatches.
///
/// Each microbatch has its own host thread, CUDA stream and expert exchange.
/// [`Self::run`] submits one call per microbatch: calls start in index order
/// and pass the host turn to the next active microbatch whenever they yield
/// after an expert dispatch, so every rank submits its microbatches'
/// collectives in the same order while their device work overlaps on
/// distinct streams. Threads persist so thread-local numerical state (cuBLAS
/// handles) exists before graph capture.
pub struct MicrobatchLanes {
    control: Arc<Microbatches>,
    threads: Vec<LaneThread>,
    /// Recorded on the parent stream; every microbatch stream waits on it.
    fork: uniserve_cuda::Event,
    /// Recorded on each microbatch stream; the parent waits on them.
    joins: Vec<uniserve_cuda::Event>,
}

impl MicrobatchLanes {
    /// Start `count` lane threads, named `microbatch-{index}`.
    pub fn new(count: usize) -> Result<Self> {
        let control = Arc::new(Microbatches::new(count)?);
        let mut threads = Vec::with_capacity(count);
        for index in 0..count {
            let (sender, receiver) = crossbeam_channel::unbounded::<Job>();
            let thread = std::thread::Builder::new()
                .name(format!("microbatch-{index}"))
                .spawn(move || {
                    // Jobs report their own outcome; a closed channel ends
                    // the thread.
                    while let Ok(job) = receiver.recv() {
                        job();
                    }
                })
                .map_err(|error| Error::Transport(format!("microbatch thread: {error}")))?;
            threads.push(LaneThread {
                jobs: Some(sender),
                thread: Some(thread),
            });
        }
        let event = || uniserve_cuda::Event::new(uniserve_cuda::EventFlags::default());
        Ok(Self {
            control,
            threads,
            fork: event(),
            joins: (0..count).map(|_| event()).collect(),
        })
    }

    pub fn count(&self) -> usize {
        self.threads.len()
    }

    /// Run `call(index)` once on every lane thread, outside the rotation,
    /// for thread-local preparation.
    pub fn each(&self, call: Arc<dyn Fn(usize) -> Result<()> + Send + Sync>) -> Result<()> {
        let calls: Vec<MicrobatchCall<()>> = (0..self.count())
            .map(|index| {
                let call = Arc::clone(&call);
                Box::new(move || call(index)) as MicrobatchCall<()>
            })
            .collect();
        self.submit(calls, None)
            .into_iter()
            .collect::<Result<Vec<()>>>()?;
        Ok(())
    }

    /// Run one call per microbatch inside the rotation and return their
    /// results in index order.
    ///
    /// Every microbatch stream first waits on `parent`'s submitted work, and
    /// `parent` waits on every microbatch stream before this returns, also
    /// on failure, so the caller reads results (or retires inputs) in
    /// `parent`'s order. On failure every suspended call is woken and every
    /// host turn retires before the first failing microbatch's error is
    /// returned.
    pub fn run<T: Send + 'static>(
        &self,
        parent: &uniserve_cuda::Stream,
        streams: &[uniserve_cuda::Stream],
        calls: Vec<MicrobatchCall<T>>,
    ) -> Result<Vec<T>> {
        if streams.len() != self.count() || calls.len() != self.count() {
            return Err(Error::Invalid(
                "each microbatch needs one stream and one call".into(),
            ));
        }
        self.fork.record(parent)?;
        for stream in streams {
            stream.wait_event(&self.fork)?;
        }
        self.control.begin(calls.len())?;
        let mut outcomes = self.submit(calls, Some(&self.control));
        let failure = self.control.failure();
        self.control.end();

        // Results are read in the parent's order whatever happened.
        let mut joined = Ok(());
        for (stream, event) in streams.iter().zip(&self.joins) {
            joined = joined
                .and_then(|()| event.record(stream).map_err(Error::from))
                .and_then(|()| parent.wait_event(event).map_err(Error::from));
        }
        // The first failing microbatch caused every other abort.
        if let Some(index) = failure
            && outcomes[index].is_err()
        {
            return outcomes.swap_remove(index).map(|_| Vec::new());
        }
        let results = outcomes.into_iter().collect::<Result<Vec<T>>>()?;
        joined?;
        Ok(results)
    }

    /// Submit one call per lane and wait for every outcome. Inside a
    /// rotation (`control`), each call runs its turn.
    fn submit<T: Send + 'static>(
        &self,
        calls: Vec<MicrobatchCall<T>>,
        control: Option<&Arc<Microbatches>>,
    ) -> Vec<Result<T>> {
        let (sender, receiver) = crossbeam_channel::bounded::<(usize, Result<T>)>(calls.len());
        let mut outcomes: Vec<Option<Result<T>>> = (0..calls.len()).map(|_| None).collect();
        let mut pending = 0;
        for (index, call) in calls.into_iter().enumerate() {
            let sender = sender.clone();
            let rotation = control.cloned();
            let job: Job = Box::new(move || {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    match &rotation {
                        Some(control) => control.run(index, call).and_then(|result| result),
                        None => call(),
                    }
                }))
                .unwrap_or_else(|_| Err(Error::Invariant(format!("microbatch {index} panicked"))));
                let _ = sender.send((index, outcome));
            });
            let sent = self.threads[index]
                .jobs
                .as_ref()
                .ok_or(Error::State("microbatch execution is closed"))
                .and_then(|jobs| {
                    jobs.send(job)
                        .map_err(|_| Error::State("microbatch thread exited"))
                });
            match sent {
                Ok(()) => pending += 1,
                Err(error) => {
                    // Unsubmitted calls never take their turn; wake the rest.
                    if let Some(control) = control {
                        control.abort();
                    }
                    outcomes[index] = Some(Err(error));
                }
            }
        }
        drop(sender);
        for _ in 0..pending {
            match receiver.recv() {
                Ok((index, outcome)) => outcomes[index] = Some(outcome),
                Err(_) => break,
            }
        }
        outcomes
            .into_iter()
            .map(|outcome| outcome.unwrap_or(Err(Error::State("microbatch call was lost"))))
            .collect()
    }

    /// Join the lane threads after every invocation and captured reader
    /// retired.
    pub fn close(&mut self) -> Result<()> {
        self.control.close()?;
        for lane in &mut self.threads {
            drop(lane.jobs.take());
            if let Some(thread) = lane.thread.take()
                && thread.join().is_err()
            {
                return Err(Error::Invariant("a microbatch thread panicked".into()));
            }
        }
        Ok(())
    }
}

impl Drop for MicrobatchLanes {
    fn drop(&mut self) {
        // Closing the channels ends the threads; they are not joined here,
        // because a failed rank may still have a call blocked on a peer.
        for lane in &mut self.threads {
            drop(lane.jobs.take());
        }
    }
}

/// One microbatch's forward in an expert step.
pub(crate) struct MicrobatchForward {
    /// The forward's operations on the microbatch's stream: a graph launch
    /// or a runner method call, and what follows it. Its microbatch thread
    /// performs them in its turn.
    pub(crate) job: crate::lane::Job,
    /// Whether the forward passes the host turn at its expert dispatch
    /// points: an eager forward does and stays suspended in Python
    /// meanwhile; a captured forward submits its whole step at once.
    pub(crate) yields: bool,
}

impl MicrobatchForward {
    /// The forward as the call of its microbatch.
    fn call(self) -> MicrobatchCall<()> {
        Box::new(move || match self.job.perform().check() {
            Ok(_) => Ok(()),
            Err(failure) => Err(failure.error),
        })
    }
}

/// Make `call` a cooperative turn of its microbatch thread: the thread
/// calls Python itself, because a call that yields at a dispatch point
/// keeps its Python frames on that thread until the rotation returns to it.
/// Only eager forwards and eager joins are turns; a captured forward or
/// join launches its graph and enters no Python on a microbatch thread.
pub(super) fn turn(call: MicrobatchCall<()>) -> MicrobatchCall<()> {
    Box::new(move || cooperative(call))
}

impl ExpertUnion {
    /// Microbatches; one without microbatch execution.
    pub(crate) fn microbatches(&self) -> usize {
        self.microbatches.len()
    }

    /// The exchange of microbatch `peer`.
    pub(crate) fn transport(&self, peer: usize) -> &Handle {
        &self.microbatches[peer].exchange
    }

    /// The stream of microbatch `peer`.
    pub(crate) fn stream(&self, peer: usize) -> &Stream {
        &self.microbatches[peer].stream
    }

    /// Order every microbatch stream after the work `job` holds for
    /// `parent`, before the job takes the microbatches' inputs.
    pub(crate) fn fork(&self, job: &crate::lane::Job, parent: &Stream) -> Result<()> {
        if self.microbatches.len() < 2 {
            return Ok(());
        }
        let event = Arc::new(uniserve_cuda::Event::new(
            uniserve_cuda::EventFlags::default(),
        ));
        job.on(Some(parent)).record(&event)?;
        for peer in &self.microbatches[1..] {
            job.on(Some(&peer.stream)).wait(&event)?;
        }
        Ok(())
    }

    /// Order `parent` after the work `job` holds for every microbatch
    /// stream.
    pub(crate) fn join_streams(&self, job: &crate::lane::Job, parent: &Stream) -> Result<()> {
        for peer in &self.microbatches[1..] {
            let event = Arc::new(uniserve_cuda::Event::new(
                uniserve_cuda::EventFlags::default(),
            ));
            job.on(Some(&peer.stream)).record(&event)?;
            job.on(Some(parent)).wait(&event)?;
        }
        Ok(())
    }

    /// Run one expert step of `capacity`: every populated microbatch's
    /// forward and every empty microbatch's join. With microbatches, they
    /// run in rotation on the microbatch threads, and `parent` waits on
    /// every microbatch stream before this returns.
    pub(crate) fn run(
        &self,
        parent: &Stream,
        forwards: Vec<Option<MicrobatchForward>>,
        capacity: usize,
    ) -> Result<()> {
        if forwards.len() != self.microbatches.len() {
            return Err(Error::Invariant(
                "an expert step has one call per microbatch".into(),
            ));
        }
        let Some(lanes) = &self.lanes else {
            return match forwards.into_iter().next().flatten() {
                Some(forward) => forward.call()(),
                None => self.peer_join(0, capacity)?(),
            };
        };
        let calls = forwards
            .into_iter()
            .enumerate()
            .map(|(peer, forward)| match forward {
                Some(forward) if forward.yields => Ok(turn(forward.call())),
                Some(forward) => Ok(forward.call()),
                None => self.peer_join(peer, capacity),
            })
            .collect::<Result<Vec<_>>>()?;
        lanes.run(parent, &self.streams(), calls).map(drop)
    }

    pub(super) fn streams(&self) -> Vec<Stream> {
        self.microbatches
            .iter()
            .map(|peer| peer.stream.clone())
            .collect()
    }
}
