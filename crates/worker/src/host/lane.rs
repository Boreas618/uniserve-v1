//! Bounded host execution shared by media encoding and cache transfer.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::thread::{self, JoinHandle};

use uniserve_ipc::Wake;

use crate::{Completion, Error, Outcome, Result, lock};

/// A one-shot observer of a host task's result.
pub(crate) type HostCallback = Box<dyn FnOnce() + Send>;

/// Work a host lane runs. The lane owns admission, scheduling, results and
/// input retirement; an action supplies its work and its inputs' lifetime.
pub(crate) trait HostAction: Send + Sync + Sized + 'static {
    type Output: Send + Sync;

    fn run(&self) -> Result<Self::Output>;

    /// Release the action's inputs once their physical producer stopped.
    fn release(&self) -> Result<()>;

    /// Whether the inputs' physical producer stopped successfully, asked
    /// once the action finished or was withdrawn. A failed or cancelled
    /// outcome leaves the producer's completion unknown: the lane then keeps
    /// the action, and every input it holds, until process exit.
    fn input_outcome(&self) -> Outcome;
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Reserved,
    Queued,
    Running,
    Finished,
}

struct TaskState<A> {
    phase: Phase,
    action: Option<Arc<A>>,
}

/// One admitted action and its result. Cancellation removes queued work before
/// returning capacity; running work retains its inputs until it completes.
pub(crate) struct HostTask<A: HostAction> {
    lane: Weak<Mutex<LaneState<A>>>,
    state: Mutex<TaskState<A>>,
    pub completion: Completion<HostCallback, Arc<A::Output>>,
}

impl<A: HostAction> HostTask<A> {
    fn state(&self) -> MutexGuard<'_, TaskState<A>> {
        lock(&self.state)
    }

    pub(crate) fn configure(&self, action: A) -> Result<()> {
        let lane = self
            .lane
            .upgrade()
            .ok_or(Error::State("host task is no longer admitted"))?;
        let lane = lock(&lane);
        let mut state = self.state();

        if lane.closed || state.phase != Phase::Reserved {
            return Err(Error::State("host task is no longer admitted"));
        }
        if state.action.is_some() {
            return Err(Error::State("host task was configured more than once"));
        }

        state.action = Some(Arc::new(action));
        Ok(())
    }

    pub(crate) fn submit(self: &Arc<Self>) -> Result<()> {
        let lane = self
            .lane
            .upgrade()
            .ok_or(Error::State("host task is no longer admitted"))?;
        let mut lane = lock(&lane);
        let mut state = self.state();

        if lane.closed || state.phase == Phase::Finished {
            return Err(Error::State("host task is no longer admitted"));
        }
        if state.phase != Phase::Reserved {
            return Err(Error::State("host task was submitted more than once"));
        }
        if state.action.is_none() {
            return Err(Error::State("host task has no action"));
        }

        lane.queue.push_back(Arc::clone(self));
        state.phase = Phase::Queued;
        lane.ready.notify_one();

        Ok(())
    }

    /// Abandon withdraws only unsubmitted work; cancel also withdraws queued
    /// work. Running actions keep their capacity and input leases to completion.
    pub(crate) fn cancel(self: &Arc<Self>, abandon: bool) -> Result<bool> {
        let lane = self.lane.upgrade();
        let mut lane = lane.as_ref().map(|lane| lock(lane));
        let mut state = self.state();

        if matches!(state.phase, Phase::Running | Phase::Finished)
            || (abandon && state.phase == Phase::Queued)
        {
            return Ok(false);
        }

        if let Some(lane) = lane.as_mut() {
            // Dequeue and cancellation use the same admission lock. Once a
            // worker claims this task, cancellation can no longer release it.
            if state.phase == Phase::Queued {
                lane.queue.retain(|task| !Arc::ptr_eq(task, self));
            }
            lane.tasks.remove(&(Arc::as_ptr(self) as usize));
        }
        state.phase = Phase::Finished;
        let (cancelled, callbacks) = self.completion.cancel();
        drop(state);
        drop(lane);

        let released = self.release_input();
        notify(callbacks);
        released.map(|()| cancelled)
    }

    /// Release the action's inputs once their physical producer stopped. An
    /// unknown completion retains the action, and its inputs, for process
    /// teardown, independently of the task's result.
    fn release_input(&self) -> Result<()> {
        let Some(action) = self.state().action.take() else {
            return Ok(());
        };
        match action.input_outcome() {
            Outcome::Success(()) => action.release(),
            Outcome::Failed(_) | Outcome::Cancelled => {
                crate::leak(action);
                Ok(())
            }
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "only configured queued tasks reach a worker; running tasks reject cancellation"
    )]
    fn run(self: &Arc<Self>, aborted: bool) {
        // Worker threads carry their lane's name, which log records name.
        let lane = thread::current().name().unwrap_or("host").to_owned();
        let (action, cancelled, callbacks) = {
            let mut state = self.state();
            let callbacks = if aborted {
                self.completion.cancel().1
            } else {
                Vec::new()
            };
            let cancelled = self.completion.done();
            state.phase = if cancelled {
                Phase::Finished
            } else {
                Phase::Running
            };
            (
                state.action.clone().expect("configured host task"),
                cancelled,
                callbacks,
            )
        };
        notify(callbacks);

        if cancelled {
            drop(action);
            if let Err(error) = self.release_input() {
                tracing::warn!(target: "uniserve_worker.host", "{lane}: releasing a withdrawn task's inputs failed: {error}");
            }
            self.remove();
            return;
        }

        let mut result = action.run();
        drop(action);
        if let Err(cleanup) = self.release_input() {
            match &result {
                Ok(_) => result = Err(cleanup),
                Err(error) => {
                    tracing::warn!(target: "uniserve_worker.host", "{lane}: releasing the inputs of a failed task also failed: {cleanup} (task: {error})")
                }
            }
        }
        self.state().phase = Phase::Finished;

        // A result observer can immediately reserve the returned capacity.
        let wake = self.remove();
        let callbacks = self
            .completion
            .complete(result.map(Arc::new))
            .expect("unfinished running task");
        notify(callbacks);
        if let Some(wake) = wake {
            wake.wake();
        }
    }

    fn remove(self: &Arc<Self>) -> Option<Wake> {
        let lane = self.lane.upgrade()?;
        let mut lane = lock(&lane);
        lane.tasks.remove(&(Arc::as_ptr(self) as usize));
        lane.wake.clone()
    }
}

/// Run the observers a completion returned, after its owner unlocked.
fn notify(callbacks: Vec<HostCallback>) {
    for callback in callbacks {
        callback();
    }
}

struct LaneState<A: HostAction> {
    capacity: usize,
    closed: bool,
    aborted: bool,
    tasks: HashMap<usize, Arc<HostTask<A>>>,
    queue: VecDeque<Arc<HostTask<A>>>,
    ready: Arc<Condvar>,
    wake: Option<Wake>,
}

/// A bounded lane with native worker threads. Abort stops admission without
/// joining tasks that may be waiting on a failed device or peer.
pub(crate) struct HostLane<A: HostAction> {
    state: Arc<Mutex<LaneState<A>>>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl<A: HostAction> HostLane<A> {
    pub(crate) fn new(capacity: usize, workers: usize, name: &str) -> Result<Self> {
        if capacity == 0 || workers == 0 || workers > capacity {
            return Err(Error::Invalid(
                "host workers must lie within positive lane capacity".into(),
            ));
        }

        let ready = Arc::new(Condvar::new());
        let state = Arc::new(Mutex::new(LaneState {
            capacity,
            closed: false,
            aborted: false,
            tasks: HashMap::new(),
            queue: VecDeque::new(),
            ready: Arc::clone(&ready),
            wake: None,
        }));

        let mut threads = Vec::new();
        for index in 0..workers {
            let shared = Arc::clone(&state);
            let ready = Arc::clone(&ready);
            match thread::Builder::new()
                .name(format!("{name}-{index}"))
                .spawn(move || run_worker(shared, ready))
            {
                Ok(thread) => threads.push(thread),
                Err(error) => {
                    {
                        let mut state = lock(&state);
                        state.closed = true;
                        state.ready.notify_all();
                    }
                    for thread in threads {
                        let _ = thread.join();
                    }
                    return Err(Error::Invariant(format!(
                        "cannot start host worker: {error}"
                    )));
                }
            }
        }

        Ok(Self {
            state,
            threads: Mutex::new(threads),
        })
    }

    fn state(&self) -> MutexGuard<'_, LaneState<A>> {
        lock(&self.state)
    }

    pub(crate) fn set_wake(&self, wake: Option<Wake>) {
        let previous = std::mem::replace(&mut self.state().wake, wake);
        drop(previous);
    }

    pub(crate) fn reserve(&self) -> Result<Arc<HostTask<A>>> {
        let mut state = self.state();
        if state.closed {
            return Err(Error::Resource("worker host lane is closed"));
        }
        if state.tasks.len() >= state.capacity {
            return Err(Error::Resource("worker host lane capacity is exhausted"));
        }
        let task = Arc::new(HostTask {
            lane: Arc::downgrade(&self.state),
            state: Mutex::new(TaskState {
                phase: Phase::Reserved,
                action: None,
            }),
            completion: Completion::default(),
        });
        state
            .tasks
            .insert(Arc::as_ptr(&task) as usize, Arc::clone(&task));
        Ok(task)
    }

    pub(crate) fn abort(&self) {
        let wake = {
            let mut state = self.state();
            state.closed = true;
            state.aborted = true;
            state.ready.notify_all();
            state.wake.take()
        };
        drop(wake);
    }

    pub(crate) fn close(&self) -> Vec<Error> {
        let unused: Vec<_> = {
            let mut state = self.state();
            state.closed = true;
            state
                .tasks
                .values()
                .filter(|task| task.state().phase == Phase::Reserved)
                .cloned()
                .collect()
        };
        let mut errors = Vec::new();
        for task in unused {
            if let Err(error) = task.cancel(true) {
                errors.push(error);
            }
        }
        self.state().ready.notify_all();
        let threads = std::mem::take(&mut *lock(&self.threads));
        for thread in threads {
            if thread.thread().id() == thread::current().id() {
                errors.push(Error::State("host worker cannot join itself"));
                continue;
            }
            if thread.join().is_err() {
                errors.push(Error::State("host worker panicked"));
            }
        }
        self.set_wake(None);
        errors
    }
}

impl<A: HostAction> Drop for HostLane<A> {
    fn drop(&mut self) {
        // A destructor must not wait on a failed peer or require the GIL.
        self.abort();
    }
}

fn run_worker<A: HostAction>(state: Arc<Mutex<LaneState<A>>>, ready: Arc<Condvar>) {
    loop {
        let mut state = ready
            .wait_while(lock(&state), |state| {
                state.queue.is_empty() && !state.closed
            })
            .unwrap_or_else(PoisonError::into_inner);
        let Some(task) = state.queue.pop_front() else {
            return;
        };

        task.state().phase = Phase::Running;
        let aborted = state.aborted;
        drop(state);

        task.run(aborted);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;

    /// An action whose inputs' producer stopped with `outcome`; it holds
    /// `input` while the lane retains it.
    struct Probe {
        outcome: Outcome,
        _input: Arc<()>,
    }

    impl HostAction for Probe {
        type Output = ();

        fn run(&self) -> Result<()> {
            Ok(())
        }

        fn release(&self) -> Result<()> {
            Ok(())
        }

        fn input_outcome(&self) -> Outcome {
            self.outcome.clone()
        }
    }

    /// Run one task of `outcome` to its end and report whether its input is
    /// still held once the lane, the task and every handle are gone.
    fn retained_after_close(outcome: Outcome) -> Result<bool> {
        let lane = HostLane::new(1, 1, "lane-test")?;
        let input = Arc::new(());
        let held = Arc::downgrade(&input);
        let task = lane.reserve()?;
        task.configure(Probe {
            outcome,
            _input: input,
        })?;
        let (done, finished) = mpsc::channel();
        if let Some(callback) = task.completion.subscribe(Box::new(move || {
            let _ = done.send(());
        })) {
            callback();
        }
        task.submit()?;
        finished
            .recv_timeout(Duration::from_secs(10))
            .map_err(|_| Error::State("the host task did not finish"))?;
        assert!(matches!(
            task.completion.outcome(),
            Some(Outcome::Success(_))
        ));

        if let Some(error) = lane.close().into_iter().next() {
            return Err(error);
        }
        drop(task);
        drop(lane);
        Ok(held.upgrade().is_some())
    }

    #[test]
    fn inputs_of_an_unknown_completion_outlive_the_lane() -> Result<()> {
        assert!(!retained_after_close(Outcome::Success(()))?);
        let unknown = Arc::new(Error::transport("device access could not be drained"));
        assert!(retained_after_close(Outcome::Failed(unknown))?);
        assert!(retained_after_close(Outcome::Cancelled)?);
        Ok(())
    }
}
