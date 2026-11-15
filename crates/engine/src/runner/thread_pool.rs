//! Bounded thread pools for the engine's blocking host work.
//!
//! The engine does blocking host work of several kinds: loading modules, and encoding, muxing and
//! reading media. None of it runs on a scheduler thread or on a worker thread: those wait only
//! for their own device work and their own queue, and a media encode that takes a second must not
//! delay the module load of an update.
//!
//! Each kind of work therefore has its own [`ThreadPool`]: its own queue, its own threads and its
//! own admission bound. The owner of the work creates the pool and submits to it
//! ([`ThreadPool::submit`], as Python's `ThreadPoolExecutor.submit`); the [`HostTask`] it gets
//! back is where the work's result is read or waited for.
//!
//! A pool is aborted when its last owner drops it: admission stops and the queued tasks are
//! cancelled, without joining work that may wait on a failed device or peer. A task's inputs are
//! released by its pool, by the rule of [`HostAction::input_status`]: an action whose inputs'
//! physical producer failed or was cancelled is kept until the process exits, because device work
//! whose completion is unknown may still address them.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread;

use crate::worker::completion::{Completion, Outcome, Waker};
use crate::worker::{Error, Result};
use service::lock;

/// Work a thread pool runs.
///
/// The pool owns admission, queueing, the result and the release of the inputs; an action
/// supplies its work and says whether its inputs may be released.
pub(crate) trait HostAction: Send + Sync + Sized + 'static {
    /// What the work returns.
    type Output: Send + Sync;

    /// Does the work, on one of the pool's threads.
    ///
    /// # Errors
    ///
    /// The work's own failure, which becomes the task's result.
    fn run(&self) -> Result<Self::Output>;

    /// Releases the action's inputs. The pool calls it once, after the work ended or was
    /// withdrawn, and only when [`Self::input_status`] reported a success.
    ///
    /// # Errors
    ///
    /// The release's failure, which replaces a successful result of the work.
    fn release(&self) -> Result<()>;

    /// How the physical producer of the action's inputs ended, asked once the work ended or was
    /// withdrawn.
    ///
    /// A failure or a cancellation leaves the producer's last device access unknown: the pool
    /// then keeps the action, and every input it holds, until the process exits.
    fn input_status(&self) -> Outcome;
}

/// One submitted action and its result.
///
/// A task holds one unit of its pool's capacity from its submission until it ended or was
/// cancelled with its pool. Its action, and the inputs the action holds, stay alive for the whole
/// run.
pub(crate) struct HostTask<A: HostAction> {
    /// The pool's admission state, which the task returns its capacity to.
    pool: Weak<Mutex<PoolState>>,
    /// The action, until the thread that runs or cancels it takes it.
    action: Mutex<Option<A>>,
    completion: Completion<Waker, Arc<A::Output>>,
}

impl<A: HostAction> HostTask<A> {
    /// Waits for the task's result.
    pub(crate) fn wait(&self) -> Outcome<Arc<A::Output>> {
        self.completion.wait()
    }

    /// Calls `wake` when the task ends, at once for a task that already has: a caller holding
    /// several tasks waits on one signal of its own and reads [`Self::wait`] of each when it
    /// fires, which then returns at once, instead of polling them.
    ///
    /// `wake` runs on the thread that ended the task, so it only signals.
    pub(crate) fn on_settled(&self, wake: &Waker) {
        if let Some(wake) = self.completion.subscribe(Arc::clone(wake)) {
            wake();
        }
    }

    /// Runs the task on the calling thread of its pool, or cancels it when the pool was
    /// `aborted` before a thread took it.
    fn run(&self, aborted: bool) {
        // The queue hands each task to one thread once, so the action is there to take.
        let Some(action) = lock(&self.action).take() else {
            return;
        };

        // The pool's threads carry their pool's name, which the log lines below name.
        let pool = thread::current().name().unwrap_or("thread-pool").to_owned();

        if aborted {
            if let Err(error) = release_inputs(action) {
                tracing::warn!("{pool}: releasing a withdrawn task's inputs failed: {error}");
            }
            self.return_capacity();
            wake(self.completion.cancel().1);
            return;
        }

        let mut result = action.run();
        if let Err(cleanup) = release_inputs(action) {
            match &result {
                Ok(_) => result = Err(cleanup),
                Err(error) => tracing::warn!(
                    "{pool}: releasing the inputs of a failed task also failed: {cleanup} \
                     (task: {error})"
                ),
            }
        }
        // The capacity goes back before the result is observable, so whoever observes the result
        // can submit again at once.
        self.return_capacity();
        // A running task rejects cancellation, so this is the completion's one result.
        wake(
            self.completion
                .complete(result.map(Arc::new))
                .unwrap_or_default(),
        );
    }

    /// Returns the task's unit of capacity to its pool.
    fn return_capacity(&self) {
        if let Some(pool) = self.pool.upgrade() {
            lock(&pool).admitted -= 1;
        }
    }
}

/// Releases `action`'s inputs when their physical producer stopped successfully. An unknown
/// completion keeps the action, and its inputs, until the process exits, whatever the task's own
/// result.
fn release_inputs<A: HostAction>(action: A) -> Result<()> {
    match action.input_status() {
        Outcome::Success(()) => action.release(),
        Outcome::Failed(_) | Outcome::Cancelled => {
            crate::worker::leak(action);
            Ok(())
        }
    }
}

/// Calls the wakers a completion returned, after its owner released its locks.
fn wake(wakers: Vec<Waker>) {
    for waker in wakers {
        waker();
    }
}

/// A queued task as its pool holds it: the call that runs it, or cancels it when its argument
/// says the pool was aborted.
type Queued = Box<dyn FnOnce(bool) + Send>;

struct PoolState {
    /// How many tasks the pool admits at once, running and queued together.
    capacity: usize,
    /// The tasks admitted and not yet ended or withdrawn.
    admitted: usize,
    /// No further task is admitted; the threads end once the queue is empty.
    closed: bool,
    /// The queued tasks are cancelled instead of run.
    aborted: bool,
    /// The queued tasks in submission order.
    queue: VecDeque<Queued>,
}

/// A bounded pool of threads for one kind of blocking host work.
///
/// The pool admits at most `capacity` tasks at once, running and queued together, and refuses a
/// submission beyond that instead of blocking its caller. Dropping the pool aborts it; its
/// threads end once the task each runs ended, and are not joined.
pub(crate) struct ThreadPool {
    state: Arc<Mutex<PoolState>>,
    /// Signalled when a task is queued and when the pool is aborted.
    ready: Arc<Condvar>,
}

impl ThreadPool {
    /// A pool of `threads` threads named `{name}-{index}` that admits `capacity` tasks at once.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] unless `threads` lies in `1..=capacity`, and [`Error::Invariant`] when
    /// the host refuses a thread.
    pub(crate) fn new(name: &str, capacity: usize, threads: usize) -> Result<Self> {
        if threads == 0 || threads > capacity {
            return Err(Error::Invalid(format!(
                "thread pool {name} needs between one thread and its capacity of {capacity}, \
                 got {threads}"
            )));
        }

        let state = Arc::new(Mutex::new(PoolState {
            capacity,
            admitted: 0,
            closed: false,
            aborted: false,
            queue: VecDeque::new(),
        }));
        let ready = Arc::new(Condvar::new());

        let mut handles = Vec::with_capacity(threads);
        for index in 0..threads {
            let pool = Arc::clone(&state);
            let queued = Arc::clone(&ready);
            match thread::Builder::new()
                .name(format!("{name}-{index}"))
                .spawn(move || run_queue(&pool, &queued))
            {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    // The threads already started end with the pool that was not built.
                    lock(&state).closed = true;
                    ready.notify_all();
                    for handle in handles {
                        let _ = handle.join();
                    }
                    return Err(Error::Invariant(format!(
                        "cannot start a thread of thread pool {name}: {error}"
                    )));
                }
            }
        }

        Ok(Self { state, ready })
    }

    /// Queues `action` and returns the task its result is read from.
    ///
    /// # Errors
    ///
    /// [`Error::Resource`] when the pool is closed or already holds `capacity` tasks; the action
    /// is dropped unrun.
    pub(crate) fn submit<A: HostAction>(&self, action: A) -> Result<Arc<HostTask<A>>> {
        let mut state = lock(&self.state);
        if state.closed {
            return Err(Error::Resource("the thread pool is closed"));
        }
        if state.admitted >= state.capacity {
            return Err(Error::Resource("the thread pool's capacity is exhausted"));
        }

        let task = Arc::new(HostTask {
            pool: Arc::downgrade(&self.state),
            action: Mutex::new(Some(action)),
            completion: Completion::default(),
        });
        let queued = Arc::clone(&task);
        state.admitted += 1;
        state
            .queue
            .push_back(Box::new(move |aborted| queued.run(aborted)));
        drop(state);

        self.ready.notify_one();
        Ok(task)
    }

    /// Stops admission and cancels the queued tasks, without joining a thread: running work may
    /// be waiting on a failed device or peer.
    pub(crate) fn abort(&self) {
        let mut state = lock(&self.state);
        state.closed = true;
        state.aborted = true;
        drop(state);

        self.ready.notify_all();
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // A destructor does not wait for work that may be waiting on a failed device or peer.
        self.abort();
    }
}

/// One thread of a pool: runs the queued tasks in submission order until the pool is closed and
/// its queue is empty.
fn run_queue(pool: &Mutex<PoolState>, ready: &Condvar) {
    loop {
        let mut state = service::wait_while(ready, lock(pool), |state| {
            state.queue.is_empty() && !state.closed
        });
        let Some(run) = state.queue.pop_front() else {
            return;
        };
        let aborted = state.aborted;
        drop(state);

        run(aborted);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// An action that tells `started` it runs and waits at `gate` when it has them, counts its
    /// run and returns the count. `producer` is how the producer of its `input` ended.
    struct Counted {
        runs: Arc<AtomicUsize>,
        started: Option<crossbeam_channel::Sender<()>>,
        gate: Option<Arc<Barrier>>,
        producer: Outcome,
        input: Arc<()>,
    }

    impl Counted {
        fn new(runs: &Arc<AtomicUsize>) -> Self {
            Self {
                runs: Arc::clone(runs),
                started: None,
                gate: None,
                producer: Outcome::Success(()),
                input: Arc::new(()),
            }
        }

        fn gated(runs: &Arc<AtomicUsize>, gate: &Arc<Barrier>) -> Self {
            Self {
                gate: Some(Arc::clone(gate)),
                ..Self::new(runs)
            }
        }
    }

    impl HostAction for Counted {
        type Output = usize;

        fn run(&self) -> Result<usize> {
            if let Some(started) = &self.started {
                let _ = started.send(());
            }
            if let Some(gate) = &self.gate {
                gate.wait();
            }
            // The pool holds the action, and so its input, for the whole run.
            assert!(Arc::strong_count(&self.input) >= 1);
            Ok(self.runs.fetch_add(1, Ordering::SeqCst) + 1)
        }

        fn release(&self) -> Result<()> {
            Ok(())
        }

        fn input_status(&self) -> Outcome {
            self.producer.clone()
        }
    }

    #[test]
    fn a_task_returns_the_result_of_the_action_submitted() -> Result<()> {
        let pool = ThreadPool::new("pool-test", 4, 1)?;
        let runs = Arc::new(AtomicUsize::new(0));

        let task = pool.submit(Counted::new(&runs))?;
        match task.wait() {
            Outcome::Success(count) => assert_eq!(*count, 1),
            other => panic!("the task did not succeed: {other:?}"),
        }

        assert_eq!(runs.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn a_full_pool_refuses_and_an_ended_task_returns_its_capacity() -> Result<()> {
        let pool = ThreadPool::new("pool-test", 2, 1)?;
        let runs = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(Barrier::new(2));

        // The first task holds the pool's one thread, so the second stays queued.
        let running = pool.submit(Counted::gated(&runs, &gate))?;
        let queued = pool.submit(Counted::new(&runs))?;
        assert!(matches!(
            pool.submit(Counted::new(&runs)),
            Err(Error::Resource(_))
        ));

        gate.wait();
        assert!(matches!(running.wait(), Outcome::Success(_)));
        assert!(matches!(queued.wait(), Outcome::Success(_)));
        // The capacity goes back before the result is observable.
        let admitted = pool.submit(Counted::new(&runs))?;
        assert!(matches!(admitted.wait(), Outcome::Success(_)));
        assert_eq!(runs.load(Ordering::SeqCst), 3);
        Ok(())
    }

    #[test]
    fn an_aborted_pool_cancels_what_it_queued() -> Result<()> {
        let pool = ThreadPool::new("pool-test", 2, 1)?;
        let runs = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(Barrier::new(2));

        // The first task runs on the pool's one thread when the pool is aborted.
        let (started, running_now) = crossbeam_channel::bounded(1);
        let running = pool.submit(Counted {
            started: Some(started),
            ..Counted::gated(&runs, &gate)
        })?;
        let queued = pool.submit(Counted::new(&runs))?;
        running_now.recv().expect("the first task starts");
        pool.abort();

        gate.wait();
        assert!(matches!(running.wait(), Outcome::Success(_)));
        assert!(matches!(queued.wait(), Outcome::Cancelled));
        assert_eq!(runs.load(Ordering::SeqCst), 1, "the queued task never ran");
        Ok(())
    }

    #[test]
    fn a_waker_fires_when_the_task_ends_and_at_once_afterwards() -> Result<()> {
        let pool = ThreadPool::new("pool-test", 1, 1)?;
        let runs = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(Barrier::new(2));
        let (fired, woken) = crossbeam_channel::unbounded();
        let wake: Waker = Arc::new(move || {
            let _ = fired.send(());
        });

        let task = pool.submit(Counted::gated(&runs, &gate))?;
        task.on_settled(&wake);
        assert!(woken.try_recv().is_err());

        gate.wait();
        // The pool's thread calls the wakers once it recorded the result.
        woken
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("the waker fires when the task ends");
        assert!(matches!(task.wait(), Outcome::Success(_)));

        task.on_settled(&wake);
        assert!(woken.try_recv().is_ok());
        Ok(())
    }

    #[test]
    fn an_aborted_pool_admits_nothing() -> Result<()> {
        let pool = ThreadPool::new("pool-test", 4, 1)?;
        pool.abort();

        let runs = Arc::new(AtomicUsize::new(0));
        assert!(pool.submit(Counted::new(&runs)).is_err());
        Ok(())
    }

    #[test]
    fn more_threads_than_capacity_is_refused() {
        assert!(ThreadPool::new("pool-test", 1, 2).is_err());
        assert!(ThreadPool::new("pool-test", 1, 0).is_err());
    }

    /// Runs one task whose input's producer ended with `producer` and returns whether the input
    /// is still held once the pool, the task and every other owner are gone. The pool releases
    /// a task's inputs before it records the task's result.
    fn held_after_close(producer: Outcome) -> Result<bool> {
        let pool = ThreadPool::new("pool-test", 1, 1)?;
        let runs = Arc::new(AtomicUsize::new(0));
        let input = Arc::new(());
        let held = Arc::downgrade(&input);

        let task = pool.submit(Counted {
            producer,
            input,
            ..Counted::new(&runs)
        })?;
        assert!(matches!(task.wait(), Outcome::Success(_)));

        drop(task);
        drop(pool);
        Ok(held.upgrade().is_some())
    }

    #[test]
    fn inputs_whose_producer_did_not_stop_outlive_the_pool() -> Result<()> {
        assert!(!held_after_close(Outcome::Success(()))?);

        let unknown = Arc::new(Error::transport("device access could not be drained"));
        assert!(held_after_close(Outcome::Failed(unknown))?);
        assert!(held_after_close(Outcome::Cancelled)?);
        Ok(())
    }
}
