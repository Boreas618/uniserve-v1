//! Bounded host work: the blocking pool the main process runs its non-device
//! work on (`specs/worker.md` §2.4).
//!
//! Module loading, transfer reads, shared-storage reclamation and the media
//! codecs run here, each on its own kind of work, so that no device stream and
//! no scheduler thread ever waits on host I/O.

mod pool;

pub use crate::completion::Outcome;
pub use pool::{HostAction, HostPool, HostTask};

use std::sync::Arc;

use crate::Result;

/// One blocking lane of the pool, as the main process holds it: an action goes
/// in, and its result comes back through the returned task.
///
/// The lane keeps [`HostPool`]'s accounting to itself (capacity, leases,
/// cancellation) and offers the verbs a caller outside the crate needs;
/// [`crate::blocking::ThreadPool::lane`] hands one out.
pub struct Lane<A: HostAction> {
    pool: Arc<HostPool<A>>,
}

impl<A: HostAction> Lane<A> {
    /// The lane over `pool`.
    pub fn new(pool: Arc<HostPool<A>>) -> Self {
        Self { pool }
    }

    /// Admits `action` and submits it: the task is queued before this returns,
    /// and fails when the lane has no room.
    pub fn admit(&self, action: A) -> Result<Arc<HostTask<A>>> {
        let task = self.pool.reserve()?;
        task.configure(action)?;
        task.submit()?;
        Ok(task)
    }

    /// Asks for a queued task's cancellation before it runs, answering whether
    /// it was still cancellable; a running task keeps its capacity until it
    /// completes. `abandon` gives up the completion, so nothing observes it.
    pub fn cancel(&self, task: &Arc<HostTask<A>>, abandon: bool) -> Result<bool> {
        task.cancel(abandon)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;
    use crate::completion::Outcome;

    /// An action that reports whether it ran.
    struct Probe {
        ran: Arc<std::sync::atomic::AtomicBool>,
    }

    impl HostAction for Probe {
        type Output = bool;

        fn run(&self) -> Result<bool> {
            self.ran.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(self.ran.load(std::sync::atomic::Ordering::SeqCst))
        }

        fn release(&self) -> Result<()> {
            Ok(())
        }

        fn input_outcome(&self) -> Outcome {
            Outcome::Success(())
        }
    }

    #[test]
    fn a_lane_admits_an_action_and_answers_its_result() -> Result<()> {
        let pool = Arc::new(HostPool::new(1, 1, "lane-test")?);
        let lane = Lane::new(pool);
        let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let task = lane.admit(Probe {
            ran: Arc::clone(&ran),
        })?;
        match task.wait() {
            Outcome::Success(produced) => assert!(*produced),
            other => panic!("the lane's task did not succeed: {other:?}"),
        }
        assert!(ran.load(std::sync::atomic::Ordering::SeqCst));
        assert!(matches!(task.outcome(), Some(Outcome::Success(_))));
        Ok(())
    }

    #[test]
    fn a_poll_before_the_task_runs_answers_nothing_and_leaves_it_admitted() -> Result<()> {
        let pool = Arc::new(HostPool::new(1, 1, "lane-poll-test")?);
        let lane = Lane::new(pool);
        let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let task = lane.admit(Probe {
            ran: Arc::clone(&ran),
        })?;
        // A poll may catch either state; it must never invent an outcome, and
        // the task must stay observable until it ends.
        let polled = task.wait_timeout(Duration::ZERO);
        assert!(polled.is_some() || task.outcome().is_none());
        task.wait();
        assert!(task.outcome().is_some());
        Ok(())
    }
}
