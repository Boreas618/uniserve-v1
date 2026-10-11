//! Work submitted to a lane's execution thread.

use crossbeam_channel::{Receiver, Sender};

use crate::handoff::{Handoff, hand};

/// One item of work a lane's execution thread runs: a forward's submission.
///
/// The work is the forward as its capability runtime issues it, with every
/// stream write, launch, read and Python call it makes. It borrows the
/// submitter's state, so the submitter waits until the job has run and no
/// other thread touches that state meanwhile.
struct Job(Handoff);

/// The submitting end of a lane's job queue.
#[derive(Clone)]
pub struct JobQueue {
    jobs: Sender<Job>,
}

/// The execution thread's end of a lane's job queue.
pub(super) struct Jobs {
    jobs: Receiver<Job>,
}

impl JobQueue {
    pub(super) fn new() -> (Self, Jobs) {
        let (sender, receiver) = crossbeam_channel::unbounded();
        (Self { jobs: sender }, Jobs { jobs: receiver })
    }

    /// Run a forward's submission on the lane's execution thread and return
    /// its result. The caller waits, and the forward may borrow its state.
    #[expect(
        clippy::expect_used,
        reason = "the execution thread lives as long as a queue, and a job stores its result"
    )]
    pub fn run<T>(&self, forward: impl FnOnce() -> T) -> T {
        let mut forward = Some(forward);
        let mut result = None;
        let mut work = || result = forward.take().map(|forward| forward());
        hand(&self.jobs, Job, &mut work);

        result.expect("a job that ran has a result")
    }
}

impl Jobs {
    /// Run jobs in arrival order on the calling thread, the lane's
    /// execution thread, until every submitting end is gone.
    pub(super) fn serve(self) {
        while let Ok(Job(handoff)) = self.jobs.recv() {
            handoff.run();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forwards_run_on_the_execution_thread_in_order_and_borrow_their_submitter() {
        let (queue, jobs) = JobQueue::new();
        let thread = std::thread::spawn(move || {
            jobs.serve();
            std::thread::current().id()
        });
        let here = std::thread::current().id();

        // A forward mutates state borrowed from its submitter and returns a
        // value; the next forward runs after it.
        let mut order = Vec::new();
        let first = queue.run(|| {
            order.push("first");
            std::thread::current().id()
        });
        let second = queue.run(|| {
            order.push("second");
            std::thread::current().id()
        });
        assert_eq!(order, ["first", "second"]);
        assert_ne!(first, here);
        assert_eq!(second, first);

        // The thread ends with its last queue.
        drop(queue);
        assert_eq!(thread.join().unwrap(), first);
    }
}
