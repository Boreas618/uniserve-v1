//! A rank's jobs: the work its builder thread performs for the control thread,
//! and the completions it posts back.
//!
//! A [`Job`] is the ordered list of what one step of a batch puts on the rank's
//! streams: copies, fills, event records and waits, executor calls, a host
//! function. The control thread builds it and submits it ([`JobQueue::start`]);
//! the builder thread, the one thread that executes Python
//! ([`crate::ffi::Interpreter`]), performs every job in submission order: it
//! issues the job's stream work and makes its executor calls itself, so each
//! stream's order is the order the jobs were submitted in. It posts a
//! [`Completion`] per job under the job's [`JobId`], to the one place the
//! control thread takes completions from ([`JobQueue::take`]), and fires the
//! control thread's wake behind it. A job that names events ([`Job::after`])
//! has its completion posted by the driver once they completed. Work that
//! issues nothing on a stream (loading, releases) is a job like any other.
//!
//! While it serves, the control thread waits for no job: it starts work and
//! takes completions when its wake fires. Before the rank serves and after it
//! stopped, a caller may wait for one job, with a deadline
//! ([`JobQueue::block_on`], [`JobQueue::drain`]).
//!
//! Because every job is submitted and every completion taken here, the queue
//! knows which jobs the builder thread has been given and not completed, and
//! since when ([`JobQueue::oldest`]): a builder thread that stays inside one
//! job, such as a driver call that does not return, is seen from the control
//! thread, which reports it to the engine.
//!
//! In a process without a started builder thread (the library used from Python,
//! tests), a job is performed on the thread that submits it.

mod operation;
mod post;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use cuda::{DeviceId, Event};
use ipc::Wake;

use crate::ffi::Interpreter;
use crate::{Error, Result, lock};

pub(crate) use operation::{Completion, Job, Recorder};
pub(crate) use post::JobId;

/// The control thread's side of the builder thread's jobs: submission, the
/// completions it posts and the jobs in flight.
#[derive(Clone)]
pub(crate) struct JobQueue {
    shared: Arc<Shared>,
}

struct Shared {
    /// Where the builder thread posts completions.
    completions: Receiver<(JobId, Completion)>,
    /// The posting end every job is given.
    posted: Sender<(JobId, Completion)>,
    /// The control thread's wake, fired behind a completion. Set and read on
    /// the control thread.
    wake: Mutex<Option<Wake>>,
    /// The name of the next job.
    next: AtomicU64,
    /// Completions taken while another job was waited for, in arrival order.
    arrived: Mutex<VecDeque<(JobId, Completion)>>,
    /// The jobs the builder thread has been given and not completed. Changed
    /// only by the thread that submits jobs and takes completions.
    running: Mutex<Running>,
}

/// The jobs the builder thread has been given and not completed, in submission
/// order. It performs its jobs in that order, so the first is the one it is
/// performing; a job whose completion waits for device events ([`Job::after`])
/// leaves the thread at once and may complete after the jobs behind it.
#[derive(Default)]
struct Running {
    /// Each job, when it was submitted and the library method it calls.
    jobs: VecDeque<(JobId, Instant, Option<String>)>,
    /// When the last completion of one of its jobs arrived.
    completed: Option<Instant>,
}

impl Default for JobQueue {
    /// The rank's one queue: every clone submits to the builder thread and
    /// takes the completions of every job any clone started.
    fn default() -> Self {
        let (posted, completions) = crossbeam_channel::unbounded();
        Self {
            shared: Arc::new(Shared {
                completions,
                posted,
                wake: Mutex::new(None),
                next: AtomicU64::new(0),
                arrived: Mutex::new(VecDeque::new()),
                running: Mutex::new(Running::default()),
            }),
        }
    }
}

impl JobQueue {
    fn arrived(&self) -> std::sync::MutexGuard<'_, VecDeque<(JobId, Completion)>> {
        lock(&self.shared.arrived)
    }

    fn running(&self) -> std::sync::MutexGuard<'_, Running> {
        lock(&self.shared.running)
    }

    /// Note that the completion of job `id` arrived now.
    fn received(&self, id: JobId) {
        let mut running = self.running();
        if let Some(position) = running.jobs.iter().position(|(job, ..)| *job == id) {
            running.jobs.remove(position);
            running.completed = Some(Instant::now());
        }
    }

    /// The job in flight that the builder thread has been performing, and since
    /// when: its first job, timed from its submission or from the arrival of
    /// the thread's last completion, whichever is later. `None` when no job is
    /// in flight.
    ///
    /// Only completions taken so far count, so a caller takes the ones that
    /// arrived before it asks.
    pub(crate) fn oldest(&self) -> Option<(JobId, Instant)> {
        let running = self.running();
        let &(id, submitted, _) = running.jobs.front()?;
        let since = running
            .completed
            .map_or(submitted, |completed| completed.max(submitted));
        Some((id, since))
    }

    /// The library method job `id` calls, for a job in flight: `Some(None)`
    /// when it calls none. `None` for a job that is not in flight.
    pub(crate) fn performing(&self, id: JobId) -> Option<Option<String>> {
        self.running()
            .jobs
            .iter()
            .find(|(job, ..)| *job == id)
            .map(|(_, _, method)| method.clone())
    }

    /// Queue `job` on the builder thread without waiting for it. Its completion
    /// arrives under the returned name. `wake` says whether the control
    /// thread's wake is fired behind it, or the completion is left to be taken
    /// when the control thread next takes one (nothing goes on from it).
    pub(crate) fn start(&self, job: Job, wake: bool) -> JobId {
        let id = self.name();
        let method = job.method();
        let wake = if wake { self.wake().clone() } else { None };
        let post = post::Post::new(id, self.shared.posted.clone(), wake);
        // Noted before the submission: in a process without a builder thread
        // the job is performed, and its completion posted, before `submit`
        // returns.
        self.running().jobs.push_back((id, Instant::now(), method));
        Interpreter::submit(move || post::perform(job, post));
        id
    }

    /// Queue a job that holds no operation and completes once `events`
    /// completed ([`Job::after`]): the builder thread orders the completion
    /// behind them on its notification stream of `device` and goes on. The
    /// control thread's wake is fired behind it. The events' records are made,
    /// or queued in jobs submitted before: a wait on an event without a record
    /// passes.
    pub(crate) fn after(&self, device: DeviceId, events: Vec<Arc<Event>>) -> JobId {
        let job = Job::new();
        job.after(device, events);
        self.start(job, true)
    }

    /// Start `job` and wait on the calling thread for its completion, at most
    /// `timeout`.
    ///
    /// For a rank that is not serving only: startup, whose captures need each
    /// job's result before the next is prepared, and shutdown. While the rank
    /// serves, the control thread starts a job and takes its completion
    /// ([`Self::start`], [`Self::take`]). Completions of other jobs that arrive
    /// meanwhile are kept for [`Self::take`].
    ///
    /// A completion, here as anywhere, says that the builder thread performed
    /// the job: it issued the job's stream work in order and ran its calls. It
    /// does not say that the device finished the work issued. Storage the job's
    /// device work addresses may be read on the host, written again outside the
    /// stream's order or released after the completion only behind something
    /// that waited for the device: a wait for the stream the job recorded after
    /// that work ([`crate::stream::Issue::synchronize`] on its recorder), the
    /// events the job completes after ([`Job::after`]), or the fences that
    /// retire the storage ([`crate::EventPool`]).
    pub(crate) fn block_on(&self, job: Job, timeout: Duration) -> Result<Completion> {
        let id = self.start(job, false);
        self.completion(id, Instant::now() + timeout)
    }

    /// Wait until the builder thread performed the jobs submitted so far, at
    /// most `timeout`: before the owners those jobs address are closed. Like
    /// [`Self::block_on`], for a rank that is not serving; performed is not
    /// finished, so the caller then synchronizes the streams.
    pub(crate) fn drain(&self, timeout: Duration) -> Result<()> {
        self.block_on(Job::new(), timeout)?
            .check()
            .map(drop)
            .map_err(|failure| failure.error)
    }

    /// The completion of job `id`, waited for until `deadline`.
    fn completion(&self, id: JobId, deadline: Instant) -> Result<Completion> {
        let mut arrived = self.arrived();
        if let Some(position) = arrived.iter().position(|(job, _)| *job == id) {
            let (_, completion) = arrived.remove(position).ok_or(LOST)?;
            return Ok(completion);
        }
        drop(arrived);

        loop {
            let received = self.shared.completions.recv_deadline(deadline);
            if let Ok((job, _)) = &received {
                self.received(*job);
            }
            match received {
                Ok((job, completion)) if job == id => return Ok(completion),
                Ok(other) => self.arrived().push_back(other),
                Err(RecvTimeoutError::Timeout) => {
                    return Err(Error::State(
                        "a job the rank waited for did not complete before its deadline",
                    ));
                }
                Err(RecvTimeoutError::Disconnected) => return Err(LOST),
            }
        }
    }

    fn wake(&self) -> std::sync::MutexGuard<'_, Option<Wake>> {
        lock(&self.shared.wake)
    }

    /// The name of the next job.
    fn name(&self) -> JobId {
        JobId(self.shared.next.fetch_add(1, Ordering::Relaxed))
    }

    /// The next completion that arrived, without waiting: one kept while
    /// another job was waited for, or one posted since.
    pub(crate) fn take(&self) -> Option<(JobId, Completion)> {
        let kept = self.arrived().pop_front();
        kept.or_else(|| {
            let posted = self.shared.completions.try_recv().ok();
            if let Some((job, _)) = &posted {
                self.received(*job);
            }
            posted
        })
    }

    /// Wait on the calling thread, at most `timeout`, for the next completion
    /// of a job in flight, and keep it for [`Self::take`]. Returns whether one
    /// arrived. For a rank that admits nothing more, whose control thread waits
    /// for the jobs it issued instead of serving: a job started without a wake
    /// ([`Self::start`]) completes unannounced.
    pub(crate) fn await_completion(&self, timeout: Duration) -> Result<bool> {
        match self.shared.completions.recv_timeout(timeout) {
            Ok(posted) => {
                self.received(posted.0);
                self.arrived().push_back(posted);
                Ok(true)
            }
            Err(RecvTimeoutError::Timeout) => Ok(false),
            Err(RecvTimeoutError::Disconnected) => Err(LOST),
        }
    }

    /// Fire `wake` behind every completion of a job started from now on, so the
    /// control thread, which blocks on its socket and its wake descriptor,
    /// learns of a completion it does not wait for.
    pub(crate) fn set_wake(&self, wake: Option<Wake>) {
        (*self.wake()).clone_from(&wake);
    }
}

const LOST: Error = Error::State("the rank's builder thread has exited");

/// How long a caller that is not serving waits for one job
/// ([`JobQueue::block_on`], [`JobQueue::drain`]). A startup job may compile
/// kernels inside its first eager forward, which has taken minutes; the bound
/// is there so that a builder thread that stopped making progress fails the
/// rank instead of hanging it.
pub(crate) const BLOCK_ON_TIMEOUT: Duration = Duration::from_secs(1800);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::Issue;

    /// The next completion that was posted.
    fn next(queue: &JobQueue) -> (JobId, Completion) {
        queue
            .take()
            .unwrap_or_else(|| panic!("no completion was posted"))
    }

    /// A job that notes the thread performing it under `name`.
    fn noting(
        name: &'static str,
        notes: &Arc<Mutex<Vec<(&'static str, std::thread::ThreadId)>>>,
    ) -> Result<Job> {
        let notes = Arc::clone(notes);
        let job = Job::new();
        job.on(None).launch_host_fn(Box::new(move || {
            let mut notes = lock(&notes);
            notes.push((name, std::thread::current().id()));
        }))?;
        Ok(job)
    }

    #[test]
    fn jobs_complete_in_their_order_under_the_names_their_submission_gave() -> Result<()> {
        // No builder thread is started in this process: a job is performed on
        // its submitter's thread.
        let queue = JobQueue::default();
        let here = std::thread::current().id();
        let notes = Arc::new(Mutex::new(Vec::new()));
        let first = queue.start(noting("first", &notes)?, true);
        let second = queue.start(noting("second", &notes)?, true);
        assert_ne!(first, second);
        for id in [first, second] {
            let (job, completion) = next(&queue);
            assert_eq!(job, id);
            assert!(completion.check().is_ok());
        }
        assert_eq!(lock(&notes)[..], [("first", here), ("second", here)]);
        assert!(queue.oldest().is_none());
        Ok(())
    }

    #[test]
    fn a_wait_before_serving_keeps_the_completions_of_other_jobs() -> Result<()> {
        let queue = JobQueue::default();
        let limit = Duration::from_secs(30);

        // A job is in flight when another is waited for: its completion arrives
        // first and is kept for whoever takes completions.
        let started = queue.start(Job::new(), true);
        assert!(queue.block_on(Job::new(), limit)?.check().is_ok());
        let (job, completion) = next(&queue);
        assert_eq!(job, started);
        assert!(completion.check().is_ok());

        // Draining returns once the builder thread performed what it was given.
        queue.drain(limit)
    }
}
