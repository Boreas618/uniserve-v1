//! Execution lanes of a rank and the rank's job queues.
//!
//! A [`Lane`] is one stream, the execution thread that issues the stream's
//! work, and that thread's job queue. Configured lanes partition one
//! device's SMs (green contexts), each serving the call kinds its
//! configuration names. A rank without configured lanes has one lane: the
//! full-device stream of its primary device, serving every call kind; a
//! second device (a generation tower) has a full-device stream that
//! forwards of that lane also submit to.
//!
//! A forward runs as jobs of its lane ([`JobQueues::start`]). A [`Job`] is
//! the ordered list of what the forward puts on the stream; the lane's
//! execution thread performs it: it writes the forward's inputs, launches
//! its graph or hands its Python call to the Python thread and waits for
//! it, and reads its outputs, so the stream's order is the thread's order.
//! The thread posts a [`Completion`] per job under the job's [`JobId`], to
//! the one place the control thread takes completions from
//! ([`JobQueues::take`]), and fires the control thread's wake behind it.
//! Several jobs of a lane may be queued: their completions arrive in the
//! lane's order.
//!
//! Work for the resource plane takes the same way to the Python thread
//! ([`JobQueues::call`]): calls that issue nothing on a stream are submitted
//! to it under a [`JobId`], and their [`Completion`] is posted to the same
//! place. So is a job no lane serves, which has no stream: its memory
//! writes and calls are performed by the Python thread.
//!
//! While it serves, the control thread waits for neither: it starts work
//! and takes completions when its wake fires. Before the rank serves and
//! after it stopped, a caller may wait for one job, with a deadline
//! ([`JobQueues::block_on`], [`JobQueues::drain`]).
//!
//! Because every job is submitted and every completion taken here, the
//! queues know which jobs each lane and the Python thread have been given
//! and not completed, and since when ([`JobQueues::oldest`]): a lane that
//! stays inside one job, such as a driver call that does not return, is
//! seen from the control thread, which reports it to the head.
//!
//! The lanes' streams belong to the rank's [`Streams`](crate::stream::Streams),
//! which starts the lanes and closes every stream in dependency order.

mod job;
mod queue;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use uniserve_cuda::{DeviceId, Event};
use uniserve_ipc::{CallKind, LaneConfig, Wake};

use crate::ffi::Interpreter;
use crate::stream::CUDAStream;
use crate::{Error, Result, lock};

pub(crate) use job::Completion;
pub(crate) use job::{Capture, ExpertStep, ExpertVariant, Job, Recorder, Value};
pub(crate) use queue::{JobId, JobQueue};

/// One execution lane.
pub(crate) struct Lane {
    /// The lane's SM partition and call kinds; `None` for the one lane of a
    /// rank without configured lanes, which serves every call kind.
    pub config: Option<LaneConfig>,
    pub stream: CUDAStream,
    jobs: JobQueue,
}

impl Lane {
    /// Start the execution thread of `stream`, named `lane-{index}`.
    pub(crate) fn start(
        index: usize,
        config: Option<LaneConfig>,
        stream: CUDAStream,
        completions: Sender<(JobId, Completion)>,
    ) -> Result<Self> {
        let (jobs, queue) = JobQueue::new(completions);
        // The thread ends when the lane and every clone of its queue are
        // gone.
        std::thread::Builder::new()
            .name(format!("lane-{index}"))
            .spawn(move || queue.run())
            .map_err(|error| Error::transport_source("execution thread", error))?;

        Ok(Self {
            config,
            stream,
            jobs,
        })
    }

    /// Whether the lane serves calls of `kind`.
    pub(crate) fn serves(&self, kind: CallKind) -> bool {
        self.config
            .as_ref()
            .is_none_or(|config| config.call_kinds.contains(&kind))
    }
}

impl JobQueues {
    /// The job queues of `lanes`, whose threads post to `posted`; the
    /// control thread takes completions from `completions`.
    pub(crate) fn of(
        lanes: &[Lane],
        posted: Sender<(JobId, Completion)>,
        completions: Receiver<(JobId, Completion)>,
    ) -> Self {
        let queues = lanes
            .iter()
            .map(|lane| {
                let kinds = lane.config.as_ref().map(|config| config.call_kinds.clone());
                (kinds, lane.jobs.clone())
            })
            .collect();
        // A configured lane is named by its id; the one lane of a rank
        // without configured lanes by its thread's name.
        let names = lanes
            .iter()
            .enumerate()
            .map(|(index, lane)| {
                lane.config
                    .as_ref()
                    .map_or_else(|| format!("lane-{index}"), |config| config.lane_id.clone())
            })
            .collect();
        Self::new(queues, names, posted, completions)
    }
}

/// The job queues of a rank's lanes, the way to its Python thread and the
/// completions of what was submitted to either: the control thread's side
/// of the execution and resource planes.
#[derive(Clone)]
pub(crate) struct JobQueues {
    shared: Arc<Shared>,
}

struct Shared {
    /// Each lane's served call kinds (`None` serves every kind) and queue.
    queues: Vec<(Option<Vec<CallKind>>, JobQueue)>,
    /// Where every lane and the Python thread post their completions.
    completions: Receiver<(JobId, Completion)>,
    /// The posting end the Python thread's work is given.
    posted: Sender<(JobId, Completion)>,
    /// The control thread's wake, fired behind a completion the Python
    /// thread posts. Set and read on the control thread.
    wake: Mutex<Option<Wake>>,
    /// The name of the next job.
    next: AtomicU64,
    /// Completions taken while another job was waited for, in arrival
    /// order.
    arrived: Mutex<VecDeque<(JobId, Completion)>>,
    /// What each lane, by position, and then the Python thread have been
    /// given and not completed. Changed only by the thread that submits
    /// jobs and takes completions.
    running: Mutex<Vec<Running>>,
}

/// The jobs one performer, a lane's execution thread or the Python thread,
/// has been given and not completed, in submission order. A performer
/// performs its jobs in that order, so the first is the one it is
/// performing; a job whose completion waits for device events
/// ([`Job::after`]) leaves its lane at once and may complete after the
/// jobs behind it.
struct Running {
    /// The lane's name in a stall report; `None` for the Python thread.
    lane: Option<String>,
    /// Each job, when it was submitted and the runner method it calls.
    jobs: VecDeque<(JobId, Instant, Option<String>)>,
    /// When the last completion of one of its jobs arrived.
    completed: Option<Instant>,
}

impl Running {
    fn new(lane: Option<String>) -> Self {
        Self {
            lane,
            jobs: VecDeque::new(),
            completed: None,
        }
    }
}

impl Default for JobQueues {
    /// The queues of a rank without lanes: every job is performed on the
    /// thread that submits it.
    fn default() -> Self {
        let (posted, completions) = crossbeam_channel::unbounded();
        Self::new(Vec::new(), Vec::new(), posted, completions)
    }
}

impl JobQueues {
    /// The queues `queues` of the lanes named `names`, in the same order.
    fn new(
        queues: Vec<(Option<Vec<CallKind>>, JobQueue)>,
        names: Vec<String>,
        posted: Sender<(JobId, Completion)>,
        completions: Receiver<(JobId, Completion)>,
    ) -> Self {
        let running = names
            .into_iter()
            .map(|name| Running::new(Some(name)))
            .chain([Running::new(None)])
            .collect();
        Self {
            shared: Arc::new(Shared {
                queues,
                completions,
                posted,
                wake: Mutex::new(None),
                next: AtomicU64::new(0),
                arrived: Mutex::new(VecDeque::new()),
                running: Mutex::new(running),
            }),
        }
    }

    fn arrived(&self) -> std::sync::MutexGuard<'_, VecDeque<(JobId, Completion)>> {
        lock(&self.shared.arrived)
    }

    fn running(&self) -> std::sync::MutexGuard<'_, Vec<Running>> {
        lock(&self.shared.running)
    }

    /// Note that job `id`, which calls `method`, was given to `performer`
    /// (a lane's position; the lanes' count for the Python thread) now.
    fn submitted(&self, performer: usize, id: JobId, method: Option<String>) {
        if let Some(running) = self.running().get_mut(performer) {
            running.jobs.push_back((id, Instant::now(), method));
        }
    }

    /// Note that the completion of job `id` arrived now.
    fn received(&self, id: JobId) {
        let now = Instant::now();
        for running in self.running().iter_mut() {
            if let Some(position) = running.jobs.iter().position(|(job, ..)| *job == id) {
                running.jobs.remove(position);
                running.completed = Some(now);
                return;
            }
        }
    }

    /// The job in flight that its performer, a lane or the Python thread,
    /// has been performing longest, and since when: of each performer's
    /// first job, timed from its submission or from the arrival of that
    /// performer's last completion, whichever is later, the earliest.
    /// `None` when no job is in flight.
    ///
    /// Only completions taken so far count, so a caller takes the ones
    /// that arrived before it asks.
    pub(crate) fn oldest(&self) -> Option<(JobId, Instant)> {
        self.running()
            .iter()
            .filter_map(|running| {
                let &(id, submitted, _) = running.jobs.front()?;
                let since = running
                    .completed
                    .map_or(submitted, |completed| completed.max(submitted));
                Some((id, since))
            })
            .min_by_key(|&(_, since)| since)
    }

    /// The lane performing job `id`, `None` for the Python thread, and the
    /// runner method the job calls; `None` when the job is not in flight.
    pub(crate) fn performing(&self, id: JobId) -> Option<(Option<String>, Option<String>)> {
        self.running().iter().find_map(|running| {
            running
                .jobs
                .iter()
                .find(|(job, ..)| *job == id)
                .map(|(_, _, method)| (running.lane.clone(), method.clone()))
        })
    }

    /// The lane serving `kind`, by its position among the rank's lanes;
    /// `None` when no lane serves it.
    pub(crate) fn lane(&self, kind: CallKind) -> Option<usize> {
        self.shared
            .queues
            .iter()
            .position(|(kinds, _)| kinds.as_ref().is_none_or(|kinds| kinds.contains(&kind)))
    }

    /// Queue `job` on the lane serving `kind` without waiting for it. Its
    /// completion arrives under the returned name. `wake` says whether the
    /// control thread's wake is fired behind it, or the completion is left
    /// to be taken when the control thread next takes one (nothing goes on
    /// from it). A job no lane serves (every job of a rank without a CUDA
    /// device) has no stream: it goes to the Python thread like the
    /// resource plane's calls ([`Self::call`]).
    pub(crate) fn start(&self, kind: CallKind, job: Job, wake: bool) -> Result<JobId> {
        match self.lane(kind) {
            Some(lane) => {
                let id = self.name();
                let method = job.method();
                self.shared.queues[lane].1.start(id, job, wake)?;
                self.submitted(lane, id, method);
                Ok(id)
            }
            None => Ok(self.call(job, wake)),
        }
    }

    /// Queue, on the lane serving `kind`, a job that holds no operation
    /// and completes once `events` completed ([`Job::after`]): the lane
    /// orders the completion behind them on its notification stream of
    /// `device` and goes on. The control thread's wake is fired behind it.
    /// The events' records are made, or queued in jobs submitted to the
    /// same lane before: a wait on an event without a record passes.
    pub(crate) fn after(
        &self,
        kind: CallKind,
        device: DeviceId,
        events: Vec<Arc<Event>>,
    ) -> Result<JobId> {
        let job = Job::new();
        job.after(device, events);
        self.start(kind, job, true)
    }

    /// Queue on lane `lane` a job that completes once `events` completed,
    /// as [`Self::after`] does on the lane serving a kind.
    pub(crate) fn after_on(
        &self,
        lane: usize,
        device: DeviceId,
        events: Vec<Arc<Event>>,
    ) -> Result<JobId> {
        let job = Job::new();
        job.after(device, events);
        let id = self.name();
        self.shared
            .queues
            .get(lane)
            .ok_or(Error::Invariant(
                "deferred work waits on an unknown lane".into(),
            ))?
            .1
            .start(id, job, true)?;
        self.submitted(lane, id, None);
        Ok(id)
    }

    /// Start `job` on the lane serving `kind` and wait on the calling
    /// thread for its completion, at most `timeout`.
    ///
    /// For a rank that is not serving only: startup, whose captures need
    /// each job's result before the next is prepared, and shutdown. While
    /// the rank serves, the control thread starts a job and takes its
    /// completion ([`Self::start`], [`Self::take`]). Completions of other
    /// jobs that arrive meanwhile are kept for [`Self::take`].
    ///
    /// A completion, here as anywhere, says that the lane performed the
    /// job: it issued the job's stream work in order, ran its calls and
    /// made its reads and waits. It does not say that the device finished
    /// the work issued. Memory the job's device work addresses may be read
    /// on the host, written again outside the stream's order or released
    /// after the completion only behind something that waited for the
    /// device: a read of the job (which waits for its stream), a wait for
    /// the stream the job recorded after that work
    /// ([`crate::stream::Issue::synchronize`] on its recorder), the events
    /// the job completes after ([`Job::after`]), or the fences that retire
    /// the storage ([`crate::EventPool`]).
    pub(crate) fn block_on(
        &self,
        kind: CallKind,
        job: Job,
        timeout: Duration,
    ) -> Result<Completion> {
        let id = self.start(kind, job, false)?;
        self.completion(id, Instant::now() + timeout)
    }

    /// Wait until every lane performed the jobs submitted to it so far, at
    /// most `timeout`: before the owners those jobs address are closed.
    /// Like [`Self::block_on`], for a rank that is not serving; performed is
    /// not finished, so the caller then synchronizes the streams.
    pub(crate) fn drain(&self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout;
        let mut last = Vec::new();
        for (index, (_, lane)) in self.shared.queues.iter().enumerate() {
            let id = self.name();
            lane.start(id, Job::new(), false)?;
            self.submitted(index, id, None);
            last.push(id);
        }
        for id in last {
            self.completion(id, deadline)?
                .check()
                .map_err(|failure| failure.error)?;
        }
        Ok(())
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

    /// Submit `calls`, a job of Python calls that issue nothing on a
    /// stream, to the Python thread without waiting for it. Its completion
    /// arrives under the returned name where the lanes' completions arrive;
    /// `wake` says whether the control thread's wake is fired behind it, or
    /// the completion is left to be taken when the control thread next
    /// takes one.
    pub(crate) fn call(&self, calls: Job, wake: bool) -> JobId {
        let id = self.name();
        let method = calls.method();
        let wake = if wake { self.wake().clone() } else { None };
        let post = queue::Post::new(id, self.shared.posted.clone(), wake);
        Interpreter::submit(move || post.now(calls.perform()));
        // The Python thread is the performer after the lanes.
        self.submitted(self.shared.queues.len(), id, method);
        id
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

    /// Wait on the calling thread, at most `timeout`, for the next
    /// completion of a job in flight, and keep it for [`Self::take`].
    /// Returns whether one arrived. For a rank that admits nothing more,
    /// whose control thread waits for the jobs it issued instead of
    /// serving: a job started without a wake ([`Self::start`]) completes
    /// unannounced.
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

    /// Fire `wake` behind every completion from now on, so the control
    /// thread, which blocks on its socket and its wake descriptor, learns
    /// of a completion it does not wait for.
    pub(crate) fn set_wake(&self, wake: Option<Wake>) {
        (*self.wake()).clone_from(&wake);
        for (_, jobs) in &self.shared.queues {
            // A lane that is gone posts nothing to be woken for; its next
            // job reports it.
            let _ = jobs.set_wake(wake.clone());
        }
    }
}

const LOST: Error = Error::State("the rank's execution threads have exited");

/// How long a caller that is not serving waits for one job
/// ([`JobQueues::block_on`], [`JobQueues::drain`]). A startup job may
/// compile kernels inside its first eager forward, which has taken minutes;
/// the bound is there so that a lane that stopped making progress fails the
/// rank instead of hanging it.
pub(crate) const BLOCK_ON_TIMEOUT: Duration = Duration::from_secs(1800);

#[cfg(test)]
mod tests {
    use uniserve_ipc::ForwardMode;

    use super::*;
    use crate::stream::Issue;

    const DECODE: CallKind = CallKind::Forward(ForwardMode::Decode);

    /// The queues of one lane that serves every kind, and its thread.
    fn lane() -> (JobQueues, std::thread::JoinHandle<std::thread::ThreadId>) {
        let (posted, completions) = crossbeam_channel::unbounded();
        let (queue, jobs) = JobQueue::new(posted.clone());
        let thread = std::thread::spawn(move || {
            jobs.run();
            std::thread::current().id()
        });
        let queues = JobQueues::new(
            vec![(None, queue)],
            vec!["lane-0".into()],
            posted,
            completions,
        );
        (queues, thread)
    }

    /// The next completion, which a lane or the test's own thread posts
    /// within the test's patience.
    fn next(queues: &JobQueues) -> (JobId, Completion) {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(completed) = queues.take() {
                return completed;
            }
            assert!(Instant::now() < deadline, "no completion arrived");
            std::thread::yield_now();
        }
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
    fn a_lane_performs_jobs_in_order_on_its_thread_while_their_submitter_goes_on() -> Result<()> {
        let (queues, thread) = lane();
        let here = std::thread::current().id();
        let notes = Arc::new(Mutex::new(Vec::new()));

        // Both jobs are in flight before either completion is taken; the
        // completions arrive under the jobs' names in the lane's order.
        let first = queues.start(DECODE, noting("first", &notes)?, true)?;
        let second = queues.start(DECODE, noting("second", &notes)?, true)?;
        assert_ne!(first, second);
        for id in [first, second] {
            let (job, completion) = next(&queues);
            assert_eq!(job, id);
            assert!(completion.check().is_ok());
        }
        let notes = lock(&notes).clone();
        assert_eq!([notes[0].0, notes[1].0], ["first", "second"]);
        assert_ne!(notes[0].1, here);
        assert_eq!(notes[1].1, notes[0].1);

        // The thread ends with its last queue.
        drop(queues);
        assert_eq!(thread.join().unwrap(), notes[0].1);
        Ok(())
    }

    #[test]
    fn a_job_that_makes_no_progress_does_not_hold_up_its_submitter() -> Result<()> {
        let (queues, _thread) = lane();
        let limit = Duration::from_secs(30);

        // The lane's thread stays inside this job until the gate opens.
        let (open, gate) = crossbeam_channel::bounded::<()>(1);
        let stuck = Job::new();
        stuck.on(None).launch_host_fn(Box::new(move || {
            let _ = gate.recv_timeout(limit);
        }))?;
        let stuck = queues.start(DECODE, stuck, true)?;

        // Its submitter goes on: it finds no completion, queues more work
        // behind the job, and has the Python thread's way open beside it.
        assert!(queues.take().is_none());
        let behind = queues.start(DECODE, Job::new(), true)?;
        let beside = queues.call(Job::new(), true);
        assert_eq!(next(&queues).0, beside);
        assert!(queues.take().is_none());

        // The lane goes on in its order once the job ends.
        open.send(())
            .map_err(|_| Error::State("the lane dropped its job"))?;
        assert_eq!(next(&queues).0, stuck);
        assert_eq!(next(&queues).0, behind);
        Ok(())
    }

    #[test]
    fn a_wait_before_serving_keeps_the_completions_of_other_jobs() -> Result<()> {
        let (queues, _thread) = lane();
        let values = [7_u8, 9];
        let read = |value: &u8| {
            let job = Job::new();
            job.on(None).read(std::ptr::from_ref(value) as u64, 1);
            job
        };

        // A job is in flight when another is waited for: its completion
        // arrives first and is kept for whoever takes completions.
        let started = queues.start(DECODE, read(&values[0]), true)?;
        let limit = Duration::from_secs(30);
        assert_eq!(
            queues.block_on(DECODE, read(&values[1]), limit)?.read()?,
            vec![9]
        );
        let (job, mut completion) = next(&queues);
        assert_eq!(job, started);
        assert_eq!(completion.read()?, vec![7]);

        // Draining returns once the lane performed what it was given.
        queues.drain(limit)
    }

    #[test]
    fn a_job_no_lane_serves_takes_the_way_of_the_python_threads_calls() -> Result<()> {
        // No interpreter is started in this process: work submitted to the
        // Python thread runs on its submitter's.
        let queues = JobQueues::default();
        let here = std::thread::current().id();
        let notes = Arc::new(Mutex::new(Vec::new()));
        let id = queues.start(DECODE, noting("job", &notes)?, true)?;

        let (job, completion) = next(&queues);
        assert_eq!(job, id);
        completion.check().map_err(|failure| failure.error)?;
        let notes = lock(&notes);
        assert_eq!(notes[..], [("job", here)]);
        Ok(())
    }
}
