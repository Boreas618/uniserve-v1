//! Execution lanes of a rank: its streams and their execution threads.
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
//! Every device also has a preparation stream for startup work outside any
//! lane. Capability runtimes fork module and microbatch streams from these
//! and register them here so they close in dependency order: preparation,
//! modules, microbatches, lanes, each newest first.

mod job;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use uniserve_cuda::DeviceId;
use uniserve_ipc::{CallKind, LaneConfig, Wake};

use crate::config::{Device, WorkerConfig};
use crate::ffi::Interpreter;
use crate::stream::CUDAStream;
use crate::{Error, Result};

pub(crate) use job::{Capture, ExpertStep, ExpertVariant, Job, Recorder};
pub use job::{Completion, JobId, JobQueue};

/// One execution lane.
pub struct Lane {
    /// The lane's SM partition and call kinds; `None` for the one lane of a
    /// rank without configured lanes, which serves every call kind.
    pub config: Option<LaneConfig>,
    pub stream: CUDAStream,
    jobs: JobQueue,
}

impl Lane {
    /// Start the execution thread of `stream`, named `lane-{index}`.
    fn start(
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
            .map_err(|error| Error::Transport(format!("execution thread: {error}")))?;

        Ok(Self {
            config,
            stream,
            jobs,
        })
    }

    /// Whether the lane serves calls of `kind`.
    pub fn serves(&self, kind: CallKind) -> bool {
        self.config
            .as_ref()
            .is_none_or(|config| config.call_kinds.contains(&kind))
    }
}

/// The job queues of a rank's lanes, the way to its Python thread and the
/// completions of what was submitted to either: the control thread's side
/// of the execution and resource planes.
#[derive(Clone)]
pub struct JobQueues {
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
}

impl Default for JobQueues {
    /// The queues of a rank without lanes: every job is performed on the
    /// thread that submits it.
    fn default() -> Self {
        let (posted, completions) = crossbeam_channel::unbounded();
        Self::new(Vec::new(), posted, completions)
    }
}

impl JobQueues {
    fn new(
        queues: Vec<(Option<Vec<CallKind>>, JobQueue)>,
        posted: Sender<(JobId, Completion)>,
        completions: Receiver<(JobId, Completion)>,
    ) -> Self {
        Self {
            shared: Arc::new(Shared {
                queues,
                completions,
                posted,
                wake: Mutex::new(None),
                next: AtomicU64::new(0),
                arrived: Mutex::new(VecDeque::new()),
            }),
        }
    }

    fn arrived(&self) -> std::sync::MutexGuard<'_, VecDeque<(JobId, Completion)>> {
        self.shared
            .arrived
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
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
                self.shared.queues[lane].1.start(id, job, wake)?;
                Ok(id)
            }
            None => Ok(self.call(job, wake)),
        }
    }

    /// Start `job` on the lane serving `kind` and wait on the calling
    /// thread for its completion, at most `timeout`.
    ///
    /// For a rank that is not serving only: startup, whose captures need
    /// each job's result before the next is prepared, and shutdown. While
    /// the rank serves, the control thread starts a job and takes its
    /// completion ([`Self::start`], [`Self::take`]). Completions of other
    /// jobs that arrive meanwhile are kept for [`Self::take`].
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
    /// Like [`Self::block_on`], for a rank that is not serving.
    pub(crate) fn drain(&self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout;
        let mut last = Vec::new();
        for (_, lane) in &self.shared.queues {
            let id = self.name();
            lane.start(id, Job::new(), false)?;
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
            match self.shared.completions.recv_deadline(deadline) {
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
        let wake = if wake { self.wake().clone() } else { None };
        let post = job::Post::new(id, self.shared.posted.clone(), wake);
        Interpreter::submit(move || post.now(calls.perform()));
        id
    }

    fn wake(&self) -> std::sync::MutexGuard<'_, Option<Wake>> {
        self.shared
            .wake
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// The name of the next job.
    fn name(&self) -> JobId {
        JobId(self.shared.next.fetch_add(1, Ordering::Relaxed))
    }

    /// The next completion that arrived, without waiting: one kept while
    /// another job was waited for, or one posted since.
    pub(crate) fn take(&self) -> Option<(JobId, Completion)> {
        let kept = self.arrived().pop_front();
        kept.or_else(|| self.shared.completions.try_recv().ok())
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

/// The rank's lanes and streams.
pub struct Lanes {
    /// The configured lanes, or the one lane of a rank without them.
    lanes: Vec<Lane>,
    /// Where the lanes post their completions; the rank's job queues take
    /// them from here, and give the posting end to the Python thread's
    /// work.
    posted: Sender<(JobId, Completion)>,
    completions: Receiver<(JobId, Completion)>,
    /// The full-device streams of the other CUDA devices of a rank without
    /// configured lanes.
    devices: Vec<CUDAStream>,
    preparation: Vec<CUDAStream>,
    modules: Vec<CUDAStream>,
    microbatches: Vec<CUDAStream>,
}

impl Lanes {
    /// Create the streams of `devices`.
    pub fn create(config: &WorkerConfig, devices: &[Device]) -> Result<Self> {
        let cuda: Vec<DeviceId> = devices
            .iter()
            .filter_map(|device| device.cuda_index().map(DeviceId))
            .collect();
        let (posted, completions) = crossbeam_channel::unbounded();
        let mut lanes = Self {
            lanes: Vec::new(),
            posted: posted.clone(),
            completions,
            devices: Vec::new(),
            preparation: Vec::new(),
            modules: Vec::new(),
            microbatches: Vec::new(),
        };
        if cuda.is_empty() {
            return Ok(lanes);
        }
        if !config.lanes.is_empty() {
            let [device] = cuda[..] else {
                return Err(Error::Unsupported(
                    "execution lanes require one physical device".into(),
                ));
            };
            if config.expert_microbatches > 1 && config.lanes.len() > 1 {
                return Err(Error::Unsupported(
                    "expert parallelism runs on exactly one execution lane".into(),
                ));
            }
            let counts: Vec<u32> = config
                .lanes
                .iter()
                .map(|lane| u32::try_from(lane.sm_budget).unwrap_or(u32::MAX))
                .collect();
            let streams = CUDAStream::partition(device, &counts)?;
            for (index, (config, stream)) in config.lanes.iter().cloned().zip(streams).enumerate() {
                lanes
                    .lanes
                    .push(Lane::start(index, Some(config), stream, posted.clone())?);
            }
        } else {
            // The primary device leads `devices`.
            for (index, &device) in cuda.iter().enumerate() {
                let stream = CUDAStream::create(device)?;
                if index == 0 {
                    lanes
                        .lanes
                        .push(Lane::start(0, None, stream, posted.clone())?);
                } else {
                    lanes.devices.push(stream);
                }
            }
        }
        for &device in &cuda {
            lanes.preparation.push(CUDAStream::create(device)?);
        }
        Ok(lanes)
    }

    /// The configured lanes; empty for a rank without them.
    pub fn lanes(&self) -> &[Lane] {
        if self.configured() { &self.lanes } else { &[] }
    }

    /// Whether the lanes are configured SM partitions.
    fn configured(&self) -> bool {
        self.lanes.first().is_some_and(|lane| lane.config.is_some())
    }

    /// The job queue of every lane with the call kinds it serves, for the
    /// executor, which submits each forward to its lane.
    pub fn job_queues(&self) -> JobQueues {
        let queues = self
            .lanes
            .iter()
            .map(|lane| {
                let kinds = lane.config.as_ref().map(|config| config.call_kinds.clone());
                (kinds, lane.jobs.clone())
            })
            .collect();
        JobQueues::new(queues, self.posted.clone(), self.completions.clone())
    }

    /// The stream of the lane serving `kind`, or `device`'s full-device
    /// stream without configured lanes.
    pub fn stream_for(&self, kind: CallKind, device: DeviceId) -> Option<&CUDAStream> {
        if self.configured() {
            return self
                .lanes
                .iter()
                .find(|lane| lane.serves(kind))
                .map(|lane| &lane.stream);
        }
        self.full_device_streams()
            .find(|stream| stream.device() == device)
    }

    /// The full-device streams of a rank without configured lanes.
    fn full_device_streams(&self) -> impl Iterator<Item = &CUDAStream> {
        self.lanes
            .iter()
            .filter(|_| !self.configured())
            .map(|lane| &lane.stream)
            .chain(&self.devices)
    }

    /// The startup preparation stream of `device`.
    pub fn preparation(&self, device: DeviceId) -> Option<&CUDAStream> {
        self.preparation
            .iter()
            .find(|stream| stream.device() == device)
    }

    /// Fork the stream serving `kind` for a module that runs beside it.
    /// With lanes, exactly one lane must serve the kind.
    pub fn fork_module(&mut self, kind: CallKind, device: DeviceId) -> Result<&CUDAStream> {
        let matching: Vec<&CUDAStream> = if self.configured() {
            self.lanes
                .iter()
                .filter(|lane| lane.serves(kind))
                .map(|lane| &lane.stream)
                .collect()
        } else {
            self.full_device_streams()
                .filter(|stream| stream.device() == device)
                .collect()
        };
        let [origin] = matching[..] else {
            return Err(Error::Invalid(format!(
                "{} is served by {} execution lanes; a module stream needs exactly one",
                kind.as_str(),
                matching.len()
            )));
        };
        let stream = origin.fork()?;
        self.modules.push(stream);
        self.modules
            .last()
            .ok_or(Error::State("module stream was not registered"))
    }

    /// Fork the first lane (or device stream) for one more microbatch.
    pub fn fork_microbatch(&mut self) -> Result<&CUDAStream> {
        let origin = self
            .lanes
            .first()
            .map(|lane| &lane.stream)
            .ok_or(Error::State("a microbatch stream requires a CUDA lane"))?;
        let stream = origin.fork()?;
        self.microbatches.push(stream);
        self.microbatches
            .last()
            .ok_or(Error::State("microbatch stream was not registered"))
    }

    /// The owned stream whose handle is `handle`.
    pub fn owner(&self, handle: usize) -> Option<&CUDAStream> {
        self.streams()
            .find(|stream| stream.handle().is_ok_and(|raw| raw == handle))
    }

    /// Every owned stream, for synchronization before teardown.
    pub fn streams(&self) -> impl Iterator<Item = &CUDAStream> {
        self.preparation
            .iter()
            .chain(&self.modules)
            .chain(&self.microbatches)
            .chain(self.lanes.iter().map(|lane| &lane.stream))
            .chain(&self.devices)
    }

    /// Close the streams in dependency order: preparation, modules,
    /// microbatches, then lanes, each newest first. Attempts every stream
    /// before returning the first failure. No job is in flight unless the
    /// close is aborted: the worker drained the lanes
    /// ([`JobQueues::drain`]).
    pub fn close(&mut self, aborted: bool) -> Result<()> {
        let mut failure = None;
        let mut close = |stream: &mut CUDAStream| {
            if let Err(error) = stream.close(aborted) {
                failure.get_or_insert(error);
            }
        };
        self.preparation.iter_mut().rev().for_each(&mut close);
        self.modules.iter_mut().rev().for_each(&mut close);
        self.microbatches.iter_mut().rev().for_each(&mut close);
        self.lanes
            .iter_mut()
            .rev()
            .for_each(|lane| close(&mut lane.stream));
        self.devices.iter_mut().rev().for_each(&mut close);
        failure.map_or(Ok(()), Err)
    }
}

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
        let queues = JobQueues::new(vec![(None, queue)], posted, completions);
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
            let mut notes = notes.lock().unwrap_or_else(PoisonError::into_inner);
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
        let notes = notes.lock().unwrap_or_else(PoisonError::into_inner).clone();
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
        let notes = notes.lock().unwrap_or_else(PoisonError::into_inner);
        assert_eq!(notes[..], [("job", here)]);
        Ok(())
    }
}
