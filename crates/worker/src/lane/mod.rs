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
//! A forward runs as jobs of its lane ([`JobQueues::submit`]). A [`Job`] is
//! the ordered list of what the forward puts on the stream; the lane's
//! execution thread performs it: it writes the forward's inputs, launches
//! its graph or hands its Python call to the Python thread and waits for
//! it, and reads its outputs, so the stream's order is the thread's order.
//! The thread posts a [`Completion`] per job under the job's [`JobId`], to
//! the one place the control thread takes completions from, and fires the
//! control thread's wake behind it. Several jobs of a lane may be queued:
//! their completions arrive in the lane's order.
//!
//! Every device also has a preparation stream for startup work outside any
//! lane. Capability runtimes fork module and microbatch streams from these
//! and register them here so they close in dependency order: preparation,
//! modules, microbatches, lanes, each newest first.

mod job;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use crossbeam_channel::{Receiver, Sender};
use uniserve_cuda::DeviceId;
use uniserve_ipc::{CallKind, LaneConfig, Wake};

use crate::config::{Device, WorkerConfig};
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

/// The job queues of a rank's lanes and the completions of their jobs: the
/// control thread's side of the lanes.
#[derive(Clone)]
pub struct JobQueues {
    shared: Arc<Shared>,
}

struct Shared {
    /// Each lane's served call kinds (`None` serves every kind) and queue.
    queues: Vec<(Option<Vec<CallKind>>, JobQueue)>,
    /// Where every lane posts its completions.
    completions: Receiver<(JobId, Completion)>,
    /// The name of the next job.
    next: AtomicU64,
    /// Completions taken while another job was waited for, and those of
    /// jobs performed on the submitting thread, in arrival order.
    arrived: Mutex<VecDeque<(JobId, Completion)>>,
}

impl Default for JobQueues {
    /// The queues of a rank without lanes: every job is performed on the
    /// thread that submits it.
    fn default() -> Self {
        let (_, completions) = crossbeam_channel::unbounded();
        Self::new(Vec::new(), completions)
    }
}

impl JobQueues {
    fn new(
        queues: Vec<(Option<Vec<CallKind>>, JobQueue)>,
        completions: Receiver<(JobId, Completion)>,
    ) -> Self {
        Self {
            shared: Arc::new(Shared {
                queues,
                completions,
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

    /// Submit a job of a forward of `kind` to the lane serving the kind and
    /// wait for its completion. A forward no lane serves (every forward of
    /// a rank without a CUDA device) has no stream, and its job is
    /// performed on the calling thread.
    pub(crate) fn submit(&self, kind: CallKind, job: Job<'_>) -> Result<(JobId, Completion)> {
        // SAFETY: the lifetime is erased only for the queue's type. This
        // call returns once the execution thread has consumed the job and
        // posted its completion, or with the job dropped where the thread
        // is gone, so nothing the job borrows is used after it.
        //
        // Only `Job::forward` gives a job something to borrow: the erasure
        // goes with it, and a job that owns its operations is started as
        // it is.
        let job = unsafe { std::mem::transmute::<Job<'_>, Job<'static>>(job) };
        let id = self.queue(kind, job, true)?;
        Ok((id, self.wait(id)?))
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
    /// completion arrives under the returned name, with the control
    /// thread's wake behind it.
    pub(crate) fn start(&self, kind: CallKind, job: Job<'static>) -> Result<JobId> {
        self.queue(kind, job, false)
    }

    /// Name `job` and queue it on the lane serving `kind`; `awaited` says
    /// that the caller blocks on its completion.
    fn queue(&self, kind: CallKind, job: Job<'static>, awaited: bool) -> Result<JobId> {
        let id = JobId(self.shared.next.fetch_add(1, Ordering::Relaxed));
        match self.lane(kind) {
            Some(lane) => self.shared.queues[lane].1.start(id, job, awaited)?,
            None => {
                let completion = job.perform();
                self.arrived().push_back((id, completion));
            }
        }
        Ok(id)
    }

    /// The next completion that arrived, without waiting: one kept while
    /// another job was waited for, or one a lane posted since.
    pub(crate) fn take(&self) -> Option<(JobId, Completion)> {
        let kept = self.arrived().pop_front();
        kept.or_else(|| self.shared.completions.try_recv().ok())
    }

    /// The completion of job `id`, waited for.
    fn wait(&self, id: JobId) -> Result<Completion> {
        loop {
            let mut arrived = self.arrived();
            if let Some(position) = arrived.iter().position(|(job, _)| *job == id) {
                let (_, completion) = arrived.remove(position).ok_or(LOST)?;
                return Ok(completion);
            }
            drop(arrived);

            // FIXME: the control thread waits here for as long as the lane
            // takes, as it waited for a forward before jobs: a lane that
            // stops making progress stops the control thread with it. The
            // control thread is to return after `start` and take the
            // completion when its wake descriptor fires (specs/planes.md
            // section 3).
            let completion = self.shared.completions.recv().map_err(|_| LOST)?;
            self.arrived().push_back(completion);
        }
    }

    /// Fire `wake` behind every completion from now on, so the control
    /// thread, which blocks on its socket and its wake descriptor, learns
    /// of a completion it does not wait for.
    pub(crate) fn set_wake(&self, wake: Option<Wake>) {
        for (_, jobs) in &self.shared.queues {
            // A lane that is gone posts nothing to be woken for; its next
            // job reports it.
            let _ = jobs.set_wake(wake.clone());
        }
    }
}

const LOST: Error = Error::State("the rank's execution threads have exited");

/// The rank's lanes and streams.
pub struct Lanes {
    /// The configured lanes, or the one lane of a rank without them.
    lanes: Vec<Lane>,
    /// Where the lanes post their completions; the rank's job queues take
    /// them from here.
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
        // Each lane's thread holds a sender, and nothing else does: when
        // the last lane is gone, whoever waits for a completion learns it.
        let (posted, completions) = crossbeam_channel::unbounded();
        let mut lanes = Self {
            lanes: Vec::new(),
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
        JobQueues::new(queues, self.completions.clone())
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
    /// before returning the first failure. No job is in flight: its
    /// submitter waits for each job.
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

    const DECODE: CallKind = CallKind::Forward(ForwardMode::Decode);

    /// The queues of one lane that serves every kind, and its thread.
    fn lane() -> (JobQueues, std::thread::JoinHandle<std::thread::ThreadId>) {
        let (posted, completions) = crossbeam_channel::unbounded();
        let (queue, jobs) = JobQueue::new(posted);
        let thread = std::thread::spawn(move || {
            jobs.run();
            std::thread::current().id()
        });
        (JobQueues::new(vec![(None, queue)], completions), thread)
    }

    #[test]
    fn a_lane_performs_jobs_in_order_on_its_thread_and_a_forward_borrows_its_submitter() {
        let (queues, thread) = lane();
        let here = std::thread::current().id();

        // A forward mutates state borrowed from its submitter; the next
        // job runs after it.
        let mut order = Vec::new();
        let mut threads = Vec::new();
        for name in ["first", "second"] {
            let job = Job::new();
            job.forward(|| {
                order.push(name);
                threads.push(std::thread::current().id());
            });
            assert!(queues.submit(DECODE, job).unwrap().1.check().is_ok());
        }
        assert_eq!(order, ["first", "second"]);
        assert_ne!(threads[0], here);
        assert_eq!(threads[1], threads[0]);

        // The thread ends with its last queue.
        drop(queues);
        assert_eq!(thread.join().unwrap(), threads[0]);
    }

    #[test]
    fn completions_name_their_jobs_whatever_order_they_are_waited_for() -> Result<()> {
        let (queues, _thread) = lane();
        let values = [7_u8, 9];

        // Two jobs are in flight on the lane before either is waited for.
        let mut ids = Vec::new();
        for value in &values {
            let job = Job::new();
            job.on(None).read(std::ptr::from_ref(value) as u64, 1);
            ids.push(queues.start(DECODE, job)?);
        }
        assert_ne!(ids[0], ids[1]);

        // The later job's completion is taken first; the earlier one's
        // arrived meanwhile and is kept for its own wait.
        assert_eq!(queues.wait(ids[1])?.read()?, vec![9]);
        assert_eq!(queues.wait(ids[0])?.read()?, vec![7]);
        Ok(())
    }

    #[test]
    fn a_job_no_lane_serves_is_performed_by_its_submitter() -> Result<()> {
        let queues = JobQueues::default();
        let here = std::thread::current().id();
        let mut ran = None;
        let job = Job::new();
        job.forward(|| ran = Some(std::thread::current().id()));
        let (_, completion) = queues.submit(DECODE, job)?;
        completion.check().map_err(|failure| failure.error)?;
        assert_eq!(ran, Some(here));
        Ok(())
    }
}
