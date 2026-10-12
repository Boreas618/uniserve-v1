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
//! The thread posts a [`Completion`] per job.
//!
//! Every device also has a preparation stream for startup work outside any
//! lane. Capability runtimes fork module and microbatch streams from these
//! and register them here so they close in dependency order: preparation,
//! modules, microbatches, lanes, each newest first.

mod job;

use uniserve_cuda::DeviceId;
use uniserve_ipc::{CallKind, LaneConfig};

use crate::config::{Device, WorkerConfig};
use crate::stream::CUDAStream;
use crate::{Error, Result};

pub use job::JobQueue;
pub(crate) use job::{Capture, Completion, ExpertStep, ExpertVariant, Job, Recorder};

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
    fn start(index: usize, config: Option<LaneConfig>, stream: CUDAStream) -> Result<Self> {
        let (jobs, queue) = JobQueue::new();
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

/// The job queues of a rank's lanes.
#[derive(Clone, Default)]
pub struct JobQueues {
    /// Each lane's served call kinds (`None` serves every kind) and queue.
    queues: Vec<(Option<Vec<CallKind>>, JobQueue)>,
}

impl JobQueues {
    /// Submit a job of a forward of `kind` to the lane serving the kind and
    /// wait for its completion. A forward no lane serves (every forward of
    /// a rank without a CUDA device) has no stream, and its job is
    /// performed on the calling thread.
    pub(crate) fn submit(&self, kind: CallKind, job: Job<'_>) -> Result<Completion> {
        let lane = self
            .queues
            .iter()
            .find(|(kinds, _)| kinds.as_ref().is_none_or(|kinds| kinds.contains(&kind)));
        match lane {
            Some((_, jobs)) => jobs.submit(job),
            None => Ok(job.perform()),
        }
    }
}

/// The rank's lanes and streams.
#[derive(Default)]
pub struct Lanes {
    /// The configured lanes, or the one lane of a rank without them.
    lanes: Vec<Lane>,
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
        let mut lanes = Self::default();
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
                lanes.lanes.push(Lane::start(index, Some(config), stream)?);
            }
        } else {
            // The primary device leads `devices`.
            for (index, &device) in cuda.iter().enumerate() {
                let stream = CUDAStream::create(device)?;
                if index == 0 {
                    lanes.lanes.push(Lane::start(0, None, stream)?);
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
        JobQueues {
            queues: self
                .lanes
                .iter()
                .map(|lane| {
                    let kinds = lane.config.as_ref().map(|config| config.call_kinds.clone());
                    (kinds, lane.jobs.clone())
                })
                .collect(),
        }
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
