//! Execution streams, the rank's set of them, and the events that order work
//! across them.

mod events;
mod issue;

pub(crate) use events::Deferral;
pub(crate) use events::Deferred;
pub use events::EventPool;
pub(crate) use issue::HostFunction;
pub use issue::{Fill, Issue, RowFill};

use crossbeam_channel::{Receiver, Sender};
use uniserve_cuda::{DeviceGuard, DeviceId, Event, EventFlags, Stream};
use uniserve_ipc::CallKind;

use crate::config::{Device, WorkerConfig};
use crate::lane::{Completion, JobId, JobQueues, Lane};
use crate::{Error, Result};

/// The stream a numerical binding names by its integer handle.
///
/// The worker gives every binding the handle of a stream the rank owns, and
/// the binding hands that integer back when it asks for stream work. This is
/// the one place the worker turns such an integer into a [`Stream`]; code
/// that holds a `Stream` shares it by cloning the handle instead.
///
/// The legacy default stream (handle 0) names no context of its own: the
/// driver resolves it against the calling thread's current context, here
/// and in every later use of the returned handle.
///
/// # Safety
/// `handle` is a live stream on `device`. Its owner keeps the stream and its
/// context alive while the returned handle, its clones and the events
/// recorded on it are used.
pub(crate) unsafe fn external(device: DeviceId, handle: usize) -> Result<Stream> {
    // SAFETY: forwarded from the caller.
    Ok(unsafe { Stream::borrowed(device, handle) }?)
}

/// An execution stream and its SM allocation.
///
/// Callers serialize context use across streams sharing an SM partition.
/// Forks independently retain an owned SM partition.
pub struct CUDAStream {
    device: DeviceId,
    stream: Option<Stream>,
    /// The fence [`Self::wait`] orders this stream after a producer with.
    /// One event serves every wait: a wait captures the record made just
    /// before it, and exclusive access keeps two waits from interleaving.
    ingress: Event,
}

impl CUDAStream {
    fn new(device: DeviceId, stream: Stream) -> Self {
        Self {
            device,
            stream: Some(stream),
            ingress: Event::new(EventFlags::default()),
        }
    }

    /// Own a new non-blocking stream on `device`'s full device, in its
    /// primary context.
    pub fn create(device: DeviceId) -> Result<Self> {
        Ok(Self::new(device, Stream::new(device)?))
    }

    /// Allocate the configured SM partitions from one disjoint split tree.
    pub(crate) fn partition(device: DeviceId, counts: &[u32]) -> Result<Vec<Self>> {
        Ok(Stream::partition(device, counts)?
            .into_iter()
            .map(|stream| Self::new(device, stream))
            .collect())
    }

    pub(crate) fn fork(&self) -> Result<Self> {
        Ok(Self::new(self.device, self.stream()?.fork()?))
    }

    pub(crate) fn device(&self) -> DeviceId {
        self.device
    }

    pub fn handle(&self) -> Result<usize> {
        Ok(self.stream()?.raw())
    }

    pub fn stream(&self) -> Result<&Stream> {
        self.stream
            .as_ref()
            .ok_or(Error::State("CUDA stream is closed"))
    }

    /// Order this stream's later work after the work `producer` holds now,
    /// without waiting for the producer on the host.
    ///
    /// The wait takes the stream exclusively: it records the producer on
    /// the stream's one ingress fence and enqueues the wait on that record,
    /// and a second wait recording between the two would replace the
    /// producer this one waits for.
    pub(crate) fn wait(&mut self, producer: &Stream) -> Result<()> {
        let stream = self.stream()?;
        if stream.raw() == producer.raw() {
            return Ok(());
        }

        // A default-stream producer records in the thread's current context.
        let _device = DeviceGuard::new(self.device)?;
        self.ingress.record(producer)?;
        stream.wait_event(&self.ingress)?;
        Ok(())
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream()?.synchronize()?;
        Ok(())
    }

    /// Drain submitted work, then release this owner's handle to the stream.
    /// The driver stream and its SM partition are released once every
    /// component sharing the handle has dropped its clone. Failed collective
    /// execution retains resources until process exit.
    #[allow(clippy::mem_forget)]
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        // A closed stream keeps an unrecorded fence, which holds no driver
        // event.
        let unrecorded = Event::new(EventFlags::default());
        if aborted {
            std::mem::forget(self.stream.take());
            std::mem::forget(std::mem::replace(&mut self.ingress, unrecorded));
            return Ok(());
        }

        if let Some(stream) = &self.stream {
            stream.synchronize()?;
        }

        self.ingress = unrecorded;
        self.stream = None;
        Ok(())
    }
}

impl Drop for CUDAStream {
    fn drop(&mut self) {
        if self.close(false).is_err() {
            let _ = self.close(true);
        }
    }
}

/// Every stream of a rank: its lanes', each device's preparation stream for
/// startup work outside any lane, the full-device streams of its other
/// devices, and the module and microbatch streams capability runtimes fork
/// from a lane. They close in dependency order: preparation, modules,
/// microbatches, lanes, devices, each newest first.
pub(crate) struct Streams {
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

impl Streams {
    /// Create the streams of `devices`.
    pub(crate) fn create(config: &WorkerConfig, devices: &[Device]) -> Result<Self> {
        let cuda: Vec<DeviceId> = devices.iter().filter_map(|device| device.cuda()).collect();
        let (posted, completions) = crossbeam_channel::unbounded();
        let mut streams = Self {
            lanes: Vec::new(),
            posted: posted.clone(),
            completions,
            devices: Vec::new(),
            preparation: Vec::new(),
            modules: Vec::new(),
            microbatches: Vec::new(),
        };
        if cuda.is_empty() {
            return Ok(streams);
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
            let partitions = CUDAStream::partition(device, &counts)?;
            for (index, (config, stream)) in
                config.lanes.iter().cloned().zip(partitions).enumerate()
            {
                streams
                    .lanes
                    .push(Lane::start(index, Some(config), stream, posted.clone())?);
            }
        } else {
            // The primary device leads `devices`.
            for (index, &device) in cuda.iter().enumerate() {
                let stream = CUDAStream::create(device)?;
                if index == 0 {
                    streams
                        .lanes
                        .push(Lane::start(0, None, stream, posted.clone())?);
                } else {
                    streams.devices.push(stream);
                }
            }
        }
        for &device in &cuda {
            streams.preparation.push(CUDAStream::create(device)?);
        }
        Ok(streams)
    }

    /// The configured lanes; empty for a rank without them.
    pub(crate) fn lanes(&self) -> &[Lane] {
        if self.configured() { &self.lanes } else { &[] }
    }

    /// Whether the lanes are configured SM partitions.
    fn configured(&self) -> bool {
        self.lanes.first().is_some_and(|lane| lane.config.is_some())
    }

    /// The job queue of every lane with the call kinds it serves, for the
    /// executor, which submits each forward to its lane.
    pub(crate) fn job_queues(&self) -> JobQueues {
        JobQueues::of(&self.lanes, self.posted.clone(), self.completions.clone())
    }

    /// The stream of the lane serving `kind`, or `device`'s full-device
    /// stream without configured lanes.
    pub(crate) fn stream_for(&self, kind: CallKind, device: DeviceId) -> Option<&CUDAStream> {
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
    pub(crate) fn preparation(&self, device: DeviceId) -> Option<&CUDAStream> {
        self.preparation
            .iter()
            .find(|stream| stream.device() == device)
    }

    /// Fork the stream serving `kind` for a module that runs beside it.
    /// With lanes, exactly one lane must serve the kind.
    pub(crate) fn fork_module(&mut self, kind: CallKind, device: DeviceId) -> Result<&CUDAStream> {
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
    pub(crate) fn fork_microbatch(&mut self) -> Result<&CUDAStream> {
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
    pub(crate) fn owner(&self, handle: usize) -> Option<&CUDAStream> {
        self.iter()
            .find(|stream| stream.handle().is_ok_and(|raw| raw == handle))
    }

    /// Every owned stream, for synchronization before teardown.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &CUDAStream> {
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
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
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
