//! Execution streams, the rank's set of them, and the events that order work
//! across them.

mod events;
mod issue;
mod scratch;

pub(crate) use events::Deferral;
pub(crate) use events::Deferred;
pub use events::EventPool;
pub(crate) use issue::HostFunction;
pub(crate) use issue::{Fill, Issue, RowFill};
pub(crate) use scratch::Scratches;

use crossbeam_channel::{Receiver, Sender};
use cuda::{DeviceId, Stream};

use crate::config::Device;
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

/// An execution stream of a rank.
pub(crate) struct CUDAStream {
    device: DeviceId,
    stream: Option<Stream>,
}

impl CUDAStream {
    fn new(device: DeviceId, stream: Stream) -> Self {
        Self {
            device,
            stream: Some(stream),
        }
    }

    /// Own a new non-blocking stream on `device`'s full device, in its
    /// primary context.
    pub(crate) fn create(device: DeviceId) -> Result<Self> {
        Ok(Self::new(device, Stream::new(device)?))
    }

    pub(crate) fn fork(&self) -> Result<Self> {
        Ok(Self::new(self.device, self.stream()?.fork()?))
    }

    pub(crate) fn device(&self) -> DeviceId {
        self.device
    }

    pub(crate) fn stream(&self) -> Result<&Stream> {
        self.stream
            .as_ref()
            .ok_or(Error::State("CUDA stream is closed"))
    }

    pub(crate) fn synchronize(&self) -> Result<()> {
        self.stream()?.synchronize()?;
        Ok(())
    }

    /// Drain submitted work, then release this owner's handle to the stream.
    /// The driver stream is released once every component sharing the
    /// handle has dropped its clone. Failed collective
    /// execution retains resources until process exit.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        if aborted {
            crate::leak(self.stream.take());
            return Ok(());
        }

        if let Some(stream) = &self.stream {
            stream.synchronize()?;
        }
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

/// Every stream of a rank: its lane's, its copy lane's, each device's
/// preparation stream for startup work outside the lane, the full-device
/// streams of its other devices, the module streams executor lanes that
/// run beside the lane's fork, and the expert microbatch streams forked
/// from the lane's. They close in dependency
/// order: preparation, modules, microbatches, the copy lane, the lane,
/// devices, each newest first.
pub(crate) struct Streams {
    /// The lane of a rank with a CUDA device, on its primary device.
    lane: Option<Lane>,
    /// The copy lane of a rank whose prefix cache copies pages through the
    /// host KV cache (`crate::lane`), on its primary device.
    copy: Option<Lane>,
    /// Where the lanes post their completions; the rank's job queues take
    /// them from here, and give the posting end to the builder thread's
    /// work.
    posted: Sender<(JobId, Completion)>,
    completions: Receiver<(JobId, Completion)>,
    /// The full-device streams of the rank's other CUDA devices.
    devices: Vec<CUDAStream>,
    preparation: Vec<CUDAStream>,
    modules: Vec<CUDAStream>,
    microbatches: Vec<CUDAStream>,
}

impl Streams {
    /// Create the streams of `devices`.
    pub(crate) fn create(devices: &[Device]) -> Result<Self> {
        let cuda: Vec<DeviceId> = devices.iter().filter_map(|device| device.cuda()).collect();
        let (posted, completions) = crossbeam_channel::unbounded();
        let mut streams = Self {
            lane: None,
            copy: None,
            posted: posted.clone(),
            completions,
            devices: Vec::new(),
            preparation: Vec::new(),
            modules: Vec::new(),
            microbatches: Vec::new(),
        };
        // The primary device leads `devices`.
        for (index, &device) in cuda.iter().enumerate() {
            let stream = CUDAStream::create(device)?;
            if index == 0 {
                streams.lane = Some(Lane::start(
                    crate::lane::LANE.into(),
                    stream,
                    posted.clone(),
                )?);
            } else {
                streams.devices.push(stream);
            }
        }
        for &device in &cuda {
            streams.preparation.push(CUDAStream::create(device)?);
        }
        Ok(streams)
    }

    /// Start the copy lane on a full-device stream of `device`, the device
    /// of the prefix cache that maps a host page pool, before the job
    /// queues are taken ([`Self::job_queues`]). Returns its stream.
    pub(crate) fn start_copy_lane(&mut self, device: DeviceId) -> Result<&CUDAStream> {
        if self.copy.is_some() {
            return Err(Error::State("the rank's copy lane is already started"));
        }
        let lane = Lane::start(
            crate::lane::COPY_LANE.into(),
            CUDAStream::create(device)?,
            self.posted.clone(),
        )?;
        Ok(&self.copy.insert(lane).stream)
    }

    /// The job queues of the lane and the copy lane, for the executor,
    /// which submits each forward to the lane.
    pub(crate) fn job_queues(&self) -> JobQueues {
        JobQueues::of(
            self.lane.as_ref(),
            self.copy.as_ref(),
            self.posted.clone(),
            self.completions.clone(),
        )
    }

    /// The full-device stream of `device`: the lane's for the primary
    /// device.
    pub(crate) fn stream_for(&self, device: DeviceId) -> Option<&CUDAStream> {
        self.lane
            .iter()
            .map(|lane| &lane.stream)
            .chain(&self.devices)
            .find(|stream| stream.device() == device)
    }

    /// The startup preparation stream of `device`.
    pub(crate) fn preparation(&self, device: DeviceId) -> Option<&CUDAStream> {
        self.preparation
            .iter()
            .find(|stream| stream.device() == device)
    }

    /// Fork `device`'s full-device stream for a module that runs beside it.
    pub(crate) fn fork_module(&mut self, device: DeviceId) -> Result<&CUDAStream> {
        let origin = self.stream_for(device).ok_or_else(|| {
            Error::Invalid(format!("the rank has no stream on cuda:{}", device.0))
        })?;
        let stream = origin.fork()?;
        self.modules.push(stream);
        self.modules
            .last()
            .ok_or(Error::State("module stream was not registered"))
    }

    /// Fork the lane's stream for one more microbatch.
    pub(crate) fn fork_microbatch(&mut self) -> Result<&CUDAStream> {
        let origin = self
            .lane
            .as_ref()
            .map(|lane| &lane.stream)
            .ok_or(Error::State("a microbatch stream requires a CUDA lane"))?;
        let stream = origin.fork()?;
        self.microbatches.push(stream);
        self.microbatches
            .last()
            .ok_or(Error::State("microbatch stream was not registered"))
    }

    /// Every owned stream, for synchronization before teardown.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &CUDAStream> {
        self.preparation
            .iter()
            .chain(&self.modules)
            .chain(&self.microbatches)
            .chain(self.copy.iter().map(|lane| &lane.stream))
            .chain(self.lane.iter().map(|lane| &lane.stream))
            .chain(&self.devices)
    }

    /// Close the streams in dependency order: preparation, modules,
    /// microbatches, the copy lane, the lane, then devices, each newest
    /// first. Attempts every stream before returning the first failure. No
    /// job is in flight unless the close is aborted: the worker drained the
    /// lanes ([`JobQueues::drain`]).
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
        if let Some(lane) = &mut self.copy {
            close(&mut lane.stream);
        }
        if let Some(lane) = &mut self.lane {
            close(&mut lane.stream);
        }
        self.devices.iter_mut().rev().for_each(&mut close);
        failure.map_or(Ok(()), Err)
    }
}
