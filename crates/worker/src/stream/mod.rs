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

use cuda::{DeviceId, Stream};

use crate::config::Device;
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

/// Every stream of a rank: its primary device's full-device stream, the copy
/// stream of a rank that copies KV or latent pages through host storage,
/// each device's preparation stream for startup work, the full-device streams
/// of its other devices, the module streams the executor's components run
/// beside the primary stream on, and the expert microbatch streams forked from
/// it. The builder thread issues the work of every one of them (`crate::job`).
/// They close in dependency order: preparation, modules, microbatches, the copy
/// stream, the primary stream, devices, each newest first.
pub(crate) struct Streams {
    /// The full-device stream of a rank with a CUDA device, on its primary
    /// device: every batch's work is issued on it or on a stream forked from
    /// it.
    primary: Option<CUDAStream>,
    /// KV and latent page copies share this stream beside numerical work.
    /// Both pools belong to the rank's primary device.
    copy: Option<CUDAStream>,
    /// The full-device streams of the rank's other CUDA devices.
    devices: Vec<CUDAStream>,
    preparation: Vec<CUDAStream>,
    modules: Vec<CUDAStream>,
    microbatches: Vec<CUDAStream>,
}

/// The names [`Streams::observed`] gives the primary and the copy stream, in
/// the device time the rank reports (`ForwardStats::stream_us`).
pub(crate) const PRIMARY: &str = "device";
pub(crate) const COPY: &str = "copy";

impl Streams {
    /// Create the streams of `devices`.
    pub(crate) fn create(devices: &[Device]) -> Result<Self> {
        let cuda: Vec<DeviceId> = devices.iter().filter_map(|device| device.cuda()).collect();
        let mut streams = Self {
            primary: None,
            copy: None,
            devices: Vec::new(),
            preparation: Vec::new(),
            modules: Vec::new(),
            microbatches: Vec::new(),
        };
        // The primary device leads `devices`.
        for (index, &device) in cuda.iter().enumerate() {
            let stream = CUDAStream::create(device)?;
            if index == 0 {
                streams.primary = Some(stream);
            } else {
                streams.devices.push(stream);
            }
        }
        for &device in &cuda {
            streams.preparation.push(CUDAStream::create(device)?);
        }
        Ok(streams)
    }

    /// Borrow the rank's copy stream, creating it for the first host pool.
    /// The rank owns its lifetime; each pool shares the stream handle.
    pub(crate) fn copy(&mut self, device: DeviceId) -> Result<&CUDAStream> {
        if self.copy.is_none() {
            self.copy = Some(CUDAStream::create(device)?);
        }
        let copy = self
            .copy
            .as_ref()
            .ok_or(Error::State("the rank has no copy stream"))?;
        if copy.device() != device {
            return Err(Error::State("host pools must share the rank's copy device"));
        }
        Ok(copy)
    }

    /// The primary and the copy stream, by the names the rank reports their
    /// device time under ([`PRIMARY`], [`COPY`]).
    pub(crate) fn observed(&self) -> impl Iterator<Item = (&'static str, &CUDAStream)> {
        self.primary
            .iter()
            .map(|stream| (PRIMARY, stream))
            .chain(self.copy.iter().map(|stream| (COPY, stream)))
    }

    /// The full-device stream of `device`: the primary stream for the primary
    /// device.
    pub(crate) fn stream_for(&self, device: DeviceId) -> Option<&CUDAStream> {
        self.primary
            .iter()
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

    /// Fork the primary stream for one more microbatch.
    pub(crate) fn fork_microbatch(&mut self) -> Result<&CUDAStream> {
        let origin = self
            .primary
            .as_ref()
            .ok_or(Error::State("a microbatch stream requires a CUDA device"))?;
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
            .chain(&self.copy)
            .chain(&self.primary)
            .chain(&self.devices)
    }

    /// Close the streams in dependency order: preparation, modules,
    /// microbatches, the copy stream, the primary stream, then devices, each
    /// newest first. Attempts every stream before returning the first failure.
    /// No job is in flight unless the close is aborted: the worker drained the
    /// job queue ([`JobQueue::drain`]).
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
        if let Some(stream) = &mut self.copy {
            close(stream);
        }
        if let Some(stream) = &mut self.primary {
            close(stream);
        }
        self.devices.iter_mut().rev().for_each(&mut close);
        failure.map_or(Ok(()), Err)
    }
}
