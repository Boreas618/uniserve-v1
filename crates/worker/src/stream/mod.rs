//! Execution streams and the events that order work across them.

mod events;
mod issue;

pub(crate) use events::Deferral;
pub use events::{Deferred, EventPool};
pub(crate) use issue::HostFunction;
pub use issue::{Fill, Issue, RowFill};

use uniserve_cuda::{DeviceGuard, DeviceId, Event, EventFlags, Stream};

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
    sm_count: u32,
    partitioned: bool,
    /// The fence [`Self::wait`] orders this stream after a producer with.
    /// One event serves every wait: a wait captures the record made just
    /// before it, and exclusive access keeps two waits from interleaving.
    ingress: Event,
}

impl CUDAStream {
    fn new(device: DeviceId, stream: Stream) -> Result<Self> {
        let sm_count = stream.sm_count()?;
        let partitioned = stream.is_partitioned()?;

        Ok(Self {
            device,
            stream: Some(stream),
            sm_count,
            partitioned,
            ingress: Event::new(EventFlags::default()),
        })
    }

    /// Own a new non-blocking stream on `device`'s full device, in its
    /// primary context.
    pub fn create(device: DeviceId) -> Result<Self> {
        Self::new(device, Stream::new(device)?)
    }

    /// Allocate the configured SM partitions from one disjoint split tree.
    pub fn partition(device: DeviceId, counts: &[u32]) -> Result<Vec<Self>> {
        Stream::partition(device, counts)?
            .into_iter()
            .map(|stream| Self::new(device, stream))
            .collect()
    }

    pub fn fork(&self) -> Result<Self> {
        Self::new(self.device, self.stream()?.fork()?)
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    pub fn sm_count(&self) -> u32 {
        self.sm_count
    }

    pub fn full_device(&self) -> bool {
        !self.partitioned
    }

    pub fn closed(&self) -> bool {
        self.stream.is_none()
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
    pub fn wait(&mut self, producer: &Stream) -> Result<()> {
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
    pub fn close(&mut self, aborted: bool) -> Result<()> {
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
