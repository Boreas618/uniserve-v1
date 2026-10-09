//! Execution streams and the events that order work across them.

mod events;

pub use events::{Deferred, EventPool};

use uniserve_cuda::{DeviceGuard, DeviceId, Event, EventFlags, Stream};

use crate::{Error, Result};

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

    /// Borrow an existing numerical stream; its creator retains the handle.
    ///
    /// # Safety
    /// `handle` is a live stream on `device`. Its creator keeps the stream and
    /// its context alive while this value, its forks and its fences are used.
    pub unsafe fn borrowed(device: DeviceId, handle: usize) -> Result<Self> {
        // The legacy default stream (handle 0) names no context of its own;
        // the driver resolves it against the calling thread's current context.
        let _device = DeviceGuard::new(device)?;
        // SAFETY: forwarded from the caller.
        let stream = unsafe { Stream::borrowed(device, handle) }?;
        Self::new(device, stream)
    }

    /// Own an independent stream in the origin's borrowed CUDA context.
    ///
    /// # Safety
    /// `origin` is a live stream on `device` whose creator keeps its context
    /// (or SM partition) alive while the returned stream is used.
    pub unsafe fn sibling(device: DeviceId, origin: usize) -> Result<Self> {
        let _device = DeviceGuard::new(device)?;
        // SAFETY: forwarded from the caller.
        let origin = unsafe { Stream::borrowed(device, origin) }?;
        Self::new(device, origin.fork()?)
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

    /// Drain submitted work before releasing the stream and its SM partition.
    /// Failed collective execution retains resources until process exit.
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
