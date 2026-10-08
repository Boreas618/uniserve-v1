//! Execution streams and reusable ingress/output fences.

mod events;

pub use events::{Deferred, EventPool};

use std::sync::Arc;

use uniserve_cuda::{DeviceGuard, DeviceId, Event, EventFlags, Stream};

use crate::{Error, Result};

/// An execution stream, its SM allocation and bounded submission fences.
///
/// Callers serialize context use across streams sharing an SM partition.
/// Output fences may be borrowed until their ring slot is reused; each fence
/// retains the context (primary or SM partition) it was recorded in, so it
/// may outlive this stream. Forks independently retain an owned SM partition.
pub struct CUDAStream {
    device: DeviceId,
    stream: Option<Stream>,
    sm_count: u32,
    partitioned: bool,
    ingress: Vec<Arc<Event>>,
    outputs: Vec<Arc<Event>>,
    cursor: usize,
}

impl CUDAStream {
    fn new(device: DeviceId, stream: Stream, event_slots: usize) -> Result<Self> {
        if event_slots == 0 {
            return Err(Error::Invalid(
                "CUDA stream requires a positive event bound".into(),
            ));
        }

        let sm_count = stream.sm_count()?;
        let partitioned = stream.is_partitioned()?;
        let fences = || {
            (0..event_slots)
                .map(|_| Arc::new(Event::new(EventFlags::default())))
                .collect()
        };

        Ok(Self {
            device,
            stream: Some(stream),
            sm_count,
            partitioned,
            ingress: fences(),
            outputs: fences(),
            cursor: 0,
        })
    }

    /// Borrow an existing numerical stream; its creator retains the handle.
    ///
    /// # Safety
    /// `handle` is a live stream on `device`. Its creator keeps the stream and
    /// its context alive while this value, its forks and its fences are used.
    pub unsafe fn borrowed(device: DeviceId, handle: usize, event_slots: usize) -> Result<Self> {
        // The legacy default stream (handle 0) names no context of its own;
        // the driver resolves it against the calling thread's current context.
        let _device = DeviceGuard::new(device)?;
        // SAFETY: forwarded from the caller.
        let stream = unsafe { Stream::borrowed(device, handle) }?;
        Self::new(device, stream, event_slots)
    }

    /// Own an independent stream in the origin's borrowed CUDA context.
    ///
    /// # Safety
    /// `origin` is a live stream on `device` whose creator keeps its context
    /// (or SM partition) alive while the returned stream is used.
    pub unsafe fn sibling(device: DeviceId, origin: usize, event_slots: usize) -> Result<Self> {
        let _device = DeviceGuard::new(device)?;
        // SAFETY: forwarded from the caller.
        let origin = unsafe { Stream::borrowed(device, origin) }?;
        Self::new(device, origin.fork()?, event_slots)
    }

    /// Allocate the configured SM partitions from one disjoint split tree.
    pub fn partition(device: DeviceId, counts: &[u32], slots: &[usize]) -> Result<Vec<Self>> {
        if counts.len() != slots.len() || slots.contains(&0) {
            return Err(Error::Invalid(
                "each CUDA stream requires a positive event-slot count".into(),
            ));
        }

        Stream::partition(device, counts)?
            .into_iter()
            .zip(slots)
            .map(|(stream, slots)| Self::new(device, stream, *slots))
            .collect()
    }

    pub fn fork(&self) -> Result<Self> {
        Self::new(self.device, self.stream()?.fork()?, self.ingress.len())
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

    /// Enqueue a device dependency without waiting for the producer on the host.
    pub fn wait(&self, producer: &Stream) -> Result<()> {
        let stream = self.stream()?;
        if stream.raw() == producer.raw() {
            return Ok(());
        }

        // A default-stream producer records in the thread's current context.
        let _device = DeviceGuard::new(self.device)?;
        let event = &self.ingress[self.cursor];
        event.record(producer)?;
        stream.wait_event(event)?;
        Ok(())
    }

    /// Fence one submission for later joins, preserving independent lane overlap.
    /// Consumers enqueue their waits before this ring slot is reused.
    pub fn record(&mut self, consumer: usize) -> Result<Option<Arc<Event>>> {
        let stream = self.stream()?;
        if stream.raw() == consumer {
            return Ok(None);
        }

        let event = &self.outputs[self.cursor];
        event.record(stream)?;
        let event = Arc::clone(event);
        self.cursor = (self.cursor + 1) % self.outputs.len();
        Ok(Some(event))
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream()?.synchronize()?;
        Ok(())
    }

    /// Drain submitted work before releasing the stream and its SM partition.
    /// Failed collective execution retains resources until process exit.
    #[allow(clippy::mem_forget)]
    pub fn close(&mut self, aborted: bool) -> Result<()> {
        if aborted {
            std::mem::forget(self.stream.take());
            std::mem::forget(std::mem::take(&mut self.ingress));
            std::mem::forget(std::mem::take(&mut self.outputs));
            return Ok(());
        }

        if let Some(stream) = &self.stream {
            stream.synchronize()?;
        }

        self.ingress.clear();
        self.outputs.clear();
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
