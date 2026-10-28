//! The shared-memory mechanism: a POSIX segment of this host holding a copy
//! of the export behind a readiness word and one acknowledgment word per
//! consumer slot ([`SharedBuffer`]), its export, its reads and in-place
//! borrows ([`SharedRead`]).

use std::io;
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use cuda::{DeviceGuard, Event, EventFlags, HostRegistration, SourceOrder, Stream};
use ipc::SharedMemory;
use ipc::{Locator, TransferTransport};

use super::backing::TransportBuffer;
use super::export_table::ExportTable;
use super::read::{NativeRead, ReadCall, payload};
use super::transport::{Backend, ExportView, Transport, producer, read_destination, row_span};
use super::{
    ACK_SLOTS, ACKNOWLEDGED, ByteReservation, CLAIMED, DRAIN_LIMIT, Spans, copy_planner,
    drain_event,
};
use crate::ffi::{Backing, MemoryKind};
use crate::storage::allocation::StorageView;
use crate::stream::Issue;
use crate::{Error, Result};

pub(crate) const SHM_HEADER_BYTES: usize = 512;
/// How long a reader waits for its producer's readiness.
const SHARED_READY_TIMEOUT: Duration = Duration::from_secs(120);
const ACK_OFFSET: usize = 64;
const PENDING: u32 = 0;
const READY: u32 = 1;
const FAILED: u32 = 2;

/// A mapping can outlive its producer while a borrowed tensor view or a CUDA
/// host function still uses it. It owns no CUDA resources: its final release
/// is safe on a CUDA callback thread.
pub(crate) struct SharedMapping {
    address: usize,
    size: usize,
}

impl SharedMapping {
    pub(crate) fn open(name: &str, size: usize) -> io::Result<Self> {
        let file = SharedMemory::open(name, true)?;
        if file.metadata()?.len() < size as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "shared buffer is shorter than its locator",
            ));
        }

        Self::new(&file, size)
    }

    fn new(file: &std::fs::File, size: usize) -> io::Result<Self> {
        // SAFETY: storage is physically backed for the full writable extent.
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        Ok(Self {
            address: address as usize,
            size,
        })
    }

    pub(crate) fn address(&self) -> usize {
        self.address
    }

    pub(crate) fn size(&self) -> usize {
        self.size
    }

    fn word(&self, offset: usize) -> &AtomicU32 {
        // SAFETY: only aligned header offsets within this mapping are used.
        // Both processes access these words atomically; payload writes start
        // at SHM_HEADER_BYTES and cannot overlap them.
        unsafe { &*((self.address + offset) as *const AtomicU32) }
    }
}

/// Shared mappings are host storage that copies read or write in place.
impl Backing for SharedMapping {
    fn device(&self) -> MemoryKind {
        MemoryKind::Host
    }

    fn address(&self) -> u64 {
        self.address as u64
    }

    fn bytes(&self) -> usize {
        self.size
    }
}

impl Drop for SharedMapping {
    fn drop(&mut self) {
        // SAFETY: this is the last owner of this live mapping.
        unsafe { libc::munmap(self.address as *mut _, self.size()) };
    }
}

/// Owns a producer's segment and optional CUDA host registration. The buffer
/// shares its producer's device stream through close; the caller retains the
/// buffer until all granted readers have acknowledged it.
pub(crate) struct SharedBuffer {
    // Declared before the mapping: the registration ends while it is mapped.
    registration: Option<HostRegistration>,
    // Views of the payload retain the mapping itself; keep this reference
    // even after close has unlinked the allocation.
    mapping: Arc<SharedMapping>,
    storage: Option<SharedMemory>,
    consumers: Vec<usize>,
    /// The producer stream of a device source; `None` for a host source.
    producer: Option<Stream>,
    submitted: bool,
    completion: Option<Arc<Event>>,
}

impl SharedBuffer {
    pub(crate) fn new(
        bytes: usize,
        consumers: Vec<usize>,
        producer: Option<Stream>,
    ) -> Result<Self> {
        if consumers.iter().any(|slot| *slot >= ACK_SLOTS) {
            return Err(Error::Invalid(
                "shared buffer acknowledgment slot is out of range".into(),
            ));
        }

        let size = SHM_HEADER_BYTES
            .checked_add(bytes.max(1))
            .ok_or_else(|| Error::Invalid("shared buffer is too large".into()))?;
        let (storage, file) = SharedMemory::create(size)
            .map_err(|error| Error::transport_source("creating a shared buffer", error))?;
        let mapping = Arc::new(
            SharedMapping::new(&file, size)
                .map_err(|error| Error::transport_source("mapping a shared buffer", error))?,
        );

        // Fresh POSIX storage is zero-filled: readiness is PENDING and all
        // reader slots are unclaimed before any locator can escape.
        let registration = match &producer {
            Some(stream) => {
                // SAFETY: the mapping owns this full range. Close unregisters
                // it before releasing the storage; a failed close retains both.
                Some(unsafe {
                    HostRegistration::new(
                        stream.device(),
                        mapping.address() as *mut u8,
                        mapping.size(),
                    )
                }?)
            }
            None => None,
        };

        Ok(Self {
            registration,
            mapping,
            storage: Some(storage),
            consumers,
            producer,
            submitted: false,
            completion: None,
        })
    }

    pub(crate) fn mapping(&self) -> Result<Arc<SharedMapping>> {
        self.name()?;
        Ok(Arc::clone(&self.mapping))
    }

    pub(crate) fn name(&self) -> Result<&str> {
        self.storage
            .as_ref()
            .map(SharedMemory::name)
            .ok_or(Error::State("shared buffer is closed"))
    }

    pub(crate) fn is_cuda(&self) -> bool {
        self.producer.is_some()
    }

    /// Announce bytes after their last write: behind the copies `issue`
    /// issued on the producer's stream, or in the order of its host writes
    /// for a CPU copy. The callback touches only host storage; it neither
    /// enters Python nor calls CUDA.
    /// The caller issues exactly one copy sequence per buffer, to `issue`,
    /// and must not modify its payload after announcing readiness.
    pub(crate) fn mark_ready(&mut self, issue: impl Issue) -> Result<()> {
        let mapping = self.mapping()?;
        let ready = Box::new(move || mapping.word(0).store(READY, Ordering::Release));

        if let Some(stream) = &self.producer {
            // A default-stream handle resolves against the current context.
            let _device = DeviceGuard::new(stream.device())?;
            issue.launch_host_fn(ready)?;

            let completion = Arc::new(Event::new(EventFlags::default()));
            issue.record(&completion)?;
            self.completion = Some(completion);
            Ok(())
        } else {
            issue.host().launch_host_fn(ready)
        }
    }

    /// Declare possible DMA before the producer submits its first copy. A
    /// failed submission may still have enqueued earlier spans.
    pub(crate) fn begin_copy(&mut self) {
        self.submitted = true;
    }

    pub(crate) fn settled(&self) -> bool {
        let mapping = &self.mapping;
        self.storage.is_none()
            || (mapping.word(0).load(Ordering::Acquire) != PENDING
                && self.consumers.iter().all(|slot| {
                    mapping
                        .word(ACK_OFFSET + slot * size_of::<u32>())
                        .load(Ordering::Acquire)
                        != CLAIMED
                }))
    }

    /// Shutdown and a close before readiness must see the producer's DMA
    /// finished even if readiness was never scheduled: wait on the host for
    /// it, at most [`DRAIN_LIMIT`]. Normal retirement observes READY and
    /// does not wait.
    pub(crate) fn synchronize(&self) -> Result<()> {
        if self.submitted
            && let Some(stream) = &self.producer
            && self.mapping.word(0).load(Ordering::Acquire) == PENDING
        {
            let deadline = Instant::now() + DRAIN_LIMIT;
            match &self.completion {
                Some(completion) => drain_event(completion, deadline)?,
                None => {
                    // Submission failed before its completion event was
                    // recorded: fence what the stream holds now.
                    let _device = DeviceGuard::new(stream.device())?;
                    let fence = Event::new(EventFlags::default());
                    fence.record(stream)?;
                    drain_event(&fence, deadline)?;
                }
            }
        }

        Ok(())
    }

    /// Unregister and release the buffer once its producer's DMA finished.
    pub(crate) fn close(&mut self) -> Result<()> {
        if self.storage.is_none() {
            return Ok(());
        }

        self.synchronize()?;
        self.release()
    }

    /// Unregister and release the buffer without waiting for its producer:
    /// for a buffer whose copies are known complete or were never issued.
    pub(crate) fn release(&mut self) -> Result<()> {
        if self.storage.is_none() {
            return Ok(());
        }

        if let Some(registration) = self.registration.take() {
            // Producer DMA has stopped and the mapping is still live.
            registration.unregister()?;
        }

        self.producer = None;
        self.storage.take();
        Ok(())
    }
}

impl Drop for SharedBuffer {
    fn drop(&mut self) {
        // After a failed close the producer's completion is unknown: the
        // registered mapping and its storage stay until the process exits.
        if self.close().is_err() {
            crate::leak(self.registration.take());
            crate::leak(Arc::clone(&self.mapping));
            crate::leak(self.storage.take());
        }
    }
}

/// A claimed payload range on the reader's host. Its mapping stays alive
/// through borrowed tensor views; release acknowledges the end of reading.
pub struct SharedRead {
    mapping: Arc<SharedMapping>,
    slot: Option<usize>,
    offset: usize,
    bytes: usize,
}

impl SharedRead {
    /// Map and claim one byte range. The caller must wait for readiness
    /// before accessing the payload and end all reads before releasing it.
    pub(crate) fn open(name: &str, offset: usize, bytes: usize, slot: usize) -> Result<Self> {
        if slot >= ACK_SLOTS {
            return Err(Error::Invalid(
                "shared buffer acknowledgment slot is out of range".into(),
            ));
        }
        let offset = SHM_HEADER_BYTES
            .checked_add(offset)
            .ok_or_else(|| Error::Invalid("shared read offset is too large".into()))?;
        let size = offset
            .checked_add(bytes)
            .ok_or_else(|| Error::Invalid("shared read is too large".into()))?;
        let mapping = SharedMapping::open(name, size).map_err(|error| match error.kind() {
            io::ErrorKind::NotFound => {
                Error::Invalid("shared buffer is no longer available".into())
            }
            io::ErrorKind::InvalidInput => Error::Invalid(error.to_string()),
            _ => Error::transport_source("opening a shared buffer", error),
        })?;
        mapping
            .word(ACK_OFFSET + slot * size_of::<u32>())
            .store(CLAIMED, Ordering::Release);

        Ok(Self {
            mapping: Arc::new(mapping),
            slot: Some(slot),
            offset,
            bytes,
        })
    }

    /// Poll shared host storage without interpreter calls. `check` observes
    /// the consuming transfer's native cancellation state between polls.
    /// Failure or timeout leaves the claim held until release or destruction.
    pub(crate) fn wait(
        &self,
        timeout: Duration,
        mut check: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        let started = Instant::now();
        let mut pause = Duration::ZERO;

        loop {
            check()?;
            match self.mapping.word(0).load(Ordering::Acquire) {
                READY => return Ok(()),
                FAILED => return Err(Error::Resource("export producer failed before readiness")),
                _ => {}
            }
            if started.elapsed() >= timeout {
                return Err(Error::Resource("export endpoint was lost before readiness"));
            }

            if !pause.is_zero() {
                std::thread::sleep(pause);
            }
            pause = (pause + Duration::from_micros(50)).min(Duration::from_millis(1));
        }
    }

    pub(crate) fn mapping(&self) -> Result<Arc<SharedMapping>> {
        if self.slot.is_none() {
            return Err(Error::State("shared read is released"));
        }
        Ok(Arc::clone(&self.mapping))
    }

    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    pub fn nbytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn release(&mut self) {
        if let Some(slot) = self.slot.take() {
            self.mapping
                .word(ACK_OFFSET + slot * size_of::<u32>())
                .store(ACKNOWLEDGED, Ordering::Release);
        }
    }
}

impl Drop for SharedRead {
    fn drop(&mut self) {
        self.release();
    }
}

impl Transport {
    /// The read of a shared-memory export of this host into `destination`
    /// (see [`Transport::issue_fetch`]).
    pub(super) fn shared_read(
        &self,
        slot: usize,
        locator: &Locator,
        device: MemoryKind,
        destination: Option<&Spans>,
        region: Option<&[Range<u64>]>,
    ) -> Result<ReadCall> {
        if locator.source.host != self.source.host {
            return Err(Error::Invalid(
                "shared storage transport requires the source host".into(),
            ));
        }
        Ok(ReadCall::Shared {
            locator: Arc::new(locator.clone()),
            slot,
            destination: read_destination(locator, device, destination, region)?,
            region: region.map(<[_]>::to_vec),
        })
    }

    pub(super) fn export_shared(
        &self,
        view: &ExportView<'_>,
        endpoint: &Arc<ExportTable>,
        consumers: &[usize],
        bytes: ByteReservation,
        stream: impl Issue,
    ) -> Result<Locator> {
        let producer = producer(view.spans, stream.stream())?;
        // The buffer keeps the producer's stream for its own lifetime: it
        // drains the stream before it unregisters or releases its bytes.
        let shared = producer.as_ref().map(|(_, stream)| (*stream).clone());
        let mut storage = SharedBuffer::new(view.nbytes as usize, consumers.to_vec(), shared)?;

        let written = (|| -> Result<Locator> {
            let locator = view.locator(
                &self.source,
                TransferTransport::PosixShm {
                    endpoint: endpoint.name().to_owned(),
                    name: storage.name()?.to_owned(),
                },
            );
            let mapping = storage.mapping()? as Arc<dyn Backing>;
            let shape: Vec<i64> = view.shape.iter().map(|&extent| extent as i64).collect();
            let payload = Spans::single(StorageView::contiguous(
                mapping,
                SHM_HEADER_BYTES,
                view.spans.dtype(),
                &shape,
            )?)?;
            match &producer {
                Some((device, _)) => {
                    // Mark in-flight storage before the first copy: a failed
                    // copy must drain before unregistration or byte reuse.
                    storage.begin_copy();
                    let mut plan = copy_planner(Some(*device), SourceOrder::Stream)?;
                    payload.plan_copy(view.spans, &mut plan)?;
                    // SAFETY: the registered mapping stays mapped until the
                    // buffer closes, which drains this stream first.
                    unsafe { stream.copy_strided(&plan) }?;
                }
                None => {
                    let mut plan = copy_planner(None, SourceOrder::Stream)?;
                    payload.plan_copy(view.spans, &mut plan)?;
                    // SAFETY: host operands, written in the producer's
                    // order; no reader maps the segment yet.
                    unsafe { stream.host().copy_strided(&plan) }?;
                }
            }
            storage.mark_ready(stream)?;
            Ok(locator)
        })();
        let locator = match written {
            Ok(locator) => locator,
            Err(error) => {
                // The attempt failed: a stream issued its copies at once
                // and drains them here; a job gives up the copies and this
                // wait with the rest of the failed attempt. Either way no
                // DMA into the mapping is outstanding when it is released.
                stream.synchronize()?;
                storage.release()?;
                return Err(error);
            }
        };

        let buffer = Arc::new(TransportBuffer::shared(storage, bytes));
        if let Err(error) = endpoint.register(&locator, Arc::clone(&buffer)) {
            TransportBuffer::reclaim(buffer, &self.events, None)?;
            return Err(error);
        }
        Ok(locator)
    }

    /// Borrow whole leading rows of a shared-memory export of this host in
    /// place. `region` is in logical coordinates; releasing the returned
    /// read ends the claim.
    pub fn borrow(&self, locator: &Locator, region: Option<&[Range<u64>]>) -> Result<SharedRead> {
        let Backend::Shared { slot, .. } = &self.backend else {
            return Err(Error::Invalid(
                "only shared storage supports host borrowing".into(),
            ));
        };
        if locator.source.host != self.source.host {
            return Err(Error::Invalid(
                "shared storage transport requires the source host".into(),
            ));
        }
        let TransferTransport::PosixShm { name, .. } = &locator.transport else {
            return Err(Error::Invalid(
                "shared storage borrow requires a shared storage locator".into(),
            ));
        };
        let (offset, bytes) = row_span(locator, region)?;
        let read = SharedRead::open(name, offset, bytes, *slot)?;
        read.wait(SHARED_READY_TIMEOUT, || Ok(()))?;
        Ok(read)
    }
}

impl NativeRead {
    pub(super) fn shared(
        &self,
        locator: &Arc<Locator>,
        slot: usize,
        destination: &Spans,
        region: Option<&[Range<u64>]>,
    ) -> Result<()> {
        let TransferTransport::PosixShm { name, .. } = &locator.transport else {
            return Err(Error::Invalid(
                "shared read requires a shared storage locator".into(),
            ));
        };
        let mut read = SharedRead::open(name, 0, locator.nbytes as usize, slot)?;
        read.wait(SHARED_READY_TIMEOUT, || self.read.check_active())?;

        let source = payload(
            read.mapping()? as Arc<dyn Backing>,
            SHM_HEADER_BYTES,
            locator,
        )?;
        let source = match region {
            Some(region) => source.region(region)?,
            None => source,
        };
        self.copy_from_host(&source, destination, || read.release())
    }
}
