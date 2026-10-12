//! Shared host tensor storage and stream-ordered readiness.

use std::io;
use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use uniserve_core::SharedMemory;
use uniserve_cuda::{DeviceGuard, Event, EventFlags, HostRegistration, Stream};

use super::{DRAIN_LIMIT, drain_event};
use crate::ffi::{Backing, Device};
use crate::stream::Issue;
use crate::{Error, Result};

pub const SHM_HEADER_BYTES: usize = 512;
const ACK_OFFSET: usize = 64;
const ACK_SLOTS: usize = 64;
const PENDING: u32 = 0;
const READY: u32 = 1;
const FAILED: u32 = 2;
const CLAIMED: u32 = 1;
const ACKNOWLEDGED: u32 = 2;

/// A mapping can outlive its producer while a borrowed tensor view or a CUDA
/// host function still uses it. It owns no CUDA resources: its final release
/// is safe on a CUDA callback thread.
pub struct SharedMapping {
    address: usize,
    size: usize,
}

impl SharedMapping {
    pub fn open(name: &str, size: usize) -> io::Result<Self> {
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

    pub fn address(&self) -> usize {
        self.address
    }

    pub fn size(&self) -> usize {
        self.size
    }

    fn word(&self, offset: usize) -> &AtomicU32 {
        // SAFETY: only aligned header offsets within this mapping are used.
        // Both processes access these words atomically; payload writes start
        // at SHM_HEADER_BYTES and cannot overlap them.
        unsafe { &*((self.address + offset) as *const AtomicU32) }
    }
}

/// Shared mappings are host memory that copies read or write in place.
impl Backing for SharedMapping {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn address(&self) -> u64 {
        self.address as u64
    }

    fn len(&self) -> usize {
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
pub struct SharedBuffer {
    // Declared before the mapping: the registration ends while it is mapped.
    registration: Option<HostRegistration>,
    // Views of the payload retain the mapping itself; keep this reference
    // even after close has unlinked the allocation.
    mapping: Arc<SharedMapping>,
    storage: Option<SharedMemory>,
    consumers: Vec<usize>,
    /// The producer stream of a device source; `None` for a host source.
    producer: Option<Stream>,
    bytes: usize,
    submitted: bool,
    completion: Option<Arc<Event>>,
}

impl SharedBuffer {
    pub fn new(bytes: usize, consumers: Vec<usize>, producer: Option<Stream>) -> Result<Self> {
        if consumers.iter().any(|slot| *slot >= ACK_SLOTS) {
            return Err(Error::Invalid(
                "shared buffer acknowledgment slot is out of range".into(),
            ));
        }

        let size = SHM_HEADER_BYTES
            .checked_add(bytes.max(1))
            .ok_or_else(|| Error::Invalid("shared buffer is too large".into()))?;
        let (storage, file) =
            SharedMemory::create(size).map_err(|error| Error::Transport(error.to_string()))?;
        let mapping = Arc::new(
            SharedMapping::new(&file, size).map_err(|error| Error::Transport(error.to_string()))?,
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
            bytes,
            submitted: false,
            completion: None,
        })
    }

    pub fn mapping(&self) -> Result<Arc<SharedMapping>> {
        self.name()?;
        Ok(Arc::clone(&self.mapping))
    }

    pub fn name(&self) -> Result<&str> {
        self.storage
            .as_ref()
            .map(SharedMemory::name)
            .ok_or(Error::State("shared buffer is closed"))
    }

    pub fn nbytes(&self) -> usize {
        self.bytes
    }

    pub fn is_cuda(&self) -> bool {
        self.producer.is_some()
    }

    /// Announce bytes after their last write: behind the copies `issue`
    /// issued on the producer's stream, or in the order of its host writes
    /// for a CPU copy. The callback touches only host memory; it neither
    /// enters Python nor calls CUDA.
    /// The caller issues exactly one copy sequence per buffer, to `issue`,
    /// and must not modify its payload after announcing readiness.
    pub fn mark_ready(&mut self, issue: impl Issue) -> Result<()> {
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
    pub fn begin_copy(&mut self) {
        self.submitted = true;
    }

    pub fn settled(&self) -> bool {
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
    pub fn synchronize(&self) -> Result<()> {
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
    pub fn close(&mut self) -> Result<()> {
        if self.storage.is_none() {
            return Ok(());
        }

        self.synchronize()?;
        self.release()
    }

    /// Unregister and release the buffer without waiting for its producer:
    /// for a buffer whose copies are known complete or were never issued.
    pub fn release(&mut self) -> Result<()> {
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
    #[expect(
        clippy::mem_forget,
        reason = "unknown device completion must retain its registered mapping"
    )]
    fn drop(&mut self) {
        if self.close().is_err() {
            std::mem::forget(self.registration.take());
            std::mem::forget(Arc::clone(&self.mapping));
            std::mem::forget(self.storage.take());
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
    pub fn open(name: &str, offset: usize, bytes: usize, slot: usize) -> Result<Self> {
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
            _ => Error::Transport(error.to_string()),
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

    /// Poll shared host memory without interpreter calls. `check` observes
    /// the consuming transfer's native cancellation state between polls.
    /// Failure or timeout leaves the claim held until release or destruction.
    pub fn wait(&self, timeout: Duration, mut check: impl FnMut() -> Result<()>) -> Result<()> {
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

    pub fn mapping(&self) -> Result<Arc<SharedMapping>> {
        if self.slot.is_none() {
            return Err(Error::State("shared read is released"));
        }
        Ok(Arc::clone(&self.mapping))
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn nbytes(&self) -> usize {
        self.bytes
    }

    /// Limit a padded media row to its actual frame bytes before encoding.
    pub fn truncate(&mut self, bytes: usize) -> Result<()> {
        if bytes > self.bytes {
            return Err(Error::Invalid(
                "shared read exceeds its borrowed range".into(),
            ));
        }
        self.bytes = bytes;
        Ok(())
    }

    pub fn release(&mut self) {
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
