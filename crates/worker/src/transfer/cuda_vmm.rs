//! CUDA VMM exports (in place or pooled) and mapped-allocation reads.
//!
//! A device product is exported where it lies when its storage is an
//! exportable allocation, and otherwise copied into a chunk of the device's
//! export pool, whose header carries one acknowledgment word per consumer
//! slot. Readers import the allocation's shareable handle (a fabric handle,
//! or a descriptor fetched from the producer's grant socket), map it on a
//! device that can read it, and copy the exported spans.

use std::collections::HashMap;
use std::ops::Range;
use std::os::fd::{AsRawFd, OwnedFd};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use uniserve_cuda::{
    DeviceGuard, DeviceId, Event, IpcEventHandle, PeerMapping, ShareableHandle, SourceOrder,
    Stream, memset_d32,
};
use uniserve_ipc::{Locator, TransferTransport};

use super::endpoint::Endpoint;
use super::read::{ACKNOWLEDGED, CLAIMED, NativeRead};
use super::source::{CudaSource, SharedPool, TransportBuffer};
use super::transport::Staged;
use super::vmm_pool::{ACK_SLOTS, VmmPool};
use super::{Capacity, DescriptorGrants, Spans, copy_planner, fetch_descriptor};
use crate::ffi::{Backing, Device};
use crate::storage::memory::{Memory, StorageView, parse_dtype};
use crate::stream::Issue;
use crate::{Error, EventPool, Result};

/// Bytes of one reader acknowledgment word in a pooled chunk's header.
const ACK_WORD_BYTES: u64 = 4;

/// A device's export pool and the shareable handle readers import.
struct DevicePool {
    pool: SharedPool,
    handle: Vec<u8>,
    /// The pool allocation's process descriptor, open for the pool's
    /// lifetime; grants duplicate it per export.
    _descriptor: Option<OwnedFd>,
}

/// An export whose producer stream could not be drained. Everything it
/// touched stays retained, its bytes stay reserved, and later exports fail.
struct FailedExport {
    error: Arc<Error>,
    _retained: Vec<Box<dyn std::any::Any + Send>>,
}

/// The outcome of one export attempt.
pub(crate) enum Exported {
    Located(Box<Locator>),
    /// The export's location is complete once the work it issued was
    /// performed ([`Transport::locate`]).
    Staged(Box<Staged>),
    /// The payload does not fit the device's export pool; another mechanism
    /// that reaches the consumers may carry it.
    PoolExhausted,
}

pub(crate) struct CudaExports {
    pub(crate) endpoint: Arc<Endpoint>,
    pub(crate) slot: usize,
    cross_host: bool,
    pool_bytes: usize,
    pools: Mutex<HashMap<i32, Arc<DevicePool>>>,
    grants: Mutex<Option<Arc<DescriptorGrants>>>,
    failed: Mutex<Option<FailedExport>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl CudaExports {
    pub(crate) fn new(
        endpoint: Arc<Endpoint>,
        slot: usize,
        cross_host: bool,
        pool_bytes: u64,
    ) -> Self {
        Self {
            endpoint,
            slot,
            cross_host,
            pool_bytes: pool_bytes as usize,
            pools: Mutex::new(HashMap::new()),
            grants: Mutex::new(None),
            failed: Mutex::new(None),
        }
    }

    pub(crate) fn check(&self) -> Result<()> {
        match &*lock(&self.failed) {
            Some(failed) => Err(Error::Shared(Arc::clone(&failed.error))),
            None => Ok(()),
        }
    }

    /// Device spans of one allocation sharing one stride.
    pub(crate) fn validate(source: &Spans) -> Result<()> {
        let Device::Cuda(_) = source.device() else {
            return Err(Error::Invalid(
                "cuda_vmm transport requires a CUDA tensor".into(),
            ));
        };
        let first = source.first();
        let backing = Arc::as_ptr(first.memory()).cast::<()>();
        if source.views().iter().any(|view| {
            Arc::as_ptr(view.memory()).cast::<()>() != backing || view.strides() != first.strides()
        }) {
            return Err(Error::Invalid(
                "CUDA VMM export spans require one allocation and stride".into(),
            ));
        }
        Ok(())
    }

    /// Export `source`, produced on `stream`, with `consumers`' slots.
    pub(crate) fn export(
        &self,
        source: &Spans,
        issue: impl Issue,
        consumers: &[usize],
        capacity: &Arc<Capacity>,
        events: &EventPool,
        locate: impl FnOnce(TransferTransport) -> Result<Locator>,
    ) -> Result<Exported> {
        let stream = issue
            .stream()
            .ok_or_else(|| Error::Invalid("a device export requires its producer stream".into()))?;
        let Device::Cuda(device) = source.device() else {
            return Err(Error::Invalid(
                "cuda_vmm transport requires a CUDA tensor".into(),
            ));
        };
        let nbytes = source.nbytes();
        let export_id = uuid::Uuid::new_v4().simple().to_string();
        let mut state = ExportState::default();

        let result = (|| -> Result<Option<Staged>> {
            let first = source.first().address();
            let (spans, handle, size, offset) = match Memory::exportable_containing(first) {
                Some((memory, offset)) => {
                    let allocation = memory.allocation().ok_or_else(|| {
                        Error::Invariant("an exportable allocation is device memory".into())
                    })?;
                    let handle = match allocation.export()? {
                        Some(ShareableHandle::Fabric(bytes)) => bytes.to_vec(),
                        Some(ShareableHandle::Descriptor(fd)) => {
                            let bytes = fd.as_raw_fd().to_ne_bytes().to_vec();
                            state.descriptor = Some(fd);
                            bytes
                        }
                        None => {
                            return Err(Error::Invariant(
                                "a registered exportable allocation has no handle".into(),
                            ));
                        }
                    };
                    (source.clone(), handle, memory.len() as u64, offset)
                }
                None => {
                    let pool = self.pool(device)?;
                    let chunk = lock(&pool.pool).reserve(nbytes as usize, issue)?;
                    let Some(chunk) = chunk else {
                        return Ok(None);
                    };
                    state.allocation = Some((Arc::clone(&pool.pool), Arc::clone(&chunk)));
                    let memory = Arc::clone(lock(&pool.pool).backing()?);
                    // Readers map the whole allocation, whose length the
                    // driver rounded up to its granularity.
                    let storage_bytes = memory.len() as u64;
                    let backing = memory as Arc<dyn Backing>;
                    let shape: Vec<i64> =
                        source.shape().iter().map(|&extent| extent as i64).collect();
                    let payload = Spans::single(StorageView::contiguous(
                        backing,
                        chunk.payload_offset(),
                        source.dtype(),
                        &shape,
                    )?)?;

                    let mut plan = copy_planner(Some(device), SourceOrder::Stream)?;
                    payload.plan_copy(source, &mut plan)?;
                    state.submitted = true;
                    // SAFETY: the chunk stays reserved and the producer's spans
                    // stay retained (`copied`) until the export retires.
                    unsafe { issue.copy_strided(&plan) }?;
                    state.copied = Some(source.clone());
                    (
                        payload,
                        pool.handle.clone(),
                        storage_bytes,
                        chunk.payload_offset() as u64,
                    )
                }
            };

            // CUDA IPC events are host-local. Cross-host exports establish
            // readiness before sending their allocation handle to readers.
            if self.cross_host {
                let _range = crate::profiling::range(c"vmm_export_synchronize", None);
                issue.synchronize()?;
            }
            let ready = events.fence(issue, !self.cross_host)?;
            state.event = Some(Arc::clone(&ready));
            if let Ok(bytes) = <[u8; 4]>::try_from(handle.as_slice()) {
                let grants = self.grants()?;
                grants
                    .register(&export_id, i32::from_ne_bytes(bytes))
                    .map_err(|error| Error::Transport(error.to_string()))?;
                state.grants = Some(grants);
            }

            let layout = layout(&spans, offset)?;
            // A same-host reader waits on the readiness event through its
            // interprocess handle, which exists once the record was made:
            // the location receives it where the export is located.
            let locator = locate(TransferTransport::CudaVmm {
                endpoint: self.endpoint.name().to_owned(),
                export_id: export_id.clone(),
                storage_size_bytes: size,
                storage_offsets_bytes: layout.offsets,
                span_lengths: layout.lengths,
                span_counts: layout.counts,
                tensor_stride: spans.first().strides().to_vec(),
                ready_event_handle: Vec::new(),
                allocation_handle: handle,
                acknowledgment_offset: state
                    .allocation
                    .as_ref()
                    .map_or(-1, |(_, chunk)| chunk.offset as i64),
            })?;

            let pooled = state.allocation.is_some();
            let buffer = Arc::new(TransportBuffer::cuda(
                CudaSource {
                    source: spans,
                    event: Arc::clone(&ready),
                    copied: state.copied.take(),
                    pool: state.allocation.clone(),
                    consumers: if pooled {
                        consumers.to_vec()
                    } else {
                        Vec::new()
                    },
                    grants: state.grants.clone(),
                    export: export_id.clone(),
                    descriptor: state.descriptor.take(),
                },
                nbytes,
                Arc::clone(capacity),
            ));
            Ok(Some(Staged::Fenced {
                locator,
                // Cross-host readers do not wait on the event: the export
                // waited for its bytes before its handle leaves.
                ready: (!self.cross_host).then_some(ready),
                buffer,
            }))
        })();

        match result {
            Ok(Some(staged)) => Ok(Exported::Staged(Box::new(staged))),
            Ok(None) => Ok(Exported::PoolExhausted),
            Err(error) => {
                self.abandon(state, stream, events, &export_id, capacity, nbytes)?;
                Err(error)
            }
        }
    }

    /// Undo an export that failed before its backing was constructed: drain
    /// the producer stream, then return the chunk, grant and bytes. A failed
    /// drain retains everything and fails every later export. A constructed
    /// backing is given up by its transport and retires through the event
    /// pool ([`Transport::discard`]).
    fn abandon(
        &self,
        mut state: ExportState,
        stream: &Stream,
        events: &EventPool,
        export_id: &str,
        capacity: &Capacity,
        nbytes: u64,
    ) -> Result<()> {
        let cleanup = (|| -> Result<()> {
            // FIXME: an export recorded into a job issued nothing yet, and
            // its failed attempt keeps only its event operations, so this
            // drain waits for nothing of it; it is still a driver call of
            // the thread that records.
            if state.submitted || state.event.is_some() {
                stream.synchronize()?;
            }
            if let Some(event) = state.event.take() {
                let copied = state.copied.take();
                events.defer(
                    vec![event],
                    Box::new(move || {
                        drop(copied);
                        Ok(())
                    }),
                )?;
            }
            Ok(())
        })();
        if let Err(cleanup) = cleanup {
            let cleanup = Arc::new(cleanup);
            *lock(&self.failed) = Some(FailedExport {
                error: Arc::clone(&cleanup),
                _retained: vec![Box::new(state)],
            });
            return Err(Error::Shared(cleanup));
        }
        if let Some(grants) = &state.grants {
            grants.release(export_id);
        }
        if let Some((pool, chunk)) = &state.allocation {
            lock(pool).release(chunk)?;
        }
        capacity.release(nbytes)
    }

    fn pool(&self, device: i32) -> Result<Arc<DevicePool>> {
        if let Some(pool) = lock(&self.pools).get(&device) {
            return Ok(Arc::clone(pool));
        }
        // Readers in this process may copy from another device.
        let memory = Memory::exportable(device, self.pool_bytes, &Memory::peers(device)?)?;
        let allocation = memory
            .allocation()
            .ok_or_else(|| Error::Invariant("an export pool is device memory".into()))?;
        let (handle, descriptor) = match allocation.export()? {
            Some(ShareableHandle::Fabric(bytes)) => (bytes.to_vec(), None),
            Some(ShareableHandle::Descriptor(fd)) => {
                (fd.as_raw_fd().to_ne_bytes().to_vec(), Some(fd))
            }
            None => {
                return Err(Error::Invariant(
                    "an export pool allocation has no shareable handle".into(),
                ));
            }
        };
        let address = memory.address() as usize;
        let capacity = memory.len();
        // SAFETY: the pool retains `memory`, which maps `capacity` writable
        // bytes at `address`; producer streams outlive the transport.
        let pool = unsafe { VmmPool::new(memory, device, address, capacity) }?;
        let pool = Arc::new(DevicePool {
            pool: Arc::new(Mutex::new(pool)),
            handle,
            _descriptor: descriptor,
        });
        Ok(Arc::clone(lock(&self.pools).entry(device).or_insert(pool)))
    }

    fn grants(&self) -> Result<Arc<DescriptorGrants>> {
        let mut grants = lock(&self.grants);
        if let Some(grants) = &*grants {
            return Ok(Arc::clone(grants));
        }
        let created = Arc::new(
            DescriptorGrants::new(self.endpoint.name())
                .map_err(|error| Error::Transport(error.to_string()))?,
        );
        *grants = Some(Arc::clone(&created));
        Ok(created)
    }

    fn pools(&self) -> Vec<Arc<DevicePool>> {
        lock(&self.pools).values().cloned().collect()
    }

    pub(crate) fn reap(&self) -> Result<()> {
        for pool in self.pools() {
            lock(&pool.pool).reap()?;
        }
        Ok(())
    }

    pub(crate) fn awaiting_acknowledgment(&self) -> bool {
        self.pools()
            .iter()
            .any(|pool| lock(&pool.pool).awaiting_acknowledgment())
    }

    pub(crate) fn close(&self) -> Result<()> {
        self.check()?;
        let mut result = Ok(());
        for pool in self.pools() {
            result = result.and(lock(&pool.pool).close().map(drop));
        }
        if result.is_ok() {
            // Every chunk retired: the pools' allocations and descriptors
            // can be released.
            let pools = std::mem::take(&mut *lock(&self.pools));
            drop(pools);
        }
        let grants = lock(&self.grants).take();
        if let Some(grants) = grants {
            result = result.and(
                grants
                    .close()
                    .map_err(|error| Error::Transport(error.to_string())),
            );
        }
        result
    }
}

impl Drop for CudaExports {
    #[expect(
        clippy::mem_forget,
        reason = "unresolved device access must retain its backing until process teardown"
    )]
    fn drop(&mut self) {
        if let Some(failed) = self
            .failed
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            std::mem::forget(failed);
        }
    }
}

/// Resources one export attempt acquired, for failure cleanup.
#[derive(Default)]
struct ExportState {
    event: Option<Arc<Event>>,
    copied: Option<Spans>,
    allocation: Option<(SharedPool, Arc<super::vmm_pool::PoolChunk>)>,
    descriptor: Option<OwnedFd>,
    grants: Option<Arc<DescriptorGrants>>,
    submitted: bool,
}

/// Span placement of an export inside its allocation.
struct Layout {
    offsets: Vec<u64>,
    lengths: Vec<u64>,
    counts: Vec<u32>,
}

/// Describe `spans` relative to an allocation in which the first span
/// starts at byte `offset`: per-span byte offsets and the run-length
/// encoded leading extents.
fn layout(spans: &Spans, offset: u64) -> Result<Layout> {
    let first = spans.first().address();
    let offsets = spans
        .views()
        .iter()
        .map(|view| {
            (offset + view.address())
                .checked_sub(first)
                .ok_or_else(|| Error::Invalid("CUDA VMM spans precede their first span".into()))
        })
        .collect::<Result<_>>()?;
    let mut lengths: Vec<u64> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    for view in spans.views() {
        let length = view.shape()[0] as u64;
        match (lengths.last(), counts.last_mut()) {
            (Some(&last), Some(count)) if last == length => *count += 1,
            _ => {
                lengths.push(length);
                counts.push(1);
            }
        }
    }
    Ok(Layout {
        offsets,
        lengths,
        counts,
    })
}

/// A mapped export: its spans, the producer's ready event, this reader's
/// acknowledgment word and the device the allocation is mapped on.
struct Imported {
    spans: Spans,
    event: Option<Arc<Event>>,
    acknowledgment: Option<u64>,
    device: i32,
}

/// One read of a CUDA VMM export.
pub(crate) struct CudaRead {
    pub(crate) locator: Arc<Locator>,
    pub(crate) destination: Spans,
    pub(crate) region: Option<Vec<Range<u64>>>,
    pub(crate) slot: usize,
    /// The export lies in this address space and is read through its
    /// registry instead of a mapping.
    pub(crate) local: bool,
    pub(crate) same_node: bool,
}

/// The CUDA ordinal of a locator's `cuda:<index>` device, if any.
fn locator_device(locator: &Locator) -> Option<i32> {
    locator.device.strip_prefix("cuda:")?.parse().ok()
}

impl CudaRead {
    pub(crate) fn destination(&self) -> &Spans {
        &self.destination
    }

    pub(crate) fn run(&self, read: &NativeRead) -> Result<()> {
        read.read.require_active()?;
        let Device::Cuda(device) = self.destination.device() else {
            return Err(Error::Invalid(
                "CUDA VMM destination must be a CUDA device".into(),
            ));
        };
        let _guard = DeviceGuard::new(DeviceId(device))?;

        let (mut mapped, mut event, acknowledgment, import_device) = if self.local {
            let (buffer, grant) = Endpoint::acquire_read(&self.locator)?;
            read.read.bind_source(grant)?;
            let spans = buffer.spans()?;
            // The export's spans live on the producer's device; its ready
            // event is interprocess, which a stream of another device cannot
            // wait on (the driver faults), so treat that device as the
            // import device.
            let source = match spans.device() {
                Device::Cuda(source) => source,
                _ => device,
            };
            (spans, buffer.event(), None, source)
        } else {
            let imported = self.import(device)?;
            (
                imported.spans,
                imported.event,
                imported.acknowledgment,
                imported.device,
            )
        };
        if let Some(region) = &self.region {
            mapped = mapped.region(region)?;
        }

        let cross_device = import_device != device;
        if cross_device && let Some(producer) = event.take() {
            // An interprocess fence cannot order another device's stream:
            // finish the producer on the host instead.
            producer.synchronize()?;
        }
        if cross_device && let Some(word) = acknowledgment {
            self.acknowledge(read, import_device, word, CLAIMED)?;
            read.copy(&mapped, &self.destination, None, None, Vec::new())?;
            self.acknowledge(read, import_device, word, ACKNOWLEDGED)
        } else {
            read.copy(
                &mapped,
                &self.destination,
                event,
                acknowledgment,
                Vec::new(),
            )
        }
    }

    /// Map the exported allocation on a device that can read it and view
    /// its spans. Returns the spans, the producer's ready event, the
    /// address of this reader's acknowledgment word and the import device.
    fn import(&self, device: i32) -> Result<Imported> {
        let TransferTransport::CudaVmm {
            endpoint,
            export_id,
            storage_size_bytes,
            storage_offsets_bytes,
            span_lengths,
            span_counts,
            tensor_stride,
            ready_event_handle,
            allocation_handle,
            acknowledgment_offset,
        } = &self.locator.transport
        else {
            return Err(Error::Invalid(
                "CUDA VMM read requires a CUDA VMM locator".into(),
            ));
        };

        let mut import_device = device;
        if self.same_node
            && let Some(source) = locator_device(&self.locator)
            && source != device
            && !DeviceId(device).can_access_peer(DeviceId(source))?
        {
            import_device = source;
        }

        let handle = match <[u8; 4]>::try_from(allocation_handle.as_slice()) {
            Ok(_) => ShareableHandle::Descriptor(
                fetch_descriptor(endpoint, export_id)
                    .map_err(|error| Error::Transport(error.to_string()))?,
            ),
            Err(_) => ShareableHandle::Fabric(
                <[u8; 64]>::try_from(allocation_handle.as_slice()).map_err(|_| {
                    Error::Invalid("a CUDA VMM handle is a descriptor or a fabric handle".into())
                })?,
            ),
        };
        let mapping = Memory::mapped(PeerMapping::import(
            DeviceId(import_device),
            &handle,
            *storage_size_bytes as usize,
        )?);
        // The mapping retains the imported allocation; the descriptor is no
        // longer needed.
        drop(handle);

        let dtype = parse_dtype(&self.locator.dtype)?;
        if storage_offsets_bytes
            .iter()
            .any(|offset| offset % dtype.bytes() as u64 != 0)
        {
            return Err(Error::Invalid(
                "CUDA VMM span offset is not element aligned".into(),
            ));
        }
        let lengths = span_lengths
            .iter()
            .zip(span_counts)
            .flat_map(|(&length, &count)| std::iter::repeat_n(length, count as usize));
        let views = storage_offsets_bytes
            .iter()
            .zip(lengths)
            .map(|(&offset, length)| {
                let mut shape = vec![length as i64];
                shape.extend(self.locator.shape[1..].iter().map(|&extent| extent as i64));
                StorageView::strided(
                    Arc::clone(&mapping) as Arc<dyn Backing>,
                    offset as usize,
                    dtype,
                    &shape,
                    tensor_stride,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let spans = Spans::new(views)?;

        let acknowledgment = (*acknowledgment_offset >= 0).then(|| {
            mapping.address() + *acknowledgment_offset as u64 + self.slot as u64 * ACK_WORD_BYTES
        });
        if self.slot >= ACK_SLOTS {
            return Err(Error::Invalid(
                "VMM acknowledgment slot is out of range".into(),
            ));
        }
        let event = if ready_event_handle.is_empty() {
            None
        } else {
            let bytes = <[u8; 64]>::try_from(ready_event_handle.as_slice()).map_err(|_| {
                Error::Invalid("CUDA IPC event handle must contain 64 bytes".into())
            })?;
            Some(Arc::new(Event::open_ipc(
                DeviceId(import_device),
                &IpcEventHandle(bytes),
            )?))
        };
        Ok(Imported {
            spans,
            event,
            acknowledgment,
            device: import_device,
        })
    }

    /// Write this reader's acknowledgment word on the import device and wait
    /// for it, for reads whose copy runs on another device.
    fn acknowledge(&self, read: &NativeRead, device: i32, word: u64, state: u32) -> Result<()> {
        let _device = DeviceGuard::new(DeviceId(device))?;
        let stream = Stream::new(DeviceId(device))?;
        // SAFETY: the word lies in the mapping the read's source spans retain.
        unsafe { memset_d32(word, state, 1, &stream) }?;
        if let Err(error) = stream.synchronize() {
            // The write may still reference the imported mapping.
            read.read.mark_undrained(
                Arc::new(stream),
                None,
                vec![Box::new(self.destination.clone())],
            );
            return Err(error.into());
        }
        Ok(())
    }
}
