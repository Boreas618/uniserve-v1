//! The CUDA VMM mechanism: exports in place or pooled, and mapped-allocation
//! reads.
//!
//! A device product is exported where it lies when its storage is an
//! exportable allocation, and otherwise copied into a chunk of the device's
//! export pool, whose header carries one acknowledgment word per consumer
//! slot. Readers import the allocation's shareable handle (a fabric handle,
//! or a descriptor fetched from the producer's grant socket), map it on a
//! device that can read it, and copy the exported spans.
//!
//! An export id names the allocation the export lies in: its first 16 hex
//! digits are the exporter's number of the allocation, the last 16 the
//! export's number within the endpoint ([`export_id_of`]). Readers key the
//! allocations they mapped by endpoint and allocation number
//! ([`Imports`]), so reads of several exports of one allocation share one
//! mapping.

mod grants;
mod handle;
mod peer;
pub(crate) mod pool;

pub(crate) use grants::DescriptorGrants;
pub(crate) use peer::PeerStorage;

use std::any::Any;
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Range;
use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};

use uniserve_cuda::{
    DeviceGuard, DeviceId, Event, IpcEventHandle, PeerMapping, ShareableHandle, SourceOrder,
};
use uniserve_ipc::{Locator, TransferTransport, WorkerEndpoint};

use super::backing::{CudaSource, SharedPool, TransportBuffer};
use super::export_table::ExportTable;
use super::read::{ImportOrder, NativeRead, ReadCall};
use super::transport::{Staged, read_destination};
use super::{ACK_SLOTS, ByteReservation, Spans, copy_planner};
use crate::ffi::{Backing, DType, MemoryKind};
use crate::storage::memory::{Memory, StorageView};
use crate::stream::Issue;
use crate::{Error, EventPool, Result, lock};
use handle::AllocationHandle;
use pool::VmmPool;

/// Bytes of one reader acknowledgment word in a pooled chunk's header.
const ACK_WORD_BYTES: u64 = 4;

/// A device's export pool and the shareable handle readers import.
struct DevicePool {
    pool: SharedPool,
    handle: AllocationHandle,
    /// The endpoint's number of the pool's allocation.
    allocation: u64,
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
    /// performed ([`crate::transfer::Transport::locate`]).
    Staged(Box<Staged>),
    /// The payload does not fit the device's export pool; another mechanism
    /// that reaches the consumers may carry it.
    PoolExhausted,
}

pub(crate) struct CudaExports {
    pub(crate) endpoint: Arc<ExportTable>,
    pub(crate) slot: usize,
    cross_host: bool,
    pool_bytes: usize,
    pools: Mutex<HashMap<DeviceId, Arc<DevicePool>>>,
    grants: Mutex<Option<Arc<DescriptorGrants>>>,
    failed: Mutex<Option<FailedExport>>,
    /// Numbers of the allocations this endpoint exported.
    allocations: Mutex<Numbering>,
    /// The number of the endpoint's next export.
    exports: AtomicU64,
    /// Other processes' allocations this rank's reads mapped.
    imports: Arc<Imports>,
}

/// An endpoint's numbers of the allocations its exports lie in, by the
/// address of their `Memory`. A number is given once and never reused
/// within the endpoint. An entry holds its allocation weakly, which keeps
/// that address from being reused by another `Memory` while the entry
/// exists.
#[derive(Default)]
struct Numbering {
    next: u64,
    numbers: HashMap<usize, (Weak<Memory>, u64)>,
}

impl Numbering {
    fn number(&mut self, memory: &Arc<Memory>) -> u64 {
        let address = Arc::as_ptr(memory) as usize;
        if let Some((_, number)) = self.numbers.get(&address) {
            return *number;
        }
        self.numbers
            .retain(|_, (allocation, _)| allocation.strong_count() > 0);
        let number = self.next;
        self.next += 1;
        self.numbers
            .insert(address, (Arc::downgrade(memory), number));
        number
    }
}

/// The id of an endpoint's export `export` of allocation `allocation`: 32
/// hex digits, which also key its descriptor grant.
fn export_id_of(allocation: u64, export: u64) -> String {
    format!("{allocation:016x}{export:016x}")
}

/// The exporter's number of the allocation export `id` lies in.
fn allocation_number(id: &str) -> Result<u64> {
    id.get(..16)
        .and_then(|digits| u64::from_str_radix(digits, 16).ok())
        .ok_or_else(|| Error::Invalid("a CUDA VMM export id names its allocation".into()))
}

/// Storage of other processes this rank reads over CUDA VMM: the
/// allocations it mapped, by exporting endpoint, the exporter's number of
/// the allocation and the device the mapping is on, and the producers'
/// readiness events it opened, by endpoint, export and device.
///
/// Entries are weak: an import lives while a read that reached it lives (a
/// read retains its imports, [`super::Read::retain_import`], and issuing a
/// read takes the imports it finds). Reads alive at once (the layers of one
/// cache import, the regions of one fetch) import an allocation once, and
/// the rank holds no import that no read uses. Keys never name two
/// allocations: an endpoint name belongs to one transport of one process,
/// which numbers its allocations once.
#[derive(Default)]
pub(crate) struct Imports {
    allocations: Mutex<HashMap<(String, u64, DeviceId), Weak<Memory>>>,
    events: Mutex<HashMap<(String, String, DeviceId), Weak<Event>>>,
}

impl Imports {
    fn find<K: Eq + Hash, V>(entries: &Mutex<HashMap<K, Weak<V>>>, key: &K) -> Option<Arc<V>> {
        lock(entries).get(key).and_then(Weak::upgrade)
    }

    /// Keep `value` under `key` unless a live entry is there already, which
    /// is returned instead: another read imported the same storage at once.
    fn keep<K: Eq + Hash, V>(
        entries: &Mutex<HashMap<K, Weak<V>>>,
        key: K,
        value: Arc<V>,
    ) -> Arc<V> {
        let mut entries = lock(entries);
        if let Some(existing) = entries.get(&key).and_then(Weak::upgrade) {
            return existing;
        }
        entries.retain(|_, entry| entry.strong_count() > 0);
        entries.insert(key, Arc::downgrade(&value));
        value
    }
}

impl CudaExports {
    pub(crate) fn new(
        endpoint: Arc<ExportTable>,
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
            allocations: Mutex::new(Numbering::default()),
            exports: AtomicU64::new(0),
            imports: Arc::new(Imports::default()),
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
        let MemoryKind::Cuda(_) = source.device() else {
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
        bytes: ByteReservation,
        events: &EventPool,
        locate: impl FnOnce(TransferTransport) -> Result<Locator>,
    ) -> Result<Exported> {
        if issue.stream().is_none() {
            return Err(Error::Invalid(
                "a device export requires its producer stream".into(),
            ));
        }
        let MemoryKind::Cuda(device) = source.device() else {
            return Err(Error::Invalid(
                "cuda_vmm transport requires a CUDA tensor".into(),
            ));
        };
        let nbytes = source.nbytes();
        let export = self.exports.fetch_add(1, Ordering::Relaxed);
        let mut state = ExportState {
            bytes: Some(bytes),
            ..ExportState::default()
        };

        let result = (|| -> Result<Option<Staged>> {
            let first = source.first().address();
            let (spans, handle, size, offset, number) = match Memory::exportable_containing(first) {
                Some((memory, offset)) => {
                    let number = lock(&self.allocations).number(&memory);
                    let allocation = memory.allocation().ok_or_else(|| {
                        Error::Invariant("an exportable allocation is device memory".into())
                    })?;
                    let shareable = allocation.export()?.ok_or_else(|| {
                        Error::Invariant("a registered exportable allocation has a handle".into())
                    })?;
                    let handle = AllocationHandle::of(&shareable);
                    if let ShareableHandle::Descriptor(fd) = shareable {
                        state.descriptor = Some(fd);
                    }
                    let size = memory.bytes() as u64;
                    (source.clone(), handle, size, offset, number)
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
                    let storage_bytes = memory.bytes() as u64;
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
                        pool.handle,
                        storage_bytes,
                        chunk.payload_offset() as u64,
                        pool.allocation,
                    )
                }
            };
            let export_id = export_id_of(number, export);
            state.export = Some(export_id.clone());

            // CUDA IPC events are host-local. Cross-host exports establish
            // readiness before sending their allocation handle to readers.
            if self.cross_host {
                let _range = crate::profiling::range(c"vmm_export_synchronize", None);
                issue.synchronize()?;
            }
            let ready = events.fence(issue, !self.cross_host)?;
            state.event = Some(Arc::clone(&ready));
            if let AllocationHandle::Descriptor(fd) = handle {
                let grants = self.grants()?;
                grants.register(&export_id, fd).map_err(|error| {
                    Error::transport_source("granting an export's descriptor", error)
                })?;
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
                allocation_handle: handle.encode(),
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
                state.bytes.take().ok_or_else(|| {
                    Error::Invariant("an export's bytes are held until its backing".into())
                })?,
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
                self.abandon(state, issue, events)?;
                Err(error)
            }
        }
    }

    /// Undo an export that failed before its backing was constructed: drain
    /// the producer stream, then return the chunk, grant and bytes. A failed
    /// drain retains everything and fails every later export. A constructed
    /// backing is given up by its transport and retires through the event
    /// pool ([`crate::transfer::Transport::discard`]).
    fn abandon(&self, mut state: ExportState, issue: impl Issue, events: &EventPool) -> Result<()> {
        let cleanup = (|| -> Result<()> {
            // A stream issued the failed attempt's copies at once and
            // drains them here; a job gives up the copies and this wait
            // with the rest of the failed attempt.
            if state.submitted || state.event.is_some() {
                issue.synchronize()?;
            }
            if let Some(event) = state.event.take() {
                let copied = state.copied.take();
                issue.defer(
                    events,
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
        if let (Some(grants), Some(export)) = (&state.grants, &state.export) {
            grants.release(export);
        }
        if let Some((pool, chunk)) = &state.allocation {
            lock(pool).release(chunk)?;
        }
        state.bytes.take().map_or(Ok(()), ByteReservation::release)
    }

    fn pool(&self, device: DeviceId) -> Result<Arc<DevicePool>> {
        if let Some(pool) = lock(&self.pools).get(&device) {
            return Ok(Arc::clone(pool));
        }
        // Readers in this process may copy from another device.
        let memory = Memory::exportable(device, self.pool_bytes, &Memory::peers(device)?)?;
        let allocation = memory
            .allocation()
            .ok_or_else(|| Error::Invariant("an export pool is device memory".into()))?;
        let shareable = allocation.export()?.ok_or_else(|| {
            Error::Invariant("an export pool allocation has a shareable handle".into())
        })?;
        let handle = AllocationHandle::of(&shareable);
        let descriptor = match shareable {
            ShareableHandle::Descriptor(fd) => Some(fd),
            ShareableHandle::Fabric(_) => None,
        };
        let address = memory.address() as usize;
        let capacity = memory.bytes();
        // The pool lives as long as the endpoint publishes exports, so its
        // number names one allocation for the endpoint's lifetime.
        let allocation = lock(&self.allocations).number(&memory);
        // SAFETY: the pool retains `memory`, which maps `capacity` writable
        // bytes at `address`; producer streams outlive the transport.
        let pool = unsafe { VmmPool::new(memory, device, address, capacity) }?;
        let pool = Arc::new(DevicePool {
            pool: Arc::new(Mutex::new(pool)),
            handle,
            allocation,
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
                .map_err(|error| Error::transport_source("opening the descriptor grants", error))?,
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
            result =
                result.and(grants.close().map_err(|error| {
                    Error::transport_source("closing the descriptor grants", error)
                }));
        }
        result
    }
}

impl CudaExports {
    /// Complete a fenced export: its location with the interprocess handle
    /// of its readiness event, once that event was recorded, registered in
    /// the export table. A failure retires the export's backing.
    pub(crate) fn locate(
        &self,
        events: &Arc<EventPool>,
        mut locator: Locator,
        ready: Option<Arc<Event>>,
        buffer: Arc<TransportBuffer>,
    ) -> Result<Locator> {
        let registered = (|| -> Result<()> {
            if let Some(ready) = &ready {
                match &mut locator.transport {
                    TransferTransport::CudaVmm {
                        ready_event_handle, ..
                    } => *ready_event_handle = ready.ipc_handle()?.0.to_vec(),
                    _ => {
                        return Err(Error::Invariant(
                            "a fenced export has a CUDA VMM location".into(),
                        ));
                    }
                }
            }
            self.endpoint.register(locator.clone(), Arc::clone(&buffer))
        })();
        if let Err(error) = registered {
            TransportBuffer::reclaim(buffer, events, None)?;
            return Err(error);
        }
        Ok(locator)
    }

    /// The read of a CUDA VMM export into `destination` on `device`, for
    /// the transport of `source` (see [`crate::transfer::Transport::issue_fetch`]). A read
    /// of another process's export takes the imports of it this rank holds.
    pub(crate) fn read(
        &self,
        source: &WorkerEndpoint,
        locator: &Locator,
        device: MemoryKind,
        destination: Option<&Spans>,
        region: Option<&[Range<u64>]>,
    ) -> Result<ReadCall> {
        let TransferTransport::CudaVmm {
            endpoint,
            export_id,
            allocation_handle,
            ..
        } = &locator.transport
        else {
            return Err(Error::Invalid(
                "CUDA VMM read requires a CUDA VMM locator".into(),
            ));
        };
        let same_node = locator.source.node == source.node;
        if !same_node && !AllocationHandle::decode(allocation_handle)?.crosses_hosts() {
            return Err(Error::Invalid(
                "a device product crosses hosts only as a fabric handle".into(),
            ));
        }
        let MemoryKind::Cuda(target) = device else {
            return Err(Error::Invalid(
                "CUDA VMM destination must be a CUDA device".into(),
            ));
        };
        let local = locator.source.address_space == source.address_space;

        let import = if local {
            None
        } else {
            // A device that cannot address the producer's device reads the
            // allocation through a mapping on the producer's device.
            let mut mapped = target;
            if same_node
                && let MemoryKind::Cuda(producer) = MemoryKind::from_locator(&locator.device)
                && producer != target
                && !target.can_access_peer(producer)?
            {
                mapped = producer;
            }
            let allocation = (endpoint.clone(), allocation_number(export_id)?, mapped);
            let event = (endpoint.clone(), export_id.clone(), mapped);
            Some(Import {
                device: mapped,
                mapping: Imports::find(&self.imports.allocations, &allocation),
                event: Imports::find(&self.imports.events, &event),
                imports: Arc::clone(&self.imports),
            })
        };
        Ok(ReadCall::Cuda(CudaRead {
            locator: Arc::new(locator.clone()),
            destination: read_destination(locator, device, destination, region)?,
            region: region.map(<[_]>::to_vec),
            slot: self.slot,
            import,
        }))
    }
}

impl Drop for CudaExports {
    fn drop(&mut self) {
        if let Some(failed) = self
            .failed
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            // Device access to a failed export never resolved: its backing
            // stays until the process exits.
            crate::leak(failed);
        }
    }
}

/// Resources one export attempt acquired, for failure cleanup.
#[derive(Default)]
struct ExportState {
    /// The export's bytes, which its backing takes once constructed.
    bytes: Option<ByteReservation>,
    event: Option<Arc<Event>>,
    copied: Option<Spans>,
    allocation: Option<(SharedPool, Arc<super::vmm::pool::PoolChunk>)>,
    descriptor: Option<OwnedFd>,
    grants: Option<Arc<DescriptorGrants>>,
    /// The export's id, once its allocation is known.
    export: Option<String>,
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

/// How a read of another process's export reaches it: the device its
/// allocation is mapped on, and the imports of the allocation and of the
/// export's readiness event the rank held when the read was issued.
pub(crate) struct Import {
    device: DeviceId,
    mapping: Option<Arc<Memory>>,
    event: Option<Arc<Event>>,
    imports: Arc<Imports>,
}

/// One read of a CUDA VMM export.
pub(crate) struct CudaRead {
    pub(crate) locator: Arc<Locator>,
    pub(crate) destination: Spans,
    pub(crate) region: Option<Vec<Range<u64>>>,
    pub(crate) slot: usize,
    /// `None` when the export lies in this address space and is read
    /// through its registry instead of a mapping.
    import: Option<Import>,
}

impl CudaRead {
    pub(crate) fn destination(&self) -> &Spans {
        &self.destination
    }

    pub(crate) fn run(&self, read: &NativeRead) -> Result<()> {
        read.read.require_active()?;
        let MemoryKind::Cuda(device) = self.destination.device() else {
            return Err(Error::Invalid(
                "CUDA VMM destination must be a CUDA device".into(),
            ));
        };
        let _guard = DeviceGuard::new(device)?;

        let (mut spans, order) = match &self.import {
            None => {
                let (buffer, grant) = ExportTable::acquire_read(&self.locator)?;
                read.read.bind_source(grant)?;
                let spans = buffer.spans()?;
                // The export's spans live on the producer's device, whose
                // stream waits for its (interprocess) ready event.
                let source = match spans.device() {
                    MemoryKind::Cuda(source) => source,
                    _ => device,
                };
                let order = ImportOrder {
                    device: source,
                    producer: buffer.event(),
                    acknowledgment: None,
                };
                (spans, order)
            }
            Some(import) => self.import(read, import)?,
        };
        if let Some(region) = &self.region {
            spans = spans.region(region)?;
        }
        read.copy(&spans, &self.destination, None, Some(order), Vec::new())
    }

    /// View the export's spans in a mapping of its allocation on the import
    /// device: the import the rank holds, or a new one. Returns the spans
    /// and the read's order on the import device: the producer's ready
    /// event and this reader's acknowledgment word. The read retains both
    /// imports while it lives.
    fn import(&self, read: &NativeRead, import: &Import) -> Result<(Spans, ImportOrder)> {
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
        let device = import.device;
        let imports = &import.imports;

        let key = (endpoint.clone(), allocation_number(export_id)?, device);
        let mapping = match import
            .mapping
            .clone()
            .or_else(|| Imports::find(&imports.allocations, &key))
        {
            Some(mapping) => mapping,
            None => {
                let handle =
                    AllocationHandle::decode(allocation_handle)?.import(endpoint, export_id)?;
                let mapping = Memory::mapped(PeerMapping::import(
                    device,
                    &handle,
                    *storage_size_bytes as usize,
                )?);
                // The mapping retains the imported allocation; the
                // descriptor is no longer needed.
                drop(handle);
                Imports::keep(&imports.allocations, key, mapping)
            }
        };
        if mapping.bytes() as u64 != *storage_size_bytes {
            return Err(Error::Invariant(
                "an imported allocation disagrees with its export's size".into(),
            ));
        }
        read.read
            .retain_import(Arc::clone(&mapping) as Arc<dyn Any + Send + Sync>);

        let dtype = DType::parse(&self.locator.dtype)?;
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

        if self.slot >= ACK_SLOTS {
            return Err(Error::Invalid(
                "VMM acknowledgment slot is out of range".into(),
            ));
        }
        let acknowledgment = (*acknowledgment_offset >= 0).then(|| {
            mapping.address() + *acknowledgment_offset as u64 + self.slot as u64 * ACK_WORD_BYTES
        });
        let producer = if ready_event_handle.is_empty() {
            None
        } else {
            let key = (endpoint.clone(), export_id.clone(), device);
            let event = match import
                .event
                .clone()
                .or_else(|| Imports::find(&imports.events, &key))
            {
                Some(event) => event,
                None => {
                    let bytes =
                        <[u8; 64]>::try_from(ready_event_handle.as_slice()).map_err(|_| {
                            Error::Invalid("CUDA IPC event handle must contain 64 bytes".into())
                        })?;
                    let event = Arc::new(Event::open_ipc(device, &IpcEventHandle(bytes))?);
                    Imports::keep(&imports.events, key, event)
                }
            };
            read.read
                .retain_import(Arc::clone(&event) as Arc<dyn Any + Send + Sync>);
            Some(event)
        };
        let order = ImportOrder {
            device,
            producer,
            acknowledgment,
        };
        Ok((spans, order))
    }
}
