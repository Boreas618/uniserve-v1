//! One rank's export and read mechanisms.
//!
//! A [`Transport`] exports device or host tensors through one mechanism and
//! reads locators of that mechanism into Rust-owned destinations:
//!
//! | Kind | Export | Read |
//! | --- | --- | --- |
//! | local | register the spans with a producer fence | borrow in place, or copy |
//! | shm | copy into a POSIX shared-memory segment | stage on the host, then copy |
//! | cuda_vmm | export in place or copy into a VMM pool chunk | map the handle, then copy |
//! | channel | copy into the locator's payload bytes | stage on the host, then copy |
//!
//! All transports of a rank share one [`Capacity`]: export bytes are held
//! until physical retirement, and reads hold a read credit plus their bytes
//! until their copy drains.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::Duration;

use uniserve_cuda::{Event, SourceOrder, Stream};
use uniserve_ipc::{Locator, TransferTransport, Wake, WorkerEndpoint};

use super::cuda_vmm::{CudaExports, CudaRead, Exported};
use super::endpoint::Endpoint;
use super::read::{self, BorrowedReads, NativeRead, Read, ReadCall, ReadContext, ReadPool};
use super::source::TransportBuffer;
use super::{
    Capacity, Notify, ReadReservation, Retirement, SHM_HEADER_BYTES, SharedBuffer, SharedRead,
    Spans, copy_planner, notify,
};
use crate::ffi::{Backing, Device};
use crate::storage::memory::{Memory, StorageView, device_name, dtype_name, parse_dtype};
use crate::stream::Issue;
use crate::{Error, EventPool, Result};

/// How long a shared-memory borrower waits for its producer's readiness.
const SHARED_READY_TIMEOUT: Duration = Duration::from_secs(120);

/// Registered exports of one shared-memory or CUDA VMM endpoint.
const REGISTERED_EXPORTS: usize = 256;

/// A transfer mechanism.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TransportKind {
    Local,
    Shm,
    CudaVmm,
    Channel,
}

impl TransportKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Shm => "shm",
            Self::CudaVmm => "cuda_vmm",
            Self::Channel => "channel",
        }
    }

    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "local" => Self::Local,
            "shm" => Self::Shm,
            "cuda_vmm" => Self::CudaVmm,
            "channel" => Self::Channel,
            _ => return Err(Error::Invalid(format!("unknown transport: {name}"))),
        })
    }

    /// The mechanism a locator was exported through.
    pub fn of(locator: &Locator) -> Self {
        match locator.transport {
            TransferTransport::Local { .. } => Self::Local,
            TransferTransport::PosixShm { .. } => Self::Shm,
            TransferTransport::CudaVmm { .. } => Self::CudaVmm,
            TransferTransport::Channel { .. } => Self::Channel,
        }
    }
}

/// Rank facts every transport of a rank shares.
#[derive(Clone, Debug)]
pub struct TransportOptions {
    /// This rank, published in every locator.
    pub source: WorkerEndpoint,
    /// This rank's reader slot in other ranks' acknowledgment headers.
    pub acknowledgment_slot: usize,
    /// Consumer slots of ranks on this host: shared memory serves them, the
    /// rank channel serves the others.
    pub host_slots: HashSet<usize>,
    /// Whether any consumer lives on another host (CUDA IPC events do not
    /// reach it; exports then synchronize before publishing).
    pub cross_host_consumers: bool,
    /// Bytes of each device's CUDA VMM export pool.
    pub vmm_pool_bytes: u64,
}

enum Backend {
    Local {
        endpoint: Arc<Endpoint>,
        next: AtomicU64,
    },
    Shared {
        endpoint: Arc<Endpoint>,
        slot: usize,
        host_slots: HashSet<usize>,
    },
    Cuda(CudaExports),
    Channel {
        name: String,
        host_slots: HashSet<usize>,
    },
}

/// One mechanism's exports and reads.
pub struct Transport {
    source: WorkerEndpoint,
    capacity: Arc<Capacity>,
    events: Arc<EventPool>,
    reads: Arc<ReadPool>,
    context: Arc<ReadContext>,
    borrowed: Arc<BorrowedReads>,
    backend: Backend,
}

/// The logical extent of an export: its shape, its offset in the logical
/// tensor, and its payload bytes.
struct ExportView<'a> {
    spans: &'a Spans,
    shape: Vec<u64>,
    offset: Vec<u64>,
    nbytes: u64,
}

impl<'a> ExportView<'a> {
    fn new(spans: &'a Spans, offset: Option<&[u64]>) -> Result<Self> {
        let shape = spans.shape();
        let offset = match offset {
            Some(offset) if offset.len() == shape.len() => offset.to_vec(),
            Some(_) => {
                return Err(Error::Invalid(
                    "export offset does not match its tensor shape".into(),
                ));
            }
            None => vec![0; shape.len()],
        };
        Ok(Self {
            spans,
            shape,
            offset,
            nbytes: spans.nbytes(),
        })
    }

    fn locator(&self, source: &WorkerEndpoint, transport: TransferTransport) -> Locator {
        Locator {
            source: source.clone(),
            transport,
            nbytes: self.nbytes,
            dtype: dtype_name(self.spans.dtype()).to_owned(),
            shape: self.shape.clone(),
            offset: self.offset.clone(),
            device: device_name(self.spans.device()),
        }
    }
}

/// An export whose location is complete once the work it issued was
/// performed. Its transport completes it ([`Transport::locate`]) or gives
/// it up ([`Transport::discard`]); the caller keeps it until one of them,
/// called only then: at once after a stream, from the job's completion on
/// after a recorder.
pub(crate) enum Staged {
    /// A channel export whose bytes are staged on the host. They travel in
    /// the location itself, so it exists once they were copied.
    Channel {
        /// The location without its payload.
        locator: Locator,
        staging: Arc<Memory>,
        bytes: usize,
    },
    /// A CUDA VMM export fenced by a readiness event. A same-host reader
    /// waits on the event through its interprocess handle, which exists
    /// once the event was recorded, and reads the export through its
    /// endpoint's registration, made with the complete location.
    Fenced {
        /// The location without its event handle.
        locator: Locator,
        /// The readiness event whose handle the location carries; `None`
        /// when readers on other hosts do not wait on it.
        ready: Option<Arc<Event>>,
        buffer: Arc<TransportBuffer>,
    },
}

/// The producer stream a device export is ordered on.
fn producer<'s>(spans: &Spans, stream: Option<&'s Stream>) -> Result<Option<(i32, &'s Stream)>> {
    let Device::Cuda(device) = spans.device() else {
        return Ok(None);
    };
    let stream = stream
        .ok_or_else(|| Error::Invalid("a device export requires its producer stream".into()))?;
    if stream.device().0 != device {
        return Err(Error::Invalid(
            "an export's producer stream is on another device".into(),
        ));
    }
    Ok(Some((device, stream)))
}

impl Transport {
    pub fn new(
        kind: TransportKind,
        capacity: Arc<Capacity>,
        events: Arc<EventPool>,
        options: &TransportOptions,
    ) -> Result<Self> {
        let suffix = uuid::Uuid::new_v4().simple();
        let endpoint = |limit| {
            Endpoint::new(
                format!("uniserve-buffers-{suffix}"),
                limit,
                Arc::clone(&events),
            )
        };
        let backend = match kind {
            TransportKind::Local => Backend::Local {
                // Nonempty exports make the byte budget an upper bound on
                // registration count as well.
                endpoint: endpoint(capacity.capacity() as usize)?,
                next: AtomicU64::new(0),
            },
            TransportKind::Shm => Backend::Shared {
                endpoint: endpoint(REGISTERED_EXPORTS)?,
                slot: options.acknowledgment_slot,
                host_slots: options.host_slots.clone(),
            },
            TransportKind::CudaVmm => Backend::Cuda(CudaExports::new(
                endpoint(REGISTERED_EXPORTS)?,
                options.acknowledgment_slot,
                options.cross_host_consumers,
                options.vmm_pool_bytes,
            )),
            TransportKind::Channel => Backend::Channel {
                name: format!("uniserve-channel-{suffix}"),
                host_slots: options.host_slots.clone(),
            },
        };
        let workers = if kind == TransportKind::Channel { 1 } else { 2 };
        let reads = Arc::new(ReadPool::new(
            Arc::clone(&capacity),
            workers,
            &format!("uniserve-{}-read", kind.name()),
        )?);
        Ok(Self {
            source: options.source.clone(),
            capacity,
            context: ReadContext::new(Arc::clone(&events)),
            events,
            reads,
            borrowed: Arc::new(Mutex::new(Some(HashMap::new()))),
            backend,
        })
    }

    pub fn kind(&self) -> TransportKind {
        match self.backend {
            Backend::Local { .. } => TransportKind::Local,
            Backend::Shared { .. } => TransportKind::Shm,
            Backend::Cuda(_) => TransportKind::CudaVmm,
            Backend::Channel { .. } => TransportKind::Channel,
        }
    }

    pub fn capacity(&self) -> &Arc<Capacity> {
        &self.capacity
    }

    fn endpoint_name(&self) -> &str {
        match &self.backend {
            Backend::Local { endpoint, .. } | Backend::Shared { endpoint, .. } => endpoint.name(),
            Backend::Cuda(cuda) => cuda.endpoint.name(),
            Backend::Channel { name, .. } => name,
        }
    }

    fn endpoint(&self) -> Option<&Arc<Endpoint>> {
        match &self.backend {
            Backend::Local { endpoint, .. } | Backend::Shared { endpoint, .. } => Some(endpoint),
            Backend::Cuda(cuda) => Some(&cuda.endpoint),
            Backend::Channel { .. } => None,
        }
    }

    /// Whether this mechanism reads `locator`: a location of its kind, which
    /// for a POSIX segment must be on this rank's node.
    pub fn reads(&self, locator: &Locator) -> bool {
        TransportKind::of(locator) == self.kind()
            && (self.kind() != TransportKind::Shm || locator.source.node == self.source.node)
    }

    /// Whether this mechanism reaches any of `consumers` (every mechanism
    /// serves an export without declared consumers).
    pub fn serves(&self, consumers: &[usize]) -> bool {
        if consumers.is_empty() {
            return true;
        }
        match &self.backend {
            Backend::Shared { host_slots, .. } => {
                consumers.iter().any(|slot| host_slots.contains(slot))
            }
            Backend::Channel { host_slots, .. } => {
                consumers.iter().any(|slot| !host_slots.contains(slot))
            }
            _ => true,
        }
    }

    /// Export `source` (produced on `stream`'s stream when it is device
    /// memory) at `offset` in its logical tensor, for `consumers`'
    /// acknowledgment slots.
    ///
    /// The export reserves its bytes before preparing backing; a successful
    /// registration holds them until producer and consumer accesses retired.
    ///
    /// `stream` is what the export's device work is issued to: its fence,
    /// its copies into shared or pooled storage with the readiness mark
    /// behind them, and the wait a mechanism needs before its bytes or
    /// handle leave the process. A job's recorder defers all of it to the
    /// job, so a location is published only from the job's completion on,
    /// and a channel export, whose bytes travel in the location, is
    /// [`Exported::Staged`] until then. This releases its storage when it
    /// fails, so its caller issues it inside an attempt
    /// ([`Issue::attempt`]), which keeps a job from performing the copies
    /// a failed export recorded.
    pub(crate) fn export(
        &self,
        source: &Spans,
        offset: Option<&[u64]>,
        consumers: &[usize],
        stream: impl Issue,
    ) -> Result<Exported> {
        let view = ExportView::new(source, offset)?;
        match &self.backend {
            Backend::Cuda(cuda) => {
                cuda.check()?;
                CudaExports::validate(source)?;
            }
            Backend::Shared { endpoint, .. } => endpoint.reap()?,
            Backend::Local { .. } | Backend::Channel { .. } => {}
        }
        self.capacity.acquire(view.nbytes)?;

        match &self.backend {
            Backend::Local { endpoint, next } => self
                .export_local(&view, endpoint, next, stream)
                .map(|locator| Exported::Located(Box::new(locator))),
            Backend::Shared { endpoint, .. } => self
                .export_shared(&view, endpoint, consumers, stream)
                .map(|locator| Exported::Located(Box::new(locator))),
            Backend::Cuda(cuda) => {
                let exported = if stream.stream().is_some() {
                    cuda.export(
                        source,
                        stream,
                        consumers,
                        &self.capacity,
                        &self.events,
                        |transport| Ok(view.locator(&self.source, transport)),
                    )
                } else {
                    Err(Error::Invalid(
                        "a device export requires its producer stream".into(),
                    ))
                };
                match exported {
                    Ok(Exported::PoolExhausted) => {
                        self.capacity.release(view.nbytes)?;
                        Ok(Exported::PoolExhausted)
                    }
                    Ok(exported) => Ok(exported),
                    // A failure retains its bytes only when its backing was
                    // registered (which retires them) or could not drain.
                    Err(error) => Err(error),
                }
            }
            Backend::Channel { name, .. } => {
                let result = self.export_channel(&view, name, stream);
                self.capacity.release(view.nbytes)?;
                result.map(|staged| Exported::Staged(Box::new(staged)))
            }
        }
    }

    fn export_local(
        &self,
        view: &ExportView<'_>,
        endpoint: &Arc<Endpoint>,
        next: &AtomicU64,
        stream: impl Issue,
    ) -> Result<Locator> {
        let event = match producer(view.spans, stream.stream()) {
            Ok(Some(_)) => self.events.fence(stream, false).map(Some),
            Ok(None) => Ok(None),
            Err(error) => Err(error),
        };
        let event = match event {
            Ok(event) => event,
            Err(error) => {
                self.capacity.release(view.nbytes)?;
                return Err(error);
            }
        };
        let buffer = Arc::new(TransportBuffer::local(
            view.spans.clone(),
            event,
            view.nbytes,
            Arc::clone(&self.capacity),
        ));
        let locator = view.locator(
            &self.source,
            TransferTransport::Local {
                endpoint: endpoint.name().to_owned(),
                key: next.fetch_add(1, Ordering::Relaxed),
            },
        );
        if let Err(error) = endpoint.register(locator.clone(), Arc::clone(&buffer)) {
            TransportBuffer::reclaim(buffer, &self.events, None)?;
            return Err(error);
        }
        Ok(locator)
    }

    fn export_shared(
        &self,
        view: &ExportView<'_>,
        endpoint: &Arc<Endpoint>,
        consumers: &[usize],
        stream: impl Issue,
    ) -> Result<Locator> {
        let producer = match producer(view.spans, stream.stream()) {
            Ok(producer) => producer,
            Err(error) => {
                self.capacity.release(view.nbytes)?;
                return Err(error);
            }
        };
        // The buffer keeps the producer's stream for its own lifetime: it
        // drains the stream before it unregisters or releases its bytes.
        let shared = producer.as_ref().map(|(_, stream)| (*stream).clone());
        let mut storage = match SharedBuffer::new(view.nbytes as usize, consumers.to_vec(), shared)
        {
            Ok(storage) => storage,
            Err(error) => {
                self.capacity.release(view.nbytes)?;
                return Err(error);
            }
        };

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
                self.capacity.release(view.nbytes)?;
                return Err(error);
            }
        };

        let buffer = Arc::new(TransportBuffer::shared(storage, Arc::clone(&self.capacity)));
        if let Err(error) = endpoint.register(locator.clone(), Arc::clone(&buffer)) {
            TransportBuffer::reclaim(buffer, &self.events, None)?;
            return Err(error);
        }
        Ok(locator)
    }

    /// Complete a staged export of this transport: the location with the
    /// channel bytes its copy staged, or with the handle of the readiness
    /// event its work recorded, registered with the endpoint.
    pub(crate) fn locate(&self, staged: Staged) -> Result<Locator> {
        match staged {
            Staged::Channel {
                mut locator,
                staging,
                bytes,
            } => {
                // SAFETY: the export's copy and the wait behind it were
                // performed, so nothing writes the staging any more.
                let payload = unsafe { staging.host_bytes() }
                    .ok_or_else(|| Error::Invariant("channel staging is host memory".into()))?
                    [..bytes]
                    .to_vec();
                match &mut locator.transport {
                    TransferTransport::Channel {
                        payload: carried, ..
                    } => *carried = payload,
                    _ => {
                        return Err(Error::Invariant(
                            "a staged channel export has a channel location".into(),
                        ));
                    }
                }
                Ok(locator)
            }
            Staged::Fenced {
                mut locator,
                ready,
                buffer,
            } => {
                let registered = (|| -> Result<()> {
                    let Backend::Cuda(cuda) = &self.backend else {
                        return Err(Error::Invariant(
                            "a fenced export belongs to a CUDA VMM transport".into(),
                        ));
                    };
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
                    cuda.endpoint.register(locator.clone(), Arc::clone(&buffer))
                })();
                if let Err(error) = registered {
                    TransportBuffer::reclaim(buffer, &self.events, None)?;
                    return Err(error);
                }
                Ok(locator)
            }
        }
    }

    /// Give up a staged export of this transport that is not published:
    /// its backing retires through the event pool.
    pub(crate) fn discard(&self, staged: Staged) -> Result<()> {
        match staged {
            // The staging is dropped; its bytes were never sent.
            Staged::Channel { .. } => Ok(()),
            Staged::Fenced { buffer, .. } => TransportBuffer::reclaim(buffer, &self.events, None),
        }
    }

    fn export_channel(
        &self,
        view: &ExportView<'_>,
        name: &str,
        stream: impl Issue,
    ) -> Result<Staged> {
        let shape: Vec<i64> = view.shape.iter().map(|&extent| extent as i64).collect();
        let bytes = view.nbytes as usize;
        let producer = producer(view.spans, stream.stream())?.map(|(device, _)| device);
        let staging = match producer {
            Some(device) => Memory::pinned(device, bytes.max(1))?,
            None => Memory::host(bytes.max(1)),
        };
        let target = Spans::single(StorageView::contiguous(
            Arc::clone(&staging) as Arc<dyn Backing>,
            0,
            view.spans.dtype(),
            &shape,
        )?)?;
        let mut plan = copy_planner(producer, SourceOrder::Stream)?;
        target.plan_copy(view.spans, &mut plan)?;

        match producer {
            Some(_) => {
                // SAFETY: the staged export keeps the staging alive until
                // its bytes are read, behind the wait below.
                unsafe { stream.copy_strided(&plan) }?;
                // The bytes leave with the result: a downstream reader cannot
                // wait on this stream.
                let _range = crate::profiling::range(c"channel_export_synchronize", None);
                stream.synchronize()?;
            }
            // SAFETY: host operands owned by this export, written in the
            // producer's order.
            None => unsafe { stream.host().copy_strided(&plan) }?,
        }

        Ok(Staged::Channel {
            locator: view.locator(
                &self.source,
                TransferTransport::Channel {
                    endpoint: name.to_owned(),
                    payload: Vec::new(),
                },
            ),
            staging,
            bytes,
        })
    }

    /// Read `region` (in the locator's own coordinates, the whole export by
    /// default) into `destination` on `device`, or borrow a process-local
    /// export in place when no destination is given.
    ///
    /// A device destination records its handoff on `stream`, the stream
    /// that orders the destination's earlier writes; transport copies start
    /// after it. `reservation` supplies the read's credit when the read is
    /// part of a planned fan-out.
    ///
    /// The read is issued on `stream` and started at once
    /// ([`Self::issue_fetch`], [`Read::start`]).
    pub fn fetch(
        &self,
        locator: &Locator,
        device: Device,
        destination: Option<&Spans>,
        region: Option<&[Range<u64>]>,
        reservation: Option<&ReadReservation<Notify>>,
        stream: Option<&Stream>,
    ) -> Result<Arc<Read>> {
        let read = self.issue_fetch(locator, device, destination, region, reservation, stream)?;
        read.start()?;
        Ok(read)
    }

    /// Issue a read as [`Self::fetch`] describes it, with its handoff fence
    /// issued to `stream`. The caller starts the read ([`Read::start`])
    /// once that issuer performed its work, so the fence's record is made
    /// before the read reaches its transport thread: at once after a
    /// stream, from the job's completion on after a recorder.
    pub fn issue_fetch(
        &self,
        locator: &Locator,
        device: Device,
        destination: Option<&Spans>,
        region: Option<&[Range<u64>]>,
        reservation: Option<&ReadReservation<Notify>>,
        stream: impl Issue,
    ) -> Result<Arc<Read>> {
        if TransportKind::of(locator) != self.kind() {
            return Err(Error::Invalid(format!(
                "{} read requires a {} locator",
                self.kind().name(),
                self.kind().name()
            )));
        }
        match &self.backend {
            Backend::Local { .. } => {
                self.fetch_local(locator, device, destination, region, reservation, stream)
            }
            Backend::Shared { slot, .. } => {
                if locator.source.node != self.source.node {
                    return Err(Error::Invalid(
                        "shared storage transport requires the source node".into(),
                    ));
                }
                let destination = read_destination(locator, device, destination, region)?;
                self.submit(
                    ReadCall::Shared {
                        locator: Arc::new(locator.clone()),
                        slot: *slot,
                        destination,
                        region: region.map(<[_]>::to_vec),
                    },
                    locator.nbytes,
                    stream,
                    reservation,
                )
            }
            Backend::Channel { .. } => {
                let destination = read_destination(locator, device, destination, region)?;
                self.submit(
                    ReadCall::Channel {
                        locator: Arc::new(locator.clone()),
                        destination,
                        region: region.map(<[_]>::to_vec),
                    },
                    locator.nbytes,
                    stream,
                    reservation,
                )
            }
            Backend::Cuda(cuda) => {
                let TransferTransport::CudaVmm {
                    allocation_handle, ..
                } = &locator.transport
                else {
                    return Err(Error::Invalid(
                        "CUDA VMM read requires a CUDA VMM locator".into(),
                    ));
                };
                let same_node = locator.source.node == self.source.node;
                if !same_node && allocation_handle.len() != 64 {
                    return Err(Error::Invalid(
                        "a device product crosses hosts only as a fabric handle".into(),
                    ));
                }
                if !matches!(device, Device::Cuda(_)) {
                    return Err(Error::Invalid(
                        "CUDA VMM destination must be a CUDA device".into(),
                    ));
                }
                let destination = read_destination(locator, device, destination, region)?;
                self.submit(
                    ReadCall::Cuda(CudaRead {
                        locator: Arc::new(locator.clone()),
                        destination,
                        region: region.map(<[_]>::to_vec),
                        slot: cuda.slot,
                        local: locator.source.address_space == self.source.address_space,
                        same_node,
                    }),
                    locator.nbytes,
                    stream,
                    reservation,
                )
            }
        }
    }

    fn submit(
        &self,
        call: ReadCall,
        nbytes: u64,
        stream: impl Issue,
        reservation: Option<&ReadReservation<Notify>>,
    ) -> Result<Arc<Read>> {
        let backend = NativeRead {
            read: Read::new(Arc::clone(&self.events), None),
            call,
            context: Arc::clone(&self.context),
        };
        read::submit(&self.reads, backend, nbytes, stream, reservation)
    }

    fn fetch_local(
        &self,
        locator: &Locator,
        device: Device,
        destination: Option<&Spans>,
        region: Option<&[Range<u64>]>,
        reservation: Option<&ReadReservation<Notify>>,
        stream: impl Issue,
    ) -> Result<Arc<Read>> {
        if let Some(error) = self.reads.error() {
            return Err(Error::Shared(error));
        }
        let (buffer, grant) = Endpoint::acquire_read(locator)?;
        let borrowed = destination.is_none();
        let borrower = borrowed.then(|| Arc::downgrade(&self.borrowed));
        let read = Read::new(Arc::clone(&self.events), borrower);
        read.bind_source(grant)?;

        let submitted = (|| -> Result<Arc<Read>> {
            let source = buffer.spans()?;
            let producer = buffer.event();
            if source.device() != device {
                return Err(Error::Invalid(
                    "local binding requires the source device".into(),
                ));
            }
            let source = match region {
                Some(region) => {
                    within(region, &locator.shape)?;
                    source.region(region)?
                }
                None => source,
            };

            match destination {
                Some(destination) => {
                    let target = read_destination(locator, device, Some(destination), region)?;
                    let backend = NativeRead {
                        read: Arc::clone(&read),
                        call: ReadCall::Copy {
                            source,
                            destination: target,
                            producer,
                        },
                        context: Arc::clone(&self.context),
                    };
                    read::submit(&self.reads, backend, locator.nbytes, stream, reservation)
                }
                None => {
                    {
                        let mut reads =
                            self.borrowed.lock().unwrap_or_else(PoisonError::into_inner);
                        let reads = reads
                            .as_mut()
                            .ok_or(Error::Resource("transfer pool is closed"))?;
                        reads.insert(Arc::as_ptr(&read) as usize, Arc::downgrade(&read));
                    }
                    read::borrow(
                        &read,
                        &self.capacity,
                        source,
                        producer,
                        reservation,
                        self.reads.completion_wake(),
                    )?;
                    Ok(Arc::clone(&read))
                }
            }
        })();
        if submitted.is_err()
            && let Err(cleanup) = read.release_source()
        {
            tracing::warn!(target: "uniserve_worker.transfer", "failed to release a local transfer source: {cleanup}");
        }
        submitted
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
        if locator.source.node != self.source.node {
            return Err(Error::Invalid(
                "shared storage transport requires the source node".into(),
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

    /// Revoke future reads of an export. The returned signal completes after
    /// physical retirement; `None` once the export left the registry.
    pub fn release(&self, locator: &Locator) -> Result<Option<Arc<Retirement>>> {
        match self.endpoint() {
            Some(endpoint) => endpoint.release(locator),
            None => {
                self.require_channel(locator)?;
                Ok(None)
            }
        }
    }

    /// The physical retirement signal of an export.
    pub fn retirement(&self, locator: &Locator) -> Result<Arc<Retirement>> {
        match self.endpoint() {
            Some(endpoint) => endpoint.retirement(locator),
            None => {
                // Channel bytes left with the locator: nothing remains.
                self.require_channel(locator)?;
                let done = Arc::new(Retirement::default());
                notify(done.complete(Ok(()))?);
                Ok(done)
            }
        }
    }

    fn require_channel(&self, locator: &Locator) -> Result<()> {
        match &locator.transport {
            TransferTransport::Channel { endpoint, .. } if endpoint == self.endpoint_name() => {
                Ok(())
            }
            _ => Err(Error::Invalid(
                "channel export belongs to another endpoint".into(),
            )),
        }
    }

    /// Install the service wake fired when reads complete or exports retire.
    pub fn set_wake(&self, wake: Option<Wake>) {
        if let Some(endpoint) = self.endpoint() {
            endpoint.set_wake(wake.clone());
        }
        self.reads
            .set_completion_wake(wake.map(|wake| -> Notify { Arc::new(move || wake.wake()) }));
    }

    pub fn reap(&self) -> Result<()> {
        match &self.backend {
            Backend::Shared { endpoint, .. } => endpoint.reap(),
            Backend::Cuda(cuda) => {
                cuda.reap()?;
                cuda.endpoint.reap()
            }
            Backend::Local { endpoint, .. } => endpoint.reap(),
            Backend::Channel { .. } => Ok(()),
        }
    }

    /// Whether remote readers still hold exports; their acknowledgments have
    /// no wake source, so the service polls while this holds.
    pub fn awaiting_acknowledgment(&self) -> bool {
        match &self.backend {
            Backend::Shared { endpoint, .. } => endpoint.awaiting_acknowledgment(),
            Backend::Cuda(cuda) => cuda.awaiting_acknowledgment(),
            _ => false,
        }
    }

    /// Join reads, drain borrowed consumers and retire every export. Fails,
    /// retaining storage, while remote readers still hold exports.
    pub fn close(&self) -> Result<()> {
        let borrowed = self
            .borrowed
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .unwrap_or_default();
        let mut result = match self.reads.close() {
            Some(error) => Err(Error::Shared(error)),
            None => Ok(()),
        };
        for read in borrowed
            .into_values()
            .filter_map(|read| Weak::upgrade(&read))
        {
            result = result.and(read.drain());
        }
        self.context.clear();
        if let Some(endpoint) = self.endpoint()
            && let Err(error) = endpoint.close()
        {
            // Active readers still own pool chunks: keep the allocation and
            // the descriptor service until the registry drains.
            return result.and(Err(error));
        }
        if let Backend::Cuda(cuda) = &self.backend {
            result = result.and(cuda.close());
        }
        result
    }
}

/// Check that `region` lies inside `shape`.
fn within(region: &[Range<u64>], shape: &[u64]) -> Result<()> {
    if region.len() != shape.len()
        || region
            .iter()
            .zip(shape)
            .any(|(range, &extent)| range.start > range.end || range.end > extent)
    {
        return Err(Error::Invalid(
            "read region exceeds the exported view".into(),
        ));
    }
    Ok(())
}

/// Validate a read's destination against the region of the export it reads.
fn read_destination(
    locator: &Locator,
    device: Device,
    destination: Option<&Spans>,
    region: Option<&[Range<u64>]>,
) -> Result<Spans> {
    let destination = destination.ok_or_else(|| {
        Error::Invalid("a transport read requires a Rust-owned destination".into())
    })?;
    let shape: Vec<u64> = match region {
        Some(region) => {
            within(region, &locator.shape)?;
            region.iter().map(|range| range.end - range.start).collect()
        }
        None => locator.shape.clone(),
    };
    destination.validate_destination(&shape, parse_dtype(&locator.dtype)?, device)?;
    Ok(destination.clone())
}

/// The byte offset and length of whole leading rows of a shared export;
/// `region`'s leading range is in logical coordinates.
fn row_span(locator: &Locator, region: Option<&[Range<u64>]>) -> Result<(usize, usize)> {
    let element = parse_dtype(&locator.dtype)?.bytes() as u64;
    let shape = &locator.shape;
    let rows = shape.first().copied().unwrap_or(1);
    let row_bytes = element * shape.iter().skip(1).product::<u64>();
    let Some(region) = region else {
        return Ok((0, (row_bytes * rows) as usize));
    };
    if region.len() != shape.len()
        || region
            .iter()
            .zip(shape)
            .skip(1)
            .any(|(range, &extent)| range.start != 0 || range.end != extent)
    {
        return Err(Error::Invalid(
            "a borrowed span covers whole rows of the leading axis".into(),
        ));
    }
    let origin = locator.offset.first().copied().unwrap_or(0);
    let first = region[0].start.checked_sub(origin);
    let last = region[0].end.checked_sub(origin);
    match (first, last) {
        (Some(first), Some(last)) if first < last && last <= rows => Ok((
            (first * row_bytes) as usize,
            ((last - first) * row_bytes) as usize,
        )),
        _ => Err(Error::Invalid(
            "a borrowed span lies outside its location".into(),
        )),
    }
}

/// The transports a rank configured, sharing one capacity and event pool.
///
/// Every configured mechanism reads other ranks' exports; a rank publishes
/// its own products only over its export mechanisms (the outgoing edges of
/// its placement), which default to all of them.
pub struct Transports {
    transports: BTreeMap<TransportKind, Arc<Transport>>,
    exports: BTreeSet<TransportKind>,
    capacity: Arc<Capacity>,
}

impl Transports {
    pub fn new(
        kinds: &[TransportKind],
        capacity: Arc<Capacity>,
        events: Arc<EventPool>,
        options: &TransportOptions,
    ) -> Result<Self> {
        let transports = kinds
            .iter()
            .map(|&kind| {
                Ok((
                    kind,
                    Arc::new(Transport::new(
                        kind,
                        Arc::clone(&capacity),
                        Arc::clone(&events),
                        options,
                    )?),
                ))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            transports,
            exports: kinds.iter().copied().collect(),
            capacity,
        })
    }

    /// Publish products only over `kinds`, each of which must be configured.
    pub fn with_exports(mut self, kinds: &[TransportKind]) -> Result<Self> {
        if let Some(kind) = kinds
            .iter()
            .find(|kind| !self.transports.contains_key(kind))
        {
            return Err(Error::Invalid(format!(
                "export mechanism {} is not a configured transport",
                kind.name()
            )));
        }
        self.exports = kinds.iter().copied().collect();
        Ok(self)
    }

    pub fn get(&self, kind: TransportKind) -> Option<&Arc<Transport>> {
        self.transports.get(&kind)
    }

    /// The transport of `kind` when the rank publishes over it.
    pub fn exporter(&self, kind: TransportKind) -> Option<&Arc<Transport>> {
        self.exports
            .contains(&kind)
            .then(|| self.transports.get(&kind))
            .flatten()
    }

    pub fn is_empty(&self) -> bool {
        self.transports.is_empty()
    }

    /// Whether the rank publishes over no mechanism.
    pub fn exports_nothing(&self) -> bool {
        self.exports.is_empty()
    }

    /// Whether the rank publishes over a mechanism other ranks can read:
    /// anything but the process-local registry.
    pub fn exports_remotely(&self) -> bool {
        self.exports
            .iter()
            .any(|&kind| kind != TransportKind::Local)
    }

    pub fn capacity(&self) -> &Arc<Capacity> {
        &self.capacity
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<Transport>> {
        self.transports.values()
    }

    pub fn set_wake(&self, wake: Option<Wake>) {
        for transport in self.iter() {
            transport.set_wake(wake.clone());
        }
    }

    pub fn reap(&self) -> Result<()> {
        self.iter().try_for_each(|transport| transport.reap())
    }

    pub fn awaiting_acknowledgment(&self) -> bool {
        self.iter()
            .any(|transport| transport.awaiting_acknowledgment())
    }

    /// Close every transport, returning the first failure.
    pub fn close(&self) -> Result<()> {
        let mut result = Ok(());
        for transport in self.iter() {
            result = result.and(transport.close());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn locator(shape: Vec<u64>, offset: Vec<u64>) -> Locator {
        Locator {
            source: WorkerEndpoint {
                worker_id: "w".into(),
                rank: 0,
                node: "n".into(),
                address_space: "a".into(),
                incarnation: "i".into(),
            },
            transport: TransferTransport::PosixShm {
                endpoint: "e".into(),
                name: "s".into(),
            },
            nbytes: shape.iter().product::<u64>() * 2,
            dtype: "bfloat16".into(),
            shape,
            offset,
            device: "cpu".into(),
        }
    }

    #[test]
    fn borrowed_rows_are_whole_rows_relative_to_the_location() -> Result<()> {
        let rows = locator(vec![4, 3], vec![10, 0]);
        assert_eq!(row_span(&rows, None)?, (0, 24));
        assert_eq!(row_span(&rows, Some(&[11..13, 0..3]))?, (6, 12));
        assert!(row_span(&rows, Some(&[11..13, 0..2])).is_err());
        assert!(row_span(&rows, Some(&[9..13, 0..3])).is_err());
        assert!(row_span(&rows, Some(&[12..15, 0..3])).is_err());
        Ok(())
    }
}
