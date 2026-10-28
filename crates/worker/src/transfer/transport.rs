//! One rank's export and read mechanisms.
//!
//! A [`Transport`] exports device or host tensors through one mechanism and
//! reads locators of that mechanism into Rust-owned destinations:
//!
//! | Kind | Export | Read |
//! | --- | --- | --- |
//! | local | register the spans with a producer fence | borrow in place, or copy |
//! | posix_shm | copy into a POSIX shared-memory segment | stage on the host, then copy |
//! | cuda_vmm | export in place or copy into a VMM pool chunk | map the handle, then copy |
//! | channel | copy into the locator's payload bytes | stage on the host, then copy |
//!
//! All transports of a rank share one [`TransferCapacity`]: export bytes
//! are held until physical retirement, and reads hold a read credit plus
//! their bytes until their copy drains.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ops::Range;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, Weak};

use cuda::{DeviceId, Event, Stream};
use ipc::{Locator, TransferTransport, Wake, WorkerEndpoint};

use super::backing::TransportBuffer;
use super::export_table::ExportTable;
use super::pool::TransferPool;
use super::read::{self, BorrowedReads, NativeRead, Read, ReadCall, ReadContext};
use super::vmm::{CudaExports, Exported};
use super::{
    ByteReservation, Notify, ReadReservation, Retirement, Spans, TransferCapacity, notify,
};
use crate::ffi::{DType, MemoryKind};
use crate::storage::allocation::Storage;
use crate::stream::Issue;
use crate::{Error, EventPool, Result, lock};

/// A transfer mechanism.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TransportKind {
    Local,
    PosixShm,
    CudaVmm,
    Channel,
}

impl TransportKind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::PosixShm => "posix_shm",
            Self::CudaVmm => "cuda_vmm",
            Self::Channel => "channel",
        }
    }

    pub fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "local" => Self::Local,
            "posix_shm" => Self::PosixShm,
            "cuda_vmm" => Self::CudaVmm,
            "channel" => Self::Channel,
            _ => return Err(Error::Invalid(format!("unknown transport: {name}"))),
        })
    }

    /// The mechanism a locator was exported through.
    pub fn of(locator: &Locator) -> Self {
        match locator.transport {
            TransferTransport::Local { .. } => Self::Local,
            TransferTransport::PosixShm { .. } => Self::PosixShm,
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

pub(super) enum Backend {
    Local {
        endpoint: Arc<ExportTable>,
        next: AtomicU64,
    },
    Shared {
        endpoint: Arc<ExportTable>,
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
    pub(super) source: WorkerEndpoint,
    pub(super) capacity: Arc<TransferCapacity>,
    pub(super) events: Arc<EventPool>,
    pub(super) reads: Arc<TransferPool>,
    pub(super) context: Arc<ReadContext>,
    pub(super) borrowed: Arc<BorrowedReads>,
    pub(super) backend: Backend,
}

/// The logical extent of an export: its shape, its offset in the logical
/// tensor, and its payload bytes.
pub(super) struct ExportView<'a> {
    pub(super) spans: &'a Spans,
    pub(super) shape: Vec<u64>,
    pub(super) offset: Vec<u64>,
    pub(super) nbytes: u64,
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

    pub(super) fn locator(&self, source: &WorkerEndpoint, transport: TransferTransport) -> Locator {
        Locator {
            source: source.clone(),
            transport,
            nbytes: self.nbytes,
            dtype: self.spans.dtype().name().to_owned(),
            shape: self.shape.clone(),
            offset: self.offset.clone(),
            device: self.spans.device().locator_name(),
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
        staging: Arc<Storage>,
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
pub(super) fn producer<'s>(
    spans: &Spans,
    stream: Option<&'s Stream>,
) -> Result<Option<(DeviceId, &'s Stream)>> {
    let MemoryKind::Cuda(device) = spans.device() else {
        return Ok(None);
    };
    let stream = stream
        .ok_or_else(|| Error::Invalid("a device export requires its producer stream".into()))?;
    if stream.device() != device {
        return Err(Error::Invalid(
            "an export's producer stream is on another device".into(),
        ));
    }
    Ok(Some((device, stream)))
}

impl Transport {
    pub(crate) fn new(
        kind: TransportKind,
        capacity: Arc<TransferCapacity>,
        events: Arc<EventPool>,
        options: &TransportOptions,
    ) -> Result<Self> {
        let suffix = uuid::Uuid::new_v4().simple();
        let endpoint =
            || ExportTable::new(format!("uniserve-buffers-{suffix}"), Arc::clone(&events));
        let backend = match kind {
            TransportKind::Local => Backend::Local {
                endpoint: endpoint(),
                next: AtomicU64::new(0),
            },
            TransportKind::PosixShm => Backend::Shared {
                endpoint: endpoint(),
                slot: options.acknowledgment_slot,
                host_slots: options.host_slots.clone(),
            },
            TransportKind::CudaVmm => Backend::Cuda(CudaExports::new(
                endpoint(),
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
        let reads = Arc::new(TransferPool::new(
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

    pub(crate) fn kind(&self) -> TransportKind {
        match self.backend {
            Backend::Local { .. } => TransportKind::Local,
            Backend::Shared { .. } => TransportKind::PosixShm,
            Backend::Cuda(_) => TransportKind::CudaVmm,
            Backend::Channel { .. } => TransportKind::Channel,
        }
    }

    pub(crate) fn capacity(&self) -> &Arc<TransferCapacity> {
        &self.capacity
    }

    fn endpoint_name(&self) -> &str {
        match &self.backend {
            Backend::Local { endpoint, .. } | Backend::Shared { endpoint, .. } => endpoint.name(),
            Backend::Cuda(cuda) => cuda.endpoint.name(),
            Backend::Channel { name, .. } => name,
        }
    }

    fn endpoint(&self) -> Option<&Arc<ExportTable>> {
        match &self.backend {
            Backend::Local { endpoint, .. } | Backend::Shared { endpoint, .. } => Some(endpoint),
            Backend::Cuda(cuda) => Some(&cuda.endpoint),
            Backend::Channel { .. } => None,
        }
    }

    /// Whether this mechanism reads `locator`, a location of its kind: a
    /// POSIX segment only on this rank's host.
    pub(crate) fn reads(&self, locator: &Locator) -> bool {
        self.kind() != TransportKind::PosixShm || locator.source.host == self.source.host
    }

    /// Whether this mechanism reaches any of `consumers` (every mechanism
    /// serves an export without declared consumers).
    pub(crate) fn serves(&self, consumers: &[usize]) -> bool {
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
    /// storage) at `offset` in its logical tensor, for `consumers`'
    /// acknowledgment slots.
    ///
    /// `bytes` is this mechanism's share of the export's byte reservation,
    /// which every mechanism publishing the same export shares; a successful
    /// registration holds it until producer and consumer accesses retired.
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
        bytes: ByteReservation,
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
        // The export's backing holds its share until it retires; a failed
        // attempt returns it, except a VMM export whose producer stream could
        // not drain, which retains it with everything it touched.

        match &self.backend {
            Backend::Local { endpoint, next } => self
                .export_local(&view, endpoint, next, bytes, stream)
                .map(|locator| Exported::Located(Box::new(locator))),
            Backend::Shared { endpoint, .. } => self
                .export_shared(&view, endpoint, consumers, bytes, stream)
                .map(|locator| Exported::Located(Box::new(locator))),
            Backend::Cuda(cuda) => cuda.export(
                source,
                stream,
                consumers,
                bytes,
                &self.events,
                |transport| Ok(view.locator(&self.source, transport)),
            ),
            Backend::Channel { name, .. } => {
                // The location carries the bytes, so this share covers only
                // the staging copy.
                let result = self.export_channel(&view, name, stream);
                bytes.release()?;
                result.map(|staged| Exported::Staged(Box::new(staged)))
            }
        }
    }

    /// Complete a staged export of this transport: the location with the
    /// channel bytes its copy staged, or with the handle of the readiness
    /// event its work recorded, registered with the endpoint.
    pub(crate) fn locate(&self, staged: Staged) -> Result<Locator> {
        match staged {
            Staged::Channel {
                locator,
                staging,
                bytes,
            } => super::channel::locate(locator, &staging, bytes),
            Staged::Fenced {
                locator,
                ready,
                buffer,
            } => match &self.backend {
                Backend::Cuda(cuda) => cuda.locate(&self.events, locator, ready, buffer),
                _ => {
                    TransportBuffer::reclaim(buffer, &self.events, None)?;
                    Err(Error::Invariant(
                        "a fenced export belongs to a CUDA VMM transport".into(),
                    ))
                }
            },
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

    /// Issue a read of `region` (in the locator's own coordinates, the whole
    /// export by default) into `destination` on `device`, or a borrow of a
    /// process-local export in place when no destination is given.
    ///
    /// A device destination's handoff fence is issued to `stream`, the
    /// stream that orders the destination's earlier writes; transport copies
    /// start after it. `reservation` supplies the read's credit when the read
    /// is part of a planned fan-out. The caller starts the read
    /// ([`Read::start`]) once that issuer performed its work, so the fence's
    /// record is made before the read reaches its transport thread: at once
    /// after a stream, from the job's completion on after a recorder.
    pub fn issue_fetch(
        &self,
        locator: &Locator,
        device: MemoryKind,
        destination: Option<&Spans>,
        region: Option<&[Range<u64>]>,
        reservation: Option<&ReadReservation>,
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
                let call = self.shared_read(*slot, locator, device, destination, region)?;
                self.submit(call, locator.nbytes, stream, reservation)
            }
            Backend::Channel { .. } => {
                let call = super::channel::read(locator, device, destination, region)?;
                self.submit(call, locator.nbytes, stream, reservation)
            }
            Backend::Cuda(cuda) => {
                let call = cuda.read(&self.source, locator, device, destination, region)?;
                self.submit(call, locator.nbytes, stream, reservation)
            }
        }
    }

    fn submit(
        &self,
        call: ReadCall,
        nbytes: u64,
        stream: impl Issue,
        reservation: Option<&ReadReservation>,
    ) -> Result<Arc<Read>> {
        let backend = NativeRead {
            read: Read::new(Arc::clone(&self.events), None),
            call,
            context: Arc::clone(&self.context),
        };
        read::submit(&self.reads, backend, nbytes, stream, reservation)
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
    pub(crate) fn retirement(&self, locator: &Locator) -> Result<Arc<Retirement>> {
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
    pub(crate) fn set_wake(&self, wake: Option<Wake>) {
        if let Some(endpoint) = self.endpoint() {
            endpoint.set_wake(wake.clone());
        }
        self.reads
            .set_completion_wake(wake.map(|wake| -> Notify { Arc::new(move || wake.wake()) }));
    }

    pub(crate) fn reap(&self) -> Result<()> {
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
    pub(crate) fn awaiting_acknowledgment(&self) -> bool {
        match &self.backend {
            Backend::Shared { endpoint, .. } => endpoint.awaiting_acknowledgment(),
            Backend::Cuda(cuda) => cuda.awaiting_acknowledgment(),
            _ => false,
        }
    }

    /// Join reads, drain borrowed consumers and retire every export. Fails,
    /// retaining storage, while remote readers still hold exports.
    pub(crate) fn close(&self) -> Result<()> {
        let borrowed = lock(&self.borrowed).take().unwrap_or_default();
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
pub(super) fn within(region: &[Range<u64>], shape: &[u64]) -> Result<()> {
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
pub(super) fn read_destination(
    locator: &Locator,
    device: MemoryKind,
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
    destination.validate_destination(&shape, DType::parse(&locator.dtype)?, device)?;
    Ok(destination.clone())
}

/// The byte offset and length of whole leading rows of a shared export;
/// `region`'s leading range is in logical coordinates.
pub(super) fn row_span(locator: &Locator, region: Option<&[Range<u64>]>) -> Result<(usize, usize)> {
    let element = DType::parse(&locator.dtype)?.bytes() as u64;
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
    capacity: Arc<TransferCapacity>,
}

impl Transports {
    pub fn new(
        kinds: &[TransportKind],
        capacity: Arc<TransferCapacity>,
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
    pub(crate) fn with_exports(mut self, kinds: &[TransportKind]) -> Result<Self> {
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
    pub(crate) fn exporter(&self, kind: TransportKind) -> Option<&Arc<Transport>> {
        self.exports
            .contains(&kind)
            .then(|| self.transports.get(&kind))
            .flatten()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.transports.is_empty()
    }

    /// Whether the rank publishes over no mechanism.
    pub(crate) fn exports_nothing(&self) -> bool {
        self.exports.is_empty()
    }

    /// Whether the rank publishes over a mechanism other ranks can read:
    /// anything but the process-local registry.
    pub(crate) fn exports_remotely(&self) -> bool {
        self.exports
            .iter()
            .any(|&kind| kind != TransportKind::Local)
    }

    pub fn capacity(&self) -> &Arc<TransferCapacity> {
        &self.capacity
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &Arc<Transport>> {
        self.transports.values()
    }

    pub(crate) fn set_wake(&self, wake: Option<Wake>) {
        for transport in self.iter() {
            transport.set_wake(wake.clone());
        }
    }

    pub fn reap(&self) -> Result<()> {
        self.iter().try_for_each(|transport| transport.reap())
    }

    pub(crate) fn awaiting_acknowledgment(&self) -> bool {
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
                host: "n".into(),
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
