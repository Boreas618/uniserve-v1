//! Transfer admission, transports, readable results, and physical retirement.
//!
//! The generic cores ([`TransferCapacity`], [`TransferTicket`],
//! [`TransferPool`], [`BufferRegistry`], [`KVImporter`]) own admission,
//! cancellation and retirement rules; the native layer instantiates them with
//! Rust-owned storage ([`Spans`] of [`crate::storage::memory::StorageView`]s),
//! driver copies and the shared [`EventPool`].

use std::collections::HashSet;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use uniserve_cuda::{DeviceGuard, DeviceId, Event, Stream};

use crate::{Completion, Error, EventPool, Outcome, Result};

mod cuda_vmm;
mod descriptor_grants;
mod endpoint;
pub mod exports;
pub mod fetch;
pub mod kv_export;
mod kv_import;
mod peer;
mod pool;
mod read;
mod registry;
mod shared_buffer;
mod source;
mod spans;
mod transport;
pub mod vmm_pool;

pub use descriptor_grants::{DescriptorGrants, fetch_descriptor};
pub(crate) use kv_import::{CacheImporter, ImportTask, KVImportBackend, Retired};
pub use kv_import::{
    ImportBackend, ImportCopy, ImportEntry, ImportWorkspace, KVImport, KVImporter,
};
pub use peer::{PeerExchange, PeerStorage};
pub use read::{Read, ReadRef};
pub use registry::{BufferRegistry, RegisteredBuffer};
pub use shared_buffer::{SHM_HEADER_BYTES, SharedBuffer, SharedMapping, SharedRead};
pub use spans::{Spans, copy_on_host, copy_planner, submit_copy};
pub use transport::{Transport, TransportKind, TransportOptions, Transports};

pub use pool::{ReadBackend, TransferPool, TransferRead};

/// A completion or readiness observer. Observers only notify (wake a
/// submission or the service loop); they never block or call back into the
/// owner that notifies them.
pub type Notify = Arc<dyn Fn() + Send + Sync>;

/// The physical retirement signal of an export, a read or a KV import.
pub type Retirement = Completion<Error, Notify>;

/// A rank's shared transfer byte budget and read credits.
pub type Capacity = TransferCapacity<Notify>;

/// Run observers returned by a completion, after its owner unlocked.
pub fn notify(callbacks: Vec<Notify>) {
    for callback in callbacks {
        callback();
    }
}

/// One source's contribution to a logical tensor fetch, in source-local and
/// destination-local coordinates.
pub struct ReadRegion {
    pub location: usize,
    pub source: Vec<Range<u64>>,
    pub destination: Vec<Range<u64>>,
}

/// Cover a requested region from bound locations in preference order.
///
/// Location bounds are already validated by their transfer descriptor. All
/// bounds use the logical tensor's axes. Missing coverage rejects the complete
/// plan, before any destination can be written. Return `None` for missing
/// coverage, and stop asking for locations as soon as coverage is complete.
/// An empty region needs no reads. Source lookup errors propagate unchanged.
pub fn plan_reads<E>(
    region: &[Range<u64>],
    locations: impl IntoIterator<Item = std::result::Result<(usize, Vec<Range<u64>>), E>>,
) -> std::result::Result<Option<Vec<ReadRegion>>, E> {
    if region.iter().any(Range::is_empty) {
        return Ok(Some(Vec::new()));
    }

    let mut missing = vec![region.to_vec()];
    let mut reads = Vec::new();
    for location in locations {
        let (location, covered) = location?;
        let mut remaining = Vec::new();
        for required in missing {
            if let Some(overlap) =
                uniserve_ipc::subtract_region(&required, &covered, &mut remaining)
            {
                reads.push(ReadRegion {
                    location,
                    source: relative_region(&overlap, &covered),
                    destination: relative_region(&overlap, region),
                });
            }
        }

        if remaining.is_empty() {
            return Ok(Some(reads));
        }
        missing = remaining;
    }

    Ok(None)
}

fn relative_region(region: &[Range<u64>], origin: &[Range<u64>]) -> Vec<Range<u64>> {
    region
        .iter()
        .zip(origin)
        .map(|(axis, base)| axis.start - base.start..axis.end - base.start)
        .collect()
}

struct CapacityState<C> {
    used: u64,
    reads_free: usize,
    returns: u64,
    waiters: Vec<C>,
}

/// Shared byte reservations and physical read credits for one worker rank.
/// The backend dispatches notifications after the capacity lock is released.
pub struct TransferCapacity<C> {
    capacity: u64,
    ticket_capacity: usize,
    state: Mutex<CapacityState<C>>,
    notify: fn(Vec<C>),
}

impl<C> TransferCapacity<C> {
    pub fn new(capacity: u64, ticket_capacity: usize, notify: fn(Vec<C>)) -> Result<Self> {
        if capacity == 0 || ticket_capacity == 0 {
            return Err(Error::Invalid(
                "transfer byte and read capacities must be positive".into(),
            ));
        }

        Ok(Self {
            capacity,
            ticket_capacity,
            state: Mutex::new(CapacityState {
                used: 0,
                reads_free: ticket_capacity,
                returns: 0,
                waiters: Vec::new(),
            }),
            notify,
        })
    }

    fn state(&self) -> MutexGuard<'_, CapacityState<C>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn ticket_capacity(&self) -> usize {
        self.ticket_capacity
    }

    pub fn used(&self) -> u64 {
        self.state().used
    }

    pub fn acquire(&self, amount: u64) -> Result<()> {
        let mut state = self.state();
        if amount > self.capacity - state.used {
            return Err(Error::Resource("transfer byte capacity is exhausted"));
        }

        state.used += amount;
        Ok(())
    }

    pub fn release(&self, amount: u64) -> Result<()> {
        let mut state = self.state();
        if amount > state.used {
            return Err(Error::State(
                "transfer byte release exceeds the live reservation",
            ));
        }

        state.used -= amount;
        Ok(())
    }

    /// Take the complete fan-out before starting any reads. Backpressure
    /// includes the return sequence used to subscribe without missing a wake.
    pub fn take_reads(&self, count: usize) -> Result<()> {
        if count == 0 {
            return Err(Error::Invalid(
                "a read ticket reservation takes at least one".into(),
            ));
        }
        if count > self.ticket_capacity {
            return Err(Error::Unsupported(format!(
                "a fetch of {count} reads exceeds the rank's {} read tickets",
                self.ticket_capacity
            )));
        }

        let mut state = self.state();
        if count > state.reads_free {
            return Err(Error::ReadBackpressure {
                returns: state.returns,
            });
        }

        state.reads_free -= count;
        Ok(())
    }

    /// Admit one borrowed or copied read from the rank's shared reservation.
    pub fn take_read(self: &Arc<Self>, reservation: Option<&ReadReservation<C>>) -> Result<()> {
        if let Some(reservation) = reservation {
            if !Arc::ptr_eq(reservation.capacity(), self) {
                return Err(Error::Invalid(
                    "read reservation belongs to another transfer capacity".into(),
                ));
            }
            reservation.use_read()
        } else {
            self.take_reads(1)
        }
    }

    /// Return credits only after a physical read retires, or if unused.
    pub fn return_reads(&self, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }

        let mut state = self.state();
        if count > self.ticket_capacity - state.reads_free {
            return Err(Error::State("read ticket return exceeds the tickets taken"));
        }

        state.reads_free += count;
        state.returns = state.returns.wrapping_add(1);
        let callbacks = std::mem::take(&mut state.waiters);
        drop(state);

        (self.notify)(callbacks);
        Ok(())
    }

    /// Register for a credit return, or return the callback for immediate
    /// dispatch if credits have already returned since the refused fetch.
    pub fn notify_reads_returned(&self, callback: C, after: u64) -> Option<C> {
        let mut state = self.state();
        if state.returns != after {
            return Some(callback);
        }

        state.waiters.push(callback);
        None
    }
}

/// A fetch's reserved read credits. Each submission takes one credit; close
/// and destruction return only those which have not been handed to a read.
pub struct ReadReservation<C> {
    capacity: Arc<TransferCapacity<C>>,
    unused: AtomicUsize,
}

impl<C> ReadReservation<C> {
    pub fn new(capacity: Arc<TransferCapacity<C>>, count: usize) -> Result<Self> {
        capacity.take_reads(count)?;
        Ok(Self {
            capacity,
            unused: AtomicUsize::new(count),
        })
    }

    pub fn capacity(&self) -> &Arc<TransferCapacity<C>> {
        &self.capacity
    }

    pub fn use_read(&self) -> Result<()> {
        self.unused
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |unused| {
                unused.checked_sub(1)
            })
            .map(|_| ())
            .map_err(|_| Error::State("read reservation has no ticket left"))
    }

    pub fn close(&self) -> Result<()> {
        // Only the counter crosses threads; read storage has its own fences.
        let unused = self.unused.swap(0, Ordering::Relaxed);
        self.capacity.return_reads(unused)
    }
}

impl<C> Drop for ReadReservation<C> {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::warn!(target: "uniserve_worker.transfer", "failed to return unused read credits: {error}");
        }
    }
}

enum ReadAccess {
    Copy {
        handoff: Option<Arc<Event>>,
    },
    Borrowed {
        streams: HashSet<(i32, usize)>,
        events: Vec<Arc<Event>>,
    },
}

/// A read's result and physical lifetime. The transport serializes mutations
/// and dispatches returned callbacks after releasing its ownership lock.
///
/// A consumable view can be followed by a device failure, so result readiness
/// is separate from the one-shot physical retirement completion.
pub struct TransferTicket<V, E, C> {
    value: Option<Arc<V>>,
    error: Option<Arc<E>>,
    cancelled: bool,
    access: ReadAccess,
    producer: Option<Arc<Event>>,
    copy_stream: Option<Arc<Stream>>,
    source_ready: Option<Arc<Event>>,
    closed: bool,
    undrained: bool,
    callbacks: Vec<C>,
    pub retirement: Arc<Completion<E, C>>,
}

impl<V, E, C> TransferTicket<V, E, C> {
    pub fn new(borrowed: bool) -> Self {
        Self {
            value: None,
            error: None,
            cancelled: false,
            access: if borrowed {
                ReadAccess::Borrowed {
                    streams: HashSet::new(),
                    events: Vec::new(),
                }
            } else {
                ReadAccess::Copy { handoff: None }
            },
            producer: None,
            copy_stream: None,
            source_ready: None,
            closed: false,
            undrained: false,
            callbacks: Vec::new(),
            retirement: Arc::new(Completion::default()),
        }
    }

    pub fn ready(&self) -> bool {
        self.value.is_some() || self.error.is_some()
    }

    pub fn retired(&self) -> bool {
        self.retirement.succeeded()
    }

    pub fn undrained(&self) -> bool {
        self.undrained
    }

    pub fn closed(&self) -> bool {
        self.closed
    }

    pub fn retirement_ready(&self) -> Result<bool> {
        if self.undrained {
            return Err(Error::Resource("transfer physical completion is unknown"));
        }

        Ok(self.retired())
    }

    pub fn add_done_callback(&mut self, callback: C) -> Option<C> {
        if self.ready() {
            return Some(callback);
        }

        self.callbacks.push(callback);
        None
    }

    /// Cancellation revokes consumption, but does not end a submitted access.
    pub fn cancel(&mut self, error: E) -> Vec<C> {
        self.cancelled = true;
        if self.error.is_none() {
            self.error = Some(Arc::new(error));
        }

        std::mem::take(&mut self.callbacks)
    }

    pub fn cancellation_error(&self) -> Option<Arc<E>> {
        if self.cancelled {
            self.error.clone()
        } else {
            None
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    /// Expose a completed view without replacing an earlier error or result.
    pub fn complete(&mut self, value: V, producer: Option<Arc<Event>>) -> Vec<C> {
        self.producer = producer;
        if self.ready() {
            return Vec::new();
        }

        self.value = Some(Arc::new(value));
        std::mem::take(&mut self.callbacks)
    }

    /// Return whether consumers could already have observed a successful view.
    /// Such a failure also invalidates the transport pool, not just this read.
    pub fn fail(&mut self, error: E) -> (bool, Vec<C>) {
        self.fail_shared(Arc::new(error))
    }

    pub fn fail_shared(&mut self, error: Arc<E>) -> (bool, Vec<C>) {
        let late = self.value.is_some();
        self.error = Some(error);
        (late, std::mem::take(&mut self.callbacks))
    }

    pub fn result(&self) -> Result<Outcome<E, Arc<V>>> {
        if let Some(error) = &self.error {
            return Ok(Outcome::Failed(Arc::clone(error)));
        }

        let value = self.value.as_ref().ok_or(Error::State(
            "transfer ticket was observed before readiness",
        ))?;
        if self.closed {
            return Err(Error::State("transfer consumption has already closed"));
        }

        Ok(Outcome::Success(Arc::clone(value)))
    }

    /// End borrowed consumption once. Copy destinations retain their ordinary
    /// value ownership and remain readable after close.
    pub fn close(&mut self) -> bool {
        if !matches!(self.access, ReadAccess::Borrowed { .. }) || self.closed {
            return false;
        }

        self.closed = true;
        true
    }

    /// Capture writes made before submission. Later work on the submitting
    /// stream must not become a dependency of the transport copy.
    pub fn prepare_copy(&mut self, pool: &EventPool, stream: &Stream) -> Result<()> {
        let ReadAccess::Copy { handoff } = &mut self.access else {
            return Err(Error::State("a borrowed transfer cannot copy into storage"));
        };
        *handoff = Some(pool.fence(stream, false)?);
        Ok(())
    }

    /// Order a transport stream after the destination's submitting stream.
    pub fn wait_for_copy(&self, stream: &Stream) -> Result<()> {
        let ReadAccess::Copy { handoff } = &self.access else {
            return Err(Error::State("a borrowed transfer cannot copy into storage"));
        };
        if let Some(event) = handoff {
            stream.wait_event(event)?;
        }
        Ok(())
    }

    /// Order a consuming stream after the producer and, for borrowed reads,
    /// remember it so close can fence its access to the source.
    pub fn consume(&mut self, stream: &Stream) -> Result<()> {
        if let Some(producer) = &self.producer {
            stream.wait_event(producer)?;
            if let ReadAccess::Borrowed { streams, .. } = &mut self.access {
                streams.insert((stream.device().0, stream.raw()));
            }
        }
        Ok(())
    }

    pub fn producer(&self) -> Option<&Arc<Event>> {
        self.producer.as_ref()
    }

    /// Fence each consumer at the end of borrowed access. The caller transfers
    /// these retained event references to deferred source release.
    pub fn record_consumers(&mut self, pool: &EventPool) -> Result<()> {
        if let ReadAccess::Borrowed { streams, events } = &mut self.access {
            for &(device, stream) in streams.iter() {
                let _device = DeviceGuard::new(DeviceId(device))?;
                // SAFETY: consumers keep their streams alive until they close
                // the borrowed read, which records this fence.
                let stream = unsafe { Stream::borrowed(DeviceId(device), stream) }?;
                events.push(pool.fence(&stream, false)?);
            }
            streams.clear();
        }
        Ok(())
    }

    pub fn consumer_events(&self) -> &[Arc<Event>] {
        match &self.access {
            ReadAccess::Borrowed { events, .. } => events,
            ReadAccess::Copy { .. } => &[],
        }
    }

    pub fn release_consumers(&mut self) {
        if let ReadAccess::Borrowed { events, .. } = &mut self.access {
            events.clear();
        }
    }

    /// Move this ticket's retained readiness fences to deferred release.
    pub fn take_events(&mut self) -> Vec<Arc<Event>> {
        let mut events: Vec<_> = self.producer.take().into_iter().collect();
        if let ReadAccess::Copy { handoff } = &mut self.access {
            events.extend(handoff.take());
        }
        events
    }

    /// The backend could not drain device access. It must retain its backing
    /// allocations; neither cancellation nor result failure permits reuse.
    pub fn mark_undrained(
        &mut self,
        stream: Option<Arc<Stream>>,
        source_ready: Option<Arc<Event>>,
    ) {
        self.undrained = true;
        self.copy_stream = stream;
        self.source_ready = source_ready;
    }

    pub fn retire(&self) -> Result<Vec<C>> {
        if self.undrained {
            return Err(Error::Invariant(
                "transfer cannot retire with unknown physical completion".into(),
            ));
        }

        self.retirement.complete(Ok(()))
    }

    pub fn value(&self) -> Option<&V> {
        self.value.as_deref()
    }

    pub fn error(&self) -> Option<&E> {
        self.error.as_deref()
    }
}
