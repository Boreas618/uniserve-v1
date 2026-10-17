//! Transfer reads: consumable results, physical retirement and copy bodies.
//!
//! A [`Read`] is one read of one export: a destination copy, or a borrowed
//! view of a process-local export. Its consumable result and its physical
//! retirement are separate completions: a result can be observed while the
//! copy's stream still runs, and a failed drain retains every storage the
//! copy addressed until process exit.
//!
//! Copies run on transfer-pool threads, on one ordinary (non-partitioned)
//! stream per thread and device: on an SM-partition stream the driver takes
//! copies of mapped peer windows only one call each (`uniserve_cuda::copy`).
//! A thread submits a device copy and its fences and moves on: no host
//! wait orders a copy. The read's [`PhysicalAccess`] is retired once its
//! last fence completed, by the transfer pool's retirement thread.

use super::{ACKNOWLEDGED, CLAIMED};
use std::any::Any;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::thread::ThreadId;

use uniserve_cuda::{DeviceGuard, DeviceId, Event, SourceOrder, Stream, memset_d32};
use uniserve_ipc::Locator;

use super::export_table::SourceGrant;
use super::pool::{TransferPool, TransferRead};
use super::staging::Staging;
use super::{
    Notify, ReadReservation, Retirement, Spans, TransferCapacity, TransferTicket, copy_on_host,
    copy_planner, notify, submit_copy,
};
use crate::ffi::{Backing, DType, MemoryKind};
use crate::host::HostTask;
use crate::storage::memory::StorageView;
use crate::stream::Issue;
use crate::{Error, EventPool, Outcome, Result, lock};

pub(crate) type BorrowedReads = Mutex<Option<HashMap<usize, Weak<Read>>>>;

struct ReadState {
    ticket: TransferTicket,
    /// A process-local reader's admission to the producer's export.
    source: Option<SourceGrant>,
    /// A borrowed read's credit, returned when its consumers finish.
    credit: Option<Arc<TransferCapacity>>,
    /// Storage an undrained copy may still address.
    retained: Vec<Box<dyn Any + Send>>,
    /// The submitted device work of a copy, until the transfer pool takes it
    /// to retire it.
    access: Option<PhysicalAccess>,
    /// Imported storage of another process the read reached (a mapping, the
    /// producer's fence), kept while the read lives: reads of one storage
    /// alive at once share one import.
    imports: Vec<Arc<dyn Any + Send + Sync>>,
    work: Option<Weak<HostTask<TransferRead>>>,
    /// The read's task was handed to its transport thread.
    started: bool,
}

/// One read's consumable result and physical retirement. Cancellation
/// revokes consumption; it cannot prove that a submitted copy has stopped.
pub struct Read {
    events: Arc<EventPool>,
    ready: Condvar,
    state: Mutex<ReadState>,
    borrower: Option<Weak<BorrowedReads>>,
}

impl Read {
    pub(crate) fn new(events: Arc<EventPool>, borrower: Option<Weak<BorrowedReads>>) -> Arc<Self> {
        Arc::new(Self {
            events,
            ready: Condvar::new(),
            state: Mutex::new(ReadState {
                ticket: TransferTicket::new(borrower.is_some()),
                source: None,
                credit: None,
                retained: Vec::new(),
                access: None,
                imports: Vec::new(),
                work: None,
                started: false,
            }),
            borrower,
        })
    }

    fn lock(&self) -> MutexGuard<'_, ReadState> {
        lock(&self.state)
    }

    /// Whether a result or an error is consumable.
    pub(crate) fn ready(&self) -> bool {
        self.lock().ticket.ready()
    }

    /// Whether the read retired; an error when its completion is unknown.
    pub(crate) fn retirement_ready(&self) -> Result<bool> {
        self.lock().ticket.retirement_ready()
    }

    /// The physical retirement signal storage owners wait on.
    pub fn retirement(&self) -> Arc<Retirement> {
        Arc::clone(&self.lock().ticket.retirement)
    }

    /// Run `callback` once a result or error is consumable.
    pub fn add_done_callback(&self, callback: Notify) {
        let immediate = self.lock().ticket.add_done_callback(callback);
        if let Some(callback) = immediate {
            callback();
        }
    }

    /// Run `callback` once physical access ended.
    pub(crate) fn add_retirement_callback(&self, callback: Notify) {
        let immediate = self.retirement().subscribe(callback);
        if let Some(callback) = immediate {
            callback();
        }
    }

    /// Revoke consumption and withdraw queued work. A running copy keeps its
    /// credits and storage until it drains.
    pub fn cancel(&self) -> Result<()> {
        let (callbacks, work) = {
            let mut state = self.lock();
            let callbacks = state
                .ticket
                .cancel(Error::Resource("transfer read was cancelled"));
            (callbacks, state.work.as_ref().and_then(Weak::upgrade))
        };
        self.ready.notify_all();
        notify(callbacks);
        if let Some(work) = work {
            work.cancel(false)?;
        }
        Ok(())
    }

    /// Fail unless the read is still wanted (checked between copy steps).
    pub(crate) fn require_active(&self) -> Result<()> {
        match self.lock().ticket.cancellation_error() {
            Some(error) => Err(Error::Shared(error)),
            None => Ok(()),
        }
    }

    /// The read's value without a host wait. A device result orders
    /// `stream` after the producing copy; a borrowed result also remembers
    /// `stream`, so [`close`](Self::close) can fence its access.
    pub fn result(&self, stream: impl Issue) -> Result<Spans> {
        let mut state = self.lock();
        let value = match state.ticket.result()? {
            Outcome::Success(value) => Spans::clone(&value),
            Outcome::Failed(error) => return Err(Error::Shared(error)),
            Outcome::Cancelled => return Err(Error::Resource("transfer read was cancelled")),
        };
        if state.ticket.producer().is_some() {
            let MemoryKind::Cuda(device) = value.device() else {
                return Err(Error::Invariant(
                    "a fenced transfer result is not device memory".into(),
                ));
            };
            let consumer = stream.stream().ok_or_else(|| {
                Error::Invalid("a device transfer result requires a consuming stream".into())
            })?;
            if consumer.device() != device {
                return Err(Error::Invalid(
                    "transfer consumer stream is on another device".into(),
                ));
            }
            state.ticket.consume(stream)?;
        }
        Ok(value)
    }

    /// Block a host thread until a value or error is consumable.
    pub fn wait_ready(&self) {
        let state = self.lock();
        let _ready = self
            .ready
            .wait_while(state, |state| !state.ticket.ready())
            .unwrap_or_else(PoisonError::into_inner);
    }

    /// End borrowed consumption once: fence every consuming stream and
    /// release the source after those fences complete. Copy destinations
    /// remain owned by their caller and stay readable after close.
    pub fn close(self: &Arc<Self>) -> Result<()> {
        let events = {
            let mut state = self.lock();
            if !state.ticket.close() {
                return Ok(());
            }
            state.ticket.record_consumers(&self.events)?;
            state.ticket.consumer_events().to_vec()
        };
        if events.is_empty() {
            return self.finish_borrow();
        }

        let read = Arc::clone(self);
        self.events
            .defer(events, Box::new(move || read.finish_borrow()))
    }

    /// Close a borrowed read and have the event pool wait on the host for
    /// its consumers, with the pool's deadline (shutdown only: nobody is
    /// left to take a completion).
    pub(crate) fn drain(self: &Arc<Self>) -> Result<()> {
        self.close()?;
        self.events.drain()
    }

    pub(crate) fn bind_source(&self, source: SourceGrant) -> Result<()> {
        let mut state = self.lock();
        if state.source.is_some() {
            drop(state);
            source.release()?;
            return Err(Error::Invariant(
                "transfer read already owns a source buffer".into(),
            ));
        }
        state.source = Some(source);
        Ok(())
    }

    fn set_work(&self, work: &Arc<HostTask<TransferRead>>) {
        self.lock().work = Some(Arc::downgrade(work));
    }

    fn set_credit(&self, capacity: Arc<TransferCapacity>) {
        self.lock().credit = Some(capacity);
    }

    pub(crate) fn prepare_copy(&self, stream: impl Issue) -> Result<()> {
        self.lock().ticket.prepare_copy(&self.events, stream)
    }

    /// Hand the read to its transport thread, whose stream waits on the
    /// destination's handoff fence.
    ///
    /// The caller starts a read once the issuer it was submitted with
    /// (`submit`) performed its work, so the fence's record is made: at
    /// once after a stream, from the job's completion on after a recorder.
    /// A read that was cancelled or already started stays as it is. When
    /// the task cannot be queued it is withdrawn.
    pub fn start(&self) -> Result<()> {
        let work = {
            let mut state = self.lock();
            if std::mem::replace(&mut state.started, true) {
                return Ok(());
            }
            state.work.as_ref().and_then(Weak::upgrade)
        };
        let Some(task) = work else {
            return Ok(());
        };
        if task.completion.done() {
            return Ok(());
        }

        let queued = task.submit();
        if queued.is_err()
            && let Err(cleanup) = task.cancel(true)
        {
            tracing::warn!(target: "uniserve_worker.transfer", "failed to withdraw a transfer read: {cleanup}");
        }
        queued
    }

    /// Expose a completed value. `event` fences a device copy; the read
    /// retains one reference to it until the read is dropped.
    pub(crate) fn complete(&self, value: Spans, event: Option<Arc<Event>>) -> Result<()> {
        if let Some(event) = &event {
            self.events.retain(event, self.events.device(event)?, 1)?;
        }
        let callbacks = self.lock().ticket.complete(value, event);
        self.ready.notify_all();
        notify(callbacks);
        Ok(())
    }

    fn fail(&self, error: Arc<Error>) {
        let (_, callbacks) = self.lock().ticket.fail_shared(error);
        self.ready.notify_all();
        notify(callbacks);
    }

    pub(crate) fn mark_undrained(
        &self,
        stream: Arc<Stream>,
        producer: Option<Arc<Event>>,
        retained: Vec<Box<dyn Any + Send>>,
    ) {
        let mut state = self.lock();
        state.ticket.mark_undrained(Some(stream), producer);
        state.retained.extend(retained);
    }

    /// Keep `import` while the read lives.
    pub(crate) fn retain_import(&self, import: Arc<dyn Any + Send + Sync>) {
        self.lock().imports.push(import);
    }

    /// The device work a copy submitted, which its retirement waits for.
    pub(crate) fn take_access(&self) -> Option<PhysicalAccess> {
        self.lock().access.take()
    }

    pub(crate) fn release_source(&self) -> Result<()> {
        let (source, credit) = {
            let mut state = self.lock();
            (state.source.take(), state.credit.take())
        };
        let released = source.as_ref().map_or(Ok(()), SourceGrant::release);
        if released.is_err() {
            // A failed producer reclamation retains its registry and backing.
            self.lock().source = source;
        }
        let returned = credit.map_or(Ok(()), |capacity| capacity.return_reads(1));
        released.and(returned)
    }

    fn finish_borrow(&self) -> Result<()> {
        self.lock().ticket.release_consumers();
        if let Some(borrower) = self.borrower.as_ref().and_then(Weak::upgrade)
            && let Some(reads) = lock(&borrower).as_mut()
        {
            reads.remove(&(std::ptr::from_ref(self) as usize));
        }

        let result = self.release_source();
        let callbacks = {
            let state = self.lock();
            match &result {
                Ok(()) => state.ticket.retire(),
                Err(error) => state
                    .ticket
                    .retirement
                    .complete(Err(Error::transport(format!(
                        "borrowed transfer source release failed: {error}"
                    )))),
            }
        }?;
        notify(callbacks);
        result
    }

    pub(crate) fn check_active(&self) -> Result<()> {
        if self.lock().ticket.is_cancelled() {
            return Err(Error::Resource("transfer read was cancelled"));
        }
        Ok(())
    }
}

impl Drop for Read {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        let events = state.ticket.take_events();
        if events.is_empty() {
            return;
        }
        // The fences' last device work may still address the value: keep it
        // until they complete.
        let value = state.ticket.value().cloned();
        let deferred = self.events.defer(
            events,
            Box::new(move || {
                drop(value);
                Ok(())
            }),
        );
        if let Err(error) = deferred {
            tracing::warn!(target: "uniserve_worker.transfer", "failed to retire transfer read fences: {error}");
        }
    }
}

/// A storage owner's reference to one read: the read itself (to cancel or
/// close it) and its physical retirement signal (to decide reuse).
pub(crate) struct ReadRef {
    read: Arc<Read>,
    retirement: Arc<Retirement>,
}

impl ReadRef {
    pub(crate) fn new(read: &Arc<Read>) -> Self {
        Self {
            read: Arc::clone(read),
            retirement: read.retirement(),
        }
    }

    pub(crate) fn read(&self) -> &Arc<Read> {
        &self.read
    }
}

impl std::ops::Deref for ReadRef {
    type Target = Retirement;

    fn deref(&self) -> &Retirement {
        &self.retirement
    }
}

/// Per-thread copy streams and the page-locked staging of one transport.
pub(crate) struct ReadContext {
    pub(crate) events: Arc<EventPool>,
    streams: Mutex<HashMap<(ThreadId, DeviceId), Arc<Stream>>>,
    /// Staging of host reads into device destinations and of channel
    /// exports; a read's lease returns when the read retires.
    pub(crate) staging: Arc<Staging>,
}

impl ReadContext {
    /// `staging_bytes` bounds the free staging the transport keeps.
    pub(crate) fn new(events: Arc<EventPool>, staging_bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            events,
            streams: Mutex::new(HashMap::new()),
            staging: Staging::new(staging_bytes),
        })
    }

    /// The calling thread's copy stream on `device`.
    fn stream(&self, device: DeviceId) -> Result<Arc<Stream>> {
        let key = (std::thread::current().id(), device);
        let mut streams = lock(&self.streams);
        if let Some(stream) = streams.get(&key) {
            return Ok(Arc::clone(stream));
        }
        let stream = Arc::new(Stream::new(device)?);
        streams.insert(key, Arc::clone(&stream));
        Ok(stream)
    }

    /// Release streams after every reader thread joined and every read
    /// retired.
    pub(crate) fn clear(&self) {
        lock(&self.streams).clear();
    }
}

/// What one read does on its transfer thread.
pub(crate) enum ReadCall {
    /// Copy a process-local export into the destination.
    Copy {
        source: Spans,
        destination: Spans,
        producer: Option<Arc<Event>>,
    },
    /// Copy a shared-memory export of this host into the destination.
    Shared {
        locator: Arc<Locator>,
        slot: usize,
        destination: Spans,
        region: Option<Vec<Range<u64>>>,
    },
    /// Copy a rank-channel payload into the destination.
    Channel {
        locator: Arc<Locator>,
        destination: Spans,
        region: Option<Vec<Range<u64>>>,
    },
    /// Map a CUDA VMM export and copy it into the destination.
    Cuda(super::vmm::CudaRead),
}

/// The native read backend of the transfer pool.
pub(crate) struct NativeRead {
    pub(crate) read: Arc<Read>,
    pub(crate) call: ReadCall,
    pub(crate) context: Arc<ReadContext>,
}

impl NativeRead {
    /// Run `action` on the read's ticket under its lock, waking result
    /// waiters when the ticket became consumable. `action` invokes no
    /// observer and no numerical code.
    pub(crate) fn with_ticket<T>(&self, action: impl FnOnce(&mut TransferTicket) -> T) -> T {
        let mut state = self.read.lock();
        let result = action(&mut state.ticket);
        let ready = state.ticket.ready();
        drop(state);
        if ready {
            self.read.ready.notify_all();
        }
        result
    }

    /// Perform the read. It returns after physical access ended; when that
    /// cannot be established it marks the ticket undrained and fails.
    pub(crate) fn read(&self) -> Result<()> {
        match &self.call {
            ReadCall::Copy {
                source,
                destination,
                producer,
            } => self.copy(source, destination, producer.clone(), None, Vec::new()),
            ReadCall::Shared {
                locator,
                slot,
                destination,
                region,
            } => self.shared(locator, *slot, destination, region.as_deref()),
            ReadCall::Channel {
                locator,
                destination,
                region,
            } => self.channel(locator, destination, region.as_deref()),
            ReadCall::Cuda(read) => read.run(self),
        }
    }

    /// Release local source ownership after the read's physical accesses
    /// ended. This also runs for a queued read withdrawn before it ran.
    pub(crate) fn release(&self) -> Result<()> {
        self.read.release_source()
    }
}

/// The device whose stream copies between `source` and `destination`.
fn copy_device(source: &Spans, destination: &Spans) -> Option<DeviceId> {
    match (destination.device(), source.device()) {
        (MemoryKind::Cuda(device), _) | (_, MemoryKind::Cuda(device)) => Some(device),
        _ => None,
    }
}

/// How the device a CUDA VMM read's source is mapped on takes part in its
/// copy: a stream of that device waits for the producer's interprocess
/// fence (no stream of another device may wait on it), and this reader's
/// acknowledgment word is claimed before the copy and acknowledged after it.
pub(crate) struct ImportOrder {
    pub(crate) device: DeviceId,
    pub(crate) producer: Option<Arc<Event>>,
    pub(crate) acknowledgment: Option<u64>,
}

/// A copy's submitted device work: every device access of the read
/// precedes `fence`, the last of `fences`. Until that fence completed the
/// read keeps `operands` (every storage the work addresses), one pool
/// reference of each fence, and the streams the work runs on.
pub(crate) struct PhysicalAccess {
    fence: Arc<Event>,
    fences: Vec<Arc<Event>>,
    streams: Vec<Arc<Stream>>,
    operands: Vec<Box<dyn Any + Send>>,
}

impl PhysicalAccess {
    /// Wait on the host for the work to complete, then release what it
    /// kept. When completion cannot be established the read keeps all of
    /// it, marked undrained, and this fails.
    pub(crate) fn retire(self, read: &NativeRead) -> Result<()> {
        let Self {
            fence,
            fences,
            mut streams,
            mut operands,
        } = self;
        if let Err(error) = fence.synchronize() {
            let stream = streams.remove(0);
            operands.push(Box::new(streams));
            operands.push(Box::new(fences));
            read.read.mark_undrained(stream, None, operands);
            return Err(error.into());
        }
        drop(operands);
        let events = &read.context.events;
        fences.iter().try_for_each(|fence| events.release(fence, 1))
    }

    /// Give the work up unobserved: the read keeps all of it, marked
    /// undrained.
    pub(crate) fn abandon(self, read: &NativeRead) {
        let Self {
            fence,
            fences,
            mut streams,
            mut operands,
        } = self;
        let stream = streams.remove(0);
        operands.push(Box::new((fence, fences, streams)));
        read.read.mark_undrained(stream, None, operands);
    }
}

impl NativeRead {
    /// Copy `source` into `destination` and complete the read with it.
    ///
    /// A device copy runs on this thread's stream after the destination's
    /// handoff fence and the producer's fence; `import` adds the ordering of
    /// a mapped VMM source. The read's result is fenced by an event of the
    /// copy stream, so consumers order their streams after it without a
    /// host wait. No host wait orders the copy either: the submitted work
    /// and every operand it addresses (with `retained`) become the read's
    /// [`PhysicalAccess`], which the transfer pool retires once it
    /// completed. A copy that fails after submitting work drains it here;
    /// a failed drain retains every operand.
    pub(crate) fn copy(
        &self,
        source: &Spans,
        destination: &Spans,
        producer: Option<Arc<Event>>,
        import: Option<ImportOrder>,
        retained: Vec<Box<dyn Any + Send>>,
    ) -> Result<()> {
        self.read.require_active()?;
        let Some(device) = copy_device(source, destination) else {
            let mut plan = copy_planner(None, SourceOrder::Stream)?;
            destination.plan_copy(source, &mut plan)?;
            // SAFETY: both operands are host memory retained by this read;
            // the destination's owner does not access it until completion.
            unsafe { copy_on_host(&plan) };
            return self.read.complete(destination.clone(), None);
        };

        let events = &self.context.events;
        let stream = self.context.stream(device)?;
        // The source's device takes part through its own stream of this
        // thread when it is not the copy's device.
        let peer = match &import {
            Some(order) if order.device != device => Some(self.context.stream(order.device)?),
            _ => None,
        };
        let (word, mapped) = import.as_ref().map_or((None, None), |order| {
            (order.acknowledgment, order.producer.as_ref())
        });
        // Pool fences of this copy, each with one retained reference, in
        // submission order: the last one follows all of the copy's work.
        let mut fences: Vec<Arc<Event>> = Vec::new();

        let submitted = (|| -> Result<()> {
            // The claim precedes either handoff wait in stream order, so the
            // producer cannot reuse the chunk while this copy waits to
            // begin; no host wait is needed for it to land first.
            let claimed = match &peer {
                Some(peer) => {
                    let _device = DeviceGuard::new(peer.device())?;
                    if let Some(word) = word {
                        // SAFETY: the word lies in the mapped chunk header
                        // the read's source spans retain.
                        unsafe { memset_d32(word, CLAIMED, 1, peer) }?;
                    }
                    if let Some(producer) = mapped {
                        peer.wait_event(producer)?;
                    }
                    let claimed = events.fence(peer.as_ref(), false)?;
                    fences.push(Arc::clone(&claimed));
                    Some(claimed)
                }
                None => None,
            };

            let _device = DeviceGuard::new(device)?;
            if peer.is_none()
                && let Some(word) = word
            {
                // SAFETY: as for the claim on the source's device.
                unsafe { memset_d32(word, CLAIMED, 1, &stream) }?;
            }
            self.read.lock().ticket.wait_for_copy(&stream)?;
            let mapped = mapped.filter(|_| peer.is_none()).map(Arc::as_ref);
            for fence in [producer.as_deref(), mapped, claimed.as_deref()]
                .into_iter()
                .flatten()
            {
                stream.wait_event(fence)?;
            }

            // SAFETY: the read retains both operands until its physical
            // access retires, or forever if that cannot be established.
            unsafe { submit_copy(destination, source, &stream) }?;
            if peer.is_none()
                && let Some(word) = word
            {
                // SAFETY: as for the claim.
                unsafe { memset_d32(word, ACKNOWLEDGED, 1, &stream) }?;
            }
            let completed = events.fence(stream.as_ref(), false)?;
            fences.push(Arc::clone(&completed));

            if let (Some(peer), Some(word)) = (&peer, word) {
                // Acknowledge on the source's device once the copy is done.
                let _device = DeviceGuard::new(peer.device())?;
                peer.wait_event(&completed)?;
                // SAFETY: as for the claim.
                unsafe { memset_d32(word, ACKNOWLEDGED, 1, peer) }?;
                fences.push(events.fence(peer.as_ref(), false)?);
            }
            self.read.complete(destination.clone(), Some(completed))
        })();

        let mut operands: Vec<Box<dyn Any + Send>> =
            vec![Box::new(source.clone()), Box::new(destination.clone())];
        operands.extend(retained);
        operands.extend(import.map(|order| Box::new(order) as Box<dyn Any + Send>));
        let streams: Vec<Arc<Stream>> = std::iter::once(stream).chain(peer).collect();

        let last = fences.last().cloned();
        match (submitted, last) {
            (Ok(()), Some(fence)) => {
                self.read.lock().access = Some(PhysicalAccess {
                    fence,
                    fences,
                    streams,
                    operands,
                });
                Ok(())
            }
            (submitted, _) => {
                let error = submitted
                    .err()
                    .unwrap_or_else(|| Error::Invariant("a submitted copy has no fence".into()));
                // The read's waiters and this call's caller observe one
                // failure.
                let shared = Arc::new(error);
                self.read.fail(Arc::clone(&shared));
                if let Err(drained) = self.drain(streams, producer, fences, operands) {
                    return Err(Error::transport(format!(
                        "{drained}; copy failed: {shared}"
                    )));
                }
                Err(Error::Shared(shared))
            }
        }
    }

    /// Wait on the host for the streams of a failed copy, which may have
    /// submitted part of its work, and release its fences. Unknown physical
    /// completion is not cancellation: the read then keeps the streams,
    /// fences and every operand.
    fn drain(
        &self,
        mut streams: Vec<Arc<Stream>>,
        producer: Option<Arc<Event>>,
        fences: Vec<Arc<Event>>,
        mut operands: Vec<Box<dyn Any + Send>>,
    ) -> Result<()> {
        let drained = streams.iter().try_for_each(|stream| -> Result<()> {
            let _device = DeviceGuard::new(stream.device())?;
            stream.synchronize()?;
            Ok(())
        });
        if let Err(error) = drained {
            let stream = streams.remove(0);
            operands.push(Box::new(streams));
            operands.push(Box::new(fences));
            self.read.mark_undrained(stream, producer, operands);
            return Err(error);
        }
        let events = &self.context.events;
        fences.iter().try_for_each(|fence| events.release(fence, 1))
    }

    /// Stage host bytes for a device destination: host-copy `source` into a
    /// page-locked staging lease, then copy the lease to `destination`. The
    /// lease returns to the transport's staging when the read retires.
    pub(super) fn copy_from_host(
        &self,
        source: &Spans,
        destination: &Spans,
        release: impl FnOnce(),
    ) -> Result<()> {
        let MemoryKind::Cuda(device) = destination.device() else {
            let mut plan = copy_planner(None, SourceOrder::Stream)?;
            destination.plan_copy(source, &mut plan)?;
            // SAFETY: host operands retained by this read and its locator.
            unsafe { copy_on_host(&plan) };
            release();
            return self.read.complete(destination.clone(), None);
        };

        let shape: Vec<i64> = source.shape().iter().map(|&extent| extent as i64).collect();
        let staging = self
            .context
            .staging
            .lease(device, source.nbytes() as usize)?;
        let staged = Spans::single(StorageView::contiguous(
            Arc::clone(&staging) as Arc<dyn Backing>,
            0,
            source.dtype(),
            &shape,
        )?)?;
        let mut plan = copy_planner(None, SourceOrder::Stream)?;
        staged.plan_copy(source, &mut plan)?;
        // SAFETY: no device copy uses a leased block: a block returns to the
        // staging only after the read that copied from it retired.
        unsafe { copy_on_host(&plan) };
        // The private copy ends the source claim before device DMA begins.
        release();

        self.copy(&staged, destination, None, None, vec![Box::new(staging)])
    }
}

/// A contiguous view of a locator's payload bytes at `offset` in `backing`.
pub(super) fn payload(
    backing: Arc<dyn Backing>,
    offset: usize,
    locator: &Locator,
) -> Result<Spans> {
    let dtype: DType = DType::parse(&locator.dtype)?;
    let shape: Vec<i64> = locator.shape.iter().map(|&extent| extent as i64).collect();
    Spans::single(StorageView::contiguous(backing, offset, dtype, &shape)?)
}

/// Reserve a read's credits and task and issue its destination handoff
/// fence to `stream`. The read is handed to its transport thread by
/// [`Read::start`], once that fence's record is made. On failure the task
/// is withdrawn.
pub(crate) fn submit(
    pool: &Arc<TransferPool>,
    backend: NativeRead,
    nbytes: u64,
    stream: impl Issue,
    reservation: Option<&ReadReservation>,
) -> Result<Arc<Read>> {
    if let Some(error) = pool.error() {
        return Err(Error::Shared(error));
    }
    let read = Arc::clone(&backend.read);
    let destination_device = match &backend.call {
        ReadCall::Copy { destination, .. }
        | ReadCall::Shared { destination, .. }
        | ReadCall::Channel { destination, .. } => destination.device(),
        ReadCall::Cuda(cuda) => cuda.destination().device(),
    };
    let task = pool.reserve(backend, nbytes, reservation)?;

    let submitted = (|| -> Result<()> {
        read.set_work(&task);
        if let MemoryKind::Cuda(device) = destination_device {
            let submitting = stream.stream().ok_or_else(|| {
                Error::Invalid("a device read destination requires its submitting stream".into())
            })?;
            if submitting.device() != device {
                return Err(Error::Invalid(
                    "a read's submitting stream is on another device".into(),
                ));
            }
            read.prepare_copy(stream)?;
        }
        if let Some(wake) = pool.completion_wake() {
            read.add_done_callback(Arc::clone(&wake));
            read.add_retirement_callback(wake);
        }
        Ok(())
    })();
    if let Err(error) = submitted {
        if let Err(cleanup) = task.cancel(true) {
            tracing::warn!(target: "uniserve_worker.transfer", "failed to withdraw a transfer read: {cleanup}");
        }
        return Err(error);
    }
    Ok(read)
}

/// Complete a borrowed read of a process-local export: the reader consumes
/// the producer's spans in place after its fence, holding a read credit and
/// the source grant until it closes.
pub(crate) fn borrow(
    read: &Arc<Read>,
    capacity: &Arc<TransferCapacity>,
    spans: Spans,
    producer: Option<Arc<Event>>,
    reservation: Option<&ReadReservation>,
    wake: Option<Notify>,
) -> Result<()> {
    capacity.take_read(reservation)?;
    read.set_credit(Arc::clone(capacity));
    if let Some(wake) = wake {
        read.add_done_callback(Arc::clone(&wake));
        read.add_retirement_callback(wake);
    }
    read.complete(spans, producer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{ErrorCode, WorkerError};

    #[test]
    fn a_cancelled_read_reports_a_resource_failure() -> Result<()> {
        let read = Read::new(EventPool::new(), None);
        read.cancel()?;

        // Both the copy's activity check and the consumer's result report
        // the cancellation with its own class.
        for observed in [
            read.require_active(),
            read.result(None::<&Stream>).map(drop),
        ] {
            let Err(error) = observed else {
                return Err(Error::State("a cancelled read stayed usable"));
            };
            let classified = WorkerError::classify(error, None);
            assert_eq!(classified.code, ErrorCode::ResourceError);
            assert_eq!(classified.response().message, "transfer read was cancelled");
        }
        Ok(())
    }
}
