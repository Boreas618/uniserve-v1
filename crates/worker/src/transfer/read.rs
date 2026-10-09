//! Transfer reads: consumable results, physical retirement and copy bodies.
//!
//! A [`Read`] is one read of one export: a destination copy, or a borrowed
//! view of a process-local export. Its consumable result and its physical
//! retirement are separate completions: a result can be observed while the
//! copy's stream still runs, and a failed drain retains every storage the
//! copy addressed until process exit.
//!
//! Copies run on transfer-pool threads, on one ordinary (non-partitioned)
//! stream per thread and device: green-context streams cannot batch-copy
//! mapped peer windows.

use std::any::Any;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::thread::ThreadId;
use std::time::Duration;

use uniserve_cuda::{DeviceGuard, DeviceId, Event, SourceOrder, Stream, memset_d32};
use uniserve_ipc::{Locator, TransferTransport};

use super::endpoint::SourceGrant;
use super::pool::{ReadBackend, TransferPool, TransferRead};
use super::{
    Capacity, Notify, Retirement, SHM_HEADER_BYTES, SharedRead, Spans, TransferTicket,
    copy_on_host, copy_planner, notify, submit_copy,
};
use crate::ffi::{Backing, DType, Device};
use crate::host::HostTask;
use crate::storage::memory::{Memory, StorageView, parse_dtype};
use crate::{Error, EventPool, Outcome, Result};

/// Reader acknowledgment states written into a VMM chunk header.
pub(crate) const CLAIMED: u32 = 1;
pub(crate) const ACKNOWLEDGED: u32 = 2;

/// How long a shared-memory reader waits for its producer's readiness.
const SHARED_READY_TIMEOUT: Duration = Duration::from_secs(120);

pub(crate) type ReadPool = TransferPool<NativeRead>;
pub(crate) type BorrowedReads = Mutex<Option<HashMap<usize, Weak<Read>>>>;

struct ReadState {
    ticket: TransferTicket<Spans, Error, Notify>,
    /// A process-local reader's admission to the producer's export.
    source: Option<SourceGrant>,
    /// A borrowed read's credit, returned when its consumers finish.
    credit: Option<Arc<Capacity>>,
    /// Storage an undrained copy may still address.
    retained: Vec<Box<dyn Any + Send>>,
    work: Option<Weak<HostTask<TransferRead<NativeRead>>>>,
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
                work: None,
            }),
            borrower,
        })
    }

    fn lock(&self) -> MutexGuard<'_, ReadState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Whether a result or an error is consumable.
    pub fn ready(&self) -> bool {
        self.lock().ticket.ready()
    }

    /// Whether physical access ended successfully.
    pub fn retired(&self) -> bool {
        self.lock().ticket.retired()
    }

    /// Whether the read retired; an error when its completion is unknown.
    pub fn retirement_ready(&self) -> Result<bool> {
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
    pub fn add_retirement_callback(&self, callback: Notify) {
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
    pub fn result(&self, stream: Option<&Stream>) -> Result<Spans> {
        let mut state = self.lock();
        let value = match state.ticket.result()? {
            Outcome::Success(value) => Spans::clone(&value),
            Outcome::Failed(error) => return Err(Error::Shared(error)),
            Outcome::Cancelled => return Err(Error::Resource("transfer read was cancelled")),
        };
        if state.ticket.producer().is_some() {
            let Device::Cuda(device) = value.device() else {
                return Err(Error::Invariant(
                    "a fenced transfer result is not device memory".into(),
                ));
            };
            let stream = stream.ok_or_else(|| {
                Error::Invalid("a device transfer result requires a consuming stream".into())
            })?;
            if stream.device().0 != device {
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
        self.events.reap()?;
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
            .defer(events.clone(), Box::new(move || read.finish_borrow()))?;
        for event in &events {
            self.events.wake_after(event)?;
        }
        self.events.reap()
    }

    /// Close a borrowed read and wait on the host for its consumers
    /// (shutdown only).
    pub(crate) fn drain(self: &Arc<Self>) -> Result<()> {
        self.close()?;
        let events = self.lock().ticket.consumer_events().to_vec();
        for event in events {
            event.synchronize()?;
        }
        self.events.reap()
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

    fn set_work(&self, work: &Arc<HostTask<TransferRead<NativeRead>>>) {
        self.lock().work = Some(Arc::downgrade(work));
    }

    fn set_credit(&self, capacity: Arc<Capacity>) {
        self.lock().credit = Some(capacity);
    }

    pub(crate) fn prepare_copy(&self, stream: &Stream) -> Result<()> {
        self.lock().ticket.prepare_copy(&self.events, stream)
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
            && let Some(reads) = borrower
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .as_mut()
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
                    .complete(Err(Error::Transport(format!(
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
        if let Err(error) = deferred.and_then(|()| self.events.reap()) {
            tracing::warn!(target: "uniserve_worker.transfer", "failed to retire transfer read fences: {error}");
        }
    }
}

/// A storage owner's reference to one read: the read itself (to cancel or
/// close it) and its physical retirement signal (to decide reuse).
pub struct ReadRef {
    read: Arc<Read>,
    retirement: Arc<Retirement>,
}

impl ReadRef {
    pub fn new(read: &Arc<Read>) -> Self {
        Self {
            read: Arc::clone(read),
            retirement: read.retirement(),
        }
    }

    pub fn read(&self) -> &Arc<Read> {
        &self.read
    }
}

impl std::ops::Deref for ReadRef {
    type Target = Retirement;

    fn deref(&self) -> &Retirement {
        &self.retirement
    }
}

/// Per-thread copy streams and page-locked staging of one transport's reads.
pub(crate) struct ReadContext {
    pub(crate) events: Arc<EventPool>,
    streams: Mutex<HashMap<(ThreadId, i32), Arc<Stream>>>,
    staging: Mutex<HashMap<(ThreadId, i32), Arc<Memory>>>,
}

impl ReadContext {
    pub(crate) fn new(events: Arc<EventPool>) -> Arc<Self> {
        Arc::new(Self {
            events,
            streams: Mutex::new(HashMap::new()),
            staging: Mutex::new(HashMap::new()),
        })
    }

    /// The calling thread's copy stream on `device`.
    fn stream(&self, device: i32) -> Result<Arc<Stream>> {
        let key = (std::thread::current().id(), device);
        let mut streams = self.streams.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(stream) = streams.get(&key) {
            return Ok(Arc::clone(stream));
        }
        let stream = Arc::new(Stream::new(DeviceId(device))?);
        streams.insert(key, Arc::clone(&stream));
        Ok(stream)
    }

    /// The calling thread's page-locked staging of at least `bytes` for
    /// `device` copies. A read drains its stream before returning, so the
    /// thread's next read may reuse it; an undrained read takes it over.
    fn staging(&self, device: i32, bytes: usize) -> Result<Arc<Memory>> {
        let key = (std::thread::current().id(), device);
        let mut staging = self.staging.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(memory) = staging.get(&key)
            && memory.len() >= bytes
        {
            return Ok(Arc::clone(memory));
        }
        let memory = Memory::pinned(device, bytes.max(1).next_power_of_two())?;
        staging.insert(key, Arc::clone(&memory));
        Ok(memory)
    }

    fn forget_staging(&self, device: i32) {
        let key = (std::thread::current().id(), device);
        self.staging
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&key);
    }

    /// Release streams after every reader thread joined.
    pub(crate) fn clear(&self) {
        self.streams
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        self.staging
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
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
    Cuda(super::cuda_vmm::CudaRead),
}

/// The native read backend of the transfer pool.
pub(crate) struct NativeRead {
    pub(crate) read: Arc<Read>,
    pub(crate) call: ReadCall,
    pub(crate) context: Arc<ReadContext>,
}

impl ReadBackend for NativeRead {
    type Value = Spans;
    type Error = Error;
    type Callback = Notify;

    fn with_ticket<T>(
        &self,
        action: impl FnOnce(&mut TransferTicket<Spans, Error, Notify>) -> T,
    ) -> T {
        let mut state = self.read.lock();
        let result = action(&mut state.ticket);
        let ready = state.ticket.ready();
        drop(state);
        if ready {
            self.read.ready.notify_all();
        }
        result
    }

    fn read(&self) -> Result<()> {
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

    fn release(&self) -> Result<()> {
        self.read.release_source()
    }

    fn notify(callbacks: Vec<Notify>) {
        notify(callbacks);
    }

    fn wake(callback: &Notify) {
        callback();
    }

    fn error(error: Error) -> Error {
        error
    }

    fn report(error: Error) {
        tracing::warn!(target: "uniserve_worker.transfer", "transfer read cleanup failed: {error}");
    }
}

/// The device whose stream copies between `source` and `destination`.
fn copy_device(source: &Spans, destination: &Spans) -> Option<i32> {
    match (destination.device(), source.device()) {
        (Device::Cuda(device), _) | (_, Device::Cuda(device)) => Some(device),
        _ => None,
    }
}

impl NativeRead {
    /// Copy `source` into `destination` and complete the read with it.
    ///
    /// A device copy runs on this thread's stream after the destination's
    /// handoff fence and the producer's fence, claims and acknowledges a VMM
    /// chunk around it when `acknowledgment` names a reader word, exposes its
    /// fence, and drains the stream before returning: the transfer task
    /// retires only after the drain. A failed drain retains every operand.
    pub(crate) fn copy(
        &self,
        source: &Spans,
        destination: &Spans,
        producer: Option<Arc<Event>>,
        acknowledgment: Option<u64>,
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

        let stream = self.context.stream(device)?;
        self.context.events.reap()?;
        let copied = (|| -> Result<()> {
            let _device = DeviceGuard::new(DeviceId(device))?;
            if let Some(word) = acknowledgment {
                // Claim before either handoff wait, so the producer cannot
                // reuse the chunk while this copy is waiting to begin.
                // SAFETY: the word lies in the mapped chunk header retained by
                // this read.
                unsafe { memset_d32(word, CLAIMED, 1, &stream) }?;
                stream.synchronize()?;
            }
            self.read.lock().ticket.wait_for_copy(&stream)?;
            if let Some(producer) = &producer {
                stream.wait_event(producer)?;
            }

            // SAFETY: the read retains both operands until the stream drains
            // below, or forever if it cannot.
            unsafe { submit_copy(destination, source, &stream) }?;
            if let Some(word) = acknowledgment {
                // SAFETY: as for the claim.
                unsafe { memset_d32(word, ACKNOWLEDGED, 1, &stream) }?;
            }
            let completed = self.context.events.acquire(device, false, false)?;
            self.context.events.record(&completed, &stream)?;
            self.read.complete(destination.clone(), Some(completed))
        })();
        // The read's waiters and this call's caller observe one failure.
        let copied = copied.map_err(|error| {
            let shared = Arc::new(error);
            self.read.fail(Arc::clone(&shared));
            Error::Shared(shared)
        });

        let drained = DeviceGuard::new(DeviceId(device))
            .map_err(Error::from)
            .and_then(|_device| stream.synchronize().map_err(Error::from));
        if let Err(error) = drained {
            // Unknown physical completion is not cancellation: keep the
            // stream, fences and every operand with the read.
            let mut kept: Vec<Box<dyn Any + Send>> =
                vec![Box::new(source.clone()), Box::new(destination.clone())];
            kept.extend(retained);
            self.read
                .mark_undrained(Arc::clone(&stream), producer, kept);
            return Err(match copied {
                Ok(()) => error,
                Err(cause) => Error::Transport(format!("{error}; copy failed: {cause}")),
            });
        }
        copied
    }

    /// Stage host bytes for a device destination: host-copy `source` into
    /// this thread's page-locked staging, then copy staging to `destination`.
    fn copy_from_host(
        &self,
        source: &Spans,
        destination: &Spans,
        release: impl FnOnce(),
    ) -> Result<()> {
        let Device::Cuda(device) = destination.device() else {
            let mut plan = copy_planner(None, SourceOrder::Stream)?;
            destination.plan_copy(source, &mut plan)?;
            // SAFETY: host operands retained by this read and its locator.
            unsafe { copy_on_host(&plan) };
            release();
            return self.read.complete(destination.clone(), None);
        };

        let shape: Vec<i64> = source.shape().iter().map(|&extent| extent as i64).collect();
        let staging = self.context.staging(device, source.nbytes() as usize)?;
        let staged = Spans::single(StorageView::contiguous(
            Arc::clone(&staging) as Arc<dyn Backing>,
            0,
            source.dtype(),
            &shape,
        )?)?;
        let mut plan = copy_planner(None, SourceOrder::Stream)?;
        staged.plan_copy(source, &mut plan)?;
        // SAFETY: the staging belongs to this thread and no device copy uses
        // it: the thread's previous read drained its stream.
        unsafe { copy_on_host(&plan) };
        // The private copy ends the source claim before device DMA begins.
        release();

        let copied = self.copy(&staged, destination, None, None, Vec::new());
        if self.read.lock().ticket.undrained() {
            // The undrained copy retains the staging; never reuse it.
            self.context.forget_staging(device);
        }
        copied
    }

    fn shared(
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

    fn channel(
        &self,
        locator: &Arc<Locator>,
        destination: &Spans,
        region: Option<&[Range<u64>]>,
    ) -> Result<()> {
        let TransferTransport::Channel { payload: bytes, .. } = &locator.transport else {
            return Err(Error::Invalid(
                "channel read requires a channel locator".into(),
            ));
        };
        if bytes.len() as u64 != locator.nbytes {
            return Err(Error::Invalid(
                "channel payload disagrees with its locator size".into(),
            ));
        }
        let backing: Arc<dyn Backing> = Arc::new(ChannelPayload(Arc::clone(locator)));
        let source = payload(backing, 0, locator)?;
        let source = match region {
            Some(region) => source.region(region)?,
            None => source,
        };
        self.copy_from_host(&source, destination, || {})
    }
}

/// A contiguous view of a locator's payload bytes at `offset` in `backing`.
fn payload(backing: Arc<dyn Backing>, offset: usize, locator: &Locator) -> Result<Spans> {
    let dtype: DType = parse_dtype(&locator.dtype)?;
    let shape: Vec<i64> = locator.shape.iter().map(|&extent| extent as i64).collect();
    Spans::single(StorageView::contiguous(backing, offset, dtype, &shape)?)
}

/// A rank-channel payload as host memory a copy can read in place. The
/// locator is immutable and retained by the read for the copy's duration.
struct ChannelPayload(Arc<Locator>);

impl ChannelPayload {
    fn bytes(&self) -> &[u8] {
        match &self.0.transport {
            TransferTransport::Channel { payload, .. } => payload,
            _ => &[],
        }
    }
}

impl Backing for ChannelPayload {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn address(&self) -> u64 {
        self.bytes().as_ptr() as u64
    }

    fn len(&self) -> usize {
        self.bytes().len()
    }
}

/// Reserve a read's credits and task, record its destination handoff on
/// `stream` and submit it. On failure the task is withdrawn.
pub(crate) fn submit(
    pool: &Arc<ReadPool>,
    backend: NativeRead,
    nbytes: u64,
    stream: Option<&Stream>,
    reservation: Option<&super::ReadReservation<Notify>>,
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
        if let Device::Cuda(device) = destination_device {
            let stream = stream.ok_or_else(|| {
                Error::Invalid("a device read destination requires its submitting stream".into())
            })?;
            if stream.device().0 != device {
                return Err(Error::Invalid(
                    "a read's submitting stream is on another device".into(),
                ));
            }
            read.prepare_copy(stream)?;
        }
        if let Some(wake) = pool.completion_wake() {
            let wake: Notify = Arc::new(move || wake());
            read.add_done_callback(Arc::clone(&wake));
            read.add_retirement_callback(wake);
        }
        task.submit()
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
    capacity: &Arc<Capacity>,
    spans: Spans,
    producer: Option<Arc<Event>>,
    reservation: Option<&super::ReadReservation<Notify>>,
    wake: Option<Arc<Notify>>,
) -> Result<()> {
    capacity.take_read(reservation)?;
    read.set_credit(Arc::clone(capacity));
    if let Some(wake) = wake {
        let wake: Notify = Arc::new(move || wake());
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
        for observed in [read.require_active(), read.result(None).map(drop)] {
            let Err(error) = observed else {
                return Err(Error::State("a cancelled read stayed usable"));
            };
            let classified = WorkerError::classify(error, None);
            assert_eq!(classified.code, ErrorCode::ResourceError);
            assert_eq!(classified.message(), "transfer read was cancelled");
        }
        Ok(())
    }
}
