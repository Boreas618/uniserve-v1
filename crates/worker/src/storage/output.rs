//! Bounded host output storage and per-batch readback lifetimes.
//!
//! A lane's batch results are copied device-to-host into one page-locked
//! word buffer per batch: integer token columns from the front, raw byte
//! captures from the back. The copies are ordered on the producing streams
//! and fenced by events; the host reads the words only after those fences
//! complete, widening narrower token columns to 64 bits on the host.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use uniserve_core::TokenLogprob;
use uniserve_cuda::{Event, SourceOrder};

use crate::ffi::{Backing, DType, Device};
use crate::storage::memory::{Memory, StorageView};
use crate::stream::Issue;
use crate::transfer::{Spans, copy_planner};
use crate::{Error, EventPool, Result};

mod logprobs;

pub use logprobs::LogprobLayout;

/// One batch's host words: page-locked for CUDA lanes, pageable for CPU
/// lanes. Only storage is reused; each batch receives a distinct
/// [`OutputBuffer`] and completion signal.
pub struct OutputStorage {
    memory: Arc<Memory>,
    words: usize,
}

impl OutputStorage {
    /// Host storage for `words` 64-bit words, pinned for `device`'s copies.
    pub fn allocate(device: Option<i32>, words: usize) -> Result<Self> {
        let bytes = words.max(1) * size_of::<i64>();
        let memory = match device {
            Some(device) => Memory::pinned(device, bytes)?,
            None => Memory::host(bytes),
        };
        Ok(Self { memory, words })
    }

    pub fn words(&self) -> usize {
        self.words
    }

    fn address(&self) -> u64 {
        self.memory.address()
    }
}

struct DeviceEvents {
    device: i32,
    start: Option<Arc<Event>>,
    copy: Option<Arc<Event>>,
    end: Option<Arc<Event>>,
}

/// A batch's readback ranges, device fences and row/CPU-reader lifetime.
/// Mutations are serialized by its owner.
pub struct OutputBuffer {
    storage: Option<OutputStorage>,
    rows: Vec<bool>,
    remaining: usize,
    tokens: usize,
    /// Token captures narrower than 64 bits: `(word offset, count, dtype)`.
    /// Their raw elements are packed at the start of their reserved words and
    /// widened on readback.
    narrow: Vec<(usize, usize, DType)>,
    readback: Option<Vec<i64>>,
    logprob_layouts: HashMap<(usize, usize), LogprobLayout>,
    logprobs: HashMap<(usize, usize), HashMap<usize, Vec<TokenLogprob>>>,
    bytes: usize,
    devices: Vec<DeviceEvents>,
    sealed: bool,
    copies_finished: bool,
    abandoned: bool,
    readers: usize,
    timing_events: bool,
    reserved: Instant,
    started: Option<Instant>,
    copying: Option<Instant>,
    sealed_at: Option<Instant>,
    ready_at: Option<Instant>,
    timing: Option<[u64; 4]>,
}

impl OutputBuffer {
    pub fn new(storage: OutputStorage, rows: usize, devices: &[i32], timing: bool) -> Self {
        Self {
            storage: Some(storage),
            rows: vec![false; rows],
            remaining: rows,
            tokens: 0,
            narrow: Vec::new(),
            readback: None,
            logprob_layouts: HashMap::new(),
            logprobs: HashMap::new(),
            bytes: 0,
            devices: devices
                .iter()
                .map(|&device| DeviceEvents {
                    device,
                    start: None,
                    copy: None,
                    end: None,
                })
                .collect(),
            sealed: false,
            copies_finished: false,
            abandoned: false,
            readers: 0,
            timing_events: timing,
            reserved: Instant::now(),
            started: None,
            copying: None,
            sealed_at: None,
            ready_at: None,
            timing: None,
        }
    }

    pub fn sealed(&self) -> bool {
        self.sealed
    }

    pub fn storage(&self) -> Result<&OutputStorage> {
        self.storage
            .as_ref()
            .ok_or_else(|| Error::Invariant("output storage has retired".into()))
    }

    /// Copy an integer token column into the next token words, on `stream`
    /// for a device column. Returns the capture's `(word offset, count)`.
    ///
    /// The source holds `int64`, `int32` or `uint32` tokens in any layout;
    /// narrower values are widened when read back. The producing stream must
    /// remain alive until the buffer is sealed.
    pub fn capture(
        &mut self,
        source: &StorageView,
        stream: impl Issue,
        events: &EventPool,
    ) -> Result<(usize, usize)> {
        let dtype = source.dtype();
        if !matches!(dtype, DType::I64 | DType::I32 | DType::U32) {
            return Err(Error::Invalid(format!(
                "token captures require 32- or 64-bit integers, not {dtype:?}"
            )));
        }
        let count = source.numel() as usize;
        let offset = self.reserve_tokens(count)?;
        let address = self.storage()?.address() + (offset * size_of::<i64>()) as u64;
        self.copy(source, address, stream, events)?;
        if dtype != DType::I64 {
            self.narrow.push((offset, count, dtype));
        }
        Ok((offset, count))
    }

    /// Copy a `uint8` tensor into the byte region at the back of the buffer
    /// and return the host view of the captured bytes, readable once the
    /// buffer's fences completed and retained by the caller as a CPU reader.
    pub fn capture_bytes(
        &mut self,
        source: &StorageView,
        stream: impl Issue,
        events: &EventPool,
    ) -> Result<StorageView> {
        if source.dtype() != DType::U8 {
            return Err(Error::Invalid(
                "byte captures require a uint8 tensor".into(),
            ));
        }
        let count = source.numel() as usize;
        let offset = self.reserve_bytes(count)?;
        let memory: Arc<dyn Backing> = Arc::clone(&self.storage()?.memory) as Arc<dyn Backing>;
        let capture = StorageView::contiguous(memory, offset, DType::U8, source.shape())?;
        self.copy(source, capture.address(), stream, events)?;
        Ok(capture)
    }

    /// Copy `source` into host storage at `address`: on `stream` for a
    /// device source (after fencing its device for timing), as a host copy
    /// in `stream`'s order for host memory.
    fn copy(
        &mut self,
        source: &StorageView,
        address: u64,
        stream: impl Issue,
        events: &EventPool,
    ) -> Result<()> {
        let memory: Arc<dyn Backing> = Arc::clone(&self.storage()?.memory) as Arc<dyn Backing>;
        let base = memory.address();
        let target = StorageView::contiguous(
            memory,
            (address - base) as usize,
            source.dtype(),
            source.shape(),
        )?;
        let target = Spans::single(target)?;
        let source_spans = Spans::single(source.clone())?;

        match source.device() {
            Device::Cuda(device) => {
                let producer = stream.stream().ok_or_else(|| {
                    Error::Invalid("a device capture requires its producing stream".into())
                })?;
                if producer.device().0 != device {
                    return Err(Error::Invalid(
                        "a capture stream must belong to its source device".into(),
                    ));
                }
                self.begin_copy(Some(device), stream, events)?;
                let mut plan = copy_planner(Some(device), SourceOrder::Stream)?;
                target.plan_copy(&source_spans, &mut plan)?;
                // SAFETY: the pinned destination stays allocated until the
                // seal fences complete; the source is the producer's storage,
                // which the producer keeps until its stream passes this copy.
                unsafe { stream.copy_strided(&plan) }?;
            }
            Device::Cpu | Device::CudaHost => {
                self.register_device(None)?;
                self.copying.get_or_insert_with(Instant::now);
                let mut plan = copy_planner(None, SourceOrder::Stream)?;
                target.plan_copy(&source_spans, &mut plan)?;
                // SAFETY: both operands are host memory; the reserved words
                // belong to this buffer and no device copy writes them. The
                // producer keeps its storage until the copy ran.
                unsafe { stream.host().copy_strided(&plan) }?;
            }
        }
        Ok(())
    }

    /// Reserve opposing word and byte ranges before submitting a copy. A failed
    /// copy abandons its batch, so its range must remain occupied until retirement.
    pub fn reserve_tokens(&mut self, count: usize) -> Result<usize> {
        self.require_open()?;
        let available = (self.storage()?.words * 8 - self.bytes) / 8;
        let end = self
            .tokens
            .checked_add(count)
            .filter(|&end| end <= available)
            .ok_or(Error::Resource(
                "completion capture exceeds pinned output capacity",
            ))?;
        let offset = self.tokens;
        self.tokens = end;
        Ok(offset)
    }

    pub fn reserve_bytes(&mut self, count: usize) -> Result<usize> {
        self.require_open()?;
        let capacity = self.storage()?.words * 8;
        self.bytes = self
            .bytes
            .checked_add(count)
            .filter(|&end| count > 0 && end <= capacity - self.tokens * 8)
            .ok_or(Error::Resource(
                "byte capture exceeds pinned output capacity",
            ))?;
        Ok(capacity - self.bytes)
    }

    pub fn token_range(&self, offset: usize, count: usize) -> Result<()> {
        if !offset
            .checked_add(count)
            .is_some_and(|end| end <= self.tokens)
        {
            return Err(Error::Invariant(
                "completion capture range is outside its registered token extent".into(),
            ));
        }
        Ok(())
    }

    /// Snapshot completed words once, before the host allocation can be
    /// recycled. Token captures narrower than 64 bits are widened here; byte
    /// captures retain their separate CPU-reader lifetime.
    pub fn readback(&mut self) -> Result<()> {
        if self.readback.is_some() {
            return Ok(());
        }
        if self.ready_at().is_none() {
            return Err(Error::Invariant(
                "completion storage was observed before its copy event was ready".into(),
            ));
        }

        let storage = self.storage()?;
        let mut words = vec![0_i64; self.tokens];
        // SAFETY: every copy writing these words completed (`ready_at`), and
        // the buffer's mutex prevents recycling the storage while it is read.
        unsafe {
            std::ptr::copy_nonoverlapping(
                storage.address() as *const i64,
                words.as_mut_ptr(),
                self.tokens,
            );
        }
        for &(offset, count, dtype) in &self.narrow {
            // The capture packed `count` 4-byte values at its first word.
            let packed: Vec<u32> = words[offset..offset + count.div_ceil(2)]
                .iter()
                .flat_map(|word| {
                    let bytes = word.to_ne_bytes();
                    [
                        u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                        u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
                    ]
                })
                .take(count)
                .collect();
            for (word, value) in words[offset..offset + count].iter_mut().zip(packed) {
                *word = match dtype {
                    DType::I32 => i64::from(value as i32),
                    _ => i64::from(value),
                };
            }
        }
        self.readback = Some(words);
        Ok(())
    }

    pub fn read_tokens(&self, offset: usize, count: usize) -> Result<&[i64]> {
        self.token_range(offset, count)?;
        self.readback
            .as_ref()
            .and_then(|words| words.get(offset..offset + count))
            .ok_or(Error::State("completion words have not been read back"))
    }

    pub fn register_logprobs(
        &mut self,
        offset: usize,
        count: usize,
        layout: LogprobLayout,
    ) -> Result<()> {
        self.require_open()?;
        self.token_range(offset, count)?;
        layout.validate(count)?;
        self.logprob_layouts.insert((offset, count), layout);
        Ok(())
    }

    /// Upper bound before token deduplication, used for output-byte admission.
    pub fn logprob_entries(&self, span: (usize, usize, usize)) -> Result<usize> {
        let (offset, count, row) = span;
        self.logprob_layouts
            .get(&(offset, count))
            .ok_or(Error::State("logprob capture has no row layout"))?
            .entries(row)
    }

    /// IPC encodes a 4-byte row length and 12 bytes per token score. Admission
    /// uses the count before deduplication, so selected/top/requested overlap
    /// can only reduce the eventual payload.
    pub fn logprob_bytes(&self, spans: &[(usize, usize, usize)]) -> Result<usize> {
        spans.iter().try_fold(0, |bytes, &span| {
            Ok(bytes + 4 + 12 * self.logprob_entries(span)?)
        })
    }

    /// Selected token first, then top and explicitly requested tokens. The
    /// packed column is decoded once for all of its consuming request rows.
    pub fn logprob_values(&mut self, span: (usize, usize, usize)) -> Result<&[TokenLogprob]> {
        let (offset, count, row) = span;
        let key = (offset, count);
        if !self.logprobs.contains_key(&key) {
            let layout = self
                .logprob_layouts
                .get(&key)
                .ok_or(Error::State("logprob capture has no row layout"))?;
            let values = layout.decode(self.read_tokens(offset, count)?)?;
            self.logprobs.insert(key, values);
        }
        self.logprobs[&key]
            .get(&row)
            .map(Vec::as_slice)
            .ok_or(Error::State("logprob capture has no requested row"))
    }

    pub fn register_device(&self, device: Option<i32>) -> Result<()> {
        self.require_open()?;
        if device.is_some_and(|device| !self.devices.iter().any(|events| events.device == device)) {
            return Err(Error::Invariant(
                "output capture uses an undeclared CUDA device".into(),
            ));
        }
        Ok(())
    }

    /// Fence the start of `device`'s work for timing, once per batch.
    pub fn begin_device(
        &mut self,
        device: Option<i32>,
        stream: impl Issue,
        pool: &EventPool,
    ) -> Result<()> {
        self.register_device(device)?;
        self.started.get_or_insert_with(Instant::now);

        if let Some(device) = device.filter(|_| self.timing_events)
            && let Some(events) = self
                .devices
                .iter_mut()
                .find(|events| events.device == device)
            && events.start.is_none()
        {
            events.start = Some(record(pool, stream, true)?);
        }
        Ok(())
    }

    pub fn begin_copy(
        &mut self,
        device: Option<i32>,
        stream: impl Issue,
        pool: &EventPool,
    ) -> Result<()> {
        self.register_device(device)?;
        self.copying.get_or_insert_with(Instant::now);

        if let Some(device) = device.filter(|_| self.timing_events) {
            self.begin_device(Some(device), stream, pool)?;
            if let Some(events) = self
                .devices
                .iter_mut()
                .find(|events| events.device == device)
                && events.copy.is_none()
            {
                events.copy = Some(record(pool, stream, true)?);
            }
        }
        Ok(())
    }

    /// Record all producer fences once, one per declared device on that
    /// device's stream. The caller transfers their pool references to
    /// deferred release and calls `copies_finished` after they complete.
    pub fn seal<I: Issue>(
        &mut self,
        streams: &[I],
        pool: &EventPool,
    ) -> Result<Option<Vec<Arc<Event>>>> {
        if self.sealed {
            return Ok(None);
        }

        for &stream in streams {
            let device = device_of(&stream)?;
            if self.timing_events {
                self.begin_copy(Some(device), stream, pool)?;
            }
            let timing = self.timing_events;
            let events = self
                .devices
                .iter_mut()
                .find(|events| events.device == device)
                .ok_or_else(|| {
                    Error::Invariant("output fence uses an undeclared CUDA device".into())
                })?;
            if events.end.is_none() {
                events.end = Some(record(pool, stream, timing)?);
            }
        }

        self.sealed = true;
        self.sealed_at = Some(Instant::now());
        // Without a fence no device work writes the words: they are ready.
        if self.fences().next().is_none() {
            self.ready_at = self.sealed_at;
        }
        Ok(Some(self.events()))
    }

    pub fn events(&self) -> Vec<Arc<Event>> {
        self.devices
            .iter()
            .flat_map(|events| [&events.start, &events.copy, &events.end])
            .filter_map(|event| event.clone())
            .collect()
    }

    pub fn fences(&self) -> impl Iterator<Item = &Arc<Event>> {
        self.devices.iter().filter_map(|events| events.end.as_ref())
    }

    /// Each fence with the device whose stream recorded it.
    pub fn device_fences(&self) -> impl Iterator<Item = (i32, &Arc<Event>)> {
        self.devices
            .iter()
            .filter_map(|events| Some((events.device, events.end.as_ref()?)))
    }

    /// When the lease learned that every copy into it completed, or `None`
    /// while it has not: a lease without fences knows when it is sealed,
    /// one with fences when it is told ([`Self::fences_completed`],
    /// [`Self::copies_finished`]). The lease does not ask the driver.
    pub fn ready_at(&self) -> Option<Instant> {
        self.ready_at.filter(|_| self.sealed)
    }

    /// The device finished every copy the fences cover: the words can be
    /// read without asking the driver.
    pub fn fences_completed(&mut self) {
        self.ready_at.get_or_insert_with(Instant::now);
    }

    pub fn copies_finished(&mut self) {
        self.copies_finished = true;
        self.ready_at.get_or_insert_with(Instant::now);
    }

    pub fn observe(&mut self, row: usize) -> Result<[u64; 4]> {
        if row >= self.rows.len() {
            return Err(Error::Invariant(
                "completion row is outside its output buffer".into(),
            ));
        }
        let Some(ready_at) = self.ready_at() else {
            return Err(Error::Invariant(
                "completion row was observed before its copies completed".into(),
            ));
        };
        let timing = match self.timing {
            Some(timing) => timing,
            None => self.measure(ready_at)?,
        };
        self.timing = Some(timing);
        self.discard(row);
        Ok(timing)
    }

    pub fn timing(&self) -> Result<[u64; 4]> {
        self.timing
            .ok_or_else(|| Error::Invariant("completion timing was read before observation".into()))
    }

    pub fn discard(&mut self, row: usize) {
        if let Some(observed) = self.rows.get_mut(row)
            && !std::mem::replace(observed, true)
        {
            self.remaining -= 1;
        }
    }

    pub fn abandon(&mut self) {
        self.abandoned = true;
    }

    pub fn retain_reader(&mut self) -> Result<()> {
        self.storage()?;
        self.readers += 1;
        Ok(())
    }

    pub fn release_reader(&mut self) -> Result<()> {
        self.readers = self
            .readers
            .checked_sub(1)
            .ok_or_else(|| Error::Invariant("output CPU reader count underflow".into()))?;
        Ok(())
    }

    fn retired(&self) -> bool {
        self.copies_finished && (self.remaining == 0 || self.abandoned) && self.readers == 0
    }

    fn require_open(&self) -> Result<()> {
        if self.sealed {
            return Err(Error::Invariant("output capture is already sealed".into()));
        }
        Ok(())
    }

    fn measure(&self, ready_at: Instant) -> Result<[u64; 4]> {
        let queued = self.started.map_or(0, |start| micros(start, self.reserved));
        let mut device = 0;
        let mut copy = 0;
        if self.timing_events {
            for events in &self.devices {
                if let (Some(start), Some(producer), Some(end)) =
                    (&events.start, &events.copy, &events.end)
                {
                    // CUDA reports milliseconds; output statistics use microseconds.
                    device = device.max(
                        (start.elapsed_ms(producer)? as f64 * 1000.0).round_ties_even() as u64,
                    );
                    copy = copy
                        .max((producer.elapsed_ms(end)? as f64 * 1000.0).round_ties_even() as u64);
                }
            }
        }
        if self.devices.is_empty()
            && let (Some(start), Some(sealed)) = (self.started, self.sealed_at)
        {
            let copying = self.copying.unwrap_or(sealed);
            device = micros(copying, start);
            copy = micros(sealed, copying);
        }
        Ok([queued, device, copy, micros(Instant::now(), ready_at)])
    }
}

fn micros(end: Instant, start: Instant) -> u64 {
    end.saturating_duration_since(start).as_micros() as u64
}

/// The device of the stream `stream` issues on.
fn device_of(stream: &impl Issue) -> Result<i32> {
    stream
        .stream()
        .map(|stream| stream.device().0)
        .ok_or_else(|| Error::Invariant("an output fence is recorded on a stream".into()))
}

fn record(pool: &EventPool, stream: impl Issue, timing: bool) -> Result<Arc<Event>> {
    let device = device_of(&stream)?;
    let event = pool.acquire(device, timing, false)?;
    pool.retain(&event, device, 1)?;
    pool.record(&event, stream)?;
    Ok(event)
}

/// A batch's output buffer, shared by its pending results and the pool.
pub type OutputLease = Arc<Mutex<OutputBuffer>>;

/// Bounded output storage of one lane. Storage is reused only after every
/// lease using it retired: its copies finished, every row was observed or
/// the batch abandoned, and no CPU reader holds a byte capture.
pub struct OutputPool {
    device: Option<i32>,
    capacity: usize,
    max_words: usize,
    active: Vec<OutputLease>,
    available: Vec<OutputStorage>,
    closed: bool,
}

impl OutputPool {
    /// `capacity` leases of at most `max_words` words, pinned for `device`
    /// copies (pageable for a CPU lane).
    pub fn new(device: Option<i32>, capacity: usize, max_words: usize) -> Self {
        Self {
            device,
            capacity,
            max_words,
            active: Vec::new(),
            available: Vec::new(),
            closed: false,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn max_words(&self) -> usize {
        self.max_words
    }

    /// Recycle the storage of retired leases; returns how many retired.
    pub fn reap(&mut self) -> usize {
        let mut retired = 0;
        let mut index = 0;
        while index < self.active.len() {
            let mut buffer = self.active[index]
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if !buffer.retired() {
                index += 1;
                continue;
            }
            if let Some(storage) = buffer.storage.take() {
                self.available.push(storage);
            }
            // Released event objects can now be reused even if a result retains
            // its consumed lease. Timing and decoded values remain per lease.
            buffer.devices.clear();
            drop(buffer);
            self.active.swap_remove(index);
            retired += 1;
        }
        retired
    }

    /// Lease storage for a batch of `rows` result rows and `words` capture
    /// words, reusing retired storage or allocating the lane's maximum.
    pub fn lease(
        &mut self,
        rows: usize,
        words: usize,
        devices: &[i32],
        timing: bool,
    ) -> Result<OutputLease> {
        if self.closed {
            return Err(Error::Resource("output pool is closed"));
        }
        if words > self.max_words {
            return Err(Error::Resource(
                "lane output exceeds its startup storage bound",
            ));
        }
        if rows == 0 || words < rows {
            return Err(Error::Resource(
                "completion row count exceeds pinned output capacity",
            ));
        }
        self.reap();
        if self.active.len() >= self.capacity {
            return Err(Error::Resource("all lane output leases are active"));
        }

        let storage = match self.available.pop() {
            Some(storage) => storage,
            None => OutputStorage::allocate(self.device, self.max_words)?,
        };
        let lease = Arc::new(Mutex::new(OutputBuffer::new(
            storage, rows, devices, timing,
        )));
        self.active.push(Arc::clone(&lease));
        Ok(lease)
    }

    /// Host bytes held by allocated storage, for capacity accounting.
    pub fn bytes(capacity: usize, max_words: usize) -> u64 {
        (capacity * max_words.max(1) * size_of::<i64>()) as u64
    }

    pub fn begin_close(&mut self) {
        self.closed = true;
    }

    /// Release storage only after shutdown has drained its device writes.
    /// Leases still referenced by results keep their own storage alive.
    pub fn finish_close(&mut self) {
        self.active.clear();
        self.available.clear();
    }
}

impl crate::executor::lifecycle::RequestStorage for OutputPool {
    fn admit_slots(
        &mut self,
        _slots: &[usize],
        _jobs: &crate::executor::lifecycle::LaneJobs,
    ) -> Result<()> {
        Ok(())
    }

    fn release_buffers(
        &mut self,
        _buffers: &std::collections::HashSet<uniserve_ipc::BufferId>,
        _jobs: &crate::executor::lifecycle::LaneJobs,
    ) -> Result<()> {
        Ok(())
    }

    fn release_requests(
        &mut self,
        _requests: &std::collections::HashSet<uniserve_ipc::RequestKey>,
        _retained: &std::collections::HashSet<uniserve_ipc::BufferId>,
        _jobs: &crate::executor::lifecycle::LaneJobs,
    ) -> Result<()> {
        Ok(())
    }

    fn retirement_ready(
        &mut self,
        _buffers: &std::collections::HashSet<uniserve_ipc::BufferId>,
        _requests: &std::collections::HashSet<uniserve_ipc::RequestKey>,
        _retained: &std::collections::HashSet<uniserve_ipc::BufferId>,
    ) -> Result<bool> {
        Ok(true)
    }

    fn retire_slots(
        &mut self,
        _slots: &[usize],
        _jobs: &crate::executor::lifecycle::LaneJobs,
    ) -> Result<()> {
        Ok(())
    }

    /// Return the storage of retired leases to the pool.
    fn reap(&mut self) -> Result<()> {
        OutputPool::reap(self);
        Ok(())
    }

    fn close(&mut self, _aborted: bool) -> Result<()> {
        self.begin_close();
        self.finish_close();
        Ok(())
    }
}
