//! Bounded host output storage and per-batch readback lifetimes.
//!
//! A batch's results are copied device-to-host into the batch's range of
//! its runtime's word arena (`OutputPool`): one arena per runtime on each
//! lane, allocated when that runtime starts, page-locked for device copies
//! and pageable on a CPU lane. Integer token columns fill the range
//! from its front, raw byte captures from its back. The copies are ordered
//! on the producing streams and fenced by events; the host reads the words
//! only after those fences complete, widening narrower token columns to 64
//! bits on the host.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use uniserve_core::TokenLogprob;
use uniserve_cuda::{DeviceId, Event, SourceOrder};

use crate::ffi::{Backing, DType, MemoryKind};
use crate::storage::memory::{Memory, StorageView};
use crate::stream::Issue;
use crate::transfer::{Spans, copy_planner};
use crate::{Error, EventPool, Result, lock};

mod logprobs;

pub use logprobs::LogprobLayout;

/// One batch's host words: `words` 64-bit words from word `offset` of
/// `memory`, page-locked for CUDA lanes and pageable for CPU lanes. Only
/// storage is reused; each batch receives a distinct [`OutputBuffer`] and
/// completion signal.
pub struct OutputStorage {
    memory: Arc<Memory>,
    offset: usize,
    words: usize,
}

impl OutputStorage {
    /// Host storage of its own for `words` 64-bit words, pinned for
    /// `device`'s copies.
    pub fn allocate(device: Option<DeviceId>, words: usize) -> Result<Self> {
        Ok(Self {
            memory: words_memory(device, words)?,
            offset: 0,
            words,
        })
    }

    pub fn words(&self) -> usize {
        self.words
    }

    /// Byte offset of the first word in `memory`.
    fn byte_offset(&self) -> usize {
        self.offset * size_of::<i64>()
    }

    fn address(&self) -> u64 {
        self.memory.address() + self.byte_offset() as u64
    }
}

/// Host memory for `words` 64-bit words, pinned for `device`'s copies.
fn words_memory(device: Option<DeviceId>, words: usize) -> Result<Arc<Memory>> {
    let bytes = words.max(1) * size_of::<i64>();
    Ok(match device {
        Some(device) => Memory::pinned(device, bytes)?,
        None => Memory::host(bytes),
    })
}

struct DeviceEvents {
    device: DeviceId,
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
    pub fn new(storage: OutputStorage, rows: usize, devices: &[DeviceId], timing: bool) -> Self {
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

    pub(crate) fn sealed(&self) -> bool {
        self.sealed
    }

    pub(crate) fn storage(&self) -> Result<&OutputStorage> {
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
        let storage = self.storage()?;
        let memory: Arc<dyn Backing> = Arc::clone(&storage.memory) as Arc<dyn Backing>;
        let capture = StorageView::contiguous(
            memory,
            storage.byte_offset() + offset,
            DType::U8,
            source.shape(),
        )?;
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
            MemoryKind::Cuda(device) => {
                let producer = stream.stream().ok_or_else(|| {
                    Error::Invalid("a device capture requires its producing stream".into())
                })?;
                if producer.device() != device {
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
            MemoryKind::Host | MemoryKind::PinnedHost => {
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
    pub(crate) fn reserve_tokens(&mut self, count: usize) -> Result<usize> {
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

    pub(crate) fn reserve_bytes(&mut self, count: usize) -> Result<usize> {
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

    pub(crate) fn token_range(&self, offset: usize, count: usize) -> Result<()> {
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

    /// Selected token first, then the top tokens. The packed column is
    /// decoded once for all of its consuming request rows.
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

    pub(crate) fn register_device(&self, device: Option<DeviceId>) -> Result<()> {
        self.require_open()?;
        if device.is_some_and(|device| !self.devices.iter().any(|events| events.device == device)) {
            return Err(Error::Invariant(
                "output capture uses an undeclared CUDA device".into(),
            ));
        }
        Ok(())
    }

    /// Fence the start of `device`'s work for timing, once per batch.
    pub(crate) fn begin_device(
        &mut self,
        device: Option<DeviceId>,
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

    pub(crate) fn begin_copy(
        &mut self,
        device: Option<DeviceId>,
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

    pub(crate) fn events(&self) -> Vec<Arc<Event>> {
        self.devices
            .iter()
            .flat_map(|events| [&events.start, &events.copy, &events.end])
            .filter_map(|event| event.clone())
            .collect()
    }

    pub(crate) fn fences(&self) -> impl Iterator<Item = &Arc<Event>> {
        self.devices.iter().filter_map(|events| events.end.as_ref())
    }

    /// Each fence with the device whose stream recorded it.
    pub(crate) fn device_fences(&self) -> impl Iterator<Item = (DeviceId, &Arc<Event>)> {
        self.devices
            .iter()
            .filter_map(|events| Some((events.device, events.end.as_ref()?)))
    }

    /// When the lease learned that every copy into it completed, or `None`
    /// while it has not: a lease without fences knows when it is sealed,
    /// one with fences when it is told ([`Self::fences_completed`],
    /// [`Self::copies_finished`]). The lease does not ask the driver.
    pub(crate) fn ready_at(&self) -> Option<Instant> {
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

    pub(crate) fn discard(&mut self, row: usize) {
        if let Some(observed) = self.rows.get_mut(row)
            && !std::mem::replace(observed, true)
        {
            self.remaining -= 1;
        }
    }

    pub(crate) fn abandon(&mut self) {
        self.abandoned = true;
    }

    pub(crate) fn retain_reader(&mut self) -> Result<()> {
        self.storage()?;
        self.readers += 1;
        Ok(())
    }

    pub(crate) fn release_reader(&mut self) -> Result<()> {
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
fn device_of(stream: &impl Issue) -> Result<DeviceId> {
    stream
        .stream()
        .map(|stream| stream.device())
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
pub(crate) type OutputLease = Arc<Mutex<OutputBuffer>>;

/// Output storage of one lane: one host word arena, allocated when the lane
/// starts, that every batch leases exactly its words from.
///
/// The arena is `blocks` blocks of `block_words` words, the largest lease
/// the lane's own bounds allow. A lease lies within one block, first fit, so
/// while fewer than `blocks` leases are live some block is entirely free and
/// any lease of at most `block_words` fits: `blocks` is the number of the
/// largest batches the lane holds at once, and smaller batches share
/// blocks. A lease's range is reused only after the lease retired: its
/// copies finished, every row was observed or the batch abandoned, and no
/// CPU reader holds a byte capture.
///
/// A batch of more words than a block, such as a readout whose candidate
/// count no admission limit bounds, reads back into storage of its own,
/// allocated for it at its lease and released when it retires: the
/// reservation follows the request.
pub(crate) struct OutputPool {
    /// Released by [`Self::finish_close`].
    arena: Option<Arc<Memory>>,
    /// The device the lane's copies read from; `None` for a CPU lane.
    device: Option<DeviceId>,
    blocks: usize,
    block_words: usize,
    /// Live leases and their word ranges in the arena; `None` for a lease
    /// with storage of its own.
    active: Vec<(OutputLease, Option<Range<usize>>)>,
    closed: bool,
}

impl OutputPool {
    /// `blocks` blocks of `block_words` words, pinned for `device` copies
    /// (pageable for a CPU lane), allocated now so that no batch within the
    /// lane's bounds allocates host memory.
    pub(crate) fn new(device: Option<DeviceId>, blocks: usize, block_words: usize) -> Result<Self> {
        let blocks = blocks.max(1);
        let words = blocks
            .checked_mul(block_words)
            .ok_or(Error::Resource("lane output storage overflows"))?;
        Ok(Self {
            arena: Some(words_memory(device, words)?),
            device,
            blocks,
            block_words,
            active: Vec::new(),
            closed: false,
        })
    }

    /// Recycle the ranges of retired leases; returns how many retired.
    pub(crate) fn reap(&mut self) -> usize {
        let mut retired = 0;
        let mut index = 0;
        while index < self.active.len() {
            let mut buffer = lock(&self.active[index].0);
            if !buffer.retired() {
                index += 1;
                continue;
            }
            // The range returns to the arena (or storage of its own is
            // released) with its entry below; a result that retains the
            // consumed lease keeps its timing and decoded values, not the
            // words.
            buffer.storage = None;
            // Released event objects can now be reused even if a result retains
            // its consumed lease.
            buffer.devices.clear();
            drop(buffer);
            self.active.swap_remove(index);
            retired += 1;
        }
        retired
    }

    /// Lease `words` capture words for a batch of `rows` result rows, after
    /// recycling retired leases: from the arena when they fit a block, else
    /// in storage of the batch's own.
    pub(crate) fn lease(
        &mut self,
        rows: usize,
        words: usize,
        devices: &[DeviceId],
        timing: bool,
    ) -> Result<OutputLease> {
        let Some(arena) = self.arena.as_ref().filter(|_| !self.closed) else {
            return Err(Error::Resource("output pool is closed"));
        };
        let arena = Arc::clone(arena);
        if rows == 0 || words < rows {
            return Err(Error::Resource(
                "completion row count exceeds pinned output capacity",
            ));
        }
        self.reap();
        if words > self.block_words {
            let storage = OutputStorage::allocate(self.device, words)?;
            let lease = Arc::new(Mutex::new(OutputBuffer::new(
                storage, rows, devices, timing,
            )));
            self.active.push((Arc::clone(&lease), None));
            return Ok(lease);
        }
        let offset = self
            .place(words)
            .ok_or(Error::Resource("all lane output storage is leased"))?;

        let storage = OutputStorage {
            memory: arena,
            offset,
            words,
        };
        let lease = Arc::new(Mutex::new(OutputBuffer::new(
            storage, rows, devices, timing,
        )));
        self.active
            .push((Arc::clone(&lease), Some(offset..offset + words)));
        Ok(lease)
    }

    /// The first word of the first gap of at least `words` words inside one
    /// block, in arena order.
    fn place(&self, words: usize) -> Option<usize> {
        let mut ranges: Vec<&Range<usize>> = self
            .active
            .iter()
            .filter_map(|(_, range)| range.as_ref())
            .collect();
        ranges.sort_unstable_by_key(|range| range.start);
        let mut ranges = ranges.into_iter().peekable();
        for block in 0..self.blocks {
            let end = (block + 1) * self.block_words;
            let mut cursor = block * self.block_words;
            // Leases never cross a block boundary, so the ranges starting in
            // this block also end in it.
            while let Some(range) = ranges.next_if(|range| range.start < end) {
                if range.start - cursor >= words {
                    return Some(cursor);
                }
                cursor = range.end;
            }
            if end - cursor >= words {
                return Some(cursor);
            }
        }
        None
    }

    pub(crate) fn begin_close(&mut self) {
        self.closed = true;
    }

    /// Release storage only after shutdown has drained its device writes.
    /// Leases still referenced by results keep the arena alive.
    pub(crate) fn finish_close(&mut self) {
        self.active.clear();
        self.arena = None;
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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{OutputBuffer, OutputPool};
    use crate::ffi::{Backing, DType};
    use crate::storage::memory::{Memory, StorageView};
    use crate::{EventPool, Result, lock};

    /// A host column holding `values`, as a CPU lane's sampler produces it.
    fn column(values: &[i64]) -> Result<StorageView> {
        let memory = Memory::host(size_of_val(values));
        // SAFETY: fresh host memory of exactly this size, owned by the test.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                memory.address() as *mut i64,
                values.len(),
            );
        }
        StorageView::contiguous(
            memory as Arc<dyn Backing>,
            0,
            DType::I64,
            &[values.len() as i64],
        )
    }

    /// Lease `words` words of `pool` for a one-row batch, with `value`
    /// captured in each word.
    fn lease_filled(
        pool: &mut OutputPool,
        words: usize,
        value: i64,
    ) -> Result<Arc<Mutex<OutputBuffer>>> {
        let lease = pool.lease(1, words, &[], false)?;
        lock(&lease).capture(&column(&vec![value; words])?, None, &EventPool::default())?;
        Ok(lease)
    }

    /// Complete a leased batch and give its words back: its copies finished
    /// and its row was observed.
    fn retire(lease: &Arc<Mutex<OutputBuffer>>) -> Result<Vec<i64>> {
        let mut buffer = lock(lease);
        // Host captures are sealed on no stream.
        buffer.seal::<&uniserve_cuda::Stream>(&[], &EventPool::default())?;
        buffer.copies_finished();
        buffer.readback()?;
        let words = buffer.read_tokens(0, buffer.storage()?.words())?.to_vec();
        buffer.discard(0);
        Ok(words)
    }

    #[test]
    fn a_lane_holds_its_largest_batches_at_once_and_smaller_ones_share_its_storage() -> Result<()> {
        // Storage for two of the lane's largest batches, of eight words each.
        let mut pool = OutputPool::new(None, 2, 8)?;

        // Two of the largest batches at once; a third waits for one to retire.
        let first = lease_filled(&mut pool, 8, 1)?;
        let second = lease_filled(&mut pool, 8, 2)?;
        assert!(pool.lease(1, 1, &[], false).is_err());
        assert_eq!(retire(&first)?, [1; 8]);
        let third = lease_filled(&mut pool, 8, 3)?;
        assert_eq!(retire(&second)?, [2; 8]);
        assert_eq!(retire(&third)?, [3; 8]);

        // Batches take their own words only: sixteen batches of one word fill
        // the storage of the two largest, each keeping its own value.
        let small = (0..16)
            .map(|value| lease_filled(&mut pool, 1, value))
            .collect::<Result<Vec<_>>>()?;
        assert!(pool.lease(1, 1, &[], false).is_err());
        for (value, lease) in small.iter().enumerate() {
            assert_eq!(retire(lease)?, [value as i64]);
        }

        // A small batch still live leaves room for one of the largest.
        let small = lease_filled(&mut pool, 3, 4)?;
        let large = lease_filled(&mut pool, 8, 5)?;
        assert!(pool.lease(1, 8, &[], false).is_err());
        assert_eq!(retire(&small)?, [4; 3]);
        let other = lease_filled(&mut pool, 8, 6)?;
        assert_eq!(retire(&large)?, [5; 8]);
        assert_eq!(retire(&other)?, [6; 8]);
        Ok(())
    }

    /// A batch of more words than a block, as a readout of more candidates
    /// than the lane's bounds state, reads back into storage of its own: it
    /// holds every word, and both blocks stay free for the largest batches.
    #[test]
    fn a_batch_larger_than_a_block_reads_back_into_storage_of_its_own() -> Result<()> {
        let mut pool = OutputPool::new(None, 2, 8)?;
        let largest = lease_filled(&mut pool, 8, 1)?;
        let beyond = lease_filled(&mut pool, 9, 2)?;
        let other = lease_filled(&mut pool, 8, 3)?;
        assert!(pool.lease(1, 1, &[], false).is_err());
        assert_eq!(retire(&beyond)?, [2; 9]);
        assert_eq!(retire(&largest)?, [1; 8]);
        assert_eq!(retire(&other)?, [3; 8]);

        // Its storage goes with it: the pool again holds two of the largest.
        assert_eq!(pool.reap(), 3);
        let first = lease_filled(&mut pool, 8, 4)?;
        let second = lease_filled(&mut pool, 8, 5)?;
        assert_eq!(retire(&first)?, [4; 8]);
        assert_eq!(retire(&second)?, [5; 8]);
        Ok(())
    }
}
