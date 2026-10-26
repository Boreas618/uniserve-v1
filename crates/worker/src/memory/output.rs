//! Bounded host output storage and per-batch readback lifetimes.
//!
//! A batch's results reach the host in the batch's range of a word arena
//! (`OutputPool`): one arena per executor computation and one for the transfer
//! runtime, allocated at startup, page-locked for device copies and pageable
//! for a computation on the host. The producer writes the range itself, on the
//! batch's stream in its own order: `int64` words from the range's front
//! ([`OutputBuffer::lend_words`]) and raw bytes, such as decoded pixels, from
//! its back ([`OutputBuffer::lend_bytes`]). The lease's seal fences those
//! copies; the host reads the words only after the fences complete.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cuda::{DeviceId, Event};
use ipc::TokenLogprob;

use crate::ffi::{Backing, DType};
use crate::memory::allocation::{Storage, StorageView};
use crate::stream::Issue;
use crate::{Error, EventPool, Result, lock};

mod logprobs;

pub use logprobs::LogprobLayout;

/// One batch's host words: `words` 64-bit words from word `offset` of `memory`,
/// page-locked for device copies and pageable on the host. Only storage is
/// reused; each batch receives a distinct [`OutputBuffer`] and completion
/// signal.
pub struct OutputStorage {
    memory: Arc<Storage>,
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

    /// Byte offset of the first word in `memory`.
    fn byte_offset(&self) -> usize {
        self.offset * size_of::<i64>()
    }

    fn address(&self) -> u64 {
        self.memory.address() + self.byte_offset() as u64
    }
}

/// Host memory for `words` 64-bit words, pinned for `device`'s copies.
fn words_memory(device: Option<DeviceId>, words: usize) -> Result<Arc<Storage>> {
    let bytes = words.max(1) * size_of::<i64>();
    Ok(match device {
        Some(device) => Storage::pinned(device, bytes)?,
        None => Storage::host(bytes),
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
    /// The token words lent to a producer that writes them itself
    /// ([`Self::lend_words`]), until it states what it wrote
    /// ([`Self::register_lent`]).
    lent: Option<Range<usize>>,
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
            lent: None,
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

    /// Lend `count` bytes at the back of the buffer to a producer that
    /// copies them in itself, on `stream` of `device` in its own order (the
    /// executor staging decoded pixels). Returns a host `uint8 [count]`
    /// view of the lent bytes, pinned for a device lease and readable once
    /// the buffer's fences completed; a CPU reader retains the buffer
    /// ([`Self::retain_reader`]). The lend marks where the buffer's copy
    /// timing begins.
    pub(crate) fn lend_bytes(
        &mut self,
        count: usize,
        device: Option<DeviceId>,
        stream: impl Issue,
        events: &EventPool,
    ) -> Result<StorageView> {
        let offset = self.reserve_bytes(count)?;
        let storage = self.storage()?;
        let memory: Arc<dyn Backing> = Arc::clone(&storage.memory) as Arc<dyn Backing>;
        let view = StorageView::contiguous(
            memory,
            storage.byte_offset() + offset,
            DType::U8,
            &[count as i64],
        )?;
        self.begin_copy(device, stream, events)?;
        Ok(view)
    }

    /// Reserve `count` bytes at the back of the buffer, which bound the
    /// token words lent from its front. A failed copy abandons its batch,
    /// so the range remains occupied until retirement.
    fn reserve_bytes(&mut self, count: usize) -> Result<usize> {
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

    /// Refuse a read outside the words lent and written so far.
    fn token_range(&self, offset: usize, count: usize) -> Result<()> {
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
    /// recycled; lent bytes retain their separate CPU-reader lifetime.
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

    /// Reserve the next `count` token words for device values the lease's owner
    /// copies in itself, on the lease's stream before its seal (a call
    /// predicate's byte, `crate::executor::batch`). The words are zero until a
    /// copy lands, so a copy of fewer bytes than a word leaves the rest zero.
    /// Returns the offset of the first word and its host address. Words are
    /// reserved before any are lent.
    pub(crate) fn reserve_words(&mut self, count: usize) -> Result<(usize, u64)> {
        self.require_open()?;
        if self.lent.is_some() {
            return Err(Error::Invariant(
                "output words are reserved before the rest are lent".into(),
            ));
        }
        let storage = self.storage()?;
        let available = (storage.words * 8 - self.bytes) / 8;
        let offset = self.tokens;
        if !offset
            .checked_add(count)
            .is_some_and(|end| end <= available)
        {
            return Err(Error::Resource(
                "reserved words exceed pinned output capacity",
            ));
        }
        let address = storage.address() + (offset * size_of::<i64>()) as u64;
        // SAFETY: the words lie inside the lease's storage, which no device
        // copy writes before this lease issues one, and no reader reads before
        // the seal's fences.
        unsafe { std::ptr::write_bytes(address as *mut i64, 0, count) };
        self.tokens = offset + count;
        Ok((offset, address))
    }

    /// Lend every token word the buffer has left to a producer that writes
    /// them itself: the executor copies its device values in on the lease's
    /// stream, in its own order, and states afterwards what it wrote
    /// ([`Self::register_lent`]). Returns the offset of the first lent word
    /// and a host `int64` view of the lent words, pinned for a device lease.
    ///
    /// The words count as written until then, so no bytes can be lent
    /// beside them.
    pub fn lend_words(&mut self) -> Result<(usize, StorageView)> {
        self.require_open()?;
        if self.lent.is_some() {
            return Err(Error::Invariant(
                "output words are lent to one producer at a time".into(),
            ));
        }
        let storage = self.storage()?;
        let available = (storage.words * 8 - self.bytes) / 8;
        let offset = self.tokens;
        let memory: Arc<dyn Backing> = Arc::clone(&storage.memory) as Arc<dyn Backing>;
        let view = StorageView::contiguous(
            memory,
            storage.byte_offset() + offset * size_of::<i64>(),
            DType::I64,
            &[(available - offset) as i64],
        )?;
        self.tokens = available;
        self.lent = Some(offset..available);
        Ok((offset, view))
    }

    /// State what the producer of the lent words wrote: the first `written`
    /// of them, and the packed logprob columns among them as `(offset,
    /// count, layout)`, offsets counted from the first lent word. Called
    /// once the lease is sealed and before it is read back; the lent words
    /// past `written` are released from the capture, so readback copies
    /// only what was written.
    ///
    /// Fails without changing the buffer when no words are lent, when the
    /// buffer was read back, or when a count or column lies outside the
    /// written words or disagrees with its layout.
    pub fn register_lent(
        &mut self,
        written: usize,
        columns: Vec<(usize, usize, LogprobLayout)>,
    ) -> Result<()> {
        let lent = self
            .lent
            .clone()
            .ok_or_else(|| Error::Invariant("no output words are lent".into()))?;
        if self.readback.is_some() {
            return Err(Error::Invariant(
                "lent output words are registered before they are read back".into(),
            ));
        }
        if written > lent.len() {
            return Err(Error::Invariant(format!(
                "{written} output words were written into {} lent words",
                lent.len()
            )));
        }
        let end = lent.start + written;
        let mut layouts = Vec::with_capacity(columns.len());
        for (offset, count, layout) in columns {
            let offset = lent.start + offset;
            if !offset.checked_add(count).is_some_and(|last| last <= end) {
                return Err(Error::Invariant(
                    "a logprob column lies outside the written output words".into(),
                ));
            }
            layout.validate(count)?;
            layouts.push(((offset, count), layout));
        }

        self.lent = None;
        self.tokens = end;
        self.logprob_layouts.extend(layouts);
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

/// Output storage of one computation: one host word arena, allocated when the
/// computation is bound, that every batch leases exactly its words from.
///
/// The arena is `blocks` blocks of `block_words` words, the largest lease the
/// computation's own bounds allow. A lease lies within one block, first fit, so
/// while fewer than `blocks` leases are live some block is entirely free and
/// any lease of at most `block_words` fits: `blocks` is the number of the
/// largest batches the computation holds at once, and smaller batches share
/// blocks. A lease's range is reused only after the lease retired: its copies
/// finished, every row was observed or the batch abandoned, and no CPU reader
/// holds a byte capture.
///
/// A batch of more words than a block, such as a readout whose candidate
/// count no admission limit bounds, reads back into storage of its own,
/// allocated for it at its lease and released when it retires: the
/// reservation follows the request.
pub(crate) struct OutputPool {
    /// Released by [`Self::finish_close`].
    arena: Option<Arc<Storage>>,
    /// The device the copies read from; `None` for a computation on the host.
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
    /// (pageable for a computation on the host), allocated now so that no batch
    /// within the computation's bounds allocates host memory.
    pub(crate) fn new(device: Option<DeviceId>, blocks: usize, block_words: usize) -> Result<Self> {
        let blocks = blocks.max(1);
        let words = blocks
            .checked_mul(block_words)
            .ok_or(Error::Resource("computation output storage overflows"))?;
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
            .ok_or(Error::Resource("all computation output storage is leased"))?;

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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{LogprobLayout, OutputBuffer, OutputPool};
    use crate::ffi::DType;
    use crate::memory::allocation::StorageView;
    use crate::{EventPool, Result, lock};

    /// Write `values` into the lent host words `lent`, as the executor
    /// does.
    fn write(lent: &StorageView, values: &[i64]) {
        assert!(values.len() as i64 <= lent.shape()[0]);
        // SAFETY: the lent host words belong to the test's lease and nothing
        // else writes them.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                lent.address() as *mut i64,
                values.len(),
            );
        }
    }

    /// Lent bytes come from the back of the buffer and bound the words
    /// lent from its front: the executor writes both itself.
    #[test]
    fn lent_bytes_fill_from_the_back_and_bound_token_capacity() -> Result<()> {
        let mut pool = OutputPool::new(None, 1, 2)?;
        let lease = pool.lease(1, 2, &[], false)?;
        let mut buffer = lock(&lease);
        let events = EventPool::default();
        let lent = buffer.lend_bytes(5, None, None::<&cuda::Stream>, &events)?;
        assert_eq!(lent.shape(), &[5]);
        assert_eq!(lent.dtype(), DType::U8);
        // Five bytes from the back leave one whole token word.
        let (offset, words) = buffer.lend_words()?;
        assert_eq!((offset, words.shape()), (0, &[1_i64][..]));
        assert!(
            buffer
                .lend_bytes(4, None, None::<&cuda::Stream>, &events)
                .is_err()
        );
        Ok(())
    }

    /// Lease `words` words of `pool` for a one-row batch, with `value`
    /// written in each word.
    fn lease_filled(
        pool: &mut OutputPool,
        words: usize,
        value: i64,
    ) -> Result<Arc<Mutex<OutputBuffer>>> {
        let lease = pool.lease(1, words, &[], false)?;
        {
            let mut buffer = lock(&lease);
            let (_, lent) = buffer.lend_words()?;
            write(&lent, &vec![value; words]);
        }
        Ok(lease)
    }

    /// Complete a leased batch of `words` written words and give them
    /// back: its copies finished and its row was observed.
    fn retire(lease: &Arc<Mutex<OutputBuffer>>, words: usize) -> Result<Vec<i64>> {
        let mut buffer = lock(lease);
        // Host leases are sealed on no stream.
        buffer.seal::<&cuda::Stream>(&[], &EventPool::default())?;
        buffer.register_lent(words, Vec::new())?;
        buffer.copies_finished();
        buffer.readback()?;
        let read = buffer.read_tokens(0, words)?.to_vec();
        buffer.discard(0);
        Ok(read)
    }

    #[test]
    fn a_pool_holds_its_largest_batches_at_once_and_smaller_ones_share_its_storage() -> Result<()> {
        // Storage for two of the computation's largest batches, of eight words
        // each.
        let mut pool = OutputPool::new(None, 2, 8)?;

        // Two of the largest batches at once; a third waits for one to retire.
        let first = lease_filled(&mut pool, 8, 1)?;
        let second = lease_filled(&mut pool, 8, 2)?;
        assert!(pool.lease(1, 1, &[], false).is_err());
        assert_eq!(retire(&first, 8)?, [1; 8]);
        let third = lease_filled(&mut pool, 8, 3)?;
        assert_eq!(retire(&second, 8)?, [2; 8]);
        assert_eq!(retire(&third, 8)?, [3; 8]);

        // Batches take their own words only: sixteen batches of one word fill
        // the storage of the two largest, each keeping its own value.
        let small = (0..16)
            .map(|value| lease_filled(&mut pool, 1, value))
            .collect::<Result<Vec<_>>>()?;
        assert!(pool.lease(1, 1, &[], false).is_err());
        for (value, lease) in small.iter().enumerate() {
            assert_eq!(retire(lease, 1)?, [value as i64]);
        }

        // A small batch still live leaves room for one of the largest.
        let small = lease_filled(&mut pool, 3, 4)?;
        let large = lease_filled(&mut pool, 8, 5)?;
        assert!(pool.lease(1, 8, &[], false).is_err());
        assert_eq!(retire(&small, 3)?, [4; 3]);
        let other = lease_filled(&mut pool, 8, 6)?;
        assert_eq!(retire(&large, 8)?, [5; 8]);
        assert_eq!(retire(&other, 8)?, [6; 8]);
        Ok(())
    }

    /// A producer that writes the lent words itself: they are reserved
    /// whole until it states what it wrote, and readback then holds exactly
    /// those words and decodes the logprob column among them.
    #[test]
    fn lent_words_read_back_as_far_as_their_producer_wrote() -> Result<()> {
        let events = EventPool::default();
        let mut pool = OutputPool::new(None, 1, 16)?;
        let lease = pool.lease(2, 10, &[], false)?;
        let mut buffer = lock(&lease);
        let (offset, lent) = buffer.lend_words()?;
        assert_eq!((offset, lent.shape()), (0, &[10_i64][..]));
        assert!(buffer.lend_words().is_err());
        assert!(
            buffer
                .lend_bytes(1, None, None::<&cuda::Stream>, &events)
                .is_err()
        );

        // One sampled row (token 42, score 0.5, rank 1) with one top token
        // (token 3, score 0.25, rank 2): a six-word column at lent word 1.
        let words = [
            9_i64,
            42,
            i64::from(0.5_f32.to_bits()),
            1,
            3,
            i64::from(0.25_f32.to_bits()),
            2,
        ];
        write(&lent, &words);
        buffer.seal::<&cuda::Stream>(&[], &events)?;
        let layout = || LogprobLayout {
            rows: vec![0],
            counts: vec![1],
        };
        // Columns outside what was written, or of another length than their
        // layout, are refused and leave the lent words registered as lent.
        assert!(buffer.register_lent(11, vec![]).is_err());
        assert!(buffer.register_lent(6, vec![(1, 6, layout())]).is_err());
        assert!(buffer.register_lent(7, vec![(1, 5, layout())]).is_err());
        buffer.register_lent(7, vec![(1, 6, layout())])?;
        assert!(buffer.register_lent(7, vec![]).is_err());

        buffer.copies_finished();
        buffer.readback()?;
        assert_eq!(buffer.read_tokens(0, 7)?, &words);
        assert!(buffer.read_tokens(0, 8).is_err());
        let values = buffer.logprob_values((1, 6, 0))?;
        assert_eq!(
            values
                .iter()
                .map(|entry| (entry.token_id, entry.logprob, entry.rank))
                .collect::<Vec<_>>(),
            [(42, 0.5, 1), (3, 0.25, 2)]
        );
        Ok(())
    }

    /// A batch of more words than a block, as a readout of more candidates than
    /// the computation's bounds state, reads back into storage of its own: it
    /// holds every word, and both blocks stay free for the largest batches.
    #[test]
    fn a_batch_larger_than_a_block_reads_back_into_storage_of_its_own() -> Result<()> {
        let mut pool = OutputPool::new(None, 2, 8)?;
        let largest = lease_filled(&mut pool, 8, 1)?;
        let beyond = lease_filled(&mut pool, 9, 2)?;
        let other = lease_filled(&mut pool, 8, 3)?;
        assert!(pool.lease(1, 1, &[], false).is_err());
        assert_eq!(retire(&beyond, 9)?, [2; 9]);
        assert_eq!(retire(&largest, 8)?, [1; 8]);
        assert_eq!(retire(&other, 8)?, [3; 8]);

        // Its storage goes with it: the pool again holds two of the largest.
        assert_eq!(pool.reap(), 3);
        let first = lease_filled(&mut pool, 8, 4)?;
        let second = lease_filled(&mut pool, 8, 5)?;
        assert_eq!(retire(&first, 8)?, [4; 8]);
        assert_eq!(retire(&second, 8)?, [5; 8]);
        Ok(())
    }
}
