//! The request slots' device state: what a step of a request reads on the device without a
//! host round trip, and what the commit that ends the step advances.
//!
//! One row per request slot of each table, in memory this module owns. Request slot `s`, as
//! the scheduler numbers it from one, owns row `s - 1` (`columns::slot_row`):
//!
//! | Table | Type and extent | Holds |
//! | --- | --- | --- |
//! | `counts` | I32 `[slots, vocabulary]` | how often the slot's request has seen each token |
//! | `bias` | F32 `[slots, vocabulary]` | the request's logit bias, `-inf` outside its allowed set |
//! | `future` | U32 `[slots + 1, width]` | column 0: the token the slot's next decode step reads |
//! | `predicates` | U8 `[slots]` | whether the slot's request continues |
//! | `logical`, `sampling`, `verified` | I32 `[slots + 1]` | the slot's logical and sampling positions and its verified cache length |
//! | `bans` | U32 `[rows, ceil(vocabulary / 32)]` | the ban masks of the step being issued |
//!
//! The tables a forward gathers by a row's slot hold one row more than the request slots: the
//! **padding row**, which a row without a slot gathers. It stays zero, because a commit is
//! given the tables' request rows alone ([`Slots::committed`]).
//!
//! ## Writes
//!
//! The forwards read the tables and the commit kernel advances them. The host writes them at
//! three moments, each as a copy on the runner's stream, ordered before the replay that reads
//! the row:
//!
//! - at construction every table is zeroed;
//! - a batch that starts a request on a slot ([`Slots::start`]) sets the slot's counts to the
//!   request's own history, its bias row when the batch installs one, and, when the request's
//!   first step here is a decode, the state the engine that sampled its last token committed;
//! - a batch whose rows ban tokens copies its masks to the head of `bans`
//!   ([`Slots::ban`]).
//!
//! The copies read page-locked memory this module keeps until the step's stream work has
//! completed: the runner brackets a step's writes with [`Slots::begin`] and
//! [`Slots::settled`].
//!
//! ## Without a device
//!
//! A runner over graphs that hold no device has no stream: the tables are meta tensors, a
//! shape and an element type with no memory, and every write validates its rows and copies
//! nothing.

use std::sync::Arc;

use ipc::execution::{BatchColumns, FreshSlot, SlotBias};
use ops::sampler::ban_words;
use platform::{PinnedBuffer, Stream};
use uniserve_tensor::{DType, Shape, Tensor};

use super::columns::slot_row;
use crate::memory::allocation::Storage;
use crate::sampler::Sampler;
use crate::{Error, Result};

/// The alignment of a table in the slot memory.
const ALIGNMENT: u64 = backend::allocator::ALIGNMENT;

/// The least bytes of page-locked memory the write buffer grows by: a step's small writes
/// (slot starts, a few ban masks) fit one such block.
const WRITE_BLOCK: usize = 1 << 16;

/// The tables of the slot memory, in the order they lie in it.
const TABLES: usize = 8;
const COUNTS: usize = 0;
const BIAS: usize = 1;
const FUTURE: usize = 2;
const PREDICATES: usize = 3;
const LOGICAL: usize = 4;
const SAMPLING: usize = 5;
const VERIFIED: usize = 6;
const BANS: usize = 7;

/// The request slots' device state (module documentation).
pub(crate) struct Slots {
    /// Request slots.
    slots: u32,
    /// The width of a logit row.
    vocabulary: u32,
    /// The rows of the largest bucket: the masks `bans` holds.
    rows: u32,
    /// The next input tokens a slot holds: one, and one per draft of a runner that speculates.
    width: u32,
    /// Each table's tensor, in the order of the table constants.
    tables: [Tensor; TABLES],
    /// The device side; `None` for a runner whose graphs hold no device.
    device: Option<Device>,
}

/// The memory and the stream of slot state on a GPU.
struct Device {
    /// The stream the runner's graphs replay on: every write is ordered on it.
    stream: Stream,
    /// The tables' memory.
    storage: Arc<Storage>,
    /// Each table's offset in `storage`.
    offsets: [u64; TABLES],
    /// The page-locked blocks the step's copies read. A step that needs more than the blocks
    /// hold adds one; the blocks are folded into one when the next step begins.
    blocks: Vec<PinnedBuffer>,
    /// The bytes of the last block the step has written.
    used: usize,
    /// Whether copies were issued that the runner has not yet seen complete.
    unsettled: bool,
}

/// The shape and the element type of each table.
fn shapes(slots: u32, vocabulary: u32, rows: u32, width: u32) -> [(Shape, DType); TABLES] {
    let (slots, vocabulary) = (slots as usize, vocabulary as usize);
    let words = ban_words(vocabulary as u32) as usize;

    [
        ((slots, vocabulary).into(), DType::I32),
        ((slots, vocabulary).into(), DType::F32),
        ((slots + 1, width as usize).into(), DType::U32),
        (slots.into(), DType::U8),
        ((slots + 1).into(), DType::I32),
        ((slots + 1).into(), DType::I32),
        ((slots + 1).into(), DType::I32),
        ((rows as usize, words).into(), DType::U32),
    ]
}

/// Each table's offset in the slot memory, and the memory's bytes.
fn offsets(shapes: &[(Shape, DType); TABLES]) -> ([u64; TABLES], u64) {
    let mut offsets = [0; TABLES];
    let mut bytes = 0_u64;

    for (offset, (shape, dtype)) in offsets.iter_mut().zip(shapes) {
        *offset = bytes;

        let element = if *dtype == DType::U8 { 1 } else { 4 };
        bytes += (shape.elem_count() as u64 * element).next_multiple_of(ALIGNMENT);
    }
    (offsets, bytes)
}

impl Slots {
    /// The bytes of device memory the state of `slots` request slots takes, over a
    /// `vocabulary`-wide logit row, for buckets of at most `rows` rows and `width` next input
    /// tokens a slot.
    pub(crate) fn bytes(slots: u32, vocabulary: u32, rows: u32, width: u32) -> u64 {
        offsets(&shapes(slots, vocabulary, rows, width)).1
    }

    /// The zeroed state of `slots` request slots.
    ///
    /// `stream` is the stream the runner's graphs replay on, and `None` for graphs that hold
    /// no device: the tables are then meta tensors.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for no slot, an empty vocabulary or no next input token; the
    /// driver's refusal of the allocation or of the fill.
    pub(crate) fn new(
        stream: Option<Stream>,
        slots: u32,
        vocabulary: u32,
        rows: u32,
        width: u32,
    ) -> Result<Self> {
        if slots == 0 || vocabulary == 0 || rows == 0 || width == 0 {
            return Err(Error::Invalid(format!(
                "slot state of {slots} slots over {vocabulary} tokens, for {rows} rows of \
                 {width} next input tokens, holds nothing"
            )));
        }

        let shapes = shapes(slots, vocabulary, rows, width);
        let (offsets, bytes) = offsets(&shapes);

        let Some(stream) = stream else {
            let meta = backend::meta();
            let mut tables = Vec::with_capacity(TABLES);
            for (shape, dtype) in shapes {
                // SAFETY: a tensor over the meta device has no memory: its address is never
                // read or written.
                tables.push(unsafe { backend::external(0, shape, dtype, &meta) }?);
            }

            return Ok(Self {
                slots,
                vocabulary,
                rows,
                width,
                tables: into_tables(tables)?,
                device: None,
            });
        };

        let bytes = usize::try_from(bytes)
            .map_err(|_| Error::Invalid("the slot state is larger than memory".into()))?;
        let storage = Storage::linear(stream.device(), bytes)?;

        // A slot has seen no token and holds no bias before its first request, and a fresh
        // allocation holds what its pages held last: the zeros are written on the stream,
        // ahead of the first forward that reads a table.
        //
        // SAFETY: the range is the whole allocation, which this value keeps for as long as the
        // tensors over it exist.
        unsafe { platform::memset_d8(storage.address(), 0, bytes, &stream) }?;

        let mut tables = Vec::with_capacity(TABLES);
        for ((shape, dtype), offset) in shapes.into_iter().zip(offsets) {
            // SAFETY: this value owns the storage and outlives the tensors' work: the runner
            // that holds it holds the graphs captured over them, and only the copies issued
            // here and the kernels of those graphs write the tables.
            tables.push(unsafe { storage.external(offset as usize, shape, dtype) }?);
        }

        Ok(Self {
            slots,
            vocabulary,
            rows,
            width,
            tables: into_tables(tables)?,
            device: Some(Device {
                stream,
                storage,
                offsets,
                blocks: Vec::new(),
                used: 0,
                unsettled: false,
            }),
        })
    }

    /// The count table, `[slots, vocabulary]` I32.
    pub(crate) const fn counts(&self) -> &Tensor {
        &self.tables[COUNTS]
    }

    /// The bias table, `[slots, vocabulary]` F32.
    pub(crate) const fn bias(&self) -> &Tensor {
        &self.tables[BIAS]
    }

    /// The next input tokens, `[slots + 1, width]` U32, the padding row included: what a
    /// decode row gathers its input token from.
    pub(crate) const fn future(&self) -> &Tensor {
        &self.tables[FUTURE]
    }

    /// The logical positions, `[slots + 1]` I32, the padding row included.
    pub(crate) const fn logical(&self) -> &Tensor {
        &self.tables[LOGICAL]
    }

    /// The sampling positions, `[slots + 1]` I32, the padding row included.
    pub(crate) const fn sampling(&self) -> &Tensor {
        &self.tables[SAMPLING]
    }

    /// The ban masks, `[rows, ceil(vocabulary / 32)]` U32.
    pub(crate) const fn bans(&self) -> &Tensor {
        &self.tables[BANS]
    }

    /// The state a commit advances, as `Sampler::commit_` takes it: `[future, predicates,
    /// logical, sampling, verified]` over the request slots alone, so no commit writes the
    /// padding row.
    ///
    /// # Errors
    ///
    /// What the tensor library refuses of a view.
    pub(crate) fn committed(&self) -> Result<[Tensor; 5]> {
        let slots = self.slots as usize;
        let requests = |table: usize| self.tables[table].narrow(0, 0, slots);

        Ok([
            requests(FUTURE)?,
            self.tables[PREDICATES].clone(),
            requests(LOGICAL)?,
            requests(SAMPLING)?,
            requests(VERIFIED)?,
        ])
    }

    /// Begins a step's writes: the copies of the step before have completed, so the memory
    /// they read is written again.
    ///
    /// # Errors
    ///
    /// When the stream cannot be waited for, or the write buffer cannot be allocated.
    pub(crate) fn begin(&mut self) -> Result<()> {
        let Some(device) = &mut self.device else {
            return Ok(());
        };

        // A step that failed before its stream event leaves copies the runner never saw
        // complete; they are waited for before the memory they read is written.
        if device.unsettled {
            device.stream.synchronize()?;
            device.unsettled = false;
        }

        if device.blocks.len() > 1 {
            let bytes = device.blocks.iter().map(PinnedBuffer::len).sum();
            device.blocks = vec![PinnedBuffer::new(device.stream.device(), bytes)?];
        }
        device.used = 0;
        Ok(())
    }

    /// States that the stream has passed every copy issued since [`Slots::begin`].
    pub(crate) fn settled(&mut self) {
        if let Some(device) = &mut self.device {
            device.unsettled = false;
        }
    }

    /// Starts the sampling state of the requests `columns` starts on their slots, ahead of the
    /// step that reads it.
    ///
    /// - Each fresh slot's counts become its request's own history (`Sampler::count_row`):
    ///   the slot's previous request counted its tokens there, and the request may have
    ///   generated tokens before it reached this runner.
    /// - Each bias row the batch installs is written (`Sampler::bias_row`).
    /// - In a decode batch (`decode`), a fresh slot's request continues from a token another
    ///   engine sampled, so the slot's commit state starts as that engine's commit left it:
    ///   the row's input token as the next input token, the token's position as the logical
    ///   position, the row's draw position as the sampling position, the keys before the token
    ///   as the verified length, and the predicate set. A prefill's commit states the slot's
    ///   state itself.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a slot the runner holds no state for, a fresh slot of a decode
    /// batch that no row of the batch runs on, or a row whose values no slot state holds; the
    /// driver's refusal of a copy.
    pub(crate) fn start(
        &mut self,
        sampler: &Sampler,
        columns: &BatchColumns,
        decode: bool,
    ) -> Result<()> {
        let row_bytes = u64::from(self.vocabulary) * 4;

        for fresh in &columns.fresh_slots {
            let own = slot_row(fresh.slot, self.slots)?;

            // An empty history is a zeroed row, which the device fills without a copy. The
            // sampler states a row's values for the row of its tables the slot owns.
            if fresh.history.is_empty() {
                self.zero(COUNTS, u64::from(own) * row_bytes, self.vocabulary)?;
            } else {
                let row = sampler.count_row(&FreshSlot {
                    slot: own,
                    history: fresh.history.clone(),
                })?;
                self.write(COUNTS, u64::from(own) * row_bytes, &row)?;
            }

            if decode {
                self.start_decode(fresh.slot, columns)?;
            }
        }

        for install in &columns.adjustments.installs {
            let own = slot_row(install.slot, self.slots)?;
            let row = sampler.bias_row(&SlotBias {
                slot: own,
                ..install.clone()
            })?;
            self.write(BIAS, u64::from(own) * row_bytes, &row)?;
        }
        Ok(())
    }

    /// Copies `masks`, the ban masks of the step being issued in row order
    /// (`sampler::adjustments`), to the head of the mask table. A step that bans nothing
    /// copies nothing: no row then names a mask.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for more masks than the table holds; the driver's refusal.
    pub(crate) fn ban(&mut self, masks: &[u32]) -> Result<()> {
        let words = ban_words(self.vocabulary) as usize;
        if masks.len() > self.rows as usize * words {
            return Err(Error::Invalid(format!(
                "{} ban masks for a runner of {} rows",
                masks.len() / words,
                self.rows
            )));
        }

        self.write(BANS, 0, masks)
    }

    /// Starts request slot `slot`'s commit state from its row of the decode batch `columns`.
    fn start_decode(&mut self, slot: u32, columns: &BatchColumns) -> Result<()> {
        let own = u64::from(slot_row(slot, self.slots)?);
        let rows = &columns.rows;
        let row = columns
            .slots
            .iter()
            .position(|held| u32::try_from(*held).ok() == Some(slot))
            .ok_or_else(|| {
                Error::Invalid(format!("fresh slot {slot} has no row in its decode batch"))
            })?;

        // A decode row is one token: its query offset is the token's index.
        let stated = || -> Option<(u32, i32, i32, i32)> {
            let token = usize::try_from(*rows.query_offsets.get(row)?).ok()?;
            let id = u32::try_from(*columns.input_ids.get(token)?).ok()?;
            let sampling = i32::try_from(columns.draws.get(row)?.position).ok()?;

            Some((
                id,
                *rows.positions.get(token)?,
                sampling,
                rows.seq_lens.get(row)?.checked_sub(1)?,
            ))
        };
        let (id, logical, sampling, verified) = stated().ok_or_else(|| {
            Error::Invalid(format!(
                "row {row} of the decode batch states no input token, position, draw or length \
                 that slot {slot} can start from"
            ))
        })?;

        self.write(FUTURE, own * u64::from(self.width) * 4, &[id])?;
        self.write(LOGICAL, own * 4, &[logical])?;
        self.write(SAMPLING, own * 4, &[sampling])?;
        self.write(VERIFIED, own * 4, &[verified])?;
        self.write(PREDICATES, own, &[1_u8])
    }

    /// Copies `values` to byte `offset` of `table`, on the stream.
    fn write<T: bytemuck::Pod>(&mut self, table: usize, offset: u64, values: &[T]) -> Result<()> {
        let Some(device) = &mut self.device else {
            return Ok(());
        };
        let source: &[u8] = bytemuck::cast_slice(values);
        if source.is_empty() {
            return Ok(());
        }

        // The bytes take the next free bytes of the last block, or a block of their own.
        let free = device
            .blocks
            .last()
            .map_or(0, |block| block.len() - device.used);
        if source.len() > free {
            let bytes = source.len().max(WRITE_BLOCK);
            device
                .blocks
                .push(PinnedBuffer::new(device.stream.device(), bytes)?);
            device.used = 0;
        }
        let Some(block) = device.blocks.last() else {
            return Err(Error::State("the slot state holds no write buffer"));
        };
        let at = device.used;
        device.used += source.len();

        // SAFETY: `at + source.len()` bytes lie inside the block, as reserved above, and no
        // copy in flight reads them: the bytes written since `begin` are disjoint, and the
        // copies of earlier steps completed before `begin` returned.
        unsafe {
            std::ptr::copy_nonoverlapping(source.as_ptr(), block.as_ptr().add(at), source.len());
        }

        let destination = device.storage.address() + device.offsets[table] + offset;
        device.unsettled = true;

        // SAFETY: the destination is `source.len()` bytes of a table of this value's storage,
        // which the callers address by a slot or a row the table holds; the source stays as
        // written until the step's copies have completed (`begin`).
        unsafe {
            platform::memcpy_async(
                destination,
                block.address() + at as u64,
                source.len(),
                &device.stream,
            )
        }?;
        Ok(())
    }

    /// Zeroes `words` 32-bit words at byte `offset` of `table`, on the stream.
    fn zero(&mut self, table: usize, offset: u64, words: u32) -> Result<()> {
        let Some(device) = &mut self.device else {
            return Ok(());
        };
        let destination = device.storage.address() + device.offsets[table] + offset;
        device.unsettled = true;

        // SAFETY: the destination is a row of a table of this value's storage, which the
        // caller addresses by a slot the table holds.
        unsafe { platform::memset_d32(destination, 0, words as usize, &device.stream) }?;
        Ok(())
    }
}

/// `tables` as the array the state holds.
fn into_tables(tables: Vec<Tensor>) -> Result<[Tensor; TABLES]> {
    tables
        .try_into()
        .map_err(|_| Error::Invariant("the slot state holds another number of tables".into()))
}
