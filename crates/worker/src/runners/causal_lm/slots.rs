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
//! | `context` | U32 `[slots + 1, 2]` | the two token ids before the slot's next row, the earlier first |
//! | `accepted` | U32 `[slots + 1]` | how many tokens of the slot's last step stand (`nn::VerifySlots::num_accepted`) |
//!
//! The last two are held only where a forward reads them: `context` for a model whose layers
//! read token ids (`nn::family::TextModel::reads_token_ids`), `accepted` for a runner that
//! speculates over a model whose state keeps a slot per verified token
//! (`nn::family::StateField::stepwise`).
//!
//! The tables a forward gathers by a row's slot hold one row more than the request slots: the
//! **padding row**, which a row without a slot gathers. It stays as the state starts it (zero,
//! and one in `accepted`), because a commit is given the tables' request rows alone
//! ([`Slots::committed`]) and a forward's writes skip a row on no slot.
//!
//! ## Writes
//!
//! The forwards read the tables and the commit kernel advances them. The host writes them at
//! three moments, each as a copy on the runner's stream, ordered before the replay that reads
//! the row:
//!
//! - at construction every table is zeroed, and every entry of `accepted` set to one;
//! - a batch that starts a request on a slot ([`Slots::start`]) sets the slot's counts to the
//!   request's own history, its bias row when the request states a bias or an allowed set,
//!   its context to the two ids before the request's first token of the batch, its accepted
//!   count to one, and, when the request's first step here is a decode, the state the engine
//!   that sampled its last token committed;
//! - a batch that restores a slot's recurrent state from a cached prefix sets the slot's
//!   accepted count to one ([`Slots::restore`]): the restore writes the slot's home row, which
//!   is what a count of one reads;
//! - a batch whose rows ban tokens copies its masks to the head of `bans`
//!   ([`Slots::ban`]).
//!
//! The copies read page-locked memory this module keeps until the step's stream work has
//! completed: the runner brackets a step's writes with [`Slots::begin`] and
//! [`Slots::settled`]. The forwards advance `context` and `accepted` themselves, as the commit
//! advances the other tables (`super` module documentation, the forward).
//!
//! ## Without a device
//!
//! A runner over graphs that hold no device has no stream: the tables are meta tensors, a
//! shape and an element type with no memory, and every write validates its rows and copies
//! nothing.

use std::sync::Arc;

use ipc::batch::{Columns, NewSlot, StateCopy, StateDirection};
use ops::sampler::ban_words;
use platform::{PinnedBuffer, Stream};
use tensor::{DType, Shape, Tensor};

use super::columns::slot_row;
use crate::memory::allocation::Storage;
use crate::sampler::Sampler;
use crate::{Error, Result};

/// The alignment of a table in the slot memory.
const ALIGNMENT: u64 = backend::allocator::ALIGNMENT;

/// The least bytes of page-locked memory the write buffer grows by: a step's small writes
/// (slot starts, a few ban masks) fit one such block.
const WRITE_BLOCK: usize = 1 << 16;

/// The tables of the slot memory, in the order they lie in it. Every runner holds the first
/// [`HELD`]; the last two are held where a forward reads them (module documentation).
const TABLES: usize = 10;
const HELD: usize = 8;
const COUNTS: usize = 0;
const BIAS: usize = 1;
const FUTURE: usize = 2;
const PREDICATES: usize = 3;
const LOGICAL: usize = 4;
const SAMPLING: usize = 5;
const VERIFIED: usize = 6;
const BANS: usize = 7;
const CONTEXT: usize = 8;
const ACCEPTED: usize = 9;

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
    /// Each table every runner holds, in the order of the table constants.
    tables: [Tensor; HELD],
    /// The context table, for a model whose layers read token ids.
    context: Option<Tensor>,
    /// The accepted counts, for a runner whose forwards lend verify slots.
    accepted: Option<Tensor>,
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

/// The shape and the element type of each table. `context` and `accepted` state whether the
/// runner holds the context table and the accepted counts; a table it does not hold has no row.
fn shapes(
    slots: u32,
    vocabulary: u32,
    rows: u32,
    width: u32,
    context: bool,
    accepted: bool,
) -> [(Shape, DType); TABLES] {
    let (slots, vocabulary) = (slots as usize, vocabulary as usize);
    let words = ban_words(vocabulary as u32) as usize;
    let held = |held: bool| if held { slots + 1 } else { 0 };

    [
        ((slots, vocabulary).into(), DType::I32),
        ((slots, vocabulary).into(), DType::F32),
        ((slots + 1, width as usize).into(), DType::U32),
        (slots.into(), DType::U8),
        ((slots + 1).into(), DType::I32),
        ((slots + 1).into(), DType::I32),
        ((slots + 1).into(), DType::I32),
        ((rows as usize, words).into(), DType::U32),
        ((held(context), 2).into(), DType::U32),
        (held(accepted).into(), DType::U32),
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
    /// tokens a slot, with the context table when `context` and the accepted counts when
    /// `accepted`.
    pub(crate) fn bytes(
        slots: u32,
        vocabulary: u32,
        rows: u32,
        width: u32,
        context: bool,
        accepted: bool,
    ) -> u64 {
        offsets(&shapes(slots, vocabulary, rows, width, context, accepted)).1
    }

    /// The state of `slots` request slots as a runner starts it: every table zeroed and every
    /// accepted count one. The context table is held when `context`, for a model whose layers
    /// read token ids, and the accepted counts when `accepted`, for a runner whose forwards
    /// lend verify slots.
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
        context: bool,
        accepted: bool,
    ) -> Result<Self> {
        if slots == 0 || vocabulary == 0 || rows == 0 || width == 0 {
            return Err(Error::Invalid(format!(
                "slot state of {slots} slots over {vocabulary} tokens, for {rows} rows of \
                 {width} next input tokens, holds nothing"
            )));
        }

        let shapes = shapes(slots, vocabulary, rows, width, context, accepted);
        let (offsets, bytes) = offsets(&shapes);

        let device = match stream {
            Some(stream) => {
                let bytes = usize::try_from(bytes)
                    .map_err(|_| Error::Invalid("the slot state is larger than memory".into()))?;
                let storage = Storage::linear(stream.device(), bytes)?;

                // A slot has seen no token and holds no bias before its first request, and a
                // fresh allocation holds what its pages held last: the zeros are written on the
                // stream, ahead of the first forward that reads a table. A slot whose previous
                // step was of no decode phase has one token of it standing, and so has the
                // padding row.
                //
                // SAFETY: the ranges are the whole allocation and the accepted counts inside
                // it, which this value keeps for as long as the tensors over it exist.
                unsafe { platform::memset_d8(storage.address(), 0, bytes, &stream) }?;
                if accepted {
                    let address = storage.address() + offsets[ACCEPTED];
                    // SAFETY: as above.
                    unsafe { platform::memset_d32(address, 1, slots as usize + 1, &stream) }?;
                }

                Some(Device {
                    stream,
                    storage,
                    offsets,
                    blocks: Vec::new(),
                    used: 0,
                    unsettled: false,
                })
            }
            None => None,
        };

        // Each table's tensor over the memory, or a meta tensor for graphs that hold no device.
        let table = |index: usize| -> Result<Tensor> {
            let (shape, dtype) = shapes[index].clone();
            match &device {
                // SAFETY: this value owns the storage and outlives the tensors' work: the
                // runner that holds it holds the graphs captured over them, and only the copies
                // issued here and the kernels of those graphs write the tables.
                Some(device) => unsafe {
                    device
                        .storage
                        .external(offsets[index] as usize, shape, dtype)
                },
                // SAFETY: a tensor over the meta device has no memory: its address is never
                // read or written.
                None => Ok(unsafe { backend::external(0, shape, dtype, &backend::meta()) }?),
            }
        };

        let mut held = Vec::with_capacity(HELD);
        for index in 0..HELD {
            held.push(table(index)?);
        }
        let context = context.then(|| table(CONTEXT)).transpose()?;
        let accepted = accepted.then(|| table(ACCEPTED)).transpose()?;

        Ok(Self {
            slots,
            vocabulary,
            rows,
            width,
            tables: into_tables(held)?,
            context,
            accepted,
            device,
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

    /// The context table, `[slots + 1, 2]` U32, the padding row included: row `s - 1` holds the
    /// two token ids before the next row of request slot `s`, the earlier first. `None` for a
    /// runner whose model reads no token ids.
    pub(crate) const fn context(&self) -> Option<&Tensor> {
        self.context.as_ref()
    }

    /// The accepted counts, `[slots + 1]` U32, the padding row included: entry `s - 1` holds
    /// how many tokens of request slot `s`'s last step stand, in `1..=Q`. `None` for a runner
    /// whose forwards lend no verify slots.
    pub(crate) const fn accepted(&self) -> Option<&Tensor> {
        self.accepted.as_ref()
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

    /// Starts the sampling state of the requests a batch starts on their slots, `new_slots`
    /// (`ipc::batch::Batch::new_slots`), ahead of the step that reads it; `columns` are the
    /// batch's.
    ///
    /// - Each new slot's counts become its request's own history (`Sampler::count_row`): the
    ///   slot's previous request counted its tokens there, and the request may have generated
    ///   tokens before it reached this runner.
    /// - A request that states a bias or an allowed set has its bias row written
    ///   (`Sampler::bias_row`). A request that states neither leaves the row as it is: none of
    ///   its rows is flagged to read it (`Columns::biased`).
    /// - Where the runner holds them, the slot's context becomes the two ids before the
    ///   request's first token of the batch (`NewSlot::context`), and its accepted count one:
    ///   the request's state starts in its home row.
    /// - In a decode batch (`decode`), a new slot's request continues from a token another
    ///   engine sampled, so the slot's commit state starts as that engine's commit left it:
    ///   the row's input token as the next input token, the token's position as the logical
    ///   position, the row's draw position as the sampling position, the keys before the token
    ///   as the verified length, and the predicate set. A prefill's commit states the slot's
    ///   state itself.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a slot the runner holds no state for, a new slot of a decode
    /// batch that no row of the batch runs on, or a row whose values no slot state holds; the
    /// driver's refusal of a copy.
    pub(crate) fn start(
        &mut self,
        sampler: &Sampler,
        columns: &Columns,
        new_slots: &[NewSlot],
        decode: bool,
    ) -> Result<()> {
        let row_bytes = u64::from(self.vocabulary) * 4;

        for new in new_slots {
            let own = slot_row(new.slot, self.slots)?;

            // An empty history is a zeroed row, which the device fills without a copy. The
            // sampler states a row's values for the row of its tables the slot owns.
            if new.history.is_empty() {
                self.zero(COUNTS, u64::from(own) * row_bytes, self.vocabulary)?;
            } else {
                let row = sampler.count_row(own, &new.history)?;
                self.write(COUNTS, u64::from(own) * row_bytes, &row)?;
            }

            if !new.bias.is_empty() || new.allowed.is_some() {
                let row = sampler.bias_row(own, &new.bias, new.allowed.as_deref())?;
                self.write(BIAS, u64::from(own) * row_bytes, &row)?;
            }

            if self.context.is_some() {
                self.write(CONTEXT, u64::from(own) * 8, &new.context)?;
            }
            if self.accepted.is_some() {
                self.write(ACCEPTED, u64::from(own) * 4, &[1_u32])?;
            }

            if decode {
                self.start_decode(new.slot, columns)?;
            }
        }
        Ok(())
    }

    /// Sets the accepted count of every slot whose recurrent state `copies` restore before the
    /// step (`ipc::batch::Batch`'s state copies) to one: a restore writes the slot's home row,
    /// which is the row a count of one reads. A runner that holds no accepted counts writes
    /// nothing.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a slot the runner holds no state for; the driver's refusal of a
    /// copy.
    pub(crate) fn restore(&mut self, copies: &[StateCopy]) -> Result<()> {
        if self.accepted.is_none() {
            return Ok(());
        }

        for copy in copies {
            if copy.direction == StateDirection::Restore {
                let own = slot_row(copy.slot, self.slots)?;
                self.write(ACCEPTED, u64::from(own) * 4, &[1_u32])?;
            }
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
    fn start_decode(&mut self, slot: u32, columns: &Columns) -> Result<()> {
        let own = u64::from(slot_row(slot, self.slots)?);
        let rows = &columns.rows;
        let row = columns
            .slots
            .iter()
            .position(|held| u32::try_from(*held).ok() == Some(slot))
            .ok_or_else(|| {
                Error::Invalid(format!("new slot {slot} has no row in its decode batch"))
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

/// `tables` as the array of the tables every runner holds.
fn into_tables(tables: Vec<Tensor>) -> Result<[Tensor; HELD]> {
    tables
        .try_into()
        .map_err(|_| Error::Invariant("the slot state holds another number of tables".into()))
}
