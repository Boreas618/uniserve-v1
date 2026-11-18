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
//! | `context` | U32 `[slots + 1, width]` | the `width` token ids before the slot's next row, the oldest first (`nn::family::TokenHistory`) |
//! | `accepted` | U32 `[slots + 1]` | how many tokens of the slot's last step stand (`nn::VerifySlots::num_accepted`) |
//!
//! The last two are held only where a forward reads them: `context` for a model whose layers
//! read token ids (`nn::family::TextModel::token_history`), `accepted` for a runner that
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
//!   its context to the ids before the request's first token of the batch, the model's fill
//!   id standing for those before the request's first token, its accepted
//!   count to one, and, when the request's first step here is a decode, the state the engine
//!   that sampled its last token committed;
//! - a batch that restores a slot's recurrent state from a cached prefix sets the slot's
//!   accepted count to one ([`Slots::restore`]): the restore writes the slot's home row, which
//!   is what a count of one reads;
//! - a batch whose rows ban tokens copies its masks to the head of `bans`
//!   ([`Slots::ban`]).
//!
//! A shape of the runner that lays the recurrent state out otherwise (`Runner::reshape`, a
//! kernel generation of other drafts) holds accepted counts of its own, every one set to one, when
//! its forwards lend verify slots ([`Slots::restated`]): its switch moves each slot's state into
//! its home row, which a count of one reads, from the counts the serving shape left
//! ([`Slots::standing`]), and a switch back sets the kept shape's counts to one again
//! ([`Slots::accept_one`]).
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

use nn::family::TokenHistory;

use crate::batch::{Columns, NewSlot, StateCopy, StateDirection};
use ops::allocator::ALIGNMENT;
use ops::sampler::ban_words;
use platform::{PinnedBuffer, Stream};
use strum::{EnumCount, EnumIter, IntoEnumIterator as _};
use tensor::{DType, Shape, Tensor};

use super::columns::slot_row;
use crate::memory::allocation::Storage;
use crate::memory::state::Standing;
use crate::runner::sampler::Sampler;
use crate::worker::{Error, Result};

/// The least bytes of page-locked memory the write buffer grows by: a step's small writes
/// (slot starts, a few ban masks) fit one such block.
const WRITE_BLOCK: usize = 1 << 16;

/// One table of the slot memory, declared in the order the tables lie in it: [`Table::iter`]
/// walks them in that order, and a table's discriminant is its place among them. Every runner
/// holds the tables up to the ban masks ([`Held`]); the context table and the accepted counts
/// are held where a forward reads them (module documentation).
#[derive(Clone, Copy, PartialEq, Eq, Debug, EnumCount, EnumIter)]
enum Table {
    Counts,
    Bias,
    Future,
    Predicates,
    Logical,
    Sampling,
    Verified,
    Bans,
    Context,
    Accepted,
}

impl Table {
    /// The shape and the element type of the table, for `slots` request slots over a
    /// `vocabulary`-wide logit row, buckets of at most `rows` rows and `width` next input tokens
    /// a slot. `context` states the token history of a model whose layers read token ids, and
    /// `accepted` whether the runner holds the accepted counts; a table it does not hold has no
    /// row.
    fn shape(
        self,
        (slots, vocabulary, rows, width): (usize, usize, usize, usize),
        context: Option<TokenHistory>,
        accepted: bool,
    ) -> (Shape, DType) {
        let held = |held: bool| if held { slots + 1 } else { 0 };
        match self {
            Self::Counts => ((slots, vocabulary).into(), DType::I32),
            Self::Bias => ((slots, vocabulary).into(), DType::F32),
            Self::Future => ((slots + 1, width).into(), DType::U32),
            Self::Predicates => (slots.into(), DType::U8),
            Self::Logical | Self::Sampling | Self::Verified => ((slots + 1).into(), DType::I32),
            Self::Bans => {
                let words = ban_words(vocabulary as u32) as usize;
                ((rows, words).into(), DType::U32)
            }
            Self::Context => {
                let history = context.map_or(0, |history| history.width as usize);
                ((held(context.is_some()), history).into(), DType::U32)
            }
            Self::Accepted => (held(accepted).into(), DType::U32),
        }
    }
}

/// The tables every runner holds, each over its part of the slot memory.
#[derive(Clone)]
struct Held {
    counts: Tensor,
    bias: Tensor,
    future: Tensor,
    predicates: Tensor,
    logical: Tensor,
    sampling: Tensor,
    verified: Tensor,
    bans: Tensor,
}

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
    /// Each table every runner holds.
    tables: Held,
    /// The context table, for a model whose layers read token ids, with the history the model
    /// reads.
    context: Option<(Tensor, TokenHistory)>,
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
    /// The ban masks' memory, when they lie apart from the other tables: those of a shape of the
    /// runner whose buckets hold other rows than the tables were made for ([`Slots::with_rows`]).
    bans: Option<Arc<Storage>>,
    /// The accepted counts' memory, when they lie apart from the other tables: those of a shape
    /// that lays the recurrent state out otherwise ([`Slots::restated`]).
    accepted: Option<Arc<Storage>>,
    /// Each table's offset in `storage`, by the table's discriminant.
    offsets: [u64; Table::COUNT],
    /// The page-locked blocks the step's copies read. A step that needs more than the blocks
    /// hold adds one; the blocks are folded into one when the next step begins.
    blocks: Vec<PinnedBuffer>,
    /// The bytes of the last block the step has written.
    used: usize,
    /// Whether copies were issued that the runner has not yet seen complete.
    unsettled: bool,
}

impl Device {
    /// The device address of `table`'s first byte.
    fn address(&self, table: Table) -> u64 {
        match (&self.bans, &self.accepted, table) {
            (Some(bans), _, Table::Bans) => bans.address(),
            (_, Some(accepted), Table::Accepted) => accepted.address(),
            _ => self.storage.address() + self.offsets[table as usize],
        }
    }
}

/// The shape and the element type of each table, by the table's discriminant (as
/// [`Table::shape`] states them).
fn shapes(
    slots: u32,
    vocabulary: u32,
    rows: u32,
    width: u32,
    context: Option<TokenHistory>,
    accepted: bool,
) -> Vec<(Shape, DType)> {
    let extents = (
        slots as usize,
        vocabulary as usize,
        rows as usize,
        width as usize,
    );
    Table::iter()
        .map(|table| table.shape(extents, context, accepted))
        .collect()
}

/// Each table's offset in the slot memory, by the table's discriminant, and the memory's bytes.
fn offsets(shapes: &[(Shape, DType)]) -> ([u64; Table::COUNT], u64) {
    let mut offsets = [0; Table::COUNT];
    let mut bytes = 0_u64;

    for (offset, (shape, dtype)) in offsets.iter_mut().zip(shapes) {
        *offset = bytes;

        let element = if *dtype == DType::U8 { 1 } else { 4 };
        bytes += (shape.elem_count() as u64 * element).next_multiple_of(ALIGNMENT);
    }
    (offsets, bytes)
}

impl Slots {
    /// The same tables, for a forward instance of the runner (`Runner::capturer`): its forward
    /// reads and writes them on the device as the runner's does, and it writes nothing into them
    /// from the host, so it holds none of the means to.
    pub(crate) fn shared(&self) -> Self {
        Self {
            slots: self.slots,
            vocabulary: self.vocabulary,
            rows: self.rows,
            width: self.width,
            tables: self.tables.clone(),
            context: self.context.clone(),
            accepted: self.accepted.clone(),
            device: None,
        }
    }

    /// The same tables for a shape of the runner whose buckets hold at most `rows` rows
    /// (`Runner::reshape`): a request's state lies in these tables whatever shape serves it, so
    /// every table is shared but the ban masks, which every step writes before its replay and
    /// which a shape of other rows holds in memory of its own, on the same stream. The two
    /// values write on the stream in turn: the one that serves writes.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for no row; the driver's refusal of the masks' allocation.
    pub(crate) fn with_rows(&self, rows: u32) -> Result<Self> {
        if rows == 0 {
            return Err(Error::Invalid(
                "slot state for buckets of no row holds no ban mask".into(),
            ));
        }
        let words = ban_words(self.vocabulary) as usize;
        let shape: Shape = (rows as usize, words).into();
        let bytes = rows as usize * words * 4;

        let mut tables = self.tables.clone();
        let device = match &self.device {
            Some(device) => {
                let bans = if rows == self.rows {
                    device.bans.clone()
                } else {
                    let storage = Storage::linear(device.stream.device(), bytes)?;
                    // SAFETY: the tensor lies over the whole allocation, which the value keeps
                    // for as long as the graphs captured over the tensor exist.
                    tables.bans = unsafe { storage.external(0, shape, DType::U32) }?;
                    Some(storage)
                };
                Some(Device {
                    stream: device.stream.clone(),
                    storage: Arc::clone(&device.storage),
                    bans,
                    accepted: device.accepted.clone(),
                    offsets: device.offsets,
                    blocks: Vec::new(),
                    used: 0,
                    unsettled: false,
                })
            }
            None => {
                // SAFETY: a tensor over the meta device has no memory: its address is never
                // read or written.
                tables.bans = unsafe { ops::external(0, shape, DType::U32, &ops::meta()) }?;
                None
            }
        };

        Ok(Self {
            slots: self.slots,
            vocabulary: self.vocabulary,
            rows,
            width: self.width,
            tables,
            context: self.context.clone(),
            accepted: self.accepted.clone(),
            device,
        })
    }

    /// The same tables for a shape of the runner that lays the recurrent state out otherwise
    /// (`Runner::reshape`): accepted counts of its own, each set to one on the stream, when
    /// `accepted`, for a shape whose forwards lend verify slots, and none otherwise (module
    /// documentation).
    ///
    /// # Errors
    ///
    /// The driver's refusal of the counts' allocation or of their fill.
    pub(crate) fn restated(mut self, accepted: bool) -> Result<Self> {
        if !accepted {
            self.accepted = None;
            if let Some(device) = &mut self.device {
                device.accepted = None;
            }
            return Ok(self);
        }

        let (shape, dtype): (Shape, DType) = ((self.slots as usize + 1).into(), DType::U32);
        let bytes = (self.slots as usize + 1) * 4;
        match &mut self.device {
            Some(device) => {
                let storage = Storage::linear(device.stream.device(), bytes)?;
                // SAFETY: the range is the whole allocation, which the device side keeps for as
                // long as the tensor over it exists.
                unsafe { platform::memset_d32(storage.address(), 1, bytes / 4, &device.stream) }?;
                // SAFETY: the tensor lies over the whole allocation, kept as above.
                self.accepted = Some(unsafe { storage.external(0, shape, dtype) }?);
                device.accepted = Some(storage);
            }
            None => {
                // SAFETY: a tensor over the meta device has no memory: its address is never
                // read or written.
                self.accepted = Some(unsafe { ops::external(0, shape, dtype, &ops::meta()) }?);
            }
        }
        Ok(self)
    }

    /// The bytes of device memory a shape's own accepted counts take ([`Slots::restated`]).
    pub(crate) fn accepted_bytes(slots: u32) -> u64 {
        (u64::from(slots) + 1) * 4
    }

    /// What each request slot's last step left, slot `s` at index `s − 1`: the accepted count
    /// the slot holds (one for a runner that holds none) and its verified length, read back once
    /// the stream passed every step issued on it. Slots of a runner that holds no device stand
    /// at one token of none.
    ///
    /// # Errors
    ///
    /// The driver's refusal of a copy or of the wait.
    pub(crate) fn standing(&self) -> Result<Vec<Standing>> {
        let slots = self.slots as usize;
        let Some(device) = &self.device else {
            return Ok(vec![
                Standing {
                    accepted: 1,
                    verified: 0
                };
                slots
            ]);
        };

        let mut verified = vec![0_i32; slots];
        let mut accepted = vec![1_u32; slots];
        // SAFETY: each destination holds the request rows the copy moves, and the stream is
        // waited for before either is read; the sources are tables of this value's storage.
        unsafe {
            platform::memcpy_async(
                verified.as_mut_ptr() as u64,
                device.address(Table::Verified),
                slots * 4,
                &device.stream,
            )?;
            if self.accepted.is_some() {
                platform::memcpy_async(
                    accepted.as_mut_ptr() as u64,
                    device.address(Table::Accepted),
                    slots * 4,
                    &device.stream,
                )?;
            }
        }
        device.stream.synchronize()?;

        Ok(accepted
            .into_iter()
            .zip(verified)
            .map(|(accepted, verified)| Standing {
                accepted: accepted.max(1),
                verified: u32::try_from(verified).unwrap_or(0),
            })
            .collect())
    }

    /// Sets every accepted count the runner holds to one, on the stream: the state of every slot
    /// stands in its home row (module documentation).
    ///
    /// # Errors
    ///
    /// The driver's refusal of the fill.
    pub(crate) fn accept_one(&mut self) -> Result<()> {
        let Some(device) = &mut self.device else {
            return Ok(());
        };
        if self.accepted.is_none() {
            return Ok(());
        }
        let destination = device.address(Table::Accepted);

        // SAFETY: the destination is the accepted counts of this value's storage, the padding
        // row included.
        unsafe { platform::memset_d32(destination, 1, self.slots as usize + 1, &device.stream) }?;
        Ok(())
    }

    /// The bytes of device memory the ban masks of buckets of at most `rows` rows take, over a
    /// `vocabulary`-wide logit row: what a shape of other rows allocates ([`Slots::with_rows`]).
    pub(crate) fn ban_bytes(vocabulary: u32, rows: u32) -> u64 {
        u64::from(rows) * u64::from(ban_words(vocabulary)) * 4
    }

    /// The bytes of device memory the state of `slots` request slots takes, over a
    /// `vocabulary`-wide logit row, for buckets of at most `rows` rows and `width` next input
    /// tokens a slot, with the context table of a model that reads the token history
    /// `context` and the accepted counts when `accepted`.
    pub(crate) fn bytes(
        slots: u32,
        vocabulary: u32,
        rows: u32,
        width: u32,
        context: Option<TokenHistory>,
        accepted: bool,
    ) -> u64 {
        offsets(&shapes(slots, vocabulary, rows, width, context, accepted)).1
    }

    /// The state of `slots` request slots as a runner starts it: every table zeroed and every
    /// accepted count one. The context table is held for a model whose layers read the token
    /// history `context`, and the accepted counts when `accepted`, for a runner whose forwards
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
        context: Option<TokenHistory>,
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
                    let address = storage.address() + offsets[Table::Accepted as usize];
                    // SAFETY: as above.
                    unsafe { platform::memset_d32(address, 1, slots as usize + 1, &stream) }?;
                }

                Some(Device {
                    stream,
                    storage,
                    bans: None,
                    accepted: None,
                    offsets,
                    blocks: Vec::new(),
                    used: 0,
                    unsettled: false,
                })
            }
            None => None,
        };

        // Each table's tensor over the memory, or a meta tensor for graphs that hold no device.
        let table = |table: Table| -> Result<Tensor> {
            let index = table as usize;
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
                None => Ok(unsafe { ops::external(0, shape, dtype, &ops::meta()) }?),
            }
        };

        let held = Held {
            counts: table(Table::Counts)?,
            bias: table(Table::Bias)?,
            future: table(Table::Future)?,
            predicates: table(Table::Predicates)?,
            logical: table(Table::Logical)?,
            sampling: table(Table::Sampling)?,
            verified: table(Table::Verified)?,
            bans: table(Table::Bans)?,
        };
        let context = context
            .map(|history| Ok::<_, Error>((table(Table::Context)?, history)))
            .transpose()?;
        let accepted = accepted.then(|| table(Table::Accepted)).transpose()?;

        Ok(Self {
            slots,
            vocabulary,
            rows,
            width,
            tables: held,
            context,
            accepted,
            device,
        })
    }

    /// The count table, `[slots, vocabulary]` I32.
    pub(crate) const fn counts(&self) -> &Tensor {
        &self.tables.counts
    }

    /// The bias table, `[slots, vocabulary]` F32.
    pub(crate) const fn bias(&self) -> &Tensor {
        &self.tables.bias
    }

    /// The next input tokens, `[slots + 1, width]` U32, the padding row included: what a
    /// decode row gathers its input token from.
    pub(crate) const fn future(&self) -> &Tensor {
        &self.tables.future
    }

    /// The logical positions, `[slots + 1]` I32, the padding row included.
    pub(crate) const fn logical(&self) -> &Tensor {
        &self.tables.logical
    }

    /// The sampling positions, `[slots + 1]` I32, the padding row included.
    pub(crate) const fn sampling(&self) -> &Tensor {
        &self.tables.sampling
    }

    /// The ban masks, `[rows, ceil(vocabulary / 32)]` U32.
    pub(crate) const fn bans(&self) -> &Tensor {
        &self.tables.bans
    }

    /// The context table, `[slots + 1, width]` U32, the padding row included: row `s - 1`
    /// holds the `width` token ids before the next row of request slot `s`, the oldest first.
    /// `None` for a runner whose model reads no token ids.
    pub(crate) fn context(&self) -> Option<&Tensor> {
        self.context.as_ref().map(|(table, _)| table)
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
        let requests = |table: &Tensor| table.narrow(0, 0, slots);

        Ok([
            requests(&self.tables.future)?,
            self.tables.predicates.clone(),
            requests(&self.tables.logical)?,
            requests(&self.tables.sampling)?,
            requests(&self.tables.verified)?,
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
    /// (`crate::batch::Batch::new_slots`), ahead of the step that reads it; `columns` are the
    /// batch's.
    ///
    /// - Each new slot's counts become its request's own history (`Sampler::count_row`): the
    ///   slot's previous request counted its tokens there, and the request may have generated
    ///   tokens before it reached this runner.
    /// - A request that states a bias or an allowed set has its bias row written
    ///   (`Sampler::bias_row`). A request that states neither leaves the row as it is: none of
    ///   its rows is flagged to read it (`Columns::biased`).
    /// - Where the runner holds them, the slot's context becomes the ids before the request's
    ///   first token of the batch (`NewSlot::context`), led by the model's fill id
    ///   (`nn::family::TokenHistory::fill`) for each position before the request's first
    ///   token, and its accepted count one: the request's state starts in its home row.
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
    /// batch that no row of the batch runs on, a context longer than the model's history, or a
    /// row whose values no slot state holds; the driver's refusal of a copy.
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
                self.zero(Table::Counts, u64::from(own) * row_bytes, self.vocabulary)?;
            } else {
                let row = sampler.count_row(own, &new.history)?;
                self.write(Table::Counts, u64::from(own) * row_bytes, &row)?;
            }

            if !new.bias.is_empty() || new.allowed.is_some() {
                let row = sampler.bias_row(own, &new.bias, new.allowed.as_deref())?;
                self.write(Table::Bias, u64::from(own) * row_bytes, &row)?;
            }

            if let Some((_, history)) = self.context {
                let width = history.width as usize;
                let Some(fill) = width.checked_sub(new.context.len()) else {
                    return Err(Error::Invalid(format!(
                        "a context of {} ids for a model that reads {width} before a token",
                        new.context.len()
                    )));
                };

                let mut row = vec![history.fill; fill];
                row.extend_from_slice(&new.context);
                self.write(
                    Table::Context,
                    u64::from(own) * 4 * u64::from(history.width),
                    &row,
                )?;
            }
            if self.accepted.is_some() {
                self.write(Table::Accepted, u64::from(own) * 4, &[1_u32])?;
            }

            if decode {
                self.start_decode(new.slot, columns)?;
            }
        }
        Ok(())
    }

    /// Sets the accepted count of every slot whose recurrent state `copies` restore before the
    /// step (`crate::batch::Batch`'s state copies) to one: a restore writes the slot's home row,
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
                self.write(Table::Accepted, u64::from(own) * 4, &[1_u32])?;
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

        self.write(Table::Bans, 0, masks)
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

        self.write(Table::Future, own * u64::from(self.width) * 4, &[id])?;
        self.write(Table::Logical, own * 4, &[logical])?;
        self.write(Table::Sampling, own * 4, &[sampling])?;
        self.write(Table::Verified, own * 4, &[verified])?;
        self.write(Table::Predicates, own, &[1_u8])
    }

    /// Copies `values` to byte `offset` of `table`, on the stream.
    fn write<T: bytemuck::Pod>(&mut self, table: Table, offset: u64, values: &[T]) -> Result<()> {
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

        let destination = device.address(table) + offset;
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
    fn zero(&mut self, table: Table, offset: u64, words: u32) -> Result<()> {
        let Some(device) = &mut self.device else {
            return Ok(());
        };
        let destination = device.address(table) + offset;
        device.unsettled = true;

        // SAFETY: the destination is a row of a table of this value's storage, which the
        // caller addresses by a slot the table holds.
        unsafe { platform::memset_d32(destination, 0, words as usize, &device.stream) }?;
        Ok(())
    }
}
