//! The input tensors of a causal runner: their layout in the input block, and the padding rules
//! that turn a batch's host columns into their bytes.
//!
//! Every forward of the runner reads the same columns at the same addresses. The layout is
//! computed once, from the most rows and the most tokens any bucket holds, and every column
//! starts at an aligned offset of the runner's input block; a bucket narrower than the largest
//! reads the head of each column.
//!
//! ## The columns
//!
//! `T` is the layout's tokens and `R` its rows. A *padding entry* is what a column holds past
//! the batch's own tokens or rows, up to the bucket's.
//!
//! | Column | Type and extent | Holds | Padding entry |
//! | --- | --- | --- | --- |
//! | [`Column::InputIds`] | U32 `[T]` | the token ids a prefill embeds | `0` |
//! | [`Column::Positions`] | I32 `[T]` | rotary positions | `0` |
//! | [`Column::SlotMapping`] | U32 `[T]` | the cache slot each token writes | [`NO_SLOT`] |
//! | [`Column::Targets`] | U32 `[T]` | the prompt token each token's logits score | [`NO_SLOT`] |
//! | [`Column::SeqLens`] | U32 `[R]` | keys visible to each row's last query | `0` |
//! | [`Column::QueryOffsets`] | U32 `[R + 1]` | packed query offsets | the batch's token count |
//! | [`Column::LogitsRows`] | U32 `[R]` | the hidden row each row's logits are computed from | `0` |
//! | [`Column::Slots`] | U32 `[R]` | each row's request slot, as the sampler and the recurrent state read it | [`NO_SLOT`] |
//! | [`Column::TableRows`] | U32 `[R]` | each row's row of the page tables the forward gathers | the padding row |
//! | [`Column::SlotRows`] | U32 `[R]` | each row's row of the slots' next input tokens | the padding row |
//! | [`Column::CallSlots`] | U32 `[R]` | each call's row of the slots' positions | the padding row |
//! | [`Column::Parameters`] | F32 `[R, PARAMETER_COLUMNS]` | sampling parameters | zeros |
//! | [`Column::Draws`] | F32 `[R]` | each row's uniform draw | `0.0` |
//! | [`Column::Adjustments`] | U32 `[R, 2]` | each row's bias flag and ban mask index | `0`, [`NO_SLOT`] |
//! | [`Column::Checkpoints`] | U32 `[R]` | the token count each row's recurrent state is kept after | `0` |
//! | [`Column::Commit`] | I32 `[COMMIT_FIELDS, R]` | the commit table, one row per call | slot and task `-1` |
//! | [`Column::Valid`], [`Column::Active`] | I32 `[R]` | whether a task's token counts | `1` |
//! | [`Column::Continuation`] | U8 `[R]` | whether a task's request continues | `0` |
//! | [`Column::Points`] | I32 `[R]` | the span a verification accepted | `0` |
//! | [`Column::LiveTokens`] | U32 `[1]` | the tokens the batch itself holds | |
//!
//! ## Indices
//!
//! The scheduler states indices as `i32` with `-1` for "none". On the device an index column
//! is U32, and what `-1` becomes follows from the kernel that reads the entry:
//!
//! - where the kernel defines an entry that addresses nothing, `-1` becomes [`NO_SLOT`]: a
//!   token that writes no cache slot, a row the sampler and the recurrent state skip, a token
//!   that scores no target, a row that bans nothing;
//! - where the entry feeds a gather, it stays a valid index: a row without a slot gathers the
//!   padding row of the pooled page tables and of the slot state (the row past the last request
//!   slot), and a row whose logits nothing reads gathers hidden row `0`, whose sample is
//!   discarded.
//!
//! The commit table keeps the scheduler's signed form, because the commit kernel defines a
//! negative slot and a negative task as a row that commits nothing.
//!
//! ## Padding
//!
//! A padding token writes no cache slot. A row the batch does not fill is an empty sequence:
//! `seq_lens` states zero keys, since a row that names the padding page and claims a key would
//! have the attention kernel read that page as a request's. The query offsets end at the
//! batch's own tokens, so padding lies inside no row's span and the batch's live token count
//! is what bounds an operator over the token axis. The commit table's rows past the batch's
//! calls are inert.
//!
//! ## A paired forward
//!
//! A paired forward computes a served batch again beside the kernels of another generation,
//! and must write nothing a serving request reads. Its columns are the batch's with three
//! differences ([`fill`] with a scratch slot mapping): the tokens write the scratch page set
//! in its own numbering ([`scratch_column`]), row `r` gathers row `r` of the scratch set's
//! page table, and the commit table is inert.

use ipc::execution::{BatchColumns, BatchRow, BatchRows, RowKind};
use ipc::philox::DrawAddress;
use nn::NO_SLOT;
use proto::module::commit::COMMIT_FIELDS;
use proto::module::sampler::PARAMETER_COLUMNS;
use uniserve_tensor::DType;

use crate::buckets::Bucket;
use crate::sampler;
use crate::{Error, Result};

/// The alignment of a column in the input block: the allocator's, which `Graphs::input` asks
/// of every input tensor.
const ALIGNMENT: u64 = backend::allocator::ALIGNMENT;

/// One input tensor of the runner (module documentation).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Column {
    InputIds,
    Positions,
    SlotMapping,
    Targets,
    SeqLens,
    QueryOffsets,
    LogitsRows,
    Slots,
    TableRows,
    SlotRows,
    CallSlots,
    Parameters,
    Draws,
    Adjustments,
    Checkpoints,
    Commit,
    Valid,
    Active,
    Continuation,
    Points,
    LiveTokens,
}

impl Column {
    /// Every column, in the order the layout places them.
    const ALL: [Self; 21] = [
        Self::InputIds,
        Self::Positions,
        Self::SlotMapping,
        Self::Targets,
        Self::SeqLens,
        Self::QueryOffsets,
        Self::LogitsRows,
        Self::Slots,
        Self::TableRows,
        Self::SlotRows,
        Self::CallSlots,
        Self::Parameters,
        Self::Draws,
        Self::Adjustments,
        Self::Checkpoints,
        Self::Commit,
        Self::Valid,
        Self::Active,
        Self::Continuation,
        Self::Points,
        Self::LiveTokens,
    ];

    /// The element type of the column's tensor.
    pub(crate) const fn dtype(self) -> DType {
        match self {
            Self::Positions | Self::Commit | Self::Valid | Self::Active | Self::Points => {
                DType::I32
            }
            Self::Parameters | Self::Draws => DType::F32,
            Self::Continuation => DType::U8,
            _ => DType::U32,
        }
    }

    /// The bytes of one element.
    const fn element(self) -> u64 {
        match self {
            Self::Continuation => 1,
            _ => 4,
        }
    }

    /// The elements of the column in a layout of `rows` rows and `tokens` tokens.
    const fn elements(self, rows: usize, tokens: usize) -> usize {
        match self {
            Self::InputIds | Self::Positions | Self::SlotMapping | Self::Targets => tokens,
            Self::QueryOffsets => rows + 1,
            Self::Parameters => rows * PARAMETER_COLUMNS as usize,
            Self::Adjustments => rows * 2,
            Self::Commit => rows * COMMIT_FIELDS,
            Self::LiveTokens => 1,
            _ => rows,
        }
    }
}

/// Where each input tensor lies in the runner's input block.
///
/// It is computed once, for the most rows and the most tokens any bucket of the runner holds,
/// which may come from two buckets: a long narrow prefill and a short wide decode.
#[derive(Clone, Debug)]
pub(crate) struct Layout {
    /// The rows a row column holds.
    rows: usize,
    /// The tokens a token column holds.
    tokens: usize,
    /// Each column's offset in the block, in the order of [`Column::ALL`].
    offsets: [u64; Column::ALL.len()],
    /// The bytes of the block.
    bytes: u64,
}

impl Layout {
    /// The layout of columns of `rows` rows and `tokens` tokens.
    pub(crate) fn new(rows: usize, tokens: usize) -> Self {
        let mut offsets = [0; Column::ALL.len()];
        let mut bytes = 0_u64;

        for (offset, column) in offsets.iter_mut().zip(Column::ALL) {
            *offset = bytes;

            let length = column.elements(rows, tokens) as u64 * column.element();
            bytes += length.next_multiple_of(ALIGNMENT);
        }

        Self {
            rows,
            tokens,
            offsets,
            bytes,
        }
    }

    /// The layout that holds every bucket of `buckets`.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for an empty set.
    pub(crate) fn of<'a>(buckets: impl IntoIterator<Item = &'a Bucket>) -> Result<Self> {
        let (mut rows, mut tokens) = (0, 0);
        for bucket in buckets {
            rows = rows.max(bucket.rows as usize);
            tokens = tokens.max(bucket.tokens as usize);
        }

        if rows == 0 || tokens == 0 {
            return Err(Error::Invalid(
                "a causal runner with no bucket has no input tensor to lay out".into(),
            ));
        }
        Ok(Self::new(rows, tokens))
    }

    /// The rows a row column holds: the capacity of the commit table.
    pub(crate) const fn rows(&self) -> usize {
        self.rows
    }

    /// The bytes of the input block.
    pub(crate) const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The offset of `column` in the input block.
    pub(crate) const fn offset(&self, column: Column) -> u64 {
        self.offsets[column as usize]
    }

    /// Writes `values` at the head of `column` in `block`, the bytes of the input block.
    fn put<T: bytemuck::Pod>(&self, block: &mut [u8], column: Column, values: &[T]) -> Result<()> {
        let source: &[u8] = bytemuck::cast_slice(values);
        let start = self.offset(column) as usize;
        let capacity = column.elements(self.rows, self.tokens) * column.element() as usize;

        if source.len() > capacity || start + source.len() > block.len() {
            return Err(Error::Invariant(format!(
                "{} bytes of {column:?} do not fit its {capacity} bytes at offset {start} of an \
                 input block of {} bytes",
                source.len(),
                block.len()
            )));
        }

        block[start..start + source.len()].copy_from_slice(source);
        Ok(())
    }

    /// The first `elements` elements of `column` in `block`, as [`Layout::put`] wrote them.
    #[cfg(test)]
    pub(crate) fn read<T: bytemuck::Pod>(
        &self,
        block: &[u8],
        column: Column,
        elements: usize,
    ) -> Vec<T> {
        let start = self.offset(column) as usize;
        let bytes = elements * std::mem::size_of::<T>();

        bytemuck::pod_collect_to_vec(&block[start..start + bytes])
    }
}

/// The unsigned index a kernel reads for the scheduler's `value`: `value` itself, or `none` for
/// a negative one.
const fn index(value: i32, none: u32) -> u32 {
    if value < 0 { none } else { value as u32 }
}

/// A count the scheduler states signed, which no batch states negative.
fn count(name: &str, value: i32) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::Invalid(format!("the batch states a negative {name} of {value}")))
}

/// Writes into `block`, the bytes of the input block, the columns of `columns` padded to
/// `bucket` (module documentation).
///
/// `slots` is the runner's request slots, whose count is also the padding row of the pooled
/// page tables and of the slot state; `vocabulary` is the width of a logit row. `scratch` is
/// the scratch slot mapping of a paired forward over the bucket's tokens ([`scratch_column`]),
/// and `None` for a serving step.
///
/// # Errors
///
/// [`Error::Invalid`] for a batch the bucket does not hold, a row whose slot the runner holds
/// no state for, a negative token id, length or offset, or columns the sampler's encoders
/// refuse; [`Error::Invariant`] for a commit table that is not one row per call.
pub(crate) fn fill(
    layout: &Layout,
    slots: u32,
    vocabulary: u32,
    columns: &BatchColumns,
    bucket: &Bucket,
    scratch: Option<&[u32]>,
    block: &mut [u8],
) -> Result<()> {
    let rows = &columns.rows;
    let (tokens, sequences, calls) = (rows.tokens(), rows.rows(), rows.calls.len());
    let (bucket_tokens, bucket_rows) = (bucket.tokens as usize, bucket.rows as usize);

    if tokens > bucket_tokens || sequences > bucket_rows || calls > bucket_rows {
        return Err(Error::Invalid(format!(
            "a batch of {sequences} rows, {calls} calls and {tokens} tokens does not fit bucket \
             {bucket}"
        )));
    }
    if bucket_tokens > layout.tokens || bucket_rows > layout.rows {
        return Err(Error::Invariant(format!(
            "bucket {bucket} is larger than the {} rows and {} tokens the input tensors hold",
            layout.rows, layout.tokens
        )));
    }

    // ---- Token columns: the batch's tokens, then padding that is stored nowhere ----

    // A decode's ids are gathered from the slot state on the device and this column is not
    // read; a prefill embeds it, so each entry is a token of the vocabulary.
    let mut ids = vec![0_u32; bucket_tokens];
    for (token, (id, held)) in ids.iter_mut().zip(&columns.input_ids).enumerate() {
        if *held < 0 && !bucket.variant.is_decode() {
            return Err(Error::Invalid(format!(
                "token {token} of the batch states the id {held}, which is no token"
            )));
        }
        *id = index(*held, 0);
    }
    layout.put(block, Column::InputIds, &ids)?;

    let mut positions = vec![0_i32; bucket_tokens];
    positions[..tokens].copy_from_slice(&rows.positions);
    layout.put(block, Column::Positions, &positions)?;

    let mut mapping = vec![NO_SLOT; bucket_tokens];
    match scratch {
        Some(scratch) if scratch.len() == bucket_tokens => mapping.copy_from_slice(scratch),
        Some(scratch) => {
            return Err(Error::Invariant(format!(
                "a scratch slot mapping of {} tokens for bucket {bucket}",
                scratch.len()
            )));
        }
        None => {
            for (slot, held) in mapping.iter_mut().zip(&rows.slot_mapping) {
                *slot = index(*held, NO_SLOT);
            }
        }
    }
    layout.put(block, Column::SlotMapping, &mapping)?;

    let mut targets = vec![NO_SLOT; bucket_tokens];
    for (target, held) in targets.iter_mut().zip(&columns.targets) {
        *target = index(*held, NO_SLOT);
    }
    layout.put(block, Column::Targets, &targets)?;

    layout.put(block, Column::LiveTokens, &[tokens as u32])?;

    // ---- Row columns: the batch's rows, then empty sequences ----

    let mut lengths = vec![0_u32; bucket_rows];
    for (length, held) in lengths.iter_mut().zip(&rows.seq_lens) {
        *length = count("sequence length", *held)?;
    }
    layout.put(block, Column::SeqLens, &lengths)?;

    // The offsets end where the batch's own tokens end: every row past the batch's holds no
    // query, and the padding tokens belong to no row.
    if rows.query_offsets.len() != sequences + 1 {
        return Err(Error::Invalid(format!(
            "the batch states {} query offsets for its {sequences} rows",
            rows.query_offsets.len()
        )));
    }
    let mut offsets = Vec::with_capacity(bucket_rows + 1);
    for held in &rows.query_offsets {
        offsets.push(count("query offset", *held)?);
    }
    let end = offsets.last().copied().unwrap_or(0);
    offsets.resize(bucket_rows + 1, end);
    layout.put(block, Column::QueryOffsets, &offsets)?;

    // A row's logits are those of the last token of its own query span, which is its own
    // index only when every row holds one token. A row that samples nothing gathers hidden
    // row zero, and so does a padding row; their samples are discarded.
    let mut logits = vec![0_u32; bucket_rows];
    for (row, (gathered, held)) in logits.iter_mut().zip(&rows.logits_rows).enumerate() {
        if *held >= 0 {
            *gathered = offsets[row + 1].min(tokens as u32).saturating_sub(1);
        }
    }
    layout.put(block, Column::LogitsRows, &logits)?;

    let mut sampler_slots = vec![NO_SLOT; bucket_rows];
    let mut slot_rows = vec![slots; bucket_rows];
    for (row, held) in columns.slots.iter().enumerate().take(sequences) {
        if *held >= 0 && *held as u32 >= slots {
            return Err(Error::Invalid(format!(
                "row {row} of the batch names slot {held} of a runner of {slots} request slots"
            )));
        }
        sampler_slots[row] = index(*held, NO_SLOT);
        slot_rows[row] = index(*held, slots);
    }
    layout.put(block, Column::Slots, &sampler_slots)?;
    layout.put(block, Column::SlotRows, &slot_rows)?;

    // A serving row's pages are its slot's row of the pooled tables; a paired forward's are
    // its own row of the scratch set's table.
    match scratch {
        Some(_) => {
            let own: Vec<u32> = (0..bucket.rows).collect();
            layout.put(block, Column::TableRows, &own)?;
        }
        None => layout.put(block, Column::TableRows, &slot_rows)?,
    }

    // A call's state is its slot's, which its rows share: the slot of the first row it owns.
    let mut call_slots = vec![slots; bucket_rows];
    for (slot, call) in call_slots.iter_mut().zip(&rows.calls) {
        *slot = slot_rows
            .get(call.rows.start as usize)
            .copied()
            .unwrap_or(slots);
    }
    layout.put(block, Column::CallSlots, &call_slots)?;

    let parameters = sampler::parameters(&columns.parameters, bucket_rows)?;
    layout.put(block, Column::Parameters, &parameters)?;

    let draws = sampler::draws(&columns.draws, bucket_rows)?;
    layout.put(block, Column::Draws, &draws)?;

    // The ban masks themselves are the slot state's: a batch that bans nothing copies none.
    let (adjustments, _) = sampler::adjustments(&columns.adjustments, bucket_rows, vocabulary)?;
    layout.put(block, Column::Adjustments, &adjustments)?;

    let mut checkpoints = vec![0_u32; bucket_rows];
    for (checkpoint, held) in checkpoints.iter_mut().zip(&columns.checkpoints) {
        *checkpoint = count("state checkpoint", *held)?;
    }
    layout.put(block, Column::Checkpoints, &checkpoints)?;

    // ---- The commit: one row per call, the rest of the table inert ----

    // A paired forward commits nothing: the slots' state is the serving requests'.
    let table = match scratch {
        Some(_) => sampler::commit_table(&sampler::inert_commit(calls), calls, layout.rows)?,
        None => sampler::commit_table(&columns.commit, calls, layout.rows)?,
    };
    layout.put(block, Column::Commit, &table)?;

    // The per-task columns of a step that verifies no draft: every task's token counts, no
    // task's request is continued by a predicate, and no span is accepted.
    layout.put(block, Column::Valid, &vec![1_i32; layout.rows])?;
    layout.put(block, Column::Active, &vec![1_i32; layout.rows])?;
    layout.put(block, Column::Continuation, &vec![0_u8; layout.rows])?;
    layout.put(block, Column::Points, &vec![0_i32; layout.rows])?;

    Ok(())
}

/// The columns of an idle step of `bucket`: a batch of the bucket's size whose rows address no
/// slot, page, token or request, at the page size `page_size` and under the key bound
/// `max_kv_len` every step of the runner states.
///
/// Every token writes no cache slot and every row is an empty sequence over no request slot,
/// which samples a token nothing reads and commits nothing. The query spans are the ones a
/// full batch of the bucket has: a decode row holds its own token, and a prefill's first row
/// holds the bucket's whole run. A forward over these makes the launches of a serving step of
/// the bucket and changes no request's state.
pub(crate) fn idle(bucket: &Bucket, page_size: u32, max_kv_len: u32) -> BatchColumns {
    let (rows, tokens) = (bucket.rows as usize, bucket.tokens as usize);
    let decode = bucket.variant.is_decode();

    // One call per row for a decode, whose row is one request's one token; one call over the
    // run for a prefill.
    let calls: Vec<BatchRow> = if decode {
        (0..bucket.rows)
            .map(|row| BatchRow {
                request: 0,
                kind: RowKind::Decode,
                cache_group: 0,
                rows: row..row + 1,
            })
            .collect()
    } else {
        vec![BatchRow {
            request: 0,
            kind: RowKind::Prefill,
            cache_group: 0,
            rows: 0..bucket.rows,
        }]
    };

    let query_offsets = if decode {
        (0..=bucket.rows as i32).collect()
    } else {
        let mut offsets = vec![bucket.tokens as i32; rows + 1];
        offsets[0] = 0;
        offsets
    };

    BatchColumns {
        rows: BatchRows {
            positions: vec![0; tokens],
            slot_mapping: vec![-1; tokens],
            seq_lens: vec![0; rows],
            query_offsets,
            logits_rows: vec![-1; rows],
            calls: calls.clone(),
        },
        input_ids: vec![0; tokens],
        slots: vec![-1; rows],
        parameters: vec![0.0; rows * PARAMETER_COLUMNS as usize],
        draws: vec![DrawAddress::default(); rows],
        commit: sampler::inert_commit(calls.len()),
        page_size,
        max_kv_len,
        ..BatchColumns::default()
    }
}

/// The cache slots a paired forward writes for a batch: the batch's own tokens, in the scratch
/// page set's numbering, as `[tokens]` U32.
///
/// A slot is flat over one plane's pages, `page · page_size + offset`, and the scratch set
/// gives row `r` the pages `r · pages .. (r + 1) · pages`, which is what its page table
/// states. Token `t` of row `r` at position `p` therefore writes
/// `(r · pages + p / page_size) · page_size + p % page_size`: a slot of the row's own scratch
/// pages, and never one a serving request reads. A token the batch does not write (a negative
/// slot) writes nothing here either and holds [`NO_SLOT`].
///
/// `positions` and `slots` are the batch's `[tokens]` columns and `query_offsets` its
/// `[rows + 1]` packed offsets.
///
/// # Errors
///
/// [`Error::Invariant`] for a scratch set of no page or columns that describe different token
/// counts; [`Error::Invalid`] for a token whose position lies past the row's scratch pages.
pub(crate) fn scratch_slot_mapping(
    positions: &[i32],
    query_offsets: &[i32],
    slots: &[i32],
    pages: u32,
    page_size: u32,
) -> Result<Vec<u32>> {
    if page_size == 0 || pages == 0 {
        return Err(Error::Invariant(
            "a scratch page set needs a page count and a page size above zero".into(),
        ));
    }

    let tokens = positions.len();
    let closes = query_offsets
        .last()
        .is_some_and(|end| usize::try_from(*end).is_ok_and(|end| end == tokens));
    if query_offsets.len() < 2 || !closes || slots.len() != tokens {
        return Err(Error::Invariant(
            "the batch's offsets, positions and slots describe different token counts".into(),
        ));
    }

    let mut mapping = vec![NO_SLOT; tokens];
    for (row, span) in query_offsets.windows(2).enumerate() {
        let (start, end) = (span[0], span[1]);
        if start < 0 || end < start {
            return Err(Error::Invariant(format!(
                "row {row}'s packed query offsets are {start}..{end}"
            )));
        }

        for token in start as usize..end as usize {
            if slots[token] < 0 {
                continue;
            }

            let position = u32::try_from(positions[token]).map_err(|_| {
                Error::Invariant(format!(
                    "token {token} writes the cache at position {}, which is no position",
                    positions[token]
                ))
            })?;
            let (page, offset) = (position / page_size, position % page_size);
            if page >= pages {
                return Err(Error::Invalid(format!(
                    "row {row} writes position {position}, which needs page {page} of a scratch \
                     set of {pages} pages a row"
                )));
            }

            let slot =
                u64::from(row as u32 * pages + page) * u64::from(page_size) + u64::from(offset);
            mapping[token] = u32::try_from(slot)
                .ok()
                .filter(|slot| *slot != NO_SLOT)
                .ok_or_else(|| {
                    Error::Invalid(format!(
                        "a scratch slot of {slot} is outside the slots a cache column addresses"
                    ))
                })?;
        }
    }
    Ok(mapping)
}

/// The scratch slot column of a paired forward over a bucket of `tokens` tokens: the batch's
/// scratch slots ([`scratch_slot_mapping`]), then [`NO_SLOT`] for every token the batch does
/// not hold.
///
/// A forward addresses the whole column, so every token of the bucket states a value: a token
/// past the batch's own names no page, where a value an earlier batch left would send the
/// paired forward's writes into a page a serving request reads.
///
/// # Errors
///
/// As [`scratch_slot_mapping`], or [`Error::Invariant`] for a batch of more tokens than the
/// bucket.
pub(crate) fn scratch_column(
    positions: &[i32],
    query_offsets: &[i32],
    slots: &[i32],
    tokens: usize,
    pages: u32,
    page_size: u32,
) -> Result<Vec<u32>> {
    if positions.len() > tokens {
        return Err(Error::Invariant(format!(
            "a batch of {} tokens in a bucket of {tokens}",
            positions.len()
        )));
    }

    let mut mapping = if positions.is_empty() {
        Vec::new()
    } else {
        scratch_slot_mapping(positions, query_offsets, slots, pages, page_size)?
    };
    mapping.resize(tokens, NO_SLOT);
    Ok(mapping)
}
