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
//! | [`Column::SlotMapping`] | U32 `[T]` | the cache slot each token writes ([`slot_mapping`]) | [`NO_SLOT`] |
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
//! | [`Column::NextContext`] | U32 `[R, 2]` | where the two token ids its slot's next row continues from lie in the id history | the row's own context |
//! | [`Column::StateIndices`] | U32 `[R, D + 1]` | each row's slots of a stepwise state field (`nn::VerifySlots::state_indices`) | [`NO_SLOT`] |
//!
//! A runner that speculates lays out the columns of its verifications and of the drafter's
//! forward behind them, which `speculation` states and fills. `D` is the drafts a verification
//! holds at most and `W` the pages of a row of the drafter's page table; without speculation
//! both are zero and the columns hold nothing. In a verification bucket the row columns above
//! are one entry per token, each token being an attention row and a sampling row, and the
//! columns below state its sequences.
//!
//! | Column | Type and extent | Holds | Padding entry |
//! | --- | --- | --- | --- |
//! | [`Column::StateOffsets`] | U32 `[R + 1]` | the packed token offsets of the batch's sequences | the batch's token count |
//! | [`Column::StateLens`] | U32 `[R]` | the tokens each sequence has seen, its own included | `0` |
//! | [`Column::StateSlots`] | U32 `[R]` | each sequence's request slot, as the recurrent state reads it | [`NO_SLOT`] |
//! | [`Column::Drafts`] | U32 `[T]` | the draft each token's sample is compared with: the next token of its row | [`NO_SLOT`] |
//! | [`Column::Extras`] | U32 `[T, D]` | the drafts before each token in its row, which its penalties count | [`NO_SLOT`] |
//! | [`Column::Windows`] | U32 `[R, D + 1]` | the token of each row at each offset, its last one past its end | [`NO_SLOT`] |
//! | [`Column::Firsts`] | U32 `[R]` | each row's first token | `0` |
//! | [`Column::DraftTokens`] | U32 `[R, D]` | each row's drafts, the tokens its count tasks count | [`NO_SLOT`] |
//! | [`Column::DraftSteps`] | U32 `[R, D]` | each draft's offset among its row's drafts | the offset |
//! | [`Column::TokenCap`] | U32 `[T]` | the largest token of the vocabulary | the same |
//! | [`Column::NextIds`] | U32 `[T]` | the token after each entry that the batch states | `0` |
//! | [`Column::NextRows`] | U32 `[T]` | the sampling row whose token is the token after each entry | [`NO_SLOT`] |
//! | [`Column::EntryRows`] | U32 `[T]` | the verification each entry belongs to | [`NO_SLOT`] |
//! | [`Column::EntrySteps`] | U32 `[T]` | each entry's offset in its verification; `0` for an entry of a prompt, `1` for a token that is no entry | `1` |
//! | [`Column::DraftSlots`] | U32 `[T]` | the draft cache slot each entry writes | [`NO_SLOT`] |
//! | [`Column::NoSlots`] | U32 `[T]` | [`NO_SLOT`] | the same |
//! | [`Column::DraftLens`] | U32 `[R]` | the keys visible to the last token of each drafter row | `0` |
//! | [`Column::DraftTables`] | U32 `[R, W]` | each drafter row's pages of the draft cache | the padding unit's pages |
//! | [`Column::LastEntries`] | U32 `[R]` | the last entry of each row a prompt states; a verification's first token | `0` |
//! | [`Column::ProposalRows`] | U32 `[R]` | the verification each row is | [`NO_SLOT`] |
//! | [`Column::ChainPositions`] | I32 `[D, R]` | the position each chained step of a row evaluates, before the drafts it accepted | `0` |
//! | [`Column::ChainLens`] | U32 `[D, R]` | the keys visible to it | `0` |
//! | [`Column::ChainSlots`] | U32 `[R, 3 D]` | the draft cache slots of the positions after a row's entries, then [`NO_SLOT`] | [`NO_SLOT`] |
//! | [`Column::ChainIndex`] | U32 `[D, R]` | where each chained step of a row reads its slot in [`Column::ChainSlots`] | a [`NO_SLOT`] cell |
//! | [`Column::ChainOffsets`] | U32 `[R + 1]` | the packed offsets of a chained step: one token a row | the batch's row count |
//! | [`Column::LiveRows`] | U32 `[1]` | the rows the batch itself holds | |
//! | [`Column::TokenContext`] | U32 `[T, 2]` | where the two token ids before each token lie in the id history | the token's own context |
//!
//! ## Token ids
//!
//! A model whose layers read token ids (`nn::family::TextModel::reads_token_ids`) reads the two
//! ids before each row's first token (`nn::TokenBatch::ngram_context`), which the request slots
//! keep (`slots`). A forward reads them from its **id history**, `[2 · rows + tokens]` U32: the
//! context of each of the bucket's rows, gathered by its slot, then the bucket's token ids.
//! Each row's slot continues from the row's last two ids in it, which
//! [`Column::NextContext`] names: the row's last two tokens, its context's later id and its one
//! token, or, for a row of no token, its own context. The forward writes them into the slot
//! as the commit advances the slot's other state. A verification's rows are its tokens, each
//! of which reads its request's context through its slot; [`Column::TokenContext`] names each
//! token's two ids before it in that history, and the ids its slot continues from are those of
//! the token its acceptance ends on (`speculation`).
//!
//! ## Verify slots
//!
//! A runner that speculates over a model whose state keeps a slot per verified token
//! (`nn::family::StateField::stepwise`) gives each request `Q = D + 1` slots of such a field
//! (`StateMemory`): request slot `s` owns its home row `s - 1` and the rows
//! `slots + (s - 1)(Q - 1) .. slots + s(Q - 1)`. [`Column::StateIndices`] states them for each
//! state row, its home row first ([`state_indices`]).
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
//! ## Cache slots
//!
//! A batch names the cache position each token writes its key and value at, `-1` for none
//! (`Rows::cache_positions`), never a unit. The cache slot is computed here through the rank's
//! device block table ([`slot_mapping`]): position `p` of a call on slot `s` lies in the block
//! at position `p / page_size` of the slot's table of the call's cache group, whose run's unit
//! `u` gives the slot `u · page_size + p % page_size`. A token is written only into an entry
//! the table states writable: a call one of whose tokens names a position whose entry is
//! missing or read-only is refused, and none of its tokens writes.
//!
//! ## Request slots
//!
//! The scheduler numbers request slots from one: slot `s` of `1..=slots`. On the device a
//! request slot owns **row `s - 1`** of every per-slot buffer (the runner's slot state, the
//! sampler's count and bias tables, the pooled page tables, the recurrent state's buffers),
//! and the padding row of the buffers that have one is the last, row `slots`. [`slot_row`]
//! is that rule, applied where each of the scheduler's values enters: the rows' slots (the
//! sampler's slot column, the table rows and the slot rows), the calls' slots, the slot field
//! of the commit table, and the slots a batch starts (`slots`). Slot zero and a slot past
//! `slots` name no request slot and are refused.
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
//! differences ([`fill`] of a paired forward): the tokens write the scratch page set in its own
//! numbering ([`scratch_column`]), row `r` gathers row `r` of the scratch set's page table, and
//! the commit table is inert.

use ipc::batch::{Columns, Phase, Row, Rows, Variant};
use ipc::philox::DrawAddress;
use nn::NO_SLOT;
use proto::module::commit::{COMMIT_FIELDS, column};
use proto::module::sampler::PARAMETER_COLUMNS;
use tensor::DType;

use crate::buckets::Bucket;
use crate::memory::tables::BlockTables;
use crate::sampler;
use crate::{Error, Result};

/// The alignment of a column in the input block: the allocator's, which `Graphs::input` asks
/// of every input tensor.
const ALIGNMENT: u64 = ops::allocator::ALIGNMENT;

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
    StateOffsets,
    StateLens,
    StateSlots,
    Drafts,
    Extras,
    Windows,
    Firsts,
    DraftTokens,
    DraftSteps,
    TokenCap,
    NextIds,
    NextRows,
    EntryRows,
    EntrySteps,
    DraftSlots,
    NoSlots,
    DraftLens,
    DraftTables,
    LastEntries,
    ProposalRows,
    ChainPositions,
    ChainLens,
    ChainSlots,
    ChainIndex,
    ChainOffsets,
    LiveRows,
    NextContext,
    StateIndices,
    TokenContext,
}

impl Column {
    /// Every column, in the order the layout places them.
    const ALL: [Self; 50] = [
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
        Self::StateOffsets,
        Self::StateLens,
        Self::StateSlots,
        Self::Drafts,
        Self::Extras,
        Self::Windows,
        Self::Firsts,
        Self::DraftTokens,
        Self::DraftSteps,
        Self::TokenCap,
        Self::NextIds,
        Self::NextRows,
        Self::EntryRows,
        Self::EntrySteps,
        Self::DraftSlots,
        Self::NoSlots,
        Self::DraftLens,
        Self::DraftTables,
        Self::LastEntries,
        Self::ProposalRows,
        Self::ChainPositions,
        Self::ChainLens,
        Self::ChainSlots,
        Self::ChainIndex,
        Self::ChainOffsets,
        Self::LiveRows,
        Self::NextContext,
        Self::StateIndices,
        Self::TokenContext,
    ];

    /// The element type of the column's tensor.
    pub(crate) const fn dtype(self) -> DType {
        match self {
            Self::Positions
            | Self::Commit
            | Self::Valid
            | Self::Active
            | Self::Points
            | Self::ChainPositions => DType::I32,
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

    /// The elements of the column in a layout of `rows` rows and `tokens` tokens, for
    /// verifications of at most `drafts` drafts and a drafter's page table of `pages` pages a
    /// row.
    const fn elements(self, rows: usize, tokens: usize, drafts: usize, pages: usize) -> usize {
        match self {
            Self::InputIds
            | Self::Positions
            | Self::SlotMapping
            | Self::Targets
            | Self::Drafts
            | Self::NextIds
            | Self::NextRows
            | Self::EntryRows
            | Self::EntrySteps
            | Self::DraftSlots
            | Self::NoSlots
            | Self::TokenCap => tokens,
            Self::QueryOffsets | Self::StateOffsets | Self::ChainOffsets => rows + 1,
            Self::Parameters => rows * PARAMETER_COLUMNS as usize,
            Self::Adjustments | Self::NextContext => rows * 2,
            Self::TokenContext => tokens * 2,
            Self::Commit => rows * COMMIT_FIELDS,
            Self::LiveTokens | Self::LiveRows => 1,
            Self::Extras => tokens * drafts,
            Self::Windows | Self::StateIndices => rows * (drafts + 1),
            Self::DraftTokens
            | Self::DraftSteps
            | Self::ChainPositions
            | Self::ChainLens
            | Self::ChainIndex => rows * drafts,
            Self::ChainSlots => rows * 3 * drafts,
            Self::DraftTables => rows * pages,
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
    /// The drafts a verification holds at most; zero for a runner that speculates nothing.
    drafts: usize,
    /// The pages of a row of the drafter's page table; zero without speculation.
    pages: usize,
    /// Each column's offset in the block, in the order of [`Column::ALL`].
    offsets: [u64; Column::ALL.len()],
    /// The bytes of the block.
    bytes: u64,
}

impl Layout {
    /// The layout of columns of `rows` rows and `tokens` tokens, for verifications of at most
    /// `drafts` drafts and a drafter's page table of `pages` pages a row.
    pub(crate) fn new(rows: usize, tokens: usize, drafts: usize, pages: usize) -> Self {
        let mut offsets = [0; Column::ALL.len()];
        let mut bytes = 0_u64;

        for (offset, column) in offsets.iter_mut().zip(Column::ALL) {
            *offset = bytes;

            let elements = column.elements(rows, tokens, drafts, pages) as u64;
            bytes += (elements * column.element()).next_multiple_of(ALIGNMENT);
        }

        Self {
            rows,
            tokens,
            drafts,
            pages,
            offsets,
            bytes,
        }
    }

    /// The layout that holds every bucket of `buckets`, for verifications of at most `drafts`
    /// drafts and a drafter's page table of `pages` pages a row.
    ///
    /// Each token of a verification is an attention row and a sampling row, so a verification
    /// bucket's row columns hold its tokens.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for an empty set.
    pub(crate) fn of<'a>(
        buckets: impl IntoIterator<Item = &'a Bucket>,
        drafts: u32,
        pages: u32,
    ) -> Result<Self> {
        let (mut rows, mut tokens) = (0, 0);
        for bucket in buckets {
            let sampled = match bucket.variant {
                Variant::VerifyGreedy | Variant::VerifySampled => bucket.tokens,
                _ => bucket.rows,
            };

            rows = rows.max(sampled as usize);
            tokens = tokens.max(bucket.tokens as usize);
        }

        if rows == 0 || tokens == 0 {
            return Err(Error::Invalid(
                "a causal runner with no bucket has no input tensor to lay out".into(),
            ));
        }
        Ok(Self::new(rows, tokens, drafts as usize, pages as usize))
    }

    /// The tokens a token column holds.
    pub(crate) const fn tokens(&self) -> usize {
        self.tokens
    }

    /// The drafts a verification holds at most.
    pub(crate) const fn drafts(&self) -> usize {
        self.drafts
    }

    /// The pages of a row of the drafter's page table.
    pub(crate) const fn pages(&self) -> usize {
        self.pages
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
    pub(crate) fn put<T: bytemuck::Pod>(
        &self,
        block: &mut [u8],
        column: Column,
        values: &[T],
    ) -> Result<()> {
        let source: &[u8] = bytemuck::cast_slice(values);
        let start = self.offset(column) as usize;
        let elements = column.elements(self.rows, self.tokens, self.drafts, self.pages);
        let capacity = elements * column.element() as usize;

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

/// The row request slot `slot` owns in every per-slot buffer of a runner of `slots` request
/// slots: `slot - 1` (module documentation).
///
/// # Errors
///
/// [`Error::Invalid`] for slot zero and for a slot past `slots`.
pub(crate) fn slot_row(slot: u32, slots: u32) -> Result<u32> {
    if slot == 0 || slot > slots {
        return Err(Error::Invalid(format!(
            "request slot {slot} is none of a runner's {slots}, which are numbered from one"
        )));
    }
    Ok(slot - 1)
}

/// A count the scheduler states signed, which no batch states negative.
fn count(name: &str, value: i32) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::Invalid(format!("the batch states a negative {name} of {value}")))
}

/// A call [`slot_mapping`] refused: its index among the batch's calls, and why.
pub(crate) type Refusal = (usize, String);

/// The cache slot each token of `rows` writes, as [`Column::SlotMapping`] holds it, and the
/// calls refused for a token whose entry may not be written (module documentation).
///
/// `tables` is the rank's device block table and `page_size` the tokens of a block of the
/// calls' cache groups, of which a call may write the first `groups`, the attention groups;
/// the class of cache group `g` is class `g`. A token at a negative cache
/// position writes nothing ([`NO_SLOT`]). A token at position `p` of a call on slot `s` writes
/// `u · page_size + p % page_size`, `u` the unit of the entry at position `p / page_size` of
/// the slot's table of the call's class, when that entry is writable
/// ([`BlockTables::writable`]). A call one of whose tokens names an entry that is missing or
/// read-only is returned among the refusals, with its index and the reason, and none of its
/// tokens writes.
///
/// # Errors
///
/// [`Error::Invalid`] for rows whose offsets or calls do not describe its tokens, a call that
/// writes a cache group past `groups`, or an entry whose block is not one unit: a token's cache
/// slot names one.
pub(crate) fn slot_mapping(
    rows: &Rows,
    tables: &BlockTables,
    page_size: u32,
    groups: u32,
) -> Result<(Vec<u32>, Vec<Refusal>)> {
    let tokens = rows.tokens();
    if page_size == 0 || rows.cache_positions.len() != tokens {
        return Err(Error::Invalid(format!(
            "a batch of {tokens} tokens states {} cache positions over pages of {page_size} \
             tokens",
            rows.cache_positions.len()
        )));
    }

    let offset = |row: u32| -> Result<usize> {
        rows.query_offsets
            .get(row as usize)
            .and_then(|offset| usize::try_from(*offset).ok())
            .filter(|offset| *offset <= tokens)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "row {row}'s query offset is none of the batch's {tokens} tokens"
                ))
            })
    };

    let mut mapping = vec![NO_SLOT; tokens];
    let mut refused = Vec::new();
    for (index, call) in rows.calls.iter().enumerate() {
        let (first, last) = (offset(call.rows.start)?, offset(call.rows.end)?);
        let class = u8::try_from(call.cache_group).map_err(|_| {
            Error::Invalid(format!(
                "call {index} writes cache group {}, which names no class",
                call.cache_group
            ))
        })?;

        // The call's slots, written only once every token of it is known to be writable.
        let mut written = Vec::with_capacity(last.saturating_sub(first));
        let mut refusal = None;
        for token in first..last {
            let Ok(position) = u32::try_from(rows.cache_positions[token]) else {
                continue;
            };
            if call.cache_group >= groups {
                return Err(Error::Invalid(format!(
                    "call {index} on slot {} writes cache group {}, and the forward writes {groups} \
                     attention groups",
                    call.slot, call.cache_group
                )));
            }
            let block = position / page_size;

            let reason = match tables.run(call.slot, class, block) {
                None => Some("holds no entry"),
                Some(_) if !tables.writable(call.slot, class, block) => Some("is read-only"),
                Some((_, unit, 1)) => {
                    let slot =
                        u64::from(unit) * u64::from(page_size) + u64::from(position % page_size);
                    let slot = u32::try_from(slot)
                        .ok()
                        .filter(|slot| *slot != NO_SLOT)
                        .ok_or_else(|| {
                            Error::Invalid(format!(
                                "cache slot {slot} lies outside the slots a cache column \
                                 addresses"
                            ))
                        })?;
                    written.push((token, slot));
                    None
                }
                Some((_, _, units)) => {
                    return Err(Error::Invalid(format!(
                        "the block at position {block} of class {class} of slot {} spans \
                         {units} units, and a token's cache slot names one",
                        call.slot
                    )));
                }
            };

            if let Some(reason) = reason {
                refusal = Some(format!(
                    "call {index} on slot {} writes cache position {position}, whose block at \
                     position {block} of class {class} {reason}",
                    call.slot
                ));
                break;
            }
        }

        match refusal {
            Some(reason) => refused.push((index, reason)),
            None => {
                for (token, slot) in written {
                    mapping[token] = slot;
                }
            }
        }
    }
    Ok((mapping, refused))
}

/// Writes into `block`, the bytes of the input block, the columns of `columns` padded to
/// `bucket` (module documentation).
///
/// `slots` is the runner's request slots, whose count is also the padding row of the pooled
/// page tables and of the slot state; `vocabulary` is the width of a logit row. `mapping` is
/// the cache slot each of the batch's tokens writes: [`slot_mapping`] for a serving step,
/// [`scratch_column`] for a paired forward (`paired`), and empty for a step that writes none;
/// the bucket's tokens past it write nothing.
///
/// # Errors
///
/// [`Error::Invalid`] for a batch the bucket does not hold, a row whose slot the runner holds
/// no state for, a negative token id, length or offset, or columns the sampler's encoders
/// refuse; [`Error::Invariant`] for a commit table that is not one row per call or a mapping
/// of more tokens than the bucket's.
#[expect(
    clippy::too_many_arguments,
    reason = "the runner's shape, the batch, its bucket, the cache slots it writes and the block"
)]
pub(crate) fn fill(
    layout: &Layout,
    slots: u32,
    vocabulary: u32,
    columns: &Columns,
    bucket: &Bucket,
    mapping: &[u32],
    paired: bool,
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

    if mapping.len() > bucket_tokens {
        return Err(Error::Invariant(format!(
            "a slot mapping of {} tokens for bucket {bucket}",
            mapping.len()
        )));
    }
    let mut padded = vec![NO_SLOT; bucket_tokens];
    padded[..mapping.len()].copy_from_slice(mapping);
    layout.put(block, Column::SlotMapping, &padded)?;

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
        // A row on no slot states a negative one, and keeps the padding entries.
        let Ok(slot) = u32::try_from(*held) else {
            continue;
        };
        let own = slot_row(slot, slots)?;

        sampler_slots[row] = own;
        slot_rows[row] = own;
    }
    layout.put(block, Column::Slots, &sampler_slots)?;
    layout.put(block, Column::SlotRows, &slot_rows)?;

    // A serving row's pages are its slot's row of the pooled tables; a paired forward's are
    // its own row of the scratch set's table.
    if paired {
        let own: Vec<u32> = (0..bucket.rows).collect();
        layout.put(block, Column::TableRows, &own)?;
    } else {
        layout.put(block, Column::TableRows, &slot_rows)?;
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
    let (adjustments, _) =
        sampler::adjustments(&columns.biased, &columns.banned, bucket_rows, vocabulary)?;
    layout.put(block, Column::Adjustments, &adjustments)?;

    let mut checkpoints = vec![0_u32; bucket_rows];
    for (checkpoint, held) in checkpoints.iter_mut().zip(&columns.checkpoints) {
        *checkpoint = count("state checkpoint", *held)?;
    }
    layout.put(block, Column::Checkpoints, &checkpoints)?;

    // ---- What a row's slot continues from: its token ids and its verify slots ----

    layout.put(block, Column::NextContext, &next_contexts(&offsets))?;
    let owned = layout.drafts as u32 + 1;
    layout.put(
        block,
        Column::StateIndices,
        &state_indices(&sampler_slots, slots, owned),
    )?;

    // ---- The commit: one row per call, the rest of the table inert ----

    // A paired forward commits nothing: the slots' state is the serving requests'.
    let mut table = if paired {
        sampler::commit_table(&sampler::inert_commit(calls), calls, layout.rows)?
    } else {
        sampler::commit_table(&columns.commit, calls, layout.rows)?
    };

    // The commit kernel indexes the slots' state by the table's slot field: a call's request
    // slot becomes its row, and a negative slot stays a row that stores nothing.
    for cell in &mut table[column::SLOT * layout.rows..][..calls] {
        if let Ok(slot) = u32::try_from(*cell) {
            *cell = slot_row(slot, slots)? as i32;
        }
    }
    layout.put(block, Column::Commit, &table)?;

    // The per-task columns of a step that verifies no draft: every task's token counts, no
    // task's request is continued by a predicate, and no span is accepted.
    layout.put(block, Column::Valid, &vec![1_i32; layout.rows])?;
    layout.put(block, Column::Active, &vec![1_i32; layout.rows])?;
    layout.put(block, Column::Continuation, &vec![0_u8; layout.rows])?;
    layout.put(block, Column::Points, &vec![0_i32; layout.rows])?;

    Ok(())
}

/// Where the two token ids each row's slot continues from lie in the id history of a forward
/// whose rows' packed query offsets are `offsets`, `[rows + 1]` (module documentation, token
/// ids): `[rows, 2]` row-major, each pair the earlier first.
///
/// History entries `2r` and `2r + 1` are row `r`'s context and entry `2 · rows + t` is token
/// `t`, so a row continues from its last two tokens, from its context's later id and its one
/// token, or, holding none, from its own context.
fn next_contexts(offsets: &[u32]) -> Vec<u32> {
    let rows = offsets.len().saturating_sub(1) as u32;
    let token = |token: u32| 2 * rows + token;

    let mut next = Vec::with_capacity(2 * rows as usize);
    for (row, span) in (0..rows).zip(offsets.windows(2)) {
        let (first, end) = (span[0], span[1]);

        next.extend(match end.saturating_sub(first) {
            0 => [2 * row, 2 * row + 1],
            1 => [2 * row + 1, token(first)],
            _ => [token(end - 2), token(end - 1)],
        });
    }
    next
}

/// Each state row's slots of a stepwise state field whose requests own `owned` slots each
/// (`nn::VerifySlots::state_indices`; module documentation, verify slots), `[homes, owned]`
/// row-major, for a runner of `slots` request slots.
///
/// `homes` holds each row's home row (`slot_row` of its request slot), or [`NO_SLOT`] for a
/// row on no slot, whose slots are all [`NO_SLOT`]. Column zero is the home row `h` and column
/// `q` the row `slots + h · (owned − 1) + q − 1`: the rows past the home rows, `owned − 1` a
/// request.
pub(crate) fn state_indices(homes: &[u32], slots: u32, owned: u32) -> Vec<u32> {
    let mut indices = Vec::with_capacity(homes.len() * owned as usize);

    for &home in homes {
        if home == NO_SLOT {
            indices.extend(std::iter::repeat_n(NO_SLOT, owned as usize));
            continue;
        }
        indices.push(home);
        indices.extend((1..owned).map(|step| slots + home * (owned - 1) + step - 1));
    }
    indices
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
pub(crate) fn idle(bucket: &Bucket, page_size: u32, max_kv_len: u32) -> Columns {
    let (rows, tokens) = (bucket.rows as usize, bucket.tokens as usize);
    let decode = bucket.variant.is_decode();

    // One call per row for a decode, whose row is one request's one token; one call over the
    // run for a prefill. Slot zero is no request slot: the calls name none.
    let calls: Vec<Row> = if decode {
        (0..bucket.rows)
            .map(|row| Row {
                slot: 0,
                kind: Phase::Decode,
                cache_group: 0,
                rows: row..row + 1,
            })
            .collect()
    } else {
        vec![Row {
            slot: 0,
            kind: Phase::Prefill,
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

    Columns {
        rows: Rows {
            positions: vec![0; tokens],
            cache_positions: vec![-1; tokens],
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
        ..Columns::default()
    }
}

/// The cache slots a paired forward writes for a batch: the batch's own tokens, in the scratch
/// page set's numbering, as `[tokens]` U32.
///
/// A slot is flat over one plane's pages, `page · page_size + offset`, and the scratch set
/// gives row `r` the pages `r · pages .. (r + 1) · pages`, which is what its page table
/// states. Token `t` of row `r` at cache position `p` therefore writes
/// `(r · pages + p / page_size) · page_size + p % page_size`: a slot of the row's own scratch
/// pages, and never one a serving request reads. A token the batch does not write (a negative
/// cache position) writes nothing here either and holds [`NO_SLOT`].
///
/// `cache_positions` is the batch's `[tokens]` column and `query_offsets` its `[rows + 1]`
/// packed offsets.
///
/// # Errors
///
/// [`Error::Invariant`] for a scratch set of no page or columns that describe different token
/// counts; [`Error::Invalid`] for a token whose position lies past the row's scratch pages.
pub(crate) fn scratch_slot_mapping(
    cache_positions: &[i32],
    query_offsets: &[i32],
    pages: u32,
    page_size: u32,
) -> Result<Vec<u32>> {
    if page_size == 0 || pages == 0 {
        return Err(Error::Invariant(
            "a scratch page set needs a page count and a page size above zero".into(),
        ));
    }

    let tokens = cache_positions.len();
    let closes = query_offsets
        .last()
        .is_some_and(|end| usize::try_from(*end).is_ok_and(|end| end == tokens));
    if query_offsets.len() < 2 || !closes {
        return Err(Error::Invariant(
            "the batch's offsets and cache positions describe different token counts".into(),
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
            let Ok(position) = u32::try_from(cache_positions[token]) else {
                continue;
            };
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
    cache_positions: &[i32],
    query_offsets: &[i32],
    tokens: usize,
    pages: u32,
    page_size: u32,
) -> Result<Vec<u32>> {
    if cache_positions.len() > tokens {
        return Err(Error::Invariant(format!(
            "a batch of {} tokens in a bucket of {tokens}",
            cache_positions.len()
        )));
    }

    let mut mapping = if cache_positions.is_empty() {
        Vec::new()
    } else {
        scratch_slot_mapping(cache_positions, query_offsets, pages, page_size)?
    };
    mapping.resize(tokens, NO_SLOT);
    Ok(mapping)
}
