//! Speculative decoding with the model's own drafter: the verification buckets, the drafter's
//! bucket, and the memory that joins them.
//!
//! A runner whose config states drafts ([`CausalLmConfig::drafts`](super::CausalLmConfig))
//! serves every step in two captured graphs, issued back to back on its stream:
//!
//! ```text
//! target    a prompt batch from its prefill bucket, or a verification batch from a
//!           verification bucket: each token of a verification is one attention row (the decode
//!           step at its position) and one sampling row, while one state row per verification
//!           advances the recurrent state through its tokens and keeps the state after each
//!           (the trail). The graph samples every token, accepts on the device the drafts its
//!           own samples agree with, settles the recurrent state on the trail's entry of the
//!           last accepted token, and commits (`slot_commit`: the verification's row, then a
//!           count row per accepted draft).
//! drafter   the drafter's bucket of the same rows and tokens: its entries over the positions
//!           whose next token the target left, then the drafts each row proposes, each chained
//!           step embedding the proposal of the step before it.
//! ```
//!
//! Nothing is read back between the two. What the drafter's graph and the completion read of
//! the target's is kept in memory this module owns ([`Kept`]), because a forward's own tensors
//! are the allocator's and the next forward takes them again: the target's hidden rows, its
//! chosen tokens with their scores, and each verification's agreed length. Both forwards
//! depend on their bucket alone for their launches, shapes and copies: draft counts, accepted
//! lengths, entries, the draft cache's pages and the chained positions are input columns
//! ([`fill_verification`], [`fill_drafter`]) or are computed on the device from them.
//!
//! ## Acceptance keeps the target distribution
//!
//! The drafter proposes deterministically (the argmax of its logits), so the draft
//! distribution `q` is a point mass at the draft `d`. The rejection rule (Leviathan et al.
//! 2023; vLLM's and SGLang's rejection samplers) accepts `d` with probability
//! `min(1, p(d) / q(d)) = p(d)` and on rejection draws from `norm(max(0, p - q))`, which is `p`
//! with `d` removed. Drawing `x ~ p` and accepting exactly when `x = d` has the same law:
//! `P(x = d) = p(d)`, and given `x ≠ d`, `x` is distributed as `p` without `d`. Each position
//! draws with the uniform of its request's stream at the position's address (`ipc::philox`),
//! the draw a decode at that position makes, so a speculated run of a seeded request
//! reproduces itself, and a greedy request's verification takes the argmax a decode would.
//!
//! ## On the device
//!
//! ```text
//! agrees  [T]      = chosen == Drafts            a row's last token names no draft
//! run     [R, D+1] = agrees gathered by Windows  offset j of row r, zero from its last token on
//! agreed  [R]      = argmin(run)                 the first offset that disagrees
//! last    [R]      = Firsts + agreed             the token whose sample ends the accepted prefix
//! ```
//!
//! A verification commits `agreed + 1` tokens: its agreed drafts and the sample after them.
//! The drafter's entry at offset `j` of a verification is written when `j <= agreed`, reads
//! the token chosen at that offset, and the row's first proposal reads the entry at `agreed`.
//!
//! ## The draft cache
//!
//! The drafter's keys and values lie in the pool's draft group, which has no pooled page
//! table: a row's table is stated per step, from the pages the scheduler installed for its
//! request slot ([`DraftPool`], [`DraftPages`]).
//!
//! ## What is refused
//!
//! A runner that speculates joins no expert union, serves no paired forward of a verification
//! (its token rows share one request's pages, and the scratch set gives each row its own), and
//! refuses a draft cache refill of a model that keeps a recurrent state, which would need a
//! scratch state per refill row.

use std::sync::Arc;

use ipc::execution::{
    Batch, BatchColumns, BatchRow, BatchRows, CallOutput, LogitAdjustments, RowKind,
};
use ipc::philox::DrawAddress;
use ipc::{CallStatus, DraftWork};
use nn::{KvCache, NO_SLOT, PhaseKind, RecurrentState, TokenBatch, functional};
use platform::Stream;
use proto::module::commit::{COMMIT_FIELDS, column, mode};
use proto::module::sampler::PARAMETER_COLUMNS;
use uniserve_tensor::{DType, Shape, Tensor};

use super::columns::{self, Column, Layout, slot_row};
use super::{CausalLmRunner, StateBuffers};
use crate::buckets::Bucket;
use crate::memory::allocation::Storage;
use crate::memory::cache::CacheLayout;
use crate::sampler::{self, Sampler};
use crate::{Error, Result};

/// The alignment of a kept tensor in its memory.
const ALIGNMENT: u64 = backend::allocator::ALIGNMENT;

/// The kept tensors, in the order they lie in their memory. The first twelve are the entries
/// of a sampling forward's outputs in their order (`output`).
pub(crate) const KEPT: usize = 14;
pub(crate) const CHOSEN: usize = 0;
pub(crate) const LOGPROBS: usize = 1;
pub(crate) const RANKS: usize = 2;
pub(crate) const LOGICAL: usize = 3;
pub(crate) const SAMPLING: usize = 4;
pub(crate) const TOP_VALUES: usize = 5;
pub(crate) const TOP_IDS: usize = 6;
pub(crate) const VALID: usize = 7;
pub(crate) const PROMPT_LOGPROBS: usize = 8;
pub(crate) const PROMPT_RANKS: usize = 9;
pub(crate) const PROMPT_VALUES: usize = 10;
pub(crate) const PROMPT_IDS: usize = 11;
pub(crate) const AGREED: usize = 12;
pub(crate) const HIDDEN: usize = 13;

/// The sampling parameters of a row that states none: a greedy row.
const NO_PARAMETERS: [f32; PARAMETER_COLUMNS as usize] = [0.0; PARAMETER_COLUMNS as usize];

/// How the draft cache lies in the unit pool: the cache layout's statement of its draft group,
/// read once ([`DraftPool::of`]).
///
/// A drafter has fewer layers than a unit has columns, so each of its layers lies over `span`
/// columns of a unit: a draft page of the scheduler holds `span` runs of `page` tokens, and
/// the drafter's attention reads run `c` of the draft page in unit `u` as page
/// `c · units + u`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DraftPool {
    /// The draft group's index among the cache groups.
    pub group: u32,
    /// The runs of `page` tokens one draft page holds.
    pub span: u32,
    /// Tokens of one page as the drafter's attention reads it.
    pub page: u32,
    /// Units of the pool.
    pub units: u32,
}

impl DraftPool {
    /// The draft pool of a cache of `layout` over `units` units.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a layout that keeps no draft group: the model's drafter states
    /// no cache.
    pub fn of(layout: &CacheLayout, units: u32) -> Result<Self> {
        let stated = layout.draft_group().zip(layout.draft_span());
        let Some((group, span)) = stated.filter(|(_, span)| *span > 0) else {
            return Err(Error::Invalid(
                "the deployment speculates and its cache keeps no draft group".into(),
            ));
        };

        Ok(Self {
            group: group as u32,
            span,
            page: layout.groups[group].block_size / span,
            units,
        })
    }

    /// The pages a row of the drafter's page table holds for sequences of at most
    /// `max_kv_len` keys.
    pub const fn width(&self, max_kv_len: u32) -> u32 {
        max_kv_len.div_ceil(if self.page == 0 { 1 } else { self.page })
    }

    /// The drafter's page of `position` in a table whose first draft page is `first` and
    /// whose pages lie in `units`, and whether the table holds a unit there.
    fn page_of(&self, first: u32, units: &[u32], position: u32) -> (u32, bool) {
        let tokens = self.page * self.span;
        let draft_page = position / tokens;
        let run = (position % tokens) / self.page;

        let unit = draft_page
            .checked_sub(first)
            .and_then(|held| units.get(held as usize).copied())
            .unwrap_or(0);
        (run * self.units + unit, unit != 0)
    }

    /// The cache slot the drafter writes `position` at, [`NO_SLOT`] where the table holds no
    /// unit: the entry is then not written.
    pub(crate) fn slot(&self, first: u32, units: &[u32], position: u32) -> u32 {
        match self.page_of(first, units, position) {
            (page, true) => page * self.page + position % self.page,
            _ => NO_SLOT,
        }
    }

    /// One row of the drafter's page table: the page of each of its `width` pages, the
    /// padding unit's where the table holds none.
    pub(crate) fn row(&self, first: u32, units: &[u32], width: u32, out: &mut Vec<u32>) {
        out.extend((0..width).map(|page| self.page_of(first, units, page * self.page).0));
    }
}

/// What states the pages a request slot holds in the draft group, as the scheduler last
/// installed them: its first draft page and the units of the pages from there on; `None` for a
/// slot that holds none ([`CausalLmRunner::set_draft_pages`]). A table the batch being served
/// carries is read from the batch itself.
pub type DraftPages = Box<dyn FnMut(u32) -> Option<(u32, Vec<u32>)> + Send>;

/// What a target forward leaves for the drafter's forward and for the completion, in memory
/// the runner owns: one tensor per entry of the sampling outputs, each verification's agreed
/// length, and the hidden rows (module documentation).
pub(crate) struct Kept {
    /// Each kept tensor, in the order of the entry constants.
    tensors: [Tensor; KEPT],
    /// The tensors' memory; `None` for a runner whose graphs hold no device.
    _storage: Option<Arc<Storage>>,
}

/// The shape and the element type of each kept tensor, for sampling rows `rows`, `tokens`
/// tokens, `candidates` reported candidates and hidden rows of `hidden` elements of `dtype`.
fn kept_shapes(
    rows: usize,
    tokens: usize,
    candidates: u32,
    hidden: u32,
    dtype: DType,
) -> [(Shape, DType); KEPT] {
    // A runner that reports no candidate keeps one unread column.
    let width = candidates.max(1) as usize;

    [
        (rows.into(), DType::U32),
        (rows.into(), DType::F32),
        (rows.into(), DType::I32),
        (rows.into(), DType::I32),
        (rows.into(), DType::I32),
        ((rows, width).into(), DType::F32),
        ((rows, width).into(), DType::U32),
        (rows.into(), DType::I32),
        (tokens.into(), DType::F32),
        (tokens.into(), DType::I32),
        ((tokens, width).into(), DType::F32),
        ((tokens, width).into(), DType::U32),
        (rows.into(), DType::U32),
        ((tokens, hidden as usize).into(), dtype),
    ]
}

/// Each kept tensor's offset in its memory, and the memory's bytes.
fn kept_offsets(shapes: &[(Shape, DType); KEPT]) -> Result<([u64; KEPT], u64)> {
    let mut offsets = [0; KEPT];
    let mut bytes = 0_u64;

    for (offset, (shape, dtype)) in offsets.iter_mut().zip(shapes) {
        *offset = bytes;

        let element = ops::abi::bytes(*dtype).ok_or_else(|| {
            Error::Invalid(format!(
                "hidden rows of {} have no whole-byte layout to keep",
                ops::abi::name(*dtype)
            ))
        })?;
        bytes += (shape.elem_count() as u64 * element).next_multiple_of(ALIGNMENT);
    }
    Ok((offsets, bytes))
}

impl Kept {
    /// The bytes of device memory the kept tensors take.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for hidden rows of an element type with no whole-byte layout.
    pub(crate) fn bytes(
        rows: usize,
        tokens: usize,
        candidates: u32,
        hidden: u32,
        dtype: DType,
    ) -> Result<u64> {
        Ok(kept_offsets(&kept_shapes(rows, tokens, candidates, hidden, dtype))?.1)
    }

    /// The zeroed kept tensors. `stream` is the stream the runner's graphs replay on, and
    /// `None` for graphs that hold no device: the tensors are then meta tensors.
    ///
    /// # Errors
    ///
    /// As [`Kept::bytes`]; the driver's refusal of the allocation or of the fill.
    pub(crate) fn new(
        stream: Option<&Stream>,
        rows: usize,
        tokens: usize,
        candidates: u32,
        hidden: u32,
        dtype: DType,
    ) -> Result<Self> {
        let shapes = kept_shapes(rows, tokens, candidates, hidden, dtype);
        let (offsets, bytes) = kept_offsets(&shapes)?;

        let mut tensors = Vec::with_capacity(KEPT);
        let storage = match stream {
            Some(stream) => {
                let bytes = usize::try_from(bytes).map_err(|_| {
                    Error::Invalid("the kept outputs are larger than memory".into())
                })?;
                let storage = Storage::linear(stream.device(), bytes)?;

                // The drafter's graph gathers from the kept tokens and agreed lengths by
                // columns that name no row for a prompt, and its idle forward reads the
                // hidden rows before any target wrote them: every byte starts defined.
                //
                // SAFETY: the range is the whole allocation, which this value keeps for as
                // long as the tensors over it exist.
                unsafe { platform::memset_d8(storage.address(), 0, bytes, stream) }?;

                for ((shape, dtype), offset) in shapes.into_iter().zip(offsets) {
                    // SAFETY: this value owns the storage and outlives the tensors' work:
                    // the runner that holds it holds the graphs captured over them, and
                    // only the kernels of those graphs write the tensors.
                    tensors.push(unsafe { storage.external(offset as usize, shape, dtype) }?);
                }
                Some(storage)
            }
            None => {
                let meta = backend::meta();
                for (shape, dtype) in shapes {
                    // SAFETY: a tensor over the meta device has no memory: its address is
                    // never read or written.
                    tensors.push(unsafe { backend::external(0, shape, dtype, &meta) }?);
                }
                None
            }
        };

        Ok(Self {
            tensors: tensors.try_into().map_err(|_| {
                Error::Invariant("the kept outputs hold another number of tensors".into())
            })?,
            _storage: storage,
        })
    }

    /// Kept tensor `entry`, over its whole extent.
    pub(crate) const fn tensor(&self, entry: usize) -> &Tensor {
        &self.tensors[entry]
    }

    /// Copies `value` to the head of kept tensor `entry` and returns that head: the tensor a
    /// later forward and the completion read in its place.
    ///
    /// # Errors
    ///
    /// A value of more rows, another trailing extent or another element type than the kept
    /// tensor's.
    pub(crate) fn keep(&self, entry: usize, value: &Tensor) -> Result<Tensor> {
        let rows = value.dims().first().copied().unwrap_or(0);
        let head = self.tensors[entry].narrow(0, 0, rows)?;

        head.slice_set(value, 0, 0)?;
        Ok(head)
    }
}

/// The speculation of one runner: its drafts, the draft cache it is lent, the sampler of its
/// verifications and what its forwards keep.
pub(crate) struct Speculation {
    /// The drafts a verification holds at most.
    pub(crate) drafts: u32,
    /// How the draft cache lies in the unit pool.
    pub(crate) pool: DraftPool,
    /// What the target forwards keep.
    pub(crate) kept: Kept,
    /// The selection of a verification: the runner's sampler over the same two tables, whose
    /// rows also count the drafts before the position they sample.
    pub(crate) verifier: Sampler,
    /// The draft cache as it is lent: the drafter's planes and the workspace. Its page table
    /// is a column of each forward.
    pub(crate) draft: Option<KvCache>,
    /// The trail buffers of the recurrent state, for a model that keeps one.
    pub(crate) trail: Option<StateBuffers>,
    /// What states a slot's installed draft pages.
    pub(crate) pages: Option<DraftPages>,
}

/// The agreed length of each verification and the token that ends its accepted prefix:
/// `(agreed, last)`, both `[rows]` U32 (module documentation).
///
/// - `chosen`: `[tokens]` U32, the token sampled at each token of the batch;
/// - `drafts`: `[tokens]` U32, the draft each sample is compared with, [`NO_SLOT`] for the
///   last token of a row and for a token no row holds;
/// - `windows`: `[rows, drafts + 1]` U32, the token of each row at each offset, the row's last
///   token past its end, and [`NO_SLOT`] for a row the batch does not hold;
/// - `firsts`: `[rows]` U32, each row's first token.
///
/// Every operation is one between tensors, so the same launches serve any batch of the
/// bucket.
///
/// # Errors
///
/// Columns of another shape or element type.
pub(crate) fn accept(
    chosen: &Tensor,
    drafts: &Tensor,
    windows: &Tensor,
    firsts: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (rows, width) = windows.dims2()?;

    // One where a token's sample is the draft after it. A row's last token names no draft,
    // and a gather through `NO_SLOT` reads zero, so every row's run ends in a zero.
    let agrees = chosen.eq(drafts)?;
    let run = agrees
        .index_select(&windows.flatten_all()?, 0)?
        .reshape((rows, width))?;

    // The first offset whose sample is not its draft: every draft before it agreed.
    let agreed = run.argmin(1)?;
    let last = firsts.add(&agreed)?;

    Ok((agreed, last))
}

/// Whether `call` is a verification: a verification, or a decode, which a speculating runner
/// serves as a verification of no draft.
const fn verifying(call: &BatchRow) -> bool {
    matches!(call.kind, RowKind::Verify | RowKind::Decode)
}

/// Whether `columns` verify: their calls are verifications ([`verifying`]).
///
/// # Errors
///
/// [`Error::Invalid`] for a batch that mixes prompt rows with verifications, a verification
/// that is not one row, that samples nothing or that holds more than `drafts` drafts.
pub(crate) fn verifies(columns: &BatchColumns, drafts: u32) -> Result<bool> {
    let rows = &columns.rows;
    let count = rows.calls.iter().filter(|call| verifying(call)).count();
    if count == 0 {
        return Ok(false);
    }
    if count != rows.calls.len() {
        return Err(Error::Invalid(
            "a batch mixes prompt rows with verifications".into(),
        ));
    }

    for (index, call) in rows.calls.iter().enumerate() {
        let row = call.rows.start as usize;
        let span = rows.query_offsets.get(row..row + 2);
        let tokens = span.map_or(0, |span| (span[1] - span[0]).max(0) as u32);

        // Verification `index` is row `index`: the device keeps each one's agreed length by
        // its row.
        let sampled = rows.logits_rows.get(row).is_some_and(|logits| *logits >= 0);
        if call.rows.len() != 1 || row != index || tokens == 0 || !sampled {
            return Err(Error::Invalid(format!(
                "a verification of slot {} holds {} rows of {tokens} tokens: one row carries a \
                 committed token and its drafts, and samples each",
                call.request,
                call.rows.len()
            )));
        }
        if tokens - 1 > drafts {
            return Err(Error::Invalid(format!(
                "a verification of slot {} carries {} drafts, and the deployment verifies \
                 {drafts}",
                call.request,
                tokens - 1
            )));
        }
    }
    Ok(true)
}

/// The columns of a verification batch with one row per token: each token attends as the
/// decode step at its position and samples the position after it.
///
/// A row of `n` tokens whose last query sees `end` keys becomes `n` rows of one token, token
/// `j` seeing `end - (n - 1 - j)` keys, on its row's slot, with its row's parameters and
/// adjustments, and drawing at the address of the position after it in its request's stream.
/// The calls keep their order and own their tokens' rows; the commit table stays one row per
/// call.
pub(crate) fn token_rows(columns: &BatchColumns) -> BatchColumns {
    let rows = &columns.rows;
    let tokens = rows.tokens();
    let width = PARAMETER_COLUMNS as usize;

    let mut expanded = BatchColumns {
        rows: BatchRows {
            positions: rows.positions.clone(),
            slot_mapping: rows.slot_mapping.clone(),
            query_offsets: (0..=tokens as i32).collect(),
            logits_rows: (0..tokens as i32).collect(),
            ..BatchRows::default()
        },
        input_ids: columns.input_ids.clone(),
        commit: columns.commit.clone(),
        checkpoints: columns.checkpoints.clone(),
        page_size: columns.page_size,
        max_kv_len: columns.max_kv_len,
        ..BatchColumns::default()
    };

    let adjustments = &columns.adjustments;
    let mut biased = Vec::new();
    let mut banned = Vec::new();

    for call in &rows.calls {
        let row = call.rows.start as usize;
        let span = |index: usize| rows.query_offsets.get(index).copied().unwrap_or(0).max(0);
        let (first, last) = (span(row), span(row + 1));

        let end = rows.seq_lens.get(row).copied().unwrap_or(0);
        let start = rows.positions.get(first as usize).copied().unwrap_or(0);
        let key = columns.draws.get(row).map_or(0, |draw| draw.key);

        for token in first..last {
            expanded
                .rows
                .seq_lens
                .push((end + 1 - (last - token)).max(0));
            expanded
                .slots
                .push(columns.slots.get(row).copied().unwrap_or(-1));
            expanded.parameters.extend(
                columns
                    .parameters
                    .get(row * width..(row + 1) * width)
                    .unwrap_or(&NO_PARAMETERS),
            );
            expanded.draws.push(DrawAddress {
                key,
                position: (start.max(0) + (token - first) + 1) as u64,
            });

            biased.push(adjustments.biased.get(row).copied().unwrap_or(false));
            banned.push(adjustments.banned.get(row).cloned().unwrap_or_default());
        }

        expanded.rows.calls.push(BatchRow {
            rows: first as u32..last as u32,
            ..call.clone()
        });
    }

    expanded.adjustments = LogitAdjustments {
        installs: Vec::new(),
        biased: if biased.iter().any(|flag| *flag) {
            biased
        } else {
            Vec::new()
        },
        banned: if banned.iter().any(|tokens| !tokens.is_empty()) {
            banned
        } else {
            Vec::new()
        },
    };
    expanded
}

/// The columns of an idle verification of `bucket`: every row a committed token and `drafts`
/// drafts that address no slot, page or request, in the form a batch states them.
pub(crate) fn idle_verification(
    bucket: &Bucket,
    drafts: u32,
    page_size: u32,
    max_kv_len: u32,
) -> BatchColumns {
    let (rows, tokens) = (bucket.rows as usize, bucket.tokens as usize);
    let width = drafts + 1;

    let calls: Vec<BatchRow> = (0..bucket.rows)
        .map(|row| BatchRow {
            request: 0,
            kind: RowKind::Verify,
            cache_group: 0,
            rows: row..row + 1,
        })
        .collect();

    BatchColumns {
        rows: BatchRows {
            positions: vec![0; tokens],
            slot_mapping: vec![-1; tokens],
            seq_lens: vec![0; rows],
            query_offsets: (0..=bucket.rows).map(|row| (row * width) as i32).collect(),
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

/// Writes into `block` the columns of the verification batch `columns` padded to `bucket`, a
/// verification bucket (`columns` documents each).
///
/// `expanded` is [`token_rows`] of `columns`: the generic columns are its rows, one per token.
/// The commit table holds the batch's row per call, whose task is the call's own, then one
/// count row per draft of each call, which counts when the device accepted the draft.
///
/// # Errors
///
/// As `columns::fill`; [`Error::Invariant`] for a bucket that is no verification bucket of
/// the layout's drafts.
pub(crate) fn fill_verification(
    layout: &Layout,
    slots: u32,
    vocabulary: u32,
    columns: &BatchColumns,
    expanded: &BatchColumns,
    bucket: &Bucket,
    block: &mut [u8],
) -> Result<()> {
    let drafts = layout.drafts();
    let (bucket_rows, bucket_tokens) = (bucket.rows as usize, bucket.tokens as usize);
    if drafts == 0 || bucket_tokens != bucket_rows * (drafts + 1) {
        return Err(Error::Invariant(format!(
            "bucket {bucket} is no verification of {drafts} drafts a row"
        )));
    }

    // Each token is an attention row and a sampling row.
    let rows_of_tokens = Bucket::prefill(bucket.tokens, bucket.tokens, bucket.variant);
    columns::fill(
        layout,
        slots,
        vocabulary,
        expanded,
        &rows_of_tokens,
        None,
        block,
    )?;

    let rows = &columns.rows;
    let calls = rows.calls.len();
    let tokens = rows.tokens();
    if calls > bucket_rows {
        return Err(Error::Invalid(format!(
            "{calls} verifications do not fit bucket {bucket}"
        )));
    }
    let span = |call: &BatchRow| {
        let row = call.rows.start as usize;
        let offset = |index: usize| rows.query_offsets.get(index).map_or(0, |at| *at as usize);

        (offset(row), offset(row + 1).min(tokens))
    };

    // ---- The sequences: one per call, which the recurrent state advances ----

    let mut state_offsets = vec![tokens as u32; bucket_rows + 1];
    let mut state_lens = vec![0_u32; bucket_rows];
    let mut state_slots = vec![NO_SLOT; bucket_rows];
    let mut firsts = vec![0_u32; bucket_rows];

    // ---- The drafts: what each sample is compared with and what each row counts ----

    let mut compared = vec![NO_SLOT; bucket_tokens];
    let mut extras = vec![NO_SLOT; bucket_tokens * drafts];
    let mut windows = vec![NO_SLOT; bucket_rows * (drafts + 1)];
    let mut draft_tokens = vec![NO_SLOT; bucket_rows * drafts];

    let mut table = sampler::commit_table(&columns.commit, calls, layout.rows())?;
    let stride = layout.rows();

    for (index, call) in rows.calls.iter().enumerate() {
        let row = call.rows.start as usize;
        let (first, last) = span(call);

        state_offsets[index] = first as u32;
        state_lens[index] = rows
            .seq_lens
            .get(row)
            .map_or(0, |length| (*length).max(0) as u32);
        firsts[index] = first as u32;

        let own = match columns.slots.get(row).copied().map(u32::try_from) {
            Some(Ok(slot)) => Some(slot_row(slot, slots)?),
            _ => None,
        };
        state_slots[index] = own.unwrap_or(NO_SLOT);

        // The row's drafts are its tokens after its first.
        let ids = columns.input_ids.get(first..last).unwrap_or_default();
        let drafted: Vec<u32> = ids.iter().skip(1).map(|id| (*id).max(0) as u32).collect();

        for (ahead, token) in (first..last).enumerate() {
            if let Some(draft) = drafted.get(ahead) {
                compared[token] = *draft;
            }
            extras[token * drafts..][..ahead].copy_from_slice(&drafted[..ahead]);
        }
        for offset in 0..=drafts {
            windows[index * (drafts + 1) + offset] = (first + offset.min(drafted.len())) as u32;
        }
        draft_tokens[index * drafts..][..drafted.len()].copy_from_slice(&drafted);

        // The call's own row installs the token that ends its accepted prefix, task `index`;
        // a row the scheduler states no task for installs nothing.
        let commits = table[column::TASK * stride + index] >= 0;
        if let Some(slot) = table.get_mut(column::SLOT * stride + index)
            && let Ok(stated) = u32::try_from(*slot)
        {
            *slot = slot_row(stated, slots)? as i32;
        }
        if !commits {
            continue;
        }
        table[column::TASK * stride + index] = index as i32;

        // An accepted draft joins the slot's counts before the installed token.
        let counts = table[column::PENALTIES * stride + index];
        for ahead in 0..drafted.len() {
            let task = bucket_rows + index * drafts + ahead;

            table[column::SLOT * stride + task] = table[column::SLOT * stride + index];
            table[column::MODE * stride + task] = mode::COUNT as i32;
            table[column::TASK * stride + task] = task as i32;
            table[column::PENALTIES * stride + task] = counts;
        }
    }
    state_offsets[calls..].fill(tokens as u32);

    let steps: Vec<u32> = (0..bucket_rows * drafts)
        .map(|cell| (cell % drafts) as u32)
        .collect();

    layout.put(block, Column::StateOffsets, &state_offsets)?;
    layout.put(block, Column::StateLens, &state_lens)?;
    layout.put(block, Column::StateSlots, &state_slots)?;
    layout.put(block, Column::Firsts, &firsts)?;
    layout.put(block, Column::Drafts, &compared)?;
    layout.put(block, Column::Extras, &extras)?;
    layout.put(block, Column::Windows, &windows)?;
    layout.put(block, Column::DraftTokens, &draft_tokens)?;
    layout.put(block, Column::DraftSteps, &steps)?;
    layout.put(block, Column::Commit, &table)
}

/// The last entry of drafter row `row` that a prompt batch states, as a token of the batch:
/// its last token when the batch samples the token after it or states it (`lookahead`), the
/// token before it otherwise, and `None` for a row with no token whose next one is known.
fn last_entry(rows: &BatchRows, row: usize, lookahead: Option<u32>) -> Option<usize> {
    let offset = |index: usize| {
        rows.query_offsets
            .get(index)
            .map_or(0, |at| (*at).max(0) as usize)
    };
    let (first, last) = (offset(row), offset(row + 1).min(rows.tokens()));

    let ends = rows.logits_rows.get(row).is_some_and(|logits| *logits >= 0) || lookahead.is_some();
    let entries = (last - first.min(last)).saturating_sub(usize::from(!ends));

    (entries > 0).then(|| first + entries - 1)
}

/// The draft work of the call that owns row `row` of `columns`, when the row is the call's
/// last: the row a call's lookahead and proposals belong to.
fn row_work(columns: &BatchColumns, row: usize) -> DraftWork {
    columns
        .rows
        .calls
        .iter()
        .position(|call| call.rows.end as usize == row + 1)
        .and_then(|call| columns.drafts.get(call).copied())
        .unwrap_or_default()
}

/// Writes into `block` the columns of the drafter's forward behind `columns`, padded to the
/// rows and tokens of `bucket` (`columns` documents each).
///
/// The drafter rows of a verification batch are its calls, whose entries are bounded on the
/// device by the length each call agreed; a prompt batch's drafter rows are its rows, each
/// entry reading the prompt's next token, the last one the token the row sampled or its
/// call's lookahead. `tables` is each drafter row's installed draft pages,
/// `(first page, units)`, in row order.
///
/// # Errors
///
/// [`Error::Invalid`] for a batch the bucket does not hold or a position that is none.
pub(crate) fn fill_drafter(
    layout: &Layout,
    pool: &DraftPool,
    vocabulary: u32,
    columns: &BatchColumns,
    tables: &[(u32, Vec<u32>)],
    bucket: &Bucket,
    block: &mut [u8],
) -> Result<()> {
    let drafts = layout.drafts();
    let rows = &columns.rows;
    let verify = rows.calls.iter().any(verifying);
    let (bucket_rows, bucket_tokens) = (bucket.rows as usize, bucket.tokens as usize);
    let (sequences, tokens) = (rows.rows(), rows.tokens());

    if sequences > bucket_rows || tokens > bucket_tokens || bucket_rows > layout.rows() {
        return Err(Error::Invalid(format!(
            "a batch of {sequences} rows and {tokens} tokens does not fit the drafter's bucket \
             {bucket}"
        )));
    }
    let position = |token: usize| -> Result<u32> {
        let stated = rows.positions.get(token).copied().unwrap_or(0);

        u32::try_from(stated).map_err(|_| {
            Error::Invalid(format!(
                "token {token} states the position {stated}, which the draft cache holds no \
                 entry for"
            ))
        })
    };
    let offset = |index: usize| {
        rows.query_offsets
            .get(index)
            .map_or(0, |at| (*at).max(0) as usize)
    };
    let empty = (0, Vec::new());

    // ---- The entries: one per token whose next token the step knows ----

    let mut next_ids = vec![0_u32; bucket_tokens];
    let mut next_rows = vec![NO_SLOT; bucket_tokens];
    let mut entry_rows = vec![NO_SLOT; bucket_tokens];
    let mut entry_steps = vec![1_u32; bucket_tokens];
    let mut draft_slots = vec![NO_SLOT; bucket_tokens];

    let mut state_offsets = vec![tokens as u32; bucket_rows + 1];
    let mut lens = vec![0_u32; bucket_rows];
    let mut pages = Vec::with_capacity(bucket_rows * layout.pages());

    // ---- The chained steps: one token a row at the positions after its entries ----

    let stride = layout.rows();
    let cells = 3 * drafts;
    let mut last_entries = vec![0_u32; bucket_rows];
    let mut proposal_rows = vec![NO_SLOT; bucket_rows];
    let mut chain_positions = vec![0_i32; drafts * stride];
    let mut chain_lens = vec![0_u32; drafts * stride];
    let mut chain_slots = vec![NO_SLOT; bucket_rows * cells];

    // A step that writes nothing reads its slot from a cell that holds none, whatever length
    // the row agreed: the cells from `2 · drafts - 1` on.
    let mut chain_index = vec![0_u32; drafts * stride];
    for step in 0..drafts {
        for row in 0..bucket_rows {
            chain_index[step * stride + row] = (row * cells + 2 * drafts - 1) as u32;
        }
    }

    for row in 0..sequences {
        let (first, last) = (offset(row), offset(row + 1).min(tokens));
        let (page, units) = tables.get(row).unwrap_or(&empty);
        let work = row_work(columns, row);

        state_offsets[row] = first as u32;
        if last > first {
            lens[row] = position(last - 1)? + 1;
        }

        // The entry of the batch's token that the row's first proposal continues from, before
        // the length a verification agreed.
        let base = if verify {
            for (ahead, token) in (first..last).enumerate() {
                next_rows[token] = token as u32;
                entry_rows[token] = row as u32;
                entry_steps[token] = ahead as u32;
                draft_slots[token] = pool.slot(*page, units, position(token)?);
            }
            proposal_rows[row] = row as u32;

            (last > first).then_some(first)
        } else {
            let entries =
                last_entry(rows, row, work.lookahead).map_or(0, |entry| entry + 1 - first);

            for token in first..first + entries {
                entry_steps[token] = 0;
                draft_slots[token] = pool.slot(*page, units, position(token)?);

                // The prompt's next token, the token the row samples, or the call's lookahead.
                if token + 1 < last {
                    next_ids[token] = columns.input_ids[token + 1].max(0) as u32;
                } else if rows.logits_rows.get(row).is_some_and(|logits| *logits >= 0) {
                    next_rows[token] = row as u32;
                } else {
                    next_ids[token] = work.lookahead.unwrap_or(0).min(vocabulary - 1);
                }
            }
            (entries > 0).then(|| first + entries - 1)
        };

        let Some(base) = base else {
            continue;
        };
        last_entries[row] = base as u32;
        let start = position(base)?;

        for cell in 0..2 * drafts - 1 {
            chain_slots[row * cells + cell] = pool.slot(*page, units, start + cell as u32 + 1);
        }
        let proposals = work.proposals.min(drafts as u32) as usize;
        for step in 1..drafts {
            chain_positions[step * stride + row] = (start + step as u32) as i32;
            chain_lens[step * stride + row] = start + step as u32 + 1;

            // A row that proposes fewer drafts writes nothing past its own.
            if step < proposals {
                chain_index[step * stride + row] = (row * cells + step - 1) as u32;
            }
        }
    }
    state_offsets[sequences..].fill(tokens as u32);

    for row in 0..bucket_rows {
        let (page, units) = tables.get(row).unwrap_or(&empty);
        pool.row(*page, units, layout.pages() as u32, &mut pages);
    }

    let mut chain_offsets: Vec<u32> = (0..=sequences as u32).collect();
    chain_offsets.resize(bucket_rows + 1, sequences as u32);

    layout.put(block, Column::NextIds, &next_ids)?;
    layout.put(block, Column::NextRows, &next_rows)?;
    layout.put(block, Column::EntryRows, &entry_rows)?;
    layout.put(block, Column::EntrySteps, &entry_steps)?;
    layout.put(block, Column::DraftSlots, &draft_slots)?;
    layout.put(block, Column::NoSlots, &vec![NO_SLOT; bucket_tokens])?;
    layout.put(
        block,
        Column::TokenCap,
        &vec![vocabulary - 1; bucket_tokens],
    )?;
    layout.put(block, Column::StateOffsets, &state_offsets)?;
    layout.put(block, Column::DraftLens, &lens)?;
    layout.put(block, Column::DraftTables, &pages)?;
    layout.put(block, Column::LastEntries, &last_entries)?;
    layout.put(block, Column::ProposalRows, &proposal_rows)?;
    layout.put(block, Column::ChainPositions, &chain_positions)?;
    layout.put(block, Column::ChainLens, &chain_lens)?;
    layout.put(block, Column::ChainSlots, &chain_slots)?;
    layout.put(block, Column::ChainIndex, &chain_index)?;
    layout.put(block, Column::ChainOffsets, &chain_offsets)?;
    layout.put(block, Column::LiveRows, &[sequences as u32])
}

/// Gives each call of `batch` the drafts its row proposed: `proposals` are the drafter's
/// outputs, one `[rows]` U32 tensor per chained step, read under the installed context.
///
/// A call receives as many as it asked for and the deployment drafts, from its last row, up
/// to the first that is no token of the vocabulary: a drafter row with no numeric logit
/// proposes none, and the next verification verifies those before it. A call that failed, or
/// whose row has no entry to continue from, receives none.
///
/// # Errors
///
/// [`Error::Invariant`] for fewer proposals than the calls' rows.
pub(crate) fn propose(
    batch: &Batch,
    vocabulary: u32,
    proposals: &[Tensor],
    calls: &mut [CallOutput],
) -> Result<()> {
    let columns = &batch.columns;

    let mut steps = Vec::with_capacity(proposals.len());
    for step in proposals {
        steps.push(step.flatten_all()?.to_vec1::<u32>()?);
    }

    for (index, (call, output)) in columns.rows.calls.iter().zip(calls).enumerate() {
        let work = columns.drafts.get(index).copied().unwrap_or_default();
        let Some(row) = (call.rows.end as usize).checked_sub(1) else {
            continue;
        };

        let continues = verifying(call) || last_entry(&columns.rows, row, work.lookahead).is_some();
        if output.status != CallStatus::Ok || !continues {
            continue;
        }

        for step in steps.iter().take(work.proposals as usize) {
            let token = *step.get(row).ok_or_else(|| {
                Error::Invariant(format!(
                    "the drafter proposed for {} rows and the batch holds row {row}",
                    step.len()
                ))
            })?;
            if token >= vocabulary {
                break;
            }
            output.draft_token_ids.push(token);
        }
    }
    Ok(())
}

impl CausalLmRunner {
    /// The speculation of a runner whose buckets verify or draft.
    fn speculating(&self) -> Result<&Speculation> {
        self.speculation.as_ref().ok_or_else(|| {
            Error::Invariant(
                "a verification or drafter bucket of a runner that speculates nothing".into(),
            )
        })
    }

    /// The forward of a verification bucket (module documentation). Its outputs are kept, in
    /// the order `output` states for a verification.
    pub(super) fn forward_verification(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        let speculation = self.speculating()?;
        let (rows, tokens) = (bucket.rows as usize, bucket.tokens as usize);
        let drafts = speculation.drafts as usize;

        // ---- The model: each token an attention row, each verification a state row ----

        let slots = self.column(Column::Slots, tokens)?;
        let cache = self.gathered_cache(&self.column(Column::TableRows, tokens)?)?;

        let batch = TokenBatch {
            positions: self.column(Column::Positions, tokens)?,
            slot_mapping: self.column(Column::SlotMapping, tokens)?,
            seq_lens: self.column(Column::SeqLens, tokens)?,
            query_offsets: self.column(Column::QueryOffsets, tokens + 1)?,
            logits_rows: self.column(Column::LogitsRows, tokens)?,
            phase: PhaseKind::Decode,
            page_size: self.config.page_size,
            num_tokens: bucket.tokens,
            max_kv_len: self.config.max_kv_len,
            draft_lens: None,
            rope_positions: None,
            live_tokens: Some(self.column(Column::LiveTokens, 1)?),
        };
        batch.check()?;

        let state_slots = self.column(Column::StateSlots, rows)?;
        let state = match (&self.state, &speculation.trail) {
            (Some((live, checkpoint)), Some(trail)) => Some(RecurrentState {
                query_offsets: self.column(Column::StateOffsets, rows + 1)?,
                seq_lens: self.column(Column::StateLens, rows)?,
                slots: state_slots.clone(),
                checkpoints: self.column(Column::Checkpoints, rows)?,
                live: live.clone(),
                checkpoint: checkpoint.clone(),
                trail: Some(trail.clone()),
            }),
            (Some(_), None) => {
                return Err(Error::State(
                    "a verification of a model that keeps a recurrent state was lent no trail",
                ));
            }
            (None, _) => None,
        };

        let hidden = {
            let embedded = self
                .model
                .embed_input_ids(&self.column(Column::InputIds, tokens)?)?;
            self.model
                .forward(&batch, &cache, state.as_ref(), None, &embedded)?
        };

        // ---- The selection of the token after every token ----

        let logits = self.model.compute_logits(&hidden)?;
        let parameters = self.column(Column::Parameters, (tokens, PARAMETER_COLUMNS as usize))?;
        let adjustments = self.column(Column::Adjustments, (tokens, 2))?;
        let bans = self.slots.bans().narrow(0, 0, tokens)?;
        let extras = self.column(Column::Extras, (tokens, drafts))?;

        let (chosen, valid) = if bucket.variant.greedy() {
            let chosen = speculation.verifier.greedy(
                &logits,
                &slots,
                &parameters,
                &adjustments,
                &bans,
                Some(&extras),
            )?;
            (chosen, None)
        } else {
            let (chosen, valid) = speculation.verifier.forward(
                &logits,
                &slots,
                &parameters,
                &self.column(Column::Draws, tokens)?,
                &adjustments,
                &bans,
                Some(&extras),
            )?;
            (chosen, Some(valid))
        };

        let scores = speculation.verifier.score(&logits, &chosen)?;
        drop(logits);

        // ---- The acceptance, and the recurrent state after the accepted prefix ----

        let (agreed, last) = accept(
            &chosen,
            &self.column(Column::Drafts, tokens)?,
            &self.column(Column::Windows, (rows, drafts + 1))?,
            &self.column(Column::Firsts, rows)?,
        )?;

        if let (Some((live, _)), Some(trail)) = (&self.state, &speculation.trail) {
            for (live, trail) in live.iter().flatten().zip(trail.iter().flatten()) {
                // A slot's state becomes the state its verification left after the last
                // token it accepted. A row on no slot names none, which the scatter skips.
                let settled = trail.index_select(&last, 0)?;

                let mut broadcast = vec![1; settled.rank()];
                broadcast[0] = rows;
                let index = state_slots
                    .reshape(broadcast)?
                    .broadcast_as(settled.shape())?;

                live.scatter_set(&index, &settled, 0)?;
            }
        }

        // ---- The commit: each verification's row, then a count row per draft ----

        let counts = rows * drafts;
        let ones = self.column(Column::Valid, tokens)?;

        let tasks = Tensor::cat(
            &[
                &chosen.index_select(&last, 0)?,
                &self
                    .column(Column::DraftTokens, (rows, drafts))?
                    .flatten_all()?,
            ],
            0,
        )?;
        let counted = match &valid {
            Some(valid) => Tensor::cat(
                &[&valid.index_select(&last, 0)?, &ones.narrow(0, 0, counts)?],
                0,
            )?,
            None => ones.clone(),
        };

        // A draft counts when it lies before the length its row agreed.
        let accepted = self
            .column(Column::DraftSteps, (rows, drafts))?
            .broadcast_lt(&agreed.reshape((rows, 1))?)?
            .flatten_all()?
            .to_dtype(DType::I32)?;
        let active = Tensor::cat(&[&self.column(Column::Active, rows)?, &accepted], 0)?;

        // A verification advances by its agreed drafts and the sample after them.
        let span = agreed.view(DType::I32)?.add(&ones.narrow(0, 0, rows)?)?;
        let points = Tensor::cat(&[&span, &self.column(Column::Points, counts)?], 0)?;

        let table = self
            .column(Column::Commit, (COMMIT_FIELDS, self.layout.rows()))?
            .narrow(1, 0, tokens)?;
        let [future, predicates, logical, sampling, verified] = self.slots.committed()?;

        speculation.verifier.commit_(
            &table,
            &tasks,
            &counted,
            &active,
            &self.column(Column::Continuation, tokens)?,
            &points,
            &future,
            &predicates,
            &logical,
            &sampling,
            &verified,
        )?;

        let call_slots = self.column(Column::CallSlots, rows)?;
        let logical = self.slots.logical().index_select(&call_slots, 0)?;
        let sampling = self.slots.sampling().index_select(&call_slots, 0)?;

        // ---- What the drafter and the completion read ----

        let kept = &speculation.kept;
        let mut scores = scores.into_iter();
        let mut outputs = vec![kept.keep(CHOSEN, &chosen)?];

        for (entry, score) in [LOGPROBS, RANKS].into_iter().zip(scores.by_ref()) {
            outputs.push(kept.keep(entry, &score)?);
        }
        outputs.push(kept.keep(LOGICAL, &logical)?);
        outputs.push(kept.keep(SAMPLING, &sampling)?);
        for (entry, score) in [TOP_VALUES, TOP_IDS].into_iter().zip(scores) {
            outputs.push(kept.keep(entry, &score)?);
        }
        if let Some(valid) = &valid {
            outputs.push(kept.keep(VALID, valid)?);
        }
        outputs.push(kept.keep(AGREED, &agreed)?);

        kept.keep(HIDDEN, &hidden)?;
        Ok(outputs)
    }

    /// The forward of the drafter's bucket (module documentation): `drafts` outputs, the
    /// proposal of each chained step for every row, `[rows]` U32.
    pub(super) fn forward_drafter(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        let speculation = self.speculating()?;
        let drafter = self.model.drafter().ok_or(Error::State(
            "a speculating runner's model carries its drafter",
        ))?;
        let lent = speculation
            .draft
            .as_ref()
            .ok_or(Error::State("a speculating runner was lent no draft cache"))?;

        let (rows, tokens) = (bucket.rows as usize, bucket.tokens as usize);
        let drafts = speculation.drafts as usize;
        let kept = &speculation.kept;
        let cap = self.column(Column::TokenCap, tokens)?;

        let cache = KvCache {
            block_tables: self.column(Column::DraftTables, (rows, self.layout.pages()))?,
            ..lent.clone()
        };

        // ---- The entries ----

        // The token after each entry: the one the batch states, or the one its sampling row
        // chose, held to the vocabulary so a row that chose none embeds a token.
        let chosen = kept
            .tensor(CHOSEN)
            .index_select(&self.column(Column::NextRows, tokens)?, 0)?;
        let next = self
            .column(Column::NextIds, tokens)?
            .add(&chosen)?
            .minimum(&cap)?;

        // An entry of a verification is written up to the length its row agreed.
        let bound = kept
            .tensor(AGREED)
            .index_select(&self.column(Column::EntryRows, tokens)?, 0)?;
        let written = self.column(Column::EntrySteps, tokens)?.le(&bound)?;
        let slot_mapping = written.where_cond(
            &self.column(Column::DraftSlots, tokens)?,
            &self.column(Column::NoSlots, tokens)?,
        )?;

        let lens = self.column(Column::DraftLens, rows)?;
        let batch = TokenBatch {
            positions: self.column(Column::Positions, tokens)?,
            slot_mapping,
            seq_lens: lens.clone(),
            query_offsets: self.column(Column::StateOffsets, rows + 1)?,
            logits_rows: lens,
            phase: PhaseKind::Prefill,
            page_size: speculation.pool.page,
            num_tokens: bucket.tokens,
            max_kv_len: self.config.max_kv_len,
            draft_lens: None,
            rope_positions: None,
            live_tokens: Some(self.column(Column::LiveTokens, 1)?),
        };
        batch.check()?;

        let drafted = {
            let embedded = self.model.embed_input_ids(&next)?;
            let hidden = kept.tensor(HIDDEN).narrow(0, 0, tokens)?;

            drafter.forward(&batch, &cache, &embedded, &hidden)?
        };

        // ---- The proposals: the first from each row's last entry, then the chained steps ----

        let ahead = kept
            .tensor(AGREED)
            .index_select(&self.column(Column::ProposalRows, rows)?, 0)?;
        let ahead_positions = ahead.view(DType::I32)?;
        let last = self.column(Column::LastEntries, rows)?.add(&ahead)?;

        let slots = self
            .column(Column::ChainSlots, (rows, 3 * drafts))?
            .flatten_all()?;
        let step_column = |column: Column, step: usize| -> Result<Tensor> {
            Ok(self
                .column(column, (drafts, self.layout.rows()))?
                .narrow(0, step, 1)?
                .squeeze(0)?
                .narrow(0, 0, rows)?)
        };

        let mut current = drafted.index_select(&last, 0)?;
        let mut proposals: Vec<Tensor> = Vec::with_capacity(drafts);

        for step in 0..drafts {
            if let Some(previous) = proposals.last() {
                let lens = step_column(Column::ChainLens, step)?.add(&ahead)?;
                let index = step_column(Column::ChainIndex, step)?.add(&ahead)?;

                let batch = TokenBatch {
                    positions: step_column(Column::ChainPositions, step)?.add(&ahead_positions)?,
                    slot_mapping: slots.index_select(&index, 0)?,
                    seq_lens: lens.clone(),
                    query_offsets: self.column(Column::ChainOffsets, rows + 1)?,
                    logits_rows: lens,
                    phase: PhaseKind::Decode,
                    page_size: speculation.pool.page,
                    num_tokens: bucket.rows,
                    max_kv_len: self.config.max_kv_len,
                    draft_lens: None,
                    rope_positions: None,
                    live_tokens: Some(self.column(Column::LiveRows, 1)?),
                };

                // A row whose step before proposed no token embeds one all the same: its
                // proposals end there, where the completion reads them.
                let embedded = self
                    .model
                    .embed_input_ids(&previous.minimum(&cap.narrow(0, 0, rows)?)?)?;
                current = drafter.forward(&batch, &cache, &embedded, &current)?;
            }

            proposals.push(functional::argmax(&self.model.compute_logits(&current)?)?);
        }
        Ok(proposals)
    }
}
