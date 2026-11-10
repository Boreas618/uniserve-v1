//! `CausalLmRunner`: a causal language model's batches, planned into captured buckets.
//!
//! The runner is the worker's half of §4.3's `CausalLm`: it owns the model, the generation of
//! kernels it serves, the captured graphs of both generations during an update, the arena their
//! activations live in, and the request-slot state a decode step advances. Per batch (design §9's
//! `每批的规划` row) it:
//!
//! 1. chooses the smallest captured bucket that holds the batch's shape ([`Buckets::select`]);
//! 2. pads the mechanism's host columns to that bucket and stages them into the pinned arena,
//!    which is sized for the largest bucket and never grows during a step;
//! 3. uploads the staged columns on the batch's stream and builds the device-side
//!    `nn::TokenBatch` and `nn::KvView` views over them;
//! 4. replays the bucket's graph.
//!
//! Padding follows the runner it replaces: a decode bucket pads with rows of slot 0, and a prefill
//! bucket adds one sequence holding every padding token and then empty sequences, so no padding
//! token writes the KV cache.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ipc::{PhaseId, TimingCounters};
use nn::family::TextModel;
use nn::{CausalLm, Forward, KvView, TokenBatch};
use ops::{DType, Generation, TensorView};
use platform::DeviceId;

use crate::arena::{Pools, StreamId};
use crate::batch::{
    Batch, BatchColumns, BatchId, BatchOutput, BatchRow, BatchRows, CallId, CallOutput, Job, Jobs,
    RowKind, RowState,
};
use crate::plan::{Bucket, Buckets, Shape};

#[cfg(test)]
use crate::plan::Variant;
use crate::runner::{Binding, Runner, RunnerId};
use crate::runners::columns::{Column, encode_f32, encode_i32};
use crate::runners::lifecycle::{Lifecycle, walk_capture};
use crate::runners::recording::{Footprint, Walk};
use crate::runners::sampler::{Sampler, SamplerState, Samples};

/// The capture sink, at the path the device tests of the rebind mechanism import it from.
pub use crate::runners::lifecycle::Recording;
use crate::{Error, Result};

/// The host columns of one batch, padded to a bucket and held for the upload.
///
/// The buffers are allocated once, at bind, sized for the largest bucket, so serving a batch writes
/// into them and allocates nothing. The row layout is the batch's `BatchRows`, padded: a
/// padded token carries `slot_mapping = -1` (it writes no cache page) and a padded row carries a
/// commit slot below zero (it keeps no token state).
#[derive(Default, Debug)]
pub struct Staging {
    rows: BatchRows,
    input_ids: Vec<i32>,
    slots: Vec<i32>,
    parameters: Vec<f32>,
    /// `[rows]` each row's draw address; a padded row's is the default, and its draw is discarded.
    draws: Vec<ipc::philox::DrawAddress>,
    commit: Vec<i64>,
    page_size: u32,
    max_kv_len: u32,
    /// The mechanism's call ids of the staged rows, in the order `rows.calls` lists them.
    calls: Vec<CallId>,
    /// The tokens the batch really holds, before the padding this staging appends to reach the
    /// bucket's count. A captured graph states the bucket's token count as a constant, so a kernel
    /// that must bound its work by what the batch holds reads this from the device instead.
    live_tokens: i32,
    /// The bucket the last [`Staging::stage`] padded the columns to.
    /// The bucket the held columns were padded to. A capture must reuse a batch of exactly the
    /// bucket it captures: two buckets of one variant differ in size, and a graph captured from
    /// the other one's staging would address the wrong extents.
    staged: Option<Bucket>,
}

impl Staging {
    /// Empty buffers, grown by [`Staging::reserve`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sizes the buffers for the largest bucket `buckets` holds.
    pub fn reserve(&mut self, buckets: &Buckets) -> Result<()> {
        let largest = buckets
            .iter()
            .max_by_key(|bucket| (bucket.rows, bucket.tokens))
            .copied()
            .ok_or_else(|| Error::Invalid("the runner has no bucket to serve".into()))?;
        self.input_ids = vec![0; largest.tokens as usize];
        self.slots = vec![0; largest.rows as usize];
        self.parameters =
            vec![0.0; largest.rows as usize * ops::sampler::PARAMETER_COLUMNS as usize];
        self.draws = vec![ipc::philox::DrawAddress::default(); largest.rows as usize];
        // The commit table's height is ops' business; `reserve` leaves it empty and `stage` sizes
        // it to whatever the mechanism wrote, which `check` has already validated.
        self.commit = Vec::new();
        self.rows = BatchRows::default();
        self.staged = None;
        Ok(())
    }

    /// Pads the batch's columns to `bucket` and holds them, and answers the shape that is staged.
    ///
    /// The mechanism's values are copied in order; padding is appended in the runner's layout, so
    /// a kernel reading the staged columns sees one well-formed batch of the bucket's size. A
    /// prefill's padding tokens extend the last call's token range (they write no cache page); a
    /// decode's padding rows each evaluate one padding token with a commit slot below zero.
    pub fn stage(
        &mut self,
        columns: &BatchColumns,
        call_ids: &[CallId],
        shape: &Shape,
        bucket: Bucket,
    ) -> Result<Shape> {
        if !bucket.holds(shape) {
            return Err(Error::Invalid(format!(
                "bucket {bucket} does not hold a batch of {shape}"
            )));
        }
        let tokens = columns.rows.tokens();
        self.live_tokens = i32::try_from(tokens)
            .map_err(|_| Error::Invalid("a batch of more tokens than an i32 counts".into()))?;
        // A batch has two counts and each of its columns belongs to one of them: the sampler's
        // `slots` and `parameters` are per **row** (a verify call carries several rows for one
        // call), and the commit table states one row per **call**.
        let rows = columns.rows.rows();
        let calls = columns.rows.calls.len();
        if tokens > bucket.tokens as usize || rows > bucket.rows as usize {
            return Err(Error::Invalid(format!(
                "the batch's {rows} rows and {tokens} tokens do not fit bucket {bucket}"
            )));
        }
        let bucket_tokens = bucket.tokens as usize;
        let bucket_rows = bucket.rows as usize;

        // Token columns: the mechanism's tokens, then padding that persists nowhere. Every column
        // is copied into the staging's own buffers, whose capacity was taken at bind, so a step
        // allocates nothing (4.5) and a captured graph's addresses cannot move under it.
        self.input_ids[..tokens].copy_from_slice(&columns.input_ids[..tokens]);
        for index in tokens..bucket_tokens {
            self.input_ids[index] = 0;
        }
        self.rows.clear();
        self.rows
            .positions
            .extend_from_slice(&columns.rows.positions);
        self.rows.positions.resize(bucket_tokens, 0);
        for index in tokens..bucket_tokens {
            self.rows.positions[index] = 0;
        }
        self.rows
            .slot_mapping
            .extend_from_slice(&columns.rows.slot_mapping);
        self.rows.slot_mapping.resize(bucket_tokens, -1);
        for index in tokens..bucket_tokens {
            self.rows.slot_mapping[index] = -1;
        }
        self.rows.seq_lens.extend_from_slice(&columns.rows.seq_lens);
        self.rows
            .query_offsets
            .extend_from_slice(&columns.rows.query_offsets);
        self.rows
            .logits_rows
            .extend_from_slice(&columns.rows.logits_rows);
        self.rows.calls.extend_from_slice(&columns.rows.calls);
        if self.rows.calls.is_empty() || rows > bucket_rows || calls > bucket_rows {
            return Err(Error::Invariant(format!(
                "the batch has {rows} rows in {calls} calls and does not fit {bucket_rows} rows"
            )));
        }

        // Row columns: the mechanism's rows, then rows that sample nothing and commit nothing.
        self.slots[..rows].copy_from_slice(&columns.slots[..rows]);
        let parameters = ops::sampler::PARAMETER_COLUMNS as usize;
        self.parameters[..rows * parameters]
            .copy_from_slice(&columns.parameters[..rows * parameters]);
        if columns.draws.len() != rows {
            return Err(Error::Invalid(format!(
                "the batch states {} draw addresses for its {rows} rows",
                columns.draws.len()
            )));
        }
        self.draws[..rows].copy_from_slice(&columns.draws);
        for index in rows..bucket_rows {
            self.slots[index] = -1;
            self.draws[index] = ipc::philox::DrawAddress::default();
            for column in 0..parameters {
                self.parameters[index * parameters + column] = 0.0;
            }
        }
        // A row the batch does not fill is an **empty sequence**, not a short one: it owns no token,
        // its gathered block-table row names the pool's padding page, and it therefore has no key
        // to attend to. `seq_lens` states that as a length of zero - a length of one would tell the
        // attention kernel that an absent row has a page of keys. The row names a page rather than
        // a sentinel because a vendor kernel forms a page's address before it checks the length,
        // and a `-1` there is an address before the pool's first byte. The row count itself stays the bucket's: inside a captured graph every per-row
        // operand has the bucket's extent and the batch's width is a mask over it (`logits_rows`
        // below, and the loader's `-1` slots), never the extent of an operand.
        self.rows.seq_lens.resize(bucket_rows, 0);
        for index in rows..bucket_rows {
            self.rows.seq_lens[index] = 0;
        }
        self.rows.logits_rows.resize(bucket_rows, -1);
        for index in rows..bucket_rows {
            self.rows.logits_rows[index] = -1;
        }

        // Padding tokens belong to the last row the batch staged - a row's token extent is the gap
        // between its two `query_offsets` entries - and every row the bucket holds beyond the
        // batch's own is a call of its own that owns that row and no token. A call's `rows` names
        // rows of the batch's device columns and not tokens, so padding never widens a call's range
        // past the row axis: a prefilling bucket has four rows and sixty-four tokens.
        self.rows.query_offsets.truncate(rows + 1);
        if self.rows.query_offsets.is_empty() {
            return Err(Error::Invariant(format!(
                "the batch states no query offsets for its {rows} rows"
            )));
        }
        // A call's span is its **real** tokens and padding is inside no span: the offsets end where
        // the batch's own tokens end, and the rest of the bucket's token axis belongs to no call.
        // Widening the last row's span over the padding fixed one reader - a gather that wanted the
        // last real token - and broke every other: a reader that takes a row's query length from the
        // span was told a twenty-four-token prompt is five hundred and twelve tokens of query, and
        // the token axis's operators were asked to work on four hundred and eighty-eight tokens no
        // call owns. What bounds those operators is the batch's own count (`live_tokens`), which is
        // staged beside these columns; what a span says is what the batch has.
        while self.rows.query_offsets.len() <= bucket_rows {
            let end = self
                .rows
                .query_offsets
                .last()
                .copied()
                .ok_or_else(|| Error::Invariant("the batch states no query offsets".into()))?;
            self.rows.query_offsets.push(end);
        }
        let mut owned = self
            .rows
            .calls
            .last()
            .map(|call| call.rows.end)
            .unwrap_or(0);
        while (owned as usize) < bucket_rows {
            let start = owned;
            owned += 1;
            self.rows.calls.push(BatchRow {
                request: 0,
                kind: if bucket.variant.is_decode() {
                    RowKind::Decode
                } else {
                    RowKind::Prefill
                },
                cache_group: 0,
                rows: start..owned,
            });
        }

        // The commit table states one row per call. Its padded rows are inert rows by the
        // operator's own sentinels - a slot and a task of -1 name no destination and sample
        // nothing - so a padding row commits nothing and the graph may run over the whole bucket.
        // As with the rows, the table is copied into the staging's buffer rather than cloned per
        // step, so a step writes fixed addresses (4.5).
        self.commit.clear();
        if !columns.commit.is_empty() {
            let fields = ops::commit::COMMIT_FIELDS;
            if columns.commit.len() != fields * calls {
                return Err(Error::Invariant(format!(
                    "the commit table holds {} values, {fields} fields of {calls} calls",
                    columns.commit.len()
                )));
            }
            self.commit.extend_from_slice(&inert_commit(bucket_rows));
            for field in 0..fields {
                self.commit[field * bucket_rows..field * bucket_rows + calls]
                    .copy_from_slice(&columns.commit[field * calls..(field + 1) * calls]);
            }
        }
        self.page_size = columns.page_size;
        self.max_kv_len = columns.max_kv_len;
        self.calls.clear();
        self.calls.extend_from_slice(call_ids);
        if self.calls.len() != calls {
            return Err(Error::Invariant(format!(
                "the batch states {} call ids for its {calls} calls",
                self.calls.len()
            )));
        }

        self.staged = Some(bucket);
        Ok(bucket.shape())
    }

    /// The shape staged by the last [`Staging::stage`], if any.
    pub fn staged(&self) -> Option<Bucket> {
        self.staged
    }

    /// The tokens the staged batch holds, before the padding to its bucket.
    pub fn live_tokens(&self) -> u64 {
        u64::try_from(self.live_tokens).unwrap_or(0)
    }

    /// The staged token ids, as the embedding upload reads them.
    pub fn input_ids(&self, tokens: usize) -> &[i32] {
        &self.input_ids[..tokens]
    }

    /// The staged sampler slot view.
    pub fn slots(&self, rows: usize) -> &[i32] {
        &self.slots[..rows]
    }

    /// The staged sampling parameters, row-major.
    pub fn parameters(&self, rows: usize) -> &[f32] {
        let parameters = ops::sampler::PARAMETER_COLUMNS as usize;
        &self.parameters[..rows * parameters]
    }

    /// The staged rows' draw addresses.
    pub fn draws(&self, rows: usize) -> &[ipc::philox::DrawAddress] {
        &self.draws[..rows]
    }

    /// `ops::commit`'s column table of the staged rows.
    pub fn commit(&self) -> &[i64] {
        &self.commit
    }

    /// The sampling rows, in batch order (4.4's rows the sampler writes a token for).
    pub fn logits_rows(&self) -> &[i32] {
        &self.rows.logits_rows
    }

    /// The rows of the batch as the model reads them, padded to the bucket.
    pub fn batch_rows(&self) -> &BatchRows {
        &self.rows
    }

    /// The mechanism's call ids of the staged rows, in the rows' order.
    pub fn calls(&self) -> &[CallId] {
        &self.calls
    }
}

/// The element type the head writes logits in.
///
/// A logit row is written in f32: the sampler and the read-back both read it, and the head's own
/// precision would decide a token from the last bit.
const LOGITS_DTYPE: DType = DType::F32;

/// The batch columns of every bucket of one runner, sized for the largest bucket it serves.
struct Columns {
    positions: Column,
    /// `[tokens]` i32 the input ids, which the model embeds into the forward's hidden states.
    input_ids: Column,
    slot_mapping: Column,
    /// `[tokens]` i32 the paired forward's cache slots: the same tokens, addressed in the scratch
    /// set's numbering, so a shadow walk writes no resident page (7.3).
    scratch_slots: Column,
    seq_lens: Column,
    query_offsets: Column,
    logits_rows: Column,
    /// `[rows]` the token row each row's logits are gathered from, `0` for a row that samples
    /// nothing (its id is discarded, so the gather may read any in-range row).
    indices: Column,
    parameters: Column,
    slots: Column,
    draws: Column,
    /// `[1]` i32 the tokens the batch really holds, which a kernel bounds its work by: the bucket's
    /// own token count is a constant inside a captured graph and cannot bound anything.
    live_tokens: Column,
    /// `[rows]` the sampled ids, read back after a replay: two, as the commit's positions are.
    readback: [platform::PinnedBuffer; 2],
    /// One row of logits, `[vocabulary]`, read back for a greedy request whose first token is taken
    /// on the host (see `read_sampled`): a bucket is homogeneous in its sampling path, so a prefill
    /// bucket that ends a greedy prompt cannot argmax in its graph - the graph draws.
    logits_row: platform::PinnedBuffer,
}

/// The width of the per-token columns and of the per-row columns, over the buckets a runner serves.
///
/// The two are maxima over **different** buckets and must not be read off one: a prefill bucket can
/// be long and narrow while a decode bucket is short and wide, and a column sized from whichever
/// bucket wins a tuple comparison would then be shorter than the batches that address it. Nothing
/// bounds an [`Column::extent`] view against its column, so a column one bucket short is a silent
/// read or write past the allocation rather than an error.
fn column_extents(buckets: &Buckets) -> Result<(u64, u64)> {
    if buckets.is_empty() {
        return Err(Error::Invalid("the runner has no bucket to serve".into()));
    }
    let tokens = buckets
        .iter()
        .map(|bucket| u64::from(bucket.tokens))
        .max()
        .unwrap_or(0);
    let rows = buckets
        .iter()
        .map(|bucket| u64::from(bucket.rows))
        .max()
        .unwrap_or(0);
    Ok((tokens, rows))
}

impl Columns {
    /// Moves every column a batch rewrites to device storage, so a batch can be staged behind a
    /// replay that still reads the batch before it.
    fn place(
        &mut self,
        device: DeviceId,
        pools: &mut crate::arena::Pools,
        stream: &platform::Stream,
    ) -> Result<()> {
        for column in self.staged() {
            column.place(device, pools, stream)?;
        }
        Ok(())
    }

    /// The columns every batch rewrites.
    fn staged(&mut self) -> [&mut Column; 11] {
        [
            &mut self.positions,
            &mut self.input_ids,
            &mut self.slot_mapping,
            &mut self.seq_lens,
            &mut self.query_offsets,
            &mut self.logits_rows,
            &mut self.indices,
            &mut self.parameters,
            &mut self.slots,
            &mut self.draws,
            &mut self.live_tokens,
        ]
    }

    /// Allocates the columns over the widest token count and row count `buckets` hold, with
    /// `vocabulary` as the width of the logits row the greedy first token is taken from.
    fn new(device: DeviceId, buckets: &Buckets, vocabulary: u32) -> Result<Self> {
        let (tokens, rows) = column_extents(buckets)?;
        let parameters = rows * u64::from(ops::sampler::PARAMETER_COLUMNS);
        Ok(Self {
            positions: Column::new(device, DType::I32, tokens)?,
            input_ids: Column::new(device, DType::I32, tokens)?,
            slot_mapping: Column::new(device, DType::I32, tokens)?,
            scratch_slots: Column::new(device, DType::I32, tokens)?,
            seq_lens: Column::new(device, DType::I32, rows)?,
            query_offsets: Column::new(device, DType::I32, rows + 1)?,
            logits_rows: Column::new(device, DType::I32, rows)?,
            indices: Column::new(device, DType::I32, rows)?,
            parameters: Column::new(device, DType::F32, parameters)?,
            slots: Column::new(device, DType::I32, rows)?,
            draws: Column::new(device, DType::F32, rows)?,
            live_tokens: Column::new(device, DType::I32, 1)?,
            readback: [
                platform::PinnedBuffer::new(device, (rows * 4) as usize)?,
                platform::PinnedBuffer::new(device, (rows * 4) as usize)?,
            ],
            logits_row: platform::PinnedBuffer::new(device, (u64::from(vocabulary) * 4) as usize)?,
        })
    }
}

/// The commit table of a warm pose: `COMMIT_FIELDS` columns of `rows` i64 values, every row's slot
/// and task `-1` and every other field zero.
///
/// The kernel treats a row with a negative slot as having no destination (no store of any kind) and
/// a row with a negative task as having sampled nothing, so this table commits nothing while the
/// launch that reads it is still recorded in the bucket's graph - which is what makes the graph a
/// warmup captures the graph a serving batch replays.
fn inert_commit(rows: usize) -> Vec<i64> {
    let mut table = vec![0_i64; ops::commit::COMMIT_FIELDS * rows];
    for row in 0..rows {
        table[ops::commit::column::SLOT * rows + row] = -1;
        table[ops::commit::column::TASK * rows + row] = -1;
    }
    table
}

/// The descriptor a plain decode step's commit uses, from the table's capacity and the vocabulary.
///
/// `ops::commit` checks the table's leading stride against `stride`, every operand's type against
/// `index`, and the block against `count`; the arithmetic is here, once, so a host test can state it
/// without a device and the runner cannot drift from it.
fn commit_descriptor(capacity: u64, block: u32, vocabulary: u32) -> ops::commit::CommitDescriptor {
    ops::commit::CommitDescriptor {
        // The table and the sampled ids are i32 end to end: `Sampler` emits i32 tokens, and the
        // operator checks `tokens` against `index`.
        index: DType::I32,
        verified: DType::I32,
        penalties: DType::I32,
        // The table is `[COMMIT_FIELDS, capacity]` and a column is contiguous.
        stride: capacity,
        // One next-input token per slot, in a `[slots, 1]` matrix.
        future_stride: 1,
        vocab: vocabulary,
        block,
    }
}

/// Every `slot_commit` descriptor a runner of `buckets` can launch, one per distinct block.
///
/// The commit is a runner-owned operator: no model calls it, so the epoch's program set does not
/// contain it unless the runner asks. Its descriptor carries the vocabulary and the launch block
/// (the smallest power of two covering the bucket's rows) while the table's capacity is the largest
/// bucket's rows for every bucket, so the keys a deployment needs are one per power of two its
/// buckets reach - eight for the default catalog.
pub fn commit_descriptors(
    buckets: &Buckets,
    vocabulary: u32,
) -> Vec<ops::commit::CommitDescriptor> {
    let capacity = buckets
        .iter()
        .map(|bucket| u64::from(bucket.rows))
        .max()
        .unwrap_or(1);
    let mut blocks: Vec<u32> = buckets
        .iter()
        .map(|bucket| commit_block(bucket.rows))
        .collect();
    blocks.sort_unstable();
    blocks.dedup();
    blocks
        .into_iter()
        .map(|block| commit_descriptor(capacity, block, vocabulary))
        .collect()
}

/// The commit table the device reads, from the mechanism's `[COMMIT_FIELDS, rows]` i64 table.
///
/// The device table is `[COMMIT_FIELDS, capacity]` i32: the staged rows fill its first `rows`
/// columns, and the rest keep whatever the previous batch left there behind `dim(1) >= count`.
fn commit_table(commit: &[i64], rows: usize, capacity: usize) -> Result<Vec<i32>> {
    let fields = ops::commit::COMMIT_FIELDS;
    if commit.len() != fields * rows {
        return Err(Error::Invariant(format!(
            "the commit table holds {} values, {fields} fields of {rows} rows",
            commit.len()
        )));
    }
    let mut table = vec![0_i32; fields * capacity];
    // Rows past the batch's own are **inert**, not zero: a zero slot is slot 0, a real destination,
    // so a launch that ever ran over the whole capacity would commit thirteen phantom tokens into
    // the first slot the deployment serves. The staged table already pads its own rows with the
    // operator's sentinels; this makes the tail inert as well, so the invariant survives whatever
    // `count` a future launch passes (`ops::commit`'s two sentinels).
    for row in rows.min(capacity)..capacity {
        table[ops::commit::column::SLOT * capacity + row] = -1;
        table[ops::commit::column::TASK * capacity + row] = -1;
    }
    for field in 0..fields {
        for row in 0..rows.min(capacity) {
            let value = commit[field * rows + row];
            table[field * capacity + row] = i32::try_from(value).map_err(|_| {
                Error::Invalid(format!("a commit column value {value} does not fit i32"))
            })?;
        }
    }
    Ok(table)
}

/// The block a commit launch needs for `rows` rows: the smallest power of two that covers them.
fn commit_block(rows: u32) -> u32 {
    rows.max(1).next_power_of_two()
}

/// The rows the largest bucket of `buckets` holds, which sizes the columns bound once.
fn largest_rows(buckets: &Buckets) -> Result<u32> {
    buckets
        .iter()
        .map(|bucket| bucket.rows)
        .max()
        .ok_or_else(|| Error::Invalid("the runner has no bucket to serve".into()))
}

/// The token-state commit's operands (`ops::commit`), bound once and rewritten per batch.
///
/// A batch's sampled tokens must advance its slots' state - the histories the repetition,
/// frequency and presence penalties subtract, and the positions a verify row carries. Without the
/// launch a slot's state never moves, so the second decode step is computed against an empty
/// history. It runs once per batch, after the sampler and before the read-back, inside the bucket's
/// capture, so a replay advances the state exactly once.
///
/// Every operand is allocated when the runner binds and only the table's contents change per
/// batch: the warm pose writes it with every row's slot and task `-1`, which the kernel treats as
/// inert on every store, so the graph recorded at warmup is the graph a serving batch replays.
struct CommitColumns {
    /// `[COMMIT_FIELDS, capacity]` i32, column-major: the table the mechanism's rows fill.
    table: Column,
    /// `[rows]` i32 whether a row's token is valid, and whether its slot is active.
    valid: Column,
    active: Column,
    /// `[rows]` u8 continuation flags and `[rows]` i32 accepted points, one per sampled task: a plain
    /// decode's rows use neither, and the commit reads them at the row's task all the same.
    continuation: Column,
    points: Column,
    /// `[slots]` the per-slot next-input predicate, logical position, sampling position and
    /// verification length. A decode row advances the two positions by one.
    predicates: Column,
    logical: Column,
    sampling: Column,
    verified: Column,
    /// `[slots]` i32 one next-input token per slot: a decode without speculation drafts none.
    future: Column,
    /// The table's leading stride, which the descriptor states.
    capacity: u64,
    /// The launch's block, which must cover every row of the largest bucket.
    block: u32,
    /// The slots the per-slot state columns address.
    slots: u32,
    /// `[slots]` i32 the per-slot trajectory positions, read back after a replay: the commit kernel
    /// advances them, so only the device knows where a call's slot stands. Two, because a replay's
    /// positions are read while the next replay's copy is on its way.
    readback: [platform::PinnedBuffer; 2],
}

impl CommitColumns {
    /// The commit operands of a runner whose largest bucket has `rows` rows over `slots` slots.
    fn new(device: DeviceId, rows: u32, slots: u32) -> Result<Self> {
        let capacity = u64::from(rows.max(1));
        let mut columns = Self {
            table: Column::new(
                device,
                DType::I32,
                u64::from(ops::commit::COMMIT_FIELDS as u32) * capacity,
            )?,
            valid: Column::new(device, DType::I32, capacity)?,
            active: Column::new(device, DType::I32, capacity)?,
            continuation: Column::new(device, DType::U8, capacity)?,
            points: Column::new(device, DType::I32, capacity)?,
            predicates: Column::new(device, DType::U8, u64::from(slots.max(1)))?,
            logical: Column::new(device, DType::I32, u64::from(slots.max(1)))?,
            sampling: Column::new(device, DType::I32, u64::from(slots.max(1)))?,
            verified: Column::new(device, DType::I32, u64::from(slots.max(1)))?,
            future: Column::new(device, DType::I32, u64::from(slots.max(1)))?,
            capacity,
            block: commit_block(rows),
            slots: slots.max(1),
            readback: [
                platform::PinnedBuffer::new(
                    device,
                    2 * slots.max(1) as usize * std::mem::size_of::<i32>(),
                )?,
                platform::PinnedBuffer::new(
                    device,
                    2 * slots.max(1) as usize * std::mem::size_of::<i32>(),
                )?,
            ],
        };
        // The roles a decode step leaves inert hold their starting values; only `table` is written
        // again per batch.
        let mut encoded = Vec::new();
        let ones = vec![1_i32; capacity as usize];
        encode_i32(&mut encoded, &ones);
        columns.valid.write(&encoded)?;
        columns.active.write(&encoded)?;
        let zeros = vec![0_i32; slots.max(1) as usize];
        encode_i32(&mut encoded, &zeros);
        for column in [
            &mut columns.logical,
            &mut columns.sampling,
            &mut columns.verified,
            &mut columns.future,
        ] {
            column.write(&encoded)?;
        }
        let zeros_i32 = vec![0_i32; capacity as usize];
        encode_i32(&mut encoded, &zeros_i32);
        columns.points.write(&encoded)?;
        let zeros_u8 = vec![0_u8; capacity as usize];
        columns.continuation.write(&zeros_u8)?;
        let zeros_u8 = vec![0_u8; slots.max(1) as usize];
        columns.predicates.write(&zeros_u8)?;
        Ok(columns)
    }

    /// The slots the per-slot state columns address.
    fn slots(&self) -> usize {
        self.slots as usize
    }

    /// States slot `slot`'s next input, for a token the host chose after the graph drew another.
    fn set_next_input(&self, slot: usize, token: i32, stream: &platform::Stream) -> Result<()> {
        let offset = (slot * std::mem::size_of::<i32>()) as u64;
        // SAFETY: the column holds one i32 per slot and the caller's slot is inside it; no launch
        // that reads the column is in flight, and the wait below ends the copy's use of the source.
        unsafe {
            *self.future.host.as_ptr().cast::<i32>().add(slot) = token;
            platform::memcpy_async(
                self.future.device.ptr.address() + offset,
                self.future.host.address() + offset,
                std::mem::size_of::<i32>(),
                stream,
            )?;
        }
        stream.synchronize()?;
        Ok(())
    }

    /// Moves the columns the host rewrites or reads to device storage. The others hold values no
    /// step changes, or values only the kernel touches, and stay where they are.
    fn place(
        &mut self,
        device: DeviceId,
        pools: &mut crate::arena::Pools,
        stream: &platform::Stream,
    ) -> Result<()> {
        for column in [
            &mut self.table,
            &mut self.logical,
            &mut self.sampling,
            &mut self.future,
        ] {
            column.place(device, pools, stream)?;
        }
        Ok(())
    }

    /// The device address of the per-slot trajectory positions.
    fn sampling_ptr(&self) -> ops::DevicePtr {
        self.sampling.device.ptr
    }

    /// The device address of the per-slot logical positions: the cache extent execution
    /// initialized.
    fn logical_ptr(&self) -> ops::DevicePtr {
        self.logical.device.ptr
    }

    /// The bytes one readback buffer holds, which bounds what one copy may read.
    fn readback_len(&self) -> usize {
        self.readback[0].len()
    }

    /// The host address a readback copies the trajectory positions into.
    fn readback_address(&self, turn: usize) -> u64 {
        self.readback[turn].address()
    }

    /// The host side of the readback, as the positions the copy filled.
    fn readback_ptr(&self, turn: usize) -> *const u8 {
        self.readback[turn].as_ptr()
    }

    /// Writes the staged commit table - `[COMMIT_FIELDS, rows]` i64, column-major - as the i32 table
    /// the descriptor asks for, then hands every operand to the device.
    fn stage(&mut self, commit: &[i64], staged_rows: usize, capacity: usize) -> Result<()> {
        if commit.is_empty() {
            return Ok(());
        }
        let table = commit_table(commit, staged_rows, capacity)?;
        let mut encoded = Vec::new();
        encode_i32(&mut encoded, &table);
        self.table.write(&encoded)?;
        Ok(())
    }

    /// The `[COMMIT_FIELDS, capacity]` i32 view the kernel reads.
    fn table_view(&self) -> TensorView {
        TensorView::with_strides(
            self.table.device.ptr,
            DType::I32,
            ops::Shape::new(&[ops::commit::COMMIT_FIELDS as u64, self.capacity]),
            ops::Strides::new(&[self.capacity, 1]),
        )
    }

    /// Copies the one operand a batch rewrites into its device buffer.
    fn upload(&mut self, stream: &platform::Stream) -> Result<()> {
        self.table.push(stream)
    }
}

/// Elements one program of a step's gather copies.
const GATHER_BLOCK: u32 = 1024;

/// A replay on its way: what its answer reads once its copies land.
struct Flight {
    bucket: Bucket,
    /// Where its sampler wrote; `None` for a bucket that samples nothing.
    sampled: Option<Sampled>,
    /// The instant the device segment is measured from.
    launched: Instant,
    /// The readback buffers its copies fill.
    turn: usize,
    slots: Vec<i32>,
    logits_rows: Vec<i32>,
    rows: BatchRows,
    calls: Vec<CallId>,
}

/// Where one bucket's sampler wrote its ids: the first `rows` i32 of the plan's sampler block.
#[derive(Clone, Copy)]
struct Sampled {
    /// `[rows]` i32 sampled token ids, in the bucket's row order.
    tokens: TensorView,
    /// `[rows, vocabulary]` the bucket's raw logits, kept for a bucket that may hold a greedy
    /// request's prompt ending: its first token is the argmax of that row, which the captured graph
    /// cannot produce because a bucket is homogeneous in its sampling path.
    ///
    /// `None` for a decode bucket, whose variant already states which path it takes.
    logits: Option<TensorView>,
}

/// The largest candidate of one row: a greedy request's token.
///
/// The first finite value wins ties (the lowest token id, which is what `torch.max` answers for a
/// row of distinct logits and what keeps two runs alike for a tied one), and a row whose values are
/// all non-finite answers candidate zero rather than an arbitrary index. This is the same rule the
/// captured `argmax` operator applies on the device, and it is here in one place so the host path
/// and its test cannot disagree about it.
/// How many tokens a slot had committed **before** the step whose readback is `readback`.
///
/// The commit table answers the sampling position after the batch, and the batch commits the row's
/// own token, so a row that ends a prompt answers 1 and a slot that already held four answers 5. The
/// greedy host path is valid only for the first: it argmaxes the raw logits row, and the penalties a
/// continuing slot carries are computed by the captured sampler and not by this path.
/// The classes of the rows a batch carries, as a reader of a refusal needs them.
///
/// A device's graph serves one class: a prefill evaluates a chunk of prompt tokens and a decode one
/// token per row. The mechanism partitions by class before it plans, so a batch that mixes them is a
/// planning disagreement - and the refusal says which rows were which, not only that two classes met.
fn row_classes(rows: &BatchRows) -> String {
    if rows.calls.is_empty() {
        return "no rows".to_owned();
    }
    let mut classes = Vec::with_capacity(rows.calls.len());
    for call in &rows.calls {
        let kind = match call.kind {
            RowKind::Prefill => "prefill",
            RowKind::Decode => "decode",
            RowKind::Verify => "verify",
        };
        classes.push(format!(
            "rows {}..{} {kind}",
            call.rows.start, call.rows.end
        ));
    }
    let mixed = rows
        .calls
        .windows(2)
        .any(|pair| pair[0].kind != pair[1].kind);
    match mixed {
        true => format!("a batch of two classes: {}", classes.join(", ")),
        false => classes.join(", "),
    }
}

fn committed_before(readback: i32) -> usize {
    usize::try_from(readback).unwrap_or(0).saturating_sub(1)
}

/// How a prompt-ending row's first token is decided (4.4), from the batch's own staged figures.
///
/// A prefill bucket's job is the KV cache; the sampler is a decode operator, and the prompt ending
/// is the one token a prefill can owe the mechanism. Three cases, and only the first needs the host:
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptToken {
    /// The row ends no prompt: whatever the graph drew for it is discarded.
    Discarded,
    /// A greedy row that has committed nothing: the host takes the argmax of its own logits row,
    /// which is the same token the captured sampler would take - every penalty is a no-op for a
    /// slot with no history.
    HostArgmax,
    /// A sampled row: its first token is the graph's draw over its penalized logits, not an argmax.
    Draw,
}

/// The decision for one row, and the refusal when the row cannot be decided here.
///
/// `readback` is the slot's sampling position **after** this batch, and this batch committed the
/// row's own token, so the figure the decision needs is [`committed_before`] of it. A greedy row
/// with earlier commits is refused by name: its token is the argmax of *penalized* logits, which
/// only the captured sampler computes, and argmaxing the raw row would silently answer a different
/// request's token.
fn prompt_token(
    samples_nothing: bool,
    temperature: f32,
    slot: i32,
    readback: i32,
) -> Result<PromptToken> {
    if samples_nothing {
        return Ok(PromptToken::Discarded);
    }
    if temperature != 0.0 {
        return Ok(PromptToken::Draw);
    }
    let committed = committed_before(readback);
    if committed > 0 {
        return Err(Error::Invalid(format!(
            "a greedy row of slot {slot} ends a prompt that already holds {committed} \
             committed token(s): its token is the argmax of its penalized logits, which only \
             the captured sampler computes"
        )));
    }
    Ok(PromptToken::HostArgmax)
}

/// The bytes the trtllm-gen paged-attention program takes as its workspace: its `workspace_size`
/// argument. A forward reserves them from its arena ([`KvView::workspace`]) and the model clears
/// the decode's arrival counters at its head once per forward.
const ATTENTION_WORKSPACE_BYTES: u64 = 512 << 20;

/// Whether a replay walks its batch eagerly instead of launching the bucket's graph.
///
/// A diagnosis of one question: is a wrong answer the batch the runner staged, or the graph it
/// launched? Both are answered by the same run, because the walk uses the same columns, the same arena
/// and the same programs. It is never a serving path: an eager walk is what capture does, and a
/// deployment that ran it would launch every operator again on every batch.
fn eager_replay() -> bool {
    static EAGER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EAGER.get_or_init(|| {
        std::env::var_os("UNISERVE_EAGER_REPLAY").is_some_and(|value| !value.is_empty())
    })
}

/// The largest logit that is not `best`'s, which is the gap a first-token choice was made by.
fn runner_up(values: &[f32], best: usize) -> f32 {
    let mut second = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if index != best && value.is_finite() && *value > second {
            second = *value;
        }
    }
    values.get(best).copied().unwrap_or(f32::NAN) - second
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0_usize;
    for (index, value) in values.iter().enumerate() {
        if value.is_finite() && (*value > values[best] || !values[best].is_finite()) {
            best = index;
        }
    }
    best
}

/// The sampling step's own device state: the slot histories its penalties read.
struct Sampling {
    sampler: Sampler,
    /// `[slots, vocabulary]` i32 counts of the tokens each slot has already seen, which the
    /// penalties subtract; zeroed once when the runner binds and advanced by `ops::commit`.
    counts: TensorView,
    _counts: platform::DeviceAllocation,
}

impl Sampling {
    /// The sampler and its zeroed count table for `slots` request slots over `vocabulary` tokens.
    ///
    /// A request may ask for any top-k it likes, so the candidate width is the vocabulary: a
    /// narrower buffer would silently sample a different set.
    fn new(
        device: DeviceId,
        stream: &platform::Stream,
        vocabulary: u32,
        slots: u32,
    ) -> Result<Self> {
        let sampler = Sampler::new(vocabulary, slots, vocabulary, 0)?;
        let bytes = u64::from(slots) * u64::from(vocabulary) * 4;
        let allocation = platform::DeviceAllocation::linear(
            device,
            usize::try_from(bytes).map_err(|_| {
                Error::Invalid("the sampler's count table is larger than memory".into())
            })?,
        )?;
        // The table starts at zero: a slot has seen no token before its first call. A fresh device
        // allocation holds whatever its pages held last, so the zeros are written by the device,
        // stream-ordered before the first sample reads a count.
        // SAFETY: the range is this allocation's own `slots × vocabulary` i32 words, and the
        // allocation outlives the stream's work on it (the runner owns both).
        unsafe {
            platform::memset_d32(
                allocation.address(),
                0,
                usize::try_from(bytes / 4).map_err(|_| {
                    Error::Invalid("the sampler's count table is larger than memory".into())
                })?,
                stream,
            )?;
        }
        stream.synchronize()?;
        let counts = TensorView::new(
            ops::DevicePtr::new(allocation.address()),
            DType::I32,
            ops::Shape::new(&[u64::from(slots), u64::from(vocabulary)]),
        );
        Ok(Self {
            sampler,
            counts,
            _counts: allocation,
        })
    }
}

/// The refusal a provider that cannot install a batch's block tables gives, over the tables alone.
///
/// A batch whose slots have no pages yet carries none, and no provider needs anything for it; one
/// whose slots do carry tables needs them written into the pooled table before the gather reads it.
/// A provider that dropped them would gather the slot's previous pages, so the non-empty case is
/// refused by name until the provider holds a cache.
fn install_block_tables_unavailable(tables: &[ipc::BlockTable]) -> Result<()> {
    match tables.first() {
        None => Ok(()),
        Some(table) => Err(Error::Invalid(format!(
            "this runner holds no cache to install request slot {}'s row of group {} into, and a \
             gather would read whatever that row held before: the cache provider is the device \
             worker's, attached before the first capture",
            table.request_pool_idx, table.group_id
        ))),
    }
}

/// The refusal a provider that cannot apply a batch's cache units gives, over the allocation alone.
///
/// A batch that reads a fresh prefix carries none, and no provider needs anything for it; one that
/// reuses a cached prefix carries the copies that put the prefix in its own pages, and a provider
/// that dropped them would serve the wrong tokens. So the empty case is served by everyone and the
/// non-empty one is refused by name until the provider holds a cache.
fn apply_cache_units_unavailable(units: &[ipc::CacheUnitAllocation]) -> Result<()> {
    if units.is_empty() {
        return Ok(());
    }
    Err(Error::Unsupported(
        "this runner's cache provider does not apply a batch's cache units, so a request reusing a \
         cached prefix would read pages that were never copied"
            .into(),
    ))
}

/// What the runner needs from the block's resident KV cache to build a batch's `nn::KvView`.
///
/// The cache is the memory owner's (8's resident pages and its block tables); the runner only
/// reads it. `scratch` answers a page set of the paired forward's own, whose pages are never a
/// serving request's (7.3).
pub trait KvProvider: Send + Sync {
    /// Gives the provider the device's one pool set, before its first gather.
    ///
    /// The runner's provider gathers block tables into columns of its own, and those columns are
    /// carved from the device's allocator (4.1); the device actor creates that allocator, so it is
    /// handed over here rather than at construction. A provider that maps nothing may ignore it.
    fn attach_pools(&mut self, pools: Arc<std::sync::Mutex<crate::arena::Pools>>) -> Result<()> {
        let _ = pools;
        Ok(())
    }

    /// Where a graph gathers `rows` rows of `group`'s block table: the pooled table, indexed by
    /// slot, and the table attention reads. `None` from a provider whose table needs no gathering.
    fn table_source(&self, group: usize, rows: u32) -> Result<Option<(TensorView, TensorView)>> {
        let _ = (group, rows);
        Ok(None)
    }

    /// The cache this provider reads, when it reads one: the handle a sleep gives back and a wake
    /// maps again (2.5).
    ///
    /// One call per cache, not per runner: two runners of a device that share one cache must be
    /// woken once, and the caller dedups by the handle's address.
    fn cache(&self) -> Option<&Arc<std::sync::Mutex<crate::memory::prefix_cache::PrefixCache>>> {
        None
    }

    /// Applies what the mechanism allocated for this batch's cache pages, on `stream`, before the
    /// batch's forward reads them.
    ///
    /// Two things happen here and nowhere else: the units the mechanism acquired for the request's
    /// slot become the cache's own (an imported unit's flags say it holds bytes, so a later read
    /// does not treat it as unwritten), and every `(source, destination)` pair is copied - the
    /// request reuses a leased prefix, and a destination that still held an older page would serve
    /// the wrong tokens. The source's lease is the mechanism's and runs to completion; the copy
    /// reads it and never writes it.
    ///
    /// Issuing on the batch's stream is what orders the copy before the graph: the forward that
    /// reads those pages runs on the same stream afterwards, so no barrier is needed and none is
    /// taken (a barrier here would serialize the batch's own work).
    ///
    /// A provider that holds no cache refuses an allocation rather than dropping it: a batch whose
    /// prefix copies were dropped serves the wrong pages, and a silent drop is the one failure this
    /// seam must not have.
    fn apply_cache_units(
        &mut self,
        units: &[ipc::CacheUnitAllocation],
        stream: &platform::Stream,
    ) -> Result<()> {
        let _ = stream;
        apply_cache_units_unavailable(units)
    }

    /// Installs the batch's slot rows into the cache's pooled table, on the batch's stream.
    ///
    /// The gather [`Self::block_table`] performs later copies **row `slot` of the pooled table** for
    /// every request slot the batch names. That row is the only place a request's pages live for the
    /// gather, and nothing else writes it: a batch whose rows are not installed gathers whatever the
    /// slot held before - zeros, or another request's pages - and the graph then runs clean against
    /// the wrong memory. It is the one failure that looks like a model defect rather than a cache
    /// one, so a provider that holds no cache refuses a table rather than dropping it.
    ///
    /// It is issued before [`Self::apply_cache_units`]: the row must exist before the graph reads it,
    /// and the acquired units are this cache's before anything is copied into them.
    fn install_block_tables(
        &mut self,
        tables: &[ipc::BlockTable],
        stream: &platform::Stream,
    ) -> Result<()> {
        let _ = stream;
        install_block_tables_unavailable(tables)
    }

    /// Records the batch's page copies on the cache's copy stream, and makes `stream` wait for the
    /// loads the forward reads. Answers the fence of the offloads, which the batch's output waits
    /// on. A provider that holds no cache refuses a copy rather than dropping it.
    fn issue_pages(
        &mut self,
        loads: &[ipc::PageCopy],
        offloads: &[ipc::PageCopy],
        stream: &platform::Stream,
    ) -> Result<Option<Arc<platform::Event>>> {
        let _ = stream;
        if loads.is_empty() && offloads.is_empty() {
            return Ok(None);
        }
        Err(Error::Invalid(
            "a batch copies pages and this runner holds no cache".to_owned(),
        ))
    }

    /// How many attention groups the model's layers index.
    fn groups(&self) -> usize;

    /// Tokens per cache page.
    fn page_size(&self) -> u32;

    /// Element type of the KV planes.
    fn dtype(&self) -> DType;

    /// The block table of `group` for the batch's rows, `[rows, max_pages]` i32.
    ///
    /// `slots[r]` is row `r`'s request slot, in the batch's row order and exactly as many entries as
    /// the bucket has rows (padding included); a row whose slot is negative has no table and its
    /// row of the view is filled with `-1`, not an error. `max_pages` is the attention
    /// descriptor's bound (`max_kv_len / page_size`), so the view has a fixed shape the captured
    /// graph can address, and it is assembled on `stream` before the replay reads it.
    fn block_table(
        &self,
        group: usize,
        slots: &[i32],
        stream: &platform::Stream,
    ) -> Result<TensorView>;

    /// The same table's view over `slots`' rows, with no copy issued.
    ///
    /// A capture records every copy with the addresses it had, so a gather inside a recording bakes
    /// the pose's pool rows into the graph, and each replay would then overwrite the batch's table
    /// with them. A recording addresses the table through this view, and `block_table` fills it
    /// before each launch.
    fn block_table_view(&self, group: usize, slots: &[i32]) -> Result<TensorView>;

    /// The plane views of `group`'s layers, in model order.
    fn planes(&self, group: usize) -> Result<Vec<nn::KvPlanes>>;

    /// Gives the provider the bytes of the device's scratch pool its page set may map.
    fn attach_scratch_bytes(&mut self, bytes: u64) {
        let _ = bytes;
    }

    /// The pages a paired forward's rows may use between them.
    fn scratch_capacity(&self) -> u32 {
        0
    }

    /// A scratch page set of `pages` pages per row for `group`, and its table over `slots`
    /// (`block_table`'s order and `-1` rule), holding each row's first `lengths` cached keys.
    ///
    /// The planes carry their own scales, as the resident ones do: an FP8 group's scratch pages are
    /// FP8 pages with a scale each, so the paired forward's kernels read a page set shaped exactly
    /// like the one they serve.
    fn scratch(
        &self,
        group: usize,
        slots: &[i32],
        lengths: &[i32],
        pages: u32,
        stream: &platform::Stream,
    ) -> Result<(TensorView, Vec<nn::KvPlanes>)>;
}

/// A batch's cache installation, carried from the submitting thread to the device thread.
///
/// The `block_tables` are the rows the gather reads and the `units` the pages the mechanism
/// allocated this batch; both are issued on the runner's stream by the replay, which is the one
/// thread that may issue there (2.4).
#[derive(Debug)]
struct PendingInstall {
    /// The batch the rows belong to: a replay of any other batch must not install them.
    batch: BatchId,
    /// The batch's slot rows, `[rows, max_pages]` in the cache's pooled table.
    tables: Vec<ipc::BlockTable>,
    /// The units the mechanism allocated for this batch's pages, with their copies.
    units: Vec<ipc::CacheUnitAllocation>,
    /// The host KV cache's pages the batch's new units load.
    loads: Vec<ipc::PageCopy>,
    /// The published pages the batch copies to the host KV cache.
    offloads: Vec<ipc::PageCopy>,
}

/// A causal language model's runner: one block's decode and prefill phases.
pub struct CausalLmRunner<M: TextModel> {
    /// The worker's name for this runner.
    id: RunnerId,
    /// The phases it serves, in scheduler order.
    phases: Vec<PhaseId>,
    /// The model the runner calls.
    model: M,
    /// The generations this runner holds and the graphs captured for each (4.5, 7.3).
    ///
    /// Capturing, rebinding, switching over, reverting and purging are the same rules for every
    /// runner, so they live in [`Lifecycle`] and a runner supplies only the forward a bucket runs.
    lifecycle: Lifecycle,
    /// Every bucket the deployment captures, in capture order.
    buckets: Buckets,
    /// The device's graph pools and each bucket's static activation plan, shared with every other
    /// runner of this device (one pool per device, never per runner); the worker attaches them
    /// before the first capture.
    pools: Option<Arc<std::sync::Mutex<Pools>>>,
    /// The device's counters, when the worker handed them over (7.3's metrics).
    counters: Option<Arc<crate::device::SharedCounters>>,
    /// The device's record of the operators an update could not compare, each with its reason
    /// (7.3 step 2). The control thread reports it when the precopy finishes, so an update that
    /// has no evidence for an operator says so rather than leaving it to the log.
    uncompared: Option<Arc<std::sync::Mutex<Vec<String>>>>,
    /// The host columns of the batch being staged.
    staging: Staging,
    /// The device the runner computes on.
    device: DeviceId,
    /// The stream the worker issues this runner's work on: a clone of the device worker's
    /// primary stream. Only the worker calls into the runner, so holding it here does not make
    /// the runner a second issuer.
    stream: platform::Stream,
    /// Where shadow execution's twins live, allocated at bind, before the first capture.
    shadow: Option<crate::shadow::Shadow>,
    /// How many request slots the deployment serves: the height of the sampler's penalty-count
    /// table and of the token-state commit's table.
    slots: u32,
    /// The bound on the keys a row reads: the attention descriptor's reduction length, the
    /// deployment's longest sequence.
    max_kv_len: u32,
    /// Every bucket's per-step columns, whom the captured graphs address directly.
    columns: Columns,
    /// The token-state commit's operands, written per batch and launched inside the capture.
    commit: CommitColumns,
    /// The sampling step's own state, when any bucket samples.
    sampling: Option<Sampling>,
    /// The sampling row's output views, per bucket's plan, written by `sample` and read by the
    /// replay's readback. Interior mutability because the walk's KV closure borrows the runner
    /// while it runs.
    bucket_samples: std::cell::RefCell<HashMap<Bucket, Sampled>>,
    /// The host encode buffer every column is written through, so a step allocates no `Vec`.
    encoded: Vec<u8>,
    /// The gather indices a step writes, in the runner's own buffer.
    indices: Vec<i32>,
    /// The sampling draws a step writes, in the runner's own buffer.
    draws: Vec<f32>,
    /// Where the batch's KV views come from: the resident cache, and a paired forward's scratch.
    kv: Box<dyn KvProvider>,
    /// Whether this runner has served a batch. A bucket captured before that may be warmed by an
    /// eager walk; one captured afterwards may not, because the walk would run a forward over an
    /// arena a serving bucket owns.
    served: bool,
    /// Whether a capture is in flight: a probe that copies and synchronises is legal between
    /// launches and **not** inside a recording, where a synchronise is refused by the driver and a
    /// failed one can leave the recording in a state it never recovers from.
    capturing: bool,
    /// The batch's cache rows and page copies, staged at submit and performed by the device
    /// thread before the replay reads them (2.4).
    ///
    /// A table's installation is work on this runner's stream exactly as a column's upload is, and
    /// only the device thread issues there. Installed from the submitting thread it races the
    /// previous batch's replay, and the pages batch N+1 writes land in the row batch N is reading.
    /// Submit therefore states which batch the rows belong to, and a replay of another batch
    /// refuses rather than installing the wrong request's pages.
    pending_install: Option<PendingInstall>,
    /// The fence of the running batch's page offloads, until its output waits on it.
    offload_fence: Option<Arc<platform::Event>>,
    /// The replays launched and not yet answered, oldest first.
    flights: std::collections::VecDeque<Flight>,
    /// The readback buffers the next replay fills.
    turn: usize,
    /// Whether the columns were moved to device storage, which the first capture does.
    placed: bool,
    /// The bucket the last warm walk made ready, without its expert step size.
    warmed: Option<Bucket>,
    /// This rank's member of the expert union, on a deployment whose experts are split.
    experts: Option<super::experts::Member>,
    /// The tensor group this rank's model is a member of: every forward issues the group's
    /// collectives, the partial sums after the row-parallel products and the gathered logits.
    tensor: Option<std::sync::Arc<nn::Group>>,
}

/// One model object shared by the runners of every device a rank serves.
///
/// A rank's runners must all read the same weights, and a model is not copyable (it holds the
/// checkpoint's mapped tensors), so the engine gives each device's runner a share of the one model
/// it loaded. Delegation keeps the model type out of the worker: a runner is generic over
/// `nn::family::TextModel`, and this is that trait over a shared object.
pub struct Shared<M: ?Sized>(Arc<M>);

impl<M: ?Sized> Shared<M> {
    /// The shared model.
    pub fn new(model: Arc<M>) -> Self {
        Self(model)
    }
}

impl<M: ?Sized> Clone for Shared<M> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<M: CausalLm + ?Sized> CausalLm for Shared<M> {
    fn embed_input_ids(
        &self,
        f: &nn::Forward,
        ids: ops::TensorView,
        out: ops::TensorView,
    ) -> nn::Result<()> {
        self.0.embed_input_ids(f, ids, out)
    }

    fn forward(
        &self,
        f: &nn::Forward,
        batch: &nn::TokenBatch,
        cache: &nn::KvView<'_>,
        hidden: ops::TensorView,
    ) -> nn::Result<()> {
        self.0.forward(f, batch, cache, hidden)
    }

    fn compute_logits(
        &self,
        f: &nn::Forward,
        hidden: ops::TensorView,
        out: ops::TensorView,
    ) -> nn::Result<()> {
        self.0.compute_logits(f, hidden, out)
    }

    fn join_expert_step(&self, f: &nn::Forward) -> nn::Result<()> {
        self.0.join_expert_step(f)
    }
}

impl<M: TextModel + ?Sized> TextModel for Shared<M> {
    fn hidden(&self) -> u32 {
        self.0.hidden()
    }

    fn vocabulary(&self) -> u32 {
        self.0.vocabulary()
    }

    fn dtype(&self) -> DType {
        self.0.dtype()
    }

    fn cache(&self) -> nn::family::CacheShape {
        self.0.cache()
    }

    fn experts(&self) -> Option<nn::family::ExpertShape> {
        self.0.experts()
    }
}

impl<M: TextModel> CausalLmRunner<M> {
    /// A runner for the block's phases on `device`, serving `generation`.
    ///
    /// The model states its hidden width, activation type and vocabulary, and `kv` its page size;
    /// `slots` is the deployment's request slots and `max_kv_len` its longest sequence, the
    /// attention descriptor's static key bound. It captures nothing here: the device worker's
    /// warmup captures every bucket after the runner is bound, largest first, so the pools' plans
    /// exist before serving starts.
    pub fn new(
        binding: Binding,
        model: M,
        generation: Arc<Generation>,
        buckets: Buckets,
        kv: Box<dyn KvProvider>,
        slots: u32,
        max_kv_len: u32,
    ) -> Result<Self> {
        let Binding {
            id,
            phases,
            device,
            stream,
        } = binding;
        let mut staging = Staging::new();
        staging.reserve(&buckets)?;
        let commit_rows = largest_rows(&buckets)?;
        let columns = Columns::new(device, &buckets, model.vocabulary())?;
        // One contract, checked once, before any bucket is captured: the attention program states
        // how many pages a row of the block table holds (`ceil(max_kv_len / page_size)`), and the
        // cache allocates rows of its own width. A program that states more pages than a row holds
        // addresses entries the row does not own - the next slot's pages, or past the table - and
        // the page index it reads there is a unit no descriptor covers. That reads as an illegal
        // address in the FMHA kernel, one batch in, and nothing on the host can see it.
        if let Some(cache) = kv.cache() {
            let width = cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .tables()
                .width();
            let pages =
                usize::try_from(max_kv_len.div_ceil(kv.page_size().max(1))).unwrap_or(usize::MAX);
            if width < pages {
                return Err(Error::Invariant(format!(
                    "the cache's block-table rows are {width} pages wide and the attention \
                     program's key bound needs {pages} (max_kv_len {} over {} tokens per page): a row \
                     this program addresses reads pages it does not own",
                    max_kv_len,
                    kv.page_size()
                )));
            }
            tracing::info!(
                width,
                pages,
                max_kv_len = max_kv_len,
                page_size = kv.page_size(),
                "the block table is wide enough for the pages the attention program addresses"
            );
        }
        let sampling = if buckets.iter().any(|bucket| bucket.variant.samples()) {
            Some(Sampling::new(device, &stream, model.vocabulary(), slots)?)
        } else {
            None
        };
        Ok(Self {
            id,
            phases,
            model,
            lifecycle: Lifecycle::new(id, Arc::clone(&generation), stream.clone()),
            buckets,
            pools: None,
            counters: None,
            uncompared: None,
            staging,
            device,
            stream,
            shadow: None,
            slots,
            max_kv_len,
            columns,
            commit: CommitColumns::new(device, commit_rows, slots)?,
            sampling,
            bucket_samples: std::cell::RefCell::new(HashMap::new()),
            served: false,
            capturing: false,
            pending_install: None,
            offload_fence: None,
            flights: std::collections::VecDeque::new(),
            turn: 0,
            placed: false,
            warmed: None,
            experts: None,
            tensor: None,
            encoded: Vec::new(),
            indices: Vec::new(),
            draws: Vec::new(),
            kv,
        })
    }

    /// Makes this runner a member of an expert union: every step goes through the union, and the
    /// model's expert layers run over the member's transport.
    #[must_use]
    pub fn in_union(mut self, member: super::experts::Member) -> Self {
        self.experts = Some(member);
        self
    }

    /// Makes this runner a rank of a tensor group: its model holds this rank's block of every
    /// layer, and every forward issues the group's collectives.
    #[must_use]
    pub fn in_tensor_group(mut self, group: std::sync::Arc<nn::Group>) -> Self {
        self.tensor = Some(group);
        self
    }

    /// The pools' books, for the caller that reports mapped bytes against the budget.
    fn pool(&self) -> Result<std::sync::MutexGuard<'_, Pools>> {
        self.pools
            .as_ref()
            .map(|pools| {
                pools
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
            })
            .ok_or_else(|| Error::State("the runner has no graph pools yet"))
    }

    /// Installs the shadow harness, sized for the largest bucket's shadow set.
    pub fn set_shadow(&mut self, shadow: crate::shadow::Shadow) {
        self.shadow = Some(shadow);
    }

    /// Records that a program-layer update could not compare `entry` (7.3 step 2), once.
    ///
    /// The same operator is shadowed once per bucket the update's requests occupy, so the record
    /// keeps the first statement of each fact and drops its repeats.
    fn note_uncompared(&self, entry: String) {
        let Some(sink) = &self.uncompared else {
            return;
        };
        let mut sink = sink
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !sink.contains(&entry) {
            sink.push(entry);
        }
    }

    /// Starts a new update's record of what it could not compare.
    fn clear_uncompared(&self) {
        let Some(sink) = &self.uncompared else {
            return;
        };
        sink.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }

    /// The shape of the bucket a batch is served from.
    fn bucket_for(&self, shape: &Shape) -> Result<Bucket> {
        let bucket = self.buckets.require(shape)?;
        // A media bucket is refused by name rather than replayed: its graph is shaped for a latent
        // canvas or a raster, and a catalogue that mixed the two families must read as the mistake
        // it is, not as wrong tokens (`plan::Bucket::causal_lm`).
        bucket.causal_lm()?;
        Ok(bucket)
    }
}

impl<M: TextModel> Runner for CausalLmRunner<M> {
    fn cache(&self) -> Option<Arc<std::sync::Mutex<crate::memory::prefix_cache::PrefixCache>>> {
        self.kv.cache().cloned()
    }

    fn phases(&self) -> &[PhaseId] {
        &self.phases
    }

    fn stream(&self) -> Option<&platform::Stream> {
        Some(&self.stream)
    }

    /// A replay is launched behind the one before it, because the driver takes the device idle for
    /// as long as a launch lasts when it waits for that replay to end. An eager replay launches no
    /// graph, so it waits. A union's ranks enter its steps in order on each device thread, so a
    /// step entered behind one in flight is the next step on every rank.
    fn overlaps_replays(&self) -> bool {
        !eager_replay()
    }

    /// A batch that keeps its logits answers its greedy rows on the host, and the next step reads
    /// what the host chose.
    fn overlaps_staged(&self) -> bool {
        self.overlaps_replays()
            && self
                .staging
                .staged()
                .is_none_or(|bucket| bucket.variant != crate::plan::Variant::PrefillOutput)
    }

    fn landed(&mut self, output: BatchOutput) -> Result<BatchOutput> {
        match self.flights.pop_front() {
            Some(flight) => self.finish(flight),
            None => Ok(output),
        }
    }

    fn submit(&mut self, batch: Batch) -> Result<Jobs> {
        let shape = batch.shape;
        // What a refusal is about, said before it happens: the classes the rows carry are the fact
        // no other owner of this batch can report, and a batch that mixes them is refused for them.
        let classes = row_classes(&batch.columns.rows);
        let refused = |error: Error| {
            tracing::debug!(
                batch = batch.id.0,
                shape = %shape,
                classes = %classes,
                %error,
                "a batch is refused before its replay"
            );
            error
        };
        batch.columns.check(&shape).map_err(refused)?;
        let bucket = self.bucket_for(&shape).map_err(refused)?;
        self.staging
            .stage(&batch.columns, &batch.calls, &shape, bucket)
            .map_err(refused)?;
        // A batch is homogeneous in its sampling path, and the check is about this batch's own
        // rows: a warm pose has no request rows, so it is made here rather than in the capture.
        refuse_mixed_sampling(bucket.variant, self.staging.parameters(shape.rows as usize))
            .map_err(refused)?;
        // The batch's slot rows and page copies are the replay's first work, not this thread's:
        // both are operations on this runner's stream and only the device thread issues there
        // (2.4). A table installed from here races the previous batch's replay, and the pages this
        // batch writes land in the row that batch is reading - which reads as a model defect and is
        // a cache one. They travel with the job like the staged columns do.
        self.pending_install = Some(PendingInstall {
            batch: batch.id,
            tables: batch.block_tables.clone(),
            units: batch.cache_units.clone(),
            loads: batch.page_loads.clone(),
            offloads: batch.page_offloads.clone(),
        });
        // The columns are staged here, on the submitting thread; everything that touches the
        // device is the replay job, which this worker's device thread performs in `serve`: the
        // staged columns are uploaded, the bucket's graph is launched - the smallest captured
        // bucket holding this shape - and the sampled tokens are read back, all before `submit`
        // returns its output (4.5, 2.4: one thread issues to the streams).
        Ok(Jobs::new([Job::Replay {
            runner: self.id,
            bucket,
            generation: batch.generation,
            batch: batch.id,
        }]))
    }

    fn shadow(&mut self, batch: Batch) -> Result<Jobs> {
        let shape = batch.shape;
        batch.columns.check(&shape)?;
        let bucket = self.bucket_for(&shape)?;
        self.staging
            .stage(&batch.columns, &batch.calls, &shape, bucket)?;
        // A shadow batch is a real batch: the same homogeneity rule applies to its rows.
        refuse_mixed_sampling(bucket.variant, self.staging.parameters(shape.rows as usize))?;
        // The operators whose kernel differs between the serving generation and the one the batch
        // names are the ones the paired forwards compare (7.3 step 2).
        let next = self.lifecycle.generation_named(batch.generation)?;
        let mut jobs = Vec::new();
        // Every program of the prepared generation ends this pass in one of three states: compared,
        // paired against another key of the same operator, or recorded as unmeasured. An operator
        // split across two backends holds one program per phase, and both are launched by their own
        // descriptor, so `covered` is what stops the second of them being neither.
        let mut covered: Vec<(&'static str, String)> = Vec::new();
        for (operator, key, kernel) in self.lifecycle.generation.iter() {
            let candidate = match next.kernel(operator, key) {
                Ok(candidate) => {
                    covered.push((operator, key.to_owned()));
                    candidate
                }
                // A recapture replaces the operator's program by another compilation, whose
                // descriptor text may differ. The comparison pairs by operator (7.3 step 2), and
                // the previous program is resolved by the key it served with, so the update is
                // still compared rather than silently committed. The walk has already recaptured
                // this bucket for the new descriptor; the comparison is what measures the program.
                //
                // That pairing is sound only while the operator holds **one** program: an operator
                // split across two backends holds two, and pairing this launch with the other one
                // would compare two different computations and call the result a measurement.
                Err(_) => {
                    let mut keys = next.keys(operator);
                    let first = keys.next();
                    let only = first.filter(|_| keys.next().is_none());
                    let Some(other) = only else {
                        tracing::debug!(
                            operator,
                            key,
                            programs = next.keys(operator).count(),
                            "the prepared generation holds no program under this key and more than \
                             one for the operator: no paired forward"
                        );
                        continue;
                    };
                    covered.push((operator, other.to_owned()));
                    next.kernel(operator, other)?
                }
            };
            if Arc::ptr_eq(candidate, kernel) {
                continue;
            }
            jobs.push(Job::Shadow {
                runner: self.id,
                operator,
                key: key.to_owned(),
                bucket,
                generation: batch.generation,
            });
        }
        // A program the prepared generation *adds* has nothing in the serving generation to be
        // compared against - a new operator, and equally a new phase of an operator the deployment
        // already serves. The update's evidence does not cover it, and saying so is the difference
        // between an update that was measured and one that merely ran.
        for (operator, key, _) in next.iter() {
            if self.lifecycle.generation.contains(operator, key) {
                continue;
            }
            if covered
                .iter()
                .any(|(covered_by, covered_key)| *covered_by == operator && covered_key == key)
            {
                continue;
            }
            let new_operator = self.lifecycle.generation.keys(operator).next().is_none();
            if new_operator {
                self.note_uncompared(format!(
                    "operator {operator} is new in generation {}: the serving generation holds no \
                     program of it, so there is nothing to compare it against (its {key} program is \
                     unmeasured)",
                    batch.generation
                ));
            } else {
                self.note_uncompared(format!(
                    "operator {operator} {key} is new in generation {}: the deployment already \
                     serves this operator under another program, and no launch of the serving one \
                     computes this descriptor, so this program is unmeasured",
                    batch.generation
                ));
            }
        }
        Ok(Jobs::new(jobs))
    }

    fn capture_plan(&self) -> Vec<Bucket> {
        self.buckets.capture_order().copied().collect()
    }

    fn warmup(&mut self) -> Result<Jobs> {
        let generation = self.lifecycle.generation.number();
        let mut jobs = Vec::new();
        // Largest first, so the smaller captures reuse the larger ones' pool blocks
        // (`specs/replaceable-processes.md` 9.3).
        for bucket in self.buckets.capture_order().copied().collect::<Vec<_>>() {
            if self
                .lifecycle
                .graphs
                .captured_for(bucket, generation)
                .is_none()
            {
                jobs.push(Job::Capture {
                    runner: self.id,
                    bucket,
                    generation,
                });
            }
        }
        Ok(Jobs::new(jobs))
    }

    fn precopy(&mut self, next: &Arc<Generation>) -> Result<Jobs> {
        // One update's evidence: a new precopy starts its record empty, so an operator this update
        // compares is never reported from the one before it.
        self.clear_uncompared();
        self.lifecycle.precopy(next)
    }

    fn switchover(&mut self, generation: u64) -> Result<()> {
        self.lifecycle.switchover(generation)
    }

    fn revert(&mut self, generation: u64) {
        self.lifecycle.revert(generation)
    }

    fn discard(&mut self, generation: u64) -> Result<()> {
        self.lifecycle.discard(generation)
    }

    fn purge(&mut self, generation: u64) {
        self.lifecycle.purge(generation)
    }

    fn attach_pools(&mut self, pools: Arc<std::sync::Mutex<Pools>>) {
        self.pools = Some(Arc::clone(&pools));
        self.lifecycle.attach_pools(Arc::clone(&pools));
        if let Err(error) = self.kv.attach_pools(pools) {
            tracing::warn!(%error, "the cache provider took no pools; a gathered table will fail");
        }
    }

    fn attach_shadow(&mut self, shadow: crate::shadow::Shadow, page_set_bytes: u64) {
        self.set_shadow(shadow);
        self.kv.attach_scratch_bytes(page_set_bytes);
    }

    fn attach_counters(&mut self, counters: Arc<crate::device::SharedCounters>) {
        self.counters = Some(counters);
    }

    /// Every bucket of the capture plan, in capture order: the step its graph captures, with the
    /// sampler and the commit of a bucket that samples, or the expert step a join takes part in.
    fn record_walk(&mut self) -> Result<Footprint> {
        let mut footprint = Footprint::default();
        for bucket in self.capture_plan() {
            footprint.walk(Some(bucket), |walk| self.walk_recorded(walk, bucket))?;
        }
        Ok(footprint)
    }

    fn attach_uncompared(&mut self, sink: Arc<std::sync::Mutex<Vec<String>>>) {
        self.uncompared = Some(sink);
    }

    fn serve(&mut self, job: &Job) -> Result<BatchOutput> {
        match job {
            Job::Join { .. } => {
                let stream = self.stream.clone();
                let member = self.experts.as_mut().ok_or_else(|| {
                    Error::Invariant("a join reached a runner outside an expert union".into())
                })?;
                if member.woken()?
                    && let Some(step) = member.step(0, &stream)?
                {
                    let generation = self.lifecycle.generation.number();
                    let join = self
                        .buckets
                        .at_capacity(Bucket::expert_join(), step.capacity)
                        .ok_or_else(|| {
                            Error::Invariant(format!(
                                "no join graph holds a step of {} rows per rank",
                                step.capacity
                            ))
                        })?;
                    let (_, captured) = self
                        .lifecycle
                        .graphs
                        .captured_for(join, generation)
                        .ok_or_else(|| {
                            Error::Invariant(format!(
                                "no join graph in generation {generation} or one it replaces"
                            ))
                        })?;
                    captured.launch(&stream)?;
                }
                Ok(BatchOutput::default())
            }
            Job::Replay {
                bucket,
                generation,
                batch,
                ..
            } => {
                // A request that never reaches a replay is invisible from every other owner: the
                // mechanism sees a submission it waits on, the main process a timer. One line per
                // replayed bucket is what tells a served request from an unplanned one.
                tracing::debug!(
                    bucket = %bucket,
                    rows = bucket.rows,
                    tokens = bucket.tokens,
                    generation,
                    batch = batch.0,
                    stream = self.stream.raw(),
                    "a bucket is replayed"
                );
                let stream = self.stream.clone();
                // The batch's cache rows and page copies first, on this thread: the gather reads
                // row `slot` of the cache's pooled table, so the row must state the request's own
                // pages before the graph reads it, and the units must be this cache's before
                // anything is copied into them.
                self.install_pending(*batch)?;
                // Then the batch's columns are uploaded here, on the device thread, before the
                // graph reads them: the scheduler hands over a `Batch` and the device thread
                // stages and uploads it, so a replay that skipped the upload would serve the pose
                // the bucket was captured with.
                self.upload(*bucket)?;
                // Every rank of an expert union runs this step: the graph's dispatches wait on their
                // devices for the peers', so the union is entered before the launch.
                let mut expert_bucket = None;
                if let Some(member) = self.experts.as_mut() {
                    let live = u32::try_from(self.staging.live_tokens())
                        .map_err(|_| Error::Invariant("a batch's tokens exceed u32".into()))?;
                    // The step's rows decide which of the bucket's graphs holds its expert work.
                    if let Some(step) = member.step(live, &stream)? {
                        expert_bucket = Some(
                            self.buckets.at_capacity(*bucket, step.capacity).ok_or_else(|| {
                                Error::Invariant(format!(
                                    "no graph of bucket {bucket} holds a step of {} rows per rank",
                                    step.capacity
                                ))
                            })?,
                        );
                    }
                }
                let chosen = expert_bucket.unwrap_or(*bucket);
                let bucket = &chosen;
                let launched = Instant::now();
                // A diagnosis, not a mode: the same batch, the same columns, the same arena and the
                // same model, launched operator by operator instead of as one graph. It splits every
                // remaining question in two - a wrong answer here is what the runner *staged*, and a
                // right answer here is capture or replay - without a second deployment to compare
                // against.
                let eager = eager_replay();
                if eager {
                    tracing::warn!(
                        bucket = %bucket,
                        generation = *generation,
                        live_tokens = self.staging.live_tokens(),
                        "this replay walks the batch eagerly instead of launching its graph"
                    );
                    // Cloned before the walk: the epoch is borrowed for the walk itself.
                    let kernels = self.lifecycle.generation_named(*generation)?;
                    self.walk(*bucket, &kernels)?;
                }
                // A bucket this generation did not prepare launches only operators that did not
                // change, so the newest generation that did prepare it is the same program: it
                // serves the batch, and the switchover of an update that changed no kernel of this
                // bucket's does not need a fresh executable per bucket.
                let (served, captured) = self
                    .lifecycle
                    .graphs
                    .captured_for(*bucket, *generation)
                    .ok_or_else(|| {
                    Error::Invariant(format!(
                        "no graph of bucket {bucket} in generation {generation} or in a \
                             generation it replaces"
                    ))
                })?;
                if served != *generation {
                    tracing::debug!(
                        bucket = %bucket,
                        generation,
                        served,
                        "the bucket was prepared for an earlier generation; its programs did not change"
                    );
                }
                let nodes = captured.kernel_nodes();
                if !eager {
                    captured.launch(&stream)?;
                }
                // The output read behind this says the batch's offloads finished: the mechanism
                // commits their pages from it, and the forward above did not wait for them.
                if let Some(fence) = self.offload_fence.take() {
                    stream.wait_event(&fence)?;
                }
                if let Some(counters) = &self.counters {
                    counters.launches(nodes);
                }
                // The captured launch's token bound is the pose the bucket was captured with, which
                // `stage_warm` states as the bucket's whole extent; the batch's span is its own. When
                // they differ the kernel is told to run work this batch does not hold, and that is a
                // fact about the two numbers, so it is stated on every batch: a bound taken from the
                // wrong pose leaves no trace in any figure the batch computes for itself.
                let live = self.staging.live_tokens();
                if !bucket.variant.is_decode() && u64::from(bucket.tokens) > live {
                    // The bound alone says the launch runs work this batch does not hold; what the
                    // work *does* is decided by the columns those rows read, and the rows past the
                    // live span are the ones no operator wrote. A slot there is either absent
                    // (`-1`, which the cache write skips) or left from another batch, and a row
                    // naming a slot this batch never allocated is a write outside its pages - which
                    // the device reports as an illegal address two commands later.
                    tracing::warn!(
                        captured_tokens = bucket.tokens,
                        live_tokens = live,
                        beyond = %self.beyond_span(live, u64::from(bucket.tokens)),
                        "a prefill bucket's captured launch states more tokens than this batch \
                         holds: the bound was taken from the pose the bucket was captured with"
                    );
                }
                self.served = true;
                // The sampled tokens are read back behind the replay. A replay that overlaps the
                // next one answers when it lands; any other waits here, and the wait covers the
                // replay as well, which is why the launch's instant is what the device segment is
                // measured from.
                let flight = self.read_sampled(*bucket, launched)?;
                if self.overlaps_staged() {
                    self.flights.push_back(flight);
                    return Ok(BatchOutput::default());
                }
                self.stream.synchronize()?;
                self.finish(flight)
            }
            Job::Capture {
                bucket, generation, ..
            } => self
                .capture(*bucket, *generation)
                .map(|()| BatchOutput::default()),
            Job::Shadow {
                operator,
                key,
                bucket,
                generation,
                ..
            } => self
                .shadow(operator, key, *bucket, *generation)
                .map(|()| BatchOutput::default()),
            Job::Rebind {
                bucket, generation, ..
            } => self
                .rebind(*bucket, *generation)
                .map(|()| BatchOutput::default()),
            Job::Prepare { call, .. } => Err(Error::Invariant(format!(
                "a runner that steps no trajectory was asked to prepare call {}",
                call.0
            ))),
            // A host call is no device work: the media stage performs it, and a device runner
            // handed one was given a job another runner returned.
            Job::Host { call, .. } => Err(Error::Invariant(format!(
                "a runner that serves device work was asked for host call {}",
                call.0
            ))),
        }
    }
}

impl<M: TextModel> CausalLmRunner<M> {
    /// Captures one bucket of `generation` (4.5): the walk runs here, the lifecycle keeps the graph.
    ///
    /// The walk is this class's alone - it stages the bucket's columns, runs the model's forward and
    /// lets the sampler or the commit table write - and the capture, the node table and the
    /// executable are the same for every runner, so those live in [`Lifecycle`].
    fn capture(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        let kernels = self.lifecycle.generation_named(generation)?;
        self.place_columns()?;
        // The same bucket for another expert step size runs the same operators over the same
        // tokens, which the walk before it already made ready: its capture needs no warm walk.
        let twin = self.warmed == Some(bucket.at_capacity(0));
        self.warmed = Some(bucket.at_capacity(0));
        if twin {
            let stream = self.stream.clone();
            self.capturing = true;
            let walked = walk_capture(&stream, || self.walk(bucket, &kernels));
            self.capturing = false;
            return self
                .lifecycle
                .capture(bucket, generation, walked?, &kernels);
        }
        // The first capture of the first generation is where the exchange's program exists, and
        // every rank reaches it in step: their plans capture the same buckets in the same order.
        if let Some(member) = self.experts.as_mut() {
            let stream = self.stream.clone();
            member.ensure(&kernels, &stream)?;
            member.write(bucket.tokens, &stream)?;
            member.set_warming(true);
        }
        let warmed = self.warm(bucket, &kernels);
        if let Some(member) = self.experts.as_ref() {
            member.set_warming(false);
        }
        warmed?;
        let stream = self.stream.clone();
        // The recording's own window: nothing that synchronises may run inside it.
        self.capturing = true;
        let walked = walk_capture(&stream, || self.walk(bucket, &kernels));
        self.capturing = false;
        let walked = walked?;
        self.lifecycle.capture(bucket, generation, walked, &kernels)
    }

    /// Moves the batch columns to device storage, before any graph addresses them.
    fn place_columns(&mut self) -> Result<()> {
        if self.placed {
            return Ok(());
        }
        let pools = self.pools.clone().ok_or(Error::State(
            "a runner captures with the device's pools attached",
        ))?;
        let device = self.device();
        let stream = self.stream.clone();
        let mut pools = pools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.columns.place(device, &mut pools, &stream)?;
        self.commit.place(device, &mut pools, &stream)?;
        drop(pools);
        stream.synchronize()?;
        self.placed = true;
        Ok(())
    }

    /// Walks a bucket's pose once *before* it is captured, so every operator's first-use host work
    /// is done outside the recording.
    ///
    /// A capture records launches, not the host work behind them. An operator that builds a runner,
    /// loads a cubin or compiles a module on its first call does that work inside the recording, and
    /// the graph then replays a launch whose preparation never happened again - which is how a
    /// captured attention writes nothing while the same call outside a graph works. One eager walk
    /// of the same pose makes every operator ready, so the capture records launches of things that
    /// exist. The operator's entry can refuse a capture that meets an unprepared runner; this is what
    /// keeps that refusal from firing on a healthy warmup.
    ///
    /// It runs only where a bucket is captured, and it walks the bucket's inert pose, never a batch:
    /// a walk executes what is staged, and the columns of the last batch stay staged after its
    /// replay. Walked again they compute that batch's step a second time and commit its slot's token
    /// state, so the slot's sampled token moves past one the mechanism never read and the request's
    /// stream loses it. A hot update's new operators are warmed by the shadow pass, and the pose is
    /// safe beside live requests: it writes no KV page (`slot_mapping` is `-1`) and its commit table
    /// is inert, and the next batch stages its own columns over it.
    fn warm(&mut self, bucket: Bucket, kernels: &Arc<Generation>) -> Result<()> {
        // A capture is its own job on this thread, dispatched between batches rather than inside
        // one (`Job::Capture`), and a batch stages and replays in one command, so no batch is ever
        // half-served here. The walk has to run for a bucket captured after serving too: it builds
        // the plans of the pose the capture records, and an operator whose plan was never built
        // cannot be captured at all (a cuBLASLt plan allocates its workspace).

        let began = std::time::Instant::now();
        if self.joins_only(bucket) {
            self.walk(bucket, kernels)?;
            self.stream.synchronize()?;
            return Ok(());
        }
        // The pose, stated before the walk that can fault on it: a warm launch that dies names the
        // numbers it was given, and the table it read, rather than leaving them to be inferred from
        // the bucket. The capture's own walk reuses what this stages.
        self.stage_warm(bucket)?;
        let (lengths, offsets) = warm_span(bucket);
        tracing::info!(
            bucket = %bucket,
            rows = bucket.rows,
            tokens = bucket.tokens,
            calls = warm_calls(bucket).len(),
            seq_lens = ?lengths,
            query_offsets = ?offsets,
            // The third statement of the same partition: the count the kernels bound by. It must
            // equal the last of `query_offsets` and the tokens the calls own, and a capture where
            // those three disagree is an operand no descriptor can match.
            live_tokens = self.staging.live_tokens(),
            page_size = self.kv.page_size(),
            max_kv_len = self.max_kv_len,
            "a bucket's warm pose"
        );
        self.walk(bucket, kernels)?;
        // The warm launches are real device work on the pool's stream; a capture begun over them is
        // legal and does not contain them, but nothing may plan a mapping while they are in flight.
        self.stream.synchronize()?;
        tracing::debug!(
            bucket = %bucket,
            warm_ms = began.elapsed().as_secs_f64() * 1e3,
            "a bucket's operators were warmed before its capture"
        );
        Ok(())
    }

    /// Prepares one bucket of the next generation from a walk of the same bucket (4.5).
    fn rebind(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        self.capturing = true;
        let walked = walk_capture(&stream, || self.walk(bucket, &kernels));
        self.capturing = false;
        self.lifecycle.rebind(bucket, generation, walked?, &kernels)
    }

    /// The pool stream this runner's graphs are planned and replayed on.
    ///
    /// A graph pool is per stream (4.5), and two runners on one device fork their own streams, so
    /// the identity is the stream's own handle rather than a position in a list the worker
    /// keeps elsewhere.
    fn stream_id(&self) -> StreamId {
        StreamId(self.stream.raw())
    }

    /// The device the runner computes on.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// Runs the paired forward of one changed operator and compares its two results (7.3).
    ///
    /// The old kernel writes the real outputs and the new one its twins; the forward continues on
    /// the old kernel's result, so the difference of one operator cannot accumulate.
    ///
    /// The paired forward runs eagerly, on the runner's stream, between two device jobs, and its KV
    /// goes to scratch pages: the KV cache is the state that persists from one step to the next, so
    /// that is what must not be written (7.3). Its activations do land in the bucket's arena region,
    /// which is the only region the model's layers are laid out in; every replay writes that region
    /// before it reads it, so a paired forward can change no value a request observes, and no
    /// serving batch can be in flight while it runs (one device thread, one stream).
    fn shadow(
        &mut self,
        operator: &'static str,
        key: &str,
        bucket: Bucket,
        generation: u64,
    ) -> Result<()> {
        let next = self.lifecycle.generation_named(generation)?;
        let previous = if self.lifecycle.generation.number() == generation {
            self.lifecycle.previous.clone()
        } else {
            Some(Arc::clone(&self.lifecycle.generation))
        }
        .ok_or_else(|| {
            Error::Invariant(
                "shadow execution needs the generation being replaced kept beside the new one"
                    .into(),
            )
        })?;
        let mut harness = self.shadow.take().ok_or_else(|| {
            Error::State(
                "the runner has no shadow harness: it is allocated when the runner is bound",
            )
        })?;
        harness.reset();
        let stream = self.stream.clone();
        let walked = self.walk_paired(bucket, &next, &previous, operator, key, harness.twins());
        // A batch whose rows' context does not fit the page set cannot be replayed beside the served
        // one, and that says nothing about the kernel: the update's answer names what it has no
        // evidence for, and another batch of the bucket may still be compared.
        if let Ok(Some(reason)) = &walked {
            let reason = reason.clone();
            harness.reset();
            self.shadow = Some(harness);
            self.note_uncompared(format!(
                "operator {operator} {key} of generation {generation} in the bucket of {} tokens \
                 and {} rows: {reason}",
                bucket.tokens, bucket.rows
            ));
            return Ok(());
        }
        // The operator's tolerance is the one its paired forward recorded: `Forward::operate`
        // calls `O::tolerance(descriptor)` when it shadows, and the harness compares with it.
        let compared = walked.and_then(|_| {
            let tolerance = harness.tolerance().ok_or_else(|| {
                Error::Invariant(format!(
                    "the paired forward of {operator} recorded no tolerance: was it shadowed?"
                ))
            })?;
            harness.compare(&stream, tolerance)
        });
        // The harness stays with the runner whatever the outcome: its failure is the precopy's,
        // which the calling device command reports.
        harness.reset();
        self.shadow = Some(harness);
        // A pair the two programs could not be compared over, and a forward that carved no twin at
        // all, are both statements of what this update has no evidence for: the operator is named
        // in the update's answer rather than left to the log (7.3 step 2).
        if let Ok(summary) = &compared {
            for entry in &summary.incomparable {
                self.note_uncompared(format!(
                    "operator {operator} {key} of generation {generation} in the bucket of {} \
                     tokens and {} rows: {entry}",
                    bucket.tokens, bucket.rows
                ));
            }
            if summary.pairs == 0 {
                self.note_uncompared(format!(
                    "operator {operator} of generation {generation} was shadowed in the bucket of \
                     {} tokens and {} rows and its launch carved no twin, so nothing was compared",
                    bucket.tokens, bucket.rows
                ));
            }
        }
        compared.map(|_| ())
    }

    /// Walks one bucket's forward with two kernels of `operator`: `previous`'s on the real outputs
    /// and `next`'s on the twins (7.3).
    ///
    /// The pair is the operator's, not the descriptor's: `key` is the key the serving generation
    /// stored the previous program under, and every launch of the operator in this bucket is
    /// compared against it (or against the serving program under the launch's own key, when that
    /// generation still holds one - an operator served under several descriptors is compared
    /// program for program).
    ///
    /// The paired forward runs over a scratch KV: every page it writes is a scratch page, so the
    /// new kernel cannot touch a page a serving request reads. The model's `Forward::operate` does
    /// the branch, once, for every layer alike.
    ///
    /// Its columns are the ones the bucket has staged. A shadow job the mechanism submits right
    /// after a batch stages that batch, so the comparison runs on served work; a precopy-driven
    /// shadow job has no batch staged, and this stages a warm pose instead - the same bucket, the
    /// same launches, every slot `-1` so the cache write is inert. Both are sound for a kernel
    /// comparison (each operator is compared on the inputs its old kernel saw), and the warm one is
    /// what a precopy can have: the update is checked before the generation that would serve it has
    /// served anything.
    fn walk_paired(
        &mut self,
        bucket: Bucket,
        next: &Arc<Generation>,
        previous: &Arc<Generation>,
        operator: &'static str,
        key: &str,
        twins: &mut ops::Twins,
    ) -> Result<Option<String>> {
        let stream = self.stream.clone();
        match self.staging.staged() {
            Some(staged) if staged == bucket => {}
            _ => {
                self.stage_warm(bucket)?;
            }
        }
        self.upload(bucket)?;
        // Every row writes its own scratch pages, and holds a copy of its context there: the paired
        // forward must not write a resident page, and it reads the keys the served row read. So a
        // row's width is the row's longest reach, the keys it reads and the positions it writes.
        let rows = bucket.rows as usize;
        let lengths: Vec<i32> = self.staging.batch_rows().seq_lens[..rows].to_vec();
        let reach = {
            let batch = self.staging.batch_rows();
            let live = usize::try_from(self.staging.live_tokens()).unwrap_or(0);
            let written = batch.positions[..live]
                .iter()
                .zip(&batch.slot_mapping[..live])
                .filter(|(_, slot)| **slot >= 0)
                .map(|(position, _)| position.saturating_add(1));
            lengths.iter().copied().chain(written).max().unwrap_or(0)
        };
        let pages = u32::try_from(reach)
            .unwrap_or(0)
            .div_ceil(self.kv.page_size().max(1))
            .max(1);
        let capacity = self.kv.scratch_capacity();
        if bucket.rows.saturating_mul(pages) > capacity {
            return Ok(Some(format!(
                "its {} rows of {pages} pages do not fit the page set's {capacity} pages",
                bucket.rows
            )));
        }
        self.upload_scratch_slots(bucket, pages)?;
        let base = self.pool()?.base(self.stream_id())?;
        let reserved = self.pool()?.reserved_on(self.stream_id());
        let arena = nn::Arena::new(ops::DevicePtr::new(base.address()), reserved);
        let batch = self.token_batch_with(
            bucket,
            self.columns
                .scratch_slots
                .extent(0..u64::from(bucket.tokens)),
        )?;
        let pairing = crate::shadow::Pairing::new(operator, key, previous, twins);
        // The count bounds the MoE sort's padded rows: the paired forward gets the same column as
        // the served one, so the comparison sees one count (7.3).
        let tensor = self.tensor.clone();
        let mut forward = Forward::new(&stream, next, &arena)
            .with_route(&pairing)
            .with_live_tokens(self.columns.live_tokens.device);
        if let Some(group) = tensor.as_deref() {
            forward = forward.with_group(group);
        }
        let hidden = arena.tensor(
            self.model.dtype(),
            ops::Shape::new(&[u64::from(bucket.tokens), u64::from(self.model.hidden())]),
        )?;
        let slots: Vec<i32> = self.staging.slots(rows).to_vec();
        let ids = self.input_ids(bucket);
        self.with_kv(&arena, &slots, Some((pages, &lengths)), false, |cache| {
            self.model
                .embed_input_ids(&forward, ids, hidden)
                .map_err(Error::from)?;
            self.model
                .forward(&forward, &batch, cache, hidden)
                .map_err(Error::from)
        })?;
        Ok(None)
    }

    /// Reads the step's device results back and states what each call did (4.4).
    ///
    /// Two columns come back after the replay's stream completed: the sampled token ids the
    /// sampling step wrote (one `i32` per sampling row) and the commit table's per-slot trajectory
    /// position, which the commit kernel advanced. `launched` is the instant the replay was issued,
    /// so the device segment is measured from the launch to this wait and the host segment from
    /// there to the output.
    ///
    /// Replaces a greedy row's drawn token with the argmax of its logits row, taken on the host.
    ///
    /// A request whose temperature is zero asks for its largest candidate, and the captured graph of
    /// a prefill bucket draws for every row of the bucket (a bucket is homogeneous in its sampling
    /// path, and a greedy prefill variant would double the largest graphs' memory to decide one
    /// token). The host has the logits - the arena buffer is live until the next `begin` - so the
    /// decision is made here, once per request, before the token leaves this process.
    ///
    /// The penalties are no-ops for a row whose slot has committed nothing *before this step*, which
    /// is what a prompt ending is: a row whose slot already held committed tokens would need the
    /// penalized logits, which the graph computes and this path does not, so it is **refused by
    /// name** rather than argmaxed over the raw row. The readback states the position after this
    /// batch's own token was committed, so the row's commit is subtracted before the test.
    fn greedy_first_tokens(
        &mut self,
        logits: TensorView,
        rows: usize,
        sampling: &[i32],
        turn: usize,
    ) -> Result<()> {
        if logits.dtype != DType::F32 {
            return Err(Error::Invalid(format!(
                "a greedy request's first token is argmaxed on the host, which reads f32 logits, \
                 and this deployment's logits are {}",
                logits.dtype
            )));
        }
        let width = u64::from(self.model.vocabulary());
        let row_bytes = (width * 4) as usize;
        if row_bytes > self.columns.logits_row.len() {
            return Err(Error::Invariant(
                "the logits row buffer is narrower than the vocabulary".into(),
            ));
        }
        let columns = ops::sampler::PARAMETER_COLUMNS as usize;
        for row in 0..rows {
            let slot = self.staging.slots(rows)[row];
            let readback = usize::try_from(slot)
                .ok()
                .and_then(|slot| sampling.get(slot).copied())
                .unwrap_or(0);
            let decision = prompt_token(
                self.staging.logits_rows()[row] < 0,
                self.staging.parameters(rows)[row * columns + ops::sampler::TEMPERATURE as usize],
                slot,
                readback,
            )?;
            // A drawn row keeps the graph's token, and a row that ends no prompt keeps nothing: the
            // host answers only the greedy prompt ending.
            if decision != PromptToken::HostArgmax {
                continue;
            }
            let source = logits.ptr.address() + row as u64 * width * 4;
            // SAFETY: the source is the row's `width` f32 inside the logits tensor the graph wrote,
            // the destination is this runner's pinned row buffer of that length, and the stream has
            // completed the launch that wrote it.
            unsafe {
                platform::memcpy_async(
                    self.columns.logits_row.address(),
                    source,
                    row_bytes,
                    &self.stream,
                )?;
            }
            self.stream.synchronize()?;
            // SAFETY: the copy above filled `width` f32 in the pinned row buffer, and the token
            // buffer holds one i32 per row of this bucket.
            // SAFETY: the copy above filled `width` f32 in the pinned row buffer.
            let row_logits = unsafe {
                std::slice::from_raw_parts(
                    self.columns.logits_row.as_ptr().cast::<f32>(),
                    width as usize,
                )
            };
            let best = argmax(row_logits);
            // The first token of a prompt is the one value a serving run and a reference stack must
            // agree on before anything downstream means anything, and a wrong token and a right one
            // read the same in a completion. The margin says which it is: a near-tie is numerics,
            // a wide one is a defect. Only on the host's greedy path, which is one call per prompt.
            tracing::debug!(
                row,
                token = best,
                logit = row_logits.get(best).copied().unwrap_or(f32::NAN),
                runner_up = runner_up(row_logits, best),
                "the prompt's first token was argmaxed on the host"
            );
            // SAFETY: the readback buffer holds `rows` i32 and `row` is inside it.
            unsafe {
                *self.columns.readback[turn].as_ptr().cast::<i32>().add(row) = best as i32;
            }
            // The graph committed its own draw as this slot's next input, and the next decode
            // embeds what the client was told.
            if let Ok(slot) = usize::try_from(slot) {
                self.commit
                    .set_next_input(slot, best as i32, &self.stream)?;
            }
        }
        Ok(())
    }

    /// A bucket that samples nothing answers with an empty output: nothing was measured and no call
    /// produced a token.
    fn read_sampled(&mut self, bucket: Bucket, launched: Instant) -> Result<Flight> {
        let turn = self.turn;
        let sampled = self.bucket_samples.borrow().get(&bucket).copied();
        let mut flight = Flight {
            bucket,
            sampled,
            launched,
            turn,
            slots: Vec::new(),
            logits_rows: Vec::new(),
            rows: BatchRows::default(),
            calls: Vec::new(),
        };
        let Some(samples) = sampled else {
            return Ok(flight);
        };
        self.turn = 1 - turn;
        let rows = bucket.rows as usize;
        let bytes = rows * std::mem::size_of::<i32>();
        if bytes > self.columns.readback[turn].len() {
            return Err(Error::Invariant(
                "the sampler's readback buffer is smaller than the bucket".into(),
            ));
        }
        let slots = self.commit.slots();
        if 2 * slots * std::mem::size_of::<i32>() > self.commit.readback_len() {
            return Err(Error::Invariant(
                "the commit table's readback buffer is smaller than the slot count".into(),
            ));
        }
        // SAFETY: the sampled ids are `rows` i32 in the bucket's plan and the readback buffer is
        // this runner's pinned memory of at least that size, as is the table's `[slots]` i32; the
        // stream has completed.
        unsafe {
            platform::memcpy_async(
                self.columns.readback[turn].address(),
                samples.tokens.ptr.address(),
                bytes,
                &self.stream,
            )?;
            // The two per-slot lengths the completion reports: the logical position execution
            // initialized and the sampling position a successor may read.
            platform::memcpy_async(
                self.commit.readback_address(turn),
                self.commit.logical_ptr().address(),
                slots * std::mem::size_of::<i32>(),
                &self.stream,
            )?;
            platform::memcpy_async(
                self.commit.readback_address(turn) + (slots * std::mem::size_of::<i32>()) as u64,
                self.commit.sampling_ptr().address(),
                slots * std::mem::size_of::<i32>(),
                &self.stream,
            )?;
        }
        // What the answer reads of the batch, because the next batch is staged before this one
        // lands.
        flight.slots = self.staging.slots(rows).to_vec();
        flight.logits_rows = self.staging.logits_rows().to_vec();
        flight.rows = self.staging.batch_rows().clone();
        flight.calls = self.staging.calls().to_vec();
        Ok(flight)
    }

    /// The answer of a replay whose copies have landed.
    fn finish(&mut self, flight: Flight) -> Result<BatchOutput> {
        let Some(samples) = flight.sampled else {
            return Ok(BatchOutput::default());
        };
        let Flight {
            bucket,
            launched,
            turn,
            ..
        } = flight;
        let rows = bucket.rows as usize;
        let slots = self.commit.slots();
        let device_us = launched.elapsed().as_micros() as u64;
        let host_start = Instant::now();
        // SAFETY: the copies above filled `rows` i32 with the sampled ids and `slots` i32 with the
        // table's trajectory positions.
        let (ids, logical, sampling) = unsafe {
            (
                std::slice::from_raw_parts(
                    self.columns.readback[turn].as_ptr().cast::<i32>(),
                    rows,
                ),
                std::slice::from_raw_parts(self.commit.readback_ptr(turn).cast::<i32>(), slots),
                std::slice::from_raw_parts(
                    (self.commit.readback_ptr(turn) as usize + slots * std::mem::size_of::<i32>())
                        as *const i32,
                    slots,
                ),
            )
        };
        // A greedy request's first token is the argmax of its prompt-ending row, taken here on the
        // host: the bucket is homogeneous in its sampling path, so a prefill bucket's graph draws,
        // and a draw cannot produce a greedy token (the candidates are in token-id order before the
        // inverse CDF). Only a bucket that kept its logits view can answer this - a decode bucket's
        // variant already states that it is greedy, and its graph argmaxes. The tokens are corrected
        // in place, in the buffer the copy above filled, so a step allocates nothing (4.5).
        if let Some(logits) = samples.logits
            && !bucket.variant.is_decode()
        {
            self.greedy_first_tokens(logits, rows, sampling, turn)?;
        }
        // A sampled id is an INDEX into the vocabulary, so it is below `vocabulary` by contract:
        // a kernel that answers the width instead of the last index is a defect and not a token,
        // and a token one past the vocabulary reaches the tokenizer as an out-of-range id. The
        // check is here, once, on the readback of every batch: it costs one comparison and it
        // turns a batch of nonsense tokens into a refusal at the first step that produced one.
        if let Some(row) = ids.iter().position(|id| {
            let id = *id;
            id < 0
                || u64::try_from(id)
                    .map(|id| id >= self.model.vocabulary() as u64)
                    .unwrap_or(true)
        }) {
            return Err(Error::Invariant(format!(
                "row {row} sampled token {} outside the vocabulary of {}",
                ids[row],
                self.model.vocabulary()
            )));
        }
        let states = states_of(
            &flight.slots,
            &flight.rows,
            flight.calls.len(),
            logical,
            sampling,
        );
        let calls = calls_output(
            ids,
            &flight.logits_rows,
            &flight.rows,
            &flight.calls,
            &states,
        )?;
        Ok(BatchOutput {
            calls,
            timing: TimingCounters {
                // The device thread states the wait it saw; the runner issues nothing before it.
                queued_us: 0,
                device_us,
                // Nothing is transferred for a completion in-process: the mechanism reads it here.
                copy_us: 0,
                host_us: host_start.elapsed().as_micros() as u64,
            },
            forward_stats: None,
        })
    }

    /// Installs the staged batch's cache rows and page copies on this runner's stream.
    ///
    /// A refusal is now the replay's, not submit's: the work itself is on the device's stream, and
    /// a batch whose pages cannot be installed fails as that batch - the operator the cache could
    /// not describe is named by the error either way.
    fn install_pending(&mut self, batch: BatchId) -> Result<()> {
        let Some(install) = self.pending_install.take() else {
            return Ok(());
        };
        if install.batch != batch {
            return Err(Error::Invariant(format!(
                "the cache rows staged for batch {} are not batch {}'s: a replay installs only its \
                 own batch's pages",
                install.batch.0, batch.0
            )));
        }
        self.kv
            .install_block_tables(&install.tables, &self.stream)?;
        self.kv.apply_cache_units(&install.units, &self.stream)?;
        self.offload_fence =
            self.kv
                .issue_pages(&install.loads, &install.offloads, &self.stream)?;
        Ok(())
    }

    /// What the slot column holds for the rows a captured bound names beyond this batch's span.
    ///
    /// The cache write reads one slot per row of the launch's grid. Rows past the live span are rows
    /// this batch never staged, so their slots are whatever the column last held: `-1` (which the
    /// write skips), a slot of an earlier batch (a write into another request's pages), or a value
    /// nothing ever wrote (a write into pages that do not exist, which is an illegal address on the
    /// device). The three are indistinguishable from the bound alone, and only the third faults.
    fn beyond_span(&self, live: u64, tokens: u64) -> String {
        let first = usize::try_from(live).unwrap_or(0);
        let count = usize::try_from(tokens.saturating_sub(live))
            .unwrap_or(0)
            .min(4096);
        if count == 0 {
            return "none".to_owned();
        }
        let address = self
            .columns
            .slot_mapping
            .device
            .ptr
            .address()
            .saturating_add((first as u64) * 4);
        let Some(raw) = self.read_device(address, count * 4) else {
            return "unreadable".to_owned();
        };
        let mut negative = 0_usize;
        let mut largest = i64::MIN;
        let mut first_few = String::new();
        for (index, word) in raw.as_chunks::<4>().0.iter().enumerate() {
            let slot = i32::from_ne_bytes(*word);
            if slot < 0 {
                negative += 1;
            } else {
                largest = largest.max(i64::from(slot));
            }
            if index < 4 {
                first_few.push_str(&format!("{slot},"));
            }
        }
        format!("rows={count} negative={negative} max={largest} first=[{first_few}]")
    }

    /// `bytes` of device memory at `address`, read on this runner's stream, or `None`.
    ///
    /// The copy is issued only from the device thread, and it synchronizes the stream before the
    /// host reads the buffer.
    fn read_device(&self, address: u64, bytes: usize) -> Option<Vec<u8>> {
        if address == 0 || bytes == 0 {
            return None;
        }
        let mut raw = vec![0_u8; bytes];
        // SAFETY: the destination is this function's own buffer of `bytes`, and the source is an
        // address the caller read out of a launch's own arguments.
        let copied = unsafe {
            platform::memcpy_async(raw.as_mut_ptr() as u64, address, bytes, &self.stream)
        };
        if copied.is_err() || self.stream.synchronize().is_err() {
            return None;
        }
        Some(raw)
    }

    /// Uploads the staged batch into the device columns the captured graphs address.
    ///
    /// Every column is copied into a buffer allocated when the runner bound, sized for the largest
    /// bucket, and the host-side encode indices and draw buffers are the runner's own: a step
    /// writes fixed addresses and allocates nothing (4.5).
    fn upload(&mut self, bucket: Bucket) -> Result<()> {
        let rows = bucket.rows as usize;
        let tokens = bucket.tokens as usize;
        if self.staging.staged().is_none() {
            return Err(Error::State("a step needs a staged batch"));
        }
        let batch = self.staging.batch_rows();

        encode_i32(&mut self.encoded, &batch.positions[..tokens]);
        self.columns.positions.write(&self.encoded)?;
        // The ids travel with the batch: `nn::CausalLm::forward` runs the stack over hidden states
        // the model already holds, so embedding them is the runner's step, not the model's. A
        // decode row's token is the slot's next input, which the graph reads for itself: the step
        // before it may still be running, and the mechanism plans ahead of its results anyway.
        encode_i32(&mut self.encoded, self.staging.input_ids(tokens));
        self.columns.input_ids.write(&self.encoded)?;
        encode_i32(&mut self.encoded, &batch.slot_mapping[..tokens]);
        self.columns.slot_mapping.write(&self.encoded)?;
        encode_i32(&mut self.encoded, &batch.seq_lens[..rows]);
        self.columns.seq_lens.write(&self.encoded)?;
        encode_i32(&mut self.encoded, &batch.query_offsets[..rows + 1]);
        self.columns.query_offsets.write(&self.encoded)?;
        encode_i32(&mut self.encoded, &batch.logits_rows[..rows]);
        self.columns.logits_rows.write(&self.encoded)?;

        // The sampler gathers rows of the `[tokens, hidden]` stack, and a row that samples reads
        // the **last token of its own query range**: a prefill's row is its whole prompt, so the
        // row index and the token index are the same only when every row holds one token. Reading
        // the row index here samples a prefill from its first token - a decode is unaffected,
        // which is why this shows as a wrong first token and not as a fault.
        //
        // A row that samples nothing still gathers some in-range token; its id is discarded.
        gather_indices(
            &batch.logits_rows,
            &batch.query_offsets,
            usize::try_from(self.staging.live_tokens()).unwrap_or(usize::MAX),
            rows,
            &mut self.indices,
        )?;
        encode_i32(&mut self.encoded, &self.indices);
        self.columns.indices.write(&self.encoded)?;

        encode_f32(&mut self.encoded, self.staging.parameters(rows));
        self.columns.parameters.write(&self.encoded)?;

        encode_i32(&mut self.encoded, self.staging.slots(rows));
        self.columns.slots.write(&self.encoded)?;

        // Each row's uniform is a function of its request's stream and the position it samples
        // (`ipc::philox`), never of a generator this runner advances: a step replayed, restored or
        // served on another engine draws the same number, and the ranks of a tensor group, which
        // stage the same batch, draw the same token from the same gathered logits.
        self.draws.clear();
        self.draws.extend(
            self.staging
                .draws(rows)
                .iter()
                .map(|address| address.uniform()),
        );
        encode_f32(&mut self.encoded, &self.draws);
        self.columns.draws.write(&self.encoded)?;

        // The batch's live token count, which no shape of the bucket states: a captured graph reads
        // the bucket's token count as a constant, so the MoE sort - which must write a sentinel into
        // every route row of a padded token - reads this column instead.
        encode_i32(&mut self.encoded, &[self.staging.live_tokens]);
        self.columns.live_tokens.write(&self.encoded)?;

        self.commit
            .stage(self.staging.commit(), rows, self.commit.capacity as usize)?;

        // A copy issued inside a recording would be part of the graph, and replay the warm pose.
        if self.capturing {
            return Ok(());
        }
        let stream = self.stream.clone();
        for column in self.columns.staged() {
            column.push(&stream)?;
        }
        self.commit.upload(&stream)?;
        Ok(())
    }

    /// Stages the paired forward's cache slots: the batch's own tokens, addressed in the scratch
    /// page set's numbering.
    ///
    /// A slot is flat over one plane's pages, and the scratch set gives row `r` the pages
    /// `r · pages .. (r + 1) · pages`, which is what its block table states; so the new kernel's
    /// writes land in the row's own scratch pages and never in a page a served request reads (7.3).
    fn upload_scratch_slots(&mut self, bucket: Bucket, pages: u32) -> Result<()> {
        let tokens = bucket.tokens as usize;
        let live = usize::try_from(self.staging.live_tokens()).unwrap_or(0);
        let batch = self.staging.batch_rows();
        let mapping = crate::shadow::scratch_column(
            &batch.positions,
            &batch.query_offsets,
            &batch.slot_mapping,
            live,
            tokens,
            pages,
            self.kv.page_size(),
        )?;
        encode_i32(&mut self.encoded, &mapping);
        self.columns.scratch_slots.write(&self.encoded)?;
        self.columns.scratch_slots.upload(&self.stream)
    }

    /// The bucket's extent of the input ids, which the forward embeds before the stack.
    fn input_ids(&self, bucket: Bucket) -> TensorView {
        self.columns.input_ids.extent(0..u64::from(bucket.tokens))
    }

    /// The device-side `TokenBatch` of `bucket`: the bucket's extent of each column.
    fn token_batch(&self, bucket: Bucket) -> Result<TokenBatch> {
        self.token_batch_with(
            bucket,
            self.columns.slot_mapping.extent(0..bucket.tokens as u64),
        )
    }

    /// The batch with `slot_mapping` in place of the served batch's cache slots: what a paired
    /// forward passes so its writes land in scratch pages (7.3).
    fn token_batch_with(&self, bucket: Bucket, slot_mapping: TensorView) -> Result<TokenBatch> {
        Ok(TokenBatch {
            positions: self.columns.positions.extent(0..bucket.tokens as u64),
            slot_mapping,
            seq_lens: self.columns.seq_lens.extent(0..bucket.rows as u64),
            query_offsets: self.columns.query_offsets.extent(0..bucket.rows as u64 + 1),
            logits_rows: self.columns.logits_rows.extent(0..bucket.rows as u64),
            phase: match bucket.variant.is_decode() {
                true => ops::PhaseKind::Decode,
                false => ops::PhaseKind::Prefill,
            },
            page_size: self.kv.page_size(),
            num_tokens: bucket.tokens,
            max_kv_len: self.max_kv_len,
            draft_lens: None,
        })
    }

    /// Runs `operation` with the batch's `KvView` over `slots`, resident or scratch.
    ///
    /// The view lives as long as the closure: `KvGroup` borrows its layer planes, so the planes,
    /// tables and groups are locals of one scope and the model reads them borrowed. A `quiet` walk
    /// reads the resident block tables where they stand rather than gathering them on the stream.
    /// The attention program's workspace is reserved from `arena` here, in the forward's order, so a
    /// replay finds it at the address its capture used.
    fn with_kv<R>(
        &self,
        arena: &nn::Arena,
        slots: &[i32],
        scratch_pages: Option<(u32, &[i32])>,
        quiet: bool,
        operation: impl FnOnce(&KvView<'_>) -> Result<R>,
    ) -> Result<R> {
        let workspace = arena.tensor(DType::U8, ops::Shape::new(&[ATTENTION_WORKSPACE_BYTES]))?;
        let groups = self.kv.groups();
        let mut layers: Vec<Vec<nn::KvPlanes>> = Vec::with_capacity(groups);
        let mut tables = Vec::with_capacity(groups);
        for group in 0..groups {
            match scratch_pages {
                None => {
                    layers.push(self.kv.planes(group)?);
                    tables.push(if quiet {
                        self.kv.block_table_view(group, slots)?
                    } else {
                        self.kv.block_table(group, slots, &self.stream)?
                    });
                }
                Some((pages, lengths)) => {
                    let (table, planes) =
                        self.kv
                            .scratch(group, slots, lengths, pages, &self.stream)?;
                    layers.push(planes);
                    tables.push(table);
                }
            }
        }
        let view: Vec<nn::KvGroup<'_>> = (0..groups)
            .map(|group| nn::KvGroup {
                block_tables: tables[group],
                page_size: self.kv.page_size(),
                dtype: self.kv.dtype(),
                // The scales are per layer and travel on each `KvPlanes` (nn's
                // `KvPlanes::key_scale`/`value_scale`); the provider sets them as it
                // builds the group's planes.
                layers: &layers[group],
            })
            .collect();
        operation(&KvView {
            groups: &view,
            workspace,
        })
    }

    /// Walks one bucket's forward under the active capture with `generation`'s kernels.
    ///
    /// It is the same forward a replay runs, over the same columns and in the same arena order, so
    /// the nodes it records are the ones the live graph has and the addresses are the same each
    /// time. It launches nothing: the stream is capturing.
    fn walk(&mut self, bucket: Bucket, generation: &Arc<Generation>) -> Result<()> {
        self.walk_body(bucket, generation)
    }

    /// The walk's own work, without the wrapper a throwaway capture needs.
    fn walk_body(&mut self, bucket: Bucket, generation: &Arc<Generation>) -> Result<()> {
        let stream = self.stream.clone();
        if self.joins_only(bucket) {
            return self.walk_join(bucket, generation);
        }
        match self.staging.staged() {
            Some(staged) if staged == bucket => {}
            _ => {
                self.stage_warm(bucket)?;
            }
        }
        let first = self.pool()?.begin(self.stream_id(), bucket)?;
        self.upload(bucket)?;
        let base = self.pool()?.base(self.stream_id())?;
        let reserved = self.pool()?.reserved_on(self.stream_id());
        let arena = nn::Arena::new(ops::DevicePtr::new(base.address()), reserved);
        let probes = crate::profiling::fences::capturing(&stream);
        let tensor = self.tensor.clone();
        let mut forward = Forward::new(&stream, generation, &arena)
            .with_live_tokens(self.columns.live_tokens.device)
            .with_fence(probes.as_ref().map(|probes| probes as &dyn nn::Fence));
        if let Some(group) = tensor.as_deref() {
            forward = forward.with_group(group);
        }
        let shaped = self
            .experts
            .as_ref()
            .and_then(super::experts::Member::views)
            .map(|views| views.at_rows(bucket.capacity));
        if let Some(views) = shaped.as_ref() {
            forward = forward.with_dispatch(views);
        }
        self.step(&forward, bucket)?;
        if first {
            self.pool()?
                .record(self.stream_id(), bucket, arena.peak())?;
        }
        Ok(())
    }

    /// One step of the staged `bucket` on `forward`: the slots' gathers, the embedding, the stack
    /// and, for a bucket that samples, the sampler and the token-state commit. The model calls a
    /// capture walk records, a warm walk launches, and the recorded walk records.
    ///
    /// Every buffer the step works in is reserved from `forward`'s arena in this order, so a
    /// recording forward over a counting arena reaches the extent the launching one does.
    fn step(&self, forward: &Forward<'_>, bucket: Bucket) -> Result<()> {
        let batch = self.token_batch(bucket)?;
        let hidden = forward.arena().tensor(
            self.model.dtype(),
            ops::Shape::new(&[u64::from(bucket.tokens), u64::from(self.model.hidden())]),
        )?;
        let slots: Vec<i32> = self.staging.slots(bucket.rows as usize).to_vec();
        let ids = self.input_ids(bucket);
        self.gather_step(forward, bucket)?;
        // A recording or a capture issues no table copy of its own: the tables are read where they
        // stand, which is what a replay addresses.
        let quiet = self.capturing || forward.is_recording();
        self.with_kv(forward.arena(), &slots, None, quiet, |cache| {
            self.model
                .embed_input_ids(forward, ids, hidden)
                .map_err(Error::from)?;
            self.model
                .forward(forward, &batch, cache, hidden)
                .map_err(Error::from)?;
            if bucket.variant.samples() {
                self.sample(forward, bucket, hidden)?;
            }
            Ok(())
        })
    }

    /// Walks `bucket` on recording forwards of `walk` ([`Runner::record_walk`]): the step its
    /// graph captures, or this rank's part in an expert step it only joins.
    ///
    /// The bucket's warm pose is staged on the host and nothing is uploaded: a recording reads no
    /// value, and the columns' addresses are what the forward is handed. The expert exchange's
    /// views are placeholders of the extents the transport lays out.
    fn walk_recorded(&mut self, walk: &Walk, bucket: Bucket) -> Result<()> {
        let views = match self.experts.as_ref() {
            Some(member) => {
                // The exchange's own program is resolved when the transport is created, before any
                // forward runs, so the walk states it beside the forward's.
                member.record_program();
                Some(member.recorded_views(|dtype, shape| walk.placeholder(dtype, shape))?)
            }
            None => None,
        };
        let shaped = views.as_ref().map(|views| views.at_rows(bucket.capacity));
        if self.joins_only(bucket) {
            let views = shaped
                .as_ref()
                .ok_or_else(|| Error::Invariant("a join walked with no expert transport".into()))?;
            let forward = walk.forward(None)?.with_dispatch(views);
            return self.model.join_expert_step(&forward).map_err(Error::from);
        }
        match self.staging.staged() {
            Some(staged) if staged == bucket => {}
            _ => {
                self.stage_warm(bucket)?;
            }
        }
        let mut forward = walk
            .forward(self.tensor.as_deref())?
            .with_live_tokens(self.columns.live_tokens.device);
        if let Some(views) = shaped.as_ref() {
            forward = forward.with_dispatch(views);
        }
        self.step(&forward, bucket)
    }

    /// What a step reads through its slots, gathered on the device.
    ///
    /// The block tables of the batch's rows, and a decode row's input token: the token the step
    /// before it wrote, which the host need not have seen, so a step can be launched behind the one
    /// it continues.
    fn gather_step(&self, forward: &Forward<'_>, bucket: Bucket) -> Result<()> {
        let rows = u64::from(bucket.rows);
        let slots = self.columns.slots.shaped(&[rows]);
        let gather = |columns: u32| ops::logits::GatherDescriptor {
            dtype: DType::I32,
            columns,
            block: GATHER_BLOCK,
            pdl: nn::layers::launch::PDL,
        };
        for group in 0..self.kv.groups() {
            let Some((pooled, gathered)) = self.kv.table_source(group, bucket.rows)? else {
                continue;
            };
            let width = u32::try_from(pooled.shape.last())
                .map_err(|_| Error::Invalid("a block table wider than a launch states".into()))?;
            forward
                .operate::<ops::logits::LogitsGather>(
                    &gather(width),
                    ops::logits::GatherArgs {
                        hidden: pooled,
                        rows: slots,
                        out: gathered,
                    },
                )
                .map_err(Error::from)?;
        }
        if bucket.variant.is_decode() {
            let future = u64::from(self.commit.slots);
            forward
                .operate::<ops::logits::LogitsGather>(
                    &gather(1),
                    ops::logits::GatherArgs {
                        hidden: self.commit.future.shaped(&[future, 1]),
                        rows: slots,
                        out: self.columns.input_ids.shaped(&[rows, 1]),
                    },
                )
                .map_err(Error::from)?;
        }
        Ok(())
    }

    /// Whether a walk of `bucket` takes this rank's part in a step with no tokens of its own.
    ///
    /// A rank that only holds experts walks every bucket that way: its captures keep the ranks in
    /// step through a warmup, because each warm walk of an attention rank needs it inside the same
    /// exchanges.
    fn joins_only(&self, bucket: Bucket) -> bool {
        bucket.variant == crate::plan::Variant::ExpertJoin
            || self
                .experts
                .as_ref()
                .is_some_and(super::experts::Member::experts_only)
    }

    /// Walks every expert layer of a step with no tokens of this rank's own.
    fn walk_join(&mut self, bucket: Bucket, generation: &Arc<Generation>) -> Result<()> {
        let stream = self.stream.clone();
        let first = self.pool()?.begin(self.stream_id(), bucket)?;
        let base = self.pool()?.base(self.stream_id())?;
        let reserved = self.pool()?.reserved_on(self.stream_id());
        let arena = nn::Arena::new(ops::DevicePtr::new(base.address()), reserved);
        let views = self
            .experts
            .as_ref()
            .and_then(super::experts::Member::views)
            .ok_or_else(|| Error::Invariant("a join walked with no expert transport".into()))?
            .at_rows(bucket.capacity);
        let forward = Forward::new(&stream, generation, &arena).with_dispatch(&views);
        self.model.join_expert_step(&forward).map_err(Error::from)?;
        if first {
            self.pool()?
                .record(self.stream_id(), bucket, arena.peak())?;
        }
        Ok(())
    }

    /// Appends the bucket's sampling step to the capture in flight (4.4, 9's `采样`).
    ///
    /// The step gathers the rows whose logits are wanted, projects them, and samples one token per
    /// row through the model's sampler; its working set comes from the bucket's arena, so a replay
    /// samples without a host round trip.
    fn sample(&self, forward: &Forward<'_>, bucket: Bucket, hidden: TensorView) -> Result<()> {
        let arena = forward.arena();
        let Some(sampling) = &self.sampling else {
            return Err(Error::State(
                "the runner has no sampler but the bucket samples",
            ));
        };
        let rows = u64::from(bucket.rows);
        // The batch is checked where a real one is staged (`submit`, `shadow`): a capture has no
        // batch contents - its warm pose states a zeroed parameter column, which reads as a greedy
        // row - so the check here would refuse a sampled bucket its own capture.
        let gathered = arena.tensor(
            self.model.dtype(),
            ops::Shape::new(&[rows, u64::from(self.model.hidden())]),
        )?;
        // The indices are the same defect one operator earlier: `IndexRows::gather` takes a flat `[count]`
        // view and derives `count` from the gathered rows, so the column's own extent - the largest
        // bucket's - is refused for every bucket narrower than it, before the sampler is reached.
        nn::IndexRows
            .gather(
                forward,
                hidden,
                self.columns.indices.shaped(&[rows]),
                gathered,
            )
            .map_err(Error::from)?;
        let logits = arena.tensor(
            LOGITS_DTYPE,
            ops::Shape::new(&[rows, u64::from(self.model.vocabulary())]),
        )?;
        self.model
            .compute_logits(forward, gathered, logits)
            .map_err(Error::from)?;
        // The sampler reads its request state per row: `parameters` is `[rows, PARAMETER_COLUMNS]`
        // and `slots` is `[rows]`. A bucket narrower than the largest reads its own rows from the
        // column's head - a captured graph addresses the column's base pointer, so the view states
        // the bucket's shape and not the column's flat extent, which is the largest bucket's.
        let parameters = self
            .columns
            .parameters
            .shaped(&[rows, u64::from(ops::sampler::PARAMETER_COLUMNS)]);
        let slots = self.columns.slots.shaped(&[rows]);
        let state = SamplerState {
            counts: sampling.counts,
            parameters,
            slots,
            extras: None,
        };
        let samples = Samples::reserve(arena, DType::I32, rows, false)?;
        // A greedy bucket takes each row's largest candidate; every other sampling bucket draws one
        // from the row's distribution. The two are different launch sequences in the sampler layer
        // (`Sampler::greedy` and `Sampler::forward`), and a draw cannot produce a greedy token: the
        // candidates are ordered by token id before the inverse CDF, so a draw of zero selects the
        // lowest id rather than the largest logit.
        if bucket.variant.greedy() {
            sampling.sampler.greedy(forward, logits, &state, &samples)?;
        } else {
            sampling.sampler.forward(
                forward,
                logits,
                &state,
                self.columns.draws.shaped(&[rows]),
                &samples,
            )?;
        }
        // A bucket that draws may end a greedy request's prompt, whose first token is the argmax of
        // the row - a decision the graph cannot make for one row of a bucket that draws. The view
        // is kept (the arena's buffer is live until the next `begin`, and `read_sampled` runs before
        // that); a greedy bucket's own launch already states every token it keeps.
        let logits = (bucket.variant == crate::plan::Variant::PrefillOutput).then_some(logits);
        // A recording's views address no memory, so only a launching walk names where a replay's
        // tokens land.
        if !forward.is_recording() {
            self.bucket_samples.borrow_mut().insert(
                bucket,
                Sampled {
                    tokens: samples.tokens,
                    logits,
                },
            );
        }
        self.commit_step(forward, bucket, &samples)?;
        Ok(())
    }

    /// Launches the token-state commit of `bucket`'s rows: their sampled tokens advance their
    /// slots' history, so the next step's penalties are computed against it (4.4, `ops::commit`).
    ///
    /// It runs inside the bucket's capture, after the sampler and before the read-back, so a replay
    /// advances the state exactly once. A batch with no commit table (a shadow batch, a chunk that
    /// commits nothing) launches nothing, and the graph recorded for a bucket always carries the
    /// launch, because the warm pose stages an inert table rather than an empty one.
    fn commit_step(&self, forward: &Forward<'_>, bucket: Bucket, samples: &Samples) -> Result<()> {
        if self.staging.commit().is_empty() {
            return Ok(());
        }
        let rows = u64::from(bucket.rows);
        let descriptor = commit_descriptor(
            self.commit.capacity,
            self.commit.block,
            self.model.vocabulary(),
        );
        let slots = u64::from(self.slots.max(1));
        let args = ops::commit::CommitArgs {
            columns: self.commit.table_view(),
            tokens: samples.tokens,
            valid: self.commit.valid.extent(0..rows),
            active: self.commit.active.extent(0..rows),
            continuation: self.commit.continuation.extent(0..1),
            points: self.commit.points.extent(0..1),
            future: TensorView::with_strides(
                self.commit.future.device.ptr,
                DType::I32,
                ops::Shape::new(&[slots, 1]),
                ops::Strides::new(&[1, 1]),
            ),
            predicates: self.commit.predicates.extent(0..slots),
            logical: self.commit.logical.extent(0..slots),
            sampling: self.commit.sampling.extent(0..slots),
            verified: self.commit.verified.extent(0..slots),
            penalties: self
                .sampling
                .as_ref()
                .map(|sampling| sampling.counts)
                .ok_or_else(|| Error::State("a sampling batch has no count table"))?,
            count: rows,
        };
        forward
            .operate::<ops::commit::SlotCommit>(&descriptor, args)
            .map_err(Error::from)
    }

    /// Stages a batch of the bucket's full size with nothing to write: what a warmup capture
    /// walks (the same allocations, the same launches as a served batch of that bucket).
    fn stage_warm(&mut self, bucket: Bucket) -> Result<Shape> {
        let tokens = bucket.tokens as usize;
        let rows = bucket.rows as usize;
        let span = warm_span(bucket);
        let columns = BatchColumns {
            rows: BatchRows {
                positions: vec![0; tokens],
                slot_mapping: vec![-1; tokens],
                // The partition the launch reads, and the calls below are the same shape a serving
                // batch states: `warm_span` and `warm_calls` state each rule once, with the reason
                // beside it, because both were once the prefill form applied to a decode bucket.
                seq_lens: span.0,
                query_offsets: span.1,
                logits_rows: vec![-1; rows],
                calls: warm_calls(bucket),
            },
            input_ids: vec![0; tokens],
            slots: vec![0; rows],
            // A warm pose of a sampled bucket must state a valid *draw*: the sampler's forward
            // divides by the row's temperature, so a zeroed row asks it to divide by zero - and a
            // capture that does so is a device nobody can tell from a hung one. The permissive
            // pose keeps every candidate (`top_k` 0), takes all of the nucleus (`top_p` 1) and
            // applies no penalty, which is a request the sampler answers and whose token is
            // discarded. A greedy bucket's pose is not read through this path at all: the launch
            // sequence follows the bucket's variant.
            parameters: warm_parameters(rows),
            // The pose's token is discarded, so any address serves; one per row, as the batch states.
            draws: vec![ipc::philox::DrawAddress::default(); rows],
            // An inert table, not an empty one, with one row per call: a row whose slot and task
            // are `-1` commits nothing, and the launch is still recorded, so a serving replay of
            // this bucket runs the operator over the bucket's rows and advances nothing.
            commit: inert_commit(if bucket.variant.is_decode() { rows } else { 1 }),
            // The page size and key bound the upload hands the model, the same two values the
            // serving pose states (`token_batch_with`): a captured node's page size and key bound
            // are its own, so a warm pose that stated its own would freeze a page of one token and a reduction
            // of one key into the graph every served batch then replays. `page_size` is the
            // cache's and `max_kv_len` the deployment's; neither is a property of a bucket, so no pose
            // states one of its own.
            page_size: self.kv.page_size(),
            max_kv_len: self.max_kv_len,
        };
        self.staging
            .stage(&columns, &warm_call_ids(bucket), &bucket.shape(), bucket)
    }
}

/// The non-negative length a per-slot table column holds for `slot`, or zero when the table has no
/// such slot: a length is never negative, and a padded row names no slot at all.
/// The state each call of a staged batch reports, from the commit table's readback.
///
/// The completion names one row per call and the state it carries is the call's **own row of the
/// commit table**: the table is per call, while the sampler's `slots` column is per row, so a
/// multi-row call reads the slot its rows share through the first row it owns. A call whose slot is
/// below zero names no state at all and reports the default.
fn states_of(
    slots: &[i32],
    rows: &BatchRows,
    calls: usize,
    logical: &[i32],
    sampling: &[i32],
) -> Vec<RowState> {
    let mut states = Vec::with_capacity(calls);
    for index in 0..calls {
        let row = rows
            .calls
            .get(index)
            .map(|call| call.rows.start as usize)
            .unwrap_or(0);
        states.push(if slots.get(row).copied().unwrap_or(-1) < 0 {
            RowState::default()
        } else {
            RowState {
                // For text the call's trajectory position is the number of tokens it has committed,
                // which the table keeps in the sampling column; a latent trajectory advances the
                // same column one step at a time.
                step_index: length_at(sampling, index),
                cache_computed_len: length_at(logical, index),
                cache_visible_len: length_at(sampling, index),
            }
        });
    }
    states
}

/// One row of a commit table column: the call at `row`, whose lengths a completion reports.
///
/// A row the table does not hold reads as zero rather than as a neighbour's value, so a batch that
/// outgrew its bucket is a state of zero length and not another call's trajectory.
fn length_at(column: &[i32], row: usize) -> u32 {
    column.get(row).copied().unwrap_or(0).max(0) as u32
}

/// Names each call's tokens and trajectory position: the step's output in the mechanism's order
/// (4.4).
///
/// `sampling_rows` are the batch rows a token is sampled for, in batch order; `calls` name the row
/// ranges of the batch, and `steps` carry the same length as `calls`: the trajectory position the
/// commit table left each call's slot at. A call with no sampling row still gets an entry - it
/// advanced, it just ended no prompt - and a sampling row outside every call is an invariant
/// violation, because the mechanism's columns and its calls would then disagree about the batch.
/// The completion of one step: one [`CallOutput`] per call, in the batch's call order.
///
/// `calls` and `states` are per **call**, and `ids` and `sampling_rows` are per *row* (the bucket's
/// rows, one sampled id and one `logits_rows` entry each). A sample belongs to the call that owns
/// its batch row; a row whose logits entry is negative contributes nothing.
pub fn calls_output(
    ids: &[i32],
    sampling_rows: &[i32],
    rows: &BatchRows,
    calls: &[CallId],
    states: &[RowState],
) -> Result<Vec<CallOutput>> {
    if ids.len() < sampling_rows.len() {
        return Err(Error::Invariant(format!(
            "the bucket sampled {} rows and the batch names {}",
            ids.len(),
            sampling_rows.len()
        )));
    }
    if states.len() != calls.len() {
        return Err(Error::Invariant(format!(
            "the batch names {} rows and {} commit positions",
            calls.len(),
            states.len()
        )));
    }
    let mut output: Vec<CallOutput> = calls
        .iter()
        .zip(states)
        .map(|(call, state)| CallOutput::completed(*call, Vec::new(), *state))
        .collect();
    for (index, logits_row) in sampling_rows.iter().copied().enumerate() {
        // `logits_rows` names, for the batch row at `index`, the token row whose logits it samples
        // from; a negative entry is a row that samples nothing (a padded row, or a prefill chunk
        // that ends no prompt). The sample belongs to the **batch row**, which is the position -
        // attributing it to the row it read its logits from would hand one request another
        // request's token as soon as a batch's logits layout is not the identity.
        if logits_row < 0 {
            continue;
        }
        let logits_row = logits_row as u32;
        if !rows
            .calls
            .iter()
            .any(|call| call.rows.contains(&logits_row))
        {
            return Err(Error::Invariant(format!(
                "sampling row {logits_row} names a token row the batch does not hold"
            )));
        }
        let row = index as u32;
        let (position, _) = rows
            .calls
            .iter()
            .enumerate()
            .find(|(_, call)| call.rows.contains(&row))
            .ok_or_else(|| {
                Error::Invariant(format!("no call of the batch owns sampling row {row}"))
            })?;
        let entry = output.get_mut(position).ok_or_else(|| {
            Error::Invariant(format!("the batch names no call id for its row {position}"))
        })?;
        entry.tokens.push(ids[index] as u32);
    }
    Ok(output)
}

/// The call ids of a bucket's warm pose: one per call `warm_calls` states.
///
/// `Staging::stage` refuses a pose whose call ids do not name every call of its rows, and a decode
/// bucket's warm pose states one call per row - so a single `CallId(0)` is a pose of one call
/// wearing eight. The ids are positional: a call's id names its slot in the batch's tables, and a
/// warm pose has no request to carry one.
fn warm_call_ids(bucket: Bucket) -> Vec<CallId> {
    warm_calls(bucket)
        .into_iter()
        .enumerate()
        .map(|(index, _)| CallId(index as u64))
        .collect()
}

/// The calls a warm pose states: one per row for a decode bucket, one over the run for a prefill.
///
/// A decode row is one request's one token, and `Staging::stage` extends a batch to one call per
/// padded row in exactly this form, so a decode pose that stated one call holding the whole token run
/// would describe a batch no serving batch has - and a capture states its pose to the model for the
/// life of the bucket. A prefill row genuinely carries a run, so its pose states one call over the
/// bucket's tokens.
fn warm_calls(bucket: Bucket) -> Vec<BatchRow> {
    let rows = bucket.rows;
    if bucket.variant.is_decode() {
        (0..rows)
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
            rows: 0..bucket.tokens,
        }]
    }
}

/// The span a warm pose states: `seq_lens` per row, and `query_offsets` per row boundary.
///
/// A prefill row carries a multi-token run, so the first row holds the bucket's whole extent and the
/// rest hold none - a partition of `num_tokens` into one run. **A decode row is one query token of
/// its own sequence**, so its partition is the identity: row `i` holds `cum_seq_lens_q[i]..[i + 1]`,
/// which is one token. Stating the prefill form for a decode bucket tells the attention kernel that
/// row 0 holds every query while the query tensor holds one row per token, and a kernel given a
/// partition no tensor has computes a tiling and a mask that no buffer matches.
fn warm_span(bucket: Bucket) -> (Vec<i32>, Vec<i32>) {
    let rows = usize::try_from(bucket.rows).unwrap_or(0);
    let tokens = bucket.tokens as i32;
    // Every warm row's slot is `-1`, and `resident.rs:258-273` states the rule that goes with it: an
    // absent row is filled with page zero - the pool's reserved padding unit - **and its `seq_lens`
    // entry states the same zero**, because a row that names page zero and claims a key makes the
    // kernel read the padding unit as if a request had written it. That is a launch with no request
    // behind it, and at more than one row it is an address the pool does not own.
    let lengths = vec![0_i32; rows];
    let offsets = if bucket.variant.is_decode() {
        (0..=rows as i32).collect()
    } else {
        let mut offsets = vec![tokens; rows + 1];
        if let Some(first) = offsets.first_mut() {
            *first = 0;
        }
        offsets
    };
    (lengths, offsets)
}

/// The sampling parameters a warm pose states: one valid, permissive draw per row.
///
/// `TEMPERATURE` is 1 rather than 0 because the sampler's forward divides by it - a zeroed row
/// asks for a division by zero, and the capture would hang or produce a value no kernel can use.
/// `TOP_K` 0 keeps every candidate, `TOP_P` 1 keeps all of the nucleus, `MIN_P` 0 keeps every
/// candidate above it, and the three penalties are 0. The token the pose draws is discarded.
fn warm_parameters(rows: usize) -> Vec<f32> {
    let columns = ops::sampler::PARAMETER_COLUMNS as usize;
    let mut parameters = vec![0.0; rows * columns];
    for row in 0..rows {
        let row = &mut parameters[row * columns..(row + 1) * columns];
        row[ops::sampler::TEMPERATURE as usize] = 1.0;
        row[ops::sampler::TOP_P as usize] = 1.0;
    }
    parameters
}

/// Refuses a decode batch whose rows disagree with the bucket's sampling path.
///
/// A bucket's variant fixes the sampler launch sequence for every row: [`Variant::DecodeGreedy`]
/// argmaxes each row's penalized logits and [`Variant::DecodeSampled`] draws one token per row from
/// its distribution. Neither sequence can serve a row of the other class - a draw cannot produce a
/// greedy token (the candidates are ordered by token id before the inverse CDF, so a draw selects an
/// id, not the largest logit), and an argmax is not what a row that states a temperature asked for.
///
/// Which requests share a batch is the batcher's decision, so the composition is not the worker's
/// to change: it states that the batch it was handed cannot be served, naming the row and the
/// bucket, rather than serving a token the request did not ask for. Prefill buckets are exempt - a
/// prompt-ending greedy row inside a drawing prefill bucket is ordinary, and its first token is the
/// host argmax of the row's kept logits (`greedy_first_tokens`).
fn refuse_mixed_sampling(variant: crate::plan::Variant, parameters: &[f32]) -> Result<()> {
    if !variant.is_decode() {
        return Ok(());
    }
    let columns = ops::sampler::PARAMETER_COLUMNS as usize;
    let temperature = ops::sampler::TEMPERATURE as usize;
    let rows = parameters.len() / columns;
    for (row, stated) in parameters.chunks_exact(columns).enumerate() {
        let greedy_row = stated[temperature] == 0.0;
        if greedy_row == variant.greedy() {
            continue;
        }
        return Err(Error::Invariant(format!(
            "a batch is homogeneous in its sampling path: the {} bucket holds {rows} row(s) and row \
             {row} states temperature {}, so its token is {} and this bucket has no launch sequence \
             for it",
            variant.name(),
            stated[temperature],
            if greedy_row {
                "the argmax of its penalized logits"
            } else {
                "drawn from its distribution"
            },
        )));
    }
    Ok(())
}

/// The rows of the `[tokens, hidden]` stack the sampler gathers, one per batch row.
///
/// `logits_rows` names, per row, the row whose logits are wanted and is negative where nothing is
/// sampled; the token whose logits that row samples is the **last token of the row's own query
/// range**, which `query_offsets` states. The two indices agree only when every row holds one
/// token - a decode - so a batch whose rows carry several tokens reads its prefill from the wrong
/// row without this mapping.
///
/// A row's range ends at its own last real token, and padding is inside no span, so the range's end
/// is the token to read. The clamp to the batch's live token count is a belt around that rule rather
/// than a second rule: a span that ever reached past the batch's tokens would read the bucket's
/// padding, which is what a widened span used to do. A row that samples nothing gathers the stack's
/// first token, whose id the sampler discards.
fn gather_indices(
    logits_rows: &[i32],
    query_offsets: &[i32],
    live_tokens: usize,
    rows: usize,
    out: &mut Vec<i32>,
) -> Result<()> {
    if logits_rows.len() < rows || query_offsets.len() < rows + 1 {
        return Err(Error::Invalid(format!(
            "a batch of {rows} rows states {} logits rows and {} query offsets",
            logits_rows.len(),
            query_offsets.len()
        )));
    }
    let live = i32::try_from(live_tokens)
        .map_err(|_| Error::Invalid("a batch of more tokens than an i32 counts".into()))?;
    out.clear();
    for row in 0..rows {
        out.push(if logits_rows[row] < 0 {
            0
        } else {
            // The batch's own tokens end at `live_tokens`, and only the last row's range was
            // extended past them.
            query_offsets[row + 1].min(live).saturating_sub(1).max(0)
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Bucket, Variant, warm_span};

    #[test]
    fn a_decode_bucket_partitions_its_queries_one_per_row() {
        // The defect this guards: the pose gave row 0 the whole run, which is the prefill partition,
        // and the attention kernel then computed four queries in row 0 of a tensor holding one row
        // per token - the launch that faulted at four rows and never at one.
        let bucket = Bucket::decode(4, Variant::DecodeSampled);
        let (seq_lens, query_offsets) = warm_span(bucket);
        assert_eq!(
            query_offsets,
            vec![0, 1, 2, 3, 4],
            "a decode row is one query token, so the partition is the identity"
        );
        assert_eq!(
            seq_lens,
            vec![0, 0, 0, 0],
            "a warm row's slot is absent, and an absent row claims no key"
        );
    }

    #[test]
    fn a_decode_bucket_states_one_call_per_row() {
        // The second pose defect: one call holding the bucket's whole token run is a shape no
        // serving decode batch has, and a capture states its pose to the model for the bucket's
        // life. A decode row is one request's one token.
        let bucket = Bucket::decode(4, Variant::DecodeSampled);
        let calls = super::warm_calls(bucket);
        assert_eq!(calls.len(), 4, "one call per decode row");
        for (row, call) in calls.iter().enumerate() {
            assert_eq!(call.kind, RowKind::Decode);
            assert_eq!(
                call.rows,
                row as u32..row as u32 + 1,
                "a decode call holds one token"
            );
        }
    }

    #[test]
    fn a_prefill_bucket_gives_its_first_row_the_whole_run() {
        // A prefill row carries a multi-token run: the partition puts every token in row 0 and the
        // reach of every other row at zero, which is what the kernel reads to split the run.
        let bucket = Bucket::prefill(2, 512, Variant::PrefillOutput);
        let (seq_lens, query_offsets) = warm_span(bucket);
        assert_eq!(query_offsets, vec![0, 512, 512]);
        assert_eq!(
            seq_lens,
            vec![0, 0],
            "a warm row's slot is absent, and an absent row claims no key"
        );
    }

    /// A prefill's row is its whole prompt, and the sampler must read the row of the stack that
    /// token ends: the gather index is the last token of the row's own range, not the row's own
    /// index. A decode is the case where the two agree, which is why reading the row index fails
    /// only on a prefill and only from the first token on.
    #[test]
    fn a_prefill_samples_the_last_token_of_its_row_and_not_the_rows_own_index() {
        use super::gather_indices;

        let mut indices = Vec::new();
        // One call of three prompt tokens, ending its prompt, in a bucket whose token axis holds
        // eight: `stage` extends the row's range over the five padding tokens, so the row's range
        // ends at the bucket and the batch's own count is what names the last real token.
        gather_indices(&[0], &[0, 8], 3, 1, &mut indices).expect("a prefill");
        assert_eq!(indices, vec![2], "the prefill samples its last real token");

        // Two calls, one of three prompt tokens and one of a single token, the last padded.
        gather_indices(&[0, 1], &[0, 3, 8], 4, 2, &mut indices).expect("two prefills");
        assert_eq!(indices, vec![2, 3]);

        // A decode of three one-token rows: the row index is the token index, and a decode bucket
        // extends no row's range over padding rows.
        gather_indices(&[0, 1, 2], &[0, 1, 2, 3], 3, 3, &mut indices).expect("a decode");
        assert_eq!(indices, vec![0, 1, 2]);

        // A chunked prefill that ends no prompt samples nothing, and its row gathers an in-range
        // token the sampler discards, so the row after it still reads its own last token.
        gather_indices(&[-1, 1], &[0, 2, 8], 3, 2, &mut indices).expect("a chunk and a row");
        assert_eq!(indices, vec![0, 2]);

        // Fewer columns than rows is refused rather than read past.
        assert!(gather_indices(&[0], &[0, 3], 3, 2, &mut indices).is_err());
    }

    use super::*;

    /// A decode batch is homogeneous in its sampling path: a row of the other class is refused by
    /// name rather than served by a launch sequence that cannot produce its token. A prefill bucket
    /// is exempt, because a prompt-ending greedy row is ordinary there.
    #[test]
    fn a_decode_bucket_refuses_a_row_of_the_other_sampling_class() {
        let columns = ops::sampler::PARAMETER_COLUMNS as usize;
        let temperature = ops::sampler::TEMPERATURE as usize;
        // One row a stated temperature, the rest of its columns the reference values: 8 floats a row,
        // as the worker writes them.
        let rows = |stated: &[f32]| {
            stated
                .iter()
                .flat_map(|stated| {
                    let mut row = vec![1.0_f32; columns];
                    row[temperature] = *stated;
                    row
                })
                .collect::<Vec<f32>>()
        };

        assert!(
            refuse_mixed_sampling(crate::plan::Variant::DecodeGreedy, &rows(&[0.0, 0.0])).is_ok(),
            "two greedy rows were refused by a greedy bucket"
        );
        assert!(
            refuse_mixed_sampling(crate::plan::Variant::DecodeSampled, &rows(&[0.7, 1.0])).is_ok(),
            "two sampled rows were refused by a sampling bucket"
        );
        assert!(
            refuse_mixed_sampling(crate::plan::Variant::PrefillOutput, &rows(&[0.0, 0.7])).is_ok(),
            "a prompt-ending greedy row was refused inside a drawing prefill bucket"
        );

        let error = refuse_mixed_sampling(crate::plan::Variant::DecodeSampled, &rows(&[0.7, 0.0]))
            .expect_err("a greedy row was drawn from in a sampling bucket");
        let text = error.to_string();
        assert!(
            text.contains("decode_sampled") && text.contains("row 1"),
            "the refusal names neither the bucket nor the row: {text}"
        );
        assert!(
            text.contains("temperature 0"),
            "the refusal does not name the temperature the row stated: {text}"
        );

        let error = refuse_mixed_sampling(crate::plan::Variant::DecodeGreedy, &rows(&[0.5]))
            .expect_err("a sampled row was argmaxed in a greedy bucket");
        assert!(
            error.to_string().contains("decode_greedy"),
            "the refusal does not name the bucket: {error}"
        );
    }

    /// The homogeneity check reads the **batch's** rows, and the rows the bucket pads around them
    /// are not rows of the batch: a padded row's parameters are zero, and zero reads as a greedy
    /// row, so a check handed the bucket's rows would refuse a perfectly homogeneous sampled batch
    /// for the rows nobody sent.
    ///
    /// This is the same boundary the capture's failure was on, with a boolean instead of a count:
    /// the figure the check reads is the batch's, the figure the graph is built from is the bucket's.
    #[test]
    fn the_rows_a_bucket_pads_are_not_rows_of_the_batch() -> Result<()> {
        let buckets = Buckets::new([
            Bucket::decode(2, Variant::DecodeSampled),
            Bucket::decode(4, Variant::DecodeSampled),
        ]);
        let mut staging = Staging::new();
        staging.reserve(&buckets)?;
        let bucket = Bucket::decode(4, Variant::DecodeSampled);
        let mut columns = decode_columns();
        let stated = ops::sampler::PARAMETER_COLUMNS as usize;
        for row in 0..2 {
            columns.parameters[row * stated + ops::sampler::TEMPERATURE as usize] = 0.7;
        }
        let shape = Shape::decode(2, Variant::DecodeSampled);
        staging.stage(&columns, &[CallId(0), CallId(1)], &shape, bucket)?;

        // The batch's own rows: both sampled, so the bucket takes them.
        assert!(
            refuse_mixed_sampling(bucket.variant, staging.parameters(shape.rows as usize)).is_ok(),
            "a homogeneous sampled batch was refused for rows it did not send"
        );
        // The bucket's rows: the two the staging padded state temperature 0, so they read as greedy
        // rows of a sampling bucket. That is what the capture used to be refused by.
        let error = refuse_mixed_sampling(bucket.variant, staging.parameters(bucket.rows as usize))
            .expect_err("a padded row read as a row of the batch");
        let text = error.to_string();
        assert!(
            text.contains("row 2") && text.contains("temperature 0"),
            "the refusal does not name the first padded row: {text}"
        );
        Ok(())
    }

    /// The cache-unit seam's two decisions, over the allocation alone: a batch that reads a fresh
    /// prefix passes through every provider, and one that reuses a leased one is refused by name by
    /// a provider that holds no cache. The non-empty refusal is the state until the resident cache
    /// applies the copies; what matters here is that it can never be a silent pass.
    #[test]
    fn a_provider_without_a_cache_serves_an_empty_allocation_and_refuses_a_prefix_copy() {
        assert!(
            apply_cache_units_unavailable(&[]).is_ok(),
            "a batch reading a fresh prefix was refused"
        );
        let allocation = vec![ipc::CacheUnitAllocation {
            request_pool_idx: 0,
            group_id: 0,
            unit_ids: vec![ipc::UnitId(7)],
            copies: vec![(ipc::UnitId(3), ipc::UnitId(7))],
        }];
        let error = apply_cache_units_unavailable(&allocation)
            .expect_err("a prefix copy was dropped without a word");
        assert!(
            error.to_string().contains("never copied"),
            "the refusal does not say what would go wrong: {error}"
        );
    }

    /// The host argmax of a greedy row is the same rule the captured `argmax` operator applies: the
    /// largest finite value, ties to the lowest index, and a row of non-finite values answering
    /// candidate zero rather than an arbitrary index.
    ///
    /// The end-to-end proof - a greedy request whose prompt spans several prefill chunks taking the
    /// argmax of that chunk's row - belongs to the caller's run, which must compare against a
    /// recomputed argmax and not merely against a rerun of this code: a drawn token is just as
    /// deterministic.
    #[test]
    fn the_host_argmax_is_the_largest_finite_candidate_and_ties_go_to_the_lowest() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[3.0, 1.0, 1.0]), 0);
        // Two equal maxima: the first wins, so two runs and two implementations agree.
        assert_eq!(argmax(&[0.5, 2.0, 2.0]), 1);
        // A non-finite value is never a candidate, at the head or in the middle.
        assert_eq!(argmax(&[f32::NAN, 1.5, 1.0]), 1);
        assert_eq!(argmax(&[1.0, f32::INFINITY, 2.0]), 2);
        // Every value non-finite: answer zero rather than an index that came from nowhere.
        assert_eq!(argmax(&[f32::NAN, f32::NAN]), 0);
    }

    /// A provider that holds no cache refuses a slot row by name, because the gather would otherwise
    /// read whatever the row held - another request's pages, or zeros - and the graph would run clean
    /// against the wrong memory.
    #[test]
    fn a_provider_without_a_cache_serves_a_batch_with_no_tables_and_refuses_a_slot_row() {
        assert!(
            install_block_tables_unavailable(&[]).is_ok(),
            "a batch whose slots have no pages was refused"
        );
        let tables = vec![ipc::BlockTable {
            request_pool_idx: 3,
            group_id: 1,
            start_page: 0,
            unit_ids: vec![ipc::UnitId(11), ipc::UnitId(12)],
            allocated_tokens: 256,
        }];
        let error = install_block_tables_unavailable(&tables)
            .expect_err("a slot row was dropped without a word");
        let message = error.to_string();
        assert!(
            message.contains("slot 3") && message.contains("group 1"),
            "the refusal does not name the row: {message}"
        );
        assert!(
            message.contains("would read whatever that row held"),
            "the refusal does not say what would go wrong: {message}"
        );
    }

    /// A mechanism commit table of one row per call, column-major `[COMMIT_FIELDS][rows]`: the one
    /// column a fixture states is the row's destination slot, and the rest keep zero.
    fn committed(slots: &[i64]) -> Vec<i64> {
        let rows = slots.len();
        let mut table = vec![0_i64; ops::commit::COMMIT_FIELDS * rows];
        table[..rows].copy_from_slice(slots);
        table
    }

    fn decode_columns() -> BatchColumns {
        BatchColumns {
            rows: BatchRows {
                positions: vec![3, 4],
                slot_mapping: vec![10, 11],
                seq_lens: vec![4, 9],
                query_offsets: vec![0, 1, 2],
                logits_rows: vec![0, 1],
                calls: vec![
                    BatchRow {
                        request: 0,
                        kind: RowKind::Decode,
                        cache_group: 0,
                        rows: 0..1,
                    },
                    BatchRow {
                        request: 1,
                        kind: RowKind::Decode,
                        cache_group: 0,
                        rows: 1..2,
                    },
                ],
            },
            input_ids: vec![7, 8],
            slots: vec![0, 1],
            parameters: vec![0.0; 2 * ops::sampler::PARAMETER_COLUMNS as usize],
            draws: vec![ipc::philox::DrawAddress::default(); 2],
            commit: committed(&[0, 1]),
            page_size: 16,
            max_kv_len: 4096,
        }
    }

    fn prefill_columns() -> BatchColumns {
        BatchColumns {
            rows: BatchRows {
                positions: vec![0, 1, 2],
                slot_mapping: vec![0, 1, 2],
                seq_lens: vec![3],
                query_offsets: vec![0, 3],
                logits_rows: vec![2],
                calls: vec![BatchRow {
                    request: 0,
                    kind: RowKind::Prefill,
                    cache_group: 0,
                    rows: 0..1,
                }],
            },
            input_ids: vec![1, 2, 3],
            slots: vec![0],
            parameters: vec![0.0; ops::sampler::PARAMETER_COLUMNS as usize],
            draws: vec![ipc::philox::DrawAddress::default()],
            commit: committed(&[0]),
            page_size: 16,
            max_kv_len: 4096,
        }
    }

    /// A batch is padded to its bucket: no padding token writes the KV cache, a decode's padding
    /// rows commit nothing, and a prefill's padding tokens extend the one call's range.
    #[test]
    fn padding_never_writes_the_cache_or_commits_a_row() -> Result<()> {
        let buckets = Buckets::new([
            Bucket::decode(4, Variant::DecodeGreedy),
            Bucket::prefill(4, 64, Variant::PrefillOutput),
        ]);
        let mut staging = Staging::new();
        staging.reserve(&buckets)?;

        let bucket = Bucket::decode(4, Variant::DecodeGreedy);
        let staged = staging.stage(
            &decode_columns(),
            &[CallId(4), CallId(5)],
            &Shape::decode(2, Variant::DecodeGreedy),
            bucket,
        )?;
        assert_eq!((staged.rows, staged.tokens), (4, 4));
        // No padding token persists and every padding row keeps its token state.
        assert_eq!(&staging.batch_rows().slot_mapping[..], &[10, 11, -1, -1]);
        assert_eq!(&staging.slots(4)[2..], &[-1, -1]);
        assert_eq!(&staging.commit()[..4], &[0, 1, -1, -1]);
        // A decode's padding rows do not sample.
        assert_eq!(&staging.logits_rows()[2..], &[-1, -1]);

        let bucket = Bucket::prefill(4, 64, Variant::PrefillOutput);
        let staged = staging.stage(
            &prefill_columns(),
            &[CallId(9)],
            &Shape::prefill(1, 3, Variant::PrefillOutput),
            bucket,
        )?;
        assert_eq!(staged.tokens, 64);
        assert_eq!(&staging.batch_rows().slot_mapping[3..], &[-1i32; 61]);
        Ok(())
    }

    /// A prefill bucket's padding tokens are **tokens the graph can serve**, not sentinels.
    ///
    /// The captured graph runs the router over every token the bucket holds, so a padded token's
    /// route is computed from its own row: its id and its rotary position must be values a kernel
    /// may index with, and the row's answer is discarded rather than left unset. A padding value a
    /// kernel would index with - a `-1` id, or a row whose attention output nothing writes - is how
    /// a padded token reached the MoE block as a route nobody could serve and faulted the launch
    /// (`moeUnpermuteKernel`, block 289 of a 512-token bucket, the first padded token of the
    /// prompt).
    #[test]
    fn a_prefill_bucket_pads_with_tokens_the_graph_can_serve() -> Result<()> {
        let bucket = Bucket::prefill(1, 64, Variant::PrefillOutput);
        let buckets = Buckets::new([bucket]);
        let mut staging = Staging::new();
        staging.reserve(&buckets)?;
        staging.stage(
            &prefill_columns(),
            &[CallId(1)],
            &Shape::prefill(1, 3, Variant::PrefillOutput),
            bucket,
        )?;

        let tokens = bucket.tokens as usize;
        assert_eq!(staging.input_ids.len(), tokens);
        // The three real tokens, then padding that is a *token*: id zero of the vocabulary and
        // rotary position zero, so the embedding and every position-dependent launch after it read
        // a value rather than a sentinel. No padding token writes a cache page.
        assert_eq!(&staging.input_ids[..3], &[1, 2, 3]);
        for token in 3..tokens {
            assert_eq!(
                staging.input_ids[token], 0,
                "padded token {token} is a real id, not a sentinel"
            );
            assert_eq!(
                staging.batch_rows().positions[token],
                0,
                "padded token {token} has a real rotary position"
            );
            assert_eq!(
                staging.batch_rows().slot_mapping[token],
                -1,
                "padded token {token} writes no cache page"
            );
        }
        // The call's span is its three real tokens: padding is inside no span, so a reader that
        // takes a row's query length from these offsets is told three and not the bucket's sixty
        // four. Everything that must work over the bucket's token axis is bounded by the batch's
        // live count instead (`staging.live_tokens()`), which is one bound with one meaning.
        assert_eq!(&staging.batch_rows().query_offsets[..], &[0, 3]);
        assert_eq!(staging.live_tokens(), 3);
        Ok(())
    }

    /// The live token count is the batch's own, not the bucket's: a captured graph reads the
    /// bucket's count as a constant, and a kernel that must bound its work - the MoE sort writes a
    /// route sentinel into every row of a token the batch does not own - reads this count instead.
    ///
    /// The batch here is the one the delivery run meets: 289 tokens in a 512-token bucket, so 223
    /// padded tokens follow it.
    #[test]
    fn the_live_token_count_is_the_batchs_and_not_its_buckets() -> Result<()> {
        let tokens = 289i32;
        let bucket = Bucket::prefill(1, 512, Variant::PrefillOutput);
        let buckets = Buckets::new([bucket]);
        let mut staging = Staging::new();
        staging.reserve(&buckets)?;
        let count = tokens as usize;
        let mut columns = BatchColumns {
            rows: BatchRows {
                positions: (0..tokens).collect(),
                slot_mapping: (0..tokens).collect(),
                seq_lens: vec![tokens],
                query_offsets: vec![0, tokens],
                logits_rows: vec![tokens - 1],
                calls: vec![BatchRow {
                    request: 0,
                    kind: RowKind::Prefill,
                    cache_group: 0,
                    rows: 0..1,
                }],
            },
            input_ids: (0..tokens).map(|token| token + 1).collect(),
            slots: vec![0],
            parameters: vec![0.0; ops::sampler::PARAMETER_COLUMNS as usize],
            draws: vec![ipc::philox::DrawAddress::default()],
            commit: committed(&[0]),
            page_size: 16,
            max_kv_len: 4096,
        };
        columns.input_ids.truncate(count);
        staging.stage(
            &columns,
            &[CallId(1)],
            &Shape::prefill(1, tokens as u32, Variant::PrefillOutput),
            bucket,
        )?;

        assert_eq!(
            staging.live_tokens(),
            count as u64,
            "the staged count is the batch's, not the {} of the bucket",
            512
        );
        assert_eq!(
            staging.input_ids.len(),
            512,
            "the bucket's tokens are staged"
        );
        // The padded tail is still a token whose output is discarded, and the count is the only
        // statement of where the batch ends.
        assert_eq!(staging.input_ids[count], 0);
        assert_eq!(staging.batch_rows().slot_mapping[count], -1);
        assert_eq!(staging.input_ids[511], 0);
        assert_eq!(staging.batch_rows().slot_mapping[511], -1);
        Ok(())
    }

    /// A batch narrower than its bucket fills the bucket's rows with **empty sequences**: the row
    /// count of every per-row operand is the bucket's, and the rows the batch does not fill state
    /// length zero, no pool slot, no logits row and inert sampling parameters.
    ///
    /// This is the warmup's normal case, not an edge case: a bucket is a catalogue size and a batch
    /// is whatever arrived. A capture is of the bucket, so the extent of `seq_lens`,
    /// `query_offsets`, `logits_rows`, the sampler's rows and the commit's count is fixed at capture
    /// and must be the bucket's at every replay.
    #[test]
    fn the_rows_a_batch_does_not_fill_are_empty_sequences() -> Result<()> {
        let buckets = Buckets::new([
            Bucket::decode(4, Variant::DecodeGreedy),
            Bucket::decode(8, Variant::DecodeGreedy),
        ]);
        let mut staging = Staging::new();
        staging.reserve(&buckets)?;
        let bucket = Bucket::decode(8, Variant::DecodeGreedy);
        staging.stage(
            &decode_columns(),
            &[CallId(0), CallId(1)],
            &Shape::decode(2, Variant::DecodeGreedy),
            bucket,
        )?;

        let rows = bucket.rows as usize;
        // The bucket's extents, not the batch's four.
        assert_eq!(staging.batch_rows().seq_lens.len(), rows);
        assert_eq!(staging.batch_rows().logits_rows.len(), rows);
        assert_eq!(staging.batch_rows().query_offsets.len(), rows + 1);
        assert_eq!(staging.slots(rows).len(), rows);
        // Rows 2..8 are empty sequences: length zero, no slot, no logits row, no sampling state.
        assert_eq!(
            &staging.batch_rows().seq_lens[..],
            &[4, 9, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(&staging.slots(rows)[2..], &[-1; 6]);
        assert_eq!(&staging.logits_rows()[2..], &[-1; 6]);
        assert!(
            staging.parameters(rows)[2 * ops::sampler::PARAMETER_COLUMNS as usize..]
                .iter()
                .all(|value| *value == 0.0),
            "an empty row states sampling parameters"
        );
        // The pack ends where the batch's last token ends and stays there: an empty row owns no
        // token, so every offset past the batch's rows repeats the end of the packed tokens.
        assert_eq!(
            &staging.batch_rows().query_offsets[..],
            &[0, 1, 2, 2, 2, 2, 2, 2, 2]
        );
        Ok(())
    }

    /// The commit launch is well formed for the rows a bucket holds, and its table is the
    /// mechanism's in the layout the operator states: `[COMMIT_FIELDS, capacity]` i32,
    /// column-major, the staged rows first.
    ///
    /// These are the operator's own conditions (`SlotCommit::launch` checks them and refuses with
    /// `Error::argument`), stated here because a wrong block or stride is otherwise discovered as a
    /// failed launch inside a captured bucket on a device.
    #[test]
    fn the_commit_launch_is_well_formed_for_a_bucket() -> Result<()> {
        let rows = 3;
        let capacity = 4;
        let vocabulary = 1_024;
        let descriptor = commit_descriptor(capacity as u64, commit_block(rows as u32), vocabulary);
        assert_eq!(descriptor.index, DType::I32);
        assert_eq!(descriptor.verified, DType::I32);
        assert_eq!(descriptor.penalties, DType::I32);
        // The table's leading stride is its capacity, the stride `table_view` states.
        assert_eq!(descriptor.stride, capacity as u64);
        // A single next-input token per slot is a `[slots, 1]` matrix whose rows are one apart.
        assert_eq!(descriptor.future_stride, 1);
        assert_eq!(descriptor.vocab, vocabulary);
        // The block covers the step's rows and is a power of two.
        assert!(descriptor.block.is_power_of_two());
        assert!(u64::from(descriptor.block) >= rows as u64);
        assert_eq!(commit_block(1), 1);
        assert_eq!(commit_block(5), 8);

        // The mechanism's `[fields, rows]` table, column-major, becomes `[fields, capacity]`.
        let mut staged = vec![0_i64; ops::commit::COMMIT_FIELDS * rows];
        for row in 0..rows {
            staged[ops::commit::column::SLOT * rows + row] = row as i64 + 7;
            staged[ops::commit::column::MODE * rows + row] = ops::commit::mode::DECODE;
        }
        let table = commit_table(&staged, rows, capacity)?;
        assert_eq!(table.len(), ops::commit::COMMIT_FIELDS * capacity);
        let slots = &table
            [ops::commit::column::SLOT * capacity..(ops::commit::column::SLOT + 1) * capacity];
        assert_eq!(&slots[..rows], &[7, 8, 9]);
        // The row past the batch is **inert**, not zero: a zero slot is slot 0, a real destination,
        // so a launch that ever ran over the whole capacity would commit phantom tokens into the
        // first slot the deployment serves. `-1` is the operator's sentinel for "no slot", and the
        // tail is `capacity - rows` rows of it - one here, not two.
        assert_eq!(&slots[rows..], &[-1]);
        // A table of the wrong length is refused rather than read past its end.
        assert!(commit_table(&staged[..rows], rows, capacity).is_err());
        Ok(())
    }

    /// The runner's own operator needs one key per block its buckets reach, not one per bucket:
    /// the table's capacity is the largest bucket's rows for every bucket, so eight blocks cover
    /// the default catalog's rows.
    #[test]
    fn the_commit_keys_are_one_per_block() {
        let buckets = Buckets::causal_lm(&[1, 2, 4, 8, 16, 32, 64, 128], &[64, 128], false);
        let descriptors = commit_descriptors(&buckets, 1_024);
        let blocks: Vec<u32> = descriptors
            .iter()
            .map(|descriptor| descriptor.block)
            .collect();
        assert_eq!(blocks, vec![1, 2, 4, 8, 16, 32, 64, 128]);
        // One capacity for all of them: the largest bucket's rows, which is the table's stride.
        for descriptor in &descriptors {
            assert_eq!(descriptor.stride, 128);
            assert_eq!(descriptor.vocab, 1_024);
            assert_eq!(descriptor.index, DType::I32);
        }
        // One key per block a catalog's rows fall into: rows 1 and 3 land in blocks 1 and 4, and a
        // catalog whose rows share one block needs one key.
        let two_blocks = Buckets::causal_lm(&[1, 3], &[4], false);
        assert_eq!(commit_descriptors(&two_blocks, 1_024).len(), 2);
        let one_block = Buckets::causal_lm(&[3], &[4], false);
        assert_eq!(commit_descriptors(&one_block, 1_024).len(), 1);
    }

    /// A warm pose's commit table is inert and complete: every column is present, no row names a
    /// slot or a task, and nothing counts a penalty. The launch is recorded and the state stays as
    /// it was, so the first request does not start from a history warmup invented.
    #[test]
    fn a_warm_pose_commits_nothing_but_is_recorded() {
        let rows = 5;
        let table = inert_commit(rows);
        assert_eq!(table.len(), ops::commit::COMMIT_FIELDS * rows);
        for field in 0..ops::commit::COMMIT_FIELDS {
            let column = &table[field * rows..(field + 1) * rows];
            if field == ops::commit::column::SLOT || field == ops::commit::column::TASK {
                assert_eq!(column, &[-1_i64; 5], "field {field} names no destination");
            } else {
                assert!(
                    column.iter().all(|value| *value == 0),
                    "field {field} is zero"
                );
            }
        }
    }

    /// The staged ids are the batch's, in order, padded after them: the forward embeds these, so a
    /// row's embedding comes from its own token and a padding token embeds nothing a request sees.
    #[test]
    fn the_staged_ids_are_the_batchs_in_order() -> Result<()> {
        let buckets = Buckets::new([
            Bucket::decode(4, Variant::DecodeGreedy),
            Bucket::prefill(4, 64, Variant::PrefillOutput),
        ]);
        let mut staging = Staging::new();
        staging.reserve(&buckets)?;

        staging.stage(
            &prefill_columns(),
            &[CallId(3)],
            &Shape::prefill(1, 3, Variant::PrefillOutput),
            Bucket::prefill(4, 64, Variant::PrefillOutput),
        )?;
        assert_eq!(&staging.input_ids(3), &[1, 2, 3]);

        // A decode's two rows keep their tokens and its padding rows carry none of them.
        staging.stage(
            &decode_columns(),
            &[CallId(4), CallId(5)],
            &Shape::decode(2, Variant::DecodeGreedy),
            Bucket::decode(4, Variant::DecodeGreedy),
        )?;
        assert_eq!(&staging.input_ids(2), &[7, 8]);
        assert_eq!(staging.input_ids(4).len(), 4);
        Ok(())
    }

    /// A completion reports the commit table's row of its own call. The slot a call writes to is a
    /// state slot, not a row of the table: a batch whose slots are numbered past its calls must not
    /// read another call's lengths, nor a row the table does not hold.
    #[test]
    fn a_completion_reads_its_own_row_of_the_commit_table() {
        let rows = BatchRows {
            positions: vec![1, 2],
            slot_mapping: vec![0, 1],
            seq_lens: vec![2, 3],
            query_offsets: vec![0, 1, 2],
            logits_rows: vec![0, 1],
            calls: vec![
                BatchRow {
                    request: 0,
                    kind: RowKind::Decode,
                    cache_group: 0,
                    rows: 0..1,
                },
                BatchRow {
                    request: 1,
                    kind: RowKind::Decode,
                    cache_group: 0,
                    rows: 1..2,
                },
            ],
        };
        // Slots 9 and 4 write state nine and four: neither is a row of the two-call table.
        let states = states_of(&[9, 4], &rows, 2, &[3, 5], &[3, 6]);
        assert_eq!(
            states,
            vec![
                RowState {
                    step_index: 3,
                    cache_computed_len: 3,
                    cache_visible_len: 3
                },
                RowState {
                    step_index: 6,
                    cache_computed_len: 5,
                    cache_visible_len: 6
                },
            ]
        );
    }

    /// A call that evaluates several rows reports one state, the one the commit table wrote for the
    /// call, and a call whose slot names no state reports nothing.
    #[test]
    fn a_multi_row_call_reports_one_state_and_no_slot_reports_none() {
        let rows = BatchRows {
            positions: vec![9, 10, 11],
            slot_mapping: vec![0, 1, 2],
            seq_lens: vec![1, 1, 1],
            query_offsets: vec![0, 1, 2, 3],
            logits_rows: vec![0, 1, 2],
            calls: vec![BatchRow {
                request: 0,
                kind: RowKind::Verify,
                cache_group: 0,
                rows: 0..3,
            }],
        };
        assert_eq!(
            states_of(&[2, 2, 2], &rows, 1, &[7], &[11]),
            vec![RowState {
                step_index: 11,
                cache_computed_len: 7,
                cache_visible_len: 11
            }]
        );
        assert_eq!(
            states_of(&[-1], &rows, 1, &[7], &[11]),
            vec![RowState::default()]
        );
    }

    /// A batch larger than the bucket it planned is refused before any staging or launch.
    #[test]
    fn a_batch_larger_than_its_bucket_is_refused() {
        let buckets = Buckets::new([Bucket::decode(2, Variant::DecodeGreedy)]);
        let mut staging = Staging::new();
        staging.reserve(&buckets).expect("one bucket");
        let bucket = Bucket::decode(2, Variant::DecodeGreedy);
        staging
            .stage(
                &decode_columns(),
                &[CallId(0)],
                &Shape::decode(4, Variant::DecodeGreedy),
                bucket,
            )
            .expect_err("two rows do not fit a bucket of two, and four do not either");
    }
}

#[cfg(test)]
mod sampled_output_tests {
    use super::*;

    fn rows(calls: &[(u32, std::ops::Range<u32>)]) -> BatchRows {
        BatchRows {
            positions: Vec::new(),
            slot_mapping: Vec::new(),
            seq_lens: Vec::new(),
            query_offsets: Vec::new(),
            logits_rows: Vec::new(),
            calls: calls
                .iter()
                .map(|(request, rows)| BatchRow {
                    request: *request,
                    kind: RowKind::Decode,
                    cache_group: 0,
                    rows: rows.clone(),
                })
                .collect(),
        }
    }

    /// The step's output names every call of the batch, and each call's tokens are its own rows'
    /// samples in row order, whatever order the mechanism listed the calls in.
    #[test]
    fn sampled_rows_are_named_by_their_calls() {
        let batch = rows(&[(0, 0..1), (1, 1..2)]);
        let output = calls_output(
            &[111, 222],
            &[1, 0],
            &batch,
            &[CallId(7), CallId(9)],
            &[
                RowState {
                    step_index: 3,
                    cache_computed_len: 3,
                    cache_visible_len: 3,
                },
                RowState {
                    step_index: 4,
                    cache_computed_len: 4,
                    cache_visible_len: 4,
                },
            ],
        )
        .expect("both rows belong to a call");
        assert_eq!(
            output,
            vec![
                CallOutput::completed(
                    CallId(7),
                    vec![111],
                    RowState {
                        step_index: 3,
                        cache_computed_len: 3,
                        cache_visible_len: 3
                    },
                ),
                CallOutput::completed(
                    CallId(9),
                    vec![222],
                    RowState {
                        step_index: 4,
                        cache_computed_len: 4,
                        cache_visible_len: 4
                    },
                ),
            ]
        );
    }

    /// A call whose rows sample nothing still completes: it advanced, it just ended no prompt. The
    /// readback still holds a token per bucket row - the sampler writes one whatever the row is -
    /// and the row's `-1` is what discards it, so no call ever receives it.
    #[test]
    fn a_call_that_samples_nothing_still_completes() {
        let output = calls_output(
            &[0],
            &[-1],
            &rows(&[(0, 0..4)]),
            &[CallId(1)],
            &[RowState {
                step_index: 7,
                cache_computed_len: 4,
                cache_visible_len: 4,
            }],
        )
        .expect("no sampling rows");
        assert_eq!(
            output,
            vec![CallOutput::completed(
                CallId(1),
                Vec::new(),
                RowState {
                    step_index: 7,
                    cache_computed_len: 4,
                    cache_visible_len: 4
                }
            )]
        );
    }

    /// The columns are sized by the widest token count and the widest row count, which can come from
    /// two different buckets: a plan holding a long narrow prefill bucket and a short wide decode
    /// bucket has no bucket that is largest in both. A column sized from one bucket alone would be
    /// shorter than the batches that address it, and nothing bounds an extent view against it.
    ///
    /// No constructor in the tree builds such a set - `Buckets::causal_lm` crosses every row size
    /// with every token size - so this is the rule stated rather than a defect a deployment reached.
    #[test]
    fn the_columns_cover_both_maxima_and_not_one_bucket() {
        let buckets = Buckets::new([
            Bucket::prefill(1, 4096, Variant::PrefillWrite),
            Bucket::decode(64, Variant::DecodeGreedy),
        ]);
        let (tokens, rows) = column_extents(&buckets).expect("a plan of two buckets");
        assert_eq!(tokens, 4096, "the longest bucket names the token columns");
        assert_eq!(rows, 64, "the widest bucket names the row columns");
        // The mistake this states against: one bucket read for both figures.
        let widest = buckets
            .iter()
            .max_by_key(|bucket| (bucket.rows, bucket.tokens))
            .copied()
            .expect("a bucket");
        assert_ne!(
            (widest.tokens as u64, widest.rows as u64),
            (tokens, rows),
            "no single bucket holds both maxima, which is why one is not read for both"
        );
    }

    /// A sampling row outside every call, or fewer sampled ids than sampling rows, is an invariant
    /// violation: silently dropping either would hand the mechanism the wrong request's token.
    #[test]
    fn a_row_no_call_owns_is_refused() {
        let batch = rows(&[(0, 0..1)]);
        calls_output(&[5], &[3], &batch, &[CallId(1)], &[RowState::default()])
            .expect_err("row 3 belongs to no call");
        calls_output(&[], &[0], &batch, &[CallId(1)], &[RowState::default()])
            .expect_err("no id was sampled for row 0");
        calls_output(&[5], &[0], &batch, &[CallId(1)], &[])
            .expect_err("one call, no trajectory position");
    }

    /// A sampling bucket narrower than the largest reads its own rows: the state the sampler is
    /// handed is `[rows, PARAMETER_COLUMNS]` over the column's head, and `[rows]` for its slots and
    /// draws, never the column's flat extent. A captured graph addresses the column's base pointer,
    /// so the two views differ in shape alone - and a flat one states the largest bucket's width,
    /// which the sampler refuses by the shape it expected.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_narrower_sampling_bucket_reads_its_own_rows() {
        let columns = u64::from(ops::sampler::PARAMETER_COLUMNS);
        let largest = 8;
        let narrow = 2;
        let parameters = Column::new(platform::DeviceId(0), DType::F32, largest * columns)
            .expect("the parameters column");
        assert_eq!(parameters.device.shape.dims(), &[largest * columns]);
        let view = parameters.shaped(&[narrow, columns]);
        assert_eq!(view.shape.dims(), &[narrow, columns]);
        assert_eq!(
            view.ptr.address(),
            parameters.device.ptr.address(),
            "the bucket's view names the column's own base, which is what a captured graph addresses"
        );
        let slots =
            Column::new(platform::DeviceId(0), DType::I32, largest).expect("the slots column");
        assert_eq!(slots.shaped(&[narrow]).shape.dims(), &[narrow]);
        let draws =
            Column::new(platform::DeviceId(0), DType::F32, largest).expect("the draws column");
        assert_eq!(draws.shaped(&[narrow]).shape.dims(), &[narrow]);
        // The gather column is the same shape one operator earlier: `IndexRows::gather` takes a flat
        // `[count]` view and derives `count` from the rows it gathers, so a flat column states the
        // largest bucket's row count and every narrower bucket is refused before the sampler.
        let indices =
            Column::new(platform::DeviceId(0), DType::I32, largest).expect("the indices column");
        let gather = indices.shaped(&[narrow]);
        assert_eq!(gather.shape.dims(), &[narrow]);
        assert_eq!(gather.numel(), narrow, "the count the gather reads");
    }

    /// The warm pose of a sampled bucket states a draw the sampler can make: temperature 1, every
    /// candidate kept. A zeroed pose asks the forward to divide by a zero temperature - and it is
    /// the very input `refuse_mixed_sampling` refuses in a real batch, which is why a capture that
    /// ran on it was a bucket the batch check would have rejected.
    #[test]
    fn a_warm_pose_states_a_draw_a_sampled_bucket_can_make() {
        let columns = ops::sampler::PARAMETER_COLUMNS as usize;
        let parameters = warm_parameters(2);
        assert_eq!(parameters.len(), 2 * columns, "a row per row");
        for row in 0..2 {
            let row = &parameters[row * columns..(row + 1) * columns];
            assert_eq!(row[ops::sampler::TEMPERATURE as usize], 1.0);
            assert_eq!(row[ops::sampler::TOP_P as usize], 1.0);
            assert_eq!(row[ops::sampler::TOP_K as usize], 0.0);
        }
        refuse_mixed_sampling(crate::plan::Variant::DecodeSampled, &parameters)
            .expect("the pose of a sampled bucket is not a greedy row");
        assert!(
            refuse_mixed_sampling(crate::plan::Variant::DecodeSampled, &vec![0.0; 2 * columns])
                .is_err(),
            "the zeroed pose the capture used to state is not a draw"
        );
    }

    /// A prompt ending's own token is one the batch committed, so the readback the greedy host path
    /// reads answers 1 for it and not 0. Refusing on that raw figure refused every deployment's
    /// first token (the run at `091731ffb`: `a greedy row of slot 1 ends a prompt that already holds 1
    /// committed token(s)`), and a slot that really did hold tokens is still refused.
    #[test]
    fn a_prompt_ending_holds_no_committed_token_before_its_own() {
        assert_eq!(committed_before(1), 0, "a prompt ending's own token only");
        assert_eq!(committed_before(5), 4, "four tokens before this step's");
        assert_eq!(committed_before(0), 0, "an unread slot has none");
        assert_eq!(
            committed_before(-1),
            0,
            "a readback that never ran has none"
        );
    }

    /// **A greedy prompt ending is argmaxed on the host.** The row is greedy (temperature zero), it
    /// ends a prompt (the readback states the position after its own commit, so one means none
    /// before it), and its token is the argmax of its own logits row - the same token the captured
    /// sampler would take, because every penalty is a no-op for a slot with no history.
    ///
    /// The device half - the copy of the row out of the arena and the readback the mechanism reads -
    /// is the caller's run; what is stated here is the decision and the argmax it rests on.
    #[test]
    fn a_greedy_prompt_ending_is_argmaxed_on_the_host() {
        assert_eq!(
            prompt_token(false, 0.0, 3, 1).expect("a prompt ending that committed only its own"),
            PromptToken::HostArgmax
        );
        // The token is the largest value of the row, and a tie is the lowest index.
        assert_eq!(argmax(&[0.5, 9.0, 9.0, -3.0]), 1);
        // A non-finite value is never the largest: the first finite value wins over a NaN before it,
        // and a row with no finite value has no largest candidate to name.
        assert_eq!(argmax(&[f32::NAN, 2.0, 1.5]), 1);
        assert_eq!(argmax(&[f32::NAN, f32::NEG_INFINITY]), 0);
        assert_eq!(argmax(&[f32::NAN, f32::NEG_INFINITY, f32::NAN]), 0);
    }

    /// **A sampled prompt ending is NOT argmaxed**: its first token is the graph's draw over the
    /// penalized logits. The host leaves the drawn token alone, and the decision is taken before any
    /// history is consulted - a sampled slot may hold as much history as it has decoded, because the
    /// captured sampler is what applies the penalties.
    #[test]
    fn a_sampled_prompt_ending_is_drawn_and_not_argmaxed() {
        for readback in [1, 9, -1] {
            assert_eq!(
                prompt_token(false, 0.7, 3, readback)
                    .expect("a sampled row is decided by the draw"),
                PromptToken::Draw,
                "a sampled prompt ending was argmaxed on the host, which would answer the largest                  candidate instead of a token of its distribution"
            );
        }
        // A row that ends no prompt is discarded whatever its temperature: the graph drew for it in
        // a bucket that draws, and the draw belongs to no call.
        for temperature in [0.0, 1.0] {
            assert_eq!(
                prompt_token(true, temperature, 3, 1).expect("a row that ends no prompt"),
                PromptToken::Discarded
            );
        }
    }

    /// **A greedy row with a committed history is refused by name.** Its token is the argmax of the
    /// *penalized* logits, which only the captured sampler computes; argmaxing the raw row would
    /// answer a different request's token, so the row is refused with both figures rather than
    /// served with the wrong one.
    #[test]
    fn a_greedy_row_with_a_committed_history_is_refused_by_name() {
        let refusal = prompt_token(false, 0.0, 7, 5).expect_err("four tokens before this step's");
        assert!(
            matches!(refusal, Error::Invalid(_)),
            "a row that cannot be decided here was not stated as a refusal: {refusal:?}"
        );
        let text = refusal.to_string();
        assert!(
            text.contains("slot 7") && text.contains("4 committed token"),
            "the refusal names neither the slot nor the history it found: {text}"
        );
    }
}
