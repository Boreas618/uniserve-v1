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

mod speculation;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ipc::{PhaseId, TimingCounters};
use nn::family::TextModel;
use nn::{CausalLm, Forward, KvView, TokenBatch};
use ops::{DType, Generation, Operand};
use platform::DeviceId;

use crate::arena::{Pools, StreamId};
use crate::batch::{Job, Jobs};
use crate::plan::{Bucket, Buckets};
use ipc::execution::Shape;
use ipc::execution::{
    Batch, BatchColumns, BatchId, BatchOutput, BatchRow, BatchRows, CallId, CallOutput, RowKind,
    RowState,
};

use crate::runner::{Binding, Runner, RunnerId};
use crate::runners::columns::{Column, encode_f32, encode_i32};
use crate::runners::lifecycle::{Lifecycle, walk_capture};
use crate::runners::recording::{Footprint, Walk};
use crate::runners::sampling::{
    Adjustments, CommitColumns, SamplerState, Samples, Sampling, Scores, commit_descriptor,
    inert_commit, scored,
};
#[cfg(test)]
use ipc::execution::Variant;

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
    /// `[tokens]` the prompt token each token's logits score, `-1` for one that scores none and
    /// for padding (`BatchColumns::targets`).
    targets: Vec<i32>,
    slots: Vec<i32>,
    /// `[rows]` each row's recurrent-state checkpoint, zero for padding and for a row that keeps
    /// none.
    checkpoints: Vec<i32>,
    /// The batch's recurrent-state copies, which the replay launches around its forward.
    state_copies: Vec<ipc::execution::StateCopy>,
    /// Each call's draft cache work, empty for a batch whose calls draft nothing.
    drafts: Vec<ipc::DraftWork>,
    parameters: Vec<f32>,
    /// `[rows]` each row's draw address; a padded row's is the default, and its draw is discarded.
    draws: Vec<ipc::philox::DrawAddress>,
    /// The batch's logit adjustments, encoded as the sampler reads them.
    adjustments: Adjustments,
    /// The request slots whose penalty counts the batch starts before it samples.
    fresh_slots: Vec<ipc::execution::FreshSlot>,
    commit: Vec<i64>,
    page_size: u32,
    max_kv_len: u32,
    /// The mechanism's call ids of the staged rows, in the order `rows.calls` lists them.
    calls: Vec<CallId>,
    /// The tokens the batch really holds, before the padding this staging appends to reach the
    /// bucket's count. A captured graph states the bucket's token count as a constant, so a kernel
    /// that must bound its work by what the batch holds reads this from the device instead.
    live_tokens: i32,
    /// The image blocks the batch's rows write, in token order.
    vision: Vec<ipc::execution::VisionRows>,
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

    /// Sizes the buffers for the largest bucket `buckets` holds, over a `vocabulary`-wide logit row.
    pub fn reserve(&mut self, buckets: &Buckets, vocabulary: u32) -> Result<()> {
        let largest = buckets
            .iter()
            .max_by_key(|bucket| (bucket.rows, bucket.tokens))
            .copied()
            .ok_or_else(|| Error::Invalid("the runner has no bucket to serve".into()))?;
        self.input_ids = vec![0; largest.tokens as usize];
        self.targets = vec![-1; largest.tokens as usize];
        self.slots = vec![0; largest.rows as usize];
        self.checkpoints = vec![0; largest.rows as usize];
        self.state_copies = Vec::new();
        self.parameters =
            vec![0.0; largest.rows as usize * proto::program::sampler::PARAMETER_COLUMNS as usize];
        self.draws = vec![ipc::philox::DrawAddress::default(); largest.rows as usize];
        self.adjustments.reserve(largest.rows as usize, vocabulary);
        self.fresh_slots = Vec::with_capacity(largest.rows as usize);
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
        // A batch that scores nothing states no targets, and padding scores nothing.
        self.targets[..bucket_tokens].fill(-1);
        if !columns.targets.is_empty() {
            self.targets[..tokens].copy_from_slice(&columns.targets[..tokens]);
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
        // A model without recurrent state states no checkpoint column; every row then keeps none.
        self.checkpoints[..bucket_rows].fill(0);
        if !columns.checkpoints.is_empty() {
            self.checkpoints[..rows].copy_from_slice(&columns.checkpoints[..rows]);
        }
        self.state_copies.clear();
        self.state_copies.extend_from_slice(&columns.state_copies);
        self.drafts.clear();
        self.drafts.extend_from_slice(&columns.drafts);
        self.vision.clear();
        self.vision.extend_from_slice(&columns.vision);
        let parameters = proto::program::sampler::PARAMETER_COLUMNS as usize;
        self.parameters[..rows * parameters]
            .copy_from_slice(&columns.parameters[..rows * parameters]);
        // A greedy row keeps one candidate: its largest. A greedy bucket argmaxes and reads no
        // width, and a drawing bucket that holds the row - a prompt ending beside a drawn one, a
        // decode beside a drawn decode - then draws from that one candidate, which is the argmax
        // of the row's shaped logits with ties to the lowest id (`top_k` orders ties by id), the
        // token the greedy launch sequence chooses.
        for row in self.parameters[..rows * parameters].chunks_exact_mut(parameters) {
            if row[proto::program::sampler::TEMPERATURE as usize] == 0.0 {
                row[proto::program::sampler::TOP_K as usize] = 1.0;
            }
        }
        if columns.draws.len() != rows {
            return Err(Error::Invalid(format!(
                "the batch states {} draw addresses for its {rows} rows",
                columns.draws.len()
            )));
        }
        self.draws[..rows].copy_from_slice(&columns.draws);
        self.adjustments
            .stage(&columns.adjustments, rows, bucket_rows)?;
        self.fresh_slots.clear();
        self.fresh_slots.extend_from_slice(&columns.fresh_slots);
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
            let fields = proto::program::commit::COMMIT_FIELDS;
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

    /// The staged prompt targets, one per token.
    pub fn targets(&self, tokens: usize) -> &[i32] {
        &self.targets[..tokens]
    }

    /// The staged sampler slot view.
    pub fn slots(&self, rows: usize) -> &[i32] {
        &self.slots[..rows]
    }

    /// The staged recurrent-state checkpoints, one per row.
    pub fn checkpoints(&self, rows: usize) -> &[i32] {
        &self.checkpoints[..rows]
    }

    /// The staged calls' draft cache work, empty when no call drafts.
    pub fn drafts(&self) -> &[ipc::DraftWork] {
        &self.drafts
    }

    /// The staged batch's recurrent-state copies.
    pub fn state_copies(&self) -> &[ipc::execution::StateCopy] {
        &self.state_copies
    }

    /// The image blocks the staged batch's rows write.
    pub fn vision(&self) -> &[ipc::execution::VisionRows] {
        &self.vision
    }

    /// The staged sampling parameters, row-major.
    pub fn parameters(&self, rows: usize) -> &[f32] {
        let parameters = proto::program::sampler::PARAMETER_COLUMNS as usize;
        &self.parameters[..rows * parameters]
    }

    /// The staged rows' draw addresses.
    pub fn draws(&self, rows: usize) -> &[ipc::philox::DrawAddress] {
        &self.draws[..rows]
    }

    /// The staged batch's logit adjustments.
    pub fn adjustments(&self) -> &Adjustments {
        &self.adjustments
    }

    /// The request slots the staged batch starts a request's sampling state on.
    pub fn fresh_slots(&self) -> &[ipc::execution::FreshSlot] {
        &self.fresh_slots
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
    /// `[tokens]` i32 the prompt token each token's logits score, which a scoring bucket reads.
    targets: Column,
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
    /// `[rows, 2]` i32 each row's logit adjustments: its bias flag and its ban mask's index.
    adjustments: Column,
    /// `[rows, ceil(vocabulary / 32)]` i32 the batch's ban masks, one per row that bans a token.
    bans: Column,
    /// `[1]` i32 the tokens the batch really holds, which a kernel bounds its work by: the bucket's
    /// own token count is a constant inside a captured graph and cannot bound anything.
    live_tokens: Column,
    /// `[rows]` the sampled ids, read back after a replay: two, as the commit's positions are.
    readback: [platform::PinnedBuffer; 2],
    /// `[rows]` f32 the raw log-probability of each sampled id, read back beside it.
    logprob_readback: [platform::PinnedBuffer; 2],
    /// `[rows]` i32 each sampled id's rank in its raw distribution, read back beside it.
    rank_readback: [platform::PinnedBuffer; 2],
    /// `[rows, candidates]` f32 then `[rows, candidates]` i32: each row's most likely tokens, read
    /// back beside them, for a runner whose sampler reports candidates ([`CausalLmRunner::reporting`]).
    top_readback: Option<[platform::PinnedBuffer; 2]>,
    /// `[tokens]` f32, `[tokens]` i32, then `[tokens, candidates]` f32 and i32: a scoring bucket's
    /// prompt scores, read back beside its samples ([`Variant::PrefillScored`]).
    prompt_readback: [platform::PinnedBuffer; 2],
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
    fn staged(&mut self) -> [&mut Column; 14] {
        [
            &mut self.positions,
            &mut self.input_ids,
            &mut self.targets,
            &mut self.slot_mapping,
            &mut self.seq_lens,
            &mut self.query_offsets,
            &mut self.logits_rows,
            &mut self.indices,
            &mut self.parameters,
            &mut self.slots,
            &mut self.draws,
            &mut self.adjustments,
            &mut self.bans,
            &mut self.live_tokens,
        ]
    }

    /// Allocates the columns over the widest token count and row count `buckets` hold, with
    /// `vocabulary` as the width of the logits row the greedy first token is taken from.
    fn new(device: DeviceId, buckets: &Buckets, vocabulary: u32) -> Result<Self> {
        let (tokens, rows) = column_extents(buckets)?;
        let parameters = rows * u64::from(proto::program::sampler::PARAMETER_COLUMNS);
        Ok(Self {
            positions: Column::new(device, DType::I32, tokens)?,
            input_ids: Column::new(device, DType::I32, tokens)?,
            targets: Column::new(device, DType::I32, tokens)?,
            slot_mapping: Column::new(device, DType::I32, tokens)?,
            scratch_slots: Column::new(device, DType::I32, tokens)?,
            seq_lens: Column::new(device, DType::I32, rows)?,
            query_offsets: Column::new(device, DType::I32, rows + 1)?,
            logits_rows: Column::new(device, DType::I32, rows)?,
            indices: Column::new(device, DType::I32, rows)?,
            parameters: Column::new(device, DType::F32, parameters)?,
            slots: Column::new(device, DType::I32, rows)?,
            draws: Column::new(device, DType::F32, rows)?,
            adjustments: Column::new(device, DType::I32, 2 * rows)?,
            bans: Column::new(
                device,
                DType::I32,
                rows * u64::from(ops::sampler::ban_words(vocabulary)),
            )?,
            live_tokens: Column::new(device, DType::I32, 1)?,
            readback: [
                platform::PinnedBuffer::new(device, (rows * 4) as usize)?,
                platform::PinnedBuffer::new(device, (rows * 4) as usize)?,
            ],
            logprob_readback: [
                platform::PinnedBuffer::new(device, (rows * 4) as usize)?,
                platform::PinnedBuffer::new(device, (rows * 4) as usize)?,
            ],
            rank_readback: [
                platform::PinnedBuffer::new(device, (rows * 4) as usize)?,
                platform::PinnedBuffer::new(device, (rows * 4) as usize)?,
            ],
            top_readback: None,
            prompt_readback: [
                platform::PinnedBuffer::new(device, (tokens * 8) as usize)?,
                platform::PinnedBuffer::new(device, (tokens * 8) as usize)?,
            ],
        })
    }

    /// Sizes the candidates' readback for `candidates` most likely tokens a row, over the widest
    /// row count `buckets` hold; none for a sampler that reports none.
    fn report(&mut self, device: DeviceId, buckets: &Buckets, candidates: u32) -> Result<()> {
        let (tokens, rows) = column_extents(buckets)?;
        let bytes = usize::try_from(2 * rows * u64::from(candidates) * 4)
            .map_err(|_| Error::Invalid("a candidates readback larger than memory".into()))?;
        self.top_readback = match bytes {
            0 => None,
            bytes => Some([
                platform::PinnedBuffer::new(device, bytes)?,
                platform::PinnedBuffer::new(device, bytes)?,
            ]),
        };
        // A scored token's log-probability and rank, then its candidates' values and ids.
        let prompt = usize::try_from(tokens * (8 + 8 * u64::from(candidates)))
            .map_err(|_| Error::Invalid("a prompt scores readback larger than memory".into()))?;
        self.prompt_readback = [
            platform::PinnedBuffer::new(device, prompt)?,
            platform::PinnedBuffer::new(device, prompt)?,
        ];
        Ok(())
    }
}

/// The rows the largest bucket of `buckets` holds, which sizes the columns bound once.
fn largest_rows(buckets: &Buckets) -> Result<u32> {
    buckets
        .iter()
        .map(|bucket| bucket.rows)
        .max()
        .ok_or_else(|| Error::Invalid("the runner has no bucket to serve".into()))
}

/// Elements one program of a step's gather copies.
const GATHER_BLOCK: u32 = 1024;

/// The tokens one chunk of a scoring bucket's prompt projection holds: its two `[rows, vocabulary]`
/// f32 buffers are 39 MB each over a 151936-token vocabulary, and sixty-four rows keep the
/// candidates' `top_k`, one program a row, over most of a device's multiprocessors.
const SCORE_ROWS: u64 = 64;

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
    /// `[tokens]` the prompt targets a scoring bucket scored, empty for any other.
    targets: Vec<i32>,
    rows: BatchRows,
    calls: Vec<CallId>,
}

/// One paired forward whose pairs are on their way to the host, and what its comparison is about.
struct PendingCompare {
    issued: crate::shadow::Issued,
    operator: &'static str,
    key: String,
    bucket: Bucket,
    generation: u64,
}

/// Where one bucket's sampler wrote its ids: the first `rows` i32 of the plan's sampler block.
#[derive(Clone, Copy)]
struct Sampled {
    /// `[rows]` i32 sampled token ids, in the bucket's row order.
    tokens: Operand,
    /// Each sampled id's scores under the raw logits.
    scores: Scores,
    /// `[tokens]` each token's prompt target's scores, in a bucket that scores its prompt.
    prompt: Option<Scores>,
}

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

/// The most nearly square patch grid of an image of `rows` merged rows, `merge` patches to a
/// merged side, near `side` patches a side: the largest image the recorded walk encodes, whose
/// tower activations bound every other image's.
fn largest_grid(rows: u32, merge: u32, side: u32) -> [u32; 3] {
    let merge = merge.max(1);
    let mut height = (side / merge).max(1);
    while height > 1 && !rows.is_multiple_of(height) {
        height -= 1;
    }
    let width = (rows / height).max(1);
    [1, height * merge, width * merge]
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
    fn table_source(&self, group: usize, rows: u32) -> Result<Option<(Operand, Operand)>> {
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
    /// loads the forward reads; a staged load converts with the programs of `kernels`, the
    /// generation the batch runs. Answers the fence of the offloads, which the batch's output
    /// waits on. A provider that holds no cache refuses a copy rather than dropping it.
    fn issue_pages(
        &mut self,
        loads: &[ipc::PageCopy],
        offloads: &[ipc::PageCopy],
        stream: &platform::Stream,
        kernels: &ops::Generation,
    ) -> Result<Option<Arc<platform::Event>>> {
        let _ = (stream, kernels);
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
    ) -> Result<Operand>;

    /// The same table's view over `slots`' rows, with no copy issued.
    ///
    /// A capture records every copy with the addresses it had, so a gather inside a recording bakes
    /// the pose's pool rows into the graph, and each replay would then overwrite the batch's table
    /// with them. A recording addresses the table through this view, and `block_table` fills it
    /// before each launch.
    fn block_table_view(&self, group: usize, slots: &[i32]) -> Result<Operand>;

    /// The plane views of `group`'s layers, in model order.
    fn planes(&self, group: usize) -> Result<Vec<nn::KvPlanes>>;

    /// The plane views of the draft cache's layers, in the drafter's order, as its attention reads
    /// them: pages of one K/V page's tokens ([`crate::memory::prefix_cache::planes::CachePlanes::draft_layer`]).
    /// A provider whose cache keeps no draft group refuses.
    fn draft_planes(&self) -> Result<Vec<nn::KvPlanes>> {
        Err(Error::Invalid(
            "this runner's cache keeps no draft cache".to_owned(),
        ))
    }

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
    ) -> Result<(Operand, Vec<nn::KvPlanes>)>;
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
    /// The last paired forward's comparison, issued and not yet read: it is compared when the next
    /// paired forward has been launched, or when it was the last its shadow batch asked for.
    pending_compare: Option<PendingCompare>,
    /// The paired forwards the shadow batches asked for that have not run yet.
    shadow_jobs_left: usize,
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
    /// The request slots' recurrent-state banks, for a model that keeps a recurrent state: carved
    /// when the device's pools are attached, before the walk and the first capture.
    state: Option<super::state::StateBanks>,
    /// The eager speculative step's columns, in a deployment that speculates with the model's
    /// drafter: every batch is then served by that step rather than a captured graph
    /// (`speculation`).
    speculation: Option<speculation::Speculation>,
    /// The sampler of a speculative step, whose rows count a verification's drafts in their
    /// penalties.
    verifier: Option<super::sampling::Sampler>,
    /// The image input of a model with a vision tower: its encodings and the feature products
    /// its prefills read.
    images: Option<super::vision::Images>,
    /// `[axes · tokens]` i32 the rotary coordinates of a batch with image rows, axis-major over
    /// the bucket's tokens; allocated for a model with a vision tower.
    rope: Option<super::columns::Column>,
    /// The staged prefill's image blocks, resolved to their feature slots at the upload.
    blocks: std::cell::RefCell<Vec<super::vision::Block>>,
    /// Whether a recorded walk walks a bucket as a batch with image rows.
    image_pose: std::cell::Cell<bool>,
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
        ids: ops::Operand,
    ) -> nn::Result<ops::Operand> {
        self.0.embed_input_ids(f, ids)
    }

    fn forward(
        &self,
        f: &nn::Forward,
        batch: &nn::TokenBatch,
        cache: &nn::KvView<'_>,
        hidden: ops::Operand,
    ) -> nn::Result<ops::Operand> {
        self.0.forward(f, batch, cache, hidden)
    }

    fn compute_logits(
        &self,
        f: &nn::Forward,
        hidden: ops::Operand,
    ) -> nn::Result<ops::Operand> {
        self.0.compute_logits(f, hidden)
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

    fn state(&self) -> Option<nn::family::StateShape> {
        self.0.state()
    }

    fn experts(&self) -> Option<nn::family::ExpertShape> {
        self.0.experts()
    }

    fn drafter(&self) -> Option<&dyn nn::family::Drafter> {
        self.0.drafter()
    }

    fn vision(&self) -> Option<&dyn nn::capability::Vision> {
        self.0.vision()
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
        staging.reserve(&buckets, model.vocabulary())?;
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
            pending_compare: None,
            shadow_jobs_left: 0,
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
            state: None,
            speculation: None,
            verifier: None,
            images: None,
            rope: None,
            blocks: std::cell::RefCell::new(Vec::new()),
            image_pose: std::cell::Cell::new(false),
            encoded: Vec::new(),
            indices: Vec::new(),
            draws: Vec::new(),
            kv,
        })
    }

    /// Makes this runner speculate with its model's drafter, verifying at most `drafts` drafts per
    /// call: every batch is served by the eager speculative step, over a cache that keeps the draft
    /// group.
    ///
    /// # Errors
    ///
    /// A model that carries no drafter, a cache without a draft group, a runner in an expert union
    /// (whose ranks step in lockstep through captured graphs), or columns the device cannot hold.
    pub fn speculating(mut self, drafts: u32) -> Result<Self> {
        if self.model.drafter().is_none() {
            return Err(Error::Invalid(
                "the deployment speculates and its model loaded no drafter".into(),
            ));
        }
        if self.experts.is_some() {
            return Err(Error::Unsupported(
                "speculative decoding of a model whose experts are split across ranks".into(),
            ));
        }
        let (tokens, rows) = column_extents(&self.buckets)?;
        let mut speculation = speculation::Speculation::new(
            self.device,
            tokens,
            rows,
            self.slots,
            drafts,
            self.model.vocabulary(),
            self.candidates(),
            self.max_kv_len,
            self.kv.page_size(),
        )?;
        let cache =
            self.kv.cache().cloned().ok_or_else(|| {
                Error::Invalid("a speculating runner reads a prefix cache".into())
            })?;
        speculation.bind(
            &cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )?;
        let vocabulary = self.model.vocabulary();
        self.verifier = Some(
            super::sampling::Sampler::new(vocabulary, self.slots, vocabulary, drafts)?
                .reporting(self.candidates())?,
        );
        self.speculation = Some(speculation);
        Ok(self)
    }

    /// The rows a speculating runner's block tables gather for a batch of at most `tokens`
    /// tokens over `slots` slots verifying `drafts` drafts each: a verification attends as one
    /// row per token.
    pub fn speculative_rows(slots: u32, drafts: u32, rows: u32) -> u32 {
        rows.max(slots.saturating_mul(drafts + 1))
    }

    /// Serves the model's image input: its vision encodings, read on `lane`, and the image rows of
    /// its prefills. A model without a vision tower takes none.
    ///
    /// # Errors
    ///
    /// A model whose vision tower states no image it could encode.
    pub fn with_images(
        mut self,
        lane: crate::host::Lane<super::vision::ImageRead>,
    ) -> Result<Self> {
        let Some(vision) = self.model.vision() else {
            return Ok(self);
        };
        let rows = vision.max_image_rows();
        if rows == 0 {
            return Err(Error::Invalid(
                "the vision tower sizes no image to any row".into(),
            ));
        }
        let (tokens, _) = column_extents(&self.buckets)?;
        self.rope = Some(super::columns::Column::new(
            self.device,
            DType::I32,
            u64::from(vision.position_axes()) * tokens,
        )?);
        self.images = Some(super::vision::Images::new(
            self.device,
            vision.patch_format(),
            self.model.dtype(),
            self.model.hidden(),
            rows,
            lane,
        ));
        Ok(self)
    }

    /// Copies the image encodings from and to the host encoder `pool`, writing those this rank
    /// computes when `writes`.
    #[must_use]
    pub fn with_encoder_pool(
        mut self,
        pool: Arc<crate::memory::encoders::EncoderHostPool>,
        writes: bool,
    ) -> Self {
        if let Some(images) = self.images.as_mut() {
            images.with_pool(pool, writes);
        }
        self
    }

    /// What the runner's image input occupies: the bytes of one feature row, and the rows one
    /// encoding holds at most; `None` for a model without a vision tower.
    pub fn image_features(&self) -> Option<(u64, u32)> {
        self.images
            .as_ref()
            .map(|images| (images.row_bytes(), images.slot_rows()))
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

    /// Makes this runner report each sampled token's `candidates` most likely alternatives beside
    /// it - the deployment's `max_logprobs`, which bounds what a request may ask for - computed in
    /// every sampling bucket's graph. Zero reports none, and adds no launch.
    ///
    /// # Errors
    ///
    /// A count above the vocabulary, or a readback the host cannot pin.
    pub fn reporting(mut self, candidates: u32) -> Result<Self> {
        if let Some(sampling) = self.sampling.take() {
            self.sampling = Some(sampling.reporting(candidates)?);
        }
        if let Some(verifier) = self.verifier.take() {
            self.verifier = Some(verifier.reporting(candidates)?);
        }
        self.columns
            .report(self.device, &self.buckets, candidates)?;
        Ok(self)
    }

    /// The most likely alternatives each sampled token is reported with.
    pub fn candidates(&self) -> u32 {
        self.sampling
            .as_ref()
            .map_or(0, |sampling| sampling.sampler.candidates())
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
    /// Stages a batch of vision encodings: each call's image is read on the host lane, and a
    /// host job per call encodes it on this stream.
    fn submit_images(&mut self, batch: &Batch) -> Result<Jobs> {
        let vision = self.model.vision().ok_or_else(|| {
            Error::Invalid("a vision encoding reached a model without a vision tower".into())
        })?;
        let images = self.images.as_mut().ok_or_else(|| {
            Error::Invalid("a vision encoding reached a runner that serves no image input".into())
        })?;
        images.stage(vision, batch)?;
        Ok(Jobs::new(batch.calls.iter().map(|call| Job::Host {
            runner: self.id,
            call: *call,
        })))
    }

    /// Encodes the staged vision encoding of `call` into its feature product.
    fn encode_image(&mut self, call: CallId) -> Result<BatchOutput> {
        let started = Instant::now();
        let vision = self.model.vision().ok_or_else(|| {
            Error::Invalid("a vision encoding reached a model without a vision tower".into())
        })?;
        let images = self.images.as_mut().ok_or_else(|| {
            Error::Invalid("a vision encoding reached a runner that serves no image input".into())
        })?;
        let index = usize::try_from(call.0)
            .map_err(|_| Error::Invalid("a call index wider than memory".into()))?;
        let kernels = self.lifecycle.generation_named(images.generation())?;
        // The encode is an eager forward over the stream's region, which the recorded walk sized
        // for the largest image; a replay that reads the region runs behind it on this stream.
        let arena = self.lifecycle.arena()?;
        images.encode(
            vision,
            index,
            &self.stream,
            (&arena, &self.lifecycle.staging),
            &kernels,
            self.tensor.as_deref(),
        )?;
        Ok(BatchOutput {
            calls: vec![ipc::execution::CallOutput::completed(
                call,
                Vec::new(),
                ipc::execution::RowState::default(),
            )],
            timing: ipc::TimingCounters {
                host_us: started.elapsed().as_micros() as u64,
                ..Default::default()
            },
            forward_stats: None,
        })
    }

    /// Stages the image rows of the staged batch, whose tokens rotate from `positions`: the blocks'
    /// feature slots, and every token's rotary coordinates on every axis, the image rows' at the
    /// coordinates the model states for their grids. A batch without image rows stages none.
    fn stage_image_rows(&mut self, positions: &[i32]) -> Result<()> {
        let blocks = match self.images.as_ref() {
            Some(images) if !self.staging.vision().is_empty() => {
                images.blocks(self.staging.vision())?
            }
            _ => Vec::new(),
        };
        if !blocks.is_empty()
            && let (Some(rope), Some(vision)) = (self.rope.as_mut(), self.model.vision())
        {
            let coordinates = super::vision::rope_positions(vision, positions, &blocks)?;
            encode_i32(&mut self.encoded, &coordinates);
            rope.write(&self.encoded)?;
            if !self.capturing {
                rope.push(&self.stream)?;
            }
        }
        *self.blocks.borrow_mut() = blocks;
        Ok(())
    }

    /// The `[axes, tokens]` rotary coordinates of a batch of `tokens` tokens with image rows, or
    /// of the recorded walk of one; `None` for a text batch.
    fn rope_view(&self, tokens: u32) -> Option<Operand> {
        match (&self.rope, self.model.vision()) {
            (Some(rope), Some(vision))
                if !self.staging.vision().is_empty() || self.image_pose.get() =>
            {
                Some(rope.shaped(&[u64::from(vision.position_axes()), u64::from(tokens)]))
            }
            _ => None,
        }
    }

    /// Writes the staged prefill's image blocks into their rows of `hidden`, on this stream.
    fn write_image_rows(&self, hidden: Operand) -> Result<()> {
        let blocks = self.blocks.borrow();
        if blocks.is_empty() {
            return Ok(());
        }
        let row_bytes = hidden.shape.last() * self.model.dtype().bytes().unwrap_or(0);
        for block in blocks.iter() {
            let destination = hidden.ptr.address() + u64::from(block.token) * row_bytes;
            let bytes = block.features.rows() * row_bytes;
            // SAFETY: the feature slot holds its product until the engine frees it, which it
            // does only after this batch completed; the hidden rows are this forward's own.
            unsafe {
                platform::memcpy_async(
                    destination,
                    block.features.ptr.address(),
                    bytes as usize,
                    &self.stream,
                )
            }?;
        }
        Ok(())
    }

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

    /// A model that takes images serves their vision encodings, which no captured graph holds: an
    /// encoding is the tower's eager forward over the image's own grid.
    fn media_plan(&self) -> Vec<crate::plan::Bucket> {
        match self.images {
            Some(_) => vec![crate::plan::Bucket::host(
                ipc::execution::Variant::ConditionEncode,
            )],
            None => Vec::new(),
        }
    }

    fn stream(&self) -> Option<&platform::Stream> {
        Some(&self.stream)
    }

    /// A replay is launched behind the one before it, because the driver takes the device idle for
    /// as long as a launch lasts when it waits for that replay to end. An eager replay launches no
    /// graph, so it waits. A union's ranks enter its steps in order on each device thread, so a
    /// step entered behind one in flight is the next step on every rank.
    fn overlaps_replays(&self) -> bool {
        !eager_replay() && self.speculation.is_none()
    }

    /// A batch that keeps its logits answers its greedy rows on the host, and the next step reads
    /// what the host chose.
    fn overlaps_staged(&self) -> bool {
        self.overlaps_replays()
            && self.staging.vision().is_empty()
            && self
                .staging
                .staged()
                .is_none_or(|bucket| bucket.variant != ipc::execution::Variant::PrefillOutput)
    }

    fn landed(&mut self, output: BatchOutput) -> Result<BatchOutput> {
        match self.flights.pop_front() {
            Some(flight) => self.finish(flight),
            None => Ok(output),
        }
    }

    fn submit(&mut self, batch: Batch) -> Result<Jobs> {
        // The products whose last reader completed are released first: their slots' last reads
        // are on this stream, ahead of whatever this batch writes there.
        if let Some(images) = self.images.as_mut() {
            images.free(&batch.frees);
        }
        // A batch of frees alone serves no call.
        if batch.calls.is_empty() && !batch.frees.is_empty() {
            return Ok(Jobs::default());
        }
        if batch
            .media
            .as_ref()
            .is_some_and(|media| media.computation.call == ipc::MediaCall::VisionEncoding)
        {
            return self.submit_images(&batch);
        }
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
        batch
            .columns
            .check(&shape)
            .map_err(|error| refused(error.into()))?;
        // A speculative step is eager and serves the batch as it is: its bucket is the batch's own
        // shape, which no graph is captured for.
        let bucket = if self.speculation.is_some() {
            Bucket {
                variant: shape.variant,
                rows: shape.rows,
                tokens: shape.tokens,
                extent: ipc::execution::Extent::None,
                serves: None,
                capacity: 0,
            }
        } else {
            self.bucket_for(&shape).map_err(refused)?
        };
        self.staging
            .stage(&batch.columns, &batch.calls, &shape, bucket)
            .map_err(refused)?;
        // A greedy bucket takes no drawn row, and the check is about this batch's own rows: a warm
        // pose has no request rows, so it is made here rather than in the capture.
        refuse_drawn_rows_in_greedy_bucket(
            bucket.variant,
            self.staging.parameters(shape.rows as usize),
        )
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
        // A shadow batch is a real batch: the same rule applies to its rows.
        refuse_drawn_rows_in_greedy_bucket(
            bucket.variant,
            self.staging.parameters(shape.rows as usize),
        )?;
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
        self.shadow_jobs_left += jobs.len();
        Ok(Jobs::new(jobs))
    }

    fn capture_plan(&self) -> Vec<Bucket> {
        self.buckets.capture_order().copied().collect()
    }

    fn lifecycle_mut(&mut self) -> Option<&mut Lifecycle> {
        Some(&mut self.lifecycle)
    }

    fn precopy(&mut self, next: &Arc<Generation>) -> Result<Jobs> {
        // One update's evidence: a new precopy starts its record empty, so an operator this update
        // compares is never reported from the one before it.
        self.clear_uncompared();
        self.lifecycle.precopy(next)
    }

    fn attach_pools(&mut self, pools: Arc<std::sync::Mutex<Pools>>) {
        self.pools = Some(Arc::clone(&pools));
        self.lifecycle.attach_pools(Arc::clone(&pools));
        if let Err(error) = self.bind_state(&pools) {
            tracing::error!(%error, "the recurrent state's banks were not bound; a forward will fail");
        }
        if let Some(images) = self.images.as_mut() {
            let mut locked = pools
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Err(error) = images.bind(&mut locked) {
                tracing::error!(%error, "the feature slots were not carved; an image will fail");
            }
        }
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
        self.record_speculation(&mut footprint)?;
        // A model with a vision tower walks every prefill bucket again as a batch with image
        // rows, which turns by the multi-axis rotary, and encodes the largest image the processor
        // sizes, whose tower activations the stream's region must hold.
        let image_input = self.images.as_ref().map(|images| {
            (
                images.format(),
                images.slot_rows(),
                images.largest_patches(),
            )
        });
        if let Some((format, slot_rows, largest)) = image_input {
            let prefills: Vec<Bucket> = self
                .capture_plan()
                .into_iter()
                .filter(|bucket| !bucket.variant.is_decode() && !self.joins_only(*bucket))
                .collect();
            self.image_pose.set(true);
            let walked = prefills.iter().try_for_each(|bucket| {
                footprint.walk(Some(*bucket), |walk| self.walk_recorded(walk, *bucket))
            });
            self.image_pose.set(false);
            walked?;
            let vision = self
                .model
                .vision()
                .ok_or_else(|| Error::Invariant("image input without a vision tower".into()))?;
            let side = f64::from(largest).sqrt() as u32;
            let grid = largest_grid(slot_rows, format.merge, side);
            let patches: u64 = grid.iter().map(|extent| u64::from(*extent)).product();
            let group = self.tensor.clone();
            footprint.walk(None, |walk| {
                let pixels = walk.placeholder(
                    format.dtype,
                    ops::Shape::new(&[patches, u64::from(format.width())]),
                )?;
                let rows = vision.image_rows(grid).map_err(Error::from)?;
                let out = walk.placeholder(
                    self.model.dtype(),
                    ops::Shape::new(&[u64::from(rows), u64::from(self.model.hidden())]),
                )?;
                let forward = walk.forward(group.as_deref())?;
                vision
                    .encode_image(&forward, grid, pixels, out)
                    .map_err(Error::from)
            })?;
        }
        // A pool that admits converted loads launches its conversion on the copy stream as a
        // batch's staged pages load, so the walk states one page of every group.
        if let Some(cache) = self.kv.cache() {
            let (plan, units) = {
                let cache = cache
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                (Arc::clone(cache.plan()), cache.num_units())
            };
            footprint.walk(None, |walk| {
                let forward = walk.forward(None)?;
                crate::memory::prefix_cache::conversion::record(
                    &plan,
                    units,
                    &forward,
                    &|dtype, shape| walk.placeholder(dtype, shape),
                )
            })?;
        }
        // A recurrent state's copies are launched eagerly around a replay, so the walk states them
        // beside the forwards: one copy of each direction states every descriptor a batch's copies
        // launch.
        let stream = self.stream.clone();
        if let Some(state) = self.state.as_mut() {
            let copies = state.recorded_copies();
            footprint.walk(None, |walk| {
                let forward = walk.forward(None)?;
                for direction in [
                    ipc::execution::StateDirection::Restore,
                    ipc::execution::StateDirection::Keep,
                ] {
                    state.copy(&forward, &copies, direction, &stream)?;
                }
                Ok(())
            })?;
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
                    let views = self
                        .experts
                        .as_ref()
                        .and_then(super::experts::Member::views)
                        .map(|views| views.at_rows(join.capacity));
                    self.lifecycle.replay(captured, |arena, staging| {
                        let mut forward = Forward::staging(arena, staging);
                        if let Some(views) = views.as_ref() {
                            forward = forward.with_dispatch(views);
                        }
                        self.model.join_expert_step(&forward).map_err(Error::from)
                    })?;
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
                if self.speculation.is_some() {
                    return self.speculate(*generation, *batch);
                }
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
                // The slots this batch starts requests on, and the bias rows it installs, before
                // the replay that reads them. Only a served step starts a slot: a paired forward
                // of the same staging keeps the slots' state as the replay left it.
                self.start_slots(&stream)?;
                // A row that resumes from a cached prefix continues from the checkpoint its prefix
                // ends at: the page's state is in the slot's live banks before the graph reads them.
                self.copy_state(*generation, ipc::execution::StateDirection::Restore)?;
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
                // A batch with image rows walks eagerly: the feature rows it writes into the
                // hidden rows are each request's own, which no captured graph holds.
                let images = !self.staging.vision().is_empty();
                let eager = eager_replay() || images;
                if images {
                    let kernels = self.lifecycle.generation_named(*generation)?;
                    self.walk(*bucket, &kernels)?;
                } else if eager {
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
                    // The staging walk runs the step the graph was captured from over this batch's
                    // columns, so the values its captured copies read are this batch's.
                    let views = self
                        .experts
                        .as_ref()
                        .and_then(super::experts::Member::views)
                        .map(|views| views.at_rows(bucket.capacity));
                    let tensor = self.tensor.clone();
                    self.lifecycle.replay(captured, |arena, staging| {
                        let mut forward = super::denoiser::grouped(
                            Forward::staging(arena, staging)
                                .with_live_tokens(self.columns.live_tokens.device),
                            tensor.as_deref(),
                        )?;
                        if let Some(views) = views.as_ref() {
                            forward = forward.with_dispatch(views);
                        }
                        self.step(&forward, *bucket)
                    })?;
                }
                // The rows' checkpoints move into the state pages the mechanism allocated for them,
                // behind the forward that wrote them.
                self.copy_state(*generation, ipc::execution::StateDirection::Keep)?;
                // The output read behind this says the batch's offloads finished: the mechanism
                // commits their pages from it, and the forward above did not wait for them.
                if let Some(fence) = self.offload_fence.take() {
                    stream.wait_event(&fence)?;
                }
                if let Some(counters) = &self.counters {
                    counters.launches(nodes);
                }
                // A prefill bucket's captured launches are bounded by the bucket's token count, not
                // the batch's: the rows past the live span are the padding `Staging::stage` wrote, a
                // slot of `-1` that the cache write skips, so they write no page.
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
            // A vision encoding: the image the host lane read, encoded on this stream.
            Job::Host { call, .. } if self.images.is_some() => self.encode_image(*call),
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

    /// Carves the request slots' recurrent-state banks from `pools`, for a model that keeps a
    /// recurrent state; a model that keeps none binds nothing.
    fn bind_state(&mut self, pools: &Arc<std::sync::Mutex<Pools>>) -> Result<()> {
        let Some(shape) = self.model.state() else {
            return Ok(());
        };
        if self.state.is_some() {
            return Ok(());
        }
        let cache = self.kv.cache().cloned().ok_or_else(|| {
            Error::Invalid("a model that keeps a recurrent state reads a prefix cache".into())
        })?;
        let plan = Arc::clone(
            cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .plan(),
        );
        let rows = largest_rows(&self.buckets)?;
        let mut locked = pools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.state = Some(super::state::StateBanks::new(
            shape,
            self.slots,
            rows,
            // A verification's recurrent state keeps a trail of its tokens.
            self.speculation
                .as_ref()
                .map_or(0, speculation::Speculation::trail_tokens),
            &plan,
            &cache,
            self.device,
            &mut locked,
        )?);
        Ok(())
    }

    /// Launches the staged batch's recurrent-state copies of `direction` on the stream with
    /// `generation`'s kernels: the restores before its forward, the checkpoints after it.
    fn copy_state(
        &mut self,
        generation: u64,
        direction: ipc::execution::StateDirection,
    ) -> Result<()> {
        if self.state.is_none() || self.staging.state_copies().is_empty() {
            return Ok(());
        }
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        let arena = nn::Arena::new(ops::DevicePtr::new(0), 0);
        let forward = Forward::new(&stream, &kernels, &arena);
        let copies = self.staging.state_copies().to_vec();
        let state = self
            .state
            .as_mut()
            .ok_or(Error::State("the recurrent state's banks are bound"))?;
        state.copy(&forward, &copies, direction, &stream)
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
        if let Some(state) = self.state.as_mut() {
            state.checkpoints.place(device, &mut pools, &stream)?;
        }
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
        let result = self.shadow_step(operator, key, bucket, generation);
        // A failure ends the shadow batch: the device drops its remaining paired forwards, so none
        // is still owed and no comparison is left to read.
        if result.is_err() {
            self.shadow_jobs_left = 0;
            self.pending_compare = None;
        }
        result
    }

    /// One paired forward of a shadow batch, and the comparison of the one before it.
    fn shadow_step(
        &mut self,
        operator: &'static str,
        key: &str,
        bucket: Bucket,
        generation: u64,
    ) -> Result<()> {
        self.shadow_jobs_left = self.shadow_jobs_left.saturating_sub(1);
        let issued = self.issue_shadow(operator, key, bucket, generation);
        // The previous operator's copies were queued before this forward, so they have landed or
        // are about to: comparing them now waits for no work this forward added. This operator's
        // pairs are compared at the next paired forward, or now when none follows.
        let previous = match issued {
            Ok(Some(pending)) => self.pending_compare.replace(pending),
            Ok(None) => self.pending_compare.take(),
            Err(error) => {
                if let Some(previous) = self.pending_compare.take() {
                    self.finish_compare(previous)?;
                }
                return Err(error);
            }
        };
        if let Some(previous) = previous {
            self.finish_compare(previous)?;
        }
        if self.shadow_jobs_left == 0
            && let Some(last) = self.pending_compare.take()
        {
            self.finish_compare(last)?;
        }
        Ok(())
    }

    /// Launches the paired forward of one changed operator and issues the copies of its pairs,
    /// answering the comparison still to be read, or `None` when the bucket's batch could not be
    /// paired (which the update's evidence then names).
    fn issue_shadow(
        &mut self,
        operator: &'static str,
        key: &str,
        bucket: Bucket,
        generation: u64,
    ) -> Result<Option<PendingCompare>> {
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
            return Ok(None);
        }
        // The operator's tolerance is the one its paired forward recorded: `Forward::operate`
        // calls `O::tolerance(descriptor)` when it shadows, and the harness compares with it.
        let issued = walked.and_then(|_| {
            let tolerance = harness.tolerance().ok_or_else(|| {
                Error::Invariant(format!(
                    "the paired forward of {operator} recorded no tolerance: was it shadowed?"
                ))
            })?;
            harness.issue(&stream, tolerance)
        });
        // The harness stays with the runner whatever the outcome: its failure is the precopy's,
        // which the calling device command reports.
        harness.reset();
        self.shadow = Some(harness);
        Ok(Some(PendingCompare {
            issued: issued?,
            operator,
            key: key.to_owned(),
            bucket,
            generation,
        }))
    }

    /// Compares one paired forward's pairs, and names in the update's evidence what they could not
    /// measure.
    fn finish_compare(&mut self, pending: PendingCompare) -> Result<()> {
        let PendingCompare {
            issued,
            operator,
            key,
            bucket,
            generation,
        } = pending;
        let harness = self.shadow.as_mut().ok_or(Error::State(
            "the runner has no shadow harness: it is allocated when the runner is bound",
        ))?;
        let summary = harness.finish(issued)?;
        // A pair the two programs could not be compared over, and a forward that carved no twin at
        // all, are both statements of what this update has no evidence for: the operator is named
        // in the update's answer rather than left to the log (7.3 step 2).
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
        Ok(())
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
        // The paired forward runs eagerly, so its uploads are copied from a staging slot.
        let staging = self.lifecycle.staging.slot()?;
        let mut forward = Forward::new(&stream, next, &arena)
            .with_staging(staging)
            .with_route(&pairing)
            .with_live_tokens(self.columns.live_tokens.device);
        if let Some(group) = tensor.as_deref() {
            forward = forward.with_group(group);
        }
        let slots: Vec<i32> = self.staging.slots(rows).to_vec();
        let ids = self.input_ids(bucket);
        self.with_kv(&forward, &slots, Some((pages, &lengths)), false, |cache| {
            let embedded = self
                .model
                .embed_input_ids(&forward, ids)
                .map_err(Error::from)?;
            self.model
                .forward(&forward, &batch, cache, embedded)
                .map_err(Error::from)
        })?;
        self.lifecycle.staging.release(&stream)?;
        Ok(None)
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
            targets: Vec::new(),
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
            platform::memcpy_async(
                self.columns.logprob_readback[turn].address(),
                samples.scores.logprobs.ptr.address(),
                bytes,
                &self.stream,
            )?;
            platform::memcpy_async(
                self.columns.rank_readback[turn].address(),
                samples.scores.ranks.ptr.address(),
                bytes,
                &self.stream,
            )?;
            // The candidates, values then ids, each `[rows, candidates]` and contiguous.
            if let (Some(top), Some(readback)) =
                (samples.scores.top, self.columns.top_readback.as_ref())
            {
                let width = top.values.shape.last() as usize;
                let plane = rows * width * std::mem::size_of::<i32>();
                if 2 * plane > readback[turn].len() {
                    return Err(Error::Invariant(
                        "the candidates' readback buffer is smaller than the bucket".into(),
                    ));
                }
                platform::memcpy_async(
                    readback[turn].address(),
                    top.values.ptr.address(),
                    plane,
                    &self.stream,
                )?;
                platform::memcpy_async(
                    readback[turn].address() + plane as u64,
                    top.ids.ptr.address(),
                    plane,
                    &self.stream,
                )?;
            }
            // A scoring bucket's prompt scores: log-probabilities, ranks, then the candidates'
            // values and ids, each over the bucket's tokens and back to back.
            if let Some(prompt) = samples.prompt {
                let tokens = bucket.tokens as usize;
                let width = prompt.candidates() as usize;
                let word = std::mem::size_of::<i32>();
                let readback = &self.columns.prompt_readback[turn];
                if tokens * word * (2 + 2 * width) > readback.len() {
                    return Err(Error::Invariant(
                        "the prompt scores' readback buffer is smaller than the bucket".into(),
                    ));
                }
                let mut offset = 0;
                let mut copy = |source: Operand, bytes: usize| -> Result<()> {
                    platform::memcpy_async(
                        readback.address() + offset as u64,
                        source.ptr.address(),
                        bytes,
                        &self.stream,
                    )?;
                    offset += bytes;
                    Ok(())
                };
                copy(prompt.logprobs, tokens * word)?;
                copy(prompt.ranks, tokens * word)?;
                if let Some(top) = prompt.top {
                    copy(top.values, tokens * width * word)?;
                    copy(top.ids, tokens * width * word)?;
                }
            }
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
        if samples.prompt.is_some() {
            flight.targets = self.staging.targets(bucket.tokens as usize).to_vec();
        }
        flight.rows = self.staging.batch_rows().clone();
        flight.calls = self.staging.calls().to_vec();
        Ok(flight)
    }

    /// The answer of a replay whose copies have landed.
    fn finish(&mut self, flight: Flight) -> Result<BatchOutput> {
        if flight.sampled.is_none() {
            return Ok(BatchOutput::default());
        };
        let Flight {
            bucket,
            launched,
            turn,
            sampled,
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
        let width = sampled
            .and_then(|sampled| sampled.scores.top)
            .map_or(0, |top| top.values.shape.last() as usize);
        // SAFETY: the copies above filled `rows` f32 with the sampled ids' log-probabilities and
        // `rows` i32 with their ranks, and, for a sampler that reports candidates, `rows · width`
        // f32 then `rows · width` i32 with each row's most likely tokens; for a scoring bucket, the
        // prompt readback with its tokens' scores in the same order.
        let samples = unsafe {
            let (values, candidates) = match (&self.columns.top_readback, width) {
                (Some(readback), width) if width > 0 => {
                    let base = readback[turn].as_ptr();
                    (
                        std::slice::from_raw_parts(base.cast::<f32>(), rows * width),
                        std::slice::from_raw_parts(
                            base.add(rows * width * std::mem::size_of::<f32>())
                                .cast::<i32>(),
                            rows * width,
                        ),
                    )
                }
                _ => (&[][..], &[][..]),
            };
            let prompt = sampled.and_then(|sampled| sampled.prompt).map(|prompt| {
                let tokens = flight.targets.len();
                let width = prompt.candidates() as usize;
                let base = self.columns.prompt_readback[turn].as_ptr();
                let word = std::mem::size_of::<i32>();
                PromptSamples {
                    targets: &flight.targets,
                    logprobs: std::slice::from_raw_parts(base.cast::<f32>(), tokens),
                    ranks: std::slice::from_raw_parts(
                        base.add(tokens * word).cast::<i32>(),
                        tokens,
                    ),
                    values: std::slice::from_raw_parts(
                        base.add(2 * tokens * word).cast::<f32>(),
                        tokens * width,
                    ),
                    candidates: std::slice::from_raw_parts(
                        base.add((2 + width) * tokens * word).cast::<i32>(),
                        tokens * width,
                    ),
                    width,
                }
            });
            RowSamples {
                ids,
                prompt,
                logprobs: std::slice::from_raw_parts(
                    self.columns.logprob_readback[turn].as_ptr().cast::<f32>(),
                    rows,
                ),
                ranks: std::slice::from_raw_parts(
                    self.columns.rank_readback[turn].as_ptr().cast::<i32>(),
                    rows,
                ),
                values,
                candidates,
                width,
                vocabulary: self.model.vocabulary(),
            }
        };
        let calls = calls_output(
            &samples,
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
        let kernels = Arc::clone(&self.lifecycle.generation);
        self.offload_fence =
            self.kv
                .issue_pages(&install.loads, &install.offloads, &self.stream, &kernels)?;
        Ok(())
    }

    /// Starts the staged batch's requests' sampling state on their slots, on `stream`, before the
    /// step that samples them: a fresh slot's penalty counts start at its request's own history -
    /// the slot's previous request committed its own tokens there, and the request may have
    /// generated tokens on another group or before it was admitted again - and the batch's bias
    /// rows are installed. Both are stream-ordered device writes, so nothing waits for them.
    ///
    /// A fresh slot of a decode batch starts a request whose prompt another worker computed and
    /// whose first token it sampled - a decode group continuing a prefill group's request - so its
    /// commit state starts as that worker's explicit commit left it ([`super::sampling::SlotStart`]):
    /// the decode reads its input token from the slot's state, not from the host.
    fn start_slots(&mut self, stream: &platform::Stream) -> Result<()> {
        let fresh = self.staging.fresh_slots();
        let installs = self.staging.adjustments().installs();
        if fresh.is_empty() && installs.is_empty() {
            return Ok(());
        }
        let starts = self.decode_starts()?;
        for start in starts {
            self.commit.start(start, stream)?;
        }
        let fresh = self.staging.fresh_slots();
        let sampling = self.sampling.as_mut().ok_or(Error::State(
            "a batch starts sampling state in a runner that samples nothing",
        ))?;
        sampling.start_counts(fresh, stream)?;
        sampling.install(installs, stream)
    }

    /// The commit state each fresh slot of a staged decode batch starts with: its row's input
    /// token at the row's position, the row's draw, and the cache before the token as verified.
    /// None for any other batch, whose first call on a slot is a prompt whose explicit commit
    /// states the slot's state.
    fn decode_starts(&self) -> Result<Vec<super::sampling::SlotStart>> {
        let Some(bucket) = self.staging.staged() else {
            return Ok(Vec::new());
        };
        let fresh = self.staging.fresh_slots();
        if !bucket.variant.is_decode() || fresh.is_empty() {
            return Ok(Vec::new());
        }
        let rows = bucket.rows as usize;
        let batch = self.staging.batch_rows();
        let slots = self.staging.slots(rows);
        let input_ids = self.staging.input_ids(bucket.tokens as usize);
        let draws = self.staging.draws(rows);
        let mut starts = Vec::with_capacity(fresh.len());
        for fresh in fresh {
            let slot = fresh.slot;
            let row = slots
                .iter()
                .position(|held| u32::try_from(*held).ok() == Some(slot))
                .ok_or_else(|| {
                    Error::Invalid(format!("fresh slot {slot} has no row in its decode batch"))
                })?;
            // A decode row is one token: its query offset is the token's index.
            let token = usize::try_from(batch.query_offsets[row])
                .map_err(|_| Error::Invalid("a negative query offset".into()))?;
            let sampling = i32::try_from(draws[row].position)
                .map_err(|_| Error::Invalid("a draw position past i32".into()))?;
            starts.push(super::sampling::SlotStart {
                slot,
                token: input_ids[token],
                logical: batch.positions[token],
                sampling,
                verified: batch.seq_lens[row] - 1,
            });
        }
        Ok(starts)
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
        let positions = self.staging.batch_rows().positions[..tokens].to_vec();
        self.stage_image_rows(&positions)?;
        let batch = self.staging.batch_rows();

        encode_i32(&mut self.encoded, &batch.positions[..tokens]);
        self.columns.positions.write(&self.encoded)?;
        // The ids travel with the batch: `nn::CausalLm::forward` runs the stack over hidden states
        // the model already holds, so embedding them is the runner's step, not the model's. A
        // decode row's token is the slot's next input, which the graph reads for itself: the step
        // before it may still be running, and the mechanism plans ahead of its results anyway.
        encode_i32(&mut self.encoded, self.staging.input_ids(tokens));
        self.columns.input_ids.write(&self.encoded)?;
        encode_i32(&mut self.encoded, self.staging.targets(tokens));
        self.columns.targets.write(&self.encoded)?;
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

        // The rows' logit adjustments, and the ban masks only as many as the batch's rows that ban
        // a token: a batch that bans nothing copies no mask.
        encode_i32(&mut self.encoded, self.staging.adjustments().rows(rows));
        self.columns.adjustments.write(&self.encoded)?;
        encode_i32(&mut self.encoded, self.staging.adjustments().bans());
        self.columns.bans.write(&self.encoded)?;

        // The batch's live token count, which no shape of the bucket states: a captured graph reads
        // the bucket's token count as a constant, so the MoE sort - which must write a sentinel into
        // every route row of a padded token - reads this column instead.
        encode_i32(&mut self.encoded, &[self.staging.live_tokens]);
        self.columns.live_tokens.write(&self.encoded)?;

        self.commit
            .stage(self.staging.commit(), rows, self.commit.capacity as usize)?;
        if let Some(state) = self.state.as_mut() {
            state.stage(self.staging.checkpoints(rows))?;
        }

        // A copy issued inside a recording would be part of the graph, and replay the warm pose.
        if self.capturing {
            return Ok(());
        }
        let stream = self.stream.clone();
        for column in self.columns.staged() {
            column.push(&stream)?;
        }
        if let Some(state) = self.state.as_mut() {
            state.checkpoints.push(&stream)?;
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
    fn input_ids(&self, bucket: Bucket) -> Operand {
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
    fn token_batch_with(&self, bucket: Bucket, slot_mapping: Operand) -> Result<TokenBatch> {
        Ok(TokenBatch {
            positions: self.columns.positions.extent(0..bucket.tokens as u64),
            slot_mapping,
            seq_lens: self.columns.seq_lens.extent(0..bucket.rows as u64),
            query_offsets: self.columns.query_offsets.extent(0..bucket.rows as u64 + 1),
            logits_rows: self.columns.logits_rows.extent(0..bucket.rows as u64),
            phase: match bucket.variant.is_decode() {
                true => proto::program::PhaseKind::Decode,
                false => proto::program::PhaseKind::Prefill,
            },
            page_size: self.kv.page_size(),
            num_tokens: bucket.tokens,
            max_kv_len: self.max_kv_len,
            draft_lens: None,
            // A batch with image rows, or the walk of one, rotates by its multi-axis coordinates.
            rope_positions: self.rope_view(bucket.tokens),
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
        forward: &Forward<'_>,
        slots: &[i32],
        scratch_pages: Option<(u32, &[i32])>,
        quiet: bool,
        operation: impl FnOnce(&KvView<'_>) -> Result<R>,
    ) -> Result<R> {
        let workspace = forward.empty(DType::U8, ops::Shape::new(&[ATTENTION_WORKSPACE_BYTES]))?;
        self.with_kv_over(workspace, slots, scratch_pages, quiet, None, operation)
    }

    /// [`Self::with_kv`] over the reserved `workspace`, with the batch's state rows `state` (and
    /// whether they keep a trail) where they are not its attention rows: a verification's.
    #[allow(
        clippy::too_many_arguments,
        reason = "the view's workspace, its rows' slots, the paired forward's pages, the table \
                  gather's mode, the state rows and the operation"
    )]
    fn with_kv_over<R>(
        &self,
        workspace: Operand,
        slots: &[i32],
        scratch_pages: Option<(u32, &[i32])>,
        quiet: bool,
        state_rows: Option<(super::state::StateRows, bool)>,
        operation: impl FnOnce(&KvView<'_>) -> Result<R>,
    ) -> Result<R> {
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
        match &self.state {
            None => operation(&KvView {
                groups: &view,
                workspace,
                state: None,
            }),
            // A paired forward writes scratch pages and must not advance a served slot's state,
            // and no scratch bank exists for it to advance instead.
            Some(_) if scratch_pages.is_some() => Err(Error::Unsupported(
                "a paired forward of a model that keeps a recurrent state has no scratch banks to \
                 advance"
                    .into(),
            )),
            Some(state) => {
                // A batch's state rows are its attention rows unless the caller states others.
                let rows = slots.len() as u64;
                let (rows, trailing) = state_rows.unwrap_or((
                    super::state::StateRows {
                        query_offsets: self.columns.query_offsets.extent(0..rows + 1),
                        seq_lens: self.columns.seq_lens.extent(0..rows),
                        slots: self.columns.slots.extent(0..rows),
                        count: rows,
                    },
                    false,
                ));
                state.with_view(rows, trailing, |state| {
                    operation(&KvView {
                        groups: &view,
                        workspace,
                        state: Some(state),
                    })
                })
            }
        }
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
        let staging = self.lifecycle.staging.captured(&stream)?;
        let mut forward = Forward::new(&stream, generation, &arena)
            .with_staging(staging)
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
        let slots: Vec<i32> = self.staging.slots(bucket.rows as usize).to_vec();
        let ids = self.input_ids(bucket);
        self.gather_step(forward, bucket)?;
        // A recording or a capture issues no table copy of its own: the tables are read where they
        // stand, which is what a replay addresses.
        let quiet = self.capturing || !forward.launches();
        self.with_kv(forward, &slots, None, quiet, |cache| {
            let embedded = self
                .model
                .embed_input_ids(forward, ids)
                .map_err(Error::from)?;
            // An image block's hidden rows are its features, written over the placeholder rows'
            // embeddings on this stream; a recording launches nothing.
            if forward.launches() {
                self.write_image_rows(embedded)?;
            }
            let hidden = self
                .model
                .forward(forward, &batch, cache, embedded)
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
        bucket.variant == ipc::execution::Variant::ExpertJoin
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
        let staging = self.lifecycle.staging.captured(&stream)?;
        let forward = Forward::new(&stream, generation, &arena)
            .with_staging(staging)
            .with_dispatch(&views);
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
    fn sample(&self, forward: &Forward<'_>, bucket: Bucket, hidden: Operand) -> Result<()> {
        let Some(sampling) = &self.sampling else {
            return Err(Error::State(
                "the runner has no sampler but the bucket samples",
            ));
        };
        let rows = u64::from(bucket.rows);
        // The batch is checked where a real one is staged (`submit`, `shadow`): a capture has no
        // batch contents - its warm pose states a zeroed parameter column, which reads as a greedy
        // row - so the check here would refuse a sampled bucket its own capture.
        // `IndexRows::gather` takes a flat `[count]` index view and answers `count` rows, so the
        // indices are this bucket's rows of the column rather than the column's own extent.
        let gathered = nn::IndexRows
            .gather(forward, hidden, self.columns.indices.shaped(&[rows]))
            .map_err(Error::from)?;
        let logits = self
            .model
            .compute_logits(forward, gathered)
            .map_err(Error::from)?;
        // The sampler reads its request state per row: `parameters` is `[rows, PARAMETER_COLUMNS]`
        // and `slots` is `[rows]`. A bucket narrower than the largest reads its own rows from the
        // column's head - a captured graph addresses the column's base pointer, so the view states
        // the bucket's shape and not the column's flat extent, which is the largest bucket's.
        let parameters = self
            .columns
            .parameters
            .shaped(&[rows, u64::from(proto::program::sampler::PARAMETER_COLUMNS)]);
        let slots = self.columns.slots.shaped(&[rows]);
        let words = u64::from(ops::sampler::ban_words(self.model.vocabulary()));
        let state = SamplerState {
            counts: sampling.counts,
            parameters,
            slots,
            extras: None,
            bias: sampling.bias,
            adjustments: self.columns.adjustments.shaped(&[rows, 2]),
            bans: self.columns.bans.shaped(&[rows, words]),
        };
        // Every sampling bucket reports its tokens' log-probabilities, ranks and most likely
        // alternatives: whether a row's request asked for them is request data a captured graph
        // cannot branch on, and the scores reuse the sampler's dead shaped row, so the extra memory
        // is a few words a row. The scheduler keeps them for the requests that asked, as many
        // alternatives as each asked for.
        let samples = Samples::reserve(
            forward,
            DType::I32,
            rows,
            self.model.vocabulary(),
            Some(sampling.sampler.candidates()),
        )?;
        // A greedy bucket takes each row's largest candidate; every other sampling bucket draws one
        // from the row's distribution. The two are different launch sequences in the sampler layer
        // (`Sampler::greedy` and `Sampler::forward`). A greedy row in a drawing bucket draws from the
        // one candidate its staged `top_k` keeps, which is its argmax (`Staging::stage`).
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
        let prompt = self.score_prompt(forward, bucket, hidden)?;
        // A recording's views address no memory, so only a launching walk names where a replay's
        // tokens land.
        if forward.launches() {
            self.bucket_samples.borrow_mut().insert(
                bucket,
                Sampled {
                    tokens: samples.tokens,
                    scores: samples
                        .scores
                        .ok_or(Error::State("a sampling bucket reserved no scores"))?,
                    prompt,
                },
            );
        }
        self.commit_step(forward, bucket, &samples)?;
        Ok(())
    }

    /// Scores the prompt tokens a scoring bucket's targets name (`prompt_logprobs`), and answers
    /// where the scores land; a bucket that scores nothing launches nothing.
    ///
    /// Every token of the bucket is projected: its hidden row's logits, their `log_softmax` (the
    /// raw distribution), then its target's log-probability, rank and the row's candidates, as a
    /// sampled token's are scored (`Sampler::score`). The projection walks the bucket in
    /// [`SCORE_ROWS`]-row chunks through one pair of `[SCORE_ROWS, vocabulary]` f32 buffers, so the
    /// working set does not grow with the bucket; a chunk is a fixed sequence of launches, and the
    /// graph holds one per chunk. A token whose target is `-1` - padding, a row of a request that
    /// asks for nothing, a prompt's last token - is projected all the same and reports rank `0`,
    /// which the readback skips.
    ///
    /// The cost is the scoring's: one head projection, one log-softmax and one `top_k` per token,
    /// which a decode or a prefill that scores nothing never pays (`Variant::PrefillScored`).
    fn score_prompt(
        &self,
        forward: &Forward<'_>,
        bucket: Bucket,
        hidden: Operand,
    ) -> Result<Option<Scores>> {
        if !bucket.variant.scores() {
            return Ok(None);
        }
        let sampler = &self
            .sampling
            .as_ref()
            .ok_or(Error::State("a scoring bucket has no sampler"))?
            .sampler;
        let tokens = u64::from(bucket.tokens);
        let vocabulary = u64::from(self.model.vocabulary());
        let scores = Scores::reserve(
            forward,
            tokens,
            self.model.vocabulary(),
            sampler.candidates(),
        )?;
        let chunk = SCORE_ROWS.min(tokens);
        let targets = self.columns.targets.shaped(&[tokens]);
        let mut start = 0;
        while start < tokens {
            let count = chunk.min(tokens - start);
            let rows = hidden.narrow(0, start, count).map_err(Error::from)?;
            // A chunk's logits and log-probabilities are dead once its scores are written, so the
            // next chunk is handed the same addresses.
            forward.scoped(|forward| -> Result<()> {
                let logits = self
                    .model
                    .compute_logits(forward, rows)
                    .map_err(Error::from)?;
                let logprobs = forward.empty(DType::F32, ops::Shape::new(&[count, vocabulary]))?;
                sampler.log_softmax(forward, logits, logprobs)?;
                sampler.score(
                    forward,
                    logprobs,
                    targets.narrow(0, start, count).map_err(Error::from)?,
                    &scores.rows(start, count)?,
                )
            })?;
            start += count;
        }
        Ok(Some(scores))
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
            future: Operand::with_strides(
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
            targets: Vec::new(),
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
            // A warm pose keeps no checkpoint and resumes from none: its rows start at position
            // zero over the padding slot, whose banks no request owns.
            checkpoints: Vec::new(),
            state_copies: Vec::new(),
            drafts: Vec::new(),
            vision: Vec::new(),
            // No row adjusts its logits: the shaper's adjustment operands are recorded all the
            // same, so the graph a serving batch with adjustments replays is this one.
            adjustments: ipc::execution::LogitAdjustments::default(),
            fresh_slots: Vec::new(),
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
    samples: &RowSamples<'_>,
    sampling_rows: &[i32],
    rows: &BatchRows,
    calls: &[CallId],
    states: &[RowState],
) -> Result<Vec<CallOutput>> {
    let ids = samples.ids;
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
        if let Some(scored) = samples.scored(index) {
            entry.token_logprobs.push(scored);
        }
    }
    // A scoring bucket's prompt scores belong to the call whose rows hold the scored tokens, in
    // token order; a token whose target is `-1` scored nothing.
    if let Some(prompt) = &samples.prompt {
        for (call, entry) in rows.calls.iter().zip(output.iter_mut()) {
            let offset = |row: u32| {
                rows.query_offsets
                    .get(row as usize)
                    .map_or(0, |offset| (*offset).max(0) as usize)
            };
            for token in offset(call.rows.start)..offset(call.rows.end) {
                if let Some(scored) = prompt.scored(token, samples.vocabulary) {
                    entry.prompt_logprobs.push(scored);
                }
            }
        }
    }
    Ok(output)
}

/// What a replay's readback holds per bucket row: the sampled ids and their scores, as the
/// completion reads them.
pub struct RowSamples<'a> {
    /// `[rows]` the sampled ids.
    pub ids: &'a [i32],
    /// `[rows]` each sampled id's raw log-probability; empty for a step that reports none.
    pub logprobs: &'a [f32],
    /// `[rows]` each sampled id's rank in its raw distribution.
    pub ranks: &'a [i32],
    /// `[rows, width]` each row's most likely tokens' log-probabilities, row-major.
    pub values: &'a [f32],
    /// `[rows, width]` their ids.
    pub candidates: &'a [i32],
    /// The candidates a row reports.
    pub width: usize,
    /// The vocabulary a candidate id lies in.
    pub vocabulary: u32,
    /// A scoring bucket's prompt scores, one per bucket token.
    pub prompt: Option<PromptSamples<'a>>,
}

/// What a scoring bucket's readback holds per bucket token: the prompt token it scored and its
/// scores.
pub struct PromptSamples<'a> {
    /// `[tokens]` the prompt token each token scored, `-1` for one that scored none.
    pub targets: &'a [i32],
    /// `[tokens]` each target's raw log-probability.
    pub logprobs: &'a [f32],
    /// `[tokens]` each target's rank.
    pub ranks: &'a [i32],
    /// `[tokens, width]` each token's most likely next tokens' log-probabilities, row-major.
    pub values: &'a [f32],
    /// `[tokens, width]` their ids.
    pub candidates: &'a [i32],
    /// The candidates a token reports.
    pub width: usize,
}

impl PromptSamples<'_> {
    /// Token `index`'s target and its scores ([`scored`]), when it scored one.
    fn scored(&self, index: usize, vocabulary: u32) -> Option<Vec<ipc::TokenLogprob>> {
        let target = u32::try_from(*self.targets.get(index)?).ok()?;
        let span = index * self.width..(index + 1) * self.width;
        Some(scored(
            target,
            *self.logprobs.get(index)?,
            self.ranks.get(index).copied().unwrap_or(0).max(0) as u32,
            self.values.get(span.clone()).unwrap_or_default(),
            self.candidates.get(span).unwrap_or_default(),
            vocabulary,
        ))
    }
}

impl RowSamples<'_> {
    /// Row `index`'s sampled token and its scores ([`scored`]), when the step reports scores.
    fn scored(&self, index: usize) -> Option<Vec<ipc::TokenLogprob>> {
        let logprob = *self.logprobs.get(index)?;
        let rank = self.ranks.get(index).copied().unwrap_or(0).max(0) as u32;
        let span = index * self.width..(index + 1) * self.width;
        Some(scored(
            self.ids[index] as u32,
            logprob,
            rank,
            self.values.get(span.clone()).unwrap_or_default(),
            self.candidates.get(span).unwrap_or_default(),
            self.vocabulary,
        ))
    }
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
    let columns = proto::program::sampler::PARAMETER_COLUMNS as usize;
    let mut parameters = vec![0.0; rows * columns];
    for row in 0..rows {
        let row = &mut parameters[row * columns..(row + 1) * columns];
        row[proto::program::sampler::TEMPERATURE as usize] = 1.0;
        row[proto::program::sampler::TOP_P as usize] = 1.0;
    }
    parameters
}

/// Refuses a greedy decode bucket a row that draws.
///
/// A bucket's variant fixes the sampler launch sequence for every row: [`Variant::DecodeGreedy`]
/// argmaxes each row's shaped logits and [`Variant::DecodeSampled`] draws one token per row. A
/// drawing bucket serves a greedy row as well - the row is staged with a top-k width of one, so its
/// draw is its argmax (`Staging::stage`) - which is what lets the batcher put greedy and drawn rows
/// in one batch. The converse is not so: an argmax is not what a row that states a temperature
/// asked for, so a greedy bucket holding one is a batch it cannot serve, refused by the row and the
/// bucket rather than served with a token the request did not ask for.
fn refuse_drawn_rows_in_greedy_bucket(
    variant: ipc::execution::Variant,
    parameters: &[f32],
) -> Result<()> {
    if !variant.greedy() {
        return Ok(());
    }
    let columns = proto::program::sampler::PARAMETER_COLUMNS as usize;
    let temperature = proto::program::sampler::TEMPERATURE as usize;
    let rows = parameters.len() / columns;
    for (row, stated) in parameters.chunks_exact(columns).enumerate() {
        if stated[temperature] == 0.0 {
            continue;
        }
        return Err(Error::Invariant(format!(
            "the {} bucket holds {rows} row(s) and row {row} states temperature {}: its token is \
             drawn from its distribution, and a greedy bucket only takes each row's argmax",
            variant.name(),
            stated[temperature],
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
    use crate::runners::sampling::{commit_block, commit_descriptors, commit_table};

    /// The logit row width a staging is sized for; no case here bans a token.
    const VOCABULARY_UNDER_TEST: u32 = 1_024;

    /// A drawing bucket serves greedy rows beside drawn ones: the staging gives each greedy row a
    /// top-k width of one, so its draw is its largest candidate, and leaves a drawn row's width as
    /// the request stated it. A greedy bucket still refuses a drawn row, by the bucket and the row.
    #[test]
    fn a_drawing_bucket_serves_greedy_rows_as_their_largest_candidate() -> Result<()> {
        let columns = proto::program::sampler::PARAMETER_COLUMNS as usize;
        let temperature = proto::program::sampler::TEMPERATURE as usize;
        let top_k = proto::program::sampler::TOP_K as usize;
        let buckets = Buckets::new([Bucket::decode(4, Variant::DecodeSampled)]);
        let mut staging = Staging::new();
        staging.reserve(&buckets, VOCABULARY_UNDER_TEST)?;
        let mut batch = decode_columns();
        batch.parameters[temperature] = 0.7;
        batch.parameters[top_k] = 40.0;
        batch.parameters[columns + temperature] = 0.0;
        batch.parameters[columns + top_k] = 40.0;
        let shape = Shape::decode(2, Variant::DecodeSampled);
        let bucket = Bucket::decode(4, Variant::DecodeSampled);
        staging.stage(&batch, &[CallId(4), CallId(5)], &shape, bucket)?;

        let staged = staging.parameters(2);
        assert_eq!(staged[top_k], 40.0, "a drawn row keeps its width");
        assert_eq!(
            staged[columns + top_k],
            1.0,
            "a greedy row keeps one candidate"
        );
        refuse_drawn_rows_in_greedy_bucket(bucket.variant, staged)
            .expect("a drawing bucket takes a greedy row");

        let error = refuse_drawn_rows_in_greedy_bucket(Variant::DecodeGreedy, staged)
            .expect_err("a greedy bucket argmaxed a drawn row");
        let text = error.to_string();
        assert!(
            text.contains("decode_greedy") && text.contains("row 0"),
            "the refusal names neither the bucket nor the row: {text}"
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
        let mut table = vec![0_i64; proto::program::commit::COMMIT_FIELDS * rows];
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
            targets: Vec::new(),
            slots: vec![0, 1],
            parameters: vec![0.0; 2 * proto::program::sampler::PARAMETER_COLUMNS as usize],
            draws: vec![ipc::philox::DrawAddress::default(); 2],
            commit: committed(&[0, 1]),
            page_size: 16,
            max_kv_len: 4096,
            checkpoints: Vec::new(),
            state_copies: Vec::new(),
            drafts: Vec::new(),
            vision: Vec::new(),
            adjustments: ipc::execution::LogitAdjustments::default(),
            fresh_slots: Vec::new(),
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
            targets: Vec::new(),
            slots: vec![0],
            parameters: vec![0.0; proto::program::sampler::PARAMETER_COLUMNS as usize],
            draws: vec![ipc::philox::DrawAddress::default()],
            commit: committed(&[0]),
            page_size: 16,
            max_kv_len: 4096,
            checkpoints: Vec::new(),
            state_copies: Vec::new(),
            drafts: Vec::new(),
            vision: Vec::new(),
            adjustments: ipc::execution::LogitAdjustments::default(),
            fresh_slots: Vec::new(),
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
        staging.reserve(&buckets, VOCABULARY_UNDER_TEST)?;

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
        staging.reserve(&buckets, VOCABULARY_UNDER_TEST)?;
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
        staging.reserve(&buckets, VOCABULARY_UNDER_TEST)?;
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
            targets: Vec::new(),
            slots: vec![0],
            parameters: vec![0.0; proto::program::sampler::PARAMETER_COLUMNS as usize],
            draws: vec![ipc::philox::DrawAddress::default()],
            commit: committed(&[0]),
            page_size: 16,
            max_kv_len: 4096,
            checkpoints: Vec::new(),
            state_copies: Vec::new(),
            drafts: Vec::new(),
            vision: Vec::new(),
            adjustments: ipc::execution::LogitAdjustments::default(),
            fresh_slots: Vec::new(),
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
        staging.reserve(&buckets, VOCABULARY_UNDER_TEST)?;
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
            staging.parameters(rows)[2 * proto::program::sampler::PARAMETER_COLUMNS as usize..]
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
        let mut staged = vec![0_i64; proto::program::commit::COMMIT_FIELDS * rows];
        for row in 0..rows {
            staged[proto::program::commit::column::SLOT * rows + row] = row as i64 + 7;
            staged[proto::program::commit::column::MODE * rows + row] =
                proto::program::commit::mode::DECODE;
        }
        let table = commit_table(&staged, rows, capacity)?;
        assert_eq!(
            table.len(),
            proto::program::commit::COMMIT_FIELDS * capacity
        );
        let slots = &table[proto::program::commit::column::SLOT * capacity
            ..(proto::program::commit::column::SLOT + 1) * capacity];
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
        assert_eq!(table.len(), proto::program::commit::COMMIT_FIELDS * rows);
        for field in 0..proto::program::commit::COMMIT_FIELDS {
            let column = &table[field * rows..(field + 1) * rows];
            if field == proto::program::commit::column::SLOT
                || field == proto::program::commit::column::TASK
            {
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
        staging.reserve(&buckets, VOCABULARY_UNDER_TEST)?;

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
        staging
            .reserve(&buckets, VOCABULARY_UNDER_TEST)
            .expect("one bucket");
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

    /// The readback of a step that sampled `ids` and reports no scores.
    fn unscored(ids: &[i32]) -> RowSamples<'_> {
        RowSamples {
            ids,
            logprobs: &[],
            ranks: &[],
            values: &[],
            candidates: &[],
            width: 0,
            vocabulary: 1_000,
            prompt: None,
        }
    }

    /// Each sampled token carries its own row's log-probability, rank and candidates, and a row
    /// that samples nothing reports none.
    #[test]
    fn a_sampled_token_carries_its_rows_scores() {
        let batch = rows(&[(0, 0..1), (1, 1..2)]);
        let samples = RowSamples {
            ids: &[111, 222],
            logprobs: &[-0.5, -2.0],
            ranks: &[2, 9],
            values: &[-0.25, -0.5, -0.75, -1.0],
            candidates: &[7, 111, 3, 4],
            width: 2,
            vocabulary: 1_000,
            prompt: None,
        };
        let output = calls_output(
            &samples,
            &[0, -1],
            &batch,
            &[CallId(7), CallId(9)],
            &[RowState::default(), RowState::default()],
        )
        .expect("the batch's calls");
        assert_eq!(output[0].tokens, vec![111]);
        let entry = |token_id, logprob, rank| ipc::TokenLogprob {
            token_id,
            logprob,
            rank,
        };
        assert_eq!(
            output[0].token_logprobs,
            vec![vec![
                entry(111, -0.5, 2),
                entry(7, -0.25, 1),
                entry(111, -0.5, 2)
            ]]
        );
        assert!(output[1].tokens.is_empty() && output[1].token_logprobs.is_empty());
    }

    /// The step's output names every call of the batch, and each call's tokens are its own rows'
    /// samples in row order, whatever order the mechanism listed the calls in.
    #[test]
    fn sampled_rows_are_named_by_their_calls() {
        let batch = rows(&[(0, 0..1), (1, 1..2)]);
        let output = calls_output(
            &unscored(&[111, 222]),
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
            &unscored(&[0]),
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
        calls_output(
            &unscored(&[5]),
            &[3],
            &batch,
            &[CallId(1)],
            &[RowState::default()],
        )
        .expect_err("row 3 belongs to no call");
        calls_output(
            &unscored(&[]),
            &[0],
            &batch,
            &[CallId(1)],
            &[RowState::default()],
        )
        .expect_err("no id was sampled for row 0");
        calls_output(&unscored(&[5]), &[0], &batch, &[CallId(1)], &[])
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
        let columns = u64::from(proto::program::sampler::PARAMETER_COLUMNS);
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
    /// candidate kept, so the capture walks the whole candidate width a served draw may need.
    #[test]
    fn a_warm_pose_states_a_draw_a_sampled_bucket_can_make() {
        let columns = proto::program::sampler::PARAMETER_COLUMNS as usize;
        let parameters = warm_parameters(2);
        assert_eq!(parameters.len(), 2 * columns, "a row per row");
        for row in 0..2 {
            let row = &parameters[row * columns..(row + 1) * columns];
            assert_eq!(row[proto::program::sampler::TEMPERATURE as usize], 1.0);
            assert_eq!(row[proto::program::sampler::TOP_P as usize], 1.0);
            assert_eq!(row[proto::program::sampler::TOP_K as usize], 0.0);
        }
    }
}
