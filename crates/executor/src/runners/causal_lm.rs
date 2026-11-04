//! `CausalLmRunner`: a causal language model's batches, planned into captured buckets.
//!
//! The runner is the executor's half of §4.3's `CausalLm`: it owns the model, the generation of
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

use ipc::PhaseId;
use nn::{CausalLm, Forward, KvView, SamplerState, Samples, TokenBatch};
use ops::{DType, Generation, TensorView};
use platform::DeviceId;

use crate::arena::{Pools, PoolMetrics, StreamId};
use crate::batch::{Batch, BatchColumns, BatchOutput, CallId, Job, Jobs, SampledToken};
use crate::captured::Captured;
use crate::graph::Graphs;
use crate::plan::{Bucket, Buckets, Shape, Variant};
use crate::runner::{Runner, RunnerId};
use crate::{Error, Result};

/// The host columns of one batch, padded to a bucket and held for the upload.
///
/// The buffers are allocated once, at bind, sized for the largest bucket, so serving a batch writes
/// into them and allocates nothing. The row layout is the model's `nn::BatchRows`, padded: a
/// padded token carries `slot_mapping = -1` (it writes no cache page) and a padded row carries a
/// commit slot below zero (it keeps no token state).
#[derive(Default, Debug)]
pub struct Staging {
    rows: nn::BatchRows,
    input_ids: Vec<i32>,
    slots: Vec<i32>,
    parameters: Vec<f32>,
    commit: Vec<i64>,
    page_size: u32,
    max_kv_len: u32,
    /// The mechanism's call ids of the staged rows, in the order `rows.calls` lists them.
    calls: Vec<CallId>,
    /// The shape staged by the last [`Staging::stage`].
    staged: Option<Shape>,
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
        self.parameters = vec![0.0; largest.rows as usize * ops::sampler::PARAMETER_COLUMNS as usize];
        // The commit table's height is ops' business; `reserve` leaves it empty and `stage` sizes
        // it to whatever the mechanism wrote, which `check` has already validated.
        self.commit = Vec::new();
        self.rows = nn::BatchRows::default();
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
        let rows = columns.rows.calls.len();
        if tokens > bucket.tokens as usize || rows > bucket.rows as usize {
            return Err(Error::Invalid(format!(
                "the batch's {rows} rows and {tokens} tokens do not fit bucket {bucket}"
            )));
        }
        let bucket_tokens = bucket.tokens as usize;
        let bucket_rows = bucket.rows as usize;

        // Token columns: the mechanism's tokens, then padding that persists nowhere.
        self.input_ids[..tokens].copy_from_slice(&columns.input_ids[..tokens]);
        for index in tokens..bucket_tokens {
            self.input_ids[index] = 0;
        }
        self.rows = columns.rows.clone();
        self.rows.positions.resize(bucket_tokens, 0);
        for index in tokens..bucket_tokens {
            self.rows.positions[index] = 0;
        }
        self.rows.slot_mapping.resize(bucket_tokens, -1);
        for index in tokens..bucket_tokens {
            self.rows.slot_mapping[index] = -1;
        }

        // Row columns: the mechanism's rows, then rows that sample nothing and commit nothing.
        self.slots[..rows].copy_from_slice(&columns.slots[..rows]);
        let parameters = ops::sampler::PARAMETER_COLUMNS as usize;
        self.parameters[..rows * parameters]
            .copy_from_slice(&columns.parameters[..rows * parameters]);
        for index in rows..bucket_rows {
            self.slots[index] = -1;
            for column in 0..parameters {
                self.parameters[index * parameters + column] = 0.0;
            }
        }
        self.rows.seq_lens.resize(bucket_rows, 1);
        for index in rows..bucket_rows {
            self.rows.seq_lens[index] = 1;
        }
        self.rows.logits_rows.resize(bucket_rows, -1);
        for index in rows..bucket_rows {
            self.rows.logits_rows[index] = -1;
        }

        // Padding tokens belong to the last call's token range, and padding rows are appended
        // after it, each evaluating the padding token that follows.
        let mut cursor = tokens;
        let calls = self.rows.calls.len();
        let _ = calls;
        if let Some(last) = self.rows.calls.last_mut() {
            if tokens < bucket_tokens {
                last.rows.end = bucket_tokens as u32;
                cursor = bucket_tokens;
            }
        }
        while self.rows.calls.len() < bucket_rows {
            let start = cursor as u32;
            let end = if bucket.variant.is_decode() && cursor < bucket_tokens {
                cursor += 1;
                cursor as u32
            } else {
                start
            };
            self.rows.calls.push(nn::BatchRow {
                request: 0,
                kind: if bucket.variant.is_decode() {
                    nn::RowKind::Decode
                } else {
                    nn::RowKind::Prefill
                },
                cache_group: 0,
                rows: start..end,
            });
        }
        if self.rows.calls.is_empty() || calls > bucket_rows {
            return Err(Error::Invariant(format!(
                "the batch has {calls} calls and does not fit {bucket_rows} rows"
            )));
        }

        // The commit table's padded rows keep their token state: a slot below zero.
        self.commit = columns.commit.clone();
        if !columns.commit.is_empty() {
            let columns = columns.commit.len() / rows.max(1);
            self.commit.resize(columns * bucket_rows, 0);
            for column in 0..columns {
                for row in rows..bucket_rows {
                    self.commit[column * bucket_rows + row] = -1;
                }
            }
        }
        self.page_size = columns.page_size;
        self.max_kv_len = columns.max_kv_len;
        self.calls.clear();
        self.calls.extend_from_slice(call_ids);
        self.calls.resize(bucket_rows, CallId(u64::MAX));

        self.staged = Some(bucket.shape());
        Ok(bucket.shape())
    }

    /// The shape staged by the last [`Staging::stage`], if any.
    pub fn staged(&self) -> Option<Shape> {
        self.staged
    }

    /// The staged rows, which the upload builds `nn::TokenBatch` from.
    pub fn rows(&self) -> &nn::BatchRows {
        &self.rows
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

    /// `ops::commit`'s column table of the staged rows.
    pub fn commit(&self) -> &[i64] {
        &self.commit
    }

    /// The page size and KV bound the upload hands the model with the rows.
    pub fn geometry(&self) -> (u32, u32) {
        (self.page_size, self.max_kv_len)
    }

    /// The sampling rows, in batch order (4.4's rows the sampler writes a token for).
    pub fn logits_rows(&self) -> &[i32] {
        &self.rows.logits_rows
    }

    /// The rows of the batch as the model reads them, padded to the bucket.
    pub fn batch_rows(&self) -> &nn::BatchRows {
        &self.rows
    }

    /// The mechanism's call ids of the staged rows, in the rows' order.
    pub fn calls(&self) -> &[CallId] {
        &self.calls
    }
}

/// The model's geometry: what the runner sizes a bucket's plan and the sampler from.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    /// The width of a hidden-state row.
    pub hidden: u32,
    /// The element type the model's forward reads and writes hidden states in.
    pub hidden_dtype: DType,
    /// The element type the head writes logits in.
    pub logits_dtype: DType,
    /// The vocabulary width of a logit row.
    pub vocabulary: u32,
    /// How many request slots the deployment serves: the sampler's penalty-count table's height.
    pub slots: u32,
    /// The largest top-k width a request may ask for: the sampler's candidate width.
    pub width: u32,
    /// Tokens per KV page.
    pub page_size: u32,
    /// The bound on the keys a row reads: the attention descriptor's reduction length.
    pub max_kv_len: u32,
}

/// One host column of a batch, uploaded into a device buffer whose address never moves.
///
/// A captured graph addresses these buffers directly, so they are allocated once when the runner
/// binds, sized for the largest bucket, and every step rewrites their contents instead of
/// allocating: what 4.5 forbids is an allocation inside a step, not a copy into a fixed buffer.
struct Column {
    device: TensorView,
    host: platform::PinnedBuffer,
}

impl Column {
    /// A column of `numel` elements of `dtype` on `device`.
    fn new(device: DeviceId, dtype: DType, numel: u64) -> Result<Self> {
        let bytes = dtype
            .bytes()
            .ok_or_else(|| Error::Invalid(format!("{dtype} has no whole-byte size")))?
            .checked_mul(numel)
            .ok_or_else(|| Error::Invalid("a batch column's extent overflows".into()))?;
        let host = platform::PinnedBuffer::new(device, bytes as usize)?;
        let device = TensorView::new(
            ops::DevicePtr::new(host.address()),
            dtype,
            ops::Shape::new(&[numel]),
        );
        Ok(Self { device, host })
    }

    /// Writes `bytes` into the host side; the device copy follows in [`Column::upload`].
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.host.len() {
            return Err(Error::Invariant(format!(
                "a column of {} bytes cannot hold {}",
                self.host.len(),
                bytes.len()
            )));
        }
        // SAFETY: the caller wrote `bytes.len()` bytes of one element type into a host buffer of at
        // least that size, allocated for exactly this purpose; the buffer is not read concurrently.
        unsafe {
            self.host.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        }
        Ok(())
    }

    /// Copies the host side into the device column; the caller's graph reads it afterwards.
    fn upload(&self, stream: &platform::Stream) -> Result<()> {
        // SAFETY: source and destination are this column's own host and device buffers, of the
        // same length, and the copy is stream-ordered before the graph that reads the device one.
        unsafe {
            platform::memcpy_async(
                self.device.ptr.address(),
                self.host.address(),
                self.host.len(),
                stream,
            )?;
        }
        Ok(())
    }

    /// The first `elements` of the column, as a view the batch reads.
    fn extent(&self, elements: std::ops::Range<u64>) -> TensorView {
        TensorView::new(
            self.device.ptr,
            self.device.dtype,
            ops::Shape::new(&[elements.end - elements.start]),
        )
    }
}

/// The batch columns of every bucket of one runner, sized for the largest bucket it serves.
struct Columns {
    /// The tokens the largest bucket holds.
    tokens: usize,
    /// The rows the largest bucket holds.
    rows: usize,
    positions: Column,
    slot_mapping: Column,
    seq_lens: Column,
    query_offsets: Column,
    logits_rows: Column,
    /// `[rows]` the token row each row's logits are gathered from, `0` for a row that samples
    /// nothing (its id is discarded, so the gather may read any in-range row).
    indices: Column,
    parameters: Column,
    slots: Column,
    draws: Column,
    /// `[rows]` the sampled ids, read back after a replay.
    readback: platform::PinnedBuffer,
}

impl Columns {
    /// Allocates the columns for the largest bucket `buckets` holds.
    fn new(device: DeviceId, buckets: &Buckets) -> Result<Self> {
        let largest = buckets
            .iter()
            .max_by_key(|bucket| (bucket.rows, bucket.tokens))
            .copied()
            .ok_or_else(|| Error::Invalid("the runner has no bucket to serve".into()))?;
        let tokens = largest.tokens as u64;
        let rows = largest.rows as u64;
        let parameters = rows * u64::from(ops::sampler::PARAMETER_COLUMNS);
        Ok(Self {
            tokens: tokens as usize,
            rows: rows as usize,
            positions: Column::new(device, DType::I32, tokens)?,
            slot_mapping: Column::new(device, DType::I32, tokens)?,
            seq_lens: Column::new(device, DType::I32, rows)?,
            query_offsets: Column::new(device, DType::I32, rows + 1)?,
            logits_rows: Column::new(device, DType::I32, rows)?,
            indices: Column::new(device, DType::I32, rows)?,
            parameters: Column::new(device, DType::F32, parameters)?,
            slots: Column::new(device, DType::I32, rows)?,
            draws: Column::new(device, DType::F32, rows)?,
            readback: platform::PinnedBuffer::new(device, (rows * 4) as usize)?,
        })
    }
}

/// Where one bucket's sampler wrote its ids: the first `rows` i32 of the plan's sampler block.
#[derive(Clone, Copy)]
struct Sampled {
    /// `[rows]` i32 sampled token ids, in the bucket's row order.
    tokens: TensorView,
}

/// The sampling step's own device state: the slot histories its penalties read.
struct Sampling {
    sampler: nn::Sampler,
    /// `[slots, vocabulary]` i32 counts of the tokens each slot has already seen, which the
    /// penalties subtract; zeroed once when the runner binds and advanced by `ops::commit`.
    counts: TensorView,
    _counts: platform::DeviceAllocation,
}

impl Sampling {
    /// The sampler and its zeroed count table for `geometry`.
    fn new(device: DeviceId, stream: &platform::Stream, geometry: Geometry) -> Result<Self> {
        let sampler = nn::Sampler::new(
            geometry.vocabulary,
            geometry.slots,
            geometry.width.min(geometry.vocabulary),
            0,
        )?;
        let bytes = u64::from(geometry.slots) * u64::from(geometry.vocabulary) * 4;
        let allocation = platform::DeviceAllocation::linear(
            device,
            usize::try_from(bytes)
                .map_err(|_| Error::Invalid("the sampler's count table is larger than memory".into()))?,
        )?;
        // The table starts at zero: a slot has seen no token before its first call. Zeroing is a
        // bind-time copy, not a step's, so a chunked upload from pinned memory is enough.
        let chunk = 1 << 22;
        let staging = platform::PinnedBuffer::new(device, chunk)?;
        // SAFETY: the staging buffer is this allocation's own pinned memory, wholly writable.
        let zeros = unsafe { staging.as_slice() };
        let mut written = 0u64;
        while written < bytes {
            let step = (bytes - written).min(chunk as u64);
            // SAFETY: source and destination are the pinned chunk and the count table's range, of
            // the same length, and the copy is stream-ordered before the first sample.
            unsafe {
                platform::memcpy_async(
                    allocation.address() + written,
                    staging.address(),
                    step as usize,
                    stream,
                )?;
            }
            written += step;
            let _ = zeros;
        }
        stream.synchronize()?;
        let counts = TensorView::new(
            ops::DevicePtr::new(allocation.address()),
            DType::I32,
            ops::Shape::new(&[
                u64::from(geometry.slots),
                u64::from(geometry.vocabulary),
            ]),
        );
        Ok(Self {
            sampler,
            counts,
            _counts: allocation,
        })
    }
}

/// What the runner needs from the block's resident KV cache to build a batch's `nn::KvView`.
///
/// The cache is the memory owner's (8's resident pages and its block tables); the runner only
/// reads it. `scratch` answers a page set of the paired forward's own, whose pages are never a
/// serving request's (7.3).
pub trait KvProvider: Send + Sync {
    /// How many attention groups the model's layers index.
    fn groups(&self) -> usize;

    /// Tokens per cache page.
    fn page_size(&self) -> u32;

    /// Element type of the KV planes.
    fn dtype(&self) -> DType;

    /// The block table of `group` for the batch rows whose slots are `slots`, `[rows, max_pages]`.
    ///
    /// The batch's rows are the calls' request slots, so this gathers one table row per batch row
    /// on `stream` and returns a view the captured graph addresses.
    fn block_table(
        &self,
        group: usize,
        slots: &[u32],
        stream: &platform::Stream,
    ) -> Result<TensorView>;

    /// The plane views of `group`'s layers, in model order.
    fn planes(&self, group: usize) -> Result<Vec<nn::KvPlanes>>;

    /// The per-page scales of `group`'s planes, for a quantized cache.
    fn scales(&self, group: usize) -> Option<(TensorView, TensorView)>;

    /// A scratch page set of `pages` pages for `group`, its table over `slots`, and its planes.
    fn scratch(
        &self,
        group: usize,
        slots: &[u32],
        pages: u32,
        stream: &platform::Stream,
    ) -> Result<(TensorView, Vec<nn::KvPlanes>, Option<(TensorView, TensorView)>)>;
}

/// A causal language model's runner: one block's decode and prefill phases.
pub struct CausalLmRunner<M: CausalLm> {
    /// The executor's name for this runner.
    id: RunnerId,
    /// The phases it serves, in scheduler order.
    phases: Vec<PhaseId>,
    /// The model the runner calls.
    model: M,
    /// The generation whose kernels serve the current batches.
    generation: Arc<Generation>,
    /// The old generation probation keeps for a revert (7.3).
    previous: Option<Arc<Generation>>,
    /// The generation precopy prepared and switchover will take (7.3).
    next: Option<Arc<Generation>>,
    /// Every bucket the deployment captures, in capture order.
    buckets: Buckets,
    /// The captured graphs of every generation the runner holds.
    graphs: Graphs,
    /// The device's graph pools and each bucket's static activation plan, shared with every other
    /// runner of this device (one pool per device, never per runner); the executor attaches them
    /// before the first capture.
    pools: Option<Arc<std::sync::Mutex<Pools>>>,
    /// The host columns of the batch being staged.
    staging: Staging,
    /// The device the runner computes on.
    device: DeviceId,
    /// The stream the executor issues this runner's work on: a clone of the device executor's
    /// primary stream. Only the executor calls into the runner, so holding it here does not make
    /// the runner a second issuer.
    stream: platform::Stream,
    /// Where shadow execution's twins live, allocated at bind, before the first capture.
    shadow: Option<crate::shadow::Shadow>,
    /// The model's geometry: hidden width, logits width, page size and the sampler's bounds.
    geometry: Geometry,
    /// Every bucket's per-step columns, whom the captured graphs address directly.
    columns: Columns,
    /// The sampling step's own state, when any bucket samples.
    sampling: Option<Sampling>,
    /// The sampling row's output views, per bucket's plan, written by `sample` and read by the
    /// replay's readback. Interior mutability because the walk's KV closure borrows the runner
    /// while it runs.
    bucket_samples: std::cell::RefCell<HashMap<Bucket, Sampled>>,
    /// Where the batch's KV views come from: the resident cache, and a paired forward's scratch.
    kv: Arc<dyn KvProvider>,
    /// The sampling step's uniform generator, seeded by the engine so a run reproduces.
    seed: std::cell::Cell<u64>,
}

impl<M: CausalLm> CausalLmRunner<M> {
    /// A runner for the block's phases on `device`, serving `generation`.
    ///
    /// It captures nothing here: the device executor's warmup captures every bucket after the
    /// runner is bound, largest first, so the pools' plans exist before serving starts.
    pub fn new(
        id: RunnerId,
        phases: Vec<PhaseId>,
        model: M,
        generation: Arc<Generation>,
        buckets: Buckets,
        device: DeviceId,
        stream: platform::Stream,
        geometry: Geometry,
        kv: Arc<dyn KvProvider>,
        seed: u64,
    ) -> Result<Self> {
        let mut staging = Staging::new();
        staging.reserve(&buckets)?;
        let columns = Columns::new(device, &buckets)?;
        let sampling = if buckets.iter().any(|bucket| bucket.variant.samples()) {
            Some(Sampling::new(device, &stream, geometry)?)
        } else {
            None
        };
        Ok(Self {
            id,
            phases,
            model,
            generation,
            previous: None,
            next: None,
            buckets,
            graphs: Graphs::new(),
            pools: None,
            staging,
            device,
            stream,
            shadow: None,
            geometry,
            columns,
            sampling,
            bucket_samples: std::cell::RefCell::new(HashMap::new()),
            kv,
            seed: std::cell::Cell::new(seed | 1),
        })
    }

    /// The runner's name in the executor.
    pub fn id(&self) -> RunnerId {
        self.id
    }

    /// The buckets the runner captures.
    pub fn buckets(&self) -> &Buckets {
        &self.buckets
    }

    /// The generation serving now.
    pub fn generation(&self) -> &Arc<Generation> {
        &self.generation
    }

    /// The captured graphs, for the executor's capture pass and the runner's tests.
    pub fn graphs(&self) -> &Graphs {
        &self.graphs
    }

    /// The graph pools this runner books its buckets in, once the executor attached them.
    pub fn pools(&self) -> Option<&Arc<std::sync::Mutex<Pools>>> {
        self.pools.as_ref()
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
            .ok_or_else(|| Error::State("the runner has no graph pools yet".into()))
    }

    /// What the graph pools hold and what they may hold, for the metrics report (4.1).
    pub fn pool_metrics(&self) -> PoolMetrics {
        self.pool().map(|pools| pools.metrics()).unwrap_or_default()
    }

    /// Installs the shadow harness, sized for the largest bucket's shadow set.
    pub fn set_shadow(&mut self, shadow: crate::shadow::Shadow) {
        self.shadow = Some(shadow);
    }

    /// The shape of the bucket a batch is served from.
    fn bucket_for(&self, shape: &Shape) -> Result<Bucket> {
        self.buckets.select(shape).ok_or_else(|| {
            Error::Invalid(format!(
                "no captured bucket holds {shape}: the bucket plan and the batch disagree"
            ))
        })
    }
}

impl<M: CausalLm + Send> Runner for CausalLmRunner<M> {
    fn phases(&self) -> &[PhaseId] {
        &self.phases
    }

    fn submit(&mut self, batch: Batch) -> Result<Jobs> {
        let shape = batch.shape;
        batch.columns.check(&shape)?;
        let bucket = self.bucket_for(&shape)?;
        self.staging.stage(&batch.columns, &batch.calls, &shape, bucket)?;
        // The upload and the device views are staged by the executor's stream work, then the
        // bucket's graph is replayed (design §9, `每批的规划`). Until `nn::TokenBatch`/`KvView`
        // and platform's allocator are published, the staging is as far as this path can go; the
        // replay job is what the executor performs once they are.
        let _ = &self.model;
        Ok(Jobs::new([Job::Replay {
            runner: self.id,
            bucket,
            generation: batch.generation,
            batch: batch.id,
        }]))
    }

    fn warmup(&mut self) -> Result<Jobs> {
        let generation = self.generation.number();
        let mut jobs = Vec::new();
        // Largest first, so the smaller captures reuse the larger ones' pool blocks
        // (`specs/replaceable-processes.md` 9.3).
        for bucket in self.buckets.capture_order().copied().collect::<Vec<_>>() {
            if self.graphs.captured(bucket, generation).is_none() {
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
        // Which buckets a changed operator forces to change, and whether the change can be a
        // rebind (same launch structure) or must be a recapture, is decided from the two
        // generations' kernels and each bucket's node table (4.5).
        let mut changed: Vec<(&'static str, String)> = Vec::new();
        for (operator, key, kernel) in self.generation.iter() {
            let same = next
                .kernel(operator, key)
                .map(|candidate| Arc::ptr_eq(candidate, kernel))
                .unwrap_or(false);
            if !same {
                changed.push((operator, key.to_owned()));
            }
        }
        // A launch the old generation does not have is a change too: it has nothing to compare
        // against, so a rebind of its bucket cannot be right and the bucket is recaptured.
        for (operator, key, _) in next.iter() {
            if !self.generation.contains(operator, key) {
                changed.push((operator, key.to_owned()));
            }
        }
        let mut jobs = Vec::new();
        for (operator, key) in changed {
            for bucket in self.graphs.using(operator, self.generation.number()) {
                // The paired forward runs first: a difference beyond the operator's tolerance
                // fails the precopy before anything is rebound or recaptured (7.3).
                jobs.push(Job::Shadow {
                    runner: self.id,
                    operator,
                    key: key.clone(),
                    bucket,
                    generation: next.number(),
                });
                let captured = self
                    .graphs
                    .captured(bucket, self.generation.number())
                    .ok_or_else(|| {
                        Error::Invariant("a bucket vanished from the graph table".into())
                    })?;
                let rebindable = captured.launches(operator) > 0
                    && captured
                        .kernels()
                        .iter()
                        .all(|kernel| kernel.metadata().is_some());
                if rebindable {
                    jobs.push(Job::Rebind {
                        runner: self.id,
                        bucket,
                        generation: next.number(),
                    });
                } else {
                    jobs.push(Job::Capture {
                        runner: self.id,
                        bucket,
                        generation: next.number(),
                    });
                }
            }
        }
        self.next = Some(Arc::clone(next));
        Ok(Jobs::new(jobs))
    }

    fn revert(&mut self, generation: u64) {
        // The prepared generation never serves: its graphs go, and with them the kernels they
        // held. The old generation's graphs, its arena plans and its sampler regions stay exactly
        // as they were, so a reverted update cannot disturb a batch in flight.
        let dropped = self.graphs.purge(generation);
        let dropped_count = dropped.len();
        if self.next.as_ref().map(|next| next.number()) == Some(generation) {
            self.next = None;
        }
        tracing::info!(
            dropped = dropped_count,
            generation,
            "reverted the prepared generation; the serving generation is unchanged"
        );
    }

    fn switchover(&mut self, generation: u64) {
        // From the next batch on, the prepared generation serves; the old one stays for a revert
        // (7.3 step 4). A generation that was not prepared here is a generation of kernels only,
        // and is taken as the serving one unchanged.
        let prepared = self
            .next
            .take()
            .filter(|next| next.number() == generation)
            .unwrap_or_else(|| {
                Arc::new(
                    self.generation
                        .replace(generation, std::iter::empty::<ops::KernelEntry>()),
                )
            });
        self.previous = Some(std::mem::replace(&mut self.generation, prepared));
    }

    fn attach_pools(&mut self, pools: Arc<std::sync::Mutex<Pools>>) {
        self.pools = Some(pools);
    }

    fn pool_metrics(&self) -> PoolMetrics {
        self.pool().map(|pools| pools.metrics()).unwrap_or_default()
    }

    fn purge(&mut self, generation: u64) {
        // The graphs go first, then their arena regions: `Captured` guarantees that order, and a
        // region released under a live executable would be reused by the next generation's
        // capture while the old graph still addressed it.
        let dropped = self.graphs.purge(generation);
        let dropped_count = dropped.len();
        let mut released = 0u64;
        for bucket in &dropped {
            let released_bucket = self
                .pool()
                .and_then(|mut pools| pools.release_bucket(StreamId(0), *bucket));
            match released_bucket {
                Ok(bytes) => released += bytes,
                // A pool that cannot shrink keeps the bytes mapped until the executor closes;
                // the purge itself still holds, so this is not a reason to fail the command.
                Err(error) => {
                    tracing::warn!(%error, %bucket, "releasing a purged bucket's arena region");
                }
            }
        }
        tracing::debug!(
            dropped = dropped_count,
            released,
            generation,
            "purged graphs of a replaced generation"
        );
        if self
            .previous
            .as_ref()
            .is_some_and(|previous| previous.number() < generation)
        {
            // The last `Arc` of the old generation drops here; every kernel whose graph was
            // rebound or destroyed unloads its library with it (7.3's purge).
            self.previous = None;
        }
    }

    fn serve(&mut self, job: &Job) -> Result<BatchOutput> {
        match job {
            Job::Replay {
                bucket,
                generation,
                batch,
                ..
            } => {
                let _ = batch;
                let stream = self.stream.clone();
                let captured = self.graphs.captured(*bucket, *generation).ok_or_else(|| {
                    Error::Invariant(format!(
                        "no graph of bucket {bucket} in generation {generation}"
                    ))
                })?;
                captured.launch(&stream)?;
                // The sampled tokens are read back now, before the submission completes: the
                // mechanism builds the next step's input ids from them (4.4).
                self.read_sampled(*bucket)
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
                .shadow(*operator, key, *bucket, *generation)
                .map(|()| BatchOutput::default()),
            Job::Rebind {
                bucket, generation, ..
            } => self
                .rebind(*bucket, *generation)
                .map(|()| BatchOutput::default()),
            Job::Issue { .. } | Job::After { .. } => Err(Error::Invariant(
                "the device queue performs its own jobs".into(),
            )),
        }
    }

}

impl<M: CausalLm> CausalLmRunner<M> {
    /// The stream the executor issues this runner's work on.
    pub fn stream(&self) -> &platform::Stream {
        &self.stream
    }

    /// The device the runner computes on.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// Captures one bucket for one generation: runs the bucket's forward eagerly once under a
    /// stream capture, records every operator launch's nodes, then instantiates the template
    /// (4.5). The forward executes nothing while it is captured.
    fn capture(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        let kernels = self.generation_named(generation)?;
        let stream = self.stream.clone();
        let capture = Arc::new(std::sync::Mutex::new(platform::Capture::begin(&stream, platform::CaptureMode::Relaxed)?));
        let mut recording = Recording::new(Arc::clone(&capture));
        // The model's forward runs with the generation being captured and the bucket's arena, so
        // every operator launch becomes a node of the template and is reported to `recording`.
        ops::capture::set_sink(Some(&mut recording));
        let walked = self.walk(bucket, &kernels);
        ops::capture::set_sink(None);
        let nodes = recording.nodes;
        if let Some(failure) = recording.failure {
            return Err(failure.into());
        }
        walked?;
        let capture = Arc::try_unwrap(capture)
            .map_err(|_| Error::Invariant("the capture is still shared by its recorder".into()))?
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let template = capture.end()?;

        // The graph holds the kernels it launches, so none unloads while it can replay; purge
        // releases them after the last batch that used this generation (7.3).
        let mut held: Vec<Arc<ops::Kernel>> = Vec::new();
        for (operator, key, _) in &nodes {
            let kernel = kernels.kernel(operator, key)?;
            if !held.iter().any(|existing| Arc::ptr_eq(existing, kernel)) {
                held.push(Arc::clone(kernel));
            }
        }
        let captured = Captured::new(template, nodes, held)?;
        self.graphs.insert(bucket, generation, captured);
        Ok(())
    }

    /// Rebinds one bucket from the next generation's precopy walk: captures the forward with the
    /// new kernels on a scratch stream, reads each changed launch's parameters from that
    /// throwaway template, and sets them on the live executable (4.5). Nothing is launched.
    fn rebind(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        let kernels = self.generation_named(generation)?;
        let current = self.generation.number();
        if self.graphs.captured(bucket, current).is_none() {
            return Err(Error::Invariant(format!(
                "no graph of bucket {bucket} in generation {current}"
            )));
        }
        let stream = self.stream.clone();
        let capture = Arc::new(std::sync::Mutex::new(platform::Capture::begin(&stream, platform::CaptureMode::Relaxed)?));
        let mut recording = Recording::new(Arc::clone(&capture));
        ops::capture::set_sink(Some(&mut recording));
        let walked = self.walk(bucket, &kernels);
        ops::capture::set_sink(None);
        let nodes = recording.nodes;
        if let Some(failure) = recording.failure {
            return Err(failure.into());
        }
        walked?;
        let capture = Arc::try_unwrap(capture)
            .map_err(|_| Error::Invariant("the capture is still shared by its recorder".into()))?
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let template = capture.end()?;

        let served = self.graphs.captured(bucket, current).ok_or_else(|| {
            Error::Invariant(format!("no graph of bucket {bucket} in generation {current}"))
        })?;
        // The walk's launches are the same ones, in the same order, as the live graph's: pair
        // them, read each node's parameters from the template, and set them on the executable.
        let mut launch_of: std::collections::HashMap<&'static str, usize> =
            std::collections::HashMap::new();
        let mut applied = 0usize;
        for (operator, _, walk_nodes) in &nodes {
            let ordinal = launch_of.entry(*operator).or_insert(0);
            // A capture adds both kernel nodes and (for example) the empty nodes that carry a
            // dependency; only kernel nodes have launch parameters to carry over.
            let mut params = Vec::with_capacity(walk_nodes.len());
            for node in walk_nodes {
                if node.kind()? != platform::GraphNodeKind::Kernel {
                    continue;
                }
                params.push(template.kernel_node_params(node)?);
            }
            applied += served.rebind_launch(operator, *ordinal, &params)?;
            *ordinal += 1;
        }
        tracing::debug!(bucket = %bucket, applied, "rebound a bucket for a new generation");
        Ok(())
    }

    /// Runs the paired forward of one changed operator and compares its two results (7.3).
    ///
    /// The old kernel writes the real outputs and the new one its twins; the forward continues on
    /// the old kernel's result, so the difference of one operator cannot accumulate. The paired
    /// forward runs without a graph, on the runner's stream, so it touches no captured bucket, and
    /// its KV goes to scratch pages: nothing a serving request reads is written.
    fn shadow(
        &mut self,
        operator: &'static str,
        key: &str,
        bucket: Bucket,
        generation: u64,
    ) -> Result<()> {
        let next = self.generation_named(generation)?;
        let previous = if self.generation.number() == generation {
            self.previous.clone()
        } else {
            Some(Arc::clone(&self.generation))
        }
        .ok_or_else(|| {
            Error::Invariant(
                "shadow execution needs the generation being replaced kept beside the new one"
                    .into(),
            )
        })?;
        let mut harness = self.shadow.take().ok_or_else(|| {
            Error::State(
                "the runner has no shadow harness: it is allocated when the runner is bound".into(),
            )
        })?;
        harness.reset();
        let stream = self.stream.clone();
        let walked = self.walk_paired(bucket, &next, &previous, operator, key, harness.twins());
        // The operator's tolerance is the one its paired forward recorded: `Forward::operate`
        // calls `O::tolerance(descriptor)` when it shadows, and the harness compares with it.
        let compared = walked.and_then(|()| {
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
        compared.map(|_| ())
    }

    /// Walks one bucket's forward with two kernels of `operator`: `previous`'s on the real outputs
    /// and `next`'s on the twins (7.3).
    ///
    /// The paired forward runs over the batch's own columns but a scratch KV: every page it writes
    /// is a scratch page, so the new kernel cannot touch a page a serving request reads. The
    /// model's `Forward::operate` does the branch, once, for every layer alike.
    fn walk_paired(
        &mut self,
        bucket: Bucket,
        next: &Arc<Generation>,
        previous: &Arc<Generation>,
        operator: &'static str,
        key: &str,
        twins: &mut ops::Twins,
    ) -> Result<()> {
        let stream = self.stream.clone();
        match self.staging.staged() {
            Some(staged) if staged.variant == bucket.variant => {}
            _ => {
                self.stage_warm(bucket)?;
            }
        }
        self.upload(bucket)?;
        let base = self.pool()?.base(StreamId(0))?;
        let arena = nn::Arena::new(
            ops::DevicePtr::new(base.address()),
            self.pool()?.reserved(),
        );
        let batch = self.token_batch(bucket)?;
        let shadow = nn::forward::Shadow::new(operator, key, previous, twins);
        let forward = Forward::new(&stream, next, &arena).with_shadow(shadow);
        let hidden = arena.tensor(
            self.geometry.hidden_dtype,
            ops::Shape::new(&[u64::from(bucket.tokens), u64::from(self.geometry.hidden)]),
        )?;
        // Every row writes its own scratch page: the paired forward must not write a resident one.
        let pages = bucket
            .tokens
            .div_ceil(self.geometry.page_size.max(1))
            .max(1);
        let slots: Vec<u32> = self
            .staging
            .batch_rows()
            .calls
            .iter()
            .map(|call| call.request)
            .collect();
        self.with_kv(&slots, Some(pages), |cache| {
            self.model
                .forward(&forward, &batch, cache, hidden)
                .map_err(|error| Error::Interface(error.to_string()))
        })
    }

    /// The generation with `number`: the serving one, the prepared one, or the old one probation
    /// keeps.
    fn generation_named(&self, number: u64) -> Result<Arc<Generation>> {
        for candidate in [Some(&self.generation), self.next.as_ref(), self.previous.as_ref()]
            .into_iter()
            .flatten()
        {
            if candidate.number() == number {
                return Ok(Arc::clone(candidate));
            }
        }
        Err(Error::Invariant(format!(
            "the runner holds no generation {number}"
        )))
    }

    /// Reads the bucket's sampled token ids back and names the call each belongs to (4.4).
    ///
    /// The sampling step wrote one `i32` per sampling row into the bucket's plan; the readback
    /// copies them into the runner's pinned buffer after the replay's stream completed, and the
    /// batch's columns say which call each row belongs to. A bucket that samples nothing, or a
    /// batch whose rows all end no prompt, answers with no token.
    fn read_sampled(&mut self, bucket: Bucket) -> Result<BatchOutput> {
        let Some(samples) = self.bucket_samples.borrow().get(&bucket).copied() else {
            return Ok(BatchOutput::default());
        };
        self.stream.synchronize()?;
        let rows = bucket.rows as usize;
        let bytes = rows * std::mem::size_of::<i32>();
        if bytes > self.columns.readback.len() {
            return Err(Error::Invariant(
                "the sampler's readback buffer is smaller than the bucket".into(),
            ));
        }
        // SAFETY: the sampled ids are `rows` i32 in the bucket's plan and the readback buffer is
        // this runner's pinned memory of at least that size; the stream has completed.
        unsafe {
            platform::memcpy_async(
                self.columns.readback.address(),
                samples.tokens.ptr.address(),
                bytes,
                &self.stream,
            )?;
        }
        self.stream.synchronize()?;
        // SAFETY: the copy above filled `rows` i32 of pinned host memory with the sampled ids.
        let ids: &[i32] = unsafe {
            std::slice::from_raw_parts(self.columns.readback.as_ptr().cast::<i32>(), rows)
        };
        sampled_output(
            ids,
            self.staging.logits_rows(),
            self.staging.batch_rows(),
            self.staging.calls(),
        )
    }

    /// Uploads the staged batch into the device columns the captured graphs address.
    ///
    /// Every column is copied into a buffer allocated when the runner bound, sized for the largest
    /// bucket, so a step writes fixed addresses and allocates nothing (4.5).
    fn upload(&mut self, bucket: Bucket) -> Result<()> {
        let rows = bucket.rows as usize;
        let tokens = bucket.tokens as usize;
        if self.staging.staged().is_none() {
            return Err(Error::State("a step needs a staged batch".into()));
        }
        let encoded = |values: &[i32]| -> Vec<u8> {
            values.iter().flat_map(|value| value.to_ne_bytes()).collect()
        };
        let batch = self.staging.batch_rows();
        self.columns.positions.write(&encoded(&batch.positions[..tokens]))?;
        self.columns
            .slot_mapping
            .write(&encoded(&batch.slot_mapping[..tokens]))?;
        self.columns.seq_lens.write(&encoded(&batch.seq_lens[..rows]))?;
        self.columns
            .query_offsets
            .write(&encoded(&batch.query_offsets[..rows + 1]))?;
        self.columns
            .logits_rows
            .write(&encoded(&batch.logits_rows[..rows]))?;
        // A row that samples nothing still gathers some in-range token; its id is discarded.
        let indices: Vec<i32> = batch.logits_rows[..rows]
            .iter()
            .map(|row| if *row < 0 { 0 } else { *row })
            .collect();
        self.columns.indices.write(&encoded(&indices))?;
        self.columns.parameters.write(
            &self
                .staging
                .parameters(rows)
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<u8>>(),
        )?;
        self.columns.slots.write(&encoded(self.staging.slots(rows)))?;
        self.columns.draws.write(
            &self
                .draw_uniforms(rows)
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<u8>>(),
        )?;
        for column in [
            &self.columns.positions,
            &self.columns.slot_mapping,
            &self.columns.seq_lens,
            &self.columns.query_offsets,
            &self.columns.logits_rows,
            &self.columns.indices,
            &self.columns.parameters,
            &self.columns.slots,
            &self.columns.draws,
        ] {
            column.upload(&self.stream)?;
        }
        Ok(())
    }

    /// `rows` uniform draws in `[0, 1)` for the sampling step, from the runner's generator.
    ///
    /// The generator is the runner's own: sampling state is the executor's (4.5), and a seeded
    /// generator is what makes a replay reproduce its tokens.
    fn draw_uniforms(&self, rows: usize) -> Vec<f32> {
        let mut state = self.seed.get();
        let mut draws = Vec::with_capacity(rows);
        while draws.len() < rows {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // The top 24 bits are a value in [0, 1) that cannot round up to one in f32.
            draws.push(((state >> 40) as f32) / ((1u64 << 24) as f32));
        }
        self.seed.set(state);
        draws
    }

    /// The device-side `TokenBatch` of `bucket`: the bucket's extent of each column.
    fn token_batch(&self, bucket: Bucket) -> Result<TokenBatch> {
        Ok(TokenBatch {
            positions: self.columns.positions.extent(0..bucket.tokens as u64),
            slot_mapping: self.columns.slot_mapping.extent(0..bucket.tokens as u64),
            seq_lens: self.columns.seq_lens.extent(0..bucket.rows as u64),
            query_offsets: self.columns.query_offsets.extent(0..bucket.rows as u64 + 1),
            logits_rows: self.columns.logits_rows.extent(0..bucket.rows as u64),
            phase: match bucket.variant.is_decode() {
                true => ops::PhaseKind::Decode,
                false => ops::PhaseKind::Prefill,
            },
            page_size: self.geometry.page_size,
            num_tokens: bucket.tokens,
            max_kv_len: self.geometry.max_kv_len,
            draft_lens: None,
        })
    }

    /// Runs `operation` with the batch's `KvView` over `slots`, resident or scratch.
    ///
    /// The view lives as long as the closure: `KvGroup` borrows its layer planes, so the planes,
    /// tables and groups are locals of one scope and the model reads them borrowed.
    fn with_kv<R>(
        &self,
        slots: &[u32],
        scratch_pages: Option<u32>,
        operation: impl FnOnce(&KvView<'_>) -> Result<R>,
    ) -> Result<R> {
        let groups = self.kv.groups();
        let mut layers: Vec<Vec<nn::KvPlanes>> = Vec::with_capacity(groups);
        let mut tables = Vec::with_capacity(groups);
        let mut scales = Vec::with_capacity(groups);
        for group in 0..groups {
            match scratch_pages {
                None => {
                    layers.push(self.kv.planes(group)?);
                    tables.push(self.kv.block_table(group, slots, &self.stream)?);
                    scales.push(self.kv.scales(group));
                }
                Some(pages) => {
                    let (table, planes, scale) = self.kv.scratch(group, slots, pages, &self.stream)?;
                    layers.push(planes);
                    tables.push(table);
                    scales.push(scale);
                }
            }
        }
        let view: Vec<nn::KvGroup<'_>> = (0..groups)
            .map(|group| nn::KvGroup {
                block_tables: tables[group],
                page_size: self.kv.page_size(),
                dtype: self.kv.dtype(),
                key_scales: scales[group].map(|(key, _)| key),
                value_scales: scales[group].map(|(_, value)| value),
                layers: &layers[group],
            })
            .collect();
        operation(&KvView { groups: &view })
    }

    /// Walks one bucket's forward under the active capture with `generation`'s kernels.
    ///
    /// It is the same forward a replay runs, over the same columns and in the same arena order, so
    /// the nodes it records are the ones the live graph has and the addresses are the same each
    /// time. It launches nothing: the stream is capturing.
    fn walk(&mut self, bucket: Bucket, generation: &Arc<Generation>) -> Result<()> {
        let stream = self.stream.clone();
        match self.staging.staged() {
            Some(staged) if staged.variant == bucket.variant => {}
            _ => {
                self.stage_warm(bucket)?;
            }
        }
        let first = self.pool()?.begin(StreamId(0), bucket)?;
        self.upload(bucket)?;
        let base = self.pool()?.base(StreamId(0))?;
        let arena = nn::Arena::new(
            ops::DevicePtr::new(base.address()),
            self.pool()?.reserved(),
        );
        let batch = self.token_batch(bucket)?;
        let forward = Forward::new(&stream, generation, &arena);
        let hidden = arena.tensor(
            self.geometry.hidden_dtype,
            ops::Shape::new(&[u64::from(bucket.tokens), u64::from(self.geometry.hidden)]),
        )?;
        let slots: Vec<u32> = self
            .staging
            .batch_rows()
            .calls
            .iter()
            .map(|call| call.request)
            .collect();
        self.with_kv(&slots, None, |cache| {
            self.model
                .forward(&forward, &batch, cache, hidden)
                .map_err(|error| Error::Interface(error.to_string()))?;
            if bucket.variant.samples() {
                self.sample(&forward, bucket, &arena, hidden)?;
            }
            Ok(())
        })?;
        if first {
            self.pool()?.record(StreamId(0), bucket, arena.used())?;
        }
        Ok(())
    }

    /// Appends the bucket's sampling step to the capture in flight (4.4, 9's `采样`).
    ///
    /// The step gathers the rows whose logits are wanted, projects them, and samples one token per
    /// row through the model's sampler; its working set comes from the bucket's arena, so a replay
    /// samples without a host round trip.
    fn sample(
        &self,
        forward: &Forward<'_>,
        bucket: Bucket,
        arena: &nn::Arena,
        hidden: TensorView,
    ) -> Result<()> {
        let Some(sampling) = &self.sampling else {
            return Err(Error::State(
                "the runner has no sampler but the bucket samples".into(),
            ));
        };
        let rows = u64::from(bucket.rows);
        let gathered = arena.tensor(
            self.geometry.hidden_dtype,
            ops::Shape::new(&[rows, u64::from(self.geometry.hidden)]),
        )?;
        nn::gather_rows(forward, hidden, self.columns.indices.device, gathered)
            .map_err(|error| Error::Interface(error.to_string()))?;
        let logits = arena.tensor(
            self.geometry.logits_dtype,
            ops::Shape::new(&[rows, u64::from(self.geometry.vocabulary)]),
        )?;
        self.model
            .logits(forward, gathered, logits)
            .map_err(|error| Error::Interface(error.to_string()))?;
        let state = SamplerState {
            counts: sampling.counts,
            parameters: self.columns.parameters.device,
            slots: self.columns.slots.device,
            extras: None,
        };
        let samples = Samples::reserve(arena, DType::I32, rows, false)?;
        sampling
            .sampler
            .forward(forward, logits, &state, self.columns.draws.device, &samples)
            .map_err(|error| Error::Interface(error.to_string()))?;
        self.bucket_samples
            .borrow_mut()
            .insert(bucket, Sampled { tokens: samples.tokens });
        Ok(())
    }

    /// Stages a batch of the bucket's full size with nothing to write: what a warmup capture
    /// walks (the same allocations, the same launches as a served batch of that bucket).
    fn stage_warm(&mut self, bucket: Bucket) -> Result<Shape> {
        let tokens = bucket.tokens as usize;
        let rows = bucket.rows as usize;
        let columns = BatchColumns {
            rows: nn::BatchRows {
                positions: vec![0; tokens],
                slot_mapping: vec![-1; tokens],
                seq_lens: vec![1; rows],
                query_offsets: (0..=rows as i32).collect(),
                logits_rows: vec![-1; rows],
                calls: vec![nn::BatchRow {
                    request: 0,
                    kind: if bucket.variant.is_decode() {
                        nn::RowKind::Decode
                    } else {
                        nn::RowKind::Prefill
                    },
                    cache_group: 0,
                    rows: 0..bucket.tokens,
                }],
            },
            input_ids: vec![0; tokens],
            slots: vec![0; rows],
            parameters: vec![0.0; rows * ops::sampler::PARAMETER_COLUMNS as usize],
            commit: Vec::new(),
            page_size: 1,
            max_kv_len: 1,
        };
        self.staging
            .stage(&columns, &[CallId(0)], &bucket.shape(), bucket)
    }

}

/// Records the nodes each operator launch adds during a capture (4.5's `Captured.nodes`).
///
/// `ops` reports every launch to the installed sink, because the executor never sees an
/// individual `Op::launch`; the sink asks the capture which nodes the launch added and keeps the
/// per-launch table. The sink must own its data (`ops::capture` keeps a thread-local pointer), so
/// it holds the capture behind an `Arc`: the capture outlives every report, and it is taken back
/// after the sink is removed.
struct Recording {
    capture: Arc<std::sync::Mutex<platform::Capture>>,
    nodes: crate::captured::Nodes,
    failure: Option<platform::Error>,
}

impl Recording {
    fn new(capture: Arc<std::sync::Mutex<platform::Capture>>) -> Self {
        Self {
            capture,
            nodes: Vec::new(),
            failure: None,
        }
    }
}

impl ops::capture::Sink for Recording {
    fn launched(&mut self, operator: &'static str, key: &str) {
        let recorded = {
            let mut capture = self
                .capture
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            capture.record()
        };
        match recorded {
            Ok(nodes) => self.nodes.push((operator, key.to_owned(), nodes)),
            Err(error) => {
                if self.failure.is_none() {
                    self.failure = Some(error);
                }
            }
        }
    }
}

/// Names each sampled row's call: the step's output in the mechanism's order (4.4).
///
/// `sampling_rows` are the batch rows a token is sampled for, in batch order; `calls` name the row
/// ranges of the batch. A sampling row outside every call is an invariant violation: the
/// mechanism's columns and its calls disagree about the batch.
pub fn sampled_output(
    ids: &[i32],
    sampling_rows: &[i32],
    rows: &nn::BatchRows,
    calls: &[CallId],
) -> Result<BatchOutput> {
    if ids.len() < sampling_rows.len() {
        return Err(Error::Invariant(format!(
            "the bucket sampled {} rows and the batch names {}",
            ids.len(),
            sampling_rows.len()
        )));
    }
    let mut tokens = Vec::with_capacity(sampling_rows.len());
    for (index, row) in sampling_rows.iter().copied().enumerate() {
        // `logits_rows` names the token row that feeds `logits`; a negative entry is a row that
        // samples nothing (a padded row, or a prefill chunk that ends no prompt).
        if row < 0 {
            continue;
        }
        let row = row as u32;
        let (position, _) = rows
            .calls
            .iter()
            .enumerate()
            .find(|(_, call)| call.rows.contains(&row))
            .ok_or_else(|| {
                Error::Invariant(format!("no call of the batch owns sampling row {row}"))
            })?;
        let call = calls.get(position).copied().ok_or_else(|| {
            Error::Invariant(format!(
                "the batch names no call id for its row {position}"
            ))
        })?;
        tokens.push(SampledToken {
            call,
            token: ids[index] as u32,
        });
    }
    Ok(BatchOutput { tokens })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_columns() -> BatchColumns {
        BatchColumns {
            rows: nn::BatchRows {
                positions: vec![3, 4],
                slot_mapping: vec![10, 11],
                seq_lens: vec![4, 9],
                query_offsets: vec![0, 1, 2],
                logits_rows: vec![0, 1],
                calls: vec![nn::BatchRow {
                    request: 0,
                    kind: nn::RowKind::Decode,
                    cache_group: 0,
                    rows: 0..2,
                }],
            },
            input_ids: vec![7, 8],
            slots: vec![0, 1],
            parameters: vec![0.0; 2 * ops::sampler::PARAMETER_COLUMNS as usize],
            commit: vec![0, 1],
            page_size: 16,
            max_kv_len: 4096,
        }
    }

    fn prefill_columns() -> BatchColumns {
        BatchColumns {
            rows: nn::BatchRows {
                positions: vec![0, 1, 2],
                slot_mapping: vec![0, 1, 2],
                seq_lens: vec![3],
                query_offsets: vec![0, 3],
                logits_rows: vec![2],
                calls: vec![nn::BatchRow {
                    request: 0,
                    kind: nn::RowKind::Prefill,
                    cache_group: 0,
                    rows: 0..3,
                }],
            },
            input_ids: vec![1, 2, 3],
            slots: vec![0],
            parameters: vec![0.0; ops::sampler::PARAMETER_COLUMNS as usize],
            commit: vec![0],
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
            &[CallId(5)],
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

    fn rows(calls: &[(u32, std::ops::Range<u32>)]) -> nn::BatchRows {
        nn::BatchRows {
            positions: Vec::new(),
            slot_mapping: Vec::new(),
            seq_lens: Vec::new(),
            query_offsets: Vec::new(),
            logits_rows: Vec::new(),
            calls: calls
                .iter()
                .map(|(request, rows)| nn::BatchRow {
                    request: *request,
                    kind: nn::RowKind::Decode,
                    cache_group: 0,
                    rows: rows.clone(),
                })
                .collect(),
        }
    }

    /// The step's output names each sampled row's call, in the batch's row order, whatever order
    /// the mechanism listed the calls in.
    #[test]
    fn sampled_rows_are_named_by_their_calls() {
        let batch = rows(&[(0, 0..1), (1, 1..2)]);
        let output = sampled_output(&[111, 222], &[1, 0], &batch, &[CallId(7), CallId(9)])
            .expect("both rows belong to a call");
        assert_eq!(
            output.tokens,
            vec![
                SampledToken {
                    call: CallId(9),
                    token: 222
                },
                SampledToken {
                    call: CallId(7),
                    token: 111
                },
            ]
        );
    }

    /// A batch with no sampling row produces no token: a prefill chunk that samples nothing and a
    /// bucket whose rows are all padding.
    #[test]
    fn a_batch_that_samples_nothing_produces_no_tokens() {
        let output = sampled_output(&[], &[-1], &rows(&[(0, 0..4)]), &[CallId(1)])
            .expect("no sampling rows");
        assert!(output.tokens.is_empty());
    }

    /// A sampling row outside every call, or fewer sampled ids than sampling rows, is an invariant
    /// violation: silently dropping either would hand the mechanism the wrong request's token.
    #[test]
    fn a_row_no_call_owns_is_refused() {
        let batch = rows(&[(0, 0..1)]);
        sampled_output(&[5], &[3], &batch, &[CallId(1)]).expect_err("row 3 belongs to no call");
        sampled_output(&[], &[0], &batch, &[CallId(1)]).expect_err("no id was sampled for row 0");
    }
}
