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
use nn::CausalLm;
use ops::Generation;
use platform::DeviceId;

use crate::arena::{Pools, StreamId};
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
        calls: &[CallId],
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
        self.calls.extend_from_slice(calls);
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
    /// The graph pools and each bucket's static activation plan.
    pools: Pools,
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
    /// The sampler output of every bucket that samples, addressed in its plan (4.4).
    samplers: HashMap<Bucket, SamplerBuffer>,
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
        pools: Pools,
        device: DeviceId,
        stream: platform::Stream,
    ) -> Result<Self> {
        let mut staging = Staging::new();
        staging.reserve(&buckets)?;
        Ok(Self {
            id,
            phases,
            model,
            generation,
            previous: None,
            next: None,
            buckets,
            graphs: Graphs::new(),
            pools,
            staging,
            device,
            stream,
            shadow: None,
            samplers: HashMap::new(),
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

    /// The graph pools.
    pub fn pools(&self) -> &Pools {
        &self.pools
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
        if self.next.as_ref().map(|next| next.number()) == Some(generation) {
            self.next = None;
        }
        tracing::info!(
            dropped,
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

    fn purge(&mut self, generation: u64) {
        // The graphs go first, then their arena regions: `Captured` guarantees that order, and a
        // region released under a live executable would be reused by the next generation's
        // capture while the old graph still addressed it.
        let dropped = self.graphs.purge(generation);
        let mut released = 0u64;
        for bucket in &dropped {
            match self.pools.release_bucket(StreamId(0), *bucket) {
                Ok(bytes) => released += bytes,
                // A pool that cannot shrink keeps the bytes mapped until the executor closes;
                // the purge itself still holds, so this is not a reason to fail the command.
                Err(error) => {
                    tracing::warn!(%error, %bucket, "releasing a purged bucket's arena region");
                }
            }
        }
        tracing::debug!(
            dropped = dropped.len(),
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
                bucket, generation, ..
            } => {
                let captured = self.graphs.captured(*bucket, *generation).ok_or_else(|| {
                    Error::Invariant(format!(
                        "no graph of bucket {bucket} in generation {generation}"
                    ))
                })?;
                self.replay(captured)
            }
            Job::Capture { bucket, generation } => self.capture(*bucket, *generation),
            Job::Shadow {
                operator,
                key,
                bucket,
                generation,
            } => self.shadow(*operator, key, *bucket, *generation),
            Job::Rebind { bucket, generation } => self.rebind(*bucket, *generation),
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
        let mut capture = platform::Capture::begin(&stream)?;
        let nodes = {
            let mut recording = Recording::new(&mut capture);
            // The model's forward runs with the generation being captured and the bucket's
            // arena, so every operator launch becomes a node of the template and is reported to
            // `recording` by the operator layer.
            ops::capture::set_sink(Some(&mut recording));
            let walked = self.walk(bucket, &kernels);
            ops::capture::set_sink(None);
            walked?;
            recording.nodes
        };
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
        if bucket.variant.samples() {
            // The sampler output is the plan's first block; the buffer is the runner's host copy.
            let base = self.pools.base(StreamId(0))?;
            let sampler = SamplerBuffer::new(self.device, base, bucket.rows)?;
            self.samplers.insert(bucket, sampler);
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
        let served = self.graphs.captured(bucket, current).ok_or_else(|| {
            Error::Invariant(format!("no graph of bucket {bucket} in generation {current}"))
        })?;
        let stream = self.stream.clone();
        let mut capture = platform::Capture::begin(&stream)?;
        let nodes = {
            let mut recording = Recording::new(&mut capture);
            ops::capture::set_sink(Some(&mut recording));
            let walked = self.walk(bucket, &kernels);
            ops::capture::set_sink(None);
            walked?;
            recording.nodes
        };
        let template = capture.end()?;

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
    /// The model's `Forward::operate` does the branch, for every layer alike: with the shadow
    /// field set it calls `O::shadow`, launches the new kernel into the twins and the old one into
    /// the real outputs; every other operator launches from `next` only.
    fn walk_paired(
        &mut self,
        _bucket: Bucket,
        _next: &Arc<Generation>,
        _previous: &Arc<Generation>,
        _operator: &'static str,
        _key: &str,
        _twins: &mut ops::Twins,
    ) -> Result<()> {
        Err(Error::State(
            "the paired forward needs nn::Forward::operate (exec-work M169/M159): the model's \
             layers call it, and the runner only sets the forward's shadow field"
                .into(),
        ))
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
    /// The bucket's sampler writes one `u32` per sampling row; the staged columns name those rows
    /// (`logits_rows`) and the call each row belongs to, so the output preserves the mechanism's
    /// request order. A bucket that samples nothing has no sampler buffer and answers empty.
    fn read_sampled(&mut self, bucket: Bucket) -> Result<BatchOutput> {
        let Some(sampler) = self.samplers.get(&bucket) else {
            return Ok(BatchOutput::default());
        };
        self.stream.synchronize()?;
        sampler.read_back(&self.stream)?;
        sampled_output(
            sampler.host_tokens(),
            self.staging.logits_rows(),
            self.staging.batch_rows(),
            self.staging.calls(),
        )
    }

    /// Walks one bucket's forward under the active capture with `generation`'s kernels.
    ///
    /// It is the same forward a replay runs, with the same staged columns, so the nodes it records
    /// are the ones the live graph has. It launches nothing: the stream is capturing.
    fn walk(&mut self, bucket: Bucket, generation: &Arc<Generation>) -> Result<()> {
        let stream = self.stream.clone();
        // A capture during warmup has no batch: the bucket's own full size is staged, with every
        // padding token writing nothing. A capture of a served batch reuses its staged columns.
        let staged = match self.staging.staged() {
            Some(staged) if staged.variant == bucket.variant => staged,
            _ => self.stage_warm(bucket)?,
        };
        // The bucket's plan in the graph pool: the sampler output first (a bucket that samples
        // owns it, so a step allocates nothing), then the arena the forward allocates from.
        let first = self.pools.begin(StreamId(0), bucket)?;
        let sampler_bytes = if bucket.variant.samples() {
            SamplerBuffer::bytes(bucket.rows)
        } else {
            0
        };
        let base = self.pools.base(StreamId(0))?;
        let arena_base = platform::DevicePtr::from_address(base.address() + sampler_bytes);
        let arena = nn::Arena::new(arena_base, self.pools.reserved().saturating_sub(sampler_bytes));
        let forward = nn::Forward {
            stream: &stream,
            kernels: generation,
            arena: &arena,
            group: None,
        };
        let batch = self.token_batch(&arena_base, &staged)?;
        let cache = self.kv_view()?;
        let hidden = self.hidden(&arena_base, &staged)?;
        self.model.forward(&forward, &batch, &cache, hidden.clone())?;
        if bucket.variant.samples() {
            // The sampler runs inside this capture too, so a replay samples in the graph: its
            // nodes are the bucket's, and its region is the plan's first block.
            self.append_sampler(&stream, bucket, arena_base, &hidden)?;
        }
        if first {
            let offsets = arena
                .offsets()
                .iter()
                .map(|offset| offset + sampler_bytes)
                .collect();
            self.pools.record(
                StreamId(0),
                bucket,
                offsets,
                sampler_bytes + arena.bytes(),
            )?;
        }
        Ok(())
    }

    /// Appends the bucket's sampler chain to the capture in flight (4.4, 9's `采样`).
    ///
    /// The chain shapes the row's logits by its parameters, keeps its candidates, and draws or
    /// picks one token per sampling row, writing the ids to the plan's first block. It is one
    /// chain per bucket, captured once, so every later replay samples without a host round trip.
    fn append_sampler(
        &mut self,
        _stream: &platform::Stream,
        bucket: Bucket,
        _arena_base: platform::DevicePtr,
        _logits: &ops::TensorView,
    ) -> Result<()> {
        Err(Error::State(format!(
            "the sampler chain of {bucket} needs the batch's sampling parameters and the model's \
             logits view; exec-work M211 asks for the two crossing columns and \
             CausalLm::logits (nn)"
        )))
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

    /// The device-side `TokenBatch` over the bucket's staged columns (9's `每批的规划`).
    fn token_batch(&self, _arena: &platform::DevicePtr, _staged: &Shape) -> Result<nn::TokenBatch> {
        Err(Error::State(
            "the token batch is built once nn::TokenBatch and the pinned upload are published"
                .into(),
        ))
    }

    /// The KV view of the batch: the request slots' block tables and the cache geometry.
    fn kv_view(&self) -> Result<nn::KvView> {
        Err(Error::State(
            "the KV view is built once nn::KvView and the slot tables are published".into(),
        ))
    }

    /// Where the forward's hidden states go for this bucket.
    fn hidden(&self, _base: &platform::DevicePtr, _staged: &Shape) -> Result<ops::TensorView> {
        Err(Error::State(
            "the hidden-state buffer comes from the bucket's static plan".into(),
        ))
    }

}

/// Records the nodes each operator launch adds during a capture (4.5's `Captured.nodes`).
///
/// `ops` reports every launch to the installed sink because the executor never sees an individual
/// `Op::launch` (see `ops::capture`); the sink turns the report into the per-launch node table.
struct Recording<'a> {
    capture: &'a mut platform::Capture,
    nodes: crate::captured::Nodes,
}

impl<'a> Recording<'a> {
    fn new(capture: &'a mut platform::Capture) -> Self {
        Self {
            capture,
            nodes: Vec::new(),
        }
    }
}

impl ops::capture::Sink for Recording<'_> {
    fn launched(&mut self, operator: &'static str, descriptor: &str) -> ops::Result<()> {
        let nodes = self
            .capture
            .record()
            .map_err(ops::Error::Platform)?;
        self.nodes.push((operator, descriptor.to_owned(), nodes));
        Ok(())
    }
}

/// The sampler output of one bucket: one `u32` token id per sampling row.
///
/// The region is the first block of the bucket's plan, so it is allocated at capture and reached at
/// the same address by every replay without allocating during a step (4.5). It is written by the
/// bucket's sampler nodes inside the captured graph, and read back into the runner's host pinned
/// buffer after the replay's stream completed (4.4).
pub struct SamplerBuffer {
    /// Where the graph writes the sampled ids.
    base: platform::DevicePtr,
    /// How many rows the bucket samples for.
    rows: usize,
    /// The host copy one step reads them from; one per runner, never per batch.
    host: platform::PinnedBuffer,
}

impl SamplerBuffer {
    /// The bytes a bucket of `rows` sampling rows needs in its plan.
    pub fn bytes(rows: u32) -> u64 {
        u64::from(rows) * std::mem::size_of::<u32>() as u64
    }

    /// The region at `base` of a bucket whose plan holds it, with `device`'s host staging.
    pub fn new(device: DeviceId, base: platform::DevicePtr, rows: u32) -> Result<Self> {
        let bytes = Self::bytes(rows);
        let host = platform::PinnedBuffer::new(device, bytes as usize)?;
        Ok(Self {
            base,
            rows: rows as usize,
            host,
        })
    }

    /// Where the graph writes the sampled ids.
    pub fn base(&self) -> platform::DevicePtr {
        self.base
    }

    /// The rows the bucket samples for.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Copies the region to the host buffer; the caller has already waited for `stream`.
    pub fn read_back(&self, stream: &platform::Stream) -> Result<()> {
        let bytes = Self::bytes(self.rows as u32);
        // SAFETY: the region is this bucket's sampler output (one u32 per row, inside the pool's
        // mapping) and the host buffer is its own pinned memory of the same size.
        unsafe {
            platform::memcpy_async(self.host.address(), self.base.address(), bytes as usize, stream)?;
        }
        stream.synchronize()
    }

    /// The sampled ids, in the order the graph wrote them.
    fn host_tokens(&self) -> &[u32] {
        // SAFETY: `read_back` filled `rows` u32 values of host-pinned memory; the buffer is aligned
        // by `PinnedBuffer` for any byte copy, and `rows` is the number it was allocated for.
        let bytes = Self::bytes(self.rows as u32) as usize;
        unsafe { std::slice::from_raw_parts(self.host.as_ptr().cast::<u32>(), bytes / 4) }
    }
}

/// Names each sampled row's call: the step's output in the mechanism's order (4.4).
///
/// `sampling_rows` are the batch rows a token is sampled for, in batch order; `calls` name the row
/// ranges of the batch. A sampling row outside every call is an invariant violation: the
/// mechanism's columns and its calls disagree about the batch.
pub fn sampled_output(
    ids: &[u32],
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
            token: ids[index],
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
