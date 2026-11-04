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

use std::sync::Arc;

use ipc::PhaseId;
use nn::CausalLm;
use ops::Generation;
use platform::DeviceId;

use crate::arena::{Pools, StreamId};
use crate::batch::{Batch, BatchColumns, Job, Jobs};
use crate::captured::Captured;
use crate::graph::Graphs;
use crate::plan::{Bucket, Buckets, Shape, Variant};
use crate::runner::{Runner, RunnerId};
use crate::{Error, Result};

/// The host columns of one batch, padded to a bucket and held for the upload.
///
/// The buffers are allocated once, at bind, sized for the largest bucket of each variant, so
/// serving a batch writes into them and allocates nothing.
#[derive(Default, Debug)]
pub struct Staging {
    input_ids: Vec<u32>,
    positions: Vec<u32>,
    slot_mapping: Vec<i64>,
    seq_lens: Vec<u32>,
    query_lens: Vec<u32>,
    write_cache: Vec<u8>,
    logits_rows: Vec<u32>,
    calls: Vec<crate::batch::BatchCall>,
    /// The rows and tokens the padded batch holds.
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
        self.positions = vec![0; largest.tokens as usize];
        self.slot_mapping = vec![-1; largest.tokens as usize];
        self.seq_lens = vec![0; largest.rows as usize];
        self.query_lens = vec![0; largest.rows as usize];
        self.write_cache = vec![0; largest.rows as usize];
        self.logits_rows = vec![0; largest.rows as usize];
        self.calls = Vec::with_capacity(largest.rows as usize);
        self.staged = None;
        Ok(())
    }

    /// Pads the batch's columns to `bucket` and holds them, and answers the shape that is staged.
    ///
    /// The mechanism's values are copied in order; padding is appended in the runner's layout, so
    /// a kernel reading the staged columns sees one well-formed batch of the bucket's size.
    pub fn stage(
        &mut self,
        columns: &BatchColumns,
        shape: &Shape,
        bucket: Bucket,
    ) -> Result<Shape> {
        if !bucket.holds(shape) {
            return Err(Error::Invalid(format!(
                "bucket {bucket} does not hold a batch of {shape}"
            )));
        }
        if bucket.tokens as usize > self.input_ids.len()
            || bucket.rows as usize > self.seq_lens.len()
        {
            return Err(Error::Invariant(format!(
                "the staging buffers are smaller than bucket {bucket}"
            )));
        }
        self.calls.clear();
        self.calls.extend_from_slice(&columns.calls);

        let tokens = columns.tokens().min(bucket.tokens as usize);
        self.input_ids[..tokens].copy_from_slice(&columns.input_ids[..tokens]);
        self.positions[..tokens].copy_from_slice(&columns.positions[..tokens]);
        self.slot_mapping[..tokens].copy_from_slice(&columns.slot_mapping[..tokens]);

        let rows = columns.rows().min(bucket.rows as usize);
        self.seq_lens[..rows].copy_from_slice(&columns.seq_lens[..rows]);
        self.query_lens[..rows].copy_from_slice(&columns.query_lens[..rows]);
        self.write_cache[..rows].copy_from_slice(&columns.write_cache[..rows]);
        self.logits_rows[..rows].copy_from_slice(&columns.logits_rows[..rows]);

        // Padding tokens write nothing: `-1` in `slot_mapping` is what the KV-write kernel reads
        // as "this token does not persist".
        for index in tokens..bucket.tokens as usize {
            self.input_ids[index] = 0;
            self.positions[index] = 0;
            self.slot_mapping[index] = -1;
        }
        for index in rows..bucket.rows as usize {
            self.seq_lens[index] = 1;
            self.query_lens[index] = 1;
            self.write_cache[index] = 0;
            self.logits_rows[index] = 0;
        }

        // One call covering every padding token of a prefill, then empty calls for the remaining
        // rows; a decode's padding rows are rows of slot 0 and need no call of their own.
        if bucket.variant.is_decode() {
            let padding = Bucket::decode(bucket.rows, bucket.variant);
            if padding != bucket {
                return Err(Error::Invariant(
                    "a decode bucket's rows and tokens must agree".into(),
                ));
            }
        } else if bucket.tokens > columns.num_tokens {
            let range = columns.num_tokens..bucket.tokens;
            self.calls.push(crate::batch::BatchCall {
                request: 0,
                kind: crate::batch::CallKind::Prefill,
                cache_group: 0,
                rows: range,
            });
        }
        self.staged = Some(bucket.shape());
        Ok(bucket.shape())
    }

    /// The shape staged by the last [`Staging::stage`].
    pub fn staged(&self) -> Option<Shape> {
        self.staged
    }

    /// The staged tokens, as the upload reads them.
    pub fn input_ids(&self, tokens: usize) -> &[u32] {
        &self.input_ids[..tokens]
    }

    /// The staged positions.
    pub fn positions(&self, tokens: usize) -> &[u32] {
        &self.positions[..tokens]
    }

    /// The staged KV slots.
    pub fn slot_mapping(&self, tokens: usize) -> &[i64] {
        &self.slot_mapping[..tokens]
    }

    /// The staged rows, as the upload reads them.
    pub fn rows(&self) -> StagedRows<'_> {
        StagedRows {
            seq_lens: &self.seq_lens,
            query_lens: &self.query_lens,
            write_cache: &self.write_cache,
            logits_rows: &self.logits_rows,
            calls: &self.calls,
        }
    }
}

/// The per-row view of the staged columns.
pub struct StagedRows<'a> {
    /// Per row: cached prefix plus query length.
    pub seq_lens: &'a [u32],
    /// Per row: query tokens of the row's call.
    pub query_lens: &'a [u32],
    /// Per row: whether the row's tokens persist in the prefix cache.
    pub write_cache: &'a [u8],
    /// Per row: the token row that feeds `logits`.
    pub logits_rows: &'a [u32],
    /// The batch's calls, padding calls included.
    pub calls: &'a [crate::batch::BatchCall],
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
        self.staging.stage(&batch.columns, &shape, bucket)?;
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
        let dropped = self.graphs.purge(generation);
        tracing::debug!(
            dropped,
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

    fn serve(&mut self, job: &Job) -> Result<()> {
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
        // The bucket's arena: the base of its plan in the graph pool. A first capture plans it; a
        // later one replays the plan, so the same allocations land at the same addresses.
        let base = self.pools.base(StreamId(0))?;
        let plan = self.pools.plan(StreamId(0), bucket).cloned().unwrap_or_default();
        let arena = nn::Arena::new(base, plan.bytes());
        let forward = nn::Forward {
            stream: &stream,
            kernels: generation,
            arena: &arena,
            group: None,
        };
        let batch = self.token_batch(&base, &staged)?;
        let cache = self.kv_view()?;
        let hidden = self.hidden(&base, &staged)?;
        self.model.forward(&forward, &batch, &cache, hidden)
    }

    /// Stages a batch of the bucket's full size with nothing to write: what a warmup capture
    /// walks (the same allocations, the same launches as a served batch of that bucket).
    fn stage_warm(&mut self, bucket: Bucket) -> Result<Shape> {
        let tokens = bucket.tokens as usize;
        let rows = bucket.rows as usize;
        let columns = BatchColumns {
            calls: vec![crate::batch::BatchCall {
                request: 0,
                kind: if bucket.variant.is_decode() {
                    crate::batch::CallKind::Decode
                } else {
                    crate::batch::CallKind::Prefill
                },
                cache_group: 0,
                rows: 0..bucket.tokens,
            }],
            input_ids: vec![0; tokens],
            positions: vec![0; tokens],
            slot_mapping: vec![-1; tokens],
            seq_lens: vec![1; rows],
            query_lens: vec![1; rows],
            write_cache: vec![0; rows],
            logits_rows: vec![0; rows],
            num_tokens: bucket.tokens,
            page_size: 1,
        };
        self.staging.stage(&columns, &bucket.shape(), bucket)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A batch is padded to its bucket: no padding token writes the KV cache, and a prefill's
    /// padding tokens belong to one padding call.
    #[test]
    fn padding_never_writes_the_cache() -> Result<()> {
        let buckets = Buckets::new([
            Bucket::decode(4, Variant::DecodeGreedy),
            Bucket::prefill(4, 64, Variant::PrefillOutput),
        ]);
        let mut staging = Staging::new();
        staging.reserve(&buckets)?;

        let decode = BatchColumns {
            input_ids: vec![7, 8],
            positions: vec![3, 4],
            slot_mapping: vec![10, 11],
            seq_lens: vec![4, 9],
            query_lens: vec![1, 1],
            write_cache: vec![1, 1],
            logits_rows: vec![0, 1],
            num_tokens: 2,
            page_size: 16,
            calls: vec![crate::batch::BatchCall {
                request: 0,
                kind: crate::batch::CallKind::Decode,
                cache_group: 0,
                rows: 0..2,
            }],
        };
        let bucket = Bucket::decode(4, Variant::DecodeGreedy);
        let staged = staging.stage(&decode, &Shape::decode(2, Variant::DecodeGreedy), bucket)?;
        assert_eq!(staged.rows, 4);
        assert_eq!(staged.tokens, 4);
        // The padding rows write nothing and persist nothing.
        assert_eq!(&staging.slot_mapping(4)[..], &[10, 11, -1, -1]);
        assert_eq!(&staging.write_cache()[2..], &[0, 0]);

        let prefill = BatchColumns {
            input_ids: vec![1, 2, 3],
            positions: vec![0, 1, 2],
            slot_mapping: vec![0, 1, 2],
            seq_lens: vec![3],
            query_lens: vec![3],
            write_cache: vec![1],
            logits_rows: vec![0],
            num_tokens: 3,
            page_size: 16,
            calls: vec![crate::batch::BatchCall {
                request: 0,
                kind: crate::batch::CallKind::Prefill,
                cache_group: 0,
                rows: 0..3,
            }],
        };
        let bucket = Bucket::prefill(4, 64, Variant::PrefillOutput);
        let staged = staging.stage(
            &prefill,
            &Shape::prefill(1, 3, Variant::PrefillOutput),
            bucket,
        )?;
        assert_eq!(staged.tokens, 64);
        assert_eq!(&staging.slot_mapping(64)[3..], &[-1i64; 61]);
        // One call for the batch and one for every padding token.
        assert_eq!(staging.rows().calls.len(), 2);
        assert_eq!(staging.rows().calls[1].rows, 3..64);
        Ok(())
    }
}

impl Staging {
    /// The staged write flags, for the upload and the padding test.
    pub fn write_cache(&self) -> &[u8] {
        &self.write_cache
    }
}
