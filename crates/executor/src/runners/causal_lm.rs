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

use crate::arena::Pools;
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
        if bucket.tokens as usize > self.input_ids.len() || bucket.rows as usize > self.seq_lens.len()
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
    /// The generation a program-layer update is preparing or has switched to, kept for a revert.
    previous: Option<Arc<Generation>>,
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
    ) -> Result<Self> {
        let mut staging = Staging::new();
        staging.reserve(&buckets)?;
        Ok(Self {
            id,
            phases,
            model,
            generation,
            previous: None,
            buckets,
            graphs: Graphs::new(),
            pools,
            staging,
            device,
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

    fn precopy(&mut self, next: &Arc<Generation>) -> Result<Jobs> {
        // Which buckets a changed operator forces to change, and whether the change can be a
        // rebind (same launch structure) or must be a recapture, is decided from the two
        // generations' kernels and each bucket's node table (4.5).
        let mut changed: Vec<&'static str> = Vec::new();
        for (operator, key, kernel) in self.generation.iter() {
            let same = next
                .kernel(operator, key)
                .map(|candidate| Arc::ptr_eq(candidate, kernel))
                .unwrap_or(false);
            if !same && !changed.contains(&operator) {
                changed.push(operator);
            }
        }
        let mut jobs = Vec::new();
        for operator in changed {
            for bucket in self.graphs.using(operator, self.generation.number()) {
                let captured = self
                    .graphs
                    .captured(bucket, self.generation.number())
                    .ok_or_else(|| Error::Invariant("a bucket vanished from the graph table".into()))?;
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
        self.previous = Some(Arc::clone(&self.generation));
        Ok(Jobs::new(jobs))
    }

    fn switchover(&mut self, generation: u64) {
        // The generation the next batch serves with; `previous` keeps the old one for a revert.
        self.generation = Arc::new(
            self.generation
                .replace(generation, std::iter::empty::<ops::KernelEntry>()),
        );
    }

    fn purge(&mut self, generation: u64) {
        let dropped = self.graphs.purge(generation);
        tracing::debug!(dropped, generation, "purged graphs of a replaced generation");
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
            Job::Replay { bucket, generation, .. } => {
                let captured = self
                    .graphs
                    .captured(*bucket, *generation)
                    .ok_or_else(|| Error::Invariant(format!("no graph of bucket {bucket} in generation {generation}")))?;
                self.replay(captured)
            }
            Job::Capture { bucket, generation } => self.capture(*bucket, *generation),
            Job::Rebind { bucket, generation } => self.rebind(*bucket, *generation),
            Job::Issue { .. } | Job::After { .. } => Err(Error::Invariant(
                "the device queue performs its own jobs".into(),
            )),
        }
    }
}

impl<M: CausalLm> CausalLmRunner<M> {
    /// Replays one captured graph. The device executor's stream is the one that launches it.
    fn replay(&mut self, captured: &Captured) -> Result<()> {
        let stream = self.owned_stream()?;
        captured.launch(stream)
    }

    /// Captures one bucket for one generation: run the bucket's forward eagerly once, then capture
    /// it into the pool's plan, recording every operator launch's nodes (4.5).
    fn capture(&mut self, _bucket: Bucket, _generation: u64) -> Result<()> {
        Err(Error::State(
            "capture needs the model's forward, the arena's plan and the capture recorder".into(),
        ))
    }

    /// Rebinds one bucket from the next generation's precopy walk: capture the forward with the
    /// new kernels on a scratch stream, read each changed launch's parameters from that template,
    /// and set them on the live executable (4.5).
    fn rebind(&mut self, _bucket: Bucket, _generation: u64) -> Result<()> {
        Err(Error::State(
            "rebind needs platform's kernel-node parameter getter".into(),
        ))
    }

    /// The stream the executor issues this runner's work on.
    fn owned_stream(&mut self) -> Result<&platform::Stream> {
        Err(Error::State(
            "the runner's stream is owned by the device executor".into(),
        ))
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
