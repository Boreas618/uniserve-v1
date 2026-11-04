//! The encoder runner: a media computation's token batch, walked once per bucket and replayed (4.5).
//!
//! An encoder takes the packed token rows of a request - a prompt, a vision tower's merged rows, a
//! latent sequence - and writes hidden states. It reads no cache and samples no token, so this
//! runner has none of the causal-LM runner's staging of pages, block tables or slots; what it does
//! share is the shape of the work: one stream, a bucket per declared extent, a captured graph per
//! bucket, and the generation lifecycle the hot update runs through.
//!
//! Its model-declared tables are the one thing it holds that a language model does not: `state`,
//! allocated when the runner loads and viewed at every capture (`runners::tables`).

use std::sync::{Arc, Mutex};

use nn::capability::{Encoder, EncoderInputs, EncoderLayout, EncoderOutputs, EncoderPrompt, TokenBatch};
use ipc::TimingCounters;
use ops::{DType, PhaseKind};
use platform::DeviceId;

use crate::arena::{Pools, StreamId};
use crate::batch::{Batch, BatchOutput, Job, Jobs};
use crate::plan::{Bucket, Buckets, Variant};
use crate::runner::{Binding, Runner, RunnerId};
use crate::runners::columns::{encode_i32, Column};
use crate::runners::lifecycle::{walk_capture, Lifecycle};
use crate::runners::tables::Tables;
use crate::{Error, Result};

/// The figures an encoder's columns are sized by, which the model package states once at open.
#[derive(Clone, Copy, Debug)]
pub struct EncoderShape {
    /// The width of the hidden states the encode writes.
    pub hidden: u32,
    /// The element type of those states.
    pub dtype: DType,
}

/// The columns an encoder stages a batch into, sized for the largest bucket it was bound with.
struct EncoderColumns {
    /// `[tokens]` u32 embedder ids: what the stateless encoder reads instead of a cache slot.
    ids: Column,
    /// `[tokens]` i32 rotary positions.
    positions: Column,
    /// `[rows]` i32 keys visible to each row.
    seq_lens: Column,
    /// `[rows + 1]` i32 packed query offsets.
    query_offsets: Column,
    /// `[rows]` i32 the hidden rows whose logits are wanted: never written, and zeroed so a model
    /// that reads the column anyway reads a defined row rather than uninitialised memory.
    logits_rows: Column,
}

impl EncoderColumns {
    fn new(device: DeviceId, tokens: u32, rows: u32) -> Result<Self> {
        let tokens64 = u64::from(tokens.max(1));
        let rows64 = u64::from(rows.max(1));
        Ok(Self {
            ids: Column::new(device, DType::I32, tokens64)?,
            positions: Column::new(device, DType::I32, tokens64)?,
            seq_lens: Column::new(device, DType::I32, rows64)?,
            query_offsets: Column::new(device, DType::I32, rows64 + 1)?,
            logits_rows: Column::new(device, DType::I32, rows64)?,
        })
    }
}

/// The encoder's runner: one device stream, the buckets its plan captures, and its model.
pub struct EncoderRunner<M: Encoder + Send + Sync> {
    id: RunnerId,
    phases: Vec<ipc::PhaseId>,
    model: Arc<M>,
    lifecycle: Lifecycle,
    buckets: Buckets,
    device: DeviceId,
    stream: platform::Stream,
    shape: EncoderShape,
    pools: Option<Arc<Mutex<Pools>>>,
    counters: Option<Arc<crate::device::SharedCounters>>,
    tables: Tables,
    columns: EncoderColumns,
    /// The host ids of the batch staged last, which `prepare_state` reads when a captured walk is
    /// taken and which no replay reads.
    ids: Vec<u32>,
    /// The bucket the columns are staged for, so a capture smaller than the largest bucket knows how
    /// far the columns it uploads run.
    staged: Option<Bucket>,
}

impl<M: Encoder + Send + Sync> EncoderRunner<M> {
    /// Binds an encoder to one device stream.
    ///
    /// The buckets are the ones the plan states for this runner (`Runner::media_plan`), and the
    /// columns are sized for the largest of them: an encoder captures per bucket and stages into
    /// one set of columns, because the buckets differ only in how many tokens they carry.
    pub fn new(
        binding: Binding,
        model: Arc<M>,
        buckets: Buckets,
        shape: EncoderShape,
        generation: Arc<ops::Generation>,
    ) -> Result<Self> {
        let largest = buckets
            .iter()
            .filter(|bucket| bucket.variant.media())
            .max_by_key(|bucket| bucket.tokens)
            .ok_or_else(|| {
                Error::Invalid(
                    "an encoder runner was bound with no media bucket to capture; its plan must \
                     state the layouts the checkpoint declares"
                        .into(),
                )
            })?;
        let columns = EncoderColumns::new(binding.device, largest.tokens, largest.rows)?;
        let lifecycle =
            Lifecycle::new(binding.id, generation, binding.stream.clone());
        Ok(Self {
            id: binding.id,
            phases: binding.phases,
            model,
            lifecycle,
            buckets,
            device: binding.device,
            stream: binding.stream,
            shape,
            pools: None,
            counters: None,
            tables: Tables::new(),
            columns,
            ids: Vec::new(),
            staged: None,
        })
    }

    /// The buckets this runner captures: the media ones, largest first.
    fn capture_order(&self) -> Vec<Bucket> {
        let mut buckets: Vec<Bucket> = self
            .buckets
            .iter()
            .filter(|bucket| bucket.variant.media())
            .copied()
            .collect();
        buckets.sort_by_key(|bucket| std::cmp::Reverse(bucket.tokens));
        buckets
    }

    /// Allocates and fills the tables of every bucket's layout, once, when the runner loads (4.3).
    ///
    /// The plan sized this memory before admission (`Tables::bytes`), so nothing here decides how
    /// much; a table the model needs and the plan did not size is a refusal rather than a request
    /// path allocation.
    pub fn load(&mut self) -> Result<()> {
        if self.pools.is_none() {
            return Err(Error::Resource(
                "the encoder runner loaded before its device's pools were attached",
            ));
        }
        let buckets = self.capture_order();
        for bucket in buckets {
            self.stage_warm(bucket)?;
            self.fill(bucket)?;
        }
        Ok(())
    }

    /// The bytes `bucket`'s tables need, as the plan books them before admission (4.3).
    ///
    /// The plan calls this once per layout the checkpoint declares, so the resident figure it
    /// reserves is the model's own statement rather than an estimate; a runner that allocated a
    /// table the plan had not sized would be allocating resident memory after admission.
    pub fn table_bytes(model: &M, bucket: Bucket) -> Result<u64> {
        let layout = EncoderLayout {
            rows: bucket.tokens,
            tokens: bucket.tokens,
            images: Vec::new(),
        };
        Tables::plan_bytes(&model.state_tables(&layout))
    }

    fn stream_id(&self) -> StreamId {
        StreamId(self.stream.raw())
    }

    fn pool(&self) -> Result<std::sync::MutexGuard<'_, Pools>> {
        let pools = self.pools.as_ref().ok_or_else(|| {
            Error::Resource("the encoder runner has no graph pools; the device attaches them at start")
        })?;
        Ok(pools.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
    }

    /// The layout of `bucket`'s encode, as the model states it.
    ///
    /// The layout is a function of the bucket alone - an encoder's rows are its tokens and its
    /// images are stated by the mechanism's own `Computation` - so the bucket's extent is the key a
    /// layout's tables are held under, and two requests of one shape share them.
    fn layout(&self, bucket: Bucket) -> EncoderLayout {
        EncoderLayout {
            rows: bucket.tokens,
            tokens: bucket.tokens,
            images: Vec::new(),
        }
    }

    /// The key a layout's tables are held under.
    ///
    /// An encode bucket states no extent (`Computation::extent` is `None` for an encode), so its
    /// shape is what names its layout: two requests of one shape share one table set, and a bucket
    /// of another shape never reads it.
    fn key(bucket: Bucket) -> String {
        format!("encode-{}-{}", bucket.rows, bucket.tokens)
    }

    /// Allocates and fills `bucket`'s layout tables, on the device thread, outside any step.
    fn fill(&mut self, bucket: Bucket) -> Result<()> {
        let layout = self.layout(bucket);
        let tables = self.model.state_tables(&layout);
        if tables.is_empty() {
            return Ok(());
        }
        let key = Self::key(bucket);
        let model = Arc::clone(&self.model);
        let ids = self.ids.clone();
        let prompt = EncoderPrompt { ids: &ids };
        self.tables.fill(
            self.device,
            &self.stream,
            self.lifecycle.generation.number(),
            &key,
            &tables,
            |host| model.prepare_state(&prompt, &layout, host).map_err(Error::from),
        )
    }

    /// The token batch a bucket's graph reads.
    fn token_batch(&self, bucket: Bucket) -> Result<TokenBatch> {
        let tokens = u64::from(bucket.tokens);
        let rows = u64::from(bucket.rows);
        Ok(TokenBatch {
            positions: self.columns.positions.extent(0..tokens),
            // An encoder reads no cache: `-1` is 4.3's statement of that, and a layer that gathers
            // by slot only reads it when it has a cache to gather from.
            slot_mapping: self.columns.logits_rows.extent(0..tokens.max(1)),
            seq_lens: self.columns.seq_lens.extent(0..rows),
            query_offsets: self.columns.query_offsets.extent(0..rows + 1),
            logits_rows: self.columns.logits_rows.extent(0..rows),
            phase: PhaseKind::Prefill,
            page_size: 0,
            num_tokens: bucket.tokens,
            max_kv_len: 0,
            draft_lens: None,
        })
    }

    /// Stages a batch's columns, sized to `bucket`, and records them for the capture.
    fn stage(&mut self, batch: &Batch, bucket: Bucket) -> Result<()> {
        let columns = &batch.columns;
        let threads = &columns.rows;
        let rows = usize::try_from(bucket.rows)
            .map_err(|_| Error::Invalid("a bucket of more rows than memory holds".into()))?;
        if threads.seq_lens.len() != rows {
            return Err(Error::Invalid(format!(
                "a batch of {} row(s) was planned into the bucket of {rows} rows",
                threads.seq_lens.len()
            )));
        }
        // The ids are the vocabulary's own indices: an encoder reads them as integers, and the
        // element type is the one its descriptor names (`DType::I32`), so the column is signed and
        // the cast cannot lose a value a vocabulary can hold.
        self.ids = columns.input_ids.iter().map(|id| *id as u32).collect();
        let mut buffer = Vec::new();

        encode_i32(&mut buffer, &self.ids.iter().map(|id| *id as i32).collect::<Vec<_>>());
        self.columns.ids.write(&buffer)?;
        encode_i32(&mut buffer, &threads.positions);
        self.columns.positions.write(&buffer)?;
        encode_i32(&mut buffer, &threads.seq_lens);
        self.columns.seq_lens.write(&buffer)?;
        encode_i32(&mut buffer, &threads.query_offsets);
        self.columns.query_offsets.write(&buffer)?;
        encode_i32(&mut buffer, &threads.logits_rows);
        self.columns.logits_rows.write(&buffer)?;
        self.staged = Some(bucket);
        // The columns are sized for the largest bucket the runner was bound with; a batch planned
        // into a smaller one writes exactly its own prefix and the graph reads only that far.
        Ok(())
    }

    /// Stages a bucket with no batch behind it, so a first capture has defined columns to read.
    fn stage_warm(&mut self, bucket: Bucket) -> Result<()> {
        if self.staged == Some(bucket) {
            return Ok(());
        }
        let tokens = bucket.tokens as usize;
        let rows = bucket.rows as usize;
        let mut buffer = Vec::new();
        self.ids = vec![0u32; tokens];
        encode_i32(&mut buffer, &vec![0i32; tokens]);
        self.columns.ids.write(&buffer)?;
        encode_i32(&mut buffer, &vec![0i32; tokens]);
        self.columns.positions.write(&buffer)?;
        encode_i32(&mut buffer, &vec![0i32; rows]);
        self.columns.seq_lens.write(&buffer)?;
        encode_i32(&mut buffer, &vec![0i32; rows + 1]);
        self.columns.query_offsets.write(&buffer)?;
        self.columns.logits_rows.write(&buffer)?;
        self.staged = Some(bucket);
        Ok(())
    }

    /// Uploads the staged columns, in one place, so a capture and a replay agree on what the graph
    /// reads.
    fn upload(&self) -> Result<()> {
        self.columns.ids.upload(&self.stream)?;
        self.columns.positions.upload(&self.stream)?;
        self.columns.seq_lens.upload(&self.stream)?;
        self.columns.query_offsets.upload(&self.stream)?;
        self.columns.logits_rows.upload(&self.stream)?;
        Ok(())
    }

    /// Walks the encode once for `bucket`, recording every operator launch's nodes into the
    /// generation's capture (4.5).
    fn walk(&mut self, bucket: Bucket, generation: &Arc<ops::Generation>) -> Result<()> {
        let stream = self.stream.clone();
        if self.staged != Some(bucket) {
            self.stage_warm(bucket)?;
        }
        let first = self.pool()?.begin(self.stream_id(), bucket)?;
        self.upload()?;
        let base = self.pool()?.base(self.stream_id())?;
        let reserved = self.pool()?.reserved();
        let arena = nn::Arena::new(ops::DevicePtr::new(base.address()), reserved);
        let batch = self.token_batch(bucket)?;
        let forward = nn::Forward::new(&stream, generation, &arena);
        let layout = self.layout(bucket);
        let tables = self.model.state_tables(&layout);
        let key = Self::key(bucket);
        if !tables.is_empty() {
            self.fill(bucket)?;
        }
        let state = self.tables.views(
            self.lifecycle.generation.number(),
            &key,
            &tables,
        )?;
        let hidden_view = arena.tensor(
            self.shape.dtype,
            ops::Shape::new(&[u64::from(bucket.tokens), u64::from(self.shape.hidden)]),
        )?;
        let tokens = u64::from(bucket.tokens);
        let inputs = EncoderInputs {
            ids: self.columns.ids.extent(0..tokens),
            batch,
            images: &[],
            layout: &layout,
            state: &state,
            diagnostics: &[],
        };
        self.model
            .encode(&forward, inputs, EncoderOutputs { hidden: hidden_view })
            .map_err(Error::from)?;
        if first {
            self.pool()?.record(self.stream_id(), bucket, arena.used())?;
        }
        Ok(())
    }
}

impl<M: Encoder + Send + Sync> Runner for EncoderRunner<M> {
    fn phases(&self) -> &[ipc::PhaseId] {
        &self.phases
    }

    fn stream(&self) -> Option<&platform::Stream> {
        Some(&self.stream)
    }

    /// Serves one encode: choose its bucket, stage its columns, and return the replay (4.5).
    ///
    /// The bucket is chosen from the computation the batch states, never from how many tokens it
    /// happens to carry: a media graph is captured for one extent, and a batch that a smaller bucket
    /// could hold is still served by the bucket its computation names, because the alternative is a
    /// graph whose capture assumed another canvas.
    fn submit(&mut self, batch: Batch) -> Result<Jobs> {
        let media = batch.media.clone().ok_or_else(|| {
            Error::Invalid(
                "an encoder runner serves a media batch; this one states no computation".into(),
            )
        })?;
        media.check()?;
        let computation = media.computation;
        if computation.variant() != Some(Variant::Encode) {
            return Err(Error::Invalid(format!(
                "an encoder runner serves an encode; this batch states a {:?}",
                computation.call
            )));
        }
        let bucket = self
            .buckets
            .select_media(&computation)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "no bucket was captured for {:?} of {} row(s) and {} token(s)",
                    computation.call, computation.rows, computation.tokens
                ))
            })?;
        self.stage(&batch, bucket)?;
        Ok(Jobs::new([Job::Replay {
            runner: self.id,
            bucket,
            generation: batch.generation,
            batch: batch.id,
        }]))
    }

    fn media_plan(&self) -> Vec<Bucket> {
        self.capture_order()
    }

    fn capture_plan(&self) -> Vec<Bucket> {
        self.capture_order()
    }

    fn cache(&self) -> Option<Arc<Mutex<crate::memory::prefix_cache::PrefixCache>>> {
        // An encoder holds no paged cache: what it reads is the rows it was handed.
        None
    }

    fn warmup(&mut self) -> Result<Jobs> {
        let generation = self.lifecycle.generation.number();
        let mut jobs = Vec::new();
        for bucket in self.capture_order() {
            if self.lifecycle.graphs.captured(bucket, generation).is_none() {
                jobs.push(Job::Capture {
                    runner: self.id,
                    bucket,
                    generation,
                });
            }
        }
        Ok(Jobs::new(jobs))
    }

    fn precopy(&mut self, next: &Arc<ops::Generation>) -> Result<Jobs> {
        self.lifecycle.precopy(next)
    }

    fn switchover(&mut self, generation: u64) -> Result<()> {
        self.lifecycle.switchover(generation)
    }

    fn revert(&mut self, generation: u64) {
        self.lifecycle.revert(generation);
    }

    fn purge(&mut self, generation: u64) {
        self.lifecycle.purge(generation);
        // A generation's tables are charged to resident and retire with it, the same rule as its
        // weights, so the release is here beside the graphs rather than on a second path (4.3).
        self.tables.release(generation);
    }

    fn discard(&mut self, generation: u64) -> Result<()> {
        self.lifecycle.discard(generation)
    }

    fn attach_pools(&mut self, pools: Arc<Mutex<Pools>>) {
        self.tables.attach(Arc::clone(&pools));
        self.lifecycle.attach_pools(pools);
    }

    fn attach_counters(&mut self, counters: Arc<crate::device::SharedCounters>) {
        self.counters = Some(counters);
    }

    fn serve(&mut self, job: &Job) -> Result<BatchOutput> {
        match job {
            Job::Replay {
                bucket, generation, ..
            } => {
                let stream = self.stream.clone();
                self.upload()?;
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
                let launched = std::time::Instant::now();
                captured.launch(&stream)?;
                if let Some(counters) = &self.counters {
                    counters.launches(captured.kernel_nodes());
                }
                // The encode's outputs are read by the denoiser or the decoder through resident
                // products, not through this completion: the stream's order is what makes them
                // readable, and the wait is the executor's on the runner's stream.
                Ok(BatchOutput {
                    calls: Vec::new(),
                    timing: TimingCounters {
                        device_us: launched.elapsed().as_micros() as u64,
                        ..Default::default()
                    },
                })
            }
            Job::Capture {
                bucket, generation, ..
            } => self
                .capture(*bucket, *generation)
                .map(|()| BatchOutput::default()),
            Job::Rebind {
                bucket, generation, ..
            } => self
                .rebind(*bucket, *generation)
                .map(|()| BatchOutput::default()),
            Job::Shadow { operator, key, .. } => Err(Error::Invariant(format!(
                "an encoder runner compares no operator, and the device asked it to shadow \
                 {operator} {key}"
            ))),
        }
    }
}

impl<M: Encoder + Send + Sync> EncoderRunner<M> {
    /// Captures one bucket of `generation`: the walk here, the graph kept by the lifecycle.
    fn capture(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        let walked = walk_capture(&stream, || {
            self.walk(bucket, &kernels)
        })?;
        self.lifecycle.capture(bucket, generation, walked, &kernels)
    }

    /// Prepares one bucket of the next generation from a walk of the same bucket (4.5).
    fn rebind(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        let walked = walk_capture(&stream, || {
            self.walk(bucket, &kernels)
        })?;
        self.lifecycle.rebind(bucket, generation, walked, &kernels)
    }
}
