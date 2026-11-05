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
//!
//! **It keeps no cache and therefore no state group.** The state group of §8 is what a recurrent
//! model must checkpoint at a boundary it declares, and an encoder computes its hidden states from
//! the rows it was handed and carries nothing across batches: so this runner takes no snapshot,
//! constructs no `Snapshot`, and has no boundary to name. A model that does keep state - a recurrent
//! or hybrid one - states the boundary itself and its runner snapshots there; this is the statement
//! that this class is not that one, written down so a runner that never checkpoints is not mistaken
//! for one that silently forgot to.

use std::sync::{Arc, Mutex};

use ipc::TimingCounters;
use nn::capability::{
    Encoder, EncoderInputs, EncoderLayout, EncoderOutputs, EncoderPrompt, TokenBatch,
};
use ops::{DType, PhaseKind};
use platform::DeviceId;

use crate::arena::{Pools, StreamId};
use crate::batch::{Batch, BatchOutput, Job, Jobs};
use crate::plan::{Bucket, Buckets, Variant};
use crate::runner::{Binding, Runner, RunnerId};
use crate::runners::columns::{Column, encode_i32};
use crate::runners::lifecycle::{Lifecycle, walk_capture};
use crate::runners::products::{Product, Products};
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
    /// `[tokens × hidden]` the encoded states, in the model's own element type.
    ///
    /// A column rather than arena scratch: the states are a product another call of the same
    /// request reads, so the region they live in must be the runner's for as long as it serves
    /// rather than a reservation the captured graph happens to address. One region holds every
    /// request of the largest bucket this runner stages, one request's rows after another.
    hidden: Column,
}

impl EncoderColumns {
    fn new(device: DeviceId, tokens: u32, rows: u32, shape: EncoderShape) -> Result<Self> {
        let tokens64 = u64::from(tokens.max(1));
        let rows64 = u64::from(rows.max(1));
        let hidden = tokens64
            .checked_mul(u64::from(shape.hidden.max(1)))
            .ok_or_else(|| Error::Invalid("an encoder's hidden column overflows".into()))?;
        Ok(Self {
            ids: Column::new(device, DType::I32, tokens64)?,
            positions: Column::new(device, DType::I32, tokens64)?,
            seq_lens: Column::new(device, DType::I32, rows64)?,
            query_offsets: Column::new(device, DType::I32, rows64 + 1)?,
            logits_rows: Column::new(device, DType::I32, rows64)?,
            hidden: Column::new(device, shape.dtype, hidden)?,
        })
    }
}

/// The token span each request of a batch owns, as `(first row, count)`.
///
/// A call's span is its **real** tokens and the mechanism's offsets are their boundaries, so the
/// first row is the offset itself rather than a sum of the rows before it: a batch whose first span
/// does not begin at zero - a padded or offset allocation - keeps the rows the mechanism named,
/// which a running sum would silently shift. A span that runs backwards, or a row count that does
/// not fit, is refused rather than wrapped.
fn spans(offsets: &[i32]) -> Result<Vec<(u32, u32)>> {
    let mut spans = Vec::with_capacity(offsets.len().saturating_sub(1));
    for window in offsets.windows(2) {
        let rows = window[1].checked_sub(window[0]).ok_or_else(|| {
            Error::Invalid(format!(
                "a call's span runs backwards: {} then {}",
                window[0], window[1]
            ))
        })?;
        let first = u32::try_from(window[0]).map_err(|_| {
            Error::Invalid("a request's first token row does not fit a count".into())
        })?;
        let rows = u32::try_from(rows)
            .map_err(|_| Error::Invalid("a request's token rows do not fit a count".into()))?;
        spans.push((first, rows));
    }
    Ok(spans)
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
    /// The epoch's product store, attached by the executor that starts this runner: what an encode
    /// produces and a denoise step of the same request reads.
    products: Option<std::sync::Arc<dyn Products>>,
    /// The requests of the batch staged last: `(request, first token row, count)` in request order.
    ///
    /// A replay publishes what its own launch wrote, and the batch numbers its calls by position,
    /// so the request's name is kept here beside the rows it covers (`Batch::media`'s staged record
    /// is the only place a request is named). The rows are the batch's own offline offsets rather
    /// than a running sum, so the publish reads the same numbers the encode read: a call's span is
    /// its real tokens, and whatever the mechanism put before or after it is not this runner's.
    pending: Vec<(ipc::RequestKey, u32, u32)>,
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
        let columns = EncoderColumns::new(binding.device, largest.tokens, largest.rows, shape)?;
        let lifecycle = Lifecycle::new(binding.id, generation, binding.stream.clone());
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
            products: None,
            pending: Vec::new(),
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
            Error::Resource(
                "the encoder runner has no graph pools; the device attaches them at start",
            )
        })?;
        Ok(pools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner))
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
            |host| {
                model
                    .prepare_state(&prompt, &layout, host)
                    .map_err(Error::from)
            },
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

        encode_i32(
            &mut buffer,
            &self.ids.iter().map(|id| *id as i32).collect::<Vec<_>>(),
        );
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
        // Which request is at which rows: a replay publishes what its own launch wrote, and a batch
        // numbers its calls by position, so the name the store is keyed by is kept here with the
        // token rows that request covers (`Batch::media`'s record is where a request is named).
        self.pending.clear();
        if let Some(media) = &batch.media {
            for (index, (first, rows)) in spans(&threads.query_offsets)?.into_iter().enumerate() {
                self.pending
                    .push((media.request(index)?.request, first, rows));
            }
        }
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
        encode_i32(&mut buffer, &vec![0i32; rows]);
        self.columns.logits_rows.write(&buffer)?;
        self.staged = Some(bucket);
        // A warm capture serves no request, so its launch publishes nothing: a row left here would
        // name a request whose states this capture overwrote.
        self.pending.clear();
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

    /// Publishes the encoded states this replay's launch wrote: one `Text` product per request of
    /// the batch staged last.
    ///
    /// The region is the runner's own hidden **column** - pinned memory the runner owns rather than
    /// a graph pool's arena - so the view stays readable until the next encode writes it, whatever
    /// else the device launches in between. That is why the states are a column and not arena
    /// scratch: a graph pool's buckets share one base (`arena::Pools::record`: a stream's plans are
    /// extents and the addresses repeat), so an arena-backed product would be good only until the
    /// next capture or replay of *any* bucket on the producer's stream.
    ///
    /// **One region, therefore one in-flight encode per runner.** The captured graph writes
    /// `EncoderOutputs::hidden` at the address the capture fixed, so a runner cannot give a second
    /// request its own span: a second encode batch overwrites the rows the first request's product
    /// still points at, silently, because the store's view is a name for the producer's memory and
    /// not a copy of it. Two requests inside **one** batch are fine - their spans are disjoint inside
    /// that write - and a second batch must wait for the first request's products to be forgotten.
    /// That is the admission rule the media path already keeps in one place (a decode bucket refuses a
    /// batch of two clips for the same reason), and it belongs to execution rather than here: this
    /// runner states the region's lifetime, and the scheduler states when a second encode may run.
    ///
    /// Only `Text` is published. The conditioning rows a `ref2va` request lays out are a different
    /// packing of a vision tower's features rather than a slice of this sequence, and this encoder
    /// writes one region (`EncoderOutputs::hidden`); publishing a second product from it would be
    /// the executor inventing the model's layout arithmetic.
    fn publish(&self) -> Result<()> {
        let Some(products) = self.products.as_ref() else {
            // A runner whose store was not attached publishes nothing. The consumer that reads a
            // product refuses by name, so the gap is named where it is felt rather than here.
            return Ok(());
        };
        let width = u64::from(self.shape.hidden.max(1));
        for (request, first, rows) in &self.pending {
            let (first, rows) = (u64::from(*first), u64::from(*rows));
            if rows == 0 {
                continue;
            }
            let view = self
                .columns
                .hidden
                .device
                .at(self.columns.hidden.device.offset_elements(first * width))
                .reshape(ops::Shape::new(&[rows, width]));
            let written = crate::runners::products::written_on(&self.stream)?;
            products.publish(
                *request,
                Product::Text,
                self.device,
                self.stream_id(),
                Some(written),
                &view,
            )?;
        }
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
        let state = self
            .tables
            .views(self.lifecycle.generation.number(), &key, &tables)?;
        let tokens = u64::from(bucket.tokens);
        let width = u64::from(self.shape.hidden.max(1));
        // The states are the column's first `tokens × width` elements: a warm capture writes them
        // and so does every replay, at the offset a request's rows begin, which is why the region
        // is the runner's rather than the walk's.
        let hidden_view = self
            .columns
            .hidden
            .extent(0..tokens * width)
            .reshape(ops::Shape::new(&[tokens, width]));
        let inputs = EncoderInputs {
            ids: self.columns.ids.extent(0..tokens),
            batch,
            images: &[],
            layout: &layout,
            state: &state,
            diagnostics: &[],
        };
        self.model
            .encode(
                &forward,
                inputs,
                EncoderOutputs {
                    hidden: hidden_view,
                },
            )
            .map_err(Error::from)?;
        if first {
            self.pool()?
                .record(self.stream_id(), bucket, arena.used())?;
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
        let bucket = self.buckets.select_media(&computation).ok_or_else(|| {
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

    fn attach_products(&mut self, products: Arc<dyn Products>) {
        self.products = Some(products);
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
                // The encoded states are a product a denoise step of the same request reads, so
                // they are named for it after this launch wrote them: the stream's order is what
                // makes them readable, and the wait is the executor's on the runner's stream.
                self.publish()?;
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
            // A host call is no device work: the media stage performs it, and a device runner
            // handed one was given a job another runner returned.
            Job::Host { call, .. } => Err(Error::Invariant(format!(
                "a runner that serves device work was asked for host call {}",
                call.0
            ))),
        }
    }
}

impl<M: Encoder + Send + Sync> EncoderRunner<M> {
    /// Captures one bucket of `generation`: the walk here, the graph kept by the lifecycle.
    fn capture(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        let walked = walk_capture(&stream, || self.walk(bucket, &kernels))?;
        self.lifecycle.capture(bucket, generation, walked, &kernels)
    }

    /// Prepares one bucket of the next generation from a walk of the same bucket (4.5).
    fn rebind(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        let walked = walk_capture(&stream, || self.walk(bucket, &kernels))?;
        self.lifecycle.rebind(bucket, generation, walked, &kernels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A request's span is the pair of offsets the mechanism wrote, not the sum of the rows before
    /// it: a batch whose first span begins at zero answers the same pair either way, and one that
    /// does not is the case a sum would shift without saying so.
    ///
    /// The publisher reads this pair to place each request's rows in the hidden column, so a shift
    /// here is a product that names another request's tokens - bytes that read as a prompt and are
    /// not one.
    #[test]
    fn a_span_is_the_offsets_and_not_a_running_sum() {
        assert_eq!(
            spans(&[0, 4, 6, 6]).expect("a tiling of the batch's own rows"),
            vec![(0, 4), (4, 2), (6, 0)]
        );
        assert_eq!(
            spans(&[5, 9]).expect("a batch whose first span begins at row five"),
            vec![(5, 4)]
        );
        assert!(
            spans(&[9, 5]).is_err(),
            "a span that runs backwards is refused rather than wrapped"
        );
    }
}
