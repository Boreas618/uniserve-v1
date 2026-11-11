//! The encoder runner: a media computation's token batch, walked once per bucket and replayed (4.5).
//!
//! An encoder takes the packed token rows of a request - a prompt, a vision tower's merged rows, a
//! latent sequence - and writes hidden states. It reads no cache and samples no token, so this
//! runner has none of the causal-LM runner's staging of pages, block tables or slots; what it does
//! share is the shape of the work: one stream, a bucket per declared extent, a captured graph per
//! bucket, and the generation lifecycle the hot update runs through.
//!
//! The values an encode reads beyond its columns (the tower's positions, the prompt rows its images
//! replace) are the model's to compute from the prompt inside `encode`, uploaded through the
//! runner's staging region; before each replay the same encode runs in staging mode with the
//! served prompt ([`crate::runners::lifecycle::Lifecycle::replay`]).
//!
//! **A prompt that carries vision blocks runs eagerly.** A graph is captured for a layout, and a
//! prompt's vision grids are the request's: a captured graph per grid would be one per image size a
//! client can send. So such a prompt is walked once, operator by operator, over its own layout
//! ([`EncoderRunner::encode_vision`]): only a
//! prompt without vision tokens is captured. That eager encode runs over the same stream region the
//! captured ones address, so the recorded walk encodes, beside each bucket's captured encode, every
//! vision prompt the deployment admits that fits the bucket ([`WarmVision`]), through the same
//! function the serving path calls ([`EncoderRunner::encode_images`]).
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
    Encoder, EncoderInputs, EncoderLayout, EncoderOutputs, EncoderPrompt, ImageInput, ImageShape,
    PatchFormat, TokenBatch, WarmPrompt,
};
use ops::{DType, Operand};
use platform::DeviceId;
use proto::module::PhaseKind;

use crate::arena::Pools;
use crate::batch::{Job, Jobs, answered};
use crate::memory::encoders::EncoderHostPool;
use crate::plan::{Bucket, Buckets};
use crate::runner::{Binding, Runner, RunnerId};
use crate::runners::columns::{Column, Region, encode_i32};
use crate::runners::lifecycle::{Lifecycle, RunnerLifecycle};
use crate::runners::products::{self, Product, Products};
use crate::runners::recording::Footprint;
use crate::{Error, Result};
use ipc::execution::Variant;
use ipc::execution::{Batch, BatchOutput, CallId};

/// The figures an encoder's columns are sized by, which the model package states once at open.
#[derive(Clone, Copy, Debug)]
pub struct EncoderShape {
    /// The width of the hidden states the encode writes.
    pub hidden: u32,
    /// The element type of those states.
    pub dtype: DType,
}

/// The prompts with vision blocks a deployment admits, which the recorded walk encodes beside the
/// captured buckets, and the patch rows their blocks read.
pub struct WarmVision {
    /// Each admitted prompt's vision blocks ([`nn::family::VideoModel::warm_prompts`]).
    pub prompts: Vec<WarmPrompt>,
    /// How the media stage writes the patch rows a vision encode reads
    /// ([`Product::VisionPixels`]): their element type and row width.
    pub patches: PatchFormat,
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
    /// A region rather than arena scratch: the states are a product another call of the same
    /// request reads, so the region they live in must be the runner's for as long as it serves
    /// rather than a reservation the captured graph happens to address. One region holds every
    /// request of the largest bucket this runner stages, one request's rows after another.
    ///
    /// **On the device, not a column**, because the stack's residual stream is these rows: every
    /// layer reads and writes them, and in pinned host memory each normalization took 1.7 ms.
    hidden: Region,
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
            hidden: Region::new(device, shape.dtype, hidden)?,
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
pub struct EncoderRunner<M: Encoder + Send + Sync + ?Sized> {
    id: RunnerId,
    phases: Vec<ipc::PhaseId>,
    model: Arc<M>,
    lifecycle: Lifecycle,
    buckets: Buckets,
    device: DeviceId,
    stream: platform::Stream,
    shape: EncoderShape,
    counters: Option<Arc<crate::device::SharedCounters>>,
    columns: EncoderColumns,
    /// The host ids of the batch staged last: the prompt every forward of the batch hands the
    /// model ([`EncoderPrompt`]), the capture's warm one included.
    ids: Vec<u32>,
    /// The bucket the columns are staged for, so a capture smaller than the largest bucket knows how
    /// far the columns it uploads run.
    /// The calls of the batch staged last, which its replay answers.
    calls: Vec<CallId>,
    staged: Option<Bucket>,
    /// The epoch's product store, attached by the worker that starts this runner: what an encode
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
    /// The vision blocks of the prompt staged last, when it carries any: the replay of that batch
    /// walks the encode instead of launching a graph.
    vision: Option<VisionPrompt>,
    /// The vision prompts the recorded walk encodes beside the captured buckets.
    warm: WarmVision,
    /// The tensor group this runner's rank reduces over, absent when one rank holds the encoder.
    group: Option<Arc<nn::Group>>,
    /// The host encoder pool of this runner's device, which a request's encoded states are read
    /// from instead of computed when the engine found them stored, and written to when it reserved
    /// pages for them. `None` on a device that maps none.
    host_pool: Option<Arc<EncoderHostPool>>,
    /// Whether this device writes the rows it computed to the pool: every rank of a tensor group
    /// holds the whole states, so one of them writes them for all.
    writes_outputs: bool,
    /// The host outputs of the batch staged last, by request.
    host_outputs: Vec<(ipc::RequestKey, ipc::HostOutput)>,
}

/// A prompt's vision blocks, as its conditioning states them.
struct VisionPrompt {
    /// The request the patch rows were read for.
    request: ipc::RequestKey,
    /// Each block's `[time, rows, columns]` patch grid, in presentation order.
    images: Vec<ImageShape>,
}

/// The vision blocks a request's conditioning presents, in presentation order.
///
/// Every condition with a conditioner's view is one block, and they are presented in request order
/// (its model's request processing). An audio reference and a `ref2va` keyframe present none.
fn vision_images(conditioning: &[u8]) -> Result<Vec<ImageShape>> {
    let decoded = proto::video::Conditions::decode(conditioning).map_err(|error| {
        Error::Invalid(format!(
            "a media request's conditioning does not decode: {error}"
        ))
    })?;
    Ok(decoded
        .conditions
        .iter()
        .filter_map(|condition| condition.vision.as_ref())
        .map(|vision| ImageShape {
            grid: [vision.grid.t, vision.grid.h, vision.grid.w],
        })
        .collect())
}

impl<M: Encoder + Send + Sync + ?Sized> EncoderRunner<M> {
    /// Binds an encoder to one device stream.
    ///
    /// The buckets are the ones the plan states for this runner (`Runner::media_plan`), and the
    /// columns are sized for the largest of them: an encoder captures per bucket and stages into
    /// one set of columns, because the buckets differ only in how many tokens they carry. `vision`
    /// is what the deployment admits of prompts with vision blocks, which the recorded walk encodes
    /// beside the buckets so the stream's region holds their eager encodes.
    pub fn new(
        binding: Binding,
        model: Arc<M>,
        buckets: Buckets,
        vision: WarmVision,
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
            counters: None,
            columns,
            ids: Vec::new(),
            calls: Vec::new(),
            staged: None,
            products: None,
            pending: Vec::new(),
            vision: None,
            warm: vision,
            group: None,
            host_pool: None,
            writes_outputs: false,
            host_outputs: Vec::new(),
        })
    }

    /// This runner with its device's host encoder pool: a request's encoded states are read from it
    /// when the engine found them stored, and, when `writes` is set, written to it when the engine
    /// reserved pages for them.
    #[must_use]
    pub fn with_host_outputs(mut self, pool: Arc<EncoderHostPool>, writes: bool) -> Self {
        self.host_pool = Some(pool);
        self.writes_outputs = writes;
        self
    }

    /// The host output of `request` in the batch staged last, when it has one.
    fn host_output(&self, request: ipc::RequestKey) -> Option<&ipc::HostOutput> {
        self.host_outputs
            .iter()
            .find(|(key, _)| *key == request)
            .map(|(_, output)| output)
    }

    /// Whether every request of the batch staged last reads its states from the host encoder pool,
    /// so the encode computes nothing.
    fn imports_all(&self) -> bool {
        self.host_pool.is_some()
            && !self.pending.is_empty()
            && self.pending.iter().all(|(request, _, _)| {
                self.host_output(*request)
                    .is_some_and(|output| output.import)
            })
    }

    /// Copies between each pending request's rows of the hidden column and its stored output's pages:
    /// from the pages when `import`, to them otherwise. The copies have completed when this returns,
    /// before the call's result lets the engine release or commit the pages.
    fn copy_host_outputs(&self, import: bool) -> Result<()> {
        let Some(pool) = self.host_pool.as_ref() else {
            return Ok(());
        };
        let element = self.shape.dtype.bytes().ok_or_else(|| {
            Error::Invalid(format!("{} has no whole-byte size", self.shape.dtype))
        })?;
        let width = u64::from(self.shape.hidden.max(1));
        let row_bytes = width * element;
        let mut copied = false;
        for (request, first, rows) in &self.pending {
            let Some(output) = self.host_output(*request) else {
                continue;
            };
            if output.import != import || (!import && !self.writes_outputs) {
                continue;
            }
            if output.pool != pool.pool() || output.row_bytes == 0 {
                return Err(Error::Invalid(format!(
                    "a host output of pool {} with rows of {} bytes reached a runner of pool {}",
                    output.pool,
                    output.row_bytes,
                    pool.pool()
                )));
            }
            // The stored output's rows are the engine's unit, which need not be this encoder's:
            // the request's rows of the hidden column are as many of the stored output's bytes as
            // they fill, from its first stored row on.
            let stored = u64::from(output.rows[1] - output.rows[0]);
            let units = (u64::from(*rows) * row_bytes / output.row_bytes).min(stored);
            let base = self.columns.hidden.device.ptr.address() + u64::from(*first) * row_bytes;
            for copy in crate::memory::encoders::page_copies(
                output.rows[0],
                0..units,
                output.row_bytes,
                pool.page_bytes(),
            ) {
                // An export leaves a page the pool holds already as it is.
                let Some(slot) = output.slots.get(copy.page).copied().flatten() else {
                    continue;
                };
                let page = pool.view(slot)?.address() + copy.offset;
                let row = base + copy.output;
                let (destination, source) = if import { (row, page) } else { (page, row) };
                // SAFETY: the pages stay pinned or reserved until this call's result, and the
                // copies complete before this returns; the hidden column is this runner's, which
                // no other launch writes while it serves this batch.
                unsafe {
                    platform::memcpy_async(destination, source, copy.bytes as usize, &self.stream)
                }?;
                copied = true;
            }
        }
        if copied {
            self.stream.synchronize()?;
        }
        Ok(())
    }

    /// This runner over the tensor `group`: every forward it runs carries the group.
    #[must_use]
    pub fn in_group(mut self, group: Option<Arc<nn::Group>>) -> Self {
        self.group = group;
        self
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

    /// The layout of `bucket`'s encode, as the model states it.
    ///
    /// The layout is a function of the bucket alone - an encoder's rows are its tokens and its
    /// images are stated by the mechanism's own `Computation` - so two requests of one shape share
    /// one graph.
    fn layout(&self, bucket: Bucket) -> EncoderLayout {
        EncoderLayout {
            rows: bucket.tokens,
            tokens: bucket.tokens,
            images: Vec::new(),
        }
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
            // A row attends over its own tokens, and a bucket holds no more than its tokens.
            max_kv_len: bucket.tokens,
            draft_lens: None,
            rope_positions: None,
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
        self.vision = None;
        if let Some(media) = &batch.media {
            for (index, (first, rows)) in spans(&threads.query_offsets)?.into_iter().enumerate() {
                let staged = media.request(index)?;
                self.pending.push((staged.request, first, rows));
                let images = vision_images(&staged.conditioning)?;
                if images.is_empty() {
                    continue;
                }
                // One walk serves one prompt: its layout is that prompt's grids, and a batch of two
                // would lay one prompt's blocks over the other's rows.
                if threads.seq_lens.len() != 1 {
                    return Err(Error::Invalid(format!(
                        "a batch of {} prompts carries vision blocks, and a prompt with vision \
                         blocks is encoded alone",
                        threads.seq_lens.len()
                    )));
                }
                self.vision = Some(VisionPrompt {
                    request: staged.request,
                    images,
                });
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
    /// Only `Text` is published, because this encoder writes one region
    /// (`EncoderOutputs::hidden`): a vision tower's features are spliced into it by the encode that
    /// read them, and a condition's rows are the denoiser's conditioner's to write.
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
                self.lifecycle.stream_id(),
                Some(written),
                &view,
            )?;
        }
        Ok(())
    }

    /// Encodes a prompt that carries vision blocks, operator by operator.
    ///
    /// The layout is the prompt's own - its presented tokens and its blocks' grids. The arena and
    /// the staging region are the stream's, whose bucket extents a graph of this runner addresses:
    /// nothing of a graph is live between its replays, so this walk may reuse them, and the recorded
    /// walk encodes the vision prompts the deployment admits so the regions hold it.
    fn encode_vision(&self, bucket: Bucket, generation: u64, prompt: VisionPrompt) -> Result<()> {
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        let tokens = u32::try_from(self.ids.len())
            .map_err(|_| Error::Invalid("a prompt of more tokens than a u32 counts".into()))?;

        let products = self
            .products
            .as_ref()
            .ok_or_else(|| products::no_store(prompt.request))?;
        let patches = products
            .ordered(prompt.request, Product::VisionPixels, &stream)?
            .ok_or_else(|| products::absent(prompt.request, Product::VisionPixels))?;

        let arena = self.lifecycle.arena()?;
        let group = self.group.clone();
        let staging = self.lifecycle.staging.slot()?;
        let forward = crate::runners::denoiser::grouped(
            nn::Forward::new(&stream, &kernels, &arena).with_staging(staging),
            group.as_deref(),
        )?;
        self.upload()?;
        self.encode_images(&forward, bucket, tokens, &prompt.images, patches)?;
        self.lifecycle.staging.release(&stream)
    }

    /// Encodes one prompt of `tokens` tokens whose vision blocks are `images` on `forward`: the
    /// layout and the model call, which the eager serving path ([`Self::encode_vision`]) and the
    /// recorded walk both run.
    ///
    /// `patches` is the `[rows, width]` patch rows of every block, one block's rows after
    /// another's, in the media stage's element type. The prompt's columns are the staged ones, and
    /// its states land in the hidden column's first rows.
    fn encode_images(
        &self,
        forward: &nn::Forward<'_>,
        bucket: Bucket,
        tokens: u32,
        images: &[ImageShape],
        patches: Operand,
    ) -> Result<()> {
        let layout = EncoderLayout {
            rows: tokens,
            tokens,
            images: images.to_vec(),
        };

        // Each block reads its own run of the patch rows: `time × rows × columns` of them.
        let width = patches.shape.dim(1);
        let mut pixels = Vec::with_capacity(images.len());
        let mut first = 0_u64;
        for image in images {
            let [time, height, columns] = image.grid;
            let rows = u64::from(time) * u64::from(height) * u64::from(columns);
            pixels.push(ImageInput {
                pixels: patches
                    .at(patches.offset_elements(first * width))
                    .reshape(ops::Shape::new(&[rows, width])),
            });
            first += rows;
        }

        // One prompt alone: one row of `tokens` keys, whose bound is the bucket's.
        let rows = u64::from(tokens);
        let hidden_width = u64::from(self.shape.hidden.max(1));
        let batch = TokenBatch {
            positions: self.columns.positions.extent(0..rows),
            slot_mapping: self.columns.logits_rows.extent(0..rows.max(1)),
            seq_lens: self.columns.seq_lens.extent(0..1),
            query_offsets: self.columns.query_offsets.extent(0..2),
            logits_rows: self.columns.logits_rows.extent(0..1),
            phase: PhaseKind::Prefill,
            page_size: 0,
            num_tokens: tokens,
            max_kv_len: bucket.tokens,
            draft_lens: None,
            rope_positions: None,
        };
        let hidden = self
            .columns
            .hidden
            .extent(0..rows * hidden_width)
            .reshape(ops::Shape::new(&[rows, hidden_width]));
        self.model
            .encode(
                forward,
                &EncoderPrompt { ids: &self.ids },
                EncoderInputs {
                    ids: self.columns.ids.extent(0..rows),
                    batch,
                    images: &pixels,
                    layout: &layout,
                },
                EncoderOutputs { hidden },
            )
            .map_err(Error::from)
    }
}

impl<M: Encoder + Send + Sync + ?Sized> EncoderRunner<M> {
    /// Encodes `bucket`'s staged prompts on `forward`: the model call a capture walk records, the
    /// staging walk before each replay repeats with the served prompt, and the recorded walk
    /// records.
    fn encode(&self, forward: &nn::Forward<'_>, bucket: Bucket) -> Result<()> {
        let layout = self.layout(bucket);
        let batch = self.token_batch(bucket)?;
        let tokens = u64::from(bucket.tokens);
        let width = u64::from(self.shape.hidden.max(1));
        // The states are the column's first `tokens × width` elements: a warm capture writes them
        // and so does every replay, at the offset a request's rows begin, which is why the region
        // is the runner's rather than the walk's.
        let hidden = self
            .columns
            .hidden
            .extent(0..tokens * width)
            .reshape(ops::Shape::new(&[tokens, width]));
        let inputs = EncoderInputs {
            ids: self.columns.ids.extent(0..tokens),
            batch,
            images: &[],
            layout: &layout,
        };
        self.model
            .encode(
                forward,
                &EncoderPrompt { ids: &self.ids },
                inputs,
                EncoderOutputs { hidden },
            )
            .map_err(Error::from)
    }
}

impl<M: Encoder + Send + Sync + ?Sized> Runner for EncoderRunner<M> {
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
    fn lifecycle_mut(&mut self) -> Option<&mut Lifecycle> {
        Some(&mut self.lifecycle)
    }

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
        self.host_outputs.clone_from(&batch.host_outputs);
        self.calls.clone_from(&batch.calls);
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

    fn purge(&mut self, generation: u64) {
        self.lifecycle.purge(generation);
    }

    fn attach_products(&mut self, products: Arc<dyn Products>) {
        self.products = Some(products);
    }

    fn attach_pools(&mut self, pools: Arc<Mutex<Pools>>) {
        self.lifecycle.attach_pools(pools);
    }

    fn attach_counters(&mut self, counters: Arc<crate::device::SharedCounters>) {
        self.counters = Some(counters);
    }

    /// The encode every bucket's graph captures, and the eager encode of every admitted vision
    /// prompt the bucket serves.
    ///
    /// A prompt with vision blocks is encoded alone (`stage`), so it rides a bucket of one row,
    /// and only one of at least its vision tokens. It is walked at the bucket's whole token count,
    /// the longest prompt that bucket serves, over placeholder patch rows of the media stage's
    /// element type and width; its eager encode starts at the region's first address, as a walk
    /// of its own does.
    fn record_walk(&mut self) -> Result<Footprint> {
        let mut footprint = Footprint::default();
        let group = self.group.clone();
        let patch_dtype = self.warm.patches.dtype;
        let patch_width = u64::from(self.warm.patches.width());
        for bucket in self.capture_order() {
            footprint.walk(Some(bucket), |walk| {
                self.encode(&walk.forward(group.as_deref())?, bucket)
            })?;

            if bucket.rows != 1 {
                continue;
            }
            for prompt in &self.warm.prompts {
                if prompt.tokens > bucket.tokens {
                    continue;
                }
                let rows: u64 = prompt
                    .images
                    .iter()
                    .map(|image| {
                        image
                            .grid
                            .iter()
                            .map(|extent| u64::from(*extent))
                            .product::<u64>()
                    })
                    .sum();
                footprint.walk(Some(bucket), |walk| {
                    let patches =
                        walk.placeholder(patch_dtype, ops::Shape::new(&[rows, patch_width]))?;
                    self.encode_images(
                        &walk.forward(group.as_deref())?,
                        bucket,
                        bucket.tokens,
                        &prompt.images,
                        patches,
                    )
                })?;
            }
        }
        Ok(footprint)
    }

    fn serve(&mut self, job: &Job) -> Result<BatchOutput> {
        match job {
            Job::Replay {
                bucket, generation, ..
            } => {
                // Every request's states are stored: they are read from the host encoder pool into
                // the rows the encode would have written, and nothing is computed.
                if self.imports_all() {
                    let launched = std::time::Instant::now();
                    self.vision = None;
                    self.copy_host_outputs(true)?;
                    self.publish()?;
                    return Ok(BatchOutput {
                        calls: answered(&mut self.calls),
                        timing: TimingCounters {
                            copy_us: launched.elapsed().as_micros() as u64,
                            ..Default::default()
                        },
                        forward_stats: None,
                    });
                }
                if let Some(prompt) = self.vision.take() {
                    let launched = std::time::Instant::now();
                    self.encode_vision(*bucket, *generation, prompt)?;
                    self.copy_host_outputs(false)?;
                    self.publish()?;
                    return Ok(BatchOutput {
                        calls: answered(&mut self.calls),
                        timing: TimingCounters {
                            device_us: launched.elapsed().as_micros() as u64,
                            ..Default::default()
                        },
                        forward_stats: None,
                    });
                }
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
                // The staging walk runs the encode the graph was captured from over the served
                // prompt, so the values its captured copies read are this prompt's.
                let group = self.group.clone();
                self.lifecycle.replay(captured, |arena, staging| {
                    let forward = crate::runners::denoiser::grouped(
                        nn::Forward::staging(arena, staging),
                        group.as_deref(),
                    )?;
                    self.encode(&forward, *bucket)
                })?;
                if let Some(counters) = &self.counters {
                    counters.launches(captured.kernel_nodes());
                }
                // The encoded states are a product a denoise step of the same request reads, so
                // they are named for it after this launch wrote them: the stream's order is what
                // makes them readable, and the wait is the worker's on the runner's stream. The ones
                // the engine reserved pages for are written there first.
                self.copy_host_outputs(false)?;
                self.publish()?;
                Ok(BatchOutput {
                    calls: answered(&mut self.calls),
                    timing: TimingCounters {
                        device_us: launched.elapsed().as_micros() as u64,
                        ..Default::default()
                    },
                    forward_stats: None,
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
            Job::Join { .. } => Err(Error::Invariant(
                "an encoder runner joins no expert step".into(),
            )),
            Job::Shadow { operator, key, .. } => Err(Error::Invariant(format!(
                "an encoder runner compares no operator, and the device asked it to shadow \
                 {operator} {key}"
            ))),
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

impl<M: Encoder + Send + Sync + ?Sized> RunnerLifecycle for EncoderRunner<M> {
    fn lifecycle(&mut self) -> &mut Lifecycle {
        &mut self.lifecycle
    }

    /// Walks the encode once for `bucket`, recording every operator launch's nodes into the
    /// generation's capture (4.5).
    fn walk(&mut self, bucket: Bucket, generation: &Arc<ops::Generation>) -> Result<()> {
        let stream = self.stream.clone();
        if self.staged != Some(bucket) {
            self.stage_warm(bucket)?;
        }
        let first = self
            .lifecycle
            .pool()?
            .begin(self.lifecycle.stream_id(), bucket)?;
        self.upload()?;
        let arena = self.lifecycle.arena()?;
        let group = self.group.clone();
        let probes = crate::profiling::fences::capturing(&stream);
        let staging = self.lifecycle.staging.captured(&stream)?;
        let forward = crate::runners::denoiser::grouped(
            nn::Forward::new(&stream, generation, &arena)
                .with_staging(staging)
                .with_fence(probes.as_ref().map(|probes| probes as &dyn nn::Fence)),
            group.as_deref(),
        )?;
        self.encode(&forward, bucket)?;
        if first {
            self.lifecycle
                .pool()?
                .record(self.lifecycle.stream_id(), bucket, arena.peak())?;
        }
        Ok(())
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
