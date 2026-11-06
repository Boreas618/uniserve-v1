//! The denoiser runner: one denoising step of a batch's requests, per layout, captured and replayed
//! (4.5).
//!
//! A denoising step is one computation over one request's latents: the model reads `Latents` and
//! `Conditioning`, runs the DiT stack once and writes the velocity, which its solver applies to the
//! same latents in place. **The loop over a request's steps is not here.** The mechanism schedules
//! one step per batch (`plan::Variant::Denoise`: "a request's steps are one batch each, never a loop
//! inside a batch"), so this runner serves one step and the trajectory belongs to whoever batches
//! the steps.
//!
//! **What the runner owns and what the model owns.** The schedule is the model's:
//! `Denoiser::schedule(task)` states the steps a request of a task runs, with the sigmas and the step
//! count the solver checks itself against. This runner passes a `Step` through without reading a
//! sigma, computing a delta, or counting a step. The coefficient a step reads reaches the device in
//! `Conditioning::step` - a `[1]` `i64` the runner writes before each replay - and the update is the
//! model's, inside `denoise`, because the trait's contract for that call is that it updates the
//! latents in place.
//!
//! **One arena per bucket, never rewound.** The arena is the graph pool's: mapped whole, never a
//! bucket at a time (4.5). The first walk of a bucket records what it used, so the pool's figure is
//! measured from the model's own launches rather than estimated beside them.
//!
//! **The tables are per layout.** `state_tables` is deterministic for a layout by the trait's own
//! rule, so the store keys them by the bucket's layout name and two requests of one shape share one
//! set. They are filled from the binder's warm request for the bucket - a request whose counts match
//! the layout it was derived from, which is why the bucket and the warm request are one object here.
//!
//! **The seam from the mechanism.** A media batch's columns state, per request, what the mechanism
//! owns: the request's conditioning exactly as the model family planned it, and the step, seed and
//! canvas the request states ([`crate::batch::StagedRequest`], written by the scheduler's media
//! arm). A step also reads two things the mechanism cannot state, because they are device products
//! of the model's own preparation: the request's latent trajectory and its encoded conditioning
//! rows. Those are the model's, and this runner reads them from [`Prepared`], a set no capability
//! prepares yet - so a batch is refused by name at the point where it would need one, rather than
//! served with a step index and no latents. When a residency layer for a request's media products
//! lands, it fills `prepared()` and `stage_request` keeps its body.

use std::sync::{Arc, Mutex};

use ipc::TimingCounters;
use nn::capability::{Conditioning, Denoiser, LatentShape, Latents, Layout, Step, VideoRequest};
use ops::{DType, Shape, TensorView};
use platform::DeviceId;

use crate::arena::{Pools, StreamId};
use crate::batch::{Batch, BatchOutput, Job, Jobs};
use crate::plan::{Bucket, Extent, Variant};
use crate::runner::{Binding, Runner, RunnerId};
use crate::runners::columns::Column;
use crate::runners::diag::Diag;
use crate::runners::floor;
use crate::runners::lifecycle::{Lifecycle, walk_capture, warmed_capture};
use crate::runners::products::{self, Product, Products};
use crate::runners::tables::Tables;
use crate::{Error, Result};

/// The request a bucket is walked with before any request of its layout arrives.
///
/// A capture happens before the first request of a shape, and the model's per-layout tables are
/// written from a request - so the deployment states one request per served task and canvas, and
/// **the bucket is derived from it** (`Denoiser::layout`) rather than stated beside it. A bucket and
/// a warm request that disagreed about a count would put a graph's tables in one shape and its
/// graph in another, which no launch refuses.
#[derive(Clone, Debug)]
pub struct WarmRequest {
    /// The task this bucket serves, and the key its schedule is asked for.
    pub task: nn::VideoTask,
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Frames of the request; the model snaps it to a legal count.
    pub frames: u32,
    /// Frames per second of the output.
    pub fps: u32,
    /// Frame-one anchors.
    pub first_frames: u32,
    /// Last-frame anchors.
    pub last_frames: u32,
    /// Reference images or clips.
    pub references: u32,
    /// Prompt tokens the request presents, vision tokens included.
    pub text_tokens: u32,
    /// The encoded condition media, in request order: anchors, then references.
    pub conditions: Vec<nn::ConditionShape>,
    /// Per-image prompt spans, in request order.
    pub vision_spans: Vec<(u32, u32)>,
}

impl WarmRequest {
    /// The request as the model's capability reads one.
    pub fn view(&self) -> VideoRequest<'_> {
        VideoRequest {
            task: self.task,
            width: self.width,
            height: self.height,
            frames: self.frames,
            fps: self.fps,
            first_frames: self.first_frames,
            last_frames: self.last_frames,
            references: self.references,
            text_tokens: self.text_tokens,
            conditions: &self.conditions,
            vision_spans: &self.vision_spans,
        }
    }
}

/// The figures a denoiser's columns are sized by, which the deployment states once at open.
#[derive(Clone, Copy, Debug)]
pub struct DenoiserShape {
    /// The element type of the **stack**: the packed conditioning rows a step reads and the
    /// checkpoint's own arithmetic. The text encoder writes these, so this is the encoder's type.
    pub stack: DType,
    /// The element type of the **sample**: the trajectory the solver accumulates into, and its
    /// noise. It is the model's own statement rather than this runner's
    /// (`minimax_h3::denoiser::SAMPLE_DTYPE` is FP32) and it is not the stack's: the reference keeps
    /// its trajectory in FP32 and refuses any other type at its entry, because a sample in the
    /// stack's type would round the trajectory once per evaluation.
    ///
    /// **Two types, therefore two sizes.** A latent column sized at the stack's type is half the
    /// bytes the model reads, so a runner that held one type here would refuse every step with
    /// "the video latents are bf16 and a trajectory is f32" and the arena floor would be short by the
    /// trajectory's own size.
    pub sample: DType,
    /// The width of the encoded conditioning rows: the text encoder's hidden width, which a layout
    /// counts rows of and states no width for.
    pub width: u32,
}

/// The elements of one latent's block.
fn numel(latent: &LatentShape) -> Result<u64> {
    let product = u64::from(latent.channels)
        .checked_mul(u64::from(latent.frames))
        .and_then(|value| value.checked_mul(u64::from(latent.height)))
        .and_then(|value| value.checked_mul(u64::from(latent.width)))
        .ok_or_else(|| Error::Invalid("a latent's extent overflows".into()))?;
    Ok(product.max(1))
}

/// The shape of one latent's block, as the model declares it.
fn latent_shape(latent: &LatentShape) -> Shape {
    Shape::new(&[
        u64::from(latent.channels),
        u64::from(latent.frames),
        u64::from(latent.height),
        u64::from(latent.width),
    ])
}

/// One latent block's bytes in `dtype`.
fn latent_bytes(latent: &LatentShape, dtype: DType) -> Result<u64> {
    let width = dtype
        .bytes()
        .ok_or_else(|| Error::Invalid(format!("{dtype} has no whole-byte size")))?;
    numel(latent)?
        .checked_mul(width)
        .ok_or_else(|| Error::Invalid("a latent's bytes overflow".into()))
}

/// The host side of a runner's columns: one flat byte buffer per column, staged whole and uploaded
/// once per batch.
///
/// A column's device address never moves, so a captured graph addresses it directly and every step
/// rewrites its contents instead of allocating (4.5): the staging buffers here are the host side of
/// that one buffer, and the offsets are the requests' own blocks.
#[derive(Default)]
struct Staged {
    video: Vec<u8>,
    audio: Vec<u8>,
    text: Vec<u8>,
    source: Vec<u8>,
    step: Vec<u8>,
}

impl Staged {
    /// Resizes `buffer` to `bytes` and zeroes it, so a warm capture reads defined values.
    fn zero(buffer: &mut Vec<u8>, bytes: u64) -> Result<()> {
        let length = usize::try_from(bytes)
            .map_err(|_| Error::Invalid("a batch column is larger than memory".into()))?;
        buffer.clear();
        buffer.resize(length, 0);
        Ok(())
    }

    /// Writes `bytes` into `buffer` at the block `offset`, refusing a block that does not fit.
    fn place(buffer: &mut [u8], offset: u64, bytes: &[u8], what: &str) -> Result<()> {
        let start = usize::try_from(offset)
            .map_err(|_| Error::Invalid(format!("a {what} block offset is larger than memory")))?;
        let end = start
            .checked_add(bytes.len())
            .ok_or_else(|| Error::Invalid(format!("a {what} block overflows")))?;
        if end > buffer.len() {
            return Err(Error::Invalid(format!(
                "a {what} block of {} byte(s) at offset {start} does not fit a column of {}",
                bytes.len(),
                buffer.len()
            )));
        }
        buffer[start..end].copy_from_slice(bytes);
        Ok(())
    }
}

/// The columns a denoiser stages a batch into, sized for the largest bucket it was bound with.
struct DenoiseColumns {
    /// `[requests x channels x frames x height x width]` video latents, one block per request.
    video: Column,
    /// The audio latents, in the same per-request blocks; one element when a bucket serves a task
    /// with no soundtrack, so the column is never empty.
    audio: Column,
    /// `[requests x (text_rows + condition_rows) x width]` conditioning rows.
    text: Column,
    /// `[requests x (text_rows + condition_rows) x width]` the text encoder's own features.
    ///
    /// A model with a conditioner between the encoder and the denoiser reads these and folds them
    /// into `text` in place ([`nn::Denoiser::prepare_conditioning`]), so both buffers are the same
    /// shape and both are the runner's: the conditioner runs inside a captured forward, and a buffer
    /// it allocated would be a capture that allocates.
    source: Column,
    /// `[1]` `i64`: the step index the coefficients are selected by (`Conditioning::step`).
    step: Column,
    /// The host side of each column.
    staged: Staged,
}

impl DenoiseColumns {
    fn new(device: DeviceId, shape: DenoiserShape, layout: &Layout, rows: u32) -> Result<Self> {
        let rows = u64::from(rows.max(1));
        let video = numel(&layout.video)?;
        let audio = match layout.audio {
            Some(audio) => numel(&audio)?,
            None => 1,
        };
        let width = (u64::from(layout.text_rows) + u64::from(layout.condition_rows)).max(1);
        let text = width * u64::from(shape.width.max(1));
        Ok(Self {
            video: Column::new(device, shape.sample, rows * video)?,
            audio: Column::new(device, shape.sample, rows * audio)?,
            text: Column::new(device, shape.stack, rows * text)?,
            source: Column::new(device, shape.stack, rows * text)?,
            step: Column::new(device, DType::I64, 1)?,
            staged: Staged::default(),
        })
    }

    /// Zeroes every column's host side for `bucket`'s layout, so a warm capture reads defined
    /// values.
    fn warm(&mut self, layout: &Layout, shape: DenoiserShape, rows: u32) -> Result<()> {
        let rows = u64::from(rows.max(1));
        Staged::zero(
            &mut self.staged.video,
            rows * latent_bytes(&layout.video, shape.sample)?,
        )?;
        let audio = match layout.audio {
            Some(audio) => latent_bytes(&audio, shape.sample)?,
            None => shape.sample.bytes().unwrap_or(1),
        };
        Staged::zero(&mut self.staged.audio, rows * audio)?;
        let width = (u64::from(layout.text_rows) + u64::from(layout.condition_rows)).max(1);
        let text = u64::from(shape.width.max(1));
        let bytes = shape
            .stack
            .bytes()
            .ok_or_else(|| Error::Invalid(format!("{} has no whole-byte size", shape.stack)))?;
        Staged::zero(&mut self.staged.text, rows * width * text * bytes)?;
        // The conditioner folds these into `text`, so a warm capture must read defined features as
        // well as defined rows: an uninitialised buffer would make the capture's output depend on
        // whatever the allocator last held.
        Staged::zero(&mut self.staged.source, rows * width * text * bytes)?;
        self.staged.step = vec![0u8; 8];
        Ok(())
    }
}

/// The products one denoising step reads, in the order a refusal names them.
///
/// **This is the list the runner reads from the store, and nothing else.** A step here reads the
/// encoded conditioning rows (`text`) and the encoder's own features (`source`), which
/// `Denoiser::prepare_conditioning` folds into `text` before the first evaluation.
///
/// **The trajectory is not on the list although the model's `Layout` states it**, because this
/// runner *owns* it: the latents are its own columns, which the captured graph updates in place, so
/// asking a store for them would be asking for a second name for memory the runner already holds.
/// The runner is what *publishes* them, as `VideoLatents`/`AudioLatents`, for the decode that reads
/// them next.
///
/// `images` and `audio` are `Conditioning` fields another model fills; this runner passes `None`
/// for both, so requiring them would refuse every step of a model that does not use them - the
/// defect of a list written from the capability's fields rather than from this file's own
/// assembly.
///
/// The request's raw condition media is not here either: it is an input to the conditioning rather
/// than a row of it, it is per condition, and the scheduler names it
/// (`crate::runners::products`).
const STEP_PRODUCTS: &[Product] = &[Product::Text];

/// The products a step of a *conditioned* layout reads: the two above, and the conditioning
/// sequence's encoded states.
///
/// `condition_elements` is the per-request fact that decides between this list and
/// [`STEP_PRODUCTS`]: it counts the elements a request's conditions cover, and a task whose request
/// carries none - a prompt alone, or a first and last frame the latents already hold - has no
/// conditioning sequence for an encoder to encode. Requiring a source product there would refuse a
/// task for something no call can produce, which is the same defect as requiring the raw condition
/// media (`crate::runners::products`).
const CONDITIONED_STEP_PRODUCTS: &[Product] = &[Product::Text, Product::Source];

/// The products a step of `layout` reads, in the order a refusal names them.
///
/// The text rows are the prompt's encoded states and every task reads them; a layout with
/// conditioning elements reads its conditioning rows as well.
fn step_products(layout: &nn::capability::Layout) -> &'static [Product] {
    if layout.condition_elements > 0 {
        CONDITIONED_STEP_PRODUCTS
    } else {
        STEP_PRODUCTS
    }
}

/// The rows and width of one request's conditioning in the columns: `(text_rows + condition_rows)`
/// rows of `width` elements each.
///
/// One statement for the two readers that need it - the view a step hands the model
/// ([`DenoiserRunner::conditioning`]) and the copies that fill it from the store
/// ([`DenoiserRunner::seed`]). Two statements of this arithmetic would be a step that reads a row
/// the copies never filled, which is a plausible clip rather than an error.
fn conditioning_row(shape: &DenoiserShape, layout: &nn::capability::Layout) -> (u64, u64) {
    let rows = (u64::from(layout.text_rows) + u64::from(layout.condition_rows)).max(1);
    let width = u64::from(shape.width.max(1));
    (rows, width)
}

/// How a product's bytes fill one row of a conditioning column: `(copied, zeroed)`.
///
/// A row is the layout's **capacity** - `text_rows + condition_rows` rows of `width` - and a
/// product holds the request's own rows, which the encoder wrote. The prefix's rows past the
/// request's are the padding the packing names, and its value is zeros
/// (`packing::DensePacking::zero_row`). So a short product fills the row's front and the rest is
/// zeroed rather than refused.
///
/// **A product longer than the row is refused**, and by name: it is not padding, it is a plan
/// disagreement - the store holds more of the request's rows than the layout those rows are copied
/// into can carry - and truncating it would drop states the denoiser was told are there.
fn fills(
    request: ipc::RequestKey,
    kind: Product,
    available: u64,
    bytes: u64,
) -> Result<(u64, u64)> {
    if available == 0 {
        return Err(Error::Invalid(format!(
            "request {} presents no byte(s) of {} where a row of {bytes} bytes must be filled: an \
             empty region is a product no call wrote",
            request.request_id.0,
            kind.name()
        )));
    }
    if available > bytes {
        return Err(Error::Invalid(format!(
            "request {} presents {available} byte(s) of {} where a row holds {bytes}: the store's \
             region is longer than the conditioning block it is copied into, so the deployment's \
             text capacity is below the rows the request was encoded with",
            request.request_id.0,
            kind.name()
        )));
    }
    Ok((available, bytes - available))
}

/// The first of [`STEP_PRODUCTS`] that `products` does not hold for `request`, if any.
///
/// A free function so the presence rule is testable without a runner: constructing one takes a
/// device, and the rule is about a store rather than about a device.
fn missing_product(
    products: &dyn Products,
    request: ipc::RequestKey,
    kinds: &[Product],
) -> Result<Option<Product>> {
    // **Presence, not placement.** Whether a product exists is a fact about the store; where a
    // reader wants it is `copy_into`'s question. Asking `borrow` here made a deployment that puts
    // the text encoder on its own GPU refuse every step with "no call produced it".
    for kind in kinds {
        if !products.holds(request, *kind) {
            return Ok(Some(*kind));
        }
    }
    Ok(None)
}

/// One row of the batch a replay serves.
///
/// The request and its row are what the store is seeded and published by; the seed is what the
/// first step's draw is addressed by, and it is kept here because the draw is issued by the
/// replay rather than by the `submit` that read it.
#[derive(Clone, Copy, Debug)]
struct Pending {
    /// The request this row belongs to.
    request: ipc::RequestKey,
    /// The row of the bucket's columns the request occupies.
    index: u32,
    /// The request's noise seed: the draw is addressed by it and the element's index.
    seed: nn::capability::Seed,
}

/// One bucket and the request it was derived from.
#[derive(Clone, Debug)]
pub struct BucketPlan {
    /// The bucket: one step over one request of the warm request's layout.
    pub bucket: Bucket,
    /// The request the bucket's layout, tables and schedule are stated by.
    pub warm: WarmRequest,
}

/// The denoiser's runner: one device stream, the buckets its plan captures, and its model.
pub struct DenoiserRunner<M: Denoiser + Send + Sync> {
    id: RunnerId,
    phases: Vec<ipc::PhaseId>,
    model: Arc<M>,
    lifecycle: Lifecycle,
    /// The buckets this runner captures, each with the warm request it was derived from: a bucket's
    /// task is what its schedule is asked for, and the two are one object so they cannot disagree.
    buckets: Vec<BucketPlan>,
    device: DeviceId,
    stream: platform::Stream,
    shape: DenoiserShape,
    pools: Option<Arc<Mutex<Pools>>>,
    counters: Option<Arc<crate::device::SharedCounters>>,
    tables: Tables,
    columns: DenoiseColumns,
    /// The epoch's product store: what a call of this request produced for this one to read.
    ///
    /// `None` until the executor attaches it, and a step that needs a product without one refuses by
    /// name rather than staging zeros.
    products: Option<Arc<dyn Products>>,
    /// The block-by-block diagnosis, off unless `UNISERVE_DIAG_DIR` names a directory.
    diag: Diag,
    /// The step of the schedule the staged batch is, which names the diagnosis dumps.
    staged_step: Option<u32>,
    /// The rows of the batch being served: what a replay seeds from the store, what it publishes
    /// its trajectory back to, and the seed its first step's draw is addressed by.
    ///
    /// A batch is one step, so this is filled by `submit` and consumed by the replay that follows;
    /// a second `submit` before a replay would be a second batch, which the executor does not do.
    pending: Vec<Pending>,
}

impl<M: Denoiser + Send + Sync> DenoiserRunner<M> {
    /// Binds a denoiser to one device stream.
    ///
    /// The buckets are derived here from the deployment's warm requests, one per task and canvas it
    /// serves: `Denoiser::layout` states the layout, and the bucket is `Bucket::denoise(1, layout)` -
    /// one request per batch, because the model's step is one request's (`Denoiser::denoise`).
    pub fn new(
        binding: Binding,
        model: Arc<M>,
        warm: Vec<WarmRequest>,
        shape: DenoiserShape,
        generation: Arc<ops::Generation>,
    ) -> Result<Self> {
        if warm.is_empty() {
            return Err(Error::Invalid(
                "a denoiser runner was bound with no served task and canvas; its plan must state \
                 one warm request for every layout the checkpoint serves"
                    .into(),
            ));
        }
        let mut buckets: Vec<BucketPlan> = Vec::with_capacity(warm.len());
        for request in warm {
            let layout = model.layout(&request.view());
            let bucket = Bucket::denoise(1, layout);
            if buckets.iter().any(|plan| plan.bucket == bucket) {
                // Two served tasks of one layout share one graph and one table set: the bucket is
                // the layout, so a second entry would capture the same extent twice.
                continue;
            }
            buckets.push(BucketPlan {
                bucket,
                warm: request,
            });
        }
        let largest = buckets
            .iter()
            .max_by_key(|plan| plan.bucket.rows)
            .ok_or_else(|| Error::Invalid("a denoiser runner derived no bucket".into()))?;
        let layout = layout_of(largest.bucket)?;
        let columns = DenoiseColumns::new(binding.device, shape, &layout, largest.bucket.rows)?;
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
            products: None,
            diag: Diag::from_env(),
            staged_step: None,
            pending: Vec::new(),
        })
    }

    /// The buckets this runner captures, largest first.
    fn capture_order(&self) -> Vec<Bucket> {
        let mut buckets: Vec<Bucket> = self.buckets.iter().map(|plan| plan.bucket).collect();
        buckets.sort_by_key(|bucket| std::cmp::Reverse(bucket.rows));
        buckets
    }

    /// The plan entry of `bucket`, or a refusal naming a bucket this runner does not serve.
    fn plan_of(&self, bucket: Bucket) -> Result<&BucketPlan> {
        self.buckets
            .iter()
            .find(|plan| plan.bucket == bucket)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "this denoiser runner serves no bucket {bucket}; its warm requests state {}",
                    self.buckets.len()
                ))
            })
    }

    /// The steps of `bucket`'s schedule, as the model states them for its task.
    ///
    /// The model refuses a task it states no schedule for, and that refusal is passed on rather
    /// than turned into an empty ladder: empty is the same value as a schedule with no steps, which
    /// is the opposite fact.
    pub fn schedule(&self, bucket: Bucket) -> Result<Vec<Step>> {
        let plan = self.plan_of(bucket)?;
        self.model.schedule(plan.warm.task).map_err(Error::from)
    }

    /// Allocates the tables of every bucket's layout, once, when the runner loads (4.3).
    ///
    /// The plan sized this memory before admission (`Tables::plan_bytes`), so nothing here decides
    /// how much; a table the model needs and the plan did not size is a refusal rather than a
    /// request-path allocation.
    pub fn load(&mut self) -> Result<()> {
        if self.pools.is_none() {
            return Err(Error::Resource(
                "the denoiser runner loaded before its device's pools were attached",
            ));
        }
        // **The plan is the floor of the device's share, checked here rather than discovered by a
        // capture that has to be thrown away.** The pool is the runtime's policy (a share of the
        // device per stream); the model states what one evaluation reserves. A share below the plan
        // is a refusal naming both numbers, and it is asked before anything is filled.
        let offered = self.pool()?.reserved();
        let plans = self.buckets.clone();
        for plan in &plans {
            floor::checked(
                plan.bucket,
                self.model.arena_bytes(&plan.warm.view()),
                offered,
            )?;
        }
        for bucket in self.capture_order() {
            self.fill(bucket)?;
        }
        Ok(())
    }

    /// The bytes `warm`'s layout needs for its tables, as the plan books them before admission
    /// (4.3).
    pub fn table_bytes(model: &M, warm: &WarmRequest) -> Result<u64> {
        let layout = model.layout(&warm.view());
        Tables::plan_bytes(&model.state_tables(&layout))
    }

    fn stream_id(&self) -> StreamId {
        StreamId(self.stream.raw())
    }

    fn pool(&self) -> Result<std::sync::MutexGuard<'_, Pools>> {
        let pools = self.pools.as_ref().ok_or_else(|| {
            Error::Resource(
                "the denoiser runner has no graph pools; the device attaches them at start",
            )
        })?;
        Ok(pools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner))
    }

    /// The key a bucket's tables are held under.
    ///
    /// The layout is the key because the layout is the extent (`Bucket::denoise`): two tasks of one
    /// extent share one table set, and a table filled from one task's request describes the other's
    /// shape just as well, which is the trait's "deterministic for a layout" rule.
    fn key(bucket: Bucket) -> String {
        match bucket.extent {
            Extent::Latents(layout) => format!(
                "denoise-{}-{}-{}-{}x{}x{}",
                layout.video.channels,
                layout.video.frames,
                layout.text_rows + layout.condition_rows,
                layout.video.height,
                layout.video.width,
                bucket.rows.max(1),
            ),
            _ => format!("denoise-{}", bucket.rows.max(1)),
        }
    }

    /// Allocates and fills `bucket`'s layout tables, on the device thread, outside any step.
    fn fill(&mut self, bucket: Bucket) -> Result<()> {
        let layout = layout_of(bucket)?;
        let tables = self.model.state_tables(&layout);
        if tables.is_empty() {
            return Ok(());
        }
        let key = Self::key(bucket);
        let model = Arc::clone(&self.model);
        let warm = self.plan_of(bucket)?.warm.clone();
        self.tables.fill(
            self.device,
            &self.stream,
            self.lifecycle.generation.number(),
            &key,
            &tables,
            |host| {
                let request = warm.view();
                model
                    .prepare_state(&request, &layout, host)
                    .map_err(Error::from)
            },
        )
    }

    /// The video latents of request `index` of `bucket`, as a view of the runner's own column.
    fn video_latents(&self, bucket: Bucket, index: u32) -> Result<TensorView> {
        let layout = layout_of(bucket)?;
        let elements = numel(&layout.video)?;
        let view = self.columns.video.shaped(&[elements]);
        Ok(view
            .at(view.offset_elements(u64::from(index) * elements))
            .reshape(latent_shape(&layout.video)))
    }

    /// The audio latents of request `index` of `bucket`, when the bucket's layout has a soundtrack.
    fn audio_latents(&self, bucket: Bucket, index: u32) -> Result<Option<TensorView>> {
        let layout = layout_of(bucket)?;
        let Some(audio) = layout.audio else {
            return Ok(None);
        };
        let elements = numel(&audio)?;
        let view = self.columns.audio.shaped(&[elements]);
        Ok(Some(
            view.at(view.offset_elements(u64::from(index) * elements))
                .reshape(latent_shape(&audio)),
        ))
    }

    /// The latents of request `index` of `bucket`.
    fn latents(&self, bucket: Bucket, index: u32) -> Result<Latents> {
        Ok(Latents {
            video: self.video_latents(bucket, index)?,
            audio: self.audio_latents(bucket, index)?,
        })
    }

    /// The conditioning of request `index` of `bucket`, over the tables of its layout.
    fn conditioning<'a>(
        &self,
        bucket: Bucket,
        index: u32,
        state: &'a [TensorView],
        media: &'a [nn::capability::MediaInput],
    ) -> Result<Conditioning<'a>> {
        let layout = layout_of(bucket)?;
        let (rows, width) = conditioning_row(&self.shape, &layout);
        let view = self.columns.text.shaped(&[rows * width]);
        let text = view
            .at(view.offset_elements(u64::from(index) * rows * width))
            .reshape(Shape::new(&[rows, width]));
        let features = self.columns.source.shaped(&[rows * width]);
        let source = features
            .at(features.offset_elements(u64::from(index) * rows * width))
            .reshape(Shape::new(&[rows, width]));
        Ok(Conditioning {
            text,
            source: Some(source),
            media,
            images: None,
            audio: None,
            state,
            step: Some(self.columns.step.shaped(&[1])),
        })
    }

    /// Copies each pending row's conditioning from the store into its row of the columns.
    ///
    /// **Device to device, on the runner's stream, after `upload()`**: the store holds views of
    /// regions other calls wrote, so a host round-trip would move a request's conditioning rows out
    /// of the device and back on every step. Issued on one stream, the copy is ordered after
    /// `upload()`'s writes and after whatever the producer issued, so the row the graph reads is the
    /// row the store holds.
    ///
    /// A view shorter than the row it must fill is refused by name: a short copy would leave the
    /// tail of the row holding the previous request's values, which is worse than a refusal because
    /// the graph runs either way.
    fn seed(&mut self, bucket: Bucket) -> Result<()> {
        // A replay always follows the `submit` that produced it, and that submit is what names the
        // rows: an empty set means the batch was served already, or a job outlived its submit. The
        // graph would then read the host columns - which are a warm capture's zeros - so this is
        // refused rather than served, because a request whose conditioning is zeros produces a
        // plausible clip and no error.
        if self.pending.is_empty() {
            return Err(Error::Invariant(format!(
                "a replay of bucket {bucket} names no staged row: its batch was served already, or the job \
                 outlived the submit that produced it, and the columns hold a warm capture's zeros \
                 rather than a request's conditioning"
            )));
        }
        let Some(products) = self.products.clone() else {
            return Ok(());
        };
        // The rows and the stream are copied out before the columns are borrowed mutably: the
        // padding below writes a column's host side, which is the memory its graph reads.
        let pending = self.pending.clone();
        let stream = self.stream.clone();
        let layout = layout_of(bucket)?;
        let (rows, width) = conditioning_row(&self.shape, &layout);
        let element = u64::from(self.shape.stack.bytes().ok_or_else(|| {
            Error::Invalid(format!("{} has no whole-byte size", self.shape.stack))
        })?);
        let bytes = rows * width * element;
        for row in &pending {
            let (request, index) = (row.request, row.index);
            for (kind, column) in [
                (Product::Text, &mut self.columns.text),
                (Product::Source, &mut self.columns.source),
            ] {
                // **The copy is the store's, not this runner's.** A deployment can put the text
                // phase on its own GPU, so the product may be on another device or another stream,
                // and only the store holds both ends: it waits on the producer's write before this
                // stream copies. A runner issuing its own `memcpy_async` from another device's
                // region would be reading bytes no ordering protects.
                let Some(published) = products.product(request, kind)? else {
                    return Err(products::absent(request, kind));
                };
                let (copied, zeroed) = fills(request, kind, published.bytes, bytes)?;
                let destination = TensorView::new(
                    ops::DevicePtr::new(column.host.address() + u64::from(index) * bytes),
                    self.shape.stack,
                    Shape::new(&[copied / u64::from(element)]),
                );
                if !products.copy_into(request, kind, &destination, &stream)? {
                    return Err(products::absent(request, kind));
                }
                // The rest of the row is the packing's zero row. A layout's prefix is the
                // **deployment's capacity** (`DensePacking::text_rows` - "a layout holds every
                // prompt that fits it") while the encoder publishes the request's own rows, so the
                // tail is the padding the packing names and its value is zeros
                // (`DensePacking::zero_row`). It is written here, on the host side of the column,
                // because that is the memory the captured graph reads: a column's device view is
                // its pinned buffer's address (`columns.rs`), so these bytes are the graph's.
                if zeroed > 0 {
                    let start = usize::try_from(u64::from(index) * bytes + copied)
                        .map_err(|_| Error::Invalid("a row is larger than memory".into()))?;
                    let end = usize::try_from(u64::from(index) * bytes + bytes)
                        .map_err(|_| Error::Invalid("a row is larger than memory".into()))?;
                    // SAFETY: `start..end` is inside this column's row `index`, which the column
                    // was sized for, and no copy or launch writes it concurrently: the front copy
                    // is issued on the stream before the launch and covers `..start` alone.
                    unsafe {
                        column.host.as_mut_slice()[start..end].fill(0);
                    }
                }
            }
        }
        Ok(())
    }

    /// Issues each pending row's first step: the model's own draw and fold, on the replay's arena
    /// and stream, before the captured graph launches.
    ///
    /// A trajectory's value at step zero is not a step: it is `Denoiser::draw_noise` (Philox
    /// addressed by the request's seed and the element's index, section 9) followed by
    /// `Denoiser::prepare_latents` (the schedule's start over that draw), and both take a
    /// `Forward` because their operators reserve scratch in an arena.
    ///
    /// The replay's arena is the one its captured graph addresses: `Pool::base` is the stream's
    /// range and `Pool::begin` answers only whether a bucket is planning or replaying, so the same
    /// range serves a replay. That is what makes the draw affordable here - its reservations are
    /// the walk's own, and the graph's nodes address their recorded offsets above a column the draw
    /// writes rather than above scratch it overwrites. A later step needs none of this: its
    /// trajectory is the column the previous replay advanced in place, which is why only step zero
    /// draws.
    fn first_step(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        if self.staged_step != Some(0) || self.pending.is_empty() {
            return Ok(());
        }
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        self.fill(bucket)?;
        let base = self.pool()?.base(self.stream_id())?;
        let reserved = self.pool()?.reserved();
        let arena = nn::Arena::new(ops::DevicePtr::new(base.address()), reserved);
        let forward = nn::Forward::new(&stream, &kernels, &arena);
        let layout = layout_of(bucket)?;
        let tables = self.model.state_tables(&layout);
        let key = Self::key(bucket);
        let state = self
            .tables
            .views(self.lifecycle.generation.number(), &key, &tables)?;
        // A denoising step reads no condition media: the media reader's views belong to a
        // conditioner's forward, and this is a denoiser's first step.
        let media: [nn::capability::MediaInput; 0] = [];
        for row in &self.pending {
            let latents = self.latents(bucket, row.index)?;
            let cond = self.conditioning(bucket, row.index, &state, &media)?;
            self.model.draw_noise(&forward, row.seed, layout, latents)?;
            self.model.prepare_latents(&forward, cond, latents)?;
        }
        Ok(())
    }

    /// Publishes each pending row's trajectory as this runner's columns hold it.
    ///
    /// **No copy**: the captured graph updates the latents in place, so the column *is* the
    /// trajectory and what the store records is a view of it. That is the handoff to a decode, and
    /// it is why the store's `VideoLatents` is a name for memory rather than a second buffer.
    ///
    /// **The same lifetime rule as the encoder's text states**: a second batch's step zero gives its
    /// own request the same row of this column (`prepare_latents` writes it), so a second media batch
    /// on this runner before the first request's trajectory is forgotten replaces the bytes the first
    /// request's product still names - silently, because the view is a name and not a copy. One
    /// in-flight batch per runner, and the rule is execution's to keep (an encode and a denoise are
    /// the same shape: a captured graph addresses one output region per row).
    fn publish(&self, bucket: Bucket) -> Result<()> {
        let Some(products) = self.products.as_ref() else {
            return Ok(());
        };
        for row in &self.pending {
            let request = row.request;
            let latents = self.latents(bucket, row.index)?;
            let written = products::written_on(&self.stream)?;
            products.publish(
                request,
                Product::VideoLatents,
                self.device,
                self.stream_id(),
                Some(written),
                &latents.video,
            )?;
            if let Some(audio) = latents.audio {
                let written = products::written_on(&self.stream)?;
                products.publish(
                    request,
                    Product::AudioLatents,
                    self.device,
                    self.stream_id(),
                    Some(written),
                    &audio,
                )?;
            }
        }
        Ok(())
    }

    /// Refuses a step whose request has no store, or whose store lacks one of the kinds it reads.
    ///
    /// The kinds are [`STEP_PRODUCTS`], and they are what this file places into a `Conditioning`.
    /// A store that answers `None` for one of them is refused by name - which is the difference
    /// between a step that fails because nobody ran the encoder and one that runs on zeros.
    fn require_products(&self, request: ipc::RequestKey, bucket: Bucket) -> Result<()> {
        let Some(products) = self.products.as_ref() else {
            return Err(products::no_store(request));
        };
        let kinds = step_products(&layout_of(bucket)?);
        match missing_product(products.as_ref(), request, kinds)? {
            Some(kind) => Err(products::absent(request, kind)),
            None => Ok(()),
        }
    }

    /// Writes the coefficient index the step reads, before a walk or a replay, and records it for
    /// the diagnosis.
    ///
    /// Little-endian is the **device's** convention for the `[1]` `i64` table a kernel reads, not the
    /// host's: the column is device memory a launch addresses, so the bytes are the device's byte
    /// order and every GPU this tree runs on is little-endian. A big-endian device would need this
    /// line changed, which is why it says whose order it is rather than leaving it to be assumed.
    fn stage_step(&mut self, step: &Step) -> Result<()> {
        self.columns.staged.step = step.index.to_le_bytes().to_vec();
        self.staged_step = Some(step.index);
        Ok(())
    }

    /// The refusal for a bucket whose task the model states no schedule for.
    ///
    /// It names the chain a reader needs - the bucket, the task the bucket's schedule was asked for,
    /// and the fact that the answer had no step - because the model's own refusal names the task and
    /// this one is reached through the bucket.
    fn no_step(&self, bucket: Bucket) -> Error {
        let task = self
            .plan_of(bucket)
            .map(|plan| format!("{:?}", plan.warm.task))
            .unwrap_or_else(|_| "no task".to_owned());
        Error::Invalid(format!(
            "the schedule of bucket {bucket} states no step to capture; the bucket asked for the \
             {task} schedule"
        ))
    }

    /// Writes the latents of the staged step under the block-by-block diagnosis' own names.
    ///
    /// The protocol's names are per evaluation: the state on entry to evaluation `e` is what both
    /// implementations dump, so a replay writes `denoiser/latents_video_e` before it launches and
    /// `latents_video_{e+1}` after it - the runner's step updates the latents in place, so the exit
    /// state of one evaluation is the entry state of the next and the two trees hold the same names.
    /// Evaluation zero's entry state is the request's noise, so it is written under that name too.
    ///
    /// A dump never fails a request (design 10): `Diag::write` reports and the step goes on.
    fn dump_latents(&self, bucket: Bucket, index: u32) {
        if !self.diag.enabled() {
            return;
        }
        let Ok(video) = self.video_latents(bucket, 0) else {
            return;
        };
        self.write_latent("video", video, index);
        if let Ok(Some(audio)) = self.audio_latents(bucket, 0) {
            self.write_latent("audio", audio, index);
        }
    }

    /// One modality's latents, at the step's own names.
    fn write_latent(&self, modality: &str, view: TensorView, index: u32) {
        let _ = self.diag.write(
            self.device,
            &self.stream,
            "denoiser",
            &format!("latents_{modality}_{index}"),
            view,
        );
        if index == 0 {
            // Evaluation zero's entry state is the request's noise, and the reference dumps it under
            // that name; the two names hold one tensor rather than two computations.
            let _ = self.diag.write(
                self.device,
                &self.stream,
                "denoiser",
                &format!("noise_{modality}"),
                view,
            );
        }
    }

    /// Stages a bucket with no batch behind it, so a first capture has defined columns to read.
    fn stage_warm(&mut self, bucket: Bucket) -> Result<()> {
        let layout = layout_of(bucket)?;
        self.columns.warm(&layout, self.shape, bucket.rows)?;
        let step = self
            .schedule(bucket)?
            .first()
            .copied()
            .ok_or_else(|| self.no_step(bucket))?;
        self.stage_step(&step)?;
        Ok(())
    }

    /// Copies the staged host sides to the device, **every column**, for a walk.
    ///
    /// A capture has no trajectory and must still read defined values, so it stages zeros into the
    /// latent columns and uploads them here. A replay does not: see [`Self::upload_step`].
    fn upload(&mut self) -> Result<()> {
        let stream = self.stream.clone();
        // The host side and the device column are two fields of one value, so the copy borrows
        // them apart rather than cloning a buffer the graph is about to read anyway.
        let DenoiseColumns {
            video,
            audio,
            text,
            source,
            step,
            staged,
        } = &mut self.columns;
        video.write(&staged.video)?;
        audio.write(&staged.audio)?;
        text.write(&staged.text)?;
        source.write(&staged.source)?;
        step.write(&staged.step)?;
        self.columns.video.upload(&stream)?;
        self.columns.audio.upload(&stream)?;
        self.columns.text.upload(&stream)?;
        self.columns.source.upload(&stream)?;
        self.columns.step.upload(&stream)?;
        Ok(())
    }

    /// Copies the columns a **replay** writes from the host: the conditioning rows and the step's
    /// index.
    ///
    /// **The trajectory is not one of them.** The captured graph advances the latents *in place* —
    /// that is the whole of a diffusion step — while the staged buffer holds a warm capture's zeros,
    /// so uploading it before every replay would restart the trajectory at zero whenever a step
    /// began. Nothing refused: step one would advance zeros, and the clip would be the draw plus one
    /// step rather than the schedule. The draw at step zero writes the whole block itself
    /// (`first_step`), so the column never needs this path for its initial values.
    fn upload_step(&mut self) -> Result<()> {
        let stream = self.stream.clone();
        let DenoiseColumns {
            text,
            source,
            step,
            staged,
            ..
        } = &mut self.columns;
        text.write(&staged.text)?;
        source.write(&staged.source)?;
        step.write(&staged.step)?;
        self.columns.text.upload(&stream)?;
        self.columns.source.upload(&stream)?;
        self.columns.step.upload(&stream)?;
        Ok(())
    }

    /// Walks one step of `bucket`'s first request, recording every operator launch into the
    /// generation's capture (4.5).
    ///
    /// One request, because the model's step is one request's: `Bucket::denoise` carries the layout
    /// and this runner walks request zero of it, the rest of the bucket being the same graph over
    /// another block of the columns.
    fn walk(&mut self, bucket: Bucket, generation: &Arc<ops::Generation>) -> Result<()> {
        let stream = self.stream.clone();
        self.stage_warm(bucket)?;
        self.fill(bucket)?;
        let first = self.pool()?.begin(self.stream_id(), bucket)?;
        self.upload()?;
        let base = self.pool()?.base(self.stream_id())?;
        let reserved = self.pool()?.reserved();
        let arena = nn::Arena::new(ops::DevicePtr::new(base.address()), reserved);
        let forward = nn::Forward::new(&stream, generation, &arena);
        let layout = layout_of(bucket)?;
        let tables = self.model.state_tables(&layout);
        let key = Self::key(bucket);
        let state = self
            .tables
            .views(self.lifecycle.generation.number(), &key, &tables)?;
        let step = self
            .schedule(bucket)?
            .first()
            .copied()
            .ok_or_else(|| self.no_step(bucket))?;
        let media: [nn::capability::MediaInput; 0] = [];
        let latents = self.latents(bucket, 0)?;
        let cond = self.conditioning(bucket, 0, &state, &media)?;
        self.model
            .denoise(&forward, step, latents, cond)
            .map_err(Error::from)?;
        if first {
            self.pool()?
                .record(self.stream_id(), bucket, arena.used())?;
        }
        Ok(())
    }
}

impl<M: Denoiser + Send + Sync> Runner for DenoiserRunner<M> {
    fn phases(&self) -> &[ipc::PhaseId] {
        &self.phases
    }

    fn stream(&self) -> Option<&platform::Stream> {
        Some(&self.stream)
    }

    /// Serves one denoising step: choose its bucket, stage its requests, and return the replay (4.5).
    ///
    /// The bucket is chosen from the computation the batch states - the layout is the extent - and
    /// never from how many requests it happens to carry: a graph is captured for one layout, and a
    /// batch of another canvas must not be served by it.
    fn submit(&mut self, batch: Batch) -> Result<Jobs> {
        let media = batch.media.clone().ok_or_else(|| {
            Error::Invalid(
                "a denoiser runner serves a media batch; this one states no computation".into(),
            )
        })?;
        media.check()?;
        let computation = media.computation;
        if computation.variant() != Some(Variant::Denoise) {
            return Err(Error::Invalid(format!(
                "a denoiser runner serves a denoise; this batch states a {:?}",
                computation.call
            )));
        }
        let bucket = self
            .buckets
            .iter()
            .map(|plan| plan.bucket)
            .find(|bucket| bucket.extent == computation.extent && bucket.rows == computation.rows)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "no bucket was captured for {} request(s) of this layout; this runner serves \
                     {} layout(s)",
                    computation.rows,
                    self.buckets.len()
                ))
            })?;
        if media.inputs.len() != computation.rows as usize {
            return Err(Error::Invalid(format!(
                "a denoise batch of {} request(s) carries {} row(s) of staged inputs",
                computation.rows,
                media.inputs.len()
            )));
        }
        let steps = self.schedule(bucket)?;
        // A batch is one step of its requests, and the replay that follows serves exactly this one:
        // a row left from an earlier batch would be seeded and published again.
        self.pending.clear();
        let mut step = None;
        for (index, _row) in media.inputs.iter().enumerate() {
            // The mechanism states the request's parameters: which step this is, and the seed and
            // canvas it runs at. `StagedRequest` is that record and this is the one reader of it.
            let stated_request = media.request(index)?;
            let index = u32::try_from(index)
                .map_err(|_| Error::Invalid("a batch of more requests than rows".into()))?;
            // What a step also reads is produced by other calls of the same request - the text
            // encoder's rows, the trajectory an earlier step advanced - so it is asked for by name
            // and never derived here.
            self.require_products(stated_request.request, bucket)?;
            // A request's first step has no trajectory yet: its values are the model's own draw
            // and fold, which the replay issues before its graph launches (`first_step`). The seed
            // travels with the row, because the draw is addressed by it and by the element's index.
            self.pending.push(Pending {
                request: stated_request.request,
                index,
                seed: nn::capability::Seed(stated_request.seed),
            });
            // Every row of one batch is the same step of the same schedule: a batch is one step
            // (`Variant::Denoise`), so a row that names another step describes another batch.
            let stated = steps
                .get(stated_request.step as usize)
                .copied()
                .ok_or_else(|| {
                    Error::Invalid(format!(
                        "a request of bucket {bucket} states step {} and the schedule has {}",
                        stated_request.step,
                        steps.len()
                    ))
                })?;
            match step {
                None => step = Some(stated),
                Some(first) if first.index == stated.index => {}
                Some(first) => {
                    return Err(Error::Invalid(format!(
                        "a batch carries two requests of different steps ({} and {}); a batch is \
                         one step",
                        first.index, stated.index
                    )));
                }
            }
        }
        let step = step.ok_or_else(|| Error::Invalid("a denoise batch of no request".into()))?;
        self.stage_step(&step)?;
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
        // A denoiser reads no paged cache: its latents are the request's, staged by its own columns.
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
        self.lifecycle.attach_pools(Arc::clone(&pools));
        self.pools = Some(pools);
    }

    fn attach_counters(&mut self, counters: Arc<crate::device::SharedCounters>) {
        self.counters = Some(counters);
    }

    fn attach_products(&mut self, products: Arc<dyn Products>) {
        self.products = Some(products);
    }

    fn serve(&mut self, job: &Job) -> Result<BatchOutput> {
        match job {
            Job::Replay {
                bucket, generation, ..
            } => {
                let stream = self.stream.clone();
                // What a replay writes from the host: the conditioning rows and the step index. The
                // trajectory is the column's own, advanced in place by the previous step.
                self.upload_step()?;
                // The conditioning is the store's, and it overwrites the rows the upload just wrote:
                // that order is why one upload covers every column it writes.
                self.seed(*bucket)?;
                // The first step's trajectory is drawn and folded here, on the arena the graph
                // below addresses, so the graph advances a trajectory rather than a warm capture's
                // zeros (4.5, section 9).
                self.first_step(*bucket, *generation)?;
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
                let step = self.staged_step;
                if let Some(step) = step {
                    self.dump_latents(*bucket, step);
                }
                let launched = std::time::Instant::now();
                captured.launch(&stream)?;
                // The graph advanced the trajectory in place; what a decode reads next is this
                // column, and the store records that view rather than a copy of it.
                self.publish(*bucket)?;
                self.pending.clear();
                if let Some(step) = step {
                    // The step updates the latents in place, so the exit state of this evaluation is
                    // the entry state of the next one, which is the name both trees dump.
                    self.dump_latents(*bucket, step + 1);
                }
                if let Some(counters) = &self.counters {
                    counters.launches(captured.kernel_nodes());
                }
                // The step's latents are the columns the graph updated in place, and the mechanism
                // reads them through its own products: the stream's order is what makes them
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
                "a denoiser runner compares no operator, and the device asked it to shadow \
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

impl<M: Denoiser + Send + Sync> DenoiserRunner<M> {
    /// Captures one bucket of `generation`: the walk here, the graph kept by the lifecycle.
    fn capture(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        let walked = warmed_capture(&stream, || self.walk(bucket, &kernels))?;
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

/// The layout a denoising bucket is captured for.
fn layout_of(bucket: Bucket) -> Result<Layout> {
    match bucket.extent {
        Extent::Latents(layout) => Ok(layout),
        other => Err(Error::Invalid(format!(
            "a denoiser runner serves a denoising bucket; Extent {other:?} is not one"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nn::capability::{Seed, StateTable};

    /// A layout whose video latent is `channels x frames x height x width`.
    fn layout(channels: u32, frames: u32, height: u32, width: u32, text_rows: u32) -> Layout {
        Layout {
            video: LatentShape {
                channels,
                frames,
                height,
                width,
            },
            audio: None,
            text_rows,
            condition_rows: 0,
            condition_elements: 0,
        }
    }

    /// A layout survives the bucket that carries it and the reader that unwraps it.
    ///
    /// The equality is the seam's own: a runner hands the model the layout its bucket carries, and
    /// every bucket was built from a layout the model stated, so anything but an identity through
    /// `Bucket::denoise` and `layout_of` is a transposition the model can never see - it receives
    /// one layout in place of another and computes correctly over the wrong one.
    ///
    /// **A square latent cannot show it**, which is why this case carries paint on four distinct
    /// numbers: the same reason a decoder's rectangle is only caught by a canvas whose extents
    /// differ. The transposed layout is asserted different too, so a change that made every layout
    /// compare equal could not pass this by being vacuous.
    #[test]
    fn a_layout_survives_its_bucket_and_back() {
        let stated = Layout {
            video: LatentShape {
                channels: 16,
                frames: 2,
                height: 3,
                width: 5,
            },
            audio: Some(LatentShape {
                channels: 8,
                frames: 3,
                height: 2,
                width: 4,
            }),
            text_rows: 7,
            condition_rows: 2,
            condition_elements: 6,
        };
        let bucket = Bucket::denoise(1, stated);
        assert_eq!(
            layout_of(bucket).expect("a denoising bucket names the layout it serves"),
            stated,
            "the layout the model stated is the layout the runner hands it back"
        );
        let transposed = Layout {
            video: LatentShape {
                channels: 16,
                frames: 2,
                height: 5,
                width: 3,
            },
            ..stated
        };
        assert_ne!(stated, transposed, "the two video extents differ");
        assert_ne!(
            layout_of(Bucket::denoise(1, transposed)).expect("a denoising bucket names its layout"),
            stated,
            "a transposed canvas is not the canvas that was stated"
        );
    }

    #[test]
    fn a_key_names_the_layout_so_two_extents_never_share_tables() {
        let first = Bucket::denoise(1, layout(16, 2, 3, 4, 8));
        let other = Bucket::denoise(1, layout(16, 2, 3, 4, 9));
        assert_ne!(
            DenoiserRunner::<NoModel>::key(first),
            DenoiserRunner::<NoModel>::key(other)
        );
    }

    #[test]
    fn two_requests_of_one_layout_share_a_key() {
        assert_eq!(
            DenoiserRunner::<NoModel>::key(Bucket::denoise(1, layout(16, 2, 3, 4, 8))),
            DenoiserRunner::<NoModel>::key(Bucket::denoise(1, layout(16, 2, 3, 4, 8)))
        );
    }

    #[test]
    fn a_latent_block_is_its_channels_by_its_extents() {
        let shape = LatentShape {
            channels: 16,
            frames: 2,
            height: 3,
            width: 4,
        };
        assert_eq!(numel(&shape).unwrap(), 384);
        assert_eq!(latent_bytes(&shape, DType::F32).unwrap(), 384 * 4);
        assert_eq!(latent_bytes(&shape, DType::BF16).unwrap(), 384 * 2);
    }

    #[test]
    fn a_bucket_that_is_not_a_denoise_states_what_it_is() {
        let error = layout_of(Bucket::prefill(1, 16, Variant::Encode)).unwrap_err();
        assert!(format!("{error}").contains("serves a denoising bucket"));
    }

    #[test]
    fn a_block_that_does_not_fit_its_column_is_refused_by_name() {
        let mut buffer = vec![0u8; 8];
        let error = Staged::place(&mut buffer, 4, &[1u8; 8], "video latents").unwrap_err();
        assert!(format!("{error}").contains("does not fit a column of 8"));
        Staged::place(&mut buffer, 4, &[1u8; 4], "video latents").unwrap();
        assert_eq!(&buffer[4..], &[1, 1, 1, 1]);
    }

    /// A denoiser that states nothing: the tests above exercise the runner's own arithmetic, which
    /// needs no model.
    struct NoModel;

    impl Denoiser for NoModel {
        fn layout(&self, _request: &VideoRequest<'_>) -> Layout {
            layout(1, 1, 1, 1, 1)
        }

        fn draw_noise(
            &self,
            _f: &nn::Forward,
            _seed: Seed,
            _layout: Layout,
            _out: Latents,
        ) -> nn::Result<()> {
            Ok(())
        }

        fn prepare_latents<'a>(
            &self,
            _f: &nn::Forward,
            _cond: Conditioning<'a>,
            _out: Latents,
        ) -> nn::Result<()> {
            Ok(())
        }

        fn denoise<'a>(
            &self,
            _f: &nn::Forward,
            _step: Step,
            _latents: Latents,
            _cond: Conditioning<'a>,
        ) -> nn::Result<()> {
            Ok(())
        }

        fn evaluations(&self) -> u32 {
            0
        }

        fn constant_tables(&self) -> Vec<StateTable> {
            Vec::new()
        }

        fn prepare_constants(&self, _f: &nn::Forward, _out: &[TensorView]) -> nn::Result<()> {
            Ok(())
        }

        fn layout_tables(&self, _layout: &Layout) -> nn::Result<Vec<StateTable>> {
            Ok(Vec::new())
        }

        fn prepare_layout(&self, _layout: &Layout, _out: &mut [Vec<u8>]) -> nn::Result<()> {
            Ok(())
        }

        fn state_tables(&self, _layout: &Layout) -> Vec<StateTable> {
            Vec::new()
        }

        fn prepare_state(
            &self,
            _request: &VideoRequest<'_>,
            _layout: &Layout,
            _out: &mut [Vec<u8>],
        ) -> nn::Result<()> {
            Ok(())
        }

        fn prepare_evaluation(&self, _index: u32, _out: &mut [Vec<u8>]) -> nn::Result<()> {
            Ok(())
        }
    }
}

#[cfg(test)]
mod product_tests {
    use super::*;
    use std::sync::Mutex;

    /// A product store that answers for exactly the kinds it was given.
    struct Store {
        held: Vec<Product>,
        asked: Mutex<Vec<Product>>,
    }

    impl Products for Store {
        fn publish(
            &self,
            _request: ipc::RequestKey,
            _kind: Product,
            _device: platform::DeviceId,
            _stream: StreamId,
            _written: Option<Arc<platform::Event>>,
            _view: &TensorView,
        ) -> Result<()> {
            // The presence rule reads the store; who published what is `Store::held`'s own setup.
            Ok(())
        }

        fn holds(&self, _request: ipc::RequestKey, kind: Product) -> bool {
            self.held.contains(&kind)
        }

        fn product(
            &self,
            _request: ipc::RequestKey,
            _kind: Product,
        ) -> Result<Option<products::Published>> {
            unreachable!("the presence rule reads existence, not a placement")
        }

        fn copy_into(
            &self,
            _request: ipc::RequestKey,
            _kind: Product,
            _destination: &TensorView,
            _stream: &platform::Stream,
        ) -> Result<bool> {
            unreachable!("the presence rule never copies")
        }

        fn forget(&self, _request: ipc::RequestKey) -> usize {
            unreachable!("the presence rule never ends a request")
        }

        fn borrow(
            &self,
            _request: ipc::RequestKey,
            kind: Product,
            _device: platform::DeviceId,
            _stream: StreamId,
        ) -> Result<Option<TensorView>> {
            self.asked.lock().unwrap().push(kind);
            Ok(self
                .held
                .contains(&kind)
                .then(|| TensorView::new(ops::DevicePtr::new(0), DType::BF16, Shape::new(&[1, 1]))))
        }
    }

    fn request() -> ipc::RequestKey {
        ipc::RequestKey::new(1, ipc::RequestId(41), 0)
    }

    /// The capacity rule a served request depends on: the layout's conditioning block is the
    /// **deployment's** capacity, the product holds the request's own rows, and the difference is
    /// the packing's zero row.
    ///
    /// Both directions matter. A short product is padded rather than refused - refusing it is every
    /// request, because a prompt is shorter than the prefix capacity a deployment plans for. A long
    /// one is refused by name: it is not padding but a plan disagreement, and truncating it would
    /// drop states the denoiser was told are there.
    #[test]
    fn a_short_product_fills_the_rows_front_and_a_long_one_is_refused() {
        let (copied, zeroed) = fills(request(), Product::Text, 300, 512).expect("a short product");
        assert_eq!((copied, zeroed), (300, 212));

        let exact = fills(request(), Product::Text, 512, 512).expect("an exact product");
        assert_eq!(exact, (512, 0));

        let refusal = fills(request(), Product::Text, 640, 512)
            .expect_err("a product longer than the block")
            .to_string();
        assert!(
            refusal.contains("640"),
            "the refusal names what it got: {refusal}"
        );
        assert!(
            refusal.contains("512"),
            "and what the block holds: {refusal}"
        );

        let empty = fills(request(), Product::Text, 0, 512)
            .expect_err("a product no call wrote")
            .to_string();
        assert!(empty.contains("no byte(s)"), "{empty}");
    }

    /// The rule a step depends on: the refusal names **the first kind the store lacks, in the
    /// runner's own order**, so a failure says which call never ran rather than that something did.
    #[test]
    fn the_first_absent_kind_is_the_one_a_step_refuses() {
        let store = Store {
            held: vec![Product::Text],
            asked: Mutex::new(Vec::new()),
        };
        let missing =
            missing_product(&store, request(), CONDITIONED_STEP_PRODUCTS).expect("a store answers");
        assert_eq!(
            missing,
            Some(Product::Source),
            "the encoder's features are the first kind this store lacks"
        );
        let refusal = products::absent(request(), Product::Source).to_string();
        assert!(
            refusal.contains("source"),
            "the refusal names the kind: {refusal}"
        );
    }

    /// A store holding every kind a step reads has nothing missing, and it is asked in one pass.
    #[test]
    fn a_store_holding_every_kind_leaves_nothing_missing() {
        let store = Store {
            held: CONDITIONED_STEP_PRODUCTS.to_vec(),
            asked: Mutex::new(Vec::new()),
        };
        let missing =
            missing_product(&store, request(), CONDITIONED_STEP_PRODUCTS).expect("a store answers");
        assert_eq!(missing, None);
        assert_eq!(
            store.asked.lock().unwrap().len(),
            CONDITIONED_STEP_PRODUCTS.len(),
            "each kind is asked once"
        );
    }

    /// A task whose request carries no condition element reads no source rows, so a store holding
    /// the prompt's states serves it: a layout with no conditioning elements asks for one kind, and a
    /// layout with them asks for two.
    ///
    /// The rule is what keeps a prompt-only task from being refused for a product no call can
    /// produce, so it is asserted from the layout side rather than from the store side.
    #[test]
    fn a_layout_without_conditioning_elements_reads_only_the_prompts_states() {
        let mut layout = Layout {
            video: nn::capability::LatentShape {
                channels: 16,
                frames: 2,
                height: 2,
                width: 2,
            },
            audio: None,
            text_rows: 64,
            condition_rows: 0,
            condition_elements: 0,
        };
        assert_eq!(step_products(&layout), STEP_PRODUCTS);
        assert_eq!(step_products(&layout).len(), 1);

        layout.condition_elements = 3;
        assert_eq!(step_products(&layout), CONDITIONED_STEP_PRODUCTS);

        let store = Store {
            held: vec![Product::Text],
            asked: Mutex::new(Vec::new()),
        };
        let missing =
            missing_product(&store, request(), CONDITIONED_STEP_PRODUCTS).expect("a store answers");
        assert_eq!(
            missing,
            Some(Product::Source),
            "a conditioned layout asks a store holding only the prompt's states for the source"
        );
    }

    /// A step with no store at all is a different fact from one whose store lacks a kind, and it says
    /// so: the first is the executor not having attached one, the second is a call that never ran.
    #[test]
    fn no_store_is_its_own_refusal() {
        let refusal = products::no_store(request()).to_string();
        assert!(
            refusal.contains("no product store"),
            "the refusal names the attached store: {refusal}"
        );
        assert_eq!(
            Product::VideoLatents.name(),
            "video latents",
            "a kind is named as the design names it"
        );
    }
}
