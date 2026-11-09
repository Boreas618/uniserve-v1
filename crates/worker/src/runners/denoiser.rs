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
//! arm). A step also reads its encoded conditioning rows, which the encoder published to the
//! product store, and the request's latent trajectory, which this runner owns.
//!
//! **The trajectory lives in the runner's latent columns from the request's opening to its last
//! step.** The scheduler holds the denoiser for one request from its latent preparation to its last
//! step, so the columns hold that request's trajectory throughout. The latent preparation is served
//! here: it states how the columns hold the request's samples (`ipc::RankLatentLayout`) and, for a
//! request resumed from a stored trajectory, copies the stored pages of the host latent pool into
//! the columns (`ipc::LatentRestore`). The request's first step on this runner then folds its
//! conditioning and draws its noise, or continues from the restored trajectory. A preempted
//! request's trajectory is written from the columns to the host latent pool
//! (`ipc::LatentOffload`, [`Runner::offload_latents`]).

use std::sync::{Arc, Mutex};

use ipc::TimingCounters;
use nn::capability::{Conditioning, Denoiser, LatentShape, Latents, Layout, Step, WarmRequest};
use ops::{DType, Shape, TensorView};
use platform::DeviceId;

use ::storage::manifest::{Encoding, Field, Logical, Segment, Span, StorageLayout};

use crate::arena::{Pools, StreamId};
use crate::batch::{Batch, BatchOutput, CallId, CallOutput, Job, Jobs, RowState};
use crate::memory::latent::host::LatentHostPool;
use crate::memory::latent::restore;
use crate::plan::{Bucket, Extent, Variant};
use crate::runner::{Binding, Runner, RunnerId};
use crate::runners::columns::Column;
use crate::runners::diag::Diag;
use crate::runners::floor;
use crate::runners::lifecycle::{Lifecycle, walk_capture, warmed_capture};
use crate::runners::products::{self, Product, Products};
use crate::runners::tables::Tables;
use crate::{Error, Result};

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
    ///
    /// **This is `source`'s width, not necessarily `text`'s.** A conditioner may widen what the
    /// denoiser reads, and the model states that width itself (`Denoiser::text_width`), so the pair
    /// is resolved in [`conditioning_width`] rather than read from here twice.
    pub width: u32,
    /// The width of the rows the denoiser reads (`nn::Denoiser::text_width`), which `text` is sized
    /// by while `source`, holding what the encoder wrote, is sized by [`Self::width`].
    ///
    /// **The shape is where the model's statement lives, not the runner.** The engine asks the model
    /// once when it binds the runner and puts the answer here, so a runner never asks the model a
    /// second time and two statements of one width cannot drift.
    pub text_width: u32,
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
///
/// **The conditioning columns are not here.** They are hundreds of megabytes, a capture only needs
/// them to hold defined values, and a request writes every row of them it reads, so they are zeroed
/// once, where they are allocated.
#[derive(Default)]
struct Staged {
    video: Vec<u8>,
    audio: Vec<u8>,
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
}

/// The columns a denoiser stages a batch into, sized for the largest bucket it was bound with.
struct DenoiseColumns {
    /// `[requests x channels x frames x height x width]` video latents, one block per request.
    video: Column,
    /// The audio latents, in the same per-request blocks; one element when a bucket serves a task
    /// with no soundtrack, so the column is never empty.
    audio: Column,
    /// `[requests x (text_rows + condition_rows + 1) x text_width]` conditioning rows: the prefix
    /// source, whose last row is the zero row every generated and padding row gathers.
    text: Column,
    /// `[requests x (text_rows + condition_rows + 1) x width]` the text encoder's own features.
    ///
    /// A model with a conditioner between the encoder and the denoiser reads these and folds them
    /// into `text` in place ([`nn::Denoiser::prepare_conditioning`]).
    ///
    /// **Both buffers are the runner's even though the conditioner runs eagerly, once per request and
    /// outside the graph** ([`DenoiserRunner::first_step`]): `text` is the region the captured graph
    /// reads, so an output buffer the conditioner allocated for itself would be a per-request
    /// allocation no capture addresses, and `source` is where [`DenoiserRunner::seed`] puts the
    /// encoder's bytes on the store's clock.
    source: Column,
    /// The encoder's width: elements in one `source` row.
    source_width: u64,
    /// What the denoiser reads: elements in one `text` row.
    ///
    /// **Two widths, because a conditioner may widen.** H3's refiner projects the encoder's 5120
    /// columns into the stack's 5376, so a runner that sized both buffers from the encoder's width
    /// handed the conditioner one too narrow for its own output - and no launch refused until the
    /// packed rows were gathered, as `out is [rows, 5376], [rows, 5120] expected`. The model states
    /// the second width (`Denoiser::text_width`); `None` is one width.
    text_width: u64,
    /// `[1]` `i64`: the step index the coefficients are selected by (`Conditioning::step`).
    step: Column,
    /// The host side of each column.
    staged: Staged,
}

impl DenoiseColumns {
    /// Columns that hold a request of any of `layouts`, `rows` requests to a batch.
    ///
    /// The extents are taken one by one rather than from a largest layout: the longest timeline and
    /// the highest condition rung are two axes, and a bucket that is largest on one need not be on
    /// the other.
    fn new(device: DeviceId, shape: DenoiserShape, layouts: &[Layout], rows: u32) -> Result<Self> {
        let rows = u64::from(rows.max(1));
        let (mut video, mut audio, mut width) = (0, 1, 0);
        for layout in layouts {
            video = video.max(numel(&layout.video)?);
            if let Some(track) = layout.audio {
                audio = audio.max(numel(&track)?);
            }
            width = width.max(prefix_rows(layout));
        }
        let (source_width, text_width) = conditioning_width(&shape);
        let mut text = Column::new(device, shape.stack, rows * width * text_width)?;
        let mut source = Column::new(device, shape.stack, rows * width * source_width)?;
        // SAFETY: the columns were allocated above and nothing reads them yet.
        unsafe {
            text.host.as_mut_slice().fill(0);
            source.host.as_mut_slice().fill(0);
        }
        Ok(Self {
            video: Column::new(device, shape.sample, rows * video)?,
            audio: Column::new(device, shape.sample, rows * audio)?,
            text,
            source,
            step: Column::new(device, DType::I64, 1)?,
            staged: Staged::default(),
            source_width,
            text_width,
        })
    }

    /// Zeroes the latents' host side for `bucket`'s layout, so a warm capture reads defined values.
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

/// The products a step of `layout` reads, in the order a refusal names them.
///
/// **One list for every layout, because one product has a producer.** `Product::Text` is what the
/// encoder publishes (`crate::runners::encoder`), and `Product::Source` is published by nobody -
/// the product table says so as a contract (`crate::runners::products`) - so a conditioned layout
/// that required it refused every step with `absent(request, Source)` before the conditioner could
/// run. The encoder's rows are the conditioner's input as well as a prompt's: where they land is
/// decided by [`DenoiserRunner::seed`], not by asking the store for a second name for them.
fn step_products(_layout: &nn::capability::Layout) -> &'static [Product] {
    STEP_PRODUCTS
}

/// `forward` over `group`, when the runner's rank is in one.
pub(crate) fn grouped<'a>(
    forward: nn::Forward<'a>,
    group: Option<&'a nn::Group>,
) -> nn::Forward<'a> {
    match group {
        Some(group) => forward.with_group(group),
        None => forward,
    }
}

/// The rows of one request's prefix source: the prompt's, the conditions' and the zero row.
///
/// The packing gathers every generated and padding row from the row past the text and condition
/// capacities (`DensePacking::zero_row`), so that row is part of the column: a column that ended
/// before it would hand every such row whatever follows the allocation.
fn prefix_rows(layout: &Layout) -> u64 {
    u64::from(layout.text_rows) + u64::from(layout.condition_rows) + 1
}

/// The rows of one request's conditioning and the encoder's width: [`prefix_rows`] rows, the
/// encoder writing `width` elements in each.
///
/// One statement for the two readers that need it - the view a step hands the model
/// ([`DenoiserRunner::conditioning`]) and the rows [`DenoiserRunner::seed`] gives their bytes to.
/// Two statements of this arithmetic would be a step that reads a row the copies never filled, which
/// is a plausible clip rather than an error.
///
/// **`width` is `source`'s alone.** A conditioner may widen what the denoiser reads, so `text`'s
/// width is the model's statement and the pair is resolved in [`conditioning_width`].
fn conditioning_row(shape: &DenoiserShape, layout: &nn::capability::Layout) -> (u64, u64) {
    let width = u64::from(shape.width.max(1));
    (prefix_rows(layout), width)
}

/// The two widths one request's conditioning columns hold: `(source, text)`.
///
/// `source` is what the text encoder wrote, which [`DenoiserShape::width`] states; `text` is what
/// the denoiser reads, which [`DenoiserShape::text_width`] states - the planning boundary asked the
/// model once and equal to the encoder's width when the model has no conditioner. **One width for both
/// is the defect this pair removes**: the conditioner's own output then does not fit the buffer it
/// writes, and the refusal arrives one operator later, at the gather.
fn conditioning_width(shape: &DenoiserShape) -> (u64, u64) {
    let source = u64::from(shape.width.max(1));
    let text = u64::from(shape.text_width.max(1));
    (source, text)
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
    // **Presence, not device and stream.** Whether a product exists is a fact about the store; where a
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
#[derive(Clone, Debug)]
struct Pending {
    /// The request this row belongs to.
    request: ipc::RequestKey,
    /// The row of the bucket's columns the request occupies.
    index: u32,
    /// The request's noise seed: the draw is addressed by it and the element's index.
    seed: nn::capability::Seed,
    /// The request as its first step reads it: its own prompt length and its own conditions.
    stated: Arc<WarmRequest>,
}

/// The request a staged row states, within the bucket `warm` plans.
///
/// A graph is captured for a bucket's capacity, and a request's state tables say how much of that
/// capacity it fills: the refiner's keys stop at its prompt, and every packed row takes its place on
/// the rotary clock from that length. So the task, the canvas and the frames stay the plan's, and
/// everything the request itself states replaces the plan's figures.
fn stated_request(warm: &WarmRequest, conditioning: &[u8]) -> Result<WarmRequest> {
    use proto::video::{ConditionRole, Conditions};

    let decoded = Conditions::decode(conditioning).map_err(|error| {
        Error::Invalid(format!(
            "a media request's conditioning does not decode: {error}"
        ))
    })?;
    let count = |role: ConditionRole| {
        decoded
            .conditions
            .iter()
            .filter(|condition| condition.role == role)
            .count() as u32
    };
    let conditions = decoded
        .conditions
        .iter()
        .map(|condition| {
            let (frames, height, width) = condition
                .media
                .pixels()
                .map_or((0, 0, 0), |(frames, canvas)| {
                    (frames, canvas.height, canvas.width)
                });
            nn::ConditionShape {
                role: match condition.role {
                    ConditionRole::FirstFrame => nn::ConditionRole::FirstFrame,
                    ConditionRole::LastFrame => nn::ConditionRole::LastFrame,
                    ConditionRole::Reference => nn::ConditionRole::Reference,
                },
                frames,
                height,
                width,
                audio_samples: condition
                    .media
                    .audio()
                    .map_or(0, |clip| u32::try_from(clip.samples).unwrap_or(u32::MAX)),
            }
        })
        .collect();
    Ok(WarmRequest {
        first_frames: count(ConditionRole::FirstFrame),
        last_frames: count(ConditionRole::LastFrame),
        references: count(ConditionRole::Reference),
        text_tokens: decoded.prompt_tokens,
        conditions,
        vision_spans: decoded.vision_spans,
        ..warm.clone()
    })
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
pub struct DenoiserRunner<M: Denoiser + Send + Sync + ?Sized> {
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
    /// `None` until the worker attaches it, and a step that needs a product without one refuses by
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
    /// a second `submit` before a replay would be a second batch, which the worker does not do.
    pending: Vec<Pending>,
    /// Whether this load's constants are written.
    constants_filled: bool,
    /// The sequence group this runner's rank exchanges over, absent at one rank.
    group: Option<Arc<nn::Group>>,
    /// The calls of the batch staged last, which its replay answers.
    calls: Vec<CallId>,
    /// The steps those calls' trajectories have completed once the replay ran: a call reports it,
    /// and that report is what moves the request to its next step.
    steps_completed: u32,
    /// The request whose trajectory the latent columns hold, and the bucket it is laid out in: the
    /// request whose first step this runner ran last. A step of any other request opens that
    /// request first ([`Self::first_step`]). The scheduler holds the denoiser for one request from
    /// its latent preparation to its last step, so a trajectory is never displaced mid-schedule.
    opened: Option<(ipc::RequestKey, Bucket)>,
    /// The request a latent preparation restored a stored trajectory for, and the step that
    /// trajectory stands at: its first step continues from the columns instead of drawing noise.
    restored: Option<(ipc::RequestKey, u32)>,
    /// The latent preparations of the batch submitted last, which its jobs perform.
    preparations: Vec<Preparation>,
    /// The host latent pool of this runner's device: a preempted request's trajectory is written to
    /// it and a resumed one's read through it. `None` on a device that maps none.
    host_pool: Option<Arc<LatentHostPool>>,
    /// The rank of the worker group this device belongs to, which the engine's page copies and
    /// restore readers name.
    pool_rank: u32,
    /// Whether this device states a trajectory's page layout for its rank: every device of a
    /// sequence group holds the whole samples, so one of them states them for all.
    reports_layout: bool,
}

/// One latent preparation of a submitted batch.
struct Preparation {
    /// The call it answers.
    call: CallId,
    /// The request whose trajectory it opens.
    request: ipc::RequestKey,
    /// The denoising bucket the request's steps run in, whose layout the trajectory has.
    bucket: Bucket,
    /// The stored trajectory it opens at, for a resumed request.
    restore: Option<ipc::LatentRestore>,
}

impl<M: Denoiser + Send + Sync + ?Sized> DenoiserRunner<M> {
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
            let bucket =
                Bucket::denoise(1, layout).serving(request.frames, request.width, request.height);
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
        let rows = buckets
            .iter()
            .map(|plan| plan.bucket.rows)
            .max()
            .ok_or_else(|| Error::Invalid("a denoiser runner derived no bucket".into()))?;
        let layouts = buckets
            .iter()
            .map(|plan| layout_of(plan.bucket))
            .collect::<Result<Vec<_>>>()?;
        let columns = DenoiseColumns::new(binding.device, shape, &layouts, rows)?;
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
            constants_filled: false,
            group: None,
            calls: Vec::new(),
            steps_completed: 0,
            opened: None,
            restored: None,
            preparations: Vec::new(),
            host_pool: None,
            pool_rank: 0,
            reports_layout: false,
        })
    }

    /// This runner as rank `rank` of the sequence `group`: every forward it runs carries the group.
    #[must_use]
    pub fn in_group(mut self, rank: u32, group: Option<Arc<nn::Group>>) -> Self {
        self.diag = std::mem::replace(&mut self.diag, Diag::from_env()).for_rank(rank);
        self.group = group;
        self
    }

    /// The bytes of the largest trajectory this runner holds: the video and audio latents of its
    /// largest layout, which a host latent pool's trajectory pages are sized for.
    pub fn trajectory_bytes(&self) -> Result<u64> {
        let element = self.shape.sample.bytes().ok_or_else(|| {
            Error::Invalid(format!("{} has no whole-byte size", self.shape.sample))
        })?;
        let mut largest = 0;
        for plan in &self.buckets {
            let layout = layout_of(plan.bucket)?;
            let audio = layout.audio.as_ref().map_or(Ok(0), numel)?;
            largest = largest.max((numel(&layout.video)? + audio) * element);
        }
        Ok(largest)
    }

    /// This runner with its device's host latent pool, as a device of rank `rank` of its worker
    /// group: preempted trajectories are written to the pool and resumed ones read through it, and
    /// when `reports` is set this device states the trajectory's page layout for the rank.
    #[must_use]
    pub fn with_host_latents(
        mut self,
        pool: Arc<LatentHostPool>,
        rank: u32,
        reports: bool,
    ) -> Self {
        self.host_pool = Some(pool);
        self.pool_rank = rank;
        self.reports_layout = reports;
        self
    }

    /// The buckets this runner captures, largest first: the longest timeline, and within it the
    /// highest condition rung.
    fn capture_order(&self) -> Vec<Bucket> {
        let mut buckets: Vec<Bucket> = self.buckets.iter().map(|plan| plan.bucket).collect();
        buckets.sort_by_key(|bucket| std::cmp::Reverse(*bucket));
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
        self.model.schedule(&plan.warm.view()).map_err(Error::from)
    }

    /// Claims the arena the largest bucket's evaluation plans.
    fn claim(&mut self) -> Result<()> {
        let plans = self.buckets.clone();
        let mut needed = 0;
        for plan in &plans {
            needed = needed.max(
                floor::checked(
                    plan.bucket,
                    self.model.arena_bytes(&plan.warm.view()),
                    u64::MAX,
                )?
                .unwrap_or(0),
            );
        }
        if needed > 0 {
            let stream = self.stream_id();
            self.pool()?.claim(stream, needed)?;
        }
        Ok(())
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
                "denoise-{}-{}-{}-{}-{}x{}x{}",
                layout.video.channels,
                layout.video.frames,
                layout.text_rows,
                layout.condition_rows,
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

    /// The rows and the two widths one request's conditioning occupies: `(rows, source, text)`.
    ///
    /// Read from the columns rather than recomputed, so the view a step hands the model and the row
    /// size [`Self::seed`] fills are the ones the buffers were allocated for.
    fn conditioning_row(&self, layout: &nn::capability::Layout) -> (u64, u64, u64) {
        let (rows, _) = conditioning_row(&self.shape, layout);
        (rows, self.columns.source_width, self.columns.text_width)
    }

    /// The conditioning of request `index` of `bucket`, over the tables of its layout.
    ///
    /// `text` and `source` are two views of **two widths**: a conditioner may widen what the
    /// denoiser reads, so a step that handed the model one width for both would address the wrong
    /// columns of one of them.
    fn conditioning<'a>(
        &self,
        bucket: Bucket,
        index: u32,
        state: &'a [TensorView],
        media: &'a [nn::capability::MediaInput],
    ) -> Result<Conditioning<'a>> {
        let layout = layout_of(bucket)?;
        let (rows, source_width, text_width) = self.conditioning_row(&layout);
        let view = self.columns.text.shaped(&[rows * text_width]);
        let text = view
            .at(view.offset_elements(u64::from(index) * rows * text_width))
            .reshape(Shape::new(&[rows, text_width]));
        let features = self.columns.source.shaped(&[rows * source_width]);
        let source = features
            .at(features.offset_elements(u64::from(index) * rows * source_width))
            .reshape(Shape::new(&[rows, source_width]));
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
        // The encoder's rows stay in `source` for the request's life, and only the step that opens
        // the request on this runner folds them: a later step's copy would write bytes no launch
        // reads.
        if !self.opens() {
            return Ok(());
        }
        self.zero_rows(bucket)?;
        let Some(products) = self.products.clone() else {
            return Ok(());
        };
        // The rows and the stream are copied out before the columns are borrowed mutably: the
        // padding below writes a column's host side, which is the memory its graph reads.
        let pending = self.pending.clone();
        let stream = self.stream.clone();
        let layout = layout_of(bucket)?;
        let (rows, source_width, text_width) = self.conditioning_row(&layout);
        let element = self.shape.stack.bytes().ok_or_else(|| {
            Error::Invalid(format!("{} has no whole-byte size", self.shape.stack))
        })?;
        // One row's size per kind: the two columns are two widths, so a single number would make
        // the shorter kind's copy land inside the wrong column's row (`conditioning_width`).
        let text_bytes = rows * text_width * element;
        let source_bytes = rows * source_width * element;
        // What the conditioner reads of `source` is the text capacity: the rows past it are the
        // conditions', which the encoders write into `text` and no launch reads here.
        let folded_bytes = u64::from(layout.text_rows) * source_width * element;
        // **The encoder's rows go where the model consumes them next.** A conditioner that widens
        // reads `source` and writes `text` in place, so the product lands in `source` and `text`
        // belongs to the fold alone - copying it into `text` as well would be a second, mis-strided
        // copy of the same bytes (`text_width != source_width`, so the rows would not line up). A
        // model with no conditioner reads the encoder's rows themselves, so they land in `text`.
        //
        // The reference settles the direction: its `condition()` takes the encoder's features as the
        // conditioner's **input** and its `text_condition` is written only by the conditioner's
        // output, for every task. The two widths are therefore the only statement the interface makes
        // about whether a conditioner runs (`Denoiser::text_width` answers `None` for no conditioner
        // and for one that keeps the encoder's width alike), which is why the branch reads them
        // rather than asking a second question.
        let folded = text_width != source_width;
        for row in &pending {
            let (request, index) = (row.request, row.index);
            let kind = Product::Text;
            // `stride` is the row of the column and `bytes` what this copy fills of it.
            let (column, stride, bytes) = if folded {
                (&mut self.columns.source, source_bytes, folded_bytes)
            } else {
                (&mut self.columns.text, text_bytes, text_bytes)
            };
            {
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
                    ops::DevicePtr::new(column.host.address() + u64::from(index) * stride),
                    self.shape.stack,
                    Shape::new(&[copied / element]),
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
                    let start = usize::try_from(u64::from(index) * stride + copied)
                        .map_err(|_| Error::Invalid("a row is larger than memory".into()))?;
                    let end = usize::try_from(u64::from(index) * stride + bytes)
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

    /// Writes the zero row of each pending request's prefix source.
    ///
    /// Every generated and padding row gathers it (`DensePacking::zero_row`), and it is the last row
    /// of the layout's prefix, so it sits at another place in the column for each condition rung:
    /// a request of a higher rung wrote a condition row where a lower rung reads its zeros. The
    /// rest of a request's `text` is the refiner's and the encoders' to write, and no gather reads
    /// a row they left.
    fn zero_rows(&mut self, bucket: Bucket) -> Result<()> {
        let layout = layout_of(bucket)?;
        let (rows, _, text_width) = self.conditioning_row(&layout);
        let element = self.shape.stack.bytes().ok_or_else(|| {
            Error::Invalid(format!("{} has no whole-byte size", self.shape.stack))
        })?;
        let stride = rows * text_width * element;
        let row = text_width * element;
        for pending in &self.pending {
            let start = usize::try_from(u64::from(pending.index) * stride + stride - row)
                .map_err(|_| Error::Invalid("a row is larger than memory".into()))?;
            let end = start + row as usize;
            // SAFETY: the row lies inside this request's block of the column, and `upload_step`
            // waited out the launches of the previous request, which are the column's only readers.
            unsafe {
                self.columns.text.host.as_mut_slice()[start..end].fill(0);
            }
        }
        Ok(())
    }

    /// Issues each pending row's first step: the model's own draw, its conditioner and its fold, on
    /// the replay's arena and stream, before the captured graph launches.
    ///
    /// A trajectory's value at step zero is not a step: it is `Denoiser::draw_noise` (Philox
    /// addressed by the request's seed and the element's index, section 9) followed by
    /// `Denoiser::prepare_latents` (the schedule's start over that draw), and both take a
    /// `Forward` because their operators reserve scratch in an arena. `Denoiser::prepare_conditioning`
    /// runs between them: the encoder's rows are raw until it refines them, and the fold reads
    /// `cond.text`, so refining after the fold would start the schedule from unrefined rows.
    ///
    /// **None of the three is in the captured graph.** Each runs once per request, and a graph
    /// replays once per step; `prepare_conditioning` writes `text` in place, so a refinement inside
    /// the graph would compound over the schedule.
    ///
    /// The replay's arena is the one its captured graph addresses: `Pool::base` is the stream's
    /// range and `Pool::begin` answers only whether a bucket is planning or replaying, so the same
    /// range serves a replay. That is what makes the draw affordable here - its reservations are
    /// the walk's own, and the graph's nodes address their recorded offsets above a column the draw
    /// writes rather than above scratch it overwrites. A later step needs none of this: its
    /// trajectory is the column the previous replay advanced in place.
    ///
    /// **This runs for the step that opens a request on this runner**, which is step zero for a
    /// request that starts here and the stored step for one a latent preparation restored. A
    /// restored request's columns already hold its trajectory at that step, so it draws no noise
    /// and starts no schedule; its conditioner, its condition draws and its state tables are
    /// computed again from the request, which is all they depend on.
    fn first_step(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        if !self.opens() {
            return Ok(());
        }
        let request = self.pending[0].request;
        let restored = match self.restored {
            Some((stored, steps)) if stored == request => Some(steps),
            _ => None,
        };
        let stands = restored.unwrap_or(0);
        if self.staged_step != Some(stands) {
            return Err(Error::Invalid(format!(
                "request {request:?} opens on this denoiser at step {:?} and its trajectory here \
                 stands at step {stands}: a request that continues past its first step on another \
                 runner has nothing to continue from unless its latent preparation restored it",
                self.staged_step
            )));
        }
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        self.fill(bucket)?;
        self.state_of_request(bucket)?;
        let base = self.pool()?.base(self.stream_id())?;
        let reserved = self.pool()?.reserved_on(self.stream_id());
        let arena = nn::Arena::new(ops::DevicePtr::new(base.address()), reserved);
        let group = self.group.clone();
        let forward = grouped(
            nn::Forward::new(&stream, &kernels, &arena),
            group.as_deref(),
        );
        let layout = layout_of(bucket)?;
        let tables = self.model.state_tables(&layout);
        let key = Self::key(bucket);
        let state = self
            .tables
            .views(self.lifecycle.generation.number(), &key, &tables)?;
        for row in &self.pending {
            let latents = self.latents(bucket, row.index)?;
            // The raw media a conditioner encodes, each visual piece with its anchoring draw: the
            // request's first step is the one forward that reads them.
            let media = self.media_of(row, &forward)?;
            let cond = self.conditioning(bucket, row.index, &state, &media)?;
            // The stream's first elements are the conditions' draws, so the trajectory's offsets
            // are the request's own and not the warm request's the bucket was captured from.
            let drawn = Layout {
                condition_elements: self.model.layout(&row.stated.view()).condition_elements,
                ..layout
            };
            if restored.is_none() {
                self.model.draw_noise(&forward, row.seed, drawn, latents)?;
            }
            // **The conditioner, eager and once per request, outside the graph.** `text` holds the
            // encoder's rows as `seed` copied them; this folds `source` into it in place, so every
            // replay of the bucket reads rows refined for its own request. It is not a graph node
            // because a graph replays once per *step* while this runs once per *request* - a
            // refinement inside the captured graph would compound over the schedule, and `text` is
            // written in place.
            if let Some(source) = cond.source {
                self.dump("conditioner", "source", source);
            }
            self.model.prepare_conditioning(&forward, cond)?;
            self.dump("conditioner", "text", cond.text);
            for (index, piece) in media.iter().enumerate() {
                if let Some(pixels) = piece.pixels {
                    self.dump("media", &format!("pixels_{index}"), pixels);
                }
                if let Some(audio) = piece.audio {
                    self.dump("media", &format!("samples_{index}"), audio);
                }
                if let Some(noise) = piece.noise {
                    self.dump("media", &format!("draw_{index}"), noise);
                }
            }
            if self.diag.enabled() {
                for (table, view) in tables.iter().zip(state.iter()) {
                    self.dump("state", table.name, *view);
                }
            }
            if restored.is_none() {
                self.model.prepare_latents(&forward, cond, latents)?;
            }
        }
        self.opened = Some((request, bucket));
        self.restored = None;
        Ok(())
    }

    /// Whether the pending request's step opens it on this runner: the step whose host work folds
    /// its conditioning and writes its state tables before the graph runs.
    fn opens(&self) -> bool {
        match self.pending.as_slice() {
            [row] => self
                .opened
                .is_none_or(|(request, _)| request != row.request),
            _ => false,
        }
    }

    /// Notes the latent preparations of `batch`: each opens a trajectory of the denoising bucket
    /// whose layout the request's shape selected, and a resumed request's opens at the stored
    /// trajectory the engine pinned.
    fn submit_preparation(
        &mut self,
        batch: &Batch,
        media: &crate::batch::MediaColumns,
    ) -> Result<Jobs> {
        let computation = media.computation;
        let bucket = self
            .buckets
            .iter()
            .map(|plan| plan.bucket)
            .find(|bucket| bucket.extent == computation.extent)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "a latent preparation names a layout no denoising bucket of this runner holds \
                     ({:?})",
                    computation.extent
                ))
            })?;
        if media.inputs.len() != batch.calls.len() {
            return Err(Error::Invalid(format!(
                "a latent preparation batch of {} call(s) carries {} row(s) of staged inputs",
                batch.calls.len(),
                media.inputs.len()
            )));
        }
        self.preparations.clear();
        for (index, call) in batch.calls.iter().enumerate() {
            let stated = media.request(index)?;
            let restore = batch
                .latent_restores
                .iter()
                .find(|(request, _)| *request == stated.request)
                .map(|(_, restore)| restore.clone());
            self.preparations.push(Preparation {
                call: *call,
                request: stated.request,
                bucket,
                restore,
            });
        }
        Ok(Jobs::new(batch.calls.iter().map(|call| Job::Prepare {
            runner: self.id,
            call: *call,
        })))
    }

    /// Performs the latent preparation of `call`: a new trajectory begins on this runner, its page
    /// layout is stated for the engine, and a resumed request's stored pages are copied into it.
    ///
    /// Whatever request the columns held is finished or was preempted and stored, because the
    /// scheduler holds the denoiser for one request at a time.
    fn prepare(&mut self, call: CallId) -> Result<BatchOutput> {
        let index = self
            .preparations
            .iter()
            .position(|preparation| preparation.call == call)
            .ok_or_else(|| {
                Error::Invariant(format!(
                    "a preparation job names call {}, which the batch submitted last did not \
                     prepare",
                    call.0
                ))
            })?;
        let preparation = self.preparations.remove(index);
        let layout = trajectory_layout(&layout_of(preparation.bucket)?, self.shape.sample)?;
        self.opened = None;
        self.restored = None;
        let step = match &preparation.restore {
            Some(restore) => {
                self.restore(preparation.bucket, &layout, restore)?;
                self.restored = Some((preparation.request, restore.steps));
                tracing::info!(
                    request = ?preparation.request,
                    steps = restore.steps,
                    "restored a stored trajectory into the denoiser"
                );
                restore.steps
            }
            None => 0,
        };
        let mut output = CallOutput::completed(
            call,
            Vec::new(),
            RowState {
                step_index: step,
                ..RowState::default()
            },
        );
        if self.reports_layout && self.host_pool.is_some() {
            output.latent_layouts.push(ipc::RankLatentLayout {
                rank: self.pool_rank,
                layout: serde_json::to_string(&layout).map_err(|error| {
                    Error::Invalid(format!(
                        "a trajectory's page layout does not serialize: {error}"
                    ))
                })?,
            });
        }
        Ok(BatchOutput {
            calls: vec![output],
            ..BatchOutput::default()
        })
    }

    /// Copies the stored pages `restore` names into the trajectory of `bucket`, laid out in this
    /// runner's columns as `layout` states: the copy manifest from the stored ranks' pages this
    /// rank reads to the rank's own (`crate::memory::latent::restore`), which copies a page laid out
    /// alike whole. The copies have completed when this returns, before the engine releases the
    /// pages it pinned.
    fn restore(
        &self,
        bucket: Bucket,
        layout: &StorageLayout,
        restore: &ipc::LatentRestore,
    ) -> Result<()> {
        let pool = self.host_pool.as_ref().ok_or_else(|| {
            Error::Invalid(
                "a latent preparation restores a stored trajectory on a device that maps no host \
                 latent pool"
                    .into(),
            )
        })?;
        let sources: Vec<&ipc::LatentSource> = restore
            .sources
            .iter()
            .filter(|source| source.readers.contains(&self.pool_rank))
            .collect();
        if sources.is_empty() {
            return Err(Error::Invalid(format!(
                "a latent restore names no stored pages rank {} reads",
                self.pool_rank
            )));
        }
        let stored = sources
            .iter()
            .map(|source| {
                Ok(restore::Stored {
                    layout: serde_json::from_str(&source.layout).map_err(|error| {
                        Error::Invalid(format!("a stored trajectory's page layout: {error}"))
                    })?,
                    page_bytes: source.pool.page_bytes,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let target = restore::Stored {
            layout: layout.clone(),
            page_bytes: pool.page_bytes(),
        };
        let gpu = self.device.uuid()?.0;
        let copies = restore::copies(&stored, &target, restore.content, gpu)?;
        let regions = trajectory_regions(&self.latents(bucket, 0)?)?;
        let total: u64 = regions.iter().map(|(_, bytes)| bytes).sum();
        for copy in &copies {
            // A page laid out alike is copied whole, and the trajectory's last page holds fewer
            // bytes than a page: what lies past the trajectory is no value of it, and the columns
            // end there for the largest layout.
            if copy.target >= total {
                continue;
            }
            let length = copy.bytes.min(total - copy.target);
            let source = sources[copy.source];
            let slot = *source.slots.get(copy.page as usize).ok_or_else(|| {
                Error::Invalid(format!(
                    "a latent restore copies stored page {} of a source of {} pages",
                    copy.page,
                    source.slots.len()
                ))
            })?;
            let page = pool.source(&source.pool, slot)?;
            let from = page.address() + copy.offset;
            for (address, offset, bytes) in pieces(&regions, copy.target, length)? {
                // SAFETY: the stored page stays pinned until this call's result, and the copies
                // complete before this returns; the destination lies in this runner's column, which
                // no graph reads while the denoiser serves this preparation.
                unsafe {
                    platform::memcpy_async(address, from + offset, bytes as usize, &self.stream)
                }?;
            }
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// The raw media of `row`'s conditions, in the order the model packs them.
    ///
    /// The media reader left each kind in one region, conditions in request order, so a piece's
    /// place in its region is the sum of the extents of the conditions before it. Those extents are
    /// the model's statement ([`nn::capability::ConditionPiece`]), and the two regions are read in
    /// place: a condition is read once, by this forward.
    fn media_of(
        &self,
        row: &Pending,
        forward: &nn::Forward<'_>,
    ) -> Result<Vec<nn::capability::MediaInput>> {
        let pieces = self
            .model
            .condition_pieces(&row.stated.view())
            .map_err(Error::from)?;
        if pieces.is_empty() {
            return Ok(Vec::new());
        }
        let products = self
            .products
            .as_ref()
            .ok_or_else(|| products::no_store(row.request))?;
        let region = |kind: Product, wanted: bool| -> Result<Option<TensorView>> {
            if !wanted {
                return Ok(None);
            }
            products
                .ordered(row.request, kind, &self.stream)?
                .map(Some)
                .ok_or_else(|| products::absent(row.request, kind))
        };
        let pixels = region(
            Product::ConditionPixels,
            pieces.iter().any(|piece| piece.pixels.is_some()),
        )?;
        let samples = region(
            Product::ConditionSamples,
            pieces.iter().any(|piece| piece.track.is_some()),
        )?;
        let mut offsets = vec![(0_u64, 0_u64); row.stated.conditions.len()];
        let elements = |extents: &[u64]| extents.iter().product::<u64>();
        let mut ordered: Vec<&nn::capability::ConditionPiece> = pieces.iter().collect();
        ordered.sort_by_key(|piece| piece.condition);
        let (mut pixel_at, mut sample_at) = (0_u64, 0_u64);
        for piece in ordered {
            offsets[piece.condition as usize] = (pixel_at, sample_at);
            pixel_at += piece.pixels.as_ref().map_or(0, |shape| elements(shape));
            sample_at += piece.track.as_ref().map_or(0, |shape| elements(shape));
        }
        let view = |source: Option<TensorView>, at: u64, shape: &[u64]| -> Result<TensorView> {
            let source = source.ok_or_else(|| {
                Error::Invalid(format!(
                    "request {} states a piece the media reader left no region for",
                    row.request.request_id.0
                ))
            })?;
            Ok(source
                .at(source.offset_elements(at))
                .reshape(Shape::new(shape)))
        };
        let mut media = Vec::with_capacity(pieces.len());
        for piece in &pieces {
            let (pixel_at, sample_at) = offsets[piece.condition as usize];
            let noise = match piece.draw {
                Some(draw) => {
                    let view = forward
                        .arena()
                        .tensor(self.shape.sample, latent_shape(&draw))?;
                    self.model
                        .draw_condition_noise(forward, row.seed, piece, view)
                        .map_err(Error::from)?;
                    Some(view)
                }
                None => None,
            };
            media.push(nn::capability::MediaInput {
                role: piece.role,
                rows: piece.rows,
                audio_rows: piece.audio_rows,
                pixels: piece
                    .pixels
                    .as_ref()
                    .map(|shape| view(pixels, pixel_at, shape))
                    .transpose()?,
                audio: piece
                    .track
                    .as_ref()
                    .map(|shape| view(samples, sample_at, shape))
                    .transpose()?,
                noise,
            });
        }
        Ok(media)
    }

    /// Writes the pending request's own state tables over the ones the capture read.
    ///
    /// A bucket's tables are one set, so a bucket serves one request at a time.
    fn state_of_request(&mut self, bucket: Bucket) -> Result<()> {
        let [row] = self.pending.as_slice() else {
            return Err(Error::Invalid(format!(
                "a first step of {} requests reached bucket {bucket}, whose state tables hold one",
                self.pending.len()
            )));
        };
        let layout = layout_of(bucket)?;
        let tables = self.model.state_tables(&layout);
        if tables.is_empty() {
            return Ok(());
        }
        let stated = Arc::clone(&row.stated);
        let model = Arc::clone(&self.model);
        self.tables.rewrite(
            &self.stream,
            self.lifecycle.generation.number(),
            &Self::key(bucket),
            &tables,
            |host| {
                model
                    .prepare_state(&stated.view(), &layout, host)
                    .map_err(Error::from)
            },
        )
    }

    /// Writes one named tensor of the block-by-block diagnosis, when a directory is named.
    fn dump(&self, block: &str, name: &str, view: TensorView) {
        if self.diag.enabled() {
            let _ = self
                .diag
                .write(self.device, &self.stream, block, name, view);
        }
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

    /// The refusal for a bucket whose request the model states no schedule for.
    ///
    /// It names the chain a reader needs - the bucket, the request the bucket's schedule was asked
    /// for, and the fact that the answer had no step - because the model's own refusal names the
    /// request and this one is reached through the bucket.
    fn no_step(&self, bucket: Bucket) -> Error {
        let request = self
            .plan_of(bucket)
            .map(|plan| {
                let warm = &plan.warm;
                format!(
                    "{}x{} of {} frames with {} anchor(s) and {} reference(s)",
                    warm.width,
                    warm.height,
                    warm.frames,
                    warm.first_frames + warm.last_frames,
                    warm.references
                )
            })
            .unwrap_or_else(|_| "no request".to_owned());
        Error::Invalid(format!(
            "the schedule of bucket {bucket} states no step to capture; the bucket asked for the \
             schedule of a request of {request}"
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
            step,
            staged,
            ..
        } = &mut self.columns;
        video.write(&staged.video)?;
        audio.write(&staged.audio)?;
        step.write(&staged.step)?;
        self.columns.video.upload(&stream)?;
        self.columns.audio.upload(&stream)?;
        self.columns.step.upload(&stream)?;
        Ok(())
    }

    /// Copies the column a **replay** writes from the host: the step's index.
    ///
    /// **The trajectory is not one of them.** The captured graph advances the latents *in place* —
    /// that is the whole of a diffusion step — while the staged buffer holds a warm capture's zeros,
    /// so uploading it before every replay would restart the trajectory at zero whenever a step
    /// began. Nothing refused: step one would advance zeros, and the clip would be the draw plus one
    /// step rather than the schedule. The draw at step zero writes the whole block itself
    /// (`first_step`), so the column never needs this path for its initial values.
    fn upload_step(&mut self) -> Result<()> {
        let stream = self.stream.clone();
        // The refined rows are the request's for its whole schedule: `first_step` folds them once,
        // and every later step reads them where they are. Only a first step writes the conditioning
        // columns from the host (`seed`), so the previous request's last launch is waited out first.
        if self.staged_step == Some(0) {
            stream.synchronize()?;
        }
        let DenoiseColumns { step, staged, .. } = &mut self.columns;
        step.write(&staged.step)?;
        self.columns.step.upload(&stream)?;
        Ok(())
    }

    /// Fills the model's constants, once per load and ahead of every capture.
    ///
    /// They are values folded from the weights, such as an AdaLN cache, and every step reads them.
    /// A model's region left unwritten is zeros, so its gates close and the stack passes its input
    /// through unchanged, with nothing refusing.
    fn fill_constants(&mut self, kernels: &Arc<ops::Generation>) -> Result<()> {
        if self.constants_filled {
            return Ok(());
        }
        if !self.model.constant_tables().is_empty() {
            return Err(Error::Invalid(
                "this model declares constant tables, and this runner holds no storage for them"
                    .into(),
            ));
        }
        let stream = self.stream.clone();
        let base = self.pool()?.base(self.stream_id())?;
        let reserved = self.pool()?.reserved_on(self.stream_id());
        let arena = nn::Arena::new(ops::DevicePtr::new(base.address()), reserved);
        let forward = nn::Forward::new(&stream, kernels, &arena);
        self.model
            .prepare_constants(&forward, &[])
            .map_err(Error::from)?;
        self.constants_filled = true;
        Ok(())
    }

    /// Writes the staged step's index into the `step` state table a replay reads.
    ///
    /// The graph selects its modulation and its solver coefficient on the device from this word, so
    /// one capture serves every evaluation. The copy is stream-ordered, so a step queued behind a
    /// running one changes nothing the running one reads.
    fn write_step(&mut self, bucket: Bucket) -> Result<()> {
        let layout = layout_of(bucket)?;
        let tables = self.model.state_tables(&layout);
        let Some(position) = tables.iter().position(|table| table.name == "step") else {
            return Ok(());
        };
        let state = self.tables.views(
            self.lifecycle.generation.number(),
            &Self::key(bucket),
            &tables,
        )?;
        let index = self.staged_step.unwrap_or(0).to_le_bytes();
        // SAFETY: the destination is the four-byte table and the source four live bytes, staged by
        // the driver before the call returns.
        unsafe {
            platform::memcpy_async(
                state[position].ptr.address(),
                index.as_ptr() as u64,
                index.len(),
                &self.stream,
            )?;
        }
        Ok(())
    }

    /// Runs the staged step operator by operator instead of replaying its graph.
    ///
    /// This is the diagnosis' own control: a step that is right here and wrong in a replay says the
    /// capture is what differs.
    fn eager_step(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        let base = self.pool()?.base(self.stream_id())?;
        let reserved = self.pool()?.reserved_on(self.stream_id());
        let arena = nn::Arena::new(ops::DevicePtr::new(base.address()), reserved);
        let group = self.group.clone();
        let forward = grouped(
            nn::Forward::new(&stream, &kernels, &arena),
            group.as_deref(),
        );
        let layout = layout_of(bucket)?;
        let tables = self.model.state_tables(&layout);
        let key = Self::key(bucket);
        let state = self
            .tables
            .views(self.lifecycle.generation.number(), &key, &tables)?;
        let index = self.staged_step.unwrap_or(0);
        let step = self
            .schedule(bucket)?
            .get(index as usize)
            .copied()
            .ok_or_else(|| self.no_step(bucket))?;
        let media: [nn::capability::MediaInput; 0] = [];
        let latents = self.latents(bucket, 0)?;
        let cond = self.conditioning(bucket, 0, &state, &media)?;
        self.model
            .denoise(&forward, step, latents, cond)
            .map_err(Error::from)
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
        let reserved = self.pool()?.reserved_on(self.stream_id());
        let arena = nn::Arena::new(ops::DevicePtr::new(base.address()), reserved);
        let group = self.group.clone();
        let probes = crate::profiling::fences::capturing(&stream);
        let forward = grouped(
            nn::Forward::new(&stream, generation, &arena)
                .with_fence(probes.as_ref().map(|probes| probes as &dyn nn::Fence)),
            group.as_deref(),
        );
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

impl<M: Denoiser + Send + Sync + ?Sized> Runner for DenoiserRunner<M> {
    /// The constants are folded from the weights, so they are folded again from the new ones. A
    /// runner that has not filled them yet does so before its first capture anyway.
    fn weights_changed(&mut self) -> Result<()> {
        if !self.constants_filled {
            return Ok(());
        }
        self.constants_filled = false;
        let generation = self.lifecycle.generation.number();
        let kernels = self.lifecycle.generation_named(generation)?;
        self.fill_constants(&kernels)
    }

    fn phases(&self) -> &[ipc::PhaseId] {
        &self.phases
    }

    /// A step's graph reads the step's index from a state table the stream writes in order, so the
    /// next step can be launched behind it. The diagnosis reads the columns between steps, so it
    /// waits for each.
    fn overlaps_replays(&self) -> bool {
        !self.diag.enabled()
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
        if computation.variant() == Some(Variant::LatentPrepare) {
            return self.submit_preparation(&batch, &media);
        }
        if computation.variant() != Some(Variant::Denoise) {
            return Err(Error::Invalid(format!(
                "a denoiser runner serves a denoise or a latent preparation; this batch states a {:?}",
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
                stated: Arc::new(self::stated_request(
                    &self.plan_of(bucket)?.warm,
                    &stated_request.conditioning,
                )?),
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
        self.calls.clone_from(&batch.calls);
        self.steps_completed = step.index + 1;
        Ok(Jobs::new([Job::Replay {
            runner: self.id,
            bucket,
            generation: batch.generation,
            batch: batch.id,
        }]))
    }

    fn media_plan(&self) -> Vec<Bucket> {
        // A latent preparation opens a trajectory of a layout this runner steps, so each denoising
        // bucket has a preparation bucket of the same extent and the request's shape selects both
        // alike. A preparation captures no graph, so `capture_plan` names the denoising buckets
        // alone.
        let steps = self.capture_order();
        let preparations: Vec<Bucket> = steps
            .iter()
            .map(|bucket| Bucket {
                variant: Variant::LatentPrepare,
                ..*bucket
            })
            .collect();
        steps.into_iter().chain(preparations).collect()
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

    fn claim_arena(&mut self) -> Result<()> {
        self.claim()
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
                self.write_step(*bucket)?;
                let (served, _) = self
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
                let eager = self.diag.eager();
                if eager {
                    self.eager_step(*bucket, *generation)?;
                }
                let (_, captured) = self
                    .lifecycle
                    .graphs
                    .captured_for(*bucket, *generation)
                    .ok_or_else(|| {
                        Error::Invariant(format!(
                            "no graph of bucket {bucket} at generation {generation}"
                        ))
                    })?;
                if !eager {
                    captured.launch(&stream)?;
                }
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
                // readable, and the wait is the worker's on the runner's stream.
                Ok(BatchOutput {
                    calls: self
                        .calls
                        .drain(..)
                        .map(|call| {
                            let state = RowState {
                                step_index: self.steps_completed,
                                ..RowState::default()
                            };
                            CallOutput::completed(call, Vec::new(), state)
                        })
                        .collect(),
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
                "a denoiser runner joins no expert step".into(),
            )),
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
            Job::Prepare { call, .. } => self.prepare(*call),
        }
    }

    fn offload_latents(&mut self, offloads: &[ipc::LatentOffload]) -> Result<bool> {
        if offloads.is_empty() {
            return Ok(false);
        }
        let Some(pool) = self.host_pool.clone() else {
            return Ok(false);
        };
        let page_bytes = pool.page_bytes();
        for offload in offloads {
            let Some((_, bucket)) = self
                .opened
                .filter(|(request, _)| *request == offload.request_key)
            else {
                return Err(Error::Invariant(format!(
                    "the engine asked for the trajectory of {:?} and this denoiser's columns hold \
                     {:?}'s: the denoiser is held for one request from its preparation to its \
                     last step, so a preempted request's trajectory is the one it holds",
                    offload.request_key,
                    self.opened.map(|(request, _)| request)
                )));
            };
            let regions = trajectory_regions(&self.latents(bucket, 0)?)?;
            let total: u64 = regions.iter().map(|(_, bytes)| bytes).sum();
            for copy in &offload.pages {
                let start = u64::from(copy.page) * page_bytes;
                if start >= total {
                    return Err(Error::Invalid(format!(
                        "latent page {} lies past a trajectory of {total} bytes in pages of \
                         {page_bytes}",
                        copy.page
                    )));
                }
                let destination = pool.address(copy)?;
                for (address, offset, bytes) in
                    pieces(&regions, start, page_bytes.min(total - start))?
                {
                    // SAFETY: the slot is reserved for this rank until the engine commits it after
                    // this command's output, which waits for the copies below; the columns are
                    // not written while the denoiser is held for the preempted request.
                    unsafe {
                        platform::memcpy_async(
                            destination + offset,
                            address,
                            bytes as usize,
                            &self.stream,
                        )
                    }?;
                }
            }
            tracing::info!(
                request = ?offload.request_key,
                pages = offload.pages.len(),
                "offloading a preempted trajectory to the host latent pool"
            );
        }
        self.stream.synchronize()?;
        Ok(true)
    }
}

/// How a runner's latent columns hold one request's trajectory, as the copy manifest library reads
/// it: the whole video latent, then the whole audio latent when the layout has a soundtrack, back to
/// back from the first byte. Every rank of a sequence group holds the whole samples (the velocity is
/// gathered to every rank before the solver's update), so two ranks or epochs whose layouts agree
/// lay out every page alike.
fn trajectory_layout(layout: &Layout, dtype: DType) -> Result<StorageLayout> {
    let element = dtype
        .bytes()
        .ok_or_else(|| Error::Invalid(format!("{dtype} has no whole-byte size")))?;
    let field = |tensor: &str, shape: &LatentShape, offset: u64| {
        let dims = [
            u64::from(shape.channels),
            u64::from(shape.frames),
            u64::from(shape.height),
            u64::from(shape.width),
        ];
        let strides = [dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1];
        Field {
            tensor: tensor.to_owned(),
            source: String::new(),
            dtype: dtype.to_string(),
            element,
            offset,
            shape: dims.to_vec(),
            strides: strides.to_vec(),
            logical: Logical {
                shape: dims.to_vec(),
                dims: ["channel", "frame", "row", "column"]
                    .map(str::to_owned)
                    .to_vec(),
                // No parallel configuration divides a sample: every rank holds it whole.
                units: vec![0; 4],
                segments: dims
                    .iter()
                    .map(|extent| {
                        vec![Segment {
                            span: Span::range(0, *extent),
                            local: 0,
                        }]
                    })
                    .collect(),
            },
            encoding: Encoding::Affine,
        }
    };
    let video = numel(&layout.video)? * element;
    let mut fields = vec![field("video_latents", &layout.video, 0)];
    let mut bytes = video;
    if let Some(audio) = &layout.audio {
        fields.push(field("audio_latents", audio, video));
        bytes += numel(audio)? * element;
    }
    Ok(StorageLayout { bytes, fields })
}

/// The device ranges a trajectory's bytes lie in, in the order [`trajectory_layout`] lays them back
/// to back: the video latent, then the audio latent.
fn trajectory_regions(latents: &Latents) -> Result<Vec<(u64, u64)>> {
    let region = |view: &TensorView| -> Result<(u64, u64)> {
        let bytes = view
            .dtype
            .bytes()
            .ok_or_else(|| Error::Invalid(format!("{} has no whole-byte size", view.dtype)))?;
        Ok((view.ptr.address(), view.shape.numel() * bytes))
    };
    let mut regions = vec![region(&latents.video)?];
    if let Some(audio) = &latents.audio {
        regions.push(region(audio)?);
    }
    Ok(regions)
}

/// The pieces of the byte range `start..start + bytes` of a trajectory laid back to back over
/// `regions`: each piece's device address, its offset from `start` and its length.
fn pieces(regions: &[(u64, u64)], start: u64, bytes: u64) -> Result<Vec<(u64, u64, u64)>> {
    let end = start + bytes;
    let mut pieces = Vec::new();
    let mut base = 0;
    for (address, length) in regions {
        let (first, last) = (start.max(base), end.min(base + length));
        if first < last {
            pieces.push((address + (first - base), first - start, last - first));
        }
        base += length;
    }
    if base < end {
        return Err(Error::Invalid(format!(
            "bytes {start}..{end} lie past a trajectory of {base} bytes"
        )));
    }
    Ok(pieces)
}

impl<M: Denoiser + Send + Sync + ?Sized> DenoiserRunner<M> {
    /// Captures one bucket of `generation`: the walk here, the graph kept by the lifecycle.
    fn capture(&mut self, bucket: Bucket, generation: u64) -> Result<()> {
        let kernels = self.lifecycle.generation_named(generation)?;
        let stream = self.stream.clone();
        self.fill_constants(&kernels)?;
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
    use nn::capability::{Seed, StateTable, VideoRequest};

    /// A trajectory's bytes lie back to back over the video and the audio column, so a range that
    /// straddles the two is two pieces, and a range past the trajectory is refused.
    #[test]
    fn a_range_across_the_two_latents_is_two_pieces() {
        let regions = [(0x1000_u64, 100_u64), (0x9000, 50)];
        assert_eq!(pieces(&regions, 0, 100).unwrap(), vec![(0x1000, 0, 100)]);
        assert_eq!(
            pieces(&regions, 90, 20).unwrap(),
            vec![(0x1000 + 90, 0, 10), (0x9000, 10, 10)]
        );
        assert_eq!(
            pieces(&regions, 140, 10).unwrap(),
            vec![(0x9000 + 40, 0, 10)]
        );
        assert!(pieces(&regions, 140, 11).is_err());
    }

    /// A trajectory stored by a rank of the same layout is restored page by page and whole: every
    /// copy is a whole stored page into the same page of the trajectory, and the pages cover every
    /// byte the trajectory holds.
    #[test]
    fn a_trajectory_of_the_same_layout_restores_whole_pages() {
        let mut layout = layout(24, 37, 48, 84, 1024);
        layout.audio = Some(LatentShape {
            channels: 64,
            frames: 251,
            height: 1,
            width: 1,
        });
        let stored = trajectory_layout(&layout, DType::F32).unwrap();
        stored.validate().unwrap();
        assert_eq!(stored.bytes, (24 * 37 * 48 * 84 + 64 * 251) * 4);
        let page = 1 << 20;
        let copies = restore::copies(
            &[restore::Stored {
                layout: stored.clone(),
                page_bytes: page,
            }],
            &restore::Stored {
                layout: stored.clone(),
                page_bytes: page,
            },
            [7; 32],
            [1; 16],
        )
        .unwrap();
        assert!(
            copies.iter().all(|copy| copy.offset == 0
                && copy.bytes == page
                && copy.target == copy.page * page),
            "a page laid out alike is copied whole into its own place: {copies:?}"
        );
        let mut targets: Vec<u64> = copies.iter().map(|copy| copy.target).collect();
        targets.sort_unstable();
        assert_eq!(
            targets,
            (0..stored.bytes.div_ceil(page))
                .map(|index| index * page)
                .collect::<Vec<_>>()
        );
    }

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

    /// A widening conditioner gets a `text` column wider than its `source` one, by exactly the
    /// width the model states - and a model that keeps the encoder's width gets one width.
    ///
    /// The stop this closes was `out is [19904, 5376], [19904, 5120] expected`. A gather cannot
    /// change a row's width (`head::gather_rows` requires `[count, table width]`), so a runner that
    /// sized both conditioning columns from the encoder's width handed the conditioner a buffer too
    /// narrow for its own output, and the refusal arrived one operator later than the mistake. Both
    /// models are asserted, so an implementation that always used one of the two numbers - either one
    /// - fails here rather than at a gather.
    #[test]
    fn a_widening_conditioner_writes_into_a_text_column_of_its_own_width() {
        // H3's refiner projects the encoder's 5120 columns into the stack's 5376.
        let widened = DenoiserShape {
            stack: DType::BF16,
            sample: DType::F32,
            width: 5120,
            text_width: 5376,
        };
        assert_eq!(conditioning_width(&widened), (5120, 5376));
        // A model with no conditioner, or one that keeps the encoder's width.
        let kept = DenoiserShape {
            text_width: 5120,
            ..widened
        };
        assert_eq!(conditioning_width(&kept), (5120, 5120));
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

    /// A denoiser that states nothing: the tests above exercise the runner's own arithmetic, which
    /// needs no model.
    struct NoModel;

    impl Denoiser for NoModel {
        fn layout(&self, _request: &VideoRequest<'_>) -> Layout {
            layout(1, 1, 1, 1, 1)
        }

        fn prepare_conditioning(
            &self,
            _f: &nn::Forward,
            _cond: Conditioning<'_>,
        ) -> nn::Result<()> {
            // The fixture has no conditioner: its denoiser reads the request's text rows alone.
            Ok(())
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

    /// A product store that answers for exactly the kinds it was given.
    ///
    /// It records nothing about how a caller asks: whether a kind is present is the store's
    /// answer, and how many questions a step put to it is not a fact about the deployment.
    struct Store {
        held: Vec<Product>,
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
            unreachable!("the presence rule reads existence, not a device and stream")
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
            held: vec![Product::VideoLatents],
        };
        let missing = missing_product(&store, request(), STEP_PRODUCTS).expect("a store answers");
        assert_eq!(
            missing,
            Some(Product::Text),
            "the encoder's rows are the first kind this store lacks"
        );
        let refusal = products::absent(request(), Product::Text).to_string();
        assert!(
            refusal.contains("text"),
            "the refusal names the kind: {refusal}"
        );
    }

    /// A store holding every kind a step reads has nothing missing, and a store holding none names
    /// the first kind the step would have read.
    #[test]
    fn a_store_holding_every_kind_leaves_nothing_missing() {
        let full = Store {
            held: STEP_PRODUCTS.to_vec(),
        };
        let missing = missing_product(&full, request(), STEP_PRODUCTS).expect("a store answers");
        assert_eq!(missing, None);

        let empty = Store { held: Vec::new() };
        let missing = missing_product(&empty, request(), STEP_PRODUCTS).expect("a store answers");
        assert_eq!(
            missing,
            STEP_PRODUCTS.first().copied(),
            "an empty store is refused for the first kind the step reads, not for the last"
        );
    }

    /// A staged row's own prompt replaces the capacity the plan captured, and the plan keeps the
    /// canvas: the refiner's keys and the rotary clock stop at the prompt the request presents.
    #[test]
    fn a_staged_request_states_its_own_prompt_length_within_the_plan() {
        let warm = WarmRequest {
            width: 768,
            height: 768,
            frames: 107,
            fps: 24,
            first_frames: 0,
            last_frames: 0,
            references: 0,
            text_tokens: 1024,
            conditions: Vec::new(),
            vision_spans: Vec::new(),
            condition_rows: None,
        };
        let conditioning = proto::video::Conditions {
            conditions: Vec::new(),
            prompt_tokens: 9,
            vision_spans: Vec::new(),
        };
        let stated = stated_request(
            &warm,
            &conditioning.encode().expect("a conditioning encodes"),
        )
        .expect("an unconditioned request states a request");
        assert_eq!(stated.text_tokens, 9);
        assert_eq!(
            (stated.width, stated.height, stated.frames),
            (768, 768, 107)
        );
        assert!(stated.conditions.is_empty());
        assert!(
            stated_request(&warm, b"not a conditioning").is_err(),
            "a blob that does not decode is refused, because the capture's capacity is no request"
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

        // A conditioned layout asks for the same one kind: the encoder's rows are the conditioner's
        // input, and no producer exists for a second name of them.
        layout.condition_elements = 3;
        assert_eq!(step_products(&layout), STEP_PRODUCTS);

        let store = Store {
            held: vec![Product::Text],
        };
        let missing =
            missing_product(&store, request(), step_products(&layout)).expect("a store answers");
        assert_eq!(
            missing, None,
            "a store holding the encoder's rows leaves a conditioned step nothing missing: the \
             conditioner reads those rows, and asking for `Source` refused a step for a product no \
             call publishes"
        );
    }

    /// A step with no store at all is a different fact from one whose store lacks a kind, and it says
    /// so: the first is the worker not having attached one, the second is a call that never ran.
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
