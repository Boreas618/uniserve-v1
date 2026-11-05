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
//! **The seam this runner needs from a batch, and its first form.** A step needs three things the
//! executor does not derive: the request's latents, its conditioning rows, and which step of the
//! schedule this batch is. `MediaColumns::inputs` is the field a media batch states its per-request
//! values in, and no runner reads it yet, so its meaning is stated here: **one `DenoiseRequest` per
//! request, in batch order, as JSON.** That is the seam's first form - everything crosses as host
//! bytes, the latents included - and its second form is device views once the worker's latent pool
//! exposes them. The change is `stage_request`'s body and nothing else.

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
use crate::runners::lifecycle::{Lifecycle, walk_capture};
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
    /// The element type the latents and the packed conditioning rows are stored in.
    pub dtype: DType,
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
            video: Column::new(device, shape.dtype, rows * video)?,
            audio: Column::new(device, shape.dtype, rows * audio)?,
            text: Column::new(device, shape.dtype, rows * text)?,
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
            rows * latent_bytes(&layout.video, shape.dtype)?,
        )?;
        let audio = match layout.audio {
            Some(audio) => latent_bytes(&audio, shape.dtype)?,
            None => shape.dtype.bytes().unwrap_or(1),
        };
        Staged::zero(&mut self.staged.audio, rows * audio)?;
        let width = (u64::from(layout.text_rows) + u64::from(layout.condition_rows)).max(1);
        let text = u64::from(shape.width.max(1));
        let bytes = shape
            .dtype
            .bytes()
            .ok_or_else(|| Error::Invalid(format!("{} has no whole-byte size", shape.dtype)))?;
        Staged::zero(&mut self.staged.text, rows * width * text * bytes)?;
        self.staged.step = vec![0u8; 8];
        Ok(())
    }
}

/// One request's facts and values, as a denoise batch stages them.
///
/// The bytes of this type are what `MediaColumns::inputs` carries, one per request in batch order.
/// It states everything a step reads that the runner cannot derive: which step of the schedule the
/// batch is, the request's seed, and the request's own latents and conditioning rows. The seed is
/// here rather than in the batch's columns because a draw is a request's, and the runner passes it
/// on unchanged (`Denoiser::draw_noise` is the model's; this runner never draws).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct DenoiseRequest {
    /// The zero-based step of the schedule this batch is.
    pub step: u32,
    /// The request's own seed, for the draw a first step makes.
    pub seed: u64,
    /// The video latents, `[channels, frames, height, width]`, row-major in the layout's element
    /// type.
    pub video: Vec<u8>,
    /// The audio latents when the task has a soundtrack, empty otherwise.
    pub audio: Vec<u8>,
    /// The encoded conditioning rows, `[text_rows + condition_rows, width]`.
    pub text: Vec<u8>,
}

impl DenoiseRequest {
    /// Decodes the bytes one request's row of `MediaColumns::inputs` holds.
    ///
    /// The row is the request's own JSON document and nothing else: a row that carries two buffers,
    /// or bytes that are not a document, is refused by name rather than served with a step index
    /// nobody stated.
    pub fn decode(row: &[Vec<u8>]) -> Result<Self> {
        let [bytes] = row else {
            return Err(Error::Invalid(format!(
                "a denoise request's row carries {} buffer(s); it states one document",
                row.len()
            )));
        };
        serde_json::from_slice(bytes).map_err(|error| {
            Error::Invalid(format!(
                "a denoise request's row is not the document this runner reads: {error}"
            ))
        })
    }
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
    /// The block-by-block diagnosis, off unless `UNISERVE_DIAG_DIR` names a directory.
    diag: Diag,
    /// The step of the schedule the staged batch is, which names the diagnosis dumps.
    staged_step: Option<u32>,
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
            diag: Diag::from_env(),
            staged_step: None,
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
        let rows = (u64::from(layout.text_rows) + u64::from(layout.condition_rows)).max(1);
        let width = u64::from(self.shape.width.max(1));
        let view = self.columns.text.shaped(&[rows * width]);
        let text = view
            .at(view.offset_elements(u64::from(index) * rows * width))
            .reshape(Shape::new(&[rows, width]));
        Ok(Conditioning {
            text,
            source: None,
            media,
            images: None,
            audio: None,
            state,
            step: Some(self.columns.step.shaped(&[1])),
        })
    }

    /// Stages one request's row of a batch into the host side of the columns.
    fn stage_request(
        &mut self,
        bucket: Bucket,
        index: u32,
        request: &DenoiseRequest,
    ) -> Result<()> {
        let layout = layout_of(bucket)?;
        let expected = latent_bytes(&layout.video, self.shape.dtype)?;
        if request.video.len() as u64 != expected {
            return Err(Error::Invalid(format!(
                "a step of bucket {bucket} carries {} byte(s) of video latents and the layout \
                 states {expected}",
                request.video.len()
            )));
        }
        let offset = u64::from(index) * expected;
        Staged::place(
            &mut self.columns.staged.video,
            offset,
            &request.video,
            "video latents",
        )?;
        if let Some(audio) = layout.audio {
            let bytes = latent_bytes(&audio, self.shape.dtype)?;
            if request.audio.len() as u64 != bytes {
                return Err(Error::Invalid(format!(
                    "a step of bucket {bucket} carries {} byte(s) of audio latents and the layout \
                     states {bytes}",
                    request.audio.len()
                )));
            }
            let offset = u64::from(index) * bytes;
            Staged::place(
                &mut self.columns.staged.audio,
                offset,
                &request.audio,
                "audio latents",
            )?;
        }
        let rows = (u64::from(layout.text_rows) + u64::from(layout.condition_rows)).max(1);
        let width = u64::from(self.shape.width.max(1));
        let element = self.shape.dtype.bytes().ok_or_else(|| {
            Error::Invalid(format!("{} has no whole-byte size", self.shape.dtype))
        })?;
        let bytes = rows * width * element;
        if request.text.len() as u64 != bytes {
            return Err(Error::Invalid(format!(
                "a step of bucket {bucket} carries {} byte(s) of conditioning rows and the layout \
                 states {bytes}",
                request.text.len()
            )));
        }
        let offset = u64::from(index) * bytes;
        Staged::place(
            &mut self.columns.staged.text,
            offset,
            &request.text,
            "conditioning rows",
        )?;
        Ok(())
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

    /// Copies the staged host sides to the device, in one place, so a capture and a replay agree on
    /// what the graph reads.
    fn upload(&mut self) -> Result<()> {
        let stream = self.stream.clone();
        // The host side and the device column are two fields of one value, so the copy borrows
        // them apart rather than cloning a buffer the graph is about to read anyway.
        let DenoiseColumns {
            video,
            audio,
            text,
            step,
            staged,
        } = &mut self.columns;
        video.write(&staged.video)?;
        audio.write(&staged.audio)?;
        text.write(&staged.text)?;
        step.write(&staged.step)?;
        self.columns.video.upload(&stream)?;
        self.columns.audio.upload(&stream)?;
        self.columns.text.upload(&stream)?;
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
        let mut step = None;
        for (index, row) in media.inputs.iter().enumerate() {
            let request = DenoiseRequest::decode(row)?;
            let index = u32::try_from(index)
                .map_err(|_| Error::Invalid("a batch of more requests than rows".into()))?;
            self.stage_request(bucket, index, &request)?;
            // Every row of one batch is the same step of the same schedule: a batch is one step
            // (`Variant::Denoise`), so a row that names another step describes another batch.
            let stated = steps.get(request.step as usize).copied().ok_or_else(|| {
                Error::Invalid(format!(
                    "a request of bucket {bucket} states step {} and the schedule has {}",
                    request.step,
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
                let step = self.staged_step;
                if let Some(step) = step {
                    self.dump_latents(*bucket, step);
                }
                let launched = std::time::Instant::now();
                captured.launch(&stream)?;
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
        }
    }
}

impl<M: Denoiser + Send + Sync> DenoiserRunner<M> {
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
    fn a_row_of_inputs_states_one_document() {
        let request = DenoiseRequest {
            step: 3,
            seed: 7,
            video: vec![0u8; 8],
            audio: Vec::new(),
            text: vec![1u8; 4],
        };
        let bytes = serde_json::to_vec(&request).unwrap();
        let decoded = DenoiseRequest::decode(&[bytes]).unwrap();
        assert_eq!(decoded.step, 3);
        assert_eq!(decoded.seed, 7);
        assert_eq!(decoded.video.len(), 8);
    }

    #[test]
    fn a_row_of_two_buffers_names_what_it_carries() {
        let error = DenoiseRequest::decode(&[Vec::new(), Vec::new()]).unwrap_err();
        assert!(matches!(error, Error::Invalid(_)), "{error:?}");
        assert!(format!("{error}").contains("carries 2 buffer(s)"));
    }

    #[test]
    fn bytes_that_are_not_a_document_are_refused_by_name() {
        let error = DenoiseRequest::decode(&[b"not a document".to_vec()]).unwrap_err();
        assert!(format!("{error}").contains("not the document this runner reads"));
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
