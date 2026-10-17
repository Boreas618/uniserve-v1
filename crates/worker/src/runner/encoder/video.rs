//! A video model's encoders: the runners of its text encoder component (the
//! prompt encoder and, for conditioned tasks, the vision tower bound with it),
//! its condition latent encoders ([`Conditions`]), and the encoding of a
//! request's prompt, vision blocks and conditions into denoiser rows.

use std::collections::HashMap;
use std::sync::Arc;

use tvm_ffi::Any;
use uniserve_core::{Canvas, VideoCondition};
use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{Batch as BatchPlan, CallKind, MediaCall};

use super::{
    ENCODER, EncoderConfig, EncoderInfo, EncoderMethods, EncoderRunner, OwnedContext, PromptVision,
    TextSize,
};
use crate::config::Device;
use crate::executor::batch::CallState;
use crate::executor::products::{OutputShape, Placement, bind_outputs};
use crate::executor::video::{
    MediaBatch, Parts, VideoCall, bind_rows, decode_range, product_name, publish, publish_bound,
    row_width, stream_of,
};
use crate::ffi::{DType, Function, Handle, Interpreter, integer_list};
use crate::lane::{Completion, Job};
use crate::runner::RunnerSetup;
use crate::runner::methods::{Argument, MethodCall, Methods, RunnerClass};
use crate::storage::PinnedPool;
use crate::storage::memory::StorageView;
use crate::{Error, Result};

/// A condition encoding was recorded before the job that constructs its
/// encoder completed.
const MISSING_ENCODER: Error =
    Error::State("a condition encoding was recorded before its encoder was constructed");

/// The condition latent encoders of a video model: the Python runners of
/// its video encoder and of its audio encoder, one per input shapes.
pub(super) struct Conditions {
    video: RunnerClass,
    audio: RunnerClass,
    loaded: Handle,
    /// Frames of each visual unit of a condition of some frame count.
    unit_frames: Function,
    /// What `unit_frames` answered, by frame count.
    frames: HashMap<i64, Vec<i64>>,
    /// The runner of each call's input shapes, constructed by the job
    /// before the first call's ([`Self::setup`]).
    video_runners: HashMap<Vec<i64>, Methods>,
    audio_runners: HashMap<Vec<i64>, Methods>,
}

/// The encoder of a condition's medium.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConditionMedium {
    Video,
    Audio,
}

/// What a condition encoding call encodes on this rank.
pub(crate) enum ConditionInput {
    /// Visual units, each `[pixel_offset, frames, width, height, unit]`,
    /// written to rows `region` of the round's `rows`.
    Units {
        units: Vec<[i64; 5]>,
        rows: u64,
        region: std::ops::Range<u64>,
    },
    /// Audio tracks of these sample counts, written to `rows` rows.
    Tracks { lengths: Vec<u64>, rows: u64 },
}

/// What the job on the lane asks the library for before a batch's
/// condition encodings are recorded.
pub(crate) enum ConditionSetup {
    /// The frames of each visual unit of conditions of these frame counts.
    Frames(Vec<i64>),
    /// The encoders of these input shapes.
    Runners(Vec<(ConditionMedium, Vec<i64>)>),
}

impl Conditions {
    fn new(video: RunnerSetup<'_>, audio: RunnerSetup<'_>) -> Result<Self> {
        Ok(Self {
            unit_frames: Interpreter::import(&format!("{ENCODER}.unit_frames"))?,
            loaded: video.loaded.clone(),
            video: RunnerClass::new(ENCODER, "encode", video.arguments())?,
            audio: RunnerClass::new(ENCODER, "encode", audio.arguments())?,
            frames: HashMap::new(),
            video_runners: HashMap::new(),
            audio_runners: HashMap::new(),
        })
    }

    /// The input shapes of a call encoding `units`: each unit's frames at
    /// its raster. `None` while the unit frames of one of the conditions
    /// were not asked for.
    fn unit_shapes(&self, units: &[[i64; 5]]) -> Result<Option<Vec<i64>>> {
        let mut shapes = Vec::with_capacity(units.len() * 4);
        for &[_, frames, width, height, unit] in units {
            let Some(lengths) = self.frames.get(&frames) else {
                return Ok(None);
            };
            let length = lengths.get(unit as usize).copied().ok_or_else(|| {
                Error::Invalid("a condition unit lies outside its condition".into())
            })?;
            shapes.extend([length, height, width, 3]);
        }
        Ok(Some(shapes))
    }

    /// The input shapes of a call encoding tracks of `lengths` samples.
    fn track_shapes(lengths: &[u64]) -> Vec<i64> {
        lengths
            .iter()
            .flat_map(|&length| [length as i64, 2])
            .collect()
    }

    /// Record into `job` what the encodings of `inputs` lack: the unit
    /// frames of conditions whose frame count is new, and once those are
    /// known the construction of the encoders of new input shapes. `None`
    /// when the encodings can be recorded.
    ///
    /// Both are calls of the library, so the job makes them; a constructed
    /// encoder is an argument of its forward, which is why the forward is
    /// recorded only from the job's completion ([`Self::accept`]).
    fn setup(&self, inputs: &[ConditionInput], job: &Job) -> Result<Option<ConditionSetup>> {
        let mut frames: Vec<i64> = inputs
            .iter()
            .filter_map(|input| match input {
                ConditionInput::Units { units, .. } => Some(units),
                ConditionInput::Tracks { .. } => None,
            })
            .flatten()
            .map(|unit| unit[1])
            .filter(|count| !self.frames.contains_key(count))
            .collect();
        frames.sort_unstable();
        frames.dedup();
        if !frames.is_empty() {
            for &count in &frames {
                job.call_returning(MethodCall::new(
                    &self.unit_frames,
                    vec![
                        Argument::Handle(self.loaded.clone()),
                        Argument::Integer(count),
                    ],
                ));
            }
            return Ok(Some(ConditionSetup::Frames(frames)));
        }

        let mut runners: Vec<(ConditionMedium, Vec<i64>)> = Vec::new();
        for input in inputs {
            let (medium, shapes) = match input {
                ConditionInput::Units { units, .. } => {
                    let shapes = self.unit_shapes(units)?.ok_or_else(|| {
                        Error::Invariant("a condition's unit frames were not asked for".into())
                    })?;
                    (ConditionMedium::Video, shapes)
                }
                ConditionInput::Tracks { lengths, .. } => {
                    (ConditionMedium::Audio, Self::track_shapes(lengths))
                }
            };
            let (class, constructed) = match medium {
                ConditionMedium::Video => (&self.video, &self.video_runners),
                ConditionMedium::Audio => (&self.audio, &self.audio_runners),
            };
            let recorded = runners
                .iter()
                .any(|(other, known)| *other == medium && *known == shapes);
            if constructed.contains_key(&shapes) || recorded {
                continue;
            }
            job.call_handle(class.construction(vec![Argument::Integers(shapes.clone())]));
            runners.push((medium, shapes));
        }
        Ok((!runners.is_empty()).then_some(ConditionSetup::Runners(runners)))
    }

    /// Keep what the job of `setup` returned.
    fn accept(&mut self, setup: ConditionSetup, completion: &mut Completion) -> Result<()> {
        match setup {
            ConditionSetup::Frames(frames) => {
                for count in frames {
                    self.frames.insert(count, completion.returned()?);
                }
            }
            ConditionSetup::Runners(runners) => {
                for (medium, shapes) in runners {
                    let runner = completion.handle()?;
                    match medium {
                        ConditionMedium::Video => self
                            .video_runners
                            .insert(shapes, self.video.constructed(runner)),
                        ConditionMedium::Audio => self
                            .audio_runners
                            .insert(shapes, self.audio.constructed(runner)),
                    };
                }
            }
        }
        Ok(())
    }

    /// Encode visual units, each `[pixel_offset, frames, width, height,
    /// unit]`, into consecutive rows of `out`.
    fn encode_video(
        &mut self,
        pixels: &StorageView,
        units: &[[i64; 5]],
        out: &StorageView,
        job: &Job,
    ) -> Result<()> {
        let runner = self
            .unit_shapes(units)?
            .and_then(|shapes| self.video_runners.get(&shapes))
            .ok_or(MISSING_ENCODER)?;
        job.call(MethodCall::new(
            &self.video.forward,
            vec![
                Argument::Handle(runner.runner().clone()),
                Argument::Tensor(pixels.clone()),
                Argument::Tensor(out.clone()),
                Argument::Integers(units.iter().flatten().copied().collect()),
            ],
        ));
        Ok(())
    }

    /// Encode audio tracks of `lengths` samples into consecutive rows of
    /// `out`.
    fn encode_audio(
        &mut self,
        samples: &StorageView,
        lengths: &[u64],
        out: &StorageView,
        job: &Job,
    ) -> Result<()> {
        let runner = self
            .audio_runners
            .get(&Self::track_shapes(lengths))
            .ok_or(MISSING_ENCODER)?;
        job.call(MethodCall::new(
            &self.audio.forward,
            vec![
                Argument::Handle(runner.runner().clone()),
                Argument::Tensor(samples.clone()),
                Argument::Tensor(out.clone()),
                Argument::Integers(lengths.iter().map(|&length| length as i64).collect()),
            ],
        ));
        Ok(())
    }

    pub(crate) fn kernel_calls(&self) -> Vec<MethodCall> {
        let mut calls = self.video.kernel_calls(self.video_runners.values());
        calls.extend(self.audio.kernel_calls(self.audio_runners.values()));
        calls
    }

    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let video = self.video.close(aborted, self.video_runners.values());
        video.and(self.audio.close(aborted, self.audio_runners.values()))
    }
}

/// What a video model's text encoder component is bound with.
pub(crate) struct TextSetup<'a> {
    pub(crate) loaded: &'a Handle,
    pub(crate) component: &'a str,
    /// The component's root module path.
    pub(crate) path: &'a str,
    pub(crate) device: DeviceId,
    pub(crate) stream: Stream,
    /// The stream's shared work areas, a Python `Scratch`.
    pub(crate) scratch: Handle,
    pub(crate) attention: String,
    /// Prompt capacities, the longest admitted prompt and, as the row
    /// budget, the most vision patch rows one request carries.
    pub(crate) config: EncoderConfig,
    /// Whether the component also encodes condition media with its vision
    /// tower.
    pub(crate) vision: bool,
    /// The model dtype the encoders compute in.
    pub(crate) dtype: DType,
}

impl EncoderRunner {
    /// The runners of a video model's text encoder component: the prompt
    /// encoder, which owns the execution context, and, for conditioned
    /// tasks, the vision tower bound with it.
    pub(crate) fn text(setup: TextSetup<'_>) -> Result<(Self, Option<Self>)> {
        let TextSetup {
            loaded,
            component,
            path,
            device,
            stream,
            scratch,
            attention,
            config,
            vision,
            dtype,
        } = setup;
        let text_info = EncoderInfo::describe(loaded, path, dtype)?;
        let vision_info = if vision {
            Some(EncoderInfo::describe(
                loaded,
                &format!("{path}.vision"),
                dtype,
            )?)
        } else {
            None
        };
        // One vision call encodes every vision block of a request. Their
        // patch rows together are within the row budget, and a block holds
        // at least one feature, so the budget also bounds the blocks.
        let config = match &vision_info {
            Some(EncoderInfo::Vision(tower)) => EncoderConfig {
                max_images: (config.row_budget / (tower.downsample as usize).pow(2)).max(1),
                ..config
            },
            _ => config,
        };
        // The context serves both encoders, so its operators are prepared
        // for the larger of their workloads: a prompt padded to the largest
        // capacity, and a request's vision blocks.
        let size = [
            text_info.context_size(&config),
            vision_info.and_then(|info| info.context_size(&config)),
        ]
        .into_iter()
        .flatten()
        .reduce(|left, right| TextSize {
            tokens: left.tokens.max(right.tokens),
            rows: left.rows.max(right.rows),
        })
        .ok_or_else(|| Error::Invariant("a text encoder declares its size".into()))?;
        let method = |name: &str| Interpreter::import(&format!("{ENCODER}.{name}"));
        let runner = Handle::new(Interpreter::import(ENCODER)?.call_values(&[
            loaded.to_any(),
            Any::from(tvm_ffi::String::from(path)),
            Any::from(stream.raw() as i64),
            Any::from(tvm_ffi::String::from(attention.as_str())),
            scratch.to_any(),
            Any::from(integer_list([size.tokens as i64, size.rows as i64])),
        ])?)?;
        let device = Device::Cuda(device);
        let vision = vision_info
            .map(|info| {
                let tower = Handle::new(method("tower")?.call_values(&[
                    runner.to_any(),
                    Any::from(tvm_ffi::String::from("vision")),
                ])?)?;
                EncoderRunner::over(
                    component,
                    device,
                    tower,
                    "vision",
                    info,
                    &config,
                    stream.clone(),
                )
            })
            .transpose()?;
        let mut text =
            EncoderRunner::over(component, device, runner, "", text_info, &config, stream)?;
        text.context = Some(OwnedContext {
            kernels: method("kernels")?,
            close: method("close")?,
        });
        Ok((text, vision))
    }

    /// The runner of a video model's condition latent encoders. `video`
    /// and `audio` name the encoders' module paths; the other fields of
    /// the setups agree.
    pub(crate) fn conditions(video: RunnerSetup<'_>, audio: RunnerSetup<'_>) -> Result<Self> {
        Ok(Self {
            owner: video.path.to_owned(),
            info: EncoderInfo::Condition,
            device: Device::Cuda(video.device),
            stream: Some(video.stream.clone()),
            methods: EncoderMethods::new(Handle::none())?,
            packed: None,
            pinned: PinnedPool::new(Some(video.device)),
            resident: None,
            noise: None,
            text: None,
            text_capacities: Vec::new(),
            config: EncoderConfig::default(),
            graphs: None,
            capturing: false,
            context: None,
            conditions: Some(Box::new(Conditions::new(video, audio)?)),
        })
    }

    fn condition_encoders(&mut self) -> Result<&mut Conditions> {
        self.conditions
            .as_deref_mut()
            .ok_or_else(|| Error::Invalid(format!("{} encodes no conditions", self.owner)))
    }

    /// Record into `job` what the condition encodings of `inputs` lack
    /// before they can be recorded; see [`Conditions::setup`].
    fn condition_setup(
        &mut self,
        inputs: &[ConditionInput],
        job: &Job,
    ) -> Result<Option<ConditionSetup>> {
        self.condition_encoders()?.setup(inputs, job)
    }

    /// Keep what the job of `setup` returned.
    fn accept_setup(&mut self, setup: ConditionSetup, completion: &mut Completion) -> Result<()> {
        self.condition_encoders()?.accept(setup, completion)
    }

    /// Encode a condition's visual units, each `[pixel_offset, frames,
    /// width, height, unit]`, into consecutive rows of `out`.
    pub(crate) fn encode_condition_units(
        &mut self,
        pixels: &StorageView,
        units: &[[i64; 5]],
        out: &StorageView,
        job: &Job,
    ) -> Result<()> {
        self.condition_encoders()?
            .encode_video(pixels, units, out, job)
    }

    /// Encode condition audio tracks of `lengths` samples into consecutive
    /// rows of `out`.
    pub(crate) fn encode_condition_tracks(
        &mut self,
        samples: &StorageView,
        lengths: &[u64],
        out: &StorageView,
        job: &Job,
    ) -> Result<()> {
        self.condition_encoders()?
            .encode_audio(samples, lengths, out, job)
    }
}

/// The patch grids of a request's vision blocks, images and videos apart,
/// each in request order as the text encoder places them.
fn vision_grids(conditions: &[VideoCondition]) -> (Vec<[u32; 3]>, Vec<[u32; 3]>) {
    let mut images = Vec::new();
    let mut videos = Vec::new();
    for condition in conditions {
        let Some(vision) = &condition.vision else {
            continue;
        };
        let grid = [vision.grid.t, vision.grid.h, vision.grid.w];
        match condition.media {
            uniserve_core::ConditionMedia::Video { .. } => videos.push(grid),
            _ => images.push(grid),
        }
    }
    (images, videos)
}

/// Encode a request's prompt into its `conditioning` rows, splicing the
/// vision features of its condition media in at their placeholders when the
/// call reads them. Every rank of the text encoder computes the features
/// and publishes its copy.
pub(crate) fn encode_text(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
    job: &Job,
) -> Result<()> {
    let call = &plan.calls[index];
    if call.outputs.len() != 1 || call.inputs.len() > 1 {
        return Err(Error::Invalid(
            "text encoding reads at most one vision product and writes its conditioning".into(),
        ));
    }
    let (tokens, conditions) = {
        let request = parts.video.request(slot, call.request_key)?;
        (Arc::clone(&request.tokens), Arc::clone(&request.conditions))
    };
    if tokens.is_empty() {
        return Err(Error::Invalid(
            "text conditioning requires admitted prompt tokens".into(),
        ));
    }
    let width = row_width(&call.outputs[0])?;
    let stream = stream_of(batch)?;
    let issue = job.on(Some(&stream));
    let visual = call
        .inputs
        .first()
        .map(|input| batch.read(parts.owners().store, input, call.call_id, issue))
        .transpose()?;
    let write = bind_rows(
        parts,
        batch,
        &call.outputs[0],
        vec![tokens.len() as u64, width],
        placement,
    )?;

    let text = &mut parts
        .encoders
        .text
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no text encoder".into()))?
        .runner;
    let (image_grids, video_grids) = vision_grids(&conditions);
    let vision = visual.as_ref().map(|read| PromptVision {
        image_grids: &image_grids,
        video_grids: &video_grids,
        rows: read.value(),
    });
    let mode = text.encode_text(&tokens, &write.view(), vision, job)?;
    batch.record(mode);
    publish(
        parts,
        batch,
        call,
        index,
        &write,
        &[tokens.len() as u64, width],
        issue,
    )
}

/// Encode the vision blocks of a request's condition media into one
/// `vision_features` row per vision placeholder, in request order.
pub(crate) fn encode_vision(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
    job: &Job,
) -> Result<()> {
    let call = &plan.calls[index];
    if call.inputs.len() != 1 || call.outputs.len() != 1 {
        return Err(Error::Invalid(
            "vision encoding requires one input and one output".into(),
        ));
    }
    let conditions = Arc::clone(&parts.video.request(slot, call.request_key)?.conditions);
    let blocks: Vec<([u32; 3], u64)> = conditions
        .iter()
        .filter_map(|condition| condition.vision.as_ref())
        .map(|vision| {
            (
                [vision.grid.t, vision.grid.h, vision.grid.w],
                u64::from(vision.tokens),
            )
        })
        .collect();
    if blocks.is_empty() {
        return Err(Error::Invalid(
            "vision encoding requires a vision block".into(),
        ));
    }
    let width = row_width(&call.outputs[0])?;
    let tokens: u64 = blocks.iter().map(|block| block.1).sum();
    let stream = stream_of(batch)?;
    let issue = job.on(Some(&stream));
    let patches = batch.read(parts.owners().store, &call.inputs[0], call.call_id, issue)?;
    let write = bind_rows(
        parts,
        batch,
        &call.outputs[0],
        vec![tokens, width],
        placement,
    )?;

    let vision = parts
        .encoders
        .vision
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank's text encoder reads no vision".into()))?;
    // Each block's features are consecutive rows of the product.
    let grids: Vec<[u32; 3]> = blocks.iter().map(|block| block.0).collect();
    vision.encode_patches(patches.value(), &grids, &write.view(), job)?;
    publish(parts, batch, call, index, &write, &[tokens, width], issue)
}

/// What condition encoding `call` encodes on this rank: a round of visual
/// units, this rank's of them, or every audio track. `None` for an audio
/// call on a rank that is not the component's first.
fn condition_input(
    parts: &Parts<'_>,
    plan: &BatchPlan,
    call: &uniserve_ipc::Call,
    slot: usize,
) -> Result<Option<ConditionInput>> {
    if call.inputs.len() != 1 || call.outputs.len() != 1 {
        return Err(Error::Invalid(
            "condition encoding requires one input and one output".into(),
        ));
    }
    let name = product_name(parts.info, call, &call.outputs[0])?;
    let conditions = &parts.video.request(slot, call.request_key)?.conditions;
    if conditions.is_empty() {
        return Err(Error::Invalid(
            "condition encoding requires admitted conditions".into(),
        ));
    }
    let latent = parts
        .encoders
        .latent
        .as_ref()
        .ok_or_else(|| Error::Invalid("this rank holds no latent encoder".into()))?;
    match name {
        "condition_video_latents" => {
            let range = decode_range(plan, call)?;
            // Every visual unit as (condition, unit, rows), in encoding order.
            let all: Vec<(usize, u32, u64)> = conditions
                .iter()
                .enumerate()
                .flat_map(|(index, condition)| {
                    condition
                        .latent_units
                        .iter()
                        .enumerate()
                        .map(move |(unit, &rows)| (index, unit as u32, u64::from(rows)))
                })
                .collect();
            let (cursor, count) = (range.cursor as usize, range.max_units as usize);
            if count == 0 || cursor + count > all.len() {
                return Err(Error::Invalid(
                    "condition encoding exceeds the admitted units".into(),
                ));
            }
            let round = &all[cursor..cursor + count];
            let run = latent.units(parts.rank, 0, count as u32)?;
            if run.is_empty() {
                return Err(Error::Invalid(
                    "condition encoding assigns this rank no unit".into(),
                ));
            }
            let rows: u64 = round.iter().map(|unit| unit.2).sum();
            let first: u64 = round[..run.start as usize].iter().map(|unit| unit.2).sum();
            let last: u64 = round[..run.end as usize].iter().map(|unit| unit.2).sum();
            let mut offsets = Vec::new();
            let mut cursor_pixels = 0u64;
            for condition in conditions.iter() {
                offsets.push(cursor_pixels);
                cursor_pixels += condition.pixel_bytes() / 3;
            }
            let units: Vec<[i64; 5]> = round[run.start as usize..run.end as usize]
                .iter()
                .map(|&(condition, unit, _)| {
                    let (frames, canvas) = conditions[condition].media.pixels().unwrap_or((
                        0,
                        Canvas {
                            width: 0,
                            height: 0,
                        },
                    ));
                    [
                        offsets[condition] as i64,
                        i64::from(frames),
                        i64::from(canvas.width),
                        i64::from(canvas.height),
                        i64::from(unit),
                    ]
                })
                .collect();
            Ok(Some(ConditionInput::Units {
                units,
                rows,
                region: first..last,
            }))
        }
        "condition_audio_latents" => {
            // The audio call is one unit, the component's first rank's.
            if latent.units(parts.rank, 0, 1)?.is_empty() {
                return Ok(None);
            }
            let lengths: Vec<u64> = conditions
                .iter()
                .filter_map(|condition| condition.media.audio().map(|track| track.samples))
                .collect();
            let rows: u64 = conditions
                .iter()
                .map(|condition| u64::from(condition.audio_rows))
                .sum();
            Ok(Some(ConditionInput::Tracks { lengths, rows }))
        }
        other => Err(Error::Invalid(format!(
            "latent encoding does not produce {other}"
        ))),
    }
}

/// Record into `job` what the batch's condition encodings lack before
/// their forwards can be recorded: answers and encoders the library gives
/// for input shapes this rank has not encoded before. `None` when the
/// batch's forwards can be recorded; otherwise the job ends here and
/// [`accept_conditions`] takes its completion.
pub(crate) fn setup_conditions(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    calls: &[CallState<VideoCall>],
    job: &Job,
) -> Result<Option<ConditionSetup>> {
    let mut inputs = Vec::new();
    for (call, state) in plan.calls.iter().zip(calls) {
        // A skipped call encodes nothing.
        if call.code != CallKind::Media(MediaCall::LatentEncoding) || !state.active {
            continue;
        }
        inputs.extend(condition_input(parts, plan, call, state.slot)?);
    }
    if inputs.is_empty() {
        return Ok(None);
    }
    let latent = parts
        .encoders
        .latent
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no latent encoder".into()))?;
    latent.runner.condition_setup(&inputs, job)
}

/// Keep what the job of `setup` returned for the batch's condition
/// encodings.
pub(crate) fn accept_conditions(
    parts: &mut Parts<'_>,
    setup: ConditionSetup,
    completion: &mut Completion,
) -> Result<()> {
    let latent = parts
        .encoders
        .latent
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no latent encoder".into()))?;
    latent.runner.accept_setup(setup, completion)
}

/// Encode a request's conditions into denoiser rows: a round of visual
/// units, this rank's of them, or every audio track.
pub(crate) fn encode_conditions(
    parts: &mut Parts<'_>,
    plan: &BatchPlan,
    batch: &mut MediaBatch,
    index: usize,
    slot: usize,
    placement: &Placement<'_>,
    job: &Job,
) -> Result<()> {
    let call = &plan.calls[index];
    let Some(input) = condition_input(parts, plan, call, slot)? else {
        return Ok(());
    };
    let owners = parts.owners();
    let stream = stream_of(batch)?;
    let issue = job.on(Some(&stream));
    let width = row_width(&call.outputs[0])?;
    let (rows, region) = match &input {
        ConditionInput::Units { rows, region, .. } => (*rows, Some(vec![region.clone(), 0..width])),
        ConditionInput::Tracks { rows, .. } => (*rows, None),
    };
    let read = batch.read(owners.store, &call.inputs[0], call.call_id, issue)?;
    let write = bind_outputs(
        owners.store,
        &[OutputShape {
            product: call.outputs[0].clone(),
            shape: vec![rows, width],
            region,
        }],
        batch.target(),
        placement,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| Error::Invariant("condition rows were not bound".into()))?;
    batch.batch.writes.push(write.clone());

    let latent = parts
        .encoders
        .latent
        .as_mut()
        .ok_or_else(|| Error::Invalid("this rank holds no latent encoder".into()))?;
    match &input {
        ConditionInput::Units { units, .. } => {
            latent
                .runner
                .encode_condition_units(read.value(), units, &write.view(), job)?;
        }
        ConditionInput::Tracks { lengths, .. } => {
            latent
                .runner
                .encode_condition_tracks(read.value(), lengths, &write.view(), job)?;
        }
    }
    publish_bound(parts, batch, call, index, &write, false, issue)
}

// Image vision and latent encoding batches (`VisionEncoding`,
// `LatentEncoding`): each active call encodes one image into its
// `encoder_output` feature product, an inline request image the host lane
// prepared before the batch launched, or a resident image product (feedback
// of a generated image), which the encoder prepares on its stream. Features
// are written in place into the product's reserved storage and published
// with the image's canvas dimensions (`Metadata::Feature`).
