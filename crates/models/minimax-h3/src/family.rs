//! MiniMax-H3 and FastH3 as execution loads them and request processing serves them ([`nn::family`]).
//!
//! [`FAMILY`] claims the `MiniMaxH3ModularPipeline` class. A checkpoint opens to its [`Config`] (an
//! export and the base release it pins read as one), a load places the stages the deployment names
//! with the deployment's sequence and tensor splits, and the loaded [`MinimaxH3`] states the model's
//! own rules to execution: the `17 * n + 5` timeline, its canvases, the requests its denoiser graphs
//! are captured for, and the description its request processing admits requests against.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use nn::capability::{Decoder, Encoder, ImageShape, MediaFormat, Splicer, WarmPrompt, WarmRequest};
use nn::family::{
    ConditionBounds, Description, Family, Model, Stage, Stages, VideoCheckpoint, VideoFamily,
    VideoModel, VideoPlacement, VideoServing,
};
use nn::{Checkpoint, DType, WeightLoader};
use processing::minimax_h3::{ConditionPacking, DenoiserInfo, PreparedCondition, REFERENCE_LIMITS};
use processing::qwen_vl::ProcessorConfig;
use proto::video::Canvas;
use qwen3::Qwen3VlModel;

use crate::audio_vae::AudioDecoder;
use crate::config::{
    AUDIO_CHANNELS, AUDIO_HOP, AUDIO_SAMPLE_RATE, CANVAS_RULE, FPS, FRAMES, MIN_SECONDS,
    RETAINED_LAYERS, named_canvases,
};
use crate::denoiser::Denoiser;
use crate::output::VideoPostprocessor;
use crate::video_vae::VideoDecoder;
use crate::{Config, Error, Result, VideoTask};

/// The MiniMax-H3 family: a diffusers modular pipeline, which request processing serves through its
/// video processor ([`processing::minimax_h3::H3Family`]).
pub const FAMILY: Family = Family {
    request: processing::minimax_h3::FAMILY,
    model: Model::Video(&MinimaxH3Family),
};

/// The family's value, which [`FAMILY`] refers to.
struct MinimaxH3Family;

impl VideoFamily for MinimaxH3Family {
    fn open(&self, checkpoint: &Checkpoint) -> nn::Result<Box<dyn VideoCheckpoint>> {
        // A checkpoint is one directory, or an export and the base release it pins.
        if checkpoint.roots().len() > 2 {
            return Err(nn::Error::Unsupported(format!(
                "a MiniMax-H3 checkpoint is one directory and the base release it pins, not {} \
                 directories",
                checkpoint.roots().len()
            )));
        }
        Ok(Box::new(Config::open(checkpoint)?))
    }

    /// A FastH3 adapter edits the base release's `transformer`, the network its student was
    /// distilled from; fused there, the checkpoint reads as that student.
    fn adapter_stage(
        &self,
        checkpoint: &Checkpoint,
        adapter: &nn::Adapter,
    ) -> nn::Result<Option<&'static str>> {
        if !crate::checkpoint::fasth3_adapter(adapter.metadata()) {
            return Err(nn::Error::Unsupported(format!(
                "{} is not a FastH3 adapter (its metadata names the student {:?} over {:?}); \
                 MiniMax-H3 fuses the FastH3 releases of FastVideo over {}",
                adapter.path().display(),
                adapter.metadata().get("finetuned_model"),
                adapter.metadata().get("base_model"),
                crate::checkpoint::FASTH3_BASE_MODEL,
            )));
        }
        // An adapter edits the base release: a FastH3 export is already a student.
        if checkpoint.sidecar("fastvideo_inference.json")?.is_some() {
            return Err(nn::Error::Unsupported(format!(
                "{} is a FastH3 export, and a FastH3 adapter is fused into the base release {} \
                 it was extracted from",
                checkpoint.root().display(),
                crate::checkpoint::FASTH3_BASE_MODEL,
            )));
        }
        Ok(Some(Stage::Transformer.name()))
    }
}

impl VideoCheckpoint for Config {
    fn denoisers(&self) -> &'static [&'static str] {
        // The two networks a release holds, of which a deployment places one.
        const NETWORKS: [&str; 2] = [Stage::Transformer.name(), Stage::TransformerRef.name()];
        &NETWORKS
    }

    fn text_encoder(&self) -> &'static str {
        Stage::TextEncoder.name()
    }

    fn load(
        &self,
        checkpoint: &Checkpoint,
        loader: &mut WeightLoader<'_>,
        placement: &VideoPlacement,
    ) -> nn::Result<Arc<dyn VideoModel>> {
        Ok(Arc::new(MinimaxH3::load(
            self, checkpoint, loader, placement,
        )?))
    }
}

/// A loaded MiniMax-H3 pipeline: the stages this rank places, each constructed once, the
/// configuration they were read with, the deployment's element type and the tasks its network
/// serves.
pub struct MinimaxH3 {
    config: Config,
    dtype: DType,
    tasks: Vec<VideoTask>,
    text_encoder: Option<Arc<Qwen3VlModel>>,
    denoiser: Option<(Stage, Arc<Denoiser>)>,
    splicer: Option<Arc<VideoPostprocessor>>,
    video_decoder: Option<Arc<VideoDecoder>>,
    audio_decoder: Option<Arc<AudioDecoder>>,
}

impl MinimaxH3 {
    /// Load the stages `placement` places of `checkpoint`, whose configuration `config` is.
    ///
    /// Every weight is converted once to the deployment's element type as it is published, and the
    /// retained key names the conversion. A stage this build cannot construct is refused by name
    /// rather than omitted: a deployment that places one fails at load, not on the device.
    fn load(
        config: &Config,
        checkpoint: &Checkpoint,
        loader: &mut WeightLoader<'_>,
        placement: &VideoPlacement,
    ) -> Result<Self> {
        let serving = &placement.serving;
        let dtype = placement.dtype;
        let tasks = served_tasks(
            &config.tasks(),
            placement.stages.as_ref(),
            checkpoint.root(),
        )?;
        let stages = stages(config, &tasks)?;
        let sequence = placement.sequence;
        // The Ulysses degree splits the attention heads between the ranks of the exchange.
        for (dit, denoiser) in &config.denoisers {
            let heads = denoiser.transformer.num_attention_heads;
            if sequence.split.size == 0 || !heads.is_multiple_of(sequence.split.size) {
                return Err(Error::config(format!(
                    "the sequence degree {} does not divide the {heads} attention heads of {}",
                    sequence.split.size,
                    dit.name()
                )));
            }
        }

        // The prompt capacity is the deployment's longest prompt, which the denoiser's packed prefix
        // holds; the condition capacity is the deployment's when it states one, and otherwise the
        // widest its tasks ask for.
        let text_rows = *serving.text_capacities.last().ok_or_else(|| {
            Error::config(
                "this deployment states no prompt capacity, so no layout could hold a prompt"
                    .to_owned(),
            )
        })?;
        let condition_rows = match serving.condition_rows {
            Some(rows) => rows,
            None => tasks
                .iter()
                .map(|task| condition_rows(config, *task))
                .collect::<Result<Vec<u32>>>()?
                .into_iter()
                .max()
                .unwrap_or(0),
        };

        let mut model = Self {
            config: config.clone(),
            dtype,
            tasks: tasks.clone(),
            text_encoder: None,
            denoiser: None,
            splicer: None,
            video_decoder: None,
            audio_decoder: None,
        };
        if stages.contains(Stage::TextEncoder) {
            let processor = config.processor.ok_or_else(|| {
                Error::checkpoint(
                    "the checkpoint states no processor/video_preprocessor_config.json, so the \
                     text encoder's vision tower cannot size an image"
                        .to_owned(),
                )
            })?;
            model.text_encoder = Some(Arc::new(Qwen3VlModel::load(
                &mut loader.reborrow(checkpoint.stage("text_encoder")?)?,
                &config.text_encoder,
                (processor, config.pixels),
                placement.tensor,
                dtype,
                RETAINED_LAYERS,
                false,
            )?));
        }
        for dit in crate::checkpoint::DITS {
            if !stages.contains(dit) {
                continue;
            }
            let denoiser = config.denoisers.get(&dit).ok_or_else(|| {
                Error::checkpoint(format!("the checkpoint holds no {} DiT", dit.name()))
            })?;
            // The two condition encoders are the denoiser's conditioning half.
            let encoders = if stages.contains(Stage::LatentEncoder) {
                Some(crate::denoiser::load_encoders(config, checkpoint, loader)?)
            } else {
                None
            };
            let source = checkpoint.stage(dit.name())?;
            let mut shards = loader.reborrow(source)?;
            // A parallel-decoding student stacks one head per fine interval; the loading boundary
            // fuses them into one head per evaluation block.
            let heads = if denoiser.transformer.output_heads > 1 {
                let schedule = crate::schedule::Schedule::new(&denoiser.schedule)?;
                Some(crate::transformer::fuse_heads(
                    source,
                    &schedule,
                    &denoiser.transformer,
                    &mut shards,
                )?)
            } else {
                None
            };
            let network = Denoiser::load(
                &mut shards,
                denoiser.clone(),
                (text_rows, condition_rows),
                sequence,
                encoders,
                heads,
            )?;
            model.denoiser = Some((dit, Arc::new(network)));
        }
        if stages.contains(Stage::VideoPostprocessor) {
            model.splicer = Some(Arc::new(VideoPostprocessor::new(
                crate::video_vae::ACTIVATION,
                crate::video_vae::clips(&config.video_vae),
            )?));
        }
        if stages.contains(Stage::AudioDecoder) {
            model.audio_decoder = Some(Arc::new(AudioDecoder::load(
                &config.audio_vae,
                &mut loader.reborrow(checkpoint.stage("audio_vae")?)?,
            )?));
        }
        if stages.contains(Stage::VideoDecoder) {
            let canvases: Vec<(u32, u32)> = deployment_canvases(&serving.sizes)?
                .into_iter()
                .map(|(_, canvas)| (canvas.width, canvas.height))
                .collect();
            model.video_decoder = Some(Arc::new(VideoDecoder::load(
                &config.video_vae,
                &canvases,
                &mut loader.reborrow(checkpoint.stage("vae")?)?,
            )?));
        }
        Ok(model)
    }

    /// The placed denoiser network.
    fn network(&self) -> Option<&Arc<Denoiser>> {
        self.denoiser.as_ref().map(|(_, network)| network)
    }
}

impl VideoModel for MinimaxH3 {
    fn text_encoder(&self) -> Option<Arc<dyn Encoder + Send + Sync>> {
        self.text_encoder
            .as_ref()
            .map(|encoder| Arc::clone(encoder) as Arc<dyn Encoder + Send + Sync>)
    }

    fn text_width(&self) -> u32 {
        self.config.text_encoder.text.hidden_size
    }

    fn denoiser(&self) -> Option<Arc<dyn nn::Denoiser + Send + Sync>> {
        self.network()
            .map(|denoiser| Arc::clone(denoiser) as Arc<dyn nn::Denoiser + Send + Sync>)
    }

    fn denoiser_stage(&self) -> &'static str {
        match self.denoiser {
            Some((Stage::TransformerRef, _)) => Stage::TransformerRef.name(),
            _ => Stage::Transformer.name(),
        }
    }

    fn sample_type(&self) -> DType {
        crate::denoiser::SAMPLE_DTYPE
    }

    fn video_decoder(&self) -> Option<Arc<dyn Decoder + Send + Sync>> {
        self.video_decoder
            .as_ref()
            .map(|decoder| Arc::clone(decoder) as Arc<dyn Decoder + Send + Sync>)
    }

    fn splicer(&self) -> Option<Arc<dyn Splicer + Send + Sync>> {
        self.splicer
            .as_ref()
            .map(|splicer| Arc::clone(splicer) as Arc<dyn Splicer + Send + Sync>)
    }

    fn audio_decoder(&self) -> Option<Arc<dyn Decoder + Send + Sync>> {
        self.audio_decoder
            .as_ref()
            .map(|decoder| Arc::clone(decoder) as Arc<dyn Decoder + Send + Sync>)
    }

    fn audio_rate(&self) -> u32 {
        self.audio_decoder
            .as_ref()
            .map_or(AUDIO_SAMPLE_RATE, |decoder| decoder.sampling_rate())
    }

    fn frame_rate(&self) -> u32 {
        FPS
    }

    fn decode_window(&self) -> u32 {
        // One decode call decodes the decoder's window of latent frames, each covering the VAE's
        // temporal compression of output frames.
        crate::video_vae::clips(&self.config.video_vae).window()
            * self.config.video_vae.temporal_compression()
    }

    fn format(&self) -> nn::Result<MediaFormat> {
        Ok(crate::encoding::media_format(&self.config, self.dtype)?)
    }

    fn timeline(&self, seconds: f64) -> nn::Result<u32> {
        Ok(timeline(seconds)?)
    }

    fn video_units(&self, frames: u32) -> nn::Result<Vec<u32>> {
        // The decoder's temporal clips are the one home of the clip schedule.
        Ok(
            crate::video_vae::splices(&crate::video_vae::clips(&self.config.video_vae), frames)?
                .into_iter()
                .map(|splice| splice.keep)
                .collect(),
        )
    }

    fn condition_bounds(&self) -> ConditionBounds {
        // A deployment that places no condition encoder takes no condition at all.
        let rows = self
            .network()
            .filter(|denoiser| denoiser.places_encoders())
            .map_or(0, |denoiser| denoiser.rules().condition_capacity());
        // A condition row is one latent patch, a cell of the canvas grid square, of one latent
        // frame, which covers up to a clip's frames over its latent frames, rounded up; a track's
        // latent frame is two rows.
        let frames = FRAMES.clip_frames.div_ceil(FRAMES.clip_latent_frames);
        let cell = CANVAS_RULE.multiple;
        let merge = self
            .config
            .vision()
            .map_or(1, |vision| vision.spatial_merge_size);
        ConditionBounds {
            rows,
            pixels: rows.saturating_mul(cell * cell * frames),
            samples: (rows / AUDIO_CHANNELS).saturating_mul(AUDIO_HOP),
            patches_per_token: merge * merge,
        }
    }

    fn candidates(&self, serving: &VideoServing) -> nn::Result<Vec<WarmRequest>> {
        let denoiser = self.network().ok_or_else(|| {
            Error::config("this rank places no denoiser, so no denoising step could run".to_owned())
        })?;
        Ok(candidates(
            &self.tasks,
            serving,
            &denoiser.rules().condition_ladder(),
            &condition_packing(denoiser)?,
            self.config.processor.as_ref(),
        )?)
    }

    fn warm_prompts(&self, serving: &VideoServing) -> nn::Result<Vec<WarmPrompt>> {
        Ok(warm_prompts(
            &self.tasks,
            serving,
            self.config.processor.as_ref(),
        )?)
    }

    fn description(&self, serving: &VideoServing) -> nn::Result<Description> {
        Ok(describe(self, serving)?)
    }
}

/// The stages a deployment serving `tasks` places over a checkpoint of `config`.
///
/// A rank places **one** DiT partition, so the tasks must share one: `t2va` and `fl2va` run the
/// base `transformer`, `ref2va` the `transformer_ref`. The decoders and the splicer are placed for
/// every task, because every task returns video and audio; the condition encoders for the tasks
/// that carry conditions.
fn stages(config: &Config, tasks: &[VideoTask]) -> Result<Stages> {
    if tasks.is_empty() {
        return Err(Error::config(
            "a deployment serves at least one task".to_owned(),
        ));
    }
    let stated = config.tasks();
    let mut stages = Stages::of([
        Stage::TextEncoder,
        Stage::VideoDecoder,
        Stage::AudioDecoder,
        Stage::VideoPostprocessor,
    ]);
    for task in tasks {
        if !stated.contains(task) {
            return Err(Error::config(format!(
                "this checkpoint states {stated:?} and not {task:?}"
            )));
        }
        stages = stages.with(crate::checkpoint::dit(*task));
        if task.conditioned() {
            stages = stages.with(Stage::LatentEncoder);
        }
    }
    if stages.contains(Stage::Transformer) && stages.contains(Stage::TransformerRef) {
        return Err(Error::config(
            "a rank cannot hold both the base and the reference denoiser: they are one stage"
                .to_owned(),
        ));
    }
    Ok(stages)
}

/// The condition rows a deployment reserves to serve `task` at the largest canvas the checkpoint
/// generates, in whole capacity units: the keyframes the task anchors, or a reference clip and
/// its soundtrack, counted row for row by the packing ([`crate::packing::condition_rows`]). A task with
/// no conditions still reserves one unit, which the packing's text region is placed against.
fn condition_rows(config: &Config, task: VideoTask) -> Result<u32> {
    let dit = crate::checkpoint::dit(task);
    let denoiser = config
        .denoisers
        .get(&dit)
        .ok_or_else(|| Error::config(format!("the checkpoint holds no {} DiT", dit.name())))?;
    let mut rows = 0;
    for (_, canvas) in named_canvases()? {
        let keyframe = |role| nn::ConditionShape {
            role,
            frames: 1,
            height: canvas.height,
            width: canvas.width,
            audio_samples: 0,
        };
        let conditions = match task {
            VideoTask::TextToVideoAudio => Vec::new(),
            VideoTask::FirstLastToVideoAudio => vec![
                keyframe(nn::ConditionRole::FirstFrame),
                keyframe(nn::ConditionRole::LastFrame),
            ],
            // A reference with a soundtrack is one clip of the frame rule's frames and a track as
            // long as the clip.
            VideoTask::ReferenceToVideoAudio => vec![nn::ConditionShape {
                role: nn::ConditionRole::Reference,
                frames: FRAMES.clip_frames,
                height: canvas.height,
                width: canvas.width,
                audio_samples: FRAMES.clip_frames * AUDIO_SAMPLE_RATE / FPS,
            }],
        };
        let segments = crate::packing::condition_segments(&conditions, &canvas)?;
        let rows_packing =
            crate::packing::condition_packing(crate::config::AttentionKind::Dense, 0)?;
        rows = rows.max(crate::packing::condition_rows(&segments, &rows_packing));
    }
    let unit: u32 = denoiser.attention.capacity_rows();
    Ok(rows.max(1).div_ceil(unit) * unit)
}

/// The tasks a deployment serves: those the checkpoint states, on the network the deployment
/// places. A deployment that places the whole pipeline serves what the checkpoint states when that
/// is one network's.
fn served_tasks(
    stated: &[VideoTask],
    stages: Option<&BTreeSet<String>>,
    checkpoint: &Path,
) -> Result<Vec<VideoTask>> {
    if stated.is_empty() {
        return Err(Error::config(format!(
            "the checkpoint {} states no task, so no denoiser partition can be sized for one",
            checkpoint.display()
        )));
    }
    let Some(stages) = stages else {
        let first = crate::checkpoint::dit(stated[0]);
        if stated
            .iter()
            .all(|task| crate::checkpoint::dit(*task) == first)
        {
            return Ok(stated.to_vec());
        }
        return Err(Error::config(format!(
            "the checkpoint {} holds two networks and this deployment places the whole model: a \
             deployment file that places one of them chooses it",
            checkpoint.display()
        )));
    };
    let served: Vec<VideoTask> = stated
        .iter()
        .copied()
        .filter(|task| stages.contains(crate::checkpoint::dit(*task).name()))
        .collect();
    if served.is_empty() {
        return Err(Error::config(format!(
            "the checkpoint {} states {stated:?} and the deployment places no network of them",
            checkpoint.display()
        )));
    }
    Ok(served)
}

/// The frames of the least timeline of the frame rule that lasts at least `seconds`.
fn timeline(seconds: f64) -> Result<u32> {
    let frames = processing::requested_frames(seconds, FPS);
    if !seconds.is_finite() || frames == 0 || frames > u32::MAX - FRAMES.clip_frames {
        return Err(Error::config(format!(
            "{seconds} seconds is not a positive number of frames at {FPS} frames per second"
        )));
    }
    Ok(FRAMES.align_up(frames))
}

/// The canvases of the sizes a deployment serves, each with the aspect ratio that names it, in the
/// deployment's order.
fn deployment_canvases(sizes: &[(u32, u32)]) -> Result<Vec<((u32, u32), Canvas)>> {
    if sizes.is_empty() {
        return Err(Error::config(
            "this deployment serves no size, so no raster could be decoded".to_owned(),
        ));
    }
    let generated = named_canvases()?;
    sizes
        .iter()
        .map(|&(width, height)| {
            generated
                .iter()
                .find(|(_, canvas)| canvas.width == width && canvas.height == height)
                .copied()
                .ok_or_else(|| {
                    Error::config(format!(
                        "this deployment serves {width}x{height}, which the checkpoint does not \
                         generate: it generates {}",
                        generated
                            .iter()
                            .map(|(_, canvas)| format!("{}x{}", canvas.width, canvas.height))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                })
        })
        .collect()
}

/// The keyframes a task's warm graph is captured with, `(first, last)`: `fl2va` anchors both,
/// every other task none (a `ref2va` reference's raster follows the client's media).
fn task_keyframes(task: VideoTask) -> (u32, u32) {
    match task {
        VideoTask::FirstLastToVideoAudio => (1, 1),
        VideoTask::TextToVideoAudio | VideoTask::ReferenceToVideoAudio => (0, 0),
    }
}

/// The warm requests a deployment captures its denoiser with: one per (task, size, timeline,
/// prompt capacity, condition rung) the deployment serves. A conditioned `fl2va` candidate also
/// states the keyframe sets a request of its layout may open with ([`keyframe_openings`]).
fn candidates(
    tasks: &[VideoTask],
    serving: &VideoServing,
    ladder: &[u32],
    packing: &ConditionPacking,
    processor: Option<&ProcessorConfig>,
) -> Result<Vec<WarmRequest>> {
    let longest = timeline(serving.max_seconds)?;
    let shortest = timeline(MIN_SECONDS)?;
    let timelines: Vec<u32> = (shortest..=longest)
        .step_by(FRAMES.clip_frames as usize)
        .collect();
    let mut candidates = Vec::new();
    for (_, canvas) in deployment_canvases(&serving.sizes)? {
        for task in tasks.iter().copied() {
            let (first_frames, last_frames) = task_keyframes(task);
            let rungs: Vec<Option<u32>> = if task.conditioned() {
                ladder.iter().copied().map(Some).collect()
            } else {
                vec![None]
            };
            let openings = if task == VideoTask::FirstLastToVideoAudio {
                keyframe_openings(&canvas, ladder, packing, processor)?
            } else {
                Vec::new()
            };
            for frames in timelines.iter().copied() {
                for capacity in &serving.text_capacities {
                    for rung in rungs.iter().copied() {
                        candidates.push(WarmRequest {
                            width: canvas.width,
                            height: canvas.height,
                            frames,
                            fps: FPS,
                            first_frames,
                            last_frames,
                            // A layout's capacities are the deployment's: a `ref2va` reference's
                            // raster follows the client's media, so its warm request states none
                            // and rides its rung's condition rows.
                            references: 0,
                            text_tokens: *capacity,
                            conditions: Vec::new(),
                            vision_spans: Vec::new(),
                            condition_rows: rung,
                            openings: openings
                                .iter()
                                .filter(|(served, _)| Some(*served) == rung)
                                .map(|(_, set)| set.clone())
                                .collect(),
                        });
                    }
                }
            }
        }
    }
    Ok(candidates)
}

/// The keyframe sets an `fl2va` request of `canvas` may present, each with the rung of `ladder`
/// that serves it: every non-empty subset of the anchors, each keyframe fitted to the canvas by
/// request processing's own rule, on the smallest rung holding the rows it packs into.
fn keyframe_openings(
    canvas: &Canvas,
    ladder: &[u32],
    packing: &ConditionPacking,
    processor: Option<&ProcessorConfig>,
) -> Result<Vec<(u32, Vec<nn::ConditionShape>)>> {
    let processor = processor.ok_or_else(|| {
        Error::checkpoint(
            "the deployment serves fl2va and the checkpoint states no \
             processor/video_preprocessor_config.json, so a keyframe cannot be prepared"
                .to_owned(),
        )
    })?;
    // An image of the canvas's own size: the preparation fits every image onto the canvas.
    let image = processing::ImageMetadata {
        width: canvas.width,
        height: canvas.height,
    };
    let anchors = [
        proto::video::ConditionRole::FirstFrame,
        proto::video::ConditionRole::LastFrame,
    ];
    let mut openings = Vec::new();
    for mask in 1_u32..(1 << anchors.len()) {
        let roles: Vec<_> = anchors
            .iter()
            .enumerate()
            .filter(|(bit, _)| mask & (1 << bit) != 0)
            .map(|(_, role)| *role)
            .collect();
        let prepared = roles
            .iter()
            .enumerate()
            .map(|(order, role)| {
                PreparedCondition::keyframe(order, *role, &image, *canvas, order, true, processor)
            })
            .collect::<std::result::Result<Vec<_>, String>>()
            .map_err(|error| {
                Error::config(format!(
                    "a keyframe of the {}x{} canvas: {error}",
                    canvas.width, canvas.height
                ))
            })?;
        let rows = processing::minimax_h3::condition_rows(&prepared, *canvas, packing);
        let Some(rung) = ladder.iter().copied().find(|rung| *rung >= rows) else {
            continue;
        };
        let shapes = roles
            .iter()
            .map(|role| nn::ConditionShape {
                role: match role {
                    proto::video::ConditionRole::FirstFrame => nn::ConditionRole::FirstFrame,
                    proto::video::ConditionRole::LastFrame => nn::ConditionRole::LastFrame,
                    proto::video::ConditionRole::Reference => nn::ConditionRole::Reference,
                },
                frames: 1,
                height: canvas.height,
                width: canvas.width,
                audio_samples: 0,
            })
            .collect();
        openings.push((rung, shapes));
    }
    Ok(openings)
}

/// How the placed denoiser packs a request's conditions, as request processing sizes them: its
/// tiles under sparse attention, row for row under dense attention, and the rows it holds when
/// this rank places the condition encoders.
fn condition_packing(denoiser: &Denoiser) -> Result<ConditionPacking> {
    let capacity = if denoiser.places_encoders() {
        denoiser.rules().condition_capacity()
    } else {
        0
    };
    crate::packing::condition_packing(denoiser.config().attention, capacity)
}

/// The prompts with vision blocks a deployment admits whose grids it fixes: per served canvas, an
/// `fl2va` prompt of each keyframe count, each keyframe at the processor's grid of the canvas. A
/// `t2va` prompt presents none, and a `ref2va` reference's grid follows the client's media.
fn warm_prompts(
    tasks: &[VideoTask],
    serving: &VideoServing,
    processor: Option<&ProcessorConfig>,
) -> Result<Vec<WarmPrompt>> {
    if !tasks.contains(&VideoTask::FirstLastToVideoAudio) {
        return Ok(Vec::new());
    }
    let processor = processor.ok_or_else(|| {
        Error::checkpoint(
            "the deployment serves fl2va and the checkpoint states no \
             processor/video_preprocessor_config.json, so the vision grid a keyframe presents \
             cannot be sized"
                .to_owned(),
        )
    })?;
    let (first_frames, last_frames) = task_keyframes(VideoTask::FirstLastToVideoAudio);
    let mut prompts: Vec<WarmPrompt> = Vec::new();
    for (_, canvas) in deployment_canvases(&serving.sizes)? {
        let grid = processor
            .image_grid(canvas.height, canvas.width)
            .map_err(|error| {
                Error::config(format!(
                    "a keyframe of the {}x{} canvas: {error}",
                    canvas.width, canvas.height
                ))
            })?;
        let image = ImageShape {
            grid: [grid.t, grid.h, grid.w],
        };
        // Each temporal step of a block presents one run of merged tokens.
        let tokens = grid.t * processor.block_tokens(grid);
        for keyframes in 1..=first_frames + last_frames {
            let prompt = WarmPrompt {
                images: vec![image; keyframes as usize],
                tokens: tokens * keyframes,
            };
            if !prompts.contains(&prompt) {
                prompts.push(prompt);
            }
        }
    }
    Ok(prompts)
}

/// What the placed denoiser serves under `serving`, as request processing admits requests against
/// it: the tasks of the placed network, its fixed schedule and shifts, the deployment's canvases,
/// the reference limits and the condition packing.
fn describe(pipeline: &MinimaxH3, serving: &VideoServing) -> Result<Description> {
    let denoiser = pipeline.network().ok_or_else(|| {
        Error::config("this media rank places no denoiser, so it serves no task".to_owned())
    })?;
    let tasks: Vec<VideoTask> = VideoTask::ALL
        .into_iter()
        .filter(|task| pipeline.tasks.contains(task))
        .collect();
    let steps = denoiser.schedule().num_steps()?;
    let shifts = &denoiser.config().schedule.shifts;
    let report = DenoiserInfo {
        tasks: tasks.iter().map(|task| task.as_str().to_owned()).collect(),
        schedule_points: steps + 1,
        video_shift: f64::from(shifts.video),
        audio_shift: f64::from(shifts.audio),
        canvases: deployment_canvases(&serving.sizes)?
            .into_iter()
            .map(
                |((width, height), canvas)| processing::minimax_h3::ServedCanvas {
                    short_edge: CANVAS_RULE.short_edge,
                    aspect_ratio: format!("{width}:{height}"),
                    canvas,
                },
            )
            .collect(),
        references: if tasks.contains(&VideoTask::ReferenceToVideoAudio) {
            REFERENCE_LIMITS
        } else {
            processing::minimax_h3::ReferenceLimits::default()
        },
        condition_packing: condition_packing(denoiser)?,
    };
    report
        .validate()
        .map_err(|reason| Error::config(format!("the denoiser's description: {reason}")))?;
    Ok(Description {
        takes_conditions: tasks.iter().any(|task| task.conditioned()),
        denoise_steps: steps,
        details: report
            .encode()
            .map_err(|error| Error::config(format!("encoding the description: {error}")))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A checkpoint with no page serves what it states when that is one network's, and a checkpoint
    /// that states no task serves nothing.
    #[test]
    fn the_served_tasks_are_those_of_the_checkpoints_one_network() {
        let checkpoint = Path::new("/models/fasth3");
        let tasks = [
            VideoTask::TextToVideoAudio,
            VideoTask::FirstLastToVideoAudio,
        ];
        assert_eq!(
            served_tasks(&tasks, None, checkpoint).expect("one network's tasks"),
            tasks.to_vec()
        );
        let none = served_tasks(&[], None, checkpoint).expect_err("no task is no partition");
        assert!(
            format!("{none:#}").contains("states no task"),
            "the refusal says the checkpoint states none: {none:#}"
        );
        let both = [
            VideoTask::TextToVideoAudio,
            VideoTask::ReferenceToVideoAudio,
        ];
        assert!(
            served_tasks(&both, None, checkpoint).is_err(),
            "two networks need a page to choose between them"
        );
    }

    /// A deployment that places one network serves that network's tasks, and one that places none
    /// of the checkpoint's networks is refused.
    #[test]
    fn a_placed_network_chooses_the_tasks() {
        let checkpoint = Path::new("/models/minimax-h3");
        let both = [
            VideoTask::TextToVideoAudio,
            VideoTask::ReferenceToVideoAudio,
        ];
        let placed = |names: &[&str]| -> BTreeSet<String> {
            names.iter().map(|name| (*name).to_owned()).collect()
        };
        assert_eq!(
            served_tasks(
                &both,
                Some(&placed(&["transformer_ref", "text_encoder"])),
                checkpoint
            )
            .expect("the reference network's tasks"),
            vec![VideoTask::ReferenceToVideoAudio]
        );
        assert!(
            served_tasks(&both, Some(&placed(&["text_encoder"])), checkpoint).is_err(),
            "a deployment that places no network serves no task"
        );
    }

    /// A timeline is `17 * n + 5` frames, and a duration is served by the least timeline covering
    /// it; a duration of no frame is refused.
    #[test]
    fn a_duration_aligns_to_the_least_timeline_covering_it() {
        assert_eq!(timeline(4.0).unwrap(), 107);
        assert_eq!(timeline(5.0).unwrap(), 124);
        assert_eq!(timeline(15.0).unwrap(), 362);
        assert_eq!(timeline(0.5).unwrap(), 22, "one clip and its tail");
        assert!(timeline(0.0).is_err());
        assert!(timeline(f64::NAN).is_err());
    }

    /// A size the checkpoint does not generate is refused by name, with the sizes it does.
    #[test]
    fn a_size_the_checkpoint_does_not_generate_is_refused() {
        let canvases = deployment_canvases(&[(1344, 768), (768, 1344)]).expect("named canvases");
        assert_eq!(canvases.len(), 2);
        let refused = deployment_canvases(&[(1000, 1000)]).expect_err("not a canvas");
        let message = refused.to_string();
        assert!(
            message.contains("1000x1000") && message.contains("1344x768"),
            "{message}"
        );
    }

    /// An `fl2va` deployment admits, per served canvas, a prompt of one keyframe and one of both
    /// anchors, each keyframe at the processor's grid of the canvas; a `t2va` one admits none.
    #[test]
    fn a_keyframe_presents_the_grid_of_its_canvas() {
        // The released processor: 16-pixel patches merged 2x2, an image budget of 64Ki to 16Mi
        // pixels.
        let processor = ProcessorConfig::parse(
            r#"{"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "size": {"shortest_edge": 65536, "longest_edge": 16777216}}"#,
            r#"{"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "size": {"shortest_edge": 4096, "longest_edge": 25165824}}"#,
        )
        .expect("the released processor");
        let serving = VideoServing {
            sizes: vec![(1344, 768)],
            max_seconds: 4.0,
            text_capacities: vec![2048],
            condition_rows: None,
        };

        // 768x1344 pixels are 48x84 patches of one temporal step, which merge to 1008 tokens.
        let keyframe = ImageShape { grid: [1, 48, 84] };
        assert_eq!(
            warm_prompts(
                &[VideoTask::FirstLastToVideoAudio],
                &serving,
                Some(&processor)
            )
            .expect("an fl2va deployment's prompts"),
            vec![
                WarmPrompt {
                    images: vec![keyframe],
                    tokens: 1008,
                },
                WarmPrompt {
                    images: vec![keyframe, keyframe],
                    tokens: 2016,
                },
            ]
        );
        assert_eq!(
            warm_prompts(&[VideoTask::TextToVideoAudio], &serving, None).expect("no vision"),
            Vec::new()
        );
        assert!(
            warm_prompts(&[VideoTask::FirstLastToVideoAudio], &serving, None).is_err(),
            "an fl2va deployment cannot size a keyframe without the processor"
        );
    }

    /// An `fl2va` layout opens with the keyframe sets its rung serves, each keyframe at the
    /// canvas's raster: one keyframe (either anchor) on the rung that holds one canvas frame, both
    /// on the rung that holds two, and a `t2va` layout with none.
    #[test]
    fn a_rung_opens_with_the_keyframes_it_holds() {
        let processor = ProcessorConfig::parse(
            r#"{"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "size": {"shortest_edge": 65536, "longest_edge": 16777216}}"#,
            r#"{"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "size": {"shortest_edge": 4096, "longest_edge": 25165824}}"#,
        )
        .expect("the released processor");
        let serving = VideoServing {
            sizes: vec![(1344, 768)],
            max_seconds: 4.0,
            text_capacities: vec![2048],
            condition_rows: None,
        };
        // Row for row: a 1344x768 keyframe is 42x24 latent patches, 1008 rows.
        let packing = ConditionPacking {
            capacity: 2048,
            tile_rows: 1,
            video_tile: [1, 1, 1],
        };
        let candidates = candidates(
            &[
                VideoTask::TextToVideoAudio,
                VideoTask::FirstLastToVideoAudio,
            ],
            &serving,
            &[1024, 2048],
            &packing,
            Some(&processor),
        )
        .expect("the deployment's candidates");

        let keyframe = |role| nn::ConditionShape {
            role,
            frames: 1,
            height: 768,
            width: 1344,
            audio_samples: 0,
        };
        let opened = |rung: Option<u32>| -> Vec<Vec<nn::ConditionShape>> {
            candidates
                .iter()
                .find(|candidate| candidate.condition_rows == rung)
                .expect("a candidate of the rung")
                .openings
                .clone()
        };
        assert_eq!(opened(None), Vec::<Vec<nn::ConditionShape>>::new());
        assert_eq!(
            opened(Some(1024)),
            vec![
                vec![keyframe(nn::ConditionRole::FirstFrame)],
                vec![keyframe(nn::ConditionRole::LastFrame)],
            ]
        );
        assert_eq!(
            opened(Some(2048)),
            vec![vec![
                keyframe(nn::ConditionRole::FirstFrame),
                keyframe(nn::ConditionRole::LastFrame),
            ]]
        );
    }
}
