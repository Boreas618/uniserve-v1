//! MiniMax-H3 and FastH3 as execution loads them and request processing serves them ([`nn::family`]).
//!
//! [`FAMILY`] claims the `MiniMaxH3ModularPipeline` class. A checkpoint opens to its configuration
//! (an export and the base release it pins read as one), a load places the network the deployment
//! file's components name with the deployment's sequence and tensor splits, and the loaded
//! pipeline states the model's own rules to execution: the `17 * n + 5` timeline, its canvases, the
//! requests its denoiser graphs are captured for, and the description its request processing
//! plans requests against.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use nn::capability::{Decoder, Encoder, ImageShape, MediaFormat, Splicer, WarmPrompt, WarmRequest};
use nn::family::{
    Component, ConditionBounds, Description, Family, Model, VideoCheckpoint, VideoFamily,
    VideoModel, VideoPlacement, VideoServing,
};
use nn::{Checkpoint, DType, WeightLoader};

use crate::config::{CANVAS_RULE, FRAMES, MIN_SECONDS, named_canvases};
use crate::denoiser::Capacity;
use crate::load::MinimaxH3;
use crate::processing::ConditionPacking;
use crate::{AUDIO_CHANNELS, AUDIO_HOP, AUDIO_SAMPLE_RATE, Config, FPS, Partition, VideoTask};
use proto::video::Canvas;
use qwen3::processor::ProcessorConfig;

/// The MiniMax-H3 family: a diffusers modular pipeline, which request processing serves through its
/// video processor ([`crate::processing`]).
pub const FAMILY: Family = Family {
    request: ::processing::Family {
        id: "minimax_h3",
        model_types: &[],
        classes: &["MiniMaxH3ModularPipeline"],
        serving: ::processing::Serving::Video(&crate::processing::H3Family),
    },
    model: Model::Video(&MinimaxH3Family),
};

/// The family's one value, which [`FAMILY`] refers to.
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
        let config = Config::open(checkpoint)?;
        Ok(Box::new(H3Checkpoint { config }))
    }

    /// A FastH3 adapter edits the base release's `transformer` partition, the one its student was
    /// distilled from; it is fused there and the checkpoint then reads as that student
    /// ([`crate::checkpoint::Contract::of_adapter`]).
    fn adapter_component(
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
        if crate::checkpoint::Contract::read(checkpoint)?.is_some() {
            return Err(nn::Error::Unsupported(format!(
                "{} is a FastH3 export, and a FastH3 adapter is fused into the base release {} \
                 it was extracted from",
                checkpoint.root().display(),
                crate::checkpoint::FASTH3_BASE_MODEL,
            )));
        }
        Ok(Some(crate::checkpoint::Dit::Base.directory()))
    }
}

/// A MiniMax-H3 checkpoint: the configuration read from it and its base, once.
struct H3Checkpoint {
    config: Config,
}

impl VideoCheckpoint for H3Checkpoint {
    fn denoisers(&self) -> &'static [&'static str] {
        // The two networks a release holds, of which a deployment places one.
        const NETWORKS: [&str; 2] = [
            Component::Transformer.name(),
            Component::TransformerRef.name(),
        ];
        &NETWORKS
    }

    fn text_encoder(&self) -> &'static str {
        Component::TextEncoder.name()
    }

    fn load(
        &self,
        checkpoint: &Checkpoint,
        loader: &mut WeightLoader<'_>,
        placement: &VideoPlacement,
    ) -> nn::Result<Arc<dyn VideoModel>> {
        let config = &self.config;
        let serving = &placement.serving;

        // **The partition serves the tasks of the network the deployment places.** One partition
        // carries one network's block set and condition capacity, so a deployment places one of
        // the release's two networks.
        let tasks = served_tasks(
            &config.tasks(),
            placement.components.as_ref(),
            checkpoint.root(),
        )?;

        // **The prompt capacity is the deployment's, and the denoiser's packed prefix holds it.**
        // A layout reserves one row per prompt token, so the text capacity the model is built with
        // and the prompt capacities the deployment captures are one number: the largest of them,
        // the longest prompt the deployment admits.
        let text_capacity = *serving.text_capacities.last().ok_or_else(|| {
            nn::Error::Unsupported(
                "this deployment states no prompt capacity, so no layout could hold a prompt"
                    .to_owned(),
            )
        })?;

        // **The rank's place, its blocks and its capacity travel apart.** The partition is the
        // split alone; the network the tasks run on, the condition encoders a conditioned task
        // reads and the postprocessor - the clip splicer (`nn::capability::Splicer`),
        // which holds no weights - are the placed blocks.
        let sequence = placement.sequence;
        let partition = Partition::new(
            config,
            sequence.split.size as usize,
            placement.tensor.size as usize,
            placement.tensor.rank as usize,
            sequence.split.rank as usize,
        )?
        .projecting(sequence.projection);
        let blocks = crate::partition::blocks(config, &tasks)?;

        // The condition capacity is the deployment's when it states one: the rows a request's
        // references take follow their rasters, which a task's own default cannot know. Without
        // one it is the widest the tasks ask for.
        let condition_rows = match serving.condition_rows {
            Some(rows) => rows,
            None => tasks
                .iter()
                .map(|task| crate::partition::condition_rows(config, *task))
                .collect::<crate::Result<Vec<u32>>>()?
                .into_iter()
                .max()
                .unwrap_or(0),
        };
        let capacity = Capacity {
            text_rows: text_capacity,
            condition_rows,
        };

        let canvases = deployment_canvases(&serving.sizes)?
            .into_iter()
            .map(|(_, canvas)| (canvas.width, canvas.height))
            .collect::<Vec<_>>();
        let model = crate::load(
            config,
            checkpoint,
            partition,
            blocks,
            capacity,
            Some(placement.dtype),
            &canvases,
            loader,
        )?;

        // **One number, checked where both halves meet.** A model built for fewer text rows than
        // the deployment's longest prompt would refuse every longer prompt at the packed prefix,
        // minutes into a load and in a component that never saw the deployment's number.
        if let Some(denoiser) = model.transformer().or(model.transformer_ref()) {
            let held = denoiser.rules().capacity().text_rows;
            if serving
                .text_capacities
                .iter()
                .any(|capacity| *capacity > held)
            {
                return Err(nn::Error::Unsupported(format!(
                    "this deployment prepares prompts of up to {text_capacity} tokens and the \
                     denoiser's layout holds {held} text rows: the prompt capacity is one number, \
                     stated by the deployment and held by the packing"
                )));
            }
        }

        Ok(Arc::new(H3Pipeline {
            model,
            config: config.clone(),
            dtype: placement.dtype,
            tasks,
        }))
    }
}

/// A loaded MiniMax-H3 pipeline: the components this rank places, the configuration they were
/// read with, the deployment's element type and the tasks its network serves.
struct H3Pipeline {
    model: MinimaxH3,
    config: Config,
    dtype: DType,
    tasks: Vec<VideoTask>,
}

impl H3Pipeline {
    /// The placed denoiser network.
    fn network(&self) -> Option<&Arc<crate::denoiser::Denoiser>> {
        self.model.transformer().or(self.model.transformer_ref())
    }
}

impl VideoModel for H3Pipeline {
    fn text_encoder(&self) -> Option<Arc<dyn Encoder + Send + Sync>> {
        self.model
            .text_encoder()
            .map(|encoder| Arc::clone(encoder) as Arc<dyn Encoder + Send + Sync>)
    }

    fn text_width(&self) -> u32 {
        self.config.text_encoder.text.hidden_size
    }

    fn denoiser(&self) -> Option<Arc<dyn nn::Denoiser + Send + Sync>> {
        self.network()
            .map(|denoiser| Arc::clone(denoiser) as Arc<dyn nn::Denoiser + Send + Sync>)
    }

    fn denoiser_component(&self) -> &'static str {
        if self.model.transformer_ref().is_some() {
            Component::TransformerRef.name()
        } else {
            Component::Transformer.name()
        }
    }

    fn sample_type(&self) -> DType {
        crate::denoiser::SAMPLE_DTYPE
    }

    fn video_decoder(&self) -> Option<Arc<dyn Decoder + Send + Sync>> {
        self.model
            .video_decoder()
            .map(|decoder| Arc::clone(decoder) as Arc<dyn Decoder + Send + Sync>)
    }

    fn splicer(&self) -> Option<Arc<dyn Splicer + Send + Sync>> {
        self.model
            .video_postprocessor()
            .map(|splicer| Arc::clone(splicer) as Arc<dyn Splicer + Send + Sync>)
    }

    fn audio_decoder(&self) -> Option<Arc<dyn Decoder + Send + Sync>> {
        self.model
            .audio_decoder()
            .map(|decoder| Arc::clone(decoder) as Arc<dyn Decoder + Send + Sync>)
    }

    fn audio_rate(&self) -> u32 {
        self.model
            .audio_decoder()
            .map_or(AUDIO_SAMPLE_RATE, |decoder| decoder.sampling_rate())
    }

    fn frame_rate(&self) -> u32 {
        FPS
    }

    fn decode_window(&self) -> u32 {
        // One decode call decodes the decoder's window of latent frames, each covering the VAE's
        // temporal compression of output frames; a longer timeline is spliced from its units.
        crate::video_vae::clips(&self.config.video_vae).window()
            * self.config.video_vae.temporal_compression()
    }

    fn format(&self) -> nn::Result<MediaFormat> {
        Ok(crate::encoding::media_format(&self.config, self.dtype))
    }

    fn timeline(&self, seconds: f64) -> nn::Result<u32> {
        timeline(seconds)
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
            .map_or(0, |denoiser| denoiser.rules().capacity().condition_rows);

        // A condition row is one latent patch, a cell of the canvas grid square, of one latent
        // frame, which covers up to a clip's frames over its latent frames, rounded up; a track's
        // latent frame is two rows.
        let frames = FRAMES.clip_frames.div_ceil(FRAMES.clip_latent_frames);
        let cell = CANVAS_RULE.multiple;
        let merge = self.config.text_encoder.vision.spatial_merge_size;
        ConditionBounds {
            rows,
            pixels: rows.saturating_mul(cell * cell * frames),
            samples: (rows / AUDIO_CHANNELS).saturating_mul(AUDIO_HOP),
            patches_per_token: merge * merge,
        }
    }

    fn candidates(&self, serving: &VideoServing) -> nn::Result<Vec<WarmRequest>> {
        let denoiser = self.network().ok_or_else(|| {
            nn::Error::Unsupported(
                "the checkpoint holds no transformer, so no denoising step could run".to_owned(),
            )
        })?;
        let ladder = denoiser.rules().condition_ladder();
        candidates(
            &self.tasks,
            serving,
            &ladder,
            &condition_packing(denoiser)?,
            self.config.processor.as_ref(),
        )
    }

    fn warm_prompts(&self, serving: &VideoServing) -> nn::Result<Vec<WarmPrompt>> {
        warm_prompts(&self.tasks, serving, self.config.processor.as_ref())
    }

    fn description(&self, serving: &VideoServing) -> nn::Result<Description> {
        describe(self, serving)
    }
}

/// The tasks a deployment serves: those the checkpoint states, on the network the deployment
/// places.
///
/// A release holds two networks, `transformer` and `transformer_ref`, and a deployment runs one:
/// its components say which. A deployment that places the whole pipeline states no such choice,
/// so it serves what the checkpoint states when that is one network's.
fn served_tasks(
    stated: &[VideoTask],
    components: Option<&BTreeSet<String>>,
    checkpoint: &Path,
) -> nn::Result<Vec<VideoTask>> {
    let refuse = |problem: String| nn::Error::Unsupported(problem);
    if stated.is_empty() {
        return Err(refuse(format!(
            "the checkpoint {} states no task, so no denoiser partition can be sized for one",
            checkpoint.display()
        )));
    }

    let reference = |task: &VideoTask| *task == VideoTask::ReferenceToVideoAudio;
    let Some(components) = components else {
        if stated.iter().all(reference) || !stated.iter().any(reference) {
            return Ok(stated.to_vec());
        }
        return Err(refuse(format!(
            "the checkpoint {} holds two networks and this deployment places the whole model: a \
             deployment file that places one of them chooses it",
            checkpoint.display()
        )));
    };

    let served: Vec<VideoTask> = stated
        .iter()
        .copied()
        .filter(|task| {
            components.contains(if reference(task) {
                Component::TransformerRef.name()
            } else {
                Component::Transformer.name()
            })
        })
        .collect();
    if served.is_empty() {
        return Err(refuse(format!(
            "the checkpoint {} states {stated:?} and the deployment places no network of them",
            checkpoint.display()
        )));
    }
    Ok(served)
}

/// The frames of the least timeline of the frame rule that lasts at least `seconds`: the duration's
/// frames as the request rules count them ([`processing::requested_frames`]), aligned up to the
/// frame rule ([`FRAMES`]), so 4 s is 96 frames and aligns to 107, 15 s is 360 and aligns to 362.
fn timeline(seconds: f64) -> nn::Result<u32> {
    let frames = processing::requested_frames(seconds, FPS);
    if !seconds.is_finite() || frames == 0 || frames > u32::MAX - FRAMES.clip_frames {
        return Err(nn::Error::Unsupported(format!(
            "{seconds} seconds is not a positive number of frames at {FPS} frames per second"
        )));
    }
    Ok(FRAMES.align_up(frames))
}

/// The canvases of the sizes a deployment serves, each with the aspect ratio that names it, in the
/// deployment's order.
///
/// **One decoder serves the whole set**: the constants a decode reads are a tile's or a width's,
/// the same for every canvas, so a size of the set is a lookup and costs no device storage.
fn deployment_canvases(sizes: &[(u32, u32)]) -> nn::Result<Vec<((u32, u32), Canvas)>> {
    if sizes.is_empty() {
        return Err(nn::Error::Unsupported(
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
                    nn::Error::Unsupported(format!(
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

/// The condition counts a task's warm graph is captured with.
///
/// A graph is captured for a layout, and a layout's capacities are the deployment's: what a request
/// fills of them reaches the graph through its state tables. So a warm request needs no condition
/// of its own, and `ref2va`'s references state none; `fl2va` anchors a first and a last frame.
fn task_conditions(task: VideoTask) -> (u32, u32, u32) {
    match task {
        VideoTask::TextToVideoAudio | VideoTask::ReferenceToVideoAudio => (0, 0, 0),
        VideoTask::FirstLastToVideoAudio => (1, 1, 0),
    }
}

/// The warm requests a deployment binds its denoiser with.
///
/// **One per (task, size, timeline, prompt capacity, condition rung) the deployment serves.** A
/// denoiser bucket is a layout, and a layout counts the prompt's rows, the request's frames and its
/// conditions; the runner keeps one candidate per distinct layout. The tasks are the checkpoint's,
/// on the placed network; the timelines are every legal one from the shortest output to the
/// deployment's longest.
///
/// A conditioned candidate also states the condition sets a request of its layout may open with
/// ([`WarmRequest::openings`]): an `fl2va` layout those of [`keyframe_openings`] its rung serves.
/// A `ref2va` reference's raster follows the client's media, so a `ref2va` layout states none.
fn candidates(
    tasks: &[VideoTask],
    serving: &VideoServing,
    ladder: &[u32],
    packing: &ConditionPacking,
    processor: Option<&ProcessorConfig>,
) -> nn::Result<Vec<WarmRequest>> {
    let longest = timeline(serving.max_seconds)?;
    let shortest = timeline(MIN_SECONDS)?;
    let timelines: Vec<u32> = (shortest..=longest)
        .step_by(FRAMES.clip_frames as usize)
        .collect();

    let mut candidates = Vec::new();
    for (_, canvas) in deployment_canvases(&serving.sizes)? {
        for task in tasks.iter().copied() {
            let (first_frames, last_frames, references) = task_conditions(task);

            // A layout's condition rows are real rows of every step, so a conditioned task
            // captures one graph per rung of the ladder and a request rides the smallest that
            // holds it. A task with no conditions holds none.
            let rungs: Vec<Option<u32>> = if task == VideoTask::TextToVideoAudio {
                vec![None]
            } else {
                ladder.iter().copied().map(Some).collect()
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
                            references,
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
/// that serves it.
///
/// A request anchors its first frame, its last, or both (`task_conditions`). Each keyframe is
/// planned by the request processing's own keyframe plan, which fits it to the canvas whatever
/// image the client sent, so its shape is the canvas's; a set rides the smallest rung that holds
/// the rows the processing packs it into under the deployment's `packing`, and a set no rung holds
/// is one the processing refuses.
fn keyframe_openings(
    canvas: &Canvas,
    ladder: &[u32],
    packing: &ConditionPacking,
    processor: Option<&ProcessorConfig>,
) -> nn::Result<Vec<(u32, Vec<nn::ConditionShape>)>> {
    use crate::processing::plan::plan_keyframe;

    let processor = processor.ok_or_else(|| {
        nn::Error::Unsupported(
            "the deployment serves fl2va and the checkpoint states no \
             processor/video_preprocessor_config.json, so a keyframe cannot be planned"
                .to_owned(),
        )
    })?;
    let size = *canvas;
    // An image of the canvas's own size: the keyframe plan fits every image onto the canvas, so
    // the image's size reaches only the media reader's crop and not the condition's shape.
    let image = processing::ImageMetadata {
        width: canvas.width,
        height: canvas.height,
    };
    let (first_frames, last_frames, _) = task_conditions(VideoTask::FirstLastToVideoAudio);
    let anchors: Vec<i64> = std::iter::repeat_n(0, first_frames as usize)
        .chain(std::iter::repeat_n(-1, last_frames as usize))
        .collect();

    let mut openings = Vec::new();
    // Every non-empty subset of the anchors, in anchor order.
    for mask in 1_u32..(1 << anchors.len()) {
        let mut plans = Vec::new();
        for (index, frame_index) in anchors
            .iter()
            .enumerate()
            .filter(|(bit, _)| mask & (1 << bit) != 0)
            .map(|(_, frame_index)| *frame_index)
            .enumerate()
        {
            let spec = processing::Reference {
                kind: processing::ReferenceKind::Image,
                frame_index: Some(frame_index),
                start_time_seconds: None,
            };
            let plan = plan_keyframe(
                index,
                &spec,
                &image,
                size,
                index,
                VideoTask::FirstLastToVideoAudio,
                processor,
            )
            .map_err(|error| {
                nn::Error::Unsupported(format!(
                    "a keyframe of the {}x{} canvas: {error}",
                    canvas.width, canvas.height
                ))
            })?;
            plans.push(plan);
        }
        let rows =
            crate::processing::plan::condition_rows(&plans, size, packing).map_err(|error| {
                nn::Error::Unsupported(format!(
                    "the keyframes of the {}x{} canvas: {error}",
                    canvas.width, canvas.height
                ))
            })?;
        let Some(rung) = ladder.iter().copied().find(|rung| *rung >= rows) else {
            continue;
        };
        openings.push((rung, plans.iter().map(|plan| plan.shape(size)).collect()));
    }
    Ok(openings)
}

/// How the placed denoiser packs a request's conditions, as the request processing sizes them:
/// its tiles under sparse attention, row for row under dense attention, and the rows it holds when
/// this rank places the condition encoders.
fn condition_packing(denoiser: &crate::denoiser::Denoiser) -> nn::Result<ConditionPacking> {
    let tiling = crate::packing::Tiling::of(denoiser.config().attention)?;
    Ok(ConditionPacking {
        capacity: if denoiser.places_encoders() {
            denoiser.rules().capacity().condition_rows
        } else {
            0
        },
        tile_rows: tiling.rows,
        video_tile: [tiling.shape.0, tiling.shape.1, tiling.shape.2],
    })
}

/// The prompts with vision blocks a deployment admits whose grids the deployment fixes.
///
/// An `fl2va` keyframe is fitted to the generated canvas and enters the conditioner at the canvas's
/// own size, which the processor resizes to its grid ([`ProcessorConfig::image_grid`], as the
/// request processing's keyframe plan calls it), so a request of each served canvas presents one
/// such grid per keyframe: one keyframe, or every anchor the task takes. A `t2va` prompt presents
/// no vision block, and a `ref2va` reference presents a grid that follows the client's media rather
/// than anything the deployment serves, so neither states a prompt here.
fn warm_prompts(
    tasks: &[VideoTask],
    serving: &VideoServing,
    processor: Option<&ProcessorConfig>,
) -> nn::Result<Vec<WarmPrompt>> {
    if !tasks.contains(&VideoTask::FirstLastToVideoAudio) {
        return Ok(Vec::new());
    }
    let processor = processor.ok_or_else(|| {
        nn::Error::Unsupported(
            "the deployment serves fl2va and the checkpoint states no \
             processor/video_preprocessor_config.json, so the vision grid a keyframe presents \
             cannot be sized"
                .to_owned(),
        )
    })?;

    let (first_frames, last_frames, _) = task_conditions(VideoTask::FirstLastToVideoAudio);
    let mut prompts: Vec<WarmPrompt> = Vec::new();
    for (_, canvas) in deployment_canvases(&serving.sizes)? {
        let grid = processor
            .image_grid(canvas.height, canvas.width)
            .map_err(|error| {
                nn::Error::Unsupported(format!(
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

/// What the placed denoiser serves under `serving`, as the request processing plans requests
/// against it.
///
/// The tasks are the checkpoint's on the placed network, which a distilled student narrows from
/// what its network could run; the schedule and the shifts are its too. The frame rule is the clip
/// rule, the rates are the model's output clock, and the canvases are the deployment's sizes.
fn describe(pipeline: &H3Pipeline, serving: &VideoServing) -> nn::Result<Description> {
    use crate::processing::{DenoiserInfo, ReferenceLimits, ServedCanvas};

    let denoiser = pipeline.network().ok_or_else(|| {
        nn::Error::Unsupported(
            "this media rank places no denoiser, so it serves no task".to_owned(),
        )
    })?;

    let tasks: Vec<VideoTask> = VideoTask::ALL
        .into_iter()
        .filter(|task| pipeline.tasks.contains(task))
        .collect();

    let steps = denoiser.schedule().num_steps()?;
    let shifts = &denoiser.config().schedule.shifts;

    let condition_packing = condition_packing(denoiser)?;

    let report = DenoiserInfo {
        tasks: tasks.iter().map(|task| task.as_str().to_owned()).collect(),
        schedule_points: steps + 1,
        video_shift: f64::from(shifts.video),
        audio_shift: f64::from(shifts.audio),
        frame_rate: FPS,
        audio_sample_rate: AUDIO_SAMPLE_RATE,
        audio_hop: AUDIO_HOP,
        frames: FRAMES,
        canvas_rule: CANVAS_RULE,
        canvases: deployment_canvases(&serving.sizes)?
            .into_iter()
            .map(|((width, height), canvas)| ServedCanvas {
                // Every canvas is generated at the rule's resolution class.
                short_edge: CANVAS_RULE.short_edge,
                aspect_ratio: format!("{width}:{height}"),
                canvas,
            })
            .collect(),
        references: if tasks.contains(&VideoTask::ReferenceToVideoAudio) {
            // The counts the checkpoint documents for a reference request.
            ReferenceLimits {
                images: 9,
                videos: 3,
                audios: 3,
                total: 12,
                image_short_edge: 2048,
            }
        } else {
            ReferenceLimits::default()
        },
        condition_packing,
    };
    report.validate().map_err(|reason| {
        nn::Error::Unsupported(format!("the denoiser's description: {reason}"))
    })?;

    Ok(Description {
        takes_conditions: tasks.iter().any(|task| task.conditioned()),
        denoise_steps: steps,
        details: report.encode().map_err(|error| {
            nn::Error::Unsupported(format!("encoding the description: {error}"))
        })?,
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
