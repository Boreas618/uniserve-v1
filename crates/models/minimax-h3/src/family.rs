//! MiniMax-H3 and FastH3 as execution loads them ([`nn::family`]).
//!
//! [`FAMILY`] claims the `MiniMaxH3ModularPipeline` class. A checkpoint opens to its configuration
//! (an export and the base release it pins read as one), a load places the network the deployment
//! file's components name with the deployment's sequence and tensor splits, and the loaded
//! pipeline states the model's own rules to execution: the `17 * n + 5` timeline, its canvases, the
//! requests its denoiser graphs are captured for, and the description its request processing
//! plans requests against.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use nn::capability::{Decoder, Encoder, MediaFormat, WarmRequest};
use nn::family::{
    ConditionBounds, Description, Family, VideoCheckpoint, VideoFamily, VideoModel, VideoPlacement,
    VideoServing,
};
use nn::{Checkpoint, DType, WeightLoader};

use crate::config::{AttentionKind, Canvas, NAMED_CANVASES};
use crate::load::MinimaxH3;
use crate::output::VideoPostprocessor;
use crate::partition::Ulysses;
use crate::{
    AUDIO_CHANNELS, AUDIO_HOP, AUDIO_SAMPLE_RATE, CANVAS_MULTIPLE, CLIP_FRAMES, CLIP_LATENTS,
    Component, Config, FPS, Partition, VideoTask,
};

/// The MiniMax-H3 family.
pub const FAMILY: Family = Family::Video(&MinimaxH3Family);

/// The deployment component of the base network, which serves `t2va` and `fl2va`.
const DENOISER_COMPONENT: &str = "transformer";

/// The deployment component of the reference network, which serves `ref2va`.
const REFERENCE_COMPONENT: &str = "transformer_ref";

/// The deployment component of the text encoder.
const TEXT_COMPONENT: &str = "text_encoder";

/// The shortest output a request may ask for, in seconds.
pub const MIN_VIDEO_SECONDS: f64 = 4.0;

/// The family's one value, which [`FAMILY`] refers to.
struct MinimaxH3Family;

impl VideoFamily for MinimaxH3Family {
    fn classes(&self) -> &'static [&'static str] {
        &["MiniMaxH3ModularPipeline"]
    }

    fn open(&self, roots: &[PathBuf]) -> nn::Result<Box<dyn VideoCheckpoint>> {
        let (root, base) = match roots {
            [root] => (root, None),
            [root, base] => (root, Some(base.as_path())),
            _ => {
                return Err(nn::Error::Unsupported(format!(
                    "a MiniMax-H3 checkpoint is one directory and the base release it pins, not \
                     {} directories",
                    roots.len()
                )));
            }
        };
        let config = Config::read(root, base).map_err(model_error)?;
        Ok(Box::new(H3Checkpoint {
            root: root.clone(),
            config,
        }))
    }
}

/// A MiniMax-H3 checkpoint: its root and the configuration read from it and its base.
struct H3Checkpoint {
    root: PathBuf,
    config: Config,
}

impl VideoCheckpoint for H3Checkpoint {
    fn denoisers(&self) -> &'static [&'static str] {
        &[DENOISER_COMPONENT, REFERENCE_COMPONENT]
    }

    fn text_encoder(&self) -> &'static str {
        TEXT_COMPONENT
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
        let tasks = served_tasks(&config.serving(), placement.components.as_ref(), &self.root)?;

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

        let sequence = placement.sequence;
        let ulysses = Ulysses::of(sequence.split.size as usize, sequence.projection);
        let mut partition = Partition::serving(
            config,
            tasks[0],
            ulysses,
            placement.tensor.size as usize,
            placement.tensor.rank as usize,
            sequence.split.rank as usize,
        )
        .map_err(model_error)?;
        for task in &tasks[1..] {
            partition = partition.also_serving(config, *task).map_err(model_error)?;
        }

        // The condition capacity is the deployment's when it states one: the rows a request's
        // references take follow their rasters, which the task's own default cannot know.
        let partition = partition.sized(
            text_capacity,
            serving.condition_rows.unwrap_or(partition.condition_rows),
        );

        // **The postprocessor is placed though it holds no weights.** It is the video decoder's
        // second entry point, the clip splice, and a rank that held the decoder without it would
        // publish a timeline's first clip and say nothing.
        let partition = partition.placing(partition.blocks.with(Component::VideoPostprocessor));

        let canvases = deployment_canvases(&serving.sizes)?
            .into_iter()
            .map(|canvas| (canvas.width, canvas.height))
            .collect::<Vec<_>>();
        let model = crate::load(
            checkpoint,
            partition,
            Some(placement.dtype),
            &canvases,
            loader,
        )
        .map_err(model_error)?;

        // **One number, checked where both halves meet.** A model built for fewer text rows than
        // the deployment's longest prompt would refuse every longer prompt at the packed prefix,
        // minutes into a load and in a component that never saw the deployment's number.
        if let Some(denoiser) = model.transformer().or(model.transformer_ref()) {
            let held = denoiser.rules().arrangement().text_rows;
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
        self.config.text_encoder.hidden_size
    }

    fn denoiser(&self) -> Option<Arc<dyn nn::Denoiser + Send + Sync>> {
        self.network()
            .map(|denoiser| Arc::clone(denoiser) as Arc<dyn nn::Denoiser + Send + Sync>)
    }

    fn denoiser_component(&self) -> &'static str {
        if self.model.transformer_ref().is_some() {
            REFERENCE_COMPONENT
        } else {
            DENOISER_COMPONENT
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

    fn splicer(&self) -> Option<Arc<dyn Decoder + Send + Sync>> {
        self.model
            .video_postprocessor()
            .map(|splicer| Arc::clone(splicer) as Arc<dyn Decoder + Send + Sync>)
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
        crate::video_vae::decode_latents(&self.config.video_vae)
            * self.config.video_vae.temporal_compression()
    }

    fn format(&self) -> nn::Result<Arc<dyn MediaFormat>> {
        let format =
            crate::encoding::ConditionFormat::new(&self.config, self.dtype).map_err(model_error)?;
        Ok(Arc::new(format))
    }

    fn timeline(&self, seconds: f64) -> nn::Result<u32> {
        let frames = (seconds * f64::from(FPS)).round();
        if !(frames.is_finite() && frames >= 1.0) {
            return Err(nn::Error::Unsupported(format!(
                "{seconds} seconds is not a positive number of frames at {FPS} frames per second"
            )));
        }
        snap_frames(frames as u32)
    }

    fn video_units(&self, frames: u32) -> nn::Result<Vec<u32>> {
        // The postprocessor is the one home of the clip schedule.
        Ok(VideoPostprocessor::splices_of(frames)?
            .into_iter()
            .map(|splice| splice.keep)
            .collect())
    }

    fn condition_bounds(&self) -> ConditionBounds {
        // A deployment that places no condition encoder takes no condition at all.
        let rows = self
            .network()
            .filter(|denoiser| denoiser.places_encoders())
            .map_or(0, |denoiser| denoiser.rules().arrangement().condition_rows);

        // A condition row is one latent patch, `CANVAS_MULTIPLE` pixels square, of one latent
        // frame, which covers up to `CLIP_FRAMES / CLIP_LATENTS` output frames rounded up; a
        // track's latent frame is two rows.
        let frames = CLIP_FRAMES.div_ceil(CLIP_LATENTS);
        let merge = self.config.text_encoder.vision.spatial_merge_size;
        ConditionBounds {
            rows,
            pixels: rows.saturating_mul(CANVAS_MULTIPLE * CANVAS_MULTIPLE * frames),
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
        candidates(&self.tasks, serving, &ladder)
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
                REFERENCE_COMPONENT
            } else {
                DENOISER_COMPONENT
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

/// The least legal timeline of at least `frames` frames (`17 * n + 5`).
///
/// A timeline is one 17-frame body per clip plus a 5-frame tail (`VideoPostprocessor::splices_of`),
/// so the legal counts are 22, 39, 56, … : 4 s is 96 frames and snaps to 107, 15 s is 360 and snaps
/// to 362.
fn snap_frames(frames: u32) -> nn::Result<u32> {
    let tail = crate::output::Rgb::OVERLAP_FRAMES;
    if frames < CLIP_FRAMES + tail {
        return Err(nn::Error::Unsupported(format!(
            "{frames} frame(s) is shorter than a timeline, which is at least {} ({CLIP_FRAMES} \
             body frames and a {tail}-frame tail)",
            CLIP_FRAMES + tail
        )));
    }
    let beyond = frames - tail;
    Ok(tail + beyond.div_ceil(CLIP_FRAMES) * CLIP_FRAMES)
}

/// The named canvases of the sizes a deployment serves, in its order.
///
/// **One decoder serves the whole set**: the constants a decode reads are a tile's or a width's,
/// the same for every canvas, so a size of the set is a lookup and costs no device storage.
fn deployment_canvases(sizes: &[(u32, u32)]) -> nn::Result<Vec<&'static Canvas>> {
    if sizes.is_empty() {
        return Err(nn::Error::Unsupported(
            "this deployment serves no size, so no raster could be decoded".to_owned(),
        ));
    }
    sizes
        .iter()
        .map(|&(width, height)| {
            NAMED_CANVASES
                .iter()
                .find(|canvas| canvas.width == width && canvas.height == height)
                .ok_or_else(|| {
                    nn::Error::Unsupported(format!(
                        "this deployment serves {width}x{height}, which the checkpoint does not \
                         generate: it generates {}",
                        NAMED_CANVASES
                            .iter()
                            .map(|canvas| format!("{}x{}", canvas.width, canvas.height))
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
fn candidates(
    tasks: &[VideoTask],
    serving: &VideoServing,
    ladder: &[u32],
) -> nn::Result<Vec<WarmRequest>> {
    let longest = snap_frames(seconds_to_frames(serving.max_seconds)?)?;
    let shortest = snap_frames(seconds_to_frames(MIN_VIDEO_SECONDS)?)?;
    let timelines: Vec<u32> = (shortest..=longest).step_by(CLIP_FRAMES as usize).collect();

    let mut candidates = Vec::new();
    for canvas in deployment_canvases(&serving.sizes)? {
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
                        });
                    }
                }
            }
        }
    }
    Ok(candidates)
}

/// `seconds` at the model's frame rate, rounded.
fn seconds_to_frames(seconds: f64) -> nn::Result<u32> {
    let frames = (seconds * f64::from(FPS)).round();
    if !(frames.is_finite() && frames >= 1.0 && frames <= f64::from(u32::MAX)) {
        return Err(nn::Error::Unsupported(format!(
            "{seconds} seconds is not a positive number of frames at {FPS} frames per second"
        )));
    }
    Ok(frames as u32)
}

/// What the placed denoiser serves under `serving`, as the request processing plans requests
/// against it.
///
/// The tasks are the checkpoint's on the placed network, which a distilled student narrows from
/// what its network could run; the schedule and the shifts are its too. The frame rule is the clip
/// rule, the rates are the model's output clock, and the canvases are the deployment's sizes.
fn describe(pipeline: &H3Pipeline, serving: &VideoServing) -> nn::Result<Description> {
    use crate::processing::{
        ConditionPacking, DenoiserInfo, FrameClips, ReferenceLimits, ServedCanvas,
    };
    use proto::video::Canvas as Size;

    let denoiser = pipeline.network().ok_or_else(|| {
        nn::Error::Unsupported(
            "this media rank places no denoiser, so it serves no task".to_owned(),
        )
    })?;

    let tasks: Vec<VideoTask> = VideoTask::ALL
        .into_iter()
        .filter(|task| pipeline.tasks.contains(task))
        .collect();

    let steps = denoiser.schedule().num_steps().map_err(model_error)?;
    let shifts = &denoiser.config().schedule.shifts;

    // The timeline rule: the fewest frames are one whole clip and the tail, and the tail's latent
    // frames are what the timeline holds past the clip's.
    let min_frames = CLIP_FRAMES + CLIP_LATENTS;
    let latent_frames = crate::packing::video_latent_frames(min_frames).map_err(model_error)?;
    let frames = FrameClips {
        min_frames,
        clip_frames: CLIP_FRAMES,
        clip_latent_frames: CLIP_LATENTS,
        tail_latent_frames: latent_frames - CLIP_LATENTS,
    };

    // How the denoiser packs a request's conditions: its tiles under sparse attention, row for
    // row under dense attention.
    let (tile_rows, shape) = match denoiser.config().attention {
        AttentionKind::Sparse { tile, .. } => {
            (tile, crate::packing::tile_shape(tile).unwrap_or((1, 1, 1)))
        }
        AttentionKind::Dense => (1, (1, 1, 1)),
    };
    let condition_packing = ConditionPacking {
        capacity: if denoiser.places_encoders() {
            denoiser.rules().arrangement().condition_rows
        } else {
            0
        },
        tile_rows,
        video_tile: [shape.0, shape.1, shape.2],
    };

    let report = DenoiserInfo {
        tasks: tasks.iter().map(|task| task.as_str().to_owned()).collect(),
        schedule_points: steps + 1,
        video_shift: f64::from(shifts.video),
        audio_shift: f64::from(shifts.audio),
        frame_rate: FPS,
        audio_sample_rate: AUDIO_SAMPLE_RATE,
        frames,
        canvases: deployment_canvases(&serving.sizes)?
            .into_iter()
            .map(|canvas| ServedCanvas {
                short_edge: canvas.short_edge,
                aspect_ratio: canvas.name(),
                canvas: Size {
                    width: canvas.width,
                    height: canvas.height,
                },
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

/// The crate's error as the family contract states it.
fn model_error(error: crate::Error) -> nn::Error {
    nn::Error::Model(Box::new(error))
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
    /// it.
    #[test]
    fn a_duration_snaps_to_the_least_timeline_covering_it() {
        assert_eq!(snap_frames(96).unwrap(), 107, "4 s");
        assert_eq!(snap_frames(120).unwrap(), 124, "5 s");
        assert_eq!(snap_frames(360).unwrap(), 362, "15 s");
        assert_eq!(snap_frames(22).unwrap(), 22, "one clip and its tail");
        assert!(snap_frames(21).is_err(), "shorter than one clip");
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
}
