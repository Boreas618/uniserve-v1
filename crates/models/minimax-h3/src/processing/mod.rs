//! MiniMax-H3's request processing: the rules its requests are held to, how each condition is
//! prepared and presented, and the processor a deployment binds ([`processor`]).
//!
//! [`plan`] holds the request rules: the canvas, the frame counts, how every condition is prepared,
//! what the conditioner reads of it and the denoiser rows it adds. [`presentation`] lists and
//! tokenizes what the conditioner reads. The model's workers describe their placed denoiser to
//! request processing as a [`DenoiserInfo`] (`crate::family`), which [`processor`] binds the
//! deployment's processor from; each planned condition reaches the workers in the pipeline's
//! shared vocabulary (`proto::video`).

pub mod plan;
pub mod presentation;
pub mod processor;

use proto::video::Canvas;
use serde::{Deserialize, Serialize};

/// MiniMax-H3 as request processing serves it.
pub const FAMILY: ::processing::Family = ::processing::Family {
    id: "minimax_h3",
    model_types: &[],
    pipeline_classes: &["MiniMaxH3ModularPipeline"],
    serving: ::processing::Serving::Video(&H3Family),
};

/// The family's one value, which [`FAMILY`] refers to.
struct H3Family;

impl ::processing::VideoFamily for H3Family {
    fn default_prompt_tokens(&self) -> u32 {
        // The checkpoint's text encoder accepts 262,144 positions; the default is the prompt
        // length a deployment prepares layouts and storage for.
        16_384
    }

    fn processor_files(&self) -> &'static [(&'static str, &'static str)] {
        processor::PROCESSOR_FILES
    }

    fn bind(
        &self,
        binding: ::processing::VideoBinding<'_>,
    ) -> Result<Box<dyn ::processing::VideoProcessor>, String> {
        Ok(Box::new(processor::H3Processor::bind(binding)?))
    }
}

/// How a model's video VAE divides a video's frames into clips.
///
/// A video has `min_frames + n * clip_frames` frames: one decode unit per
/// clip, so `n + 1` units. The VAE encodes each clip to `clip_latent_frames`
/// latent frames and the tail past the whole clips to `tail_latent_frames`.
/// The model states the rule and the worker declares it in its handshake;
/// every count a request derives from it, the generated frames, the decode
/// units and a reference video's encoded frames and latent units, follows
/// from these four values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FrameClips {
    /// Fewest frames of a video: one clip and the tail.
    pub min_frames: u32,
    /// Frames each further clip adds.
    pub clip_frames: u32,
    /// Latent frames one clip encodes to.
    pub clip_latent_frames: u32,
    /// Latent frames of the tail past the clips.
    pub tail_latent_frames: u32,
}

impl FrameClips {
    /// The decode units of a video of `frames` legal frames: one per clip.
    pub fn units(&self, frames: u32) -> u32 {
        frames.saturating_sub(self.min_frames) / self.clip_frames + 1
    }

    /// The fewest legal frame count that is at least `requested`.
    pub fn align_up(&self, requested: u32) -> u32 {
        let beyond = requested.saturating_sub(self.min_frames);
        self.min_frames + beyond.div_ceil(self.clip_frames) * self.clip_frames
    }

    /// The most legal frames of at most `available`, or `None` below the
    /// fewest.
    pub fn align_down(&self, available: u32) -> Option<u32> {
        let beyond = available.checked_sub(self.min_frames)?;
        Some(self.min_frames + beyond / self.clip_frames * self.clip_frames)
    }

    /// The latent frames the VAE encodes `frames` legal frames to.
    pub fn latent_frames(&self, frames: u32) -> u32 {
        self.units(frames) * self.clip_latent_frames + self.tail_latent_frames
    }

    /// Whether the rule is positive and leaves a tail past the first clip.
    pub fn is_valid(&self) -> bool {
        self.clip_frames > 0
            && self.min_frames > self.clip_frames
            && self.clip_latent_frames > 0
            && self.tail_latent_frames > 0
    }
}

/// A canvas a deployment serves, with the request fields that select it.
///
/// The model names each canvas it generates by the pair a request selects it
/// with; the rank declares the canvases its deployment prepares in its
/// handshake, in the order the operator named them, and the request rules
/// resolve a target against exactly those.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ServedCanvas {
    /// `target.short_edge`: the resolution class, in pixels, such as 768.
    pub short_edge: u32,
    /// `target.aspect_ratio`: the display ratio as `W:H`, such as `16:9`.
    pub aspect_ratio: String,
    /// The raster this pair selects.
    pub canvas: Canvas,
}

/// The references one `ref2va` request may carry, and the largest short
/// edge an image reference is encoded at, as a video denoiser declares them.
///
/// `total` bounds the references of every kind together, keyframes not
/// counted; a video's soundtrack counts with the video, and `audios` bounds
/// the audio references that are not one. A denoiser that takes no
/// references declares zeros.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReferenceLimits {
    /// Image references.
    pub images: u32,
    /// Video references, with or without soundtrack.
    pub videos: u32,
    /// Audio references that are not a video's soundtrack.
    pub audios: u32,
    /// References of every kind.
    pub total: u32,
    /// Short edge, in pixels, the denoiser encodes an image reference at, at
    /// most; a deployment may encode them smaller.
    pub image_short_edge: u32,
}

/// How a denoiser lays a request's conditions into its rows, and how many it
/// holds.
///
/// A sparse denoiser attends whole tiles, so each condition's audio and each
/// condition's pixels fill tiles of their own, and a reference video fills
/// the tiles of its token grid, which its rows need not fill. A dense
/// denoiser packs row for row: its tile is one row. What a request takes is
/// therefore the denoiser's arithmetic and not its rows' sum, and it decides
/// both whether the deployment serves the request and which of its graphs
/// does ([`plan::VideoParams::condition_rows`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ConditionPacking {
    /// Rows one request's conditions may take: the deployment's capacity,
    /// zero for a denoiser that takes none.
    pub capacity: u32,
    /// Rows of one tile.
    pub tile_rows: u32,
    /// Tokens of one video tile along time, height and width.
    pub video_tile: [u32; 3],
}

/// What the video denoiser a deployment places serves, as the worker that
/// denoises reports it to the frontend.
///
/// The denoiser fixes its schedule, and with it every request's step count;
/// the tasks, canvases and reference limits bound what a request may ask
/// for, and its condition packing what its conditions may take. The engine carries it JSON-encoded
/// ([`Self::encode`]), so a frontend reads the report of workers of another
/// build.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenoiserInfo {
    /// Task names the denoiser serves, in canonical order.
    pub tasks: Vec<String>,
    /// Sigma points of the fixed schedule, the clean endpoint included: one
    /// more than the network evaluations ([`Self::denoise_steps`]).
    pub schedule_points: u32,
    /// Shift of the video schedule.
    pub video_shift: f64,
    /// Shift of the audio schedule.
    pub audio_shift: f64,
    /// Frame rate, in frames per second, of every video the deployment
    /// generates: the output clock durations and reference videos are
    /// counted on.
    pub frame_rate: u32,
    /// Sample rate, in Hz, of the audio track generated with each video.
    pub audio_sample_rate: u32,
    /// How the VAE divides a video's frames into clips, which counts the
    /// frames the deployment generates and their decode units.
    pub frames: FrameClips,
    /// The only canvases the deployment serves: those it prepares, each
    /// with the `(short_edge, aspect_ratio)` its model names it by, in the
    /// order the deployment names them.
    pub canvases: Vec<ServedCanvas>,
    /// The references one request may carry and the largest short edge an
    /// image reference is encoded at; zeros for a denoiser that takes none.
    pub references: ReferenceLimits,
    /// How the denoiser packs a request's conditions and how many rows they
    /// may take.
    pub condition_packing: ConditionPacking,
}

impl DenoiserInfo {
    /// The network evaluations of the fixed schedule: one fewer than its
    /// sigma points, and the step count of every request.
    pub fn denoise_steps(&self) -> u32 {
        self.schedule_points.saturating_sub(1)
    }

    /// The JSON encoding the engine carries.
    ///
    /// # Errors
    ///
    /// Fails only when serialization fails, which these plain values do not.
    pub fn encode(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Decodes [`Self::encode`]'s encoding.
    ///
    /// # Errors
    ///
    /// Fails when `bytes` are not a denoiser's report.
    pub fn decode(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    /// Checks that tasks are named once each, the schedule has an endpoint
    /// and positive finite shifts, the frame and audio sample rates and the
    /// frame rule are declared, at least one canvas is prepared and each is a
    /// positive raster named by its own `(short_edge, aspect_ratio)` pair,
    /// and the reference limits bound each kind within the total, with an
    /// image edge where images are taken.
    pub fn validate(&self) -> Result<(), &'static str> {
        let mut names = std::collections::HashSet::new();
        if self.tasks.is_empty()
            || !self
                .tasks
                .iter()
                .all(|task| !task.is_empty() && names.insert(task.as_str()))
        {
            return Err("a video denoiser names each of its tasks once");
        }
        if self.schedule_points < 2
            || ![self.video_shift, self.audio_shift]
                .iter()
                .all(|shift| shift.is_finite() && *shift > 0.0)
        {
            return Err("a video denoiser schedule needs points and positive shifts");
        }
        if self.frame_rate == 0 {
            return Err("a video denoiser declares no frame rate");
        }
        if self.audio_sample_rate == 0 {
            return Err("a video denoiser declares no audio sample rate");
        }
        let packing = self.condition_packing;
        if packing.capacity > 0 && (packing.tile_rows == 0 || packing.video_tile.contains(&0)) {
            return Err("a video denoiser takes conditions and declares no tile to pack them in");
        }
        if !self.frames.is_valid() {
            return Err("a video denoiser declares an empty frame rule");
        }
        let references = &self.references;
        if references
            .images
            .max(references.videos)
            .max(references.audios)
            > references.total
            || (references.images > 0 && references.image_short_edge == 0)
        {
            return Err(
                "a video denoiser declares reference limits beyond their total or images without an edge",
            );
        }
        if self.canvases.is_empty() {
            return Err("a video denoiser prepared no canvas");
        }
        if !self.canvases.iter().all(|served| {
            served.short_edge > 0
                && !served.aspect_ratio.is_empty()
                && served.canvas.width > 0
                && served.canvas.height > 0
        }) {
            return Err("a video denoiser declares an empty canvas");
        }
        let mut named = std::collections::HashSet::new();
        if !self
            .canvases
            .iter()
            .all(|served| named.insert((served.short_edge, served.aspect_ratio.as_str())))
        {
            return Err("a video denoiser declares a canvas name twice");
        }
        Ok(())
    }
}
