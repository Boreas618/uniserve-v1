//! MiniMax-H3 video requests: their rules and their conditions.
//!
//! [`plan`] holds the request rules: the canvas, the frame counts, how every
//! condition is prepared, what the conditioner reads of it and the denoiser
//! rows it adds. [`presentation`] lists and tokenizes what the conditioner
//! reads. The rules exist once, here: the server plans requests with them
//! before admission, and the model package reads the canvas rule and the
//! frame counts from them through the worker library.
//!
//! The server plans every condition before admission. A [`VideoCondition`]
//! carries that plan to the workers: the shared-memory object holding the
//! fetched bytes, the exact media the worker's media reader decodes them
//! into, the conditioner's view, and the rows the latent encoders produce.
//! A request's [`Conditioning`] (its task, presentation tags and conditions)
//! crosses the engine encoded, beside the sizes the engine schedules the
//! calls that read and encode the conditions by; the engine never decodes
//! it. The workers describe their denoiser to the server the same way
//! ([`DenoiserInfo`]).

pub mod plan;
pub mod presentation;

use serde::{Deserialize, Serialize};

/// A video raster in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Canvas {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// A video-and-audio generation task, by its request name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoTask {
    /// Text to video and audio.
    T2va,
    /// First and/or last keyframe to video and audio.
    Fl2va,
    /// Image, video and audio references to video and audio.
    Ref2va,
}

impl VideoTask {
    /// Every task, in canonical order.
    pub const ALL: [Self; 3] = [Self::T2va, Self::Fl2va, Self::Ref2va];

    /// The task's request name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::T2va => "t2va",
            Self::Fl2va => "fl2va",
            Self::Ref2va => "ref2va",
        }
    }

    /// The task a request name denotes, if any.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|task| task.as_str() == name)
    }

    /// Whether the task's requests carry conditions.
    pub const fn conditioned(self) -> bool {
        !matches!(self, Self::T2va)
    }
}

/// How a condition relates to the generated video.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionRole {
    /// A keyframe anchoring the first generated frame.
    FirstFrame,
    /// A keyframe anchoring the last generated frame.
    LastFrame,
    /// A reference conditioning the generation as a whole.
    Reference,
}

impl ConditionRole {
    /// Whether the condition anchors a generated frame.
    pub const fn is_keyframe(self) -> bool {
        matches!(self, Self::FirstFrame | Self::LastFrame)
    }
}

/// Where a condition's fetched media bytes are: a POSIX shared-memory object
/// on the host that runs the server and the media reader.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MediaLocator {
    /// Object name without its leading `/`.
    pub name: String,
    /// Published byte count.
    pub bytes: u64,
}

/// A decoded still image put on a raster: resized with LANCZOS to `resized`,
/// then the `size` window whose top-left corner is (`left`, `top`) kept.
///
/// A stretched image is resized straight to its raster (`resized == size`,
/// no offset); a cover-cropped keyframe is resized past the canvas and
/// centred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImageFit {
    /// Size the decoded image is resized to.
    pub resized: Canvas,
    /// Left edge of the kept window in the resized image.
    pub left: u32,
    /// Top edge of the kept window in the resized image.
    pub top: u32,
    /// The kept raster, which the video encoder encodes.
    pub size: Canvas,
}

/// The frames of a reference video the media reader produces.
///
/// The video is resampled to the output frame rate and scaled onto `canvas`
/// in one decoding pass; its first `start_frame` frames are skipped and the
/// next `frames` kept. The conditioner samples the kept frames; the video encoder encodes
/// their leading `vae_frames`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VideoClip {
    /// The canvas of the video's own display aspect.
    pub canvas: Canvas,
    /// Frames of the output-rate timeline skipped at the start.
    pub start_frame: u32,
    /// Frames of the output-rate timeline kept after the start offset.
    pub frames: u32,
    /// The leading kept frames the video encoder encodes.
    pub vae_frames: u32,
}

/// The samples of a soundtrack or audio reference the media reader produces.
///
/// The track is decoded at its native rate, its first `start_sample` samples
/// skipped and the next `source_samples` kept, then resampled once to the
/// model's audio rate, giving `samples` stereo samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AudioClip {
    /// The track's native sample rate.
    pub sample_rate: u32,
    /// Native samples skipped at the start.
    pub start_sample: u64,
    /// Native samples kept after the start offset.
    pub source_samples: u64,
    /// Samples after resampling to the model's audio rate.
    pub samples: u64,
}

/// What the media reader decodes a condition's bytes into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionMedia {
    /// A keyframe or an image reference: one frame.
    Image(ImageFit),
    /// A reference video and, when the request conditions on it, its
    /// soundtrack.
    Video {
        /// The frames.
        clip: VideoClip,
        /// The soundtrack.
        soundtrack: Option<AudioClip>,
    },
    /// An audio reference.
    Audio(AudioClip),
}

impl ConditionMedia {
    /// The audio track the condition carries, if any.
    pub const fn audio(&self) -> Option<&AudioClip> {
        match self {
            Self::Image(_) => None,
            Self::Video { soundtrack, .. } => soundtrack.as_ref(),
            Self::Audio(clip) => Some(clip),
        }
    }

    /// The frame count and raster of the pixels the video encoder encodes,
    /// if the condition has any.
    pub const fn pixels(&self) -> Option<(u32, Canvas)> {
        match self {
            Self::Image(fit) => Some((1, fit.size)),
            Self::Video { clip, .. } => Some((clip.vae_frames, clip.canvas)),
            Self::Audio(_) => None,
        }
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

/// A vision patch grid: `t` temporal patches of `h` by `w` spatial patches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VisionGrid {
    /// Temporal patches; 1 for an image.
    pub t: u32,
    /// Patch rows.
    pub h: u32,
    /// Patch columns.
    pub w: u32,
}

impl VisionGrid {
    /// Patches of the grid.
    pub const fn patches(&self) -> u64 {
        self.t as u64 * self.h as u64 * self.w as u64
    }
}

/// What the conditioner reads of a visual condition.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConditionVision {
    /// The patch grid the condition's frames are resized to.
    pub grid: VisionGrid,
    /// Vision tokens the condition occupies in the presentation, which is
    /// the count of its vision placeholders.
    pub tokens: u32,
    /// The kept frames of a reference video the conditioner samples, in
    /// order; empty for an image.
    pub frame_indices: Vec<u32>,
}

/// One condition of a video request.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VideoCondition {
    /// Keyframe anchor or reference.
    pub role: ConditionRole,
    /// The fetched media bytes.
    pub source: MediaLocator,
    /// What the media reader decodes them into.
    pub media: ConditionMedia,
    /// What the conditioner reads, for a condition it reads.
    pub vision: Option<ConditionVision>,
    /// Denoiser rows each temporal unit of the condition's pixels encodes
    /// to, in unit order: one unit for a still image, one per video encoder
    /// window for a video. Empty for an audio reference.
    pub latent_units: Vec<u32>,
    /// Denoiser audio rows of the condition's audio track; zero without one.
    pub audio_rows: u32,
}

impl VideoCondition {
    /// Denoiser video rows of the condition.
    pub fn video_rows(&self) -> u64 {
        self.latent_units.iter().map(|rows| u64::from(*rows)).sum()
    }

    /// Bytes of the RGB24 pixels the video encoder encodes.
    pub fn pixel_bytes(&self) -> u64 {
        self.media.pixels().map_or(0, |(frames, canvas)| {
            u64::from(frames) * u64::from(canvas.width) * u64::from(canvas.height) * 3
        })
    }

    /// Checks the condition's internal consistency: a keyframe is one still
    /// image the conditioner may read; a condition with pixels encodes to
    /// positive rows in at least one unit and one without has none; a
    /// condition with an audio track encodes to positive audio rows and one
    /// without has none; a video's conditioner view samples kept frames in
    /// increasing order and an image's samples none; every extent is
    /// positive.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.source.name.is_empty() || self.source.name.contains('/') || self.source.bytes == 0 {
            return Err("a condition's media locator is invalid");
        }
        if self.role.is_keyframe() && !matches!(self.media, ConditionMedia::Image(_)) {
            return Err("a keyframe is one still image");
        }
        let canvases_positive = |canvas: Canvas| canvas.width > 0 && canvas.height > 0;
        match &self.media {
            ConditionMedia::Image(fit) => {
                if !canvases_positive(fit.size)
                    || !canvases_positive(fit.resized)
                    || u64::from(fit.left) + u64::from(fit.size.width)
                        > u64::from(fit.resized.width)
                    || u64::from(fit.top) + u64::from(fit.size.height)
                        > u64::from(fit.resized.height)
                {
                    return Err("an image fit keeps a window outside its resized image");
                }
            }
            ConditionMedia::Video { clip, .. } => {
                if !canvases_positive(clip.canvas)
                    || clip.vae_frames == 0
                    || clip.vae_frames > clip.frames
                {
                    return Err("a video clip encodes frames it does not keep");
                }
            }
            ConditionMedia::Audio(_) => {}
        }
        if let Some(track) = self.media.audio()
            && (track.sample_rate == 0 || track.source_samples == 0 || track.samples == 0)
        {
            return Err("an audio track keeps no samples");
        }
        let has_pixels = self.media.pixels().is_some();
        if has_pixels == self.latent_units.is_empty() || self.latent_units.contains(&0) {
            return Err("a condition's latent units disagree with its pixels");
        }
        if self.media.audio().is_some() != (self.audio_rows > 0) {
            return Err("a condition's audio rows disagree with its audio track");
        }
        if let Some(vision) = &self.vision {
            let video = match &self.media {
                ConditionMedia::Video { clip, .. } => Some(clip),
                ConditionMedia::Image(_) => None,
                ConditionMedia::Audio(_) => {
                    return Err("the conditioner reads no audio reference");
                }
            };
            if vision.grid.patches() == 0 || vision.tokens == 0 {
                return Err("a conditioner view covers no patches");
            }
            match video {
                None if !vision.frame_indices.is_empty() || vision.grid.t != 1 => {
                    return Err("an image's conditioner view samples one frame");
                }
                Some(clip)
                    if vision.frame_indices.is_empty()
                        || !vision
                            .frame_indices
                            .windows(2)
                            .all(|pair| pair[0] < pair[1])
                        || vision
                            .frame_indices
                            .last()
                            .is_some_and(|last| *last >= clip.frames) =>
                {
                    return Err("a video's conditioner view samples frames it does not keep");
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// What a video request brings to the denoiser beyond its prompt and its
/// sampling: its task, the denoiser's tag of each presentation token and its
/// conditions.
///
/// The frontend plans it and the engine carries it, JSON-encoded
/// ([`Self::encode`]), to the workers, which decode it ([`Self::decode`]);
/// the engine reads only the sizes it needs for scheduling. JSON keeps the
/// encoding readable by a frontend and workers of different builds as long
/// as fields are only added.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conditioning {
    /// The task the request runs.
    pub task: VideoTask,
    /// The denoiser's AdaLN tag of each presentation token: 0 for a vision
    /// token or vision marker, 1 for text.
    pub text_tags: Vec<u8>,
    /// The conditions in request order; empty for `t2va`.
    pub conditions: Vec<VideoCondition>,
}

impl Conditioning {
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
    /// Fails when `bytes` are not a conditioning.
    pub fn decode(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    /// Checks the conditioning of a request whose presentation holds
    /// `prompt_tokens` tokens: every token has a tag; `t2va` has no
    /// conditions, `fl2va` only keyframes and `ref2va` at least one
    /// condition; every condition is consistent
    /// ([`VideoCondition::validate`]).
    pub fn validate(&self, prompt_tokens: usize) -> Result<(), &'static str> {
        if self.text_tags.len() != prompt_tokens {
            return Err("every video prompt token requires one tag");
        }
        let served = self.conditions.iter().all(|condition| match self.task {
            VideoTask::T2va => false,
            VideoTask::Fl2va => condition.role.is_keyframe(),
            VideoTask::Ref2va => true,
        });
        if !served || (self.task != VideoTask::T2va && self.conditions.is_empty()) {
            return Err("the conditions do not fit the task");
        }
        self.conditions
            .iter()
            .try_for_each(VideoCondition::validate)
    }
}

/// What the video denoiser a deployment places serves, as the worker that
/// denoises reports it to the frontend.
///
/// The denoiser fixes its schedule, and with it every request's step count;
/// the tasks, canvases and reference limits bound what a request may ask
/// for. The worker that denoises a request admits its conditions into the
/// condition capacity it provisioned. The engine carries it JSON-encoded
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

#[cfg(test)]
mod tests {
    use super::*;

    /// One condition of each media kind: a cover-cropped keyframe, an image
    /// reference, a video reference with its soundtrack and an audio reference.
    fn reference_conditions() -> Vec<VideoCondition> {
        let canvas = Canvas {
            width: 1024,
            height: 768,
        };
        let source = |index: u32| MediaLocator {
            name: format!("uniserve-media-7-{index}"),
            bytes: 1_000 + u64::from(index),
        };
        let soundtrack = AudioClip {
            sample_rate: 44_100,
            start_sample: 22_050,
            source_samples: 40_425,
            samples: 29_334,
        };
        vec![
            VideoCondition {
                role: ConditionRole::LastFrame,
                source: source(0),
                media: ConditionMedia::Image(ImageFit {
                    resized: Canvas {
                        width: 1365,
                        height: 768,
                    },
                    left: 170,
                    top: 0,
                    size: canvas,
                }),
                vision: None,
                latent_units: vec![768],
                audio_rows: 0,
            },
            VideoCondition {
                role: ConditionRole::Reference,
                source: source(1),
                media: ConditionMedia::Image(ImageFit {
                    resized: Canvas {
                        width: 2048,
                        height: 2048,
                    },
                    left: 0,
                    top: 0,
                    size: Canvas {
                        width: 2048,
                        height: 2048,
                    },
                }),
                vision: Some(ConditionVision {
                    grid: VisionGrid {
                        t: 1,
                        h: 128,
                        w: 128,
                    },
                    tokens: 4096,
                    frame_indices: Vec::new(),
                }),
                latent_units: vec![4096],
                audio_rows: 0,
            },
            VideoCondition {
                role: ConditionRole::Reference,
                source: source(2),
                media: ConditionMedia::Video {
                    clip: VideoClip {
                        canvas,
                        start_frame: 12,
                        frames: 22,
                        vae_frames: 22,
                    },
                    soundtrack: Some(soundtrack),
                },
                vision: Some(ConditionVision {
                    grid: VisionGrid { t: 1, h: 32, w: 48 },
                    tokens: 384,
                    frame_indices: vec![0, 12],
                }),
                latent_units: vec![3840, 1536],
                audio_rows: 74,
            },
            VideoCondition {
                role: ConditionRole::Reference,
                source: source(3),
                media: ConditionMedia::Audio(soundtrack),
                vision: None,
                latent_units: Vec::new(),
                audio_rows: 74,
            },
        ]
    }

    /// A `ref2va` conditioning keeps its task, presentation tags and every
    /// condition's media, conditioner view and rows through its encoding.
    #[test]
    fn a_conditioning_round_trips_through_its_encoding() {
        let conditioning = Conditioning {
            task: VideoTask::Ref2va,
            text_tags: vec![1, 0, 0, 1],
            conditions: reference_conditions(),
        };
        let decoded = Conditioning::decode(&conditioning.encode().unwrap()).unwrap();
        assert_eq!(decoded, conditioning);
    }

    /// A conditioning whose tags do not cover its prompt, or whose `t2va`
    /// task carries conditions, is refused.
    #[test]
    fn an_inconsistent_conditioning_is_refused() {
        let conditioning = |task, text_tags: Vec<u8>, conditions| Conditioning {
            task,
            text_tags,
            conditions,
        };
        assert!(
            conditioning(VideoTask::T2va, vec![1, 1], Vec::new())
                .validate(3)
                .is_err()
        );
        assert!(
            conditioning(VideoTask::T2va, vec![1, 1, 1], reference_conditions())
                .validate(3)
                .is_err()
        );
        // The first condition is a keyframe, which `ref2va` also takes.
        assert!(
            conditioning(VideoTask::Ref2va, vec![1, 1, 1], reference_conditions())
                .validate(3)
                .is_ok()
        );
    }

    fn image(role: ConditionRole) -> VideoCondition {
        let canvas = Canvas {
            width: 64,
            height: 32,
        };
        VideoCondition {
            role,
            source: MediaLocator {
                name: "uniserve-media-1".to_owned(),
                bytes: 10,
            },
            media: ConditionMedia::Image(ImageFit {
                resized: canvas,
                left: 0,
                top: 0,
                size: canvas,
            }),
            vision: Some(ConditionVision {
                grid: VisionGrid { t: 1, h: 2, w: 4 },
                tokens: 2,
                frame_indices: Vec::new(),
            }),
            latent_units: vec![2],
            audio_rows: 0,
        }
    }

    /// A still image keyframe or reference is consistent, and its sizes
    /// follow from its raster and units.
    #[test]
    fn an_image_condition_is_consistent() {
        let condition = image(ConditionRole::FirstFrame);
        assert_eq!(condition.validate(), Ok(()));
        assert_eq!(condition.video_rows(), 2);
        assert_eq!(condition.pixel_bytes(), 64 * 32 * 3);
    }

    /// Inconsistent conditions are refused: a keyframe that is not an image,
    /// rows without pixels, and a video view sampling frames past the clip.
    #[test]
    fn inconsistent_conditions_are_refused() {
        let mut audio = image(ConditionRole::FirstFrame);
        audio.media = ConditionMedia::Audio(AudioClip {
            sample_rate: 48_000,
            start_sample: 0,
            source_samples: 48_000,
            samples: 32_000,
        });
        assert!(audio.validate().is_err());

        audio.role = ConditionRole::Reference;
        audio.vision = None;
        assert!(audio.validate().is_err(), "rows without pixels");
        audio.latent_units.clear();
        audio.audio_rows = 80;
        assert_eq!(audio.validate(), Ok(()));

        let mut video = image(ConditionRole::Reference);
        video.media = ConditionMedia::Video {
            clip: VideoClip {
                canvas: Canvas {
                    width: 64,
                    height: 32,
                },
                start_frame: 0,
                frames: 24,
                vae_frames: 22,
            },
            soundtrack: None,
        };
        video.vision = Some(ConditionVision {
            grid: VisionGrid { t: 1, h: 2, w: 4 },
            tokens: 2,
            frame_indices: vec![0, 12, 24],
        });
        assert!(video.validate().is_err(), "frame 24 is not kept");
    }
}
