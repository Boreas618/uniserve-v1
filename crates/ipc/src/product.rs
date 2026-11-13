//! Product names, as a main process's stages declare their outputs and a request's graph
//! asks for them.

use crate::batch::Phase;

/// The media reader's RGB24 pixels of every visual condition, `[pixels,
/// 3]` bytes in request order, each condition's frames whole.
pub const CONDITION_PIXELS: &str = "condition_pixels";
/// The media reader's stereo PCM of every audio track, `[samples, 2]` in
/// request order.
pub const CONDITION_SAMPLES: &str = "condition_samples";
/// The media reader's vision processor patches of every vision block,
/// `[patches, features]` in presentation order.
pub const VISION_PIXELS: &str = "vision_pixels";
/// The vision encoder's token features, one row per vision placeholder.
pub const VISION_FEATURES: &str = "vision_features";
/// The text encoder's prompt features, one row per presentation token.
pub const CONDITIONING: &str = "conditioning";
/// The latent encoder's visual condition rows in request order.
pub const CONDITION_VIDEO_LATENTS: &str = "condition_video_latents";
/// The latent encoder's audio condition rows in request order.
pub const CONDITION_AUDIO_LATENTS: &str = "condition_audio_latents";
/// The denoiser's final video latents.
pub const VIDEO_LATENTS: &str = "video_latents";
/// The denoiser's final audio latents.
pub const AUDIO_LATENTS: &str = "audio_latents";
/// The video decoder's RGB media units.
pub const VIDEO_UNITS: &str = "video_units";
/// The video decoder's splice tail of each media unit, `[3, overlap frames, rows, columns]`
/// per unit: the last frames of unit `n` that the decode of unit `n + 1` splices onto.
pub const VIDEO_TAIL: &str = "video_tail";
/// The video codec's encoded media unit rows.
pub const ENCODED_UNITS: &str = "encoded_units";
/// The audio decoder's PCM track.
pub const AUDIO_SAMPLES: &str = "audio_samples";

/// The product kinds of a video request, in the order of a session's product classes, each with
/// the phase whose calls write it and the phase whose calls read it (the declared results of a
/// video request's calls, `specs/scheduler-design.md` §3.3). The class of kind `k` is the `k`-th
/// of the classes just before the class of stored trajectories, and `Capacity::media_bytes`
/// states the bytes of a block of each kind in this order. The scheduler and the ranks read this
/// one list.
pub const KINDS: [(&str, Phase, Phase); 10] = [
    (CONDITION_PIXELS, Phase::MediaRead, Phase::LatentPrepare),
    (CONDITION_SAMPLES, Phase::MediaRead, Phase::LatentPrepare),
    (VISION_PIXELS, Phase::MediaRead, Phase::TextEncode),
    (CONDITIONING, Phase::TextEncode, Phase::LatentPrepare),
    (VIDEO_LATENTS, Phase::Denoise, Phase::VideoDecode),
    (AUDIO_LATENTS, Phase::Denoise, Phase::AudioDecode),
    (VIDEO_UNITS, Phase::VideoDecode, Phase::VideoEncode),
    (VIDEO_TAIL, Phase::VideoDecode, Phase::VideoDecode),
    (ENCODED_UNITS, Phase::VideoEncode, Phase::Mux),
    (AUDIO_SAMPLES, Phase::AudioDecode, Phase::AudioEncode),
];

/// The index of product kind `name` in [`KINDS`]; `None` for a name that is no kind of it.
pub fn kind(name: &str) -> Option<usize> {
    KINDS.iter().position(|(held, ..)| *held == name)
}
