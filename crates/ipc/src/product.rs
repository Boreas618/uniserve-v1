//! Product names, as a main process's components declare their outputs and a request's graph
//! asks for them.

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
/// The video codec's encoded media unit rows.
pub const ENCODED_UNITS: &str = "encoded_units";
/// The audio decoder's PCM track.
pub const AUDIO_SAMPLES: &str = "audio_samples";
