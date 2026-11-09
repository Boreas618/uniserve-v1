//! The Qwen-VL processor: how an image or a video becomes the vision tower's patch grid and patch
//! rows.
//!
//! - [`ProcessorConfig`] is the checkpoint's image and video processor geometry, with the
//!   `smart_resize` of each ([`ProcessorConfig::image_grid`], [`ProcessorConfig::video_grid`]) and
//!   the tokens one temporal block of a grid presents ([`ProcessorConfig::block_tokens`]);
//! - [`sample_frames`] samples a clip at the video processor's 2 fps and labels its blocks;
//! - [`patch_format`] states how a clip's pixels become the tower's patch rows: normalized by the
//!   processor's mean and deviation, one channel-major tubelet per row, rows over merge blocks, and
//!   rounded once to the tower's element type; the runtime writes them.
//!
//! The arithmetic reproduces the transformers processors operation for operation, in `f64` where
//! they compute in Python floats and in `f32` where they compute in tensors, so that every rounding
//! lands on the same side.

use nn::DType;
use nn::capability::PixelNorm;
use nn::config::Fields;
use proto::video::VisionGrid;
use serde::{Deserialize, Serialize};

use crate::config::VisionConfig;
use crate::{Error, Result};

/// The rate the video processor samples a clip at, in frames per second.
pub const SAMPLE_FPS: f64 = 2.0;

/// The widest aspect a processor resizes: 200:1.
const MAX_ASPECT: f64 = 200.0;

/// A checkpoint's processor files that [`ProcessorConfig::parse`] cannot use.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The image processor config is not a processor config.
    #[error("parsing the image processor config")]
    Image(#[source] serde_json::Error),
    /// The video processor config is not a processor config.
    #[error("parsing the video processor config")]
    Video(#[source] serde_json::Error),
    /// The image and video processors patch differently.
    #[error("the image and video processors disagree on the patch geometry")]
    Geometry,
}

/// Media the processor cannot size, named in request terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum VisionError {
    /// An image or video frame whose sides differ by more than 200 times.
    #[error("an aspect of {width}x{height} exceeds 200:1")]
    Aspect {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// Video frames smaller than one merged vision token.
    #[error("video frames of {width}x{height} are too small")]
    Frames {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// A video too short for one temporal patch of samples.
    #[error("a video must sample at least {required} frames at {SAMPLE_FPS} fps, got {sampled}")]
    Sampled {
        /// Samples one temporal patch holds.
        required: usize,
        /// Samples the video yields.
        sampled: usize,
    },
}

/// The patch geometry and pixel budgets a Qwen-VL processor sizes media by.
///
/// It is the image and video processors' configuration, as transformers and SGLang name it (their
/// `VisionConfig` is the vision tower's model configuration). Read from the checkpoint's
/// `preprocessor_config.json` and `video_preprocessor_config.json` with [`ProcessorConfig::parse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Pixels per patch side.
    pub patch_size: u32,
    /// Frames merged into one temporal patch.
    pub temporal_patch_size: u32,
    /// Patches per merged vision token side.
    pub merge_size: u32,
    /// Lower pixel budget of an image.
    pub image_min_pixels: u64,
    /// Upper pixel budget of an image.
    pub image_max_pixels: u64,
    /// Lower budget of a video: frames times pixels.
    pub video_min_pixels: u64,
    /// Upper budget of a video: frames times pixels.
    pub video_max_pixels: u64,
}

#[derive(Deserialize)]
struct ProcessorFile {
    patch_size: u32,
    temporal_patch_size: u32,
    merge_size: u32,
    size: ProcessorSize,
}

#[derive(Deserialize)]
struct ProcessorSize {
    shortest_edge: u64,
    longest_edge: u64,
}

impl ProcessorConfig {
    /// Parses a checkpoint's image processor config (`preprocessor_config.json`) and video
    /// processor config (`video_preprocessor_config.json`), given as their texts.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] when a text is not a processor config, or when the two configs
    /// disagree on the patch geometry.
    pub fn parse(image: &str, video: &str) -> std::result::Result<Self, ConfigError> {
        Self::of_files(
            serde_json::from_str(image).map_err(ConfigError::Image)?,
            serde_json::from_str(video).map_err(ConfigError::Video)?,
        )
    }

    /// [`Self::parse`] over the two configs already read as JSON values, as the model's loading
    /// boundary holds its sidecars.
    ///
    /// # Errors
    ///
    /// As [`Self::parse`].
    pub fn from_values(
        image: &serde_json::Value,
        video: &serde_json::Value,
    ) -> std::result::Result<Self, ConfigError> {
        Self::of_files(
            ProcessorFile::deserialize(image).map_err(ConfigError::Image)?,
            ProcessorFile::deserialize(video).map_err(ConfigError::Video)?,
        )
    }

    /// The geometry the image and video processor files state, which must patch alike.
    fn of_files(
        image: ProcessorFile,
        video: ProcessorFile,
    ) -> std::result::Result<Self, ConfigError> {
        let geometry =
            |file: &ProcessorFile| (file.patch_size, file.temporal_patch_size, file.merge_size);
        if geometry(&image) != geometry(&video) {
            return Err(ConfigError::Geometry);
        }
        Ok(Self {
            patch_size: image.patch_size,
            temporal_patch_size: image.temporal_patch_size,
            merge_size: image.merge_size,
            image_min_pixels: image.size.shortest_edge,
            image_max_pixels: image.size.longest_edge,
            video_min_pixels: video.size.shortest_edge,
            video_max_pixels: video.size.longest_edge,
        })
    }

    /// What a resized side is a multiple of: one merged token.
    const fn factor(&self) -> u32 {
        self.patch_size * self.merge_size
    }

    /// The patch grid the image processor resizes an image to.
    ///
    /// This is Qwen2-VL's `smart_resize`: each side rounds half to even to the factor, and a size
    /// outside the pixel budget is rescaled with floor (above) or ceil (below) rounding.
    ///
    /// # Errors
    ///
    /// Returns [`VisionError::Aspect`] for an aspect beyond 200:1.
    pub fn image_grid(
        &self,
        height: u32,
        width: u32,
    ) -> std::result::Result<VisionGrid, VisionError> {
        check_aspect(height, width)?;
        let factor = f64::from(self.factor());
        let (height, width) = (f64::from(height), f64::from(width));
        let mut resized_height = (height / factor).round_ties_even() * factor;
        let mut resized_width = (width / factor).round_ties_even() * factor;
        if resized_height * resized_width > self.image_max_pixels as f64 {
            let beta = (height * width / self.image_max_pixels as f64).sqrt();
            resized_height = factor.max((height / beta / factor).floor() * factor);
            resized_width = factor.max((width / beta / factor).floor() * factor);
        } else if resized_height * resized_width < self.image_min_pixels as f64 {
            let beta = (self.image_min_pixels as f64 / (height * width)).sqrt();
            resized_height = (height * beta / factor).ceil() * factor;
            resized_width = (width * beta / factor).ceil() * factor;
        }
        Ok(self.grid(1, resized_height, resized_width))
    }

    /// The patch grid of `frames` sampled video frames.
    ///
    /// This is Qwen3-VL's video `smart_resize`: the budget covers all frames, which are padded to
    /// a whole temporal patch by repeating the last one.
    ///
    /// # Errors
    ///
    /// Returns [`VisionError::Frames`] for frames smaller than one merged token and
    /// [`VisionError::Aspect`] for an aspect beyond 200:1.
    pub fn video_grid(
        &self,
        frames: u32,
        height: u32,
        width: u32,
    ) -> std::result::Result<VisionGrid, VisionError> {
        let factor = self.factor();
        if height < factor || width < factor {
            return Err(VisionError::Frames { width, height });
        }
        check_aspect(height, width)?;
        let temporal = self.temporal_patch_size;
        let padded_frames = frames.div_ceil(temporal) * temporal;
        let factor = f64::from(factor);
        let pixels = f64::from(frames) * f64::from(height) * f64::from(width);
        let (height, width) = (f64::from(height), f64::from(width));
        let mut resized_height = (height / factor).round_ties_even() * factor;
        let mut resized_width = (width / factor).round_ties_even() * factor;
        let budget = f64::from(padded_frames) * resized_height * resized_width;
        if budget > self.video_max_pixels as f64 {
            let beta = (pixels / self.video_max_pixels as f64).sqrt();
            resized_height = factor.max((height / beta / factor).floor() * factor);
            resized_width = factor.max((width / beta / factor).floor() * factor);
        } else if budget < self.video_min_pixels as f64 {
            let beta = (self.video_min_pixels as f64 / pixels).sqrt();
            resized_height = (height * beta / factor).ceil() * factor;
            resized_width = (width * beta / factor).ceil() * factor;
        }
        Ok(self.grid(padded_frames / temporal, resized_height, resized_width))
    }

    /// Vision tokens of one temporal block of `grid`.
    pub const fn block_tokens(&self, grid: VisionGrid) -> u32 {
        grid.h * grid.w / (self.merge_size * self.merge_size)
    }

    fn grid(&self, t: u32, resized_height: f64, resized_width: f64) -> VisionGrid {
        // Resized sides are positive multiples of the factor below 2^32.
        VisionGrid {
            t,
            h: resized_height as u32 / self.patch_size,
            w: resized_width as u32 / self.patch_size,
        }
    }
}

/// Refuse an image or frame whose sides differ by more than 200 times.
fn check_aspect(height: u32, width: u32) -> std::result::Result<(), VisionError> {
    let (long, short) = (height.max(width), height.min(width));
    if f64::from(long) / f64::from(short) > MAX_ASPECT {
        return Err(VisionError::Aspect { width, height });
    }
    Ok(())
}

/// Samples a clip of `frames` frames at `frame_rate` frames per second at [`SAMPLE_FPS`], and labels
/// its temporal blocks.
///
/// Every `frame_rate / 2`-th frame is sampled (every twelfth at 24 fps), the index rounded half to
/// even as Python's `round` does. The tower merges the samples in groups of `temporal_patch`,
/// repeating the last one to fill a group, and a group is labelled with the mean of its first and
/// last timestamps. Answers the sampled frame indices and the block timestamps in seconds.
///
/// # Errors
///
/// Returns [`VisionError::Sampled`] when fewer frames are sampled than one group holds.
pub fn sample_frames(
    frames: u32,
    temporal_patch: u32,
    frame_rate: u32,
) -> std::result::Result<(Vec<u32>, Vec<f64>), VisionError> {
    let stride = f64::from(frame_rate) / SAMPLE_FPS;
    let mut indices: Vec<u32> = Vec::new();
    let mut cursor = 0.0_f64;
    loop {
        let index = cursor.round_ties_even();
        if index >= f64::from(frames) {
            break;
        }
        let index = index as u32;
        if indices.last().is_none_or(|&last| index > last) {
            indices.push(index);
        }
        cursor += stride;
    }
    let temporal_patch = temporal_patch as usize;
    if indices.len() < temporal_patch {
        return Err(VisionError::Sampled {
            required: temporal_patch,
            sampled: indices.len(),
        });
    }
    let mut timestamps: Vec<f64> = (0..indices.len())
        .map(|index| index as f64 / SAMPLE_FPS)
        .collect();
    let last = timestamps[timestamps.len() - 1];
    timestamps.resize(
        indices.len().div_ceil(temporal_patch) * temporal_patch,
        last,
    );
    let blocks = timestamps
        .chunks(temporal_patch)
        .map(|group| (group[0] + group[temporal_patch - 1]) / 2.0)
        .collect();
    Ok((indices, blocks))
}

/// The pixel normalization a processor's `image_mean` and `image_std` state: each RGB channel maps
/// `x` in `[0, 255]` to `(x / 255 - mean) / std` ([`nn::capability::PixelNorm`]).
///
/// # Errors
///
/// Fields of the wrong type or length, or a deviation that is not positive.
pub fn pixel_norm(values: &Fields<'_>) -> Result<PixelNorm> {
    let owner = values.owner();
    let mean = values.fixed_numbers("image_mean", 3)?;
    let std = values.fixed_numbers("image_std", 3)?;
    let mean: [f32; 3] = [mean[0] as f32, mean[1] as f32, mean[2] as f32];
    let std: [f32; 3] = [std[0] as f32, std[1] as f32, std[2] as f32];
    if std.iter().any(|value| *value <= 0.0) {
        return Err(Error::Config(format!(
            "{owner}.image_std must be positive, got {std:?}"
        )));
    }
    Ok(PixelNorm { mean, std })
}

/// How the processor lays a clip's pixels out as the vision tower's patch rows, which the runtime
/// writes ([`nn::capability::PatchFormat`]).
///
/// A row is one tubelet, channel-major (`channels · temporal · patch²`, the layout Qwen-VL
/// processors emit), each element `(byte - 255 · mean) / (255 · std)` in FP32 rounded once to the
/// tower's element type `dtype` (`TubeletEmbed`: `tubelets.to(self.weight.dtype)`). The rows run
/// over each temporal patch in merge-block order, and a temporal patch short of frames repeats the
/// clip's last frame: an image is one frame repeated.
pub fn patch_format(
    vision: &VisionConfig,
    norm: PixelNorm,
    dtype: DType,
) -> nn::capability::PatchFormat {
    nn::capability::PatchFormat {
        side: vision.patch_size,
        temporal: vision.temporal_patch_size,
        merge: vision.spatial_merge_size,
        channels: vision.in_channels,
        norm,
        dtype,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released processor: 16-pixel patches merged 2x2, an image budget of 64Ki to 16Mi
    /// pixels.
    fn processor() -> ProcessorConfig {
        ProcessorConfig::parse(
            r#"{"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "size": {"shortest_edge": 65536, "longest_edge": 16777216}}"#,
            r#"{"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "size": {"shortest_edge": 4096, "longest_edge": 25165824}}"#,
        )
        .expect("the released processor")
    }

    #[test]
    fn processors_that_patch_differently_are_refused() {
        let error = ProcessorConfig::parse(
            r#"{"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "size": {"shortest_edge": 1, "longest_edge": 2}}"#,
            r#"{"patch_size": 14, "temporal_patch_size": 2, "merge_size": 2,
                "size": {"shortest_edge": 1, "longest_edge": 2}}"#,
        )
        .expect_err("two patch sizes");
        assert!(matches!(error, ConfigError::Geometry), "{error}");
    }

    #[test]
    fn an_image_resizes_to_whole_merged_tokens_within_its_budget() {
        let processor = processor();
        // 768 x 1344 is 24 x 42 merged tokens of 32 pixels already: 48 x 84 patches.
        let grid = processor.image_grid(768, 1344).expect("a keyframe");
        assert_eq!((grid.t, grid.h, grid.w), (1, 48, 84));
        assert_eq!(processor.block_tokens(grid), 24 * 42);
        // Each side rounds half to even to 32 pixels: 784 / 32 = 24.5 rounds to 24.
        let grid = processor.image_grid(784, 1344).expect("an image");
        assert_eq!(grid.h, 48);
        // An image under the lower budget grows: 100 x 200 rounds to 96 x 192, below 65536
        // pixels, so each side scales by `sqrt(65536 / 20000)` and rounds up to whole tokens: 6
        // and 12 merged tokens.
        let grid = processor.image_grid(100, 200).expect("a small image");
        assert_eq!((grid.h, grid.w), (12, 24));
        assert!(matches!(
            processor.image_grid(1, 201),
            Err(VisionError::Aspect {
                width: 201,
                height: 1
            })
        ));
    }

    #[test]
    fn a_video_pads_its_frames_to_whole_temporal_patches() {
        let processor = processor();
        let grid = processor.video_grid(5, 256, 256).expect("five frames");
        assert_eq!((grid.t, grid.h, grid.w), (3, 16, 16));
        assert!(matches!(
            processor.video_grid(4, 16, 256),
            Err(VisionError::Frames { .. })
        ));
    }

    #[test]
    fn a_clip_is_sampled_at_two_frames_a_second_and_labelled_per_block() {
        // 48 frames at 24 fps: every twelfth frame, four samples in two blocks of two.
        let (indices, blocks) = sample_frames(48, 2, 24).expect("two seconds");
        assert_eq!(indices, vec![0, 12, 24, 36]);
        assert_eq!(blocks, vec![0.25, 1.25]);
        // Three samples fill a second block by repeating the last timestamp.
        let (indices, blocks) = sample_frames(25, 2, 24).expect("three samples");
        assert_eq!(indices, vec![0, 12, 24]);
        assert_eq!(blocks, vec![0.25, 1.0]);
        assert_eq!(
            sample_frames(12, 2, 24),
            Err(VisionError::Sampled {
                required: 2,
                sampled: 1
            })
        );
    }

    #[test]
    fn a_processor_states_three_positive_deviations() {
        let fields =
            serde_json::json!({"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.0, 0.5]});
        let error =
            pixel_norm(&Fields::of("preprocessor_config.json", &fields).expect("an object"))
                .expect_err("a zero deviation is refused");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("image_std")),
            "{error}"
        );
    }
}
