//! The Qwen-VL image and video processor rules: how an image or a video is sized to the vision
//! tower's patch grid, how a clip is sampled, and the context block an image takes.
//!
//! - [`ProcessorConfig`] is the checkpoint's image and video processor geometry, with the
//!   `smart_resize` of each ([`ProcessorConfig::image_grid`], [`ProcessorConfig::video_grid`]) and
//!   the tokens one temporal block of a grid presents ([`ProcessorConfig::block_tokens`]);
//! - [`sample_frames`] samples a clip at the video processor's 2 fps and labels its blocks;
//! - [`position_span`] is how far an image block advances the rotary position of the text after it.
//!
//! The arithmetic follows transformers' `Qwen2VLImageProcessor` and `Qwen3VLVideoProcessor`
//! operation for operation, in `f64` where they compute in Python floats, so that every rounding
//! lands on the same side. How the sized pixels become the tower's patch rows is the model's
//! (`qwen3::vision::patch_format`).

use proto::video::VisionGrid;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::image::{ImageBlock, ImageProcessor};

/// The rate the video processor samples a clip at, in frames per second.
pub const SAMPLE_FPS: f64 = 2.0;

/// The widest aspect a processor resizes: 200:1.
const MAX_ASPECT: f64 = 200.0;

/// The token text Qwen chat templates write for one image, which request processing replaces by
/// the image's block.
pub const PLACEHOLDER: &str = "<|image_pad|>";

/// The markup Qwen chat templates write for one image part.
pub const MARKUP: &str = "<|vision_start|><|image_pad|><|vision_end|>";

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

/// The patch geometry and pixel budgets a Qwen-VL processor sizes media by: the checkpoint's
/// `preprocessor_config.json` and `video_preprocessor_config.json` ([`ProcessorConfig::FILES`]).
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

impl ProcessorConfig {
    /// The image and video processor files a checkpoint names, relative to the directory that
    /// holds them.
    pub const FILES: [&'static str; 2] =
        ["preprocessor_config.json", "video_preprocessor_config.json"];

    /// Parses the image processor config and the video processor config, given as their texts in
    /// [`Self::FILES`]' order.
    ///
    /// # Errors
    ///
    /// A text that is not a processor config, or two configs that patch differently.
    pub fn parse(image: &str, video: &str) -> Result<Self, String> {
        let parse = |name: &str, text: &str| {
            serde_json::from_str::<Value>(text).map_err(|error| format!("{name}: {error}"))
        };
        Self::from_values(
            &parse(Self::FILES[0], image)?,
            &parse(Self::FILES[1], video)?,
        )
    }

    /// [`Self::parse`] over the two configs already read as JSON values.
    ///
    /// # Errors
    ///
    /// As [`Self::parse`].
    pub fn from_values(image: &Value, video: &Value) -> Result<Self, String> {
        let [image_file, video_file] = Self::FILES;
        let patching = |file: &str, config: &Value| -> Result<(u32, u32, u32), String> {
            Ok((
                count(file, config, "patch_size")?,
                count(file, config, "temporal_patch_size")?,
                count(file, config, "merge_size")?,
            ))
        };
        let (patch_size, temporal_patch_size, merge_size) = patching(image_file, image)?;
        if patching(video_file, video)? != (patch_size, temporal_patch_size, merge_size) {
            return Err("the image and video processors disagree on the patch geometry".to_owned());
        }
        let (image_min_pixels, image_max_pixels) = budget(image_file, image)?;
        let (video_min_pixels, video_max_pixels) = budget(video_file, video)?;
        Ok(Self {
            patch_size,
            temporal_patch_size,
            merge_size,
            image_min_pixels,
            image_max_pixels,
            video_min_pixels,
            video_max_pixels,
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
    pub fn image_grid(&self, height: u32, width: u32) -> Result<VisionGrid, VisionError> {
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
    ) -> Result<VisionGrid, VisionError> {
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

    /// Vision tokens of one temporal block of `grid`: its patches over one merge block's.
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

impl ImageProcessor for ProcessorConfig {
    fn placeholder(&self) -> &str {
        PLACEHOLDER
    }

    fn markup(&self) -> &str {
        MARKUP
    }

    /// The block an image takes in a context: its merged rows, and the positions the text after
    /// it advances by ([`position_span`]).
    fn block(&self, width: u32, height: u32) -> Result<ImageBlock, String> {
        let grid = self
            .image_grid(height, width)
            .map_err(|error| error.to_string())?;
        Ok(ImageBlock {
            tokens: grid.t * self.block_tokens(grid),
            positions: position_span(grid, self.merge_size),
        })
    }
}

/// The positions one vision block of `grid` (patches, whole `merge` blocks) advances the text
/// after it by: the larger of its merged spatial sides, as Qwen3-VL's `get_rope_index` advances
/// its origin past a vision block.
pub const fn position_span(grid: VisionGrid, merge: u32) -> u32 {
    let side = if grid.h > grid.w { grid.h } else { grid.w };
    side / merge
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
) -> Result<(Vec<u32>, Vec<f64>), VisionError> {
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

    // Pad the timestamps to whole groups with the last one, as the tower pads the samples.
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

/// Refuse an image or frame whose sides differ by more than 200 times.
fn check_aspect(height: u32, width: u32) -> Result<(), VisionError> {
    let (long, short) = (height.max(width), height.min(width));
    if f64::from(long) / f64::from(short) > MAX_ASPECT {
        return Err(VisionError::Aspect { width, height });
    }
    Ok(())
}

/// The positive integer `key` of the processor config `file`.
fn count(file: &str, config: &Value, key: &str) -> Result<u32, String> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{file}: `{key}` must be a positive integer"))
}

/// The `(shortest_edge, longest_edge)` pixel budget of the processor config `file`'s `size`.
fn budget(file: &str, config: &Value) -> Result<(u64, u64), String> {
    let edge = |key: &str| {
        config
            .get("size")
            .and_then(|size| size.get(key))
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value >= 0.0)
            // Pixel budgets are whole numbers; a JSON float states one exactly below 2^53.
            .map(|value| value as u64)
            .ok_or_else(|| format!("{file}: `size.{key}` must be a non-negative number"))
    };
    Ok((edge("shortest_edge")?, edge("longest_edge")?))
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
        assert!(error.contains("patch geometry"), "{error}");

        let missing = ProcessorConfig::parse(r#"{"patch_size": 16}"#, "{}").unwrap_err();
        assert!(missing.contains("temporal_patch_size"), "{missing}");
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
    fn an_image_block_takes_its_merged_rows_and_spans_its_longer_side() {
        let block = processor().block(1344, 768).expect("a keyframe");
        assert_eq!(block.tokens, 24 * 42);
        assert_eq!(block.positions, 42);
        assert!(processor().block(201, 1).is_err());
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
}
