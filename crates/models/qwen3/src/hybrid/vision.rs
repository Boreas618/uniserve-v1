//! The vision tower of the Qwen3.5-MoE language model (`model.visual`): Qwen3-VL's tower and patch
//! merger without DeepStack features, sized by the checkpoint's Qwen-VL processor.
//!
//! An image is one block of merged rows in the request's context. The tower encodes it on its own
//! ([`nn::capability::Vision::encode_image`]), and a prefill places its rows where the prompt's
//! image placeholder stood, rotating them by the configuration's interleaved M-RoPE at the
//! coordinates of Qwen3-VL's `rope_index` ([`positions::rope_index`]): the block's first row takes
//! the logical position after the text before it on every axis, and its rows advance the height
//! and width axes by their merged row and column. The text after the block continues at the block's
//! start plus the larger of its merged sides.

use nn::capability::{ImageInput, PatchFormat, PixelNorm, StateTable, TableValues, Vision};
use nn::weights::WeightLoader;
use nn::{DType, Forward, TensorView};

use crate::config::VisionConfig;
use crate::processor::{ProcessorConfig, patch_format};
use crate::vl::positions;
use crate::vl::vision::{BlockRows, ImageTables, Outputs, VisionTower};
use crate::{Error, Result};

/// The rotary axes of the language model's M-RoPE: time, height and width.
const AXES: u32 = 3;

/// The processor rules a checkpoint states for its images: the patch grid an image resizes to and
/// the normalization of its pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageProcessor {
    /// The image and video processors' patch geometry and pixel budgets.
    pub config: ProcessorConfig,
    /// The image processor's `image_mean` and `image_std`.
    pub norm: PixelNorm,
}

/// The token text Qwen chat templates write for one image (`<|vision_start|><|image_pad|>
/// <|vision_end|>`), which the image's block replaces.
pub const PLACEHOLDER: &str = "<|image_pad|>";

/// The markup Qwen chat templates write for one image part.
pub const MARKUP: &str = "<|vision_start|><|image_pad|><|vision_end|>";

impl ImageProcessor {
    /// The block an image of `width` x `height` pixels takes in a request's context.
    ///
    /// # Errors
    ///
    /// An image the processor refuses to size.
    pub fn block(&self, width: u32, height: u32) -> Result<processing::ImageBlock> {
        let grid = self
            .config
            .image_grid(height, width)
            .map_err(|error| Error::Config(error.to_string()))?;
        let grid = [grid.t, grid.h, grid.w];
        let merge = self.config.merge_size;
        Ok(processing::ImageBlock {
            tokens: grid[0] * (grid[1] / merge) * (grid[2] / merge),
            positions: positions::span(grid, merge),
        })
    }
}

/// The image side of the Qwen3.5-MoE family as request processing serves it.
pub struct Images;

impl processing::ImageFamily for Images {
    fn processor_files(&self) -> &'static [&'static str] {
        &ImageProcessor::FILES
    }

    fn bind(
        &self,
        files: &[String],
    ) -> std::result::Result<Box<dyn processing::ImageProcessor>, String> {
        let [image, video] = files else {
            return Err(format!("{} processor files, two expected", files.len()));
        };
        let parse = |text: &str| serde_json::from_str::<serde_json::Value>(text);
        let image =
            parse(image).map_err(|error| format!("{}: {error}", ImageProcessor::FILES[0]))?;
        let video =
            parse(video).map_err(|error| format!("{}: {error}", ImageProcessor::FILES[1]))?;
        let processor =
            ImageProcessor::of_values(&image, &video).map_err(|error| error.to_string())?;
        Ok(Box::new(processor))
    }
}

impl processing::ImageProcessor for ImageProcessor {
    fn placeholder(&self) -> &str {
        PLACEHOLDER
    }

    fn markup(&self) -> &str {
        MARKUP
    }

    fn block(
        &self,
        width: u32,
        height: u32,
    ) -> std::result::Result<processing::ImageBlock, String> {
        ImageProcessor::block(self, width, height).map_err(|error| error.to_string())
    }
}

/// The tower, its configuration and the processor rules it reads images by.
pub struct HybridVision {
    config: VisionConfig,
    processor: ImageProcessor,
    format: PatchFormat,
    tower: VisionTower,
}

impl ImageProcessor {
    /// The image and video processor files the checkpoint names, relative to its root.
    pub const FILES: [&'static str; 2] =
        ["preprocessor_config.json", "video_preprocessor_config.json"];

    /// Read the checkpoint's processor files, or `None` when it states none: a checkpoint without
    /// them serves text alone.
    ///
    /// # Errors
    ///
    /// A file that is not a Qwen-VL processor config, or a normalization it does not state.
    pub fn open(checkpoint: &nn::Checkpoint) -> Result<Option<Self>> {
        let [image, video] = Self::FILES;
        let (Some(image), Some(video)) = (checkpoint.sidecar(image)?, checkpoint.sidecar(video)?)
        else {
            return Ok(None);
        };
        Self::of_values(&image, &video).map(Some)
    }

    /// The rules the parsed image and video processor files state.
    ///
    /// # Errors
    ///
    /// As [`Self::open`].
    pub fn of_values(image: &serde_json::Value, video: &serde_json::Value) -> Result<Self> {
        let config = ProcessorConfig::from_values(image, video)
            .map_err(|error| Error::Config(format!("the checkpoint's processor: {error}")))?;
        let norm = crate::processor::pixel_norm(&nn::config::Fields::of(Self::FILES[0], image)?)?;
        Ok(Self { config, norm })
    }
}

impl HybridVision {
    /// The tower of `config` in `dtype`, reading `model.visual` through `loader`, its images sized
    /// and normalized by `processor`.
    ///
    /// # Errors
    ///
    /// A tower with DeepStack features (no Qwen3.5 checkpoint states one, and the language model
    /// adds none), a processor whose patch geometry is not the tower's, or a module the loader
    /// refuses.
    pub fn load(
        loader: &mut WeightLoader<'_>,
        config: &VisionConfig,
        processor: ImageProcessor,
        dtype: DType,
    ) -> Result<Self> {
        if !config.deepstack_visual_indexes.is_empty() {
            return Err(Error::Config(
                "a hybrid stack adds no DeepStack features, but the vision tower states some"
                    .to_owned(),
            ));
        }
        let geometry = &processor.config;
        if (
            geometry.patch_size,
            geometry.temporal_patch_size,
            geometry.merge_size,
        ) != (
            config.patch_size,
            config.temporal_patch_size,
            config.spatial_merge_size,
        ) {
            return Err(Error::Config(format!(
                "the processor patches {}x{} over {} frames merging {}, the tower {}x{} over {} \
                 merging {}",
                geometry.patch_size,
                geometry.patch_size,
                geometry.temporal_patch_size,
                geometry.merge_size,
                config.patch_size,
                config.patch_size,
                config.temporal_patch_size,
                config.spatial_merge_size
            )));
        }
        Ok(Self {
            format: patch_format(config, processor.norm, dtype),
            tower: VisionTower::load(loader, config, dtype)?,
            config: config.clone(),
            processor,
        })
    }

    /// The row counts of one image of `grid`, which must be a single still.
    fn rows(&self, grid: [u32; 3]) -> Result<BlockRows> {
        if grid[0] != 1 {
            return Err(Error::Config(format!(
                "an image is one temporal patch, the grid {grid:?} states {}",
                grid[0]
            )));
        }
        Ok(BlockRows::of(&self.config, grid)?)
    }
}

impl Vision for HybridVision {
    fn image_grid(&self, height: u32, width: u32) -> nn::Result<[u32; 3]> {
        let grid = self
            .processor
            .config
            .image_grid(height, width)
            .map_err(|error| nn::Error::from(Error::Config(error.to_string())))?;
        Ok([grid.t, grid.h, grid.w])
    }

    fn patch_format(&self) -> PatchFormat {
        self.format
    }

    fn image_rows(&self, grid: [u32; 3]) -> nn::Result<u32> {
        Ok(self.rows(grid)?.merged)
    }

    fn max_image_rows(&self) -> u32 {
        // A resized image's area is at most the processor's pixel budget, in whole merged tokens.
        let geometry = &self.processor.config;
        let token = u64::from(geometry.patch_size * geometry.merge_size).pow(2);
        u32::try_from(geometry.image_max_pixels / token.max(1)).unwrap_or(u32::MAX)
    }

    fn position_axes(&self) -> u32 {
        AXES
    }

    fn image_positions(
        &self,
        grid: [u32; 3],
        start: u32,
        out: &mut [&mut [i32]],
    ) -> nn::Result<()> {
        let rows = self.rows(grid)?.merged as usize;
        if out.len() != AXES as usize || out.iter().any(|axis| axis.len() != rows) {
            return Err(nn::Error::from(Error::Config(format!(
                "{} coordinate rows for an image of {rows} rows over {AXES} axes",
                out.len()
            ))));
        }
        // The block alone, as `rope_index` places a run of placeholders at origin zero; the start
        // is added to every axis.
        const PLACEHOLDER: u32 = 1;
        let coordinates = positions::rope_index(
            &vec![PLACEHOLDER; rows],
            &[grid],
            &[],
            PLACEHOLDER,
            PLACEHOLDER + 1,
            self.config.spatial_merge_size,
        )?;
        for (row, coordinate) in coordinates.iter().enumerate() {
            for (axis, value) in coordinate.iter().enumerate() {
                out[axis][row] = i32::try_from(i64::from(start) + value).map_err(|_| {
                    nn::Error::from(Error::Config(
                        "an M-RoPE coordinate wider than an i32".to_owned(),
                    ))
                })?;
            }
        }
        Ok(())
    }

    fn image_tables(&self, grid: [u32; 3]) -> nn::Result<Vec<StateTable>> {
        let rows = self.rows(grid)?;
        Ok(crate::vl::image_table_specs(
            &self.config,
            u64::from(rows.patches),
            u64::from(rows.time) + 1,
        ))
    }

    fn prepare_image(&self, grid: [u32; 3], out: &mut [TableValues]) -> nn::Result<()> {
        let rows = self.rows(grid)?;
        Ok(crate::vl::write_image_tables(
            &self.config,
            &[grid],
            &[rows.patches],
            out,
        )?)
    }

    fn encode_image(
        &self,
        f: &Forward,
        grid: [u32; 3],
        pixels: TensorView,
        tables: &[TensorView],
        out: TensorView,
    ) -> nn::Result<()> {
        let rows = self.rows(grid)?;
        let image = ImageTables::new(
            &self.config,
            &ImageInput { pixels },
            rows.patches,
            rows.time,
            tables,
        )?;
        // The tower's buffers are dead once its features are written: the caller's arena takes
        // them back after the call.
        let mark = f.arena().mark();
        let buffers = self.tower.buffers(f.arena(), rows.patches)?;
        self.tower.encode(
            f,
            &image,
            &buffers,
            Outputs {
                features: out,
                deepstack: &[],
            },
        )?;
        f.arena().release(mark);
        Ok(())
    }
}
