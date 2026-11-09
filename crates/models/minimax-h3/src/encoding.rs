//! The H3 condition encoders: the causal video encoder that turns a clip of raster frames into
//! posterior moments, and the audio encoder that turns a reference waveform into latents.
//!
//! Only `ref2va` reads them: a reference video or audio clip is encoded once at
//! the request boundary and the latents join the packed sequence as clean condition rows. The rest of
//! this module is the encoders' arithmetic — the stage table, the causal extents, the tile grid and
//! the band rule a partial encoding follows — and their parameter tables. The numerical bodies follow
//! the shared layers, as the decoders' do.
//!
//! ## Precision
//!
//! Every convolution of the video encoder runs FP32 with TF32 allowed because the checkpoint's VAE
//! encoder is evaluated at the quality preset. The audio encoder's convolutions and its `Snake`
//! activations run FP32 and keep no TF32: its magnitudes are amplitudes, where the reference
//! compares samples, not feature maps.
//!
//! **The audio encoder's attention is FP32 throughout** — query, key and value projections, the
//! scores, the softmax and the output — whatever the checkpoint's preset says: FP32 attention for
//! `ref2va`'s audio encoder and the VSA compression branch, as a cuBLASLt SGEMM plus a softmax
//! kernel program. The model declares it instead of leaning on `torch.autocast`, and it is the
//! opposite of the video VAE decoder's rule (one element type for both parameters and residual,
//! whatever the deployment's preset - F32 under `dtype = "auto"`, because the checkpoint stores the
//! VAE in F32 and auto converts nothing; FP16 is a *converted* deployment's element, and every size
//! halves with it). The layer is [`nn::layers::Attention::fp32`]; no call site may construct the
//! dense backend for these blocks.
//!
//! ## The encoder's input
//!
//! A reference video reaches the encoder as `[frames, height, width, 3]` **uint8** pixels — the
//! same raster the decoder writes ([`crate::output::Rgb`]) — and is normalized per channel by
//! `(pixel / 255 − mean) / std` with the ImageNet constants before the input convolution. A clip
//! shorter than the checkpoint's [`crate::config::CLIP_FRAMES`] repeats its last frame; a single
//! frame stays alone.
//!
//! ## Causal extents
//!
//! A stride-`s` temporal convolution pads two zero frames in front and none behind, so it maps `T`
//! frames to `ceil(T / s)`: one 17-frame clip becomes five latent frames, and a single frame stays
//! one. A spatial downsampling pads the bottom and the right by one pixel and strides by two, so it
//! halves an even extent exactly and floors an odd one.
//!
//! ## Tiles and bands
//!
//! A tile is 256 input pixels square with at least 64 pixels of overlap, both multiples of the 16-pixel
//! spatial compression, so every seam spans whole latent positions. Adjacent tiles cross-fade over
//! `overlap / 16` latent positions in the encoder's output precision — unlike the decoder, which blends
//! decoded pixels, the encoder blends latents. A band of latent rows can be encoded on its own: the
//! tile rows it reads are chosen so that the band's rows equal those rows of the whole encoding.

/// `nn.LayerNorm`'s default epsilon, which the reference's audio projection does not override.
const NORM_EPS: f32 = 1e-5;
use crate::config::VideoVaeConfig;
use crate::video_vae::{MINIMUM_OVERLAP, Seams, TILE_PIXELS};
use crate::{Error, Result};
use ops::values::f32_to_bf16;

/// The spatial compression of the video encoder: input pixels per latent position.
pub const SPATIAL_RATIO: u32 = 16;

/// The channels of a condition's raster: RGB.
pub const PIXEL_CHANNELS: u32 = 3;

/// How H3's encoders read a request's decoded media, on the host.
///
/// Three layouts, one per encoder: the video encoder's normalised planar pixels, the vision tower's
/// patch rows and the audio encoder's zero-padded channel rows.
#[derive(Clone, Debug)]
pub struct ConditionFormat {
    vision: crate::config::VisionConfig,
    pixels: crate::config::PixelConfig,
    /// The element type the tower's patch rows are rounded to: its weights' own.
    dtype: DType,
    /// Samples of one audio latent frame.
    hop: u64,
}

impl ConditionFormat {
    /// The format of `config`'s encoders, with the tower at `dtype`.
    pub fn new(config: &crate::Config, dtype: DType) -> Result<Self> {
        Ok(Self {
            vision: config.text_encoder.vision.clone(),
            pixels: config.text_encoder.pixels,
            dtype,
            hop: u64::from(AudioStack::of(&config.audio_vae)?.samples_per_latent),
        })
    }
}

/// The threads a host stage on a request's critical path shares its work over.
fn workers() -> usize {
    std::thread::available_parallelism().map_or(1, |count| count.get().min(16))
}

/// Runs `work` over `items`, a run of them on each of `workers` threads.
fn share<T: Send>(items: &mut [T], workers: usize, work: impl Fn(&mut T) + Sync) {
    let run = items.len().div_ceil(workers.max(1)).max(1);
    let work = &work;
    std::thread::scope(|scope| {
        for items in items.chunks_mut(run) {
            scope.spawn(move || items.iter_mut().for_each(work));
        }
    });
}

impl nn::capability::MediaFormat for ConditionFormat {
    fn condition_pixels(
        &self,
        rgb: &[u8],
        frames: u32,
        height: u32,
        width: u32,
        out: &mut [f32],
    ) -> nn::Result<()> {
        let area = height as usize * width as usize;
        let frames = frames as usize;
        let channels = PIXEL_CHANNELS as usize;
        if rgb.len() != frames * area * channels || out.len() != rgb.len() {
            return Err(nn::Error::from(Error::Unsupported(format!(
                "{} RGB bytes and {} elements for {frames} frames of {height}x{width}",
                rgb.len(),
                out.len()
            ))));
        }
        // `[frames, rows, columns, 3]` to `[3, frames, rows, columns]`, each byte through the
        // normalisation the reference applies in FP32. A byte has 256 values, so the normalisation
        // is a table of them, and a frame's pixels are shared out in bands: this runs on a request's
        // critical path, and a reference image at its encoding size is twenty million elements.
        let mut table = [[0.0_f32; 256]; 3];
        for byte in 0..=u8::MAX {
            let values = normalize([byte; 3]);
            for (channel, value) in values.iter().enumerate() {
                table[channel][usize::from(byte)] = *value;
            }
        }
        let table = &table;
        let workers = workers();
        let band = area.div_ceil(workers.div_ceil(frames.max(1))).max(1);
        let mut planes: Vec<_> = out
            .chunks_mut(area)
            .map(|plane| plane.chunks_mut(band))
            .collect();
        let mut bands = Vec::new();
        for frame in 0..frames {
            let mut first = frame * area;
            while let (Some(red), Some(green), Some(blue)) = (
                planes[frame].next(),
                planes[frames + frame].next(),
                planes[2 * frames + frame].next(),
            ) {
                let pixels = red.len();
                bands.push((first, red, green, blue));
                first += pixels;
            }
        }
        share(&mut bands, workers, |(first, red, green, blue)| {
            let source = &rgb[*first * channels..(*first + red.len()) * channels];
            for (index, pixel) in source.chunks_exact(channels).enumerate() {
                red[index] = table[0][usize::from(pixel[0])];
                green[index] = table[1][usize::from(pixel[1])];
                blue[index] = table[2][usize::from(pixel[2])];
            }
        });
        Ok(())
    }

    fn patch_dtype(&self) -> DType {
        self.dtype
    }

    fn patch_width(&self) -> u32 {
        crate::vision::BlockRows::tubelet(&self.vision)
    }

    fn patch_side(&self) -> u32 {
        self.vision.patch_size
    }

    /// The processor's patch rows, op for op: `(byte - 255 · mean) / (255 · std)` in FP32, the last
    /// frame repeated to a whole temporal patch, and rows over each temporal patch in merge-block
    /// order, each one channel-major tubelet. The tower's projection rounds the pixels to its weights'
    /// type, so the row is stored rounded (`TubeletEmbed`: `tubelets.to(self.weight.dtype)`).
    fn patches(
        &self,
        planes: &[u8],
        frames: u32,
        grid: [u32; 3],
        out: &mut [u8],
    ) -> nn::Result<()> {
        let [time, rows, columns] = grid;
        let (patch, merge) = (self.vision.patch_size, self.vision.spatial_merge_size);
        let temporal = self.vision.temporal_patch_size;
        let channels = self.vision.in_channels as usize;
        let (height, width) = ((rows * patch) as usize, (columns * patch) as usize);
        let area = height * width;
        let width_bytes = self.patch_width() as usize * 2;
        let patches = time as usize * rows as usize * columns as usize;
        if self.dtype != DType::BF16 {
            return Err(nn::Error::from(Error::Unsupported(format!(
                "H3's vision tower reads BF16 patch rows, not {}",
                self.dtype
            ))));
        }
        if planes.len() != frames as usize * channels * area
            || out.len() != patches * width_bytes
            || frames == 0
            || frames.div_ceil(temporal) != time
        {
            return Err(nn::Error::from(Error::Unsupported(format!(
                "{} plane bytes and {} row bytes for {frames} frames on a {grid:?} grid",
                planes.len(),
                out.len()
            ))));
        }
        let mean = self.pixels.mean.map(|value| value * 255.0);
        let std = self.pixels.std.map(|value| value * 255.0);
        let (patch, merge, temporal) = (patch as usize, merge as usize, temporal as usize);
        // A byte has 256 values, so a channel's normalised BF16 is a table of them.
        let mut table = vec![[0_u8; 2]; channels * 256];
        for channel in 0..channels {
            for byte in 0..=u8::MAX {
                table[channel * 256 + usize::from(byte)] =
                    f32_to_bf16((f32::from(byte) - mean[channel]) / std[channel]).to_le_bytes();
            }
        }
        let table = &table;
        // One band is one temporal patch's row of merge blocks: its bytes follow its predecessor's,
        // so the bands are shared out as they lie in `out`.
        let block_rows = rows as usize / merge;
        let block_columns = columns as usize / merge;
        let band = patches / (time as usize * block_rows) * width_bytes;
        let mut bands: Vec<(usize, &mut [u8])> = out.chunks_mut(band).enumerate().collect();
        share(&mut bands, workers(), |(index, out)| {
            let (step, block_row) = (*index / block_rows, *index % block_rows);
            let mut at = 0_usize;
            for block_column in 0..block_columns {
                for merge_row in 0..merge {
                    for merge_column in 0..merge {
                        let top = (block_row * merge + merge_row) * patch;
                        let left = (block_column * merge + merge_column) * patch;
                        for channel in 0..channels {
                            for offset in 0..temporal {
                                // A short temporal patch repeats the last frame: one image is one
                                // frame repeated.
                                let frame = (step * temporal + offset).min(frames as usize - 1);
                                let plane = (frame * channels + channel) * area;
                                for y in 0..patch {
                                    let line = plane + (top + y) * width + left;
                                    for x in 0..patch {
                                        let bits =
                                            table[channel * 256 + usize::from(planes[line + x])];
                                        out[at] = bits[0];
                                        out[at + 1] = bits[1];
                                        at += 2;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
        Ok(())
    }

    fn track_samples(&self, samples: u64) -> nn::Result<u64> {
        padded_samples(samples, self.hop).map_err(nn::Error::from)
    }

    /// `[samples, 2]` interleaved to `[2, 1, padded]`, each channel's tail zero.
    fn track(&self, pcm: &[f32], samples: u64, out: &mut [f32]) -> nn::Result<()> {
        let padded = self.track_samples(samples)? as usize;
        let samples = samples as usize;
        if pcm.len() != 2 * samples || out.len() != 2 * padded {
            return Err(nn::Error::from(Error::Unsupported(format!(
                "{} PCM values and {} elements for a track of {samples} samples",
                pcm.len(),
                out.len()
            ))));
        }
        out.fill(0.0);
        for sample in 0..samples {
            out[sample] = pcm[2 * sample];
            out[padded + sample] = pcm[2 * sample + 1];
        }
        Ok(())
    }
}

/// The extents of one tensor of the encoder's stack, in frames, rows and columns.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Extent {
    /// Frames.
    pub frames: u32,
    /// Rows.
    pub height: u32,
    /// Columns.
    pub width: u32,
}

impl Extent {
    /// The extent one stage of the encoder's stack produces: `frame_stride` in time, and a halving in
    /// each spatial axis when `spatial_stride` is two.
    pub fn stage(self, frame_stride: u32, spatial_stride: u32) -> Self {
        let frames = if frame_stride <= 1 {
            self.frames
        } else {
            self.frames.div_ceil(frame_stride)
        };
        let halve = |extent: u32| {
            if spatial_stride <= 1 {
                extent
            } else {
                // Two pixels of padding in front (bottom and right), a three-pixel kernel and a
                // stride of two: `floor((extent + 1 - 3) / 2) + 1`.
                extent.saturating_sub(2) / 2 + 1
            }
        };
        Self {
            frames,
            height: halve(self.height),
            width: halve(self.width),
        }
    }
}

/// The encoder's stage table, read from the checkpoint's config once at the loading boundary.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Stack {
    /// The raster channels the input convolution reads.
    pub input_channels: u32,
    /// Each stage's width, in order.
    pub widths: Vec<u32>,
    /// Each stage's temporal stride.
    pub temporal_strides: Vec<u32>,
    /// Each stage's spatial stride.
    pub spatial_strides: Vec<u32>,
    /// Blocks per stage.
    pub blocks: u32,
    /// Latent channels.
    pub latent_channels: u32,
    /// The clip the network reads: its frames and what one clip yields.
    pub clip: Clip,
}

/// The clip a causal encoder cuts a raster into, and what its frames are worth.
///
/// Three numbers of one geometry, stated together because every use needs all three: the frames the
/// network reads at once, the latent frames one complete clip yields, and the latent frames a
/// condition drops from its last clip because a later clip of a generated video would overlap them.
/// A condition's own latent count follows from them ([`Clip::latent_frames`]), and packing takes this
/// value rather than three loose numbers so that the count has one statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Clip {
    /// Frames one clip of the encoder's raster is.
    pub frames: u32,
    /// Latent frames one complete clip yields.
    pub latents: u32,
    /// Latent frames a condition drops from its last clip.
    pub token_drop: u32,
}

impl Clip {
    /// The clip a checkpoint's video VAE config states.
    pub fn of(config: &VideoVaeConfig) -> Result<Self> {
        let clip = Self {
            frames: config.clip_length,
            latents: config.clip_latents(),
            token_drop: config.token_drop,
        };
        if clip.frames == 0 || clip.latents <= clip.token_drop {
            return Err(Error::config(format!(
                "a clip of {} frames yields {} latent frames and drops {}; the drop must leave some",
                clip.frames, clip.latents, clip.token_drop
            )));
        }
        Ok(clip)
    }

    /// The latent frames a **condition** of `frames` frames encodes to.
    ///
    /// A condition is cut into clips of [`Self::frames`] frames, each yielding [`Self::latents`] latent
    /// frames, and the last clip's trailing [`Self::token_drop`] latent frames are dropped - so 17
    /// frames are 2 and 22 are 7. A single frame is a still and yields one: the network's causal front
    /// padding makes one frame one latent frame, and the reference encodes
    /// an image or a keyframe through the same network alone.
    ///
    /// This is a **different statement** from a generated timeline's (`packing::video_latent_frames`:
    /// `17n + 5` frames to `5n + 2` latents). The two agree on the counts a generation uses, and only
    /// this one accepts the counts a condition may be - a condition is cut and padded, not generated.
    pub fn latent_frames(&self, frames: u32) -> Result<u32> {
        if frames == 0 {
            return Err(Error::config(
                "a video condition of no frame encodes nothing".to_owned(),
            ));
        }
        if frames == 1 {
            return Ok(1);
        }
        Ok(frames.div_ceil(self.frames) * self.latents - self.token_drop)
    }

    /// The clips a condition of `frames` frames is cut into, in order.
    ///
    /// Each clip is encoded alone, as the reference's `clips` and `latent_slices` cut it:
    /// the frames it reads and the latent frames it writes, the last clip
    /// holding what is left of both. A still is one clip of one frame.
    pub fn spans(&self, frames: u32) -> Result<Vec<ClipSpan>> {
        let total = self.latent_frames(frames)?;
        let mut spans = Vec::with_capacity(frames.div_ceil(self.frames) as usize);
        let (mut first, mut latent) = (0, 0);
        while first < frames {
            let length = self.frames.min(frames - first);
            let latents = self.latents.min(total - latent);
            spans.push(ClipSpan {
                first,
                frames: length,
                latent,
                latents,
            });
            first += length;
            latent += latents;
        }
        Ok(spans)
    }
}

impl Stack {
    /// Whether stage `index` ends in a downsampler, which is whether it halves space or strides time.
    ///
    /// The load builds the convolution from this answer and the call pads its input from the same
    /// one: a stage with both factors one has no convolution, so it has no padded frame either.
    pub fn downsamples(&self, index: usize) -> bool {
        self.temporal_strides[index] * self.spatial_strides[index] > 1
    }

    /// The stack the config states.
    pub fn of(config: &VideoVaeConfig) -> Result<Self> {
        let stages = config.block_out_channels.len();
        if config.temporal_downsample_factors.len() != stages
            || config.spatial_downsample_factors.len() != stages
        {
            return Err(Error::config(format!(
                "the video VAE's {stages} widths do not match its downsample factors"
            )));
        }
        if config.block_out_channels.is_empty() || config.layers_per_block == 0 {
            return Err(Error::config(
                "the video encoder has no stage or no block".to_owned(),
            ));
        }
        if config.latent_channels == 0 {
            return Err(Error::config(
                "the video VAE has no latent channel".to_owned(),
            ));
        }
        for (index, stride) in config
            .temporal_downsample_factors
            .iter()
            .chain(config.spatial_downsample_factors.iter())
            .enumerate()
        {
            if *stride == 0 {
                return Err(Error::config(format!(
                    "the video encoder's stride {} is zero",
                    index + 1
                )));
            }
        }
        let compression: u32 = config
            .spatial_downsample_factors
            .iter()
            .try_fold(1u32, |product, stride| product.checked_mul(*stride))
            .ok_or_else(|| Error::config("the video encoder's strides overflow".to_owned()))?;
        if compression != SPATIAL_RATIO {
            return Err(Error::config(format!(
                "the video encoder compresses space {compression}-fold, not {SPATIAL_RATIO}"
            )));
        }
        Ok(Self {
            input_channels: config.in_channels,
            widths: config.block_out_channels.clone(),
            temporal_strides: config.temporal_downsample_factors.clone(),
            spatial_strides: config.spatial_downsample_factors.clone(),
            blocks: config.layers_per_block,
            latent_channels: config.latent_channels,
            clip: Clip::of(config)?,
        })
    }

    /// The extents the stack visits for one clip: the raster, one entry per stage, and the latent
    /// raster the output convolution produces.
    pub fn extents(&self, raster: Extent) -> Vec<Extent> {
        let mut extents = vec![raster];
        let mut current = raster;
        for (temporal, spatial) in self
            .temporal_strides
            .iter()
            .zip(self.spatial_strides.iter())
        {
            current = current.stage(*temporal, *spatial);
            extents.push(current);
        }
        extents
    }

    /// The extent of the posterior moments: `2 * latent_channels` channels over the latent raster.
    pub fn moments(&self, raster: Extent) -> Result<Extent> {
        self.extents(raster)
            .last()
            .copied()
            .ok_or_else(|| Error::config("the video encoder's stack has no level".to_owned()))
    }
}

/// The tile grid of one raster, in input pixels, with the latent positions each seam shares.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Grid {
    /// The tile rows.
    pub rows: Seams,
    /// The tile columns.
    pub columns: Seams,
}

impl Grid {
    /// The grid of a raster whose extents are multiples of the spatial compression.
    pub fn of(height: u32, width: u32) -> Result<Self> {
        Ok(Self {
            rows: Seams::split(height, TILE_PIXELS, MINIMUM_OVERLAP, SPATIAL_RATIO)?,
            columns: Seams::split(width, TILE_PIXELS, MINIMUM_OVERLAP, SPATIAL_RATIO)?,
        })
    }

    /// The latent rows each tile row owns: where it starts in latent rows, then the latent height.
    ///
    /// A stitched tile row owns the latent rows from its own start up to the next row's start, so the
    /// first seams of a run cross-fade and the last row keeps its whole lower overlap.
    pub fn owned(&self, latent_height: u32) -> Vec<u32> {
        let mut owned: Vec<u32> = self
            .rows
            .starts
            .iter()
            .map(|start| start / SPATIAL_RATIO)
            .collect();
        owned.push(latent_height);
        owned
    }

    /// The tile rows a band of latent rows `[start, stop)` reads, as an inclusive range.
    ///
    /// The band's first tile row is the last one starting at or before `start`; it is preceded by one
    /// more row when `start` falls inside the cross-fade that row shares, because those latent rows
    /// read the row above. The last row is the last one starting before `stop`.
    pub fn band(&self, latent_height: u32, start: u32, stop: u32) -> Result<(usize, usize)> {
        if start >= stop || stop > latent_height {
            return Err(Error::config(format!(
                "the latent band [{start}, {stop}) is not a run of the {latent_height} rows"
            )));
        }
        let owned = self.owned(latent_height);
        let first = (0..self.rows.len())
            .rev()
            .find(|index| owned[*index] <= start)
            .ok_or_else(|| Error::config("the band starts before the first tile".to_owned()))?;
        let last = (0..self.rows.len())
            .rev()
            .find(|index| owned[*index] < stop)
            .ok_or_else(|| Error::config("the band ends before the first tile".to_owned()))?;
        let begin = match first.checked_sub(1) {
            Some(above) if start < owned[first] + self.rows.overlaps[above] / SPATIAL_RATIO => {
                above
            }
            _ => first,
        };
        Ok((begin, last))
    }

    /// The latent positions each seam of an axis cross-fades.
    pub fn blends(seams: &Seams) -> Vec<u32> {
        seams
            .overlaps
            .iter()
            .map(|overlap| overlap / SPATIAL_RATIO)
            .collect()
    }
}

/// One pixel's normalized channels: `(pixel / 255 − mean) / std`.
///
/// The inverse of the decoder's byte step (`value * std + mean`, clamped to [0, 1], scaled to 255
/// and rounded half to even), and the input the video encoder reads. A kernel that normalizes the
/// raster computes this; the round trip through the decoder's quantization recovers the byte.
pub fn normalize(pixel: [u8; 3]) -> [f32; 3] {
    let mut values = [0.0f32; 3];
    for (channel, byte) in pixel.iter().enumerate() {
        values[channel] = (f32::from(*byte) / crate::output::BYTE_MAX
            - crate::output::PIXEL_MEAN[channel])
            / crate::output::PIXEL_STD[channel];
    }
    values
}

/// The audio encoder's stack: DAC's strided widening network and the latent projection.
///
/// The decoder's rates are the encoder's reversed only in effect: the encoder's five strided stages
/// multiply to the 800 samples of one latent frame, so a whole number of latent frames in gives whole
/// latent frames out. `encoder_rates` are [2, 4, 4, 5, 5] and each stage doubles the width, starting
/// from `encoder_dim` after the input convolution.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AudioStack {
    /// Width after the input convolution.
    pub encoder_dim: u32,
    /// Each strided stage's rate, in order.
    pub rates: Vec<u32>,
    /// The width of each stage's output: `encoder_dim * 2^(stage + 1)`.
    pub widths: Vec<u32>,
    /// The encoder's features before the latent projection.
    pub latent_dim: u32,
    /// Latent channels.
    pub latent_channels: u32,
    /// Attention heads of the latent projection.
    pub heads: u32,
    /// Samples per latent frame: the product of the rates.
    pub samples_per_latent: u32,
}

impl AudioStack {
    /// The stack the config states.
    pub fn of(config: &crate::AudioVaeConfig) -> Result<Self> {
        if config.encoder_rates.is_empty() || config.encoder_dim == 0 {
            return Err(Error::config(
                "the audio encoder has no stage or no input width".to_owned(),
            ));
        }
        if config.latent_channels == 0 || config.num_attention_heads == 0 {
            return Err(Error::config(
                "the audio encoder has no latent channel or no head".to_owned(),
            ));
        }
        if !config.latent_dim.is_multiple_of(config.num_attention_heads) {
            return Err(Error::config(format!(
                "{} heads do not divide the audio encoder's width {}",
                config.num_attention_heads, config.latent_dim
            )));
        }
        let samples_per_latent = config
            .encoder_rates
            .iter()
            .try_fold(1u32, |product, rate| product.checked_mul(*rate))
            .ok_or_else(|| Error::config("the audio encoder's rates overflow".to_owned()))?;
        let mut width = config.encoder_dim;
        let widths = config
            .encoder_rates
            .iter()
            .map(|_| {
                width *= 2;
                width
            })
            .collect();
        Ok(Self {
            encoder_dim: config.encoder_dim,
            rates: config.encoder_rates.clone(),
            widths,
            latent_dim: config.latent_dim,
            latent_channels: config.latent_channels,
            heads: config.num_attention_heads,
            samples_per_latent,
        })
    }

    /// The length after each stage, from the samples to the latent frames.
    ///
    /// A strided stage is a convolution of `2 * rate` taps with `ceil(rate / 2)` samples of padding
    /// each side, so it maps `L` to `(L + 2 * ceil(rate / 2) - 2 * rate) / rate + 1`: 800 samples
    /// become 400, 100, 25, 5 and 1.
    pub fn lengths(&self, samples: u32) -> Vec<u32> {
        let mut lengths = vec![samples];
        let mut current = samples;
        for rate in &self.rates {
            let pad = rate.div_ceil(2);
            let padded = current + 2 * pad;
            current = if padded < 2 * rate {
                0
            } else {
                (padded - 2 * rate) / rate + 1
            };
            lengths.push(current);
        }
        lengths
    }

    /// The latent frames a track of `samples` encodes to, and the length the stack must be given.
    pub fn latent_frames(&self, samples: u32) -> Result<u32> {
        let frames = audio_latent_frames(u64::from(samples), u64::from(self.samples_per_latent))?;
        u32::try_from(frames)
            .map_err(|_| Error::config(format!("{samples} samples exceed a latent frame count")))
    }
}

/// The latent frames a track of `samples` encodes to: `ceil(samples / hop)`.
///
/// One statement of the rule, because two callers need it: the stack's geometry and the encoder's
/// own output extent. **This is the encode's direction:** a requested duration is covered, so the
/// track is `+0..hop-1` samples longer than the request and never short.
/// [`crate::packing::audio_latent_frames`] is the other direction - a request's video frame count
/// to its latents, rounded to nearest - and the two are deliberately named alike and documented
/// against each other, because a reader who takes one for the other gets a figure that looks right.
/// The ceiling is the reference's, and it is a ceiling because its `unpack_samples` right-pads the
/// track with zeros to [`padded_samples`] first, so every strided convolution divides exactly and
/// the last latent frame reads those zeros.
pub fn audio_latent_frames(samples: u64, hop: u64) -> Result<u64> {
    if samples == 0 || hop == 0 {
        return Err(Error::config(
            "an audio track of no samples encodes to no latent frames".to_owned(),
        ));
    }
    Ok(samples.div_ceil(hop))
}

/// The length the encoder must be given for a track of `samples`: the ceiling's own multiple.
///
/// The padding is [`audio_latent_frames`]' multiple less the track, so a whole-frame track is unchanged
/// and any other is right-padded with zeros exactly as the reference's `unpack_samples` does.
pub fn padded_samples(samples: u64, hop: u64) -> Result<u64> {
    Ok(audio_latent_frames(samples, hop)? * hop)
}

// ---------------------------------------------------------------------------------------------
// The video encoder's device path.
// ---------------------------------------------------------------------------------------------

use nn::Forward;
use nn::layers::{
    AdaptivePool, Attention, Conv1d, Conv3d, ConvGeometry, FrameMoments, FrameNormPad, FramePad,
    GatedActivation, LayerNorm, Linear, Posterior, Residual, Snake, Transpose, WeightOrder,
};
use nn::weights::{Scope, Weight, WeightLoader};
use nn::{DType, Shape, Strides, TensorView};

use crate::audio_vae::folded;

/// One causal convolution of the encoder and the normalization that feeds it.
///
/// The reference's `CausalConv3d.forward(norm=…)` normalizes each input frame, activates it, pads the
/// result with a reflection of the edge rows and columns and two zero frames in front, and convolves;
/// no output frame therefore reads a later input frame. The bias belongs to the convolution here,
/// where the reference adds it in a second pass to keep its rounding equal to `conv(x) + bias` — the
/// same value, one rounding apart.
struct Normalized {
    norm: FrameNormPad,
    conv: Conv3d,
    /// The source width the normalization reduces.
    channels: u32,
    /// The frame groups its statistics are taken over.
    groups: u32,
    /// The variance floor of those statistics.
    epsilon: f32,
}

/// One residual block: two normalized causal convolutions added to the input's projection.
struct Block {
    first: Normalized,
    second: Normalized,
    /// The one-by-one projection of the input when the width changes.
    shortcut: Option<Conv3d>,
}

/// One encoder level: its blocks, then the convolution that halves space and strides time.
struct Level {
    blocks: Vec<Block>,
    downsample: Option<Conv3d>,
}

/// The video encoder: the causal convolutional network that turns a clip's pixels into the posterior
/// moments of its latent.
///
/// One call encodes one raster: the caller tiles a clip as `Grid` states, and a band of latent rows
/// may be encoded on its own through `Grid::band`. Every convolution runs FP32 with TF32 allowed, the
/// precision the checkpoint's preset declares for the video VAE's encoder.
pub struct VideoEncoder {
    stack: Stack,
    /// The input convolution, over raw pixels.
    input: Conv3d,
    levels: Vec<Level>,
    /// The final normalization and the moment convolution, `2 · latent_channels` wide.
    output: Normalized,
    /// The one-by-one projection after the moments.
    moments: Conv3d,
    /// The latent space's per-channel affine, `(sample − mean) / std` as a one-by-one convolution.
    normalize: Conv3d,
    /// The posterior's epilogue: the sample a condition reads, from the projected moments.
    sampler: Posterior,
    /// Whether the spatial padding mirrors the edge rows and columns (the checkpoint's mode) or
    /// replicates them.
    reflect: bool,
}

/// The geometry every causal convolution of the encoder is built for: three taps, no padding of its
/// own (the frame pad supplies it) and unit dilation.
fn geometry(kernel: u32, stride: [u32; 3]) -> Result<ConvGeometry> {
    Ok(ConvGeometry::new(
        3,
        [kernel; 3],
        stride,
        [0; 3],
        [1; 3],
        1,
    )?)
}

/// The per-channel affine of the latent space, in the encoder's direction: `(sample − mean) / std`.
///
/// Multiplying by `1 / std` and shifting by `−mean / std` is a one-by-one convolution whose weight is
/// the diagonal, so the encoder undoes the decoder's `value · std + mean` with one operator of the
/// same shape as the projection beside it (`audio_vae::denormalize_tables` is that other direction).
///
/// **Declared deviation.** The reference's `(lat - mean) / std` is a subtraction and a division in
/// f32 - two roundings - while a convolution's multiply-add is one. The values agree to at most 1 ulp
/// of the f32 sample (6e-8 relative), against the 1e-4 the conditioned path is compared at; the
/// epilogue before it is bit-exact, so this is the only step that differs from torch.
pub fn normalize_tables(mean: &[f32], std: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
    let channels = mean.len();
    if channels == 0 || std.len() != channels {
        return Err(Error::config(format!(
            "the latent statistics hold {} means and {} deviations",
            mean.len(),
            std.len()
        )));
    }
    let mut scale = vec![0.0_f32; channels * channels];
    let mut shift = vec![0.0_f32; channels];
    for channel in 0..channels {
        let deviation = std[channel];
        if !deviation.is_finite() || deviation <= 0.0 {
            return Err(Error::config(format!(
                "latent channel {channel} has a deviation of {deviation}, which cannot be divided by"
            )));
        }
        scale[channel * channels + channel] = 1.0 / deviation;
        shift[channel] = -mean[channel] / deviation;
    }
    Ok((scale, shift))
}

/// The geometry of one posterior sample: the positions it reads and the channels it writes.
///
/// The sample is `[latent_channels, frames, rows, columns]` like every other raster here, and
/// `ops::posterior` reads `[positions, channels]` - so the call states the positions
/// (`frames · rows · columns`) once, here, and both the strides and the operator's channel count come
/// from it. Every refusal is on this type rather than inside a forward, because a raster of the wrong
/// channel count is a configuration mistake and not a runtime condition.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Sample {
    /// Positions the sample has: `frames · rows · columns`.
    pub positions: u64,
    /// Latent channels of one position.
    pub channels: u64,
}

impl Sample {
    /// The sample of `latent_channels` channels over the raster `moments`, `noise` and `out` share.
    ///
    /// `moments` holds both fields of the posterior, so it has twice the sample's channels; `noise` and
    /// `out` hold the sample's own. All three are FP32 rasters of one geometry, packed as the encoder
    /// writes them: the call addresses them through two strides, and a plane that is not packed has no
    /// single pair of them.
    pub fn of(
        latent_channels: u32,
        moments: TensorView,
        noise: TensorView,
        out: TensorView,
    ) -> Result<Self> {
        let channels = u64::from(latent_channels);
        if channels == 0 {
            return Err(Error::config(
                "a latent raster with no channels has no sample".to_owned(),
            ));
        }
        let wide = 2 * channels;
        for (name, view, expected) in [
            ("moments", moments, wide),
            ("noise", noise, channels),
            ("out", out, channels),
        ] {
            if view.dtype != DType::F32 {
                return Err(Error::config(format!(
                    "the {name} raster is {}, and a condition's posterior is FP32 throughout: the \
                     reference draws and samples in f32",
                    view.dtype
                )));
            }
            if view.shape.rank() != 4 {
                return Err(Error::config(format!(
                    "the {name} raster is {:?}, a [channels, frames, rows, columns] raster expected",
                    view.shape.dims()
                )));
            }
            if view.shape.dim(0) != expected {
                return Err(Error::config(format!(
                    "the {name} raster has {} channels, {expected} expected for a {latent_channels}-\
                     channel latent",
                    view.shape.dim(0)
                )));
            }
            let packed = Strides::row_major(view.shape);
            if view.strides != packed {
                return Err(Error::config(format!(
                    "the {name} raster's strides are {:?} over {:?}: a channel-major sample \
                     addresses a packed raster through two strides, and a view is refused rather than \
                     copied",
                    view.strides.as_slice(),
                    view.shape.dims()
                )));
            }
        }
        if noise.shape != out.shape || moments.shape.dims()[1..] != out.shape.dims()[1..] {
            return Err(Error::config(format!(
                "the posterior's rasters disagree: moments {:?}, noise {:?}, out {:?}",
                moments.shape.dims(),
                noise.shape.dims(),
                out.shape.dims()
            )));
        }
        let positions = moments.shape.dims()[1..].iter().product::<u64>();
        Ok(Self {
            positions,
            channels,
        })
    }
}

/// Read one convolution of the checkpoint: `from` input channels, `to` output, `kernel` taps.
///
/// **`channels_last` is the operand's storage, not a preference.** The kernel derives every
/// operand's strides from the flag, so a convolution declared channels-last over a channels-first
/// buffer is wrong numbers rather than a refusal (`nn::Conv3d::forward` refuses it).
///
/// **Every convolution here is channel-first, and the reference says why that costs no number.**
/// Its stride-1 padded convolutions run under `torch.channels_last_3d` as a cuDNN speed path: "it
/// selects the same algorithm for either order, which leaves the encoding unchanged".
/// The order is therefore a throughput choice
/// rather than a fidelity one, and this encoder takes it only against a measured encode-convolution
/// time. The same comment is why its strided downsamplers stay channel-first even there: under
/// channels-last cuDNN picks another algorithm "whose rounding differs", so the reference keeps
/// those two out of the speed path itself.
fn convolution(
    scope: &mut Scope<'_, '_>,
    from: u32,
    to: u32,
    kernel: u32,
    stride: [u32; 3],
    channels_last: bool,
) -> Result<Conv3d> {
    let weight = scope.tensor(
        "weight",
        &[
            u64::from(to),
            u64::from(from),
            u64::from(kernel),
            u64::from(kernel),
            u64::from(kernel),
        ],
    )?;
    let bias = scope.tensor("bias", &[u64::from(to)])?;
    Ok(Conv3d::new(
        weight,
        Some(bias),
        geometry(kernel, stride)?,
        // `tf32`: the encoder's convolutions are FP32 and the reference keeps cuDNN's TF32 for
        // them (design section 9). Stated with its name because a bare `true` beside another bool
        // reads as neither, and a review of this crate took it for absent while it was here.
        /* tf32 */
        true,
        /* channels_last */
        channels_last,
    )?)
}

impl VideoEncoder {
    /// Read the encoder's weights. Every convolution is stored plain, so nothing is folded.
    pub fn load(config: &VideoVaeConfig, loader: &mut WeightLoader) -> Result<Self> {
        let stack = Stack::of(config)?;
        let groups = config.norm_num_groups;
        let epsilon = config.norm_eps;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::config(format!(
                "the video encoder's normalization epsilon is {epsilon}"
            )));
        }
        let reflect = matches!(
            config.spatial_padding_mode,
            crate::config::SpatialPadding::Reflect
        );
        // A normalized convolution reads its affine parameters from `norm` and its filter from `conv`.
        let normalized = |loader: &mut WeightLoader,
                          norm_name: &str,
                          conv_name: &str,
                          channels: u32,
                          to: u32,
                          kernel: u32|
         -> Result<Normalized> {
            let mut norm = loader.scope(norm_name);
            let weight = norm.tensor("weight", &[u64::from(channels)])?;
            let bias = norm.tensor("bias", &[u64::from(channels)])?;
            Ok(Normalized {
                norm: FrameNormPad::new(u64::from(groups), Some(weight), Some(bias), reflect)?,
                conv: convolution(
                    &mut loader.scope(conv_name),
                    channels,
                    to,
                    kernel,
                    [1; 3],
                    false,
                )?,
                channels,
                groups,
                epsilon,
            })
        };

        let input = convolution(
            &mut loader.scope("encoder.conv_in"),
            stack.input_channels,
            stack.widths[0],
            3,
            [1; 3],
            false,
        )?;
        let mut width = stack.widths[0];
        let mut levels = Vec::with_capacity(stack.widths.len());
        for (index, out) in stack.widths.iter().enumerate() {
            let stage = format!("encoder.down_blocks.{index}");
            let mut blocks = Vec::with_capacity(stack.blocks as usize);
            for block in 0..stack.blocks {
                let base = format!("{stage}.resnets.{block}");
                let from = if block == 0 { width } else { *out };
                let first = normalized(
                    loader,
                    &format!("{base}.norm1"),
                    &format!("{base}.conv1"),
                    from,
                    *out,
                    3,
                )?;
                let second = normalized(
                    loader,
                    &format!("{base}.norm2"),
                    &format!("{base}.conv2"),
                    *out,
                    *out,
                    3,
                )?;
                let shortcut = if from == *out {
                    None
                } else {
                    Some(convolution(
                        &mut loader.scope(format!("{base}.conv_shortcut")),
                        from,
                        *out,
                        1,
                        [1; 3],
                        false,
                    )?)
                };
                blocks.push(Block {
                    first,
                    second,
                    shortcut,
                });
            }
            let stride = [
                stack.temporal_strides[index],
                stack.spatial_strides[index],
                stack.spatial_strides[index],
            ];
            let downsample = if stack.downsamples(index) {
                Some(convolution(
                    &mut loader.scope(format!("{stage}.downsamplers.0.conv")),
                    *out,
                    *out,
                    3,
                    stride,
                    false,
                )?)
            } else {
                None
            };
            levels.push(Level { blocks, downsample });
            width = *out;
        }
        let output = normalized(
            loader,
            "encoder.norm_out",
            "encoder.conv_out",
            width,
            2 * stack.latent_channels,
            3,
        )?;
        let moments = convolution(
            &mut loader.scope("quant_conv"),
            2 * stack.latent_channels,
            2 * stack.latent_channels,
            1,
            [1; 3],
            false,
        )?;
        // The latent space's affine: the checkpoint states its statistics, not a tensor for them.
        let latent = u64::from(stack.latent_channels);
        let (scale, shift) = normalize_tables(&config.latents_mean, &config.latents_std)?;
        let mut constants = loader.scope("video_vae.normalize");
        let normalize = Conv3d::new(
            constants.constant_f32("weight", &[latent, latent, 1, 1, 1], &scale)?,
            Some(constants.constant_f32("bias", &[latent], &shift)?),
            geometry(1, [1; 3])?,
            true,
            false,
        )?;
        Ok(Self {
            stack,
            input,
            levels,
            output,
            moments,
            normalize,
            sampler: Posterior::new(),
            reflect,
        })
    }
}
impl Normalized {
    /// Normalize each frame, activate it, pad the result and convolve it.
    ///
    /// The padding is one pixel on each spatial side, mirrored or replicated, and two zero frames in
    /// front, so the convolution's own geometry carries none and no output frame reads a later input
    /// frame.
    /// `pad` is `[1, channels, frames + 2, height + 2, width + 2]` and `mean`/`rstd` are
    /// `[frames, groups]`, all reserved by the encoder's plan.
    ///
    /// **Nothing here allocates.** These three buffers were taken from the arena on every call, which
    /// made the encode ask for roughly another set's worth of bytes per tile that no plan could state:
    /// the first of them alone measured 647,534,592 B at 17 frames of 256x256 (level 0 at 128
    /// channels). A reserved buffer is a buffer the plan can count, and the blocks run one after
    /// another, so one pad per distinct channel count per level is enough.
    fn run(
        &self,
        f: &Forward,
        values: TensorView,
        pad: TensorView,
        mean: TensorView,
        rstd: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let (_, channels, frames, height, width) = extent(values)?;
        if channels != u64::from(self.channels)
            || pad.shape.rank() != 5
            || pad.shape.dims()
                != [
                    1,
                    u64::from(self.channels),
                    frames + 2,
                    height + 2,
                    width + 2,
                ]
        {
            return Err(Error::config(format!(
                "a normalization of {} channels got {:?} to carry its padded input",
                self.channels,
                pad.shape.dims()
            )));
        }
        FrameMoments::new(u64::from(self.groups), self.epsilon).forward(f, values, mean, rstd)?;
        self.norm.forward(f, values, mean, rstd, pad, 1, 1, 2)?;
        Ok(self.conv.forward(f, pad, out)?)
    }
}

/// A `[channels, frames, rows, columns]` raster as the one sample a convolution reads.
fn batched(view: TensorView) -> TensorView {
    let dims = view.shape.dims();
    view.reshape(Shape::new(&[1, dims[0], dims[1], dims[2], dims[3]]))
}

/// The extent of a five-axis tensor: `(batch, channels, frames, height, width)`.
fn extent(view: TensorView) -> Result<(u64, u64, u64, u64, u64)> {
    if view.shape.rank() != 5 {
        return Err(Error::config(format!(
            "a video tensor is {:?}, five axes expected",
            view.shape.dims()
        )));
    }
    Ok((
        view.shape.dim(0),
        view.shape.dim(1),
        view.shape.dim(2),
        view.shape.dim(3),
        view.shape.dim(4),
    ))
}

/// A five-axis tensor as the two-axis view an elementwise operator reads.
fn rows(view: TensorView) -> Result<TensorView> {
    let columns = view.shape.dim(4);
    if columns == 0 {
        return Err(Error::config("a tensor of no column".to_owned()));
    }
    let rows = view.shape.numel() / columns;
    Ok(view.reshape(Shape::new(&[rows, columns])))
}

/// The distinct extents of a run, in the order they are first named.
///
/// A grid names every tile's extent and a uniform grid names the same one 28 times; the reservation is per
/// distinct extent, so this is what keeps a uniform run to one set. The order is first-seen because a
/// plan's sets are reported in reservation order.
fn distinct(extents: &[Extent]) -> Vec<Extent> {
    let mut seen: Vec<Extent> = Vec::with_capacity(extents.len());
    for extent in extents {
        if !seen.contains(extent) {
            seen.push(*extent);
        }
    }
    seen
}

/// One clip of a condition: the frames it reads and the latent frames it writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClipSpan {
    /// The clip's first frame.
    pub first: u32,
    /// Its frames.
    pub frames: u32,
    /// The first latent frame it writes.
    pub latent: u32,
    /// The latent frames it writes.
    pub latents: u32,
}

/// The buffers one tile extent's encode works in, reserved once for a whole run of tiles.
///
/// A run's tiles are not uniform - the right and bottom edges of a grid are shorter - so a scratch holds
/// one set per **distinct** tile extent and each call takes the set its own extent names. A set's buffers
/// are then exactly the call's shape, which matters because a convolution writes the view it is handed:
/// a sub-window of a larger buffer would be strided, and the convolution path takes a destination's
/// address and shape, not its strides.
pub struct EncoderScratch {
    sets: Vec<(Extent, EncoderSet)>,
}

impl EncoderScratch {
    /// The set a tile of this extent encodes in.
    fn of(&self, extent: Extent) -> Result<&EncoderSet> {
        self.sets
            .iter()
            .find(|(known, _)| *known == extent)
            .map(|(_, set)| set)
            .ok_or_else(|| {
                Error::config(format!(
                    "no encoder buffers were reserved for a {} by {} tile of {} frames",
                    extent.height, extent.width, extent.frames
                ))
            })
    }
}

/// One tile extent's buffers: the input pad, the input convolution's output and a set per level.
///
/// The moment convolution writes the caller's `out`, so no buffer is held for it: the tile's result is
/// the moments, and the one-by-one projection that follows runs once, on the stitched raster
/// ([`VideoEncoder::project`]).
struct EncoderSet {
    padded: TensorView,
    stream: TensorView,
    levels: Vec<LevelSet>,
    /// The last normalization's own padded input, mean and inverse standard deviation.
    output_pad: TensorView,
    output_mean: TensorView,
    output_rstd: TensorView,
}

/// One level's buffers: the pair its blocks alternate between, one working buffer the block reads back,
/// its shortcut projection where the first block widens, and the pad and strided output where the level
/// downsamples.
///
/// The pair is the level's **own** width, not the buffer the level was handed. Alternating with the
/// incoming buffer instead reads a level's second block against its input width whenever the level widens
/// it - which three of the six levels do - so a two-block level would write a wider value into a narrower
/// view.
struct LevelSet {
    first: TensorView,
    second: TensorView,
    working: TensorView,
    projected: Option<TensorView>,
    pad: Option<TensorView>,
    strided: Option<TensorView>,
    /// One padded-input buffer per distinct channel count among the level's normalizations, keyed by
    /// that count. A level's first block widens and its later blocks do not, so a level needs two; the
    /// blocks run one after another and never hold two pads at once, so one buffer per count is enough.
    norms: Vec<(u32, TensorView)>,
    /// The statistics of every normalization at this level, `[frames, groups]` each. Every block of a
    /// level runs at the level's own frame count, so one pair serves them all.
    mean: TensorView,
    rstd: TensorView,
}

impl VideoEncoder {
    /// Reserve the buffers one run of tiles encodes in, one set per distinct tile extent.
    ///
    /// The arena never rewinds, so a driver that encodes a grid of tiles must reserve once and hand the
    /// same scratch to every tile: a reservation inside the loop is multiplied by the tile count, and one
    /// tile's buffers are most of a gigabyte at the served canvas.
    pub fn reserve(&self, f: &Forward, extents: &[Extent]) -> Result<EncoderScratch> {
        if extents.is_empty() {
            return Err(Error::config(
                "an encoder run with no tile extent has nothing to reserve".to_owned(),
            ));
        }
        let mut sets: Vec<(Extent, EncoderSet)> = Vec::new();
        for extent in distinct(extents) {
            sets.push((extent, self.set(f, extent)?));
        }
        Ok(EncoderScratch { sets })
    }

    /// The buffers one tile extent needs.
    fn set(&self, f: &Forward, raster: Extent) -> Result<EncoderSet> {
        let extents = self.stack.extents(raster);
        let buffer = |channels: u64, extent: &Extent| -> Result<TensorView> {
            Ok(f.arena().tensor(
                DType::F32,
                Shape::new(&[
                    1,
                    channels,
                    u64::from(extent.frames),
                    u64::from(extent.height),
                    u64::from(extent.width),
                ]),
            )?)
        };
        // One normalization's padded input: one pixel wider and taller on each side and two frames in
        // front, exactly the shape `Normalized::run` checks.
        let pad_buffer = |channels: u64, extent: &Extent| -> Result<TensorView> {
            Ok(f.arena().tensor(
                DType::F32,
                Shape::new(&[
                    1,
                    channels,
                    u64::from(extent.frames) + 2,
                    u64::from(extent.height) + 2,
                    u64::from(extent.width) + 2,
                ]),
            )?)
        };
        let moments_buffer = |groups: u64, extent: &Extent| -> Result<TensorView> {
            Ok(f.arena()
                .tensor(DType::F32, Shape::new(&[u64::from(extent.frames), groups]))?)
        };
        let padded = pad_buffer(u64::from(self.stack.input_channels), &raster)?;
        let stream = buffer(u64::from(self.stack.widths[0]), &extents[0])?;
        let mut levels = Vec::with_capacity(extents.len() - 1);
        for index in 0..extents.len() - 1 {
            let width = u64::from(self.stack.widths[index]);
            let groups = u64::from(
                self.levels[index]
                    .blocks
                    .first()
                    .map(|block| block.first.groups)
                    .unwrap_or(self.output.groups),
            );
            let extent = &extents[index];
            let next = &extents[index + 1];
            let downsample = self.levels[index].downsample.is_some();
            levels.push(LevelSet {
                first: buffer(width, extent)?,
                second: buffer(width, extent)?,
                working: buffer(width, extent)?,
                // The first block of a level widens from the previous level's width; later blocks add to
                // their own width and project nothing.
                projected: match self.levels[index]
                    .blocks
                    .first()
                    .map(|block| block.shortcut.is_some())
                {
                    Some(true) => Some(buffer(width, extent)?),
                    _ => None,
                },
                pad: match downsample {
                    true => Some(f.arena().tensor(
                        DType::F32,
                        Shape::new(&[
                            1,
                            width,
                            u64::from(extent.frames) + 2,
                            u64::from(extent.height) + 1,
                            u64::from(extent.width) + 1,
                        ]),
                    )?),
                    false => None,
                },
                strided: match downsample {
                    true => Some(buffer(width, next)?),
                    false => None,
                },
                norms: {
                    let mut norms: Vec<(u32, TensorView)> = Vec::new();
                    for block in &self.levels[index].blocks {
                        for normalized in [&block.first, &block.second] {
                            if norms.iter().any(|(count, _)| *count == normalized.channels) {
                                continue;
                            }
                            // Level zero's first block normalizes raw pixels, so its pad is the pixel
                            // pad the input convolution has already read: one buffer, not two.
                            let view = match index == 0
                                && normalized.channels == self.stack.input_channels
                            {
                                true => padded,
                                false => pad_buffer(u64::from(normalized.channels), extent)?,
                            };
                            norms.push((normalized.channels, view));
                        }
                    }
                    norms
                },
                mean: moments_buffer(groups, extent)?,
                rstd: moments_buffer(groups, extent)?,
            });
        }
        let last = extents[extents.len() - 1];
        let output_pad = pad_buffer(u64::from(self.output.channels), &last)?;
        let output_groups = u64::from(self.output.groups);
        let output_mean = moments_buffer(output_groups, &last)?;
        let output_rstd = moments_buffer(output_groups, &last)?;

        // The normalizations' own padded inputs: one per distinct channel count, because a level's
        // first block widens and its later blocks do not. Level zero's narrowest is the pixel pad the
        // input convolution already read, so it is the same buffer rather than a second one.
        Ok(EncoderSet {
            padded,
            stream,
            levels,
            output_pad,
            output_mean,
            output_rstd,
        })
    }

    /// Encode one tile into a reserved set of buffers.
    ///
    /// This is the entry point a driver of several tiles uses: it reserves once with [`Self::reserve`]
    /// and hands the same scratch to every tile, which is the difference between one tile's buffers and
    /// one tile's buffers times the grid. `pixels` is `[1, in_channels, frames, height, width]` FP32 -
    /// one tile of a clip, normalized by [`normalize`] - and `out` is
    /// `[1, 2 · latent_channels, frames', height / 16, width / 16]`. A tile whose extent was not reserved
    /// is refused by name rather than read past the buffers.
    pub fn encode_tile(
        &self,
        f: &Forward,
        scratch: &EncoderScratch,
        pixels: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let (batch, channels, frames, height, width) = extent(pixels)?;
        if batch != 1 {
            return Err(Error::config(format!(
                "the video encoder reads one tile at a time, not {batch}"
            )));
        }
        if channels != u64::from(self.stack.input_channels) {
            return Err(Error::config(format!(
                "{channels} input channels for a stack of {}",
                self.stack.input_channels
            )));
        }
        if frames == 0 || frames > u64::from(self.stack.clip.frames) {
            return Err(Error::config(format!(
                "a video encoder reads one still or one clip of at most {} frames, not {frames}",
                self.stack.clip.frames
            )));
        }
        let raster = Extent {
            frames: u32::try_from(frames)
                .map_err(|_| Error::config("too many frames".to_owned()))?,
            height: u32::try_from(height).map_err(|_| Error::config("too tall".to_owned()))?,
            width: u32::try_from(width).map_err(|_| Error::config("too wide".to_owned()))?,
        };
        let set = scratch.of(raster)?;

        // The input convolution reads padded pixels; it has no normalization of its own.
        FramePad::new(self.reflect).forward(f, pixels, set.padded, 1, 1, 2)?;
        self.input.forward(f, set.padded, set.stream)?;

        // A level's blocks alternate between its own pair of buffers, and the level's input is a third
        // buffer it never writes: the first block reads the previous level's output, each later block
        // reads the one before it, and every block writes the level's width.
        let mut source = set.stream;
        for (index, level) in self.levels.iter().enumerate() {
            let buffers = &set.levels[index];
            for (at, block) in level.blocks.iter().enumerate() {
                let target = match at % 2 {
                    0 => buffers.first,
                    _ => buffers.second,
                };
                self.block(f, block, source, target, buffers)?;
                source = target;
            }
            if let Some(downsample) = &level.downsample {
                let (Some(pad), Some(strided)) = (buffers.pad, buffers.strided) else {
                    return Err(Error::config(format!(
                        "level {index} downsamples without the buffers for it"
                    )));
                };
                FramePad::new(self.reflect).forward(f, source, pad, 0, 0, 2)?;
                downsample.forward(f, pad, strided)?;
                source = strided;
            }
        }

        // The moment convolution writes `2 · latent_channels` into the caller's view. The one-by-one
        // projection that follows it runs **once for the whole raster**, after the tiles are stitched
        // (`project`), because that is the reference's order: the tiles produce moments, the tiler
        // blends them, and the projection reads the stitched result.
        self.output.run(
            f,
            source,
            set.output_pad,
            set.output_mean,
            set.output_rstd,
            out,
        )
    }

    /// The latent frames a **condition** of `frames` frames encodes to.
    ///
    /// A condition is cut into clips of `Stack::clip_frames` frames, each yielding
    /// `Stack::clip_latents` latent frames, and the last clip's trailing `Stack::token_drop` latent
    /// frames are dropped - so 17 frames are 2 latent frames and 22 are 7. A single frame is a still and
    /// yields one (the network's causal front padding makes one frame one latent frame).
    ///
    /// This is a **different statement** from the generated timeline's `packing::video_latent_frames`
    /// (`17n + 5` frames to `5n + 2` latents). The two agree on the counts a generation uses and this
    /// one alone accepts the counts a condition may be - 17, 30, any length - because a condition is
    /// cut and padded, not generated.
    pub fn latent_frames(&self, frames: u32) -> Result<u32> {
        self.stack.clip.latent_frames(frames)
    }

    /// The clips a **condition** of `frames` frames is cut into, in order ([`Clip::spans`]).
    pub fn clips(&self, frames: u32) -> Result<Vec<ClipSpan>> {
        self.stack.clip.spans(frames)
    }

    /// Project the stitched moments to the posterior's two fields: one call, after the stitch.
    ///
    /// The reference's `Encoder` wraps the tiled network and runs `quant_conv` on the stitched
    /// moments, so a tile's own output is not yet projected. A one-by-one convolution commutes with the
    /// blend in real arithmetic - the blend's weights do not depend on the channel - so projecting per
    /// tile would be the same value for more work; it is written this way because it is also the order
    /// the posterior is sampled in, which does depend on it: the sample reads a mean and a log-variance
    /// of the same position, and a per-tile projection would entangle two tiles' roundings before they
    /// were blended.
    ///
    /// `moments` is a tile's or the whole run's `[2 · latent_channels, frames, rows, columns]`; `out` is
    /// the same shape. Nothing is reserved: the projection reads and writes the caller's views.
    pub fn project(&self, f: &Forward, moments: TensorView, out: TensorView) -> Result<()> {
        self.moments
            .forward(f, batched(moments), batched(out))
            .map_err(Error::from)
    }

    /// The posterior sample a condition reads: `moments` and the runner's `noise` to `out`.
    ///
    /// The reference's `DiagonalGaussian` and the tail of `LatentEncoder`: the channel axis splits into
    /// mean and log-variance, the log-variance is clamped to `(-30, 20)`, scaled by `exp(0.5 · ·)`,
    /// mixed with the noise and added to the mean; the sum is rounded through FP16 **as a step**; and
    /// the per-channel affine of [`normalize_tables`] turns it into the latent the conditioner reads.
    /// The epilogue is `ops::posterior`, exact by contract, so the only rounding that differs from
    /// torch is the affine's fused multiply-add ([`normalize_tables`] declares it).
    ///
    /// `moments` is the projected moments of the whole stitched raster and `out` is the condition's
    /// latent, both `[channels, frames, rows, columns]` FP32; `noise` is the runner's draw in the same
    /// order - the reference's `condition_noise_shapes` is channel-major and the draw is addressed by
    /// element index, so that order is what reproduces it. The operator addresses every operand
    /// through two strides, so a channel-major raster needs **no copy and no transpose**: the moments'
    /// two halves are one buffer cut by an offset, the sample is written into one scratch raster (the
    /// affine cannot read and write `out` at once), and the affine reads that raster into `out`.
    pub fn posterior(
        &self,
        f: &Forward,
        moments: TensorView,
        noise: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let sample = Sample::of(self.stack.latent_channels, moments, noise, out)?;
        let positions = sample.positions;
        let channels = sample.channels;
        // A channel-major raster: neighbouring positions are neighbouring elements, and one channel is
        // one whole plane away. The two halves of the moments differ by exactly `channels` planes.
        let strides = Strides::new(&[1, positions]);
        let view = |ptr| {
            TensorView::with_strides(ptr, DType::F32, Shape::new(&[positions, channels]), strides)
        };
        let drawn = f.arena().tensor(DType::F32, out.shape)?;
        self.sampler.forward(
            f,
            view(moments.ptr),
            view(moments.offset_elements(channels * positions)),
            view(noise.ptr),
            view(drawn.ptr),
        )?;
        self.normalize
            .forward(f, batched(drawn), batched(out))
            .map_err(Error::from)
    }

    /// One residual block: two normalized convolutions, added to the input or its projection.
    ///
    /// The level's own buffers come in as one argument because they are one thing: `set` holds, for a
    /// level, the pair of alternating outputs, the scratch its first convolution writes, the projection
    /// a widening block needs, and the normalization state (`norms`, `mean`, `rstd`) its convolutions
    /// share. Passing them one by one made a ten-argument call whose argument list was the struct's
    /// field list - `clippy::too_many_arguments` was reporting a fact about the signature rather than a
    /// style preference.
    fn block(
        &self,
        f: &Forward,
        block: &Block,
        values: TensorView,
        out: TensorView,
        set: &LevelSet,
    ) -> Result<()> {
        let pad_of = |normalized: &Normalized| -> Result<TensorView> {
            set.norms
                .iter()
                .find(|(count, _)| *count == normalized.channels)
                .map(|(_, view)| *view)
                .ok_or_else(|| {
                    Error::config(format!(
                        "no reserved pad for a normalization of {} channels",
                        normalized.channels
                    ))
                })
        };
        block.first.run(
            f,
            values,
            pad_of(&block.first)?,
            set.mean,
            set.rstd,
            set.working,
        )?;
        block.second.run(
            f,
            set.working,
            pad_of(&block.second)?,
            set.mean,
            set.rstd,
            out,
        )?;
        let residual = match (&block.shortcut, set.projected) {
            (Some(shortcut), Some(projected)) => {
                shortcut.forward(f, values, projected)?;
                projected
            }
            (Some(_), None) => {
                return Err(Error::config(
                    "a block that widens its input has no buffer to project it into".to_owned(),
                ));
            }
            (None, _) => values,
        };
        // The reference groups the sum as `residual + (values + bias)` and adds the convolution's
        // bias in a pass of its own; cuDNN adds that bias in a pass of its own too, and IEEE addition
        // commutes, so `(values + bias) + residual` here is the same rounded value.
        let sum = rows(out)?;
        Ok(Residual::default().forward(f, sum, rows(residual)?, None, sum)?)
    }
}

/// The longest track the model supports, in latent frames.
///
/// A request is at most [`crate::config::MAX_FRAMES`] frames of video at [`crate::config::FPS`], and its
/// audio lasts as long, so the longest track is that duration at [`crate::config::AUDIO_LATENTS_PER_SECOND`]
/// latents, rounded up to the next latent. The encoder's attention is built for this bound and a longer
/// track is refused rather than read past it.
pub const LONGEST_TRACK: u32 = (crate::config::MAX_FRAMES
    * crate::config::AUDIO_LATENTS_PER_SECOND)
    .div_ceil(crate::config::FPS);

/// The DAC-style audio encoder `ref2va` conditions on: a waveform's posterior mean as latent frames.
///
/// `[1, 1, samples]` mono audio in, `[latent_channels, frames]` latents out, with `frames =
/// samples / 800` — the product of the checkpoint's encoder rates `(2, 4, 4, 5, 5)` and the hop the
/// decoder inverts. The network is the reference's `Encoder`: an input convolution of width
/// `encoder_dim`, five stages that each run three dilated residual units at the incoming width and then
/// a strided widening convolution, a Snake and a convolution to `latent_dim`, and then the attention
/// projection that every frame of the track attends over its predecessors.
///
/// **The projection's attention is causal** (`DenseInput(causal=True)` in the reference), which is what
/// makes a latent frame depend only on the audio before it. It runs in FP32 like everything else in this
/// encoder, on the deployment's longest track, which `load` takes as the attention's key bound.
pub struct AudioEncoder {
    config: crate::AudioVaeConfig,
    /// The most latent frames one track may have: the attention's key bound.
    frames: u32,
    input: Conv1d,
    stages: Vec<EncoderStage>,
    tail: Snake,
    tail_alpha: Weight,
    output: Conv1d,
    projection: Projection,
    mean: Conv1d,
    /// The latent space's affine, `(latent - mean) / std` per channel. The reference's `LatentEncoder`
    /// applies it to the posterior mode, and the DiT's audio input reads normalized rows: the
    /// statistics are the same ones the audio decoder restores.
    normalize: Conv1d,
    transpose: Transpose,
}

/// One encoder stage: three dilated residual units at `inner`, then a widening strided convolution.
struct EncoderStage {
    /// The width the units and the Snake run at: half the stage's output width.
    inner: u32,
    /// Latent samples divide by this.
    rate: u32,
    units: Vec<EncoderUnit>,
    snake: Snake,
    alpha: Weight,
    widen: Conv1d,
}

/// One residual unit: snake, dilated convolution, snake, one-by-one-wide convolution, added back.
struct EncoderUnit {
    convs: [Conv1d; 2],
    snakes: [Snake; 2],
    alphas: [Weight; 2],
}

/// The attention projection that narrows latent features to the latent channels.
struct Projection {
    shortcut_norm: LayerNorm,
    shortcut: Linear,
    attention_norm: LayerNorm,
    /// Three matrices over the checkpoint's fused query-key-value weight, so each head group is one
    /// contiguous block the attention can read.
    query: Linear,
    key: Linear,
    value: Linear,
    attention: Attention,
    attention_output: Linear,
    /// The head average and the head width pooled down to the latent channels, in one pass.
    pool: AdaptivePool,
    mlp_norm: LayerNorm,
    mlp: GeGlu,
}

/// A normalized tanh-GELU gated linear unit: the reference's `GeGLU`.
struct GeGlu {
    norm: LayerNorm,
    gate: Linear,
    value: Linear,
    output: Linear,
    /// `gelu_tanh(gate) · value`, the reference's `F.gelu(gate, approximate="tanh") * value`.
    product: GatedActivation,
}

impl AudioEncoder {
    /// The latent frames `samples` samples encode to: `ceil(samples / hop)`.
    ///
    /// This is the reference's rule and the reason it is a ceiling is the pad its
    /// The batch the track must carry: **the stereo pair is the batch**, and a mono track is refused.
    ///
    /// The reference expands a mono track to two identical channels before the encoder
    /// (`unpack_samples`: `waveform.expand(2, -1)`, refusing a channel count other than one or two),
    /// and the encoder's own output is `2 * latent_frames` rows. A caller that hands a mono track
    /// unexpanded gets **half a track of latent frames** with every count downstream still
    /// consistent - `[1, 1, samples]` is a legal shape on its own - so the check is that the batch is
    /// exactly the stereo pair, and it names the expansion the reference performs.
    fn check_track(shape: Shape) -> Result<u64> {
        if shape.rank() != 3 || shape.dim(1) != 1 {
            return Err(Error::config(format!(
                "the audio encoder reads a [batch, 1, samples] track, not {:?}",
                shape.dims()
            )));
        }
        let batch = shape.dim(0);
        if batch != u64::from(crate::config::AUDIO_CHANNELS) {
            return Err(Error::config(format!(
                "the audio encoder reads the stereo pair ({} batch rows) and was given {batch}: a \
                 mono track must be expanded to two identical channels before it, as the \
                 reference's `unpack_samples` does - read as one row it encodes half a track of \
                 latent frames, which every count downstream accepts",
                crate::config::AUDIO_CHANNELS
            )));
        }
        Ok(batch)
    }

    /// own `unpack_samples` applies first: a track that is not a whole number of latent frames is
    /// **right-padded with zeros** to `ceil(samples / hop) · hop`, so every stage's strided convolution
    /// divides exactly and the last latent frame reads those zeros. A track of any positive length
    /// therefore encodes, and [`Self::encode`] states what it needs to do the same.
    pub fn frames(&self, samples: u64) -> Result<u64> {
        let frames = audio_latent_frames(samples, self.hop())?;
        if frames > u64::from(LONGEST_TRACK) {
            return Err(Error::config(format!(
                "{frames} latent frames exceed the {LONGEST_TRACK}-frame track the encoder's \
                 attention is built for"
            )));
        }
        Ok(frames)
    }

    /// The sample count each stage's residual units run at, from the track's padded length.
    ///
    /// The units of stage `k` add over the extent that stage receives - the track's length halved
    /// and fifthed once per rate, `[2, 4, 4, 5, 5]` here - so a track of `L` samples launches the
    /// residual add at `L`, `L/2`, `L/8`, `L/32` and `L/160`, five programs and not one.
    /// [`Self::encode`] walks this list.
    ///
    /// The length is the **padded** one, because the pad is applied before the first convolution
    /// (see [`Self::encode`]) - a track that is not a whole number of latent frames never reaches a
    /// stage, so it states no row.
    pub fn stage_extents(config: &crate::AudioVaeConfig, samples: u64) -> Result<Vec<u64>> {
        let hop = config
            .encoder_rates
            .iter()
            .map(|rate| u64::from(*rate))
            .product::<u64>()
            .max(1);
        if !samples.is_multiple_of(hop) {
            let padded = padded_samples(samples, hop)?;
            return Err(Error::config(format!(
                "{samples} samples is not a whole number of {hop}-sample latent frames: the stages \
                 run the padded track ({padded} samples), so an unpadded length states no program"
            )));
        }
        let mut extent = samples;
        let mut extents = Vec::with_capacity(config.encoder_rates.len());
        for rate in &config.encoder_rates {
            extents.push(extent);
            extent /= u64::from(*rate).max(1);
        }
        Ok(extents)
    }

    /// The samples one latent frame covers: the product of the encoder's rates.
    pub fn hop(&self) -> u64 {
        self.stages
            .iter()
            .map(|stage| u64::from(stage.rate))
            .product()
    }

    /// The encoder the checkpoint's `encoder`, `pre_block` and `mean_proj` modules state, read in
    /// FP32.
    ///
    /// The encoder is DAC's network - `encoder.block` is the input convolution, one entry per stage
    /// (three dilated residual units, the stage's Snake and the strided widening convolution), a
    /// Snake and the output convolution - followed by the latent projection (`pre_block`) and the
    /// mean convolution (`mean_proj`). Every convolution of the network is weight-normalized and
    /// folded once here. The attention is built for [`LONGEST_TRACK`], the longest track the model
    /// supports; a longer track is refused by [`Self::frames`] rather than read past the bound.
    pub fn load(config: &crate::AudioVaeConfig, loader: &mut WeightLoader) -> Result<Self> {
        let frames = LONGEST_TRACK;
        // The audio encoder computes in FP32 throughout, so every weight is read in it.
        let mut root = loader.scope("");
        let mut root = root.pinned(DType::F32);
        let mut network = root.scope("encoder.block");
        let mut width = config.encoder_dim;
        let (weight, bias) = folded(&mut network.scope("0"), Some(u64::from(width)))?;
        let input = Conv1d::new(weight, bias, crate::audio_vae::geometry(7, 1, 3)?, false)?;

        let mut stages = Vec::with_capacity(config.encoder_rates.len());
        for (index, rate) in config.encoder_rates.iter().enumerate() {
            let inner = width;
            width = width.checked_mul(2).ok_or_else(|| {
                Error::config("the encoder's widths overflow a descriptor".to_owned())
            })?;
            // `EncoderBlock.block` is five entries: three residual units at 0, 1 and 2, the stage's
            // Snake at 3, and the widening convolution at 4.
            let mut stage = network.scope(format!("{}.block", index + 1));
            let mut units = Vec::with_capacity(3);
            // The three units are the same shape at dilations 1, 3 and 9: the first convolution is
            // seven wide and dilated, its padding keeping the length, and the second is the reference's
            // `Conv1d(channels, channels, 1)`, a pointwise mix.
            for (unit, dilation) in [1u32, 3, 9].into_iter().enumerate() {
                // A residual unit's `block` is its Snake (0), the dilated convolution (1), a second
                // Snake (2) and the pointwise convolution (3).
                let mut module = stage.scope(format!("{unit}.block"));
                let (weight, bias) = folded(&mut module.scope("1"), Some(u64::from(inner)))?;
                let first = Conv1d::new(
                    weight,
                    bias,
                    crate::audio_vae::dilated_geometry(7, 1, 3 * dilation, dilation)?,
                    false,
                )?;
                let (weight, bias) = folded(&mut module.scope("3"), Some(u64::from(inner)))?;
                let second =
                    Conv1d::new(weight, bias, crate::audio_vae::geometry(1, 1, 0)?, false)?;
                units.push(EncoderUnit {
                    convs: [first, second],
                    snakes: [Snake::new(u64::from(inner))?, Snake::new(u64::from(inner))?],
                    alphas: [
                        module.scope("0").tensor("alpha", &[u64::from(inner)])?,
                        module.scope("2").tensor("alpha", &[u64::from(inner)])?,
                    ],
                });
            }
            let snake = Snake::new(u64::from(inner))?;
            let alpha = stage.scope("3").tensor("alpha", &[u64::from(inner)])?;
            let (weight, bias) = folded(&mut stage.scope("4"), Some(u64::from(width)))?;
            let widen = Conv1d::new(
                weight,
                bias,
                crate::audio_vae::geometry(2 * rate, *rate, rate.div_ceil(2))?,
                false,
            )?;
            stages.push(EncoderStage {
                inner,
                rate: *rate,
                units,
                snake,
                alpha,
                widen,
            });
        }

        let tail = Snake::new(u64::from(width))?;
        // The network ends with a Snake and the output convolution.
        let stages_end = config.encoder_rates.len();
        let tail_alpha = network
            .scope(stages_end + 1)
            .tensor("alpha", &[u64::from(width)])?;
        // The network's last convolution is three wide, not seven: the reference's `Conv1d(width,
        // latent_dim, 3, padding=1)`.
        let (weight, bias) = folded(
            &mut network.scope(stages_end + 2),
            Some(u64::from(config.latent_dim)),
        )?;
        let output = Conv1d::new(weight, bias, crate::audio_vae::geometry(3, 1, 1)?, false)?;

        let projection = Projection::load(config, &mut root.scope("pre_block"), frames)?;
        let channels = u64::from(config.latent_channels);
        // The reference's mean projection is `nn.Conv1d(latent_channels, latent_channels, 1)`, so the
        // checkpoint stores a three-dimensional filter `[out, in, kernel]` and not a matrix: a kernel
        // of one makes the two the same arithmetic but not the same rank, and `Conv1d::new` refuses a
        // filter that is not 3-d.
        let mut mean = root.scope("mean_proj");
        let mean = Conv1d::new(
            mean.tensor("weight", &[channels, channels, 1])?,
            Some(mean.tensor("bias", &[channels])?),
            crate::audio_vae::geometry(1, 1, 0)?,
            false,
        )?;
        // The latent space's affine: the config states its statistics, not a tensor for them.
        let (scale, shift) = normalize_tables(&config.latents_mean, &config.latents_std)?;
        let mut constants = loader.scope("audio_vae.normalize");
        let normalize = Conv1d::new(
            constants.constant_f32("weight", &[channels, channels, 1], &scale)?,
            Some(constants.constant_f32("bias", &[channels], &shift)?),
            crate::audio_vae::geometry(1, 1, 0)?,
            false,
        )?;
        Ok(Self {
            config: config.clone(),
            frames,
            input,
            stages,
            tail,
            tail_alpha,
            output,
            projection,
            mean,
            normalize,
            transpose: Transpose::new(DType::F32)?,
        })
    }

    /// Encode a `[batch, 1, length]` track into `out [batch · frames, latent_channels]` F32 rows of
    /// the normalized latent, `(mode - mean) / std`.
    ///
    /// The encoder is mono per batch row (it encodes `[batch, 1, samples]` waveforms) and its
    /// latent is a **two-row batch**: a stereo reference arrives as `[2, 1, samples]` and its
    /// latent is `[2, latent_channels, frames]`, the same two-row shape the audio decoder reads.
    /// The rows are written channel-major, `[batch · frames, latent_channels]` with batch
    /// outermost, because that is the form the DiT's audio conditioning reads (`[2 · frames,
    /// audio_channels]`) - so no caller transposes.
    ///
    /// The length must be a whole number of hops and no longer than the deployment's bound. The
    /// activation of every stage runs at the stage's inner width, which is half of the width the
    /// widening convolution writes. The projection attends over frame 2, so it runs once per batch row
    /// and never mixes them.
    pub fn encode(&self, f: &Forward, samples: TensorView, out: TensorView) -> Result<()> {
        let batch = Self::check_track(samples.shape)?;
        let length = samples.shape.dim(2);
        let frames = self.frames(length)?;
        // The network's stages divide the length exactly, so a track that is not a whole number of
        // latent frames must be right-padded with zeros to one - what the reference's `unpack_samples`
        // does before it convolves. **The pad is the caller's**, and that is not a punt: the padded
        // span depends on the request's own sample count, so it is a request buffer, which the runner
        // allocates and may fill on the host before it uploads. `padded_samples` states the length and
        // this refusal states the length too, so a caller that meets it needs no operator: truncating
        // instead would drop the last latent frame, which is why this is a refusal and not a rounding.
        let hop = self.hop();
        if !length.is_multiple_of(hop) {
            let padded = padded_samples(length, hop)?;
            return Err(Error::config(format!(
                "{length} samples is not a whole number of {hop}-sample latent frames: hand this call \
                 the track right-padded with zeros to {padded} samples (`padded_samples`), as the \
                 reference's `unpack_samples` does before it convolves - the pad is a request \
                 buffer's tail, so it is the caller's to fill"
            )));
        }
        if frames > u64::from(self.frames) {
            return Err(Error::config(format!(
                "a {frames}-frame track is longer than the {} the deployment declared",
                self.frames
            )));
        }
        let channels = u64::from(self.config.latent_channels);
        let rows = batch
            .checked_mul(frames)
            .ok_or_else(|| Error::config("the audio latent's row count overflows".to_owned()))?;
        if out.shape.rank() != 2 || out.shape.dims() != [rows, channels] {
            return Err(Error::config(format!(
                "the audio encoder writes [{rows}, {channels}] channel-major rows, not {:?}",
                out.shape.dims()
            )));
        }

        let width = u64::from(self.config.encoder_dim);
        let mut hidden = f
            .arena()
            .tensor(DType::F32, Shape::new(&[batch, width, length]))?;
        self.input.forward(f, samples, hidden)?;

        // A stage runs its three units at the stage's inner width, which is the width it receives, then
        // snaked and widened into the next width. Each unit needs two buffers - the two its convolutions
        // write, because the second snake reads what the first convolution wrote - and they are the
        // stage's shape, so the stage reserves them once and the three units take turns: a unit writes
        // `first` from the residual stream before reading it, and `second` from its own convolution. The
        // residual stream itself is `hidden`, which no unit writes.
        // The extents this track launches at, from the one expression the statement also reads.
        let extents = Self::stage_extents(&self.config, length)?;
        for (index, stage) in self.stages.iter().enumerate() {
            let extent = extents.get(index).copied().unwrap_or(frames);
            let inner = u64::from(stage.inner);
            let inner_shape = Shape::new(&[batch, inner, extent]);
            let first = f.arena().tensor(DType::F32, inner_shape)?;
            let second = f.arena().tensor(DType::F32, inner_shape)?;
            for unit in &stage.units {
                unit.snakes[0].forward(f, hidden, unit.alphas[0].view(), first)?;
                unit.convs[0].forward(f, first, second)?;
                unit.snakes[1].forward(f, second, unit.alphas[1].view(), first)?;
                unit.convs[1].forward(f, first, second)?;
                let residual = matrix(hidden)?;
                Residual::default().forward(f, residual, matrix(second)?, None, residual)?;
            }
            let widened = extent / u64::from(stage.rate);
            let next = f.arena().tensor(
                DType::F32,
                Shape::new(&[batch, u64::from(self.width_of(stage)), widened]),
            )?;
            let scratch = f.arena().tensor(DType::F32, inner_shape)?;
            stage
                .snake
                .forward(f, hidden, stage.alpha.view(), scratch)?;
            stage.widen.forward(f, scratch, next)?;
            hidden = next;
        }

        // The tail runs at the latent frame count, which is the last stage's own `extent` divided by
        // its rate - `frames`, from the same expression that refused an unpadded track.
        let extent = frames;

        // The tail Snake and the convolution to the latent width.
        let wide = u64::from(self.config.latent_dim);
        let tail_shape = Shape::new(&[batch, wide, extent]);
        let tail_scratch = f.arena().tensor(DType::F32, tail_shape)?;
        self.tail
            .forward(f, hidden, self.tail_alpha.view(), tail_scratch)?;
        let features = f.arena().tensor(DType::F32, tail_shape)?;
        self.output.forward(f, tail_scratch, features)?;

        // The projection attends over the frame axis, so it runs once per batch row and never mixes
        // rows: a row's features are a `[wide, extent]` matrix, transposed to frame-major for the
        // attention, and the mean head's `[channels, extent]` is transposed again into the row-major
        // channel-major block the conditioning reads.
        let per_row = wide * extent;
        for row in 0..batch {
            let features = TensorView::new(
                features.ptr.offset(row * per_row * 4),
                DType::F32,
                Shape::new(&[wide, extent]),
            );
            let frames_view = f.arena().tensor(DType::F32, Shape::new(&[extent, wide]))?;
            self.transpose.forward(f, features, frames_view)?;
            let pooled = self.projection.forward(f, frames_view, extent)?;
            let pooled_matrix = f
                .arena()
                .tensor(DType::F32, Shape::new(&[channels, extent]))?;
            self.transpose.forward(f, pooled, pooled_matrix)?;
            let batched = TensorView::new(
                pooled_matrix.ptr,
                DType::F32,
                Shape::new(&[1, channels, extent]),
            );
            let narrowed = f
                .arena()
                .tensor(DType::F32, Shape::new(&[1, channels, extent]))?;
            self.mean.forward(f, batched, narrowed)?;
            let normalized = f
                .arena()
                .tensor(DType::F32, Shape::new(&[1, channels, extent]))?;
            self.normalize.forward(f, narrowed, normalized)?;
            // One row's latent, channel-major rows: `[extent, channels]` at the row's place in `out`.
            let plane =
                TensorView::new(normalized.ptr, DType::F32, Shape::new(&[channels, extent]));
            let written = TensorView::new(
                out.ptr.offset(row * extent * channels * 4),
                DType::F32,
                Shape::new(&[extent, channels]),
            );
            self.transpose.forward(f, plane, written)?;
        }
        Ok(())
    }

    /// The width a stage writes: its inner width doubled.
    fn width_of(&self, stage: &EncoderStage) -> u32 {
        stage.inner * 2
    }
}

/// A `[batch, rows, columns]` view as `[batch · rows, columns]`: the row axis the operators that work
/// a row at a time read.
///
/// Valid for a contiguous view, which every arena tensor and every slice of one is: the batch axis is
/// the outermost, so folding it into the row axis keeps every row where it is.
fn matrix(view: TensorView) -> Result<TensorView> {
    if view.shape.rank() != 3 || view.shape.dim(0) == 0 {
        return Err(Error::config(format!(
            "{:?} is not a batched matrix",
            view.shape.dims()
        )));
    }
    let rows = view.shape.dim(0) * view.shape.dim(1);
    Ok(TensorView::new(
        view.ptr,
        view.dtype,
        Shape::new(&[rows, view.shape.dim(2)]),
    ))
}

impl Projection {
    /// The latent projection module at `scope` (`pre_block`), at the latent width.
    fn load(
        config: &crate::AudioVaeConfig,
        scope: &mut Scope<'_, '_>,
        frames: u32,
    ) -> Result<Self> {
        let wide = u64::from(config.latent_dim);
        let channels = u64::from(config.latent_channels);
        let heads = u64::from(config.num_attention_heads);
        if heads == 0 || !wide.is_multiple_of(heads) {
            return Err(Error::config(format!(
                "{wide} latent channels over {heads} attention heads"
            )));
        }
        let shortcut_norm = LayerNorm::load(&mut scope.scope("norm3"), wide, NORM_EPS)?;
        let attention_norm = LayerNorm::load(&mut scope.scope("norm1"), wide, NORM_EPS)?;
        let mut shortcut = scope.scope("proj");
        let shortcut = Linear::new(
            shortcut.tensor("weight", &[channels, wide])?,
            Some(shortcut.tensor("bias", &[channels])?),
            WeightOrder::NxK,
        )?;
        // The checkpoint fuses query, key and value into one `[3 · wide, wide]` matrix whose thirds
        // are contiguous rows, and keeps the three bias windows as three tensors: each head group is
        // its own layer over its rows. `zero_k_bias` is the checkpoint's stored key window - a
        // vector it carries like the other two - so it is loaded, never substituted by a constant
        // on the strength of its name: a buffer placed in a learned slot changes the arithmetic.
        let third = wide;
        let mut attention = scope.scope("attn");
        let mut linear = |index: u64, bias: &str| -> Result<Linear> {
            let weight = attention
                .scope("qkv")
                .rows("weight", index * third, &[third, wide])?;
            let bias = attention.tensor(bias, &[third])?;
            Ok(Linear::new(weight, Some(bias), WeightOrder::NxK)?)
        };
        let query = linear(0, "q_bias")?;
        let key = linear(1, "zero_k_bias")?;
        let value = linear(2, "v_bias")?;
        let head_dim = u32::try_from(wide / heads)
            .map_err(|_| Error::config("a head wider than a launch holds".to_owned()))?;
        let heads_u32 = u32::try_from(heads)
            .map_err(|_| Error::config("more heads than a launch holds".to_owned()))?;
        let attention = Attention::dense(
            head_dim,
            heads_u32,
            heads_u32,
            frames,
            true,
            None,
            1.0 / (head_dim as f32).sqrt(),
        )?;
        let mut output = scope.scope("attn.proj");
        let attention_output = Linear::new(
            output.tensor("weight", &[channels, channels])?,
            Some(output.tensor("bias", &[channels])?),
            WeightOrder::NxK,
        )?;
        let pool = AdaptivePool::new(heads, wide / heads, channels)?;
        let mlp_norm = LayerNorm::load(&mut scope.scope("norm2"), channels, NORM_EPS)?;
        let mlp = GeGlu::load(config, &mut scope.scope("mlp"))?;
        Ok(Self {
            shortcut_norm,
            shortcut,
            attention_norm,
            query,
            key,
            value,
            attention,
            attention_output,
            pool,
            mlp_norm,
            mlp,
        })
    }

    /// Project `values [frames, latent_dim]` to `[frames, latent_channels]`.
    fn forward(&self, f: &Forward, values: TensorView, frames: u64) -> Result<TensorView> {
        let shape = values.shape;
        let channels = u64::from(self.pool.output());
        let heads = shape.dim(1) / u64::from(self.attention.head_dim());
        let shortcut = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, channels]))?;
        let normed = f.arena().tensor(DType::F32, shape)?;
        self.shortcut_norm.forward(f, values, None, normed)?;
        self.shortcut.forward(f, normed, shortcut)?;
        self.attention_norm.forward(f, values, None, normed)?;
        // Three contiguous head-group blocks, then causal attention over the frames.
        let head_shape = Shape::new(&[frames, heads, u64::from(self.attention.head_dim())]);
        let attended = f.arena().tensor(DType::F32, head_shape)?;
        let mut head_views = Vec::with_capacity(3);
        for linear in [&self.query, &self.key, &self.value] {
            let out = f.arena().tensor(DType::F32, shape)?;
            linear.forward(f, normed, out)?;
            head_views.push(TensorView::new(out.ptr, DType::F32, head_shape));
        }
        self.attention.attend_dense(
            f,
            head_views[0],
            head_views[1],
            head_views[2],
            None,
            None,
            u32::try_from(frames)
                .map_err(|_| Error::config("a track longer than a launch bound".to_owned()))?,
            attended,
        )?;
        let pooled = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, channels]))?;
        self.pool.forward(f, attended, pooled)?;
        let projected = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, channels]))?;
        self.attention_output.forward(f, pooled, projected)?;
        Residual::default().forward(f, shortcut, projected, None, shortcut)?;
        // The normalized GeGLU residual.
        let normed = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, channels]))?;
        self.mlp_norm.forward(f, shortcut, None, normed)?;
        let updated = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, channels]))?;
        self.mlp.forward(f, normed, updated)?;
        Residual::default().forward(f, shortcut, updated, None, shortcut)?;
        Ok(shortcut)
    }
}

impl GeGlu {
    /// The normalized tanh-GELU gated linear unit at `scope` over the latent channels: its `norm`,
    /// the gate `w0`, the value `w1` and the output `w2`.
    fn load(config: &crate::AudioVaeConfig, scope: &mut Scope<'_, '_>) -> Result<Self> {
        let channels = u64::from(config.latent_channels);
        let hidden = 2 * channels;
        let norm = LayerNorm::load(&mut scope.scope("norm"), channels, NORM_EPS)?;
        let mut linear = |module: &str, out: u64, input: u64| -> Result<Linear> {
            let mut module = scope.scope(module);
            Ok(Linear::new(
                module.tensor("weight", &[out, input])?,
                Some(module.tensor("bias", &[out])?),
                WeightOrder::NxK,
            )?)
        };
        let gate = linear("w0", hidden, channels)?;
        let value = linear("w1", hidden, channels)?;
        let output = linear("w2", channels, hidden)?;
        Ok(Self {
            norm,
            gate,
            value,
            output,
            product: GatedActivation::new(ops::activation::Activation::GeluTanh),
        })
    }

    /// `output(gelu_tanh(gate(norm(x))) · value(norm(x)))` into `out`.
    fn forward(&self, f: &Forward, x: TensorView, out: TensorView) -> Result<()> {
        let (frames, hidden) = (x.shape.dim(0), 2 * x.shape.dim(1));
        let normed = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, x.shape.dim(1)]))?;
        self.norm.forward(f, x, None, normed)?;
        let gate = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, hidden]))?;
        let value = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, hidden]))?;
        self.gate.forward(f, normed, gate)?;
        self.value.forward(f, normed, value)?;
        let product = f
            .arena()
            .tensor(DType::F32, Shape::new(&[frames, hidden]))?;
        self.product.forward(f, gate, value, product)?;
        Ok(self.output.forward(f, product, out)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nn::capability::MediaFormat as _;
    use ops::values::bf16_to_f32;

    /// The format of a tower of `patch`-pixel patches, two-frame tubelets and `2 x 2` merges.
    fn format(patch: u32) -> ConditionFormat {
        ConditionFormat {
            vision: crate::config::VisionConfig {
                depth: 1,
                hidden_size: 8,
                intermediate_size: 8,
                num_heads: 1,
                hidden_act: ops::activation::Activation::GeluTanh,
                in_channels: 3,
                patch_size: patch,
                temporal_patch_size: 2,
                spatial_merge_size: 2,
                out_hidden_size: 8,
                num_position_embeddings: 16,
                deepstack_visual_indexes: Vec::new(),
            },
            pixels: crate::config::PixelConfig {
                mean: [0.5; 3],
                std: [0.5; 3],
            },
            dtype: DType::BF16,
            hop: 800,
        }
    }

    fn row_value(rows: &[u8], index: usize) -> f32 {
        let bits = u16::from_le_bytes([rows[2 * index], rows[2 * index + 1]]);
        bf16_to_f32(bits)
    }

    /// Patch rows run over merge blocks: the four patches of a `2 x 2` block are consecutive rows,
    /// and a block's rows come before the next block's. A row-major walk of the patches would give
    /// every merged token the wrong four, which no shape refuses.
    #[test]
    fn patch_rows_run_over_merge_blocks_and_a_row_is_one_channel_major_tubelet() {
        let format = format(1);
        // One frame of 2 rows by 4 columns, channel c of pixel (y, x) holding 8 c + 4 y + x.
        let (height, width) = (2_usize, 4_usize);
        let mut planes = vec![0_u8; 3 * height * width];
        for channel in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    planes[(channel * height + y) * width + x] = (8 * channel + 4 * y + x) as u8;
                }
            }
        }
        let rows = 8;
        let row = format.patch_width() as usize;
        assert_eq!(row, 3 * 2);
        let mut out = vec![0_u8; rows * row * 2];
        format
            .patches(&planes, 1, [1, 2, 4], &mut out)
            .expect("one frame fills one temporal patch");
        let pixel = |byte: u8| (f32::from(byte) - 127.5) / 127.5;
        let rounded = |value: f32| bf16_to_f32(f32_to_bf16(value));
        // The rows of the first block are pixels (0,0), (0,1), (1,0), (1,1); the second block's
        // first row is pixel (0,2). A row is channel-major with its two frames inside a channel, and
        // the one frame fills both.
        let order = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
        ];
        for (index, (y, x)) in order.iter().enumerate() {
            for channel in 0..3 {
                let expected = rounded(pixel((8 * channel + 4 * y + x) as u8));
                for frame in 0..2 {
                    assert_eq!(
                        row_value(&out, index * row + channel * 2 + frame),
                        expected,
                        "row {index} channel {channel} frame {frame}"
                    );
                }
            }
        }
    }

    /// An image's temporal patch repeats its one frame, and a video's short last patch repeats its
    /// last frame: a zero there would hand the tower a frame nobody decoded.
    #[test]
    fn a_short_temporal_patch_repeats_the_last_frame() {
        let format = format(2);
        let frame = 3 * 4 * 4;
        // Three frames of 4 x 4, each one value: two temporal patches, the second one frame short.
        let mut planes = vec![0_u8; 3 * frame];
        for (index, value) in [10_u8, 20, 30].iter().enumerate() {
            planes[index * frame..(index + 1) * frame].fill(*value);
        }
        let row = format.patch_width() as usize;
        let rows = 2 * 2 * 2;
        let mut out = vec![0_u8; rows * row * 2];
        format
            .patches(&planes, 3, [2, 2, 2], &mut out)
            .expect("three frames fill two temporal patches");
        let pixel = |byte: u8| bf16_to_f32(f32_to_bf16((f32::from(byte) - 127.5) / 127.5));
        // Inside a channel the tubelet holds its frames in order, four pixels each.
        assert_eq!(row_value(&out, 0), pixel(10));
        assert_eq!(row_value(&out, 4), pixel(20));
        let second = 4 * row;
        assert_eq!(row_value(&out, second), pixel(30));
        assert_eq!(row_value(&out, second + 4), pixel(30));
    }

    /// A condition's pixels are planar and normalised as the encoder reads them, and the track is
    /// two channel rows padded to whole latent frames.
    #[test]
    fn a_conditions_pixels_are_planar_and_a_track_is_padded_channel_rows() {
        let format = format(16);
        let rgb = [255_u8, 0, 128, 0, 255, 128];
        let mut out = [0.0_f32; 6];
        format
            .condition_pixels(&rgb, 1, 1, 2, &mut out)
            .expect("one frame of two pixels");
        let first = normalize([255, 0, 128]);
        let second = normalize([0, 255, 128]);
        assert_eq!(
            out,
            [
                first[0], second[0], first[1], second[1], first[2], second[2]
            ]
        );

        assert_eq!(format.track_samples(801).expect("a whole frame"), 1600);
        let pcm = [1.0_f32, -1.0, 2.0, -2.0];
        let mut track = vec![9.0_f32; 2 * 800];
        format.track(&pcm, 2, &mut track).expect("two samples");
        assert_eq!(&track[..3], &[1.0, 2.0, 0.0]);
        assert_eq!(&track[800..803], &[-1.0, -2.0, 0.0]);
    }

    use crate::video_vae::tests::video_config;

    /// A video encoder over the fixture's config: the stack is all `latent_frames` needs, so no
    /// weights are loaded.
    fn encoder_stack() -> Stack {
        Stack::of(&video_config()).expect("the fixture's stack")
    }

    /// A condition's clips cover its frames and its latent frames, whole clips and then what is
    /// left, the reference's `clips` and `latent_slices`.
    #[test]
    fn a_condition_is_cut_into_its_clips() {
        let clip = encoder_stack().clip;
        assert_eq!(
            clip.spans(1).unwrap(),
            [ClipSpan {
                first: 0,
                frames: 1,
                latent: 0,
                latents: 1
            }]
        );
        let video = clip.spans(39).unwrap();
        assert_eq!(video.len(), 3);
        assert_eq!(
            video[2],
            ClipSpan {
                first: 34,
                frames: 5,
                latent: 10,
                latents: 2
            }
        );
        for frames in [22, 39, 56, 124, 362] {
            let spans = clip.spans(frames).unwrap();
            let last = spans.last().unwrap();
            assert_eq!(last.first + last.frames, frames);
            assert_eq!(
                last.latent + last.latents,
                clip.latent_frames(frames).unwrap()
            );
        }
    }

    #[test]
    fn a_condition_is_counted_by_its_own_rule() {
        // The reference counts a condition in clips: every clip of `clip_length` frames yields
        // `clip_latents` latent frames and the last clip's trailing `token_drop` are dropped, so the
        // canonical 22 frames (`clip_length + 5`) are two clips less three.
        let clip = encoder_stack().clip;
        assert_eq!(clip.latent_frames(1).expect("a still"), 1);
        assert_eq!(
            clip.latent_frames(clip.frames).expect("one clip"),
            clip.latents - clip.token_drop
        );
        let tail = clip.frames + crate::config::CLIP_LATENTS;
        assert_eq!(
            clip.latent_frames(tail).expect("a clip and a tail"),
            2 * clip.latents - clip.token_drop
        );
        // A length no generation produces is a condition all the same: two clips and a short tail.
        assert_eq!(
            clip.latent_frames(2 * clip.frames - 4)
                .expect("a short last clip"),
            2 * clip.latents - clip.token_drop
        );
        assert!(clip.latent_frames(0).is_err());

        // The generated timeline's rule agrees on the canonical count, which is why the two must be
        // stated separately rather than one standing in for the other.
        assert_eq!(
            clip.latent_frames(tail).expect("a clip and a tail"),
            crate::packing::video_latent_frames(tail).expect("the generated rule")
        );
    }

    #[test]
    fn a_run_reserves_one_set_per_distinct_tile_extent() {
        let tile = |height: u32, width: u32| Extent {
            frames: 25,
            height,
            width,
        };
        // A uniform grid names one extent many times and reserves once; a grid with shorter edges names
        // two, in the order they first appear.
        assert_eq!(distinct(&[tile(256, 256); 28]), vec![tile(256, 256)]);
        assert_eq!(
            distinct(&[
                tile(256, 256),
                tile(256, 192),
                tile(256, 256),
                tile(256, 192)
            ]),
            vec![tile(256, 256), tile(256, 192)]
        );
        assert_eq!(
            distinct(&[tile(192, 192), tile(256, 256)]),
            vec![tile(192, 192), tile(256, 256)],
            "the order is the order they are named"
        );
    }

    #[test]
    fn a_pixel_normalizes_and_quantizes_back_to_itself() {
        // The decoder's byte step, per channel: restore, clamp to [0, 1], scale and round half to
        // even.
        let quantize = |normalized: [f32; 3]| -> [u8; 3] {
            let mut bytes = [0u8; 3];
            for (channel, value) in normalized.iter().enumerate() {
                let restored =
                    value * crate::output::PIXEL_STD[channel] + crate::output::PIXEL_MEAN[channel];
                bytes[channel] =
                    (restored.clamp(0.0, 1.0) * crate::output::BYTE_MAX).round_ties_even() as u8;
            }
            bytes
        };
        for pixel in [
            [0u8, 0, 0],
            [255, 255, 255],
            [124, 116, 104],
            [1, 2, 3],
            [200, 10, 77],
        ] {
            let restored = quantize(normalize(pixel));
            for (channel, (got, want)) in restored.iter().zip(pixel.iter()).enumerate() {
                assert!(
                    got.abs_diff(*want) <= 1,
                    "channel {channel}: {got} for {want} from {pixel:?}"
                );
            }
        }
        // The extremes land outside the band the decoder works in - the byte step maps a value
        // back through `value · std + mean` and clamps to the byte range, so one inside [-1, 1]
        // loses nothing. Being outside is the property; how far outside is the checkpoint's
        // statistics' business, not this test's.
        for channel in 0..3 {
            let low = normalize([0, 0, 0])[channel];
            let high = normalize([255, 255, 255])[channel];
            assert!(low < -1.0, "channel {channel}: {low} from zero");
            assert!(high > 1.0, "channel {channel}: {high} from 255");
        }
    }

    /// A raster view of `channels` over one frame's `rows × columns`, for the shape tests: nothing
    /// reads it, so the pointer is the null one.
    fn raster(channels: u64, frames: u64, rows: u64, columns: u64, dtype: DType) -> TensorView {
        TensorView::new(
            nn::DevicePtr::new(0),
            dtype,
            Shape::new(&[channels, frames, rows, columns]),
        )
    }

    #[test]
    fn the_posterior_sample_is_the_rasters_own_geometry() {
        let config = video_config();
        let channels = u64::from(config.latent_channels);
        let moments = raster(2 * channels, 7, 30, 52, DType::F32);
        let noise = raster(channels, 7, 30, 52, DType::F32);
        let out = raster(channels, 7, 30, 52, DType::F32);
        let sample = Sample::of(config.latent_channels, moments, noise, out).expect("a sample");
        assert_eq!(sample.channels, channels);
        assert_eq!(sample.positions, 7 * 30 * 52);
        // The moments' rows are twice the sample's, which is what makes the two fields one tensor.
        assert_eq!(moments.shape.dim(0), 2 * sample.channels);

        // A moments raster of the sample's own width has no log-variance to read.
        let narrow = raster(channels, 7, 30, 52, DType::F32);
        assert!(Sample::of(config.latent_channels, narrow, noise, out).is_err());
        // A noise raster of the moments' width is two draws where the sample wants one.
        let wide_noise = raster(2 * channels, 7, 30, 52, DType::F32);
        assert!(Sample::of(config.latent_channels, moments, wide_noise, out).is_err());
        // An FP16 raster is the decoder's precision, not the encoder's posterior.
        let half = raster(channels, 7, 30, 52, DType::F16);
        assert!(Sample::of(config.latent_channels, moments, half, out).is_err());
        // A view whose planes overlap has no single channel stride.
        let strided = TensorView::with_strides(
            nn::DevicePtr::new(0),
            DType::F32,
            Shape::new(&[channels, 7, 30, 52]),
            Strides::new(&[30 * 52, 30 * 52, 52, 1]),
        );
        assert!(Sample::of(config.latent_channels, moments, strided, out).is_err());
        // A noise raster of another clip is a different sample.
        let other = raster(channels, 5, 30, 52, DType::F32);
        assert!(Sample::of(config.latent_channels, moments, other, out).is_err());
    }

    #[test]
    fn the_encoders_affine_inverts_the_latent_statistics() {
        let config = video_config();
        let channels = config.latents_mean.len();
        let (scale, shift) =
            normalize_tables(&config.latents_mean, &config.latents_std).expect("the affine");
        assert_eq!(scale.len(), channels * channels);
        assert_eq!(shift.len(), channels);
        for channel in 0..channels {
            assert_eq!(
                scale[channel * channels + channel],
                1.0 / config.latents_std[channel]
            );
            assert_eq!(
                shift[channel],
                -config.latents_mean[channel] / config.latents_std[channel]
            );
        }
        // Off the diagonal there is nothing: each channel scales on its own.
        let off = scale
            .iter()
            .enumerate()
            .filter(|(index, value)| index % (channels + 1) != 0 && **value != 0.0)
            .count();
        assert_eq!(off, 0);

        // A deviation of zero cannot be divided by, and a shorter table names the wrong channel count.
        let mut zeroed = config.latents_std.clone();
        zeroed[3] = 0.0;
        assert!(normalize_tables(&config.latents_mean, &zeroed).is_err());
        assert!(normalize_tables(&config.latents_mean, &config.latents_std[..2]).is_err());
    }

    #[test]
    fn a_clip_of_seventeen_frames_becomes_five_latent_frames() {
        let stack = Stack::of(&video_config()).expect("the stack");
        let extents = stack.extents(Extent {
            frames: 17,
            height: 256,
            width: 256,
        });
        // 17 frames, then 17, 9, 5, 5, 5, 5: one 17-frame clip is five latent frames.
        assert_eq!(
            extents
                .iter()
                .map(|extent| extent.frames)
                .collect::<Vec<_>>(),
            vec![17, 17, 9, 5, 5, 5, 5]
        );
        // 256 pixels halve four times: the 16-pixel compression.
        assert_eq!(
            extents
                .iter()
                .map(|extent| extent.height)
                .collect::<Vec<_>>(),
            vec![256, 128, 64, 32, 16, 16, 16]
        );
        assert_eq!(stack.moments(extents[0]).expect("moments").frames, 5);
        // A single frame stays one latent frame.
        let single = stack
            .moments(Extent {
                frames: 1,
                height: 16,
                width: 16,
            })
            .expect("moments");
        assert_eq!(single.frames, 1);
        assert_eq!((single.height, single.width), (1, 1));
    }

    #[test]
    fn the_tile_grid_is_the_decoders_grid_in_input_pixels() {
        let grid = Grid::of(1344, 768).expect("the grid");
        // The decoder's grid at 16 pixels per latent position: seven rows of 256 with overlaps
        // [80, 80, 80, 80, 64, 64] and four columns with [96, 80, 80].
        assert_eq!(grid.rows.starts, vec![0, 176, 352, 528, 704, 896, 1088]);
        assert_eq!(grid.rows.overlaps, vec![80, 80, 80, 80, 64, 64]);
        assert_eq!(grid.columns.starts, vec![0, 160, 336, 512]);
        // Every seam spans whole latent positions: 80/16 = 5, 64/16 = 4, 96/16 = 6.
        assert_eq!(Grid::blends(&grid.rows), vec![5, 5, 5, 5, 4, 4]);
        assert_eq!(Grid::blends(&grid.columns), vec![6, 5, 5]);
        // The rows a raster of tiles owns: each start in latent rows, then the height.
        assert_eq!(grid.owned(84), vec![0, 11, 22, 33, 44, 56, 68, 84]);
    }

    #[test]
    fn a_band_reads_the_tile_rows_it_cross_fades_with() {
        let grid = Grid::of(1344, 768).expect("the grid");
        let latent_height = 1344 / SPATIAL_RATIO;
        assert_eq!(latent_height, 84);
        // The first band starts at the top: no row above it.
        assert_eq!(grid.band(84, 0, 11).expect("band"), (0, 0));
        // A band that starts on a row's own first latent row reads only that row and below.
        assert_eq!(grid.band(84, 16, 33).expect("band"), (1, 2));
        // A band that starts inside a row's cross-fade reads the row above.
        let owned = grid.owned(84);
        assert_eq!(owned[1], 11);
        assert_eq!(grid.rows.overlaps[0] / SPATIAL_RATIO, 5);
        assert_eq!(grid.band(84, 13, 20).expect("band"), (0, 1));
        // Past the end is refused.
        assert!(grid.band(84, 80, 90).is_err());
        assert!(grid.band(84, 5, 5).is_err());
    }

    #[test]
    fn the_audio_encoder_stages_land_on_whole_latent_frames() {
        let stack = AudioStack::of(&crate::audio_vae::tests::audio_config()).expect("the stack");
        assert_eq!(stack.samples_per_latent, 800);
        assert_eq!(stack.lengths(800), vec![800, 400, 100, 25, 5, 1]);
        assert_eq!(stack.lengths(1600), vec![1600, 800, 200, 50, 10, 2]);
        assert_eq!(stack.latent_frames(800).expect("one frame"), 1);
        assert_eq!(stack.latent_frames(24000).expect("thirty frames"), 30);
        // A partial frame rounds up, and the length the network is given is the ceiling's own multiple:
        // the reference right-pads the track with zeros, so nothing is dropped.
        assert_eq!(stack.latent_frames(801).expect("rounds up"), 2);
        assert_eq!(padded_samples(801, 800).expect("padded"), 1600);
        assert_eq!(padded_samples(1600, 800).expect("whole"), 1600);
        // Each stage doubles the width from the input convolution's.
        assert_eq!(stack.widths, vec![128, 256, 512, 1024, 2048]);
        assert_eq!(stack.latent_dim, 2048);
    }

    #[test]
    fn a_stack_with_no_stage_is_refused() {
        let mut config = video_config();
        config.block_out_channels.clear();
        assert!(Stack::of(&config).is_err());
        let mut config = video_config();
        config.spatial_downsample_factors.pop();
        assert!(Stack::of(&config).is_err());
        let mut config = video_config();
        config.temporal_downsample_factors[1] = 0;
        assert!(Stack::of(&config).is_err());
    }

    #[test]
    fn a_mono_track_is_refused_where_a_stereo_pair_passes() {
        // The stereo pair is the batch. A mono track is legal as a shape and halves the latent
        // frames, so nothing downstream notices - the reference expands it before the encoder and
        // this refuses it here, naming that expansion.
        assert_eq!(
            AudioEncoder::check_track(Shape::new(&[2, 1, 44_100])).unwrap(),
            2
        );
        let refused = AudioEncoder::check_track(Shape::new(&[1, 1, 44_100])).unwrap_err();
        let message = refused.to_string();
        assert!(message.contains("stereo pair"), "{message}");
        assert!(message.contains("unpack_samples"), "{message}");
        // And three batch rows are not "more stereo": the reference refuses a channel count that is
        // neither one nor two, and so does this.
        assert!(AudioEncoder::check_track(Shape::new(&[3, 1, 44_100])).is_err());
        // The two shapes the encoder never took are still refused, and by the other message.
        assert!(AudioEncoder::check_track(Shape::new(&[2, 3, 44_100])).is_err());
        assert!(AudioEncoder::check_track(Shape::new(&[2, 44_100])).is_err());
    }
}
