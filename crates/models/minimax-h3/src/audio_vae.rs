//! H3's audio VAE decoder: BigVGAN's stage arithmetic, its alias-free activation, and the stereo PCM
//! it ends in.
//!
//! Diffusers' `autoencoders/autoencoder_kl_minimax_h3_audio.py` is the reference. The decoder is a
//! BigVGAN: a 1×1 input projection to `latent_dim`, a kernel-7 pre-convolution, seven transposed-conv
//! upsampling stages (each followed by three anti-aliased multi-periodicity blocks whose outputs are
//! summed and averaged), one alias-free activation, and a kernel-7 post-convolution to mono, clamped
//! to [-1, 1].
//!
//! ## What this module owns
//!
//! * the stage table a deployment builds from [`AudioVaeConfig`]: every convolution's channel pair,
//!   rate, kernel and padding, so the layer stack and its weight names are derived, never transcribed;
//! * the decoder network itself ([`BigVgan`]), whose alias-free activations and stereo PCM step run
//!   as fused `nn` layers.
//!
//! ## Precision
//!
//! The audio decoder is FP32 throughout, unquantized and unconverted. Its 135 weight-normalized
//! convolutions are folded once at the loading boundary (`nn::WeightLoader::weight_norm`), so no
//! layer carries the magnitude/direction pair, and its convolutions run through cuDNN without TF32:
//! TF32 is kept for the video VAE's FP32 `conv3d` alone.

use crate::{AudioVaeConfig, Error, Result};
use nn::layers::{
    AudioActivation, Axpby, Cast, ChannelAffine, Conv1d, ConvGeometry, ConvTranspose1d,
    PcmInterleave, Residual,
};
use nn::weights::Scope;
use nn::{
    DType, DecodedMedia, Decoder, Forward, Shape, StateTable, TensorView, Weight, WeightLoader,
};

/// One upsampling stage of the decoder.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Stage {
    /// Input channels.
    pub channels_in: u32,
    /// Output channels: half the input channels.
    pub channels_out: u32,
    /// The transposed convolution's stride, i.e. the stage's upsampling rate.
    pub rate: u32,
    /// The transposed convolution's kernel.
    pub kernel: u32,
    /// The transposed convolution's zero padding: `(kernel − rate) / 2`.
    pub padding: u32,
}

/// The decoder's stages and the channel count it starts from.
///
/// The reference builds the stages from `decoder_dim` and halves the channel count at each one; a
/// sidecar whose rates and kernels disagree, or whose halving does not divide `decoder_dim`, is
/// refused at the loading boundary rather than producing a stack nobody can name.
pub fn stages(config: &AudioVaeConfig) -> Result<Vec<Stage>> {
    if config.decoder_rates.len() != config.decoder_kernel_sizes.len() {
        return Err(Error::config(format!(
            "the audio decoder has {} rates and {} kernels",
            config.decoder_rates.len(),
            config.decoder_kernel_sizes.len()
        )));
    }
    let mut channels = config.decoder_dim;
    let mut stages = Vec::with_capacity(config.decoder_rates.len());
    for (index, (&rate, &kernel)) in config
        .decoder_rates
        .iter()
        .zip(&config.decoder_kernel_sizes)
        .enumerate()
    {
        if channels < 2 || !channels.is_multiple_of(2) {
            return Err(Error::config(format!(
                "audio decoder stage {index} halves {channels} channels"
            )));
        }
        let channels_out = channels / 2;
        // `(kernel - rate) / 2` in the reference's integer arithmetic: the kernel is wider than the
        // stride at every stage of the checkpoint, so the difference is never negative.
        let padding = kernel.saturating_sub(rate) / 2;
        if rate == 0 || kernel < rate {
            return Err(Error::config(format!(
                "audio decoder stage {index} has rate {rate} and kernel {kernel}"
            )));
        }
        stages.push(Stage {
            channels_in: channels,
            channels_out,
            rate,
            kernel,
            padding,
        });
        channels = channels_out;
    }
    if stages.is_empty() {
        return Err(Error::config(
            "the audio decoder has no upsampling stage".to_owned(),
        ));
    }
    Ok(stages)
}

/// The samples one latent frame decodes to.
pub fn samples_per_latent(config: &AudioVaeConfig) -> u32 {
    config.decoder_hop()
}

/// The latent frames a sample count needs: `ceil(samples / samples_per_latent)`.
pub fn latent_frames(config: &AudioVaeConfig, samples: u32) -> Result<u32> {
    let hop = samples_per_latent(config);
    if samples == 0 {
        return Err(Error::config(
            "an audio duration of zero samples has no latent frame".to_owned(),
        ));
    }
    Ok(samples.div_ceil(hop))
}

/// The samples a latent frame count decodes to.
pub fn track_samples(config: &AudioVaeConfig, latent_frames: u32) -> u64 {
    u64::from(latent_frames) * u64::from(samples_per_latent(config))
}

/// The four buffers one anti-aliased block works in, reserved once per stage and reused by its blocks.
///
/// All four are the stage's shape because a block's channels are its stage's; the block overwrites each
/// before reading it, so the blocks of a stage take turns rather than each taking a share of the arena.
#[derive(Clone, Copy)]
struct BlockBuffers {
    running: TensorView,
    activated: TensorView,
    convolved: TensorView,
    residual: TensorView,
}

/// The anti-aliased multi-periodicity blocks one decoder stage holds.
pub const BLOCKS_PER_STAGE: usize = 3;

/// One dilation step of an anti-aliased multi-periodicity block.
///
/// A step is two convolutions of the block's kernel: the first is dilated by `dilation` and padded by
/// `dilation · (kernel − 1) / 2`, the second is dilated by one and padded by `(kernel − 1) / 2`. Both
/// keep the length.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Step {
    /// The first convolution's dilation.
    pub dilation: u32,
    /// The first convolution's zero padding; the second's is `(kernel − 1) / 2`.
    pub padding: u32,
}

impl Step {
    /// The step of `kernel` at `dilation`.
    pub fn new(kernel: u32, dilation: u32) -> Result<Self> {
        if kernel == 0 || dilation == 0 || !(kernel - 1).is_multiple_of(2) {
            return Err(Error::config(format!(
                "a branch of kernel {kernel} and dilation {dilation} has no whole padding"
            )));
        }
        Ok(Self {
            dilation,
            padding: dilation * (kernel - 1) / 2,
        })
    }

    /// The second convolution's padding, which undoes no dilation.
    pub fn padding_plain(kernel: u32) -> u32 {
        (kernel - 1) / 2
    }
}

/// One anti-aliased multi-periodicity block: one residual kernel run at each of its dilations.
///
/// The steps accumulate rather than run in parallel — each adds its convolutional residual to the
/// running value — and a block's kernel is fixed, so a stage holds one block per residual kernel and
/// sums their outputs, dividing by the kernel count.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Block {
    /// The stage the block belongs to.
    pub stage: usize,
    /// The block's index within its stage, which is also its residual kernel's index.
    pub index: usize,
    /// The block's residual kernel.
    pub kernel: u32,
    /// Channels in and out: the stage's width.
    pub channels: u32,
    /// The dilation steps, in the order the checkpoint names them.
    pub steps: Vec<Step>,
}

/// The decoder's blocks, in stage order: one per residual kernel, three per stage.
///
/// A block runs its kernel at every dilation the configuration lists for that kernel, so a kernel with
/// three dilations contributes six convolutions; the stage sums its three blocks and divides by three.
pub fn blocks(config: &AudioVaeConfig) -> Result<Vec<Block>> {
    let stages = stages(config)?;
    if config.resblock_kernel_sizes.len() != BLOCKS_PER_STAGE
        || config.resblock_dilation_sizes.len() != BLOCKS_PER_STAGE
    {
        return Err(Error::config(format!(
            "the audio decoder names {} residual kernels and {} dilation lists, not {BLOCKS_PER_STAGE}",
            config.resblock_kernel_sizes.len(),
            config.resblock_dilation_sizes.len()
        )));
    }
    let mut blocks = Vec::with_capacity(stages.len() * BLOCKS_PER_STAGE);
    for (index, stage) in stages.iter().enumerate() {
        for block in 0..BLOCKS_PER_STAGE {
            let kernel = config.resblock_kernel_sizes[block];
            let dilations = &config.resblock_dilation_sizes[block];
            if dilations.is_empty() {
                return Err(Error::config(format!(
                    "the audio residual kernel {kernel} names no dilation"
                )));
            }
            let steps = dilations
                .iter()
                .map(|dilation| Step::new(kernel, *dilation))
                .collect::<Result<Vec<_>>>()?;
            blocks.push(Block {
                stage: index,
                index: block,
                kernel,
                channels: stage.channels_out,
                steps,
            });
        }
    }
    Ok(blocks)
}

/// The latent statistics the decoder inverts before its first convolution.
///
/// A generated audio latent is normalized per channel; the decoder restores it with
/// `value * std + mean` (`LatentDecoder` in the reference) before the input projection. The two vectors
/// are load-time constants of the checkpoint, one entry per latent channel.
#[derive(Clone, PartialEq, Debug)]
pub struct Statistics {
    /// The per-channel mean.
    pub mean: Vec<f32>,
    /// The per-channel standard deviation.
    pub std: Vec<f32>,
}

impl Statistics {
    /// The statistics the config states, one entry per latent channel.
    pub fn of(config: &AudioVaeConfig) -> Result<Self> {
        let channels = config.latent_channels as usize;
        if config.latents_mean.len() != channels || config.latents_std.len() != channels {
            return Err(Error::config(format!(
                "the audio latent statistics hold {} means and {} deviations for {channels} channels",
                config.latents_mean.len(),
                config.latents_std.len()
            )));
        }
        for (index, (mean, std)) in config
            .latents_mean
            .iter()
            .zip(config.latents_std.iter())
            .enumerate()
        {
            if !mean.is_finite() || !std.is_finite() || *std == 0.0 {
                return Err(Error::config(format!(
                    "the audio latent channel {index} has mean {mean} and deviation {std}"
                )));
            }
        }
        Ok(Self {
            mean: config.latents_mean.clone(),
            std: config.latents_std.clone(),
        })
    }
}

/// The audio decoder: the checkpoint's BigVGAN network, one call per run of latent frames.
///
/// Layout everywhere is `[batch, channels, samples]`, the form the one-dimensional convolutions take.
/// The batch is two because the reference decodes a stereo pair of identical rows and interleaves them;
/// nothing in the network mixes the rows, so both output channels carry the same signal.
pub struct BigVgan {
    config: AudioVaeConfig,
    /// The latent statistics as one one-by-one convolution: a diagonal of the deviations and the means
    /// as its bias, which is `value · std + mean` per latent channel.
    denormalize: ChannelAffine,
    input: Conv1d,
    pre: Conv1d,
    stages: Vec<DecoderStage>,
    post: AudioActivation,
    output: Conv1d,
    pcm: PcmInterleave,
    /// Moves the track from the denoiser's rows into the decoder's channel-major layout.
    track: nn::layers::Copy,
}

/// One upsampling stage: a transposed convolution whose output is the sum of three anti-aliased blocks,
/// divided by three.
struct DecoderStage {
    upsample: ConvTranspose1d,
    blocks: Vec<DecoderBlock>,
    /// Latent samples become `rate` times as many.
    rate: u32,
    /// The stage's width, which its blocks keep.
    channels: u32,
}

/// One anti-aliased multi-periodicity block: one residual kernel run at each of its dilations, the steps
/// accumulating on the running value.
struct DecoderBlock {
    steps: Vec<DecoderStep>,
}

/// One dilation step: an alias-free activation, the dilated convolution, a second activation and the
/// plain convolution, whose result is added to the running value.
struct DecoderStep {
    first: Conv1d,
    first_activation: AudioActivation,
    second: Conv1d,
    second_activation: AudioActivation,
}

/// The alias-free activation's geometry, from the checkpoint's own constants.
const ACTIVATION_KERNEL: u32 = 12;
const ACTIVATION_RATIO: u32 = 2;
/// The program's width, as the operator states it - **a key field**, not a request's length.
///
/// The launcher passes this to `AudioActivation::new`, so it is part of the key the plan walk records.
const ACTIVATION_BLOCK: u32 = 256;
/// The two channels the reference decodes and interleaves.
/// The rows of the PCM window a decode writes: one per stereo channel, interleaved.
///
/// Public because a caller sizes `DecodedMedia.audio` before a graph exists - the window is
/// `[track_samples(config, frames), PCM_CHANNELS]` - and a row count restated outside this crate would
/// be a second statement of the wire format.
pub const PCM_CHANNELS: u32 = 2;

impl BigVgan {
    /// The sample clock the checkpoint was trained at, in hertz.
    ///
    /// It is the audio vae config's own number (`audio_vae/config.json`'s `sampling_rate`), not a
    /// constant this crate types: the muxer builds the artifact's timeline from it, so a checkpoint
    /// trained at another rate would state that rate rather than be resampled silently.
    pub fn sampling_rate(&self) -> u32 {
        self.config.sampling_rate
    }

    /// The audio decoder the checkpoint's `dec_in_proj` and `decoder` modules state, read in FP32.
    ///
    /// The checkpoint's names are the reference's after its name translation: the input
    /// projection `dec_in_proj`, then `decoder.conv_pre`, the upsamplers `decoder.ups.<i>.0`, the
    /// residual blocks `decoder.resblocks.<stage · 3 + block>`, `decoder.activation_post` and
    /// `decoder.conv_post`. The body's convolutions are weight-normalized and folded once here.
    pub fn load(config: &AudioVaeConfig, loader: &mut WeightLoader) -> Result<Self> {
        let statistics = Statistics::of(config)?;
        let (scale, shift) = denormalize_tables(&statistics)?;
        let channels = u64::from(config.latent_channels);
        let denormalize = {
            let mut constants = loader.scope("audio_vae.denormalize");
            // The latents are `[batch, channels, frames]`: the channel is axis one.
            ChannelAffine::new(
                constants.constant_f32("scale", &[channels], &scale)?,
                constants.constant_f32("shift", &[channels], &shift)?,
                1,
            )?
        };
        // The audio decoder computes in FP32 throughout (`DType::F32` below), so every weight is
        // read in it whatever the checkpoint stores.
        let mut root = loader.scope("");
        let mut root = root.pinned(DType::F32);
        // The decoder's input is a plain 1x1 convolution in the reference - `nn.Conv1d(
        // latent_channels, latent_dim, 1)` outside `_normalized` - and the export holds its weight
        // as a **3-d filter** `[latent_dim, latent_channels, 1]`, the rank `Conv1d::new` requires.
        let input = {
            let mut input = root.scope("dec_in_proj");
            Conv1d::new(
                input.tensor(
                    "weight",
                    &[
                        u64::from(config.latent_dim),
                        u64::from(config.latent_channels),
                        1,
                    ],
                )?,
                Some(input.tensor("bias", &[u64::from(config.latent_dim)])?),
                geometry(1, 1, 0)?,
                false,
            )?
        };
        let mut decoder = root.scope("decoder");
        let (weight, bias) = folded(
            &mut decoder.scope("conv_pre"),
            Some(u64::from(config.decoder_dim)),
        )?;
        let pre = Conv1d::new(weight, bias, geometry(7, 1, 3)?, false)?;
        let stages = stages(config)?;
        let blocks = blocks(config)?;
        let mut built = Vec::with_capacity(stages.len());
        for (index, stage) in stages.iter().enumerate() {
            let (weight, bias) = folded(
                &mut decoder.scope(format!("ups.{index}.0")),
                Some(u64::from(stage.channels_out)),
            )?;
            let upsample = ConvTranspose1d::new(
                weight,
                bias,
                geometry(stage.kernel, stage.rate, stage.padding)?,
                false,
            )?;
            let width = u64::from(stage.channels_out);
            let mut stage_blocks = Vec::with_capacity(BLOCKS_PER_STAGE);
            for block in blocks.iter().filter(|block| block.stage == index) {
                let mut module = decoder.scope(format!(
                    "resblocks.{}",
                    index * BLOCKS_PER_STAGE + block.index
                ));
                let mut steps = Vec::with_capacity(block.steps.len());
                for (step, dilation) in block.steps.iter().enumerate() {
                    let (weight, bias) =
                        folded(&mut module.scope(format!("convs1.{step}")), Some(width))?;
                    let first = Conv1d::new(
                        weight,
                        bias,
                        dilated_geometry(block.kernel, 1, dilation.padding, dilation.dilation)?,
                        false,
                    )?;
                    let first_activation = activation(
                        &mut module.scope(format!("activations.{}", 2 * step)),
                        width,
                    )?;
                    let (weight, bias) =
                        folded(&mut module.scope(format!("convs2.{step}")), Some(width))?;
                    let second = Conv1d::new(
                        weight,
                        bias,
                        geometry(block.kernel, 1, Step::padding_plain(block.kernel))?,
                        false,
                    )?;
                    let second_activation = activation(
                        &mut module.scope(format!("activations.{}", 2 * step + 1)),
                        width,
                    )?;
                    steps.push(DecoderStep {
                        first,
                        first_activation,
                        second,
                        second_activation,
                    });
                }
                // A block runs at its stage's width and needs no field of its own for it: the stage
                // reserves the buffers and states the width they are sized by.
                stage_blocks.push(DecoderBlock { steps });
            }
            built.push(DecoderStage {
                upsample,
                blocks: stage_blocks,
                rate: stage.rate,
                channels: stage.channels_out,
            });
        }
        let last = stages
            .last()
            .ok_or_else(|| Error::config("the audio decoder has no stage".to_owned()))?;
        let post = activation(
            &mut decoder.scope("activation_post"),
            u64::from(last.channels_out),
        )?;
        let (weight, _) = folded(&mut decoder.scope("conv_post"), None)?;
        let output = Conv1d::new(weight, None, geometry(7, 1, 3)?, false)?;
        Ok(Self {
            config: config.clone(),
            denormalize,
            input,
            pre,
            stages: built,
            post,
            output,
            pcm: PcmInterleave::new(DType::F32)?,
            track: nn::layers::Copy::new(DType::F32, nn::layers::Access::Element)?,
        })
    }

    /// The denoiser's track in the layout [`Self::forward`] reads.
    ///
    /// The denoiser publishes its audio rows as they are: `[frames, 2 · latent_channels]` over the
    /// bytes of `[2 · frames, latent_channels]`, the first stereo channel's rows before the second's.
    /// The decoder's convolutions run along time, so each stereo channel's latent channels must
    /// become the outer axis: this is the reference's `reshape(2, frames, channels).permute(0, 2, 1)`.
    fn channel_major(&self, f: &Forward, track: TensorView) -> Result<TensorView> {
        let channels = u64::from(self.config.latent_channels);
        let stereo = u64::from(PCM_CHANNELS);
        let shape = track.shape;
        if shape.rank() != 2 || shape.dim(1) != stereo * channels {
            return Err(Error::config(format!(
                "{:?} is not a track of [frames, {}]",
                shape.dims(),
                stereo * channels
            )));
        }
        let frames = shape.dim(0);
        let major = self.buffer(f, self.config.latent_channels, frames)?;
        // Four axes are what a block copy moves: `[stereo, channels, 1, frames]`.
        let block = Shape::new(&[stereo, channels, 1, frames]);
        let source = TensorView::with_strides(
            track.offset_elements(0),
            track.dtype,
            block,
            nn::Strides::new(&[frames * channels, 1, 1, channels]),
        );
        let strides = [channels * frames, frames, frames, 1].map(|stride| stride as i64);
        let destination = TensorView::with_strides(
            major.offset_elements(0),
            major.dtype,
            block,
            nn::Strides::new(&[channels * frames, frames, frames, 1]),
        );
        self.track.forward(f, source, destination, &strides)?;
        Ok(major)
    }

    /// Decode `latents [2, latent_channels, frames]` into interleaved PCM `[samples, 2]`.
    ///
    /// The two rows of the latent are the stereo pair the reference decodes; both hold the same latent,
    /// so the network's two output rows are equal and [`PcmInterleave`] writes them interleaved. Every
    /// intermediate is FP32 — the audio decoder keeps no reduced precision — and the
    /// length grows only through the stages' transposed convolutions, so `samples` is `frames · Π rates`.
    pub fn forward(&self, f: &Forward, latents: TensorView, out: TensorView) -> Result<()> {
        let shape = latents.shape;
        if shape.rank() != 3
            || shape.dim(0) != u64::from(PCM_CHANNELS)
            || shape.dim(1) != u64::from(self.config.latent_channels)
        {
            return Err(Error::config(format!(
                "{:?} is not [{PCM_CHANNELS}, {}, frames] latents",
                shape.dims(),
                self.config.latent_channels
            )));
        }
        let mut length = shape.dim(2);
        // The reference restores each latent channel (`value · std + mean`) before the decoder's first
        // convolution, and this decoder's first convolution is weight-normalized, so its values exist
        // only after the fold: the statistics are one map of constants instead of a fold into that
        // projection.
        let restored = self.buffer(f, self.config.latent_channels, length)?;
        self.denormalize.forward(f, latents, restored)?;
        let projected = self.buffer(f, self.config.latent_dim, length)?;
        self.input.forward(f, restored, projected)?;
        let mut current = self.buffer(f, self.config.decoder_dim, length)?;
        self.pre.forward(f, projected, current)?;
        for stage in &self.stages {
            length = length
                .checked_mul(u64::from(stage.rate))
                .ok_or_else(|| Error::config("the audio track's length overflows".to_owned()))?;
            let upsampled = self.buffer(f, stage.channels, length)?;
            stage.upsample.forward(f, current, upsampled)?;
            current = self.stage(f, stage, upsampled, length)?;
        }
        let activated = self.buffer(f, self.last_channels()?, length)?;
        self.post.forward(f, current, activated)?;
        let mono = f.arena().tensor(
            DType::F32,
            Shape::new(&[u64::from(PCM_CHANNELS), 1, length]),
        )?;
        self.output.forward(f, activated, mono)?;
        let planes = mono.reshape(Shape::new(&[u64::from(PCM_CHANNELS), length]));
        self.pcm.forward(f, planes, out)?;
        Ok(())
    }

    /// One stage's output: its three blocks summed and divided by three.
    fn stage(
        &self,
        f: &Forward,
        stage: &DecoderStage,
        input: TensorView,
        length: u64,
    ) -> Result<TensorView> {
        let accumulated = self.buffer(f, stage.channels, length)?;
        // The four buffers a block needs are the stage's shape, not the block's, so they are reserved
        // once here and reused by every block: a block overwrites all four before it reads them (its
        // first `Cast` writes `running` from the stage's input), while the running
        // sum is a buffer of its own because it must survive the loop. Reserving them per block instead
        // multiplied the decoder's arena by the block count, and the arena never rewinds.
        let scratch = BlockBuffers {
            running: self.buffer(f, stage.channels, length)?,
            activated: self.buffer(f, stage.channels, length)?,
            convolved: self.buffer(f, stage.channels, length)?,
            residual: self.buffer(f, stage.channels, length)?,
        };
        for (index, block) in stage.blocks.iter().enumerate() {
            let block_out = self.block(f, block, input, scratch)?;
            // The first block initialises the sum; the rest add to it.
            let beta = if index == 0 { 0.0 } else { 1.0 };
            Axpby.forward(f, rows(accumulated)?, rows(block_out)?, 1.0, beta)?;
        }
        // The reference divides the sum by the number of kernels; multiplying by the reciprocal differs
        // only in the last bit of a FP32 value, three orders below one PCM step.
        Axpby.forward(
            f,
            rows(accumulated)?,
            rows(accumulated)?,
            0.0,
            1.0 / BLOCKS_PER_STAGE as f32,
        )?;
        Ok(accumulated)
    }

    /// One block: its steps accumulate on the stage's input, which the block never modifies.
    fn block(
        &self,
        f: &Forward,
        block: &DecoderBlock,
        input: TensorView,
        scratch: BlockBuffers,
    ) -> Result<TensorView> {
        if block.steps.is_empty() {
            return Err(Error::config("an audio block has no step".to_owned()));
        }
        let BlockBuffers {
            running,
            activated,
            convolved,
            residual,
        } = scratch;
        // The running value starts as a copy of the stage's input, which stays untouched. Every
        // block does this first, which is what makes one set of buffers reusable across the stage.
        Cast.forward(f, rows(input)?, rows(running)?)?;
        for step in &block.steps {
            step.first_activation.forward(f, running, activated)?;
            step.first.forward(f, activated, convolved)?;
            step.second_activation.forward(f, convolved, activated)?;
            step.second.forward(f, activated, residual)?;
            let running_rows = rows(running)?;
            Residual::default().forward(f, running_rows, rows(residual)?, None, running_rows)?;
        }
        Ok(running)
    }

    /// The width the last stage leaves, which the post activation and the output convolution read.
    fn last_channels(&self) -> Result<u32> {
        self.stages
            .last()
            .map(|stage| stage.channels)
            .ok_or_else(|| Error::config("the audio decoder has no stage".to_owned()))
    }

    /// A FP32 buffer of the audio network's layout.
    fn buffer(&self, f: &Forward, channels: u32, length: u64) -> Result<TensorView> {
        Ok(f.arena().tensor(
            DType::F32,
            Shape::new(&[u64::from(PCM_CHANNELS), u64::from(channels), length]),
        )?)
    }
}

/// The statistics as the scale and the shift of the decoder's restore, `value · std + mean` per
/// latent channel: the deviations and the means.
pub fn denormalize_tables(statistics: &Statistics) -> Result<(Vec<f32>, Vec<f32>)> {
    let channels = statistics.mean.len();
    if channels == 0 || statistics.std.len() != channels {
        return Err(Error::config(format!(
            "the latent statistics hold {} means and {} deviations",
            statistics.mean.len(),
            statistics.std.len()
        )));
    }
    Ok((statistics.std.clone(), statistics.mean.clone()))
}

impl Decoder for BigVgan {
    /// The element type this decoder computes in.
    ///
    /// `F32`: the audio decoder's arithmetic is single precision throughout - its weights are
    /// stored in the checkpoint's type and stream at this one - so the latents column and the PCM
    /// raster it writes are `f32` and `i16` respectively, and the PCM half states its own type at
    /// the write (`DecodedMedia::audio`).
    fn element_type(&self) -> nn::DType {
        nn::DType::F32
    }

    fn layout_tables(&self, _layout: &nn::capability::Layout) -> nn::Result<Vec<StateTable>> {
        // Every table this decoder reads is a weight, bound at load: nothing follows from the
        // layout, so the runner uploads nothing for it. `Ok` of nothing for every layout is the
        // correct answer, and different from a failure to answer.
        Ok(Vec::new())
    }

    fn prepare_layout(
        &self,
        _layout: &nn::capability::Layout,
        out: &mut [nn::TableValues],
    ) -> nn::Result<()> {
        if out.is_empty() {
            Ok(())
        } else {
            Err(nn::Error::Shape {
                owner: "BigVgan",
                problem: format!(
                    "the audio decoder has no layout tables; {} buffers were offered",
                    out.len()
                ),
            })
        }
    }

    /// The interleaved PCM window one decode of `audio` writes.
    ///
    /// The window's frames are the track's latent steps multiplied by the product of the seven
    /// stages' rates, which is what [`track_samples`] answers from the checkpoint's own constants;
    /// the two channels are [`PCM_CHANNELS`]. Both are stated once here so a runner sizes its column
    /// from the model rather than restating the rates.
    fn pcm_window(&self, audio: &nn::capability::LatentShape) -> nn::Result<(u64, u32)> {
        Ok((track_samples(&self.config, audio.frames), PCM_CHANNELS))
    }

    /// Decode one run of latent frames into interleaved PCM.
    ///
    /// The audio decoder has no layout constants: every table it reads is a weight, bound at load, so
    /// `state` must be empty and the runner uploads nothing for it. `out.audio` is the request's
    /// `[samples, 2]` i16 window and `out.video` must be absent — this path decodes sound only.
    ///
    /// **Refused at the capability:** a `state` entry of any kind, a request with no audio window, and a
    /// request that asks for video as well. The window's length is the latent's frames times the product
    /// of the seven stages' rates, and sizing it is the runner's.
    fn decode(
        &self,
        f: &Forward,
        latents: TensorView,
        state: &[TensorView],
        out: DecodedMedia,
    ) -> nn::Result<()> {
        if !state.is_empty() {
            return Err(nn::Error::Shape {
                owner: "BigVgan",
                problem: format!(
                    "the audio decoder reads no layout table, {} were given",
                    state.len()
                ),
            });
        }
        let audio = out.audio.ok_or_else(|| nn::Error::Shape {
            owner: "BigVgan",
            problem: "an audio decode writes the request's PCM window".to_owned(),
        })?;
        if out.video.is_some() {
            return Err(nn::Error::Shape {
                owner: "BigVgan",
                problem: "the audio decoder writes no video raster".to_owned(),
            });
        }
        self.channel_major(f, latents)
            .and_then(|latents| self.forward(f, latents, audio))
            .map_err(|error| nn::Error::Shape {
                owner: "BigVgan",
                problem: format!("{error}"),
            })
    }
}

/// The 1-d convolution geometry: `kernel` taps, `stride`, `padding`, no dilation of its own, one group.
pub(crate) fn geometry(kernel: u32, stride: u32, padding: u32) -> Result<ConvGeometry> {
    dilated_geometry(kernel, stride, padding, 1)
}

/// A one-dimensional convolution's geometry with a dilation, which the encoder's residual units use at
/// `1`, `3` and `9`.
pub(crate) fn dilated_geometry(
    kernel: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
) -> Result<ConvGeometry> {
    // The operator states a 1-d convolution's length as the first spatial axis, `[length, 1, 1]`, so
    // the geometry's other two axes are the identity: a filter of one, no padding.
    Ok(ConvGeometry::new(
        1,
        [kernel, 1, 1],
        [stride, 1, 1],
        [padding, 0, 0],
        [dilation, 1, 1],
        1,
    )?)
}

/// A view of `[channels, rows, columns]` as `[rows · channels, columns]`, the form the in-place
/// accumulation takes.
fn rows(view: TensorView) -> Result<TensorView> {
    let columns = view.shape.dim(view.shape.rank() - 1);
    if columns == 0 {
        return Err(Error::config("a tensor of no column".to_owned()));
    }
    Ok(view.reshape(Shape::new(&[view.shape.numel() / columns, columns])))
}

/// The weight-normalized convolution module at `scope`: its `weight`, folded from the stored
/// `weight_v`/`weight_g` pair, and its `[bias]` of `bias` channels when it has one.
///
/// # Errors
///
/// When the checkpoint lacks a tensor or holds another extent.
pub(crate) fn folded(
    scope: &mut Scope<'_, '_>,
    bias: Option<u64>,
) -> Result<(Weight, Option<Weight>)> {
    let weight = scope.weight_norm("weight")?;
    let bias = bias
        .map(|channels| scope.tensor("bias", &[channels]))
        .transpose()?;
    Ok((weight, bias))
}

/// The alias-free activation module at `scope`: its two log-space frequencies (`act.alpha`,
/// `act.beta`) of `channels`, and its two Kaiser-sinc filters (`upsample.filter`,
/// `downsample.lowpass.filter`).
///
/// # Errors
///
/// When the checkpoint lacks a tensor or holds another extent.
fn activation(scope: &mut Scope<'_, '_>, channels: u64) -> Result<AudioActivation> {
    let alpha = scope.tensor("act.alpha", &[channels])?;
    let beta = scope.tensor("act.beta", &[channels])?;
    let filter = [1, 1, u64::from(ACTIVATION_KERNEL)];
    let up = scope.tensor("upsample.filter", &filter)?;
    let down = scope.tensor("downsample.lowpass.filter", &filter)?;
    Ok(AudioActivation::new(
        DType::F32,
        u32::try_from(channels)
            .map_err(|_| Error::config("an audio width wider than 32 bits".to_owned()))?,
        ACTIVATION_KERNEL,
        ACTIVATION_RATIO,
        ACTIVATION_BLOCK,
    )?
    .with_weights(alpha, beta, up, down)?)
}

#[cfg(test)]
pub(crate) mod tests {

    use super::*;

    pub(crate) fn audio_config() -> AudioVaeConfig {
        let sidecar = serde_json::json!({
            "encoder_dim": 64,
            "encoder_rates": [2, 4, 4, 5, 5],
            "latent_dim": 2048,
            "latent_channels": 32,
            "decoder_dim": 1024,
            "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
            "decoder_kernel_sizes": [9, 9, 4, 4, 4, 4, 4],
            "num_attention_heads": 8,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "sampling_rate": 32000,
            "latents_mean": vec![0.0; 32],
            "latents_std": vec![1.0; 32],
        });
        AudioVaeConfig::parse(&sidecar).expect("the checkpoint's audio VAE sidecar")
    }

    #[test]
    fn the_stage_table_is_the_checkpoints() {
        let stages = stages(&audio_config()).expect("stages");
        let table: Vec<(u32, u32, u32, u32, u32)> = stages
            .iter()
            .map(|s| (s.channels_in, s.channels_out, s.rate, s.kernel, s.padding))
            .collect();
        assert_eq!(
            table,
            vec![
                (1024, 512, 5, 9, 2),
                (512, 256, 5, 9, 2),
                (256, 128, 2, 4, 1),
                (128, 64, 2, 4, 1),
                (64, 32, 2, 4, 1),
                (32, 16, 2, 4, 1),
                (16, 8, 2, 4, 1),
            ]
        );
    }

    #[test]
    fn the_decoder_hop_is_eight_hundred_samples_per_latent() {
        let config = audio_config();
        assert_eq!(samples_per_latent(&config), 800);
        assert_eq!(latent_frames(&config, 800).expect("one frame"), 1);
        assert_eq!(latent_frames(&config, 801).expect("two frames"), 2);
        assert!(latent_frames(&config, 0).is_err());
        assert_eq!(track_samples(&config, 5), 4000);
        // A five-second track at 32 kHz.
        let samples = 5 * 32000;
        let frames = latent_frames(&config, samples).expect("frames");
        assert_eq!(frames, 200);
        assert_eq!(track_samples(&config, frames), u64::from(samples));
    }

    #[test]
    fn the_restore_scales_by_the_deviations_and_shifts_by_the_means() {
        let config = crate::audio_vae::tests::audio_config();
        let statistics = Statistics::of(&config).expect("the statistics");
        let (scale, shift) = denormalize_tables(&statistics).expect("the tables");
        assert_eq!(scale, statistics.std);
        assert_eq!(shift, statistics.mean);
    }

    #[test]
    fn latent_statistics_of_the_wrong_count_or_a_zero_deviation_are_refused() {
        let config = audio_config();
        let statistics = Statistics::of(&config).expect("the statistics");
        assert_eq!(statistics.mean.len(), 32);
        assert_eq!(statistics.std.len(), 32);
        // A sidecar with the wrong counts, or a zero deviation, is refused at the boundary.
        let mut broken = config.clone();
        broken.latents_mean.pop();
        assert!(Statistics::of(&broken).is_err());
        let mut broken = config.clone();
        broken.latents_std[3] = 0.0;
        assert!(Statistics::of(&broken).is_err());
    }

    #[test]
    fn every_block_runs_its_kernel_at_every_dilation() {
        let config = audio_config();
        let table = blocks(&config).expect("the blocks");
        // Seven stages of three blocks, one block per residual kernel: BigVGAN's 21, with 126
        // convolutions between them — six per block, two per dilation.
        assert_eq!(table.len(), 21);
        let convolutions: usize = table.iter().map(|block| block.steps.len() * 2).sum();
        assert_eq!(convolutions, 126);
        let first = &table[0];
        assert_eq!((first.stage, first.index), (0, 0));
        assert_eq!(first.kernel, 3);
        assert_eq!(first.channels, 512);
        // Kernel 3 at dilations 1, 3 and 5: padding `dilation · (kernel − 1) / 2`.
        assert_eq!(
            first.steps,
            vec![
                Step {
                    dilation: 1,
                    padding: 1
                },
                Step {
                    dilation: 3,
                    padding: 3
                },
                Step {
                    dilation: 5,
                    padding: 5
                },
            ]
        );
        // Each step's two convolutions keep the length: `2·padding = dilation · (kernel − 1)` for the
        // dilated one, and `kernel − 1` for the plain one.
        for step in &first.steps {
            assert_eq!(2 * step.padding, step.dilation * (first.kernel - 1));
        }
        assert_eq!(2 * Step::padding_plain(first.kernel), first.kernel - 1);
        // The three blocks of a stage are the three residual kernels.
        let kernels: Vec<u32> = table[0..3].iter().map(|block| block.kernel).collect();
        assert_eq!(kernels, vec![3, 7, 11]);
        assert_eq!(table[1].steps.len(), 3);
        assert_eq!(table[1].steps[0].padding, 3);
        // The last stage is the narrowest.
        assert_eq!(table.last().expect("a block").channels, 8);
        // A kernel with no dilation is refused rather than silently contributing nothing.
        let mut broken = config.clone();
        broken.resblock_dilation_sizes[0].clear();
        assert!(blocks(&broken).is_err());
        // So is a configuration that does not name three residual kernels.
        let mut broken = config.clone();
        broken.resblock_kernel_sizes.pop();
        assert!(blocks(&broken).is_err());
    }
}
