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
    AudioActivation, Conv1d, ConvGeometry, ConvTranspose1d, PcmInterleave, ScaledAdd,
};
use nn::weights::WeightRequest;
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

/// One logical parameter of the audio decoder and the checkpoint tensor that fills it.
///
/// The names are composed from the sidecar's stage table, never transcribed: `parameter` is the
/// decoder's own path and `tensor` is the checkpoint's key (`audio_vae/diffusion_pytorch_model.
/// safetensors` after the reference's name translation, `dec_in_proj.*` and `decoder.*`).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Source {
    /// The decoder's path for the parameter.
    pub parameter: String,
    /// Where the checkpoint stores it.
    pub tensor: Tensor,
}

/// How the checkpoint stores one tensor.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Tensor {
    /// One tensor under one key.
    Region(String),
    /// A weight-normalized convolution: `value` holds the weight's extents and `gain` one magnitude
    /// per output channel. `nn::WeightLoader::weight_norm` folds the pair once at load, so no layer
    /// below carries it.
    Normalized {
        /// The key of the direction tensor (`weight_v`).
        value: String,
        /// The key of the magnitude tensor (`weight_g`).
        gain: String,
    },
}

/// Every parameter of the decoder, with its checkpoint key.
///
/// The decoder is 135 weight-normalized convolutions (the pre- and post-convolutions, the seven
/// upsamplers, and 126 anti-aliased multi-periodicity convolutions), 127 alias-free activations —
/// each with a Kaiser-sinc filter pair and a log-space `alpha`/`beta` — and the input projection with
/// its bias.
pub fn sources(config: &AudioVaeConfig) -> Result<Vec<Source>> {
    let stages = stages(config)?;
    let mut sources = Vec::new();
    let region = |parameter: String, tensor: String| Source {
        parameter,
        tensor: Tensor::Region(tensor),
    };
    let normalized = |parameter: String, base: String| Source {
        parameter,
        tensor: Tensor::Normalized {
            value: format!("{base}.weight_v"),
            gain: format!("{base}.weight_g"),
        },
    };
    let activation = |sources: &mut Vec<Source>, parameter: String, base: String| {
        for field in [
            "act.alpha",
            "act.beta",
            "upsample.filter",
            "downsample.lowpass.filter",
        ] {
            sources.push(region(
                format!("{parameter}.{field}"),
                format!("{base}.{field}"),
            ));
        }
    };

    // The decoder's input is a plain 1x1 convolution in the reference - `nn.Conv1d(latent_channels,
    // latent_dim, 1)` outside `_normalized` - and the checkpoint stores it as one weight, so a
    // normalized pair here asked for two tensors the export does not have and left the one it does
    // have unread. The body's 136 convolutions *are* weight-normalized and stay so.
    sources.push(region(
        "input.weight".to_owned(),
        "dec_in_proj.weight".to_owned(),
    ));
    sources.push(region(
        "input.bias".to_owned(),
        "dec_in_proj.bias".to_owned(),
    ));
    sources.push(normalized(
        "pre.weight".to_owned(),
        "decoder.conv_pre".to_owned(),
    ));
    sources.push(region(
        "pre.bias".to_owned(),
        "decoder.conv_pre.bias".to_owned(),
    ));

    for (index, _stage) in stages.iter().enumerate() {
        let base = format!("decoder.ups.{index}.0");
        sources.push(normalized(
            format!("stages.{index}.upsample.weight"),
            base.clone(),
        ));
        sources.push(region(
            format!("stages.{index}.upsample.bias"),
            format!("{base}.bias"),
        ));
        // Three anti-aliased multi-periodicity blocks per stage, in the reference's order.
        for block in 0..BLOCKS_PER_STAGE {
            let base = format!("decoder.resblocks.{}", index * BLOCKS_PER_STAGE + block);
            let parameter = |name: &str| format!("stages.{index}.blocks.{block}.{name}");
            for dilation in 0..config.resblock_dilation_sizes.len() {
                for pair in ["convs1", "convs2"] {
                    sources.push(normalized(
                        parameter(&format!("{pair}.{dilation}.weight")),
                        format!("{base}.{pair}.{dilation}"),
                    ));
                    sources.push(region(
                        parameter(&format!("{pair}.{dilation}.bias")),
                        format!("{base}.{pair}.{dilation}.bias"),
                    ));
                }
                // A dilation contributes two activations: one before each convolution.
                for half in 0..2 {
                    let at = 2 * dilation + half;
                    activation(
                        &mut sources,
                        parameter(&format!("activations.{at}")),
                        format!("{base}.activations.{at}"),
                    );
                }
            }
        }
    }

    activation(
        &mut sources,
        "post".to_owned(),
        "decoder.activation_post".to_owned(),
    );
    sources.push(normalized(
        "output.weight".to_owned(),
        "decoder.conv_post".to_owned(),
    ));
    Ok(sources)
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
    denormalize: Conv1d,
    input: Conv1d,
    pre: Conv1d,
    stages: Vec<DecoderStage>,
    post: AudioActivation,
    output: Conv1d,
    pcm: PcmInterleave,
    /// Moves the track from the denoiser's rows into the decoder's channel-major layout.
    track: nn::layers::CopyBlock,
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

    /// Bind every parameter the checkpoint's audio decoder has, through [`sources`].
    ///
    /// The table is consumed exactly: a tensor it names but the load never reads is an error, so a
    /// checkpoint field cannot be silently dropped, and a parameter read twice is an error too.
    pub fn load(config: &AudioVaeConfig, loader: &mut WeightLoader) -> Result<Self> {
        let table = sources(config)?;
        let statistics = Statistics::of(config)?;
        let (scale, shift) = denormalize_tables(&statistics)?;
        let channels = u64::from(config.latent_channels);
        let denormalize = Conv1d::new(
            constants(
                loader,
                "audio_vae.denormalize.weight",
                Shape::new(&[channels, channels, 1]),
                &scale,
            )?,
            Some(constants(
                loader,
                "audio_vae.denormalize.bias",
                Shape::new(&[channels]),
                &shift,
            )?),
            geometry(1, 1, 0)?,
            false,
        )?;
        let mut binder = Binder::new(loader, table);
        // The input projection is one weight and a bias, and the weight is a **3-d filter**: the
        // export holds `[latent_dim, latent_channels, 1]` - a kernel of one is the same arithmetic
        // and not the same rank, and `Conv1d::new` refuses anything that is not 3-d. Asking for two
        // dimensions would trade a mapping refusal for a rank refusal one line later.
        let input = Conv1d::new(
            binder.region(
                "input.weight",
                &[
                    u64::from(config.latent_dim),
                    u64::from(config.latent_channels),
                    1,
                ],
            )?,
            Some(binder.region("input.bias", &[u64::from(config.latent_dim)])?),
            geometry(1, 1, 0)?,
            false,
        )?;
        let pre = Conv1d::new(
            binder.normalized("pre.weight")?,
            Some(binder.region("pre.bias", &[u64::from(config.decoder_dim)])?),
            geometry(7, 1, 3)?,
            false,
        )?;
        let stages = stages(config)?;
        let blocks = blocks(config)?;
        let mut built = Vec::with_capacity(stages.len());
        for (index, stage) in stages.iter().enumerate() {
            let upsample = ConvTranspose1d::new(
                binder.normalized(&format!("stages.{index}.upsample.weight"))?,
                Some(binder.region(
                    &format!("stages.{index}.upsample.bias"),
                    &[u64::from(stage.channels_out)],
                )?),
                geometry(stage.kernel, stage.rate, stage.padding)?,
                false,
            )?;
            let width = u64::from(stage.channels_out);
            let mut stage_blocks = Vec::with_capacity(BLOCKS_PER_STAGE);
            for block in blocks.iter().filter(|block| block.stage == index) {
                let at = |name: &str| format!("stages.{index}.blocks.{}.{name}", block.index);
                let mut steps = Vec::with_capacity(block.steps.len());
                for (step, dilation) in block.steps.iter().enumerate() {
                    let first = Conv1d::new(
                        binder.normalized(&at(&format!("convs1.{step}.weight")))?,
                        Some(binder.region(&at(&format!("convs1.{step}.bias")), &[width])?),
                        dilated_geometry(block.kernel, 1, dilation.padding, dilation.dilation)?,
                        false,
                    )?;
                    let first_activation =
                        binder.activation(&at(&format!("activations.{}", 2 * step)), width)?;
                    let second = Conv1d::new(
                        binder.normalized(&at(&format!("convs2.{step}.weight")))?,
                        Some(binder.region(&at(&format!("convs2.{step}.bias")), &[width])?),
                        geometry(block.kernel, 1, Step::padding_plain(block.kernel))?,
                        false,
                    )?;
                    let second_activation =
                        binder.activation(&at(&format!("activations.{}", 2 * step + 1)), width)?;
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
        let post = binder.activation("post", u64::from(last.channels_out))?;
        let output = Conv1d::new(
            binder.normalized("output.weight")?,
            None,
            geometry(7, 1, 3)?,
            false,
        )?;
        binder.finish()?;
        Ok(Self {
            config: config.clone(),
            denormalize,
            input,
            pre,
            stages: built,
            post,
            output,
            pcm: PcmInterleave::new(DType::F32)?,
            track: nn::layers::CopyBlock::new(DType::F32)?,
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
        // only after the fold: the statistics are one one-by-one convolution of constants instead of a
        // fold into that projection.
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
        // first `ScaledAdd` writes `running` from the stage's input with `beta = 0`), while the running
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
            ScaledAdd.forward(f, rows(accumulated)?, rows(block_out)?, 1.0, beta)?;
        }
        // The reference divides the sum by the number of kernels; multiplying by the reciprocal differs
        // only in the last bit of a FP32 value, three orders below one PCM step.
        ScaledAdd.forward(
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
        // `beta = 0` initialises the running value from the stage's input without touching it. Every
        // block does this first, which is what makes one set of buffers reusable across the stage.
        ScaledAdd.forward(f, rows(running)?, rows(input)?, 1.0, 0.0)?;
        for step in &block.steps {
            step.first_activation.forward(f, running, activated)?;
            step.first.forward(f, activated, convolved)?;
            step.second_activation.forward(f, convolved, activated)?;
            step.second.forward(f, activated, residual)?;
            ScaledAdd.forward(f, rows(running)?, rows(residual)?, 1.0, 1.0)?;
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

/// The statistics as the diagonal and the shift of the decoder's restoring convolution.
///
/// `value · std + mean` per latent channel is a one-by-one convolution whose weight is the diagonal of
/// the deviations and whose bias is the means, so the restore is one operator of the same shape as the
/// projection that follows it. The diagonal is `[channels, channels, 1]` in the convolution's
/// `[out, in, kernel]` order.
pub fn denormalize_tables(statistics: &Statistics) -> Result<(Vec<f32>, Vec<f32>)> {
    let channels = statistics.mean.len();
    if channels == 0 || statistics.std.len() != channels {
        return Err(Error::config(format!(
            "the latent statistics hold {} means and {} deviations",
            statistics.mean.len(),
            statistics.std.len()
        )));
    }
    let mut scale = vec![0.0_f32; channels * channels];
    for channel in 0..channels {
        scale[channel * channels + channel] = statistics.std[channel];
    }
    Ok((scale, statistics.mean.clone()))
}

/// A constant weight of `values` in FP32, published once at load.
fn constants(
    loader: &mut nn::WeightLoader<'_>,
    name: &str,
    shape: Shape,
    values: &[f32],
) -> Result<Weight> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    Ok(loader.constant(name, DType::F32, shape, &bytes, "minimax-h3")?)
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
        out: &mut [Vec<u8>],
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

/// The parameters of one load, with every table entry tracked so none is skipped.
pub(crate) struct Binder<'a, 'b> {
    loader: &'a mut WeightLoader<'b>,
    table: Vec<Source>,
    used: Vec<bool>,
}

impl<'a, 'b> Binder<'a, 'b> {
    pub(crate) fn new(loader: &'a mut WeightLoader<'b>, table: Vec<Source>) -> Self {
        let used = vec![false; table.len()];
        Self {
            loader,
            table,
            used,
        }
    }

    /// The table entry of `parameter`, marked used.
    fn entry(&mut self, parameter: &str) -> Result<Tensor> {
        let index = self
            .table
            .iter()
            .position(|source| source.parameter == parameter)
            .ok_or_else(|| {
                Error::checkpoint(format!(
                    "{}: {}",
                    parameter.to_owned(),
                    "the checkpoint's audio decoder has no such parameter".to_owned()
                ))
            })?;
        if self.used[index] {
            return Err(Error::checkpoint(format!(
                "{}: {}",
                parameter.to_owned(),
                "the parameter is bound twice".to_owned()
            )));
        }
        self.used[index] = true;
        Ok(self.table[index].tensor.clone())
    }

    /// A weight-normalized convolution's folded weight.
    pub(crate) fn normalized(&mut self, parameter: &str) -> Result<Weight> {
        match self.entry(parameter)? {
            Tensor::Normalized { value, gain } => {
                let weight = self.loader.weight_norm(parameter, &value, &gain)?;
                self.declared(weight, parameter)
            }
            other => Err(Error::checkpoint(format!(
                "{parameter}: the parameter is not weight-normalized: {other:?}"
            ))),
        }
    }

    /// A weight as the audio VAE's declared arithmetic requires it: FP32.
    ///
    /// The two audio VAEs compute in FP32 - the alias-free resample's Kaiser-sinc filters, the SnakeBeta
    /// exponentials and the seven stages' transposed convolutions are all stated in it - and the loader
    /// converts nothing by default, so a checkpoint whose audio weights are stored in a narrower type
    /// would reach an operator whose activations are FP32 with weights that are not. That mismatch is
    /// refused here, at the loading boundary, naming the parameter and the representation it is in.
    fn declared(&self, weight: Weight, parameter: &str) -> Result<Weight> {
        if weight.dtype() != DType::F32 {
            return Err(Error::checkpoint(format!(
                "{parameter}: the audio VAE computes in FP32 and this weight is {}; the checkpoint's \
                 audio weights must be converted at load rather than mixed with FP32 activations",
                weight.dtype()
            )));
        }
        Ok(weight)
    }

    /// One tensor, read at `shape`.
    pub(crate) fn region(&mut self, parameter: &str, shape: &[u64]) -> Result<Weight> {
        match self.entry(parameter)? {
            Tensor::Region(tensor) => {
                let weight =
                    self.loader
                        .load(&WeightRequest::region(tensor, 0, Shape::new(shape)))?;
                self.declared(weight, parameter)
            }
            other => Err(Error::checkpoint(format!(
                "{parameter}: the parameter is not one tensor: {other:?}"
            ))),
        }
    }

    /// `shape` elements of one tensor, starting `offset` elements into it.
    ///
    /// A row slice of a fused matrix is contiguous, which is what splits the audio encoder's query-key-
    /// value projection into three matrices: the checkpoint stores them fused by rows, and each one is
    /// then read on its own by its own linear layer.
    ///
    /// **One entry may be sliced any number of times**, which is why this does not go through
    /// [`Self::entry`]: that guard refuses a parameter bound twice, and a fused matrix is bound once
    /// and read in three windows. What the guard protects is a parameter read twice *whole* - two
    /// layers sharing one checkpoint tensor - and [`Self::region`] and [`Self::normalized`] still
    /// refuse it.
    pub(crate) fn slice(&mut self, parameter: &str, offset: u64, shape: &[u64]) -> Result<Weight> {
        let index = self
            .table
            .iter()
            .position(|source| source.parameter == parameter)
            .ok_or_else(|| {
                Error::checkpoint(format!(
                    "{}: {}",
                    parameter.to_owned(),
                    "the checkpoint's audio decoder has no such parameter".to_owned()
                ))
            })?;
        self.used[index] = true;
        match self.table[index].tensor.clone() {
            Tensor::Region(tensor) => {
                let weight =
                    self.loader
                        .load(&WeightRequest::region(tensor, offset, Shape::new(shape)))?;
                self.declared(weight, parameter)
            }
            other => Err(Error::checkpoint(format!(
                "{parameter}: the parameter is not one tensor: {other:?}"
            ))),
        }
    }

    /// One alias-free activation: its two frequencies and its two filters.
    fn activation(&mut self, parameter: &str, channels: u64) -> Result<AudioActivation> {
        let alpha = self.region(&format!("{parameter}.act.alpha"), &[channels])?;
        let beta = self.region(&format!("{parameter}.act.beta"), &[channels])?;
        let up = self.region(
            &format!("{parameter}.upsample.filter"),
            &[1, 1, u64::from(ACTIVATION_KERNEL)],
        )?;
        let down = self.region(
            &format!("{parameter}.downsample.lowpass.filter"),
            &[1, 1, u64::from(ACTIVATION_KERNEL)],
        )?;
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

    /// Fail if the table names a parameter the load did not read.
    pub(crate) fn finish(self) -> Result<()> {
        match self.table.iter().zip(&self.used).find(|(_, used)| !**used) {
            Some((source, _)) => Err(Error::checkpoint(format!(
                "{}: the checkpoint's parameter is never bound",
                source.parameter
            ))),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {

    /// A weight the audio VAE cannot compute in is refused at the load, by name.
    ///
    /// Both audio halves compute in FP32 - the alias-free resample's Kaiser-sinc filters, the
    /// SnakeBeta exponentials and the seven stages' transposed convolutions are stated in it - and
    /// the loader converts nothing by default. A checkpoint whose audio weights are stored narrower
    /// would therefore reach an FP32 operator with weights that are not, and the refusal at the
    /// loading boundary is the whole of what stops it: without this, the mismatch appears at the
    /// kernel that reads one type and is handed another, if the kernel checks at all.
    ///
    /// A meta loader answers each request at the type the checkpoint's header states, so a
    /// checkpoint holding the first weight the load reads, stored as F16, is enough to reach the
    /// refusal.
    #[test]
    fn a_weight_the_audio_vae_cannot_compute_in_is_refused_by_name() {
        let config = audio_config();
        let input = sources(&config)
            .expect("the decoder's sources")
            .into_iter()
            .find(|source| source.parameter == "input.weight")
            .expect("the input projection");
        let Tensor::Region(tensor) = input.tensor else {
            panic!("the input projection is one region");
        };
        let directory = tempfile::tempdir().expect("a temp directory");
        let checkpoint = crate::video_vae::tests::header_checkpoint(
            directory.path(),
            &[(
                tensor,
                DType::F16,
                vec![
                    u64::from(config.latent_dim),
                    u64::from(config.latent_channels),
                    1,
                ],
            )],
        );
        let mut loader = WeightLoader::meta(&checkpoint, None, None);
        // `BigVgan` is not `Debug`, so the outcome is matched rather than unwrapped.
        let error = match BigVgan::load(&config, &mut loader) {
            Ok(_) => panic!("an F16 audio weight was accepted"),
            Err(error) => error,
        };
        let message = error.to_string();
        assert!(
            message.contains("must be converted at load"),
            "the refusal names the type the audio VAE cannot compute in: {message}"
        );
    }
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
    fn the_restoring_convolution_is_the_diagonal_of_the_deviations() {
        let config = crate::audio_vae::tests::audio_config();
        let statistics = Statistics::of(&config).expect("the statistics");
        let channels = statistics.mean.len();
        let (scale, shift) = denormalize_tables(&statistics).expect("the tables");
        // A one-by-one convolution's weight is `[out, in, 1]`: the deviations on the diagonal and zero
        // everywhere else, so each channel is scaled by its own deviation and by nothing else.
        assert_eq!(scale.len(), channels * channels);
        for out in 0..channels {
            for input in 0..channels {
                let want = if out == input {
                    statistics.std[input]
                } else {
                    0.0
                };
                assert_eq!(scale[out * channels + input], want);
            }
        }
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

    #[test]
    fn the_weight_table_names_every_parameter_of_the_checkpoint() {
        let config = audio_config();
        let table = sources(&config).expect("the decoder's parameters");
        let normalized = table
            .iter()
            .filter(|source| matches!(source.tensor, Tensor::Normalized { .. }))
            .count();
        // The same count by structure: the pre- and post-convolutions plus every stage's upsampler
        // and its blocks' convolutions. The input projection is not among them.
        let stages = config.decoder_rates.len();
        let kernels = config.resblock_kernel_sizes.len();
        let normalized_by_structure = 2 + stages * (1 + BLOCKS_PER_STAGE * kernels * 2);
        assert_eq!(normalized, normalized_by_structure);
        // The table's total is not asserted here: it would be a restatement of the same structure.
        // What matters beyond the count is that the table names the checkpoint's parameters in the
        // right representation - and the pair rule below is that check, stated directly.
        let names: Vec<&str> = table
            .iter()
            .map(|source| source.parameter.as_str())
            .collect();
        assert!(names.contains(&"input.weight"));
        assert!(names.contains(&"pre.weight"));
        assert!(names.contains(&"stages.6.upsample.weight"));
        assert!(names.contains(&"stages.0.blocks.2.convs1.2.weight"));
        assert!(names.contains(&"stages.6.blocks.2.activations.5.act.alpha"));
        assert!(names.contains(&"post.downsample.lowpass.filter"));
        assert!(names.contains(&"output.weight"));
        // No parameter repeats: the stack is named once.
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), names.len());
        // A weight-normalized convolution names both directions and one gain.
        let pre = table
            .iter()
            .find(|source| source.parameter == "pre.weight")
            .expect("the pre-convolution");
        assert_eq!(
            pre.tensor,
            Tensor::Normalized {
                value: "decoder.conv_pre.weight_v".to_owned(),
                gain: "decoder.conv_pre.weight_g".to_owned(),
            }
        );
    }
}
