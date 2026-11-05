//! H3's audio VAE decoder: BigVGAN's stage arithmetic, its alias-free activation, and the stereo PCM
//! it ends in.
//!
//! `uniserve/models/minimax_h3/audio_vae.py` and diffusers'
//! `autoencoders/autoencoder_kl_minimax_h3_audio.py` are the reference. The decoder is a
//! BigVGAN: a 1×1 input projection to `latent_dim`, a kernel-7 pre-convolution, seven transposed-conv
//! upsampling stages (each followed by three anti-aliased multi-periodicity blocks whose outputs are
//! summed and averaged), one alias-free activation, and a kernel-7 post-convolution to mono, clamped
//! to [-1, 1].
//!
//! ## What this module owns
//!
//! * the stage table a deployment builds from [`AudioVaeConfig`]: every convolution's channel pair,
//!   rate, kernel and padding, so the layer stack and its weight names are derived, never transcribed;
//! * the alias-free activation's length arithmetic ([`Resample`], [`Activation`]) — the two replicate
//!   pads, the crop after the transposed convolution and the stride, which together map `length` to
//!   `length · ratio` and back to `length`;
//! * [`receptive_field`]: the latent frames of context one decoded sample needs on each side, which is
//!   what the runner carries as a halo around a media window;
//! * [`Pcm::quantize`]: the exact arithmetic of the `pcm_interleave` program, stated here so the
//!   operator's implementation has one reference.
//!
//! ## Precision
//!
//! The audio decoder is FP32 throughout (specs/worker.md 9; `precision.py` keeps it unquantized and
//! unconverted). Its 135 weight-normalized convolutions are folded once at the loading boundary
//! (`nn::WeightLoader::weight_norm`), so no layer carries the magnitude/direction pair, and its
//! convolutions run through cuDNN without TF32: spec 9 keeps TF32 for the video VAE's FP32 `conv3d`
//! alone.

use crate::programs::Programs;
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

/// The alias-free resampler's geometry: one direction of the activation's two passes.
///
/// The upsample replicate-pads by `pad` on both sides, runs a transposed depthwise convolution of
/// stride `ratio`, and crops `pad_left` samples from the front and `pad_right` from the end; the
/// downsample replicate-pads the same way and runs a depthwise convolution of stride `ratio`. Both
/// formulas are the reference's, and the operator refuses an output length they do not produce.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Resample {
    /// The upsampling factor; 2 in the checkpoint.
    pub ratio: u32,
    /// The Kaiser-sinc filter's length; 12 in the checkpoint.
    pub kernel: u32,
    /// Samples replicate-padded on each side before the convolution.
    pub pad: u32,
    /// Samples cropped from the front after the transposed convolution.
    pub pad_left: u32,
    /// Samples cropped from the end after the transposed convolution.
    pub pad_right: u32,
}

impl Resample {
    /// The geometry of ratio `ratio` with a `kernel`-long filter.
    ///
    /// The reference states the three paddings once, for the upsampling direction; the downsampling
    /// direction pads its own convolution's input, which is why [`Resample::upsample_length`] and
    /// [`Resample::downsample_padding`] are separate.
    pub fn new(ratio: u32, kernel: u32) -> Result<Self> {
        if ratio == 0 || kernel == 0 {
            return Err(Error::config(
                "a resampler needs a positive ratio and kernel".to_owned(),
            ));
        }
        if kernel < ratio || !kernel.is_multiple_of(ratio) {
            return Err(Error::config(format!(
                "a {kernel}-long filter does not divide a rate of {ratio}"
            )));
        }
        Ok(Self {
            ratio,
            kernel,
            pad: kernel / ratio - 1,
            pad_left: (kernel / ratio - 1) * ratio + (kernel - ratio) / 2,
            pad_right: (kernel / ratio - 1) * ratio + (kernel - ratio).div_ceil(2),
        })
    }

    /// The output length of the upsampling direction for `length` input samples.
    ///
    /// The transposed convolution of a `length + 2·pad` input, stride `ratio` and kernel `kernel`
    /// emits `(length + 2·pad − 1)·ratio + kernel`, and the crop removes `pad_left + pad_right`; the
    /// two cancel to `length · ratio` for the kernel this constructor admits.
    pub fn upsample_length(&self, length: u32) -> u32 {
        (u64::from(length) * u64::from(self.ratio)) as u32
    }

    /// The samples the upsampling direction's transposed convolution emits before the crop.
    pub fn transposed_length(&self, length: u32) -> u32 {
        (u64::from(length) + 2 * u64::from(self.pad) - 1) as u32 * self.ratio + self.kernel
    }

    /// The padding of the downsampling direction's convolution: `[pad_left, pad_right]`.
    pub fn downsample_padding(&self) -> (u32, u32) {
        let even = self.kernel.is_multiple_of(2);
        let left = self.kernel / 2 - u32::from(even);
        (left, self.kernel / 2)
    }

    /// The output length of the downsampling direction for `length` input samples.
    pub fn downsample_length(&self, length: u32) -> u32 {
        let (left, right) = self.downsample_padding();
        let padded = u64::from(length) + u64::from(left) + u64::from(right);
        // `conv1d` with stride `ratio` and no padding: `⌊(padded − kernel)/ratio⌋ + 1`.
        ((padded - u64::from(self.kernel)) / u64::from(self.ratio) + 1) as u32
    }
}

/// The alias-free activation: upsample, activate, low-pass back down.
///
/// The activation's length is unchanged; the operator that runs it fuses the five passes (pad,
/// transposed depthwise convolution, crop, SnakeBeta, pad, depthwise convolution) into one program
/// (specs/worker.md 9, the BigVGAN row).
///
/// The passes, exactly (`autoencoder_kl_minimax_h3_audio.py`, `MiniMaxH3AudioActivation1d`):
///
/// 1. replicate-pad the track by [`Resample::pad`] on each side;
/// 2. `ratio` times a depthwise transposed convolution of stride `ratio` with the Kaiser-sinc
///    upsampling filter (`cutoff 0.5/ratio`, `half_width 0.6/ratio`) — the `ratio` factor is part of
///    the result, not of the length, and a program that drops it is off by that factor;
/// 3. crop [`Resample::pad_left`] from the front and [`Resample::pad_right`] from the end;
/// 4. the activation itself: `x + (exp(beta) + 1e-9)^-1 * sin(exp(alpha) * x)^2` per channel, with
///    `alpha` and `beta` stored in log space (the DAC encoder's `Snake` is the same formula with a
///    single `alpha` and no exponentiation, which is why the two are not one layer);
/// 5. replicate-pad by [`Resample::downsample_padding`] and run a depthwise convolution of stride
///    `ratio` with the same Kaiser-sinc filter.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Activation {
    /// The upsampling factor of both passes.
    pub ratio: u32,
    /// The Kaiser-sinc filter's length.
    pub kernel: u32,
}

impl Activation {
    /// The activation of `ratio` and `kernel`.
    pub fn new(ratio: u32, kernel: u32) -> Result<Self> {
        Resample::new(ratio, kernel)?;
        Ok(Self { ratio, kernel })
    }

    /// The activation of the checkpoint's decoder: ratio 2, kernel 12.
    pub fn bigvgan() -> Self {
        Self {
            ratio: 2,
            kernel: 12,
        }
    }

    /// The output length for `length` input samples, which is `length` itself.
    ///
    /// Stated as the two passes so that [`Resample`]'s arithmetic is the only place the formulas live.
    pub fn length(&self, length: u32) -> Result<u32> {
        let up = Resample::new(self.ratio, self.kernel)?;
        Ok(up.downsample_length(up.upsample_length(length)))
    }
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

/// The latent frames of context one decoded sample depends on, per side.
///
/// This is the reference `receptive_field`: it walks the decoder in forward order, tracking how far
/// one output sample's dependency spreads, with the residual branches of a stage running in parallel
/// (so a stage contributes its widest branch, not their sum) and each alias-free activation spreading
/// its own upsample, filter and low-pass. The result rounds up and may exceed the exact field, which
/// costs context and never correctness — it is the halo the runner carries around a media window.
pub fn receptive_field(config: &AudioVaeConfig) -> Result<u32> {
    let stages = stages(config)?;
    if config.resblock_kernel_sizes.len() != config.resblock_dilation_sizes.len() {
        return Err(Error::config(format!(
            "the audio decoder has {} residual kernels and {} dilation lists",
            config.resblock_kernel_sizes.len(),
            config.resblock_dilation_sizes.len()
        )));
    }
    let activation = Activation::bigvgan();
    let mut spread = 0.0f64;
    let mut rate = 1.0f64;

    // A convolution's spread is `(kernel − 1) · dilation / rate` output samples at the current rate.
    let convolution = |kernel: u32, dilation: u32, rate: f64| -> f64 {
        let width = f64::from(kernel) - 1.0;
        width * f64::from(dilation) / rate
    };
    // One alias-free activation spreads its upsampled convolution and its low-pass, at the current
    // rate: the reference computes the first pass with `ceil` because the crop only takes whole
    // samples.
    let activation_spread = |rate: f64| -> f64 {
        let up = (f64::from(activation.kernel) / f64::from(activation.ratio)).ceil() - 1.0;
        up / rate + (f64::from(activation.kernel) - 1.0) / (rate * f64::from(activation.ratio))
    };

    spread += convolution(7, 1, rate);
    for stage in &stages {
        spread += (f64::from(stage.kernel) / f64::from(stage.rate)).ceil() / rate - 1.0 / rate;
        rate *= f64::from(stage.rate);
        let mut widest = 0.0f64;
        for (kernel, dilations) in config
            .resblock_kernel_sizes
            .iter()
            .zip(&config.resblock_dilation_sizes)
        {
            let mut branch = 0.0f64;
            for &dilation in dilations {
                branch += activation_spread(rate) + convolution(*kernel, dilation, rate);
                branch += activation_spread(rate) + convolution(*kernel, 1, rate);
            }
            widest = widest.max(branch);
        }
        spread += widest;
    }
    spread += activation_spread(rate) + convolution(7, 1, rate);

    // The spread covers both sides of one dependency, so one side is half, rounded up.
    Ok((spread / 2.0).ceil() as u32)
}

/// The arithmetic of the stereo PCM the audio decoder ends in.
///
/// The reference converts `[stereo, samples]` float waveforms to `[samples, 2]` `int16`:
/// `clamp(x, −1, 1) · 32767`, `torch.round` (half to even), then the integer cast. `pcm_interleave`
/// states the same thing on the device; this is the one reference an implementation is checked
/// against.
pub struct Pcm;

impl Pcm {
    /// The full-scale sample value: the scale the reference multiplies by.
    pub const FULL_SCALE: f32 = 32767.0;

    /// One sample's PCM value.
    ///
    /// `x` is clamped to [-1, 1] first, so the result never leaves `[-32767, 32767]` and the cast is
    /// total.
    pub fn quantize(x: f32) -> i16 {
        let clamped = x.clamp(-1.0, 1.0) * Self::FULL_SCALE;
        // `f32::round_ties_even` is `torch.round`'s rule; a value that is not finite cannot leave the
        // clamp, so the cast is defined.
        clamped.round_ties_even() as i16
    }
}

/// Every buffer one audio decode takes from `Forward::arena()`, in reservation order.
///
/// The decoder's `forward` reserves: the restoring and projecting pair, the input convolution's output,
/// then per stage the upsampled signal, the running sum, and **one** set of four buffers the stage's three
/// blocks take turns in (the block's running value, its two convolution outputs and the second's residual).
/// The post activation's output and the mono signal the PCM step reads close the list. `frames` is the
/// request's latent frame count, so every length grows by the rates.
///
/// The list matters because the arena never rewinds - its only rewind is the executor's per-forward
/// `reset` - so every reservation in this walk accumulates and the sum is what a request needs. It is
/// therefore large: at 207 latent frames each buffer of stages one to six is `[2, c, L]` with `c · L` near
/// 1.32 M, about 10.6 MB, so those six stages carry most of the total against 0.12 GiB of weights. The
/// remaining reduction is the shape-family one - `c · L` is within a factor of the patch area across the
/// stages, so a set sized for the largest and viewed per stage would serve all seven - and it would make
/// the plan a *bound* rather than a sum of distinct buffers. The stage loop states one shape per stage
/// today, so this list states what is actually reserved.
pub fn audio_arena(
    config: &crate::AudioVaeConfig,
    frames: u32,
) -> Result<Vec<crate::decoding::Buffer>> {
    let mut plan = Vec::new();
    let mut push = |name: String, channels: u64, length: u64| {
        plan.push(crate::decoding::Buffer {
            name,
            element: "f32",
            elements: u64::from(PCM_CHANNELS) * channels * length,
        });
    };
    let mut length = u64::from(frames);
    push(
        "restored".to_owned(),
        u64::from(config.latent_channels),
        length,
    );
    push("projected".to_owned(), u64::from(config.latent_dim), length);
    push("input".to_owned(), u64::from(config.decoder_dim), length);
    // The stage list the decoder itself builds, so the plan cannot disagree with the code about a
    // channel count or a rate: both read this one derivation.
    let stages = stages(config)?;
    let mut channels = u64::from(config.decoder_dim);
    for (index, stage) in stages.iter().enumerate() {
        channels = u64::from(stage.channels_out);
        length = length
            .checked_mul(u64::from(stage.rate))
            .ok_or_else(|| Error::config("an audio length overflows".to_owned()))?;
        push(format!("stage{index}.upsampled"), channels, length);
        push(format!("stage{index}.sum"), channels, length);
        // One set for the stage's blocks, not one per block: a block overwrites all four before it
        // reads them.
        for part in ["running", "activated", "convolved", "residual"] {
            push(format!("stage{index}.{part}"), channels, length);
        }
    }
    push("post".to_owned(), channels, length);
    push("mono".to_owned(), 1, length);
    Ok(plan)
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

/// Every checkpoint tensor a source list reads, in the order the load binds them.
///
/// One parameter can read more than one tensor: a weight-normalized convolution reads its direction
/// and its magnitude, and a fused request reads every part. This is the other column of a component's
/// source table rather than a second spelling of it, so a `keys` list built here cannot drift from the
/// load that table drives.
pub fn tensor_keys(sources: &[Source]) -> Vec<String> {
    let mut keys = Vec::new();
    for source in sources {
        match &source.tensor {
            Tensor::Region(tensor) => keys.push(tensor.clone()),
            Tensor::Normalized { value, gain } => {
                keys.push(value.clone());
                keys.push(gain.clone());
            }
        }
    }
    keys
}

/// How the checkpoint stores one tensor.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Tensor {
    /// One tensor under one key.
    Region(String),
    /// A weight-normalized convolution: `value` holds the weight's extents and `gain` one magnitude
    /// per output channel. `nn::WeightLoader::weight_norm` folds the pair once at load, so no layer
    /// below carries it (specs/worker.md 9, the BigVGAN row).
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

/// One buffer of the decoder's forward: the samples and channels it holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Workspace {
    /// The stage the buffer belongs to: 0 is the input projection and the pre-convolution, `i + 1` is
    /// the output of stage `i`, and the last entry is the decoder's single output channel.
    pub stage: u32,
    /// Samples the buffer holds.
    pub samples: u32,
    /// Channels the buffer holds.
    pub channels: u32,
}

impl Workspace {
    /// The elements the buffer holds.
    pub fn elements(&self) -> u64 {
        u64::from(self.samples) * u64::from(self.channels)
    }
}

/// The buffers one decode of `latent_frames` needs, in the order the stages run.
///
/// The reference decodes a track's whole latent timeline at once, so the runner sizes these from the
/// request; a window shorter than the track carries the halo [`receptive_field`] states on each side.
pub fn workspace(config: &AudioVaeConfig, latent_frames: u32) -> Result<Vec<Workspace>> {
    if latent_frames == 0 {
        return Err(Error::config(
            "an audio decode of no latent frame has no workspace".to_owned(),
        ));
    }
    let stages = stages(config)?;
    let mut buffers = vec![Workspace {
        stage: 0,
        samples: latent_frames,
        channels: config.latent_dim,
    }];
    let mut samples = latent_frames;
    for (index, stage) in stages.iter().enumerate() {
        samples = samples
            .checked_mul(stage.rate)
            .ok_or_else(|| Error::config("the audio track's length overflows".to_owned()))?;
        buffers.push(Workspace {
            stage: index as u32 + 1,
            samples,
            channels: stage.channels_out,
        });
    }
    buffers.push(Workspace {
        stage: stages.len() as u32 + 1,
        samples,
        channels: 1,
    });
    Ok(buffers)
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

    /// Restore one channel's value: `value * std + mean`.
    ///
    /// The reference of the affine the decoder's input stage applies; a kernel that folds it into the
    /// input projection's epilogue must agree with it.
    pub fn denormalize(&self, channel: usize, value: f32) -> Result<f32> {
        let mean = *self.mean.get(channel).ok_or_else(|| {
            Error::config(format!(
                "channel {channel} of {} audio latent channels",
                self.mean.len()
            ))
        })?;
        let std = self.std[channel];
        Ok(value * std + mean)
    }
}

/// The audio decoder: the checkpoint's BigVGAN network, one call per run of latent frames.
///
/// Layout everywhere is `[batch, channels, samples]`, the form the one-dimensional convolutions take.
/// The batch is two because the reference decodes a stereo pair of identical rows and interleaves them;
/// nothing in the network mixes the rows, so both output channels carry the same signal.
pub struct AudioDecoder {
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
/// The launcher passes this to `AudioActivation::new`, so it is what the walk records and what the
/// deployment list must hold. It is public because the fixture generator states the rows a store is
/// built from, and a second copy of this number in that generator is how the file came to say 1024
/// for seven rows the device launches at 256.
pub const ACTIVATION_BLOCK: u32 = 256;
/// The two channels the reference decodes and interleaves.
const PCM_CHANNELS: u32 = 2;

/// Every parameter name the audio decoder's load binds, in the order it binds them.
///
/// One source of truth for the load and its check: [`AudioDecoder::load`] binds exactly these names, and
/// [`sources`] must name exactly the same set — a name in one and not the other is a load failure rather
/// than a checkpoint field silently dropped or a tensor bound to the wrong layer.
pub fn parameters(config: &AudioVaeConfig) -> Result<Vec<String>> {
    let stages = stages(config)?;
    let blocks = blocks(config)?;
    let mut names = vec![
        "input.weight".to_owned(),
        "input.bias".to_owned(),
        "pre.weight".to_owned(),
        "pre.bias".to_owned(),
    ];
    let activation = |names: &mut Vec<String>, parameter: &str| {
        for field in [
            "act.alpha",
            "act.beta",
            "upsample.filter",
            "downsample.lowpass.filter",
        ] {
            names.push(format!("{parameter}.{field}"));
        }
    };
    for (index, _stage) in stages.iter().enumerate() {
        names.push(format!("stages.{index}.upsample.weight"));
        names.push(format!("stages.{index}.upsample.bias"));
        for block in blocks.iter().filter(|block| block.stage == index) {
            let at = |name: &str| format!("stages.{index}.blocks.{}.{name}", block.index);
            for step in 0..block.steps.len() {
                names.push(at(&format!("convs1.{step}.weight")));
                names.push(at(&format!("convs1.{step}.bias")));
                names.push(at(&format!("convs2.{step}.weight")));
                names.push(at(&format!("convs2.{step}.bias")));
                for half in 0..2 {
                    activation(&mut names, &at(&format!("activations.{}", 2 * step + half)));
                }
            }
        }
    }
    activation(&mut names, "post");
    names.push("output.weight".to_owned());
    Ok(names)
}

impl AudioDecoder {
    /// Every checkpoint tensor the audio decoder reads, by name.
    ///
    /// The decoder is a component whose names were already data ([`sources`] maps each parameter to
    /// the key that fills it), so this list is that mapping's other column: a weight-normalized pair
    /// names both tensors and a fused request names every part. A checkpoint that lacks one is refused
    /// by block and name before a byte is read. `audio_vae/` ships no weights in the export this tree
    /// was checked against, so the list's honest verdict against that checkpoint is "all absent" -
    /// which is the right refusal for a task that needs the audio path and not a load failure from
    /// inside a layer.
    pub fn keys(config: &AudioVaeConfig) -> Result<Vec<String>> {
        Ok(tensor_keys(&sources(config)?))
    }

    /// Bind every parameter the checkpoint's audio decoder has, through [`sources`].
    ///
    /// The table is consumed exactly: a tensor it names but the load never reads is an error, so a
    /// checkpoint field cannot be silently dropped, and a parameter read twice is an error too.
    pub fn load(config: &AudioVaeConfig, loader: &mut WeightLoader) -> Result<Self> {
        let table = sources(config)?;
        let bound = parameters(config)?;
        check_table(&table, &bound)?;
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
                        geometry(block.kernel, 1, dilation.padding)?,
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
        })
    }

    /// Decode `latents [2, latent_channels, frames]` into interleaved PCM `[samples, 2]`.
    ///
    /// The two rows of the latent are the stereo pair the reference decodes; both hold the same latent,
    /// so the network's two output rows are equal and [`PcmInterleave`] writes them interleaved. Every
    /// intermediate is FP32 — `specs/worker.md` 9's audio rows keep no reduced precision — and the
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

    /// The programs one decode of `config` launches, with no weights and no device.
    ///
    /// Every launch goes through the same construction the layer uses: `nn`'s `descriptor_of`
    /// accessors for the convolutions, the activation and the interleave, and the model's own
    /// config arithmetic ([`stages`], [`blocks`]) for the shapes they are built at.
    ///
    /// The per-row programs — the residual accumulation and the stage's divide — carry their width in
    /// the descriptor, so they are stated once per stage width and per sample count the deployment
    /// serves (`buckets`). The per-channel ones do not: a convolution's key is its geometry, and a
    /// clip of another length replays the compiled program.
    pub fn programs(config: &AudioVaeConfig, buckets: &[u32]) -> Result<Vec<(String, String)>> {
        use ops::audio_activation::AudioActivation as AudioActivationOp;
        use ops::conv::{Conv1d as Conv1dOp, ConvTranspose1d as ConvTranspose1dOp};
        use ops::pcm::PcmInterleave as PcmInterleaveOp;
        use ops::residual::ScaledAdd as ScaledAddOp;

        let mut programs = Programs::default();
        // The decoder computes in FP32 and its convolutions' weights are normalized at load, so every
        // convolution below reads and writes that one type.
        let dtype = DType::F32;
        let decoder_dim = config.decoder_dim;
        // The channel pairs are the weight's, not the key's: a convolution's descriptor is its
        // geometry and its element types, so the three opening convolutions are their three kernels.
        for (kernel, stride, padding) in [
            // The statistics as a one-by-one convolution, the projection into the latent width, and
            // the pre-convolution of kernel seven.
            (1, 1, 0),
            (1, 1, 0),
            (7, 1, 3),
        ] {
            programs.launch::<Conv1dOp>(&Conv1d::descriptor_of(
                dtype,
                dtype,
                true,
                dtype,
                geometry(kernel, stride, padding)?,
                false,
            ));
        }

        let stages = stages(config)?;
        for stage in &stages {
            programs.launch::<ConvTranspose1dOp>(&ConvTranspose1d::descriptor_of(
                dtype,
                dtype,
                true,
                dtype,
                geometry(stage.kernel, stage.rate, stage.padding)?,
                false,
            ));
        }

        for block in &blocks(config)? {
            let channels = block.channels;
            for step in &block.steps {
                // A step is an activation, a convolution, an activation and a convolution.
                for padding in [step.padding, (block.kernel - 1) / 2] {
                    programs.launch::<AudioActivationOp>(&AudioActivation::descriptor_of(
                        dtype,
                        channels,
                        ACTIVATION_KERNEL,
                        ACTIVATION_RATIO,
                        ACTIVATION_BLOCK,
                    ));
                    programs.launch::<Conv1dOp>(&Conv1d::descriptor_of(
                        dtype,
                        dtype,
                        true,
                        dtype,
                        geometry(block.kernel, 1, padding)?,
                        false,
                    ));
                }
            }
        }

        let last = stages
            .last()
            .map_or(decoder_dim, |stage| stage.channels_out);
        programs.launch::<AudioActivationOp>(&AudioActivation::descriptor_of(
            dtype,
            last,
            ACTIVATION_KERNEL,
            ACTIVATION_RATIO,
            ACTIVATION_BLOCK,
        ));
        programs.launch::<Conv1dOp>(&Conv1d::descriptor_of(
            dtype,
            dtype,
            true,
            dtype,
            geometry(1, 1, 0)?,
            false,
        ));
        programs.launch::<PcmInterleaveOp>(&PcmInterleave::descriptor_of(dtype)?);

        // The residual add's width is the **sample count at that stage**, not the channel count:
        // the stage's buffers are `[PCM_CHANNELS, channels, length]` and the add reshapes them to
        // `[PCM_CHANNELS · channels, length]` (`rows`), so the descriptor's `columns` is the length
        // the stage's upsampling leaves. Stating `channels × bucket` named keys no launch asks for.
        for bucket in buckets {
            let mut length = u64::from(*bucket);
            for stage in &stages {
                length *= u64::from(stage.rate);
                programs.launch::<ScaledAddOp>(&ScaledAdd.descriptor(dtype, dtype, length)?);
            }
        }
        Ok(programs.finish())
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

/// The table and the load's own list must name the same parameters.
fn check_table(table: &[Source], bound: &[String]) -> Result<()> {
    for name in bound {
        if !table.iter().any(|source| source.parameter == *name) {
            return Err(Error::checkpoint(format!(
                "{name}: the audio decoder binds a parameter its table does not name"
            )));
        }
    }
    for source in table {
        if !bound.contains(&source.parameter) {
            return Err(Error::checkpoint(format!(
                "{}: the table names a parameter the audio decoder does not bind",
                source.parameter
            )));
        }
    }
    Ok(())
}

impl Decoder for AudioDecoder {
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
                owner: "AudioDecoder",
                problem: format!(
                    "the audio decoder has no layout tables; {} buffers were offered",
                    out.len()
                ),
            })
        }
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
                owner: "AudioDecoder",
                problem: format!(
                    "the audio decoder reads no layout table, {} were given",
                    state.len()
                ),
            });
        }
        let audio = out.audio.ok_or_else(|| nn::Error::Shape {
            owner: "AudioDecoder",
            problem: "an audio decode writes the request's PCM window".to_owned(),
        })?;
        if out.video.is_some() {
            return Err(nn::Error::Shape {
                owner: "AudioDecoder",
                problem: "the audio decoder writes no video raster".to_owned(),
            });
        }
        self.forward(f, latents, audio)
            .map_err(|error| nn::Error::Shape {
                owner: "AudioDecoder",
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
    Ok(ConvGeometry::new(
        1,
        [kernel; 3],
        [stride; 3],
        [padding; 3],
        [dilation; 3],
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

    /// The audio decoder's names are the checkpoint's, tensor for tensor.
    ///
    /// A mapping that invents one name and drops another still asks the same count, so a count is not
    /// the check - the set is. This is the decoder half of the same export the encoder's fixture
    /// records: 912 `decoder.*` tensors and the 2 of `dec_in_proj.*`. It is what caught the input
    /// projection being asked as a weight-normalized pair, one name each way, and the same class the
    /// normalized-pair test above cannot see: that test reads the crate's own list.
    #[test]
    fn the_audio_decoders_names_are_the_checkpoints() {
        let config = audio_config();
        let mut ours = AudioDecoder::keys(&config).expect("the decoder's keys");
        ours.sort_unstable();
        let fixture: Vec<&str> = include_str!("../tests/data/audio-decoder-names.txt")
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect();
        assert_eq!(fixture.len(), 914, "the recorded export's decoder half");
        let missing: Vec<&str> = fixture
            .iter()
            .copied()
            .filter(|name| !ours.iter().any(|ours| ours == name))
            .collect();
        let extra: Vec<&str> = ours
            .iter()
            .map(String::as_str)
            .filter(|name| !fixture.contains(name))
            .collect();
        assert!(
            missing.is_empty(),
            "the checkpoint has decoder names this crate does not ask for: {missing:?}"
        );
        assert!(
            extra.is_empty(),
            "this crate asks for decoder names the checkpoint does not have: {extra:?}"
        );
    }

    #[test]
    fn the_audio_decoders_keys_name_both_tensors_of_every_normalized_pair() {
        let config = audio_config();
        let keys = AudioDecoder::keys(&config).expect("the decoder's keys");
        let mut unique = keys.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), keys.len(), "no tensor is named twice");
        // The input projection is the one convolution the reference does not normalize, and the
        // checkpoint therefore holds one weight for it: a list that asked for a pair would name two
        // tensors the export does not have and leave the one it has unread (`M5621`'s measurement:
        // 915 asked against 914 stored, one name each way).
        assert!(keys.iter().any(|key| key == "dec_in_proj.weight"));
        assert!(
            keys.iter()
                .all(|key| !key.starts_with("dec_in_proj.weight_")),
            "the input projection holds no normalized pair"
        );
        // A weight-normalized convolution reads a direction and a magnitude, and the load needs both:
        // a list that named only the direction would refuse a checkpoint that is complete.
        assert!(keys.iter().any(|key| key == "decoder.conv_pre.weight_v"));
        let magnitudes = keys.iter().filter(|key| key.ends_with(".weight_g")).count();
        let directions = keys.iter().filter(|key| key.ends_with(".weight_v")).count();
        assert_eq!(magnitudes, directions, "every magnitude has its direction");
        // One pre-convolution, one post-convolution, and per stage its upsampler plus two per
        // residual kernel in each of the three blocks. The input projection is not among them: the
        // reference declares it plain and the export stores it as one weight.
        let stages = config.decoder_rates.len();
        let kernels = config.resblock_kernel_sizes.len();
        assert_eq!(
            magnitudes,
            2 + stages * (1 + BLOCKS_PER_STAGE * kernels * 2),
            "the decoder's weight-normalized convolutions, by structure rather than by count"
        );
        // The encoder reads a different half of a different directory; a shared name would be one
        // block binding the other's tensor.
        let encoder = crate::encoding::AudioEncoder::keys(&config).expect("the encoder's keys");
        assert!(encoder.iter().all(|key| !keys.contains(key)));
    }

    #[test]
    fn the_load_binds_every_parameter_the_table_names() {
        let config = audio_config();
        let table = sources(&config).expect("the table");
        let bound = parameters(&config).expect("the load's list");
        assert_eq!(bound.len(), table.len());
        for name in &bound {
            assert!(
                table.iter().any(|source| source.parameter == *name),
                "{name} is bound but not in the table"
            );
        }
        // Each of the seven stages' three blocks runs three dilated steps of two convolutions, and each
        // of those six convolutions carries its own alias-free activation. One activation is **four**
        // tensors — the SnakeBeta pair (`act.alpha`, `act.beta`) and the two anti-aliasing filters
        // (`upsample.filter`, `downsample.lowpass.filter`), which `Binder::activation` reads together —
        // so the number of names under `.activations.` is the site count times four, not the site count.
        assert_eq!(
            bound
                .iter()
                .filter(|name| name.contains(".activations."))
                .count(),
            7 * BLOCKS_PER_STAGE * 6 * 4
        );
        // A configuration with one stage fewer keeps the two lists in step: the table and the load's
        // list are checked against each other, not against a count written down here.
        let mut fewer = config.clone();
        fewer.decoder_rates.pop();
        fewer.decoder_kernel_sizes.pop();
        let table = sources(&fewer).expect("six stages' table");
        let bound = parameters(&fewer).expect("six stages' list");
        assert!(bound.len() < parameters(&config).expect("seven stages' list").len());
        check_table(&table, &bound).expect("the two agree with six stages");
    }
    use super::*;

    /// The checkpoint's audio VAE config (`audio_vae/config.json`).
    /// The decoder states its programs from its own configuration, and the statement is a **set**: a
    /// convolution's key carries element types only (its geometry travels in the call structure), so
    /// a kernel-one projection and a kernel-nine filter are one program, while the fused activation
    /// is one per channel count.
    #[test]
    fn the_decoder_states_its_programs_from_its_own_config() {
        let config = audio_config();
        let programs = AudioDecoder::programs(&config, &[64, 256]).expect("the decoder's programs");
        assert!(!programs.is_empty());
        for (operator, key) in &programs {
            assert!(!operator.is_empty() && !key.is_empty(), "{operator} {key}");
        }
        let count = |operator: &str| programs.iter().filter(|(name, _)| name == operator).count();

        // One program for every one-dimensional convolution of the decoder, in either direction: the
        // window, the bias and the stride are launch values.
        assert_eq!(count("conv1d"), 1);
        assert_eq!(count("conv_transpose1d"), 1);
        // The interleave converts two channels whatever the clip's length.
        assert_eq!(count("pcm_interleave"), 1);

        let stages = stages(&config).expect("the stages");
        // The fused activation's key is the channel count, so a decoder has one per stage width plus
        // the post-activation, which repeats the last width and shares its program.
        assert_eq!(count("audio_activation"), stages.len());

        // A row program carries its width, so the stage sums and the divide are per width and per
        // sample count — deduplicated where a width and a count collide.
        let row_programs = count("scaled_add");
        assert!(
            row_programs > 0 && row_programs < 2 * stages.len(),
            "{programs:?}"
        );
        assert_eq!(
            programs.len(),
            count("conv1d")
                + count("conv_transpose1d")
                + count("pcm_interleave")
                + count("audio_activation")
                + row_programs
        );
    }

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
    fn the_audio_decoders_arena_names_every_buffer_it_reserves() {
        let config = tests::audio_config();
        let frames = 207;
        let plan = audio_arena(&config, frames).expect("the plan");
        // Three buffers before the stages (restore, project, input), six per stage (the upsampled signal,
        // the running sum and the four the stage's blocks share), and the post activation and the mono
        // signal.
        assert_eq!(plan.len(), 3 + 6 * config.decoder_rates.len() + 2);
        // The lengths follow the rates, so the last stage's buffers cover `frames · Π rates` samples.
        let longest = stages(&config)
            .expect("stages")
            .iter()
            .try_fold(u64::from(frames), |length, stage| {
                length.checked_mul(u64::from(stage.rate))
            })
            .expect("the rates");
        assert_eq!(longest, u64::from(frames) * 800);
        let mono = plan.last().expect("the mono signal");
        assert_eq!(mono.elements, u64::from(PCM_CHANNELS) * longest);
        // The plan's per-stage channels and lengths are the decoder's own stage list, walked once -
        // the same derivation `forward` reserves from, so a channel count cannot drift between them.
        for (index, stage) in stages(&config).expect("stages").iter().enumerate() {
            let entries: Vec<&crate::decoding::Buffer> = plan
                .iter()
                .filter(|buffer| buffer.name.starts_with(&format!("stage{index}.")))
                .collect();
            assert_eq!(entries.len(), 6, "stage {index} reserves six buffers");
            for buffer in entries {
                // `[2, channels, length]` f32, with the length already multiplied by every rate up to
                // and including this stage's.
                let length = stages(&config)
                    .expect("stages")
                    .iter()
                    .take(index + 1)
                    .try_fold(u64::from(frames), |length, stage| {
                        length.checked_mul(u64::from(stage.rate))
                    })
                    .expect("the rates");
                assert_eq!(
                    buffer.elements,
                    u64::from(PCM_CHANNELS) * u64::from(stage.channels_out) * length,
                    "{}",
                    buffer.name
                );
            }
        }
        // Every buffer is named once, and the total is the sum of what the walk reserves.
        let mut names: Vec<&str> = plan.iter().map(|buffer| buffer.name.as_str()).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count);
        let total: u64 = plan.iter().map(|buffer| buffer.bytes()).sum();
        assert!(
            total > 100 * 1024 * 1024,
            "the walk reserves the stages' buffers: {total} B"
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
    fn the_alias_free_activation_keeps_its_length() {
        let activation = Activation::bigvgan();
        assert_eq!(activation.ratio, 2);
        assert_eq!(activation.kernel, 12);
        for length in [1u32, 2, 7, 40, 1000, 4000] {
            assert_eq!(
                activation.length(length).expect("a length"),
                length,
                "length {length}"
            );
        }
        let resample = Resample::new(2, 12).expect("the checkpoint's resampler");
        assert_eq!(resample.pad, 5);
        assert_eq!(resample.pad_left, 15);
        assert_eq!(resample.pad_right, 15);
        assert_eq!(resample.transposed_length(40), 110);
        assert_eq!(resample.upsample_length(40), 80);
        assert_eq!(resample.downsample_padding(), (5, 6));
        assert_eq!(resample.downsample_length(80), 40);
    }

    #[test]
    fn a_resampler_refuses_a_kernel_its_ratio_does_not_divide() {
        assert!(Resample::new(2, 11).is_err());
        assert!(Resample::new(1, 12).is_ok());
        assert!(Resample::new(0, 12).is_err());
        assert!(Activation::new(3, 4).is_err());
    }

    #[test]
    fn the_receptive_field_is_the_references_thirty_latent_frames() {
        // Computed from the reference `receptive_field`: spread 58.31 output samples, one side 30.
        assert_eq!(receptive_field(&audio_config()).expect("field"), 30);
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
    fn the_latent_statistics_restore_a_channel() {
        let config = audio_config();
        let statistics = Statistics::of(&config).expect("the statistics");
        assert_eq!(statistics.mean.len(), 32);
        assert_eq!(statistics.std.len(), 32);
        // A zero latent restores the channel's mean; a unit latent the mean plus its deviation.
        assert_eq!(
            statistics.denormalize(0, 0.0).expect("a value"),
            statistics.mean[0]
        );
        assert_eq!(
            statistics.denormalize(0, 1.0).expect("a value"),
            statistics.mean[0] + statistics.std[0]
        );
        // A channel the statistics do not hold is refused rather than read past the end.
        assert!(statistics.denormalize(32, 0.0).is_err());
        // A sidecar with the wrong counts, or a zero deviation, is refused at the boundary.
        let mut broken = config.clone();
        broken.latents_mean.pop();
        assert!(Statistics::of(&broken).is_err());
        let mut broken = config.clone();
        broken.latents_std[3] = 0.0;
        assert!(Statistics::of(&broken).is_err());
    }

    #[test]
    fn the_workspace_grows_to_one_stereo_track() {
        let config = audio_config();
        let buffers = workspace(&config, 1).expect("workspace");
        // One latent frame is 800 samples: the projection's own buffer, one per stage and the mono
        // output. The pre-convolution shares the projection's width, so it needs no buffer of its own.
        assert_eq!(buffers.len(), 1 + 7 + 1);
        assert_eq!(buffers[0].samples, 1);
        assert_eq!(buffers[0].channels, 2048);
        let samples: Vec<u32> = buffers.iter().map(|buffer| buffer.samples).collect();
        assert_eq!(samples, vec![1, 5, 25, 50, 100, 200, 400, 800, 800]);
        let channels: Vec<u32> = buffers.iter().map(|buffer| buffer.channels).collect();
        assert_eq!(channels, vec![2048, 512, 256, 128, 64, 32, 16, 8, 1]);
        // A whole second of audio is forty latent frames and 32000 samples.
        let buffers = workspace(&config, 40).expect("workspace");
        assert_eq!(buffers.last().expect("the output").samples, 32000);
        assert_eq!(buffers.last().expect("the output").channels, 1);
        assert!(workspace(&config, 0).is_err());
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
        // The table's total is not asserted here: it would be a fourth restatement of the same
        // structure, and the set is already pinned exactly against the export's own 914 names
        // (`the_audio_decoders_names_are_the_checkpoints`), which is the check a count can only
        // approximate. What the count would catch and the set does not is a table that names the
        // checkpoint's parameters in the wrong representation - and the pair rule below is that
        // check, stated directly.
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

    #[test]
    fn pcm_rounds_half_to_even_and_clamps() {
        assert_eq!(Pcm::quantize(0.0), 0);
        assert_eq!(Pcm::quantize(1.0), 32767);
        assert_eq!(Pcm::quantize(-1.0), -32767);
        assert_eq!(Pcm::quantize(2.0), 32767);
        assert_eq!(Pcm::quantize(-2.0), -32767);
        // 0.5 / 32767 lies halfway between 0 and 1: to even is 0; 1.5 / 32767 rounds to 2.
        assert_eq!(Pcm::quantize(0.5 / Pcm::FULL_SCALE), 0);
        assert_eq!(Pcm::quantize(1.5 / Pcm::FULL_SCALE), 2);
        assert_eq!(Pcm::quantize(2.5 / Pcm::FULL_SCALE), 2);
    }
}
