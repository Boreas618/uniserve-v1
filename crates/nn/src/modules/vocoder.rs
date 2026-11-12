//! The BigVGAN vocoder: latent features to a waveform through transposed-convolution upsampling
//! stages of anti-aliased multi-periodicity blocks.
//!
//! ```text
//! conv_pre                      kernel 7, padding 3
//! for each stage:
//!     ups.<s>.0                 ConvTranspose1d, stride rate, padding (kernel − rate) / 2,
//!                               halving the channels
//!     resblocks.<s · K + k>     one AmpBlock per residual kernel k of K; their outputs summed
//!                               and multiplied by 1 / K
//! activation_post               Activation1d(SnakeBeta)
//! conv_post                     kernel 7, padding 3, to one channel
//! ```
//!
//! [`Activation1d`] is BigVGAN's alias-free activation ([`functional::activation1d`]). Every
//! convolution of the generator is weight-normalized in the checkpoint and folded once at load.
//!
//! **The stage average multiplies by `1 / K`.** The reference divides the sum by the kernel
//! count; the product with the FP32 reciprocal differs from the quotient in the last bit of
//! some values, and the served outputs are this product's, so it is kept. The final `tanh` or
//! clamp is the caller's: a PCM step clamps as it quantizes.

use uniserve_tensor::{DType, Module, Tensor};

use crate::error::{Error, Result};
use crate::functional;
use crate::modules::conv::{Conv1d, ConvConfig, ConvTranspose1d};
use crate::var_builder::VarBuilder;

/// What a checkpoint states of an [`Activation1d`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Activation1dConfig {
    /// The channels of the signal: one SnakeBeta parameter pair each.
    pub channels: u32,
    /// The up- and down-sampling ratio; the filters hold `int(6 · ratio / 2) · 2` taps.
    pub ratio: u32,
}

/// The alias-free activation: SnakeBeta between a `ratio`-times upsample and its low-pass, the
/// reference's `Activation1d(SnakeBeta(channels, alpha_logscale=True))`.
pub struct Activation1d {
    /// `[channels]` SnakeBeta's `alpha`, in log space as the checkpoint stores it.
    alpha: Tensor,
    /// `[channels]` SnakeBeta's `beta`, in log space.
    beta: Tensor,
    /// `[1, 1, kernel]` the upsample's filter, shared by every channel.
    filter_up: Tensor,
    /// `[1, 1, kernel]` the low-pass's filter.
    filter_down: Tensor,
    ratio: u32,
}

impl Activation1d {
    /// The module at `vb`: `act.alpha` and `act.beta` in FP32 and the two Kaiser-sinc filters
    /// `upsample.filter` and `downsample.lowpass.filter`, in the element type the builder reads.
    ///
    /// # Errors
    ///
    /// No channels or ratio, or a tensor the checkpoint lacks or holds at another extent.
    pub fn load(config: &Activation1dConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        if config.channels == 0 || config.ratio == 0 {
            return Err(Error::shape(
                "Activation1d",
                "an activation has channels and a ratio".to_owned(),
            ));
        }

        // The reference's default taps: `int(6 · ratio // 2) · 2`, even and no shorter than
        // the ratio, so the pads and crops of `kernel / 2` keep the low-pass in phase.
        let kernel = (6 * config.ratio / 2 * 2) as usize;
        let channels = config.channels as usize;
        let parameters = vb.pp("act").to_dtype(DType::F32);

        Ok(Self {
            alpha: parameters.get(channels, "alpha")?,
            beta: parameters.get(channels, "beta")?,
            filter_up: vb.pp("upsample").get((1, 1, kernel), "filter")?,
            filter_down: vb
                .pp("downsample")
                .pp("lowpass")
                .get((1, 1, kernel), "filter")?,
            ratio: config.ratio,
        })
    }

    /// The activation of `x`, `[channels, length]` or `[batch, channels, length]`: a tensor of
    /// its shape and element type.
    ///
    /// # Errors
    ///
    /// A signal of other channels or another element type than the filters, an empty signal,
    /// or the operator's refusal.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        functional::activation1d(
            x,
            &self.alpha,
            &self.beta,
            &self.filter_up,
            &self.filter_down,
            self.ratio,
        )
    }
}

impl Module for Activation1d {
    fn forward(&self, x: &Tensor) -> uniserve_tensor::Result<Tensor> {
        Ok(Self::forward(self, x)?)
    }
}

/// What a checkpoint states of an [`AmpBlock`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AmpBlockConfig {
    /// The channels the block reads and writes.
    pub channels: u32,
    /// The residual kernel's taps, odd.
    pub kernel: u32,
    /// The dilation of each step's first convolution.
    pub dilations: Vec<u32>,
    /// The alias-free activations' ratio.
    pub ratio: u32,
}

/// One dilation step of an [`AmpBlock`]: activation, dilated convolution, activation, plain
/// convolution, whose result is added to the running value.
struct AmpStep {
    first_activation: Activation1d,
    first: Conv1d,
    second_activation: Activation1d,
    second: Conv1d,
}

/// BigVGAN's anti-aliased multi-periodicity block (`AMPBlock1`): one residual kernel run at
/// each of its dilations, every step adding its convolutional residual to the running value.
pub struct AmpBlock {
    steps: Vec<AmpStep>,
}

impl AmpBlock {
    /// The block at `vb`: `convs1.<i>` dilated by `dilations[i]` and padded by
    /// `dilations[i] · (kernel − 1) / 2`, `convs2.<i>` padded by `(kernel − 1) / 2`, both
    /// weight-normalized, and `activations.<2i>`, `activations.<2i + 1>` around them.
    ///
    /// # Errors
    ///
    /// An even kernel (no whole padding), no dilation, or a module the checkpoint does not
    /// hold.
    pub fn load(config: &AmpBlockConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let (channels, kernel) = (config.channels, config.kernel);
        if kernel == 0 || kernel.is_multiple_of(2) || config.dilations.is_empty() {
            return Err(Error::shape(
                "AmpBlock",
                format!(
                    "a kernel of {kernel} taps at dilations {:?} keeps no length",
                    config.dilations
                ),
            ));
        }
        let conv = |padding: u32, dilation: u32| ConvConfig {
            padding: [padding],
            dilation: [dilation],
            weight_norm: true,
            ..ConvConfig::new(channels, channels, [kernel])
        };
        let activation = Activation1dConfig {
            channels,
            ratio: config.ratio,
        };

        let steps = config
            .dilations
            .iter()
            .enumerate()
            .map(|(index, dilation)| {
                let activations = vb.pp("activations");

                Ok(AmpStep {
                    first_activation: Activation1d::load(&activation, activations.pp(2 * index))?,
                    first: Conv1d::load(
                        &conv(dilation * (kernel - 1) / 2, *dilation),
                        vb.pp("convs1").pp(index),
                    )?,
                    second_activation: Activation1d::load(
                        &activation,
                        activations.pp(2 * index + 1),
                    )?,
                    second: Conv1d::load(&conv((kernel - 1) / 2, 1), vb.pp("convs2").pp(index))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { steps })
    }

    /// The block over `x`, `[batch, channels, length]`: a tensor of its shape and element type,
    /// each step's residual added to the running value.
    ///
    /// # Errors
    ///
    /// A signal of other channels, or an operator's refusal.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut running = x.clone();

        for step in &self.steps {
            let convolved = step
                .first
                .forward(&step.first_activation.forward(&running)?)?;
            let residual = step
                .second
                .forward(&step.second_activation.forward(&convolved)?)?;

            running = (running + residual)?;
        }
        Ok(running)
    }
}

impl Module for AmpBlock {
    fn forward(&self, x: &Tensor) -> uniserve_tensor::Result<Tensor> {
        Ok(Self::forward(self, x)?)
    }
}

/// A BigVGAN generator's architecture, as its checkpoint's configuration states it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BigVganConfig {
    /// Channels `conv_pre` reads.
    pub in_channels: u32,
    /// Channels `conv_pre` writes (`upsample_initial_channel`); each stage halves them.
    pub channels: u32,
    /// Each stage's upsampling rate.
    pub upsample_rates: Vec<u32>,
    /// Each stage's transposed-convolution kernel.
    pub upsample_kernels: Vec<u32>,
    /// The residual kernels, one [`AmpBlock`] each per stage.
    pub resblock_kernels: Vec<u32>,
    /// Each residual kernel's dilations.
    pub resblock_dilations: Vec<Vec<u32>>,
    /// The alias-free activations' up- and down-sampling ratio.
    pub activation_ratio: u32,
    /// Whether `conv_post` has a bias (`use_bias_at_final`).
    pub final_bias: bool,
}

/// One upsampling stage: a transposed convolution and the averaged sum of its blocks.
struct Stage {
    upsample: ConvTranspose1d,
    blocks: Vec<AmpBlock>,
    rate: u32,
}

/// The BigVGAN generator from `conv_pre` to `conv_post`.
pub struct BigVgan {
    pre: Conv1d,
    stages: Vec<Stage>,
    /// `[1]` FP32: the reciprocal of a stage's block count, which its summed blocks are
    /// multiplied by.
    average: Tensor,
    post: Activation1d,
    output: Conv1d,
}

impl BigVgan {
    /// The generator at `vb` (`conv_pre`, `ups.<s>.0`, `resblocks.<s · K + k>`,
    /// `activation_post`, `conv_post`), every convolution weight-normalized.
    ///
    /// # Errors
    ///
    /// A config whose stage or residual lists disagree in length, a stage that cannot halve
    /// its channels or whose kernel is narrower than its rate, or a module the checkpoint
    /// lacks.
    pub fn load(config: &BigVganConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        const OWNER: &str = "BigVgan";

        if config.upsample_rates.len() != config.upsample_kernels.len()
            || config.upsample_rates.is_empty()
            || config.resblock_kernels.len() != config.resblock_dilations.len()
            || config.resblock_kernels.is_empty()
        {
            return Err(Error::shape(
                OWNER,
                format!("{config:?} has stages or residual kernels of disagreeing counts"),
            ));
        }
        let normalized = |from: u32, to: u32, kernel: u32, padding: u32| ConvConfig {
            padding: [padding],
            weight_norm: true,
            ..ConvConfig::new(from, to, [kernel])
        };
        let kernels = config.resblock_kernels.len();

        let mut channels = config.channels;
        let mut stages = Vec::with_capacity(config.upsample_rates.len());
        for (index, (&rate, &kernel)) in config
            .upsample_rates
            .iter()
            .zip(&config.upsample_kernels)
            .enumerate()
        {
            if channels < 2 || !channels.is_multiple_of(2) || rate == 0 || kernel < rate {
                return Err(Error::shape(
                    OWNER,
                    format!(
                        "stage {index} halves {channels} channels at rate {rate} with a kernel \
                         of {kernel}"
                    ),
                ));
            }
            let halved = channels / 2;

            // `(kernel − rate) / 2` in the reference's integer arithmetic.
            let upsample = ConvTranspose1d::load(
                &ConvConfig {
                    stride: [rate],
                    ..normalized(channels, halved, kernel, (kernel - rate) / 2)
                },
                vb.pp("ups").pp(index).pp(0),
            )?;
            let blocks = config
                .resblock_kernels
                .iter()
                .zip(&config.resblock_dilations)
                .enumerate()
                .map(|(block, (kernel, dilations))| {
                    AmpBlock::load(
                        &AmpBlockConfig {
                            channels: halved,
                            kernel: *kernel,
                            dilations: dilations.clone(),
                            ratio: config.activation_ratio,
                        },
                        vb.pp("resblocks").pp(index * kernels + block),
                    )
                })
                .collect::<Result<Vec<_>>>()?;

            stages.push(Stage {
                upsample,
                blocks,
                rate,
            });
            channels = halved;
        }

        Ok(Self {
            pre: Conv1d::load(
                &normalized(config.in_channels, config.channels, 7, 3),
                vb.pp("conv_pre"),
            )?,
            stages,
            // The reciprocal's product, not the reference's quotient: see the module
            // documentation.
            average: vb.constant("stage_average", DType::F32, 1, &[1.0 / kernels as f32])?,
            post: Activation1d::load(
                &Activation1dConfig {
                    channels,
                    ratio: config.activation_ratio,
                },
                vb.pp("activation_post"),
            )?,
            output: Conv1d::load(
                &ConvConfig {
                    bias: config.final_bias,
                    ..normalized(channels, 1, 7, 3)
                },
                vb.pp("conv_post"),
            )?,
        })
    }

    /// The samples `length` input frames become: the product of the stages' rates times
    /// `length`.
    pub fn samples(&self, length: usize) -> usize {
        self.stages
            .iter()
            .fold(length, |samples, stage| samples * stage.rate as usize)
    }

    /// The waveform of `x`, `[batch, in_channels, length]`: `[batch, 1, samples(length)]` in
    /// its element type.
    ///
    /// # Errors
    ///
    /// A latent of another rank or channel count, or an operator's refusal.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() != 3 {
            return Err(Error::shape(
                "BigVgan",
                format!("{:?}: a [batch, channels, frames] latent", x.dims()),
            ));
        }

        let mut current = self.pre.forward(x)?;
        for stage in &self.stages {
            let upsampled = stage.upsample.forward(&current)?;

            // The blocks' outputs summed in block order, then the average.
            let mut sum: Option<Tensor> = None;
            for block in &stage.blocks {
                let output = block.forward(&upsampled)?;
                sum = Some(match sum {
                    Some(sum) => (sum + output)?,
                    None => output,
                });
            }
            let Some(sum) = sum else {
                return Err(Error::shape("BigVgan", "a stage of no block".to_owned()));
            };

            current = sum
                .to_dtype(DType::F32)?
                .broadcast_mul(&self.average)?
                .to_dtype(sum.dtype())?;
        }
        self.output.forward(&self.post.forward(&current)?)
    }
}

impl Module for BigVgan {
    fn forward(&self, x: &Tensor) -> uniserve_tensor::Result<Tensor> {
        Ok(Self::forward(self, x)?)
    }
}
