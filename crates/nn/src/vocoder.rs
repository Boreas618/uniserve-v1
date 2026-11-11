//! The BigVGAN vocoder: latent features to a waveform through transposed-convolution upsampling
//! stages of anti-aliased multi-periodicity blocks.
//!
//! ```text
//! conv_pre                      kernel 7, padding 3
//! for each stage:
//!     ups.<s>.0                 ConvTranspose1d, stride rate, padding (kernel - rate) / 2, halving
//!                               the channels
//!     resblocks.<s · K + k>     one AmpBlock per residual kernel k of K; their outputs summed and
//!                               multiplied by 1 / K
//! activation_post               Activation1d(SnakeBeta)
//! conv_post                     kernel 7, padding 3, to one channel
//! ```
//!
//! [`Activation1d`] is BigVGAN's alias-free activation: a SnakeBeta between a `ratio`-times
//! Kaiser-sinc upsample and the matching low-pass, one fused program. Every convolution of the
//! generator is weight-normalized in the checkpoint and folded once at load.
//!
//! **The stage average multiplies by `1 / K`.** The reference divides the sum by the kernel count;
//! the product with the FP32 reciprocal differs from the quotient in the last bit of some values,
//! and the served outputs are this product's, so it is kept (a change is a protocol change). The
//! final `tanh` or clamp is the caller's: a PCM step clamps as it quantizes.

use ops::activation1d::{self, Activation1d as Activation1dOp};
use ops::{Shape, Operand};
use uniserve_tensor::DType;

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;
use crate::layers::{Axpby, Bias, Cast, Conv1d, ConvSpec, ConvTranspose1d, Residual};
use crate::weights::{Scope, Weight};

/// Channels one `audio_activation` program handles: the operator's tuning constant.
const ACTIVATION_BLOCK: u32 = 256;

/// The alias-free activation: SnakeBeta between a `ratio`-times upsample and its low-pass.
///
/// The reference's `Activation1d(SnakeBeta(channels, alpha_logscale=True))` with the up- and
/// down-sampling ratio `ratio` and its default filter of `int(6 · ratio / 2) · 2` taps. The signal
/// length is a launch value; channels, taps and ratio are compiled.
pub struct Activation1d {
    dtype: DType,
    channels: u32,
    kernel: u32,
    ratio: u32,
    /// `[channels]` SnakeBeta's `alpha`, in log space as the checkpoint stores it.
    alpha: Weight,
    /// `[channels]` SnakeBeta's `beta`, in log space.
    beta: Weight,
    /// `[1, 1, kernel]` the upsample's filter, shared by every channel.
    filter_up: Weight,
    /// `[1, 1, kernel]` the low-pass's filter.
    filter_down: Weight,
}

impl Activation1d {
    /// The module at `scope` over `channels` channels: `act.alpha` and `act.beta` (FP32) and the
    /// two Kaiser-sinc filters `upsample.filter` and `downsample.lowpass.filter`, in the element
    /// type the activation computes in.
    ///
    /// # Errors
    ///
    /// No channels or ratio, a tensor the checkpoint lacks, or parameters of another extent or
    /// element type.
    pub fn load(scope: &mut Scope<'_, '_>, channels: u32, ratio: u32) -> Result<Self> {
        const OWNER: &str = "Activation1d";
        if channels == 0 || ratio == 0 {
            return Err(Error::shape(
                OWNER,
                "an activation has channels and a ratio".to_owned(),
            ));
        }
        // The reference's default taps: `int(6 · ratio // 2) · 2`, even and no shorter than the
        // ratio, so the pads and crops of `kernel / 2` keep the low-pass in phase.
        let kernel = 6 * ratio / 2 * 2;
        let width = u64::from(channels);
        let alpha = scope.tensor("act.alpha", &[width])?;
        let beta = scope.tensor("act.beta", &[width])?;
        let filter = [1, 1, u64::from(kernel)];
        let filter_up = scope.tensor("upsample.filter", &filter)?;
        let filter_down = scope.tensor("downsample.lowpass.filter", &filter)?;
        let dtype = filter_up.dtype();
        for (name, weight) in [("alpha", &alpha), ("beta", &beta)] {
            if weight.dtype() != DType::F32 {
                return Err(Error::shape(
                    OWNER,
                    format!("the {name} is f32 per channel, not {}", ops::abi::name(weight.dtype())),
                ));
            }
        }
        if filter_down.dtype() != dtype {
            return Err(Error::shape(
                OWNER,
                "the two filters are one element type".to_owned(),
            ));
        }
        Ok(Self {
            dtype,
            channels,
            kernel,
            ratio,
            alpha,
            beta,
            filter_up,
            filter_down,
        })
    }

    /// The descriptor a block of this geometry launches; the length is a launch value.
    fn descriptor(&self) -> activation1d::Descriptor {
        activation1d::Descriptor {
            dtype: self.dtype,
            channels: self.channels,
            kernel: self.kernel,
            ratio: self.ratio,
            block: ACTIVATION_BLOCK,
            pdl: PDL,
        }
    }

    /// The activation of `input` `[channels, length]` or `[batch, channels, length]`: a fresh tensor
    /// of the same rank and the activation's element type, the length after the ratio's chain.
    ///
    /// A batch is the same channel set read again, so it is one launch per batch row over one
    /// descriptor.
    ///
    /// # Errors
    ///
    /// Operands of another shape or element type, an empty signal, or the operator's refusal.
    pub fn forward(&self, f: &Forward, input: Operand) -> Result<Operand> {
        const OWNER: &str = "Activation1d";
        let descriptor = self.descriptor();
        let channels = u64::from(self.channels);
        let batch = match input.shape.rank() {
            2 => 1,
            3 => input.shape.dim(0),
            _ => {
                return Err(Error::shape(
                    OWNER,
                    format!(
                        "{:?} is neither [{channels}, samples] nor [batch, {channels}, samples]",
                        input.shape.dims()
                    ),
                ));
            }
        };
        let rank = input.shape.rank();
        if input.shape.dim(rank - 2) != channels {
            return Err(Error::shape(
                OWNER,
                format!("{:?} does not have {channels} channels", input.shape.dims()),
            ));
        }
        let length = input.shape.dim(rank - 1);
        if length == 0 {
            return Err(Error::shape(
                OWNER,
                "a signal of no samples has no activation".to_owned(),
            ));
        }
        let expected = descriptor
            .output(length)
            .map_err(|error| Error::shape(OWNER, format!("{error}")))?;
        if input.dtype != self.dtype {
            return Err(Error::shape(
                OWNER,
                format!(
                    "the activation is {} and the signal {}",
                    ops::abi::name(self.dtype), ops::abi::name(input.dtype)
                ),
            ));
        }
        let mut dims = input.shape.dims().to_vec();
        dims[rank - 1] = expected;
        let out = f.empty(self.dtype, Shape::new(&dims))?;
        let length = u32::try_from(length)
            .map_err(|_| Error::shape(OWNER, format!("{length} samples exceed a launch value")))?;
        let element = ops::abi::bytes(self.dtype)
            .ok_or_else(|| Error::shape(OWNER, format!("{} has no byte width", ops::abi::name(self.dtype))))?;
        let (input_row, out_row) = (
            channels * u64::from(length) * element,
            channels * expected * element,
        );
        for row in 0..batch {
            f.operate::<Activation1dOp>(
                &descriptor,
                activation1d::Args {
                    values: Operand::new(
                        input.ptr.offset(row * input_row),
                        self.dtype,
                        Shape::new(&[channels, u64::from(length)]),
                    ),
                    alpha: self.alpha.view(),
                    beta: self.beta.view(),
                    filter_up: self.filter_up.view(),
                    filter_down: self.filter_down.view(),
                    out: Operand::new(
                        out.ptr.offset(row * out_row),
                        self.dtype,
                        Shape::new(&[channels, expected]),
                    ),
                    length,
                },
            )?;
        }
        Ok(out)
    }
}

/// One dilation step of an [`AmpBlock`]: activation, dilated convolution, activation, plain
/// convolution, whose result is added to the running value.
struct AmpStep {
    first_activation: Activation1d,
    first: Conv1d,
    second_activation: Activation1d,
    second: Conv1d,
}

/// BigVGAN's anti-aliased multi-periodicity block (`AMPBlock1`): one residual kernel run at each of
/// its dilations, every step adding its convolutional residual to the running value.
pub struct AmpBlock {
    channels: u32,
    steps: Vec<AmpStep>,
}

impl AmpBlock {
    /// The block at `scope` over `channels` channels with a `kernel`-tap residual kernel at each of
    /// `dilations`: `convs1.<i>` dilated by `dilations[i]` and padded by
    /// `dilations[i] · (kernel - 1) / 2`, `convs2.<i>` padded by `(kernel - 1) / 2`, both
    /// weight-normalized, and `activations.<2i>`, `activations.<2i + 1>` around them.
    ///
    /// # Errors
    ///
    /// An even kernel (no whole padding), no dilation, or a module the checkpoint does not hold.
    pub fn load(
        scope: &mut Scope<'_, '_>,
        channels: u32,
        kernel: u32,
        dilations: &[u32],
        ratio: u32,
    ) -> Result<Self> {
        if kernel == 0 || kernel.is_multiple_of(2) || dilations.is_empty() {
            return Err(Error::shape(
                "AmpBlock",
                format!("a kernel of {kernel} taps at dilations {dilations:?} keeps no length"),
            ));
        }
        let conv = |padding: u32, dilation: u32| ConvSpec {
            padding: [padding],
            dilation: [dilation],
            weight_norm: true,
            ..ConvSpec::new(channels, channels, [kernel])
        };
        let mut steps = Vec::with_capacity(dilations.len());
        for (index, dilation) in dilations.iter().enumerate() {
            steps.push(AmpStep {
                first: Conv1d::load(
                    &mut scope.scope(format!("convs1.{index}")),
                    conv(dilation * (kernel - 1) / 2, *dilation),
                )?,
                first_activation: Activation1d::load(
                    &mut scope.scope(format!("activations.{}", 2 * index)),
                    channels,
                    ratio,
                )?,
                second: Conv1d::load(
                    &mut scope.scope(format!("convs2.{index}")),
                    conv((kernel - 1) / 2, 1),
                )?,
                second_activation: Activation1d::load(
                    &mut scope.scope(format!("activations.{}", 2 * index + 1)),
                    channels,
                    ratio,
                )?,
            });
        }
        Ok(Self { channels, steps })
    }

    /// The block over `input` `[batch, channels, length]`: a fresh tensor of its shape and element
    /// type, each step's residual added into it in place. A step's intermediates are handed back
    /// when its add is enqueued.
    ///
    /// # Errors
    ///
    /// An input of another shape, or an operator's refusal.
    pub fn forward(&self, f: &Forward, input: Operand) -> Result<Operand> {
        if input.shape.rank() != 3 || input.shape.dim(1) != u64::from(self.channels) {
            return Err(Error::shape(
                "AmpBlock",
                format!(
                    "{:?} for a block of {} channels",
                    input.shape.dims(),
                    self.channels
                ),
            ));
        }
        let out = Cast
            .forward(f, rows(input)?, input.dtype)?
            .reshape(input.shape);
        for step in &self.steps {
            f.scoped(|f| {
                let convolved = step
                    .first
                    .forward(f, step.first_activation.forward(f, out)?)?;
                let residual = step
                    .second
                    .forward(f, step.second_activation.forward(f, convolved)?)?;
                Residual::default().forward(f, rows(out)?, rows(residual)?, None)?;
                Ok::<_, Error>(())
            })?;
        }
        Ok(out)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BigVganSpec {
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
    pub final_bias: Bias,
}

/// One upsampling stage: a transposed convolution and the averaged sum of its blocks.
struct Stage {
    upsample: ConvTranspose1d,
    blocks: Vec<AmpBlock>,
    rate: u32,
    channels: u32,
}

/// The BigVGAN generator from `conv_pre` to `conv_post`.
pub struct BigVgan {
    pre: Conv1d,
    stages: Vec<Stage>,
    post: Activation1d,
    output: Conv1d,
}

impl BigVgan {
    /// The generator at `scope` (`conv_pre`, `ups.<s>.0`, `resblocks.<s · K + k>`,
    /// `activation_post`, `conv_post`), every convolution weight-normalized.
    ///
    /// # Errors
    ///
    /// A spec whose stage or residual lists disagree in length, a stage that cannot halve its
    /// channels or whose kernel is narrower than its rate, or a module the checkpoint lacks.
    pub fn load(scope: &mut Scope<'_, '_>, spec: &BigVganSpec) -> Result<Self> {
        const OWNER: &str = "BigVgan";
        if spec.upsample_rates.len() != spec.upsample_kernels.len()
            || spec.upsample_rates.is_empty()
            || spec.resblock_kernels.len() != spec.resblock_dilations.len()
            || spec.resblock_kernels.is_empty()
        {
            return Err(Error::shape(
                OWNER,
                format!("{spec:?} has stages or residual kernels of disagreeing counts"),
            ));
        }
        let normalized = |from: u32, to: u32, kernel: u32, padding: u32| ConvSpec {
            padding: [padding],
            weight_norm: true,
            ..ConvSpec::new(from, to, [kernel])
        };
        let pre = Conv1d::load(
            &mut scope.scope("conv_pre"),
            normalized(spec.in_channels, spec.channels, 7, 3),
        )?;
        let kernels = spec.resblock_kernels.len();
        let mut channels = spec.channels;
        let mut stages = Vec::with_capacity(spec.upsample_rates.len());
        for (index, (&rate, &kernel)) in spec
            .upsample_rates
            .iter()
            .zip(&spec.upsample_kernels)
            .enumerate()
        {
            if channels < 2 || !channels.is_multiple_of(2) || rate == 0 || kernel < rate {
                return Err(Error::shape(
                    OWNER,
                    format!(
                        "stage {index} halves {channels} channels at rate {rate} with a kernel of \
                         {kernel}"
                    ),
                ));
            }
            let halved = channels / 2;
            // `(kernel - rate) / 2` in the reference's integer arithmetic.
            let upsample = ConvTranspose1d::load(
                &mut scope.scope(format!("ups.{index}.0")),
                ConvSpec {
                    stride: [rate],
                    ..normalized(channels, halved, kernel, (kernel - rate) / 2)
                },
            )?;
            let blocks = spec
                .resblock_kernels
                .iter()
                .zip(&spec.resblock_dilations)
                .enumerate()
                .map(|(block, (kernel, dilations))| {
                    AmpBlock::load(
                        &mut scope.scope(format!("resblocks.{}", index * kernels + block)),
                        halved,
                        *kernel,
                        dilations,
                        spec.activation_ratio,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            stages.push(Stage {
                upsample,
                blocks,
                rate,
                channels: halved,
            });
            channels = halved;
        }
        let post = Activation1d::load(
            &mut scope.scope("activation_post"),
            channels,
            spec.activation_ratio,
        )?;
        let output = Conv1d::load(
            &mut scope.scope("conv_post"),
            ConvSpec {
                bias: spec.final_bias,
                ..normalized(channels, 1, 7, 3)
            },
        )?;
        Ok(Self {
            pre,
            stages,
            post,
            output,
        })
    }

    /// The samples `length` input frames become: the product of the stages' rates times `length`.
    pub fn samples(&self, length: u64) -> u64 {
        self.stages
            .iter()
            .fold(length, |samples, stage| samples * u64::from(stage.rate))
    }

    /// The waveform of `input` `[batch, in_channels, length]`: a fresh `[batch, 1,
    /// samples(length)]` tensor of its element type.
    ///
    /// Each stage's averaged output is reserved before the stage and its upsampled input and
    /// blocks' outputs are handed back once the average is enqueued.
    ///
    /// # Errors
    ///
    /// An input of another rank, or an operator's refusal.
    pub fn forward(&self, f: &Forward, input: Operand) -> Result<Operand> {
        if input.shape.rank() != 3 {
            return Err(Error::shape(
                "BigVgan",
                format!(
                    "{:?}: a [batch, channels, frames] latent",
                    input.shape.dims()
                ),
            ));
        }
        let (batch, mut length) = (input.shape.dim(0), input.shape.dim(2));
        let mut current = self.pre.forward(f, input)?;
        for stage in &self.stages {
            length *= u64::from(stage.rate);
            let accumulated = f.empty(
                input.dtype,
                Shape::new(&[batch, u64::from(stage.channels), length]),
            )?;
            f.scoped(|f| {
                let upsampled = stage.upsample.forward(f, current)?;
                self.stage(f, stage, upsampled, accumulated)
            })?;
            current = accumulated;
        }
        let out = self.output.reserve(f, current)?;
        f.scoped(|f| {
            let activated = self.post.forward(f, current)?;
            self.output.launch(f, activated, out)
        })?;
        Ok(out)
    }

    /// One stage's blocks over `input`, summed into `accumulated` and multiplied by the reciprocal
    /// of their count.
    fn stage(
        &self,
        f: &Forward,
        stage: &Stage,
        input: Operand,
        accumulated: Operand,
    ) -> Result<()> {
        for (index, block) in stage.blocks.iter().enumerate() {
            f.scoped(|f| {
                let block_out = block.forward(f, input)?;
                // The first block initialises the sum (a zero `beta` reads no target); the rest
                // add.
                let beta = if index == 0 { 0.0 } else { 1.0 };
                Axpby.forward(f, rows(accumulated)?, rows(block_out)?, 1.0, beta)
            })?;
        }
        // The reciprocal's product, not the reference's quotient: see the module documentation.
        let reciprocal = 1.0 / stage.blocks.len() as f32;
        Axpby.forward(f, rows(accumulated)?, rows(accumulated)?, 0.0, reciprocal)
    }
}

/// A contiguous `[..., columns]` view as the `[rows, columns]` matrix the row-wise operators read.
fn rows(view: Operand) -> Result<Operand> {
    let columns = view.shape.dim(view.shape.rank() - 1);
    if columns == 0 {
        return Err(Error::shape("BigVgan", "a tensor of no column".to_owned()));
    }
    Ok(view.reshape(Shape::new(&[view.shape.numel() / columns, columns])))
}
