//! `SnakeBeta`: the audio decoder's periodic activation, elementwise over `[rows, channels, width]`.
//!
//! ```text
//! out = x + sin²(exp(α) · x) / (exp(β) + 1e-9)
//! ```
//!
//! `α` and `β` are stored in log space as `[channels]`, one pair per channel, so a channel's
//! frequency and its reciprocal gain are learned independently
//! (`diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py:133-148`). The epsilon makes
//! the reciprocal finite at `β = -∞`; it is part of the formula and not a guard the kernel may drop.
//!
//! The program is a kernel program (specs/worker.md 9, "pad, upsample, SnakeBeta, downsample"):
//! one element of `out` per thread, the channel taken from the linear index so each thread reads one
//! `α` and one `β`. `x` and `out` must be row-major, which is what the convolution that feeds the
//! activation produces; the views still carry their own strides, so the same program reads a
//! strided batch row.
//!
//! The result is elementwise, so its tolerance is the output's own rounding plus one envelope: the
//! two implementations evaluate the same formula with their own `exp` and `sin`, each accurate to
//! about one ulp of its own result, and an ulp of `sin²` is bounded by an ulp of the value 1 over
//! the envelope `|exp(α)·x| ≤ 8` and `exp(β) ≥ 1` the decoder's parameters stay in.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, Strides, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The envelope of `|exp(α) · x|` and of `exp(β)`: the absolute part of the tolerance follows it.
const VALUE_ENVELOPE: f32 = 8.0;

/// One compilation of `snake_beta`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the activation and of the two parameters.
    pub dtype: DType,
    /// Channels of the audio signal: the length of `α` and `β`.
    pub channels: u32,
    /// Elements one thread block evaluates.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("channels", self.channels)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `snake_beta`.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[rows, channels, width]`, row-major, the signal to activate.
    pub x: TensorView,
    /// `[channels]` log-frequency `α`.
    pub alpha: TensorView,
    /// `[channels]` log-gain `β`.
    pub beta: TensorView,
    /// `[rows, channels, width]`, row-major, the activated signal.
    pub out: TensorView,
}

/// The periodic activation of the audio decoder.
pub struct SnakeBeta;

impl Op for SnakeBeta {
    const NAME: &'static str = "snake_beta";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        activate(
            Self::NAME,
            kernel,
            descriptor.dtype,
            descriptor.channels,
            descriptor.block,
            descriptor.pdl,
            args.x,
            args.alpha,
            Some(args.beta),
            args.out,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // Elementwise: the output's rounding, plus an ulp of the value 1 for each competing
        // `exp`/`sin`, scaled by the envelope the reciprocal and the angle stay inside.
        let unit = descriptor.dtype.unit_roundoff() as f32;
        Tolerance::rounding(1, DType::F32, descriptor.dtype).with_atol(8.0 * VALUE_ENVELOPE * unit)
    }
}

/// The launch both activations share: one thread per element, the channel from the linear index.
///
/// `beta` is the log-space gain of [`SnakeBeta`]; the encoder's [`SnakeAlpha`] has no second
/// parameter, so the program that implements it takes `x`, `alpha`, `out` and no gain at all — the
/// two parameter lists differ, and a program for one is not a program for the other.
#[allow(clippy::too_many_arguments)]
fn activate(
    operator: &'static str,
    kernel: &Kernel,
    dtype: DType,
    channels: u32,
    block: u32,
    pdl: bool,
    x: TensorView,
    alpha: TensorView,
    beta: Option<TensorView>,
    out: TensorView,
    stream: &Stream,
) -> Result<()> {
    let channel_count = channels;
    let channels = u64::from(channels);
    x.require_rank(operator, 3)?;
    let shape = x.shape;
    let width = shape.dim(2);
    let rows = shape.dim(0);
    if shape.dim(1) != channels {
        return Err(Error::argument(
            operator,
            format!(
                "expected {channels} channels on the axis after the rows, found {}",
                shape.dim(1)
            ),
        ));
    }
    x.require_row(operator, dtype, width)?;
    if x.strides != Strides::row_major(shape) {
        return Err(Error::argument(
            operator,
            format!(
                "the activation is read row-major; found strides {}",
                x.strides
            ),
        ));
    }
    out.require_shape(operator, shape)?;
    out.require_row(operator, dtype, width)?;
    let parameters = Shape::new(&[channels]);
    for (view, what) in [(alpha, "alpha"), (beta.unwrap_or(alpha), "beta")] {
        view.require_shape(operator, parameters)?;
        if view.require_row(operator, dtype, channels).is_err() {
            return Err(Error::argument(
                operator,
                format!("{what} is not {channels} {dtype} elements with a unit last stride"),
            ));
        }
    }

    let mut arguments = Arguments::new();
    arguments.view(x).view(alpha);
    if let Some(beta) = beta {
        arguments.view(beta);
    }
    arguments
        .view(out)
        .u64(rows)
        .u32(channel_count)
        .u32(u32::try_from(width).map_err(|_| {
            Error::argument(
                operator,
                format!("a width of {width} elements does not fit a kernel parameter"),
            )
        })?)
        .flag(pdl);
    kernel.launch(
        LaunchGrid::covering(shape.numel(), u64::from(block))?,
        &mut arguments,
        stream,
    )
}

/// One compilation of `snake_alpha`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AlphaDescriptor {
    /// Element type of the activation and of the parameter.
    pub dtype: DType,
    /// Channels of the audio signal: the length of `α`.
    pub channels: u32,
    /// Elements one thread block evaluates.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for AlphaDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("channels", self.channels)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `snake_alpha`.
#[derive(Clone, Copy)]
pub struct AlphaArgs {
    /// `[rows, channels, width]`, row-major, the signal to activate.
    pub x: TensorView,
    /// `[channels]` the frequency, stored as it is used.
    pub alpha: TensorView,
    /// `[rows, channels, width]`, row-major, the activated signal.
    pub out: TensorView,
}

/// The periodic activation of the audio encoder: `x + sin²(α·x)/(α + 1e-9)`.
///
/// The DAC encoder's form (`uniserve/models/minimax_h3/audio_vae.py:297`): one per-channel parameter
/// serving as both the frequency and the reciprocal gain, stored directly rather than in log space.
/// It is not [`SnakeBeta`] with `β = α` — that would divide by `exp(α)` — so the two are separate
/// operators with separate programs.
pub struct SnakeAlpha;

impl Op for SnakeAlpha {
    const NAME: &'static str = "snake_alpha";

    type Descriptor = AlphaDescriptor;
    type Args<'a> = AlphaArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        activate(
            Self::NAME,
            kernel,
            descriptor.dtype,
            descriptor.channels,
            descriptor.block,
            descriptor.pdl,
            args.x,
            args.alpha,
            None,
            args.out,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        AlphaArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The same elementwise bound as the decoder's activation: the output's rounding plus an ulp
        // of the value 1 for each competing `sin`, scaled by the envelope the reciprocal and the
        // angle stay inside (`α >= 1` and `|α·x| <= 8` in the encoder's parameter range).
        let unit = descriptor.dtype.unit_roundoff() as f32;
        Tolerance::rounding(1, DType::F32, descriptor.dtype).with_atol(8.0 * VALUE_ENVELOPE * unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> Descriptor {
        Descriptor {
            dtype: DType::F32,
            channels: 1536,
            block: 1024,
            pdl: false,
        }
    }

    #[test]
    fn the_key_states_every_compiled_field() {
        assert_eq!(
            descriptor().to_string(),
            "dtype_f32_channels_1536_block_1024_pdl_false"
        );
    }

    #[test]
    fn the_tolerance_is_the_rounding_plus_the_transcendental_envelope() {
        let tolerance = SnakeBeta::tolerance(&descriptor());
        assert_eq!(
            tolerance.atol,
            8.0 * VALUE_ENVELOPE * DType::F32.unit_roundoff() as f32
        );
        // A reduction of one term in FP32: two gammas and the output's own unit roundoff.
        assert!(tolerance.rtol > 0.0);
    }

    #[test]
    fn the_encoder_activation_has_its_own_key_and_the_same_bound() {
        let alpha = AlphaDescriptor {
            dtype: DType::F32,
            channels: 96,
            block: 1024,
            pdl: false,
        };
        assert_eq!(
            alpha.to_string(),
            "dtype_f32_channels_96_block_1024_pdl_false"
        );
        // One parameter, one sine: the bound is the decoder's, from the same envelope.
        assert_eq!(
            SnakeAlpha::tolerance(&alpha).atol,
            SnakeBeta::tolerance(&Descriptor {
                dtype: DType::F32,
                channels: 96,
                block: 1024,
                pdl: false,
            })
            .atol
        );
        assert_eq!(SnakeAlpha::NAME, "snake_alpha");
        assert_ne!(SnakeBeta::NAME, SnakeAlpha::NAME);
    }

    #[test]
    fn a_bf16_activation_holds_its_own_rounding() {
        let bf16 = Descriptor {
            dtype: DType::BF16,
            ..descriptor()
        };
        assert!(SnakeBeta::tolerance(&bf16).atol > SnakeBeta::tolerance(&descriptor()).atol);
        assert_eq!(
            bf16.to_string(),
            "dtype_bf16_channels_1536_block_1024_pdl_false"
        );
    }
}
