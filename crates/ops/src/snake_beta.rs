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
        let channels = u64::from(descriptor.channels);
        args.x.require_rank(Self::NAME, 3)?;
        let shape = args.x.shape;
        let width = shape.dim(2);
        let rows = shape.dim(0);
        if shape.dim(1) != channels {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {channels} channels on the axis after the rows, found {}",
                    shape.dim(1)
                ),
            ));
        }
        args.x.require_row(Self::NAME, descriptor.dtype, width)?;
        if args.x.strides != Strides::row_major(shape) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the activation is read row-major; found strides {}",
                    args.x.strides
                ),
            ));
        }
        args.out.require_shape(Self::NAME, shape)?;
        args.out.require_row(Self::NAME, descriptor.dtype, width)?;
        let parameters = Shape::new(&[channels]);
        args.alpha.require_shape(Self::NAME, parameters)?;
        args.alpha
            .require_row(Self::NAME, descriptor.dtype, channels)?;
        args.beta.require_shape(Self::NAME, parameters)?;
        args.beta
            .require_row(Self::NAME, descriptor.dtype, channels)?;

        // One thread per element; the plane stride is the row-major one the check above pins.
        let elements = shape.numel();
        let mut arguments = Arguments::new();
        arguments
            .view(args.x)
            .view(args.alpha)
            .view(args.beta)
            .view(args.out)
            .u64(rows)
            .u32(descriptor.channels)
            .u32(u32::try_from(width).map_err(|_| {
                Error::argument(
                    Self::NAME,
                    format!("a width of {width} elements does not fit a kernel parameter"),
                )
            })?)
            .flag(descriptor.pdl);
        kernel.launch(
            LaunchGrid::covering(elements, u64::from(descriptor.block))?,
            &mut arguments,
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
