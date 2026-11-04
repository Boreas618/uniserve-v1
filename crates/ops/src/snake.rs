//! `Snake`: the periodic activation of the H3 audio VAE, elementwise over `[rows, channels, width]`.
//!
//! ```text
//! out = x + sin²(α · x) / (α + 1e-9)
//! ```
//!
//! `α` is `[channels]`, stored as it is used (not in log space), one frequency and reciprocal gain per
//! channel: `uniserve/models/minimax_h3/audio_vae.py:297-311` defines the decoder's activation with a
//! single `nn.Parameter(torch.ones(1, channels, 1))` and evaluates exactly the formula above. The
//! epsilon keeps the reciprocal finite for a zero frequency; it is part of the formula and not a
//! guard a kernel may drop.
//!
//! This is the H3 decoder's activation and is **not** `snake_beta`: BigVGAN-style blocks store a
//! frequency and a gain separately, in log space (`snake_beta`, `x + sin²(exp(α)·x)/(exp(β) + 1e-9)`),
//! while H3's audio VAE stores one linear parameter that serves as both. The two have different
//! parameters, so folding them behind a flag would make one operator read a weight the other's
//! checkpoint does not have.
//!
//! The program is a kernel program (specs/worker.md 9, "pad, upsample, SnakeBeta, downsample"): one
//! element of `out` per thread, the channel taken from the linear index so each thread reads one
//! `α`. `x` and `out` must be row-major, which is what the convolution that feeds the activation
//! produces.
//!
//! The result is elementwise, so its tolerance is the output's own rounding plus one envelope: the
//! two implementations evaluate the same formula with their own `sin` and their own reciprocal, each
//! accurate to about one ulp of its result, and over the decoder's envelope `|α·x| ≤ 8` and
//! `α + 1e-9 ≥ 1` an ulp of `sin²` is bounded by an ulp of the value 1.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, Strides, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The envelope of `|α · x|`: the absolute part of the tolerance follows it.
const VALUE_ENVELOPE: f32 = 8.0;

/// One compilation of `snake`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the activation and of the parameter.
    pub dtype: DType,
    /// Channels of the audio signal: the length of `α`.
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

/// The arguments of one `snake`.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[rows, channels, width]`, row-major, the signal to activate.
    pub x: TensorView,
    /// `[channels]` frequency `α`, as the checkpoint stores it.
    pub alpha: TensorView,
    /// `[rows, channels, width]`, row-major, the activated signal.
    pub out: TensorView,
}

/// The periodic activation of the H3 audio decoder.
pub struct Snake;

impl Op for Snake {
    const NAME: &'static str = "snake";

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
        args.alpha
            .require_shape(Self::NAME, Shape::new(&[channels]))?;
        args.alpha
            .require_row(Self::NAME, descriptor.dtype, channels)?;

        // One thread per element; the plane stride is the row-major one the check above pins.
        let elements = shape.numel();
        let mut arguments = Arguments::new();
        arguments
            .view(args.x)
            .view(args.alpha)
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
        // `sin`, its square and the reciprocal are three roundings in FP32 before the output's own;
        // the envelope above bounds the addend by the value 1.
        let unit = descriptor.dtype.unit_roundoff() as f32;
        Tolerance::rounding(3, DType::F32, descriptor.dtype).with_atol(8.0 * VALUE_ENVELOPE * unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_activation_is_compiled_for_a_channel_count_and_a_block() {
        let descriptor = Descriptor {
            dtype: DType::BF16,
            channels: 512,
            block: 1024,
            pdl: false,
        };
        assert_eq!(
            crate::descriptor::key(&descriptor),
            "dtype_bf16_channels_512_block_1024_pdl_false"
        );
        // The parameter count is the channel count, so a second channel layout is a second module.
        assert_ne!(
            crate::descriptor::key(&descriptor),
            crate::descriptor::key(&Descriptor {
                channels: 256,
                ..descriptor
            })
        );
    }
}
