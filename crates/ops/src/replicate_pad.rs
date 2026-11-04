//! `replicate_pad`: ATen's `F.pad(..., mode="replicate")` over the trailing axes.
//!
//! Every output element is the input element at the same index, clamped into the input's extent
//! along each padded axis — the edge value repeats outward. A negative pad crops, as ATen's does.
//! The audio decoder pads the ends of a waveform before its low-pass (`pad_left = k/2 - even`,
//! `pad_right = k/2`) and before its transposed convolution (`pad = k//ratio - 1`); the video
//! decoder pads the last axes of a latent. There is no arithmetic: the value at an interior index
//! moves unchanged, which is why the tolerance is exact (specs/worker.md 9 wants it its own kernel
//! program so the pad never becomes a copy through the host).
//!
//! Padded axes are the last `axes` axes of the input; `axes` is one to three, and the axes before it
//! are flattened into `leading`, so a `[batch, channels, length]` waveform is padded as
//! `leading = batch · channels` with one padded axis.
//!
//! The kernel's parameters, in order: `x`, `out`, `leading`, then for each of the three axis slots
//! `in_extent`, `out_extent`, `left`, `right` (unused slots are zero), then the PDL flag. The kernel
//! is Triton, so its parameters are scalars and pointers only: shapes travel as parameters rather
//! than through the pointer, and an axis slot past `axes` must be ignored by the program.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Strides, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// How many axes one compilation may pad.
pub const MAX_AXES: usize = 3;

/// One compilation of `replicate_pad`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the padded tensor.
    pub dtype: DType,
    /// How many trailing axes are padded: one to [`MAX_AXES`].
    pub axes: u32,
    /// Elements one thread block copies.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("axes", self.axes)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one replicate pad.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[leading, width[0], .., width[axes-1]]`, row-major.
    pub x: TensorView,
    /// The padded tensor, row-major, its padded axes `width[i] + left[i] + right[i]` long.
    pub out: TensorView,
    /// The pad before each padded axis, in the order `x`'s axes take; negative crops.
    pub left: [i32; MAX_AXES],
    /// The pad after each padded axis.
    pub right: [i32; MAX_AXES],
}

/// Edge replication over the trailing axes.
pub struct ReplicatePad;

impl Op for ReplicatePad {
    const NAME: &'static str = "replicate_pad";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let axes = usize::try_from(descriptor.axes).unwrap_or(0);
        if !(1..=MAX_AXES).contains(&axes) {
            return Err(Error::argument(
                Self::NAME,
                format!("expected one to {MAX_AXES} padded axes, found {axes}"),
            ));
        }
        let rank = args.x.shape.rank();
        if rank < axes {
            return Err(Error::argument(
                Self::NAME,
                format!("a rank-{rank} tensor cannot have {axes} padded trailing axes"),
            ));
        }
        args.x
            .require_row(Self::NAME, descriptor.dtype, args.x.last())?;
        if args.x.strides != Strides::row_major(args.x.shape) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the pad is read row-major; found strides {}",
                    args.x.strides
                ),
            ));
        }
        args.out
            .require_row(Self::NAME, descriptor.dtype, args.out.last())?;
        if args.out.strides != Strides::row_major(args.out.shape) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the pad writes row-major; found strides {}",
                    args.out.strides
                ),
            ));
        }

        let leading = args.x.shape.dims()[..rank - axes].iter().product();
        if args.out.shape.dims()[..rank - axes].iter().product::<u64>() != leading {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the leading axes of {:?} cannot hold {} of them",
                    args.out.shape.dims(),
                    leading
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments.view(args.x).view(args.out).u64(leading);
        for axis in 0..MAX_AXES {
            let (extent, padded) = if axis < axes {
                let at = rank - axes + axis;
                let left = args.left[axis];
                let right = args.right[axis];
                let extent = args.x.shape.dim(at);
                let padded =
                    i64::try_from(extent).unwrap_or(0) + i64::from(left) + i64::from(right);
                if padded <= 0 {
                    return Err(Error::argument(
                        Self::NAME,
                        format!(
                            "axis {axis} of {extent} elements with a pad of {left} and {right} \
                             leaves nothing"
                        ),
                    ));
                }
                let expected = args.out.shape.dim(at);
                if u64::try_from(padded) != Ok(expected) {
                    return Err(Error::argument(
                        Self::NAME,
                        format!(
                            "axis {axis} of {extent} elements with a pad of {left} and {right} \
                             is {padded} long, and the output's is {expected}"
                        ),
                    ));
                }
                (extent, u64::try_from(padded).unwrap_or(0))
            } else {
                (0, 0)
            };
            arguments
                .u64(extent)
                .u64(padded)
                .i32(args.left[axis])
                .i32(args.right[axis]);
        }
        arguments.flag(descriptor.pdl);
        kernel.launch(
            LaunchGrid::covering(args.out.numel(), u64::from(descriptor.block))?,
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
        let _ = descriptor;
        // Every output element is a copy of one input element — the padding repeats the edge rather
        // than inventing a value — so two implementations agree bit for bit.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> Descriptor {
        Descriptor {
            dtype: DType::F32,
            axes: 1,
            block: 1024,
            pdl: false,
        }
    }

    #[test]
    fn the_key_states_every_compiled_field() {
        assert_eq!(
            descriptor().to_string(),
            "dtype_f32_axes_1_block_1024_pdl_false"
        );
        assert_eq!(
            Descriptor {
                axes: 3,
                dtype: DType::BF16,
                ..descriptor()
            }
            .to_string(),
            "dtype_bf16_axes_3_block_1024_pdl_false"
        );
    }

    #[test]
    fn a_pad_moves_elements_and_is_exact() {
        let tolerance = ReplicatePad::tolerance(&descriptor());
        assert_eq!(tolerance.atol, 0.0);
        assert_eq!(tolerance.rtol, 0.0);
    }
}
