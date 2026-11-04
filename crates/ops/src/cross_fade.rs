//! `cross_fade`: the seam between two decoded tiles, blended over their overlap.
//!
//! The video decoder decodes overlapping tiles and blends their shared band so the seam does not
//! show. The reference's weights are not floats derived at run time but the integer ratio of the
//! position, `i/extent`, evaluated in the decoded dtype — which is what the reference's own arithmetic
//! does, and reproducing it exactly is the difference between a fade and a slightly different fade
//! (`.venv`/`uniserve` reference: `cross_fade` blends `previous[…, -extent + i, …] · (1 - i/extent)`
//! with `current[…, i, …] · (i/extent)` for the leading `extent` positions of one axis).
//!
//! ```text
//! for i < extent along `axis`:
//!     out[..., i, ...] = previous[..., -extent + i, ...] · (1 - i/extent)
//!                      + current [..., i,          ...] · (i/extent)
//! elsewhere along that axis:
//!     out[..., i, ...] = current[..., i, ...]
//! ```
//!
//! `out` may alias `current`: every blended position reads `current` at the same position it writes,
//! and `previous` only at the band, so a single buffer works — which is what the decoder wants, since
//! the tile it keeps is the one it just decoded.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Strides, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// Elements one thread block blends.
pub const BLOCK: u32 = 1024;

/// One compilation of `cross_fade`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of both tiles and of the output.
    pub dtype: DType,
    /// Voxels of overlap, blended from `previous` into `current`.
    pub extent: u32,
    /// Which axis of `[channels, frames, height, width]` the overlap runs along: `2` or `3`.
    pub axis: u32,
    /// Elements one thread block blends.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The axis is a compiled loop, not a launch value, so it is in the key; `extent` is too,
        // because it is the number of blended positions and the divisor of the weights.
        Key::new()
            .field("dtype", self.dtype)
            .field("extent", self.extent)
            .field("axis", self.axis)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one blend.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[channels, frames, height, width]` the tile already written.
    pub previous: TensorView,
    /// `[channels, frames, height, width]` the tile just decoded; may be the same buffer as `out`.
    pub current: TensorView,
    /// The blended tile.
    pub out: TensorView,
}

/// The seam between two decoded tiles.
pub struct CrossFade;

impl Op for CrossFade {
    const NAME: &'static str = "cross_fade";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        if descriptor.axis != 2 && descriptor.axis != 3 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the overlap runs along the height (2) or the width (3) of a \
                     [channels, frames, height, width] tile, not along {}",
                    descriptor.axis
                ),
            ));
        }
        let dims = args.current.shape.dims();
        if dims.len() != 4 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected a four-axis tile, found {:?}", dims),
            ));
        }
        let extent = u64::from(descriptor.extent);
        let along = dims[descriptor.axis as usize];
        if extent == 0 || extent > along {
            return Err(Error::argument(
                Self::NAME,
                format!("an overlap of {extent} voxels does not fit an axis of {along}"),
            ));
        }
        for (name, view) in [("previous", args.previous), ("current", args.current)] {
            view.require_shape(Self::NAME, args.current.shape)?;
            if view.dtype != descriptor.dtype {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected {} {name}, found {}", descriptor.dtype, view.dtype),
                ));
            }
        }
        args.out.require_shape(Self::NAME, args.current.shape)?;
        if args.out.dtype != descriptor.dtype {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} out, found {}",
                    descriptor.dtype, args.out.dtype
                ),
            ));
        }
        if args.out.strides != Strides::row_major(args.out.shape) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the output is written row-major; found strides {}",
                    args.out.strides
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.previous)
            .view(args.current)
            .view(args.out)
            .u32(descriptor.extent)
            .u32(descriptor.axis)
            .u32(u32::try_from(dims[0]).map_err(|_| {
                Error::argument(
                    Self::NAME,
                    format!("{} channels do not fit a kernel parameter", dims[0]),
                )
            })?)
            .u32(u32::try_from(dims[1]).map_err(|_| {
                Error::argument(
                    Self::NAME,
                    format!("{} frames do not fit a kernel parameter", dims[1]),
                )
            })?)
            .u32(descriptor.block)
            .flag(descriptor.pdl);
        // One thread per output voxel: the leading band blends and everything after it copies, and
        // both branches write the position they read `current` at, which is what lets `out` alias it.
        kernel.launch(
            LaunchGrid::linear(args.out.numel().div_ceil(u64::from(descriptor.block)))?,
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

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // Two products and a sum in the decoded dtype, with the weights the reference computes in
        // that same dtype: one implementation produces the reference's bits.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> Descriptor {
        Descriptor {
            dtype: DType::F16,
            extent: 5,
            axis: 2,
            block: BLOCK,
            pdl: false,
        }
    }

    #[test]
    fn the_key_names_the_overlap_and_the_axis() {
        assert_eq!(
            descriptor().to_string(),
            "dtype_f16_extent_5_axis_2_block_1024_pdl_false"
        );
        // A different seam width is a different program: `extent` divides the weights.
        assert_ne!(
            descriptor().to_string(),
            Descriptor {
                extent: 9,
                ..descriptor()
            }
            .to_string()
        );
        // And so is the other axis, because the stride walk is compiled in.
        assert_ne!(
            descriptor().to_string(),
            Descriptor {
                axis: 3,
                ..descriptor()
            }
            .to_string()
        );
    }

    #[test]
    fn the_weights_are_integer_ratios_so_the_blend_is_exact() {
        // `i/extent` for i in 0..extent, which sums to one across the pair by construction: the
        // reference computes them in the decoded dtype, so a faithful implementation is bit-exact.
        let extent = 5u32;
        let weights: Vec<f32> = (0..extent).map(|i| i as f32 / extent as f32).collect();
        assert_eq!(weights.first().copied(), Some(0.0));
        assert_eq!(weights.last().copied(), Some(0.8));
        assert_eq!(CrossFade::tolerance(&descriptor()), Tolerance::exact());
    }
}
