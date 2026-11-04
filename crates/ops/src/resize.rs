//! `antialiased_resize_uint8`: the video pipeline's downscaling of a uint8 frame.
//!
//! This is torchvision's antialiased resize — `F.interpolate(scale, mode="bilinear" | "bicubic",
//! antialias=True, align_corners=False)` — written as its own kernel program because the reference
//! does it through ATen and specs/worker.md 9 gives it a program of its own. The filter is separable:
//! for one output index `i` along an axis, `center = (i + 0.5) · scale - 0.5`, the taps are the
//! input indices in `[center - support, center + support]` clamped to the input's extent, and each
//! tap's weight is the cubic (α = -0.75) or linear kernel evaluated at its distance over the
//! support, normalised by the taps' sum so a constant image stays constant.
//!
//! `scale = input_extent / output_extent` and `support = (bicubic ? 2 : 1) · max(1, scale)`, which
//! is what makes the resize antialiased: when the image shrinks, the filter widens to the fraction
//! of the source that each destination pixel covers. The two axes are computed the same way with
//! their own scale, so one program covers a non-square resize.
//!
//! The program is a kernel program; its parameters, in order: `x`, `out`, `rows`, `channels`,
//! `height`, `width`, `out_height`, `out_width`, `scale_h`, `scale_w`, the PDL flag. Weights are not
//! passed in: the program derives them from the two scales, deterministically, which keeps a graph
//! capture free of a host-side coefficient table and makes the program's answer independent of the
//! caller.
//!
//! The output is quantised, so the tolerance is one code value and nothing else: the two
//! implementations perform the same arithmetic in FP32, and may break a rounding tie differently.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Strides, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The interpolation kernel of a resize.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mode {
    /// Two taps per axis.
    Bilinear,
    /// Four taps per axis, cubic convolution with α = -0.75 as torchvision uses it.
    Bicubic,
}

impl fmt::Display for Mode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Bilinear => "bilinear",
            Self::Bicubic => "bicubic",
        })
    }
}

/// One compilation of `antialiased_resize_uint8`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Which interpolation the program applies.
    pub mode: Mode,
    /// Channels of the frame: the axis the resize leaves alone.
    pub channels: u32,
    /// Output pixels one thread block produces.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .variant(
                "mode",
                match self.mode {
                    Mode::Bilinear => "bilinear",
                    Mode::Bicubic => "bicubic",
                },
            )
            .field("channels", self.channels)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one resize.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[rows, channels, height, width]` uint8, contiguous.
    pub x: TensorView,
    /// `[rows, channels, out_height, out_width]` uint8, contiguous.
    pub out: TensorView,
}

/// The antialiased uint8 resize of the video path.
pub struct AntialiasedResizeUint8;

impl Op for AntialiasedResizeUint8 {
    const NAME: &'static str = "antialiased_resize_uint8";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        args.x.require_rank(Self::NAME, 4)?;
        args.out.require_rank(Self::NAME, 4)?;
        if args.x.dtype != DType::U8 || args.out.dtype != DType::U8 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a uint8 resize was given {} and {}",
                    args.x.dtype, args.out.dtype
                ),
            ));
        }
        // The program walks both images as contiguous planes: the frame the decoder hands over is
        // packed, and a strided source would make the tap arithmetic silently wrong.
        if args.x.strides != Strides::row_major(args.x.shape) {
            return Err(Error::argument(
                Self::NAME,
                format!("the frame is read packed; found strides {}", args.x.strides),
            ));
        }
        if args.out.strides != Strides::row_major(args.out.shape) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the resized frame is written packed; found strides {}",
                    args.out.strides
                ),
            ));
        }

        let channels = u64::from(descriptor.channels);
        let (rows, height, width) = (
            args.x.shape.dim(0),
            args.x.shape.dim(2),
            args.x.shape.dim(3),
        );
        let (out_height, out_width) = (args.out.shape.dim(2), args.out.shape.dim(3));
        if args.x.shape.dim(1) != channels {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {channels} channels, found {}",
                    args.x.shape.dim(1)
                ),
            ));
        }
        if args.out.shape.dim(0) != rows
            || args.out.shape.dim(1) != channels
            || out_height == 0
            || out_width == 0
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected [{rows}, {channels}, _, _] as the resized frame, found {:?}",
                    args.out.shape.dims()
                ),
            ));
        }
        if height == 0 || width == 0 {
            return Err(Error::argument(
                Self::NAME,
                "an empty frame has nothing to resize",
            ));
        }

        // ATen's `area_pixel_compute_scale` with `align_corners = false`: the scale is the ratio of
        // the extents, and it is what the antialias support widens with.
        let scale_h = height as f32 / out_height as f32;
        let scale_w = width as f32 / out_width as f32;

        let mut arguments = Arguments::new();
        arguments
            .view(args.x)
            .view(args.out)
            .u64(rows)
            .u32(descriptor.channels)
            .u32(u32_of(Self::NAME, height, "the height")?)
            .u32(u32_of(Self::NAME, width, "the width")?)
            .u32(u32_of(Self::NAME, out_height, "the output height")?)
            .u32(u32_of(Self::NAME, out_width, "the output width")?)
            .f32(scale_h)
            .f32(scale_w)
            .flag(descriptor.pdl);
        // One thread block per output row of one channel of one image; the block covers the row's
        // output pixels, each of them a separable sum of at most four taps per axis.
        kernel.launch(
            LaunchGrid::covering(rows * channels * out_height, u64::from(descriptor.block))?,
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
        // One code value: the same FP32 arithmetic, a possibly different rounding tie.
        Tolerance::exact().with_atol(1.0)
    }
}

/// A host extent as the `u32` a kernel parameter holds.
fn u32_of(operator: &'static str, value: u64, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        Error::argument(
            operator,
            format!("{what} of {value} pixels does not fit a kernel parameter"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> Descriptor {
        Descriptor {
            mode: Mode::Bicubic,
            channels: 3,
            block: 1024,
            pdl: false,
        }
    }

    #[test]
    fn the_key_states_the_interpolation_it_was_compiled_for() {
        assert_eq!(
            descriptor().to_string(),
            "mode_bicubic_channels_3_block_1024_pdl_false"
        );
        assert_eq!(
            Descriptor {
                mode: Mode::Bilinear,
                ..descriptor()
            }
            .to_string(),
            "mode_bilinear_channels_3_block_1024_pdl_false"
        );
    }

    #[test]
    fn a_quantised_result_allows_one_code_value_and_no_relative_slack() {
        let tolerance = AntialiasedResizeUint8::tolerance(&descriptor());
        assert_eq!(tolerance.atol, 1.0);
        assert_eq!(tolerance.rtol, 0.0);
    }
}
