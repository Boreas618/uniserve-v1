//! `rgb_bytes`: the video path's last step, from a normalized float raster to 8-bit RGB.
//!
//! The decoder's raster is ImageNet-normalized channel-major (`[channels, frames, rows, columns]`,
//! the layout the VAE speaks), and the artifact is packed `[frames, rows, columns, channels]` bytes.
//! One arithmetic per element, but it is the step whose rounding is visible in the output, so the
//! quantization is written out rather than left to host code beside the graph:
//!
//! ```text
//! out[f, r, c, k] = quantize((input[k, f, r, c] · std[k] + mean[k]) · 255)
//!
//! quantize(scaled) = round_half_to_even(clamp(scaled, 0, 255))        as u8
//! ```
//!
//! **The arithmetic is FP32 throughout**, whatever the input's element type: the reference converts
//! its normalized pixels to `float32` before the de-normalization and the clamp (`uniserve/media/
//! pixels.py`, `quantize_image_hwc`), and the same order matters here — a BF16 multiply then a BF16
//! add would round twice inside a range of 255 worth of bytes.
//!
//! De-normalization is multiply then add, left to right: the encoder normalized as
//! `(v / 255 - mean) / std`, so `(x · std + mean) · 255` returns `v`, and no reassociation
//! (`x·std + mean` against `x·std + (mean)`, or folding the 255 into `std`) is the same floating
//! point expression. `round_half_to_even` is the reference's: ties go to the even neighbour, which is
//! what `numpy.rint` and Rust's `f32::round_ties_even` do.
//!
//! **`out` may be a sub-region of a larger raster, and so may `input`**: the write goes through
//! `out`'s own strides, so a channel tile placed inside a canvas is one launch rather than a pass and
//! a copy — which is what the decoder needs, since each decoded tile lands in a sub-region of one
//! clip raster. The rule is the crate's: the **innermost** axis is unit-strided (the channels of
//! `[frames, rows, columns, 3]`, the columns of the read side), **rows must not overlap** (each
//! outer stride is at least the next axis's extent times its stride), and every outer stride is
//! otherwise arbitrary. A transposed view is refused, because it would put a gather in the inner
//! loop for a layout nobody needs; where the strides are row-major the same loop vectorizes, so the
//! general form costs nothing in the common case.
//!
//! `mean` and `std` are **views, not host scalars**: they are the checkpoint's ImageNet constants,
//! published once as device weights and held by the layer, so no call re-uploads three floats and a
//! captured forward sees them as the memory it already has.
//!
//! The geometry is the views': the operation is elementwise, so no compiled program depends on the
//! frame count, the raster's sides or the element count, and all three are launch values (the extent
//! rule's middle tier, exactly like `pcm_interleave`'s sample count).

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// Channels of the packed output: the ImageNet constants' length, and the only value the reference's
/// de-normalization has constants for.
pub const CHANNELS: u32 = 3;

/// Elements one thread block converts.
pub const BLOCK: u32 = 1024;

/// The de-normalization's scale onto bytes: the reference multiplies by 255 before rounding.
pub const BYTE_SCALE: f32 = 255.0;

/// The quantization of one element, in host form: the reference's arithmetic, used by the tests and
/// available as an oracle to anything that needs one.
///
/// `value` is the normalized input element, `mean` and `std` its channel's constants. The result is
/// the byte the device program writes for that element.
pub fn quantize(value: f32, mean: f32, std: f32) -> u8 {
    quantize_scaled((value * std + mean) * BYTE_SCALE)
}

/// The last step alone: clamp into `[0, 255]` and round to the nearest byte, ties to even.
///
/// Split from [`quantize`] so a test can name a tie exactly (`0.5`, `1.5`, `2.5`), which is what the
/// rounding's contract is about.
pub fn quantize_scaled(scaled: f32) -> u8 {
    // A NaN has no byte; the reference's `.clamp(0, 1)` leaves it NaN and `round()` keeps it, and its
    // conversion to `uint8` is then implementation-defined, so a launch that produces one is a
    // decoder defect rather than a rounding. Saturating to 0 here keeps the host oracle total.
    if scaled.is_nan() {
        return 0;
    }
    scaled.clamp(0.0, BYTE_SCALE).round_ties_even() as u8
}

/// The outer strides of a raster view, refusing a broadcast or an overlapping one.
///
/// A rank-4 raster's axes are `[channels, frames, rows, columns]` on the read side and
/// `[frames, rows, columns, channels]` on the write side; the innermost is unit-strided by the
/// crate's row rule (`require_row`), so what the program needs is the three that carry it between
/// planes, rows and columns. Two views are refused here: one with a **zero** stride on an axis (every
/// element of that axis would be the same byte — a broadcast, not a raster), and one whose axes
/// **overlap** (`stride[i]` below `extent[i+1] · stride[i+1]`, so two elements would share a byte).
///
/// A sub-region of a larger raster passes unchanged: it is only a bigger outer stride, which is what
/// the decoder's tile-in-a-canvas write needs.
fn strides_of(operator: &'static str, name: &str, view: &TensorView) -> Result<[u64; 3]> {
    let rank = view.shape.rank();
    let strides = view.strides.as_slice();
    let mut outer = [0_u64; 3];
    for axis in 0..rank {
        if strides[axis] == 0 {
            return Err(Error::argument(
                operator,
                format!("the {name} raster has a zero stride on axis {axis}"),
            ));
        }
        if axis + 1 < rank {
            let needed = view.shape.dim(axis + 1) * strides[axis + 1];
            if strides[axis] < needed {
                return Err(Error::argument(
                    operator,
                    format!(
                        "the {name} raster overlaps on axis {axis}: stride {} is below {needed}",
                        strides[axis]
                    ),
                ));
            }
        }
        if axis < 3 {
            outer[axis] = strides[axis];
        }
    }
    Ok(outer)
}

/// One compilation of `rgb_bytes`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the normalized input raster.
    pub dtype: DType,
    /// Channels: it must be [`CHANNELS`], the constants' length.
    pub channels: u32,
    /// Elements one thread block converts.
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

/// The arguments of one `rgb_bytes` launch.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[channels, frames, rows, columns]` the normalized raster.
    pub input: TensorView,
    /// `[channels]` f32: each channel's ImageNet mean, the addend of the de-normalization.
    pub mean: TensorView,
    /// `[channels]` f32: each channel's ImageNet standard deviation, the multiplier.
    pub std: TensorView,
    /// `[frames, rows, columns, channels]` u8: the packed image bytes.
    pub out: TensorView,
}

/// The normalized raster to packed 8-bit RGB.
pub struct RgbBytes;

impl Op for RgbBytes {
    const NAME: &'static str = "rgb_bytes";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        if descriptor.channels != CHANNELS {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the de-normalization has {CHANNELS} constants, the descriptor says {} channels",
                    descriptor.channels
                ),
            ));
        }
        if !matches!(descriptor.dtype, DType::F32 | DType::BF16) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected an f32 or bf16 raster, the descriptor says {}",
                    descriptor.dtype
                ),
            ));
        }

        let channels = u64::from(descriptor.channels);
        args.input.require_rank(Self::NAME, 4)?;
        if args.input.dtype != descriptor.dtype {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} input, found {}",
                    descriptor.dtype, args.input.dtype
                ),
            ));
        }
        let frames = args.input.shape.dim(1);
        let rows = args.input.shape.dim(2);
        let columns = args.input.shape.dim(3);
        if args.input.shape.dim(0) != channels {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the raster's first axis is {}, the descriptor says {channels} channels",
                    args.input.shape.dim(0)
                ),
            ));
        }
        // The program reads along the columns axis, so that axis is unit-stride; the channel-major
        // outer axes may be strided.
        args.input
            .require_row(Self::NAME, descriptor.dtype, columns)?;

        for (name, view) in [("mean", &args.mean), ("std", &args.std)] {
            view.require_shape(Self::NAME, Shape::new(&[channels]))?;
            if view.dtype != DType::F32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected f32 {name}, found {}", view.dtype),
                ));
            }
        }

        args.out.require_rank(Self::NAME, 4)?;
        if args.out.dtype != DType::U8 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected u8 output, found {}", args.out.dtype),
            ));
        }
        args.out
            .require_shape(Self::NAME, Shape::new(&[frames, rows, columns, channels]))?;
        // The packed image is contiguous, and the program writes it as a flat run of pixels.
        args.out.require_row(Self::NAME, DType::U8, channels)?;

        // A view carries its base address into the launch, not its strides, so a strided operand has
        // to hand them over: the program walks `out` as a block of a larger raster and `input` as a
        // slice of the latent, and both are only correct if it is told how far apart the axes are.
        let input = strides_of(Self::NAME, "input", &args.input)?;
        let out = strides_of(Self::NAME, "out", &args.out)?;

        let mut arguments = Arguments::new();
        arguments
            .view(args.input)
            .view(args.mean)
            .view(args.std)
            .view(args.out)
            .u64(frames)
            .u64(rows)
            .u64(columns)
            .u32(descriptor.channels)
            .u64(input[0])
            .u64(input[1])
            .u64(input[2])
            .u64(out[0])
            .u64(out[1])
            .u64(out[2])
            .flag(descriptor.pdl);
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
        // **One byte, not half.** The final rounding moves a value by at most half a byte, but the
        // bound is between two implementations of the same real arithmetic, and a value within the
        // float pipeline's own error of a half-byte boundary rounds to either side — and the two
        // sides are a byte apart. So `|candidate - reference| <= 1` is the tight bound for a four
        // step FP32 pipeline (multiply, add, multiply by 255, round), and half a byte would reject
        // exactly the tie case that the arithmetic must accept.
        //
        // No relative term: a byte near 255 is a relative error of 4e-3 and a byte near 1 is one of
        // 1, so a relative bound would say nothing about the small values and everything about the
        // large ones. The error is absolute in the output's units by construction.
        Tolerance::new(1.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_names_the_element_type_the_channels_and_the_block() {
        let descriptor = Descriptor {
            dtype: DType::BF16,
            channels: 3,
            block: 1024,
            pdl: false,
        };
        assert_eq!(
            crate::descriptor::key(&descriptor),
            "dtype_bf16_channels_3_block_1024_pdl_false"
        );
        // The geometry is not in the key, so a resolution or a clip length cannot make a new module.
        assert_eq!(
            crate::descriptor::key(&descriptor),
            crate::descriptor::key(&Descriptor {
                block: 1024,
                ..descriptor
            })
        );
        assert_ne!(
            crate::descriptor::key(&descriptor),
            crate::descriptor::key(&Descriptor {
                dtype: DType::F32,
                ..descriptor
            })
        );
    }

    #[test]
    fn a_tie_rounds_to_the_even_neighbour() {
        // The reference's `numpy.rint`/`torch.round` behaviour, which is what a program must match:
        // away-from-zero would give 1 and 3 here and differ on exactly the half bytes.
        assert_eq!(quantize_scaled(0.5), 0);
        assert_eq!(quantize_scaled(1.5), 2);
        assert_eq!(quantize_scaled(2.5), 2);
        assert_eq!(quantize_scaled(3.5), 4);
        // Just either side of a tie is not a tie.
        assert_eq!(quantize_scaled(0.4999), 0);
        assert_eq!(quantize_scaled(0.5001), 1);
    }

    #[test]
    fn both_ends_clamp_to_the_byte_range() {
        // A decoder artifact outside the normalized range is clipped at the ends, not wrapped.
        assert_eq!(quantize_scaled(-1.0), 0);
        assert_eq!(quantize_scaled(0.0), 0);
        assert_eq!(quantize_scaled(255.0), 255);
        assert_eq!(quantize_scaled(256.0), 255);
        assert_eq!(quantize_scaled(1e9), 255);
    }

    #[test]
    fn the_denormalization_is_the_inverse_of_the_encoder() {
        // The encoder divided by 255 and normalized; this is the same arithmetic backwards, so the
        // bytes of a raster that went through the encoder come back unchanged at every end of the
        // range.
        let std = 0.229f32;
        let mean = 0.485f32;
        for byte in [0u32, 1, 17, 64, 128, 200, 254, 255] {
            let normalized = (byte as f32 / 255.0 - mean) / std;
            let back = quantize(normalized, mean, std);
            // The round trip is exact except at the ends, where the clamp is what catches it — and
            // at most one byte everywhere, which is the declared bound.
            assert!(
                (i32::from(back) - byte as i32).abs() <= 1,
                "byte {byte} came back as {back}"
            );
            if byte == 0 {
                assert_eq!(back, 0, "the low end clamps");
            }
            if byte == 255 {
                assert_eq!(back, 255, "the high end clamps");
            }
        }
    }

    #[test]
    fn a_tile_inside_a_canvas_is_accepted_and_a_broadcast_or_an_overlap_is_not() {
        use crate::DevicePtr;
        use crate::tensor::{Shape, Strides};

        // One tile of a larger packed raster: 2 frames of rows × columns pixels of 3 bytes, placed
        // every 1000 bytes along the canvas's rows (a row is 96 bytes) and every 2000 rows in frames.
        let tile = TensorView {
            ptr: DevicePtr::new(0),
            dtype: DType::U8,
            shape: Shape::new(&[2, 4, 32, 3]),
            strides: Strides::new(&[4 * 1000, 1000, 3, 1]),
        };
        let outer = strides_of("test", "out", &tile).expect("a sub-region is a valid raster");
        assert_eq!(outer, [4000, 1000, 3]);

        // The same view squeezed against its neighbour: the rows would share bytes.
        let overlapping = TensorView {
            strides: Strides::new(&[4 * 90, 90, 3, 1]),
            ..tile
        };
        assert!(strides_of("test", "out", &overlapping).is_err());

        // A broadcast axis is not a raster.
        let broadcast = TensorView {
            strides: Strides::new(&[0, 1000, 3, 1]),
            ..tile
        };
        assert!(strides_of("test", "out", &broadcast).is_err());
    }

    #[test]
    fn the_bound_admits_a_tie_and_nothing_looser() {
        let tolerance = <RgbBytes as Op>::tolerance(&Descriptor {
            dtype: DType::BF16,
            channels: 3,
            block: 1024,
            pdl: false,
        });
        // A tie can land a byte either side of the boundary: 128 against 127 is the case a
        // half-byte bound would have rejected.
        assert!(tolerance.allows(127.0, 128.0));
        assert!(tolerance.allows(0.0, 1.0));
        assert!(tolerance.allows(255.0, 254.0));
        // Two bytes is a different raster, not a rounding.
        assert!(!tolerance.allows(127.0, 129.0));
        assert!(!tolerance.allows(0.0, 2.0));
    }
}
