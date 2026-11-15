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
//! **The arithmetic is FP32 throughout**, whatever the input's element type — a decoder emits f16
//! (its raster is `float16`, its own state buffer type), so f16 is the element type
//! this operator is declared for, with bf16 and f32 accepted because the arithmetic is the same
//! either way: the normalized pixels convert to `float32` before the de-normalization and the
//! clamp, and the same order matters here — a BF16 multiply then a BF16 add would round twice
//! inside a range of 255 worth of bytes.
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

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::{self, Key};
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::operand::Operand;
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// Channels of the packed output: the ImageNet constants' length, and the only value the reference's
/// de-normalization has constants for.
pub const CHANNELS: u32 = 3;

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
/// rounding's rule is about.
pub fn quantize_scaled(scaled: f32) -> u8 {
    // A NaN has no byte; the reference's `.clamp(0, 1)` leaves it NaN and `round()` keeps it, and its
    // conversion to `uint8` is then implementation-defined, so a launch that produces one is a
    // decoder defect rather than a rounding. Saturating to 0 here keeps the host oracle total.
    if scaled.is_nan() {
        return 0;
    }
    scaled.clamp(0.0, BYTE_SCALE).round_ties_even() as u8
}

/// Refuses a raster view that is a broadcast or that overlaps itself.
///
/// A rank-4 raster's axes are `[channels, frames, rows, columns]` on the read side and
/// `[frames, rows, columns, channels]` on the write side; the innermost is unit-strided by the
/// crate's row rule (`require_row`), so what the program needs is the three outer strides that carry
/// it between planes, rows and columns ([`outer_strides`]). Two views are refused here: one with a
/// **zero** stride on an axis (every element of that axis would be the same byte — a broadcast, not a
/// raster), and one whose axes **overlap** (`stride[i]` below `extent[i+1] · stride[i+1]`, so two
/// elements would share a byte).
///
/// A sub-region of a larger raster passes unchanged: it is only a bigger outer stride, which is what
/// the decoder's tile-in-a-canvas write needs.
fn raster(operator: &'static str, name: &str, view: &Operand) -> Result<()> {
    let rank = view.rank();
    for axis in 0..rank {
        let stride = view.stride(axis);
        if stride == 0 {
            return Err(Error::argument(
                operator,
                format!("the {name} raster has a zero stride on axis {axis}"),
            ));
        }
        if axis + 1 < rank {
            let needed = view.dim(axis + 1) * view.stride(axis + 1);
            if stride < needed {
                return Err(Error::argument(
                    operator,
                    format!(
                        "the {name} raster overlaps on axis {axis}: stride {stride} is below {needed}"
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// The three outer strides of a rank-4 raster view that [`raster`] accepted.
fn outer_strides(view: &Operand) -> [u64; 3] {
    [view.stride(0), view.stride(1), view.stride(2)]
}

/// One compilation of `rgb_bytes`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the normalized input raster: the decoder's own (f16, bf16 or f32).
    pub dtype: DType,
    /// Channels: it must be [`CHANNELS`], the constants' length.
    pub channels: u32,
    /// Elements one thread block converts.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl Descriptor {
    /// The packing of a `dtype` raster into [`CHANNELS`] bytes a pixel, over the operators'
    /// block ([`descriptor::BLOCK`]).
    pub const fn new(dtype: DType, pdl: bool) -> Self {
        Self {
            dtype,
            channels: CHANNELS,
            block: descriptor::BLOCK,
            pdl,
        }
    }
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("channels", self.channels)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `rgb_bytes` launch.
#[derive(Clone)]
pub struct Args {
    /// `[channels, frames, rows, columns]` the normalized raster.
    pub input: Operand,
    /// `[channels]` f32: each channel's ImageNet mean, the addend of the de-normalization.
    pub mean: Operand,
    /// `[channels]` f32: each channel's ImageNet standard deviation, the multiplier.
    pub std: Operand,
    /// `[frames, rows, columns, channels]` u8: the packed image bytes.
    pub out: Operand,
}

/// The normalized raster to packed 8-bit RGB.
pub struct RgbBytes;

impl Op for RgbBytes {
    const NAME: &'static str = proto::module::operator::RGB_BYTES;

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        if descriptor.channels != CHANNELS {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the de-normalization has {CHANNELS} constants, the descriptor says {} channels",
                    descriptor.channels
                ),
            ));
        }
        if !matches!(descriptor.dtype, DType::F32 | DType::BF16 | DType::F16) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected an f16, bf16 or f32 raster, the descriptor says {}",
                    abi::name(descriptor.dtype)
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
                    abi::name(descriptor.dtype),
                    abi::name(args.input.dtype)
                ),
            ));
        }
        let frames = args.input.dim(1);
        let rows = args.input.dim(2);
        let columns = args.input.dim(3);
        if args.input.dim(0) != channels {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the raster's first axis is {}, the descriptor says {channels} channels",
                    args.input.dim(0)
                ),
            ));
        }
        // The program reads along the columns axis, so that axis is unit-stride; the channel-major
        // outer axes may be strided.
        args.input
            .require_row(Self::NAME, descriptor.dtype, columns)?;

        for (name, view) in [("mean", &args.mean), ("std", &args.std)] {
            view.require_shape(Self::NAME, &[channels])?;
            if view.dtype != DType::F32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected f32 {name}, found {}", abi::name(view.dtype)),
                ));
            }
        }

        args.out.require_rank(Self::NAME, 4)?;
        if args.out.dtype != DType::U8 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected u8 output, found {}", abi::name(args.out.dtype)),
            ));
        }
        args.out
            .require_shape(Self::NAME, &[frames, rows, columns, channels])?;
        // The packed image is contiguous, and the program writes it as a flat run of pixels.
        args.out.require_row(Self::NAME, DType::U8, channels)?;

        // The program walks `out` as a block of a larger raster and `input` as a slice of the
        // latent through their outer strides, so neither may broadcast or overlap.
        raster(Self::NAME, "input", &args.input)?;
        raster(Self::NAME, "out", &args.out)?;
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let frames = args.input.dim(1);
        let rows = args.input.dim(2);
        let columns = args.input.dim(3);
        // A view carries its base address into the launch, not its strides, so a strided operand has
        // to hand them over: the program walks `out` as a block of a larger raster and `input` as a
        // slice of the latent, and both are only correct if it is told how far apart the axes are.
        let input = outer_strides(&args.input);
        let out = outer_strides(&args.out);

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("input", &args.input)
            .view("mean", &args.mean)
            .view("std", &args.std)
            .view("out", &args.out)
            .u64("frames", frames)
            .u64("rows", rows)
            .u64("columns", columns)
            .u32("channels", descriptor.channels)
            .u64("in0", input[0])
            .u64("in1", input[1])
            .u64("in2", input[2])
            .u64("out0", out[0])
            .u64("out1", out[1])
            .u64("out2", out[2]);
        kernel.launch(
            LaunchGrid::linear(args.out.numel().div_ceil(u64::from(descriptor.block)))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
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
        let descriptor = Descriptor::new(DType::BF16, false);
        assert_eq!(
            crate::descriptor::key(&descriptor),
            "dtype_bf16_channels_3_block_1024_pdl_false"
        );
        // The geometry is not in the key, so a resolution or a clip length cannot make a new module.
        assert_eq!(
            crate::descriptor::key(&descriptor),
            crate::descriptor::key(&Descriptor::new(DType::BF16, false))
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
        use platform::DevicePtr;

        let view = |strides: &[u64]| {
            Operand::with_strides(
                DevicePtr::from_address(0),
                DType::U8,
                &[2, 4, 32, 3],
                strides,
            )
        };

        // One tile of a larger packed raster: 2 frames of rows × columns pixels of 3 bytes, placed
        // every 1000 bytes along the canvas's rows (a row is 96 bytes) and every 2000 rows in frames.
        let tile = view(&[4 * 1000, 1000, 3, 1]);
        raster("test", "out", &tile).expect("a sub-region is a valid raster");
        assert_eq!(outer_strides(&tile), [4000, 1000, 3]);

        // The same view squeezed against its neighbour: the rows would share bytes.
        let overlapping = view(&[4 * 90, 90, 3, 1]);
        assert!(raster("test", "out", &overlapping).is_err());

        // A broadcast axis is not a raster.
        let broadcast = view(&[0, 1000, 3, 1]);
        assert!(raster("test", "out", &broadcast).is_err());
    }

    #[test]
    fn the_bound_admits_a_tie_and_nothing_looser() {
        let tolerance = <RgbBytes as Op>::tolerance(&Descriptor::new(DType::BF16, false));
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
