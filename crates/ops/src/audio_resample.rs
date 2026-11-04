//! The audio decoder's anti-aliased up- and downsampler, pad and crop included.
//!
//! Both directions are one depthwise convolution with a Kaiser-windowed sinc filter whose taps sum
//! to one (`kaiser_sinc_filter1d`), and both reproduce the reference module's replicate pad
//! exactly, because the pad is what keeps the resampler from ringing at the ends of a waveform
//! (`autoencoder_kl_minimax_h3_audio.py:151-193`):
//!
//! * [`Mode::Downsample`] pads `pad_left = kernel/2 - (kernel even)` on the left and `pad_right =
//!   kernel/2` on the right, replicates, and convolves with stride `ratio`, one filter per channel.
//!   Its output length is ATen's `floor((padded - kernel)/ratio) + 1`; the span need not divide, and
//!   a length that differs from that floor is refused.
//! * [`Mode::Upsample`] pads `pad = kernel/ratio - 1` on both sides, replicates, applies the
//!   transposed convolution of stride `ratio` scaled by `ratio`, and crops `pad_left = pad·ratio +
//!   (kernel - ratio)/2` from the front and `pad_right = pad·ratio + (kernel - ratio + 1)/2` from the
//!   end, which leaves exactly `width · ratio` samples.
//!
//! It is one operator with a mode rather than two, because everything the caller supplies has the
//! same shape in both: the signal, the shared filter, the output, the pad amounts. The two
//! convolutions differ in the kernel program, and the descriptor names which one was compiled.
//!
//! The program is a kernel program (specs/worker.md 9 fuses pad, upsample, SnakeBeta and downsample
//! into one program where the activation sits between them; each of the four is also its own
//! operator, which is what this one is). Parameters, in order: `x`, `filter`, `out`, `rows`, `width`,
//! `out_width`, `pad`, `pad_left`, `pad_right`, the PDL flag. The filter is FP32 and shared by every
//! channel; the kernel expands it, so the caller never materialises `[channels, 1, kernel]`.
//!
//! The tolerance is a `kernel`-term weighted sum in FP32: the taps carry signs, so the bound is
//! [`Tolerance::dot`] rather than the plain sum bound, and the taps' sum of one keeps the result
//! from growing beyond the signal's own range.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, Strides, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// Which of the two resamplers a compilation holds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mode {
    /// A transposed depthwise convolution of stride `ratio`, scaled by `ratio`, then cropped.
    Upsample,
    /// A depthwise convolution of stride `ratio` behind a Kaiser-sinc low-pass.
    Downsample,
}

impl fmt::Display for Mode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Upsample => "upsample",
            Self::Downsample => "downsample",
        })
    }
}

/// One compilation of `audio_resample`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Which direction this module resamples.
    pub mode: Mode,
    /// Element type of the signal and of the output.
    pub dtype: DType,
    /// Channels of the signal: the depth of the depthwise convolution.
    pub channels: u32,
    /// Taps of the shared Kaiser-sinc filter.
    pub kernel: u32,
    /// Samples out per sample in.
    pub ratio: u32,
    /// Samples one thread block produces.
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
                    Mode::Upsample => "upsample",
                    Mode::Downsample => "downsample",
                },
            )
            .field("dtype", self.dtype)
            .field("channels", self.channels)
            .field("kernel", self.kernel)
            .field("ratio", self.ratio)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one resample.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[rows, channels, width]`, row-major, the signal before its pad.
    pub x: TensorView,
    /// `[kernel]` FP32, the Kaiser-sinc taps, the same for every channel.
    pub filter: TensorView,
    /// `[rows, channels, out_width]`, row-major, after the crop.
    pub out: TensorView,
    /// The pad each side of `x` when upsampling; unused when downsampling.
    pub pad: u32,
    /// The samples the crop drops at the front of the transposed convolution's output.
    pub pad_left: u32,
    /// The samples the crop drops at the end.
    pub pad_right: u32,
}

/// The anti-aliased resampler of the audio decoder.
pub struct AudioResample;

impl Op for AudioResample {
    const NAME: &'static str = "audio_resample";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let channels = u64::from(descriptor.channels);
        let kernel_size = u64::from(descriptor.kernel);
        let ratio = u64::from(descriptor.ratio);
        if ratio == 0 || kernel_size == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a resampler needs a nonzero ratio and filter length",
            ));
        }
        if kernel_size < ratio {
            return Err(Error::argument(
                Self::NAME,
                format!("a filter of {kernel_size} taps cannot span a ratio of {ratio}"),
            ));
        }

        args.x.require_rank(Self::NAME, 3)?;
        let (rows, x_channels, width) = (
            args.x.shape.dim(0),
            args.x.shape.dim(1),
            args.x.shape.dim(2),
        );
        if x_channels != channels {
            return Err(Error::argument(
                Self::NAME,
                format!("expected {channels} channels, found {x_channels}"),
            ));
        }
        args.x.require_row(Self::NAME, descriptor.dtype, width)?;
        if args.x.strides != Strides::row_major(args.x.shape) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the signal is read row-major; found strides {}",
                    args.x.strides
                ),
            ));
        }

        args.filter
            .require_shape(Self::NAME, Shape::new(&[kernel_size]))?;
        args.filter
            .require_row(Self::NAME, DType::F32, kernel_size)?;

        args.out.require_rank(Self::NAME, 3)?;
        let out_width = args.out.shape.dim(2);
        if args.out.shape.dim(0) != rows || args.out.shape.dim(1) != channels {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected [{rows}, {channels}, {out_width}], found {:?}",
                    args.out.shape.dims()
                ),
            ));
        }
        args.out
            .require_row(Self::NAME, descriptor.dtype, out_width)?;
        if args.out.strides != Strides::row_major(args.out.shape) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the output is written row-major; found strides {}",
                    args.out.strides
                ),
            ));
        }

        // Each mode's length follows from the reference's own formula; a mismatch is a caller that
        // built the wrong signal for this module, and is worth refusing here rather than cropping on
        // device.
        let expected = expected_width(
            Self::NAME,
            descriptor.mode,
            width,
            kernel_size,
            ratio,
            args.pad,
            args.pad_left,
            args.pad_right,
        )?;
        if out_width != expected {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} of {width} samples produces {expected} samples, and the output has \
                     {out_width}",
                    descriptor.mode
                ),
            ));
        }
        // The crop takes exactly what the pad and the filter add, `2·pad·ratio + kernel - ratio`:
        // the transposed convolution of the padded signal is `width·ratio` samples longer than that,
        // so any other crop keeps a different window of samples than the reference does.
        if descriptor.mode == Mode::Upsample
            && u64::from(args.pad_left) + u64::from(args.pad_right)
                != 2 * u64::from(args.pad) * ratio + kernel_size - ratio
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "an upsample of {width} samples with a pad of {} and a filter of \
                     {kernel_size} crops {} from the front and {} from the end, which does not leave \
                     {out_width} samples",
                    args.pad, args.pad_left, args.pad_right
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.x)
            .view(args.filter)
            .view(args.out)
            .u64(rows)
            .u32(descriptor.channels)
            .u32(u32_of(Self::NAME, width, "the width")?)
            .u32(u32_of(Self::NAME, out_width, "the output width")?)
            .u32(args.pad)
            .u32(args.pad_left)
            .u32(args.pad_right)
            .flag(descriptor.pdl);
        // One thread per output sample of one row: the padding and the crop live in the index.
        kernel.launch(
            LaunchGrid::covering(rows * out_width, u64::from(descriptor.block))?,
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
        Tolerance::dot(u64::from(descriptor.kernel), DType::F32, descriptor.dtype)
    }
}

/// The output length each mode produces, from the reference module's own arithmetic.
///
/// `Mode::Upsample` returns `width · ratio`: the transposed convolution of the padded signal is
/// `2 · pad · ratio + kernel - ratio` samples longer than that, and the crop takes exactly that much,
/// which is why the crop is checked here rather than trusted.
///
/// `Mode::Downsample` returns ATen's `floor((padded - kernel) / ratio) + 1`. The span need not be a
/// multiple of the ratio: the alias-free activation composes a 12-tap filter at ratio 2 over a
/// padded span of `2L - 1`, and PyTorch floors that to `L` rather than refusing.
#[allow(clippy::too_many_arguments)]
fn expected_width(
    operator: &'static str,
    mode: Mode,
    width: u64,
    kernel: u64,
    ratio: u64,
    pad: u32,
    pad_left: u32,
    pad_right: u32,
) -> Result<u64> {
    match mode {
        Mode::Upsample => {
            // The crop must take exactly what the pad and the filter add, or the samples the output
            // carries are not the ones the reference's crop keeps.
            let crop = u64::from(pad_left) + u64::from(pad_right);
            let expected_crop = 2 * u64::from(pad) * ratio + kernel - ratio;
            if crop != expected_crop {
                return Err(Error::argument(
                    operator,
                    format!(
                        "an upsample of {width} samples with a pad of {pad} and a filter of {kernel} \
                         crops {pad_left} from the front and {pad_right} from the end, {crop} in \
                         all, where the transposed convolution adds {expected_crop}"
                    ),
                ));
            }
            Ok(width * ratio)
        }
        Mode::Downsample => {
            let padded = width + u64::from(pad_left) + u64::from(pad_right);
            if padded < kernel {
                return Err(Error::argument(
                    operator,
                    format!(
                        "{width} samples with a pad of {pad_left} and {pad_right} are shorter than \
                         the filter"
                    ),
                ));
            }
            Ok((padded - kernel) / ratio + 1)
        }
    }
}

/// A host extent as the `u32` a kernel parameter holds.
fn u32_of(operator: &'static str, value: u64, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        Error::argument(
            operator,
            format!("{what} of {value} elements does not fit a kernel parameter"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> Descriptor {
        Descriptor {
            mode: Mode::Downsample,
            dtype: DType::F32,
            channels: 1536,
            kernel: 12,
            ratio: 6,
            block: 256,
            pdl: false,
        }
    }

    #[test]
    fn the_key_states_the_mode_the_filter_and_the_ratio() {
        assert_eq!(
            descriptor().to_string(),
            "mode_downsample_dtype_f32_channels_1536_kernel_12_ratio_6_block_256_pdl_false"
        );
        assert_eq!(
            Descriptor {
                mode: Mode::Upsample,
                ..descriptor()
            }
            .to_string(),
            "mode_upsample_dtype_f32_channels_1536_kernel_12_ratio_6_block_256_pdl_false"
        );
    }

    #[test]
    fn an_upsample_crops_exactly_what_the_transposed_convolution_adds() {
        // ratio 2, 12 taps: the pad is `kernel/ratio - 1 = 5` on each side and the crop
        // `2·5·2 + 12 - 2 = 30`, which is what leaves `width · ratio` samples.
        let width = 100;
        assert_eq!(
            expected_width("audio_resample", Mode::Upsample, width, 12, 2, 5, 15, 15).unwrap(),
            width * 2
        );
        // A crop that takes a different window is refused, not silently accepted.
        assert!(expected_width("audio_resample", Mode::Upsample, width, 12, 2, 5, 14, 15).is_err());
    }

    #[test]
    fn a_downsample_length_floors_where_the_span_does_not_divide() {
        // The alias-free activation composes exactly this: a 12-tap filter at ratio 2 over a
        // padded signal of 211 samples (`5` and `6` of pad on `2L = 200`), a span of 199.
        // PyTorch floors to `L = 100`; a refusal of the span would reject the reference's own
        // resampler.
        assert_eq!(
            expected_width("audio_resample", Mode::Downsample, 200, 12, 2, 0, 5, 6).unwrap(),
            100
        );
        // A span shorter than the filter has nothing to convolve.
        assert!(expected_width("audio_resample", Mode::Downsample, 4, 12, 2, 0, 5, 6).is_err());
    }

    #[test]
    fn both_directions_are_a_filter_length_dot_in_fp32() {
        let tolerance = AudioResample::tolerance(&descriptor());
        assert_eq!(tolerance.atol, 0.0);
        // `kernel` terms, each an FP32 product, plus the output's own rounding.
        assert!(tolerance.rtol > 0.0);
        assert_eq!(
            tolerance.rtol,
            Tolerance::dot(12, DType::F32, DType::F32).rtol
        );
    }
}
