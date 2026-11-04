//! `audio_activation`: the BigVGAN decoder's activation block, as one program.
//!
//! The decoder's every residual block ends in `Activation1d(SnakeBeta(C))`
//! (`.venv/lib/python3.12/site-packages/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py:207`,
//! `:426`, `:470`; `uniserve/models/minimax_h3/audio_vae.py:297`). Written out it is five stages:
//!
//! ```text
//! 1. replicate pad the signal by `kernel/2 - 1` (left) and `kernel/2` (right)
//! 2. a depthwise transposed convolution, stride `ratio`, `kernel` taps — the ×ratio upsample
//! 3. crop `kernel/ratio - 1` samples from each end, which leaves `width · ratio` samples
//! 4. SnakeBeta:  y = x + sin²(exp(alpha)·x) / (exp(beta) + 1e-9), per channel
//! 5. replicate pad by `kernel/2 - 1` and `kernel/2`, then a depthwise convolution of `kernel`
//!    taps at stride `ratio` — the ×ratio low-pass, `floor((padded - kernel)/ratio) + 1` samples
//! ```
//!
//! Four launches and two intermediate buffers if each stage is its own kernel, and the buffers are
//! `ratio` times the signal: at 44.1 kHz stereo and 1024 channels that is the decoder's largest
//! allocation by far. One program keeps the upsampled signal inside the block that produced it and
//! writes only the final `[rows, channels, width]` result — so this operator is the whole chain, and
//! it is the only form of it the catalogue declares. The tolerance is the whole chain's: `2·kernel`
//! taps enter the low-pass, each an FP32 product of an FP32 product chain, plus the output's own
//! rounding — [`Tolerance::dot`] over `2·kernel + 1` terms in FP32, which is H3's stated envelope.
//!
//! The low-pass's output length is ATen's, not a divisibility rule:
//! `floor((padded - kernel)/ratio) + 1`, so a padded span that the ratio does not divide still
//! yields a length, and a span shorter than the filter has nothing to convolve and is refused by
//! name.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The reciprocal gain's floor, as the reference writes it: `1 / (exp(beta) + 1e-9)`'s denominator.
pub const BETA_FLOOR: f32 = 1e-9;

/// One compilation of `audio_activation`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the signal, the weights and the output.
    pub dtype: DType,
    /// Channels: the depthwise groups, one weight row each.
    pub channels: u32,
    /// Taps of both filters. The reference's decoder uses 12 for every rate.
    pub kernel: u32,
    /// The upsample's stride and the low-pass's stride; the chain is width-preserving because both
    /// use it.
    pub ratio: u32,
    /// Samples of one row before the chain, which fixes the intermediate and output lengths.
    pub width: u32,
    /// Channels one thread block handles.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl Descriptor {
    /// Samples after stage 3: `width · ratio`.
    pub fn upsampled(&self) -> u64 {
        u64::from(self.width) * u64::from(self.ratio)
    }

    /// Samples after the low-pass, by ATen's floor.
    pub fn output(&self) -> u64 {
        let kernel = u64::from(self.kernel);
        lowpass_length(self.upsampled(), kernel, u64::from(self.ratio)).unwrap_or(0)
    }
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Geometry only: every field here is a shape, a stride or a tap count, so the key names no
        // host float and a recompiled module with the same shapes reuses the same program.
        Key::new()
            .field("dtype", self.dtype)
            .field("channels", self.channels)
            .field("kernel", self.kernel)
            .field("ratio", self.ratio)
            .field("width", self.width)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The low-pass's output length: ATen's floor, and a refusal when the span is shorter than the
/// filter.
///
/// `width` is the signal entering the low-pass, `kernel` its taps, `ratio` its stride. The pads are
/// `kernel/2 - 1` and `kernel/2` for an even kernel, so the padded span is `width + kernel - 1`.
pub fn lowpass_length(width: u64, kernel: u64, ratio: u64) -> Result<u64> {
    let padded = width + kernel - 1;
    if padded < kernel {
        return Err(Error::argument(
            "audio_activation",
            format!(
                "a {width}-sample signal padded to {padded} is shorter than the {kernel}-tap \
                 filter, so the low-pass has nothing to convolve"
            ),
        ));
    }
    Ok((padded - kernel) / ratio + 1)
}

/// The arguments of one fused activation.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[rows, channels, width]` the signal entering the block.
    pub x: TensorView,
    /// `[channels, 1, kernel]` the transposed convolution's depthwise weight.
    pub up_weight: TensorView,
    /// `[channels]` its bias per channel.
    pub up_bias: TensorView,
    /// `[channels]` SnakeBeta's `alpha`, in log space as the checkpoint stores it.
    pub alpha: TensorView,
    /// `[channels]` SnakeBeta's `beta`, in log space.
    pub beta: TensorView,
    /// `[channels, 1, kernel]` the low-pass's depthwise weight.
    pub down_weight: TensorView,
    /// `[channels]` its bias per channel.
    pub down_bias: TensorView,
    /// `[rows, channels, out_width]` the result; `out_width` is [`Descriptor::output`].
    pub out: TensorView,
}

/// The decoder's upsampled SnakeBeta and its low-pass, as one program.
pub struct AudioActivation;

impl Op for AudioActivation {
    const NAME: &'static str = "audio_activation";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let channels = u64::from(descriptor.channels);
        let kernel_taps = u64::from(descriptor.kernel);
        let ratio = u64::from(descriptor.ratio);
        let width = u64::from(descriptor.width);
        if descriptor.kernel == 0 || descriptor.ratio == 0 || width == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a filter with no taps, a ratio of zero or a signal of no samples has no activation"
                    .to_owned(),
            ));
        }
        // The reference pads `kernel/2 - 1` and `kernel/2`, which needs an even kernel of at least
        // two taps; an odd kernel would put the pads off by half a tap and shift the phase.
        if descriptor.kernel < 2 || descriptor.kernel % 2 != 0 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected an even kernel of at least two taps, found {}",
                    descriptor.kernel
                ),
            ));
        }
        let out_width = lowpass_length(width * ratio, kernel_taps, ratio)?;

        let rows = args.x.shape.dims().first().copied().unwrap_or(0);
        args.x
            .require_shape(Self::NAME, Shape::new(&[rows, channels, width]))?;
        args.out
            .require_shape(Self::NAME, Shape::new(&[rows, channels, out_width]))?;
        for (name, view) in [("x", args.x), ("out", args.out)] {
            if view.dtype != descriptor.dtype {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected {} {name}, found {}", descriptor.dtype, view.dtype),
                ));
            }
        }
        for (name, view) in [
            ("up_weight", args.up_weight),
            ("down_weight", args.down_weight),
        ] {
            view.require_shape(Self::NAME, Shape::new(&[channels, 1, kernel_taps]))?;
            if view.dtype != descriptor.dtype {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected {} {name}, found {}", descriptor.dtype, view.dtype),
                ));
            }
        }
        for (name, view) in [
            ("up_bias", args.up_bias),
            ("down_bias", args.down_bias),
            ("alpha", args.alpha),
            ("beta", args.beta),
        ] {
            view.require_shape(Self::NAME, Shape::new(&[channels]))?;
            // SnakeBeta's parameters are read in FP32 whatever the signal's dtype: `exp` of a half
            // is where the reference would lose the activation's shape.
            if view.dtype != DType::F32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected f32 {name}, found {}", view.dtype),
                ));
            }
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.x)
            .view(args.up_weight)
            .view(args.up_bias)
            .view(args.alpha)
            .view(args.beta)
            .view(args.down_weight)
            .view(args.down_bias)
            .view(args.out)
            .u32(descriptor.channels)
            .u32(descriptor.kernel)
            .u32(descriptor.ratio)
            .u32(descriptor.width)
            .u32(u32::try_from(out_width).map_err(|_| {
                Error::argument(
                    Self::NAME,
                    format!("{out_width} output samples do not fit a kernel parameter"),
                )
            })?)
            .u32(descriptor.block)
            .flag(descriptor.pdl);
        // One block owns one (row, channel) and its whole width: the upsampled signal between
        // stages never leaves the block, so no intermediate buffer exists to allocate, and the
        // low-pass's sum order is fixed by the block's own walk.
        kernel.launch(
            LaunchGrid::new([
                rows,
                channels.div_ceil(u64::from(descriptor.block)).max(1),
                1,
            ])?,
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
        // `2·kernel` taps enter the low-pass — each the product chain the upsample and the activation
        // leave behind — and the sum is accumulated in FP32 with one rounding into the output's
        // dtype. That is H3's stated envelope, `rounding(2·kernel + 1, F32, F32)`.
        let terms = 2 * u64::from(descriptor.kernel) + 1;
        Tolerance::dot(terms, DType::F32, descriptor.dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> Descriptor {
        Descriptor {
            dtype: DType::F32,
            channels: 1024,
            kernel: 12,
            ratio: 2,
            width: 100,
            block: 8,
            pdl: false,
        }
    }

    #[test]
    fn the_chain_preserves_the_width_and_the_floor_decides_the_rest() {
        // 100 samples, ×2 upsample, 12-tap low-pass at stride 2: the padded span is 211, so the
        // output is `floor(199/2) + 1 = 100`.
        assert_eq!(descriptor().upsampled(), 200);
        assert_eq!(descriptor().output(), 100);
        // A span the ratio does not divide still floors, and the boundary is the filter's length.
        assert_eq!(lowpass_length(211, 12, 2).unwrap(), 100);
        assert_eq!(lowpass_length(12, 12, 2).unwrap(), 1);
        assert!(lowpass_length(11, 12, 2).is_err());
    }

    #[test]
    fn the_key_names_geometry_and_no_host_float() {
        assert_eq!(
            descriptor().to_string(),
            "dtype_f32_channels_1024_kernel_12_ratio_2_width_100_block_8_pdl_false"
        );
        // The ratio and the kernel are the program's loop bounds, so either one compiles again.
        assert_ne!(
            descriptor().to_string(),
            Descriptor {
                ratio: 4,
                ..descriptor()
            }
            .to_string()
        );
    }

    #[test]
    fn the_bound_is_the_whole_chain() {
        // `2·kernel + 1` FP32 terms: every low-pass tap is a product chain the upsample and the
        // activation leave behind, and the sum rounds once into the output.
        assert_eq!(
            AudioActivation::tolerance(&descriptor()),
            Tolerance::dot(25, DType::F32, DType::F32)
        );
        assert!(AudioActivation::tolerance(&descriptor()).rtol > 0.0);
    }
}
