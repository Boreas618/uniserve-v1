//! `audio_activation`: the BigVGAN decoder's activation block, as one program.
//!
//! The decoder's every residual block ends in `Activation1d(SnakeBeta(C))`
//! (`.venv/lib/python3.12/site-packages/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py:207`,
//! `:426`, `:470`; `uniserve/models/minimax_h3/audio_vae.py:297`). Written out it is five stages:
//!
//! ```text
//! 1. replicate pad the signal by `kernel/ratio - 1` on each side
//! 2. a depthwise transposed convolution, stride `ratio`, `kernel` taps, no other padding, the
//!    filter expanded over channels — the ×ratio upsample
//! 3. crop `3·(kernel - ratio)/2` samples from each end, which leaves `length · ratio` samples
//! 4. SnakeBeta:  y = x + sin²(exp(alpha)·x) / (exp(beta) + 1e-9), per channel
//! 5. replicate pad by `kernel/2 - 1` and `kernel/2`, then a depthwise convolution of `kernel` taps
//!    at stride `ratio` — the ×ratio low-pass, `floor((padded - kernel)/ratio) + 1` samples
//! ```
//!
//! Four launches and two intermediate buffers if each stage is its own kernel, and the buffers are
//! `ratio` times the signal: at 44.1 kHz stereo and 1024 channels that is the decoder's largest
//! allocation by far. One program keeps the upsampled signal inside the block that produced it and
//! writes only the final `[channels, length]` result, so this operator is the whole chain and it is
//! the only form of it the catalogue declares.
//!
//! At `ratio = 2` and `kernel = 12` — the decoder's own numbers — the constants read as the envelope
//! h3 states: replicate pad `5`, crop `15` on each side, low-pass pads `(5, 6)`. They are derived
//! here rather than written as literals, because a changed rate changes all four.
//!
//! Both lengths are launch values: an audio length follows the request's duration, so a descriptor
//! keyed by it would compile a program per duration. The descriptor carries what a program is
//! compiled from — element type, channels, kernel, ratio, block — and nothing that varies
//! continuously. The `ratio ×` factor of stage 2 is counted once, inside the upsample: a program
//! that forgets it is short by exactly `ratio`.
//!
//! The tolerance is the whole chain's: `2·kernel` taps enter the low-pass, each an FP32 product of
//! FP32 products, plus the output's own rounding — [`Tolerance::dot`] over `2·kernel + 1` terms in
//! FP32, which is H3's envelope.
//!
//! The low-pass's output length is ATen's, not a divisibility rule:
//! `floor((padded - kernel)/ratio) + 1`, so a padded span the ratio does not divide still yields a
//! length, and a span shorter than the filter has nothing to convolve and is refused by name.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The reciprocal gain's floor, as the reference writes it: `exp(beta) + 1e-9`.
pub const BETA_FLOOR: f32 = 1e-9;

/// One compilation of `audio_activation`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the signal, the filters and the output: FP32, the block's own precision.
    pub dtype: DType,
    /// Channels: the depthwise groups, one parameter each.
    pub channels: u32,
    /// Taps of both filters. The decoder uses 12 at every rate.
    pub kernel: u32,
    /// The upsample's stride and the low-pass's stride; the chain preserves the length because both
    /// use it.
    pub ratio: u32,
    /// Channels one thread block handles.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl Descriptor {
    /// The replicate pad of stage 1: `kernel/ratio - 1` samples on each side.
    pub fn upsample_pad(&self) -> u64 {
        u64::from(self.kernel / self.ratio) - 1
    }

    /// The crop of stage 3: `3·(kernel - ratio)/2` samples on each side, which leaves `length·ratio`.
    ///
    /// Derived from the transposed convolution's own length, `(length + 2p - 1)·ratio + kernel` with
    /// `p` the pad of stage 1, so a changed rate changes it with no second constant.
    pub fn crop(&self) -> u64 {
        (3 * (u64::from(self.kernel) - u64::from(self.ratio))) / 2
    }

    /// The low-pass's pads of stage 5: `kernel/2 - 1` on the left and `kernel/2` on the right.
    pub fn lowpass_pads(&self) -> (u64, u64) {
        (u64::from(self.kernel) / 2 - 1, u64::from(self.kernel) / 2)
    }

    /// Samples after stage 3, for a launch of `length` samples: `length · ratio`.
    pub fn upsampled(&self, length: u64) -> u64 {
        length * u64::from(self.ratio)
    }

    /// Samples after the low-pass, for a launch of `length` samples: the length is preserved.
    pub fn output(&self, length: u64) -> Result<u64> {
        lowpass_length(
            self.upsampled(length),
            u64::from(self.kernel),
            u64::from(self.ratio),
        )
    }
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Geometry only: a shape, a stride or a tap count, so the key names no host float and no
        // length that varies with a request.
        Key::new()
            .field("dtype", self.dtype)
            .field("channels", self.channels)
            .field("kernel", self.kernel)
            .field("ratio", self.ratio)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The low-pass's output length from an **unpadded** width: ATen's floor, and a refusal when the
/// padded span is shorter than the filter.
///
/// The pads are `kernel - 1` in total (`kernel/2 - 1` and `kernel/2`), so the span is
/// `width + kernel - 1` and the length is `floor((width - 1)/ratio) + 1`. Passing the padded span
/// here would add the pads twice, which is exactly the off-by-`kernel` the caller's run caught; the
/// two ops agree when both take the unpadded width (`audio_resample`'s 4-sample signal with pads 5
/// and 6 is a span of 15 and yields 2 here too).
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
    /// `[channels, length]` the signal entering the block.
    pub values: TensorView,
    /// `[channels]` SnakeBeta's `alpha`, in log space as the checkpoint stores it.
    pub alpha: TensorView,
    /// `[channels]` SnakeBeta's `beta`, in log space.
    pub beta: TensorView,
    /// `[1, 1, kernel]` the transposed convolution's filter, one filter shared by every channel and
    /// expanded over channels by the program.
    pub filter_up: TensorView,
    /// `[1, 1, kernel]` the low-pass's filter.
    pub filter_down: TensorView,
    /// `[channels, length · ratio]` the result; the length is preserved by the chain.
    pub out: TensorView,
    /// Samples per channel entering the block.
    pub length: u32,
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
        let taps = u64::from(descriptor.kernel);
        let ratio = u64::from(descriptor.ratio);
        let length = u64::from(args.length);
        if descriptor.kernel == 0 || descriptor.ratio == 0 || length == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a filter with no taps, a ratio of zero or a signal of no samples has no activation"
                    .to_owned(),
            ));
        }
        // The reference pads `kernel/2 - 1` and `kernel/2`, which needs an even kernel of at least
        // two taps and a kernel no shorter than the ratio; an odd kernel would put the pads off by
        // half a tap and shift the phase.
        if descriptor.kernel < 2
            || descriptor.kernel % 2 != 0
            || descriptor.kernel < descriptor.ratio
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected an even kernel of at least two taps and no shorter than the ratio \
                     {}, found {}",
                    descriptor.ratio, descriptor.kernel
                ),
            ));
        }
        let out_length = descriptor.output(length)?;

        args.values
            .require_shape(Self::NAME, Shape::new(&[channels, length]))?;
        args.out
            .require_shape(Self::NAME, Shape::new(&[channels, out_length]))?;
        for (name, view) in [
            ("values", args.values),
            ("out", args.out),
            ("filter_up", args.filter_up),
            ("filter_down", args.filter_down),
        ] {
            if view.dtype != descriptor.dtype {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected {} {name}, found {}", descriptor.dtype, view.dtype),
                ));
            }
        }
        for (name, view) in [
            ("filter_up", args.filter_up),
            ("filter_down", args.filter_down),
        ] {
            view.require_shape(Self::NAME, Shape::new(&[1, 1, taps]))?;
        }
        for (name, view) in [("alpha", args.alpha), ("beta", args.beta)] {
            view.require_shape(Self::NAME, Shape::new(&[channels]))?;
            if view.dtype != DType::F32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected f32 {name}, found {}", view.dtype),
                ));
            }
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.values)
            .view(args.alpha)
            .view(args.beta)
            .view(args.filter_up)
            .view(args.filter_down)
            .view(args.out)
            .u32(descriptor.channels)
            .u32(descriptor.kernel)
            .u32(descriptor.ratio)
            .u32(args.length)
            .u32(u32::try_from(out_length).map_err(|_| {
                Error::argument(
                    Self::NAME,
                    format!("{out_length} output samples do not fit a kernel parameter"),
                )
            })?)
            .u32(descriptor.block)
            .flag(descriptor.pdl);
        // One block owns a channel's whole length: the upsampled signal between stages never leaves
        // the block, so no intermediate buffer exists to allocate, and the low-pass's sum order is
        // fixed by the block's own walk.
        kernel.launch(
            LaunchGrid::new([
                channels,
                u64::from(args.length).div_ceil(u64::from(descriptor.block)),
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
        // leave behind — and the sum is accumulated in FP32 with one rounding into the output.
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
            block: 8,
            pdl: false,
        }
    }

    #[test]
    fn the_constants_are_the_envelopes_read_back() {
        // ratio 2, kernel 12: replicate pad 5, crop 15 a side, low-pass pads (5, 6) — h3's numbers,
        // derived rather than written.
        assert_eq!(descriptor().upsample_pad(), 5);
        assert_eq!(descriptor().crop(), 15);
        assert_eq!(descriptor().lowpass_pads(), (5, 6));
    }

    #[test]
    fn the_chain_preserves_the_length_and_the_floor_decides_the_rest() {
        // 100 samples, ×2 upsample, 12-tap low-pass at stride 2: the padded span is 211, so
        // `floor(199/2) + 1 = 100` and the block is width-preserving as the reference is.
        assert_eq!(descriptor().upsampled(100), 200);
        assert_eq!(descriptor().output(100).unwrap(), 100);
        for length in 1..64 {
            assert_eq!(descriptor().output(length).unwrap(), length);
        }
        // A span the ratio does not divide still floors, and the boundary is the filter's length:
        // the argument is the *unpadded* width and the pads are derived inside, so the two ops agree
        // -- `audio_resample`'s case is a 4-sample signal whose pads make a span of 15.
        assert_eq!(lowpass_length(4, 12, 2).unwrap(), 2);
        assert_eq!(lowpass_length(200, 12, 2).unwrap(), 100);
        assert_eq!(lowpass_length(1, 12, 2).unwrap(), 1);
        assert!(lowpass_length(0, 12, 2).is_err());
    }

    #[test]
    fn the_key_names_geometry_and_no_length_or_host_float() {
        assert_eq!(
            descriptor().to_string(),
            "dtype_f32_channels_1024_kernel_12_ratio_2_block_8_pdl_false"
        );
        // Two lengths of the same geometry are one program; a different rate is another.
        assert_eq!(descriptor(), descriptor());
        assert_ne!(
            descriptor().to_string(),
            Descriptor {
                ratio: 4,
                kernel: 16,
                ..descriptor()
            }
            .to_string()
        );
    }

    #[test]
    fn the_bound_is_the_whole_chain() {
        assert_eq!(
            AudioActivation::tolerance(&descriptor()),
            Tolerance::dot(25, DType::F32, DType::F32)
        );
        assert!(AudioActivation::tolerance(&descriptor()).rtol > 0.0);
    }
}
