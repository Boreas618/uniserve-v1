//! `pcm_interleave`: the audio decoder's last step, from two float channels to interleaved i16.
//!
//! The decoder ends with one `[samples]` float channel per side, and the artifact is 16-bit PCM with
//! the sides interleaved — left, right, left, right. Writing it is one arithmetic per sample, but it
//! is the step whose rounding is audible if it is wrong, so it is a device program with the
//! quantization written out rather than host code beside the graph:
//!
//! ```text
//! out[j, 0] = quantize(left[j])
//! out[j, 1] = quantize(right[j])
//!
//! quantize(x) = round_half_to_even(clamp(x, -1, 1) · 32767)      as i16
//! ```
//!
//! `clamp` is the reference's: an out-of-range sample is a decoder artifact clipped at full scale,
//! not a wrapping overflow. `round_half_to_even` is the reference's too — ties go to the even
//! neighbour, which is what `numpy.rint`/Rust's `f32::round_ties_even` do and what the whole-chain
//! comparison is against, so a program that rounds ties away from zero differs on exactly the half
//! samples and nowhere else.
//!
//! The sample count is a launch value, not a descriptor field: a request's duration varies while the
//! program's arithmetic does not, so one compiled module serves every clip.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// Full-scale value of the 16-bit output: the multiplier after the clamp.
pub const FULL_SCALE: f32 = 32767.0;

/// Elements one thread block converts.
pub const BLOCK: u32 = 1024;

/// One compilation of `pcm_interleave`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the two input channels: FP32, the decoder's own output.
    pub dtype: DType,
    /// Samples one thread block converts.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The output is always i16 and the sample count is a launch value, so neither is in the key.
        Key::new()
            .field("dtype", self.dtype)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one interleave.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[samples]` the left channel.
    pub left: TensorView,
    /// `[samples]` the right channel.
    pub right: TensorView,
    /// `[samples, 2]` i16 interleaved output.
    pub out: TensorView,
    /// Samples per channel.
    pub samples: u32,
}

/// Quantize two float channels into interleaved 16-bit PCM.
pub struct PcmInterleave;

impl Op for PcmInterleave {
    const NAME: &'static str = "pcm_interleave";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let samples = u64::from(args.samples);
        if samples == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a clip of no samples has nothing to interleave".to_owned(),
            ));
        }
        if descriptor.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected f32 channels, the decoder's output, found {}",
                    descriptor.dtype
                ),
            ));
        }
        for (name, view) in [("left", args.left), ("right", args.right)] {
            view.require_shape(Self::NAME, Shape::new(&[samples]))?;
            if view.dtype != DType::F32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected f32 {name}, found {}", view.dtype),
                ));
            }
        }
        args.out
            .require_shape(Self::NAME, Shape::new(&[samples, 2]))?;
        if args.out.dtype != DType::I16 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected i16 output, found {}", args.out.dtype),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.left)
            .view(args.right)
            .view(args.out)
            .u32(args.samples)
            .u32(descriptor.block)
            .flag(descriptor.pdl);
        // One block per range of samples: a sample's two outputs are written by the same block beside
        // each other, so the interleaving needs no second pass.
        kernel.launch(
            LaunchGrid::new([samples.div_ceil(u64::from(descriptor.block)), 1, 1])?,
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
        // The output is an integer the formula fixes exactly: there is no rounding to bound, only a
        // formula to reproduce tie for tie.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> Descriptor {
        Descriptor {
            dtype: DType::F32,
            block: BLOCK,
            pdl: false,
        }
    }

    #[test]
    fn the_key_holds_no_sample_count_because_a_clip_length_is_a_launch_value() {
        assert_eq!(descriptor().to_string(), "dtype_f32_block_1024_pdl_false");
    }

    #[test]
    fn the_conversion_is_exact() {
        assert_eq!(PcmInterleave::tolerance(&descriptor()), Tolerance::exact());
        assert_eq!(FULL_SCALE, 32767.0);
    }
}
