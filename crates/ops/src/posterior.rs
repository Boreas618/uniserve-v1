//! `posterior`: the VAE encoder's moments to the sample a condition reads, through its FP16 round trip.
//!
//! The condition encoder does not condition on the encoder's moments; it conditions on a *sample*
//! drawn from them and normalised. The chain, in order:
//!
//! ```text
//! log_variance = clamp(log_variance, -30, 20)
//! sample       = mean + exp(0.5 · log_variance) · noise
//! out          = round_to_f16(sample)                       // the FP16 round trip, stored as f32
//! ```
//!
//! Each step rounds in f32 and the round trip rounds again, so there are five roundings and the order
//! is the specification: `exp` after the clamp and the round trip after the sum are the two a fused
//! implementation drops first, and both change the tensor rather than its last bit. The clamp's bounds
//! are the reference's constants; `-30` is why an encoder variance of `1e-40` still conditions.
//!
//! The **per-channel affine is not here**. Normalising by `latents_std` and `latents_mean` is a
//! diagonal `1×1` convolution — the same layer a decoder composes for exactly this arithmetic — and
//! an operator that fused it would make this kernel's arithmetic depend on a weight layout. This
//! operator ends at the round trip; the caller applies the affine.
//!
//! `tolerance` is exact, on `cross_fade`'s precedent: the arithmetic above *is* the specification, so
//! two implementations of it agree bit for bit or one of them is wrong.

use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::operand::Operand;
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// Elements one thread block produces.
pub const BLOCK: u32 = 1024;

/// The lower clamp of the log-variance, as the reference states it.
pub const CLAMP_MIN: f32 = -30.0;

/// The upper clamp of the log-variance, as the reference states it.
pub const CLAMP_MAX: f32 = 20.0;

/// One compilation of `posterior`.
///
/// The arithmetic is defined on f32 and on nothing else: the moments are the encoder's own f32 output
/// and the sample is stored as f32 after the round trip, so the element type is not a choice a caller
/// makes and is not in the key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Elements one thread block produces.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one sample.
///
/// All four views are `[positions, channels]` and every one of them is read or written through **its
/// own two strides**. The reference's rasters are channel-major (`[channels, frames, rows, columns]`,
/// because that is what every convolution reads), so a caller there passes a row stride of one and a
/// channel stride of the position count; a channels-last caller passes the same two words the other way
/// round. Either way the moments' two halves are one buffer cut by an offset, with no transpose and no
/// second buffer, which is what the strides are in the signature for.
#[derive(Clone)]
pub struct Args {
    /// `[positions, channels]` f32: the mean half of the moments.
    pub mean: Operand,
    /// `[positions, channels]` f32: the log-variance half, clamped inside the kernel.
    pub log_variance: Operand,
    /// `[positions, channels]` f32: the draw, one per element.
    pub noise: Operand,
    /// `[positions, channels]` f32: the sample after the FP16 round trip.
    pub out: Operand,
}

/// The posterior sample of the encoder's moments.
pub struct Posterior;

impl Op for Posterior {
    const NAME: &'static str = proto::module::operator::POSTERIOR;

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn check(_descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        if args.mean.rank() != 2 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected [positions, channels] moments, found the extents {:?}",
                    args.mean.dims()
                ),
            ));
        }
        let positions = args.mean.dim(0);
        let channels = args.mean.dim(1);
        if channels == 0 || positions == 0 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a sample of {positions} positions by {channels} channels computes nothing"
                ),
            ));
        }
        // The element type and the channel count, but not the innermost stride: a channel-major caller
        // advances by the position count between channels, and refusing that is what forced three
        // transposes before these words existed.
        for (name, view) in [
            ("mean", &args.mean),
            ("log_variance", &args.log_variance),
            ("noise", &args.noise),
            ("out", &args.out),
        ] {
            view.require_dims(Self::NAME, args.mean.dims())?;
            if view.dtype != DType::F32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "expected {} {name}, found {}",
                        abi::name(DType::F32),
                        abi::name(view.dtype)
                    ),
                ));
            }
            if view.dim(1) != channels {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "expected {channels} channels in the {name}, found {}",
                        view.dim(1)
                    ),
                ));
            }
        }
        // Each view is addressed through two strides, and the pair must not have two elements of the
        // block land on one address: either the channels are packed inside a position (a row of at
        // least `channels · channel stride`) or the positions are packed inside a channel (a channel
        // stride of at least `positions · row stride`). Both physical layouts are legal; a stride pair
        // that overlaps is not, because which write won would depend on the block's order.
        for (name, view) in [
            ("mean", &args.mean),
            ("log_variance", &args.log_variance),
            ("noise", &args.noise),
            ("out", &args.out),
        ] {
            let row = view.stride(0);
            let channel = view.stride(1);
            let packed_channels = row >= channels * channel;
            let packed_positions = channel >= positions * row;
            if !packed_channels && !packed_positions {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the {name} advances by {row} elements per position and {channel} per channel \
                         over {positions} positions and {channels} channels: no single element per \
                         (position, channel) fits, so two of them would share an address"
                    ),
                ));
            }
        }
        u32::try_from(channels).map_err(|_| {
            Error::argument(
                Self::NAME,
                format!("{channels} channels do not fit a kernel parameter"),
            )
        })?;
        u32::try_from(args.mean.stride(0)).map_err(|_| {
            Error::argument(
                Self::NAME,
                format!(
                    "the mean's row stride {} does not fit a kernel parameter",
                    args.mean.stride(0)
                ),
            )
        })?;
        u32::try_from(args.mean.stride(1)).map_err(|_| {
            Error::argument(
                Self::NAME,
                format!(
                    "the mean's channel stride {} does not fit a kernel parameter",
                    args.mean.stride(1)
                ),
            )
        })?;
        u32::try_from(args.log_variance.stride(0)).map_err(|_| {
            Error::argument(
                Self::NAME,
                format!(
                    "the log-variance's row stride {} does not fit a kernel parameter",
                    args.log_variance.stride(0)
                ),
            )
        })?;
        u32::try_from(args.log_variance.stride(1)).map_err(|_| {
            Error::argument(
                Self::NAME,
                format!(
                    "the log-variance's channel stride {} does not fit a kernel parameter",
                    args.log_variance.stride(1)
                ),
            )
        })?;
        u32::try_from(args.noise.stride(0)).map_err(|_| {
            Error::argument(
                Self::NAME,
                format!(
                    "the noise's row stride {} does not fit a kernel parameter",
                    args.noise.stride(0)
                ),
            )
        })?;
        u32::try_from(args.noise.stride(1)).map_err(|_| {
            Error::argument(
                Self::NAME,
                format!(
                    "the noise's channel stride {} does not fit a kernel parameter",
                    args.noise.stride(1)
                ),
            )
        })?;
        u32::try_from(args.out.stride(0)).map_err(|_| {
            Error::argument(
                Self::NAME,
                format!(
                    "the sample's row stride {} does not fit a kernel parameter",
                    args.out.stride(0)
                ),
            )
        })?;
        u32::try_from(args.out.stride(1)).map_err(|_| {
            Error::argument(
                Self::NAME,
                format!(
                    "the sample's channel stride {} does not fit a kernel parameter",
                    args.out.stride(1)
                ),
            )
        })?;
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let positions = args.mean.dim(0);
        let channels = args.mean.dim(1);

        // Every narrowing below is of a value `check` holds within its kernel parameter.
        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("mean_ptr", &args.mean)
            .view("log_variance_ptr", &args.log_variance)
            .view("noise_ptr", &args.noise)
            .view("out_ptr", &args.out)
            .u64("channels", channels)
            .u64("mean_row", args.mean.stride(0))
            .u64("mean_channel", args.mean.stride(1))
            .u64("log_variance_row", args.log_variance.stride(0))
            .u64("log_variance_channel", args.log_variance.stride(1))
            .u64("noise_row", args.noise.stride(0))
            .u64("noise_channel", args.noise.stride(1))
            .u64("out_row", args.out.stride(0))
            .u64("out_channel", args.out.stride(1))
            .u64("total", positions * channels);
        kernel.launch(
            LaunchGrid::linear((positions * channels).div_ceil(u64::from(descriptor.block)))?,
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
        // Five f32 roundings and an FP16 round trip, in one order the reference states: one
        // implementation reproduces its bits, so the comparison is equality rather than a bound.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> Descriptor {
        Descriptor {
            block: BLOCK,
            pdl: false,
        }
    }

    #[test]
    fn the_key_names_the_tiling_and_nothing_a_caller_chooses() {
        assert_eq!(descriptor().to_string(), "block_1024_pdl_false");
        // A different tiling is a different program.
        assert_ne!(
            descriptor().to_string(),
            Descriptor {
                block: 512,
                ..descriptor()
            }
            .to_string()
        );
    }
}
