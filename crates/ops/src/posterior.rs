//! `posterior`: the VAE encoder's moments to the sample a condition reads, through its FP16 round trip.
//!
//! The condition encoder does not condition on the encoder's moments; it conditions on a *sample*
//! drawn from them and normalised, the same tensor the reference's sampler produces. The chain, in the
//! reference's order (`uniserve/nn/vae/layers.py:134-203`, `uniserve/nn/vae/encoder.py:88-108`):
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
//! diagonal `1×1` convolution — the same layer the audio decoder already composes for exactly this
//! arithmetic — and an operator that fused it would make this kernel's arithmetic depend on a weight
//! layout. This operator ends at the round trip; the caller applies the affine.
//!
//! `tolerance` is exact, on `cross_fade`'s precedent: the arithmetic above *is* the specification, so
//! two implementations of it agree bit for bit or one of them is wrong.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, TensorView};
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
/// All four views are `[positions, channels]` and every one of them is read or written through its own
/// row stride: the two moments are the halves of the encoder's `[positions, 2 · channels]` output, cut
/// on the channel axis, so their rows are twice as wide as the sample's.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[positions, channels]` f32: the mean half of the moments.
    pub mean: TensorView,
    /// `[positions, channels]` f32: the log-variance half, clamped inside the kernel.
    pub log_variance: TensorView,
    /// `[positions, channels]` f32: the draw, one per element.
    pub noise: TensorView,
    /// `[positions, channels]` f32: the sample after the FP16 round trip.
    pub out: TensorView,
}

/// The posterior sample of the encoder's moments.
pub struct Posterior;

impl Op for Posterior {
    const NAME: &'static str = "posterior";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        if args.mean.shape.rank() != 2 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected [positions, channels] moments, found the extents {:?}",
                    args.mean.shape.dims()
                ),
            ));
        }
        let positions = args.mean.shape.dim(0);
        let channels = args.mean.shape.dim(1);
        if channels == 0 || positions == 0 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a sample of {positions} positions by {channels} channels computes nothing"
                ),
            ));
        }
        for view in [args.mean, args.log_variance, args.noise] {
            view.require_shape(Self::NAME, args.mean.shape)?;
            view.require_row(Self::NAME, DType::F32, channels)?;
        }
        args.out.require_shape(Self::NAME, args.mean.shape)?;
        args.out.require_row(Self::NAME, DType::F32, channels)?;
        // Every view is addressed a row at a time, so a row may be wider than the sample but not
        // narrower: a row stride under `channels` would have two rows share channels.
        for (name, view) in [
            ("mean", args.mean),
            ("log_variance", args.log_variance),
            ("noise", args.noise),
            ("out", args.out),
        ] {
            let row = view.strides.get(0);
            if row < channels || !row.is_multiple_of(channels) {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the {name}'s row advances by {row} elements for {channels} channels: a row \
                         must hold whole samples, and a moments half advances by twice its channels"
                    ),
                ));
            }
        }
        let mut arguments = Arguments::new();
        arguments
            .view(args.mean)
            .view(args.log_variance)
            .view(args.noise)
            .view(args.out)
            .u32(u32::try_from(channels).map_err(|_| {
                Error::argument(
                    Self::NAME,
                    format!("{channels} channels do not fit a kernel parameter"),
                )
            })?)
            .u32(u32::try_from(args.mean.strides.get(0)).map_err(|_| {
                Error::argument(
                    Self::NAME,
                    format!(
                        "the mean's row stride {} does not fit a kernel parameter",
                        args.mean.strides.get(0)
                    ),
                )
            })?)
            .u32(
                u32::try_from(args.log_variance.strides.get(0)).map_err(|_| {
                    Error::argument(
                        Self::NAME,
                        format!(
                            "the log-variance's row stride {} does not fit a kernel parameter",
                            args.log_variance.strides.get(0)
                        ),
                    )
                })?,
            )
            .u32(u32::try_from(args.noise.strides.get(0)).map_err(|_| {
                Error::argument(
                    Self::NAME,
                    format!(
                        "the noise's row stride {} does not fit a kernel parameter",
                        args.noise.strides.get(0)
                    ),
                )
            })?)
            .u32(u32::try_from(args.out.strides.get(0)).map_err(|_| {
                Error::argument(
                    Self::NAME,
                    format!(
                        "the sample's row stride {} does not fit a kernel parameter",
                        args.out.strides.get(0)
                    ),
                )
            })?)
            .u64(positions * channels)
            .flag(descriptor.pdl);
        kernel.launch(
            LaunchGrid::linear((positions * channels).div_ceil(u64::from(descriptor.block)))?,
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
