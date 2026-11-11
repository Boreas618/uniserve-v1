//! The diagonal Gaussian posterior of a VAE encoder and the sample drawn from it
//! (`ops::posterior`).
//!
//! An encoder's moments hold the posterior's mean and log-variance as the two halves of their
//! channel axis. The sample is
//!
//! ```text
//! sample = round_to_f16(mean + exp(0.5 · clamp(log_variance, -30, 20)) · noise)
//! ```
//!
//! in FP32 with the FP16 rounding as a step of its own, the reference's `DiagonalGaussian.sample()`
//! under its autocast. The moments, the draw and the sample are channel-major rasters
//! `[channels, ...]`, packed; the operator reads each through two strides, so the two halves of the
//! moments are one buffer cut by an offset and nothing is copied or transposed.

use ops::posterior::{self, Posterior as PosteriorOp};
use ops::{Operand, Shape, Strides};
use uniserve_tensor::DType;

use crate::error::{Error, Result};
use crate::forward::Forward;

/// The posterior sample of a VAE encoder's moments.
#[derive(Clone, Copy, Default)]
pub struct Posterior;

impl Posterior {
    /// A sample drawn from `moments` and the caller's standard-normal `noise`: a fresh packed FP32
    /// `[channels, ...]` raster.
    ///
    /// `moments` is `[2 · channels, ...]`, the mean's channels then the log-variance's; `noise` is
    /// `[channels, ...]` of the same trailing extents. Both are packed FP32.
    ///
    /// # Errors
    ///
    /// Rasters of another element type, rank, channel count or trailing extent, a strided raster,
    /// an arena that cannot hold the sample, or the operator's refusal.
    pub fn sample(&self, f: &Forward, moments: Operand, noise: Operand) -> Result<Operand> {
        const OWNER: &str = "Posterior";
        let channels = noise.shape.dims().first().copied().unwrap_or(0);
        if channels == 0 {
            return Err(Error::shape(
                OWNER,
                format!("a sample of {:?} has no channels", noise.shape.dims()),
            ));
        }
        let out = f.empty(DType::F32, noise.shape)?;
        for (name, view, expected) in [
            ("moments", moments, 2 * channels),
            ("noise", noise, channels),
        ] {
            if view.dtype != DType::F32 {
                return Err(Error::shape(
                    OWNER,
                    format!(
                        "the {name} raster is {}, and a posterior is FP32",
                        ops::abi::name(view.dtype)
                    ),
                ));
            }
            if view.shape.rank() < 2
                || view.shape.dim(0) != expected
                || view.shape.dims()[1..] != out.shape.dims()[1..]
            {
                return Err(Error::shape(
                    OWNER,
                    format!(
                        "the {name} raster is {:?}, not [{expected}, {:?}]",
                        view.shape.dims(),
                        &out.shape.dims()[1..]
                    ),
                ));
            }
            if view.strides != Strides::row_major(view.shape) {
                return Err(Error::shape(
                    OWNER,
                    format!(
                        "the {name} raster's strides are {:?} over {:?}: a channel-major raster is \
                         read packed through two strides",
                        view.strides.as_slice(),
                        view.shape.dims()
                    ),
                ));
            }
        }
        // Each raster as the operator's `[positions, channels]`: neighbouring positions are
        // neighbouring elements and one channel is one whole plane away, and the moments' two halves
        // are exactly `channels` planes apart.
        let positions = out.shape.dims()[1..].iter().product::<u64>();
        let view = |ptr| {
            Operand::with_strides(
                ptr,
                DType::F32,
                Shape::new(&[positions, channels]),
                Strides::new(&[1, positions]),
            )
        };
        let descriptor = posterior::Descriptor {
            block: posterior::BLOCK,
            pdl: false,
        };
        f.operate::<PosteriorOp>(
            &descriptor,
            posterior::Args {
                mean: view(moments.ptr),
                log_variance: view(moments.offset_elements(channels * positions)),
                noise: view(noise.ptr),
                out: view(out.ptr),
            },
        )?;
        Ok(out)
    }
}
