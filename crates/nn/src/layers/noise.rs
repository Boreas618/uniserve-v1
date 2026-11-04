//! The initial noise of a video diffusion request, drawn on the device.
//!
//! A request's noise is a function of the request's sampling key and the element's position, so the
//! same request regenerates the same latents without the host holding a sample (specs/worker.md 9,
//! "Philox"; `ops::noise`). The layer holds only the element type and the tile; the key is the
//! caller's, because the executor computes it from the request seed and the phase.

use ops::TensorView;
use ops::noise::{self, NoiseKind, PhiloxNoise as NoiseOp};

use crate::error::{Error, Result};
use crate::forward::Forward;

/// The elements one noise program draws.
const BLOCK: u32 = noise::NOISE_BLOCK;

/// Noise addressed by (sampling key, element): standard normal, or uniform in `[0, 1)`.
pub struct Noise {
    kind: NoiseKind,
}

impl Default for Noise {
    fn default() -> Self {
        Self::new()
    }
}

impl Noise {
    /// The standard-normal draw, the initial noise of a request.
    pub const fn new() -> Self {
        Self {
            kind: NoiseKind::Normal,
        }
    }

    /// The same stream drawn as uniforms in `[0, 1)`, for a schedule that keeps a request's noise
    /// reproducible without a normal's tail.
    pub const fn uniform(self) -> Self {
        Self {
            kind: NoiseKind::Uniform,
        }
    }

    /// Draw `out` `[rows, elements]` in its own element type.
    ///
    /// `keys` is `[rows]` i64 with each row's sampling key as its bit pattern; `stage` and `draw`
    /// place the draw within the request's schedule so that two draws never coincide.
    pub fn forward(
        &self,
        f: &Forward,
        out: TensorView,
        keys: TensorView,
        stage: u32,
        draw: u32,
    ) -> Result<()> {
        if keys.shape.rank() != 1 || keys.shape.dim(0) != out.shape.dim(0) {
            return Err(Error::shape(
                "Noise",
                format!(
                    "{} sampling keys for {} rows",
                    keys.shape.numel(),
                    out.shape.dim(0)
                ),
            ));
        }
        let descriptor = noise::NoiseDescriptor {
            dtype: out.dtype,
            block: BLOCK,
            kind: self.kind,
        };
        let args = noise::NoiseArgs {
            out,
            keys,
            stage,
            draw,
        };

        f.operate::<NoiseOp>(&descriptor, args)
    }
}
