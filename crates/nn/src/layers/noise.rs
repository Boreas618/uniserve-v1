//! The initial noise of a video diffusion request, drawn on the device.
//!
//! A request's noise is a function of the request's sampling key and the element's position, so the
//! same request regenerates the same latents without the host holding a sample (specs/worker.md 9,
//! "Philox"; `ops::noise`). The layer holds only the element type and the tile; the key is the
//! caller's, because the executor computes it from the request seed and the phase.
//!
//! The key comes in one of two forms, and which one a caller has decides the launch shape:
//!
//! - [`Key::Rows`] — a `[rows]` i64 view with one sampling key per row, for a batched causal model
//!   whose requests are independent streams. One program draws one row's block.
//! - [`Key::Scalar`] — one `u64` seed for the whole buffer, drawn over it flattened. A diffusion
//!   denoiser holds a host seed and no device key view, and its latent's element at a flat index
//!   must draw from `(seed, index)` whatever the buffer's shape is.

use ops::TensorView;
use ops::noise::{self, KeySource, NoiseKind, PhiloxNoise as NoiseOp};

use crate::error::{Error, Result};
use crate::forward::Forward;

/// The elements one noise program draws.
const BLOCK: u32 = noise::NOISE_BLOCK;

/// Where a draw's key comes from.
#[derive(Clone, Copy)]
pub enum Key {
    /// `[rows]` i64 sampling keys, one per row: one independent stream per row of `out`.
    Rows(TensorView),
    /// One seed for the whole draw, over `out`'s flattened elements.
    Scalar(u64),
}

/// Noise addressed by (key, element): standard normal, or uniform in `[0, 1)`.
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
    /// `key` is the request's, in whichever of the two forms the caller holds: per-row keys for a
    /// batch, or one seed for a buffer whose values must depend on their flat position alone.
    /// `stage` and `draw` place the draw within the request's schedule so that two draws never
    /// coincide; under [`Key::Scalar`] they keep the values a pure function of `(seed, element)`
    /// for the request, since they are fixed for the launch.
    pub fn draw(
        &self,
        f: &Forward,
        out: TensorView,
        key: Key,
        stage: u32,
        draw: u32,
    ) -> Result<()> {
        let (source, keys, seed) = match key {
            Key::Rows(keys) => {
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
                (KeySource::Rows, Some(keys), 0)
            }
            Key::Scalar(seed) => (KeySource::Scalar, None, seed),
        };
        let descriptor = noise::NoiseDescriptor {
            dtype: out.dtype,
            block: BLOCK,
            kind: self.kind,
            source,
        };
        let args = noise::NoiseArgs {
            out,
            keys,
            key: seed,
            stage,
            draw,
        };

        f.operate::<NoiseOp>(&descriptor, args)
    }
}
