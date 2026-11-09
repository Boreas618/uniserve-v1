//! The initial noise of a video diffusion request, drawn on the device.
//!
//! A request's noise is a function of the request's sampling key and the element's position, so the
//! same request regenerates the same latents without the host holding a sample (specs/worker.md 9,
//! "Philox"; `ops::noise`). The layer holds only the element type and the tile; the key is the
//! caller's, because the worker computes it from the request seed and the phase.
//!
//! The key comes in one of two forms, and which one a caller has decides the launch shape:
//!
//! - [`Key::Rows`] — a `[rows]` i64 view with one sampling key per row, for a batched causal model
//!   whose requests are independent streams. One program draws one row's block.
//! - [`Key::Scalar`] — one `u64` seed for the whole buffer, drawn over it flattened. A diffusion
//!   denoiser holds a host seed and no device key view, and its latent's element at a flat index
//!   must draw from `(seed, index)` whatever the buffer's shape is.

use ops::noise::{self, KeySource, NoiseKind, PhiloxNoise as NoiseOp};
use ops::{Shape, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;

/// The elements one noise program draws.
const BLOCK: u32 = noise::NOISE_BLOCK;

/// The two-axis shape a scalar-keyed draw is launched on: every axis but the last folded into the
/// rows, the last kept as the width.
///
/// `philox_noise` reads a two-axis buffer and, under a scalar key, addresses it **flattened** - a
/// value is a function of `(seed, flat index)`. Where the fold falls therefore changes no value, and
/// the rank of the caller's buffer is not part of the draw: a latent arrives as a four-axis raster
/// from one caller and as a two-axis view from another, and a layer that made the rank a
/// precondition refused the second kind after the first kind had already drawn.
fn flat_shape(shape: Shape) -> Shape {
    let dims = shape.dims();
    let Some((width, folded)) = dims.split_last() else {
        return shape;
    };
    let mut rows = 1_u64;
    for extent in folded {
        rows = rows.saturating_mul(*extent);
    }
    Shape::new(&[rows, *width])
}

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


    /// Draw `out` `[rows, elements]` in its own element type.
    ///
    /// `key` is the request's, in whichever of the two forms the caller holds: per-row keys for a
    /// batch, or one seed for a buffer whose values must depend on their flat position alone.
    /// `stage` and `draw` place the draw within the request's schedule so that two draws never
    /// coincide; under [`Key::Scalar`] they keep the values a pure function of `(seed, element)`
    /// for the request, since they are fixed for the launch.
    ///
    /// `offset` is the element of the request's stream this draw starts at. The reference draws
    /// every condition, the video latent and the audio latent through one Philox stream, so a
    /// caller that draws part of it — a rank's shard of the latent, or the audio after the video —
    /// passes the global flat index it begins at, and two draws at adjacent offsets reproduce one
    /// longer draw. An odd offset is legal and takes the high half of a word first; a caller with a
    /// whole buffer of its own passes 0.
    /// The descriptor one draw takes, stated without a device: the element type, the distribution and
    /// the key source. The keys, the stage, the draw's index and its offset travel with the call.
    pub fn descriptor_of(
        dtype: ops::DType,
        kind: ops::noise::NoiseKind,
        source: ops::noise::KeySource,
    ) -> noise::NoiseDescriptor {
        noise::NoiseDescriptor {
            dtype,
            block: BLOCK,
            kind,
            source,
        }
    }

    pub fn draw(
        &self,
        f: &Forward,
        mut out: TensorView,
        key: Key,
        stage: u32,
        draw: u32,
        offset: u64,
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
            Key::Scalar(seed) => {
                // The scalar form addresses the buffer flattened, so the view the operator reads is
                // the two-axis one whatever rank the caller's buffer has.
                out = out.reshape(flat_shape(out.shape));
                (KeySource::Scalar, None, seed)
            }
        };
        let descriptor = Self::descriptor_of(out.dtype, self.kind, source);
        let args = noise::NoiseArgs {
            out,
            keys,
            key: seed,
            stage,
            draw,
            offset,
        };

        f.operate::<NoiseOp>(&descriptor, args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The draw is addressed by `(seed, flat index)`, so a buffer's rank is not part of it and every
    /// rank folds to the two-axis view the operator reads. The four-axis raster is the shape a
    /// denoiser hands, and the two-axis one is what a piece's draw arrives as - a layer that made
    /// the rank a precondition refused the second after the first had already drawn.
    #[test]
    fn every_rank_folds_to_the_two_axis_view_the_draw_reads() {
        let raster = flat_shape(Shape::new(&[24, 37, 48, 84]));
        assert_eq!(raster.dims(), &[24 * 37 * 48, 84]);

        let piece = flat_shape(Shape::new(&[2, 207]));
        assert_eq!(piece.dims(), &[2, 207], "a two-axis buffer is unchanged");

        let row = flat_shape(Shape::new(&[96768]));
        assert_eq!(row.dims(), &[1, 96768]);
    }

    /// The fold keeps every element and the row-major order, which is what makes the address a
    /// function of the element alone.
    #[test]
    fn the_fold_loses_no_element() {
        for dims in [&[24_u64, 37, 48, 84][..], &[2, 207][..], &[4, 1, 1, 64][..]] {
            let shape = Shape::new(dims);
            assert_eq!(flat_shape(shape).numel(), shape.numel(), "{dims:?}");
        }
    }
}
