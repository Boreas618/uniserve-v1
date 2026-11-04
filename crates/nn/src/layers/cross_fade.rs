//! The seam between two decoded video tiles, blended over their overlap.
//!
//! The decoder decodes overlapping tiles and blends the band they share so the seam does not show.
//! The reference derives the weights rather than taking them: position `i` of an `extent`-voxel
//! overlap reads `1 - i/extent` of the previous tile and `i/extent` of the current one, both in the
//! decoded element type. Reproducing that arithmetic exactly is the difference between a fade and a
//! slightly different one (`ops::cross_fade`), so the layer carries the geometry and nothing else —
//! no weight table is prepared, and none may be: a caller's own weights would be a different fade.
//!
//! `out` may be `current`: every blended position reads the current tile at the position it writes,
//! which is what a decoder wants — the tile it keeps is the one it just decoded.

use ops::cross_fade::{self, CrossFade as CrossFadeOp};
use ops::{DType, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;

/// Elements one program blends.
const BLOCK: u32 = cross_fade::BLOCK;

/// The overlap between two decoded tiles.
#[derive(Clone, Copy)]
pub struct CrossFade {
    dtype: DType,
    extent: u32,
    axis: u32,
}

impl CrossFade {
    /// A blend over `extent` voxels along `axis` of a `[channels, frames, rows, columns]` tile.
    ///
    /// The overlap runs along the rows (`2`) or the columns (`3`), which are the axes a tile grid
    /// shares; the axis is compiled in, because it is a loop shape rather than a launch value, so a
    /// decoder that fades both ways needs two layers.
    pub fn new(dtype: DType, extent: u32, axis: u32) -> Result<Self> {
        if extent == 0 {
            return Err(Error::shape(
                "CrossFade",
                "an overlap of no voxels blends nothing".to_owned(),
            ));
        }
        if axis != 2 && axis != 3 {
            return Err(Error::shape(
                "CrossFade",
                format!(
                    "the overlap runs along the rows (2) or the columns (3) of a \
                     [channels, frames, rows, columns] tile, not along {axis}"
                ),
            ));
        }
        Ok(Self {
            dtype,
            extent,
            axis,
        })
    }

    /// The voxels the two tiles share.
    pub const fn extent(&self) -> u32 {
        self.extent
    }

    /// Blend `previous` and `current` into `out`, all `[channels, frames, rows, columns]`.
    ///
    /// The leading `extent` positions of `axis` blend; every later position is `current`'s, copied.
    pub fn forward(
        &self,
        f: &Forward,
        previous: TensorView,
        current: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let descriptor = cross_fade::Descriptor {
            dtype: self.dtype,
            extent: self.extent,
            axis: self.axis,
            block: BLOCK,
            pdl: PDL,
        };
        f.operate::<CrossFadeOp>(
            &descriptor,
            cross_fade::Args {
                previous,
                current,
                out,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_overlap_runs_along_one_of_the_two_shared_axes() {
        assert!(CrossFade::new(DType::BF16, 32, 2).is_ok());
        assert!(CrossFade::new(DType::BF16, 32, 3).is_ok());
        assert!(CrossFade::new(DType::BF16, 0, 2).is_err());
        assert!(
            CrossFade::new(DType::BF16, 32, 1).is_err(),
            "frames are decoded together, not faded across"
        );
        assert!(CrossFade::new(DType::BF16, 32, 4).is_err());
    }
}
