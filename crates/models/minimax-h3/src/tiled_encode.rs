//! The video encoder's tiled path: a raster larger than one tile becomes its latent by tiles that
//! cross-fade where they meet.
//!
//! [`crate::encoding::VideoEncoder::encode_tile`] encodes **one** tile; [`ClipEncoder`] runs it over
//! the checkpoint's tile grid through the shared driver [`nn::tiling::TiledEncode`], which shares
//! the tiles among the forward's ranks, gathers them and stitches the moments: vertical seams, then
//! horizontal ones, each tile's neighbours read as encoded. What stays here is the checkpoint's
//! geometry (its [`TILE_PIXELS`] tiles with at least [`MINIMUM_OVERLAP`] shared, compressed
//! [`LATENT_COMPRESSION`]-fold) and the copy that packs a tile cut from a wider raster before the
//! encoder reads it.

use crate::encoding::VideoEncoder;
use crate::packing::LATENT_COMPRESSION;
use crate::video_vae::{MINIMUM_OVERLAP, TILE_PIXELS};
use crate::{Error, Result};
use nn::tiling::{Tile, TiledEncode};
use nn::{DType, Forward, Shape, Operand};

/// A tile's block of a wider raster, `[channels, frames, height, width]` at the raster's strides,
/// copied into a fresh packed `[1, channels, frames, height, width]` sample.
///
/// The encoder's first pad reads each frame's rows in place, so it needs the tile's own row stride,
/// and a block of a wider raster has the raster's.
fn pack_tile(f: &Forward, block: Operand) -> Result<Operand> {
    let dims = block.shape.dims();
    let (channels, frames, height, width) = (dims[0], dims[1], dims[2], dims[3]);
    let packed = f.empty(DType::F32, Shape::new(&[channels, frames, height, width]))?;
    let mut strides = [0_i64; ops::copy_block::RANK];
    for (axis, stride) in [frames * height * width, height * width, width, 1]
        .into_iter()
        .enumerate()
    {
        strides[axis] = i64::try_from(stride).map_err(|_| {
            Error::config("a condition tile is larger than i64 strides address".to_owned())
        })?;
    }
    nn::layers::Copy::new(DType::F32, nn::layers::Access::Element)?
        .forward(f, block, packed, &strides)?;
    Ok(packed.reshape(Shape::new(&[1, channels, frames, height, width])))
}

/// The tiled encoding of one raster through the checkpoint's video encoder.
///
/// The raster's extents are input pixels and must be multiples of the spatial compression.
pub struct ClipEncoder {
    tiling: TiledEncode,
}

impl ClipEncoder {
    /// The tiled encoding of a `height` by `width` raster, in input pixels.
    pub fn new(height: u32, width: u32) -> Result<Self> {
        Ok(Self {
            tiling: TiledEncode::new(
                height,
                width,
                TILE_PIXELS,
                MINIMUM_OVERLAP,
                LATENT_COMPRESSION,
                DType::F32,
            )?,
        })
    }

    /// Every tile, row-major, in latent cells.
    pub fn tiles(&self) -> &[Tile] {
        self.tiling.tiles()
    }

    /// The latent raster one encoding produces, `(rows, columns)`.
    pub fn latent_extent(&self) -> (u32, u32) {
        self.tiling.latent()
    }

    /// Encode one clip of `raster` `[in_channels, frames, height, width]` FP32 into its stitched
    /// moments `[moment channels, latent frames, latent rows, latent columns]`.
    ///
    /// Each tile's block is packed and encoded alone; the shared driver shares the tiles among the
    /// forward's ranks, gathers them and blends the seams.
    pub fn encode(
        &self,
        f: &Forward,
        encoder: &VideoEncoder,
        raster: Operand,
        moments: (u64, u64),
    ) -> Result<Operand> {
        let compression = u64::from(LATENT_COMPRESSION);
        let encode = |index: usize| -> nn::Result<Operand> {
            let tile = &self.tiles()[index];
            let block = raster
                .narrow(
                    2,
                    u64::from(tile.rows.0) * compression,
                    u64::from(tile.rows.1) * compression,
                )?
                .narrow(
                    3,
                    u64::from(tile.columns.0) * compression,
                    u64::from(tile.columns.1) * compression,
                )?;
            Ok(encoder.encode_tile(f, pack_tile(f, block)?)?)
        };
        Ok(self.tiling.forward(f, moments, encode)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 768 by 1344 canvas: 48 by 84 latents, four tile rows and seven columns.
    fn canvas() -> ClipEncoder {
        ClipEncoder::new(768, 1344).expect("the verified canvas")
    }

    #[test]
    fn a_raster_within_one_tile_is_one_tile_with_no_seam() {
        let small = ClipEncoder::new(256, 256).expect("one tile");
        let latent = small.latent_extent();
        assert_eq!(latent, (256 / LATENT_COMPRESSION, 256 / LATENT_COMPRESSION));
        let tiles = small.tiles();
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].rows, (0, latent.0));
        assert_eq!(tiles[0].columns, (0, latent.1));
        assert_eq!((tiles[0].above, tiles[0].left), (0, 0));
        assert!(
            small.tiling.blends().is_empty(),
            "a raster inside one tile has no seam"
        );
    }

    #[test]
    fn a_raster_that_is_not_whole_patches_is_refused_rather_than_truncated() {
        assert!(ClipEncoder::new(256 + LATENT_COMPRESSION / 2, 256).is_err());
        assert!(ClipEncoder::new(256, 256 + LATENT_COMPRESSION / 2).is_err());
        assert!(ClipEncoder::new(768, 1344).is_ok());
        assert!(ClipEncoder::new(480, 832).is_ok());
    }

    /// The seams are the reference's, in latent cells: `split_tiles` on this raster spreads the
    /// slack over the seams in tile order, three seams of 6, 5 and 5 latents down the 48 rows and six
    /// of 5, 5, 5, 5, 4 and 4 across the 84 columns, and the run compiles one blend per width and
    /// axis.
    #[test]
    fn a_seam_fades_over_the_overlap_the_split_was_given() {
        let encoder = canvas();
        let tiles = encoder.tiles();
        assert_eq!(tiles.len(), 4 * 7);
        let above: Vec<u32> = tiles.iter().step_by(7).map(|tile| tile.above).collect();
        assert_eq!(above, vec![0, 6, 5, 5]);
        let left: Vec<u32> = tiles[..7].iter().map(|tile| tile.left).collect();
        assert_eq!(left, vec![0, 5, 5, 5, 5, 4, 4]);
        assert_eq!(
            encoder.tiling.blends().len(),
            4,
            "6 and 5 down, 5 and 4 across"
        );
    }

    /// Every latent row and column is written by some tile, and the tiles are one extent, which is
    /// what lets one reservation serve the run.
    #[test]
    fn the_tiles_cover_the_latent_raster_in_one_extent() {
        let encoder = canvas();
        let latent = encoder.latent_extent();
        let mut rows = vec![false; latent.0 as usize];
        let mut columns = vec![false; latent.1 as usize];
        for tile in encoder.tiles() {
            assert_eq!((tile.rows.1, tile.columns.1), (16, 16));
            for row in tile.rows.0..tile.rows.0 + tile.rows.1 {
                rows[row as usize] = true;
            }
            for column in tile.columns.0..tile.columns.0 + tile.columns.1 {
                columns[column as usize] = true;
            }
        }
        assert!(rows.iter().all(|row| *row));
        assert!(columns.iter().all(|column| *column));
    }
}
