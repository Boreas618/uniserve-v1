//! The video encoder's tiled path: a raster larger than one tile becomes its latent by tiles that
//! cross-fade where they meet.
//!
//! [`crate::encoding::VideoEncoder::encode_tile`] encodes **one** tile; [`ClipEncoder`] runs it over
//! the checkpoint's tile grid through the shared driver [`nn::tiling::TiledEncode`], which shares
//! the tiles among the forward's ranks, gathers them and stitches the moments: vertical seams, then
//! horizontal ones, each tile's neighbours read as encoded. What stays here is the checkpoint's
//! geometry (its [`TILE_PIXELS`] tiles with at least [`MINIMUM_OVERLAP`] shared, compressed
//! [`SPATIAL_RATIO`]-fold), the encoder's buffers, and the copy that packs a tile cut from a wider
//! raster before the encoder reads it.

use crate::encoding::{EncoderScratch, Extent, SPATIAL_RATIO, VideoEncoder};
use crate::video_vae::{MINIMUM_OVERLAP, TILE_PIXELS};
use crate::{Error, Result};
use nn::tiling::{Tile, TileBlocks, TiledEncode};
use nn::{DType, Forward, Shape, Strides, TensorView};

/// The device views one tiled encoding reads and writes, as the caller uploaded and reserved them.
#[derive(Clone, Copy, Debug)]
pub struct Scene<'a> {
    /// Tile `i`'s pixels, in tile order.
    ///
    /// Without [`Self::packed`] each view is `[1, in_channels, frames, tile_height, tile_width]` and
    /// **packed**: `nn::layers::CausalPad` pads a frame's rows in place of the source's own, so it
    /// requires a frame's rows to be contiguous in the view it reads. With it, each view is the
    /// tile's block of a wider raster, `[in_channels, frames, tile_height, tile_width]` at the
    /// raster's strides.
    pub pixels: &'a [TensorView],
    /// The one buffer a tile cut from a wider raster is packed into, just before it is encoded.
    ///
    /// One buffer serves every tile because a tile is copied only once the tile before it was
    /// encoded.
    pub packed: Option<TensorView>,
    /// The stitched moments, `[channels, frames, latent_rows, latent_columns]` FP32.
    pub blended: TensorView,
}

/// What the clips of one raster encode in: the encoder's buffers and the tiles' blocks.
///
/// Reserved once ([`ClipEncoder::reserve`]) and used by each clip of the raster in turn.
pub struct ClipRun {
    scratch: EncoderScratch,
    blocks: TileBlocks,
}

/// Copies a tile's block of a wider raster into `buffer` and answers it as the encoder reads it.
fn pack_tile(f: &Forward, block: TensorView, buffer: TensorView) -> Result<TensorView> {
    let (channels, frames, height, width) = (
        block.shape.dim(0),
        block.shape.dim(1),
        block.shape.dim(2),
        block.shape.dim(3),
    );
    let packed = [frames * height * width, height * width, width, 1];
    let mut destination_strides = [0_i64; ops::copy_block::RANK];
    for (axis, stride) in packed.iter().enumerate() {
        destination_strides[axis] = i64::try_from(*stride).map_err(|_| {
            Error::config("a condition tile is larger than i64 strides address".to_owned())
        })?;
    }
    let destination = TensorView::with_strides(
        buffer.offset_elements(0),
        DType::F32,
        Shape::new(&[channels, frames, height, width]),
        Strides::new(&packed),
    );
    nn::layers::Copy::new(DType::F32, nn::layers::Access::Element)?.forward(
        f,
        block,
        destination,
        &destination_strides,
    )?;
    // The encoder's tile view is the packed copy at the shape it pads from, with the batch axis the
    // scene's contract asks for.
    Ok(TensorView::with_strides(
        buffer.offset_elements(0),
        DType::F32,
        Shape::new(&[1, channels, frames, height, width]),
        Strides::new(&[
            channels * frames * height * width,
            frames * height * width,
            height * width,
            width,
            1,
        ]),
    ))
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
                SPATIAL_RATIO,
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

    /// One reservation for clips of `frames` frames each, whose stitched moments are `raster`'s
    /// channels and at most its latent frames.
    ///
    /// The arena never rewinds inside a forward, so the encoder's buffers are reserved once for the
    /// whole run - one set per distinct tile extent, and a split's tiles are one extent per clip
    /// length.
    pub fn reserve(
        &self,
        f: &Forward,
        encoder: &VideoEncoder,
        frames: &[u32],
        raster: TensorView,
    ) -> Result<ClipRun> {
        let extents: Vec<Extent> = frames
            .iter()
            .flat_map(|frames| {
                self.tiles().iter().map(move |tile| Extent {
                    frames: *frames,
                    height: tile.rows.1 * SPATIAL_RATIO,
                    width: tile.columns.1 * SPATIAL_RATIO,
                })
            })
            .collect();
        let scratch = encoder.reserve(f, &extents)?;
        let blocks =
            self.tiling
                .reserve(f, DType::F32, raster.shape.dim(0), raster.shape.dim(1))?;
        Ok(ClipRun { scratch, blocks })
    }

    /// Encode one clip's tiles into its stitched moments, `scene.blended`, in a reservation of
    /// [`Self::reserve`].
    pub fn encode(
        &self,
        f: &Forward,
        encoder: &VideoEncoder,
        scene: Scene<'_>,
        run: &ClipRun,
    ) -> Result<()> {
        let Scene {
            pixels,
            packed,
            blended,
        } = scene;
        if pixels.len() != self.tiles().len() {
            return Err(Error::config(format!(
                "{} pixel views for {} tiles",
                pixels.len(),
                self.tiles().len()
            )));
        }
        let encode = |index: usize, decoded: TensorView| -> nn::Result<()> {
            // The encoder reads `[1, channels, frames, rows, columns]`; the block is that, batched by
            // one.
            let strides = decoded.strides.as_slice();
            let dims = decoded.shape.dims();
            let out = TensorView::with_strides(
                decoded.element(&[0, 0, 0, 0]),
                DType::F32,
                Shape::new(&[1, dims[0], dims[1], dims[2], dims[3]]),
                Strides::new(&[strides[0], strides[0], strides[1], strides[2], strides[3]]),
            );
            let tile_pixels = match packed {
                Some(buffer) => pack_tile(f, pixels[index], buffer)?,
                None => pixels[index],
            };
            Ok(encoder.encode_tile(f, &run.scratch, tile_pixels, out)?)
        };
        Ok(self.tiling.forward(f, &run.blocks, encode, blended)?)
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
        assert_eq!(latent, (256 / SPATIAL_RATIO, 256 / SPATIAL_RATIO));
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
        assert!(ClipEncoder::new(256 + SPATIAL_RATIO / 2, 256).is_err());
        assert!(ClipEncoder::new(256, 256 + SPATIAL_RATIO / 2).is_err());
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
