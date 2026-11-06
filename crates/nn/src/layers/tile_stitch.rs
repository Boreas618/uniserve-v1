//! A clip's decoded tiles into its raster: the unpack, the seam blends and the crop as one launch
//! (`ops::tile_stitch`).

use ops::TensorView;
use ops::tile_stitch::{self, Args, BLOCK, Descriptor, TileStitch as TileStitchOp};

use crate::error::{Error, Result};
use crate::forward::Forward;

/// The stitch of one patch geometry.
#[derive(Clone, Copy)]
pub struct TileStitch {
    descriptor: Descriptor,
}

/// One grid's numbers, which a canvas states.
#[derive(Clone, Copy, Debug)]
pub struct TileGrid {
    /// Tiles in one row of the grid.
    pub columns: u32,
    /// Token rows between one tile's tokens and the next's.
    pub token_rows: u32,
    /// Pixel rows of a tile.
    pub height: u32,
    /// Pixel columns of a tile.
    pub width: u32,
}

impl TileStitch {
    /// The stitch of tokens of `dtype`, each `patch_frames` frames of a `patch_side`-pixel square of
    /// `channels` channels.
    pub fn new(
        dtype: ops::DType,
        channels: u32,
        patch_frames: u32,
        patch_side: u32,
    ) -> Result<Self> {
        if channels == 0 || patch_frames == 0 || !patch_side.is_power_of_two() {
            return Err(Error::shape(
                "TileStitch",
                format!(
                    "a patch of {channels} channels, {patch_frames} frames and a side of \
                     {patch_side}"
                ),
            ));
        }
        Ok(Self {
            descriptor: Descriptor {
                dtype,
                channels,
                patch_frames,
                patch_side,
                block: BLOCK,
            },
        })
    }

    /// The descriptor one stitch launches, for a walk without a device.
    pub const fn descriptor(&self) -> Descriptor {
        self.descriptor
    }

    /// Stitch `tokens` into `out`, `[channels, frames, height, width]`, through the two axis tables.
    pub fn forward(
        &self,
        f: &Forward,
        tokens: TensorView,
        rows: TensorView,
        columns: TensorView,
        grid: TileGrid,
        out: TensorView,
    ) -> Result<()> {
        f.operate::<TileStitchOp>(
            &self.descriptor,
            Args {
                tokens,
                rows,
                columns,
                out,
                tile_columns: grid.columns,
                token_rows: grid.token_rows,
                tile_height: grid.height,
                tile_width: grid.width,
            },
        )
    }
}

const _: () = assert!(tile_stitch::TABLE_WIDTH == 3);
