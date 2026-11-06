//! `tile_stitch`: a clip's decoded tiles straight from their tokens into its raster.
//!
//! A video decoder decodes overlapping tiles, each a run of token rows whose rows hold one patch of
//! pixels. The clip's raster is what the reference's unpack and `stitch_tiles` make of them: every
//! pixel belongs to exactly one tile, the first rows and columns of a tile fade against the tile above
//! it and the tile to its left, and what a later tile overlaps is cropped away. So a pixel of the raster
//! is a function of at most three token values, and this operator computes it from them directly:
//!
//! ```text
//! value = token value of the pixel's own tile
//! if the pixel is in the seam with the tile above:    value = fade(above, value, row, seam)
//! if the pixel is in the seam with the tile to the left: value = fade(left, value, column, seam)
//! ```
//!
//! `above` and `left` are those tiles as they were decoded, never as they were blended, which is the
//! reference's rule (`uniserve/nn/vae/tiled.py`, `stitch_tiles`): a neighbour stays unmodified so a
//! corner is computed in the same order as an edge. `fade` is the reference's arithmetic in the
//! element type at every step (`uniserve/nn/video.py`, `blend_decoded_overlap`): `position / extent`
//! rounds, `1 - that` rounds, each product rounds and their sum rounds.
//!
//! The grid's geometry arrives as two tables, one row per pixel of each axis: the tile of that
//! position along the axis, the position inside that tile, and the width of the seam the position
//! lies in, zero outside one. A table is the model's arithmetic on its canvas, so the kernel holds no
//! tile logic and one module serves every canvas.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView, row_writable};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// Raster columns one program writes.
pub const BLOCK: u32 = 1024;

/// The words of one axis table's row: tile, position inside it, seam width.
pub const TABLE_WIDTH: u64 = 3;

/// One compilation of `tile_stitch`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the tokens and of the raster.
    pub dtype: DType,
    /// Channels of the raster.
    pub channels: u32,
    /// Frames one token carries.
    pub patch_frames: u32,
    /// Rows and columns one token carries.
    pub patch_side: u32,
    /// Raster columns one program writes.
    pub block: u32,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("channels", self.channels)
            .field("patch_frames", self.patch_frames)
            .field("patch_side", self.patch_side)
            .field("block", self.block)
            .finish(formatter)
    }
}

/// The arguments of one stitch.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[tiles · token_rows, channels · patch_frames · patch_side²]`: each tile's tokens in turn.
    pub tokens: TensorView,
    /// `[height, 3]` i32: each raster row's tile row, its row inside that tile and its seam.
    pub rows: TensorView,
    /// `[width, 3]` i32: each raster column's tile column, its column inside that tile and its seam.
    pub columns: TensorView,
    /// `[channels, frames, height, width]`: the clip's raster.
    pub out: TensorView,
    /// Tiles in one row of the grid.
    pub tile_columns: u32,
    /// Token rows between one tile's tokens and the next's.
    pub token_rows: u32,
    /// Pixel rows of a tile.
    pub tile_height: u32,
    /// Pixel columns of a tile.
    pub tile_width: u32,
}

/// A clip's tiles into its raster.
pub struct TileStitch;

impl Op for TileStitch {
    const NAME: &'static str = "tile_stitch";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        args.out.require_rank(Self::NAME, 4)?;
        let dims = args.out.shape.dims();
        let (channels, frames, height, width) = (dims[0], dims[1], dims[2], dims[3]);
        if channels != u64::from(descriptor.channels) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the raster has {channels} channels and the module {}",
                    descriptor.channels
                ),
            ));
        }
        args.out.require_row(Self::NAME, descriptor.dtype, width)?;
        let patch = u64::from(descriptor.channels)
            * u64::from(descriptor.patch_frames)
            * u64::from(descriptor.patch_side)
            * u64::from(descriptor.patch_side);
        args.tokens
            .require_row(Self::NAME, descriptor.dtype, patch)?;
        args.rows
            .require_shape(Self::NAME, Shape::new(&[height, TABLE_WIDTH]))?;
        args.columns
            .require_shape(Self::NAME, Shape::new(&[width, TABLE_WIDTH]))?;
        for (name, table) in [("row", args.rows), ("column", args.columns)] {
            if table.dtype != DType::I32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} table is {}, not i32", table.dtype),
                ));
            }
        }
        let strides = row_writable(&args.out, Self::NAME)?;
        let word = |name: &str, value: u64| -> Result<u32> {
            u32::try_from(value).map_err(|_| {
                Error::argument(
                    Self::NAME,
                    format!("the {name} {value} does not fit a kernel parameter"),
                )
            })
        };
        let mut arguments = Arguments::new();
        arguments
            .view(args.tokens)
            .view(args.rows)
            .view(args.columns)
            .view(args.out)
            .u32(word("frame count", frames)?)
            .u32(word("width", width)?)
            .u64(strides[0])
            .u64(strides[1])
            .u64(strides[2])
            .u32(args.tile_columns)
            .u32(args.token_rows)
            .u32(args.tile_height)
            .u32(args.tile_width);
        kernel.launch(
            LaunchGrid::new([
                width.div_ceil(u64::from(descriptor.block)),
                height,
                channels * frames,
            ])?,
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
        // The reference's own arithmetic, step for step in the element type: two implementations
        // produce one set of bits.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_names_the_type_and_the_patch() {
        let descriptor = Descriptor {
            dtype: DType::F16,
            channels: 3,
            patch_frames: 4,
            patch_side: 16,
            block: BLOCK,
        };
        assert_eq!(
            descriptor.to_string(),
            "dtype_f16_channels_3_patch_frames_4_patch_side_16_block_1024"
        );
    }
}
