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
//! stitching rule: a neighbour stays unmodified so a
//! corner is computed in the same order as an edge. `fade` is the overlap blend's arithmetic in the
//! element type at every step: `position / extent`
//! rounds, `1 - that` rounds, each product rounds and their sum rounds.
//!
//! The grid's geometry arrives as two tables, one row per pixel of each axis: the tile of that
//! position along the axis, the position inside that tile, and the width of the seam the position
//! lies in, zero outside one. A table is the model's arithmetic on its canvas, so the kernel holds no
//! tile logic and one module serves every canvas.

use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::{self, Key};
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::operand::{Operand, row_writable};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

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

impl Descriptor {
    /// The stitch of `dtype` tokens of `channels`, `patch_frames` frames and `patch_side`
    /// rows and columns, over the operators' block ([`descriptor::BLOCK`]).
    pub const fn new(dtype: DType, channels: u32, patch_frames: u32, patch_side: u32) -> Self {
        Self {
            dtype,
            channels,
            patch_frames,
            patch_side,
            block: descriptor::BLOCK,
        }
    }
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("channels", self.channels)
            .field("patch_frames", self.patch_frames)
            .field("patch_side", self.patch_side)
            .field("block", self.block)
            .finish(formatter)
    }
}

/// The arguments of one stitch.
#[derive(Clone)]
pub struct Args {
    /// `[tiles · token_rows, channels · patch_frames · patch_side²]`: each tile's tokens in turn.
    pub tokens: Operand,
    /// `[height, 3]` i32: each raster row's tile row, its row inside that tile and its seam.
    pub rows: Operand,
    /// `[width, 3]` i32: each raster column's tile column, its column inside that tile and its seam.
    pub columns: Operand,
    /// `[channels, frames, height, width]`: the clip's raster.
    pub out: Operand,
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
    const NAME: &'static str = proto::module::operator::TILE_STITCH;

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        args.out.require_rank(Self::NAME, 4)?;
        let (channels, frames, height, width) = (
            args.out.dim(0),
            args.out.dim(1),
            args.out.dim(2),
            args.out.dim(3),
        );
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
            .require_shape(Self::NAME, &[height, TABLE_WIDTH])?;
        args.columns
            .require_shape(Self::NAME, &[width, TABLE_WIDTH])?;
        for (name, table) in [("row", &args.rows), ("column", &args.columns)] {
            if table.dtype != DType::I32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} table is {}, not i32", abi::name(table.dtype)),
                ));
            }
        }
        row_writable(&args.out, Self::NAME)?;
        let word = |name: &str, value: u64| -> Result<u32> {
            u32::try_from(value).map_err(|_| {
                Error::argument(
                    Self::NAME,
                    format!("the {name} {value} does not fit a kernel parameter"),
                )
            })
        };
        word("frame count", frames)?;
        word("width", width)?;
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let (channels, frames, height, width) = (
            args.out.dim(0),
            args.out.dim(1),
            args.out.dim(2),
            args.out.dim(3),
        );
        // The outer strides of the layout `check` accepted.
        let strides = crate::operand::outer_strides(&args.out);

        // `check` keeps the frame count and the width within kernel parameters.
        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("tokens_ptr", &args.tokens)
            .view("rows_ptr", &args.rows)
            .view("columns_ptr", &args.columns)
            .view("out_ptr", &args.out)
            .u64("frames", frames)
            .u64("width", width)
            .u64("out_stride0", strides[0])
            .u64("out_stride1", strides[1])
            .u64("out_stride2", strides[2])
            .u32("tile_columns", args.tile_columns)
            .u32("token_rows", args.token_rows)
            .u32("tile_height", args.tile_height)
            .u32("tile_width", args.tile_width);
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

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        // The stitched window may be a strided view of the frame it lands in: its shadow is the
        // storage the window reaches, with the window's strides inside it.
        Args {
            out: crate::strided::shadow_window(args.out, twins),
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
        let descriptor = Descriptor::new(DType::F16, 3, 4, 16);
        assert_eq!(
            descriptor.to_string(),
            "dtype_f16_channels_3_patch_frames_4_patch_side_16_block_1024"
        );
    }
}
