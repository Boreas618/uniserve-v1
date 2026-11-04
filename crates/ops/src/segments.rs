//! Selecting the key tiles a video sparse attention reads: the segment tables' kernels.
//!
//! Video sparse attention (VSA) splits the sequence into `tile`-row tiles and attends a query tile
//! only to the key tiles it selects. Five programs prepare and choose those tiles, and none of them
//! allocates: the caller owns every buffer, and the indices stay on the device so a captured graph
//! replays them (specs/worker.md 9, "VSA 分块").
//!
//! ```text
//! vsa_prepare_tiles       normalize and rotate Q and K in place, zero the rows past each tile's
//!                         valid size, and pool each tile's per-head means into FP32
//! vsa_zero_padding        zero the rows past each tile's valid size, in place
//! vsa_add_gated_tiles     out += compressed[:, tile] · gate, in place, rounded per operation
//! vsa_compose_gated_rows  write one row interval of every member's gated rows, member-major
//! vsa_select_tiles        choose each query tile's key tiles from its pooled scores
//! ```
//!
//! Rows are `[tiles · tile, heads, width]` with a unit column stride; rows are processed in
//! [`BLOCK_ROWS`]-row blocks, and every tile is a whole number of them. The selection is by
//! descending pooled score with ties in ascending tile order, and each query tile's ke pt tiles come
//! first in ascending order, followed by the tiles it did not keep.

use std::fmt;

use crate::descriptor::{Float, Key};
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The rows one program handles at a time.
pub const BLOCK_ROWS: u32 = 16;

/// The members one composition launch can cover, which is the widest tensor-parallel group the
/// design allows.
pub const MAX_MEMBERS: usize = 8;

/// One compilation of `vsa_zero_padding`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ZeroPaddingDescriptor {
    /// Element type of the rows.
    pub dtype: DType,
    /// Heads per row.
    pub heads: u32,
    /// Rows of one tile.
    pub tile_rows: u32,
    /// Features per head.
    pub width: u32,
    /// The rows one program handles.
    pub block_rows: u32,
}

impl fmt::Display for ZeroPaddingDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("heads", self.heads)
            .field("tile_rows", self.tile_rows)
            .field("width", self.width)
            .field("block_rows", self.block_rows)
            .finish(formatter)
    }
}

/// The arguments of `vsa_zero_padding`.
#[derive(Clone, Copy)]
pub struct ZeroPaddingArgs {
    /// `[tiles · tile_rows, heads, width]`, unit column stride; rows past a tile's valid size become
    /// zeros.
    pub values: TensorView,
    /// `[tiles]` i32: the valid rows of each tile.
    pub sizes: TensorView,
}

/// Zero the rows of every tile past its valid size.
pub struct VsaZeroPadding;

impl Op for VsaZeroPadding {
    const NAME: &'static str = "vsa_zero_padding";

    type Descriptor = ZeroPaddingDescriptor;
    type Args<'a> = ZeroPaddingArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        check_rows(
            Self::NAME,
            descriptor.dtype,
            descriptor.heads,
            descriptor.width,
            &args.values,
        )?;
        check_block(Self::NAME, descriptor.tile_rows, descriptor.block_rows)?;
        let rows = args.values.shape.dim(0);
        if rows % u64::from(descriptor.tile_rows) != 0 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{rows} rows are not whole tiles of {} rows",
                    descriptor.tile_rows
                ),
            ));
        }
        let tiles = rows / u64::from(descriptor.tile_rows);
        args.sizes.require_shape(Self::NAME, Shape::new(&[tiles]))?;
        args.sizes.require_row(Self::NAME, DType::I32, tiles)?;

        let mut arguments = Arguments::new();
        arguments
            .view(args.values)
            .view(args.sizes)
            .i32(stride(&args.values, 0)?)
            .i32(stride(&args.values, 1)?);
        kernel.launch(
            LaunchGrid::new([
                tiles,
                u64::from(descriptor.tile_rows / descriptor.block_rows),
                1,
            ])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        ZeroPaddingArgs {
            values: twins.in_place(args.values, stream),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // The rows past the valid size are zeros and the rest are untouched.
        Tolerance::exact()
    }
}

/// One compilation of `vsa_add_gated_tiles`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AddGatedTilesDescriptor {
    /// Element type of the rows, the compression and the gate.
    pub dtype: DType,
    /// Rows of one tile.
    pub tile_rows: u32,
    /// Features per head.
    pub width: u32,
    /// The rows one program handles.
    pub block_rows: u32,
}

impl fmt::Display for AddGatedTilesDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("tile_rows", self.tile_rows)
            .field("width", self.width)
            .field("block_rows", self.block_rows)
            .finish(formatter)
    }
}

/// The arguments of `vsa_add_gated_tiles`.
#[derive(Clone, Copy)]
pub struct AddGatedTilesArgs {
    /// `[tiles · tile_rows, heads, width]`, unit column stride; receives the gated compression.
    pub output: TensorView,
    /// `[heads, tiles, width]`, unit column stride: each tile's compression.
    pub compressed: TensorView,
    /// `[tiles · tile_rows, heads, width]`, unit column stride: the per-row gate.
    pub gate: TensorView,
}

/// Add each tile's gated compression to its rows, in place.
pub struct VsaAddGatedTiles;

impl Op for VsaAddGatedTiles {
    const NAME: &'static str = "vsa_add_gated_tiles";

    type Descriptor = AddGatedTilesDescriptor;
    type Args<'a> = AddGatedTilesArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let (rows, heads) = (args.output.shape.dim(0), descriptor.heads_hint(&args)?);
        let width = u64::from(descriptor.width);
        let tiles = rows / u64::from(descriptor.tile_rows);
        args.compressed
            .require_shape(Self::NAME, Shape::new(&[heads, tiles, width]))?;
        for view in [&args.output, &args.gate] {
            check_rows(
                Self::NAME,
                descriptor.dtype,
                heads as u32,
                descriptor.width,
                view,
            )?;
        }
        check_block(Self::NAME, descriptor.tile_rows, descriptor.block_rows)?;
        if rows % u64::from(descriptor.tile_rows) != 0 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{rows} rows are not whole tiles of {} rows",
                    descriptor.tile_rows
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.output)
            .view(args.compressed)
            .view(args.gate)
            .i32(stride(&args.output, 0)?)
            .i32(stride(&args.output, 1)?)
            .i32(stride(&args.gate, 0)?)
            .i32(stride(&args.gate, 1)?)
            .i32(stride(&args.compressed, 1)?)
            .i32(stride(&args.compressed, 0)?);
        kernel.launch(
            LaunchGrid::new([rows / u64::from(descriptor.block_rows), heads, 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        AddGatedTilesArgs {
            output: twins.in_place(args.output, stream),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The product rounds, then the sum rounds, both in the common dtype.
        Tolerance::rounding(2, descriptor.dtype, descriptor.dtype)
    }
}

impl AddGatedTilesDescriptor {
    /// The heads the launch's shapes imply; the descriptor does not carry them because the kernel
    /// reads them from the grid.
    fn heads_hint(&self, args: &AddGatedTilesArgs) -> Result<u64> {
        let heads = args.output.shape.dim(1);
        if args.compressed.shape.dim(0) != heads {
            return Err(Error::argument(
                VsaAddGatedTiles::NAME,
                format!(
                    "the compression holds {} heads, the rows {heads}",
                    args.compressed.shape.dim(0)
                ),
            ));
        }
        Ok(heads)
    }
}

/// One compilation of `vsa_compose_gated_rows`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ComposeGatedRowsDescriptor {
    /// Element type of the rows, the compression, the gate and the payload.
    pub dtype: DType,
    /// Heads per row.
    pub heads: u32,
    /// Rows of one tile.
    pub tile_rows: u32,
    /// Features per head.
    pub width: u32,
    /// The rows one program handles.
    pub block_rows: u32,
    /// The payload slot of each member.
    pub slots: [u32; MAX_MEMBERS],
    /// The members this launch covers.
    pub members: u32,
}

impl fmt::Display for ComposeGatedRowsDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let slots: Vec<String> = self.slots[..self.members as usize]
            .iter()
            .map(ToString::to_string)
            .collect();
        Key::new()
            .field("dtype", self.dtype)
            .field("heads", self.heads)
            .field("tile_rows", self.tile_rows)
            .field("width", self.width)
            .field("block_rows", self.block_rows)
            .field("slots", slots.join(","))
            .field("members", self.members)
            .finish(formatter)
    }
}

/// The arguments of `vsa_compose_gated_rows`.
#[derive(Clone, Copy)]
pub struct ComposeGatedRowsArgs {
    /// `[members, count, heads, width]`, contiguous: receives the members' gated rows.
    pub output: TensorView,
    /// `[members · owner_rows, heads, width]`, unit column stride: the fine rows of every member.
    pub fine: TensorView,
    /// `[heads, tiles, width]`, unit column stride: each tile's compression.
    pub compressed: TensorView,
    /// `[members · owner_rows, heads, width]`, unit column stride: the per-row gate.
    pub gate: TensorView,
    /// Rows one member owns in `fine` and `gate`.
    pub owner_rows: u64,
    /// First row of the interval within each member's rows.
    pub start: u64,
    /// Rows of the interval.
    pub count: u64,
}

/// Write one row interval of every member's gated rows, member-major.
pub struct VsaComposeGatedRows;

impl Op for VsaComposeGatedRows {
    const NAME: &'static str = "vsa_compose_gated_rows";

    type Descriptor = ComposeGatedRowsDescriptor;
    type Args<'a> = ComposeGatedRowsArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let members = u64::from(descriptor.members);
        if members == 0 || members > MAX_MEMBERS as u64 {
            return Err(Error::argument(
                Self::NAME,
                format!("a composition covers 1..={MAX_MEMBERS} members, not {members}"),
            ));
        }
        let mut slots: Vec<u32> = descriptor.slots[..members as usize].to_vec();
        slots.sort_unstable();
        if slots != (0..members as u32).collect::<Vec<u32>>() {
            return Err(Error::argument(
                Self::NAME,
                format!("the payload slots {slots:?} are not one per member of {members}"),
            ));
        }
        let heads = args.output.shape.dim(2);
        let width = args.output.shape.dim(3);
        if args.output.shape.dim(0) != members {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the payload holds {} members, the descriptor {members}",
                    args.output.shape.dim(0)
                ),
            ));
        }
        if u64::from(descriptor.heads) != heads || u64::from(descriptor.width) != width {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the descriptor compiles for {heads} heads of {width} features, it says {} and {}",
                    descriptor.heads, descriptor.width
                ),
            ));
        }
        let rows = members * args.owner_rows;
        for (name, view) in [("fine rows", &args.fine), ("gate", &args.gate)] {
            if view.shape.dim(0) != rows
                || view.shape.dim(1) != heads
                || view.shape.dim(2) != width
                || view.dtype != descriptor.dtype
                || view.strides.last() != 1
            {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the {name} are not [{rows}, {heads}, {width}] {} rows with a unit column \
                         stride",
                        descriptor.dtype
                    ),
                ));
            }
        }
        let tiles = rows / u64::from(descriptor.tile_rows);
        if rows % u64::from(descriptor.tile_rows) != 0 {
            return Err(Error::argument(
                Self::NAME,
                format!("{rows} owned rows are not whole tiles"),
            ));
        }
        args.compressed
            .require_shape(Self::NAME, Shape::new(&[heads, tiles, width]))?;
        if args.compressed.strides.last() != 1 {
            return Err(Error::argument(
                Self::NAME,
                "the compression's column stride is not one".to_owned(),
            ));
        }
        if descriptor.block_rows == 0
            || !descriptor.tile_rows.is_multiple_of(descriptor.block_rows)
            || !BLOCK_ROWS.is_multiple_of(descriptor.block_rows)
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "blocks of {} rows do not divide a tile of {} rows",
                    descriptor.block_rows, descriptor.tile_rows
                ),
            ));
        }
        if args.count == 0 {
            return Err(Error::argument(
                Self::NAME,
                "an empty row interval composes nothing".to_owned(),
            ));
        }
        if args.start % u64::from(descriptor.block_rows) != 0
            || args.count % u64::from(descriptor.block_rows) != 0
            || args.start + args.count > args.owner_rows
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the interval {}..{} is not whole blocks within {} owned rows",
                    args.start,
                    args.start + args.count,
                    args.owner_rows
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.output)
            .view(args.fine)
            .view(args.compressed)
            .view(args.gate)
            .i32(stride(&args.fine, 0)?)
            .i32(stride(&args.fine, 1)?)
            .i32(stride(&args.gate, 0)?)
            .i32(stride(&args.gate, 1)?)
            .i32(stride(&args.compressed, 1)?)
            .i32(stride(&args.compressed, 0)?)
            .i64(args.owner_rows as i64)
            .i64(args.start as i64)
            .i64(args.count as i64);
        kernel.launch(
            LaunchGrid::new([
                args.count / u64::from(descriptor.block_rows),
                heads,
                members,
            ])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        ComposeGatedRowsArgs {
            output: twins.shadow_of(args.output),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The product rounds, then the sum rounds, both in the common dtype.
        Tolerance::rounding(2, descriptor.dtype, descriptor.dtype)
    }
}

/// One compilation of `vsa_select_tiles`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SelectTilesDescriptor {
    /// The lanes one program loads: the tile count rounded up to a power of two.
    pub block: u32,
}

impl fmt::Display for SelectTilesDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new().field("block", self.block).finish(formatter)
    }
}

/// The arguments of `vsa_select_tiles`.
#[derive(Clone, Copy)]
pub struct SelectTilesArgs {
    /// `[heads, query tiles, tiles]` FP32, contiguous: the pooled (query tile, key tile) scores.
    pub scores: TensorView,
    /// `[tiles]` i32: the segment of each key tile.
    pub segments: TensorView,
    /// `[tiles]` i32: the valid rows of each key tile.
    pub sizes: TensorView,
    /// `[segments]` i32: how many tiles of each segment to keep.
    pub keep: TensorView,
    /// `[heads, query tiles, tiles]` i32: receives each query tile's key tiles, kept first.
    pub indices: TensorView,
    /// `[heads, query tiles]` i32: receives each query tile's kept-tile count.
    pub counts: TensorView,
    /// The sequence tile of the scores' first query tile.
    pub query_offset: u64,
}

/// Choose each query tile's key tiles from its pooled scores.
pub struct VsaSelectTiles;

impl Op for VsaSelectTiles {
    const NAME: &'static str = "vsa_select_tiles";

    type Descriptor = SelectTilesDescriptor;
    type Args<'a> = SelectTilesArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        args.scores.require_rank(Self::NAME, 3)?;
        if args.scores.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                "the pooled scores are FP32".to_owned(),
            ));
        }
        let (heads, query_tiles, tiles) = (
            args.scores.shape.dim(0),
            args.scores.shape.dim(1),
            args.scores.shape.dim(2),
        );
        if args.scores.strides != crate::tensor::Strides::row_major(args.scores.shape) {
            return Err(Error::argument(
                Self::NAME,
                "the scores are not contiguous".to_owned(),
            ));
        }
        for (name, table) in [("segment", &args.segments), ("size", &args.sizes)] {
            table.require_shape(Self::NAME, Shape::new(&[tiles]))?;
            table.require_row(Self::NAME, DType::I32, tiles)?;
            if table.dtype != DType::I32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} table is not i32"),
                ));
            }
        }
        let segments = args.segments.shape.dim(0);
        if args.keep.shape.dim(0) != segments || args.keep.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                "the keep table does not hold one i32 entry per segment".to_owned(),
            ));
        }
        args.indices.require_shape(Self::NAME, args.scores.shape)?;
        args.indices.require_row(Self::NAME, DType::I32, tiles)?;
        if args.indices.strides != crate::tensor::Strides::row_major(args.indices.shape) {
            return Err(Error::argument(
                Self::NAME,
                "the index matrix is not contiguous".to_owned(),
            ));
        }
        args.counts
            .require_shape(Self::NAME, Shape::new(&[heads, query_tiles]))?;
        args.counts
            .require_row(Self::NAME, DType::I32, query_tiles)?;
        if args.query_offset > tiles - query_tiles {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "query tiles {query_tiles} at offset {} leave the {tiles} tiles",
                    args.query_offset
                ),
            ));
        }
        if descriptor.block < tiles.next_power_of_two() as u32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the descriptor's {} lanes cannot load {tiles} scores",
                    descriptor.block
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.scores)
            .view(args.segments)
            .view(args.sizes)
            .view(args.keep)
            .view(args.indices)
            .view(args.counts)
            .i32(query_tiles as i32)
            .i32(tiles as i32)
            .i32(args.query_offset as i32);
        kernel.launch(
            LaunchGrid::new([query_tiles, heads, 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        SelectTilesArgs {
            indices: twins.shadow_of(args.indices),
            counts: twins.shadow_of(args.counts),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // Every output is an index or a count of a total order over the scores.
        Tolerance::exact()
    }
}

/// One compilation of `vsa_prepare_tiles`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PrepareTilesDescriptor {
    /// Element type of the Q, K and V rows.
    pub dtype: DType,
    /// Heads per row.
    pub heads: u32,
    /// Features per head.
    pub width: u32,
    /// Features the rotation covers; zero when Q and K are used as they are.
    pub rotated: u32,
    /// Rows of one tile.
    pub tile_rows: u32,
    /// The rows one program handles.
    pub block_rows: u32,
    /// The epsilon of the Q/K RMS normalization.
    pub epsilon: Float,
    /// Whether Q and K are normalized and rotated before the tiles are pooled.
    pub norm: bool,
    /// Elements between query rows, and between their heads.
    pub query_row_stride: u64,
    /// Elements between query heads.
    pub query_head_stride: u64,
    /// Elements between key rows.
    pub key_row_stride: u64,
    /// Elements between key heads.
    pub key_head_stride: u64,
    /// Elements between value rows.
    pub value_row_stride: u64,
    /// Elements between value heads.
    pub value_head_stride: u64,
    /// Elements between the factor table's rows.
    pub factor_stride: u64,
}

impl fmt::Display for PrepareTilesDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("heads", self.heads)
            .field("width", self.width)
            .field("rotated", self.rotated)
            .field("tile_rows", self.tile_rows)
            .field("block_rows", self.block_rows)
            .field("epsilon", self.epsilon)
            .field("norm", self.norm)
            .field("query_row_stride", self.query_row_stride)
            .field("query_head_stride", self.query_head_stride)
            .field("key_row_stride", self.key_row_stride)
            .field("key_head_stride", self.key_head_stride)
            .field("value_row_stride", self.value_row_stride)
            .field("value_head_stride", self.value_head_stride)
            .field("factor_stride", self.factor_stride)
            .finish(formatter)
    }
}

/// The arguments of `vsa_prepare_tiles`.
#[derive(Clone, Copy)]
pub struct PrepareTilesArgs {
    /// `[tiles · tile_rows, heads, width]`, unit column stride: the query rows, normalized and
    /// rotated in place.
    pub query: TensorView,
    /// The key rows, of the query's shape.
    pub key: TensorView,
    /// The value rows, of the query's shape.
    pub value: TensorView,
    /// `[width]`: the Q normalization weight, present exactly when the descriptor normalizes.
    pub query_weight: Option<TensorView>,
    /// `[width]`: the K normalization weight, on the same condition.
    pub key_weight: Option<TensorView>,
    /// `[rows, rotated / 2]`: the cosine factor table, on the same condition.
    pub cosine: Option<TensorView>,
    /// `[rows, rotated / 2]`: the sine factor table, on the same condition.
    pub sine: Option<TensorView>,
    /// `[tiles]` i32: the valid rows of each tile.
    pub sizes: TensorView,
    /// `[tiles, heads, width]` FP32, contiguous: receives the query tiles' means.
    pub pooled_query: TensorView,
    /// Receives the key tiles' means.
    pub pooled_key: TensorView,
    /// Receives the value tiles' means.
    pub pooled_value: TensorView,
}

/// Normalize, rotate, zero and pool whole tiles of projected Q/K/V rows.
pub struct VsaPrepareTiles;

impl Op for VsaPrepareTiles {
    const NAME: &'static str = "vsa_prepare_tiles";

    type Descriptor = PrepareTilesDescriptor;
    type Args<'a> = PrepareTilesArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let heads = args.query.shape.dim(1);
        let width = args.query.shape.dim(2);
        if u64::from(descriptor.heads) != heads || u64::from(descriptor.width) != width {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the descriptor compiles for {} heads of {} features, the rows hold {heads} of \
                     {width}",
                    descriptor.heads, descriptor.width
                ),
            ));
        }
        for (name, view, row_stride, head_stride) in [
            (
                "query",
                &args.query,
                descriptor.query_row_stride,
                descriptor.query_head_stride,
            ),
            (
                "key",
                &args.key,
                descriptor.key_row_stride,
                descriptor.key_head_stride,
            ),
            (
                "value",
                &args.value,
                descriptor.value_row_stride,
                descriptor.value_head_stride,
            ),
        ] {
            check_rows(
                Self::NAME,
                descriptor.dtype,
                descriptor.heads,
                descriptor.width,
                view,
            )?;
            if view.shape != args.query.shape {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} rows do not have the query's shape"),
                ));
            }
            if stride(view, 0)? as u64 != row_stride || stride(view, 1)? as u64 != head_stride {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the {name} strides are ({}, {}), the descriptor says ({row_stride}, \
                         {head_stride})",
                        stride(view, 0)?,
                        stride(view, 1)?
                    ),
                ));
            }
        }
        if descriptor.tile_rows % descriptor.block_rows != 0 {
            return Err(Error::argument(
                Self::NAME,
                "the block rows do not divide a tile".to_owned(),
            ));
        }
        let rows = args.query.shape.dim(0);
        if rows % u64::from(descriptor.tile_rows) != 0 {
            return Err(Error::argument(
                Self::NAME,
                format!("{rows} rows are not whole tiles"),
            ));
        }
        let tiles = rows / u64::from(descriptor.tile_rows);
        args.sizes.require_shape(Self::NAME, Shape::new(&[tiles]))?;
        args.sizes.require_row(Self::NAME, DType::I32, tiles)?;
        let pooled = Shape::new(&[tiles, heads, width]);
        for (name, view) in [
            ("query", &args.pooled_query),
            ("key", &args.pooled_key),
            ("value", &args.pooled_value),
        ] {
            view.require_shape(Self::NAME, pooled)?;
            view.require_row(Self::NAME, DType::F32, width)?;
            if view.dtype != DType::F32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the pooled {name} rows are not FP32"),
                ));
            }
        }
        match (
            &args.query_weight,
            &args.key_weight,
            &args.cosine,
            &args.sine,
            descriptor.norm,
        ) {
            (Some(query_weight), Some(key_weight), Some(cosine), Some(sine), true) => {
                for view in [query_weight, key_weight] {
                    view.require_shape(Self::NAME, Shape::new(&[width]))?;
                    view.require_row(Self::NAME, descriptor.dtype, width)?;
                }
                for (name, view) in [("cosine", cosine), ("sine", sine)] {
                    view.require_shape(
                        Self::NAME,
                        Shape::new(&[rows, u64::from(descriptor.rotated / 2)]),
                    )?;
                    if view.dtype != descriptor.dtype {
                        return Err(Error::argument(
                            Self::NAME,
                            format!("the {name} table is not {}", descriptor.dtype),
                        ));
                    }
                    if view.strides.get(0) != descriptor.factor_stride {
                        return Err(Error::argument(
                            Self::NAME,
                            format!(
                                "the {name} table's rows are {} elements apart, the descriptor says \
                                 {}",
                                view.strides.get(0),
                                descriptor.factor_stride
                            ),
                        ));
                    }
                }
            }
            (None, None, None, None, false) => {}
            _ => {
                return Err(Error::argument(
                    Self::NAME,
                    "the normalization operands are all present or all absent".to_owned(),
                ));
            }
        }

        // The stepwise recipe's operands stand in with the rows they would have touched.
        let query_weight = args.query_weight.unwrap_or(args.query);
        let key_weight = args.key_weight.unwrap_or(args.key);
        let cosine = args.cosine.unwrap_or(args.query);
        let sine = args.sine.unwrap_or(args.query);
        let mut arguments = Arguments::new();
        arguments
            .view(args.query)
            .view(args.key)
            .view(args.value)
            .view(query_weight)
            .view(key_weight)
            .view(cosine)
            .view(sine)
            .view(args.sizes)
            .view(args.pooled_query)
            .view(args.pooled_key)
            .view(args.pooled_value);
        kernel.launch(LaunchGrid::new([tiles, heads, 1])?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        PrepareTilesArgs {
            query: twins.in_place(args.query, stream),
            key: twins.in_place(args.key, stream),
            value: twins.in_place(args.value, stream),
            pooled_query: twins.shadow_of(args.pooled_query),
            pooled_key: twins.shadow_of(args.pooled_key),
            pooled_value: twins.shadow_of(args.pooled_value),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The RMS statistics reduce over the head in FP32 and round once per stored feature; the
        // pooling then averages `tile_rows` rows in FP32. Both are bounded by the reduction's own
        // roundings.
        let terms = u64::from(descriptor.tile_rows) * u64::from(descriptor.width);
        Tolerance::rounding(terms, DType::F32, descriptor.dtype)
    }
}

/// Check a `[rows, heads, width]` row tensor's dtype and layout.
fn check_rows(
    operator: &'static str,
    dtype: DType,
    heads: u32,
    width: u32,
    view: &TensorView,
) -> Result<()> {
    view.require_rank(operator, 3)?;
    if view.shape.dim(1) != u64::from(heads) || view.shape.dim(2) != u64::from(width) {
        return Err(Error::argument(
            operator,
            format!(
                "the rows hold {} heads of {} features, the descriptor {heads} of {width}",
                view.shape.dim(1),
                view.shape.dim(2)
            ),
        ));
    }
    if view.dtype != dtype {
        return Err(Error::argument(
            operator,
            format!("expected {dtype} rows, found {}", view.dtype),
        ));
    }
    if view.strides.last() != 1 {
        return Err(Error::argument(
            operator,
            format!(
                "the rows' column stride is {}, not one",
                view.strides.last()
            ),
        ));
    }
    Ok(())
}

/// Check a tile and block that divide it.
fn check_block(operator: &'static str, tile_rows: u32, block_rows: u32) -> Result<()> {
    if block_rows == 0
        || !tile_rows.is_multiple_of(block_rows)
        || !BLOCK_ROWS.is_multiple_of(block_rows)
    {
        return Err(Error::argument(
            operator,
            format!("blocks of {block_rows} rows do not divide a tile of {tile_rows} rows"),
        ));
    }
    Ok(())
}

/// A stride the kernel takes as an `int32`.
fn stride(view: &TensorView, axis: usize) -> Result<i32> {
    let stride = view.strides.get(axis);
    i32::try_from(stride)
        .map_err(|_| Error::Shape(format!("the stride {stride} does not fit int32")))
}
