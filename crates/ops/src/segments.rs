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

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
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
            .i32(stride(&args.compressed, 0)?)
            .i32(stride(&args.compressed, 1)?);
        kernel.launch(
            LaunchGrid::new([rows / u64::from(descriptor.block_rows), heads, 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
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

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
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

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
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
                    // The factors are the rotary operator's output, FP32 whatever the rows' type:
                    // the kernel rounds each product to the rows' type as two eager operations do.
                    if view.dtype != DType::F32 {
                        return Err(Error::argument(
                            Self::NAME,
                            format!("the {name} table is not f32"),
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

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
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

/// One compilation of `vsa_scores`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ScoresDescriptor {
    /// Heads per row.
    pub heads: u32,
    /// Features per head: the contraction.
    pub width: u32,
    /// Lanes one program uses for the contraction, rounded up to a power of two.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl ScoresDescriptor {
    /// The descriptor of a pooled-score product.
    ///
    /// There is no element-type field because there is only one type the operands can have:
    /// `vsa_prepare_tiles` writes each tile's per-head means as FP32 — the accumulation its pooling
    /// runs in — and the reference computes the similarities in FP32 as well, so a BF16 or F16
    /// product would have to narrow operands that are already FP32.
    pub fn new(heads: u32, width: u32, block: u32, pdl: bool) -> Self {
        Self {
            heads,
            width,
            block,
            pdl,
        }
    }
}

impl fmt::Display for ScoresDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("heads", self.heads)
            .field("width", self.width)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `vsa_scores`.
#[derive(Clone, Copy)]
pub struct ScoresArgs {
    /// `[query tiles, heads, width]` FP32: the pooled query rows, in the layout
    /// `vsa_prepare_tiles` writes them.
    pub pooled_query: TensorView,
    /// `[tiles, heads, width]` FP32: the pooled key rows, in that same layout.
    pub pooled_key: TensorView,
    /// `[heads, query tiles, tiles]` FP32: one dot product per (query tile, key tile).
    pub out: TensorView,
}

/// The pooled (query tile, key tile) similarities VSA's selection ranks.
///
/// This is the step between pooling and selection: `vsa_prepare_tiles` pools each tile's per-head
/// means, `vsa_scores` contracts them per head, and `vsa_select_tiles` ranks the result. Without it
/// the sparse path has no scores, and nothing else fills the gap: `gemm` has no batch axis, and the
/// contraction is per head.
///
/// It is exactly a per-head dot product — no scale and no softmax, because the caller applies both
/// afterwards (the selection only compares, and the attention scales its own products), so the
/// program writes the raw sum of products and nothing else.
///
/// **The operands are the pooled tiles in the layout the pooling writes them**: a tile is the
/// leading axis, then the head, then the feature. The scores are the other way round — head first —
/// because `vsa_select_tiles` reads one row of `tiles` per (head, query tile) contiguously, and the
/// selection is what consumes them.
pub struct VsaScores;

impl Op for VsaScores {
    const NAME: &'static str = "vsa_scores";

    type Descriptor = ScoresDescriptor;
    type Args<'a> = ScoresArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let width = u64::from(descriptor.width);
        let heads = u64::from(descriptor.heads);
        if width == 0 || heads == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a score product needs at least one head of at least one feature".to_owned(),
            ));
        }
        if args.pooled_query.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the pooled query rows are FP32, the view says {}",
                    args.pooled_query.dtype
                ),
            ));
        }
        if args.pooled_key.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the pooled key rows are FP32, the view says {}",
                    args.pooled_key.dtype
                ),
            ));
        }
        args.pooled_query.require_rank(Self::NAME, 3)?;
        args.pooled_key.require_rank(Self::NAME, 3)?;
        // A tile's pooled row is `[heads, width]` and the tile is its leading axis, so both operands
        // carry the same two inner extents; a view whose planes are another head count or another
        // width would contract against the wrong operands and produce plausible scores.
        for (name, view) in [("query", args.pooled_query), ("key", args.pooled_key)] {
            if view.shape.dim(1) != heads || view.shape.dim(2) != width {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the pooled {name} tiles are [_, {}, {}], the descriptor's [_, {heads}, \
                         {width}]",
                        view.shape.dim(1),
                        view.shape.dim(2)
                    ),
                ));
            }
        }
        let query_tiles = args.pooled_query.shape.dim(0);
        let key_tiles = args.pooled_key.shape.dim(0);
        if args.out.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                "the scores are FP32: the selection compares them".to_owned(),
            ));
        }
        args.out
            .require_shape(Self::NAME, Shape::new(&[heads, query_tiles, key_tiles]))?;
        // The selection reads the scores as a contiguous `[heads, query tiles, tiles]` matrix, so
        // this operator writes exactly that layout and no other.
        if args.out.strides != crate::tensor::Strides::row_major(args.out.shape) {
            return Err(Error::argument(
                Self::NAME,
                "the score matrix is written row-major".to_owned(),
            ));
        }
        if u64::from(descriptor.block) < width.next_power_of_two() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the descriptor's {} lanes cannot hold the {width}-feature contraction",
                    descriptor.block
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.pooled_query)
            .view(args.pooled_key)
            .view(args.out)
            .i32(query_tiles as i32)
            .i32(key_tiles as i32)
            .flag(descriptor.pdl);
        // One program per (head, query tile): it walks the key tiles and writes one row of scores.
        kernel.launch(
            LaunchGrid::new([query_tiles, heads, 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        ScoresArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The standard bound for the contraction and nothing else: `width` products per score,
        // accumulated in FP32 and written as FP32, with no activation between them — so the reduction
        // length is the feature width and the output's rounding is the accumulator's (whose `u` is
        // already in the bound at FP32).
        Tolerance::dot(u64::from(descriptor.width), DType::F32, DType::F32)
    }
}

#[cfg(test)]
mod scores_tests {
    use super::*;

    #[test]
    fn the_key_names_the_contraction_and_the_lanes() {
        let descriptor = ScoresDescriptor::new(56, 128, 128, false);
        assert_eq!(
            descriptor.to_string(),
            "heads_56_width_128_block_128_pdl_false"
        );
    }

    #[test]
    fn the_bound_is_the_contraction_and_the_accumulator() {
        // Two widths differ only in the reduction length, and the bound follows it: a wider
        // contraction cannot share a bound with a narrower one, which is why `width` is a key field.
        let narrow = VsaScores::tolerance(&ScoresDescriptor::new(56, 64, 64, false));
        let wide = VsaScores::tolerance(&ScoresDescriptor::new(56, 256, 256, false));
        assert!(wide.rtol > narrow.rtol);
    }
}

/// One compilation of `vsa_compress_pooled`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CompressPooledDescriptor {
    /// Element type of the compressed rows: the attention's own, which the gate adds to.
    pub dtype: DType,
    /// Heads per row.
    pub heads: u32,
    /// Features per head: the width one program sums.
    pub width: u32,
    /// The bound on the key tiles one query tile's softmax reduces over: the deployment's own tile
    /// count, and the reduction length its tolerance is written against.
    ///
    /// The **key names its power of two**, because that is what the package compiles the reduction
    /// to: a program compiled for a bound of 640 and one for 1024 are the same module, and a key
    /// that told them apart would demand a compilation per tile count a deployment happens to
    /// request - none of which any enumeration could list. The bound itself stays this value, and so
    /// does the tolerance, which is a property of the reduction and not of the module.
    pub max_tiles: u32,
    /// Lanes one program holds for the weighted sum of features, rounded up to a power of two.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl CompressPooledDescriptor {
    /// The descriptor of one pooled compression.
    ///
    /// The scores and the pooled values are FP32 and only the result narrows: `vsa_prepare_tiles`
    /// pools in FP32 and the reference's softmax runs in FP32, so a descriptor naming a narrower
    /// operand would describe a program that narrows before the softmax and biases every weight.
    pub fn new(
        dtype: DType,
        heads: u32,
        width: u32,
        max_tiles: u32,
        block: u32,
        pdl: bool,
    ) -> Self {
        Self {
            dtype,
            heads,
            width,
            max_tiles,
            block,
            pdl,
        }
    }
}

impl fmt::Display for CompressPooledDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("heads", self.heads)
            .field("width", self.width)
            .field("max_tiles", self.max_tiles.next_power_of_two())
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `vsa_compress_pooled`.
#[derive(Clone, Copy)]
pub struct CompressPooledArgs {
    /// `[heads, query tiles, tiles]` FP32: the pooled similarities, in `vsa_scores`' layout.
    pub scores: TensorView,
    /// `[tiles, heads, width]` FP32: the pooled value rows, in `vsa_prepare_tiles`' layout.
    pub values: TensorView,
    /// `[tiles]` i32: the live rows of each key tile.
    ///
    /// A tile of no live rows is invisible to the softmax — its score is masked, not merely zeroed —
    /// because a zeroed pooled value under a live score would still take weight and scale the sum
    /// of the others down.
    pub sizes: TensorView,
    /// `[heads, query tiles, width]` in the descriptor's element type: each query tile's
    /// tile-weighted mean of the pooled value rows, which `vsa_add_gated_tiles` gates and adds.
    pub out: TensorView,
}

/// The compression branch of VSA: each query tile's softmax over the live key tiles weights their
/// pooled values.
///
/// The reference (`uniserve/nn/attention/vsa/segments.py:466-479`) is
/// `weights = scores.masked_fill(sizes <= 0, -inf).softmax(-1)` over the key tiles, then
/// `compressed = weights @ pooled_value` in FP32 and one rounding to the rows' element type. Its
/// result is per (head, query tile) and is the same for every row of the tile, which is why the
/// output carries no row axis: the gate step broadcasts it.
///
/// Without this operator the sparse branch has no compression and a sparse checkpoint would have to
/// be served as a different model — a dense masked softmax over every live tile — which is why the
/// model refuses one until it exists.
pub struct VsaCompressPooled;

impl Op for VsaCompressPooled {
    const NAME: &'static str = "vsa_compress_pooled";

    type Descriptor = CompressPooledDescriptor;
    type Args<'a> = CompressPooledArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let heads = u64::from(descriptor.heads);
        let width = u64::from(descriptor.width);
        if heads == 0 || width == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a compression needs at least one head of at least one feature".to_owned(),
            ));
        }
        if !matches!(descriptor.dtype, DType::BF16 | DType::F16 | DType::F32) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the compression is written as the attention's rows, not {}",
                    descriptor.dtype
                ),
            ));
        }
        if args.scores.dtype != DType::F32 || args.values.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the pooled scores and values are FP32, the views say {} and {}",
                    args.scores.dtype, args.values.dtype
                ),
            ));
        }
        args.scores.require_rank(Self::NAME, 3)?;
        args.values.require_rank(Self::NAME, 3)?;
        if args.scores.shape.dim(0) != heads || args.values.shape.dim(1) != heads {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the scores are [{}, _, _] and the values [_, {}, _], the descriptor's {heads} \
                     heads",
                    args.scores.shape.dim(0),
                    args.values.shape.dim(1)
                ),
            ));
        }
        if args.values.shape.dim(2) != width {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the pooled values carry {} features, the descriptor's {width}",
                    args.values.shape.dim(2)
                ),
            ));
        }
        let query_tiles = args.scores.shape.dim(1);
        let tiles = args.scores.shape.dim(2);
        if args.values.shape.dim(0) != tiles {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} key tiles of scores against {} of pooled values",
                    tiles,
                    args.values.shape.dim(0)
                ),
            ));
        }
        if tiles > u64::from(descriptor.max_tiles) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{tiles} key tiles beyond the {} this module was accepted for",
                    descriptor.max_tiles
                ),
            ));
        }
        args.sizes.require_shape(Self::NAME, Shape::new(&[tiles]))?;
        args.sizes.require_row(Self::NAME, DType::I32, tiles)?;
        args.out
            .require_shape(Self::NAME, Shape::new(&[heads, query_tiles, width]))?;
        if args.out.dtype != descriptor.dtype {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the compression is {} and the view says {}",
                    descriptor.dtype, args.out.dtype
                ),
            ));
        }
        // The gate step reads one contiguous `[heads, query tiles, width]` matrix, so this operator
        // writes exactly that layout and no other.
        if args.out.strides != crate::tensor::Strides::row_major(args.out.shape) {
            return Err(Error::argument(
                Self::NAME,
                "the compression is written row-major".to_owned(),
            ));
        }
        if u64::from(descriptor.block) < width.next_power_of_two() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the descriptor's {} lanes cannot hold the {width}-feature sum",
                    descriptor.block
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.scores)
            .view(args.values)
            .view(args.sizes)
            .view(args.out)
            .i32(query_tiles as i32)
            .i32(tiles as i32)
            .flag(descriptor.pdl);
        // One program per (head, query tile): it reduces the key tiles and writes one row.
        kernel.launch(
            LaunchGrid::new([query_tiles, heads, 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        CompressPooledArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // Two reductions in one program: the softmax over the key tiles and the weighted sum of the
        // features, both accumulated in FP32 with one rounding to the rows' element type. The
        // softmax's weights are a scale of positive terms and the sum is of bounded weights, so the
        // bound is the standard one over the longer reduction — the tile bound — with FP32
        // accumulation and the output's own rounding.
        Tolerance::dot(
            u64::from(descriptor.max_tiles.max(descriptor.width)),
            DType::F32,
            descriptor.dtype,
        )
    }
}

#[cfg(test)]
mod compress_pooled_tests {
    use super::*;

    #[test]
    fn the_key_names_the_geometry_the_tile_bound_and_the_lanes() {
        let descriptor = CompressPooledDescriptor::new(DType::BF16, 56, 128, 4_096, 128, false);
        assert_eq!(
            descriptor.to_string(),
            "dtype_bf16_heads_56_width_128_max_tiles_4096_block_128_pdl_false"
        );
    }

    #[test]
    fn the_bound_follows_the_tile_bound_and_the_output_rounding() {
        let narrow = VsaCompressPooled::tolerance(&CompressPooledDescriptor::new(
            DType::F32,
            56,
            128,
            1_024,
            128,
            false,
        ));
        let wide = VsaCompressPooled::tolerance(&CompressPooledDescriptor::new(
            DType::F32,
            56,
            128,
            8_192,
            128,
            false,
        ));
        assert!(wide.rtol > narrow.rtol);
        // The output's rounding is in the bound: a narrower result type cannot share it.
        let narrowed = VsaCompressPooled::tolerance(&CompressPooledDescriptor::new(
            DType::BF16,
            56,
            128,
            1_024,
            128,
            false,
        ));
        assert!(narrowed.rtol > narrow.rtol);
    }
}
