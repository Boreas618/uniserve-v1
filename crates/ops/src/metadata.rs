//! Gathering a call's attention metadata on the device.
//!
//! The executor holds the request tables (a slot's installed unit pages, each group's block size and
//! window) and the host columns of one call; attention needs them as the batch's own tensors — block
//! tables, start pages, lengths and positions. Gathering them on the device keeps the step free of
//! host round trips, so a captured graph replays a call whose rows moved
//! (`uniserve/nn/attention/gather.py`, specs/worker.md 9's metadata rows).
//!
//! ```text
//! attention_gather_rows            per table: the rows' selected pages, each token's append address,
//!                                 and the shared length and offset columns
//! attention_gather_decode_inputs   per table: one page per row, its token, position and slot
//! ```
//!
//! Both walk the same device copy of the host column array: [`ROW_SECTIONS`] sections of
//! [`row_stride`](GatherRowsDescriptor::row_stride) int64 elements (per row the slot, the prefix and
//! query lengths, whether the row writes the cache, then the exclusive query and prefix offsets),
//! then one section per table of its selected page count per row, then the tokens' rows. One program
//! per `(table, program index)` writes a disjoint part of the outputs, so no program reads another's
//! result and no barrier is needed. Every output is an index or a copy of one, so two
//! implementations agree bit for bit.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The sections of the host column array that are shared by every table: the slot, the prefix and
/// query lengths, the write flag, and the exclusive query and prefix offsets.
pub const ROW_SECTIONS: u32 = 6;

/// The cells one program copies.
pub const GATHER_BLOCK: u32 = 256;

/// The int64 elements one row section holds: one per row and its end offset.
pub fn row_columns_stride(max_rows: u64) -> u64 {
    max_rows + 1
}

/// The int64 elements the host column array of a call needs.
pub fn row_columns_size(max_rows: u64, tables: u64, max_tokens: u64) -> u64 {
    (u64::from(ROW_SECTIONS) + tables) * row_columns_stride(max_rows) + max_tokens
}

/// One compilation of `attention_gather_rows`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GatherRowsDescriptor {
    /// Element type of the host column array.
    pub columns: DType,
    /// Element type of the request and output tables.
    pub index: DType,
    /// The numerical tables of the call.
    pub tables: u32,
    /// Elements per row section.
    pub row_stride: u64,
    /// Elements from the columns' start to the tokens' section.
    pub token_section: u64,
    /// Elements between the unit tables.
    pub unit_table_stride: u64,
    /// Elements between a slot's rows in one unit table.
    pub unit_row_stride: u64,
    /// Elements in one request row of a unit table.
    pub request_width: u32,
    /// Elements between the start pages of two groups.
    pub start_group_stride: u64,
    /// Elements between two tables' block tables.
    pub block_table_stride: u64,
    /// Elements between two rows of one block table.
    pub block_table_row_stride: u64,
    /// Elements between two tables' start pages.
    pub start_page_stride: u64,
    /// Elements between two tables' append addresses.
    pub write_stride: u64,
    /// The lanes one program loads: `row_stride` rounded up to a power of two.
    pub row_block: u32,
    /// The cells one program copies.
    pub block: u32,
}

impl fmt::Display for GatherRowsDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("columns", self.columns)
            .field("index", self.index)
            .field("tables", self.tables)
            .field("row_stride", self.row_stride)
            .field("token_section", self.token_section)
            .field("unit_table_stride", self.unit_table_stride)
            .field("unit_row_stride", self.unit_row_stride)
            .field("request_width", self.request_width)
            .field("start_group_stride", self.start_group_stride)
            .field("block_table_stride", self.block_table_stride)
            .field("block_table_row_stride", self.block_table_row_stride)
            .field("start_page_stride", self.start_page_stride)
            .field("write_stride", self.write_stride)
            .field("row_block", self.row_block)
            .field("block", self.block)
            .finish(formatter)
    }
}

/// The arguments of `attention_gather_rows`.
#[derive(Clone, Copy)]
pub struct GatherRowsArgs {
    /// `[row_columns_size]` int64: the device copy of the call's host columns.
    pub columns: TensorView,
    /// `[tables, slots, request_width]`: each slot's installed units per table.
    pub request_unit_tables: TensorView,
    /// `[groups, slots]`: the slot's installed start page in each group.
    pub request_start_pages: TensorView,
    /// `[tables, 3]`: each table's group, block size and window (-1 for none).
    pub table_shapes: TensorView,
    /// `[tables, rows, width]`: receives each row's block table.
    pub block_tables: TensorView,
    /// `[tables, rows]`: receives each row's first page.
    pub start_pages: TensorView,
    /// `[rows]`: receives each row's cache length.
    pub cache_lengths: TensorView,
    /// `[rows]`: receives each row's query length.
    pub query_lengths: TensorView,
    /// `[rows + 1]`: receives the exclusive query offsets.
    pub query_offsets: TensorView,
    /// `[rows + 1]`: receives the exclusive prefix offsets.
    pub prefix_offsets: TensorView,
    /// `[tables, tokens]`: receives each token's append address, -1 for a row that does not write.
    pub slot_mapping: TensorView,
    /// The call's rows.
    pub rows: u64,
    /// The call's tokens.
    pub tokens: u64,
    /// The programs that copy table cells: `ceil(rows · width / block)`.
    pub cell_programs: u64,
    /// The cells of a row the call selects.
    pub width: u64,
}

/// Gather every table's columns of a call's request rows.
pub struct AttentionGatherRows;

impl Op for AttentionGatherRows {
    const NAME: &'static str = "attention_gather_rows";

    type Descriptor = GatherRowsDescriptor;
    type Args<'a> = GatherRowsArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let tables = u64::from(descriptor.tables);
        if tables == 0 || descriptor.block == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a call needs at least one table and a non-empty copy block".to_owned(),
            ));
        }
        if !descriptor.row_block.is_power_of_two()
            || u64::from(descriptor.row_block) < descriptor.row_stride
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a row block of {} lanes does not cover {} columns",
                    descriptor.row_block, descriptor.row_stride
                ),
            ));
        }
        if descriptor.token_section != (u64::from(ROW_SECTIONS) + tables) * descriptor.row_stride {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the token section {} is not past the {} shared and per-table sections",
                    descriptor.token_section,
                    u64::from(ROW_SECTIONS) + tables
                ),
            ));
        }
        check_type(
            Self::NAME,
            "the host columns",
            &args.columns,
            descriptor.columns,
        )?;
        args.columns.require_rank(Self::NAME, 1)?;
        if args.columns.shape.dim(0)
            < row_columns_size(
                args.block_tables.shape.dim(1),
                tables,
                args.slot_mapping.shape.dim(1),
            )
        {
            return Err(Error::argument(
                Self::NAME,
                "the host columns are shorter than the call's rows, tables and tokens need"
                    .to_owned(),
            ));
        }
        for (name, view, dtype) in [
            (
                "request unit tables",
                &args.request_unit_tables,
                descriptor.index,
            ),
            (
                "request start pages",
                &args.request_start_pages,
                descriptor.index,
            ),
            ("table shapes", &args.table_shapes, descriptor.index),
            ("block tables", &args.block_tables, descriptor.index),
            ("start pages", &args.start_pages, descriptor.index),
            ("slot mapping", &args.slot_mapping, descriptor.index),
        ] {
            check_type(Self::NAME, name, view, dtype)?;
        }
        for (name, view) in [
            ("cache lengths", &args.cache_lengths),
            ("query lengths", &args.query_lengths),
        ] {
            check_type(Self::NAME, name, view, descriptor.index)?;
            view.require_shape(Self::NAME, Shape::new(&[args.rows]))?;
            view.require_row(Self::NAME, descriptor.index, args.rows)?;
        }
        for (name, view) in [
            ("query offsets", &args.query_offsets),
            ("prefix offsets", &args.prefix_offsets),
        ] {
            check_type(Self::NAME, name, view, descriptor.index)?;
            view.require_shape(Self::NAME, Shape::new(&[args.rows + 1]))?;
        }
        if args.request_unit_tables.shape.dim(0) != tables {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the request tables hold {} tables, the descriptor {tables}",
                    args.request_unit_tables.shape.dim(0)
                ),
            ));
        }
        args.table_shapes
            .require_shape(Self::NAME, Shape::new(&[tables, 3]))?;
        let max_rows = args.block_tables.shape.dim(1);
        let max_width = args.block_tables.shape.dim(2);
        if args.rows == 0 || args.rows > max_rows || args.width == 0 || args.width > max_width {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} rows of {} cells leave a table of {max_rows} rows of {max_width} cells",
                    args.rows, args.width
                ),
            ));
        }
        if args.tokens == 0 || args.tokens > args.slot_mapping.shape.dim(1) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} tokens leave a mapping of {} tokens",
                    args.tokens,
                    args.slot_mapping.shape.dim(1)
                ),
            ));
        }
        if args.cell_programs != (args.rows * args.width).div_ceil(u64::from(descriptor.block)) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} cell programs do not cover {} cells in blocks of {}",
                    args.cell_programs,
                    args.rows * args.width,
                    descriptor.block
                ),
            ));
        }
        for (name, view, stride, axis) in [
            (
                "request unit tables",
                &args.request_unit_tables,
                descriptor.unit_table_stride,
                0,
            ),
            (
                "request unit tables",
                &args.request_unit_tables,
                descriptor.unit_row_stride,
                1,
            ),
            (
                "request start pages",
                &args.request_start_pages,
                descriptor.start_group_stride,
                0,
            ),
            (
                "block tables",
                &args.block_tables,
                descriptor.block_table_stride,
                0,
            ),
            (
                "block tables",
                &args.block_tables,
                descriptor.block_table_row_stride,
                1,
            ),
            (
                "start pages",
                &args.start_pages,
                descriptor.start_page_stride,
                0,
            ),
            (
                "slot mapping",
                &args.slot_mapping,
                descriptor.write_stride,
                0,
            ),
        ] {
            if view.shape.rank() > axis && view.strides.get(axis) != stride {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the {name} stride {axis} is {}, the descriptor says {stride}",
                        view.strides.get(axis)
                    ),
                ));
            }
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.columns)
            .view(args.request_unit_tables)
            .view(args.request_start_pages)
            .view(args.table_shapes)
            .view(args.block_tables)
            .view(args.start_pages)
            .view(args.cache_lengths)
            .view(args.query_lengths)
            .view(args.query_offsets)
            .view(args.prefix_offsets)
            .view(args.slot_mapping)
            .i64(args.rows as i64)
            .i64(args.tokens as i64)
            .i64(args.cell_programs as i64)
            .i64(args.width as i64);
        kernel.launch(
            LaunchGrid::new([
                1 + args.cell_programs + args.tokens.div_ceil(u64::from(descriptor.block)),
                tables,
                1,
            ])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        GatherRowsArgs {
            block_tables: twins.shadow_of(args.block_tables),
            start_pages: twins.shadow_of(args.start_pages),
            cache_lengths: twins.shadow_of(args.cache_lengths),
            query_lengths: twins.shadow_of(args.query_lengths),
            query_offsets: twins.shadow_of(args.query_offsets),
            prefix_offsets: twins.shadow_of(args.prefix_offsets),
            slot_mapping: twins.shadow_of(args.slot_mapping),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // Every output is an index, a length or a copy of one.
        Tolerance::exact()
    }
}

/// Check a view's element type against the descriptor.
fn check_type(
    operator: &'static str,
    name: &str,
    view: &TensorView,
    expected: DType,
) -> Result<()> {
    if view.dtype != expected {
        return Err(Error::argument(
            operator,
            format!("{name} are {}, the descriptor says {expected}", view.dtype),
        ));
    }
    Ok(())
}

/// One compilation of `attention_gather_decode_inputs`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GatherDecodeInputsDescriptor {
    /// Element type of the request and output tables, the ids and the positions.
    pub index: DType,
    /// The numerical tables of the call.
    pub tables: u32,
    /// Elements between the unit tables.
    pub unit_table_stride: u64,
    /// Elements between a slot's rows in one unit table.
    pub unit_row_stride: u64,
    /// Elements in one request row of a unit table.
    pub request_width: u32,
    /// Elements between the start pages of two groups.
    pub start_group_stride: u64,
    /// Elements between two slots of the request token rows.
    pub request_token_stride: u64,
    /// Elements between two slots of the request position rows.
    pub request_position_stride: u64,
    /// Elements between two position axes.
    pub position_axis_stride: u64,
    /// The position axes of the call: one, or three for video.
    pub position_axes: u32,
    /// Elements between two tables' block tables.
    pub block_table_stride: u64,
    /// Elements between two rows of one block table.
    pub block_table_row_stride: u64,
    /// Elements between two tables' start pages.
    pub start_page_stride: u64,
    /// Elements between two tables' append addresses.
    pub write_stride: u64,
    /// The lanes one program loads: the row count rounded up to a power of two.
    pub row_block: u32,
    /// The cells one program copies.
    pub block: u32,
}

impl fmt::Display for GatherDecodeInputsDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("index", self.index)
            .field("tables", self.tables)
            .field("unit_table_stride", self.unit_table_stride)
            .field("unit_row_stride", self.unit_row_stride)
            .field("request_width", self.request_width)
            .field("start_group_stride", self.start_group_stride)
            .field("request_token_stride", self.request_token_stride)
            .field("request_position_stride", self.request_position_stride)
            .field("position_axis_stride", self.position_axis_stride)
            .field("position_axes", self.position_axes)
            .field("block_table_stride", self.block_table_stride)
            .field("block_table_row_stride", self.block_table_row_stride)
            .field("start_page_stride", self.start_page_stride)
            .field("write_stride", self.write_stride)
            .field("row_block", self.row_block)
            .field("block", self.block)
            .finish(formatter)
    }
}

/// The arguments of `attention_gather_decode_inputs`.
#[derive(Clone, Copy)]
pub struct GatherDecodeInputsArgs {
    /// `[rows]`: each decode row's scheduler-owned request slot; slot 0 pads the row.
    pub request_pool_indices: TensorView,
    /// `[tables, slots, request_width]`: each slot's installed units per table.
    pub request_unit_tables: TensorView,
    /// `[groups, slots]`: the slot's installed start page in each group.
    pub request_start_pages: TensorView,
    /// `[tables, 3]`: each table's group, block size and window (-1 for none).
    pub table_shapes: TensorView,
    /// `[slots]`: each slot's cache length.
    pub request_cache_lengths: TensorView,
    /// `[slots, ...]`: each slot's next token id.
    pub request_tokens: TensorView,
    /// `[slots, ...]`: each slot's next position per axis.
    pub request_positions: TensorView,
    /// `[rows]`: receives the rows' token ids, 1 for padding rows.
    pub input_ids: TensorView,
    /// `[axes, rows]`: receives the rows' positions, zero for padding rows.
    pub positions: TensorView,
    /// `[tables, rows, columns]`: receives each row's block table.
    pub block_tables: TensorView,
    /// `[tables, rows]`: receives each row's first page.
    pub start_pages: TensorView,
    /// `[rows]`: receives each row's cache length.
    pub cache_lengths: TensorView,
    /// `[rows]`: receives each row's query length, one for padding rows.
    pub query_lengths: TensorView,
    /// `[rows + 1]`: receives the exclusive query offsets.
    pub query_offsets: TensorView,
    /// `[rows + 1]`: receives the exclusive prefix offsets.
    pub prefix_offsets: TensorView,
    /// `[tables, rows]`: receives each row's append address, -1 for a row that does not write.
    pub slot_mapping: TensorView,
    /// The cells of a row the call selects.
    pub columns: u64,
}

/// Gather the request state of one decode step into the batch's fixed-capacity buffers.
pub struct AttentionGatherDecodeInputs;

impl Op for AttentionGatherDecodeInputs {
    const NAME: &'static str = "attention_gather_decode_inputs";

    type Descriptor = GatherDecodeInputsDescriptor;
    type Args<'a> = GatherDecodeInputsArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let tables = u64::from(descriptor.tables);
        if tables == 0 || descriptor.block == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a call needs at least one table and a non-empty copy block".to_owned(),
            ));
        }
        if !descriptor.row_block.is_power_of_two() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the row block {} is not a power of two",
                    descriptor.row_block
                ),
            ));
        }
        if !matches!(descriptor.position_axes, 1 | 3) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} position axes are neither one nor three",
                    descriptor.position_axes
                ),
            ));
        }
        args.request_pool_indices.require_rank(Self::NAME, 1)?;
        let rows = args.request_pool_indices.shape.dim(0);
        if rows == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a decode step needs at least one row".to_owned(),
            ));
        }
        if u64::from(descriptor.row_block) < rows {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a row block of {} lanes cannot load {rows} rows",
                    descriptor.row_block
                ),
            ));
        }
        for (name, view) in [
            ("the request slots", &args.request_pool_indices),
            ("the request tables", &args.request_unit_tables),
            ("the request start pages", &args.request_start_pages),
            ("the table shapes", &args.table_shapes),
            ("the request cache lengths", &args.request_cache_lengths),
            ("the request tokens", &args.request_tokens),
            ("the request positions", &args.request_positions),
            ("the token ids", &args.input_ids),
            ("the positions", &args.positions),
            ("the block tables", &args.block_tables),
            ("the start pages", &args.start_pages),
            ("the cache lengths", &args.cache_lengths),
            ("the query lengths", &args.query_lengths),
            ("the query offsets", &args.query_offsets),
            ("the prefix offsets", &args.prefix_offsets),
            ("the slot mapping", &args.slot_mapping),
        ] {
            check_type(Self::NAME, name, view, descriptor.index)?;
        }
        args.request_unit_tables.require_rank(Self::NAME, 3)?;
        if args.request_unit_tables.shape.dim(0) != tables {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the request tables hold {} tables, the descriptor {tables}",
                    args.request_unit_tables.shape.dim(0)
                ),
            ));
        }
        args.request_start_pages.require_rank(Self::NAME, 2)?;
        args.table_shapes
            .require_shape(Self::NAME, Shape::new(&[tables, 3]))?;
        args.block_tables.require_rank(Self::NAME, 3)?;
        if args.block_tables.shape.dim(0) != tables {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the block tables hold {} tables, the descriptor {tables}",
                    args.block_tables.shape.dim(0)
                ),
            ));
        }
        let capacity = args.block_tables.shape.dim(1);
        let max_columns = args.block_tables.shape.dim(2);
        if rows > capacity || args.columns == 0 || args.columns > max_columns {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{rows} rows of {} cells leave a table of {capacity} rows of {max_columns} \
                     cells",
                    args.columns
                ),
            ));
        }
        for (name, view) in [
            ("the start pages", &args.start_pages),
            ("the slot mapping", &args.slot_mapping),
        ] {
            view.require_rank(Self::NAME, 2)?;
            if view.shape.dim(0) != tables || view.shape.dim(1) < rows {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} do not hold {tables} tables of {rows} rows"),
                ));
            }
        }
        for (name, view, extent) in [
            ("the token ids", &args.input_ids, rows),
            ("the cache lengths", &args.cache_lengths, rows),
            ("the query lengths", &args.query_lengths, rows),
            ("the query offsets", &args.query_offsets, rows + 1),
            ("the prefix offsets", &args.prefix_offsets, rows + 1),
        ] {
            if view.shape.numel() < extent {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} hold fewer than {extent} entries"),
                ));
            }
        }
        if args.positions.shape.rank() == 2 {
            if args.positions.shape.dim(0) != u64::from(descriptor.position_axes)
                || args.positions.shape.dim(1) < rows
            {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the positions are not [{} axes, {rows} rows]",
                        descriptor.position_axes
                    ),
                ));
            }
        } else {
            return Err(Error::argument(
                Self::NAME,
                "the positions are not one row per axis".to_owned(),
            ));
        }
        for (name, view, stride, axis) in [
            (
                "request tables",
                &args.request_unit_tables,
                descriptor.unit_table_stride,
                0,
            ),
            (
                "request tables",
                &args.request_unit_tables,
                descriptor.unit_row_stride,
                1,
            ),
            (
                "request start pages",
                &args.request_start_pages,
                descriptor.start_group_stride,
                0,
            ),
            (
                "request tokens",
                &args.request_tokens,
                descriptor.request_token_stride,
                0,
            ),
            (
                "request positions",
                &args.request_positions,
                descriptor.request_position_stride,
                0,
            ),
            (
                "positions",
                &args.positions,
                descriptor.position_axis_stride,
                0,
            ),
            (
                "block tables",
                &args.block_tables,
                descriptor.block_table_stride,
                0,
            ),
            (
                "block tables",
                &args.block_tables,
                descriptor.block_table_row_stride,
                1,
            ),
            (
                "start pages",
                &args.start_pages,
                descriptor.start_page_stride,
                0,
            ),
            (
                "slot mapping",
                &args.slot_mapping,
                descriptor.write_stride,
                0,
            ),
        ] {
            if view.shape.rank() > axis && view.strides.get(axis) != stride {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the {name} stride {axis} is {}, the descriptor says {stride}",
                        view.strides.get(axis)
                    ),
                ));
            }
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.request_pool_indices)
            .view(args.request_unit_tables)
            .view(args.request_start_pages)
            .view(args.table_shapes)
            .view(args.request_cache_lengths)
            .view(args.request_tokens)
            .view(args.request_positions)
            .view(args.input_ids)
            .view(args.positions)
            .view(args.block_tables)
            .view(args.start_pages)
            .view(args.cache_lengths)
            .view(args.query_lengths)
            .view(args.query_offsets)
            .view(args.prefix_offsets)
            .view(args.slot_mapping)
            .i64(rows as i64)
            .i64(args.columns as i64);
        kernel.launch(
            LaunchGrid::new([
                1 + (rows * args.columns).div_ceil(u64::from(descriptor.block)),
                tables,
                1,
            ])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        GatherDecodeInputsArgs {
            input_ids: twins.shadow_of(args.input_ids),
            positions: twins.shadow_of(args.positions),
            block_tables: twins.shadow_of(args.block_tables),
            start_pages: twins.shadow_of(args.start_pages),
            cache_lengths: twins.shadow_of(args.cache_lengths),
            query_lengths: twins.shadow_of(args.query_lengths),
            query_offsets: twins.shadow_of(args.query_offsets),
            prefix_offsets: twins.shadow_of(args.prefix_offsets),
            slot_mapping: twins.shadow_of(args.slot_mapping),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // Every output is an index, a length, a position or a token id copied from the request
        // tables; padding rows take the fixed values the reference initializes them with.
        Tolerance::exact()
    }
}
