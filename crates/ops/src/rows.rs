//! `rows`: place or accumulate the rows a model picks out by index.
//!
//! Two modalities feed rows into a sequence: an image's vision tokens replace the placeholder rows
//! the text has reserved for them, and a deep-stack tap adds a tower's rows into rows that already
//! hold a residual. They differ in one operation and share everything else — the index of the
//! destination row, the width of a row, and the order the rows are copied in — so they are one
//! operator with a mode rather than two, and the mode is compiled in:
//!
//! ```text
//! replace: destination[indices[j]] = source[j]          (a copy)
//! add:     destination[indices[j]] += source[j]         (one rounding into the output's dtype)
//! ```
//!
//! `indices` is `[count]` i32 and may repeat: a repeated index is written, or added, more than once,
//! in the order it appears, so the caller controls the sum order and repeated calls of one launch
//! agree bit for bit. The rows themselves are `[rows, columns]` and the source is `[count, columns]`;
//! an index outside `[0, rows)` is refused here rather than read on the device.
//!
//! The program is a kernel program: one thread per (row, element) of the source, the row taken from
//! the second axis and the element block from the third, so a row's elements are never split across
//! two blocks that could race on the same destination row. Parameters, in order: `source`,
//! `destination`, `indices`, `count` (u32), `columns` (u32), `block` (u32), the PDL flag.
//!
//! The tolerances differ because the operations do: a replace moves elements and is exact, an add
//! performs one addition in the output's dtype and is bounded by two units of its own rounding.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, Strides, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// What a launch does with the rows it picks out.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mode {
    /// The destination row becomes the source row.
    Replace,
    /// The destination row gains the source row.
    Add,
}

impl fmt::Display for Mode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Replace => "replace",
            Self::Add => "add",
        })
    }
}

/// One compilation of `rows`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the destination and the source: both, because they are the same tensor's.
    pub dtype: DType,
    /// What the launch does with the picked-out rows.
    pub mode: Mode,
    /// Elements per row.
    pub columns: u32,
    /// Elements one thread block handles.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("mode", self.mode)
            .field("columns", self.columns)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `rows`.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[count, columns]` the rows to place.
    pub source: TensorView,
    /// `[rows, columns]` the sequence the rows land in.
    pub destination: TensorView,
    /// `[count]` i32: the destination row of each source row.
    pub indices: TensorView,
}

/// Row placement and accumulation.
pub struct Rows;

impl Op for Rows {
    const NAME: &'static str = "rows";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let columns = u64::from(descriptor.columns);
        let count = args.indices.numel();

        args.indices
            .require_shape(Self::NAME, Shape::new(&[count]))?;
        if args.indices.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected i32 row indices, found {}", args.indices.dtype),
            ));
        }
        args.source
            .require_shape(Self::NAME, Shape::new(&[count, columns]))?;
        args.source
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        args.destination
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        // The destination is walked as a grid of rows: a row that is not `columns` elements long
        // would make the row stride meaningless, and a non-unit last stride would make an element
        // block read across two rows.
        if args.destination.strides.last() != 1 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected a unit stride on the last axis, found {}",
                    args.destination.strides.last()
                ),
            ));
        }
        if args.destination.strides != Strides::row_major(args.destination.shape) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the destination is written row-major; found strides {}",
                    args.destination.strides
                ),
            ));
        }

        let block = u64::from(descriptor.block);
        let mut arguments = Arguments::new();
        arguments
            .view(args.source)
            .view(args.destination)
            .view(args.indices)
            .u32(u32::try_from(count).map_err(|_| {
                Error::argument(
                    Self::NAME,
                    format!("{count} rows do not fit a kernel parameter"),
                )
            })?)
            .u32(descriptor.columns)
            .u32(descriptor.block)
            .flag(descriptor.pdl);
        // One block covers one row's element block, so the row's writes are ordered inside a block
        // and two blocks never touch the same destination element.
        kernel.launch(
            LaunchGrid::new([count, columns.div_ceil(block), 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        Args {
            source: twins.shadow_of(args.source),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        match descriptor.mode {
            // A copy: the element arrives unchanged.
            Mode::Replace => Tolerance::exact(),
            // One addition in the output's dtype: its own rounding, twice (the sum and the store).
            Mode::Add => {
                Tolerance::exact().with_atol(2.0 * descriptor.dtype.unit_roundoff() as f32)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(mode: Mode) -> Descriptor {
        Descriptor {
            dtype: DType::BF16,
            mode,
            columns: 5120,
            block: 1024,
            pdl: false,
        }
    }

    #[test]
    fn the_key_states_the_mode_and_the_row_width() {
        assert_eq!(
            descriptor(Mode::Replace).to_string(),
            "dtype_bf16_mode_replace_columns_5120_block_1024_pdl_false"
        );
        assert_eq!(
            descriptor(Mode::Add).to_string(),
            "dtype_bf16_mode_add_columns_5120_block_1024_pdl_false"
        );
    }

    #[test]
    fn a_copy_is_exact_and_an_addition_carries_two_units_of_its_dtype() {
        assert_eq!(Rows::tolerance(&descriptor(Mode::Replace)).atol, 0.0);
        let add = Rows::tolerance(&descriptor(Mode::Add));
        assert_eq!(add.atol, 2.0 * DType::BF16.unit_roundoff() as f32);
        assert_eq!(add.rtol, 0.0);
    }
}
