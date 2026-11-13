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
//! place:   destination[indices[j]] = source[j]          for indices[j] ≥ 0; a negative index
//!                                                       names no row and writes nothing
//! ```
//!
//! `place` is the write of per-request state a batch's tokens leave (a ring of recent rows, the
//! key of a group a token completes): a token that leaves none, or a padding token of a bucket
//! wider than its batch, states a negative index, which is `u32::MAX`, the slot that addresses
//! nothing, read as i32.
//!
//! `indices` is `[count]` i32 and may repeat: a repeated index is written, or added, more than once,
//! in the order it appears, so the caller controls the sum order and repeated calls of one launch
//! agree bit for bit. The rows themselves are `[rows, columns]`, the source is `[count, columns]` and
//! the index view is `[count]`, which is the one relation [`Rows::launch`] can hold.
//!
//! # What an index outside `[0, rows)` does
//!
//! **It writes there.** The kernel takes the index as an address, `destination[indices[j]]`, and masks
//! only the columns: an index of 16 127 against a 1 797-row destination stores into whatever the arena
//! put next, because a device value cannot be checked on the host and this kernel states no bound.
//!
//! The guard is therefore **the object that builds the table**: its provenance must be the same canvas
//! as the destination it is launched against, and a caller that cannot say which canvas a table came
//! from must not launch it. A kernel-side assert was considered and rejected: it would put one
//! instruction per scatter on every launch to detect a table a refusal upstream already makes
//! unrepresentable.
//!
//! The program is a kernel package: one thread per (row, element) of the source, the row taken from
//! the second axis and the element block from the third, so a row's elements are never split across
//! two blocks that could race on the same destination row. Parameters, in order: `source`,
//! `destination`, `indices`, `count` (u32), `columns` (u32), `block` (u32), the PDL flag.
//!
//! The tolerances differ because the operations do: a replace moves elements and is exact, an add
//! performs one addition in the output's dtype and is bounded by two units of its own rounding.

use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::operand::Operand;
use crate::tolerance::{self, Tolerance};
use crate::{Error, Op, Result, Twins};

/// What a launch does with the rows it picks out.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Mode {
    /// The destination row becomes the source row.
    Replace,
    /// The destination row gains the source row.
    Add,
    /// The destination row becomes the source row, and a negative index writes nothing.
    Place,
}

impl fmt::Display for Mode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Replace => "replace",
            Self::Add => "add",
            Self::Place => "place",
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
            .field("dtype", abi::name(self.dtype))
            .field("mode", self.mode)
            .field("columns", self.columns)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `rows`.
#[derive(Clone)]
pub struct Args {
    /// `[count, columns]` the rows to place.
    pub source: Operand,
    /// `[rows, columns]` the sequence the rows land in.
    pub destination: Operand,
    /// `[count]` i32: the destination row of each source row.
    pub indices: Operand,
}

/// Row writes and accumulation.
pub struct Rows;

impl Op for Rows {
    const NAME: &'static str = proto::module::operator::ROWS;

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let columns = u64::from(descriptor.columns);
        let count = args.indices.numel();

        args.indices.require_shape(Self::NAME, &[count])?;
        if args.indices.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected i32 row indices, found {}",
                    abi::name(args.indices.dtype)
                ),
            ));
        }
        args.source.require_shape(Self::NAME, &[count, columns])?;
        args.source
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        args.destination
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        // The destination is walked as a grid of rows: a row that is not `columns` elements long
        // would make the row stride meaningless, and a non-unit last stride would make an element
        // block read across two rows.
        if args.destination.last_stride() != 1 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected a unit stride on the last axis, found {}",
                    args.destination.last_stride()
                ),
            ));
        }
        if !args.destination.is_row_major() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the destination is written row-major; found strides {:?}",
                    args.destination.strides()
                ),
            ));
        }
        u32::try_from(count).map_err(|_| {
            Error::argument(
                Self::NAME,
                format!("{count} rows do not fit a kernel parameter"),
            )
        })?;
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let columns = u64::from(descriptor.columns);
        let count = args.indices.numel();

        let block = u64::from(descriptor.block);
        let mut arguments = Arguments::new();
        arguments
            .view(&args.source)
            .view(&args.destination)
            .view(&args.indices)
            // `check` keeps the row count within a kernel parameter.
            .u32(count as u32)
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

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        // The destination is what this operator writes, so the twin has to *be* the destination the
        // comparison replaces: the new program writes the copy while the previous generation writes
        // the real buffer, and a twin of the source instead would leave both programs writing the
        // same buffer and compare a source with itself, which no run can fail.
        Args {
            destination: twins.in_place(args.destination),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        match descriptor.mode {
            // A copy: the element arrives unchanged.
            Mode::Replace | Mode::Place => Tolerance::exact(),
            // One addition in the output's dtype: its own rounding, twice (the sum and the store).
            Mode::Add => Tolerance::exact()
                .with_atol(2.0 * tolerance::unit_roundoff(descriptor.dtype) as f32),
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
        assert_eq!(
            descriptor(Mode::Place).to_string(),
            "dtype_bf16_mode_place_columns_5120_block_1024_pdl_false"
        );
    }

    #[test]
    fn a_copy_is_exact_and_an_addition_carries_two_units_of_its_dtype() {
        assert_eq!(Rows::tolerance(&descriptor(Mode::Replace)).atol, 0.0);
        let add = Rows::tolerance(&descriptor(Mode::Add));
        assert_eq!(add.atol, 2.0 * tolerance::unit_roundoff(DType::BF16) as f32);
        assert_eq!(add.rtol, 0.0);
    }
}
