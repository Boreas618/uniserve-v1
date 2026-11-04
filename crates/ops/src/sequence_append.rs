//! `sequence_append`: a register prefix and a zero tail around rows a caller already wrote.
//!
//! The decoder's sequence is assembled in a buffer the runner allocated: the caller writes its own
//! `rows` rows, then this operator puts a small learned register prefix after them and zeroes the
//! tail a bucket reserves. There is no composition that does this today — `block_copy` moves cache
//! blocks between planes with span metadata, and nothing fills a row with a constant — so it is its
//! own operator, and a small one:
//!
//! ```text
//! out[rows + r, :] = registers_weight[r, :]        for r < registers
//! out[rows + registers + t, :] = 0                 for t < tail
//! ```
//!
//! No other row is touched: the caller's own rows and anything past the tail are not this operator's,
//! so a buffer shared with another producer is safe to pass. `rows` is a launch value because a
//! request's token count varies while the register count is the model's.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// Elements one thread block writes.
pub const BLOCK: u32 = 1024;

/// One compilation of `sequence_append`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the register rows and of the buffer.
    pub dtype: DType,
    /// Elements per row.
    pub width: u32,
    /// Register rows the launch writes after the caller's rows.
    pub registers: u32,
    /// Zero rows the launch writes after the registers.
    pub tail: u32,
    /// Elements one thread block writes.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("width", self.width)
            .field("registers", self.registers)
            .field("tail", self.tail)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one append.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[rows + registers + tail, width]` the buffer, of which this launch writes the last
    /// `registers + tail` rows.
    pub out: TensorView,
    /// `[registers, width]` the learned register rows.
    pub registers_weight: TensorView,
    /// Rows the caller has already written; the registers follow them.
    pub rows: u32,
}

/// The register prefix and the zero tail of a packed sequence.
pub struct SequenceAppend;

impl Op for SequenceAppend {
    const NAME: &'static str = "sequence_append";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let width = u64::from(descriptor.width);
        let registers = u64::from(descriptor.registers);
        let tail = u64::from(descriptor.tail);
        let rows = u64::from(args.rows);
        if width == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a row of no elements has nothing to append".to_owned(),
            ));
        }
        if registers + tail == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a launch with no register rows and no tail writes nothing".to_owned(),
            ));
        }
        args.out
            .require_shape(Self::NAME, Shape::new(&[rows + registers + tail, width]))?;
        args.out.require_row(Self::NAME, descriptor.dtype, width)?;
        args.registers_weight
            .require_shape(Self::NAME, Shape::new(&[registers, width]))?;
        if args.registers_weight.dtype != descriptor.dtype {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} register rows, found {}",
                    descriptor.dtype, args.registers_weight.dtype
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.out)
            .view(args.registers_weight)
            .u32(args.rows)
            .u32(descriptor.width)
            .u32(descriptor.registers)
            .u32(descriptor.tail)
            .u32(descriptor.block)
            .flag(descriptor.pdl);
        // One thread per element of the rows this launch owns: the caller's rows are not in the grid
        // at all, so no thread can write them even by accident.
        let elements = (registers + tail) * width;
        kernel.launch(
            LaunchGrid::linear(elements.div_ceil(u64::from(descriptor.block)))?,
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
        // A copy and a constant: two implementations produce the same bits.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> Descriptor {
        Descriptor {
            dtype: DType::BF16,
            width: 5376,
            registers: 4,
            tail: 12,
            block: BLOCK,
            pdl: false,
        }
    }

    #[test]
    fn the_key_names_the_register_count_and_the_tail() {
        assert_eq!(
            descriptor().to_string(),
            "dtype_bf16_width_5376_registers_4_tail_12_block_1024_pdl_false"
        );
        // A different register count is a different program: the loop bound is compiled in.
        assert_ne!(
            descriptor().to_string(),
            Descriptor {
                registers: 8,
                ..descriptor()
            }
            .to_string()
        );
    }

    #[test]
    fn the_append_is_exact() {
        assert_eq!(SequenceAppend::tolerance(&descriptor()), Tolerance::exact());
    }
}
