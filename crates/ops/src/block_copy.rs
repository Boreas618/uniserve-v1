//! Moving whole cache blocks: the snapshot and the scatter of a block copy.
//!
//! A state bank stores each field block-major: axis 0 selects a block and the trailing axes are the
//! block's own rows, of any strides. A copy names its blocks with a device `int64` vector, so no
//! host read stands between two launches and a captured graph replays it:
//!
//! ```text
//! cache_block_snapshot   snapshot[row, :] = field[blocks[row], :]
//! cache_block_scatter    field[blocks[row], :] = snapshot[row, :]
//! ```
//!
//! The two launches run in that order: every source block is read into the snapshot before any
//! target is written, so the vectors may name overlapping blocks. A field that stores an encoded
//! format one byte wide is copied as bytes, which is why the element type here is the storage's.
//! A negative index reads block zero (the
//! inactive pair's placeholder) and writes nothing.

use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::operand::Operand;
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The elements one program writes or reads at a time.
pub const COPY_BLOCK: u32 = 1024;

/// What one compilation of a block copy is: the storage's shape, its strides and the tile.
///
/// The shape and the strides are compile-time values of the kernel package, because the reference
/// kernel unrolls the trailing axes' address arithmetic and normalizes the element count once.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct BlockShape {
    /// Element type of the raw storage.
    pub dtype: DType,
    /// The field's block-major extents: axis 0 is the block, the rest is one block's extent.
    pub shape: Vec<u64>,
    /// The field's element strides, one per axis of `shape`.
    pub strides: Vec<u64>,
    /// Elements between entries of the index vector.
    pub index_stride: u64,
    /// Elements in one flattened block.
    pub width: u64,
    /// The columns one program covers.
    pub block: u32,
}

impl fmt::Display for BlockShape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("shape", axes(&self.shape))
            .field("strides", axes(&self.strides))
            .field("index_stride", self.index_stride)
            .field("width", self.width)
            .field("block", self.block)
            .finish(formatter)
    }
}

/// Extents or strides as a key spells them: each value in decimal, joined by `x` (`4x2x8`), and
/// nothing for a field of no axis.
fn axes(values: &[u64]) -> String {
    values
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

impl BlockShape {
    /// Check the launch against the compiled variant and the index vector.
    fn check(&self, operator: &'static str, field: &Operand, blocks: &Operand) -> Result<u64> {
        field.require_shape(operator, &self.shape)?;
        let strides = field.strides().iter().map(|&stride| stride as u64);
        if !strides.eq(self.strides.iter().copied()) {
            return Err(Error::argument(
                operator,
                format!(
                    "the field's strides are {:?}, the descriptor says {:?}",
                    field.strides(),
                    self.strides
                ),
            ));
        }
        if field.dtype != self.dtype {
            return Err(Error::argument(
                operator,
                format!(
                    "the descriptor copies {} elements, the field holds {}",
                    abi::name(self.dtype),
                    abi::name(field.dtype)
                ),
            ));
        }
        // A field of no axis holds one element and counts as one block.
        let blocks_held = self.shape.first().map_or(1, |&blocks| blocks.max(1));
        let per_block = self.shape.iter().product::<u64>() / blocks_held;
        if self.width == 0 || self.width != per_block {
            return Err(Error::argument(
                operator,
                format!(
                    "the descriptor walks {} elements per block, the field holds {per_block}",
                    self.width
                ),
            ));
        }
        blocks.require_rank(operator, 1)?;
        if blocks.dtype != DType::I64 {
            return Err(Error::argument(
                operator,
                format!("the block indices are {}, not i64", abi::name(blocks.dtype)),
            ));
        }
        if blocks.stride(0) != self.index_stride {
            return Err(Error::argument(
                operator,
                format!(
                    "the index vector's entries are {} elements apart, the descriptor says {}",
                    blocks.stride(0),
                    self.index_stride
                ),
            ));
        }
        Ok(blocks.dim(0))
    }
}

/// One compilation of `cache_block_snapshot`.
pub type SnapshotDescriptor = BlockShape;

/// The arguments of `cache_block_snapshot`.
#[derive(Clone)]
pub struct SnapshotArgs {
    /// The field's block-major storage, at any strides.
    pub field: Operand,
    /// `[count]` i64 device indices: the block each row of the snapshot takes.
    pub blocks: Operand,
    /// `[count, width]`, contiguous: receives the selected blocks.
    pub snapshot: Operand,
}

/// Read selected blocks of a state bank into a contiguous snapshot.
pub struct CacheBlockSnapshot;

impl Op for CacheBlockSnapshot {
    const NAME: &'static str = proto::module::operator::CACHE_BLOCK_SNAPSHOT;

    type Descriptor = SnapshotDescriptor;
    type Args<'a> = SnapshotArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let count = descriptor.check(Self::NAME, &args.field, &args.blocks)?;
        args.snapshot.require_rank(Self::NAME, 2)?;
        args.snapshot
            .require_row(Self::NAME, descriptor.dtype, descriptor.width)?;
        if args.snapshot.dim(0) != count {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the snapshot holds {} rows, the index vector selects {count}",
                    args.snapshot.dim(0)
                ),
            ));
        }
        if !args.snapshot.is_row_major() {
            return Err(Error::argument(
                Self::NAME,
                "the snapshot is not contiguous".to_owned(),
            ));
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        // One block per selected index: `check` holds the index vector to one axis.
        let count = args.blocks.dim(0);

        let mut arguments = Arguments::new();
        arguments
            .view(&args.field)
            .view(&args.blocks)
            .view(&args.snapshot);
        kernel.launch(
            LaunchGrid::new([
                count,
                descriptor.width.div_ceil(u64::from(descriptor.block)),
                1,
            ])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        SnapshotArgs {
            snapshot: twins.shadow_of(args.snapshot),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // Whole blocks of raw storage move unchanged.
        Tolerance::exact()
    }
}

/// One compilation of `cache_block_scatter`.
pub type ScatterDescriptor = BlockShape;

/// The arguments of `cache_block_scatter`.
#[derive(Clone)]
pub struct ScatterArgs {
    /// The field's block-major storage, at any strides; the selected blocks are overwritten.
    pub field: Operand,
    /// `[count]` i64 device indices: the block each row of the snapshot goes to.
    pub blocks: Operand,
    /// `[count, width]`, contiguous: the blocks to write.
    pub snapshot: Operand,
}

/// Write a contiguous snapshot into selected blocks of a state bank.
pub struct CacheBlockScatter;

impl Op for CacheBlockScatter {
    const NAME: &'static str = proto::module::operator::CACHE_BLOCK_SCATTER;

    type Descriptor = ScatterDescriptor;
    type Args<'a> = ScatterArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let count = descriptor.check(Self::NAME, &args.field, &args.blocks)?;
        args.snapshot.require_rank(Self::NAME, 2)?;
        args.snapshot
            .require_row(Self::NAME, descriptor.dtype, descriptor.width)?;
        if args.snapshot.dim(0) != count {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the snapshot holds {} rows, the index vector names {count}",
                    args.snapshot.dim(0)
                ),
            ));
        }
        if !args.snapshot.is_row_major() {
            return Err(Error::argument(
                Self::NAME,
                "the snapshot is not contiguous".to_owned(),
            ));
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        // One block per selected index: `check` holds the index vector to one axis.
        let count = args.blocks.dim(0);

        let mut arguments = Arguments::new();
        arguments
            .view(&args.field)
            .view(&args.blocks)
            .view(&args.snapshot);
        kernel.launch(
            LaunchGrid::new([
                count,
                descriptor.width.div_ceil(u64::from(descriptor.block)),
                1,
            ])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        // The field is written in place, so the shadow runs on a copy of it and the snapshot is read
        // from the caller's.
        ScatterArgs {
            field: twins.in_place(args.field),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // Whole blocks of raw storage move unchanged.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block's extents and strides are part of the key, spelled as the stored modules name them.
    #[test]
    fn a_block_shape_names_its_extents_and_strides_in_its_key() {
        let shape = BlockShape {
            dtype: DType::BF16,
            shape: vec![4, 2, 8],
            strides: vec![16, 8, 1],
            index_stride: 1,
            width: 16,
            block: COPY_BLOCK,
        };
        assert_eq!(
            shape.to_string(),
            "dtype_bf16_shape_4x2x8_strides_16x8x1_index_stride_1_width_16_block_1024"
        );
    }
}
