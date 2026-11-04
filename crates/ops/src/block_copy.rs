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
//! format one byte wide is copied as bytes, which is why the element type here is the storage's
//! (specs/worker.md 8, the block copy of the prefix cache). A negative index reads block zero (the
//! inactive pair's placeholder) and writes nothing.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, Strides, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The elements one program writes or reads at a time.
pub const COPY_BLOCK: u32 = 1024;

/// What one compilation of a block copy is: the storage's shape, its strides and the tile.
///
/// The shape and the strides are compile-time values of the kernel program, because the reference
/// kernel unrolls the trailing axes' address arithmetic and normalizes the element count once.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockShape {
    /// Element type of the raw storage.
    pub dtype: DType,
    /// The field's block-major shape: axis 0 is the block, the rest is one block's extent.
    pub shape: Shape,
    /// The field's element strides.
    pub strides: Strides,
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
            .field("dtype", self.dtype)
            .field("shape", self.shape)
            .field("strides", self.strides)
            .field("index_stride", self.index_stride)
            .field("width", self.width)
            .field("block", self.block)
            .finish(formatter)
    }
}

impl BlockShape {
    /// The shape of a field with `blocks` blocks of `trailing` extent and its strides.
    ///
    /// `width` and `block` are the flattened block length and the copy tile; the index vector's
    /// stride is 1 for a contiguous device vector.
    pub const fn new(
        dtype: DType,
        shape: Shape,
        strides: Strides,
        index_stride: u64,
        width: u64,
        block: u32,
    ) -> Self {
        Self {
            dtype,
            shape,
            strides,
            index_stride,
            width,
            block,
        }
    }

    /// The tile a copy of `field` uses: the flattened block, in [`COPY_BLOCK`] element pieces.
    pub fn of(field: &TensorView, index_stride: u64) -> Result<Self> {
        let shape = field.shape;
        let width = shape.numel() / shape.dim(0).max(1);
        Ok(Self {
            dtype: field.dtype,
            shape,
            strides: field.strides,
            index_stride,
            width,
            block: COPY_BLOCK,
        })
    }

    /// Check the launch against the compiled variant and the index vector.
    fn check(
        &self,
        operator: &'static str,
        field: &TensorView,
        blocks: &TensorView,
    ) -> Result<u64> {
        field.require_shape(operator, self.shape)?;
        if field.strides != self.strides {
            return Err(Error::argument(
                operator,
                format!(
                    "the field's strides are {}, the descriptor says {}",
                    field.strides, self.strides
                ),
            ));
        }
        if field.dtype != self.dtype {
            return Err(Error::argument(
                operator,
                format!(
                    "the descriptor copies {} elements, the field holds {}",
                    self.dtype, field.dtype
                ),
            ));
        }
        if self.width == 0 || self.width != self.shape.numel() / self.shape.dim(0).max(1) {
            return Err(Error::argument(
                operator,
                format!(
                    "the descriptor walks {} elements per block, the field holds {}",
                    self.width,
                    self.shape.numel() / self.shape.dim(0).max(1)
                ),
            ));
        }
        blocks.require_rank(operator, 1)?;
        if blocks.dtype != DType::I64 {
            return Err(Error::argument(
                operator,
                format!("the block indices are {}, not i64", blocks.dtype),
            ));
        }
        if blocks.strides.get(0) != self.index_stride {
            return Err(Error::argument(
                operator,
                format!(
                    "the index vector's entries are {} elements apart, the descriptor says {}",
                    blocks.strides.get(0),
                    self.index_stride
                ),
            ));
        }
        Ok(blocks.shape.dim(0))
    }
}

/// One compilation of `cache_block_snapshot`.
pub type SnapshotDescriptor = BlockShape;

/// The arguments of `cache_block_snapshot`.
#[derive(Clone, Copy)]
pub struct SnapshotArgs {
    /// The field's block-major storage, at any strides.
    pub field: TensorView,
    /// `[count]` i64 device indices: the block each row of the snapshot takes.
    pub blocks: TensorView,
    /// `[count, width]`, contiguous: receives the selected blocks.
    pub snapshot: TensorView,
}

/// Read selected blocks of a state bank into a contiguous snapshot.
pub struct CacheBlockSnapshot;

impl Op for CacheBlockSnapshot {
    const NAME: &'static str = "cache_block_snapshot";

    type Descriptor = SnapshotDescriptor;
    type Args<'a> = SnapshotArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let count = descriptor.check(Self::NAME, &args.field, &args.blocks)?;
        args.snapshot.require_rank(Self::NAME, 2)?;
        args.snapshot
            .require_row(Self::NAME, descriptor.dtype, descriptor.width)?;
        if args.snapshot.shape.dim(0) != count {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the snapshot holds {} rows, the index vector selects {count}",
                    args.snapshot.shape.dim(0)
                ),
            ));
        }
        if args.snapshot.strides != Strides::row_major(args.snapshot.shape) {
            return Err(Error::argument(
                Self::NAME,
                "the snapshot is not contiguous".to_owned(),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.field)
            .view(args.blocks)
            .view(args.snapshot);
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

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
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
#[derive(Clone, Copy)]
pub struct ScatterArgs {
    /// The field's block-major storage, at any strides; the selected blocks are overwritten.
    pub field: TensorView,
    /// `[count]` i64 device indices: the block each row of the snapshot goes to.
    pub blocks: TensorView,
    /// `[count, width]`, contiguous: the blocks to write.
    pub snapshot: TensorView,
}

/// Write a contiguous snapshot into selected blocks of a state bank.
pub struct CacheBlockScatter;

impl Op for CacheBlockScatter {
    const NAME: &'static str = "cache_block_scatter";

    type Descriptor = ScatterDescriptor;
    type Args<'a> = ScatterArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let count = descriptor.check(Self::NAME, &args.field, &args.blocks)?;
        args.snapshot.require_rank(Self::NAME, 2)?;
        args.snapshot
            .require_row(Self::NAME, descriptor.dtype, descriptor.width)?;
        if args.snapshot.shape.dim(0) != count {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the snapshot holds {} rows, the index vector names {count}",
                    args.snapshot.shape.dim(0)
                ),
            ));
        }
        if args.snapshot.strides != Strides::row_major(args.snapshot.shape) {
            return Err(Error::argument(
                Self::NAME,
                "the snapshot is not contiguous".to_owned(),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.field)
            .view(args.blocks)
            .view(args.snapshot);
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

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        // The field is written in place, so the shadow runs on a copy of it and the snapshot is read
        // from the caller's.
        ScatterArgs {
            field: twins.in_place(args.field, stream),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // Whole blocks of raw storage move unchanged.
        Tolerance::exact()
    }
}
