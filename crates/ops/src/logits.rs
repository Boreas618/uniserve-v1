//! The rows of a vocabulary matrix the logits head consumes, and the un-padding of a gathered
//! sharded vocabulary.
//!
//! `lm_head` computes the last position of every request only: `logits_gather` packs those rows of
//! the final hidden state into one matrix, and after the tensor-parallel head has gathered the
//! sharded logits, `vocab_unpad` drops the padding the sharded vocabulary carries (specs/worker.md
//! 9, "`lm_head` 与 logits"). Both operators only move values, so two implementations of either
//! agree bit for bit.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// One compilation of `logits_gather`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GatherDescriptor {
    /// Element type of the hidden state.
    pub dtype: DType,
    /// Elements per row.
    pub columns: u32,
    /// Rows one program copies.
    pub block: u32,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for GatherDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("columns", self.columns)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `logits_gather`.
pub struct GatherArgs {
    /// `[tokens, columns]` hidden state of the whole batch.
    pub hidden: TensorView,
    /// `[rows]` i32 indices of the rows to keep, in request order.
    pub rows: TensorView,
    /// `[count, columns]` the packed rows.
    pub out: TensorView,
}

/// Gather the last position of every request into one matrix.
pub struct LogitsGather;

impl Op for LogitsGather {
    const NAME: &'static str = "logits_gather";

    type Descriptor = GatherDescriptor;
    type Args<'a> = GatherArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let columns = u64::from(descriptor.columns);
        args.hidden
            .require_row(Self::NAME, descriptor.dtype, columns)?;
        args.rows.require_rank(Self::NAME, 1)?;
        if args.rows.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected i32 row indices, found {}", args.rows.dtype),
            ));
        }
        let count = args.rows.shape.dim(0);
        args.out
            .require_shape(Self::NAME, Shape::new(&[count, columns]))?;
        args.out
            .require_row(Self::NAME, descriptor.dtype, columns)?;

        let mut arguments = Arguments::new();
        arguments
            .view(args.hidden)
            .view(args.rows)
            .view(args.out)
            .u64(count)
            .u32(descriptor.columns);
        kernel.launch(
            LaunchGrid::covering(count * columns, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        GatherArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        Tolerance::exact()
    }
}

/// One compilation of `vocab_unpad`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct UnpadDescriptor {
    /// Element type of the gathered logits.
    pub dtype: DType,
    /// The vocabulary size the model uses.
    pub vocabulary: u32,
    /// The width the gathered matrix carries, at least `vocabulary`.
    pub padded: u32,
    /// Rows one program copies.
    pub block: u32,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for UnpadDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("vocabulary", self.vocabulary)
            .field("padded", self.padded)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `vocab_unpad`.
pub struct UnpadArgs {
    /// `[rows, padded]` the gathered logits.
    pub gathered: TensorView,
    /// `[rows, vocabulary]`.
    pub out: TensorView,
}

/// Drop the padding of a gathered sharded vocabulary.
pub struct VocabUnpad;

impl Op for VocabUnpad {
    const NAME: &'static str = "vocab_unpad";

    type Descriptor = UnpadDescriptor;
    type Args<'a> = UnpadArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let vocabulary = u64::from(descriptor.vocabulary);
        let padded = u64::from(descriptor.padded);
        if padded < vocabulary {
            return Err(Error::argument(
                Self::NAME,
                format!("the gathered width {padded} is below the vocabulary {vocabulary}"),
            ));
        }
        let rows = args.gathered.rows();
        args.gathered
            .require_shape(Self::NAME, Shape::new(&[rows, padded]))?;
        args.out
            .require_shape(Self::NAME, Shape::new(&[rows, vocabulary]))?;
        if args.gathered.dtype != descriptor.dtype || args.out.dtype != descriptor.dtype {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} logits on both sides, found {} and {}",
                    descriptor.dtype, args.gathered.dtype, args.out.dtype
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.gathered)
            .view(args.out)
            .u64(rows)
            .u32(descriptor.padded)
            .u32(descriptor.vocabulary);
        kernel.launch(
            LaunchGrid::covering(rows * vocabulary, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        UnpadArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        Tolerance::exact()
    }
}
