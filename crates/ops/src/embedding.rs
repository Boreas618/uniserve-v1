//! Masked vocabulary embedding.
//!
//! The vocabulary is sharded across tensor-parallel ranks, and the mathematical embedding of a
//! token that lies in another rank's shard is zero on this rank: every rank adds its own rows and
//! the ranks then all-reduce, which the layer performs (specs/worker.md 9, "词嵌入"). The operator
//! is therefore a gather with a mask, and two implementations of it agree bit for bit.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// One compilation of `embedding`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the weight rows and the output.
    pub dtype: DType,
    /// Element type of the token ids.
    pub ids: DType,
    /// Elements per row of the vocabulary.
    pub hidden: u32,
    /// The first vocabulary id this shard holds.
    pub first: u32,
    /// The number of vocabulary rows this shard holds.
    pub rows: u32,
    /// Tokens one program handles.
    pub block: u32,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("ids", self.ids)
            .field("hidden", self.hidden)
            .field("first", self.first)
            .field("rows", self.rows)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `embedding`.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[tokens]` token ids.
    pub ids: TensorView,
    /// `[rows, hidden]`, this shard's vocabulary rows.
    pub weight: TensorView,
    /// `[tokens, hidden]`; a token outside this shard's range is written as zeros.
    pub out: TensorView,
}

/// Vocabulary-sharded embedding lookup.
pub struct Embedding;

impl Op for Embedding {
    const NAME: &'static str = "embedding";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        if args.ids.dtype != descriptor.ids {
            return Err(Error::argument(
                Self::NAME,
                format!("expected {} ids, found {}", descriptor.ids, args.ids.dtype),
            ));
        }
        args.ids.require_rank(Self::NAME, 1)?;
        let hidden = u64::from(descriptor.hidden);
        let tokens = args.ids.shape.dim(0);
        args.weight.require_shape(
            Self::NAME,
            Shape::new(&[u64::from(descriptor.rows), hidden]),
        )?;
        args.weight
            .require_row(Self::NAME, descriptor.dtype, hidden)?;
        args.out
            .require_shape(Self::NAME, Shape::new(&[tokens, hidden]))?;
        args.out.require_row(Self::NAME, descriptor.dtype, hidden)?;

        let mut arguments = Arguments::new();
        arguments
            .view(args.ids)
            .view(args.weight)
            .view(args.out)
            .u64(tokens)
            .u32(descriptor.hidden)
            .u32(descriptor.first)
            .u32(descriptor.rows)
            .flag(descriptor.pdl);
        kernel.launch(
            LaunchGrid::covering(tokens, u64::from(descriptor.block))?,
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
        // A lookup copies values; two implementations agree bit for bit.
        Tolerance::exact()
    }
}
