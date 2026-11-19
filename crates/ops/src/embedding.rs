//! Masked vocabulary embedding, and the lookup of an MXFP8 table's rows.
//!
//! The vocabulary is sharded across tensor-parallel ranks, and the mathematical embedding of a
//! token that lies in another rank's shard is zero on this rank: every rank adds its own rows and
//! the ranks then all-reduce, which the layer performs. The operator
//! is therefore a gather with a mask, and two implementations of it agree bit for bit.
//!
//! A table stored in MXFP8 (E4M3 values with one UE8M0 power-of-two scale per 32, the format of
//! a hashed n-gram memory's rows) is read the same way by [`EmbeddingMxfp8`], which dequantizes
//! each row it reads; a shard's rows are named by their ids in the whole table.

use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::launch::PDL;
use crate::operand::Operand;
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

/// Tokens one program of the embedding gathers.
const TOKENS: u32 = 8;

impl Descriptor {
    /// The compilation that gathers rows of `hidden` `dtype` elements by `ids` ids from a shard
    /// of `rows` rows that opens at vocabulary id `first`.
    pub fn new(dtype: DType, ids: DType, hidden: u32, first: u32, rows: u32) -> Self {
        Self {
            dtype,
            ids,
            hidden,
            first,
            rows,
            block: TOKENS,
            pdl: PDL,
        }
    }
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("ids", abi::name(self.ids))
            .field("hidden", self.hidden)
            .field("first", self.first)
            .field("rows", self.rows)
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `embedding`.
#[derive(Clone)]
pub struct Args {
    /// `[tokens]` token ids.
    pub ids: Operand,
    /// `[rows, hidden]`, this shard's vocabulary rows.
    pub weight: Operand,
    /// `[tokens, hidden]`; a token outside this shard's range is written as zeros.
    pub out: Operand,
}

/// Vocabulary-sharded embedding lookup.
pub struct Embedding;

impl Op for Embedding {
    const NAME: &'static str = proto::module::operator::EMBEDDING;

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        if args.ids.dtype != descriptor.ids {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} ids, found {}",
                    abi::name(descriptor.ids),
                    abi::name(args.ids.dtype)
                ),
            ));
        }
        args.ids.require_rank(Self::NAME, 1)?;
        let hidden = u64::from(descriptor.hidden);
        let tokens = args.ids.dim(0);
        args.weight
            .require_shape(Self::NAME, &[u64::from(descriptor.rows), hidden])?;
        args.weight
            .require_row(Self::NAME, descriptor.dtype, hidden)?;
        args.out.require_shape(Self::NAME, &[tokens, hidden])?;
        args.out.require_row(Self::NAME, descriptor.dtype, hidden)?;
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let tokens = args.ids.dim(0);

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("ids_ptr", &args.ids)
            .view("weight_ptr", &args.weight)
            .view("out_ptr", &args.out)
            .u64("tokens", tokens)
            .u32("hidden", descriptor.hidden)
            .u32("first", descriptor.first)
            .u32("rows", descriptor.rows);
        // No dependent-launch word: the module takes that flag as a constexpr and the driver takes it
        // as a launch attribute.
        kernel.launch(
            LaunchGrid::covering(tokens, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
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

/// Ids one program of `embedding_mxfp8` dequantizes.
const MXFP8_IDS: u32 = 16;

/// Elements one UE8M0 scale of an MXFP8 row covers.
pub const MXFP8_GROUP: u32 = 32;

/// One compilation of `embedding_mxfp8`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Mxfp8Descriptor {
    /// Element type of the dequantized rows.
    pub dtype: DType,
    /// Element type of the ids.
    pub ids: DType,
    /// Elements per row of the table.
    pub hidden: u32,
    /// Elements one scale covers.
    pub group: u32,
    /// Ids one program handles.
    pub block: u32,
}

impl Mxfp8Descriptor {
    /// The compilation that dequantizes MXFP8 rows of `hidden` elements, named by `ids` ids,
    /// into `dtype` rows.
    pub fn new(dtype: DType, ids: DType, hidden: u32) -> Self {
        Self {
            dtype,
            ids,
            hidden,
            group: MXFP8_GROUP,
            block: MXFP8_IDS,
        }
    }
}

impl fmt::Display for Mxfp8Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("ids", abi::name(self.ids))
            .field("hidden", self.hidden)
            .field("group", self.group)
            .field("block", self.block)
            .finish(formatter)
    }
}

/// The arguments of `embedding_mxfp8`.
#[derive(Clone)]
pub struct Mxfp8Args {
    /// `[count]`, contiguous: the table rows to read, as ids of the whole table.
    pub ids: Operand,
    /// `[rows, hidden]` E4M3, contiguous: this shard's rows of the table, whose first is row
    /// `first` of the whole table.
    pub values: Operand,
    /// `[rows, hidden / group]` U8, contiguous: each group's UE8M0 scale, an exponent byte `e`
    /// meaning `2^(e − 127)`.
    pub scales: Operand,
    /// `[count, hidden]`, contiguous: receives the dequantized rows; an id outside the shard is
    /// written as zeros.
    pub out: Operand,
    /// The id of the shard's first row.
    pub first: u64,
}

/// A lookup of an MXFP8 table's rows (E4M3 values, a UE8M0 scale per group of 32): each value
/// times its group's power of two in FP32, rounded once, and zeros for a row another shard holds.
///
/// The shard's first row and its extent are launch values, so every shard of a table is one
/// compilation.
pub struct EmbeddingMxfp8;

impl Op for EmbeddingMxfp8 {
    const NAME: &'static str = proto::module::operator::EMBEDDING_MXFP8;

    type Descriptor = Mxfp8Descriptor;
    type Args<'a> = Mxfp8Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let Mxfp8Descriptor {
            dtype,
            ids,
            hidden,
            group,
            ..
        } = *descriptor;
        if !matches!(ids, DType::U32 | DType::I32 | DType::I64) {
            return Err(Error::argument(
                Self::NAME,
                format!("ids of {}", abi::name(ids)),
            ));
        }
        if group == 0 || hidden == 0 || !hidden.is_multiple_of(group) {
            return Err(Error::argument(
                Self::NAME,
                format!("rows of {hidden} elements are not whole groups of {group}"),
            ));
        }
        args.ids.require_rank(Self::NAME, 1)?;
        let count = args.ids.dim(0);
        args.ids
            .require_row(Self::NAME, ids, count)
            .map_err(|error| Error::argument(Self::NAME, format!("ids: {error}")))?;

        let (width, scales) = (u64::from(hidden), u64::from(hidden / group));
        args.values.require_rank(Self::NAME, 2)?;
        let rows = args.values.dim(0);
        for (name, view, dims, element) in [
            ("values", &args.values, [rows, width], DType::F8E4M3),
            ("scales", &args.scales, [rows, scales], DType::U8),
            ("rows", &args.out, [count, width], dtype),
        ] {
            let named = |error: Error| Error::argument(Self::NAME, format!("{name}: {error}"));
            view.require_shape(Self::NAME, &dims).map_err(named)?;
            view.require_row(Self::NAME, element, dims[1])
                .map_err(named)?;
            if !view.is_row_major() {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} are not contiguous"),
                ));
            }
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let count = args.ids.dim(0);

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("ids_ptr", &args.ids)
            .view("values_ptr", &args.values)
            .view("scales_ptr", &args.scales)
            .view("out_ptr", &args.out)
            .u64("count", count)
            .u64("first", args.first)
            .u64("rows", args.values.dim(0));
        kernel.launch(
            LaunchGrid::covering(count, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        Mxfp8Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // An E4M3 value times a power of two is one rounding of an exact product: two
        // implementations agree bit for bit.
        Tolerance::exact()
    }
}
