//! Sampling: shaping the logits of a request, choosing among its candidates, and reading back the
//! selected values.
//!
//! The steps are separate operators because each is one kernel program and each is replaced on its
//! own during a program-layer update; the `Sampler` layer calls them in this order (specs/worker.md
//! 9, "采样"):
//!
//! 1. `logits_shape` applies the request's penalties and divides by its temperature.
//! 2. `top_k` keeps the `width` largest logits of each row, sorted, with the vocabulary id of each.
//! 3. `filter_candidates` applies the row's top-k width, its nucleus threshold and its min-p
//!    threshold, and reports whether the row's candidates form a usable distribution.
//! 4. `softmax` over the candidates (the shared operator in [`crate::norm`]).
//! 5. `inverse_cdf` draws one token per row from one uniform draw, ordering the candidates by
//!    vocabulary id so a draw is deterministic for a fixed random value.
//! 6. `argmax` picks the largest candidate for a row without a positive temperature.
//! 7. `log_softmax` and `logprob_gather` report the log-probability of the tokens the caller asks
//!    for.
//!
//! Tolerances: the shaping and the filtering only combine FP32 values step by step and are compared
//! with the bound of their own few roundings; a chosen token id is exact — two implementations that
//! disagree about a token disagree about the sample, which no tolerance can cover.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The columns of one row of sampling parameters, in the order the executor writes them.
///
/// The offsets are fixed by `uniserve/sampling/top_k.py` (`PARAMETER_COLUMNS`, with the temperature
/// first, the repetition, frequency and presence penalties at 4, 5 and 6, and the top-k width, the
/// nucleus threshold and the min-p threshold after them); the op reads only the parameters it names.
pub const PARAMETER_COLUMNS: u32 = 8;

/// The offset of the temperature among a row's parameters.
pub const TEMPERATURE: u32 = 0;
/// The offset of the top-k width (0 to keep every candidate).
pub const TOP_K: u32 = 1;
/// The offset of the nucleus threshold.
pub const TOP_P: u32 = 2;
/// The offset of the min-p threshold.
pub const MIN_P: u32 = 3;
/// The offset of the repetition penalty.
pub const REPETITION: u32 = 4;
/// The offset of the frequency penalty.
pub const FREQUENCY: u32 = 5;
/// The offset of the presence penalty.
pub const PRESENCE: u32 = 6;

/// One compilation of `logits_shape`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ShapeDescriptor {
    /// Element type of the input logits; the shaped logits are always FP32.
    pub dtype: DType,
    /// The vocabulary width of a row.
    pub vocabulary: u32,
    /// Elements one program shapes.
    pub block: u32,
    /// How many extra token columns a row carries besides its slot's counts.
    pub extras: u32,
    /// Whether the row is divided by its temperature.
    pub divide: bool,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for ShapeDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("vocabulary", self.vocabulary)
            .field("block", self.block)
            .field("extras", self.extras)
            .field("divide", self.divide)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `logits_shape`.
#[derive(Clone, Copy)]
pub struct ShapeArgs {
    /// `[rows, vocabulary]` raw logits.
    pub logits: TensorView,
    /// `[slots, vocabulary]` i32 counts of the tokens every slot has already seen.
    pub counts: TensorView,
    /// `[rows]` i32: the slot whose counts each row reads.
    pub slots: TensorView,
    /// `[rows, PARAMETER_COLUMNS]` f32 request parameters.
    pub parameters: TensorView,
    /// `[rows, extras]` i32 tokens counted besides the slot's, `-1` counting nothing.
    pub extras: Option<TensorView>,
    /// `[rows, vocabulary]` FP32 shaped logits.
    pub out: TensorView,
}

/// Penalize and temperature-scale the logits of every row.
pub struct LogitsShape;

impl Op for LogitsShape {
    const NAME: &'static str = "logits_shape";

    type Descriptor = ShapeDescriptor;
    type Args<'a> = ShapeArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let vocabulary = u64::from(descriptor.vocabulary);
        let rows = args.logits.rows();
        args.logits
            .require_row(Self::NAME, descriptor.dtype, vocabulary)?;
        args.out.require_row(Self::NAME, DType::F32, vocabulary)?;
        if args.out.rows() != rows {
            return Err(Error::argument(
                Self::NAME,
                format!("{rows} logit rows but {} output rows", args.out.rows()),
            ));
        }
        args.slots.require_shape(Self::NAME, Shape::new(&[rows]))?;
        args.parameters.require_shape(
            Self::NAME,
            Shape::new(&[rows, u64::from(PARAMETER_COLUMNS)]),
        )?;
        args.counts
            .require_row(Self::NAME, DType::I32, vocabulary)?;
        let extras = match (descriptor.extras, args.extras) {
            (0, None) => args.slots,
            (extras, Some(extra)) => {
                extra.require_shape(Self::NAME, Shape::new(&[rows, u64::from(extras)]))?;
                extra
            }
            (extras, None) => {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the descriptor declares {extras} extra tokens and the call passed none"
                    ),
                ));
            }
        };

        let mut arguments = Arguments::new();
        arguments
            .view(args.logits)
            .view(args.counts)
            .view(args.slots)
            .view(args.parameters)
            .view(extras)
            .view(args.out)
            .u64(rows)
            .u32(descriptor.vocabulary)
            .flag(descriptor.divide);
        kernel.launch(
            LaunchGrid::new([rows, vocabulary.div_ceil(u64::from(descriptor.block)), 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        ShapeArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // Four rounded FP32 operations per element at most: the penalty, its two parts and the
        // temperature division. The comparison is against the same inputs, so the bound is those
        // roundings twice over.
        Tolerance::rounding(4, DType::F32, DType::F32)
    }
}

/// One compilation of `top_k`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TopKDescriptor {
    /// Element type of the input logits.
    pub dtype: DType,
    /// The vocabulary width of a row.
    pub vocabulary: u32,
    /// How many candidates a row keeps.
    pub width: u32,
    /// Element type of the emitted vocabulary ids.
    pub ids: DType,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for TopKDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("vocabulary", self.vocabulary)
            .field("width", self.width)
            .field("ids", self.ids)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `top_k`.
#[derive(Clone, Copy)]
pub struct TopKArgs {
    /// `[rows, vocabulary]` shaped logits.
    pub logits: TensorView,
    /// `[rows, width]` the kept logits, descending; a tie keeps the lower vocabulary id first.
    pub values: TensorView,
    /// `[rows, width]` the vocabulary ids of the kept logits.
    pub ids: TensorView,
}

/// The `width` largest logits of every row, sorted.
pub struct TopK;

impl Op for TopK {
    const NAME: &'static str = "top_k";

    type Descriptor = TopKDescriptor;
    type Args<'a> = TopKArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let vocabulary = u64::from(descriptor.vocabulary);
        let width = u64::from(descriptor.width);
        if width > vocabulary {
            return Err(Error::argument(
                Self::NAME,
                format!("a width of {width} is above the vocabulary {vocabulary}"),
            ));
        }
        let rows = args.logits.rows();
        args.logits
            .require_row(Self::NAME, descriptor.dtype, vocabulary)?;
        args.values
            .require_shape(Self::NAME, Shape::new(&[rows, width]))?;
        args.ids
            .require_shape(Self::NAME, Shape::new(&[rows, width]))?;
        if args.ids.dtype != descriptor.ids {
            return Err(Error::argument(
                Self::NAME,
                format!("expected {} ids, found {}", descriptor.ids, args.ids.dtype),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.logits)
            .view(args.values)
            .view(args.ids)
            .u64(rows)
            .u32(descriptor.vocabulary)
            .u32(descriptor.width);
        kernel.launch(LaunchGrid::linear(rows)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        TopKArgs {
            values: twins.shadow_of(args.values),
            ids: twins.shadow_of(args.ids),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // The selection moves values and ids; two implementations must keep the same ones.
        Tolerance::exact()
    }
}

/// One compilation of `filter_candidates`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FilterDescriptor {
    /// How many candidates a row carries.
    pub width: u32,
    /// Element type of the emitted vocabulary ids.
    pub ids: DType,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for FilterDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("width", self.width)
            .field("ids", self.ids)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `filter_candidates`.
#[derive(Clone, Copy)]
pub struct FilterArgs {
    /// `[rows, width]` FP32 candidates, descending; a dropped candidate is written as `-inf`.
    pub values: TensorView,
    /// `[rows, width]` the vocabulary ids of the candidates.
    pub ids: TensorView,
    /// `[rows, PARAMETER_COLUMNS]` f32 request parameters.
    pub parameters: TensorView,
    /// `[rows]` i32: 1 when the row's kept candidates form a usable distribution.
    pub valid: TensorView,
}

/// Apply the top-k width, the nucleus threshold and the min-p threshold to the candidates.
pub struct FilterCandidates;

impl Op for FilterCandidates {
    const NAME: &'static str = "filter_candidates";

    type Descriptor = FilterDescriptor;
    type Args<'a> = FilterArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let width = u64::from(descriptor.width);
        let rows = args.values.rows();
        args.values.require_row(Self::NAME, DType::F32, width)?;
        args.ids
            .require_shape(Self::NAME, Shape::new(&[rows, width]))?;
        if args.ids.dtype != descriptor.ids {
            return Err(Error::argument(
                Self::NAME,
                format!("expected {} ids, found {}", descriptor.ids, args.ids.dtype),
            ));
        }
        args.parameters.require_shape(
            Self::NAME,
            Shape::new(&[rows, u64::from(PARAMETER_COLUMNS)]),
        )?;
        args.valid.require_shape(Self::NAME, Shape::new(&[rows]))?;
        if args.valid.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected i32 validity flags, found {}", args.valid.dtype),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.values)
            .view(args.ids)
            .view(args.parameters)
            .view(args.valid)
            .u64(rows)
            .u32(descriptor.width);
        kernel.launch(LaunchGrid::linear(rows)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        FilterArgs {
            values: twins.in_place(args.values, stream),
            valid: twins.shadow_of(args.valid),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // A threshold comparison over values the previous step produced; both implementations see
        // the same candidates. A decision that differs is a sampling defect, not a rounding.
        Tolerance::exact()
    }
}

/// One compilation of `inverse_cdf`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct InverseCdfDescriptor {
    /// How many candidates a row carries.
    pub width: u32,
    /// Element type of the candidate vocabulary ids.
    pub ids: DType,
    /// The element type of an emitted token.
    pub tokens: DType,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for InverseCdfDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("width", self.width)
            .field("ids", self.ids)
            .field("tokens", self.tokens)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `inverse_cdf`.
#[derive(Clone, Copy)]
pub struct InverseCdfArgs {
    /// `[rows, width]` FP32 candidate probabilities, in descending logit order.
    pub probabilities: TensorView,
    /// `[rows, width]` the vocabulary ids of the candidates.
    pub ids: TensorView,
    /// `[rows]` f32 uniform draws in `[0, 1)`.
    pub draws: TensorView,
    /// `[rows]` the chosen token of each row.
    pub tokens: TensorView,
}

/// One token per row, drawn from its candidate probabilities.
pub struct InverseCdf;

impl Op for InverseCdf {
    const NAME: &'static str = "inverse_cdf";

    type Descriptor = InverseCdfDescriptor;
    type Args<'a> = InverseCdfArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let width = u64::from(descriptor.width);
        let rows = args.probabilities.rows();
        args.probabilities
            .require_row(Self::NAME, DType::F32, width)?;
        args.ids
            .require_shape(Self::NAME, Shape::new(&[rows, width]))?;
        if args.ids.dtype != descriptor.ids {
            return Err(Error::argument(
                Self::NAME,
                format!("expected {} ids, found {}", descriptor.ids, args.ids.dtype),
            ));
        }
        args.draws.require_shape(Self::NAME, Shape::new(&[rows]))?;
        if args.draws.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected f32 draws, found {}", args.draws.dtype),
            ));
        }
        args.tokens.require_shape(Self::NAME, Shape::new(&[rows]))?;
        if args.tokens.dtype != descriptor.tokens {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} tokens, found {}",
                    descriptor.tokens, args.tokens.dtype
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.probabilities)
            .view(args.ids)
            .view(args.draws)
            .view(args.tokens)
            .u64(rows)
            .u32(descriptor.width);
        kernel.launch(LaunchGrid::linear(rows)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        InverseCdfArgs {
            tokens: twins.shadow_of(args.tokens),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // A sample is a token id: two implementations that differ have sampled from different
        // distributions, which no rounding bound excuses.
        Tolerance::exact()
    }
}

/// One compilation of `argmax`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ArgMaxDescriptor {
    /// Element type of the input logits.
    pub dtype: DType,
    /// The width of a row.
    pub width: u32,
    /// Element type of the emitted ids.
    pub ids: DType,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for ArgMaxDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("width", self.width)
            .field("ids", self.ids)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `argmax`.
#[derive(Clone, Copy)]
pub struct ArgMaxArgs {
    /// `[rows, width]` logits or candidates.
    pub values: TensorView,
    /// `[rows]` the index of each row's largest value, the lowest index among ties.
    pub out: TensorView,
}

/// The largest value's index, per row.
pub struct ArgMax;

impl Op for ArgMax {
    const NAME: &'static str = "argmax";

    type Descriptor = ArgMaxDescriptor;
    type Args<'a> = ArgMaxArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let width = u64::from(descriptor.width);
        let rows = args.values.rows();
        args.values
            .require_row(Self::NAME, descriptor.dtype, width)?;
        args.out.require_shape(Self::NAME, Shape::new(&[rows]))?;
        if args.out.dtype != descriptor.ids {
            return Err(Error::argument(
                Self::NAME,
                format!("expected {} ids, found {}", descriptor.ids, args.out.dtype),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.values)
            .view(args.out)
            .u64(rows)
            .u32(descriptor.width);
        kernel.launch(LaunchGrid::linear(rows)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        ArgMaxArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        Tolerance::exact()
    }
}

/// One compilation of `log_softmax`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LogSoftmaxDescriptor {
    /// Element type of the input logits; the output is FP32.
    pub dtype: DType,
    /// Elements per row.
    pub columns: u32,
    /// The tile one program loads at a time.
    pub tile: u32,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for LogSoftmaxDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("columns", self.columns)
            .field("tile", self.tile)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `log_softmax`.
#[derive(Clone, Copy)]
pub struct LogSoftmaxArgs {
    /// `[rows, columns]`.
    pub x: TensorView,
    /// `[rows, columns]` FP32 log-probabilities.
    pub out: TensorView,
}

/// `log_softmax` over the last axis.
pub struct LogSoftmax;

impl Op for LogSoftmax {
    const NAME: &'static str = "log_softmax";

    type Descriptor = LogSoftmaxDescriptor;
    type Args<'a> = LogSoftmaxArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let columns = u64::from(descriptor.columns);
        args.x.require_row(Self::NAME, descriptor.dtype, columns)?;
        args.out.require_row(Self::NAME, DType::F32, columns)?;
        if args.out.rows() != args.x.rows() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} input rows but {} output rows",
                    args.x.rows(),
                    args.out.rows()
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.x)
            .view(args.out)
            .u32(descriptor.columns)
            .flag(descriptor.pdl);
        kernel.launch(LaunchGrid::linear(args.x.rows())?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        LogSoftmaxArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The denominator is a sum of `columns` positive terms in FP32; each output subtracts the
        // selected logit from its logarithm.
        Tolerance::rounding(u64::from(descriptor.columns) + 1, DType::F32, DType::F32)
    }
}

/// One compilation of `logprob_gather`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LogprobDescriptor {
    /// The width of a log-probability row.
    pub columns: u32,
    /// Element type of the token ids.
    pub ids: DType,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for LogprobDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("columns", self.columns)
            .field("ids", self.ids)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `logprob_gather`.
#[derive(Clone, Copy)]
pub struct LogprobArgs {
    /// `[rows, columns]` FP32 log-probabilities.
    pub logprobabilities: TensorView,
    /// `[rows]` the tokens whose log-probability is reported.
    pub tokens: TensorView,
    /// `[rows]` FP32.
    pub out: TensorView,
}

/// The log-probability of one chosen token per row.
pub struct LogprobGather;

impl Op for LogprobGather {
    const NAME: &'static str = "logprob_gather";

    type Descriptor = LogprobDescriptor;
    type Args<'a> = LogprobArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let columns = u64::from(descriptor.columns);
        let rows = args.logprobabilities.rows();
        args.logprobabilities
            .require_row(Self::NAME, DType::F32, columns)?;
        args.tokens.require_shape(Self::NAME, Shape::new(&[rows]))?;
        if args.tokens.dtype != descriptor.ids {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} tokens, found {}",
                    descriptor.ids, args.tokens.dtype
                ),
            ));
        }
        args.out.require_shape(Self::NAME, Shape::new(&[rows]))?;
        if args.out.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected f32 log-probabilities, found {}", args.out.dtype),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.logprobabilities)
            .view(args.tokens)
            .view(args.out)
            .u64(rows)
            .u32(descriptor.columns);
        kernel.launch(LaunchGrid::linear(rows)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        LogprobArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        Tolerance::exact()
    }
}
