//! Sampling: shaping the logits of a request, choosing among its candidates, and reading back the
//! selected values.
//!
//! The steps are separate operators because each is one kernel program and each is replaced on its
//! own during a program-layer update; the `Sampler` layer calls them in this order (specs/worker.md
//! 9, "采样"):
//!
//! 1. `logits_shape` applies the request's penalties and divides by its temperature. **A row whose
//!    temperature is zero divides by one**, not by zero: `uniserve/sampling/select.py:443` states
//!    the rule ("a zero temperature divides by one; callers select those rows by argmax"), and without
//!    it every greedy request would be shaped into `inf` and `nan` logits. The shaped row is then the
//!    penalized logits themselves, so the row's largest candidate is its greedy token.
//! 2. `top_k` keeps the `width` largest logits of each row, sorted, with the vocabulary id of each.
//! 3. `filter_candidates` applies the row's top-k width, its nucleus threshold and its min-p
//!    threshold, and reports whether the row's candidates form a usable distribution.
//! 4. `softmax` over the candidates (the shared operator in [`crate::norm`]).
//! 5. `inverse_cdf` draws one token per row from one uniform draw, inverting the row's cumulative
//!    distribution in the candidate order the layer supplies; the operator has no order of its own
//!    (`uniserve/sampling/categorical.py` inverts in the order it receives). **That order is the
//!    candidates sorted by token id**, not the descending logit order `top_k` leaves behind: the
//!    reference reorders by token id before the draw "so inverse-CDF draws are deterministic for a
//!    fixed random value independent of top-k kernel ordering, and matches the general path, which
//!    selects in vocabulary order" (`uniserve/sampling/top_k.py:307-314`). The sort is a step of its
//!    own between the filtering and the draw, because the operator that filters does not reorder
//!    rows. A consequence worth stating: a draw of zero selects the *lowest token id* of each row's
//!    candidates, not its largest logit — no draw path produces a greedy token.
//! 6. `argmax` picks the largest candidate, which is a row's greedy token: with the divide-by-one
//!    rule the shaped row keeps the logits' order, so its largest candidate is the temperature-zero
//!    token. The reference computes it separately — `torch.max`, with an `all_gather` across the
//!    tensor-parallel ranks (`uniserve/sampling/greedy.py`) — because no draw can produce it once the
//!    candidates are in token order.
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
    ///
    /// A launch size, not a compilation: the program reads the row at runtime
    /// (`do_not_specialize = ["rows", "vocabulary", "width"]`), so it does not enter the key - a
    /// key that named it would demand a module per vocabulary the deployment happens to serve.
    pub vocabulary: u32,
    /// How many candidates a row keeps: a **launch** value, and the buffers' row stride.
    ///
    /// Nothing of the kept count is compiled in. The program walks the row in `tile`-lane chunks
    /// and orders the kept pairs with radix passes over its scratch, so one module serves any kept
    /// count up to the row's width - which is why the key names none of it.
    pub width: u32,
    /// The lanes one pass reads, and the only size the key names.
    ///
    /// A pass's buffers and its nibble ranks are `tile` lanes wide, so the chunk is program content
    /// the way a normalization's row block is: one module per chunk, whatever the vocabulary and
    /// whatever the candidate count.
    pub tile: u32,
    /// Element type of the emitted vocabulary ids.
    pub ids: DType,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

/// The bins of one ordering pass of `top_k`: a nibble of the order-preserving key.
pub const TOP_K_BINS: u32 = 16;

/// The scratch one `top_k` launch of `rows` rows keeps: for every row two key planes, two id
/// planes, and then two `TOP_K_BINS`-element tables - the descending base of each nibble and the
/// count of it the pass has already placed.
///
/// The planes are the kept count wide, not the vocabulary's: the ordering works on the compacted
/// candidates. It is one tensor per launch rather than five because a launch's operands are its
/// arguments, and the program addresses the planes by offset from its own row.
pub fn top_k_workspace(rows: u64, width: u64) -> Shape {
    Shape::new(&[rows, 4 * width + 2 * u64::from(TOP_K_BINS)])
}

impl fmt::Display for TopKDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `vocabulary` and the kept count are absent on purpose: the program takes both as runtime
        // scalars. What the key names is what a program can be compiled to disagree about, which is
        // the element types and the chunk one pass reads.
        Key::new()
            .field("dtype", self.dtype)
            .field("tile", self.tile)
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
    /// The ordering's scratch, [`top_k_workspace`]'s shape: `[rows, 4 · width + 2 · TOP_K_BINS]`
    /// i32. The plan reserves it; the program never grows one.
    pub workspace: TensorView,
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
        if descriptor.tile == 0 || !descriptor.tile.is_power_of_two() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a tile of {} is not a power of two: one pass reads a whole number of lanes",
                    descriptor.tile
                ),
            ));
        }
        args.logits
            .require_row(Self::NAME, descriptor.dtype, vocabulary)?;
        args.values
            .require_shape(Self::NAME, Shape::new(&[rows, width]))?;
        if args.values.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected the kept logits as f32, found {}: the program recovers them from its \
                     ordered keys as f32, and a narrower buffer would round them a second time",
                    args.values.dtype
                ),
            ));
        }
        args.ids
            .require_shape(Self::NAME, Shape::new(&[rows, width]))?;
        args.workspace
            .require_shape(Self::NAME, top_k_workspace(rows, width))?;
        if args.workspace.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected the ordering's scratch as i32, found {}",
                    args.workspace.dtype
                ),
            ));
        }
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
            .view(args.workspace)
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
    ///
    /// A runtime value, not a compilation: the program walks the row in `tile`-sized chunks, so one
    /// module serves every candidate count and neither this nor the tile is a request the deployment
    /// has to enumerate.
    pub width: u32,
    /// The tile the program walks the row in: a chunk, never the row.
    pub tile: u32,
    /// Element type of the emitted vocabulary ids.
    pub ids: DType,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for FilterDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("width", self.width)
            .field("tile", self.tile)
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
            .u64(width);
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
    /// Lanes one program walks at a time: a chunk of the row, not the row, so the walk costs the
    /// same registers at any width.
    pub tile: u32,
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
            .field("tile", self.tile)
            .field("ids", self.ids)
            .field("tokens", self.tokens)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `inverse_cdf`.
#[derive(Clone, Copy)]
pub struct InverseCdfArgs {
    /// `[rows, width]` FP32 candidate probabilities, in the order the caller supplies them; the layer
    /// sorts by token id first (`uniserve/sampling/top_k.py:307-314`), so the draw does not depend on
    /// the order the top-k kernel happened to produce.
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
        // The row count and the width are both runtime values: the program walks the row in tiles
        // of the descriptor's `tile` lanes, so no block covers the row and nothing in the
        // signature depends on its width.
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
        arguments.view(args.values).view(args.out).u64(rows);
        // The candidate width is compiled into the program, so it is not a parameter here: a word the
        // module does not declare would push the two scratch addresses the compiler's launcher appends
        // out of place.
        kernel.launch(LaunchGrid::linear(rows)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        ArgMaxArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    // The largest value of a row and its index: a comparison, not an arithmetic result, so an
    // implementation that differs has seen a different logits tensor rather than rounded one.
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

    // A copy of one already-computed log-probability per row; the arithmetic happened in
    // `log_softmax`, and this moves its result.
    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        Tolerance::exact()
    }
}

/// One compilation of `candidate_order`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct OrderDescriptor {
    /// Element type of the candidate values, which the permutation does not change.
    pub values: DType,
    /// Element type of the candidate ids.
    pub ids: DType,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for OrderDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The candidate count is a launch value: the reference's width follows the request's top-k,
        // so an extent here would recompile the sampler per request.
        Key::new()
            .field("values", self.values)
            .field("ids", self.ids)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `candidate_order`.
#[derive(Clone, Copy)]
pub struct OrderArgs {
    /// `[rows, width]` the candidate values (probabilities or shaped logits — the permutation is the
    /// same for either).
    pub values: TensorView,
    /// `[rows, width]` the candidate token ids.
    pub ids: TensorView,
    /// `[rows, width]` the values in ascending token order.
    pub ordered_values: TensorView,
    /// `[rows, width]` the ids in ascending token order.
    pub ordered_ids: TensorView,
    /// `[rows, vocabulary]` i32 scratch, one count per token id and then that id's rank.
    ///
    /// The program counts rather than compares, because a row's candidates may be the whole
    /// vocabulary: an ordering that compares every candidate with every other is quadratic in the
    /// vocabulary, and 151936 candidates in one program is sixteen seconds. This buffer is the id
    /// space's, and the program owns it for the duration of the launch.
    pub counts: TensorView,
    /// The id space `counts` covers: the model's vocabulary.
    pub vocabulary: u64,
}

/// Put each row's candidates in ascending token order.
///
/// This is the step between filtering and drawing. [`InverseCdf`] inverts a cumulative distribution
/// in the order it is given and sorts nothing, while the reference reorders the candidates **by token
/// id** before the draw "so inverse-CDF draws are deterministic for a fixed random value independent
/// of top-k kernel ordering, and matches the general path, which selects in vocabulary order"
/// (`uniserve/sampling/top_k.py:307-314`). Without this step a draw would depend on the order the
/// top-k program happened to produce, which is exactly what the reference refuses to depend on.
///
/// It permutes and nothing else: the row's greedy token is not read here. `Sampler::greedy` takes it
/// from `argmax` over the shaped **full-vocabulary** row, where a column is a token id, so the index
/// is the token and "lowest index among ties" is `torch.max`'s rule (`uniserve/sampling/greedy.py`);
/// reading a second greedy token from the candidate set here would be a second answer to the same
/// question, and the two would eventually disagree about a tie.
pub struct CandidateOrder;

impl Op for CandidateOrder {
    const NAME: &'static str = "candidate_order";

    type Descriptor = OrderDescriptor;
    type Args<'a> = OrderArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let rows = args.values.rows();
        let width = args.values.last();
        if width == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a row without candidates has no order".to_owned(),
            ));
        }
        for (name, view, dtype) in [
            ("values", args.values, descriptor.values),
            ("ids", args.ids, descriptor.ids),
            ("ordered_values", args.ordered_values, descriptor.values),
            ("ordered_ids", args.ordered_ids, descriptor.ids),
        ] {
            view.require_shape(Self::NAME, Shape::new(&[rows, width]))?;
            if view.dtype != dtype {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the {name} are {dtype} over [{rows}, {width}], the view says {}",
                        view.dtype
                    ),
                ));
            }
            // A permutation written through rows that interleave would move each other's elements:
            // the program writes one row contiguously, so every view must be packed.
            if view.strides != crate::tensor::Strides::row_major(view.shape) {
                return Err(Error::argument(
                    Self::NAME,
                    format!("the {name} are written row-major; found {}", view.strides),
                ));
            }
        }

        if args.vocabulary < width {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a row of {width} candidates names ids in a vocabulary of {}: the id space \
                     cannot be narrower than the candidates it holds",
                    args.vocabulary
                ),
            ));
        }
        args.counts
            .require_shape(Self::NAME, Shape::new(&[rows, args.vocabulary]))?;
        if args.counts.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the counts are i32 over [{rows}, {}], the view says {}",
                    args.vocabulary, args.counts.dtype
                ),
            ));
        }
        if args.counts.strides != crate::tensor::Strides::row_major(args.counts.shape) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the counts are written row-major; found {}",
                    args.counts.strides
                ),
            ));
        }

        let mut arguments = Arguments::new();
        // The scratch is the program's own: neither twin rule applies to it, and a shadow run gets
        // its own buffer only if the caller gives it one.
        arguments
            .view(args.values)
            .view(args.ids)
            .view(args.ordered_values)
            .view(args.ordered_ids)
            .view(args.counts)
            .u64(rows)
            .u32(u32::try_from(width).map_err(|_| {
                Error::argument(
                    Self::NAME,
                    format!("{width} candidates do not fit a kernel parameter"),
                )
            })?)
            .u32(u32::try_from(args.vocabulary).map_err(|_| {
                Error::argument(
                    Self::NAME,
                    format!(
                        "a vocabulary of {} does not fit a kernel parameter",
                        args.vocabulary
                    ),
                )
            })?);
        kernel.launch(LaunchGrid::linear(rows)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        OrderArgs {
            ordered_values: twins.shadow_of(args.ordered_values),
            ordered_ids: twins.shadow_of(args.ordered_ids),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // A permutation: an implementation that differs has ordered a different candidate set, and
        // the token it selects from a draw is then a different token, which no rounding bound covers.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod order_tests {
    use super::*;

    #[test]
    fn the_key_names_the_types_and_not_the_candidate_count() {
        let descriptor = OrderDescriptor {
            values: DType::F32,
            ids: DType::I32,
            pdl: false,
        };
        assert_eq!(descriptor.to_string(), "values_f32_ids_i32_pdl_false");
    }

    #[test]
    fn the_bound_is_exact_because_a_permutation_has_no_rounding() {
        let descriptor = OrderDescriptor {
            values: DType::F32,
            ids: DType::I32,
            pdl: false,
        };
        let bound = CandidateOrder::tolerance(&descriptor);
        assert_eq!(bound.atol, 0.0);
        assert_eq!(bound.rtol, 0.0);
    }
}
