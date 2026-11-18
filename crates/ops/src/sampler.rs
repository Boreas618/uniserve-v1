//! Sampling: shaping the logits of a request, choosing among its candidates, and reading back the
//! selected values.
//!
//! The steps are separate operators because each is one kernel package and each is replaced on its
//! own during a generation update; the `Sampler` layer calls them in this order:
//!
//! 1. `logits_shape` applies the request's logit adjustments, then its penalties, and divides by its
//!    temperature. The adjustments come first, in vLLM's order (allowed set, bad words and the
//!    `min_tokens` floor, then the logit bias, then the penalties): a row that reads its slot's bias
//!    row adds it, which is `-inf` outside the request's allowed set, and a token the row bans
//!    becomes `-inf`. A `-inf` logit passes the penalties and the temperature unchanged, so masking
//!    before or after them is the same, and the bias is the one adjustment whose place matters: the
//!    repetition penalty scales the biased logit. A row that reads no bias row and bans nothing is
//!    shaped exactly as without the adjustments, rounding for rounding. **A row whose
//!    temperature is zero divides by one**, not by zero: callers select those rows by argmax, and
//!    without that rule every greedy request would be shaped into `inf` and `nan` logits. The
//!    shaped row is then the penalized logits themselves, so the row's largest candidate is its
//!    greedy token.
//! 2. `sampling_from_logits` draws one token per row from one uniform draw, under the row's top-k
//!    width, its nucleus threshold and its min-p threshold, and reports whether the row's kept
//!    candidates form a usable distribution. It inverts the kept candidates' cumulative
//!    distribution **in ascending token id**, so a draw is deterministic for a fixed random value
//!    whatever order equal logits are kept in. A consequence worth stating: a draw of zero selects
//!    the *lowest token id* of each row's kept candidates, not its largest logit — no draw path
//!    produces a greedy token.
//! 3. `argmax` picks the largest shaped logit, which is a row's greedy token: with the divide-by-one
//!    rule the shaped row keeps the logits' order, so its largest entry is the temperature-zero
//!    token. The layer computes it separately — an argmax, gathered across the tensor-parallel
//!    ranks — because no draw produces it.
//! 4. `top_k` keeps the `width` largest entries of each row, sorted, with the vocabulary id of
//!    each: a selection of a model's own (Qwen4's kept attention groups), not a step of the draw.
//! 5. `log_softmax` and `logprob_gather` report the log-probability of the tokens the caller asks
//!    for.
//!
//! Tolerances: the shaping and the filtering only combine FP32 values step by step and are compared
//! with the bound of their own few roundings; a chosen token id is exact — two implementations that
//! disagree about a token disagree about the sample, which no tolerance can cover.

use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::{Key, row_tile};
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::launch::PDL;
use crate::operand::Operand;
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

use proto::module::sampler::PARAMETER_COLUMNS;

/// The element type the sampling kernels read an index as: a token id, a slot or a mask index
/// reaches them as `I32`, where the entry that names nothing is `-1`.
pub const INDEX: DType = DType::I32;

/// The most columns one program of `logits_shape` shapes: one register tile of its kernel, so a
/// row is shaped by `ceil(vocabulary / SHAPE_TILE)` programs side by side. The shaping is one
/// element at a time, so the tile changes no value; a program per row would leave a decode
/// step's few rows to as few programs, each walking a vocabulary-wide row alone.
const SHAPE_TILE: u32 = 4096;

/// Columns one pass of `top_k` reads: the only size its key names, since the vocabulary and the
/// kept count are launch values.
const TOP_TILE: u32 = 2048;

/// Columns one program of the first, direct `top_logprobs` selection reads: a vocabulary-wide
/// row is selected in chunks of this many columns in parallel.
const LOGPROB_TILE: u32 = 4096;

/// The widest row one `top_logprobs` program selects from, which bounds the chunks' candidates
/// the second selection reads.
const LOGPROB_MAX_TILE: u32 = 8192;

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

impl ShapeDescriptor {
    /// The compilation that shapes `dtype` logits of `vocabulary` tokens with `extras` extra
    /// token columns, dividing by the temperature when `divide`: a program shapes a row's tile,
    /// at most [`SHAPE_TILE`] columns.
    pub fn new(dtype: DType, vocabulary: u32, extras: u32, divide: bool) -> Self {
        Self {
            dtype,
            vocabulary,
            block: row_tile(vocabulary).min(SHAPE_TILE),
            extras,
            divide,
            pdl: PDL,
        }
    }
}

impl fmt::Display for ShapeDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("vocabulary", self.vocabulary)
            .field("block", self.block)
            .field("extras", self.extras)
            .field("divide", self.divide)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `logits_shape`.
#[derive(Clone)]
pub struct ShapeArgs {
    /// `[rows, vocabulary]` raw logits.
    pub logits: Operand,
    /// `[slots, vocabulary]` i32 counts of the tokens every slot has already seen.
    pub counts: Operand,
    /// `[rows]` i32: the slot whose counts each row reads.
    pub slots: Operand,
    /// `[rows, PARAMETER_COLUMNS]` f32 request parameters.
    pub parameters: Operand,
    /// `[rows, extras]` i32 tokens counted besides the slot's, `-1` counting nothing.
    pub extras: Option<Operand>,
    /// `[slots, vocabulary]` f32 each slot's bias row: its request's logit bias, `-inf` outside its
    /// allowed set, `0` elsewhere. Read only by a row whose adjustment flag names it.
    pub bias: Operand,
    /// `[rows, 2]` i32 each row's adjustments: whether it adds its slot's bias row (`1`) or not
    /// (`0`), and the index of its ban mask among `bans`, `-1` for none.
    pub adjustments: Operand,
    /// `[masks, ceil(vocabulary / 32)]` i32 ban masks: bit `t % 32` of word `t / 32` set bans token
    /// `t`. Read only through a row's mask index.
    pub bans: Operand,
    /// `[rows, vocabulary]` FP32 shaped logits.
    pub out: Operand,
}

/// The words of one ban mask over a `vocabulary`-wide row: one bit per token.
pub const fn ban_words(vocabulary: u32) -> u32 {
    vocabulary.div_ceil(32)
}

/// Adjust, penalize and temperature-scale the logits of every row.
pub struct LogitsShape;

impl Op for LogitsShape {
    const NAME: &'static str = proto::module::operator::LOGITS_SHAPE;

    type Descriptor = ShapeDescriptor;
    type Args<'a> = ShapeArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
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
        args.slots.require_shape(Self::NAME, &[rows])?;
        args.parameters
            .require_shape(Self::NAME, &[rows, u64::from(PARAMETER_COLUMNS)])?;
        args.counts
            .require_row(Self::NAME, DType::I32, vocabulary)?;
        args.bias.require_row(Self::NAME, DType::F32, vocabulary)?;
        if args.bias.rows() != args.counts.rows() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} bias rows for {} count rows: both are one row per slot",
                    args.bias.rows(),
                    args.counts.rows()
                ),
            ));
        }
        args.adjustments.require_shape(Self::NAME, &[rows, 2])?;
        args.bans.require_row(
            Self::NAME,
            DType::I32,
            u64::from(ban_words(descriptor.vocabulary)),
        )?;
        match (descriptor.extras, &args.extras) {
            (0, None) => {}
            (extras, Some(extra)) => {
                extra.require_shape(Self::NAME, &[rows, u64::from(extras)])?;
            }
            (extras, None) => {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the descriptor declares {extras} extra tokens and the call passed none"
                    ),
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
        let vocabulary = u64::from(descriptor.vocabulary);
        let rows = args.logits.rows();
        // A descriptor without extra tokens reads the slots in their place.
        let extras = args.extras.as_ref().unwrap_or(&args.slots);

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("logits_ptr", &args.logits)
            .view("counts_ptr", &args.counts)
            .view("slots_ptr", &args.slots)
            .view("parameters_ptr", &args.parameters)
            .view("extras_ptr", extras)
            .view("bias_ptr", &args.bias)
            .view("adjustments_ptr", &args.adjustments)
            .view("bans_ptr", &args.bans)
            .view("out_ptr", &args.out)
            .u64("rows", rows)
            .u32("vocabulary", descriptor.vocabulary)
            .flag("divide", descriptor.divide);
        kernel.launch(
            LaunchGrid::new([rows, vocabulary.div_ceil(u64::from(descriptor.block)), 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        ShapeArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // Five rounded FP32 operations per element at most: the bias, the penalty, its two parts
        // and the temperature division. The comparison is against the same inputs, so the bound is
        // those roundings twice over.
        Tolerance::rounding(5, DType::F32, DType::F32)
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
pub fn top_k_workspace(rows: u64, width: u64) -> [u64; 2] {
    [rows, 4 * width + 2 * u64::from(TOP_K_BINS)]
}

impl TopKDescriptor {
    /// The compilation that keeps the `width` largest of rows of `vocabulary` `dtype` logits.
    pub fn new(dtype: DType, vocabulary: u32, width: u32) -> Self {
        Self {
            dtype,
            vocabulary,
            width,
            tile: TOP_TILE,
            ids: INDEX,
            pdl: PDL,
        }
    }
}

impl fmt::Display for TopKDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `vocabulary` and the kept count are absent on purpose: the program takes both as runtime
        // scalars. What the key names is what a program can be compiled to disagree about, which is
        // the element types and the chunk one pass reads.
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("tile", self.tile)
            .field("ids", abi::name(self.ids))
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `top_k`.
#[derive(Clone)]
pub struct TopKArgs {
    /// `[rows, vocabulary]` shaped logits.
    pub logits: Operand,
    /// `[rows, width]` the kept logits, descending; a tie keeps the lower vocabulary id first.
    pub values: Operand,
    /// `[rows, width]` the vocabulary ids of the kept logits.
    pub ids: Operand,
    /// The ordering's scratch, [`top_k_workspace`]'s shape: `[rows, 4 · width + 2 · TOP_K_BINS]`
    /// i32. The caller allocates it; the program never grows one.
    pub workspace: Operand,
}

/// The `width` largest logits of every row, sorted.
pub struct TopK;

impl Op for TopK {
    const NAME: &'static str = proto::module::operator::TOP_K;

    type Descriptor = TopKDescriptor;
    type Args<'a> = TopKArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
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
        args.values.require_shape(Self::NAME, &[rows, width])?;
        if args.values.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected the kept logits as f32, found {}: the program recovers them from its \
                     ordered keys as f32, and a narrower buffer would round them a second time",
                    abi::name(args.values.dtype)
                ),
            ));
        }
        args.ids.require_shape(Self::NAME, &[rows, width])?;
        args.workspace
            .require_shape(Self::NAME, &top_k_workspace(rows, width))?;
        if args.workspace.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected the ordering's scratch as i32, found {}",
                    abi::name(args.workspace.dtype)
                ),
            ));
        }
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
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let rows = args.logits.rows();

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("logits_ptr", &args.logits)
            .view("values_ptr", &args.values)
            .view("ids_ptr", &args.ids)
            .view("workspace_ptr", &args.workspace)
            .u64("rows", rows)
            .u32("vocabulary", descriptor.vocabulary)
            .u32("width", descriptor.width);
        kernel.launch(LaunchGrid::linear(rows)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
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

/// One compilation of `sampling_from_logits`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SamplingDescriptor {
    /// Element type of the shaped logits.
    pub logits: DType,
    /// Element type of the emitted tokens.
    pub tokens: DType,
}

impl SamplingDescriptor {
    /// The compilation that draws from FP32 shaped logits.
    pub fn new() -> Self {
        Self {
            logits: DType::F32,
            tokens: INDEX,
        }
    }
}

impl Default for SamplingDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SamplingDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The vocabulary and the row count are launch values: one module serves every model.
        Key::new()
            .field("logits", abi::name(self.logits))
            .field("tokens", abi::name(self.tokens))
            .finish(formatter)
    }
}

/// The widest vocabulary `sampling_from_logits` draws from: the kernel's draw sums its row in at
/// most one segment of 1024 tokens per thread of its block of 1024.
pub const SAMPLING_MAX_VOCABULARY: u64 = 1 << 20;

/// The arguments of `sampling_from_logits`.
#[derive(Clone)]
pub struct SamplingArgs {
    /// `[rows, vocabulary]` FP32 shaped logits, unit stride along the vocabulary.
    pub logits: Operand,
    /// `[rows, PARAMETER_COLUMNS]` FP32 request parameters: the top-k width, the nucleus
    /// threshold and the min-p threshold are read.
    pub parameters: Operand,
    /// `[rows]` FP32 uniform draws in `[0, 1)`.
    pub draws: Operand,
    /// `[rows]` the drawn token of each row, written.
    pub tokens: Operand,
    /// `[rows]` I32, written: `1` for a row whose kept candidates form a distribution.
    pub valid: Operand,
}

/// One token per row, drawn from the row's shaped logits under its top-k width, its nucleus
/// threshold and its min-p threshold.
///
/// In the row's descending order (equal logits in ascending token id; `+0.0` above `-0.0`), the
/// three filters each keep a prefix, applied in this order:
///
/// ```text
/// limit    = top_k > 0 ? floor(min(top_k, vocabulary)) : vocabulary
/// kept     = the first `limit` positions                               the top-k width
/// m_j      = exp(x_j − max x) in units of 2⁻⁴⁰, rounded                each position's mass
/// kept     = kept positions j with Σ_{i<j} m_i ≤ floor(top_p · Σ_kept m)  the nucleus (top_p < 1)
/// kept     = kept positions with x_j ≥ max x + ln(min_p)               min-p (min_p > 0)
/// token    = the first kept token by id whose cumulative mass exceeds floor(draw · Σ_kept m)
/// ```
///
/// The draw inverts the kept candidates' cumulative distribution in ascending token id, so a fixed
/// draw selects the same token whatever order equal logits are kept in, and a draw of zero selects
/// the lowest kept token id with mass. The masses are integers, so the nucleus's comparison and
/// the cumulative are exact and a launch is deterministic. A row reports `0` and holds token `0`
/// when it keeps nothing (a width below one, a negative nucleus, a min-p threshold above the
/// largest logit or NaN, no finite logit) or when its kept candidates hold a NaN or `+inf`.
///
/// The kernel orders nothing: it finds each filter's boundary by radix selection over the logits'
/// order-preserving keys and draws in one more walk of the row in token order (the package's
/// source states the passes).
pub struct SamplingFromLogits;

impl Op for SamplingFromLogits {
    const NAME: &'static str = proto::module::operator::SAMPLING_FROM_LOGITS;

    type Descriptor = SamplingDescriptor;
    type Args<'a> = SamplingArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        if descriptor.logits != DType::F32 {
            return Err(Error::Unsupported {
                operation: Self::NAME,
                reason: format!(
                    "shaped logits of {} are not drawn from: the shaping writes FP32",
                    abi::name(descriptor.logits)
                ),
            });
        }
        if args.logits.rank() != 2 {
            return Err(Error::argument(
                Self::NAME,
                format!("the logits are a matrix, found {:?}", args.logits.dims()),
            ));
        }
        let rows = args.logits.rows();
        let vocabulary = args.logits.last();
        if vocabulary == 0 || vocabulary > SAMPLING_MAX_VOCABULARY {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a vocabulary of {vocabulary} is outside 1..={SAMPLING_MAX_VOCABULARY}, the \
                     widest row the draw sums"
                ),
            ));
        }
        args.logits
            .require_row(Self::NAME, descriptor.logits, vocabulary)?;
        i32::try_from(rows).map_err(|_| {
            Error::argument(
                Self::NAME,
                format!("{rows} rows do not fit a kernel parameter"),
            )
        })?;
        args.parameters
            .require_shape(Self::NAME, &[rows, u64::from(PARAMETER_COLUMNS)])?;
        if args.parameters.dtype != DType::F32 || !args.parameters.is_row_major() {
            return Err(Error::argument(
                Self::NAME,
                "the parameters are row-major FP32 rows".to_owned(),
            ));
        }
        args.draws.require_shape(Self::NAME, &[rows])?;
        if args.draws.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected f32 draws, found {}", abi::name(args.draws.dtype)),
            ));
        }
        args.tokens.require_shape(Self::NAME, &[rows])?;
        if args.tokens.dtype != descriptor.tokens {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} tokens, found {}",
                    abi::name(descriptor.tokens),
                    abi::name(args.tokens.dtype)
                ),
            ));
        }
        args.valid.require_shape(Self::NAME, &[rows])?;
        if args.valid.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected i32 validity flags, found {}",
                    abi::name(args.valid.dtype)
                ),
            ));
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        _descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let rows = args.logits.rows();
        let vocabulary = args.logits.last();
        let row_stride = args.logits.stride(0);

        // One block draws one row.
        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("logits", &args.logits)
            .i64(
                "row_stride",
                i64::try_from(row_stride)
                    .map_err(|_| Error::argument(Self::NAME, "a row stride above i64::MAX"))?,
            )
            .i32("rows", rows as i32)
            .i32("vocabulary", vocabulary as i32)
            .view("parameters", &args.parameters)
            .view("draws", &args.draws)
            .view("tokens", &args.tokens)
            .view("valid", &args.valid);
        kernel.launch(LaunchGrid::linear(rows)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        SamplingArgs {
            tokens: twins.shadow_of(args.tokens),
            valid: twins.shadow_of(args.valid),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // A sample is a token id: two implementations that differ have drawn from different
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

impl ArgMaxDescriptor {
    /// The compilation that finds the largest of rows of `width` `dtype` values.
    pub fn new(dtype: DType, width: u32) -> Self {
        Self {
            dtype,
            width,
            ids: INDEX,
            pdl: PDL,
        }
    }
}

impl fmt::Display for ArgMaxDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("width", self.width)
            .field("ids", abi::name(self.ids))
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `argmax`.
#[derive(Clone)]
pub struct ArgMaxArgs {
    /// `[rows, width]` logits or candidates.
    pub values: Operand,
    /// `[rows]` the index of each row's largest value, the lowest index among ties.
    pub out: Operand,
}

/// The largest value's index, per row.
///
/// **A row with no numeric value answers the row's width, not an index.** The program's running
/// index starts at `width` and every tile's index is the smallest column equal to that tile's
/// largest value, falling back to `width`; a row whose elements are all NaN or all `-inf` therefore
/// answers `width`, which is one past the last column. When the row is a vocabulary that value is
/// the vocabulary itself — a number a caller can mistake for a token id. So a reader of `out`
/// refuses `index >= width` where it reads a token, and the refusal names the row, because only the
/// reader can say that the row it sampled had no finite logit.
pub struct ArgMax;

impl Op for ArgMax {
    const NAME: &'static str = proto::module::operator::ARGMAX;

    type Descriptor = ArgMaxDescriptor;
    type Args<'a> = ArgMaxArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let width = u64::from(descriptor.width);
        let rows = args.values.rows();
        args.values
            .require_row(Self::NAME, descriptor.dtype, width)?;
        args.out.require_shape(Self::NAME, &[rows])?;
        if args.out.dtype != descriptor.ids {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} ids, found {}",
                    abi::name(descriptor.ids),
                    abi::name(args.out.dtype)
                ),
            ));
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        _descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let rows = args.values.rows();

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("values_ptr", &args.values)
            .view("out_ptr", &args.out)
            .u64("rows", rows);
        // The candidate width is compiled into the program, so it is no launch word.
        kernel.launch(LaunchGrid::linear(rows)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
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

impl LogSoftmaxDescriptor {
    /// The compilation that takes the log-softmax of rows of `columns` `dtype` logits: one
    /// program loads its whole row in one tile.
    pub fn new(dtype: DType, columns: u32) -> Self {
        Self {
            dtype,
            columns,
            tile: row_tile(columns),
            pdl: PDL,
        }
    }
}

impl fmt::Display for LogSoftmaxDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("columns", self.columns)
            .field("tile", self.tile)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `log_softmax`.
#[derive(Clone)]
pub struct LogSoftmaxArgs {
    /// `[rows, columns]`.
    pub x: Operand,
    /// `[rows, columns]` FP32 log-probabilities.
    pub out: Operand,
}

/// `log_softmax` over the last axis.
pub struct LogSoftmax;

impl Op for LogSoftmax {
    const NAME: &'static str = proto::module::operator::LOG_SOFTMAX;

    type Descriptor = LogSoftmaxDescriptor;
    type Args<'a> = LogSoftmaxArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
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
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("x_ptr", &args.x)
            .view("out_ptr", &args.out)
            .u32("columns", descriptor.columns);
        kernel.launch(LaunchGrid::linear(args.x.rows())?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
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
    /// The lanes the rank's count reads at a time: a chunk of the row, so the count costs the same
    /// registers at any width.
    pub tile: u32,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl LogprobDescriptor {
    /// The compilation that gathers one log-probability of rows of `columns` and ranks it: one
    /// program loads its whole row in one tile.
    pub fn new(columns: u32) -> Self {
        Self {
            columns,
            ids: INDEX,
            tile: row_tile(columns),
            pdl: PDL,
        }
    }
}

impl fmt::Display for LogprobDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("columns", self.columns)
            .field("ids", abi::name(self.ids))
            .field("tile", self.tile)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of `logprob_gather`.
#[derive(Clone)]
pub struct LogprobArgs {
    /// `[rows, columns]` FP32 log-probabilities.
    pub logprobabilities: Operand,
    /// `[rows]` the tokens whose log-probability is reported; a negative token names no position.
    pub tokens: Operand,
    /// `[rows]` FP32: the token's log-probability, `0` for a row whose token names no position.
    pub out: Operand,
    /// `[rows]` i32: the token's one-based competition rank in its row - one plus the entries
    /// strictly above it, so tied entries share a rank - and `0` for a row whose token names no
    /// position.
    pub ranks: Operand,
}

/// The log-probability of one chosen token per row, and its rank in the row.
pub struct LogprobGather;

impl Op for LogprobGather {
    const NAME: &'static str = proto::module::operator::LOGPROB_GATHER;

    type Descriptor = LogprobDescriptor;
    type Args<'a> = LogprobArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let columns = u64::from(descriptor.columns);
        let rows = args.logprobabilities.rows();
        args.logprobabilities
            .require_row(Self::NAME, DType::F32, columns)?;
        args.tokens.require_shape(Self::NAME, &[rows])?;
        if args.tokens.dtype != descriptor.ids {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} tokens, found {}",
                    abi::name(descriptor.ids),
                    abi::name(args.tokens.dtype)
                ),
            ));
        }
        args.out.require_shape(Self::NAME, &[rows])?;
        if args.out.dtype != DType::F32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected f32 log-probabilities, found {}",
                    abi::name(args.out.dtype)
                ),
            ));
        }
        args.ranks.require_shape(Self::NAME, &[rows])?;
        if args.ranks.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected i32 ranks, found {}", abi::name(args.ranks.dtype)),
            ));
        }
        if descriptor.tile == 0 {
            return Err(Error::argument(Self::NAME, "a rank's count needs a tile"));
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let rows = args.logprobabilities.rows();

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("logprobabilities_ptr", &args.logprobabilities)
            .view("tokens_ptr", &args.tokens)
            .view("out_ptr", &args.out)
            .view("ranks_ptr", &args.ranks)
            .u64("rows", rows)
            .u32("columns", descriptor.columns);
        kernel.launch(LaunchGrid::linear(rows)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        LogprobArgs {
            out: twins.shadow_of(args.out),
            ranks: twins.shadow_of(args.ranks),
            ..args
        }
    }

    // A copy of one already-computed log-probability per row, and a count of the entries above it;
    // the arithmetic happened in `log_softmax`, and this moves and compares its result.
    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        Tolerance::exact()
    }
}

/// One compilation of `top_logprobs`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TopLogprobsDescriptor {
    /// The entries one chunk keeps: the deployment's `max_logprobs`.
    pub candidates: u32,
    /// The lanes one program reads: a chunk of the row, a power of two up to 8192.
    pub tile: u32,
    /// Whether each entry's id is read beside it (a selection over a first launch's candidates)
    /// rather than being its column.
    pub indirect: bool,
    /// Launch with programmatic dependent launch.
    pub pdl: bool,
}

impl TopLogprobsDescriptor {
    /// The two selections of the `candidates` largest entries of rows `width` wide: `(chunks,
    /// merge)`. The first keeps each [`LOGPROB_TILE`]-column chunk's own; the second reads every
    /// chunk's candidates of a row, with their ids, in one tile of the power of two that holds
    /// them.
    ///
    /// # Errors
    ///
    /// [`Error::Argument`] when `candidates` is zero, or when the chunks' candidates of a row are
    /// more than one program selects from ([`LOGPROB_MAX_TILE`]).
    pub fn selections(width: u64, candidates: u32) -> Result<(Self, Self)> {
        let chunks = Self {
            candidates,
            tile: LOGPROB_TILE,
            indirect: false,
            pdl: PDL,
        };
        let kept = chunks.written(width);
        let merge_tile = kept.next_power_of_two();
        if candidates == 0 || merge_tile > u64::from(LOGPROB_MAX_TILE) {
            return Err(Error::argument(
                TopLogprobs::NAME,
                format!(
                    "{candidates} candidates keep {kept} of the chunks of a row of {width}: one \
                     to {LOGPROB_MAX_TILE} expected"
                ),
            ));
        }

        let merge = Self {
            tile: merge_tile as u32,
            indirect: true,
            ..chunks
        };
        Ok((chunks, merge))
    }

    /// The entries a launch of this compilation writes a row of a `width`-wide input:
    /// `chunks · candidates`.
    pub fn written(&self, width: u64) -> u64 {
        top_logprobs_chunks(width, self.tile) * u64::from(self.candidates)
    }
}

impl fmt::Display for TopLogprobsDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("candidates", self.candidates)
            .field("tile", self.tile)
            .field("indirect", self.indirect)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The chunks a `top_logprobs` launch of `tile` lanes splits a `width`-wide row into: the
/// launch writes `chunks · candidates` entries a row.
pub fn top_logprobs_chunks(width: u64, tile: u32) -> u64 {
    width.div_ceil(u64::from(tile.max(1)))
}

/// The arguments of `top_logprobs`.
#[derive(Clone)]
pub struct TopLogprobsArgs {
    /// `[rows, width]` FP32 entries: log-probabilities, or a first launch's candidates.
    pub values: Operand,
    /// `[rows, width]` i32 each entry's id, for an indirect launch.
    pub ids: Option<Operand>,
    /// `[rows, chunks · candidates]` FP32: each chunk's kept entries, the largest first.
    pub out_values: Operand,
    /// `[rows, chunks · candidates]` i32: their ids, `-1` past a chunk's entries.
    pub out_ids: Operand,
}

/// The `candidates` largest entries of every chunk of every row, with their ids.
///
/// Each chunk's are in descending order, ties by ascending id; a launch whose row is one chunk
/// wide is the row's own selection, and an indirect launch over a first launch's output selects
/// the row's from its chunks' (`Sampler::score`).
pub struct TopLogprobs;

impl Op for TopLogprobs {
    const NAME: &'static str = proto::module::operator::TOP_LOGPROBS;

    type Descriptor = TopLogprobsDescriptor;
    type Args<'a> = TopLogprobsArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        if descriptor.candidates == 0 {
            return Err(Error::argument(Self::NAME, "a selection keeps a candidate"));
        }
        if !descriptor.tile.is_power_of_two() || descriptor.tile > 8192 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a tile of {} is not a power of two up to 8192",
                    descriptor.tile
                ),
            ));
        }
        let rows = args.values.rows();
        let width = args.values.last();
        args.values.require_row(Self::NAME, DType::F32, width)?;
        match (descriptor.indirect, &args.ids) {
            (true, Some(ids)) => ids.require_shape(Self::NAME, &[rows, width])?,
            (false, None) => {}
            _ => {
                return Err(Error::argument(
                    Self::NAME,
                    "an indirect selection reads ids, and only it does",
                ));
            }
        }
        let kept = top_logprobs_chunks(width, descriptor.tile) * u64::from(descriptor.candidates);
        args.out_values.require_shape(Self::NAME, &[rows, kept])?;
        args.out_ids.require_shape(Self::NAME, &[rows, kept])?;
        for (name, view, dtype) in [
            ("ids", args.ids.as_ref(), DType::I32),
            ("kept values", Some(&args.out_values), DType::F32),
            ("kept ids", Some(&args.out_ids), DType::I32),
        ] {
            if let Some(view) = view
                && view.dtype != dtype
            {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "expected {} {name}, found {}",
                        abi::name(dtype),
                        abi::name(view.dtype)
                    ),
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
        let rows = args.values.rows();
        let width = args.values.last();
        let chunks = top_logprobs_chunks(width, descriptor.tile);

        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("values_ptr", &args.values)
            // A direct launch reads no ids; the slot still takes a device address.
            .view("ids_ptr", args.ids.as_ref().unwrap_or(&args.out_ids))
            .view("out_values_ptr", &args.out_values)
            .view("out_ids_ptr", &args.out_ids)
            .u64("width", width)
            .u64("tiles", chunks);
        kernel.launch(LaunchGrid::new([rows, chunks, 1])?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        TopLogprobsArgs {
            out_values: twins.shadow_of(args.out_values),
            out_ids: twins.shadow_of(args.out_ids),
            ..args
        }
    }

    // A selection moves values and ids; two implementations must keep the same ones.
    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        Tolerance::exact()
    }
}
