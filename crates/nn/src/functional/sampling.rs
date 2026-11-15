//! Token selection: shaping a row's logits, choosing among its candidates, and reporting the
//! log-probabilities of the chosen token.
//!
//! ```text
//! logits_shape        shaped = penalize(adjust(logits)) / temperature
//! top_k               the `width` largest entries of a row, descending, with their token ids
//! filter_candidates_  the row's top-k width, nucleus and min-p thresholds: a dropped one is -inf
//! candidate_order     the candidates in ascending token id
//! softmax             p = exp(x − max x) / Σ exp(x − max x)
//! inverse_cdf         the first candidate whose cumulative mass exceeds draw · Σ p
//! argmax              the largest entry's column, the lowest among equals
//! log_softmax         x − (max x + ln Σ exp(x − max x))
//! logprob_gather      one token's log-probability and its competition rank
//! top_logprobs        the `n` largest entries of a row, descending, with their token ids
//! ```
//!
//! A drawn token is `logits_shape`, `top_k`, `filter_candidates_`, `candidate_order`, `softmax`
//! and `inverse_cdf`, in that order; a greedy token is `logits_shape` and `argmax`. The candidates
//! reach the draw in ascending token id, so a fixed draw selects the same token whatever order
//! the selection kept equal scores in. No function here generates randomness: a draw is a
//! uniform value its caller computes.
//!
//! ## Rows and columns
//!
//! Every function reads `[rows, columns]`: one row per sequence being sampled, one column per
//! token of the vocabulary or per candidate. A row's request parameters are one row of
//! `[rows, PARAMETER_COLUMNS]` FP32 in the order `proto::module::sampler` states: the
//! temperature at 0, the top-k width at 1, the nucleus threshold at 2, the min-p threshold at 3
//! and the repetition, frequency and presence penalties at 4, 5 and 6.
//!
//! ## Indices
//!
//! Token ids, slots and mask indices are `U32`, and [`NO_SLOT`] is the entry that names none: a
//! padding row's slot, a row without a ban mask, a candidate slot no candidate took, a row
//! whose token is not scored. The kernels are compiled to read an index as `I32`, where that
//! entry is `-1`, so a launch states each index tensor read as `I32`, bit for bit, and the keys
//! name `i32`. Counts, ranks and validity flags are `I32` values and are stated as they are.

use ops::Operand;
use ops::norm::{Softmax, SoftmaxArgs, SoftmaxDescriptor};
use ops::sampler::{
    ArgMax, ArgMaxArgs, ArgMaxDescriptor, CandidateOrder, FilterArgs, FilterCandidates,
    FilterDescriptor, INDEX, InverseCdf, InverseCdfArgs, InverseCdfDescriptor, LogSoftmax,
    LogSoftmaxArgs, LogSoftmaxDescriptor, LogitsShape, LogprobArgs, LogprobDescriptor,
    LogprobGather, OrderArgs, OrderDescriptor, ShapeArgs, ShapeDescriptor, TopK, TopKArgs,
    TopKDescriptor, TopLogprobs, TopLogprobsArgs, TopLogprobsDescriptor, ban_words,
    top_k_workspace,
};
use proto::module::sampler::{PARAMETER_COLUMNS, TEMPERATURE, TOP_K, TOP_P};
use tensor::{D, DType, Device, Tensor};

use super::nvfp4::integer;
use super::{ACCUMULATE, Fused, as_rows, compute, like_rows};
use crate::capability::NO_SLOT;
use crate::error::{Error, Result};

/// The offset of the min-p threshold among a row's parameters. `proto::module::sampler` names
/// the first three offsets and states this one and the three below by position.
const MIN_P: usize = 3;

/// The offset of the repetition penalty among a row's parameters.
const REPETITION: usize = 4;

/// The offset of the frequency penalty among a row's parameters.
const FREQUENCY: usize = 5;

/// The offset of the presence penalty among a row's parameters.
const PRESENCE: usize = 6;

/// The first index the kernels read as negative: an index at or above it names nothing.
const UNNAMED: u32 = 1 << 31;

/// Columns the CPU formulas accumulate a row's cumulative sum over at a time. The tensor
/// library's cumulative sum is a product with a triangular matrix of the row's width squared,
/// so a vocabulary-wide row is summed chunk by chunk with the total carried between chunks, as
/// the kernels carry it between tiles.
const CUMULATIVE_CHUNK: usize = 1024;

/// The widest row the CPU formulas order: a key and a column are combined in one FP64 value,
/// which is exact while `2³² · columns` stays below `2⁵³`.
const MAX_ORDERED: usize = 1 << 21;

/// The kernel argument of `indices`, a `U32` index tensor, read as the `I32` the sampling
/// kernels are compiled for: the same elements, bit for bit, with [`NO_SLOT`] as `-1`.
pub(super) fn signed(indices: &Tensor) -> Result<Operand> {
    Ok(ops::operand(&indices.view(INDEX)?)?)
}

/// Checks that `x`, the tensor `name`, holds `dtype` elements in `shape`, where `None` is an
/// extent the caller does not fix.
pub(super) fn require(
    owner: &'static str,
    name: &str,
    x: &Tensor,
    dtype: DType,
    shape: &[Option<usize>],
) -> Result<()> {
    let fits = x.dtype() == dtype
        && x.rank() == shape.len()
        && x.dims()
            .iter()
            .zip(shape)
            .all(|(dim, expected)| expected.is_none_or(|expected| expected == *dim));
    if fits {
        return Ok(());
    }

    let extents: Vec<String> = shape
        .iter()
        .map(|extent| extent.map_or_else(|| "_".to_owned(), |extent| extent.to_string()))
        .collect();
    Err(Error::shape(
        owner,
        format!(
            "the {name} is {} {:?}, {} [{}] expected",
            ops::abi::name(x.dtype()),
            x.dims(),
            ops::abi::name(dtype),
            extents.join(", ")
        ),
    ))
}

/// The rows and columns of `x`, the tensor `name` of `[rows, columns]` with at least one of
/// each.
fn matrix(owner: &'static str, name: &str, x: &Tensor) -> Result<(usize, usize)> {
    match *x.dims() {
        [rows, columns] if rows > 0 && columns > 0 => Ok((rows, columns)),
        _ => Err(Error::shape(
            owner,
            format!("the {name} is {:?}, [rows, columns] expected", x.dims()),
        )),
    }
}

/// `extent` as a descriptor states it.
fn extent(owner: &'static str, name: &str, extent: usize) -> Result<u32> {
    u32::try_from(extent).map_err(|_| {
        Error::shape(
            owner,
            format!("{extent} {name} are more than a kernel takes"),
        )
    })
}

/// `on_true` where `mask` is set and `on_false` elsewhere, the first two broadcast to
/// `on_false`'s shape.
fn choose(mask: &Tensor, on_true: &Tensor, on_false: &Tensor) -> Result<Tensor> {
    let shape = on_false.shape();

    Ok(mask
        .broadcast_as(shape)?
        .where_cond(&on_true.broadcast_as(shape)?, on_false)?)
}

/// Which entries of `indices`, a `U32` CPU tensor, name something: the ones a kernel reads as
/// zero or more.
fn named(indices: &Tensor) -> Result<Tensor> {
    Ok(indices.broadcast_lt(&integer(UNNAMED, indices)?)?)
}

/// The inclusive cumulative sum of each row of `x`, FP32 `[rows, columns]`, summed in chunks of
/// [`CUMULATIVE_CHUNK`] columns with each chunk's total carried into the next.
fn cumulative(x: &Tensor) -> Result<Tensor> {
    let (rows, columns) = (x.dims()[0], x.dims()[1]);
    let chunks = columns.div_ceil(CUMULATIVE_CHUNK);
    if chunks == 1 {
        return Ok(x.cumsum(D::Minus1)?);
    }

    let padded = chunks * CUMULATIVE_CHUNK;
    let within = x
        .pad_with_zeros(1, 0, padded - columns)?
        .reshape((rows, chunks, CUMULATIVE_CHUNK))?
        .cumsum(D::Minus1)?;

    // The mass before a chunk is the sum of the totals of the chunks before it.
    let totals = within.narrow(2, CUMULATIVE_CHUNK - 1, 1)?.squeeze(2)?;
    let before = totals
        .narrow(1, 0, chunks - 1)?
        .pad_with_zeros(1, 1, 0)?
        .cumsum(D::Minus1)?;

    Ok(within
        .broadcast_add(&before.unsqueeze(2)?)?
        .reshape((rows, padded))?
        .narrow(1, 0, columns)?)
}

/// The order-preserving unsigned keys of `values`, FP32: a negative value's bits inverted and
/// every other value's sign bit set, the map of IEEE floats onto unsigned integers `top_k`
/// orders by. `-0.0` is below `+0.0` under it.
fn monotone_keys(values: &Tensor) -> Result<Tensor> {
    let bits = values.view(DType::U32)?;
    let sign = integer(UNNAMED, &bits)?;
    let negative = bits.broadcast_ge(&sign)?;

    Ok(negative.where_cond(
        &bits.broadcast_bitwise_xor(&integer(u32::MAX, &bits)?)?,
        &bits.broadcast_bitwise_or(&sign)?,
    )?)
}

/// The columns of each row of `keys`, `[rows, columns]` U32, in descending key order with equal
/// keys in ascending column order: `[rows, columns]` U32.
fn descending(owner: &'static str, keys: &Tensor) -> Result<Tensor> {
    let columns = keys.dims()[1];
    if columns > MAX_ORDERED {
        return Err(Error::shape(
            owner,
            format!("the CPU formula orders rows of up to {MAX_ORDERED} columns, not {columns}"),
        ));
    }

    // One sort over `key · columns + (columns − 1 − column)`, which is distinct for every entry
    // of a row: the sort is left no tie to order in its own way.
    let column = Tensor::arange(0_u32, columns as u32, &Device::Cpu)?.to_dtype(DType::F64)?;
    let combined = keys
        .to_dtype(DType::F64)?
        .affine(columns as f64, (columns - 1) as f64)?
        .broadcast_sub(&column.unsqueeze(0)?)?;

    Ok(combined.arg_sort_last_dim(false)?)
}

/// The shaped logits of each row: its logit adjustments, then its penalties, then the division
/// by its temperature, FP32 `[rows, vocabulary]`.
///
/// ```text
/// x = logits + bias[slot]                         when the row's flag names its slot's bias row
/// x = -inf                                        at every token the row's ban mask sets
/// n = counts[slot] + the row's extra tokens       how often the row has seen each token
/// x = (x > 0 ? x / repetition : x · repetition) − frequency · n − presence
///                                                 where n > 0 and x is not -inf
/// shaped = x / (temperature > 0 ? temperature : 1)
/// ```
///
/// The order is vLLM's: the allowed set, the banned tokens and the logit bias before the
/// penalties, so the repetition penalty scales a biased logit, and the temperature last. A row
/// of temperature zero divides by one: its shaped logits keep the logits' order and its largest
/// one is its greedy token. Every operation is one FP32 rounding, and a row with no adjustment
/// and no seen token is shaped as its own logits over its temperature.
///
/// - `logits`: `[rows, vocabulary]` FP32, BF16 or FP16;
/// - `counts`: `[slots, vocabulary]` I32, how often each slot has seen each token;
/// - `slots`: `[rows]` U32, the slot each row reads; a row of [`NO_SLOT`] reads no counts and
///   no bias row;
/// - `parameters`: `[rows, PARAMETER_COLUMNS]` FP32 (module documentation);
/// - `extras`: `[rows, extras]` U32 tokens a row counts besides its slot's, once per
///   occurrence, [`NO_SLOT`] counting nothing; `None` for rows that count none;
/// - `bias`: `[slots, vocabulary]` FP32, each slot's additive row: a request's logit bias,
///   `-inf` outside its allowed set and zero elsewhere;
/// - `adjustments`: `[rows, 2]` U32, each row's bias flag (nonzero adds its slot's bias row)
///   and the row of `bans` that is its ban mask, [`NO_SLOT`] for none;
/// - `bans`: `[masks, ceil(vocabulary / 32)]` U32 ban masks, bit `t mod 32` of word `t / 32`
///   set for a banned token `t`.
///
/// # Errors
///
/// When a tensor is not the shape or element type stated above, or the operator refuses its
/// arguments.
#[expect(
    clippy::too_many_arguments,
    reason = "the operands of the shaping kernel, in its order"
)]
pub fn logits_shape(
    logits: &Tensor,
    counts: &Tensor,
    slots: &Tensor,
    parameters: &Tensor,
    extras: Option<&Tensor>,
    bias: &Tensor,
    adjustments: &Tensor,
    bans: &Tensor,
) -> Result<Tensor> {
    const OWNER: &str = LogitsShapeCall::OWNER;

    let (rows, width) = matrix(OWNER, "logits", logits)?;
    let vocabulary = extent(OWNER, "tokens", width)?;
    if !matches!(logits.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        return Err(Error::shape(
            OWNER,
            format!(
                "{} logits: f32, bf16 or f16 expected",
                ops::abi::name(logits.dtype())
            ),
        ));
    }

    let words = ban_words(vocabulary) as usize;
    let parameter_columns = PARAMETER_COLUMNS as usize;
    require(OWNER, "counts", counts, DType::I32, &[None, Some(width)])?;
    require(
        OWNER,
        "bias table",
        bias,
        DType::F32,
        &[Some(counts.dims()[0]), Some(width)],
    )?;
    require(OWNER, "slots", slots, DType::U32, &[Some(rows)])?;
    require(
        OWNER,
        "parameters",
        parameters,
        DType::F32,
        &[Some(rows), Some(parameter_columns)],
    )?;
    require(
        OWNER,
        "adjustments",
        adjustments,
        DType::U32,
        &[Some(rows), Some(2)],
    )?;
    require(OWNER, "ban masks", bans, DType::U32, &[None, Some(words)])?;
    if let Some(extras) = extras {
        require(
            OWNER,
            "extra tokens",
            extras,
            DType::U32,
            &[Some(rows), None],
        )?;
    }
    let extra_columns = extras.map_or(0, |extras| extras.dims()[1]);

    compute(LogitsShapeCall {
        logits,
        counts,
        slots,
        parameters,
        extras,
        bias,
        adjustments,
        bans,
        rows,
        width,
        vocabulary,
        extra_columns,
    })
}

/// A checked call of [`logits_shape`]: `rows` rows of a `vocabulary` (`width` columns) and
/// `extra_columns` extra tokens a row.
struct LogitsShapeCall<'t> {
    logits: &'t Tensor,
    counts: &'t Tensor,
    slots: &'t Tensor,
    parameters: &'t Tensor,
    extras: Option<&'t Tensor>,
    bias: &'t Tensor,
    adjustments: &'t Tensor,
    bans: &'t Tensor,
    rows: usize,
    width: usize,
    vocabulary: u32,
    extra_columns: usize,
}

impl LogitsShapeCall<'_> {
    const OWNER: &'static str = "logits_shape";
}

impl Fused for LogitsShapeCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.logits.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let Self {
            logits,
            counts,
            slots,
            parameters,
            extras,
            bias,
            adjustments,
            bans,
            rows,
            width,
            vocabulary,
            extra_columns,
        } = self;

        let x = logits.to_dtype(ACCUMULATE)?;
        let dropped = Tensor::full(f32::NEG_INFINITY, (rows, width), &Device::Cpu)?;
        let token = Tensor::arange(0_u32, vocabulary, &Device::Cpu)?;

        // A padding row names no slot: it reads row 0 of both tables and keeps neither.
        let live = named(slots)?;
        let read = live.where_cond(slots, &slots.zeros_like()?)?;
        let live = live.unsqueeze(1)?;

        // 1. The slot's bias row, for a row whose flag names it.
        let biased = adjustments.narrow(1, 0, 1)?.ne(0_u32)?.mul(&live)?;
        let x = choose(&biased, &(&x + bias.index_select(&read, 0)?)?, &x)?;

        // 2. The row's ban mask: bit `t mod 32` of word `t / 32`.
        let mask_index = adjustments.narrow(1, 1, 1)?.squeeze(1)?;
        let banning = named(&mask_index)?;
        let x = if bans.dims()[0] == 0 {
            x
        } else {
            let mask_row = banning.where_cond(&mask_index, &mask_index.zeros_like()?)?;
            let word = token.broadcast_div(&integer(32, &token)?)?;
            let bit = token.broadcast_sub(&word.broadcast_mul(&integer(32, &token)?)?)?;
            let set = bans
                .index_select(&mask_row, 0)?
                .index_select(&word, 1)?
                .broadcast_bitwise_right_shift(&bit.unsqueeze(0)?)?
                .broadcast_bitwise_and(&integer(1, &token)?)?
                .ne(0_u32)?;

            choose(&set.broadcast_mul(&banning.unsqueeze(1)?)?, &dropped, &x)?
        };

        // 3. How often the row has seen each token: its slot's counts and its extra tokens.
        let slot_counts = counts.index_select(&read, 0)?.to_dtype(ACCUMULATE)?;
        let mut seen = choose(&live, &slot_counts, &slot_counts.zeros_like()?)?;
        if let Some(extras) = extras {
            for index in 0..extra_columns {
                let counted = token
                    .unsqueeze(0)?
                    .broadcast_eq(&extras.narrow(1, index, 1)?)?;
                seen = (seen + counted.to_dtype(ACCUMULATE)?)?;
            }
        }

        // 4. The penalties, where the row has seen the token and its logit is not dropped.
        let parameter = |offset: usize| parameters.narrow(1, offset, 1);
        let repetition = parameter(REPETITION)?;
        let repeated = x.gt(0.0_f32)?.where_cond(
            &x.broadcast_div(&repetition)?,
            &x.broadcast_mul(&repetition)?,
        )?;
        let penalized = repeated
            .broadcast_sub(&seen.broadcast_mul(&parameter(FREQUENCY)?)?)?
            .broadcast_sub(&parameter(PRESENCE)?)?;
        let applies = seen.gt(0.0_f32)?.mul(&x.ne(f32::NEG_INFINITY)?)?;
        let x = applies.where_cond(&penalized, &x)?;

        // 5. The temperature, and one for a row that states none.
        let temperature = parameter(TEMPERATURE as usize)?;
        let divisor = temperature
            .gt(0.0_f32)?
            .where_cond(&temperature, &temperature.ones_like()?)?;

        Ok(x.broadcast_div(&divisor)?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self {
            logits,
            counts,
            slots,
            parameters,
            extras,
            bias,
            adjustments,
            bans,
            rows,
            width,
            vocabulary,
            extra_columns,
        } = self;

        let out = ops::empty((rows, width), DType::F32)?;
        ops::launch::<LogitsShape>(
            &ShapeDescriptor::new(
                logits.dtype(),
                vocabulary,
                extent(Self::OWNER, "extra tokens", extra_columns)?,
                true,
            ),
            ShapeArgs {
                logits: ops::operand(logits)?,
                counts: ops::operand(counts)?,
                slots: signed(slots)?,
                parameters: ops::operand(parameters)?,
                extras: extras.map(signed).transpose()?,
                bias: ops::operand(bias)?,
                adjustments: signed(adjustments)?,
                bans: signed(bans)?,
                out: ops::operand(&out)?,
            },
        )?;
        Ok(out)
    }
}

/// The `width` largest entries of each row of `logits`, `[rows, vocabulary]`: `(values, ids)`,
/// FP32 and U32 `[rows, width]`, descending, with equal entries in ascending token id.
///
/// The order is the kernel's, which orders the entries' bit patterns rather than comparing
/// them: `+0.0` is above `-0.0`. Every later step reads the result as an order, not a set:
/// the filter takes the largest at column 0 and a row's own top-k at column `top_k − 1`.
///
/// # Errors
///
/// When `logits` is not a matrix, `width` is not between one and the vocabulary, or the
/// operator refuses its arguments.
pub fn top_k(logits: &Tensor, width: u32) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = TopKCall::OWNER;

    let (rows, row_width) = matrix(OWNER, "logits", logits)?;
    let vocabulary = extent(OWNER, "tokens", row_width)?;
    if width == 0 || width > vocabulary {
        return Err(Error::shape(
            OWNER,
            format!("a width of {width} is outside 1..={vocabulary}"),
        ));
    }
    let kept = width as usize;

    compute(TopKCall {
        logits,
        width,
        rows,
        vocabulary,
        kept,
    })
}

/// A checked call of [`top_k`]: `rows` rows of a `vocabulary`, each keeping `kept`.
struct TopKCall<'t> {
    logits: &'t Tensor,
    width: u32,
    rows: usize,
    vocabulary: u32,
    kept: usize,
}

impl TopKCall<'_> {
    const OWNER: &'static str = "top_k";
}

impl Fused for TopKCall<'_> {
    type Output = (Tensor, Tensor);

    fn device(&self) -> &Device {
        self.logits.device()
    }

    fn cpu(self) -> Result<(Tensor, Tensor)> {
        let Self { logits, kept, .. } = self;

        let values = logits.to_dtype(ACCUMULATE)?.contiguous()?;
        let ids = descending(Self::OWNER, &monotone_keys(&values)?)?
            .narrow(1, 0, kept)?
            .contiguous()?;

        Ok((values.gather(&ids, 1)?, ids))
    }

    fn launch(self) -> Result<(Tensor, Tensor)> {
        let Self {
            logits,
            width,
            rows,
            vocabulary,
            kept,
        } = self;

        let values = ops::empty((rows, kept), DType::F32)?;
        let ids = ops::empty((rows, kept), DType::U32)?;

        // The ordering's scratch is the kernel's for the launch and nothing reads it afterwards.
        let scratch = top_k_workspace(rows as u64, u64::from(width));
        let workspace = ops::empty((rows, scratch[1] as usize), DType::I32)?;

        ops::launch::<TopK>(
            &TopKDescriptor::new(logits.dtype(), vocabulary, width),
            TopKArgs {
                logits: ops::operand(logits)?,
                values: ops::operand(&values)?,
                ids: signed(&ids)?,
                workspace: ops::operand(&workspace)?,
            },
        )?;
        Ok((values, ids))
    }
}

/// Applies each row's own thresholds to its candidates, in place: `(values, valid)`, the
/// candidates with every dropped one `-inf`, and `[rows]` I32 that is `1` for a row whose
/// candidates form a usable distribution.
///
/// ```text
/// limit    = top_k > 0 ? floor(min(top_k, width)) : width
/// x        = column < limit ? values : -inf                       the row's top-k width
/// p        = softmax(x);  before = Σ p of the columns before
/// x        = before > top_p ? -inf : x                            the nucleus
/// x        = min_p ≤ 0 or x ≥ max(x) + ln(min_p) ? x : -inf       min-p
/// valid    = no x is NaN or +inf, and some x is finite
/// ```
///
/// `values` is FP32 `[rows, width]` in descending order ([`top_k`]'s), `ids` its U32 token
/// ids, which the kernel takes and does not read, and `parameters` `[rows,
/// PARAMETER_COLUMNS]` FP32. The nucleus keeps the first candidate whose cumulative mass
/// crosses `top_p`, so a row keeps its largest candidate under any threshold of zero or more;
/// min-p is measured against the largest candidate the nucleus left. A row of a min-p above
/// one, or of logits that are all `-inf`, keeps nothing and reports `0`: the function does
/// not branch on that, its reader does.
///
/// # Errors
///
/// When a tensor is not the shape or element type stated above, `values` is a strided view
/// (it is filtered in its own storage), or the operator refuses its arguments.
pub fn filter_candidates_(
    values: Tensor,
    ids: &Tensor,
    parameters: &Tensor,
) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = FilterCandidatesCall::OWNER;

    let (rows, width) = matrix(OWNER, "candidates", &values)?;
    let parameter_columns = PARAMETER_COLUMNS as usize;
    require(
        OWNER,
        "candidates",
        &values,
        DType::F32,
        &[Some(rows), Some(width)],
    )?;
    require(
        OWNER,
        "candidate ids",
        ids,
        DType::U32,
        &[Some(rows), Some(width)],
    )?;
    require(
        OWNER,
        "parameters",
        parameters,
        DType::F32,
        &[Some(rows), Some(parameter_columns)],
    )?;
    if !values.is_contiguous() {
        return Err(Error::shape(
            OWNER,
            "strided candidates have no storage of their own to be filtered in".to_owned(),
        ));
    }

    compute(FilterCandidatesCall {
        values,
        ids,
        parameters,
        rows,
        width,
    })
}

/// A checked call of [`filter_candidates_`]: `rows` rows of `width` candidates.
struct FilterCandidatesCall<'t> {
    values: Tensor,
    ids: &'t Tensor,
    parameters: &'t Tensor,
    rows: usize,
    width: usize,
}

impl FilterCandidatesCall<'_> {
    const OWNER: &'static str = "filter_candidates_";
}

impl Fused for FilterCandidatesCall<'_> {
    type Output = (Tensor, Tensor);

    fn device(&self) -> &Device {
        self.values.device()
    }

    fn cpu(self) -> Result<(Tensor, Tensor)> {
        let Self {
            values,
            parameters,
            rows,
            width,
            ..
        } = self;

        let dropped = Tensor::full(f32::NEG_INFINITY, (rows, width), &Device::Cpu)?;
        let column = Tensor::arange(0_u32, width as u32, &Device::Cpu)?.to_dtype(ACCUMULATE)?;
        let parameter = |offset: usize| parameters.narrow(1, offset, 1);
        let (top_k, top_p, min_p) = (
            parameter(TOP_K as usize)?,
            parameter(TOP_P as usize)?,
            parameter(MIN_P)?,
        );

        // 1. The row's top-k width: a width that is not positive keeps every candidate.
        let whole = Tensor::full(width as f32, (rows, 1), &Device::Cpu)?;
        let limit = top_k
            .gt(0.0_f32)?
            .where_cond(&top_k.minimum(width as f32)?.floor()?, &whole)?;
        let kept = column
            .unsqueeze(0)?
            .broadcast_lt(&limit)?
            .where_cond(&values, &dropped)?;

        // 2. The nucleus: a candidate is dropped when the mass before it is above `top_p`.
        let terms = kept.broadcast_sub(&kept.max_keepdim(1)?)?.exp()?;
        let probabilities = terms.broadcast_div(&terms.sum_keepdim(1)?)?;
        let before = (cumulative(&probabilities)? - &probabilities)?;
        let kept = before.broadcast_gt(&top_p)?.where_cond(&dropped, &kept)?;

        // 3. Min-p, against the largest candidate the nucleus left.
        let threshold = (kept.max_keepdim(1)? + min_p.log()?)?;
        let passes = kept
            .broadcast_ge(&threshold)?
            .broadcast_maximum(&min_p.le(0.0_f32)?)?;
        let kept = passes.where_cond(&kept, &dropped)?;

        // 4. Whether what is left is a distribution.
        let count =
            |mask: Tensor| -> Result<Tensor> { Ok(mask.to_dtype(ACCUMULATE)?.sum_keepdim(1)?) };
        let not_a_number = kept.ne(&kept)?;
        let infinite = kept.eq(f32::INFINITY)?;
        let finite = kept
            .eq(&kept)?
            .mul(&kept.ne(f32::INFINITY)?)?
            .mul(&kept.ne(f32::NEG_INFINITY)?)?;
        let valid = count(not_a_number)?
            .eq(0.0_f32)?
            .mul(&count(infinite)?.eq(0.0_f32)?)?
            .mul(&count(finite)?.gt(0.0_f32)?)?
            .to_dtype(DType::I32)?
            .squeeze(1)?;

        values.slice_set(&kept, 0, 0)?;
        Ok((values, valid))
    }

    fn launch(self) -> Result<(Tensor, Tensor)> {
        let Self {
            values,
            ids,
            parameters,
            rows,
            width,
        } = self;

        let valid = ops::empty(rows, DType::I32)?;
        ops::launch::<FilterCandidates>(
            &FilterDescriptor::new(extent(Self::OWNER, "candidates", width)?),
            FilterArgs {
                values: ops::operand(&values)?,
                ids: signed(ids)?,
                parameters: ops::operand(parameters)?,
                valid: ops::operand(&valid)?,
            },
        )?;
        Ok((values, valid))
    }
}

/// Each row's candidates in ascending token id: `(values, ids)` in the shapes and element types
/// of the candidates given.
///
/// Column `k` of a row is the candidate with the row's `k`-th smallest id, so the first
/// columns hold the candidates whose id is a token of the `vocabulary` and the columns after
/// them hold `-inf` and [`NO_SLOT`], which carry no mass and name no token. The permutation is
/// by id alone: a dropped candidate keeps its place among the ids with its `-inf`. A row holds
/// each token once; where it does not, the equal ids take consecutive columns in an order the
/// kernel does not fix.
///
/// `values` is FP32 `[rows, width]` and `ids` U32 `[rows, width]`.
///
/// # Errors
///
/// When the two are not candidates of one shape, the vocabulary is narrower than a row, or the
/// operator refuses its arguments.
pub fn candidate_order(values: &Tensor, ids: &Tensor, vocabulary: u32) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = CandidateOrderCall::OWNER;

    let (rows, width) = matrix(OWNER, "candidates", values)?;
    require(
        OWNER,
        "candidates",
        values,
        DType::F32,
        &[Some(rows), Some(width)],
    )?;
    require(
        OWNER,
        "candidate ids",
        ids,
        DType::U32,
        &[Some(rows), Some(width)],
    )?;
    if (vocabulary as usize) < width {
        return Err(Error::shape(
            OWNER,
            format!("rows of {width} candidates of a vocabulary of {vocabulary}"),
        ));
    }

    compute(CandidateOrderCall {
        values,
        ids,
        vocabulary,
        rows,
        width,
    })
}

/// A checked call of [`candidate_order`]: `rows` rows of `width` candidates.
struct CandidateOrderCall<'t> {
    values: &'t Tensor,
    ids: &'t Tensor,
    vocabulary: u32,
    rows: usize,
    width: usize,
}

impl CandidateOrderCall<'_> {
    const OWNER: &'static str = "candidate_order";
}

impl Fused for CandidateOrderCall<'_> {
    type Output = (Tensor, Tensor);

    fn device(&self) -> &Device {
        self.values.device()
    }

    fn cpu(self) -> Result<(Tensor, Tensor)> {
        let Self {
            values,
            ids,
            vocabulary,
            rows,
            width,
        } = self;

        let unnamed = Tensor::full(NO_SLOT, (rows, width), &Device::Cpu)?;
        let dropped = Tensor::full(f32::NEG_INFINITY, (rows, width), &Device::Cpu)?;

        // An id that is no token sorts past every token and leaves its column unnamed.
        let token = ids.broadcast_lt(&integer(vocabulary, ids)?)?;
        let order = token.where_cond(ids, &unnamed)?.arg_sort_last_dim(true)?;
        let placed = token.gather(&order, 1)?;

        Ok((
            placed.where_cond(&values.gather(&order, 1)?, &dropped)?,
            placed.where_cond(&ids.gather(&order, 1)?, &unnamed)?,
        ))
    }

    fn launch(self) -> Result<(Tensor, Tensor)> {
        let Self {
            values,
            ids,
            vocabulary,
            rows,
            width,
        } = self;

        let ordered_values = ops::empty((rows, width), DType::F32)?;
        let ordered_ids = ops::empty((rows, width), DType::U32)?;

        // The kernel counts the candidates of each id rather than comparing them, in scratch as
        // wide as the id space, which is its own for the launch.
        let counts = ops::empty((rows, vocabulary as usize), DType::I32)?;

        ops::launch::<CandidateOrder>(
            &OrderDescriptor::new(),
            OrderArgs {
                values: ops::operand(values)?,
                ids: signed(ids)?,
                ordered_values: ops::operand(&ordered_values)?,
                ordered_ids: signed(&ordered_ids)?,
                counts: ops::operand(&counts)?,
                vocabulary: u64::from(vocabulary),
            },
        )?;
        Ok((ordered_values, ordered_ids))
    }
}

/// The softmax of the rows of `x`: `exp(x − max x) / Σ exp(x − max x)`, in `x`'s shape and
/// element type.
///
/// The maximum, the exponentials and their sum are FP32 and each result is rounded once. An
/// entry of `-inf` has no mass; a row with no finite entry has no distribution and is NaN. The
/// kernel reads a row in chunks, so a row may be as wide as a vocabulary.
///
/// # Errors
///
/// When `x` has no rows, or the operator refuses its arguments.
pub fn softmax(x: &Tensor) -> Result<Tensor> {
    const OWNER: &str = SoftmaxCall::OWNER;

    let width = super::columns(OWNER, x)?;

    compute(SoftmaxCall { x, width })
}

/// A checked call of [`softmax`]: rows of `width` columns.
struct SoftmaxCall<'t> {
    x: &'t Tensor,
    width: u32,
}

impl SoftmaxCall<'_> {
    const OWNER: &'static str = "softmax";
}

impl Fused for SoftmaxCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.x.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let Self { x, .. } = self;

        let rows = x.to_dtype(ACCUMULATE)?;
        let terms = rows.broadcast_sub(&rows.max_keepdim(D::Minus1)?)?.exp()?;

        Ok(terms
            .broadcast_div(&terms.sum_keepdim(D::Minus1)?)?
            .to_dtype(x.dtype())?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self { x, width } = self;

        let rows = as_rows(Self::OWNER, x)?;
        let out = ops::empty(rows.shape(), x.dtype())?;

        ops::launch::<Softmax>(
            &SoftmaxDescriptor::new(x.dtype(), width),
            SoftmaxArgs {
                x: ops::operand(&rows)?,
                out: ops::operand(&out)?,
            },
        )?;
        like_rows(out, x)
    }
}

/// One token per row, drawn from its candidates by inverting their cumulative distribution:
/// `[rows]` U32.
///
/// ```text
/// token = ids[first column c with p[0] + … + p[c] > draw · Σ p]
///       = the largest id that holds mass, when no column crosses
/// ```
///
/// `probabilities` is FP32 `[rows, width]`, `ids` the candidates' U32 token ids and `draws`
/// `[rows]` FP32 uniform values in `[0, 1)`. The cumulative runs in the order given, which a
/// caller makes ascending token id ([`candidate_order`]). A candidate without mass is never
/// selected, so a draw of zero selects the first candidate that holds some. A row whose
/// candidates hold no mass at all, or a NaN, selects token `0`: it has no distribution, which
/// [`filter_candidates_`] reports.
///
/// # Errors
///
/// When a tensor is not the shape or element type stated above, or the operator refuses its
/// arguments.
pub fn inverse_cdf(probabilities: &Tensor, ids: &Tensor, draws: &Tensor) -> Result<Tensor> {
    const OWNER: &str = InverseCdfCall::OWNER;

    let (rows, width) = matrix(OWNER, "probabilities", probabilities)?;
    require(
        OWNER,
        "probabilities",
        probabilities,
        DType::F32,
        &[Some(rows), Some(width)],
    )?;
    require(
        OWNER,
        "candidate ids",
        ids,
        DType::U32,
        &[Some(rows), Some(width)],
    )?;
    require(OWNER, "draws", draws, DType::F32, &[Some(rows)])?;

    compute(InverseCdfCall {
        probabilities,
        ids,
        draws,
        rows,
        width,
    })
}

/// A checked call of [`inverse_cdf`]: `rows` rows of `width` candidates.
struct InverseCdfCall<'t> {
    probabilities: &'t Tensor,
    ids: &'t Tensor,
    draws: &'t Tensor,
    rows: usize,
    width: usize,
}

impl InverseCdfCall<'_> {
    const OWNER: &'static str = "inverse_cdf";
}

impl Fused for InverseCdfCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.probabilities.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let Self {
            probabilities,
            ids,
            draws,
            rows,
            width,
        } = self;

        let target = draws
            .unsqueeze(1)?
            .broadcast_mul(&probabilities.sum_keepdim(1)?)?;
        let crossed = cumulative(probabilities)?.broadcast_gt(&target)?;

        // The first column that crosses, or the width when none does.
        let column = Tensor::arange(0_u32, width as u32, &Device::Cpu)?;
        let past = Tensor::full(width as u32, (rows, width), &Device::Cpu)?;
        let first = choose(&crossed, &column.unsqueeze(0)?, &past)?.min_keepdim(1)?;
        let found = first.lt(width as u32)?;
        let drawn = ids.gather(&found.where_cond(&first, &first.zeros_like()?)?, 1)?;

        let fallback = probabilities
            .gt(0.0_f32)?
            .where_cond(ids, &ids.zeros_like()?)?
            .max_keepdim(1)?;

        Ok(found.where_cond(&drawn, &fallback)?.squeeze(1)?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self {
            probabilities,
            ids,
            draws,
            rows,
            width,
        } = self;

        let tokens = ops::empty(rows, DType::U32)?;
        ops::launch::<InverseCdf>(
            &InverseCdfDescriptor::new(extent(Self::OWNER, "candidates", width)?),
            InverseCdfArgs {
                probabilities: ops::operand(probabilities)?,
                ids: signed(ids)?,
                draws: ops::operand(draws)?,
                tokens: signed(&tokens)?,
            },
        )?;
        Ok(tokens)
    }
}

/// The column of each row's largest entry, the lowest column among equal entries: `[rows]`
/// U32. Over a vocabulary-wide row the column is the token id, so this is a row's greedy token.
///
/// A row with no numeric entry, all `-inf` or NaN, returns its width, which is one past its
/// last column and names no token: a reader of the result refuses `token ≥ width`.
///
/// `values` is `[rows, width]` in a floating-point type.
///
/// # Errors
///
/// When `values` is not a matrix, or the operator refuses its arguments.
pub fn argmax(values: &Tensor) -> Result<Tensor> {
    const OWNER: &str = ArgmaxCall::OWNER;

    let (rows, row_width) = matrix(OWNER, "values", values)?;
    let width = extent(OWNER, "columns", row_width)?;

    compute(ArgmaxCall {
        values,
        rows,
        row_width,
        width,
    })
}

/// A checked call of [`argmax`]: `rows` rows of `row_width` columns (`width` as a launch states
/// it).
struct ArgmaxCall<'t> {
    values: &'t Tensor,
    rows: usize,
    row_width: usize,
    width: u32,
}

impl ArgmaxCall<'_> {
    const OWNER: &'static str = "argmax";
}

impl Fused for ArgmaxCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.values.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let Self {
            values,
            rows,
            row_width,
            width,
        } = self;

        let x = values.to_dtype(ACCUMULATE)?;
        let largest = x.max_keepdim(1)?;

        let column = Tensor::arange(0_u32, width, &Device::Cpu)?;
        let past = Tensor::full(width, (rows, row_width), &Device::Cpu)?;
        let first =
            choose(&x.broadcast_eq(&largest)?, &column.unsqueeze(0)?, &past)?.min_keepdim(1)?;

        let numeric = largest.gt(f32::NEG_INFINITY)?;
        let none = Tensor::full(width, (rows, 1), &Device::Cpu)?;
        Ok(numeric.where_cond(&first, &none)?.squeeze(1)?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self {
            values,
            rows,
            width,
            ..
        } = self;

        let out = ops::empty(rows, DType::U32)?;
        ops::launch::<ArgMax>(
            &ArgMaxDescriptor::new(values.dtype(), width),
            ArgMaxArgs {
                values: ops::operand(values)?,
                out: signed(&out)?,
            },
        )?;
        Ok(out)
    }
}

/// The log-probabilities of each row of `logits`: `x − (max x + ln Σ exp(x − max x))`, FP32
/// `[rows, columns]` whatever the logits' element type.
///
/// This is the distribution a token's reported log-probability is read from: over a model's
/// logits as they are, it is vLLM's `raw_logprobs`.
///
/// # Errors
///
/// When `logits` is not a matrix, or the operator refuses its arguments.
pub fn log_softmax(logits: &Tensor) -> Result<Tensor> {
    const OWNER: &str = LogSoftmaxCall::OWNER;

    let (rows, width) = matrix(OWNER, "logits", logits)?;
    let columns = extent(OWNER, "columns", width)?;

    compute(LogSoftmaxCall {
        logits,
        rows,
        width,
        columns,
    })
}

/// A checked call of [`log_softmax`]: `rows` rows of `width` columns (`columns` as a launch
/// states it).
struct LogSoftmaxCall<'t> {
    logits: &'t Tensor,
    rows: usize,
    width: usize,
    columns: u32,
}

impl LogSoftmaxCall<'_> {
    const OWNER: &'static str = "log_softmax";
}

impl Fused for LogSoftmaxCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.logits.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let Self { logits, .. } = self;

        let x = logits.to_dtype(ACCUMULATE)?;
        let largest = x.max_keepdim(1)?;
        let total = x.broadcast_sub(&largest)?.exp()?.sum_keepdim(1)?;

        Ok(x.broadcast_sub(&(largest + total.log()?)?)?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self {
            logits,
            rows,
            width,
            columns,
        } = self;

        let out = ops::empty((rows, width), DType::F32)?;
        ops::launch::<LogSoftmax>(
            &LogSoftmaxDescriptor::new(logits.dtype(), columns),
            LogSoftmaxArgs {
                x: ops::operand(logits)?,
                out: ops::operand(&out)?,
            },
        )?;
        Ok(out)
    }
}

/// One token's log-probability and rank per row: `(logprobs, ranks)`, `[rows]` FP32 and
/// `[rows]` I32.
///
/// ```text
/// logprob = logprobs[row, token]
/// rank    = 1 + the entries of the row strictly above logprob
/// ```
///
/// The rank is the competition rank vLLM reports: equal entries share one. `logprobs` is FP32
/// `[rows, columns]` and `tokens` `[rows]` U32; a row whose token is outside the row, as
/// [`NO_SLOT`] is, reports `0.0` and rank `0`, which no scored token has.
///
/// # Errors
///
/// When a tensor is not the shape or element type stated above, or the operator refuses its
/// arguments.
pub fn logprob_gather(logprobs: &Tensor, tokens: &Tensor) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = LogprobGatherCall::OWNER;

    let (rows, width) = matrix(OWNER, "log-probabilities", logprobs)?;
    let columns = extent(OWNER, "columns", width)?;
    require(
        OWNER,
        "log-probabilities",
        logprobs,
        DType::F32,
        &[Some(rows), Some(width)],
    )?;
    require(OWNER, "tokens", tokens, DType::U32, &[Some(rows)])?;

    compute(LogprobGatherCall {
        logprobs,
        tokens,
        rows,
        columns,
    })
}

/// A checked call of [`logprob_gather`]: `rows` rows of `columns` log-probabilities.
struct LogprobGatherCall<'t> {
    logprobs: &'t Tensor,
    tokens: &'t Tensor,
    rows: usize,
    columns: u32,
}

impl LogprobGatherCall<'_> {
    const OWNER: &'static str = "logprob_gather";
}

impl Fused for LogprobGatherCall<'_> {
    type Output = (Tensor, Tensor);

    fn device(&self) -> &Device {
        self.logprobs.device()
    }

    fn cpu(self) -> Result<(Tensor, Tensor)> {
        let Self {
            logprobs,
            tokens,
            columns,
            ..
        } = self;

        let inside = tokens.broadcast_lt(&integer(columns, tokens)?)?;
        let read = inside.where_cond(tokens, &tokens.zeros_like()?)?;
        let inside = inside.unsqueeze(1)?;

        let gathered = logprobs.gather(&read.unsqueeze(1)?, 1)?;
        let value = inside.where_cond(&gathered, &gathered.zeros_like()?)?;
        let above = logprobs
            .broadcast_gt(&value)?
            .to_dtype(DType::I32)?
            .sum_keepdim(1)?;
        let one = Tensor::new(1_i32, &Device::Cpu)?;
        let rank = inside.where_cond(&above.broadcast_add(&one)?, &above.zeros_like()?)?;

        Ok((value.squeeze(1)?, rank.squeeze(1)?))
    }

    fn launch(self) -> Result<(Tensor, Tensor)> {
        let Self {
            logprobs,
            tokens,
            rows,
            columns,
        } = self;

        let out = ops::empty(rows, DType::F32)?;
        let ranks = ops::empty(rows, DType::I32)?;

        ops::launch::<LogprobGather>(
            &LogprobDescriptor::new(columns),
            LogprobArgs {
                logprobabilities: ops::operand(logprobs)?,
                tokens: signed(tokens)?,
                out: ops::operand(&out)?,
                ranks: ops::operand(&ranks)?,
            },
        )?;
        Ok((out, ranks))
    }
}

/// The `candidates` largest entries of each row of `logprobs`, FP32 `[rows, columns]`:
/// `(values, ids)`, FP32 and U32 `[rows, candidates]`, descending, with equal entries in
/// ascending token id.
///
/// A row narrower than `candidates` fills its remaining columns with `-inf` and [`NO_SLOT`].
///
/// On the device this is two launches of one operator: each chunk of 4096 columns keeps its
/// own `candidates` largest, the chunks of a row in parallel, and a second selection keeps the
/// row's among the chunks'. One kernel of that second selection holds every chunk's
/// candidates of a row, which bounds `candidates` by the vocabulary: 215 over 151936 tokens.
///
/// # Errors
///
/// When `logprobs` is not an FP32 matrix, `candidates` is zero or more than the second
/// selection holds for rows of this width, or the operator refuses its arguments.
pub fn top_logprobs(logprobs: &Tensor, candidates: u32) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = TopLogprobsCall::OWNER;

    let (rows, width) = matrix(OWNER, "log-probabilities", logprobs)?;
    require(
        OWNER,
        "log-probabilities",
        logprobs,
        DType::F32,
        &[Some(rows), Some(width)],
    )?;

    // The two selections, and what the first keeps of a row, which the second reads whole.
    let (chunks, merge) = TopLogprobsDescriptor::selections(width as u64, candidates)?;
    let chunk_candidates = chunks.written(width as u64);
    let kept = candidates as usize;

    compute(TopLogprobsCall {
        logprobs,
        rows,
        width,
        chunks,
        merge,
        chunk_candidates,
        kept,
    })
}

/// A checked call of [`top_logprobs`]: `rows` rows of `width` log-probabilities, each keeping
/// `kept`, by the two selections `chunks` and `merge`, the first of which keeps
/// `chunk_candidates` of a row.
struct TopLogprobsCall<'t> {
    logprobs: &'t Tensor,
    rows: usize,
    width: usize,
    chunks: TopLogprobsDescriptor,
    merge: TopLogprobsDescriptor,
    chunk_candidates: u64,
    kept: usize,
}

impl TopLogprobsCall<'_> {
    const OWNER: &'static str = "top_logprobs";
}

impl Fused for TopLogprobsCall<'_> {
    type Output = (Tensor, Tensor);

    fn device(&self) -> &Device {
        self.logprobs.device()
    }

    fn cpu(self) -> Result<(Tensor, Tensor)> {
        let Self {
            logprobs,
            rows,
            width,
            kept,
            ..
        } = self;

        let values = logprobs.contiguous()?;

        // The kernel compares values, so the two zeros are one value and tie by id.
        let compared = values
            .eq(0.0_f32)?
            .where_cond(&values.zeros_like()?, &values)?;
        let present = kept.min(width);
        let ids = descending(Self::OWNER, &monotone_keys(&compared)?)?
            .narrow(1, 0, present)?
            .contiguous()?;
        let values = values.gather(&ids, 1)?;
        if present == kept {
            return Ok((values, ids));
        }

        let absent = (rows, kept - present);
        Ok((
            Tensor::cat(
                &[
                    values,
                    Tensor::full(f32::NEG_INFINITY, absent, &Device::Cpu)?,
                ],
                1,
            )?,
            Tensor::cat(&[ids, Tensor::full(NO_SLOT, absent, &Device::Cpu)?], 1)?,
        ))
    }

    fn launch(self) -> Result<(Tensor, Tensor)> {
        let Self {
            logprobs,
            rows,
            chunks,
            merge,
            chunk_candidates,
            kept,
            ..
        } = self;

        let chunk_values = ops::empty((rows, chunk_candidates as usize), DType::F32)?;
        let chunk_ids = ops::empty((rows, chunk_candidates as usize), DType::U32)?;
        ops::launch::<TopLogprobs>(
            &chunks,
            TopLogprobsArgs {
                values: ops::operand(logprobs)?,
                ids: None,
                out_values: ops::operand(&chunk_values)?,
                out_ids: signed(&chunk_ids)?,
            },
        )?;

        let values = ops::empty((rows, kept), DType::F32)?;
        let ids = ops::empty((rows, kept), DType::U32)?;

        ops::launch::<TopLogprobs>(
            // The second selection reads each candidate's id beside it, in one chunk of the chunks'
            // candidates.
            &merge,
            TopLogprobsArgs {
                values: ops::operand(&chunk_values)?,
                ids: Some(signed(&chunk_ids)?),
                out_values: ops::operand(&values)?,
                out_ids: signed(&ids)?,
            },
        )?;
        Ok((values, ids))
    }
}

#[cfg(test)]
mod tests {
    use tensor::WithDType;
    use tensor::shape::ShapeWithOneHole;

    use super::*;

    /// The CPU tensor of `values` in `shape`.
    fn tensor<T: WithDType>(values: &[T], shape: impl ShapeWithOneHole) -> Tensor {
        Tensor::from_slice(values, shape, &Device::Cpu).unwrap()
    }

    /// The elements of `tensor`, in row-major order.
    fn read<T: WithDType>(tensor: &Tensor) -> Vec<T> {
        tensor.flatten_all().unwrap().to_vec1().unwrap()
    }

    /// The bit patterns of `values`, which compare `-inf` and the two zeros as themselves.
    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// One row of request parameters in the operators' column order.
    fn parameter_row(
        temperature: f32,
        top_k: f32,
        top_p: f32,
        min_p: f32,
        penalties: [f32; 3],
    ) -> [f32; PARAMETER_COLUMNS as usize] {
        let [repetition, frequency, presence] = penalties;
        [
            temperature,
            top_k,
            top_p,
            min_p,
            repetition,
            frequency,
            presence,
            0.0,
        ]
    }

    /// A row's probabilities, the softmax of its values, in FP64.
    fn probabilities(values: &[f32]) -> Vec<f64> {
        let largest = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let terms: Vec<f64> = values
            .iter()
            .map(|value| f64::from(*value - largest).exp())
            .collect();
        let total: f64 = terms.iter().sum();

        terms.iter().map(|term| term / total).collect()
    }

    /// A meta tensor of `shape` elements of `dtype`: an input of a profile run.
    fn meta(shape: impl Into<tensor::Shape>, dtype: DType) -> Tensor {
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT: AtomicU64 = AtomicU64::new(1 << 40);
        let address = NEXT.fetch_add(1 << 28, Ordering::Relaxed);

        ops::retained(address, shape, dtype, &ops::meta()).unwrap()
    }

    /// Runs `forward` under a meta context while it records, and returns what it returned with
    /// the launch table as `(operator, key)` in launch order.
    fn recorded<T>(
        forward: impl FnOnce() -> Result<T>,
    ) -> (Result<T>, Vec<(&'static str, String)>) {
        let context = ops::Context::meta(None);
        context.record_launches();

        let output = {
            let _guard = context.enter();
            forward()
        };
        let launches = context
            .take_launches()
            .into_iter()
            .map(|(operator, key, ..)| (operator, key))
            .collect();
        (output, launches)
    }

    /// `launches` as the `(operator, key)` pairs a test states.
    fn stated<'a>(launches: &'a [(&'static str, String)]) -> Vec<(&'static str, &'a str)> {
        launches
            .iter()
            .map(|(operator, key)| (*operator, key.as_str()))
            .collect()
    }

    // ---- logits_shape -------------------------------------------------------------------------

    /// The shaping formula of one row, written out: the bias row, the ban mask, the penalties
    /// over `seen`, and the temperature.
    fn shaped_row(
        logits: &[f32],
        bias: Option<&[f32]>,
        banned: &[usize],
        seen: &[f32],
        parameters: &[f32],
    ) -> Vec<f32> {
        let (temperature, repetition) = (parameters[0], parameters[REPETITION]);
        let (frequency, presence) = (parameters[FREQUENCY], parameters[PRESENCE]);

        (0..logits.len())
            .map(|token| {
                let mut x = logits[token];
                if let Some(bias) = bias {
                    x += bias[token];
                }
                if banned.contains(&token) {
                    x = f32::NEG_INFINITY;
                }
                if seen[token] > 0.0 && x != f32::NEG_INFINITY {
                    let repeated = if x > 0.0 {
                        x / repetition
                    } else {
                        x * repetition
                    };
                    x = repeated - frequency * seen[token] - presence;
                }
                x / if temperature > 0.0 { temperature } else { 1.0 }
            })
            .collect()
    }

    #[test]
    fn logits_shape_adjusts_penalizes_and_scales_each_row() {
        const VOCABULARY: usize = 40;

        let row = |seed: usize| -> Vec<f32> {
            (0..VOCABULARY)
                .map(|token| ((token * 7 + seed * 3) % 11) as f32 * 0.5 - 2.0)
                .collect()
        };
        let logits: Vec<f32> = (0..4).flat_map(row).collect();

        // Slot 0 has seen nothing; slot 1 has seen every token `t mod 3` times.
        let seen: Vec<i32> = (0..VOCABULARY).map(|token| (token % 3) as i32).collect();
        let counts: Vec<i32> = [vec![0; VOCABULARY], seen.clone()].concat();
        let mut slot_bias: Vec<f32> = (0..VOCABULARY).map(|t| 0.25 * (t % 4) as f32).collect();
        slot_bias[5] = f32::NEG_INFINITY;
        slot_bias[37] = f32::NEG_INFINITY;
        let bias: Vec<f32> = [vec![0.0; VOCABULARY], slot_bias.clone()].concat();

        // One mask banning tokens 0, 33 and 39: bit 0 of word 0, bits 1 and 7 of word 1.
        let bans = [1_u32, (1 << 1) | (1 << 7)];

        let parameters = [
            // Row 0: no adjustment, no penalty, no temperature.
            parameter_row(0.0, 0.0, 1.0, 0.0, [1.0, 0.0, 0.0]),
            // Row 1: slot 1's bias row, the ban mask, every penalty and a temperature.
            parameter_row(0.7, 0.0, 1.0, 0.0, [1.3, 0.5, 0.25]),
            // Row 2: a padding row, whose flag and penalties read no slot.
            parameter_row(2.0, 0.0, 1.0, 0.0, [2.0, 1.0, 1.0]),
            // Row 3: slot 1's counts under the repetition penalty alone.
            parameter_row(1.0, 0.0, 1.0, 0.0, [1.5, 0.0, 0.0]),
        ];
        let adjustments = [0, NO_SLOT, 1, 0, 1, NO_SLOT, 0, NO_SLOT];

        let shaped = logits_shape(
            &tensor(&logits, (4, VOCABULARY)),
            &tensor(&counts, (2, VOCABULARY)),
            &tensor(&[0, 1, NO_SLOT, 1], 4),
            &tensor(&parameters.concat(), (4, PARAMETER_COLUMNS as usize)),
            None,
            &tensor(&bias, (2, VOCABULARY)),
            &tensor(&adjustments, (4, 2)),
            &tensor(&bans, (1, 2)),
        )
        .unwrap();

        let unseen = vec![0.0; VOCABULARY];
        let seen: Vec<f32> = seen.iter().map(|count| *count as f32).collect();
        let expected = [
            shaped_row(&row(0), None, &[], &unseen, &parameters[0]),
            shaped_row(
                &row(1),
                Some(&slot_bias),
                &[0, 33, 39],
                &seen,
                &parameters[1],
            ),
            shaped_row(&row(2), None, &[], &unseen, &parameters[2]),
            shaped_row(&row(3), None, &[], &seen, &parameters[3]),
        ]
        .concat();

        // Every step is one FP32 rounding in the formula's order, so the result is exact.
        assert_eq!(bits(&read::<f32>(&shaped)), bits(&expected));

        // A row with no adjustment, no seen token and no temperature is its own logits.
        assert_eq!(read::<f32>(&shaped)[..VOCABULARY], row(0)[..]);
    }

    #[test]
    fn logits_shape_counts_a_rows_extra_tokens_once_per_occurrence() {
        let logits = [1.0_f32, 2.0, -1.0, 0.5];
        let parameters = parameter_row(1.0, 0.0, 1.0, 0.0, [2.0, 0.5, 0.25]);

        let shaped = logits_shape(
            &tensor(&logits, (1, 4)),
            &tensor(&[0_i32, 0, 1, 0], (1, 4)),
            &tensor(&[0_u32], 1),
            &tensor(&parameters, (1, PARAMETER_COLUMNS as usize)),
            Some(&tensor(&[1_u32, 1, NO_SLOT], (1, 3))),
            &tensor(&[0.0_f32; 4], (1, 4)),
            &tensor(&[0, NO_SLOT], (1, 2)),
            &tensor(&[0_u32], (1, 1)),
        )
        .unwrap();

        // Token 1 is counted twice by the extras and token 2 once by the slot: 2 / 2 less two
        // frequency penalties and the presence penalty, and -1 · 2 less one of each.
        let expected = shaped_row(&logits, None, &[], &[0.0, 2.0, 1.0, 0.0], &parameters);
        assert_eq!(read::<f32>(&shaped), expected);
        assert_eq!(expected, [1.0, -0.25, -2.75, 0.5]);
    }

    #[test]
    fn logits_shape_refuses_tables_that_are_not_one_row_per_slot() {
        let refused = logits_shape(
            &tensor(&[0.0_f32; 8], (2, 4)),
            &tensor(&[0_i32; 12], (3, 4)),
            &tensor(&[0_u32, 1], 2),
            &tensor(&[0.0_f32; 16], (2, PARAMETER_COLUMNS as usize)),
            None,
            &tensor(&[0.0_f32; 8], (2, 4)),
            &tensor(&[0_u32; 4], (2, 2)),
            &tensor(&[0_u32; 2], (2, 1)),
        );

        let text = refused.unwrap_err().to_string();
        assert!(
            text.contains("bias table") && text.contains("[3, 4]"),
            "{text}"
        );
    }

    // ---- top_k --------------------------------------------------------------------------------

    #[test]
    fn top_k_keeps_the_largest_in_descending_order_with_ties_by_token() {
        let rows = [
            [1.0_f32, 3.0, -0.0, 3.0, 0.0, -2.0, 3.0, 1.0],
            [-0.0, 0.0, -1.0, f32::NEG_INFINITY, -1.0, 0.0, -0.0, -5.0],
        ];
        let (values, ids) = top_k(&tensor(&rows.concat(), (2, 8)), 5).unwrap();

        // The kernel's order: the entries' IEEE total order, descending, then the token.
        let mut expected_values = Vec::new();
        let mut expected_ids = Vec::new();
        for row in rows {
            let mut order: Vec<usize> = (0..row.len()).collect();
            order.sort_by(|a, b| row[*b].total_cmp(&row[*a]).then(a.cmp(b)));
            expected_ids.extend(order[..5].iter().map(|token| *token as u32));
            expected_values.extend(order[..5].iter().map(|token| row[*token]));
        }

        assert_eq!(read::<u32>(&ids), expected_ids);
        assert_eq!(bits(&read::<f32>(&values)), bits(&expected_values));

        // Three equal largest entries in token order, then the two next; and `+0.0` above `-0.0`.
        assert_eq!(expected_ids[..5], [1, 3, 6, 0, 7]);
        assert_eq!(expected_ids[5..], [1, 5, 0, 6, 2]);
    }

    #[test]
    fn top_k_refuses_a_width_outside_the_vocabulary() {
        let logits = tensor(&[0.0_f32; 8], (2, 4));

        assert!(top_k(&logits, 0).is_err());
        assert!(top_k(&logits, 5).is_err());
        assert_eq!(top_k(&logits, 4).unwrap().0.dims(), [2, 4]);
    }

    // ---- filter_candidates_ -------------------------------------------------------------------

    /// The filter's rule over one descending row, in FP64: the kept candidates and the rest
    /// `-inf`, with the row's validity.
    fn filtered_row(values: &[f32], top_k: f32, top_p: f32, min_p: f32) -> (Vec<f32>, i32) {
        let limit = if top_k > 0.0 {
            top_k.min(values.len() as f32) as usize
        } else {
            values.len()
        };
        let mut kept: Vec<f32> = values
            .iter()
            .enumerate()
            .map(|(column, value)| {
                if column < limit {
                    *value
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();

        let mass = probabilities(&kept);
        let mut before = 0.0;
        for (column, mass) in mass.iter().enumerate() {
            if before > f64::from(top_p) {
                kept[column] = f32::NEG_INFINITY;
            }
            before += mass;
        }

        let largest = kept.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for value in &mut kept {
            if min_p > 0.0 && f64::from(*value) < f64::from(largest) + f64::from(min_p).ln() {
                *value = f32::NEG_INFINITY;
            }
        }

        let valid = kept.iter().any(|value| value.is_finite());
        (kept, i32::from(valid))
    }

    #[test]
    fn filter_candidates_applies_the_top_k_width_the_nucleus_and_min_p() {
        // Candidates of probabilities 0.5, 0.3, 0.15 and 0.05, descending.
        let row: Vec<f32> = [0.5_f32, 0.3, 0.15, 0.05].iter().map(|p| p.ln()).collect();
        let cases = [
            // The row's own top-k of two.
            (2.0, 1.0, 0.0),
            // A nucleus of 0.7: the mass before the third candidate is 0.8.
            (0.0, 0.7, 0.0),
            // A nucleus of zero keeps the largest candidate alone.
            (0.0, 0.0, 0.0),
            // Min-p of 0.2 drops what is below a fifth of the largest probability.
            (0.0, 1.0, 0.2),
            // All three: three candidates renormalized, a nucleus that keeps them, and a min-p
            // of a half that drops the third.
            (3.0, 0.9, 0.5),
            // A min-p above one drops every candidate.
            (0.0, 1.0, 2.0),
        ];

        let parameters: Vec<f32> = cases
            .iter()
            .flat_map(|(top_k, top_p, min_p)| {
                parameter_row(1.0, *top_k, *top_p, *min_p, [1.0, 0.0, 0.0])
            })
            .collect();
        let values = tensor(&row.repeat(cases.len()), (cases.len(), 4));
        let ids = tensor(&[0_u32, 1, 2, 3].repeat(cases.len()), (cases.len(), 4));

        let (kept, valid) = filter_candidates_(
            values.clone(),
            &ids,
            &tensor(&parameters, (cases.len(), PARAMETER_COLUMNS as usize)),
        )
        .unwrap();

        let expected: Vec<(Vec<f32>, i32)> = cases
            .iter()
            .map(|(top_k, top_p, min_p)| filtered_row(&row, *top_k, *top_p, *min_p))
            .collect();
        let expected_values: Vec<f32> =
            expected.iter().flat_map(|(kept, _)| kept.clone()).collect();
        let expected_valid: Vec<i32> = expected.iter().map(|(_, valid)| *valid).collect();

        assert_eq!(bits(&read::<f32>(&kept)), bits(&expected_values));
        assert_eq!(read::<i32>(&valid), expected_valid);

        // The kept counts of the six cases, and the one row that is no distribution.
        let counts: Vec<usize> = expected
            .iter()
            .map(|(kept, _)| kept.iter().filter(|value| value.is_finite()).count())
            .collect();
        assert_eq!(counts, [2, 2, 1, 3, 2, 0]);
        assert_eq!(expected_valid, [1, 1, 1, 1, 1, 0]);

        // The candidates are filtered where they lie.
        assert_eq!(bits(&read::<f32>(&values)), bits(&expected_values));
    }

    #[test]
    fn filter_candidates_reports_a_row_with_no_finite_candidate_or_a_nan() {
        let rows = [
            [f32::NEG_INFINITY; 3],
            [1.0, f32::NAN, 0.0],
            [f32::INFINITY, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ];
        let parameters = parameter_row(1.0, 0.0, 1.0, 0.0, [1.0, 0.0, 0.0]).repeat(4);

        let (_, valid) = filter_candidates_(
            tensor(&rows.concat(), (4, 3)),
            &tensor(&[0_u32, 1, 2].repeat(4), (4, 3)),
            &tensor(&parameters, (4, PARAMETER_COLUMNS as usize)),
        )
        .unwrap();

        assert_eq!(read::<i32>(&valid), [0, 0, 0, 1]);
    }

    // ---- candidate_order ----------------------------------------------------------------------

    #[test]
    fn candidate_order_sorts_by_token_and_fills_the_columns_no_token_takes() {
        let dropped = f32::NEG_INFINITY;
        let values = [0.5, dropped, 0.1, 0.2, 0.7, 0.6, 0.4, 0.3];
        let ids = [7, 2, 9, 0, 5, 12, 1, NO_SLOT];

        let (ordered_values, ordered_ids) =
            candidate_order(&tensor(&values, (2, 4)), &tensor(&ids, (2, 4)), 10).unwrap();

        // Row 0 is a permutation: a dropped candidate keeps its place among the ids. Row 1
        // holds two ids that are no token of a vocabulary of ten.
        assert_eq!(
            read::<u32>(&ordered_ids),
            [0, 2, 7, 9, 1, 5, NO_SLOT, NO_SLOT]
        );
        assert_eq!(
            bits(&read::<f32>(&ordered_values)),
            bits(&[0.2, dropped, 0.5, 0.1, 0.4, 0.7, dropped, dropped])
        );
    }

    // ---- softmax ------------------------------------------------------------------------------

    #[test]
    fn softmax_normalizes_each_row_and_gives_a_dropped_candidate_no_mass() {
        let rows = [
            [1.0_f32, -2.0, 0.5, 3.0],
            [f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY, 1.0],
            [-40.0, -41.0, -39.5, -60.0],
        ];
        let out = read::<f32>(&softmax(&tensor(&rows.concat(), (3, 4))).unwrap());

        for (row, out) in rows.iter().zip(out.chunks(4)) {
            for (expected, actual) in probabilities(row).iter().zip(out) {
                assert!((expected - f64::from(*actual)).abs() < 1e-6, "{out:?}");
            }
        }
        assert_eq!(out[4], 0.0);
        assert_eq!(out[6], 0.0);

        // A row with no finite entry has no distribution.
        let empty = softmax(&tensor(&[f32::NEG_INFINITY; 3], (1, 3))).unwrap();
        assert!(read::<f32>(&empty).iter().all(|value| value.is_nan()));
    }

    // ---- inverse_cdf --------------------------------------------------------------------------

    /// The first column whose inclusive cumulative mass exceeds `draw` times the row's mass.
    fn first_crossing(mass: &[f32], draw: f32) -> Option<usize> {
        let target = f64::from(draw) * mass.iter().map(|mass| f64::from(*mass)).sum::<f64>();
        let mut cumulative = 0.0;

        mass.iter().position(|mass| {
            cumulative += f64::from(*mass);
            cumulative > target
        })
    }

    #[test]
    fn inverse_cdf_selects_the_first_candidate_whose_cumulative_mass_passes_the_draw() {
        let mass = [0.1_f32, 0.0, 0.2, 0.3, 0.4];
        let ids = [3_u32, 5, 8, 20, 31];
        let draws = [0.0_f32, 0.05, 0.15, 0.45, 0.95, 0.999];

        let tokens = inverse_cdf(
            &tensor(&mass.repeat(draws.len()), (draws.len(), 5)),
            &tensor(&ids.repeat(draws.len()), (draws.len(), 5)),
            &tensor(&draws, draws.len()),
        )
        .unwrap();

        let expected: Vec<u32> = draws
            .iter()
            .map(|draw| ids[first_crossing(&mass, *draw).unwrap()])
            .collect();
        assert_eq!(read::<u32>(&tokens), expected);

        // The cumulative masses are 0.1, 0.1, 0.3, 0.6 and 1: a draw of zero selects the first
        // candidate with mass, and the candidate without mass is never selected.
        assert_eq!(expected, [3, 3, 8, 20, 31, 31]);
    }

    #[test]
    fn inverse_cdf_draws_against_the_rows_own_mass() {
        // Masses that sum to two: a draw is a share of that.
        let tokens = inverse_cdf(
            &tensor(&[0.5_f32, 1.5, 0.5, 1.5], (2, 2)),
            &tensor(&[4_u32, 9, 4, 9], (2, 2)),
            &tensor(&[0.2_f32, 0.3], 2),
        )
        .unwrap();

        assert_eq!(read::<u32>(&tokens), [4, 9]);
    }

    #[test]
    fn inverse_cdf_selects_token_zero_for_a_row_with_no_mass() {
        let tokens = inverse_cdf(
            &tensor(&[0.0_f32, 0.0, f32::NAN, f32::NAN], (2, 2)),
            &tensor(&[4_u32, 9, 4, 9], (2, 2)),
            &tensor(&[0.5_f32, 0.5], 2),
        )
        .unwrap();

        assert_eq!(read::<u32>(&tokens), [0, 0]);
    }

    /// A row wider than one chunk of the cumulative sum: the mass before a chunk is carried
    /// into it, so a draw selects the same candidate wherever the chunks' borders fall.
    #[test]
    fn inverse_cdf_carries_the_cumulative_mass_across_a_wide_row() {
        const WIDTH: usize = 2500;

        let mass = vec![1.0 / WIDTH as f32; WIDTH];
        let ids: Vec<u32> = (0..WIDTH as u32).map(|column| 2 * column).collect();
        // The first columns past the middle and past the first chunk's 1024, and the last one.
        let draws = [0.5002_f32, 0.4098, 0.9999];

        let tokens = inverse_cdf(
            &tensor(&mass.repeat(draws.len()), (draws.len(), WIDTH)),
            &tensor(&ids.repeat(draws.len()), (draws.len(), WIDTH)),
            &tensor(&draws, draws.len()),
        )
        .unwrap();

        let expected: Vec<u32> = draws
            .iter()
            .map(|draw| ids[first_crossing(&mass, *draw).unwrap()])
            .collect();
        assert_eq!(read::<u32>(&tokens), expected);
        assert_eq!(expected, [2 * 1250, 2 * 1024, 2 * 2499]);
    }

    // ---- argmax -------------------------------------------------------------------------------

    #[test]
    fn argmax_takes_the_lowest_column_among_equal_entries() {
        let dropped = f32::NEG_INFINITY;
        let rows = [
            [1.0, 3.0, 3.0, 0.0],
            [-2.0, dropped, -2.0, -7.0],
            [dropped, dropped, dropped, -1.0],
            // No numeric entry: the row returns its width.
            [dropped; 4],
        ];

        let tokens = argmax(&tensor(&rows.concat(), (4, 4))).unwrap();

        assert_eq!(read::<u32>(&tokens), [1, 0, 3, 4]);
    }

    // ---- log_softmax, logprob_gather, top_logprobs --------------------------------------------

    #[test]
    fn log_softmax_is_each_logit_less_the_log_of_the_rows_mass() {
        let rows = [[1.0_f32, -2.0, 0.5, 3.0], [30.0, 31.5, 29.0, -4.0]];
        let out = read::<f32>(&log_softmax(&tensor(&rows.concat(), (2, 4))).unwrap());

        for (row, out) in rows.iter().zip(out.chunks(4)) {
            let mass: f64 = row.iter().map(|logit| f64::from(*logit).exp()).sum();
            for (logit, actual) in row.iter().zip(out) {
                let expected = f64::from(*logit) - mass.ln();
                assert!((expected - f64::from(*actual)).abs() < 1e-5, "{out:?}");
            }
        }
    }

    #[test]
    fn logprob_gather_reports_a_tokens_entry_and_its_competition_rank() {
        let row = [-1.0_f32, -0.5, -1.0, -3.0];
        // The two tied entries, the largest, the smallest, and two tokens outside the row.
        let tokens = [0, 2, 1, 3, NO_SLOT, 4];

        let (logprobs, ranks) = logprob_gather(
            &tensor(&row.repeat(tokens.len()), (tokens.len(), 4)),
            &tensor(&tokens, tokens.len()),
        )
        .unwrap();

        // One plus the entries strictly above: the tied entries share rank two.
        assert_eq!(read::<f32>(&logprobs), [-1.0, -1.0, -0.5, -3.0, 0.0, 0.0]);
        assert_eq!(read::<i32>(&ranks), [2, 2, 1, 4, 0, 0]);
    }

    #[test]
    fn top_logprobs_keeps_the_largest_entries_with_ties_by_token() {
        let rows = [
            [-1.0_f32, -0.5, -1.0, -3.0, -0.5],
            [-0.0, -2.0, 0.0, -2.0, -9.0],
        ];
        let (values, ids) = top_logprobs(&tensor(&rows.concat(), (2, 5)), 3).unwrap();

        // Values are compared, so the two zeros of the second row tie and keep token order.
        assert_eq!(read::<u32>(&ids), [1, 4, 0, 0, 2, 1]);
        assert_eq!(read::<f32>(&values), [-0.5, -0.5, -1.0, 0.0, 0.0, -2.0]);

        // A row narrower than the candidates asked for fills the rest.
        let (values, ids) = top_logprobs(&tensor(&[-1.0_f32, -0.5], (1, 2)), 4).unwrap();
        assert_eq!(read::<u32>(&ids), [1, 0, NO_SLOT, NO_SLOT]);
        assert_eq!(
            bits(&read::<f32>(&values)),
            bits(&[-0.5, -1.0, f32::NEG_INFINITY, f32::NEG_INFINITY])
        );
    }

    #[test]
    fn top_logprobs_refuses_more_candidates_than_its_second_selection_holds() {
        let narrow = tensor(&[0.0_f32; 4], (1, 4));
        assert!(top_logprobs(&narrow, 0).is_err());

        // A 151936-token row is 38 chunks, and one selection holds 8192 candidates of them.
        let (output, launches) = recorded(|| {
            let logprobs = meta((2, 151_936), DType::F32);
            let refused = top_logprobs(&logprobs, 216).is_err();
            let (values, _) = top_logprobs(&logprobs, 215)?;

            Ok((refused, values.dims().to_vec()))
        });

        assert_eq!(output.unwrap(), (true, vec![2, 215]));
        assert_eq!(
            stated(&launches),
            [
                (
                    "top_logprobs",
                    "candidates_215_tile_4096_indirect_false_pdl_false"
                ),
                (
                    "top_logprobs",
                    "candidates_215_tile_8192_indirect_true_pdl_false"
                ),
            ]
        );
    }

    // ---- Kernels stated under a meta context --------------------------------------------------

    #[test]
    fn the_shaper_states_its_kernel() {
        let (output, launches) = recorded(|| {
            let shape = |logits: DType, extras: Option<usize>| {
                logits_shape(
                    &meta((3, 100), logits),
                    &meta((4, 100), DType::I32),
                    &meta(3, DType::U32),
                    &meta((3, PARAMETER_COLUMNS as usize), DType::F32),
                    extras.map(|extras| meta((3, extras), DType::U32)).as_ref(),
                    &meta((4, 100), DType::F32),
                    &meta((3, 2), DType::U32),
                    &meta((3, 4), DType::U32),
                )
            };
            let plain = shape(DType::F32, None)?;
            let counted = shape(DType::BF16, Some(2))?;

            Ok([plain, counted].map(|shaped| (shaped.dims().to_vec(), shaped.dtype())))
        });

        assert_eq!(
            output.unwrap(),
            [(vec![3, 100], DType::F32), (vec![3, 100], DType::F32)]
        );
        assert_eq!(
            stated(&launches),
            [
                (
                    "logits_shape",
                    "dtype_f32_vocabulary_100_block_128_extras_0_divide_true_pdl_false"
                ),
                (
                    "logits_shape",
                    "dtype_bf16_vocabulary_100_block_128_extras_2_divide_true_pdl_false"
                ),
            ]
        );
    }

    #[test]
    fn a_draw_states_its_five_kernels() {
        let (output, launches) = recorded(|| {
            let (values, ids) = top_k(&meta((3, 100), DType::F32), 16)?;
            let (values, valid) = filter_candidates_(
                values,
                &ids,
                &meta((3, PARAMETER_COLUMNS as usize), DType::F32),
            )?;
            let (values, ids) = candidate_order(&values, &ids, 100)?;
            let tokens = inverse_cdf(&softmax(&values)?, &ids, &meta(3, DType::F32))?;

            Ok([values, ids, valid, tokens].map(|out| (out.dims().to_vec(), out.dtype())))
        });

        assert_eq!(
            output.unwrap(),
            [
                (vec![3, 16], DType::F32),
                (vec![3, 16], DType::U32),
                (vec![3], DType::I32),
                (vec![3], DType::U32),
            ]
        );
        assert_eq!(
            stated(&launches),
            [
                ("top_k", "dtype_f32_tile_2048_ids_i32_pdl_false"),
                ("filter_candidates", "width_16_tile_8192_ids_i32_pdl_false"),
                ("candidate_order", "values_f32_ids_i32_pdl_false"),
                ("softmax", "dtype_f32_columns_16_tile_8192_pdl_false"),
                (
                    "inverse_cdf",
                    "width_16_tile_8192_ids_i32_tokens_i32_pdl_false"
                ),
            ]
        );
    }

    #[test]
    fn the_greedy_choice_and_the_reported_scores_state_their_kernels() {
        let (output, launches) = recorded(|| {
            let logits = meta((3, 100), DType::F32);
            let tokens = argmax(&logits)?;
            let logprobs = log_softmax(&logits)?;
            let (logprob, rank) = logprob_gather(&logprobs, &tokens)?;
            let (values, ids) = top_logprobs(&logprobs, 5)?;

            Ok([tokens, logprob, rank, values, ids].map(|out| (out.dims().to_vec(), out.dtype())))
        });

        assert_eq!(
            output.unwrap(),
            [
                (vec![3], DType::U32),
                (vec![3], DType::F32),
                (vec![3], DType::I32),
                (vec![3, 5], DType::F32),
                (vec![3, 5], DType::U32),
            ]
        );
        assert_eq!(
            stated(&launches),
            [
                ("argmax", "dtype_f32_width_100_ids_i32_pdl_false"),
                ("log_softmax", "dtype_f32_columns_100_tile_128_pdl_false"),
                ("logprob_gather", "columns_100_ids_i32_tile_128_pdl_false"),
                (
                    "top_logprobs",
                    "candidates_5_tile_4096_indirect_false_pdl_false"
                ),
                (
                    "top_logprobs",
                    "candidates_5_tile_8_indirect_true_pdl_false"
                ),
            ]
        );
    }
}
