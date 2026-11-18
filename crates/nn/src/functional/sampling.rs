//! Token selection: shaping a row's logits, choosing among its candidates, and reporting the
//! log-probabilities of the chosen token.
//!
//! ```text
//! logits_shape          shaped = penalize(adjust(logits)) / temperature
//! sampling_from_logits  one token per row under its top-k width, nucleus and min-p thresholds,
//!                       drawn in ascending token id from one uniform draw
//! argmax                the largest entry's column, the lowest among equals
//! log_softmax           x − (max x + ln Σ exp(x − max x))
//! logprob_gather        one token's log-probability and its competition rank
//! top_logprobs          the `n` largest entries of a row, descending, with their token ids
//! top_k                 the `width` largest entries of a row, descending, with their token ids
//! softmax               p = exp(x − max x) / Σ exp(x − max x)
//! ```
//!
//! A drawn token is `logits_shape` and `sampling_from_logits`; a greedy token is `logits_shape`
//! and `argmax`. The draw walks the kept candidates in ascending token id, so a fixed draw
//! selects the same token whatever order equal scores are kept in. No function here generates
//! randomness: a draw is a uniform value its caller computes.
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
    ArgMax, ArgMaxArgs, ArgMaxDescriptor, INDEX, LogSoftmax, LogSoftmaxArgs, LogSoftmaxDescriptor,
    LogitsShape, LogprobArgs, LogprobDescriptor, LogprobGather, SamplingArgs, SamplingDescriptor,
    SamplingFromLogits, ShapeArgs, ShapeDescriptor, TopK, TopKArgs, TopKDescriptor, TopLogprobs,
    TopLogprobsArgs, TopLogprobsDescriptor, ban_words, top_k_workspace,
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

/// One token per row, drawn from the row's shaped logits under its top-k width, its nucleus
/// threshold and its min-p threshold: `(tokens, valid)`, `[rows]` U32 and `[rows]` I32.
///
/// ```text
/// order    = the row's columns by descending logit, equal logits in ascending token id
/// limit    = top_k > 0 ? floor(min(top_k, vocabulary)) : vocabulary
/// kept     = the first `limit` positions of the order                  the top-k width
/// m_j      = exp(x_j − max x) in units of 2⁻⁴⁰, rounded to the nearest   a position's mass
/// kept     = the kept positions whose predecessors' Σ m is at most
///            floor(top_p · Σ_kept m)                                     the nucleus, top_p < 1
/// kept     = the kept positions with x ≥ max x + ln(min_p)               min-p, min_p > 0
/// token    = the first kept token by id whose cumulative Σ m exceeds floor(draw · Σ_kept m)
/// valid    = the kept candidates hold no NaN and no +inf, and the largest logit is finite
/// ```
///
/// The draw inverts the kept candidates' distribution in ascending token id, so a fixed draw
/// selects the same token whatever order equal logits were kept in, and a draw of zero selects the
/// lowest kept token id with mass. A candidate's mass is the softmax numerator held as an integer,
/// so the nucleus's comparison and the draw's cumulative are exact sums and the kernel's launch
/// is deterministic. The order is of the logits' bit patterns, so `+0.0` is above `-0.0`; a
/// positive NaN is above every logit and a negative NaN below. The nucleus keeps the first
/// candidate whatever its threshold of zero or more, and min-p is measured against the row's
/// largest logit. A row whose width keeps nothing, whose nucleus is negative, whose min-p
/// threshold is NaN or above its largest logit, or whose logits are all `-inf` keeps nothing;
/// it reports `0` and holds token `0`, as does a row whose kept candidates hold a NaN or `+inf`.
/// The function does not branch on that, its reader does.
///
/// `logits` is FP32 `[rows, vocabulary]`, `parameters` `[rows, PARAMETER_COLUMNS]` FP32 and
/// `draws` `[rows]` FP32 uniform values in `[0, 1)`.
///
/// # Errors
///
/// When a tensor is not the shape or element type stated above, or the operator refuses its
/// arguments.
pub fn sampling_from_logits(
    logits: &Tensor,
    parameters: &Tensor,
    draws: &Tensor,
) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = SamplingCall::OWNER;

    let (rows, vocabulary) = matrix(OWNER, "logits", logits)?;
    require(
        OWNER,
        "logits",
        logits,
        DType::F32,
        &[Some(rows), Some(vocabulary)],
    )?;
    require(
        OWNER,
        "parameters",
        parameters,
        DType::F32,
        &[Some(rows), Some(PARAMETER_COLUMNS as usize)],
    )?;
    require(OWNER, "draws", draws, DType::F32, &[Some(rows)])?;

    compute(SamplingCall {
        logits,
        parameters,
        draws,
        rows,
    })
}

/// A checked call of [`sampling_from_logits`] over `rows` rows.
struct SamplingCall<'t> {
    logits: &'t Tensor,
    parameters: &'t Tensor,
    draws: &'t Tensor,
    rows: usize,
}

impl SamplingCall<'_> {
    const OWNER: &'static str = "sampling_from_logits";
}

impl Fused for SamplingCall<'_> {
    type Output = (Tensor, Tensor);

    fn device(&self) -> &Device {
        self.logits.device()
    }

    fn cpu(self) -> Result<(Tensor, Tensor)> {
        let Self {
            logits,
            parameters,
            draws,
            rows,
        } = self;

        let logits = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let parameters = parameters.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let draws = draws.to_dtype(DType::F32)?.to_vec1::<f32>()?;

        let mut tokens = Vec::with_capacity(rows);
        let mut valid = Vec::with_capacity(rows);
        for ((row, parameters), &draw) in logits.iter().zip(&parameters).zip(&draws) {
            let drawn = draw_from(
                row,
                parameters[TOP_K as usize],
                parameters[TOP_P as usize],
                parameters[MIN_P],
                draw,
            );
            tokens.push(drawn.unwrap_or(0));
            valid.push(i32::from(drawn.is_some()));
        }

        Ok((
            Tensor::from_vec(tokens, rows, &Device::Cpu)?,
            Tensor::from_vec(valid, rows, &Device::Cpu)?,
        ))
    }

    fn launch(self) -> Result<(Tensor, Tensor)> {
        let Self {
            logits,
            parameters,
            draws,
            rows,
        } = self;

        let tokens = ops::empty(rows, DType::U32)?;
        let valid = ops::empty(rows, DType::I32)?;
        ops::launch::<SamplingFromLogits>(
            &SamplingDescriptor::new(),
            SamplingArgs {
                logits: ops::operand(logits)?,
                parameters: ops::operand(parameters)?,
                draws: ops::operand(draws)?,
                tokens: signed(&tokens)?,
                valid: ops::operand(&valid)?,
            },
        )?;
        Ok((tokens, valid))
    }
}

/// A mass's unit: 2⁻⁴⁰ of the largest logit's.
const MASS_UNIT: f32 = 1_099_511_627_776.0;

/// The order-preserving key of an FP32 value: a negative value's bits inverted and every other
/// value's sign bit set, so keys compare as the values do and `+0.0` is above `-0.0`.
const fn key_of(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits >> 31 == 1 {
        !bits
    } else {
        bits | 1 << 31
    }
}

/// `floor(fraction · total)` for a fraction in `[0, 1)`: an FP32 fraction times `2⁶⁴` is an
/// integer, so the product is exact.
fn share(total: u64, fraction: f32) -> u64 {
    let scaled = (f64::from(fraction) * 2.0_f64.powi(64)) as u128;
    ((u128::from(total) * scaled) >> 64) as u64
}

/// The token [`sampling_from_logits`] draws from one row, `None` for a row that keeps nothing
/// usable: the operator's rule, position by position.
fn draw_from(row: &[f32], top_k: f32, top_p: f32, min_p: f32, draw: f32) -> Option<u32> {
    let vocabulary = row.len();
    let keys: Vec<u32> = row.iter().map(|&value| key_of(value)).collect();
    let mut order: Vec<usize> = (0..vocabulary).collect();
    order.sort_by_key(|&token| (std::cmp::Reverse(keys[token]), token));

    // The keys of the infinities: above `+inf`'s are the positive NaNs, below `-inf`'s the
    // negative ones.
    let largest = keys[order[0]];
    let negative_nans = keys
        .iter()
        .filter(|&&key| key < key_of(f32::NEG_INFINITY))
        .count();
    let limit = if top_k > 0.0 {
        top_k.min(vocabulary as f32).floor() as usize
    } else {
        vocabulary
    };
    let maximum = row[order[0]];
    // A NaN min-p applies, and its threshold keeps nothing.
    let min_p_applies = min_p > 0.0 || min_p.is_nan();
    let threshold = maximum + min_p.ln();
    let usable = largest < key_of(f32::INFINITY)
        && largest > key_of(f32::NEG_INFINITY)
        && limit > 0
        && limit <= vocabulary - negative_nans
        && (top_p >= 0.0 || top_p.is_nan())
        && (!min_p_applies || maximum >= threshold);
    if !usable {
        return None;
    }

    // The masses, then the prefix each filter keeps: the width, the nucleus, min-p.
    let mass = |token: usize| ((row[token] - maximum).exp() * MASS_UNIT).round_ties_even() as u64;
    let mut kept = limit;
    if top_p < 1.0 && !top_p.is_nan() {
        let nucleus = share(order[..limit].iter().map(|&token| mass(token)).sum(), top_p);
        let mut before = 0;
        kept = 0;
        for &token in &order[..limit] {
            if before > nucleus {
                break;
            }
            kept += 1;
            before += mass(token);
        }
    }
    if min_p_applies {
        kept = order[..kept]
            .iter()
            .take_while(|&&token| row[token] >= threshold)
            .count();
    }

    // The draw, over the kept tokens in token order.
    let mut chosen = order[..kept].to_vec();
    chosen.sort_unstable();
    let total = chosen.iter().map(|&token| mass(token)).sum();
    let target = share(total, draw.clamp(0.0, 1.0 - f32::EPSILON / 2.0));
    let mut running = 0;
    chosen.into_iter().find_map(|token| {
        running += mass(token);
        (mass(token) > 0 && running > target).then_some(token as u32)
    })
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

    // ---- sampling_from_logits ----------------------------------------------------------------

    /// The tokens a row draws for `draws` evenly spaced draws, `(i + 0.5) / draws`, and the
    /// share of the draws each token took.
    fn draw_shares(row: &[f32], top_k: f32, top_p: f32, min_p: f32, draws: usize) -> Vec<f64> {
        let uniforms: Vec<f32> = (0..draws)
            .map(|draw| (draw as f32 + 0.5) / draws as f32)
            .collect();
        let parameters = parameter_row(1.0, top_k, top_p, min_p, [1.0, 0.0, 0.0]).repeat(draws);
        let (tokens, valid) = sampling_from_logits(
            &tensor(&row.repeat(draws), (draws, row.len())),
            &tensor(&parameters, (draws, PARAMETER_COLUMNS as usize)),
            &tensor(&uniforms, draws),
        )
        .unwrap();
        assert!(read::<i32>(&valid).iter().all(|&valid| valid == 1));

        let mut shares = vec![0.0; row.len()];
        for token in read::<u32>(&tokens) {
            shares[token as usize] += 1.0 / draws as f64;
        }
        shares
    }

    #[test]
    fn sampling_from_logits_draws_from_the_top_k_nucleus_and_min_p_candidates() {
        // Probabilities 0.05, 0.5, 0.15 and 0.3 at tokens 0 to 3: in descending order tokens 1,
        // 3, 2 and 0.
        let row: Vec<f32> = [0.05_f32, 0.5, 0.15, 0.3].iter().map(|p| p.ln()).collect();
        let cases: [((f32, f32, f32), [f64; 4]); 6] = [
            // The whole row.
            ((0.0, 1.0, 0.0), [0.05, 0.5, 0.15, 0.3]),
            // A top-k of two: tokens 1 and 3, renormalized.
            ((2.0, 1.0, 0.0), [0.0, 0.625, 0.0, 0.375]),
            // A nucleus of 0.7: the mass before token 2 is 0.8, above it.
            ((0.0, 0.7, 0.0), [0.0, 0.625, 0.0, 0.375]),
            // A nucleus of zero keeps the largest candidate alone.
            ((0.0, 0.0, 0.0), [0.0, 1.0, 0.0, 0.0]),
            // Min-p of 0.2 drops what is below a fifth of the largest probability.
            ((0.0, 1.0, 0.2), [0.0, 0.5 / 0.95, 0.15 / 0.95, 0.3 / 0.95]),
            // A top-k of three, a nucleus that keeps the three (the mass before token 2 is 0.8 of
            // 0.95) and a min-p of a half that drops token 2.
            ((3.0, 0.9, 0.5), [0.0, 0.625, 0.0, 0.375]),
        ];

        // Evenly spaced draws take each kept token's share of the kept mass, to the spacing.
        const DRAWS: usize = 4000;
        for ((top_k, top_p, min_p), expected) in cases {
            let shares = draw_shares(&row, top_k, top_p, min_p, DRAWS);
            for (share, expected) in shares.iter().zip(expected) {
                assert!(
                    (share - expected).abs() <= 2.0 / DRAWS as f64,
                    "top-k {top_k}, top-p {top_p}, min-p {min_p}: {shares:?}, {expected:?}"
                );
            }
        }
    }

    #[test]
    fn sampling_from_logits_inverts_the_kept_distribution_in_token_order() {
        // Tokens 2, 5 and 7 hold the mass, equally; token 6 is banned.
        let mut row = [-30.0_f32; 8];
        row[2] = 1.0;
        row[5] = 1.0;
        row[6] = f32::NEG_INFINITY;
        row[7] = 1.0;
        let draws = [0.0_f32, 0.3, 0.34, 0.66, 0.67, 0.999_999_9];
        let parameters = parameter_row(1.0, 3.0, 1.0, 0.0, [1.0, 0.0, 0.0]).repeat(draws.len());

        let (tokens, valid) = sampling_from_logits(
            &tensor(&row.repeat(draws.len()), (draws.len(), 8)),
            &tensor(&parameters, (draws.len(), PARAMETER_COLUMNS as usize)),
            &tensor(&draws, draws.len()),
        )
        .unwrap();

        // A draw of zero takes the lowest kept token, the highest draws the last one.
        assert_eq!(read::<u32>(&tokens), [2, 2, 5, 5, 7, 7]);
        assert_eq!(read::<i32>(&valid), [1; 6]);
    }

    #[test]
    fn sampling_from_logits_keeps_the_lowest_tokens_of_equal_logits_a_width_cuts() {
        // Five equal largest logits; a top-k of three keeps tokens 1, 4 and 6 of them.
        let row = [0.0_f32, 2.0, 0.0, -1.0, 2.0, 0.0, 2.0, 2.0, 2.0];
        let shares = draw_shares(&row, 3.0, 1.0, 0.0, 300);

        let kept: Vec<usize> = (0..row.len())
            .filter(|&token| shares[token] > 0.0)
            .collect();
        assert_eq!(kept, [1, 4, 6]);
    }

    #[test]
    fn sampling_from_logits_reports_a_row_that_keeps_nothing_usable() {
        let rows = [
            // No finite logit.
            [f32::NEG_INFINITY; 3],
            // A NaN or +inf among the candidates.
            [1.0, f32::NAN, 0.0],
            [f32::INFINITY, 1.0, 0.0],
            // A width that keeps nothing, a negative nucleus, a min-p above one.
            [2.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            // A usable row.
            [2.0, 1.0, 0.0],
        ];
        let parameters = [
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 1.0, 0.0),
            (0.0, -0.1, 0.0),
            (0.0, 1.0, 2.0),
            (0.0, 1.0, 0.0),
        ]
        .iter()
        .flat_map(|(top_k, top_p, min_p)| {
            parameter_row(1.0, *top_k, *top_p, *min_p, [1.0, 0.0, 0.0])
        })
        .collect::<Vec<f32>>();

        let (tokens, valid) = sampling_from_logits(
            &tensor(&rows.concat(), (rows.len(), 3)),
            &tensor(&parameters, (rows.len(), PARAMETER_COLUMNS as usize)),
            &tensor(&[0.9_f32; 7], 7),
        )
        .unwrap();

        // The usable row's masses are 0.665, 0.245 and 0.090 of its total, so a draw of 0.9
        // lands on token 1.
        assert_eq!(read::<i32>(&valid), [0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(read::<u32>(&tokens), [0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn sampling_from_logits_refuses_columns_of_another_shape_or_type() {
        let logits = tensor(&[0.0_f32; 8], (2, 4));
        let parameters = tensor(&[0.0_f32; 16], (2, PARAMETER_COLUMNS as usize));
        let draws = tensor(&[0.5_f32; 2], 2);

        assert!(sampling_from_logits(&logits, &parameters, &tensor(&[0.5_f32; 3], 3)).is_err());
        assert!(sampling_from_logits(&logits, &tensor(&[0.0_f32; 8], (2, 4)), &draws).is_err());
        assert!(
            sampling_from_logits(&logits.to_dtype(DType::BF16).unwrap(), &parameters, &draws)
                .is_err()
        );
        assert!(sampling_from_logits(&logits, &parameters, &draws).is_ok());
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
    fn a_draw_states_its_kernel() {
        let (output, launches) = recorded(|| {
            let (tokens, valid) = sampling_from_logits(
                &meta((3, 100), DType::F32),
                &meta((3, PARAMETER_COLUMNS as usize), DType::F32),
                &meta(3, DType::F32),
            )?;

            Ok([tokens, valid].map(|out| (out.dims().to_vec(), out.dtype())))
        });

        assert_eq!(
            output.unwrap(),
            [(vec![3], DType::U32), (vec![3], DType::I32)]
        );
        assert_eq!(
            stated(&launches),
            [("sampling_from_logits", "logits_f32_tokens_i32")]
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
