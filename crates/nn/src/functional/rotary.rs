//! Rotary position embedding: the factors of a batch's positions and their application to the
//! heads of an attention.
//!
//! ```text
//! rotary_factors   cos[t, j], sin[t, j] of  position[axis(j), t] · inv_freq[j]
//! rotary           out = rotate(x)
//! qk_norm_rope     out = rotate(x · rsqrt(mean(x²) + eps) · gain)       queries and keys
//! qk_norm_rope_    the same without a gain, in place over two column ranges of wider rows
//!
//! rotate(x)[i]         = x[i] · cos[i] − x[i + p] · sin[i]              i < p pairs
//! rotate(x)[i + p]     = x[i + p] · cos[i] + x[i] · sin[i]
//! rotate(x)[2p..]      = x[2p..]
//! ```
//!
//! The factors are computed once per forward and every attention layer turns its queries and
//! keys by them. The rotation is the half-split pairing (`x[i]` against `x[i + p]`, GPT-NeoX's),
//! and a head may rotate only its leading channels: the factor tables are as wide as the pairs
//! that rotate, and a head wider than twice that passes its trailing channels through.
//!
//! A position of more than one axis (M-RoPE, an N-d grid) reads one coordinate per axis, and
//! each frequency reads the coordinate of the axis that owns it: the axes own contiguous blocks
//! of the frequencies, or cycle over the whole spectrum (interleaved M-RoPE), where frequency
//! `j` reads axis `j mod axes` while `j < widths[j mod axes] · axes` and axis 0 otherwise, so
//! every axis keeps frequencies from the whole range.
//!
//! The fused normalization is not the arithmetic of a separate per-head RMS normalization
//! followed by the rotation, which rounds in between: [`qk_norm_rope`] normalizes in FP32,
//! rotates and rounds once. A model keeps the form its reference computes.

use std::ops::Range;

use ops::rope::{
    FactorsArgs, FactorsDescriptor, MAX_AXES, QkNormRope, QkNormRopeArgs, QkNormRopeDescriptor,
    RopeApply, RopeApplyArgs, RopeApplyDescriptor, RotaryFactors, Rotation,
};
use tensor::{D, DType, Device, Tensor};

use super::norm::{gain, rms_scaled};
use super::nvfp4::integer;
use super::{ACCUMULATE, Fused, compute, vector};
use crate::error::{Error, Result};

/// The heads and the head width of `x`, the tensor `name` of `[tokens, heads, head_dim]`.
pub(super) fn heads_of(owner: &'static str, name: &str, x: &Tensor) -> Result<(u32, u32)> {
    if let [_, heads, head_dim] = *x.dims()
        && let (Ok(heads), Ok(head_dim)) = (u32::try_from(heads), u32::try_from(head_dim))
        && heads > 0
        && head_dim > 0
    {
        return Ok((heads, head_dim));
    }

    Err(Error::shape(
        owner,
        format!(
            "{name} is {:?}, [tokens, heads, head_dim] expected",
            x.dims()
        ),
    ))
}

/// The channels of a `head_dim`-wide head the factor tables `cos` and `sin` rotate: twice
/// their columns. The tables are `[rows, pairs]` of one shape and element type.
fn rotated(owner: &'static str, cos: &Tensor, sin: &Tensor, head_dim: u32) -> Result<u32> {
    if let [_, pairs] = *cos.dims()
        && sin.dims() == cos.dims()
        && sin.dtype() == cos.dtype()
        && pairs > 0
        && 2 * pairs <= head_dim as usize
    {
        return Ok(2 * pairs as u32);
    }

    Err(Error::shape(
        owner,
        format!(
            "factor tables of {:?} and {:?} for a head of {head_dim} channels: two tables of \
             [tokens, pairs] expected, at most half the head wide",
            cos.dims(),
            sin.dims()
        ),
    ))
}

/// Refuses factor tables that are not one FP32 row per token of `tokens`.
fn per_token(owner: &'static str, cos: &Tensor, tokens: usize) -> Result<()> {
    if cos.dims()[0] != tokens || cos.dtype() != DType::F32 {
        return Err(Error::shape(
            owner,
            format!(
                "factor tables of {} {:?} for {tokens} tokens: one f32 row per token expected",
                ops::abi::name(cos.dtype()),
                cos.dims()
            ),
        ));
    }
    Ok(())
}

/// The rotation of the module documentation over `x`, FP32 `[tokens, heads, head_dim]`, by the
/// FP32 tables `cos` and `sin`, `[tokens, pairs]`.
fn rotate(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (head_dim, pairs) = (x.dims()[2], cos.dims()[1]);
    let (cos, sin) = (cos.unsqueeze(1)?, sin.unsqueeze(1)?);

    let first = x.narrow(D::Minus1, 0, pairs)?;
    let second = x.narrow(D::Minus1, pairs, pairs)?;
    let turned_first = (first.broadcast_mul(&cos)? - second.broadcast_mul(&sin)?)?;
    let turned_second = (second.broadcast_mul(&cos)? + first.broadcast_mul(&sin)?)?;

    if head_dim == 2 * pairs {
        Ok(Tensor::cat(&[turned_first, turned_second], D::Minus1)?)
    } else {
        let through = x.narrow(D::Minus1, 2 * pairs, head_dim - 2 * pairs)?;
        Ok(Tensor::cat(
            &[turned_first, turned_second, through],
            D::Minus1,
        )?)
    }
}

/// The axis each of `columns` frequencies reads, `[columns]` U32, under `sections`, the rotated
/// channels each axis owns (twice its frequencies).
fn frequency_axes(sections: &[u32], columns: usize, interleaved: bool) -> Result<Tensor> {
    let axes = sections.len();
    let frequency = Tensor::arange(0_u32, columns as u32, &Device::Cpu)?;
    let widths: Vec<u32> = sections.iter().map(|section| section / 2).collect();
    let widths = Tensor::new(widths.as_slice(), &Device::Cpu)?;

    if interleaved {
        // The axes cycle over the spectrum; a frequency its axis cannot claim falls to axis 0.
        let cycle = frequency.broadcast_remainder(&integer(axes as u32, &frequency)?)?;
        let claimed = widths
            .index_select(&cycle, 0)?
            .broadcast_mul(&integer(axes as u32, &frequency)?)?;

        Ok(frequency
            .lt(&claimed)?
            .where_cond(&cycle, &cycle.zeros_like()?)?)
    } else {
        // The axes own contiguous blocks: a frequency's axis is the number of blocks that end
        // at or before it.
        let ends = widths
            .to_dtype(ACCUMULATE)?
            .cumsum(0)?
            .to_dtype(DType::U32)?;

        Ok(frequency
            .unsqueeze(1)?
            .broadcast_ge(&ends.unsqueeze(0)?)?
            .to_dtype(DType::U32)?
            .sum(1)?)
    }
}

/// The cosine and sine factors of a batch's positions: `(cos, sin)`, FP32 `[tokens, columns]`
/// each, of the angle `position · inv_freq[j]`.
///
/// `inv_freq` is `[columns]`, one inverse frequency per rotated pair. With no `sections` the
/// positions are one axis and every element of `positions` is one token's. With `sections`, the
/// rotated channels each position axis owns in axis order, `positions` is `[axes, tokens]`,
/// axis-major, and frequency `j` reads the coordinate of its axis: the axes own contiguous
/// blocks of the frequencies, or cycle over them when `interleaved` (module documentation).
/// Positions are I32, or FP32 coordinates of a grid ([`grid_positions`]).
///
/// # Errors
///
/// Sections that do not cover the frequencies or are more than the operator's axes, positions
/// that are not one coordinate per axis and token, or the operator's refusal.
pub fn rotary_factors(
    positions: &Tensor,
    inv_freq: &Tensor,
    sections: &[u32],
    interleaved: bool,
) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = RotaryFactorsCall::OWNER;

    let &[columns] = inv_freq.dims() else {
        return Err(Error::shape(
            OWNER,
            format!(
                "the inverse frequencies are {:?}, one vector expected",
                inv_freq.dims()
            ),
        ));
    };
    let axes = sections.len().max(1);
    if axes > MAX_AXES
        || sections.contains(&0)
        || (!sections.is_empty() && sections.iter().sum::<u32>() as usize != 2 * columns)
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "sections {sections:?} over {columns} frequencies: one to {MAX_AXES} axes that \
                 own {} rotated channels together",
                2 * columns
            ),
        ));
    }
    if !sections.is_empty() && (positions.rank() < 2 || positions.dims()[0] != axes) {
        return Err(Error::shape(
            OWNER,
            format!(
                "positions of {:?}: sectioned positions have one leading coordinate per axis, \
                 {axes} in all",
                positions.dims()
            ),
        ));
    }
    compute(RotaryFactorsCall {
        positions,
        inv_freq,
        sections,
        interleaved,
        columns,
        axes,
    })
}

/// A checked call of [`rotary_factors`]: `columns` frequencies over `axes` position axes.
struct RotaryFactorsCall<'t> {
    positions: &'t Tensor,
    inv_freq: &'t Tensor,
    sections: &'t [u32],
    interleaved: bool,
    columns: usize,
    axes: usize,
}

impl RotaryFactorsCall<'_> {
    const OWNER: &'static str = "rotary_factors";
}

impl Fused for RotaryFactorsCall<'_> {
    type Output = (Tensor, Tensor);

    fn device(&self) -> &Device {
        self.positions.device()
    }

    fn cpu(self) -> Result<(Tensor, Tensor)> {
        let Self {
            positions,
            sections,
            columns,
            axes,
            ..
        } = self;
        let tokens = positions.elem_count() / axes;
        let coordinates = positions.to_dtype(ACCUMULATE)?.reshape((axes, tokens))?;

        // Each frequency's coordinate of each token, `[tokens, columns]`.
        let per_frequency = if sections.is_empty() {
            coordinates.t()?
        } else {
            coordinates
                .index_select(&frequency_axes(sections, columns, self.interleaved)?, 0)?
                .t()?
        };
        let angle =
            per_frequency.broadcast_mul(&self.inv_freq.to_dtype(ACCUMULATE)?.unsqueeze(0)?)?;

        Ok((angle.cos()?, angle.sin()?))
    }

    fn launch(self) -> Result<(Tensor, Tensor)> {
        let Self {
            positions, columns, ..
        } = self;
        let tokens = positions.elem_count() / self.axes;
        let cos = ops::empty((tokens, columns), DType::F32)?;
        let sin = ops::empty((tokens, columns), DType::F32)?;

        ops::launch::<RotaryFactors>(
            &FactorsDescriptor::new(
                DType::F32,
                positions.dtype(),
                columns as u32,
                1.0,
                self.sections,
                self.interleaved,
            )?,
            FactorsArgs {
                positions: ops::operand(positions)?,
                frequencies: ops::operand(self.inv_freq)?,
                cosine: ops::operand(&cos)?,
                sine: ops::operand(&sin)?,
            },
        )?;
        Ok((cos, sin))
    }
}

/// `x`, `[tokens, heads, head_dim]`, turned by the factors `cos` and `sin`, FP32 `[tokens,
/// pairs]`: the rotation alone, in FP32, rounded once into `x`'s element type.
///
/// # Errors
///
/// Tables that are not one FP32 row per token or are wider than half the head, or the
/// operator's refusal.
pub fn rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    const OWNER: &str = RotaryCall::OWNER;

    let (_, head_dim) = heads_of(OWNER, "the input", x)?;
    rotated(OWNER, cos, sin, head_dim)?;
    per_token(OWNER, cos, x.dims()[0])?;

    compute(RotaryCall { x, cos, sin })
}

/// A checked call of [`rotary`].
struct RotaryCall<'t> {
    x: &'t Tensor,
    cos: &'t Tensor,
    sin: &'t Tensor,
}

impl RotaryCall<'_> {
    const OWNER: &'static str = "rotary";
}

impl Fused for RotaryCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.x.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let Self { x, cos, sin } = self;
        Ok(rotate(&x.to_dtype(ACCUMULATE)?, cos, sin)?.to_dtype(x.dtype())?)
    }

    fn launch(self) -> Result<Tensor> {
        let Self { x, cos, sin } = self;
        let out = ops::empty(x.shape(), x.dtype())?;
        let args = RopeApplyArgs {
            x: ops::operand(x)?,
            cosine: ops::operand(cos)?,
            sine: ops::operand(sin)?,
            out: ops::operand(&out)?,
        };
        ops::launch::<RopeApply>(&RopeApplyDescriptor::of(&args, Rotation::Half)?, args)?;
        Ok(out)
    }
}

/// Turns each head's trailing `2 · pairs` channels of `x`, `[tokens, heads, head_dim]`, in
/// adjacent pairs by the factors `cos` and `sin`, FP32 `[tokens, pairs]`, in place: the pair
/// `(a, b)` becomes `(a·cos − b·sin, b·cos + a·sin)`, or by the negated angles when `inverse`,
/// in FP32, rounded once into `x`'s element type. The leading channels stay as they are.
///
/// A latent attention rotates the trailing channels of its queries before it attends and
/// rotates the attended values back by the inverse afterwards, so its cache keeps one rotated
/// form of every entry.
///
/// # Errors
///
/// Tables that are not one FP32 row per token or are wider than half the head, a strided `x`,
/// or the operator's refusal.
pub fn rotary_trailing_(x: &Tensor, cos: &Tensor, sin: &Tensor, inverse: bool) -> Result<()> {
    const OWNER: &str = TrailingCall::OWNER;

    let (_, head_dim) = heads_of(OWNER, "the input", x)?;
    let rotated = rotated(OWNER, cos, sin, head_dim)?;
    per_token(OWNER, cos, x.dims()[0])?;
    if !x.is_contiguous() {
        return Err(Error::shape(
            OWNER,
            "a rotation in place needs a contiguous input".to_owned(),
        ));
    }

    compute(TrailingCall {
        x,
        cos,
        sin,
        inverse,
        first: (head_dim - rotated) as usize,
    })
}

/// A checked call of [`rotary_trailing_`]: the rotation starts at channel `first`.
struct TrailingCall<'t> {
    x: &'t Tensor,
    cos: &'t Tensor,
    sin: &'t Tensor,
    inverse: bool,
    first: usize,
}

impl TrailingCall<'_> {
    const OWNER: &'static str = "rotary_trailing";
}

impl Fused for TrailingCall<'_> {
    type Output = ();

    fn device(&self) -> &Device {
        self.x.device()
    }

    fn cpu(self) -> Result<()> {
        let Self {
            x,
            cos,
            sin,
            inverse,
            first,
        } = self;
        let (tokens, heads, head_dim) = (x.dims()[0], x.dims()[1], x.dims()[2]);
        let pairs = cos.dims()[1];

        // `[tokens, heads, pairs, 2]`: each adjacent pair of the trailing channels.
        let tail = x
            .narrow(D::Minus1, first, head_dim - first)?
            .to_dtype(ACCUMULATE)?
            .reshape((tokens, heads, pairs, 2))?;
        let (a, b) = (tail.narrow(3, 0, 1)?, tail.narrow(3, 1, 1)?);
        let cos = cos.reshape((tokens, 1, pairs, 1))?;
        let sin = if inverse {
            sin.neg()?.reshape((tokens, 1, pairs, 1))?
        } else {
            sin.reshape((tokens, 1, pairs, 1))?
        };
        let turned_a = (a.broadcast_mul(&cos)? - b.broadcast_mul(&sin)?)?;
        let turned_b = (b.broadcast_mul(&cos)? + a.broadcast_mul(&sin)?)?;
        let turned = Tensor::cat(&[turned_a, turned_b], 3)?
            .reshape((tokens, heads, head_dim - first))?
            .to_dtype(x.dtype())?;

        x.slice_set(&turned, 2, first)?;
        Ok(())
    }

    fn launch(self) -> Result<()> {
        let Self {
            x,
            cos,
            sin,
            inverse,
            ..
        } = self;
        let args = RopeApplyArgs {
            x: ops::operand(x)?,
            cosine: ops::operand(cos)?,
            sine: ops::operand(sin)?,
            out: ops::operand(x)?,
        };
        let descriptor = RopeApplyDescriptor::of(&args, Rotation::Interleaved)?.trailing(inverse);
        ops::launch::<RopeApply>(&descriptor, args)?;
        Ok(())
    }
}

/// The per-head RMS normalization of an attention's queries and keys fused with their rotation:
/// `(query, key)`, each `rotate(x · rsqrt(mean(x²) + eps) · gain)` in its input's shape and
/// element type.
///
/// `query` is `[tokens, heads, head_dim]` and `key` `[tokens, kv_heads, head_dim]`; the mean is
/// over a head's channels. `weights` are the learned per-head gains of the queries and of the
/// keys, `[head_dim]` each: the gain is the weight, or one plus it when the checkpoint stores
/// it zero-centred (`centred`), the one added in FP32. Without weights the normalization has no
/// gain, which is another kernel. `cos` and `sin` are FP32 `[tokens, pairs]`.
///
/// # Errors
///
/// Queries and keys of other token counts or head widths, weights of another width, tables
/// that are not one FP32 row per token or are wider than half the head, or the operator's
/// refusal.
pub fn qk_norm_rope(
    query: &Tensor,
    key: &Tensor,
    weights: Option<(&Tensor, &Tensor)>,
    cos: &Tensor,
    sin: &Tensor,
    eps: f32,
    centred: bool,
) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = QkNormRopeCall::OWNER;

    let (_, head_dim) = heads_of(OWNER, "the query", query)?;
    let (_, key_dim) = heads_of(OWNER, "the key", key)?;
    let tokens = query.dims()[0];
    if key.dims()[0] != tokens || key_dim != head_dim || key.dtype() != query.dtype() {
        return Err(Error::shape(
            OWNER,
            format!(
                "queries of {} {:?} and keys of {} {:?}: one token count, head width and element \
                 type expected",
                ops::abi::name(query.dtype()),
                query.dims(),
                ops::abi::name(key.dtype()),
                key.dims()
            ),
        ));
    }
    rotated(OWNER, cos, sin, head_dim)?;
    per_token(OWNER, cos, tokens)?;
    if let Some((query_weight, key_weight)) = weights {
        vector(OWNER, "query gain", query_weight, head_dim)?;
        vector(OWNER, "key gain", key_weight, head_dim)?;
    }

    // Only a learned gain can be stored zero-centred.
    let centred = weights.is_some() && centred;

    compute(QkNormRopeCall {
        query,
        key,
        weights,
        cos,
        sin,
        eps,
        centred,
    })
}

/// A checked call of [`qk_norm_rope`]: `centred` only with learned gains.
struct QkNormRopeCall<'t> {
    query: &'t Tensor,
    key: &'t Tensor,
    weights: Option<(&'t Tensor, &'t Tensor)>,
    cos: &'t Tensor,
    sin: &'t Tensor,
    eps: f32,
    centred: bool,
}

impl QkNormRopeCall<'_> {
    const OWNER: &'static str = "qk_norm_rope";
}

impl Fused for QkNormRopeCall<'_> {
    type Output = (Tensor, Tensor);

    fn device(&self) -> &Device {
        self.query.device()
    }

    fn cpu(self) -> Result<(Tensor, Tensor)> {
        let Self {
            query,
            key,
            weights,
            cos,
            sin,
            eps,
            centred,
        } = self;
        let turned = |x: &Tensor, weight: Option<&Tensor>| -> Result<Tensor> {
            let mut normalized = rms_scaled(&x.to_dtype(ACCUMULATE)?, eps)?;
            if let Some(weight) = weight {
                normalized = normalized.broadcast_mul(&gain(weight, centred)?)?;
            }
            Ok(rotate(&normalized, cos, sin)?.to_dtype(x.dtype())?)
        };

        Ok((
            turned(query, weights.map(|(query, _)| query))?,
            turned(key, weights.map(|(_, key)| key))?,
        ))
    }

    fn launch(self) -> Result<(Tensor, Tensor)> {
        let Self {
            query,
            key,
            weights,
            cos,
            sin,
            ..
        } = self;
        let query_out = ops::empty(query.shape(), query.dtype())?;
        let key_out = ops::empty(key.shape(), key.dtype())?;

        // With learned gains the kernel loads them; without, it has no weight arguments.
        let args = QkNormRopeArgs {
            query: ops::operand(query)?,
            key: ops::operand(key)?,
            q_weight: weights.map(|(query, _)| ops::operand(query)).transpose()?,
            k_weight: weights.map(|(_, key)| ops::operand(key)).transpose()?,
            cosine: ops::operand(cos)?,
            sine: ops::operand(sin)?,
            q_out: ops::operand(&query_out)?,
            k_out: ops::operand(&key_out)?,
        };
        ops::launch::<QkNormRope>(
            &QkNormRopeDescriptor::of(&args, Rotation::Half, self.eps, self.centred, false)?,
            args,
        )?;
        Ok((query_out, key_out))
    }
}

/// [`qk_norm_rope`] without a gain, in place over the queries and keys of a merged projection:
/// returns `projected` with the heads of both ranges normalized and turned where they lie.
///
/// `projected` is `[tokens, width]`, contiguous, each row a whole number of `head_dim`-wide
/// heads; `query` and `key` are two ranges of those heads, of one length. The factor tables are
/// `[rows, pairs]` of an element type the caller chose, and token `t` reads row `t mod rows`,
/// so tiles computed in one launch share one table.
///
/// # Errors
///
/// Rows that are not whole heads, ranges that are not two equal runs of the row's heads, a
/// table the tokens do not repeat, a strided `projected` (the heads are turned in its own
/// storage), or the operator's refusal.
pub fn qk_norm_rope_(
    projected: Tensor,
    query: Range<usize>,
    key: Range<usize>,
    head_dim: u32,
    cos: &Tensor,
    sin: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    const OWNER: &str = QkNormRopeUpdateCall::OWNER;

    let width = head_dim as usize;
    let (tokens, heads_per_row) = match *projected.dims() {
        [tokens, row] if width > 0 && row.is_multiple_of(width) => (tokens, row / width),
        _ => {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a projection of {:?} is not rows of whole {head_dim}-channel heads",
                    projected.dims()
                ),
            ));
        }
    };
    let heads = query.len();
    if heads == 0 || key.len() != heads || query.end > heads_per_row || key.end > heads_per_row {
        return Err(Error::shape(
            OWNER,
            format!(
                "heads {query:?} and {key:?} of rows of {heads_per_row}: two runs of one length \
                 expected"
            ),
        ));
    }
    if !projected.is_contiguous() {
        return Err(Error::shape(
            OWNER,
            "a strided projection has no storage of its own to turn its heads in".to_owned(),
        ));
    }
    rotated(OWNER, cos, sin, head_dim)?;
    let table = cos.dims()[0];
    if table == 0 || !tokens.is_multiple_of(table) {
        return Err(Error::shape(
            OWNER,
            format!("{tokens} tokens do not repeat a factor table of {table} rows"),
        ));
    }

    compute(QkNormRopeUpdateCall {
        projected,
        query,
        key,
        width,
        cos,
        sin,
        eps,
    })
}

/// A checked call of [`qk_norm_rope_`]: the ranges are runs of heads of `width` channels of the
/// projection's rows, which the factor tables' rows divide.
struct QkNormRopeUpdateCall<'t> {
    projected: Tensor,
    query: Range<usize>,
    key: Range<usize>,
    width: usize,
    cos: &'t Tensor,
    sin: &'t Tensor,
    eps: f32,
}

impl QkNormRopeUpdateCall<'_> {
    const OWNER: &'static str = "qk_norm_rope_";

    /// The heads of both ranges, as views of the projection's own rows.
    fn heads(&self) -> Result<(Tensor, Tensor)> {
        let (tokens, row) = self.projected.dims2()?;
        let by_head = self
            .projected
            .reshape((tokens, row / self.width, self.width))?;

        Ok((
            by_head.narrow(1, self.query.start, self.query.len())?,
            by_head.narrow(1, self.key.start, self.key.len())?,
        ))
    }
}

impl Fused for QkNormRopeUpdateCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.projected.device()
    }

    fn cpu(self) -> Result<Tensor> {
        let (query_heads, key_heads) = self.heads()?;
        let Self {
            projected,
            query,
            key,
            width,
            cos,
            sin,
            eps,
        } = self;
        let tokens = projected.dims()[0];
        let (heads, table) = (query.len(), cos.dims()[0]);
        let tile = |factors: &Tensor| -> Result<Tensor> {
            Ok(factors.to_dtype(ACCUMULATE)?.repeat((tokens / table, 1))?)
        };
        let (cos, sin) = (tile(cos)?, tile(sin)?);

        for (heads_view, first) in [(&query_heads, query.start), (&key_heads, key.start)] {
            let normalized = rms_scaled(&heads_view.to_dtype(ACCUMULATE)?, eps)?;
            let turned = rotate(&normalized, &cos, &sin)?.to_dtype(projected.dtype())?;

            // The columns of the range, in every row: the turned heads replace them.
            let columns = heads * width;
            let index = (Tensor::arange(0_u32, columns as u32, &Device::Cpu)?
                + (first * width) as f64)?
                .unsqueeze(0)?
                .broadcast_as((tokens, columns))?
                .contiguous()?;
            projected.scatter_set(&index, &turned.reshape((tokens, columns))?, 1)?;
        }
        Ok(projected)
    }

    fn launch(self) -> Result<Tensor> {
        let (query_heads, key_heads) = self.heads()?;

        // Each kernel owns some tokens of one head, the queries' heads and the keys' alike, and
        // the outputs are the inputs.
        let (query_operand, key_operand) = (ops::operand(&query_heads)?, ops::operand(&key_heads)?);
        let args = QkNormRopeArgs {
            query: query_operand.clone(),
            key: key_operand.clone(),
            q_weight: None,
            k_weight: None,
            cosine: ops::operand(self.cos)?,
            sine: ops::operand(self.sin)?,
            q_out: query_operand,
            k_out: key_operand,
        };
        ops::launch::<QkNormRope>(
            &QkNormRopeDescriptor::of(&args, Rotation::Half, self.eps, false, true)?,
            args,
        )?;
        Ok(self.projected)
    }
}

/// The rotary coordinates of every cell of a grid of `extents` cells per axis: an
/// `[axes, cells]` FP32 CPU tensor, axis-major.
///
/// Cells run in raster order, the first axis slowest. Cell `i` of an axis of `n` cells sits at
/// its centre in `[-1, 1)`, and its coordinate is that centre times `scale`, evaluated in FP32
/// as `scale · (2 · ((i + 0.5) / n) − 1)`: inverse frequencies then turn a cell by `scale ·
/// centre · frequency`, so a scale of `2π` gives the angle of a reference that multiplies its
/// grid by `2π` where it rotates. Axis-major because a rotation of several axes reads one
/// leading coordinate per axis ([`rotary_factors`]).
///
/// # Errors
///
/// A grid with no axis or an axis of no cells.
pub fn grid_positions(extents: &[u32], scale: f32) -> Result<Tensor> {
    if extents.is_empty() || extents.contains(&0) {
        return Err(Error::shape(
            "grid_positions",
            format!("a grid of {extents:?} cells has no axis or an empty one"),
        ));
    }
    let grid: Vec<usize> = extents.iter().map(|extent| *extent as usize).collect();
    let cells: usize = grid.iter().product();

    let axes = grid
        .iter()
        .enumerate()
        .map(|(axis, &size)| -> Result<Tensor> {
            // The centres of the axis's cells, divided in FP32, then scaled.
            let centre = (Tensor::arange(0_f32, size as f32, &Device::Cpu)? + 0.5)?
                .broadcast_div(&Tensor::new(size as f32, &Device::Cpu)?)?;
            let coordinate = centre.affine(2.0, -1.0)?.affine(f64::from(scale), 0.0)?;

            // Along its own axis of the raster, and the same in every other.
            let mut along = vec![1; grid.len()];
            along[axis] = size;
            Ok(coordinate
                .reshape(along)?
                .broadcast_as(grid.as_slice())?
                .reshape(cells)?)
        })
        .collect::<Result<Vec<Tensor>>>()?;

    Ok(Tensor::stack(&axes, 0)?)
}
