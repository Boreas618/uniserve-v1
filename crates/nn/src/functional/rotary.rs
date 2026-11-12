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
use uniserve_tensor::{D, DType, Device, Tensor};

use super::linear::{integer, remainder};
use super::norm::{gain, rms_scaled};
use super::{ACCUMULATE, PDL, on_cpu, vector};
use crate::error::{Error, Result};

/// Elements one `rotary_factors` kernel handles.
const FACTORS_BLOCK: u32 = 256;

/// Head rows one `qk_norm_rope` kernel handles: a token holds one per head, so a step of few
/// tokens still spreads over the device.
const QK_NORM_ROWS: u32 = 16;

/// The tokens one kernel of the in-place q/k rotation normalizes, per head.
const IN_PLACE_TOKENS: u32 = 8;

/// The tokens one `rope_apply` kernel rotates.
const APPLY_TOKENS: u32 = 8;

/// The head elements one `rope_apply` kernel loads at a time.
const APPLY_BLOCK: u32 = 128;

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
        let cycle = remainder(&frequency, axes as u32)?;
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
    const OWNER: &str = "rotary_factors";

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
    let tokens = positions.elem_count() / axes;

    if on_cpu(positions) {
        let coordinates = positions.to_dtype(ACCUMULATE)?.reshape((axes, tokens))?;

        // Each frequency's coordinate of each token, `[tokens, columns]`.
        let per_frequency = if sections.is_empty() {
            coordinates.t()?
        } else {
            coordinates
                .index_select(&frequency_axes(sections, columns, interleaved)?, 0)?
                .t()?
        };
        let angle = per_frequency.broadcast_mul(&inv_freq.to_dtype(ACCUMULATE)?.unsqueeze(0)?)?;

        return Ok((angle.cos()?, angle.sin()?));
    }

    let cos = backend::empty((tokens, columns), DType::F32)?;
    let sin = backend::empty((tokens, columns), DType::F32)?;

    let mut owned = [0; MAX_AXES];
    owned[..sections.len()].copy_from_slice(sections);
    backend::launch::<RotaryFactors>(
        &FactorsDescriptor {
            dtype: DType::F32,
            positions: positions.dtype(),
            columns: columns as u32,
            block: FACTORS_BLOCK,
            scale: 1.0_f32.into(),
            pdl: PDL,
            axes: axes as u32,
            sections: owned,
            interleaved: !sections.is_empty() && interleaved,
        },
        FactorsArgs {
            positions: backend::operand(positions)?,
            frequencies: backend::operand(inv_freq)?,
            cosine: backend::operand(&cos)?,
            sine: backend::operand(&sin)?,
        },
    )?;
    Ok((cos, sin))
}

/// `x`, `[tokens, heads, head_dim]`, turned by the factors `cos` and `sin`, FP32 `[tokens,
/// pairs]`: the rotation alone, in FP32, rounded once into `x`'s element type.
///
/// # Errors
///
/// Tables that are not one FP32 row per token or are wider than half the head, or the
/// operator's refusal.
pub fn rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    const OWNER: &str = "rotary";

    let (heads, head_dim) = heads_of(OWNER, "the input", x)?;
    let rotated = rotated(OWNER, cos, sin, head_dim)?;
    per_token(OWNER, cos, x.dims()[0])?;

    if on_cpu(x) {
        return Ok(rotate(&x.to_dtype(ACCUMULATE)?, cos, sin)?.to_dtype(x.dtype())?);
    }

    let out = backend::empty(x.shape(), x.dtype())?;
    backend::launch::<RopeApply>(
        &RopeApplyDescriptor {
            dtype: x.dtype(),
            output: out.dtype(),
            head_dim,
            heads,
            rotation: Rotation::Half,
            rotated,
            rows: APPLY_TOKENS,
            block: APPLY_BLOCK,
            pdl: PDL,
        },
        RopeApplyArgs {
            x: backend::operand(x)?,
            cosine: backend::operand(cos)?,
            sine: backend::operand(sin)?,
            out: backend::operand(&out)?,
        },
    )?;
    Ok(out)
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
    const OWNER: &str = "qk_norm_rope";

    let (heads, head_dim) = heads_of(OWNER, "the query", query)?;
    let (kv_heads, key_dim) = heads_of(OWNER, "the key", key)?;
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
    let rotated = rotated(OWNER, cos, sin, head_dim)?;
    per_token(OWNER, cos, tokens)?;
    if let Some((query_weight, key_weight)) = weights {
        vector(OWNER, "query gain", query_weight, head_dim)?;
        vector(OWNER, "key gain", key_weight, head_dim)?;
    }

    // Only a learned gain can be stored zero-centred.
    let centred = weights.is_some() && centred;

    if on_cpu(query) {
        let turned = |x: &Tensor, weight: Option<&Tensor>| -> Result<Tensor> {
            let mut normalized = rms_scaled(&x.to_dtype(ACCUMULATE)?, eps)?;
            if let Some(weight) = weight {
                normalized = normalized.broadcast_mul(&gain(weight, centred)?)?;
            }
            Ok(rotate(&normalized, cos, sin)?.to_dtype(x.dtype())?)
        };

        return Ok((
            turned(query, weights.map(|(query, _)| query))?,
            turned(key, weights.map(|(_, key)| key))?,
        ));
    }

    let query_out = backend::empty(query.shape(), query.dtype())?;
    let key_out = backend::empty(key.shape(), key.dtype())?;
    backend::launch::<QkNormRope>(
        // With learned gains the kernel loads them; without, it has no weight arguments.
        &QkNormRopeDescriptor {
            dtype: query.dtype(),
            output: query_out.dtype(),
            head_dim,
            q_heads: heads,
            k_heads: kv_heads,
            rotation: Rotation::Half,
            epsilon: eps.into(),
            rotated,
            rows: QK_NORM_ROWS,
            block: head_tile(head_dim),
            affine: weights.is_some(),
            pdl: PDL,
            factors: DType::F32,
            strided: false,
            centred,
        },
        QkNormRopeArgs {
            query: backend::operand(query)?,
            key: backend::operand(key)?,
            q_weight: weights
                .map(|(query, _)| backend::operand(query))
                .transpose()?,
            k_weight: weights.map(|(_, key)| backend::operand(key)).transpose()?,
            cosine: backend::operand(cos)?,
            sine: backend::operand(sin)?,
            q_out: backend::operand(&query_out)?,
            k_out: backend::operand(&key_out)?,
        },
    )?;
    Ok((query_out, key_out))
}

/// The head tile one `qk_norm_rope` kernel loads at a time.
fn head_tile(head_dim: u32) -> u32 {
    head_dim.next_power_of_two().min(1024)
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
    const OWNER: &str = "qk_norm_rope_";

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
    let rotated = rotated(OWNER, cos, sin, head_dim)?;
    let table = cos.dims()[0];
    if table == 0 || !tokens.is_multiple_of(table) {
        return Err(Error::shape(
            OWNER,
            format!("{tokens} tokens do not repeat a factor table of {table} rows"),
        ));
    }

    // The heads of both ranges, as views of the projection's own rows.
    let by_head = projected.reshape((tokens, heads_per_row, width))?;
    let query_heads = by_head.narrow(1, query.start, heads)?;
    let key_heads = by_head.narrow(1, key.start, heads)?;

    if on_cpu(&projected) {
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
        return Ok(projected);
    }

    let (query_operand, key_operand) = (
        backend::operand(&query_heads)?,
        backend::operand(&key_heads)?,
    );
    backend::launch::<QkNormRope>(
        // Each kernel owns some tokens of one head, the queries' heads and the keys' alike, and
        // the outputs are the inputs.
        &QkNormRopeDescriptor {
            dtype: projected.dtype(),
            output: projected.dtype(),
            head_dim,
            q_heads: heads as u32,
            k_heads: heads as u32,
            rotation: Rotation::Half,
            epsilon: eps.into(),
            rotated,
            rows: IN_PLACE_TOKENS,
            block: head_tile(head_dim),
            affine: false,
            pdl: PDL,
            factors: cos.dtype(),
            strided: true,
            centred: false,
        },
        QkNormRopeArgs {
            query: query_operand,
            key: key_operand,
            q_weight: None,
            k_weight: None,
            cosine: backend::operand(cos)?,
            sine: backend::operand(sin)?,
            q_out: query_operand,
            k_out: key_operand,
        },
    )?;
    Ok(projected)
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
