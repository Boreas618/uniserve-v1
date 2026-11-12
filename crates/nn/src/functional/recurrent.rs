//! The recurrent operators of a gated delta network: the short causal convolution and the gated
//! delta rule, each advancing a state the batch's request slots keep.
//!
//! A gated delta network (Yang et al., "Gated Delta Networks", 2024; Qwen3-Next and Qwen3.5)
//! mixes tokens through a depthwise causal convolution of a few taps and a matrix-valued state
//! updated by the delta rule under a scalar decay per value head:
//!
//! ```text
//! causal_conv1d      out[t, c] = silu(Σ_j w[c, j] · x[t − (K − 1) + j, c])           j < K
//! gated_delta_rule   g = −exp(A_log) · softplus(a + dt_bias),   β = sigmoid(b)
//!                    q̂ = q · rsqrt(Σ q² + 1e-6) / √d_k,         k̂ = k · rsqrt(Σ k² + 1e-6)
//!                    S ← exp(g) · S;   S ← S + k̂ ⊗ β (v − k̂ᵀ S);   out = q̂ᵀ S
//! ```
//!
//! Every sum, the state and every intermediate are FP32, and each output rounds once into its
//! input's element type.
//!
//! ## Rows, slots and checkpoints
//!
//! Both read the state rows of a [`RecurrentState`]: `query_offsets` packs each row's new
//! tokens and `seq_lens` counts every token the row has seen, so a row's new tokens sit at the
//! absolute positions `seq_lens[r] − count[r] .. seq_lens[r]`. A row that starts at position
//! zero begins from a zero state; every other row continues from the state its slot holds in
//! the live buffer, and leaves there the state after its last token. A row whose slot is
//! [`NO_SLOT`] is padding: it reads and writes no state and its output rows are zeros, so a
//! bucket wider than its batch holds finite values.
//!
//! A row whose `checkpoints` entry lies within its new tokens (`start < checkpoint ≤ end`)
//! also leaves the state after the token at `checkpoint − 1` in its slot of the checkpoint
//! buffer, for the prefix cache. A batch that keeps a trail has each function write the state
//! after **every** token as well, row `t` of the trail buffer for the batch's token `t`.
//!
//! The buffers are execution's, lent for the forward and written in place; a function reads
//! and writes one field of one recurrent layer of them.

use std::ops::Range;

use ops::recurrent::{
    CausalConv1d, CausalConv1dArgs, CausalConv1dDescriptor, Form, GatedDeltaRule,
    GatedDeltaRuleArgs, GatedDeltaRuleDescriptor, RowColumns, Trail,
};
use proto::module::PhaseKind;
use uniserve_tensor::{D, DType, Device, Tensor};

use super::{ACCUMULATE, on_cpu, vector};
use crate::capability::{NO_SLOT, RecurrentState, TokenBatch};
use crate::error::{Error, Result};

/// Channels one `causal_conv1d` kernel covers.
const CONV_BLOCK: u32 = 256;

/// Tokens of one chunk of the chunked delta rule: a cache page holds a whole number of them,
/// so a checkpoint at a page boundary lands on a chunk end.
pub const DELTA_RULE_CHUNK: u32 = 64;

/// Value columns one `gated_delta_rule` kernel owns of a head's state; the chunked form's
/// tensor memory holds `2 · chunk + 4 · block` columns, which bounds it at 64.
const DELTA_RULE_BLOCK: u32 = 64;

/// The floor under the square root of a query's or a key's normalization.
const L2_EPS: f64 = 1e-6;

/// The input above which the softplus is its argument, as PyTorch evaluates it.
const SOFTPLUS_LINEAR: f64 = 20.0;

/// The buffer of field `field` of recurrent layer `layer` among `buffers`, a state's live,
/// checkpoint or trail buffers.
fn buffer<'s>(
    owner: &'static str,
    buffers: &'s [Vec<Tensor>],
    layer: usize,
    field: usize,
) -> Result<&'s Tensor> {
    buffers
        .get(layer)
        .and_then(|fields| fields.get(field))
        .ok_or_else(|| {
            Error::shape(
                owner,
                format!(
                    "field {field} of recurrent layer {layer} was asked for, and the state holds \
                     {} layers",
                    buffers.len()
                ),
            )
        })
}

/// Checks that `buffer`, the `name` buffer of a field, holds `dtype` rows of `dims`, and
/// returns its row count.
fn rows_of(
    owner: &'static str,
    name: &str,
    buffer: &Tensor,
    dtype: DType,
    dims: &[usize],
) -> Result<usize> {
    if buffer.rank() != dims.len() + 1 || buffer.dims()[1..] != *dims || buffer.dtype() != dtype {
        return Err(Error::shape(
            owner,
            format!(
                "the {name} buffer is {} {:?}, {} [rows, {dims:?}] expected",
                ops::abi::name(buffer.dtype()),
                buffer.dims(),
                ops::abi::name(dtype)
            ),
        ));
    }
    Ok(buffer.dims()[0])
}

/// Checks the three buffers of one field against `dims`, the extents of one slot's state, in
/// `dtype`: as many checkpoint slots as live ones, and a trail row for each of `tokens` tokens.
fn check_buffers(
    owner: &'static str,
    (live, checkpoint, trail): (&Tensor, &Tensor, Option<&Tensor>),
    dtype: DType,
    dims: &[usize],
    tokens: usize,
) -> Result<()> {
    let slots = rows_of(owner, "live", live, dtype, dims)?;
    let kept = rows_of(owner, "checkpoint", checkpoint, dtype, dims)?;
    if kept != slots {
        return Err(Error::shape(
            owner,
            format!("{kept} checkpoint slots for {slots} live slots"),
        ));
    }

    if let Some(trail) = trail
        && rows_of(owner, "trail", trail, dtype, dims)? < tokens
    {
        return Err(Error::shape(
            owner,
            format!("a trail of {} rows for {tokens} tokens", trail.dims()[0]),
        ));
    }
    Ok(())
}

/// The row columns a recurrent kernel reads: the state's rows.
fn row_columns(state: &RecurrentState) -> Result<RowColumns> {
    Ok(RowColumns {
        query_offsets: backend::operand(&state.query_offsets)?,
        seq_lens: backend::operand(&state.seq_lens)?,
        slots: backend::operand(&state.slots)?,
        checkpoints: backend::operand(&state.checkpoints)?,
    })
}

/// The trail argument of a launch.
fn trail_operand(trail: Option<&Tensor>) -> Result<Option<Trail>> {
    Ok(trail
        .map(backend::operand)
        .transpose()?
        .map(|bank| Trail { bank }))
}

/// The values of `column`, a U32 index column of a CPU state.
fn column(column: &Tensor) -> Result<Vec<u32>> {
    Ok(column.flatten_all()?.to_vec1::<u32>()?)
}

/// Runs `advance(tokens, slot, start, checkpoint)` over each state row of a CPU `state` that
/// names a slot, in row order: `tokens` are the row's packed tokens, `start` the absolute
/// position of the first of them, and `checkpoint` the absolute token count the row keeps its
/// state at, zero for none.
fn for_each_row(
    owner: &'static str,
    state: &RecurrentState,
    mut advance: impl FnMut(Range<usize>, usize, usize, usize) -> Result<()>,
) -> Result<()> {
    let offsets = column(&state.query_offsets)?;
    let seq_lens = column(&state.seq_lens)?;
    let slots = column(&state.slots)?;
    let checkpoints = column(&state.checkpoints)?;

    let rows = seq_lens.len();
    if offsets.len() != rows + 1 || slots.len() != rows || checkpoints.len() != rows {
        return Err(Error::shape(
            owner,
            format!(
                "{} offsets, {} slots and {} checkpoints for {rows} state rows",
                offsets.len(),
                slots.len(),
                checkpoints.len()
            ),
        ));
    }

    for row in 0..rows {
        let tokens = offsets[row] as usize..offsets[row + 1] as usize;
        if slots[row] == NO_SLOT || tokens.is_empty() {
            continue;
        }

        let start = (seq_lens[row] as usize).checked_sub(tokens.len());
        let start = start.ok_or_else(|| {
            Error::shape(
                owner,
                format!(
                    "state row {row} has seen {} tokens and brings {}",
                    seq_lens[row],
                    tokens.len()
                ),
            )
        })?;
        advance(
            tokens,
            slots[row] as usize,
            start,
            checkpoints[row] as usize,
        )?;
    }
    Ok(())
}

/// The depthwise causal convolution of `x`, `[tokens, channels]`, with its SiLU, continuing
/// each state row from its slot's last inputs: `[tokens, channels]` in `x`'s element type.
///
/// `weight` is `[channels, width]`, each channel's taps with the oldest input's first. The
/// field `field` of recurrent layer `layer` of `state` holds each slot's last `width − 1`
/// inputs, `[slots, width − 1, channels]` in `x`'s element type, oldest first; it is read for
/// a row that continues and written in place with the row's last inputs, the checkpoint buffer
/// with the inputs before a row's checkpoint, and the trail with the inputs up to and including
/// each token (module documentation).
///
/// # Errors
///
/// Taps that are not a matrix over `x`'s channels, a state that does not hold the field or
/// holds another extent or element type, or the operator's refusal.
pub fn causal_conv1d(
    x: &Tensor,
    weight: &Tensor,
    state: &RecurrentState,
    layer: usize,
    field: usize,
) -> Result<Tensor> {
    const OWNER: &str = "causal_conv1d";

    let (&[tokens, channels], &[tap_channels, width]) = (x.dims(), weight.dims()) else {
        return Err(Error::shape(
            OWNER,
            format!(
                "inputs of {:?} and taps of {:?}: [tokens, channels] and [channels, width] \
                 expected",
                x.dims(),
                weight.dims()
            ),
        ));
    };
    if tap_channels != channels || width < 2 || channels == 0 {
        return Err(Error::shape(
            OWNER,
            format!(
                "taps of {:?} over {channels} channels: one row of at least two taps per \
                 channel expected",
                weight.dims()
            ),
        ));
    }

    let live = buffer(OWNER, &state.live, layer, field)?;
    let checkpoint = buffer(OWNER, &state.checkpoint, layer, field)?;
    let trail = match &state.trail {
        Some(trail) => Some(buffer(OWNER, trail, layer, field)?),
        None => None,
    };
    check_buffers(
        OWNER,
        (live, checkpoint, trail),
        x.dtype(),
        &[width - 1, channels],
        tokens,
    )?;

    if on_cpu(x) {
        return conv_formula(x, weight, state, (live, checkpoint, trail));
    }

    let out = backend::empty((tokens, channels), x.dtype())?;
    backend::launch::<CausalConv1d>(
        &CausalConv1dDescriptor {
            dtype: x.dtype(),
            channels: channels as u32,
            width: width as u32,
            block: CONV_BLOCK.min((channels as u32).next_power_of_two()),
        },
        CausalConv1dArgs {
            x: backend::operand(x)?,
            weight: backend::operand(weight)?,
            out: backend::operand(&out)?,
            rows: row_columns(state)?,
            state: backend::operand(live)?,
            checkpoint: backend::operand(checkpoint)?,
            trail: trail_operand(trail)?,
        },
    )?;
    Ok(out)
}

/// [`causal_conv1d`] on CPU tensors: `buffers` are the field's live, checkpoint and trail
/// buffers.
fn conv_formula(
    x: &Tensor,
    weight: &Tensor,
    state: &RecurrentState,
    (live, checkpoint, trail): (&Tensor, &Tensor, Option<&Tensor>),
) -> Result<Tensor> {
    let (tokens, channels) = x.dims2()?;
    let width = weight.dims()[1];
    let kept = width - 1;
    let taps = weight.to_dtype(ACCUMULATE)?;
    let out = Tensor::zeros((tokens, channels), x.dtype(), &Device::Cpu)?;

    for_each_row("causal_conv1d", state, |row, slot, start, keep| {
        let count = row.len();

        // The inputs the row's taps read: the slot's last inputs, or zeros for a row that
        // starts its sequence, then the row's own.
        let before = if start == 0 {
            Tensor::zeros((kept, channels), x.dtype(), &Device::Cpu)?
        } else {
            live.get(slot)?
        };
        let window = Tensor::cat(&[before, x.narrow(0, row.start, count)?], 0)?;
        let inputs = window.to_dtype(ACCUMULATE)?;

        let mut sum = Tensor::zeros((count, channels), ACCUMULATE, &Device::Cpu)?;
        for tap in 0..width {
            let weights = taps.narrow(1, tap, 1)?.t()?;
            sum = (sum + inputs.narrow(0, tap, count)?.broadcast_mul(&weights)?)?;
        }
        out.slice_set(&sum.silu()?.to_dtype(x.dtype())?, 0, row.start)?;

        // The window after `seen` of the row's tokens is the inputs a later token reads.
        let after = |seen: usize| -> Result<Tensor> {
            Ok(window.narrow(0, seen, kept)?.unsqueeze(0)?.contiguous()?)
        };
        if let Some(trail) = trail {
            for token in 0..count {
                trail.slice_set(&after(token + 1)?, 0, row.start + token)?;
            }
        }
        if keep > start && keep <= start + count {
            checkpoint.slice_set(&after(keep - start)?, 0, slot)?;
        }
        live.slice_set(&after(count)?, 0, slot)?;
        Ok(())
    })?;
    Ok(out)
}

/// `x`, FP32 `[..., width]`, scaled to unit length along its last axis: `x · rsqrt(Σ x² +
/// 1e-6)`.
fn l2_normalized(x: &Tensor) -> Result<Tensor> {
    let length = (x.sqr()?.sum_keepdim(D::Minus1)? + L2_EPS)?.sqrt()?;

    Ok(x.broadcast_div(&length)?)
}

/// `heads`, FP32 `[tokens, key_heads, width]`, with each key head repeated for the `group`
/// consecutive value heads it serves: `[tokens, key_heads · group, width]`.
fn per_value_head(heads: &Tensor, group: usize) -> Result<Tensor> {
    let (tokens, key_heads, width) = heads.dims3()?;

    Ok(heads
        .unsqueeze(2)?
        .broadcast_as((tokens, key_heads, group, width))?
        .reshape((tokens, key_heads * group, width))?)
}

/// The gated delta rule over `query` and `key`, `[tokens, key_heads, key_dim]`, and `value`,
/// `[tokens, value_heads, value_dim]`, each state row continuing from its slot's state:
/// `[tokens, value_heads, value_dim]` in the query's element type.
///
/// `a` and `b`, `[tokens, value_heads]`, are the inputs of the decay and of the step size, and
/// `a_log` and `dt_bias`, `[value_heads]`, each head's log decay rate and time-step bias. A key
/// head serves `value_heads / key_heads` consecutive value heads. The field `field` of
/// recurrent layer `layer` of `state` holds each slot's FP32 state, `[slots, value_heads,
/// key_dim, value_dim]`, read and written in place with its checkpoint and trail buffers
/// (module documentation).
///
/// A decode-phase batch (a decode, or a verification) advances its rows one token at a time
/// and keeps the trail the state has; any other advances them in chunks of
/// [`DELTA_RULE_CHUNK`] tokens at absolute positions, each solved at once, which keeps no trail
/// and puts its checkpoints at chunk ends.
///
/// # Errors
///
/// Heads of other token counts or widths, value heads the key heads do not group, gate inputs
/// or parameters that are not one per value head, a state that does not hold the field or
/// holds another extent, a trail kept by a batch that advances in chunks, a page that is not
/// whole chunks, or the operator's refusal.
#[expect(
    clippy::too_many_arguments,
    reason = "the operands of the rule, as the reference's kernel takes them"
)]
pub fn gated_delta_rule(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    batch: &TokenBatch,
    state: &RecurrentState,
    layer: usize,
    field: usize,
) -> Result<Tensor> {
    const OWNER: &str = "gated_delta_rule";

    let (&[tokens, key_heads, key_dim], &[value_tokens, value_heads, value_dim]) =
        (query.dims(), value.dims())
    else {
        return Err(Error::shape(
            OWNER,
            format!(
                "queries of {:?} and values of {:?}: [tokens, heads, dim] expected of both",
                query.dims(),
                value.dims()
            ),
        ));
    };
    if key.dims() != query.dims()
        || value_tokens != tokens
        || key_heads == 0
        || !value_heads.is_multiple_of(key_heads)
        || key_dim == 0
        || value_dim == 0
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "queries of {:?}, keys of {:?} and values of {:?}: one token count, and value \
                 heads the key heads group",
                query.dims(),
                key.dims(),
                value.dims()
            ),
        ));
    }
    for (name, gate) in [("a", a), ("b", b)] {
        if gate.dims() != [tokens, value_heads] {
            return Err(Error::shape(
                OWNER,
                format!(
                    "{name} is {:?}, [{tokens}, {value_heads}] expected: one input per token \
                     and value head",
                    gate.dims()
                ),
            ));
        }
    }
    vector(OWNER, "A_log", a_log, value_heads as u32)?;
    vector(OWNER, "dt_bias", dt_bias, value_heads as u32)?;

    let live = buffer(OWNER, &state.live, layer, field)?;
    let checkpoint = buffer(OWNER, &state.checkpoint, layer, field)?;
    let trail = match &state.trail {
        Some(trail) => Some(buffer(OWNER, trail, layer, field)?),
        None => None,
    };
    check_buffers(
        OWNER,
        (live, checkpoint, trail),
        DType::F32,
        &[value_heads, key_dim, value_dim],
        tokens,
    )?;

    let (form, chunk) = match batch.phase {
        PhaseKind::Decode => (Form::Recurrent, 1),
        _ => (Form::Chunked, DELTA_RULE_CHUNK),
    };
    if form == Form::Chunked {
        if !batch.page_size.is_multiple_of(DELTA_RULE_CHUNK) {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a page of {} tokens is no whole number of {DELTA_RULE_CHUNK}-token chunks, \
                     so its checkpoints fall inside a chunk",
                    batch.page_size
                ),
            ));
        }
        if trail.is_some() {
            return Err(Error::shape(
                OWNER,
                "a batch that advances in chunks holds no state between its tokens, so it keeps \
                 no trail"
                    .to_owned(),
            ));
        }
    }

    if on_cpu(query) {
        return rule_formula(
            (query, key, value),
            (a, b),
            (a_log, dt_bias),
            state,
            (live, checkpoint, trail),
        );
    }

    let out = backend::empty((tokens, value_heads, value_dim), query.dtype())?;
    backend::launch::<GatedDeltaRule>(
        &GatedDeltaRuleDescriptor {
            dtype: query.dtype(),
            key_heads: key_heads as u32,
            value_heads: value_heads as u32,
            key_dim: key_dim as u32,
            value_dim: value_dim as u32,
            block: DELTA_RULE_BLOCK.min(value_dim as u32),
            form,
            chunk,
        },
        GatedDeltaRuleArgs {
            query: backend::operand(query)?,
            key: backend::operand(key)?,
            value: backend::operand(value)?,
            a: backend::operand(a)?,
            b: backend::operand(b)?,
            a_log: backend::operand(a_log)?,
            dt_bias: backend::operand(dt_bias)?,
            rows: row_columns(state)?,
            state: backend::operand(live)?,
            checkpoint: backend::operand(checkpoint)?,
            trail: trail_operand(trail)?,
            out: backend::operand(&out)?,
        },
    )?;
    Ok(out)
}

/// [`gated_delta_rule`] on CPU tensors, as the recurrence of the module documentation:
/// `heads` are the queries, keys and values, `gates` the inputs `a` and `b`, `parameters`
/// `A_log` and `dt_bias`, and `buffers` the field's live, checkpoint and trail buffers.
fn rule_formula(
    (query, key, value): (&Tensor, &Tensor, &Tensor),
    (a, b): (&Tensor, &Tensor),
    (a_log, dt_bias): (&Tensor, &Tensor),
    state: &RecurrentState,
    (live, checkpoint, trail): (&Tensor, &Tensor, Option<&Tensor>),
) -> Result<Tensor> {
    let (tokens, key_heads, key_dim) = query.dims3()?;
    let (_, value_heads, value_dim) = value.dims3()?;
    let group = value_heads / key_heads;
    let wide = |x: &Tensor| x.to_dtype(ACCUMULATE);

    // Every token's operands of the recurrence, per value head.
    let scale = f64::from((f64::from(key_dim as u32).sqrt().recip()) as f32);
    let queries = per_value_head(&l2_normalized(&wide(query)?)?.affine(scale, 0.0)?, group)?;
    let keys = per_value_head(&l2_normalized(&wide(key)?)?, group)?;
    let values = wide(value)?;

    let rate = wide(a)?.broadcast_add(&wide(dt_bias)?)?;
    let softplus = rate
        .gt(SOFTPLUS_LINEAR)?
        .where_cond(&rate, &(rate.exp()? + 1.0)?.log()?)?;
    let decay = softplus.broadcast_mul(&wide(a_log)?.exp()?)?.neg()?.exp()?;
    let step = (wide(b)?.neg()?.exp()? + 1.0)?.recip()?;

    let out = Tensor::zeros(
        (tokens, value_heads, value_dim),
        query.dtype(),
        &Device::Cpu,
    )?;
    let per_head = (value_heads, 1, 1);

    for_each_row("gated_delta_rule", state, |row, slot, start, keep| {
        let mut memory = if start == 0 {
            Tensor::zeros((value_heads, key_dim, value_dim), ACCUMULATE, &Device::Cpu)?
        } else {
            live.get(slot)?.copy()?
        };

        for (seen, token) in row.clone().enumerate() {
            let key = keys.get(token)?;

            // Decay, then the delta-rule step toward the token's value along its key.
            memory = memory.broadcast_mul(&decay.get(token)?.reshape(per_head)?)?;
            let recalled = key.unsqueeze(1)?.matmul(&memory)?;
            let error = (values.get(token)?.unsqueeze(1)? - recalled)?
                .broadcast_mul(&step.get(token)?.reshape(per_head)?)?;
            memory = (memory + key.unsqueeze(2)?.matmul(&error)?)?;

            let read = queries.get(token)?.unsqueeze(1)?.matmul(&memory)?;
            out.slice_set(
                &read
                    .reshape((1, value_heads, value_dim))?
                    .to_dtype(query.dtype())?,
                0,
                token,
            )?;

            let after = memory.unsqueeze(0)?;
            if let Some(trail) = trail {
                trail.slice_set(&after, 0, token)?;
            }
            if start + seen + 1 == keep {
                checkpoint.slice_set(&after, 0, slot)?;
            }
        }
        live.slice_set(&memory.unsqueeze(0)?, 0, slot)?;
        Ok(())
    })?;
    Ok(out)
}
