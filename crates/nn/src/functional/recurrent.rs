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
//! ## Request slots
//!
//! A state that lends [`VerifySlots`] is a deployment's whose verifications copy no state: a
//! request owns `Q` slots of the delta rule's field, whose checkpoint buffer keeps one slot per
//! request, and one window of `Q − 1` rows more than the convolution's history, and a row starts from what its previous step left for the tokens
//! that stand ([`VerifySlots`] states the rule; vLLM's `state_indices` and `num_accepted`). A
//! decode-phase batch then leaves the state after each of a row's tokens for the next step to
//! choose from, and every other batch the state after a row's last token in the row's first
//! slot. Such a state keeps no trail.
//!
//! The buffers are execution's, lent for the forward and written in place; a function reads
//! and writes one field of one recurrent layer of them.

use std::ops::Range;

pub use ops::recurrent::StateAxes;
use ops::recurrent::{
    CausalConv1d, CausalConv1dArgs, CausalConv1dDescriptor, DELTA_RULE_CHUNK, Form, GatedDeltaRule,
    GatedDeltaRuleArgs, GatedDeltaRuleDescriptor, RequestSlots, RowColumns, Trail,
};
use proto::module::Phase;
use tensor::{D, DType, Device, Tensor};

use super::{ACCUMULATE, on_cpu, vector};
use crate::capability::{NO_SLOT, RecurrentState, TokenBatch, VerifySlots};
use crate::error::{Error, Result};

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
/// `dtype`: as many checkpoint slots as live ones, or one per request when a request owns
/// `owned` live slots of the field (module documentation, Request slots), and a trail row for
/// each of `tokens` tokens.
fn check_buffers(
    owner: &'static str,
    (live, checkpoint, trail): (&Tensor, &Tensor, Option<&Tensor>),
    dtype: DType,
    dims: &[usize],
    tokens: usize,
    owned: usize,
) -> Result<()> {
    let slots = rows_of(owner, "live", live, dtype, dims)?;
    let kept = rows_of(owner, "checkpoint", checkpoint, dtype, dims)?;
    if kept != slots && kept * owned != slots {
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
        query_offsets: ops::operand(&state.query_offsets)?,
        seq_lens: ops::operand(&state.seq_lens)?,
        slots: ops::operand(&state.slots)?,
        checkpoints: ops::operand(&state.checkpoints)?,
    })
}

/// The trail argument of a launch.
fn trail_operand(trail: Option<&Tensor>) -> Result<Option<Trail>> {
    Ok(trail
        .map(ops::operand)
        .transpose()?
        .map(|bank| Trail { bank }))
}

/// The values of `column`, a U32 index column of a CPU state.
fn column(column: &Tensor) -> Result<Vec<u32>> {
    Ok(column.flatten_all()?.to_vec1::<u32>()?)
}

/// The slots operand of a launch: the two columns of a state that lends request slots.
fn slots_operand(verify: Option<&VerifySlots>) -> Result<Option<RequestSlots>> {
    verify
        .map(|verify| {
            Ok(RequestSlots {
                state_indices: ops::operand(&verify.state_indices)?,
                num_accepted: ops::operand(&verify.num_accepted)?,
            })
        })
        .transpose()
}

/// The slots a request owns under `verify`, checked against `rows` state rows.
fn owned_slots(owner: &'static str, verify: &VerifySlots, rows: usize) -> Result<usize> {
    let (indices, accepted) = (&verify.state_indices, &verify.num_accepted);

    match indices.dims() {
        &[stated, owned]
            if stated == rows
                && owned > 0
                && indices.dtype() == DType::U32
                && accepted.dims() == [rows]
                && accepted.dtype() == DType::U32 =>
        {
            Ok(owned)
        }
        _ => Err(Error::shape(
            owner,
            format!(
                "state_indices of {:?} and num_accepted of {:?} for {rows} state rows: \
                 [{rows}, Q] and [{rows}] u32 expected",
                indices.dims(),
                accepted.dims()
            ),
        )),
    }
}

/// The request slots of a CPU state on the host: each row's slots, and how many tokens of
/// each row's previous step stand.
type HostSlots = (Vec<Vec<u32>>, Vec<u32>);

/// The request slots of a CPU state, read to the host.
fn request_slots(owner: &'static str, verify: Option<&VerifySlots>) -> Result<Option<HostSlots>> {
    let Some(verify) = verify else {
        return Ok(None);
    };

    let accepted = column(&verify.num_accepted)?;
    let indices = verify.state_indices.to_vec2::<u32>()?;
    let owned = indices.first().map_or(0, Vec::len) as u32;
    if let Some(row) = accepted
        .iter()
        .position(|count| *count == 0 || *count > owned)
    {
        return Err(Error::shape(
            owner,
            format!(
                "state row {row} states {} standing tokens of a step of {owned}",
                accepted[row]
            ),
        ));
    }
    Ok(Some((indices, accepted)))
}

/// Runs `advance(row, tokens, slot, start, checkpoint)` over each state row of a CPU `state`
/// that names a slot, in row order: `tokens` are the row's packed tokens, `start` the absolute
/// position of the first of them, and `checkpoint` the absolute token count the row keeps its
/// state at, zero for none.
fn for_each_row(
    owner: &'static str,
    state: &RecurrentState,
    mut advance: impl FnMut(usize, Range<usize>, usize, usize, usize) -> Result<()>,
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
            row,
            tokens,
            slots[row] as usize,
            start,
            checkpoints[row] as usize,
        )?;
    }
    Ok(())
}

/// The depthwise causal convolution of `x`, `[tokens, channels]`, continuing each state row
/// from its slot's last inputs: `[tokens, channels]` in `x`'s element type.
///
/// `weight` is `[channels, width]`, each channel's taps with the oldest input's first; tap `j`
/// reads the input `(width − 1 − j) · dilation` tokens back, so a token's history is
/// `(width − 1) · dilation` inputs. With `activation` the FP32 sum passes through its SiLU
/// before it rounds; without, the sum rounds once.
///
/// The field `field` of recurrent layer `layer` of `state` holds each slot's window,
/// `[slots, window, channels]` in `x`'s element type, oldest first: the history, and one row
/// more per slot a request owns beyond its first (module documentation, Request slots). It is
/// read for a row that continues and written in place: with the row's last inputs from window
/// row zero, or, for a decode-phase `batch` of a state that lends request slots, with the
/// inputs after the first one read. The checkpoint buffer receives the inputs before a row's
/// checkpoint from window row zero, and the trail the inputs up to and including each token.
///
/// # Errors
///
/// Taps that are not a matrix over `x`'s channels, a state that does not hold the field or
/// holds another extent or element type, a window that is not the history and the request's
/// further slots, a trail beside request slots, or the operator's refusal.
#[expect(
    clippy::too_many_arguments,
    reason = "the convolution's operands and the state execution lends it"
)]
pub fn causal_conv1d(
    x: &Tensor,
    weight: &Tensor,
    dilation: u32,
    activation: bool,
    batch: &TokenBatch,
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

    // The window is the history and a row per slot a request owns beyond its first.
    let history = (width - 1) * dilation.max(1) as usize;
    let owned = match &state.verify {
        Some(verify) => owned_slots(OWNER, verify, state.seq_lens.elem_count())?,
        None => 1,
    };
    let window = history + owned - 1;
    if dilation == 0 || (state.verify.is_some() && trail.is_some()) {
        return Err(Error::shape(
            OWNER,
            format!(
                "taps {dilation} inputs apart{}",
                if trail.is_some() {
                    ", and a trail beside request slots"
                } else {
                    ""
                }
            ),
        ));
    }
    check_buffers(
        OWNER,
        (live, checkpoint, None),
        x.dtype(),
        &[window, channels],
        tokens,
        1,
    )?;
    if let Some(trail) = trail
        && rows_of(OWNER, "trail", trail, x.dtype(), &[history, channels])? < tokens
    {
        return Err(Error::shape(
            OWNER,
            format!("a trail of {} rows for {tokens} tokens", trail.dims()[0]),
        ));
    }

    // A decode-phase batch of a state that lends request slots leaves its window for a step
    // that chooses how many of a row's tokens stand.
    let stepwise = state.verify.is_some() && batch.phase == Phase::Decode;

    if on_cpu(x) {
        return conv_formula(
            x,
            weight,
            (dilation as usize, activation, stepwise),
            state,
            (live, checkpoint, trail),
        );
    }

    let out = ops::empty((tokens, channels), x.dtype())?;
    ops::launch::<CausalConv1d>(
        &CausalConv1dDescriptor::new(
            x.dtype(),
            channels as u32,
            width as u32,
            dilation,
            activation,
            window as u32,
            stepwise,
        ),
        CausalConv1dArgs {
            x: ops::operand(x)?,
            weight: ops::operand(weight)?,
            out: ops::operand(&out)?,
            rows: row_columns(state)?,
            state: ops::operand(live)?,
            checkpoint: ops::operand(checkpoint)?,
            trail: trail_operand(trail)?,
            slots: slots_operand(state.verify.as_ref())?,
        },
    )?;
    Ok(out)
}

/// [`causal_conv1d`] on CPU tensors: `form` is the taps' dilation, whether the sum is
/// activated and whether the window is left for a step that chooses, and `buffers` are the
/// field's live, checkpoint and trail buffers.
fn conv_formula(
    x: &Tensor,
    weight: &Tensor,
    (dilation, activation, stepwise): (usize, bool, bool),
    state: &RecurrentState,
    (live, checkpoint, trail): (&Tensor, &Tensor, Option<&Tensor>),
) -> Result<Tensor> {
    const OWNER: &str = "causal_conv1d";

    let (tokens, channels) = x.dims2()?;
    let width = weight.dims()[1];
    let history = (width - 1) * dilation;
    let window = live.dims()[1];
    let taps = weight.to_dtype(ACCUMULATE)?;
    let out = Tensor::zeros((tokens, channels), x.dtype(), &Device::Cpu)?;
    let slots = request_slots(OWNER, state.verify.as_ref())?;

    for_each_row(OWNER, state, |row, tokens, slot, start, keep| {
        let count = tokens.len();

        // The window row the row's history begins at: past the tokens of its previous step
        // that do not stand.
        let offset = match &slots {
            Some((_, accepted)) => (accepted[row] as usize - 1).min(window - history),
            None => 0,
        };
        if stepwise && count > window - history + 1 {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a decode-phase row of {count} tokens against requests of {} slots",
                    window - history + 1
                ),
            ));
        }

        // The inputs the row's taps read: the slot's last inputs, or zeros for a row that
        // starts its sequence, then the row's own.
        let before = if start == 0 {
            Tensor::zeros((history, channels), x.dtype(), &Device::Cpu)?
        } else {
            live.get(slot)?.narrow(0, offset, history)?
        };
        let inputs = Tensor::cat(&[before, x.narrow(0, tokens.start, count)?], 0)?;
        let wide = inputs.to_dtype(ACCUMULATE)?;

        let mut sum = Tensor::zeros((count, channels), ACCUMULATE, &Device::Cpu)?;
        for tap in 0..width {
            let weights = taps.narrow(1, tap, 1)?.t()?;
            sum = (sum
                + wide
                    .narrow(0, tap * dilation, count)?
                    .broadcast_mul(&weights)?)?;
        }
        let sum = if activation { sum.silu()? } else { sum };
        out.slice_set(&sum.to_dtype(x.dtype())?, 0, tokens.start)?;

        // The inputs from input `first` on as a slot's window, zeros past the row's last.
        let window_from = |first: usize| -> Result<Tensor> {
            let held = (history + count - first).min(window);
            let rows = inputs.narrow(0, first, held)?;

            Ok(rows
                .pad_with_zeros(0, 0, window - held)?
                .unsqueeze(0)?
                .contiguous()?)
        };
        if let Some(trail) = trail {
            // A trail's rows are the history alone: the state after each token.
            for token in 0..count {
                let after = inputs.narrow(0, token + 1, history)?.unsqueeze(0)?;
                trail.slice_set(&after.contiguous()?, 0, tokens.start + token)?;
            }
        }
        if keep > start && keep <= start + count {
            checkpoint.slice_set(&window_from(keep - start)?, 0, slot)?;
        }

        // A step that chooses reads the history before the row's token `t + 1` from window
        // row `t`; any other step reads the history after the row from row zero.
        let first = if stepwise { 1 } else { count };
        live.slice_set(&window_from(first)?, 0, slot)?;
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
/// key_dim, value_dim]`, or `[slots, value_heads, value_dim, key_dim]` under
/// [`StateAxes::ValueKey`], read and written in place with its checkpoint and trail buffers
/// (module documentation).
///
/// A decode-phase batch (a decode, or a verification) advances its rows one token at a time
/// and keeps the trail the state has; any other advances them in chunks of
/// [`DELTA_RULE_CHUNK`] tokens at absolute positions, each solved at once, which keeps no trail
/// and puts its checkpoints at chunk ends. Under a state that lends request slots a row starts
/// from the slot of the tokens that stand, and a decode-phase batch leaves the state after its
/// token `t` in the row's slot `t` (module documentation, Request slots).
///
/// # Errors
///
/// Heads of other token counts or widths, value heads the key heads do not group, gate inputs
/// or parameters that are not one per value head, a state that does not hold the field or
/// holds another extent, a trail kept by a batch that advances in chunks or beside request
/// slots, a page that is not whole chunks, or the operator's refusal.
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
    axes: StateAxes,
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
    let dims = match axes {
        StateAxes::KeyValue => [value_heads, key_dim, value_dim],
        StateAxes::ValueKey => [value_heads, value_dim, key_dim],
    };
    let owned = match &state.verify {
        Some(verify) => owned_slots(OWNER, verify, state.seq_lens.elem_count())?,
        None => 0,
    };
    check_buffers(
        OWNER,
        (live, checkpoint, trail),
        DType::F32,
        &dims,
        tokens,
        owned.max(1),
    )?;
    if owned > 0 && trail.is_some() {
        return Err(Error::shape(
            OWNER,
            "a state that lends request slots keeps no trail".to_owned(),
        ));
    }

    // A page holds whole chunks of the chunked form, so a checkpoint at a page boundary lands on
    // a chunk end.
    let form = match batch.phase {
        Phase::Decode => Form::Recurrent,
        _ => Form::Chunked,
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
            (axes, form == Form::Recurrent),
            state,
            (live, checkpoint, trail),
        );
    }

    let out = ops::empty((tokens, value_heads, value_dim), query.dtype())?;
    let args = GatedDeltaRuleArgs {
        query: ops::operand(query)?,
        key: ops::operand(key)?,
        value: ops::operand(value)?,
        a: ops::operand(a)?,
        b: ops::operand(b)?,
        a_log: ops::operand(a_log)?,
        dt_bias: ops::operand(dt_bias)?,
        rows: row_columns(state)?,
        state: ops::operand(live)?,
        checkpoint: ops::operand(checkpoint)?,
        trail: trail_operand(trail)?,
        out: ops::operand(&out)?,
        slots: slots_operand(state.verify.as_ref())?,
    };
    ops::launch::<GatedDeltaRule>(&GatedDeltaRuleDescriptor::of(&args, form, axes)?, args)?;
    Ok(out)
}

/// [`gated_delta_rule`] on CPU tensors, as the recurrence of the module documentation:
/// `heads` are the queries, keys and values, `gates` the inputs `a` and `b`, `parameters`
/// `A_log` and `dt_bias`, `form` the stored state's axis order and whether the batch advances
/// token by token, and `buffers` the field's live, checkpoint and trail buffers.
///
/// The recurrence is computed on states of key rows, `[value_heads, key_dim, value_dim]`; a
/// state stored in the other order is transposed as it is read and as it is written.
fn rule_formula(
    (query, key, value): (&Tensor, &Tensor, &Tensor),
    (a, b): (&Tensor, &Tensor),
    (a_log, dt_bias): (&Tensor, &Tensor),
    (axes, stepwise): (StateAxes, bool),
    state: &RecurrentState,
    (live, checkpoint, trail): (&Tensor, &Tensor, Option<&Tensor>),
) -> Result<Tensor> {
    const OWNER: &str = "gated_delta_rule";
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
    let slots = request_slots(OWNER, state.verify.as_ref())?;

    // A state as it is stored, `[1, …]`, and a stored state as the recurrence reads it.
    let stored = |memory: &Tensor| -> Result<Tensor> {
        Ok(match axes {
            StateAxes::KeyValue => memory.unsqueeze(0)?,
            StateAxes::ValueKey => memory.transpose(1, 2)?.contiguous()?.unsqueeze(0)?,
        })
    };
    let read = |slot: usize| -> Result<Tensor> {
        Ok(match axes {
            StateAxes::KeyValue => live.get(slot)?.copy()?,
            StateAxes::ValueKey => live.get(slot)?.transpose(1, 2)?.contiguous()?,
        })
    };

    for_each_row(OWNER, state, |row, tokens, slot, start, keep| {
        // A row starts from the state its previous step left for the tokens that stand, read
        // before any of the row's slots is written.
        let owned = slots.as_ref().map(|(indices, _)| indices[row].as_slice());
        let source = match &slots {
            Some((indices, accepted)) => indices[row][accepted[row] as usize - 1] as usize,
            None => slot,
        };
        if stepwise && owned.is_some_and(|owned| tokens.len() > owned.len()) {
            return Err(Error::shape(
                OWNER,
                format!(
                    "a decode-phase row of {} tokens against requests of {} slots",
                    tokens.len(),
                    owned.map_or(0, <[u32]>::len)
                ),
            ));
        }

        let mut memory = if start == 0 {
            Tensor::zeros((value_heads, key_dim, value_dim), ACCUMULATE, &Device::Cpu)?
        } else {
            read(source)?
        };

        for (seen, token) in tokens.clone().enumerate() {
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

            let after = stored(&memory)?;
            if let Some(trail) = trail {
                trail.slice_set(&after, 0, token)?;
            }
            if start + seen + 1 == keep {
                checkpoint.slice_set(&after, 0, slot)?;
            }

            // A step that chooses keeps the state after every token, in the row's slot of it.
            if let (true, Some(owned)) = (stepwise, owned) {
                live.slice_set(&after, 0, owned[seen] as usize)?;
            }
        }
        if !(stepwise && owned.is_some()) {
            live.slice_set(&stored(&memory)?, 0, slot)?;
        }
        Ok(())
    })?;
    Ok(out)
}
