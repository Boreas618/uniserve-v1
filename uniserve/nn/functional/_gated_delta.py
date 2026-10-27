"""The gated delta rule over packed rows, with its causal convolution.

The gated delta rule (Yang, Kautz and Hatamizadeh, "Gated Delta Networks",
2024) keeps a ``[key_dim, value_dim]`` state ``S`` per head. Token ``t``
with query ``q_t``, key ``k_t``, value ``v_t``, log decay ``g_t <= 0`` and
write strength ``beta_t`` updates and reads it as::

    S = exp(g_t) * S
    S = S + k_t (beta_t * (v_t - S^T k_t))^T
    o_t = S^T q_t

These functions evaluate it in FP32 with tensor operations over packed
rows: the ``lengths[r]`` tokens of row ``r`` start at packed token
``starts[r]`` and continue a sequence at absolute position ``prefixes[r]``
from the row's initial state. :func:`gated_delta_rule` uses the chunked
form (the WY representation of Yang et al., in the arithmetic of the
Transformers reference ``torch_chunk_gated_delta_rule``) on chunks aligned
to absolute positions, so the state at every multiple of the chunk length
within a row is the state the chunk ends with; :func:`gated_delta_step`
applies one token per row. Every shape follows the call's sizes and
nothing reads a device value on the host; the chunked form launches its
recurrence once per chunk of the call.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def packed_rows(
    offsets: torch.Tensor, tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The row of each of ``tokens`` packed tokens and its index in the row.

    ``offsets`` is the int64 ``[rows + 1]`` row offsets. A token past the
    last row belongs to row ``rows``, the padding row, at index zero.
    """
    rows = offsets.shape[0] - 1
    position = torch.arange(tokens, dtype=torch.int64, device=offsets.device)
    row = torch.searchsorted(offsets[1:].contiguous(), position, right=True)
    start = torch.cat((offsets[:-1], offsets[-1:]))[row.clamp_max(rows)]
    index = torch.where(row < rows, position - start, 0)
    return row, index


def _previous(
    x: torch.Tensor,
    window: torch.Tensor,
    row: torch.Tensor,
    index: torch.Tensor,
    first: torch.Tensor,
    lag: int,
) -> torch.Tensor:
    """The input ``lag`` positions before in-row index ``index`` of ``row``.

    It is the row's packed input at ``first + index - lag`` when that index
    is in the row, and else the row's window entry: the window holds the
    ``kernel - 1`` inputs before the row's first token, oldest first.
    ``x`` is ``[tokens, channels]``, ``window`` ``[rows + 1, channels,
    kernel - 1]`` (the last row is padding), ``first`` each position's row
    offset; ``row``, ``index`` and ``first`` share one shape.
    """
    tokens = x.shape[0]
    taps = window.shape[-1]
    inside = index >= lag
    packed = x[(first + index - lag).clamp(0, max(tokens - 1, 0))]
    slot = (taps + index - lag).clamp(0, max(taps - 1, 0))
    stored = window[row, :, slot]
    return torch.where(inside.unsqueeze(-1), packed, stored.to(x.dtype))


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    window: torch.Tensor,
    *,
    starts: torch.Tensor,
    row: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    """SiLU of each channel's causal convolution over its row's inputs.

    ``x`` is ``[tokens, channels]``, ``weight`` ``[channels, 1, kernel]``
    (``torch.nn.Conv1d``'s depthwise layout) and ``window`` ``[rows,
    channels, kernel - 1]``, each row's inputs before its first token,
    oldest first, and ``starts`` int64 ``[rows]`` each row's first packed
    token. ``row`` and ``index`` are each token's row and in-row index
    (:func:`packed_rows`); a token of row ``rows`` is padding. Products
    accumulate in FP32 and the convolution rounds to ``x``'s dtype before
    the SiLU, which rounds again, as a depthwise ``torch.nn.Conv1d``
    followed by ``F.silu`` does.
    """
    kernel = weight.shape[-1]
    taps = weight[:, 0, :].float()
    padded = torch.cat((window, window[:1]))
    first = torch.cat((starts, starts.new_zeros(1)))[row]
    total = x.float() * taps[:, kernel - 1]
    for lag in range(1, kernel):
        previous = _previous(x, padded, row, index, first, lag)
        total = total + previous.float() * taps[:, kernel - 1 - lag]
    return F.silu(total.to(x.dtype))


def conv_window(
    x: torch.Tensor,
    window: torch.Tensor,
    *,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """Each row's last ``kernel - 1`` inputs before in-row index ``ends``.

    ``x`` is ``[tokens, channels]``, ``window`` ``[rows, channels, kernel -
    1]`` the inputs before each row's first token, ``starts`` int64
    ``[rows]`` each row's first packed token and ``ends`` int64 ``[rows]``.
    Returns ``[rows, channels, kernel - 1]`` in ``window``'s dtype, oldest
    first: the window a row continuing at ``ends`` reads.
    """
    rows, _, taps = window.shape
    row = torch.arange(rows, dtype=torch.int64, device=x.device)
    padded = torch.cat((window, window[:1]))
    entries = [
        _previous(x, padded, row, ends, starts, lag).to(window.dtype)
        for lag in range(taps, 0, -1)
    ]
    return torch.stack(entries, dim=-1)


def gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    lengths: torch.Tensor,
    prefixes: torch.Tensor,
    row: torch.Tensor,
    index: torch.Tensor,
    initial: torch.Tensor,
    snapshot: torch.Tensor,
    chunk: int,
    chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The gated delta rule over packed rows, in chunks of ``chunk`` tokens.

    Args:
        query: ``[tokens, heads, key_dim]`` FP32, normalized and scaled.
        key: ``[tokens, heads, key_dim]`` FP32, normalized.
        value: ``[tokens, heads, value_dim]`` FP32.
        gate: ``[tokens, heads]`` FP32 log decay.
        beta: ``[tokens, heads]`` FP32 write strength.
        lengths: int64 ``[rows]`` tokens of each row.
        prefixes: int64 ``[rows]`` absolute position of each row's first
            token.
        row: int64 ``[tokens]`` each token's row (:func:`packed_rows`).
        index: int64 ``[tokens]`` each token's index in its row.
        initial: ``[rows, heads, key_dim, value_dim]`` FP32 state before
            each row's first token.
        snapshot: int64 ``[rows]`` an absolute position, a multiple of
            ``chunk`` within each row's tokens, whose state to return; -1
            for none.
        chunk: Chunk length; chunks start at multiples of it.
        chunks: A static bound on the chunks of every row together; one row
            of ``q`` tokens spans at most ``q // chunk + 2``.

    Returns:
        Each token's output ``[tokens, heads, value_dim]``, each row's
        final state and each row's state at ``snapshot`` (unspecified for a
        row without one), FP32.
    """
    tokens, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    rows = initial.shape[0]
    device = key.device

    # Each row's chunks: the first holds the row's first absolute position,
    # and chunk j of the row starts at position (first + j) * chunk.
    first = prefixes // chunk
    count = torch.where(
        lengths > 0, (prefixes + lengths - 1) // chunk - first + 1, 0
    )
    ends = torch.cumsum(count, 0)
    starts = ends - count

    # The packed slot of each token: its row's chunk and its offset in it.
    # Tokens past the last row go to slot `chunks * chunk`, outside every
    # chunk the scan reads.
    absolute = prefixes[row.clamp_max(rows - 1)] + index
    target = (
        starts[row.clamp_max(rows - 1)]
        + absolute // chunk
        - first[row.clamp_max(rows - 1)]
    ) * chunk + absolute % chunk
    target = torch.where(row < rows, target, chunks * chunk)
    source = torch.full(
        (chunks * chunk + 1,), tokens, dtype=torch.int64, device=device
    )
    source.scatter_(0, target, torch.arange(tokens, device=device))
    source = source[:-1]

    def chunked(values: torch.Tensor) -> torch.Tensor:
        """``[tokens, heads, ...]`` values as ``[chunks, heads, chunk,
        ...]``, zero at empty slots.
        """  # noqa: D205
        padded = torch.cat((values, values.new_zeros((1, *values.shape[1:]))))
        gathered = padded[source].view(chunks, chunk, *values.shape[1:])
        return gathered.transpose(1, 2)

    q, k, v = chunked(query), chunked(key), chunked(value)
    g, b = chunked(gate), chunked(beta)

    # Within-chunk terms of every chunk at once (Transformers'
    # `torch_chunk_gated_delta_rule`): the cumulative decay, the decay
    # between positions, and the WY factors u = W (beta v) and
    # w = W (beta k exp(G)) with W = (I + tril(beta K K^T Γ, -1))^{-1}.
    decay = g.cumsum(-1)
    lower = torch.ones(chunk, chunk, dtype=torch.bool, device=device).tril()
    between = (
        (decay.unsqueeze(-1) - decay.unsqueeze(-2)).masked_fill(~lower, 0)
    ).exp() * lower
    keyed = k * b.unsqueeze(-1)
    strictly = lower.logical_and(
        ~torch.eye(chunk, dtype=torch.bool, device=device)
    )
    mixing = ((keyed @ k.transpose(-1, -2)) * between) * strictly
    eye = torch.eye(chunk, dtype=key.dtype, device=device)
    inverse = torch.linalg.solve_triangular(
        eye + mixing, eye.expand_as(mixing), upper=False, unitriangular=True
    )
    u = inverse @ (v * b.unsqueeze(-1))
    w = inverse @ (keyed * decay.exp().unsqueeze(-1))
    attended = (q @ k.transpose(-1, -2)) * between
    queried = q * decay.exp().unsqueeze(-1)
    last = decay[..., -1:]
    carried = k * (last - decay).exp().unsqueeze(-1)

    # The chunks of each row in order; padding chunks (past every row's
    # chunks) belong to row `rows`, whose state nothing reads.
    chunk_row = torch.searchsorted(
        ends,
        torch.arange(chunks, dtype=torch.int64, device=device),
        right=True,
    )
    owner = chunk_row.clamp_max(rows - 1)
    position = torch.arange(chunks, dtype=torch.int64, device=device)
    begins = (position == starts[owner]) & (chunk_row < rows)
    finishes = (position == ends[owner] - 1) & (chunk_row < rows)
    boundary = (first[owner] + position - starts[owner] + 1) * chunk
    snaps = (boundary == snapshot[owner]) & (chunk_row < rows)

    initial = torch.cat((initial, initial.new_zeros((1, *initial.shape[1:]))))
    final = initial.clone()
    snapped = initial.clone()
    outputs = torch.empty(
        (chunks, heads, chunk, value_dim), dtype=value.dtype, device=device
    )
    state = initial.new_zeros((heads, key_dim, value_dim))
    for c in range(chunks):
        current = chunk_row[c : c + 1]
        state = torch.where(begins[c], initial[current][0], state)
        update = u[c] - w[c] @ state
        outputs[c] = queried[c] @ state + attended[c] @ update
        state = state * last[c].exp().unsqueeze(-1) + (
            carried[c].transpose(-1, -2) @ update
        )
        final.index_copy_(
            0,
            current,
            torch.where(finishes[c], state, final[current][0]).unsqueeze(0),
        )
        snapped.index_copy_(
            0,
            current,
            torch.where(snaps[c], state, snapped[current][0]).unsqueeze(0),
        )

    # Back to packed tokens; tokens past every row read zero.
    flat = outputs.transpose(1, 2).reshape(chunks * chunk, heads, value_dim)
    flat = torch.cat((flat, flat.new_zeros((1, heads, value_dim))))
    output = flat[torch.where(row < rows, target, chunks * chunk)]
    return output, final[:rows], snapped[:rows]


def gated_delta_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One token of the gated delta rule per row, in FP32.

    ``query`` and ``key`` are ``[rows, heads, key_dim]`` (the query
    normalized and scaled), ``value`` ``[rows, heads, value_dim]``,
    ``gate`` and ``beta`` ``[rows, heads]`` and ``state`` ``[rows, heads,
    key_dim, value_dim]``. Returns each row's output ``[rows, heads,
    value_dim]`` and its state after the token, in the arithmetic of the
    Transformers reference ``torch_recurrent_gated_delta_rule``.
    """
    state = state * gate.exp()[..., None, None]
    remembered = (state * key.unsqueeze(-1)).sum(dim=-2)
    delta = (value - remembered) * beta.unsqueeze(-1)
    state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
    output = (state * query.unsqueeze(-1)).sum(dim=-2)
    return output, state
