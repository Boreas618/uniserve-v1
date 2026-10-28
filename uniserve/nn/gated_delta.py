"""Gated delta attention: a linear attention layer with recurrent state.

:class:`GatedDeltaNet` is the token mixer of Gated DeltaNet layers (Yang,
Kautz and Hatamizadeh, "Gated Delta Networks", 2024), as Qwen3-Next and
Qwen3.5 checkpoints compose it: a causal depthwise convolution over the
query, key and value channels, L2-normalized queries and keys, and the
gated delta rule (:mod:`uniserve.nn.functional._gated_delta`). Its input
projections and its gated output normalization belong to the model.

The layer is a cache layer kept per request slot
(:class:`GatedDeltaState`): a request's convolution window and recurrent
state live in its slot from one forward to the next, and its checkpoints at
token prefixes in the pool's state group
(:mod:`uniserve.prefix_cache.checkpoints`).

A verification forward (:func:`uniserve.prefix_cache.slots.verification`)
evaluates each row's tokens one recurrence step at a time, the arithmetic
of a decode step, and leaves the rows' slots as they were: only the
verification knows how many of a row's tokens its request accepts. It
then advances each slot over its accepted tokens from the same inputs, so
the state a request keeps is the one decoding those tokens would have left,
whatever drafts followed them. A forward whose rows keep no state
(:func:`uniserve.prefix_cache.slots.kept`, a request's earlier tokens
evaluated again) leaves their slots unchanged and writes only their
checkpoints.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
from torch import nn
from torch.nn import functional as F

from uniserve import prefix_cache
from uniserve.prefix_cache import checkpoints as recurrent
from uniserve.prefix_cache import slots
from uniserve.prefix_cache.state import State, StateConfig
from uniserve.tensors import BufferConfig

from . import functional
from .attention import AttentionBatch

# Tokens of one chunk of the chunked gated delta rule. Every chunk length
# computes the same function; this one is the length of the reference
# kernels' chunks. Chunks end at the checkpoints' boundaries, so a layer
# whose checkpoints are closer runs chunks of their interval.
CHUNK = 64


@dataclass(frozen=True)
class GatedDeltaState(StateConfig):
    """The state of a gated delta layer kept per request slot.

    ``conv`` holds the last ``kernel - 1`` inputs of each convolution
    channel, oldest first, in the activation dtype; ``recurrent`` the
    ``[key_dim, value_dim]`` state of each of the rank's value heads in
    ``state_dtype``. The state is partitioned by value heads across
    tensor-parallel ranks.
    """

    channels: int
    taps: int
    num_key_heads: int
    key_heads: tuple[int, ...]
    num_value_heads: int
    value_heads: tuple[int, ...]
    key_dim: int
    value_dim: int
    dtype: torch.dtype
    state_dtype: torch.dtype
    indexing: ClassVar[Literal["states"]] = "states"

    @property
    def heads(self) -> int:
        return self.num_value_heads

    @property
    def state_heads(self) -> tuple[int, ...]:
        return self.value_heads

    def logical_tensors(self):
        """Convolution channels and recurrent heads in global head order."""
        from uniserve.distributed.distribution import Segment, Span
        from uniserve.distributed.logical import Logical

        key_width = self.num_key_heads * self.key_dim
        channels = []
        local = 0
        for base, heads, width in (
            (0, self.key_heads, self.key_dim),
            (key_width, self.key_heads, self.key_dim),
            (2 * key_width, self.value_heads, self.value_dim),
        ):
            for head in heads:
                channels.append(
                    Segment(Span(base + head * width, width), local)
                )
                local += width
        recurrent = Logical(
            (self.num_value_heads, self.key_dim, self.value_dim),
            ("head", "key", "value"),
            (1, 0, 0),
            (
                tuple(
                    Segment(Span(head, 1), index)
                    for index, head in enumerate(self.value_heads)
                ),
                (Segment(Span(0, self.key_dim), 0),),
                (Segment(Span(0, self.value_dim), 0),),
            ),
        )
        return {
            "conv.values": Logical(
                (
                    2 * key_width + self.num_value_heads * self.value_dim,
                    self.taps,
                ),
                ("channel", "tap"),
                (1, 0),
                (tuple(channels), (Segment(Span(0, self.taps), 0),)),
            ),
            "conv.initialized": Logical.whole(()),
            "recurrent.values": recurrent,
            "recurrent.initialized": Logical.whole(()),
        }

    def buffers(
        self, *, num_blocks, block_size, dtype, quantizer
    ) -> Mapping[str, BufferConfig]:
        heads = len(self.value_heads)
        return {
            "conv.values": BufferConfig(
                (num_blocks, self.channels, self.taps), self.dtype
            ),
            "conv.initialized": BufferConfig((num_blocks,), torch.bool),
            "recurrent.values": BufferConfig(
                (num_blocks, heads, self.key_dim, self.value_dim),
                self.state_dtype,
            ),
            "recurrent.initialized": BufferConfig((num_blocks,), torch.bool),
        }

    def bind(self, tensors, *, block_size, dtype, quantizer) -> State:
        return State(
            {
                "conv": tensors["conv.values"],
                "recurrent": tensors["recurrent.values"],
            },
            {
                "conv": tensors["conv.initialized"],
                "recurrent": tensors["recurrent.initialized"],
            },
            block_size,
        )


class GatedDeltaNet(nn.Module):
    """Causal convolution and the gated delta rule over packed rows.

    ``forward(mixed, a, b, attention)`` takes each token's projected
    channels ``mixed`` (``[tokens, channels]``: the queries of the key
    heads, their keys, then the values of the value heads), its decay input
    ``a`` and write input ``b`` (``[tokens, value_heads]``), and the
    forward's attention batch, whose rows' prefix and query lengths place
    the tokens. It returns ``[tokens, value_heads, value_dim]`` in
    ``mixed``'s dtype. Value head ``h`` reads key head ``h // (value_heads
    / key_heads)``.

    Per token, ``beta = sigmoid(b)`` (rounded to ``b``'s dtype), ``g =
    -exp(a_log) * softplus(a + dt_bias)`` in FP32, queries and keys are
    L2-normalized over their head width (``eps`` 1e-6) in FP32 and queries
    scaled by ``key_dim ** -0.5``.

    Parameters, in the checkpoint's layout: ``conv_weight`` ``[channels, 1,
    kernel]``, ``a_log`` and ``dt_bias`` ``[value_heads]``. Under tensor
    parallelism each rank holds whole key heads and the value heads that
    read them (``parallelize_`` narrows the parameters and the heads).
    """

    function = "nn.gated_delta.GatedDeltaNet/1"

    # The tensor-operation implementation runs a host loop over every chunk
    # of a forward, launching too many kernels for serving to capture its
    # calls; an implementation of fused kernels captures them.
    captures = False

    def __init__(
        self,
        num_key_heads: int,
        num_value_heads: int,
        key_dim: int,
        value_dim: int,
        kernel: int,
        *,
        cache_name: str,
        state_dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if (
            min(num_key_heads, num_value_heads, key_dim, value_dim) < 1
            or num_value_heads % num_key_heads
            or kernel < 2
        ):
            raise ValueError(
                "gated delta attention requires value heads that are a "
                "multiple of key heads and a kernel of two or more taps"
            )
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.kernel = kernel
        self.cache_name = cache_name
        self.state_dtype = state_dtype
        # The heads this rank computes; ``parallelize_`` narrows them.
        self.key_heads = range(num_key_heads)
        self.value_heads = range(num_value_heads)

        channels = 2 * num_key_heads * key_dim + num_value_heads * value_dim
        self.conv_weight = nn.Parameter(
            torch.empty((channels, 1, kernel), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.a_log = nn.Parameter(
            torch.empty(num_value_heads, device=device, dtype=dtype),
            requires_grad=False,
        )
        self.dt_bias = nn.Parameter(
            torch.empty(num_value_heads, device=device, dtype=dtype),
            requires_grad=False,
        )

    def channel_slices(self) -> tuple[slice, slice, slice]:
        """This rank's channels of the checkpoint's convolution: its query,
        key and value channels, in the order ``mixed`` holds them.
        """  # noqa: D205
        keys = self.num_key_heads * self.key_dim
        return (
            slice(
                self.key_heads.start * self.key_dim,
                self.key_heads.stop * self.key_dim,
            ),
            slice(
                keys + self.key_heads.start * self.key_dim,
                keys + self.key_heads.stop * self.key_dim,
            ),
            slice(
                2 * keys + self.value_heads.start * self.value_dim,
                2 * keys + self.value_heads.stop * self.value_dim,
            ),
        )

    def cache_state(self, dtype: torch.dtype) -> GatedDeltaState:
        return GatedDeltaState(
            channels=int(self.conv_weight.shape[0]),
            taps=self.kernel - 1,
            num_key_heads=self.num_key_heads,
            key_heads=tuple(self.key_heads),
            num_value_heads=self.num_value_heads,
            value_heads=tuple(self.value_heads),
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            dtype=dtype,
            state_dtype=self.state_dtype,
        )

    def forward(
        self,
        mixed: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        attention: AttentionBatch,
    ) -> torch.Tensor:
        binding = prefix_cache.bound(self)
        queries = attention.queries
        if queries is None:
            raise ValueError("gated delta attention reads packed rows")
        tokens = mixed.shape[0]
        rows = queries.batch_size
        offsets = queries.offsets.to(torch.int64)
        starts, lengths = offsets[:-1], offsets[1:] - offsets[:-1]
        prefixes = _prefixes(attention)
        slot_rows = slots.rows()
        stateful = binding is not None and slot_rows is not None
        verification = slots.verification()
        if stateful and verification is not None:
            return self._verify(
                binding,
                attention,
                mixed,
                a,
                b,
                slot_rows[:rows],
                offsets,
                prefixes,
                verification,
            )

        # Each row's state at its prefix: its checkpoint, its slot's state,
        # or zero for a request's first forward. Without a cache binding
        # (a standalone call) every row starts its sequence.
        if stateful:
            checkpoints = binding.checkpoints
            restores = writes = checkpoints is not None
            host = _host_lengths(attention)
            if checkpoints is not None and host is not None:
                # Most forwards cross no page boundary (a decode step does
                # once every interval): skip the checkpoint lookups then.
                restores, writes = _crossings(*host, checkpoints.interval)
            initial = recurrent.resume(
                binding, attention, slot_rows[:rows], restore=restores
            )
            window = initial["conv.values"].to(mixed.dtype)
            state = initial["recurrent.values"].float()
            boundary, units = (
                checkpoints.boundary(attention) if writes else (None, None)
            )
        else:
            window = mixed.new_zeros((rows, mixed.shape[-1], self.kernel - 1))
            state = torch.zeros(
                (rows, len(self.value_heads), self.key_dim, self.value_dim),
                dtype=torch.float32,
                device=mixed.device,
            )
            boundary = units = None
        marks = torch.full_like(prefixes, -1) if boundary is None else boundary

        if _single(attention):
            # One token per row, of distinct requests: the recurrence once
            # per row.
            output, final_window, last = self._steps(
                mixed, a, b, window, state, offsets
            )
            snapshot, snapshot_window, written = last, final_window, marks >= 0
        else:
            # Consecutive rows of one request in one forward continue each
            # other: each chain of them is one sequence, which starts from
            # its first row's state and ends with its last row's.
            first = (
                recurrent.chains(slot_rows[:rows], prefixes, lengths)
                if stateful
                else torch.arange(rows, device=mixed.device)
            )
            chain = torch.zeros_like(lengths).index_add_(0, first, lengths)
            # The deepest boundary any row of a chain checkpoints is the
            # chain's: the row holding it writes the chain's state there.
            deepest = torch.full_like(marks, -1).scatter_reduce_(
                0, first, marks, reduce="amax"
            )
            written = (marks >= 0) & (marks == deepest[first])

            row, _ = functional.packed_rows(offsets, tokens)
            row = torch.cat((first, first.new_full((1,), rows)))[row]
            index = torch.where(
                row < rows,
                torch.arange(tokens, device=mixed.device)
                - torch.cat((starts, starts.new_zeros(1)))[row],
                0,
            )
            convolved = functional.causal_conv1d(
                mixed,
                self.conv_weight,
                window,
                starts=starts,
                row=row,
                index=index,
            )
            final_window = functional.conv_window(
                mixed, window, starts=starts, ends=chain
            )[first]
            snapshot_window = functional.conv_window(
                mixed, window, starts=starts, ends=(deepest - prefixes).clamp(0)
            )[first]
            query, key, value = self._heads(convolved)
            gate, beta = self._gates(a, b)
            chunk = (
                CHUNK
                if binding is None or binding.checkpoints is None
                else math.gcd(CHUNK, binding.checkpoints.interval)
            )
            # The chunks every row spans: counted from the host mirrors of
            # the lengths when the forward has them, else bounded by its
            # sizes (a row of q tokens spans at most q // chunk + 2).
            chunks = _chunks(attention, chunk)
            if chunks is None:
                chunks = tokens // chunk + 2 * rows + 1
            output, last, snapshot = functional.gated_delta_rule(
                query,
                key,
                value,
                gate,
                beta,
                lengths=chain,
                prefixes=prefixes,
                row=row,
                index=index,
                initial=state,
                snapshot=deepest,
                chunk=chunk,
                chunks=max(chunks, 1),
            )
            last, snapshot = last[first], snapshot[first]

        if stateful:
            flags = torch.ones(rows, dtype=torch.bool, device=mixed.device)
            if binding.checkpoints is not None and boundary is not None:
                binding.checkpoints.write(
                    units,
                    {
                        "conv.values": snapshot_window,
                        "conv.initialized": flags,
                        "recurrent.values": snapshot,
                        "recurrent.initialized": flags,
                    },
                    written,
                )
            # Every row of a chain stores the chain's final state; a row
            # that keeps none stores into the padding slot, which no request
            # reads.
            kept = slots.kept()
            recurrent.keep(
                binding,
                slot_rows[:rows]
                if kept is None
                else torch.where(kept[:rows], slot_rows[:rows], 0),
                {
                    "conv.values": final_window,
                    "conv.initialized": flags,
                    "recurrent.values": last,
                    "recurrent.initialized": flags,
                },
            )
        return output.to(mixed.dtype)

    def _verify(
        self,
        binding,
        attention: AttentionBatch,
        mixed: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        rows: torch.Tensor,
        offsets: torch.Tensor,
        prefixes: torch.Tensor,
        verification: slots.Verification,
    ) -> torch.Tensor:
        """Evaluate verification rows and defer their slots' advance.

        Each row's tokens run one recurrence step at a time from its state
        at its prefix; the slots and checkpoints stay as they were. The
        verification's acceptance then advances each slot over its accepted
        tokens (:meth:`_accept`), reading this forward's inputs, which it
        keeps until then.
        """
        lengths = offsets[1:] - offsets[:-1]
        steps = _steps_bound(attention, mixed.shape[0])
        initial = recurrent.resume(binding, attention, rows)
        output, _, _, _, _ = self._advance(
            mixed,
            a,
            b,
            initial["conv.values"].to(mixed.dtype),
            initial["recurrent.values"].float(),
            offsets,
            lengths,
            steps,
        )

        def accept(counts: torch.Tensor) -> None:
            self._accept(
                binding,
                attention,
                mixed,
                a,
                b,
                rows,
                offsets,
                prefixes,
                steps,
                counts,
            )

        verification.defer(accept)
        return output.to(mixed.dtype)

    def _accept(
        self,
        binding,
        attention: AttentionBatch,
        mixed: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        rows: torch.Tensor,
        offsets: torch.Tensor,
        prefixes: torch.Tensor,
        steps: int,
        counts: torch.Tensor,
    ) -> None:
        """Advance each verification row's slot over its first ``counts``
        tokens, int64 ``[rows]`` (zero leaves a slot as it was).

        The steps repeat the verification forward's from the same state and
        inputs, so the kept state is the forward's after those tokens. A
        row whose accepted tokens reach a page boundary its tables give a
        checkpoint writes its state there.
        """  # noqa: D205
        initial = recurrent.resume(binding, attention, rows)
        checkpoints = binding.checkpoints
        boundary = None
        if checkpoints is not None:
            interval = checkpoints.interval
            reached = (prefixes + counts) // interval * interval
            pages = torch.where(reached > prefixes, reached // interval - 1, -1)
            units = checkpoints.units(attention, pages)
            boundary = torch.where(units[:, 0] != 0, reached, -1)
        _, window, state, snapshot_window, snapshot = self._advance(
            mixed,
            a,
            b,
            initial["conv.values"].to(mixed.dtype),
            initial["recurrent.values"].float(),
            offsets,
            counts,
            steps,
            prefixes=prefixes,
            boundary=boundary,
        )
        flags = torch.ones(rows.shape[0], dtype=torch.bool, device=rows.device)
        if boundary is not None:
            checkpoints.write(
                units,
                {
                    "conv.values": snapshot_window,
                    "conv.initialized": flags,
                    "recurrent.values": snapshot,
                    "recurrent.initialized": flags,
                },
                boundary >= 0,
            )
        recurrent.keep(
            binding,
            rows,
            {
                "conv.values": window,
                "conv.initialized": flags,
                "recurrent.values": state,
                "recurrent.initialized": flags,
            },
        )

    def _advance(
        self,
        mixed: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        window: torch.Tensor,
        state: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        steps: int,
        *,
        prefixes: torch.Tensor | None = None,
        boundary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Step each row through its first ``lengths`` tokens, one at a time.

        Row ``r``'s tokens start at ``offsets[r]``; ``steps`` bounds every
        length. Returns each token's output (``[tokens, heads, value_dim]``,
        zero for tokens past their row's length), each row's window and
        state after its tokens, and its window and state after the token
        ending at ``boundary`` (an absolute position, -1 for none; rows
        start at ``prefixes``), the final ones where it reaches none.
        """
        tokens = mixed.shape[0]
        starts = offsets[:-1]
        heads = len(self.value_heads)
        output = torch.zeros(
            (tokens + 1, heads, self.value_dim),
            dtype=torch.float32,
            device=mixed.device,
        )
        snapshot_window, snapshot = window, state
        last = max(tokens - 1, 0)
        for step in range(steps):
            live = step < lengths
            index = (starts + step).clamp(0, last)
            out, shifted, stepped = self._step(
                mixed[index], a[index], b[index], window, state
            )
            window = torch.where(live.view(-1, 1, 1), shifted, window)
            state = torch.where(live.view(-1, 1, 1, 1), stepped, state)
            # A row's output belongs to its token; a row past its length
            # writes the extra row, which no token reads.
            output.index_copy_(0, torch.where(live, index, tokens), out)
            if boundary is not None and prefixes is not None:
                hit = live & (prefixes + step + 1 == boundary)
                snapshot_window = torch.where(
                    hit.view(-1, 1, 1), window, snapshot_window
                )
                snapshot = torch.where(hit.view(-1, 1, 1, 1), state, snapshot)
        if boundary is None:
            snapshot_window, snapshot = window, state
        return output[:tokens], window, state, snapshot_window, snapshot

    def _heads(
        self, convolved: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """FP32 queries and keys per value head, normalized (queries scaled),
        and values, from the convolution's channels.
        """  # noqa: D205
        tokens = convolved.shape[0]
        key_heads = len(self.key_heads)
        value_heads = len(self.value_heads)
        width = key_heads * self.key_dim
        query, key, value = convolved.split(
            (width, width, value_heads * self.value_dim), dim=-1
        )
        query = query.reshape(tokens, key_heads, self.key_dim).float()
        key = key.reshape(tokens, key_heads, self.key_dim).float()
        value = value.reshape(tokens, value_heads, self.value_dim).float()
        query = _l2norm(query) * self.key_dim**-0.5
        key = _l2norm(key)
        repeats = value_heads // key_heads
        if repeats > 1:
            query = query.repeat_interleave(repeats, dim=1)
            key = key.repeat_interleave(repeats, dim=1)
        return query, key, value

    def _gates(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each token's FP32 log decay and write strength per value head."""
        beta = b.sigmoid().float()
        gate = -self.a_log.float().exp() * F.softplus(
            a.float() + self.dt_bias.float()
        )
        return gate, beta

    def _step(self, current, a, b, window, state):
        """One token of each row: its convolution and recurrence step.

        ``current`` is each row's projected channels ``[rows, channels]``,
        ``a`` and ``b`` its decay and write inputs ``[rows, heads]``,
        ``window`` and ``state`` its convolution window and recurrent state
        before the token. Returns the token's output and the window and
        state after it.
        """
        kernel = self.kernel
        taps = self.conv_weight[:, 0, :].float()
        total = current.float() * taps[:, kernel - 1] + (
            window.float() * taps[:, : kernel - 1]
        ).sum(-1)
        convolved = F.silu(total.to(current.dtype))
        shifted = torch.cat((window[..., 1:], current.unsqueeze(-1)), dim=-1)
        query, key, value = self._heads(convolved)
        gate, beta = self._gates(a, b)
        output, stepped = functional.gated_delta_step(
            query, key, value, gate, beta, state
        )
        return output, shifted, stepped

    def _steps(self, mixed, a, b, window, state, offsets):
        """One token per row; a row without tokens keeps its state.

        Returns each token's output and each row's convolution window and
        recurrent state after its token, which is also its state at its
        last boundary: a single token reaches one only at its end.
        """
        tokens = mixed.shape[0]
        lengths = offsets[1:] - offsets[:-1]
        live = lengths > 0
        first = offsets[:-1].clamp(0, max(tokens - 1, 0))
        output, shifted, stepped = self._step(
            mixed[first], a[first], b[first], window, state
        )
        final_window = torch.where(live.view(-1, 1, 1), shifted, window)
        last = torch.where(live.view(-1, 1, 1, 1), stepped, state)

        # Each row's output belongs to its one token; tokens past every row
        # read zero.
        row, _ = functional.packed_rows(offsets, tokens)
        padded = torch.cat((output, output.new_zeros((1, *output.shape[1:]))))
        return padded[row], final_window, last


def _host_lengths(
    attention: AttentionBatch,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Each row's prefix and query lengths from the forward's host mirrors,
    ``None`` without them.

    The layer's calls are never captured (``GatedDeltaNet.captures``), so
    the mirrors hold the lengths of the call that runs, not of a captured
    call another call replays.
    """  # noqa: D205
    queries = attention.queries
    for entry in attention.entries.values():
        prefixes = getattr(entry, "prefixes", None)
        if prefixes is None:
            continue
        if queries is None or queries.host is None or prefixes.host is None:
            return None
        return prefixes.host, queries.host
    return None


def _steps_bound(attention: AttentionBatch, tokens: int) -> int:
    """The most tokens any row of ``attention`` holds: from the host
    mirrors of the lengths, else ``tokens``, the forward's.
    """  # noqa: D205
    host = _host_lengths(attention)
    if host is None:
        return tokens
    return max(host[1], default=0)


def _crossings(
    prefixes: tuple[int, ...], queries: tuple[int, ...], interval: int
) -> tuple[bool, bool]:
    """Whether any row starts at a page boundary (where it may restore a
    checkpoint), and whether any row reaches a boundary past its prefix
    (where it may write one), boundaries every ``interval`` tokens.
    """  # noqa: D205
    restores = any(prefix > 0 and prefix % interval == 0 for prefix in prefixes)
    writes = any(
        (prefix + query) // interval * interval > prefix
        for prefix, query in zip(prefixes, queries, strict=True)
    )
    return restores, writes


def _single(attention: AttentionBatch) -> bool:
    """Whether every row holds at most one token and no row can continue
    the one before it, from the host mirrors of the lengths: a row
    continues another of its request only at the position after it.
    """  # noqa: D205
    host = _host_lengths(attention)
    if host is None:
        return False
    prefixes, queries = host
    if max(queries, default=0) > 1:
        return False
    return not any(
        query and prefixes[row + 1] == prefix + query
        for row, (prefix, query) in enumerate(
            zip(prefixes[:-1], queries[:-1], strict=True)
        )
    )


def _chunks(attention: AttentionBatch, chunk: int) -> int | None:
    """The chunks the rows of ``attention`` span, chunks starting at
    multiples of ``chunk``, from the host mirrors of their lengths; ``None``
    without them.
    """  # noqa: D205
    host = _host_lengths(attention)
    if host is None:
        return None
    return sum(
        (prefix + query - 1) // chunk - prefix // chunk + 1
        for prefix, query in zip(*host, strict=True)
        if query
    )


def _prefixes(attention: AttentionBatch) -> torch.Tensor:
    """Each row's absolute start position, int64 ``[rows]``."""
    for entry in attention.entries.values():
        prefixes = getattr(entry, "prefixes", None)
        if prefixes is not None:
            return prefixes.values.to(torch.int64)
    raise ValueError("gated delta attention reads paged attention rows")


def _l2norm(values: torch.Tensor) -> torch.Tensor:
    """``values`` over the L2 norm of their last axis, ``eps`` 1e-6 inside
    the root (the flash-linear-attention ``l2norm``).
    """  # noqa: D205
    return values * torch.rsqrt(values.square().sum(-1, keepdim=True) + 1e-6)
