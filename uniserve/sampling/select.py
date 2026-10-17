"""Batched token selection, speculative acceptance and terminal decisions.

A task is one sampled call; it owns one or more candidate logits rows (a
verification task owns one row per draft token plus its bonus row). The
selectors take the tasks of one homogeneous group:

- :func:`select_greedy`: argmax of each task's single row, optionally after
  masking suppressed tokens;
- :func:`select_top_k`: each task samples from its top-k candidates under
  its own parameters (:func:`uniserve.sampling.top_k.sample_top_k`, which a
  decode forward also captures);
- :func:`select_categorical`: the general path that shapes logits
  (:func:`shape_logits`), inverts the categorical CDF with each row's
  precomputed semantic draw (argmax at zero temperature) and resolves
  speculative acceptance.

Every selector then resolves predicates, finish and transition decisions as
device tensors. With a ``broadcast``, tensor-parallel ranks replace their
selection with the source rank's before those decisions, so every rank
commits identical state.

Per-row values (temperature, cutoffs, penalty strengths, draws) are tensor
columns. What a caller knows on the host arrives as flat integer lists:
which rows take a shaping stage, and token sets as scatter indices
``row * vocab + token`` into the group's ``[rows, vocab]`` logits, with
out-of-vocabulary and repeated tokens already removed. Per-task token sets
(finish and transition tokens) are row-major ``[tasks, width]`` matrices
padded with -1. Each selector copies its lists to the device together
(:func:`~uniserve.sampling.upload.upload`), without waiting for the
forward that produced its logits.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .categorical import sample_categorical
from .logprobs import LogprobRequest, logprob_details
from .top_k import (
    MIN_P,
    TEMPERATURE,
    TOP_P,
    TYPICAL_P,
    penalize,
    sample_top_k,
    temperature_divisors,
    top_k_candidates,
)
from .upload import upload

# Token relays carry continuation in bit 31 and the token in the low 31 bits.
TOKEN_CONTINUATION_BIT = 1 << 31
TOKEN_VALUE_MASK = TOKEN_CONTINUATION_BIT - 1


@dataclass(frozen=True, slots=True)
class Selection:
    """One group's selections, one entry per task.

    Attributes:
        tokens: Selected (emitted) token per task.
        valid: The task's shaped distribution was usable; under
            speculation, every consumed candidate row was.
        active: The task's predicate.
        finish: The task finishes (false when invalid or inactive).
        continuation: ``valid & active & ~finish``.
        tagged: ``tokens`` with ``TOKEN_CONTINUATION_BIT`` where continuing.
        accepted: Accepted draft tokens.
        points: Emitted tokens: accepted drafts plus a correction or bonus
            token unless an accepted draft finished the task.
        transition: Transition decision per task (false without a
            transition output).
        logprobs: Packed logprob column of the requesting tasks, in task
            order, or ``None``.
    """

    tokens: torch.Tensor
    valid: torch.Tensor
    active: torch.Tensor
    finish: torch.Tensor
    continuation: torch.Tensor
    tagged: torch.Tensor
    accepted: torch.Tensor
    points: torch.Tensor
    transition: torch.Tensor
    logprobs: torch.Tensor | None = None


Broadcast = Callable[[torch.Tensor], object] | None


def _member(tokens: torch.Tensor, sets: torch.Tensor) -> torch.Tensor:
    """Whether each task's token is in its row of the flat ``sets`` matrix."""
    rows = sets.view(int(tokens.numel()), -1)
    return tokens.unsqueeze(1).eq(rows).any(dim=1)


def decisions(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    force_finish: Sequence[int],
    finish_ids: Sequence[int],
    transition_ids: Sequence[int],
    terminal: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finish, continuation, tagged token and transition per task.

    A task finishes when it is valid and active and either forced to
    finish (``force_finish``, one flag per task), its token is in its row
    of ``finish_ids``, or it accepted a terminal draft prefix. It continues
    when valid, active and not finished. A transition requires a valid,
    active selection in the task's row of ``transition_ids``; a task
    without a transition output has an empty row. Both matrices are
    row-major ``[tasks, width]`` with at least one column, padded with -1.
    """
    eligible = valid & active
    forced, finish_sets, transition_sets = upload(
        tokens.device, force_finish, finish_ids, transition_ids
    )
    finish = forced.bool() | _member(tokens, finish_sets)
    if terminal is not None:
        finish = finish | terminal
    finish = finish & eligible
    continuation = eligible & ~finish
    tagged = tokens.to(torch.int64) | torch.where(
        continuation, TOKEN_CONTINUATION_BIT, 0
    )
    transition = _member(tokens, transition_sets) & eligible
    return finish, continuation, tagged, transition


def _valid(work: torch.Tensor) -> torch.Tensor:
    """A row is usable with no NaN or +inf and at least one finite entry."""
    return (
        ~torch.isnan(work).any(dim=-1)
        & ~torch.isposinf(work).any(dim=-1)
        & torch.isfinite(work).any(dim=-1)
    )


def select_greedy(
    logits: torch.Tensor,
    active: torch.Tensor,
    *,
    suppressed: Sequence[int],
    force_finish: Sequence[int],
    finish_ids: Sequence[int],
    transition_ids: Sequence[int],
    broadcast: Broadcast = None,
) -> Selection:
    """Argmax of each task's single ``[tasks, vocab]`` logits row.

    ``suppressed`` scatter indices are masked on a float32 copy before the
    argmax.
    """
    work = logits
    if len(suppressed):
        work = logits.to(dtype=torch.float32, copy=True)
        (indices,) = upload(work.device, suppressed)
        work.reshape(-1).index_fill_(0, indices, float("-inf"))
    tokens = torch.argmax(work, dim=-1)
    valid = _valid(work)
    if broadcast is not None:
        broadcast(tokens)
    finish, continuation, tagged, transition = decisions(
        tokens, valid, active, force_finish, finish_ids, transition_ids
    )
    zeros = torch.zeros_like(tokens)
    return Selection(
        tokens,
        valid,
        active,
        finish,
        continuation,
        tagged,
        zeros,
        zeros + 1,
        transition,
    )


def select_top_k(
    logits: torch.Tensor,
    draws: torch.Tensor,
    params: torch.Tensor,
    active: torch.Tensor,
    width: int,
    *,
    counts: torch.Tensor,
    slots: torch.Tensor,
    force_finish: Sequence[int],
    finish_ids: Sequence[int],
    transition_ids: Sequence[int],
    broadcast: Broadcast = None,
) -> Selection:
    """Sample each task's single row from its top-k candidates.

    The tasks' shaping is decided by their parameter rows alone
    (:func:`~uniserve.sampling.top_k.sample_top_k`); ``counts`` holds every
    slot's committed token counts and ``slots`` each task's slot.
    """
    tokens, valid = sample_top_k(logits, draws, params, width, counts, slots)
    if broadcast is not None:
        broadcast(tokens)
    finish, continuation, tagged, transition = decisions(
        tokens, valid, active, force_finish, finish_ids, transition_ids
    )
    zeros = torch.zeros_like(tokens)
    return Selection(
        tokens,
        valid,
        active,
        finish,
        continuation,
        tagged,
        zeros,
        zeros + 1,
        transition,
    )


def select_categorical(
    logits: torch.Tensor,
    draws: torch.Tensor,
    params: torch.Tensor,
    active: torch.Tensor,
    *,
    row_counts: Sequence[int],
    drafts: Sequence[int],
    terminal_prefixes: Sequence[int],
    candidates: int,
    top_ks: Sequence[int],
    top_p_rows: Sequence[int],
    min_p_rows: Sequence[int],
    typical_p_rows: Sequence[int],
    allowed_rows: Sequence[int],
    allowed: Sequence[int],
    suppressed: Sequence[int],
    penalty_rows: Sequence[int],
    penalty_counts: torch.Tensor | None,
    bias: Sequence[int],
    bias_values: Sequence[float],
    logprob_tasks: Sequence[int],
    logprob_requests: Sequence[LogprobRequest],
    force_finish: Sequence[int],
    finish_ids: Sequence[int],
    transition_ids: Sequence[int],
    broadcast: Broadcast = None,
) -> Selection:
    """Sample tasks through the general shaping path.

    ``logits`` holds every task's candidate rows back to back
    (``row_counts`` rows per task), ``draws`` and ``params`` one entry per
    candidate row. ``drafts`` concatenates the tasks' draft tokens, one
    fewer than each task's rows, and ``terminal_prefixes`` holds per task
    the one-based length of its draft prefix ending at the first terminal
    draft token, or zero. The shaping arguments are those of
    :func:`shape_logits`, the decision arguments those of
    :func:`decisions`, and ``logprob_tasks`` names the tasks reporting one
    of ``logprob_requests`` each. Reported logprobs are those of the
    model's distribution, the unshaped ``logits``: the sampling controls
    decide which token is selected, not how likely the model found it.

    Speculative tasks accept the longest prefix where row ``i``'s selection
    equals draft ``i``. Without a terminal prefix, the output row is the
    first mismatching row, whose selection is the correction, or the bonus
    row after a full match. Reaching a terminal prefix clamps acceptance to
    it and emits its final draft token instead of the output row's
    selection, so the emitted-token count adds no extra token.
    """
    device = logits.device
    work, valid = shape_logits(
        logits.float(),
        params,
        candidates=candidates,
        top_ks=top_ks,
        top_p_rows=top_p_rows,
        min_p_rows=min_p_rows,
        typical_p_rows=typical_p_rows,
        allowed_rows=allowed_rows,
        allowed=allowed,
        suppressed=suppressed,
        penalty_rows=penalty_rows,
        penalty_counts=penalty_counts,
        bias=bias,
        bias_values=bias_values,
    )

    # Temperature-zero rows use deterministic argmax; the others invert the
    # categorical CDF with their precomputed draw. Masked tokens carry zero
    # mass, which the selector never picks.
    probabilities = torch.softmax(work, dim=-1)
    sampled = sample_categorical(probabilities, draws)
    row_tokens = torch.where(
        params[:, TEMPERATURE] > 0.0, sampled, torch.argmax(work, dim=-1)
    )

    # Speculative acceptance over each task's candidate rows laid out as one
    # ``[tasks, width]`` matrix, ``width`` the most rows of a task: a task's
    # rows padded with its first row, and its draft tokens padded with -1,
    # which no selection equals. The padding is never consumed: a task's
    # emitted points stay within its own rows.
    counts = [int(count) for count in row_counts]
    tasks, width = len(counts), max(counts)
    candidate_rows: list[int] = []
    draft_tokens: list[int] = []
    first = taken = 0
    for count in counts:
        candidate_rows.extend(first + index for index in range(count))
        candidate_rows.extend((first,) * (width - count))
        draft_tokens.extend(drafts[taken : taken + count - 1])
        draft_tokens.extend((-1,) * (width - count))
        first += count
        taken += count - 1
    rows, proposed, prefixes, reporting = upload(
        device, candidate_rows, draft_tokens, terminal_prefixes, logprob_tasks
    )
    rows = rows.view(tasks, width)
    selections = row_tokens.index_select(0, rows.reshape(-1)).view(tasks, width)
    matches = selections[:, :-1] == proposed.view(tasks, width - 1)
    raw = torch.cumprod(matches.to(torch.long), dim=1).sum(dim=1)

    # Reaching a terminal prefix clamps acceptance to it. Its tasks matched
    # every draft through the prefix, so their selection at the prefix's
    # last row is that terminal draft token.
    prefixed = prefixes > 0
    terminal = prefixed & (raw >= prefixes)
    accepted = torch.where(prefixed, torch.minimum(raw, prefixes), raw)
    terminal_tokens = selections.gather(
        1, (prefixes - 1).clamp_min(0).unsqueeze(1)
    ).squeeze(1)
    output = rows[:, 0] + accepted
    tokens = torch.where(
        terminal,
        terminal_tokens,
        selections.gather(1, accepted.unsqueeze(1)).squeeze(1),
    )
    points = accepted + (~terminal).to(dtype=torch.long)

    # Broadcast the complete selection so every tensor-parallel rank
    # advances request state from identical speculative decisions.
    if broadcast is not None:
        selection = torch.stack(
            (
                tokens.to(torch.int64),
                accepted.to(torch.int64),
                points.to(torch.int64),
                terminal.to(torch.int64),
            )
        )
        broadcast(selection)
        tokens = selection[0].to(dtype=tokens.dtype)
        accepted = selection[1].to(dtype=accepted.dtype)
        points = selection[2].to(dtype=points.dtype)
        terminal = selection[3].to(dtype=torch.bool)

    # A task is valid only when every consumed speculative point retained a
    # legal candidate under its sampling constraints: its first ``points``
    # rows.
    positions = torch.arange(width, device=device)
    consumed = positions.unsqueeze(0) < points.unsqueeze(1)
    legal = valid.index_select(0, rows.reshape(-1)).view(tasks, width)
    task_valid = (legal | ~consumed).all(dim=1)
    finish, continuation, tagged, transition = decisions(
        tokens,
        task_valid,
        active,
        force_finish,
        finish_ids,
        transition_ids,
        terminal,
    )

    # A terminal selection emits the finish draft token, which the
    # verification row before the output row scored.
    packed = None
    if len(logprob_tasks):
        scored = output - terminal.to(dtype=torch.long)
        packed = logprob_details(
            logits.float(),
            scored.index_select(0, reporting),
            tokens.index_select(0, reporting),
            logprob_requests,
        )
    return Selection(
        tokens,
        task_valid,
        active,
        finish,
        continuation,
        tagged,
        accepted,
        points,
        transition,
        packed,
    )


def shape_logits(
    work: torch.Tensor,
    params: torch.Tensor,
    *,
    candidates: int,
    top_ks: Sequence[int],
    top_p_rows: Sequence[int] = (),
    min_p_rows: Sequence[int] = (),
    typical_p_rows: Sequence[int] = (),
    allowed_rows: Sequence[int] = (),
    allowed: Sequence[int] = (),
    suppressed: Sequence[int] = (),
    penalty_rows: Sequence[int] = (),
    penalty_counts: torch.Tensor | None = None,
    bias: Sequence[int] = (),
    bias_values: Sequence[float] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the canonical shaping and truncation order to logits rows.

    Stages run in this order on a float32 copy: allowed-token restriction,
    suppression, penalties, logit bias, temperature, top-k with nested
    top-p, full-vocabulary top-p, min-p and typical-p. Float values come
    from the ``params`` columns; the lists decide which rows take each
    stage.

    Args:
        work: ``[rows, vocab]`` logits.
        params: ``[rows, PARAMETER_COLUMNS]`` float parameters
            (:mod:`uniserve.sampling.top_k`).
        candidates: The top-k sampler's candidate width: a row whose top-k
            width is at most this keeps the candidates that sampler keeps.
        top_ks: Per row, its top-k width, or zero without one.
        top_p_rows: Rows with a nucleus cutoff below one and no top-k width.
        min_p_rows: Rows with a minimum-probability cutoff.
        typical_p_rows: Rows with a typical-probability cutoff below one.
        allowed_rows: Rows restricted to the tokens ``allowed`` (scatter
            indices) names for them.
        suppressed: Scatter indices of tokens masked from selection.
        penalty_rows: Rows with repetition, frequency or presence penalties;
            ``penalty_counts`` holds their ``[len(penalty_rows), vocab]``
            token counts.
        bias: Scatter indices of the additive offsets ``bias_values``.

    Returns:
        ``(work, valid)``: the shaped ``[rows, vocab]`` float32 logits with
        dropped tokens at -inf, and whether each row keeps a usable
        distribution.
    """
    work = work.to(dtype=torch.float32, copy=True)
    device = work.device

    # Top-k truncation groups. Rows whose width is at most ``candidates``
    # keep their candidates as the top-k sampler (``sample_top_k``) does:
    # the shared candidate width, then each row's own width. Both selectors
    # therefore keep the same tokens, including the order among equal logits
    # at a cutoff. Wider rows sharing a width are truncated in one
    # gather/sort pass of that width.
    narrow = [
        index
        for index, top_k in enumerate(top_ks)
        if 0 < int(top_k) <= candidates
    ]
    own = [int(top_ks[index]) for index in narrow]
    widths: dict[int, list[int]] = defaultdict(list)
    for index, top_k in enumerate(top_ks):
        if int(top_k) > candidates:
            widths[int(top_k)].append(index)

    # Every list a stage indexes with, as one copy.
    (
        allowed_row_index,
        allowed_index,
        suppressed_index,
        penalty_index,
        bias_index,
        top_p_index,
        min_p_index,
        typical_p_index,
        narrow_index,
        own_widths,
        *wide_index,
    ) = upload(
        device,
        allowed_rows,
        allowed,
        suppressed,
        penalty_rows,
        bias,
        top_p_rows,
        min_p_rows,
        typical_p_rows,
        narrow,
        own,
        *widths.values(),
    )

    # Allowed-token restriction masks every other vocabulary entry. The
    # allowed entries are filled with a scalar: assigning one through an
    # index (``mask[index] = True``) copies it to the device and waits.
    if len(allowed_rows):
        mask = torch.ones_like(work, dtype=torch.bool)
        mask.index_fill_(0, allowed_row_index, False)
        mask.reshape(-1).index_fill_(0, allowed_index, True)
        work.masked_fill_(~mask, float("-inf"))

    # Suppressed tokens are masked regardless of any other shaping.
    if len(suppressed):
        work.reshape(-1).index_fill_(0, suppressed_index, float("-inf"))

    # Penalties apply to tokens with a positive count (see ``penalize``).
    if len(penalty_rows) and penalty_counts is not None:
        work.index_copy_(
            0,
            penalty_index,
            penalize(
                work.index_select(0, penalty_index),
                params.index_select(0, penalty_index),
                penalty_counts,
            ),
        )

    # Logit bias adds per-token offsets on top of the penalized logits.
    if len(bias):
        (offsets,) = upload(device, bias_values, dtype=work.dtype)
        work.reshape(-1).index_put_((bias_index,), offsets, accumulate=True)

    # A zero temperature divides by one; callers select those rows by argmax.
    work.div_(temperature_divisors(params).unsqueeze(1))

    # Top-k truncation; top-p applies within each retained top-k set. Each
    # pass: its gather width, its rows and, for the narrow rows, each row's
    # own width.
    passes: list[tuple[int, torch.Tensor, torch.Tensor | None]] = [
        (width, selected, None)
        for width, selected in zip(widths, wide_index, strict=True)
    ]
    if narrow:
        passes.append((candidates, narrow_index, own_widths))
    for width, selected, row_widths in passes:
        subset = work.index_select(0, selected)
        top_p = params.index_select(0, selected)[:, TOP_P]
        ordered, tokens = top_k_candidates(subset, width, top_p, row_widths)
        truncated = torch.full_like(subset, float("-inf"))
        truncated.scatter_(1, tokens, ordered)
        work.index_copy_(0, selected, truncated)

    # Full-vocabulary top-p for rows without a top-k restriction. The token
    # that first crosses top_p is kept: each drop decision shifts right.
    if len(top_p_rows):
        subset = work.index_select(0, top_p_index)
        ordered, tokens = torch.sort(subset, dim=-1, descending=True)
        top_p = params.index_select(0, top_p_index)[:, TOP_P]
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        over = cumulative > top_p.unsqueeze(1)
        drop = torch.cat(
            (
                torch.zeros(
                    (len(top_p_rows), 1), dtype=torch.bool, device=device
                ),
                over[:, :-1],
            ),
            dim=1,
        )
        ordered.masked_fill_(drop, float("-inf"))
        truncated = torch.full_like(subset, float("-inf"))
        truncated.scatter_(1, tokens, ordered)
        work.index_copy_(0, top_p_index, truncated)

    # Min-p drops tokens below a fraction of the row maximum, compared in log
    # space so the shaped logits are used directly.
    if len(min_p_rows):
        subset = work.index_select(0, min_p_index)
        min_p = params.index_select(0, min_p_index)[:, MIN_P]
        threshold = subset.max(dim=-1).values + torch.log(min_p)
        subset.masked_fill_(subset < threshold.unsqueeze(1), float("-inf"))
        work.index_copy_(0, min_p_index, subset)

    # Typical-p keeps the tokens whose surprise is closest to the row entropy,
    # covering the requested probability mass.
    if len(typical_p_rows):
        subset = work.index_select(0, typical_p_index)
        typical_p = params.index_select(0, typical_p_index)[:, TYPICAL_P]
        probs = torch.softmax(subset, dim=-1)
        log_probs = probs.log()
        entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1, keepdim=True)
        scores = ((-log_probs) - entropy).abs()
        order = torch.argsort(scores, dim=-1)
        cumulative = probs.gather(1, order).cumsum(dim=-1)
        over = cumulative >= typical_p.unsqueeze(1)
        drop = torch.cat(
            (
                torch.zeros(
                    (len(typical_p_rows), 1), dtype=torch.bool, device=device
                ),
                over[:, :-1],
            ),
            dim=1,
        )
        subset.scatter_(
            1, order, subset.gather(1, order).masked_fill(drop, float("-inf"))
        )
        work.index_copy_(0, typical_p_index, subset)

    return work, _valid(work)


__all__ = [
    "TOKEN_CONTINUATION_BIT",
    "TOKEN_VALUE_MASK",
    "Selection",
    "decisions",
    "select_categorical",
    "select_greedy",
    "select_top_k",
    "shape_logits",
]
