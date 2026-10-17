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
padded with -1.
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


def _index(values: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.long, device=device)


def _member(tokens: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    """Whether each task's token is in its row of ``ids``."""
    sets = torch.tensor(
        list(ids), dtype=tokens.dtype, device=tokens.device
    ).view(int(tokens.numel()), -1)
    return tokens.unsqueeze(1).eq(sets).any(dim=1)


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
    forced = torch.tensor(
        list(force_finish), dtype=torch.long, device=tokens.device
    ).bool()
    finish = forced | _member(tokens, finish_ids)
    if terminal is not None:
        finish = finish | terminal
    finish = finish & eligible
    continuation = eligible & ~finish
    tagged = tokens.to(torch.int64) | torch.where(
        continuation, TOKEN_CONTINUATION_BIT, 0
    )
    transition = _member(tokens, transition_ids) & eligible
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
        work.reshape(-1).index_fill_(
            0, _index(suppressed, work.device), float("-inf")
        )
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


def _logprobs(
    work: torch.Tensor,
    rows: torch.Tensor,
    tokens: torch.Tensor,
    logprob_tasks: Sequence[int],
    logprob_requests: Sequence[LogprobRequest],
) -> torch.Tensor:
    selected = _index(logprob_tasks, work.device)
    return logprob_details(
        work,
        rows.index_select(0, selected),
        tokens.index_select(0, selected),
        logprob_requests,
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

    counts = [int(count) for count in row_counts]
    offsets, drafted = [], []
    offset = taken = 0
    for count in counts:
        offsets.append(offset)
        drafted.append(
            tuple(int(token) for token in drafts[taken : taken + count - 1])
        )
        offset += count
        taken += count - 1
    accepted_counts, point_counts, terminals = [], [], []
    output_rows, terminal_tokens = [], []
    for first, tokens_, prefix in zip(
        offsets, drafted, terminal_prefixes, strict=True
    ):
        if tokens_:
            draft = torch.tensor(tokens_, dtype=row_tokens.dtype, device=device)
            matches = row_tokens[first : first + len(tokens_)] == draft
            raw = torch.cumprod(matches.to(torch.long), dim=0).sum()
        else:
            draft = torch.empty((0,), dtype=row_tokens.dtype, device=device)
            raw = torch.zeros((), dtype=torch.long, device=device)
        prefix = int(prefix)
        if not prefix:
            accepted = raw
            terminal = torch.zeros((), dtype=torch.bool, device=device)
            terminal_token = torch.zeros(
                (), dtype=row_tokens.dtype, device=device
            )
        else:
            accepted = raw.clamp_max(prefix)
            terminal = raw >= prefix
            terminal_token = draft[prefix - 1]
        accepted_counts.append(accepted)
        point_counts.append(accepted + (~terminal).to(dtype=torch.long))
        terminals.append(terminal)
        output_rows.append(accepted + first)
        terminal_tokens.append(terminal_token)

    output = torch.stack(output_rows)
    terminal = torch.stack(terminals)
    tokens = torch.where(
        terminal,
        torch.stack(terminal_tokens),
        row_tokens.index_select(0, output),
    )
    accepted = torch.stack(accepted_counts)
    points = torch.stack(point_counts)

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
    # legal candidate under its sampling constraints.
    task_valid = torch.stack(
        tuple(
            valid[first : first + count][
                torch.arange(count, device=device) < points[index]
            ].all()
            for index, (count, first) in enumerate(
                zip(counts, offsets, strict=True)
            )
        )
    )
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
        packed = _logprobs(
            logits.float(),
            output - terminal.to(dtype=torch.long),
            tokens,
            logprob_tasks,
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

    # Allowed-token restriction masks every other vocabulary entry.
    if len(allowed_rows):
        mask = torch.ones_like(work, dtype=torch.bool)
        mask.index_fill_(0, _index(allowed_rows, device), False)
        mask.reshape(-1)[_index(allowed, device)] = True
        work.masked_fill_(~mask, float("-inf"))

    # Suppressed tokens are masked regardless of any other shaping.
    if len(suppressed):
        work.reshape(-1).index_fill_(
            0, _index(suppressed, device), float("-inf")
        )

    # Penalties apply to tokens with a positive count (see ``penalize``).
    if len(penalty_rows) and penalty_counts is not None:
        selected = _index(penalty_rows, device)
        work.index_copy_(
            0,
            selected,
            penalize(
                work.index_select(0, selected),
                params.index_select(0, selected),
                penalty_counts,
            ),
        )

    # Logit bias adds per-token offsets on top of the penalized logits.
    if len(bias):
        work.reshape(-1).index_put_(
            (_index(bias, device),),
            torch.tensor(list(bias_values), dtype=work.dtype, device=device),
            accumulate=True,
        )

    # A zero temperature divides by one; callers select those rows by argmax.
    work.div_(temperature_divisors(params).unsqueeze(1))

    # Top-k truncation; top-p applies within each retained top-k set, so rows
    # sharing a width are truncated in one gather/sort pass. The candidates
    # are shared with ``sample_top_k``, so both paths keep the same tokens.
    widths: dict[int, list[int]] = defaultdict(list)
    for index, top_k in enumerate(top_ks):
        if int(top_k) > 0:
            widths[int(top_k)].append(index)
    for top_k, members in widths.items():
        selected = _index(members, device)
        subset = work.index_select(0, selected)
        top_p = params.index_select(0, selected)[:, TOP_P]
        ordered, tokens = top_k_candidates(subset, top_k, top_p)
        truncated = torch.full_like(subset, float("-inf"))
        truncated.scatter_(1, tokens, ordered)
        work.index_copy_(0, selected, truncated)

    # Full-vocabulary top-p for rows without a top-k restriction. The token
    # that first crosses top_p is kept: each drop decision shifts right.
    if len(top_p_rows):
        selected = _index(top_p_rows, device)
        subset = work.index_select(0, selected)
        ordered, tokens = torch.sort(subset, dim=-1, descending=True)
        top_p = params.index_select(0, selected)[:, TOP_P]
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
        work.index_copy_(0, selected, truncated)

    # Min-p drops tokens below a fraction of the row maximum, compared in log
    # space so the shaped logits are used directly.
    if len(min_p_rows):
        selected = _index(min_p_rows, device)
        subset = work.index_select(0, selected)
        min_p = params.index_select(0, selected)[:, MIN_P]
        threshold = subset.max(dim=-1).values + torch.log(min_p)
        subset.masked_fill_(subset < threshold.unsqueeze(1), float("-inf"))
        work.index_copy_(0, selected, subset)

    # Typical-p keeps the tokens whose surprise is closest to the row entropy,
    # covering the requested probability mass.
    if len(typical_p_rows):
        selected = _index(typical_p_rows, device)
        subset = work.index_select(0, selected)
        typical_p = params.index_select(0, selected)[:, TYPICAL_P]
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
        work.index_copy_(0, selected, subset)

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
