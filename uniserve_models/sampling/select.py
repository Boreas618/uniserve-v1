"""Batched token selection, speculative acceptance and terminal decisions.

Tasks are sampled calls; each owns one or more candidate logits rows (a
verification task owns one row per draft token plus its bonus row). The
selectors take the tasks of one homogeneous group:

- :func:`select_greedy`: argmax of each task's single row, optionally after
  masking suppressed tokens;
- :func:`select_top_k`: the compiled fixed-``k`` sampler
  (``uniserve_models.sampling.sample_top_k``);
- :func:`select_categorical`: the general path that shapes logits
  (:func:`shape_logits`), inverts the categorical CDF with each row's
  precomputed semantic draw (argmax at zero temperature) and resolves
  speculative acceptance.

Every selector then resolves predicates, finish and transition decisions as
device tensors. With a ``broadcast``, tensor-parallel ranks replace their
selection with the source rank's before those decisions, so every rank
commits identical state.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .categorical import sample_categorical
from .logprobs import LogprobRequest, logprob_details
from .top_k import sample_top_k, top_k_candidates

# Token relays carry continuation in bit 31 and the token in the low 31 bits.
TOKEN_CONTINUATION_BIT = 1 << 31
TOKEN_VALUE_MASK = TOKEN_CONTINUATION_BIT - 1

# Columns of the per-row float sampling parameters.
TEMPERATURE, TOP_P, MIN_P, TYPICAL_P = 0, 1, 2, 3
REPETITION, FREQUENCY, PRESENCE = 4, 5, 6
PARAMETER_COLUMNS = 8


@dataclass(frozen=True, slots=True)
class Task:
    """One sampled call's host controls.

    Attributes:
        row: First candidate row in the step's logits.
        rows: Candidate rows: drafts plus the bonus row.
        slot: Request slot (penalty counts row).
        force_finish: Grammar state forcing termination.
        return_transition: The call publishes a transition decision.
        tagged: The call is gated by its request's continuation flag.
        penalties: Repetition, frequency or presence penalties apply.
        top_k: Top-k width, zero when disabled.
        top_p: A nucleus cutoff below one applies.
        min_p: A minimum-probability cutoff applies.
        typical_p: A typical-probability cutoff below one applies.
        finish_ids: Terminal tokens, sorted and deduplicated.
        transition_ids: Tokens that select a transition.
        suppress: Tokens masked from selection.
        logit_bias: ``(token, bias)`` additive offsets.
        logprobs: What the call reports, or ``None``.
        drafts: Draft tokens a verification compares row by row.
        terminal_prefix: One-based length of the draft prefix ending at the
            first terminal draft token, or ``None``.
        allowed: Per candidate row, the only selectable tokens, or ``None``.
    """

    row: int
    rows: int
    slot: int
    force_finish: bool = False
    return_transition: bool = False
    tagged: bool = False
    penalties: bool = False
    top_k: int = 0
    top_p: bool = False
    min_p: bool = False
    typical_p: bool = False
    finish_ids: tuple[int, ...] = ()
    transition_ids: tuple[int, ...] = ()
    suppress: tuple[int, ...] = ()
    logit_bias: tuple[tuple[int, float], ...] = ()
    logprobs: LogprobRequest | None = None
    drafts: tuple[int, ...] = ()
    terminal_prefix: int | None = None
    allowed: tuple[tuple[int, ...] | None, ...] = (None,)


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


def _member(
    tokens: torch.Tensor, sets: Sequence[tuple[int, ...]]
) -> torch.Tensor:
    """Whether each task's token is in its own token set."""
    first = sets[0]
    if all(values == first for values in sets[1:]):
        # One shared set: a vectorized comparison per member.
        matched = torch.zeros_like(tokens, dtype=torch.bool)
        for token in first:
            matched |= tokens == token
        return matched
    # Mixed sets: a padded matrix with -1 as the non-token sentinel.
    width = max(1, *(len(values) for values in sets))
    padded = torch.tensor(
        [(*values, *((-1,) * (width - len(values)))) for values in sets],
        dtype=tokens.dtype,
        device=tokens.device,
    )
    return tokens.unsqueeze(1).eq(padded).any(dim=1)


def decisions(
    tasks: Sequence[Task],
    tokens: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    terminal: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finish, continuation, tagged token and transition per task.

    A task finishes when it is valid and active and either forced to
    finish, its token is in its finish set, or it accepted a terminal draft
    prefix. It continues when valid, active and not finished. A transition
    requires a valid, active selection in the task's transition set.
    """
    eligible = valid & active
    forced = torch.tensor(
        [task.force_finish for task in tasks],
        dtype=torch.bool,
        device=tokens.device,
    )
    finish = forced | _member(tokens, [task.finish_ids for task in tasks])
    if terminal is not None:
        finish = finish | terminal
    finish = finish & eligible
    continuation = eligible & ~finish
    tagged = tokens.to(torch.int64) | torch.where(
        continuation, TOKEN_CONTINUATION_BIT, 0
    )
    participates = torch.tensor(
        [task.return_transition for task in tasks],
        dtype=torch.bool,
        device=tokens.device,
    )
    transition = (
        _member(tokens, [task.transition_ids for task in tasks])
        & participates
        & eligible
    )
    return finish, continuation, tagged, transition


def _valid(work: torch.Tensor) -> torch.Tensor:
    """A row is usable with no NaN or +inf and at least one finite entry."""
    return (
        ~torch.isnan(work).any(dim=-1)
        & ~torch.isposinf(work).any(dim=-1)
        & torch.isfinite(work).any(dim=-1)
    )


def select_greedy(
    tasks: Sequence[Task],
    logits: torch.Tensor,
    active: torch.Tensor,
    *,
    suppress: bool,
    broadcast: Broadcast = None,
) -> Selection:
    """Argmax of each task's single ``[tasks, vocab]`` logits row.

    With ``suppress``, each task's suppressed ids are masked on a float32
    copy before the argmax.
    """
    work = logits
    if suppress:
        work = logits.to(dtype=torch.float32, copy=True)
        vocab = int(work.shape[1])
        for row, task in enumerate(tasks):
            for token in dict.fromkeys(task.suppress):
                if 0 <= token < vocab:
                    work[row, token].fill_(float("-inf"))
    tokens = torch.argmax(work, dim=-1)
    valid = _valid(work)
    if broadcast is not None:
        broadcast(tokens)
    finish, continuation, tagged, transition = decisions(
        tasks, tokens, valid, active
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
    tasks: Sequence[Task],
    logits: torch.Tensor,
    draws: torch.Tensor,
    params: torch.Tensor,
    active: torch.Tensor,
    top_k: int,
    *,
    broadcast: Broadcast = None,
) -> Selection:
    """Sample a group sharing one top-k width with the compiled kernel.

    Logprobs, when any task requests them, are scored from the general
    shaping path while the kernel's tokens are kept.
    """
    tokens, valid = sample_top_k(
        logits, draws, params[:, [TEMPERATURE, TOP_P, MIN_P]], int(top_k)
    )
    if broadcast is not None:
        broadcast(tokens)
    finish, continuation, tagged, transition = decisions(
        tasks, tokens, valid, active
    )
    packed = None
    if any(task.logprobs is not None for task in tasks):
        # Observations do not select another numerical sampler: even draws
        # at a floating-point CDF boundary keep the kernel's chosen token.
        work, _ = shape_logits(logits, params, _rows(tasks), None)
        packed = _logprobs(
            work,
            torch.arange(len(tasks), device=tokens.device),
            tokens,
            tasks,
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
        packed,
    )


@dataclass(frozen=True, slots=True)
class _Row:
    """Shaping controls of one candidate row."""

    task: Task
    allowed: tuple[int, ...] | None


def _rows(tasks: Sequence[Task]) -> list[_Row]:
    return [_Row(task, allowed) for task in tasks for allowed in task.allowed]


def _logprobs(
    work: torch.Tensor,
    rows: torch.Tensor,
    tokens: torch.Tensor,
    tasks: Sequence[Task],
) -> torch.Tensor:
    requesting = [
        index for index, task in enumerate(tasks) if task.logprobs is not None
    ]
    selected = torch.tensor(requesting, dtype=torch.long, device=work.device)
    return logprob_details(
        work,
        rows.index_select(0, selected),
        tokens.index_select(0, selected),
        [task.logprobs for task in tasks if task.logprobs is not None],
    )


def select_categorical(
    tasks: Sequence[Task],
    logits: torch.Tensor,
    draws: torch.Tensor,
    params: torch.Tensor,
    active: torch.Tensor,
    penalty_counts: Sequence[torch.Tensor | None],
    *,
    broadcast: Broadcast = None,
) -> Selection:
    """Sample tasks through the general shaping path.

    ``logits`` holds every task's candidate rows back to back, ``draws`` and
    ``params`` one entry per candidate row and ``penalty_counts`` each
    row's token counts (``None`` without penalties).

    Speculative tasks accept the longest prefix where row ``i``'s selection
    equals draft ``i``. Without a terminal prefix, the output row is the
    first mismatching row, whose selection is the correction, or the bonus
    row after a full match. Reaching a terminal prefix clamps acceptance to
    it and emits its final draft token instead of the output row's
    selection, so the emitted-token count adds no extra token.
    """
    device = logits.device
    rows = _rows(tasks)
    work, valid = shape_logits(
        logits.float(), params, rows, list(penalty_counts)
    )

    # Temperature-zero rows use deterministic argmax; the others invert the
    # categorical CDF with their precomputed draw. Masked tokens carry zero
    # mass, which the selector never picks.
    probabilities = torch.softmax(work, dim=-1)
    sampled = sample_categorical(probabilities, draws)
    row_tokens = torch.where(
        params[:, TEMPERATURE] > 0.0, sampled, torch.argmax(work, dim=-1)
    )

    offsets = []
    offset = 0
    for task in tasks:
        offsets.append(offset)
        offset += task.rows
    accepted_counts, point_counts, terminals = [], [], []
    output_rows, terminal_tokens = [], []
    for task, first in zip(tasks, offsets, strict=True):
        if task.drafts:
            draft = torch.tensor(
                task.drafts, dtype=row_tokens.dtype, device=device
            )
            matches = row_tokens[first : first + len(task.drafts)] == draft
            raw = torch.cumprod(matches.to(torch.long), dim=0).sum()
        else:
            draft = torch.empty((0,), dtype=row_tokens.dtype, device=device)
            raw = torch.zeros((), dtype=torch.long, device=device)
        if task.terminal_prefix is None:
            accepted = raw
            terminal = torch.zeros((), dtype=torch.bool, device=device)
            terminal_token = torch.zeros(
                (), dtype=row_tokens.dtype, device=device
            )
        else:
            prefix = int(task.terminal_prefix)
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
            valid[first : first + task.rows][
                torch.arange(task.rows, device=device) < points[index]
            ].all()
            for index, (task, first) in enumerate(
                zip(tasks, offsets, strict=True)
            )
        )
    )
    finish, continuation, tagged, transition = decisions(
        tasks, tokens, task_valid, active, terminal
    )

    # A terminal selection emits the finish draft token, which the
    # verification row before the output row scored.
    packed = None
    if any(task.logprobs is not None for task in tasks):
        packed = _logprobs(
            work, output - terminal.to(dtype=torch.long), tokens, tasks
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
    rows: Sequence[_Row],
    penalty_counts: Sequence[torch.Tensor | None] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the canonical shaping and truncation order to logits rows.

    Stages run in this order on a float32 copy: allowed-token restriction,
    suppression, penalties, logit bias, temperature, top-k with nested
    top-p, full-vocabulary top-p, min-p and typical-p. Float values come
    from the ``params`` columns; the host controls decide which rows take
    each stage. Out-of-vocabulary ids in the allowed, suppressed and bias
    sets are ignored.

    Returns:
        ``(work, valid)``: the shaped ``[rows, vocab]`` float32 logits with
        dropped tokens at -inf, and whether each row keeps a usable
        distribution.
    """
    work = work.to(dtype=torch.float32, copy=True)
    count, vocab = (int(value) for value in work.shape)
    device = work.device
    if len(rows) != count or int(params.shape[0]) != count:
        raise ValueError("sampling controls do not align with logits rows")

    # Allowed-token restriction masks every other vocabulary entry.
    allowed_rows: list[int] = []
    allowed_flat: list[int] = []
    for index, row in enumerate(rows):
        if row.allowed is None:
            continue
        allowed_rows.append(index)
        allowed_flat.extend(
            index * vocab + token
            for token in dict.fromkeys(row.allowed)
            if 0 <= token < vocab
        )
    if allowed_rows:
        mask = torch.ones_like(work, dtype=torch.bool)
        restricted = torch.tensor(allowed_rows, dtype=torch.long, device=device)
        mask.index_fill_(0, restricted, False)
        mask.reshape(-1)[
            torch.tensor(allowed_flat, dtype=torch.long, device=device)
        ] = True
        work.masked_fill_(~mask, float("-inf"))

    # Suppressed tokens are masked regardless of any other shaping.
    suppressed = tuple(
        index * vocab + token
        for index, row in enumerate(rows)
        for token in dict.fromkeys(row.task.suppress)
        if 0 <= token < vocab
    )
    if suppressed:
        work.reshape(-1).index_fill_(
            0,
            torch.tensor(suppressed, dtype=torch.long, device=device),
            float("-inf"),
        )

    # Penalties apply to tokens with a positive count: repetition is
    # multiplicative and sign-aware, frequency scales with the count and
    # presence is flat. Masked (-inf) entries are preserved.
    penalized = [
        index
        for index, value in enumerate(penalty_counts or ())
        if value is not None
    ]
    if penalized:
        selected = torch.tensor(penalized, dtype=torch.long, device=device)
        counts = torch.stack(
            [value for value in penalty_counts or () if value is not None]
        ).to(dtype=work.dtype)
        values = params.index_select(0, selected)
        repetition = values[:, REPETITION].unsqueeze(1)
        frequency = values[:, FREQUENCY].unsqueeze(1)
        presence = values[:, PRESENCE].unsqueeze(1)
        current = work.index_select(0, selected)
        seen = counts > 0
        repeated = torch.where(
            current > 0.0, current / repetition, current * repetition
        )
        adjusted = repeated - frequency * counts - presence
        apply = seen & ~torch.isneginf(current)
        work.index_copy_(0, selected, torch.where(apply, adjusted, current))

    # Logit bias adds per-token offsets on top of the penalized logits.
    bias_indexes: list[int] = []
    bias_values: list[float] = []
    for index, row in enumerate(rows):
        for token, bias in row.task.logit_bias:
            if 0 <= token < vocab:
                bias_indexes.append(index * vocab + token)
                bias_values.append(float(bias))
    if bias_indexes:
        work.reshape(-1).index_put_(
            (torch.tensor(bias_indexes, dtype=torch.long, device=device),),
            torch.tensor(bias_values, dtype=work.dtype, device=device),
            accumulate=True,
        )

    # A zero temperature divides by one; callers select those rows by argmax.
    temperatures = params[:, TEMPERATURE]
    divisors = torch.where(
        temperatures > 0.0,
        temperatures,
        torch.ones((), dtype=work.dtype, device=device),
    )
    work.div_(divisors.unsqueeze(1))

    # Top-k truncation; top-p applies within each retained top-k set, so rows
    # sharing a width are truncated in one gather/sort pass. The candidates
    # are shared with ``sample_top_k``, so both paths keep the same tokens.
    widths: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if 0 < row.task.top_k < vocab:
            widths[row.task.top_k].append(index)
    for top_k, members in widths.items():
        selected = torch.tensor(members, dtype=torch.long, device=device)
        subset = work.index_select(0, selected)
        top_p = params.index_select(0, selected)[:, TOP_P]
        ordered, tokens = top_k_candidates(subset, top_k, top_p)
        truncated = torch.full_like(subset, float("-inf"))
        truncated.scatter_(1, tokens, ordered)
        work.index_copy_(0, selected, truncated)

    # Full-vocabulary top-p for rows without a top-k restriction. The token
    # that first crosses top_p is kept: each drop decision shifts right.
    nucleus = [
        index
        for index, row in enumerate(rows)
        if row.task.top_p and not 0 < row.task.top_k < vocab
    ]
    if nucleus:
        selected = torch.tensor(nucleus, dtype=torch.long, device=device)
        subset = work.index_select(0, selected)
        ordered, tokens = torch.sort(subset, dim=-1, descending=True)
        top_p = params.index_select(0, selected)[:, TOP_P]
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        over = cumulative > top_p.unsqueeze(1)
        drop = torch.cat(
            (
                torch.zeros((len(nucleus), 1), dtype=torch.bool, device=device),
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
    floor = [index for index, row in enumerate(rows) if row.task.min_p]
    if floor:
        selected = torch.tensor(floor, dtype=torch.long, device=device)
        subset = work.index_select(0, selected)
        min_p = params.index_select(0, selected)[:, MIN_P]
        threshold = subset.max(dim=-1).values + torch.log(min_p)
        subset.masked_fill_(subset < threshold.unsqueeze(1), float("-inf"))
        work.index_copy_(0, selected, subset)

    # Typical-p keeps the tokens whose surprise is closest to the row entropy,
    # covering the requested probability mass.
    typical = [index for index, row in enumerate(rows) if row.task.typical_p]
    if typical:
        selected = torch.tensor(typical, dtype=torch.long, device=device)
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
                torch.zeros((len(typical), 1), dtype=torch.bool, device=device),
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
    "PARAMETER_COLUMNS",
    "TOKEN_CONTINUATION_BIT",
    "TOKEN_VALUE_MASK",
    "Selection",
    "Task",
    "decisions",
    "select_categorical",
    "select_greedy",
    "select_top_k",
    "shape_logits",
]
