"""Batched token selection, speculative acceptance and terminal decisions.

The tasks of one forward select in one call, :func:`select`, whose inputs
are all tensors (:class:`Controls`), so one captured graph serves every
forward of its shape. A task is one sampled call; it owns ``W`` candidate
logits rows (a verification's draft tokens plus its bonus row, padded by
repeating its first row; one row otherwise) and selects with one of three
samplers, chosen per task (``paths``):

- ``GREEDY``: the argmax of its row, after masking the tokens its token set
  excludes;
- ``TOP_K``: its row sampled from its top-k candidates under its own
  parameters (:func:`uniserve.sampling.top_k.sample_top_k`, which a decode
  forward also captures);
- ``CATEGORICAL``: the general path, which shapes the logits
  (:func:`shape_logits`), inverts the categorical CDF with each row's
  precomputed semantic draw (argmax at zero temperature) and resolves
  speculative acceptance.

Each sampler the caller enables runs over every task, and a task keeps its
own sampler's selection, so it selects what that sampler selects for it
alone. Every task then resolves its predicate, finish and transition
decisions as device tensors. With a ``broadcast``, tensor-parallel ranks
replace their selection with the source rank's before those decisions, so
every rank commits identical state.

Per-row values (temperature, cutoffs, penalty strengths, draws) are tensor
columns, and which shaping stages a row takes follows from them. Token sets
are bitmasks: row ``i`` of a ``[n, words]`` int32 matrix holds token
``32 * w + b`` in bit ``b`` of word ``w`` (:func:`unpack`, :func:`members`).
The logit biases of a request are a row of a slot-indexed table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .categorical import sample_categorical
from .top_k import (
    MIN_P,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    TYPICAL_P,
    sample_top_k,
    shape_rows,
    temperature_divisors,
    top_k_candidates,
)

# Token relays carry continuation in bit 31 and the token in the low 31 bits.
TOKEN_CONTINUATION_BIT = 1 << 31
TOKEN_VALUE_MASK = TOKEN_CONTINUATION_BIT - 1

# A task's sampler (``Controls.paths``).
GREEDY, TOP_K_PATH, CATEGORICAL = 0, 1, 2


@dataclass(frozen=True, slots=True)
class Controls:
    """The selection inputs of ``T`` tasks of ``W`` candidate rows each.

    Attributes:
        rows: ``[T, W]`` int64, each candidate row's row of the source
            logits.
        paths: ``[T]`` int64, each task's sampler.
        params: ``[T * W, PARAMETER_COLUMNS]`` float32 parameters
            (:mod:`uniserve.sampling.top_k`), candidate rows task by task.
        draws: ``[T * W]`` float32 uniform draws.
        slots: ``[T]`` int64, each task's request slot: the row of the
            penalty counts and of the bias table it reads.
        active: ``[T]`` bool predicates.
        masks: ``[M, words]`` int32 token sets a masked row may select.
        mask_rows: ``[T * W]`` int64, each candidate row's set of ``masks``,
            or -1 for a row every token may take.
        biased: ``[T]`` bool, tasks whose slot's bias row applies.
        force: ``[T]`` bool, tasks that finish whatever they select.
        finish: ``[T, words]`` int32 finish token sets.
        transitions: ``[T, words]`` int32 token sets that select a
            transition; empty for a task without a transition output.
        drafts: ``[T, W - 1]`` int64 draft tokens, -1 past a task's own.
        terminal: ``[T]`` int64, the one-based length of a task's draft
            prefix ending at its first finish token, or zero.
    """

    rows: torch.Tensor
    paths: torch.Tensor
    params: torch.Tensor
    draws: torch.Tensor
    slots: torch.Tensor
    active: torch.Tensor
    masks: torch.Tensor
    mask_rows: torch.Tensor
    biased: torch.Tensor
    force: torch.Tensor
    finish: torch.Tensor
    transitions: torch.Tensor
    drafts: torch.Tensor
    terminal: torch.Tensor


@dataclass(frozen=True, slots=True)
class Selection:
    """One forward's selections, one entry per task.

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
        scored: The source logits row whose distribution produced the
            emitted token, which reported logprobs score.
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
    scored: torch.Tensor


Broadcast = Callable[[torch.Tensor], object] | None


def unpack(sets: torch.Tensor, vocab: int) -> torch.Tensor:
    """``[n, vocab]`` bool membership of each row of ``[n, words]`` bitmasks."""
    bits = torch.arange(32, dtype=torch.int32, device=sets.device)
    flags = (sets.unsqueeze(-1) >> bits) & 1
    return flags.reshape(int(sets.shape[0]), -1)[:, :vocab] != 0


def members(sets: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Whether each row's token is in its row of ``[n, words]`` bitmasks."""
    tokens = tokens.to(torch.int64)
    words = sets.gather(1, (tokens >> 5).unsqueeze(1)).squeeze(1)
    return ((words.to(torch.int64) >> (tokens & 31)) & 1) != 0


def decisions(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    force: torch.Tensor,
    finish: torch.Tensor,
    transitions: torch.Tensor,
    terminal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finish, continuation, tagged token and transition per task.

    A task finishes when it is valid and active and either forced to
    finish, its token is in its ``finish`` set, or it accepted a terminal
    draft prefix (``terminal``). It continues when valid, active and not
    finished. A transition requires a valid, active selection in the task's
    ``transitions`` set.
    """
    eligible = valid & active
    finish = (force | members(finish, tokens) | terminal) & eligible
    continuation = eligible & ~finish
    tagged = tokens.to(torch.int64) | torch.where(
        continuation, TOKEN_CONTINUATION_BIT, 0
    )
    transition = members(transitions, tokens) & eligible
    return finish, continuation, tagged, transition


def _valid(work: torch.Tensor) -> torch.Tensor:
    """A row is usable with no NaN or +inf and at least one finite entry."""
    return (
        ~torch.isnan(work).any(dim=-1)
        & ~torch.isposinf(work).any(dim=-1)
        & torch.isfinite(work).any(dim=-1)
    )


def select(
    logits: torch.Tensor,
    controls: Controls,
    *,
    candidates: int,
    top_k: bool,
    categorical: bool,
    sorts: bool,
    counts: torch.Tensor,
    bias: torch.Tensor,
    broadcast: Broadcast = None,
) -> Selection:
    """Select a token for each task of ``controls``.

    Args:
        logits: ``[rows, vocab]`` global source logits.
        controls: The tasks' inputs.
        candidates: The top-k sampler's candidate width.
        top_k: Some task takes the ``TOP_K`` sampler.
        categorical: Some task takes the ``CATEGORICAL`` sampler; ``sorts``
            when one of its rows sorts its whole vocabulary (a top-k width
            beyond ``candidates``, a nucleus cutoff without a top-k width or
            a typical-probability cutoff).
        counts: ``[slots, vocab]`` int32 committed token counts.
        bias: ``[slots, vocab]`` float32 logit biases.

    Without ``top_k`` and ``categorical`` every task is greedy. Reported
    logprobs (:mod:`uniserve.sampling.logprobs`) score the ``scored`` rows
    of the unshaped source logits.
    """
    tasks, width = int(controls.rows.shape[0]), int(controls.rows.shape[1])
    vocab = int(logits.shape[1])
    source = logits.index_select(0, controls.rows.reshape(-1))
    masked = controls.mask_rows >= 0
    allowed = unpack(
        controls.masks.index_select(0, controls.mask_rows.clamp_min(0)), vocab
    )
    blocked = masked.unsqueeze(1) & ~allowed
    first = torch.arange(tasks, device=source.device) * width

    # Greedy: the argmax of each task's first row with its excluded tokens
    # masked.
    masked_rows = source.index_select(0, first).float()
    masked_rows.masked_fill_(blocked.index_select(0, first), float("-inf"))
    tokens = torch.argmax(masked_rows, dim=-1)
    valid = _valid(masked_rows)
    accepted = torch.zeros_like(tokens)
    points = accepted + 1
    terminal = torch.zeros(tasks, dtype=torch.bool, device=source.device)
    scored = controls.rows[:, 0]

    if top_k:
        sampled, usable = sample_top_k(
            source.index_select(0, first),
            controls.draws.index_select(0, first),
            controls.params.index_select(0, first),
            candidates,
            counts,
            controls.slots,
        )
        chosen = controls.paths == TOP_K_PATH
        tokens = torch.where(chosen, sampled, tokens)
        valid = torch.where(chosen, usable, valid)

    legal = None
    if categorical:
        row_slots = controls.slots.repeat_interleave(width)
        drafts = controls.drafts
        extra = None
        if width > 1:
            # Each candidate row's penalties count the drafts before it.
            before = torch.arange(width - 1, device=source.device).unsqueeze(
                0
            ) < torch.arange(width, device=source.device).unsqueeze(1)
            extra = torch.where(
                before.unsqueeze(0), drafts.unsqueeze(1), -1
            ).reshape(tasks * width, width - 1)
        work, usable = shape_logits(
            source,
            controls.params,
            slots=row_slots,
            counts=counts,
            extra=extra,
            blocked=blocked,
            biased=controls.biased.repeat_interleave(width),
            bias=bias,
            candidates=candidates,
            sorts=sorts,
        )

        # Temperature-zero rows use deterministic argmax; the others invert
        # the categorical CDF with their precomputed draw. Masked tokens
        # carry zero mass, which the selector never picks.
        probabilities = torch.softmax(work, dim=-1)
        sampled = sample_categorical(probabilities, controls.draws)
        row_tokens = torch.where(
            controls.params[:, TEMPERATURE] > 0.0,
            sampled,
            torch.argmax(work, dim=-1),
        )

        # Speculative acceptance: the longest prefix where row ``i``'s
        # selection equals draft ``i``. Draft padding (-1) equals no
        # selection, and a task's padding rows are never consumed.
        selections = row_tokens.view(tasks, width)
        matches = selections[:, :-1] == drafts
        raw = torch.cumprod(matches.to(torch.long), dim=1).sum(dim=1)

        # Reaching a terminal prefix clamps acceptance to it. Its tasks
        # matched every draft through the prefix, so their selection at the
        # prefix's last row is that terminal draft token.
        prefixes = controls.terminal
        prefixed = prefixes > 0
        reached = prefixed & (raw >= prefixes)
        drafted = torch.where(prefixed, torch.minimum(raw, prefixes), raw)
        terminal_tokens = selections.gather(
            1, (prefixes - 1).clamp_min(0).unsqueeze(1)
        ).squeeze(1)
        emitted = torch.where(
            reached,
            terminal_tokens,
            selections.gather(1, drafted.unsqueeze(1)).squeeze(1),
        )
        chosen = controls.paths == CATEGORICAL
        tokens = torch.where(chosen, emitted, tokens)
        accepted = torch.where(chosen, drafted, accepted)
        points = torch.where(
            chosen, drafted + (~reached).to(dtype=torch.long), points
        )
        terminal = chosen & reached
        legal = usable.view(tasks, width)

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

    if legal is not None:
        # A categorical task is valid only when every consumed speculative
        # point retained a legal candidate: its first ``points`` rows.
        positions = torch.arange(width, device=source.device)
        consumed = positions.unsqueeze(0) < points.unsqueeze(1)
        speculative = (legal | ~consumed).all(dim=1)
        chosen = controls.paths == CATEGORICAL
        valid = torch.where(chosen, speculative, valid)
        # A terminal selection emits the finish draft token, which the
        # verification row before the output row scored.
        column = (accepted - terminal.to(dtype=torch.long)).clamp_min(0)
        scored = torch.where(
            chosen,
            controls.rows.gather(1, column.unsqueeze(1)).squeeze(1),
            scored,
        )

    finish, continuation, tagged, transition = decisions(
        tokens,
        valid,
        controls.active,
        controls.force,
        controls.finish,
        controls.transitions,
        terminal,
    )
    return Selection(
        tokens,
        valid,
        controls.active,
        finish,
        continuation,
        tagged,
        accepted,
        points,
        transition,
        scored,
    )


def shape_logits(
    source: torch.Tensor,
    params: torch.Tensor,
    *,
    slots: torch.Tensor,
    counts: torch.Tensor,
    extra: torch.Tensor | None,
    blocked: torch.Tensor,
    biased: torch.Tensor,
    bias: torch.Tensor,
    candidates: int,
    sorts: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the canonical shaping and truncation order to logits rows.

    Stages, in order, on float32 rows: penalties over each row's slot
    counts and its ``extra`` tokens (:func:`~uniserve.sampling.top_k.
    shape_rows`), the ``blocked`` tokens' exclusion, the slot's logit bias
    on ``biased`` rows, temperature, top-k with nested top-p, full
    vocabulary top-p, min-p and typical-p. A stage changes only the rows its
    parameter selects, so each row is shaped as if alone: penalties and the
    bias are exact for neutral rows, and the cutoffs are applied where a
    row's parameter asks for them. Penalties keep excluded (-inf) entries,
    so excluding after penalizing equals excluding first.

    Rows whose top-k width is at most ``candidates`` keep their candidates
    as the top-k sampler (``sample_top_k``) does: the shared candidate
    width, then each row's own width, so both selectors keep the same
    tokens, including the order among equal logits at a cutoff. The stages
    that sort a whole vocabulary row (a wider top-k width, a nucleus cutoff
    without a top-k width, a typical-probability cutoff) run only with
    ``sorts``.

    Args:
        source: ``[rows, vocab]`` global logits.
        params: ``[rows, PARAMETER_COLUMNS]`` float parameters.
        slots: ``[rows]`` int64, each row's request slot.
        counts: ``[slots, vocab]`` int32 committed token counts.
        extra: ``[rows, E]`` int64 tokens a row's penalties count besides
            its slot's, or ``None``.
        blocked: ``[rows, vocab]`` bool, the tokens each row may not select.
        biased: ``[rows]`` bool, rows that add their slot's bias row.
        bias: ``[slots, vocab]`` float32 logit biases.

    Returns:
        ``(work, valid)``: the shaped ``[rows, vocab]`` float32 logits with
        dropped tokens at -inf, and whether each row keeps a usable
        distribution.
    """
    device = source.device
    vocab = int(source.shape[1])
    work = shape_rows(source, params, counts, slots, extra, divide=False)
    work.masked_fill_(blocked, float("-inf"))
    work = torch.where(
        biased.unsqueeze(1), work + bias.index_select(0, slots), work
    )

    # A zero temperature divides by one; callers select those rows by argmax.
    work = work / temperature_divisors(params).unsqueeze(1)

    # Top-k truncation; top-p applies within each retained top-k set.
    top_k = params[:, TOP_K].to(torch.int64)
    top_p = params[:, TOP_P]
    narrow = (top_k > 0) & (top_k <= candidates)
    ordered, tokens = top_k_candidates(
        work, candidates, top_p, top_k.clamp(1, candidates)
    )
    truncated = torch.full_like(work, float("-inf")).scatter_(
        1, tokens, ordered
    )
    work = torch.where(narrow.unsqueeze(1), truncated, work)
    if sorts:
        wide = top_k > candidates
        ordered, tokens = top_k_candidates(
            work, vocab, top_p, torch.where(wide, top_k, vocab)
        )
        truncated = torch.full_like(work, float("-inf")).scatter_(
            1, tokens, ordered
        )
        work = torch.where(wide.unsqueeze(1), truncated, work)

        # Full-vocabulary top-p for rows without a top-k width. The token
        # that first crosses top_p is kept: each drop decision shifts right.
        nucleus = (top_k == 0) & (top_p > 0.0) & (top_p < 1.0)
        ordered, order = torch.sort(work, dim=-1, descending=True)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        over = cumulative > top_p.unsqueeze(1)
        drop = torch.cat((torch.zeros_like(over[:, :1]), over[:, :-1]), dim=1)
        ordered = ordered.masked_fill(drop, float("-inf"))
        truncated = torch.full_like(work, float("-inf")).scatter_(
            1, order, ordered
        )
        work = torch.where(nucleus.unsqueeze(1), truncated, work)

    # Min-p drops tokens below a fraction of the row maximum, compared in log
    # space so the shaped logits are used directly.
    min_p = params[:, MIN_P]
    threshold = work.max(dim=-1).values + torch.log(min_p)
    cut = (min_p > 0.0).unsqueeze(1) & (work < threshold.unsqueeze(1))
    work = work.masked_fill(cut, float("-inf"))

    if sorts:
        # Typical-p keeps the tokens whose surprise is closest to the row
        # entropy, covering the requested probability mass.
        typical_p = params[:, TYPICAL_P]
        typical = typical_p < 1.0
        probs = torch.softmax(work, dim=-1)
        log_probs = probs.log()
        entropy = -(probs * log_probs).nan_to_num(0.0).sum(dim=-1, keepdim=True)
        scores = ((-log_probs) - entropy).abs()
        order = torch.argsort(scores, dim=-1)
        cumulative = probs.gather(1, order).cumsum(dim=-1)
        over = cumulative >= typical_p.unsqueeze(1)
        drop = torch.cat(
            (
                torch.zeros(
                    (int(work.shape[0]), 1), dtype=torch.bool, device=device
                ),
                over[:, :-1],
            ),
            dim=1,
        )
        truncated = work.clone()
        truncated.scatter_(
            1, order, work.gather(1, order).masked_fill(drop, float("-inf"))
        )
        work = torch.where(typical.unsqueeze(1), truncated, work)

    return work, _valid(work)


__all__ = [
    "CATEGORICAL",
    "GREEDY",
    "TOKEN_CONTINUATION_BIT",
    "TOKEN_VALUE_MASK",
    "TOP_K_PATH",
    "Controls",
    "Selection",
    "decisions",
    "members",
    "select",
    "shape_logits",
    "unpack",
]
