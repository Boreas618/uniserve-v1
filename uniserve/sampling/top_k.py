"""Capture-safe sampling of each row from its top-k candidates.

:func:`sample_top_k` is the sampler of every row whose shaping a per-row
parameter decides: repetition, frequency and presence penalties,
temperature, top-k with nested top-p, and min-p, in the canonical order of
the general path (:func:`uniserve.sampling.select.shape_logits`). Every
changing value is the contents of a tensor: the per-row parameters
(``params``, one row of ``PARAMETER_COLUMNS``), one precomputed uniform
draw per row, and the committed token counts the penalties read. The
candidate width is the only Python integer. The sampler therefore has
fixed shapes, no host synchronization and no random draw of its own, and
a decode forward captures it into its CUDA graph exactly as it runs
eagerly.

Penalties and temperature are applied in one pass over the vocabulary
(:func:`shape_rows`). On CUDA the pass is a Triton kernel whose float32
operations use explicitly rounded intrinsics, so it reproduces the
separately rounded steps of :func:`penalize` and the temperature division
bit for bit; elsewhere the same steps run as tensor operations.
"""

from __future__ import annotations

import torch
from uniserve_kernels.triton import launchable

from .categorical import sample_categorical

try:  # pragma: no cover - depends on the installed accelerator stack.
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except Exception:  # pragma: no cover
    triton = None
    tl = None
    libdevice = None

# Columns of the per-row float sampling parameters. ``TOP_K`` holds the
# row's top-k width inside the vocabulary, or zero without one; widths are
# integers far below 2**24, which float32 represents exactly.
TEMPERATURE, TOP_P, MIN_P, TYPICAL_P = 0, 1, 2, 3
REPETITION, FREQUENCY, PRESENCE, TOP_K = 4, 5, 6, 7
PARAMETER_COLUMNS = 8

# Vocabulary entries one program of the shaping kernel covers.
_SHAPE_BLOCK = 2048


def penalize(
    work: torch.Tensor, params: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    """Apply repetition, frequency and presence penalties to logits rows.

    Penalties apply to tokens with a positive count: repetition is
    multiplicative and sign-aware, frequency scales with the count and
    presence is flat. Masked (-inf) entries are preserved. A row with the
    neutral parameters (repetition 1, frequency 0, presence 0) is returned
    unchanged bit for bit, whatever its counts.

    Args:
        work: ``[rows, vocab]`` float32 logits.
        params: ``[rows, PARAMETER_COLUMNS]`` float32 parameters.
        counts: ``[rows, vocab]`` integer committed token counts.

    Returns:
        The penalized ``[rows, vocab]`` float32 logits.
    """
    counts = counts.to(dtype=work.dtype)
    repetition = params[:, REPETITION].unsqueeze(1)
    frequency = params[:, FREQUENCY].unsqueeze(1)
    presence = params[:, PRESENCE].unsqueeze(1)
    seen = counts > 0
    repeated = torch.where(work > 0.0, work / repetition, work * repetition)
    adjusted = repeated - frequency * counts - presence
    apply = seen & ~torch.isneginf(work)
    return torch.where(apply, adjusted, work)


def temperature_divisors(params: torch.Tensor) -> torch.Tensor:
    """Each row's temperature; a zero temperature divides by one.

    Callers select zero-temperature rows by argmax.
    """
    temperatures = params[:, TEMPERATURE]
    return torch.where(
        temperatures > 0.0, temperatures, torch.ones_like(temperatures)
    )


if triton is not None:

    @triton.jit(
        do_not_specialize=[
            "vocab",
            "logits_stride",
            "counts_stride",
            "params_stride",
            "out_stride",
        ]
    )
    def _shape_kernel(
        logits_ptr,
        counts_ptr,
        slots_ptr,
        params_ptr,
        out_ptr,
        vocab,
        logits_stride,
        counts_stride,
        params_stride,
        out_stride,
        block: tl.constexpr,
    ):
        """Penalize and scale one block of one row's logits.

        The program of row ``r`` reads the counts of slot ``slots[r]``.
        Every float32 operation is a separately rounded intrinsic, as each
        step of :func:`penalize` and the temperature division is one tensor
        operation, so the result equals theirs bit for bit.
        """
        row = tl.program_id(0)
        columns = tl.program_id(1) * block + tl.arange(0, block)
        mask = columns < vocab

        current = tl.load(
            logits_ptr + row * logits_stride + columns, mask=mask
        ).to(tl.float32)
        slot = tl.load(slots_ptr + row).to(tl.int64)
        counts = tl.load(
            counts_ptr + slot * counts_stride + columns, mask=mask, other=0
        ).to(tl.float32)
        parameters = params_ptr + row * params_stride
        temperature = tl.load(parameters + 0)
        repetition = tl.load(parameters + 4)
        frequency = tl.load(parameters + 5)
        presence = tl.load(parameters + 6)

        repeated = tl.where(
            current > 0.0,
            libdevice.div_rn(current, repetition),
            libdevice.mul_rn(current, repetition),
        )
        adjusted = libdevice.sub_rn(
            libdevice.sub_rn(repeated, libdevice.mul_rn(frequency, counts)),
            presence,
        )
        apply = (counts > 0.0) & (current != float("-inf"))
        penalized = tl.where(apply, adjusted, current)
        divisor = tl.where(temperature > 0.0, temperature, 1.0)
        tl.store(
            out_ptr + row * out_stride + columns,
            libdevice.div_rn(penalized, divisor),
            mask=mask,
        )


def shape_rows(
    logits: torch.Tensor,
    params: torch.Tensor,
    counts: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Penalize and temperature-scale logits rows in one pass.

    Args:
        logits: ``[rows, vocab]`` logits in any floating dtype.
        params: ``[rows, PARAMETER_COLUMNS]`` float32 parameters.
        counts: ``[slots, vocab]`` int32 committed token counts of every
            request slot.
        slots: ``[rows]`` int64, the slot whose counts each row reads.

    Returns:
        ``[rows, vocab]`` float32 logits: :func:`penalize`, then division by
        :func:`temperature_divisors`.
    """
    rows, vocab = int(logits.shape[0]), int(logits.shape[1])
    if launchable(logits.device) and rows > 0:
        params = params.contiguous()
        out = torch.empty(
            (rows, vocab), dtype=torch.float32, device=logits.device
        )
        _shape_kernel[(rows, triton.cdiv(vocab, _SHAPE_BLOCK))](
            logits,
            counts,
            slots,
            params,
            out,
            vocab,
            logits.stride(0),
            counts.stride(0),
            params.stride(0),
            out.stride(0),
            block=_SHAPE_BLOCK,
        )
        return out
    work = penalize(
        logits.to(dtype=torch.float32),
        params,
        counts.index_select(0, slots),
    )
    return work / temperature_divisors(params).unsqueeze(1)


def top_k_candidates(
    logits: torch.Tensor,
    width: int,
    top_p: torch.Tensor,
    top_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retain top-k candidates through the first nucleus threshold crossing.

    Each row keeps its ``width`` largest logits, sorted in descending order;
    with ``top_k`` (``[rows]`` int64, each at most ``width``) a row keeps
    only its first ``top_k`` of them. Nucleus filtering then retains the
    candidate that first crosses the row's ``top_p``. General and top-k
    sampling use the same candidate ordering, including ties at the top-k
    and nucleus boundaries. Returned IDs address the input vocabulary;
    dropped candidates have negative-infinite logits.
    """
    values, indexes = torch.topk(logits, width, dim=-1, sorted=True)
    if top_k is not None:
        positions = torch.arange(width, device=logits.device)
        values = values.masked_fill(
            positions.unsqueeze(0) >= top_k.unsqueeze(1), float("-inf")
        )
    cumulative = torch.softmax(values, dim=-1).cumsum(dim=-1)
    over = cumulative > top_p.unsqueeze(1)
    drop = torch.cat((torch.zeros_like(over[:, :1]), over[:, :-1]), dim=1)
    return torch.where(drop, float("-inf"), values), indexes


def sample_top_k(
    logits: torch.Tensor,
    draws: torch.Tensor,
    params: torch.Tensor,
    width: int,
    counts: torch.Tensor,
    slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample each row from its top-k candidates under its own parameters.

    A row with a positive temperature samples among at most ``width``
    candidates: its top-k width (``params[:, TOP_K]``) must be positive and
    at most ``width``. A zero-temperature row selects its largest shaped
    logit, the lowest vocabulary ID among ties, whatever its top-k width.

    Args:
        logits: ``[rows, vocab]`` global logits.
        draws: ``[rows]`` float32 uniform draws in ``[0, 1)``.
        params: ``[rows, PARAMETER_COLUMNS]`` float32 parameters.
        width: Candidates each row keeps, at most the vocabulary.
        counts: ``[slots, vocab]`` int32 committed token counts.
        slots: ``[rows]`` int64, the slot whose counts each row's penalties
            read.

    Returns:
        ``(tokens, valid)``: ``[rows]`` int64 tokens, and whether each row's
        shaped candidates form a usable distribution.
    """
    work = shape_rows(logits, params, counts, slots)
    temperature = params[:, TEMPERATURE]

    # A row without a top-k width keeps every candidate, which serves only
    # a zero-temperature row.
    top_k = params[:, TOP_K].to(torch.int64)
    top_k = torch.where(top_k > 0, top_k, width).clamp_max(width)
    candidates, token_indexes = top_k_candidates(
        work, width, params[:, TOP_P], top_k
    )

    # Min-p is measured relative to the maximum candidate probability in log
    # space.
    min_p = params[:, MIN_P]
    min_threshold = candidates[:, 0] + torch.log(min_p)
    candidates = torch.where(
        (min_p.unsqueeze(1) <= 0.0)
        | (candidates >= min_threshold.unsqueeze(1)),
        candidates,
        float("-inf"),
    )
    probabilities = torch.softmax(candidates, dim=-1)
    # Sorting candidates by token ID makes inverse-CDF draws deterministic for
    # a fixed random value independent of top-k kernel ordering, and matches
    # the general path, which selects in vocabulary order. Candidates dropped
    # by top-k, top-p or min-p carry zero mass, which the shared selector
    # never picks.
    token_order = torch.argsort(token_indexes, dim=-1)
    ordered_probabilities = probabilities.gather(1, token_order)
    sampled_order = sample_categorical(ordered_probabilities, draws)
    sampled = token_order.gather(1, sampled_order.unsqueeze(1))[:, 0]
    greedy = (
        torch.where(
            candidates == candidates.max(dim=-1, keepdim=True).values,
            token_indexes,
            logits.shape[1],
        )
        .min(dim=-1)
        .values
    )
    tokens = torch.where(
        temperature > 0.0,
        token_indexes.gather(1, sampled.unsqueeze(1))[:, 0],
        greedy,
    )

    # Rows with NaN or +inf candidates, or with no finite candidate at all,
    # have no well-defined distribution; the caller rejects their tokens.
    valid = (
        ~torch.isnan(candidates).any(dim=-1)
        & ~torch.isposinf(candidates).any(dim=-1)
        & torch.isfinite(candidates).any(dim=-1)
    )
    return tokens, valid


__all__ = [
    "PARAMETER_COLUMNS",
    "penalize",
    "sample_top_k",
    "shape_rows",
    "temperature_divisors",
    "top_k_candidates",
]
