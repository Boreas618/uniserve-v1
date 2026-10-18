"""Selected and top token log probabilities in one packed column.

``logprob_details`` scores logits rows and packs every requesting row's
fields into one int64 column that the worker reads back and decodes
(``uniserve_worker::LogprobLayout``). ``prompt_logprobs`` scores a prompt
chunk against the logits that predicted each of its tokens. Both score the
model's distribution, the logits before any sampling control: a reported
log probability is how likely the model found a token, whatever cutoffs,
penalties or temperature selected it.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def packed_words(counts: Sequence[int]) -> int:
    """Words ``logprob_details`` packs for rows of top ``counts``."""
    if not counts:
        return 0
    return len(counts) * (3 + 3 * max(counts))


def _float_bits(values: torch.Tensor) -> torch.Tensor:
    """Reinterpret float32 values as int32 bit patterns widened to int64.

    The widening sign-extends, so decoders keep only the low 32 bits.
    """
    return (
        values.to(dtype=torch.float32)
        .contiguous()
        .view(torch.int32)
        .to(torch.long)
    )


def logprob_details(
    work: torch.Tensor,
    output_rows: torch.Tensor,
    output_tokens: torch.Tensor,
    counts: Sequence[int],
) -> torch.Tensor:
    """Score the selected and top tokens of the requesting rows.

    Scores are ``log_softmax`` of the logits rows. Ranks are 1-based
    competition ranks: one plus the number of strictly larger scores in the
    row, so equal scores share a rank.

    Args:
        work: ``[rows, vocab]`` float logits.
        output_rows: Per requesting row, the ``work`` row that scored it.
        output_tokens: Per requesting row, the selected token.
        counts: Per requesting row, how many top tokens it reports, already
            clamped to the vocabulary.

    Returns:
        An int64 column concatenating, row-major over the requesting rows:
        selected token, selected score bits, selected rank, top token ids,
        top score bits and top ranks. Top fields are ``max(counts)`` wide.
    """
    count = len(counts)
    device = work.device
    scores = torch.log_softmax(work.index_select(0, output_rows), dim=-1)
    selected = output_tokens.reshape(-1).to(torch.long)
    selected_values = scores.gather(1, selected.unsqueeze(1))[:, 0]
    selected_ranks = (scores > selected_values.unsqueeze(1)).sum(
        dim=-1, dtype=torch.long
    ) + 1

    max_count = max(counts, default=0)
    if max_count:
        top_values, top_indexes = torch.topk(
            scores, max_count, dim=-1, sorted=True
        )
        positions = torch.arange(
            1, max_count + 1, dtype=torch.long, device=device
        ).unsqueeze(0)
        # Competition ranking: equal values share a rank, and the next
        # strictly lower value resumes at its own position.
        starts = torch.cat(
            (
                torch.ones((count, 1), dtype=torch.bool, device=device),
                top_values[:, 1:] < top_values[:, :-1],
            ),
            dim=1,
        )
        top_ranks = torch.where(starts, positions, 0).cummax(dim=1).values
    else:
        top_values = scores.new_empty((count, 0))
        top_indexes = torch.empty((count, 0), dtype=torch.long, device=device)
        top_ranks = torch.empty_like(top_indexes)

    return torch.cat(
        (
            selected,
            _float_bits(selected_values.reshape(-1)),
            selected_ranks.reshape(-1),
            top_indexes.reshape(-1).to(torch.long),
            _float_bits(top_values.reshape(-1)),
            top_ranks.reshape(-1).to(torch.long),
        )
    )


def prompt_logprobs(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    previous: torch.Tensor | None,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Score a prompt chunk and return the logits predicting the next chunk.

    ``logits[i]`` predicts the token after ``tokens[i]``. A first chunk has
    no prediction for its first token, which is excluded from scoring; a
    continued chunk supplies the preceding chunk's last logits row as
    ``previous``, which predicts its first token.

    Returns:
        ``(retained, packed)``: the chunk's last logits row and the packed
        column of every scored position (``logprob_details`` with ``count``
        top tokens per position), or ``None`` when no position is scored.
    """
    tokens = tokens.reshape(-1).to(device=logits.device, dtype=torch.long)
    if logits.ndim != 2 or int(logits.shape[0]) != int(tokens.numel()):
        raise ValueError("prompt scoring logits do not align with input tokens")

    if previous is None:
        score_logits, targets = logits[:-1], tokens[1:]
    else:
        previous = previous.reshape(1, -1).to(
            device=logits.device, dtype=logits.dtype
        )
        score_logits = torch.cat((previous, logits[:-1]), dim=0)
        targets = tokens

    retained = logits[-1]
    if targets.numel() == 0:
        return retained, None
    rows = torch.arange(
        int(targets.numel()), dtype=torch.long, device=score_logits.device
    )
    packed = logprob_details(
        score_logits.float(),
        rows,
        targets,
        (int(count),) * int(targets.numel()),
    )
    return retained, packed


__all__ = [
    "logprob_details",
    "packed_words",
    "prompt_logprobs",
]
