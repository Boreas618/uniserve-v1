"""Selected, top and requested token log probabilities in one packed column.

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
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class LogprobRequest:
    """What one sampled row reports: top ``count`` and explicit token ids.

    ``count`` is already clamped to the vocabulary and ``token_ids`` are
    deduplicated in-range ids, in request order.
    """

    count: int
    token_ids: tuple[int, ...]


def packed_words(requests: Sequence[LogprobRequest]) -> int:
    """Words ``logprob_details`` packs for ``requests`` (one per row)."""
    if not requests:
        return 0
    top = max(request.count for request in requests)
    explicit = max(len(request.token_ids) for request in requests)
    return len(requests) * (3 + 3 * top + 2 * explicit)


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
    requests: Sequence[LogprobRequest],
) -> torch.Tensor:
    """Score selected, top and requested tokens of the requesting rows.

    Scores are ``log_softmax`` of the logits rows. Ranks are 1-based
    competition ranks: one plus the number of strictly larger scores in the
    row, so equal scores share a rank.

    Args:
        work: ``[rows, vocab]`` float logits.
        output_rows: Per requesting row, the ``work`` row that scored it.
        output_tokens: Per requesting row, the selected token.
        requests: Per requesting row, what it reports.

    Returns:
        An int64 column concatenating, row-major over the requesting rows:
        selected token, selected score bits, selected rank, top token ids,
        top score bits, top ranks, requested score bits and requested ranks.
        Top fields are ``max(count)`` wide and requested fields
        ``max(len(token_ids))`` wide; padding columns hold zeros.
    """
    count = len(requests)
    device = work.device
    scores = torch.log_softmax(work.index_select(0, output_rows), dim=-1)
    selected = output_tokens.reshape(-1).to(torch.long)
    selected_values = scores.gather(1, selected.unsqueeze(1))[:, 0]
    selected_ranks = (scores > selected_values.unsqueeze(1)).sum(
        dim=-1, dtype=torch.long
    ) + 1

    max_count = max((request.count for request in requests), default=0)
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

    max_requested = max(
        (len(request.token_ids) for request in requests), default=0
    )
    if max_requested:
        # Rows with fewer requested ids keep zero-padding columns; the host
        # layout records how many leading columns are real.
        padded = [
            list(request.token_ids)
            + [0] * (max_requested - len(request.token_ids))
            for request in requests
        ]
        candidates = torch.tensor(padded, dtype=torch.long, device=device)
        candidate_values = scores.gather(1, candidates)
        candidate_ranks = torch.stack(
            tuple(
                (scores > candidate_values[:, index].unsqueeze(1)).sum(
                    dim=-1, dtype=torch.long
                )
                + 1
                for index in range(max_requested)
            ),
            dim=1,
        )
    else:
        candidate_values = scores.new_empty((count, 0))
        candidate_ranks = torch.empty(
            (count, 0), dtype=torch.long, device=device
        )

    return torch.cat(
        (
            selected,
            _float_bits(selected_values.reshape(-1)),
            selected_ranks.reshape(-1),
            top_indexes.reshape(-1).to(torch.long),
            _float_bits(top_values.reshape(-1)),
            top_ranks.reshape(-1).to(torch.long),
            _float_bits(candidate_values.reshape(-1)),
            candidate_ranks.reshape(-1).to(torch.long),
        )
    )


def prompt_logprobs(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    previous: torch.Tensor | None,
    request: LogprobRequest,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Score a prompt chunk and return the logits predicting the next chunk.

    ``logits[i]`` predicts the token after ``tokens[i]``. A first chunk has
    no prediction for its first token, which is excluded from scoring; a
    continued chunk supplies the preceding chunk's last logits row as
    ``previous``, which predicts its first token.

    Returns:
        ``(retained, packed)``: the chunk's last logits row and the packed
        column of every scored position (``logprob_details`` with ``request``
        per position), or ``None`` when no position is scored.
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
        (request,) * int(targets.numel()),
    )
    return retained, packed


__all__ = [
    "LogprobRequest",
    "logprob_details",
    "packed_words",
    "prompt_logprobs",
]
