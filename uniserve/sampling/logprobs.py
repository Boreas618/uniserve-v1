"""Selected and top token log probabilities in one packed column.

``logprob_details`` scores logits rows and packs every row's fields into
one int64 column that the worker reads back and decodes
(``worker::LogprobLayout``). ``scatter_logprobs`` scores rows that belong
to blocks, each packed compactly in its own place: a forward's reporting
tasks, or its prompt chunks, scored against the logits that predicted
each of their tokens. Both score the model's distribution, the logits
before any sampling control: a reported log probability is how likely the
model found a token, whatever cutoffs, penalties or temperature selected
it.
"""

from __future__ import annotations

import torch


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
    top: int,
) -> torch.Tensor:
    """Score the selected and top tokens of rows.

    Scores are ``log_softmax`` of the logits rows. Ranks are 1-based
    competition ranks: one plus the number of strictly larger scores in the
    row, so equal scores share a rank.

    Args:
        work: ``[rows, vocab]`` float logits.
        output_rows: Per scored row, the ``work`` row that scored it.
        output_tokens: Per scored row, the selected token.
        top: Top tokens each row reports (a row may request fewer, which the
            reader keeps), at most the vocabulary.

    Returns:
        An int64 column concatenating, field by field over the rows:
        selected token, selected score bits, selected rank, top token ids,
        top score bits and top ranks. Top fields are ``top`` wide.
    """
    count = int(output_rows.numel())
    device = work.device
    scores = torch.log_softmax(work.index_select(0, output_rows), dim=-1)
    selected = output_tokens.reshape(-1).to(torch.long)
    selected_values = scores.gather(1, selected.unsqueeze(1))[:, 0]
    selected_ranks = (scores > selected_values.unsqueeze(1)).sum(
        dim=-1, dtype=torch.long
    ) + 1

    max_count = int(top)
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


def compact_offsets(
    bases: torch.Tensor,
    index: torch.Tensor,
    rows: torch.Tensor,
    prefix: torch.Tensor,
    total: torch.Tensor,
    counts: torch.Tensor,
    top: int,
    dump: int,
) -> torch.Tensor:
    """Column offsets of the words ``logprob_details`` packs for rows.

    Row ``r`` is row ``index[r]`` of a block of ``rows[r]`` rows packed from
    ``bases[r]``, field by field over the block's rows: the selected tokens,
    their scores and their ranks, then each row's own top tokens, scores and
    ranks, ``counts[r]`` of them, after the ``prefix[r]`` of the block's
    rows before it, of ``total[r]`` in the block (``worker::LogprobLayout``).
    The packed words have ``top`` top tokens per row; those past a row's
    count, and every word of a row with a negative count, go to ``dump``.
    The offsets follow the packed order of the rows' words: selected
    tokens, scores and ranks, then top ids, scores and ranks.
    """
    reports = counts >= 0
    first = bases + index
    columns = torch.arange(int(top), dtype=torch.long, device=bases.device)
    kept = reports.unsqueeze(1) & (columns < counts.unsqueeze(1))
    start = (bases + 3 * rows + prefix).unsqueeze(1) + columns
    span = total.unsqueeze(1)

    def selected(offsets: torch.Tensor) -> torch.Tensor:
        return torch.where(reports, offsets, dump)

    def tops(offsets: torch.Tensor) -> torch.Tensor:
        return torch.where(kept, offsets, dump).reshape(-1)

    return torch.cat(
        (
            selected(first),
            selected(first + rows),
            selected(first + 2 * rows),
            tops(start),
            tops(start + span),
            tops(start + 2 * span),
        )
    )


def scatter_logprobs(
    work: torch.Tensor,
    output_rows: torch.Tensor,
    targets: torch.Tensor,
    layout: tuple[torch.Tensor, ...],
    top: int,
    logprobs: torch.Tensor,
    dump: int,
) -> None:
    """Score rows' targets and top tokens into their blocks' compact words.

    Row ``r`` scores ``targets[r]`` under row ``output_rows[r]`` of
    ``work`` (``logprob_details`` with ``top`` top tokens) and writes its
    words where ``layout``, the ``bases, index, rows, prefix, total,
    counts`` of :func:`compact_offsets`, places them; the words it does not
    report go to ``dump``, which no one reads.
    """
    packed = logprob_details(work, output_rows, targets, int(top))
    bases, index, rows, prefix, total, counts = layout
    offsets = compact_offsets(
        bases, index, rows, prefix, total, counts, int(top), int(dump)
    )
    logprobs.index_copy_(0, offsets, packed)


__all__ = [
    "compact_offsets",
    "logprob_details",
    "scatter_logprobs",
]
