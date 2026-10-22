"""The adapter of every token of a forward (``uniserve.adapters``).

A decode's tokens are its rows, each reading its slot's adapter; a
prefill's packed tokens read the adapter of their row's slot, through the
row columns the worker packs (`crates/worker/src/runner/attention.rs`,
`pack_rows`): each row's slot in the first section, and each token's row in
the last.
"""

import torch

from uniserve import adapters
from uniserve.nn.attention.gather import ROW_SECTIONS, row_columns_stride


def _row_columns(slots, queries, max_rows, tables):
    """The packed columns of rows of ``slots`` taking ``queries`` tokens."""
    stride = row_columns_stride(max_rows)
    token_section = (ROW_SECTIONS + tables) * stride
    columns = [0] * (token_section + sum(queries))
    offset = 0
    for row, (slot, query) in enumerate(zip(slots, queries, strict=True)):
        columns[row] = slot
        for token in range(query):
            columns[token_section + offset + token] = row
        offset += query
    return torch.tensor(columns, dtype=torch.int64)


def test_a_prefill_token_reads_its_rows_slots_adapter():
    # Slot 3 computes with adapter 2 and slot 5 with adapter 1; slot 4 and
    # the padding slot 0 with none.
    slot_adapters = torch.tensor([0, 0, 0, 2, 0, 1, 0, 0], dtype=torch.int64)
    columns = _row_columns([3, 4, 5], [2, 1, 3], max_rows=4, tables=2)
    token_slots = torch.full((6,), -1, dtype=torch.int64)
    out = torch.full((6,), -1, dtype=torch.int64)
    gathered = adapters.gather_tokens(
        slot_adapters, columns, 4, 2, token_slots, out
    )
    assert gathered.data_ptr() == out.data_ptr()
    assert token_slots.tolist() == [3, 3, 4, 5, 5, 5]
    assert out.tolist() == [2, 2, 0, 1, 1, 1]


def test_a_decode_row_reads_its_slots_adapter():
    slot_adapters = torch.tensor([0, 0, 0, 2, 0, 1], dtype=torch.int64)
    out = torch.full((4,), -1, dtype=torch.int64)
    slots = torch.tensor([5, 3, 4, 0], dtype=torch.int64)
    assert adapters.gather_rows(slot_adapters, slots, out).tolist() == [
        1,
        2,
        0,
        0,
    ]


def test_the_running_forwards_adapters_are_selected_while_it_runs():
    assert adapters.tokens() is None
    selected = torch.zeros(3, dtype=torch.int64)
    with adapters.selected(selected):
        assert adapters.tokens() is selected
    assert adapters.tokens() is None
