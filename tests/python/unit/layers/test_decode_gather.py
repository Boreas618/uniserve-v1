"""Decode rows gathered from request slot state."""

from __future__ import annotations

import pytest
import torch

from uniserve_models.nn.attention.gather import gather_request_decode_inputs

pytestmark = pytest.mark.unit


def test_decode_rows_read_their_slots_and_slot_zero_is_padding() -> None:
    # Three slots (slot 0 is padding) over a full table (group 0) and a
    # table windowed to 4 tokens (group 1), both of 4-token pages. Slot 1
    # installed the windowed table from its page 1 on.
    unit_tables = torch.tensor(
        [
            [[0, 0, 0], [10, 11, 12], [20, 21, 22]],
            [[0, 0, 0], [31, 32, 33], [40, 41, 42]],
        ],
        dtype=torch.int32,
    )
    start_pages = torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.int32)
    table_shapes = torch.tensor([[0, 4, -1], [1, 4, 4]], dtype=torch.int32)
    cache_lengths = torch.tensor([0, 9, 2], dtype=torch.int32)
    tokens = torch.tensor([[1], [101], [102]], dtype=torch.int64)
    positions = torch.tensor([0, 9, 7], dtype=torch.int64)

    rows, columns = 3, 2
    outputs = {
        "input_ids": torch.full((rows,), -7, dtype=torch.int64),
        "positions": torch.full((1, rows), -7, dtype=torch.int64),
        "block_tables": torch.full((2, rows, columns), -7, dtype=torch.int32),
        "start_pages": torch.full((2, rows), -7, dtype=torch.int32),
        "cache_lengths": torch.full((rows,), -7, dtype=torch.int32),
        "query_lengths": torch.full((rows,), -7, dtype=torch.int32),
        "query_offsets": torch.full((rows + 1,), -7, dtype=torch.int32),
        "prefix_offsets": torch.full((rows + 1,), -7, dtype=torch.int32),
        "write_indices": torch.full((2, rows), -7, dtype=torch.int64),
    }
    gather_request_decode_inputs(
        request_pool_indices=torch.tensor([2, 0, 1], dtype=torch.int64),
        request_unit_tables=unit_tables,
        request_start_pages=start_pages,
        table_shapes=table_shapes,
        request_cache_lengths=cache_lengths,
        request_tokens=tokens[:, 0],
        request_positions=positions,
        columns=columns,
        **outputs,
    )

    # Each live row carries its slot's next token, position and cache
    # length; the padding row is one inert token.
    assert outputs["input_ids"].tolist() == [102, 1, 101]
    assert outputs["positions"].tolist() == [[7, 0, 9]]
    assert outputs["cache_lengths"].tolist() == [2, 0, 9]
    assert outputs["query_lengths"].tolist() == [1, 1, 1]
    assert outputs["query_offsets"].tolist() == [0, 1, 2, 3]
    assert outputs["prefix_offsets"].tolist() == [0, 2, 2, 11]

    # A full table selects from page zero; the windowed table from the
    # first page the row's window reaches, relative to the installed page.
    assert outputs["start_pages"].tolist() == [[0, 0, 0], [0, 0, 1]]
    assert outputs["block_tables"].tolist() == [
        [[20, 21], [0, 0], [10, 11]],
        [[40, 41], [0, 0], [31, 32]],
    ]

    # The token is appended at its cache length; padding writes nothing.
    assert outputs["write_indices"].tolist() == [
        [20 * 4 + 2, -1, 12 * 4 + 1],
        [40 * 4 + 2, -1, 32 * 4 + 1],
    ]
