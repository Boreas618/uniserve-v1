"""The columns and resident banks of a diffusion LM computation
(``uniserve.executor.diffusion_lm.buffers``).
"""  # noqa: D205

from __future__ import annotations

import pytest
import torch

from uniserve.executor.diffusion_lm.buffers import (
    Banks,
    BankShape,
    Columns,
    Dimensions,
)
from uniserve.executor.diffusion_lm.shapes import CANDIDATES

pytestmark = pytest.mark.unit

CPU = torch.device("cpu")


def _banks() -> Banks:
    shape = BankShape(slots=3, length=4, depth=1, hidden=2, dtype=torch.float32)
    return Banks(shape, CPU)


def test_gathers_copy_slot_rows_and_padding_rows_read_the_zero_slot():
    banks = _banks()
    # Slots 1..3 hold distinct states; the padding slot zero stays zero.
    banks.canvas[1:] = torch.arange(12).view(3, 4)
    banks.history[1:] = torch.arange(100, 112).view(3, 1, 4)
    banks.self_conditioning[1:] = torch.arange(24.0).view(3, 4, 2)
    # Stale contents of the step rows must not survive a gather.
    canvas = torch.full((3, 4), 7, dtype=torch.int64)
    history = torch.full((3, 1, 4), 7, dtype=torch.int64)
    soft = torch.full((3, 4, 2), 7.0)

    banks.gather(torch.tensor([2, 1, 0]), 3, canvas, history, soft)

    assert canvas.tolist() == [[4, 5, 6, 7], [0, 1, 2, 3], [0, 0, 0, 0]]
    assert history[:, 0].tolist() == [
        [104, 105, 106, 107],
        [100, 101, 102, 103],
        [0, 0, 0, 0],
    ]
    assert torch.equal(soft[0], banks.self_conditioning[2])
    assert torch.equal(soft[1], banks.self_conditioning[1])
    assert torch.equal(soft[2], torch.zeros(4, 2))


def test_gathers_leave_rows_past_the_step_untouched():
    banks = _banks()
    banks.canvas[3] = 9
    canvas = torch.full((2, 4), 7, dtype=torch.int64)
    history = torch.zeros((2, 1, 4), dtype=torch.int64)
    soft = torch.zeros((2, 4, 2))

    banks.gather(torch.tensor([3, 1]), 1, canvas, history, soft)
    assert canvas.tolist() == [[9, 9, 9, 9], [7, 7, 7, 7]]


def test_banks_fit_the_bytes_capacity_planning_reserves():
    # Planning reserves the fixed bytes plus the slot bytes per slot.
    for slots in (1, 3, 17):
        shape = BankShape(slots, 5, 2, 3, torch.bfloat16)
        banks = Banks(shape, CPU)
        allocated = banks.canvas.untyped_storage().nbytes()
        assert allocated <= shape.fixed_bytes() + slots * shape.slot_bytes()


def _dimensions(**overrides) -> Dimensions:
    arguments = {
        "rows": 8,
        "tokens": 64,
        "tables": 2,
        "width": 4,
        "hidden": 16,
        "attend_widths": (16, 8),
        "compute_dtype": torch.bfloat16,
        "length": 16,
        "denoise_rows": 3,
        "depth": 2,
        "embedding_dtype": torch.float32,
    }
    arguments.update(overrides)
    return Dimensions(**arguments)


def test_columns_hold_the_forwards_tensors_in_disjoint_storage():
    dimensions = _dimensions()
    columns = Columns(dimensions, CPU)
    shapes = {
        "input_ids": ((64,), torch.int64),
        "positions": ((64,), torch.int64),
        "slot_rows": ((64,), torch.int64),
        "finished_rows": ((256,), torch.int64),
        "candidate_index": ((CANDIDATES,), torch.int64),
        "candidate_logprobs": ((CANDIDATES,), torch.float32),
        "candidate_words": ((CANDIDATES,), torch.int64),
        "attend_state.0": ((64, 16), torch.bfloat16),
        "attend_state.1": ((64, 8), torch.bfloat16),
        "finished": ((64, 16), torch.bfloat16),
        "denoise.columns": ((4, 3), torch.int64),
        "denoise.relays": ((3,), torch.int64),
        "denoise.tokens": ((3, 16), torch.int64),
        "denoise.history": ((3, 2, 16), torch.int64),
        "denoise.self_conditioning": ((3, 16, 16), torch.float32),
        "denoise.results": ((3, 17), torch.int64),
        "denoise.continued": ((3,), torch.uint8),
    }
    for name, (shape, dtype) in shapes.items():
        assert columns[name].shape == shape, name
        assert columns[name].dtype == dtype, name
    assert [state.shape for state in columns.attend_state] == [
        (64, 16),
        (64, 8),
    ]
    # The packed attention rows: six sections and one per table, of the
    # rows and their end offset each, then every token's row.
    assert columns["row_columns"].shape == ((6 + 2) * 9 + 64,)

    # Writing every column leaves every other column as it was.
    names = [name for name, _, _ in dimensions.columns()]
    for value, name in enumerate(names, start=1):
        columns[name].view(-1).view(torch.uint8).fill_(value)
    for value, name in enumerate(names, start=1):
        written = columns[name].reshape(-1).view(torch.uint8)
        assert bool(torch.all(written == value)), name
    assert columns.arena.numel() <= dimensions.bytes()


def test_readout_computations_off_the_final_stage_hold_no_state_or_steps():
    dimensions = _dimensions(attend_widths=(), denoise_rows=0)
    assert Columns(dimensions, CPU).attend_state == []
    names = {name for name, _, _ in dimensions.columns()}
    assert "finished" not in names
    assert "denoise.columns" not in names
    assert "candidate_index" in names
