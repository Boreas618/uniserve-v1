"""The bounds a diffusion LM computation derives from the worker's limits,
the prefix cache and the model (``uniserve.executor.diffusion_lm.bounds``).
"""  # noqa: D205

from __future__ import annotations

import pytest
import torch

from uniserve.executor.diffusion_lm.bounds import (
    Bounds,
    Denoising,
    Facts,
    ServedSampling,
)
from uniserve.executor.diffusion_lm.shapes import chunk_rows
from uniserve.executor.text.buffers import PageGroup

pytestmark = pytest.mark.unit

LENGTH, VOCAB = 256, 1024


def _facts(*, generates: bool = True) -> Facts:
    return Facts(
        sequence_length=LENGTH,
        vocab_size=VOCAB,
        hidden_size=64,
        embedding_dtype=torch.bfloat16,
        attend_widths=(64,),
        compute_dtype=torch.bfloat16,
        generates=generates,
        local_tail=True,
    )


def _sampling(length: int = LENGTH) -> ServedSampling:
    return ServedSampling(
        canvas_length=length,
        max_steps=8,
        entropy_bound=0.5,
        t_min=0.4,
        t_max=0.8,
        confidence_threshold=0.2,
        stability_threshold=1,
    )


def _group(
    block_size: int, units_per_page: int, window: int | None = None
) -> PageGroup:
    return PageGroup(block_size, units_per_page, window)


def _bounds(**overrides) -> Bounds:
    """Derive the bounds of a generating CUDA computation with graphs, eight
    calls and 2048 tokens a batch and 16 slots, unless ``overrides`` says
    otherwise.
    """  # noqa: D205
    arguments = {
        "max_batch_calls": 8,
        "max_batch_tokens": 2048,
        "max_model_len": 4096,
        "slots": 16,
        "groups": (_group(16, 1),),
        "units": None,
        "facts": _facts(),
        "cuda": True,
        "graph_policy": "full",
        "sampling": _sampling(),
    }
    arguments.update(overrides)
    return Bounds.derive(**arguments)


def test_capacity_is_the_least_of_the_rows_the_pool_and_the_tokens():
    # The batch's calls bound the rows; 2048 tokens hold 8 blocks.
    assert _bounds().capacity == 8
    # Fewer request slots bound the rows.
    assert _bounds(slots=3).capacity == 3
    # A pool of 7 units holds 6 blocks of 2 units a page (unit 0 is the
    # sentinel), so 3 blocks.
    assert _bounds(groups=(_group(16, 2),), units=7).capacity == 3
    # Before the pool is fitted, it bounds nothing.
    assert _bounds(groups=(_group(16, 2),), units=None).capacity == 8
    # The token bound holds 4 blocks.
    assert _bounds(max_batch_tokens=1100).capacity == 4

    with pytest.raises(ValueError):
        _bounds(max_batch_tokens=LENGTH - 1)
    with pytest.raises(ValueError):
        _bounds(units=1)


def test_graphs_double_the_packed_sequences_of_a_readout():
    bounds = _bounds(slots=6)
    assert bounds.graphs
    assert (bounds.readout_rows, bounds.rows) == (6, 12)
    # Without graphs a forward packs only its live sequences.
    for eager in (
        _bounds(slots=6, graph_policy="off"),
        _bounds(slots=6, cuda=False),
    ):
        assert not eager.graphs
        assert (eager.readout_rows, eager.rows) == (6, 6)
    # A readout never holds more blocks than tokens.
    bounds = _bounds(max_batch_calls=300, slots=300, max_batch_tokens=280)
    assert (bounds.capacity, bounds.readout_rows, bounds.rows) == (1, 280, 560)


def test_table_widths_follow_the_window_and_the_pool():
    full, windowed = _group(16, 1), _group(16, 2, window=1023)
    bounds = _bounds(groups=(full, windowed), max_batch_tokens=512)
    # One table per unit of a page.
    assert bounds.block_size == (16, 16, 16)
    assert bounds.windowed == (False, True, True)
    # Full attention selects the 4096-token sequence's 256 pages; the
    # window selects its 1023 history and 512 query tokens plus the partial
    # page at their start.
    assert bounds.widths == (256, 97, 97)

    # A table selects no more pages than the pool holds.
    bounds = _bounds(groups=(full, windowed), max_batch_tokens=512, units=61)
    assert bounds.widths == (60, 30, 30)


def test_a_generating_rank_steps_its_slots_with_the_served_sampling():
    bounds = _bounds()
    assert bounds.denoise == Denoising(8, chunk_rows(LENGTH, VOCAB, 8))
    # Without graphs the denoising columns hold every row a batch carries.
    eager = _bounds(max_batch_tokens=1100, graph_policy="off")
    assert eager.denoise == Denoising(4, chunk_rows(LENGTH, VOCAB, 4))
    assert _bounds(slots=3).denoise.rows == 3

    # A rank that cannot denoise, or a deployment without block sampling,
    # serves readouts only.
    assert _bounds(facts=_facts(generates=False)).denoise is None
    assert _bounds(sampling=None).denoise is None

    # The served canvas is the model's block.
    with pytest.raises(ValueError):
        _bounds(sampling=_sampling(LENGTH // 2))
    readouts = _bounds(facts=_facts(generates=False), sampling=_sampling(16))
    assert readouts.denoise is None


def test_shapes_follow_the_bounds():
    bounds = _bounds(slots=6)
    shapes = bounds.shapes()
    assert shapes.capacity() == bounds.capacity == 6
    assert shapes.rows() == (1, 2, 3, 4, 6)
    assert max(rows for rows, _ in shapes.attend_buckets()) == 6
    assert shapes.finish_buckets()[-1] == bounds.tokens


def test_served_sampling_decodes_the_worker_settings():
    value = {
        "canvas_length": 256,
        "max_steps": 8,
        "entropy_bound": 0.5,
        "t_min": 0.4,
        "t_max": 0.8,
        "confidence_threshold": 0.2,
        "stability_threshold": 1,
    }
    assert ServedSampling.from_json(value) == _sampling()
