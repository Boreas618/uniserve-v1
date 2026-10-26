"""Image denoisers assemble their own typed input from borrowed views.

Execution calls `bind_packed` on whichever denoiser a model exposes and passes
the same borrowed numerical views to each. These run on CPU so an implementation
that reads a field the contract does not carry is caught without a GPU.
"""

from __future__ import annotations

import json

import pytest
import torch

from tests.python.fixtures.model_metadata import neo_metadata
from uniserve.executor.image.layout import ImageLayout
from uniserve.executor.image.rows import ImageRow, RowColumns, expand_rows
from uniserve.loading import Config as IOConfig
from uniserve.nn.attention import (
    AttentionBatch,
    SequenceLengths,
    VarlenInput,
)

pytestmark = pytest.mark.unit


def _sensenova_denoiser(tmp_path):
    from uniserve.models.sensenova_u1 import read_config
    from uniserve.models.sensenova_u1.denoiser import Denoiser
    from uniserve.models.sensenova_u1.transformer import Transformer

    (tmp_path / "config.json").write_text(json.dumps(neo_metadata()))
    config = read_config(tmp_path, IOConfig(), sources={})
    with torch.device("meta"):
        return Denoiser(config, Transformer(config.text))


def _stub_denoiser(tmp_path):
    from uniserve.models.stub import Config, Model

    del tmp_path
    return Model(Config()).denoiser


def _bind(denoiser):
    """Bind one two-by-two patch image's framed row and its attention."""
    layout = ImageLayout.of(denoiser)
    side = denoiser.downsample * 2
    columns = RowColumns.of(
        layout, [ImageRow(0, 0, side, side, (3, 3, 3), 0.5)]
    )
    expanded = expand_rows(*columns.tensors(), tokens=columns.tokens, dump=4)
    source = torch.zeros((4, layout.width), dtype=torch.float32)
    lengths = SequenceLengths.from_lengths((columns.tokens,), device="cpu")
    bound = denoiser.bind_packed(
        source=source,
        rows=expanded.rows,
        attention=AttentionBatch.single(
            VarlenInput(lengths, lengths, (False,))
        ),
    )
    return bound, source, expanded.rows


@pytest.mark.parametrize(
    "build", [_sensenova_denoiser, _stub_denoiser], ids=["sensenova_u1", "stub"]
)
def test_image_denoisers_bind_borrowed_rows(build, tmp_path):
    bound, source, rows = _bind(build(tmp_path))

    # Solver updates write the step workspace in place, and a captured step
    # reads its rows as device data, so the bound input must borrow them
    # rather than hold copies.
    assert bound.source is source
    assert bound.rows is rows


def test_bound_input_drives_one_packed_prediction(tmp_path):
    denoiser = _stub_denoiser(tmp_path)
    bound, source, rows = _bind(denoiser)

    (output,) = denoiser(bound, state={}, constants={}, workspace={})["image"]

    assert output.tensor.shape == (rows.tokens, source.shape[1])
    assert output.tensor.dtype == denoiser.prediction_dtype
    assert output.layout.shape == (rows.tokens, source.shape[1])


def test_packed_inputs_cover_their_attention_queries(tmp_path):
    denoiser = _stub_denoiser(tmp_path)
    bound, source, rows = _bind(denoiser)
    lengths = SequenceLengths.from_lengths((rows.tokens + 1,), device="cpu")

    with pytest.raises(ValueError, match="cover"):
        denoiser.bind_packed(
            source=source,
            rows=rows,
            attention=AttentionBatch.single(
                VarlenInput(lengths, lengths, (False,))
            ),
        )
