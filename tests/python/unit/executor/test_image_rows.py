"""Packed image rows expand into the per-token metadata their images need."""

import numpy as np
import pytest
import torch

from uniserve.executor.image.layout import ImageLayout
from uniserve.executor.image.rows import (
    ImageRow,
    RowColumns,
    RowField,
    expand_rows,
    pack_attention,
)
from uniserve.models.inputs import TokenKind
from uniserve.processing import BranchSource

pytestmark = pytest.mark.unit

DUMP = 99


def _layout(framing: int) -> ImageLayout:
    # 16-pixel latent patches, at most 12 per image.
    return ImageLayout(
        patch_size=2,
        latent_channels=2,
        downsample=16,
        framing_tokens=framing,
        image_unconditional=BranchSource.START,
        max_sequence_tokens=12,
        rope_advance=2,
        guidance_branches=3,
        prediction_dtype=torch.bfloat16,
    )


def _rows(layout: ImageLayout) -> list[ImageRow]:
    # A 32x48 image (2x3 grid) evaluated on two guidance branches, then a
    # 16x32 image (1x2 grid) written as conditioning at position 10.
    p = 10
    return [
        ImageRow(0, 0, 32, 48, (7, 7, 7), 0.25, 1.5, prediction=0),
        ImageRow(0, 0, 32, 48, (3, 3, 3), 0.25, 1.5, prediction=6),
        ImageRow(
            1,
            6,
            16,
            32,
            (p, p + 1, p + layout.rope_advance),
            0.0,
            prediction=12,
            slot=2,
            prefix=10,
            write=True,
        ),
    ]


def _expand(columns: RowColumns):
    return expand_rows(*columns.tensors(), tokens=columns.tokens, dump=DUMP)


@pytest.mark.parametrize("framing", [0, 2])
def test_rows_expand_to_their_images_positions(framing):
    layout = _layout(framing)
    rows = _rows(layout)
    columns = RowColumns.of(layout, rows)
    expanded = _expand(columns)

    expected = [
        layout.positions(32, 48, 7),
        layout.positions(32, 48, 3),
        layout.positions(16, 32, 11),
    ]
    if framing:
        expected[2] = layout.conditioning_positions(16, 32, 10)
    np.testing.assert_array_equal(
        expanded.rows.positions.numpy(), np.concatenate(expected, axis=1)
    )
    assert expanded.rows.tokens == sum(part.shape[1] for part in expected)


def test_latent_tokens_read_their_samples_and_predict_their_rows():
    layout = _layout(2)
    expanded = _expand(RowColumns.of(layout, _rows(layout)))
    rows = expanded.rows

    marker = [TokenKind.OPEN, *[TokenKind.LATENT] * 6, TokenKind.CLOSE]
    assert rows.kind.tolist() == [
        *marker,
        *marker,
        TokenKind.OPEN,
        TokenKind.LATENT,
        TokenKind.LATENT,
        TokenKind.CLOSE,
    ]
    assert rows.sample_index.tolist() == [
        -1, 0, 1, 2, 3, 4, 5, -1,
        -1, 0, 1, 2, 3, 4, 5, -1,
        -1, 6, 7, -1,
    ]  # fmt: skip
    assert expanded.prediction_index.tolist() == [
        DUMP, 0, 1, 2, 3, 4, 5, DUMP,
        DUMP, 6, 7, 8, 9, 10, 11, DUMP,
        DUMP, 12, 13, DUMP,
    ]  # fmt: skip
    assert rows.token_row.tolist() == [0] * 8 + [1] * 8 + [2] * 4
    assert rows.timesteps.tolist() == [0.25] * 16 + [0.0] * 4
    assert rows.row_grid.tolist() == [[2, 3], [2, 3], [1, 2]]
    assert rows.noise_scale.tolist() == [1.5, 1.5, 0.0]


def test_a_bucket_pads_with_empty_rows_and_one_padding_row():
    layout = _layout(2)
    live = RowColumns.of(layout, _rows(layout))
    columns = live.padded(26, 6, layout.max_row_tokens)
    expanded = _expand(columns)
    rows = expanded.rows

    # Live rows lead, two empty rows follow and the last row holds the
    # bucket's six remaining tokens.
    assert columns.column(RowField.TOKENS).tolist() == [8, 8, 4, 0, 0, 6]
    assert columns.column(RowField.TOKEN_START).tolist() == [
        0, 8, 16, 20, 20, 20,
    ]  # fmt: skip
    # Padding and empty rows attend to slot 0 without prefix and write
    # nothing.
    for field in (RowField.SLOT, RowField.PREFIX, RowField.WRITE):
        assert columns.column(field)[3:].tolist() == [0, 0, 0]

    padding = slice(20, 26)
    assert rows.kind[padding].tolist() == [TokenKind.PAD] * 6
    assert rows.sample_index[padding].tolist() == [-1] * 6
    assert expanded.prediction_index[padding].tolist() == [DUMP] * 6
    assert rows.timesteps[padding].tolist() == [0.0] * 6
    assert rows.positions[:, padding].tolist() == [[0] * 6] * 3
    assert rows.token_row[padding].tolist() == [5] * 6

    # The live tokens expand as without padding.
    unpadded = _expand(live)
    torch.testing.assert_close(
        rows.positions[:, :20], unpadded.rows.positions, rtol=0, atol=0
    )
    torch.testing.assert_close(
        expanded.prediction_index[:20],
        unpadded.prediction_index,
        rtol=0,
        atol=0,
    )


def test_a_bucket_serves_fewer_rows_and_tokens_than_it_holds():
    layout = _layout(2)
    live = RowColumns.of(layout, _rows(layout))
    with pytest.raises(ValueError, match="cannot pad"):
        live.padded(20, 6, layout.max_row_tokens)
    with pytest.raises(ValueError, match="cannot pad"):
        live.padded(26, 3, layout.max_row_tokens)
    # A padding row holds at most one row's longest sequence.
    with pytest.raises(ValueError, match="exceed"):
        live.padded(20 + layout.max_row_tokens + 1, 6, layout.max_row_tokens)


def test_an_all_padding_bucket_reads_and_predicts_nothing():
    layout = _layout(2)
    columns = RowColumns.of(layout, []).padded(8, 2, layout.max_row_tokens)
    expanded = _expand(columns)
    assert expanded.rows.kind.tolist() == [TokenKind.PAD] * 8
    assert expanded.prediction_index.tolist() == [DUMP] * 8
    assert columns.regions() == ()


def test_images_that_exceed_the_longest_sequence_are_refused():
    with pytest.raises(ValueError, match="exceeds"):
        RowColumns.of(_layout(2), [ImageRow(0, 0, 64, 64, (0, 0, 0), 0.5)])


def test_regions_name_each_images_latent_tokens():
    layout = _layout(2)
    columns = RowColumns.of(layout, _rows(layout)).padded(
        26, 6, layout.max_row_tokens
    )
    assert columns.regions() == ((1, 2, 3), (9, 2, 3), (17, 1, 2))


def test_attention_rows_hold_each_rows_queries_and_writes():
    layout = _layout(2)
    columns = RowColumns.of(layout, _rows(layout)).padded(
        26, 6, layout.max_row_tokens
    )
    stride, capacity = 7, 32
    words = pack_attention(
        columns, [np.array([0, 0, 3])], stride=stride, tokens=capacity
    )
    sections = words[: 7 * stride].reshape(7, stride)
    assert sections[0, :6].tolist() == [0, 0, 2, 0, 0, 0]
    assert sections[1, :6].tolist() == [0, 0, 10, 0, 0, 0]
    assert sections[2, :6].tolist() == [8, 8, 4, 0, 0, 6]
    assert sections[3, :6].tolist() == [0, 0, 1, 0, 0, 0]
    assert sections[6, :6].tolist() == [0, 0, 3, 0, 0, 0]
    token_rows = words[7 * stride :]
    assert token_rows[:26].tolist() == [0] * 8 + [1] * 8 + [2] * 4 + [5] * 6
