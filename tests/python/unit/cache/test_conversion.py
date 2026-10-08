"""Imported K/V spans convert into a unit pool exactly like local writes."""

import math

import pytest
import torch

from uniserve_models.cache import Config, mha
from uniserve_models.cache.conversion import copy_converted
from uniserve_models.cache.units import UnitPool
from uniserve_models.quantization import QuantizedTensor, Quantizer

pytestmark = pytest.mark.unit

LAYERS, TOTAL_HEADS, DIM = 2, 4, 4


def _pool(heads, page_tokens, compute, quantized):
    """One destination group of ``LAYERS`` layers storing ``heads``."""
    config = Config(
        {
            f"layers.{index}": mha.Config(
                TOTAL_HEADS, DIM, tuple(heads), compute
            )
            for index in range(LAYERS)
        }
    )
    return UnitPool.allocate(
        config,
        num_units=8,
        block_size=page_tokens,
        device="cpu",
        quantization={name: Quantizer("fp8", axis=0) for name in config.layers}
        if quantized
        else None,
    )


def _export(generator, start, end, page_tokens, scale_heads, compute):
    """A logical FP8 export of tokens ``[start, end)``.

    Returns encoded ``[2, tokens, layers, heads, dim]`` values, their
    ``[pages, 2, layers, head groups]`` scales and the values they decode to,
    rounded through ``compute`` as every reader decodes them.
    """
    tokens = end - start
    pages = math.ceil((start % page_tokens + tokens) / page_tokens)
    encoded = (
        torch.randn(2, tokens, LAYERS, TOTAL_HEADS, DIM, generator=generator)
        * 200
    ).clamp(-448, 448)
    encoded = encoded.to(torch.float8_e4m3fn)
    scales = (
        torch.rand(
            pages, 2, LAYERS, TOTAL_HEADS // scale_heads, generator=generator
        )
        * 3
        + 0.5
    )
    page = (torch.arange(tokens) + start % page_tokens) // page_tokens
    group = torch.arange(TOTAL_HEADS) // scale_heads
    # [2, tokens, layers, heads] scales selected per element.
    selected = scales.permute(1, 0, 2, 3)[:, page][..., group]
    decoded = (encoded.float() * selected[..., None]).to(compute).float()
    return encoded, scales, decoded


def _convert(
    pool, export, heads, start, end, page_tokens, scale_heads, compute
):
    """Import ``[start, end)`` as the worker does: one call per page span."""
    encoded, scales, _ = export
    group = pool.groups[0]
    destination_tokens = group.page_tokens
    h0, count_heads = heads[0], len(heads)
    elements = destination_tokens * LAYERS * count_heads * DIM
    rounding = tuple(torch.zeros(8 * elements, dtype=torch.uint8) for _ in "kv")
    values = torch.empty(elements, dtype=torch.float32)
    position = start
    while position < end:
        page, offset = divmod(position, destination_tokens)
        count = min(end - position, destination_tokens - offset)
        source_offset = position % page_tokens
        first = position // page_tokens - start // page_tokens
        pages = math.ceil((source_offset + count) / page_tokens)
        # The raw span sits at the start of each field's byte buffer.
        source = []
        for field in (0, 1):
            span = encoded[
                field, position - start : position - start + count, :, heads
            ]
            raw = rounding[field][: span.numel()].view(torch.float8_e4m3fn)
            raw.copy_(span.reshape(-1))
            source.append(raw.view(span.shape))
        groups = slice(h0 // scale_heads, -(-(h0 + count_heads) // scale_heads))
        copy_converted(
            pool,
            0,
            tuple(source),
            scales[first : first + pages, :, :, groups].contiguous(),
            values,
            rounding,
            (page + 1,),
            offset,
            source_offset,
            page_tokens,
            scale_heads,
            compute,
        )
        position += count


def _reference(pool, decoded, heads, start, end):
    """Write the decoded values through the layers' own write path."""
    page_tokens = pool.groups[0].page_tokens
    position = start
    while position < end:
        page, offset = divmod(position, page_tokens)
        count = min(end - position, page_tokens - offset)
        rows = slice(position - start, position - start + count)
        for layer in range(LAYERS):
            pool.state(f"layers.{layer}").write(
                (page + 1,),
                start=offset,
                key=decoded[0, rows, layer][:, heads],
                value=decoded[1, rows, layer][:, heads],
            )
        position += count


def _assert_same_pool(actual, expected):
    for layer in range(LAYERS):
        state = actual.state(f"layers.{layer}")
        reference = expected.state(f"layers.{layer}")
        for field in ("key", "value"):
            mine, theirs = state.tensors[field], reference.tensors[field]
            if isinstance(mine, QuantizedTensor):
                mine, theirs = mine.buffers(), theirs.buffers()
            else:
                mine, theirs = {"values": mine}, {"values": theirs}
            for name, tensor in mine.items():
                other = theirs[name]
                assert torch.equal(
                    tensor.view(torch.uint8), other.view(torch.uint8)
                ), (layer, field, name)
            assert torch.equal(
                state.initialized[field], reference.initialized[field]
            )


@pytest.mark.parametrize(
    ("source_pages", "destination_pages", "start", "end"),
    [
        # Equal pages, a span starting mid-page.
        (4, 4, 2, 11),
        # Resharded page sizes in both directions.
        (4, 8, 6, 21),
        (8, 4, 3, 19),
    ],
)
@pytest.mark.parametrize(
    ("heads", "scale_heads"),
    [
        # The destination's heads lie in one source scale group.
        ((0, 1), 2),
        # They split across two source scale groups.
        ((1, 2), 2),
        # Every source rank held one head.
        ((1, 2, 3), 1),
    ],
)
@pytest.mark.parametrize("compute", (torch.bfloat16, torch.float32))
def test_fp8_spans_encode_like_writes_of_their_decoded_values(
    source_pages, destination_pages, start, end, heads, scale_heads, compute
):
    generator = torch.Generator().manual_seed(7)
    export = _export(generator, start, end, source_pages, scale_heads, compute)
    actual = _pool(heads, destination_pages, compute, True)
    expected = _pool(heads, destination_pages, compute, True)

    _convert(
        actual,
        export,
        list(heads),
        start,
        end,
        source_pages,
        scale_heads,
        compute,
    )
    _reference(expected, export[2], list(heads), start, end)
    _assert_same_pool(actual, expected)


@pytest.mark.parametrize("resident", (0.25, 64.0))
def test_fp8_append_preserves_resident_tokens_and_grows_only_as_needed(
    resident,
):
    heads, start, end = [0, 1], 3, 8
    generator = torch.Generator().manual_seed(3)
    export = _export(generator, start, end, 8, 2, torch.bfloat16)
    pools = [_pool(heads, 8, torch.bfloat16, True) for _ in range(2)]
    prefix = torch.full((start, len(heads), DIM), resident)
    for pool in pools:
        for layer in range(LAYERS):
            pool.state(f"layers.{layer}").write(
                (1,), start=0, key=prefix, value=-prefix
            )
    before = pools[0].scales_of("key")[:, 1].clone()

    _convert(pools[0], export, heads, start, end, 8, 2, torch.bfloat16)
    _reference(pools[1], export[2], heads, start, end)
    _assert_same_pool(*pools)

    # Scales never shrink; the resident prefix still decodes to its values.
    after = pools[0].scales_of("key")[:, 1]
    assert torch.all(after >= before)
    for layer in range(LAYERS):
        read = pools[0].state(f"layers.{layer}").read((1,), start=0, length=3)
        torch.testing.assert_close(read[0].float(), prefix, rtol=0.07, atol=0)


def test_dense_spans_convert_their_dtype_like_writes():
    heads, start, end = [1, 2], 5, 14
    generator = torch.Generator().manual_seed(11)
    values = torch.randn(
        2, end - start, LAYERS, TOTAL_HEADS, DIM, generator=generator
    )
    actual = _pool(heads, 4, torch.bfloat16, False)
    expected = _pool(heads, 4, torch.bfloat16, False)
    elements = 4 * LAYERS * len(heads) * DIM
    rounding = tuple(torch.zeros(8 * elements, dtype=torch.uint8) for _ in "kv")
    position = start
    while position < end:
        page, offset = divmod(position, 4)
        count = min(end - position, 4 - offset)
        rows = slice(position - start, position - start + count)
        source = tuple(
            values[field, rows][:, :, heads].contiguous() for field in (0, 1)
        )
        copy_converted(
            actual,
            0,
            source,
            None,
            torch.empty(elements),
            rounding,
            (page + 1,),
            offset,
            0,
            4,
            0,
            torch.float32,
        )
        position += count
    _reference(expected, values, heads, start, end)
    _assert_same_pool(actual, expected)
