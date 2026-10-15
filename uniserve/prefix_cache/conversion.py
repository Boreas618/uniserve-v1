"""Conversion of imported K/V pages into a unit pool's representation.

An importing rank receives a span of another rank's K/V pages as raw
``[tokens, layers, heads, dim]`` keys and values of the source dtype and,
for FP8 sources, the source pages' scales. ``copy_converted`` decodes the
span and encodes it into the destination units with the pool's own write
numerics, so imported values round and scale exactly as locally written ones.
The caller owns every scratch buffer and the stream the work runs on.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import TYPE_CHECKING, Literal, cast

import torch

from uniserve.quantization import Quantizer

from . import mha
from .state import decode_region
from .units import UnitPool, named_fields, stream_scope

if TYPE_CHECKING:
    from .plan import CachePlan

__all__ = ["copy_converted"]

_FIELDS: tuple[Literal["key", "value"], ...] = ("key", "value")


def copy_converted(
    pool: UnitPool,
    group: int,
    source: tuple[torch.Tensor, torch.Tensor],
    scales: torch.Tensor | None,
    values: torch.Tensor,
    rounding: tuple[torch.Tensor, torch.Tensor],
    units: Sequence[int],
    offset: int,
    source_offset: int,
    source_page_tokens: int,
    scale_heads: int,
    compute_dtype: torch.dtype,
) -> None:
    """Decode one source span and encode it into the selected destination units.

    Args:
        pool: The destination unit pool.
        group: The destination cache group.
        source: Raw keys and values ``[count, layers, heads, dim]`` of the
            source dtype over this rank's layers and K/V heads of ``group``.
        scales: For an FP8 source, float32 ``[pages, 2, layers,
            scale_groups]``: one scale per source page the span touches, per
            key/value field, layer and source head group (the heads one
            source rank stored). ``None`` for a dense source.
        values: Flat float32 conversion scratch of at least
            ``count * layers * heads * dim`` elements.
        rounding: Per field, the raw byte buffer ``source[field]`` lies at the
            start of. Bytes past ``4 * count * layers * heads * dim`` are the
            rounding scratch of narrow compute dtypes: an FP8 input occupies
            at most the first quarter of that prefix.
        units: The destination page's unit per row of ``units_per_page``.
        offset: First destination token within the page.
        source_offset: Token offset of the span's first token within its
            first source page.
        source_page_tokens: Tokens per source page, which select scale rows.
        scale_heads: Heads of one source scale group; 0 for a dense source.
        compute_dtype: The source's logical dtype, through which decoded
            values round before destination encoding.

    The destination encoding (FP8 scale growth and rescaling of resident
    values, initialization flags) is ``mha.State.copy_region``.
    """
    cache_group = pool.groups[group]
    # Scale groups are aligned to global heads; this rank's heads start at
    # its first global K/V head.
    head_offset = pool.head_offset(cache_group.layers[0])
    columns = pool.planes.columns
    count, num_layers, num_heads, head_dim = source[0].shape
    trailing = (num_layers, num_heads, head_dim)
    elements = source[0].numel()

    for field_index, raw in enumerate(source):
        converted = raw
        if scales is not None:
            converted = values[:elements].view(count, *trailing)
            # Rounding scratch lies in the raw page past the FP8 input, which
            # occupies at most its first quarter of these bytes. FP32 and
            # FP64 compute dtypes need no rounding scratch.
            rounded = (
                rounding[field_index][elements * 4 :].view(compute_dtype)
                if compute_dtype not in {torch.float32, torch.float64}
                else rounding[field_index][:0].view(compute_dtype)
            )
            # Walk source pages (one scale row each) and, within each, this
            # rank's heads grouped by source scale group.
            cursor, scale_index, token_offset = 0, 0, source_offset
            while cursor < count:
                length = min(source_page_tokens - token_offset, count - cursor)
                head, head_group = 0, 0
                while head < num_heads:
                    heads = min(
                        scale_heads - (head_offset + head) % scale_heads,
                        num_heads - head,
                    )
                    for layer in range(num_layers):
                        page_region = (
                            slice(cursor, cursor + length),
                            layer,
                            slice(head, head + heads),
                        )
                        encoded = raw[page_region]
                        numerical = Quantizer("fp8").from_tensors(
                            {
                                "values": encoded,
                                "scale": scales[
                                    scale_index,
                                    field_index,
                                    layer,
                                    head_group,
                                ].reshape(()),
                            },
                            shape=tuple(encoded.shape),
                            dtype=compute_dtype,
                        )
                        decode_region(
                            numerical,
                            tuple(slice(0, size) for size in encoded.shape),
                            device=converted.device,
                            workspace={
                                "values": converted[page_region],
                                "rounded": rounded,
                            },
                        )
                    head += heads
                    head_group += 1
                cursor += length
                scale_index += 1
                token_offset = 0

        # Store the converted tokens into each layer's unit of the
        # destination page: layer k lies in unit k // columns.
        for column_index, name in enumerate(cache_group.layers):
            unit = units[column_index // columns]
            trailing_slice = (slice(0, num_heads), slice(0, head_dim))
            # A unit pool binds only MHA state layers.
            state = cast(mha.State, pool.state(name))
            state.copy_region(
                converted[:, column_index],
                field=_FIELDS[field_index],
                block=unit,
                source_slice=(slice(0, count), *trailing_slice),
                target_slice=(slice(offset, offset + count), *trailing_slice),
                workspace={},
            )


class PageConversion:
    """The dtype and FP8 conversion of imported pages, on one stream.

    Args:
        plan: The ``cache_plan`` result.
        num_units: The units the fields hold, sentinel included.
        scratch: uint8 ``[20 * elements + 8 * scales]``: raw keys at bytes
            ``[0, 8E)``, raw values at ``[8E, 16E)``, the FP32 conversion
            scratch at ``[16E, 20E)`` and FP32 source scales at ``[20E, 20E +
            8S)``, where ``E`` is ``elements`` (the largest group page's
            elements) and ``S`` is ``scales`` (its scale entries).
        elements: ``E`` above.
        scales: ``S`` above.
        stream: The stream conversions are enqueued on.
        names, fields: The pool's unit fields, as :class:`UnitResets` takes
            them.

    Raises:
        ValueError: The scratch disagrees with its layout.
    """

    def __init__(
        self,
        plan: CachePlan,
        num_units: int,
        scratch: torch.Tensor,
        elements: int,
        scales: int,
        stream: int,
        names: Sequence[str],
        *fields: torch.Tensor,
    ) -> None:
        elements, scales = int(elements), int(scales)
        if scratch.dtype != torch.uint8 or scratch.numel() < (
            20 * elements + 8 * scales
        ):
            raise ValueError("the conversion scratch disagrees with its layout")
        self._pool = UnitPool(
            plan.planes,
            named_fields(names, fields),
            plan.configs,
            num_units=int(num_units),
        )
        self._raw = scratch[: 16 * elements].view(2, 8 * elements)
        self._values = scratch[16 * elements : 20 * elements].view(
            torch.float32
        )
        self._scales = scratch[20 * elements : 20 * elements + 8 * scales].view(
            torch.float32
        )
        self._scope = stream_scope(int(stream), scratch.device)

    def convert(
        self,
        group: int,
        source_dtype: str,
        compute_dtype: str,
        count: int,
        offset: int,
        source_offset: int,
        source_page_tokens: int,
        scale_heads: int,
        pages: int,
        units: Sequence[int],
    ) -> None:
        """Write one imported span into the destination page's units.

        Reads ``[count, layers, heads, dim]`` raw keys and values of
        ``source_dtype`` from the scratch (with FP8, also ``[pages, 2,
        layers, scale groups]`` scales) and writes the destination page's
        ``units`` at token ``offset`` with ``copy_converted``.
        """
        pool, raw = self._pool, self._raw
        cache_group = pool.groups[int(group)]
        dtype = getattr(torch, source_dtype)
        layers, heads = len(cache_group.layers), cache_group.num_kv_heads
        shape = (int(count), layers, heads, cache_group.head_dim)
        nbytes = prod(shape) * dtype.itemsize
        source = (
            raw[0, :nbytes].view(dtype).view(shape),
            raw[1, :nbytes].view(dtype).view(shape),
        )
        source_scales = None
        if int(scale_heads) > 0:
            first = pool.head_offset(cache_group.layers[0])
            groups = -(-(first + heads) // int(scale_heads)) - first // int(
                scale_heads
            )
            source_scales = self._scales[
                : int(pages) * 2 * layers * groups
            ].view(int(pages), 2, layers, groups)
        with self._scope():
            copy_converted(
                pool,
                int(group),
                source,
                source_scales,
                self._values,
                (raw[0], raw[1]),
                tuple(int(unit) for unit in units),
                int(offset),
                int(source_offset),
                int(source_page_tokens),
                int(scale_heads),
                getattr(torch, compute_dtype),
            )
