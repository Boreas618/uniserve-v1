"""What the video sparse attention checks share.

- the descriptor key a launch is stated for (:func:`key`), and row and tile
  buffers whose strides are the caller's (:func:`strided_rows`,
  :func:`padded_tiles`);
- the zero-padding rule (:func:`zero_padding`): the rows of a tile past its
  valid size read as zeros;
- the gated-tile arithmetic (:func:`gated_sum`). ``vsa_add_gated_tiles`` and
  ``vsa_compose_gated_rows`` both add a tile's compression, gated per row, to
  a row as the operator's two eager steps:

```text
total = narrow(row + narrow(compressed[tile] · gate[row]))
```

each product and sum evaluated in FP32 and narrowed to the storage type. With
the product narrowed before the sum there is nothing left for a bound to
cover, so both checks compare bits. A device narrows to nearest; Triton's
interpreter narrows into BF16 by truncation (its F16 narrowing rounds to
nearest), so the reference narrows as the session does (:func:`narrowed`).

A kernel whose compiler contracted the product into the sum would round once
(:func:`fused_sum`). The fusion trap is data on which the two arithmetics
differ: BF16 values (spacing ``2^-7`` in ``[1, 2)``, ``2^-6`` in ``[2, 4)``)
where the FP32 product ``(1 + 2^-4)^2 = 1 + 2^-3 + 2^-8`` lies halfway between
two BF16 neighbours and narrows to the even one, ``1 + 2^-3``. Added to
``1 + 2^-7``, that narrowed product gives ``2 + 2^-3 + 2^-7``, halfway again,
which rounds to the even ``2 + 2^-3``; the fused sum is ``2^-8`` above that
tie and rounds up to ``2 + 2^-3 + 2^-6``. A product merely not representable
in BF16 is not enough: the sum's own rounding can absorb its tail. Contraction
is a device compiler's decision and the trap turns on rounding to nearest, so
a trap case needs a device.
"""

from __future__ import annotations

import numpy as np
import torch

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}

#: The fusion trap's values of the rows, of the gate and of the compression.
TRAP_ROW = 1.0 + 2.0**-7
TRAP_FACTOR = 1.0 + 2.0**-4


def key(**fields) -> str:
    """The descriptor key a descriptor's ``Display`` prints for `fields`, in
    their order.
    """  # noqa: D205
    return "_".join(f"{name}_{value}" for name, value in fields.items())


def zero_padding(rows: torch.Tensor, sizes: torch.Tensor, tile: int) -> None:
    """Zero every row of `rows` at or past its tile's valid size, in place.

    `rows` is ``[tiles · tile, heads, width]`` with whole tiles of rows,
    `sizes` the valid rows of each tile, on the same device.
    """
    live = torch.arange(tile, device=rows.device) < sizes[:, None]
    rows.view(sizes.numel(), tile, *rows.shape[1:]).masked_fill_(
        ~live[:, :, None, None], 0
    )


def narrowed(values: torch.Tensor, element: torch.dtype, interpreted: bool):
    """FP32 `values` narrowed to `element` as the session narrows them."""
    if interpreted and element == torch.bfloat16:
        bits = values.to(torch.float32).contiguous().numpy().view(np.uint32)
        bits = bits & np.uint32(0xFFFF0000)
        return torch.from_numpy(bits.view(np.float32)).to(element)
    return values.to(torch.float32).to(element)


def gated_sum(compressed, rows, gate, interpreted):
    """The operator's two steps over aligned host tensors of one storage type:
    the product narrowed, then the sum narrowed.
    """  # noqa: D205
    element = rows.dtype
    product = narrowed(
        compressed.to(torch.float32) * gate.to(torch.float32),
        element,
        interpreted,
    )
    return narrowed(
        rows.to(torch.float32) + product.to(torch.float32),
        element,
        interpreted,
    )


def fused_sum(compressed, rows, gate):
    """The same sum with the product left in FP32 and one rounding to nearest:
    what a contracted multiply-add stores.
    """  # noqa: D205
    total = rows.to(torch.float32) + compressed.to(torch.float32) * gate.to(
        torch.float32
    )
    return total.to(rows.dtype)


def drawn(shape, scale, element, trap, value):
    """Normal values of deviation `scale` in `element`, or the trap's `value`
    in every position.
    """  # noqa: D205
    if trap:
        return torch.full(shape, value).to(element)
    return (torch.randn(shape) * scale).to(element)


def strided_rows(values, row_stride, device):
    """`[rows, heads, width]` values copied into a buffer on `device` whose
    rows are `row_stride` elements apart, and that view.
    """  # noqa: D205
    rows, _, width = values.shape
    flat = torch.zeros(rows * row_stride, dtype=values.dtype, device=device)
    view = torch.as_strided(flat, values.shape, (row_stride, width, 1))
    view.copy_(values.to(device))
    return view


def padded_tiles(values, tile_stride, device):
    """`[heads, tiles, width]` values copied into a buffer on `device` whose
    tiles are `tile_stride` elements apart, and that view.
    """  # noqa: D205
    heads, tiles, width = values.shape
    flat = torch.zeros(
        heads * tiles * tile_stride, dtype=values.dtype, device=device
    )
    view = torch.as_strided(
        flat, values.shape, (tiles * tile_stride, tile_stride, 1)
    )
    view.copy_(values.to(device))
    return view


def assert_same(label, got, expected):
    """Hold `got` to `expected` element for element, the worst one named."""
    if torch.equal(got, expected):
        return
    difference = (got.to(torch.float64) - expected.to(torch.float64)).abs()
    position = int(difference.reshape(-1).argmax())
    raise AssertionError(
        f"{label}: {int((difference > 0).sum())} of {got.numel()} elements "
        f"differ; worst at {position}: {float(got.reshape(-1)[position])} vs "
        f"{float(expected.reshape(-1)[position])}"
    )
