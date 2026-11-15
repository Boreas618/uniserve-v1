"""The vsa_add_gated_tiles program against the operator's two steps.

The operator adds one tile's gated compression to each of its rows, in place,
as two eager steps (:mod:`uniserve_testing.kernels.vsa.operands`):

```text
row += narrow(compressed[tile] · gate[row])      the product narrowed first
```

and declares the result within two roundings of the storage type; with the
product narrowed before the sum, the comparison is bit for bit. Each case
launches the compilation the package's launch table states for its
descriptor key, which compiles without FP32 contraction, so the product and
the sum round separately on a device as on the interpreter.

The cases:

- **a strided output and gate** whose row stride is wider than
  ``heads · width``, and a **padded compression** whose tile stride is longer
  than ``width``;
- **``block_rows`` narrower than ``tile_rows``**, with rows in a tile's second
  and later blocks, so the block-to-tile division is exercised;
- **bf16, f16 and f32**, since the product's narrowing is the type's;
- **the fusion trap**, on packed and on strided rows, where a contracted
  multiply-add stores another value; each trap case first asserts that its
  data tells the two arithmetics apart, so its pass is evidence about the
  kernel rather than about the data. It needs a device.
"""

import dataclasses

import pytest
import torch
from uniserve_testing.kernels.vsa import operands

pytestmark = pytest.mark.unit


@dataclasses.dataclass(frozen=True)
class Rows:
    """One case: the storage type, the rows, heads and features, the tile and
    block rows, the row stride of the output and the gate and the compression's
    tile stride (the packed ones when unset), and whether the values are the
    fusion trap's.
    """  # noqa: D205

    dtype: str
    rows: int
    heads: int
    width: int
    tile: int
    block: int
    row_stride: int | None = None
    tile_stride: int | None = None
    trap: bool = False


CASES = [
    pytest.param(Rows("bf16", 128, 4, 64, 64, 16), id="bf16"),
    pytest.param(
        Rows("f16", 128, 3, 128, 64, 16, row_stride=640, tile_stride=160),
        id="f16 strided",
    ),
    pytest.param(Rows("f32", 64, 2, 32, 64, 32), id="f32"),
    pytest.param(
        Rows("bf16", 128, 4, 64, 64, 16, trap=True),
        id="the fusion trap",
        marks=pytest.mark.gpu,
    ),
    pytest.param(
        Rows("bf16", 64, 2, 32, 64, 16, row_stride=96, trap=True),
        id="the fusion trap, strided rows",
        marks=pytest.mark.gpu,
    ),
]

#: The seed every case draws its operands from.
SEED = 31


def launch(stated, output, compressed, gate):
    """One launch over every row block and head of `output`, in place, as
    the package states it.
    """  # noqa: D205
    rows, heads, _ = output.shape
    block = stated.constexprs["block_rows"]
    stated.kernel[(rows // block, heads, 1)](
        output,
        compressed,
        gate,
        output.stride(0),
        output.stride(1),
        gate.stride(0),
        gate.stride(1),
        compressed.stride(0),
        compressed.stride(1),
        **stated.constexprs,
        **stated.options,
    )


def per_row(compressed, rows, tile):
    """Each row's tile compression: ``[heads, tiles, width]`` read at
    ``row // tile``.
    """  # noqa: D205
    index = (torch.arange(rows) // tile).clamp(max=compressed.shape[1] - 1)
    return compressed[:, index, :].permute(1, 0, 2)


@pytest.mark.parametrize("case", CASES)
def test_every_row_gains_its_narrowed_gated_tile(
    case, torch_device, interpreted, stated_launch, launch_target
):
    dtype, rows, heads, width = case.dtype, case.rows, case.heads, case.width
    tile, block, trap = case.tile, case.block, case.trap
    element = operands.DTYPES[dtype]
    tiles = rows // tile
    torch.manual_seed(SEED)
    before = operands.drawn(
        (rows, heads, width), 0.25, element, trap, operands.TRAP_ROW
    )
    gate = operands.drawn(
        (rows, heads, width), 0.5, element, trap, operands.TRAP_FACTOR
    )
    compressed = operands.drawn(
        (heads, tiles, width), 0.5, element, trap, operands.TRAP_FACTOR
    )
    row_stride = case.row_stride or heads * width
    tile_stride = case.tile_stride or width
    output = operands.strided_rows(before, row_stride, torch_device)

    stated = stated_launch(
        "vsa_add_gated_tiles",
        operands.key(
            dtype=dtype, tile_rows=tile, width=width, block_rows=block
        ),
        launch_target,
    )
    launch(
        stated,
        output,
        operands.padded_tiles(compressed, tile_stride, torch_device),
        operands.strided_rows(gate, row_stride, torch_device),
    )

    tile_values = per_row(compressed, rows, tile)
    expected = operands.gated_sum(tile_values, before, gate, interpreted)
    if trap:
        assert not torch.equal(
            expected, operands.fused_sum(tile_values, before, gate)
        ), (
            "the fusion case cannot tell a fused multiply-add from the "
            "operator's two roundings, so it would pass either way"
        )
    operands.assert_same(
        f"{dtype} rows {rows} heads {heads} width {width} tile {tile} block "
        f"{block} row stride {row_stride} tile stride {tile_stride}",
        output.cpu(),
        expected,
    )
