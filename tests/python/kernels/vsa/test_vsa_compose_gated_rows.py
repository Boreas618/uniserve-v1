"""The vsa_compose_gated_rows program against the operator's steps.

The operator writes one interval of every member's gated rows into a payload,
member-major, with the arithmetic of ``vsa_add_gated_tiles``
(:mod:`tests.python.kernels.vsa.operands`):

```text
row  = member · owner_rows + start + i          the member's own row numbering
tile = row / tile_rows                          the compression that row reads
out[slots[member], i] = narrow(fine[row] + narrow(compressed[tile] · gate[row]))
```

The two narrowings are the operator's, and the comparison is bit for bit.
Each case launches the compilation the package's launch table states for its
descriptor key, which compiles without FP32 contraction. The cases:

- **a permuted slot table**, ``slots = (2, 0, 1)``: a kernel that wrote
  members in order passes any case with the identity and fails this one;
- **an interval that is not the whole member** (``start > 0``,
  ``count < owner_rows``), so the row base and the tile division are exercised
  at an offset;
- **a row block inside one tile** (``block_rows`` 16 of ``tile_rows`` 64), the
  operator's stated invariant;
- **strided ``fine`` and ``gate``** whose row stride is wider than
  ``heads · width``, and a **padded compression** whose tile stride is longer
  than ``width``;
- **bf16, f16 and f32**;
- **the fusion trap**, where a contracted multiply-add stores another value;
  the case first asserts that its data tells the two arithmetics apart. It
  needs a device.
"""

import dataclasses

import pytest
import torch

from tests.python.kernels.vsa import operands

pytestmark = pytest.mark.unit


@dataclasses.dataclass(frozen=True)
class Interval:
    """One case: the storage type, the members and the rows each owns, the
    interval's start and count, the heads and features, the tile and block
    rows, the payload slot of each member (the identity when unset), the row
    stride of the fine rows and the gate and the compression's tile stride
    (the packed ones when unset), and whether the values are the fusion
    trap's.
    """  # noqa: D205

    dtype: str
    members: int
    owner_rows: int
    start: int
    count: int
    heads: int
    width: int
    tile: int
    block: int
    slots: tuple[int, ...] | None = None
    fine_stride: int | None = None
    tile_stride: int | None = None
    trap: bool = False


CASES = [
    pytest.param(
        Interval("bf16", 3, 128, 32, 64, 4, 64, 64, 16, slots=(2, 0, 1)),
        id="bf16, permuted slots, an interval inside the member",
    ),
    pytest.param(
        Interval(
            "f16",
            2,
            64,
            16,
            48,
            3,
            128,
            64,
            16,
            fine_stride=768,
            tile_stride=192,
        ),
        id="f16, strided rows and tiles",
    ),
    pytest.param(Interval("f32", 1, 64, 0, 64, 2, 32, 64, 32), id="f32"),
    pytest.param(
        Interval(
            "bf16", 2, 128, 0, 128, 2, 64, 64, 16, fine_stride=192, trap=True
        ),
        id="the fusion trap, strided rows",
        marks=pytest.mark.gpu,
    ),
]

#: The seed every case draws its operands from.
SEED = 37


def launch(stated, output, fine, compressed, gate, owner_rows, start):
    """One launch over every row block, head and member of `output`, as the
    package states it.
    """  # noqa: D205
    members, count, heads, _ = output.shape
    block = stated.constexprs["block_rows"]
    stated.kernel[(count // block, heads, members)](
        output,
        fine,
        compressed,
        gate,
        fine.stride(0),
        fine.stride(1),
        gate.stride(0),
        gate.stride(1),
        compressed.stride(0),
        compressed.stride(1),
        owner_rows,
        start,
        count,
        **stated.constexprs,
        **stated.options,
    )


def member_rows(tensor, slots, owner_rows, start, count, tile):
    """The rows of `tensor` each member's interval reads, laid out as the
    payload ``[members, count, ...]`` in slot order, and the tile of each.
    """  # noqa: D205
    order = [0] * len(slots)
    for member, slot in enumerate(slots):
        order[slot] = member
    rows = torch.tensor(
        [
            [member * owner_rows + start + index for index in range(count)]
            for member in order
        ]
    )
    return tensor[rows], rows // tile


@pytest.mark.parametrize("case", CASES)
def test_every_member_interval_lands_in_its_slot(
    case, torch_device, interpreted, stated_launch, launch_target
):
    dtype, members, heads, width = (
        case.dtype,
        case.members,
        case.heads,
        case.width,
    )
    owner_rows, start, count = case.owner_rows, case.start, case.count
    tile, block, trap = case.tile, case.block, case.trap
    element = operands.DTYPES[dtype]
    slots = case.slots or tuple(range(members))
    rows = members * owner_rows
    tiles = rows // tile
    torch.manual_seed(SEED)
    fine = operands.drawn(
        (rows, heads, width), 0.25, element, trap, operands.TRAP_ROW
    )
    gate = operands.drawn(
        (rows, heads, width), 0.5, element, trap, operands.TRAP_FACTOR
    )
    compressed = operands.drawn(
        (heads, tiles, width), 0.5, element, trap, operands.TRAP_FACTOR
    )
    fine_stride = case.fine_stride or heads * width
    tile_stride = case.tile_stride or width
    output = torch.full(
        (members, count, heads, width),
        float("nan"),
        dtype=element,
        device=torch_device,
    )

    stated = stated_launch(
        "vsa_compose_gated_rows",
        operands.key(
            dtype=dtype,
            heads=heads,
            tile_rows=tile,
            width=width,
            block_rows=block,
            slots=",".join(str(slot) for slot in slots),
            members=members,
        ),
        launch_target,
    )
    launch(
        stated,
        output,
        operands.strided_rows(fine, fine_stride, torch_device),
        operands.padded_tiles(compressed, tile_stride, torch_device),
        operands.strided_rows(gate, fine_stride, torch_device),
        owner_rows,
        start,
    )

    fine_rows, tile_of = member_rows(
        fine, slots, owner_rows, start, count, tile
    )
    gate_rows, _ = member_rows(gate, slots, owner_rows, start, count, tile)
    # Each payload row's compression: `[heads, width]` of its tile.
    tile_values = compressed.permute(1, 0, 2)[tile_of]
    expected = operands.gated_sum(
        tile_values, fine_rows, gate_rows, interpreted
    )
    if trap:
        assert not torch.equal(
            expected, operands.fused_sum(tile_values, fine_rows, gate_rows)
        ), (
            "the fusion case cannot tell a fused multiply-add from the "
            "operator's two roundings, so it would pass either way"
        )
    operands.assert_same(
        f"{dtype} members {members} owner rows {owner_rows} interval "
        f"{start}..{start + count} heads {heads} width {width} tile {tile} "
        f"block {block} slots {slots} fine stride {fine_stride} tile stride "
        f"{tile_stride}",
        output.cpu(),
        expected,
    )
