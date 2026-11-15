"""The vsa_prepare_tiles program against the stepwise recipe and torch's mean.

One program prepares one tile of one head: with ``norm`` it normalizes and
rotates the tile's valid Q and K rows in place as ``qk_norm_rope`` evaluates
its stepwise rounding, it stores zeros to the Q, K and V rows past the tile's
valid size, and it writes the tile's FP32 means of the stored rows to the
contiguous ``[tiles, heads, width]`` pooled outputs.

Each case launches the compilation the package's launch table states for its
descriptor key (the epsilon the key's bits encode, two warps, no FP32
contraction) on the operator's operands: 16-bit Q/K/V rows of the
descriptor's type, ``[width]`` weights of that type and FP32
``[rows, rotated / 2]`` factor tables. Without normalization the rows stand in
for the weights and the factor tables, which the module then does not read
(``VsaPrepareTiles::launch``). One program reads seven caller strides, so the
row buffers' row stride is wider than ``heads · width`` in every case.

**The rows.** The reference is the stepwise recipe in eager torch, each step
an operation of the source type's values rounded once into it:

```text
inverse = rsqrt(mean(row²) + ε)                   FP32, over the whole head
x       = narrow(row · inverse · weight)          the weighted normalization
c, s    = narrow(cos), narrow(sin)                the factors
x[a]    = narrow(narrow(x[a]·c) − narrow(x[b]·s)) a < h = rotated/2, b = a + h
x[b]    = narrow(narrow(x[b]·c) + narrow(x[a]·s))
```

with the features past ``rotated`` only normalized. Every step after the
inverse is exact arithmetic rounded once, which the kernel and eager torch
evaluate alike; the inverse is an FP32 reduction associated as the kernel's
tile is laid out, and its last bits move a narrowed value by a unit where the
FP32 product lies at a rounding boundary. So a valid row is held to the
operator's bound, ``rounding(tile_rows · width, f32, dtype)``, applied to the
sum of the magnitudes of the terms that produced each element
(``|x[a]·c| + |x[b]·s|`` for a rotated feature, ``|x|`` for the others): a
rotation can cancel far below its terms, and a bound relative to the result
fails there on an error of one unit of its terms. Below the type's smallest
normal value, where its grid is uniform, one step of that grid is allowed. The
rows past a tile's valid size are zeros, and the V rows are otherwise
untouched.

**The means** sum each tile's stored rows in the order in which
``torch.sum(dtype=torch.float32)`` reduces a strided row axis on a device, and
divide by the valid size (at least one), so on a device they equal torch's own
mean of the stored rows bit for bit. The host's ``torch.sum`` associates
otherwise, so on the interpreter they are held to the operator's bound.

**Where the cases run.** The normalizing path spells its packed 16-bit pair
arithmetic as PTX, which Triton's interpreter cannot execute, so every case
with ``norm`` needs a device; the cases without it run in both modes.

The cases:

- **the fixture's cell**: 128-feature heads of which 96 rotate, two tiles whose
  second is partly empty, on a row stride of 1200 against ``heads · width``
  1024, so the row and the head stride are both honoured by the loads, the
  stores and the means;
- **a tile with fewer valid rows than a row block**: the masked tail inside a
  block, and the mean dividing by the valid size;
- **``rotated < width``**: with ``cosine = 0, sine = 1`` the rotated features
  exchange pairs, and the features past ``rotated`` must come out identical to
  a run with other factors, the pass-through mask;
- **no normalization**: only the zeroing and the pooling;
- **an empty tile**: its means are zeros, not a division by the tile's row
  count and not NaNs.
"""

import dataclasses
import struct

import pytest
import torch

from tests.python.kernels.vsa import operands

pytestmark = pytest.mark.unit

#: The bits of the descriptor's epsilon (``PrepareTilesDescriptor``'s
#: ``epsilon``), the fixture's cell; a key prints a float's bits, and the
#: descriptor means the float they encode.
EPSILON = 0x3727C5AC

#: `ops::tolerance::unit_roundoff` of each element type.
UNIT_ROUNDOFF = {"bf16": 2.0**-8, "f16": 2.0**-11, "f32": 2.0**-24}

#: The seed every case draws its rows, weights and factors from.
SEED = 11


def epsilon():
    """The float :data:`EPSILON`'s bits encode."""
    return struct.unpack("<f", struct.pack("<I", EPSILON))[0]


@dataclasses.dataclass(frozen=True)
class Tiles:
    """One launch's descriptor and operands' extents: the rows' type, the
    tiles' valid sizes, the heads and features, the rotated features, the tile
    and block rows, whether the rows are normalized, the rows' stride and
    whether the factor tables rotate (``cosine = 0, sine = 1`` otherwise).
    """  # noqa: D205

    dtype: str
    sizes: tuple[int, ...]
    heads: int
    width: int
    rotated: int
    tile: int
    block: int
    norm: bool
    row_stride: int
    rotating: bool = True

    @property
    def rows(self) -> int:
        return len(self.sizes) * self.tile

    @property
    def factor_stride(self) -> int:
        """The factor tables' row stride: the rows' own, or none without a
        rotation.
        """  # noqa: D205
        return self.row_stride if self.rotated else 0

    def key(self) -> str:
        """The key ``PrepareTilesDescriptor``'s ``Display`` prints."""
        return operands.key(
            dtype=self.dtype,
            heads=self.heads,
            width=self.width,
            rotated=self.rotated,
            tile_rows=self.tile,
            block_rows=self.block,
            epsilon=f"{EPSILON:#x}",
            norm="true" if self.norm else "false",
            query_row_stride=self.row_stride,
            query_head_stride=self.width,
            key_row_stride=self.row_stride,
            key_head_stride=self.width,
            value_row_stride=self.row_stride,
            value_head_stride=self.width,
            factor_stride=self.factor_stride,
        )


FIXTURE = Tiles("f16", (64, 37), 8, 128, 96, 64, 16, True, 1200)
SMALL_VALID = Tiles("f16", (5,), 2, 128, 96, 64, 16, True, 1200)
NARROW_ROTATION = Tiles("f16", (64,), 4, 64, 32, 64, 16, True, 320)
NO_NORM = Tiles("bf16", (64, 31), 4, 64, 0, 64, 16, False, 256)
EMPTY_TILE = Tiles("bf16", (0, 40), 4, 64, 0, 64, 16, False, 256)


@dataclasses.dataclass
class Operands:
    """The host values of one launch: Q, K and V rows, the two weights and
    the two factor tables.
    """  # noqa: D205

    rows: dict[str, torch.Tensor]
    weights: dict[str, torch.Tensor]
    cosine: torch.Tensor
    sine: torch.Tensor


def drawn(tiles):
    """The operands of `tiles`: Q, K and V from one normal draw, weights of the
    rows' type, and the rotation's factors per row.
    """  # noqa: D205
    element = operands.DTYPES[tiles.dtype]
    torch.manual_seed(SEED)
    features = torch.randn(tiles.rows, tiles.heads, tiles.width) * 0.5
    rows = {
        "query": features.to(element),
        "key": (features * 0.75).to(element),
        "value": (features - 0.25).to(element),
    }
    weights = {
        name: (torch.randn(tiles.width) * 0.25).to(element)
        for name in ("query", "key")
    }
    pairs = torch.linspace(0.0, 1.0, max(tiles.rotated // 2, 1))
    if tiles.rotating:
        cosine, sine = torch.cos(pairs), torch.sin(pairs)
    else:
        cosine, sine = torch.zeros_like(pairs), torch.ones_like(pairs)
    return Operands(
        rows,
        weights,
        cosine.expand(tiles.rows, -1).contiguous(),
        sine.expand(tiles.rows, -1).contiguous(),
    )


def factor_table(values, stride, device):
    """An FP32 ``[rows, rotated / 2]`` table on `device` at row stride
    `stride`.
    """  # noqa: D205
    rows, pairs = values.shape
    flat = torch.zeros(rows * max(stride, 1), device=device)
    table = torch.as_strided(flat, (rows, pairs), (max(stride, 1), 1))
    table.copy_(values.to(device))
    return table


def launch(stated, tiles, rows, weights, cosine, sine, sizes, pooled):
    """One launch over every tile and head of the rows, in place, as the
    package states it; the rows stand in for the normalization operands
    without ``norm``.
    """  # noqa: D205
    query, key, value = rows
    stated.kernel[(len(tiles.sizes), tiles.heads)](
        query,
        key,
        value,
        weights[0] if tiles.norm else query,
        weights[1] if tiles.norm else key,
        cosine if tiles.norm else query,
        sine if tiles.norm else query,
        sizes,
        *pooled,
        **stated.constexprs,
        **stated.options,
    )


def prepared(tiles, values, stated, device):
    """Launch `tiles` over the host `values`; the rows it stored and its means,
    on `device`.
    """  # noqa: D205
    rows = [
        operands.strided_rows(values.rows[name], tiles.row_stride, device)
        for name in ("query", "key", "value")
    ]
    pooled = [
        torch.full(
            (len(tiles.sizes), tiles.heads, tiles.width),
            float("nan"),
            device=device,
        )
        for _ in range(3)
    ]
    launch(
        stated,
        tiles,
        rows,
        [values.weights[name].to(device) for name in ("query", "key")],
        factor_table(values.cosine, tiles.factor_stride, device),
        factor_table(values.sine, tiles.factor_stride, device),
        torch.tensor(tiles.sizes, dtype=torch.int32, device=device),
        pooled,
    )
    return rows, pooled


def stepwise(rows, weight, cosine, sine, rotated):
    """The stepwise recipe over host ``[rows, heads, width]`` rows: the
    prepared rows, and the magnitude of the terms behind each element.
    """  # noqa: D205
    element = rows.dtype
    values = rows.to(torch.float32)
    inverse = torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon())
    normalized = ((values * inverse) * weight.to(torch.float32)).to(element)
    terms = normalized.to(torch.float32).abs()
    result = normalized.clone()
    half = rotated // 2
    if half:
        # One factor pair per rotated pair of the row, shared by its heads.
        c = cosine.to(element)[:, None, :]
        s = sine.to(element)[:, None, :]
        first, second = normalized[..., :half], normalized[..., half:rotated]
        result[..., :half] = first * c - second * s
        result[..., half:rotated] = second * c + first * s
        terms[..., :half] = (first * c).float().abs() + (
            second * s
        ).float().abs()
        terms[..., half:rotated] = (second * c).float().abs() + (
            first * s
        ).float().abs()
    return result, terms


def valid(tiles):
    """Each row's validity: inside its tile's valid size."""
    sizes = torch.tensor(tiles.sizes)
    return (torch.arange(tiles.tile) < sizes[:, None]).reshape(-1)


def pool(rows, sizes, tile):
    """The FP32 sum of each tile's rows, divided by its valid size (at least
    one), as torch evaluates it on the rows' device.
    """  # noqa: D205
    values = rows.view(sizes.numel(), tile, *rows.shape[1:])
    total = values.sum(1, dtype=torch.float32)
    return total / sizes.clamp_min(1).view(-1, 1, 1)


def assert_rows(tiles, name, stored, original, weight, values):
    """Hold one tensor's stored rows to the recipe: the valid Q and K rows
    within the operator's bound of the stepwise reference, the V rows as they
    were, and every padding row zero.
    """  # noqa: D205
    live = valid(tiles)
    got = stored.cpu()
    padding = got[~live]
    assert not bool(padding.any()), (
        f"the prepared {name} rows hold {int((padding != 0).sum())} non-zero "
        "padding values"
    )
    if not tiles.norm or name == "value":
        assert torch.equal(got[live], original[live]), (
            f"the {name} rows inside the valid sizes were written"
        )
        return
    expected, terms = stepwise(
        original, weight, values.cosine, values.sine, tiles.rotated
    )
    reduction = tiles.tile * tiles.width * UNIT_ROUNDOFF["f32"]
    rtol = 2.0 * reduction / (1.0 - reduction) + UNIT_ROUNDOFF[tiles.dtype]
    # A relative bound holds where the type's values are relative: below its
    # smallest normal value its grid is uniform, and one step of it is room.
    limits = torch.finfo(got.dtype)
    error = (got.float() - expected.float()).abs()[live]
    outside = ~(error <= rtol * terms[live] + limits.tiny * limits.eps)
    if outside.any():
        position = int(outside.reshape(-1).nonzero()[0])
        raise AssertionError(
            f"the prepared {name} rows: {int(outside.sum())} of "
            f"{outside.numel()} valid values outside rtol {rtol:.3e} of their "
            f"terms; first at {position}: "
            f"{float(got[live].reshape(-1)[position])} vs "
            f"{float(expected[live].reshape(-1)[position])}"
        )


def assert_means(tiles, name, pooled, stored, interpreted):
    """Hold one tensor's means to torch's own mean of its stored rows: bit
    for bit on a device; on the interpreter within the operator's bound,
    relative to the mean of the rows' magnitudes, since a mean of values of
    both signs may cancel below its terms.
    """  # noqa: D205
    sizes = torch.tensor(tiles.sizes, dtype=torch.int32, device=stored.device)
    got = pooled.cpu()
    expected = pool(stored, sizes, tiles.tile).cpu()
    assert not bool(torch.isnan(got).any()), f"the pooled {name} holds NaNs"
    if not interpreted:
        operands.assert_same(f"the pooled {name}", got, expected)
        return
    magnitude = pool(stored.abs(), sizes, tiles.tile).cpu()
    reduction = tiles.tile * tiles.width * UNIT_ROUNDOFF["f32"]
    rtol = 2.0 * reduction / (1.0 - reduction) + UNIT_ROUNDOFF["f32"]
    outside = ~((got - expected).abs() <= rtol * magnitude)
    assert not outside.any(), (
        f"the pooled {name}: {int(outside.sum())} of {got.numel()} means "
        f"outside rtol {rtol:.3e} of their terms; max |error| = "
        f"{(got - expected).abs().max().item():.3e}"
    )


def assert_prepared(tiles, values, rows, pooled, interpreted):
    """Every assertion one launch's outputs answer to."""
    for index, name in enumerate(("query", "key", "value")):
        assert_rows(
            tiles,
            name,
            rows[index],
            values.rows[name],
            values.weights.get(name),
            values,
        )
        assert_means(tiles, name, pooled[index], rows[index], interpreted)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "tiles",
    [FIXTURE, SMALL_VALID],
    ids=["the fixture's cell", "fewer valid rows than a block"],
)
def test_normalized_tiles_follow_the_stepwise_recipe(
    tiles, torch_device, interpreted, stated_launch, launch_target
):
    values = drawn(tiles)
    stated = stated_launch("vsa_prepare_tiles", tiles.key(), launch_target)

    rows, pooled = prepared(tiles, values, stated, torch_device)

    assert_prepared(tiles, values, rows, pooled, interpreted)


@pytest.mark.gpu
def test_a_narrowed_rotation_passes_the_rest_through(
    torch_device, interpreted, stated_launch, launch_target
):
    turned_tiles = NARROW_ROTATION
    exchanged_tiles = dataclasses.replace(NARROW_ROTATION, rotating=False)
    stated = stated_launch(
        "vsa_prepare_tiles", turned_tiles.key(), launch_target
    )
    queries = []
    for tiles in (turned_tiles, exchanged_tiles):
        values = drawn(tiles)
        rows, pooled = prepared(tiles, values, stated, torch_device)
        assert_prepared(tiles, values, rows, pooled, interpreted)
        queries.append(rows[0].cpu())

    turned, exchanged = queries
    rotated = NARROW_ROTATION.rotated
    operands.assert_same(
        "the pass-through features (past `rotated`) of a narrowed rotation",
        exchanged[..., rotated:],
        turned[..., rotated:],
    )
    assert not torch.equal(exchanged[..., :rotated], turned[..., :rotated]), (
        "the rotation moved nothing, so the case does not test its coverage"
    )


@pytest.mark.parametrize(
    "tiles", [NO_NORM, EMPTY_TILE], ids=["no normalization", "an empty tile"]
)
def test_plain_tiles_are_zeroed_and_pooled(
    tiles, torch_device, interpreted, stated_launch, launch_target
):
    values = drawn(tiles)
    stated = stated_launch("vsa_prepare_tiles", tiles.key(), launch_target)

    rows, pooled = prepared(tiles, values, stated, torch_device)

    assert_prepared(tiles, values, rows, pooled, interpreted)
    for name, means in zip(("query", "key", "value"), pooled, strict=True):
        for tile, size in enumerate(tiles.sizes):
            assert size or not bool(means[tile].any()), (
                f"the pooled {name} of empty tile {tile} is not zero"
            )
