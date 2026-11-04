#!/usr/bin/env python3
"""Run `vsa_prepare_tiles` on a device and check it against the tree's own reference.

    crates/program-server/python/.venv/bin/python prepare_check.py [--package DIR] [--device cuda]

**This check needs a GPU, and the reason is in the kernel rather than in this script.** The
stepwise normalize-and-rotate recipe stores bits, so the package spells its arithmetic as PTX:
`add.rn.f32`, `mul.rn.f32`, and the packed `cvt.rn.bf16x2.f32`/`fma.rn.f16x2` forms of the narrowed
halves. `tl.inline_asm_elementwise` does not execute under `TRITON_INTERPRET`, and the pooling's own
`_add_rn` uses it as well — so no part of this kernel can run on the build host, not even the
`norm = false` path. On a host without a device the script says exactly that and stops.

Two independent references, because this is the operator where a stride mix-up cannot be caught by
reading the key — one program reads seven caller strides at once:

- the tree's Triton kernel, `uniserve_kernels.attention.vsa_segments._prepare_tiles_kernel`, which is
  the definition of the arithmetic (`prepare_tiles` itself refuses to run without Triton, so the
  check launches the kernel with the same constexpr wiring that function passes);
- `uniserve/nn/attention/vsa/segments.py`'s `pool` and `zero_padding`, copied verbatim, for the half
  of the operator the kernel need not be trusted about: the padding zeros and the FP32 mean, which
  must equal torch's own reduction bit for bit.

The cases:
- **the fixture's cell**: 128-feature heads of which 96 rotate, two tiles whose second is partly
  empty, on a **strided** buffer whose row stride is wider than `heads · width` — so both the row and
  the head stride have to be honoured for the loads, the stores and the means;
- **`rotated < width`**, the narrowed rotation: with `cosine = 0, sine = 1` the rotated features
  exchange pairs, and the features **past `rotated`** must come out identical to the run with
  `cosine = 1, sine = 0`. That is the pass-through mask, and skipping it is what `rope_apply`'s
  narrowed form did wrong;
- **no normalization**: only zeroing and pooling, compared bit for bit against `pool` of the zeroed
  rows rather than against the reference kernel;
- **an empty tile**: its means are zeros — not a division by the tile's row count, and not NaNs.
"""

import argparse
import importlib.util
import pathlib
import sys

import torch

#: `PrepareTilesDescriptor`'s epsilon, `0x3727c5ac` — the fixture's cell.
EPSILON = 0x3727C5AC

ROOT = pathlib.Path(__file__).resolve().parents[5]


def reference_kernel():
    """The tree's `_prepare_tiles_kernel`, which defines the arithmetic this package computes."""
    sys.path.insert(0, str(ROOT / "uniserve_kernels" / "src"))
    from uniserve_kernels.attention.vsa_segments import _prepare_tiles_kernel

    return _prepare_tiles_kernel


def zero_padding(rows, variable_block_sizes, tile):
    """`uniserve/nn/attention/vsa/segments.py::zero_padding`, for the device as for the host."""
    live = torch.arange(tile, device=rows.device) < variable_block_sizes[:, None]
    rows.view(variable_block_sizes.numel(), tile, *rows.shape[1:]).masked_fill_(
        ~live[:, :, None, None], 0
    )


def pool(rows, variable_block_sizes, tile):
    """`uniserve/nn/attention/vsa/segments.py::pool`: the FP32 sum of a tile's rows, then divide."""
    values = rows.view(variable_block_sizes.numel(), tile, *rows.shape[1:])
    total = values.sum(1, dtype=torch.float32)
    return total / variable_block_sizes.clamp_min(1).view(-1, 1, 1)


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "vsa_prepare_tiles.py"
    spec = importlib.util.spec_from_file_location("vsa_prepare_tiles_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Case:
    """One launch's operands and the constexprs its descriptor gives them."""

    def __init__(
        self,
        *,
        device,
        dtype,
        rows,
        heads,
        width,
        rotated,
        tile,
        block,
        norm,
        sizes,
        row_stride=None,
        cosines=True,
        seed=11,
    ):
        self.device = device
        self.element = {"bf16": torch.bfloat16, "f16": torch.float16}[dtype]
        self.rows = rows
        self.heads = heads
        self.width = width
        self.rotated = rotated
        self.tile = tile
        self.block = block
        self.norm = norm
        self.sizes = sizes
        self.tiles = len(sizes)
        self.row_stride = heads * width if row_stride is None else row_stride
        self.head_stride = width

        torch.manual_seed(seed)
        features = torch.randn(rows, heads, width, dtype=torch.float32) * 0.5
        self.query = self.buffer(features)
        self.key = self.buffer(features * 0.75)
        self.value = self.buffer(features - 0.25)
        self.query_weight = self.weight()
        self.key_weight = self.weight()
        self.cosine = self.factor("cosine" if cosines else "zero")
        self.sine = self.factor("sine" if cosines else "one")

    def buffer(self, features):
        """Hold `[rows, heads, width]` features in a buffer whose row stride the case sets."""
        flat = torch.zeros(self.rows * self.row_stride, dtype=torch.float32, device=self.device)
        self.rows_view(flat).copy_(features.to(torch.float32))
        return flat

    def rows_view(self, flat, count=None):
        count = self.rows if count is None else count
        return torch.as_strided(
            flat, (count, self.heads, self.width), (self.row_stride, self.head_stride, 1)
        )

    def weight(self):
        flat = torch.zeros(self.row_stride, dtype=torch.float32, device=self.device)
        weight = torch.as_strided(flat, (1, 1, self.width), (self.width, self.width, 1))
        weight.copy_(torch.randn(1, 1, self.width, dtype=torch.float32) * 0.25)
        return flat

    def weight_view(self, flat):
        return torch.as_strided(flat, (1, 1, self.width), (self.width, self.width, 1))

    @property
    def factor_stride(self):
        return self.row_stride if self.rotated else 0

    def factor(self, kind):
        stride = max(self.factor_stride, 1)
        flat = torch.zeros(self.rows * stride, dtype=torch.float32, device=self.device)
        if not self.rotated:
            return flat
        table = torch.as_strided(flat, (self.rows, self.rotated // 2), (stride, 1))
        if kind == "cosine":
            table.copy_(torch.cos(torch.linspace(0.0, 1.0, self.rotated // 2, device=self.device)))
        elif kind == "sine":
            table.copy_(torch.sin(torch.linspace(0.0, 1.0, self.rotated // 2, device=self.device)))
        elif kind == "one":
            table.fill_(1.0)
        return flat

    def factor_view(self, flat):
        stride = max(self.factor_stride, 1)
        return torch.as_strided(flat, (self.rows, max(self.rotated // 2, 1)), (stride, 1))

    def constexprs(self):
        """The constexprs `src/build.py` passes for this descriptor."""
        return {
            "QUERY_STRIDES": (self.row_stride, self.head_stride),
            "KEY_STRIDES": (self.row_stride, self.head_stride),
            "VALUE_STRIDES": (self.row_stride, self.head_stride),
            # The stepwise routine reads a block's rows as one head's consecutive tokens.
            "QUERY_ROWS": (self.row_stride, 0),
            "KEY_ROWS": (self.row_stride, 0),
            "FACTOR_STRIDE": self.factor_stride,
            "HEADS": self.heads,
            "DIM": self.width,
            "ROTATED": self.rotated,
            "EPS": EPSILON,
            "TILE": self.tile,
            "BLOCK": self.block,
            "NORM": self.norm,
        }


def launch(kernel, case, buffers):
    """One launch of a `_prepare_tiles_kernel` over the case's buffers; returns its pooled tensors."""
    sizes = torch.tensor(case.sizes, dtype=torch.int32, device=case.device)
    pooled = [
        torch.full(
            (case.tiles, case.heads, case.width), float("nan"), dtype=torch.float32,
            device=case.device,
        )
        for _ in range(3)
    ]
    kernel[(case.tiles, case.heads)](
        case.rows_view(buffers["query"]),
        case.rows_view(buffers["key"]),
        case.rows_view(buffers["value"]),
        case.weight_view(buffers["query_weight"]) if case.norm else case.rows_view(buffers["query"]),
        case.weight_view(buffers["key_weight"]) if case.norm else case.rows_view(buffers["key"]),
        case.factor_view(buffers["cosine"]) if case.norm else case.rows_view(buffers["query"]),
        case.factor_view(buffers["sine"]) if case.norm else case.rows_view(buffers["query"]),
        sizes,
        pooled[0],
        pooled[1],
        pooled[2],
        **case.constexprs(),
    )
    return pooled


def clone(case):
    return {
        name: getattr(case, name).clone()
        for name in ("query", "key", "value", "query_weight", "key_weight", "cosine", "sine")
    }


def compare(what, mine, theirs):
    """Bit-for-bit equality, with the worst disagreement in the message."""
    if torch.equal(mine, theirs):
        return
    difference = (mine.to(torch.float64) - theirs.to(torch.float64)).abs()
    position = int(difference.reshape(-1).argmax())
    raise SystemExit(
        f"{what}: {int((difference > 0).sum())} of {mine.numel()} values differ; worst at "
        f"{position}: {float(mine.reshape(-1)[position])} vs {float(theirs.reshape(-1)[position])}"
    )


def check_fixture_geometry(module, reference, device, row_stride=1200):
    """The fixture's cell against the tree's kernel, on a strided buffer.

    `row_stride` must be wider than `heads · width` for the case to be about strides at all: a tile
    walk that assumed contiguity passes every identity case.
    """
    if row_stride <= 8 * 128:
        raise SystemExit(
            f"a row stride of {row_stride} is not wider than heads · width (1024), so this case "
            f"would not exercise the strides the descriptor carries"
        )
    case = Case(
        device=device, dtype="f16", rows=128, heads=8, width=128, rotated=96, tile=64, block=16,
        norm=True, sizes=[64, 37], row_stride=row_stride,
    )
    mine_buffers, their_buffers = clone(case), clone(case)
    mine = launch(module._prepare_tiles_kernel, case, mine_buffers)
    theirs = launch(reference, case, their_buffers)
    for name, a, b in zip(("query", "key", "value"), mine, theirs):
        compare(f"the pooled {name}", a, b)
    for name in ("query", "key", "value"):
        compare(f"the prepared {name} rows", mine_buffers[name], their_buffers[name])
    # The rows past each tile's valid size are zeros, in all three row tensors.
    sizes = torch.tensor(case.sizes, dtype=torch.int32, device=device)
    for name in ("query", "key", "value"):
        rows = case.rows_view(mine_buffers[name])
        live = torch.arange(case.tile, device=device) < sizes[:, None]
        padding = rows.view(case.tiles, case.tile, case.heads, case.width)[~live]
        if bool(padding.any()):
            raise SystemExit(f"the prepared {name} rows hold padding instead of zeros")
    return 3 * mine[0].numel() + 3 * case.rows * case.heads * case.width


def check_pass_through(module, reference, device):
    """With `rotated < width`, the features past the rotation do not move."""
    def run(cosines):
        case = Case(
            device=device, dtype="f16", rows=64, heads=4, width=64, rotated=32, tile=64, block=16,
            norm=True, sizes=[64], row_stride=320, cosines=cosines,
        )
        buffers = clone(case)
        pooled = launch(module._prepare_tiles_kernel, case, buffers)
        theirs = launch(reference, case, clone(case))
        compare(f"the pooled query with cosines={cosines}", pooled[0], theirs[0])
        return case.rows_view(buffers["query"]).clone()

    turned = run(True)
    # `cosine = 0, sine = 1` exchanges each rotated pair; the pass-through features stay put.
    exchanged = torch.zeros_like(turned)
    case = Case(
        device=device, dtype="f16", rows=64, heads=4, width=64, rotated=32, tile=64, block=16,
        norm=True, sizes=[64], row_stride=320, cosines=False,
    )
    buffers = clone(case)
    launch(module._prepare_tiles_kernel, case, buffers)
    exchanged.copy_(case.rows_view(buffers["query"]))
    rotated = case.rotated
    compare(
        "the pass-through features (past `rotated`) of a narrowed rotation",
        exchanged[..., rotated:],
        turned[..., rotated:],
    )
    if torch.equal(exchanged[..., :rotated], turned[..., :rotated]):
        raise SystemExit("the rotation moved nothing, so the case does not test its coverage")
    return int(exchanged.numel())


def check_no_norm(module, reference, device):
    """Without normalization: the padding and the mean, against `pool` itself."""
    case = Case(
        device=device, dtype="bf16", rows=128, heads=4, width=64, rotated=0, tile=64, block=16,
        norm=False, sizes=[64, 31], row_stride=256,
    )
    buffers = clone(case)
    mine = launch(module._prepare_tiles_kernel, case, buffers)
    sizes = torch.tensor(case.sizes, dtype=torch.int32, device=device)
    for name, flat, pooled in zip(
        ("query", "key", "value"),
        (buffers["query"], buffers["key"], buffers["value"]),
        mine,
    ):
        rows = case.rows_view(flat)
        expected = pool(rows, sizes, case.tile)
        compare(f"the pooled {name}", pooled, expected)
        zeroed = case.rows_view(flat).clone()
        zero_padding(zeroed, sizes, case.tile)
        compare(f"the zeroed {name} rows", rows, zeroed)
    return 3 * mine[0].numel()


def check_empty_tile(module, reference, device):
    """An empty tile's means are zeros: no division by the tile's row count, no NaN."""
    case = Case(
        device=device, dtype="bf16", rows=128, heads=4, width=64, rotated=0, tile=64, block=16,
        norm=False, sizes=[0, 40], row_stride=256,
    )
    buffers = clone(case)
    mine = launch(module._prepare_tiles_kernel, case, buffers)
    sizes = torch.tensor(case.sizes, dtype=torch.int32, device=device)
    for name, flat, pooled in zip(
        ("query", "key", "value"),
        (buffers["query"], buffers["key"], buffers["value"]),
        mine,
    ):
        if bool(torch.isnan(pooled).any()):
            raise SystemExit(f"the pooled {name} holds NaNs for an empty tile")
        if bool(pooled[0].any()):
            raise SystemExit(f"the pooled {name} of an empty tile is not zero")
        compare(f"the pooled {name}", pooled, pool(case.rows_view(flat), sizes, case.tile))
    return 3 * mine[0].numel()


def check_small_valid(module, reference, device, row_stride=1200):
    """A tile whose valid size is smaller than one row block: the masked tail inside a block."""
    case = Case(
        device=device, dtype="f16", rows=64, heads=2, width=128, rotated=96, tile=64, block=16,
        norm=True, sizes=[5], row_stride=row_stride,
    )
    mine_buffers, their_buffers = clone(case), clone(case)
    mine = launch(module._prepare_tiles_kernel, case, mine_buffers)
    theirs = launch(reference, case, their_buffers)
    for name, a, b in zip(("query", "key", "value"), mine, theirs):
        compare(f"the pooled {name} of a tile with 5 valid rows", a, b)
    # The mean divides by the valid size, not by the tile's row count.
    sizes = torch.tensor(case.sizes, dtype=torch.int32, device=device)
    expected = pool(case.rows_view(mine_buffers["query"]), sizes, case.tile)
    compare("the pooled query of a tile with 5 valid rows", mine[0], expected)
    return 3 * mine[0].numel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/vsa_prepare_tiles.py",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--row-stride",
        type=int,
        default=1200,
        help="the query/key/value row stride of the fixture case; the fixture's own is 7168, and "
        "any value wider than heads · width is a stride set that is not the identity",
    )
    options = parser.parse_args()

    device = torch.device(options.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit(
            "this kernel needs a device: its arithmetic is inline PTX (add.rn.f32, mul.rn.f32 and "
            "the packed cvt.rn.bf16x2.f32 forms), which the interpreter cannot execute, so a CPU "
            "run would prove nothing about what the module stores"
        )
    module = load_kernel(options.package)
    reference = reference_kernel()

    checked = 0
    checked += check_fixture_geometry(module, reference, device, options.row_stride)
    checked += check_small_valid(module, reference, device, options.row_stride)
    checked += check_pass_through(module, reference, device)
    checked += check_no_norm(module, reference, device)
    checked += check_empty_tile(module, reference, device)
    print(
        f"ok: {checked} values — the fixture's strided cell and the narrowed rotation against the "
        f"tree's reference kernel, bit for bit, the padding zeros, the pooled means against pool, "
        f"and an empty tile's zeros"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
