#!/usr/bin/env python3
"""Run `vsa_prepare_tiles` on a device and check it against the tree's own reference.

    crates/build-server/python/.venv/bin/python prepare_check.py [--package DIR] [--device cuda]

**This check needs a GPU, and the reason is in the kernel rather than in this script.** The
stepwise normalize-and-rotate recipe stores bits, so the package spells its arithmetic as PTX:
`add.rn.f32`, `mul.rn.f32`, and the packed `cvt.rn.bf16x2.f32`/`fma.rn.f16x2` forms of the narrowed
halves. `tl.inline_asm_elementwise` does not execute under `TRITON_INTERPRET`, and the pooling's own
`_add_rn` uses it as well — so no part of this kernel can run on the build host, not even the
`norm = false` path. On a host without a device the script says exactly that and stops.

The package's kernel is launched as its module is built: `src/build.py`'s `spec` for the case's
descriptor fields gives the kernel, its compile-time values (the epsilon decoded from the key's bits)
and its options (two warps, no FP32 fusion), so the check cannot launch something the build does
not compile. The operands are the operator's: 16-bit Q/K/V rows of the descriptor's type, `[width]`
weights of that type and FP32 `[rows, rotated / 2]` factor tables. The stepwise routine addresses
the rows as words of two 16-bit features, so rows of any other element type are outside its domain:
they are read and written at the wrong addresses, across other programs' heads.

Two independent references, because this is the operator where a stride mix-up cannot be caught by
reading the key — one program reads seven caller strides at once:

- the retired Python tree's operator, `uniserve_kernels.attention.vsa_segments.prepare_tiles`, which
  is the definition of the arithmetic; it is called as its callers called it, so it checks the
  operands and launches its kernel with its own wiring. This repository no longer holds it:
  `UNISERVE_REFERENCE_TREE` names a checkout of the last tree that did;
- eager torch transcriptions of the pooling and zero-padding rules (`pool`, `zero_padding`), for the half
  of the operator the kernel need not be trusted about: the padding zeros and the FP32 mean, which
  must equal torch's own reduction bit for bit.

The cases:
- **the fixture's cell**: 128-feature heads of which 96 rotate, two tiles whose second is partly
  empty, on a **strided** buffer whose row stride is wider than `heads · width` — so both the row and
  the head stride have to be honoured for the loads, the stores and the means;
- **`rotated < width`**, the narrowed rotation: with `cosine = 0, sine = 1` the rotated features
  exchange pairs, and the features **past `rotated`** must come out identical to a run with other
  factors. That is the pass-through mask, and skipping it is what `rope_apply`'s
  narrowed form did wrong;
- **no normalization**: only zeroing and pooling, compared bit for bit against `pool` of the zeroed
  rows rather than against the retired operator;
- **an empty tile**: its means are zeros — not a division by the tile's row count, and not NaNs.
"""

import argparse
import importlib.util
import os
import pathlib
import struct
import sys

import torch

#: The repository root: this check lives in `tests/python/kernels/<family>/`.
ROOT = pathlib.Path(__file__).resolve().parents[4]
#: The kernel packages' root.
KERNELS = ROOT / "crates" / "ops" / "kernels"
#: The package this check covers.
PACKAGE = KERNELS / "vsa_prepare_tiles" / "triton"
#: The Build Server's Python package, which every compile script imports its driver from.
BUILD_SERVER = ROOT / "crates" / "build-server" / "python" / "src"

#: `PrepareTilesDescriptor`'s epsilon, `0x3727c5ac` — the fixture's cell. A key prints a float's
#: bits; the descriptor means the float they encode.
EPSILON = 0x3727C5AC
#: The architecture the compile script is asked about. Its `spec` states the same kernel,
#: compile-time values and options for every target.
ARCHITECTURE = "sm_100a"
#: The descriptor's names of the row types.
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16}


def epsilon():
    """The float `EPSILON`'s bits encode, which the retired operator took as `eps`."""
    return struct.unpack("<f", struct.pack("<I", EPSILON))[0]


def reference_operator():
    """The retired tree's `prepare_tiles`, which defines the arithmetic this package computes."""
    tree = os.environ.get("UNISERVE_REFERENCE_TREE")
    if not tree:
        sys.exit(
            "prepare_check: UNISERVE_REFERENCE_TREE names no checkout of the retired tree"
        )
    source = pathlib.Path(tree) / "uniserve_kernels" / "src"
    if not source.is_dir():
        sys.exit(
            f"prepare_check: {source} holds no reference kernels; set UNISERVE_REFERENCE_TREE"
        )
    sys.path.insert(0, str(source))
    from uniserve_kernels.attention.vsa_segments import prepare_tiles

    return prepare_tiles


def zero_padding(rows, variable_block_sizes, tile):
    """Zero every row at or past its tile's valid size, on the device as on the host."""
    live = (
        torch.arange(tile, device=rows.device) < variable_block_sizes[:, None]
    )
    rows.view(variable_block_sizes.numel(), tile, *rows.shape[1:]).masked_fill_(
        ~live[:, :, None, None], 0
    )


def pool(rows, variable_block_sizes, tile):
    """The FP32 sum of a tile's rows, divided by its valid size (at least one)."""
    values = rows.view(variable_block_sizes.numel(), tile, *rows.shape[1:])
    total = values.sum(1, dtype=torch.float32)
    return total / variable_block_sizes.clamp_min(1).view(-1, 1, 1)


def load_build(package):
    """The package's compile script, `src/build.py`, imported with the kernel source it imports."""
    for path in (BUILD_SERVER, package / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(
        "vsa_prepare_tiles_build", package / "src" / "build.py"
    )
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


class Case:
    """One launch's operands and the descriptor fields that describe them."""

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
        self.dtype = dtype
        self.element = DTYPES[dtype]
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
        """Hold `[rows, heads, width]` rows of the case's type in a buffer whose row stride the case sets."""
        flat = torch.zeros(
            self.rows * self.row_stride, dtype=self.element, device=self.device
        )
        self.rows_view(flat).copy_(features)
        return flat

    def rows_view(self, flat, count=None):
        count = self.rows if count is None else count
        return torch.as_strided(
            flat,
            (count, self.heads, self.width),
            (self.row_stride, self.head_stride, 1),
        )

    def weight(self):
        """A contiguous `[width]` normalization weight of the rows' type, as the operator takes it."""
        drawn = torch.randn(self.width, dtype=torch.float32) * 0.25
        return drawn.to(self.element).to(self.device)

    @property
    def factor_stride(self):
        return self.row_stride if self.rotated else 0

    def factor(self, kind):
        """An FP32 factor table, `[rows, rotated / 2]` at the case's factor stride."""
        stride = max(self.factor_stride, 1)
        flat = torch.zeros(
            self.rows * stride, dtype=torch.float32, device=self.device
        )
        if not self.rotated:
            return flat
        table = torch.as_strided(
            flat, (self.rows, self.rotated // 2), (stride, 1)
        )
        if kind == "cosine":
            table.copy_(
                torch.cos(
                    torch.linspace(
                        0.0, 1.0, self.rotated // 2, device=self.device
                    )
                )
            )
        elif kind == "sine":
            table.copy_(
                torch.sin(
                    torch.linspace(
                        0.0, 1.0, self.rotated // 2, device=self.device
                    )
                )
            )
        elif kind == "one":
            table.fill_(1.0)
        return flat

    def factor_view(self, flat):
        stride = max(self.factor_stride, 1)
        return torch.as_strided(
            flat, (self.rows, max(self.rotated // 2, 1)), (stride, 1)
        )

    def fields(self):
        """The descriptor's fields, spelled as `PrepareTilesDescriptor`'s key prints them."""
        return {
            "dtype": self.dtype,
            "heads": str(self.heads),
            "width": str(self.width),
            "rotated": str(self.rotated),
            "tile_rows": str(self.tile),
            "block_rows": str(self.block),
            "epsilon": f"{EPSILON:#x}",
            "norm": "true" if self.norm else "false",
            "query_row_stride": str(self.row_stride),
            "query_head_stride": str(self.head_stride),
            "key_row_stride": str(self.row_stride),
            "key_head_stride": str(self.head_stride),
            "value_row_stride": str(self.row_stride),
            "value_head_stride": str(self.head_stride),
            "factor_stride": str(self.factor_stride),
        }

    def outputs(self):
        """The valid-size table and three NaN-filled FP32 `[tiles, heads, width]` means."""
        sizes = torch.tensor(self.sizes, dtype=torch.int32, device=self.device)
        pooled = [
            torch.full(
                (self.tiles, self.heads, self.width),
                float("nan"),
                dtype=torch.float32,
                device=self.device,
            )
            for _ in range(3)
        ]
        return sizes, pooled


def launch(build, case, buffers):
    """One launch of the package's module for the case's descriptor; returns its pooled tensors.

    The kernel, its compile-time values and its options are what `build.spec` states for the
    case's fields (the manifest declares no `[parameters]`). The operands are packed as
    `VsaPrepareTiles::launch` packs them: without normalization the rows stand in for the weights
    and factor tables, which the module then does not read.
    """
    stated = build.spec(case.fields(), {}, ARCHITECTURE)
    sizes, pooled = case.outputs()
    query, key, value = (
        case.rows_view(buffers[name]) for name in ("query", "key", "value")
    )
    stated["kernel"][(case.tiles, case.heads)](
        query,
        key,
        value,
        buffers["query_weight"] if case.norm else query,
        buffers["key_weight"] if case.norm else key,
        case.factor_view(buffers["cosine"]) if case.norm else query,
        case.factor_view(buffers["sine"]) if case.norm else query,
        sizes,
        *pooled,
        **stated["constexprs"],
        **stated["options"],
    )
    return pooled


def launch_reference(prepare_tiles, case, buffers):
    """The retired operator over the case's buffers; returns its pooled tensors."""
    sizes, pooled = case.outputs()
    normalization = (
        {
            "query_weight": buffers["query_weight"],
            "key_weight": buffers["key_weight"],
            "cosine": case.factor_view(buffers["cosine"]),
            "sine": case.factor_view(buffers["sine"]),
            "eps": epsilon(),
        }
        if case.norm
        else {}
    )
    prepare_tiles(
        case.rows_view(buffers["query"]),
        case.rows_view(buffers["key"]),
        case.rows_view(buffers["value"]),
        sizes,
        case.tile,
        *pooled,
        **normalization,
    )
    return pooled


def clone(case):
    return {
        name: getattr(case, name).clone()
        for name in (
            "query",
            "key",
            "value",
            "query_weight",
            "key_weight",
            "cosine",
            "sine",
        )
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


def check_fixture_geometry(build, reference, device, row_stride=1200):
    """The fixture's cell against the retired operator, on a strided buffer.

    `row_stride` must be wider than `heads · width` for the case to be about strides at all: a tile
    walk that assumed contiguity passes every identity case.
    """
    if row_stride <= 8 * 128:
        raise SystemExit(
            f"a row stride of {row_stride} is not wider than heads · width (1024), so this case "
            f"would not exercise the strides the descriptor carries"
        )
    case = Case(
        device=device,
        dtype="f16",
        rows=128,
        heads=8,
        width=128,
        rotated=96,
        tile=64,
        block=16,
        norm=True,
        sizes=[64, 37],
        row_stride=row_stride,
    )
    mine_buffers, their_buffers = clone(case), clone(case)
    mine = launch(build, case, mine_buffers)
    theirs = launch_reference(reference, case, their_buffers)
    for name, a, b in zip(("query", "key", "value"), mine, theirs):
        compare(f"the pooled {name}", a, b)
    for name in ("query", "key", "value"):
        compare(
            f"the prepared {name} rows", mine_buffers[name], their_buffers[name]
        )
    # The rows past each tile's valid size are zeros, in all three row tensors.
    sizes = torch.tensor(case.sizes, dtype=torch.int32, device=device)
    for name in ("query", "key", "value"):
        rows = case.rows_view(mine_buffers[name])
        live = torch.arange(case.tile, device=device) < sizes[:, None]
        padding = rows.view(case.tiles, case.tile, case.heads, case.width)[
            ~live
        ]
        if bool(padding.any()):
            raise SystemExit(
                f"the prepared {name} rows hold padding instead of zeros"
            )
    return 3 * mine[0].numel() + 3 * case.rows * case.heads * case.width


def check_pass_through(build, reference, device):
    """With `rotated < width`, the features past the rotation do not move."""

    def run(cosines):
        case = Case(
            device=device,
            dtype="f16",
            rows=64,
            heads=4,
            width=64,
            rotated=32,
            tile=64,
            block=16,
            norm=True,
            sizes=[64],
            row_stride=320,
            cosines=cosines,
        )
        buffers = clone(case)
        pooled = launch(build, case, buffers)
        theirs = launch_reference(reference, case, clone(case))
        compare(
            f"the pooled query with cosines={cosines}", pooled[0], theirs[0]
        )
        return case.rows_view(buffers["query"]).clone()

    turned = run(True)
    # `cosine = 0, sine = 1` exchanges each rotated pair; the pass-through features stay put.
    exchanged = torch.zeros_like(turned)
    case = Case(
        device=device,
        dtype="f16",
        rows=64,
        heads=4,
        width=64,
        rotated=32,
        tile=64,
        block=16,
        norm=True,
        sizes=[64],
        row_stride=320,
        cosines=False,
    )
    buffers = clone(case)
    launch(build, case, buffers)
    exchanged.copy_(case.rows_view(buffers["query"]))
    rotated = case.rotated
    compare(
        "the pass-through features (past `rotated`) of a narrowed rotation",
        exchanged[..., rotated:],
        turned[..., rotated:],
    )
    if torch.equal(exchanged[..., :rotated], turned[..., :rotated]):
        raise SystemExit(
            "the rotation moved nothing, so the case does not test its coverage"
        )
    return int(exchanged.numel())


def check_no_norm(build, reference, device):
    """Without normalization: the padding and the mean, against `pool` itself."""
    case = Case(
        device=device,
        dtype="bf16",
        rows=128,
        heads=4,
        width=64,
        rotated=0,
        tile=64,
        block=16,
        norm=False,
        sizes=[64, 31],
        row_stride=256,
    )
    buffers = clone(case)
    mine = launch(build, case, buffers)
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


def check_empty_tile(build, reference, device):
    """An empty tile's means are zeros: no division by the tile's row count, no NaN."""
    case = Case(
        device=device,
        dtype="bf16",
        rows=128,
        heads=4,
        width=64,
        rotated=0,
        tile=64,
        block=16,
        norm=False,
        sizes=[0, 40],
        row_stride=256,
    )
    buffers = clone(case)
    mine = launch(build, case, buffers)
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
        compare(
            f"the pooled {name}",
            pooled,
            pool(case.rows_view(flat), sizes, case.tile),
        )
    return 3 * mine[0].numel()


def check_small_valid(build, reference, device, row_stride=1200):
    """A tile whose valid size is smaller than one row block: the masked tail inside a block."""
    case = Case(
        device=device,
        dtype="f16",
        rows=64,
        heads=2,
        width=128,
        rotated=96,
        tile=64,
        block=16,
        norm=True,
        sizes=[5],
        row_stride=row_stride,
    )
    mine_buffers, their_buffers = clone(case), clone(case)
    mine = launch(build, case, mine_buffers)
    theirs = launch_reference(reference, case, their_buffers)
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
        default=PACKAGE,
        help="the package directory holding src/build.py and the kernel source it imports",
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
    build = load_build(options.package)
    reference = reference_operator()

    checked = 0
    checked += check_fixture_geometry(
        build, reference, device, options.row_stride
    )
    checked += check_small_valid(build, reference, device, options.row_stride)
    checked += check_pass_through(build, reference, device)
    checked += check_no_norm(build, reference, device)
    checked += check_empty_tile(build, reference, device)
    print(
        f"ok: {checked} values — the fixture's strided cell and the narrowed rotation against the "
        f"retired operator, bit for bit, the padding zeros, the pooled means against pool, "
        f"and an empty tile's zeros"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
