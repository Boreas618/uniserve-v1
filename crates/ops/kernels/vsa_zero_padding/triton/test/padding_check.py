#!/usr/bin/env python3
"""Run `vsa_zero_padding` and check it against the operator's masking, on the CPU.

    TRITON_INTERPRET=1 crates/program-server/python/.venv/bin/python padding_check.py [--package DIR]

The operator's whole claim is an absence with a boundary: the rows a tile has past its valid size
read as zeros, and the rows inside it are not touched. Both halves are checked, because a kernel
that zeroed the whole tile would pass any check that only looked at the padding, and one that
zeroed the wrong block would pass any check that only looked at the tile's first rows.

`uniserve/nn/attention/vsa/segments.py::zero_padding` is the reference, copied verbatim: it masks a
`[tiles, tile, heads, width]` view by the same comparison. The cases are the ones a plausible-but
-wrong kernel passes:

- **a tile whose valid size is zero**, so every row blocks' mask is full;
- **a partial block**, where the valid size ends inside one `block_rows` block — the only place the
  mask has both kinds of row in one store;
- **a strided row layout**, row stride wider than `heads · width`, so an address that ignored the
  stride would write inside the tile and a head that ignored the head stride would land on its
  neighbour;
- **several heads**, since the kernel walks them with `static_range` and a mask that is computed
  once and reused;
- **`f32` next to `bf16`**, because the zeros are stored in the values' own element type.
"""

import argparse
import importlib.util
import os
import pathlib
import sys

import torch

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "vsa_zero_padding.py"
    spec = importlib.util.spec_from_file_location("vsa_zero_padding_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(rows, sizes, tile):
    """`uniserve/nn/attention/vsa/segments.py::zero_padding`, on the CPU."""
    live = torch.arange(tile) < sizes[:, None]
    rows.view(sizes.numel(), tile, *rows.shape[1:]).masked_fill_(~live[:, :, None, None], 0)


def run(module, *, dtype, sizes, heads, width, tile, block, row_stride=None, seed=13):
    """One launch, and the reference's masking of the same rows."""
    element = DTYPES[dtype]
    stride_row = heads * width if row_stride is None else row_stride
    tiles = len(sizes)
    rows = tiles * tile
    torch.manual_seed(seed)
    # Non-zero everywhere, so a row the kernel failed to zero cannot pass as already zero, and a
    # row it wrote by mistake cannot pass as untouched.
    flat = (torch.randn(rows * stride_row, dtype=torch.float32) * 2.0 + 3.0).to(element)
    view = torch.as_strided(flat, (rows, heads, width), (stride_row, width, 1))
    before = view.clone()
    table = torch.tensor(sizes, dtype=torch.int32)

    module._zero_tile_padding_kernel[(tiles, tile // block)](
        view, table, stride_row, width, heads=heads, tile_rows=tile, width=width, block_rows=block
    )

    expected = before.clone()
    reference(expected.reshape(rows, heads, width), table, tile)
    if not torch.equal(view.reshape(rows, heads, width), expected):
        difference = (view.reshape(rows, heads, width).to(torch.float64)
                      - expected.to(torch.float64)).abs()
        position = int(difference.reshape(-1).argmax())
        raise SystemExit(
            f"{dtype} sizes {sizes} heads {heads} width {width} stride {stride_row}: "
            f"{int((difference > 0).sum())} of {view.numel()} elements differ; worst at "
            f"{position}: {float(view.reshape(-1)[position])} vs "
            f"{float(expected.reshape(-1)[position])}"
        )
    # The rows inside the valid size must be bit for bit what they were: the reference's masking
    # says so, and the check below states it separately so a failure names the half that broke.
    live = torch.arange(tile) < table[:, None]
    inside = before.reshape(tiles, tile, heads, width)[live]
    after = view.reshape(tiles, tile, heads, width)[live]
    if not torch.equal(inside, after):
        raise SystemExit(f"{dtype} sizes {sizes}: a live row was written")
    return int(view.numel())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/vsa_zero_padding.py",
    )
    options = parser.parse_args()
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise SystemExit("run this with TRITON_INTERPRET=1: the kernel is checked on the CPU")
    module = load_kernel(options.package)

    checked = 0
    # A partial tile and a full one; a zero tile; a partial block; a strided buffer; f32 next to
    # bf16.
    checked += run(module, dtype="bf16", sizes=[64, 5], heads=4, width=64, tile=64, block=16)
    checked += run(module, dtype="bf16", sizes=[0, 40, 64], heads=2, width=64, tile=64, block=16)
    checked += run(module, dtype="f32", sizes=[17], heads=3, width=32, tile=64, block=16)
    checked += run(
        module, dtype="f16", sizes=[33, 64], heads=8, width=128, tile=64, block=16, row_stride=1200
    )
    checked += run(module, dtype="bf16", sizes=[1, 63], heads=1, width=16, tile=64, block=16)
    print(f"ok: {checked} elements, the padding zeroed and every live row bit for bit untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
