#!/usr/bin/env python3
"""Run `vsa_add_gated_tiles` on a device and check it bit for bit against the operator's two steps.

    crates/program-server/python/.venv/bin/python gated_check.py [--package DIR] [--device cuda]

**This check needs a GPU**: the kernel pins its rounding with inline PTX (`mul.rn.f32`, `add.rn.f32`)
so that the compiler cannot contract the multiply with the sum, and `tl.inline_asm_elementwise` does
not execute under `TRITON_INTERPRET`. On a host without a device the script says that and stops.

The operator adds one tile's gated compression to its rows as **two** eager steps:

```text
row += narrow(compressed[tile] · gate[row])       the product rounded to the storage dtype
```

and declares the result within two units of that dtype's rounding. The reference below is those two
steps in torch, and the comparison is **bit for bit**: with the product rounded to the storage dtype
before the sum, there is nothing left for a bound to cover.

The cases, and the one that matters most:
- **a contraction trap**: values whose product is not representable in the storage dtype, chosen so
  that contracting the multiply into the sum (what `enable_fp_fusion` would do, and what the inline
  assembly exists to prevent) gives a different answer. The check asserts the case can tell the two
  apart, so a passing run is evidence about this kernel rather than about the data;
- **a strided output and gate** whose row stride is wider than `heads · width`, and a strided
  compression whose tile stride is longer than `width`;
- **`block_rows` narrower than `tile_rows`**, with rows in the second and third blocks of a tile, so
  the block-to-tile division is exercised rather than the first block alone;
- **bf16, f16 and f32**, since the product's narrowing is the dtype's.
"""

import argparse
import importlib.util
import pathlib
import sys

import torch

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "vsa_add_gated_tiles.py"
    spec = importlib.util.spec_from_file_location("vsa_add_gated_tiles_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def per_row(compressed, rows, tile):
    """Each row's tile compression: `[heads, tiles, width]` read at `row // tile`."""
    tiles = compressed.shape[1]
    index = (torch.arange(rows) // tile).clamp(max=tiles - 1)
    return compressed[:, index, :].permute(1, 0, 2)


def reference(tile_values, before, gate, element):
    """The operator's two steps: the product narrowed, then the sum, then the store."""
    product = (tile_values.to(torch.float32) * gate.to(torch.float32)).to(element).to(
        torch.float32
    )
    return (before.to(torch.float32) + product).to(element)


def contracted(tile_values, before, gate, element):
    """The same sum with the multiply left in FP32: what a fused compiler would produce."""
    return (before.to(torch.float32) + tile_values.to(torch.float32) * gate.to(torch.float32)).to(
        element
    )


def run(module, *, dtype, rows, heads, width, tile, block, row_stride=None, tile_stride=None,
        contraction_trap=False, seed=31):
    """One launch, the operator's reference, and the contracted variant for contrast."""
    element = DTYPES[dtype]
    device = torch.device("cuda")
    stride_row = heads * width if row_stride is None else row_stride
    stride_tile = width if tile_stride is None else tile_stride
    tiles = rows // tile
    torch.manual_seed(seed)

    if contraction_trap:
        # Values one BF16/F16 ulp apart in their product's tail: 1 + 2**-6 times 1 + 2**-6 is
        # representable in FP32 and not in 16 bits, so the narrowing decides the answer.
        before = torch.full((rows, heads, width), 0.5, dtype=torch.float32, device=device)
        gate = torch.full((rows, heads, width), 1.0 + 2.0**-6, dtype=torch.float32, device=device)
        compressed = torch.full(
            (heads, tiles, width), 1.0 + 2.0**-6, dtype=torch.float32, device=device
        )
    else:
        before = torch.randn(rows, heads, width, dtype=torch.float32, device=device) * 0.25
        gate = torch.randn(rows, heads, width, dtype=torch.float32, device=device) * 0.5
        compressed = torch.randn(heads, tiles, width, dtype=torch.float32, device=device) * 0.5

    before = before.to(element)
    gate = gate.to(element)
    compressed = compressed.to(element)

    output_flat = torch.zeros(rows * stride_row, dtype=element, device=device)
    output = torch.as_strided(output_flat, (rows, heads, width), (stride_row, width, 1))
    output.copy_(before)
    gate_flat = torch.zeros(rows * stride_row, dtype=element, device=device)
    gate_view = torch.as_strided(gate_flat, (rows, heads, width), (stride_row, width, 1))
    gate_view.copy_(gate)
    # The compression is `[heads, tiles, width]` whose tile axis has the descriptor's stride: a tile
    # stride of `width` is the contiguous case, anything longer pads every tile.
    compressed_flat = torch.zeros(heads * tiles * stride_tile, dtype=element, device=device)
    compressed_view = torch.as_strided(
        compressed_flat.view(heads, tiles, stride_tile),
        (heads, tiles, width),
        (tiles * stride_tile, stride_tile, 1),
    )
    compressed_view.copy_(compressed)

    module._add_gated_tiles_kernel[(rows // block, heads, 1)](
        output,
        compressed_view,
        gate_view,
        stride_row,
        width,
        stride_row,
        width,
        tiles * stride_tile,
        stride_tile,
        tile_rows=tile,
        width=width,
        block_rows=block,
    )

    rows_of = per_row(compressed, rows, tile)
    expected = reference(rows_of, before, gate, element)
    differing = contracted(rows_of, before, gate, element)
    if contraction_trap and torch.equal(expected, differing):
        raise SystemExit(
            "the contraction case cannot tell a fused multiply-add from the operator's two "
            "roundings, so it would pass either way"
        )
    got = torch.as_strided(output_flat, (rows, heads, width), (stride_row, width, 1))
    if not torch.equal(got, expected):
        difference = (got.to(torch.float64) - expected.to(torch.float64)).abs()
        position = int(difference.reshape(-1).argmax())
        raise SystemExit(
            f"{dtype} rows {rows} heads {heads} width {width} tile {tile} block {block} "
            f"stride {stride_row} tile stride {stride_tile}: "
            f"{int((difference > 0).sum())} of {got.numel()} elements differ; worst at "
            f"{position}: {float(got.reshape(-1)[position])} vs "
            f"{float(expected.reshape(-1)[position])}"
        )
    return int(got.numel())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/vsa_add_gated_tiles.py",
    )
    parser.add_argument("--device", default="cuda")
    options = parser.parse_args()
    if options.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit(
            "this kernel needs a device: it pins the product's and the sum's rounding with inline "
            "PTX, which the interpreter cannot execute, so a CPU run would prove nothing"
        )

    module = load_kernel(options.package)
    checked = 0
    checked += run(module, dtype="bf16", rows=128, heads=4, width=64, tile=64, block=16)
    checked += run(
        module, dtype="bf16", rows=128, heads=4, width=64, tile=64, block=16, contraction_trap=True
    )
    checked += run(
        module, dtype="f16", rows=128, heads=3, width=128, tile=64, block=16, row_stride=640,
        tile_stride=160,
    )
    checked += run(
        module, dtype="bf16", rows=64, heads=2, width=32, tile=64, block=16, contraction_trap=True,
        row_stride=96,
    )
    checked += run(module, dtype="f32", rows=64, heads=2, width=32, tile=64, block=32)
    print(
        f"ok: {checked} elements, bit for bit against the operator's narrowed product and its sum, "
        f"on strided buffers and in all three storage types, with a case that a fused multiply-add "
        f"would fail"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
