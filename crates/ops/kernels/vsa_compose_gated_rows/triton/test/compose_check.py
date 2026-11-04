#!/usr/bin/env python3
"""Run `vsa_compose_gated_rows` on a device and check it bit for bit against the operator's steps.

    crates/program-server/python/.venv/bin/python compose_check.py [--package DIR] [--device cuda]

**This check needs a GPU**: the kernel pins its rounding with inline PTX (`mul.rn.f32`,
`add.rn.f32`), which exists so the compiler cannot contract the multiply into the sum, and
`tl.inline_asm_elementwise` does not execute under `TRITON_INTERPRET`. Without a device the script
says that and stops.

The operator writes one interval of every member's rows into a payload, member-major:

```text
row  = member · owner_rows + start + i          the member's own row numbering
tile = row / tile_rows                          the compression that row reads
out[slots[member], i] = narrow(fine[row] + narrow(compressed[tile] · gate[row]))
```

The two narrowings are the operator's, and the comparison is **bit for bit**. The cases:
- **a permuted slot table**, `slots = (2, 0, 1)`: a kernel that wrote members in order would pass
  any case with the identity and fail this one;
- **an interval that is not the whole member** (`start > 0`, `count < owner_rows`), so the row base
  and the tile division are exercised at an offset rather than at zero;
- **a row block that crosses no tile but sits inside one** (`block_rows` 16 of `tile_rows` 64), the
  operator's stated invariant;
- **strided `fine` and `gate`** whose row stride is wider than `heads · width`, and a **padded
  compression** whose tile stride is longer than `width`;
- **a contraction trap**: data whose product is not representable in the storage dtype, so the
  narrowed product and a fused multiply-add give different answers — the check asserts the case can
  tell them apart;
- **bf16, f16 and f32**.
"""

import argparse
import importlib.util
import pathlib
import sys

import torch

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "vsa_compose_gated_rows.py"
    spec = importlib.util.spec_from_file_location("vsa_compose_gated_rows_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(output, fine, compressed, gate, slots, owner_rows, start, count, tile, element):
    """The operator's text, walked per member and per row."""
    heads, width = fine.shape[1], fine.shape[2]
    for member, slot in enumerate(slots):
        for index in range(count):
            row = member * owner_rows + start + index
            tile_index = row // tile
            product = (compressed[:, tile_index, :].to(torch.float32) * gate[row].to(torch.float32))
            product = product.to(element).to(torch.float32)
            total = fine[row].to(torch.float32) + product
            output[slot, index] = total.to(element)
    return output


def contracted(fine, compressed, gate, slots, owner_rows, start, count, tile, element):
    """The same composition with the products left in FP32: what a fused compiler would produce."""
    shape = (len(slots), count, fine.shape[1], fine.shape[2])
    output = torch.full(shape, float("nan"), dtype=element, device=fine.device)
    for member, slot in enumerate(slots):
        for index in range(count):
            row = member * owner_rows + start + index
            tile_index = row // tile
            product = compressed[:, tile_index, :].to(torch.float32) * gate[row].to(torch.float32)
            output[slot, index] = (fine[row].to(torch.float32) + product).to(element)
    return output


def run(module, *, dtype, members, owner_rows, start, count, heads, width, tile, block,
        slots=None, fine_stride=None, tile_stride=None, contraction_trap=False, seed=37):
    """One launch, the operator's reference, and the contracted variant for contrast."""
    element = DTYPES[dtype]
    device = torch.device("cuda")
    slots = tuple(range(members)) if slots is None else slots
    fine_rows = members * owner_rows
    fine_stride = heads * width if fine_stride is None else fine_stride
    tile_stride = width if tile_stride is None else tile_stride
    tiles = fine_rows // tile
    torch.manual_seed(seed)

    if contraction_trap:
        fine = torch.full((fine_rows, heads, width), 0.5, dtype=torch.float32, device=device)
        gate = torch.full(
            (fine_rows, heads, width), 1.0 + 2.0**-6, dtype=torch.float32, device=device
        )
        compressed = torch.full(
            (heads, tiles, width), 1.0 + 2.0**-6, dtype=torch.float32, device=device
        )
    else:
        fine = torch.randn(fine_rows, heads, width, dtype=torch.float32, device=device) * 0.25
        gate = torch.randn(fine_rows, heads, width, dtype=torch.float32, device=device) * 0.5
        compressed = torch.randn(heads, tiles, width, dtype=torch.float32, device=device) * 0.5

    fine, gate, compressed = fine.to(element), gate.to(element), compressed.to(element)

    def strided(values, row_stride):
        flat = torch.zeros(values.shape[0] * row_stride, dtype=element, device=device)
        view = torch.as_strided(flat, values.shape, (row_stride, width, 1))
        view.copy_(values)
        return flat, view

    fine_flat, fine_view = strided(fine, fine_stride)
    gate_flat, gate_view = strided(gate, fine_stride)
    compressed_flat = torch.zeros(heads * tiles * tile_stride, dtype=element, device=device)
    compressed_view = torch.as_strided(
        compressed_flat.view(heads, tiles, tile_stride),
        (heads, tiles, width),
        (tiles * tile_stride, tile_stride, 1),
    )
    compressed_view.copy_(compressed)

    output = torch.full((members, count, heads, width), float("nan"), dtype=element, device=device)
    module._compose_gated_rows_kernel[(count // block, heads, members)](
        output,
        fine_view,
        compressed_view,
        gate_view,
        fine_stride,
        width,
        fine_stride,
        width,
        tiles * tile_stride,
        tile_stride,
        owner_rows,
        start,
        count,
        heads=heads,
        tile_rows=tile,
        width=width,
        block_rows=block,
        slots=slots,
    )

    expected = torch.full_like(output, float("nan"))
    reference(expected, fine, compressed, gate, slots, owner_rows, start, count, tile, element)
    if contraction_trap:
        differing = contracted(fine, compressed, gate, slots, owner_rows, start, count, tile, element)
        if torch.equal(expected, differing):
            raise SystemExit(
                "the contraction case cannot tell a fused multiply-add from the operator's two "
                "roundings, so it would pass either way"
            )
    if not torch.equal(output, expected):
        difference = (output.to(torch.float64) - expected.to(torch.float64)).abs()
        position = int(difference.reshape(-1).argmax())
        raise SystemExit(
            f"{dtype} members {members} owner rows {owner_rows} interval {start}..{start + count} "
            f"heads {heads} width {width} tile {tile} block {block} slots {slots} "
            f"fine stride {fine_stride} tile stride {tile_stride}: "
            f"{int((difference > 0).sum())} of {output.numel()} elements differ; worst at "
            f"{position}: {float(output.reshape(-1)[position])} vs "
            f"{float(expected.reshape(-1)[position])}"
        )
    return int(output.numel())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/vsa_compose_gated_rows.py",
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
    checked += run(
        module, dtype="bf16", members=3, owner_rows=128, start=32, count=64, heads=4, width=64,
        tile=64, block=16, slots=(2, 0, 1),
    )
    checked += run(
        module, dtype="bf16", members=2, owner_rows=128, start=0, count=128, heads=2, width=64,
        tile=64, block=16, contraction_trap=True, fine_stride=192,
    )
    checked += run(
        module, dtype="f16", members=2, owner_rows=64, start=16, count=48, heads=3, width=128,
        tile=64, block=16, fine_stride=768, tile_stride=192,
    )
    checked += run(
        module, dtype="f32", members=1, owner_rows=64, start=0, count=64, heads=2, width=32,
        tile=64, block=32,
    )
    print(
        f"ok: {checked} elements, bit for bit against the operator's narrowed product and sum, with "
        f"a permuted slot table, an interval inside the member, strided rows and tiles, and a case a "
        f"fused multiply-add would fail"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
