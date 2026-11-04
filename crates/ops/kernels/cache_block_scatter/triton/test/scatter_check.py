#!/usr/bin/env python3
"""Run `cache_block_scatter` on Triton's interpreter and check the blocks it writes.

    crates/program-server/python/.venv/bin/python scatter_check.py [--package DIR]

A state bank stores each field block-major: axis 0 selects a block and the trailing axes are the
block's own, **at the field's own strides** — one manifest cell's blocks start 2048 elements apart
while the content of a block covers 1088, and another cell is packed. This program turns a column of
the flattened block into an offset through those axes, so the address arithmetic is what can be wrong,
and it is wrong invisibly: a dense-looking bank is filled with the right *count* of right values in
the wrong places, and every later read is a shifted tensor.

The check therefore builds the field as a **strided view of one buffer**: the gaps between blocks and
inside a block carry a sentinel, and every element the program does not name must still hold it
afterwards. A block index of `-1` (the inactive pair's placeholder) and one past the bank write
nothing, which is what keeps an unvalidated device vector inside its allocation — both are in the
cases — and the snapshot is read-only here, so it must come back unchanged.
"""

import argparse
import importlib.util
import os
import pathlib
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402  (Triton reads TRITON_INTERPRET at import)

print("scatter_check runs Triton's interpreter (TRITON_INTERPRET=1)")

#: The sentinel a bank is primed with, per element type: a byte no small integer produces and a half
#: no value of these cases is, so a program writing where it should not shows wherever it lands.
SENTINELS = {torch.uint8: 0xA5, torch.bfloat16: 1234.5}

#: One manifest cell per case: `shape` is the bank's block-major shape, `strides` its element strides,
#: `width` the elements in one flattened block and `block` the columns one program writes, exactly as
#: the descriptor compiles them.
CASES = [
    dict(what="a packed block bank", dtype=torch.uint8, shape=(2048, 64, 1, 128),
         strides=(8192, 128, 128, 1), width=8192, block=1024, blocks=[3, 0, 7, 1]),
    # Blocks 2048 elements apart whose content covers 1088: the gaps inside a block are the check.
    dict(what="a bank whose blocks hold 2048 elements in 1088", dtype=torch.bfloat16,
         shape=(1024, 16, 2, 64), strides=(2048, 64, 64, 1), width=2048, block=1024,
         blocks=[0, 5, 2]),
    # The placeholders: an inactive pair and an index past the bank write nothing, while the rows
    # between them still land in their own blocks.
    dict(what="a placeholding index", dtype=torch.uint8, shape=(1024, 16, 2, 64),
         # The bank's last block and the first index past it sit either side of the bound, so a
         # bound that is off by one, or short by any number, changes an answer here.
         strides=(2048, 64, 64, 1), width=2048, block=1024,
         blocks=[-1, 4, 1019, 1023, 1024, 9]),
]


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "cache_block_scatter.py"
    spec = importlib.util.spec_from_file_location("scatter_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bank_of(case):
    """The buffer a bank is a strided view of, primed with the sentinel, and that view."""
    extents = [(count - 1) * stride for count, stride in zip(case["shape"], case["strides"])]
    first, rest = extents[0], extents[1:]
    buffer = torch.full((first + sum(rest) + 1,), SENTINELS[case["dtype"]], dtype=case["dtype"])
    field = torch.as_strided(buffer, case["shape"], case["strides"])
    return buffer, field


def snapshot_of(case):
    """The rows to scatter: distinct in every element, so a shifted write cannot pass."""
    rows = len(case["blocks"])
    generator = torch.Generator().manual_seed(7 + rows)
    if case["dtype"] == torch.uint8:
        return torch.randint(0, 256, (rows, case["width"]), generator=generator, dtype=torch.uint8)
    values = torch.randn((rows, case["width"]), generator=generator, dtype=torch.float32)
    return values.to(case["dtype"])


def run(module, case):
    """One launch, and the bank it wrote against the bank it should have written."""
    buffer, field = bank_of(case)
    snapshot = snapshot_of(case)
    snapshot_before = snapshot.clone()
    blocks = torch.tensor(case["blocks"], dtype=torch.int64)

    # The expected bank is built by torch's own strided copy, not by the kernel's address formula:
    # `copy_` walks the destination's axes in their logical order, so the reference does not restate
    # the arithmetic under test.
    expected = buffer.clone()
    written = 0
    for row, block in enumerate(case["blocks"]):
        if 0 <= block < case["shape"][0]:
            block_view = torch.as_strided(
                expected, case["shape"][1:], case["strides"][1:],
                storage_offset=block * case["strides"][0],
            )
            block_view.copy_(snapshot[row].view(case["shape"][1:]))
            written += 1

    tiles = (case["width"] + case["block"] - 1) // case["block"]
    module._scatter_kernel[(len(case["blocks"]), tiles)](
        field, blocks, snapshot, 1, case["shape"], case["strides"], case["width"], case["block"],
    )

    where = (f"{case['what']}: {len(case['blocks'])} rows into blocks {case['blocks']} of a "
             f"{case['shape'][0]}-block bank at strides {case['strides']}")
    if not torch.equal(snapshot, snapshot_before):
        raise SystemExit(f"{where}: the snapshot changed; this program only reads it")
    if not torch.equal(buffer, expected):
        wrong = (buffer != expected).reshape(-1)
        position = int(wrong.nonzero()[0])
        raise SystemExit(
            f"{where}: {int(wrong.sum())} of {buffer.numel()} elements of the bank differ from what "
            f"the rows should have written ({written} of them land inside the bank); first at "
            f"{position}: {int(buffer.reshape(-1)[position])} vs "
            f"{int(expected.reshape(-1)[position])}")
    return written, case["width"] * written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/cache_block_scatter.py",
    )
    options = parser.parse_args()
    module = load_kernel(options.package)
    total = 0
    for case in CASES:
        written, elements = run(module, case)
        total += elements
        print(f"  {case['what']}: {written} rows, {elements} elements written, the rest untouched")
    print(f"ok: {total} elements scattered into selected blocks, in place, with every element no row "
          f"named left as it was")
    return 0


if __name__ == "__main__":
    sys.exit(main())
