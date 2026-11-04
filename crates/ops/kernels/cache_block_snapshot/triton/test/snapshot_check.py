#!/usr/bin/env python3
"""Run `cache_block_snapshot` on Triton's interpreter and check the rows it reads out.

    crates/program-server/python/.venv/bin/python snapshot_check.py [--package DIR]

The first half of the prefix cache's block copy: ``snapshot[row, :] = field[blocks[row], :]``, so the
second half can write the same blocks elsewhere without a host read between the two launches. It reads
a block-major bank at the field's own strides, which is the arithmetic that can be wrong without a
build noticing — a bank whose blocks are 2048 elements apart while a block's content covers 1088 comes
back with a shifted row and no error.

So the check fills the bank with distinct values at its own strides and the gaps with a sentinel, and
requires every row of the snapshot to be the block the index named, element for element. A block index
of `-1` (the inactive pair's placeholder) and one past the bank copy **nothing**: those rows keep
what the caller left in them, which is the property that keeps an unvalidated device vector inside its
allocation. The bank is read-only here, so it must come back unchanged.
"""

import argparse
import importlib.util
import os
import pathlib
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402  (Triton reads TRITON_INTERPRET at import)

print("snapshot_check runs Triton's interpreter (TRITON_INTERPRET=1)")

#: The sentinel a snapshot is primed with: a row that should not be written keeps a value no block of
#: these cases holds.
SENTINELS = {torch.uint8: 0xA5, torch.bfloat16: 1234.5}

#: One manifest cell per case: `shape` is the bank's block-major shape, `strides` its element strides,
#: `width` the elements in one flattened block and `block` the columns one program reads, exactly as
#: the descriptor compiles them.
CASES = [
    dict(what="a packed block bank", dtype=torch.uint8, shape=(2048, 64, 1, 128),
         strides=(8192, 128, 128, 1), width=8192, block=1024, blocks=[3, 0, 7, 1]),
    # Blocks 2048 elements apart whose content covers 1088: the gaps inside a block are the check.
    dict(what="a bank whose blocks hold 2048 elements in 1088", dtype=torch.bfloat16,
         shape=(1024, 16, 2, 64), strides=(2048, 64, 64, 1), width=2048, block=1024,
         blocks=[0, 5, 2]),
    # The placeholders: an inactive pair and an index past the bank copy nothing, while the rows
    # between them still come back with their own blocks.
    dict(what="a placeholding index", dtype=torch.uint8, shape=(1024, 16, 2, 64),
         # The bank's last block and the first index past it sit either side of the bound, so a
         # bound that is off by one, or short by any number, changes an answer here.
         strides=(2048, 64, 64, 1), width=2048, block=1024,
         blocks=[-1, 4, 1019, 1023, 1024, 9]),
]


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "cache_block_snapshot.py"
    spec = importlib.util.spec_from_file_location("snapshot_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bank_of(case):
    """The bank as a strided view of one buffer, distinct in every element."""
    extents = [(count - 1) * stride for count, stride in zip(case["shape"], case["strides"])]
    buffer = torch.zeros(extents[0] + sum(extents[1:]) + 1, dtype=case["dtype"])
    field = torch.as_strided(buffer, case["shape"], case["strides"])
    if case["dtype"] == torch.uint8:
        # Every element its own value, so a row read from the wrong offset cannot pass by chance.
        field.copy_(torch.arange(field.numel(), dtype=torch.int64).remainder(256).to(torch.uint8)
                    .view(case["shape"]))
    else:
        field.copy_(torch.arange(field.numel(), dtype=torch.float32).remainder(997).to(torch.bfloat16)
                    .view(case["shape"]))
    return buffer, field


def run(module, case):
    """One launch, and the snapshot it filled against the rows it should have read."""
    buffer, field = bank_of(case)
    buffer_before = buffer.clone()
    rows = len(case["blocks"])
    snapshot = torch.full((rows, case["width"]), SENTINELS[case["dtype"]], dtype=case["dtype"])
    snapshot_before = snapshot.clone()
    blocks = torch.tensor(case["blocks"], dtype=torch.int64)

    expected = snapshot.clone()
    read = 0
    for row, block in enumerate(case["blocks"]):
        if 0 <= block < case["shape"][0]:
            # torch's own strided read, not the kernel's address formula.
            block_view = torch.as_strided(
                buffer, case["shape"][1:], case["strides"][1:],
                storage_offset=block * case["strides"][0],
            )
            expected[row] = block_view.reshape(-1)
            read += 1

    tiles = (case["width"] + case["block"] - 1) // case["block"]
    module._snapshot_kernel[(rows, tiles)](
        field, blocks, snapshot, 1, case["shape"], case["strides"], case["width"], case["block"],
    )

    where = (f"{case['what']}: {rows} rows from blocks {case['blocks']} of a "
             f"{case['shape'][0]}-block bank at strides {case['strides']}")
    if not torch.equal(buffer, buffer_before):
        raise SystemExit(f"{where}: the bank changed; this program only reads it")
    if not torch.equal(snapshot, expected):
        wrong = (snapshot != expected).reshape(-1)
        position = int(wrong.nonzero()[0])
        raise SystemExit(
            f"{where}: {int(wrong.sum())} of {snapshot.numel()} elements of the snapshot are not the "
            f"bank's ({read} of {rows} rows name a block inside it, and the rest must be left as "
            f"they were); first at {position}: {float(snapshot.reshape(-1)[position])!r} vs "
            f"{float(expected.reshape(-1)[position])!r}")
    return read, case["width"] * read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/cache_block_snapshot.py",
    )
    options = parser.parse_args()
    module = load_kernel(options.package)
    total = 0
    for case in CASES:
        read, elements = run(module, case)
        total += elements
        print(f"  {case['what']}: {read} rows read, {elements} elements, the rest left alone")
    print(f"ok: {total} elements copied out of selected blocks, every row the block its index named, "
          f"and every row whose index names no block left as it was")
    return 0


if __name__ == "__main__":
    sys.exit(main())
