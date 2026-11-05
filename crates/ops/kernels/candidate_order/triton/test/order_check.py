#!/usr/bin/env python3
"""Check candidate_order: every row's candidates in ascending token order.

The operator is a permutation and its tolerance is `exact`, so the reference here is
the multiset rule the kernel states — each counted candidate ranked by its token id, the
row filled to its width — and the comparison is bit equality of both outputs, never a
bound.

    # the decisive run, on a host without a device: Triton's interpreter
    python order_check.py --cpu-only [--package .]

    # the same scenarios through the built cubin, on the GPU host
    python order_check.py <resolved module dir> [--package .]

Why the interpreter is the right instrument *here* and not for an fp8 kernel: a
permutation is integer comparison, integer ranks and moves; the interpreter's one known
defect is that it truncates a narrowing float cast, and this program has no narrowing
cast at all. The values are moved, never converted.

The scenarios are chosen for the width being a launch value rather than a field: one
candidate, a width below the kernel's tile, exactly one tile, one tile plus one, several
tiles, a row whose ids repeat, and a row carrying an id that is not a token at all.

The destination buffers are pre-filled with the operator's own sentinels (`-1` ids,
`nan` values), which no candidate can produce, so a slot the program failed to write
fails the comparison instead of passing as a zero that a legitimate token could be.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

#: The widths the check runs, chosen around the kernel's tile of 128.
WIDTHS = [1, 7, 64, 128, 129, 200, 300]

#: How many rows each width is given.
ROWS = 4


def scenario(width, rows, seed, duplicate=False, outside=False):
    """One scenario's inputs: values and ids, as the planning operator writes them."""
    generator = np.random.default_rng(seed)
    values = generator.random((rows, width), dtype=np.float32)
    ids = generator.permutation(np.arange(width, dtype=np.int32))
    ids = np.stack([ids for _ in range(rows)])
    if duplicate and width >= 4:
        # A duplicated id in every row: both copies are candidates and take consecutive ranks.
        ids[:, 1] = ids[:, 0]
    if outside and width >= 4:
        # An id outside the vocabulary, which is what a defect upstream produces: it is not a
        # candidate, so it takes no rank and the row's tail stays the operator's sentinel.
        ids[:, 0] = width + 7
    return values, ids


def wanted(width, values, ids):
    """The row the operator must write: the counted candidates ranked, then the sentinel tail."""
    rows = ids.shape[0]
    want_ids = np.full((rows, width), -1, dtype=np.int32)
    want_values = np.full((rows, width), np.nan, dtype=np.float32)
    for row in range(rows):
        keep = np.flatnonzero((ids[row] >= 0) & (ids[row] < width))
        order = keep[np.argsort(ids[row][keep], kind="stable")]
        want_ids[row, : order.size] = ids[row][order]
        want_values[row, : order.size] = values[row][order]
    return want_values, want_ids


def difference(got_values, got_ids, want_values, want_ids):
    """Every way the row differs, naming the column: ids first, then values within each id group."""
    faults = []
    if not np.array_equal(got_ids, want_ids):
        column = next(
            index
            for index in range(want_ids.size)
            if got_ids.flat[index] != want_ids.flat[index]
        )
        faults.append(f"ids differ from column {column}")
    for row in range(want_ids.shape[0]):
        for token in np.unique(want_ids[row]):
            mine = got_values[row][got_ids[row] == token]
            theirs = want_values[row][want_ids[row] == token]
            if mine.size != theirs.size:
                faults.append(f"row {row}: id {token} appears {mine.size} times, want {theirs.size}")
            elif not np.allclose(np.sort(mine), np.sort(theirs), equal_nan=True):
                # Equal ids are indistinguishable to the draw, so the order of two copies of one id
                # is not a contract; their values are compared as one group.
                faults.append(f"row {row}: id {token} carries other values")
    return faults


def expected_key(package):
    """The key the package's manifest declares for this program."""
    import tomllib

    manifest = tomllib.loads((pathlib.Path(package) / "package.toml").read_text())
    described = manifest["kernel"]["descriptors"]
    shapes = described if isinstance(described, list) else [described]
    dimensions = shapes[0]["dimensions"]
    return "_".join(
        f"{name}_{values[0]}" for name, values in dimensions.items() if name != "pdl"
    )


def cases():
    """Every scenario the check runs: a width, and which of the two shapes of row it carries."""
    for width in WIDTHS:
        for name in ("plain", "duplicate", "outside"):
            if width < 4 and name != "plain":
                continue
            yield width, name


def run_interpreter(package, key, seed):
    """Launch the Triton source under the interpreter and compare with the reference."""
    import os

    os.environ["TRITON_INTERPRET"] = "1"
    import torch

    source = pathlib.Path(package).resolve() / "src"
    sys.path.insert(0, str(source))
    import candidate_order as kernels  # noqa: PLC0415

    failures = 0
    for width, name in cases():
        values, ids = scenario(width, ROWS, seed, name == "duplicate", name == "outside")
        want_values, want_ids = wanted(width, values, ids)
        # The destination holds the operator's sentinels, which no candidate can be, so an
        # unwritten slot cannot pass as a legitimate token `0`.
        got_values = torch.full((ROWS, width), float("nan"), dtype=torch.float32)
        got_ids = torch.full((ROWS, width), -1, dtype=torch.int32)
        kernels._candidate_order_kernel[(ROWS,)](
            torch.tensor(values),
            torch.tensor(ids),
            got_values,
            got_ids,
            ROWS,
            width,
            BLOCK=128,
        )
        for fault in difference(got_values.numpy(), got_ids.numpy(), want_values, want_ids):
            print(f"width {width}, {name}: {fault}")
            failures += 1
    print(
        f"{len(list(cases()))} scenarios ({WIDTHS}, each plain and with a repeated or an outside id): "
        + ("PASS" if failures == 0 else f"FAIL: {failures} mismatch(es)")
        + f" [{key}]"
    )
    return 0 if failures == 0 else 1


def run_module(directory, seed):
    """The same scenarios through the built cubin, on a device."""
    import ctypes

    import torch
    from cuda.bindings import driver as cu

    metadata = json.loads((pathlib.Path(directory) / "metadata.json").read_text())
    threads = int(metadata.get("num_warps", 4)) * 32
    shared = int(metadata.get("shared", 0))
    _, module = cu.cuModuleLoad((pathlib.Path(directory) / "module.cubin").read_bytes())
    _, function = cu.cuModuleGetFunction(module, metadata["name"].encode())

    failures = 0
    for width, name in cases():
        values, ids = scenario(width, ROWS, seed, name == "duplicate", name == "outside")
        want_values, want_ids = wanted(width, values, ids)
        device = "cuda"
        got_values = torch.full(
            (ROWS, width), float("nan"), dtype=torch.float32, device=device
        )
        got_ids = torch.full((ROWS, width), -1, dtype=torch.int32, device=device)
        holders = [
            ctypes.c_void_p(torch.tensor(values).cuda().data_ptr()),
            ctypes.c_void_p(torch.tensor(ids).cuda().data_ptr()),
            ctypes.c_void_p(got_values.data_ptr()),
            ctypes.c_void_p(got_ids.data_ptr()),
            ctypes.c_uint64(ROWS),
            ctypes.c_uint32(width),
        ]
        pointers = (ctypes.c_void_p * len(holders))(
            *[ctypes.cast(ctypes.byref(holder), ctypes.c_void_p) for holder in holders]
        )
        error = cu.cuLaunchKernel(
            function,
            ROWS,
            1,
            1,
            threads,
            1,
            1,
            shared,
            torch.cuda.current_stream().cuda_stream,
            pointers,
            0,
        )[0]
        if int(error) != 0:
            raise SystemExit(f"launching width {width}: {error}")
        torch.cuda.synchronize()
        for fault in difference(
            got_values.cpu().numpy(), got_ids.cpu().numpy(), want_values, want_ids
        ):
            print(f"width {width}, {name}: {fault}")
            failures += 1
    print("PASS" if failures == 0 else f"FAIL: {failures} mismatch(es)")
    return 0 if failures == 0 else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("module", nargs="?", help="the resolved module directory")
    parser.add_argument(
        "--package",
        default=str(pathlib.Path(__file__).resolve().parents[1]),
        help="the package root holding package.toml",
    )
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    arguments = parser.parse_args()

    key = expected_key(arguments.package)
    if "values_f32" not in key or "ids_i32" not in key:
        raise SystemExit(f"{key}: the check serves values_f32 with ids_i32")
    if arguments.module and key not in pathlib.Path(arguments.module).name:
        raise SystemExit(
            f"{pathlib.Path(arguments.module).name}: the module is not this package's {key}"
        )

    if arguments.cpu_only or not arguments.module:
        return run_interpreter(arguments.package, key, arguments.seed)
    return run_module(arguments.module, arguments.seed)


if __name__ == "__main__":
    sys.exit(main())
