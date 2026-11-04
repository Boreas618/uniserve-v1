#!/usr/bin/env python3
"""Check candidate_order: every row's candidates in ascending token order.

The operator is a permutation and its tolerance is `exact`, so the reference here
is a stable sort by token id — written with numpy's stable argsort, which is the
same rule the kernel states (ascending id, ties by source position) — and the
comparison is bit equality of both outputs, never a bound.

    # the decisive run, on a host without a device: Triton's interpreter
    python order_check.py --cpu-only [--package .]

    # the same scenarios through the built cubin, on the GPU host
    python order_check.py <resolved module dir> [--package .]

Why the interpreter is the right instrument *here* and not for an fp8 kernel: a
permutation is integer comparison, integer ranks and moves; the interpreter's one
known defect is that it truncates a narrowing float cast, and this program has no
narrowing cast at all. The values are moved, never converted.

The scenarios are chosen for the width being a launch value rather than a field:
one candidate, a width below the kernel's tile, exactly one tile, one tile plus
one, several tiles, and a row whose ids repeat — the last one makes the tie rule
a checked outcome instead of an accident.
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


def reference(values, ids):
    """Ascending id order, ties by source position: numpy's stable sort."""
    order = np.argsort(ids, axis=1, kind="stable")
    return (
        np.take_along_axis(values, order, axis=1),
        np.take_along_axis(ids, order, axis=1),
    )


def scenario(width, rows, seed, duplicate=False):
    """One scenario's inputs: values, ids, and the order the operator must produce."""
    generator = np.random.default_rng(seed)
    values = generator.random((rows, width), dtype=np.float32)
    ids = generator.permutation(np.arange(width, dtype=np.int32))
    ids = np.stack([ids for _ in range(rows)])
    if duplicate and width >= 4:
        # A duplicated id in every row: the tie rule decides, and the check says
        # so rather than accepting either order.
        ids[:, 1] = ids[:, 0]
    return values, ids, reference(values, ids)


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


def run_interpreter(package, key, seed):
    """Launch the Triton source under the interpreter and compare with the reference."""
    import os

    os.environ["TRITON_INTERPRET"] = "1"
    import torch

    source = pathlib.Path(package).resolve() / "src"
    sys.path.insert(0, str(source))
    import candidate_order as kernels  # noqa: PLC0415

    failures = 0
    for width in WIDTHS:
        for duplicate in (False, True):
            values, ids, (want_values, want_ids) = scenario(width, ROWS, seed, duplicate)
            # The interpreter writes into the tensors it is handed, so the
            # comparison reads the module's own destination buffers.
            got_values = torch.zeros(ROWS, width, dtype=torch.float32)
            got_ids = torch.zeros(ROWS, width, dtype=torch.int32)
            kernels._candidate_order_kernel[(ROWS,)](
                torch.tensor(values),
                torch.tensor(ids),
                got_values,
                got_ids,
                ROWS,
                width,
                BLOCK=128,
            )
            if not np.array_equal(got_ids.numpy(), want_ids):
                print(f"ids differ at width {width}, duplicate {duplicate}")
                failures += 1
            if not np.array_equal(got_values.numpy(), want_values):
                print(f"values differ at width {width}, duplicate {duplicate}")
                failures += 1
    print(
        f"{len(WIDTHS) * 2} scenarios ({WIDTHS}, each with and without a repeated id): "
        + ("PASS" if failures == 0 else f"FAIL: {failures} mismatch(es)")
        + f" [{key}]"
    )
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
    for width in WIDTHS:
        for duplicate in (False, True):
            values, ids, (want_values, want_ids) = scenario(width, ROWS, seed, duplicate)
            device = "cuda"
            got_values = torch.zeros(ROWS, width, dtype=torch.float32, device=device)
            got_ids = torch.zeros(ROWS, width, dtype=torch.int32, device=device)
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
            if not np.array_equal(got_ids.cpu().numpy(), want_ids):
                print(f"ids differ at width {width}, duplicate {duplicate}")
                failures += 1
            if not np.array_equal(got_values.cpu().numpy(), want_values):
                print(f"values differ at width {width}, duplicate {duplicate}")
                failures += 1
    print("PASS" if failures == 0 else f"FAIL: {failures} mismatch(es)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
