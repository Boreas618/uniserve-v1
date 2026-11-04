#!/usr/bin/env python3
"""Run copy_block's kernel and check every value it stores against the reference copy.

The operator's tolerance is ``exact``: it moves elements, so a correct implementation stores the
source's bits and this check compares them as bits rather than against a bound. The interpreter's
arithmetic is the right instrument because the kernel performs none: the only operations are index
decompositions and address arithmetic, which a device computes identically.

    TRITON_INTERPRET=1 python copy_block_check.py [--package .]

What is checked, all four cases a launch can express:

* a contiguous copy, where the strides are the block's own;
* a permutation, the source's stride vector in the destination's axis order -- the channel-first to
  channel-last reorder the video decoder needs;
* a destination placed inside a larger raster, so its strides are not its extents' product;
* a destination whose innermost axis is **not** unit stride, which the operator explicitly allows.
"""

import argparse
import importlib.util

import numpy as np


def load_kernel(package):
    """The package's kernel, imported from its source directory."""
    path = f"{package}/src/copy_block.py"
    spec = importlib.util.spec_from_file_location("copy_block_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._copy_block_kernel


def run(kernel, shape, source_strides, out_strides, dtype):
    """Launch the kernel once on a host buffer and answer the destination's region."""
    total = int(np.prod(shape))
    source = np.arange(total, dtype=np.float64).reshape(shape).astype(dtype)
    # A destination region large enough for the outermost address of each of its axes.
    span = 1
    for extent, stride in zip(shape, out_strides):
        span += (extent - 1) * stride
    out = np.full(span, -1.0, dtype=dtype)
    kernel[(1,)](
        source,
        out,
        *shape,
        *source_strides,
        *out_strides,
        total,
        1024,
        0,
        BLOCK=1024,
    )
    return source, out


def reference(shape, source_strides, out_strides, source, out, dtype):
    """Write the block into `out` the way the operator defines it, and compare."""
    for index in np.ndindex(*shape):
        source_offset = sum(coordinate * stride for coordinate, stride in zip(index, source_strides))
        out_offset = sum(coordinate * stride for coordinate, stride in zip(index, out_strides))
        expected = source.reshape(-1)[source_offset]
        if out[out_offset] != expected:
            return (index, out_offset, float(out[out_offset]), float(expected))
    return None


def cases():
    """The four copies: plain, permuted, placed, and an inner stride above one."""
    shape = (2, 3, 4, 5)

    def strided(shape, order):
        """Row-major strides of `shape`, permuted to `order`."""
        strides = [0] * len(shape)
        running = 1
        for axis in reversed(order):
            strides[axis] = running
            running *= shape[axis]
        return strides

    yield "contiguous", shape, strided(shape, (3, 2, 1, 0)), strided(shape, (3, 2, 1, 0))
    # The video decoder's reorder: [channels, frames, h, w] to [frames, h, w, channels].
    yield "permuted", shape, strided(shape, (3, 2, 1, 0)), [1, shape[0] * 1, shape[0] * shape[1], shape[0] * shape[1] * shape[2]]
    # Placed inside a larger raster: every stride is wider than the extent needs.
    yield "placed", shape, strided(shape, (3, 2, 1, 0)), [64, 128, 256, 512]
    # An innermost axis that is not unit stride, which the operator permits.
    yield "inner_two", shape, strided(shape, (3, 2, 1, 0)), [2, 16, 128, 1024]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default=".")
    arguments = parser.parse_args()
    kernel = load_kernel(arguments.package)
    failures = 0
    for name, shape, source_strides, out_strides in cases():
        for dtype in (np.float32, np.float16):
            source, out = run(kernel, shape, source_strides, out_strides, dtype)
            problem = reference(shape, source_strides, out_strides, source, out, dtype)
            if problem is None:
                print(f"ok   {name} {np.dtype(dtype).name}")
            else:
                failures += 1
                index, offset, got, expected = problem
                print(
                    f"FAIL {name} {np.dtype(dtype).name}: index {index} at {offset} is "
                    f"{got}, expected {expected}"
                )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
