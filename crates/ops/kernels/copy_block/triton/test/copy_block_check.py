#!/usr/bin/env python3
"""Run copy_block's kernel under Triton's interpreter and compare every stored value.

The operator's tolerance is ``exact``: it moves elements, so a correct implementation stores the
source's bits and this check compares values, not a bound. The interpreter is the right instrument
because the kernel performs no arithmetic — the only operations are index decompositions and address
arithmetic, which a device computes identically — and because a wrong stride then shows up as a wrong
value rather than as a slow kernel.

    TRITON_INTERPRET=1 python copy_block_check.py [--package .]

Both sides are torch tensors with **explicit strides** (``torch.as_strided``), which is what lets the
four cases below be four launches of one module rather than four programs:

* a contiguous copy;
* a permutation — the video decoder's channel-first to channel-last reorder, expressed by the
  source's stride vector in the destination's axis order;
* a destination placed inside a larger raster, so its strides are wider than its extents;
* a destination whose innermost axis is **not** unit stride, which the operator explicitly allows
  (it has no unit-stride rule).
"""

import argparse
import importlib.util

import torch


def load_kernel(package):
    """The package's kernel, imported from its source directory."""
    path = f"{package}/src/copy_block.py"
    spec = importlib.util.spec_from_file_location("copy_block_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._copy_block_kernel


def strided_view(shape, strides, dtype):
    """A view of `shape` with `strides` over the smallest base buffer that holds it."""
    span = 1
    for extent, stride in zip(shape, strides):
        span += (extent - 1) * stride
    base = torch.zeros(span, dtype=dtype)
    return base.as_strided(shape, strides), base


def run(kernel, shape, source_strides, out_strides, dtype):
    """Launch the kernel once and answer the source's and the destination's logical views."""
    source, _ = strided_view(shape, source_strides, dtype)
    values = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float64).reshape(shape)
    source.copy_(values.to(dtype))
    out, base = strided_view(shape, out_strides, dtype)
    base.fill_(-1.0)
    total = int(torch.tensor(shape).prod())
    block = 1024
    kernel[((total + block - 1) // block,)](
        source,
        out,
        *shape,
        *source_strides,
        *out_strides,
        total,
        block,
        0,
        BLOCK=block,
    )
    return source, out


def check(shape, source_strides, out_strides, dtype):
    """Compare the destination's logical block with the source's, element by element."""
    source, out = run(KERNEL, shape, source_strides, out_strides, dtype)
    expected = source.clone()
    same = torch.equal(out, expected)
    if same:
        return None
    for index in torch.cartesian_prod(*[torch.arange(extent) for extent in shape]):
        wanted = expected[tuple(int(one) for one in index)].item()
        got = out[tuple(int(one) for one in index)].item()
        if got != wanted:
            return tuple(int(one) for one in index), got, wanted
    return None


def padded(shape, innermost):
    """Row-major strides of `shape` whose innermost stride is `innermost`.

    A view's stride vector must not overlap: each outer stride is at least the extent of the axis it
    skips times the stride inside it. Overlapping strides would let two elements of one block share a
    byte, and the comparison below would then be about which write landed last rather than about the
    copy.
    """
    strides = [0] * len(shape)
    running = innermost
    for axis in reversed(range(len(shape))):
        strides[axis] = running
        running *= shape[axis]
    return strides


def cases():
    """The four copies: plain, permuted, placed, and an inner stride above one."""
    shape = (2, 3, 4, 5)

    def row_major(shape, order):
        """Row-major strides of `shape`, permuted to `order`."""
        strides = [0] * len(shape)
        running = 1
        for axis in reversed(order):
            strides[axis] = running
            running *= shape[axis]
        return strides

    plain = row_major(shape, (3, 2, 1, 0))
    yield "contiguous", shape, plain, plain
    # The video decoder's reorder: [channels, frames, h, w] -> [frames, h, w, channels].
    yield (
        "permuted",
        shape,
        plain,
        [1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]],
    )
    # Placed inside a larger raster: the innermost stride is eight for a run of five, so the gaps the
    # destination leaves are the ones a placed block has.
    yield "placed", shape, plain, padded(shape, 8)
    # An innermost axis that is not unit stride, which the operator permits.
    yield "inner_two", shape, plain, padded(shape, 2)


def main():
    global KERNEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default=".")
    arguments = parser.parse_args()
    KERNEL = load_kernel(arguments.package)
    failures = 0
    for name, shape, source_strides, out_strides in cases():
        for dtype in (torch.float32, torch.float16):
            problem = check(shape, source_strides, out_strides, dtype)
            if problem is None:
                print(f"ok   {name} {str(dtype).split('.')[-1]}")
            else:
                failures += 1
                index, got, wanted = problem
                print(
                    f"FAIL {name} {str(dtype).split('.')[-1]}: index {index} is {got}, "
                    f"expected {wanted}"
                )
    return 1 if failures else 0


KERNEL = None

if __name__ == "__main__":
    raise SystemExit(main())
