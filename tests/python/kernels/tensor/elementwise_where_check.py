#!/usr/bin/env python3
"""Run `elementwise_where`'s kernel and check the selection against torch, on the CPU.

    python elementwise_where_check.py             # Triton's interpreter, no device
    python elementwise_where_check.py --device    # compiled for a device

Each case launches the package's kernel as `ElementwiseWhere::launch` does — the condition, both
values and the output merged into the axes a program traverses, the extents and the three stride
vectors packed right-aligned — and compares the output with `torch.where` **bit for bit**: a
selection moves one of its two values, so the operator's tolerance is exact.

What is covered:

- **every value type under every condition type** (`u8`, `u32`, `i64`), contiguous, an FP8
  (`f8e4m3`) value selected as the byte it is;
- **a condition that is any nonzero value**, not only one: 2, 200 and, where the type is signed, a
  negative number select `on_true` as 1 does, which a kernel comparing against one would miss;
- **every layout** of `operands.views` as the condition, with a permuted, stepped or broadcast
  `on_true` of the same shape and an `on_false` that is one value repeated everywhere;
- **a NaN and a negative zero among the values**, which a bitwise comparison tells from any other.
"""

import sys

# `operands` chooses between the interpreter and a device, and is imported before Triton.
import operands
import torch

CONDITIONS = ("u8", "u32", "i64")

#: The name `ops::DType` prints for each torch element type.
NAMES = {element: name for name, element in operands.ELEMENTS.items()}


def conditions(dtype, shape, seed):
    """A condition of zeros and of nonzero values that are not all one."""
    generator = torch.Generator().manual_seed(seed)
    truths = [0, 0, 1, 2, 200] if dtype != "i64" else [0, 0, 1, -1, 1 << 40]
    drawn = torch.randint(0, len(truths), shape, generator=generator)
    return operands.tensor(torch.tensor(truths)[drawn], dtype)


def check(name, dtype, condition, on_true, on_false):
    """One launch over values of `dtype` held to torch's selection, bit for bit."""
    out = operands.empty(condition.shape, on_true.dtype)
    extents, strides = operands.merged([condition, on_true, on_false, out])
    elements = out.numel()

    # `WhereDescriptor`'s `Display`.
    key = (
        f"dtype_{dtype}_condition_{NAMES[condition.dtype]}"
        f"_rank_{len(extents)}_block_{operands.BLOCK}_pdl_false"
    )
    operands.launch(
        "elementwise_where",
        key,
        operands.grid(elements),
        condition,
        on_true,
        on_false,
        out,
        elements,
        *operands.extent_words(extents),
        *operands.stride_words(strides[0]),
        *operands.stride_words(strides[1]),
        *operands.stride_words(strides[2]),
        0,
    )

    # The selection as torch states it, on contiguous copies so that no layout is involved.
    holds = condition.contiguous() != 0
    expected = torch.where(holds, on_true.contiguous(), on_false.contiguous())
    label = f"{dtype} under {condition.dtype} {name} {tuple(condition.shape)}"
    return operands.same(label, out, expected)


def main():
    if not operands.ready():
        return 0

    checked = 0
    dtypes = operands.FLOATS + operands.INTEGERS + tuple(operands.MOVED)

    # Every value type under every condition type.
    for dtype in dtypes:
        for seed, condition_dtype in enumerate(CONDITIONS):
            condition = conditions(condition_dtype, (5, 7), seed)
            on_true = operands.values(dtype, (5, 7), seed + 10)
            on_false = operands.values(dtype, (5, 7), seed + 20)
            checked += check("contiguous", dtype, condition, on_true, on_false)

    # Values a bitwise comparison tells apart.
    for dtype in operands.FLOATS:
        on_true = operands.tensor(
            [float("nan"), -0.0, 1.0, float("inf")], dtype
        )
        on_false = operands.tensor([0.0, 0.0, float("nan"), -0.0], dtype)
        for pattern in ([1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]):
            condition = operands.tensor(pattern, "u8")
            checked += check(
                "a NaN and a negative zero", dtype, condition, on_true, on_false
            )

    # Every layout as the condition; the values take other layouts of the same shape.
    layouts = 0
    for dtype, condition_dtype in (
        ("bf16", "u8"),
        ("f32", "u32"),
        ("i64", "i64"),
        ("u8", "u8"),
        ("f8e4m3", "u32"),
    ):
        cases = zip(
            operands.views(condition_dtype, 5, make=conditions),
            operands.views(dtype, 105),
            strict=True,
        )
        for (name, condition), (_, on_true) in cases:
            shape = tuple(condition.shape)
            on_false = operands.values(dtype, (1,) * len(shape), 205).expand(
                shape
            )
            checked += check(name, dtype, condition, on_true, on_false)

            # The same condition repeated along its last axis, over contiguous values.
            repeated = condition[..., :1].expand(shape)
            checked += check(
                f"{name}, repeated",
                dtype,
                repeated,
                operands.values(dtype, shape, 305),
                on_false,
            )
            layouts += 2

    print(
        f"ok: {checked} elements bit for bit over {len(dtypes)} value types under "
        f"{len(CONDITIONS)} condition types and {layouts} strided and broadcast launches of rank 1 "
        f"to 6 on {operands.where()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
