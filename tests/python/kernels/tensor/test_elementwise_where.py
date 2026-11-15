"""The elementwise_where program against torch's selection.

Each case launches the package's kernel as ``ElementwiseWhere::launch`` does
(the condition, both values and the output merged into the axes a program
traverses, the extents and the three stride vectors packed right-aligned) and
compares the output with ``torch.where`` **bit for bit**: a selection moves
one of its two values, so the operator's tolerance is exact.

What is covered:

- **every value type under every condition type** (``u8``, ``u32``,
  ``i64``), contiguous, an FP8 (``f8e4m3``) value selected as the byte it is;
- **a condition that is any nonzero value**, not only one: 2, 200 and, where
  the type is signed, a negative number select ``on_true`` as 1 does, which a
  kernel comparing against one would miss;
- **every layout** of :data:`~uniserve_testing.kernels.tensor.operands.LAYOUTS`
  as the condition, with a permuted, stepped or broadcast ``on_true`` of the
  same shape and an ``on_false`` that is one value repeated everywhere, and
  the same condition repeated along its last axis over contiguous values;
- **a NaN and a negative zero among the values**, which a bitwise comparison
  tells from any other.
"""

import functools

import pytest
import torch
from uniserve_testing.kernels.tensor import operands

pytestmark = pytest.mark.unit

CONDITIONS = ("u8", "u32", "i64")

#: Every value type the selection moves.
DTYPES = operands.FLOATS + operands.INTEGERS + tuple(operands.MOVED)

#: The value types and condition types every layout is selected under.
LAYOUT_TYPES = (
    ("bf16", "u8"),
    ("f32", "u32"),
    ("i64", "i64"),
    ("u8", "u8"),
    ("f8e4m3", "u32"),
)


def conditions(primitives, dtype, shape, seed):
    """A condition of zeros and of nonzero values that are not all one."""
    generator = torch.Generator().manual_seed(seed)
    truths = [0, 0, 1, 2, 200] if dtype != "i64" else [0, 0, 1, -1, 1 << 40]
    drawn = torch.randint(0, len(truths), shape, generator=generator)
    return primitives.tensor(torch.tensor(truths)[drawn], dtype)


def check(primitives, name, dtype, condition, on_true, on_false):
    """One launch over values of `dtype` held to torch's selection, bit for
    bit.
    """  # noqa: D205
    out = primitives.empty(condition.shape, on_true.dtype)
    extents, strides = operands.merged([condition, on_true, on_false, out])
    elements = out.numel()
    # `WhereDescriptor`'s `Display`.
    key = (
        f"dtype_{dtype}_condition_{operands.NAMES[condition.dtype]}"
        f"_rank_{len(extents)}_block_{operands.BLOCK}_pdl_false"
    )
    primitives.launch(
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
    )

    # The selection as torch states it, on contiguous copies so that no layout
    # is involved.
    holds = condition.contiguous() != 0
    expected = torch.where(holds, on_true.contiguous(), on_false.contiguous())
    label = f"{dtype} under {condition.dtype} {name} {tuple(condition.shape)}"
    operands.assert_same(label, out, expected)


@pytest.mark.parametrize(
    ("seed", "condition_dtype"), list(enumerate(CONDITIONS))
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_every_value_type_is_selected_under_every_condition_type(
    dtype, seed, condition_dtype, primitives
):
    condition = conditions(primitives, condition_dtype, (5, 7), seed)
    on_true = primitives.values(dtype, (5, 7), seed + 10)
    on_false = primitives.values(dtype, (5, 7), seed + 20)
    check(primitives, "contiguous", dtype, condition, on_true, on_false)


@pytest.mark.parametrize("pattern", [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]])
@pytest.mark.parametrize("dtype", operands.FLOATS)
def test_a_nan_and_a_negative_zero_are_selected_as_they_are(
    dtype, pattern, primitives
):
    on_true = primitives.tensor([float("nan"), -0.0, 1.0, float("inf")], dtype)
    on_false = primitives.tensor([0.0, 0.0, float("nan"), -0.0], dtype)
    condition = primitives.tensor(pattern, "u8")
    check(
        primitives,
        "a NaN and a negative zero",
        dtype,
        condition,
        on_true,
        on_false,
    )


@pytest.mark.parametrize("layout", operands.LAYOUTS)
@pytest.mark.parametrize(("dtype", "condition_dtype"), LAYOUT_TYPES)
def test_every_layout_is_a_condition(
    dtype, condition_dtype, layout, primitives
):
    """The values take other layouts of the same shape."""
    make = functools.partial(conditions, primitives)
    condition = primitives.view(layout, condition_dtype, 5, make=make)
    on_true = primitives.view(layout, dtype, 105)
    shape = tuple(condition.shape)
    on_false = primitives.values(dtype, (1,) * len(shape), 205).expand(shape)
    check(primitives, layout, dtype, condition, on_true, on_false)

    # The same condition repeated along its last axis, over contiguous values.
    repeated = condition[..., :1].expand(shape)
    check(
        primitives,
        f"{layout}, repeated",
        dtype,
        repeated,
        primitives.values(dtype, shape, 305),
        on_false,
    )
