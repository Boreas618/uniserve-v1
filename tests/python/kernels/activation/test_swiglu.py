"""The SwiGLU programs against torch: their output and their magnitude record.

One module covers both operators because they are one operation over two
layouts: ``value * silu(gate)``, optionally with a bias on each half and
optionally recording each program's largest output magnitude for a later
quantization. ``value_first_swiglu`` reads a **packed projection** whose rows
start with the value half and end with the gate half, one bias of
``2 * width``; ``swiglu`` reads separate value and gate views, each with its
own bias and its own row stride.

The scenarios: random values; gates swept to +-80, where the sigmoid has
saturated and ``silu`` of a very negative gate is already about ``1e-35``
while ``exp(-gate)`` is still finite in FP32; and a zeroed row, whose output
and magnitude are exactly zero. A gate past ``log(FLT_MAX) = 88.7`` is no
case: every FP32 implementation of ``gate / (1 + exp(-gate))`` returns zero
there, below the exact value, and with ``atol = 0`` the formula's own limit
reads as a miss.

**Every stride a case builds is a different number from the width, and from
the other stride.** A stride that equals the width makes a wrong address
arithmetic indistinguishable from the right one. With ``value_stride`` twice
the width and ``gate_stride`` equal to it, a kernel that assumed rows are
adjacent reads the wrong row's channels.

The bound is each operator's own, from ``activation.rs::tolerance``: five
roundings in the output type for the separate form (two bias adds, the
sigmoid, its product with the gate, the product with the value) and four for
the packed one, which has a single bias. The magnitudes are compared
**exactly**: a maximum of values already in the output type has nothing to
round.

A ``bf16`` store under Triton's interpreter truncates where hardware rounds,
and an ``f16`` store there rounds; the reference takes the same cast as the
run's store, so what is compared is the arithmetic and not the interpreter's
store model.
"""

import pytest
import torch

pytestmark = pytest.mark.unit

#: The output types' unit roundoff, as `ops::tolerance::unit_roundoff` gives
#: it: the narrow types, whose stores the reference models.
UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-8}

TORCH_TYPES = {"f16": torch.float16, "bf16": torch.bfloat16}

#: The operators, and the bound's factor from `activation.rs::tolerance`.
ROUNDINGS = {"swiglu": 5.0, "value_first_swiglu": 4.0}

#: The scenarios, each with the seed its values are drawn from.
SCENARIOS = {"random values": 1, "saturating gates": 2, "a zeroed row": 3}

#: The rows, their width and the block one program covers.
ROWS, WIDTH, BLOCK = 2, 4096, 1024

#: `value_first_swiglu`'s packed row stride; `swiglu`'s value rows are twice
#: the width apart and its gate rows the width apart.
INPUT_STRIDE = 8192
VALUE_STRIDE, GATE_STRIDE = 2 * WIDTH, WIDTH


def silu(gate):
    """`gate * sigmoid(gate)` in torch, in FP64."""
    return gate * torch.sigmoid(gate)


def truncate(values):
    """The reference as Triton's interpreter stores it: truncated, not rounded.

    The `bf16` significand is eight bits, so the value is truncated on its own
    mantissa.
    """
    bits = values.view(torch.int64)
    return (bits & ~((1 << 45) - 1)).view(torch.float64)


def store(values, dtype, truncated):
    """A value as the run's store keeps it: truncated or rounded."""
    if dtype == "bf16" and truncated:
        return truncate(values)
    return values.to(TORCH_TYPES[dtype]).double()


def launch_packed(kernels, packed, bias, out, partials, absmax):
    """One launch of `value_first_swiglu` over a packed projection; an absent
    bias or magnitude record is the output's buffer, which the program does not
    read or write then.
    """  # noqa: D205
    elements = out.numel()
    kernels._value_first_swiglu_kernel[(-(-elements // BLOCK),)](
        packed,
        bias if bias is not None else out.reshape(-1),
        out,
        partials if absmax else out.reshape(-1),
        elements,
        INPUT_STRIDE,
        width=WIDTH,
        HAS_BIAS=bias is not None,
        RETURN_ABSMAX=absmax,
        BLOCK=BLOCK,
    )


def launch_separate(kernels, value, gate, biases, out, partials, absmax):
    """One launch of `swiglu` over separate value and gate views; an absent
    bias or magnitude record is the output's buffer, which the program does not
    read or write then.
    """  # noqa: D205
    value_bias, gate_bias = biases
    elements = out.numel()
    kernels._swiglu_kernel[(-(-elements // BLOCK),)](
        value,
        gate,
        value_bias if value_bias is not None else out.reshape(-1),
        gate_bias if gate_bias is not None else out.reshape(-1),
        out,
        partials if absmax else out.reshape(-1),
        elements,
        width=WIDTH,
        value_row_stride=VALUE_STRIDE,
        gate_row_stride=GATE_STRIDE,
        HAS_VALUE_BIAS=value_bias is not None,
        HAS_GATE_BIAS=gate_bias is not None,
        RETURN_ABSMAX=absmax,
        BLOCK=BLOCK,
    )


def packed_case(
    kernels, scenario, element, bias, out, partials, absmax, device
):
    """Launch `value_first_swiglu` on one scenario; the exact value and gate."""
    generator = torch.Generator().manual_seed(SCENARIOS[scenario])
    buffer = torch.randn(ROWS, INPUT_STRIDE, generator=generator)
    if scenario == "saturating gates":
        buffer[:, WIDTH:] = (
            torch.linspace(-80.0, 80.0, WIDTH).unsqueeze(0).repeat(ROWS, 1)
        )
    if scenario == "a zeroed row":
        buffer[0, :] = 0.0
    buffer = buffer.to(element)
    biases = (
        torch.randn(2 * WIDTH, generator=generator).to(element)
        if bias
        else None
    )
    launch_packed(
        kernels,
        buffer.to(device)[:, : 2 * WIDTH],
        None if biases is None else biases.to(device),
        out,
        partials,
        absmax,
    )

    exact = buffer.double()
    value, gate = exact[:, :WIDTH], exact[:, WIDTH : 2 * WIDTH]
    if biases is not None:
        value = value + biases.double()[:WIDTH]
        gate = gate + biases.double()[WIDTH:]
    return value, gate


def separate_case(
    kernels, scenario, element, bias, out, partials, absmax, device
):
    """Launch `swiglu` on one scenario; the exact value and gate."""
    generator = torch.Generator().manual_seed(SCENARIOS[scenario])
    # Rows that are not adjacent: each half is its own buffer with its own
    # stride, and the padding holds a value a wrong stride would read.
    values = torch.randn(ROWS, VALUE_STRIDE, generator=generator)
    gates = torch.randn(ROWS, GATE_STRIDE, generator=generator)
    if scenario == "saturating gates":
        gates[:, :WIDTH] = torch.linspace(-80.0, 80.0, WIDTH)
    if scenario == "a zeroed row":
        values[0, :] = 0.0
        gates[0, :] = 0.0
    values, gates = values.to(element), gates.to(element)
    biases = tuple(
        torch.randn(WIDTH, generator=generator).to(element) if bias else None
        for _ in range(2)
    )
    launch_separate(
        kernels,
        values.to(device)[:, :WIDTH],
        gates.to(device)[:, :WIDTH],
        tuple(None if b is None else b.to(device) for b in biases),
        out,
        partials,
        absmax,
    )

    value, gate = values.double()[:, :WIDTH], gates.double()[:, :WIDTH]
    value_bias, gate_bias = biases
    if value_bias is not None:
        value = value + value_bias.double()
    if gate_bias is not None:
        gate = gate + gate_bias.double()
    return value, gate


@pytest.mark.parametrize("absmax", [False, True], ids=["no absmax", "absmax"])
@pytest.mark.parametrize("bias", [False, True], ids=["no bias", "bias"])
@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("op", ROUNDINGS)
def test_the_output_and_its_magnitudes_match_torch(
    op, scenario, dtype, bias, absmax, torch_device, interpreted, kernel_source
):
    element = TORCH_TYPES[dtype]
    programs = -(-ROWS * WIDTH // BLOCK)
    out = torch.full(
        (ROWS, WIDTH), float("nan"), dtype=element, device=torch_device
    )
    partials = torch.full(
        (programs,), float("nan"), dtype=torch.float32, device=torch_device
    )
    case = packed_case if op == "value_first_swiglu" else separate_case
    value, gate = case(
        kernel_source(op),
        scenario,
        element,
        bias,
        out,
        partials,
        absmax,
        torch_device,
    )

    expected = store(value * silu(gate), dtype, interpreted and dtype == "bf16")
    got = out.cpu().double()
    difference = (got - expected).abs()
    allowed = ROUNDINGS[op] * UNIT_ROUNDOFF[dtype] * expected.abs()
    outside = int((~(difference <= allowed)).sum())
    largest = (
        (difference / expected.abs().clamp(min=1e-30))
        .nan_to_num(nan=float("inf"))
        .max()
        .item()
    )
    assert outside == 0, (
        f"{op}, {scenario}, {dtype}: {outside} of {expected.numel()} outside "
        f"the bound, max relative error {largest:.3e}"
    )

    if absmax:
        # Each program's record is the largest magnitude of its own lanes.
        lanes = expected.reshape(programs, BLOCK).abs().amax(dim=1)
        record = partials.cpu().double()
        differing = (record != lanes).nonzero()
        assert differing.numel() == 0, (
            f"{op}, {scenario}, {dtype}: {differing.numel()} of {programs} "
            f"magnitudes differ; program {int(differing[0])} records "
            f"{float(record[differing[0]]):.9e}, its lanes' largest "
            f"{float(lanes[differing[0]]):.9e}"
        )
