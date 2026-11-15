"""What the tensor primitives' checks share.

A check launches one package's kernel the way its operator launches it and
compares the result with numpy or torch on the same values. It runs on
Triton's interpreter, which needs no device, or with ``--device`` compiled
for a device, where the same cases run on device tensors. This module holds
what every such check needs:

- the launch of a descriptor key: the key is read against the package's
  manifest and given to its compile script as the Build Server does for
  a build, and the kernel runs with the compile-time values that script
  states, so a check launches the module a key names;
- the strided inputs a check traverses: contiguous, sliced, stepped, permuted
  and broadcast (zero-stride) views of rank 1 to 6;
- the launch arithmetic of ``crates/ops/src/strided.rs``: the axes a program
  traverses once every run the operands step through together is merged, and the
  extents and strides packed right-aligned;
- the operators' bounds, in the expressions of ``crates/ops/src/tolerance.rs``.

**BF16 on the interpreter.** A device rounds an FP32 value to the nearest
BF16 when it stores one; Triton's interpreter truncates it instead, which
leaves the result up to one BF16 unit in the last place short. A bound
compared on the interpreter therefore gains that unit for a BF16 output, and
no other type needs it. A result that is already a BF16 value (a moved
element, a sign, a whole number) is stored unchanged either way and is
compared exactly.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import os
import pathlib
import sys

#: Whether this run compiles the kernels for a device instead of interpreting
#: them. Triton reads ``TRITON_INTERPRET`` when it is imported, so the mode is
#: chosen here, which every check imports before Triton.
DEVICE = "--device" in sys.argv
if not DEVICE:
    os.environ["TRITON_INTERPRET"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

#: Where a check's tensors live.
TARGET = torch.device("cuda" if DEVICE else "cpu")

#: The kernel packages' root: the checks live in
#: ``tests/python/kernels/<family>/``.
KERNELS = (
    pathlib.Path(__file__).resolve().parents[4] / "crates" / "ops" / "kernels"
)

#: The packages' compile driver is the Build Server's Python package.
sys.path.insert(0, str(KERNELS.parents[1] / "build-server" / "python" / "src"))

from uniserve_build_server import triton as driver  # noqa: E402
from uniserve_build_server.request import Target  # noqa: E402

#: The architecture a check asks a compile script about. It has no
#: programmatic dependent launch, so a key's ``pdl`` compiles to nothing and a
#: check launches one grid at a time.
ARCHITECTURE = "sm_90a"

#: The most axes an operand has (``ops::MAX_RANK``).
MAX_RANK = 6

#: The elements one program covers in a check: small, so that an input of a
#: few hundred elements spans several programs and a masked tail.
BLOCK = 64

#: Torch's element type for each element type the primitives compute with.
ELEMENTS = {
    "u8": torch.uint8,
    "u32": torch.uint32,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
}

FLOATS = ("f16", "bf16", "f32", "f64")
INTEGERS = ("u8", "u32", "i16", "i32", "i64")

#: The element types the primitives move and never compute with
#: (``strided::MOVED`` beyond ``ELEMENTS``), with the torch type that holds
#: their storage. Torch has no FP8 element type, and such an element is the
#: byte it is stored in: a check holds an FP8 tensor as its bytes and names
#: the element type in the key.
MOVED = {"f8e4m3": torch.uint8}

#: The unit round-off of each floating element type
#: (``ops::tolerance::unit_roundoff``).
UNIT = {"f16": 2.0**-11, "bf16": 2.0**-8, "f32": 2.0**-24, "f64": 2.0**-53}


def ready() -> bool:
    """Whether the mode this run asked for can run here.

    A device run on a host without a device says so and is no failure: the
    interpreter run is the one every host can make.
    """
    if DEVICE and not torch.cuda.is_available():
        print(
            "skipped: --device needs a CUDA device; without it the same cases "
            "run on Triton's interpreter",
            file=sys.stderr,
        )
        return False
    return True


def where() -> str:
    """The name of what this run launches on, for a check's last line."""
    return "a device" if DEVICE else "Triton's interpreter"


def load(op: str):
    """The Triton source of ``op``'s package, imported from its ``src/``."""
    source = str(KERNELS / op / "triton" / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    return importlib.import_module(op)


@functools.cache
def described(op: str, key: str) -> dict:
    """What ``op``'s package states for the module of descriptor ``key``.

    The key is read against the package's manifest and given to its
    ``spec``, which is how the Build Server asks for a build: a key the
    manifest does not declare, or a field the compile script misreads, fails
    here rather than passing as a launch of something else.
    """
    load(op)
    root = KERNELS / op / "triton"
    found = importlib.util.spec_from_file_location(
        f"{op}_build", root / "src" / "build.py"
    )
    build = importlib.util.module_from_spec(found)
    found.loader.exec_module(build)
    return driver.stated_for(
        build.spec,
        root,
        key,
        Target.named(ARCHITECTURE),
        os.environ.get("UNISERVE_BINARY", "uniserve"),
    )


def launch(op: str, key: str, programs: tuple, *words) -> None:
    """Launch the module ``key`` names over ``programs``, with ``words``.

    ``words`` are the launch's arguments in the order the operator packs
    them; the compile-time values are the package's own.
    """
    stated = described(op, key)
    signature = stated["signature"]
    packed = [name for name, kind in signature.items() if kind != "constexpr"]
    if len(words) != len(packed):
        raise SystemExit(
            f"{op} {key}: {len(words)} words launched, the signature states "
            f"{len(packed)}: {packed}"
        )
    stated["kernel"][programs](*words, **stated["constexprs"])


def accumulate(dtype: str) -> str:
    """The element type a bound is accumulated in (``DType::accumulate``)."""
    return "f64" if dtype == "f64" else "f32"


def element(dtype: str) -> torch.dtype:
    """Torch's element type for the storage of ``dtype`` elements."""
    return MOVED[dtype] if dtype in MOVED else ELEMENTS[dtype]


def values(dtype: str, shape, seed: int, scale: float = 3.0) -> torch.Tensor:
    """A contiguous tensor of ``dtype`` values drawn from one seed.

    A floating tensor is normal with deviation ``scale``; an integer one is
    uniform over a range its element type holds, both signs where it has one.
    An element type that only moves is uniform over every byte, which holds
    its NaNs, both zeros and its largest values.
    """
    generator = torch.Generator().manual_seed(seed)
    if dtype in MOVED:
        return tensor(torch.randint(0, 256, shape, generator=generator), dtype)
    if dtype in FLOATS:
        drawn = torch.randn(shape, generator=generator, dtype=torch.float64)
        return tensor(drawn * scale, dtype)
    low = 0 if dtype in ("u8", "u32") else -100
    return tensor(torch.randint(low, 101, shape, generator=generator), dtype)


def tensor(data, dtype: str) -> torch.Tensor:
    """``data`` as a contiguous tensor of ``dtype`` where the check runs."""
    return torch.as_tensor(data).to(element(dtype)).to(TARGET)


def empty(shape, element: torch.dtype) -> torch.Tensor:
    """An output of ``shape`` for a launch to write, where the check runs."""
    return torch.empty(tuple(shape), dtype=element, device=TARGET)


def views(dtype: str, seed: int, make=values):
    """Strided inputs of rank 1 to 6, as ``(name, view)``.

    The layouts are the ones an address computation gets wrong: an offset
    window, a step that skips elements, a permutation that leaves no run of
    axes to merge, a broadcast whose stride is zero, and an input wider than
    one program.
    """
    yield "rank 1 contiguous", make(dtype, (37,), seed)
    yield "rank 1 over several blocks", make(dtype, (3 * BLOCK + 17,), seed + 1)
    yield "rank 1 stepped", make(dtype, (40,), seed + 2)[3::3]
    yield "one element", make(dtype, (1,), seed + 3)
    yield "rank 2 contiguous", make(dtype, (5, 7), seed + 4)
    yield "rank 2 transposed", make(dtype, (7, 5), seed + 5).t()
    yield (
        "rank 2 broadcast rows",
        make(dtype, (1, 7), seed + 6).expand(5, 7),
    )
    yield "rank 3 permuted", make(dtype, (3, 4, 5), seed + 7).permute(2, 0, 1)
    yield "rank 3 window", make(dtype, (4, 6, 8), seed + 8)[1:3, ::2, 2:7]
    yield (
        "rank 4 permuted",
        make(dtype, (2, 3, 4, 5), seed + 9).permute(0, 3, 1, 2),
    )
    yield (
        "rank 4 broadcast",
        make(dtype, (2, 1, 4, 1), seed + 10).expand(2, 3, 4, 5),
    )
    yield "rank 5 contiguous", make(dtype, (2, 3, 2, 3, 2), seed + 11)
    yield (
        "rank 5 reversed",
        make(dtype, (2, 3, 2, 3, 2), seed + 12).permute(4, 3, 2, 1, 0),
    )
    yield (
        "rank 6 reversed",
        make(dtype, (2, 2, 3, 2, 2, 2), seed + 13).permute(5, 4, 3, 2, 1, 0),
    )


def real(tensor: torch.Tensor) -> np.ndarray:
    """The values a tensor names, as float64 or int64, contiguous."""
    wide = torch.float64 if tensor.dtype.is_floating_point else torch.int64
    return tensor.to(wide).contiguous().cpu().numpy()


def bits(tensor: torch.Tensor) -> torch.Tensor:
    """A tensor's elements as integers of their own width, contiguous.

    What a moved element is compared by: a NaN's payload and a zero's sign
    are part of it.
    """
    widths = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    return tensor.contiguous().view(widths[tensor.element_size()])


def same(label: str, got: torch.Tensor, expected: torch.Tensor) -> int:
    """Hold ``got`` to the bits of ``expected``, element for element.

    Returns the number of elements compared.
    """
    assert got.dtype == expected.dtype, (label, got.dtype, expected.dtype)
    assert got.shape == expected.shape, (label, got.shape, expected.shape)
    differing = (bits(got) != bits(expected)).nonzero()
    if differing.numel():
        at = tuple(int(axis) for axis in differing[0])
        raise SystemExit(
            f"{label}: {differing.size(0)} of {got.numel()} elements differ; "
            f"first at {at}: got {got[at].item()!r}, expected "
            f"{expected[at].item()!r}"
        )
    return got.numel()


def merge(shape, strides) -> tuple[list[int], list[list[int]]]:
    """The extents and each operand's strides once the shared runs are merged.

    The mirror of ``strided::merge_axes``, over one ``shape`` and one stride
    vector per operand: two adjacent axes merge when every operand's outer
    stride is the inner extent times its inner stride, an axis of one element
    merges with its neighbour whatever its stride, and an operand of no axis
    is one axis of one element.
    """
    # Built from the last axis outward: entry 0 is the innermost run.
    extents: list[int] = []
    runs: list[list[int]] = [[] for _ in strides]
    for axis in reversed(range(len(shape))):
        extent = shape[axis]
        steps = [operand[axis] for operand in strides]
        together = bool(extents) and (
            extent == 1
            or extents[-1] == 1
            or all(
                step == extents[-1] * run[-1]
                for step, run in zip(steps, runs, strict=True)
            )
        )
        if together:
            if extents[-1] == 1:
                for step, run in zip(steps, runs, strict=True):
                    run[-1] = step
            extents[-1] *= extent
            continue
        extents.append(extent)
        for step, run in zip(steps, runs, strict=True):
            run.append(step)

    if not extents:
        extents = [1]
        runs = [[0] for _ in strides]
    return extents[::-1], [run[::-1] for run in runs]


def merged(operands) -> tuple[list[int], list[list[int]]]:
    """The merged extents and strides of tensors that share one shape."""
    shape = list(operands[0].shape)
    assert all(list(operand.shape) == shape for operand in operands)
    return merge(shape, [list(operand.stride()) for operand in operands])


def extent_words(extents) -> list[int]:
    """An index space's extents as a launch packs them: slots 1 to 5."""
    pad = MAX_RANK - len(extents)
    return [1] * (pad - 1) + list(extents[max(0, 1 - pad) :])


def stride_words(strides) -> list[int]:
    """An operand's strides as a launch packs them: six, right-aligned."""
    return [0] * (MAX_RANK - len(strides)) + list(strides)


def grid(elements: int, block: int = BLOCK) -> tuple[int]:
    """The grid of a launch over ``elements`` (``LaunchGrid::covering``)."""
    return (-(-elements // block),)


def gamma(reduction: int, unit: float) -> float:
    """``tolerance::gamma``: ``n u / (1 - n u)``, saturating at one."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


#: A bound is ``(atol, rtol, cancels)``, as ``Tolerance`` holds it.
EXACT = (0.0, 0.0, False)


def rounding(reduction: int, dtype: str):
    """``Tolerance::rounding(reduction, dtype.accumulate(), dtype)``."""
    unit = UNIT[accumulate(dtype)]
    return (0.0, 2.0 * gamma(reduction, unit) + UNIT[dtype], False)


def dot(reduction: int, dtype: str):
    """``Tolerance::dot(reduction, dtype.accumulate(), dtype)``."""
    unit = UNIT[accumulate(dtype)]
    relative = 2.0 * gamma(reduction, unit) * reduction**0.5 + UNIT[dtype]
    return (0.0, relative, True)


def with_atol(bound, atol: float):
    """``Tolerance::with_atol``."""
    return (atol, bound[1], bound[2])


def compare(label: str, got: torch.Tensor, expected: np.ndarray, bound) -> int:
    """Hold ``got`` to ``bound`` around ``expected``, element for element.

    ``expected`` is the exact value of each element in float64 or int64. A
    bound of zero is equality, with a NaN equal to a NaN; any other bound is
    ``|got - expected| <= atol + rtol * magnitude``, the magnitude being the
    element's own or, for a bound that cancels, the output's largest. A BF16
    output on the interpreter gains the store's truncation (module
    docstring). Returns the number of elements compared.

    A rounding bound is relative, which holds where the element type's
    values are: a value past its largest finite one is stored as an infinity,
    and below its smallest normal one the type's grid is uniform, so the
    comparison there allows one step of that grid.
    """
    result = real(got)
    expected = np.asarray(expected).reshape(result.shape)
    atol, rtol, cancels = bound

    if atol == 0.0 and rtol == 0.0:
        wrong = ~(
            (result == expected) | ((result != result) & (expected != expected))
        )
    else:
        limits = torch.finfo(got.dtype)
        if got.dtype == torch.bfloat16 and not DEVICE:
            rtol += 2.0 * UNIT["bf16"]
        with np.errstate(invalid="ignore"):
            overflows = np.abs(expected) > limits.max
        expected = np.where(overflows, np.copysign(np.inf, expected), expected)

        finite = np.isfinite(expected)
        magnitude = np.abs(np.where(finite, expected, 0.0))
        if cancels:
            magnitude = np.full_like(magnitude, magnitude.max(initial=0.0))
        allowed = atol + rtol * magnitude + limits.tiny * limits.eps
        with np.errstate(invalid="ignore"):
            close = np.abs(result - expected) <= allowed
        # An infinity or a NaN has no neighbourhood: it is matched as it is.
        same = (result == expected) | (
            (result != result) & (expected != expected)
        )
        wrong = ~np.where(finite, close, same)

    if wrong.any():
        at = tuple(int(axis) for axis in np.argwhere(wrong)[0])
        raise SystemExit(
            f"{label}: {int(wrong.sum())} of {result.size} elements outside "
            f"atol {atol:.3e} rtol {rtol:.3e}; first at {at}: got "
            f"{result[at]!r}, expected {expected[at]!r}"
        )
    return int(result.size)
