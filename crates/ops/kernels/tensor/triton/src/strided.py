"""What the tensor primitives' Triton sources share.

The primitives (``elementwise_unary``, ``elementwise_binary``,
``elementwise_where``, ``cast``, ``copy_strided``, ``reduce``, ``index`` and
``fill``) read their operands through strides: element ``[i0, .., ir-1]`` of
an operand is ``sum(ik * stride_k)`` elements from its address, and a zero
stride repeats an axis. Extents and strides are words of the launch, so a
module is compiled for a rank and serves every shape of that rank.

Each package links this file into its own ``src/``: the address arithmetic of
a strided operand is one implementation, and an installation copies the file
the link names.

**The words.** A kernel takes ``MAX_RANK`` strides per strided operand and
``MAX_RANK - 1`` extents per index space, right-aligned: axis ``k`` of
``RANK`` axes is slot ``MAX_RANK - RANK + k``. The slots before it are unit
extents and zero strides, which a module compiled for ``RANK`` never reads.
The extent of slot 0 does not travel: only an outermost axis can sit there,
and a row-major index needs no modulus on its outermost axis.

**The computing type.** A half-precision element is widened to FP32 when it
is loaded and rounded to nearest even once when it is stored; FP32, FP64 and
the integers compute in their own type, the integers wrapping. Every
conversion to or from a half-precision type goes through FP32, the one such
conversion Triton lowers.

**Elements that only move.** An FP8 (``f8e4m3``) element is one byte, and the
primitives that convert nothing (``fill``, ``copy_strided``,
``elementwise_where``, and the ``gather`` and ``scatter_set`` of ``index``)
address it as ``u8`` and store the bits they loaded. No primitive computes with
one (``compute`` refuses it). Two NVFP4 elements share a byte, so no strided
kernel addresses one: its storage is moved as the ``u8`` tensor of its packed
bytes.
"""

import triton
import triton.language as tl

#: The most axes an operand has (``ops::MAX_RANK``).
MAX_RANK = 6

#: The element types the primitives move and never compute with: an element is
#: the byte it is stored in.
MOVED = ("f8e4m3",)

#: Triton's name for each element type an operand holds, by the name
#: ``ops::abi::name`` prints. An element that only moves is addressed as its
#: byte.
ELEMENTS = {
    "u8": "u8",
    "u32": "u32",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
    "f16": "fp16",
    "bf16": "bf16",
    "f32": "fp32",
    "f64": "fp64",
    "f8e4m3": "u8",
}

#: The unsigned integer as wide as each element type: what holds its bits.
BITS = {
    "u8": "u8",
    "u32": "u32",
    "i16": "u16",
    "i32": "u32",
    "i64": "u64",
    "f16": "u16",
    "bf16": "u16",
    "f32": "u32",
    "f64": "u64",
    "f8e4m3": "u8",
}


def pointer(dtype):
    """The signature type of an operand of ``dtype`` elements."""
    return "*" + ELEMENTS[dtype]


def compute(dtype):
    """The type a kernel computes ``dtype`` elements in.

    An element type that only moves has none: a descriptor that asks a kernel
    to compute with one names no module.
    """
    if dtype in MOVED:
        raise SystemExit(
            f"{dtype} elements move as their bytes: no primitive computes "
            f"with them"
        )
    if dtype == "f64":
        return "fp64"
    if dtype in ("f16", "bf16", "f32"):
        return "fp32"
    return "int"


def extents(prefix):
    """The signature of one index space's extents, slots 1 and up."""
    return {f"{prefix}{axis}": "i64" for axis in range(1, MAX_RANK)}


def strides(prefix):
    """The signature of one operand's strides."""
    return {f"{prefix}{axis}": "i64" for axis in range(MAX_RANK)}


def warps_of(block):
    """The warp count one block of elements needs."""
    return 8 if block >= 2048 else 4


@triton.jit
def pdl_prologue(PDL: tl.constexpr):  # noqa: N803
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def offsets(
    index,
    d1,
    d2,
    d3,
    d4,
    d5,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    RANK: tl.constexpr,  # noqa: N803
):
    """The element offset of each row-major ``index`` in one operand.

    ``index`` counts the elements of an index space of ``RANK`` axes in
    row-major order; ``d1..d5`` are its extents and ``s0..s5`` the operand's
    strides, both right-aligned. The index is taken apart from the last axis
    outward, one modulus and one division per axis but the outermost, so a
    rank-1 module performs neither.
    """
    rest = index
    offset = index * 0

    if RANK > 1:
        offset += (rest % d5) * s5
        rest = rest // d5
    if RANK > 2:
        offset += (rest % d4) * s4
        rest = rest // d4
    if RANK > 3:
        offset += (rest % d3) * s3
        rest = rest // d3
    if RANK > 4:
        offset += (rest % d2) * s2
        rest = rest // d2
    if RANK > 5:
        offset += (rest % d1) * s1
        rest = rest // d1

    # What is left indexes the outermost of the `RANK` axes.
    if RANK == 1:
        offset += rest * s5
    elif RANK == 2:
        offset += rest * s4
    elif RANK == 3:
        offset += rest * s3
    elif RANK == 4:
        offset += rest * s2
    elif RANK == 5:
        offset += rest * s1
    else:
        offset += rest * s0
    return offset


@triton.jit
def widened(value, COMPUTE: tl.constexpr):  # noqa: N803
    """A loaded value in the type its kernel computes in."""
    if COMPUTE == "fp32":
        result = value.to(tl.float32)
    elif COMPUTE == "fp64":
        result = value.to(tl.float64)
    else:
        result = value
    return result


@triton.jit
def narrowed(value, pointer, COMPUTE: tl.constexpr):  # noqa: N803
    """A computed value in the element type of ``pointer``, for a store.

    A floating value rounds to nearest even, once; an integer wraps.
    """
    if COMPUTE == "int":
        result = value.to(pointer.dtype.element_ty)
    else:
        result = value.to(pointer.dtype.element_ty, fp_downcast_rounding="rtne")
    return result
