"""The rope_apply program against the rotation alone.

The tables arrive already evaluated, one unit pair per turned feature and
token, so the call costs only the rotation of each head's leading ``ROTATED``
channels::

    out[t,h,low]  = x[t,h,low] cos - x[t,h,high] sin
    out[t,h,high] = x[t,h,high] cos + x[t,h,low] sin
    out[t,h,k]    = x[t,h,k]          k >= ROTATED: a partial rotation copies

with ``low = j, high = j + ROTATED/2`` in the half pairing and
``low = 2j, high = 2j + 1`` in the interleaved one. A head need not be a power
of two: the fixture's is 72, where a kernel that rounds its tile up and
forgets the mask writes past the row.

The subtraction ``x[low] cos - x[high] sin`` cancels, so the operator's
relative factor ``2 gamma(2) + u`` (`RopeApply::tolerance`,
``rounding(2, f32, out)``) is applied to the **terms**
(``|low cos| + |high sin|``) rather than to the result: the forward bound for
a cancelling sum, and stronger than the operator's where the element does not
cancel. A channel past the rotation is held to its input at the same factor,
on an output pre-filled with NaN. The store is checked apart from the
arithmetic (:mod:`tests.python.kernels.rope.rounding`).
"""

import math

import pytest
import torch

from tests.python.kernels.rope.rounding import (
    TORCH_TYPES,
    UNIT_ROUNDOFF,
    gamma,
    outside,
    stored_copy,
)

pytestmark = pytest.mark.unit

#: The seed every case draws its values and angles from.
SEED = 23

#: The cases, each named for the mistake it catches: the output and input
#: types, the pairing, the head width, the heads, the turned channels, the
#: rows one program covers, the heads' tile and the tokens.
CASES = {
    "the fixture's cell, a 72-channel head and a tail of rows": (
        "bf16",
        "bf16",
        False,
        72,
        16,
        72,
        8,
        128,
        11,
    ),
    "the half pairing": ("f32", "bf16", False, 64, 8, 64, 4, 64, 6),
    "the interleaved pairing on the same data": (
        "f32",
        "bf16",
        True,
        64,
        8,
        64,
        4,
        64,
        6,
    ),
    "a partial rotation, 96 of 128": (
        "bf16",
        "f32",
        False,
        128,
        32,
        96,
        2,
        64,
        5,
    ),
    "a quarter rotation, 32 of 64": ("f16", "bf16", True, 64, 4, 32, 1, 64, 3),
    "one token against a program's whole block of rows": (
        "f32",
        "f32",
        False,
        96,
        2,
        96,
        8,
        64,
        1,
    ),
}


def factor(output):
    """`RopeApply::tolerance`'s relative factor: two products, one rounding
    out.
    """  # noqa: D205
    return 2.0 * gamma(2) + UNIT_ROUNDOFF[output]


def launch(
    kernels, x, cosine, sine, out, rotated, rows, heads_tile, interleaved
):
    """One launch of the program the descriptor selects: the narrowed one,
    which takes the turned width as a word, when the rotation is partial.
    """  # noqa: D205
    tokens, heads, head_dim = x.shape
    narrowed = rotated != head_dim
    words = [x, cosine, sine, out, tokens, 0] + ([rotated] if narrowed else [])
    kernel = (
        kernels._rope_apply_narrowed_kernel
        if narrowed
        else kernels._rope_apply_full_kernel
    )
    kernel[((tokens + rows - 1) // rows, 1, 1)](
        *words,
        ROTATED=rotated,
        PAIRS=rotated // 2,
        PAIRS_TILE=1 << (rotated // 2 - 1).bit_length(),
        HEAD_DIM=head_dim,
        HEADS=heads,
        ROWS=rows,
        HEADS_TILE=heads_tile,
        INTERLEAVED=interleaved,
        PDL=False,
    )


def reference(x, cosine, sine, rotated, interleaved):
    """The expected output, and the terms each element's error is measured
    against.
    """  # noqa: D205
    x = x.to(torch.float64)
    half = rotated // 2
    if interleaved:
        low_column = 2 * torch.arange(half)
        high_column = low_column + 1
    else:
        low_column = torch.arange(half)
        high_column = low_column + half
    cos = cosine.to(torch.float64).unsqueeze(1)
    sin = sine.to(torch.float64).unsqueeze(1)
    expected = x.clone()
    scale = x.abs().clone()
    low, high = x[..., low_column], x[..., high_column]
    expected[..., low_column] = low * cos - high * sin
    expected[..., high_column] = high * cos + low * sin
    scale[..., low_column] = (low * cos).abs() + (high * sin).abs()
    scale[..., high_column] = (high * cos).abs() + (low * sin).abs()
    return expected, scale


@pytest.mark.parametrize("case", CASES)
def test_the_rotation_and_its_store_match_the_reference(
    case, torch_device, kernel_source
):
    output, dtype, interleaved, head_dim, heads, rotated, rows, tile, tokens = (
        CASES[case]
    )
    torch.manual_seed(SEED)
    x = torch.randn(tokens, heads, head_dim).to(TORCH_TYPES[dtype])
    angle = torch.rand(tokens, rotated // 2) * 2.0 * math.pi
    cosine, sine = torch.cos(angle), torch.sin(angle)
    kernels = kernel_source("rope_apply")

    results = {}
    for stored in dict.fromkeys((torch.float32, TORCH_TYPES[output])):
        out = torch.full(
            (tokens, heads, head_dim),
            float("nan"),
            dtype=stored,
            device=torch_device,
        )
        launch(
            kernels,
            x.to(torch_device),
            cosine.to(torch_device),
            sine.to(torch_device),
            out,
            rotated,
            rows,
            tile,
            interleaved,
        )
        results[stored] = out.cpu()

    where = (
        f"{case}: {output} out of {dtype} in, {tokens} tokens, {heads} heads "
        f"of {head_dim}, {rotated} turned "
        f"{'interleaved' if interleaved else 'half'}"
    )
    expected, scale = reference(x, cosine, sine, rotated, interleaved)
    wide = results[torch.float32]
    relative = factor(torch.float32)
    missed = outside(
        (wide.to(torch.float64) - expected).abs(), relative * scale
    )
    if missed.any():
        position = int(missed.reshape(-1).nonzero()[0])
        pytest.fail(
            f"{where}: the arithmetic is outside the bound at "
            f"{int(missed.sum())} of {wide.numel()} elements; first at "
            f"{position}: {float(wide.reshape(-1)[position])} vs "
            f"{float(expected.reshape(-1)[position])}, terms "
            f"{float(scale.reshape(-1)[position])}, factor {relative:.3e}"
        )

    if output != "f32":
        stored_copy(where, results[TORCH_TYPES[output]], wide)
