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
arithmetic (:mod:`uniserve_testing.kernels.rope.rounding`).

A rotation may turn a span from channel ``first`` on (a latent attention turns
its heads' trailing channels), by the negated angles (the inverse rotation,
``sin -> -sin``), and in place, where the channels outside the span are left
as they are rather than copied: the placed cases hold the span to the same
bound, and the rotation in place to the one into another tensor of the input's
type, bit for bit, with every other channel untouched.
"""

import math

import pytest
import torch
from uniserve_testing.kernels.rope.rounding import (
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
    kernels,
    x,
    cosine,
    sine,
    out,
    rotated,
    rows,
    heads_tile,
    interleaved,
    placement=(0, False),
):
    """One launch of the program the descriptor selects: the narrowed one,
    which takes the turned width as a word, when the rotation is partial.

    ``placement`` is the first turned channel and whether the rotation is the
    inverse one. A launch whose output is its input is in place: the narrowed
    program then leaves the channels it does not turn alone, and copies them
    otherwise, as the package's derivation compiles it.
    """  # noqa: D205
    tokens, heads, head_dim = x.shape
    first, inverse = placement
    narrowed = rotated != head_dim
    words = [x, cosine, sine, out, tokens] + ([rotated] if narrowed else [])
    constants = {
        "ROTATED": rotated,
        "PAIRS": rotated // 2,
        "PAIRS_TILE": 1 << (rotated // 2 - 1).bit_length(),
        "HEAD_DIM": head_dim,
        "HEADS": heads,
        "ROWS": rows,
        "HEADS_TILE": heads_tile,
        "INTERLEAVED": interleaved,
        "INVERSE": inverse,
        "PDL": False,
    }
    if narrowed:
        kernel = kernels._rope_apply_narrowed_kernel
        constants["FIRST"] = first
        constants["COPY"] = x.data_ptr() != out.data_ptr()
    else:
        kernel = kernels._rope_apply_full_kernel
    kernel[((tokens + rows - 1) // rows, 1, 1)](*words, **constants)


def reference(x, cosine, sine, rotated, interleaved, placement=(0, False)):
    """The expected output, and the terms each element's error is measured
    against: the rotation of the channels from ``placement``'s first on, by
    the negated angles for the inverse rotation.
    """  # noqa: D205
    first, inverse = placement
    x = x.to(torch.float64)
    half = rotated // 2
    if interleaved:
        low_column = first + 2 * torch.arange(half)
        high_column = low_column + 1
    else:
        low_column = first + torch.arange(half)
        high_column = low_column + half
    cos = cosine.to(torch.float64).unsqueeze(1)
    sin = sine.to(torch.float64).unsqueeze(1)
    if inverse:
        sin = -sin
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


#: The rotations of a span that does not start at a head's first channel, or
#: turn backwards, each named for the mistake it catches: the element type,
#: the pairing, the head width, the heads, the turned channels, the rows one
#: program covers, the heads' tile, the tokens, the first turned channel and
#: whether the rotation is the inverse one.
PLACED = {
    "a latent's trailing 64 of 512 channels, adjacent pairs": (
        "bf16",
        True,
        512,
        16,
        64,
        8,
        16,
        5,
        448,
        False,
    ),
    "the inverse rotation of the same span, which a forward one fails": (
        "bf16",
        True,
        512,
        16,
        64,
        8,
        16,
        5,
        448,
        True,
    ),
    "an index query's trailing 64 of 128 channels": (
        "f32",
        True,
        128,
        32,
        64,
        4,
        32,
        3,
        64,
        False,
    ),
    "a span inside the head in the half pairing, inverted": (
        "f32",
        False,
        64,
        4,
        16,
        2,
        4,
        7,
        24,
        True,
    ),
}


@pytest.mark.parametrize("case", PLACED)
def test_a_placed_rotation_turns_its_span_and_in_place_leaves_the_rest(
    case, torch_device, kernel_source
):
    (
        dtype,
        interleaved,
        head_dim,
        heads,
        rotated,
        rows,
        tile,
        tokens,
        first,
        inverse,
    ) = PLACED[case]
    torch.manual_seed(SEED)
    element = TORCH_TYPES[dtype]
    x = torch.randn(tokens, heads, head_dim).to(element)
    angle = torch.rand(tokens, rotated // 2) * 2.0 * math.pi
    cosine = torch.cos(angle).to(torch_device)
    sine = torch.sin(angle).to(torch_device)
    kernels = kernel_source("rope_apply")
    placement = (first, inverse)
    where = (
        f"{case}: {dtype}, {tokens} tokens, {heads} heads of {head_dim}, "
        f"{rotated} turned from channel {first}"
        f"{', inverse' if inverse else ''}"
    )

    def into(out, source):
        launch(
            kernels,
            source,
            cosine,
            sine,
            out,
            rotated,
            rows,
            tile,
            interleaved,
            placement,
        )
        return out

    # Into another tensor: the span turned within the operator's bound, every
    # other channel copied, on an output that starts as NaN.
    source = x.to(torch_device)
    wide = into(
        torch.full(x.shape, float("nan"), device=torch_device), source
    ).cpu()
    expected, scale = reference(
        x, cosine.cpu(), sine.cpu(), rotated, interleaved, placement
    )
    relative = factor(torch.float32)
    missed = outside(
        (wide.to(torch.float64) - expected).abs(), relative * scale
    )
    assert not missed.any(), (
        f"{where}: {int(missed.sum())} of {wide.numel()} elements outside the "
        f"bound, first at {int(missed.reshape(-1).nonzero()[0])}"
    )

    # In place: the span is what the same rotation stores into another tensor
    # of the input's type, bit for bit, and every other channel keeps its
    # input's bits, untouched.
    narrow = into(
        torch.full(x.shape, float("nan"), dtype=element, device=torch_device),
        source,
    ).cpu()
    in_place = source.clone()
    into(in_place, in_place)
    in_place = in_place.cpu()
    span = torch.zeros(head_dim, dtype=torch.bool)
    span[first : first + rotated] = True
    assert torch.equal(in_place[..., span], narrow[..., span]), (
        f"{where}: the span rotated in place differs from the rotation into "
        f"another tensor"
    )
    assert torch.equal(in_place[..., ~span], x[..., ~span]), (
        f"{where}: a channel outside the span changed in place"
    )


def test_the_key_of_a_latent_output_inverse_in_place_turns_its_tail(
    torch_device, stated_launch, launch_target
):
    """The key `RopeApplyDescriptor::trailing(true)` prints for a latent
    attention's output, rotated back in place: the compilation it states
    turns the trailing 64 of 512 channels backwards and leaves the rest.
    """  # noqa: D205
    tokens, heads, head_dim, rotated = 3, 16, 512, 64
    stated = stated_launch(
        "rope_apply",
        f"dtype_f32_output_f32_head_dim_{head_dim}_heads_{heads}_rotation_"
        f"interleaved_rows_8_block_128_pdl_false_rotated_{rotated}_first_"
        f"{head_dim - rotated}_inverse_true_inplace_true",
        launch_target,
    )
    assert stated.words == ["x", "cosine", "sine", "out", "tokens", "rotated"]

    torch.manual_seed(SEED)
    x = torch.randn(tokens, heads, head_dim)
    angle = torch.rand(tokens, rotated // 2) * 2.0 * math.pi
    cosine, sine = torch.cos(angle), torch.sin(angle)
    in_place = x.clone().to(torch_device)
    stated.kernel[(1, 1, 1)](
        in_place,
        cosine.to(torch_device),
        sine.to(torch_device),
        in_place,
        tokens,
        rotated,
        **stated.constexprs,
    )

    got = in_place.cpu()
    placement = (head_dim - rotated, True)
    expected, scale = reference(x, cosine, sine, rotated, True, placement)
    missed = outside(
        (got.to(torch.float64) - expected).abs(),
        factor(torch.float32) * scale,
    )
    assert not missed.any(), f"{int(missed.sum())} elements outside the bound"
    assert torch.equal(got[..., : head_dim - rotated], x[..., :-rotated])
