"""The qk_norm_rope program: per-head normalization plus rotation.

One program normalizes and rotates the query heads and then the key heads of a
block of tokens::

    row        = x[t, h, :]                  the whole head, DIM channels
    inverse    = rsqrt(mean(row^2) + eps)    over the whole head
    row        = row inverse weight          one weight per channel
    out[t,h,k] = row[k]                      k >= ROTATED: not turned
    out[t,h,a] = row[a] cos - row[b] sin     the turned pairs
    out[t,h,b] = row[b] cos + row[a] sin

with one unit ``cos``/``sin`` pair per turned feature of the token, ``half``
pairing channel ``i`` with ``i + ROTATED/2`` and ``interleaved`` pairing
``2i`` with ``2i + 1``.

**How a subtractive rotation is checked.** The operator's bound,
``rounding(head_dim, f32, output)`` (`QkNormRope::tolerance`), is relative to
the result, the wrong denominator for a rotation: ``row[a] cos - row[b] sin``
can cancel to a thousandth of its terms. So the comparison applies the same
relative factor ``2 gamma(head_dim) + u`` to the **sum of the magnitudes of
the terms that produced the element**, the textbook bound for a cancelling
sum, and a strictly tighter statement of the operator's claim wherever the
element does not cancel::

    |got - reference| <= (2 gamma(head_dim) + u) (|first cos| + |second sin|)

The store is checked apart from the arithmetic
(:mod:`uniserve_testing.kernels.rope.rounding`).
"""

import math
import struct

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

#: The seed every case draws its values from.
SEED = 19

#: The epsilon the fixture's key names, as its FP32 bits: 1e-5.
FIXTURE_EPSILON = struct.unpack("<f", struct.pack("<I", 0x3727C5AC))[0]

#: The program of each form: with a learned scale or not, and narrowed to a
#: partial rotation or not.
ENTRIES = {
    (True, False): "_qk_norm_rope_kernel",
    (True, True): "_qk_norm_rope_narrowed_kernel",
    (False, False): "_qk_norm_rope_unweighted_kernel",
    (False, True): "_qk_norm_rope_unweighted_narrowed_kernel",
}

#: The cases, each named for the mistake it catches: the output and input
#: types, the pairing, the head width, the query and key heads, the turned
#: channels, the rows one program covers, the tokens, the epsilon, and what
#: the case adds (no learned scale, a zero-centred one, a zero row and a tiny
#: one).
CASES = {
    "the fixture's cell, 56 + 56 heads, 96 of 128 turned": {
        "output": "bf16",
        "dtype": "bf16",
        "interleaved": True,
        "head_dim": 128,
        "heads": (56, 56),
        "rotated": 96,
        "rows": 1,
        "tokens": 2,
        "epsilon": FIXTURE_EPSILON,
    },
    "the half pairing": {
        "output": "f32",
        "dtype": "bf16",
        "interleaved": False,
        "head_dim": 128,
        "heads": (4, 2),
        "rotated": 64,
        "rows": 2,
        "tokens": 3,
        "epsilon": FIXTURE_EPSILON,
    },
    "the interleaved pairing on the same data": {
        "output": "f32",
        "dtype": "bf16",
        "interleaved": True,
        "head_dim": 128,
        "heads": (4, 2),
        "rotated": 64,
        "rows": 2,
        "tokens": 3,
        "epsilon": FIXTURE_EPSILON,
    },
    "a full-width rotation, which skips the first store": {
        "output": "f32",
        "dtype": "f32",
        "interleaved": True,
        "head_dim": 64,
        "heads": (2, 1),
        "rotated": 64,
        "rows": 4,
        "tokens": 6,
        "epsilon": 1e-6,
    },
    "the epsilon, a zero row and a tiny one over the whole head": {
        "output": "bf16",
        "dtype": "bf16",
        "interleaved": False,
        "head_dim": 64,
        "heads": (3, 3),
        "rotated": 32,
        "rows": 2,
        "tokens": 2,
        "epsilon": 1e-1,
        "zeros": True,
        "tiny": True,
    },
    "a zero-centred scale, 16 + 2 heads of 256, a quarter turned": {
        "output": "bf16",
        "dtype": "bf16",
        "interleaved": False,
        "head_dim": 256,
        "heads": (16, 2),
        "rotated": 64,
        "rows": 16,
        "tokens": 5,
        "epsilon": 1e-6,
        "centred": True,
    },
    "the unweighted program, a separate kernel rather than a null scale": {
        "output": "f16",
        "dtype": "f16",
        "interleaved": False,
        "head_dim": 64,
        "heads": (2, 2),
        "rotated": 32,
        "rows": 1,
        "tokens": 3,
        "epsilon": 1e-5,
        "affine": False,
    },
}


def factor(head_dim, output):
    """`QkNormRope::tolerance`'s relative factor: the head's reduction, one
    rounding out.
    """  # noqa: D205
    return 2.0 * gamma(head_dim) + UNIT_ROUNDOFF[output]


def launch(kernels, operands, tokens, epsilon, spec, outputs):
    """One launch of the program the form selects over every query and key
    head, `ceil(heads * tokens / rows)` programs per side.

    `operands` are the query and key, their weights when the form is affine,
    and the tables; `outputs` the query's and the key's.
    """  # noqa: D205
    head_dim, rotated, rows = spec["head_dim"], spec["rotated"], spec["rows"]
    q_heads, k_heads = spec["heads"]
    affine = spec.get("affine", True)
    narrowed = rotated != head_dim
    words = [*operands, *outputs, tokens, epsilon]
    if narrowed:
        words.append(rotated)
    kernel = getattr(kernels, ENTRIES[(affine, narrowed)])
    grid = (tokens * q_heads + rows - 1) // rows
    grid += (tokens * k_heads + rows - 1) // rows
    kernel[(grid, 1, 1)](
        *words,
        Q_HEADS=q_heads,
        K_HEADS=k_heads,
        DIM=head_dim,
        ROTATED=rotated,
        # A lane count is a power of two: 48 pairs take 64 lanes and the last
        # 16 are masked off.
        PAIRS=1 << (rotated // 2 - 1).bit_length(),
        ROWS=rows,
        INTERLEAVED=spec["interleaved"],
        **(
            {"AFFINE": True, "CENTRED": spec.get("centred", False)}
            if affine
            else {}
        ),
        PDL=False,
    )


def reference(side, weight, cosine, sine, spec):
    """One side's expected values, and the sum of the magnitudes of the terms
    that produced each element.
    """  # noqa: D205
    values = side.to(torch.float64)
    inverse = 1.0 / torch.sqrt(
        (values * values).mean(dim=-1, keepdim=True) + spec["epsilon"]
    )
    normalized = values * inverse
    if spec.get("affine", True):
        # A zero-centred scale is `1 + w`.
        normalized = normalized * (
            weight.to(torch.float64) + float(spec.get("centred", False))
        )

    expected = normalized.clone()
    scale = normalized.abs().clone()
    half = spec["rotated"] // 2
    if spec["interleaved"]:
        a = 2 * torch.arange(half)
        b = a + 1
    else:
        a = torch.arange(half)
        b = a + half
    first, second = normalized[..., a], normalized[..., b]
    # One unit pair per turned feature of the token, shared by its heads.
    cos = cosine.to(torch.float64).unsqueeze(1)
    sin = sine.to(torch.float64).unsqueeze(1)
    expected[..., a] = first * cos - second * sin
    expected[..., b] = second * cos + first * sin
    scale[..., a] = (first * cos).abs() + (second * sin).abs()
    scale[..., b] = (second * cos).abs() + (first * sin).abs()
    return expected, scale


@pytest.mark.parametrize("case", CASES)
def test_the_heads_and_their_store_match_the_reference(
    case, torch_device, kernel_source
):
    spec = CASES[case]
    output, dtype, tokens = spec["output"], spec["dtype"], spec["tokens"]
    head_dim, (q_heads, k_heads) = spec["head_dim"], spec["heads"]
    centred = float(spec.get("centred", False))
    affine = spec.get("affine", True)
    element = TORCH_TYPES[dtype]
    torch.manual_seed(SEED)
    query = torch.randn(tokens, q_heads, head_dim)
    key = torch.randn(tokens, k_heads, head_dim)
    if spec.get("zeros"):
        query[0, 0] = 0.0
        key[0, 0] = 0.0
    if spec.get("tiny"):
        query[1 % tokens, 0] *= 1e-6
    query, key = query.to(element), key.to(element)
    angle = torch.rand(tokens, spec["rotated"] // 2) * 2.0 * math.pi
    cosine, sine = torch.cos(angle), torch.sin(angle)
    # A zero-centred scale is stored around zero, so its gain `1 + w` lies
    # where a plain one does.
    q_weight = torch.rand(head_dim) + 0.5 - centred
    k_weight = torch.rand(head_dim) + 0.5 - centred
    operands = [query, key] + ([q_weight, k_weight] if affine else [])
    operands = [tensor.to(torch_device) for tensor in [*operands, cosine, sine]]
    kernels = kernel_source("qk_norm_rope")

    results = {}
    for stored in dict.fromkeys((torch.float32, TORCH_TYPES[output])):
        outputs = [
            torch.full(
                (tokens, heads, head_dim),
                float("nan"),
                dtype=stored,
                device=torch_device,
            )
            for heads in (q_heads, k_heads)
        ]
        launch(kernels, operands, tokens, spec["epsilon"], spec, outputs)
        results[stored] = [out.cpu() for out in outputs]

    where = (
        f"{case}: {output} out of {dtype} in, {tokens} tokens, {q_heads}+"
        f"{k_heads} heads of {head_dim}, {spec['rotated']} turned "
        f"{'interleaved' if spec['interleaved'] else 'half'}"
    )
    relative = factor(head_dim, torch.float32)
    for name, wide, side, weight in zip(
        ("query", "key"),
        results[torch.float32],
        (query, key),
        (q_weight, k_weight),
    ):
        expected, scale = reference(side, weight, cosine, sine, spec)
        missed = outside(
            (wide.to(torch.float64) - expected).abs(), relative * scale
        )
        if missed.any():
            position = int(missed.reshape(-1).nonzero()[0])
            pytest.fail(
                f"{where}: the {name}'s arithmetic is outside the bound at "
                f"{int(missed.sum())} of {wide.numel()} elements; first at "
                f"{position}: {float(wide.reshape(-1)[position])} vs "
                f"{float(expected.reshape(-1)[position])}, terms "
                f"{float(scale.reshape(-1)[position])}, factor "
                f"{relative:.3e}"
            )

    if output != "f32":
        for name, narrow, wide in zip(
            ("query", "key"),
            results[TORCH_TYPES[output]],
            results[torch.float32],
        ):
            stored_copy(f"{where}, the {name}", narrow, wide)
