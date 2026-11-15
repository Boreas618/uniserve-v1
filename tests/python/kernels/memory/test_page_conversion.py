"""The page_conversion program decodes one field of a staged page exactly.

A page stored in another representation is staged in the conversion scratch
in the host page encoding: layer-major keys ``[layers, block_size, heads,
head_dim]``, values in the same shape, and for an FP8 page FP32 scales ``[2,
layers, groups]`` (one group per source pool's head range; the host page's
own tail is ``groups = 1``). The program decodes one field into the FP32
scratch block ``cache_copy_region`` then writes units from. Two things in it
can be wrong without a build noticing:

- **which scale a row takes.** A row of layer ``l`` of field ``f`` takes the
  scale ``[f, l, head // scale_heads]``; another layer's or field's entry
  scales a layer by a foreign factor, and every number still looks like a
  number. Every scale entry of a case is a distinct factor, and the cases
  cover keys and values, one and two scale groups, and several layers.
- **the rounding the decoded value takes.** A scaled page holds ``key /
  scale``; the value times its scale is what the source rank wrote, and it
  rounds through the source's *logical* type so a loaded value equals a
  locally written one. The FP32 block must equal a rounding model exactly,
  and a case proves it separates the models before its answer is trusted.

The scratch is ``20 E + 8 S`` bytes, and every byte outside the FP32 block
(the raw areas and the scale entries) must be untouched.

**Where the cases run.** The program only *loads* the source's elements and
*stores* FP32. Triton's interpreter widens 254 of E4M3's 256 patterns to
torch's own value (the two NaN patterns, which no case carries, read back as
``+-480``), so every case runs on the interpreter too. Its one quirk in play
is the cast into a narrow logical type: a narrow store there truncates
(BF16) where a device rounds to nearest, so on the interpreter the block may
follow either model, and on a device only the operator's nearest rounding.
"""

import pytest
import torch
import triton.language as tl

pytestmark = pytest.mark.unit

FIELD_KEYS = 0
FIELD_VALUES = 1

#: The stored element types: Triton's spelling, the bytes one takes, torch's.
SOURCES = {
    "f8e4m3": (tl.float8e4nv, 1, torch.float8_e4m3fn),
    "bf16": (tl.bfloat16, 2, torch.bfloat16),
}

#: The source's logical types, which decide the decoded value's rounding.
COMPUTES = {
    "bf16": (tl.bfloat16, torch.bfloat16),
    "f16": (tl.float16, torch.float16),
    "f32": (tl.float32, torch.float32),
}

#: The sentinel the FP32 block is primed with, so a stray write shows.
SENTINEL = -1234.5

#: The cases. `layers` and `heads` are compiled into a module; the page's
#: block size and the scale group's heads are launch arguments.
CASES = {
    # The served path's widest source: an FP8 page decoded to FP32, one scale
    # group per field and layer (the host page's own `[2, layers]` tail).
    "fp8 keys decoded to f32 from one source pool": {
        "field": FIELD_KEYS,
        "source": "f8e4m3",
        "compute": "f32",
        "scaled": True,
        "heads": 4,
        "layers": 4,
        "block_size": 16,
        "scale_heads": 4,
        "seed": 11,
    },
    # The values sit eight E bytes along and take the tail's second row, so a
    # program that decoded keys twice would pass every key case.
    "fp8 values decoded to f32 from one source pool": {
        "field": FIELD_VALUES,
        "source": "f8e4m3",
        "compute": "f32",
        "scaled": True,
        "heads": 4,
        "layers": 4,
        "block_size": 16,
        "scale_heads": 4,
        "seed": 12,
    },
    # Two source pools of two heads each: a head that took the other group's
    # scale lands on another factor; the BF16 rounding is measured here.
    "fp8 keys to bf16 from two source pools": {
        "field": FIELD_KEYS,
        "source": "f8e4m3",
        "compute": "bf16",
        "scaled": True,
        "heads": 4,
        "layers": 3,
        "block_size": 8,
        "scale_heads": 2,
        "seed": 13,
    },
    "bf16 keys to bf16 with scales": {
        "field": FIELD_KEYS,
        "source": "bf16",
        "compute": "bf16",
        "scaled": True,
        "heads": 4,
        "layers": 2,
        "block_size": 16,
        "scale_heads": 4,
        "seed": 14,
    },
    # Dense pages carry no scales: the value is widened and rounded.
    "dense bf16 keys to f16": {
        "field": FIELD_KEYS,
        "source": "bf16",
        "compute": "f16",
        "scaled": False,
        "heads": 4,
        "layers": 3,
        "block_size": 16,
        "scale_heads": 0,
        "seed": 15,
    },
    "dense bf16 values to f32": {
        "field": FIELD_VALUES,
        "source": "bf16",
        "compute": "f32",
        "scaled": False,
        "heads": 4,
        "layers": 2,
        "block_size": 16,
        "scale_heads": 0,
        "seed": 16,
    },
    # A BF16 source times an FP32 factor is an FP32 product with no rounding
    # to argue about: the scale rule in exact arithmetic.
    "bf16 keys to f32 with per-layer scales": {
        "field": FIELD_KEYS,
        "source": "bf16",
        "compute": "f32",
        "scaled": True,
        "heads": 4,
        "layers": 4,
        "block_size": 8,
        "scale_heads": 4,
        "seed": 17,
    },
    "bf16 values to f32 from two source pools": {
        "field": FIELD_VALUES,
        "source": "bf16",
        "compute": "f32",
        "scaled": True,
        "heads": 4,
        "layers": 3,
        "block_size": 4,
        "scale_heads": 2,
        "seed": 18,
    },
}

#: The head width every case decodes.
HEAD_DIM = 64


def launch(kernels, scratch, case, elements, scale_entries):
    """One launch: a program per token row of every layer."""
    source, source_bytes, _ = SOURCES[case["source"]]
    kernels._page_conversion_kernel[(case["layers"] * case["block_size"], 1)](
        scratch,
        elements,
        scale_entries,
        case["block_size"],
        case["scale_heads"] if case["scaled"] else 0,
        FIELD=case["field"],
        SOURCE=source,
        COMPUTE=COMPUTES[case["compute"]][0],
        HEAD_DIM=HEAD_DIM,
        HEADS=case["heads"],
        LAYERS=case["layers"],
        SCALED=case["scaled"],
        SOURCE_BYTES=source_bytes,
    )


def groups_of(case):
    """The scale groups of a case's page: one per source pool's head range."""
    if not case["scaled"]:
        return 1
    return -(-case["heads"] // case["scale_heads"])


def staged(case):
    """The scratch bytes a case starts from, and what they hold.

    Returns the scratch, the keys and values in the source's type, the
    scales (or None), the elements of one field and the scale entries.
    """
    shape = (case["layers"], case["block_size"], case["heads"], HEAD_DIM)
    elements = shape[0] * shape[1] * shape[2] * shape[3]
    groups = groups_of(case)
    scale_entries = 2 * case["layers"] * groups if case["scaled"] else 0
    scratch = torch.zeros(20 * elements + 8 * scale_entries, dtype=torch.uint8)

    generator = torch.Generator().manual_seed(case["seed"])
    # The stored elements, in the source's own type: the source wrote
    # `value / scale`.
    stored = SOURCES[case["source"]][2]
    magnitude = {"f8e4m3": 32.0, "bf16": 512.0}[case["source"]]
    keys, values = (
        (
            torch.rand(shape, generator=generator, dtype=torch.float32)
            * magnitude
            - magnitude / 2
        ).to(stored)
        for _ in range(2)
    )
    # Each raw area is eight E bytes, more than one field needs in any source
    # type: the field lands at its start, as the staging copy puts it, and
    # the rest stays zero.
    keys_bytes = keys.contiguous().view(torch.uint8).reshape(-1)
    values_bytes = values.contiguous().view(torch.uint8).reshape(-1)
    scratch[: keys_bytes.numel()] = keys_bytes
    scratch[8 * elements : 8 * elements + values_bytes.numel()] = values_bytes

    # The FP32 block starts at 16 E, the scale entries past 20 E.
    scratch[16 * elements : 20 * elements].view(torch.float32).fill_(SENTINEL)
    scales = None
    if case["scaled"]:
        # Factors near 1/100 as a real FP8 cache carries them, which no
        # narrow mantissa covers, so the product must round; every entry its
        # own factor, so a program reading another layer's, field's or
        # group's cannot pass.
        scales = (
            torch.rand(
                (2, case["layers"], groups),
                generator=generator,
                dtype=torch.float32,
            )
            * 0.02
        )
        scratch[20 * elements : 20 * elements + 4 * scale_entries] = (
            scales.contiguous().view(torch.uint8).reshape(-1)
        )
    return scratch, keys, values, scales, elements, scale_entries


def decoded(case, keys, values, scales):
    """The decoded product before the logical type's rounding."""
    field = case["field"]
    product = (keys if field == FIELD_KEYS else values).to(torch.float32)
    if case["scaled"]:
        for head in range(case["heads"]):
            # The `[layers]` factors of this field and head's group,
            # broadcast over tokens and features.
            factor = scales[field, :, head // case["scale_heads"]]
            product[:, :, head, :] *= factor[:, None, None]
    return product


def truncation_model(product, compute):
    """The product rounded toward zero to the logical type: the other model a
    narrow cast may follow.
    """  # noqa: D205
    if compute == "bf16":
        bits = product.contiguous().view(torch.int32) & ~0xFFFF
        return bits.view(torch.float32)
    narrow = product.to(torch.float16).view(torch.int16)
    # Clearing the low thirteen bits of a half's pattern lowers its magnitude
    # to the next representable value at or below it, which is truncation
    # toward zero for a normal half.
    return (narrow & ~0x1FFF).view(torch.float16).to(torch.float32)


def first_difference(got, expected):
    """The count of differing values and the first, as a phrase."""
    apart = (got != expected).reshape(-1)
    position = int(apart.nonzero()[0])
    return (
        f"{int(apart.sum())} of {got.numel()} decoded values differ; first at "
        f"{position}: {float(got.reshape(-1)[position])!r} vs "
        f"{float(expected.reshape(-1)[position])!r}"
    )


@pytest.mark.parametrize("case", CASES)
def test_the_block_is_the_decoded_field_and_nothing_else_moves(
    case, torch_device, interpreted, kernel_source
):
    name, case = case, CASES[case]
    scratch, keys, values, scales, elements, scale_entries = staged(case)
    launched = scratch.clone().to(torch_device)
    launch(
        kernel_source("page_conversion"),
        launched,
        case,
        elements,
        scale_entries,
    )
    launched = launched.cpu()

    # Nothing outside the FP32 block moves: the raw areas hold the source's
    # bytes, and the scale entries are the next launch's input.
    assert torch.equal(launched[: 16 * elements], scratch[: 16 * elements]), (
        f"{name}: the raw areas changed; this program writes only the FP32 "
        "block"
    )
    assert torch.equal(launched[20 * elements :], scratch[20 * elements :]), (
        f"{name}: the scale bytes changed; this program writes only the FP32 "
        "block"
    )

    product = decoded(case, keys, values, scales)
    got = launched[16 * elements : 20 * elements].view(torch.float32)
    got = got.reshape(product.shape)
    narrow = COMPUTES[case["compute"]][1]
    nearest = product.to(narrow).to(torch.float32)
    if torch.equal(nearest, product):
        # The logical type holds every decoded value (the FP32 cases, and a
        # source no wider than its logical type): the cast cannot round, and
        # the answer is exact rather than one of two models.
        assert torch.equal(got, product), (
            f"{name}: the logical type holds every value exactly, yet "
            + first_difference(got, product)
        )
        return

    # A narrow store on the interpreter truncates where a device rounds to
    # nearest; the operator's rule is the nearest rounding.
    truncated = truncation_model(product, case["compute"])
    models = {"nearest": nearest}
    if interpreted:
        models["truncated"] = truncated
        assert not torch.equal(nearest, truncated), (
            f"{name}: the nearest and truncating models are identical, so the "
            "case cannot tell which rounding the program takes"
        )
    if any(torch.equal(got, model) for model in models.values()):
        return
    apart = torch.stack([got != model for model in models.values()]).all(0)
    position = int(apart.reshape(-1).nonzero()[0])
    candidates = ", ".join(
        f"{float(model.reshape(-1)[position])!r} ({label})"
        for label, model in models.items()
    )
    raise AssertionError(
        f"{name}: the FP32 block follows none of the models; first at "
        f"{position}: {float(got.reshape(-1)[position])!r} vs {candidates}"
    )
