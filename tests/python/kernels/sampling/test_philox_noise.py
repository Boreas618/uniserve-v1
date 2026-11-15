"""The philox_noise program against the reference implementation's draws.

A noise program is where a plausible-looking program is worthless: a wrong
round count, a swapped counter word or a sign in the angle produces values
that *look* random and silently change every sampled sequence. So the check
does not compare with a re-derivation of the same formula alone. It compares
with the tree's **recorded draws**: ``tests/python/generated/
sampling_rng_parity.json`` holds 144 draws of the reference implementation
(the stream key, the semantic token index, the processor stage and draw
index, the four Philox words and the FP32 bits of the sampler's uniform). The
published Random123 known-answer vectors anchor the Philox permutation of this
module's reference before it checks anything.

What is compared, all in FP32 output:

- the uniform kind in both key forms, as rows (one stream per row) and flat
  (one key over the whole buffer), against the fixture's ``uniform_bits``, bit
  for bit. Philox is integer arithmetic and the uniform a 24-bit dyadic value,
  so the interpreter answers it as a device does;
- the uniform kind over positions the fixture does not hold, against the
  anchored reference, bit for bit, at a stage and draw that are not zero, so
  the counter's upper words take values no record gives them;
- the normal kind against the reference's ``sqrt(-2 ln(1 - u1))`` times the
  cosine (even element) or sine (odd element) of ``2 pi u2``, at the
  operator's bound ``Tolerance::new(0.0, 4 u_out)``: the transcendentals are
  the library's, so this one is a bound and not a bit comparison.

A draw's position is the element index of the buffer the call passes, and the
call carries no base, so a record whose coordinate a buffer cannot hold
(beyond :data:`ADDRESSABLE` elements) is not reachable by any launch of the
operator and is not compared.
"""

import json
import math

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit

#: The published Random123 Philox4x32-10 known-answer vectors.
KNOWN_ANSWERS = [
    (([0, 0, 0, 0], [0, 0]), [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]),
    (
        ([0xFFFFFFFF] * 4, [0xFFFFFFFF] * 2),
        [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD],
    ),
]

M0, M1 = 0xD2511F53, 0xCD9E8D57
KEY_BUMP_0, KEY_BUMP_1 = 0x9E3779B9, 0xBB67AE85
MASK32 = 0xFFFFFFFF
UNIFORM_SCALE = np.float32(1.0 / 16_777_216.0)

#: The unit round-off of FP32 (``ops::tolerance::unit_roundoff``).
UNIT_ROUNDOFF = 2.0**-24

#: The largest semantic coordinate one launch addresses in a check: a
#: coordinate beyond the buffer is reachable by no launch of the operator.
ADDRESSABLE = 1 << 20


#: The recorded draws, relative to the repository.
RECORDED_DRAWS = ("tests", "python", "generated", "sampling_rng_parity.json")


@pytest.fixture(scope="module")
def recorded_draws(repository):
    """The reference implementation's recorded draws."""
    return json.loads(repository.joinpath(*RECORDED_DRAWS).read_text())


def launch(kernels, out, keys, elements, stage, draw, key, block, normal):
    """One launch: a program per row (when `keys` holds one key per row) and
    block of elements, the stream's base counter zero.
    """  # noqa: D205
    rows = out.shape[0] if out.dim() == 2 else 0
    grid = (rows, -(-elements // block)) if rows else (-(-elements // block), 1)
    kernels._philox_noise_kernel[grid](
        out,
        keys,
        elements,
        stage,
        draw,
        key,
        0,
        BLOCK=block,
        NORMAL=normal,
        ROWS=rows,
    )


def philox4x32_10(counter, key):
    """The ten-round Philox4x32 bijection in exact 32-bit integer arithmetic."""
    counter = [int(word) & MASK32 for word in counter]
    key = [int(word) & MASK32 for word in key]
    for round_index in range(10):
        if round_index > 0:
            key = [
                (key[0] + KEY_BUMP_0) & MASK32,
                (key[1] + KEY_BUMP_1) & MASK32,
            ]
        product0 = M0 * counter[0]
        product1 = M1 * counter[2]
        counter = [
            ((product1 >> 32) ^ counter[1] ^ key[0]) & MASK32,
            product1 & MASK32,
            ((product0 >> 32) ^ counter[3] ^ key[1]) & MASK32,
            product0 & MASK32,
        ]
    return counter


def words(key, position, stage, draw):
    """The four Philox words of one draw."""
    counter = [position & MASK32, position >> 32, stage & MASK32, draw & MASK32]
    return philox4x32_10(counter, [key & MASK32, key >> 32])


def uniform(word):
    """``(word >> 8) * 2**-24``, the 24-bit dyadic uniform, as FP32."""
    return np.float32(word >> 8) * UNIFORM_SCALE


def keys_of(keys, device):
    """Stream keys as the program reads them: a signed int64 holds each
    64-bit key bit for bit (the fixture's keys reach above 2**63).
    """  # noqa: D205
    held = np.asarray(keys, dtype=np.uint64).view(np.int64).copy()
    return torch.from_numpy(held).to(device)


def bits(tensor):
    """FP32 values as their bit patterns."""
    return tensor.cpu().numpy().view(np.uint32)


def test_the_reference_reproduces_random123():
    for (counter, key), expected in KNOWN_ANSWERS:
        got = philox4x32_10(counter, key)
        assert got == expected, (
            f"the reference disagrees with Random123 at counter {counter}: "
            f"{[hex(word) for word in got]}"
        )


def test_every_recorded_draw_in_the_row_form(
    recorded_draws, torch_device, kernel_source
):
    """Every addressable record, each stage and draw one launch of a row per
    record.
    """  # noqa: D205
    kernels = kernel_source("philox_noise")
    groups = {}
    for case in recorded_draws["cases"]:
        if case["semantic_token_index"] < ADDRESSABLE:
            stage = (case["processor_stage"], case["draw_index"])
            groups.setdefault(stage, []).append(case)
    assert groups, "the fixture holds no addressable record"

    faults = []
    for (stage, draw), members in sorted(groups.items()):
        columns = max(case["semantic_token_index"] for case in members) + 1
        out = torch.zeros((len(members), columns), device=torch_device)
        keys = keys_of([case["key"] for case in members], torch_device)
        launch(kernels, out, keys, columns, stage, draw, 0, 8, False)
        drawn = bits(out)
        for row, case in enumerate(members):
            got = int(drawn[row, case["semantic_token_index"]])
            if got != case["uniform_bits"]:
                faults.append(
                    f"key {case['key']}, position "
                    f"{case['semantic_token_index']}, stage {stage}, draw "
                    f"{draw}: {got:#010x}, recorded "
                    f"{case['uniform_bits']:#010x}"
                )
    assert not faults, f"{len(faults)} draws differ: " + "; ".join(faults[:6])


def test_recorded_draws_in_the_flat_form(
    recorded_draws, torch_device, kernel_source
):
    """The flat form on a few records: one launch per stream key, and an
    interpreted launch costs seconds. The addressing is the row form's code
    with ``ROWS`` false.
    """  # noqa: D205
    kernels = kernel_source("philox_noise")
    cases = [
        case
        for case in recorded_draws["cases"]
        if case["semantic_token_index"] < ADDRESSABLE
    ][:3]
    keys = torch.zeros(1, dtype=torch.int64, device=torch_device)
    faults = []
    for case in cases:
        position = case["semantic_token_index"]
        out = torch.zeros(position + 1, device=torch_device)
        launch(
            kernels,
            out,
            keys,
            position + 1,
            case["processor_stage"],
            case["draw_index"],
            # The program reads the key as a signed 64-bit word.
            int(np.uint64(case["key"]).view(np.int64)),
            8,
            False,
        )
        got = int(bits(out)[position])
        if got != case["uniform_bits"]:
            faults.append(
                f"key {case['key']}, position {position}: {got:#010x}, "
                f"recorded {case['uniform_bits']:#010x}"
            )
    assert not faults, "; ".join(faults)


def test_a_run_of_positions_at_a_stage_and_draw_no_record_holds(
    torch_device, kernel_source
):
    """The stage and the draw are the counter's upper two words; the fixture
    holds them at 0 and 2 / 0 and 7, and a third pair exercises the key
    schedule where no record does.
    """  # noqa: D205
    key, stage, draw, elements = 0x0DEADBEEFCAFEF00, 3, 5, 37
    out = torch.zeros((1, elements), device=torch_device)
    launch(
        kernel_source("philox_noise"),
        out,
        keys_of([key], torch_device),
        elements,
        stage,
        draw,
        0,
        64,
        False,
    )

    expected = np.array(
        [
            uniform(words(key, position, stage, draw)[0])
            for position in range(elements)
        ],
        dtype=np.float32,
    )
    differing = int((bits(out[0]) != expected.view(np.uint32)).sum())
    assert differing == 0, (
        f"{differing} of {elements} draws at stage {stage} and draw {draw} "
        "differ from the reference"
    )


def test_the_normal_kind_within_the_bound(torch_device, kernel_source):
    """The Box-Muller pair of each draw's first two words: an even element
    takes the cosine, an odd one the sine.
    """  # noqa: D205
    key, stage, draw, elements = 0x0123456789ABCDEF, 1, 2, 32
    out = torch.zeros((1, elements), device=torch_device)
    launch(
        kernel_source("philox_noise"),
        out,
        keys_of([key], torch_device),
        elements,
        stage,
        draw,
        0,
        32,
        True,
    )

    expected = np.empty(elements, dtype=np.float32)
    for position in range(elements):
        first, second, _, _ = words(key, position // 2, stage, draw)
        radius = np.sqrt(
            np.float32(-2.0) * np.log(np.float32(1.0) - uniform(first))
        )
        angle = np.float32(2.0 * math.pi) * uniform(second)
        turn = np.cos(angle) if position % 2 == 0 else np.sin(angle)
        expected[position] = radius * turn
    tolerance = 4.0 * UNIT_ROUNDOFF
    error = np.abs(out[0].cpu().numpy() - expected)
    allowed = tolerance * np.abs(expected) + np.finfo(np.float32).tiny
    # `~(error <= allowed)`: a NaN never compares above the bound.
    outside = int((~(error <= allowed)).sum())
    assert outside == 0, (
        f"{outside} of {elements} normals outside rtol {tolerance:.3e}; max "
        f"|error| {np.nan_to_num(error).max():.3e}"
    )
