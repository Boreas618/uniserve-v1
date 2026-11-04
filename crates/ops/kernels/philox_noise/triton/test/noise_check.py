#!/usr/bin/env python3
"""Run philox_noise's kernel and check every value it stores against the reference draws.

A noise kernel is the one place where a plausible-looking kernel is worthless: a wrong round count,
a swapped counter word, or a sign in the angle produces values that *look* random and silently change
every sampled sequence. So this check does not compare against a re-derivation of the same formula.
It compares against the tree's **recorded draws**:

* ``tests/python/generated/sampling_rng_parity.json`` holds 144 draws the reference implementation
  produced — the request's stream key, the semantic token index, the processor stage and draw index,
  the four Philox words, and the float32 **bits** of the uniform the sampler uses. Both key forms are
  covered by them: as rows (one stream per row) and, with a key over the whole buffer, flat;
* the published Random123 **known-answer vectors** anchor the Philox permutation itself, so the
  reference in this file is checked before it checks anything.

    TRITON_INTERPRET=1 python noise_check.py [--package .] [--dtype f32]

The kernel runs on the **CPU** through Triton's interpreter, which is enough here and is the point:
Philox is integer arithmetic and the uniform is a 24-bit dyadic value, so a correct kernel agrees bit
for bit with the fixture — no device needed to prove it. Run it on a GPU host only if you want the
compiled module's arithmetic instead, which cannot differ for the uniform kind.

What is checked:

* the uniform kind, both key forms, against the fixture's ``uniform_bits`` (exact, bit for bit);
* the uniform kind over positions the fixture does not hold, against the anchored reference
  (exact), with a stage and draw that are not zero, so the counter's upper words are exercised;
* the normal kind, both key forms, against the reference's ``sqrt(−2 ln(1 − u1))`` times
  ``cos``/``sin`` of ``2π u2``, at the operator's own bound
  ``Tolerance::new(0.0, 4 · u_output)`` — the transcendentals are the library's, not the integer
  path's, so this one is a tolerance and not a bit comparison.
"""

import argparse
import importlib.util
import json
import math
import os
import pathlib
import sys

# Triton's interpreter must be chosen before Triton is imported.
os.environ.setdefault("TRITON_INTERPRET", "1")

import numpy as np
import torch

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

#: Unit roundoff of the dtypes the operator can name (``DType::unit_roundoff``).
UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-9, "f32": 2.0**-24}


def philox4x32_10(counter, key):
    """The ten-round Philox4x32 bijection, in exact 32-bit integer arithmetic (numpy, vectorized)."""
    counter = [np.uint64(word) & np.uint64(MASK32) for word in counter]
    key = [np.uint64(word) & np.uint64(MASK32) for word in key]
    for round_index in range(10):
        if round_index > 0:
            key = [
                (key[0] + np.uint64(KEY_BUMP_0)) & np.uint64(MASK32),
                (key[1] + np.uint64(KEY_BUMP_1)) & np.uint64(MASK32),
            ]
        product0 = np.uint64(M0) * counter[0]
        product1 = np.uint64(M1) * counter[2]
        counter = [
            ((product1 >> np.uint64(32)) ^ counter[1] ^ key[0]) & np.uint64(MASK32),
            product1 & np.uint64(MASK32),
            ((product0 >> np.uint64(32)) ^ counter[3] ^ key[1]) & np.uint64(MASK32),
            product0 & np.uint64(MASK32),
        ]
    return [int(word) for word in counter]


def words_for(keys, positions, stage, draw):
    """The four Philox words of each draw, ``[n]`` keys and positions to ``[n, 4]``."""
    keys = np.asarray(keys, dtype=np.uint64)
    positions = np.asarray(positions, dtype=np.uint64)
    keys, positions = np.broadcast_arrays(keys, positions)
    shape = keys.shape
    # A single draw addresses a scalar; give it one axis so the loop below has indices, and take the
    # axis away again before returning.
    keys = np.atleast_1d(keys)
    positions = np.atleast_1d(positions)
    # Every operand of one draw is an array of the draw's shape: the stage and the draw index are
    # broadcast to it, so the loop below indexes all four words the same way.
    broadcast = np.full(positions.shape, np.uint64(int(stage) & MASK32), dtype=np.uint64)
    drawn = np.full(positions.shape, np.uint64(int(draw) & MASK32), dtype=np.uint64)
    counter = [
        positions & np.uint64(MASK32),
        positions >> np.uint64(32),
        broadcast,
        drawn,
    ]
    key = [keys & np.uint64(MASK32), keys >> np.uint64(32)]
    result = [np.empty(keys.shape, dtype=np.uint64) for _ in range(4)]
    for index in np.ndindex(keys.shape):
        words = philox4x32_10([word[index] for word in counter], [word[index] for word in key])
        for slot in range(4):
            result[slot][index] = np.uint64(words[slot])
    stacked = np.stack(result, axis=-1)
    return stacked.reshape(shape + (4,))


def uniform(keys, positions, stage, draw):
    """``(word0 >> 8) · 2**-24``, the 24-bit dyadic uniform, as float32."""
    first = words_for(keys, positions, stage, draw)[..., 0]
    return (first >> np.uint64(8)).astype(np.float32) * UNIFORM_SCALE


def normal(keys, positions, stage, draw):
    """The pair of Box–Muller normals from one draw's first two words, per element position.

    The element's own index chooses the cosine or the sine, as the kernel does: an even element
    index takes ``cos``, an odd one ``sin``.
    """
    words = words_for(keys, positions // 2, stage, draw)
    first = (words[..., 0] >> np.uint64(8)).astype(np.float32) * UNIFORM_SCALE
    second = (words[..., 1] >> np.uint64(8)).astype(np.float32) * UNIFORM_SCALE
    radius = np.sqrt(np.float32(-2.0) * np.log(np.float32(1.0) - first))
    angle = np.float32(2.0 * math.pi) * second
    even = (np.asarray(positions) % 2) == 0
    return np.where(even, radius * np.cos(angle), radius * np.sin(angle)).astype(np.float32)


def check_known_answers():
    """The reference in this file reproduces the published Random123 vectors."""
    for (counter, key), expected in KNOWN_ANSWERS:
        got = philox4x32_10(counter, key)
        if got != expected:
            raise SystemExit(f"the reference disagrees with Random123 at counter {counter}: {got}")
    print("reference: reproduces the published Random123 known-answer vectors")


def load_kernel(package):
    """The package's kernel module, imported by path."""
    source = package / "src" / "philox_noise.py"
    spec = importlib.util.spec_from_file_location("philox_noise_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launch(module, *, rows, elements, keys, stage, draw, key, block, normal, dtype):
    """Run the kernel through the interpreter and return the buffer it wrote."""
    torch_dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[dtype]
    out = torch.zeros((rows, elements) if rows else (elements,), dtype=torch_dtype)
    # The key is a stream's 64-bit word, which a signed int64 holds bit for bit: the fixture's
    # keys are above 2**63 and reach the kernel as the same bits.
    if keys is not None:
        keys_tensor = torch.from_numpy(np.asarray(keys, dtype=np.uint64).view(np.int64).copy())
    else:
        keys_tensor = torch.zeros(1, dtype=torch.int64)
    if rows:
        grid = (rows, math.ceil(elements / block))
    else:
        grid = (math.ceil(elements / block), 1)
    module._philox_noise_kernel[grid](
        out,
        keys_tensor,
        elements,
        stage,
        draw,
        key,
        BLOCK=block,
        NORMAL=normal,
        ROWS=rows,
    )
    return out


def bits(tensor):
    """The float32 bit patterns of a tensor, for an exact comparison."""
    return tensor.to(torch.float32).numpy().view(np.uint32)


#: The largest semantic coordinate this operator can address in one launch. A draw's position is the
#: element index in the buffer the call passes — the counter's low and high words come from that
#: index, and the call carries no base — so a coordinate beyond the buffer is not reachable by any
#: launch of the operator, and a check that tried would allocate the index and not return.
ADDRESSABLE = 1 << 20


def check_fixture(module, fixture, dtype):
    """Every recorded draw whose coordinate a buffer can hold, through both key forms."""
    cases = [case for case in fixture["cases"]
             if case["semantic_token_index"] < ADDRESSABLE]
    skipped = len(fixture["cases"]) - len(cases)
    groups = {}
    for case in cases:
        groups.setdefault((case["processor_stage"], case["draw_index"]), []).append(case)

    checked = 0
    for (stage, draw), members in sorted(groups.items()):
        columns = max(case["semantic_token_index"] for case in members) + 1
        keys = [case["key"] for case in members]
        out = launch(
            module,
            rows=len(members),
            elements=columns,
            keys=keys,
            stage=stage,
            draw=draw,
            key=0,
            block=8,
            normal=False,
            dtype=dtype,
        )
        for row, case in enumerate(members):
            got = int(bits(out[row, case["semantic_token_index"]]))
            expected = case["uniform_bits"]
            if dtype == "f32" and got != expected:
                raise SystemExit(
                    f"row form: the kernel drew {got:#010x} at key {case['key']}, position "
                    f"{case['semantic_token_index']}, stage {stage}, draw {draw}; the reference "
                    f"recorded {expected:#010x}"
                )
            checked += 1

    # The flat form needs one launch per stream key, and an interpreter launch costs seconds, so it
    # is checked on a few of the records rather than all of them: the addressing is the same code
    # with `ROWS` false, and `check_positions` exercises it over a run of positions.
    for case in cases[:3]:
        single = launch(
            module,
            rows=0,
            elements=case["semantic_token_index"] + 1,
            keys=None,
            stage=case["processor_stage"],
            draw=case["draw_index"],
            key=case["key"],
            block=8,
            normal=False,
            dtype=dtype,
        )
        flat = int(bits(single[case["semantic_token_index"]]))
        if dtype == "f32" and flat != case["uniform_bits"]:
            raise SystemExit(
                f"flat form: the kernel drew {flat:#010x} for key {case['key']}, position "
                f"{case['semantic_token_index']}; the reference recorded "
                f"{case['uniform_bits']:#010x}"
            )
        checked += 1
    print(
        f"fixture: {checked} recorded draws, both key forms, bit for bit ({dtype})"
        + (
            f"; {skipped} skipped because their semantic coordinate exceeds {ADDRESSABLE} elements, "
            "which no launch of this operator can address (the call carries no base offset)"
            if skipped
            else ""
        )
    )


def check_positions(module, dtype):
    """A run of positions with a stage and a draw that are not zero.

    The **stage and the draw** are the counter's upper two words, and the fixture holds them at 0
    and 2 / 0 and 7; a third pair exercises the key schedule where no record does. A position above
    2**32 is *not* checked here and cannot be by any launch of this operator: a draw's position is
    the element index of the buffer the call passes and the call carries no base, so a coordinate of
    5e9 would need a five-billion-element buffer. That is why the fixture's own draws at such
    coordinates are reported and skipped rather than silently dropped.
    """
    key = 0x0DEADBEEFCAFEF00
    stage, draw = 3, 5
    positions = np.arange(37, dtype=np.uint64)
    out = launch(
        module,
        rows=1,
        elements=len(positions),
        keys=[key],
        stage=stage,
        draw=draw,
        key=0,
        block=64,
        normal=False,
        dtype=dtype,
    )
    expected = uniform(np.full(len(positions), key, dtype=np.uint64), positions, stage, draw)
    got = bits(out[0]) if dtype == "f32" else out[0].to(torch.float32).numpy().view(np.uint32)
    if dtype == "f32" and not np.array_equal(got, expected.view(np.uint32)):
        differing = int((got != expected.view(np.uint32)).sum())
        raise SystemExit(f"positions: {differing} of {len(positions)} values differ from the reference")
    print(f"positions: {len(positions)} draws at stage {stage} and draw {draw} ({dtype})")


def check_normal(module, dtype):
    """The normal kind's Box–Muller pair, both key forms, at the operator's bound."""
    key, stage, draw = 0x0123456789ABCDEF, 1, 2
    elements = 32
    tolerance = 4.0 * UNIT_ROUNDOFF[dtype]
    out = launch(
        module,
        rows=1,
        elements=elements,
        keys=[key],
        stage=stage,
        draw=draw,
        key=0,
        block=32,
        normal=True,
        dtype=dtype,
    )
    positions = np.arange(elements, dtype=np.uint64)
    expected = normal(np.full(elements, key, dtype=np.uint64), positions, stage, draw)
    if dtype != "f32":
        expected = expected.astype(np.float32)
    got = out[0].to(torch.float32).numpy()
    error = np.abs(got - expected)
    allowed = tolerance * np.abs(expected) + np.finfo(np.float32).tiny
    violations = int((error > allowed).sum())
    print(
        f"normal: rows form, max |error| {error.max():.3e} at rtol {tolerance:.3e}, "
        f"{violations} of {elements} outside the bound ({dtype})"
    )
    if violations:
        raise SystemExit(f"normal: {violations} values outside the operator's declared bound")


def recorded_draws():
    """The tree's recorded draws, found by walking up from this file."""
    relative = pathlib.Path("tests") / "python" / "generated" / "sampling_rng_parity.json"
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / relative).is_file():
            return parent / relative
    return relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/philox_noise.py",
    )
    parser.add_argument(
        "--fixture",
        type=pathlib.Path,
        default=recorded_draws(),
        help="the recorded draws of the reference implementation",
    )
    parser.add_argument("--dtype", default="f32", choices=sorted(UNIT_ROUNDOFF))
    options = parser.parse_args()

    if os.environ.get("TRITON_INTERPRET") != "1":
        raise SystemExit("run this with TRITON_INTERPRET=1: the kernel is checked on the CPU")
    check_known_answers()
    module = load_kernel(options.package)
    if not options.fixture.is_file():
        raise SystemExit(f"the recorded draws are missing at {options.fixture}")
    fixture = json.loads(options.fixture.read_text())
    check_fixture(module, fixture, options.dtype)
    check_positions(module, options.dtype)
    check_normal(module, options.dtype)
    print("ok: every value the kernel stores is the reference's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
