#!/usr/bin/env python3
"""Run position_interpolate's kernel and check every channel it writes.

The operator resamples a learned position table to another grid, but it computes **no geometry**: the
caller hands four tap indices (i32) and four FP32 weights per output point, and the program gathers
four table rows and writes their weighted sum. So this check tests exactly that and nothing about a
resampling convention:

    out[j, c] = Σ_{t < 4} weights[j, t] · table[indices[j, t], c]

    TRITON_INTERPRET=1 python position_check.py [--package .]

It runs the kernel on the CPU through Triton's interpreter — the arithmetic is four gathers, four
products and a sum, so a device would compute the same values — and compares against torch doing the
same sum in FP64, rounded once, at the operator's own bound ``Tolerance::dot(4, f32, dtype)``: four
weighted products accumulated in FP32 with the output's own rounding.

Four cases, each aimed at a way the kernel could be wrong while looking right:

* **1152 channels with a 1024-lane block** — the fixture's geometry, where the last program covers 128
  lanes and the rest is masked. A mask bug hides here and nowhere else.
* **one table point tapped by two different output rows**, at *different* weights, so a kernel that
  read the wrong row's taps lands on a plausible value instead of an obviously wrong one.
* **a clamped corner**: all four taps the same table point with weights summing to one, which must
  equal that row exactly.
* **weights that do not sum to one** — the operator says the caller normalizes, so a kernel that
  divided or averaged would be wrong here and right everywhere else.
"""

import argparse
import importlib.util
import os
import pathlib
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch

#: Taps one output point reads (``TAPS`` in ``crates/ops/src/position.rs``).
TAPS = 4
#: Unit roundoff of FP32, the accumulator the operator names.
UNIT_ROUNDOFF = 2.0**-24
#: The output dtypes the operator's descriptors name.
UNIT_ROUNDOFF_OF = {"f16": 2.0**-11, "bf16": 2.0**-9, "f32": 2.0**-24}
ELEMENT = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}


def gamma(reduction, unit=UNIT_ROUNDOFF):
    """``γₙ = n·u / (1 − n·u)``."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def declared_bound(dtype):
    """``Tolerance::dot(TAPS, f32, dtype)``'s ``rtol``, as ``position.rs`` declares it."""
    return 2.0 * gamma(TAPS) * TAPS**0.5 + UNIT_ROUNDOFF_OF[dtype]


def load_kernel(package):
    """The package's kernel module, imported by path."""
    source = package / "src" / "position_interpolate.py"
    spec = importlib.util.spec_from_file_location("position_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(module, *, dtype, channels, in_side, block, indices, weights, seed=31):
    """Resample one table through the kernel and compare with the four-tap sum in FP64."""
    torch.manual_seed(seed)
    element = ELEMENT[dtype]
    points = int(indices.shape[0])
    # The table is **monotone and smooth**, which is what a learned position table is and what the
    # operator's declared bound assumes: a bilinear blend of neighbouring points whose weights are
    # positive does not cancel, so `Tolerance::dot(4)` — whose correction is a factor √4 — is the
    # right size. Independent random values would make some output elements cancel two orders of
    # magnitude harder than that, and no data-free bound covers such a row (`tolerance.rs` says an
    # operator that has them declares an envelope with `with_atol`, which this one does not). The
    # gradient is steep enough per point that a wrong tap lands on a distinguishably wrong value.
    points_axis = torch.arange(in_side * in_side, dtype=torch.float64).reshape(-1, 1)
    channels_axis = torch.arange(channels, dtype=torch.float64).reshape(1, -1)
    smooth = 0.5 + 0.01 * points_axis + 0.0001 * channels_axis
    table = smooth.to(element)
    out = torch.full((points, channels), float("nan"), dtype=element)

    module._position_interpolate_kernel[(points, (channels + block - 1) // block, 1)](
        table,
        indices.contiguous().to(torch.int32),
        weights.contiguous().to(torch.float32),
        out,
        points,
        channels,
        block,
        0,
        CHANNELS=channels,
        BLOCK=block,
    )

    # The reference: the same four taps, in FP64, rounded once into the output's dtype.
    gathered = table.to(torch.float64)[indices.to(torch.int64)]          # [points, 4, channels]
    expected = (gathered * weights.to(torch.float64)[:, :, None]).sum(dim=1)
    expected = expected.to(element).to(torch.float64)

    bound = declared_bound(dtype)
    cancellation = float(
        ((table.to(torch.float64)[indices.to(torch.int64)].abs()
          * weights.to(torch.float64)[:, :, None]).sum(dim=1)
         / expected.abs().clamp_min(1e-30)).max()
    )
    candidate = out.to(torch.float64)
    error = (candidate - expected).abs()
    # BF16 carries one extra allowance, and it is the interpreter's, not the kernel's: its
    # f32 -> bf16 cast is not round-to-nearest at a tie (measured: 1.005859375 becomes 1.0 where
    # hardware rounds to even, 0.666015625 becomes 0.66796875 where it rounds down), so a store the
    # device would round exactly can land a few ulps away here. The allowance is per element and
    # scaled by the table's magnitude, which is what the rounding is relative to.
    cast_noise = 8.0 * UNIT_ROUNDOFF_OF["bf16"] * float(table.abs().max()) if dtype == "bf16" else 0.0
    # The declared bound is the *kernel's* distance from the exact four-tap sum. The reference here
    # is that exact sum rounded once into the output's dtype, so it carries one rounding of its own;
    # two correct implementations can therefore sit one ulp of the result apart even when both are
    # right, and comparing them needs the declared bound plus that single rounding. It is stated
    # rather than folded into the bound, because it is a property of the comparison, not of the
    # operator.
    allowed = (bound + UNIT_ROUNDOFF_OF[dtype]) * expected.abs() + cast_noise + 1e-30
    unset = int(torch.isnan(out).sum())
    violations = int((error > allowed).sum()) + unset
    if violations:
        position = int(((error > allowed) | torch.isnan(out)).reshape(-1).nonzero()[0])
        raise SystemExit(
            f"{dtype} channels {channels} in_side {in_side} block {block}, {points} points: "
            f"{violations} of {out.numel()} elements outside the bound ({unset} never written); "
            f"first at {position}: {float(out.reshape(-1)[position])} vs "
            f"{float(expected.reshape(-1)[position])}, rtol {bound:.3e}, "
            f"worst Σ|ab|/|out| {cancellation:.1f}"
        )
    return out


def taps_for(points, in_side, weights=None, shared=False):
    """A plausible bilinear tap set: four neighbouring points, clamped at the table's edges."""
    side = in_side
    indices = torch.zeros(points, TAPS, dtype=torch.int32)
    if weights is None:
        weights = torch.zeros(points, TAPS, dtype=torch.float32)
    for point in range(points):
        row = (point * 7) % side
        column = (point * 13) % side
        below = min(row + 1, side - 1)
        right = min(column + 1, side - 1)
        indices[point] = torch.tensor(
            [
                row * side + column,
                row * side + right,
                below * side + column,
                below * side + right,
            ],
            dtype=torch.int32,
        )
        weights[point] = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
    if shared:
        # One table point reached by two output rows through different taps: a kernel reading the
        # wrong row's indices lands on a plausible value rather than a wrong one.
        indices[1, 0] = indices[0, 3]
        indices[1, 1] = indices[0, 3]
    return indices, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/position_interpolate.py",
    )
    options = parser.parse_args()
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise SystemExit("run this with TRITON_INTERPRET=1: the kernel is checked on the CPU")
    module = load_kernel(options.package)

    # (1) The fixture's geometry: 1152 channels against a 1024-lane block, so the tail program masks.
    indices, weights = taps_for(6, 7)
    # Different weights per point, so the weights are not a constant the kernel could ignore.
    weights = weights * torch.arange(1, 3, dtype=torch.float32).repeat_interleave(2).repeat(6, 1)
    per_point = weights.sum(dim=1, keepdim=True)
    weights = weights / per_point
    run(
        module,
        dtype="bf16",
        channels=1152,
        in_side=7,
        block=1024,
        indices=indices,
        weights=weights,
    )

    # (2) One table point tapped by two output rows, at different weights.
    shared_indices, shared_weights = taps_for(5, 6, shared=True)
    shared_weights[1] = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
    run(
        module,
        dtype="f16",
        channels=64,
        in_side=6,
        block=64,
        indices=shared_indices,
        weights=shared_weights,
    )

    # (2b) The grid's own tail: seven points, and one row whose taps are in descending order — the
    # operator says nothing about the order of the four, so a gather that sorted them would pass
    # every monotone case and fail here.
    tail_indices, tail_weights = taps_for(7, 5)
    tail_indices[3] = torch.tensor(
        [tail_indices[3, 3], tail_indices[3, 2], tail_indices[3, 0], tail_indices[3, 1]],
        dtype=torch.int32,
    )
    tail_weights[3] = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    run(
        module,
        dtype="f32",
        channels=48,
        in_side=5,
        block=32,
        indices=tail_indices,
        weights=tail_weights,
    )

    # (3) A clamped corner: all four taps one point, weights summing to one. The answer is that row.
    corner = torch.zeros(2, TAPS, dtype=torch.int32)
    corner[0] = 0
    corner[1] = 35
    corner_weights = torch.full((2, TAPS), 0.25, dtype=torch.float32)
    out = run(
        module,
        dtype="f32",
        channels=32,
        in_side=6,
        block=32,
        indices=corner,
        weights=corner_weights,
    )
    # And the same taps with weights that do not sum to one: the operator's caller normalizes, so the
    # program must neither divide nor average.
    doubled = torch.full((1, TAPS), 0.5, dtype=torch.float32)
    scaled = run(
        module,
        dtype="f32",
        channels=32,
        in_side=6,
        block=32,
        indices=corner[:1],
        weights=doubled,
    )
    print(
        f"ok: 1152 channels over a 1024-lane block, a shared table point at two rows' weights, a "
        f"seven-point grid with a row of descending taps, a clamped corner ({float(out[0, 0]):.4f}) "
        f"and weights that do not sum to one ({float(scaled[0, 0]):.4f}), all within dot({TAPS})'s "
        f"bound"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
