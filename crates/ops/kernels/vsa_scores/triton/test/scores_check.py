#!/usr/bin/env python3
"""Run vsa_scores' kernel and check every score it writes against the operator's product.

The kernel is a per-head contraction, so its arithmetic is checkable without a device: Triton's
interpreter executes the kernel body with numpy-backed tensors, one program at a time, and the
result is compared with the operator's definition from ``crates/ops/src/segments.rs`` — one dot
product per (head, query tile, key tile) over the pooled rows, accumulated in FP32, with no scale
and no softmax:

    scores[head, i, j] = Σ_w pooled_query[head, i, w] · pooled_key[head, j, w]

    TRITON_INTERPRET=1 python scores_check.py [--package .] [--heads 3] [--width 128]

What is checked:

* the scores themselves, at the operator's own bound — ``Tolerance::dot(width, f32, f32)``, the
  standard bound for a ``width``-term contraction — computed here by torch in FP64 and rounded once,
  so the reference is exact and the bound is what the operator declares;
* a key-tile count that is not a multiple of the kernel's 16-tile step, so the step's mask is
  exercised rather than implied;
* a contraction narrower than the block of lanes, so the lane mask is exercised — the case where a
  kernel that summed the whole block would still pass a width that fills it;
* the destination's layout, which the operator requires to be the contiguous
  ``[heads, query tiles, key tiles]`` matrix the selection reads: the check reads it back in that
  order, so a transposed or padded store fails rather than scoring plausibly.
"""

import argparse
import importlib.util
import os
import pathlib
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch

#: Unit roundoff of FP32, the accumulator the operator names.
UNIT_ROUNDOFF = 2.0**-24


def gamma(reduction, unit=UNIT_ROUNDOFF):
    """``γₙ = n·u / (1 − n·u)``."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def declared_bound(width):
    """``Tolerance::dot(width, f32, f32)``'s ``rtol``, as ``segments.rs`` declares it."""
    return 2.0 * gamma(width) * width**0.5 + UNIT_ROUNDOFF


def load_kernel(package):
    """The package's kernel module, imported by path."""
    source = package / "src" / "vsa_scores.py"
    spec = importlib.util.spec_from_file_location("vsa_scores_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(module, *, heads, width, block, query_tiles, key_tiles, seed=23):
    """Contract one pair of pooled tilings and compare with the operator's product."""
    torch.manual_seed(seed)
    # Both operands are FP32: that is the type `vsa_prepare_tiles` pools a tile's per-head means
    # in, and the operator refuses anything else, so there is no element type to vary here.
    query = torch.randn(heads, query_tiles, width, dtype=torch.float32)
    key = torch.randn(heads, key_tiles, width, dtype=torch.float32)
    out = torch.full((heads, query_tiles, key_tiles), float("nan"), dtype=torch.float32)

    module._vsa_scores_kernel[(query_tiles, heads, 1)](
        query,
        key,
        out,
        query_tiles,
        key_tiles,
        0,
        WIDTH=width,
        BLOCK=block,
        KEY_TILES=16,
    )

    # The reference: the same operands in FP64, rounded once into FP32.
    expected = torch.einsum(
        "hiw,hjw->hij", query.to(torch.float64), key.to(torch.float64)
    ).to(torch.float32)

    bound = declared_bound(width)
    error = (out - expected).abs()
    allowed = bound * expected.abs()
    unset = int(torch.isnan(out).sum())
    violations = int((error > allowed).sum()) + unset
    if violations:
        position = int(((error > allowed) | torch.isnan(out)).reshape(-1).nonzero()[0])
        raise SystemExit(
            f"heads {heads} width {width} block {block} "
            f"{query_tiles}x{key_tiles} tiles: {violations} of {out.numel()} scores outside the "
            f"bound ({unset} never written); first at {position}: "
            f"{float(out.reshape(-1)[position])} vs {float(expected.reshape(-1)[position])}, "
            f"rtol {bound:.3e}"
        )
    return out.numel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/vsa_scores.py",
    )
    options = parser.parse_args()
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise SystemExit("run this with TRITON_INTERPRET=1: the kernel is checked on the CPU")
    module = load_kernel(options.package)

    checked = 0
    # The fixture's cell, then the two edges: a key count that is not a multiple of the 16-tile
    # step, and a contraction narrower than the block of lanes.
    checked += run(module, heads=2, width=128, block=128, query_tiles=3, key_tiles=7)
    checked += run(module, heads=3, width=64, block=128, query_tiles=2, key_tiles=5)
    checked += run(module, heads=1, width=32, block=128, query_tiles=4, key_tiles=20)
    print(f"ok: {checked} scores within the contraction's bound, both masks exercised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
