#!/usr/bin/env python3
"""Run vsa_scores' kernel and check every score it writes against the operator's product.

The kernel is a per-head contraction, compared with the operator's definition (``VsaScores`` in
``crates/ops/src/vsa.rs``) — one dot product per (head, query tile, key tile) over the pooled rows,
accumulated in FP32 and scaled, with no softmax, over operands in the layout the pooling writes:

    scores[head, i, j] = Σ_w pooled_query[i, head, w] · pooled_key[j, head, w] / sqrt(width)

    python vsa_scores_check.py [--package .]             # Triton's interpreter
    python vsa_scores_check.py --device [--package .]    # a device

Triton's interpreter executes the kernel body with numpy-backed tensors, one program at a time, and
proves the indexing, the masks and the layout; its ``tl.dot`` is an FP32 product whatever the
``input_precision``. The product's own precision is the tensor cores', so ``--device`` runs the
module's extents on a GPU, where the three TF32 passes of ``tf32x3`` multiply.

What is checked:

* the scores themselves, at the operator's own bound — ``Tolerance::split_dot(width,
  tf32x3(width), f32, f32)``, the bound for a ``width``-term contraction with the three-pass
  product's excess counted — computed here by torch in FP64 and rounded once, so the reference is
  exact and the bound is what the operator declares;
* a key-tile count that is not a multiple of the kernel's key step, so the step's mask is
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

#: Triton reads `TRITON_INTERPRET` when it is imported, so the mode is chosen from the command line
#: before the kernel is loaded: `--device` compiles for the device, anything else interprets.
DEVICE = "--device" in sys.argv
if not DEVICE:
    os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402  (after TRITON_INTERPRET, which Triton reads at import)

#: The kernel packages' root: this check lives in `tests/python/kernels/<family>/`.
KERNELS = (
    pathlib.Path(__file__).resolve().parents[4] / "crates" / "ops" / "kernels"
)
#: The package this check covers.
PACKAGE = KERNELS / "vsa_scores" / "triton"

#: Unit roundoff of FP32, the accumulator the operator names.
UNIT_ROUNDOFF = 2.0**-24


def gamma(reduction, unit=UNIT_ROUNDOFF):
    """``γₙ = n·u / (1 − n·u)``."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def tf32x3(reduction):
    """``ops::tolerance::tf32x3``: the roundings the three-pass product costs beyond FP32's."""
    return 21 + -(-3 * reduction // 1024)


def declared_bound(width):
    """``VsaScores::tolerance``: ``split_dot(width, tf32x3(width), f32, f32)``'s ``rtol``."""
    return 2.0 * gamma(width + tf32x3(width)) * width**0.5 + UNIT_ROUNDOFF


def load_kernel(package):
    """The package's kernel module, imported by path."""
    source = package / "src" / "vsa_scores.py"
    spec = importlib.util.spec_from_file_location("vsa_scores_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(
    module,
    *,
    heads,
    width,
    block,
    query_tiles,
    key_tiles,
    key_step=16,
    seed=23,
    device="cpu",
):
    """Multiply one pair of pooled tilings and compare with the operator's product."""
    torch.manual_seed(seed)
    # Both operands are FP32: that is the type `vsa_prepare_tiles` pools a tile's per-head means
    # in, and the operator refuses anything else, so there is no element type to vary here.
    query = torch.randn(query_tiles, heads, width, dtype=torch.float32)
    key = torch.randn(key_tiles, heads, width, dtype=torch.float32)
    out = torch.full(
        (heads, query_tiles, key_tiles),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )

    module._vsa_scores_kernel[((query_tiles + 15) // 16, heads, 1)](
        query.to(device),
        key.to(device),
        out,
        query_tiles,
        key_tiles,
        0,
        WIDTH=width,
        BLOCK=block,
        KEY_TILES=key_step,
        QUERIES=16,
        PDL=False,
    )
    out = out.cpu()

    # The reference: the same operands in FP64, rounded once into FP32.
    expected = (
        torch.einsum(
            "ihw,jhw->hij", query.to(torch.float64), key.to(torch.float64)
        )
        / width**0.5
    ).to(torch.float32)

    bound = declared_bound(width)
    error = (out - expected).abs()
    # A dot product's error scales with the sum of its terms' magnitudes, and a score that cancels
    # to near zero has no such scale of its own.
    magnitude = (
        torch.einsum(
            "ihw,jhw->hij",
            query.abs().to(torch.float64),
            key.abs().to(torch.float64),
        )
        / width**0.5
    )
    allowed = bound * magnitude.to(torch.float32)
    unset = int(torch.isnan(out).sum())
    # `~(error <= allowed)` counts a NaN difference as outside, which also covers the scores the
    # kernel never wrote — so `unset` is reported but not added.
    violations = int((~(error <= allowed)).sum())
    if violations:
        position = int((~(error <= allowed)).reshape(-1).nonzero()[0])
        raise SystemExit(
            f"heads {heads} width {width} block {block} "
            f"{query_tiles}x{key_tiles} tiles: {violations} of {out.numel()} scores outside the "
            f"bound ({unset} not finite); first at {position}: "
            f"{float(out.reshape(-1)[position])} vs {float(expected.reshape(-1)[position])}, "
            f"rtol {bound:.3e}"
        )
    return out.numel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=PACKAGE,
        help="the package directory holding src/vsa_scores.py",
    )
    parser.add_argument(
        "--device", action="store_true", help="run on a CUDA device"
    )
    options = parser.parse_args()
    module = load_kernel(options.package)

    checked = 0
    if options.device:
        # The module's own extents (the launch table's: a key step of 32, sixteen query
        # tiles a program), which the tensor cores need at least sixteen wide: query and key
        # tile counts that fill no step, and a contraction narrower than its lanes.
        checked += run(
            module,
            heads=2,
            width=128,
            block=128,
            query_tiles=37,
            key_tiles=70,
            key_step=32,
            device="cuda",
        )
        checked += run(
            module,
            heads=3,
            width=96,
            block=128,
            query_tiles=20,
            key_tiles=45,
            key_step=32,
            device="cuda",
        )
        where = "the device"
    else:
        # The fixture's cell, then the two edges: a key count that is not a multiple of the 16-tile
        # step, and a contraction narrower than the block of lanes.
        checked += run(
            module, heads=2, width=128, block=128, query_tiles=3, key_tiles=7
        )
        checked += run(
            module, heads=3, width=64, block=128, query_tiles=2, key_tiles=5
        )
        checked += run(
            module, heads=1, width=32, block=128, query_tiles=4, key_tiles=20
        )
        where = "the interpreter"
    print(
        f"ok on {where}: {checked} scores within the contraction's bound, both masks exercised"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
