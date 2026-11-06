#!/usr/bin/env python3
"""Compare one served request's dump with the reference's, block by block, and write the report.

    python tools/diagnosis/compare_served.py <served dump> <reference dump> <report.md>

The two dumps come from `UNISERVE_DIAG_DIR` and `tools/diagnosis/reference_dump.py`, the second run
with `INJECT_NOISE` naming the first, so both trajectories start from the same numbers. The packed
row order of the reference is recovered from those numbers themselves: a value of the draw names
the served element it came from, so nothing here restates the tile order.

This is a diagnosis and never a gate (design 10): it reports numbers and fails on nothing.
"""

import sys
from pathlib import Path

import numpy as np


def relative(got, want):
    """The relative L2 distance of `got` from `want`."""
    got = np.asarray(got, dtype=np.float64).ravel()
    want = np.asarray(want, dtype=np.float64).ravel()
    return float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-30))


def packed_order(served, reference):
    """The served element of each reference element, from the shared noise."""
    order = np.argsort(served, kind="stable")
    positions = np.clip(np.searchsorted(served[order], reference), 0, len(served) - 1)
    return order[positions]


def main():
    served, reference, report = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    lines = ["| block | tensor | relative L2 |", "| --- | --- | --- |"]

    prompt = np.load(reference / "prompt_ids.npy")
    tokens = len(prompt)
    source = np.load(served / "conditioner/source.npy")
    text = np.load(served / "conditioner/text.npy")
    features = np.load(reference / "encoder/output.npy")
    refined = np.load(reference / "conditioner/output.npy")
    lines.append(f"| text encoder | {tokens} prompt rows | {relative(source[:tokens], features[:tokens]):.3e} |")
    lines.append(f"| refiner | {tokens} prompt rows | {relative(text[:tokens], refined[:tokens]):.3e} |")

    video = np.load(served / "denoiser/latents_video_0.npy").ravel()
    want = np.load(reference / "denoiser/latents_video_0.npy").ravel()
    order = packed_order(video, want)
    steps = len(list((served / "denoiser").glob("latents_video_*.npy"))) - 1
    for step in range(steps + 1):
        got = np.load(served / f"denoiser/latents_video_{step}.npy").ravel()[order]
        want = np.load(reference / f"denoiser/latents_video_{step}.npy").ravel()
        lines.append(f"| denoiser | video latents after {step} evaluations | {relative(got, want):.3e} |")
        got = np.load(served / f"denoiser/latents_audio_{step}.npy").ravel()
        want = np.load(reference / f"denoiser/latents_audio_{step}.npy").ravel()
        lines.append(f"| denoiser | audio latents after {step} evaluations | {relative(got, want):.3e} |")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
