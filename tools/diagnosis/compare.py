#!/usr/bin/env python3
"""Compare two block-by-block dump trees (design 10's diagnosis, never a gate).

The protocol is ``notes/h3/diagnosis.md``: both implementations write one
``.npy`` per (block, tensor name) under their own directory, and this script
matches them by relative path and reports, per tensor, the relative L2 error,
the maximum absolute difference and the cosine similarity.

Usage::

    python tools/diagnosis/compare.py --ref DIR --got DIR --out report.json
    python tools/diagnosis/compare.py --self-test          # proves the harness

It needs only NumPy: the diagnosis must run without a GPU to prove that the
harness itself is sound before any checkpoint is loaded.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

EPS = 1e-12


def metrics(got: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Relative L2, maximum absolute difference and cosine similarity.

    Both arrays are flattened as float64 so that a difference in element type
    (BF16 against FP32) is measured as the numbers it represents.
    """
    a = np.asarray(got, dtype=np.float64).ravel()
    b = np.asarray(ref, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: got {a.shape}, ref {b.shape}")
    difference = a - b
    denominator = max(float(np.linalg.norm(b)), EPS)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    cosine = (
        float(np.dot(a, b)) / max(norm_a * norm_b, EPS)
        if norm_a > 0.0 and norm_b > 0.0
        else 1.0
    )
    return {
        "rel_l2": float(np.linalg.norm(difference)) / denominator,
        "max_abs": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "cosine": cosine,
        "elements": float(difference.size),
        "ref_norm": norm_b,
    }


def _files(root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Every ``.npy`` under ``root``, keyed by its POSIX relative path."""
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*.npy"))
    }


def compare(ref: pathlib.Path, got: pathlib.Path) -> dict[str, object]:
    """Match the two trees by relative path and compare every pair."""
    if not ref.is_dir():
        raise SystemExit(f"reference tree is not a directory: {ref}")
    if not got.is_dir():
        raise SystemExit(f"dump tree is not a directory: {got}")
    ref_files = _files(ref)
    got_files = _files(got)
    tensors: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    extra: list[str] = []
    failed: dict[str, str] = {}
    for name in sorted(ref_files):
        if name not in got_files:
            missing.append(name)
            continue
        try:
            tensors[name] = metrics(np.load(got_files[name]), np.load(ref_files[name]))
        except ValueError as error:  # shape mismatch is a finding, not a crash
            failed[name] = str(error)
    for name in sorted(got_files):
        if name not in ref_files:
            extra.append(name)
    return {
        "tensors": tensors,
        "missing_in_got": missing,
        "only_in_got": extra,
        "shape_mismatch": failed,
    }


def _table(result: dict[str, object]) -> str:
    """A readable table: one line per tensor, largest relative error first."""
    tensors: dict[str, dict[str, float]] = result["tensors"]  # type: ignore[assignment]
    ordered = sorted(tensors.items(), key=lambda item: -item[1]["rel_l2"])
    lines = [f"{'tensor':<48}{'rel_l2':>12}{'max_abs':>14}{'cosine':>12}"]
    for name, values in ordered:
        lines.append(
            f"{name:<48}{values['rel_l2']:>12.3e}"
            f"{values['max_abs']:>14.3e}{values['cosine']:>12.6f}"
        )
    if result["missing_in_got"]:
        lines.append("")
        lines.append("not dumped by the Rust side: " + ", ".join(result["missing_in_got"]))
    if result["only_in_got"]:
        lines.append("only in the Rust dump: " + ", ".join(result["only_in_got"]))
    for name, error in result["shape_mismatch"].items():  # type: ignore[union-attr]
        lines.append(f"shape mismatch {name}: {error}")
    return "\n".join(lines)


def self_test() -> int:
    """Check the comparator's own arithmetic on arrays with known answers."""
    rng = np.random.default_rng(0)
    reference = rng.standard_normal((4, 5, 6), dtype=np.float32)
    cases = {
        "identical": (reference.copy(), 0.0, 1.0),
        "bf16_rounded": (
            reference.astype(np.float32).astype(np.float16).astype(np.float32),
            None,
            1.0,
        ),
        "scaled": (reference * 1.01, 1e-2, 1.0),
        "zero_ref": (np.zeros_like(reference), None, 1.0),
    }
    failures = 0
    for name, (got, expected_rel, expected_cos) in cases.items():
        result = metrics(got, reference)
        if expected_rel is not None and abs(result["rel_l2"] - expected_rel) > 1e-6:
            print(f"self-test {name}: rel_l2 {result['rel_l2']} != {expected_rel}")
            failures += 1
        if abs(result["cosine"] - expected_cos) > 1e-5 and name != "zero_ref":
            print(f"self-test {name}: cosine {result['cosine']} != {expected_cos}")
            failures += 1
        if name == "identical" and result["max_abs"] != 0.0:
            print(f"self-test {name}: max_abs {result['max_abs']} != 0")
            failures += 1
    print(f"self-test: {len(cases) - failures}/{len(cases)} cases correct")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", type=pathlib.Path, help="reference dump tree")
    parser.add_argument("--got", type=pathlib.Path, help="Rust dump tree")
    parser.add_argument("--out", type=pathlib.Path, help="report.json to write")
    parser.add_argument("--self-test", action="store_true", help="check the harness itself")
    arguments = parser.parse_args(argv)
    if arguments.self_test:
        return self_test()
    if not (arguments.ref and arguments.got):
        parser.error("--ref and --got are required unless --self-test is given")
    result = compare(arguments.ref, arguments.got)
    table = _table(result)
    print(table)
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(result, indent=2) + "\n")
        arguments.out.with_suffix(".txt").write_text(table + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
