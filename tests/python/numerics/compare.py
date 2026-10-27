"""Refuse a code change that keeps a component's identity but not what it
computes.

A component's identity (``uniserve.numerics.identity``) keys the derived
state it produces, so an unchanged identity asserts that the component
computes what it computed before, up to the rounding that batching itself
causes: the prefix cache already reuses a page computed in one batch for a
request in another. This program verifies the assertion for a revision
pair. It checks the base revision out into a temporary git worktree and runs
``capture.py`` once on each tree, the base and the head (by default the
working tree this program belongs to), each importing its own ``uniserve``
and ``uniserve_kernels``, on one checkpoint and one seeded workload. Then,
for every leaf, a tensor a module call returns, whose call has the same
identity chain in both revisions (its own identity and those of everything
computed before it), it requires

    difference <= variation

``difference`` is the largest difference between the base's and the head's
probe rows over every composition of the leaf's phase; ``variation`` is the
larger of the two revisions' batch variations, each the largest difference
between the probe's rows in any composition (including the probe computed
alone a second time) and its rows computed alone. A floating-point
difference is the largest absolute elementwise difference; an integer
leaf's (an index) is the number of elements that differ. Taking the larger
variation makes the verdict symmetric in the two revisions: either
implementation's own batching admits the difference. Where batching changes
nothing in either revision, such as rotary tables and the layers before the
first batch-dependent kernel, the change must reproduce the base bit for
bit; a computation that is batch invariant throughout is held to bitwise
equality everywhere.

A call whose component changed its identity, or whose inputs did, is
reported and not compared. The program writes ``report.json`` beside the
captures and exits 1 when any leaf exceeds its variation.
"""  # noqa: D205

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

CAPTURE = Path(__file__).with_name("capture.py")
REPOSITORY = Path(__file__).resolve().parents[3]


def difference(left: torch.Tensor, right: torch.Tensor) -> float:
    """The largest absolute difference of two floating-point tensors, or the
    number of differing elements of two integer tensors.

    Elements equal in both, NaN included, contribute nothing; a NaN against
    a number is an infinite difference.
    """  # noqa: D205
    if left.shape != right.shape:
        return math.inf
    if not left.dtype.is_floating_point:
        return float((left != right).sum())
    left, right = left.double(), right.double()
    same = (left == right) | (left.isnan() & right.isnan())
    gap = (left - right).abs().nan_to_num(nan=math.inf)
    return float(torch.where(same, 0.0, gap).max()) if gap.numel() else 0.0


@dataclass(frozen=True)
class Capture:
    """One revision's capture: its description and its leaves' rows."""

    description: dict
    path: Path

    @classmethod
    def read(cls, stem: Path) -> Capture:
        description = json.loads(stem.with_suffix(".json").read_text())
        return cls(description, stem.with_suffix(".safetensors"))

    def phases(self) -> dict[str, list[str]]:
        workload = self.description["workload"]["compositions"]
        return {phase: list(names) for phase, names in workload.items()}

    def leaves(self) -> dict[tuple[str, str], set[str]]:
        """The compositions each ``(phase, leaf)`` has rows in."""
        found: dict[tuple[str, str], set[str]] = {}
        with safe_open(self.path, "pt") as tensors:
            for key in tensors.keys():  # noqa: SIM118
                phase, composition, leaf = key.split("/", 2)
                found.setdefault((phase, leaf), set()).add(composition)
        return found


def variation(tensors, phase: str, leaf: str, compositions: list[str]) -> float:
    """The largest difference batching makes to a leaf's probe rows."""
    alone = tensors.get_tensor(f"{phase}/alone/{leaf}")
    return max(
        difference(tensors.get_tensor(f"{phase}/{name}/{leaf}"), alone)
        for name in compositions
        if name != "alone"
    )


def _call(leaf: str) -> str:
    """The module call a leaf belongs to: ``<path>#<number>``."""
    path, _, rest = leaf.partition("#")
    number = ""
    for character in rest:
        if not character.isdigit():
            break
        number += character
    return f"{path}#{number}"


def compare(base: Capture, head: Capture) -> dict:
    """Compare two captures of one workload; see the module description."""
    if base.description["workload"] != head.description["workload"]:
        raise SystemExit("the captures evaluated different workloads")
    phases = base.phases()
    base_calls = base.description["calls"]
    head_calls = head.description["calls"]
    base_components = base.description["components"]
    head_components = head.description["components"]

    # Which calls are compared, and why the others are not.
    status: dict[str, str] = {}
    for call in sorted(set(base_calls) | set(head_calls)):
        if call not in base_calls:
            status[call] = "added"
        elif call not in head_calls:
            status[call] = "removed"
        else:
            path = base_calls[call]["component"]
            if (
                base_components[path]["identity"]
                != head_components[path]["identity"]
            ):
                status[call] = "changed"
            elif base_calls[call]["chain"] != head_calls[call]["chain"]:
                status[call] = "upstream"
            else:
                status[call] = "compared"

    base_leaves, head_leaves = base.leaves(), head.leaves()
    results, violations = [], []
    with (
        safe_open(base.path, "pt") as base_tensors,
        safe_open(head.path, "pt") as head_tensors,
    ):
        for phase, leaf in sorted(set(base_leaves) & set(head_leaves)):
            call = _call(leaf)
            if status.get(call) != "compared":
                continue
            compositions = phases[phase]
            # A leaf is attributed to rows only when every composition of
            # its phase, in both revisions, returned it row by row.
            if base_leaves[phase, leaf] != set(compositions) or head_leaves[
                phase, leaf
            ] != set(compositions):
                continue
            allowed = max(
                variation(base_tensors, phase, leaf, compositions),
                variation(head_tensors, phase, leaf, compositions),
            )
            measured = 0.0
            for name in compositions:
                key = f"{phase}/{name}/{leaf}"
                measured = max(
                    measured,
                    difference(
                        head_tensors.get_tensor(key),
                        base_tensors.get_tensor(key),
                    ),
                )
            alone = base_tensors.get_tensor(f"{phase}/alone/{leaf}")
            magnitude = (
                float(alone.double().abs().nan_to_num(nan=0.0).max())
                if alone.dtype.is_floating_point and alone.numel()
                else None
            )
            path = base_calls[call]["component"]
            result = {
                "phase": phase,
                "leaf": leaf,
                "component": path,
                "function": base_components[path]["function"],
                "difference": measured,
                "variation": allowed,
                "magnitude": magnitude,
                "floating": alone.dtype.is_floating_point,
            }
            results.append(result)
            if not measured <= allowed:
                violations.append(result)

    def described(call: str) -> dict:
        path = (base_calls.get(call) or head_calls[call])["component"]
        return {
            "call": call,
            "base": base_components.get(path, {}).get("function"),
            "head": head_components.get(path, {}).get("function"),
        }

    return {
        "base": base.description["revision"],
        "head": head.description["revision"],
        "workload": base.description["workload"],
        "leaves": results,
        "violations": violations,
        "not_compared": {
            reason: [
                described(call)
                for call, kind in status.items()
                if kind == reason
            ]
            for reason in ("changed", "upstream", "added", "removed")
        },
    }


def _relative(value: float, magnitude: float | None) -> str:
    if magnitude is None:
        return f"{value:.0f} elements"
    if not magnitude:
        return f"{value:.3e}"
    return f"{value:.3e} ({value / magnitude:.2e} of {magnitude:.3e})"


def summary(report: dict) -> str:
    """The report as text: what was compared, skipped and refused."""
    lines = []
    compared = {item["component"] for item in report["leaves"]}
    lines.append(
        f"compared {len(report['leaves'])} leaves of {len(compared)} "
        "components with unchanged identity chains"
    )
    for reason, calls in report["not_compared"].items():
        if not calls:
            continue
        lines.append(f"not compared, {reason}: {len(calls)} calls")
        for item in calls[:20]:
            functions = (
                item["base"]
                if item["base"] == item["head"]
                else f"{item['base']} -> {item['head']}"
            )
            lines.append(f"  {item['call']} ({functions})")
        if len(calls) > 20:
            lines.append(f"  ... {len(calls) - 20} more")
    for item in report["violations"]:
        lines.append(
            f"REFUSED {item['phase']} {item['leaf']} "
            f"[{item['function']}]: difference "
            f"{_relative(item['difference'], item['magnitude'])} exceeds "
            f"batch variation {_relative(item['variation'], item['magnitude'])}"
        )
    if not report["violations"]:
        lines.append("every compared leaf is within its batch variation")
    return "\n".join(lines)


def _git(tree: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(tree), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _label(tree: Path) -> str:
    """A tree's commit, marked when its tracked files differ from it, or
    its directory name outside a git checkout.
    """  # noqa: D205
    try:
        label = _git(tree, "rev-parse", "--short=12", "HEAD")
        if _git(tree, "status", "--porcelain", "--untracked-files=no"):
            label += "-modified"
    except subprocess.CalledProcessError:
        label = tree.name
    return label


def _capture(tree: Path, stem: Path, args: argparse.Namespace) -> Capture:
    """Run ``capture.py`` with ``tree``'s ``uniserve`` and kernels.

    ``PYTHONPATH`` entries precede the environment's editable installs, and
    a program run by path adds only its own directory to ``sys.path``, so
    the capture imports the tree's packages and nothing else of either tree.
    """
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(tree), str(tree / "uniserve_kernels" / "src"))
    )
    subprocess.run(
        [
            sys.executable,
            str(CAPTURE),
            "--checkpoint",
            str(args.checkpoint),
            "--output",
            str(stem),
            "--device",
            args.device,
            "--prompt",
            str(args.prompt),
            "--context",
            str(args.context),
            "--tokens",
            str(args.tokens),
            "--rows",
            str(args.rows),
        ],
        check=True,
        env=environment,
    )
    return Capture.read(stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base",
        required=True,
        help="the revision of this repository the change applies to",
    )
    parser.add_argument(
        "--head",
        type=Path,
        default=REPOSITORY,
        help="the tree of the change (default: this working tree)",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="directory of the captures and the report (default: "
        "artifacts/numerics/<base>-<head> in this working tree)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", type=int, default=64)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--rows", type=int, default=512)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()

    head = args.head.resolve()
    base = _git(REPOSITORY, "rev-parse", "--verify", f"{args.base}^{{commit}}")
    if args.output is None:
        args.output = (
            REPOSITORY
            / "artifacts"
            / "numerics"
            / f"{base[:12]}-{_label(head)}"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # The base is checked out into a temporary worktree of this repository
    # for its capture alone and removed even when the capture fails.
    with tempfile.TemporaryDirectory(prefix="numerics-base-") as scratch:
        tree = Path(scratch) / "tree"
        _git(REPOSITORY, "worktree", "add", "--detach", str(tree), base)
        try:
            base_capture = _capture(tree, output / "base", args)
        finally:
            _git(REPOSITORY, "worktree", "remove", "--force", str(tree))
    head_capture = _capture(head, output / "head", args)

    report = compare(base_capture, head_capture)
    (output / "report.json").write_text(json.dumps(report, indent=1))
    print(summary(report))
    print(f"report: {output / 'report.json'}")
    sys.exit(1 if report["violations"] else 0)


if __name__ == "__main__":
    main()
