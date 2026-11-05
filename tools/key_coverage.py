#!/usr/bin/env python3
"""Which of a fixture's keys no package declares, without compiling anything.

`build_store.py` answers this as a side effect of building, and a build needs a host
whose toolchain carries every package - so on a host that cannot compile, coverage is
unmeasured and a fixture reads as complete because nothing said otherwise.

This asks the same question with the same functions (`load_packages`, `covers`) and
stops before the compile. It answers what a package *declares*, not what it can build:
a key can be covered and still fail at compile, and that is `build_store`'s answer.

The distinction matters because the two failures have different cures. A key nothing
declares is a `dimensions` line or a missing package; a key that is declared and does
not compile is a kernel. `slot_commit`'s rows were the first kind for weeks and looked
like the second.

Usage:
    tools/key_coverage.py --fixture notes/programs/qwen3-moe-keys.txt [--target sm_100a]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_store  # noqa: E402


def declares(package, target):
    """Whether the package serves `target` at all."""
    kernel = package["manifest"].get("kernel") or {}
    architectures = kernel.get("architectures") or []
    return not architectures or target in architectures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--kernels", default="crates/ops/kernels")
    parser.add_argument("--target", default="sm_100a")
    args = parser.parse_args()

    packages = build_store.load_packages(Path(args.kernels))
    rows = [
        tuple(line.rstrip("\n").split("\t", 1))
        for line in Path(args.fixture).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not rows:
        sys.exit(f"{args.fixture} names no rows: an empty fixture is a walk that printed nothing")

    uncovered, doubled = [], []
    for op, key in rows:
        claiming = [
            package
            for package in packages.get(op, [])
            if declares(package, args.target) and build_store.covers(package, key)
        ]
        if not claiming:
            uncovered.append((op, key))
        elif len(claiming) > 1:
            doubled.append((op, key, [package["directory"].name for package in claiming]))

    print(f"{args.fixture}: {len(rows)} rows, target {args.target}")
    if doubled:
        print(f"{len(doubled)} key(s) claimed by more than one package:")
        for op, key, names in doubled:
            print(f"  {op}\t{key}\t{names}")

    if not uncovered:
        print(f"every key is covered for {args.target}")
        return 0 if not doubled else 1

    print(f"{len(uncovered)} of {len(rows)} keys are not covered for {args.target}:")
    for op, key in uncovered:
        print(f"  {op}\t{key}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
