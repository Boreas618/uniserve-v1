#!/usr/bin/env python3
"""Assemble a program store for a key fixture, offline, without a GPU.

For every fixture row whose descriptor some package covers, run that package's compile script with
the environment the Program Server passes (`UNISERVE_OP`, `UNISERVE_BACKEND`, `UNISERVE_DESCRIPTOR`,
`UNISERVE_TARGET`, `UNISERVE_PACKAGE`, `UNISERVE_OUT`), then publish the result under the store
layout `<root>/<op>/<backend>/<revision>/<target>/<descriptor>/` with the `.committed` mark.

The output is a store a reader can use: `program_server::store::Store::open([root])`, or the
`qwen3_moe_slice` example's `--store`. It is a stopgap for a host where the Program Server has not
run; the published layout and the environment are the Program Server's own, so the two cannot
disagree about what a module looks like.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

TARGETS = {"sm_100a", "sm_90a"}


def load_packages(kernels: Path):
    """Every package under <kernels>/<op>/<backend>/, by op."""
    packages = {}
    for manifest_path in sorted(kernels.glob("*/*/package.toml")):
        with open(manifest_path, "rb") as handle:
            manifest = tomllib.load(handle)
        package = {
            "root": manifest_path.parent,
            "manifest": manifest,
            "op": manifest["op"],
            "backend": manifest["backend"],
        }
        packages.setdefault(package["op"], []).append(package)
    return packages


def split_key(dimensions, key):
    """The key's `dimension_value` pairs, longest dimension name first."""
    names = sorted(dimensions, key=len, reverse=True)
    pairs, rest = [], key
    while rest:
        name = next((n for n in names if rest.startswith(n)), None)
        if name is None or not rest[len(name):].startswith("_"):
            return None
        after = rest[len(name) + 1:]
        ends = [after.find("_" + n + "_") for n in names]
        ends = [end for end in ends if end >= 0]
        end = min(ends) if ends else len(after)
        if not after[:end]:
            return None
        pairs.append((name, after[:end]))
        rest = after[end:].lstrip("_")
    return pairs


def shapes(kernel):
    """The descriptor tables a package declares: one, or one per shape."""
    descriptors = kernel.get("descriptors")
    if descriptors is None:
        return []
    return descriptors if isinstance(descriptors, list) else [descriptors]


def covers_one(descriptors, key):
    if key in descriptors.get("keys", []):
        return True
    dimensions = descriptors.get("dimensions") or {}
    if not dimensions:
        return False
    pairs = split_key(dimensions, key)
    if pairs is None:
        return False
    return all(value in dimensions.get(name, []) for name, value in pairs)


def covers(package, key):
    """Whether the package covers this key: it covers one of the shapes it declares."""
    kernel = package["manifest"].get("kernel")
    if kernel is None:
        return False
    return any(covers_one(descriptors, key) for descriptors in shapes(kernel))


def build(package, key, target, out_dir):
    """Compile one descriptor into `out_dir`; returns a failure message or None."""
    manifest = package["manifest"]
    source = manifest["source"]
    script = source.get("script")
    if not script:
        return "the package declares no compile script"
    environment = dict(os.environ)
    environment.update({
        "UNISERVE_OP": package["op"],
        "UNISERVE_BACKEND": package["backend"],
        "UNISERVE_DESCRIPTOR": key,
        "UNISERVE_TARGET": target,
        "UNISERVE_PACKAGE": str(package["root"].resolve()),
        "UNISERVE_OUT": str(out_dir.resolve()),
    })
    if source["kind"] == "triton":
        # Triton's cache is build scratch, not part of the module: keep it out of the store so a store
        # can be copied or published without a compilation cache inside it.
        environment.setdefault("TRITON_CACHE_DIR", str(Path(tempfile.gettempdir()) / "uniserve-triton-cache"))
    # A package that fetches a vendored artifact asks for the Program Server's own interpreter by
    # importing `uniserve_program_server`; making that package importable lets the script keep this
    # interpreter, which is the one that carries the toolchain (Triton, CuTe DSL, nvcc).
    python_path = str((Path(__file__).resolve().parents[1]
                       / "crates/program-server/python/src").resolve())
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = python_path + (os.pathsep + existing if existing else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    command = {
        "triton": ["/opt/uv/venv/bin/python", script],
        "cuda": ["/opt/uv/venv/bin/python", script],
        "cutlass": ["/opt/uv/venv/bin/python", script],
        "cute_dsl": ["/opt/uv/venv/bin/python", script],
        "nvcc": ["/opt/uv/venv/bin/python", script],
    }.get(source["kind"])
    if command is None:
        command = ["/opt/uv/venv/bin/python", script]
    try:
        finished = subprocess.run(
            command,
            cwd=package["root"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=1200,
        )
    except subprocess.TimeoutExpired:
        return "the compile script did not finish in 20 minutes"
    if finished.returncode != 0:
        tail = (finished.stderr or finished.stdout).strip().splitlines()[-3:]
        return "compile script failed: " + " | ".join(tail)
    form = manifest["module"]["form"]
    if form == "cubin" and not (out_dir / "module.cubin").is_file():
        return "no module.cubin was written"
    if form == "native" and not any(out_dir.glob("*.so")):
        return "no shared object was written"
    # A native module's C entry is declared by its manifest and is not in the shared object; publish
    # it beside the library so the store directory is enough to load the module from.
    entries = manifest["module"].get("entries")
    if form == "native" and entries:
        (out_dir / "launch.json").write_text(
            json.dumps({"name": entries.get("launch"), "entries": entries}, indent=1) + "\n"
        )
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--root", required=True, help="the store root to assemble")
    parser.add_argument("--kernels", default="crates/ops/kernels")
    parser.add_argument("--target", default="sm_100a")
    parser.add_argument("--revision", type=int, default=1)
    args = parser.parse_args()

    if args.target not in TARGETS:
        sys.exit(f"target {args.target} is not one a package here declares ({sorted(TARGETS)})")

    packages = load_packages(Path(args.kernels))
    root = Path(args.root)
    # A fixture with no rows is a walk that failed to print, and a store built from it is empty and
    # says so as a success: "built 0 of 0" reads like a complete store and is no evidence at all.
    if not [line for line in Path(args.fixture).read_text().splitlines()
            if line.strip() and not line.startswith("#")]:
        raise SystemExit(
            f"{args.fixture} names no rows: an empty fixture is a walk that printed nothing, and "
            f"a store built from it would be empty rather than correct"
        )
    rows = [
        line.rstrip("\n").split("\t")
        for line in Path(args.fixture).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    # A key names one program, so exactly one package may claim it. Two packages claiming one key
    # would let two hosts that build the same fixture store different modules under the same name,
    # with nothing in the build log saying so — and the loader, which keys modules by (operator, key),
    # would then run whichever the directory walk offered last. The claim is checked over the whole
    # fixture before anything compiles, because a build that succeeds differently on another host is
    # worse than one that stops.
    claims = {}
    for op, key in rows:
        claiming = [
            package for package in packages.get(op, [])
            if covers(package, key)
            and args.target in package["manifest"]["kernel"]["architectures"]
        ]
        if len(claiming) > 1:
            claims.setdefault((op, key), []).append(claiming)
    if claims:
        report = []
        for (op, key), _ in sorted(claims.items()):
            directories = " and ".join(
                str(package["root"]) for package in packages[op] if covers(package, key)
                and args.target in package["manifest"]["kernel"]["architectures"]
            )
            report.append(f"  {op}\t{key}\n    claimed by {directories}")
        sys.exit(
            "the store cannot be built: a key must name one program, and these name two or more\n"
            + "\n".join(report)
        )

    built, skipped = [], []
    for op, key in rows:
        candidates = [
            package for package in packages.get(op, [])
            if covers(package, key)
            and args.target in package["manifest"]["kernel"]["architectures"]
        ]
        if not candidates:
            skipped.append((op, key, "no package covers this key for the target"))
            continue
        # A package may cover a key by its dimensions and still refuse the descriptor's variant
        # (the FA4 attention backend serves prefill only), so every candidate is tried before the
        # key is reported uncovered.
        failures = []
        for package in candidates:
            directory = root / op / package["backend"] / str(args.revision) / args.target / key
            failure = build(package, key, args.target, directory)
            if failure:
                failures.append(f"{package['backend']}: {failure}")
                continue
            (directory / ".committed").write_text("")
            built.append((op, package["backend"], key))
            break
        else:
            skipped.append((op, key, "; ".join(failures)))
            continue
    print(f"# built {len(built)} of {len(rows)} keys into {root} (revision {args.revision}, {args.target})")
    for op, backend, key in built:
        print(f"built\t{op}\t{backend}\t{key}")
    for op, key, reason in skipped:
        print(f"SKIPPED\t{op}\t{reason}\t{key}")
    return 0 if not skipped else 1


if __name__ == "__main__":
    sys.exit(main())
