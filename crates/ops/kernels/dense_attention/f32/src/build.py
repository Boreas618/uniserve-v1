"""The dense_attention f32 backend's compile script: one descriptor -> one module.

The mask is the only thing this backend compiles in: a causal module refuses a
dense call and the other way round, so the masking kernel carries no branch on it.
Everything else the descriptor names (heads, head width, the length cap) arrives in
the call and is checked there, and the history window is a bound the call carries,
so one module serves a windowed and an unwindowed call of the same descriptor.

The export name is read from the package's manifest, so the symbol the module
exports and the symbol the package declares cannot disagree.

A descriptor this backend does not implement — an element type other than FP32, or
heads that do not divide — is refused here by name rather than compiled into a
module that would refuse every launch.

The Program Server runs this script with its own Python environment, its working
directory the package root, and `UNISERVE_OP`, `UNISERVE_BACKEND`,
`UNISERVE_DESCRIPTOR`, `UNISERVE_TARGET`, `UNISERVE_PACKAGE`, `UNISERVE_OUT` and
`MAX_JOBS` in the environment.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tomllib

#: The translation unit of the module.
SOURCES = ["src/dense.cu"]

#: The operator crate's `DType::as_i64()` numbering.
ELEMENT_TYPES = {
    "u8": 0,
    "i32": 1,
    "i64": 2,
    "f16": 3,
    "bf16": 4,
    "f32": 5,
    "f8e4m3": 6,
    "f4e2m1": 7,
}

#: The operator's masks, as `Mask`'s `Display` prints them, to the compiled flag.
MASKS = {"causal": 1, "dense": 0}

#: The fields a descriptor of this operator carries.
FIELDS = (
    "dtype",
    "kv",
    "out",
    "head_dim",
    "q_heads",
    "kv_heads",
    "mask",
    "window",
    "len",
)


def arguments():
    """The package, descriptor, target and output directory of one build."""
    reader = argparse.ArgumentParser(description=__doc__)
    reader.add_argument("--package")
    reader.add_argument("--descriptor")
    reader.add_argument("--target")
    reader.add_argument("--out")
    known = reader.parse_args()
    values = {
        "package": known.package or os.environ.get("UNISERVE_PACKAGE"),
        "descriptor": known.descriptor or os.environ.get("UNISERVE_DESCRIPTOR"),
        "target": known.target or os.environ.get("UNISERVE_TARGET"),
        "out": known.out or os.environ.get("UNISERVE_OUT"),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)}")
    return (
        pathlib.Path(values["package"]).resolve(),
        values["descriptor"],
        values["target"],
        pathlib.Path(values["out"]),
    )


def manifest_of(package):
    """The package's manifest."""
    return tomllib.loads((package / "package.toml").read_text())


def launch_entry(manifest):
    """The launch symbol the package's manifest declares."""
    entry = manifest.get("module", {}).get("entries", {}).get("launch")
    if not entry:
        raise SystemExit("the package's [module.entries] names no launch symbol")
    return entry


def split(key, names):
    """The `name_value` pairs of a descriptor key, by the manifest's field names.

    A field name may contain an underscore (`head_dim`), and a value is a single
    token, so the key is split by the names the manifest declares rather than by
    every underscore.
    """
    ordered = sorted(names, key=len, reverse=True)
    pairs = {}
    rest = key
    while rest:
        name = next((one for one in ordered if rest.startswith(one)), None)
        if name is None:
            raise SystemExit(f"the descriptor {key} names no field at {rest!r}")
        after = rest[len(name) :]
        if not after.startswith("_"):
            raise SystemExit(f"the descriptor {key} gives {name} no value")
        after = after[1:]
        ends = [after.find(f"_{one}_") for one in ordered]
        ends = [end for end in ends if end >= 0]
        end = min(ends) if ends else len(after)
        if not after[:end]:
            raise SystemExit(f"the descriptor {key} gives {name} no value")
        pairs[name] = after[:end]
        rest = after[end:]
        rest = rest[1:] if rest.startswith("_") else ""
    missing = [name for name in FIELDS if name not in pairs]
    if missing:
        raise SystemExit(f"the descriptor {key} names no {', '.join(missing)}")
    return pairs


def main():
    """Compile the module for one descriptor into ``module.so``."""
    package, descriptor, target, out = arguments()
    manifest = manifest_of(package)
    entry = launch_entry(manifest)
    dimensions = manifest["kernel"]["descriptors"]["dimensions"]
    named = split(descriptor, dimensions.keys())

    for field in ("dtype", "kv", "out"):
        if named[field] != "f32":
            raise SystemExit(
                f"{field} = {named[field]!r}: this backend serves FP32 activations"
            )
    if named["mask"] not in MASKS:
        raise SystemExit(
            f"mask = {named['mask']!r} is not one of {sorted(MASKS)}"
        )
    if named["window"] not in {"false", "true"}:
        raise SystemExit(
            f"window = {named['window']!r} is not one of ['false', 'true']"
        )
    counts = {}
    for field in ("head_dim", "q_heads", "kv_heads", "len"):
        try:
            counts[field] = int(named[field])
        except ValueError:
            raise SystemExit(f"{field} = {named[field]!r} is not a number")
        if counts[field] <= 0:
            raise SystemExit(f"{field} = {counts[field]} names no work")
    if counts["q_heads"] % counts["kv_heads"] != 0:
        raise SystemExit(
            f"{counts['q_heads']} query heads do not divide over "
            f"{counts['kv_heads']} key heads"
        )
    # The value of an element type in the descriptor is the operator's numbering;
    # the module compares against it, so it is passed rather than assumed.
    for field in ("dtype", "kv", "out"):
        if named[field] not in ELEMENT_TYPES:
            raise SystemExit(f"{field} = {named[field]!r} is not an element type")

    out.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        f"-arch={target}",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        f"-DUNISERVE_ENTRY={entry}",
        f"-DUNISERVE_CAUSAL={MASKS[named['mask']]}",
        "-I",
        str(package / "include"),
        *[str(package / source) for source in SOURCES],
        "-lcublasLt",
        "-lcudart",
        "-o",
        str(out / "module.so"),
    ]
    print(" ".join(command), file=sys.stderr)
    built = subprocess.run(command, capture_output=True, text=True, check=False)
    if built.returncode != 0:
        raise SystemExit(
            f"nvcc failed with {built.returncode}:\n{built.stderr[-4000:]}"
        )

    exported = subprocess.run(
        ["nm", "-D", "--defined-only", str(out / "module.so")],
        capture_output=True,
        text=True,
        check=True,
    )
    symbols = {
        line.split()[-1]
        for line in exported.stdout.splitlines()
        if line.split()
    }
    if entry not in symbols:
        raise SystemExit(f"module.so does not export {entry}")
    print(
        json.dumps(
            {
                "descriptor": descriptor,
                "target": target,
                "entry": entry,
                "mask": named["mask"],
                "window": named["window"],
            }
        )
    )


if __name__ == "__main__":
    main()
