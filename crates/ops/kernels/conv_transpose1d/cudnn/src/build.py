"""The convolution packages' compile script: one descriptor -> one module.

The module is the convolution itself: cuDNN 9's backend API, asked once for an
engine and then executed on the caller's stream. A descriptor is the element
types and the geometry, both known when the module is compiled, so the script
passes them to the compiler as defines, and the entry is the package's operator.

`tuning/engines.json`, when the package carries one, maps a descriptor key to
the cuDNN engine's global index a tuning pass chose for it (design section 5.4);
this descriptor's index is compiled in, and the module then takes that engine
among cuDNN's proposals or fails naming it. Without such data the module takes
the heuristic's first proposal, which is stable for the cuDNN version the
package's toolchain pins.

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

#: The translation unit of the module: the entry, the plan cache and the bias.
SOURCES = ["src/conv.cu"]

#: The operator crate's `DType::as_i64()` numbering: the value each descriptor
#: token compiles to.
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

#: The engine index of a module compiled without tuning data.
UNTUNED = -1


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


def launch_entry(package):
    """The launch symbol the package's manifest declares.

    The manifest and the module must name the same entry, so the script reads
    the symbol out of the manifest instead of building it from the operator's
    name.
    """
    manifest = tomllib.loads((package / "package.toml").read_text())
    entry = manifest.get("module", {}).get("entries", {}).get("launch")
    if not entry:
        raise SystemExit(
            "the package's [module.entries] names no launch symbol"
        )
    return entry


def descriptor_field(key, field):
    """The value a descriptor key gives one field, such as its input type."""
    tokens = key.split("_")
    for at, token in enumerate(tokens[:-1]):
        if token == field:
            return tokens[at + 1]
    raise SystemExit(f"the descriptor {key} names no {field}")


def element_type(key, field):
    """The `DType` number a descriptor's field names."""
    token = descriptor_field(key, field)
    if token not in ELEMENT_TYPES:
        raise SystemExit(
            f"{field} = {token!r} is not an element type the operator has"
        )
    return ELEMENT_TYPES[token]


def engine(package, descriptor):
    """The tuned engine index of this descriptor, or the untuned marker."""
    tuning = package / "tuning" / "engines.json"
    if not tuning.is_file():
        return UNTUNED
    recorded = json.loads(tuning.read_text())
    return recorded.get(descriptor, UNTUNED)


def main():
    """Compile the convolution for one descriptor into ``module.so``."""
    package, descriptor, target, out = arguments()
    out.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    entry = launch_entry(package)

    command = [
        compiler,
        "-O3",
        "-std=c++17",
        f"-arch={target}",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        f"-I{package / 'include'}",
        f"-DUNISERVE_CONV_ENTRY={entry}",
        f"-DUNISERVE_CONV_INPUT_DTYPE={element_type(descriptor, 'input')}",
        f"-DUNISERVE_CONV_WEIGHT_DTYPE={element_type(descriptor, 'weight')}",
        f"-DUNISERVE_CONV_OUTPUT_DTYPE={element_type(descriptor, 'output')}",
        f"-DUNISERVE_CONV_ENGINE={engine(package, descriptor)}",
        *[str(package / source) for source in SOURCES],
        "-lcudart",
        "-ldl",
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
        json.dumps({"descriptor": descriptor, "target": target, "entry": entry})
    )


if __name__ == "__main__":
    main()
