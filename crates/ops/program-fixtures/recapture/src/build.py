"""The recapture fixture's compile script: one descriptor -> one module.

The module is a `gemm` that launches two kernels (the product into FP32 scratch,
then the epilogue) where the serving `cublas_lt` program launches one. Its
descriptor's element types, weight order, epilogue and contraction length are
known when it is compiled, so the script passes them as defines; the launch
form itself is read from the package's manifest, so the symbol the module
exports and the symbol the package declares cannot disagree.

A descriptor this fixture does not implement — an accumulate type other than
FP32, or an input and weight of different element types — is refused here by
name, rather than compiled into a module that would refuse every launch.

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
SOURCES = ["src/split_k.cu"]

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

#: The operator's weight orders, as `WeightOrder::Display` prints them.
ORDERS = {"kxn": 0, "nxk": 1}

#: The operator's epilogues, as `Epilogue::Display` prints them.
EPILOGUES = {"none": 0, "bias": 1}


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


def field(key, name):
    """The value a descriptor key gives one field, such as its input type."""
    tokens = key.split("_")
    for at, token in enumerate(tokens[:-1]):
        if token == name:
            return tokens[at + 1]
    raise SystemExit(f"the descriptor {key} names no {name}")


def element_type(key, name):
    """The `DType` number a descriptor's field names."""
    token = field(key, name)
    if token not in ELEMENT_TYPES:
        raise SystemExit(f"{name} = {token!r} is not an element type the operator has")
    return ELEMENT_TYPES[token]


def launch_entry(package):
    """The launch symbol the package's manifest declares."""
    manifest = tomllib.loads((package / "package.toml").read_text())
    entry = manifest.get("module", {}).get("entries", {}).get("launch")
    if not entry:
        raise SystemExit("the package's [module.entries] names no launch symbol")
    return entry


def main():
    """Compile the fixture for one descriptor into ``module.so``."""
    package, descriptor, target, out = arguments()
    entry = launch_entry(package)
    input_type = element_type(descriptor, "input")
    weight_type = element_type(descriptor, "weight")
    output_type = element_type(descriptor, "output")
    accumulate = element_type(descriptor, "accumulate")
    order = field(descriptor, "order")
    epilogue = field(descriptor, "epilogue")
    if order not in ORDERS:
        raise SystemExit(f"order = {order!r} is not a weight order the operator has")
    if epilogue not in EPILOGUES:
        raise SystemExit(f"epilogue = {epilogue!r} is not an epilogue the operator has")
    if accumulate != ELEMENT_TYPES["f32"]:
        raise SystemExit("this fixture accumulates in FP32 only")
    if input_type != weight_type:
        raise SystemExit("this fixture takes one element type for its input and its weight")
    if output_type not in {ELEMENT_TYPES["f16"], ELEMENT_TYPES["bf16"], ELEMENT_TYPES["f32"]}:
        raise SystemExit("this fixture writes f16, bf16 or f32 products")

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
        f"-DUNISERVE_GEMM_ENTRY={entry}",
        f"-DUNISERVE_GEMM_INPUT={input_type}",
        f"-DUNISERVE_GEMM_WEIGHT={weight_type}",
        f"-DUNISERVE_GEMM_ACCUMULATE={accumulate}",
        f"-DUNISERVE_GEMM_OUTPUT={output_type}",
        f"-DUNISERVE_GEMM_ORDER={ORDERS[order]}",
        f"-DUNISERVE_GEMM_EPILOGUE={EPILOGUES[epilogue]}",
        f"-DUNISERVE_GEMM_K={field(descriptor, 'k')}",
        *[str(package / source) for source in SOURCES],
        "-lcudart",
        "-o",
        str(out / "module.so"),
    ]
    print(" ".join(command), file=sys.stderr)
    built = subprocess.run(command, capture_output=True, text=True, check=False)
    if built.returncode != 0:
        raise SystemExit(f"nvcc failed with {built.returncode}:\n{built.stderr[-4000:]}")

    exported = subprocess.run(
        ["nm", "-D", "--defined-only", str(out / "module.so")],
        capture_output=True,
        text=True,
        check=True,
    )
    symbols = {line.split()[-1] for line in exported.stdout.splitlines() if line.split()}
    if entry not in symbols:
        raise SystemExit(f"module.so does not export {entry}")
    print(json.dumps({"descriptor": descriptor, "target": target, "entry": entry}))


if __name__ == "__main__":
    main()
