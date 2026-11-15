"""The recapture fixture's build module: one descriptor -> one module.

The module is a `gemm` that launches two kernels (the product into FP32
scratch, then the epilogue) where the serving `cublas_lt` package launches one.
Its descriptor's element types, weight order and epilogue are known when it is
compiled, so the build passes them as defines; the launch form itself is read
from the package's manifest, so the symbol the module exports and the symbol
the package declares cannot disagree.

A descriptor this fixture does not implement — an accumulate type other than
FP32, or an input and weight of different element types — is refused here by
name, rather than compiled into a module that would refuse every launch.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

import json
import subprocess
import sys
import tomllib

from uniserve_build_server.request import Request

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


def element_type(request, name):
    """The `DType` number a descriptor's field names."""
    token = request.field(name)
    if token not in ELEMENT_TYPES:
        raise SystemExit(
            f"{name} = {token!r} is not an element type the operator has"
        )
    return ELEMENT_TYPES[token]


def launch_entry(package):
    """The launch symbol the package's manifest declares."""
    manifest = tomllib.loads((package / "package.toml").read_text())
    entry = manifest.get("module", {}).get("entries", {}).get("launch")
    if not entry:
        raise SystemExit(
            "the package's [module.entries] names no launch symbol"
        )
    return entry


def compile(request: Request) -> None:
    """Compile the fixture for the request's descriptor into ``module.so``.

    The descriptor is checked first, so one this fixture does not implement is
    refused by name; a compiler failure refuses the build, with the compiler's
    output in the build log.
    """
    package = request.package
    entry = launch_entry(package)
    input_type = element_type(request, "input")
    weight_type = element_type(request, "weight")
    output_type = element_type(request, "output")
    accumulate = element_type(request, "accumulate")
    order = request.field("order")
    epilogue = request.field("epilogue")
    if order not in ORDERS:
        raise SystemExit(
            f"order = {order!r} is not a weight order the operator has"
        )
    if epilogue not in EPILOGUES:
        raise SystemExit(
            f"epilogue = {epilogue!r} is not an epilogue the operator has"
        )
    if accumulate != ELEMENT_TYPES["f32"]:
        raise SystemExit("this fixture accumulates in FP32 only")
    if input_type != weight_type:
        raise SystemExit(
            "this fixture takes one element type for its input and its weight"
        )
    if output_type not in {
        ELEMENT_TYPES["f16"],
        ELEMENT_TYPES["bf16"],
        ELEMENT_TYPES["f32"],
    }:
        raise SystemExit("this fixture writes f16, bf16 or f32 products")

    # The nvcc the Build Server checked against this package's declared version.
    compiler = request.toolchain.require_cuda().nvcc
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        f"-arch={request.target}",
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
        *[str(package / source) for source in SOURCES],
        "-lcudart",
        "-o",
        str(request.output / "module.so"),
    ]
    print(" ".join(command), file=sys.stderr)
    built = subprocess.run(command, check=False)
    if built.returncode != 0:
        raise SystemExit(f"nvcc failed with {built.returncode}")

    print(
        json.dumps(
            {
                "descriptor": request.descriptor,
                "target": str(request.target),
                "entry": entry,
            }
        )
    )
