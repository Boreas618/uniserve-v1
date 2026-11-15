"""The convolution packages' build module: one descriptor -> one module.

The module is the convolution itself: cuDNN 9's backend API, asked once for an
engine and then executed on the caller's stream. A descriptor is the three
element types and nothing else: the geometry, the bias and the TF32 choice
arrive in the call structure, which the entry reads, so one module serves every
window, filter, group count and rank of its types. The build passes the types
and the operator's launch entry, as the manifest resolves it, to the compiler as
defines.

`tuning/engines.json`, when the package carries one, maps a descriptor key to
the cuDNN engine's global index a tuning pass chose for it (design section 5.4);
this descriptor's index is compiled in, and the module then takes that engine
among cuDNN's proposals or fails naming it. Without such data the module takes
the heuristic's first proposal, which is stable for the cuDNN version the
package's toolchain pins.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

import json
import subprocess
import sys

from uniserve_build_server.request import Request

#: The translation unit of the module: the entry, the executable cache and the
#: bias.
SOURCES = ["src/conv.cu"]

#: The operator crate's package ABI numbering (`ops::abi::code`): the value
#: each descriptor token compiles to.
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


def element_type(request, field):
    """The `DType` number a descriptor's field names."""
    token = request.field(field)
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


def compile(request: Request) -> None:
    """Compile the convolution for the request's descriptor into ``module.so``.

    The module is written into the request's output directory; a compiler
    failure refuses the build, with the compiler's output in the build log.
    """
    package = request.package
    descriptor = request.descriptor
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
        f"-I{package / 'include'}",
        f"-DUNISERVE_CONV_ENTRY={request.entries['launch']}",
        f"-DUNISERVE_CONV_INPUT_DTYPE={element_type(request, 'input')}",
        f"-DUNISERVE_CONV_WEIGHT_DTYPE={element_type(request, 'weight')}",
        f"-DUNISERVE_CONV_OUTPUT_DTYPE={element_type(request, 'output')}",
        f"-DUNISERVE_CONV_ENGINE={engine(package, descriptor)}",
        *[str(package / source) for source in SOURCES],
        "-lcudart",
        "-ldl",
        "-o",
        str(request.output / "module.so"),
    ]
    print(" ".join(command), file=sys.stderr)
    built = subprocess.run(command, check=False)
    if built.returncode != 0:
        raise SystemExit(f"nvcc failed with {built.returncode}")
