"""The moe_unpermute package's build module: one descriptor -> one module.

The module is trtllm-gen's combine kernel, compiled from the FlashInfer
distribution's sources into a shared object with this package's C entry in
front of it. A descriptor is the problem's hidden width, top-k expansion,
expert count and the three element types; the kernel reads them at launch, so
every descriptor of this package compiles the same module.

The sources come from the FlashInfer package the toolchain pins (its `data`
tree holds the trtllm-gen kernels and their headers), not from this tree: the
kernel is NVIDIA's, and the package's `[toolchain] packages` names where it
comes from.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

import pathlib
import tempfile

from uniserve_build_server import trtllm
from uniserve_build_server.request import Request

#: The element types the combine kernel is instantiated for.
DTYPES = {"bf16": "bf16", "fp16": "f16", "f32": "f32"}


#: The translation units of the FlashInfer distribution this module links: the
#: combine kernel and the TensorRT-LLM common library its checks call.
SOURCES = [
    "csrc/nv_internal/tensorrt_llm/kernels/cuteDslKernels/moeUtils.cu",
    "csrc/nv_internal/cpp/common/tllmException.cpp",
    "csrc/nv_internal/cpp/common/stringUtils.cpp",
    "csrc/nv_internal/cpp/common/logger.cpp",
    "csrc/nv_internal/cpp/common/envUtils.cpp",
]

#: The defines the distribution's own build uses for these kernels.
DEFINES = [
    "TLLM_GEN_EXPORT_INTERFACE",
    "ENABLE_BF16",
    "ENABLE_FP8",
    "ENABLE_FP4",
    "NDEBUG",
]


def includes(data):
    """The include paths the distribution's own build uses."""
    return [
        data / "csrc/nv_internal",
        data / "csrc/nv_internal/include",
        data / "csrc",
        data / "include",
        data / "cutlass/include",
        data / "cutlass/tools/util/include",
    ]


def compile_unit(compiler, source, out, data, target):
    """Compile one translation unit into ``out``."""
    command = [
        compiler,
        "-c",
        "-O3",
        "-std=c++17",
        f"-arch={target}",
        "--expt-relaxed-constexpr",
        "-Xcompiler",
        "-fPIC",
        *[f"-D{one}" for one in DEFINES],
        *[f"-I{one}" for one in includes(data)],
        "-o",
        str(out),
        str(source),
    ]
    trtllm.run(command, f"compiling {source}")


def compile(request: Request) -> None:
    """Compile the request's descriptor into ``module.so``.

    The objects are the build's own intermediates, in a dot directory of the
    request's output that the Build Server does not publish, so only
    ``module.so`` is.
    """
    package = request.package
    target = request.target
    data = trtllm.data_tree()
    # The nvcc the Build Server checked against this package's declared version.
    compiler = request.toolchain.require_cuda().nvcc

    with tempfile.TemporaryDirectory(dir=request.output, prefix=".") as scratch:
        staging = pathlib.Path(scratch)
        objects = []
        for relative in SOURCES:
            source = data / relative
            if not source.is_file():
                raise SystemExit(f"the distribution has no {relative}")
            unit = staging / (source.stem + ".o")
            compile_unit(compiler, source, unit, data, target)
            objects.append(unit)

        entry = staging / "entry.o"
        compile_unit(compiler, package / "src/entry.cpp", entry, data, target)
        objects.append(entry)

        trtllm.run(
            [
                compiler,
                "-shared",
                f"-arch={target}",
                *[str(one) for one in objects],
                "-lcudart",
                "-o",
                str(request.output / "module.so"),
            ],
            "linking module.so",
        )
