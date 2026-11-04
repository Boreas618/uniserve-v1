"""The moe_unpermute package's compile script: one descriptor -> one module.

The module is trtllm-gen's combine kernel, compiled from the FlashInfer
distribution's sources into a shared object with this package's C entry in front
of it. A descriptor is the problem's hidden width, top-k expansion, expert count
and the three element types; the kernel reads them at launch, so every
descriptor of this package compiles the same module.

The sources come from the FlashInfer package the toolchain pins (its `data`
tree holds the trtllm-gen kernels and their headers), not from this tree: the
kernel is NVIDIA's, and the package's `[toolchain] packages` names where it comes
from.
"""

import argparse
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys

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
    package = pathlib.Path(values["package"])
    return package, values["descriptor"], values["target"], pathlib.Path(values["out"])


def distribution():
    """The FlashInfer `data` tree the kernels and their headers live in."""
    found = importlib.util.find_spec("flashinfer")
    roots = []
    if found is not None and found.submodule_search_locations:
        roots.extend(pathlib.Path(one) for one in found.submodule_search_locations)
    # The Program Server's own Python environment pins FlashInfer even when this
    # script runs under another interpreter.
    repository = pathlib.Path(__file__).resolve().parents[5]
    roots.extend(repository.glob("crates/program-server/python/.venv/lib/python*/site-packages/flashinfer"))
    roots.extend(repository.glob(".venv/lib/python*/site-packages/flashinfer"))
    for root in roots:
        data = root / "data"
        if (data / "csrc").is_dir():
            return data
    raise SystemExit("no FlashInfer distribution with its data tree was found")


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
    built = subprocess.run(command, capture_output=True, text=True, check=False)
    if built.returncode != 0:
        raise SystemExit(f"compiling {source} failed:\n{built.stderr[-4000:]}")


def main():
    """Compile one descriptor into ``module.so``."""
    package, descriptor, target, out = arguments()
    data = distribution()
    out.mkdir(parents=True, exist_ok=True)
    staging = out / "objects"
    staging.mkdir(exist_ok=True)
    compiler = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"

    objects = []
    for relative in SOURCES:
        source = data / relative
        if not source.is_file():
            raise SystemExit(f"the distribution has no {relative}")
        unit = staging / (source.stem + ".o")
        print(f"compiling {relative}", file=sys.stderr)
        compile_unit(compiler, source, unit, data, target)
        objects.append(unit)

    entry = staging / "entry.o"
    compile_unit(compiler, package / "src/entry.cpp", entry, data, target)
    objects.append(entry)

    module = out / "module.so"
    linked = subprocess.run(
        [
            compiler,
            "-shared",
            f"-arch={target}",
            *[str(one) for one in objects],
            "-L/usr/local/cuda/lib64",
            "-lcudart",
            "-o",
            str(module),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if linked.returncode != 0:
        raise SystemExit(f"linking module.so failed:\n{linked.stderr[-4000:]}")
    shutil.rmtree(staging, ignore_errors=True)

    exported = subprocess.run(
        ["nm", "-D", "--defined-only", str(module)], capture_output=True, text=True, check=True
    )
    if "uniserve_moe_unpermute" not in {line.split()[-1] for line in exported.stdout.splitlines() if line.split()}:
        raise SystemExit("module.so does not export uniserve_moe_unpermute")
    print(json.dumps({"descriptor": descriptor, "target": target, "entry": "uniserve_moe_unpermute"}))


if __name__ == "__main__":
    main()
