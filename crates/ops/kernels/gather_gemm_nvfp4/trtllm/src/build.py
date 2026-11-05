"""The gather_gemm_nvfp4 package's trtllm-gen compile script: one descriptor -> one module.

The module is trtllm-gen's routed, block-scaled first matrix -- the launch code
FlashInfer calls `PermuteGemm1::Runner` -- compiled out of the FlashInfer
distribution's data tree against the pinned `trtllm-gen-bmm` artifact, with this
package's C entry in front of it and the artifact's cubins published beside the
module under `cubins/`.

Steps:
1. stage the artifact: `include/flashinferMetaInfo.h`, the export headers and the
   cubins this descriptor selects, through the Program Server's own fetcher
   (`python -m uniserve_program_server.trtllm_gen --artifact trtllm-gen-bmm`), which
   verifies every file against the pin;
2. select the kernels: the artifact's table is parsed entry by entry and kept by its
   **options** -- the operand and output element types, the MMA tile, the cluster, the
   activation, the block-scale layouts and the architecture family -- never by
   spelling the name, because a name is a rendering of those options and a selection
   that reads it silently changes when the vendor renames a suffix;
3. write a filtered `flashinferMetaInfo.h` into the compile's include path, so the
   runner's configuration list is exactly this descriptor's tactics: a cubin that is
   not published cannot be selected, and one that is published is one this module's
   geometry and layouts allow;
4. compile the entry unit and the two vendor units whose symbols the entry's runner
   resolves, and link `module.so`.

The module needs the CUDA driver (`libcuda`) as well as the runtime: trtllm-gen's
launcher calls `cuModuleLoadData`, `cuModuleGetFunction` and `cuLaunchKernelEx`
directly, and the entry writes the padded route count with `cuMemsetD32Async`.

A descriptor is the operator's: hidden, intermediate, experts, top, activation, dtype,
scale, layout, tile, cluster.
"""

import argparse
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

#: The operator's element types, as the artifact's table names them.
NAMES = {"bf16": "Bfloat16", "f16": "Fp16", "f8e4m3": "E4m3", "e2m1": "E2m1", "u8": "E2m1"}

#: The architecture families each target runs. A GB200 reports SM 100 and runs the
#: family-conditional `Sm100f` kernels beside the specific `Sm100a` ones.
FAMILIES = {"sm_100a": ("Sm100a", "Sm100f"), "sm_100f": ("Sm100f",)}

#: The gated activation each of the operator's activations is, in the artifact's own
#: enumeration (`gemmGatedAct::ActType`).
ACTIVATIONS = {"silu": 0, "gelu_tanh": 1, "none": 3}

#: The bias types (`gemm::BiasType`). A gated product's BF16-output kernels all take `M`: one f32
#: per output column, which this operator has none of, so the entry hands them zeros.
BIAS_TYPES = {"none": 0, "m": 1, "n": 2, "mn": 3}

#: The block-scale layouts (`trtllm::gen::SfLayout`). The artifact serves the
#: activation side in the 128x4 swizzled form on every E2M1 kernel, and the weight
#: side in either the plain or the 8x4 form; this package serves the plain one, which
#: is what a ModelOpt shard stores.
SF_LAYOUTS = {"linear": 0, "8x4": 1, "8x16": 2, "swizzled_128x4": 3}

#: The compile defines the artifact's own build uses.
DEFINES = [
    "TLLM_GEN_EXPORT_INTERFACE",
    "TLLM_GEN_EXPORT_FLASHINFER",
    # The artifact's `KernelParamsSetup` -- the function that turns an options record into the
    # pointers and launch geometry a cubin takes -- is inside `#ifdef TLLM_ENABLE_CUDA`, so a build
    # that leaves the macro off sees an empty namespace and fails on the runner's own call into it.
    "TLLM_ENABLE_CUDA",
    "ENABLE_BF16",
    "ENABLE_FP8",
    "ENABLE_FP4",
    "NDEBUG",
]

#: The vendor units the entry's runner resolves against. `PermuteGemm1::Runner` and
#: `Gemm2::Runner` live in the first; the batched-GEMM runner they hold in the second.
#: The runner units and the runtime they reference.
#:
#: A runner unit calls into TensorRT-LLM's commons - the exception class, the logger, the
#: string formatting and the two NVFP4 environment flags - and those are sources in this
#: distribution, not a library the host is assumed to have. Compiled here, the module has
#: no unresolved reference; omitted, it builds, publishes and then cannot be loaded, and
#: the failure arrives at the first launch naming no symbol at all.
SOURCES = [
    "csrc/trtllm_fused_moe_runner.cu",
    "csrc/trtllm_batched_gemm_runner.cu",
    "csrc/nv_internal/cpp/common/tllmException.cpp",
    "csrc/nv_internal/cpp/common/stringUtils.cpp",
    "csrc/nv_internal/cpp/common/logger.cpp",
    "csrc/nv_internal/cpp/common/envUtils.cpp",
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
    return (
        pathlib.Path(values["package"]),
        values["descriptor"],
        values["target"],
        pathlib.Path(values["out"]),
    )


def fields_of(package, descriptor):
    """Parse a descriptor key against the package manifest's dimensions."""
    import tomllib

    manifest = tomllib.loads((package / "package.toml").read_text())
    dimensions = manifest["kernel"]["descriptors"]["dimensions"]
    names = sorted(dimensions, key=len, reverse=True)
    fields = {}
    remaining = descriptor
    while remaining:
        name = next((one for one in names if remaining.startswith(one + "_")), None)
        if name is None:
            raise SystemExit(f"{descriptor}: no dimension starts {remaining!r}")
        rest = remaining[len(name) + 1 :]
        values = sorted(dimensions[name], key=len, reverse=True)
        value = next((one for one in values if rest == one or rest.startswith(one + "_")), None)
        if value is None:
            raise SystemExit(f"{descriptor}: {name} takes none of {dimensions[name]} at {rest!r}")
        fields[name] = value
        remaining = rest[len(value) :].removeprefix("_")
    missing = sorted(set(dimensions) - set(fields))
    if missing:
        raise SystemExit(f"{descriptor}: the descriptor names no {missing}")
    return fields


def distribution():
    """The FlashInfer distribution's data tree: the launch code's headers and units."""
    found = importlib.util.find_spec("flashinfer")
    roots = []
    if found is not None and found.submodule_search_locations:
        roots.extend(pathlib.Path(one) for one in found.submodule_search_locations)
    repository = pathlib.Path(__file__).resolve().parents[6]
    roots.extend(
        repository.glob("crates/program-server/python/.venv/lib/python*/site-packages/flashinfer")
    )
    roots.extend(repository.glob(".venv/lib/python*/site-packages/flashinfer"))
    for root in roots:
        data = root / "data"
        if (data / "include/flashinfer/trtllm/fused_moe/runner.h").is_file():
            return data
    raise SystemExit("no FlashInfer distribution with the fused-MoE launch code was found")


def fetcher_python():
    """The interpreter that carries the Program Server's artifact fetcher."""
    if importlib.util.find_spec("uniserve_program_server") is not None:
        return sys.executable
    repository = pathlib.Path(__file__).resolve().parents[6]
    for candidate in sorted(repository.glob("crates/program-server/python/.venv/bin/python*")):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise SystemExit("no interpreter carries uniserve_program_server's artifact fetcher")


def fetch(staged, names):
    """Stage the artifact's named files into ``staged`` through the pinned fetcher."""
    command = [
        fetcher_python(),
        "-m",
        "uniserve_program_server.trtllm_gen",
        "--artifact",
        "trtllm-gen-bmm",
        "--out",
        str(staged),
    ]
    for name in names:
        command.extend(["--cubin", name])
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    if run.returncode != 0:
        raise SystemExit(f"staging the trtllm-gen batched-GEMM artifact failed:\n{run.stderr[-4000:]}")


def entry_blocks(text):
    """The artifact's kernel table, entry by entry, as (name, block) pairs.

    The table is a C array of `{...}` initializers; an entry is one balanced brace
    block whose first string is the kernel's name. Parsing it structurally is what
    lets the selection read its options: every field is a `/* mField */ value` pair.
    """
    start = text.find("tllmGenBatchedGemmList[]")
    if start < 0:
        raise SystemExit("the artifact's header carries no tllmGenBatchedGemmList")
    # The array's own brace opens the collection; every entry is a brace block inside it.
    index = text.find("{", text.find("=", start))
    index = text.find("{", index + 1)
    blocks = []
    while True:
        index = text.find("{", index)
        if index < 0:
            break
        depth = 0
        end = index
        for end in range(index, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    break
        block = text[index : end + 1]
        if "mFunctionName" in block or re.search(r'"bmm_[^"]+"', block):
            name = re.search(r'"(bmm_[^"]+)"', block)
            if name is not None:
                blocks.append((name.group(1), block))
        index = end + 1
    return blocks


def option(block, name):
    """One option's value, as the artifact's `/* mName */ value` pair states it."""
    found = re.search(r"/\* " + name + r" \*/ ([^,\n]+)", block)
    return found.group(1).strip() if found is not None else None


def dtype_code(type_name):
    """`trtllm::gen::Dtype`'s encoding of one type's name."""
    table = {
        "Bfloat16": (0, 1, 0, 16, 0),
        "E2m1": (1, 1, 0, 4, 2),
        "E4m3": (0, 1, 0, 8, 5),
        "Fp16": (0, 1, 0, 16, 7),
        "Fp32": (0, 1, 0, 32, 8),
    }
    block, signed, integer, bits, uid = table[type_name]
    return (block << 24) | (signed << 20) | (integer << 16) | (bits << 8) | uid


def sealed(block, name, expected):
    """Whether an option's encoded value is the one expected."""
    text = option(block, name)
    if text is None:
        return False
    found = re.search(r"\((\d+)\)", text)
    return found is not None and int(found.group(1)) == expected


def family_of(block):
    """The architecture family a kernel was compiled for, as the table states it.

    The sm version is the entry's own trailing field (`gemm::SmVersion::Sm100f`), not a
    suffix of the kernel's name: the name is a rendering of the options, the field is
    the option.
    """
    found = re.search(r"gemm::SmVersion::(\w+)", block)
    return found.group(1) if found is not None else None


#: The enum renderings the artifact's table states a selection's values in.
RENDERINGS = {
    "mDtypeA": lambda name: f"trtllm::gen::Dtype({dtype_code(name)})",
    "mDtypeB": lambda name: f"trtllm::gen::Dtype({dtype_code(name)})",
    "mDtypeC": lambda name: f"trtllm::gen::Dtype({dtype_code(name)})",
    "mDtypeAcc": lambda name: f"trtllm::gen::Dtype({dtype_code(name)})",
    "mBiasDtype": lambda name: f"trtllm::gen::Dtype({dtype_code(name)})",
    "mSfLayoutA": lambda name: f"trtllm::gen::SfLayout({SF_LAYOUTS[name]})",
    "mSfLayoutB": lambda name: f"trtllm::gen::SfLayout({SF_LAYOUTS[name]})",
    "mSfLayoutC": lambda name: f"trtllm::gen::SfLayout({SF_LAYOUTS[name]})",
    "mBiasType": lambda name: f"gemm::BiasType({BIAS_TYPES[name]})",
    "mPerTokenSfDtype": lambda name: f"trtllm::gen::Dtype({dtype_code(name)})",
    "mActType": lambda name: f"gemmGatedAct::ActType({ACTIVATIONS[name]})",
}


def selection(blocks, fields, selected, target):
    """The kernel names this descriptor's options select.

    The filter is the options the kernels were built with, as the manifest's `[parameters.selected]`
    states them, and nothing else -- not a suffix of a name: a name is a rendering of those options,
    and a selection that reads it changes silently when the vendor reorders a suffix. Two options are
    the descriptor's rather than the manifest's: the MMA tile's rows and columns and the cluster.
    """
    families = FAMILIES.get(target)
    if families is None:
        raise SystemExit(f"{target} names no trtllm-gen architecture family")
    rows, columns = (int(one) for one in fields["tile"].split("x"))
    cluster_m, cluster_n = (int(one) for one in fields["cluster"].split("x"))
    expected = {}
    for name, value in selected.items():
        render = RENDERINGS.get(name)
        expected[name] = render(value) if render is not None else str(value)
    kept = []
    for name, block in blocks:
        if family_of(block) not in families:
            continue
        if option(block, "mTileM") != str(rows) or option(block, "mTileN") != str(columns):
            continue
        if option(block, "mClusterDimX") != str(cluster_m):
            continue
        if option(block, "mClusterDimY") != str(cluster_n):
            continue
        if any(option(block, option_name) != value for option_name, value in expected.items()):
            continue
        kept.append(name)
    if not kept:
        raise SystemExit(
            f"the artifact has no kernel for tile {rows}x{columns}, cluster {cluster_m}x{cluster_n} "
            f"on {target} with the options {json.dumps(selected, sort_keys=True)}"
        )
    return sorted(set(kept))


def filtered_header(original, blocks, kept):
    """The artifact's table, filtered to the kernels this module published."""
    chosen = [block for name, block in blocks if name in set(kept)]
    # The artifact states its own count just before the table; the filtered table states its own, so
    # the preamble ends where the artifact's count begins.
    preamble = original[: original.find("static constexpr size_t tllmGenBatchedGemmListLen")]
    preamble = preamble[: preamble.rfind("\n", 0, len(preamble)) + 1]
    collection = (
        "static const batchedGemm::BatchedGemmConfig tllmGenBatchedGemmList[] = {\n"
        + "\n".join(block + "," for block in chosen)
        + "\n};\n"
    )
    # The artifact closes the namespaces it opened after the table; dropping the tail would leave
    # every later include inside them, and the interface's own types would then be unqualified.
    suffix = original[original.rfind("};") + 2 :]
    return (
        preamble
        + f"static constexpr size_t tllmGenBatchedGemmListLen = {len(chosen)};\n\n"
        + collection
        + suffix
    )


def manifest_selected(package):
    """The options the manifest states this package's kernels must have been built with."""
    import tomllib

    manifest = tomllib.loads((package / "package.toml").read_text())
    selected = manifest.get("parameters", {}).get("selected")
    if not selected:
        raise SystemExit("the manifest states no [parameters.selected] options")
    return selected


def cubin_name(name):
    """The artifact's cubin file name for one kernel: its name, first letter raised."""
    return name[0].upper() + name[1:] + ".cubin"


def compiler_and_version():
    """The CUDA compiler this build names, and the version line it prints."""
    candidates = [os.environ.get("CUDACXX"), shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"]
    tried = []
    for candidate in candidates:
        if not candidate or not pathlib.Path(candidate).is_file():
            continue
        answered = subprocess.run([candidate, "--version"], capture_output=True, text=True)
        if answered.returncode == 0 and "Cuda compilation tools" in answered.stdout:
            return candidate, answered.stdout.strip().splitlines()[-1].strip()
        tried.append(f"{candidate}: {answered.stdout[-300:]}{answered.stderr[-300:]}")
    raise SystemExit(
        "no CUDA compiler: name one with CUDACXX, or put nvcc on PATH"
        + (f"\ntried:\n" + "\n".join(tried) if tried else "")
    )


def headers(artifact, data):
    """The include paths, the artifact's exported headers first."""
    # The vendor's launch code includes the artifact's headers under the distribution's
    # own path, so the export tree is linked where it looks for it: the artifact is the
    # build's input, not the module's, and never enters the directory a runner is handed.
    facade = artifact / "flashinfer/trtllm/batched_gemm"
    facade.mkdir(parents=True, exist_ok=True)
    target = facade / "trtllmGen_bmm_export"
    if not target.exists():
        target.symlink_to((artifact / "include/trtllmGen_bmm_export").resolve(), target_is_directory=True)
    return [
        artifact / "filtered/include",
        artifact,
        artifact / "include",
        data / "include",
        data / "csrc",
        data / "csrc/nv_internal",
        data / "csrc/nv_internal/include",
        data / "cutlass/include",
        data / "cutlass/tools/util/include",
        data / "cccl/libcudacxx/include",
        data / "cccl/cub",
        data / "cccl/thrust",
    ]


def geometry(package, fields):
    """The descriptor's own values, as the entry's compile-time constants."""
    rows, columns = (int(one) for one in fields["tile"].split("x"))
    cluster_m, cluster_n = (int(one) for one in fields["cluster"].split("x"))
    return [
        f"-DUNISERVE_HIDDEN={fields['hidden']}",
        f"-DUNISERVE_INTERMEDIATE={fields['intermediate']}",
        f"-DUNISERVE_EXPERTS={fields['experts']}",
        f"-DUNISERVE_TOP={fields['top']}",
        f"-DUNISERVE_TILE_TOKENS={rows}",
        f"-DUNISERVE_TILE_COLUMNS={columns}",
        f"-DUNISERVE_CLUSTER_ROWS={cluster_m}",
        f"-DUNISERVE_CLUSTER_COLUMNS={cluster_n}",
    ]


def main():
    """Compile one descriptor into ``module.so`` with its cubins."""
    package, descriptor, target, out = arguments()
    fields = fields_of(package, descriptor)
    data = distribution()
    out.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix="uniserve-trtllm-bmm-"))
    try:
        build(package, descriptor, fields, target, out, data, staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def build(package, descriptor, fields, target, out, data, staging):
    """Select, publish and compile one descriptor's module."""
    fetch(staging, ["include/flashinferMetaInfo.h"])
    fetch(
        staging,
        [
            f"include/trtllmGen_bmm_export/{name}"
            for name in (
                "BatchedGemmEnums.h",
                "BatchedGemmInterface.h",
                "BatchedGemmOptions.h",
                "Enums.h",
                "GemmGatedActOptions.h",
                "GemmOptions.h",
                "KernelParams.h",
                "KernelParamsDecl.h",
                "KernelTraits.h",
                "TmaDescriptor.h",
                "trtllm/gen/CommonUtils.h",
                "trtllm/gen/CudaArchDecl.h",
                "trtllm/gen/CudaKernelLauncher.h",
                "trtllm/gen/DtypeDecl.h",
                "trtllm/gen/MmaDecl.h",
                "trtllm/gen/SfLayoutDecl.h",
                "trtllm/gen/SparsityDecl.h",
            )
        ],
    )
    header = (staging / "include/flashinferMetaInfo.h").read_text()
    blocks = entry_blocks(header)
    kept = selection(blocks, fields, manifest_selected(package), target)
    fetch(staging, [cubin_name(name) for name in kept])
    published = out / "cubins"
    published.mkdir(parents=True, exist_ok=True)
    for name in kept:
        source = staging / cubin_name(name)
        if not source.is_file():
            raise SystemExit(f"the fetcher staged no {source}")
        shutil.copyfile(source, published / cubin_name(name))
    print(f"published {len(kept)} cubins", file=sys.stderr)

    # The compile reads a table of this descriptor's tactics and nothing else.
    # The fetcher hardlinks its cache into the staging tree, so a write through that path would
    # rewrite the artifact cache -- which every other module and host shares, and which is
    # content-addressed by the artifact's own hash. The filtered table therefore gets a file and an
    # include directory of its own, searched first.
    filtered = staging / "filtered/include"
    filtered.mkdir(parents=True, exist_ok=True)
    (filtered / "flashinferMetaInfo.h").write_text(filtered_header(header, blocks, kept))

    compiler, version = compiler_and_version()
    unit = out / "entry.o"
    command = [
        compiler,
        "-c",
        "-x",
        "cu",
        "-O3",
        "-std=c++17",
        f"-arch={target}",
        "--expt-relaxed-constexpr",
        "-Xcompiler",
        "-fPIC",
        *[f"-D{one}" for one in DEFINES],
        *geometry(package, fields),
        '-DTLLM_GEN_GEMM_CUBIN_PATH="."',
        *[f"-I{one}" for one in headers(staging, data)],
        "-o",
        str(unit),
        str(package / "src/entry.cpp"),
    ]
    compiled = subprocess.run(command, capture_output=True, text=True, check=False)
    if compiled.returncode != 0:
        raise SystemExit(
            f"compiling the entry unit failed with {compiler} ({version}):\n"
            f"{' '.join(command)}\n{compiled.stderr[-6000:]}"
        )

    objects = [unit]
    for relative in SOURCES:
        source = data / relative
        if not source.is_file():
            raise SystemExit(f"the distribution has no {relative}")
        object_unit = out / (source.stem + ".o")
        # The vendor units carry no descriptor: they are the launch code, compiled once.
        vendor = [one for one in command if not one.startswith("-DUNISERVE_")]
        vendor[vendor.index("-o") + 1] = str(object_unit)
        vendor[vendor.index(str(package / "src/entry.cpp"))] = str(source)
        built = subprocess.run(vendor, capture_output=True, text=True, check=False)
        if built.returncode != 0:
            raise SystemExit(
                f"compiling {relative} failed with {compiler} ({version}):\n"
                f"{' '.join(vendor)}\n{built.stderr[:8000]}"
            )
        objects.append(object_unit)

    # trtllm-gen's launcher calls the CUDA driver API directly, and the entry writes the
    # padded route count through it; the toolkit's stub is a link-time input only.
    stub = pathlib.Path("/usr/local/cuda/lib64/stubs")
    if not (stub / "libcuda.so").is_file():
        raise SystemExit(f"no libcuda to link against: the toolkit's stub is missing from {stub}")
    module = out / "module.so"
    linked = subprocess.run(
        [
            compiler,
            "-shared",
            f"-arch={target}",
            *[str(one) for one in objects],
            f"-L{stub}",
            "-lcuda",
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
        raise SystemExit(f"linking module.so failed:\n{linked.stderr[-6000:]}")

    exported = subprocess.run(
        ["nm", "-D", "--defined-only", str(module)], capture_output=True, text=True, check=True
    )
    symbols = {line.split()[-1] for line in exported.stdout.splitlines() if line.split()}
    if "uniserve_gather_gemm_nvfp4" not in symbols:
        raise SystemExit("module.so does not export uniserve_gather_gemm_nvfp4")
    print(json.dumps({"descriptor": descriptor, "target": target, "cubins": len(kept)}))


if __name__ == "__main__":
    main()
