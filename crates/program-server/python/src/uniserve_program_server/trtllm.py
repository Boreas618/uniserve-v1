"""The trtllm-gen packages' shared build.

FlashInfer's launch code is compiled against a pinned trtllm-gen kernel
library.

trtllm-gen's kernels ship as cubins plus a generated header carrying their table
(the `trtllm-gen-fmha` and `trtllm-gen-bmm` pins of
:mod:`uniserve_program_server.fetch`); FlashInfer's distribution carries the
launch code that selects and launches them. A package that launches them
compiles that launch code with its own C entry into ``module.so`` and publishes
the cubins its descriptor can select beside the module under ``cubins/``.

This module holds what those packages share: the distribution's data tree,
staging the pinned library, and the batched-GEMM build of the routed NVFP4
products (`gather_gemm_nvfp4`'s first matrix and `route_gemm_nvfp4`'s second),
which differ only in their entry and in the options their manifests select. The
attention package's kernel selection is its own.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable

from uniserve_program_server import fetch, package

#: The architecture families each target runs. A GB200 reports SM 100 and runs
#: the family-conditional `Sm100f` kernels beside the specific `Sm100a` ones.
FAMILIES = {"sm_100a": ("Sm100a", "Sm100f"), "sm_100f": ("Sm100f",)}


def distribution(marker: str) -> pathlib.Path:
    """The FlashInfer distribution's data tree that holds ``marker``.

    ``marker`` is a launch-code header the package compiles against.
    """
    found = importlib.util.find_spec("flashinfer")
    roots = found.submodule_search_locations if found is not None else None
    for root in roots or []:
        data = pathlib.Path(root) / "data"
        if (data / marker).is_file():
            return data
    raise SystemExit(
        f"no FlashInfer distribution with {marker} was found in the "
        "interpreter running this build "
        f"({sys.executable})"
    )


def stage(library: str, staged: pathlib.Path, names: Iterable[str]) -> None:
    """Stages files of a pinned trtllm-gen library into ``staged``.

    Each is verified against the pin and staged as a link into the artifact
    cache.
    """
    names = list(names)
    try:
        pinned = fetch.pin(library)
        fetch.fetch(pinned, names)
        fetch.stage(pinned, staged, names)
    except fetch.FetchError as error:
        raise SystemExit(
            f"staging the {library} artifact failed: {error}"
        ) from error


# The batched-GEMM library's routed NVFP4 products.

#: The batched-GEMM library's launch-code header, which marks the distribution's
#: data tree.
FUSED_MOE_RUNNER = "include/flashinfer/trtllm/fused_moe/runner.h"

#: The gated activation each of the operator's activations is, in the artifact's
#: own enumeration (`gemmGatedAct::ActType`).
ACTIVATIONS = {"silu": 0, "gelu_tanh": 1, "none": 3}

#: The bias types (`gemm::BiasType`). A gated product's BF16-output kernels all
#: take `M`: one f32 per output column, which these operators have none of, so
#: the entry hands them zeros.
BIAS_TYPES = {"none": 0, "m": 1, "n": 2, "mn": 3}

#: The block-scale layouts (`trtllm::gen::SfLayout`). The artifact serves the
#: activation side in the 128x4 swizzled form on every E2M1 kernel, and the
#: weight side in either the plain or the 8x4 form; these packages serve the
#: plain one, which is what a ModelOpt shard stores.
SF_LAYOUTS = {"linear": 0, "8x4": 1, "8x16": 2, "swizzled_128x4": 3}

#: The compile defines the artifact's own build uses.
DEFINES = [
    "TLLM_GEN_EXPORT_INTERFACE",
    "TLLM_GEN_EXPORT_FLASHINFER",
    # The artifact's `KernelParamsSetup` -- the function that turns an options
    # record into the pointers and launch geometry a cubin takes -- is inside
    # `#ifdef TLLM_ENABLE_CUDA`, so a build that leaves the macro off sees an
    # empty namespace and fails on the runner's own call into it.
    "TLLM_ENABLE_CUDA",
    "ENABLE_BF16",
    "ENABLE_FP8",
    "ENABLE_FP4",
    "NDEBUG",
]

#: The vendor units the entry's runner resolves against, and the runtime they
#: reference. The routed products' runners (`PermuteGemm1::Runner`,
#: `Gemm2::Runner`) and the kernel table they select live in the first; the
#: batched-GEMM runner they hold in the second.
#:
#: A runner unit calls into TensorRT-LLM's commons - the exception class, the
#: logger, the string formatting and the two NVFP4 environment flags - and those
#: are sources in this distribution, not a library the host is assumed to have.
#: Compiled here, the module has no unresolved reference; omitted, it builds,
#: publishes and then cannot be loaded, and the failure arrives at the first
#: launch naming no symbol at all.
SOURCES = [
    "csrc/trtllm_fused_moe_runner.cu",
    "csrc/trtllm_batched_gemm_runner.cu",
    "csrc/nv_internal/cpp/common/tllmException.cpp",
    "csrc/nv_internal/cpp/common/stringUtils.cpp",
    "csrc/nv_internal/cpp/common/logger.cpp",
    "csrc/nv_internal/cpp/common/envUtils.cpp",
]

#: The export headers of the batched-GEMM library the launch code includes.
EXPORT_HEADERS = [
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
]


def entry_blocks(text: str) -> list[tuple[str, str]]:
    """The artifact's kernel table, entry by entry, as (name, block) pairs.

    The table is a C array of `{...}` initializers; an entry is one balanced
    brace block whose first string is the kernel's name. Parsing it structurally
    is what lets the selection read its options: every field is a `/* mField */
    value` pair.
    """
    start = text.find("tllmGenBatchedGemmList[]")
    if start < 0:
        raise SystemExit(
            "the artifact's header carries no tllmGenBatchedGemmList"
        )
    # The array's own brace opens the collection; every entry is a brace block
    # inside it.
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


def option(block: str, name: str) -> str | None:
    """One option's value, as the table's `/* mName */ value` pair states it."""
    found = re.search(r"/\* " + name + r" \*/ ([^,\n]+)", block)
    return found.group(1).strip() if found is not None else None


def dtype_code(type_name: str) -> int:
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


def family_of(block: str) -> str | None:
    """The architecture family a kernel was compiled for, per the table.

    The sm version is the entry's own trailing field
    (`gemm::SmVersion::Sm100f`), not a suffix of the kernel's name: the name is
    a rendering of the options, the field is the option.
    """
    found = re.search(r"gemm::SmVersion::(\w+)", block)
    return found.group(1) if found is not None else None


#: The enum renderings the artifact's table states a selection's values in.
RENDERINGS: dict[str, Callable[[str], str]] = {
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


def selection(blocks, fields, selected, target, renderings):
    """The kernel names this descriptor's options select.

    The filter is the options the kernels were built with, as the manifest's
    `[parameters.selected]` states them, and nothing else -- not a suffix of a
    name: a name is a rendering of those options, and a selection that reads it
    changes silently when the vendor reorders a suffix. Two options are the
    descriptor's rather than the manifest's: the MMA tile's rows and columns and
    the cluster.
    """
    families = FAMILIES.get(target)
    if families is None:
        raise SystemExit(f"{target} names no trtllm-gen architecture family")
    rows, columns = (int(one) for one in fields["tile"].split("x"))
    cluster_m, cluster_n = (int(one) for one in fields["cluster"].split("x"))
    expected = {}
    for name, value in selected.items():
        render = renderings.get(name)
        expected[name] = render(value) if render is not None else str(value)
    kept = []
    for name, block in blocks:
        if family_of(block) not in families:
            continue
        if option(block, "mTileM") != str(rows) or option(
            block, "mTileN"
        ) != str(columns):
            continue
        if option(block, "mClusterDimX") != str(cluster_m):
            continue
        if option(block, "mClusterDimY") != str(cluster_n):
            continue
        if any(
            option(block, option_name) != value
            for option_name, value in expected.items()
        ):
            continue
        kept.append(name)
    if not kept:
        raise SystemExit(
            f"the artifact has no kernel for tile {rows}x{columns}, cluster "
            f"{cluster_m}x{cluster_n} on {target} with the options "
            f"{json.dumps(selected, sort_keys=True)}"
        )
    return sorted(set(kept))


def filtered_header(original: str, blocks, kept) -> str:
    """The artifact's table, filtered to the kernels this module published."""
    chosen = [block for name, block in blocks if name in set(kept)]
    # The artifact states its own count just before the table; the filtered
    # table states its own, so the preamble ends where the artifact's count
    # begins.
    preamble = original[
        : original.find("static constexpr size_t tllmGenBatchedGemmListLen")
    ]
    preamble = preamble[: preamble.rfind("\n", 0, len(preamble)) + 1]
    collection = (
        "static const batchedGemm::BatchedGemmConfig "
        "tllmGenBatchedGemmList[] = {\n"
        + "\n".join(block + "," for block in chosen)
        + "\n};\n"
    )
    # The artifact closes the namespaces it opened after the table; dropping the
    # tail would leave every later include inside them, and the interface's own
    # types would then be unqualified.
    suffix = original[original.rfind("};") + 2 :]
    return (
        preamble
        + "static constexpr size_t tllmGenBatchedGemmListLen = "
        + f"{len(chosen)};\n\n"
        + collection
        + suffix
    )


def manifest_selected(root: pathlib.Path) -> dict:
    """The options the manifest states the package's kernels were built with."""
    selected = package.manifest(root).get("parameters", {}).get("selected")
    if not selected:
        raise SystemExit("the manifest states no [parameters.selected] options")
    return selected


def cubin_name(name: str) -> str:
    """The artifact's cubin file name for one kernel: its name, capitalized."""
    return name[0].upper() + name[1:] + ".cubin"


def headers(artifact: pathlib.Path, data: pathlib.Path) -> list[pathlib.Path]:
    """The include paths, the artifact's exported headers first."""
    # The vendor's launch code includes the artifact's headers under the
    # distribution's own path, so the export tree is linked where it looks for
    # it: the artifact is the build's input, not the module's, and never enters
    # the directory a runner is handed.
    facade = artifact / "flashinfer/trtllm/batched_gemm"
    facade.mkdir(parents=True, exist_ok=True)
    target = facade / "trtllmGen_bmm_export"
    if not target.exists():
        target.symlink_to(
            (artifact / "include/trtllmGen_bmm_export").resolve(),
            target_is_directory=True,
        )
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


def geometry(fields: dict[str, str]) -> list[str]:
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


def build_routed_gemm(
    entry: str, renderings: dict[str, Callable[[str], str]] | None = None
) -> None:
    """Compiles one descriptor of a routed NVFP4 batched-GEMM package.

    This is the package's compile script: it writes ``module.so`` with its
    cubins.

    Steps:

    1. stage the `trtllm-gen-bmm` library's table
       (`include/flashinferMetaInfo.h`) and export headers, each verified
       against the pin;
    2. select the kernels: the table is parsed entry by entry and kept by its
       **options** -- the operand and output element types, the MMA tile, the
       cluster, the activation, the block-scale layouts and the architecture
       family -- never by spelling the name;
    3. publish the selected cubins under ``cubins/`` and write a filtered table
       into the compile's include path, so the runner's configuration list is
       exactly this descriptor's tactics: a cubin that is not published cannot
       be selected;
    4. compile the entry unit and the vendor units its runner resolves, and link
       ``module.so``.

    ``entry`` is the C entry the module must export; ``renderings`` adds the
    enum renderings of options only one package's manifest selects. The module
    needs the CUDA driver (`libcuda`) as well as the runtime: trtllm-gen's
    launcher calls `cuModuleLoadData`, `cuModuleGetFunction` and
    `cuLaunchKernelEx` directly, and the entry writes the padded route count
    with `cuMemsetD32Async`.
    """
    root, descriptor, target, out = package.arguments()
    fields = package.fields(root, descriptor)
    data = distribution(FUSED_MOE_RUNNER)
    out.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix="uniserve-trtllm-bmm-"))
    try:
        _build_routed_gemm(
            root,
            descriptor,
            fields,
            target,
            out,
            data,
            staging,
            entry,
            {**RENDERINGS, **(renderings or {})},
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _build_routed_gemm(
    root, descriptor, fields, target, out, data, staging, entry, renderings
):
    stage(
        "trtllm-gen-bmm",
        staging,
        ["include/flashinferMetaInfo.h", *EXPORT_HEADERS],
    )
    header = (staging / "include/flashinferMetaInfo.h").read_text()
    blocks = entry_blocks(header)
    kept = selection(
        blocks, fields, manifest_selected(root), target, renderings
    )
    stage("trtllm-gen-bmm", staging, [cubin_name(name) for name in kept])
    published = out / "cubins"
    published.mkdir(parents=True, exist_ok=True)
    for name in kept:
        source = staging / cubin_name(name)
        if not source.is_file():
            raise SystemExit(f"the fetcher staged no {source}")
        shutil.copyfile(source, published / cubin_name(name))
    print(f"published {len(kept)} cubins", file=sys.stderr)

    # The compile reads a table of this descriptor's tactics and nothing else.
    # The staging tree's files are links into the artifact cache, which every
    # other module and host shares and which is content-addressed by the
    # artifact's own hash, so a write through them would corrupt it. The
    # filtered table therefore gets a file and an include directory of its own,
    # searched first.
    filtered = staging / "filtered/include"
    filtered.mkdir(parents=True, exist_ok=True)
    (filtered / "flashinferMetaInfo.h").write_text(
        filtered_header(header, blocks, kept)
    )

    compiler, version = package.nvcc()
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
        *geometry(fields),
        '-DTLLM_GEN_GEMM_CUBIN_PATH="."',
        *[f"-I{one}" for one in headers(staging, data)],
        "-o",
        str(unit),
        str(root / "src/entry.cpp"),
    ]
    compiled = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
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
        # The vendor units carry no descriptor: they are the launch code,
        # compiled once.
        vendor = [one for one in command if not one.startswith("-DUNISERVE_")]
        vendor[vendor.index("-o") + 1] = str(object_unit)
        vendor[vendor.index(str(root / "src/entry.cpp"))] = str(source)
        built = subprocess.run(
            vendor, capture_output=True, text=True, check=False
        )
        if built.returncode != 0:
            raise SystemExit(
                f"compiling {relative} failed with {compiler} ({version}):\n"
                f"{' '.join(vendor)}\n{built.stderr[:8000]}"
            )
        objects.append(object_unit)

    # trtllm-gen's launcher calls the CUDA driver API directly, and the entry
    # writes the padded route count through it; the toolkit's stub is a
    # link-time input only, and nvcc's own profile puts the toolkit's
    # `lib/stubs` on the link line.
    module = out / "module.so"
    linked = subprocess.run(
        [
            compiler,
            "-shared",
            f"-arch={target}",
            *[str(one) for one in objects],
            "-lcuda",
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

    package.require_exports(module, [entry])
    print(
        json.dumps(
            {"descriptor": descriptor, "target": target, "cubins": len(kept)}
        )
    )
