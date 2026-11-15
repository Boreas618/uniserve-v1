"""The shared build of the packages that compile FlashInfer's TensorRT-LLM code.

FlashInfer's distribution carries TensorRT-LLM's launch code: the runners that
select and launch trtllm-gen's kernels, and kernels of its own (the MoE
combine). trtllm-gen's kernels ship as cubins plus a generated header carrying
their table (the `trtllm-gen-fmha` and `trtllm-gen-bmm` pins of
:mod:`uniserve_build_server.fetch`). A package compiles the launch code it
needs with its own C entry into ``module.so``, and a package launching
trtllm-gen kernels publishes the cubins its descriptor can select beside the
module under ``cubins/``.

This module is the one statement of what those packages share:

- the FlashInfer distribution: its data tree (:func:`data_tree`, found through
  the locked distribution's metadata) and the include directories in it
  (:data:`INCLUDES`);
- the vendor's compile definitions (:data:`TENSORRT_LLM`,
  :data:`FLASHINFER_TYPES`, :data:`BATCHED_GEMM`) and the translation units of
  each part of the launch code, with the definitions it is compiled with
  (:class:`LaunchCode`: :data:`COMBINE`, :data:`FMHA`, :data:`ROUTED_GEMM`);
- the one build routine (:func:`build`): a package's entry unit and the launch
  code's units as the CUDA driver's units, compiled and linked into
  ``module.so`` by :func:`uniserve_build_server.cuda.native`;
- staging a pinned trtllm-gen library (:func:`stage`), and the whole build of
  the routed NVFP4 products (`gather_gemm_nvfp4`'s first matrix and
  `route_gemm_nvfp4`'s second), which differ only in their entry and in the
  options their manifests select.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import pathlib
import re
import shutil
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from uniserve_build_server import cuda, fetch
from uniserve_build_server.request import Request, Target

#: The distribution whose data tree holds the launch code.
FLASHINFER = "flashinfer-python"


def families(target: Target) -> tuple[str, ...]:
    """The architecture families of ``target``, as trtllm-gen spells them.

    A device of ``sm_100a`` runs the family-conditional ``Sm100f`` kernels
    beside the specific ``Sm100a`` ones: the target's families are both.
    """
    return tuple(
        "Sm" + family.removeprefix("sm_") for family in target.families
    )


def data_tree() -> pathlib.Path:
    """The locked FlashInfer distribution's data tree."""
    try:
        installed = importlib.metadata.distribution(FLASHINFER)
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(
            f"{sys.executable} has no {FLASHINFER} distribution"
        ) from None
    return pathlib.Path(installed.locate_file("flashinfer/data"))


#: The include directories of the data tree, relative to it: FlashInfer's own
#: headers and sources, TensorRT-LLM's (``csrc/nv_internal``), CUTLASS, and
#: the CCCL FlashInfer is written against, which is searched before the
#: toolkit's own copy as FlashInfer's build searches it.
INCLUDES = (
    "include",
    "csrc",
    "csrc/nv_internal",
    "csrc/nv_internal/include",
    "cutlass/include",
    "cutlass/tools/util/include",
    "cccl/libcudacxx/include",
    "cccl/cub",
    "cccl/thrust",
)

#: The definitions TensorRT-LLM's sources are compiled with in FlashInfer's
#: own build: the export interface (the trtllm-gen headers include each other
#: relative to themselves), the BF16, FP8 and FP4 element types, and no
#: assertions.
TENSORRT_LLM: Mapping[str, cuda.Definition] = {
    "TLLM_GEN_EXPORT_INTERFACE": None,
    "ENABLE_BF16": None,
    "ENABLE_FP8": None,
    "ENABLE_FP4": None,
    "NDEBUG": None,
}

#: The element types FlashInfer's own headers instantiate their code for.
FLASHINFER_TYPES: Mapping[str, cuda.Definition] = {
    "FLASHINFER_ENABLE_F16": None,
    "FLASHINFER_ENABLE_BF16": None,
    "FLASHINFER_ENABLE_FP8_E4M3": None,
    "FLASHINFER_ENABLE_FP8_E5M2": None,
    "FLASHINFER_ENABLE_FP8_E8M0": None,
    "FLASHINFER_ENABLE_FP4_E2M1": None,
}

#: The batched-GEMM library's FlashInfer export: its kernels are loaded from
#: cubins through ``getCubin`` rather than linked in. The library's
#: `KernelParamsSetup` -- the function that turns an options record into the
#: pointers and launch geometry a cubin takes -- is inside
#: ``#ifdef TLLM_ENABLE_CUDA``, so a build that leaves that macro off sees an
#: empty namespace and fails on the runner's own call into it.
BATCHED_GEMM: Mapping[str, cuda.Definition] = {
    "TLLM_GEN_EXPORT_FLASHINFER": None,
    "TLLM_ENABLE_CUDA": None,
}

#: TensorRT-LLM's commons -- the exception class, the logger, the string
#: formatting and the environment flags -- which its runners and checks call.
#: They are sources of the distribution, not a library the host is assumed to
#: have: compiled in, the module has no unresolved reference; left out, it
#: builds, publishes and then cannot be loaded.
COMMONS = (
    "csrc/nv_internal/cpp/common/tllmException.cpp",
    "csrc/nv_internal/cpp/common/stringUtils.cpp",
    "csrc/nv_internal/cpp/common/logger.cpp",
    "csrc/nv_internal/cpp/common/envUtils.cpp",
)


@dataclasses.dataclass(frozen=True)
class LaunchCode:
    """One part of the distribution's launch code a module compiles.

    ``units`` are its translation units, relative to the data tree, and
    ``definitions`` the definitions every unit of a module compiling it
    reads, the package's entry unit included. ``library`` is the pinned
    trtllm-gen library whose kernels it launches, for launch code that
    launches them.
    """

    units: tuple[str, ...]
    definitions: Mapping[str, cuda.Definition]
    library: str | None = None


#: trtllm-gen's MoE combine (`moeUnpermute`) and the commons its checks call.
COMBINE = LaunchCode(
    units=(
        "csrc/nv_internal/tensorrt_llm/kernels/cuteDslKernels/moeUtils.cu",
        *COMMONS,
    ),
    definitions=TENSORRT_LLM,
)

#: The FMHA launch code: `TllmGenFmhaRunner` is headers the entry includes,
#: and its reduction (`tensorrt_llm::kernels::runFmhaReduction`) the one unit
#: it calls into.
FMHA = LaunchCode(
    units=("csrc/fmhaReduction.cu",),
    definitions={**TENSORRT_LLM, **FLASHINFER_TYPES},
    library="trtllm-gen-fmha",
)

#: The routed MoE products' runners (`PermuteGemm1::Runner`,
#: `Gemm2::Runner`) with the kernel table they select, the batched-GEMM
#: runner they hold, and the commons both call.
ROUTED_GEMM = LaunchCode(
    units=(
        "csrc/trtllm_fused_moe_runner.cu",
        "csrc/trtllm_batched_gemm_runner.cu",
        *COMMONS,
    ),
    definitions={**TENSORRT_LLM, **BATCHED_GEMM},
    library="trtllm-gen-bmm",
)


def stage(
    request: Request, library: str, staged: pathlib.Path, names: Iterable[str]
) -> None:
    """Stages files of a pinned trtllm-gen library into ``staged``.

    Each is verified against the pin and staged as a link into the Build
    Server's cache of pins.
    """
    names = list(names)
    try:
        pinned = fetch.pin(library)
        fetch.fetch(pinned, request.cache, names, request.jobs)
        fetch.stage(pinned, request.cache, staged, names)
    except fetch.FetchError as error:
        raise SystemExit(
            f"staging the {library} artifact failed: {error}"
        ) from error


def build(
    request: Request,
    code: LaunchCode,
    includes: Sequence[pathlib.Path] = (),
    definitions: Mapping[str, cuda.Definition] | None = None,
    entry_definitions: Mapping[str, cuda.Definition] | None = None,
) -> None:
    """Compiles the package's entry unit and ``code`` into ``module.so``.

    The entry unit is the manifest's ``[source] unit``. Every unit reads the
    launch code's definitions and ``definitions``; the entry unit also reads
    ``entry_definitions``, the descriptor's values it is compiled for. The
    include path is ``includes`` (a staged library's headers, searched
    first) and then the data tree's :data:`INCLUDES`. Every unit is C++17 in
    the language its file names, as FlashInfer's own build compiles it, and
    lets device code call constexpr host functions, as CUTLASS's headers
    need.

    The module links the driver API beside the CUDA runtime: trtllm-gen's
    launcher calls `cuModuleLoadData`, `cuModuleGetFunction` and
    `cuLaunchKernelEx` directly.
    """
    if request.unit is None:
        raise SystemExit(
            f"{request.op}/{request.backend} names its entry unit in "
            "[source] unit"
        )
    data = data_tree()
    search = [*includes, *(data / one for one in INCLUDES)]
    shared = {**code.definitions, **(definitions or {})}
    sources = [
        (
            request.package / request.unit,
            {**shared, **(entry_definitions or {})},
        ),
        *((data / unit, shared) for unit in code.units),
    ]
    units = []
    for source, unit_definitions in sources:
        if not source.is_file():
            raise SystemExit(f"{request.op}/{request.backend} has no {source}")
        units.append(
            cuda.Unit(
                source=source,
                standard="c++17",
                definitions=unit_definitions,
                includes=search,
                relaxed_constexpr=True,
            )
        )
    cuda.native(request, units, libraries=["cuda"])


# The batched-GEMM library's routed NVFP4 products.

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


def selection(blocks, fields, selected, target: Target, renderings):
    """The kernel names this descriptor's options select.

    The filter is the options the kernels were built with, as the manifest's
    `[parameters.selected]` states them, and nothing else -- not a suffix of a
    name: a name is a rendering of those options, and a selection that reads it
    changes silently when the vendor reorders a suffix. Two options are the
    descriptor's rather than the manifest's: the MMA tile's rows and columns and
    the cluster.
    """
    runs = families(target)
    rows, columns = fields["tile"]
    cluster_m, cluster_n = fields["cluster"]
    expected = {}
    for name, value in selected.items():
        render = renderings.get(name)
        expected[name] = render(value) if render is not None else str(value)
    kept = []
    for name, block in blocks:
        if family_of(block) not in runs:
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


def manifest_selected(parameters: dict, fields: dict[str, Any]) -> dict:
    """The options the manifest states the package's kernels were built with.

    An option the descriptor decides is stated as the field it takes its value
    from, ``{ field = "activation" }``, and reads that field.
    """
    selected = parameters.get("selected")
    if not selected:
        raise SystemExit("the manifest states no [parameters.selected] options")
    return {
        name: fields[value["field"]] if isinstance(value, dict) else value
        for name, value in selected.items()
    }


def cubin_name(name: str) -> str:
    """The artifact's cubin file name for one kernel: its name, capitalized."""
    return name[0].upper() + name[1:] + ".cubin"


def artifact_includes(artifact: pathlib.Path) -> list[pathlib.Path]:
    """The staged artifact's include paths, its filtered table first."""
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
    return [artifact / "filtered/include", artifact, artifact / "include"]


def geometry(fields: dict[str, Any]) -> dict[str, cuda.Definition]:
    """The descriptor's own values, as the entry's compile-time definitions."""
    rows, columns = fields["tile"]
    cluster_m, cluster_n = fields["cluster"]
    return {
        "UNISERVE_HIDDEN": fields["hidden"],
        "UNISERVE_INTERMEDIATE": fields["intermediate"],
        "UNISERVE_EXPERTS": fields["experts"],
        "UNISERVE_TOP": fields["top"],
        "UNISERVE_TILE_TOKENS": rows,
        "UNISERVE_TILE_COLUMNS": columns,
        "UNISERVE_CLUSTER_ROWS": cluster_m,
        "UNISERVE_CLUSTER_COLUMNS": cluster_n,
    }


def build_routed_gemm(
    request: Request,
    renderings: dict[str, Callable[[str], str]] | None = None,
    entry_definitions: Mapping[str, cuda.Definition] | None = None,
) -> None:
    """Compiles one descriptor of a routed NVFP4 batched-GEMM package.

    This is the package's whole compilation: it writes ``module.so`` with its
    cubins into the request's output directory.

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
    4. compile the entry unit and :data:`ROUTED_GEMM` and link ``module.so``
       (:func:`build`).

    ``renderings`` adds the enum renderings of options only one package's
    manifest selects, and ``entry_definitions`` the preprocessor definitions
    only one package's entry unit reads. The entry writes the padded route count
    with `cuMemsetD32Async`, a driver call of its own.
    """
    out = request.output
    renderings = {**RENDERINGS, **(renderings or {})}
    with cuda.scratch(request) as staging:
        stage(
            request,
            ROUTED_GEMM.library,
            staging,
            ["include/flashinferMetaInfo.h", *EXPORT_HEADERS],
        )
        header = (staging / "include/flashinferMetaInfo.h").read_text()
        blocks = entry_blocks(header)
        kept = selection(
            blocks,
            request.fields,
            manifest_selected(request.parameters, request.fields),
            request.target,
            renderings,
        )
        stage(
            request,
            ROUTED_GEMM.library,
            staging,
            [cubin_name(name) for name in kept],
        )
        published = out / "cubins"
        published.mkdir(parents=True, exist_ok=True)
        for name in kept:
            shutil.copyfile(
                staging / cubin_name(name), published / cubin_name(name)
            )
        print(f"published {len(kept)} cubins", file=sys.stderr)

        # The compile reads a table of this descriptor's tactics and nothing
        # else. The staged files are links into the cache of pins, which every
        # other build shares and which is content-addressed by the artifact's
        # own hash, so a write through them would corrupt it. The filtered
        # table therefore gets a file and an include directory of its own,
        # searched first.
        filtered = staging / "filtered/include"
        filtered.mkdir(parents=True, exist_ok=True)
        (filtered / "flashinferMetaInfo.h").write_text(
            filtered_header(header, blocks, kept)
        )

        build(
            request,
            ROUTED_GEMM,
            includes=artifact_includes(staging),
            definitions={"TLLM_GEN_GEMM_CUBIN_PATH": '"."'},
            entry_definitions={
                **geometry(request.fields),
                **(entry_definitions or {}),
            },
        )
