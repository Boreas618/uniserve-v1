"""The shared build of the packages that compile FlashInfer's TensorRT-LLM code.

FlashInfer's distribution carries TensorRT-LLM's launch code: the runners that
select and launch trtllm-gen's kernels, and kernels of its own (the MoE
combine). trtllm-gen's kernels ship as cubins plus a generated header carrying
their table (the `trtllm-gen-fmha` and `trtllm-gen-bmm` pins of
:mod:`uniserve_build_server.fetch`). A package compiles the launch code it
needs with its own C entry into ``module.so``, and a package launching
trtllm-gen kernels publishes the cubins its descriptor can select beside the
module, under :data:`CUBINS`, and reads them through the cubin loader of the
kernel packages' shared sources (``uniserve_trtllm/cubins.h`` of
``uniserve_kernels``), from the directory the loader hands its load entry.

This module is the one statement of what those packages share:

- the FlashInfer distribution: its data tree (:func:`data_tree`, found through
  the locked distribution's metadata) and the include directories in it
  (:data:`INCLUDES`);
- the vendor's compile definitions (:data:`TENSORRT_LLM`,
  :data:`FLASHINFER_TYPES`, :data:`BATCHED_GEMM`) and the translation units of
  each part of the launch code, with the definitions it is compiled with
  (:class:`LaunchCode`: :data:`COMBINE`, :data:`FMHA`, :data:`ROUTED_GEMM`);
- the one build routine (:func:`build`): a package's entry unit and the launch
  code's units as the CUDA driver's units, with the kernel packages' shared
  C++ sources on the include path, compiled and linked into ``module.so`` by
  :func:`uniserve_build_server.cuda.native`;
- staging a pinned trtllm-gen library (:func:`stage`);
- the libraries' tables read into typed kernels (:func:`fmha_table`,
  :func:`bmm_table`, with the vendor's enumerations, :func:`enumerations`),
  the one selection of a package's kernels by the options they were built
  with (:func:`select`), and the publication of their cubins
  (:func:`publish`);
- the whole build of the routed NVFP4 products (`gather_gemm_nvfp4`'s first
  matrix and `route_gemm_nvfp4`'s second), which differ only in their entry
  and in the options their manifests select.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.resources
import pathlib
import re
import shutil
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from uniserve_build_server import cuda, fetch, toolchain
from uniserve_build_server.request import Request, Target

#: The distribution whose data tree holds the launch code.
FLASHINFER = "flashinfer-python"

#: The directory of a module that holds the cubins it publishes. The launch
#: code's loader prefixes it to a cubin's name (:attr:`LaunchCode.cubin_path`),
#: and the module's loader reads that name in the module's directory.
CUBINS = "cubins"


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
    reads, the package's entry unit included. Launch code that launches
    trtllm-gen kernels names the pinned ``library`` of them and the macro
    whose value its loader prefixes to a cubin's name (``cubin_path``), which
    the vendor's headers refuse to compile without.
    """

    units: tuple[str, ...]
    definitions: Mapping[str, cuda.Definition]
    library: str | None = None
    cubin_path: str | None = None


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
    cubin_path="TLLM_GEN_FMHA_CUBIN_PATH",
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
    cubin_path="TLLM_GEN_GEMM_CUBIN_PATH",
)


def headers(request: Request, library: str) -> list[str]:
    """The headers a pinned trtllm-gen library publishes, by their paths in it.

    They are its manifest's files under ``include/``: the library's kernel
    table and, for the batched-GEMM library, the export headers its launch
    code includes, which declare the table's struct and the enumerations of
    its options.
    """
    try:
        files = fetch.manifest(fetch.pin(library), request.cache)
    except fetch.FetchError as error:
        raise SystemExit(f"reading the {library} manifest: {error}") from error
    return sorted(name for name in files if name.startswith("include/"))


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
    first), the kernel packages' shared C++ sources and then the data tree's
    :data:`INCLUDES`; launch code with a cubin path defines it as
    :data:`CUBINS`. Every unit is C++17 in the language its file names, as
    FlashInfer's own build compiles it, and lets device code call constexpr
    host functions, as CUTLASS's headers need.

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
    shared = {**code.definitions, **(definitions or {})}
    if code.cubin_path is not None:
        shared[code.cubin_path] = f'"{CUBINS}"'
    sources = [
        (
            request.package / request.unit,
            {**shared, **(entry_definitions or {})},
        ),
        *((data / unit, shared) for unit in code.units),
    ]
    for source, _ in sources:
        if not source.is_file():
            raise SystemExit(f"{request.op}/{request.backend} has no {source}")

    with importlib.resources.as_file(
        importlib.resources.files(toolchain.KERNELS).joinpath("include")
    ) as kernel_sources:
        search = [*includes, kernel_sources, *(data / one for one in INCLUDES)]
        units = [
            cuda.Unit(
                source=source,
                standard="c++17",
                definitions=unit_definitions,
                includes=search,
                relaxed_constexpr=True,
            )
            for source, unit_definitions in sources
        ]
        cuda.native(request, units, libraries=["cuda"])


# The tables of trtllm-gen's libraries.


@dataclasses.dataclass(frozen=True)
class Enumerator:
    """One enumerator of a vendor enumeration: the enumeration and its value."""

    enum: str
    value: int


#: An option's value: as a table states it, or as a package states what it
#: selects. A string is a string literal, an identifier that names no
#: enumerator, or an initializer this reader does not interpret (``{}``).
Value = int | bool | str | Enumerator | None


@dataclasses.dataclass(frozen=True)
class Kernel:
    """One kernel of a trtllm-gen library's table."""

    #: The kernel's function name in its cubin.
    name: str
    #: The cubin's file name in the library.
    cubin: str
    #: The architecture family it was compiled for, as targets name families
    #: (``sm_100f``).
    family: str
    #: The sha256 of the cubin, as the table states it.
    sha256: str
    #: Every option the kernel was built with, by the table's field name.
    options: dict[str, Value]
    #: The table entry's initializer, as the table spells it.
    entry: str


#: Comments, and the preprocessor's lines with their continuations.
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_DIRECTIVE = re.compile(r"^[ \t]*#(?:[^\n]*\\\n)*[^\n]*$", re.M)

#: What decides an initializer's structure: string literals and comments
#: (whose commas and braces are text), brackets and commas.
_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|/\*.*?\*/|[{}(),]', re.S)


def _declarations(text: str) -> str:
    """A header's declarations: its text without comments and directives."""
    return _DIRECTIVE.sub("", _COMMENT.sub(" ", text))


def _parts(text: str) -> list[str]:
    """The comma-separated parts of an initializer's contents, at its level.

    A comma inside braces, parentheses, a string literal or a comment
    separates nothing at this level.
    """
    parts, depth, start = [], 0, 0
    for token in _TOKEN.finditer(text):
        found = token.group()
        if found in ("{", "("):
            depth += 1
        elif found in ("}", ")"):
            depth -= 1
        elif found == "," and depth == 0:
            parts.append(text[start : token.start()].strip())
            start = token.end()
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _closing(text: str, opening: int) -> int:
    """The index of the brace that closes the one at ``opening``."""
    depth = 0
    for token in _TOKEN.finditer(text, opening):
        found = token.group()
        if found in ("{", "("):
            depth += 1
        elif found in ("}", ")"):
            depth -= 1
            if depth == 0:
                return token.start()
    raise SystemExit("an initializer of the table is not closed")


def _encode_dtype(
    block: int, signed: int, integer: int, bits: int, uid: int
) -> int:
    """`trtllm::gen::Dtype`'s value of an element type.

    The batched-GEMM library declares every element type through its macro
    ``TLLM_ENCODE_DTYPE(BlockFormatBit, SignedBit, IntegerBit, NumBits,
    Uid)`` (``trtllm/gen/DtypeDecl.h``), whose value this is.
    """
    return (block << 24) | (signed << 20) | (integer << 16) | (bits << 8) | uid


def _evaluate(expression: str, known: dict[str, int]) -> int:
    """An enumerator's initializer, evaluated.

    It is a number, an earlier enumerator of the same enumeration, or an
    element type's encoding.
    """
    number = re.fullmatch(r"(0[xX][0-9a-fA-F]+|\d+)[uUlL]*", expression)
    if number is not None:
        return int(number.group(1), 0)
    if expression in known:
        return known[expression]
    encoded = re.fullmatch(r"TLLM_ENCODE_DTYPE\s*\((.*)\)", expression, re.S)
    if encoded is not None:
        return _encode_dtype(
            *(_evaluate(one, known) for one in _parts(encoded.group(1)))
        )
    raise ValueError(f"cannot evaluate {expression!r}")


@dataclasses.dataclass(frozen=True)
class Enumerations:
    """The enumerations vendor headers declare, by name.

    An enumeration whose initializers this reader cannot evaluate is known
    by name with the reason, and refused when a selection names it.
    """

    values: dict[str, dict[str, int]]
    #: The enumerators of the unscoped enumerations (`enum`, not
    #: `enum class`), which a table names bare.
    unscoped: dict[str, Enumerator]
    unreadable: dict[str, str]

    def enumerator(self, qualified: str) -> Enumerator:
        """The enumerator ``Enum::Name`` states, or a refusal naming why."""
        enum, _, name = qualified.partition("::")
        if enum in self.unreadable:
            raise SystemExit(
                f"{qualified}: the enumeration {enum} is not one this reader "
                f"evaluates ({self.unreadable[enum]})"
            )
        if name not in self.values.get(enum, {}):
            raise SystemExit(f"the vendor's headers declare no {qualified}")
        return Enumerator(enum, self.values[enum][name])


def enumerations(texts: Iterable[str]) -> Enumerations:
    """Every enumeration ``texts`` declare, with its enumerators' values."""
    values: dict[str, dict[str, int]] = {}
    unscoped: dict[str, Enumerator] = {}
    unreadable: dict[str, str] = {}
    declaration = re.compile(
        r"\benum\s+(class\s+|struct\s+)?(\w+)\s*(?::[^{;]*)?\{([^{}]*)\}\s*;"
    )
    for text in texts:
        for found in declaration.finditer(_declarations(text)):
            scoped, enum, body = found.groups()
            known: dict[str, int] = {}
            following = 0
            try:
                for item in _parts(body):
                    name, _, expression = item.partition("=")
                    if expression.strip():
                        following = _evaluate(expression.strip(), known)
                    known[name.strip()] = following
                    following += 1
            except ValueError as error:
                unreadable[enum] = str(error)
                continue
            if values.get(enum, known) != known:
                unreadable[enum] = "two declarations state different values"
                continue
            values[enum] = known
            if not scoped:
                unscoped.update(
                    (name, Enumerator(enum, value))
                    for name, value in known.items()
                )
    return Enumerations(values, unscoped, unreadable)


def _literal(text: str, enums: Enumerations) -> Value:
    """One value of a table entry, typed."""
    if text.startswith('"'):
        return text[1:-1]
    if text in ("true", "false"):
        return text == "true"
    if text == "nullptr":
        return None
    number = re.fullmatch(r"(-?\d+)[uUlL]*", text)
    if number is not None:
        return int(number.group(1))
    # `gemm::BiasType(1)`: an enumerator as its enumeration and value.
    cast = re.fullmatch(r"(?:\w+::)*(\w+)\((-?\d+)\)", text)
    if cast is not None:
        return Enumerator(cast.group(1), int(cast.group(2)))
    if text in enums.unscoped:
        return enums.unscoped[text]
    return text


def _fields(header: str, struct: str) -> list[str]:
    """The data members of ``struct``, in declaration order."""
    text = _declarations(header)
    found = re.search(r"\bstruct\s+" + struct + r"\s*\{", text)
    if found is None:
        raise SystemExit(f"the table's headers declare no struct {struct}")
    body = text[found.end() : _closing(text, found.end() - 1)]
    members = []
    for statement in body.split(";"):
        # A member is the last name of its declaration, before a default
        # initializer (`uint32_t mSize{0}`); a member function is none.
        declared = re.search(r"(\w+)\s*(?:\{[^{}]*\}|=[^=]*)?\s*$", statement)
        if declared is not None and "(" not in statement:
            members.append(declared.group(1))
    return members


def _entries(header: str, struct: str) -> list[str]:
    """The initializers of the table array whose elements are ``struct``."""
    found = re.search(r"\b" + struct + r"\s+\w+\[\]\s*=\s*\{", header)
    if found is None:
        raise SystemExit(f"the header carries no table of {struct}")
    contents = header[found.end() : _closing(header, found.end() - 1)]
    # The table guards each architecture's entries with a directive
    # (`#ifndef EXCLUDE_SM_100`); every entry is in the library either way.
    return _parts(_DIRECTIVE.sub("", contents))


def _record(fields: list[str], entry: str) -> dict[str, str]:
    """One entry's values, by the struct's members in order."""
    values = _parts(entry[1:-1])
    if len(values) != len(fields):
        raise SystemExit(
            f"a table entry has {len(values)} values and its struct "
            f"{len(fields)} members: {entry[:120]}"
        )
    return dict(zip(fields, values, strict=True))


def fmha_table(header: str, enums: Enumerations) -> list[Kernel]:
    """The kernels of the FMHA library's table (`flashInferMetaInfo.h`).

    An entry is a `TllmGenFmhaKernelMetaInfo`, every member positional;
    ``enums`` are FlashInfer's enumerations its members take (the element
    types' `Data_type` among them). The library's loader names a cubin by
    the kernel's name (`fmhaKernels.cuh`). Its member `mSM` is FlashInfer's
    constant of the architecture (`trtllm/common.h`): `kSM_100` for the
    architecture-specific kernels of `sm_100a`, `kSM_100f` for the
    family's.
    """
    struct = "TllmGenFmhaKernelMetaInfo"
    fields = _fields(header, struct)
    kernels = []
    for entry in _entries(header, struct):
        options = {
            field: _literal(value, enums)
            for field, value in _record(fields, entry).items()
        }
        architecture = re.fullmatch(r"kSM_(\d+)(f?)", str(options["mSM"]))
        if architecture is None:
            raise SystemExit(f"a kernel's architecture is {options['mSM']}")
        number, family = architecture.groups()
        name = str(options["mFuncName"])
        kernels.append(
            Kernel(
                name=name,
                cubin=f"{name}.cubin",
                family=f"sm_{number}{family or 'a'}",
                sha256=str(options["sha256"]),
                options=options,
                entry=entry,
            )
        )
    return kernels


def bmm_table(
    header: str, declarations: str, enums: Enumerations
) -> list[Kernel]:
    """The kernels of the batched-GEMM library's table.

    An entry is a `BatchedGemmConfig` (declared in ``declarations``, the
    library's export headers): positional members, of which `mOptions` is the
    options record, each of its fields labelled `/* mName */ value`, and
    `mSm` the architecture (`gemm::SmVersion::Sm100f`). The library's loader
    names a cubin by the kernel's name with its first letter capitalized
    (`loadCubinData`, `GemmOptions.h`).
    """
    fields = _fields(declarations, "BatchedGemmConfig")
    kernels = []
    for entry in _entries(header, "BatchedGemmConfig"):
        record = _record(fields, entry)
        options = {}
        for part in _parts(record["mOptions"][1:-1]):
            labelled = re.fullmatch(r"/\*\s*(\w+)\s*\*/\s*(.*)", part, re.S)
            if labelled is None:
                raise SystemExit(f"an option is not labelled: {part[:80]}")
            options[labelled.group(1)] = _literal(labelled.group(2), enums)
        architecture = re.fullmatch(r"(?:\w+::)*Sm(\d+)([af])", record["mSm"])
        if architecture is None:
            raise SystemExit(f"a kernel's architecture is {record['mSm']}")
        name = str(_literal(record["mFunctionName"], enums))
        kernels.append(
            Kernel(
                name=name,
                cubin=name[0].upper() + name[1:] + ".cubin",
                family="sm_" + "".join(architecture.groups()),
                sha256=str(_literal(record["mHash"], enums)),
                options=options,
                entry=entry,
            )
        )
    return kernels


#: A stated option: one value, or any of several.
Stated = Value | frozenset


def resolve(stated: Any, enums: Enumerations) -> Stated:
    """The value a package states, typed.

    ``Enum::Name`` is that enumerator of the vendor's headers, and a list
    states any of its values.
    """
    if isinstance(stated, list | tuple | frozenset):
        return frozenset(resolve(one, enums) for one in stated)
    if isinstance(stated, str) and "::" in stated:
        return enums.enumerator(stated)
    return stated


def _matches(stated: Stated, actual: Value) -> bool:
    if isinstance(stated, frozenset):
        return any(_matches(one, actual) for one in stated)
    if isinstance(stated, Enumerator) and not isinstance(actual, Enumerator):
        # The FMHA table states its enumerated members as `int`s.
        return type(actual) is int and actual == stated.value
    return type(stated) is type(actual) and stated == actual


def select(
    kernels: Sequence[Kernel], options: dict[str, Stated], target: Target
) -> list[Kernel]:
    """The kernels built with every one of ``options``, for ``target``.

    A kernel is selected by the options the table states it was built with
    and the architecture family it was compiled for, never by its name: a
    name is a rendering of the options, and a selection that reads it
    changes silently when the vendor renders a name differently. An option
    the table has no member of is refused, as is a selection no kernel meets.

    The kernels keep the table's order: a filtered table is the runner's
    configuration list, whose order its choice among valid tactics follows.
    """
    known = kernels[0].options if kernels else {}
    unknown = sorted(set(options) - set(known))
    if unknown:
        raise SystemExit(f"the table has no option {', '.join(unknown)}")
    kept = [
        kernel
        for kernel in kernels
        if kernel.family in target.families
        and all(
            _matches(value, kernel.options[name])
            for name, value in options.items()
        )
    ]
    if not kept:
        raise SystemExit(
            f"the library has no kernel for {target} with the options "
            + ", ".join(f"{name} = {value}" for name, value in options.items())
        )
    return kept


def publish(
    request: Request,
    library: str,
    kernels: Sequence[Kernel],
    staging: pathlib.Path,
) -> None:
    """Publishes the kernels' cubins beside the module, under :data:`CUBINS`.

    Each cubin is staged from the pinned library into ``staging`` (verified
    against the pin's manifest) and copied into the module's output, its own
    bytes and not a link into the cache of pins. A cubin whose digest in the
    manifest is not the hash the library's table states for its kernel is
    refused: the module's loader reads the published file in place of the
    hash the launch code hands it.
    """
    stage(request, library, staging, [kernel.cubin for kernel in kernels])
    try:
        digests = fetch.manifest(fetch.pin(library), request.cache)
    except fetch.FetchError as error:
        raise SystemExit(f"reading the {library} manifest: {error}") from error
    for kernel in kernels:
        if digests[kernel.cubin] != kernel.sha256:
            raise SystemExit(
                f"{library} lists {kernel.cubin} with sha256 "
                f"{digests[kernel.cubin]} and its table states {kernel.sha256}"
            )
    published = request.output / CUBINS
    published.mkdir(parents=True, exist_ok=True)
    for kernel in kernels:
        shutil.copyfile(staging / kernel.cubin, published / kernel.cubin)
    print(f"published {len(kernels)} cubins", file=sys.stderr)


# The batched-GEMM library's routed NVFP4 products.

#: The batched-GEMM library's kernel table.
BMM_TABLE = "include/flashinferMetaInfo.h"


def filtered_header(original: str, kernels: Sequence[Kernel]) -> str:
    """The library's table, filtered to the kernels this module published."""
    # The library states its own count just before the table; the filtered
    # table states its own, so the preamble ends where the library's count
    # begins.
    preamble = original[
        : original.find("static constexpr size_t tllmGenBatchedGemmListLen")
    ]
    preamble = preamble[: preamble.rfind("\n", 0, len(preamble)) + 1]
    collection = (
        "static const batchedGemm::BatchedGemmConfig "
        "tllmGenBatchedGemmList[] = {\n"
        + "\n".join(kernel.entry + "," for kernel in kernels)
        + "\n};\n"
    )
    # The library closes the namespaces it opened after the table; dropping
    # the tail would leave every later include inside them, and the
    # interface's own types would then be unqualified.
    suffix = original[original.rfind("};") + 2 :]
    return (
        preamble
        + "static constexpr size_t tllmGenBatchedGemmListLen = "
        + f"{len(kernels)};\n\n"
        + collection
        + suffix
    )


def manifest_selected(parameters: dict, fields: dict[str, Any]) -> dict:
    """The options the manifest's ``[parameters.selected]`` states.

    An option the descriptor decides is stated as the field it takes its
    value from with the value each of the field's values selects:
    ``{ field = "activation", values = { silu = "ActType::SwiGlu" } }``.
    """
    selected = parameters.get("selected")
    if not selected:
        raise SystemExit("the manifest states no [parameters.selected] options")
    stated = {}
    for name, value in selected.items():
        if not isinstance(value, dict):
            stated[name] = value
            continue
        taken = str(fields[value["field"]])
        if taken not in value["values"]:
            raise SystemExit(
                f"{name} states no value for {value['field']} = {taken}"
            )
        stated[name] = value["values"][taken]
    return stated


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
    entry_definitions: Mapping[str, cuda.Definition] | None = None,
) -> None:
    """Compiles one descriptor of a routed NVFP4 batched-GEMM package.

    This is the package's whole compilation: it writes ``module.so`` with its
    cubins into the request's output directory.

    Steps:

    1. stage the `trtllm-gen-bmm` library's headers (:func:`headers`: its
       table and export headers), each verified against the pin;
    2. select the kernels (:func:`select`) by the options the manifest's
       `[parameters.selected]` states and the descriptor's MMA tile and
       cluster, on the target's architecture families;
    3. publish the selected cubins under :data:`CUBINS` (:func:`publish`) and
       write a filtered table into the compile's include path, so the
       runner's configuration list is exactly this descriptor's tactics: a
       cubin that is not published cannot be selected;
    4. compile the entry unit and :data:`ROUTED_GEMM` and link ``module.so``
       (:func:`build`).

    ``entry_definitions`` are the preprocessor definitions only one
    package's entry unit reads. The entry writes the padded route count with
    `cuMemsetD32Async`, a driver call of its own.
    """
    library = ROUTED_GEMM.library
    with cuda.scratch(request) as staging:
        published = headers(request, library)
        stage(request, library, staging, published)
        header = (staging / BMM_TABLE).read_text()
        declarations = "\n".join(
            (staging / name).read_text()
            for name in published
            if name != BMM_TABLE
        )
        enums = enumerations([declarations])
        options = {
            name: resolve(value, enums)
            for name, value in manifest_selected(
                request.parameters, request.fields
            ).items()
        }
        # The MMA tile and the cluster are the descriptor's own.
        rows, columns = request.field("tile")
        cluster_m, cluster_n = request.field("cluster")
        options.update(
            mTileM=rows,
            mTileN=columns,
            mClusterDimX=cluster_m,
            mClusterDimY=cluster_n,
        )
        kernels = select(
            bmm_table(header, declarations, enums), options, request.target
        )
        publish(request, library, kernels, staging)

        # The compile reads a table of this descriptor's tactics and nothing
        # else. The staged files are links into the cache of pins, which every
        # other build shares and which is content-addressed by the library's
        # own hash, so a write through them would corrupt it. The filtered
        # table therefore gets a file and an include directory of its own,
        # searched first.
        filtered = staging / "filtered/include"
        filtered.mkdir(parents=True, exist_ok=True)
        (filtered / "flashinferMetaInfo.h").write_text(
            filtered_header(header, kernels)
        )

        build(
            request,
            ROUTED_GEMM,
            includes=artifact_includes(staging),
            entry_definitions={
                **geometry(request.fields),
                **(entry_definitions or {}),
            },
        )
