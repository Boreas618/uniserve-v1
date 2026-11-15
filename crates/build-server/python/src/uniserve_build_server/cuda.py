"""The CUDA compile driver: nvcc from the request's toolchain description.

Every CUDA package's module is compiled here, one way:

- a package that declares its compilation (``[compile]`` in its manifest,
  ``crates/build-server/src/compile.rs``) is compiled by :func:`build` from its
  request alone, with no code of its own;
- a package whose compilation is computed from what it compiles against (a
  vendor's kernel table it selects from, a launcher another compiler exports,
  a pinned source tree) keeps a build module, ``src/build.py``, whose
  ``compile(request)`` decides its translation units, their definitions and
  include directories, and hands them to this module's routines instead of
  writing a command line.

The routines are this module's stable interface:

- :class:`Unit`: one translation unit and what it is compiled with;
- :func:`native`: compile units into objects and link them into the module's
  ``module.so``, a native library launched through its C entries;
- :func:`cubin`: compile one unit into ``module.cubin`` and write its launch
  metadata from a :class:`Kernel`;
- :func:`objects` and :func:`link`: the two steps of :func:`native`, for a
  build that links objects it made otherwise (a CuTe DSL export) or links
  several modules;
- :func:`scratch`: the build's directory for intermediates;
- :func:`includes` and :func:`library`: the include directories every native
  compile gets, and where the toolchain description states a library.

Every compilation runs the nvcc of the request's toolchain description at
``-O3`` for the request's target, compiles position-independent host code
with the include directories every native compile gets, runs at most the
request's ``jobs`` compiler processes at once, keeps its intermediates in a
scratch directory of the output the Build Server does not publish, and prints
each command and the compiler's whole output into the build's log; a failed
step refuses the build, naming it.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import json
import pathlib
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from uniserve_build_server import fetch
from uniserve_build_server.binding import argument_type, constant
from uniserve_build_server.request import Request

#: The optimization level of every unit.
OPTIMIZATION = "-O3"

#: The name ``ops`` loads a native module by, and a cubin module and its
#: launch metadata by (``crates/build-server/src/source.rs``).
MODULE_SO = "module.so"
MODULE_CUBIN = "module.cubin"
METADATA = "metadata.json"

#: The value of a definition: an integer, a boolean (defined as 1 or 0), the
#: text of a token sequence such as an enumerator's name, or ``None`` for a
#: macro defined without a value.
Definition = int | bool | str | None


@dataclasses.dataclass(frozen=True)
class Unit:
    """One translation unit and what it is compiled with.

    ``standard`` is the C++ standard as nvcc spells it (``c++17``);
    ``definitions`` are its preprocessor definitions by macro name;
    ``includes`` are include directories searched after the ones every native
    compile gets (:func:`includes`). ``cuda`` compiles a file as CUDA whatever
    its extension says, for a vendor's unit named ``.cpp``;
    ``relaxed_constexpr`` lets device code call constexpr host functions, as
    CuTe's host helpers need; ``host_contraction = False`` compiles host code
    with every floating-point operation rounded.
    """

    source: pathlib.Path
    standard: str
    definitions: Mapping[str, Definition] = dataclasses.field(
        default_factory=dict
    )
    includes: Sequence[pathlib.Path] = ()
    cuda: bool = False
    relaxed_constexpr: bool = False
    host_contraction: bool = True


@dataclasses.dataclass(frozen=True)
class Kernel:
    """The kernel of a cubin module, as its launch metadata states it.

    ``signature`` maps each parameter the kernel takes at a launch, in its
    parameter order, to its type as a module's signature spells one (``*bf16``,
    ``i32``); ``shared`` is a block's dynamic shared memory in bytes.
    """

    name: str
    warps: int
    signature: Mapping[str, str]
    shared: int = 0


@dataclasses.dataclass(frozen=True)
class Library:
    """Where the toolchain description states one library a module links."""

    #: The directories its headers are in, when they are not the toolkit's.
    includes: tuple[pathlib.Path, ...]
    #: The directories the linker finds it in.
    directories: tuple[pathlib.Path, ...]
    #: Its name as the linker spells it (``-l<name>``).
    name: str


def includes(request: Request) -> list[pathlib.Path]:
    """The include directories every native compile gets: the package's own
    ``include/``, which holds its operator's C header.
    """  # noqa: D205
    return [request.package / "include"]


def library(request: Request, name: str) -> Library:
    """Where the request's toolchain description states the library ``name``
    (the names ``crates/build-server/src/compile.rs`` declares).

    ``cuda`` is the driver API, linked against the toolkit's stub so the
    serving process loads the driver's own library; ``cublasLt`` is a library
    of the toolkit; ``cudnn`` is the distribution package the description
    names, whose library the serving process's loader finds by the same
    default search.
    """  # noqa: D205
    cuda = request.toolchain.require_cuda()
    if name == "cuda":
        return Library((), (pathlib.Path(cuda.stubs),), "cuda")
    if name == "cublasLt":
        return Library((), (pathlib.Path(cuda.lib),), "cublasLt")
    if name == "cudnn":
        cudnn = request.toolchain.require_cudnn()
        return Library(
            (pathlib.Path(cudnn.include),), (pathlib.Path(cudnn.lib),), "cudnn"
        )
    raise SystemExit(
        f"{name!r} is no library the toolchain description states; the "
        "libraries are cuda, cublasLt and cudnn"
    )


@contextlib.contextmanager
def scratch(request: Request) -> Iterator[pathlib.Path]:
    """A directory for the build's intermediates, removed when it ends: a dot
    directory of the request's output, which the Build Server does not publish.
    """  # noqa: D205
    with tempfile.TemporaryDirectory(dir=request.output, prefix=".") as made:
        yield pathlib.Path(made)


def definition(name: str, value: Definition) -> str:
    """The ``-D`` argument defining ``name`` as ``value``."""
    if value is None:
        return f"-D{name}"
    if isinstance(value, bool):
        return f"-D{name}={int(value)}"
    if isinstance(value, int | str):
        return f"-D{name}={value}"
    raise SystemExit(
        f"the definition {name} = {value!r} is no integer, boolean or token"
    )


def compile_command(
    request: Request, unit: Unit, output: pathlib.Path, *, cubin: bool = False
) -> list[str]:
    """The nvcc command compiling ``unit`` into ``output``: an object, or a
    cubin when ``cubin`` is set.
    """  # noqa: D205
    command = [
        request.toolchain.require_cuda().nvcc,
        "-cubin" if cubin else "-c",
        OPTIMIZATION,
        f"-std={unit.standard}",
        f"-arch={request.target}",
    ]
    if unit.cuda:
        command += ["-x", "cu"]
    if unit.relaxed_constexpr:
        command.append("--expt-relaxed-constexpr")
    if not cubin:
        command += ["-Xcompiler", "-fPIC"]
        if not unit.host_contraction:
            command += ["-Xcompiler", "-ffp-contract=off"]
    command += [f"-I{one}" for one in [*includes(request), *unit.includes]]
    command += [
        definition(name, value) for name, value in unit.definitions.items()
    ]
    command += ["-o", str(output), str(unit.source)]
    return command


def run(request: Request, steps: Sequence[tuple[list[str], str]]) -> None:
    """Run each ``(command, what)`` step, the request's ``jobs`` at once.

    Each command and its whole output are printed into the build's log when it
    ends, so concurrent steps do not interleave; the first step that failed
    refuses the build, naming what it was.
    """

    def one(step: tuple[list[str], str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            step[0], capture_output=True, text=True, check=False
        )

    with concurrent.futures.ThreadPoolExecutor(max(1, request.jobs)) as pool:
        completed = list(pool.map(one, steps))
    failed = None
    for (command, what), result in zip(steps, completed, strict=True):
        print(" ".join(command), file=sys.stderr)
        print(result.stdout, end="", file=sys.stderr)
        print(result.stderr, end="", file=sys.stderr, flush=True)
        if result.returncode != 0 and failed is None:
            failed = f"{what} failed with {result.returncode}"
    if failed is not None:
        raise SystemExit(failed)


def objects(
    request: Request, units: Sequence[Unit], directory: pathlib.Path
) -> list[pathlib.Path]:
    """Compile each unit into an object in ``directory``; the objects, in the
    units' order.
    """  # noqa: D205
    made = [
        directory / f"{index}-{unit.source.name}.o"
        for index, unit in enumerate(units)
    ]
    run(
        request,
        [
            (compile_command(request, unit, output), f"compiling {unit.source}")
            for unit, output in zip(units, made, strict=True)
        ],
    )
    return made


def link(
    request: Request,
    linked: Sequence[pathlib.Path],
    output: pathlib.Path,
    *,
    standard: str,
    libraries: Sequence[str] = (),
) -> None:
    """Link objects into the shared library ``output`` with the named
    libraries (:func:`library`) and the CUDA runtime.

    nvcc compiles a registration stub of its own into every shared library it
    links, with the optimization, the standard and the position independence
    the link states, so those are the units' (``standard``): a module linked
    from objects is then the module one nvcc command would have compiled.
    """  # noqa: D205
    found = [library(request, name) for name in libraries]
    command = [
        request.toolchain.require_cuda().nvcc,
        "--shared",
        OPTIMIZATION,
        f"-std={standard}",
        f"-arch={request.target}",
        "-Xcompiler",
        "-fPIC",
        *[str(one) for one in linked],
        *[f"-L{directory}" for one in found for directory in one.directories],
        *[f"-l{one.name}" for one in found],
        "-lcudart",
        "-o",
        str(output),
    ]
    run(request, [(command, f"linking {output.name}")])


def native(
    request: Request, units: Sequence[Unit], libraries: Sequence[str] = ()
) -> None:
    """Compile ``units`` and link them with the named libraries into the
    request's ``module.so``; a unit that names a library's headers gets them
    (:func:`library`).
    """  # noqa: D205
    headers = [
        directory
        for name in libraries
        for directory in library(request, name).includes
    ]
    standards = {unit.standard for unit in units}
    if len(standards) != 1:
        raise SystemExit(
            f"the units of one module are written in one C++ standard, not "
            f"{sorted(standards)}"
        )
    with scratch(request) as directory:
        made = objects(
            request,
            [
                dataclasses.replace(unit, includes=[*unit.includes, *headers])
                for unit in units
            ],
            directory,
        )
        link(
            request,
            made,
            request.output / MODULE_SO,
            standard=standards.pop(),
            libraries=libraries,
        )


def cubin(request: Request, unit: Unit, kernel: Kernel) -> None:
    """Compile ``unit`` into ``module.cubin`` and write its launch metadata.

    The metadata is what ``ops`` reads of a cubin module
    (``ops::CubinMetadata``): the kernel's symbol, its warps, one block per
    cluster, its dynamic shared memory, an ordinary launch without global or
    profile scratch, and its signature in parameter order; it also records the
    definitions the unit was compiled with, the module's compile constants.
    """
    output = request.output
    run(
        request,
        [
            (
                compile_command(
                    request, unit, output / MODULE_CUBIN, cubin=True
                ),
                f"compiling {unit.source}",
            )
        ],
    )
    metadata = {
        "name": kernel.name,
        "num_warps": kernel.warps,
        "num_ctas": 1,
        "shared": kernel.shared,
        "launch_pdl": False,
        "launch_cooperative_grid": False,
        "global_scratch_size": 0,
        "global_scratch_align": 1,
        "profile_scratch_size": 0,
        "profile_scratch_align": 1,
        "signature": dict(kernel.signature),
        "definitions": dict(unit.definitions),
    }
    (output / METADATA).write_text(json.dumps(metadata, indent=2) + "\n")


def build(request: Request) -> None:
    """Compile the descriptor ``request`` names by its declared compilation."""
    declared: dict[str, Any] | None = request.compile
    if declared is None:
        raise SystemExit("the request declares no compilation")

    def bound(stated: Any) -> Any:
        return constant(
            stated, request.fields, request.parameters, request.descriptor
        )

    definitions = {
        name: bound(stated) for name, stated in declared["definitions"].items()
    }
    try:
        headers = [
            fetch.installed(stated["distribution"], stated["path"])
            for stated in declared["headers"]
        ]
    except fetch.FetchError as error:
        raise SystemExit(str(error)) from error
    units = [
        Unit(
            source=request.package / relative,
            standard=declared["standard"],
            definitions=definitions,
            includes=headers,
            relaxed_constexpr=declared["relaxed_constexpr"],
            host_contraction=declared["host_contraction"],
        )
        for relative in declared["units"]
    ]
    stated = declared["kernel"]
    if stated is None:
        native(request, units, declared["libraries"])
        return
    (unit,) = units
    cubin(
        request,
        unit,
        Kernel(
            name=stated["name"],
            warps=bound(stated["warps"]),
            shared=stated["shared"],
            signature={
                name: argument_type(
                    stated_type, request.fields, request.descriptor
                )
                for name, stated_type in stated["signature"]
            },
        ),
    )
