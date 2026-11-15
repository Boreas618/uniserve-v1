"""The kernel packages as the kernel checks reach them, outside a build.

A check reaches a package of ``crates/ops/kernels/<op>/<backend>/`` in three
ways, each through the package's own files or the Build Server's interface:

- its kernel source, imported in this process from the package's ``src/``
  directory by its location (:func:`source_module`), so the kernels are
  decorated in the session's Triton mode (``tests/python/kernels/conftest.py``);
- the compilation a Triton package's launch table states for a descriptor key,
  as the Build Server states it for a build of that key (:class:`Launches`);
- a built module, resolved at the session's Build Server
  (:mod:`uniserve_testing.builds`), which builds it when its store holds none
  (:class:`Modules`).

A native module's call structure is declared by the C ABI headers the operator
crate generates into the kernel library (``uniserve_kernels``'s ``include/``,
``ops::abi``), each ending with the structure's size and every field offset as
Rust lays it out, asserted; a check's ctypes mirror of a call is held to those
assertions (:func:`assert_abi_layout`).
"""

from __future__ import annotations

import ctypes
import dataclasses
import importlib.machinery
import importlib.resources
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from types import ModuleType
from typing import IO, Any

from uniserve_testing.builds import BuildServer
from uniserve_testing.servers import environment


def source_module(package: Path, name: str) -> ModuleType:
    """The module ``name`` of a package's source directory, imported from it.

    A kernel package is a directory, which the Build Server installs by its
    location, not an installed distribution. The module is found in
    ``package / "src"`` alone, by the rule the Build Server's driver imports a
    package's kernel module by when it states a launch
    (``uniserve_build_server.triton.launches``: the standard path finder over
    the source directory, nothing added to the search path), and is executed
    here: every ``@triton.jit`` function it defines takes the mode Triton is
    in now.

    Raises:
        LookupError: When the source directory holds no such module.
    """
    source = package / "src"
    spec = importlib.machinery.PathFinder.find_spec(name, [str(source)])
    if spec is None:
        raise LookupError(f"{source} holds no module {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclasses.dataclass(frozen=True)
class AbiLayout:
    """A call structure's size and field offsets, in bytes, as its generated
    header asserts them.
    """  # noqa: D205

    size: int
    offsets: dict[str, int]


#: The kernel library and its directory of generated C ABI headers.
KERNEL_LIBRARY = "uniserve_kernels"
ABI_HEADERS = "include"


def abi_layout(structure: str) -> AbiLayout:
    """The size and field offsets the generated ABI header of the installed
    kernel library asserts for the C structure ``structure``.

    The headers are the ones every native module is compiled against; their
    ``static_assert(sizeof(...) == ...)`` and ``static_assert(offsetof(...,
    field) == ...)`` lines state the layout the operator crate passes.

    Raises:
        LookupError: When no generated header asserts the structure's size.
    """  # noqa: D205
    size = re.compile(rf"static_assert\(sizeof\({structure}\) == (\d+),")
    offset = re.compile(
        rf"static_assert\(offsetof\({structure}, (\w+)\) == (\d+),"
    )
    headers = importlib.resources.files(KERNEL_LIBRARY) / ABI_HEADERS
    for header in sorted(headers.iterdir(), key=lambda one: one.name):
        if not header.name.endswith(".h"):
            continue
        text = header.read_text()
        stated = size.search(text)
        if stated is not None:
            return AbiLayout(
                size=int(stated.group(1)),
                offsets={
                    field: int(value) for field, value in offset.findall(text)
                },
            )
    raise LookupError(f"no generated ABI header asserts {structure}'s size")


def assert_abi_layout(mirror: Any, structure: str) -> None:
    """Assert that the ctypes structure ``mirror`` has the size and every
    field offset the generated header asserts for ``structure``: a stale mirror
    would hand the module a call it reads at other offsets.
    """  # noqa: D205
    layout = abi_layout(structure)
    fields = [field for field, *_ in mirror._fields_]
    assert fields == list(layout.offsets), (
        f"the check's {mirror.__name__} has the fields {fields}, and the "
        f"generated header asserts {list(layout.offsets)} for {structure}"
    )
    assert ctypes.sizeof(mirror) == layout.size, (
        f"the check's {mirror.__name__} is {ctypes.sizeof(mirror)} bytes, and "
        f"the generated header asserts {layout.size} for {structure}"
    )
    for field, offset in layout.offsets.items():
        stated = getattr(mirror, field).offset
        assert stated == offset, (
            f"the check's {mirror.__name__}.{field} is at {stated}, and the "
            f"generated header asserts {offset} for {structure}"
        )


def abi_enumerator(name: str) -> int:
    """The number of the enumerator ``name`` of a generated ABI header of
    the installed kernel library, such as ``UNISERVE_DTYPE_BF16``.

    The headers' enumerations are the numbers a call structure's fields carry
    (an element type, ``ops::abi::code``; a phase, a layout, a normalization
    branch, a direction), each enumerator its enumeration's name and its
    variant in capitals, so a check passes the operator crate's number rather
    than a copy of it.

    Raises:
        LookupError: When no generated header states the enumerator.
    """  # noqa: D205
    enumerator = re.compile(
        rf"^\s*{re.escape(name)} = (-?\d+),$", flags=re.MULTILINE
    )
    headers = importlib.resources.files(KERNEL_LIBRARY) / ABI_HEADERS
    for header in sorted(headers.iterdir(), key=lambda one: one.name):
        if not header.name.endswith(".h"):
            continue
        stated = enumerator.search(header.read_text())
        if stated is not None:
            return int(stated.group(1))
    raise LookupError(f"no generated ABI header states the enumerator {name}")


def compile_time(value: Any) -> Any:
    """A compile-time value as a launch passes it.

    A tuple a launch table states (a derived stride pair, a slot table)
    crosses the command's JSON as an array, and Triton takes a compile-time
    sequence as a tuple only, so every array becomes a tuple again. A Triton
    type crosses it as ``{"type": <Triton's name>}`` and becomes the type.
    """
    if isinstance(value, list):
        return tuple(compile_time(item) for item in value)
    if isinstance(value, dict):
        # Imported here, after the session has fixed Triton's mode
        # (`conftest.py`), which an import at this module's load could precede.
        import triton.language as tl

        return tl.dtype(value["type"])
    return value


class RefusedLaunchError(LookupError):
    """A package states no launch for a descriptor key: its dimensions do not
    declare the key, or its launch table or derivation refuses it. The message
    is the package's reason.
    """  # noqa: D205


@dataclasses.dataclass(frozen=True)
class StatedLaunch:
    """The compilation a launch table states for one descriptor key.

    ``kernel`` is the ``@triton.jit`` function the table names, of this
    session's import of the package's source; ``signature`` holds every kernel
    parameter in order, a Triton type or ``"constexpr"``; ``constexprs`` the
    compile-time values a launch passes by name; ``options`` the warp count,
    the dependent launch and the further ``triton.compile`` options.
    """

    kernel: Any
    signature: dict[str, str]
    constexprs: dict[str, Any]
    options: dict[str, Any]

    @property
    def words(self) -> list[str]:
        """The parameters a launch passes in order: the non-constexpr ones."""
        return [
            name for name, kind in self.signature.items() if kind != "constexpr"
        ]


class Launches:
    """What Triton packages' launch tables state, from the Build Server.

    One ``uniserve build-server launch <package> --target <target>`` runs per
    package and target for the session and is asked key by key: the command
    reads each key with the Build Server's reader of the key grammar and has
    the driver a build compiles through state its compilation, so a check
    launches the compilation the key's module is built with. An answer is kept
    for the session.
    """

    def __init__(
        self, uniserve: Path, kernels: Path, logs: Path, source: Any
    ) -> None:
        """``uniserve`` is the executable, ``kernels`` the package tree,
        ``logs`` a directory for each command's standard error, and
        ``source(op)`` this session's import of a package's kernel module.
        """  # noqa: D205
        self._uniserve = uniserve
        self._kernels = kernels
        self._logs = logs
        self._source = source
        self._commands: dict[tuple[str, str], subprocess.Popen[str]] = {}
        self._errors: dict[tuple[str, str], IO[str]] = {}
        self._stated: dict[tuple[str, str, str], StatedLaunch] = {}

    def __call__(self, op: str, key: str, target: str) -> StatedLaunch:
        """The launch ``op``'s Triton package states for ``key`` on ``target``.

        Raises:
            RefusedLaunchError: When the package states none, with its reason.
            RuntimeError: When the command ends without answering.
        """
        stated = self._stated.get((op, key, target))
        if stated is not None:
            return stated

        command = self._command(op, target)
        command.stdin.write(key + "\n")
        command.stdin.flush()
        line = command.stdout.readline()
        if not line:
            log = self._errors[(op, target)].name
            raise RuntimeError(
                f"uniserve build-server launch of {op} for {target} ended "
                f"(exit {command.wait()}) before it answered {key}:\n"
                f"{Path(log).read_text(errors='replace')}"
            )
        answer = json.loads(line)
        if "refused" in answer:
            raise RefusedLaunchError(f"{op} {key}: {answer['refused']}")

        launch = answer["launch"]
        stated = StatedLaunch(
            kernel=getattr(self._source(op), launch["kernel"]),
            signature=launch["signature"],
            constexprs={
                name: compile_time(value)
                for name, value in launch["constexprs"].items()
            },
            options=launch["options"],
        )
        self._stated[(op, key, target)] = stated
        return stated

    def _command(self, op: str, target: str) -> subprocess.Popen[str]:
        """The running command of ``op``'s Triton package for ``target``.

        The command runs with the tests' declared environment
        (:func:`uniserve_testing.servers.environment`): its search path names
        the Build Server's toolchain program, and no setting of the session
        (Triton's mode among them) reaches the driver.
        """
        command = self._commands.get((op, target))
        if command is None:
            errors = (self._logs / f"launch-{op}-{target}.log").open("w")
            command = subprocess.Popen(
                [
                    str(self._uniserve),
                    "build-server",
                    "launch",
                    str(self._kernels / op / "triton"),
                    "--target",
                    target,
                ],
                env=environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=errors,
                text=True,
            )
            self._commands[(op, target)] = command
            self._errors[(op, target)] = errors
        return command

    def close(self) -> None:
        """End every command: each ends when its input does."""
        for key, command in self._commands.items():
            command.stdin.close()
            command.wait()
            command.stdout.close()
            self._errors[key].close()
        self._commands.clear()


class Modules:
    """Built modules of the kernel packages, resolved at a Build Server.

    A package is installed at the server the first time a module of it is
    asked for, and resolving a module names its files in the server's store,
    the server building the module first when its store holds none.
    """

    def __init__(self, server: BuildServer, kernels: Path, target: str) -> None:
        """``server`` is the session's Build Server, ``kernels`` the package
        tree and ``target`` what every module is built for.
        """  # noqa: D205
        self._server = server
        self._kernels = kernels
        self._target = target
        self._revisions: dict[tuple[str, str], int] = {}

    def __call__(self, op: str, backend: str, key: str) -> Path:
        """The directory holding the module of descriptor ``key`` of
        ``op``'s ``backend`` package.

        Raises:
            RuntimeError: When the server cannot install the package or
                resolve the module, with its reason.
        """  # noqa: D205
        revision = self._revisions.get((op, backend))
        if revision is None:
            installed = self._server.install(self._kernels / op / backend)
            revision = installed.revision
            self._revisions[(op, backend)] = revision
        resolved = self._server.resolve(
            op, backend, revision, key=key, target=self._target
        )
        return resolved.directory
