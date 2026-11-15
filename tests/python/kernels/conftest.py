"""The kernel checks: the fixtures and conventions every family shares.

A kernel check launches one package's kernel the way its operator launches it
and holds the result to a reference, inside the operator's own bound. The
checks of the packages under ``crates/ops/kernels`` live in
``tests/python/kernels/<family>/``, one module of plain pytest test functions
per check, ``test_<op>.py``; a check several operators share (one reference
over two layouts) is one module named for them, parametrized over the
operator.

**Layout of a module.** ``pytestmark = pytest.mark.unit``; the cases a check
covers are parameters (``pytest.mark.parametrize``, with readable ids): the
cases it states, and of each descriptor option the values it states it
covers, each marked for the mode it runs in. Each assertion carries the
measured error and the bound it was held to. A
bound, a tolerance and a reference are the operator's (``crates/ops/src``) and
are written once, beside the check that uses them. A launch of a kernel is
written once per module, in one function the tests call.

**Where the kernels run.** ``--kernel-device`` (registered in
``tests/python/conftest.py``) chooses once per session: ``interpreter``, the
default, runs every kernel on Triton's interpreter with host tensors and needs
no device; ``cuda`` compiles the kernels for the session's CUDA device and
launches them there. Triton fixes a function's mode when ``@triton.jit``
decorates it (``triton.knobs.runtime.interpret``), it decorates its own
language library (``tl.cdiv``, the reductions) when it is first imported, and
a kernel and every ``@triton.jit`` function it calls (``uniserve_kernels``,
``triton.language``) must share one mode; so the mode is one per process,
fixed by :func:`pytest_configure` before Triton is first imported, and no
check sets it again.

- A test that runs in both modes takes its tensors on ``torch_device``; one
  whose reference models the interpreter's stores (Triton's interpreter
  truncates a store into BF16 where a device rounds) reads ``interpreted``.
- A test, or a parameter (``pytest.param(..., marks=pytest.mark.gpu)``), that
  needs a device is marked ``gpu``: it is skipped with the reason in the
  interpreter mode, and ``-m "unit and not gpu"`` (``just
  test-python-fast``) deselects it. A test the interpreter cannot run (a
  libdevice call, a dependent launch) is such a test.
- In the ``cuda`` mode a host without a CUDA device skips every check, naming
  the reason.

**What a check reaches** (:mod:`tests.python.kernels.packages`), each through a
fixture of this module:

- ``kernels``: the package tree, ``crates/ops/kernels`` of pytest's root
  directory.
- ``kernel_source(op, backend="triton")``: the package's kernel module
  ``src/<op>.py``, imported from the package by its location and kept for the
  session, decorated in the session's mode.
- ``stated_launch(op, key, target)``: the compilation a Triton package's
  launch table states for descriptor ``key`` on ``target`` (the kernel of
  ``kernel_source(op)``, its signature, its compile-time values and its
  options), from ``uniserve build-server launch``; a key the package does not
  state raises :class:`~tests.python.kernels.packages.RefusedLaunchError`
  with its reason. The ``uniserve`` executable is ``UNISERVE_BINARY``, or
  ``uniserve`` on the search path; it runs the Build Server's Python
  environment's program ``uniserve-build-server``, which must be on the
  search path.
- ``kernel_module(op, backend, key)``: the directory of the built module of a
  descriptor key, resolved at a Build Server the session starts for the
  session's device (``uniserve build-server resolve``, which builds a missing
  module), for a check of a native or cubin module. It needs a device.

A check of a dependent launch runs its operator behind a producer that writes
the inputs late (:mod:`tests.python.kernels.chain`); a family's shared
reference arithmetic is a helper module beside its checks.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest
import torch

from tests.python.kernels.packages import (
    Launches,
    Modules,
    source_module,
)

#: The kernel package tree, from pytest's root directory, the repository's.
PACKAGES = ("crates", "ops", "kernels")


def interpreting(config: pytest.Config) -> bool:
    """Whether the session runs the kernels on Triton's interpreter."""
    return config.getoption("kernel_device") == "interpreter"


#: Triton's setting of its mode, which ``triton.knobs.runtime.interpret``
#: reads when Triton is imported and its language library is decorated.
TRITON_MODE = "TRITON_INTERPRET"


def pytest_configure(config: pytest.Config) -> None:
    """Fix Triton's mode for the session before Triton is first imported.

    No check and no kernel source imports Triton before this hook runs, which
    pytest calls once this file is loaded and before it imports a test module
    of this directory. The choice is Triton's own setting, the only one it
    reads before it decorates its language library; the processes a check
    starts inherit it with the rest of the session's environment.

    Raises:
        pytest.UsageError: When Triton was imported earlier in another mode,
            which decorated its language library in that mode.
    """
    if "triton" in sys.modules:
        triton = sys.modules["triton"]
        if triton.knobs.runtime.interpret != interpreting(config):
            raise pytest.UsageError(
                "Triton was imported in the other mode before the kernel "
                "checks chose theirs (--kernel-device)"
            )
        return
    os.environ[TRITON_MODE] = "1" if interpreting(config) else "0"


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip a check that needs a device when the session interprets."""
    if interpreting(item.config) and item.get_closest_marker("gpu"):
        pytest.skip("needs a CUDA device: run with --kernel-device cuda")


@pytest.fixture(scope="session")
def kernels(pytestconfig: pytest.Config) -> Path:
    """The kernel package tree."""
    return pytestconfig.rootpath.joinpath(*PACKAGES)


@pytest.fixture(scope="session")
def interpreted(pytestconfig: pytest.Config) -> bool:
    """Whether the kernels run on Triton's interpreter."""
    return interpreting(pytestconfig)


@pytest.fixture(scope="session")
def torch_device(interpreted: bool) -> torch.device:
    """Where a check's tensors live: the host for the interpreter, else the
    session's CUDA device.
    """  # noqa: D205
    if interpreted:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        pytest.skip("--kernel-device cuda: this host shows no CUDA device")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def kernel_source(kernels: Path) -> Callable[..., ModuleType]:
    """``kernel_source(op, backend="triton")``: a package's kernel module."""
    imported: dict[tuple[str, str], ModuleType] = {}

    def source(op: str, backend: str = "triton") -> ModuleType:
        module = imported.get((op, backend))
        if module is None:
            module = source_module(kernels / op / backend, op)
            imported[(op, backend)] = module
        return module

    return source


@pytest.fixture(scope="session")
def uniserve() -> str:
    """The ``uniserve`` executable: ``UNISERVE_BINARY``, or ``uniserve`` on the
    search path.
    """  # noqa: D205
    return os.environ.get("UNISERVE_BINARY", "uniserve")


@pytest.fixture(scope="session")
def stated_launch(
    uniserve: str,
    kernels: Path,
    kernel_source: Callable[..., ModuleType],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Launches]:
    """``stated_launch(op, key, target)``: what a launch table states."""
    launches = Launches(
        uniserve, kernels, tmp_path_factory.mktemp("launch"), kernel_source
    )
    yield launches
    launches.close()


@pytest.fixture(scope="session")
def module_target(torch_device: torch.device) -> str:
    """The target the session's device runs: the architecture-specific one of
    its compute capability, as the Build Server names it.
    """  # noqa: D205
    major, minor = torch.cuda.get_device_capability(torch_device)
    return f"sm_{major}{minor}a"


@pytest.fixture(scope="session")
def build_server(
    uniserve: str,
    module_target: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Path]:
    """A Build Server for the session's modules, over an empty store; its
    socket.
    """  # noqa: D205
    # The servers' helpers are the end-to-end tests', whose HTTP stack only a
    # session that builds modules needs.
    from tests.python.e2e.build_server import (
        BUILD_SERVER_START_S,
        require_toolchain_environment,
    )
    from tests.python.e2e.http_helpers import start_server, stop_server

    require_toolchain_environment()
    work = tmp_path_factory.mktemp("build-server")
    # A Unix socket's path is bounded (`SUN_LEN`, 108 bytes), so the socket
    # lives in a short directory of its own.
    sockets = Path(tempfile.mkdtemp(prefix="uniserve-kernels-"))
    socket = sockets / "build-server.sock"
    server = start_server(
        [
            uniserve,
            "build-server",
            "serve",
            "--socket",
            str(socket),
            "--package-root",
            str(work / "packages"),
            "--store-root",
            str(work / "modules"),
            "--target",
            module_target,
        ],
        work / "build-server.log",
    )
    try:
        server.wait("serving", BUILD_SERVER_START_S)
        yield socket
    finally:
        stop_server(server.process)
        server.close()
        shutil.rmtree(sockets, ignore_errors=True)


@pytest.fixture(scope="session")
def kernel_module(
    uniserve: str, build_server: Path, kernels: Path, module_target: str
) -> Modules:
    """``kernel_module(op, backend, key)``: a built module's directory."""
    return Modules(uniserve, build_server, kernels, module_target)
