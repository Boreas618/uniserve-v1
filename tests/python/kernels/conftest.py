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
measured error and the bound it was held to. A bound, a tolerance and a
reference are the operator's (``crates/ops/src``) and are written once,
beside the check that uses them. A launch of a kernel is written once per
module, in one function the tests call.

**Where the kernels run.** ``--kernel-device`` (an option of the tests'
plugin, :mod:`uniserve_testing.plugin`) chooses once per session:
``interpreter``, the default, runs every kernel on Triton's interpreter with
host tensors and needs no device; ``cuda`` compiles the kernels for the
session's CUDA device and launches them there. Triton fixes a function's mode
when ``@triton.jit`` decorates it (``triton.knobs.runtime.interpret``), it
decorates its own language library (``tl.cdiv``, the reductions) when it is
first imported, and a kernel and every ``@triton.jit`` function it calls
(``uniserve_kernels``, ``triton.language``) must share one mode; so the mode
is one per process, fixed by :func:`pytest_configure` before Triton is first
imported, and no check sets it again.

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

**What a check reaches** (:mod:`uniserve_testing.kernels.packages`), each
through a fixture of this module or of the plugin:

- ``kernels``: the package tree, ``crates/ops/kernels`` of the repository
  (the plugin's ``repository``).
- ``kernel_source(op, backend="triton")``: the package's kernel module
  ``src/<op>.py``, imported from the package by its location and kept for the
  session, decorated in the session's mode.
- ``stated_launch(op, key, target)``: the compilation a Triton package's
  launch table states for descriptor ``key`` on ``target`` (the kernel of
  ``kernel_source(op)``, its signature, its compile-time values and its
  options), from ``uniserve build-server launch`` run by the environment's
  ``uniserve`` (the plugin's ``uniserve``); a key the package does not state
  raises :class:`~uniserve_testing.kernels.packages.RefusedLaunchError` with
  its reason. The command runs the Build Server's toolchain program
  ``uniserve-build-server``, which must be on the search path.
- ``launch_target``: the target a check's stated launch names, the session
  device's (``module_target``) or, on the interpreter,
  :data:`INTERPRETED_TARGET`.
- ``kernel_module(op, backend, key)``: the directory of the built module of a
  descriptor key, resolved at the session's Build Server
  (:mod:`uniserve_testing.builds`, which builds a missing module for this
  host's devices), for a check of a native or cubin module. It needs a
  device.

A check of a dependent launch runs its operator behind a producer that writes
the inputs late (:mod:`uniserve_testing.kernels.chain`); a family's shared
reference arithmetic and operands are a module of the family's subpackage of
:mod:`uniserve_testing.kernels`.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest
import torch
from uniserve_testing import builds
from uniserve_testing.kernels.packages import (
    Launches,
    Modules,
    source_module,
)


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
    reads before it decorates its language library. A process a check starts
    runs with the declared environment of the tests' servers
    (:func:`uniserve_testing.servers.environment`), which does not carry it:
    the Build Server's driver decorates a package's kernels in Triton's
    compiled mode itself.

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
def kernels(repository: Path) -> Path:
    """The kernel package tree."""
    return repository / builds.KERNEL_TREE


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
def stated_launch(
    uniserve: Path,
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
    # `uniserve build-server resolve` and `launch` take a kernel module's
    # target by its name, and no command names the target of a device, so the
    # name is the Build Server's rule for a device of this capability
    # (`Target::arch`).
    major, minor = torch.cuda.get_device_capability(torch_device)
    return f"sm_{major}{minor}a"


#: The target a launch is stated for on the interpreter. The interpreter
#: compiles for no target, and ``uniserve build-server launch`` states a
#: compilation for a GPU target of the Build Server's description. The
#: launches the interpreted checks state read no fact of a target: their keys
#: ask for no dependent launch and no derivation of their packages reads a
#: target's facts, so every target states the same kernel, compile-time values
#: and options, and this one, the description's first, states them.
INTERPRETED_TARGET = "sm_90a"


@pytest.fixture(scope="session")
def launch_target(interpreted: bool, request: pytest.FixtureRequest) -> str:
    """The target a check's launch is stated for: the session device's, or on
    the interpreter :data:`INTERPRETED_TARGET`.
    """  # noqa: D205
    if interpreted:
        return INTERPRETED_TARGET
    return request.getfixturevalue("module_target")


@pytest.fixture(scope="session")
def build_server(
    uniserve: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[builds.BuildServer]:
    """The session's Build Server, over an empty package root and store."""
    with builds.build_server(
        uniserve, tmp_path_factory.mktemp("build-server")
    ) as server:
        yield server


@pytest.fixture(scope="session")
def kernel_module(
    build_server: builds.BuildServer, kernels: Path, module_target: str
) -> Modules:
    """``kernel_module(op, backend, key)``: a built module's directory."""
    return Modules(build_server, kernels, module_target)
