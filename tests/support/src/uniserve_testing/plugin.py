"""The pytest plugin of UniServe's Python tests.

pytest loads this module in every session of an environment that installs
``uniserve-testing`` (its ``pytest11`` entry point). It declares the options
the test directories read and the session fixtures every directory shares; a
directory's own fixtures stay in its ``conftest.py``. Which cases a run
selects is pytest's marker expression alone (``pyproject.toml``'s
``addopts`` deselects the GPU end-to-end cases unless a run names its own).

Options:

- ``--kernel-device {interpreter,cuda}``: where the kernel checks
  (``tests/python/kernels``) run.
- ``--e2e-devices``: the GPUs an end-to-end run may lease, as device indices;
  every GPU the driver reports by default (:mod:`uniserve_testing.devices`).
- ``--engine``: the engine package every test deployment runs, by its backend
  name; the tree's own engine package by default.
- ``--serve-set``: a ``PATH=VALUE`` assignment every test deployment adds
  (repeatable), such as a run under compute-sanitizer states.
- ``--time-scale``: the factor every wait of an end-to-end case is stretched
  by, for a run whose processes run under a tool.
- ``--peer-host``, ``--peer-address`` and ``--leader-address``: the second
  host of a two-host case (an ssh destination that has ``uniserve``
  installed), the address this host reaches it at, and the address it reaches
  this host at.

With pytest-xdist, ``-n auto`` starts one worker per GPU the run may lease
(:func:`pytest_xdist_auto_num_workers`).
"""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from pathlib import Path

import pytest

from uniserve_testing.deployments import Settings
from uniserve_testing.devices import host_devices
from uniserve_testing.remote import RemoteHost


def pytest_addoption(parser: pytest.Parser) -> None:
    """Declare the options of UniServe's tests (module documentation)."""
    kernels = parser.getgroup("uniserve kernels")
    kernels.addoption(
        "--kernel-device",
        choices=("interpreter", "cuda"),
        default="interpreter",
        help="run the kernel checks on Triton's interpreter, which needs no "
        "device (the default), or compiled for and launched on a CUDA device",
    )

    e2e = parser.getgroup("uniserve end-to-end")
    e2e.addoption(
        "--e2e-devices",
        type=_device_indices,
        default=None,
        help="the GPUs the end-to-end cases may lease, as comma-separated "
        "device indices; every GPU the driver reports when unset",
    )
    e2e.addoption(
        "--engine",
        default=None,
        help="the engine package every test deployment runs, by its backend "
        "name; the tree's own engine package when unset",
    )
    e2e.addoption(
        "--serve-set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="an option every test deployment adds, as `serve --set` takes "
        "it; repeatable",
    )
    e2e.addoption(
        "--time-scale",
        type=float,
        default=1.0,
        help="the factor every wait of an end-to-end case is stretched by",
    )
    e2e.addoption(
        "--peer-host",
        default=None,
        help="the second host of a two-host case: an ssh destination whose "
        "login has `uniserve` installed",
    )
    e2e.addoption(
        "--peer-address",
        default=None,
        help="the address this host reaches the second host at",
    )
    e2e.addoption(
        "--leader-address",
        default=None,
        help="the address the second host reaches this host at",
    )
    e2e.addoption(
        "--mooncake-master",
        default=None,
        metavar="HOST:PORT",
        help="a running Mooncake master, the remote store of the cases that "
        "keep host blocks in one",
    )


def _device_indices(text: str) -> tuple[int, ...]:
    """The device indices of ``--e2e-devices``."""
    return tuple(int(index) for index in text.split(",") if index.strip())


def interpreting(config: pytest.Config) -> bool:
    """Whether the session runs the kernels on Triton's interpreter."""
    return config.getoption("kernel_device") == "interpreter"


#: Triton's setting of its mode, which ``triton.knobs.runtime.interpret``
#: reads when Triton is imported and its language library is decorated.
TRITON_MODE = "TRITON_INTERPRET"


def pytest_configure(config: pytest.Config) -> None:
    """Fix Triton's mode for the session (``--kernel-device``) before Triton
    is first imported.

    Triton fixes a function's mode when ``@triton.jit`` decorates it, and
    decorates its own language library when it is first imported, so the
    mode is one per process and must be chosen before any test module of
    the session, whichever directory it lies in, imports Triton (a test
    module that imports torch or transformers may). This hook runs when
    pytest loads the plugin, before collection. The choice is Triton's own
    setting, the only one it reads before it decorates its language
    library; a process a test starts runs with the declared environment of
    :func:`uniserve_testing.servers.environment`, which does not carry it.

    Raises:
        pytest.UsageError: When Triton was imported earlier in another mode.
    """  # noqa: D205
    triton = sys.modules.get("triton")
    if triton is not None:
        if triton.knobs.runtime.interpret != interpreting(config):
            raise pytest.UsageError(
                "Triton was imported in the other mode before the session "
                "chose its own (--kernel-device)"
            )
        return
    os.environ[TRITON_MODE] = "1" if interpreting(config) else "0"


#: The program the root project installs into its environment's scripts
#: directory: every UniServe process's executable.
EXECUTABLE = "uniserve"


def environment_program(name: str) -> Path:
    """The console program ``name`` of the environment that runs pytest.

    The repository's environment installs the ``uniserve`` executable, and
    ``uniserve-manager`` beside it, into its scripts directory as the root
    project's binaries; the tests run what that environment installed, never
    a build found elsewhere.

    Raises:
        pytest.UsageError: When the environment has no such program.
    """
    scripts = sysconfig.get_path("scripts")
    found = shutil.which(name, path=scripts)
    if found is None:
        raise pytest.UsageError(
            f"this environment has no `{name}` in {scripts}: install the "
            "root project into it (`uv sync --locked`)"
        )
    return Path(found)


@pytest.fixture(scope="session")
def uniserve() -> Path:
    """The ``uniserve`` executable of the environment."""
    return environment_program(EXECUTABLE)


@pytest.fixture(scope="session")
def repository(pytestconfig: pytest.Config) -> Path:
    """The repository's root: pytest's root directory, which the root
    project's ``pyproject.toml`` declares.
    """  # noqa: D205
    return pytestconfig.rootpath


@pytest.fixture(scope="session")
def e2e_settings(pytestconfig: pytest.Config) -> Settings:
    """What the run's options state for every deployment of the run."""
    return Settings(
        time_scale=pytestconfig.getoption("time_scale"),
        serve_set=tuple(pytestconfig.getoption("serve_set")),
        engine=pytestconfig.getoption("engine"),
    )


def _run_devices(config: pytest.Config) -> tuple[int, ...]:
    """The GPUs the run's cases may lease: ``--e2e-devices``, else every GPU
    NVML reports on this host.
    """  # noqa: D205
    stated = config.getoption("e2e_devices")
    return stated if stated is not None else host_devices()


@pytest.fixture(scope="session")
def run_devices(pytestconfig: pytest.Config) -> tuple[int, ...]:
    """The GPUs the run's cases may lease (:func:`_run_devices`)."""
    return _run_devices(pytestconfig)


# pytest-xdist calls this hook for `-n auto` and takes the first answer, so
# this one runs before xdist's own (the host's CPU count); it is optional so
# that a session of an environment without pytest-xdist still loads the
# plugin.
@pytest.hookimpl(optionalhook=True, tryfirst=True)
def pytest_xdist_auto_num_workers(config: pytest.Config) -> int:
    """The worker count of ``pytest -n auto``: one per GPU the run may lease.

    Each end-to-end case serves on the GPUs it leases and blocks until they
    are free (:mod:`uniserve_testing.devices`), so one worker per leasable
    GPU lets the one-GPU cases fill every GPU, and a further worker could
    only wait for a lease while it holds a session of its own (a Build
    Server and the session's imports).
    """
    return len(_run_devices(config))


@pytest.fixture(scope="session")
def peer_host(pytestconfig: pytest.Config) -> RemoteHost:
    """The second host of a two-host case; the case is skipped when the run
    names none (``--peer-host``, ``--peer-address``, ``--leader-address``).
    """  # noqa: D205
    destination = pytestconfig.getoption("peer_host")
    address = pytestconfig.getoption("peer_address")
    leader = pytestconfig.getoption("leader_address")
    if not (destination and address and leader):
        pytest.skip(
            "a two-host case needs --peer-host, --peer-address and "
            "--leader-address"
        )
    return RemoteHost(destination, address, leader)


@pytest.fixture(scope="session")
def mooncake_master(pytestconfig: pytest.Config) -> str:
    """The Mooncake master a remote store case keeps its objects at
    (``kv_cache_config.mooncake.master``, ``host:port``): a service of the
    host the run names with ``--mooncake-master``; the case is skipped
    without one. The Storage Server's connector opens Mooncake Store's C API
    from the loader's default search path, where the Mooncake package
    installs it.
    """  # noqa: D205
    master = pytestconfig.getoption("mooncake_master")
    if master is None:
        pytest.skip("a remote store case needs --mooncake-master")
    return master
