"""GPU leases and the Program Server of the end-to-end tests.

The GPU recipe runs the end-to-end cases in parallel (``pytest -n`` with
``--dist loadgroup``). Each case serves on the GPUs it leases for its
duration, so independent cases share the host while no two of them share a
GPU:

- a case leases one GPU unless it is marked ``gpus(count)``; ``gpus(0)``
  leases none;
- a module whose own ``pytestmark`` carries ``gpus(count)`` leases them once
  for the whole module, because its server is module-scoped, and its cases
  run on one worker (one ``xdist_group``); every other case is scheduled on
  its own;
- a case marked ``host_wide`` observes state of the whole host, such as its
  shared memory, and runs alone, holding every GPU.

A lease exports the devices it holds in the variables the tests read:
``UNISERVE_WORLD_DEVICE``, ``UNISERVE_LIVE_DEVICE`` and
``UNISERVE_EPOCH_DEVICE`` (the first), ``UNISERVE_SECOND_DEVICE`` (the
second), and ``UNISERVE_WORLD_DEVICES`` and ``UNISERVE_EPOCH_DEVICES`` (all
of them, comma-separated). The GPUs come from ``UNISERVE_E2E_DEVICES`` (a
comma-separated list of device ordinals), else every GPU ``nvidia-smi``
lists. A case that needs more GPUs than that is skipped, naming both counts.

Leases are advisory ``flock`` locks on one file per GPU, plus a host lock
every case holds shared and a host-wide case holds exclusively. They live in
a directory of the system's temporary directory, so concurrent runs on one
host respect one another.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import IO

import pytest

from tests.python.e2e.http_helpers import programs_from
from tests.python.e2e.program_server import (
    program_server as running_program_server,
)

#: How long a case waits between attempts to lease its GPUs, in seconds.
LEASE_RETRY_S = 2.0


def _devices() -> list[str]:
    """The GPU ordinals the run may lease."""
    stated = os.environ.get("UNISERVE_E2E_DEVICES", "")
    if stated:
        return [device.strip() for device in stated.split(",") if device]
    listed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line.strip() for line in listed.splitlines() if line.strip()]


def _lock_directory() -> Path:
    directory = Path(tempfile.gettempdir()) / f"uniserve-e2e-gpus-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    return directory


def _try_lock(path: Path) -> IO[str] | None:
    """`path` locked exclusively without waiting, or `None` when it is held."""
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


@contextmanager
def lease(count: int, host_wide: bool) -> Iterator[list[str]]:
    """Hold `count` GPUs (every GPU when `host_wide`) and export them.

    The host lock is taken first, shared or exclusive, and the GPU locks are
    taken all at once or not at all: a case that finds too few free releases
    what it took and retries, so two cases never hold part of each other's
    GPUs while they wait.
    """
    devices = _devices()
    wanted = len(devices) if host_wide else count
    if wanted > len(devices):
        pytest.skip(
            f"the case needs {wanted} GPUs and the run may lease "
            f"{len(devices)} (UNISERVE_E2E_DEVICES or nvidia-smi)"
        )
    directory = _lock_directory()
    with ExitStack() as stack:
        host = stack.enter_context((directory / "host.lock").open("a+"))
        fcntl.flock(host, fcntl.LOCK_EX if host_wide else fcntl.LOCK_SH)
        held: list[tuple[str, IO[str]]] = []
        while len(held) < wanted:
            for device in devices:
                if len(held) == wanted:
                    break
                handle = _try_lock(directory / f"gpu{device}.lock")
                if handle is not None:
                    held.append((device, handle))
            if len(held) < wanted:
                for _, handle in held:
                    handle.close()
                held = []
                time.sleep(LEASE_RETRY_S)
        for _, handle in held:
            stack.callback(handle.close)
        granted = [device for device, _ in held]
        with pytest.MonkeyPatch.context() as patch:
            if granted:
                every = ",".join(granted)
                for name in (
                    "UNISERVE_WORLD_DEVICE",
                    "UNISERVE_LIVE_DEVICE",
                    "UNISERVE_EPOCH_DEVICE",
                ):
                    patch.setenv(name, granted[0])
                patch.setenv("UNISERVE_WORLD_DEVICES", every)
                patch.setenv("UNISERVE_EPOCH_DEVICES", every)
                if len(granted) > 1:
                    patch.setenv("UNISERVE_SECOND_DEVICE", granted[1])
            yield granted


def _module_gpus(module: object) -> int | None:
    """The GPU count a module's own ``pytestmark`` leases for the module."""
    marks = getattr(module, "pytestmark", [])
    if not isinstance(marks, list):
        marks = [marks]
    for mark in marks:
        if mark.name == "gpus":
            return int(mark.args[0])
    return None


# Before pytest-xdist reads the groups from the marks, in its own such hook.
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Keep the cases of a module that leases for itself on one worker."""
    for item in items:
        module = getattr(item, "module", None)
        if module is not None and _module_gpus(module) is not None:
            item.add_marker(pytest.mark.xdist_group(module.__name__))


def _is_gpu_e2e(node: pytest.Item | pytest.Module) -> bool:
    keywords = node.keywords
    return "e2e" in keywords and "gpu" in keywords


@pytest.fixture(scope="module", autouse=True)
def module_gpus(request: pytest.FixtureRequest) -> Iterator[None]:
    """The module's lease, for a module that states ``gpus`` itself."""
    count = _module_gpus(request.module)
    if count is None or not _is_gpu_e2e(request.node):
        yield
        return
    with lease(count, host_wide=False):
        yield


@pytest.fixture(autouse=True)
def case_gpus(request: pytest.FixtureRequest) -> Iterator[None]:
    """The case's lease, unless its module holds one."""
    if not _is_gpu_e2e(request.node) or (
        _module_gpus(request.module) is not None
    ):
        yield
        return
    mark = request.node.get_closest_marker("gpus")
    count = int(mark.args[0]) if mark else 1
    host_wide = request.node.get_closest_marker("host_wide") is not None
    with lease(count, host_wide):
        yield


@pytest.fixture(scope="module", autouse=True)
def program_server(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Path | None]:
    """The module's Program Server, which every deployment it starts names.

    A deployment's epoch is resolved at this host's Program Server, and the
    one a daemon starts by itself holds no package, so the module starts one
    with every package of the tree installed, and `start_server` gives each
    deployment that names no other ``program_config.socket`` this one. A
    module that installs packages of its own overrides this fixture, and a
    module that leases no GPU starts none.
    """
    # A module that leases no GPU starts no deployment.
    if not _is_gpu_e2e(request.node) or _module_gpus(request.module) == 0:
        yield None
        return
    with (
        running_program_server(tmp_path_factory) as socket,
        programs_from(socket),
    ):
        yield socket
