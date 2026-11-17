"""The GPU leases, the Build Server and the deployments of the end-to-end
cases.

The GPU recipe runs the end-to-end cases in parallel (``pytest -n`` with
``--dist loadgroup``). Each case serves on the GPUs it leases for its
duration (:mod:`uniserve_testing.devices`), so independent cases share the
host while no two of them share a GPU:

- a case leases one GPU unless it is marked ``gpus(count)``; ``gpus(0)``
  leases none;
- a module whose own ``pytestmark`` carries ``gpus(count)`` leases them once
  for the whole module, because its server is module-scoped, and its cases
  run on one worker (one ``xdist_group``); every other case is scheduled on
  its own;
- a case marked ``host_wide`` observes state of the whole host, such as its
  shared memory, and runs alone, holding every GPU.

A case that needs more GPUs than the run may lease (``--e2e-devices``, else
every GPU of the host) is skipped, naming both counts. The lease is the
``lease`` fixture's value, and the deployments a case starts
(``deployments``, :class:`uniserve_testing.deployments.Deployments`) serve on
it at the run's Build Server, which holds the tree's packages.
"""  # noqa: D205

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from uniserve_testing import builds
from uniserve_testing.builds import BuildServer
from uniserve_testing.deployments import Deployments, Settings
from uniserve_testing.devices import Lease, lease_gpus


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


@contextmanager
def _leased(
    devices: tuple[int, ...], count: int, host_wide: bool
) -> Iterator[Lease]:
    """`count` of `devices` leased (all when `host_wide`), or the case
    skipped when the run may lease fewer.
    """  # noqa: D205
    wanted = len(devices) if host_wide else count
    if wanted > len(devices):
        pytest.skip(
            f"the case needs {wanted} GPUs and the run may lease {len(devices)}"
        )
    with lease_gpus(devices, wanted, host_wide=host_wide) as granted:
        yield granted


@pytest.fixture(scope="module")
def module_lease(
    request: pytest.FixtureRequest, run_devices: tuple[int, ...]
) -> Iterator[Lease | None]:
    """The module's lease, for a module that states ``gpus`` itself."""
    count = _module_gpus(request.module)
    if count is None or not _is_gpu_e2e(request.node):
        yield None
        return
    with _leased(run_devices, count, host_wide=False) as granted:
        yield granted


@pytest.fixture(autouse=True)
def lease(
    request: pytest.FixtureRequest,
    run_devices: tuple[int, ...],
    module_lease: Lease | None,
) -> Iterator[Lease | None]:
    """The GPUs the case serves on: its module's lease, or its own."""
    if not _is_gpu_e2e(request.node) or module_lease is not None:
        yield module_lease
        return
    mark = request.node.get_closest_marker("gpus")
    count = int(mark.args[0]) if mark else 1
    host_wide = request.node.get_closest_marker("host_wide") is not None
    with _leased(run_devices, count, host_wide) as granted:
        yield granted


@pytest.fixture(scope="session")
def build_server(
    uniserve: Path,
    repository: Path,
    e2e_settings: Settings,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[BuildServer]:
    """The run's Build Server, with every package of the tree installed
    (:func:`uniserve_testing.builds.build_server`).

    A deployment's epoch is resolved at its host's Build Server, which the
    run starts itself so that its package root and store are the run's own.
    A module whose cases install
    revisions of their own overrides this fixture with a server of its own,
    since a revision installed here is what every later deployment of the
    session would select.
    """  # noqa: D205
    with builds.build_server(
        uniserve,
        repository,
        tmp_path_factory.mktemp("build-server"),
        scale=e2e_settings.time_scale,
    ) as server:
        yield server


@pytest.fixture
def deployments(
    uniserve: Path,
    build_server: BuildServer,
    e2e_settings: Settings,
    lease: Lease | None,
    tmp_path: Path,
) -> Deployments:
    """The case's deployments, on its lease at the run's Build Server."""
    return Deployments(uniserve, build_server, e2e_settings, lease, tmp_path)


@pytest.fixture(scope="module")
def module_deployments(
    uniserve: Path,
    build_server: BuildServer,
    e2e_settings: Settings,
    module_lease: Lease | None,
    tmp_path_factory: pytest.TempPathFactory,
) -> Deployments:
    """The deployments of a module that serves one deployment for all its
    cases, on the module's lease.
    """  # noqa: D205
    return Deployments(
        uniserve,
        build_server,
        e2e_settings,
        module_lease,
        tmp_path_factory.mktemp("deployments"),
    )
