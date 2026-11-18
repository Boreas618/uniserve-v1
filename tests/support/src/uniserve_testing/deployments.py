"""The deployments an end-to-end case starts.

A case asks the ``deployments`` fixture (``tests/python/e2e/conftest.py``)
for a :class:`Deployments`, which writes each deployment's policy package
(:mod:`uniserve_testing.packages`) into the case's directory, starts it with
``uniserve serve <package>`` and stops it with the case:

- on the GPUs the case leases, which the daemon sees as ``cuda:0`` onwards
  (CUDA's device visibility, stated in its environment);
- at the run's Build Server (``build_config``), which holds the tree's
  packages;
- under a runtime directory of its own (``XDG_RUNTIME_DIR``, which the
  caller of a UniServe server creates and names), where its daemon starts a
  Storage Server that serves this deployment alone and that the case stops
  when the deployment ends;
- with the persistent state of its deployments (``XDG_STATE_HOME``, which
  holds each deployment's revisions under the package directory's name) in
  the case's directory, so that no case starts from a revision another case
  left;
- with the run's engine package (``--engine``) and options
  (``--serve-config``), and the waits stretched by its time scale.

The daemon binds port 0 and reports what it bound in its readiness record
(:class:`uniserve_testing.servers.Server`).
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from uniserve_testing import builds, packages
from uniserve_testing.builds import BuildServer
from uniserve_testing.devices import Lease
from uniserve_testing.packages import Config, merged
from uniserve_testing.remote import RemoteHost
from uniserve_testing.servers import (
    Server,
    environment,
    launch,
    socket_path,
)

#: The host tier a deployment holds unless its case sizes it, in GiB: every
#: case runs its own Storage Server, which budgets the host's available
#: storage as if it were the host's only one, so concurrent cases sized by
#: the default ratio would together commit more host storage than the host
#: has.
HOST_CACHE_GIB = 16

#: How long a deployment's second host waits for its leader to end before
#: the case fails it, in seconds: it ends once the leader's connection
#: closes.
FOLLOWER_EXIT_S = 120.0

#: How long the second host's transfer calibration may take, in seconds: it
#: times a few copies of each of its paths (``storage-server calibrate``).
CALIBRATION_S = 600.0


@dataclasses.dataclass(frozen=True)
class Follower:
    """A deployment's second-host daemon: its runtime directory there, where
    its Storage Server's socket lies (``storage-server`` commands run there
    with ``XDG_RUNTIME_DIR`` set to it).
    """  # noqa: D205

    runtime: str


@dataclasses.dataclass(frozen=True)
class Settings:
    """What the run's options state for every deployment of the run: the
    wait scale (``--time-scale``), the options every package states over its
    own (``--serve-config``) and the engine package (``--engine``).
    """  # noqa: D205

    time_scale: float
    config: Mapping[str, Any]
    engine: str | None


def this_host() -> str:
    """This host's name, ``uname -n``: how a node finds itself in a
    package's ``hosts``.
    """  # noqa: D205
    return os.uname().nodename


def free_port() -> int:
    """A TCP port no listener of this host holds now, which the kernel picks."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("", 0))
        return int(probe.getsockname()[1])


def two_hosts(host: RemoteHost) -> dict[str, Any]:
    """The options of a deployment over this host, its leader, and `host`:
    both by their host names (``hosts``), and a join port of its own
    (``admin_config.join``), so that concurrent cases' leaders bind no port
    one another holds.
    """  # noqa: D205
    return {
        "hosts": [this_host(), host.name],
        "admin_config": {"join": free_port()},
    }


@contextmanager
def runtime_directory(
    uniserve: Path, kept: Path | None = None
) -> Iterator[Path]:
    """A runtime directory (``XDG_RUNTIME_DIR``) of the caller's own, whose
    Storage Server is stopped when the block ends.

    A daemon started under it starts its host's Storage Server there, which
    outlives the daemon with the deployment's host and device storage; the
    block's end stops it, keeps the logs it wrote beside its socket in the
    directory `kept` when one is named, and removes the runtime directory. A
    Unix socket's path holds at most 107 bytes, so the directory lies in the
    system's temporary directory.
    """  # noqa: D205
    runtime = Path(tempfile.mkdtemp(prefix="uniserve-"))
    try:
        yield runtime
    finally:
        stop_storage_server(uniserve, runtime, kept)
        shutil.rmtree(runtime, ignore_errors=True)


def storage_status(uniserve: Path, runtime: Path) -> dict | None:
    """The readiness record of the Storage Server serving under the runtime
    directory `runtime` (``storage-server status``), or ``None`` when none
    serves there.
    """  # noqa: D205
    answered = subprocess.run(
        [str(uniserve), "storage-server", "status"],
        env=environment(XDG_RUNTIME_DIR=str(runtime)),
        capture_output=True,
        text=True,
        check=False,
    )
    if answered.returncode != 0:
        return None
    return json.loads(answered.stdout)


def stop_storage_server(
    uniserve: Path, runtime: Path, kept: Path | None = None
) -> None:
    """Stop the Storage Server serving under `runtime`, if one serves, with
    its ``stop`` request, which returns once it ended; keep the logs beside
    its socket in `kept` when one is named.
    """  # noqa: D205
    record = storage_status(uniserve, runtime)
    if record is None:
        return
    subprocess.run(
        [str(uniserve), "storage-server", "stop"],
        env=environment(XDG_RUNTIME_DIR=str(runtime)),
        capture_output=True,
        text=True,
        check=True,
    )
    if kept is not None:
        kept.mkdir(parents=True, exist_ok=True)
        directory = socket_path(record["endpoints"]["storage"]).parent
        for log in directory.glob("*.log"):
            shutil.copy(log, kept / log.name)


class Deployments:
    """Starts a case's deployments (module documentation)."""

    def __init__(
        self,
        uniserve: Path,
        build_server: BuildServer | None,
        settings: Settings,
        lease: Lease | None,
        directory: Path,
    ) -> None:
        """Deployments of the executable `uniserve` at `build_server`, on
        the GPUs of `lease`, with their packages, state and logs under
        `directory`.
        """  # noqa: D205
        self.uniserve = uniserve
        self.build_server = build_server
        self.settings = settings
        self.lease = lease
        self.directory = directory

    def run_config(self) -> dict[str, Any]:
        """The options every deployment of the run states over its own: its
        engine package (``--engine``) and the run's ``--serve-config``.
        """  # noqa: D205
        engine: Config = (
            {"worker_config": {"engine": self.settings.engine}}
            if self.settings.engine is not None
            else {}
        )
        return merged(engine, self.settings.config)

    def config(
        self, config: Config, build_settings: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """The ``config.toml`` of a deployment of the run whose own options
        are `config`: its host's Build Server, stated by the settings
        `build_settings` (the run's server's by default,
        :func:`uniserve_testing.builds.build_config`), and the run's
        :meth:`run_config` over `config`.
        """  # noqa: D205
        if build_settings is None and self.build_server is not None:
            build_settings = self.build_server.settings
        build = (
            builds.build_config(build_settings)
            if build_settings is not None
            else {}
        )
        return merged(config, build, self.run_config())

    def package(self, name: str, config: Config) -> Path:
        """Write the package of the deployment `name` whose own options are
        `config` (:meth:`config`) into the case's directory; its directory,
        whose name is the deployment's.
        """  # noqa: D205
        return packages.write(self.directory / name, self.config(config))

    def environment(self, runtime: Path) -> dict[str, str]:
        """The environment of a daemon under the runtime directory
        `runtime`: on the leased GPUs, with the case's own state directory.
        """  # noqa: D205
        declared = {
            "XDG_RUNTIME_DIR": str(runtime),
            "XDG_STATE_HOME": str(self.directory / "state"),
        }
        if self.lease is not None:
            declared["CUDA_VISIBLE_DEVICES"] = self.lease.visible
        return environment(**declared)

    @contextmanager
    def serve(
        self,
        config: Config,
        *,
        timeout_s: float | None = None,
        name: str = "server",
        serving: bool = True,
        host_gib: int | None = HOST_CACHE_GIB,
        runtime: Path | None = None,
        calibration: Path | None = None,
    ) -> Iterator[Server]:
        """Serve the deployment `name` of the options `config` until the
        block ends, once its readiness record states it serves (with
        `serving` unset, once it states its listeners, for a leader whose
        members join it next: :meth:`follower`).

        Args:
            config: The deployment's own options, its ``model`` among them.
            timeout_s: A bound on its start, in seconds (scaled); none by
                default: the server states its record or ends.
            name: The deployment's name: its package's directory in the
                case's directory, and its log's (``<name>.log``).
            serving: Whether to wait for its serving record.
            host_gib: Its host tier in GiB (:data:`HOST_CACHE_GIB`); ``None``
                for a case whose `config` sizes it.
            runtime: A runtime directory the caller owns
                (:func:`runtime_directory`), for a case whose deployments
                share their Storage Server; a fresh one by default.
            calibration: The transfer calibration its Storage Server copies
                by (:meth:`calibration`), for a deployment that transfers;
                without one its Storage Server runs no Transfer Server.

        The yielded server's ``package`` is the package's directory.
        """  # noqa: D205
        sized: dict[str, Any] = (
            {"kv_cache_config": {"host_gib": host_gib}}
            if host_gib is not None
            else {}
        )
        if calibration is not None:
            sized = merged(
                sized,
                {"storage_config": {"transfer_calibration": str(calibration)}},
            )
        package = self.package(name, merged(config, sized))
        with ExitStack() as owned:
            if runtime is None:
                runtime = owned.enter_context(
                    runtime_directory(
                        self.uniserve, self.directory / f"{name}.storage"
                    )
                )
            server = launch(
                [str(self.uniserve), "serve", str(package)],
                self.directory / f"{name}.log",
                self.environment(runtime),
                scale=self.settings.time_scale,
            )
            server.package = package
            owned.callback(server.stop)
            server.wait("serving" if serving else "listening", timeout_s)
            yield server

    def calibration(self, *, reads: bool = False) -> Path:
        """The transfer calibration of the case's GPUs and this host's
        storage, which ``uniserve storage-server calibrate`` measures on the
        leased GPUs (and with `reads` NIXL's READs through the host's RDMA
        ports), written into the case's directory once per case.
        """  # noqa: D205
        calibration = self.directory / "transfer-calibration.json"
        if not calibration.exists():
            declared = {}
            if self.lease is not None:
                declared["CUDA_VISIBLE_DEVICES"] = self.lease.visible
            subprocess.run(
                [
                    str(self.uniserve),
                    "storage-server",
                    "calibrate",
                    "--output",
                    str(calibration),
                    *(["--reads"] if reads else []),
                ],
                env=environment(**declared),
                check=True,
            )
        return calibration

    @contextmanager
    def follower(
        self,
        host: RemoteHost,
        leader: Server,
        repository: Path,
        *,
        timeout_s: float | None = None,
        devices: Sequence[int] | None = None,
        calibration: Path | None = None,
        reads: bool = False,
        name: str = "follower",
    ) -> Iterator[Follower]:
        """The daemon of a deployment's second host, serving the package of
        `leader` there until the block ends.

        `leader` is a leader started with ``serving=False`` whose package
        lists this host and `host` (:func:`two_hosts`): the second host's
        daemon finds itself in the package's ``hosts`` by its host name and
        joins the leader at the first host's ``admin_config.join``, and the
        leader serves once it joined (within `timeout_s` seconds, scaled,
        when stated), which this waits for before the block begins. Each
        host resolves the other's name (``docs/testing.md``, Two hosts).

        The second host serves the leader's package with its own Build
        Server (:func:`remote_build_server`) as ``build_config``, in a
        package directory of the same name, so the deployment is the same
        on both hosts. Its daemon runs on `devices` of the host (all of its
        GPUs without them) under a runtime directory of its own there,
        which also holds its state directory. A deployment that names a
        transfer calibration names it on every host it serves on, so with
        `calibration` (the path the leader's package names) the host's own
        is measured at that path there first, with NIXL's READs when
        `reads`. On exit the leader stops first; the daemon ends when the
        leader's connection closes, which this waits for, and a daemon that
        outlives its leader fails the case. Its log is ``<name>.log`` in
        the case's directory.
        """  # noqa: D205
        assert leader.package is not None, "the leader serves no package"
        with ExitStack() as owned:
            runtime = owned.enter_context(host.runtime_directory())
            settings = owned.enter_context(
                remote_build_server(host, repository, runtime)
            )
            config = merged(
                packages.read(leader.package), builds.build_config(settings)
            )
            package = f"{runtime}/{leader.package.name}"
            host.run("mkdir", "-p", package)
            host.write(f"{package}/{packages.CONFIG}", packages.text(config))
            declared = {
                "XDG_RUNTIME_DIR": runtime,
                "XDG_STATE_HOME": f"{runtime}/state",
            }
            if devices is not None:
                declared["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
            if calibration is not None:
                host.run("mkdir", "-p", str(calibration.parent))
                host.run(
                    "env",
                    *(f"{key}={value}" for key, value in declared.items()),
                    "uniserve",
                    "storage-server",
                    "calibrate",
                    "--output",
                    str(calibration),
                    *(["--reads"] if reads else []),
                    timeout_s=CALIBRATION_S,
                )
                owned.callback(host.run, "rm", "-f", str(calibration))
            member = host.start(
                ["uniserve", "serve", package],
                self.directory / f"{name}.log",
                environment=declared,
            )
            owned.callback(stop_remote_storage_server, host, runtime)
            try:
                leader.wait("serving", timeout_s)
                yield Follower(runtime=runtime)
            finally:
                leader.stop()
                try:
                    member.wait(timeout=leader.scaled(FOLLOWER_EXIT_S))
                except subprocess.TimeoutExpired:
                    member.kill()
                    member.wait()
                    raise AssertionError(
                        "the second host's daemon outlived its leader"
                    ) from None


@contextmanager
def remote_build_server(
    host: RemoteHost, repository: Path, runtime: str
) -> Iterator[dict]:
    """A Build Server on `host` under its runtime directory `runtime`, with
    the tree's packages installed, until the block ends; the settings it
    serves with, as its readiness record states them.

    A main process resolves its modules at its own host's Build Server, which
    installs the tree's packages from its source tree, this repository's
    path, which the other host must mount at the same path. The server starts
    detached (``--daemon``) and its starter prints its readiness record once
    it has installed them and serves.
    """  # noqa: D205
    started = host.run(
        "env",
        f"XDG_RUNTIME_DIR={runtime}",
        "uniserve",
        "build-server",
        "serve",
        "--source",
        str(repository),
        "--package-root",
        f"{runtime}/packages",
        "--store-root",
        f"{runtime}/modules",
        "--daemon",
        timeout_s=builds.START_S,
    )
    record = json.loads(started.splitlines()[-1])
    socket = socket_path(record["endpoints"]["build"])
    try:
        yield record["settings"]
    finally:
        # The Build Server has no stop request: it ends on SIGTERM, and it
        # holds the lock beside its socket for its life, so taking the lock
        # waits until it ended, before its directory goes.
        host.run("kill", str(record["pid"]))
        host.run("flock", str(socket.parent / "build-server.lock"), "true")


def stop_remote_storage_server(host: RemoteHost, runtime: str) -> None:
    """Stop the Storage Server serving under the runtime directory `runtime`
    of `host`, if one serves, with its ``stop`` request.
    """  # noqa: D205
    under = ("env", f"XDG_RUNTIME_DIR={runtime}", "uniserve", "storage-server")
    try:
        host.run(*under, "status")
    except subprocess.CalledProcessError:
        return
    host.run(*under, "stop")
