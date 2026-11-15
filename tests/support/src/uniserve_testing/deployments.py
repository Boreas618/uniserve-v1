"""The deployments an end-to-end case starts.

A case asks the ``deployments`` fixture (``tests/python/e2e/conftest.py``)
for a :class:`Deployments`, which starts each deployment with
``uniserve serve`` and stops it with the case:

- on the GPUs the case leases, which the daemon sees as ``cuda:0`` onwards
  (CUDA's device visibility, stated in its environment);
- at the run's Build Server (``build_config.socket``), which holds the
  tree's packages;
- under a runtime directory of its own (``XDG_RUNTIME_DIR``, which the
  caller of a UniServe server creates and names), where its daemon starts a
  Storage Server that serves this deployment alone and that the case stops
  when the deployment ends;
- with the run's engine package (``--engine``) and assignments
  (``--serve-set``), and the waits stretched by its time scale.

The daemon binds port 0 and reports what it bound in its readiness record
(:class:`uniserve_testing.servers.Server`).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path

from uniserve_testing import builds
from uniserve_testing.builds import BuildServer
from uniserve_testing.devices import Lease
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
    """What the run's options state for every deployment of the run."""

    time_scale: float
    serve_set: tuple[str, ...]
    engine: str | None


def two_host_leader(host: RemoteHost) -> tuple[str, ...]:
    """The options of the leader of a deployment over this host and `host`:
    node ``rank-0`` of two, which binds its join endpoint at the address
    `host` reaches this host at on a port it picks, and reports it in its
    listening record (:meth:`Deployments.follower`).
    """  # noqa: D205
    return (
        "--nnodes",
        "2",
        "--node-rank",
        "0",
        "--dist-init-addr",
        f"{host.leader_address}:0",
        "--host-identity",
        "rank-0",
    )


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
        the GPUs of `lease`, with their logs under `directory`.
        """  # noqa: D205
        self.uniserve = uniserve
        self.build_server = build_server
        self.settings = settings
        self.lease = lease
        self.directory = directory

    def assignments(self) -> list[str]:
        """The ``PATH=VALUE`` assignments every deployment of the run
        states: its engine package (``--engine``) and its ``--serve-set``,
        as a deployment that a model-less node brings up (``PUT
        /admin/config``) states them too.
        """  # noqa: D205
        engine = (
            [f'worker_config.engine="{self.settings.engine}"']
            if self.settings.engine is not None
            else []
        )
        return [*engine, *self.settings.serve_set]

    def options(self, build_socket: str | None = None) -> list[str]:
        """The options every deployment of the run states after its own: the
        Build Server at `build_socket` (the run's by default) and the run's
        :meth:`assignments`.
        """  # noqa: D205
        if build_socket is None and self.build_server is not None:
            build_socket = str(self.build_server.socket)
        stated: list[str] = []
        if build_socket is not None:
            stated += ["--set", f'build_config.socket="{build_socket}"']
        for assignment in self.assignments():
            stated += ["--set", assignment]
        return stated

    def command(self, *options: str, model: str | None = None) -> list[str]:
        """The ``uniserve serve`` command of a deployment of `model` (a
        repository, :mod:`uniserve_testing.assets`) with `options`, as this
        run starts it.
        """  # noqa: D205
        return [
            str(self.uniserve),
            "serve",
            *([model] if model is not None else []),
            *options,
            *self.options(),
        ]

    def environment(self, runtime: Path) -> dict[str, str]:
        """The environment of a daemon under the runtime directory
        `runtime`: on the leased GPUs.
        """  # noqa: D205
        declared = {"XDG_RUNTIME_DIR": str(runtime)}
        if self.lease is not None:
            declared["CUDA_VISIBLE_DEVICES"] = self.lease.visible
        return environment(**declared)

    @contextmanager
    def serve(
        self,
        *options: str,
        model: str | None = None,
        timeout_s: float | None = None,
        name: str = "server",
        serving: bool = True,
        host_gib: int | None = HOST_CACHE_GIB,
        runtime: Path | None = None,
        calibration: Path | None = None,
    ) -> Iterator[Server]:
        """Serve a deployment of `model` with `options` until the block ends,
        once its readiness record states it serves (with `serving` unset,
        once it states its listeners, for a leader whose members join it
        next: :meth:`follower`).

        Args:
            options: The deployment's own options.
            model: The repository it serves; unset when `options` name a
                checkpoint directory or none.
            timeout_s: A bound on its start, in seconds (scaled); none by
                default: the server states its record or ends.
            name: The name of its log (``<name>.log``) in the case's
                directory.
            serving: Whether to wait for its serving record.
            host_gib: Its host tier in GiB (:data:`HOST_CACHE_GIB`); ``None``
                for a case whose `options` size it.
            runtime: A runtime directory the caller owns
                (:func:`runtime_directory`), for a case whose deployments
                share their Storage Server; a fresh one by default.
            calibration: The transfer calibration its Storage Server copies
                by (:meth:`calibration`), for a deployment that transfers;
                without one its Storage Server runs no Transfer Server.
        """  # noqa: D205
        sized = (
            ("--set", f"kv_cache_config.host_gib={host_gib}")
            if host_gib is not None
            else ()
        )
        if calibration is not None:
            sized = (
                *sized,
                "--set",
                f'storage_config.transfer_calibration="{calibration}"',
            )
        with ExitStack() as owned:
            if runtime is None:
                runtime = owned.enter_context(
                    runtime_directory(
                        self.uniserve, self.directory / f"{name}.storage"
                    )
                )
            server = launch(
                self.command(*options, *sized, model=model),
                self.directory / f"{name}.log",
                self.environment(runtime),
                scale=self.settings.time_scale,
            )
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
        *options: str,
        timeout_s: float | None = None,
        devices: Sequence[int] | None = None,
        calibration: Path | None = None,
        reads: bool = False,
        name: str = "follower",
    ) -> Iterator[Follower]:
        """The daemon of a deployment's second host, joining `leader` as
        node ``rank-1`` with `options` after its own, until the block ends.

        `leader` is a leader started with ``serving=False``: its listening
        record states the endpoint its members join at, and it serves once
        this member joined (within `timeout_s` seconds, scaled, when
        stated), which this waits for before the block begins. The daemon
        runs on `devices` of the host (all of its GPUs without them) under a
        runtime directory of its own there, at a Build Server of its own
        there (:func:`remote_build_server`). A deployment that names a
        transfer calibration names it on every host it serves on, so with
        `calibration` (the path the leader's options name) the host's own is
        measured at that path there first, with NIXL's READs when `reads`.
        On exit the leader stops first; the daemon ends when the leader's
        connection closes, which this waits for, and a daemon that outlives
        its leader fails the case.
        """  # noqa: D205
        with ExitStack() as owned:
            runtime = owned.enter_context(host.runtime_directory())
            socket = owned.enter_context(
                remote_build_server(host, repository, runtime)
            )
            declared = {"XDG_RUNTIME_DIR": runtime}
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
            argv = [
                "uniserve",
                "serve",
                "--nnodes",
                "2",
                "--node-rank",
                "1",
                "--dist-init-addr",
                leader.join,
                "--host-identity",
                "rank-1",
                *options,
                *self.options(build_socket=socket),
            ]
            member = host.start(
                argv, self.directory / f"{name}.log", environment=declared
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
) -> Iterator[str]:
    """A Build Server on `host` under its runtime directory `runtime`, with
    the tree's packages installed, until the block ends; its socket there.

    A main process resolves its modules at its own host's Build Server, and
    nothing distributes the leader's package revisions to another host's
    server, so the case installs the tree's packages there itself from this
    repository's path, which the other host must mount at the same path
    (``specs/ad-hoc.md`` §9, AU-001). The server starts detached
    (``--daemon``) and its starter prints its readiness record once it
    serves.
    """  # noqa: D205
    started = host.run(
        "env",
        f"XDG_RUNTIME_DIR={runtime}",
        "uniserve",
        "build-server",
        "serve",
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
        for package in builds.tree_packages(repository):
            host.run(
                "uniserve",
                "build-server",
                "install",
                "--socket",
                str(socket),
                str(package),
            )
        yield str(socket)
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
