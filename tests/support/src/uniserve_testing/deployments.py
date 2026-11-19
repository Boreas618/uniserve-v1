"""The deployments an end-to-end case starts.

A case asks the ``deployments`` fixture (``tests/python/e2e/conftest.py``)
for a :class:`Deployments`, which writes each deployment's policy package
(:mod:`uniserve_testing.packages`) into the case's directory, starts it with
``uniserve serve <package>`` and stops it with the case:

- on the GPUs the case leases, which the daemon sees as ``cuda:0`` onwards
  (CUDA's device visibility, stated in its environment);
- with a Build Server of its own (``build_config``), which reads the run's
  store, where the tree's packages are installed, and keeps what it compiles
  in the run's data directory under the deployment's name
  (:func:`uniserve_testing.builds.deployment_config`);
- under a runtime directory of its own (``XDG_RUNTIME_DIR``, which the
  caller of a UniServe server creates and names), where its daemon starts
  the deployment's Storage Server and Build Server, which serve this
  deployment alone and which the case stops when the deployment ends;
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

import contextlib
import dataclasses
import fcntl
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
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


#: How long a deployment's Build Server may take to end once it was sent
#: SIGTERM, in seconds: it ends its builds' compiler processes.
BUILD_SERVER_STOP_S = 60.0


@contextmanager
def runtime_directory(
    uniserve: Path, kept: Path | None = None
) -> Iterator[Path]:
    """A runtime directory (``XDG_RUNTIME_DIR``) of the caller's own, whose
    deployments' servers are stopped when the block ends.

    A daemon started under it starts its deployment's Storage Server and
    Build Server there, which outlive the daemon; the block's end stops them
    (:func:`stop_deployment_servers`), keeps the logs they wrote beside their
    sockets in the directory `kept` when one is named, and removes the
    runtime directory. A Unix socket's path holds at most 107 bytes, so the
    directory lies in the system's temporary directory.
    """  # noqa: D205
    runtime = Path(tempfile.mkdtemp(prefix="uniserve-"))
    try:
        yield runtime
    finally:
        stop_deployment_servers(uniserve, runtime, kept)
        shutil.rmtree(runtime, ignore_errors=True)


def deployments_under(runtime: Path) -> list[str]:
    """The deployments whose servers keep their sockets under the runtime
    directory `runtime`, by name.
    """  # noqa: D205
    directory = runtime / "uniserve" / "deployments"
    if not directory.is_dir():
        return []
    return sorted(entry.name for entry in directory.iterdir() if entry.is_dir())


def storage_socket(runtime: Path, deployment: str | None = None) -> Path:
    """The socket of the Storage Server of `deployment` under the runtime
    directory `runtime`, or of its one deployment when unnamed.
    """  # noqa: D205
    if deployment is None:
        (deployment,) = deployments_under(runtime)
    return builds.deployment_socket(runtime, deployment, "storage-server")


def storage_status(
    uniserve: Path, runtime: Path, deployment: str | None = None
) -> dict | None:
    """The readiness record of the Storage Server of `deployment` (the one
    deployment under `runtime` when unnamed) serving under the runtime
    directory `runtime` (``storage-server status``), or ``None`` when none
    serves there.
    """  # noqa: D205
    if deployment is None and len(deployments_under(runtime)) != 1:
        return None
    answered = subprocess.run(
        [
            str(uniserve),
            "storage-server",
            "status",
            "--socket",
            str(storage_socket(runtime, deployment)),
        ],
        env=environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    if answered.returncode != 0:
        return None
    return json.loads(answered.stdout)


def stop_deployment_servers(
    uniserve: Path, runtime: Path, kept: Path | None = None
) -> None:
    """Stop the servers of every deployment under `runtime` and keep the
    logs beside their sockets in `kept` when one is named.

    A Storage Server stops on its ``stop`` request, which returns once it
    ended. A Build Server has no stop request: it ends on SIGTERM, and it
    holds the lock beside its socket for its life, so taking the lock waits
    until it ended, before its package root and its directory go.
    """  # noqa: D205
    for deployment in deployments_under(runtime):
        if storage_status(uniserve, runtime, deployment) is not None:
            subprocess.run(
                [
                    str(uniserve),
                    "storage-server",
                    "stop",
                    "--socket",
                    str(storage_socket(runtime, deployment)),
                ],
                env=environment(),
                capture_output=True,
                text=True,
                check=True,
            )
        build = builds.deployment_socket(runtime, deployment, builds.SERVICE)
        ready = build.with_suffix(".ready")
        if ready.exists():
            pid = json.loads(ready.read_text())["pid"]
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)
            with build.with_suffix(".lock").open("a") as lock:
                deadline = time.monotonic() + BUILD_SERVER_STOP_S
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() > deadline:
                            with contextlib.suppress(ProcessLookupError):
                                os.kill(pid, signal.SIGKILL)
                            fcntl.flock(lock, fcntl.LOCK_EX)
                            break
                        time.sleep(0.1)
        if kept is not None:
            kept.mkdir(parents=True, exist_ok=True)
            for log in build.parent.glob("*.log"):
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
        #: The case's directory on this host's own storage, which holds its
        #: transfer calibration (:meth:`calibration`); made once one is
        #: measured.
        self.local: Path | None = None

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

    def config(self, name: str, config: Config) -> dict[str, Any]:
        """The ``config.toml`` of the deployment `name` of the run whose own
        options are `config`: its Build Server over the run's store, with a
        package root of the case's own
        (:func:`uniserve_testing.builds.deployment_config`), and the run's
        :meth:`run_config` over `config`.
        """  # noqa: D205
        build = (
            builds.deployment_config(
                self.build_server, self.directory / f"{name}.packages"
            )
            if self.build_server is not None
            else {}
        )
        return merged(config, build, self.run_config())

    def package(self, name: str, config: Config) -> Path:
        """Write the package of the deployment `name` whose own options are
        `config` (:meth:`config`) into the case's directory; its directory,
        whose name is the deployment's.
        """  # noqa: D205
        return packages.write(self.directory / name, self.config(name, config))

    def environment(self, runtime: Path) -> dict[str, str]:
        """The environment of a daemon under the runtime directory
        `runtime`: on the leased GPUs, with the case's own state directory,
        and with the run's data directory for deployments
        (:attr:`uniserve_testing.builds.BuildServer.data`), where each
        deployment's Build Server keeps its modules under its name.
        """  # noqa: D205
        declared = {
            "XDG_RUNTIME_DIR": str(runtime),
            "XDG_STATE_HOME": str(self.directory / "state"),
        }
        if self.build_server is not None and self.build_server.data is not None:
            declared["XDG_DATA_HOME"] = str(self.build_server.data / "data")
            declared["XDG_CACHE_HOME"] = str(self.build_server.data / "cache")
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
        log: str | None = None,
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
                case's directory, and its log's (``<name>.log``) unless
                `log` names it. A case that starts the same deployment
                again under one runtime directory names it alike each
                time, so that the start finds the deployment's Storage
                Server and Build Server, with the storage it retained.
            log: The name of this start's log in the case's directory
                (``<log>.log``), for a case that starts one deployment more
                than once and keeps each start's log.
            serving: Whether to wait for its serving record.
            host_gib: Its host tier in GiB (:data:`HOST_CACHE_GIB`); ``None``
                for a case whose `config` sizes it.
            runtime: A runtime directory the caller owns
                (:func:`runtime_directory`), for a case that reads a
                deployment's Storage Server or starts the deployment again
                beside the one it left; a fresh one by default.
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
                self.directory / f"{log or name}.log",
                self.environment(runtime),
                scale=self.settings.time_scale,
            )
            server.package = package
            server.runtime = runtime
            owned.callback(server.stop)
            if serving:
                server.wait("serving", timeout_s)
                self.bind_build_server(server)
            else:
                # A leader starts its deployment's Build Server once the
                # deployment's other nodes joined it: :meth:`follower` binds
                # it once the leader serves.
                server.wait("listening", timeout_s)
            yield server

    def bind_build_server(self, server: Server) -> None:
        """Bind the Build Server of the serving daemon `server`'s deployment
        (``server.build_server``), which its daemon started before its
        engines: where a case installs a revision the deployment selects
        next.
        """  # noqa: D205
        assert server.package is not None and server.runtime is not None
        server.build_server = BuildServer.of(
            self.uniserve, server.runtime, server.package.name
        )

    def calibration(self, *, reads: bool = False) -> Path:
        """The transfer calibration of the case's GPUs and this host's
        storage, which ``uniserve storage-server calibrate`` measures on the
        leased GPUs (and with `reads` NIXL's READs through the host's RDMA
        ports), once per case.

        It lies in a directory of the case's own in the system's temporary
        directory, which is the host's own, rather than in the case's
        directory: a deployment over two hosts names one calibration path,
        which each host's Storage Server reads its own calibration from
        (:meth:`follower` measures the second host's there), and the case's
        directory may lie on storage both hosts mount. :meth:`close` removes
        it.
        """  # noqa: D205
        if self.local is None:
            self.local = Path(tempfile.mkdtemp(prefix="uniserve-calibration-"))
        calibration = self.local / "transfer-calibration.json"
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

    def close(self) -> None:
        """Remove what the case's deployments keep on this host outside the
        case's directory: the transfer calibration (:meth:`calibration`).
        """  # noqa: D205
        if self.local is not None:
            shutil.rmtree(self.local, ignore_errors=True)
            self.local = None

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
        killed: bool = False,
        name: str = "follower",
    ) -> Iterator[Follower]:
        """The daemon of a deployment's second host, serving the package of
        `leader` there until the block ends.

        `leader` is a leader started with ``serving=False`` whose package
        lists this host and `host` (:func:`two_hosts`): the second host's
        daemon finds itself in the package's ``hosts`` by its host name and
        joins the leader at the first host's ``admin_config.join``, and the
        leader serves once it joined (within `timeout_s` seconds, scaled,
        when stated), which this waits for before the block begins, and
        binds the leader's ``build_server`` then. Each
        host resolves the other's name (``docs/testing.md``, Two hosts).

        The second host serves the leader's package, with the deployment's
        Build Server of that host its daemon starts from the repository's
        packages as ``build_config``, in a package directory of the same
        name, so the deployment is the same
        on both hosts. Its daemon runs on `devices` of the host (all of its
        GPUs without them) under a runtime directory of its own there,
        which also holds its state directory. A deployment that names a
        transfer calibration names it on every host it serves on, so with
        `calibration` (the path the leader's package names, on this host's
        own storage: :meth:`calibration`) the host's own is measured at that
        path there first, with NIXL's READs when `reads`, and its directory
        there is removed with the block. On exit the leader stops first;
        the daemon ends when the leader's connection closes, which this
        waits for, and a daemon that outlives its leader fails the case, as
        does a process of it that did not exit with status 0, such as a main
        process that ended by a signal in its close; a case that kills a
        process of the second host itself states it (`killed`), and the kills
        (``SIGKILL``) are not counted then.
        Its log is ``<name>.log`` in the case's directory.
        """  # noqa: D205
        assert leader.package is not None, "the leader serves no package"
        with ExitStack() as owned:
            runtime = owned.enter_context(host.runtime_directory())
            # The deployment's Build Server on the second host is that host's
            # own, which its daemon starts: it installs the tree's packages
            # from the repository's path, which the other host mounts at the
            # same path, into a package root of the case's runtime directory.
            config = packages.read(leader.package)
            config["build_config"] = {
                "source": str(repository),
                "package_root": f"{runtime}/packages",
            }
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
                owned.callback(host.run, "rm", "-rf", str(calibration.parent))
            log = self.directory / f"{name}.log"
            member = host.start(
                ["uniserve", "serve", package],
                log,
                environment=declared,
            )
            owned.callback(
                stop_remote_servers, host, runtime, leader.package.name
            )
            try:
                leader.wait("serving", timeout_s)
                self.bind_build_server(leader)
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
            failed = [
                status
                for status in failed_exits(log)
                if not (killed and status.startswith(KILLED_STATUS))
            ]
            assert not failed, (
                f"a process of the second host's daemon failed: {failed}"
            )


# A daemon's report of a supervised process that exited otherwise than with
# status 0: its supervisor states the status only then
# (`crates/daemon/src/supervision/supervisor.rs`), on one line of its log.
FAILED_EXIT = re.compile(r"a process exited .*\bstatus=(?P<status>.*)$")

# How the daemon states the status of a process a kill ended (Rust's
# `ExitStatus` display).
KILLED_STATUS = "signal: 9 (SIGKILL)"

# The terminal colours of a log line, which the pattern reads past.
COLOURS = re.compile(r"\x1b\[[0-9;]*m")


def failed_exits(log: Path) -> list[str]:
    """The statuses the daemon whose log is `log` reported for processes
    it supervised that did not exit with status 0, in the log's order.
    """  # noqa: D205
    text = log.read_text(encoding="utf-8", errors="replace")
    return [
        found["status"]
        for line in text.splitlines()
        if (found := FAILED_EXIT.search(COLOURS.sub("", line)))
    ]


def stop_remote_servers(
    host: RemoteHost, runtime: str, deployment: str
) -> None:
    """Stop the servers of the deployment `deployment` under the runtime
    directory `runtime` of `host`: its Storage Server with its ``stop``
    request, and its Build Server, which has no stop request, with SIGTERM,
    waiting on the lock it holds for its life until it ended.
    """  # noqa: D205
    under = (
        "env",
        f"XDG_RUNTIME_DIR={runtime}",
        "uniserve",
        "storage-server",
    )
    try:
        host.run(*under, "status", "--deployment", deployment)
    except subprocess.CalledProcessError:
        pass
    else:
        host.run(*under, "stop", "--deployment", deployment)
    build = builds.deployment_socket(Path(runtime), deployment, builds.SERVICE)
    try:
        record = json.loads(host.run("cat", str(build.with_suffix(".ready"))))
    except subprocess.CalledProcessError:
        return
    host.run("kill", str(record["pid"]))
    host.run("flock", str(build.with_suffix(".lock")), "true")
