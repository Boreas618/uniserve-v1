"""ManagedServer releases every resource it acquires for a launch.

The servers are :mod:`uniserve_testing.standins.reporter` processes, which
take a load time to start and then report their start to a listener of the
test's own, on a port the kernel chose. That listener is the address each
server's profile names: it accepts connections from the moment the test opens
it and answers its readiness path with 503 until its server reports, then
with 200.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from uniserve_eval.config import ServerLaunch, ServerProfile
from uniserve_eval.server import ManagedDeployment, ManagedServer

pytestmark = pytest.mark.unit

# The path of a profile's readiness probe; the listener answers any path.
_READY_PATH = "/ready"


class _Listener(ThreadingHTTPServer):
    """The address a stand-in server's profile names, and its report's sink.

    It answers every ``GET`` with 503 until a report arrives and with 200
    after it; the report is a ``POST`` whose body ``report`` keeps.
    """

    def __init__(self) -> None:
        """Bind a loopback port the kernel chooses and listen on it."""
        super().__init__(("127.0.0.1", 0), _ListenerHandler)
        self.reported = threading.Event()
        self.report = b""

    @property
    def report_url(self) -> str:
        """The URL a stand-in reports its start to."""
        return f"http://127.0.0.1:{self.server_port}/report"

    def profile(
        self, name: str, command: tuple[str, ...], ready_path: str | None
    ) -> ServerProfile:
        """A profile whose readiness this listener answers."""
        return ServerProfile(
            name,
            command,
            "127.0.0.1",
            self.server_port,
            {},
            ready_path=ready_path,
        )


class _ListenerHandler(BaseHTTPRequestHandler):
    """Answers readiness probes and takes a stand-in's report."""

    server: _Listener

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler method name.
        """Answer a readiness probe: 200 once the report arrived."""
        self.send_response(200 if self.server.reported.is_set() else 503)
        self.send_header("content-length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler method name.
        """Keep the report's body and mark the server started."""
        length = int(self.headers.get("content-length", "0"))
        self.server.report = self.rfile.read(length)
        self.server.reported.set()
        self.send_response(204)
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        """Keep the probes out of the test's output."""


@pytest.fixture
def listener() -> Iterator[Callable[[], _Listener]]:
    """Open listeners on demand; close each at the test's end."""
    opened: list[_Listener] = []

    def open_listener() -> _Listener:
        server = _Listener()
        threading.Thread(target=server.serve_forever, daemon=True).start()
        opened.append(server)
        return server

    yield open_listener

    for server in opened:
        server.shutdown()
        server.server_close()


def _reporter(listener: _Listener, *options: str) -> tuple[str, ...]:
    """The command of a stand-in server that reports to ``listener``."""
    return (
        sys.executable,
        "-m",
        "uniserve_testing.standins.reporter",
        "--report",
        listener.report_url,
        *options,
    )


def _open_descriptors(path: Path) -> list[str]:
    """Return this process's file descriptors that refer to ``path``."""
    descriptors = []
    for descriptor in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except FileNotFoundError:
            # The descriptor `listdir` itself used is closed by now.
            continue
        if target == str(path):
            descriptors.append(descriptor)
    return descriptors


def test_failed_launch_closes_the_server_log(tmp_path: Path) -> None:
    log_path = tmp_path / "server.log"
    executable = tmp_path / "missing-server"
    server = ManagedServer(
        ServerProfile("local", (str(executable),), "127.0.0.1", 9, {}),
        ServerLaunch((str(executable),), tmp_path, {}),
        log_path,
        timeout_s=1.0,
    )

    with pytest.raises(FileNotFoundError):
        with server:
            pass

    assert _open_descriptors(log_path) == []


@pytest.mark.parametrize("start_worker", [True, False], ids=["worker", "alone"])
def test_stop_terminates_the_group_after_the_server_was_killed(
    tmp_path: Path,
    listener: Callable[[], _Listener],
    wait_exit: Callable[[int, float], bool],
    start_worker: bool,
) -> None:
    address = listener()
    command = _reporter(address, *(["--worker"] if start_worker else []))
    server = ManagedServer(
        address.profile("local", command, _READY_PATH),
        ServerLaunch(command, tmp_path, {}),
        tmp_path / "server.log",
        timeout_s=30.0,
    )

    worker: int | None = None
    try:
        with server:
            # The server is ready once it reported, with its worker's pid.
            if start_worker:
                worker = int(address.report)
            # An abrupt server exit, such as SIGKILL or an OOM kill, skips
            # the server's own worker cleanup.
            assert server.process is not None
            os.kill(server.process.pid, signal.SIGKILL)
            server.process.wait()

        # Exiting the context signals the group; a group left with no
        # member must not make it raise.
        if worker is not None:
            assert wait_exit(worker, 5.0)
    finally:
        if worker is not None:
            with suppress(ProcessLookupError):
                os.kill(worker, signal.SIGKILL)


def test_startup_time_runs_from_launch_until_the_process_is_ready(
    tmp_path: Path, listener: Callable[[], _Listener]
) -> None:
    """A deployment reports how long its slowest process took to be ready."""
    # Each stand-in loads for its delay before it reports its start.
    delays = (0.2, 0.6)
    processes = []
    for index, delay in enumerate(delays):
        address = listener()
        command = _reporter(address, "--load-s", str(delay))
        processes.append(
            (
                address.profile(f"replica-{index}", command, _READY_PATH),
                ServerLaunch(command, tmp_path, {}),
                tmp_path / f"{index}.log",
            )
        )

    deployment = ManagedDeployment(processes, timeout_s=30.0)
    assert deployment.startup_s is None
    with deployment:
        startup = deployment.startup_s

    assert startup is not None and startup >= max(delays)
    assert startup == max(server.startup_s for server in deployment.servers)


@pytest.mark.parametrize("ready_path", [None, _READY_PATH])
def test_a_readiness_path_holds_startup_until_the_server_reports_ready(
    tmp_path: Path, listener: Callable[[], _Listener], ready_path: str | None
) -> None:
    """A server that listens before it has started is ready only then."""
    address = listener()
    load_s = 1.0
    command = _reporter(address, "--load-s", str(load_s))
    server = ManagedServer(
        address.profile("local", command, ready_path),
        ServerLaunch(command, tmp_path, {}),
        tmp_path / "server.log",
        timeout_s=30.0,
    )

    with server:
        startup = server.startup_s

    assert startup is not None
    if ready_path is None:
        # The listener alone accepts long before the load ends.
        assert startup < load_s
    else:
        assert startup >= load_s
