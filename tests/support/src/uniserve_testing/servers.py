"""Servers a test starts, and what their readiness records state.

Every UniServe server binds all of its listeners when it starts and reports
them in a readiness record (``docs/admin.md``, Listeners): one JSON line per
state, written to the descriptor its caller passes with ``--ready-fd``,
``listening`` once its listeners are bound and ``serving`` once it serves,
after which it closes the descriptor. :func:`launch` starts a server with
the write end of a pipe at that option and returns a :class:`Server` that
reads the records, so a test neither chooses a port nor probes one.

A launched server runs with a declared environment (:func:`environment`):
the search path its programs are found on, the user's home (where the
Hugging Face cache lies unless the standard ``HF_HOME`` names it), the
standard Hugging Face variables the run states, and what the caller states
for the server itself (its runtime directory, the GPUs it may see). Nothing
else of the test's environment reaches it.
"""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

#: The test's variables a launched server inherits: the search path its
#: programs are run from (the Build Server runs its environment's program
#: from it), the user's home, and Hugging Face's standard variables, which
#: name the cache, the endpoint, the token and offline operation.
INHERITED = ("PATH", "HOME")
HUGGING_FACE = "HF_"


def environment(**declared: str) -> dict[str, str]:
    """The environment a launched server runs with: :data:`INHERITED` and
    the run's Hugging Face variables as the test has them, and `declared`.
    """  # noqa: D205
    inherited = {
        name: value
        for name, value in os.environ.items()
        if name in INHERITED or name.startswith(HUGGING_FACE)
    }
    return {**inherited, **declared}


def log_tail(log_path: Path | None, lines: int = 40) -> str:
    """The last `lines` lines of a server's log, marked, for a failure message.

    The tail is what usually names a server's failure; a missing or
    unreadable file says so rather than failing.
    """
    if log_path is None:
        return ""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return f"\n(server log: no file at {log_path})"
    except OSError as error:
        return f"\n(server log unreadable at {log_path}: {error})"
    tail = "\n".join(text.splitlines()[-lines:])
    return (
        f"\n--- server log tail ({log_path}) ---\n{tail}\n--- end of tail ---"
    )


def socket_path(endpoint: str) -> Path:
    """The path of the Unix socket a record's endpoint (``unix:PATH``) names."""
    if not endpoint.startswith("unix:"):
        raise ValueError(f"{endpoint} is no Unix socket")
    return Path(endpoint.removeprefix("unix:"))


class Server:
    """A server this process started, and what its readiness records state.

    ``scale`` stretches every wait on the server (``--time-scale``): a run
    whose processes run under a tool lets them take longer.
    """

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        records: int,
        log_path: Path | None,
        *,
        scale: float = 1.0,
        address: str | None = None,
    ) -> None:
        """A server run as `process`, its records arriving on the descriptor
        `records`, its log at `log_path`. `address` replaces an unspecified
        host (``0.0.0.0``) of its client endpoint in :attr:`base_url`, for a
        server another host reaches.
        """  # noqa: D205
        self.process = process
        self.log_path = log_path
        self.scale = scale
        self.record: dict[str, Any] = {}
        #: The policy package a daemon serves, its directory, as its starter
        #: wrote it (:meth:`uniserve_testing.deployments.Deployments.serve`);
        #: ``None`` for another server.
        self.package: Path | None = None
        #: The runtime directory a daemon runs under (``XDG_RUNTIME_DIR``),
        #: where it starts its deployment's servers; ``None`` for another
        #: server.
        self.runtime: Path | None = None
        #: The Build Server of a daemon's deployment, which its daemon
        #: started (:class:`uniserve_testing.builds.BuildServer`), once the
        #: daemon serves; ``None`` for another server.
        self.build_server: Any = None
        self._records = records
        self._pending = b""
        self._address = address

    def scaled(self, seconds: float) -> float:
        """`seconds` stretched by the run's time scale."""
        return seconds * self.scale

    def wait(
        self, state: str, timeout_s: float | None = None
    ) -> dict[str, Any]:
        """The record that states `state` (``listening`` or ``serving``),
        once it arrived.

        A server either states the record or ends, which closes its
        descriptor, so the wait needs no bound of its own: the case's own
        bound (pytest-timeout) covers a server that hangs. A caller that
        states `timeout_s` (scaled) holds the start to it instead.

        Raises:
            RuntimeError: With the server's log tail, when the server closed
                its descriptor first, as a server that ends does.
            TimeoutError: When no such record arrived within `timeout_s`.
        """  # noqa: D205
        deadline = (
            time.monotonic() + self.scaled(timeout_s)
            if timeout_s is not None
            else None
        )
        while self.record.get("state") != state:
            if b"\n" not in self._pending:
                remaining = (
                    deadline - time.monotonic()
                    if deadline is not None
                    else None
                )
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        f"the server was not {state} within "
                        f"{self.scaled(timeout_s or 0):.0f} s"
                        f"{log_tail(self.log_path)}"
                    )
                readable, _, _ = select.select(
                    [self._records], [], [], remaining
                )
                if not readable:
                    continue
                chunk = os.read(self._records, 65536)
                if not chunk:
                    raise RuntimeError(
                        f"the server ended before it was {state} (exit "
                        f"{self.process.poll()}){log_tail(self.log_path)}"
                    )
                self._pending += chunk
                continue
            line, self._pending = self._pending.split(b"\n", 1)
            record = json.loads(line)
            if not isinstance(record, dict) or "state" not in record:
                raise ValueError(f"no readiness record: {line!r}")
            self.record = record
        return self.record

    def wait_until(
        self,
        predicate: Callable[[], object],
        timeout_s: float,
        what: str,
        interval_s: float = 1.0,
    ) -> None:
        """Wait until `predicate` holds, reading it every `interval_s`
        seconds for at most `timeout_s` seconds (scaled), for a state the
        server announces no event of.

        Raises:
            TimeoutError: Naming `what`, with the server's log tail.
        """  # noqa: D205
        deadline = time.monotonic() + self.scaled(timeout_s)
        while not predicate():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{what} within {self.scaled(timeout_s):.0f} s"
                    f"{log_tail(self.log_path)}"
                )
            time.sleep(interval_s)

    def endpoint(self, name: str) -> str:
        """The endpoint `name` (``client``, ``admin``, ``join``, ``storage``,
        ``peer``, ``build``) the latest record states.
        """  # noqa: D205
        endpoint = self.record.get("endpoints", {}).get(name)
        if endpoint is None:
            raise KeyError(f"the server's record states no {name} endpoint")
        return endpoint

    @property
    def base_url(self) -> str:
        """The base URL of the server's client API."""
        host, _, port = self.endpoint("client").rpartition(":")
        if self._address is not None and host in ("0.0.0.0", "[::]"):
            host = self._address
        return f"http://{host}:{port}"

    @property
    def admin_socket(self) -> Path:
        """The daemon's operator socket."""
        return socket_path(self.endpoint("admin"))

    @property
    def join(self) -> str:
        """The endpoint a deployment's other nodes join the leader at."""
        return self.endpoint("join")

    @property
    def pid(self) -> int:
        """The server's process, as its record states it."""
        return int(self.record["pid"])

    def stop(self) -> None:
        """Stop the server's process group and close the record descriptor
        (:func:`stop_process`).
        """  # noqa: D205
        stop_process(self.process)
        self.close()

    def close(self) -> None:
        """Close this process's end of the server's readiness descriptor."""
        if self._records >= 0:
            os.close(self._records)
            self._records = -1


def stop_process(process: subprocess.Popen[bytes]) -> None:
    """End `process`'s session (it leads one, :func:`launch`) with SIGTERM
    and wait for it.

    A UniServe server ends on SIGTERM within its own shutdown bound
    (``server_config.shutdown_timeout`` for a daemon), so the wait needs no
    bound of its own: the case's bound (pytest-timeout) covers one that
    hangs.
    """  # noqa: D205
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    process.wait()


def launch(
    argv: list[str],
    log_path: Path,
    env: Mapping[str, str],
    *,
    scale: float = 1.0,
) -> Server:
    """Start the server `argv` in a session of its own, with the environment
    `env` (:func:`environment`), its output to `log_path`.

    The server is passed the write end of a pipe as ``--ready-fd``, whose
    records the returned :class:`Server` reads.
    """  # noqa: D205
    records, ready = os.pipe()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [*argv, "--ready-fd", str(ready)],
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(ready,),
            )
    finally:
        # The child holds the write end now: the read end sees the end of
        # file once the child closed it or ended.
        os.close(ready)
    return Server(process, records, log_path, scale=scale)
