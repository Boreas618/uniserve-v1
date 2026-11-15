from __future__ import annotations

import base64
import io
import json
import os
import select
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from uniserve_eval.transport.sse import iter_sse_events

# The host tier a test's deployment holds unless the test sizes it itself:
# the host regions of its buffer pool. Every test runs its own Storage
# Server, which budgets the host's available storage as if it were the host's
# only one, so concurrent tests sized by `kv_cache_config.host_ratio` alone
# would together commit more host storage than the host has.
HOST_CACHE_GIB = 16

#: The attention package the suite's Qwen3 deployments serve: trtllm-gen, the
#: package these deployments' shapes take on Blackwell.
# Explicit until the Build Server selects a package left at `auto` (W4).
ATTENTION_BACKEND = "trtllm"

#: What ``UNISERVE_QWEN3_MODEL`` names.
QWEN3_CHECKPOINT = "a Qwen3-MoE BF16 checkpoint directory (Qwen3-30B-A3B)"

#: What ``UNISERVE_QWEN3_UPDATED_MODEL`` names.
QWEN3_UPDATED_CHECKPOINT = (
    "another copy of the Qwen3-MoE checkpoint's files with other weights"
)

#: What ``UNISERVE_H3_MODEL`` names.
H3_CHECKPOINT = (
    "a FastH3 checkpoint directory (FastVideo-FastH3-8-Step-V2; "
    "docs/minimax_h3/minimax_h3.md lists the supported ones)"
)

#: What ``UNISERVE_MINIMAX_H3_MODEL`` names.
MINIMAX_H3_CHECKPOINT = "the MiniMax-H3 diffusers root"

#: What ``UNISERVE_FASTH3_ADAPTER`` names.
FASTH3_ADAPTER = (
    "a FastH3 Preview v1 adapter (FastVideo-FastH3-4-step-Preview-v1-LoRA's "
    "vsa-datafree or dense-datafree directory)"
)

# Options every test server adds to its own, as `PATH=VALUE` assignments
# separated by whitespace (``UNISERVE_E2E_SET``), and the factor every wait
# of a test is stretched by (``UNISERVE_E2E_TIME_SCALE``): a run whose
# workers run under a tool, such as compute-sanitizer, leaves the tool
# device storage and lets the workers' jobs and the tests' waits take
# longer.
E2E_SET = os.environ.get("UNISERVE_E2E_SET", "").split()
TIME_SCALE = float(os.environ.get("UNISERVE_E2E_TIME_SCALE", "1"))


def scaled(seconds: float) -> float:
    """`seconds` stretched by `TIME_SCALE`."""
    return seconds * TIME_SCALE


# Attention on the first GPU and the routed experts on the second, exchanging
# tokens through an expert transport: two worker groups of one expert epoch.
DEDICATED_EXPERTS = """
[[worker_groups]]
id = "attention"
ranks = ["cuda:0"]
queue_depth = 2

[worker_groups.blocks.model]

[[worker_groups]]
id = "experts"
role = "experts"
ranks = ["cuda:1"]
queue_depth = 2
"""


def dedicated_experts(directory: Path) -> Path:
    """Write :data:`DEDICATED_EXPERTS` under ``directory``; return its path."""
    rank_layout = directory / "attention_experts.toml"
    rank_layout.write_text(DEDICATED_EXPERTS)
    return rank_layout


def granted_devices(count: int) -> str:
    """The first `count` GPUs the GPU locks granted
    (``UNISERVE_WORLD_DEVICES``), comma-separated; the first `count` GPUs
    when no lock granted any.
    """  # noqa: D205
    granted = os.environ.get("UNISERVE_WORLD_DEVICES", "")
    devices = (
        granted.split(",")
        if granted
        else [str(index) for index in range(count)]
    )
    if len(devices) < count:
        raise RuntimeError(f"{count} GPUs are needed, {granted!r} were granted")
    return ",".join(devices[:count])


def require_uniserve_binary() -> Path:
    """The `uniserve` binary the tests serve with.

    `UNISERVE_BINARY` names it.
    """
    configured = os.environ.get("UNISERVE_BINARY")
    if not configured:
        raise FileNotFoundError(
            "UNISERVE_BINARY must name the uniserve binary; "
            "`cargo build -p uniserve` builds one"
        )
    path = Path(configured)
    if not path.exists():
        raise FileNotFoundError(
            f"UNISERVE_BINARY names {path}, which does not exist"
        )
    return path


def engine_selection() -> list[str]:
    """The `serve` arguments that select a test server's engine package.

    The tree's own engine package (`uniserve`, `crates/engine`) is the
    default and needs no argument: every test's Build Server holds it.
    `UNISERVE_ENGINE` names another installed engine package, such as the
    sanitizer's wrapper (`UNISERVE_ENGINE_PACKAGE` is its directory, which
    the test's Build Server installs).
    """
    configured = os.environ.get("UNISERVE_ENGINE")
    if configured:
        return ["--set", f'worker_config.engine="{configured}"']
    return []


def build_server_socket() -> Path:
    """The socket of the Build Server the module's deployments use."""
    assert _BUILD_SERVER is not None, "the module started no Build Server"
    return _BUILD_SERVER


def engine_revision() -> str:
    """An assignment that selects a new revision of the tree's engine.

    The revision is installed and built now at the module's Build Server. It
    runs the same code as the serving one under another revision, which is a
    change of the epoch's engine and of nothing else, and an update to it
    builds nothing while it prepares: a request started after this still
    streams when the switch comes.
    """
    socket = build_server_socket()
    revision = install_engine(socket)
    engine_executable(socket, revision)
    return f"worker_config.engine_revision={revision}"


def engine_executable(socket: Path, revision: int) -> Path:
    """The executable of the tree's engine package at `revision`.

    The Build Server builds the module when its store misses it, and names
    the file in its store.
    """
    resolved = subprocess.run(
        [
            str(require_uniserve_binary()),
            "build-server",
            "resolve",
            "--socket",
            str(socket),
            "engine",
            "uniserve",
            str(revision),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(resolved.stdout.splitlines()[0])


def install_engine(socket: Path, package: Path | None = None) -> int:
    """Installs an engine package at a Build Server and returns its revision.

    Args:
        socket: The Build Server's socket.
        package: The package directory; the tree's own engine package when
            omitted. A new revision of an unchanged package runs the same
            code, which is what a test applies
            (`worker_config.engine_revision`) to change the epoch's engine
            and nothing else.

    Returns:
        The revision the Build Server numbered the installation.
    """
    directory = package or Path.cwd() / "crates/engine"
    installed = subprocess.run(
        [
            str(require_uniserve_binary()),
            "build-server",
            "install",
            "--socket",
            str(socket),
            str(directory),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # `engine <backend> revision <n>`.
    return int(installed.stdout.split()[-1])


def wait_until(
    predicate: Callable[[], object],
    timeout_s: float,
    what: str,
    interval_s: float = 1.0,
    context: Callable[[], str] | None = None,
) -> None:
    """Wait until `predicate` holds, for at most `timeout_s` seconds stretched
    by `TIME_SCALE`; raise :class:`TimeoutError` naming `what` otherwise,
    followed by what `context` returns then, such as a server's log tail.
    """  # noqa: D205
    deadline = time.monotonic() + scaled(timeout_s)
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval_s)
    detail = context() if context is not None else ""
    raise TimeoutError(f"{what} within {scaled(timeout_s):.0f} s{detail}")


def runs(pid: int) -> bool:
    """Whether process `pid` runs.

    A child of the test that exited and that no one waited for yet (a
    zombie) still has its entry in ``/proc``; it does not run.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    # The state follows the parenthesized command name.
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def host_asset(variable: str, what: str) -> Path:
    """The path of a host asset that `variable` names.

    Model checkpoints and input sets are facts of the host, which its
    environment states. A case whose asset is unset is skipped, naming the
    variable and what it must name, and never passes; a variable that names a
    path that does not exist is a misconfigured run and fails.
    """
    value = os.environ.get(variable, "")
    if not value:
        pytest.skip(f"{variable} is unset: it names {what}")
    path = Path(value)
    if not path.exists():
        pytest.fail(f"{variable} names {path}, which does not exist: {what}")
    return path


def qwen3_checkpoint() -> str:
    """The Qwen3-MoE checkpoint ``UNISERVE_QWEN3_MODEL`` names."""
    return str(host_asset("UNISERVE_QWEN3_MODEL", QWEN3_CHECKPOINT))


#: The Build Server every deployment that names none is given
#: (:func:`modules_from`).
_BUILD_SERVER: Path | None = None


@contextmanager
def modules_from(server: Path) -> Iterator[None]:
    """Give every deployment started in the block the Build Server whose
    socket is `server`, unless it names one itself (``build_config.socket``).
    """  # noqa: D205
    global _BUILD_SERVER
    previous, _BUILD_SERVER = _BUILD_SERVER, server
    try:
        yield
    finally:
        _BUILD_SERVER = previous


def log_tail(log_path: Path | None, lines: int = 40) -> str:
    """The last `lines` lines of a server's log, marked, for a failure message.

    The log lives on the host that ran the server, so a failure whose message
    does not carry it costs a round trip to read; the tail is what usually names
    the cause. A missing or unreadable file says so rather than failing.
    """  # noqa: D205
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


def _serves_a_deployment(args: list[str]) -> bool:
    """Whether `args` starts a deployment: the `serve` subcommand of the one
    executable.

    The subcommand is the second word. `uniserve build-server serve` and
    `uniserve storage-server serve` are other commands: `serve` is theirs too,
    and they take no deployment options.
    """  # noqa: D205
    return len(args) > 1 and args[1] == "serve"


def _reports_readiness(args: list[str]) -> bool:
    """Whether `args` starts a server that reports its readiness record on
    the descriptor `--ready-fd` names: a deployment's daemon, a Build Server
    or a Storage Server.
    """  # noqa: D205
    return _serves_a_deployment(args) or args[1:3] in (
        ["build-server", "serve"],
        ["storage-server", "serve"],
    )


#: The admin socket of every daemon this process started, by the base URL of
#: its client API, as the daemon's readiness record states both
#: (:func:`admin_socket`).
_ADMIN_SOCKETS: dict[str, Path] = {}


def _socket_path(endpoint: str) -> Path:
    """The path of the Unix socket a record's endpoint (``unix:PATH``) names."""
    if not endpoint.startswith("unix:"):
        raise ValueError(f"{endpoint} is no Unix socket")
    return Path(endpoint.removeprefix("unix:"))


class Server:
    """A server this process started, and what its readiness records state.

    A server writes one JSON line per state to the descriptor its caller
    passes (``--ready-fd``, ``docs/admin.md``, Listeners): ``listening`` once
    its listeners are bound, with their endpoints, and ``serving`` once it
    serves; then it closes the descriptor. :meth:`wait` reads the records as
    they arrive, so a caller neither chooses a port nor probes one.
    """

    def __init__(
        self,
        process: subprocess.Popen[str],
        records: int,
        log_path: Path | None,
        *,
        timeout_s: float = 0.0,
        address: str | None = None,
        remote: bool = False,
    ) -> None:
        """A server run as `process`, its records arriving on the descriptor
        `records`, its log at `log_path`.

        `timeout_s` is the longest its start may take, for a caller that
        waits on it later (:func:`second_host`). `address` replaces an
        unspecified host (``0.0.0.0``) of its client endpoint in
        :attr:`base_url`. A `remote` server's records cross an ssh
        connection's standard error, where ssh's own warnings may precede
        them, so a line that is no record is skipped there.
        """  # noqa: D205
        self.process = process
        self.log_path = log_path
        self.timeout_s = timeout_s
        self.record: dict[str, Any] = {}
        self._records = records
        self._pending = b""
        self._address = address
        self._remote = remote

    def wait(self, state: str, timeout_s: float) -> dict[str, Any]:
        """The record that states `state` (``listening`` or ``serving``),
        once it arrived within `timeout_s` seconds stretched by `TIME_SCALE`.

        Raises :class:`RuntimeError`, with the server's log tail, when the
        server closed its descriptor first, as a server that ends does, and
        :class:`TimeoutError` when no such record arrived in time.
        """  # noqa: D205
        deadline = time.monotonic() + scaled(timeout_s)
        while self.record.get("state") != state:
            if b"\n" not in self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"the server was not {state} within "
                        f"{scaled(timeout_s):.0f} s{log_tail(self.log_path)}"
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
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if self._remote:
                    continue
                raise
            if not isinstance(record, dict) or "state" not in record:
                if self._remote:
                    continue
                raise ValueError(f"no readiness record: {line!r}")
            self.record = record
            admin = record["endpoints"].get("admin")
            if record["endpoints"].get("client") and admin:
                _ADMIN_SOCKETS[self.base_url] = _socket_path(admin)
        return self.record

    def endpoint(self, name: str) -> str:
        """The endpoint `name` (``client``, ``admin``, ``join``, ``peer``,
        ...) the latest record states.
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
        return _socket_path(self.endpoint("admin"))

    @property
    def join(self) -> str:
        """The endpoint a deployment's other nodes join the leader at."""
        return self.endpoint("join")

    @property
    def pid(self) -> int:
        """The server's process, as its record states it."""
        return int(self.record["pid"])

    def close(self) -> None:
        """Closes this process's end of the server's readiness descriptor."""
        if self._records >= 0:
            os.close(self._records)
            self._records = -1


def start_server(
    args: list[str], log_path: Path, env: dict[str, str] | None = None
) -> Server:
    """Start `args` in a session of its own, its output to `log_path`.

    A server that reports its readiness (:func:`_reports_readiness`) is
    passed the write end of a pipe (``--ready-fd``), whose records the
    returned :class:`Server` reads. A deployment that sizes no host tier
    (``kv_cache_config.host_*``) holds `HOST_CACHE_GIB` of it; one that names
    no Build Server is given the module's (:func:`modules_from`); and every
    deployment adds `E2E_SET`.
    """
    if _serves_a_deployment(args):
        named = any("build_config.socket" in arg for arg in args)
        if not named and _BUILD_SERVER is not None:
            args = [
                *args,
                "--set",
                f'build_config.socket="{_BUILD_SERVER}"',
            ]
        sized = any("kv_cache_config.host_" in arg for arg in args)
        if not sized:
            args = [
                *args,
                "--set",
                f"kv_cache_config.host_gib={HOST_CACHE_GIB}",
            ]
        for assignment in E2E_SET:
            args = [*args, "--set", assignment]
    records, ready = os.pipe()
    passed: tuple[int, ...] = ()
    if _reports_readiness(args):
        args = [*args, "--ready-fd", str(ready)]
        passed = (ready,)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        process = subprocess.Popen(
            args,
            cwd=Path.cwd(),
            env=merged_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            pass_fds=passed,
        )
    finally:
        # The child holds the write end now: the read end sees the end of
        # file once the child closed it or ended.
        os.close(ready)
    return Server(process, records, log_path)


def stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def storage_socket(runtime: Path | str) -> str:
    """The socket of the Storage Server a server under the runtime directory
    `runtime` (its ``XDG_RUNTIME_DIR``) uses.
    """  # noqa: D205
    return f"{runtime}/uniserve/storage-server.sock"


def storage_status(runtime: Path) -> dict[str, Any]:
    """The readiness record of the Storage Server serving under the runtime
    directory `runtime` (``storage-server status``): its process, its
    endpoints (``storage``, ``peer``) and its settings.
    """  # noqa: D205
    answered = subprocess.run(
        [
            str(require_uniserve_binary()),
            "storage-server",
            "status",
            "--socket",
            storage_socket(runtime),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(answered.stdout)


def stop_storage_server(runtime: Path) -> None:
    """Stop the Storage Server serving under the runtime directory `runtime`
    (its ``XDG_RUNTIME_DIR``), which outlives the deployment that started it
    and holds its host and device storage until it stops: its ``stop``
    request returns once it ended. A runtime directory whose server is gone
    holds no socket, and nothing is stopped.
    """  # noqa: D205
    socket = storage_socket(runtime)
    if not Path(socket).exists():
        return
    subprocess.run(
        [
            str(require_uniserve_binary()),
            "storage-server",
            "stop",
            "--socket",
            socket,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


@contextmanager
def runtime_directory(kept_log: Path | None = None) -> Iterator[Path]:
    """A runtime directory (``XDG_RUNTIME_DIR``) of the caller's own, whose
    Storage Server is stopped when the block ends.

    A server started under it starts its Storage Server there, which
    outlives the server; the block's end stops it, keeps its log at
    `kept_log` when one is named, and removes the directory. A socket path
    holds at most 107 bytes, so the directory lies in the system's short
    temporary directory.
    """  # noqa: D205
    runtime = Path(tempfile.mkdtemp(prefix="uniserve-"))
    try:
        yield runtime
    finally:
        stop_storage_server(runtime)
        log = runtime / "uniserve" / "storage-server.log"
        if kept_log is not None and log.is_file():
            shutil.copy(log, kept_log)
        shutil.rmtree(runtime, ignore_errors=True)


#: The variables a second host's processes take from this test's
#: environment: their search and package paths, native libraries, kernel
#: build settings and the network devices UCX may use.
REMOTE_VARIABLES = (
    "PATH",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "MAX_JOBS",
    "FLASHINFER_WORKSPACE_BASE",
    "UCX_NET_DEVICES",
)


def remote_environment(extra: dict[str, str] | None = None) -> str:
    """The ``NAME=VALUE`` words, shell-quoted, of :data:`REMOTE_VARIABLES`
    as this test has them, with `extra` beside them.
    """  # noqa: D205
    variables = {name: os.environ.get(name, "") for name in REMOTE_VARIABLES}
    variables.update(extra or {})
    return " ".join(
        f"{name}={shlex.quote(value)}" for name, value in variables.items()
    )


def _ssh(host: str) -> list[str]:
    """The ssh command that runs a command on `host`, with ssh's own
    messages limited to errors.
    """  # noqa: D205
    return ["ssh", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", host]


def on_host(host: str, command: str, timeout: float = 120) -> str:
    """Runs `command` on `host` over ssh; returns its output."""
    done = subprocess.run(
        [*_ssh(host), command],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return done.stdout


def remote_runtime_directory(host: str, label: str) -> str:
    """A new runtime directory (mode 0700) on `host`, named after `label`
    and created there by ``mktemp``.
    """  # noqa: D205
    return on_host(
        host, f"mktemp -d -t {shlex.quote(label)}.XXXXXXXXXX"
    ).strip()


def start_remote_server(
    host: str,
    words: list[str],
    log: Path,
    *,
    environment: str,
    timeout_s: float,
    address: str | None = None,
) -> Server:
    """Run the server `words` on the ssh destination `host`, in this
    checkout's path there and with the variables `environment` states, its
    output to `log` on this host.

    Its readiness records cross the ssh connection's standard error: the
    server writes them to its descriptor 3, which the remote shell points at
    that stream before its own output goes to the log. `address` is the
    host's address as this host reaches it, which replaces an unspecified
    host in the server's base URL.
    """  # noqa: D205
    remote = (
        f"cd {shlex.quote(str(Path.cwd()))} && exec env {environment} "
        f"{shlex.join(words)} --ready-fd 3 3>&2 2>&1"
    )
    process = subprocess.Popen(
        [*_ssh(host), remote],
        stdout=log.open("w", encoding="utf-8"),
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stderr is not None
    return Server(
        process,
        process.stderr.fileno(),
        log,
        timeout_s=timeout_s,
        address=address,
        remote=True,
    )


def stop_remote_storage_server(host: str, runtime: str, kept_log: Path) -> None:
    """Stop the Storage Server serving under the runtime directory `runtime`
    (its ``XDG_RUNTIME_DIR``) on the ssh destination `host` with its ``stop``
    request, keep its log at `kept_log` on this host, and remove the
    directory.

    The log travels back over the ssh connection: `kept_log` lies in this
    test's temporary directory, which exists on this host alone. A server
    that wrote no log leaves `kept_log` empty.
    """  # noqa: D205
    socket = shlex.quote(storage_socket(runtime))
    directory = shlex.quote(runtime)
    stop = shlex.join(
        [str(require_uniserve_binary()), "storage-server", "stop", "--socket"]
    )
    log = on_host(
        host,
        f"test -S {socket} && env {remote_environment()} {stop} {socket}; "
        f"cat {directory}/uniserve/storage-server.log 2>/dev/null; "
        f"rm -rf {directory}; true",
    )
    kept_log.write_text(log, encoding="utf-8")


class Follower:
    """The daemon of a deployment's second host (:func:`second_host`)."""

    def __init__(self, server: Server, runtime: str) -> None:
        self.server = server
        #: Its runtime directory there, holding its Storage Server's socket.
        self.runtime = runtime


@contextmanager
def second_host(
    host: str,
    log: Path,
    leader: Server,
    *,
    devices: str | None = None,
    env: dict[str, str] | None = None,
    arguments: tuple[str, ...] = (),
) -> Iterator[Follower]:
    """The daemon of a deployment's second host, joining `leader` as node
    `rank-1`, with `arguments` after its own options.

    `leader` is a leader started with ``serving=False``
    (:func:`server_process`): its listening record states the endpoint its
    members join at, and it serves once this member joined, which this
    waits for before the block begins. `host` is an ssh destination whose
    checkout path holds this build. The daemon runs in this test's
    environment (:data:`REMOTE_VARIABLES`), with `env` beside it and on
    `devices` (all of the host's without them), under a runtime directory
    of its own: the Storage Server it starts serves this deployment alone.
    On exit the leader stops first, the daemon ends when the leader's
    connection closes, which this waits for, and its Storage Server is
    stopped with its log kept beside `log`. A daemon that outlives its
    leader fails the test.
    """  # noqa: D205
    runtime = remote_runtime_directory(host, "uniserve-follower")
    variables = dict(env or {})
    variables["XDG_RUNTIME_DIR"] = runtime
    if devices is not None:
        variables["CUDA_VISIBLE_DEVICES"] = devices
    launcher = os.environ.get("UNISERVE_FOLLOWER_LAUNCHER")
    words = [
        *([launcher] if launcher else []),
        str(require_uniserve_binary()),
        "serve",
        "--nnodes",
        "2",
        "--node-rank",
        "1",
        "--dist-init-addr",
        leader.join,
        "--host-identity",
        "rank-1",
        *engine_selection(),
        *arguments,
    ]
    member = start_remote_server(
        host,
        words,
        log,
        environment=remote_environment(variables),
        timeout_s=leader.timeout_s,
    )
    try:
        member.wait("listening", leader.timeout_s)
        leader.wait("serving", leader.timeout_s)
        yield Follower(member, runtime)
    finally:
        stop_server(leader.process)
        outlived = False
        try:
            member.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            if member.record:
                on_host(host, f"kill {member.pid}")
            member.process.wait(timeout=120)
            outlived = True
        stop_remote_storage_server(
            host, runtime, log.with_suffix(".storage.log")
        )
        if outlived:
            raise AssertionError("the second host's daemon outlived its leader")


@contextmanager
def server_process(
    args: list[str],
    log_path: Path,
    timeout_s: float,
    env: dict[str, str] | None = None,
    *,
    serving: bool = True,
) -> Iterator[Server]:
    """Serve `args` until the block ends, once its readiness record states
    it serves; with `serving` unset, once it states its listeners, for a
    leader whose members join it next (:func:`second_host`).

    The server binds a port of its own choosing (``--port`` defaults to 0)
    and states it in its record (:attr:`Server.base_url`). Unless `env`
    names them, the server runs on the GPUs the GPU locks granted
    (``UNISERVE_WORLD_DEVICES``) and under a runtime directory of its own
    (``XDG_RUNTIME_DIR``), whose Storage Server, which outlives the server
    with the deployment's host and device storage, is stopped at the end. A
    caller that names its runtime directory owns that Storage Server.
    """  # noqa: D205
    env = dict(env or {})
    granted = os.environ.get("UNISERVE_WORLD_DEVICES")
    if granted and "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = granted
    with ExitStack() as owned:
        if "XDG_RUNTIME_DIR" not in env:
            env["XDG_RUNTIME_DIR"] = str(
                owned.enter_context(runtime_directory())
            )
        server = start_server(args, log_path, env)
        server.timeout_s = timeout_s
        try:
            server.wait("serving" if serving else "listening", timeout_s)
            yield server
        finally:
            stop_server(server.process)
            server.close()


def admin_socket(base_url: str) -> Path:
    """The operator socket of the daemon whose client API is `base_url`.

    A daemon serves its admin API apart from the client API, on a Unix
    socket of its own (``docs/admin.md``, Listeners), which its readiness
    record states beside its client endpoint; every daemon this process
    started is registered by its base URL once its record arrived. A hot
    restart's successor goes on with both, so the socket stays the
    daemon's.
    """
    try:
        return _ADMIN_SOCKETS[base_url]
    except KeyError:
        raise FileNotFoundError(
            f"no daemon this process started states the client API {base_url}"
        ) from None


def taken_over(base_url: str, pid: int, timeout_s: float = 120) -> str:
    """The base URL of the daemon `pid` that took the deployment serving at
    `base_url` over in a hot restart, as its readiness record states it.

    A successor goes on in its predecessor's instance directory and binds the
    admin sockets over its predecessor's, so `base_url`'s admin socket
    answers for the successor once it holds the deployment; its record
    (``GET /admin/status``, ``readiness``) states the client API it serves,
    a new one when the restart changed the listening address. The returned
    URL is registered with the same admin socket.
    """  # noqa: D205

    def readiness() -> dict[str, Any] | None:
        try:
            record = admin_status(base_url).get("readiness")
        except (httpx.HTTPError, OSError):
            return None
        return record if record and record["pid"] == pid else None

    wait_until(
        lambda: readiness() is not None,
        timeout_s,
        f"daemon {pid} did not take the deployment over",
    )
    record = readiness()
    assert record is not None
    host, _, port = record["endpoints"]["client"].rpartition(":")
    url = f"http://{host}:{port}"
    _ADMIN_SOCKETS[url] = admin_socket(base_url)
    return url


def admin_request(
    base_url: str,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout_s: float = 1800,
) -> httpx.Response:
    """`method` on the admin route `path` (``/admin/...``) of the daemon
    whose client API is `base_url`, through its operator socket, waiting at
    most `timeout_s` seconds stretched by `TIME_SCALE`.
    """  # noqa: D205
    transport = httpx.HTTPTransport(uds=str(admin_socket(base_url)))
    with httpx.Client(
        transport=transport,
        base_url="http://admin",
        timeout=scaled(timeout_s),
    ) as client:
        return client.request(method, path, json=body, params=params)


def admin(
    base_url: str, path: str, body: dict[str, Any] | None = None
) -> httpx.Response:
    """``POST /admin/<path>`` with `body`, answered once the operation ends."""
    return admin_request(base_url, "POST", f"/admin/{path}", body=body or {})


def admin_status(base_url: str) -> dict[str, Any]:
    """``GET /admin/status``: what the deployment serves and its operations."""
    response = admin_request(base_url, "GET", "/admin/status", timeout_s=10)
    response.raise_for_status()
    return response.json()


def admin_config(base_url: str) -> dict[str, Any]:
    """``GET /admin/config``: the configuration, the epoch that executes
    (``executing``), the update in flight or the last one (``update``), the
    epoch records (``epochs``), the node's processes (``processes``) and the
    quotas (``quotas``).
    """  # noqa: D205
    response = admin_request(base_url, "GET", "/admin/config", timeout_s=60)
    response.raise_for_status()
    return response.json()


def apply_config(
    base_url: str, *assignments: str, **fields: object
) -> httpx.Response:
    """``PUT /admin/config`` with `assignments` (``set``) and the request's
    other `fields` (``force``, ``in_flight``, ``dry_run``, ``deadline_s``,
    ``wait``): answered, for an update, once it reached its trial or ended
    before one (``docs/admin.md``, Updates).
    """  # noqa: D205
    return admin_request(
        base_url,
        "PUT",
        "/admin/config",
        body={"set": list(assignments), **fields},
    )


def update_report(base_url: str) -> dict[str, Any]:
    """The update in flight, or the one that ended last, as ``GET
    /admin/config`` reports it (``update``); empty before any.
    """  # noqa: D205
    return admin_config(base_url).get("update") or {}


def wait_for_phase(
    base_url: str, *phases: str, timeout_s: float = 900
) -> dict[str, Any]:
    """Waits until the update stands in one of `phases` (``preparing``,
    ``switching``, ``trial``, ``committed``, ``rolled_back``,
    ``abandoned``); returns its report.
    """  # noqa: D205
    wait_until(
        lambda: update_report(base_url).get("phase") in phases,
        timeout_s,
        f"the update did not reach {' or '.join(phases)}",
        context=lambda: f"\nupdate: {update_report(base_url)}",
    )
    return update_report(base_url)


def engine_processes(base_url: str) -> list[dict[str, Any]]:
    """The deployment's main processes on the leader's node: the rows of
    ``GET /admin/config``'s ``processes`` of group ``engine``, each with its
    ``pid``, launch ``generation`` and ``state``.
    """  # noqa: D205
    return [
        process
        for process in admin_config(base_url)["processes"]
        if process["group"] == "engine"
    ]


def serving_engine(base_url: str) -> dict[str, Any]:
    """The main process that serves: the row of :func:`engine_processes`
    whose launch generation ``GET /admin/status`` names.
    """  # noqa: D205
    generation = admin_status(base_url)["deployment"]["generation"]
    assert generation is not None, "no main process serves"
    (serving,) = [
        process
        for process in engine_processes(base_url)
        if process["generation"] == generation
    ]
    return serving


def metric(base_url: str, name: str, label: str | None = None) -> float:
    """The sum of metric `name` from ``/metrics`` (a counter's ``_total``
    samples) over its label sets, or over those holding `label`, such as
    ``mode="eager"``.
    """  # noqa: D205
    text = httpx.get(f"{base_url}/metrics", timeout=10).text
    total = 0.0
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        sample = line.split("{", 1)[0].split(" ", 1)[0]
        if sample not in (name, f"{name}_total"):
            continue
        if label is not None and label not in line:
            continue
        total += float(line.rsplit(" ", 1)[1])
    return total


#: The prefix cache's counters, in tokens (``uniserve:prefix_cache_<name>``):
#: the tokens admissions looked up (``queries``, a request's prompt less its
#: last token), the tokens of their hits (``hits``), those of the hits loaded
#: from the host tier (``host_hits``), and those of the host hits a prefetch
#: from the peers brought (``peer_hits``).
PREFIX_COUNTERS = ("queries", "hits", "host_hits", "peer_hits")

#: How long the statistics a scheduler reports once a second take to include
#: what came before: two report intervals.
REPORT_SETTLE_S = 2.0


def prefix_counters(base_url: str) -> dict[str, float]:
    """The prefix cache's counters of every epoch so far."""
    return {
        name: metric(base_url, f"uniserve:prefix_cache_{name}")
        for name in PREFIX_COUNTERS
    }


def tokens_found(
    base_url: str, ask: Callable[[], Any]
) -> tuple[Any, dict[str, float]]:
    """What `ask` returns, and how much each prefix cache counter grew over
    it: the tokens its admissions found cached, per tier and per source.

    A hit is counted when its admission's answers arrive, before the request
    computes, so the counters are read once the report holding the query
    arrived and another followed. The counters before `ask` are read the same
    way, after the reports of the admissions before it, so the growth counts
    `ask`'s admissions alone. `ask` must look its prompt up: a deployment
    without ``prefix_caching`` counts no query.
    """  # noqa: D205
    time.sleep(scaled(REPORT_SETTLE_S))
    before = prefix_counters(base_url)
    result = ask()
    wait_until(
        lambda: (
            metric(base_url, "uniserve:prefix_cache_queries")
            > before["queries"]
        ),
        30,
        "the request's prefix cache query was not reported",
    )
    time.sleep(scaled(REPORT_SETTLE_S))
    after = prefix_counters(base_url)
    return result, {name: after[name] - before[name] for name in after}


def post_sse(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float = 600.0,
) -> list[dict[str, Any]]:
    with httpx.stream(
        "POST", f"{base_url}{endpoint}", json=payload, timeout=timeout_s
    ) as response:
        response.raise_for_status()
        return list(iter_sse_events(response.iter_lines()))


def t2va_request(
    model: str,
    prompt: str,
    num_frames: int,
    seed: int,
    *,
    size: str = "1344x768",
) -> dict[str, Any]:
    """A text-to-video-and-audio request for an exact size and frame count.

    The body carries no references, which is how MiniMax-H3 tells a t2va
    request; ``num_frames`` must already be a count the model generates.
    """
    return {
        "model": model,
        "prompt": prompt,
        "size": size,
        "num_frames": num_frames,
        "seed": seed,
    }


def png_size_from_b64(pixels_png_b64: str) -> tuple[int, int]:
    image = Image.open(io.BytesIO(base64.b64decode(pixels_png_b64)))
    image.load()
    return image.width, image.height


def tiny_input_png_b64() -> str:
    image = Image.new("RGB", (32, 32), (32, 96, 180))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def written_deployment(directory: Path, groups: list[dict[str, Any]]) -> Path:
    """Write the deployment file a serve invocation is given with `--config`.

    A deployment states every worker group's ranks and the blocks placed
    on them in its configuration's `worker_groups`; a test states one the
    same way a deployment does. Each entry of `groups` is one
    `[[worker_groups]]` table, such as `{"id": "model", "ranks":
    ["cuda:0-1"], "queue_depth": 2, "blocks": {"model":
    {"tensor_parallel_size": 2}}}`.
    """
    path = directory / "deployment.toml"
    tables = (
        "[[worker_groups]]\n"
        + "".join(f"{key} = {_toml(item)}\n" for key, item in group.items())
        for group in groups
    )
    path.write_text("".join(tables))
    return path


def _toml(value: Any) -> str:
    """Write a value as TOML: inline tables and arrays, JSON-escaped strings."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        # A JSON string is a valid TOML basic string.
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml(element) for element in value) + "]"
    if isinstance(value, dict):
        entries = (f"{key} = {_toml(item)}" for key, item in value.items())
        return "{ " + ", ".join(entries) + " }"
    raise TypeError(f"no TOML form for {value!r}")
