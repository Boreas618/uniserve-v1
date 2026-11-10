from __future__ import annotations

import base64
import io
import json
import os
import shlex
import shutil
import signal
import socket
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

# The host KV cache a test's epoch holds unless the test sizes it itself.
# Every test runs its own Storage Server, which budgets the host's available
# storage as if it were the host's only one, so concurrent tests sized by
# `kv_cache_config.host_ratio` alone would together commit more host storage
# than the host has.
HOST_CACHE_GIB = 16

#: The attention program the suite's Qwen3 deployments serve: trtllm-gen, the
#: program these deployments' shapes take on Blackwell.
# Explicit until the Program Server selects a program left at `auto` (W4).
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


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def worker_executable() -> Path:
    """The `uniserve` executable a test server's workers run as.

    `UNISERVE_WORKER_EXECUTABLE` names it, such as the sanitizer's wrapper;
    otherwise the workers run the binary the daemon runs
    (:func:`require_uniserve_binary`), so a server never mixes two builds.
    """
    configured = os.environ.get("UNISERVE_WORKER_EXECUTABLE")
    if configured:
        return Path(configured)
    return require_uniserve_binary()


def wait_until(
    predicate: Callable[[], object],
    timeout_s: float,
    what: str,
    interval_s: float = 1.0,
) -> None:
    """Wait until `predicate` holds, for at most `timeout_s` seconds stretched
    by `TIME_SCALE`; raise :class:`TimeoutError` naming `what` otherwise.
    """  # noqa: D205
    deadline = time.monotonic() + scaled(timeout_s)
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval_s)
    raise TimeoutError(f"{what} within {scaled(timeout_s):.0f} s")


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


#: The Program Server every deployment that names none is given
#: (:func:`programs_from`).
_PROGRAM_SERVER: Path | None = None


@contextmanager
def programs_from(server: Path) -> Iterator[None]:
    """Give every deployment started in the block the Program Server whose
    socket is `server`, unless it names one itself (``program_config.socket``).
    """  # noqa: D205
    global _PROGRAM_SERVER
    previous, _PROGRAM_SERVER = _PROGRAM_SERVER, server
    try:
        yield
    finally:
        _PROGRAM_SERVER = previous


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


def wait_health(
    base_url: str,
    process: subprocess.Popen[str],
    timeout_s: float,
    log_path: Path | None = None,
) -> None:
    deadline = time.monotonic() + scaled(timeout_s)
    last_error: str | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited early with code {process.returncode}"
                f"{log_tail(log_path)}"
            )
        try:
            response = httpx.get(f"{base_url}/health", timeout=2.0)
            if response.status_code == 200:
                return
            last_error = (
                f"health returned {response.status_code}: {response.text[:200]}"
            )
        except Exception as error:  # noqa: BLE001 - readiness polling reports the last failure.
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(1.0)
    raise TimeoutError(
        f"server did not become healthy within {timeout_s}s; last error: "
        f"{last_error}{log_tail(log_path)}"
    )


def _serves_a_deployment(args: list[str]) -> bool:
    """Whether `args` starts a deployment: the `serve` subcommand of the one
    executable.

    The subcommand is the second word. `uniserve program-server serve` and
    `uniserve storage-server serve` are other commands: `serve` is theirs too,
    and they take no deployment options.
    """  # noqa: D205
    return len(args) > 1 and args[1] == "serve"


def start_server(
    args: list[str], log_path: Path, env: dict[str, str] | None = None
) -> subprocess.Popen[str]:
    """Start `args` in a session of its own, its output to `log_path`.

    A deployment that sizes no host KV cache (``kv_cache_config.host_*``) holds
    `HOST_CACHE_GIB` of it; one that names no Program Server is given the
    module's (:func:`programs_from`); and every deployment adds `E2E_SET`.
    """
    if _serves_a_deployment(args):
        named = any("program_config.socket" in arg for arg in args)
        if not named and _PROGRAM_SERVER is not None:
            args = [
                *args,
                "--set",
                f"program_config.socket={_PROGRAM_SERVER}",
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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.Popen(
        args,
        cwd=Path.cwd(),
        env=merged_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def stop_storage_server(runtime: Path) -> None:
    """Stop the Storage Server a server started under the runtime directory
    `runtime` (its ``XDG_RUNTIME_DIR``), found by its socket on its command
    line: it outlives the server, and holds the deployment's host and
    device storage until it stops.
    """  # noqa: D205
    socket = str(runtime / "uniserve" / "storage-server.sock").encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"serve" in command and socket in command:
            os.kill(int(entry.name), signal.SIGTERM)


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


# Stops the Storage Server serving the socket given as its argument, found by
# its command line, as `stop_storage_server` does on this host.
_STOP_STORAGE = """
import os
import signal
import sys

for entry in os.listdir("/proc"):
    if not entry.isdigit():
        continue
    try:
        command = open(f"/proc/{entry}/cmdline", "rb").read().split(b"\\0")
    except OSError:
        continue
    if b"serve" in command and sys.argv[1].encode() in command:
        os.kill(int(entry), signal.SIGTERM)
"""


def _on_host(host: str, command: str, timeout: float = 120) -> str:
    """Runs `command` on `host` over ssh; returns its output."""
    done = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, command],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return done.stdout


@contextmanager
def second_host(
    host: str,
    log: Path,
    address: str,
    *,
    devices: str | None = None,
    env: dict[str, str] | None = None,
    arguments: tuple[str, ...] = (),
) -> Iterator[None]:
    """The daemon of a deployment's second host, joining the leader at
    `address` as node `rank-1`, with `arguments` after its own options.

    `host` is an ssh destination whose checkout path holds this build. The
    daemon runs in this test's environment (its search and package paths,
    native libraries, kernel build settings and the network devices UCX
    may use), with `env` beside it and on `devices` (all of the host's
    without them), under a runtime directory of its own: the Storage Server
    it starts serves this deployment alone, in that environment. It starts
    when the block begins: a member retries its join until the leader binds
    `address`, which the leader does first when it starts, within the join
    timeout (300 s), so the caller starts the leader in the block. On exit,
    which follows the leader's, it waits until the daemon ended, which it
    does when the leader's connection closes, keeps the Storage Server's
    log beside `log` and stops that server. A daemon that outlives its
    leader fails the test.
    """  # noqa: D205
    runtime = os.path.join(
        tempfile.gettempdir(),
        f"uniserve-follower-{os.getpid()}-{find_free_port()}",
    )
    variables = {
        name: os.environ.get(name, "")
        for name in (
            "PATH",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "MAX_JOBS",
            "UNISERVE_PROGRAM_SERVER_PYTHON",
            "FLASHINFER_WORKSPACE_BASE",
            "UCX_NET_DEVICES",
        )
    }
    variables.update(env or {})
    variables["XDG_RUNTIME_DIR"] = runtime
    if devices is not None:
        variables["CUDA_VISIBLE_DEVICES"] = devices
    environment = " ".join(
        f"{name}={shlex.quote(value)}" for name, value in variables.items()
    )
    launcher = os.environ.get("UNISERVE_FOLLOWER_LAUNCHER")
    command_prefix = f"{shlex.quote(launcher)} " if launcher else ""
    remote = (
        f"mkdir -m 700 {runtime} && cd {shlex.quote(str(Path.cwd()))} && "
        f"echo follower pid $$ && exec env {environment} "
        f"{command_prefix}{shlex.quote(str(require_uniserve_binary()))} "
        "serve --nnodes 2 "
        f"--node-rank 1 --dist-init-addr {address} --host-identity rank-1 "
        f"--worker-executable {shlex.quote(str(worker_executable()))}"
    )
    if arguments:
        remote += " " + shlex.join(arguments)
    process = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", host, remote],
        stdout=log.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        yield
    finally:
        outlived = False
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            # The remote shell echoed its pid before it became the daemon.
            pid = log.read_text().split("follower pid ")[1].split()[0]
            _on_host(host, f"kill {pid}")
            process.wait(timeout=120)
            outlived = True
        storage_socket = f"{runtime}/uniserve/storage-server.sock"
        kept = shlex.quote(str(log.with_suffix(".storage.log")))
        _on_host(
            host,
            f"python3 -c {shlex.quote(_STOP_STORAGE)} {storage_socket}; "
            f"cp {runtime}/uniserve/storage-server.log {kept} 2>/dev/null; "
            f"rm -rf {runtime}",
        )
        if outlived:
            raise AssertionError("the second host's daemon outlived its leader")


@contextmanager
def server_process(
    args: list[str],
    base_url: str,
    log_path: Path,
    timeout_s: float,
    env: dict[str, str] | None = None,
) -> Iterator[subprocess.Popen[str]]:
    """Serve `args` until the block ends, once its health answers.

    Unless `env` names them, the server runs on the GPUs the GPU locks
    granted (``UNISERVE_WORLD_DEVICES``) and under a runtime directory of
    its own (``XDG_RUNTIME_DIR``), whose Storage Server, which outlives the
    server with the deployment's host and device storage, is stopped at the
    end. A caller that names its runtime directory owns that Storage Server.
    """
    env = dict(env or {})
    granted = os.environ.get("UNISERVE_WORLD_DEVICES")
    if granted and "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = granted
    with ExitStack() as owned:
        if "XDG_RUNTIME_DIR" not in env:
            env["XDG_RUNTIME_DIR"] = str(
                owned.enter_context(runtime_directory())
            )
        process = start_server(args, log_path, env)
        try:
            wait_health(base_url, process, timeout_s, log_path)
            yield process
        finally:
            stop_server(process)


def admin(
    base_url: str, path: str, body: dict[str, Any] | None = None
) -> httpx.Response:
    """``POST /admin/<path>`` with `body`, answered once the operation ends."""
    return httpx.post(
        f"{base_url}/admin/{path}", json=body or {}, timeout=scaled(1800)
    )


def admin_status(base_url: str) -> dict[str, Any]:
    """``GET /admin/status``: what the deployment serves and its operations."""
    response = httpx.get(f"{base_url}/admin/status", timeout=10)
    response.raise_for_status()
    return response.json()


def admin_config(base_url: str) -> dict[str, Any]:
    """``GET /admin/config``: the configuration, the epoch that executes
    (``executing``), the update in flight or the last one (``update``), the
    epoch records (``epochs``), the node's processes (``processes``) and the
    storage ledgers (``ledgers``).
    """  # noqa: D205
    response = httpx.get(f"{base_url}/admin/config", timeout=60)
    response.raise_for_status()
    return response.json()


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
    """Write the deployment file a serve invocation is given.

    `--worker-groups` names a TOML file because a deployment states every
    worker group's ranks and the components placed on them; a test states
    one the same way a deployment does. Each entry of `groups` is one
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
