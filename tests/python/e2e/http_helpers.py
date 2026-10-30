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
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from uniserve_eval.transport.sse import iter_sse_events

# The host KV cache a test's world holds unless the test sizes it itself.
# Every test runs its own Storage Server, which budgets the host's available
# storage as if it were the host's only one, so concurrent tests sized by
# `kv_cache_config.host_ratio` alone would together commit more host storage
# than the host has.
HOST_CACHE_GIB = 16

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
# tokens through an expert transport: two worker groups of one expert world.
DEDICATED_EXPERTS = """
[[worker_groups]]
id = "attention"
ranks = ["cuda:0"]
queue_depth = 2

[worker_groups.components.model]

[[worker_groups]]
id = "experts"
role = "experts"
ranks = ["cuda:1"]
queue_depth = 2
"""


def dedicated_experts(directory: Path) -> Path:
    """Write :data:`DEDICATED_EXPERTS` under ``directory``; return its path."""
    placement = directory / "attention_experts.toml"
    placement.write_text(DEDICATED_EXPERTS)
    return placement


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


def gpu_device() -> str:
    """The GPU a one-GPU server runs on (``UNISERVE_WORLD_DEVICE``, 0 by
    default), so that one-GPU cases run beside each other on other GPUs.
    """  # noqa: D205
    return os.environ.get("UNISERVE_WORLD_DEVICE", "0")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def require_uniserve_binary() -> Path:
    configured = os.environ.get("UNISERVE_BINARY")
    path = (
        Path(configured)
        if configured
        else Path.cwd() / "target" / "debug" / "uniserve"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run `cargo build -p uniserve` first"
        )
    return path


def worker_executable() -> Path:
    """The `uniserve` executable a test server's workers run as.

    `UNISERVE_WORKER_EXECUTABLE` names it; otherwise it is the one installed
    beside the interpreter running the tests, which runs the workers in that
    environment.
    """
    configured = os.environ.get("UNISERVE_WORKER_EXECUTABLE")
    if configured:
        return Path(configured)
    return Path(sys.executable).parent / "uniserve"


def wait_health(
    base_url: str, process: subprocess.Popen[str], timeout_s: float
) -> None:
    deadline = time.monotonic() + scaled(timeout_s)
    last_error: str | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited early with code {process.returncode}"
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
        f"{last_error}"
    )


def start_server(
    args: list[str], log_path: Path, env: dict[str, str] | None = None
) -> subprocess.Popen[str]:
    """Start `args` in a session of its own, its output to `log_path`.

    A `serve` command that sizes no host KV cache
    (``kv_cache_config.host_*``) holds `HOST_CACHE_GIB` of it, and every
    `serve` command adds `E2E_SET`.
    """
    sized = any("kv_cache_config.host_" in arg for arg in args)
    if "serve" in args and not sized:
        args = [*args, "--set", f"kv_cache_config.host_gib={HOST_CACHE_GIB}"]
    if "serve" in args:
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
    socket = str(runtime / "uniserve" / "storage.sock").encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"serve" in command and socket in command:
            os.kill(int(entry.name), signal.SIGTERM)


# What a deployment's leader logs while it awaits the nodes below `--nnodes`.
AWAITING_NODES = "awaiting the deployment's other nodes"

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
    leader_log: Path | None = None,
) -> Iterator[Callable[[], None]]:
    """The daemon of a deployment's second host, joining the leader at
    `address` as node `rank-1`.

    `host` is an ssh destination whose checkout path holds this build. The
    daemon runs in this test's environment (its search and package paths,
    native libraries, kernel build settings and the network devices UCX
    may use), with `env` beside it and on `devices` (all of the host's
    without them), under a runtime directory of its own: the Storage Server
    it starts serves this deployment alone, in that environment. It starts
    once `leader_log` shows the leader awaits it, or when the test calls the
    callable this yields. On exit, which follows the leader's, it waits
    until the daemon ended, which it does when the leader's connection
    closes, keeps the Storage Server's log beside `log` and stops that
    server. A daemon that outlives its leader fails the test.
    """  # noqa: D205
    runtime = f"/tmp/uniserve-follower-{os.getpid()}-{find_free_port()}"
    variables = {
        name: os.environ.get(name, "")
        for name in (
            "PATH",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "MAX_JOBS",
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
    started: list[subprocess.Popen[str]] = []
    leaving = threading.Event()

    def start() -> None:
        if not started:
            started.append(
                subprocess.Popen(
                    ["ssh", "-o", "BatchMode=yes", host, remote],
                    stdout=log.open("w", encoding="utf-8"),
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )

    def start_when_awaited() -> None:
        while not leaving.is_set():
            if (
                leader_log is not None
                and leader_log.exists()
                and AWAITING_NODES in leader_log.read_text(errors="replace")
            ):
                start()
                return
            time.sleep(1)

    starter = threading.Thread(target=start_when_awaited, daemon=True)
    if leader_log is not None:
        starter.start()
    try:
        yield start
    finally:
        leaving.set()
        if leader_log is not None:
            starter.join()
        outlived = False
        for process in started:
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                pid = log.read_text().split("follower pid ")[1].split()[0]
                _on_host(host, f"kill {pid}")
                process.wait(timeout=120)
                outlived = True
        if started:
            storage_socket = f"{runtime}/uniserve/storage.sock"
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
    runtime = None
    if "XDG_RUNTIME_DIR" not in env:
        runtime = Path(tempfile.mkdtemp(prefix="uniserve-"))
        env["XDG_RUNTIME_DIR"] = str(runtime)
    granted = os.environ.get("UNISERVE_WORLD_DEVICES")
    if granted and "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = granted
    process = start_server(args, log_path, env)
    try:
        wait_health(base_url, process, timeout_s)
        yield process
    finally:
        stop_server(process)
        if runtime is not None:
            stop_storage_server(runtime)
            shutil.rmtree(runtime, ignore_errors=True)


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
    seconds: float,
    seed: int,
    *,
    aspect_ratio: str = "16:9",
) -> dict[str, Any]:
    """A MiniMax-H3 text-to-video-and-audio request at a named canvas."""
    return {
        "model": model,
        "prompt": prompt,
        "task": "t2va",
        "target": {
            "short_edge": 768,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": seconds,
        },
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
    ["cuda:0-1"], "queue_depth": 2, "components": {"model":
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
