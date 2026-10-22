"""A deployment's lifecycle and live updates through the admin API.

Qwen3 serves on one GPU of the host, with a Storage Server of the test's
own (``XDG_RUNTIME_DIR`` points the daemon at a directory of the test's,
and the server the daemon starts there is stopped at the end).

- A node (``uniserve node``) brings deployments up and down through its
  admin API without restarting: each deployment's page pools and weights
  belong to it, so bringing it down returns the host's shared memory and
  the GPU's memory to what they were before it, and a deployment whose
  page pools the host's available memory cannot hold is refused with the
  sizes that decided it.
- A serving deployment answers its admin API: new requests wait while
  generation is paused and are served once it continues; a flush revokes
  the cached prefix pages; scheduling limits within the batch the workers
  were started for change in place while a request streams, and a change
  of the decode batch sizes is a hot update under the same stream; a
  worker hook no plugin implements fails with that reason; the manager
  attaches to the node, follows its events and detaches while it serves.
- A plugin installed beside the deployment's environment is taken up by a
  hot update and dropped by the next one after its removal: the worker
  hook it registers answers from every worker in between, and fails before
  and after.
- A hot update while another process holds the device's free memory gives
  back the tail of the KV cache for the standby engine, recomputing the
  requests that held it, and the updated engine creates the tail again.
- A sleeping deployment gives back its KV cache and graph memory, holds
  new requests and recomputes the ones it ran once it wakes, and answers
  as before.
- A LoRA adapter loads into a slot the daemon assigns, through the plugin
  hooks of a plugin that holds the pool: requests naming it are served
  under its name, the engine a hot update starts loads it again, and once
  it is unloaded its name is no model's.
- A plugin's communication buffer is allocated before the KV cache is
  sized and counts against it: the cache takes as many units fewer as
  the buffer's bytes hold.
- A weight update from another checkpoint of the model rewrites the
  weights in place under a streaming request: the deployment then answers
  as a deployment started on that checkpoint does, and as before once it
  updates back. A plugin's weight source that streams the tensors of the
  other checkpoint does the same through the same layout conversion, and
  a hot update after it binds the weights it left.
  ``UNISERVE_QWEN3_UPDATED_MODEL`` names the other checkpoint.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
    start_server,
    stop_server,
    worker_executable,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

# Options every deployment of the tests serves with: few captured graphs, so
# that engines start in about a minute, and a small device KV cache.
OPTIONS = (
    "dtype=bfloat16",
    "kv_cache_config.max_tokens=32768",
    "chat_config.reasoning_parser=none",
    "cuda_graph_config.prefill_token_sizes=[512]",
    "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
)
# Each worker's host KV cache, in GiB.
HOST_GIB = 8
PROMPT = (
    "Count from 1 to 3000, separated by commas and spaces. "
    "Write only the numbers."
)
# The plugin a test installs: a worker hook that answers its version.
PLUGIN = """
VERSION = "1.0"


def register(registry):
    registry.hook("plugin_probe", probe)


def probe(payload, models):
    return {"version": VERSION, "models": len(models), "payload": payload}
"""
PLUGIN_METADATA = (
    "Metadata-Version: 2.1\nName: uniserve-test-plugin\nVersion: 1.0\n"
)
PLUGIN_ENTRY_POINTS = (
    "[uniserve.plugins]\ntest = uniserve_test_plugin:register\n"
)


def _checkpoint() -> str:
    checkpoint = os.environ.get("UNISERVE_QWEN3_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail("UNISERVE_QWEN3_MODEL must name the model checkpoint")
    return checkpoint


def _shmem_gib() -> float:
    """The host's shared memory in use (``Shmem`` of ``/proc/meminfo``)."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("Shmem:"):
            return int(line.split()[1]) / (1 << 20)
    raise AssertionError("/proc/meminfo states no Shmem")


def _available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1 << 20)
    raise AssertionError("/proc/meminfo states no MemAvailable")


def _gpu_used_mib(device: str) -> int:
    """The memory in use on GPU ``device``, as ``nvidia-smi`` reports it."""
    output = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return int(output.strip())


def _stop_storage_server(socket: Path) -> None:
    """Stop the Storage Server serving `socket`, found by its command line."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"serve" in command and str(socket).encode() in command:
            os.kill(int(entry.name), signal.SIGTERM)


def _storage_list(runtime: Path) -> str:
    """What the test's Storage Server holds (``uniserve storage list``)."""
    return subprocess.run(
        [str(require_uniserve_binary()), "storage", "list"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "XDG_RUNTIME_DIR": str(runtime)},
    ).stdout


@pytest.fixture
def runtime(tmp_path: Path) -> Iterator[Path]:
    """The runtime directory of the test's own Storage Server, whose log
    the test keeps beside its own.
    """  # noqa: D205
    directory = Path(tempfile.mkdtemp(prefix="uniserve-", dir="/tmp"))
    try:
        yield directory
    finally:
        _stop_storage_server(directory / "uniserve" / "storage.sock")
        log = directory / "uniserve" / "storage-server.log"
        if log.is_file():
            shutil.copy(log, tmp_path / "storage-server.log")
        shutil.rmtree(directory, ignore_errors=True)


def _log_lines(log: Path) -> list[str]:
    """The lines of `log` without their terminal colors."""
    colors = re.compile(r"\x1b\[[0-9;]*m")
    return [colors.sub("", line) for line in log.read_text().splitlines()]


def _chat(base_url: str, model: str, prompt: str, tokens: int = 16) -> dict:
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "max_completion_tokens": tokens,
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def _stream_count(
    base_url: str, model: str, read: list[str], done: threading.Event
) -> None:
    """Streams the count, appending each content delta to `read`."""
    with httpx.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "max_completion_tokens": 8192,
            "stream": True,
        },
        timeout=1800,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line == "data: [DONE]":
                break
            if line.startswith("data: "):
                chunk = json.loads(line[6:])
                for choice in chunk.get("choices", ()):
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        read.append(content)
    done.set()


def _wait_until(predicate, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(1.0)
    raise TimeoutError(f"{what} within {timeout_s} s")


@contextmanager
def _node(runtime: Path, device: str, log: Path) -> Iterator[tuple[str, int]]:
    """A node serving no deployment yet; its URL and process id."""
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_server(
        [
            str(require_uniserve_binary()),
            "node",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        log,
        env={"CUDA_VISIBLE_DEVICES": device, "XDG_RUNTIME_DIR": str(runtime)},
    )
    try:

        def answers() -> bool:
            if process.poll() is not None:
                raise RuntimeError(f"the node exited with {process.returncode}")
            try:
                return (
                    httpx.get(f"{base_url}/admin/node", timeout=2).status_code
                    == 200
                )
            except httpx.HTTPError:
                return False

        _wait_until(answers, 60, "the node did not answer")
        yield base_url, process.pid
    finally:
        stop_server(process)


def _node_state(base_url: str) -> dict:
    return httpx.get(f"{base_url}/admin/node", timeout=10).json()


def _deploy(base_url: str, name: str, *assignments: str) -> None:
    response = httpx.post(
        f"{base_url}/admin/deploy",
        json={
            "config": "",
            "set": [
                f"model={_checkpoint()}",
                f"served_model_name={name}",
                f"worker_config.executable={worker_executable()}",
                f"kv_cache_config.host_gib={HOST_GIB}",
                *OPTIONS,
                *assignments,
            ],
        },
        timeout=60,
    )
    assert response.status_code == 202, response.text


def _settle(base_url: str, timeout_s: float = 900) -> dict:
    """Waits until the node serves or idles, and returns its state."""
    state: dict = {}

    def settled() -> bool:
        state.update(_node_state(base_url))
        return state["state"] in ("serving", "idle")

    _wait_until(settled, timeout_s, "the node did not settle")
    return state


def test_a_node_serves_deployments_one_after_another_and_releases_their_memory(
    tmp_path: Path, runtime: Path
):
    device = os.environ.get("UNISERVE_LIVE_DEVICE", "0")
    with _node(runtime, device, tmp_path / "node.log") as (base_url, pid):
        shmem = _shmem_gib()
        gpu = _gpu_used_mib(device)
        print(f"before: Shmem {shmem:.2f} GiB, GPU {gpu} MiB")
        for index in range(3):
            name = f"qwen3-live-{index}"
            _deploy(base_url, name)
            state = _settle(base_url)
            assert state == {
                "state": "serving",
                "deployment": name,
                "last_error": None,
            }, state
            reply = _chat(base_url, name, "Say hello.")
            assert reply["choices"][0]["message"]["content"]
            serving = _shmem_gib()
            print(
                f"{name} serving: Shmem {serving:.2f} GiB, "
                f"GPU {_gpu_used_mib(device)} MiB"
            )
            # The deployment's page pools: its worker's host KV cache.
            assert serving - shmem >= HOST_GIB * 0.99, (serving, shmem)
            assert name in _storage_list(runtime)

            response = httpx.post(
                f"{base_url}/admin/shutdown", json={}, timeout=60
            )
            assert response.status_code == 202, response.text
            state = _settle(base_url)
            assert state["state"] == "idle", state
            assert state["last_error"] is None, state
            after_shmem = _shmem_gib()
            after_gpu = _gpu_used_mib(device)
            print(
                f"{name} down: Shmem {after_shmem:.2f} GiB, GPU {after_gpu} MiB"
            )
            assert name not in _storage_list(runtime)
            # The host's other processes move its shared memory by less than
            # a page pool's gigabytes.
            assert abs(after_shmem - shmem) < 1.0, (after_shmem, shmem)
            assert after_gpu <= gpu + 1024, (after_gpu, gpu)

        # A host KV cache the host's available memory cannot hold.
        too_large = int(_available_gib()) * 2
        _deploy(
            base_url,
            "qwen3-live-too-large",
            f"kv_cache_config.host_gib={too_large}",
        )
        state = _settle(base_url)
        assert state["state"] == "idle", state
        error = state["last_error"] or ""
        print(f"refused: {error}")
        assert "available" in error and "reserve" in error, error
        # The node is the same process throughout.
        assert (
            httpx.get(f"{base_url}/admin/node", timeout=10).status_code == 200
        )
        assert _runs(pid)


@contextmanager
def _serve(
    runtime: Path,
    log: Path,
    name: str,
    env: dict[str, str] | None = None,
    options: tuple[str, ...] = OPTIONS,
    checkpoint: str | None = None,
) -> Iterator[str]:
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(require_uniserve_binary()),
        "serve",
        checkpoint or _checkpoint(),
        "--served-model-name",
        name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-executable",
        str(worker_executable()),
        "--set",
        f"kv_cache_config.host_gib={HOST_GIB}",
    ]
    for option in options:
        command += ["--set", option]
    device = os.environ.get("UNISERVE_LIVE_DEVICE", "0")
    environment = {
        "CUDA_VISIBLE_DEVICES": device,
        "XDG_RUNTIME_DIR": str(runtime),
        **(env or {}),
    }
    with server_process(command, base_url, log, timeout_s=900, env=environment):
        yield base_url


def _admin(
    base_url: str, path: str, body: dict | None = None
) -> httpx.Response:
    return httpx.post(f"{base_url}/admin/{path}", json=body or {}, timeout=1800)


def test_a_serving_deployment_answers_its_admin_api(
    tmp_path: Path, runtime: Path
):
    name = "qwen3-live-admin"
    log = tmp_path / "server.log"
    with _serve(runtime, log, name) as base_url:
        status = httpx.get(f"{base_url}/admin/status", timeout=10).json()
        assert status["deployment"]["deployment"] == name, status
        assert status["deployment"]["engine"] == "ready", status

        # New requests wait while generation is paused.
        paused = _admin(base_url, "pause_generation")
        assert paused.status_code == 200, paused.text
        answered: dict = {}
        waiting = threading.Thread(
            target=lambda: answered.update(_chat(base_url, name, "Say hello."))
        )
        waiting.start()
        time.sleep(5)
        assert not answered, "a request was served while generation paused"
        assert _admin(base_url, "continue_generation").status_code == 200
        waiting.join(timeout=300)
        assert answered["choices"][0]["message"]["content"], answered

        # A flush revokes the pages the same prompt published.
        prompt = "Describe the sea in one sentence. " * 40
        _chat(base_url, name, prompt)
        _chat(base_url, name, prompt)
        flushed = _admin(base_url, "flush_cache")
        assert flushed.status_code == 200, flushed.text
        _wait_until(
            lambda: any(
                re.search(r"flushed the prefix cache.*pages=[1-9]", line)
                for line in _log_lines(log)
            ),
            30,
            "the engine logged no flushed pages",
        )

        # Scheduling limits change in place under a streaming request, and
        # other decode batch sizes by a hot update under the same stream.
        read: list[str] = []
        done = threading.Event()
        streaming = threading.Thread(
            target=_stream_count, args=(base_url, name, read, done)
        )
        streaming.start()
        _wait_until(lambda: len(read) >= 32, 300, "the count did not stream")
        in_place = _admin(
            base_url,
            "update",
            {"set": ["max_running_requests=4"]},
        )
        assert in_place.status_code == 200, in_place.text
        result = in_place.json()["result"]
        assert result["in_place"] is True, result
        assert result["pause_seconds"] == 0, result
        before_update = len(read)
        hot = _admin(
            base_url,
            "update",
            {"set": ["cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8, 16]"]},
        )
        assert hot.status_code == 200, hot.text
        hot_result = hot.json()["result"]
        assert hot_result["in_place"] is False, hot_result
        print(f"hot update pause: {hot_result['pause_seconds']:.3f} s")
        streaming.join(timeout=1800)
        assert done.is_set(), "the count did not end"
        text = "".join(read)
        assert len(read) > before_update, "the stream ended before the update"
        numbers = [int(word) for word in re.findall(r"\d+", text)]
        assert numbers[:100] == list(range(1, 101)), text[:400]
        assert any(
            "the engine took the new scheduling limits in place" in line
            for line in _log_lines(log)
        )

        # A worker hook no plugin implements fails with that reason.
        hook = _admin(base_url, "hooks/plugin_probe", {"probe": 1})
        assert hook.status_code >= 400, hook.text
        assert "no plugin implements the worker hook" in hook.text, hook.text

        # The manager attaches, follows the node's events and detaches.
        manager = shutil.which("uniserve-manager") or str(
            Path(require_uniserve_binary()).with_name("uniserve-manager")
        )
        status_line = subprocess.run(
            [manager, "--node", base_url, "status"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
        assert name in status_line and "engine ready" in status_line, (
            status_line
        )
        watching = subprocess.Popen(
            [manager, "--node", base_url, "watch", "--from-start"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(5)
        watching.send_signal(signal.SIGINT)
        output, _ = watching.communicate(timeout=60)
        assert watching.returncode == 0, output
        assert (
            "update" in output and "detached; the nodes serve on" in output
        ), output
        reply = _chat(base_url, name, "Say hello.")
        assert reply["choices"][0]["message"]["content"]

        events = httpx.get(
            f"{base_url}/admin/events?after=0", timeout=10
        ).json()
        kinds = [event["kind"] for event in events["events"]]
        assert "operation" in kinds, events


def test_a_hot_update_takes_up_and_drops_a_plugin(
    tmp_path: Path, runtime: Path
):
    name = "qwen3-live-plugin"
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    log = tmp_path / "server.log"
    environment = {
        "PYTHONPATH": os.pathsep.join(
            [str(plugins), *filter(None, [os.environ.get("PYTHONPATH")])]
        )
    }
    with _serve(runtime, log, name, environment) as base_url:

        def probe() -> httpx.Response:
            return _admin(base_url, "hooks/plugin_probe", {"probe": 7})

        before = probe()
        assert before.status_code >= 400, before.text

        package = plugins / "uniserve_test_plugin"
        package.mkdir()
        (package / "__init__.py").write_text(PLUGIN)
        metadata = plugins / "uniserve_test_plugin-1.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(PLUGIN_METADATA)
        (metadata / "entry_points.txt").write_text(PLUGIN_ENTRY_POINTS)
        installed = _admin(base_url, "update")
        assert installed.status_code == 200, installed.text
        assert installed.json()["result"]["in_place"] is False

        answered = probe()
        assert answered.status_code == 200, answered.text
        results = answered.json()["result"]["results"]
        assert results and all(
            result == {"version": "1.0", "models": 1, "payload": {"probe": 7}}
            for result in results
        ), results
        reply = _chat(base_url, name, "Say hello.")
        assert reply["choices"][0]["message"]["content"]

        shutil.rmtree(package)
        shutil.rmtree(metadata)
        removed = _admin(base_url, "update")
        assert removed.status_code == 200, removed.text
        after = probe()
        assert after.status_code >= 400, after.text
        reply = _chat(base_url, name, "Say hello.")
        assert reply["choices"][0]["message"]["content"]


def _pids(log: Path, role: str) -> set[int]:
    """The process ids the log names for `role` (``Daemon`` or ``Engine``)."""
    pattern = re.compile(rf"\({role} pid=(\d+)\)")
    return {
        int(match.group(1))
        for line in _log_lines(log)
        if (match := pattern.search(line))
    }


def _runs(pid: int) -> bool:
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


def _stop_pid(pid: int) -> None:
    """Stops daemon `pid` as SIGTERM stops a daemon, if it still runs."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    _wait_until(
        lambda: not _runs(pid),
        120,
        f"daemon {pid} did not stop",
    )


def test_a_hot_restart_hands_the_deployment_to_a_successor_daemon(
    tmp_path: Path, runtime: Path
):
    name = "qwen3-live-restart"
    log = tmp_path / "server.log"
    # The daemons a restart starts, which the fixture, stopping the daemon
    # it started, does not stop.
    successors: list[int] = []
    try:
        with _serve(runtime, log, name) as base_url:
            _restart_while_streaming(base_url, name, log, successors)
    finally:
        for pid in successors:
            _stop_pid(pid)


def _restart_while_streaming(
    base_url: str, name: str, log: Path, successors: list[int]
) -> None:
    engines_before = _pids(log, "Engine")
    status = httpx.get(f"{base_url}/admin/status", timeout=10).json()
    assert status["deployment"]["engine"] == "ready", status
    (old_daemon,) = _pids(log, "Daemon")

    read: list[str] = []
    done = threading.Event()
    streaming = threading.Thread(
        target=_stream_count, args=(base_url, name, read, done)
    )
    streaming.start()
    _wait_until(lambda: len(read) >= 32, 300, "the count did not stream")
    restarted = _admin(base_url, "restart")
    assert restarted.status_code == 200, restarted.text
    successor = restarted.json()["result"]["pid"]
    successors.append(successor)
    assert successor != old_daemon
    before_restart = len(read)

    # New requests are served while the old daemon finishes the count.
    for _ in range(3):
        reply = _chat(base_url, name, "Say hello.")
        assert reply["choices"][0]["message"]["content"]
    streaming.join(timeout=1800)
    assert done.is_set(), "the count did not end"
    assert len(read) > before_restart, "the stream ended at the restart"
    numbers = [int(word) for word in re.findall(r"\d+", "".join(read))]
    assert numbers[:200] == list(range(1, 201)), "".join(read)[:400]

    # The old daemon exits once its requests ended; the engine and its
    # workers are the same processes throughout.
    _wait_until(
        lambda: not _runs(old_daemon),
        120,
        "the replaced daemon did not exit",
    )
    assert _runs(successor)
    status = httpx.get(f"{base_url}/admin/status", timeout=10).json()
    assert status["deployment"]["engine"] == "ready", status
    assert _pids(log, "Engine") == engines_before
    lines = _log_lines(log)
    assert any("the engine attached" in line for line in lines)
    # The workers read the checkpoint once, when the deployment started.
    assert (
        sum(
            "uniserve.loading.rank: loaded numerical model" in line
            for line in lines
        )
        == 1
    )
    events = httpx.get(f"{base_url}/admin/events?after=0", timeout=10).json()
    messages = [event["message"] for event in events["events"]]
    assert any("after a hot restart" in message for message in messages), (
        messages
    )

    # The successor restarts in turn, and serves an update.
    again = _admin(base_url, "restart")
    assert again.status_code == 200, again.text
    successors.append(again.json()["result"]["pid"])
    _wait_until(
        lambda: not _runs(successor),
        120,
        "the first successor did not exit",
    )
    reply = _chat(base_url, name, "Say hello.")
    assert reply["choices"][0]["message"]["content"]
    update = _admin(base_url, "update", {"set": ["max_running_requests=8"]})
    assert update.status_code == 200, update.text
    assert update.json()["result"]["in_place"] is True


def _storage_pid(runtime: Path) -> int:
    """The process id of the Storage Server serving the test's socket."""
    socket = str(runtime / "uniserve" / "storage.sock").encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"serve" in command and socket in command:
            return int(entry.name)
    raise AssertionError("no Storage Server serves the test's socket")


def test_a_storage_server_restart_keeps_the_deployment_serving(
    tmp_path: Path, runtime: Path
):
    name = "qwen3-live-storage"
    log = tmp_path / "server.log"
    with _serve(runtime, log, name) as base_url:
        prompt = "Describe the sea in one sentence. " * 40
        _chat(base_url, name, prompt)
        before = sorted(_storage_list(runtime).splitlines())
        old_server = _storage_pid(runtime)

        read: list[str] = []
        done = threading.Event()
        streaming = threading.Thread(
            target=_stream_count, args=(base_url, name, read, done)
        )
        streaming.start()
        _wait_until(lambda: len(read) >= 32, 300, "the count did not stream")
        restarted = subprocess.run(
            [str(require_uniserve_binary()), "storage", "restart"],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "XDG_RUNTIME_DIR": str(runtime)},
            timeout=120,
        ).stdout
        successor = int(
            re.search(r"successor server (\d+)", restarted).group(1)
        )
        _wait_until(
            lambda: not _runs(old_server),
            120,
            "the replaced Storage Server did not exit",
        )
        assert _storage_pid(runtime) == successor
        assert sorted(_storage_list(runtime).splitlines()) == before

        # The deployment serves throughout: the count goes on, a new
        # request is answered, and a hot update maps the weights the
        # successor holds.
        reply = _chat(base_url, name, prompt)
        assert reply["choices"][0]["message"]["content"]
        streaming.join(timeout=1800)
        assert done.is_set(), "the count did not end"
        numbers = [int(word) for word in re.findall(r"\d+", "".join(read))]
        assert numbers[:200] == list(range(1, 201)), "".join(read)[:400]
        hot = _admin(base_url, "update")
        assert hot.status_code == 200, hot.text
        reply = _chat(base_url, name, "Say hello.")
        assert reply["choices"][0]["message"]["content"]
        lines = _log_lines(log)
        assert (
            sum(
                "uniserve.loading.rank: loaded numerical model" in line
                for line in lines
            )
            == 1
        ), (
            "an engine read the checkpoint again rather than the weights "
            "the Storage Server retains"
        )
        assert any(
            re.search(r"mapped the retained weights .*missing=0", line)
            for line in lines
        ), "the updated engine did not map the retained weights"


# The options of a deployment whose KV cache takes the device memory its
# share leaves, rather than a fixed token count.
SIZED_OPTIONS = tuple(
    option
    for option in OPTIONS
    if not option.startswith("kv_cache_config.max_tokens")
)


@contextmanager
def _ballast(device: str, leave_gib: float) -> Iterator[int]:
    """Hold GPU `device`'s free memory but `leave_gib` GiB in a process.

    The process holds it until the context ends; yields the bytes it holds.
    """
    code = (
        "import sys, torch\n"
        "free, _ = torch.cuda.mem_get_info()\n"
        "hold = max(0, free - int(float(sys.argv[1]) * (1 << 30)))\n"
        "ballast = torch.empty(hold, dtype=torch.uint8, device='cuda')\n"
        "print(hold, flush=True)\n"
        "sys.stdin.read()\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(leave_gib)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": device},
    )
    try:
        assert process.stdout is not None
        line = process.stdout.readline()
        assert line.strip().isdigit(), f"the ballast process answered {line!r}"
        yield int(line)
    finally:
        # Closing its input ends the process, which frees the memory.
        assert process.stdin is not None
        process.stdin.close()
        process.wait(timeout=120)


def test_a_hot_update_under_memory_pressure_shrinks_the_kv_cache(
    tmp_path: Path, runtime: Path
):
    name = "qwen3-live-pressure"
    log = tmp_path / "server.log"
    device = os.environ.get("UNISERVE_LIVE_DEVICE", "0")
    prompt = "Describe the sea in one sentence."
    with _serve(runtime, log, name, options=SIZED_OPTIONS) as base_url:
        before = _chat(base_url, name, prompt, 32)["choices"][0]["message"][
            "content"
        ]
        read: list[str] = []
        done = threading.Event()
        streaming = threading.Thread(
            target=_stream_count, args=(base_url, name, read, done)
        )
        streaming.start()
        _wait_until(lambda: len(read) >= 32, 300, "the count did not stream")
        with _ballast(device, leave_gib=2.0) as held:
            print(f"ballast holds {held / (1 << 30):.1f} GiB; GPU free 2 GiB")
            hot = _admin(
                base_url,
                "update",
                {
                    "set": [
                        "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8, 16]"
                    ]
                },
            )
            assert hot.status_code == 200, hot.text
            result = hot.json()["result"]
            assert result["in_place"] is False, result
            print(
                "hot update under pressure, pause: "
                f"{result['pause_seconds']:.3f} s"
            )
        streaming.join(timeout=1800)
        assert done.is_set(), "the count did not end"
        numbers = [int(word) for word in re.findall(r"\d+", "".join(read))]
        assert numbers[:200] == list(range(1, 201)), "".join(read)[:400]

        lines = _log_lines(log)
        withdrew = [
            line
            for line in lines
            if "withdrew the tail of the KV cache" in line
        ]
        gave = [
            line
            for line in lines
            if "gave back the tail of the KV cache" in line
        ]
        restored = [line for line in lines if "restored the KV cache" in line]
        for line in withdrew + gave + restored:
            print(line)
        assert withdrew, "the running engine did not shrink its KV cache"
        assert gave, "the running engine's workers did not give the tail back"
        assert restored, "the updated engine did not create the tail again"
        after = _chat(base_url, name, prompt, 32)["choices"][0]["message"][
            "content"
        ]
        assert after == before


def test_a_sleeping_deployment_gives_back_its_cache_and_graph_memory(
    tmp_path: Path, runtime: Path
):
    name = "qwen3-live-sleep"
    log = tmp_path / "server.log"
    device = os.environ.get("UNISERVE_LIVE_DEVICE", "0")
    prompt = "Describe the sea in one sentence."
    with _serve(runtime, log, name, options=SIZED_OPTIONS) as base_url:
        before = _chat(base_url, name, prompt, 32)["choices"][0]["message"][
            "content"
        ]
        awake = _gpu_used_mib(device)
        read: list[str] = []
        done = threading.Event()
        streaming = threading.Thread(
            target=_stream_count, args=(base_url, name, read, done)
        )
        streaming.start()
        _wait_until(lambda: len(read) >= 32, 300, "the count did not stream")

        slept = _admin(base_url, "sleep")
        assert slept.status_code == 200, slept.text
        status = httpx.get(f"{base_url}/admin/status", timeout=10).json()
        assert status["deployment"]["asleep"] is True, status
        asleep = _gpu_used_mib(device)
        answered: dict = {}
        waiting = threading.Thread(
            target=lambda: answered.update(_chat(base_url, name, prompt, 32))
        )
        waiting.start()
        time.sleep(5)
        assert not answered, "a sleeping deployment served a request"

        woke = _admin(base_url, "wake_up")
        assert woke.status_code == 200, woke.text
        waiting.join(timeout=300)
        assert answered["choices"][0]["message"]["content"], answered
        streaming.join(timeout=1800)
        assert done.is_set(), "the count did not end"
        numbers = [int(word) for word in re.findall(r"\d+", "".join(read))]
        assert numbers[:200] == list(range(1, 201)), "".join(read)[:400]
        after = _gpu_used_mib(device)
        print(
            f"GPU memory: awake {awake} MiB, asleep {asleep} MiB, "
            f"woken {after} MiB"
        )
        assert awake - asleep >= 10 * 1024, (awake, asleep)
        # Alone, as before the sleep, the deployment answers as it did: the
        # request that waited ran beside the recomputed count, whose batch
        # changes the numerics.
        alone = _chat(base_url, name, prompt, 32)
        assert alone["choices"][0]["message"]["content"] == before, alone
        assert abs(after - awake) <= 2 * 1024, (awake, after)
        lines = _log_lines(log)
        assert any("the worker sleeps" in line for line in lines)
        assert any("the worker woke" in line for line in lines)
        status = httpx.get(f"{base_url}/admin/status", timeout=10).json()
        assert status["deployment"]["asleep"] is False, status
        assert status["deployment"]["paused"] is False, status


# The weight source plugin a test installs: ``renormed://<checkpoint>``
# streams the checkpoint's tensors with the final norm scaled as the other
# checkpoint scales it.
SOURCE_PLUGIN = """
import json
from pathlib import Path

import torch
from safetensors import safe_open

DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "I64": torch.int64,
    "I32": torch.int32,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def register(registry):
    registry.weight_source("renormed", Renormed)


class Renormed:
    def __init__(self, uri):
        self.root = Path(uri.removeprefix("renormed://"))
        index = json.loads(
            (self.root / "model.safetensors.index.json").read_text()
        )
        self.files = index["weight_map"]
        self.handles = {}

    def identity(self):
        return f"renormed:{self.root}:sin-0.5"

    def _handle(self, file):
        if file not in self.handles:
            self.handles[file] = safe_open(self.root / file, framework="pt")
        return self.handles[file]

    def tensors(self, source):
        listed = []
        for name, file in sorted(self.files.items()):
            piece = self._handle(file).get_slice(name)
            listed.append(
                (name, tuple(piece.get_shape()), DTYPES[piece.get_dtype()])
            )
        return listed

    def read(self, source, name):
        value = self._handle(self.files[name]).get_tensor(name)
        if name == "model.norm.weight":
            scale = 1 + 0.5 * torch.sin(
                torch.arange(value.numel(), dtype=torch.float32)
            )
            value = (value.float() * scale).to(value.dtype)
        return value
"""


def _updated_checkpoint() -> str:
    checkpoint = os.environ.get("UNISERVE_QWEN3_UPDATED_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail(
            "UNISERVE_QWEN3_UPDATED_MODEL must name another checkpoint of "
            "the model, with other weights"
        )
    return checkpoint


def test_a_weight_update_answers_as_the_new_checkpoint_does(
    tmp_path: Path, runtime: Path
):
    name = "qwen3-live-weights"
    updated = _updated_checkpoint()
    prompts = ("Describe the sea in one sentence.", "Name three prime numbers.")

    def answers(base_url: str) -> list[str]:
        return [
            _chat(base_url, name, prompt, 32)["choices"][0]["message"][
                "content"
            ]
            for prompt in prompts
        ]

    with _serve(
        runtime, tmp_path / "cold.log", name, checkpoint=updated
    ) as base_url:
        expected = answers(base_url)
    plugins = tmp_path / "plugins"
    package = plugins / "uniserve_source_plugin"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(SOURCE_PLUGIN)
    metadata = plugins / "uniserve_source_plugin-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: uniserve-source-plugin\nVersion: 1.0\n"
    )
    (metadata / "entry_points.txt").write_text(
        "[uniserve.plugins]\nsource = uniserve_source_plugin:register\n"
    )
    environment = {
        "PYTHONPATH": os.pathsep.join(
            [str(plugins), *filter(None, [os.environ.get("PYTHONPATH")])]
        )
    }
    log = tmp_path / "server.log"
    with _serve(runtime, log, name, environment) as base_url:
        original = answers(base_url)
        assert original != expected, "the checkpoints answer alike"

        read: list[str] = []
        done = threading.Event()
        streaming = threading.Thread(
            target=_stream_count, args=(base_url, name, read, done)
        )
        streaming.start()
        _wait_until(lambda: len(read) >= 32, 300, "the count did not stream")
        response = _admin(base_url, "update_weights", {"model_path": updated})
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert result["model_path"] == updated, result
        assert all(report["page_contents"] for report in result["results"]), (
            result
        )
        streaming.join(timeout=1800)
        assert done.is_set(), "the stream did not end after the update"
        assert answers(base_url) == expected

        back = _admin(base_url, "update_weights", {"model_path": _checkpoint()})
        assert back.status_code == 200, back.text
        assert answers(base_url) == original

        refused = _admin(
            base_url,
            "update_weights",
            {"model_path": str(tmp_path / "nowhere")},
        )
        assert refused.status_code == 400, refused.text
        assert answers(base_url) == original

        # The plugin streams the base checkpoint's tensors with the other
        # checkpoint's norm: the same answers as that checkpoint's.
        streamed = _admin(
            base_url,
            "update_weights",
            {"model_path": f"renormed://{_checkpoint()}"},
        )
        assert streamed.status_code == 200, streamed.text
        assert answers(base_url) == expected
        # A hot update's standby binds the weights the source left.
        hot = _admin(base_url, "update")
        assert hot.status_code == 200, hot.text
        assert answers(base_url) == expected
        assert any(
            re.search(r"mapped the retained weights .*missing=0", line)
            for line in _log_lines(log)
        ), "the standby did not map the streamed weights"
        back = _admin(base_url, "update_weights", {"model_path": _checkpoint()})
        assert back.status_code == 200, back.text
        assert answers(base_url) == original
        lines = _log_lines(log)
        # Every update frees the checkpoint it loaded: what stays allocated
        # does not grow from one update to the next.
        left = [
            float(match.group(1))
            for line in lines
            if (match := re.search(r"tensors in place .*; ([\d.]+) GiB", line))
        ]
        assert len(left) >= 4, left
        assert max(left) - min(left) < 1, left
        assert any("renamed the retained weights" in line for line in lines)


# The LoRA plugin a test installs: it records which adapter each slot of its
# pool holds, and computes nothing with them.
LORA_PLUGIN = """
loaded = {}


def register(registry):
    registry.hook("load_lora_adapter", load)
    registry.hook("unload_lora_adapter", unload)
    registry.hook("lora_probe", probe)


def load(payload, models):
    loaded[payload["lora_name"]] = payload["adapter"]
    return {"adapter": payload["adapter"]}


def unload(payload, models):
    loaded.pop(payload["lora_name"], None)
    return {"adapter": payload["adapter"]}


def probe(payload, models):
    return sorted(loaded.items())
"""


def test_a_lora_adapter_is_served_under_its_name_across_a_hot_update(
    tmp_path: Path, runtime: Path
):
    name = "qwen3-live-lora"
    plugins = tmp_path / "plugins"
    package = plugins / "uniserve_lora_plugin"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(LORA_PLUGIN)
    metadata = plugins / "uniserve_lora_plugin-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: uniserve-lora-plugin\nVersion: 1.0\n"
    )
    (metadata / "entry_points.txt").write_text(
        "[uniserve.plugins]\nlora = uniserve_lora_plugin:register\n"
    )
    environment = {
        "PYTHONPATH": os.pathsep.join(
            [str(plugins), *filter(None, [os.environ.get("PYTHONPATH")])]
        )
    }
    prompt = "Describe the sea in one sentence."
    with _serve(
        runtime, tmp_path / "server.log", name, environment
    ) as base_url:

        def probe() -> list:
            response = _admin(base_url, "hooks/lora_probe", {})
            assert response.status_code == 200, response.text
            return response.json()["result"]["results"]

        def ask(model: str) -> httpx.Response:
            return httpx.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "temperature": 0,
                    "max_completion_tokens": 16,
                },
                timeout=600,
            )

        loaded = _admin(
            base_url,
            "load_lora_adapter",
            {"lora_name": "tuned", "lora_path": "/adapters/tuned"},
        )
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()["result"]["results"] == [{"adapter": 1}]
        assert probe() == [[["tuned", 1]]]
        models = httpx.get(f"{base_url}/v1/models", timeout=10).json()
        assert [model["id"] for model in models["data"]] == [name, "tuned"]
        # The plugin computes nothing with its adapters: the adapter's
        # answer is the base model's.
        base = ask(name)
        tuned = ask("tuned")
        assert tuned.status_code == 200, tuned.text
        assert tuned.json()["model"] == "tuned"
        assert (
            tuned.json()["choices"][0]["message"]["content"]
            == base.json()["choices"][0]["message"]["content"]
        )

        # The engine a hot update starts loads the adapter again.
        updated = _admin(base_url, "update")
        assert updated.status_code == 200, updated.text
        assert probe() == [[["tuned", 1]]]
        assert ask("tuned").status_code == 200

        unloaded = _admin(
            base_url, "unload_lora_adapter", {"lora_name": "tuned"}
        )
        assert unloaded.status_code == 200, unloaded.text
        assert probe() == [[]]
        assert ask("tuned").status_code == 404


# The communication buffer plugin a test installs: 4 GiB of device memory.
BUFFER_PLUGIN = """
import torch


def register(registry):
    registry.buffer("test", allocate)


def allocate(models, device):
    return torch.empty(4 << 30, dtype=torch.uint8, device=f"cuda:{device}")
"""


def _fitted_units(log: Path) -> tuple[int, int]:
    """The cache units the worker's free memory fits, before it rounds them
    down to whole segments, and the bytes of each.
    """  # noqa: D205
    fitted = None
    for line in _log_lines(log):
        match = re.search(r"down to whole segments.*fitted=(\d+)", line)
        if match:
            fitted = int(match.group(1))
        match = re.search(
            r"planned capacity: (\d+) cache units of (\d+) bytes", line
        )
        if match and fitted is not None:
            return fitted, int(match.group(2))
    raise AssertionError("the worker logged no fitted and planned capacity")


def test_a_plugins_communication_buffer_counts_against_the_kv_cache(
    tmp_path: Path, runtime: Path
):
    name = "qwen3-live-buffer"
    plugins = tmp_path / "plugins"
    package = plugins / "uniserve_buffer_plugin"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(BUFFER_PLUGIN)
    metadata = plugins / "uniserve_buffer_plugin-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: uniserve-buffer-plugin\nVersion: 1.0\n"
    )
    (metadata / "entry_points.txt").write_text(
        "[uniserve.plugins]\nbuffer = uniserve_buffer_plugin:register\n"
    )
    environment = {
        "PYTHONPATH": os.pathsep.join(
            [str(plugins), *filter(None, [os.environ.get("PYTHONPATH")])]
        )
    }
    plain = tmp_path / "plain.log"
    with _serve(runtime, plain, name, options=SIZED_OPTIONS):
        pass
    buffered = tmp_path / "buffered.log"
    with _serve(
        runtime, buffered, name, environment, options=SIZED_OPTIONS
    ) as base_url:
        reply = _chat(base_url, name, "Say hello.")
        assert reply["choices"][0]["message"]["content"]
    assert any(
        "allocated the plugins' communication buffers" in line
        for line in _log_lines(buffered)
    )
    (units, unit_bytes), (fewer, _) = (
        _fitted_units(plain),
        _fitted_units(buffered),
    )
    expected = (4 << 30) // unit_bytes
    print(f"cache units: {units} without the buffer, {fewer} with it")
    assert abs((units - fewer) - expected) <= expected // 100 + 16, (
        units,
        fewer,
        expected,
    )


# The graph policy plugin a test installs: decode graphs of 1, 2 and 4 rows.
GRAPH_PLUGIN = """
def register(registry):
    registry.graph_policy(choose)


def choose(models, configured):
    assert configured["mode"] == "auto", configured
    return {"decode": [1, 2, 4]}
"""


def test_a_plugins_graph_policy_chooses_the_captured_buckets(
    tmp_path: Path, runtime: Path
):
    name = "qwen3-live-graphs"
    plugins = tmp_path / "plugins"
    package = plugins / "uniserve_graph_plugin"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(GRAPH_PLUGIN)
    metadata = plugins / "uniserve_graph_plugin-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: uniserve-graph-plugin\nVersion: 1.0\n"
    )
    (metadata / "entry_points.txt").write_text(
        "[uniserve.plugins]\ngraphs = uniserve_graph_plugin:register\n"
    )
    environment = {
        "PYTHONPATH": os.pathsep.join(
            [str(plugins), *filter(None, [os.environ.get("PYTHONPATH")])]
        )
    }
    log = tmp_path / "graphs.log"
    with _serve(
        runtime, log, name, environment, options=SIZED_OPTIONS
    ) as base_url:
        reply = _chat(base_url, name, "Say hello.")
        assert reply["choices"][0]["message"]["content"]
    lines = _log_lines(log)
    assert any("a plugin chose the graph policy" in line for line in lines)
    # The three chosen sizes and the bound of the rows a decode batch admits.
    captured = [
        int(match.group(1))
        for line in lines
        if (match := re.search(r"text graphs: (\d+) decode", line))
    ]
    assert captured and all(buckets == 4 for buckets in captured), captured
