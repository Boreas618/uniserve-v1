"""A deployment's lifecycle and live updates through the admin API.

Qwen3 serves on one GPU of the host, with a Storage Server of the test's
own (``XDG_RUNTIME_DIR`` points the daemon at a directory of the test's,
and the server the daemon starts there is stopped at the end).

- A node (``uniserve serve`` without a model) brings deployments up
  (``PUT /admin/config``) and down through its admin API without
  restarting: each deployment's host regions (the host tier of its buffer
  pool) and weights belong to it, so bringing it down returns the host's
  shared memory and the GPU's storage to what they were before it, and a
  deployment whose host regions the host's available storage cannot hold is
  refused with the sizes that decided it.
- A serving deployment answers its admin API, on the daemon's operator
  socket: new requests wait while generation is paused and are served once
  it continues; a flush evicts the device copies of the cached blocks;
  scheduling limits within the batch the workers were started for change in
  place while a request streams, and a change of the decode batch sizes is
  an epoch change under the same stream, which the operator commits; the
  manager attaches to the node, follows its events and detaches while it
  serves.
- A sleeping deployment gives back its KV cache and graph storage, holds
  new requests and recomputes the ones it ran once it wakes, and answers
  as before.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.build_server import (
    build_server as running_build_server,
)
from tests.python.e2e.http_helpers import (
    QWEN3_UPDATED_CHECKPOINT,
    admin,
    admin_config,
    admin_request,
    admin_socket,
    admin_status,
    apply_config,
    build_server_socket,
    engine_executable,
    engine_processes,
    engine_revision,
    engine_selection,
    granted_devices,
    host_asset,
    install_engine,
    log_tail,
    metric,
    modules_from,
    qwen3_checkpoint,
    require_uniserve_binary,
    runs,
    runtime_directory,
    server_process,
    serving_engine,
    start_server,
    stop_server,
    storage_status,
    taken_over,
    tokens_found,
    update_report,
    wait_for_phase,
    wait_until,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

# Options every deployment of the tests serves with: few captured graphs, so
# that engines start in about a minute, and a small device KV cache.
OPTIONS = (
    "dtype=bfloat16",
    # The attention package the deployment's paged attention executes, as a
    # package name (one of the store's packages, ``KEYS``). ``auto`` names no
    # package, so the deployment names the backend design 9's default choice
    # picks for this shape: trtllm-gen.
    "attention_backend=trtllm",
    "kv_cache_config.max_tokens=32768",
    "chat_config.reasoning_parser=none",
    "cuda_graph_config.prefill_token_sizes=[512]",
    "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
    # The step's token budget is the prefill graph's size, and it is the bound
    # the scheduling policy's own limits are given: unstated, the daemon has no
    # token bound to hand the policy, the policy falls back to its own default
    # (a 8192-token chunk) and the mechanism refuses it against the build
    # config's ceiling of 512 -- the number this deployment's captured prefill
    # graph states. Stating the budget makes the deployment, its graphs and its
    # policy agree on one number instead of three.
    "max_num_batched_tokens=512",
)
# A change of the options that only a new epoch executes: graphs of other
# decode batch sizes.
CHANGE = "cuda_graph_config.decode_batch_sizes=[1, 2, 8]"
# The deployment's host tier on the host, in GiB.
HOST_GIB = 8
PROMPT = (
    "Count from 1 to 3000, separated by commas and spaces. "
    "Write only the numbers."
)


#: The socket of the module's Build Server, which every deployment of this
#: file names as ``build_config.socket``. Set by :func:`build_server`.
BUILD_SERVER: Path | None = None


@pytest.fixture(scope="module", autouse=True)
def build_server(tmp_path_factory: pytest.TempPathFactory):
    """The host's Build Server, with the deployment's modules installed.

    Autouse and module-scoped because every deployment of this file needs it:
    the daemon's leader resolves each epoch's modules at this server. It is
    also the server the shared helpers install engine revisions at
    (:func:`modules_from`, which :func:`engine_revision` reads).
    """
    global BUILD_SERVER
    with running_build_server(tmp_path_factory) as socket, modules_from(socket):
        BUILD_SERVER = socket
        try:
            yield socket
        finally:
            BUILD_SERVER = None


def build_server_option() -> list[str]:
    """The command-line option that gives a deployment its modules' server.

    Every deployment serves a scheduling policy, and a policy's module comes
    from this host's Build Server (design 6.1); without the option the daemon
    starts a Build Server of its own on the default socket, which holds none
    of the module's packages.
    """
    if BUILD_SERVER is None:
        pytest.fail("the module's Build Server is not running")
    return ["--set", f"build_config.socket={BUILD_SERVER}"]


def _host_region_gib(runtime: Path) -> float:
    """The host regions the test's Storage Server holds, in GiB: the sizes
    of its ``uniserve-host-region`` memfds, one per rank and region, whose
    storage the host frees once the server and the ranks that map them let
    them go.
    """  # noqa: D205
    pid = _storage_pid(runtime)
    held = 0
    for descriptor in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(descriptor)
            if target.startswith("/memfd:uniserve-host-region"):
                held += descriptor.stat().st_size
        except OSError:
            continue
    return held / (1 << 30)


def _available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1 << 20)
    raise AssertionError("/proc/meminfo states no MemAvailable")


def _gpu_used_mib(device: str) -> int:
    """The storage in use on GPU ``device``, as ``nvidia-smi`` reports it."""
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


def _storage_list(runtime: Path) -> str:
    """What the test's Storage Server holds (``storage-server list``)."""
    return subprocess.run(
        [str(require_uniserve_binary()), "storage-server", "list"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "XDG_RUNTIME_DIR": str(runtime)},
    ).stdout


def _held(runtime: Path) -> list[str]:
    """What the test's Storage Server holds: its tensors, deployments and
    quotas. Its Transfer Server's process, contexts, link measurements and
    tasks are no storage it holds: a restarted server's Transfer Server
    measures anew. Its buffer pools' blocks, groups and copies change with
    every step of the requests the deployment serves.
    """  # noqa: D205
    return sorted(
        line
        for line in _storage_list(runtime).splitlines()
        if not line.startswith(("transfer ", "link ", "buffer pool of "))
    )


@pytest.fixture
def runtime(tmp_path: Path) -> Iterator[Path]:
    """The runtime directory of the test's own Storage Server, whose log
    the test keeps beside its own.
    """  # noqa: D205
    with runtime_directory(tmp_path / "storage-server.log") as directory:
        yield directory


def _log_trouble(log: Path, limit: int = 40) -> str:
    """The last `limit` warnings and errors of a server's log, marked.

    A tail of the last lines is the wrong excerpt for a failure whose cause is
    minutes earlier: a relaunch logs a capture per bucket per device, so two
    hundred tail lines can hold nothing but captures and stop short of the
    engine's death. The warnings and errors are what name a cause, and there
    are few of them whatever the moment.
    """  # noqa: D205
    try:
        # The engine writes `WARN` and `ERROR`; a process that dies writes
        # neither, so a panic and the daemon's own account of the loss are
        # matched by their text as well. The handover's own steps are `INFO`
        # and are the only record of how far it got, so the ones that decide
        # whether the successor ever became a front are named here: a run that
        # fails at the handover without them says nothing about which step.
        marks = (
            " WARN",
            " ERROR",
            "panicked",
            "the engine was lost",
            "a daemon attached",
            "the main process ends",
            "this process's interface",
            "the daemon closed the process interface",
            "the attached daemons still",
            "the successor received the service",
            "the handed-over control connection closed",
            "the successor owns it, so it is not relaunched here",
            "the engine read an engine command",
            "a serve pass took longer than five seconds",
            "closed its front; the",
            "the daemon hands its service",
            "attaching a daemon failed",
            "an engine connected to this daemon's request socket",
            "asked the engine to dial",
            "was not asked to dial",
            "a request was taken over",
            "the successor took the responses in flight over",
            "the service goes on from the previous daemon",
            "the responses in flight are paused",
            "adopted the processes the previous daemon supervised",
        )
        trouble = []
        colors = re.compile(r"\x1b\[[0-9;]*m")
        for line in colors.sub("", log.read_text()).splitlines():
            if any(mark in line for mark in marks):
                # A repeated line is one fact, not many: a storm of the same
                # warning evicted three runs' evidence, because a storm fills
                # every line the excerpt has. Consecutive repeats collapse to
                # one line and a count, so what differs from the storm still
                # reaches the artifact.
                if trouble and trouble[-1][0] == line:
                    trouble[-1][1] += 1
                else:
                    trouble.append([line, 1])
    except OSError as error:
        return f"\n(server log unreadable at {log}: {error})"
    if not trouble:
        return f"\n--- no warning or error in {log} ---"
    shown = trouble[-limit:]
    lines = [
        line if count == 1 else f"{line} (repeated {count} times)"
        for line, count in shown
    ]
    return (
        f"\n--- {len(lines)} warnings and errors of {log} ---\n"
        + "\n".join(lines)
        + "\n--- end of excerpts ---"
    )


def _chat(
    base_url: str,
    model: str,
    prompt: str,
    tokens: int = 16,
    timeout_s: float = 600,
) -> dict:
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "max_completion_tokens": tokens,
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as error:
        # A body that is not JSON is either an empty success or a frame that
        # belongs to a stream; the two have different owners, and the status
        # and the first bytes are what tell them apart.
        raise AssertionError(
            f"the completion's body is not JSON: {response.status_code} "
            f"{response.headers.get('content-type')!r} {response.text[:400]!r}"
        ) from error


def _stream_count(
    base_url: str,
    model: str,
    read: list[str],
    done: threading.Event,
    reasons: list[str] | None = None,
    prompt: str = PROMPT,
    errors: list[BaseException] | None = None,
    arrivals: list[float] | None = None,
) -> None:
    """Streams the count, appending each content delta to `read`, the moment
    it arrived to `arrivals`, and each finish reason to `reasons`.

    Runs on a thread, where an exception would otherwise be printed to a log
    the test's failure does not carry and read exactly like a stream that
    produced nothing: a refusal, a timeout and a stall would all show as a
    count of zero. `errors` keeps what went wrong so the failure states it.
    """  # noqa: D205
    try:
        _stream_count_body(base_url, model, read, reasons, prompt, arrivals)
    except BaseException as error:  # noqa: BLE001 - reported, never swallowed
        if errors is not None:
            errors.append(error)
        else:
            raise
    finally:
        done.set()


def _stream_count_body(
    base_url: str,
    model: str,
    read: list[str],
    reasons: list[str] | None,
    prompt: str,
    arrivals: list[float] | None = None,
) -> None:
    with httpx.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
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
                        if arrivals is not None:
                            arrivals.append(time.monotonic())
                    if reasons is not None and choice.get("finish_reason"):
                        reasons.append(choice["finish_reason"])


def _requests_running(base_url: str) -> float:
    """The requests the schedulers serve now.

    The gauge is ``uniserve:num_requests_running``, reported every second.
    """
    return metric(base_url, "uniserve:num_requests_running")


#: How often a case reads a stream it acts in the middle of: a kill or a switch
#: lands a few tokens after the count it waits for, not up to a second after.
STREAM_POLL_S = 0.01


@contextmanager
def _node(runtime: Path, device: str, log: Path) -> Iterator[tuple[str, int]]:
    """A node serving no deployment yet; its URL and process id. The node
    serves its own routes once its readiness record states it serves.
    """  # noqa: D205
    server = start_server(
        [
            str(require_uniserve_binary()),
            "serve",
            "--host",
            "127.0.0.1",
            *build_server_option(),
        ],
        log,
        env={"CUDA_VISIBLE_DEVICES": device, "XDG_RUNTIME_DIR": str(runtime)},
    )
    try:
        server.wait("serving", 60)
        yield server.base_url, server.process.pid
    finally:
        stop_server(server.process)
        server.close()


def _node_state(base_url: str) -> dict:
    """``GET /admin/node``: the node's state and its last deployment's error."""
    return admin_request(base_url, "GET", "/admin/node", timeout_s=10).json()


def _deploy(base_url: str, name: str, *assignments: str) -> None:
    response = admin_request(
        base_url,
        "PUT",
        "/admin/config",
        body={
            "config": "",
            "set": [
                f"model={qwen3_checkpoint()}",
                f"served_model_name={name}",
                f"kv_cache_config.host_gib={HOST_GIB}",
                *OPTIONS,
                *assignments,
            ],
        },
        timeout_s=60,
    )
    assert response.status_code == 202, response.text


def _settle(base_url: str, timeout_s: float = 900) -> dict:
    """Waits until the node serves or idles, and returns its state."""
    state: dict = {}

    def settled() -> bool:
        state.update(_node_state(base_url))
        return state["state"] in ("serving", "idle")

    wait_until(settled, timeout_s, "the node did not settle")
    return state


# It compares the host's shared memory before and after, so it runs alone.
@pytest.mark.host_wide
def test_a_node_serves_deployments_one_after_another_and_releases_their_storage(
    tmp_path: Path, runtime: Path
):
    device = granted_devices(1)
    with _node(runtime, device, tmp_path / "node.log") as (base_url, pid):
        gpu = _gpu_used_mib(device)
        print(f"before: GPU {gpu} MiB")
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
            regions = _host_region_gib(runtime)
            print(
                f"{name} serving: host regions {regions:.2f} GiB, "
                f"GPU {_gpu_used_mib(device)} MiB"
            )
            # The deployment's host regions: its host tier of `HOST_GIB`.
            assert regions >= HOST_GIB * 0.99, regions
            assert name in _storage_list(runtime)

            response = admin_request(
                base_url, "POST", "/admin/shutdown", body={}, timeout_s=60
            )
            assert response.status_code == 202, response.text
            state = _settle(base_url)
            assert state["state"] == "idle", state
            assert state["last_error"] is None, state
            after_regions = _host_region_gib(runtime)
            after_gpu = _gpu_used_mib(device)
            print(
                f"{name} down: host regions {after_regions:.2f} GiB, "
                f"GPU {after_gpu} MiB"
            )
            assert name not in _storage_list(runtime)
            assert after_regions == 0, after_regions
            assert after_gpu <= gpu + 1024, (after_gpu, gpu)

        # A host tier the host's available storage cannot hold.
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
            admin_request(base_url, "GET", "/admin/node", timeout_s=10)
        ).status_code == 200
        assert runs(pid)


@contextmanager
def _serving(
    runtime: Path,
    log: Path,
    name: str,
    env: dict[str, str] | None = None,
    options: tuple[str, ...] = OPTIONS,
    checkpoint: str | None = None,
) -> Iterator[tuple[str, int]]:
    """A deployment serving `name`, its URL and its daemon's process id."""
    command = [
        str(require_uniserve_binary()),
        "serve",
        checkpoint or qwen3_checkpoint(),
        "--served-model-name",
        name,
        "--host",
        "127.0.0.1",
        *engine_selection(),
        "--set",
        f"kv_cache_config.host_gib={HOST_GIB}",
        *build_server_option(),
    ]
    for option in options:
        command += ["--set", option]
    environment = {
        "CUDA_VISIBLE_DEVICES": granted_devices(1),
        "XDG_RUNTIME_DIR": str(runtime),
        **(env or {}),
    }
    with server_process(command, log, timeout_s=900, env=environment) as server:
        yield server.base_url, server.pid


@contextmanager
def _serve(
    runtime: Path,
    log: Path,
    name: str,
    env: dict[str, str] | None = None,
    options: tuple[str, ...] = OPTIONS,
    checkpoint: str | None = None,
) -> Iterator[str]:
    """A deployment serving `name` (:func:`_serving`); its URL."""
    with _serving(runtime, log, name, env, options, checkpoint) as (
        base_url,
        _,
    ):
        yield base_url


def _held_requests(base_url: str) -> int:
    """The new requests the daemon holds before an engine takes them."""
    return admin_status(base_url)["deployment"]["held"]


def _engine_pids(base_url: str) -> set[int]:
    """The process ids of the deployment's main processes."""
    return {process["pid"] for process in engine_processes(base_url)}


def _quota_reservations(base_url: str) -> list[dict]:
    """Every entry of the quotas ``GET /admin/config`` reports."""
    return [
        entry
        for node in admin_config(base_url)["quotas"]
        for quota in node["quotas"]
        for entry in quota["reservations"]
    ]


def _weight_bytes(base_url: str) -> int:
    """The bytes of weights the deployment's quotas reserve."""
    return sum(
        entry["bytes"]
        for entry in _quota_reservations(base_url)
        if entry["item"] == "weights"
    )


def _apply(base_url: str, *assignments: str) -> dict:
    """Applies `assignments` (``PUT /admin/config``); returns its result."""
    response = apply_config(base_url, *assignments)
    assert response.status_code == 200, response.text
    return response.json()["result"]


def _apply_settled(base_url: str, *assignments: str) -> None:
    """Applies `assignments` once the switch that ended left its replica."""
    answers: list[httpx.Response] = []

    def applied() -> bool:
        answers.append(apply_config(base_url, *assignments))
        return answers[-1].status_code != 409

    wait_until(applied, 120, "the replica stayed with the switch that ended")
    assert answers[-1].status_code == 200, answers[-1].text


def _change_epoch(base_url: str, *assignments: str) -> dict:
    """Applies `assignments` as an epoch switch and commits the target; returns
    the update's report.
    """  # noqa: D205
    result = _apply(base_url, *assignments)
    assert result["change"] == "epoch", result
    report = _on_trial(base_url)
    committed = admin(base_url, "config/commit")
    assert committed.status_code == 200, committed.text
    return report


def test_a_serving_deployment_answers_its_admin_api(
    tmp_path: Path, runtime: Path
):
    name = "qwen3-live-admin"
    log = tmp_path / "server.log"
    with _serve(runtime, log, name) as base_url:
        status = admin_status(base_url)
        assert status["deployment"]["deployment"] == name, status
        assert status["deployment"]["engine"] == "serving", status

        # New requests wait while generation is paused.
        paused = admin(base_url, "pause_generation")
        assert paused.status_code == 200, paused.text
        answered: dict = {}
        waiting = threading.Thread(
            target=lambda: answered.update(_chat(base_url, name, "Say hello."))
        )
        waiting.start()
        wait_until(
            lambda: _held_requests(base_url) >= 1,
            60,
            "the daemon reported no held request while generation paused",
        )
        assert not answered, "a request was served while generation paused"
        assert admin(base_url, "continue_generation").status_code == 200
        waiting.join(timeout=300)
        assert answered["choices"][0]["message"]["content"], answered

        # A flush evicts the device copies of the blocks the same prompt
        # committed: asked again, it finds none of them on the device,
        # whatever the host tier still holds.
        prompt = "Describe the sea in one sentence. " * 40
        _chat(base_url, name, prompt)
        cached = _device_tokens_read(
            base_url, lambda: _chat(base_url, name, prompt)
        )
        assert cached > 0, cached
        flushed = admin(base_url, "flush_cache")
        assert flushed.status_code == 200, flushed.text
        after_flush = _device_tokens_read(
            base_url, lambda: _chat(base_url, name, prompt)
        )
        assert after_flush == 0, after_flush

        # Scheduling limits change in place under a streaming request, and
        # other decode batch sizes by an epoch change under the same stream.
        read: list[str] = []
        done = threading.Event()
        streaming = threading.Thread(
            target=_stream_count, args=(base_url, name, read, done)
        )
        streaming.start()
        wait_until(lambda: len(read) >= 32, 300, "the count did not stream")
        update = _apply(base_url, "scheduler_config.max_batch=4")
        assert update["change"] == "generation", update
        committed = admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text
        before_update = len(read)
        report = _change_epoch(base_url, CHANGE)
        print(f"epoch switch: {report['switch_ms']} ms")
        streaming.join(timeout=1800)
        assert done.is_set(), "the count did not end"
        text = "".join(read)
        assert len(read) > before_update, "the stream ended before the update"
        numbers = [int(word) for word in re.findall(r"\d+", text)]
        assert numbers[:100] == list(range(1, 101)), text[:400]

        # The manager attaches to the node by its operator socket, follows
        # its events and detaches.
        manager = shutil.which("uniserve-manager") or str(
            Path(require_uniserve_binary()).with_name("uniserve-manager")
        )
        node = f"unix:{admin_socket(base_url)}"
        status_line = subprocess.run(
            [manager, "--node", node, "status"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
        assert name in status_line and "engine serving" in status_line, (
            status_line
        )
        watching = subprocess.Popen(
            [manager, "--node", node, "watch", "--from-start"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        printed: list[str] = []

        def read() -> None:
            assert watching.stdout is not None
            for line in watching.stdout:
                printed.append(line)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        # The watch replays the events from the start, the config
        # operations among them.
        wait_until(
            lambda: any("config" in line for line in printed),
            60,
            "the manager printed no config event",
            context=lambda: "".join(printed),
        )
        watching.send_signal(signal.SIGINT)
        watching.wait(timeout=60)
        reader.join(timeout=60)
        output = "".join(printed)
        assert watching.returncode == 0, output
        assert "detached; the nodes serve on" in output, output
        reply = _chat(base_url, name, "Say hello.")
        assert reply["choices"][0]["message"]["content"]

        events = admin_request(
            base_url, "GET", "/admin/events", params={"after": 0}, timeout_s=10
        ).json()
        kinds = [event["kind"] for event in events["events"]]
        assert "operation" in kinds, events


def _stop_pid(pid: int) -> None:
    """Stops daemon `pid` as SIGTERM stops a daemon, if it still runs."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    wait_until(
        lambda: not runs(pid),
        120,
        f"daemon {pid} did not stop",
    )


#: The streams a hot restart hands over: three counts of different lengths, so
#: their event boundaries fall at different points of the response, and the
#: number each stream must carry with no gap and no repetition.
HANDED_STREAMS = (200, 500, 900)


def test_a_hot_restart_hands_the_deployment_to_a_successor_daemon(
    tmp_path: Path, runtime: Path
):
    name = "qwen3-live-restart"
    log = tmp_path / "server.log"
    # The daemons a restart starts, which the fixture, stopping the daemon
    # it started, does not stop.
    successors: list[int] = []
    try:
        with _serving(runtime, log, name) as (
            base_url,
            daemon,
        ):
            _restart_while_streaming(base_url, name, log, daemon, successors)
    finally:
        for pid in successors:
            _stop_pid(pid)


def test_a_switch_after_a_hot_restart_finds_the_storage_of_the_epoch_before(
    tmp_path: Path, runtime: Path
):
    """A successor daemon learns what a target maps from the process that serves.

    The epoch that serves took its storage from the one before it, under that
    epoch's name. The daemon that switched it is gone, and the next target still
    maps that storage.
    """  # noqa: E501
    name = "qwen3-live-restart-switch"
    log = tmp_path / "server.log"
    revisions = [engine_revision() for _ in range(2)]
    successors: list[int] = []
    try:
        with _serve(runtime, log, name) as base_url:
            answer = _chat(base_url, name, RECOVERY_PROMPT, tokens=32)
            _change_epoch(base_url, revisions[0])
            _committed(base_url)
            restarted = admin(base_url, "restart")
            assert restarted.status_code == 200, (
                f"{restarted.text}{_log_trouble(log)}"
            )
            successors.append(restarted.json()["result"]["pid"])
            _chat(base_url, name, RECOVERY_PROMPT, tokens=8)
            _change_epoch(base_url, revisions[1])
            after = _chat(base_url, name, RECOVERY_PROMPT, tokens=32)
            assert (
                after["choices"][0]["message"]["content"]
                == answer["choices"][0]["message"]["content"]
            )
    finally:
        for pid in successors:
            _stop_pid(pid)


def _first_stream_frame(
    base_url: str, model: str, timeout_s: float = 120.0
) -> None:
    """Reads one streamed frame, or says why none arrived.

    A stream that produces nothing for five minutes and one that produces
    tokens slowly are the same failure to a test that waits for a count. One
    frame tells them apart in a minute: it proves the streaming path carries a
    frame end to end, and when it does not it reports the status, the content
    type and the opening bytes instead of a timeout with no cause.
    """
    with httpx.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Count from 1 to 20."}],
            "temperature": 0,
            "max_completion_tokens": 8,
            "stream": True,
        },
        timeout=timeout_s,
    ) as response:
        content_type = response.headers.get("content-type", "")
        if response.status_code != 200:
            raise AssertionError(
                "a streamed request was refused: "
                f"{response.status_code}: {response.read()[:400]!r}"
            )
        for line in response.iter_lines():
            if line.startswith("data: "):
                return
    raise AssertionError(
        f"the stream carried no frame in {timeout_s:.0f} s "
        f"(content-type {content_type!r})"
    )


def _restart_while_streaming(
    base_url: str,
    name: str,
    log: Path,
    old_daemon: int,
    successors: list[int],
) -> None:
    engines_before = _engine_pids(base_url)
    status = admin_status(base_url)
    assert status["deployment"]["engine"] == "serving", status

    # One completion before any stream: a deployment that answers nothing says
    # so here, in seconds and with its own body, instead of as a stream that
    # produced no token in five minutes — a silence the two paths share and
    # that therefore names neither. A refusal is the likelier failure (a
    # relaunched process that was not handed the epoch's modules answers 503
    # until it times out), and `raise_for_status` keeps the status but not the
    # body, so the refusal's own `detail` is raised here as well.
    probe = "Count from 1 to 3, separated by commas."
    # Three numbers from a deployment that is serving take seconds; two
    # minutes is generous and turns a deployment that answers nothing at all
    # into a failure this test names, instead of ten minutes of silence.
    try:
        answered = _chat(base_url, name, probe, timeout_s=120)
    except httpx.HTTPStatusError as error:
        raise AssertionError(
            "the deployment refused a completion: "
            f"{error.response.status_code} {error.response.text[:2000]}"
            f"{log_tail(log)}"
        ) from error
    choice = (answered.get("choices") or [{}])[0]
    content = choice.get("message", {}).get("content")
    assert content, (
        f"the deployment answered no content: {answered}{log_tail(log)}"
    )

    # And one frame of one stream: the three streams below each need 32 counts,
    # so a silent stream and a slow one are the same failure at 300 s. This
    # says which in a minute.
    try:
        _first_stream_frame(base_url, name)
    except AssertionError as error:
        raise AssertionError(f"{error}{log_tail(log)}") from error

    # Three streams in flight at the restart, of different lengths so their
    # boundaries fall at different events of different requests.
    reads: list[list[str]] = [[] for _ in HANDED_STREAMS]
    dones = [threading.Event() for _ in HANDED_STREAMS]
    # The finish reason each stream ends with: a response the successor closes
    # instead of finishing carries none, and the count alone would not say so.
    reasons: list[list[str]] = [[] for _ in HANDED_STREAMS]
    # What each stream's thread raised, if anything: a refusal or a timeout
    # otherwise reads as a count of zero, the same as a stall.
    errors: list[BaseException] = []
    streaming = [
        threading.Thread(
            target=_stream_count,
            args=(base_url, name, read, done, reason),
            kwargs={
                "prompt": (
                    f"Count from 1 to {last}, separated by commas and spaces. "
                    "Write only the numbers."
                ),
                "errors": errors,
            },
        )
        for read, done, reason, last in zip(
            reads, dones, reasons, HANDED_STREAMS, strict=True
        )
    ]
    for thread in streaming:
        thread.start()
    try:
        wait_until(
            lambda: all(len(read) >= 32 for read in reads),
            300,
            "the counts did not stream",
        )
    except TimeoutError as error:
        # Three streams open at once is the first place the deployment serves
        # more than one request at a time; a stop here is the engine's own
        # account of what it did with them, not a number of seconds.
        # Order is what survives: the harness keeps the end of this output, so
        # the verdict and the warnings/errors come last and the long tail
        # first. With the excerpts reversed, the last two runs' artifacts held
        # thirty lines of graph captures and none of the engine's death.
        raise AssertionError(
            f"{log_tail(log, 30)}{_log_trouble(log)}"
            f"\n{error} (counts so far: {[len(read) for read in reads]}; "
            f"stream errors: {[repr(one) for one in errors]})"
        ) from error
    restarted = admin(base_url, "restart")
    # A refusal here is the successor's own answer, and its reason names the
    # step that stopped; the daemons' log says what happened to the processes
    # the handover was carrying, which the answer cannot.
    assert restarted.status_code == 200, f"{restarted.text}{_log_trouble(log)}"
    successor = restarted.json()["result"]["pid"]
    successors.append(successor)
    assert successor != old_daemon
    before_restart = [len(read) for read in reads]

    # The successor serves the same deployment, so it holds the same modules'
    # server: a deployment reaches its scheduling policy only through it, and
    # the successor reads the configuration the old process served rather than
    # a command line of its own (design 7.5).
    inherited = tomllib.loads(admin_config(base_url)["config"])
    assert inherited["build_config"]["socket"] == str(BUILD_SERVER), (
        f"the successor did not inherit the modules' server: {inherited}"
    )

    # New requests are served while the old daemon finishes its streams.
    for _ in range(3):
        try:
            reply = _chat(base_url, name, "Say hello.")
        except Exception as error:  # noqa: BLE001 - the successor's answer, whole, with the log
            raise AssertionError(
                f"a new request after the restart failed: {error!r}"
                f"{_log_trouble(log)}"
            ) from error
        assert reply["choices"][0]["message"]["content"], (
            f"a new request after the restart answered an empty completion: "
            f"{reply}{_log_trouble(log)}"
        )
    for thread in streaming:
        thread.join(timeout=1800)
    assert all(done.is_set() for done in dones), "a stream did not end"
    for index, (read, done, last) in enumerate(
        zip(reads, dones, HANDED_STREAMS, strict=True)
    ):
        assert len(read) > before_restart[index], (
            f"stream {index} ended at the restart (reads={len(read)} of "
            f"{before_restart[index]} before it, done={done.is_set()}, "
            f"last={last}){_log_trouble(log)}"
        )
        # Every stream carries its count whole: a frame lost, repeated or
        # reordered at the boundary — the one thing the handoff must not do —
        # breaks the sequence, and the boundary lies inside it, not after the
        # first two hundred numbers.
        numbers = [int(word) for word in re.findall(r"\d+", "".join(read))]
        assert numbers == list(range(1, len(numbers) + 1)), (
            f"stream {index} broke: {''.join(read)[:400]}"
        )
        # `before_restart` counts frames, and a frame need not hold a number.
        before = re.findall(r"\d+", "".join(read[: before_restart[index]]))
        assert len(numbers) > len(before), (
            f"stream {index} carried no number after the restart"
        )
        assert numbers and numbers[-1] >= last // 2, (
            f"stream {index} stopped at {numbers[-1] if numbers else 0}"
        )
        # It finished rather than being closed: the successor's last chunk
        # names why it stopped.
        assert reasons[index], f"stream {index} ended without a finish reason"
        assert reasons[index][-1] in ("stop", "length"), reasons[index]

    # A one-body chat response in flight at a further restart is written by
    # its successor too, whole (design 7.5.1: every response in flight moves).
    _restart_with_a_json_body(base_url, name, log, successors)

    # The old daemon exits once its requests ended; the engine and its
    # workers are the same processes throughout.
    wait_until(
        lambda: not runs(old_daemon),
        120,
        "the replaced daemon did not exit",
    )
    # The one-body restart replaced `successor` in turn, so the daemon that
    # serves now is the last of them.
    assert runs(successors[-1])
    status = admin_status(base_url)
    assert status["deployment"]["engine"] == "serving", status
    assert _engine_pids(base_url) == engines_before
    events = admin_request(
        base_url, "GET", "/admin/events", params={"after": 0}, timeout_s=10
    ).json()
    messages = [event["message"] for event in events["events"]]
    assert any("after a hot restart" in message for message in messages), (
        messages
    )

    # The successor restarts in turn, and takes a change in place.
    again = admin(base_url, "restart")
    assert again.status_code == 200, again.text
    successors.append(again.json()["result"]["pid"])
    wait_until(
        lambda: not runs(successor),
        120,
        "the first successor did not exit",
    )
    reply = _chat(base_url, name, "Say hello.")
    assert reply["choices"][0]["message"]["content"]
    # A scheduling limit is a parameter of the policy package, so it reaches
    # the serving epoch as a generation update; `max_running_requests` is build
    # config and would switch the epoch.
    update = _apply(base_url, "scheduler_config.max_batch=4")
    assert update["change"] == "generation", update
    committed = admin(base_url, "config/commit")
    assert committed.status_code == 200, committed.text

    # A change of the listening address restarts the daemon on it: the new
    # address serves, and the old one closes once the replaced daemon exits.
    previous = successors[-1]
    moved = _apply(base_url, "admin_config.host=localhost")
    assert moved["change"] == "daemon", moved
    successors.append(moved["pid"])
    # The answer comes from the daemon on the previous address, and the
    # successor binds a new address, on a port it picks, once it holds the
    # deployment; its readiness record states it.
    moved_url = taken_over(base_url, moved["pid"])
    reply = _chat(moved_url, name, "Say hello.")
    assert reply["choices"][0]["message"]["content"]
    wait_until(
        lambda: not runs(previous),
        120,
        "the daemon on the previous address did not exit",
    )
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"{base_url}/health", timeout=5)
    status = admin_config(moved_url)
    server = tomllib.loads(status["config"])["admin_config"]
    assert server["host"] == "localhost", server

    # A change of both the daemon's options and what an epoch executes
    # restarts the daemon on its new options first, and its successor then
    # changes the epoch in an operation of its own.
    previous = successors[-1]
    both = _apply(moved_url, "admin_config.host=127.0.0.1", CHANGE)
    assert both["change"] == "daemon", both
    assert both["then"] == "update", both
    successors.append(both["pid"])
    both_url = taken_over(moved_url, both["pid"])

    def changed() -> dict | None:
        status = admin_status(both_url)
        last = status["last_operation"]
        if status["operation"] is None and last and last["kind"] == "config":
            return last
        return None

    wait_until(lambda: changed() is not None, 900, "the epoch did not change")
    last = changed()
    assert last["state"] == "succeeded", last
    assert last["result"]["change"] == "epoch", last
    committed = admin(both_url, "config/commit")
    assert committed.status_code == 200, committed.text

    def served_config() -> dict:
        return tomllib.loads(admin_config(both_url)["config"])

    # A commit may be asked while the target is still being promoted, and it is
    # applied when the trial starts: the configuration moves once the target
    # serves.
    wait_until(
        lambda: (
            served_config()["cuda_graph_config"]["decode_batch_sizes"]
            == [1, 2, 8]
        ),
        300,
        "the switch's target never served its configuration",
    )
    assert served_config()["admin_config"]["host"] == "127.0.0.1"
    reply = _chat(both_url, name, "Say hello.")
    assert reply["choices"][0]["message"]["content"]
    wait_until(
        lambda: not runs(previous),
        120,
        "the daemon on the previous address did not exit",
    )


#: The prompt of the one-body request the restart test hands over: long enough
#: that its generation is certainly in flight when the restart pauses it.
JSON_PROMPT = (
    "Count from 1 to 1500, separated by commas and spaces. "
    "Write only the numbers."
)


def _restart_with_a_json_body(
    base_url: str, name: str, log: Path, successors: list[int]
) -> None:
    """A non-streaming chat body in flight across a restart is written whole.

    The body is one response built from a collector's state, so a field lost in
    the crossing is silent: the request still answers 200 with a response that
    parses. What catches it is a body that finished, whose text is a contiguous
    count and whose usage adds up.
    """
    bodies: list[dict] = []
    failures: list[BaseException] = []

    def collect() -> None:
        try:
            bodies.append(_chat(base_url, name, JSON_PROMPT, tokens=8192))
        except BaseException as error:  # noqa: BLE001 - a failure is the test's
            failures.append(error)

    request = threading.Thread(target=collect)
    request.start()
    # The restart pauses what the engine serves, so it is issued only once the
    # one-body request is being served: the schedulers say so, not a sleep.
    wait_until(
        lambda: _requests_running(base_url) >= 1,
        120,
        "the one-body request was never served",
    )
    restarted = admin(base_url, "restart")
    # A refusal here is the successor's own answer, and its reason names the
    # step that stopped; the daemons' log says what happened to the processes
    # the handover was carrying, which the answer cannot.
    assert restarted.status_code == 200, f"{restarted.text}{_log_trouble(log)}"
    successor = restarted.json()["result"]["pid"]
    successors.append(successor)
    request.join(timeout=1800)
    assert not failures, failures
    assert bodies, "the one-body request answered nothing"
    body = bodies[0]
    assert body["choices"][0]["finish_reason"] not in (None, "migratable"), body
    text = body["choices"][0]["message"]["content"]
    numbers = [int(word) for word in re.findall(r"\d+", text)]
    assert numbers == list(range(1, len(numbers) + 1)), text[:400]
    assert len(numbers) > 100, text[:400]
    usage = body["usage"]
    assert usage["prompt_tokens"] > 0, usage
    assert usage["completion_tokens"] >= len(numbers), usage
    assert (
        usage["total_tokens"]
        == usage["prompt_tokens"] + usage["completion_tokens"]
    ), usage


def _storage_pid(runtime: Path) -> int:
    """The process id of the Storage Server serving the test's socket, as
    its readiness record states it.
    """  # noqa: D205
    return storage_status(runtime)["pid"]


def test_a_storage_server_restart_keeps_the_deployment_serving(
    tmp_path: Path, runtime: Path
):
    name = "qwen3-live-storage"
    log = tmp_path / "server.log"
    with _serve(runtime, log, name) as base_url:
        prompt = "Describe the sea in one sentence. " * 40
        _chat(base_url, name, prompt)
        before = _held(runtime)
        old_server = _storage_pid(runtime)

        read: list[str] = []
        done = threading.Event()
        streaming = threading.Thread(
            target=_stream_count, args=(base_url, name, read, done)
        )
        streaming.start()
        wait_until(lambda: len(read) >= 32, 300, "the count did not stream")
        restarted = subprocess.run(
            [str(require_uniserve_binary()), "storage-server", "restart"],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "XDG_RUNTIME_DIR": str(runtime)},
            timeout=120,
        ).stdout
        successor = int(
            re.search(r"successor server (\d+)", restarted).group(1)
        )
        wait_until(
            lambda: not runs(old_server),
            120,
            "the replaced Storage Server did not exit",
        )
        assert _storage_pid(runtime) == successor
        assert _held(runtime) == before

        # The deployment serves throughout: the count goes on, a new
        # request is answered, and an epoch change maps the weights the
        # successor holds.
        reply = _chat(base_url, name, prompt)
        assert reply["choices"][0]["message"]["content"]
        streaming.join(timeout=1800)
        assert done.is_set(), "the count did not end"
        numbers = [int(word) for word in re.findall(r"\d+", "".join(read))]
        assert numbers[:200] == list(range(1, 201)), "".join(read)[:400]
        weights = _weight_bytes(base_url)
        _change_epoch(base_url, CHANGE)
        reply = _chat(base_url, name, "Say hello.")
        assert reply["choices"][0]["message"]["content"]
        # The quota reserves a storage once however many epochs map it, so a
        # target that read the checkpoint again would reserve its weights
        # beside these.
        assert _weight_bytes(base_url) == weights


# The options of a deployment whose KV cache takes the device storage its
# share leaves, rather than a fixed token count.
SIZED_OPTIONS = tuple(
    option
    for option in OPTIONS
    if not option.startswith("kv_cache_config.max_tokens")
)


@contextmanager
def _ballast(device: str, leave_gib: float) -> Iterator[int]:
    """Hold GPU `device`'s free storage but `leave_gib` GiB in a process.

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
        # Closing its input ends the process, which frees the storage.
        assert process.stdin is not None
        process.stdin.close()
        process.wait(timeout=120)


def test_a_target_that_finds_no_storage_leaves_the_serving_epoch_serving(
    tmp_path: Path, runtime: Path
):
    """A target beside a full GPU ends the update and changes nothing that serves.

    A target prepares beside the serving epoch, which keeps its regions until
    it drains: the target starts with what the GPU's quota has free. With two
    GiB left on the GPU it finds no room for its graph pool, so its start
    fails, the update is abandoned before any switch (the apply answers
    ``409`` with the abandoned update, ``docs/admin.md``), and the stream goes
    on with the serving epoch's regions as they were.
    """  # noqa: E501
    name = "qwen3-live-pressure"
    log = tmp_path / "server.log"
    device = granted_devices(1)
    prompt = "Describe the sea in one sentence."
    with _serve(runtime, log, name, options=SIZED_OPTIONS) as base_url:
        before = _chat(base_url, name, prompt, 32)["choices"][0]["message"][
            "content"
        ]
        serving = admin_config(base_url)["executing"]["epoch"]
        cache = _cache_bytes(base_url, serving)
        read: list[str] = []
        done = threading.Event()
        streaming = threading.Thread(
            target=_stream_count, args=(base_url, name, read, done)
        )
        streaming.start()
        wait_until(lambda: len(read) >= 32, 300, "the count did not stream")
        with _ballast(device, leave_gib=2.0) as held:
            print(f"ballast holds {held / (1 << 30):.1f} GiB; GPU free 2 GiB")
            # The apply waits until the update reached its trial or ended
            # before it: abandoned, it answers 409 with the update.
            response = apply_config(base_url, CHANGE)
            assert response.status_code == 409, response.text
            assert response.json()["result"]["change"] == "epoch", response.text
            wait_until(
                lambda: update_report(base_url).get("phase") == "abandoned",
                600,
                "the update did not end",
                context=lambda: _log_trouble(log),
            )
            report = update_report(base_url)
            print("epoch switch under pressure:", json.dumps(report))
            # The target's start failed before it joined: a failed
            # preparation, whose reason names the storage it did not find.
            assert report["failure"]["kind"] == "prepare_failed", report
            assert "graph pool" in report["failure"]["reason"], report
        streaming.join(timeout=1800)
        assert done.is_set(), "the count did not end"
        numbers = [int(word) for word in re.findall(r"\d+", "".join(read))]
        assert numbers[:200] == list(range(1, 201)), "".join(read)[:400]
        # The serving epoch's quota reserves at least the KV cache it held.
        assert admin_config(base_url)["executing"]["epoch"] == serving
        assert _cache_bytes(base_url, serving) >= cache, (
            "the abandoned update shrank the serving epoch's KV regions"
        )
        after = _chat(base_url, name, prompt, 32)["choices"][0]["message"][
            "content"
        ]
        assert after == before


def _cache_bytes(base_url: str, epoch: int) -> int:
    """The bytes of KV cache segments `epoch` holds on the quotas."""
    return sum(
        entry["bytes"]
        for entry in _quota_reservations(base_url)
        if entry["item"] == "cache_segment" and epoch in entry["epochs"]
    )


def test_a_sleeping_deployment_gives_back_its_cache_and_graph_storage(
    tmp_path: Path, runtime: Path
):
    name = "qwen3-live-sleep"
    log = tmp_path / "server.log"
    device = granted_devices(1)
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
        wait_until(lambda: len(read) >= 32, 300, "the count did not stream")

        slept = admin(base_url, "sleep")
        assert slept.status_code == 200, slept.text
        status = admin_status(base_url)
        assert status["deployment"]["asleep"] is True, status
        asleep = _gpu_used_mib(device)
        answered: dict = {}
        waiting = threading.Thread(
            target=lambda: answered.update(_chat(base_url, name, prompt, 32))
        )
        waiting.start()
        wait_until(
            lambda: _held_requests(base_url) >= 1,
            60,
            "the daemon reported no held request while the engine slept",
        )
        assert not answered, "a sleeping deployment served a request"

        woke = admin(base_url, "wake_up")
        assert woke.status_code == 200, woke.text
        waiting.join(timeout=300)
        assert answered["choices"][0]["message"]["content"], answered
        streaming.join(timeout=1800)
        assert done.is_set(), "the count did not end"
        numbers = [int(word) for word in re.findall(r"\d+", "".join(read))]
        assert numbers[:200] == list(range(1, 201)), "".join(read)[:400]
        after = _gpu_used_mib(device)
        print(
            f"GPU storage: awake {awake} MiB, asleep {asleep} MiB, "
            f"woken {after} MiB"
        )
        assert awake - asleep >= 10 * 1024, (awake, asleep)
        # Alone, as before the sleep, the deployment answers as it did: the
        # request that waited ran beside the recomputed count, whose batch
        # changes the numerics.
        alone = _chat(base_url, name, prompt, 32)
        assert alone["choices"][0]["message"]["content"] == before, alone
        assert abs(after - awake) <= 2 * 1024, (awake, after)
        status = admin_status(base_url)
        assert status["deployment"]["asleep"] is False, status
        assert status["deployment"]["paused"] is False, status


#: How long a lost main process has to put an engine back in front of requests
#: before the test calls it a stall. The design states no recovery time, and a
#: migrating request waits in the hold until an engine serves; the test's
#: deployment captures few graphs, so its relaunch ends well inside this bound:
#: what a case measures is recorded, and the bound is only ever a stall bound.
RECOVERY_BOUND_S = 120.0

#: The count a recovery case streams across its kill: long enough that the kill
#: lands in the middle of the response, short enough to be compared whole.
#:
#: A count to 60 is about 230 tokens, which a single request on one GB200 now
#: streams in under a second, so a kill made after the eighth number landed
#: after the response had ended and no request was left to migrate. A count to
#: 300 streams for several seconds, which the kill and a switch both land
#: inside.
RECOVERY_PROMPT = (
    "Count from 1 to 300, separated by commas and spaces. "
    "Write only the numbers."
)

#: The count a request asks for when its tokens are compared: the same prompt
#: every time, so the same numbers are the same answer to the same question.
RECOVERY_TOKENS = 256


def _numbers(read: list[str]) -> list[int]:
    """The integers a streamed answer wrote, in order."""
    return [int(word) for word in re.findall(r"\d+", "".join(read))]


def _lose_the_serving_engine(
    base_url: str,
    log: Path,
    name: str,
    case: str,
    standby: bool,
) -> dict:
    """Kills the engine that serves under a request in flight.

    The request in flight is a count: it must carry every number it wrote
    before the loss and every number after it, with no gap and no repetition,
    because the response migrates to the engine that takes over. The answer to
    the same question after the recovery must be the answer the deployment gave
    before it — that is what says the surviving engine serves the same model and
    not merely a live one.
    """
    if standby:
        # The standby is what takes over, so the loss waits for it to stand
        # by with its graphs.
        wait_until(
            lambda: _prepared_standby(base_url) is not None,
            600,
            "the deployment reported no prepared standby",
            context=lambda: _log_trouble(log),
        )
    answer = _chat(base_url, name, RECOVERY_PROMPT, tokens=RECOVERY_TOKENS)
    baseline = answer["choices"][0]["message"]["content"]
    baseline_numbers = _numbers([baseline])
    # The prompt's own token count is in the answer, and it is what separates a
    # deployment that reads the question wrongly from one that answers wrongly:
    # this sentence is about twenty tokens long, so a `prompt_tokens` of a
    # handful says the request never reached the model as text it could read.
    assert baseline_numbers[:10] == list(range(1, 11)), (
        f"{baseline[:200]!r} usage={answer.get('usage')}"
    )

    read: list[str] = []
    done = threading.Event()
    errors: list[BaseException] = []
    arrivals: list[float] = []
    reasons: list[str] = []
    streaming = threading.Thread(
        target=_stream_count,
        args=(base_url, name, read, done),
        kwargs={
            "reasons": reasons,
            "prompt": RECOVERY_PROMPT,
            "errors": errors,
            "arrivals": arrivals,
        },
    )
    streaming.start()
    wait_until(
        lambda: len(_numbers(read)) >= 8,
        300,
        "the count did not stream",
        STREAM_POLL_S,
    )
    before = len(_numbers(read))
    serving = admin_status(base_url)["deployment"]
    standby_generations = serving["standby"]
    killed = serving_engine(base_url)["pid"]
    started = time.monotonic()
    _stop_pid(killed)

    resumed: list[float] = []

    def continues() -> bool:
        if len(_numbers(read)) > before:
            resumed.append(time.monotonic())
            return True
        return False

    wait_until(
        continues,
        RECOVERY_BOUND_S,
        f"the response in flight did not continue within "
        f"{RECOVERY_BOUND_S:.0f} s of the engine {killed} ending",
    )

    streaming.join(timeout=1800)
    # What the client lived through: the longest silence of the stream from the
    # last token before the loss on. The arrival of each token is its own
    # reading, so this is exact to the token; a poll of the count would round up
    # to its interval, and tokens already on their way would read as a recovery.
    last_before = [moment for moment in arrivals if moment < started][-1:]
    after = [moment for moment in arrivals if moment >= started]
    spans = last_before + after
    recovery_s = max(
        (later - earlier for earlier, later in zip(spans, spans[1:])),
        default=resumed[0] - started,
    )
    # The measurement is taken before the tokens are judged: a run whose served
    # path is broken still states how long the recovery took, which is the one
    # number this case exists to record.
    evidence_time = round(recovery_s, 3)
    assert done.is_set(), "the count did not end after the recovery"
    assert not errors, f"the count failed: {errors[0]!r}"
    numbers = _numbers(read)
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"the response lost or repeated a count across the loss: {numbers[:40]}"
    )
    assert len(numbers) >= 300, f"the count stopped at {len(numbers)} numbers"
    assert reasons == ["stop"], f"the count ended {reasons}"

    after = _chat(base_url, name, RECOVERY_PROMPT, tokens=RECOVERY_TOKENS)[
        "choices"
    ][0]["message"]["content"]
    assert _numbers([after]) == baseline_numbers, (
        "the engine that serves after the loss answers differently: "
        f"{after[:200]}"
    )

    # The main process that serves now: the standby's, promoted, or one
    # launched anew.
    generation = admin_status(base_url)["deployment"]["generation"]
    promoted = bool(standby_generations) and (
        generation == standby_generations[0]
    )
    relaunched = generation not in (
        None,
        serving["generation"],
        *standby_generations,
    )
    assert promoted or relaunched, (
        f"no engine took the lost one's place: {_log_trouble(log)}"
    )
    expected = promoted if standby else relaunched
    assert expected, (
        f"a deployment with standby={standby} recovered by "
        f"promoted={promoted} relaunched={relaunched}:\n{_log_trouble(log)}"
    )

    evidence = {
        "case": case,
        "standby": standby,
        "engine_lost": killed,
        "recovery_s": evidence_time,
        "promoted": promoted,
        "relaunched": relaunched,
        "numbers_across_the_loss": len(numbers),
        "answer_matches_the_answer_before": True,
        "recovery_bound_s": RECOVERY_BOUND_S,
    }
    print(f"{case}: {json.dumps(evidence, sort_keys=True)}")
    return evidence


def _prepared_standby(base_url: str) -> list[int] | None:
    """The launch generations of the parked standby, which parks once it
    captured its graphs; `None` while there is none.
    """  # noqa: D205
    return admin_status(base_url)["deployment"]["standby"] or None


def test_a_lost_main_process_is_promoted_from_its_standby(
    tmp_path: Path, runtime: Path
):
    """A deployment with a standby loses its serving process and serves on.

    The standby stands beside it for exactly this: it is promoted in place,
    with the graphs it captured at their addresses, so the requests in flight
    and the ones after them are answered by the process that was already
    parked rather than by one that starts now.
    """
    name = "qwen3-live-standby"
    log = tmp_path / "server.log"
    options = (*OPTIONS, "worker_config.standby=true")
    with _serve(runtime, log, name, options=options) as base_url:
        evidence = _lose_the_serving_engine(
            base_url, log, name, "standby-promoted", standby=True
        )
        assert evidence["promoted"]


def test_a_lost_main_process_is_relaunched_without_a_standby(
    tmp_path: Path, runtime: Path
):
    """The same loss with no standby beside the serving process.

    The daemon launches a replacement and hands it the epoch's own modules and
    the requests in flight: the count migrates to the new process, which answers
    the next request as the lost one did.
    """
    name = "qwen3-live-relaunched"
    log = tmp_path / "server.log"
    with _serve(runtime, log, name) as base_url:
        evidence = _lose_the_serving_engine(
            base_url, log, name, "relaunched", standby=False
        )
        assert evidence["relaunched"]


def test_a_standby_started_after_a_switch_maps_what_its_epoch_serves_from(
    tmp_path: Path, runtime: Path
):
    """A standby of a target maps storage another epoch created.

    A target serves from the storage of the epoch before it, under that epoch's
    name. The standby started beside the target finds it under the same name,
    and takes over when the target's process is lost.
    """
    name = "qwen3-live-standby-switch"
    log = tmp_path / "server.log"
    options = (*OPTIONS, "worker_config.standby=true")
    with _serve(runtime, log, name, options=options) as base_url:
        wait_until(
            lambda: _prepared_standby(base_url) is not None,
            600,
            "the first epoch's standby did not start",
        )
        first = _prepared_standby(base_url)
        result = _apply(base_url, engine_revision())
        assert result["change"] == "epoch", result
        _on_trial(base_url)
        committed = admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text
        _committed(base_url)
        wait_until(
            lambda: _prepared_standby(base_url) not in (None, first),
            600,
            "the target's standby did not start",
            context=lambda: _log_trouble(log),
        )
        evidence = _lose_the_serving_engine(
            base_url, log, name, "standby-after-a-switch", standby=True
        )
        assert evidence["promoted"]


def test_a_relaunched_epoch_hands_its_own_storage_to_the_next_target(
    tmp_path: Path, runtime: Path
):
    """A process started again keys the storage it creates by its own epoch.

    The epoch that replaces it maps that storage. So it does after a target's
    process is lost, when the first epoch's storage is gone with it, and after a
    rollback that started the epoch again on the weights it kept.
    """
    name = "qwen3-live-relaunched-storage"
    log = tmp_path / "server.log"
    revisions = [engine_revision() for _ in range(3)]
    with _serve(runtime, log, name) as base_url:
        answer = _chat(base_url, name, RECOVERY_PROMPT, tokens=32)
        result = _apply(base_url, revisions[0])
        assert result["change"] == "epoch", result
        _on_trial(base_url)
        committed = admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text
        _committed(base_url)
        _lose_the_serving_engine(
            base_url, log, name, "relaunched-after-a-switch", standby=False
        )
        result = _apply(
            base_url,
            revisions[1],
            "update_config.previous_epoch=weights",
        )
        assert result["change"] == "epoch", result
        _on_trial(base_url)
        rolled = admin(base_url, "config/rollback", {"reason": "the check"})
        assert rolled.status_code == 200, rolled.text
        wait_until(
            lambda: update_report(base_url).get("phase") == "rolled_back",
            300,
            "the rollback did not end",
            context=lambda: _log_trouble(log),
        )
        _apply_settled(base_url, revisions[2])
        _on_trial(base_url)
        committed = admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text
        after = _chat(base_url, name, RECOVERY_PROMPT, tokens=32)
        assert (
            after["choices"][0]["message"]["content"]
            == answer["choices"][0]["message"]["content"]
        )


def test_an_epoch_switch_to_a_new_executable_fails_no_request(
    tmp_path: Path, runtime: Path
):
    """A deployment switches to another worker executable under a request.

    The executable is a copy of the worker the test runs, so the switch replaces
    every main process of the deployment with a process of a different file. A
    request in flight crosses the switch with its count whole and nothing of it
    fails: an epoch switch is not a restart of the service the client sees.
    """
    name = "qwen3-live-executable"
    log = tmp_path / "server.log"
    with _serve(runtime, log, name) as base_url:
        before = _engine_pids(base_url)
        failed = _failed_requests(base_url)
        baseline = _chat(
            base_url, name, RECOVERY_PROMPT, tokens=RECOVERY_TOKENS
        )["choices"][0]["message"]["content"]
        baseline_numbers = _numbers([baseline])

        read: list[str] = []
        done = threading.Event()
        errors: list[BaseException] = []
        reasons: list[str] = []
        streaming = threading.Thread(
            target=_stream_count,
            args=(base_url, name, read, done),
            kwargs={
                "reasons": reasons,
                "prompt": RECOVERY_PROMPT,
                "errors": errors,
            },
        )
        streaming.start()
        wait_until(
            lambda: len(_numbers(read)) >= 8,
            300,
            "the count did not stream",
            STREAM_POLL_S,
        )
        report = _change_epoch(base_url, engine_revision())
        assert report, "the epoch switch reported no epoch"
        for pid in before:
            wait_until(
                lambda pid=pid: not runs(pid),
                RECOVERY_BOUND_S,
                f"engine {pid} outlived the switch to the new executable",
            )

        streaming.join(timeout=1800)
        assert done.is_set(), "the count did not end after the switch"
        assert not errors, f"the count failed: {errors[0]!r}"
        numbers = _numbers(read)
        assert numbers == list(range(1, len(numbers) + 1)), (
            f"the response lost or repeated a count across the switch: "
            f"{numbers[:40]}"
        )
        assert reasons == ["stop"], f"the count ended {reasons}"

        after = _chat(base_url, name, RECOVERY_PROMPT, tokens=RECOVERY_TOKENS)[
            "choices"
        ][0]["message"]["content"]
        assert _numbers([after]) == baseline_numbers, (
            "the new executable answers differently from the one it replaced: "
            f"{after[:200]}"
        )
        assert _failed_requests(base_url) == failed, "a request failed"


def _failed_requests(base_url: str) -> float:
    """The requests that ended in an error so far."""
    return metric(
        base_url, "uniserve:request_success", 'finished_reason="error"'
    )


# The span a trial's step time and throughput are averaged over.
TRIAL_WINDOW_S = 5


#: How many times the slower attention package (:func:`_slower_target`) runs
#: trtllm-gen's decode attention per call.
ATTENTION_RUNS = 12


def _slower_target(directory: Path) -> tuple[str, ...]:
    """Installs what makes an epoch step slower under the same stream.

    The target is a new revision of the tree's engine whose decode attention
    is a package of its own: the tree's trtllm-gen paged attention package
    (``crates/ops/kernels/paged_attention/trtllm``) under the backend name
    ``trtllm_repeated``, whose entry runs the kernel ``ATTENTION_RUNS`` times
    per call. Every run reads the same inputs and writes the same output, so
    the target computes the same tokens, later, as a release that lost a fast
    path would serve them, with no fault to report. On one GB200 serving this
    file's deployment, twelve runs made a step of one count stream 69% longer
    (15.2 and 15.3 ms of batch time against 8.9 to 9.1 ms for the tree's
    package before and after, points of 2000 tokens each), about 0.57 ms per
    run past the first. A key whose backend no deployment names takes the
    package whose name sorts first among those of one priority, ``trtllm``,
    and the engine revision is what makes the change an epoch (a package
    selection alone is a generation update).

    Both modules are built here, for the decode keys the serving deployment's
    attention runs: an update's preparation then builds nothing, and a stream
    started after this still runs when the switch comes.

    Returns:
        The assignments that select the target.
    """
    socket = build_server_socket()
    binary = str(require_uniserve_binary())
    engine = install_engine(socket)
    engine_executable(socket, engine)

    package = directory / "trtllm-repeated"
    shutil.copytree(
        Path.cwd() / "crates/ops/kernels/paged_attention/trtllm", package
    )
    manifest = package / "package.toml"
    manifest.write_text(
        manifest.read_text().replace(
            'backend = "trtllm"', 'backend = "trtllm_repeated"', 1
        )
    )
    entry = package / "src/entry.cpp"
    launch = "    runner.run(params);\n"
    source = entry.read_text()
    assert source.count(launch) == 1, "the entry's launch moved"
    entry.write_text(
        source.replace(
            launch,
            f"    for (int run = 0; run < {ATTENTION_RUNS}; ++run) {{\n"
            f"  {launch}"
            "    }\n",
        )
    )
    # A kernel package installs as an engine package does.
    revision = install_engine(socket, package)

    # `list` heads each package revision with `<op> <backend> revision <n>`
    # and follows it with its modules, one `<target> <key> <form>` line each.
    listed = subprocess.run(
        [binary, "build-server", "list", "--socket", str(socket)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    decode: set[tuple[str, str]] = set()
    package_of = None
    for line in listed.splitlines():
        if not line.startswith(" "):
            package_of = line.split()[:2]
        elif package_of == ["paged_attention", "trtllm"]:
            target, key, _ = line.split()
            if key.startswith("phase_decode_"):
                decode.add((target, key))
    assert decode, f"the deployment's attention built no decode key:\n{listed}"
    for target, key in sorted(decode):
        subprocess.run(
            [
                binary,
                "build-server",
                "resolve",
                "--socket",
                str(socket),
                "--target",
                target,
                "--key",
                key,
                "paged_attention",
                "trtllm_repeated",
                str(revision),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    return (
        f"worker_config.engine_revision={engine}",
        "decode_attention_backend=trtllm_repeated",
    )


def _start_count(base_url: str, name: str) -> tuple[threading.Thread, dict]:
    """Starts streaming the count; returns its thread and what it reads."""
    streamed: dict = {
        "read": [],
        "done": threading.Event(),
        "errors": [],
        "reasons": [],
    }
    streaming = threading.Thread(
        target=_stream_count,
        args=(base_url, name, streamed["read"], streamed["done"]),
        kwargs={"reasons": streamed["reasons"], "errors": streamed["errors"]},
    )
    streaming.start()
    wait_until(
        lambda: len(_numbers(streamed["read"])) >= 8,
        300,
        "the count did not stream",
    )
    return streaming, streamed


def _assert_whole(streaming: threading.Thread, streamed: dict) -> int:
    """Waits for the count's end; it lost and repeated nothing."""
    streaming.join(timeout=1800)
    assert streamed["done"].is_set(), "the count did not end"
    assert not streamed["errors"], (
        f"the count failed: {streamed['errors'][0]!r}"
    )
    numbers = _numbers(streamed["read"])
    wrong = next(
        (index for index, number in enumerate(numbers) if number != index + 1),
        None,
    )
    if wrong is not None:
        pieces = streamed["read"]
        text = "".join(pieces)
        at = text.find(f"{wrong}, ")
        pytest.fail(
            f"the response lost or repeated a count at {wrong}: "
            f"{text[max(at - 60, 0) : at + 80]!r}, deltas "
            f"{list(pieces[3 * wrong - 12 : 3 * wrong + 12])}"
        )
    assert len(streamed["reasons"]) == 1, (
        f"the count ended {streamed['reasons']}"
    )
    return len(numbers)


def _host_tokens_read(base_url: str, ask: Callable[[], object]) -> float:
    """The prompt tokens `ask` loads from the host tier."""
    return tokens_found(base_url, ask)[1]["host_hits"]


def _device_tokens_read(base_url: str, ask: Callable[[], object]) -> float:
    """The prompt tokens `ask` finds cached on the device."""
    found = tokens_found(base_url, ask)[1]
    return found["hits"] - found["host_hits"]


def _committed(base_url: str) -> None:
    """Waits until the committed switch released its previous epoch."""
    wait_for_phase(base_url, "committed", timeout_s=300)


def _on_trial(base_url: str) -> dict:
    """Waits until the update in flight stands on trial."""
    return wait_for_phase(base_url, "trial", timeout_s=600)


@pytest.mark.parametrize("previous_epoch", ["parked", "weights", "host"])
def test_an_operator_rollback_returns_to_the_previous_epoch_under_a_request(
    tmp_path: Path, runtime: Path, previous_epoch: str
):
    """A rollback by hand ends the trial with no request lost.

    The previous epoch is kept as ``update_config.previous_epoch`` states: its
    drained processes stay alive through the trial, or they exit with its
    weights kept, and the rollback starts them again on those.
    """
    name = "qwen3-live-rollback"
    log = tmp_path / "server.log"
    with _serve(runtime, log, name) as base_url:
        before = _engine_pids(base_url)
        previous = admin_config(base_url)["executing"]["epoch"]
        answer = _chat(base_url, name, RECOVERY_PROMPT, tokens=32)
        streaming, streamed = _start_count(base_url, name)
        result = _apply(
            base_url,
            engine_revision(),
            f"update_config.previous_epoch={previous_epoch}",
        )
        assert result["change"] == "epoch", result
        trial = _on_trial(base_url)
        if previous_epoch != "parked":
            for pid in before:
                wait_until(
                    lambda pid=pid: not runs(pid),
                    120,
                    f"engine {pid} outlived the promotion",
                )
        else:
            assert all(runs(pid) for pid in before)

        # A node that cannot move the weights keeps them on its GPUs, and says
        # why.
        def kept_as_stated() -> bool:
            report = update_report(base_url)
            return (
                previous_epoch == "parked"
                or report["moved_to_host_bytes"] > 0
                or report["previous_epoch_kept"] == previous_epoch
            )

        wait_until(
            kept_as_stated,
            120,
            "the previous epoch's storage was not kept",
        )
        trial = update_report(base_url)
        print("kept:", json.dumps(trial))
        if not trial["host_refused"]:
            assert trial["previous_epoch_kept"] == previous_epoch, trial
        if previous_epoch == "weights":
            # The previous epoch keeps its weights alone: each GPU's quota
            # reserves nothing else of it once its processes exited, the
            # segments the target maps being the target's.
            def kept() -> set[str]:
                return {
                    entry["item"]
                    for entry in _quota_reservations(base_url)
                    if previous in entry["epochs"]
                }

            wait_until(
                lambda: kept() == {"weights"},
                60,
                "the previous epoch kept more than its weights",
                context=lambda: str(kept()),
            )
        rolled = admin(base_url, "config/rollback", {"reason": "the check"})
        assert rolled.status_code == 200, rolled.text
        _assert_whole(streaming, streamed)
        # The previous epoch serves again once its processes stopped for the
        # trial started again; the target serves until then.
        update = wait_for_phase(base_url, "rolled_back", timeout_s=600)
        assert update["failure"]["reason"] == "the check", update
        after = _chat(base_url, name, RECOVERY_PROMPT, tokens=32)
        assert (
            after["choices"][0]["message"]["content"]
            == answer["choices"][0]["message"]["content"]
        )


def test_a_regression_of_the_epoch_on_trial_rolls_it_back(
    tmp_path: Path, runtime: Path
):
    """A target that steps slower past the threshold rolls back by itself.

    The target's decode attention runs its kernel several times per call
    (:func:`_slower_target`), so the same stream steps slower on it with no
    fault. The stream crosses the switch and the rollback whole, the update
    ends unavailable with the threshold's reason, the status states the rates
    that decided it, and applying it again needs ``force``. Forced without
    the threshold, the same slower target stands its trial: the threshold,
    and nothing else, rolled it back.
    """
    name = "qwen3-live-regression"
    log = tmp_path / "server.log"
    # The baseline is the previous epoch's steps up to the switch, which the
    # target's loading and capture on the same GPU lengthen (to 1.16 times
    # their length before the preparation, as the transfer control read them
    # in one run): a fifth past such a baseline is still well short of the
    # target's 69%.
    regression = 0.2
    thresholds = (
        f"update_config.rollback.step_time_regression={regression}",
        f'update_config.rollback.window="{TRIAL_WINDOW_S}s"',
    )
    with _serve(runtime, log, name) as base_url:
        answer = _chat(base_url, name, RECOVERY_PROMPT, tokens=32)
        slower = _slower_target(tmp_path)
        # The stream runs through the preparation, the switch, a whole window
        # of the trial and the rollback's drain, which the count outlasts only
        # when it starts with the update: it takes little more than half a
        # minute at the deployment's steps. The preparation's own seconds give
        # the baseline its window.
        streaming, streamed = _start_count(base_url, name)
        result = _apply(base_url, *slower, *thresholds)
        assert result["change"] == "epoch", result
        wait_until(
            lambda: update_report(base_url).get("phase") == "rolled_back",
            600,
            "the slower epoch was not rolled back",
        )
        update = update_report(base_url)
        assert update["failure"]["kind"] == "step_time_regression", update
        print("rolled back:", json.dumps(update))
        # The status states the rates the decision read.
        (replica,) = update["trial"]["replicas"]
        assert replica["baseline"]["tokens_per_second"] > 0, update
        baseline = replica["baseline"]["step_us"]
        assert replica["rates"]["step_us"] > baseline * (1 + regression), update
        _assert_whole(streaming, streamed)
        after = _chat(base_url, name, RECOVERY_PROMPT, tokens=32)
        assert (
            after["choices"][0]["message"]["content"]
            == answer["choices"][0]["message"]["content"]
        )

        again = apply_config(base_url, *slower)
        assert again.status_code != 200, again.text
        assert "force" in again.text, again.text

        streaming, streamed = _start_count(base_url, name)
        time.sleep(TRIAL_WINDOW_S + 2)
        forced = apply_config(base_url, *slower, force=True)
        assert forced.status_code == 200, forced.text
        _on_trial(base_url)
        time.sleep(2 * TRIAL_WINDOW_S)
        unthresholded = update_report(base_url)
        assert unthresholded["phase"] == "trial", unthresholded
        committed = admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text
        _assert_whole(streaming, streamed)


def test_a_package_update_past_a_threshold_rolls_back_in_its_epoch(
    tmp_path: Path, runtime: Path
):
    """A generation update stands the same trial as an epoch switch.

    The serving epoch computes a stream before the update and a trickle after
    it, so its token throughput falls past the threshold. The update rolls
    back in the same main process, with the threshold as its failure.
    """
    name = "qwen3-live-package-trial"
    log = tmp_path / "server.log"
    with _serve(runtime, log, name) as base_url:
        streaming, streamed = _start_count(base_url, name)
        time.sleep(TRIAL_WINDOW_S + 2)
        result = _apply(
            base_url,
            "scheduler_config.max_batch=4",
            "update_config.rollback.throughput_regression=0.5",
            f'update_config.rollback.window="{TRIAL_WINDOW_S}s"',
        )
        assert result["change"] == "generation", result
        _assert_whole(streaming, streamed)

        def rolled_back() -> bool:
            _chat(base_url, name, RECOVERY_PROMPT, tokens=2)
            time.sleep(1)
            return bool(update_report(base_url).get("failure"))

        wait_until(rolled_back, 120, "the update was not rolled back")
        update = update_report(base_url)
        print("rolled back:", json.dumps(update))
        assert update["failure"]["kind"] == "throughput_regression", update


def test_a_request_that_ends_at_once_leaves_its_blocks_in_the_host_tier(
    tmp_path: Path, runtime: Path
):
    """A prompt's blocks reach the host tier though no batch follows them.

    A request of one output token ends with the step that fills its blocks,
    and nothing else runs, so the copies its stores need have no later batch
    to go with: the scheduler sends its executor an empty batch, which runs
    them on the device's copy stream (``specs/buffer-pool.md`` §10).
    """
    name = "qwen3-live-copies-alone"
    log = tmp_path / "server.log"
    prompt = " ".join(f"entry {index} of the table" for index in range(800))
    with _serve(runtime, log, name) as base_url:
        _chat(base_url, name, prompt, tokens=1)
        time.sleep(2)
        flushed = admin(base_url, "flush_cache")
        assert flushed.status_code == 200, flushed.text
        found = _host_tokens_read(
            base_url, lambda: _chat(base_url, name, prompt, tokens=1)
        )
        print("host tokens of a request of one token:", found)
        assert found > 0, found
        # What moved the blocks is on the report beside what computed them.
        metrics = httpx.get(f"{base_url}/metrics", timeout=10).text
        copied = [
            line
            for line in metrics.splitlines()
            if line.startswith("uniserve:worker_stream_us")
            and 'stream="copy"' in line
        ]
        print("copy stream:", copied)
        assert copied, metrics[:2000]


def test_host_hits_beyond_the_device_cache_all_finish(
    tmp_path: Path, runtime: Path
):
    """Requests that find more blocks in the host tier than the device holds
    all end.

    A host hit takes the units of its blocks at admission, and a running
    request keeps its units until it ends. Admitted together, requests whose
    tails run past their first chunk would hold every unit with none left for
    those tails, so admission keeps available the units the running requests
    still need to reach their next call and low watermark
    (``specs/scheduler-design.md`` §4.4).
    """  # noqa: D205
    name = "qwen3-live-host-hits"
    log = tmp_path / "server.log"
    prompts = [
        " ".join(f"{turn}-{word}" for word in range(2495)) for turn in range(4)
    ]
    tail = " ".join(f"tail-{word}" for word in range(300))
    with _serve(runtime, log, name) as base_url:
        for prompt in prompts:
            _chat(base_url, name, prompt, tokens=1)
        time.sleep(2)
        flushed = admin(base_url, "flush_cache")
        assert flushed.status_code == 200, flushed.text
        errors: list[BaseException] = []

        def ask(prompt: str) -> None:
            try:
                _chat(base_url, name, prompt, tokens=1, timeout_s=300)
            except httpx.HTTPError as error:
                errors.append(error)

        def ask_together() -> None:
            asking = [
                threading.Thread(target=ask, args=(f"{prompt} {tail}",))
                for prompt in prompts
            ]
            for thread in asking:
                thread.start()
            for thread in asking:
                thread.join(timeout=400)

        found = _host_tokens_read(base_url, ask_together)
        print("host tokens of the four requests:", found)
        assert not errors, f"{errors[0]!r}{_log_trouble(log)}"
        assert found > 0, found


@pytest.mark.gpus(2)
@pytest.mark.parametrize(
    "parallel",
    [
        ("expert", ("data_parallel_size=2", "moe_config.expert_parallel=true")),
        ("data", ("data_parallel_size=2",)),
    ],
    ids=lambda parallel: parallel[0],
)
def test_a_process_of_two_devices_keeps_the_host_kv_cache(
    tmp_path: Path, runtime: Path, parallel: tuple[str, tuple[str, ...]]
):
    """A main process over two devices finds a prompt's blocks in the host
    tier.

    Each of its schedulers serves one device with a queue pair of its own on
    the host's buffer pool, whose host regions keep the blocks it stores,
    which an expert union computes on both devices. A prompt whose device
    copies a flush evicted is loaded back from the host, and the answer is the
    one it gave before.
    """  # noqa: D205
    kind, options = parallel
    name = f"qwen3-live-host-{kind}"
    log = tmp_path / "server.log"
    prompt = " ".join(f"item {index} of the list" for index in range(800))
    with _serve(
        runtime,
        log,
        name,
        env={"CUDA_VISIBLE_DEVICES": granted_devices(2)},
        options=(*OPTIONS, *options),
    ) as base_url:
        answer = _chat(base_url, name, prompt, tokens=16)
        time.sleep(2)
        flushed = admin(base_url, "flush_cache")
        assert flushed.status_code == 200, flushed.text
        answers = []
        found = _host_tokens_read(
            base_url,
            lambda: answers.append(_chat(base_url, name, prompt, tokens=16)),
        )
        print(f"host tokens on {kind} parallel devices:", found)
        assert found > 0, f"{found}{_log_trouble(log)}"
        assert (
            answers[0]["choices"][0]["message"]["content"]
            == answer["choices"][0]["message"]["content"]
        )


@pytest.mark.parametrize(
    "tiers",
    [
        (),
        ("kv_cache_config.host_tier=false",),
        ("kv_cache_config.prefix_caching=false",),
    ],
    ids=["host", "device", "uncached"],
)
def test_requests_that_outgrow_the_cache_in_decode_all_finish(
    tmp_path: Path, runtime: Path, tiers: tuple[str, ...]
):
    """A cache that fills while its requests decode gives one of them up.

    Admission holds what a request's prompt takes and not what it generates, so
    eight counts on a cache of 8192 tokens outgrow it as they decode. The one
    admitted last gives its units back and continues from the tokens it
    generated, so every count ends whole and none starves. A count alone that
    fills the whole cache ends at the length the cache holds, with the finish
    reason ``cache_capacity`` (``specs/scheduler-design.md`` §3.6).
    """
    name = "qwen3-live-decode-growth"
    log = tmp_path / "server.log"
    options = (
        *SIZED_OPTIONS,
        "kv_cache_config.max_tokens=8192",
        *tiers,
    )
    # About 3400 tokens each (a number of three digits, its comma and its
    # space are five tokens), so eight of them need more than three caches of
    # 8192 tokens by their ends, and the cache fills while every one decodes.
    prompt = (
        "Count from 1 to 700, separated by commas and spaces. "
        "Write only the numbers."
    )
    with _serve(runtime, log, name, options=options) as base_url:
        streams = []
        for _ in range(8):
            stream = {
                "read": [],
                "done": threading.Event(),
                "reasons": [],
                "errors": [],
            }
            stream["thread"] = threading.Thread(
                target=_stream_count,
                args=(base_url, name, stream["read"], stream["done"]),
                kwargs={
                    "reasons": stream["reasons"],
                    "errors": stream["errors"],
                    "prompt": prompt,
                },
            )
            streams.append(stream)
        for stream in streams:
            stream["thread"].start()
        for stream in streams:
            stream["thread"].join(timeout=1800)
        counts = []
        for stream in streams:
            assert stream["done"].is_set(), "a count did not end"
            assert not stream["errors"], (
                f"{stream['errors'][0]!r}{_log_trouble(log)}"
            )
            numbers = _numbers(stream["read"])
            assert numbers == list(range(1, 701)), numbers[-60:]
            assert stream["reasons"] == ["stop"], stream["reasons"]
            counts.append(len(numbers))
        # Eight such counts outgrow the cache by their length alone, so every
        # one ending whole is the request given up continuing.
        print("numbers of each count:", counts)
        # A count longer than the cache holds runs alone until it fills it, and
        # ends there: its next token needs a block the whole cache lacks.
        read: list[str] = []
        done = threading.Event()
        reasons: list[str] = []
        errors: list[BaseException] = []
        _stream_count(
            base_url, name, read, done, reasons=reasons, errors=errors
        )
        assert not errors, f"{errors[0]!r}{_log_trouble(log)}"
        *whole, last = _numbers(read)
        print("numbers of a count the cache cannot hold:", len(whole), reasons)
        assert whole == list(range(1, len(whole) + 1)), whole[-60:]
        # The cache is full where it ends, which may be inside its last number.
        assert str(len(whole) + 1).startswith(str(last)), (whole[-5:], last)
        assert reasons == ["cache_capacity"], reasons


def test_a_weight_update_leaves_the_blocks_of_the_old_weights_unread(
    tmp_path: Path, runtime: Path
):
    """A block in the host tier belongs to the weights that computed it.

    A prompt finds its blocks in the host tier once their device copies are
    flushed. After an update to other weights the engine keys its blocks by
    the new weights, so the same prompt finds none of them, although the new
    weights name the same model: it computes its own, which the next flush
    makes it find. The other checkpoint is another copy of the files, so the
    two weights answer alike and a stale block would go unseen in the text.
    """
    other = str(
        host_asset("UNISERVE_QWEN3_UPDATED_MODEL", QWEN3_UPDATED_CHECKPOINT)
    )
    name = "qwen3-live-weights-blocks"
    log = tmp_path / "server.log"
    prompt = " ".join(f"item {index} of the list" for index in range(800))

    def found_in_the_host_cache() -> float:
        """Sends the prompt on a flushed device; the host tokens it reads."""
        flushed = admin(base_url, "flush_cache")
        assert flushed.status_code == 200, flushed.text
        return _host_tokens_read(
            base_url, lambda: _chat(base_url, name, prompt, tokens=16)
        )

    with _serve(runtime, log, name) as base_url:
        first = _chat(base_url, name, prompt, tokens=16)
        time.sleep(2)
        old_weights = found_in_the_host_cache()
        print("host tokens of the first weights:", old_weights)
        assert old_weights > 0, old_weights
        updated = admin(base_url, "update_weights", {"model_path": other})
        assert updated.status_code == 200, updated.text
        print("updated:", updated.text[:400])
        answers = []
        stale = _host_tokens_read(
            base_url,
            lambda: answers.append(_chat(base_url, name, prompt, tokens=16)),
        )
        second = answers[0]
        print("host tokens after the update:", stale)
        assert stale == 0, stale
        assert (
            second["choices"][0]["message"]["content"]
            == first["choices"][0]["message"]["content"]
        )
        time.sleep(2)
        new_weights = found_in_the_host_cache()
        print("host tokens of the new weights:", new_weights)
        assert new_weights > 0, new_weights


def test_a_weight_update_keeps_the_stream_in_flight_going(
    tmp_path: Path, runtime: Path
):
    """A weight update never holds the stream in flight as long as it takes.

    With ``in_flight = keep`` the batches go on while the new checkpoint loads
    beside the KV cache, and the weights are put in place at one batch boundary
    (``docs/admin.md``, the KV cache's storage and the weights), so the
    stream's longest silence from its last token before the update to its end
    stays below the update's duration, and the stream is whole.

    The weights put in place are mapped at the addresses the captured graphs
    read, which a process subscribed to CUPTI follows by instantiating each
    graph's executable again (``profiling::storage_remapped``); that renewal
    belongs to each graph's next replay and never holds a batch for every
    bucket the runner captured. The case states its measures whatever the
    process's CUPTI, and guards the renewal where the deployment's environment
    gives it one (``VIRTUAL_ENV`` naming the project's environment, whose
    ``nvidia-cuda-cupti`` the engine subscribes with).
    """
    other = str(
        host_asset("UNISERVE_QWEN3_UPDATED_MODEL", QWEN3_UPDATED_CHECKPOINT)
    )
    name = "qwen3-live-weights-stream"
    log = tmp_path / "server.log"
    with _serve(runtime, log, name) as base_url:
        read: list[str] = []
        done = threading.Event()
        errors: list[BaseException] = []
        arrivals: list[float] = []
        reasons: list[str] = []
        streaming = threading.Thread(
            target=_stream_count,
            args=(base_url, name, read, done),
            kwargs={"reasons": reasons, "errors": errors, "arrivals": arrivals},
        )
        streaming.start()
        wait_until(
            lambda: len(_numbers(read)) >= 8,
            300,
            "the count did not stream",
            STREAM_POLL_S,
        )

        started = time.monotonic()
        updated = admin(base_url, "update_weights", {"model_path": other})
        took = time.monotonic() - started
        assert updated.status_code == 200, updated.text
        streaming.join(timeout=1800)

    # The stream's silences from its last token before the update on, each
    # token's arrival its own reading.
    spans = [moment for moment in arrivals if moment < started][-1:] + [
        moment for moment in arrivals if moment >= started
    ]
    longest = max(
        (later - earlier for earlier, later in zip(spans, spans[1:])),
        default=0.0,
    )
    print(f"update {took:.3f} s, the stream's longest silence {longest:.3f} s")
    assert done.is_set(), "the count did not end"
    assert not errors, f"the count failed: {errors[0]!r}"
    numbers = _numbers(read)
    assert numbers == list(range(1, len(numbers) + 1)), numbers[-40:]
    assert len(spans) > 1, "the count ended before the update"
    assert longest < took, (
        f"the stream in flight waited {longest:.3f} s, longer than the update "
        f"took ({took:.3f} s)"
    )


def test_a_previous_epoch_in_host_storage_moves_its_weights_and_returns(
    tmp_path: Path, runtime: Path
):
    """``previous_epoch = "host"`` moves bytes, and a rollback brings them back.

    A target of other model files maps none of the serving epoch's weights, so
    the weights that epoch alone retains leave its GPU for host storage while
    the target serves. The rollback returns them and starts the epoch on them,
    under a stream that crosses both moves whole, and it answers as it did.
    """
    other = str(
        host_asset("UNISERVE_QWEN3_UPDATED_MODEL", QWEN3_UPDATED_CHECKPOINT)
    )
    name = "qwen3-live-host-weights"
    log = tmp_path / "server.log"
    # The target loads its checkpoint beside the serving one, and the
    # deployment's quota of the GPU holds both only past the default share.
    options = (*OPTIONS, "mem_fraction_static=0.9")
    with _serve(runtime, log, name, options=options) as base_url:
        answer = _chat(base_url, name, RECOVERY_PROMPT, tokens=32)
        streaming, streamed = _start_count(base_url, name)
        result = _apply(
            base_url, f"model={other}", "update_config.previous_epoch=host"
        )
        assert result["change"] == "epoch", result
        _on_trial(base_url)
        wait_until(
            lambda: (
                update_report(base_url)["moved_to_host_bytes"] > 0
                or update_report(base_url)["host_refused"]
            ),
            300,
            "the previous epoch's weights did not move",
        )
        trial = update_report(base_url)
        print("kept:", json.dumps(trial))
        assert not trial["host_refused"], trial
        assert trial["previous_epoch_kept"] == "host", trial
        rolled = admin(base_url, "config/rollback", {"reason": "the check"})
        assert rolled.status_code == 200, rolled.text
        _assert_whole(streaming, streamed)
        after = _chat(base_url, name, RECOVERY_PROMPT, tokens=32)
        assert (
            after["choices"][0]["message"]["content"]
            == answer["choices"][0]["message"]["content"]
        )
        update = update_report(base_url)
        assert update["phase"] == "rolled_back", update


def test_package_updates_under_a_request_keep_its_stream_whole(
    tmp_path: Path, runtime: Path
):
    """In-place switches under a request leave its stream as a run without them.

    A generation update reaches a process that serves: its switch captures
    nothing a batch of the running request can be mistaken for, so every token
    the request computes is the one its stream delivers. The count crosses as
    many switches as fit under it, each one committed.
    """
    name = "qwen3-live-switches"
    log = tmp_path / "server.log"
    switches = 0
    with _serve(runtime, log, name) as base_url:
        for _ in range(2):
            streaming, streamed = _start_count(base_url, name)
            while not streamed["done"].is_set():
                switches += 1
                size = 4 if switches % 2 else 8
                result = _apply(base_url, f"scheduler_config.max_batch={size}")
                assert result["change"] == "generation", result
                committed = admin(base_url, "config/commit")
                assert committed.status_code == 200, committed.text
                time.sleep(0.5)
            _assert_whole(streaming, streamed)
    print("switches:", switches)
    assert switches >= 20, switches
