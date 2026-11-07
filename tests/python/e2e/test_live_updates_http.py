"""A deployment's lifecycle and live updates through the admin API.

Qwen3 serves on one GPU of the host, with a Storage Server of the test's
own (``XDG_RUNTIME_DIR`` points the daemon at a directory of the test's,
and the server the daemon starts there is stopped at the end).

- A node (``uniserve serve`` without a model) brings deployments up
  (``PUT /admin/config``) and down through its admin API without
  restarting: each deployment's page pools and weights
  belong to it, so bringing it down returns the host's shared memory and
  the GPU's storage to what they were before it, and a deployment whose
  page pools the host's available storage cannot hold is refused with the
  sizes that decided it.
- A serving deployment answers its admin API: new requests wait while
  generation is paused and are served once it continues; a flush revokes
  the cached prefix pages; scheduling limits within the batch the workers
  were started for change in place while a request streams, and a change
  of the decode batch sizes is a world change under the same stream,
  which the operator commits; the manager attaches to the node, follows
  its events and detaches while it serves.
- A world change while another process holds the device's free storage
  gives back the tail of the serving world's KV cache for the successor,
  and the committed successor grows its cache into the storage the
  previous world freed.
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
import tempfile
import threading
import time
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    log_tail,
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
    # The attention program the deployment's paged attention executes, as a
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
# A change of the options that only a new world executes: graphs of other
# decode batch sizes.
CHANGE = "cuda_graph_config.decode_batch_sizes=[1, 2, 8]"
# The deployment's host KV cache on the host, in GiB.
HOST_GIB = 8
PROMPT = (
    "Count from 1 to 3000, separated by commas and spaces. "
    "Write only the numbers."
)


# The packages this file installs: the scheduling policy, and one kernel
# package for every operator the deployment selects. A deployment's epoch is
# resolved at this host's Program Server before its main processes serve
# (design 5.1, 6.1): the policy fills the scheduling slot, and the operators
# the build config names fill theirs, so a server that has the policy alone
# refuses the epoch by the first kernel it lacks.
#
# ``FilledPrograms::of`` (``crates/daemon/src/epoch.rs``) fills paged attention
# with the attention program's backend; ``moe_runner_backend`` is unset here, so
# no grouped expert GEMM is filled, and the expert exchange is filled only for
# an expert-parallel deployment — the model launches no exchange when every
# expert is local, so this one-rank deployment names no such program.
#
# A program's backend is a package's name: the Program Server selects a program
# by the backend its manifest declares, and no package is named ``auto``. The
# deployment therefore states the attention backend it means (``OPTIONS``), and
# it is the one design 9's default choice resolves to for this shape — BF16, no
# window, head_dim 128, 64-token pages, one rank: trtllm-gen.
POLICY_PACKAGE = "crates/ops/program-fixtures/policy"

#: The descriptor keys the deployment launches: the model's own ``keys()`` plus
#: the executor's sampler launches, as ``crates/ops`` generators state them
#: (``notes/programs/qwen3-moe-keys.txt``). The store the fixture serves holds a
#: module for every one of them.
#:
#: This file used to install one kernel package, the attention one the daemon
#: names. That is enough for the epoch to *resolve* — the daemon names one
#: operator and the Program Server fills that slot — but not to serve: a main
#: process installs the kernels the model actually launches, and it resolves
#: them from this host's store (``listed_programs`` in ``src/engine_main.rs``).
#: With two packages there, the listing described nothing to overlay and the
#: precopy stood in ``this node's devices`` until its 300 s budget ended it.
KEYS = "notes/programs/qwen3-moe-keys.txt"

#: The precisions whose rows in ``KEYS`` belong to deployments other than this
#: file's. The key file covers three deployments — bf16 dense weights, NVFP4
#: weights, and an fp8 KV variant — in one store, as its own header states.
#: This file serves the bf16 dense one, so a row that names an NVFP4 encoding or
#: an ``f8e4m3`` scale or page is another deployment's, and one operator
#: (``kv_write_scaled``) stands in ``kv_write``'s place there. Building those
#: rows here fails on packages that deployment's owner owes, which says nothing
#: about whether this deployment can serve.
FOREIGN_PRECISIONS = ("nvfp4", "f8e4m3")
FOREIGN_OPERATORS = frozenset({"kv_write_scaled"})

#: The socket of the module's Program Server, which every deployment of this
#: file reaches through ``--program-server``. Set by :func:`program_server`.
PROGRAM_SERVER: Path | None = None


def _repository() -> Path:
    """The repository root: the workspace the policy package is built in."""
    return Path(__file__).resolve().parents[3]


def _precompiler() -> Path:
    """The wasm precompiler the Program Server turns a policy module into.

    A policy is built for wasm32 and precompiled for this host's CPU, so the
    server needs the precompiler before it can install a policy (design 5.3).
    It is named by the environment, found on the path, shipped beside the
    ``uniserve`` binary, or built from this tree, in that order.
    """
    named = os.environ.get("UNISERVE_SCHEDULING_PRECOMPILE")
    if named:
        return Path(named)
    found = shutil.which("scheduling-precompile")
    if found:
        return Path(found)
    beside = Path(require_uniserve_binary()).with_name("scheduling-precompile")
    if beside.exists():
        return beside
    root = _repository()
    done = subprocess.run(
        [
            "cargo",
            "build",
            "--package",
            "scheduling",
            "--bin",
            "scheduling-precompile",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if done.returncode != 0:
        pytest.fail(
            f"scheduling-precompile did not build: {done.stderr[-2000:]}"
        )
    return root / "target" / "debug" / "scheduling-precompile"


def _program_target() -> str:
    """The target this host's modules are built for.

    The engine derives it from the device it serves — ``sm_{major}{minor}a``
    (``src/engine_plan.rs``) — and a store built for another target describes
    nothing to overlay: the precopy then fails at the module lane rather than
    the build. So the default is the served device's own compute capability,
    read the way this file already reads that device; the environment's
    ``UNISERVE_PROGRAM_TARGET`` names a target instead, for a host whose store
    is provisioned elsewhere.
    """
    named = os.environ.get("UNISERVE_PROGRAM_TARGET")
    if named:
        return named
    device = os.environ.get("UNISERVE_LIVE_DEVICE", "0")
    reported = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-gpu=compute_cap",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    major, _, minor = reported.partition(".")
    if not major.isdigit() or not minor.isdigit():
        raise AssertionError(
            f"nvidia-smi reported no compute capability for GPU {device}: "
            f"{reported!r}"
        )
    return f"sm_{major}{minor}a"


def _deployment_keys(store: Path) -> Path:
    """The key lines this file's deployment serves, written beside `store`.

    ``KEYS`` is the union of three deployments' rows; ``build_fixtures`` fails
    on any row that does not build, so handing it the union would let another
    deployment's missing package stop this one from serving. The rows kept here
    are the bf16 dense deployment's: everything whose key names none of
    ``FOREIGN_PRECISIONS`` and whose operator is not ``FOREIGN_OPERATORS``.
    Comments and blank lines are dropped — the store's build reads rows, and
    the selection remains readable in the file it came from.
    """
    kept = []
    for line in (_repository() / KEYS).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        operator, _, key = stripped.partition("\t")
        if operator in FOREIGN_OPERATORS:
            continue
        lowered = stripped.lower()
        if any(precision in lowered for precision in FOREIGN_PRECISIONS):
            continue
        kept.append(stripped)
    if not kept:
        raise AssertionError(
            f"{KEYS} states no row for this deployment: "
            f"{sorted(FOREIGN_PRECISIONS)} and "
            f"{sorted(FOREIGN_OPERATORS)} excluded all of them"
        )
    path = store.parent / "deployment-keys.txt"
    path.write_text("\n".join(kept) + "\n")
    return path


def _build_store(store: Path, target: str, python: Path | None) -> None:
    """Builds a module for every kernel key this deployment launches.

    `store` is ``build_fixtures``'s store, which it fills in its own shape: the
    installed package sources under ``<store>/packages`` and the prepared
    modules under ``<store>/modules``. The directory a Program Server serves is
    the second of those — a store root holds the modules, not the store's
    parent.

    ``build_fixtures`` (``crates/program-server/examples``) is the store's own
    producer: for each ``operator<TAB>key`` line of the key file it installs the
    operator's package, prepares that key for `target`, and resolves it to the
    module it names. It is given :func:`_deployment_keys`, not ``KEYS``: this
    deployment's rows only, so a package another deployment owes fails that
    deployment's run and not this one. The builds are real — Triton, nvcc and
    CuTe DSL all run — so this takes minutes rather than seconds, once per
    module. The Program Server then
    serves the store as it stands, which is why ``--store-root`` below is this
    directory and no kernel package is installed a second time.

    Every row it is given is one this deployment launches, so a row that does
    not build is this deployment's failure and the file stops here: a short
    store would only move the same news to the precopy a minute later, and the
    build's own output names the key and the reason.
    """
    command = [
        "cargo",
        "run",
        "--package",
        "program-server",
        "--example",
        "build_fixtures",
        "--",
        "--keys",
        str(_deployment_keys(store)),
        "--target",
        target,
        "--store",
        str(store),
    ]
    if python is not None:
        command += ["--python", str(python)]
    built = subprocess.run(
        command,
        cwd=_repository(),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if built.returncode != 0:
        missing = [
            line.removeprefix("FAILED\t").replace("\t", " ")
            for line in built.stdout.splitlines()
            if line.startswith("FAILED\t")
        ]
        raise AssertionError(
            "the deployment's own kernels did not build, so nothing can serve"
            + (f": {', '.join(missing)}" if missing else "")
            + f"\n{built.stdout[-4000:]}\n{built.stderr[-4000:]}"
        )


@pytest.fixture(scope="module", autouse=True)
def program_server(tmp_path_factory: pytest.TempPathFactory):
    """The host's Program Server, with the deployment's programs installed.

    Autouse and module-scoped because every deployment of this file needs it:
    the daemon's leader resolves each epoch's programs at this server, and a
    program's module exists nowhere else. Two sets of programs are here, built
    once for the module:

    * the kernels — a module for every descriptor key the deployment launches,
      built into the store by ``build_fixtures`` (``KEYS``). The epoch is
      refused by the first slot no installed program fills, and a main process
      serves nothing until it can install the kernels the model launches;
    * the policy — ``POLICY_PACKAGE``, installed below, because a policy is a
      wasm package this host precompiles rather than a kernel key.

    The package root and the store live as long as the module, and every test
    of the module deploys against this one server, so a module is built once and
    every later deployment of the module reuses it (design 5.3). They are not
    shared with another run: a store entry is keyed by operator, backend,
    revision, target and descriptor, and a revision counts from 1 in a fresh
    package root, so a store kept across runs would answer a changed package
    with the module the older source built.
    """
    global PROGRAM_SERVER
    binary = require_uniserve_binary()
    work = tmp_path_factory.mktemp("program-server")
    socket = work / "program-server.sock"
    log = work / "program-server.log"
    target = _program_target()
    interpreter = _repository() / ".venv" / "bin" / "python"
    built = work / "programs"
    _build_store(built, target, interpreter if interpreter.exists() else None)
    # The builder's two directories are the two the server is given: installed
    # sources for a rebuild, prepared modules for what it serves.
    packages = built / "packages"
    store = built / "modules"
    command = [
        str(binary),
        "program-server",
        "serve",
        "--socket",
        str(socket),
        "--package-root",
        str(packages),
        "--store-root",
        str(store),
        "--target",
        target,
        "--wasm",
        str(_precompiler()),
    ]
    if interpreter.exists():
        command += ["--python", str(interpreter)]
    process = start_server(command, log)
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the Program Server exited: {log.read_text()}")
        if socket.exists():
            break
        time.sleep(0.2)
    else:
        raise TimeoutError(f"the Program Server left no socket at {socket}")

    for package in (POLICY_PACKAGE,):
        installed = subprocess.run(
            [
                str(binary),
                "program-server",
                "install",
                "--socket",
                str(socket),
                str(_repository() / package),
            ],
            capture_output=True,
            text=True,
            # A kernel package builds its modules on the first prepare, not
            # here: installing copies the source and registers its revision.
            timeout=3600,
        )
        if installed.returncode != 0:
            raise AssertionError(
                f"installing {package} failed: "
                f"{installed.stdout}\n{installed.stderr}"
            )
    PROGRAM_SERVER = socket
    try:
        yield socket
    finally:
        PROGRAM_SERVER = None
        stop_server(process)


def program_server_option() -> list[str]:
    """The command-line option that gives a deployment its programs' server.

    Every deployment serves a scheduling policy, and a policy's module comes
    from this host's Program Server (design 6.1); without the option the leader
    resolves the epoch's programs nowhere and the deployment never serves.
    """
    if PROGRAM_SERVER is None:
        pytest.fail("the module's Program Server is not running")
    return ["--program-server", str(PROGRAM_SERVER)]


def _checkpoint() -> str:
    checkpoint = os.environ.get("UNISERVE_QWEN3_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail("UNISERVE_QWEN3_MODEL must name the model checkpoint")
    return checkpoint


def _pool_gib(runtime: Path) -> float:
    """The host page pools the test's Storage Server holds, in GiB: the
    sizes of its ``uniserve-page-pool`` memfds, whose storage the host frees
    once the server and the workers that map them let them go.
    """  # noqa: D205
    pid = _storage_pid(runtime)
    held = 0
    for descriptor in Path(f"/proc/{pid}/fd").iterdir():
        try:
            if os.readlink(descriptor).startswith("/memfd:uniserve-page-pool"):
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


def _held(runtime: Path) -> list[str]:
    """What the test's Storage Server holds, its page pools without their
    slot counters, which the deployment's offloads move as it serves. Its
    Transfer Server's process, contexts, link measurements and tasks are no
    storage it holds: a restarted server's Transfer Server measures anew.
    """  # noqa: D205
    return sorted(
        line.split(": ", 1)[0] if line.startswith("page pool") else line
        for line in _storage_list(runtime).splitlines()
        if not line.startswith(("transfer ", "link "))
    )


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


def _log_trouble(log: Path, limit: int = 40) -> str:
    """The last `limit` warnings and errors of a server's log, marked.

    A tail of the last lines is the wrong excerpt for a failure whose cause is
    minutes earlier: a relaunch logs a capture per bucket per device, so two
    hundred tail lines can hold nothing but captures and stop short of the
    engine's death. The warnings and errors are what name a cause, and there
    are few of them whatever the moment.
    """  # noqa: D205
    try:
        # The engine writes `WARNING` and `ERROR`; a process that dies writes
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
        for line in _log_lines(log):
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
    text = httpx.get(f"{base_url}/metrics", timeout=10).text
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith("uniserve:num_requests_running")
    )


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
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            *program_server_option(),
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
    response = httpx.put(
        f"{base_url}/admin/config",
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


def test_a_node_serves_deployments_one_after_another_and_releases_their_storage(
    tmp_path: Path, runtime: Path
):
    device = os.environ.get("UNISERVE_LIVE_DEVICE", "0")
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
            pools = _pool_gib(runtime)
            print(
                f"{name} serving: page pools {pools:.2f} GiB, "
                f"GPU {_gpu_used_mib(device)} MiB"
            )
            # The deployment's page pools: its host KV cache of `HOST_GIB`.
            assert pools >= HOST_GIB * 0.99, pools
            assert name in _storage_list(runtime)

            response = httpx.post(
                f"{base_url}/admin/shutdown", json={}, timeout=60
            )
            assert response.status_code == 202, response.text
            state = _settle(base_url)
            assert state["state"] == "idle", state
            assert state["last_error"] is None, state
            after_pools = _pool_gib(runtime)
            after_gpu = _gpu_used_mib(device)
            print(
                f"{name} down: page pools {after_pools:.2f} GiB, "
                f"GPU {after_gpu} MiB"
            )
            assert name not in _storage_list(runtime)
            assert after_pools == 0, after_pools
            assert after_gpu <= gpu + 1024, (after_gpu, gpu)

        # A host KV cache the host's available storage cannot hold.
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
        *program_server_option(),
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


def _apply(base_url: str, *assignments: str) -> dict:
    """Applies `assignments` (``PUT /admin/config``); returns its result."""
    response = httpx.put(
        f"{base_url}/admin/config",
        json={"set": list(assignments)},
        timeout=1800,
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]


def _change_world(base_url: str, *assignments: str) -> dict:
    """Applies `assignments` as a world change and commits the successor;
    returns the change's report.
    """  # noqa: D205
    result = _apply(base_url, *assignments)
    assert result["change"] == "epoch", result
    committed = _admin(base_url, "config/commit")
    assert committed.status_code == 200, committed.text
    return result["epoch"]


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
        # other decode batch sizes by a world change under the same stream.
        read: list[str] = []
        done = threading.Event()
        streaming = threading.Thread(
            target=_stream_count, args=(base_url, name, read, done)
        )
        streaming.start()
        _wait_until(lambda: len(read) >= 32, 300, "the count did not stream")
        update = _apply(base_url, "scheduler_config.max_batch=4")
        assert update["change"] == "program", update
        committed = _admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text
        before_update = len(read)
        report = _change_world(base_url, CHANGE)
        print(f"world change pause: {report['pause_seconds']:.3f} s")
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
            "config" in output and "detached; the nodes serve on" in output
        ), output
        reply = _chat(base_url, name, "Say hello.")
        assert reply["choices"][0]["message"]["content"]

        events = httpx.get(
            f"{base_url}/admin/events?after=0", timeout=10
        ).json()
        kinds = [event["kind"] for event in events["events"]]
        assert "operation" in kinds, events


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


#: The streams a hot restart hands over: three counts of different lengths, so
#: their event boundaries fall at different points of the response, and the
#: number each stream must carry with no gap and no repetition.
HANDED_STREAMS = (200, 500, 900)


def _handoff_records(log: Path) -> dict[str, list[dict]]:
    """The handoff records the daemons logged, by kind.

    The predecessor writes `handoff stream` per response it froze and
    `handoff streams adopted` once; the successor writes `handoff adopted` when
    it starts writing a taken-over response and `handoff finished` when it
    ends. Each is one log line whose message ends in one JSON object.
    """
    records: dict[str, list[dict]] = {
        "frozen": [],
        "paused": [],
        "adopted": [],
        "finished": [],
    }
    # The prefixes are matched longest first: "handoff stream " is a prefix of
    # "handoff streams adopted ".
    prefixes = (
        ("paused", "handoff streams adopted "),
        ("finished", "handoff finished "),
        ("adopted", "handoff adopted "),
        ("frozen", "handoff stream "),
    )
    for line in _log_lines(log):
        for kind, prefix in prefixes:
            index = line.find(prefix)
            if index < 0:
                continue
            try:
                records[kind].append(json.loads(line[index + len(prefix) :]))
            except json.JSONDecodeError:
                continue
            break
    return records


def _write_handoff_artifacts(
    log: Path, records: dict[str, list[dict]], numbers: list[int]
) -> Path:
    """Writes the handoff evidence under `artifacts/handoff-streams/<UTC>/`.

    `streams.jsonl` holds one line per handed-over response, with what each
    process recorded of it: the freeze, the takeover and the end, and the pause
    between them. `summary.json` holds the counts and the pauses.
    """
    stamp = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    directory = (
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "handoff-streams"
        / stamp
    )
    directory.mkdir(parents=True, exist_ok=True)
    frozen = {record["request_id"]: record for record in records["frozen"]}
    adopted = {record["request_id"]: record for record in records["adopted"]}
    finished = {record["request_id"]: record for record in records["finished"]}
    pauses: list[int] = []
    with (directory / "streams.jsonl").open("w") as out:
        for request_id, freeze in frozen.items():
            taken = adopted.get(request_id, {})
            end = finished.get(request_id, {})
            pauses.append(takeover_pause(freeze, taken))
            line = {
                "request_id": request_id,
                "frozen": freeze,
                "adopted": taken,
                "finished": end,
                "now_ns": time.monotonic_ns(),
            }
            out.write(json.dumps(line) + "\n")
    summary = {
        "streams": len(frozen),
        "adopted": len(adopted),
        "finished": len(finished),
        "pause_ms": {
            "min": min(pauses) if pauses else None,
            "max": max(pauses) if pauses else None,
        },
        "numbers_per_stream": numbers,
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"handoff artifacts: {directory}")
    return directory


def takeover_pause(freeze: dict, taken: dict) -> int:
    """The pause a handed-over response took, in milliseconds.

    It is measured from the predecessor's freeze to the successor's first
    frame, from the two processes' monotonic clocks, which share an origin.
    """
    at_ns = taken.get("at_ns")
    if at_ns is None:
        return freeze["pause_ms"]
    return max(0, (at_ns - freeze["frozen_at_ns"]) // 1_000_000)


def test_a_hot_restart_hands_the_deployment_to_a_successor_daemon(
    tmp_path: Path, runtime: Path
):
    name = "qwen3-live-restart"
    log = tmp_path / "server.log"
    # The daemons a restart starts, which the fixture, stopping the daemon
    # it started, does not stop.
    successors: list[int] = []
    try:
        # The records this test parses are the predecessor's info lines, and a
        # RUST_LOG in the host's environment would drop them; the successor
        # inherits this environment, so both processes log them.
        with _serve(runtime, log, name, {"RUST_LOG": "info"}) as base_url:
            _restart_while_streaming(base_url, name, log, successors)
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
    base_url: str, name: str, log: Path, successors: list[int]
) -> None:
    engines_before = _pids(log, "Engine")
    status = httpx.get(f"{base_url}/admin/status", timeout=10).json()
    assert status["deployment"]["engine"] == "ready", status
    (old_daemon,) = _pids(log, "Daemon")

    # One completion before any stream: a deployment that answers nothing says
    # so here, in seconds and with its own body, instead of as a stream that
    # produced no token in five minutes — a silence the two paths share and
    # that therefore names neither. A refusal is the likelier failure (a
    # relaunched process that was not handed the epoch's programs answers 503
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
        _wait_until(
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
    restarted = _admin(base_url, "restart")
    # A refusal here is the successor's own answer, and its reason names the
    # step that stopped; the daemons' log says what happened to the processes
    # the handover was carrying, which the answer cannot.
    assert restarted.status_code == 200, f"{restarted.text}{_log_trouble(log)}"
    successor = restarted.json()["result"]["pid"]
    successors.append(successor)
    assert successor != old_daemon
    before_restart = [len(read) for read in reads]

    # The successor serves the same deployment, so it holds the same programs'
    # server: a deployment reaches its scheduling policy only through it, and
    # the successor reads the configuration the old process served rather than
    # a command line of its own (design 7.5).
    inherited = httpx.get(f"{base_url}/admin/config", timeout=60.0).json()
    assert program_server_option()[1] in inherited["config"], (
        "the successor did not inherit the programs' server: "
        f"{inherited['config'][:2000]}"
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

    # The records prove the streams moved rather than ended: each was frozen,
    # taken over by the successor at the pause the successor measured, and
    # finished there.
    _wait_until(
        lambda: len(_handoff_records(log)["finished"]) >= len(HANDED_STREAMS),
        300,
        "the successor did not finish the handed-over streams",
    )
    records = _handoff_records(log)
    frozen = records["frozen"]
    assert len(frozen) == len(HANDED_STREAMS), frozen
    assert all(record["adopted"] for record in frozen), frozen
    assert {record["successor_pid"] for record in frozen} == {successor}, frozen
    assert {record["pid"] for record in frozen} == {old_daemon}, frozen
    assert all(record["pause_ms"] >= 0 for record in frozen), frozen
    assert records["paused"], "the predecessor recorded no pause"
    assert records["paused"][-1]["pid"] == old_daemon, records["paused"]
    assert records["paused"][-1]["successor_pid"] == successor, records[
        "paused"
    ]
    adopted = {record["request_id"]: record for record in records["adopted"]}
    finished = {record["request_id"]: record for record in records["finished"]}
    frozen_at = {
        record["request_id"]: record["frozen_at_ns"] for record in frozen
    }
    assert set(adopted) == set(frozen_at), (adopted, frozen_at)
    assert set(finished) == set(frozen_at), (finished, frozen_at)
    for request_id, record in adopted.items():
        assert record["pid"] == successor, record
        assert record["frozen_at_ns"] == frozen_at[request_id], record
        assert record["at_ns"] >= record["frozen_at_ns"], record
        assert finished[request_id]["pid"] == successor, finished[request_id]
        assert finished[request_id]["at_ns"] > record["at_ns"], (
            record,
            finished[request_id],
        )
    _write_handoff_artifacts(
        log,
        records,
        [len(re.findall(r"\d+", "".join(read))) for read in reads],
    )

    # A one-body chat response in flight at a further restart is written by
    # its successor too, whole (design 7.5.1: every response in flight moves).
    _restart_with_a_json_body(base_url, name, log, successors)

    # The old daemon exits once its requests ended; the engine and its
    # workers are the same processes throughout.
    _wait_until(
        lambda: not _runs(old_daemon),
        120,
        "the replaced daemon did not exit",
    )
    # The one-body restart replaced `successor` in turn, so the daemon that
    # serves now is the last of them.
    assert _runs(successors[-1])
    status = httpx.get(f"{base_url}/admin/status", timeout=10).json()
    assert status["deployment"]["engine"] == "ready", status
    assert _pids(log, "Engine") == engines_before
    lines = _log_lines(log)
    assert any("the engine attached" in line for line in lines)
    # The main process read the checkpoint once, when the deployment started.
    assert sum("the epoch's retained weights" in line for line in lines) == 1
    events = httpx.get(f"{base_url}/admin/events?after=0", timeout=10).json()
    messages = [event["message"] for event in events["events"]]
    assert any("after a hot restart" in message for message in messages), (
        messages
    )

    # The successor restarts in turn, and takes a change in place.
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
    # A scheduling limit is a parameter of the policy program, so it reaches the
    # serving epoch as a program update; `max_running_requests` is build config
    # and would switch the epoch.
    update = _apply(base_url, "scheduler_config.max_batch=4")
    assert update["change"] == "program", update
    committed = _admin(base_url, "config/commit")
    assert committed.status_code == 200, committed.text

    # A change of the listening address restarts the daemon on it: the new
    # address serves, and the old one closes once the replaced daemon exits.
    previous = successors[-1]
    port = find_free_port()
    moved = _apply(base_url, f"server_config.port={port}")
    assert moved["change"] == "daemon", moved
    successors.append(moved["pid"])
    moved_url = f"http://127.0.0.1:{port}"
    reply = _chat(moved_url, name, "Say hello.")
    assert reply["choices"][0]["message"]["content"]
    _wait_until(
        lambda: not _runs(previous),
        120,
        "the daemon on the previous address did not exit",
    )
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"{base_url}/health", timeout=5)
    status = httpx.get(f"{moved_url}/admin/config", timeout=10).json()
    server = tomllib.loads(status["config"])["server_config"]
    assert server["port"] == port, server

    # A change of both the daemon's options and what a world executes
    # restarts the daemon on its new options first, and its successor then
    # changes the world in an operation of its own.
    previous = successors[-1]
    port = find_free_port()
    both = _apply(moved_url, f"server_config.port={port}", CHANGE)
    assert both["change"] == "daemon", both
    assert both["then"] == "epoch", both
    successors.append(both["pid"])
    both_url = f"http://127.0.0.1:{port}"

    def changed() -> dict | None:
        status = httpx.get(f"{both_url}/admin/status", timeout=10).json()
        last = status["last_operation"]
        if status["operation"] is None and last and last["kind"] == "config":
            return last
        return None

    _wait_until(lambda: changed() is not None, 900, "the world did not change")
    last = changed()
    assert last["state"] == "succeeded", last
    assert last["result"]["change"] == "epoch", last
    committed = _admin(both_url, "config/commit")
    assert committed.status_code == 200, committed.text

    def served_config() -> dict:
        status = httpx.get(f"{both_url}/admin/config", timeout=10).json()
        return tomllib.loads(status["config"])

    # A commit may be asked while the target is still being promoted, and it is
    # applied when the trial starts: the configuration moves once the target serves.
    _wait_until(
        lambda: (
            served_config()["cuda_graph_config"]["decode_batch_sizes"]
            == [1, 2, 8]
        ),
        300,
        "the switch's target never served its configuration",
    )
    assert served_config()["server_config"]["port"] == port
    reply = _chat(both_url, name, "Say hello.")
    assert reply["choices"][0]["message"]["content"]
    _wait_until(
        lambda: not _runs(previous),
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
    parses. What catches it is the records — the successor adopted the request
    and finished it — together with a body whose text is a contiguous count and
    whose usage adds up.
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
    _wait_until(
        lambda: _requests_running(base_url) >= 1,
        120,
        "the one-body request was never served",
    )
    restarted = _admin(base_url, "restart")
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

    # The records name that request: frozen here, adopted by the successor at
    # the pause it measured, and finished there.
    _wait_until(
        lambda: len(_handoff_records(log)["finished"]) >= 1,
        300,
        "the successor did not finish the handed-over body",
    )
    # The streaming restart's records are in this same log — every daemon of
    # the run inherits one — so the pause under test is the one whose
    # successor is this restart's: the earlier freezes named another.
    records = _handoff_records(log)
    frozen = [
        record
        for record in records["frozen"]
        if record["successor_pid"] == successor
    ]
    adopted = [
        record for record in records["adopted"] if record["pid"] == successor
    ]
    finished = [
        record for record in records["finished"] if record["pid"] == successor
    ]
    assert len(frozen) == 1, frozen
    assert frozen[0]["adopted"], frozen
    assert frozen[0]["successor_pid"] == successor, frozen
    assert frozen[0]["pause_ms"] >= 0, frozen
    assert [record["request_id"] for record in adopted] == [
        frozen[0]["request_id"]
    ], (adopted, frozen)
    assert adopted[0]["pid"] == successor, adopted
    assert finished[0]["pid"] == successor, finished
    assert finished[0]["at_ns"] > adopted[0]["at_ns"], (adopted, finished)


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
        before = _held(runtime)
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
        assert _held(runtime) == before

        # The deployment serves throughout: the count goes on, a new
        # request is answered, and a world change maps the weights the
        # successor holds.
        reply = _chat(base_url, name, prompt)
        assert reply["choices"][0]["message"]["content"]
        streaming.join(timeout=1800)
        assert done.is_set(), "the count did not end"
        numbers = [int(word) for word in re.findall(r"\d+", "".join(read))]
        assert numbers[:200] == list(range(1, 201)), "".join(read)[:400]
        _change_world(base_url, CHANGE)
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
        ), "the successor world did not map the retained weights"


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


def test_a_world_change_under_storage_pressure_shrinks_the_kv_cache(
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
            report = _change_world(base_url, CHANGE)
            print(
                "world change under pressure, pause: "
                f"{report['pause_seconds']:.3f} s"
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
        grew = [
            line
            for line in lines
            if "growing the KV cache into the storage freed" in line
        ]
        for line in withdrew + gave + grew:
            print(line)
        assert withdrew, "the serving world did not shrink its KV cache"
        assert gave, "the serving world's workers did not give the tail back"
        assert grew, "the committed successor did not grow its KV cache"
        after = _chat(base_url, name, prompt, 32)["choices"][0]["message"][
            "content"
        ]
        assert after == before


def test_a_sleeping_deployment_gives_back_its_cache_and_graph_storage(
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
        lines = _log_lines(log)
        assert any("the worker sleeps" in line for line in lines)
        assert any("the worker woke" in line for line in lines)
        status = httpx.get(f"{base_url}/admin/status", timeout=10).json()
        assert status["deployment"]["asleep"] is False, status
        assert status["deployment"]["paused"] is False, status


#: How long a lost main process has to put an engine back in front of requests
#: before the test calls it a stall. The design states no recovery time, so
#: this is the daemon's own migration bound (``MIGRATION_TIMEOUT``): what a case
#: measures is recorded, and the bound is only ever a stall bound.
RECOVERY_BOUND_S = 120.0

#: The count a recovery case streams across its kill: long enough that the kill
#: lands in the middle of the response, short enough to be compared whole.
RECOVERY_PROMPT = (
    "Count from 1 to 60, separated by commas and spaces. "
    "Write only the numbers."
)

#: The count a request asks for when its tokens are compared: the same prompt
#: every time, so the same numbers are the same answer to the same question.
RECOVERY_TOKENS = 256


def _numbers(read: list[str]) -> list[int]:
    """The integers a streamed answer wrote, in order."""
    return [int(word) for word in re.findall(r"\d+", "".join(read))]


def _serving_engine(log: Path) -> int:
    """The engine process that serves now: the role's pid the log names most.

    A deployment with a standby has two engine processes and only one of them
    answers requests, so the busiest pid of the log's tail is the one to kill;
    a parked standby states itself once and then writes nothing.
    """
    counts: dict[int, int] = {}
    for line in _log_lines(log)[-400:]:
        match = re.search(r"\(Engine pid=(\d+)\)", line)
        if match is not None:
            pid = int(match.group(1))
            counts[pid] = counts.get(pid, 0) + 1
    assert counts, "no engine process is named in the log"
    return max(counts, key=counts.__getitem__)


def _recovery_artifacts(case: str, evidence: dict) -> Path:
    """Writes one case's evidence under `artifacts/main-process-loss/<UTC>/`.

    The recovery time, the tokens the request carried across the loss, and the
    process that was lost are the record: the design states no recovery time, so
    what the case measures is the only statement of it.
    """
    stamp = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    directory = (
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "main-process-loss"
        / stamp
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{case}.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return path


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
    _wait_until(
        lambda: len(_numbers(read)) >= 8, 300, "the count did not stream"
    )
    before = len(_numbers(read))
    killed = _serving_engine(log)
    started = time.monotonic()
    _stop_pid(killed)

    resumed: list[float] = []

    def continues() -> bool:
        if len(_numbers(read)) > before:
            resumed.append(time.monotonic())
            return True
        return False

    _wait_until(
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
    assert len(numbers) >= 60, f"the count stopped at {len(numbers)} numbers"
    assert reasons == ["stop"], f"the count ended {reasons}"

    after = _chat(base_url, name, RECOVERY_PROMPT, tokens=RECOVERY_TOKENS)[
        "choices"
    ][0]["message"]["content"]
    assert _numbers([after]) == baseline_numbers, (
        "the engine that serves after the loss answers differently: "
        f"{after[:200]}"
    )

    lines = _log_lines(log)
    promoted = any("the standby promoted in place" in line for line in lines)
    relaunched = any(
        "the engine was lost; starting another" in line for line in lines
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
    _recovery_artifacts(case, evidence)
    return evidence


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

    The daemon launches a replacement and hands it the epoch's own programs and
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
        before = _pids(log, "Engine")
        baseline = _chat(
            base_url, name, RECOVERY_PROMPT, tokens=RECOVERY_TOKENS
        )["choices"][0]["message"]["content"]
        baseline_numbers = _numbers([baseline])
        successor = tmp_path / worker_executable().name
        shutil.copy2(worker_executable(), successor)

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
        _wait_until(
            lambda: len(_numbers(read)) >= 8, 300, "the count did not stream"
        )
        report = _change_world(
            base_url, f"worker_config.executable={successor}"
        )
        assert report, "the epoch switch reported no world"
        for pid in before:
            _wait_until(
                lambda pid=pid: not _runs(pid),
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
        lines = _log_lines(log)
        failed = [line for line in lines if "chat completion failed" in line]
        assert not failed, failed[:3]
        _recovery_artifacts(
            "epoch-switch-to-a-new-executable",
            {
                "case": "epoch-switch-to-a-new-executable",
                "executable": str(successor),
                "numbers_across_the_switch": len(numbers),
                "answer_matches_the_answer_before": True,
                "failed_requests": 0,
            },
        )


def _update(base_url: str) -> dict:
    """The update in flight or the last one, as ``GET /admin/config`` reports it."""
    status = httpx.get(f"{base_url}/admin/config", timeout=60).json()
    return status.get("update") or {}


# The span a trial's step time and throughput are averaged over.
TRIAL_WINDOW_S = 5


def _slower_executable(directory: Path) -> Path:
    """An executable whose main process steps slower under the same stream.

    It walks each batch operator by operator (``UNISERVE_EAGER_REPLAY``) where
    the deployment launches a captured graph: the same tokens, later, as a
    release that lost a fast path would serve them, with no fault to report.
    """
    path = directory / "uniserve-slower"
    path.write_text(
        "#!/bin/sh\n"
        f'UNISERVE_EAGER_REPLAY=1 exec "{worker_executable()}" "$@"\n'
    )
    path.chmod(0o755)
    return path


def _metric(base_url: str, name: str) -> float:
    """The sum of counter `name` over its label sets."""
    text = httpx.get(f"{base_url}/metrics", timeout=10).text
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith(name) and not line.startswith("#")
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
    _wait_until(
        lambda: len(_numbers(streamed["read"])) >= 8,
        300,
        "the count did not stream",
    )
    return streaming, streamed


def _assert_whole(streaming: threading.Thread, streamed: dict) -> int:
    """Waits for the count's end; it lost and repeated nothing."""
    streaming.join(timeout=1800)
    assert streamed["done"].is_set(), "the count did not end"
    assert not streamed["errors"], f"the count failed: {streamed['errors'][0]!r}"
    numbers = _numbers(streamed["read"])
    wrong = next(
        (
            index
            for index, number in enumerate(numbers)
            if number != index + 1
        ),
        None,
    )
    if wrong is not None:
        pieces = streamed["read"]
        text = "".join(pieces)
        at = text.find(f"{wrong}, ")
        pytest.fail(
            f"the response lost or repeated a count at {wrong}: "
            f"{text[max(at - 60, 0):at + 80]!r}, deltas "
            f"{[piece for piece in pieces[3 * wrong - 12:3 * wrong + 12]]}"
        )
    assert len(streamed["reasons"]) == 1, f"the count ended {streamed['reasons']}"
    return len(numbers)


def _on_trial(base_url: str) -> dict:
    """Waits until the update in flight stands on probation."""
    _wait_until(
        lambda: _update(base_url).get("phase") == "probation",
        600,
        "the update did not reach its trial",
    )
    return _update(base_url)


def _artifact(case: str, evidence: dict) -> None:
    """Writes one case's evidence under ``artifacts/update-options/``."""
    directory = (
        Path(__file__).resolve().parents[3] / "artifacts" / "update-options"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{case}.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize("previous_world", ["parked", "weights", "host"])
def test_an_operator_rollback_returns_to_the_previous_epoch_under_a_request(
    tmp_path: Path, runtime: Path, previous_world: str
):
    """A revert by hand ends the trial with no request lost.

    The previous epoch is kept as ``update_config.previous_world`` states: its
    drained processes stay alive through the trial, or they exit with its
    weights kept, and the revert starts them again on those.
    """
    name = "qwen3-live-rollback"
    log = tmp_path / "server.log"
    with _serve(runtime, log, name) as base_url:
        before = _pids(log, "Engine")
        answer = _chat(base_url, name, RECOVERY_PROMPT, tokens=32)
        successor = tmp_path / worker_executable().name
        shutil.copy2(worker_executable(), successor)
        streaming, streamed = _start_count(base_url, name)
        result = _apply(
            base_url,
            f"worker_config.executable={successor}",
            f"update_config.previous_world={previous_world}",
        )
        assert result["change"] == "epoch", result
        trial = _on_trial(base_url)
        if previous_world != "parked":
            for pid in before:
                _wait_until(
                    lambda pid=pid: not _runs(pid),
                    120,
                    f"engine {pid} outlived the promotion",
                )
        else:
            assert all(_runs(pid) for pid in before)
        # A node that cannot move the weights keeps them on its GPUs, and says why.
        _wait_until(
            lambda: previous_world == "parked"
            or _update(base_url)["moved_to_host_bytes"] > 0
            or _update(base_url)["previous_world_kept"] == previous_world,
            120,
            "the previous epoch's storage was not kept",
        )
        trial = _update(base_url)
        print("kept:", json.dumps(trial))
        if not trial["host_refused"]:
            assert trial["previous_world_kept"] == previous_world, trial
        rolled = _admin(base_url, "config/rollback", {"reason": "the check"})
        assert rolled.status_code == 200, rolled.text
        count = _assert_whole(streaming, streamed)
        after = _chat(base_url, name, RECOVERY_PROMPT, tokens=32)
        assert (
            after["choices"][0]["message"]["content"]
            == answer["choices"][0]["message"]["content"]
        )
        update = _update(base_url)
        assert update["phase"] == "abandoned", update
        assert update["failure"]["reason"] == "the check", update
        _artifact(
            f"operator-rollback-{previous_world}",
            {"numbers_across_the_switch": count, "update": update},
        )


def test_a_regression_of_the_epoch_on_trial_reverts_it(
    tmp_path: Path, runtime: Path
):
    """A target that steps slower past the threshold reverts by itself.

    The target's executable walks every batch instead of replaying a graph, so
    the same stream steps slower on it with no fault. The stream crosses the
    switch and the revert whole, the update ends unavailable with the
    threshold's reason, and applying it again needs ``force``. Forced without
    the threshold, the same slower target stands its trial: the threshold, and
    nothing else, reverted it.
    """
    name = "qwen3-live-regression"
    log = tmp_path / "server.log"
    thresholds = (
        "update_config.rollback.step_time_regression=0.2",
        f'update_config.rollback.window="{TRIAL_WINDOW_S}s"',
    )
    with _serve(runtime, log, name) as base_url:
        answer = _chat(base_url, name, RECOVERY_PROMPT, tokens=32)
        slower = f"worker_config.executable={_slower_executable(tmp_path)}"
        streaming, streamed = _start_count(base_url, name)
        time.sleep(TRIAL_WINDOW_S + 2)
        result = _apply(base_url, slower, *thresholds)
        assert result["change"] == "epoch", result
        _wait_until(
            lambda: _update(base_url).get("phase") == "abandoned",
            600,
            "the slower epoch was not reverted",
        )
        update = _update(base_url)
        assert update["failure"]["kind"] == "step_time_regression", update
        print("reverted:", json.dumps(update))
        (replica,) = update["trial"]["replicas"]
        assert replica["baseline"]["tokens_per_second"] > 0, update
        assert replica["rates"]["step_us"] > replica["baseline"]["step_us"], update
        count = _assert_whole(streaming, streamed)
        after = _chat(base_url, name, RECOVERY_PROMPT, tokens=32)
        assert (
            after["choices"][0]["message"]["content"]
            == answer["choices"][0]["message"]["content"]
        )

        again = httpx.put(
            f"{base_url}/admin/config",
            json={"set": [slower]},
            timeout=1800,
        )
        assert again.status_code != 200, again.text
        assert "force" in again.text, again.text

        streaming, streamed = _start_count(base_url, name)
        time.sleep(TRIAL_WINDOW_S + 2)
        forced = httpx.put(
            f"{base_url}/admin/config",
            json={"set": [slower], "force": True},
            timeout=1800,
        )
        assert forced.status_code == 200, forced.text
        _on_trial(base_url)
        time.sleep(2 * TRIAL_WINDOW_S)
        unthresholded = _update(base_url)
        assert unthresholded["phase"] == "probation", unthresholded
        committed = _admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text
        forced_count = _assert_whole(streaming, streamed)
        _artifact(
            "step-time-regression",
            {
                "numbers_across_the_revert": count,
                "numbers_across_the_forced_commit": forced_count,
                "update": update,
                "without_the_threshold": unthresholded,
            },
        )


def test_a_program_update_past_a_threshold_reverts_in_its_epoch(
    tmp_path: Path, runtime: Path
):
    """A program-layer update stands the same trial as an epoch switch.

    The serving epoch computes a stream before the update and a trickle after
    it, so its token throughput falls past the threshold. The update reverts in
    the same main process, with the threshold as its failure.
    """
    name = "qwen3-live-program-trial"
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
        assert result["change"] == "program", result
        streaming.join(timeout=1800)
        assert not streamed["errors"], streamed["errors"][0]

        def reverted() -> bool:
            _chat(base_url, name, RECOVERY_PROMPT, tokens=2)
            time.sleep(1)
            return bool(_update(base_url).get("failure"))

        _wait_until(reverted, 120, "the update was not reverted")
        update = _update(base_url)
        print("reverted:", json.dumps(update))
        assert update["failure"]["kind"] == "throughput_regression", update
        _artifact("program-throughput-regression", {"update": update})


@pytest.mark.xfail(
    strict=True,
    reason="a main process maps no host page pool, so no page reaches the host "
    "(crates/engine/src/scheduler/submission.rs, map_host_cache)",
)
def test_the_requests_in_flight_warm_the_target_through_the_host_cache(
    tmp_path: Path, runtime: Path
):
    """The switchover waits for the tail ``cache_tail_tokens`` states.

    A request's pages reach the host KV cache as they complete, so the target's
    admission finds all but its tail there: it computes that tail and not the
    request. The prefix counters of the target say how much it read.
    """
    name = "qwen3-live-warming"
    log = tmp_path / "server.log"
    with _serve(runtime, log, name) as base_url:
        successor = tmp_path / worker_executable().name
        shutil.copy2(worker_executable(), successor)
        streaming, streamed = _start_count(base_url, name)
        _wait_until(
            lambda: len(_numbers(streamed["read"])) >= 400,
            300,
            "the count did not grow",
        )
        hits = _metric(base_url, "uniserve:prefix_cache_host_hits")
        result = _apply(
            base_url,
            f"worker_config.executable={successor}",
            "update_config.cache_tail_tokens=64",
            "update_config.cache_prefill_tokens=128",
        )
        assert result["change"] == "epoch", result
        trial = _on_trial(base_url)
        warming = trial["warming"]
        print("warming:", json.dumps(warming))
        assert warming["warmed_requests"] == 1, warming
        assert warming["limit_tokens"] == 64, warming
        assert warming["tail_tokens"] <= 64, warming
        assert warming["warmed_tokens"] > 0, warming
        time.sleep(3)
        read = _metric(base_url, "uniserve:prefix_cache_host_hits") - hits
        print("host hits since the switch:", read)
        assert read >= warming["warmed_tokens"] - 64, (read, warming)
        committed = _admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text
        count = _assert_whole(streaming, streamed)
        _artifact(
            "warming",
            {
                "numbers_across_the_switch": count,
                "warming": warming,
                "host_hit_tokens_on_the_target": read,
            },
        )
