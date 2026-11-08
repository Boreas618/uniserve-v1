"""A program-layer update while a deployment serves (design 7.3, 11 step 2).

One Qwen3-MoE deployment serves on one GPU while the test replaces kernel
programs and the scheduling policy under it. The pass criteria are the delivery
test's:

- one kernel program swapped by **rebind**, one by **recapture** together with
  the **policy**, then **reverted**, with no request failing at any step;
- a deliberately **wrong kernel** refused by shadow execution at precopy: the
  serving generation does not change and requests keep succeeding;
- the allocator's mapped bytes stay within the device's budget while both
  generations' graphs exist.

The Program Server runs beside the deployment (design 6.1): the test starts it,
installs a variant package into it, and applies the unchanged deployment
configuration, which is what triggers a program-layer update (5.1). Phases are
asserted on the daemon's events, never on sleeps.

Inputs, all optional except the model, so the file skips where the fixtures do
not exist:

- ``UNISERVE_QWEN3_MOE_MODEL``: the Qwen3-MoE BF16 checkpoint.
- ``UNISERVE_PROGRAM_REBIND_PACKAGE``: a package of an operator the slice
  launches every step, with the serving descriptor key and the same launch
  structure (a rebind swap).
- ``UNISERVE_PROGRAM_RECAPTURE_PACKAGE``: a package of one operator whose single
  entry launches two kernels (a native module; the recapture case of 4.5).
- ``UNISERVE_PROGRAM_POLICY_PACKAGE``: a scheduling-policy package (wasm) whose
  parameters differ from the serving one.
- ``UNISERVE_PROGRAM_WRONG_PACKAGE``: a package of the rebind operator whose
  result is outside ``Op::tolerance``.
- ``UNISERVE_PROGRAM_TARGET``: the Program Server's target (``sm_100a``).

The admin surface is front's (M488): `POST /admin/config/commit` and
`/admin/config/rollback`, events whose `data.phase` names the phase and whose
`kind` is `deployment`, and `/admin/config`'s `epochs` array whose elements
carry `kernels` and `previous_kernels` as a list of
`{op, phase, program: {backend, revision, arguments}}`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
    start_server,
    stop_server,
    wait_health,
    worker_executable,
)

#: Options every deployment this file serves with. They are the live-update file's, deliberately:
#: the two files serve the same model on the same host, and a deployment that stated a different
#: attention backend, graph catalogue or token budget would be a second deployment whose numbers
#: nobody could compare with the first's.
#:
#: `attention_backend` in particular is not optional: a program is selected by its backend's name,
#: so a deployment left at `auto` states no package and its epoch cannot be prepared at all.
DEPLOYMENT_OPTIONS = (
    "dtype=bfloat16",
    "attention_backend=trtllm",
    "kv_cache_config.max_tokens=32768",
    "chat_config.reasoning_parser=none",
    "cuda_graph_config.prefill_token_sizes=[512]",
    "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
    "max_num_batched_tokens=512",
)

#: Host KV cache this deployment holds, in GiB. The live-update file serves the same model
#: with the same figure, so the two files' deployments size their host memory alike.
HOST_GIB = 8

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

PROMPTS = [
    "Explain what a KV cache is in two sentences.",
    "Name three prime numbers above one hundred.",
    "Write one sentence about the sea.",
    "What is 17 times 23?",
    "Give a synonym for 'quick'.",
    "Summarize the plot of a story about a lighthouse.",
    "Translate 'good morning' into French.",
    "List two uses of a thermometer.",
]

CONCURRENCY = 8
MAX_TOKENS = 64
STEP_TIMEOUT_S = 900.0

# The phases the daemon reports for one update (design 7.3 steps 2-6).
PRECOPY = ["precopy", "precopied", "switchover", "switched", "probation"]
COMMITTED = ["committed", "purged"]
REVERTED = ["reverted", "purged"]


def _required(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} must name a package directory")
    path = Path(value)
    if not path.is_dir():
        pytest.fail(f"{name} names {path}, which is not a directory")
    return path


@pytest.fixture(scope="module")
def model() -> Path:
    value = os.environ.get("UNISERVE_QWEN3_MOE_MODEL")
    if not value:
        pytest.skip(
            "UNISERVE_QWEN3_MOE_MODEL must name the Qwen3-MoE checkpoint"
        )
    path = Path(value)
    if not path.exists():
        pytest.fail(
            f"UNISERVE_QWEN3_MOE_MODEL names {path}, which does not exist"
        )
    return path


#: The descriptor keys this deployment launches: the model's own ``keys()`` plus the executor's
#: sampler launches (``notes/programs/qwen3-moe-keys.txt``). A main process resolves the kernels the
#: model actually launches from this host's store, so the store must hold a module for every one.
KEYS = "notes/programs/qwen3-moe-keys.txt"

#: The rows of ``KEYS`` that belong to the file's other two deployments (NVFP4 weights and an fp8
#: KV variant). This file serves the bf16 dense one; building another deployment's rows fails on
#: packages that deployment's owner owes, which says nothing about whether this one can serve.
FOREIGN_PRECISIONS = ("nvfp4", "f8e4m3")
FOREIGN_OPERATORS = frozenset({"kv_write_scaled"})


def _repository() -> Path:
    """The repository root."""
    return Path(__file__).resolve().parents[3]


def _precompiler() -> Path:
    """The wasm precompiler the Program Server turns a policy module into (design 5.3)."""
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
        ["cargo", "build", "--package", "scheduling", "--bin", "scheduling-precompile"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if done.returncode != 0:
        pytest.fail(f"scheduling-precompile did not build: {done.stderr[-2000:]}")
    return root / "target" / "debug" / "scheduling-precompile"


def _program_target() -> str:
    """The target this host's modules are built for: the served device's own ``sm_{major}{minor}a``.

    ``src/model_runner.rs`` derives the same string from the device it serves, and a store built for
    another target describes nothing to overlay. ``UNISERVE_PROGRAM_TARGET`` names one instead, for
    a host whose store is provisioned elsewhere.
    """
    named = os.environ.get("UNISERVE_PROGRAM_TARGET")
    if named:
        return named
    device = os.environ.get("UNISERVE_LIVE_DEVICE", "0")
    reported = subprocess.run(
        ["nvidia-smi", f"--id={device}", "--query-gpu=compute_cap", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    major, _, minor = reported.partition(".")
    if not major.isdigit() or not minor.isdigit():
        raise AssertionError(
            f"nvidia-smi reported no compute capability for GPU {device}: {reported!r}"
        )
    return f"sm_{major}{minor}a"


def _deployment_keys(store: Path) -> Path:
    """This deployment's rows of ``KEYS``, written beside ``store``."""
    kept = []
    for line in (_repository() / KEYS).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        operator, _, _key = stripped.partition("\t")
        if operator in FOREIGN_OPERATORS:
            continue
        lowered = stripped.lower()
        if any(precision in lowered for precision in FOREIGN_PRECISIONS):
            continue
        kept.append(stripped)
    if not kept:
        raise AssertionError(f"{KEYS} states no row for this deployment")
    path = store.parent / "deployment-keys.txt"
    path.write_text("\n".join(kept) + "\n")
    return path


def _build_store(store: Path, target: str, python: Path | None) -> None:
    """Build a module for every kernel key this deployment launches.

    ``build_fixtures`` is the store's own producer: for each ``operator<TAB>key`` row it installs
    the operator's package, prepares that key for `target`, and resolves it to the module it names.
    The builds are real — Triton, nvcc and CuTe DSL all run — so this takes minutes once per module.
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
        command, cwd=_repository(), capture_output=True, text=True, timeout=3600
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


@pytest.fixture(scope="module")
def program_server(tmp_path_factory: pytest.TempPathFactory):
    """The host's Program Server, serving a store built for this deployment.

    The server's package root and store are built here rather than left empty: an empty store
    answers `no paged_attention program of backend trtllm is installed`, because a program's module
    exists nowhere else. The kernels come from ``build_fixtures`` over this deployment's own rows,
    and the scheduling policy is installed on top of them (a policy is a wasm package this host
    precompiles rather than a kernel key).

    The two directories live as long as the module and are not shared with another run: a store
    entry is keyed by operator, backend, revision, target and descriptor, and a revision counts
    from 1 in a fresh package root.
    """
    binary = require_uniserve_binary()
    work = tmp_path_factory.mktemp("program-server")
    socket = work / "program-server.sock"
    log = work / "program-server.log"
    target = _program_target()
    interpreter = _repository() / "crates/program-server/python/.venv/bin/python"
    built = work / "programs"
    _build_store(built, target, interpreter if interpreter.exists() else None)
    command = [
        str(binary),
        "program-server",
        "serve",
        "--socket",
        str(socket),
        "--package-root",
        str(built / "packages"),
        "--store-root",
        str(built / "modules"),
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

    installed = subprocess.run(
        [
            str(binary),
            "program-server",
            "install",
            "--socket",
            str(socket),
            str(_repository() / "crates/ops/program-fixtures/policy"),
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if installed.returncode != 0:
        raise AssertionError(
            f"installing the policy package failed: {installed.stdout}\n{installed.stderr}"
        )
    try:
        yield {"socket": socket, "binary": binary, "log": log}
    finally:
        stop_server(process)


@pytest.fixture(scope="module")
def server(
    model: Path, program_server, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[str]:
    """One deployment of `model` beside this host's Program Server.

    Module-scoped, like the epoch history the tests read back.

    Every test here requests `server`, and until this fixture existed none of
    them could start: pytest fails the setup of a test whose fixture is
    undefined and reports it as an error rather than a failure, so a run of the
    file read "no failures" with no test executed.

    The deployment is module-scoped because the tests are one epoch history —
    a rebind, a recapture with its policy, a revert and a refusal — each read
    back through `/admin/status` and `/metrics` against the same deployment's
    records. The options are `DEPLOYMENT_OPTIONS`, the live-update file's
    deliberately, and the host KV cache is `HOST_GIB`, the figure that file
    serves the same model with.
    """
    directory = tmp_path_factory.mktemp("deployment")
    log = directory / "deployment.log"
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(require_uniserve_binary()),
        "serve",
        str(model),
        "--served-model-name",
        "qwen3-moe",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-executable",
        str(worker_executable()),
        "--program-server",
        str(program_server["socket"]),
        "--set",
        f"kv_cache_config.host_gib={HOST_GIB}",
    ]
    for option in DEPLOYMENT_OPTIONS:
        command += ["--set", option]
    with server_process(command, base_url, log, timeout_s=1800):
        yield base_url


RESIDENT_TOKENS = 512

# The two sentences a wrong-kernel refusal can carry. The comparison's own
# sentence names the element
# that left tolerance and the bound it left; the idle sentence says no request
# ran a step, which means
# nothing was compared. A guard that accepts both cannot tell a refused kernel
# from an update that
# never ran one.
COMPARED = "differ beyond tolerance"
IDLE = "no resident request ran a step"


class Resident:
    """One streamed completion, held resident for a whole update window.

    Shadow execution pairs a changed operator's kernel on one step of the
    batches a scheduler forms from its **resident** requests. A closed loop of
    short requests has moments with none in flight, and a precopy that lands in
    one of them compares nothing and fails by name ("no resident request ran a
    step"). Holding one request open for as long as the caller keeps it makes
    the wrong-kernel assertion mean what it says.

    The stream is drained on its own thread so the daemon is never blocked by
    backpressure, and `close` ends it: the caller closes it after the update's
    answer has arrived and before any failure is read, so a request in flight
    never races the assertions.
    """

    def __init__(self, base_url: str, prompt: str) -> None:
        self._closed = threading.Event()
        self._client = httpx.Client(timeout=STEP_TIMEOUT_S)
        request = self._client.build_request(
            "POST",
            f"{base_url}/v1/chat/completions",
            json={
                "model": "qwen3-moe",
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": RESIDENT_TOKENS,
                "stream": True,
            },
        )
        self._response = self._client.send(request, stream=True)
        if self._response.status_code != 200:
            body = self._response.read()[:300]
            self.close()
            raise AssertionError(
                "the resident request was refused: "
                f"{self._response.status_code}: {body!r}"
            )
        self._lines = self._response.iter_lines()
        # The first line proves the engine formed a batch for this request: it
        # is resident, not waiting behind admission.
        # An SSE comment or a blank line can precede the first frame, so a few
        # are read before the stream is called silent.
        first = ""
        for _ in range(4):
            first = next(self._lines, "")
            if "data:" in first:
                break
        if "data:" not in first:
            self.close()
            raise AssertionError(
                f"the resident request produced no stream: {first!r}"
            )
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        try:
            for _ in self._lines:
                if self._closed.is_set():
                    return
        except (httpx.HTTPError, OSError):
            # A stream the client closed is the caller's own doing, not a
            # failure of the deployment.
            return

    def close(self) -> None:
        self._closed.set()
        try:
            self._response.close()
        finally:
            self._client.close()


class Traffic:
    """A closed loop of requests that records every failure as it happens."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.failures: list[str] = []
        self.completed = 0
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(target=self._loop, args=(prompt,), daemon=True)
            for prompt in PROMPTS[:CONCURRENCY]
        ]
        for thread in self._threads:
            thread.start()

    def _one(self, prompt: str) -> None:
        body = {
            "model": "qwen3-moe",
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": MAX_TOKENS,
            "stream": False,
        }
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions", json=body, timeout=600.0
        )
        if response.status_code != 200:
            self.failures.append(
                f"{response.status_code}: {response.text[:300]}"
            )
            return
        payload = response.json()
        choice = payload["choices"][0]
        if not choice.get("message", {}).get("content"):
            self.failures.append(f"empty content: {json.dumps(payload)[:300]}")
            return
        if choice.get("finish_reason") in (None, "migratable"):
            self.failures.append(
                f"no finish reason: {json.dumps(choice)[:300]}"
            )
            return
        self.completed += 1

    def _loop(self, prompt: str) -> None:
        while not self._stop.is_set():
            try:
                self._one(prompt)
            except Exception as error:  # noqa: BLE001 - a request that cannot be
                # made is a failure of the update under test, not of the test.
                self.failures.append(f"{type(error).__name__}: {error}")

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=120.0)


@pytest.fixture(scope="module")
def traffic(server: str):
    loop = Traffic(server)
    deadline = time.monotonic() + 300.0
    while loop.completed < CONCURRENCY and time.monotonic() < deadline:
        time.sleep(0.5)
    assert loop.completed >= CONCURRENCY, "the deployment served nothing"
    try:
        yield loop
    finally:
        loop.stop()


def install(program_server, package: Path) -> int:
    """Install `package` on the Program Server and answer the new revision."""
    done = subprocess.run(
        [
            str(program_server["binary"]),
            "program-server",
            "install",
            str(package),
            "--socket",
            str(program_server["socket"]),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if done.returncode != 0:
        raise AssertionError(f"install failed: {done.stdout}\n{done.stderr}")
    match = re.search(r"revision[^0-9]*([0-9]+)", done.stdout)
    if not match:
        raise AssertionError(f"install printed no revision: {done.stdout}")
    return int(match.group(1))


def apply_config(base_url: str, *, force: bool = False) -> httpx.Response:
    """Apply the unchanged configuration, which triggers an update."""
    body: dict[str, object] = {}
    if force:
        body["force"] = True
    return httpx.put(
        f"{base_url}/admin/config", json=body, timeout=STEP_TIMEOUT_S
    )


def end_update(base_url: str, action: str) -> None:
    """Commit or revert the update on probation (design 7.3 step 5)."""
    response = httpx.post(
        f"{base_url}/admin/config/{action}", timeout=STEP_TIMEOUT_S
    )
    assert response.status_code == 200, (
        f"POST /admin/config/{action} answered {response.status_code}: "
        f"{response.text[:500]}"
    )


def _phase_of(event: dict) -> str:
    """The phase an event names, in the daemon's vocabulary (front, M357)."""
    for source in (event.get("data"), event):
        if isinstance(source, dict):
            for key in ("phase", "kind"):
                value = source.get(key)
                if isinstance(value, str):
                    return value
    return ""


def wait_for_phases(
    base_url: str, phases: list[str], after: int = 0
) -> list[dict]:
    """Long-poll the daemon's events until every phase arrived."""
    seen: list[dict] = []
    pending = list(phases)
    deadline = time.monotonic() + STEP_TIMEOUT_S
    next_seq = after
    while pending:
        if time.monotonic() > deadline:
            raise TimeoutError(f"phases {pending} did not arrive; saw {seen}")
        response = httpx.get(
            f"{base_url}/admin/events",
            params={"after": next_seq, "wait": 30},
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
        next_seq = payload.get("next", next_seq)
        for event in payload.get("events", []):
            seen.append(event)
            phase = _phase_of(event)
            if phase in pending:
                pending.remove(phase)
    return seen


def epoch_record(base_url: str) -> dict:
    """The epoch the deployment serves, as the daemon records it (5.1).

    `/admin/config` reports every epoch, ascending; the one that serves is the
    highest-numbered epoch in `serving` or `probation`, and `/admin/events`
    names it by number (`data.epoch`).
    """
    response = httpx.get(f"{base_url}/admin/config", timeout=60.0)
    response.raise_for_status()
    epochs = response.json()["epochs"]
    live = [
        epoch
        for epoch in epochs
        if epoch.get("status") in ("serving", "probation")
    ]
    assert live, f"the deployment serves no epoch: {epochs}"
    return max(live, key=lambda epoch: epoch["epoch"])


def metrics(
    base_url: str,
) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    """The deployment's Prometheus text, keyed by name and labels."""
    text = httpx.get(f"{base_url}/metrics", timeout=30.0).text
    values: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    pattern = re.compile(
        r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9.eE+-]+)$"
    )
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        raw = match.group(2) or ""
        labels = tuple(sorted(re.findall(r'([a-z_]+)="([^"]*)"', raw)))
        values[(match.group(1), labels)] = float(match.group(3))
    return values


def mapped_within_budget(base_url: str) -> None:
    values = metrics(base_url)
    budgets = {
        labels: value
        for (name, labels), value in values.items()
        if name == "uniserve:memory_budget"
    }
    mapped: dict[tuple[tuple[str, str], ...], float] = {}
    for (name, labels), value in values.items():
        if name != "uniserve:memory_mapped":
            continue
        labels_by_name = dict(labels)
        key = (("device", labels_by_name.get("device")),)
        mapped[key] = mapped.get(key, 0.0) + value
    assert budgets, "the deployment published no memory budget"
    for device, budget in budgets.items():
        assert mapped.get(device, 0.0) <= budget, (
            f"device {device}: mapped {mapped.get(device)}"
            f" exceeds budget {budget}"
        )


def assert_no_failures(step: str, traffic: Traffic) -> None:
    assert not traffic.failures, (
        f"requests failed during {step}: {traffic.failures}"
    )


def test_a_kernel_program_is_swapped_by_rebind_while_requests_stream(
    server: str, program_server, traffic: Traffic
) -> None:
    package = _required("UNISERVE_PROGRAM_REBIND_PACKAGE")
    before = epoch_record(server)
    revision = install(program_server, package)

    assert apply_config(server).status_code == 200
    wait_for_phases(server, PRECOPY)
    after = epoch_record(server)
    assert after["kernels"] != before["kernels"], (
        "the program-layer update did not change the recorded kernel programs"
    )
    revisions = {
        entry["program"]["revision"]
        for entry in after["kernels"]
        if isinstance(entry.get("program"), dict)
    }
    assert revision in revisions, (
        f"the recorded programs name no revision {revision}: {after['kernels']}"
    )

    end_update(server, "commit")
    wait_for_phases(server, COMMITTED)
    assert_no_failures("the rebind swap", traffic)
    mapped_within_budget(server)


def test_the_recapture_swap_and_the_policy_swap_keep_serving(
    server: str, program_server, traffic: Traffic
) -> None:
    package = _required("UNISERVE_PROGRAM_RECAPTURE_PACKAGE")
    policy = _required("UNISERVE_PROGRAM_POLICY_PACKAGE")
    before = epoch_record(server)
    install(program_server, package)
    install(program_server, policy)

    # Shadow execution compares on a step of a resident request; without one the
    # precopy refuses by name and the swap below would be asserted on an update
    # that compared nothing.
    resident = Resident(server, PROMPTS[0])
    try:
        assert apply_config(server).status_code == 200
        wait_for_phases(server, PRECOPY)
    finally:
        # Closed after the update answered and before any failure is read: a
        # request in flight here would race `assert_no_failures` below.
        resident.close()
    after = epoch_record(server)
    assert after["kernels"] != before["kernels"], (
        "the recapture swap did not change the recorded kernel programs"
    )
    assert after["policy"] != before["policy"], (
        "the policy swap did not change the recorded policy"
    )
    assert_no_failures("the recapture and policy swap", traffic)
    mapped_within_budget(server)

    # The update stays on probation for the revert below (7.3 step 4).
    assert after["status"] == "probation", (
        f"expected probation, got {after['status']}"
    )


def test_revert_restores_the_previous_generation(
    server: str, traffic: Traffic
) -> None:
    before = epoch_record(server)
    end_update(server, "rollback")
    wait_for_phases(server, REVERTED)
    assert before["previous_kernels"] is not None, (
        "the epoch on probation kept no previous kernel programs to revert to"
    )
    after = epoch_record(server)
    assert after["kernels"] == before["previous_kernels"], (
        "the revert did not restore the previous kernel programs"
    )
    assert after["policy"] == before["previous_policy"], (
        "the revert did not restore the previous policy"
    )
    assert_no_failures("the revert", traffic)
    mapped_within_budget(server)


def test_a_wrong_kernel_is_refused_by_shadow_execution(
    server: str, program_server, traffic: Traffic
) -> None:
    package = _required("UNISERVE_PROGRAM_WRONG_PACKAGE")
    before = epoch_record(server)
    install(program_server, package)

    # The comparison needs a request the scheduler is stepping when the precopy
    # shadows: a wrong
    # kernel can only be refused by a comparison that ran.
    resident = Resident(server, PROMPTS[1])
    try:
        response = apply_config(server)
        if response.status_code == 200:
            events = wait_for_phases(server, ["precopy_failed"])
            reasons = [
                json.dumps(event)
                for event in events
                if _phase_of(event) == "precopy_failed"
            ]
            assert reasons, "precopy_failed carried no reason"
            refusal = " ".join(reasons)
        else:
            refusal = response.text
    finally:
        resident.close()

    # The comparison's own sentence, not a refusal for any reason at all: the
    # element past tolerance and the bound it left, and never the sentence that
    # says nothing was compared.
    # The refusal is printed whole: its first clause names the step that failed
    # and the rest carries the engine's own sentence, so a slice at 800
    # characters can hide the very string these assertions are about.
    assert COMPARED in refusal, (
        "a wrong kernel was refused without the comparison's sentence: "
        f"{refusal}"
    )
    assert "tolerance atol" in refusal, (
        f"the refusal names no tolerance: {refusal}"
    )
    assert IDLE not in refusal, (
        "the wrong kernel was not compared at all: the update was refused "
        f"because no resident request ran a step: {refusal}"
    )

    after = epoch_record(server)
    assert after["kernels"] == before["kernels"], (
        "a refused update changed the serving generation"
    )
    assert after["status"] in ("serving", "probation"), after["status"]
    assert_no_failures("the refused wrong kernel", traffic)
    mapped_within_budget(server)


def test_no_request_failed_and_every_mapping_stayed_within_budget(
    server: str, traffic: Traffic
) -> None:
    mapped_within_budget(server)
    assert traffic.completed > 0
    assert not traffic.failures, (
        f"requests failed during the update: {traffic.failures}"
    )
