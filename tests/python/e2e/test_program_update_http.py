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
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    start_server,
    stop_server,
    wait_health,
)

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


@pytest.fixture(scope="module")
def program_server(tmp_path_factory: pytest.TempPathFactory):
    """The host's Program Server, with a writable package root of its own."""
    binary = require_uniserve_binary()
    work = tmp_path_factory.mktemp("program-server")
    socket = work / "program-server.sock"
    packages = work / "packages"
    packages.mkdir()
    log = work / "program-server.log"
    process = start_server(
        [
            str(binary),
            "program-server",
            "serve",
            "--socket",
            str(socket),
            "--package-root",
            str(packages),
            "--target",
            os.environ.get("UNISERVE_PROGRAM_TARGET", "sm_100a"),
        ],
        log,
    )
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the Program Server exited: {log.read_text()}")
        if socket.exists():
            break
        time.sleep(0.2)
    else:
        raise TimeoutError(f"the Program Server left no socket at {socket}")
    try:
        yield {"socket": socket, "binary": binary, "log": log}
    finally:
        stop_server(process)


@pytest.fixture(scope="module")
def server(
    model: Path, program_server, tmp_path_factory: pytest.TempPathFactory
):
    binary = require_uniserve_binary()
    work = tmp_path_factory.mktemp("deployment")
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = start_server(
        [
            str(binary),
            "serve",
            str(model),
            "--served-model-name",
            "qwen3-moe",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--program-server",
            str(program_server["socket"]),
        ],
        work / "server.log",
    )
    wait_health(base_url, process, 900.0)
    try:
        yield base_url
    finally:
        stop_server(process)


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
    """The epoch the deployment serves, as the daemon records it (5.1)."""
    response = httpx.get(f"{base_url}/admin/config", timeout=60.0)
    response.raise_for_status()
    epochs = response.json()["epochs"]
    serving = [
        epoch
        for epoch in epochs
        if epoch.get("status") in ("serving", "probation", "precopy")
    ]
    assert len(serving) == 1, f"expected one serving epoch, got {epochs}"
    return serving[0]


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
    assert any(
        str(revision) in json.dumps(entry)
        for entry in after["kernels"].values()
        if isinstance(entry, dict)
    ), f"the recorded programs name no revision {revision}: {after['kernels']}"

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

    assert apply_config(server).status_code == 200
    wait_for_phases(server, PRECOPY)
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

    response = apply_config(server)
    if response.status_code == 200:
        events = wait_for_phases(server, ["precopy_failed"])
        reasons = [
            json.dumps(event)
            for event in events
            if _phase_of(event) == "precopy_failed"
        ]
        assert reasons, "precopy_failed carried no reason"
    else:
        assert (
            "precopy" in response.text.lower()
            or "tolerance" in response.text.lower()
        ), (
            "a wrong kernel was refused for the wrong reason: "
            f"{response.text[:500]}"
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
