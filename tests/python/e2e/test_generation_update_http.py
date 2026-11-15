"""A generation update while a deployment serves (``specs/updates.md`` §3.1).

One Qwen3-MoE deployment serves on one GPU while the test replaces kernel
packages and the scheduling policy under it:

- one kernel package swapped by **rebind**, one by **recapture** together with
  the **policy**, then **rolled back**, with no request failing at any step;
- a deliberately **wrong kernel** refused by the shadow while the update
  prepares: the update is abandoned, the serving generation does not change
  and requests keep succeeding;
- the allocator's mapped bytes stay within the device's budget while both
  generations' graphs exist.

The Build Server runs beside the deployment: the test starts it, installs a
variant package into it, and applies the unchanged deployment configuration,
which is what moves a newly installed revision into a generation update
(``docs/admin.md``, Updates). Phases are asserted on the daemon's events,
never on sleeps.

The model is ``UNISERVE_QWEN3_MODEL``'s checkpoint; the module is skipped
when it is unset. The packages are the tree's fixtures under
``crates/ops/package-fixtures``, and each variable below names another:

- ``UNISERVE_REBIND_PACKAGE``: a package of an operator the slice
  launches every step, with the serving descriptor key and the same launch
  structure (a rebind swap).
- ``UNISERVE_RECAPTURE_PACKAGE``: a package of one operator whose single
  entry launches two kernels (a native module; the recapture case of 4.5).
- ``UNISERVE_POLICY_PACKAGE``: a scheduling-policy package (wasm) of the
  serving policy's backend, which installing makes its next revision.
- ``UNISERVE_WRONG_PACKAGE``: a package of the rebind operator whose
  result is outside ``Op::tolerance``.
- ``UNISERVE_MODULE_TARGET``: the Build Server's target, derived from the
  served device (``sm_100a`` on Blackwell) when unset.

The Build Server's toolchain environment is its program on the search path
(``uniserve-build-server``), and the binary is named by
``UNISERVE_BINARY``.

The admin surface is the operator socket's: `POST /admin/config/commit` and
`/admin/config/rollback`, events whose `kind` is `deployment` and whose
`data` is the update as it stood, its `phase` among them, and
`/admin/config`'s `epochs` array whose elements carry `kernels` and
`previous_kernels` as a list of `{op, phase, package: {backend, revision,
arguments}}`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.build_server import POLICY_PACKAGE, repository
from tests.python.e2e.build_server import (
    build_server as running_build_server,
)
from tests.python.e2e.http_helpers import (
    QWEN3_CHECKPOINT,
    admin,
    admin_config,
    admin_request,
    admin_status,
    apply_config,
    engine_selection,
    host_asset,
    require_uniserve_binary,
    server_process,
)

#: Options every deployment this file serves with. They are the live-update
#: file's, deliberately: the two files serve the same model on the same host,
#: and a deployment that stated a different attention backend, graph catalogue
#: or token budget would be a second deployment whose numbers nobody could
#: compare with the first's.
#:
#: `attention_backend` in particular is not optional: a package is selected by
#: its backend's name, so a deployment left at `auto` states no package and its
#: epoch cannot be prepared at all.
DEPLOYMENT_OPTIONS = (
    "dtype=bfloat16",
    "attention_backend=trtllm",
    "kv_cache_config.max_tokens=32768",
    "chat_config.reasoning_parser=none",
    "cuda_graph_config.prefill_token_sizes=[512]",
    "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
    "max_num_batched_tokens=512",
)

#: The host tier this deployment holds, in GiB. The live-update file serves the
#: same model with the same figure, so the two files' deployments size their
#: host memory alike.
HOST_GIB = 8

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("qwen3"),
    # The module's cases are one deployment's history.
    pytest.mark.gpus(1),
]

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

# The phases the daemon reports for one update, up to its trial and after it
# (``docs/admin.md``, One update, two forms).
TO_TRIAL = ["preparing", "switching", "trial"]
COMMITTED = ["committed"]
ROLLED_BACK = ["rolled_back"]


#: The tree's package each variable stands for when it is unset: a kernel
#: fixture, or the deployment's own policy package, whose second installation is
#: the next revision of its backend.
FIXTURE_PACKAGES = {
    "UNISERVE_REBIND_PACKAGE": "crates/ops/package-fixtures/rms_norm_rebind",
    "UNISERVE_RECAPTURE_PACKAGE": "crates/ops/package-fixtures/recapture",
    "UNISERVE_POLICY_PACKAGE": POLICY_PACKAGE,
    "UNISERVE_WRONG_PACKAGE": "crates/ops/package-fixtures/rms_norm_wrong",
}


def _package(name: str) -> Path:
    """The package directory `name` names, else the tree's fixture."""
    value = os.environ.get(name)
    path = Path(value) if value else repository() / FIXTURE_PACKAGES[name]
    if not path.is_dir():
        pytest.fail(f"{name}: {path} is not a package directory")
    return path


@pytest.fixture(scope="module")
def model() -> Path:
    return host_asset("UNISERVE_QWEN3_MODEL", QWEN3_CHECKPOINT)


@pytest.fixture(scope="module")
def build_server(tmp_path_factory: pytest.TempPathFactory):
    """The host's Build Server, with every package installed.

    The deployment's startup builds the kernels its recorded walk names, and
    the scheduling policy is installed beside them; the shared helper does
    both, so this file and the others serve one store the same way.
    """
    with running_build_server(tmp_path_factory) as socket:
        yield {"socket": socket, "binary": require_uniserve_binary()}


@pytest.fixture(scope="module")
def server(
    model: Path, build_server, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[str]:
    """One deployment of `model` beside this host's Build Server.

    Module-scoped, like the epoch history the tests read back.

    Every test here requests `server`, and until this fixture existed none of
    them could start: pytest fails the setup of a test whose fixture is
    undefined and reports it as an error rather than a failure, so a run of the
    file read "no failures" with no test executed.

    The deployment is module-scoped because the tests are one epoch history —
    a rebind, a recapture with its policy, a rollback and a refusal — each read
    back through `/admin/status` and `/metrics` against the same deployment's
    records. The options are `DEPLOYMENT_OPTIONS`, the live-update file's
    deliberately, and the host tier is `HOST_GIB`, the figure that file
    serves the same model with.
    """
    directory = tmp_path_factory.mktemp("deployment")
    log = directory / "deployment.log"
    command = [
        str(require_uniserve_binary()),
        "serve",
        "--model-dir",
        str(model),
        "--served-model-name",
        "qwen3-moe",
        "--host",
        "127.0.0.1",
        *engine_selection(),
        "--set",
        f'build_config.socket="{build_server["socket"]}"',
        "--set",
        f"kv_cache_config.host_gib={HOST_GIB}",
    ]
    for option in DEPLOYMENT_OPTIONS:
        command += ["--set", option]
    with server_process(command, log, timeout_s=1800) as server:
        base_url = server.base_url
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
IDLE = "no running request ran a step"


class Resident:
    """One streamed completion, held running for a whole update window.

    The shadow pairs a changed operator's kernel on one step of the batches a
    scheduler forms from its **running** requests. A closed loop of short
    requests has moments with none in flight, and a preparation that lands in
    one of them compares nothing and fails by name ("no running request ran a
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
                "the running request was refused: "
                f"{self._response.status_code}: {body!r}"
            )
        self._lines = self._response.iter_lines()
        # The first line proves the engine formed a batch for this request: it
        # is running, not waiting behind admission.
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
                f"the running request produced no stream: {first!r}"
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


def install(build_server, package: Path) -> int:
    """Install `package` on the Build Server and answer the new revision."""
    done = subprocess.run(
        [
            str(build_server["binary"]),
            "build-server",
            "install",
            str(package),
            "--socket",
            str(build_server["socket"]),
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


def next_event(base_url: str) -> int:
    """The sequence number the daemon's next event takes."""
    return admin_status(base_url)["next_event"]


def end_update(base_url: str, action: str) -> None:
    """Commit or roll back the update on trial."""
    response = admin(base_url, f"config/{action}")
    assert response.status_code == 200, (
        f"POST /admin/config/{action} answered {response.status_code}: "
        f"{response.text[:500]}"
    )


def _phase_of(event: dict) -> str:
    """The phase of the update a `deployment` event states."""
    if event.get("kind") != "deployment":
        return ""
    phase = (event.get("data") or {}).get("phase")
    return phase if isinstance(phase, str) else ""


def wait_for_phases(base_url: str, phases: list[str], after: int) -> list[dict]:
    """Long-poll the daemon's events after sequence number `after` until
    every phase arrived.
    """  # noqa: D205
    seen: list[dict] = []
    pending = list(phases)
    deadline = time.monotonic() + STEP_TIMEOUT_S
    next_seq = after
    while pending:
        if time.monotonic() > deadline:
            raise TimeoutError(f"phases {pending} did not arrive; saw {seen}")
        response = admin_request(
            base_url,
            "GET",
            "/admin/events",
            params={"after": next_seq, "wait": 30},
            timeout_s=60,
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
    """The epoch the deployment serves, as the daemon records it.

    `/admin/config` reports every epoch, ascending; the one that serves is the
    highest-numbered epoch in `serving` or `trial`, and `/admin/events` names
    it by number (`data.epoch`).
    """
    epochs = admin_config(base_url)["epochs"]
    live = [
        epoch for epoch in epochs if epoch.get("status") in ("serving", "trial")
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


def test_a_kernel_package_is_swapped_by_rebind_while_requests_stream(
    server: str, build_server, traffic: Traffic
) -> None:
    package = _package("UNISERVE_REBIND_PACKAGE")
    before = epoch_record(server)
    revision = install(build_server, package)

    after_event = next_event(server)
    assert apply_config(server).status_code == 200
    wait_for_phases(server, TO_TRIAL, after_event)
    after = epoch_record(server)
    assert after["kernels"] != before["kernels"], (
        "the generation update did not change the recorded kernel packages"
    )
    revisions = {
        entry["package"]["revision"]
        for entry in after["kernels"]
        if isinstance(entry.get("package"), dict)
    }
    assert revision in revisions, (
        f"the recorded modules name no revision {revision}: {after['kernels']}"
    )

    after_event = next_event(server)
    end_update(server, "commit")
    wait_for_phases(server, COMMITTED, after_event)
    assert_no_failures("the rebind swap", traffic)
    mapped_within_budget(server)


def test_the_recapture_swap_and_the_policy_swap_keep_serving(
    server: str, build_server, traffic: Traffic
) -> None:
    package = _package("UNISERVE_RECAPTURE_PACKAGE")
    policy = _package("UNISERVE_POLICY_PACKAGE")
    before = epoch_record(server)
    install(build_server, package)
    install(build_server, policy)

    # The shadow compares on a step of a running request; without one the
    # preparation refuses by name and the swap below would be asserted on an
    # update that compared nothing.
    resident = Resident(server, PROMPTS[0])
    try:
        after_event = next_event(server)
        assert apply_config(server).status_code == 200
        wait_for_phases(server, TO_TRIAL, after_event)
    finally:
        # Closed after the update answered and before any failure is read: a
        # request in flight here would race `assert_no_failures` below.
        resident.close()
    after = epoch_record(server)
    assert after["kernels"] != before["kernels"], (
        "the recapture swap did not change the recorded kernel packages"
    )
    assert after["policy"] != before["policy"], (
        "the policy swap did not change the recorded policy"
    )
    assert_no_failures("the recapture and policy swap", traffic)
    mapped_within_budget(server)

    # The update stays on trial for the rollback below.
    assert after["status"] == "trial", f"expected trial, got {after['status']}"


def test_a_rollback_restores_the_previous_generation(
    server: str, traffic: Traffic
) -> None:
    before = epoch_record(server)
    after_event = next_event(server)
    end_update(server, "rollback")
    wait_for_phases(server, ROLLED_BACK, after_event)
    assert before["previous_kernels"] is not None, (
        "the epoch on trial kept no previous kernel packages to roll back to"
    )
    after = epoch_record(server)
    assert after["kernels"] == before["previous_kernels"], (
        "the rollback did not restore the previous kernel packages"
    )
    assert after["policy"] == before["previous_policy"], (
        "the rollback did not restore the previous policy"
    )
    assert_no_failures("the rollback", traffic)
    mapped_within_budget(server)


def test_a_wrong_kernel_is_refused_by_shadow_execution(
    server: str, build_server, traffic: Traffic
) -> None:
    package = _package("UNISERVE_WRONG_PACKAGE")
    before = epoch_record(server)
    install(build_server, package)

    # The comparison needs a request the scheduler is stepping when the
    # preparation shadows: a wrong kernel can only be refused by a comparison
    # that ran.
    resident = Resident(server, PROMPTS[1])
    try:
        # The update is abandoned while it prepares, which the answer states
        # with the update as it ended (`409`).
        response = apply_config(server)
    finally:
        resident.close()
    assert response.status_code == 409, response.text
    update = response.json()["result"]["update"]
    assert update["phase"] == "abandoned", update
    assert update["failure"]["kind"] == "prepare_failed", update
    refusal = update["failure"]["reason"]

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
        f"because no running request ran a step: {refusal}"
    )

    after = epoch_record(server)
    assert after["kernels"] == before["kernels"], (
        "a refused update changed the serving generation"
    )
    assert after["status"] in ("serving", "trial"), after["status"]
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
