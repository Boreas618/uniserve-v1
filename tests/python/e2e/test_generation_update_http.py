"""A generation update while a deployment serves (``specs/updates.md`` §3.1).

One Qwen3-MoE deployment serves on one GPU, as the deployment both update
modules serve (:data:`uniserve_testing.updates.PROFILE`), while the cases
replace kernel packages and the scheduling policy under it:

- one kernel package swapped by **rebind**, one by **recapture** together with
  the **policy**, then **rolled back**, with no request failing at any step;
- a deliberately **wrong kernel** refused by the shadow while the update
  prepares: the update is abandoned, the serving generation does not change
  and requests keep succeeding;
- the allocator's mapped bytes stay within the device's budget while both
  generations' graphs exist.

Each case installs a revision at the run's Build Server and applies the
unchanged deployment configuration, which is what moves a newly installed
revision into a generation update (``docs/admin.md``, Updates). Phases are
asserted on the daemon's events, never on sleeps. The revisions are derived
from the serving packages by the mutations ``tests/python/e2e/packages``
declares (:func:`uniserve_testing.updates.derived`):

- ``rms_norm`` ``rebind``: a package of an operator the slice launches every
  step, with the serving descriptor key and the same launch structure (a
  rebind swap);
- ``gemm`` ``recapture``: a package of one operator whose single entry
  launches two kernels (a native module; the recapture case of 4.5);
- the serving scheduling policy (wasm) installed again, which makes its next
  revision;
- ``rms_norm`` ``wrong``: a package of the rebind operator whose result is
  outside ``Op::tolerance``.

The Build Server is the run's, so the module installs the serving packages
it replaced again when it ends: their latest revisions are then the tree's
kernels, which every later deployment of the run serves.

The admin surface is the operator socket's: `POST /admin/config/commit` and
`/admin/config/rollback`, events whose `kind` is `update` and whose `data` is
the update as it stood, its `phase` among them, and `/admin/config`'s `epochs`
array whose elements carry `kernels` and `previous_kernels` as a list of
`{op, phase, package: {backend, revision, arguments}}`.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from uniserve_testing import admin, builds, chat, updates
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.builds import BuildServer
from uniserve_testing.client import metric
from uniserve_testing.deployments import Deployments, Settings
from uniserve_testing.servers import Server
from uniserve_testing.updates import Stream

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("qwen3"),
    # The module's cases are one deployment's history.
    pytest.mark.gpus(1),
]

#: The deployment's served model name.
NAME = "qwen3-moe"

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

#: The tokens the running request of a preparation's shadow asks for: more
#: than an update takes to prepare at the deployment's steps.
RESIDENT_TOKENS = 512

# The two sentences a wrong-kernel refusal can carry. The comparison's own
# sentence names the element that left tolerance and the bound it left; the
# idle sentence says no request ran a step, which means nothing was compared.
# A guard that accepts both cannot tell a refused kernel from an update that
# never ran one.
COMPARED = "differ beyond tolerance"
IDLE = "no running request ran a step"


@pytest.fixture(scope="module")
def server(module_deployments: Deployments) -> Iterator[Server]:
    """The module's deployment, beside the run's Build Server.

    Module-scoped because the cases are one epoch history — a rebind, a
    recapture with its policy, a rollback and a refusal — each read back
    through ``/admin/config`` and ``/metrics`` against the same deployment's
    records.
    """
    with module_deployments.serve(
        "--served-model-name",
        NAME,
        *updates.assigned(*updates.PROFILE),
        model=CHECKPOINTS["qwen3"].model,
        name="deployment",
    ) as served:
        yield served


@pytest.fixture(scope="module")
def build_server(
    uniserve: Path,
    repository: Path,
    e2e_settings: Settings,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[BuildServer]:
    """The module's own Build Server, with the tree's packages installed.

    A revision the module installs is what every later epoch selects, so the
    module installs its revisions at a server of its own, which no other
    module's deployment resolves at.
    """
    with builds.build_server(
        uniserve,
        repository,
        tmp_path_factory.mktemp("build-server"),
        scale=e2e_settings.time_scale,
    ) as server:
        yield server


def _resident(base_url: str, prompt: str) -> Stream:
    """One streamed completion, held running for a whole update window,
    once its first frame arrived.

    The shadow pairs a changed operator's kernel on one step of the batches a
    scheduler forms from its **running** requests. A closed loop of short
    requests has moments with none in flight, and a preparation that lands in
    one of them compares nothing and fails by name ("no running request ran a
    step"). Holding one request open for as long as the caller keeps it makes
    the wrong-kernel assertion mean what it says. Its first frame proves the
    engine formed a batch for it: it is running, not waiting behind
    admission.

    The caller closes it after the update's answer has arrived and before any
    failure is read, so a request in flight never races the assertions.
    """  # noqa: D205
    stream = Stream(
        base_url, chat.body(NAME, prompt, RESIDENT_TOKENS, stream=True)
    )
    stream.wait_for(
        lambda streamed: streamed.frames >= 1,
        "the running request produced no stream",
    )
    return stream


class Traffic:
    """A closed loop of requests that records every failure as it happens."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.failures: list[str] = []
        self.completed = 0
        # The failures and completions change under this condition, which
        # wakes a caller waiting for the loop's progress.
        self._progress = threading.Condition()
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(target=self._loop, args=(prompt,), daemon=True)
            for prompt in PROMPTS[:CONCURRENCY]
        ]
        for thread in self._threads:
            thread.start()

    def served(self, count: int) -> None:
        """Wait until the loop completed `count` requests, or one failed."""
        with self._progress:
            self._progress.wait_for(
                lambda: self.completed >= count or bool(self.failures)
            )

    def _fail(self, failure: str) -> None:
        with self._progress:
            self.failures.append(failure)
            self._progress.notify_all()

    def _one(self, prompt: str) -> None:
        try:
            answer = chat.complete(
                self.base_url, chat.body(NAME, prompt, MAX_TOKENS)
            )
        except httpx.HTTPStatusError as error:
            self._fail(
                f"{error.response.status_code}: {error.response.text[:300]}"
            )
            return
        choice = answer["choices"][0]
        if not choice.get("message", {}).get("content"):
            self._fail(f"empty content: {json.dumps(answer)[:300]}")
            return
        if choice.get("finish_reason") in (None, "migratable"):
            self._fail(f"no finish reason: {json.dumps(choice)[:300]}")
            return
        with self._progress:
            self.completed += 1
            self._progress.notify_all()

    def _loop(self, prompt: str) -> None:
        while not self._stop.is_set():
            try:
                self._one(prompt)
            except Exception as error:  # noqa: BLE001 - a request that cannot be
                # made is a failure of the update under test, not of the test.
                self._fail(f"{type(error).__name__}: {error}")

    def stop(self) -> None:
        """End the loop once each request in flight answered."""
        self._stop.set()
        for thread in self._threads:
            thread.join()


@pytest.fixture(scope="module")
def traffic(server: Server) -> Iterator[Traffic]:
    loop = Traffic(server.base_url)
    try:
        loop.served(CONCURRENCY)
        assert not loop.failures, f"the deployment failed: {loop.failures}"
        yield loop
    finally:
        loop.stop()


def _to_trial(server: Server) -> None:
    """Apply the unchanged configuration, which moves the newly installed
    revisions into a generation update, and see it reach its trial.
    """  # noqa: D205
    before = admin.next_event(server)
    applied = admin.apply(server)
    assert applied.status_code == 200, applied.text
    update = admin.wait_for_phase(
        server, "trial", "abandoned", "rolled_back", after=before
    )
    assert update["phase"] == "trial", update


def _end_update(server: Server, action: str, phase: str) -> None:
    """Commit or roll back the update on trial (`action`), and see it enter
    `phase`.
    """  # noqa: D205
    before = admin.next_event(server)
    response = admin.operation(server, f"config/{action}")
    assert response.status_code == 200, (
        f"POST /admin/config/{action} answered {response.status_code}: "
        f"{response.text[:500]}"
    )
    admin.wait_for_phase(server, phase, after=before)


def epoch_record(server: Server) -> dict:
    """The epoch the deployment serves, as the daemon records it.

    `/admin/config` reports every epoch, ascending; the one that serves is the
    highest-numbered epoch in `serving` or `trial`, and `/admin/events` names
    it by number (`data.epoch`).
    """
    epochs = admin.config(server)["epochs"]
    live = [
        epoch for epoch in epochs if epoch.get("status") in ("serving", "trial")
    ]
    assert live, f"the deployment serves no epoch: {epochs}"
    return max(live, key=lambda epoch: epoch["epoch"])


def mapped_within_budget(server: Server) -> None:
    """The allocator's mapped bytes stay within the device's budget.

    The deployment serves one device, so the bytes every pool mapped there
    and the device's budget are the sums of their metrics.
    """
    budget = metric(server.base_url, "uniserve:memory_budget")
    mapped = metric(server.base_url, "uniserve:memory_mapped")
    assert budget > 0, "the deployment published no memory budget"
    assert mapped <= budget, f"mapped {mapped} exceeds the budget {budget}"


def assert_no_failures(step: str, traffic: Traffic) -> None:
    assert not traffic.failures, (
        f"requests failed during {step}: {traffic.failures}"
    )


def test_a_kernel_package_is_swapped_by_rebind_while_requests_stream(
    server: Server,
    build_server: BuildServer,
    repository: Path,
    traffic: Traffic,
    tmp_path: Path,
) -> None:
    package = updates.derived(repository, "rms_norm", "rebind", tmp_path)
    before = epoch_record(server)
    revision = build_server.install(package).revision

    _to_trial(server)
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

    _end_update(server, "commit", "committed")
    assert_no_failures("the rebind swap", traffic)
    mapped_within_budget(server)


def test_the_recapture_swap_and_the_policy_swap_keep_serving(
    server: Server,
    build_server: BuildServer,
    repository: Path,
    traffic: Traffic,
    tmp_path: Path,
) -> None:
    package = updates.derived(repository, "gemm", "recapture", tmp_path)
    before = epoch_record(server)
    build_server.install(package)
    # The serving policy installed again: its next revision, the same code.
    build_server.install(repository / builds.POLICY_PACKAGE)

    # The shadow compares on a step of a running request; without one the
    # preparation refuses by name and the swap below would be asserted on an
    # update that compared nothing.
    resident = _resident(server.base_url, PROMPTS[0])
    try:
        _to_trial(server)
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
    server: Server, traffic: Traffic
) -> None:
    before = epoch_record(server)
    _end_update(server, "rollback", "rolled_back")
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
    server: Server,
    build_server: BuildServer,
    repository: Path,
    traffic: Traffic,
    tmp_path: Path,
) -> None:
    package = updates.derived(repository, "rms_norm", "wrong", tmp_path)
    before = epoch_record(server)
    build_server.install(package)

    # The comparison needs a request the scheduler is stepping when the
    # preparation shadows: a wrong kernel can only be refused by a comparison
    # that ran.
    resident = _resident(server.base_url, PROMPTS[1])
    try:
        # The update is abandoned while it prepares, which the answer states
        # with the update as it ended (`409`).
        response = admin.apply(server)
    finally:
        resident.close()
    assert response.status_code == 409, response.text
    update = response.json()["result"]["update"]
    assert update["phase"] == "abandoned", update
    assert update["failure"]["kind"] == "prepare_failed", update
    refusal = update["failure"]["reason"]

    # The comparison's own sentence, not a refusal for any reason at all: the
    # element past tolerance and the bound it left, and never the sentence that
    # says nothing was compared. The refusal is printed whole: its first clause
    # names the step that failed and the rest carries the engine's own
    # sentence, so a slice can hide the very string these assertions are about.
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
    server: Server, traffic: Traffic
) -> None:
    mapped_within_budget(server)
    assert traffic.completed > 0
    assert not traffic.failures, (
        f"requests failed during the update: {traffic.failures}"
    )
