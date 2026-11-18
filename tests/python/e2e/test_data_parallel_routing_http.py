"""Requests are served by the data-parallel replica the policy routes them to.

Qwen3 is served by two data-parallel schedulers whose ranks share their
experts (``data_parallel_size = 2`` with expert parallelism): one main
process, one scheduler per replica, each on its own GPU. The control thread
asks the epoch's scheduling policy where each request goes (its ``route``),
and the policy the deployment serves names the live replica with the fewest
requests in flight, rotating among equally loaded ones
(``docs/architecture.md``, The request interface). Each replica's scheduler
reports its admissions under its own ``scheduler`` label
(``uniserve:scheduler_admitted``).

Requests sent one at a time find both replicas idle, so the policy alternates
between them and each admits half. While one replica holds a long request,
every request sent meanwhile finds the other replica with fewer in flight, so
the other replica admits all of them.

An answer does not state the replica that served it, and the schedulers'
statistics reach ``/metrics`` in a report once a report period
(``worker_config.report_period``, a second) that no event announces and no
sequence number orders, so the case reads each replica's admissions from the
reports: before a phase once they held still over several reports with
nothing running, and after it once they count the phase's requests.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from uniserve_testing import chat, client
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.deployments import Deployments
from uniserve_testing.servers import Server

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("qwen3"),
    pytest.mark.gpus(2),
]

SERVED_MODEL = "qwen3-data-parallel"
# The schedulers of the deployment's two replicas, by their label.
SCHEDULERS = (0, 1)
# Requests sent one at a time; an even count splits evenly.
SEQUENTIAL = 6
# Requests sent while a long one runs on one replica.
BESIDE = 4
ADMITTED = "uniserve:scheduler_admitted"
RUNNING = "uniserve:scheduler_running"
# The successive reports, a report period apart, that must state the same
# admissions with nothing running before a phase starts: the serving path's
# own requests at start are counted by then.
QUIET_READS = 3
# How long the replicas may take to hold still before a phase, in seconds:
# the serving path's requests at start end within it.
QUIET_S = 120
# How long the reports may take to state what a phase did, in seconds: sixty
# report periods.
REPORTED_S = 60


def _per_scheduler(base_url: str, name: str) -> dict[int, float]:
    """The metric `name` of each replica's scheduler."""
    return {
        scheduler: client.metric(base_url, name, scheduler=str(scheduler))
        for scheduler in SCHEDULERS
    }


def _quiet(server: Server) -> dict[int, float]:
    """Each replica's admissions once :data:`QUIET_READS` successive reads,
    a report period apart, state the same counts and nothing running.
    """  # noqa: D205
    last: dict[int, float] | None = None
    held = 0

    def still() -> bool:
        nonlocal last, held
        now = _per_scheduler(server.base_url, ADMITTED)
        running = any(_per_scheduler(server.base_url, RUNNING).values())
        held = held + 1 if now == last and not running else 0
        last = now
        return held >= QUIET_READS

    server.wait_until(still, QUIET_S, "the replicas did not settle")
    assert last is not None
    return last


def _admitted_since(
    server: Server, before: dict[int, float], total: int
) -> dict[int, float]:
    """The admissions each replica added since `before`, once the reports
    count `total` of them.
    """  # noqa: D205
    grown: dict[int, float] = {}

    def counted() -> bool:
        now = _per_scheduler(server.base_url, ADMITTED)
        grown.update(
            {scheduler: now[scheduler] - before[scheduler] for scheduler in now}
        )
        return sum(grown.values()) >= total

    server.wait_until(
        counted, REPORTED_S, f"the replicas did not report {total} admissions"
    )
    return grown


def _ask(base_url: str, prompt: str, max_tokens: int) -> dict:
    """The answer to `prompt`, `max_tokens` tokens long."""
    return chat.complete(
        base_url,
        chat.body(
            SERVED_MODEL,
            prompt,
            max_tokens,
            **chat.NO_THINKING,
            temperature=0,
            ignore_eos=True,
        ),
    )


def test_requests_are_served_by_the_replicas_the_policy_routes_them_to(
    deployments: Deployments,
):
    with deployments.serve(
        {
            "model": CHECKPOINTS["qwen3"].model,
            "served_model_name": SERVED_MODEL,
            "data_parallel_size": 2,
            "moe_config": {"expert_parallel": True},
        }
    ) as server:
        base_url = server.base_url
        # Idle replicas: the policy rotates, and each admits half.
        before = _quiet(server)
        for index in range(SEQUENTIAL):
            _ask(base_url, f"Say the number {index}.", 4)
        alternated = _admitted_since(server, before, SEQUENTIAL)
        assert alternated == {0: SEQUENTIAL / 2, 1: SEQUENTIAL / 2}, alternated

        # One replica holds a long request: every request beside it is routed
        # to the other one.
        before = _quiet(server)
        with ThreadPoolExecutor(max_workers=1) as pool:
            long = pool.submit(_ask, base_url, "Count from 1 to 3000.", 3000)
            server.wait_until(
                lambda: any(_per_scheduler(base_url, RUNNING).values()),
                REPORTED_S,
                "the long request did not start",
            )
            running = _per_scheduler(base_url, RUNNING)
            held = next(
                scheduler for scheduler, count in running.items() if count > 0
            )
            for index in range(BESIDE):
                _ask(base_url, f"Say the letter {'abcd'[index]}.", 4)
            long.result()
        beside = _admitted_since(server, before, BESIDE + 1)
        assert beside == {held: 1, 1 - held: BESIDE}, (held, beside)
