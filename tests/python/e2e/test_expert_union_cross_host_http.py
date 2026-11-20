"""An expert union whose members two hosts hold serves every scheduler.

Qwen3 is served by four data-parallel schedulers of one rank each whose ranks
share their experts in one expert union (``data_parallel_size = 4`` with
expert parallelism, ``docs/expert_parallelism.md``): ranks 0 and 1 on this
host, whose main process holds the four schedulers, and ranks 2 and 3 on a
second host, whose main process is a peer. The union's members meet across
the two processes: one formation at the epoch's rendezvous connects them for
each step's agreement and for the joins, and the exchange maps every rank's
window by its fabric handle over the hosts' NVLink domain.

Requests sent together are spread over the four schedulers, so every step of
one scheduler runs with members of both hosts, some joining it with no tokens
of their own. Each greedy count must come back as the count, and every
scheduler must have admitted requests. The same count asked twice of the idle
deployment, whose one stepping scheduler the three others join, must hold to
vLLM's ``check_logprobs_close`` rule
(:func:`uniserve_testing.chat.assert_logprobs_close`), since the two answers
may run on schedulers of different hosts.

The second host is the run's peer host (``--peer-host``,
:mod:`uniserve_testing.remote`); without one the case is skipped. Its GPUs
must share an NVLink domain with this host's. The case needs two free GPUs on
each host: the leased two here and the first two of the second host.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from uniserve_testing import admin, chat
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.deployments import Deployments, this_host, two_hosts
from uniserve_testing.packages import merged
from uniserve_testing.remote import RemoteHost

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("qwen3"),
    pytest.mark.gpus(2),
]

NAME = "qwen3-union-two-hosts"
# The schedulers, by their label: 0 and 1 drive this host's ranks, 2 and 3
# the second host's.
SCHEDULERS = (0, 1, 2, 3)
# Requests sent together: twice the schedulers, so each admits some.
TOGETHER = 8
# The count each request asks for, which its answer must begin with.
COUNT_TO = 40
ADMITTED = "uniserve_scheduler_admitted_requests_total"
# How long the reports may take to count every request, in seconds.
REPORTED_S = 60
# The ranked tokens a greedy answer states at each position.
TOP_LOGPROBS = 5


def _scheduler(host: str, device: int) -> dict:
    """The worker group of one scheduler's one rank, on `device` of
    `host`.
    """  # noqa: D205
    return {"ranks": [f"{host}/cuda:{device}"], "queue_depth": 2}


def _count(index: int) -> dict:
    """A greedy request for the count to :data:`COUNT_TO`, worded by
    `index` so requests sent together are distinct.
    """  # noqa: D205
    return chat.body(
        NAME,
        f"Request {index}: count from 1 to {COUNT_TO}, separated by commas "
        "and spaces. Write only the numbers.",
        256,
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
        **chat.GREEDY,
        **chat.NO_THINKING,
    )


def test_an_expert_union_over_two_hosts_serves_every_scheduler(
    deployments: Deployments, peer_host: RemoteHost, repository: Path
):
    expected = ", ".join(str(number) for number in range(1, COUNT_TO + 1))
    groups = [
        _scheduler(this_host(), 0),
        _scheduler(this_host(), 1),
        _scheduler(peer_host.name, 0),
        _scheduler(peer_host.name, 1),
    ]
    config = merged(
        two_hosts(peer_host),
        {
            "model": CHECKPOINTS["qwen3"].model,
            "served_model_name": NAME,
            "data_parallel_size": len(SCHEDULERS),
            "moe_config": {"expert_parallel": True},
            "worker_groups": [
                {"id": f"model-{index}", **group, "blocks": {"model": {}}}
                for index, group in enumerate(groups)
            ],
            # A small share of each GPU: a rank's quarter of the experts and the
            # dense weights take about 16 GB, beside a cache of 64K tokens.
            "mem_fraction_static": 0.18,
            "kv_cache_config": {"max_tokens": 65536},
            "chat_config": {"reasoning_parser": "none"},
            "cuda_graph_config": {
                "prefill_token_sizes": [512],
                "decode_batch_sizes": [1, 2, 4],
            },
        },
    )
    with (
        deployments.serve(config, serving=False) as server,
        deployments.follower(peer_host, server, repository, devices=(0, 1)),
    ):
        before = {
            scheduler: admin.counter(server, ADMITTED, scheduler=str(scheduler))
            for scheduler in SCHEDULERS
        }
        with ThreadPoolExecutor(max_workers=TOGETHER) as pool:
            answers = list(
                pool.map(
                    lambda index: chat.complete(
                        server.base_url,
                        _count(index),
                        server.scaled(chat.COMPLETION_S),
                    ),
                    range(TOGETHER),
                )
            )
        for index, answer in enumerate(answers):
            assert chat.content(answer).startswith(expected), (index, answer)

        admitted: dict[int, float] = {}

        def counted() -> bool:
            for scheduler in SCHEDULERS:
                now = admin.counter(server, ADMITTED, scheduler=str(scheduler))
                admitted[scheduler] = now - before[scheduler]
            spread = all(admitted[scheduler] > 0 for scheduler in SCHEDULERS)
            return spread and sum(admitted.values()) >= TOGETHER

        server.wait_until(
            counted,
            REPORTED_S,
            "the reports did not count the requests on every scheduler",
        )

        # Alone on an idle deployment, one scheduler steps and the three
        # others join it, two of them from the other host.
        first = chat.complete(
            server.base_url, _count(0), server.scaled(chat.COMPLETION_S)
        )
        again = chat.complete(
            server.base_url, _count(0), server.scaled(chat.COMPLETION_S)
        )
        assert chat.content(first).startswith(expected), first
        chat.assert_logprobs_close(
            chat.ranked_tokens(first),
            chat.ranked_tokens(again),
            "one count asked twice of the idle union",
        )
