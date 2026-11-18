"""A prefill group transfers each request to a decode group.

Qwen3 is served by a deployment of two worker groups: a ``prefill`` group
that computes each prompt and samples its first token, and a ``decode``
group that continues the request from that token. One scheduler drives both
as separate groups of its session with the host's buffer pool: once the
prefill group's call that sampled the first token is output, the scheduler
submits a ``Transfer`` of the request's owner to the decode group, stamped by
the prefill group's executor; a task of the Transfer Server copies the
request's blocks that are not valid in the decode group's memory, device to
device, the owner gets a slot there, and the request decodes on the decode
group (``specs/buffer-pool.md`` §11, ``specs/scheduler-design.md`` §4.1). Each
case compares the greedy tokens of fixed prompts, one of several blocks with
a partial last one, with an undivided deployment's on one GPU: a prefill
group computes no decode, so the tokens after the first are the decode
group's, computed over the blocks it was transferred. Each case also reads
the scheduler's transfer counters in the observation store
(``/admin/query``, ``specs/scheduler-design.md`` §7.5) once a report counts
a transfer for every request it answered and states nothing running or
waiting: no transfer failed.

The one-host case places both groups on the module's two GPUs. The two-host
case places the decode group on the first GPU of the run's second host
(``--peer-host``, ``uniserve_testing.plugin``): the package lists this host,
the leader, and the second host by their host names
(:func:`uniserve_testing.deployments.two_hosts`), and the second host's
daemon serves the same package there and joins the leader
(:meth:`uniserve_testing.deployments.Deployments.follower`), at a Build
Server there that installs the tree's packages from this repository's path,
which the second host must mount at the same path. Its Transfer Server reads
the prefill rank's planes through the NIXL package each host has installed
(``docs/nixl.md``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from uniserve_testing import admin, chat, client
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.deployments import Deployments, two_hosts
from uniserve_testing.packages import merged
from uniserve_testing.remote import RemoteHost
from uniserve_testing.servers import Server

# The module's cases share two GPUs: the undivided reference serves on the
# first, and the one-host prefill/decode deployment on both.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("qwen3"),
    pytest.mark.gpus(2),
]

SERVED_MODEL = "qwen3-prefill-decode"
MAX_TOKENS = 48
PROMPTS = (
    "Name the planets of the solar system in order from the sun.",
    "Write a haiku about a lighthouse in winter.",
    "Explain in two sentences why the sky is blue.",
    # A prompt of several blocks, whose last block is partial.
    chat.long_prompt(40),
)

# Every deployment of the module serves these options: the divided
# deployments capture the graphs the undivided one does, so that each group
# computes its part of a request as the undivided deployment computes it.
OPTIONS: dict[str, Any] = {
    "model": CHECKPOINTS["qwen3"].model,
    "served_model_name": SERVED_MODEL,
    "cuda_graph_config": {
        "prefill_token_sizes": [1024],
        "decode_batch_sizes": [1, 2, 4, 8],
    },
}

# The scheduler's transfer counters (``specs/scheduler-design.md`` §7.5): the
# transfers answered ``Transferred`` and those answered ``Failed``.
TRANSFERS = "uniserve_request_transfers_total"
FAILED_TRANSFERS = "uniserve_request_transfer_failures_total"

# Each prompt's greedy tokens, each by its token id with no candidate ranked
# (`uniserve_testing.chat.ranked_tokens`).
Decoded = list[list[tuple[str, set[str]]]]

# How long the scheduler's report may take to count the transfers of the
# requests a case answered, in seconds: thirty report periods
# (``worker_config.report_period``, a second).
REPORTED_S = 30


def _split(decode_host: str | None) -> dict[str, Any]:
    """The layout of a prefill group on this host's first GPU and a decode
    group on the next one, or on `decode_host`'s first.
    """  # noqa: D205
    decode = "cuda:1" if decode_host is None else f"{decode_host}/cuda:0"
    return client.layout(
        [
            {
                "id": "prefill",
                "role": "prefill",
                "ranks": ["cuda:0"],
                "queue_depth": 2,
                "blocks": {"model": {}},
            },
            {
                "id": "decode",
                "role": "decode",
                "ranks": [decode],
                "queue_depth": 2,
                "blocks": {"model": {}},
            },
        ],
    )


def _decoded(base_url: str) -> Decoded:
    """The tokens each prompt decodes to greedily, each by its token id."""
    return [
        chat.ranked_tokens(
            chat.complete(
                base_url,
                chat.body(
                    SERVED_MODEL,
                    prompt,
                    MAX_TOKENS,
                    **chat.GREEDY,
                    **chat.NO_THINKING,
                    logprobs=True,
                    return_tokens_as_token_ids=True,
                ),
            )
        )
        for prompt in PROMPTS
    ]


def _failed_transfers(server: Server, requests: int) -> float:
    """The transfers the scheduler answered as failed, once a report counts
    at least `requests` transfers, as many as the requests the case
    answered, and states nothing running or waiting.

    A failed transfer ends its request as migratable, and the request is
    computed and transferred again, so its failure is counted before its
    answer arrives. The scheduler reports its statistics once a report
    period, and no event announces a report, so the case reads the
    observation store until one states the transfers.
    """  # noqa: D205

    def reported() -> bool:
        transferred = admin.counter(server, TRANSFERS) + admin.counter(
            server, FAILED_TRANSFERS
        )
        busy = sum(
            sample.value
            for name in (
                "uniserve_scheduler_running_requests",
                "uniserve_scheduler_waiting_requests",
            )
            for sample in admin.gauge(server, name)
        )
        return transferred >= requests and busy == 0

    server.wait_until(
        reported, REPORTED_S, "the requests' transfers were not reported"
    )
    return admin.counter(server, FAILED_TRANSFERS)


@pytest.fixture(scope="module")
def reference(module_deployments: Deployments) -> Decoded:
    """Each prompt's greedy tokens on an undivided deployment of one GPU."""
    with module_deployments.serve(OPTIONS, name="undivided") as server:
        return _decoded(server.base_url)


def test_a_prefill_group_transfers_requests_to_a_decode_group_on_one_host(
    deployments: Deployments,
    reference: Decoded,
):
    with deployments.serve(
        merged(OPTIONS, _split(None)),
        calibration=deployments.calibration(),
    ) as server:
        decoded = _decoded(server.base_url)
        failed = _failed_transfers(server, len(PROMPTS))

    assert decoded == reference
    assert failed == 0


def test_a_prefill_group_transfers_requests_to_a_decode_group_on_another_host(
    repository: Path,
    deployments: Deployments,
    peer_host: RemoteHost,
    reference: Decoded,
):
    # The decode group's Transfer Server reads the prefill rank's cache on
    # this host through NIXL: both hosts' calibrations time those READs.
    calibration = deployments.calibration(reads=True)
    with (
        deployments.serve(
            merged(OPTIONS, _split(peer_host.name), two_hosts(peer_host)),
            serving=False,
            calibration=calibration,
        ) as server,
        deployments.follower(
            peer_host,
            server,
            repository,
            calibration=calibration,
            reads=True,
        ),
    ):
        decoded = _decoded(server.base_url)
        failed = _failed_transfers(server, len(PROMPTS))

    assert decoded == reference
    assert failed == 0
