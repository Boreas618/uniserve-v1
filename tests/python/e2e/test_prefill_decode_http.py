"""A prefill replica hands each request off to a decode replica.

Qwen3 is served by a deployment of two replicas, two schedulers of one GPU
each: replica ``0`` of the ``prefill`` role computes each prompt and samples
its first token, and replica ``1`` of the ``decode`` role continues the
request from that token (``replicas.<id>.role``, ``docs/configuration.md``
under Replicas). Once the prefill replica's call that sampled the first
token is output, its buffer pool lends the request's pages and the request
migrates with the lend; the decode replica's buffer pool imports every lent
page into blocks of its own, by a task of the Transfer Server, and the
request decodes there. Each case compares the greedy tokens of fixed prompts,
one of several pages with a partial last one, with an undivided deployment's
on one GPU: a prefill replica generates the first token alone, so the tokens
after it are the decode replica's, computed over the pages it imported. Each
case also reads the decode replica's import counters in the observation store
(``/admin/query``) once a report counts an import for every request it
answered and states nothing running or waiting: no import failed.

The one-host case places both replicas on the module's two GPUs. The two-host
case places the decode replica on the first GPU of the run's second host
(``--peer-host``, ``uniserve_testing.plugin``): the package lists this host,
the leader, and the second host by their host names
(:func:`uniserve_testing.deployments.two_hosts`), and the second host's
daemon serves the same package there and joins the leader
(:meth:`uniserve_testing.deployments.Deployments.follower`), at a Build
Server there that installs the tree's packages from this repository's path,
which the second host must mount at the same path. Its Transfer Server reads
the prefill replica's pages through the NIXL package each host has installed
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
# first, and the one-host deployment of a prefill and a decode replica on
# both.
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
# deployments capture the graphs the undivided one does, so that each replica
# computes its part of a request as the undivided deployment computes it.
OPTIONS: dict[str, Any] = {
    "model": CHECKPOINTS["qwen3"].model,
    "served_model_name": SERVED_MODEL,
    "cuda_graph_config": {
        "prefill_token_sizes": [1024],
        "decode_batch_sizes": [1, 2, 4, 8],
    },
}

# The decode replica's import counters: the imports answered ``Imported`` and
# those answered ``Failed``.
IMPORTS = "uniserve_request_imports_total"
FAILED_IMPORTS = "uniserve_request_import_failures_total"

# Each prompt's greedy tokens, each by its token id with no candidate ranked
# (`uniserve_testing.chat.ranked_tokens`).
Decoded = list[list[tuple[str, set[str]]]]

# How long the scheduler's report may take to count the imports of the
# requests a case answered, in seconds: thirty report periods
# (``worker_config.report_period``, a second).
REPORTED_S = 30


def _split(decode_host: str | None) -> dict[str, Any]:
    """Two replicas of one GPU each: a prefill replica on this host's first
    GPU and a decode replica on the next one, or on `decode_host`'s first.
    """  # noqa: D205
    decode = "cuda:1" if decode_host is None else f"{decode_host}/cuda:0"
    layout = client.layout(
        [
            {
                "id": "model-0",
                "ranks": ["cuda:0"],
                "queue_depth": 2,
                "blocks": {"model": {}},
            },
            {
                "id": "model-1",
                "ranks": [decode],
                "queue_depth": 2,
                "blocks": {"model": {}},
            },
        ],
    )
    return merged(
        layout,
        {
            "data_parallel_size": 2,
            "replicas": {"0": {"role": "prefill"}, "1": {"role": "decode"}},
        },
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


def _failed_imports(server: Server, requests: int) -> float:
    """The imports the decode replica answered as failed, once a report counts
    at least `requests` imports, as many as the requests the case answered,
    and states nothing running or waiting.

    A failed import leaves its request to compute its prompt on the decode
    replica, so its failure is counted before its answer arrives. The
    scheduler reports its statistics once a report period, and no event
    announces a report, so the case reads the observation store until one
    states the imports.
    """  # noqa: D205

    def reported() -> bool:
        imported = admin.counter(server, IMPORTS) + admin.counter(
            server, FAILED_IMPORTS
        )
        busy = sum(
            sample.value
            for name in (
                "uniserve_scheduler_running_requests",
                "uniserve_scheduler_waiting_requests",
            )
            for sample in admin.gauge(server, name)
        )
        return imported >= requests and busy == 0

    server.wait_until(
        reported, REPORTED_S, "the requests' imports were not reported"
    )
    return admin.counter(server, FAILED_IMPORTS)


@pytest.fixture(scope="module")
def reference(module_deployments: Deployments) -> Decoded:
    """Each prompt's greedy tokens on an undivided deployment of one GPU."""
    with module_deployments.serve(OPTIONS, name="undivided") as server:
        return _decoded(server.base_url)


def test_a_prefill_replica_hands_requests_off_to_a_decode_replica_on_one_host(
    deployments: Deployments,
    reference: Decoded,
):
    with deployments.serve(
        merged(OPTIONS, _split(None)),
        calibration=deployments.calibration(),
    ) as server:
        decoded = _decoded(server.base_url)
        failed = _failed_imports(server, len(PROMPTS))

    assert decoded == reference
    assert failed == 0


def test_a_prefill_replica_hands_requests_off_to_a_decode_replica_across_hosts(
    repository: Path,
    deployments: Deployments,
    peer_host: RemoteHost,
    reference: Decoded,
):
    # The decode replica's Transfer Server reads the prefill replica's pages
    # on this host through NIXL: both hosts' calibrations time those READs.
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
        failed = _failed_imports(server, len(PROMPTS))

    assert decoded == reference
    assert failed == 0
