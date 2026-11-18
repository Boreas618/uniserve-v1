"""A worker group over two hosts survives the loss of one host's rank.

Qwen3 is served by a tensor group of two ranks: rank 0 on this host, whose main
process holds the scheduler, and rank 1 on a second host, whose main process is
a peer that registers with the scheduler node over the peer channel and runs
every batch the scheduler sends it (`docs/architecture.md`, The channel). The
deployment keeps a standby (``worker_config.standby``): once it serves, the
daemon's failure-domain rules start a second main process on each host, one
standby epoch over the same two hosts, which parks.

While a long greedy count streams, the second host's serving main process is
killed. Its connection's close ends the scheduler node's process too, the
domain's standbys are promoted together, and the request continues on the
promoted epoch: it migrates after every token it generated, and the client
reads one uninterrupted stream, which matches the count asked again afterwards.

The second host is the run's peer host (``--peer-host``,
:mod:`uniserve_testing.remote`); without one the test is skipped. The
deployment's package lists both hosts by their host names, this host first
(:func:`uniserve_testing.deployments.two_hosts`), so the leader accepts the
second host's daemon at its join port, and the second host's daemon serves
the same package there and joins it (``Deployments.follower``). The case
needs one free GPU on each host: the leased one here and the first of the
second host, which runs no other main process of the user.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from uniserve_testing import admin, chat
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.deployments import Deployments, two_hosts
from uniserve_testing.packages import merged
from uniserve_testing.remote import RemoteHost

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

NAME = "qwen3-cross-host"
# Tokens the client reads before the second host's rank is killed.
EARLY_TOKENS = 64

#: The count, greedy and with each token's logprob, whose tokens a stream
#: and a later answer compare.
COUNT = chat.body(
    NAME,
    "Count from 1 to 5000, separated by commas and spaces. "
    "Write only the numbers.",
    16384,
    logprobs=True,
    return_tokens_as_token_ids=True,
    **chat.GREEDY,
    **chat.NO_THINKING,
)

#: The program of a main process, whose name the second host's process list
#: shows (``crates/engine/package.toml``).
ENGINE_PROGRAM = "uniserve-engine"

#: How long the standby epoch over both hosts takes to park, in seconds
#: (scaled): it starts as the deployment's engines did, but maps the weights
#: the serving epoch holds instead of loading them, and captures the same few
#: graphs.
STANDBY_S = 1800.0

#: How long the stream may wait for its next line, in seconds (scaled):
#: across the kill, until the promoted standbys serve the migrated request.
KILL_GAP_S = 1800.0


def _kill_serving_peer(host: RemoteHost) -> None:
    """Kill the second host's serving main process: the oldest main process
    of the user there, since the standby beside it started after it served.

    The leader's admin API lists its own node's processes alone, so the
    process is named by its program on the second host.
    """  # noqa: D205
    host.run("pkill", "-KILL", "--oldest", "--exact", ENGINE_PROGRAM)


def test_a_group_over_two_hosts_promotes_its_standby_when_a_hosts_rank_is_lost(
    deployments: Deployments, peer_host: RemoteHost, repository: Path
):
    config = merged(
        two_hosts(peer_host),
        {
            "model": CHECKPOINTS["qwen3"].model,
            "served_model_name": NAME,
            # Rank 0 on this host and rank 1 on the second: the shorthand
            # spreads the group's ranks over this host and the hosts after
            # it in equal blocks.
            "tensor_parallel_size": 2,
            "device_config": {"worker_hosts": [peer_host.name]},
            # Few captured graphs, so that the standby epoch parks soon, and
            # a small KV cache.
            "kv_cache_config": {"max_tokens": 32768},
            "chat_config": {"reasoning_parser": "none"},
            "cuda_graph_config": {
                "prefill_token_sizes": [512],
                "decode_batch_sizes": [1, 2, 4, 8],
            },
            "worker_config": {"standby": True},
        },
    )
    streamed: list[str] = []
    usage = None
    with (
        deployments.serve(config, serving=False) as server,
        deployments.follower(peer_host, server, repository),
    ):
        # The standby epoch over both hosts parked with its graphs: the
        # leader states its launch generations once it parked.
        server.wait_until(
            lambda: admin.replica(server).standby,
            STANDBY_S,
            "the deployment reported no parked standby",
        )
        before = admin.replica(server)
        with httpx.stream(
            "POST",
            f"{server.base_url}/v1/chat/completions",
            json=chat.stream_options(COUNT),
            timeout=server.scaled(KILL_GAP_S),
        ) as response:
            response.raise_for_status()
            killed = False
            for line in response.iter_lines():
                tokens, line_usage, done = chat.stream_tokens(line)
                if done:
                    break
                streamed.extend(tokens)
                usage = line_usage or usage
                if not killed and len(streamed) >= EARLY_TOKENS:
                    _kill_serving_peer(peer_host)
                    killed = True
        answer = chat.complete(
            server.base_url, COUNT, server.scaled(chat.COMPLETION_S)
        )
        reference = [
            entry["token"]
            for entry in answer["choices"][0]["logprobs"]["content"]
        ]
        after = admin.replica(server, after=admin.now(server))

    # The domain's standby epoch serves now, rather than an engine launched
    # anew, and the request continued on it where the kill stopped it.
    assert after.generation == before.standby[0], (before, after)
    assert streamed == reference
    assert usage == answer["usage"]
