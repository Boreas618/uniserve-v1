"""A streaming request survives the loss of its engine's process.

Qwen3 streams a long greedy count to a client. While it streams, the engine
process that serves it is killed: the process holds the scheduler and every
rank of the deployment's groups on this host, one executor thread per
device. The buffer pool sees the process exit and detaches its owners; what
the request committed stays cached (``specs/buffer-pool.md`` §12.4, §14). The
leader's daemon starts a new engine process, which finds its weights retained
at the Storage Server, and submits the request again with the tokens it
generated once the new engine serves; the request continues the count where
it stopped. The client reads one uninterrupted stream.

The stream decodes the tokens the engine decodes for the same request alone
afterwards, and its usage charges the same prompt and completion. The count
is a continuation that no near tie decides, so rounding differences between
the lost engine's decode of the generated tokens and the new engine's
prefill of them cannot change it. The deployment captures few graphs (decode
batches up to 8, prefill buckets of 512 tokens and the batch bound), so that
the engine starts again while the count streams.

Two schedulers that share their experts (``data_parallel_size = 2`` with
expert parallelism), or attention and expert workers exchanging through
DeepEP, form one expert epoch, whose main process keeps a parked standby
(``worker_config.standby``): once it serves, a second main process of the
same epoch starts beside it, captures its graphs and parks, its device
storage kept resident or suspended (``worker_config.standby_storage``:
suspended, it also releases its all-to-all workspaces and NCCL
communicators, which its ranks restore together at the promotion, and frees
DeepEP's windows, which cannot keep their addresses: the promotion rebuilds
them and captures the graphs that run expert steps again). Killing the
serving main process promotes the standby in place of a relaunch, and the
request continues on it.

Which main process serves before and after a kill is what ``GET
/admin/status`` reports: the serving process's launch generation
(``generation``) and the parked standby's (``standby``), which a promotion
makes the serving one; the status's ``report`` names the process of each
launch generation (``processes``). Each test's deployment runs a Storage
Server of its own.
"""

from __future__ import annotations

import os
import signal
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from uniserve_testing import admin, chat, client
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.deployments import Deployments
from uniserve_testing.packages import Config, merged

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

NAME = "qwen3-relaunch"
# Tokens the client reads before a process is killed.
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

#: Qwen3 under :data:`NAME` with few captured graphs, so that an engine
#: starts again while the count streams, and a small KV cache.
OPTIONS: dict[str, Any] = {
    "model": CHECKPOINTS["qwen3"].model,
    "served_model_name": NAME,
    "kv_cache_config": {"max_tokens": 32768},
    "chat_config": {"reasoning_parser": "none"},
    "cuda_graph_config": {
        "prefill_token_sizes": [512],
        "decode_batch_sizes": [1, 2, 4, 8],
    },
}

#: How long a standby takes to park, in seconds (scaled): it starts as the
#: deployment's engine did, but maps the weights the serving epoch holds
#: instead of loading them, and captures the same few graphs.
STANDBY_S = 900.0

#: How long the stream may wait for its next line, in seconds (scaled):
#: across the kill, until another main process serves the request, which a
#: relaunch starts as the deployment's engine started, on retained weights.
KILL_GAP_S = 1800.0


@dataclass
class Killed:
    """What a stream through a kill read, and what the deployment reported.

    `streamed` and `usage` are the stream's tokens and usage, `reference`
    and `reference_usage` those of the count asked again afterwards.
    `before` and `after` are the deployment of ``GET /admin/status`` before
    the kill and once the count was asked again.
    """

    streamed: list[str]
    usage: dict[str, Any] | None
    reference: list[str]
    reference_usage: dict[str, Any]
    before: dict[str, Any]
    after: dict[str, Any]


def _stream_through_a_kill(
    deployments: Deployments, options: Config, standby: bool = False
) -> Killed:
    """Stream the count, killing the serving engine's process after the
    first tokens.

    The deployment serves with `options` over :data:`OPTIONS`; with
    `standby`, the stream starts once the deployment reports its standby
    parked.
    """  # noqa: D205
    streamed: list[str] = []
    usage = None
    with deployments.serve(merged(OPTIONS, options)) as server:
        if standby:
            # No event of the node's log announces a standby that parks: the
            # status states it once it parked.
            server.wait_until(
                lambda: admin.status(server)["deployment"]["standby"],
                STANDBY_S,
                "the deployment reported no parked standby",
            )
        before = admin.status(server)["deployment"]
        engine = admin.serving_engine(server)["pid"]
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
                    os.kill(engine, signal.SIGKILL)
                    killed = True
        answer = chat.complete(
            server.base_url, COUNT, server.scaled(chat.COMPLETION_S)
        )
        after = admin.status(server)["deployment"]
    return Killed(
        streamed=streamed,
        usage=usage,
        reference=[
            entry["token"]
            for entry in answer["choices"][0]["logprobs"]["content"]
        ],
        reference_usage=answer["usage"],
        before=before,
        after=after,
    )


def _continued(killed: Killed) -> None:
    """The stream read the count the deployment answers alone, with the
    same usage: it continued where the kill stopped it, and the deployment
    serves again.
    """  # noqa: D205
    assert killed.streamed == killed.reference
    assert killed.usage == killed.reference_usage
    assert killed.after["engine"] == "serving", killed.after


def test_a_request_continues_after_its_engine_process_is_lost(
    deployments: Deployments,
):
    killed = _stream_through_a_kill(deployments, {})
    _continued(killed)
    # The daemon started another engine process, which serves.
    assert killed.after["generation"] not in (
        None,
        killed.before["generation"],
    ), (killed.before, killed.after)


def _expert_epoch(topology: str) -> dict[str, Any]:
    """The options of an expert epoch of two groups on the case's two GPUs.

    ``replicas``: two schedulers sharing their experts over the all-to-all.
    ``dedicated``: attention on the first GPU and the routed experts on the
    second, exchanging through DeepEP, whose windows a suspended standby
    frees and its promotion rebuilds.
    """
    if topology == "replicas":
        return {"data_parallel_size": 2}
    return client.dedicated_experts()


@pytest.mark.gpus(2)
@pytest.mark.parametrize("topology", ["replicas", "dedicated"])
@pytest.mark.parametrize("standby_storage", ["resident", "suspended"])
def test_an_expert_epoch_promotes_its_standby_when_its_engine_is_lost(
    deployments: Deployments,
    standby_storage: str,
    topology: str,
):
    options = merged(
        _expert_epoch(topology),
        {
            "moe_config": {"expert_parallel": True},
            "worker_config": {
                "standby": True,
                "standby_storage": standby_storage,
            },
        },
    )
    killed = _stream_through_a_kill(deployments, options, standby=True)
    _continued(killed)
    # The parked standby serves now, rather than an engine launched anew.
    standby = killed.before["standby"]
    assert killed.after["generation"] == standby[0], (
        killed.before,
        killed.after,
    )
