"""A prompt's KV blocks cross hosts through the Storage Servers' peers.

**Peer reuse across hosts.** Qwen3 is served by two deployments of the same
checkpoint and served name: deployment A on this host, and deployment B,
whose engine runs on a second host (``engine_host``), with its rank, its
buffer pool and its host KV cache there, while its daemon serves B's API on
this host. B's ``kv_cache_config.peers`` names A's Storage Server. A answers
a long prompt; its write policy stores every full block it commits to A's
host tier, which A's own admission then loads back once its device copies
are evicted (a change of the package's ``flush``): that answer is computed
over the stored blocks. B then answers the same prompt: its admission finds
the keys in neither its device nor its host tier, and since the peers could
deepen the hit by at least ``prefetch_min_tokens``, it submits a
``Prefetch`` from the peers (``specs/scheduler-design.md`` §4.4). A's
Storage Server holds the host copies by key (``PeerHold``) while the
Transfer Server of B's host reads them through NIXL into B's host tier, and
B's admission loads them. B's first answer finds nothing cached on B itself,
so the prompt tokens it states as cached (:mod:`uniserve_testing.caching`)
are tokens the peers brought, and its greedy tokens equal A's answer over
the stored blocks, computed over the same keys and values. A prompt A never
saw brings nothing from A; once A's Storage Server stops, B computes a
prompt A stored, with the same tokens as A computed it, and no request fails
(a tier's failure never fails a request, ``specs/buffer-pool.md`` §12.4).

**Converted reuse across hosts.** Deployment B keeps its device KV blocks in
FP8 (``kv_cache_config.dtype=float8_e4m3fn``) while A keeps BF16: a key
names the tokens and the weights that compute them, not the cache's
precision, so B's prefetch finds A's BF16 host copies, which A's Storage
Server states in the encoding A's class wrote them in; B's buffer pool reads
them into its converted-load source region (``Class::source``), and B's load
converts each into its FP8 device block (``specs/buffer-pool.md`` §13, §20),
as B's own writes encode it. B's answer reuses the peers' blocks, and it
agrees with its answer over the same prompt's blocks it computed and wrote
itself (another ``cache_salt``, asked twice: computed cold, then pinned on
its device) by the rule of vLLM's ``check_logprobs_close``
(:func:`uniserve_testing.chat.assert_logprobs_close`) over the
``TOP_LOGPROBS`` most likely tokens. Both answers compute the prompt's last
tokens over FP8 blocks; FP8 blocks written whole from a block's amax and
blocks written token by token round differently, so the two are not
required to be identical.

The second host is the run's ``--peer-host`` (``uniserve_testing.plugin``):
B's package lists this host, its leader, and the second host by their host
names (:func:`uniserve_testing.deployments.two_hosts`), and the second
host's daemon serves the same package there and joins the leader
(:meth:`uniserve_testing.deployments.Deployments.follower`), at a Build
Server there that installs the tree's packages from this repository's path,
which the second host must mount at the same path, and on its first GPU.
B's engine there opens the checkpoint's snapshot at the path the leader
resolved, which the second host's Hugging Face cache must hold. Both hosts'
Storage Servers are their daemons' own, which they start with the settings
the deployments state: A's accepts other hosts at this host's name, which
the second host resolves, on a port it picks, which A's readiness record
states as the peer endpoint B names, and both read through the NIXL
package each host has installed (``docs/nixl.md``). The peer endpoints are
unauthenticated: the hosts share an isolated network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from uniserve_testing import admin, chat
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.caching import cached_tokens
from uniserve_testing.deployments import (
    Deployments,
    runtime_directory,
    stop_storage_server,
    this_host,
    two_hosts,
)
from uniserve_testing.packages import merged
from uniserve_testing.remote import RemoteHost
from uniserve_testing.servers import Server

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

SERVED_MODEL = "qwen3"
MAX_TOKENS = 32
# The fewest tokens B's answer must find cached from the peers, and A's
# prompt must reuse: sixteen blocks of the deployments' block size
# (`kv_cache_config.block_size`, 64 tokens by default).
PEER_TOKENS = 1024
# The candidates two greedy answers are compared at, as vLLM's model
# correctness tests compare logprobs.
TOP_LOGPROBS = 5
# How long the host tier of A may take to hold a prompt's blocks after A
# answered it, in seconds: the write policy's copies run with the batches
# that follow the commits.
STORED_S = 120


def _prompt(label: str) -> str:
    """A long prompt of records under `label`, whose first block is the
    label's own, so that prompts of two labels share no block.
    """  # noqa: D205
    return f"{label}. {chat.long_prompt(260)}"


def _ask(base_url: str, prompt: str, **fields: object) -> dict:
    """The greedy answer to `prompt`, each token reported by its token id,
    with `fields` added to the request.
    """  # noqa: D205
    return chat.complete(
        base_url,
        chat.body(
            SERVED_MODEL,
            prompt,
            MAX_TOKENS,
            **chat.GREEDY,
            logprobs=True,
            return_tokens_as_token_ids=True,
            **fields,
        ),
    )


def _stored(server: Server, prompt: str) -> tuple[dict, dict]:
    """A's answers to `prompt`: computed whole, and computed over the blocks
    A's host tier stores once it holds every reusable one.

    The prompt asked again finds its whole reuse limit on the device. Once the
    device copies are evicted (:func:`uniserve_testing.admin.flush`), the
    prompt asked once more loads what the host tier holds, A's only other
    tier, so its answer reuses the whole reuse limit once the host tier
    holds every block. The write policy stores a block once it is committed,
    by a copy its executor runs with the next batch, so the copies may still
    run when the request has answered, and a flush converges without waiting
    for them: the case evicts and asks again until the answer loads every
    block, a state the deployment announces no event of.
    """  # noqa: D205
    computed = _ask(server.base_url, prompt)
    reusable = cached_tokens(_ask(server.base_url, prompt))
    answers: list[dict] = []

    def loaded() -> bool:
        admin.flush(server)
        answers.append(_ask(server.base_url, prompt))
        return cached_tokens(answers[-1]) == reusable

    server.wait_until(
        loaded, STORED_S, "the host tier did not hold the prompt's blocks"
    )
    return computed, answers[-1]


def _writer() -> dict[str, Any]:
    """Deployment A's options: a Storage Server that also accepts other
    hosts at this host's name, which the second host resolves, on a port it
    picks, which A's readiness record states as its peer endpoint.
    """  # noqa: D205
    return {
        "model": CHECKPOINTS["qwen3"].model,
        "served_model_name": SERVED_MODEL,
        "storage_config": {"tcp": f"{this_host()}:0"},
    }


def _reader(peer_host: RemoteHost, peer: str) -> dict[str, Any]:
    """Deployment B's options: a leader here, the first of two hosts, whose
    engine runs on the second host, and whose peers are the Storage Server
    at `peer`.
    """  # noqa: D205
    return merged(
        two_hosts(peer_host),
        {
            "model": CHECKPOINTS["qwen3"].model,
            "served_model_name": SERVED_MODEL,
            "engine_host": peer_host.name,
            "kv_cache_config": {"peers": [peer]},
        },
    )


def test_a_host_loads_a_prompts_blocks_from_a_peer_hosts_host_tier(
    uniserve: Path,
    repository: Path,
    deployments: Deployments,
    peer_host: RemoteHost,
    tmp_path: Path,
):
    shared = _prompt("Shared")
    dropped = _prompt("Dropped")
    unseen = _prompt("Unseen")
    # B's host reads A's blocks through NIXL: both hosts' calibrations
    # time those READs.
    calibration = deployments.calibration(reads=True)
    with (
        runtime_directory(uniserve, tmp_path / "a.storage") as runtime,
        deployments.serve(
            _writer(), name="a", runtime=runtime, calibration=calibration
        ) as a,
    ):
        # A computes both prompts and stores their full blocks to its host
        # tier. Its answer to the shared prompt over the stored blocks
        # computes only the last tokens over them, as B's does over the
        # blocks it reads: the two compute the same tokens over the same
        # cached keys and values (a prompt computed whole may round
        # otherwise, as `test_host_kv_cache_http.py` states).
        _, a_shared = _stored(a, shared)
        a_dropped, _ = _stored(a, dropped)

        with (
            deployments.serve(
                _reader(peer_host, a.endpoint("peer")),
                name="b",
                serving=False,
                calibration=calibration,
            ) as b,
            deployments.follower(
                peer_host,
                b,
                repository,
                calibration=calibration,
                reads=True,
                name="b-member",
            ),
        ):
            # B reads the shared prompt's blocks from A: B holds no block
            # of its own yet, so what its answer finds cached the peers
            # brought.
            b_shared = _ask(b.base_url, shared)
            # A prompt A never saw brings nothing from A, and B's answer is
            # A's.
            b_unseen = _ask(b.base_url, unseen)
            a_unseen = _ask(a.base_url, unseen)
            # A's Storage Server is gone: B computes what A stored.
            stop_storage_server(uniserve, runtime, tmp_path / "a.storage")
            b_dropped = _ask(b.base_url, dropped)

    assert cached_tokens(a_shared) >= PEER_TOKENS, a_shared["usage"]
    assert cached_tokens(b_shared) >= PEER_TOKENS, b_shared["usage"]
    assert chat.ranked_tokens(b_shared) == chat.ranked_tokens(a_shared)
    assert cached_tokens(b_unseen) == 0, b_unseen["usage"]
    assert chat.ranked_tokens(b_unseen) == chat.ranked_tokens(a_unseen)
    assert cached_tokens(b_dropped) == 0, b_dropped["usage"]
    assert chat.ranked_tokens(b_dropped) == chat.ranked_tokens(a_dropped)


def test_an_fp8_host_loads_a_bf16_hosts_blocks_by_converting_them(
    repository: Path,
    deployments: Deployments,
    peer_host: RemoteHost,
):
    shared = _prompt("Converted")
    calibration = deployments.calibration(reads=True)
    with deployments.serve(_writer(), name="a", calibration=calibration) as a:
        # A computes the prompt in BF16 and stores its full blocks to its
        # host tier.
        _, bf16 = _stored(a, shared)

        with (
            deployments.serve(
                merged(
                    _reader(peer_host, a.endpoint("peer")),
                    {"kv_cache_config": {"dtype": "float8_e4m3fn"}},
                ),
                name="b",
                serving=False,
                calibration=calibration,
            ) as b,
            deployments.follower(
                peer_host,
                b,
                repository,
                calibration=calibration,
                reads=True,
                name="b-member",
            ),
        ):
            # The same prompt on B under a salt no cache holds: computed
            # cold, then over the FP8 blocks B wrote itself. Their keys are
            # the salt's, so what the unsalted answer finds cached the peers
            # brought.
            _ask(b.base_url, shared, cache_salt="local")
            local = _ask(
                b.base_url,
                shared,
                cache_salt="local",
                top_logprobs=TOP_LOGPROBS,
            )
            converted = _ask(b.base_url, shared, top_logprobs=TOP_LOGPROBS)

    assert cached_tokens(bf16) >= PEER_TOKENS, bf16["usage"]
    assert cached_tokens(converted) >= PEER_TOKENS, converted["usage"]
    chat.assert_logprobs_close(
        chat.ranked_tokens(local),
        chat.ranked_tokens(converted),
        "an answer over converted BF16 blocks and one over FP8 blocks",
    )
