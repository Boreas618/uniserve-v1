"""A profiling session's device storage in its GPU's quota.

Qwen3 serves on the GPU the module's lease granted with its device KV
region sized by the deployment's quota. A ``range_metrics`` session whose
pass takes more device storage than the quota has room for first has the
region shrink (``Resize``, which evicts unpinned blocks and never takes a
private or pinned one) and its rank release the tail segments the region no
longer covers; the rank's process item reserves the pass while the session
runs, and the region grows back into new segments once it ends. A pass that
even every segment past the first cannot make room for is refused, naming
its bytes.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.python.e2e.http_helpers import admin_config, admin_request
from tests.python.e2e.profiling_helpers import ended, serve, start, stop

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("qwen3"),
    # The module serves one deployment for all of its cases.
    pytest.mark.gpus(1),
]

# No kv_cache_config.max_tokens: the KV region takes the quota's rest beside
# the graph pool the profile run sized.
OPTIONS = (
    "dtype=bfloat16",
    "attention_backend=trtllm",
    "max_num_batched_tokens=512",
    "chat_config.reasoning_parser=none",
    "cuda_graph_config.prefill_token_sizes=[512]",
    "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
)

# A pass's device storage per range, and the unit CUPTI takes it in
# (docs/profiling.md).
RANGE_STORAGE = 1_420_288
GRANULARITY = 2 << 20


def _pass_storage(ranges: int) -> int:
    return -(-ranges * RANGE_STORAGE // GRANULARITY) * GRANULARITY


def _ranges_past(storage: int) -> int:
    """The fewest ranges whose pass takes more than `storage` bytes."""
    return storage // RANGE_STORAGE + 1


@pytest.fixture(scope="module")
def served(
    tmp_path_factory: pytest.TempPathFactory, build_server: Path
) -> Iterator[str]:
    """A Qwen3 deployment whose KV cache is sized by its GPU's quota."""
    with serve(
        tmp_path_factory.mktemp("served"), OPTIONS, build_server
    ) as base_url:
        yield base_url


def _quota(base_url: str) -> dict:
    """The quota of the worker's GPU."""
    status = admin_config(base_url)
    [quota] = [
        quota
        for node in status["quotas"]
        for quota in node["quotas"]
        if any(entry["item"] == "process" for entry in quota["reservations"])
    ]
    return quota


def _bytes(quota: dict, item: str) -> int:
    return sum(
        entry["bytes"]
        for entry in quota["reservations"]
        if entry["item"] == item
    )


def _room(quota: dict) -> int:
    return quota["limit"] - sum(
        entry["bytes"] for entry in quota["reservations"]
    )


def test_the_cache_makes_room_for_a_session_and_grows_back(
    served: str, tmp_path: Path
):
    before = _quota(served)
    # A pass 4 GiB past the quota's room.
    ranges = _ranges_past(_room(before) + (4 << 30))
    storage = _pass_storage(ranges)
    start(
        served,
        {
            "collectors": ["range_metrics"],
            "range_metrics": {"pass_ranges": ranges},
            "stages": ["decode"],
            "output_dir": str(tmp_path / "profiles"),
        },
    )
    during = _quota(served)
    assert _bytes(during, "process") == _bytes(before, "process") + storage
    given_back = _bytes(before, "cache_segment") - _bytes(
        during, "cache_segment"
    )
    assert given_back >= storage - _room(before), during
    assert _room(during) >= 0, during

    stopped = stop(served)
    assert stopped.status_code == 200, stopped.text
    assert ended(served, 60)["end_reason"] == "operator"
    after = _quota(served)
    assert _bytes(after, "process") == _bytes(before, "process"), after
    assert _bytes(after, "cache_segment") == _bytes(before, "cache_segment"), (
        after
    )


def test_a_pass_no_withdrawal_makes_room_for_is_refused(
    served: str, tmp_path: Path
):
    before = _quota(served)
    # Past the room and every segment of the cache together.
    ranges = _ranges_past(_room(before) + _bytes(before, "cache_segment"))
    refused = admin_request(
        served,
        "POST",
        "/admin/profile",
        body={
            "collectors": ["range_metrics"],
            "range_metrics": {"pass_ranges": ranges},
            "output_dir": str(tmp_path),
        },
        timeout_s=600,
    )
    assert refused.status_code == 400, refused.text
    assert str(_pass_storage(ranges)) in refused.text, refused.text
    after = _quota(served)
    assert _bytes(after, "cache_segment") == _bytes(before, "cache_segment"), (
        after
    )
    assert _bytes(after, "process") == _bytes(before, "process"), after
