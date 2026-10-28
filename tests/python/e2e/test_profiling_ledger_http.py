"""A profiling session's device storage in its GPU's storage ledger.

Qwen3 serves on one GPU (``UNISERVE_WORLD_DEVICE``, default 0) with its
KV cache sized by the deployment's quota. A ``range_metrics`` session whose
pass takes more device storage than the ledger has room for first has the
KV cache give back tail segments for it; the worker's process item books
the pass while the session runs, and the cache grows back once it ends. A
pass that even every segment past the first cannot make room for is
refused, naming its bytes.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.profiling_helpers import ended, serve, start

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

# No kv_cache_config.max_tokens: the cache takes the quota's rest beside the
# share the capacity plan keeps for graphs.
OPTIONS = (
    "dtype=bfloat16",
    "chat_config.reasoning_parser=none",
    "cuda_graph_config.prefill_token_sizes=[512]",
    "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
)

# A pass's device storage per range, and the unit CUPTI takes it in
# (docs/profiling.md).
RANGE_STORAGE = 1_421_312
GRANULE = 2 << 20


def _pass_storage(ranges: int) -> int:
    return -(-ranges * RANGE_STORAGE // GRANULE) * GRANULE


def _ranges_past(storage: int) -> int:
    """The fewest ranges whose pass takes more than `storage` bytes."""
    return storage // RANGE_STORAGE + 1


@pytest.fixture(scope="module")
def served(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A Qwen3 deployment whose KV cache is sized by its GPU's quota."""
    with serve(tmp_path_factory.mktemp("served"), OPTIONS) as base_url:
        yield base_url


def _ledger(base_url: str) -> dict:
    """The storage ledger of the worker's GPU."""
    status = httpx.get(f"{base_url}/admin/config", timeout=60).json()
    [ledger] = [
        ledger
        for node in status["ledgers"]
        for ledger in node["ledgers"]
        if any(entry["item"] == "process" for entry in ledger["entries"])
    ]
    return ledger


def _bytes(ledger: dict, item: str) -> int:
    return sum(
        entry["bytes"] for entry in ledger["entries"] if entry["item"] == item
    )


def _room(ledger: dict) -> int:
    return ledger["quota"] - sum(entry["bytes"] for entry in ledger["entries"])


def test_the_cache_makes_room_for_a_session_and_grows_back(
    served: str, tmp_path: Path
):
    before = _ledger(served)
    # A pass 4 GiB past the ledger's room.
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
    during = _ledger(served)
    assert _bytes(during, "process") == _bytes(before, "process") + storage
    given_back = _bytes(before, "cache_segment") - _bytes(
        during, "cache_segment"
    )
    assert given_back >= storage - _room(before), during
    assert _room(during) >= 0, during

    stopped = httpx.delete(f"{served}/admin/profile", timeout=600)
    assert stopped.status_code == 200, stopped.text
    assert ended(served, 60)["end_reason"] == "operator"
    after = _ledger(served)
    assert _bytes(after, "process") == _bytes(before, "process"), after
    assert _bytes(after, "cache_segment") == _bytes(before, "cache_segment"), (
        after
    )


def test_a_pass_no_withdrawal_makes_room_for_is_refused(
    served: str, tmp_path: Path
):
    before = _ledger(served)
    # Past the room and every segment of the cache together.
    ranges = _ranges_past(_room(before) + _bytes(before, "cache_segment"))
    refused = httpx.post(
        f"{served}/admin/profile",
        json={
            "collectors": ["range_metrics"],
            "range_metrics": {"pass_ranges": ranges},
            "output_dir": str(tmp_path),
        },
        timeout=600,
    )
    assert refused.status_code == 400, refused.text
    assert str(_pass_storage(ranges)) in refused.text, refused.text
    after = _ledger(served)
    assert _bytes(after, "cache_segment") == _bytes(before, "cache_segment"), (
        after
    )
    assert _bytes(after, "process") == _bytes(before, "process"), after
