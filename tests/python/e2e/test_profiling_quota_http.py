"""A profiling session's device storage in its GPU's quota.

Qwen3 serves on the GPU the module's lease holds with its device KV region
sized by the deployment's quota. A ``range_metrics`` session whose pass takes
more device storage than the quota has room for first has the region shrink
(``Resize``, which evicts unpinned blocks and never takes a private or pinned
one) and its rank release the tail segments the region no longer covers; the
rank's process item reserves the pass while the session runs, and the region
grows back into new segments once it ends. A pass that even every segment
past the first cannot make room for is refused, naming its bytes.

A pass's device storage is what its worker measures when the session starts
(``docs/profiling.md``, Range profiler passes), which the session reports
(``workers``' ``device_storage``).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from uniserve_testing import admin, profiling
from uniserve_testing.deployments import Deployments
from uniserve_testing.servers import Server

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("qwen3"),
    # The module serves one deployment for all of its cases.
    pytest.mark.gpus(1),
]

#: The ranges of a pass a session states nothing for (``pass_ranges``).
DEFAULT_RANGES = 512


@pytest.fixture(scope="module")
def served(module_deployments: Deployments) -> Iterator[Server]:
    """A Qwen3 deployment whose KV cache is sized by its GPU's quota: no
    ``kv_cache_config.max_tokens``, so the KV region takes the quota's rest
    beside the graph pool the profile run sized.
    """  # noqa: D205
    with profiling.serve(module_deployments, {}) as server:
        yield server


def _quota(server: Server) -> dict[str, Any]:
    """The quota of the worker's GPU: the one its process item is in."""
    [quota] = [
        quota
        for quota in admin.quotas(server)
        if any(entry["item"] == "process" for entry in quota["reservations"])
    ]
    return quota


def _bytes(quota: dict[str, Any], item: str) -> int:
    """The bytes the reservations of `item` hold in `quota`."""
    return sum(
        entry["bytes"]
        for entry in quota["reservations"]
        if entry["item"] == item
    )


def _room(quota: dict[str, Any]) -> int:
    """The bytes `quota` leaves unreserved."""
    return quota["limit"] - sum(
        entry["bytes"] for entry in quota["reservations"]
    )


def _session(output: Path, ranges: int = DEFAULT_RANGES) -> dict[str, Any]:
    """A range_metrics session of decode steps whose pass holds `ranges`."""
    return {
        "collectors": ["range_metrics"],
        "range_metrics": {"pass_ranges": ranges},
        "stages": ["decode"],
        "output_dir": str(output),
    }


def test_the_cache_makes_room_for_a_session_and_grows_back(
    served: Server, tmp_path: Path
):
    before = _quota(served)
    session = profiling.start(served, _session(tmp_path / "profiles"))
    [worker] = session["workers"]
    storage = worker["device_storage"]
    # The KV region takes the quota's rest to within an allocation granule of
    # each of its planes, which leaves a pass less room than it takes.
    assert storage > _room(before), (worker, before)

    during = _quota(served)
    assert _bytes(during, "process") == _bytes(before, "process") + storage
    given_back = _bytes(before, "cache_segment") - _bytes(
        during, "cache_segment"
    )
    assert given_back >= storage - _room(before), during
    assert _room(during) >= 0, during

    stopped = profiling.stop(served)
    assert stopped["end_reason"] == "operator", stopped
    after = _quota(served)
    assert _bytes(after, "process") == _bytes(before, "process"), after
    assert _bytes(after, "cache_segment") == _bytes(before, "cache_segment"), (
        after
    )


def test_a_pass_no_withdrawal_makes_room_for_is_refused(
    served: Server, tmp_path: Path
):
    # A pass of the default ranges, as its worker measures it.
    measured = profiling.start(served, _session(tmp_path / "measured"))
    [worker] = measured["workers"]
    default_pass = worker["device_storage"]
    profiling.stop(served)

    before = _quota(served)
    # A pass's storage grows with its ranges, in proportion to them
    # (docs/profiling.md): enough default passes' ranges to take the room
    # and every segment of the cache together, and one more pass's to cover
    # the rounding of its storage.
    beyond = _room(before) + _bytes(before, "cache_segment")
    ranges = DEFAULT_RANGES * (-(-beyond // default_pass) + 1)
    refused = admin.request(
        served, "POST", profiling.PROFILE, body=_session(tmp_path, ranges)
    )
    assert refused.status_code == 400, refused.text
    # The refusal names the bytes the pass holds, as its worker measured
    # them: more than every withdrawal could make room for.
    named = re.search(r"holds (\d+) bytes", refused.text)
    assert named is not None and int(named[1]) > beyond, refused.text
    after = _quota(served)
    assert _bytes(after, "cache_segment") == _bytes(before, "cache_segment"), (
        after
    )
    assert _bytes(after, "process") == _bytes(before, "process"), after
