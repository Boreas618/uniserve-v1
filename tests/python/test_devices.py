"""The GPU leases end-to-end cases hold (``uniserve_testing.devices``)."""

from __future__ import annotations

import fcntl
import threading
from pathlib import Path
from typing import IO

import pytest
from uniserve_testing import devices

pytestmark = pytest.mark.unit

#: How long, in seconds, a lease that should proceed may take to be granted.
GRANTED_S = 10.0


@pytest.fixture
def locks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The leases' lock directory, in a temporary directory of the test's
    own (the system's temporary directory is the leases' environment).
    """  # noqa: D205
    monkeypatch.setattr(devices.tempfile, "gettempdir", lambda: str(tmp_path))
    directory = devices._lock_directory()
    return directory


def _hold(directory: Path, device: int) -> IO[str]:
    """GPU `device` held as another case's lease holds it."""
    handle = (directory / f"gpu{device}.lock").open("a+")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def _lease_in_background(
    count: int,
) -> tuple[threading.Thread, list[tuple[int, ...]]]:
    """A lease of `count` of GPUs 0-2 taken on a thread, which records the
    GPUs it was granted and releases them.
    """  # noqa: D205
    granted: list[tuple[int, ...]] = []

    def lease() -> None:
        with devices.lease_gpus((0, 1, 2), count) as held:
            granted.append(held.devices)

    thread = threading.Thread(target=lease, daemon=True)
    thread.start()
    return thread, granted


def test_a_lease_takes_the_free_gpus(locks: Path) -> None:
    holder = _hold(locks, 1)

    with devices.lease_gpus((0, 1, 2), 2) as lease:
        assert lease.devices == (0, 2)

    holder.close()


def test_a_short_lease_takes_whichever_gpu_frees_first(locks: Path) -> None:
    first = _hold(locks, 1)
    second = _hold(locks, 2)
    thread, granted = _lease_in_background(2)

    # GPU 2 frees while GPU 1 stays held: the lease proceeds on 0 and 2.
    second.close()
    thread.join(GRANTED_S)

    assert granted == [(0, 2)]
    first.close()


def test_a_lease_waits_while_too_few_gpus_are_free(locks: Path) -> None:
    first = _hold(locks, 1)
    second = _hold(locks, 2)
    thread, granted = _lease_in_background(2)

    thread.join(3 * devices.POLL_S)
    assert granted == [], "a lease proceeded short of the GPUs it needs"

    first.close()
    thread.join(GRANTED_S)
    assert granted == [(0, 1)]
    second.close()
