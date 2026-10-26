"""How a text batch's calls advance and commit.

``uniserve.executor.text.commit``.
"""

from __future__ import annotations

import numpy as np
import pytest

from uniserve.executor.text.commit import (
    CommitWork,
    commit_bucket,
    finish_rows,
)
from uniserve.executor.text.protocol import CallKind
from uniserve.sampling.state import CACHE, DECODE, EXPLICIT, VERIFY

pytestmark = pytest.mark.unit

PREFILL, DECODE_CALL, VERIFY_CALL = (
    CallKind.PREFILL,
    CallKind.DECODE,
    CallKind.VERIFY,
)

# One call of each kind: a prefill that only writes the cache, a sampled
# prefill, a decode, a verification of two drafts, and a media prompt
# without and with a sample.
CALLS = {
    "kind": [PREFILL, PREFILL, DECODE_CALL, VERIFY_CALL, PREFILL, PREFILL],
    "slot": [1, 2, 3, 4, 5, 6],
    "prefix": [4, 0, 9, 6, 10, 0],
    "query": [3, 5, 1, 3, 4, 6],
    "logical": [4, 0, 9, 6, 3, 0],
    "rng": [2, 0, 3, 5, 1, 0],
    "allocated": [8, 8, 10, 9, 14, 6],
    "task": [-1, 0, 1, 2, -1, 3],
    "penalties": [True, True, False, True, True, True],
    "prompt_ready": [False, False, False, False, True, False],
    "chunks": [False, True, False, False, False, True],
    "media": [False, False, False, False, True, True],
    "media_position": [0, 0, 0, 0, 20, 30],
    "media_continued": [0, 0, 0, 0, 0, 25],
}  # fmt: skip


def _finish(**changes: list) -> object:
    calls = {**CALLS, **changes}
    return finish_rows(**{name: np.asarray(v) for name, v in calls.items()})


def test_each_call_kind_advances_and_commits_its_own_row():
    finished = _finish()
    assert finished.rows.tolist() == [
        # A prefill without a task sets its cache length alone.
        [1, CACHE, -1, 7, 0, 0, 0, 0],
        # A sampled prefill installs its token at host coordinates.
        [2, EXPLICIT, 0, 5, 5, 1, 0, 1],
        # A decode advances on the device.
        [3, DECODE, 1, 0, 0, 0, 0, 0],
        # A verification advances from its base by its emitted tokens.
        [4, VERIFY, 2, 0, 6, 5, 6, 1],
        # A media prompt writes its rows and continues at explicit
        # positions.
        [5, CACHE, -1, 14, 0, 0, 0, 0],
        [6, EXPLICIT, 3, 6, 25, 1, 0, 1],
    ]
    # A cache-only prefill keeps its logical position; a verification
    # reports its speculation base.
    assert finished.logical.tolist() == [4, 5, 10, 6, 20, 30]
    assert finished.rng.tolist() == [2, 1, 4, 5, 1, 1]
    assert finished.visible.tolist() == [7, 5, 10, 6, 14, 6]
    assert finished.computed.tolist() == [7, 5, 10, 9, 14, 6]
    assert finished.speculation.tolist() == [0, 0, 0, 1, 0, 0]
    # Scored chunks retain their prompt logits; readiness is otherwise
    # kept.
    assert finished.prompt_ready.tolist() == [0, 1, 0, 0, 1, 1]


def test_calls_that_cannot_commit_are_invariant_violations():
    # A decode always samples.
    with pytest.raises(AssertionError):
        _finish(task=[-1, 0, -1, 2, -1, 3])
    # A call writes within its slot's allocation.
    with pytest.raises(AssertionError):
        _finish(allocated=[6, 8, 10, 9, 14, 6])


def test_commit_rows_pad_to_their_bucket_with_inert_rows():
    # The smallest decode bucket holding the rows, else the binding's rows.
    assert commit_bucket(3, [8, 4, 16], 32) == 4
    assert commit_bucket(17, [8, 4, 16], 32) == 32

    rows = _finish().rows[:3]
    work = CommitWork(rows=rows, tails=np.zeros((0, 2), dtype=np.int64))
    columns = work.columns(4)
    assert columns.shape == (8, 4)
    assert columns[:, :3].T.tolist() == rows.tolist()
    assert columns[:, 3].tolist() == [0, CACHE, -1, 0, 0, 0, 0, 0]
    with pytest.raises(AssertionError):
        work.columns(2)
