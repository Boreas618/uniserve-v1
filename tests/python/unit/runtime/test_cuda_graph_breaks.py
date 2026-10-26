"""A captured step with breaks replays its segments around each break's work.

:class:`uniserve.cuda_graph.BreakableCUDAGraph` launches its segments in
submission order and performs the work recorded at each break between the
two segments around it, as distributed expert weight copies require.
"""

import pytest

from uniserve.cuda_graph import BreakableCUDAGraph

pytestmark = pytest.mark.unit


class _Segment:
    """A segment graph that records its launch and release."""

    def __init__(self, name: str, log: list[str]) -> None:
        self.name, self.log = name, log

    def replay(self) -> None:
        self.log.append(self.name)

    def reset(self) -> None:
        self.log.append(f"reset {self.name}")


def test_replay_performs_each_break_between_its_segments():
    log: list[str] = []
    graph = BreakableCUDAGraph(
        [_Segment(f"segment {index}", log) for index in range(3)],
        output="output",
        capture_stream=None,
        breaks=[lambda: log.append("copy 0"), lambda: log.append("copy 1")],
    )

    assert graph.replay() == "output"
    assert log == ["segment 0", "copy 0", "segment 1", "copy 1", "segment 2"]


def test_segments_and_breaks_must_alternate():
    with pytest.raises(ValueError, match="one break between each two"):
        BreakableCUDAGraph(
            [_Segment("segment 0", [])],
            None,
            capture_stream=None,
            breaks=[lambda: None],
        )


def test_release_resets_every_segment_and_drops_the_break_work():
    log: list[str] = []
    graph = BreakableCUDAGraph(
        [_Segment("segment 0", log), _Segment("segment 1", log)],
        output="output",
        capture_stream=None,
        breaks=[lambda: log.append("copy")],
    )

    graph.release()

    assert log == ["reset segment 0", "reset segment 1"]
    assert graph.output is None
