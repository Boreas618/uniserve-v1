"""What a video rank admits, the layouts it prepares and where a request's
state lives (``uniserve.executor.video.plan``).
"""  # noqa: D205

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from uniserve.errors import SetupError
from uniserve.executor.video import condition_spans
from uniserve.executor.video.plan import (
    Canvas,
    Layout,
    LayoutFields,
    MediaPlan,
    SampleLayout,
    SamplePages,
    SampleRows,
    StateEnvelope,
    StateField,
    VideoSize,
    _CanvasExtent,
    admitted_canvases,
    preparation_order,
    prompt_capacities,
    storage_leading,
    vision_spans,
)

pytestmark = pytest.mark.unit


def test_vision_spans_are_maximal_runs_of_vision_tags():
    assert vision_spans([1, 0, 0, 1, 1, 0, 0, 0]) == ((1, 3), (5, 8))
    assert vision_spans([0, 1]) == ((0, 1),)
    assert vision_spans([1, 1, 2]) == ()
    assert vision_spans([]) == ()


def test_a_deployment_admits_the_offered_canvases_it_names():
    offered = [
        (768, "21:9", Canvas(1536, 672)),
        (768, "16:9", Canvas(1344, 768)),
        (768, "9:16", Canvas(768, 1344)),
        (480, "16:9", Canvas(832, 480)),
    ]
    # Without a selection every offered canvas is prepared.
    assert admitted_canvases(offered, [], []) == offered
    # Resolutions cross aspect ratios resolution-major, in the order the
    # deployment names them.
    assert admitted_canvases(offered, ["480p", "768p"], ["16:9"]) == [
        offered[3],
        offered[1],
    ]
    assert admitted_canvases(offered, ["768p"], ["9:16", "16:9"]) == [
        offered[2],
        offered[1],
    ]
    # Without resolutions, every offered class takes their place.
    assert admitted_canvases(offered, [], ["16:9"]) == [offered[1], offered[3]]
    # A pair the denoiser does not name refuses the deployment, naming what
    # it generates.
    with pytest.raises(SetupError, match="768p 21:9, 768p 16:9") as refused:
        admitted_canvases(offered, ["480p"], ["21:9"])
    assert "not 480p 21:9" in str(refused.value)
    # A name that is not a class, and a repeated name, are invalid.
    for resolutions, ratios in [
        (["768"], []),
        (["768p", "768p"], []),
        ([], ["16:9", "16:9"]),
    ]:
        with pytest.raises(ValueError):
            admitted_canvases(offered, resolutions, ratios)
    with pytest.raises(SetupError):
        admitted_canvases([], [], [])


def test_canvases_prepare_by_generated_elements_and_storage_sizes_the_maximum():
    wide = Canvas(1536, 672)
    landscape = Canvas(1344, 768)
    portrait = Canvas(768, 1344)
    small = Canvas(832, 480)
    # The canvases follow by generated elements, the equal ones in the order
    # they were named, whatever storage this rank's share takes; 21:9
    # generates as many elements as 16:9 and takes more storage, so its
    # layout sizes the rank's storage.
    extents = [
        _CanvasExtent(small, 40, 30),
        _CanvasExtent(landscape, 90, 100),
        _CanvasExtent(wide, 96, 100),
        _CanvasExtent(portrait, 90, 100),
    ]
    assert preparation_order(extents) == [landscape, wide, portrait, small]
    assert storage_leading(extents) == 2
    # Two ranks whose shares of the same canvases differ prepare them in one
    # order; the first canvas sizes the storage among equal ones.
    first = [
        _CanvasExtent(portrait, 90, 100),
        _CanvasExtent(landscape, 90, 100),
    ]
    second = [
        _CanvasExtent(portrait, 80, 100),
        _CanvasExtent(landscape, 95, 100),
    ]
    assert preparation_order(first) == preparation_order(second)
    assert preparation_order(first) == [portrait, landscape]
    assert (storage_leading(first), storage_leading(second)) == (0, 1)


def test_the_prompt_capacity_is_the_largest_text_capacity():
    # A list names the capacities below it, an entry equal to it repeats it,
    # one above it is refused, and the default steps stop below it.
    assert prompt_capacities([1024, 4096], 16384) == [1024, 4096, 16384]
    assert prompt_capacities([1024, 10240, 16384], 16384) == [
        1024,
        10240,
        16384,
    ]
    assert prompt_capacities([], 8192) == [1024, 2048, 4096, 6144, 8192]
    assert prompt_capacities([], 1000) == [1000]
    for listed in ([1024, 32768], [0, 1024]):
        with pytest.raises(ValueError):
            prompt_capacities(listed, 16384)


def _plan() -> MediaPlan:
    """A plan of two canvases, two frame counts and two prompt capacities,
    whose startup layouts name what they were sized for.
    """  # noqa: D205
    canvases = [Canvas(1344, 768), Canvas(768, 1344)]
    layouts = [
        Layout(frames, canvas.width, canvas.height, capacity, 0)
        for canvas in canvases
        for frames in (121, 49)
        for capacity in (2048, 1024)
    ]
    return MediaPlan(
        frame_counts=[49, 121],
        admitted=[(768, "16:9", canvases[0]), (768, "9:16", canvases[1])],
        canvases=canvases,
        text_capacities=[1024, 2048],
        max_text_tokens=2048,
        condition_rows=512,
        maximum=Layout(121, 1344, 768, 2048, 0),
        layouts=layouts,
        conditioned=[],
        largest=[],
    )


def test_ranks_prepare_their_startup_layouts_in_one_order():
    # Under sequence parallelism a rank's share of a layout differs between
    # ranks, so ranks select different maximum layouts; the layouts they
    # warm, whose steps exchange tensors, follow in one order.
    first = _plan()
    first.largest = [
        Layout(121, 1344, 768, 2048, 0),
        Layout(121, 768, 1344, 2048, 0),
    ]
    second = _plan()
    second.largest = list(first.largest)
    first.maximum, second.maximum = first.largest[0], first.largest[1]
    assert first.startup() == second.startup()
    assert first.startup() == [
        *first.layouts,
        *(layout for layout in first.largest if layout not in first.layouts),
    ]
    assert len(set(first.startup())) == len(first.startup())


def test_a_size_without_conditions_takes_its_startup_layout():
    plan = _plan()
    for canvas in plan.canvases:
        for frames in plan.frame_counts:
            for capacity in plan.text_capacities:
                layout = plan.startup_layout(frames, canvas, capacity, False)
                assert layout is not None
                assert (layout.canvas, layout.frames, layout.text_rows) == (
                    canvas,
                    frames,
                    capacity,
                )
    # Nothing outside the plan has a startup layout.
    assert plan.startup_layout(73, plan.canvases[0], 1024, False) is None
    assert plan.startup_layout(49, Canvas(832, 480), 1024, False) is None
    assert plan.startup_layout(49, plan.canvases[0], 4096, False) is None


def test_a_size_with_conditions_takes_its_widened_startup_layout_under_graphs():
    # Without graphs a size with conditions has no startup layout and is
    # evaluated in a layout of its own.
    plan = _plan()
    assert plan.startup_layout(49, plan.canvases[0], 1024, True) is None
    plan.conditioned = [
        layout._replace(condition_rows=plan.condition_rows)
        for layout in plan.layouts
    ]
    for canvas in plan.canvases:
        for frames in plan.frame_counts:
            for capacity in plan.text_capacities:
                layout = plan.startup_layout(frames, canvas, capacity, True)
                assert layout is not None
                assert (
                    layout.canvas,
                    layout.frames,
                    layout.text_rows,
                    layout.condition_rows,
                ) == (canvas, frames, capacity, 512)


def _model(condition_rows: int, holds: dict[int, bool]) -> SimpleNamespace:
    """A denoiser that sizes every request at ``condition_rows`` and holds
    it in the layouts of the text rows ``holds`` names.
    """  # noqa: D205
    return SimpleNamespace(
        size=lambda size: SimpleNamespace(condition_rows=condition_rows),
        resolve=lambda layout: layout,
        denoiser=SimpleNamespace(
            holds=lambda layout, resolved: holds[layout.text_rows]
        ),
    )


def test_a_request_evaluates_in_the_smallest_capacity_that_holds_it():
    plan = _plan()
    size = VideoSize(49, plan.canvases[0], 1500)
    admitted = plan.admit(_model(0, {1024: False, 2048: True}), size)
    assert not isinstance(admitted, str)
    assert (admitted.text_capacity, admitted.layout) == (
        2048,
        plan.startup_layout(49, size.canvas, 2048, False),
    )
    admitted = plan.admit(_model(0, {1024: True, 2048: True}), size)
    assert not isinstance(admitted, str)
    assert (admitted.text_capacity, admitted.layout) == (
        1024,
        plan.startup_layout(49, size.canvas, 1024, False),
    )
    # No layout holds the request, or its conditions exceed the plan's:
    # either is the request's refusal, the second naming the capacity the
    # deployment states.
    refusal = plan.admit(_model(0, {1024: False, 2048: False}), size)
    assert isinstance(refusal, str)
    refusal = plan.admit(_model(513, {1024: True, 2048: True}), size)
    assert isinstance(refusal, str)
    assert "513" in refusal and "video_config.max_condition_rows" in refusal
    # A size outside the plan's frame counts, canvases or prompt capacity is
    # refused before the denoiser sizes it.
    for outside in (
        VideoSize(73, plan.canvases[0], 1500),
        VideoSize(49, Canvas(832, 480), 1500),
        VideoSize(49, plan.canvases[0], 4096),
    ):
        assert isinstance(plan.admit(_model(0, {}), outside), str)


def test_condition_latents_follow_their_rows_across_encoding_rounds():
    # A keyframe of 4 rows, a reference video of 10 rows with 3 audio rows
    # and a voice of 2 audio rows; the encoder published the visual rows in
    # rounds of 6, 5 and 3 rows.
    conditions = [(4, 0), (10, 3), (0, 2)]
    assert condition_spans(conditions, [6, 5, 3], 5) == [
        [(0, 0, 4)],
        # The video continues through the second round into the third.
        [(0, 4, 2), (1, 0, 5), (2, 0, 3)],
        # The audio read follows the three visual rounds.
        [(3, 0, 3)],
        [(3, 3, 2)],
    ]
    # Reads that hold other rows than the request planned are refused.
    for rounds, tracks in (([6, 5], 5), ([6, 5, 3], 4), ([6, 5, 3], None)):
        with pytest.raises(ValueError):
            condition_spans(conditions, rounds, tracks)
    # Without audio, every read is a visual round.
    assert condition_spans([(4, 0)], [4], None) == [[(0, 0, 4)]]


def _field(
    name: str, dtype: torch.dtype, elements: int, sample: bool
) -> StateField:
    return StateField(name, dtype, elements, sample)


def _layout(
    video: int, audio: int, table: int, rows: int, shared: int = 0
) -> LayoutFields:
    """One layout's fields on a rank that holds every sample row but the
    video's last ``shared`` rows: a video and an audio sample of width 8, a
    timestep table and ``rows`` prompt rows of width 4.
    """  # noqa: D205
    return LayoutFields(
        state=(
            _field("video", torch.float32, (video - shared) * 8, True),
            _field("audio", torch.float32, audio * 8, True),
            _field("tables", torch.int64, table, False),
        ),
        samples=(
            ("video", SampleRows(video, 8, 0, video - shared)),
            ("audio", SampleRows(audio, 8, 0, audio)),
        ),
        noise=(("video", video * 8), ("audio", audio * 8)),
        condition_noise=0,
        text_condition=(rows, 4),
        workspace=0,
    )


def test_samples_start_on_aligned_offsets_within_their_pages():
    # 40 video rows of 8 elements take 320 elements, aligned to 512.
    samples = SampleLayout.of(_layout(40, 3, 5, 16), 256)
    assert samples.offsets == (0, 512)
    assert samples.elements == (320, 24)
    assert samples.pages == 3
    fields = _layout(1, 1, 1, 1)
    depth = LayoutFields(
        state=fields.state,
        samples=(("depth", SampleRows(1, 1, 0, 1)),),
        noise=fields.noise,
        condition_noise=0,
        text_condition=fields.text_condition,
        workspace=0,
    )
    with pytest.raises(SetupError):
        SampleLayout.of(depth, 256)


def test_the_envelope_holds_every_layout_field_at_its_largest():
    envelope = StateEnvelope.of(
        [_layout(40, 3, 5, 16), _layout(8, 30, 9, 4)], SamplePages(256, 4)
    )
    # Samples live in latent pages; only tables and the conditioning are
    # slot state on the device.
    assert "video" not in envelope.device
    assert envelope.device["tables"].elements == 9
    assert envelope.device["text_condition"].elements == 64
    assert envelope.slot_bytes() == 9 * 8 + 64 * 2
    # Host sources cover every field; noise and condition draws are FP32.
    assert envelope.host["video_source"].elements == 320
    assert envelope.host["audio_noise"].elements == 240
    assert envelope.host["audio_noise"].dtype == torch.float32
    assert envelope.host["condition_noise"].elements == 1
    # The scratch holds a slot's pages, any rank's share of a sample.
    assert envelope.samples == 4 * 256
    # A layout whose samples exceed a slot's pages is refused.
    with pytest.raises(SetupError):
        StateEnvelope.of([_layout(40, 3, 5, 16)], SamplePages(256, 2))
    # A rank's share of a sample grows and shrinks with the request's prompt
    # and conditions, so its source holds the complete sample wherever the
    # prepared layouts leave the rank fewer rows.
    envelope = StateEnvelope.of(
        [_layout(40, 3, 5, 16, shared=30)], SamplePages(256, 4)
    )
    assert envelope.host["video_source"].elements == 320
    # A field whose dtype changes across layouts is refused.
    changed = _layout(8, 30, 9, 4)
    changed = LayoutFields(
        state=(
            *changed.state[:2],
            _field("tables", torch.int32, 9, False),
        ),
        samples=changed.samples,
        noise=changed.noise,
        condition_noise=0,
        text_condition=changed.text_condition,
        workspace=0,
    )
    with pytest.raises(SetupError):
        StateEnvelope.of([_layout(40, 3, 5, 16), changed], SamplePages(256, 4))


def test_slots_own_consecutive_pages_after_the_padding_page():
    pages = SamplePages(256, 3)
    assert pages.for_slot(1) == range(1, 4)
    assert pages.for_slot(2) == range(4, 7)
    assert pages.pool_pages(2) == 7
