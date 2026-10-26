"""Image trajectories (``uniserve.executor.image.trajectory``)."""

from __future__ import annotations

import pytest

from tests.python.fixtures.stub import load_stub
from uniserve.diffusion import Branch, Renorm
from uniserve.executor.image.trajectory import (
    BranchSet,
    Coordinates,
    ForwardSlot,
    GuidancePlan,
    ImageParams,
    ImageTrajectory,
    Prefix,
    PrefixFill,
    Trajectories,
    admitted_prefix,
)
from uniserve.processing import BranchSource

pytestmark = pytest.mark.unit

# A request's identity: engine, request id and epoch.
KEY = (1, 7, 1)

# The request's own conditioning prefix: 40 visible tokens of slot 3.
CONDITIONING = Coordinates(slot=3, visible=40, capacity=64)

# The prefix each source conditions on: the request's own context, three
# negative tokens and the one-token start of a conversation.
PREFIXES = {
    BranchSource.CONDITIONING: Prefix(tokens=(), reuses_conditioning=True),
    BranchSource.NEGATIVE_OR_START: Prefix(
        tokens=(1, 2, 3), reuses_conditioning=False
    ),
    BranchSource.START: Prefix(tokens=(1,), reuses_conditioning=False),
}


def _plan(masks: list[int]) -> GuidancePlan:
    return GuidancePlan(
        timesteps=tuple(float(step) for step in range(len(masks))),
        branches=tuple(BranchSet(mask) for mask in masks),
    )


def _trajectory_of(masks: list[int]) -> ImageTrajectory:
    trajectory = ImageTrajectory(KEY, (64, 64), _plan(masks), PREFIXES)
    trajectory.begin_call(CONDITIONING)
    return trajectory


def _table(slot: int, visible: int) -> Coordinates:
    return Coordinates(slot=slot, visible=visible, capacity=16)


def test_the_negative_branch_conditions_on_exactly_the_admitted_tokens():
    given = admitted_prefix(BranchSource.NEGATIVE_OR_START, [4, 5])
    assert given is not None and given.tokens == (4, 5)
    conditioning = admitted_prefix(BranchSource.CONDITIONING, [4, 5])
    assert conditioning is not None and conditioning.reuses_conditioning
    assert admitted_prefix(BranchSource.START, [4, 5]) is None

    # A request admitted without negative tokens: the engine reserves its
    # alternative prefix no tokens and gives that branch's denoising row no
    # context. The branch conditions on an empty prefix, which fits the
    # slot and needs no prefill.
    empty = admitted_prefix(BranchSource.NEGATIVE_OR_START, [])
    assert empty == Prefix(tokens=(), reuses_conditioning=False)
    trajectory = ImageTrajectory(
        KEY,
        (64, 64),
        _plan([3]),
        {
            BranchSource.CONDITIONING: PREFIXES[BranchSource.CONDITIONING],
            BranchSource.NEGATIVE_OR_START: empty,
        },
    )
    trajectory.begin_call(CONDITIONING)
    rows = [
        ForwardSlot(slot=3, seq_len=41, query_len=1),
        ForwardSlot(slot=5, seq_len=1, query_len=1),
    ]
    fills = trajectory.resolve_branches(
        BranchSet(3),
        BranchSource.START,
        rows,
        lambda slot, visible: Coordinates(slot, visible, capacity=0),
    )
    assert fills == []


def test_branch_sets_keep_evaluation_order_and_require_the_conditioned_branch():
    assert list(BranchSet(7)) == [
        Branch.CONDITIONED,
        Branch.TEXT_UNCONDITIONAL,
        Branch.IMAGE_UNCONDITIONAL,
    ]
    assert len(BranchSet(5)) == 2
    with pytest.raises(ValueError):
        BranchSet(2)
    with pytest.raises(ValueError):
        BranchSet(9)


def test_a_first_interval_prefills_each_new_prefix_once_per_slot():
    trajectory = _trajectory_of([7])
    branches, _ = trajectory.plan.step(0)
    # Two prefix prefills (slots 5 and 6), then three denoising rows.
    descriptors = [
        ForwardSlot(slot=5, seq_len=3, query_len=3),
        ForwardSlot(slot=6, seq_len=1, query_len=1),
        ForwardSlot(slot=3, seq_len=40, query_len=0),
        ForwardSlot(slot=5, seq_len=3, query_len=3),
        ForwardSlot(slot=6, seq_len=1, query_len=1),
    ]
    fills = trajectory.resolve_branches(
        branches, BranchSource.START, descriptors, _table
    )
    assert fills == [
        PrefixFill(slot=5, tokens=(1, 2, 3)),
        PrefixFill(slot=6, tokens=(1,)),
    ]
    trajectory.prefilled(5, 3)
    trajectory.prefilled(6, 1)

    # The conditioned branch continues the request; the others follow their
    # prefixes.
    assert trajectory.branch_rows(branches, 41) == [
        (Branch.CONDITIONED, CONDITIONING, 41),
        (Branch.TEXT_UNCONDITIONAL, Coordinates(5, visible=3, capacity=16), 3),
        (Branch.IMAGE_UNCONDITIONAL, Coordinates(6, visible=1, capacity=16), 1),
    ]


def test_a_later_interval_reads_resident_prefixes_without_prefill():
    trajectory = _trajectory_of([3, 3])
    branches, _ = trajectory.plan.step(1)
    descriptors = [
        ForwardSlot(slot=3, seq_len=40, query_len=0),
        ForwardSlot(slot=5, seq_len=3, query_len=0),
    ]
    fills = trajectory.resolve_branches(
        branches, BranchSource.START, descriptors, _table
    )
    assert fills == []
    assert trajectory.branch_rows(branches, 41)[1].coordinates.visible == 3

    # A resident prefix whose length disagrees with the tokens is refused.
    trajectory.begin_call(CONDITIONING)
    stale = [descriptors[0], ForwardSlot(slot=5, seq_len=2, query_len=0)]
    with pytest.raises(ValueError):
        trajectory.resolve_branches(branches, BranchSource.START, stale, _table)


def test_prefixes_beyond_the_slot_capacity_and_short_descriptors_are_refused():
    trajectory = _trajectory_of([3])
    branches = BranchSet(3)
    descriptors = [ForwardSlot(slot=3, seq_len=40, query_len=0)]
    with pytest.raises(ValueError):
        trajectory.resolve_branches(
            branches, BranchSource.START, descriptors, _table
        )

    def small(slot: int, visible: int) -> Coordinates:
        return Coordinates(slot=slot, visible=visible, capacity=2)

    descriptors = [
        descriptors[0],
        ForwardSlot(slot=5, seq_len=3, query_len=3),
    ]
    with pytest.raises(ValueError):
        trajectory.resolve_branches(
            branches, BranchSource.START, descriptors, small
        )


def test_reused_slots_never_see_an_earlier_requests_trajectory():
    trajectories = Trajectories()
    first = (1, 7, 1)
    trajectories.open(3, ImageTrajectory(first, (64, 64), _plan([1]), {}))
    assert trajectories.get(3, first) is not None
    assert trajectories.get(3, (1, 8, 1)) is None
    trajectories.release([3])
    assert trajectories.get(3, first) is None


# Mask bits of the guidance branches (``BranchSet.mask``).
BITS = {
    Branch.CONDITIONED: 1,
    Branch.TEXT_UNCONDITIONAL: 2,
    Branch.IMAGE_UNCONDITIONAL: 4,
}


@pytest.mark.parametrize(
    "values",
    [
        # Text and image guidance inside an interval, on a shifted schedule.
        {
            "steps": 6,
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 2.0,
            "cfg_renorm_type": "global",
            "cfg_renorm_min": 0.0,
            "cfg_interval": [0.3, 0.8],
            "timestep_shift": 3.0,
            "height": 64,
            "width": 64,
            "seed": 11,
        },
        # Text guidance on the denoiser's default shift.
        {
            "steps": 4,
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.0,
            "cfg_renorm_type": "text_channel",
            "cfg_renorm_min": 0.0,
            "cfg_interval": [0.0, 0.6],
            "timestep_shift": 0.0,
            "height": 32,
            "width": 48,
        },
    ],
)
def test_plans_evaluate_what_the_denoiser_selects_per_step(values):
    denoiser = load_stub().module.denoiser
    params = ImageParams.from_json(values)
    plan = GuidancePlan.make(denoiser, params)

    # The trajectory as the denoiser makes it from the request's parameters:
    # a non-positive shift keeps the denoiser's default.
    shift = params.timestep_shift
    schedules = denoiser.make_schedules(
        params.steps, shift=shift if shift > 0 else None, device="cpu"
    )
    guidance = denoiser.make_guidance(
        text_scale=params.cfg_text_scale,
        image_scale=params.cfg_img_scale,
        interval=params.cfg_interval,
        renorm=Renorm(params.cfg_renorm_type),
        renorm_min=params.cfg_renorm_min,
    )
    schedule = schedules["image"]
    steps = schedule.num_steps
    masks = [
        sum(BITS[branch] for branch in guidance.branches(schedule, index))
        for index in range(steps)
    ]
    assert plan.steps == steps == values["steps"]
    assert [plan.step(index)[0].mask for index in range(steps)] == masks
    assert [plan.step(index)[1] for index in range(steps)] == (
        schedule.timesteps.tolist()[:steps]
    )
    assert len(set(masks)) > 1
    with pytest.raises(ValueError):
        plan.step(steps)


def test_image_parameters_read_their_wire_fields():
    values = {
        "steps": 50,
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.5,
        "cfg_renorm_type": "global",
        "cfg_renorm_min": 0.0,
        "cfg_interval": [0.4, 1.0],
        "timestep_shift": 3.0,
        "height": 512,
        "width": 768,
    }
    assert ImageParams.from_json(values) == ImageParams(
        steps=50,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_renorm_type="global",
        cfg_renorm_min=0.0,
        # The request holds its parameters as f32.
        cfg_interval=(0.4000000059604645, 1.0),
        timestep_shift=3.0,
        height=512,
        width=768,
        seed=None,
    )
    assert ImageParams.from_json({**values, "seed": 9}).seed == 9

    with pytest.raises(ValueError, match="renormalization"):
        ImageParams.from_json({**values, "cfg_renorm_type": "channel"})
    with pytest.raises(ValueError, match="steps"):
        ImageParams.from_json(
            {name: value for name, value in values.items() if name != "steps"}
        )
