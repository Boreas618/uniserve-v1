"""Image trajectories of admitted requests and their guidance prefixes.

An image request denoises one trajectory per generated image. When its
latent is prepared, its batch opens an :class:`ImageTrajectory`: the
:class:`GuidancePlan` the model's schedules and guidance select per solver
step, made once per trajectory and kept with the schedules and guidance the
solver update combines predictions with, and the request's guidance
prefixes.

Each classifier-free guidance branch (``uniserve.diffusion.Branch``)
attends to a cache prefix: the conditioned branch to the request's own
context; the unconditional branches to prefixes the scheduler placed in
other request slots, which the executor prefills the first time a branch is
evaluated. A prefix's tokens are resolved once per source
(``uniserve.processing.BranchSource``) when the trajectory opens and kept
with it: the conditioning source reads the request's own context, the
negative source's are the request's admitted negative tokens, which the
engine sized that prefix's cache from (:func:`admitted_prefix`), and the
model frames the others (``denoiser.framed_prefix``, its prompt
framing and tokenizer). A prefix's physical coordinates are resolved again
for every denoising call (:meth:`ImageTrajectory.begin_call`), because the
scheduler may place a later interval's prefixes in other slots.

:class:`Trajectories` keeps the open trajectory of each request slot; a
reused slot never sees an earlier request's trajectory.
"""

from __future__ import annotations

import operator
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, cast, get_args

import numpy as np

from uniserve.diffusion import Branch, Guidance, Renorm, Schedule
from uniserve.models import ImageDenoiser
from uniserve.processing import BranchSource

__all__ = [
    "BranchRow",
    "BranchSet",
    "Coordinates",
    "ForwardSlot",
    "GuidancePlan",
    "ImageParams",
    "ImageTrajectory",
    "Prefix",
    "PrefixFill",
    "RenormName",
    "Trajectories",
    "admitted_prefix",
    "branch_source",
]

# The mask bit of each guidance branch. Bits ascend in evaluation order,
# which is the declaration order of ``Branch``: conditioned first.
_BITS = {
    Branch.CONDITIONED: 1,
    Branch.TEXT_UNCONDITIONAL: 2,
    Branch.IMAGE_UNCONDITIONAL: 4,
}

# The bit every branch set holds, and every bit a branch set may hold.
_CONDITIONED = _BITS[Branch.CONDITIONED]
_ALL_BITS = sum(_BITS.values())

# The schedule an image trajectory steps along (``make_schedules`` keys its
# schedules by latent modality).
_IMAGE = "image"

# A request's guidance renormalization names (``CfgRenorm`` on the wire).
RenormName = Literal["none", "global", "text_channel"]


@dataclass(frozen=True, slots=True)
class BranchSet:
    """The guidance branches one denoising step evaluates.

    Iterating yields the branches in evaluation order, conditioned first.

    Attributes:
        mask: Bit 0 conditioned, bit 1 text-unconditional, bit 2
            image-unconditional.

    Raises:
        ValueError: ``mask`` omits the conditioned branch, which every step
            evaluates, or holds a bit naming no branch.
    """

    mask: int

    def __post_init__(self) -> None:
        if type(self.mask) is not int or not 0 <= self.mask <= _ALL_BITS:
            raise ValueError(
                f"guidance branch mask {self.mask!r} names an unknown branch"
            )
        if not self.mask & _CONDITIONED:
            raise ValueError(
                f"guidance branch mask {self.mask:#b} omits the conditioned "
                "branch"
            )

    @classmethod
    def of(cls, branches: Iterable[Branch]) -> BranchSet:
        """The set of ``branches`` (``Guidance.branches`` of a step).

        Raises:
            ValueError: ``branches`` omits the conditioned branch.
        """
        mask = 0
        for branch in branches:
            mask |= _BITS[branch]
        return cls(mask)

    def __contains__(self, branch: object) -> bool:
        return isinstance(branch, Branch) and bool(self.mask & _BITS[branch])

    def __iter__(self) -> Iterator[Branch]:
        return (branch for branch, bit in _BITS.items() if self.mask & bit)

    def __len__(self) -> int:
        return self.mask.bit_count()


def branch_source(
    branch: Branch, image_unconditional: BranchSource
) -> BranchSource:
    """The prefix source ``branch`` conditions on.

    ``image_unconditional`` is the denoiser's source for its
    image-unconditional branch (``ImageLayout.image_unconditional``).
    """
    if branch is Branch.CONDITIONED:
        return BranchSource.CONDITIONING
    if branch is Branch.TEXT_UNCONDITIONAL:
        return BranchSource.NEGATIVE_OR_START
    return image_unconditional


@dataclass(frozen=True, slots=True)
class Prefix:
    """A guidance prefix: its tokens, or the request's own context.

    Attributes:
        tokens: The prefix's token ids; empty for a branch reusing the
            conditioning context.
        reuses_conditioning: The branch attends to the request's own
            conditioning prefix cache rather than a prefix of its own.
    """

    tokens: tuple[int, ...]
    reuses_conditioning: bool


def admitted_prefix(
    source: BranchSource, negative: Sequence[int]
) -> Prefix | None:
    """The prefix a branch of ``source`` conditions on that admission fixes.

    The conditioning source reads the request's own context, where the
    engine places the conditioned branch's row. The negative source
    conditions on exactly the request's admitted negative tokens
    ``negative``, also when none were admitted: they are the request's final
    negative conditioning tokens, and the engine sizes the prefix's cache and
    its prefill from their count, so a prefix the rank framed itself would
    not fit the slot the engine placed it in.

    Returns:
        The prefix, or None for a source the model frames (``START``).
    """
    if source is BranchSource.CONDITIONING:
        return Prefix(tokens=(), reuses_conditioning=True)
    if source is BranchSource.NEGATIVE_OR_START:
        return Prefix(
            tokens=tuple(int(token) for token in negative),
            reuses_conditioning=False,
        )
    return None


@dataclass(frozen=True, slots=True)
class Coordinates:
    """Cache coordinates of a request slot.

    Attributes:
        slot: The request slot.
        visible: Tokens of the slot's cache the forward reads.
        capacity: Tokens the slot's cache has allocated.
    """

    slot: int
    visible: int
    capacity: int


@dataclass(frozen=True, slots=True)
class ForwardSlot:
    """One scheduler forward descriptor of a denoising call.

    Attributes:
        slot: The request slot the row addresses.
        seq_len: The row's sequence length: its context and query tokens.
        query_len: The row's query tokens.
    """

    slot: int
    seq_len: int
    query_len: int


@dataclass(frozen=True, slots=True)
class PrefixFill:
    """A prefix prefill submitted before a step's denoising rows.

    ``tokens`` are written causally as prefix cache from position zero of
    ``slot``; the prefill's hidden states are discarded.
    """

    slot: int
    tokens: tuple[int, ...]


class BranchRow(NamedTuple):
    """One denoising row of a guidance branch.

    Attributes:
        branch: The guidance branch.
        coordinates: The branch prefix's slot and visible tokens.
        temporal: The temporal position of the row's image sequence.
    """

    branch: Branch
    coordinates: Coordinates
    temporal: int


def _integer(value: Any) -> int:
    # A wire integer: a float or a string is refused rather than truncated.
    return operator.index(value)


def _f32(value: Any) -> float:
    # The request carries its parameters as f32; JSON states each as its
    # shortest decimal, which rounds back to the request's exact value.
    return float(np.float32(float(value)))


@dataclass(frozen=True, slots=True)
class ImageParams:
    """An image request's generation parameters (``ipc::ImageParams``).

    Attributes:
        steps: Denoising steps.
        cfg_text_scale: Classifier-free guidance scale of text conditioning.
        cfg_img_scale: Classifier-free guidance scale of image conditioning.
        cfg_renorm_type: Guidance renormalization.
        cfg_renorm_min: Lower bound of guidance renormalization.
        cfg_interval: Inclusive ``(low, high)`` schedule coordinates where
            guidance is active.
        timestep_shift: Shift of the timestep schedule; a non-positive
            shift keeps the denoiser's default.
        height: Output height in pixels.
        width: Output width in pixels.
        seed: Diffusion seed, or None.

    Raises:
        ValueError: ``cfg_renorm_type`` names no renormalization a request
            may choose.
    """

    steps: int
    cfg_text_scale: float
    cfg_img_scale: float
    cfg_renorm_type: RenormName
    cfg_renorm_min: float
    cfg_interval: tuple[float, float]
    timestep_shift: float
    height: int
    width: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.cfg_renorm_type not in get_args(RenormName):
            raise ValueError(
                f"unsupported CFG renormalization mode {self.cfg_renorm_type!r}"
            )

    @property
    def draw_seed(self) -> int:
        """The seed the request's noise draws are keyed by; 0 without one."""
        return 0 if self.seed is None else self.seed

    @classmethod
    def from_json(cls, values: Mapping[str, Any]) -> ImageParams:
        """The parameters of a request's JSON image parameters.

        Reads the wire field names; a missing ``seed`` is None.

        Raises:
            ValueError: A field is missing or the interval is not two
                values.
        """
        try:
            low, high = values["cfg_interval"]
            seed = values.get("seed")
            return cls(
                steps=_integer(values["steps"]),
                cfg_text_scale=_f32(values["cfg_text_scale"]),
                cfg_img_scale=_f32(values["cfg_img_scale"]),
                cfg_renorm_type=cast(
                    RenormName, str(values["cfg_renorm_type"])
                ),
                cfg_renorm_min=_f32(values["cfg_renorm_min"]),
                cfg_interval=(_f32(low), _f32(high)),
                timestep_shift=_f32(values["timestep_shift"]),
                height=_integer(values["height"]),
                width=_integer(values["width"]),
                seed=None if seed is None else _integer(seed),
            )
        except KeyError as error:
            raise ValueError(
                f"image parameters lack the field {error.args[0]!r}"
            ) from None
        except (TypeError, ValueError) as error:
            raise ValueError(f"malformed image parameters: {error}") from None


@dataclass(frozen=True, slots=True)
class GuidancePlan:
    """The guidance plan of one trajectory.

    Each solver step's model timestep and the branches it evaluates, with
    the schedules and guidance the solver update combines that step's
    predictions with. A plan made by :meth:`make` keeps both; a plan stated
    directly may omit them.

    Attributes:
        timesteps: The model timestep of each solver step.
        branches: The branches each solver step evaluates.
        schedules: The trajectory's schedules by latent modality, on the
            CPU, or None.
        guidance: The trajectory's guidance, or None.

    Raises:
        ValueError: ``timesteps`` and ``branches`` cover different steps.
    """

    timesteps: tuple[float, ...]
    branches: tuple[BranchSet, ...]
    schedules: Mapping[str, Schedule] | None = field(
        default=None, compare=False, repr=False
    )
    guidance: Guidance | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if len(self.timesteps) != len(self.branches):
            raise ValueError(
                "a guidance plan states one timestep and one branch set per "
                "solver step"
            )

    @classmethod
    def make(cls, network: ImageDenoiser, params: ImageParams) -> GuidancePlan:
        """The plan of a trajectory of ``params`` on ``network``.

        The network's schedules are made on the CPU, a non-positive
        ``timestep_shift`` keeping its default shift, and its guidance from
        the request's scales, interval and renormalization; each step
        evaluates the branches the guidance selects at the image schedule's
        coordinate.

        Raises:
            ValueError: The network makes no image schedule, or rejects the
                parameters.
        """
        shift = params.timestep_shift
        schedules = dict(
            network.make_schedules(
                params.steps,
                shift=shift if shift > 0 else None,
                device="cpu",
            )
        )
        guidance = network.make_guidance(
            text_scale=params.cfg_text_scale,
            image_scale=params.cfg_img_scale,
            interval=params.cfg_interval,
            renorm=Renorm(params.cfg_renorm_type),
            renorm_min=params.cfg_renorm_min,
        )

        schedule = schedules.get(_IMAGE)
        if schedule is None:
            raise ValueError("the image denoiser makes no image schedule")
        steps = schedule.num_steps
        # A schedule holds one more timestep than it has steps: the endpoint
        # the last step reaches, which no step evaluates at.
        timesteps = schedule.timesteps.tolist()[:steps]
        return cls(
            timesteps=tuple(float(time) for time in timesteps),
            branches=tuple(
                BranchSet.of(guidance.branches(schedule, index))
                for index in range(steps)
            ),
            schedules=MappingProxyType(schedules),
            guidance=guidance,
        )

    @property
    def steps(self) -> int:
        """The trajectory's solver steps."""
        return len(self.branches)

    def step(self, index: int) -> tuple[BranchSet, float]:
        """The branches and model timestep of solver step ``index``.

        Raises:
            ValueError: ``index`` lies outside the schedule.
        """
        if not 0 <= index < self.steps:
            raise ValueError(
                f"denoising step {index} is outside its {self.steps}-step "
                "schedule"
            )
        return self.branches[index], self.timesteps[index]


class ImageTrajectory:
    """One image request's open trajectory.

    Attributes:
        key: The request's identity; any hashable value.
        size: Image ``(height, width)`` in pixels.
        plan: The trajectory's guidance plan.
    """

    __slots__ = (
        "_branches",
        "_conditioning",
        "_prefixes",
        "key",
        "plan",
        "size",
    )

    def __init__(
        self,
        key: Hashable,
        size: tuple[int, int],
        plan: GuidancePlan,
        prefixes: Mapping[BranchSource, Prefix],
    ) -> None:
        """Open a trajectory of ``plan``.

        Args:
            key: The request's identity.
            size: Image ``(height, width)`` in pixels.
            plan: The trajectory's guidance plan.
            prefixes: The prefix of each source a guidance branch may
                condition on, resolved when the trajectory opens.
        """
        self.key = key
        self.size = size
        self.plan = plan
        self._prefixes = dict(prefixes)
        # The request's own conditioning coordinates for the current call.
        self._conditioning = Coordinates(slot=0, visible=0, capacity=0)
        # Each branch's prefix coordinates, resolved in the current call.
        self._branches: dict[Branch, Coordinates] = {}

    def reuses_conditioning(
        self, branch: Branch, image_unconditional: BranchSource
    ) -> bool:
        """Whether ``branch`` attends to the request's own conditioning.

        ``image_unconditional`` is the denoiser's source of its
        image-unconditional branch.

        Raises:
            ValueError: The trajectory was opened without the branch's
                prefix source.
        """
        return self._prefix(branch, image_unconditional).reuses_conditioning

    def _prefix(
        self, branch: Branch, image_unconditional: BranchSource
    ) -> Prefix:
        """The prefix ``branch`` conditions on, resolved at opening."""
        source = branch_source(branch, image_unconditional)
        prefix = self._prefixes.get(source)
        if prefix is None:
            raise ValueError(
                "an image trajectory was opened without its "
                f"{source.value} prefix"
            )
        return prefix

    def begin_call(self, conditioning: Coordinates) -> None:
        """Start a denoising call at the request's ``conditioning``.

        Every branch's coordinates resolve again in the call.
        """
        self._conditioning = conditioning
        self._branches.clear()

    def resolve_branches(
        self,
        branches: BranchSet,
        image_unconditional: BranchSource,
        descriptors: Sequence[ForwardSlot],
        coordinates: Callable[[int, int], Coordinates],
    ) -> list[PrefixFill]:
        """Resolve the prefix coordinates of ``branches`` for this call.

        Branches already resolved in the call keep their coordinates.

        Args:
            branches: The branches the step evaluates.
            image_unconditional: The denoiser's source of its
                image-unconditional branch.
            descriptors: The call's forward descriptors: optional prefix
                prefills, then one denoising descriptor per branch of
                ``branches`` in evaluation order.
            coordinates: The allocated coordinates of a slot with the given
                visible tokens, as ``coordinates(slot, visible)``.

        Returns:
            The prefix prefills to submit before the denoising rows, one per
            physical prefix: branches sharing a slot share its prefill.

        Raises:
            ValueError: The descriptors are fewer than the branches, a
                prefix exceeds its slot's capacity, a resident prefix's
                length disagrees with its tokens, or the trajectory was
                opened without a branch's prefix source.
        """
        count = len(branches)
        if len(descriptors) < count:
            raise ValueError(
                "a denoising call has fewer forward rows than guidance branches"
            )
        split = len(descriptors) - count
        prefills, denoising = descriptors[:split], descriptors[split:]

        fills: list[PrefixFill] = []
        filled: set[int] = set()
        for index, branch in enumerate(branches):
            if branch in self._branches:
                continue
            prefix = self._prefix(branch, image_unconditional)

            if prefix.reuses_conditioning:
                located = self._conditioning
                length = located.visible
            else:
                # A non-reusing branch's prefix lives in its denoising
                # descriptor's slot. It is empty when the call also carries
                # a prefill of exactly the prefix's tokens into that slot;
                # otherwise it is resident, and the descriptor's context is
                # its visible length.
                descriptor = denoising[index]
                length = len(prefix.tokens)
                prefilled = any(
                    row.slot == descriptor.slot
                    and row.seq_len == row.query_len == length
                    for row in prefills
                )
                visible = (
                    0
                    if prefilled
                    else max(descriptor.seq_len - descriptor.query_len, 0)
                )
                located = coordinates(descriptor.slot, visible)

            if length > located.capacity:
                raise ValueError(
                    "a guidance prefix exceeds its slot's allocated tokens"
                )
            if located.visible not in (0, length):
                raise ValueError(
                    "a resident guidance prefix's length disagrees with its "
                    "tokens"
                )
            self._branches[branch] = located
            if (
                located.visible == 0
                and prefix.tokens
                and located.slot not in filled
            ):
                filled.add(located.slot)
                fills.append(PrefixFill(located.slot, prefix.tokens))
        return fills

    def prefilled(self, slot: int, count: int) -> None:
        """Record that a prefix prefill wrote ``count`` tokens of ``slot``.

        The branches whose empty prefix lives in ``slot`` now read them.
        """
        for branch, located in self._branches.items():
            if located.slot == slot and located.visible == 0:
                self._branches[branch] = replace(located, visible=count)

    def branch_rows(
        self, branches: BranchSet, logical_position: int
    ) -> list[BranchRow]:
        """The denoising rows of ``branches``, in evaluation order.

        The conditioned branch's image continues the request at its
        ``logical_position``; an unconditional branch's image follows its
        prefix.

        Raises:
            ValueError: A branch's prefix was not resolved in this call.
        """
        rows = []
        for branch in branches:
            located = self._branches.get(branch)
            if located is None:
                raise ValueError("guidance branch has no cache prefix")
            temporal = (
                logical_position
                if branch is Branch.CONDITIONED
                else located.visible
            )
            rows.append(BranchRow(branch, located, temporal))
        return rows


class Trajectories:
    """The open trajectory of each request slot."""

    __slots__ = ("_slots",)

    def __init__(self) -> None:
        self._slots: dict[int, ImageTrajectory] = {}

    def get(self, slot: int, key: Hashable) -> ImageTrajectory | None:
        """The open trajectory of the request ``key`` at ``slot``.

        None when the slot holds no trajectory or holds an earlier
        request's: a reused slot never sees another request's trajectory.
        """
        trajectory = self._slots.get(slot)
        if trajectory is None or trajectory.key != key:
            return None
        return trajectory

    def open(self, slot: int, trajectory: ImageTrajectory) -> ImageTrajectory:
        """Open ``trajectory`` at ``slot``, replacing the slot's last one."""
        self._slots[slot] = trajectory
        return trajectory

    def release(self, slots: Iterable[int]) -> None:
        """Forget the trajectories of ``slots``.

        Called for slots whose requests retired or restarted, and for slots
        a batch admits, so a reused slot never continues an earlier
        request's trajectory.
        """
        for slot in slots:
            self._slots.pop(slot, None)
