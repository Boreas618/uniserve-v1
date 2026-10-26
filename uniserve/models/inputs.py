"""Borrowed numerical model inputs with no request or execution ownership."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Generic, Literal, TypeVar

import torch

from uniserve.distributed import DeviceMesh
from uniserve.nn.attention import AttentionBatch
from uniserve.nn.routing import IndexedRoutes

SizeT = TypeVar("SizeT")

# The entry name used when a deployment serves one undivided model.
DEFAULT_COMPONENT = "model"


@dataclass(frozen=True, slots=True)
class EntryPoint:
    """A callable numerical method with its pipeline-stage participation.

    ``stage`` restricts which pipeline ranks execute the method; ``groups``
    names the communication axes it joins.
    """

    method: str
    stage: Literal["all", "first", "last"] = "all"
    groups: tuple[str, ...] = ()

    def communication_axes(
        self, mesh: DeviceMesh
    ) -> tuple[tuple[str, ...], ...]:
        """Resolve participation roles to the mesh's ordered axes."""
        return tuple(
            tuple(
                axis
                for axis in mesh.axes
                if axis.startswith("cp") or (role == "sp" and axis == "ulysses")
            )
            if role in {"cp", "sp"}
            else (role,)
            for role in self.groups
        )

    def __post_init__(self):
        if not self.method or self.stage not in {"all", "first", "last"}:
            raise ValueError(
                "entry points require a method and a valid pipeline "
                "participation stage"
            )
        if len(set(self.groups)) != len(self.groups) or any(
            not name for name in self.groups
        ):
            raise ValueError(
                "entry-point communication axes must be distinct nonempty names"
            )


@dataclass(frozen=True, slots=True)
class ComponentEntry:
    """One IPC-addressable component and the numerical methods it exports.

    ``component`` is the owning module path relative to the model root; the
    empty string names the model itself. ``points`` are the methods serving
    ranks may invoke, with nested methods written relative to ``component``.

    A model declares one of these per IPC entry name, so the entry name, its
    owning module and its callable methods are stated once together.
    """

    component: str
    points: tuple[EntryPoint, ...]

    def __post_init__(self):
        if not self.points:
            raise ValueError(
                "a component entry must export at least one method"
            )


@dataclass(frozen=True, slots=True)
class EmbeddingReplacement:
    """Precomputed embeddings spliced in place of token embeddings.

    Where ``mask`` is set.
    """

    values: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class TextInput:
    """Borrowed tensors for one transformer forward over packed tokens."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    attention: AttentionBatch
    embeddings: EmbeddingReplacement | None = None
    # Expert routes of a routed decoder; None takes its default route.
    routes: IndexedRoutes | None = None

    @property
    def batch_size(self) -> int:
        if self.attention.queries is None:
            return self.input_ids.shape[0] if self.input_ids.ndim == 2 else 1
        return self.attention.queries.batch_size


@dataclass(frozen=True, slots=True)
class VisionInput:
    """Per-sample images, either CHW rasters or packed patch rows with grids.

    A packed sample's grid is its patch extent: ``(height, width)``, or
    ``(time, height, width)`` for encoders whose patches also span frames.
    ``grids`` holds it as a ``[1, axes]`` device tensor and ``grid_shapes``
    as host integers; rasters derive both from their pixels.
    """

    images: tuple[torch.Tensor, ...]
    grids: tuple[torch.Tensor | None, ...]
    grid_shapes: tuple[tuple[int, ...] | None, ...]

    def __post_init__(self):
        if len(self.images) != len(self.grids) or len(self.images) != len(
            self.grid_shapes
        ):
            raise ValueError(
                "vision tensors and grid descriptions must align by sample"
            )

    @property
    def batch_size(self) -> int:
        return len(self.images)


@dataclass(frozen=True, slots=True)
class LatentInput:
    """One sample's latent tensor and its diffusion timestep."""

    tensor: torch.Tensor
    timestep: torch.Tensor


@dataclass(frozen=True)
class DenoiserInput(Generic[SizeT]):
    """Per-modality latents aligned with sample sizes at one solver step.

    ``step`` names the solver step as device data: a [1] int64 tensor holding
    the evaluation index (``Schedule.step``). Per-step tables (schedule
    endpoints, step-indexed model products) are gathered through it, so one
    captured computation evaluates every step once the value is copied in.
    """

    latents: Mapping[str, tuple[LatentInput, ...]]
    sizes: tuple[SizeT, ...]
    step: torch.Tensor

    def __post_init__(self):
        if (
            not isinstance(self.step, torch.Tensor)
            or self.step.dtype != torch.int64
            or tuple(self.step.shape) != (1,)
        ):
            raise ValueError(
                "denoiser step must be a [1] int64 tensor of its evaluation"
            )
        if any(
            not name or len(values) != len(self.sizes)
            for name, values in self.latents.items()
        ):
            raise ValueError(
                "latent modalities must align with the numerical sample sizes"
            )
        object.__setattr__(
            self, "latents", MappingProxyType(dict(self.latents))
        )

    @property
    def batch_size(self) -> int:
        return len(self.sizes)


class TokenKind(IntEnum):
    """What one token of a packed image forward holds."""

    # Padding: reads no sample and predicts nothing.
    PAD = 0
    # The opening framing token of an image sequence.
    OPEN = 1
    # One latent patch row of an image, in raster order.
    LATENT = 2
    # The closing framing token of an image sequence.
    CLOSE = 3


@dataclass(frozen=True, slots=True)
class PackedImageRows:
    """Per-token metadata of a packed image denoiser forward.

    A forward packs ``R`` *rows*, each one image's framed sequence (one
    guidance branch of a denoising step, or one latent image written as
    conditioning), into ``T`` tokens; rows past the live ones are empty or
    hold padding tokens. Every field is device data at a static shape, so
    one captured computation serves any rows whose tokens fit ``T``; the
    network reads nothing of the rows on the host.

    Attributes:
        sample_index: ``int64 [T]`` the source row each latent token reads,
            -1 for every other token.
        kind: ``int64 [T]`` each token's ``TokenKind``.
        positions: ``int64 [3, T]`` rotary coordinates (temporal, row,
            column): a latent token's raster coordinates in its image's
            latent grid; zero spatial coordinates elsewhere.
        timesteps: ``float32 [T]`` the network time of each token's row; zero
            on padding.
        token_row: ``int64 [T]`` the row each token belongs to.
        row_grid: ``int64 [R, 2]`` each row's latent grid ``(rows,
            columns)``; ``(0, 0)`` for a row without latent tokens.
        noise_scale: ``float32 [R]`` the scale of each row's initial noise
            draw, which a network may embed; zero for rows without latents.
        regions: For a network with a computation that is not row-local
            (an eager per-image region, such as a convolution over an
            image's token grid), the host ``(first latent token, grid rows,
            grid columns)`` of each row holding latents, in row order;
            ``None`` where the rows are device data alone.
    """

    sample_index: torch.Tensor
    kind: torch.Tensor
    positions: torch.Tensor
    timesteps: torch.Tensor
    token_row: torch.Tensor
    row_grid: torch.Tensor
    noise_scale: torch.Tensor
    regions: tuple[tuple[int, int, int], ...] | None = None

    def __post_init__(self) -> None:
        tokens, rows = self.kind.shape[0], self.row_grid.shape[0]
        if (
            any(
                value.shape != (tokens,)
                for value in (
                    self.sample_index,
                    self.kind,
                    self.timesteps,
                    self.token_row,
                )
            )
            or self.positions.shape != (3, tokens)
            or self.row_grid.shape != (rows, 2)
            or self.noise_scale.shape != (rows,)
        ):
            raise ValueError(
                "packed image rows require [T] token columns, [3, T] "
                "positions and [R] row columns"
            )
        if any(
            value.dtype != torch.int64
            for value in (
                self.sample_index,
                self.kind,
                self.positions,
                self.token_row,
                self.row_grid,
            )
        ) or any(
            value.dtype != torch.float32
            for value in (self.timesteps, self.noise_scale)
        ):
            raise ValueError(
                "packed image rows hold int64 indices and float32 times"
            )
        if self.regions is not None and any(
            len(region) != 3 or any(type(value) is not int for value in region)
            for region in self.regions
        ):
            raise ValueError(
                "image regions are (first token, rows, columns) integers"
            )

    @property
    def tokens(self) -> int:
        """The static token count ``T``."""
        return self.kind.shape[0]

    @property
    def rows(self) -> int:
        """The static row count ``R``."""
        return self.row_grid.shape[0]


@dataclass(frozen=True, slots=True)
class PackedImageInput:
    """Borrowed tensors of one packed image denoiser forward.

    Attributes:
        source: ``[U, width]`` canonical latent patch rows that latent tokens
            read through ``rows.sample_index``: the step workspace of a
            denoising step, or the latents written as conditioning.
        rows: The forward's per-token metadata.
        attention: Attention metadata over the rows' packed sequences.
    """

    source: torch.Tensor
    rows: PackedImageRows
    attention: AttentionBatch

    def __post_init__(self) -> None:
        if self.source.ndim != 2:
            raise ValueError(
                "a packed image input reads [U, width] latent rows"
            )
        # Query lengths are compared through their optional host mirror
        # only, so validation never reads device lengths.
        queries = self.attention.queries
        if (
            queries is not None
            and queries.num_tokens is not None
            and queries.num_tokens != self.rows.tokens
        ):
            raise ValueError(
                "image attention queries must cover the packed tokens"
            )


@dataclass(frozen=True, slots=True)
class TextSize:
    """Token and sequence counts describing a text workload."""

    num_tokens: int
    batch_size: int

    def __post_init__(self):
        if any(
            type(value) is not int or value < 0
            for value in (self.num_tokens, self.batch_size)
        ):
            raise ValueError(
                "text sizes require nonnegative token and sequence counts"
            )
