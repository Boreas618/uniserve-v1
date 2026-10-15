"""Component placements, declared calls and capability tags.

A model declares its IPC-addressable components as ``ComponentEntry`` values:
an owning module path and the ``EntryPoint`` methods serving ranks invoke.
:func:`describe_components` resolves those declarations against a model
(a meta skeleton or a loaded pipeline stage) into :class:`Call` values. The
Rust worker decides call kinds, host components and placement policy from
the reported capability tags; this module only reads the numerical model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Any

import torch
from torch import nn

from uniserve.distributed import Communicator, DeviceMesh
from uniserve.errors import SetupError
from uniserve.models import (
    AudioDecoder,
    AudioEncoder,
    CausalLM,
    ComponentEntry,
    Denoiser,
    DiffusionLM,
    Encoder,
    EntryPoint,
    ImageDecoder,
    ImageDenoiser,
    PatchEncoder,
    TextEncoder,
    VideoDecoder,
    VideoDenoiser,
    VideoEncoder,
    VideoPostprocessor,
)
from uniserve.nn.attention import (
    AttentionParallelConfig,
    ContextParallelConfig,
    Ulysses,
)
from uniserve.nn.vae.patch import PatchAutoencoder

# Capability tags the worker understands, with the class each one names.
CAPABILITIES: tuple[tuple[str, type], ...] = (
    ("causal_lm", CausalLM),
    ("diffusion_lm", DiffusionLM),
    ("denoiser", Denoiser),
    ("image_denoiser", ImageDenoiser),
    ("video_denoiser", VideoDenoiser),
    ("video_postprocessor", VideoPostprocessor),
    ("text_encoder", TextEncoder),
    ("patch_encoder", PatchEncoder),
    ("patch_autoencoder", PatchAutoencoder),
    ("video_encoder", VideoEncoder),
    ("audio_encoder", AudioEncoder),
    ("encoder", Encoder),
    ("image_decoder", ImageDecoder),
    ("video_decoder", VideoDecoder),
    ("audio_decoder", AudioDecoder),
)

#: Distinguishes an attribute a model never declared from one it cleared.
_MISSING = object()


def capability_tags(module: nn.Module) -> list[str]:
    """Every capability tag whose class ``module`` is an instance of."""
    return [tag for tag, kind in CAPABILITIES if isinstance(module, kind)]


@dataclass(frozen=True, slots=True)
class Placement:
    """One component's rank membership and logical parallel degrees.

    Mirrors the worker's ``ComponentConfig`` launch value.
    """

    ranks: tuple[int, ...]
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence: str = "local"
    ulysses_degree: int = 1
    allgather_degree: int = 1
    temporal: bool = False
    units_per_rank: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Placement:
        """Read the serialized ``ComponentConfig`` launch value."""
        parallel = dict(value.get("parallel_config") or {})
        sequence = dict(parallel.get("sequence_parallel") or {"kind": "local"})
        kind = str(sequence.get("kind", "local"))
        distribution = value.get("distribution")
        if distribution not in (None, "temporal_units"):
            raise SetupError(f"unknown component distribution {distribution!r}")
        return cls(
            ranks=tuple(int(rank) for rank in value["ranks"]),
            tensor_parallel_size=int(parallel.get("tensor_parallel_size", 1)),
            pipeline_parallel_size=int(
                parallel.get("pipeline_parallel_size", 1)
            ),
            sequence=kind,
            ulysses_degree=int(sequence.get("ulysses_degree", 1)),
            allgather_degree=int(sequence.get("allgather_degree", 1)),
            temporal=distribution is not None,
            units_per_rank=int(value.get("units_per_rank", 1)),
        )

    def dimensions(self) -> tuple[tuple[str, int], ...]:
        """Row-major rank order: pipeline, context, tensor, then Ulysses."""
        context = (
            self.allgather_degree
            if self.sequence in {"allgather", "hybrid"}
            else 1
        )
        ulysses = (
            self.ulysses_degree if self.sequence in {"ulysses", "hybrid"} else 1
        )
        return (
            ("pp", self.pipeline_parallel_size),
            ("cp", context),
            ("tp", self.tensor_parallel_size),
            ("ulysses", ulysses),
        )

    def topology(self, rank: int) -> DeviceMesh | None:
        """The numerical mesh this rank binds, or ``None`` when it binds none.

        A temporally distributed component's members each run a local
        instance on a one-rank mesh; non-members bind none.
        """
        if self.temporal:
            if rank not in self.ranks:
                return None
            ranks: tuple[int, ...] = (rank,)
        else:
            ranks = self.ranks
        dimensions = self.dimensions()
        return DeviceMesh(
            ranks=ranks,
            shape=tuple(size for _, size in dimensions),
            axes=tuple(axis for axis, _ in dimensions),
            rank=rank,
        )


def placements(value: Mapping[str, Any]) -> dict[str, Placement]:
    """Read every component placement of a launch."""
    return {
        str(name): Placement.from_mapping(dict(config))
        for name, config in value.items()
    }


def attention_parallel(sequence: str) -> AttentionParallelConfig:
    """Translate a sequence-parallel kind into attention partition axes.

    ``ulysses`` and ``hybrid`` partition attention heads; ``allgather`` and
    ``hybrid`` gather the context partition over the ``cp`` axis.
    """
    heads = Ulysses() if sequence in {"ulysses", "hybrid"} else None
    context = (
        ContextParallelConfig(gather_axis="cp")
        if sequence in {"allgather", "hybrid"}
        else None
    )
    return AttentionParallelConfig(heads=heads, context=context)


@dataclass(frozen=True, slots=True)
class Call:
    """A capability method of a resolved module and its communicators.

    ``groups`` lists the communicators the call exchanges tensors in on this
    rank; it is empty on a meta skeleton.
    """

    path: str
    module: nn.Module
    entry_point: EntryPoint
    groups: tuple[Communicator, ...] = ()

    @property
    def forward(self) -> Callable[..., Any]:
        """The bound entry-point method, resolved on the module per access."""
        return getattr(self.module, self.entry_point.method)


def describe_components(
    model: nn.Module,
    *,
    entries: Mapping[str, ComponentEntry] | None = None,
) -> dict[str, tuple[Call, ...]]:
    """Resolve each IPC entry's methods to their resident modules.

    EntryPoint methods are relative to their component, including nested
    methods such as ``conditioner.encode``.

    Args:
        model: A meta-device skeleton or a loaded pipeline stage.
        entries: The model's entry declarations; by default those of the
            ``entry_points(config)`` function of the module that defines
            ``type(model)``.

    Returns:
        Every entry name mapped to its calls in declaration order. A call
        whose module a pipeline stage cleared on this rank is left out.

    Raises:
        SetupError: A (path, method) pair is declared twice, a path names no
            module, a cleared module has no ancestor with a ``DeviceMesh`` or
            belongs to a stage that runs the method, or a method is not
            callable.
    """
    if entries is None:
        entries = import_module(type(model).__module__).entry_points(
            model.config
        )

    owners: dict[tuple[str, str], str] = {}
    calls: dict[str, list[Call]] = {name: [] for name in entries}
    for name, entry in entries.items():
        for point in entry.points:
            nested, _, method = point.method.rpartition(".")
            path = ".".join(part for part in (entry.component, nested) if part)
            key = (path, method)
            if key in owners:
                raise SetupError(f"model repeats numerical method {key}")
            owners[key] = name
            module = _entry_module(model, path)
            if module is None:
                _require_cleared_stage(model, path, point)
                continue
            if not callable(getattr(module, method, None)):
                raise SetupError(
                    f"component {path!r} has no callable {method!r}"
                )
            calls[name].append(
                Call(path, module, replace(point, method=method))
            )
    return {name: tuple(items) for name, items in calls.items()}


def _require_cleared_stage(model: nn.Module, path: str, point) -> None:
    """Excuse a cleared module only on a stage that does not run its method.

    Walks up to the nearest ancestor with a ``DeviceMesh``: a ``first`` or
    ``last`` stage method is excused on the other pipeline stages.
    """
    parent_path = path
    while parent_path:
        parent_path = parent_path.rpartition(".")[0]
        parent = model.get_submodule(parent_path)
        mesh = getattr(parent, "mesh", None)
        if isinstance(mesh, DeviceMesh):
            pipeline = mesh.get_group("pp" if "pp" in mesh.axes else ())
            if (point.stage == "first" and pipeline.rank != 0) or (
                point.stage == "last" and pipeline.rank != pipeline.size - 1
            ):
                return
            raise SetupError(f"participating component {path!r} was removed")
    raise SetupError(f"component {path!r} has no numerical module")


def _entry_module(model: nn.Module, path: str) -> nn.Module | None:
    """Resolve a declared path, or ``None`` where its attribute is cleared.

    A pipeline stage that does not participate in a submodule clears the
    attribute, which ``get_submodule`` reports like a path the model never
    declared; this tells the two apart.
    """
    try:
        return model.get_submodule(path)
    except AttributeError as error:
        parent_path, _, attribute = path.rpartition(".")
        try:
            parent = model.get_submodule(parent_path) if parent_path else model
        except AttributeError:
            parent = None
        if parent is not None and getattr(parent, attribute, _MISSING) is None:
            return None
        raise SetupError(f"entry module {path!r} does not exist") from error


def device(name: str | None) -> torch.device | None:
    """A torch device from the worker's canonical device name."""
    return None if name is None else torch.device(name)
