"""Reading a checkpoint's configuration and loading a rank's model.

A rank loads a model in two calls around process-group creation, each
reached by import path with explicit arguments:

1. :func:`read_config` reads the checkpoint configuration and builds a meta
   skeleton, before any collective exists. The rank reads the declared
   components, their calls, capability tags and communication axes from the
   returned :class:`Description`, one scalar or flat list per method.
2. :func:`load_model` narrows the checkpoint to this rank's modules and the
   launch's weight choices and materializes them on the meshes the rank
   created.

The objects returned here are opaque to the rank and passed back unchanged.
What the rank reports of a loaded model it reads from the functions below
:func:`load_model`; a fact of one capability belongs to that capability's
runner.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import torch
from torch import nn

from uniserve import loading as library
from uniserve.distributed import Communicator, communication_axes
from uniserve.errors import SetupError
from uniserve.loading.precision import weight_config
from uniserve.models import (
    CausalLM,
    Denoiser,
    ImageDecoder,
    VideoDecoder,
    VideoPostprocessor,
    registry,
)
from uniserve.models.components import (
    Call,
    Placement,
    attention_parallel,
    capability_tags,
    describe_components,
    placements,
)
from uniserve.nn.moe import FusedMoE
from uniserve.nn.vae.patch import PatchAutoencoder

logger = logging.getLogger(__name__)

# A capability class a model's module is looked up by.
_Kind = TypeVar("_Kind")


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


@dataclass
class Description:
    """A checkpoint's configuration and meta skeleton before loading.

    The rank reads its declarations through the methods: each returns a
    scalar or one flat list. A method over a component returns one value per
    declared call of that component, in declared order; ``capabilities`` and
    ``selection_axes`` concatenate their per-call and per-selection lists,
    with the lengths in ``capability_counts`` and ``selection_sizes``.

    ``metadata`` is ``None`` for the stub model.
    """

    stub: bool
    model: nn.Module
    metadata: registry.Config | None
    path: str | None
    base: str | None
    io: library.Config
    entries: Mapping[str, Any]
    calls: dict[str, tuple[Call, ...]]
    # Axis selections each component's calls communicate over on the reading
    # rank's mesh; empty for a component the rank does not place.
    selections: Mapping[str, tuple[tuple[str, ...], ...]]
    # Module paths of the routed experts a split deployment places apart.
    expert_paths: tuple[str, ...]

    def model_name(self) -> str:
        """``module.QualName`` of the model class."""
        return _name(self.model)

    def components(self) -> list[str]:
        """The declared components, in declared order."""
        return list(self.calls)

    def call_paths(self, component: str) -> list[str]:
        """Each call's module path within the model; empty for the root."""
        return [call.path for call in self.calls[str(component)]]

    def call_methods(self, component: str) -> list[str]:
        """Each call's method name on its module."""
        return [call.entry_point.method for call in self.calls[str(component)]]

    def call_stages(self, component: str) -> list[str]:
        """The pipeline stages running each call.

        ``"all"``, ``"first"`` or ``"last"``.
        """
        return [call.entry_point.stage for call in self.calls[str(component)]]

    def capability_counts(self, component: str) -> list[int]:
        """How many of ``capabilities`` belong to each call."""
        return [
            len(capability_tags(call.module))
            for call in self.calls[str(component)]
        ]

    def capabilities(self, component: str) -> list[str]:
        """Every call's capability tags, concatenated in call order."""
        return [
            tag
            for call in self.calls[str(component)]
            for tag in capability_tags(call.module)
        ]

    def selection_sizes(self, component: str) -> list[int]:
        """The axis count of each selection the component communicates over."""
        return [len(axes) for axes in self.selections[str(component)]]

    def selection_axes(self, component: str) -> list[str]:
        """Every selection's mesh axes, concatenated in selection order."""
        return [
            axis for axes in self.selections[str(component)] for axis in axes
        ]

    def routed_experts(self) -> list[str]:
        """Module paths of the routed expert layers placed on dedicated ranks.

        Empty unless routed experts run apart from attention.
        """
        return list(self.expert_paths)

    def decodes_video(self) -> bool:
        """Whether the model reconstructs video."""
        return any(
            isinstance(module, VideoDecoder) for module in self.model.modules()
        )


@dataclass
class Resolved:
    """A checkpoint narrowed to this rank's modules, with its weights."""

    source: registry.Config
    description: Description


@dataclass
class Loaded:
    """A rank's loaded model and what serving needs from it."""

    module: nn.Module
    description: Description
    calls: dict[str, tuple[Call, ...]] = field(default_factory=dict)
    attention: Any = None
    attention_backend: str = "auto"
    image_processor: Any = None
    flow_prompt: Any = None
    tokenizer: Path | None = None

    @property
    def model(self) -> nn.Module:
        return self.module


def read_config(
    stub: bool,
    path: str | None,
    base_model: str | None,
    io_format: str,
    io_mode: str,
    download_dir: str | None,
    num_threads: int | None,
    checksum_manifest: str | None,
    device: str,
    generation_device: str | None,
    components: str,
    rank: int,
    role: str,
    expert_split: bool,
) -> Description:
    """Read a checkpoint's configuration and this rank's view of it.

    Args:
        stub: Describe the stub model instead of a checkpoint.
        path: The checkpoint directory.
        base_model: The base checkpoint an adapter refers to.
        io_format, io_mode, download_dir, num_threads, checksum_manifest:
            How weights are read (``uniserve.loading.Config``).
        device: The rank's primary device.
        generation_device: The device of a generation tower placed apart.
        components: The launch's component placement as JSON object text:
            each component's ranks and parallel configuration.
        rank: This rank.
        role: ``"model"`` or ``"experts"``.
        expert_split: Routed experts run on dedicated ranks.

    Raises:
        SetupError: The placement does not fit the declarations, a
            generation device has no flow route to place, or a split expert
            deployment has no routed expert layer.
    """
    stub = bool(stub)
    # The caller sends plain strings; ``Config`` rejects unknown values.
    io = library.Config(
        format=cast(Literal["auto", "safetensors", "pt"], str(io_format)),
        mode=cast(Literal["eager", "layered", "dummy"], str(io_mode)),
        download_dir=None if download_dir is None else Path(str(download_dir)),
        num_threads=None if num_threads is None else int(num_threads),
        checksum_manifest=None
        if checksum_manifest is None
        else Path(str(checksum_manifest)),
    )
    path, base = _text(path), _text(base_model)
    rank, role = int(rank), str(role)
    placed = placements(json.loads(str(components)))
    if stub:
        from uniserve.models.stub import Model, entry_points

        with torch.device("meta"):
            model = Model()
        metadata = None
        entries = entry_points(model.config)
    else:
        if path is None:
            raise SetupError("--model is required for a model worker")
        metadata = registry.read_config(
            path, io=io, modules=frozenset(), base=base
        )
        with torch.device("meta"):
            model = metadata.model_class(metadata.model)
        entries = metadata.entry_points

    calls = (
        {} if role == "experts" else describe_components(model, entries=entries)
    )
    generation = _text(generation_device)
    if generation is not None:
        _devices(model, generation)

    expert_paths: list[str] = []
    if expert_split:
        expert_paths = [
            name
            for name, module in model.named_modules()
            if isinstance(module, FusedMoE)
        ]
        if not expert_paths:
            raise SetupError(
                "disaggregated workers require routed expert layers"
            )

    selections = {}
    for name, items in calls.items():
        placement = placed.get(name)
        topology = None if placement is None else placement.topology(rank)
        selections[name] = tuple(
            sorted(
                {
                    axes
                    for call in items
                    for axes in _axes(call, placement, topology)
                }
            )
        )
    return Description(
        stub,
        model,
        metadata,
        path,
        base,
        io,
        entries,
        calls,
        selections,
        tuple(expert_paths),
    )


def _axes(
    call: Call, placement: Placement | None, topology
) -> set[tuple[str, ...]]:
    """Axis selections a call communicates over on this rank's mesh."""
    if placement is None or topology is None:
        return set()
    axes = set(
        communication_axes(
            call.module,
            topology,
            attention=attention_parallel(placement.sequence),
        )
    )
    axes |= set(call.entry_point.communication_axes(topology))
    return {tuple(str(axis) for axis in selection) for selection in axes}


def _resolve(
    description: Description,
    modules: Sequence[str],
    exclude_modules: Sequence[str],
    quantization: str,
    model_dtype: str,
) -> Description | Resolved:
    """Narrow the checkpoint to this rank's modules and weight choices.

    The stub model needs no checkpoint and is returned unchanged.
    """
    if description.stub:
        return description
    # ``read_config`` requires a checkpoint path of every model but the stub.
    assert description.path is not None
    source = registry.read_config(
        description.path,
        io=description.io,
        modules=frozenset(str(path) for path in modules),
        base=description.base,
        exclude_modules=frozenset(str(path) for path in exclude_modules),
    )
    options = json.loads(str(quantization))
    weights = weight_config(source, options, str(model_dtype))
    return Resolved(replace(source, weights=weights), description)


def load_model(
    description: Description,
    modules: Sequence[str],
    exclude_modules: Sequence[str],
    quantization: str,
    model_dtype: str,
    device: str,
    generation_device: str | None,
    role: str,
    host_only: bool,
    mesh_components: Sequence[str],
    mesh_path_counts: Sequence[int],
    mesh_paths: Sequence[str],
    meshes: Sequence[Any],
    mesh_units: Sequence[Any],
    mesh_sequences: Sequence[str],
    expert_world: Any,
    attention_ranks: int,
    expert_dedicated: bool,
    attention_backend: str,
    workspace_size: int,
    decode_backend: str,
    prefill_backend: str,
    disable_split_kv: bool,
    decode_split_tile_size: int | None,
    prefill_split_tile_size: int | None,
    use_tensor_core: bool | None,
) -> Loaded:
    """Materialize this rank's modules of a described model on its meshes.

    The stub model is built directly on the device and serves with the
    portable ``torch`` attention backend. A host-only rank keeps the
    weight-less description. Otherwise each listed outermost path loads on
    its component's mesh with that component's attention partition.

    Args:
        description: The description :func:`read_config` returned.
        modules: Module paths resident on this rank.
        exclude_modules: Module paths left unloaded below them.
        quantization: The launch's precision selectors as JSON object text.
        model_dtype: The compute element type's name.
        device, generation_device, role: As in :func:`read_config`.
        host_only: The rank holds only host components.
        mesh_components: One entry per component with a mesh.
        mesh_path_counts: How many of ``mesh_paths`` each component has.
        mesh_paths: Every component's outermost module paths, back to back.
        meshes: Each component's device mesh.
        mesh_units: Each component's unit mesh, or ``None``.
        mesh_sequences: Each component's sequence-parallel kind.
        expert_world: The torch world holding the expert union's
            communicator; ``None`` without an expert union.
        attention_ranks: Attention ranks of a disaggregated union.
        expert_dedicated: This rank holds only routed experts.
        attention_backend: The attention backend's name.
        workspace_size, decode_backend, prefill_backend, disable_split_kv,
        decode_split_tile_size, prefill_split_tile_size, use_tensor_core:
            FlashInfer options (``flashinfer.Config``); ``None`` keeps an
            option's default.
    """
    resolved = _resolve(
        description, modules, exclude_modules, quantization, model_dtype
    )
    device = str(device)
    generation = _text(generation_device)
    dedicated = str(role) == "experts"
    host_only = bool(host_only)
    backend = str(attention_backend)
    options: dict[str, Any] = {
        name: value
        for name, value in (
            ("workspace_size", int(workspace_size)),
            ("decode_backend", str(decode_backend)),
            ("prefill_backend", str(prefill_backend)),
            ("disable_split_kv", bool(disable_split_kv)),
            ("decode_split_tile_size", decode_split_tile_size),
            ("prefill_split_tile_size", prefill_split_tile_size),
            ("use_tensor_core", use_tensor_core),
        )
        if value is not None
    }

    if isinstance(resolved, Description):
        description, source = resolved, None
    else:
        description, source = resolved.description, resolved.source

    placed, attention, units = {}, {}, {}
    first = 0
    for index, component in enumerate(mesh_components):
        count = int(mesh_path_counts[index])
        if mesh_units[index] is not None:
            units[str(component)] = mesh_units[index].get_group("units")
        for path in mesh_paths[first : first + count]:
            placed[str(path)] = meshes[index]
            attention[str(path)] = attention_parallel(
                str(mesh_sequences[index])
            )
        first += count

    module: nn.Module
    if description.stub:
        from uniserve.models.stub import Model, image_processor

        module = Model().to(device)
        for path, target in (_devices(module, generation) or {}).items():
            module.get_submodule(path).to(target)
        loaded = Loaded(
            module,
            description,
            image_processor=image_processor(),
            attention_backend="torch",
        )
    elif host_only:
        # Only the stub model resolves to a bare description.
        assert source is not None
        # A host rank holds only host components: the weight-less
        # description carries the media geometry, rates and unit division.
        logger.info(
            "described numerical model %s for host work",
            type(description.model).__qualname__,
        )
        loaded = Loaded(
            description.model,
            description,
            attention_backend=backend,
            image_processor=source.image_processor,
            flow_prompt=source.flow_prompt,
        )
    else:
        # Only the stub model resolves to a bare description.
        assert source is not None
        partition = _expert_partition(
            expert_world, int(attention_ranks), bool(expert_dedicated)
        )
        result = registry.load_model(
            source,
            device=device,
            meshes=placed,
            attention=attention,
            devices=_devices(description.model, generation),
            experts=partition,
        )
        module = result.model
        if dedicated:
            # The worker owns invocation of the loaded expert modules, in
            # numerical traversal order; no other capability is fabricated.
            module = nn.ModuleList(
                child
                for child in module.modules()
                if isinstance(child, FusedMoE)
            )
        logger.info("loaded numerical model %s", type(module).__qualname__)
        loaded = Loaded(
            module,
            description,
            attention_backend=backend,
            image_processor=source.image_processor,
            flow_prompt=source.flow_prompt,
            tokenizer=None if dedicated else source.tokenizer,
        )

    if not dedicated and not host_only:
        loaded.calls = _bind_calls(loaded.module, description, units)
    from uniserve.backends import attention as attention_backends
    from uniserve.backends.attention import flashinfer

    loaded.attention = attention_backends.resolve(
        loaded.attention_backend,
        device=torch.device(device),
        flashinfer=flashinfer.Backend(flashinfer.Config(**options)),
    )
    return loaded


def _name(module: nn.Module) -> str:
    return f"{type(module).__module__}.{type(module).__qualname__}"


def _expert_partition(
    world, attention_ranks: int, dedicated: bool
) -> Communicator | None:
    """The communicator a rank's routed experts shard over.

    Colocated expert parallelism partitions over the whole union. A
    disaggregated attention rank keeps its experts unloaded (meta), and a
    dedicated expert rank shards over the expert ranks with an unbound
    communicator: the exchange, not torch, moves its tokens.
    """
    if world is None:
        return None
    union = world.experts
    if not attention_ranks:
        return None if union.size == 1 else union
    if not dedicated:
        return None
    return Communicator(
        union.ranks[attention_ranks:],
        union.rank - attention_ranks,
        "experts",
        union.device,
    )


def _bind_calls(
    module: nn.Module, description: Description, units: Mapping[str, Any]
) -> dict[str, tuple[Call, ...]]:
    """Resolve declared calls on the loaded model and attach unit rings.

    A cleared pipeline stage drops its calls. A temporally distributed
    ``VideoPostprocessor`` exchanges reconstruction overlap with its
    neighbors through the component's unit ring.
    """
    calls = describe_components(module, entries=description.entries)
    for name, items in calls.items():
        ring = units.get(name)
        if ring is None:
            continue
        for call in items:
            if isinstance(call.module, VideoPostprocessor):
                call.module.units = ring
    return calls


def _devices(
    model: nn.Module, generation_device: str | None
) -> Mapping[str, str] | None:
    """Place the flow route and denoiser modules on the generation device.

    Selected modules are every ``Denoiser``'s direct children that are not
    part of a ``CausalLM``, every ``flow`` entry of a ``ModuleDict``, and
    every ``ImageDecoder`` and ``PatchAutoencoder``. Every alias path maps
    to the device.

    Raises:
        SetupError: A generation device is set but nothing can be placed.
    """
    if generation_device is None:
        return None
    text_modules = {
        id(child)
        for module in model.modules()
        if isinstance(module, CausalLM)
        for child in module.modules()
    }
    placed = set()
    for module in model.modules():
        if isinstance(module, Denoiser):
            for child in module.children():
                if id(child) not in text_modules:
                    placed.add(id(child))
        if isinstance(module, nn.ModuleDict) and "flow" in module:
            placed.add(id(module["flow"]))
        if isinstance(module, (ImageDecoder, PatchAutoencoder)):
            placed.add(id(module))
    # Shared backbones and codecs have several module paths; every alias must
    # express the same placement.
    paths = {
        path: generation_device
        for path, module in model.named_modules(remove_duplicate=False)
        if id(module) in placed
    }
    if not paths:
        raise SetupError(
            "generation device requires a model with a distinct flow route"
        )
    return paths


def capability(model: nn.Module, kind: type[_Kind]) -> _Kind | None:
    """The model's unique module of ``kind``, if any."""
    found: list[_Kind] = []
    for module in model.modules():
        if isinstance(module, kind) and all(module is not m for m in found):
            found.append(module)
    if len(found) > 1:
        raise SetupError(
            f"model has more than one {kind.__name__} capability module"
        )
    return found[0] if found else None


def attention_backend(loaded: Loaded) -> str:
    """The name of the attention backend the model was loaded with."""
    return loaded.attention_backend


def attention_captures(loaded: Loaded) -> bool:
    """Whether serving captures the loaded attention provider's calls.

    See ``uniserve.backends.attention.Backend.captures``.
    """
    return bool(loaded.attention.captures)


def weight_formats(loaded: Loaded) -> list[str]:
    """The storage formats of the loaded parameters, sorted.

    ``"dense"`` or a quantizer's format name.
    """
    from uniserve.quantization import QuantizedTensor

    return sorted(
        {
            value.quantizer.format
            if isinstance(value, QuantizedTensor)
            else "dense"
            for value in loaded.module.parameters()
        }
    )


def activation_formats(loaded: Loaded) -> list[str]:
    """The input formats of the loaded linear layers, sorted.

    ``"dense"`` or an input quantizer's format name.
    """
    from uniserve.nn import Linear

    return sorted(
        {
            "dense"
            if child.input_quantizer is None
            else child.input_quantizer.format
            for child in loaded.module.modules()
            if isinstance(child, Linear)
        }
    )
