"""Reading a checkpoint's configuration and loading a rank's model.

A rank loads a model in two calls around process-group creation, each
reached by import path with explicit arguments:

1. :func:`read_config` reads the checkpoint configuration and builds a meta
   skeleton, before any collective exists. The rank reads the declared
   components, their calls, capability tags and communication axes from the
   returned :class:`Description`, one scalar or flat list per method.
2. :func:`load_model` narrows the checkpoint to this rank's modules and the
   launch's weight choices and materializes them on the meshes the rank
   created, or binds the tensors the rank retained when it last loaded the
   same model at the same position of its parallel configuration
   (:func:`fingerprint`, ``uniserve.loading.retained``).

The objects returned here are opaque to the rank and passed back unchanged.
What the rank reports of a loaded model it reads from the functions below
:func:`load_model`; a fact of one capability belongs to that capability's
runner.
"""

from __future__ import annotations

import gc
import json
import logging
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import torch
from torch import nn

from uniserve import loading as library
from uniserve import numerics, plugins
from uniserve.distributed import Communicator, communication_axes
from uniserve.errors import SetupError
from uniserve.loading import retained as retention
from uniserve.loading.layout import ExpertKernels
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
    io: library.Config
    entries: Mapping[str, Any]
    calls: dict[str, tuple[Call, ...]]
    # Axis selections each component's calls communicate over on the reading
    # rank's mesh; empty for a component the rank does not place.
    selections: Mapping[str, tuple[tuple[str, ...], ...]]
    # Module paths of the routed experts a split deployment places apart.
    expert_paths: tuple[str, ...]

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
    """A rank's loaded model and what serving needs from it.

    ``module`` is what the rank calls: the model, or a dedicated expert
    rank's routed expert layers. ``root`` is the whole loaded model, whose
    tensors the rank retains under their paths in it.
    """

    module: nn.Module
    description: Description
    root: nn.Module
    calls: dict[str, tuple[Call, ...]] = field(default_factory=dict)
    # The checkpoint source of each loaded tensor, by tensor object identity
    # (``uniserve.loading.loader.Result.origins``).
    origins: dict[int, str] = field(default_factory=dict)
    attention: Any = None
    attention_backend: str = "auto"
    image_processor: Any = None
    flow_prompt: Any = None
    tokenizer: Path | None = None
    # Loads the rank's modules of the checkpoint at a path, its tensors
    # streamed from a weight source when one is named, into new storage, as
    # the rank loaded this model (:func:`update_weights`); ``None`` for a
    # rank without checkpoint weights.
    reload: Callable[[str, str | None], nn.Module] | None = None
    # Runtime owner of the expert routing tensors borrowed by the model.
    routing: Any = None

    @property
    def model(self) -> nn.Module:
        return self.module


def read_config(
    stub: bool,
    path: str | None,
    io_format: str,
    io_mode: str,
    download_dir: str | None,
    num_threads: int | None,
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
        io_format, io_mode, download_dir, num_threads: How weights are read
            (``uniserve.loading.Config``).
        device: The rank's primary device.
        generation_device: The device of a generation tower placed apart.
        components: The launch's component placement as JSON object text:
            each component's ranks and parallel configuration.
        rank: This rank.
        role: ``"experts"`` for a rank holding only routed experts; a
            rank of any other role (``"model"``, ``"prefill"``,
            ``"decode"``) holds its request components.
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
    )
    path = _text(path)
    rank, role = int(rank), str(role)
    placed = placements(json.loads(str(components)))
    if stub:
        from uniserve.models.stub import Config, Model, entry_points

        with torch.device("meta"):
            model = Model(Config.read(path))
        metadata = None
        entries = entry_points(model.config)
    else:
        if path is None:
            raise SetupError("--model is required for a model worker")
        metadata = registry.read_config(path, io=io, modules=frozenset())
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
    weights: str | None = None,
) -> Description | Resolved:
    """Narrow the checkpoint to this rank's modules and weight choices.

    With ``weights``, a plugin's weight source (``scheme://...``,
    ``uniserve.plugins.weight_source``) streams the checkpoint's tensors in
    place of its files. The stub model needs no checkpoint and is returned
    unchanged.
    """
    if description.stub:
        return description
    # ``read_config`` requires a checkpoint path of every model but the stub.
    assert description.path is not None
    source = registry.read_config(
        description.path,
        io=description.io,
        modules=frozenset(str(path) for path in modules),
        exclude_modules=frozenset(str(path) for path in exclude_modules),
    )
    source = _streamed(source, weights)
    options = json.loads(str(quantization))
    chosen = weight_config(source, options, str(model_dtype))
    return Resolved(replace(source, weights=chosen), description)


def _streamed(source, weights: str | None):
    """``source`` with its checkpoints read from the weight source
    ``weights`` (``scheme://...``) in place of their files, if any.
    """  # noqa: D205
    if not weights:
        return source
    stream = plugins.weight_source(str(weights))
    identity = str(stream.identity())
    return replace(
        source,
        checkpoint=tuple(
            replace(checkpoint, files=(), stream=stream, identity=identity)
            for checkpoint in source.checkpoint
        ),
    )


def _expert_kernels(
    exchange: str | None, attention_ranks: int
) -> ExpertKernels:
    """Which kernels read the routed experts of a launch.

    A ``megamoe`` exchange runs them in the fused MegaMoE kernel, or in
    split MegaMoE's expert kernels when experts have ranks of their own;
    every other launch runs them in the provider ``auto`` resolves.
    """
    if exchange != "megamoe":
        return "local"
    return "split" if attention_ranks else "megamoe"


def attach_fences(loaded: Loaded, patterns: str) -> str:
    """Attach the worker's fence probes to the rank's whole loaded model
    (``uniserve.profiling.attach_fences`` of ``loaded.root``).
    """  # noqa: D205
    from uniserve.profiling import attach_fences as attach

    return attach(loaded.root, patterns)


def identity(loaded: Loaded) -> str:
    """The identity of what a rank's loaded model computes
    (``uniserve.numerics.identity`` of its whole model), which keys the
    cache pages its forwards write.
    """  # noqa: D205
    return numerics.identity(loaded.root)


def cache_identity(loaded: Loaded) -> str:
    """Cache-component identities of a model without checkpoint weights."""
    return json.dumps(
        numerics.cache_identities(loaded.description.model), sort_keys=True
    )


def fingerprint(
    description: Description,
    modules: Sequence[str],
    exclude_modules: Sequence[str],
    quantization: str,
    model_dtype: str,
    expert_exchange: str | None,
    attention_ranks: int,
    weights: str | None = None,
) -> str:
    """The digests of what determines this rank's loaded weights.

    The regions a rank holds aside (``uniserve.loading.retained.fingerprints``),
    as JSON text; the arguments are those of :func:`load_model`. Empty for
    the stub model, whose weights are not retained.
    """
    resolved = _resolve(
        description,
        modules,
        exclude_modules,
        quantization,
        model_dtype,
        _text(weights),
    )
    if isinstance(resolved, Description):
        return ""
    kernels = _expert_kernels(_text(expert_exchange), int(attention_ranks))
    # Tensor values and cache identities cover every component of the
    # model, whether this rank holds it or not, so every rank keys them
    # alike: a rank narrowed to some components also reads the checkpoint
    # metadata of the others, resolved as a rank holding them resolves them.
    # Only the rank's own tensors enter its retained-weight manifest.
    source = resolved.source
    held = {checkpoint.name: checkpoint for checkpoint in source.checkpoint}
    whole = _checkpoints(description, _text(weights))
    if any(checkpoint.name not in held for checkpoint in whole):
        source = replace(
            source,
            checkpoint=tuple(
                held.get(checkpoint.name, checkpoint) for checkpoint in whole
            ),
        )
    return json.dumps(
        retention.fingerprints(source, description.io, kernels),
        sort_keys=True,
    )


def _checkpoints(description: Description, weights: str | None) -> tuple:
    """Every checkpoint source of the described model, of every component.

    Resolved as for a rank that loads the whole model: a Hub snapshot
    fetches their files.
    """
    assert description.path is not None
    source = registry.read_config(description.path, io=description.io)
    return _streamed(source, weights).checkpoint


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
    mesh_projections: Sequence[str],
    mesh_projection_rows: Sequence[int | None],
    expert_world: Any,
    attention_ranks: int,
    expert_dedicated: bool,
    expert_exchange: str | None,
    tensor_experts: bool,
    retained: retention.Retained | None,
    load: Sequence[str],
    attention_backend: str,
    workspace_size: int,
    decode_backend: str,
    prefill_backend: str,
    disable_split_kv: bool,
    decode_split_tile_size: int | None,
    prefill_split_tile_size: int | None,
    use_tensor_core: bool | None,
    weights: str | None = None,
    unread: bool = False,
    expert_order: str = "{}",
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
        mesh_projections: Each component's Ulysses head projection mode,
            ``gather`` or ``exchange``.
        mesh_projection_rows: Each component's exchange step in rows, or
            ``None`` to exchange its whole shard at once.
        expert_world: The torch world holding the expert union's
            communicator; ``None`` without an expert union.
        attention_ranks: Attention ranks of a disaggregated union.
        expert_dedicated: This rank holds only routed experts.
        expert_exchange: The expert union's exchange transport, which
            selects the kernels its routed experts are laid out for;
            ``None`` without an expert union.
        tensor_experts: The tensor-parallel group of each mesh divides the
            routed experts among its ranks, each keeping whole experts,
            instead of every expert's channels.
        retained: The tensors this rank retained when it last loaded this
            model at this position, which it binds instead of reading the
            checkpoint; ``None`` loads.
        load: With ``retained``, the tensors to read from the checkpoint
            instead, which the rank's loading peers agreed on
            (``uniserve.loading.loader.load_model``).
        attention_backend: The attention backend's name.
        workspace_size, decode_backend, prefill_backend, disable_split_kv,
        decode_split_tile_size, prefill_split_tile_size, use_tensor_core:
            FlashInfer options (``flashinfer.Config``); ``None`` keeps an
            option's default.
        weights: A plugin's weight source (``scheme://...``) that streams
            the checkpoint's tensors in place of its files; ``None`` reads
            the files.
        unread: Materialize every tensor without reading the checkpoint
            or computing a value (the ``dummy`` mode of
            ``uniserve.loading.Config`` without ``values``): the model has
            the storages, shapes, element types and encodings a load gives
            it, and uninitialized bytes. A rank plans the layout of the
            weights it receives from another world with it.
        expert_order: Global expert permutations by layer path, as JSON.
    """
    weights = _text(weights)
    resolved = _resolve(
        description,
        modules,
        exclude_modules,
        quantization,
        model_dtype,
        weights,
    )
    if bool(unread) and isinstance(resolved, Resolved):
        source = resolved.source
        resolved = replace(
            resolved,
            source=replace(
                source, io=replace(source.io, mode="dummy", values=False)
            ),
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
            rows = mesh_projection_rows[index]
            attention[str(path)] = attention_parallel(
                str(mesh_sequences[index]),
                str(mesh_projections[index]),
                None if rows is None else int(rows),
            )
        first += count

    module: nn.Module
    if description.stub:
        from uniserve.models.stub import Model, image_processor

        described = description.model
        assert isinstance(described, Model)
        module = Model(described.config).to(device)
        for path, target in (_devices(module, generation) or {}).items():
            module.get_submodule(path).to(target)
        loaded = Loaded(
            module,
            description,
            module,
            image_processor=image_processor(),
            attention_backend="torch",
        )
    elif host_only:
        # Only the stub model resolves to a bare description.
        assert source is not None
        # A CPU worker holds only host components: the weight-less
        # description carries the media geometry, rates and unit division.
        logger.info(
            "described numerical model %s for host work",
            type(description.model).__qualname__,
        )
        loaded = Loaded(
            description.model,
            description,
            description.model,
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
        kernels = _expert_kernels(_text(expert_exchange), int(attention_ranks))
        started = time.perf_counter()
        result = registry.load_model(
            source,
            device=device,
            meshes=placed,
            attention=attention,
            devices=_devices(description.model, generation),
            experts=partition,
            tensor_experts=bool(tensor_experts),
            expert_kernels=kernels,
            retained=retained,
            load=frozenset(str(name) for name in load),
            expert_order=json.loads(str(expert_order)),
        )
        checkpoint = description

        def reload(path: str, weights: str | None) -> nn.Module:
            fresh = _resolve(
                replace(checkpoint, path=path),
                modules,
                exclude_modules,
                quantization,
                model_dtype,
                weights,
            )
            assert isinstance(fresh, Resolved)
            return registry.load_model(
                fresh.source,
                device=device,
                meshes=placed,
                attention=attention,
                devices=_devices(checkpoint.model, generation),
                experts=partition,
                tensor_experts=bool(tensor_experts),
                expert_kernels=kernels,
                expert_order=result.routing.orders,
            ).model

        module = result.model
        if dedicated:
            # The worker owns invocation of the loaded expert modules, in
            # numerical traversal order; no other capability is fabricated.
            module = nn.ModuleList(
                child
                for child in module.modules()
                if isinstance(child, FusedMoE)
            )
        if unread:
            action = "materialized without reading"
        elif retained is None:
            action = "loaded"
        elif load:
            action = f"bound retained tensors and read {len(load)} of"
        else:
            action = "bound retained"
        logger.info(
            "%s numerical model %s in %.1f s",
            action,
            type(module).__qualname__,
            time.perf_counter() - started,
        )
        loaded = Loaded(
            module,
            description,
            result.model,
            attention_backend=backend,
            image_processor=source.image_processor,
            flow_prompt=source.flow_prompt,
            tokenizer=None if dedicated else source.tokenizer,
            origins=result.origins,
            reload=reload,
            routing=result.routing,
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
    # Worker hooks reach the modules a rank loaded (``uniserve.hooks``).
    from uniserve import hooks

    # Admin hooks use the same checkpoint module paths as expert orders and
    # retained manifests, including on dedicated expert ranks. The callable
    # subset's ModuleList indices are not model identities.
    hooks.register_model(loaded.root)
    return loaded


def update_weights(loaded: Loaded, path: str) -> bool:
    """Rewrite the rank's loaded tensors in place from the checkpoint at
    ``path``; whether the rank holds checkpoint weights to rewrite.

    The checkpoint loads into new storage as the rank loaded its model (the
    same modules, meshes and weight choices: ``Loaded.reload``); every
    tensor of the loaded model then takes its new value where it lies, so
    the CUDA graphs and the retained storages that address it read the new
    weights. Every tensor is checked before any is written: the checkpoint
    must load the same tensors in the same representations, shapes and
    element types, else the loaded model stays as it was.

    ``path`` is a checkpoint directory, or a plugin's weight source
    (``scheme://...``) that streams the tensors of the rank's checkpoint,
    whose configuration stays its own. A rank without checkpoint weights
    (the stub model, a host-only rank) has nothing to rewrite.

    Raises:
        ValueError: The checkpoint loads other tensors, or a tensor in
            another representation, shape or element type.
    """  # noqa: D205
    if loaded.reload is None:
        return False
    path = str(path)
    started = time.perf_counter()
    try:
        replaced = _update_weights(loaded, path)
    except BaseException as error:
        # The frames of a failed load hold what it allocated; drop them so
        # the storage returns before the workers wake.
        traceback.clear_frames(error.__traceback__)
        _release_device_storage()
        raise
    _release_device_storage()
    logger.info(
        "updated %d tensors in place from %s in %.1f s; %.2f GiB of device "
        "storage stay allocated",
        replaced,
        path,
        time.perf_counter() - started,
        torch.cuda.memory_allocated() / 2**30
        if torch.cuda.is_available()
        else 0.0,
    )
    return True


def _release_device_storage() -> None:
    """Free what an update's discarded model held: its modules may hold
    each other in cycles, which only the cycle collector frees.
    """  # noqa: D205
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _update_weights(loaded: Loaded, path: str) -> int:
    """Load the checkpoint at ``path`` and copy every tensor into place
    (``update_weights``); the tensors copied.
    """  # noqa: D205
    streamed = "://" in path
    checkpoint = loaded.description.path
    assert checkpoint is not None
    assert loaded.reload is not None
    fresh = (
        loaded.reload(checkpoint, path)
        if streamed
        else loaded.reload(path, None)
    )

    def entries(model: nn.Module) -> dict[str, Any]:
        return {
            name: retention.tensor_entry(name, tensor)
            for name, tensor in retention.tensors(model)
            if not tensor.is_meta
        }

    current, incoming = entries(loaded.root), entries(fresh)
    if current.keys() != incoming.keys():
        missing = sorted(current.keys() - incoming.keys())[:4]
        extra = sorted(incoming.keys() - current.keys())[:4]
        raise ValueError(
            f"the checkpoint at {path} loads other tensors: it lacks "
            f"{missing} and adds {extra}"
        )
    pairs = []
    for name, entry in current.items():
        other = incoming[name]
        if (
            entry.representation != other.representation
            or entry.fields.keys() != other.fields.keys()
        ):
            raise ValueError(
                f"the checkpoint at {path} loads {name} in another "
                "representation"
            )
        for key, target in entry.fields.items():
            value = other.fields[key]
            if (target.shape, target.dtype) != (value.shape, value.dtype):
                raise ValueError(
                    f"the checkpoint at {path} loads {name} ({key}) as "
                    f"{tuple(value.shape)} {value.dtype}, not "
                    f"{tuple(target.shape)} {target.dtype}"
                )
            pairs.append((target, value))
    with torch.no_grad():
        for target, value in pairs:
            target.copy_(value)
    if not streamed:
        loaded.description = replace(loaded.description, path=path)
    return len(current)


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


def uncaptured(loaded: Loaded) -> str:
    """Why serving does not capture the loaded model's calls, or ``""``
    when it captures them all.

    Serving captures a call of the attention provider when the provider
    says so (``uniserve.backends.attention.Backend.captures``); a module
    whose prompt forwards it does not capture is :func:`uncaptured_prompts`'
    concern.
    """  # noqa: D205
    if not loaded.attention.captures:
        return (
            f"the {loaded.attention_backend} attention provider's calls are "
            "not captured"
        )
    return ""


def uncaptured_prompts(loaded: Loaded) -> str:
    """Why serving does not capture the loaded model's prompt forwards, or
    ``""`` when it captures them.

    A module class whose step forwards (decodes and verifications) serving
    captures but whose prompt forwards it does not says so
    (``captures_prompts = False``, an implementation whose prompt forward
    runs a host loop over its chunks): serving runs the model's prompt
    forwards eagerly and replays the graphs of its step forwards.
    """  # noqa: D205
    for module in loaded.module.modules():
        if getattr(type(module), "captures_prompts", True) is False:
            return f"{type(module).function}'s prompt forwards are not captured"
    return ""


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
