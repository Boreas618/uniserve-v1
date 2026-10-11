"""Model description, checkpoint resolution, loading and model facts.

The worker loads a model in phases around process-group creation:

1. ``model.describe`` reads the checkpoint configuration, builds a meta
   skeleton and reports the declared components, their capability tags and
   the axes they communicate over, before any collective exists.
2. ``model.resolve`` narrows the checkpoint to this rank's modules and the
   launch's weight choices.
3. ``model.load`` materializes the resident modules on the meshes Rust
   created.
4. ``model.facts`` reports every model-derived quantity capacity planning
   and reporting need, so the worker never inspects Python objects.

Requests arrive as TVM-FFI maps and arrays; handles returned here are opaque
to the worker and passed back unchanged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import torch
from torch import nn

from uniserve_models import catalog
from uniserve_models import loading as library
from uniserve_models.distributed import Communicator, communication_axes
from uniserve_models.loading.precision import weight_config
from uniserve_models.model import (
    AudioEncoder,
    CausalLM,
    Denoiser,
    ImageDecoder,
    ImageDenoiser,
    PatchEncoder,
    TextEncoder,
    VideoDecoder,
    VideoEncoder,
    VideoPostprocessor,
)
from uniserve_models.nn.attention.vsa import BlockAttention
from uniserve_models.nn.moe import FusedMoE
from uniserve_models.nn.vae.patch import PatchAutoencoder

from .components import (
    Call,
    Placement,
    attention_parallel,
    capability_tags,
    describe_components,
    placements,
)
from .errors import SetupError

logger = logging.getLogger(__name__)

# A capability class a model's module is looked up by.
_Kind = TypeVar("_Kind")


def functions() -> dict[str, Callable[..., Any]]:
    return {
        "model.describe": describe,
        "model.resolve": resolve,
        "model.load": load,
        "model.facts": facts,
    }


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


@dataclass
class Description:
    """A checkpoint's configuration and meta skeleton before loading.

    ``metadata`` is ``None`` for the stub model.
    """

    stub: bool
    model: nn.Module
    metadata: catalog.Config | None
    path: str | None
    base: str | None
    io: library.Config
    entries: Mapping[str, Any]
    calls: dict[str, tuple[Call, ...]]


@dataclass
class Resolved:
    """A checkpoint narrowed to this rank's modules, with its weights."""

    source: catalog.Config
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
    model_name: str = ""

    @property
    def model(self) -> nn.Module:
        return self.module


def _io(value: Mapping[str, Any]) -> library.Config:
    directory = _text(value.get("download_dir"))
    manifest = _text(value.get("checksum_manifest"))
    threads = value.get("num_threads")
    # The worker sends plain strings; ``Config`` rejects unknown values.
    return library.Config(
        format=cast(Literal["auto", "safetensors", "pt"], str(value["format"])),
        mode=cast(Literal["eager", "layered", "dummy"], str(value["mode"])),
        download_dir=None if directory is None else Path(directory),
        num_threads=None if threads is None else int(threads),
        checksum_manifest=None if manifest is None else Path(manifest),
    )


def describe(request: Mapping[str, Any]) -> list[Any]:
    """Describe the model and this rank's view of its components.

    Returns:
        ``[description, info]`` where ``info`` is ``{"model_name",
        "entries": [{"name", "calls": [{"path", "method", "stage",
        "capabilities", "axes"}]}], "expert_paths", "video_decoder"}``.

    Raises:
        SetupError: The placement does not fit the declarations, no installed
            sparse attention provider serves a resident tile on the device,
            a generation device has no flow route to place, or a split expert
            deployment has no routed expert layer.
    """
    stub = bool(request["stub"])
    io = _io(request["io"])
    path, base = _text(request.get("path")), _text(request.get("base_model"))
    rank, role = int(request["rank"]), str(request["role"])
    placed = placements(request["components"])
    if stub:
        from uniserve_models.stub import Model, entry_points

        with torch.device("meta"):
            model = Model()
        metadata = None
        entries = entry_points(model.config)
    else:
        if path is None:
            raise SetupError("--model is required for a model worker")
        metadata = catalog.read_config(
            path, io=io, modules=frozenset(), base=base
        )
        with torch.device("meta"):
            model = metadata.model_class(metadata.model)
        entries = metadata.entry_points

    calls = (
        {} if role == "experts" else describe_components(model, entries=entries)
    )
    resident = {
        call.path
        for name, placement in placed.items()
        if rank in placement.ranks
        for call in calls.get(name, ())
    }
    _require_sparse_tiles(model, resident, str(request["device"]))
    generation = _text(request.get("generation_device"))
    if generation is not None:
        _devices(model, generation)

    expert_paths: list[str] = []
    if request.get("expert_split"):
        expert_paths = [
            name
            for name, module in model.named_modules()
            if isinstance(module, FusedMoE)
        ]
        if not expert_paths:
            raise SetupError(
                "disaggregated workers require routed expert layers"
            )

    reported = []
    for name, items in calls.items():
        placement = placed.get(name)
        topology = None if placement is None else placement.topology(rank)
        reported.append(
            {
                "name": name,
                "calls": [
                    {
                        "path": call.path,
                        "method": call.entry_point.method,
                        "stage": call.entry_point.stage,
                        "capabilities": capability_tags(call.module),
                        "axes": [
                            list(axes)
                            for axes in _axes(call, placement, topology)
                        ],
                    }
                    for call in items
                ],
            }
        )
    description = Description(
        stub, model, metadata, path, base, io, entries, calls
    )
    info = {
        "model_name": f"{type(model).__module__}.{type(model).__qualname__}",
        "entries": reported,
        "expert_paths": expert_paths,
        "video_decoder": any(
            isinstance(module, VideoDecoder) for module in model.modules()
        ),
    }
    return [description, info]


def _axes(call: Call, placement: Placement | None, topology):
    """Axis selections a call communicates over on this rank's mesh."""
    if placement is None or topology is None:
        return []
    axes = set(
        communication_axes(
            call.module,
            topology,
            attention=attention_parallel(placement.sequence),
        )
    )
    axes |= set(call.entry_point.communication_axes(topology))
    return sorted(axes)


def resolve(
    description: Description, request: Mapping[str, Any]
) -> Description | Resolved:
    """Narrow the checkpoint to this rank's modules and weight choices.

    ``request`` carries ``modules``, ``exclude_modules``, ``quantization``
    (the launch's precision selectors as JSON object text) and
    ``model_dtype``. The stub model needs no checkpoint and is returned
    unchanged.
    """
    if description.stub:
        return description
    # ``describe`` requires a checkpoint path of every model but the stub.
    assert description.path is not None
    source = catalog.read_config(
        description.path,
        io=description.io,
        modules=frozenset(str(path) for path in request["modules"]),
        base=description.base,
        exclude_modules=frozenset(
            str(path) for path in request["exclude_modules"]
        ),
    )
    options = json.loads(str(request["quantization"]))
    weights = weight_config(source, options, str(request["model_dtype"]))
    return Resolved(replace(source, weights=weights), description)


def load(
    resolved: Description | Resolved, request: Mapping[str, Any]
) -> Loaded:
    """Materialize this rank's selected modules on their meshes.

    The stub model is built directly on the device and serves with the
    portable ``torch`` attention backend. A host-only rank keeps the
    weight-less description. Otherwise each listed outermost path loads on
    its component's mesh with that component's attention partition.
    """
    device = str(request["device"])
    generation = _text(request.get("generation_device"))
    role = str(request["role"])
    dedicated = role == "experts"
    backend = str(request["attention_backend"])
    options = {
        str(key): value for key, value in dict(request["attention"]).items()
    }

    if isinstance(resolved, Description):
        description, source = resolved, None
    else:
        description, source = resolved.description, resolved.source

    meshes, attention, units = {}, {}, {}
    for item in request.get("meshes") or ():
        component = str(item["component"])
        if item.get("units") is not None:
            units[component] = item["units"].get_group("units")
        for path in item["paths"]:
            meshes[str(path)] = item["mesh"]
            attention[str(path)] = attention_parallel(str(item["sequence"]))

    module: nn.Module
    if description.stub:
        from uniserve_models.stub import Model, image_processor

        module = Model().to(device)
        for path, target in (_devices(module, generation) or {}).items():
            module.get_submodule(path).to(target)
        loaded = Loaded(
            module,
            description,
            image_processor=image_processor(),
            attention_backend="torch",
            model_name=f"{type(module).__module__}.{type(module).__qualname__}",
        )
    elif request.get("host_only"):
        # ``resolve`` returns a bare description for the stub model alone.
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
            model_name=_name(description.model),
        )
    else:
        # ``resolve`` returns a bare description for the stub model alone.
        assert source is not None
        partition = _expert_partition(request.get("experts"))
        result = catalog.load_model(
            source,
            device=device,
            meshes=meshes,
            attention=attention,
            devices=_devices(description.model, generation),
            experts=partition,
        )
        module = result.model
        name = _name(module)
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
            model_name=name,
        )

    if not dedicated and not request.get("host_only"):
        loaded.calls = _bind_calls(loaded.module, description, units)
    from uniserve_models.backends import attention as attention_backends
    from uniserve_models.backends.attention import flashinfer

    loaded.attention = attention_backends.resolve(
        loaded.attention_backend,
        device=torch.device(device),
        flashinfer=flashinfer.Backend(flashinfer.Config(**options)),
    )
    return loaded


def _name(module: nn.Module) -> str:
    return f"{type(module).__module__}.{type(module).__qualname__}"


def _expert_partition(experts) -> Communicator | None:
    """The communicator a rank's routed experts shard over.

    Colocated expert parallelism partitions over the whole union. A
    disaggregated attention rank keeps its experts unloaded (meta), and a
    dedicated expert rank shards over the expert ranks with an unbound
    communicator: the exchange, not torch, moves its tokens.
    """
    if experts is None:
        return None
    union = experts["world"].experts
    attention_ranks = int(experts["attention_ranks"])
    if not attention_ranks:
        return None if union.size == 1 else union
    if not experts["dedicated"]:
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


def _require_sparse_tiles(
    model: nn.Module, paths: set[str], device: str
) -> None:
    """Refuse a device that no installed VSA provider serves for a tile.

    A resident sparse attention layer's tile size fixes the block kernel it
    needs; 128-row tiles have an SM100 kernel only.
    """
    from uniserve_models.backends.attention import vsa as providers

    tiles = sorted(
        {
            child.tile_size
            for path in paths
            for child in model.get_submodule(path).modules()
            if isinstance(child, BlockAttention)
        }
    )
    for tile in tiles:
        try:
            providers.resolve("auto", device=torch.device(device), tile=tile)
        except RuntimeError as error:
            raise SetupError(str(error)) from error


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


# Torch dtypes a product may carry on the wire.
_PRODUCT_DTYPES = {
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
}


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


def product_name(module: nn.Module, name: str) -> str:
    """Name a numerical output by its downstream protocol use."""
    if isinstance(module, Denoiser):
        return {"video": "video_latents", "audio": "audio_latents"}.get(
            name, name
        )
    if isinstance(module, PatchEncoder) and name == "features":
        return "vision_features"
    if isinstance(module, VideoEncoder) and name == "video":
        return "condition_video_latents"
    if isinstance(module, AudioEncoder) and name == "audio":
        return "condition_audio_latents"
    if isinstance(module, VideoDecoder) and name == "video":
        return "video_units"
    return name


def _outputs(
    calls: Mapping[str, Sequence[Call]], max_sequence_tokens: int, dtype
) -> dict[str, list[dict[str, Any]]]:
    """Products the loaded calls publish, bounded by the admitted sizes.

    A text encoder's ``encode`` publishes its encoded conditioning. Products
    of media timelines (denoisers, decoders and condition encoders, sized by
    a media builder) are declared by the media runtimes' facts.
    """
    result = {}
    for component, items in calls.items():
        outputs = []
        for call in items:
            if not (
                isinstance(call.module, TextEncoder)
                and call.entry_point.method == "encode"
            ):
                continue
            layouts = call.module.output_layout(max_sequence_tokens, dtype)
            for name, layout in layouts.items():
                if layout.dtype not in _PRODUCT_DTYPES:
                    raise SetupError(
                        f"result {component}.{name} has no protocol dtype"
                    )
                if len(layout.variable_axes) > 1:
                    raise SetupError(
                        f"result {component}.{name} exceeds the protocol "
                        "dynamic axes"
                    )
                outputs.append(
                    {
                        "name": product_name(call.module, name),
                        "dtype": str(layout.dtype).removeprefix("torch."),
                        "shape": list(layout.shape),
                        "variable_axes": list(layout.variable_axes),
                        "raster_axes": None,
                    }
                )
        if outputs:
            result[component] = outputs
    return result


def _vision_feature_bytes(model: nn.Module, processor) -> int:
    """Bytes of the largest vision feature product the processor admits.

    Rows come from the processor's ViT transform pixel bound and the
    encoder stride; zero without a ``PatchEncoder``, processor or ViT.
    """
    encoder = capability(model, PatchEncoder)
    if encoder is None or processor is None or processor.vit is None:
        return 0
    from uniserve_models.media import image
    from uniserve_models.processing import PatchTransform

    transform = processor.vit
    pixels = (
        transform.pixel_bound()
        if isinstance(transform, PatchTransform)
        else min(transform.resize.max_pixels, transform.resize.max_size**2)
    )
    stride = encoder.patch_size * encoder.downsample
    tokens = pixels // stride**2
    if tokens == 0:
        return 0
    feature = encoder.output_layout(image.Config(stride, stride))["features"]
    return tokens * feature.shape[1] * feature.dtype.itemsize


def _image_flow(model: nn.Module) -> dict[str, int] | None:
    """The image denoiser's latent geometry for the latent pool plan."""
    # mypy rejects an abstract class as a ``type`` argument; the lookup
    # only tests membership.
    denoiser = capability(model, ImageDenoiser)  # type: ignore[type-abstract]
    if denoiser is None:
        return None
    return {
        "max_tokens": int(denoiser.max_sequence_tokens),
        "latent_channels": int(denoiser.latent_channels),
        "patch_size": int(denoiser.patch_size),
    }


def facts(loaded: Loaded, bounds: Mapping[str, Any]) -> dict[str, Any]:
    """Report model-derived quantities for capacity planning and reporting.

    ``bounds`` carries ``max_sequence_tokens``, ``model_dtype`` and
    ``role``.
    """
    from uniserve_models.nn import Linear
    from uniserve_models.quantization import QuantizedTensor

    model = loaded.module
    dtype = getattr(torch, str(bounds["model_dtype"]).removeprefix("torch."))
    text = (
        None
        if str(bounds["role"]) == "experts"
        else capability(model, CausalLM)
    )
    weight_formats = sorted(
        {
            value.quantizer.format
            if isinstance(value, QuantizedTensor)
            else "dense"
            for value in model.parameters()
        }
    )
    activation_formats = sorted(
        {
            "dense"
            if child.input_quantizer is None
            else child.input_quantizer.format
            for child in model.modules()
            if isinstance(child, Linear)
        }
    )
    experts = str(bounds["role"]) == "experts"
    return {
        "model_name": loaded.model_name or _name(model),
        "attention_backend": loaded.attention_backend,
        "vocab_size": None if text is None else int(text.backbone.vocab_size),
        "hidden_size": None if text is None else int(text.backbone.hidden_size),
        # Expert results return through the bound exchange, not request IPC.
        "outputs": {}
        if experts
        else _outputs(loaded.calls, int(bounds["max_sequence_tokens"]), dtype),
        "weight_formats": weight_formats,
        "activation_formats": activation_formats,
        "image_processor": loaded.image_processor is not None,
        "vision_feature_bytes": _vision_feature_bytes(
            model, loaded.image_processor
        ),
        "image_flow": None if experts else _image_flow(model),
    }
