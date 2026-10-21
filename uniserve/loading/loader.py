"""Public construction and weight loading over ordinary numerical mappings."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

import torch
from torch import nn

from uniserve.distributed import (
    Communicator,
    DeviceMesh,
    parallelize_,
    partition_experts,
)
from uniserve.nn.attention import AttentionParallelConfig

from . import checkpoint as checkpoint_module
from . import weights as weight_options
from .config import Config
from .layout import ExpertKernels, finalize
from .retained import Retained, tensors

ConfigT = TypeVar("ConfigT")
ModelT = TypeVar("ModelT", bound=nn.Module)


@dataclass(frozen=True, slots=True)
class Result(Generic[ModelT]):
    """A materialized model, its resolved sources and completeness reports.

    ``origins`` names the checkpoint source of each tensor a declared
    mapping loads, by tensor object identity.
    """

    model: ModelT
    sources: tuple[checkpoint_module.Source, ...]
    reports: tuple[weight_options.Report, ...]
    origins: dict[int, str]


def _origins(
    declared: tuple[weight_options.ModuleMapping, ...],
) -> dict[int, str]:
    """The checkpoint source of every tensor under each mapping's module.

    A module under two mappings takes the first's source.
    """
    origins: dict[int, str] = {}
    for component in declared:
        for _, tensor in tensors(component.module):
            origins.setdefault(id(tensor), component.source)
    return origins


def _reads(
    model: nn.Module,
    declared: tuple[weight_options.ModuleMapping, ...],
    load: frozenset[str],
) -> frozenset[str]:
    """The tensors to read so that ``load`` (tensor identities) is read.

    A mapping that derives constants after loading (``post_load``) reads
    its whole module once any of its tensors is read, and a merged
    projection reads all of its branches, which loading lays out in one
    backing. Every rank computes the same set from the same ``load``.
    """
    from uniserve.nn.linear import MergedColumnParallelLinear

    names = {id(tensor): name for name, tensor in tensors(model)}
    reads = set(load)

    def within(module: nn.Module) -> set[str]:
        return {names[id(tensor)] for _, tensor in tensors(module)}

    for module in model.modules():
        if isinstance(module, MergedColumnParallelLinear):
            members = within(module)
            if members & reads:
                reads |= members
    for component in declared:
        if component.post_load is None:
            continue
        members = within(component.module)
        if members & reads:
            reads |= members
    return frozenset(reads)


def select_modules(
    model: nn.Module,
    modules: frozenset[str] | None = None,
    *,
    exclude_modules: frozenset[str] = frozenset(),
) -> frozenset[nn.Module]:
    """Resolve selected subtrees, removing excluded subtrees by identity.

    ``None`` selects the model root. Exclusion also removes aliases of the
    same module, so a shared backbone has one residency decision. This lets
    callers retain a parent's own parameters while placing a nested expert
    subtree elsewhere. Unknown paths raise ``ValueError``.
    """
    try:
        selected = {
            child
            for path in (("",) if modules is None else modules)
            for child in model.get_submodule(path).modules()
        }
        excluded = {
            child
            for path in exclude_modules
            for child in model.get_submodule(path).modules()
        }
    except AttributeError as error:
        raise ValueError(
            "module selection requires actual module paths"
        ) from error
    return frozenset(selected - excluded)


def _load(
    model,
    checkpoint,
    mappings,
    *,
    device,
    weights,
    io,
    devices,
    excluded=(),
    selected=None,
    known_paths=None,
):
    loader = weight_options._Loader(
        model,
        checkpoint,
        mappings,
        device=device,
        weights=weights,
        io=io,
        devices=devices,
        excluded=excluded,
        selected=selected,
        known_paths=known_paths,
    )
    try:
        with torch.no_grad():
            return loader.load()
    finally:
        loader.close()


def load_weights(
    model: ModelT,
    checkpoint: tuple[checkpoint_module.Source, ...],
    *,
    mapping: Callable[[ModelT], tuple[weight_options.ModuleMapping, ...]],
    device: torch.device | str,
    weights: weight_options.Config = weight_options.Config(),
    io: Config = Config(),
    devices: Mapping[str, torch.device | str] | None = None,
    expert_kernels: ExpertKernels = "local",
) -> tuple[weight_options.Report, ...]:
    """Load an existing module via the construction-time assignment path.

    The same assignment and conversion path applies as in construction.
    Complete-source matrix assignments follow already bound mathematical
    partitions. Readers retire before return; Parameters retain their storage
    and shared identities independently of checkpoint file handles. Encoded
    weights end in the layout of the kernels that read them
    (``uniserve.loading.layout``; ``expert_kernels`` names those of the
    routed experts).
    """
    known_paths = frozenset(
        path for path, _ in model.named_modules(remove_duplicate=False)
    )
    reports = _load(
        model,
        checkpoint,
        mapping(model),
        device=device,
        weights=weights,
        io=io,
        devices=devices,
        known_paths=known_paths,
    )
    finalize(model, experts=expert_kernels)
    return reports


def _release_unselected(model: nn.Module, selected: set[int]) -> None:
    """Leave every module this rank does not select on meta.

    A constructor may build a tensor on a concrete device even under the
    meta device, such as a VAE's latent statistics computed from its
    configuration. A module the rank does not hold keeps no such tensor: its
    own parameters and buffers become meta placeholders, as everything
    unselected is, and a selected module's tensors stay as they are.
    """
    for module in model.modules():
        if id(module) in selected:
            continue
        for name, parameter in module._parameters.items():
            if parameter is not None and not parameter.is_meta:
                module._parameters[name] = nn.Parameter(
                    parameter.to("meta"), requires_grad=parameter.requires_grad
                )
        for name, buffer in module._buffers.items():
            if buffer is not None and not buffer.is_meta:
                module._buffers[name] = buffer.to("meta")


def load_model(
    model_class: Callable[[ConfigT], ModelT],
    config: ConfigT,
    *,
    checkpoint: tuple[checkpoint_module.Source, ...],
    mapping: Callable[[ModelT], tuple[weight_options.ModuleMapping, ...]],
    device: torch.device | str,
    weights: weight_options.Config = weight_options.Config(),
    io: Config = Config(),
    meshes: Mapping[str, DeviceMesh] | None = None,
    attention: Mapping[str, AttentionParallelConfig] | None = None,
    devices: Mapping[str, torch.device | str] | None = None,
    modules: frozenset[str] | None = None,
    exclude_modules: frozenset[str] = frozenset(),
    experts: Communicator | None = None,
    expert_kernels: ExpertKernels = "local",
    retained: Retained | None = None,
    load: frozenset[str] = frozenset(),
) -> Result[ModelT]:
    """Construct on meta, bind partitions, then materialize selected modules.

    The bound partitions are mathematical. Module paths choose resources and
    numerical settings. Unselected modules remain on meta so architecture
    dimensions and layout queries stay available. Neither constructors nor
    the resulting model retain loading state. ``experts`` names the
    expert-parallel group every ``FusedMoE`` shards its experts over
    (``partition_experts``), so this rank reads only its own experts.
    ``exclude_modules`` leaves named subtrees on meta even when their
    ancestors are selected, as when attention ranks borrow remote experts.
    These subtrees also retain their own mathematical partition instead of
    inheriting a selected ancestor's mesh. Encoded weights end in the layout
    of the kernels that read them (``uniserve.loading.layout``);
    ``expert_kernels`` names those of the routed experts.

    With ``retained``, the selected modules take the tensors a worker
    published when it last loaded this model on this placement
    (``uniserve.loading.retained``) instead of reading the checkpoint; they
    are already in their final layout. ``load`` names the tensors
    (``uniserve.loading.retained`` identities) the worker reads from the
    checkpoint instead, which ``retained`` lacks: those, and what reading
    them takes (``_reads``), load as without ``retained``, and the reports
    cover them alone.
    """
    with torch.device("meta"):
        model = model_class(config)
    if not isinstance(model, nn.Module):
        raise TypeError("model constructor must return torch.nn.Module")

    known_paths = frozenset(
        path for path, _ in model.named_modules(remove_duplicate=False)
    )
    selected = {
        id(child)
        for child in select_modules(
            model, modules, exclude_modules=exclude_modules
        )
    }

    meshes = {} if meshes is None else meshes
    attention = {} if attention is None else attention
    if set(attention).difference(meshes):
        raise ValueError(
            "attention parallel settings require a mesh at the same module path"
        )

    # A mesh this rank does not participate in makes its whole subtree remote:
    # those modules stay on meta and are never materialized here.
    remote = {
        id(child)
        for path, mesh in meshes.items()
        if mesh.rank not in mesh.ranks
        for child in model.get_submodule(path).modules()
    }
    local = {
        id(child)
        for path, mesh in meshes.items()
        if mesh.rank in mesh.ranks
        for child in model.get_submodule(path).modules()
    }
    if remote.intersection(local):
        raise ValueError(
            "shared numerical modules cannot have conflicting mesh "
            "participation"
        )
    selected.difference_update(remote)
    _release_unselected(model, selected)
    excluded = select_modules(model, exclude_modules)
    for path, mesh in meshes.items():
        child = model.get_submodule(path)
        if mesh.rank in mesh.ranks:
            parallelize_(
                child,
                mesh,
                attention=attention.get(path, AttentionParallelConfig()),
                exclude=excluded,
            )
    if experts is not None:
        partition_experts(model, experts)

    # Declaring the mappings may also prune the model to the modules a
    # pipeline stage holds, so it precedes installing retained tensors too.
    declared = mapping(model)
    if retained is not None and not load:
        # Indexing the weight settings applies their module-level choices,
        # such as each projection's activation quantizer, as loading does.
        weight_options._Loader(
            model,
            checkpoint,
            (),
            device=device,
            weights=weights,
            io=io,
            devices=devices,
            selected=selected,
            known_paths=known_paths,
        ).close()
        retained.install(model, selected)
        model.eval()
        return Result(model, checkpoint, (), _origins(declared))

    parameters = {
        id(parameter)
        for child in model.modules()
        if id(child) in selected
        for parameter in child.parameters(recurse=False)
    }
    if retained is not None:
        # The retained tensors install first; loading reads the rest.
        reads = _reads(model, declared, load)
        retained.install(model, selected, skip=reads)
        parameters = {
            id(tensor)
            for name, tensor in tensors(model)
            if name in reads and id(tensor) in parameters
        }

    # Narrow each declared mapping to the selected modules: unselected
    # parameters keep their meta placeholders and their assignments drop out.
    mapped = []
    for component in declared:
        names = {
            name
            for name, parameter in component.module.named_parameters(
                remove_duplicate=False
            )
            if id(parameter) in parameters
        }
        resident = names.intersection(component.required | component.optional)
        if not resident and (
            retained is not None or id(component.module) not in selected
        ):
            continue
        if component.post_load is not None and not component.required.issubset(
            names
        ):
            raise ValueError(
                "checkpoint-derived modules must load their complete "
                "numerical inputs"
            )

        def assign(reader, mapping=component.map_weights):
            return tuple(
                value
                for value in mapping(reader)
                if id(value.target) in parameters
            )

        mapped.append(
            replace(
                component,
                map_weights=assign,
                required=component.required & names,
                optional=component.optional & names,
            )
        )

    reports = _load(
        model,
        checkpoint,
        mapped,
        device=device,
        weights=weights,
        io=io,
        devices=devices,
        excluded=declared,
        selected=selected,
        known_paths=known_paths,
    )
    finalize(model, experts=expert_kernels)
    model.eval()
    return Result(model, checkpoint, reports, _origins(declared))
