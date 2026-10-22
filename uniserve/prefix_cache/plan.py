"""The prefix cache's unit pool as a loaded model needs it.

The Rust worker owns the K/V unit pool: it allocates the planes, flags and
scales, assigns units, and copies imported pages. These functions supply
what only the numerical library knows: the pool's layout for a loaded model
(``cache_plan``, a :class:`CachePlan` the worker reads column by column and
passes back unchanged), whether the layers' readers read a page shape
(``reads_pages``) and the layers' view of the pool (``pool``). Cache layers
of a type indexed by request slot (``StateConfig.indexing == "states"``)
stay out of the pool: the plan lists their fields (``state_fields``), which
the worker allocates per request slot and binds with the pool.
The scattered unit resets are :class:`uniserve.prefix_cache.units.UnitResets`
and the dtype/FP8 conversion of imported pages is
:class:`uniserve.prefix_cache.conversion.PageConversion`. Importing this
module does not initialize CUDA.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from uniserve.prefix_cache import mha, slots
from uniserve.prefix_cache.state import Config, StateConfig
from uniserve.prefix_cache.units import (
    Planes,
    UnitPool,
    named_fields,
    plan_units,
)
from uniserve.quantization import Quantizer

if TYPE_CHECKING:
    from uniserve.models import CausalLM

__all__ = [
    "CachePlan",
    "cache_plan",
    "pool",
    "reads_pages",
]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _causal_lm(loaded) -> CausalLM:
    """Return the loaded model's unique ``CausalLM`` module.

    ``loaded`` is the handle the loading interface returned: either the
    model module itself or an object whose ``model`` attribute is it.
    """
    from uniserve.models import CausalLM

    module = loaded if isinstance(loaded, nn.Module) else loaded.model
    found = [child for child in module.modules() if isinstance(child, CausalLM)]
    if len(found) != 1:
        raise ValueError("the K/V cache requires exactly one text model")
    return found[0]


def _storage(prefix_cache_dtype: str) -> tuple[torch.dtype | None, bool]:
    """Resolve a configured K/V storage name as ``(dtype, fp8)``.

    An empty name stores each layer's compute dtype. FP8 storage is the
    compute dtype plus a per-block FP8 quantizer.
    """
    if not prefix_cache_dtype:
        return None, False
    dtype = getattr(torch, prefix_cache_dtype.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown K/V cache dtype {prefix_cache_dtype!r}")
    if dtype is torch.float8_e4m3fn:
        return None, True
    return dtype, False


def _paged(model: CausalLM) -> dict[str, mha.Config]:
    """The model's paged cache layers, which the unit pool holds.

    Raises:
        TypeError: A paged layer's type does not subclass ``mha.Config``.
    """
    paged = {}
    for name, layout in model.cache_config.layers.items():
        if layout.indexing != "tokens":
            continue
        if not isinstance(layout, mha.Config):
            raise TypeError("a paged cache layer type subclasses mha.Config")
        paged[name] = layout
    return paged


def _states(model: CausalLM) -> dict[str, StateConfig]:
    """The model's cache layers kept per request slot."""
    return {
        name: layout
        for name, layout in model.cache_config.layers.items()
        if layout.indexing == "states"
    }


def _plan(model: CausalLM, block_size: int, prefix_cache_dtype: str) -> Planes:
    """Plan the pool with ``block_size``-token pages of the widest rows."""
    layers = _paged(model)
    if not layers:
        raise ValueError(
            "the worker K/V protocol requires resident paged cache layers"
        )
    dtype, fp8 = _storage(prefix_cache_dtype)
    return plan_units(
        Config(layers),
        block_size=block_size,
        dtype=dtype,
        quantization={name: Quantizer("fp8", axis=0) for name in layers}
        if fp8
        else None,
    )


def _group_layers(model: CausalLM, planes: Planes) -> list[list[int]]:
    """Return every group's global cache-layer ids across pipeline stages.

    A logical cache layer joins the group whose history window, logical
    K/V heads and head width it shares; the decoder records these for every
    layer, including those resident on other stages. A group's layers in
    ascending global order form its export layer axis.

    Raises:
        ValueError: A resident group's layers are not the logical layers of
            one shape, or a logical layer belongs to several groups.
    """
    backbone = model.backbone
    names = backbone.cache_names
    layouts = model.cache_config.layers
    result = []
    for group in planes.groups:
        layout = layouts[group.layers[0]]
        key = (layout.window, layout.num_kv_heads, layout.head_dim)
        members = [
            index
            for index, layer in enumerate(backbone.cache_layers)
            if (layer.window, layer.num_kv_heads, layer.head_dim) == key
        ]
        local = {names.index(name) for name in group.layers}
        if not local.issubset(members):
            raise ValueError(
                "resident cache layers disagree with their logical geometry"
            )
        result.append(members)
    if len({layer for members in result for layer in members}) != sum(
        map(len, result)
    ):
        raise ValueError(
            "logical cache layers belong to several resident groups"
        )
    return result


@dataclass(frozen=True)
class CachePlan:
    """A rank's K/V unit pool for one block size, read column by column.

    ``columns``, ``plane_bytes``, ``quantized``, ``unit_bytes`` and
    ``dtype`` describe the pool. Every other method returns one value per
    cache group, in table order, except ``layers`` and ``layer_ids``, which
    concatenate the groups' resident layers (``layer_counts`` of them per
    group, in column order).
    """

    planes: Planes
    # Each resident cache layer's layout.
    configs: Mapping[str, mha.Config]
    # The published stored element type.
    stored: torch.dtype
    # Each group's resident layers as global cache-layer ids.
    ids: tuple[tuple[int, ...], ...]
    # Where each group's resident layers lie in the group's transfer layer
    # axis (every stage's layers of the group in ascending global order):
    # the offset of their consecutive run and the axis length.
    axes: tuple[tuple[int, int], ...]
    # Each resident cache layer kept per request slot, by name.
    states: Mapping[str, StateConfig]

    def _layouts(self) -> list[mha.Config]:
        """Each group's layer layout; a group's layers share one."""
        return [self.configs[group.layers[0]] for group in self.planes.groups]

    def columns(self) -> int:
        """Columns per unit."""
        return int(self.planes.columns)

    def plane_bytes(self) -> int:
        """Bytes of one ``(column, field)`` plane of one unit."""
        return int(self.planes.plane_bytes)

    def quantized(self) -> bool:
        """Whether the planes hold FP8 values with per-plane scales."""
        return bool(self.planes.quantized)

    def unit_bytes(self) -> int:
        """Bytes one unit occupies: its planes, flags and FP8 scales."""
        return int(self.planes.unit_bytes)

    def dtype(self) -> str:
        """The published stored element type's name."""
        return _dtype_name(self.stored)

    def windows(self) -> list[int]:
        """Each group's history window in tokens; -1 for full attention."""
        return [
            -1 if group.window is None else int(group.window)
            for group in self.planes.groups
        ]

    def block_sizes(self) -> list[int]:
        """Tokens of each group's logical page, its block size."""
        return [int(group.block_size) for group in self.planes.groups]

    def units_per_page(self) -> list[int]:
        """Units each group's logical page occupies."""
        return [int(group.units_per_page) for group in self.planes.groups]

    def layer_counts(self) -> list[int]:
        """Resident layers of each group."""
        return [len(group.layers) for group in self.planes.groups]

    def layers(self) -> list[str]:
        """Every group's resident layer names, concatenated."""
        return [name for group in self.planes.groups for name in group.layers]

    def layer_ids(self) -> list[int]:
        """Every group's resident global cache-layer ids, concatenated."""
        return [layer for ids in self.ids for layer in ids]

    def axis_offsets(self) -> list[int]:
        """Where each group's resident run starts in its transfer axis."""
        return [offset for offset, _ in self.axes]

    def axis_sizes(self) -> list[int]:
        """Layers of each group's transfer axis across pipeline stages."""
        return [size for _, size in self.axes]

    def num_kv_heads(self) -> list[int]:
        """K/V heads each group stores per layer on this rank."""
        return [len(layout.head_indices) for layout in self._layouts()]

    def total_kv_heads(self) -> list[int]:
        """Logical K/V heads of each group's layers."""
        return [int(layout.num_kv_heads) for layout in self._layouts()]

    def kv_head_offsets(self) -> list[int]:
        """The first logical K/V head each group stores on this rank."""
        return [int(layout.head_indices[0]) for layout in self._layouts()]

    def head_dims(self) -> list[int]:
        """Each group's head width."""
        return [int(group.head_dim) for group in self.planes.groups]

    def dtypes(self) -> list[str]:
        """The stored element type of each group's planes."""
        return [_dtype_name(group.dtype) for group in self.planes.groups]

    def compute_dtypes(self) -> list[str]:
        """The compute element type of each group's layers."""
        return [_dtype_name(layout.compute_dtype) for layout in self._layouts()]

    def state_fields(self) -> str:
        """The fields of every layer kept per request slot, as JSON: a list
        of ``{"name": "layer/field", "dtype", "shape"}``, ``shape`` one
        slot's extent; the worker allocates each ``[slots + 1, *shape]``.
        """  # noqa: D205
        return json.dumps(
            [
                {
                    "name": f"{name}/{field}",
                    "dtype": _dtype_name(dtype),
                    "shape": list(shape),
                }
                for name, layout in self.states.items()
                for field, (shape, dtype) in slots.fields(layout).items()
            ]
        )


def cache_plan(loaded, block_size: int, prefix_cache_dtype: str) -> CachePlan:
    """Describe the rank's K/V unit pool for ``block_size``-token pages.

    ``block_size`` is the block size of the group with the widest token rows;
    every other group's page holds as many tokens as fit the same plane.
    ``prefix_cache_dtype`` is ``""`` for each layer's compute dtype, a dense
    dtype name, or ``"float8_e4m3fn"`` for FP8 storage.

    Raises:
        ValueError: As ``plan_units``; when the model has no resident cache
            layer; when a group's local heads are not consecutive; or when
            a group's resident layers are not one consecutive run of its
            transfer axis.
    """
    model = _causal_lm(loaded)
    planes = _plan(model, block_size, prefix_cache_dtype)
    names = model.backbone.cache_names
    layouts = _paged(model)

    ids, axes = [], []
    for group, axis in zip(
        planes.groups, _group_layers(model, planes), strict=True
    ):
        heads = layouts[group.layers[0]].head_indices
        if heads != tuple(range(heads[0], heads[0] + len(heads))):
            raise ValueError(
                "the worker K/V protocol requires consecutive logical cache "
                "heads"
            )
        resident = tuple(names.index(name) for name in group.layers)
        offset = axis.index(resident[0])
        if tuple(axis[offset : offset + len(resident)]) != resident:
            raise ValueError(
                "cache group layers must be one consecutive transfer run"
            )
        ids.append(resident)
        axes.append((offset, len(axis)))

    # The published dtype names the stored element type: FP8 for encoded
    # storage, otherwise the configured or compute dtype.
    dtype, fp8 = _storage(prefix_cache_dtype)
    stored = (
        torch.float8_e4m3fn
        if fp8
        else dtype
        if dtype is not None
        else layouts[planes.groups[0].layers[0]].compute_dtype
    )
    configs = {
        name: layouts[name] for group in planes.groups for name in group.layers
    }
    return CachePlan(
        planes, configs, stored, tuple(ids), tuple(axes), _states(model)
    )


def reads_pages(loaded, backend: str, plan: CachePlan) -> bool:
    """Report whether every resident paged layer's reader reads its pages.

    ``plan`` is a ``cache_plan`` result. Each resident paged cache layer is
    read by the module owning it. An attention layer reads through
    ``backend``, which must serve the layer's head shape and window on its
    group's pages, in the dtype the layer computes in over them
    (``CachePages.compute_dtype``); another module answers through its own
    ``reads_pages(pages, dtype)``.

    Raises:
        ValueError: A resident paged cache layer has no owning module, or
            its owner is not attention and declares no ``reads_pages``.
    """
    from uniserve.backends.attention import CachePages, resolve
    from uniserve.nn.attention import Attention

    model = _causal_lm(loaded)
    readers: dict[str, nn.Module] = {
        str(getattr(module, "cache_name")): module
        for module in model.modules()
        if getattr(module, "cache_name", None) is not None
    }
    layouts = _paged(model)
    if any(
        name not in readers
        or not (
            isinstance(readers[name], Attention)
            or hasattr(readers[name], "reads_pages")
        )
        for name in layouts
    ):
        raise ValueError(
            "every resident paged cache layer requires a module reading it: "
            "attention, or a module declaring reads_pages"
        )
    device = next(model.parameters()).device
    provider = resolve(backend, device=device)
    for group in plan.planes.groups:
        pages = CachePages(
            int(group.block_size), group.dtype, plan.planes.quantized
        )
        for name in group.layers:
            reader = readers[name]
            dtype = pages.compute_dtype(layouts[name].compute_dtype)
            if not isinstance(reader, Attention):
                if not getattr(reader, "reads_pages")(pages, dtype):
                    return False
                continue
            if not provider.reads_pages(
                num_heads=reader.local_heads,
                num_kv_heads=reader.local_kv_heads,
                head_dim=reader.head_dim,
                window=reader.window,
                dtype=dtype,
                pages=pages,
            ):
                return False
    return True


def pool(
    plan: CachePlan, num_units: int, names: Sequence[str], *fields: torch.Tensor
) -> UnitPool:
    """Bind the cache layers' view of the Rust-owned unit pool and banks.

    Args:
        plan: The ``cache_plan`` result.
        num_units: The units the fields hold, sentinel included.
        names: The name of each of ``fields``: ``{key,value}.values``,
            ``{key,value}.initialized`` and, for FP8, ``{key,value}.scale``,
            then the request state banks ``"layer/field"`` of
            ``state_fields``.
        fields: The pool's unit fields, each with a ``[columns, num_units]``
            leading layout, then the banks, each ``[slots + 1, ...]``.

    Returns:
        The pool object an execution context binds its cache layers to, an
        opaque handle to the worker.

    Raises:
        ValueError: A field or bank disagrees with the plan's layout.
    """
    named = named_fields(names, fields)
    banks = {name: field for name, field in named.items() if "/" in name}
    return UnitPool(
        plan.planes,
        {name: field for name, field in named.items() if name not in banks},
        plan.configs,
        num_units=int(num_units),
        states=slots.bind(plan.states, banks),
    )
