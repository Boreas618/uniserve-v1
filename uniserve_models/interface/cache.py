"""K/V cache description, unit operations and import conversion bindings.

The Rust worker owns the K/V unit pool: it allocates the planes, flags and
scales, assigns units, and copies imported pages. These functions supply
what only the numerical library knows: the pool's layout for a loaded model
(``cache_plan``), whether an attention backend reads a page shape
(``reads_pages``), the scattered unit resets (``bind_units``) and the
dtype/FP8 conversion of imported pages (``bind_conversion``).

Binding functions run once, at bind time, over Rust-owned tensors and a
Rust-owned stream; the returned per-call functions take only integers.
Importing this module does not initialize CUDA.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import TYPE_CHECKING

import torch
from torch import nn

from uniserve_models.cache.units import Planes, UnitPool, plan_units
from uniserve_models.quantization import Quantizer

from .errors import binding

if TYPE_CHECKING:
    from uniserve_models.model import CausalLM

__all__ = [
    "bind_conversion",
    "bind_units",
    "cache_plan",
    "functions",
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
    from uniserve_models.model import CausalLM

    module = loaded if isinstance(loaded, nn.Module) else loaded.model
    found = [child for child in module.modules() if isinstance(child, CausalLM)]
    if len(found) != 1:
        raise ValueError("the K/V cache requires exactly one text model")
    return found[0]


def _storage(kv_dtype: str) -> tuple[torch.dtype | None, bool]:
    """Resolve a configured K/V storage name as ``(dtype, fp8)``.

    An empty name stores each layer's compute dtype. FP8 storage is the
    compute dtype plus a per-block FP8 quantizer.
    """
    if not kv_dtype:
        return None, False
    dtype = getattr(torch, kv_dtype.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown K/V cache dtype {kv_dtype!r}")
    if dtype is torch.float8_e4m3fn:
        return None, True
    return dtype, False


def _plan(model: CausalLM, block_size: int, kv_dtype: str) -> Planes:
    """Plan the pool with ``block_size``-token pages of the widest rows."""
    layers = model.cache_config.layers
    if not layers:
        raise ValueError(
            "the worker K/V protocol requires resident cache layers"
        )
    dtype, fp8 = _storage(kv_dtype)
    return plan_units(
        model.cache_config,
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


def cache_plan(loaded, block_size: int, kv_dtype: str) -> dict:
    """Describe the rank's K/V unit pool for ``block_size``-token pages.

    ``block_size`` is the page size of the group with the widest token rows;
    every other group's page holds as many tokens as fit the same plane.
    ``kv_dtype`` is ``""`` for each layer's compute dtype, a dense dtype
    name, or ``"float8_e4m3fn"`` for FP8 storage.

    Returns:
        ``{"columns", "plane_bytes", "quantized", "unit_bytes", "dtype",
        "groups"}``: ``dtype`` is the published stored dtype and each group
        is ``{"window"`` (-1 for full attention), ``"page_tokens",
        "units_per_page", "layers"`` (resident names in column order),
        ``"layer_ids"`` (their global cache-layer ids), ``"axis_layers"``
        (every global id of the group across stages, ascending),
        ``"num_kv_heads"`` (local), ``"total_kv_heads", "kv_head_offset",
        "head_dim", "dtype"`` (stored element dtype), ``"compute_dtype"}``.

    Raises:
        ValueError: As ``plan_units``; when the model has no resident cache
            layer; or when a group's local heads are not consecutive.
    """
    model = _causal_lm(loaded)
    planes = _plan(model, block_size, kv_dtype)
    names = model.backbone.cache_names
    layouts = model.cache_config.layers
    axes = _group_layers(model, planes)

    groups = []
    for group, axis in zip(planes.groups, axes, strict=True):
        layout = layouts[group.layers[0]]
        heads = layout.head_indices
        if heads != tuple(range(heads[0], heads[0] + len(heads))):
            raise ValueError(
                "the worker K/V protocol requires consecutive logical cache "
                "heads"
            )
        groups.append(
            {
                "window": -1 if group.window is None else int(group.window),
                "page_tokens": int(group.page_tokens),
                "units_per_page": int(group.units_per_page),
                "layers": list(group.layers),
                "layer_ids": [names.index(name) for name in group.layers],
                "axis_layers": list(axis),
                "num_kv_heads": len(heads),
                "total_kv_heads": int(layout.num_kv_heads),
                "kv_head_offset": int(heads[0]),
                "head_dim": int(group.head_dim),
                "dtype": _dtype_name(group.dtype),
                "compute_dtype": _dtype_name(layout.compute_dtype),
            }
        )

    # The published dtype names the stored element type: FP8 for encoded
    # storage, otherwise the configured or compute dtype.
    dtype, fp8 = _storage(kv_dtype)
    stored = (
        torch.float8_e4m3fn
        if fp8
        else dtype
        if dtype is not None
        else layouts[planes.groups[0].layers[0]].compute_dtype
    )
    return {
        "columns": int(planes.columns),
        "plane_bytes": int(planes.plane_bytes),
        "quantized": bool(planes.quantized),
        "unit_bytes": int(planes.unit_bytes),
        "dtype": _dtype_name(stored),
        "groups": groups,
    }


def reads_pages(loaded, backend: str, plan: Mapping) -> bool:
    """Report whether ``backend`` reads every resident layer's pages.

    ``plan`` is a ``cache_plan`` result. Each resident cache layer is read
    by the attention layer naming it; the backend must serve that reader's
    head shape, window and compute dtype on its group's pages.

    Raises:
        ValueError: A resident cache layer has no attention reader.
    """
    from uniserve_models.backends.attention import CachePages, resolve
    from uniserve_models.nn.attention import Attention

    model = _causal_lm(loaded)
    readers = {
        module.cache_name: module
        for module in model.modules()
        if isinstance(module, Attention) and module.cache_name is not None
    }
    layouts = model.cache_config.layers
    if any(name not in readers for name in layouts):
        raise ValueError(
            "every resident cache layer requires an attention layer reading it"
        )
    device = next(model.parameters()).device
    provider = resolve(backend, device=device)
    quantized = bool(plan["quantized"])
    for group in plan["groups"]:
        pages = CachePages(
            int(group["page_tokens"]),
            getattr(torch, str(group["dtype"])),
            quantized,
        )
        for name in group["layers"]:
            reader = readers[name]
            if not provider.reads_pages(
                num_heads=reader.local_heads,
                num_kv_heads=reader.local_kv_heads,
                head_dim=reader.head_dim,
                window=reader.window,
                dtype=layouts[name].compute_dtype,
                pages=pages,
            ):
                return False
    return True


def _stream_scope(stream: int, device: torch.device) -> Callable:
    """Return a factory of scopes enqueuing on the Rust-owned ``stream``.

    Host tensors need no stream. The external stream is wrapped once, at
    bind time; each call enters a fresh scope.
    """
    from contextlib import nullcontext

    if device.type != "cuda":
        return nullcontext
    external = torch.cuda.ExternalStream(stream, device=device)
    return lambda: torch.cuda.stream(external)


def _tensor(value) -> torch.Tensor:
    """View a worker-owned DLPack tensor (or a torch tensor) in torch."""
    return (
        value if isinstance(value, torch.Tensor) else torch.from_dlpack(value)
    )


def _fields(cache: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Select the unit fields of a cache tensor map, viewed in torch."""
    return {
        str(name): _tensor(tensor)
        for name, tensor in cache.items()
        if str(name).split(".", 1)[0] in {"key", "value"}
    }


def bind_units(
    cache: Mapping[str, torch.Tensor], units: torch.Tensor, stream: int
) -> dict[str, Callable[[int], None]]:
    """Bind scattered unit resets over Rust-owned unit fields.

    Args:
        cache: The pool's unit fields (``{key,value}.values``,
            ``{key,value}.initialized`` and, for FP8, ``{key,value}.scale``,
            each with a ``[columns, num_units]`` leading layout); other
            entries are ignored.
        units: An int64 ``[num_units]`` column on the pool's device. Rust
            writes the unit ids into ``units[:count]`` before each call.
        stream: The stream handle every reset is enqueued on.

    Returns:
        ``{"zero", "recycle", "mark_initialized"}``, each taking ``count``:
        ``zero`` clears values and flags and resets scales to one;
        ``recycle`` clears flags and resets scales (the reset of a quantized
        pool of one element type; Rust zeroes mixed-type pools and skips
        dense pools of one type); ``mark_initialized`` sets both fields'
        flags. Each is one indexed fill per field, without a host read.
    """
    fields = _fields(cache)
    for name in ("key.values", "value.values"):
        if name not in fields:
            raise ValueError(f"the unit fields lack {name!r}")
    units = _tensor(units)
    scope = _stream_scope(int(stream), units.device)

    def fill(count: int, selected: Sequence[str], value) -> None:
        index = units[: int(count)]
        with scope():
            for name in selected:
                fields[name].index_fill_(1, index, value)

    values = [name for name in fields if name.endswith(".values")]
    flags = [name for name in fields if name.endswith(".initialized")]
    scales = [name for name in fields if name.endswith(".scale")]

    def zero(count: int) -> None:
        fill(count, values + flags, 0)
        fill(count, scales, 1)

    def recycle(count: int) -> None:
        fill(count, flags, 0)
        fill(count, scales, 1)

    def mark_initialized(count: int) -> None:
        fill(count, flags, 1)

    return binding(
        {
            "zero": zero,
            "recycle": recycle,
            "mark_initialized": mark_initialized,
        }
    )


def bind_conversion(
    plan: Mapping,
    cache: Mapping[str, torch.Tensor],
    scratch: torch.Tensor,
    elements: int,
    scales: int,
    stream: int,
) -> dict[str, Callable[..., None]]:
    """Bind the dtype and FP8 conversion of imported pages.

    Args:
        plan: The ``cache_plan`` result plus ``"num_units"``.
        cache: The pool's unit fields, as ``UnitPool.from_plan`` takes them.
        scratch: uint8 ``[20 * elements + 8 * scales]``: raw keys at bytes
            ``[0, 8E)``, raw values at ``[8E, 16E)``, the FP32 conversion
            scratch at ``[16E, 20E)`` and FP32 source scales at ``[20E, 20E +
            8S)``, where ``E`` is ``elements`` (the largest group page's
            elements) and ``S`` is ``scales`` (its scale entries).
        elements: ``E`` above.
        scales: ``S`` above.
        stream: The stream conversions are enqueued on.

    Returns:
        ``{"convert": convert}``. ``convert(group, source_dtype,
        compute_dtype, count, offset, source_offset, source_page_tokens,
        scale_heads, pages, units)`` reads ``[count, layers, heads, dim]``
        raw keys and values of ``source_dtype`` from the scratch (with FP8,
        also ``[pages, 2, layers, scale groups]`` scales) and writes the
        destination page's ``units`` at token ``offset`` with
        ``copy_converted``.
    """
    from uniserve_models.cache.conversion import copy_converted

    elements, scales = int(elements), int(scales)
    scratch = _tensor(scratch)
    if scratch.dtype != torch.uint8 or scratch.numel() < (
        20 * elements + 8 * scales
    ):
        raise ValueError("the conversion scratch disagrees with its layout")
    pool = UnitPool.from_plan(plan, _fields(cache))
    raw = scratch[: 16 * elements].view(2, 8 * elements)
    values = scratch[16 * elements : 20 * elements].view(torch.float32)
    scale_buffer = scratch[20 * elements : 20 * elements + 8 * scales].view(
        torch.float32
    )
    scope = _stream_scope(int(stream), scratch.device)

    def convert(
        group: int,
        source_dtype: str,
        compute_dtype: str,
        count: int,
        offset: int,
        source_offset: int,
        source_page_tokens: int,
        scale_heads: int,
        pages: int,
        units: Sequence[int],
    ) -> None:
        cache_group = pool.groups[int(group)]
        dtype = getattr(torch, source_dtype)
        layers, heads = len(cache_group.layers), cache_group.num_kv_heads
        shape = (int(count), layers, heads, cache_group.head_dim)
        nbytes = prod(shape) * dtype.itemsize
        source = (
            raw[0, :nbytes].view(dtype).view(shape),
            raw[1, :nbytes].view(dtype).view(shape),
        )
        source_scales = None
        if int(scale_heads) > 0:
            first = pool.head_offset(cache_group.layers[0])
            groups = -(-(first + heads) // int(scale_heads)) - first // int(
                scale_heads
            )
            source_scales = scale_buffer[
                : int(pages) * 2 * layers * groups
            ].view(int(pages), 2, layers, groups)
        with scope():
            copy_converted(
                pool,
                int(group),
                source,
                source_scales,
                values,
                (raw[0], raw[1]),
                tuple(int(unit) for unit in units),
                int(offset),
                int(source_offset),
                int(source_page_tokens),
                int(scale_heads),
                getattr(torch, compute_dtype),
            )

    return binding({"convert": convert})


def pool(layout: Mapping, views: Mapping[str, torch.Tensor]) -> UnitPool:
    """Bind the attention layers' view of the Rust-owned K/V unit pool.

    Args:
        layout: The ``cache_plan`` result plus ``"num_units"``.
        views: The pool's unit fields; other entries (block tables) are
            ignored.

    Returns:
        The pool object an execution context binds its attention layers to
        (``component.prepare``'s ``cache``), an opaque handle to the worker.
    """
    return UnitPool.from_plan(layout, _fields(views))


def functions() -> dict[str, Callable]:
    """Return this module's interface-table entries."""
    return {
        "model.cache_plan": cache_plan,
        "model.reads_pages": reads_pages,
        "cache.bind_units": bind_units,
        "cache.bind_conversion": bind_conversion,
        "cache.pool": pool,
    }
