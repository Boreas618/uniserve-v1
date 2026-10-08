"""K/V cache planning facts of a loaded causal language model.

The worker chooses the page size and the unit count; this module reports
the numerical layout one page size implies and whether every resident
attention layer can read it. The storage runtime extends this module with
the bindings of the planned pool.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from uniserve_models.model import CausalLM
from uniserve_models.nn.attention import Attention
from uniserve_models.quantization import Quantizer
from uniserve_models.runtime.prefix_cache import Planes, plan_units

from .errors import SetupError


def functions() -> dict[str, Callable[..., Any]]:
    return {"model.cache_plan": cache_plan}


def _text(module) -> CausalLM | None:
    found = {
        id(child): child
        for child in module.modules()
        if isinstance(child, CausalLM)
    }
    if len(found) > 1:
        raise SetupError("model has more than one CausalLM capability module")
    return next(iter(found.values()), None)


def storage(kv_dtype: str) -> tuple[torch.dtype | None, bool]:
    """Resolve the K/V storage choice as ``(dtype, fp8)``.

    ``""`` stores each layer's compute dtype; ``float8_e4m3fn`` stores the
    compute dtype encoded with a per-block FP8 quantizer.
    """
    if not kv_dtype:
        return None, False
    dtype = getattr(torch, kv_dtype.removeprefix("torch."))
    if dtype is torch.float8_e4m3fn:
        return None, True
    return dtype, False


def _plan(model: CausalLM, block_size: int, kv_dtype: str) -> Planes:
    layers = model.cache_config.layers
    if not layers:
        raise SetupError(
            "the worker K/V protocol requires resident cache layers"
        )
    dtype, fp8 = storage(kv_dtype)
    return plan_units(
        model.cache_config,
        block_size=block_size,
        dtype=dtype,
        quantization={name: Quantizer("fp8", axis=0) for name in layers}
        if fp8
        else None,
    )


def _readable(model: CausalLM, planes: Planes, backend) -> bool:
    """Whether every resident layer's attention reads these pages."""
    from uniserve_models.backends.attention import CachePages

    readers = {
        module.cache_name: module
        for module in model.modules()
        if isinstance(module, Attention) and module.cache_name is not None
    }
    layouts = model.cache_config.layers
    if any(name not in readers for name in layouts):
        raise SetupError(
            "every resident cache layer requires an attention layer reading it"
        )
    for group in planes.groups:
        pages = CachePages(group.page_tokens, group.dtype, planes.quantized)
        for name in group.layers:
            reader = readers[name]
            if not backend.reads_pages(
                num_heads=reader.local_heads,
                num_kv_heads=reader.local_kv_heads,
                head_dim=reader.head_dim,
                window=reader.window,
                dtype=layouts[name].compute_dtype,
                pages=pages,
            ):
                return False
    return True


def cache_plan(loaded, block_size: int, kv_dtype: str) -> dict | None:
    """Describe the unit pool that ``block_size``-token pages imply.

    Returns ``None`` for a model without a causal language model. Groups
    carry their resident layers in configuration order, their global cache
    layer ids, and this rank's consecutive share of each layer's K/V heads.

    Raises:
        SetupError: The model has no resident cache layer, a cache layer has
            no attention reader, or a group's heads are not consecutive.
    """
    model = _text(loaded.module)
    if model is None:
        return None
    block_size = int(block_size)
    kv_dtype = str(kv_dtype)
    planes = _plan(model, block_size, kv_dtype)
    names = model.backbone.cache_names
    layouts = model.cache_config.layers
    _, fp8 = storage(kv_dtype)

    groups = []
    for group in planes.groups:
        layout = layouts[group.layers[0]]
        heads = layout.head_indices
        if heads != tuple(range(heads[0], heads[0] + len(heads))):
            raise SetupError(
                "the worker K/V protocol requires consecutive logical cache "
                "heads"
            )
        groups.append(
            {
                "window": -1 if group.window is None else group.window,
                "page_tokens": group.page_tokens,
                "units_per_page": group.units_per_page,
                "layers": list(group.layers),
                "layer_ids": [names.index(name) for name in group.layers],
                "num_kv_heads": len(heads),
                "total_kv_heads": layout.num_kv_heads,
                "kv_head_offset": heads[0],
                "head_dim": group.head_dim,
                "dtype": str(group.dtype).removeprefix("torch."),
            }
        )

    # The published dtype names the stored element type: FP8 for encoded
    # storage, otherwise the configured or compute dtype.
    dtype, _ = storage(kv_dtype)
    stored = (
        torch.float8_e4m3fn
        if fp8
        else dtype
        if dtype is not None
        else layouts[planes.groups[0].layers[0]].compute_dtype
    )
    return {
        "readable": _readable(model, planes, loaded.attention),
        "columns": planes.columns,
        "plane_bytes": planes.plane_bytes,
        "quantized": planes.quantized,
        "unit_bytes": planes.unit_bytes,
        "stored_dtype": str(stored).removeprefix("torch."),
        "groups": groups,
    }
