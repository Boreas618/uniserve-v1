"""Translate launch precision selectors into a weight configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from importlib import import_module
from typing import Any

import torch

from uniserve.quantization import QuantizationConfig, Quantizer

from . import weights

# Keys a launch's precision selectors may carry.
_KEYS = {
    "mode",
    "quant_method",
    "components",
    "ignored_layers",
}

# Formats a generic ``quant_method`` applies to every module.
_GENERIC = {"unquantized", "fp8", "mxfp8", "nvfp4"}


def weight_config(
    source, options: Mapping[str, Any], model_dtype: str
) -> weights.Config:
    """Translate launch precision selectors into the public loading value.

    ``options`` is the launch's precision configuration. Exactly one source
    selects the base weight configuration: ``components`` selectors passed to
    the model package's ``weight_config`` factory (with ``mode`` or
    ``quant_method`` as its preset, the factory's own default without one);
    a named precision in ``source.precisions``; a ``quant_method`` among the
    generic formats, applied to every module; or, with no selector, the
    checkpoint's own weights. ``ignored_layers`` then maps each listed module
    path to no quantization, and the dtype comes from ``model_dtype``.

    Raises:
        ValueError: An unknown key, both ``mode`` and ``quant_method``,
            numerical selectors for a checkpoint whose format owns its
            numerics, component selectors for a model without a
            ``weight_config`` factory, or an unknown precision.
        TypeError: ``components`` is not a mapping or ``ignored_layers`` is
            not a list of strings.
    """
    unknown = options.keys() - _KEYS
    if unknown:
        raise ValueError(
            f"quantization_config has unknown fields {sorted(unknown)}"
        )
    if "mode" in options and "quant_method" in options:
        raise ValueError(
            "precision mode and quant_method are mutually exclusive"
        )

    checkpoint_format = source.checkpoint_format
    numerical_overrides = options.keys() & {
        "mode",
        "quant_method",
        "components",
        "ignored_layers",
    }
    if checkpoint_format is not None and numerical_overrides:
        raise ValueError(
            f"checkpoint format {checkpoint_format!r} owns its numerical "
            "configuration; remove quantization_config"
        )

    selected = options.get("mode", options.get("quant_method"))
    components = options.get("components", {})
    if not isinstance(components, Mapping):
        raise TypeError("precision components must be an object")

    package = import_module(source.model_class.__module__.rsplit(".", 1)[0])
    if components:
        factory = getattr(package, "weight_config", None)
        if factory is None:
            raise ValueError(
                "this model exposes complete precision presets "
                "without component selectors"
            )
        # Without a selector the factory applies its own default preset.
        preset = {} if selected is None else {"preset": selected}
        result = factory(source.model, **preset, **components)
    elif selected is None:
        result = source.weights
    elif selected in source.precisions:
        result = source.precisions[selected]
    # The quantization mapping resolves by longest module-path prefix, so the
    # empty key applies the generic format to every module.
    elif "quant_method" in options and selected in _GENERIC:
        quantizer = (
            None
            if selected == "unquantized"
            else Quantizer(selected, axis=0 if selected == "fp8" else None)
        )
        result = replace(
            source.weights,
            quantization={
                "": None
                if quantizer is None
                else QuantizationConfig(quantizer, quantizer)
            },
        )
    else:
        raise ValueError(
            f"unknown precision {selected!r}; "
            f"choose from {tuple(source.precisions)}"
        )

    ignored = options.get("ignored_layers", ())
    if not isinstance(ignored, (tuple, list)) or any(
        not isinstance(path, str) for path in ignored
    ):
        raise TypeError("ignored_layers must contain numerical module paths")

    return replace(
        result,
        dtype=getattr(torch, model_dtype),
        quantization={**result.quantization, **dict.fromkeys(ignored)},
    )
