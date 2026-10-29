"""The function and format identities of a model's numerical components.

Every derived state is keyed by what determines its bytes
(``specs/online-operations.md``, section 3.2), and only by that, so that a
change of the code that leaves what a component computes as it was keeps
the weights, cache pages and encoder outputs it derived:

- Every module class of the computation library and of the model packages
  declares its function: ``function = "<component>/<revision>"``, a stable
  name and a revision that changes whenever what the component computes
  changes, and only then. Fusing kernels, tiling, selecting another kernel
  or another reduction order is an implementation of the same function and
  keeps its revision. Checkpoint derivations declare their own functions
  and inputs (``uniserve.loading.weights.Derivation``); assignment and
  quantization revisions belong to their shared implementations.
- A module's format is that of the tensors it holds itself: each
  parameter's and buffer's element type, or a quantized tensor's encoding
  and the layout its kernels read.
- A module's identity digests its function, declared mathematical
  arguments, tensor formats and children's identities. Cache-writing
  phases additionally identify their upstream computations and individual
  tensor loading inputs; downstream heads contribute no cache dependency.
  Weight loading (:func:`loading`) independently keys retained weights.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from importlib import metadata

import torch
from torch import nn

from uniserve.errors import SetupError

# What the loader does to every tensor it reads: slicing a checkpoint
# tensor to the rank's region, casting it to the parameter's element type
# and copying it into place.
LOADER = "loading.loader/1"


@functools.cache
def _dependencies() -> dict[str, str]:
    """The installed version of each distribution uniserve depends on, by
    the top-level package it provides.
    """  # noqa: D205
    required = {
        re.split(r"[\s;<>=!~\[(]", requirement, maxsplit=1)[0]
        .lower()
        .replace("_", "-")
        for requirement in metadata.requires("uniserve") or ()
    }
    versions = {}
    for package, distributions in metadata.packages_distributions().items():
        for distribution in distributions:
            if distribution.lower().replace("_", "-") in required:
                versions[package] = metadata.version(distribution)
    return versions


def function(cls: type) -> str:
    """The function ``cls`` declares, or one of its bases does.

    A module of PyTorch's computes PyTorch's documented function. A module
    of a library uniserve depends on (a model package composing a
    library's network, as MiniMax-H3's audio decoder composes diffusers'
    BigVGAN) computes what the library's installed version computes: its
    function is the class's qualified name at that version, which a
    library upgrade changes.

    Raises:
        SetupError: No class of ``cls``'s hierarchy declares one.
    """
    # PyTorch's own modules compute what PyTorch documents them to compute.
    if cls.__module__.startswith("torch.nn"):
        return f"torch.nn.{cls.__qualname__}/1"
    version = _dependencies().get(cls.__module__.split(".")[0])
    if version is not None and "function" not in vars(cls):
        return f"{cls.__module__}.{cls.__qualname__}/{version}"
    declared = getattr(cls, "function", None)
    if not isinstance(declared, str) or "/" not in declared:
        raise SetupError(
            f"{cls.__module__}.{cls.__qualname__} declares no function: a "
            'numerical module states `function = "<component>/<revision>"`'
        )
    return declared


def tensor_format(tensor: torch.Tensor) -> dict[str, object]:
    """What a tensor's bytes encode its values as: its element type, or a
    quantized tensor's encoding and the layout its kernels read.
    """  # noqa: D205
    from uniserve.quantization import QuantizedTensor

    if isinstance(tensor, QuantizedTensor):
        quantizer = tensor.quantizer
        return {
            "encoding": quantizer.format,
            "axis": quantizer.axis,
            "calibrated_scale": quantizer.calibrated_scale,
            "scale_layout": tensor.scale_layout.value,
            "row_order": tensor.row_order.value,
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
    return {"dtype": str(tensor.dtype).removeprefix("torch.")}


def module_format(module: nn.Module) -> list[tuple[str, dict[str, object]]]:
    """The formats of the tensors ``module`` holds itself, by name."""
    held = [
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ]
    return [
        (name, tensor_format(tensor))
        for name, tensor in held
        if tensor is not None
    ]


def parameters(module: nn.Module) -> dict[str, object]:
    """Declared mathematical arguments, excluding execution resources.

    A component names its scalar or immutable configuration attributes in
    ``function_parameters``. Those values are part of its function, just as
    its revision is. Tensor values are supplied separately by the loader.
    """
    return {
        name: getattr(module, name)
        for name in getattr(module, "function_parameters", ())
    }


def canonical(value):
    """A stable JSON value for immutable numerical arguments."""
    if is_dataclass(value) and not isinstance(value, type):
        return canonical(asdict(value))
    if isinstance(value, Enum):
        return f"{type(value).__qualname__}.{value.name}"
    if isinstance(value, Mapping):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(item) for item in value]
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    raise TypeError(f"not a numerical argument: {type(value).__qualname__}")


def identity(
    module: nn.Module,
    *,
    values: Mapping[int, str] | None = None,
    select: tuple[str, ...] | None = None,
) -> str:
    """The digest of what ``module`` computes: its function, its format and
    its children's identities, in registration order, as lowercase
    hexadecimal SHA-256.

    ``values`` supplies each tensor's loading identity by object identity.
    ``select`` restricts the computation to named child paths or tensors;
    intermediate components still contribute their function declarations.
    A cache-writing phase uses this to exclude downstream computations.

    Raises:
        SetupError: A module of the tree declares no function.
    """  # noqa: D205
    memo: dict[tuple[int, tuple[str, ...] | None], str] = {}

    def visit(node: nn.Module, selected: tuple[str, ...] | None) -> str:
        key = (id(node), selected)
        known = memo.get(key)
        if known is not None:
            return known
        digest = hashlib.sha256()
        _hash(digest, function(type(node)))
        arguments = parameters(node)
        if selected is not None:
            arguments = {
                name: value
                for name, value in arguments.items()
                if name in selected
            }
        _hash(digest, json.dumps(canonical(arguments), sort_keys=True))
        for name, tensor in (
            *node.named_parameters(recurse=False),
            *node.named_buffers(recurse=False),
        ):
            if selected is not None and name not in selected:
                continue
            _hash(digest, name)
            _hash(digest, json.dumps(tensor_format(tensor), sort_keys=True))
            if values is not None:
                _hash(digest, values[id(tensor)])
        for name, child in node.named_children():
            paths = None
            if selected is not None and name not in selected:
                paths = tuple(
                    path[len(name) + 1 :]
                    for path in selected
                    if path.startswith(f"{name}.")
                )
                if not paths:
                    continue
            _hash(digest, name)
            _hash(digest, visit(child, paths))
        memo[key] = digest.hexdigest()
        return memo[key]

    return visit(module, select)


def cache_identities(
    module: nn.Module, values: Mapping[int, str] | None = None
) -> dict[str, str]:
    """Keys of cache writes, before parallel binding prunes the model.

    Each decoder layer reads the embedding and every preceding layer. Its
    cache-writing phase declares only the computation before that write.
    The final norm and vocabulary projection therefore cannot invalidate
    target K/V; draft state additionally depends on the final hidden state
    and its own numerical modules.
    """
    from uniserve.models import CausalLM, Drafter, TransformerDecoder

    result = {}
    for decoder in module.modules():
        if not isinstance(decoder, TransformerDecoder):
            continue
        upstream = hashlib.sha256()
        for model in module.modules():
            if isinstance(model, CausalLM) and model.backbone is decoder:
                _hash(
                    upstream,
                    identity(
                        model, values=values, select=tuple(parameters(model))
                    ),
                )
        _hash(
            upstream,
            identity(
                decoder,
                values=values,
                select=("embedding", *parameters(decoder)),
            ),
        )
        for layer in decoder.layers.values():
            selected = getattr(layer, "cache_dependencies", None)
            if selected is None:
                raise SetupError(
                    f"{type(layer).__qualname__} declares no cache_dependencies"
                )
            current = upstream.copy()
            _hash(current, identity(layer, values=values, select=selected))
            for child in layer.modules():
                name = getattr(child, "cache_name", None)
                if name is not None and hasattr(child, "cache_state"):
                    result[name] = current.hexdigest()
            _hash(upstream, identity(layer, values=values))
        if decoder.norm is not None:
            _hash(upstream, identity(decoder.norm, values=values))
        for model in module.modules():
            if not isinstance(model, CausalLM) or model.backbone is not decoder:
                continue
            for drafter in model.modules():
                if isinstance(drafter, Drafter):
                    draft = upstream.copy()
                    _hash(draft, identity(drafter, values=values))
                    for name in drafter.cache_config.layers:
                        result[name] = draft.hexdigest()
    return result


def loading(encodings: Iterable[str]) -> str:
    """The shared assignment and quantization functions of one weight.

    Checkpoint selection and rectangular assignments are explicit inputs to
    its key. Additional mathematical transforms declare their own revisions
    and inputs through ``uniserve.loading.weights.Derivation``.

    Raises:
        SetupError: The quantizer of an encoding declares no revision.
    """  # noqa: D205
    from uniserve.quantization.quantizer import FUNCTIONS

    digest = hashlib.sha256()
    _hash(digest, "uniserve-weight-loading")
    _hash(digest, LOADER)
    for encoding in sorted(set(encodings)):
        quantizer = FUNCTIONS.get(encoding)
        if quantizer is None:
            raise SetupError(f"no quantizer declares the encoding {encoding}")
        _hash(digest, quantizer)
    return digest.hexdigest()


def _hash(digest, value: str) -> None:
    encoded = value.encode()
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
