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
  keeps its revision. A model package declares the revision of its
  checkpoint loading the same way (``LOADING`` of the package), as the
  quantizers do per format (``uniserve.quantization.quantizer.FUNCTIONS``)
  and the loader for what it does to every tensor (``uniserve.loading.
  LOADING``).
- A module's format is that of the tensors it holds itself: each
  parameter's and buffer's element type, or a quantized tensor's encoding
  and the layout its kernels read.
- A module's identity digests its function, its format and its children's
  identities in registration order. A model's identity keys what its
  forwards compute, such as its KV cache pages; the identity of its weight
  loading (:func:`loading`) keys its retained weights.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
import re
from collections.abc import Iterable
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


def identity(module: nn.Module) -> str:
    """The digest of what ``module`` computes: its function, its format and
    its children's identities, in registration order, as lowercase
    hexadecimal SHA-256.

    Raises:
        SetupError: A module of the tree declares no function.
    """  # noqa: D205
    memo: dict[int, str] = {}

    def visit(node: nn.Module) -> str:
        known = memo.get(id(node))
        if known is not None:
            return known
        digest = hashlib.sha256()
        _hash(digest, function(type(node)))
        _hash(digest, json.dumps(module_format(node), sort_keys=True))
        for name, child in node.named_children():
            _hash(digest, name)
            _hash(digest, visit(child))
        memo[id(node)] = digest.hexdigest()
        return memo[id(node)]

    return visit(module)


def loading(model_class: type, encodings: Iterable[str]) -> str:
    """The digest of how a model's weights are derived from its checkpoint:
    the loading its package declares, the loader's, and the quantizer of
    each encoding the model loads, as lowercase hexadecimal SHA-256.

    Raises:
        SetupError: The model's package, or the quantizer of an encoding,
            declares no revision.
    """  # noqa: D205
    from uniserve.quantization.quantizer import FUNCTIONS

    package = _package(model_class)
    declared = getattr(importlib.import_module(package), "LOADING", None)
    if not isinstance(declared, str) or "/" not in declared:
        raise SetupError(
            f"model package {package} declares no LOADING revision of its "
            "checkpoint loading"
        )
    digest = hashlib.sha256()
    _hash(digest, "uniserve-weight-loading")
    _hash(digest, declared)
    _hash(digest, LOADER)
    for encoding in sorted(set(encodings)):
        quantizer = FUNCTIONS.get(encoding)
        if quantizer is None:
            raise SetupError(f"no quantizer declares the encoding {encoding}")
        _hash(digest, quantizer)
    return digest.hexdigest()


def _package(model_class: type) -> str:
    """The model package ``model_class`` belongs to: ``uniserve.models.<name>``
    for a built-in model, a plugin's top-level package otherwise.
    """  # noqa: D205
    parts = model_class.__module__.split(".")
    if parts[:2] == ["uniserve", "models"] and len(parts) > 2:
        return ".".join(parts[:3])
    return parts[0]


def _hash(digest, value: str) -> None:
    encoded = value.encode()
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
