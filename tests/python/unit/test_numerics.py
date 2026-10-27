"""The identities that key what a model's components compute and load."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest
import torch
from torch import nn

import uniserve
from uniserve import numerics
from uniserve.errors import SetupError
from uniserve.nn.linear import Linear
from uniserve.nn.norm import RMSNorm
from uniserve.quantization import quantizer

pytestmark = pytest.mark.unit


class _Block(nn.Module):
    """A component of a test model."""

    function = "tests.block/1"

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.norm = RMSNorm(8, dtype=dtype)
        self.projection = Linear(8, 8, dtype=dtype)
        self.layers = nn.ModuleList([RMSNorm(8, dtype=dtype)])


def _module_classes() -> list[type]:
    """Every module class of the computation library and the model
    packages.
    """  # noqa: D205
    found = set()
    for info in pkgutil.walk_packages(uniserve.__path__, "uniserve."):
        if info.name == "uniserve._worker":
            continue
        module = importlib.import_module(info.name)
        for _, value in inspect.getmembers(module, inspect.isclass):
            if issubclass(value, nn.Module) and value.__module__ == info.name:
                found.add(value)
    return sorted(
        found, key=lambda value: f"{value.__module__}.{value.__qualname__}"
    )


def test_every_module_class_declares_its_own_function():
    undeclared = [
        f"{value.__module__}.{value.__qualname__}"
        for value in _module_classes()
        if "function" not in vars(value)
    ]
    assert not undeclared, undeclared
    declared = [vars(value)["function"] for value in _module_classes()]
    assert len(declared) == len(set(declared)), (
        "two classes declare one function"
    )


def test_equal_models_have_one_identity_and_a_revision_or_format_changes_it(
    monkeypatch,
):
    identity = numerics.identity(_Block())
    assert identity == numerics.identity(_Block())

    # Another revision of a component's function, however deep in the tree.
    monkeypatch.setattr(RMSNorm, "function", "nn.norm.RMSNorm/2")
    revised = numerics.identity(_Block())
    assert revised != identity
    monkeypatch.undo()

    # Another element type of the tensors a component holds.
    assert numerics.identity(_Block(torch.bfloat16)) != identity


def test_an_undeclared_component_has_no_identity():
    class Undeclared(nn.Module):
        """A module that states no function."""

    model = _Block()
    model.extra = Undeclared()
    with pytest.raises(SetupError, match="declares no function"):
        numerics.identity(model)


def test_pytorch_modules_compute_their_documented_function():
    assert numerics.function(nn.ModuleList) == "torch.nn.ModuleList/1"
    assert numerics.function(nn.Embedding) == "torch.nn.Embedding/1"


def test_weight_loading_follows_the_package_and_the_quantizers_in_use(
    monkeypatch,
):
    from uniserve.models.qwen3 import Model

    dense = numerics.loading(Model, [])
    assert dense == numerics.loading(Model, [])
    nvfp4 = numerics.loading(Model, ["nvfp4"])
    assert nvfp4 != dense

    # A quantizer the model does not load leaves its weights' identity.
    monkeypatch.setitem(quantizer.FUNCTIONS, "fp8", "quantization.fp8/2")
    assert numerics.loading(Model, ["nvfp4"]) == nvfp4
    monkeypatch.setitem(quantizer.FUNCTIONS, "nvfp4", "quantization.nvfp4/2")
    assert numerics.loading(Model, ["nvfp4"]) != nvfp4

    # Another revision of the package's loading.
    import uniserve.models.qwen3 as package

    monkeypatch.setattr(package, "LOADING", "models.qwen3/2")
    assert numerics.loading(Model, []) != dense


def test_providers_key_the_weights_bytes_and_not_their_values(monkeypatch):
    from types import SimpleNamespace

    from uniserve.loading import retained
    from uniserve.loading import weights as weight_options
    from uniserve.models.qwen3 import Model

    source = SimpleNamespace(
        model_class=Model,
        model={"hidden_size": 8},
        checkpoint_format=None,
        modules=None,
        exclude_modules=frozenset(),
        weights=weight_options.Config(),
        precisions={},
        checkpoint=(),
    )
    io = SimpleNamespace(mode="safetensors")

    def fingerprints():
        return retained.fingerprints(source, io, "bfloat16", "local")

    first = fingerprints()
    # Another kernel library selects other layouts: other bytes, the same
    # values.
    monkeypatch.setattr(retained, "_providers", lambda: "other providers")
    providers = fingerprints()
    assert providers["structure"] != first["structure"]
    assert providers["values"] == first["values"]
    # Another revision of the package's loading derives other values.
    import uniserve.models.qwen3 as package

    monkeypatch.setattr(package, "LOADING", "models.qwen3/2")
    assert fingerprints()["values"] != first["values"]
