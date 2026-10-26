"""Plugins add kernel providers and model packages through entry points."""

from __future__ import annotations

import importlib.metadata
import sys
import types

import pytest
import torch

from uniserve import plugins
from uniserve.backends import attention, moe
from uniserve.backends.matmul import base as matmul
from uniserve.models import registry


class _Attention:
    """A stand-in attention provider."""


class _Matmul:
    """A stand-in matmul provider."""


def _register(added: plugins.Registry) -> None:
    added.provider("attention", "example", _Attention)
    added.provider("matmul", "example", _Matmul)
    added.model("ExampleForCausalLM", "example_plugin.model")


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """One plugin package installed: ``example_plugin``, version 1.2."""
    module = types.ModuleType("example_plugin")
    module.register = _register
    monkeypatch.setitem(sys.modules, "example_plugin", module)

    class _Distribution:
        name = "example-plugin"
        version = "1.2"

    entry = importlib.metadata.EntryPoint(
        name="example", value="example_plugin:register", group=plugins.GROUP
    )
    # ``EntryPoint`` is immutable; ``dist`` is set as ``importlib.metadata``
    # sets it for an installed distribution.
    object.__setattr__(entry, "dist", _Distribution())

    def entry_points(*, group: str):
        return [entry] if group == plugins.GROUP else []

    monkeypatch.setattr(
        plugins.importlib.metadata, "entry_points", entry_points
    )
    monkeypatch.setattr(plugins, "_registry", None)


def test_a_plugins_providers_resolve_by_name(installed: None) -> None:
    device = torch.device("cpu")
    assert isinstance(attention.resolve("example", device=device), _Attention)
    assert isinstance(matmul.resolve("example", torch.empty(1)), _Matmul)
    # Names no built-in provider and no plugin has are refused, and the
    # attention family lists the plugin's name among the accepted ones.
    with pytest.raises(ValueError, match="example"):
        attention.resolve("absent", device=device)
    with pytest.raises(ValueError, match="unknown expert backend"):
        moe.resolve("absent", module=None, device=device)


def test_a_plugins_architectures_join_the_catalog(installed: None) -> None:
    catalog = registry._with_plugins()
    assert catalog["ExampleForCausalLM"] == "example_plugin.model"
    assert set(registry._catalog) <= set(catalog)


def test_a_plugin_may_not_shadow_a_built_in_architecture(
    installed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    built_in = next(iter(registry._catalog))
    plugins.registry().models[built_in] = "example_plugin.model"
    with pytest.raises(ValueError, match="built-in architectures"):
        registry._with_plugins()


def test_a_plugin_may_not_shadow_a_built_in_provider(
    installed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def shadowing(added: plugins.Registry) -> None:
        added.provider("moe", "cutlass", object)

    monkeypatch.setattr(sys.modules["example_plugin"], "register", shadowing)
    with pytest.raises(ValueError, match="built-in moe providers cutlass"):
        plugins.registry()


def test_two_plugins_may_not_add_one_name() -> None:
    added = plugins.Registry()
    added.provider("moe", "example", object)
    with pytest.raises(ValueError, match="two plugins"):
        added.provider("moe", "example", object)
    with pytest.raises(ValueError, match="not a kernel family"):
        added.provider("sampler", "example", object)


def test_installed_plugins_enter_the_weights_fingerprint(
    installed: None,
) -> None:
    assert plugins.versions() == ["example-plugin==1.2"]


def test_a_plugins_buffers_are_allocated_in_name_order_for_the_device(
    installed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []

    def factory(name: str):
        def allocate(models: list, device: int) -> str:
            calls.append((name, device))
            return f"{name} on {device}"

        return allocate

    def buffers(added: plugins.Registry) -> None:
        added.buffer("symmetric", factory("symmetric"))
        added.buffer("multicast", factory("multicast"))

    monkeypatch.setattr(sys.modules["example_plugin"], "register", buffers)
    held = plugins.allocate_buffers(3)
    assert held == ["multicast on 3", "symmetric on 3"]
    assert calls == [("multicast", 3), ("symmetric", 3)]


def test_two_plugins_may_not_allocate_one_buffer() -> None:
    added = plugins.Registry()
    added.buffer("symmetric", lambda models, device: None)
    with pytest.raises(ValueError, match="two plugins allocate"):
        added.buffer("symmetric", lambda models, device: None)


def test_plugin_hooks_may_not_take_the_engines_names() -> None:
    added = plugins.Registry()
    with pytest.raises(ValueError, match="engine's own"):
        added.hook("uniserve.sleep", lambda payload, models: None)


def test_a_plugins_graph_policy_changes_the_entries_it_returns(
    installed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    seen: list[dict] = []

    def policy(models: list, configured: dict) -> dict:
        seen.append(configured)
        return {"decode": [1, 2], "mode": "full"}

    def choose(added: plugins.Registry) -> None:
        added.graph_policy(policy)

    monkeypatch.setattr(sys.modules["example_plugin"], "register", choose)
    configured = {
        "mode": "auto",
        "decode": [1, 2, 4, 8],
        "prefill": [512],
        "flow_shapes": [[1024, 1024]],
    }
    chosen = json.loads(plugins.graph_policy(json.dumps(configured)))
    assert chosen == {**configured, "decode": [1, 2], "mode": "full"}
    assert seen == [configured]


def test_a_graph_policy_may_not_choose_unknown_entries(
    installed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def choose(added: plugins.Registry) -> None:
        added.graph_policy(lambda models, configured: {"streams": 2})
        with pytest.raises(ValueError, match="two plugins choose"):
            added.graph_policy(lambda models, configured: {})

    monkeypatch.setattr(sys.modules["example_plugin"], "register", choose)
    with pytest.raises(ValueError, match="unknown entries streams"):
        plugins.graph_policy('{"mode": "auto"}')


def test_without_a_graph_policy_the_configuration_stands(
    installed: None,
) -> None:
    assert plugins.graph_policy('{"mode": "auto"}') == ""
