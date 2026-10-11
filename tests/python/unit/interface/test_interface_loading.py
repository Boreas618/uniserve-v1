"""Describing, loading and planning the stub model through the interface."""

import pytest
import torch

from tests.python.fixtures.stub import load_model, read_config
from uniserve_models import interface
from uniserve_models.interface import cache, loading
from uniserve_models.interface.errors import UnsupportedSetup
from uniserve_models.runner import causal_lm

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def loaded():
    table = interface.table()
    return table, load_model(read_config())


def test_description_reports_calls_with_capability_tags():
    description = read_config()
    assert description.model_name() == "uniserve_models.stub.model.Model"
    assert description.components() == ["model", "vision_encoder"]
    assert description.call_paths("model")[:2] == ["", ""]
    assert description.call_methods("model")[:2] == [
        "forward",
        "embed_input_ids",
    ]
    assert set(description.call_stages("model")) == {"all"}
    # The first call's tags lead the concatenated column.
    assert description.capability_counts("model")[0] == 1
    assert description.capabilities("model")[0] == "causal_lm"
    assert description.call_paths("vision_encoder") == ["vision_encoder"]
    assert description.capability_counts("vision_encoder") == [2]
    assert set(description.capabilities("vision_encoder")) == {
        "patch_encoder",
        "encoder",
    }
    # The stub's modules communicate over no mesh axis.
    assert description.selection_sizes("model") == []
    assert description.selection_axes("model") == []
    assert description.routed_experts() == []
    assert description.decodes_video() is False


def test_a_generation_device_places_the_flow_route():
    # The stub's image denoiser and decoder form a flow route to place.
    assert read_config(generation_device="cpu").components()


def test_split_experts_require_routed_expert_layers():
    with pytest.raises(UnsupportedSetup, match="routed expert layers"):
        read_config(expert_split=True)


def test_a_loaded_model_reports_its_backend_sizes_and_formats(loaded):
    _, model = loaded
    assert loading.attention_backend(model) == "torch"
    assert causal_lm.vocab_size(model) == 151671
    assert causal_lm.hidden_size(model) == 4
    assert loading.weight_formats(model) == ["dense"]
    # The stub has no quantizable linear layer.
    assert loading.activation_formats(model) == []


def test_cache_plan_describes_pages_of_the_requested_size(loaded):
    _, model = loaded
    plan = cache.cache_plan(model, 32, "")
    assert cache.reads_pages(model, "torch", plan) is True
    assert plan.dtype() == "bfloat16"
    assert plan.page_tokens() == [32]
    assert plan.layers() == ["backbone.layers.0.attention"]
    assert plan.layer_ids() == [0]
    assert plan.windows() == [-1]
    # One column holds key and value planes of 32 one-element BF16 rows,
    # plus one initialization flag per plane.
    assert plan.unit_bytes() == 2 * (32 * 2 + 1)


def test_component_binding_serves_its_declared_workspace(loaded):
    table, model = loaded
    prepared, requirements = table["component.prepare"](
        model,
        {
            "component": "model",
            "path": "",
            "device": -1,
            "stream": 0,
            "size": {"tokens": 16, "rows": 2},
            "attention": "auto",
            "vsa": "auto",
            "matmul": "auto",
            "moe": "auto",
            "derive_host_lengths": True,
            "experts": None,
            "weights": None,
            "cache": None,
        },
    )
    tensors = {
        item["name"]: torch.zeros(
            item["shape"], dtype=getattr(torch, item["dtype"])
        )
        for item in requirements["workspaces"]
    }
    functions = table["component.bind"](prepared, {"workspaces": tensors})
    paths = {record["path"] for record in functions["kernels"]()}
    assert "backbone.layers.0.attention" in paths
    assert prepared.context is not None
    functions["close"](False)
    assert prepared.context is None


def test_a_single_rank_world_binds_local_communicators():
    table = interface.table()
    # One CPU rank outside a union: no backend choice, store or union.
    world = table["distributed.initialize"](
        0, 0, 1, "cpu", None, None, None, None, None, None
    )
    assert table["distributed.create_groups"](world, [], [], [], []) == []
    mesh = table["distributed.mesh"](
        world,
        [0],
        [1, 1, 1, 1],
        ["pp", "cp", "tp", "ulysses"],
        0,
        ["tp"],
        [1],
        [0],
        [None],
    )
    group = mesh.get_group("tp")
    assert (group.name, group.ranks, group.rank) == ("tp", (0,), 0)
    table["distributed.destroy"](world)
