"""Describing, loading and planning the stub model through the interface."""

import pytest
import torch
import tvm_ffi

from tests.python.fixtures.stub import load_model, read_config
from uniserve_models import interface
from uniserve_models.interface.errors import UnsupportedSetup

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def loaded():
    table = interface.table()
    description, _ = read_config()
    return table, load_model(description)


def test_description_reports_calls_with_capability_tags():
    _, info = read_config()
    assert info["model_name"] == "uniserve_models.stub.model.Model"
    entries = {entry["name"]: entry["calls"] for entry in info["entries"]}
    assert set(entries) == {"model", "vision_encoder"}
    forward = entries["model"][0]
    assert (forward["path"], forward["method"], forward["stage"]) == (
        "",
        "forward",
        "all",
    )
    assert forward["capabilities"] == ["causal_lm"]
    assert forward["axes"] == []
    vision = entries["vision_encoder"][0]
    assert set(vision["capabilities"]) == {"patch_encoder", "encoder"}
    assert info["expert_paths"] == []
    assert info["video_decoder"] is False


def test_a_generation_device_places_the_flow_route():
    # The stub's image denoiser and decoder form a flow route to place.
    _, info = read_config(generation_device="cpu")
    assert info["entries"]


def test_split_experts_require_routed_expert_layers():
    with pytest.raises(UnsupportedSetup, match="routed expert layers"):
        read_config(expert_split=True)


def test_facts_report_text_geometry_and_formats(loaded):
    table, model = loaded
    facts = table["model.facts"](
        model,
        {
            "max_sequence_tokens": 64,
            "model_dtype": "bfloat16",
            "role": "model",
        },
    )
    assert facts["attention_backend"] == "torch"
    assert facts["vocab_size"] == 151671
    assert facts["hidden_size"] == 4
    assert facts["weight_formats"] == ["dense"]
    assert facts["outputs"] == {}
    assert facts["image_flow"] == {
        "max_tokens": 1024,
        "latent_channels": 3,
        "patch_size": 16,
    }


def test_cache_plan_describes_pages_of_the_requested_size(loaded):
    table, model = loaded
    plan = table["model.cache_plan"](model, 32, "")
    assert table["model.reads_pages"](model, "torch", plan) is True
    assert plan["dtype"] == "bfloat16"
    (group,) = plan["groups"]
    assert group["page_tokens"] == 32
    assert group["layers"] == ["backbone.layers.0.attention"]
    assert group["layer_ids"] == [0]
    assert group["window"] == -1
    # One column holds key and value planes of 32 one-element BF16 rows,
    # plus one initialization flag per plane.
    assert plan["unit_bytes"] == 2 * (32 * 2 + 1)


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
    world = table["distributed.initialize"](
        {
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "device": "cpu",
            "backend": None,
            "store": None,
            "union": None,
        }
    )
    assert table["distributed.create_groups"](world, []) == []
    mesh = table["distributed.mesh"](
        world,
        tvm_ffi.convert(
            {
                "ranks": [0],
                "shape": [1, 1, 1, 1],
                "axes": ["pp", "cp", "tp", "ulysses"],
                "rank": 0,
                "groups": {"tp": {"members": [0], "group": None}},
            }
        ),
    )
    group = mesh.get_group("tp")
    assert (group.name, group.ranks, group.rank) == ("tp", (0,), 0)
    table["distributed.destroy"](world)
