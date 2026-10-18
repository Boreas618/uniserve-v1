"""Describing, loading and planning the stub model as a rank does."""

import pytest

from tests.python.fixtures.stub import load_model, read_config
from uniserve.distributed import world
from uniserve.errors import UnsupportedSetup
from uniserve.loading import rank
from uniserve.prefix_cache import plan as cache
from uniserve.runner import causal_lm

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def loaded():
    return load_model(read_config())


def test_description_reports_calls_with_capability_tags():
    description = read_config()
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
    model = loaded
    assert rank.attention_backend(model) == "torch"
    assert causal_lm.vocab_size(model) == 151671
    assert causal_lm.hidden_size(model) == 4
    assert rank.weight_formats(model) == ["dense"]
    # The stub has no quantizable linear layer.
    assert rank.activation_formats(model) == []


def test_cache_plan_describes_pages_of_the_requested_size(loaded):
    model = loaded
    plan = cache.cache_plan(model, 32, "")
    assert cache.reads_pages(model, "torch", plan) is True
    assert plan.dtype() == "bfloat16"
    assert plan.block_sizes() == [32]
    assert plan.layers() == ["backbone.layers.0.attention"]
    assert plan.layer_ids() == [0]
    assert plan.windows() == [-1]
    # One column holds key and value planes of 32 one-element BF16 rows,
    # plus one initialization flag per plane.
    assert plan.unit_bytes() == 2 * (32 * 2 + 1)


def test_a_single_rank_world_binds_local_communicators():
    # One CPU rank outside a union: no backend choice, store or union.
    process = world.initialize(
        0, 0, 1, "cpu", None, None, None, None, None, None
    )
    assert world.create_groups(process, [], [], [], []) == []
    mesh = world.mesh(
        process,
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
    world.destroy(process)
