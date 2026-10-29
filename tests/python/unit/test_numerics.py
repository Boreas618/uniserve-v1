"""The identities that key what a model's components compute and load."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from uniserve import numerics
from uniserve.errors import SetupError
from uniserve.nn.linear import Linear
from uniserve.nn.norm import RMSNorm

pytestmark = pytest.mark.unit


class _Block(nn.Module):
    """A component of a test model."""

    function = "tests.block/1"

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.norm = RMSNorm(8, dtype=dtype)
        self.projection = Linear(8, 8, dtype=dtype)
        self.layers = nn.ModuleList([RMSNorm(8, dtype=dtype)])


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


def test_cache_keys_follow_upstream_computations_and_tensor_inputs():
    from uniserve.models.qwen3.config import Config
    from uniserve.models.qwen3.model import Model

    config = Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=128,
        attention_bias=False,
        tie_word_embeddings=False,
        num_experts=0,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        norm_topk_prob=False,
        decoder_sparse_step=1,
        mlp_only_layers=(),
    )
    with torch.device("meta"):
        model = Model(config)
    values = {id(tensor): name for name, tensor in model.named_parameters()}
    values.update({id(tensor): name for name, tensor in model.named_buffers()})
    before = numerics.cache_identities(model, values)

    # Logit projection does not produce any K/V, even at the last layer.
    values[id(model.lm_head.weight)] = "another-head"
    model.lm_head.softcap = 30.0
    assert numerics.cache_identities(model, values) == before

    # A layer's feed-forward computation affects subsequent cache writes,
    # while that layer's K/V was already computed.
    values[id(model.backbone.layers["1"].mlp.down.weight)] = "another-mlp"
    after = numerics.cache_identities(model, values)
    assert {name for name in before if before[name] != after[name]} == {
        "backbone.layers.2.attention.attention"
    }

    # Normalization changes the current write and every downstream layer,
    # even though neither tensor values nor dtypes changed.
    model.backbone.layers["1"].input_norm.eps = 1e-5
    after = numerics.cache_identities(model, values)
    assert {name for name in before if before[name] != after[name]} == {
        "backbone.layers.1.attention.attention",
        "backbone.layers.2.attention.attention",
    }


def test_rotary_keys_ignore_an_unused_context_bound():
    from uniserve.nn.rope import DynamicScaling, RotaryEmbedding

    assert numerics.identity(
        RotaryEmbedding(128, max_position_embeddings=4096)
    ) == numerics.identity(RotaryEmbedding(128, max_position_embeddings=8192))
    assert numerics.identity(
        RotaryEmbedding(
            128, scaling=DynamicScaling(2.0), max_position_embeddings=4096
        )
    ) != numerics.identity(
        RotaryEmbedding(
            128, scaling=DynamicScaling(2.0), max_position_embeddings=8192
        )
    )


def test_a_dependencys_modules_compute_its_installed_versions_function():
    from importlib import metadata

    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
        MiniMaxH3AudioBigVGANDecoder,
    )

    assert numerics.function(MiniMaxH3AudioBigVGANDecoder) == (
        "diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio."
        f"MiniMaxH3AudioBigVGANDecoder/{metadata.version('diffusers')}"
    )
