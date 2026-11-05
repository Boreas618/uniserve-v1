"""Tiny checkpoint builders shared by model loading and runtime tests.

Each builder writes a seeded safetensors checkpoint under ``root`` whose
dimensions are small enough for CPU equation references.
"""

from __future__ import annotations

import json

import torch
from safetensors.torch import save_file
from transformers import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from uniserve import loading
from uniserve.loading import checkpoint, weights
from uniserve.models import registry as models


def qwen_checkpoint(root, tied=False, theta=1_000_000.0):
    """Save a two-layer Qwen3 checkpoint and return its reference model."""
    config = Qwen3Config(
        vocab_size=37,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=tied,
        attention_bias=True,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=theta,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(481)
    model = Qwen3ForCausalLM(config).eval()
    model.save_pretrained(root)
    return model


def qwen_moe_checkpoint(root, *, norm_topk_prob=True, **overrides):
    """Save a two-layer Qwen3-MoE checkpoint and return its reference model.

    Transformers writes each expert's gate, up and down projections as
    separate tensors, the layout of released Qwen3-MoE checkpoints.
    ``overrides`` replace any ``Qwen3MoeConfig`` field.
    """
    fields = {
        "vocab_size": 37,
        "hidden_size": 32,
        "intermediate_size": 48,
        "moe_intermediate_size": 16,
        "num_experts": 6,
        "num_experts_per_tok": 2,
        "norm_topk_prob": norm_topk_prob,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
    }
    config = Qwen3MoeConfig(**(fields | overrides))
    config._attn_implementation = "eager"
    torch.manual_seed(907)
    model = Qwen3MoeForCausalLM(config).eval()
    model.save_pretrained(root)
    return model


# The multi-token prediction tensors of `qwen3_5_checkpoint`, in its file of
# their own.
QWEN3_5_MTP = "mtp.safetensors"


def qwen3_5_checkpoint(root, layers: int = 4):
    """Save a Qwen3.5 checkpoint and return its reference model.

    The checkpoint has ``layers`` layers, a multiple of four: three gated
    delta layers precede each full-attention layer, as every
    fourth layer of released checkpoints is full attention. The checkpoint
    also holds one multi-token prediction layer, which Transformers does
    not model: its tensors are written beside the reference's, and the
    function returns them too.
    """
    from safetensors import safe_open
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    hidden, heads, kv_heads, head_dim, intermediate = 32, 4, 2, 32, 48
    text = {
        "vocab_size": 61,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
        "mtp_num_hidden_layers": 1,
        "mamba_ssm_dtype": "float32",
    }
    vision = {
        "depth": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_heads": 2,
        "out_hidden_size": hidden,
        "patch_size": 2,
        "num_position_embeddings": 16,
    }
    config = Qwen3_5Config(text_config=text, vision_config=vision)
    config.text_config._attn_implementation = "eager"
    torch.manual_seed(358)
    model = Qwen3_5ForConditionalGeneration(config).eval()
    # Zero-centered norms initialize to zero and the gated output norm to
    # one; draw them so that every norm's scale differs from one.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight"):
                parameter.normal_(0, 0.2)
    model.save_pretrained(root)

    generator = torch.Generator().manual_seed(359)

    def draw(*shape):
        return torch.randn(shape, generator=generator) * 0.2

    layer = "mtp.layers.0"
    mtp = {
        "mtp.fc.weight": draw(hidden, 2 * hidden),
        "mtp.pre_fc_norm_embedding.weight": draw(hidden),
        "mtp.pre_fc_norm_hidden.weight": draw(hidden),
        "mtp.norm.weight": draw(hidden),
        f"{layer}.input_layernorm.weight": draw(hidden),
        f"{layer}.post_attention_layernorm.weight": draw(hidden),
        f"{layer}.self_attn.q_proj.weight": draw(2 * heads * head_dim, hidden),
        f"{layer}.self_attn.k_proj.weight": draw(kv_heads * head_dim, hidden),
        f"{layer}.self_attn.v_proj.weight": draw(kv_heads * head_dim, hidden),
        f"{layer}.self_attn.o_proj.weight": draw(hidden, heads * head_dim),
        f"{layer}.self_attn.q_norm.weight": draw(head_dim),
        f"{layer}.self_attn.k_norm.weight": draw(head_dim),
        f"{layer}.mlp.gate_proj.weight": draw(intermediate, hidden),
        f"{layer}.mlp.up_proj.weight": draw(intermediate, hidden),
        f"{layer}.mlp.down_proj.weight": draw(hidden, intermediate),
    }
    save_file(mtp, root / QWEN3_5_MTP)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        with safe_open(root / "model.safetensors", "pt") as handle:
            index = {
                "metadata": {},
                "weight_map": dict.fromkeys(handle.keys(), "model.safetensors"),
            }
    index["weight_map"].update(dict.fromkeys(mtp, QWEN3_5_MTP))
    index_path.write_text(json.dumps(index))
    return model, mtp
