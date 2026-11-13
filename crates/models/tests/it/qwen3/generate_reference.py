"""Generates the reference vectors of the qwen3 crate's CPU tests.

Each fixture is a tiny random checkpoint (`model/config.json` and
`model/model.safetensors`, FP32) and the outputs the reference implementation
computes from it on the CPU (`reference.safetensors`):

- `qwen3_moe`: transformers' `Qwen3MoeForCausalLM`, saved with one module per
  expert as the released Qwen3-30B-A3B stores them.
- `qwen3_5_moe`: transformers' `Qwen3_5MoeForConditionalGeneration` (gated
  delta networks, gated attention, a shared expert, the vision tower), with
  the multi-token-prediction module the released checkpoints carry under
  `mtp.*`. Transformers has no class for that module, so
  `MultiTokenPredictor` below states vLLM's `Qwen3_5MultiTokenPredictor`
  (`refs/vllm/vllm/model_executor/models/qwen3_5_mtp.py:68-193`) over
  transformers' modules: its `ColumnParallelLinear` is `nn.Linear`, its
  `Qwen3_5DecoderLayer` and `GemmaRMSNorm` are transformers'
  `Qwen3_5MoeDecoderLayer` and `Qwen3_5MoeRMSNorm`, and the layer's
  `(hidden, residual)` pair is the sum transformers' layer returns. Nothing
  else differs: no device string, no arithmetic.
- `qwen3_vl`: transformers' `Qwen3VLModel` with DeepStack features, the model
  a pipeline uses as its text encoder, whole and truncated before its final
  norm.

Run from the repository root with the repository's interpreter:

    .venv/bin/python crates/models/tests/it/qwen3/generate_reference.py

The script is deterministic: every weight and input is drawn from a seeded
generator, so a rerun on the same PyTorch build writes the same bytes.
"""

import copy
import json
import pathlib

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    Qwen3VLConfig,
    Qwen3VLModel,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeRMSNorm,
)

DATA = pathlib.Path(__file__).parent / "data"

# The placeholder tokens of the two vision fixtures, the last two entries of
# their 32-entry vocabularies.
IMAGE_TOKEN = 30
VIDEO_TOKEN = 31

# The tokens of the two text rows of a causal model's fixture: one that
# crosses a 64-token cache page and a 64-token chunk of the gated delta rule,
# and a short one.
ROWS = (70, 7)

# The normalizations whose weight is a zero-centred gain, by class name.
CENTRED_NORMS = ("Qwen3_5MoeRMSNorm",)


def randomize(model: nn.Module, seed: int) -> None:
    """Draws every parameter of `model` from a generator seeded by `seed`.

    Transformers initializes matrices at a standard deviation of 0.02, under
    which a tiny model's attention is uniform and its router's scores tie. A
    matrix is drawn at `1 / sqrt(fan_in)`, an embedding at one, a gain around
    one (or around zero where the class stores it zero-centred), a bias
    around zero and a decay rate inside the range transformers draws it from,
    so every operation of a forward moves its input.

    Args:
        model: The module whose parameters are overwritten in place.
        seed: The generator's seed.
    """
    generator = torch.Generator().manual_seed(seed)

    def normal(parameter: torch.Tensor, std: float) -> torch.Tensor:
        return torch.randn(parameter.shape, generator=generator) * std

    with torch.no_grad():
        for module in model.modules():
            kind = type(module).__name__
            for name, parameter in module.named_parameters(recurse=False):
                if kind.endswith(("RMSNorm", "RMSNormGated", "LayerNorm")):
                    centre = 0.0 if kind in CENTRED_NORMS else 1.0
                    centre = centre if name == "weight" else 0.0
                    value = centre + normal(parameter, 0.1)
                elif name == "A_log":
                    uniform = torch.rand(parameter.shape, generator=generator)
                    value = torch.log(1.0 + 15.0 * uniform)
                elif name == "dt_bias":
                    value = 1.0 + normal(parameter, 0.1)
                elif isinstance(module, nn.Embedding):
                    value = normal(parameter, 1.0)
                elif parameter.ndim == 1:
                    value = normal(parameter, 0.1)
                else:
                    # A stack of experts and a convolution reduce over the
                    # extents after the output channels' own.
                    stacked = parameter.ndim == 3 and name.endswith("_proj")
                    fan_in = parameter[0, 0] if stacked else parameter[0]
                    value = normal(parameter, fan_in.numel() ** -0.5)
                parameter.copy_(value)


def token_ids(count: int, seed: int) -> torch.Tensor:
    """`count` text token ids of a 32-entry vocabulary, as `[1, count]`."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, IMAGE_TOKEN, (1, count), generator=generator)


def pixels(grid: list[int], tubelet: int, seed: int) -> torch.Tensor:
    """The patch rows of a `[time, height, width]` grid, one tubelet each."""
    generator = torch.Generator().manual_seed(seed)
    patches = grid[0] * grid[1] * grid[2]
    return torch.randn((patches, tubelet), generator=generator)


def write(
    name: str,
    config: dict,
    weights: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    processor: bool = False,
) -> None:
    """Writes the fixture `name`: its checkpoint and its reference vectors.

    Args:
        name: The fixture's directory under `data`.
        config: The checkpoint's `config.json`.
        weights: The checkpoint's tensors by name.
        reference: The inputs and expected outputs by name.
        processor: Whether the checkpoint states the Qwen-VL processor files
            its images are sized by.
    """
    model = DATA / name / "model"
    model.mkdir(parents=True, exist_ok=True)

    (model / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(
        {key: value.contiguous() for key, value in weights.items()},
        model / "model.safetensors",
    )
    if processor:
        rules = {
            "patch_size": config["vision_config"]["patch_size"],
            "temporal_patch_size": 2,
            "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
        for file, longest in [
            ("preprocessor_config.json", 4096),
            ("video_preprocessor_config.json", 16384),
        ]:
            size = {"shortest_edge": 64, "longest_edge": longest}
            text = json.dumps(rules | {"size": size}, indent=2)
            (model / file).write_text(text + "\n")

    stored = {}
    for key, value in reference.items():
        value = value.detach()
        if not value.dtype.is_floating_point:
            value = value.to(torch.int32)
        stored[key] = value.contiguous()
    save_file(stored, DATA / name / "reference.safetensors")


def qwen3_moe() -> None:
    """Qwen3-MoE: two rows of text through `Qwen3MoeForCausalLM`."""
    config = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "hidden_act": "silu",
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "attention_bias": False,
        "tie_word_embeddings": False,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "norm_topk_prob": True,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
    }
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**config)).eval()
    randomize(model, 1)

    # The released checkpoints store one module per expert; transformers
    # stacks them, each expert's gate rows before its up rows.
    weights = {}
    for name, value in model.state_dict().items():
        if name.endswith("experts.gate_up_proj"):
            gates, ups = value.chunk(2, dim=1)
            for expert, (gate, up) in enumerate(zip(gates, ups, strict=True)):
                prefix = f"{name.removesuffix('gate_up_proj')}{expert}"
                weights[f"{prefix}.gate_proj.weight"] = gate
                weights[f"{prefix}.up_proj.weight"] = up
        elif name.endswith("experts.down_proj"):
            for expert, down in enumerate(value):
                prefix = f"{name.removesuffix('down_proj')}{expert}"
                weights[f"{prefix}.down_proj.weight"] = down
        else:
            weights[name] = value

    reference = {}
    with torch.no_grad():
        for row, count in enumerate(ROWS):
            ids = token_ids(count, 10 + row)
            hidden = model.model(input_ids=ids).last_hidden_state[0]
            reference[f"rows.{row}.ids"] = ids[0]
            reference[f"rows.{row}.hidden"] = hidden
            reference[f"rows.{row}.logits"] = model.lm_head(hidden)

    config["architectures"] = ["Qwen3MoeForCausalLM"]
    write("qwen3_moe", config, weights, reference)


class MultiTokenPredictor(nn.Module):
    """vLLM's `Qwen3_5MultiTokenPredictor` over transformers' modules.

    `refs/vllm/vllm/model_executor/models/qwen3_5_mtp.py:68-193`; the module
    documentation lists what stands in for vLLM's layers.
    """

    def __init__(self, config) -> None:
        super().__init__()
        hidden = config.hidden_size

        # Every multi-token-prediction layer attends (`qwen3_5_mtp.py:123`).
        layer = copy.deepcopy(config)
        layer.layer_types = ["full_attention"]

        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        self.layers = nn.ModuleList([Qwen3_5MoeDecoderLayer(layer, 0)])
        self.norm = Qwen3_5MoeRMSNorm(hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5MoeRMSNorm(
            hidden, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_embedding = Qwen3_5MoeRMSNorm(
            hidden, eps=config.rms_norm_eps
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)

        # A decoder layer over the entries, each attending the ones before it.
        tokens = hidden_states.shape[1]
        causal = torch.full((tokens, tokens), float("-inf")).triu(1)
        hidden_states = self.layers[0](
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=causal[None, None],
        )
        return self.norm(hidden_states)


def qwen3_5_moe() -> None:
    """Qwen3.5-MoE: text rows, the drafter, an image and a prompt with it."""
    text = {
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 3,
        "layer_types": [
            "linear_attention",
            "full_attention",
            "linear_attention",
        ],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "hidden_act": "silu",
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "attn_output_gate": True,
        "tie_word_embeddings": False,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 4,
        "moe_intermediate_size": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "shared_expert_intermediate_size": 8,
        "mtp_num_hidden_layers": 1,
        "partial_rotary_factor": 0.5,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "partial_rotary_factor": 0.5,
            "rope_theta": 10000.0,
            "rope_type": "default",
        },
    }
    vision = {
        "depth": 2,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_heads": 2,
        "hidden_act": "gelu_pytorch_tanh",
        "in_channels": 3,
        "patch_size": 2,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "out_hidden_size": 16,
        "num_position_embeddings": 16,
        "deepstack_visual_indexes": [],
    }
    config = {
        "text_config": text,
        "vision_config": vision,
        "image_token_id": IMAGE_TOKEN,
        "video_token_id": VIDEO_TOKEN,
    }
    model = Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeConfig(**config))
    model = model.eval()
    randomize(model, 2)
    language = model.model.language_model

    mtp = MultiTokenPredictor(language.config).eval()
    randomize(mtp, 3)

    weights = dict(model.state_dict())
    for name, value in mtp.state_dict().items():
        weights[f"mtp.{name}"] = value

    reference = {}
    with torch.no_grad():
        for row, count in enumerate(ROWS):
            ids = token_ids(count, 20 + row)
            hidden = language(input_ids=ids).last_hidden_state
            reference[f"rows.{row}.ids"] = ids[0]
            reference[f"rows.{row}.hidden"] = hidden[0]
            reference[f"rows.{row}.logits"] = model.lm_head(hidden[0])

            # The drafter's entry at `i` reads the model's hidden state at
            # `i` and the embedding of the token at `i + 1`, at position `i`.
            entries = count - 1
            positions = torch.arange(entries).view(1, 1, -1).expand(3, 1, -1)
            rotary = language.rotary_emb(hidden, positions)
            drafted = mtp(
                language.embed_tokens(ids[:, 1:]), rotary, hidden[:, :entries]
            )
            reference[f"rows.{row}.drafted"] = drafted[0]

        # One image of a 4x6 patch grid, alone and inside a prompt.
        grid = [1, 4, 6]
        image = pixels(grid, 3 * 2 * 2 * 2, 30)
        features = model.model.visual(image, grid_thw=torch.tensor([grid]))
        reference["image.pixels"] = image
        reference["image.features"] = features.pooler_output

        ids = token_ids(11, 31)
        ids[0, 2:8] = IMAGE_TOKEN
        prompt = model.model(
            input_ids=ids,
            pixel_values=image,
            image_grid_thw=torch.tensor([grid]),
            mm_token_type_ids=(ids == IMAGE_TOKEN).int(),
        )
        reference["prompt.ids"] = ids[0]
        reference["prompt.hidden"] = prompt.last_hidden_state[0]

    config["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    write("qwen3_5_moe", config, weights, reference, processor=True)


def qwen3_vl() -> None:
    """Qwen3-VL: a prompt with an image and a video through `Qwen3VLModel`."""
    text = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "hidden_act": "silu",
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "rope_theta": 10000.0,
            "rope_type": "default",
        },
    }
    vision = {
        "depth": 2,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_heads": 2,
        "hidden_act": "gelu_pytorch_tanh",
        "in_channels": 3,
        "patch_size": 2,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "out_hidden_size": 16,
        "num_position_embeddings": 16,
        "deepstack_visual_indexes": [0, 1],
    }
    config = {
        "text_config": text,
        "vision_config": vision,
        "image_token_id": IMAGE_TOKEN,
        "video_token_id": VIDEO_TOKEN,
    }
    model = Qwen3VLModel(Qwen3VLConfig(**config)).eval()
    randomize(model, 4)
    weights = {
        f"model.{name}": value for name, value in model.state_dict().items()
    }

    # Two text tokens, an image of six merged rows, a text token, and a
    # video of two time steps of two merged rows, each behind its timestamp.
    image_grid, video_grid = [1, 4, 6], [2, 2, 4]
    ids = token_ids(15, 40)
    ids[0, 2:8] = IMAGE_TOKEN
    ids[0, 9:11] = VIDEO_TOKEN
    ids[0, 12:14] = VIDEO_TOKEN
    kinds = (ids == IMAGE_TOKEN).int() + 2 * (ids == VIDEO_TOKEN).int()

    tubelet = 3 * 2 * 2 * 2
    image = pixels(image_grid, tubelet, 41)
    video = pixels(video_grid, tubelet, 42)
    inputs = {
        "input_ids": ids,
        "pixel_values": image,
        "image_grid_thw": torch.tensor([image_grid]),
        "pixel_values_videos": video,
        "video_grid_thw": torch.tensor([video_grid]),
        "mm_token_type_ids": kinds,
    }

    reference = {
        "prompt.ids": ids[0],
        "image.pixels": image,
        "video.pixels": video,
    }
    with torch.no_grad():
        # The tower's rows of the image: its merged features, then each
        # DeepStack feature, side by side as vLLM's tower returns them
        # (`refs/vllm/vllm/model_executor/models/qwen3_vl.py:905-906`).
        tower = model.visual(image, grid_thw=torch.tensor([image_grid]))
        reference["image.features"] = torch.cat(
            [tower.pooler_output, *tower.deepstack_features], dim=1
        )

        encoded = model(**inputs).last_hidden_state[0]
        reference["prompt.encoded"] = encoded

        # A pipeline reads the state after a prefix of the layers, before
        # the final norm.
        truncated = copy.deepcopy(model)
        truncated.language_model.layers = truncated.language_model.layers[:2]
        truncated.language_model.config.num_hidden_layers = 2
        truncated.language_model.norm = nn.Identity()
        state = truncated(**inputs).last_hidden_state[0]
        reference["prompt.truncated"] = state

    config["architectures"] = ["Qwen3VLForConditionalGeneration"]
    write("qwen3_vl", config, weights, reference, processor=True)


if __name__ == "__main__":
    qwen3_moe()
    qwen3_5_moe()
    qwen3_vl()
