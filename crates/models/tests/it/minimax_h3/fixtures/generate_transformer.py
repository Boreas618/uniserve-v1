"""Generate the transformer fixtures from the reference implementation.

The reference is vLLM-Omni's MiniMax-H3 network: its block
(`MiniMaxH3DiTBlock` with `MiniMaxH3Attention`, `MiniMaxH3MLP` and
`MiniMaxH3AdalnProj`), its timestep embedder, its rotary embedding, its
token refiner and its final layer, with the layers they call
(`RMSNorm`, `SiluAndMul`, the indexed modulation functions and the fused
q/k norm with rotation, each on its own CPU path). They run on CPU with
torch in FP32, loaded by path.

What stands in for the serving framework, which this environment does not
install:

- its parallel linear layers are `torch.nn.Linear` returning
  `(output, None)`, which is what they compute on one rank;
- its attention layer, which dispatches to a device backend, is
  `torch.nn.functional.scaled_dot_product_attention` over the packed rows,
  the reference's own fallback (`minimax_h3_transformer.py:410-439`); the
  sequence holds no padding row, so no key is masked;
- the module constant `_BF16_DTYPE` of the two reference modules is set to
  FP32 and every module is converted with `Module.float()`, so every
  parameter and every rounding point is an FP32 one;
- every other framework name the modules import is an inert stand-in that
  no CPU path reads.

No reference source is modified. The sparse attention is not generated: its
block-sparse kernels have no CPU path.

Run from this directory:

    ../../../../../../.venv/bin/python generate_transformer.py \
        --reference ../../../../../../refs/vllm-omni

It writes `transformer/model.safetensors` (a tiny checkpoint in the released
checkpoint's names) and `transformer.expected.safetensors` (the inputs and
the reference's outputs) beside this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

PACKAGE = "vllm_omni.diffusion.models.minimax_h3"
LAYERS = "vllm_omni.diffusion.layers"

# The tiny network: two blocks of two heads of sixteen channels, twelve of
# them rotated, and one refiner block.
HIDDEN = 32
HEADS = 2
HEAD_DIM = 16
FFN = 48
LAYERS_COUNT = 2
REFINER_LAYERS = 1
VIDEO_CHANNELS = 4
AUDIO_CHANNELS = 8
TEXT_DIM = 24
FREQUENCY_DIM = 16
TIME_HIDDEN = 32
TIME_DIM = 16
ROPE_FREQUENCIES = 2
ROPE_THETA = 10000.0

# The packed sequence the block runs over, the prompt the refiner refines
# and the distinct timesteps of the step.
ROWS = 40
PROMPT_ROWS = 9
TIMESTEPS = (0.25, 0.5, 0.999, 1.0)


class _Absent:
    """A framework value no CPU path reads: falsy, callable, any attribute."""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return self

    def __call__(self, *args, **kwargs):
        # As a decorator it leaves the function it decorates unchanged.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return self

    def __bool__(self) -> bool:
        return False


class _Stub(types.ModuleType):
    """A stand-in for a framework module: every attribute is absent."""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Absent()


class Linear(nn.Linear):
    """The framework's parallel linear layers on one rank."""

    def __init__(self, input_size, output_size, bias=True, **_):
        if not isinstance(output_size, int):
            output_size = sum(output_size)
        super().__init__(input_size, output_size, bias=bias)

    def forward(self, x):
        return super().forward(x), None


class QkvLinear(Linear):
    """The fused query, key and value projection: `[q | k | v]` rows."""

    def __init__(self, hidden_size, head_size, total_num_heads, **options):
        super().__init__(
            hidden_size, 3 * total_num_heads * head_size, bias=options["bias"]
        )
        self.num_heads = total_num_heads
        self.num_kv_heads = options.get("total_num_kv_heads", total_num_heads)


class _Backend:
    """What the reference asks of the attention backend it packs for."""

    supports_prefix_kv_slicing = True

    @staticmethod
    def supports_packed_mask_free() -> bool:
        return True

    @staticmethod
    def supports_multi_doc_packed_varlen() -> bool:
        return False

    @staticmethod
    def get_name() -> str:
        return "SDPA"


class Attention(nn.Module):
    """The attention layer: scaled dot-product attention over the rows."""

    def __init__(self, *, softmax_scale, **_):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.attn_backend = _Backend

    def effective_attention(self):
        return None, _Backend

    def add_schedule_candidate_check(self, check) -> None:
        del check

    def forward(self, query, key, value, metadata):
        del metadata
        # `[1, rows, heads, head_dim]` in, heads-major inside the product.
        attended = nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            scale=self.softmax_scale,
        )
        return attended.transpose(1, 2)


def _stub(name: str) -> types.ModuleType:
    """Register a stand-in for `name` and for every parent package."""
    parts = name.split(".")
    for depth in range(1, len(parts) + 1):
        module = ".".join(parts[:depth])
        if module not in sys.modules:
            stub = _Stub(module)
            stub.__path__ = []
            sys.modules[module] = stub
    return sys.modules[name]


def _load(name: str, path: Path):
    """Load the reference source at `path` as the module `name`."""
    _stub(name.rpartition(".")[0])
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_reference(root: Path):
    """The reference's block and transformer modules."""
    diffusion = root / "vllm_omni" / "diffusion"
    model = diffusion / "models" / "minimax_h3"

    for name in (
        "cache_dit",
        "vllm.distributed",
        "vllm.logger",
        "vllm.platforms",
        "vllm.triton_utils",
        "vllm.utils.torch_utils",
        "vllm.model_executor.model_loader.weight_utils",
        "vllm_omni.platforms",
        "vllm_omni.diffusion.envs",
        "vllm_omni.diffusion.forward_context",
        "vllm_omni.diffusion.attention.backends.abstract",
        "vllm_omni.diffusion.attention.backends.flash_attn",
        "vllm_omni.diffusion.attention.backends.vdnh3_attn",
        "vllm_omni.diffusion.cache.cachedit",
        "vllm_omni.diffusion.distributed.sp_plan",
        "vllm_omni.diffusion.models.host_weight_contract",
        "vllm_omni.diffusion.offloader.config",
        "vllm_omni.diffusion.offloader.module_residency",
        f"{LAYERS}.rope",
        f"{PACKAGE}.adaln_cache",
        f"{PACKAGE}.attention.fastvideo_h3",
        f"{PACKAGE}.controlnet",
        f"{PACKAGE}.fasth3",
        f"{PACKAGE}.vdnh3",
    ):
        _stub(name)

    linear = _stub("vllm.model_executor.layers.linear")
    linear.ColumnParallelLinear = Linear
    linear.RowParallelLinear = Linear
    linear.MergedColumnParallelLinear = Linear
    linear.QKVParallelLinear = QkvLinear
    _stub("vllm_omni.diffusion.attention.layer").Attention = Attention

    # The layers the modules call, each on the path it takes on CPU.
    for layer in (
        "custom_op",
        "norm",
        "activation",
        "indexed_modulation",
        "fused_qk_norm_rope",
    ):
        _load(f"{LAYERS}.{layer}", diffusion / "layers" / f"{layer}.py")

    blocks = _load(
        f"{PACKAGE}.minimax_h3_blocks", model / "minimax_h3_blocks.py"
    )
    transformer = _load(
        f"{PACKAGE}.minimax_h3_transformer", model / "minimax_h3_transformer.py"
    )
    blocks._BF16_DTYPE = torch.float32
    transformer._BF16_DTYPE = torch.float32

    return blocks, transformer


def checkpoint(generator: torch.Generator) -> dict[str, torch.Tensor]:
    """A tiny checkpoint in the released checkpoint's names."""
    inner = HEADS * HEAD_DIM
    tensors: dict[str, torch.Tensor] = {}

    def linear(name: str, outputs: int, inputs: int, bias: bool) -> None:
        scale = inputs**-0.5
        tensors[f"{name}.weight"] = scale * torch.randn(
            outputs, inputs, generator=generator
        )
        if bias:
            tensors[f"{name}.bias"] = 0.1 * torch.randn(
                outputs, generator=generator
            )

    def gain(name: str, width: int) -> None:
        tensors[f"{name}.weight"] = 1.0 + 0.1 * torch.randn(
            width, generator=generator
        )

    linear("proj_in", HIDDEN, 4 * VIDEO_CHANNELS, True)
    linear("audio_proj_in", HIDDEN, AUDIO_CHANNELS, True)
    linear("proj_out", 4 * VIDEO_CHANNELS, HIDDEN, True)
    linear("audio_proj_out", AUDIO_CHANNELS, HIDDEN, True)
    linear("context_embedder", HIDDEN, TEXT_DIM, True)
    linear("time_embedder.linear_1", TIME_HIDDEN, FREQUENCY_DIM, True)
    linear("time_embedder.linear_2", TIME_DIM, TIME_HIDDEN, True)
    linear("norm_out.linear", 2 * HIDDEN, TIME_DIM, True)
    gain("norm_out.norm", HIDDEN)
    gain("token_refiner.final_norm", HIDDEN)

    blocks = [
        f"token_refiner.refiner_blocks.{index}"
        for index in range(REFINER_LAYERS)
    ] + [f"transformer_blocks.{index}" for index in range(LAYERS_COUNT)]
    for block in blocks:
        for projection in ("to_q", "to_k", "to_v"):
            linear(f"{block}.attn.{projection}", inner, HIDDEN, False)
        linear(f"{block}.attn.to_out.0", HIDDEN, inner, False)
        linear(f"{block}.ff.net.0.proj", 2 * FFN, HIDDEN, False)
        linear(f"{block}.ff.net.2", HIDDEN, FFN, False)
        for norm, width in (
            ("norm1", HIDDEN),
            ("norm2", HIDDEN),
            ("attn.norm_q", HEAD_DIM),
            ("attn.norm_k", HEAD_DIM),
        ):
            gain(f"{block}.{norm}", width)
        if block.startswith("transformer_blocks"):
            linear(f"{block}.adaln_proj.linear", 18 * HIDDEN, TIME_DIM, True)

    return tensors


def assign(module: nn.Module, values: dict[str, torch.Tensor]) -> None:
    """Set the parameters of a reference module from their native names."""
    # A norm's gain takes its element type from a default bound at import.
    module.float()
    parameters = dict(module.named_parameters())
    assert set(parameters) == set(values), (
        sorted(set(parameters) ^ set(values)),
    )
    with torch.no_grad():
        for name, value in values.items():
            parameters[name].copy_(value)


def block_weights(tensors: dict, block: str, adaln: bool) -> dict:
    """One block's parameters in the reference's native names.

    The release stores the three attention projections apart and the
    feed-forward's value half before its gate half; the native block fuses
    `[q | k | v]` and puts the gate first (`fasth3.py:183-194`).
    """

    def weight(name: str) -> torch.Tensor:
        return tensors[f"{block}.{name}.weight"]

    value, gate = weight("ff.net.0.proj").chunk(2, dim=0)
    native = {
        "norm1.weight": weight("norm1"),
        "norm2.weight": weight("norm2"),
        "attn.qkv_proj.weight": torch.cat(
            [weight("attn.to_q"), weight("attn.to_k"), weight("attn.to_v")]
        ),
        "attn.q_norm.weight": weight("attn.norm_q"),
        "attn.k_norm.weight": weight("attn.norm_k"),
        "attn.out_proj.weight": weight("attn.to_out.0"),
        "mlp.fc1.weight": torch.cat([gate, value]),
        "mlp.fc2.weight": weight("ff.net.2"),
    }
    if adaln:
        native["adaln_proj.linear.weight"] = weight("adaln_proj.linear")
        native["adaln_proj.linear.bias"] = tensors[
            f"{block}.adaln_proj.linear.bias"
        ]
    return native


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True)
    arguments = parser.parse_args()

    blocks, transformer = load_reference(arguments.reference.resolve())
    generator = torch.Generator().manual_seed(20261112)
    tensors = checkpoint(generator)

    arch = blocks.MiniMaxH3DiTArchConfig(
        num_layers=LAYERS_COUNT,
        token_refiner_num_layers=REFINER_LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
        ffn_hidden_size=FFN,
        latents_dim=VIDEO_CHANNELS,
        audio_latents_dim=AUDIO_CHANNELS,
        text_dim=TEXT_DIM,
        timestep_input_dim=FREQUENCY_DIM,
        time_embed_hidden_size=TIME_HIDDEN,
        time_embed_dim=TIME_DIM,
        adaln_out_features=18 * HIDDEN,
        final_adaln_out_features=2 * HIDDEN,
        rope_inv_freq_len=ROPE_FREQUENCIES,
    )

    # The reference modules, with the checkpoint's values.
    embedder = transformer.MiniMaxH3TimeEmbedder(arch, prefix="time_embedder")
    assign(
        embedder,
        {
            "proj_in.weight": tensors["time_embedder.linear_1.weight"],
            "proj_in.bias": tensors["time_embedder.linear_1.bias"],
            "proj_out.weight": tensors["time_embedder.linear_2.weight"],
            "proj_out.bias": tensors["time_embedder.linear_2.bias"],
        },
    )

    stack = []
    for index in range(LAYERS_COUNT):
        block = blocks.MiniMaxH3DiTBlock(
            arch, None, prefix=f"blocks.{index}", layer_index=index
        )
        assign(
            block,
            block_weights(tensors, f"transformer_blocks.{index}", adaln=True),
        )
        stack.append(block)

    final = transformer.MiniMaxH3FinalLayer(arch, None, prefix="final_layer")
    assign(
        final,
        {
            "norm.weight": tensors["norm_out.norm.weight"],
            "adaln_proj.linear.weight": tensors["norm_out.linear.weight"],
            "adaln_proj.linear.bias": tensors["norm_out.linear.bias"],
            "video_out.weight": tensors["proj_out.weight"],
            "video_out.bias": tensors["proj_out.bias"],
            "audio_out.weight": tensors["audio_proj_out.weight"],
            "audio_out.bias": tensors["audio_proj_out.bias"],
        },
    )

    refiner = transformer.MiniMaxH3TokenRefiner(
        arch, None, prefix="token_refiner"
    )
    native = {"final_norm.weight": tensors["token_refiner.final_norm.weight"]}
    for index in range(REFINER_LAYERS):
        weights = block_weights(
            tensors, f"token_refiner.refiner_blocks.{index}", adaln=False
        )
        native.update(
            {f"blocks.{index}.{name}": value for name, value in weights.items()}
        )
    assign(refiner, native)

    rope = transformer.MiniMaxH3Rope(ROPE_FREQUENCIES)
    # The network computes the buffer the release does not store
    # (`minimax_h3_transformer.py:1102-1106`).
    exponent = torch.arange(0, 2 * ROPE_FREQUENCIES, 2, dtype=torch.float32)
    rope.inv_freq.copy_(
        1.0 / (ROPE_THETA ** (exponent / (2 * ROPE_FREQUENCIES)))
    )

    # The inputs: packed rows, their coordinates, and each row's timestep
    # and tag.
    hidden = torch.randn(ROWS, HIDDEN, generator=generator)
    positions = torch.rand(ROWS, 3, generator=generator) * 40.0
    timesteps = torch.tensor(TIMESTEPS, dtype=torch.float32)
    groups = torch.randint(0, len(TIMESTEPS), (ROWS,), generator=generator)
    tags = torch.randint(0, 3, (ROWS,), generator=generator)
    features = torch.randn(PROMPT_ROWS, TEXT_DIM, generator=generator)
    combined = groups * 3 + tags
    offsets = torch.tensor([0, ROWS, ROWS], dtype=torch.int32)

    with torch.no_grad():
        embedding = embedder(timesteps)
        table = transformer._build_rope_table(rope(positions[None]))

        expected = {
            "hidden": hidden,
            "positions": positions,
            "timesteps": timesteps,
            "groups": groups.to(torch.int32),
            "tags": tags.to(torch.int32),
            "features": features,
            "time_embedding": embedding,
            "rope_table": table,
            "final_modulation": torch.cat(final.adaln_proj(embedding), dim=-1),
        }

        stream = hidden
        for index, block in enumerate(stack):
            expected[f"block{index}.modulation"] = torch.cat(
                block.adaln_proj(embedding), dim=-1
            )
            stream = block(
                stream,
                t_emb=embedding,
                combined_indices=combined,
                rope_table=table,
                cu_seqlens=offsets,
                max_seqlen=ROWS,
                packed_total=ROWS,
            )
            expected[f"block{index}.out"] = stream

        video, audio = final(stream, t_emb=embedding, inverse_indices=groups)
        expected["video_velocity"] = video
        expected["audio_velocity"] = audio

        embedded = nn.functional.linear(
            features,
            tensors["context_embedder.weight"],
            tensors["context_embedder.bias"],
        )
        expected["refined"] = refiner(
            embedded,
            cu_seqlens=torch.tensor([0, PROMPT_ROWS], dtype=torch.int32),
            max_seqlen=PROMPT_ROWS,
        )

    out = Path(__file__).resolve().parent / "transformer"
    out.mkdir(exist_ok=True)
    save_file(
        {name: value.contiguous() for name, value in tensors.items()},
        out / "model.safetensors",
    )
    save_file(
        {name: value.contiguous() for name, value in expected.items()},
        out.with_suffix(".expected.safetensors"),
    )
    print(f"{len(tensors)} checkpoint tensors, {len(expected)} expected")


if __name__ == "__main__":
    main()
