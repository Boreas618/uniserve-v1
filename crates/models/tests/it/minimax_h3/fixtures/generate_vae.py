"""Generates the fixtures of `vae_reference.rs` from the reference.

Each case builds a tiny module of the SGLang MiniMax-H3 VAE with random FP32
parameters, runs it on the CPU, and writes `vae/<case>/model.safetensors`:
the parameters under the names the released checkpoint stores them by, the
inputs as `fixture.<name>` and the reference's output as `fixture.expected`.

The reference packages import frameworks this environment does not hold, so
the source files are loaded by path and the framework names they import are
stubbed in `sys.modules`. Every stub is a name the CPU FP32 path never calls,
or a switch whose CPU value selects the reference's own torch expression
(`stub_frameworks`). One hyperparameter is changed: the group count of the
encoder's group normalizations, which the reference fixes at 32 and the tiny
modules state as 2 (`GROUPS`), so that widths of four and eight channels are
legal.

Run from the repository root:

    .venv/bin/python crates/models/tests/it/minimax_h3/fixtures/generate_vae.py
"""

import argparse
import enum
import importlib.util
import logging
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file

VAES = "sglang.multimodal_gen.runtime.models.vaes"
VIDEO = f"{VAES}.minimax_h3_video_vae"
AUDIO = f"{VAES}.minimax_h3_audio_vae"

# The group count of the tiny encoder's normalizations; the reference's is 32.
GROUPS = 2

# The latent statistics of the tiny audio VAE, which the tests' config states.
AUDIO_MEAN = [0.25, -0.5, 0.125, 0.75]
AUDIO_STD = [1.5, 0.75, 2.0, 1.25]


def package(name: str) -> types.ModuleType:
    """Returns the empty package `name`, registered with those above it."""
    parts = name.split(".")
    for depth in range(1, len(parts) + 1):
        prefix = ".".join(parts[:depth])

        if prefix not in sys.modules:
            module = types.ModuleType(prefix)
            module.__path__ = []
            sys.modules[prefix] = module

            if depth > 1:
                parent = sys.modules[".".join(parts[: depth - 1])]
                setattr(parent, parts[depth - 1], module)

    return sys.modules[name]


def stub(name: str, **attributes) -> None:
    """Registers the module `name` holding `attributes`."""
    module = package(name)

    for key, value in attributes.items():
        setattr(module, key, value)


def load(name: str, path: Path) -> types.ModuleType:
    """Loads the reference source file `path` as the module `name`."""
    package(name.rsplit(".", 1)[0])

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stub_frameworks() -> None:
    """Stubs the framework names the reference files import.

    On the CPU in FP32 the reference takes its own torch expressions: no
    platform is CUDA, so the attention is `F.scaled_dot_product_attention`,
    the gated activation is `act(gate) * value` and the scaled residual is
    `residual + x * scale`.
    """

    def unreachable(*_args, **_kwargs):
        raise AssertionError("a CUDA fast path ran on the CPU")

    class Platform:
        def is_cuda(self) -> bool:
            return False

        def is_npu(self) -> bool:
            return False

        def is_rocm(self) -> bool:
            return False

        def is_sm120(self) -> bool:
            return False

    class AttentionBackendEnum(enum.Enum):
        FA = "fa"
        TORCH_SDPA = "torch_sdpa"

    stub(
        "diffusers.utils",
        logging=types.SimpleNamespace(get_logger=logging.getLogger),
    )
    stub("diffusers.utils.torch_utils", maybe_allow_in_graph=lambda cls: cls)

    # No fused kernel: `None` selects the reference's `residual + x * scale`.
    stub(
        "sglang.kernels.ops.diffusion",
        try_fused_scaled_residual_add_exact=lambda residual, x, scale: None,
    )
    stub(
        "sglang.kernels.ops.activation.activation",
        silu_and_mul_with_activation_rounding=unreachable,
    )

    # The released default; in FP32 both values are the same arithmetic.
    stub(
        "sglang.multimodal_gen.envs",
        MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM=True,
    )
    stub(
        "sglang.multimodal_gen.runtime.layers.activation",
        SiluAndMul=unreachable,
    )
    stub(
        "sglang.multimodal_gen.runtime.layers.attention",
        USPAttention=unreachable,
    )
    stub(
        "sglang.multimodal_gen.runtime.platforms",
        current_platform=Platform(),
        AttentionBackendEnum=AttentionBackendEnum,
    )


def randomize(module: torch.nn.Module, generator: torch.Generator) -> None:
    """Replaces every parameter of `module` with random values.

    The magnitudes keep the activations near one, and no parameter is left at
    an initial value that would hide it: a gain of one, a scale of zero.
    """

    def normal(shape, scale: float) -> torch.Tensor:
        return torch.randn(shape, generator=generator) * scale

    vectors = ("alpha", "original0", "register_tokens")
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            shape = parameter.shape

            if parameter.ndim > 1 and leaf not in vectors:
                # A projection or a filter: unit variance over its fan-in.
                fan_in = parameter[0].numel()
                parameter.copy_(normal(shape, fan_in**-0.5))
            elif leaf == "original0":
                # A weight norm's magnitude, positive.
                uniform = torch.rand(shape, generator=generator)
                parameter.copy_(0.75 + 0.5 * uniform)
            elif leaf == "weight" or (leaf == "alpha" and parameter.ndim == 3):
                # A normalization's gain or a Snake's raw frequency: near one.
                parameter.copy_(1.0 + normal(shape, 0.1))
            elif leaf in ("scale1", "scale2"):
                parameter.copy_(normal(shape, 0.5))
            else:
                # A bias, or a SnakeBeta parameter in log space.
                parameter.copy_(normal(shape, 0.1))


def checkpoint_names(state: dict, renames: list[tuple[str, str]]) -> dict:
    """Returns `state` under the released checkpoint's names.

    `renames` are applied in order, and torch's weight-norm parametrization is
    stored as the `weight_g` and `weight_v` pair.
    """
    renames = [
        *renames,
        (".parametrizations.weight.original0", ".weight_g"),
        (".parametrizations.weight.original1", ".weight_v"),
    ]
    named = {}

    for name, tensor in state.items():
        for old, new in renames:
            name = name.replace(old, new)
        named[name] = tensor

    return named


def write(directory: Path, case: str, tensors: dict) -> None:
    """Writes one case as `<directory>/<case>/model.safetensors`, in FP32."""
    target = directory / case
    target.mkdir(parents=True, exist_ok=True)

    packed = {}
    for name, tensor in tensors.items():
        assert tensor.dtype == torch.float32, (name, tensor.dtype)
        assert torch.isfinite(tensor).all(), name
        packed[name] = tensor.detach().clone().contiguous()

    save_file(packed, str(target / "model.safetensors"))
    elements = sum(tensor.numel() for tensor in packed.values())
    print(f"{case}: {len(packed)} tensors, {elements} elements")


def resnet_block(cnn, generator: torch.Generator) -> dict:
    """The encoder's residual block with a shortcut: `ResnetBlock3D`."""
    block = cnn.ResnetBlock3D(
        4, 8, padding_mode="reflect", causal=True, use_t_isolated_gn=True
    ).eval()
    randomize(block, generator)

    clip = torch.randn((1, 4, 3, 6, 6), generator=generator)
    with torch.no_grad():
        expected = block(clip)

    renames = [("nin_shortcut", "conv_shortcut")]
    return {
        **checkpoint_names(block.state_dict(), renames),
        "fixture.input": clip,
        "fixture.expected": expected,
    }


def video_encoder(cnn, generator: torch.Generator) -> dict:
    """The encoder network of four stages and its projected moments.

    The reference's `EncoderFCN3D`, and `quant_conv` over its output as
    `AutoencoderKLLegacy.encode` applies it (`klvae.py:1319-1321`).
    """
    encoder = cnn.EncoderFCN3D(
        ch=4,
        ch_mult=[1, 1, 2, 2],
        space_down=[2, 2, 2, 2],
        time_down=[1, 2, 2, 1],
        num_res_blocks=1,
        in_channels=3,
        z_channels=2,
        double_z=True,
        padding_mode="reflect",
        causal=True,
        use_t_isolated_gn=True,
    ).eval()
    quant_conv = torch.nn.Conv3d(4, 4, 1).eval()
    randomize(encoder, generator)
    randomize(quant_conv, generator)

    pixels = torch.randn((1, 3, 5, 32, 32), generator=generator)
    with torch.no_grad():
        expected = encoder(pixels)
        moments = quant_conv(expected)

    # `down.<i>.block.<j>` and `down.<i>.downsample.conv` are diffusers'
    # `down_blocks.<i>.resnets.<j>` and `down_blocks.<i>.downsamplers.0.conv`.
    state = {}
    for name, tensor in encoder.state_dict().items():
        parts = name.split(".")

        if parts[0] == "down" and parts[2] == "block":
            parts = ["down_blocks", parts[1], "resnets", *parts[3:]]
        elif parts[0] == "down":
            parts = ["down_blocks", parts[1], "downsamplers", "0", *parts[3:]]
        state["encoder." + ".".join(parts)] = tensor

    for name, tensor in quant_conv.state_dict().items():
        state[f"quant_conv.{name}"] = tensor

    renames = [("nin_shortcut", "conv_shortcut")]
    return {
        **checkpoint_names(state, renames),
        "fixture.input": pixels,
        "fixture.expected": expected,
        "fixture.moments": moments,
    }


def transformer_block(base, vit, generator: torch.Generator) -> dict:
    """One block of the decoder over two sequences, and their rotary tables.

    The reference's `TransformerBlock`, `RotaryEmbeddingND` and
    `create_token_ids`.
    """
    heads, dim_head, rotated = 2, 8, 6
    block = base.TransformerBlock(
        heads=heads,
        dim_head=dim_head,
        norm_type="rms_norm",
        norm_affine=True,
        qk_norm_type="rms_norm",
        qk_norm_affine=False,
        ffn_activation_fn="silu",
        ffn_use_gated=True,
        eps=1e-5,
    ).eval()
    randomize(block, generator)

    # Two sequences of the twelve voxels of a [2, 2, 3] raster and five
    # trailing tokens, which read no position (`vae_vit.py:409-414`).
    sequences, suffix = 2, 5
    ids = vit.create_token_ids([2, 2, 3], "cpu", torch.float32)
    ids = torch.cat([ids, torch.zeros((1, suffix, 3))], dim=1)
    rows = ids.shape[1]

    pos_embed = base.RotaryEmbeddingND(rotated, 100.0, n_dim=3, use_angle=True)
    cos, sin = pos_embed(ids)
    rope = (
        cos.expand(sequences, -1, -1, -1),
        sin.expand(sequences, -1, -1, -1),
    )

    width = heads * dim_head
    hidden = torch.randn((sequences, rows, width), generator=generator)
    with torch.no_grad():
        expected = block(hidden, rope)

    # The reference stores one `to_qkv` whose rows are grouped per head, and
    # its `w1` gate half first; the checkpoint stores `to_q`, `to_k` and
    # `to_v`, and `ff.net.0.proj` value half first (vLLM-Omni's
    # `lightvae/checkpoint.py:109-167` states the mapping).
    state = {}
    for name, tensor in block.state_dict().items():
        if name.startswith("attn.to_qkv."):
            grouped = tensor.reshape(heads, 3, dim_head, *tensor.shape[1:])

            for index, projection in enumerate(("to_q", "to_k", "to_v")):
                key = name.replace("to_qkv", projection)
                state[key] = grouped[:, index].reshape(width, *tensor.shape[1:])
        elif name.startswith("ff.w1."):
            gate, value = tensor.chunk(2, dim=0)
            key = name.replace("ff.w1.", "ff.net.0.proj.")
            state[key] = torch.cat((value, gate), dim=0)
        elif name.startswith("ff.w2."):
            state[name.replace("ff.w2.", "ff.net.2.")] = tensor
        elif name.startswith("attn.to_out."):
            state[name.replace("attn.to_out.", "attn.to_out.0.")] = tensor
        else:
            state[name] = tensor

    # The tables hold each rotated pair's factor twice (`angles.tile(2)`); one
    # half is the `[rows, pairs]` table a half-split rotation reads.
    pairs = rotated // 2
    return {
        **{f"block.{name}": tensor for name, tensor in state.items()},
        "fixture.input": hidden.reshape(sequences * rows, width),
        "fixture.cos": cos[0, :, 0, :pairs],
        "fixture.sin": sin[0, :, 0, :pairs],
        "fixture.expected": expected.reshape(sequences * rows, width),
    }


def audio_decoder(bigvgan, generator: torch.Generator) -> dict:
    """The audio decode of one stereo track.

    The reference's `BigVGAN` (with its `AMPBlock1`, `SnakeBeta` and
    `Activation1d`) after `dec_in_proj`, driven as `DacAudioVAE.decode` and
    vLLM-Omni's `decode_latent` drive them (`audio_vae.py:325-339`,
    `vae.py:1285-1308`).
    """
    conf = bigvgan.AttrDict(
        resblock="1",
        num_mels=16,
        upsample_rates=[2, 2],
        upsample_kernel_sizes=[4, 4],
        upsample_initial_channel=8,
        resblock_kernel_sizes=[3, 5],
        resblock_dilation_sizes=[[1, 3], [1, 3]],
        use_tanh_at_final=False,
        use_bias_at_final=False,
        activation="snakebeta",
        snake_logscale=True,
    )
    vocoder = bigvgan.BigVGAN(conf).eval()
    dec_in_proj = torch.nn.Conv1d(4, 16, 1).eval()
    randomize(vocoder, generator)
    randomize(dec_in_proj, generator)

    # A quiet output convolution keeps the waveform inside the final clamp.
    with torch.no_grad():
        vocoder.conv_post.parametrizations.weight.original0.mul_(0.125)

    # The denoiser's rows: `[2 * frames, channels]`, the first stereo
    # channel's frames first.
    frames, channels = 5, 4
    rows = torch.randn((2 * frames, channels), generator=generator)
    mean = torch.tensor(AUDIO_MEAN).view(1, channels, 1)
    std = torch.tensor(AUDIO_STD).view(1, channels, 1)

    with torch.no_grad():
        latent = rows.reshape(2, frames, channels).permute(0, 2, 1)
        waveform = vocoder(dec_in_proj(latent * std + mean))

    # The generator clamps its output to [-1, 1], which execution does as it
    # quantizes; this waveform lies inside, so the clamp changed no sample.
    assert waveform.abs().max() < 1.0, waveform.abs().max()

    state = {
        f"decoder.{name}": tensor
        for name, tensor in vocoder.state_dict().items()
    }
    for name, tensor in dec_in_proj.state_dict().items():
        state[f"dec_in_proj.{name}"] = tensor

    return {
        **checkpoint_names(state, []),
        "fixture.input": rows,
        "fixture.expected": waveform.reshape(2, -1),
    }


def audio_encoder(dac, generator: torch.Generator) -> dict:
    """The audio encode of one stereo track.

    The reference's DAC `Encoder`, `AttnProjection` and `mean_proj`, driven as
    `minimax_h3_encode_reference_audio_rows` drives them
    (`reference_encoding.py:422-447`).
    """
    channels, wide, heads = 4, 16, 2
    encoder = dac.Encoder(d_model=4, strides=[2, 2], d_latent=wide).eval()
    pre_block = dac.AttnProjection(wide, channels, num_heads=heads).eval()
    mean_proj = torch.nn.Conv1d(channels, channels, 1).eval()
    for module in (encoder, pre_block, mean_proj):
        randomize(module, generator)

    # The key bias is a buffer the checkpoint stores; a nonzero one shows
    # that it is read.
    key_bias = torch.randn((wide,), generator=generator) * 0.1
    pre_block.attn.zero_k_bias.copy_(key_bias)

    samples = torch.randn((2, 1, 24), generator=generator)
    mean = torch.tensor(AUDIO_MEAN).view(1, 1, channels)
    std = torch.tensor(AUDIO_STD).view(1, 1, channels)

    with torch.no_grad():
        features = encoder(samples)
        features = pre_block(features.transpose(1, 2)).transpose(1, 2)
        latent = mean_proj(features).float().transpose(1, 2).contiguous()
        latent.sub_(mean).div_(std)
        expected = latent.reshape(-1, channels)

    state = {}
    for prefix, module in (
        ("encoder", encoder),
        ("pre_block", pre_block),
        ("mean_proj", mean_proj),
    ):
        for name, tensor in module.state_dict().items():
            state[f"{prefix}.{name}"] = tensor

    return {
        **checkpoint_names(state, []),
        "fixture.input": samples,
        "fixture.expected": expected,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[6]
    fixtures = Path(__file__).resolve().parent / "vae"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refs", type=Path, default=root / "refs")
    parser.add_argument("--out", type=Path, default=fixtures)
    arguments = parser.parse_args()

    package_path = "sglang/python/sglang/multimodal_gen/runtime/models/vaes"
    vaes = (arguments.refs / package_path).resolve()
    video, audio = vaes / "minimax_h3_video_vae", vaes / "minimax_h3_audio_vae"

    stub_frameworks()
    load(f"{VIDEO}.conv", video / "conv.py")
    norm = load(f"{VIDEO}.norm", video / "norm.py")
    cnn = load(f"{VIDEO}.vae_cnn", video / "vae_cnn.py")
    vit = load(f"{VIDEO}.vit_utils", video / "vit_utils.py")
    load(f"{VIDEO}.attention", video / "attention.py")
    base = load(f"{VIDEO}.base_module", video / "base_module.py")
    load(f"{VAES}.alias_free", vaes / "alias_free.py")
    bigvgan = load(f"{AUDIO}.bigvgan", audio / "bigvgan.py")
    dac = load(f"{AUDIO}.audio_vae", audio / "audio_vae.py")

    # The same normalization class at the tiny modules' group count.
    def group_norm(num_channels, use_t_isolated_gn=False):
        isolated = norm.TemporalIsolatedGroupNorm
        norm_class = isolated if use_t_isolated_gn else torch.nn.GroupNorm
        return norm_class(
            num_groups=GROUPS, num_channels=num_channels, eps=1e-6, affine=True
        )

    cnn.get_group_norm_3d = group_norm

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20260511)
    cases = {
        "resnet_block": lambda: resnet_block(cnn, generator),
        "video_encoder": lambda: video_encoder(cnn, generator),
        "transformer_block": lambda: transformer_block(base, vit, generator),
        "audio_decoder": lambda: audio_decoder(bigvgan, generator),
        "audio_encoder": lambda: audio_encoder(dac, generator),
    }

    for case, build in cases.items():
        write(arguments.out, case, build())


if __name__ == "__main__":
    main()
