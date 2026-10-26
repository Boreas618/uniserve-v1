"""BAGEL public text and image capabilities share checkpoint-backed experts."""

import pytest
import torch
from torch.nn import functional as F

from tests.python.fixtures.checkpoints import bagel_checkpoint, load_bagel
from uniserve.bindings.context import ExecutionContext
from uniserve.diffusion import normal_noise
from uniserve.media import image
from uniserve.models import TextInput, TextSize
from uniserve.models.inputs import PackedImageRows, TokenKind
from uniserve.nn.attention import (
    AttentionBatch,
    PagedInput,
    SequenceLengths,
    VarlenInput,
)
from uniserve.prefix_cache.units import UnitPool

pytestmark = pytest.mark.integration


def test_text_prefill_decode_and_zero_query_match_qwen_equations(tmp_path):
    reference, _, config = bagel_checkpoint(tmp_path)
    model = load_bagel(tmp_path, config)
    assert model.text.backbone is model.denoiser.backbone
    tokens = torch.tensor([1, 4, 8, 3])
    with torch.no_grad():
        expected = reference(tokens[None]).logits[0]
    cache = UnitPool.allocate(
        model.text.cache_config, num_units=1, block_size=4, device="cpu"
    )
    with ExecutionContext(
        model.text, cache=cache, attention="torch"
    ) as context:
        context.prepare(TextSize(4, 1))
        for start, stop in ((0, 3), (3, 4), (4, 4)):
            batch = AttentionBatch.single(
                PagedInput.from_blocks(
                    blocks=((0,),),
                    query_lengths=(stop - start,),
                    prefix_lengths=(start,),
                    block_size=4,
                    causal=True,
                    device="cpu",
                )
            )
            context.bind_attention(batch)
            hidden = model.text(
                TextInput(tokens[start:stop], torch.arange(start, stop), batch)
            )
            actual = model.text.compute_logits(
                hidden, token_indices=torch.arange(stop - start)
            ).gather()
            torch.testing.assert_close(
                actual, expected[start:stop], rtol=2e-2, atol=2e-3
            )


def test_image_markers_use_text_expert_and_flow_preserves_residual(tmp_path):
    _, state, config = bagel_checkpoint(tmp_path)
    model = load_bagel(tmp_path, config)
    size = image.Config(8, 8)
    # The initial latent is the seeded draw in the denoiser's native noise
    # shape, converted by ``prepare_latents``.
    sample = torch.empty((4, 8), dtype=torch.bfloat16)
    noise = torch.empty(
        (1, *model.denoiser.noise_shape("image", size)), dtype=sample.dtype
    )
    normal_noise((71,), out=(noise,))
    model.denoiser.prepare_latents(
        (size,),
        noise={"image": noise},
        state={"image": sample.unsqueeze(0)},
        constants={},
        workspace={},
    )
    expected_noise = torch.randn(
        (4, 8),
        generator=torch.Generator().manual_seed(71),
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(sample, expected_noise, rtol=0, atol=0)
    timestep = torch.tensor(0.5)
    lengths = SequenceLengths.from_lengths((6,), device="cpu")
    # One framed row of a 2x2 latent grid at temporal position 9: the
    # markers keep zero spatial coordinates, read no sample and route as
    # text.
    rows = PackedImageRows(
        sample_index=torch.tensor([-1, 0, 1, 2, 3, -1]),
        kind=torch.tensor(
            [
                TokenKind.OPEN,
                *[TokenKind.LATENT] * 4,
                TokenKind.CLOSE,
            ]
        ),
        positions=torch.tensor(
            [[9, 9, 9, 9, 9, 9], [0, 0, 0, 1, 1, 0], [0, 0, 1, 0, 1, 0]]
        ),
        timesteps=timestep.expand(6).clone(),
        token_row=torch.zeros(6, dtype=torch.int64),
        row_grid=torch.tensor([[2, 2]]),
        noise_scale=torch.ones(1),
    )
    inputs = model.denoiser.bind_packed(
        source=sample,
        rows=rows,
        attention=AttentionBatch.single(
            VarlenInput(lengths, lengths, (False,))
        ),
    )
    before = sample.clone()
    # GLIDE time features and the checkpoint's zero-update flow residual give
    # an independent closed form for the denoiser prediction, including casts.
    frequencies = torch.exp(
        -torch.log(torch.tensor(10000.0)) * torch.arange(128) / 128
    )
    angles = timestep * frequencies
    time = torch.cat((angles.cos(), angles.sin())).bfloat16().expand(4, -1)
    time = F.linear(
        time,
        state["time_embedder.mlp.0.weight"],
        state["time_embedder.mlp.0.bias"],
    )
    time = F.linear(
        F.silu(time),
        state["time_embedder.mlp.2.weight"],
        state["time_embedder.mlp.2.bias"],
    )
    hidden = F.linear(sample, state["vae2llm.weight"], state["vae2llm.bias"])
    hidden = hidden + time + state["latent_pos_embed.pos_embed"][[0, 1, 4, 5]]
    normalized = (
        hidden.float()
        * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + 1e-6)
    ).bfloat16()
    expected = F.linear(
        normalized, state["llm2vae.weight"], state["llm2vae.bias"]
    )
    with torch.no_grad():
        prediction = model.denoiser(
            inputs, state={}, constants={}, workspace={}
        )["image"][0]
    # The packed prediction holds every token; the latent tokens' rows are
    # the image's predictions.
    torch.testing.assert_close(prediction.tensor[1:5], expected, rtol=0, atol=0)
    torch.testing.assert_close(sample, before, rtol=0, atol=0)
    assert prediction.layout.shape == (6, sample.shape[1])
