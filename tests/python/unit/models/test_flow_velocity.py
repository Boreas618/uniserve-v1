"""SenseNova's velocity heads predict each packed image as it is alone."""

import pytest
import torch
from torch import nn

from uniserve.models.sensenova_u1 import flow
from uniserve.nn.functional import patchify

pytestmark = pytest.mark.unit

# 8-channel token features decoded to 4x4-pixel RGB tokens.
HIDDEN, STRIDE = 8, 4


def _pixel_velocity() -> flow.Velocity:
    torch.manual_seed(3)
    return flow.Velocity(
        nn.Identity(),
        flow.Decoder(HIDDEN, hidden_size=8, final_upscale=STRIDE // 4),
        patch_size=STRIDE,
    )


def _alone(velocity, hidden, timestep, sample, rows, columns):
    """One image's velocity from its whole token grid."""
    spatial = hidden.reshape(1, rows, columns, HIDDEN).permute(0, 3, 1, 2)
    prediction = patchify(velocity.decoder(spatial), patch_size=STRIDE)[0]
    return (prediction - sample) / (1 - timestep).clamp_min(0.02)


def test_the_pixel_decoder_predicts_each_image_region_alone():
    velocity = _pixel_velocity()
    # A 2x3 token image, a 2x2 one, then two padding tokens.
    hidden = torch.randn(12, HIDDEN)
    sample = torch.randn(12, 3 * STRIDE**2)
    timesteps = torch.tensor([0.25] * 6 + [0.75] * 4 + [0.0] * 2)
    regions = ((0, 2, 3), (6, 2, 2))

    with torch.no_grad():
        packed = velocity(hidden, timesteps, sample, regions)
        first = _alone(velocity, hidden[:6], timesteps[0], sample[:6], 2, 3)
        second = _alone(
            velocity, hidden[6:10], timesteps[6], sample[6:10], 2, 2
        )

    # The region and the reference convolve differently laid out inputs, so
    # only the FP32 summation order of the two 3x3 convolutions (18-term
    # reductions over two channels) may differ: about 2 * 18 * eps32 ~ 4e-6
    # of the values' scale, which 1e-5 bounds.
    assert packed.dtype == torch.float32
    torch.testing.assert_close(packed[:6], first, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(packed[6:10], second, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(packed[10:]).all()


def test_the_pixel_decoder_is_an_eager_region_reading_host_regions():
    velocity = _pixel_velocity()
    hidden, sample = torch.randn(4, HIDDEN), torch.randn(4, 3 * STRIDE**2)
    timesteps = torch.zeros(4)
    with pytest.raises(ValueError, match="host regions"):
        velocity(hidden, timesteps, sample, None)
    with pytest.raises(ValueError, match="outside"):
        velocity(hidden, timesteps, sample, ((2, 2, 2),))


def test_mlp_heads_predict_each_token_alone():
    torch.manual_seed(4)
    head = nn.Sequential(
        nn.Linear(HIDDEN, 16), nn.GELU(), nn.Linear(16, 3 * STRIDE**2)
    )
    velocity = flow.Velocity(head, nn.Identity(), patch_size=STRIDE)
    hidden = torch.randn(5, HIDDEN)
    sample = torch.randn(5, 3 * STRIDE**2)
    timesteps = torch.tensor([0.1, 0.1, 0.5, 0.99, 0.0])

    with torch.no_grad():
        packed = velocity(hidden, timesteps, sample, None)
        alone = torch.cat(
            [
                velocity(
                    hidden[token : token + 1],
                    timesteps[token : token + 1],
                    sample[token : token + 1],
                    None,
                )
                for token in range(5)
            ]
        )

    # One GEMM over five rows against five one-row GEMMs: only the FP32
    # summation order of the 16-wide reduction may differ.
    torch.testing.assert_close(packed, alone, rtol=1e-5, atol=1e-5)
