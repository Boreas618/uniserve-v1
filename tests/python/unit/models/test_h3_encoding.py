"""H3 conditioning encoders partition media into independent units.

A video condition is encoded in 17-frame clips that ranks encode separately,
so the clip partition, the latent frames each clip contributes and the rows
of the complete latent are the contract a distributed encoding assembles
from. Audio conditions are encoded whole as channel-major stereo rows.
"""

import pytest
import torch

from tests.python.fixtures.h3 import audio_vae_config, video_vae_config
from uniserve.media import image, video
from uniserve.models.minimax_h3.encoding import AudioEncoder, VideoEncoder
from uniserve.models.minimax_h3.packing import video_latent_frames

pytestmark = pytest.mark.unit


def _video_encoder() -> VideoEncoder:
    # Partition and layout queries need no materialized weights.
    with torch.device("meta"):
        return VideoEncoder(video_vae_config())


def _audio_encoder() -> AudioEncoder:
    with torch.device("meta"):
        return AudioEncoder(audio_vae_config(), sample_rate=32000)


@pytest.mark.parametrize("num_frames", (1, 2, 17, 18, 22, 39, 56, 362))
def test_video_clips_partition_frames_and_latent_frames(num_frames):
    encoder = _video_encoder()
    clips = encoder.clips(num_frames)
    latents = encoder.latent_slices(num_frames)

    assert len(clips) == len(latents)
    assert clips[0].start == 0 and clips[-1].stop == num_frames
    assert latents[0].start == 0
    for earlier, later in zip(clips, clips[1:], strict=False):
        assert earlier.stop == later.start
        assert earlier.stop - earlier.start == 17
    for earlier, later in zip(latents, latents[1:], strict=False):
        assert earlier.stop == later.start
        assert earlier.stop - earlier.start == 5
    if num_frames == 1:
        assert latents == (slice(0, 1),)
    elif num_frames % 17 == 5:
        assert latents[-1].stop == video_latent_frames(num_frames)
    else:
        assert latents[-1].stop == 5 * len(clips) - 3


@pytest.mark.parametrize(
    ("height", "width"), ((768, 1344), (1344, 768), (2048, 3648))
)
@pytest.mark.parametrize("num_frames", (1, 39))
def test_video_rows_are_denoiser_patch_tokens(num_frames, height, width):
    encoder = _video_encoder()
    layout = encoder.output_layout(
        video.Config(num_frames, image.Config(height, width))
    )["video"]

    frames = encoder.latent_slices(num_frames)[-1].stop
    assert layout.shape == (frames * (height // 32) * (width // 32), 96)
    assert layout.dtype == torch.float32
    assert layout.variable_axes == (0,)


def test_video_encoding_rejects_inputs_outside_the_clip_contract():
    encoder = _video_encoder()
    frames = torch.zeros((17, 64, 64, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="one of the video's complete clips"):
        encoder.encode((frames[:10],), clips=(slice(0, 10),), num_frames=(39,))
    with pytest.raises(ValueError, match="covering their clip"):
        encoder.encode((frames[:16],), clips=(slice(17, 34),), num_frames=(39,))
    with pytest.raises(ValueError, match="32-pixel latent patches"):
        encoder.encode(
            (torch.zeros((1, 48, 64, 3), dtype=torch.uint8),),
            clips=(slice(0, 1),),
            num_frames=(1,),
        )


def test_video_encoding_takes_its_conditions_posterior_draw():
    # The posterior is sampled, so every unit takes its condition's complete
    # draw, which spans the condition's whole latent.
    encoder = _video_encoder()
    frames = torch.zeros((17, 64, 64, 3), dtype=torch.uint8)
    size = video.Config(39, image.Config(64, 64))
    shape = encoder.posterior_shape(size)
    assert shape == (1, encoder.config.latent_channels, 12, 4, 4)

    with pytest.raises(ValueError, match="takes its video's draw"):
        encoder.encode((frames,), clips=(slice(0, 17),), num_frames=(39,))
    with pytest.raises(ValueError, match="span its video's latent"):
        encoder.encode(
            (frames,),
            clips=(slice(0, 17),),
            num_frames=(39,),
            noise=(torch.zeros(shape[:2] + (5,) + shape[3:]),),
        )


def test_posterior_draw_depends_on_the_condition_size_alone():
    # Every condition of one size samples the same posterior, so a draw
    # serves every unit, round and request of that size.
    encoder = _video_encoder()
    size = video.Config(22, image.Config(64, 96))
    first, second = (torch.empty(encoder.posterior_shape(size)) for _ in "ab")
    encoder.draw_noise(size, first)
    encoder.draw_noise(size, second)

    assert torch.equal(first, second)
    assert first.std() > 0.5


@pytest.mark.parametrize(
    ("num_samples", "frames"), ((160000, 200), (160001, 201), (799, 1))
)
def test_audio_rows_hold_both_channels_of_whole_latent_frames(
    num_samples, frames
):
    encoder = _audio_encoder()
    layout = encoder.output_layout(num_samples)["audio"]

    assert encoder.latent_frames(num_samples) == frames
    assert layout.shape == (2 * frames, 32)
    assert layout.dtype == torch.float32


def test_audio_encoding_rejects_more_than_two_channels():
    encoder = _audio_encoder()

    with pytest.raises(ValueError, match="mono or stereo"):
        encoder.encode((torch.zeros((800, 3)),))
