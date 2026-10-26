"""A packed vision call serves every packing of its patch rows.

``PatchEncoder.encode_segments`` encodes images packed back to back and
described by device segment grids (``uniserve.executor.image.plan.
segment_grids``). Each image's soft tokens must equal the features
``encode`` returns for it alone, whatever the images' aspect ratios, and one
CUDA graph captured over padding alone must replay any later packing of its
patch rows and segments with those same features. The DiffusionGemma vision
tower runs on the SM100 kernels at the released head width.
"""

import pytest
import torch

from tests.python.fixtures.checkpoints import diffusion_gemma_checkpoint
from uniserve import cuda_graph
from uniserve.bindings.context import ExecutionContext
from uniserve.executor.image.plan import segment_grids
from uniserve.loading import weights
from uniserve.models import VisionInput
from uniserve.models import registry as models

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

DEVICE = torch.device("cuda:0")
# Vision heads of the released width, 72, which FlashAttention-4 serves.
VISION = {
    "hidden_size": 144,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 72,
    "intermediate_size": 96,
}
# Patch grids of three aspect ratios, sides in multiples of the 3 x 3
# pooling kernel.
SHAPES = ((3, 6), (6, 3), (6, 6))


def _encoder(root):
    diffusion_gemma_checkpoint(root, vision=VISION)
    return models.load_model(
        models.read_config(root),
        device=DEVICE,
        weights=weights.Config(dtype=torch.bfloat16),
    ).model.vision_encoder


def _pack(encoder, pixels, images, shapes, slots):
    """Write ``images`` back to back into ``pixels``; return their grids."""
    offset = 0
    for image in images:
        pixels[offset : offset + image.shape[0]] = image
        offset += image.shape[0]
    return torch.tensor(
        segment_grids(shapes, slots, pixels.shape[0], encoder.downsample),
        dtype=torch.long,
        device=DEVICE,
    )


def _features(encoder, output, shapes):
    """Each packed image's soft tokens, in packing order."""
    unit = encoder.downsample**2
    tokens, offset = [], 0
    for rows, columns in shapes:
        count = rows * columns // unit
        tokens.append(output[offset : offset + count])
        offset += count
    return tuple(tokens)


@torch.inference_mode()
def test_packed_images_encode_as_they_do_alone(tmp_path):
    encoder = _encoder(tmp_path)
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    images = tuple(
        torch.rand(rows * columns, 48, device=DEVICE, generator=generator)
        for rows, columns in SHAPES
    )
    stream = torch.cuda.Stream(DEVICE)
    context = ExecutionContext(
        encoder, stream=stream, derive_host_lengths=False
    )
    context.prepare(None)
    with context, context.activate():
        alone = tuple(
            encoder.encode(
                VisionInput(
                    (image,),
                    (torch.tensor([shape], device=DEVICE),),
                    (shape,),
                )
            )[0]
            for image, shape in zip(images, SHAPES, strict=True)
        )

        # Four segments hold the three images and one empty segment, then
        # the padding rows.
        slots = 4
        pixels = torch.zeros(slots * encoder.max_patches, 48, device=DEVICE)
        grids = _pack(encoder, pixels, images, SHAPES, slots)
        packed = _features(
            encoder, encoder.encode_segments(pixels, grids), SHAPES
        )
        for actual, expected in zip(packed, alone, strict=True):
            torch.testing.assert_close(actual, expected)

        # A graph captured over padding alone replays two later packings.
        static_pixels = torch.zeros_like(pixels)
        static_grids = torch.tensor(
            segment_grids(
                (), slots, static_pixels.shape[0], encoder.downsample
            ),
            dtype=torch.long,
            device=DEVICE,
        )
        encoder.encode_segments(static_pixels, static_grids)
        graph = cuda_graph.capture(
            lambda: encoder.encode_segments(static_pixels, static_grids),
            stream=stream,
        )
        try:
            for order in ((0, 1, 2), (2, 0)):
                shapes = tuple(SHAPES[index] for index in order)
                static_grids.copy_(
                    _pack(
                        encoder,
                        static_pixels,
                        tuple(images[index] for index in order),
                        shapes,
                        slots,
                    )
                )
                replayed = _features(encoder, graph.replay(), shapes)
                for actual, index in zip(replayed, order, strict=True):
                    torch.testing.assert_close(actual, alone[index])
        finally:
            graph.release()
