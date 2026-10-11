"""A packed vision call serves every packing of its image slots.

``PatchEncoder.encode_packed`` encodes images laid out in fixed slots of
``max_patches`` patch rows (``packed_grids``). Each image's soft tokens must
equal the features ``encode`` returns for it alone, whatever the images'
aspect ratios, and one CUDA graph captured over empty slots must replay any
later packing with those same features. The worker's encoder runner packs
a batch's images into captured slot counts and replays them. The
DiffusionGemma vision tower runs on the SM100 kernels at the released head
width.
"""

import base64
import io

import numpy as np
import pytest
import torch
import tvm_ffi
from PIL import Image

from tests.python.fixtures.checkpoints import diffusion_gemma_checkpoint
from tests.python.fixtures.image_harness import image_harness
from tests.python.fixtures.stub import load_model, read_config
from uniserve_models import catalog as models
from uniserve_models import interface
from uniserve_models.bindings.context import ExecutionContext
from uniserve_models.interface import graphs
from uniserve_models.loading import weights
from uniserve_models.media.pixels import prepare_host_image
from uniserve_models.model import VisionInput

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
    """Write ``images`` into the leading slots of ``pixels``; return grids."""
    capacity = encoder.max_patches
    for index, image in enumerate(images):
        pixels[index * capacity : index * capacity + image.shape[0]] = image
    return torch.tensor(
        encoder.packed_grids(shapes, slots), dtype=torch.long, device=DEVICE
    )


def _features(encoder, output, shapes):
    """Each packed image's soft tokens, which lead its slot."""
    tokens = encoder.max_patches // encoder.downsample**2
    return tuple(
        output[index * tokens : index * tokens + rows * columns // 9]
        for index, (rows, columns) in enumerate(shapes)
    )


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

        # Four slots hold the three images and one empty slot of padding.
        slots = 4
        pixels = torch.zeros(slots * encoder.max_patches, 48, device=DEVICE)
        grids = _pack(encoder, pixels, images, SHAPES, slots)
        packed = _features(
            encoder, encoder.encode_packed(pixels, grids), SHAPES
        )
        for actual, expected in zip(packed, alone, strict=True):
            torch.testing.assert_close(actual, expected)

        # A graph captured over empty slots replays two later packings.
        static_pixels = torch.zeros_like(pixels)
        static_grids = torch.tensor(
            encoder.packed_grids((), slots), dtype=torch.long, device=DEVICE
        )
        encoder.encode_packed(static_pixels, static_grids)
        graph = graphs.capture(
            lambda: encoder.encode_packed(static_pixels, static_grids),
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


def _payload(height: int, width: int, seed: int) -> str:
    """A random RGB PNG of ``height`` x ``width`` pixels."""
    raw = np.random.default_rng(seed).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )
    encoded = io.BytesIO()
    Image.fromarray(raw).save(encoded, format="PNG")
    return base64.b64encode(encoded.getvalue()).decode()


def _load(root):
    description, _ = read_config(stub=False, path=str(root), device="cuda:0")
    return load_model(description, modules=["vision_encoder"], device="cuda:0")


@torch.inference_mode()
def test_worker_packed_vision_preserves_rows_and_reports_replay(tmp_path):
    diffusion_gemma_checkpoint(tmp_path, vision=VISION)
    table = interface.table()
    model = _load(tmp_path)
    # The processor scales each image to the tower's patch budget, so
    # aspect ratios near one keep every grid side within the fixture's
    # position table.
    payloads = [
        _payload(height, width, seed)
        for seed, (height, width) in enumerate(((64, 64), (48, 64), (64, 48)))
    ]

    # Each image encoded alone through the library is the reference.
    encoder = model.module.vision_encoder
    expected = []
    with ExecutionContext(encoder) as context:
        context.prepare(None)
        with context.activate():
            for payload in payloads:
                host = prepare_host_image(
                    model.image_processor,
                    "vit",
                    payload,
                    input_images=len(payloads),
                    pin=False,
                )
                pixels = host.pixels.to(DEVICE)
                (features,) = encoder.encode(
                    VisionInput(
                        (pixels,),
                        (torch.tensor([host.grid_shape], device=DEVICE),),
                        (host.grid_shape,),
                    )
                )
                expected.append(features)

    # Batches of three images (one replay), of seventeen images (sixteen
    # captured slots, then one) and of two images (one replay). Mixed patch
    # counts share packed calls.
    orders = [[0, 1, 2], [0, 1, 2] * 5 + [2, 0], [2, 0]]
    results = list(
        image_harness().encode(
            tvm_ffi.convert(
                {
                    "table": table,
                    "loaded": model,
                    "component": "model",
                    "root": "vision_encoder",
                    "path": "",
                    "tower": "vit",
                    "payloads": payloads,
                    "orders": orders,
                    "device": 0,
                    "capture": True,
                }
            )
        )
    )
    for order, replays in zip(orders, (1, 2, 1), strict=True):
        features, counts = results[: len(order)], results[len(order)]
        results = results[len(order) + 1 :]
        assert list(counts) == [replays, 0]
        for value, index in zip(features, order, strict=True):
            torch.testing.assert_close(
                torch.from_dlpack(value), expected[index]
            )
