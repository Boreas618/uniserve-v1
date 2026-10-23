"""BAGEL serves its image forwards from CUDA graphs captured at startup.

Under CUDA graphs the workers capture the image size a request gets
without naming one, 512x512, and its transpose (the same size): the
denoising steps, the decode of a generated image, the latent encode of a
generated image fed back as context and the write of its latent into the
prompt. A generation, a conversation whose generated images are fed back,
and an edit of an input image whose latent size is 512x512 are served; an
input image whose latent size is not captured is refused with the sizes the
workers serve. The SigLIP tower packs input images into fixed slots. Under
graphs a worker refuses any forward no captured graph holds, so each served
request ran its image forwards from graphs.

The checkpoint directory comes from ``UNISERVE_BAGEL_MODEL``.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import httpx
import pytest
from PIL import Image

from tests.python.e2e.http_helpers import (
    find_free_port,
    png_size_from_b64,
    post_sse,
    require_uniserve_binary,
    server_process,
    tiny_input_png_b64,
    worker_executable,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("bagel")]

SERVED_MODEL = "bagel"
# BAGEL's start-of-image token, which opens a generated image.
START_OF_IMAGE = 151652


def _checkpoint() -> str:
    checkpoint = os.environ.get("UNISERVE_BAGEL_MODEL")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail("UNISERVE_BAGEL_MODEL must name the BAGEL checkpoint")
    return checkpoint


def _png(width: int, height: int) -> str:
    """A base64 PNG of one color."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (180, 96, 32)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_chat(payload: str, text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{payload}"},
                },
            ],
        }
    ]


@pytest.mark.timeout(1800)
def test_image_forwards_replay_captured_graphs(tmp_path: Path) -> None:
    try:
        binary = require_uniserve_binary()
    except FileNotFoundError as error:
        pytest.fail(str(error))
    port = find_free_port()
    base = f"http://127.0.0.1:{port}"
    log = tmp_path / "server.log"
    with (
        server_process(
            [
                str(binary),
                "serve",
                _checkpoint(),
                "--served-model-name",
                SERVED_MODEL,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--worker-executable",
                str(worker_executable()),
                "--set",
                "cuda_graph_config.policy=full",
            ],
            base,
            log,
            timeout_s=1800.0,
        ),
        httpx.Client(base_url=base, timeout=600) as client,
    ):
        generated = client.post(
            "/v1/images/generations",
            json={
                "model": SERVED_MODEL,
                "prompt": "A blue square.",
                "steps": 2,
                "seed": 7,
            },
        )
        assert generated.status_code == 200, generated.text
        (image,) = generated.json()["data"]
        assert png_size_from_b64(image["b64_json"]) == (512, 512)

        # Each generated image is fed back as context: its latent encode
        # and the write of its latent into the prompt replay their graphs.
        interleaved = post_sse(
            base,
            "/v1/chat/completions",
            {
                "model": SERVED_MODEL,
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [
                    {"role": "user", "content": "Draw two red circles."}
                ],
                "modalities": ["text", "image"],
                "max_completion_tokens": 4,
                "temperature": 0.0,
                "logit_bias": {str(START_OF_IMAGE): 100.0},
                "image_config": {"num_images": 2, "steps": 2, "seed": 7},
            },
            timeout_s=600,
        )
        usage = next(
            event["usage"] for event in interleaved if event.get("usage")
        )
        assert usage["image_count"] == 2, usage

        # A 32x32 input image resizes to the 512x512 latent canvas: its
        # latent block is captured, so the edit is served.
        edited = client.post(
            "/v1/chat/completions",
            json={
                "model": SERVED_MODEL,
                "messages": _image_chat(
                    tiny_input_png_b64(), "Make this image brighter."
                ),
                "modalities": ["image"],
                "image_config": {"steps": 2, "seed": 7},
            },
        )
        assert edited.status_code == 200, edited.text

        # A 640x512 input image keeps its size on the latent canvas, which
        # is not captured.
        refused = client.post(
            "/v1/chat/completions",
            json={
                "model": SERVED_MODEL,
                "messages": _image_chat(_png(640, 512), "Describe it."),
                "modalities": ["text"],
                "max_completion_tokens": 2,
            },
        )
        assert refused.status_code == 400, refused.text
        message = refused.json()["error"]["message"]
        assert "640x512" in message and "512x512" in message, message
