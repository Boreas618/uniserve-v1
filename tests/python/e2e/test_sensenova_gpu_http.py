from __future__ import annotations

import copy
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from tokenizers import Tokenizer

from tests.python.e2e.http_helpers import (
    find_free_port,
    granted_devices,
    png_size_from_b64,
    post_sse,
    require_uniserve_binary,
    server_process,
    tiny_input_png_b64,
    worker_executable,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.sensenova,
    pytest.mark.model("sensenova"),
]

MODEL_ENV = "UNISERVE_SENSENOVA_MODEL"
# Comma-separated CUDA devices the server ranks use, one rank per device;
# the first two the GPU locks granted by default.
DEVICES_ENV = "UNISERVE_SENSENOVA_DEVICES"
# Two comma-separated CUDA devices: the rank's device and the generation
# tower's; the first two the GPU locks granted by default.
TOWER_DEVICES_ENV = "UNISERVE_SENSENOVA_TOWER_DEVICES"
SERVED_MODEL = "SenseNova-U1"
IMAGE_SIZE = (2048, 1152)


def active_model() -> Path:
    model_value = os.environ.get(MODEL_ENV)
    if not model_value:
        pytest.fail(f"{MODEL_ENV} is required for the SenseNova GPU gate")
    model = Path(model_value)
    if not model.is_dir():
        pytest.fail(f"configured checkpoint directory is missing: {model}")
    return model


def image_start_token_id(model: Path) -> int:
    tokenizer_path = model / "tokenizer.json"
    if not tokenizer_path.is_file():
        pytest.fail(
            f"configured checkpoint has no tokenizer.json: {tokenizer_path}"
        )
    token_id = Tokenizer.from_file(str(tokenizer_path)).token_to_id("<img>")
    if token_id is None:
        pytest.fail("configured SenseNova tokenizer has no <img> control")
    return token_id


@contextmanager
def production_server(
    tmp_path: Path,
    model: Path,
    graph_policy: str,
    *,
    devices: str | None = None,
    ranks: int | None = None,
    extra: tuple[str, ...] = (),
) -> Iterator[str]:
    """Serve ``model`` on ``devices`` (one rank per device by default)."""
    try:
        binary = require_uniserve_binary()
    except FileNotFoundError as error:
        pytest.fail(str(error))
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    devices = devices or os.environ.get(DEVICES_ENV) or granted_devices(2)
    args = [
        str(binary),
        "serve",
        str(model),
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-executable",
        str(worker_executable()),
        "--set",
        "kv_cache_config.max_tokens=65536",
        "--max-running-requests",
        "1",
        "--set",
        "max_num_batched_tokens=4096",
        "--set",
        f"cuda_graph_config.policy={graph_policy}",
        "--tensor-parallel-size",
        str(ranks or len(devices.split(","))),
        "--set",
        "logging_config.log_stats=false",
        *extra,
    ]
    environment = {"CUDA_VISIBLE_DEVICES": devices}
    with server_process(
        args,
        base_url,
        tmp_path / f"sensenova-production-{port}.log",
        timeout_s=300,
        env=environment,
    ):
        yield base_url


def post_chat(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={"model": SERVED_MODEL, "temperature": 0.0, **payload},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def assert_bounded_text_completion(payload: dict[str, Any]) -> None:
    assert payload["model"] == SERVED_MODEL
    assert payload["choices"][0]["finish_reason"] == "length"
    assert payload["usage"]["prompt_tokens"] > 0
    assert payload["usage"]["completion_tokens"] == 2
    assert payload["usage"]["image_count"] == 0


def assert_generated_image(image: dict[str, Any]) -> None:
    assert (image["width"], image["height"]) == IMAGE_SIZE
    assert png_size_from_b64(image["b64_json"]) == IMAGE_SIZE
    assert image["bytes"] > 0


def visible_stream_delta(event: dict[str, Any]) -> dict[str, Any] | None:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return None
    if any(
        delta.get(field) for field in ("content", "reasoning_content", "images")
    ):
        return delta
    return None


@pytest.mark.timeout(600)
@pytest.mark.parametrize("graph_policy", ["full", "off"])
def test_sensenova_public_production_lineage(tmp_path: Path, graph_policy: str):
    model = active_model()
    input_image_url = f"data:image/png;base64,{tiny_input_png_b64()}"
    with production_server(tmp_path, model, graph_policy) as base_url:
        delivered = []
        snapshots = []
        for _round in range(2):
            text = post_chat(
                base_url,
                {
                    "messages": [
                        {"role": "user", "content": "Reply with one word."}
                    ],
                    "modalities": ["text"],
                    "max_completion_tokens": 2,
                },
            )
            assert_bounded_text_completion(text)

            image_to_text = post_chat(
                base_url,
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe the dominant color.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": input_image_url},
                                },
                            ],
                        }
                    ],
                    "modalities": ["text"],
                    "max_completion_tokens": 2,
                },
            )
            assert_bounded_text_completion(image_to_text)

            image_response = httpx.post(
                f"{base_url}/v1/images/generations",
                json={
                    "model": SERVED_MODEL,
                    "prompt": "A blue square.",
                    "steps": 1,
                    "seed": 7,
                },
                timeout=600,
            )
            image_response.raise_for_status()
            generated_images = image_response.json()["data"]
            assert len(generated_images) == 1
            assert_generated_image(generated_images[0])

            interleaved = post_sse(
                base_url,
                "/v1/chat/completions",
                {
                    "model": SERVED_MODEL,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Use this color in two generated "
                                        "images."
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": input_image_url},
                                },
                            ],
                        }
                    ],
                    "modalities": ["text", "image"],
                    "max_completion_tokens": 4,
                    "temperature": 0.0,
                    "logit_bias": {str(image_start_token_id(model)): 100.0},
                    "image_config": {"num_images": 2, "steps": 2, "seed": 7},
                },
                timeout_s=600,
            )
            assert interleaved[-1]["type"] == "sse_done"
            finish_reasons = [
                choice["finish_reason"]
                for event in interleaved
                for choice in event.get("choices", [])
                if choice.get("finish_reason") is not None
            ]
            assert finish_reasons == ["length"]
            usage = next(
                event["usage"] for event in interleaved if event.get("usage")
            )
            assert usage["completion_tokens"] == 4
            assert usage["image_count"] == 2
            assert usage["image_steps"] == 4
            assert usage["image_steps_per_image"] == [2, 2]
            visible_events = [
                (event, delta)
                for event in interleaved
                if (delta := visible_stream_delta(event)) is not None
            ]
            assert visible_events
            images = [
                image
                for _, delta in visible_events
                for image in delta.get("images", [])
            ]
            assert len(images) == 2
            for image in images:
                image_url = image["image_url"]["url"]
                assert image_url.startswith("data:image/png;base64,")
                assert (
                    png_size_from_b64(image_url.split(",", 1)[1]) == IMAGE_SIZE
                )
            delivered.append(
                (text, image_to_text, generated_images, interleaved)
            )
            snapshots.append(copy.deepcopy(delivered[-1]))
            assert delivered == snapshots


@pytest.mark.timeout(900)
def test_sensenova_generation_tower_serves_one_device_outputs(tmp_path: Path):
    """A generation tower on a second device leaves served outputs unchanged.

    The tower holds the flow route, the denoiser's own modules and the image
    decoder; the rank's device keeps the text model and the prefix cache. Image
    understanding, seeded generation (replaying captured denoising steps)
    and prompt scoring must equal those of the same rank on one device.
    """
    model = active_model()
    devices = os.environ.get(TOWER_DEVICES_ENV) or granted_devices(2)
    if len(devices.split(",")) != 2:
        pytest.fail(f"{TOWER_DEVICES_ENV} must name two CUDA devices")
    input_image_url = f"data:image/png;base64,{tiny_input_png_b64()}"

    def serve(placement: str, **options) -> tuple:
        with production_server(
            tmp_path / placement, model, "full", **options
        ) as base_url:
            understanding = post_chat(
                base_url,
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe it."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": input_image_url},
                                },
                            ],
                        }
                    ],
                    "modalities": ["text"],
                    "max_completion_tokens": 4,
                    "prompt_logprobs": 1,
                },
            )
            generation = httpx.post(
                f"{base_url}/v1/images/generations",
                json={
                    "model": SERVED_MODEL,
                    "prompt": "A blue square.",
                    "steps": 2,
                    "seed": 7,
                },
                timeout=600,
            )
            generation.raise_for_status()
            (image,) = generation.json()["data"]
            assert_generated_image(image)
            return (
                understanding["choices"][0]["message"],
                understanding["prompt_logprobs"],
                image["b64_json"],
            )

    for path in (tmp_path / "one", tmp_path / "tower"):
        path.mkdir()
    one = serve("one", devices=devices.split(",")[0])
    tower = serve(
        "tower",
        devices=devices,
        ranks=1,
        extra=("--set", "worker_config.mesh=tower=gen:cuda:1"),
    )
    assert tower == one


@pytest.mark.timeout(900)
def test_image_guidance_with_a_negative_prompt_generates_its_image(
    tmp_path: Path,
):
    """Image guidance beside a negative prompt generates the image.

    SenseNova's image-unconditional branch conditions on the start of a
    conversation, while its text-unconditional branch conditions on the
    rendered negative prompt. A request with a negative prompt and image
    guidance therefore evaluates a guidance prefix other than its negative
    prompt's: with image guidance alone, and with text and image guidance
    together. Both generate the requested image.
    """
    model = active_model()
    with production_server(tmp_path, model, "full") as base_url:
        for guidance_scale, image_guidance_scale in [(1.0, 2.0), (4.0, 2.0)]:
            response = httpx.post(
                f"{base_url}/v1/images/generations",
                json={
                    "model": SERVED_MODEL,
                    "prompt": "A blue square.",
                    "negative_prompt": "blurry, low quality",
                    "guidance_scale": guidance_scale,
                    "image_guidance_scale": image_guidance_scale,
                    "steps": 2,
                    "seed": 7,
                },
                timeout=600,
            )
            assert response.status_code == 200, (
                guidance_scale,
                image_guidance_scale,
                response.text,
            )
            images = response.json()["data"]
            assert len(images) == 1
            assert_generated_image(images[0])
