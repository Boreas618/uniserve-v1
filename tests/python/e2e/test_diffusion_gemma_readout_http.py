"""System One readouts served from the published DiffusionGemma checkpoints.

Each precision's checkpoint is served by ``uniserve serve`` on one GPU and
answers ``POST /v1/systemone`` requests end to end: the server plans the
readout prompts, the engine prefills them into the paged prefix cache (image
features included) and denoises every canvas once, and the answers come
back in the official shape. The tests check what a client observes:

- every golden readout plan of ``diffusion_gemma_readout_prompts.json``
  without remote images answers every question and charges exactly the
  plan's prompt tokens;
- a fact stated at the start of the prompt is read back at prompt lengths
  around the 1023-token sliding window and beyond it, where only the full
  attention layers still see it;
- with one compact canvas per question, a state asked more questions than
  the worker's widest readout graph holds answers every one of them;
- attached images, one or eight, are read in ``x_images`` order;
- a stream of distinct images longer than the encoder cache retains is read
  in full, each image answered after the cache fills.

The checkpoint directories come from ``UNISERVE_DIFFUSION_GEMMA_MODEL``
(BF16) and ``UNISERVE_DIFFUSION_GEMMA_NVFP4_MODEL`` (NVFP4). The worker
executable is ``UNISERVE_WORKER_EXECUTABLE``, else the test environment's
``uniserve``. ``UNISERVE_DIFFUSION_GEMMA_SERVE_ARGS`` adds
``uniserve serve`` arguments that select the deployment topology, such as
``--data-parallel-size 4``; every topology serves the same answers.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import shlex
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from PIL import Image

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    scaled,
    server_process,
    worker_executable,
)
from uniserve.models import registry as models

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("diffusion_gemma"),
]

SERVED_MODEL = "diffusion-gemma"
CHECKPOINTS = {
    "bf16": "UNISERVE_DIFFUSION_GEMMA_MODEL",
    "nvfp4": "UNISERVE_DIFFUSION_GEMMA_NVFP4_MODEL",
}
# Each deployment a test may be served by: its checkpoint precision and the
# serve options it adds.
DEPLOYMENTS = {
    "bf16": ("bf16", []),
    "nvfp4": ("nvfp4", []),
    "bf16-compact": (
        "bf16",
        [
            "--set",
            "readout_config.layout=independent",
            "--set",
            "readout_config.canvas=compact",
        ],
    ),
}
FIXTURES = Path(__file__).parents[1] / "fixtures"
FIXTURE = FIXTURES / "diffusion_gemma_readout_prompts.json"
# The worker's default encoder cache budget, in retained feature products.
ENCODER_CACHE_ENTRIES = 256
COLORS = {
    "red": (220, 30, 30),
    "green": (30, 170, 60),
    "blue": (30, 60, 220),
    "yellow": (235, 210, 40),
}


def _checkpoint(precision: str) -> Path:
    variable = CHECKPOINTS[precision]
    value = os.environ.get(variable)
    if not value:
        pytest.fail(f"{variable} is required for the DiffusionGemma GPU gate")
    path = Path(value)
    if not path.is_dir():
        pytest.fail(f"configured checkpoint directory is missing: {path}")
    return path


def _serve(
    precision: str, tmp_path_factory, options: list[str]
) -> Iterator[tuple[str, Path]]:
    """Serve a checkpoint with extra `options`; yield its URL and directory."""
    checkpoint = _checkpoint(precision)
    try:
        binary = require_uniserve_binary()
    except FileNotFoundError as error:
        pytest.fail(str(error))
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    args = [
        str(binary),
        "serve",
        str(checkpoint),
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-executable",
        str(worker_executable()),
        "--max-model-len",
        "8192",
        *options,
        *shlex.split(os.environ.get("UNISERVE_DIFFUSION_GEMMA_SERVE_ARGS", "")),
    ]
    log = tmp_path_factory.mktemp(precision) / "server.log"
    with server_process(args, base_url, log, timeout_s=1800.0):
        yield base_url, checkpoint


@pytest.fixture(scope="module", params=sorted(CHECKPOINTS))
def served(request, tmp_path_factory) -> Iterator[tuple[str, Path]]:
    """Serve one deployment; yield its URL and checkpoint directory.

    By default the deployments are each precision's checkpoint as it ships.
    Every deployment is a parameter of this one module-scoped fixture, so
    pytest stops one deployment's server before it starts the next and a
    single server holds the device at a time. A test served by another
    deployment names it with indirect parametrization.
    """
    precision, options = DEPLOYMENTS[request.param]
    yield from _serve(precision, tmp_path_factory, options)


def _readout(base_url: str, body: dict) -> dict:
    response = httpx.post(
        f"{base_url}/v1/systemone", json=body, timeout=scaled(600.0)
    )
    assert response.status_code == 200, response.text
    return response.json()


def _distribution(answer: dict) -> dict[str, float]:
    """The normalized key distribution of one answer."""
    if answer["type"] == "noul":
        return {"false": 1.0 - answer["noul"], "true": answer["noul"]}
    return answer["probabilities"]


def _assert_well_formed(answer: dict) -> None:
    distribution = _distribution(answer)
    assert all(0.0 <= value <= 1.0 for value in distribution.values())
    assert math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-6)
    assert 0.0 < answer["x_candidate_mass"] <= 1.0 + 1e-6


def _choice(answer: dict) -> str:
    distribution = _distribution(answer)
    return max(distribution, key=distribution.get)


def test_golden_plans_answer_every_question(served):
    base_url, _ = served
    fixture = json.loads(FIXTURE.read_text())
    cases = [
        case
        for case in fixture["cases"]
        if (case["layout"], case["canvas"]) == ("joint", "full")
        and "x_images" not in case["request"]
    ]
    assert cases
    for case in cases:
        body = {**case["request"], "model": SERVED_MODEL}

        response = _readout(base_url, body)

        assert response["model"] == SERVED_MODEL
        assert response["usage"] == {
            "input_tokens": case["input_tokens"],
            "output_tokens": 0,
        }, case["name"]
        assert list(response["answers"]) == list(case["request"]["questions"])
        for answer in response["answers"].values():
            _assert_well_formed(answer)


@pytest.mark.parametrize("length", [1022, 1023, 1024, 1025, 2000])
def test_a_fact_before_the_sliding_window_is_read_back(served, length):
    """The canvas reads a fact from the prompt's start at every length.

    One filler token per `` x`` pads the state until the prompt holds
    ``length`` tokens, which the response's usage confirms.
    """
    base_url, _ = served
    question = {
        "color": {
            "type": "choice",
            "instructions": "What is the secret color named at the start "
            "of the state?",
            "criteria": dict.fromkeys(COLORS),
        }
    }

    def body(filler: int) -> dict:
        return {
            "model": SERVED_MODEL,
            "state": "The secret color is green." + " x" * filler,
            "questions": question,
        }

    base = _readout(base_url, body(0))["usage"]["input_tokens"]
    response = _readout(base_url, body(length - base))

    assert response["usage"]["input_tokens"] == length
    answer = response["answers"]["color"]
    _assert_well_formed(answer)
    assert _choice(answer) == "green"


def _png(color: tuple[int, int, int], size: tuple[int, int]) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@pytest.mark.parametrize("count", [1, 8])
def test_attached_images_are_read_in_order(served, count):
    """The last attached image's color is read, and usage counts its tokens.

    Every image before the last has another color, so an answer read from
    the wrong placement names a different color.
    """
    base_url, checkpoint = served
    names = list(COLORS)
    colors = [names[index % 3] for index in range(count - 1)] + ["yellow"]
    sizes = [(96 + 48 * index, 64 + 16 * index) for index in range(count)]
    vit = models.read_config(checkpoint).image_processor.vit
    soft = sum(vit.tokens(height, width, 1) for width, height in sizes)
    body = {
        "model": SERVED_MODEL,
        "state": "The attached images are plain color swatches.",
        "questions": {
            "last": {
                "type": "choice",
                "instructions": f"Which color fills Image {count}?",
                "criteria": dict.fromkeys(COLORS),
            }
        },
        "x_images": [
            _png(COLORS[color], size)
            for color, size in zip(colors, sizes, strict=True)
        ],
    }
    without_images = {
        key: value for key, value in body.items() if key != "x_images"
    }

    response = _readout(base_url, body)
    text_only = _readout(base_url, without_images)

    answer = response["answers"]["last"]
    _assert_well_formed(answer)
    assert _choice(answer) == "yellow"
    # Each image adds its soft tokens between its two marker tokens, and
    # the prompt names the images in one line of text.
    added = (
        response["usage"]["input_tokens"] - text_only["usage"]["input_tokens"]
    )
    assert added >= soft + 2 * count


def test_distinct_images_beyond_the_encoder_cache_are_read(served):
    """Every image of a stream longer than the encoder cache is read.

    The served worker keeps the default encoder cache budget. Each request
    attaches one image no earlier request used, so every image is encoded
    and, past the budget, encoded while the cache holds its full set of
    retained features.
    """
    base_url, _ = served
    entries = ENCODER_CACHE_ENTRIES
    names = list(COLORS)
    with httpx.Client(timeout=scaled(600.0)) as client:
        for index in range(entries + 8):
            # The size identifies the image; the color is the expected answer.
            color = names[index % len(names)]
            size = (64 + index % 64, 64 + index // 64)
            body = {
                "model": SERVED_MODEL,
                "state": "The attached image is a plain color swatch.",
                "questions": {
                    "color": {
                        "type": "choice",
                        "instructions": "Which color fills Image 1?",
                        "criteria": dict.fromkeys(COLORS),
                    }
                },
                "x_images": [_png(COLORS[color], size)],
            }
            response = client.post(f"{base_url}/v1/systemone", json=body)
            assert response.status_code == 200, (index, response.text)
            answer = response.json()["answers"]["color"]
            _assert_well_formed(answer)
            assert _choice(answer) == color, (index, answer)


@pytest.mark.parametrize("served", ["bf16-compact"], indirect=True)
def test_more_questions_than_a_readout_graph_holds_are_answered(served):
    """Every one of 48 questions about one state is answered.

    Each question has a compact canvas of its own, which costs few tokens, so
    the engine reads all 48 canvases in one pass: more than the widest
    readout graph holds, which the worker sizes for full canvases (8192
    batched tokens over 256-token canvases make 32). The pass runs without
    a graph and answers each question from the state.
    """
    base_url, _ = served
    questions = {
        f"color_{index}": {
            "type": "choice",
            "instructions": "What is the secret color named in the state?",
            "criteria": dict.fromkeys(COLORS),
        }
        for index in range(48)
    }

    response = _readout(
        base_url,
        {
            "model": SERVED_MODEL,
            "state": "The secret color is green.",
            "questions": questions,
        },
    )

    assert list(response["answers"]) == list(questions)
    for answer in response["answers"].values():
        _assert_well_formed(answer)
        assert _choice(answer) == "green"
