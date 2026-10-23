"""A served model's startup kernel table names the kernel of every call.

``uniserve serve`` logs one ``uniserve-kernel-table`` line when startup
completes. Startup runs
every call kind, image encoders and decoders included, so for an
image-understanding model the table already names the kernels that serve
image inputs: here BAGEL's SigLIP vision tower, its autoencoder's encoder
and decoder, the image-feature rows its prefill appends, and the image
latents its denoiser writes into the prompt. The server serves under CUDA
graphs (the default policy on a CUDA device), where a prefill's image rows
run in the captured bucket of prompts with vision blocks, whose rows state
their causality on the device. A call kind without a native kernel would
fail startup instead of its first request.

The checkpoint directory comes from ``UNISERVE_BAGEL_MODEL``; the worker
executable is ``UNISERVE_WORKER_EXECUTABLE``, else the test environment's
``uniserve``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
    worker_executable,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("bagel")]

TAG = "uniserve-kernel-table "


@pytest.fixture(scope="module")
def attention(tmp_path_factory):
    """The attention call sites of each runner in BAGEL's startup table.

    A runner is named by its component and module path, for example
    ``model.vision_encoder``.
    """
    checkpoint = os.environ.get("UNISERVE_BAGEL_MODEL")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail("UNISERVE_BAGEL_MODEL must name the BAGEL checkpoint")
    try:
        binary = require_uniserve_binary()
    except FileNotFoundError as error:
        pytest.fail(str(error))
    port = find_free_port()
    log = tmp_path_factory.mktemp("startup") / "server.log"
    with server_process(
        [
            str(binary),
            "serve",
            checkpoint,
            "--served-model-name",
            "bagel",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--worker-executable",
            str(worker_executable()),
        ],
        f"http://127.0.0.1:{port}",
        log,
        timeout_s=1800.0,
    ):
        pass

    tables = [
        json.loads(line.split(TAG, 1)[1])
        for line in log.read_text(errors="replace").splitlines()
        if TAG in line
    ]
    (startup,) = (table for table in tables if table["stage"] == "startup")
    return {
        runner["runner"]: [
            site for site in runner["call_sites"] if site["op"] == "attention"
        ]
        for runner in startup["runners"]
    }


def inputs(sites):
    """The attention inputs the call sites have served."""
    return {name for site in sites for name in site.get("inputs", {})}


def test_bagel_startup_table_names_every_image_kernel(attention):
    # Every SigLIP layer, the autoencoder's encoder mid-block and its
    # decoder mid-block (each runner also holds the other half of the
    # autoencoder, which it never calls) have chosen a kernel.
    vision = attention["model.vision_encoder"]
    assert vision and all(site["provider"] is not None for site in vision)
    for runner, half in (
        ("model.latent_encoder", "encoder.middle"),
        ("model.image_decoder", "decoder.decoder.middle"),
    ):
        called = [
            site
            for site in attention[runner]
            if any(layer.startswith(half) for layer in site["layers"])
        ]
        assert called and all(site["provider"] for site in called), runner


def test_bagel_startup_serves_image_rows_before_the_first_request(attention):
    # Prefill has served causal text rows and, in the captured bucket of
    # prompts with vision blocks, image rows whose causality the device
    # holds; the denoiser has written an input image's latent into the
    # prompt.
    assert {
        "paged attention: causal rows",
        "paged attention: device row causality",
    } <= inputs(attention["model.text"])
    assert "paged attention: non-causal rows" in inputs(
        attention["model.denoiser"]
    )
