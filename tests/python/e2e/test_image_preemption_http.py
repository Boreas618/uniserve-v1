"""Interleaved image requests preempted mid-request continue exactly.

Serves BAGEL (``UNISERVE_BAGEL_MODEL``) and SenseNova-U1
(``UNISERVE_SENSENOVA_MODEL``), each with its end-to-end setup, on the one
GPU the GPU locks granted (``UNISERVE_WORLD_DEVICE``). One interleaved
streaming chat request, greedy and seeded, whose image start is biased so
that it generates its two images back to back and then text, is served
uninterrupted as the reference. The same request is then preempted:

- by a sleep while it generates its second image: the engine ends it as
  migratable, the daemon keeps its tokens (the image starts among them) and
  the decoded values of its first image, and once the engine wakes the
  request continues: the engine rebuilds its context by feeding the carried
  image back through the model's feedback encoder, and continues the second
  image from its stored denoising step;
- by the loss of a worker while it generates its second image: no latent
  trajectory can be stored from the lost worker, so the resumed request
  computes that image from its seeded noise;
- by a world change whose numerics are unchanged (a change of the workers'
  watchdog), whose previous world is kept as its weights alone
  (``update_config.previous_world``), so that the successor's KV cache
  grows into the previous world's at its promotion and holds the whole
  cache the request reserves again. A request generates its images in
  seconds while a successor prepares for tens of them, so the request
  starts once the successor prepares, and the serving world's workers are
  paused (``SIGSTOP``) once the client receives its first image and resumed
  once the change promotes: the park then lands where
  the paused calls end, and the successor continues the request.

Each run has its own cache salt. The test waits until the client receives
its first image before initiating preemption. It checks that the stream
continues with two images, the same usage and finish reason, and preserves
the already delivered image:

- for SenseNova, its whole stream is the reference's, byte for byte: the
  request decodes no text between its images and its input image keeps its
  prompt out of the prefix cache, so its continuation runs the calls the
  uninterrupted request ran, with the same shapes (its context prefill, each
  image's cache close, its first image's feedback over the decoded values
  it carried, which are the values the uninterrupted request fed back, and
  its second image from the same noise). A BAGEL continuation that resumes
  in its second image writes its first image's end token with a prefill
  where the request decoded it, another kernel, so its later outputs follow
  that kernel's rounding: the test reports whether they equal the
  reference's.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any

import httpx
import pytest
from tokenizers import Tokenizer

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    scaled,
    server_process,
    tiny_input_png_b64,
    worker_executable,
)
from uniserve_eval.transport.sse import iter_sse_events

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]

# Denoising steps leave enough time after receiving the first image to
# preempt the request while it generates its second image.
STEPS = 24


@dataclass(frozen=True)
class Setup:
    """How a model's end-to-end tests serve it and the request they send."""

    name: str
    model_env: str
    # The token that opens a generated image.
    image_start: str
    arguments: tuple[str, ...]
    # Whether the request carries an input image.
    input_image: bool
    # Whether a continuation runs the uninterrupted request's calls with the
    # same shapes (the module documentation), so that its whole stream is
    # the reference's.
    repeats_calls: bool


SETUPS = {
    # `tests/python/e2e/test_bagel_graphs_http.py`.
    "bagel": Setup(
        name="bagel",
        model_env="UNISERVE_BAGEL_MODEL",
        image_start="<|vision_start|>",
        arguments=("--set", "cuda_graph_config.policy=full"),
        input_image=False,
        repeats_calls=False,
    ),
    # `tests/python/e2e/test_sensenova_gpu_http.py`, on one rank, with the
    # KV cache the device holds: a successor world's own segments must hold
    # the whole cache an image request reserves again, which they do not
    # beside a 65536-token cache, whose world change is refused.
    "sensenova": Setup(
        name="SenseNova-U1",
        model_env="UNISERVE_SENSENOVA_MODEL",
        image_start="<img>",
        arguments=(
            "--max-running-requests",
            "1",
            "--set",
            "max_num_batched_tokens=4096",
            "--set",
            "cuda_graph_config.policy=full",
            "--tensor-parallel-size",
            "1",
        ),
        input_image=True,
        repeats_calls=True,
    ),
}


def _checkpoint(setup: Setup) -> Path:
    model = os.environ.get(setup.model_env)
    if not model or not Path(model).is_dir():
        pytest.fail(f"{setup.model_env} must name the model's checkpoint")
    return Path(model)


def _device() -> str:
    """The GPU the deployment serves on, as its lock granted it."""
    device = os.environ.get("UNISERVE_WORLD_DEVICE", "")
    if not device.isdigit():
        pytest.fail(
            "UNISERVE_WORLD_DEVICE must name the GPU whose lock this run holds"
        )
    return device


def _image_start(model: Path, setup: Setup) -> int:
    token = Tokenizer.from_file(str(model / "tokenizer.json")).token_to_id(
        setup.image_start
    )
    if token is None:
        pytest.fail(f"the tokenizer has no {setup.image_start} token")
    return token


def _stop_storage_server(socket: Path) -> None:
    """Stop the Storage Server serving `socket`, found by its command line."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"serve" in command and str(socket).encode() in command:
            os.kill(int(entry.name), signal.SIGTERM)


def _workers(runtime: Path) -> list[int]:
    """The worker processes the deployment serving from `runtime` runs now,
    found by their command line and environment.
    """  # noqa: D205
    workers = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
            environment = (entry / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        if (
            b"worker" in command
            and b"--worker-id" in command
            and f"XDG_RUNTIME_DIR={runtime}".encode() in environment
        ):
            workers.append(int(entry.name))
    return workers


@contextmanager
def _serve(
    tmp_path: Path, setup: Setup, model: Path
) -> Iterator[tuple[str, Path, Path]]:
    """The deployment, with a Storage Server of its own; its URL, log and
    runtime directory.
    """  # noqa: D205
    runtime = Path(tempfile.mkdtemp(prefix="uniserve-", dir="/tmp"))
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(require_uniserve_binary()),
        "serve",
        str(model),
        "--served-model-name",
        setup.name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-executable",
        str(worker_executable()),
        *setup.arguments,
    ]
    log = tmp_path / "server.log"
    environment = {
        "CUDA_VISIBLE_DEVICES": _device(),
        "XDG_RUNTIME_DIR": str(runtime),
    }
    try:
        with server_process(
            command, base_url, log, timeout_s=1800, env=environment
        ):
            yield base_url, log, runtime
    finally:
        _stop_storage_server(runtime / "uniserve" / "storage.sock")
        shutil.rmtree(runtime, ignore_errors=True)


def _request(setup: Setup, image_start: int, salt: str) -> dict[str, Any]:
    """The interleaved request, under cache salt `salt`: its two images back
    to back, then text.
    """  # noqa: D205
    text = {"type": "text", "text": "Draw a red circle, then a blue square."}
    content: Any = text["text"]
    if setup.input_image:
        image = f"data:image/png;base64,{tiny_input_png_b64()}"
        content = [text, {"type": "image_url", "image_url": {"url": image}}]
    return {
        "model": setup.name,
        "cache_salt": salt,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": content}],
        "modalities": ["text", "image"],
        "max_completion_tokens": 4,
        "temperature": 0.0,
        "logit_bias": {str(image_start): 100.0},
        "image_config": {"num_images": 2, "steps": STEPS, "seed": 7},
    }


@dataclass
class Streamed:
    """What a client read of a streamed chat request."""

    text: str
    images: list[str]
    finish_reasons: list[str]
    usage: dict[str, Any]
    seconds: float


def _stream(
    base_url: str, payload: dict[str, Any], received: Queue[str] | None = None
) -> Streamed:
    started = time.monotonic()
    events = []
    with httpx.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=scaled(1800),
    ) as response:
        response.raise_for_status()
        for event in iter_sse_events(response.iter_lines()):
            events.append(event)
            if received is not None:
                for choice in event.get("choices") or []:
                    for image in (choice.get("delta") or {}).get(
                        "images"
                    ) or []:
                        received.put(image["image_url"]["url"])
    seconds = time.monotonic() - started
    assert events[-1]["type"] == "sse_done", events[-1]
    text, images, reasons, usage = [], [], [], None
    for event in events:
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            text.append(delta.get("reasoning_content") or "")
            text.append(delta.get("content") or "")
            images += [
                image["image_url"]["url"] for image in delta.get("images") or []
            ]
            if choice.get("finish_reason") is not None:
                reasons.append(choice["finish_reason"])
    assert usage is not None, events
    return Streamed("".join(text), images, reasons, usage, seconds)


def _admin(
    base_url: str, path: str, body: dict | None = None
) -> httpx.Response:
    return httpx.post(
        f"{base_url}/admin/{path}", json=body or {}, timeout=scaled(1800)
    )


# A world change's phases in their order (``GET /admin/config``).
_PHASES = ("shrink", "prepare", "promote")


def _wait_promotion(base_url: str, change: Future) -> None:
    """Waits until the world change `change` promotes its successor, or has
    answered: a park whose requests have no call in flight completes at
    once, and the promotion with it.
    """  # noqa: D205
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline and not change.done():
        config = httpx.get(
            f"{base_url}/admin/config", timeout=scaled(60)
        ).json()
        if (config.get("update") or {}).get("phase") == "promote":
            return
        time.sleep(0.05)
    if not change.done():
        raise TimeoutError("the world change did not promote")


def _wait_phase(base_url: str, change: Future, phase: str) -> None:
    """Waits until the world change `change` is in `phase`, as
    ``GET /admin/config`` reports it; fails when the change answers or
    passes the phase before.
    """  # noqa: D205
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if change.done():
            response = change.result()
            pytest.fail(
                f"the change ended before its {phase} phase: "
                f"{response.status_code} {response.text}"
            )
        config = httpx.get(
            f"{base_url}/admin/config", timeout=scaled(60)
        ).json()
        current = (config.get("update") or {}).get("phase")
        if current == phase:
            return
        if current in _PHASES and _PHASES.index(current) > _PHASES.index(phase):
            pytest.fail(f"the change passed its {phase} phase: {current}")
        time.sleep(0.05)
    raise TimeoutError(f"the world change did not reach its {phase} phase")


def _assert_continued(
    preempted: Streamed, reference: Streamed, carried: int, setup: Setup
) -> None:
    """The preempted request, which carried `carried` images, streamed each
    image and token once, its images finished before the preemption are the
    reference's, and, when its continuation repeats the uninterrupted
    request's calls, its whole stream is the reference's.
    """  # noqa: D205
    assert len(preempted.images) == 2, preempted
    assert preempted.usage["image_steps_per_image"] == [STEPS, STEPS]
    assert preempted.usage == reference.usage
    assert preempted.finish_reasons == reference.finish_reasons
    assert preempted.images[:carried] == reference.images[:carried]
    same = (
        preempted.images == reference.images
        and preempted.text == reference.text
    )
    images = [
        image == wanted
        for image, wanted in zip(
            preempted.images, reference.images, strict=True
        )
    ]
    text = "equal" if preempted.text == reference.text else "different"
    print(
        f"{setup.name}: carried {carried} image(s); the whole stream "
        f"{'equals' if same else 'differs from'} the reference's (text "
        f"{text}, each image equal: {images})"
    )
    if setup.repeats_calls:
        assert preempted.images == reference.images
        assert preempted.text == reference.text


@pytest.mark.timeout(scaled(3600))
@pytest.mark.parametrize(
    "model",
    ["bagel", pytest.param("sensenova", marks=pytest.mark.missing_state)],
)
def test_a_preempted_image_request_continues_exactly(
    tmp_path: Path, model: str
) -> None:
    setup = SETUPS[model]
    checkpoint = _checkpoint(setup)
    image_start = _image_start(checkpoint, setup)
    with _serve(tmp_path, setup, checkpoint) as (base_url, log, runtime):
        reference = _stream(base_url, _request(setup, image_start, "reference"))
        assert len(reference.images) == 2, reference
        assert reference.usage["image_count"] == 2, reference.usage
        assert reference.usage["image_steps_per_image"] == [STEPS, STEPS]

        # A sleep while the request generates its second image: the request
        # leaves with its first image and continues once the engine wakes.
        streamed: Future
        received: Queue[str] = Queue()
        with ThreadPoolExecutor(max_workers=1) as pool:
            streamed = pool.submit(
                _stream,
                base_url,
                _request(setup, image_start, "sleep"),
                received,
            )
            assert received.get(timeout=scaled(600)) == reference.images[0]
            assert not streamed.done(), "the request ended before the sleep"
            slept = _admin(base_url, "sleep")
            assert slept.status_code == 200, slept.text
            woke = _admin(base_url, "wake_up")
            assert woke.status_code == 200, woke.text
            slept_stream = streamed.result()
        _assert_continued(slept_stream, reference, 1, setup)

        # Losing the worker leaves no stored trajectory for this request.
        # Recovery must compute its unfinished image from the request facts.
        received = Queue()
        with ThreadPoolExecutor(max_workers=1) as pool:
            streamed = pool.submit(
                _stream,
                base_url,
                _request(setup, image_start, "worker-loss"),
                received,
            )
            assert received.get(timeout=scaled(600)) == reference.images[0]
            assert not streamed.done(), "the request ended before worker loss"
            workers = _workers(runtime)
            assert len(workers) == 1, workers
            worker = workers[0]
            subprocess.run(
                ["ps", "-o", "pid,ppid,args", "--forest", "-p", str(worker)],
                check=True,
            )
            assert worker in _workers(runtime), "the worker already exited"
            os.kill(worker, signal.SIGKILL)
            recovered_stream = streamed.result()
        _assert_continued(recovered_stream, reference, 1, setup)

        # A world change whose numerics are unchanged, the serving world's
        # workers paused once the request ran for most of the reference's
        # time and resumed once the change promotes, well within the
        # workers' five-minute watchdog.
        received = Queue()
        workers = _workers(runtime)
        assert workers, "the serving world's workers"
        with ThreadPoolExecutor(max_workers=2) as pool:
            change = pool.submit(
                httpx.put,
                f"{base_url}/admin/config",
                json={
                    "set": [
                        "worker_config.watchdog_timeout=6m",
                        "update_config.previous_world=weights",
                    ]
                },
                timeout=scaled(3600),
            )
            _wait_phase(base_url, change, "prepare")
            streamed = pool.submit(
                _stream,
                base_url,
                _request(setup, image_start, "world"),
                received,
            )
            assert received.get(timeout=scaled(600)) == reference.images[0]
            assert not streamed.done(), "the request ended before promotion"
            for worker in workers:
                os.kill(worker, signal.SIGSTOP)
            try:
                _wait_promotion(base_url, change)
            finally:
                # The previous world's processes exit once it is kept as
                # its weights.
                for worker in workers:
                    try:
                        os.kill(worker, signal.SIGCONT)
                    except ProcessLookupError:
                        pass
            changed = change.result()
            changed_stream = streamed.result()
        assert changed.status_code == 200, changed.text
        result = changed.json()["result"]
        assert result["change"] == "world", result
        assert result["world"]["numerics"] == "unchanged", result
        _assert_continued(changed_stream, reference, 1, setup)
        committed = _admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text
