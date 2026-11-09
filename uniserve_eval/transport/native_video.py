"""Native video APIs, ending only when all encoded media bytes arrive.

A request's payload is UniServe's model-agnostic ``/v1/videos`` body: the
prompt, the stated ``size``, ``num_frames`` and ``fps``, the seed and any
``input_references``. UniServe takes it as is, reading reference media from
their ``file://`` URLs. The baselines serve MiniMax-H3 in their own forms,
which this module translates the body into: SGLang's native H3 body names a
task, ``conditions`` and a ``target`` of short edge, aspect ratio and
duration; vLLM-Omni and FastVideo take the raster, frame rate and frame count
in their own fields, vLLM-Omni with the media uploaded as request parts and
FastVideo with plain local paths. No decoding, hashing or filesystem work
belongs in these logical requests: the media bytes were read when the request
was built. The caller owns the overall deadline and retains bodies for later
inspection.

A server's own timings of a request (its inference time, named stage
durations and peak device storage) are recorded as it reports them: in the
``X-Inference-Time-S``, ``X-Stage-Durations`` and ``X-Peak-Storage-MB``
headers of a synchronous response, or the ``inference_time_s``,
``stage_durations`` and ``peak_storage_mb`` fields of a completed job. The
shape a server reports generating is recorded the same way: the
``x-video-size``, ``x-video-num-frames`` and ``x-video-fps`` headers, or the
job's ``size``, ``num_frames`` and ``fps``.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin

import aiohttp

from ..types import RequestRecord, TaskRequest, VideoShape, parse_size

# Response headers in which a synchronous video route reports its timings.
_TIMING_HEADERS = (
    "x-inference-time-s",
    "x-stage-durations",
    "x-peak-memory-mb",
)

# Response headers in which a synchronous video route reports the shape it
# generated.
_SHAPE_HEADERS = ("x-video-size", "x-video-num-frames", "x-video-fps")

# vLLM-Omni and FastVideo present a request's references grouped by media
# type in this order, whatever order they arrive in.
_TYPE_ORDER = {"image": 0, "video": 1, "audio": 2}


async def receive_video(
    client: aiohttp.ClientSession,
    base_url: str,
    request: TaskRequest,
    record: RequestRecord,
) -> None:
    """Submit the native payload, poll as needed, and collect original bytes."""
    video = request.video
    if video is None:
        raise ValueError("a video request carries its point's video settings")
    kwargs = native_request(request)
    url = base_url.rstrip("/") + request.endpoint

    if request.endpoint.endswith("/sync"):
        await _body(client, "POST", url, record, **kwargs)
        return

    async with client.post(url, **kwargs) as response:
        record.note_http(response.status)
        if await _failed_status(response, record):
            return
        job = await response.json()
    job_id = quote(str(job["id"]), safe="")
    poll_url = f"{url}/{job_id}"
    while job.get("status") != "completed":
        if job.get("status") in {"failed", "cancelled"}:
            record.close_now()
            record.mark_failure("video_job_failed", str(job.get("error", job)))
            return
        await asyncio.sleep(video.poll_interval_s)
        async with client.get(poll_url) as response:
            if await _failed_status(response, record):
                return
            job = await response.json()
    _record_timings(
        record,
        job.get("inference_time_s"),
        job.get("stage_durations"),
        job.get("peak_storage_mb"),
    )
    record.served_shape = _served_shape(
        job.get("size"), job.get("num_frames"), job.get("fps")
    )

    # SGLang may publish a storage URL instead of a local content route.
    media_url = urljoin(base_url + "/", job.get("url") or f"{poll_url}/content")
    await _body(client, "GET", media_url, record)


def native_request(request: TaskRequest) -> dict[str, Any]:
    """Return the JSON or multipart body arguments for the request's backend.

    A baseline's request always states its schedule in sigma points
    including the clean endpoint (``VideoConfig`` requires the fields each
    baseline takes), and ``request.video_shape`` carries the stated shape.

    Raises:
        ValueError: The request carries no video settings, or the backend
            cannot be sent the request's references as the same work: in
            their order, or with a field it lacks.
    """
    video = request.video
    if video is None:
        raise ValueError("a video request carries its point's video settings")
    backend = video.backend
    payload = dict(request.payload)
    if backend == "uniserve":
        return {"json": payload}

    shape = request.video_shape
    if shape is None:
        raise ValueError(f"a {backend} video request needs its video shape")
    references = list(payload.get("input_references", []))
    task = h3_task(references)
    if backend == "sglang":
        # Configured serving options, such as x264_preset, join the native
        # body; the configuration cannot restate the work fields.
        return {
            "json": {
                **_sglang_body(request, payload, references, task),
                **video.extra_params,
            }
        }

    _check_type_order(backend, references)
    common = {
        "model": payload["model"],
        "prompt": payload["prompt"],
        "seed": payload["seed"],
    }
    if backend == "vllm-omni":
        return _vllm_omni_request(request, references, task, common)
    if backend == "fastvideo":
        return _fastvideo_request(request, references, task, common)
    raise ValueError(f"unknown video backend {backend!r}")


def h3_task(references: list[dict[str, Any]]) -> str:
    """Return the MiniMax-H3 task the baselines' native forms name.

    UniServe derives it from the references; the H3 baselines take it as a
    field. No reference is ``t2va``; only image anchor keyframes (items with
    ``frame_index``) are ``fl2va``; any reference without ``frame_index`` is
    ``ref2va``.

    Raises:
        ValueError: A video or audio reference carries ``frame_index``, which
            no task takes.
    """
    if not references:
        return "t2va"
    if any("frame_index" not in reference for reference in references):
        return "ref2va"
    if all(reference["type"] == "image" for reference in references):
        return "fl2va"
    raise ValueError("only image references may anchor a keyframe")


def _sglang_body(
    request: TaskRequest,
    payload: dict[str, Any],
    references: list[dict[str, Any]],
    task: str,
) -> dict[str, Any]:
    """Return SGLang's native MiniMax-H3 body for the request.

    Native H3 admission names its canvas by the short edge and an aspect
    ratio and its frames by a duration, which the point states as the ones
    resolving to its stated shape, and counts its schedule in sigma points.
    Each reference becomes a condition whose role is ``keyframe`` for an
    anchor and ``reference`` otherwise.
    """
    video = request.video
    shape = request.video_shape
    assert video is not None and shape is not None
    conditions = []
    for reference in references:
        condition: dict[str, Any] = {
            "type": reference["type"],
            "uri": reference["url"],
            "role": "keyframe" if "frame_index" in reference else "reference",
        }
        for name in ("frame_index", "start_time_seconds"):
            if name in reference:
                condition[name] = reference[name]
        conditions.append(condition)
    return {
        "model": payload["model"],
        "prompt": payload["prompt"],
        "task": task,
        "conditions": conditions,
        "target": {
            "short_edge": min(shape.width, shape.height),
            "aspect_ratio": video.aspect_ratio,
            "duration_seconds": video.duration_seconds,
        },
        "seed": payload["seed"],
        **video.schedule(),
    }


def _vllm_omni_request(
    request: TaskRequest,
    references: list[dict[str, Any]],
    task: str,
    common: dict[str, Any],
) -> dict[str, Any]:
    """Return vLLM-Omni's multipart body for the request.

    Omni counts schedule intervals (denoiser forwards), not points. The
    raster, frame rate and frame count are the stated shape, beside the
    point's requested aspect ratio and duration. Reference media
    go up as repeated ``input_references`` parts, which the server classifies
    by media type; keyframes take their frame indices in ``extra_params``, and
    a video's soundtrack conditions the request whenever the file has one.
    """
    video = request.video
    shape = request.video_shape
    assert video is not None and shape is not None
    keyframes = [
        reference["frame_index"]
        for reference in references
        if "frame_index" in reference
    ]
    if keyframes and task != "fl2va":
        raise ValueError("vLLM-Omni takes keyframes only for fl2va")
    assert video.num_inference_steps is not None
    assert video.flow_shift is not None and video.audio_flow_shift is not None
    assert video.duration_seconds is not None
    extra: dict[str, Any] = {
        "task": task,
        # Omni's native duration field preserves fractional seconds without
        # the generic OpenAI form field's integer restriction.
        "duration": float(video.duration_seconds),
        "audio_flow_shift": float(video.audio_flow_shift),
    }
    if keyframes:
        extra["frame_indices"] = keyframes
    starts = [
        float(reference.get("start_time_seconds", 0.0))
        for reference in references
        if reference["type"] == "video"
    ]
    if any(starts):
        # One offset per video, in the videos' order.
        extra["start_time_seconds"] = starts

    fields = {
        **common,
        "width": shape.width,
        "height": shape.height,
        "aspect_ratio": video.aspect_ratio,
        "fps": shape.fps,
        "num_frames": shape.frames,
        "num_inference_steps": video.num_inference_steps - 1,
        "guidance_scale": 1.0,
        "flow_shift": float(video.flow_shift),
        # Configured serving options, such as preencode_mp4; the
        # configuration cannot restate the work fields.
        "extra_params": json.dumps({**extra, **video.extra_params}),
    }
    parts = aiohttp.FormData(default_to_multipart=True)
    for key, value in fields.items():
        parts.add_field(key, str(value))
    for media in request.reference_media:
        parts.add_field(
            "input_references",
            media.data,
            filename=Path(media.path).name,
            content_type=media.mime,
        )
    return {"data": parts}


def _fastvideo_request(
    request: TaskRequest,
    references: list[dict[str, Any]],
    task: str,
    common: dict[str, Any],
) -> dict[str, Any]:
    """Return FastVideo's JSON body for the request.

    FastVideo takes the shifts from the checkpoint and counts the schedule
    in sigma points, or in fused-block forwards for a parallel decoding
    student. It validates an already aligned causal-VAE frame count, so the
    stated frames carry the work, and an integral requested duration is kept
    beside them. References name local files by path in its
    typed reference fields; it has no keyframe or start-offset fields.
    """
    video = request.video
    shape = request.video_shape
    assert video is not None and shape is not None
    assert video.num_inference_steps is not None
    points = video.num_inference_steps
    body: dict[str, Any] = {
        **common,
        "size": shape.size,
        "fps": shape.fps,
        "num_inference_steps": (
            points - 1 if video.parallel_decoding else points
        ),
        "num_frames": shape.frames,
    }
    assert video.duration_seconds is not None
    if float(video.duration_seconds).is_integer():
        # The generic seconds field accepts integers only; the stated frame
        # count alone carries a fractional duration.
        body["seconds"] = float(video.duration_seconds)
    if task == "t2va":
        return {"json": body}

    typed: dict[str, list[dict[str, str]]] = {}
    fields = {
        "image": ("image_reference", "image_url"),
        "video": ("video_reference", "video_url"),
        "audio": ("audio_reference", "audio_url"),
    }
    for reference, media in zip(
        references, request.reference_media, strict=True
    ):
        if "frame_index" in reference:
            raise ValueError("FastVideo takes references without keyframes")
        if reference.get("start_time_seconds"):
            raise ValueError("FastVideo has no reference start offset")
        field, key = fields[reference["type"]]
        typed.setdefault(field, []).append({key: media.path})
    return {"json": {**body, "task": task, **typed}}


def _check_type_order(backend: str, references: list[dict[str, Any]]) -> None:
    """Require references in the order a type-grouping backend presents.

    Raises:
        ValueError: A later reference has an earlier media type, so the
            backend would present the references in another order.
    """
    order = [_TYPE_ORDER[reference["type"]] for reference in references]
    if order != sorted(order):
        raise ValueError(
            f"{backend} presents images, then videos, then audio; the "
            "request's references must already be in that order"
        )


def _served_shape(size: Any, frames: Any, fps: Any) -> VideoShape | None:
    """Return the shape a server reported, or ``None`` when it is not one.

    A server reports these as strings in headers or as JSON values in a job;
    a malformed or absent value leaves the shape unset, which validation of a
    server that must report it then fails.
    """
    try:
        width, height = parse_size(str(size))
        count, rate = int(str(frames)), int(str(fps))
    except (TypeError, ValueError):
        return None
    if count < 1 or rate < 1:
        return None
    return VideoShape(width, height, count, rate)


def _record_timings(
    record: RequestRecord,
    inference_s: Any,
    stages: Any,
    peak_storage_mib: Any,
) -> None:
    """Record the server-reported timings that parse as finite numbers.

    A server reports these as it defines them; a malformed or absent value
    is left unset rather than failing a request that returned its media.
    """
    record.server_inference_s = _finite(inference_s)
    record.server_peak_storage_mib = _finite(peak_storage_mib)
    if isinstance(stages, str):
        try:
            stages = json.loads(stages)
        except json.JSONDecodeError:
            stages = None
    if isinstance(stages, Mapping):
        record.server_stage_s = {
            str(name): value
            for name, value in (
                (name, _finite(seconds)) for name, seconds in stages.items()
            )
            if value is not None
        }


def _finite(value: Any) -> float | None:
    """Return a value as a finite float, or ``None`` when it is not one."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


async def _failed_status(
    response: aiohttp.ClientResponse, record: RequestRecord
) -> bool:
    if response.status < 400:
        return False
    record.status_code = response.status
    if record.final_event_time is None:
        record.close_now()
    record.mark_failure(
        f"transport_status_{response.status}",
        (await response.text(errors="replace"))[:500],
    )
    return True


async def _body(
    client: aiohttp.ClientSession,
    method: str,
    url: str,
    record: RequestRecord,
    **kwargs,
) -> None:
    async with client.request(
        method, url, allow_redirects=True, **kwargs
    ) as response:
        record.note_http(response.status)
        record.video_mime = response.headers.get("content-type", "")
        # Encoded bodies are buffered in host storage; no media work delays the
        # next slot. Keep even invalid bodies for post-run failure artifacts.
        record.video_body = await response.read()
        record.close_now()
        if not await _failed_status(response, record):
            record.mark_success()
            # A synchronous route reports its timings in headers, any of
            # which a server may omit; a job's content download carries
            # none, and its job fields stand.
            if any(name in response.headers for name in _TIMING_HEADERS):
                _record_timings(
                    record,
                    response.headers.get("x-inference-time-s"),
                    response.headers.get("x-stage-durations"),
                    response.headers.get("x-peak-memory-mb"),
                )
            # Likewise its generated shape; a job's stands.
            if any(name in response.headers for name in _SHAPE_HEADERS):
                record.served_shape = _served_shape(
                    *(response.headers.get(name) for name in _SHAPE_HEADERS)
                )
