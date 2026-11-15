"""Issues benchmark HTTP requests and records normalized observable outputs.

``send_request`` is the single entry point the benchmark runner
(``uniserve_eval.pipeline.run``) uses per request. It routes a video request
to the native video transport and other requests by endpoint to a transport
for image generations, streamed chat, or non-streaming chat, and folds
everything observable into one ``RequestRecord``: HTTP status, success and a
stable failure classifier, client timing, token usage with its provenance,
generated text, and media. Video bodies remain encoded until the runner
inspects them after measurement.

All timestamps are ``time.perf_counter`` values on the same clock as
``RequestRecord.start_time``. Streamed events carry the client receipt time
that ``SseParser`` stamps into their ``_client_t`` field. Requests run on the
caller's ``aiohttp.ClientSession``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiohttp

from ..types import (
    IMAGES_GENERATIONS,
    RequestRecord,
    TaskRequest,
)
from .http import post_json, response_lines
from .images import ImageOutputError, decode_openai_image_parts
from .native_video import receive_video
from .openai import OpenAIChat, StreamOutcome
from .sse import aiter_sse_events


async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    request: TaskRequest,
    request_id: str,
    *,
    task: str,
    prompt_len: int = 0,
    output_len_fallback: int = 0,
    scheduled_time: float | None = None,
    timeout_s: float = 6 * 60 * 60,
) -> RequestRecord:
    """Dispatch one task request through its endpoint-specific transport.

    A video request, which carries its point's video settings, takes the
    native video transport; the endpoint selects among the others. Both are
    decided before ``request.stream`` is consulted, so video and
    image-generation requests never take the SSE path.
    ``output_len_fallback`` becomes the record's ``requested_output_len`` for
    every endpoint. On chat endpoints,
    ``prompt_len`` and ``output_len_fallback`` stand in for token counts the
    server does not report.

    Returns:
        The closed record. HTTP status and response-content failures carry
        their own classifiers. Expiration of the overall logical deadline
        becomes ``request_deadline``; other raised ``Exception`` instances,
        including aiohttp connection errors, become ``transport_failure``.
        Exceptions outside ``Exception``, such as ``asyncio.CancelledError``,
        propagate.
    """
    payload = {
        key: value
        for key, value in request.payload.items()
        if value is not None
    }
    url = base_url.rstrip("/") + request.endpoint
    record = RequestRecord(request_id=request_id, task=task)
    record.begin(
        endpoint=request.endpoint,
        scheduled_time=scheduled_time,
        requested_output_len=int(output_len_fallback),
    )
    record.base_url = base_url
    try:
        if request.video is not None:
            record.prompt_len = prompt_len
            record.video_shape = request.video_shape
            # One deadline covers dispatch, polling and the complete media body.
            async with asyncio.timeout(timeout_s):
                await receive_video(session, base_url, request, record)
        elif request.endpoint == IMAGES_GENERATIONS:
            await _send_images(session, url, payload, record)
        elif request.stream:
            await _send_chat_stream(
                session, url, payload, record, prompt_len, output_len_fallback
            )
        else:
            await _send_chat(
                session, url, payload, record, prompt_len, output_len_fallback
            )
    except TimeoutError:
        record.close_at(record.start_time + timeout_s)
        record.mark_failure(
            "request_deadline", f"logical request exceeded {timeout_s}s"
        )
    except Exception as error:  # noqa: BLE001 - benchmarks emit structured failures.
        record.mark_transport_exception(error)
    return record


async def _send_images(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    record: RequestRecord,
) -> None:
    """Execute an image-generations request and validate embedded outputs.

    Latency closes when the complete body has arrived, before parsing. A
    status of 400 or above is classified ``transport_status_<code>``
    whatever the body holds, since error bodies such as UniServe's
    ``{"error": {...}}`` do not follow the images schema; so is a non-JSON
    body at any status. Otherwise structural and image-decoding classifiers
    apply, and every image of a successful response is assigned the
    whole-response latency.
    """
    response = await post_json(session, url, payload, record)
    if response.status >= 400:
        record.mark_failure(
            f"transport_status_{response.status}", response.text[:500]
        )
        return
    try:
        data = response.json()
    except Exception:
        record.mark_failure(
            f"transport_status_{response.status}", response.text[:500]
        )
        return
    body_ok, classifier = _classify_images(data)
    if not body_ok:
        record.mark_failure(classifier)
        return

    # `_classify_images` inspects only the first entry; every entry is
    # checked here and decoded by `_attach_images`, which records its own
    # failure classifier.
    images = data["data"]
    if not all(isinstance(image, dict) for image in images):
        record.mark_failure("invalid_image_part", "invalid_image_part")
        return
    if _attach_images(record, images, assign_json_latency=True) is None:
        record.mark_success()


async def _send_chat(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    record: RequestRecord,
    prompt_len: int,
    output_len_fallback: int,
) -> None:
    """Execute a non-streaming chat request and record text, images, and usage."""  # noqa: E501
    # Latency closes when the complete body has arrived, before parsing.
    response = await post_json(session, url, payload, record)
    try:
        data = response.json()
    except Exception:
        record.mark_failure(
            f"transport_status_{response.status}", response.text[:500]
        )
        return
    choices = data.get("choices") if isinstance(data, dict) else None
    content = ""
    images: list[dict[str, Any]] = []
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        record.apply_choice_metadata(choice)
        message = choice.get("message")
        content = OpenAIChat.message_text(
            message if isinstance(message, dict) else None
        )
        images = OpenAIChat.message_images(
            message if isinstance(message, dict) else None
        )
    # A single JSON body has no per-token arrival times, so TTFT and ITL are
    # unavailable and every image is assigned the whole-response latency.
    record.generated_text = content
    record.token_timing_available = False
    image_error = _attach_images(record, images, assign_json_latency=True)
    if isinstance(data, dict):
        usage = data.get("usage")
        if isinstance(usage, dict):
            record.apply_usage(usage)
        record.apply_cached_prompt_tokens(data)
    record.apply_token_fallbacks(
        prompt_len=prompt_len, output_len_fallback=output_len_fallback
    )

    # Usage is recorded above even when image decoding already marked the
    # request failed.
    if image_error is not None:
        return
    if response.status < 400 and bool(content or record.decoded_images):
        record.mark_success()
    else:
        record.mark_failure(
            f"transport_status_{response.status}"
            if response.status >= 400
            else "empty_completion"
        )


async def _send_chat_stream(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    record: RequestRecord,
    prompt_len: int,
    output_len_fallback: int,
) -> None:
    """Execute an SSE chat request and record event-level output timing.

    Latency, TTFT, ITL, and image latencies derive from per-event receipt
    stamps, taken as each event's line is read. Each event is folded into
    the record as it completes and is not kept: a load holds dozens of
    streams of hundreds of events, and holding their decoded events until
    each stream ends would make them outlive young collections, so that
    CPython's full collections, which pause the event loop that stamps
    every stream, would come more often and take longer.
    """
    async with session.post(url, json=payload) as response:
        # Reject transport or framing mismatches before interpreting event data.
        # Any status other than 200 fails, including other 2xx codes.
        record.note_http(response.status)
        if response.status != 200:
            body = await response.read()
            record.close_now()
            record.mark_failure(
                f"transport_status_{response.status}",
                body.decode("utf-8", errors="replace")[:500],
            )
            return
        # A JSON body in reply to a stream request carries no event timing and
        # gets its own classifier.
        if _is_json_content_type(response.headers.get("content-type", "")):
            await response.read()
            record.close_now()
            record.mark_failure(
                "response_expected_sse",
                "stream request received a JSON response",
            )
            return
        # Fallback token counts are written first; `apply_usage` overwrites
        # them and their provenance when the stream reports usage. Token
        # timing becomes available only once `add_text` observes a text delta.
        record.prompt_len = prompt_len
        record.output_len = output_len_fallback
        record.token_timing_available = False

        # Malformed event JSON becomes a `parse_error` event, which
        # `StreamOutcome` rejects, instead of raising into the generic
        # transport failure. The body is read until the server closes it;
        # latency closes at the last event stamp. A stream whose transport
        # fails keeps what it folded before the failure.
        outcome = StreamOutcome()
        last_event_time: float | None = None
        last_text_time: float | None = None
        image_since_last_text = False
        image_parts: list[dict[str, Any]] = []
        async for event in aiter_sse_events(
            response_lines(response.content),
            stamp_time=True,
            on_parse_error="record",
        ):
            outcome.observe(event)
            # An event that is not a JSON object fails the stream and holds
            # nothing to fold.
            if not isinstance(event, dict):
                continue

            # Fold deltas in arrival order. A text gap that spans an image
            # event is excluded from ITL, and each image part records the
            # latency of the event that delivered it. Tokens keep the arrival
            # of their chunk.
            timestamp = event.get("_client_t")
            timestamp_f = float(timestamp) if timestamp is not None else None
            if timestamp_f is not None:
                last_event_time = (
                    timestamp_f
                    if last_event_time is None
                    else max(last_event_time, timestamp_f)
                )
                if event.get("type") != "sse_done":
                    record.add_chunk(timestamp_f)
            choices = event.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if isinstance(choice, dict):
                        record.apply_choice_metadata(choice)
                        record.add_tokens(choice, timestamp_f)
            usage = event.get("usage")
            if isinstance(usage, dict):
                record.apply_usage(usage)
            record.apply_cached_prompt_tokens(event)
            content = OpenAIChat.delta_text(event)
            images = OpenAIChat.delta_images(event)
            if content:
                record.add_text(
                    content,
                    timestamp_f,
                    last_text_time=last_text_time,
                    count_itl=not image_since_last_text,
                )
                if timestamp_f is not None:
                    last_text_time = timestamp_f
                    image_since_last_text = False
            if images:
                image_parts.extend(images)
                image_since_last_text = True
                record.add_image_arrival(len(images), timestamp_f)

    record.close_at(
        last_event_time if last_event_time is not None else time.perf_counter()
    )
    ok, classifier = outcome.classify()
    if ok:
        record.mark_success()
    else:
        # An error event the stream carried is the server's own statement of
        # the failure, which the record keeps beside its classifier.
        record.mark_failure(classifier, outcome.error)

    # Images decode after timing is folded; a decode failure replaces the
    # stream's classification.
    _attach_images(record, image_parts)


def _classify_images(payload: Any) -> tuple[bool, str]:
    """Classify the structural validity of an image-generations payload.

    Only the first ``data`` entry is inspected, and only for a non-empty
    ``b64_json``; ``_send_images`` decodes every entry.

    Returns:
        ``(True, "ok")``, or ``False`` with ``empty_image_data`` or
        ``missing_image_payload``.
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        return False, "empty_image_data"
    first = data[0]
    if not isinstance(first, dict) or not first.get("b64_json"):
        return False, "missing_image_payload"
    return True, "ok"


def _attach_images(
    record: RequestRecord,
    parts: list[dict[str, Any]],
    *,
    assign_json_latency: bool = False,
) -> str | None:
    """Decode image parts into a record and return any stable error classifier."""  # noqa: E501
    try:
        decoded = decode_openai_image_parts(parts)
    except ImageOutputError as error:
        record.mark_failure(error.classifier, error.classifier)
        return error.classifier
    record.attach_images(decoded, assign_json_latency=assign_json_latency)
    return None


def _is_json_content_type(content_type: str) -> bool:
    """Report whether a response media type denotes JSON."""
    media_type = content_type.partition(";")[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")
