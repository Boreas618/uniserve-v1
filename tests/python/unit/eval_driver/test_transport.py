from __future__ import annotations

import gc
import json
from collections.abc import Awaitable, Callable

import aiohttp
import pytest
from aiohttp import web
from pytest_aiohttp import AiohttpRawServer

from uniserve_eval.transport.client import send_request
from uniserve_eval.types import (
    CHAT_COMPLETIONS,
    IMAGES_GENERATIONS,
    RequestRecord,
    TaskRequest,
)

pytestmark = pytest.mark.unit

# Answers every request the loopback server receives, whatever its path.
Handler = Callable[[web.BaseRequest], Awaitable[web.StreamResponse]]


async def _send(
    serve: AiohttpRawServer,
    handler: Handler,
    request: TaskRequest,
    *,
    task: str,
    output_len_fallback: int = 0,
) -> RequestRecord:
    """Send one request to a loopback server that answers with `handler`.

    `serve` is the `aiohttp_raw_server` fixture, which closes the server at
    the test's teardown.
    """
    server = await serve(handler)
    async with aiohttp.ClientSession() as session:
        return await send_request(
            session,
            str(server.make_url("")),
            request,
            "req-1",
            task=task,
            output_len_fallback=output_len_fallback,
        )


def _event_stream(body: str) -> Handler:
    async def handler(request: web.BaseRequest) -> web.Response:
        return web.Response(text=body, content_type="text/event-stream")

    return handler


@pytest.mark.asyncio
async def test_image_generation_error_body_is_classified_by_status(
    aiohttp_raw_server: AiohttpRawServer,
) -> None:
    # UniServe's OpenAI-compatible error body for a failed request.
    async def handler(request: web.BaseRequest) -> web.Response:
        return web.json_response(
            {
                "error": {
                    "message": "engine failed",
                    "type": "server_error",
                    "param": None,
                    "code": None,
                }
            },
            status=500,
        )

    record = await _send(
        aiohttp_raw_server,
        handler,
        TaskRequest(IMAGES_GENERATIONS, {"model": "m"}, stream=False),
        task="t2i",
    )

    assert record.success is False
    assert record.status_code == 500
    assert record.classifier == "transport_status_500"
    assert record.images == 0


@pytest.mark.asyncio
async def test_stream_request_rejects_a_json_response(
    aiohttp_raw_server: AiohttpRawServer,
) -> None:
    async def handler(request: web.BaseRequest) -> web.Response:
        return web.json_response({"choices": [{"message": {"content": "ok"}}]})

    record = await _send(
        aiohttp_raw_server,
        handler,
        TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=True),
        task="text",
    )

    assert record.success is False
    assert record.classifier == "response_expected_sse"


@pytest.mark.asyncio
async def test_chat_json_records_visible_text_and_server_usage(
    aiohttp_raw_server: AiohttpRawServer,
) -> None:
    async def handler(request: web.BaseRequest) -> web.Response:
        return web.json_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "caption"},
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }
        )

    record = await _send(
        aiohttp_raw_server,
        handler,
        TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=False),
        task="i2t",
    )

    assert record.success is True
    assert record.generated_text == "caption"
    assert record.record_dict()["generated_text"] == "caption"
    assert record.prompt_len_source == "server_usage"
    assert record.output_len_source == "server_usage"
    assert record.token_timing_available is False


@pytest.mark.asyncio
async def test_stream_records_reasoning_content_as_public_text(
    aiohttp_raw_server: AiohttpRawServer,
) -> None:
    body = "".join(
        (
            'data: {"choices":[{"delta":{"reasoning_content":"reason"},'
            '"finish_reason":null}]}\n\n',
            'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n',
            'data: {"choices":[],"usage":{"prompt_tokens":4,'
            '"completion_tokens":1}}\n\n',
            "data: [DONE]\n\n",
        )
    )

    record = await _send(
        aiohttp_raw_server,
        _event_stream(body),
        TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=True),
        task="text",
        output_len_fallback=1,
    )

    assert record.success is True
    assert record.classifier == "ok"
    assert record.generated_text == "reason"
    assert record.record_dict()["generated_text"] == "reason"
    assert record.token_timing_available is True
    assert record.prompt_len_source == "server_usage"
    assert record.output_len_source == "server_usage"


@pytest.mark.asyncio
async def test_stream_event_longer_than_any_line_buffer_is_read_whole(
    aiohttp_raw_server: AiohttpRawServer,
) -> None:
    # One committed 256-token canvas, or an encoded image, arrives as a single
    # data line; CRLF framing is accepted as well.
    block = "x" * 300_000
    body = (
        "data: "
        + json.dumps({"choices": [{"delta": {"content": block}}]})
        + "\r\n\r\n"
        + 'data: {"choices":[{"delta":{},"finish_reason":"length"}],'
        '"usage":{"prompt_tokens":4,"completion_tokens":256}}\r\n\r\n'
        + "data: [DONE]\r\n\r\n"
    )

    record = await _send(
        aiohttp_raw_server,
        _event_stream(body),
        TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=True),
        task="text",
        output_len_fallback=256,
    )

    assert record.success is True
    assert record.generated_text == block
    assert record.output_len == 256
    assert record.finish_reason == "length"


@pytest.mark.asyncio
async def test_chat_json_records_reasoning_content_as_public_text(
    aiohttp_raw_server: AiohttpRawServer,
) -> None:
    async def handler(request: web.BaseRequest) -> web.Response:
        return web.json_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "reasoning_content": "reason",
                            "content": None,
                        },
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            }
        )

    record = await _send(
        aiohttp_raw_server,
        handler,
        TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=False),
        task="i2t",
    )

    assert record.success is True
    assert record.generated_text == "reason"
    assert record.token_timing_available is False


@pytest.mark.asyncio
async def test_stream_rejects_non_object_events(
    aiohttp_raw_server: AiohttpRawServer,
) -> None:
    record = await _send(
        aiohttp_raw_server,
        _event_stream('data: ["invalid"]\n\ndata: [DONE]\n\n'),
        TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=True),
        task="text",
    )

    assert record.success is False
    assert record.classifier == "response_invalid_event"
    assert record.token_timing_available is False


@pytest.mark.asyncio
async def test_stream_records_the_error_event_it_carries(
    aiohttp_raw_server: AiohttpRawServer,
) -> None:
    # A request the engine refused after the stream began: UniServe writes the
    # error event of the OpenAI error body, then the sentinel.
    error = {
        "message": "request exceeds total KV capacity",
        "type": "invalid_request_error",
        "code": "invalid_request_error",
    }
    body = f"data: {json.dumps({'error': error})}\n\ndata: [DONE]\n\n"
    record = await _send(
        aiohttp_raw_server,
        _event_stream(body),
        TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=True),
        task="text",
    )

    assert record.success is False
    assert record.classifier == "model_error"
    assert record.record_dict()["error"] == (
        "invalid_request_error: request exceeds total KV capacity"
    )


@pytest.mark.asyncio
async def test_a_closed_stream_record_keeps_its_tokens_out_of_the_collector(
    aiohttp_raw_server: AiohttpRawServer,
) -> None:
    # Two token events with their top candidates, as UniServe streams them
    # under `logprobs`, `top_logprobs` and `return_token_ids`. A load keeps
    # every record to its end, so a full collection that visited each
    # token's values would pause the loop that stamps every stream.
    def event(token: int) -> str:
        top = [
            {"token": f"token_id:{token + rank}", "logprob": -float(rank)}
            for rank in range(5)
        ]
        choice = {
            "delta": {"content": f"t{token}"},
            "token_ids": [token],
            "logprobs": {"content": [{**top[0], "top_logprobs": top}]},
        }
        return "data: " + json.dumps({"choices": [choice]}) + "\n\n"

    body = (
        event(10)
        + event(20)
        + 'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
        + "data: [DONE]\n\n"
    )
    record = await _send(
        aiohttp_raw_server,
        _event_stream(body),
        TaskRequest(CHAT_COMPLETIONS, {"model": "m"}, stream=True),
        task="text",
    )
    # The collector stops tracking a tuple at a pass that finds each of its
    # items untracked, so each pass may reach one more level of nesting: a
    # candidate pair, a token's candidates, the record's sequence of them.
    for _ in range(3):
        gc.collect()

    # The record as the run writes it to `requests.jsonl`.
    written = json.loads(json.dumps(record.record_dict()))
    assert record.success is True
    assert written["token_ids"] == [10, 20]
    assert written["top_logprobs"][1] == [
        [20 + rank, -float(rank)] for rank in range(5)
    ]
    for values in (
        record.itl,
        record.text_times,
        record.chunk_times,
        record.token_ids,
        record.token_times,
        record.top_logprobs,
    ):
        assert not gc.is_tracked(values)
