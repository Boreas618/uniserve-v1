"""Chat completions the text cases ask, and the rules they compare answers by.

One request shape, one streamed-token reader, one long prompt and one
closeness rule, shared by every text case instead of a copy per file.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx

#: Greedy sampling with every filter off: the same prompt gives the same
#: tokens on the same weights.
GREEDY = {"temperature": 0, "top_p": 1.0, "top_k": 0, "min_p": 0.0}

#: The chat template arguments of a Qwen3 deployment that answers without
#: its thinking section, so a short answer is the answer itself.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}

#: How long one completion may take, in seconds.
COMPLETION_S = 600.0


def body(
    model: str, prompt: str, max_tokens: int, **fields: Any
) -> dict[str, Any]:
    """A chat completion request for `prompt` with one user message, at
    most `max_tokens` completion tokens and `fields` beside them.
    """  # noqa: D205
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        **fields,
    }


def stream_options(request: dict[str, Any]) -> dict[str, Any]:
    """`request` streamed, with its usage in the last event."""
    return {
        **request,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def complete(
    base_url: str, request: dict[str, Any], timeout_s: float = COMPLETION_S
) -> dict[str, Any]:
    """The answer to the chat completion `request`.

    Raises:
        httpx.HTTPStatusError: When the server refuses it.
        AssertionError: When a successful answer's body is no JSON, naming
            its status, type and first bytes.
    """
    response = httpx.post(
        f"{base_url}/v1/chat/completions", json=request, timeout=timeout_s
    )
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as error:
        raise AssertionError(
            f"the completion's body is not JSON: {response.status_code} "
            f"{response.headers.get('content-type')!r} {response.text[:400]!r}"
        ) from error


def content(answer: dict[str, Any]) -> str:
    """The text of an answer's first choice."""
    return answer["choices"][0]["message"]["content"]


def stream_tokens(line: str) -> tuple[list[str], dict[str, Any] | None, bool]:
    """The tokens one line of a streamed answer carries (from its logprobs),
    its usage, and whether it ends the stream.
    """  # noqa: D205
    if line == "data: [DONE]":
        return [], None, True
    if not line.startswith("data: "):
        return [], None, False
    chunk = json.loads(line.removeprefix("data: "))
    tokens = [
        entry["token"]
        for choice in chunk.get("choices", ())
        for entry in (choice.get("logprobs") or {}).get("content") or ()
    ]
    return tokens, chunk.get("usage"), False


def ranked_tokens(answer: dict[str, Any]) -> list[tuple[str, set[str]]]:
    """Each sampled token of an answer's first choice with the tokens its
    ``top_logprobs`` rank beside it.
    """  # noqa: D205
    return [
        (entry["token"], {top["token"] for top in entry["top_logprobs"]})
        for entry in answer["choices"][0]["logprobs"]["content"]
    ]


def assert_logprobs_close(
    expected: Sequence[tuple[str, set[str]]],
    actual: Sequence[tuple[str, set[str]]],
    context: str,
) -> None:
    """Hold two greedy answers to vLLM's ``check_logprobs_close`` rule.

    The answers agree token for token until their first difference, where
    each one's token must be among the other's ranked tokens; past it they
    may differ, since each continues from its own token.
    """
    for position, ((token, top), (other, other_top)) in enumerate(
        zip(expected, actual, strict=False)
    ):
        if token == other:
            continue
        assert token in other_top and other in top, (
            f"{context}: the answers diverge at token {position}: {token!r} "
            f"is not among the other's ranked tokens {sorted(other_top)}, or "
            f"{other!r} not among its {sorted(top)}"
        )
        return


def long_prompt(records: int) -> str:
    """A prompt of `records` numbered records (about 19 tokens each) and a
    question about them, whose prefix blocks a cache can hold and reuse.
    """  # noqa: D205
    listing = " ".join(
        f"Record {index}: the value of item {index * 7919 % 1000} is "
        f"{index * 104729 % 997}."
        for index in range(records)
    )
    return f"{listing} Summarize the first three records in one sentence."
