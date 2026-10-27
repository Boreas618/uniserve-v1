"""Qwen3.5 serves its language model and resumes requests from checkpoints.

Qwen3.8-27B (``UNISERVE_QWEN3_5_MODEL``), a ``qwen3_5`` checkpoint of 48
gated delta attention layers and 16 full-attention layers, serves on one GPU.
Each request keeps its recurrent state in its request slot and its state's
checkpoints at page boundaries in the KV cache's state group: the last
boundary of each prompt chunk, and every ``kv_cache_config.state_interval``
decoded tokens.

- Greedy decoding follows Transformers' ``Qwen3_5ForConditionalGeneration``
  (the installed reference implementation) in bfloat16, compared as vLLM
  compares two bfloat16 implementations (``check_logprobs_close`` of
  ``refs/vllm/tests/models/utils.py``): the generated tokens agree until
  the first position where they differ, where each implementation's token
  is among the other's five most likely, and the sequences may diverge
  after it.
- A prompt reissued, or continued by another, reuses the earlier prompt's
  pages up to the checkpoint at its last chunk's page boundary; reissued,
  it generates the earlier request's tokens with the same probabilities.
- A world change moves a streaming request to a successor world, which
  resumes it from the checkpoint its decoding kept: it computes only the
  tokens after the checkpoint, and the client reads the same greedy tokens
  as for the request alone.

Each test runs a Storage Server of its own, which it stops at the end.
"""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import signal
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
    worker_executable,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3_5")]

# Options of every deployment: a KV cache that holds the tests' requests
# and their checkpoints, prompt chunks of 512 tokens, each of which
# checkpoints its last page boundary, no reasoning parser, so the content is
# the generated text alone, and FlashInfer's attention, which serves the
# checkpoint's six query heads per KV head of width 256 (the automatic
# provider's prefix-block kernel reads none of their non-causal rows).
OPTIONS = (
    "dtype=bfloat16",
    "kv_cache_config.max_tokens=65536",
    "scheduler_config.chunked_prefill_size=512",
    "chat_config.reasoning_parser=none",
    "attention_backend=flashinfer",
)
# Prompts of the comparison with the reference: short ones, and one of
# several prompt pages and chunk boundaries.
PROMPTS = (
    "Name the capital of France in one word.",
    "Write one sentence about the sea.",
    "List three prime numbers.",
    "Here are some records. "
    + " ".join(
        f"Record {index}: value {index * 7 % 31}." for index in range(90)
    )
    + " Which record has the largest value?",
)
GENERATED = 32
# The prompt of the reuse test, of more than one thousand tokens.
LONG_PROMPT = "Summarize the following log in one sentence. " + " ".join(
    f"Event {index}: sensor {index % 13} read {index * 37 % 101} units."
    for index in range(110)
)
# The streaming request of the world change, and the change: decode batch
# sizes only a new world applies.
COUNT_PROMPT = (
    "Count from 1 to 300, separated by commas and spaces. "
    "Write only the numbers."
)
COUNT_TOKENS = 1200
EARLY_TOKENS = 320
CHANGE = "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8, 16]"


def _checkpoint() -> str:
    checkpoint = os.environ.get("UNISERVE_QWEN3_5_MODEL", "")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail("UNISERVE_QWEN3_5_MODEL must name the model checkpoint")
    return checkpoint


def _device() -> str:
    """The GPU the deployment serves on (``UNISERVE_WORLD_DEVICE``, 0 by
    default).
    """  # noqa: D205
    return os.environ.get("UNISERVE_WORLD_DEVICE", "0")


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


@contextmanager
def _serve(tmp_path: Path, name: str) -> Iterator[tuple[str, Path]]:
    """A deployment serving `name` on one GPU, its URL and its log."""
    runtime = Path(tempfile.mkdtemp(prefix="uniserve-", dir="/tmp"))
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(require_uniserve_binary()),
        "serve",
        _checkpoint(),
        "--served-model-name",
        name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-executable",
        str(worker_executable()),
    ]
    for option in OPTIONS:
        command += ["--set", option]
    log = tmp_path / "server.log"
    environment = {
        "CUDA_VISIBLE_DEVICES": _device(),
        "XDG_RUNTIME_DIR": str(runtime),
    }
    try:
        with server_process(
            command, base_url, log, timeout_s=1800, env=environment
        ):
            yield base_url, log
    finally:
        _stop_storage_server(runtime / "uniserve" / "storage.sock")
        storage_log = runtime / "uniserve" / "storage-server.log"
        if storage_log.is_file():
            shutil.copy(storage_log, tmp_path / "storage-server.log")
        shutil.rmtree(runtime, ignore_errors=True)


def _log_lines(log: Path) -> list[str]:
    colors = re.compile(r"\x1b\[[0-9;]*m")
    return [colors.sub("", line) for line in log.read_text().splitlines()]


def _wait_until(predicate, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    raise TimeoutError(f"{what} within {timeout_s} s")


def _body(name: str, prompt: str, tokens: int, stream: bool = False) -> dict:
    body = {
        "model": name,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_completion_tokens": tokens,
        "logprobs": True,
        "top_logprobs": 5,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def _token_id(token: str) -> int:
    """The id of a logprob entry's `token_id:<id>` token."""
    prefix, _, value = token.partition(":")
    assert prefix == "token_id", token
    return int(value)


def _generate(base_url: str, name: str, prompt: str, tokens: int) -> dict:
    """A greedy request's prompt ids, generated ids and each generated
    position's five most likely ids.
    """  # noqa: D205
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json=_body(name, prompt, tokens),
        timeout=1800,
    )
    response.raise_for_status()
    answer = response.json()
    choice = answer["choices"][0]
    entries = choice["logprobs"]["content"]
    return {
        "prompt": answer["prompt_token_ids"],
        "tokens": [_token_id(entry["token"]) for entry in entries],
        "top": [
            [_token_id(top["token"]) for top in entry["top_logprobs"]]
            for entry in entries
        ],
        "usage": answer["usage"],
    }


def _prefix_hits(base_url: str) -> float:
    """The prefix cache hit tokens every world counted so far
    (``uniserve:prefix_cache_hits``).
    """  # noqa: D205
    text = httpx.get(f"{base_url}/metrics", timeout=10).text
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith("uniserve:prefix_cache_hits")
    )


def _reference(prompts: list[list[int]], tokens: int) -> list[dict]:
    """Transformers' greedy decoding of each prompt in bfloat16 on the GPU:
    the generated ids and each position's five most likely ids.
    """  # noqa: D205
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    # Loaded on the host and moved: a device map needs ``accelerate``,
    # which the test environment does not carry.
    model = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            _checkpoint(), dtype=torch.bfloat16
        )
        .to("cuda:0")
        .eval()
    )
    results = []
    with torch.inference_mode():
        for prompt in prompts:
            output = model.generate(
                torch.tensor([prompt], device="cuda:0"),
                attention_mask=torch.ones(1, len(prompt), device="cuda:0"),
                max_new_tokens=tokens,
                do_sample=False,
                output_logits=True,
                return_dict_in_generate=True,
                eos_token_id=None,
            )
            generated = output.sequences[0, len(prompt) :].tolist()
            top = [
                logits[0].float().log_softmax(-1).topk(5).indices.tolist()
                for logits in output.logits
            ]
            results.append({"tokens": generated, "top": top})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def _close(served: dict, reference: dict, prompt: str) -> int:
    """Check the served greedy tokens against the reference's as vLLM's
    ``check_logprobs_close`` does; return the positions that agree.
    """  # noqa: D205
    for index, (left, right) in enumerate(
        zip(served["tokens"], reference["tokens"], strict=False)
    ):
        if left != right:
            message = (
                f"{prompt[:40]!r} diverges at {index}: served {left} "
                f"(top {served['top'][index]}), reference {right} "
                f"(top {reference['top'][index]})"
            )
            assert left in reference["top"][index], message
            assert right in served["top"][index], message
            return index
    return min(len(served["tokens"]), len(reference["tokens"]))


def test_greedy_decoding_follows_transformers(tmp_path: Path):
    name = "qwen3_5-reference"
    with _serve(tmp_path, name) as (base_url, _):
        served = [
            _generate(base_url, name, prompt, GENERATED) for prompt in PROMPTS
        ]
    reference = _reference([answer["prompt"] for answer in served], GENERATED)
    # A request may end at its end-of-sequence token, which the reference
    # decodes past.
    for answer, expected, prompt in zip(
        served, reference, PROMPTS, strict=True
    ):
        agreed = _close(answer, expected, prompt)
        print(
            f"{len(answer['prompt'])} prompt tokens: {agreed} of "
            f"{len(answer['tokens'])} generated tokens agree"
        )


def test_a_prompt_resumes_from_the_checkpoint_of_an_earlier_one(
    tmp_path: Path,
):
    """A reissued prompt resumes from the checkpoint of the first's last
    chunk: it computes what that chunk computed, from the state the
    checkpoint holds instead of the first request's slot, and generates the
    same tokens with the same probabilities.
    """  # noqa: D205
    name = "qwen3_5-checkpoints"
    with _serve(tmp_path, name) as (base_url, _):
        first = _generate(base_url, name, LONG_PROMPT, 24)
        hits = _prefix_hits(base_url)
        again = _generate(base_url, name, LONG_PROMPT, 24)
        reused = _prefix_hits(base_url) - hits
        # Each chunk of 512 tokens checkpointed its last page boundary; the
        # last chunk's is the deepest boundary that leaves a token to
        # compute, where the last chunk started.
        prompt = len(first["prompt"])
        boundary = (prompt - 1) // 64 * 64
        assert boundary % 512 == 0, prompt
        assert reused == boundary, (reused, prompt)
        assert again["tokens"] == first["tokens"]
        assert again["top"] == first["top"]

        # A prompt that continues the first reuses its checkpoint too.
        hits = _prefix_hits(base_url)
        longer = _generate(
            base_url, name, LONG_PROMPT + " Name the busiest sensor.", 8
        )
        assert _prefix_hits(base_url) - hits >= 64
        assert len(longer["tokens"]) == 8


class _Streaming:
    """A greedy request streaming on a thread of its own: the tokens it
    read and its usage.
    """  # noqa: D205

    def __init__(self, base_url: str, name: str):
        self.tokens: list[int] = []
        self.usage: dict | None = None
        self.error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._read, args=(base_url, name), daemon=True
        )
        self._thread.start()

    def _read(self, base_url: str, name: str) -> None:
        try:
            with httpx.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=_body(name, COUNT_PROMPT, COUNT_TOKENS, stream=True),
                timeout=1800,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line == "data: [DONE]":
                        break
                    if not line.startswith("data: "):
                        continue
                    chunk = json.loads(line[6:])
                    self.usage = chunk.get("usage") or self.usage
                    for choice in chunk.get("choices", ()):
                        content = (choice.get("logprobs") or {}).get(
                            "content"
                        ) or ()
                        self.tokens.extend(
                            _token_id(entry["token"]) for entry in content
                        )
        except BaseException as error:  # noqa: BLE001 - the test reports it.
            self.error = error

    def wait_for(self, count: int) -> None:
        _wait_until(
            lambda: len(self.tokens) >= count or not self._thread.is_alive(),
            1200,
            "the request did not stream",
        )
        assert len(self.tokens) >= count, "the stream ended early"

    def join(self) -> None:
        self._thread.join(timeout=1800)
        assert not self._thread.is_alive(), "the request did not end"
        assert self.error is None, f"the stream failed: {self.error!r}"


def test_a_world_change_resumes_a_request_from_its_checkpoint(
    tmp_path: Path,
):
    name = "qwen3_5-world-change"
    with _serve(tmp_path, name) as (base_url, log):
        streaming = _Streaming(base_url, name)
        streaming.wait_for(EARLY_TOKENS)
        hits = _prefix_hits(base_url)
        response = httpx.put(
            f"{base_url}/admin/config",
            json={"set": [CHANGE]},
            timeout=1800,
        )
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert result["change"] == "world", result
        moved = len(streaming.tokens)
        streaming.join()
        # The successor resumed the request from the checkpoint its decoding
        # kept: every multiple of the state interval (256 tokens) before the
        # tokens it had reached.
        resumed = _prefix_hits(base_url) - hits
        assert resumed >= 256, resumed
        committed = httpx.post(
            f"{base_url}/admin/config/commit", json={}, timeout=1800
        )
        assert committed.status_code == 200, committed.text
        alone = _generate(base_url, name, COUNT_PROMPT, COUNT_TOKENS)

    lines = _log_lines(log)
    assert any("a request migrated" in line for line in lines), (
        "the streaming request did not migrate to the successor"
    )
    assert len(streaming.tokens) > moved, (
        "the stream ended before the successor served"
    )
    assert streaming.tokens == alone["tokens"]
    assert streaming.usage == alone["usage"]
