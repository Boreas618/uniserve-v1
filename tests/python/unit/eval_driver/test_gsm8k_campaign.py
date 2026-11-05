"""The GSM8K campaign's seam: a measured run scores its rows.

The neighbouring unit tests cover the dataset, the answer extraction, the
task's scoring function and the comparability premises. This one covers the
join the campaign depends on: ``run_point`` drives the GSM8K task against a
real HTTP endpoint to a completed run, and the two artifacts the protocol
reads carry the verdict — ``run.json``'s ``status``/``valid`` and
``summary.json``'s ``validation`` and ``metrics``.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from uniserve_eval.pipeline.run import run_point
from uniserve_eval.types import (
    BenchmarkPoint,
    LoadConfig,
    MetricDefinition,
    SamplingConfig,
    TaskName,
)

pytestmark = pytest.mark.unit

#: The completion every stub response carries. Two rows hold ``#### 18`` as
#: their gold and one holds ``#### 7``, so a constant answer scores 2 of 3.
COMPLETION = "#### 18"


class _CompletionHandler(BaseHTTPRequestHandler):
    """Answers every chat completion with one streamed GSM8K answer.

    The task asks for a stream with usage, so the reply is the event stream a
    server sends: a content delta, a stop chunk, a usage chunk and the
    terminator. The score is decided by each row's gold, not by this reply, so
    the handler reads no request body.
    """

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler method name.
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        events = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": COMPLETION},
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            },
        ]
        payload = (
            b"".join(
                f"data: {json.dumps(event)}\n\n".encode() for event in events
            )
            + b"data: [DONE]\n\n"
        )
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def _point(dataset: Path) -> BenchmarkPoint:
    """Return one GSM8K cell over the given rows and sampling."""
    return BenchmarkPoint(
        name="gsm8k",
        server="server",
        task=TaskName.GSM8K,
        model="Qwen3-30B-A3B",
        dataset="jsonl",
        dataset_path=str(dataset),
        metrics=(MetricDefinition(("accuracy",), "higher"),),
        load=LoadConfig(num_prompts=3, max_concurrency=1, warmup_requests=0),
        sampling=SamplingConfig(
            temperature=0.0,
            top_p=1.0,
            max_tokens=16,
            ignore_eos=False,
            stream=True,
        ),
        provenance={
            "checkpoint": "Qwen3-30B-A3B",
            "precision": "bfloat16",
            "kv_cache_dtype": "bfloat16",
        },
    )


@pytest.fixture()
def rows(tmp_path: Path) -> Path:
    """Write three GSM8K-shaped rows whose golds score 2 of 3."""
    dataset = tmp_path / "rows.jsonl"
    content = [
        {
            "id": "row-0",
            "prompt": "Ann has 9 apples and buys 9.",
            "metadata": {"answer": "#### 18"},
        },
        {
            "id": "row-1",
            "prompt": "A box holds 3 by 6 tiles.",
            "metadata": {"answer": "#### 18"},
        },
        {
            "id": "row-2",
            "prompt": "Sam had 9 and lost 2.",
            "metadata": {"answer": "#### 7"},
        },
    ]
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in content), encoding="utf-8"
    )
    return dataset


def test_a_measured_run_records_the_score_it_computed(
    tmp_path: Path, rows: Path
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CompletionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "result"
    try:
        asyncio.run(
            run_point(
                [f"http://127.0.0.1:{server.server_port}"],
                _point(rows),
                output,
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    # run.json carries the outcome the protocol's first premise reads.
    state = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["valid"] is True
    assert state["selected_rows"]["count"] == 3

    # Every row was measured, in file order.
    measured = [
        json.loads(line)
        for line in (output / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["request_id"] for record in measured] == [
        "row-0",
        "row-1",
        "row-2",
    ]
    assert all(record["success"] for record in measured)

    # summary.json carries the validation and the counts the comparison reads.
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["task"] == "gsm8k"
    assert summary["validation"]["valid"] is True
    assert summary["validation"]["checks"]["gold_answers"] is True
    assert summary["validation"]["statistics"]["correct"] == 2
    assert summary["validation"]["statistics"]["total"] == 3
    assert summary["metrics"]["correct"] == 2
    assert summary["metrics"]["total"] == 3
    assert summary["metrics"]["accuracy"] == pytest.approx(2 / 3)
