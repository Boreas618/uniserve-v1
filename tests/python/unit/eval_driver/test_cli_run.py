"""`uniserve-eval run` against a stand-in server, through the command line.

The stand-in is a real process that the driver launches from the profile's
command: an OpenAI-compatible chat server that streams one token per
request with its usage.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
from pathlib import Path

import pytest

from uniserve_eval.artifacts import supersede
from uniserve_eval.cli import main
from uniserve_eval.config import load_config
from uniserve_eval.pipeline.run import run_point
from uniserve_eval.types import (
    BenchmarkPoint,
    LoadConfig,
    MetricDefinition,
    TaskName,
    VideoConfig,
)

pytestmark = pytest.mark.unit

# The stand-in server. Its argument is the port it listens on.
_SERVER = """
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"version": "stand-in"}).encode()
        status = 200 if self.path == "/version" else 404
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", "0")))
        print("completion", flush=True)
        token = {"choices": [{"index": 0, "delta": {"content": "hi"}}]}
        final = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        body = b"".join(
            b"data: " + json.dumps(event).encode() + b"\\n\\n"
            for event in (token, final)
        ) + b"data: [DONE]\\n\\n"
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _profile(tmp_path: Path, server: str = _SERVER) -> Path:
    """Write a one-point text profile served by `server`."""
    (tmp_path / "server.py").write_text(server, encoding="utf-8")
    (tmp_path / "rows.jsonl").write_text(
        json.dumps({"id": "row", "prompt": "hello", "output_len": 1}) + "\n",
        encoding="utf-8",
    )
    port = _free_port()
    profile = tmp_path / "profile.toml"
    profile.write_text(
        f"""
artifact_root = "results"

[servers.local]
port = {port}
command = [{json.dumps(sys.executable)}, "server.py", "{port}"]

[benchmarks.point]
server = "local"
task = "text"
model = "model"
dataset = "jsonl"
dataset_path = "rows.jsonl"

[benchmarks.point.load]
num_prompts = 1
warmup_requests = 0

[benchmarks.point.metrics]
output_throughput = "higher"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return profile


def _run(argv: list[str]) -> int:
    """Invoke the command-line entry point and return its exit status."""
    previous = sys.argv
    sys.argv = ["uniserve-eval", *argv]
    try:
        try:
            main()
        except SystemExit as exit_info:
            return int(exit_info.code or 0)
    finally:
        sys.argv = previous
    return 0


def test_a_rerun_keeps_the_earlier_run_outside_the_canonical_paths(
    tmp_path: Path,
) -> None:
    """The second run owns the canonical bundle; the first is archived."""
    profile = _profile(tmp_path)
    results = tmp_path / "results"

    assert _run(["--config", str(profile), "run", "point"]) == 0
    first = json.loads((results / "point" / "run.json").read_text())
    assert _run(["--config", str(profile), "run", "point"]) == 0
    second = json.loads((results / "point" / "run.json").read_text())

    # The canonical bundle and server log are the second run's alone.
    assert second["status"] == "completed"
    assert second["started_at"] > first["started_at"]
    assert (results / "server-logs" / "point.log").read_text().count(
        "completion"
    ) == 1
    # The first run's bundle and server log are kept, unchanged, under one
    # archive of the point.
    (archive,) = (results / "superseded" / "point").iterdir()
    assert json.loads((archive / "bundle" / "run.json").read_text()) == first
    assert (archive / "server-logs" / "point.log").read_text().count(
        "completion"
    ) == 1


def test_supersede_moves_bundle_media_and_logs_together(
    tmp_path: Path,
) -> None:
    """An earlier run's evidence leaves the canonical paths intact."""
    results = tmp_path / "results"
    bundle = results / "point"
    media = tmp_path / "media" / "point"
    log = results / "server-logs" / "point.log"
    for path, content in (
        (bundle / "run.json", "{}"),
        (media / "row.mp4", "video"),
        (log, "log"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    empty = tmp_path / "empty"
    empty.mkdir()

    archive = supersede(
        results,
        "point",
        {
            "bundle": bundle,
            "media": media,
            "server-logs/point.log": log,
            "unused": empty,
        },
    )

    assert archive is not None
    assert archive.parent == results / "superseded" / "point"
    assert (archive / "bundle" / "run.json").read_text() == "{}"
    assert (archive / "media" / "row.mp4").read_text() == "video"
    assert (archive / "server-logs" / "point.log").read_text() == "log"
    assert not bundle.exists() and not media.exists() and not log.exists()
    # An empty directory holds no evidence and stays where it is.
    assert empty.is_dir() and not (archive / "unused").exists()
    assert supersede(results, "point", {"bundle": bundle}) is None


def test_a_point_refuses_a_media_directory_holding_other_videos(
    tmp_path: Path,
) -> None:
    """Videos named by request ID cannot mix with another run's."""
    media = tmp_path / "media"
    media.mkdir()
    (media / "row.mp4").write_bytes(b"earlier run")
    point = BenchmarkPoint(
        name="video",
        server="server",
        task=TaskName.VIDEO,
        model="model",
        dataset="jsonl",
        dataset_path=str(tmp_path / "rows.jsonl"),
        metrics=(MetricDefinition(("videos_per_second",), "higher"),),
        load=LoadConfig(num_prompts=1),
        video=VideoConfig(media_dir=str(media)),
    )

    with pytest.raises(FileExistsError, match="media directory"):
        asyncio.run(run_point(["http://127.0.0.1:9"], point, tmp_path / "out"))

    assert (media / "row.mp4").read_bytes() == b"earlier run"
    assert not (tmp_path / "out").exists()


def test_two_points_may_not_share_a_media_directory(tmp_path: Path) -> None:
    """Each point's media directory belongs to that point's runs."""
    points = "\n".join(
        f"""
[benchmarks.{name}]
server = "local"
task = "video"
model = "model"
dataset = "jsonl"
dataset_path = "rows.jsonl"

[benchmarks.{name}.video]
size = "1344x768"
fps = 24
audio_sample_rate = 32000
audio_channels = 2
media_dir = "media"

[benchmarks.{name}.metrics]
videos_per_second = "higher"
"""
        for name in ("first", "second")
    )
    profile = tmp_path / "profile.toml"
    profile.write_text(
        '[servers.local]\nport = 8000\ncommand = ["server"]\n' + points,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="media directory of benchmarks"):
        load_config(profile)
