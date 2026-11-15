"""`uniserve-eval run` against a stand-in server, through the command line.

The stand-in is a real process that the driver launches from the profile's
command, :mod:`uniserve_testing.standins.server`: an OpenAI-compatible chat
server that prints ``completion`` for each request it answers. Like
`uniserve serve`, it first starts a Storage Server in its runtime directory as
a detached process of its own session, which outlives the server, and records
that process's pid beside itself.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from uniserve_eval.artifacts import supersede
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


def _profile(tmp_path: Path, port: int) -> Path:
    """Write a one-point text profile served by the stand-in on ``port``.

    The profile states the port the stand-in listens on, since the evaluator
    probes the address its profile names.
    """
    (tmp_path / "rows.jsonl").write_text(
        json.dumps({"id": "row", "prompt": "hello", "output_len": 1}) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "uniserve_testing.standins.server",
        "--port",
        str(port),
    ]
    profile = tmp_path / "profile.toml"
    profile.write_text(
        f"""
artifact_root = "results"

[servers.local]
port = {port}
command = {json.dumps(command)}

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


def test_a_rerun_keeps_the_earlier_run_outside_the_canonical_paths(
    tmp_path: Path,
    unused_tcp_port: int,
    evaluate: Callable[[Sequence[str]], int],
) -> None:
    """The second run owns the canonical bundle; the first is archived."""
    profile = _profile(tmp_path, unused_tcp_port)
    results = tmp_path / "results"

    assert evaluate(["--config", str(profile), "run", "point"]) == 0
    first = json.loads((results / "point" / "run.json").read_text())
    assert evaluate(["--config", str(profile), "run", "point"]) == 0
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


def test_a_point_ends_with_the_storage_server_it_started_stopped(
    tmp_path: Path,
    unused_tcp_port: int,
    evaluate: Callable[[Sequence[str]], int],
    wait_exit: Callable[[int, float], bool],
) -> None:
    """No Storage Server, and so no storage, outlives its point."""
    profile = _profile(tmp_path, unused_tcp_port)
    results = tmp_path / "results"

    assert evaluate(["--config", str(profile), "run", "point"]) == 0

    pid, services = (tmp_path / "storage-server.pid").read_text().split()
    # The point's end waits for its services, so the server has exited by
    # the time the command returns.
    assert wait_exit(int(pid), 0.0)
    # The service's log, in which it announced its pid, is kept with the
    # deployment's.
    kept = results / "server-logs" / "point.storage-server.log"
    assert kept.read_text() == f"{pid}\n"
    # The deployment's runtime directory is gone with its services.
    assert not Path(services).exists()
