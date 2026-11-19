"""A deployment moves to a later build in place within its version windows.

Every interface between a deployment's processes is spoken in a window of
versions, and two peers agree at their hello on the newest version both speak
(``docs/architecture.md``, Interface versions). The case serves Qwen3 on this
tree's build, whose engine interface's window is ``ENGINE_INTERFACE``
(``crates/proto/src/lib.rs``), and builds a later release from a copy of the
tree whose window holds one more version: its base executable (``uniserve``)
with cargo, and its engine as a revision of the engine package installed at
the deployment's Build Server. While short requests run one after another and
counts stream, the deployment

- restarts its daemon into the later build in place (``executable``): the
  successor takes the running engines over in the version they agreed on with
  the daemon of this build, and the counts in flight stream on;
- restarts its Storage Server into the later build in place
  (``storage-server restart`` of the later ``uniserve``);
- moves its engines to the later build's engine revision in an epoch update,
  whose engines agree with the successor on the later version.

No request fails, and each count carries every number it wrote.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest
from uniserve_testing import admin, builds, chat, updates
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.deployments import (
    Deployments,
    runtime_directory,
    storage_socket,
    storage_status,
)
from uniserve_testing.packages import Config, merged
from uniserve_testing.servers import Server, environment
from uniserve_testing.updates import Stream, numbers

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

NAME = "interface-versions"

#: The source of the engine interface's window, relative to the repository.
WINDOW_SOURCE = Path("crates", "proto", "src", "lib.rs")

#: The statement of the window in it: its oldest and newest versions.
WINDOW = re.compile(
    r"pub const ENGINE_INTERFACE: service::Versions = "
    r"service::Versions::(?:one\((?P<one>\d+)\)|new\((?P<oldest>\d+), "
    r"(?P<newest>\d+)\));"
)

#: The options of the deployment: the update modules' (few captured graphs, a
#: small device KV cache) with a trial the epoch update commits at once after.
OPTIONS: Config = merged(
    updates.PROFILE,
    {
        "model": CHECKPOINTS["qwen3"].model,
        "served_model_name": NAME,
        "update_config": {"trial": "5s"},
    },
)

#: The short requests sent one after another throughout the case.
SHORT = chat.body(NAME, "Name a color.", 16, **chat.GREEDY, **chat.NO_THINKING)

#: The counts that stream across the daemon's restart and across the epoch
#: update: long enough to be in flight when each begins.
COUNT_TO = 3000
COUNT_TOKENS = 8192

#: How long a successor daemon may take to serve the admin socket, and a
#: successor Storage Server to take its socket over, in seconds.
TAKEOVER_S = 300.0

#: How long the later build's base executable may take to build, in seconds:
#: a release build of the workspace's root package.
BUILD_S = 3600.0


def _window(source: str) -> tuple[int, int]:
    """The oldest and newest versions of the engine interface's window that
    the source text of ``crates/proto/src/lib.rs`` states.
    """  # noqa: D205
    stated = WINDOW.search(source)
    assert stated is not None, "the engine interface's window moved"
    if stated["one"] is not None:
        version = int(stated["one"])
        return version, version
    return int(stated["oldest"]), int(stated["newest"])


def _later_tree(repository: Path, root: Path) -> tuple[Path, tuple[int, int]]:
    """A copy, under `root`, of the repository's tree whose engine interface
    speaks one version more than this tree's: the files Git tracks or would
    track, as they are, with the window's newest version raised by one. Returns
    the tree and its window.
    """  # noqa: D205
    listed = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        capture_output=True,
        check=True,
    ).stdout
    tree = root / "later-tree"
    for name in filter(None, listed.decode().split("\0")):
        source = repository / name
        if not source.is_file():
            continue
        target = tree / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    window_source = tree / WINDOW_SOURCE
    text = window_source.read_text()
    oldest, newest = _window(text)
    later = (oldest, newest + 1)
    window_source.write_text(
        WINDOW.sub(
            "pub const ENGINE_INTERFACE: service::Versions = "
            f"service::Versions::new({later[0]}, {later[1]});",
            text,
        )
    )
    return tree, later


def _base_executable(tree: Path, work: Path) -> Path:
    """The ``uniserve`` of the source tree `tree`, built with cargo at the
    release its ``rust-toolchain.toml`` pins, locked, into a target directory
    of its own under `work`: cargo names a workspace member's artifacts
    independently of where its tree lies, so the build of another tree in one
    target directory would replace the environment's build's artifacts.
    """  # noqa: D205
    target = work / "later-target"
    built = environment(CARGO_TARGET_DIR=str(target))
    if jobs := os.environ.get("CARGO_BUILD_JOBS"):
        built["CARGO_BUILD_JOBS"] = jobs
    with (work / "later-build.log").open("w") as log:
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--locked",
                "--package",
                "uniserve",
                "--bin",
                "uniserve",
            ],
            cwd=tree,
            env=built,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=BUILD_S,
        )
    return target / "release" / "uniserve"


def _engine_revision(server: Server, tree: Path) -> int:
    """The revision the deployment's Build Server installs of the engine
    package of the source tree `tree`, its executable built now, so the epoch
    update that names it builds nothing while it prepares.
    """  # noqa: D205
    installed = server.build_server.install(tree / builds.ENGINE_PACKAGE)
    server.build_server.resolve(
        installed.op, installed.backend, installed.revision
    )
    return installed.revision


class Load:
    """Short requests sent one after another on a thread of their own until
    `stop`, each with its status and its body.
    """  # noqa: D205

    def __init__(self, server: Server) -> None:
        """Start sending to `server`."""
        self.answers: list[tuple[int, str]] = []
        self._stopped = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._done = self._pool.submit(self._send, server)

    def _send(self, server: Server) -> None:
        while not self._stopped.is_set():
            try:
                response = httpx.post(
                    f"{server.base_url}/v1/chat/completions",
                    json=SHORT,
                    timeout=server.scaled(TAKEOVER_S),
                )
                self.answers.append((response.status_code, response.text))
            except httpx.HTTPError as error:
                self.answers.append((0, repr(error)))

    def stop(self) -> list[tuple[int, str]]:
        """Stop sending once the request in flight ended; every answer."""
        self._stopped.set()
        self._done.result()
        self._pool.shutdown()
        return self.answers


def _count(server: Server) -> Stream:
    """A greedy count streamed from the deployment, once it wrote its first
    numbers.
    """  # noqa: D205
    request = chat.body(
        NAME,
        f"Count from 1 to {COUNT_TO}, separated by commas and spaces. "
        "Write only the numbers.",
        COUNT_TOKENS,
        **chat.GREEDY,
        **chat.NO_THINKING,
    )
    stream = Stream(server.base_url, {**request, "stream": True})
    stream.wait_for(
        lambda streamed: len(numbers(streamed.text)) >= 8,
        "the count did not stream",
    )
    return stream


def _assert_whole(stream: Stream, before: int) -> None:
    """The count ended by itself, with every number from 1 in order and more
    numbers than the `before` it wrote when the change began.
    """  # noqa: D205
    stream.join()
    assert stream.error is None, f"the count failed: {stream.error!r}"
    counted = numbers(stream.text)
    assert counted == list(range(1, len(counted) + 1)), stream.text[:400]
    assert len(counted) > before, "the count carried no number after the change"
    assert stream.reasons and stream.reasons[-1] in ("stop", "length"), (
        stream.reasons
    )


def _engine(server: Server) -> dict[str, Any]:
    """The serving engine's latest process state, with the version of the
    engine interface its links agreed on (``interface``).
    """  # noqa: D205
    return admin.serving_engine(server)


def test_a_deployment_moves_to_a_later_build_in_place(
    deployments: Deployments,
    repository: Path,
    uniserve: Path,
    tmp_path: Path,
):
    oldest, newest = _window((repository / WINDOW_SOURCE).read_text())
    later, (later_oldest, later_newest) = _later_tree(repository, tmp_path)
    assert later_oldest <= newest < later_newest, (oldest, newest, later_newest)
    successor_image = _base_executable(later, tmp_path)

    with (
        runtime_directory(uniserve, tmp_path / "storage-server") as runtime,
        deployments.serve(OPTIONS, name=NAME, runtime=runtime) as server,
        updates.successors() as successors,
    ):
        # The engines of this build agree with its daemon on its newest version.
        engine = _engine(server)
        assert engine.get("interface") == newest, engine
        engine_pids = {
            process["pid"]
            for process in admin.processes(server, group="engine")
            if process["state"] not in admin.ENDED
        }
        load = Load(server)
        # The load runs until the case ends, whether it passes or fails.
        try:
            # The daemon restarts into the later build while a count streams:
            # the successor speaks the engines' version, and serves them on.
            stream = _count(server)
            before = len(numbers(stream.text))
            previous = admin.daemon_pid(server)
            after = admin.now(server)
            response = admin.change(
                server, {"executable": str(successor_image)}
            )
            assert response.status_code == 202, response.text
            assert response.json()["effect"] == "daemon", response.text

            def bound() -> int | None:
                try:
                    pid = admin.daemon_pid(server)
                except OSError:
                    return None
                return None if pid == previous else pid

            server.wait_until(
                lambda: bound() is not None,
                TAKEOVER_S,
                f"no daemon took the admin socket over from {previous}",
            )
            successor = bound()
            assert successor is not None
            successors.append(successor)
            admin.wait_for_log(
                server,
                "node",
                after=after,
                timeout_s=TAKEOVER_S,
                what=f"daemon {successor} did not serve the deployment",
                state="serving",
            )
            assert Path(f"/proc/{successor}/exe").resolve() == (
                successor_image.resolve()
            ), "the successor runs another executable than the later build's"
            _assert_whole(stream, before)
            assert {
                process["pid"]
                for process in admin.processes(server, group="engine")
                if process["state"] not in admin.ENDED
            } >= engine_pids, "the successor did not keep the engines"
            engine = _engine(server)
            assert engine.get("interface") == newest, engine
            updates.wait_for_end(previous)

            # The Storage Server restarts into the later build in place.
            storage = storage_status(uniserve, runtime)
            assert storage is not None, (
                "no Storage Server serves the deployment"
            )
            subprocess.run(
                [
                    str(successor_image),
                    "storage-server",
                    "restart",
                    "--socket",
                    str(storage_socket(runtime)),
                ],
                env=environment(),
                check=True,
                capture_output=True,
                text=True,
            )

            def storage_pid() -> int | None:
                record = storage_status(uniserve, runtime)
                return None if record is None else record["pid"]

            server.wait_until(
                lambda: storage_pid() not in (None, storage["pid"]),
                TAKEOVER_S,
                "the later build's Storage Server did not take over",
            )
            storage_successor = storage_pid()
            assert storage_successor is not None
            assert Path(f"/proc/{storage_successor}/exe").resolve() == (
                successor_image.resolve()
            ), "the Storage Server runs another build than the later one"

            # The later build's engine is installed and built now: a package of
            # the deployment that names no engine revision takes the latest,
            # so one installed before the daemon's restart would be the epoch
            # its successor converges to, draining the engine whose responses
            # it took over, which a taken-over response does not migrate from.
            revision = _engine_revision(server, later)

            # The epoch update moves the engines to the later build's engine
            # revision while a count the successor admitted streams; its engines
            # agree with the successor on the later build's newest version.
            stream = _count(server)
            before = len(numbers(stream.text))
            moved = {"worker_config": {"engine_revision": revision}}
            record = admin.revision(admin.change(server, moved))
            assert record["effect"] == "epoch", record
            update = admin.logs(server, "update")[-1].fields
            assert update["phase"] == "committed", update
            _assert_whole(stream, before)
            engine = _engine(server)
            assert engine.get("interface") == later_newest, engine

        finally:
            answers = load.stop()
        failed = [answer for answer in answers if answer[0] != 200]
        assert answers, "no request ran across the changes"
        assert not failed, failed[:3]
        errors = admin.counter(
            server, "uniserve_requests_total", outcome="error"
        )
        assert errors == 0, errors
