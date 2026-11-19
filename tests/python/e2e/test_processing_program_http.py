"""An epoch change brings a family its serving processing program lacks.

A deployment's request side (which classes a family claims, how a chat request
becomes a prompt, how its completion is parsed) is the processing program its
epoch's module set fixes (``worker_config.processing``), built from the same
source tree as the epoch's engine. The case serves Qwen3-MoE on a program
revision installed from a tree whose families omit Qwen3.5-MoE
(``Qwen3_5MoeForConditionalGeneration``): a change to a checkpoint of that
family, Qwen3.6-35B-A3B, is refused, its class unserved. The tree's own
program, installed as a later revision with a later engine revision, carries
the family: ``PUT /admin/policy`` of the Qwen3.6 checkpoint with both revisions
starts the target epoch beside the serving one, which runs every request it
holds to its end (the checkpoints process requests otherwise), while every new
request is processed by the target's program. Requests run against the
deployment throughout, a long count streaming across the switch: none fails,
and each carries every token it asked for.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest
from uniserve_testing import admin, chat, updates
from uniserve_testing.assets import CHECKPOINTS
from uniserve_testing.deployments import Deployments
from uniserve_testing.packages import Config
from uniserve_testing.servers import Server

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("qwen3"),
    pytest.mark.model("qwen3.6"),
]

NAME = "processing-program"

#: The tree's processing program, relative to the repository.
PROCESSING_PACKAGE = Path("crates", "processing")

#: The release's family list in the processing program's source, and the
#: list of the earlier tree, which omits Qwen3.5-MoE.
FAMILIES_SOURCE = Path("crates", "processing", "src", "family.rs")
FAMILIES = (
    "pub const FAMILIES: [Family; 4] = [\n"
    "    crate::qwen::QWEN3,\n"
    "    crate::qwen::QWEN3_5_MOE,\n"
)
EARLIER_FAMILIES = (
    "pub const FAMILIES: [Family; 3] = [\n    crate::qwen::QWEN3,\n"
)

#: The options every revision of the deployment states: a small KV cache, few
#: captured graphs and most of the GPU as the deployment's quota, which both
#: epochs share, so the target epoch starts beside the serving one on the same
#: GPU with both checkpoints' weights resident; and a trial the change commits
#: at once after.
OPTIONS: Config = {
    "model": CHECKPOINTS["qwen3"].model,
    "served_model_name": NAME,
    "mem_fraction_static": 0.9,
    "max_model_len": 8192,
    "kv_cache_config": {"max_tokens": 32768},
    "cuda_graph_config": {
        "prefill_token_sizes": [512],
        "decode_batch_sizes": [1, 2, 4, 8],
    },
    "update_config": {"trial": "1s"},
}

#: A greedy count long enough to stream across the switch.
COUNT = chat.body(
    NAME,
    "Count from 1 to 3000, separated by commas and spaces. "
    "Write only the numbers.",
    4096,
    temperature=0,
    logprobs=True,
    **chat.NO_THINKING,
)

#: The short requests sent one after another throughout the change.
SHORT = chat.body(NAME, "Name a color.", 16, temperature=0, **chat.NO_THINKING)

#: How long a stream may wait for its next line, in seconds (scaled): across
#: the switch, until the target's engine serves.
STREAM_GAP_S = 1800.0


def _earlier_tree(repository: Path, root: Path) -> Path:
    """A copy, under `root`, of the repository's tree whose processing
    program omits the Qwen3.5-MoE family: the files Git tracks or would
    track, as they are, with the release's family list edited.
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
    tree = root / "earlier-tree"
    for name in filter(None, listed.decode().split("\0")):
        source = repository / name
        if not source.is_file():
            continue
        target = tree / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    families = tree / FAMILIES_SOURCE
    text = families.read_text()
    assert FAMILIES in text, "the release's family list moved"
    families.write_text(text.replace(FAMILIES, EARLIER_FAMILIES))
    return tree


def _processing(server: Server, package: Path) -> int:
    """The revision the deployment's Build Server installs of the processing
    program in `package`, its module built now, so an epoch that names it
    builds nothing while it prepares.
    """  # noqa: D205
    installed = server.build_server.install(package)
    server.build_server.resolve(
        installed.op,
        installed.backend,
        installed.revision,
        target="wasm32-unknown-unknown",
    )
    return installed.revision


class Load:
    """Short requests sent one after another on a thread of their own until
    `stop`, each with its status and the completion tokens it was answered
    with.
    """  # noqa: D205

    def __init__(self, server: Server) -> None:
        """Start sending to `server`."""
        self.answers: list[tuple[int, int, str]] = []
        self._stopped = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._done = self._pool.submit(self._send, server)

    def _send(self, server: Server) -> None:
        while not self._stopped.is_set():
            response = httpx.post(
                f"{server.base_url}/v1/chat/completions",
                json=SHORT,
                timeout=server.scaled(STREAM_GAP_S),
            )
            tokens = (
                response.json()["usage"]["completion_tokens"]
                if response.status_code == 200
                else 0
            )
            self.answers.append((response.status_code, tokens, response.text))

    def stop(self) -> list[tuple[int, int, str]]:
        """Stop sending once the request in flight ended; every answer."""
        self._stopped.set()
        self._done.result()
        self._pool.shutdown()
        return self.answers


def _stream(server: Server) -> dict[str, Any]:
    """The count streamed to its end: its tokens and its usage."""
    tokens: list[str] = []
    usage: dict[str, Any] | None = None
    with httpx.stream(
        "POST",
        f"{server.base_url}/v1/chat/completions",
        json=chat.stream_options(COUNT),
        timeout=server.scaled(STREAM_GAP_S),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            read, read_usage, done = chat.stream_tokens(line)
            if done:
                break
            tokens.extend(read)
            usage = read_usage or usage
    return {"tokens": tokens, "usage": usage}


def test_a_later_program_serves_a_family_the_earlier_one_did_not(
    deployments: Deployments, repository: Path, tmp_path: Path
):
    """The deployment serves a family its earlier program refused once an
    epoch change brings the later program, with no failed request.
    """  # noqa: D205
    later = CHECKPOINTS["qwen3.6"].model
    with deployments.serve(OPTIONS, name=NAME) as server:
        # The earlier tree's program serves the deployment: its families
        # claim no Qwen3.5-MoE class, so a change to such a checkpoint is
        # refused rather than served.
        tree = _earlier_tree(repository, tmp_path)
        earlier = _processing(server, tree / PROCESSING_PACKAGE)
        admin.revision(
            admin.change(
                server, {"worker_config": {"processing_revision": earlier}}
            )
        )
        configured = admin.policy(server).config["worker_config"]
        assert configured["processing_revision"] == earlier, configured
        refused = admin.change(
            server,
            {"model": later, "worker_config": {"processing_revision": earlier}},
        )
        assert refused.status_code != 200, refused.text
        assert "Qwen3_5MoeForConditionalGeneration" in refused.text, (
            refused.text
        )

        # The tree's program, a later revision, carries the family, as does
        # the later engine revision installed beside it.
        current = _processing(server, repository / PROCESSING_PACKAGE)
        assert current > earlier
        change = updates.engine_revision(server.build_server, repository)
        change["model"] = later
        change["worker_config"]["processing_revision"] = current

        load = Load(server)
        with ThreadPoolExecutor(max_workers=1) as pool:
            streamed = pool.submit(_stream, server)
            applied = admin.change(server, change)
            record = admin.revision(applied)
            print(f"changed: {record}")
            counted = streamed.result()
        # The target serves the new family's requests, its prompt rendered
        # by the later program's template and its output parsed by it.
        answered = chat.complete(server.base_url, SHORT)
        answers = load.stop()

    assert record["result"] == "converged", record
    failed = [answer for answer in answers if answer[0] != 200]
    assert not failed, failed[:3]
    assert answers, "no request ran during the change"
    assert all(tokens > 0 for _, tokens, _ in answers), answers[:3]
    assert counted["usage"]["completion_tokens"] == len(counted["tokens"])
    assert counted["tokens"], "the count streamed nothing"
    assert chat.content(answered), answered
