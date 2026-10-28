"""PC sampling of a deployment's CUDA graphs.

Qwen3 serves on one GPU (``UNISERVE_PROFILING_DEVICE``, default 0) with
``profiling_config.graph_pc_sampling``: every worker enables PC sampling
before it captures its graphs, so a session's PC samples name the kernels
the graphs launch, which a timeline in node mode records inside graphs.
Sessions of the collectors CUPTI cannot run beside it are refused, naming
the option.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.profiling_helpers import Load, ended, serve, start
from uniserve_eval.profile import ProfileFile

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

OPTIONS = (
    "dtype=bfloat16",
    "kv_cache_config.max_tokens=32768",
    "chat_config.reasoning_parser=none",
    "cuda_graph_config.prefill_token_sizes=[512]",
    "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8]",
    "profiling_config.graph_pc_sampling=true",
)


@pytest.fixture(scope="module")
def served(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A Qwen3 deployment whose workers keep PC sampling enabled."""
    with serve(tmp_path_factory.mktemp("served"), OPTIONS) as base_url:
        yield base_url


@pytest.mark.parametrize(
    "request_body",
    [
        {"collectors": ["pm_sampling"]},
        {"collectors": ["range_metrics"]},
        {"collectors": ["timeline"]},
        {"collectors": ["timeline"], "timeline": {"graphs": "whole"}},
    ],
)
def test_collectors_cupti_cannot_run_beside_it_are_refused(
    served: str, tmp_path: Path, request_body: dict
):
    response = httpx.post(
        f"{served}/admin/profile",
        json={**request_body, "output_dir": str(tmp_path)},
        timeout=600,
    )
    assert response.status_code == 400, response.text
    assert "profiling_config.graph_pc_sampling" in response.text


def test_pc_samples_name_the_kernels_graphs_launch(served: str, tmp_path: Path):
    output = tmp_path / "profiles"
    with Load(served):
        start(
            served,
            {
                "collectors": ["timeline", "pc_sampling"],
                "timeline": {"graphs": "nodes"},
                "stages": ["decode"],
                "start_step": 5,
                "steps": 40,
                "output_dir": str(output),
            },
        )
        session = ended(served, 600)
    assert session["end_reason"] == "window", session

    [path] = sorted(output.glob("*.sqlite"))
    with ProfileFile(path) as profile:
        states = {
            (row["name"], row["state"]) for row in profile.rows("collectors")
        }
        assert {state for _, state in states} == {"prepared", "ended"}, states
        names = {row["id"]: row["text"] for row in profile.rows("strings")}
        in_graphs = {
            names[kernel["name"]]
            for kernel in profile.rows("kernel")
            if kernel["graph"]
        }
        sampled = {row["function"] for row in profile.rows("pc_sample")}
        takes = list(profile.rows("pc_take"))
    assert in_graphs, "the timeline recorded no kernel inside a CUDA graph"
    # Samples of the kernels that ran inside the decode step's graphs are
    # attributed to their functions.
    assert sampled & in_graphs, (sorted(sampled)[:10], sorted(in_graphs)[:10])
    total = sum(take["total_samples"] for take in takes)
    non_user = sum(take["non_user_samples"] for take in takes)
    assert total > 0 and non_user < total, takes
