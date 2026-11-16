"""PC sampling of a deployment's CUDA graphs.

Qwen3 serves on the GPU the module's lease holds with
``profiling_config.graph_pc_sampling``: every worker enables PC sampling
before it captures its graphs, so a session's PC samples name the kernels
the graphs launch, which a timeline in node mode records inside graphs.
Sessions of the collectors CUPTI cannot run beside it are refused, naming
the option.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from uniserve_testing import admin, profiling
from uniserve_testing.deployments import Deployments
from uniserve_testing.profiling import Load
from uniserve_testing.servers import Server

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("qwen3"),
    # The module serves one deployment for all of its cases.
    pytest.mark.gpus(1),
]

OPTIONS = (
    "--set",
    "kv_cache_config.max_tokens=32768",
    "--set",
    "profiling_config.graph_pc_sampling=true",
)


@pytest.fixture(scope="module")
def served(module_deployments: Deployments) -> Iterator[Server]:
    """A Qwen3 deployment whose workers keep PC sampling enabled."""
    with profiling.serve(module_deployments, *OPTIONS) as server:
        yield server


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
    served: Server, tmp_path: Path, request_body: dict[str, Any]
):
    response = admin.request(
        served,
        "POST",
        profiling.PROFILE,
        body={**request_body, "output_dir": str(tmp_path)},
    )
    assert response.status_code == 400, response.text
    assert "profiling_config.graph_pc_sampling" in response.text


def test_pc_samples_name_the_kernels_graphs_launch(
    served: Server, tmp_path: Path
):
    output = tmp_path / "profiles"
    with Load(served):
        profiling.start(
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
        session = profiling.ended(served)
    assert session["end_reason"] == "window", session

    [path] = sorted(output.glob("*.sqlite"))
    states = {
        (row["name"], row["state"])
        for row in profiling.table(path, "collectors")
    }
    assert {state for _, state in states} == {"prepared", "ended"}, states
    names = {row["id"]: row["text"] for row in profiling.table(path, "strings")}
    in_graphs = {
        names[kernel["name"]]
        for kernel in profiling.table(path, "kernel")
        if kernel["graph"]
    }
    sampled = {row["function"] for row in profiling.table(path, "pc_sample")}
    takes = profiling.table(path, "pc_take")
    assert in_graphs, "the timeline recorded no kernel inside a CUDA graph"
    # Samples of the kernels that ran inside the decode step's graphs are
    # attributed to their functions.
    assert sampled & in_graphs, (sorted(sampled)[:10], sorted(in_graphs)[:10])
    total = sum(take["total_samples"] for take in takes)
    non_user = sum(take["non_user_samples"] for take in takes)
    assert total > 0 and non_user < total, takes
