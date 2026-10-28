"""Expert redistribution preserves a live request within the serving world.

The admin operation reverses one layer's logical experts across physical
slots and restores them while the same request streams. Its tokens and
usage must equal ordinary decoding before either redistribution. The
expert-parallel case moves those experts between GPUs as well as slots.
"""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.test_world_changes_http import (
    Streaming,
    _admin,
    _body,
    _config,
    _pids,
    _plugin,
    _runs,
    _serve,
    _topology,
    _wait_until,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]

# Discover the real numerical module paths through the public plugin hook
# interface. The test neither guesses checkpoint paths nor alters weights.
EXPERT_PLUGIN = """
from uniserve.nn.moe import FusedMoE


def register(registry):
    registry.hook("expert_layers", expert_layers)
    registry.hook("expert_distribution", expert_distribution)


def expert_layers(payload, models):
    return {
        path: layer.num_experts
        for model in models
        for path, layer in model.named_modules()
        if isinstance(layer, FusedMoE)
    }


def expert_distribution(payload, models):
    return {
        path: list(layer.resident_experts)
        for model in models
        for path, layer in model.named_modules()
        if isinstance(layer, FusedMoE) and path == payload["layer"]
    }
"""


def _layers(base_url: str) -> dict[str, int]:
    response = _admin(base_url, "hooks/expert_layers")
    assert response.status_code == 200, response.text
    layers = {}
    for report in response.json()["result"]["results"]:
        layers.update(report)
    assert layers, "the served model must contain routed expert layers"
    return layers


@pytest.mark.parametrize("topology", ["single", "experts", "dwdp", "host"])
def test_expert_rebalancing_preserves_a_stream_and_its_world(
    tmp_path: Path, topology: str
):
    """Both redistribution directions preserve the request and world."""
    name = "qwen3-expert-rebalance"
    if topology == "dwdp":
        devices, options = _topology("experts", tmp_path)
        options = (*options, "--set", "moe_config.exchange=dwdp")
    elif topology == "host":
        devices, _ = _topology("single", tmp_path)
        groups = tmp_path / "host-weights.toml"
        groups.write_text(
            '[[worker_groups]]\nid = "model"\nranks = ["cuda:0"]\n'
            "queue_depth = 8\n[worker_groups.components.model]\n"
            'weights = "host"\n'
        )
        options = (
            "--worker-groups",
            str(groups),
            "--set",
            "cuda_graph_config.policy=off",
        )
    else:
        devices, options = _topology(topology, tmp_path)
    environment = _plugin(tmp_path / "plugins", "experts", EXPERT_PLUGIN)
    options = (*options, "--set", "kv_cache_config.host_gib=8")
    with _serve(tmp_path, name, devices, options, env=environment) as (
        base_url,
        _log,
    ):
        layers = _layers(base_url)
        layer = min(layers)
        original = list(range(layers[layer]))
        assert len(original) > 1
        world = _config(base_url)["slots"][0]["world"]

        # This bound leaves thousands of decode steps for the two transfers
        # while keeping the reference independent of the changed world.
        body = {**_body(name, stream=False), "max_completion_tokens": 4096}
        response = httpx.post(
            f"{base_url}/v1/chat/completions", json=body, timeout=600
        )
        assert response.status_code == 200, response.text
        reference = response.json()
        expected = [
            item["token"]
            for item in reference["choices"][0]["logprobs"]["content"]
        ]

        streaming = Streaming(
            base_url,
            name,
            body={
                **body,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        streaming.wait_for(64)
        reports = []
        for order in (list(reversed(original)), original):
            changed = _admin(
                base_url, "rebalance_experts", {"order": {layer: order}}
            )
            assert changed.status_code == 200, changed.text
            report = changed.json()["result"]
            reports.append(report)
            (replica,) = report["replicas"]
            assert replica["world"] == world, report
            assert replica["order"] == {layer: order}, report
            status = _config(base_url)
            assert status["slots"][0]["world"] == world, status
            assert status["trial"] is None, status
            # At least 64 tokens must execute with each published order.
            streaming.mark()
            streaming.wait_for(streaming.marked + 64)

        streaming.join()
        assert streaming.tokens == expected
        assert streaming.usage == reference["usage"]
        (tmp_path / "redistribution.json").write_text(
            json.dumps({"layer": layer, "operations": reports}, indent=2)
        )


def test_expert_distribution_survives_daemon_and_engine_recovery(
    tmp_path: Path,
):
    """A daemon handoff preserves the order used to relaunch its engine."""
    name = "qwen3-expert-recovery"
    environment = _plugin(tmp_path / "plugins", "experts", EXPERT_PLUGIN)
    options = ("--set", "kv_cache_config.host_gib=8")
    with _serve(tmp_path, name, options=options, env=environment) as (
        base_url,
        log,
    ):
        successor = None
        try:
            layers = _layers(base_url)
            layer = min(layers)
            order = list(reversed(range(layers[layer])))
            body = {**_body(name, stream=False), "max_completion_tokens": 4096}
            reference = httpx.post(
                f"{base_url}/v1/chat/completions", json=body, timeout=600
            )
            assert reference.status_code == 200, reference.text
            reference = reference.json()
            expected = [
                item["token"]
                for item in reference["choices"][0]["logprobs"]["content"]
            ]
            changed = _admin(
                base_url, "rebalance_experts", {"order": {layer: order}}
            )
            assert changed.status_code == 200, changed.text

            streaming = Streaming(
                base_url,
                name,
                body={
                    **body,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
            streaming.wait_for(64)
            (engine,) = _pids(log, "Engine")
            restarted = _admin(base_url, "restart")
            assert restarted.status_code == 200, restarted.text
            successor = restarted.json()["result"]["pid"]
            streaming.mark()
            streaming.wait_for(streaming.marked + 64)

            # This PID belongs to the isolated deployment started above.
            # The successor daemon must reconstruct its committed order.
            os.kill(engine, signal.SIGTERM)
            _wait_until(
                lambda: (
                    not _runs(engine)
                    and bool(_pids(log, "Engine") - {engine})
                    and httpx.get(f"{base_url}/health", timeout=10).status_code
                    == 200
                ),
                180,
                "the successor daemon did not recover its engine",
            )
            distribution = _admin(
                base_url, "hooks/expert_distribution", {"layer": layer}
            )
            assert distribution.status_code == 200, distribution.text
            assert distribution.json()["result"]["results"] == [{layer: order}]
            streaming.mark()
            streaming.wait_for(streaming.marked + 64)
            streaming.join()
            assert streaming.tokens == expected
            assert streaming.usage == reference["usage"]
        finally:
            # A hot-restarted daemon is no longer the fixture's child.
            if successor is not None and _runs(successor):
                os.kill(successor, signal.SIGTERM)
                _wait_until(
                    lambda: not _runs(successor),
                    120,
                    "the successor daemon did not stop",
                )
