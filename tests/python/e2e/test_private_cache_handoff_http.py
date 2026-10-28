"""A kept numerical state remains reusable through later mapped worlds."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.test_world_changes_http import (
    CHANGE,
    REPORT_SETTLE,
    REVISION_PLUGIN,
    Streaming,
    _admin,
    _body,
    _config,
    _plugin,
    _prefix_hits,
    _serve,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("qwen3")]


def test_a_private_continuation_survives_mapped_promotion_and_rollback(
    tmp_path: Path,
):
    """Keep makes the stream's later pages private to its continuation.

    A subsequent promotion and rollback must reuse those later pages while
    delivering the same tokens and usage as the uninterrupted reference.
    """
    name = "qwen3-private-continuation"
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    environment = {
        "PYTHONPATH": os.pathsep.join(
            [str(plugins), *filter(None, [os.environ.get("PYTHONPATH")])]
        )
    }
    block_size = 64
    options = (
        "--set",
        f"kv_cache_config.block_size={block_size}",
        "--set",
        "kv_cache_config.host_gib=8",
    )
    with _serve(tmp_path, name, options=options, env=environment) as (
        base_url,
        _log,
    ):
        body = {**_body(name, stream=False), "max_completion_tokens": 4096}
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
        print("initial stream: 64 tokens", flush=True)

        # Only successor processes import the numerical revision. Its
        # arithmetic is unchanged, while Keep retains the original state
        # seed and disables public publication of this request's later pages.
        _plugin(plugins, "revision", REVISION_PLUGIN)
        changed = httpx.put(
            f"{base_url}/admin/config",
            json={"set": [CHANGE], "in_flight": "keep"},
            timeout=1800,
        )
        assert changed.status_code == 200, changed.text
        first = changed.json()["result"]["world"]
        assert first["numerics"] == "changed", first
        assert first["kv_cache"] == "mapped", first
        committed = _admin(base_url, "config/commit")
        assert committed.status_code == 200, committed.text
        previous = _config(base_url)["slots"][0]["world"]
        streaming.wait_for(len(streaming.tokens) + 512)
        before = len(streaming.tokens)
        hits = _prefix_hits(base_url)
        print(
            f"private continuation: {before} tokens before promotion",
            flush=True,
        )

        changed = httpx.put(
            f"{base_url}/admin/config",
            json={
                "set": [
                    "cuda_graph_config.decode_batch_sizes=[1, 2, 4, 8, 16, 32]"
                ],
                "in_flight": "keep",
            },
            timeout=1800,
        )
        assert changed.status_code == 200, changed.text
        second = changed.json()["result"]["world"]
        assert second["kv_cache"] == "mapped", second
        streaming.wait_for(len(streaming.tokens) + 64)
        time.sleep(REPORT_SETTLE)
        resumed = _prefix_hits(base_url) - hits
        assert resumed >= before - block_size, resumed
        print(f"mapped promotion: {resumed:.0f} cached tokens", flush=True)

        hits = _prefix_hits(base_url)
        rolled = _admin(
            base_url, "config/rollback", {"reason": "operator rollback"}
        )
        assert rolled.status_code == 200, rolled.text
        assert _config(base_url)["slots"][0]["world"] == previous
        streaming.wait_for(len(streaming.tokens) + 64)
        time.sleep(REPORT_SETTLE)
        restored = _prefix_hits(base_url) - hits
        assert restored >= before - block_size, restored
        print(f"rollback: {restored:.0f} cached tokens", flush=True)
        streaming.join()
        # Generate the independent reference last: doing it first would
        # leave public pages of the entire answer that could conceal a
        # missing private continuation during the second migration.
        reference = httpx.post(
            f"{base_url}/v1/chat/completions", json=body, timeout=600
        )
        reference.raise_for_status()
        reference = reference.json()
        expected = [
            entry["token"]
            for entry in reference["choices"][0]["logprobs"]["content"]
        ]
        print(f"reference: {len(expected)} tokens", flush=True)
        assert streaming.tokens == expected
        assert streaming.usage == reference["usage"]
        (tmp_path / "continuation.json").write_text(
            json.dumps(
                {
                    "first": first,
                    "second": second,
                    "promotion_cached_tokens": resumed,
                    "rollback_cached_tokens": restored,
                    "usage": streaming.usage,
                },
                indent=2,
            )
        )
