"""The numerical comparison of a change against its base revision.

Each test runs ``tests/python/numerics/compare.py`` with this repository's
committed tree as the base and an export of the same tree, edited by the
test, as the change, on the two-layer Qwen3-MoE fixture checkpoint. The
captures run on the host, where every computation is batch invariant at the
fixture's sizes, so the gate is exercised wherever the integration suite
runs.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from tests.python.fixtures.checkpoints import qwen_moe_checkpoint

pytestmark = pytest.mark.integration

REPOSITORY = Path(__file__).resolve().parents[3]
COMPARE = REPOSITORY / "tests" / "python" / "numerics" / "compare.py"
EXPERTS = "uniserve/models/qwen3/transformer.py"
# The routed experts' combination in Qwen3's MoE layer.
COMBINATION = "        return self.experts(hidden, ids, weights)\n"
FUNCTION = '    function = "models.qwen3.transformer.MoE/1"\n'


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("qwen3-moe")
    qwen_moe_checkpoint(root, max_position_embeddings=64)
    return root


def _change(root: Path, *edits: tuple[str, str]) -> Path:
    """The committed tree exported to ``root``, with ``edits`` applied to
    its MoE layer as ``(text, replacement)`` pairs.
    """  # noqa: D205
    archive = subprocess.run(
        ["git", "-C", str(REPOSITORY), "archive", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        tree.extractall(root, filter="data")
    source = root / EXPERTS
    text = source.read_text()
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    source.write_text(text)
    return root


def _compare(change: Path, checkpoint: Path, output: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            "--base",
            "HEAD",
            "--head",
            str(change),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--prompt",
            "12",
            "--context",
            "40",
            "--tokens",
            "128",
            "--rows",
            "16",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode in (0, 1), result.stderr
    return result.returncode, json.loads((output / "report.json").read_text())


def _components(leaves: list[dict]) -> set[str]:
    return {leaf["component"] for leaf in leaves}


def test_an_unchanged_computation_passes(tmp_path, checkpoint):
    code, report = _compare(
        _change(tmp_path / "tree"), checkpoint, tmp_path / "report"
    )

    assert code == 0
    assert not report["violations"]
    # Every component of the model was compared, in both phases.
    compared = _components(report["leaves"])
    assert {"backbone.layers.0.mlp", "backbone.layers.1.mlp", "lm_head"} <= (
        compared
    )
    assert {leaf["phase"] for leaf in report["leaves"]} == {"prefill", "decode"}


def test_a_changed_computation_keeping_its_function_is_refused(
    tmp_path, checkpoint
):
    # The experts' combination scaled by 1 + 2^-7, beyond BF16 rounding.
    change = _change(
        tmp_path / "tree",
        (COMBINATION, COMBINATION.replace(")\n", ") * 1.0078125\n")),
    )
    code, report = _compare(change, checkpoint, tmp_path / "report")

    assert code == 1
    refused = _components(report["violations"])
    assert "backbone.layers.0.mlp" in refused
    assert {
        leaf["function"]
        for leaf in report["violations"]
        if leaf["component"] == "backbone.layers.0.mlp"
    } == {"models.qwen3.transformer.MoE/1"}
    # What the first MoE layer's inputs depend on is unchanged and passes.
    assert not refused & {
        "backbone.embedding",
        "backbone.layers.0.attention",
        "backbone.layers.0.mlp.router",
        "backbone.layers.0.mlp.experts",
    }


def test_a_changed_computation_declaring_its_revision_is_not_compared(
    tmp_path, checkpoint
):
    change = _change(
        tmp_path / "tree",
        (COMBINATION, COMBINATION.replace(")\n", ") * 1.0078125\n")),
        (FUNCTION, FUNCTION.replace("MoE/1", "MoE/2")),
    )
    code, report = _compare(change, checkpoint, tmp_path / "report")

    assert code == 0
    changed = {item["call"] for item in report["not_compared"]["changed"]}
    assert {"backbone.layers.0.mlp#0", "backbone.layers.1.mlp#0"} <= changed
    # Everything computed after the first MoE layer has other inputs now.
    upstream = {item["call"] for item in report["not_compared"]["upstream"]}
    assert "backbone.layers.1.attention#0" in upstream
    # Everything before it is still compared.
    assert "backbone.layers.0.attention" in _components(report["leaves"])
