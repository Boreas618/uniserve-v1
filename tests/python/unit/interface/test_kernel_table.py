"""The kernel table the library renders from the runners' records."""

import json

import pytest

from uniserve.interface import kernels
from uniserve.runner.context import kernel_records

pytestmark = pytest.mark.unit


def _table(runners, stage="startup"):
    """Render ``{label: (device, records)}`` as a rank does."""
    reported = {
        label: kernel_records(records)
        for label, (_, records) in runners.items()
    }
    return json.loads(
        kernels.table(
            stage,
            0,
            "cuda:0",
            list(runners),
            [device for device, _ in runners.values()],
            [len(texts) for texts in reported.values()],
            [text for texts in reported.values() for text in texts],
        )
    )


def test_numbered_layers_compress_into_ranges():
    assert kernels.layer_paths(
        ["layers.0.mlp", "layers.1.mlp", "layers.2.mlp", "layers.5.mlp", "head"]
    ) == ["layers.{0-2,5}.mlp", "head"]
    # Two varying positions cannot share one template.
    assert kernels.layer_paths(["a.0.b.1", "a.1.b.0"]) == ["a.0.b.1", "a.1.b.0"]


def test_equal_records_share_a_call_site_and_portable_sites_are_listed():
    def site(path, provider):
        return {"path": path, "op": "matmul", "provider": provider}

    table = _table(
        {
            "model": (
                "cuda:0",
                [
                    site("layers.0.proj", "cublas"),
                    site("layers.1.proj", "cublas"),
                    site("head", "torch"),
                ],
            )
        }
    )

    assert (table["stage"], table["rank"], table["device"]) == (
        "startup",
        0,
        "cuda:0",
    )
    (runner,) = table["runners"]
    assert (runner["runner"], runner["device"]) == ("model", "cuda:0")
    assert runner["call_sites"] == [
        {"op": "matmul", "provider": "cublas", "layers": ["layers.{0-1}.proj"]},
        {"op": "matmul", "provider": "torch", "layers": ["head"]},
    ]
    # Portable torch code serves the head on a CUDA runner.
    assert table["portable"] == [
        {
            "runner": "model",
            "op": "matmul",
            "layers": ["head"],
            "inputs": ["every input"],
        }
    ]


def test_runners_sharing_a_label_report_each_layer_once():
    record = {"path": "layers.0.proj", "op": "matmul", "provider": "cublas"}
    # Two runners of one label (one per expert microbatch) report alike.
    table = _table({"model": ("cuda:0", [record, dict(record)])})
    assert table["runners"][0]["call_sites"] == [
        {"op": "matmul", "provider": "cublas", "layers": ["layers.0.proj"]}
    ]


def test_a_prepared_attention_site_hides_its_unprepared_default():
    def attention(provider, inputs=None):
        record = {"path": "layers.0.attention", "op": "attention"}
        record["provider"] = provider
        if inputs is not None:
            record["inputs"] = inputs
        return record

    table = _table(
        {
            "model": (
                "cuda:0",
                [
                    attention(None),
                    attention(
                        "auto",
                        {
                            "causal rows": "flashinfer",
                            "non-causal rows": "torch",
                        },
                    ),
                ],
            ),
            # A host runner's torch sites are not portable fallbacks.
            "host": ("cpu", [attention("torch")]),
        }
    )

    model, host = table["runners"]
    assert [site["provider"] for site in model["call_sites"]] == ["auto"]
    assert [site["provider"] for site in host["call_sites"]] == ["torch"]
    # Automatic selection names the input class torch serves.
    assert table["portable"] == [
        {
            "runner": "model",
            "op": "attention",
            "layers": ["layers.0.attention"],
            "inputs": ["non-causal rows"],
        }
    ]
