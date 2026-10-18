"""SenseNova U1 checkpoint metadata normalizes into the served architecture.

The reader refuses checkpoint semantics the dual-route MoT decoder does not
compute, before any numerical construction.
"""

import json

import pytest

from tests.python.fixtures.model_metadata import neo_metadata
from uniserve import loading
from uniserve.models import sensenova_u1

pytestmark = pytest.mark.unit


def _read(tmp_path, metadata):
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    return sensenova_u1.read_config(tmp_path, loading.Config(), sources={})


def test_window_fields_without_windowing_are_served(tmp_path):
    metadata = neo_metadata()
    metadata["llm_config"].update(
        use_sliding_window=False,
        sliding_window=64,
        max_window_layers=1,
        layer_types=["full_attention"] * 3,
    )

    config = _read(tmp_path, metadata)

    assert config.text.num_hidden_layers == 3
    assert config.text.pad_token_id == 3


@pytest.mark.parametrize(
    "fields",
    [
        {"use_sliding_window": True, "sliding_window": 64},
        # Windowing enabled with window fields that select no layer is
        # still a windowed checkpoint.
        {"use_sliding_window": True, "max_window_layers": 3},
        {
            "layer_types": [
                "full_attention",
                "sliding_attention",
                "full_attention",
            ]
        },
    ],
    ids=["enabled", "enabled-without-layers", "sliding-layer"],
)
def test_sliding_window_checkpoints_are_refused(tmp_path, fields):
    metadata = neo_metadata()
    metadata["llm_config"].update(fields)

    with pytest.raises(ValueError, match="sliding-window attention"):
        _read(tmp_path, metadata)


def test_an_activation_without_a_gated_form_is_refused(tmp_path):
    metadata = neo_metadata()
    metadata["llm_config"]["hidden_act"] = "relu"

    with pytest.raises(ValueError, match="unsupported SenseNova hidden_act"):
        _read(tmp_path, metadata)
