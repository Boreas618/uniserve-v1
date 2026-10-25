"""A packed FastH3 export's manifest names its DiT partition by export root.

The packed ModelOpt FastH3 exports list the calibrated modules of their DiT
partition in ``modelopt_manifest.json`` under the root ``denoiser``. The
registry reads those paths as the package's ``transformer`` modules and still
refuses a listed path that names no ``Linear`` of the model.
"""

import json
import shutil
from pathlib import Path

import pytest

from uniserve.loading import Config as IOConfig
from uniserve.models import registry as models

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).parents[2] / "fixtures/models/fasth3"

# The static NVFP4 W4A4 representation with 16-element blocks that a packed
# manifest declares.
NVFP4 = {
    "activation": "a4",
    "block_scale": "fp8_e4m3",
    "block_size": 16,
    "output": "bf16",
    "tensor_scale": "fp32",
    "values": "e2m1",
    "weight": "w4",
}


def _export(tmp_path, modules):
    """The FastH3 fixture as a packed export calibrating ``modules``."""
    root = tmp_path / "export"
    shutil.copytree(FIXTURE, root)
    (root / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3ModularPipeline"})
    )
    (root / "modelopt_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "numerical_format": NVFP4,
                "components": {
                    "denoiser": {
                        "enabled": True,
                        "modules": {
                            path: {"activation_amax": 2688.0}
                            for path in modules
                        },
                    }
                },
            }
        )
    )
    return root


def test_the_manifest_names_the_dit_partition_under_its_export_root(tmp_path):
    root = _export(
        tmp_path,
        [
            "denoiser.transformer.layers.0.mlp.down",
            "denoiser.transformer.layers.0.mlp.gate_up.projections.gate",
        ],
    )

    config = models.read_config(
        root, io=IOConfig(mode="dummy"), modules=frozenset()
    )

    # The export owns its numerical representation: no runtime presets.
    assert config.checkpoint_format == "modelopt_nvfp4"
    assert not config.precisions


def test_a_listed_module_the_partition_lacks_is_refused(tmp_path):
    root = _export(tmp_path, ["denoiser.transformer.layers.0.mlp.missing"])

    with pytest.raises(
        ValueError, match=r"denoiser\.transformer\.layers\.0\.mlp\.missing"
    ):
        models.read_config(root, io=IOConfig(mode="dummy"), modules=frozenset())
