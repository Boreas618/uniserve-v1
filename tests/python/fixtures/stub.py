"""The stub model read and loaded on the host as a rank loads a model."""

import json

from uniserve.interface import runtime

# The functions as a rank reaches them: by import path, with failures
# carrying the interface's error kinds.
_READ_CONFIG = runtime.import_path("uniserve.interface.loading.read_config")
_LOAD_MODEL = runtime.import_path("uniserve.interface.loading.load_model")

LOCAL = {
    "ranks": [0],
    "parallel_config": {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "sequence_parallel": {"kind": "local"},
    },
    "units_per_rank": 1,
}


def read_config(**overrides):
    """Read the stub model's configuration as a rank does."""
    arguments = {
        "stub": True,
        "path": None,
        "base_model": None,
        "io_format": "auto",
        "io_mode": "eager",
        "download_dir": None,
        "num_threads": None,
        "checksum_manifest": None,
        "device": "cpu",
        "generation_device": None,
        "components": json.dumps({"model": LOCAL}),
        "rank": 0,
        "role": "model",
        "expert_split": False,
        **overrides,
    }
    return _READ_CONFIG(*arguments.values())


def load_model(description, **overrides):
    """Load a described model on one rank without meshes or experts."""
    arguments = {
        "modules": [],
        "exclude_modules": [],
        "quantization": "{}",
        "model_dtype": "bfloat16",
        "device": "cpu",
        "generation_device": None,
        "role": "model",
        "host_only": False,
        "mesh_components": [],
        "mesh_path_counts": [],
        "mesh_paths": [],
        "meshes": [],
        "mesh_units": [],
        "mesh_sequences": [],
        "expert_world": None,
        "attention_ranks": 0,
        "expert_dedicated": False,
        "attention_backend": "auto",
        "workspace_size": 1 << 20,
        "decode_backend": "fa2",
        "prefill_backend": "auto",
        "disable_split_kv": False,
        "decode_split_tile_size": None,
        "prefill_split_tile_size": None,
        "use_tensor_core": None,
        **overrides,
    }
    return _LOAD_MODEL(description, *arguments.values())


def load_stub():
    """Read the stub model's configuration and load it on the CPU.

    Returns the loaded model.
    """
    description = read_config(
        components=json.dumps({"model": LOCAL, "vision_encoder": LOCAL})
    )
    return load_model(description)
