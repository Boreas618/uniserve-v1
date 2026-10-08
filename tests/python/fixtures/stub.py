"""The stub model loaded on the host through the numerical interface."""

import tvm_ffi

LOCAL = {
    "ranks": [0],
    "parallel_config": {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "sequence_parallel": {"kind": "local"},
    },
    "units_per_rank": 1,
}
IO = {
    "format": "auto",
    "mode": "eager",
    "download_dir": None,
    "num_threads": None,
    "checksum_manifest": None,
}
ATTENTION = {
    "workspace_size": 1 << 20,
    "decode_backend": "fa2",
    "prefill_backend": "auto",
    "disable_split_kv": False,
    "decode_split_tile_size": None,
    "prefill_split_tile_size": None,
    "use_tensor_core": None,
}


def load_stub(table):
    """Describe, resolve and load the stub model on the CPU.

    Requests are TVM-FFI containers, as the worker sends them. Returns the
    loaded model handle.
    """
    description, _ = table["model.describe"](
        tvm_ffi.convert(
            {
                "stub": True,
                "path": None,
                "base_model": None,
                "io": IO,
                "device": "cpu",
                "generation_device": None,
                "components": {
                    "model": LOCAL,
                    "vision_encoder": LOCAL,
                },
                "rank": 0,
                "role": "model",
                "expert_split": False,
            }
        )
    )
    resolved = table["model.resolve"](
        description,
        {
            "modules": [],
            "exclude_modules": [],
            "quantization": "{}",
            "model_dtype": "bfloat16",
        },
    )
    return table["model.load"](
        resolved,
        tvm_ffi.convert(
            {
                "device": "cpu",
                "generation_device": None,
                "role": "model",
                "host_only": False,
                "meshes": [],
                "experts": None,
                "attention_backend": "auto",
                "attention": ATTENTION,
            }
        ),
    )
