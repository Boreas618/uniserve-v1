# MiniMax-H3 computation library

The model package loads base MiniMax-H3, FastH3 DMD and FastH3 OmniRef checkpoints through the public model catalog. Each checkpoint determines its denoising components, attention layout, schedule and output heads. Component exports resolve their missing text encoder, processor and VAE components from the caller's local base directory, or from the Hub revision declared by the export when no local base is supplied.

Models are ordinary PyTorch modules. Text and latent encoders, denoisers, video and audio decoders compose the shared numerical capabilities. Execution contexts own kernel workspaces and communication backing; callers own request state and advance the solver. Serving workers use these same modules and numerical calls.

## Generate from Python

The eager library entry accepts a model loaded as a serving rank loads it and a presented text prompt. It runs the request through the runners serving uses (`uniserve.runner`): the text encoder and the denoiser's conditioner, every solver step of the denoiser, and the video and audio decoders, on a CUDA stream of its own. This example uses a local base checkpoint, generates a portrait video and returns RGB frames, stereo PCM and final latents. The checkpoint and its active components must fit on the selected GPU.

```python
import json
from pathlib import Path

from transformers import AutoTokenizer

from uniserve.loading import rank
from uniserve.models.minimax_h3 import processing
from uniserve.models.minimax_h3.generation import generate

root = Path("/path/to/MiniMax-H3")
# Every component on this rank alone.
local = {
    "ranks": [0],
    "parallel_config": {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "sequence_parallel": {"kind": "local"},
    },
    "units_per_rank": 1,
}
description = rank.read_config(
    stub=False,
    path=str(root),
    base_model=None,
    io_format="auto",
    io_mode="eager",
    download_dir=None,
    num_threads=None,
    device="cuda:0",
    generation_device=None,
    components=json.dumps(
        dict.fromkeys(
            ("text_encoder", "denoiser", "video_decoder", "audio_decoder"),
            local,
        )
    ),
    rank=0,
    role="model",
    expert_split=False,
)
loaded = rank.load_model(
    description,
    modules=[
        "text_encoder",
        "denoiser",
        "video_decoder",
        "video_postprocessor",
        "audio_decoder",
    ],
    exclude_modules=[],
    quantization="{}",
    model_dtype="bfloat16",
    device="cuda:0",
    generation_device=None,
    role="model",
    host_only=False,
    mesh_components=[],
    mesh_path_counts=[],
    mesh_paths=[],
    meshes=[],
    mesh_units=[],
    mesh_sequences=[],
    expert_world=None,
    attention_ranks=0,
    expert_dedicated=False,
    attention_backend="auto",
    workspace_size=512 << 20,
    decode_backend="fa2",
    prefill_backend="auto",
    disable_split_kv=False,
    decode_split_tile_size=None,
    prefill_split_tile_size=None,
    use_tensor_core=None,
)
plan = processing.plan_request(
    processing.Task.T2VA,
    processing.Target(aspect_ratio="9:16", duration_seconds=4.0),
    [],
    processing.VisionConfig.from_processor(root / "processor"),
)
prompt = processing.present(
    AutoTokenizer.from_pretrained(root / "tokenizer"),
    plan,
    "A lighthouse keeper climbs a spiral staircase while waves crash below.",
)
result = generate(
    loaded,
    prompt_token_ids=prompt.token_ids,
    num_frames=plan.num_frames,
    canvas=plan.canvas,
    seed=7,
)
```

`result.frames` contains `[frames, height, width, 3]` uint8 RGB; `result.audio` contains stereo int16 PCM at 32 kHz. Frame counts follow the video VAE's 17-frame clips, so this four-second request produces 107 frames. `generate` supports text conditioning; keyframe and reference computation uses the encoders and denoiser's public calls directly.

## Layouts and conditions

A denoiser's `make_size` describes the request's frame count, canvas, prompt and ordered conditions. `layout_size` derives the numerical layout; `holds` checks whether a capacity layout can evaluate a request. Dense layouts pack keyframes and references into their declared prefix. Segment-sparse layouts retain each reference segment's independent selection domain. Layout indices and latent buffers are borrowed tensors, without request identities or pool ownership.

Visual conditions are encoded in independent temporal units and assembled in the declared latent order. A still image may also be encoded in bands of whole latent row groups (`VideoEncoder.row_group` rows each): `VideoEncoder.encode(..., rows=...)` encodes one band from the tiles it depends on, and the bands assemble the whole encoding exactly. An `ExecutionContext` created with `tiles={encoder: operator}` encodes that `TiledEncoder`'s tiles through a borrowed operator equal to its one-tile encoding, such as the replay of a graph captured at `VideoEncoder.still_tile`. Audio conditions supply whole stereo tracks. The denoiser's `encode_conditions`, `prepare_latents`, `prepare_state` and solver calls preserve the checkpoint's condition-noise order, modality schedules and output-head mathematics. Encoded conditions remain fixed while generated video and audio evolve.

For a partitioned model, load and bind component meshes through the public loader and execution context interfaces. Every participating rank executes the same numerical entry; runtime communication owns collective resources. Passing the server's capacity layout to `generate(..., layout=...)`, as the runners describe it (`[frames, width, height, text_rows, condition_rows]`), also preserves the prompt padding and conditioning layout used by serving.
