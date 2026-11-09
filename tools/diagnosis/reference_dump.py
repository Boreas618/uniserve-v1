#!/usr/bin/env python3
"""Run the reference's eager generation for one t2va request and write its tensors.

**This tool needs the retired Python tree.** It reads `uniserve`, the Python package that left this
repository with the Python worker, as the reference half of design 10's diagnosis. Run it with a
checkout of the last tree that held the package and that tree's interpreter, with the tree on the
path; `UNISERVE_REFERENCE_TREE` names that checkout:

    PYTHONPATH=$UNISERVE_REFERENCE_TREE:$UNISERVE_REFERENCE_TREE/uniserve_kernels/src \\
        $UNISERVE_REFERENCE_TREE/.venv/bin/python <this file> ...

This is the reference half of the served diagnosis. Run it with the
Python tree's interpreter:

    OUT=<dir> INJECT_NOISE=<served dump> CUDA_VISIBLE_DEVICES=<gpu> python tools/diagnosis/reference_dump.py

`INJECT_NOISE` names a served run's `UNISERVE_DIAG_DIR`. The design draws the served noise on the
device with Philox, and the reference draws it on the CPU with `torch.randn`, so one seed gives two
samples. Copying the served noise into the reference is what makes the two trajectories comparable.

Environment: `OUT` (required), `MODEL`, `PROMPT`, `SECONDS_`, `SEED`, `RATIO` as `h:w`,
`TEXT_CAPACITY`, `INJECT_NOISE`, and `FINAL_DUMP`, a served dump whose final latents the reference
decodes in place of its own, which separates the denoiser from the decoders.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from uniserve.loading import rank
from uniserve.models.minimax_h3 import generation, processing
from uniserve.runner.denoiser import DenoiserRunner
from uniserve.runner.encoder import EncoderRunner

ROOT = Path(os.environ["MODEL"])
OUT = Path(os.environ["OUT"])
PROMPT = os.environ.get("PROMPT", "A red fox walks through falling snow at dusk")
SECONDS = float(os.environ.get("SECONDS_", "4.0"))
SEED = int(os.environ.get("SEED", "1234"))
RATIO = tuple(int(part) for part in os.environ.get("RATIO", "1:1").split(":"))
CAPACITY = int(os.environ.get("TEXT_CAPACITY", "1024"))
MODULES = ["text_encoder", "transformer", "video_decoder", "video_postprocessor", "audio_decoder"]


def save(name, tensor):
    """Write `tensor` as `<OUT>/<name>.npy`, widened to FP32 from a half type."""
    path = OUT / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach()
    if value.dtype in (torch.bfloat16, torch.float16):
        value = value.float()
    np.save(path, value.cpu().numpy())


def load():
    """Load every component on this process's one GPU, as a one-rank worker does."""
    local = {
        "ranks": [0],
        "parallel_config": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "sequence_parallel": {"kind": "local"},
        },
        "units_per_rank": 1,
    }
    components = ("text_encoder", "transformer", "video_decoder", "audio_decoder")
    description = rank.read_config(
        stub=False, path=str(ROOT), io_format="auto", io_mode="eager", download_dir=None,
        num_threads=None, device="cuda:0", generation_device=None,
        components=json.dumps(dict.fromkeys(components, local)), rank=0, role="model",
        expert_split=False,
    )
    return rank.load_model(
        description, modules=MODULES, exclude_modules=[], quantization="{}",
        model_dtype="bfloat16", device="cuda:0", generation_device=None, role="model",
        host_only=False, mesh_components=[], mesh_path_counts=[], mesh_paths=[], meshes=[],
        mesh_units=[], mesh_sequences=[], mesh_projections=[], mesh_projection_rows=[],
        expert_world=None, attention_ranks=0, expert_dedicated=False, expert_exchange=None,
        tensor_experts=False, retained=None, load=MODULES, attention_backend="auto",
        workspace_size=512 << 20, decode_backend="fa2", prefill_backend="auto",
        disable_split_kv=False, decode_split_tile_size=None, prefill_split_tile_size=None,
        use_tensor_core=None,
    )


def hook_encoders():
    """Save the text encoder's features and the conditioner's refined rows."""
    encode = EncoderRunner.encode

    def wrapped(self, *args):
        result = encode(self, *args)
        torch.cuda.synchronize()
        block = "encoder" if len(args) == 2 else "conditioner"
        save(f"{block}/output", args[-1])
        return result

    EncoderRunner.encode = wrapped


def hook_steps():
    """Save each modality's state on entry to every evaluation and after the last."""
    denoise = DenoiserRunner.denoise
    evaluation = [0]

    def snapshot(state, index):
        # The layout's state fields lead with the two samples, video first.
        for name, value in zip(("video", "audio"), state):
            save(f"denoiser/latents_{name}_{index}", value)

    def wrapped(self, prepared, step, conditioning, *state):
        torch.cuda.synchronize()
        snapshot(state, evaluation[0])
        result = denoise(self, prepared, step, conditioning, *state)
        torch.cuda.synchronize()
        evaluation[0] += 1
        snapshot(state, evaluation[0])
        return result

    DenoiserRunner.denoise = wrapped


def hook_noise(served):
    """Replace the reference's draw with the served run's, element for element."""
    draw = DenoiserRunner.draw_noise

    def wrapped(self, seed, layout, size, conditions, spans, condition_noise, *noise):
        draw(self, seed, layout, size, conditions, spans, condition_noise, *noise)
        for tensor, name in zip(noise, ("noise_video", "noise_audio")):
            flat = np.load(f"{served}/denoiser/{name}.npy").reshape(-1).copy()
            tensor.copy_(torch.from_numpy(flat))

    DenoiserRunner.draw_noise = wrapped


def hook_final(served, order):
    """Decode the served final latents, `order` being the served element of each packed element."""
    denoise = generation._denoise

    def wrapped(*args, **kwargs):
        latents = denoise(*args, **kwargs)
        steps = len(list(Path(served, "denoiser").glob("latents_video_*.npy"))) - 1
        video = np.load(f"{served}/denoiser/latents_video_{steps}.npy").ravel()[order]
        audio = np.load(f"{served}/denoiser/latents_audio_{steps}.npy").ravel()
        latents["video"].copy_(torch.from_numpy(video.reshape(latents["video"].shape).copy()))
        latents["audio"].copy_(torch.from_numpy(audio.reshape(latents["audio"].shape).copy()))
        return latents

    generation._denoise = wrapped


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = load()
    hook_encoders()
    hook_steps()
    if os.environ.get("INJECT_NOISE"):
        hook_noise(os.environ["INJECT_NOISE"])
    if os.environ.get("FINAL_DUMP"):
        hook_final(os.environ["FINAL_DUMP"], np.load(os.environ["FINAL_ORDER"]))
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "tokenizer")
    ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    frames = processing.frame_count(SECONDS)
    canvas = processing.canvas(*RATIO)
    model = generation.VideoModel(loaded, "transformer")
    request = generation.VideoSize(
        int(frames), generation.Canvas(canvas.width, canvas.height), len(ids)
    )
    layout = [int(extent) for extent in model.layout_size(request)]
    layout[3] = CAPACITY
    started = time.time()
    result = generation.generate(
        loaded, prompt_token_ids=ids, num_frames=frames, canvas=canvas, seed=SEED, layout=layout
    )
    print(f"generated {frames} frames at {canvas} in {time.time() - started:.2f} s", flush=True)
    save("final/video", result.video_latent)
    save("final/audio", result.audio_latent)
    np.save(OUT / "prompt_ids.npy", np.array(ids, dtype=np.int64))
    np.save(OUT / "pcm.npy", result.audio.cpu().numpy())
    result.frames.cpu().numpy().tofile(OUT / "frames.rgb")


if __name__ == "__main__":
    main()
