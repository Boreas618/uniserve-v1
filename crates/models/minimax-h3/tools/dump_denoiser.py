#!/usr/bin/env python3
r"""Dump one probe of the MiniMax-H3 denoiser from today's implementation.

The reference half of the block-by-block numeric comparison (`specs/worker.md` 10) for the block that
the other dumpers do not reach: the two VAE dumpers cover the decoders, the encoder dumper covers the
text tower, `dump_tables.py` covers a request's packing tables, and this one covers the **initial
noise** and the two attention boundaries that the packing computes but `dump_tables.py` does not
write.

Names are taken from the protocol's table (`notes/h3/diagnosis.md` §2), never re-derived here, so a
tensor that changes shape changes in one place.

    .venv/bin/python crates/models/minimax-h3/tools/dump_denoiser.py --list
    .venv/bin/python crates/models/minimax-h3/tools/dump_denoiser.py \\
        --checkpoint /opt/uv/models/MiniMax-H3 --case <case.json> --probe noise \\
        --out artifacts/diagnosis/denoiser/ref

One probe per invocation, so a failure costs one tensor set and not the batch. The plan is printed
before anything is loaded: a wrong case file, a canvas that is not a multiple of 32 or a checkpoint
whose transformer config is absent are three different first lines rather than one opaque failure.

**What runs where.** The `noise` and `cu_seqlens` probes are host arithmetic over the request and the
checkpoint's transformer config: they need no weights, no device and no compiled extension, and they
are the probes whose tensors carry a convention (a draw's addressing, a row's boundaries) rather than
a matrix product. The `state` and `velocity` probes need the reference's own runner
(`uniserve.runner.denoiser.DenoiserRunner`), which takes a CUDA stream, a loaded model and the
bindings' scratch areas: they are listed here with their plan so a caller can see what remains, and
they refuse with the reason rather than pretending. **This script has not been run on the run host** -
the caller is its first exerciser, and the reason it can still be read with confidence is that its
sister dumpers write the same container through the same `dump_tables.dump` function.

The noise probe writes **two** tensors from one call, because both come from one RNG stream: the
reference draws the video's complete native latent and then the audio's from a generator seeded per
request (`uniserve/diffusion/noise.py:83-93`), so drawing them in separate invocations would draw
them from two streams and neither tensor would be the reference's.

A conditioned request's draws are **not** covered: the conditions' own draws come first in the stream
and the reference makes them in the runner (`DenoiserRunner.draw_noise`), which the host half cannot
reach. Run this probe on a `t2va` case, and read the state/velocity probes' absence as the reason.

This is a **diagnosis, never an acceptance gate** (`specs/worker.md` 10).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The probe catalogue: name -> (what it writes, what it needs). The order is the order a caller
# should run them in: the cheapest and most convention-heavy first.
CATALOGUE = {
    "noise": (
        ("noise_video", "noise_audio"),
        "host: the request's initial noise, drawn as the reference draws it",
    ),
    "rows": (
        ("packed_rows", "cu_seqlens"),
        "host: the row order and the attention boundaries a dense layout implies",
    ),
    "state": (
        ("latents_video_e", "latents_audio_e"),
        "run host: the samples on entry to evaluation `e`, which the runner owns",
    ),
    "velocity": (
        ("velocity_video_e", "velocity_audio_e"),
        "run host: the network's prediction at evaluation `e`, which the runner owns",
    ),
}

# The probes this script can produce without the reference's runner, a device or weights.
HOST_PROBES = frozenset({"noise", "rows"})


def reference(name: str):
    """Import one of the reference's modules, naming a missing compiled extension as the series does.

    Every dumper under this directory reads the reference's own arithmetic, so every one of them
    needs an environment where the reference's extensions are built - the run host, not every
    sandbox. A caller who runs this where they are absent gets the first line of that failure rather
    than a traceback from three modules deep.
    """
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError as error:
        print(f"IMPORT FAILED: {type(error).__name__}: {error}")
        print("  the reference's compiled extensions are not built in this environment")
        raise SystemExit(1) from error


def dump(out: Path, name: str, values) -> None:
    """Write one tensor of the `denoiser` block, in both containers the two comparators read.

    The **relative path is the whole matching rule**: `tools/diagnosis/compare.py` keys each tree by
    `path.relative_to(root)` and pairs by that key, so a tensor written flat can never match a
    runner's `<block>/<name>.npy`. This writes `<out>/denoiser/<name>` for exactly that reason, and
    the names are the protocol's (`notes/h3/diagnosis.md` 2), not this file's.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dump_tables import dump as write

    block = out / "denoiser"
    block.mkdir(parents=True, exist_ok=True)
    write(block, name, values)


def canvas_of(size: dict):
    """The case's canvas as the reference's image configuration."""
    image = reference("uniserve.media.image")

    return image.Config(height=size["height"], width=size["width"])


def transformer_config(checkpoint: Path) -> dict:
    """The checkpoint's transformer config, read as the file it is.

    Read as JSON rather than through the reference's config module on purpose: the module resolves
    the whole checkpoint and imports the compiled extensions, and this half of the dumper is meant to
    run where those are absent. The keys are the checkpoint's own (`in_channels`,
    `audio_in_channels`), which are the names the reference's config maps to the latent widths.
    """
    path = checkpoint / "transformer" / "config.json"
    if not path.is_file():
        raise SystemExit(f"{path} is absent; point --checkpoint at the checkpoint's root")
    return json.loads(path.read_text())


def plan_noise(case: dict, config: dict) -> dict:
    """The two native draws a request's stream holds, and the order they are drawn in."""
    packing = reference("uniserve.models.minimax_h3.packing")

    canvas = canvas_of(case["canvas"])
    height, width = packing.latent_raster(canvas)
    frames = case["num_frames"]
    latent_frames = packing.video_latent_frames(frames)
    audio_frames = packing.audio_latent_frames(frames)
    return {
        "video": (
            1,
            int(config["in_channels"]),
            int(latent_frames),
            int(height),
            int(width),
        ),
        # The reference's `noise_shape("audio")` is the packed shape: channel-major timeline rows.
        "audio": (1, int(2 * audio_frames), int(config["audio_in_channels"])),
        "order": ("video", "audio"),
        "seed": int(case.get("seed", 0)),
        "canvas": (int(canvas.height), int(canvas.width)),
        "frames": int(frames),
    }


def run_noise(case: dict, config: dict, out: Path) -> None:
    """Draw the request's noise exactly as the reference draws it, and write both tensors."""
    torch = reference("torch")
    noise_module = reference("uniserve.diffusion.noise")

    plan = plan_noise(case, config)
    print(f"canvas  {plan['canvas'][0]}x{plan['canvas'][1]} -> latent raster, {plan['frames']} frames")
    print(f"seed    {plan['seed']}, one stream, drawn in the order {plan['order']}")
    tensors = {}
    for name in plan["order"]:
        tensors[name] = torch.empty(plan[name], dtype=torch.float32, device="cpu")
        print(f"plan    {name:<6} {tuple(tensors[name].shape)} float32 on the host")
    noise_module.normal_noise(
        (plan["seed"],), tuple(tensors[name] for name in plan["order"])
    )
    for name, label in (("video", "noise_video"), ("audio", "noise_audio")):
        dump(out, label, tensors[name])
        print(f"wrote   {out / 'denoiser' / (label + '.h3t')}")


def plan_rows(case: dict) -> dict:
    """The row order and the two attention boundaries of a dense layout.

    `packed_rows` is `[rows, 4]`: each row's modulation tag followed by its three rotary
    coordinates - the columns the protocol names, assembled from the reference's own `token_tags`
    and `position_ids` rather than re-derived here. `cu_seqlens` is `[2, 2]`: one row per boundary,
    `{0, padded}` for the queries and `{0, used}` for the keys, because a single attention run is
    named by its two ends and not by every row.
    """
    models = reference("uniserve.models")
    packing = reference("uniserve.models.minimax_h3.packing")

    canvas = canvas_of(case["canvas"])
    conditions = tuple(
        _condition(condition, models.Condition, models.ConditionRole)
        for condition in case.get("conditions", ())
    )
    attention = case["attention"]
    if attention["kind"] != "dense":
        raise SystemExit(
            "the segment packing's rows are per tile and dump_tables.py already writes them under "
            "`segment.`; this probe is the dense pair the protocol's table names"
        )
    plan = packing.dense_packing(
        num_frames=case["num_frames"],
        canvas=canvas,
        text_rows=case["text_capacity"],
        condition_rows=case["condition_capacity"],
        token_multiple=case["token_multiple"],
    )
    tables = packing.dense_tables(
        plan,
        num_frames=case["num_frames"],
        canvas=canvas,
        num_text_tokens=case["num_text_tokens"],
        segments=packing.condition_segments(conditions, canvas),
        vision_spans=tuple(tuple(span) for span in case.get("vision_spans", ())),
    )
    positions = tables.position_ids.to("cpu").numpy().astype(np.float32)
    tags = tables.token_tags.to("cpu").numpy().astype(np.float32)
    rows = np.concatenate((tags[:, None], positions), axis=1)
    return {
        "packed_rows": np.ascontiguousarray(rows),
        "cu_seqlens": np.asarray(
            [[0, int(plan.padded_tokens)], [0, int(tables.used)]], dtype=np.int32
        ),
        "padded_tokens": int(plan.padded_tokens),
        "used": int(tables.used),
    }


def _condition(case: dict, condition, role):
    """One condition of the request, from the case's flat description."""
    video = reference("uniserve.media.video")

    roles = {
        "first_frame": role.FIRST_FRAME,
        "last_frame": role.LAST_FRAME,
        "reference": role.REFERENCE,
    }
    pixels = case.get("canvas")
    return condition(
        role=roles[case["role"]],
        video=None
        if pixels is None
        else video.Config(frame=canvas_of(pixels), num_frames=case.get("frames", 1)),
        audio_samples=case.get("audio_samples", 0),
    )


def run_rows(case: dict, out: Path) -> None:
    """Write the dense layout's row order and boundaries."""
    plan = plan_rows(case)
    print(f"layout  {plan['padded_tokens']} packed rows, {plan['used']} of them visible")
    for name in ("packed_rows", "cu_seqlens"):
        dump(out, name, plan[name])
        print(f"shape   {name} {plan[name].shape} {plan[name].dtype}")
        print(f"wrote   {out / 'denoiser' / (name + '.h3t')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print the probe catalogue and exit")
    parser.add_argument("--probe", choices=sorted(CATALOGUE), help="the probe to run")
    parser.add_argument("--case", type=Path, help="the case description dump_tables.py also reads")
    parser.add_argument("--checkpoint", type=Path, help="the MiniMax-H3 checkpoint directory")
    parser.add_argument("--out", type=Path, help="where to write the dump")
    options = parser.parse_args(argv)

    if options.list:
        for name, (writes, description) in CATALOGUE.items():
            print(f"{name}\t{', '.join(writes)}\t{description}")
        return 0
    if not (options.probe and options.case and options.out):
        parser.error("--probe, --case and --out are required unless --list")

    case = json.loads(options.case.read_text())
    writes = CATALOGUE[options.probe][0]
    print(f"probe   {options.probe} -> {', '.join(writes)}")
    print(f"case    {options.case} ({case['num_frames']} frames)")

    if options.probe not in HOST_PROBES:
        # The plan is printed before the refusal, so a caller sees what the tensor would be and why
        # it is not produced here.
        print(
            f"plan    {options.probe} is the runner's: it needs "
            "`uniserve.runner.denoiser.DenoiserRunner` (a CUDA stream, a loaded model and the "
            "bindings' scratch), which the host half of this dumper cannot construct"
        )
        print(
            "refused the tensors are real and named in notes/h3/diagnosis.md 2; producing them "
            "belongs with the runner's own diagnostics hook, not with a weight-free dumper"
        )
        return 1
    if options.probe == "noise":
        if options.checkpoint is None:
            parser.error("--probe noise needs --checkpoint for the latent channel counts")
        config = transformer_config(options.checkpoint)
        print(f"config  {options.checkpoint / 'transformer' / 'config.json'}")
        options.out.mkdir(parents=True, exist_ok=True)
        run_noise(case, config, options.out)
        return 0
    options.out.mkdir(parents=True, exist_ok=True)
    run_rows(case, options.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
