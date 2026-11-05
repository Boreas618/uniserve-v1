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
a matrix product. The `stack_out` probe is the reference producer of the first device run's
compare step: it reads the candidate's `stack_in.h3t` and `case.json`, runs the reference's own
stack of the case's layer count at step 0 over those bytes, and writes `stack_out.h3t`, so both
sides of the comparison compute on one input:

    .venv/bin/python crates/models/minimax-h3/tools/dump_denoiser.py --probe stack_out \
        --stack <candidate out>/denoiser --checkpoint <checkpoint> --out <reference out>

It needs the checkpoint's weights and the reference's compiled extensions - the run host - but not
the runner, which is why it can be produced at all. The `state` and `velocity` probes need the
reference's own runner
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
    "stack_out": (
        ("stack_out",),
        "run host: the reference's own stack on the candidate's input, one step",
    ),
    "state": (
        ("latents_video_e", "latents_audio_e"),
        "run host: the samples on entry to evaluation `e`, which the runner owns",
    ),
    "velocity": (
        ("velocity_video_e", "velocity_audio_e"),
        "run host: the network's prediction at evaluation `e`, which the runner owns",
    ),
    "trajectory": (
        ("latents_video_e0..eN", "latents_audio_e0..eN", "latents_*_final"),
        "run host: the reference's own trajectory - its draw, then one evaluation per step",
    ),
}

# The probe the sampler stage pairs on. It reads the candidate's own prefix source and case, runs the
# reference's whole denoiser for the case's prefix of the checkpoint's schedule, and writes each
# evaluation's samples under the candidate's own names, so a page is about arithmetic rather than
# about who drew the noise.
TRAJECTORY = "trajectory"

# The schedule kinds a checkpoint can ship, in the reference's class names and the case's words. A
# kind outside this table is refused rather than compared loosely: a FastH3 contract read as the base
# grid is the one mistake a trajectory page cannot show.
# The reference's class names against the model's own three words (`ScheduleGrid::kind` in
# `config/denoiser.rs`): `uniform`, `dmd`, `blocks`. The mapping is many-to-one because both
# block-derived grids are `blocks` - they differ in how the nodes are derived, not in the grid the
# case names, and the case carries the model's word rather than a second vocabulary of its own.
SCHEDULE_KINDS = {"UniformGrid": "uniform", "BlockGrid": "blocks", "PddGrid": "blocks"}

# The probes this script can produce without the reference's runner, a device or weights.
HOST_PROBES = frozenset({"noise", "rows"})

# The probes that need the reference's own modules and the checkpoint's weights, but not its
# runner: they run on the run host, and a host without the compiled extensions refuses them with
# the import line rather than a traceback (`reference`).
WEIGHED_PROBES = frozenset({"stack_out", TRAJECTORY})


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
        print(
            "  the reference's compiled extensions are not built in this environment"
        )
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


# The words a case uses for an element type, and the torch types they mean. A case's `dtype` is the
# element type the deployment computes in - the descriptor keys' own word (`dtype_f32`) - and it is
# not `sample_dtype`, which is FP32 whatever the deployment states.
CASE_DTYPES = {"f32": "float32", "f16": "float16", "bf16": "bfloat16"}


def case_dtype(case: dict, checkpoint: Path):
    """The element type the reference must build its model in for this case.

    The case's `dtype` is the deployment's and is the authority for the pair: a reference built at
    another type would compare two precisions, and the residual would read as arithmetic. A case
    that predates the field gets the checkpoint's own answer from the transformer's shards, printed,
    so the difference between a stated type and an observed one is never silent.
    """
    torch = reference("torch")
    stated = case.get("dtype")
    if stated is None:
        from dump_encoder import shard_dtype

        own = shard_dtype(checkpoint / "transformer")
        print(
            f"dtype   {str(own).removeprefix('torch.')}, the checkpoint's own (the case states"
            f" none)"
        )
        return own
    name = CASE_DTYPES.get(stated)
    if name is None:
        raise SystemExit(
            f"the case states the element type {stated!r}; this probe knows f32, f16 and bf16"
        )
    print(f"dtype   {stated}, the case's (the deployment's)")
    return getattr(torch, name)


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
        raise SystemExit(
            f"{path} is absent; point --checkpoint at the checkpoint's root"
        )
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
    print(
        f"canvas  {plan['canvas'][0]}x{plan['canvas'][1]} -> latent raster, {plan['frames']} frames"
    )
    print(
        f"seed    {plan['seed']}, one stream, drawn in the order {plan['order']}"
    )
    tensors = {}
    for name in plan["order"]:
        tensors[name] = torch.empty(
            plan[name], dtype=torch.float32, device="cpu"
        )
        print(
            f"plan    {name:<6} {tuple(tensors[name].shape)} float32 on the host"
        )
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
        vision_spans=tuple(
            tuple(span) for span in case.get("vision_spans", ())
        ),
    )
    positions = tables.position_ids.to("cpu").numpy().astype(np.float32)
    tags = tables.token_tags.to("cpu").numpy().astype(np.float32)
    rows = np.concatenate((tags[:, None], positions), axis=1)
    return {
        "packed_rows": np.ascontiguousarray(rows),
        "cu_seqlens": np.asarray(
            [[0, int(plan.padded_tokens)], [0, int(tables.used)]],
            dtype=np.int32,
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
        else video.Config(
            frame=canvas_of(pixels), num_frames=case.get("frames", 1)
        ),
        audio_samples=case.get("audio_samples", 0),
    )


def run_rows(case: dict, out: Path) -> None:
    """Write the dense layout's row order and boundaries."""
    plan = plan_rows(case)
    print(
        f"layout  {plan['padded_tokens']} packed rows, {plan['used']} of them visible"
    )
    for name in ("packed_rows", "cu_seqlens"):
        dump(out, name, plan[name])
        print(f"shape   {name} {plan[name].shape} {plan[name].dtype}")
        print(f"wrote   {out / 'denoiser' / (name + '.h3t')}")


def stack_packing(case: dict):
    """The dense layout the candidate's case describes, in the reference's own terms.

    The case describes a `t2va` request: a conditioned one carries its condition rows in
    `padded_tokens` without naming their tags or extents, so the layout here would be a different
    request's and the comparison would measure the difference rather than the arithmetic. Such a
    case is refused by task, not silently packed as if it had no conditions.

    The case carries the *padded* row count rather than the alignment that produced it, so the
    layout is rebuilt from the reference's own row functions and the padding the candidate used is
    checked against them: a case whose padded count cannot hold its content rows names that
    arithmetic instead of comparing two different requests.
    """
    packing = reference("uniserve.models.minimax_h3.packing")

    task = case.get("task")
    if task != "t2va":
        raise SystemExit(
            f"the case's task is {task!r}, and this probe rebuilds a dense layout without "
            "conditions; a conditioned request's rows are not described by the case"
        )
    canvas = canvas_of({"height": case["height"], "width": case["width"]})
    frames = int(case["frames"])
    text_rows = int(case["text_tokens"])
    tokens = int(case["padded_tokens"])
    latent_height, latent_width = packing.latent_raster(canvas)
    video_rows = (
        packing.video_latent_frames(frames)
        * (latent_height // 2)
        * (latent_width // 2)
    )
    audio_rows = packing.AUDIO_CHANNELS * packing.audio_latent_frames(frames)
    content = video_rows + audio_rows + text_rows
    if content > tokens:
        raise SystemExit(
            f"the case's {tokens} padded rows cannot hold its {content} content rows "
            f"({video_rows} video, {audio_rows} audio, {text_rows} text)"
        )
    plan = packing.DensePacking(
        video_rows=video_rows,
        audio_rows=audio_rows,
        text_rows=text_rows,
        condition_rows=0,
        padded_tokens=tokens,
    )
    return packing, canvas, plan


def stack_tables(case: dict, config) -> dict:
    """The row tables one stack evaluation reads, from the reference's own packing.

    The candidate's input file carries its activations, not its tables: the rotary factors and the
    per-row modulation indices are rebuilt here from the reference's arithmetic. A table that
    disagrees makes the comparison fail on a row's coordinates, which is a different diagnosis from
    a wrong matrix product and is named as such by the comparator's per-tensor report.
    """
    torch = reference("torch")
    rotary_module = reference("uniserve.nn")

    packing, canvas, plan = stack_packing(case)
    tables = packing.dense_tables(
        plan,
        num_frames=int(case["frames"]),
        canvas=canvas,
        num_text_tokens=int(case["text_tokens"]),
    )
    rotary = rotary_module.RotaryEmbedding(
        2 * config.rope_frequency_dim, theta=config.rope_theta
    )
    cosine, sine = rotary(
        tables.position_ids,
        dtype=torch.float32,
        sequence_length=plan.padded_tokens,
    )
    print(
        f"layout  {plan.padded_tokens} packed rows: {plan.video_rows} video, "
        f"{plan.audio_rows} audio, {plan.text_rows} text, {tables.used} attended"
    )
    return {
        "plan": plan,
        "tables": {
            "cos": cosine.flatten(1),
            "sin": sine.flatten(1),
            # Row `3 * group + tag` of each step's modulation products.
            "modulation_indices": tables.groups * 3 + tables.token_tags,
        },
    }


def stack_source(checkpoint: Path):
    """The checkpoint source of the DiT partition the stack runs."""
    from uniserve.loading import checkpoint as checkpoint_module

    directory = checkpoint / "transformer"
    files = tuple(sorted(directory.glob("*.safetensors")))
    if not files:
        index = directory / "diffusion_pytorch_model.safetensors.index.json"
        hint = (
            f" ({index.name} lists them, so the shards are not fetched)"
            if index.is_file()
            else ""
        )
        raise SystemExit(
            f"{directory} holds no weight file{hint}; this probe runs the reference's own "
            f"arithmetic and needs the weights the serving loader fetches"
        )
    return checkpoint_module.Source("transformer", root=directory, files=files)


def stack_model(case: dict, checkpoint: Path):
    """The reference's transformer, at the case's layer count, loaded through its own loader."""
    import dataclasses

    torch = reference("torch")
    h3_config = reference("uniserve.models.minimax_h3.config")
    denoiser_module = reference("uniserve.models.minimax_h3.denoiser")
    transformer_module = reference("uniserve.models.minimax_h3.transformer")
    weights_module = reference("uniserve.models.minimax_h3.weights")
    from uniserve.loading import loader
    from uniserve.loading import weights as weight_options

    sidecars = h3_config.read_config(checkpoint, io=None, sources=())
    denoiser = sidecars.denoisers["transformer"]
    layers = int(case["layers"])
    config = dataclasses.replace(
        denoiser,
        transformer=dataclasses.replace(
            denoiser.transformer, num_hidden_layers=layers
        ),
    )
    host = denoiser_module.schedules(config, device="cpu")
    _, entries, table = denoiser_module.adaln_timesteps(config, host)
    model = transformer_module.Transformer(
        config.transformer,
        attention=config.attention,
        entries=entries.numel(),
        steps=table.shape[0],
        groups=table.shape[1],
    )
    mapping = weights_module.transformer_component(model, config, "transformer")
    source = stack_source(checkpoint)
    print(
        f"load    {source.name} from {source.root} ({len(source.files)} file(s))"
    )
    reports = loader.load_weights(
        model,
        (source,),
        mapping=lambda _root: (mapping,),
        device="cpu",
        weights=weight_options.Config(dtype=case_dtype(case, checkpoint)),
    )
    for report in reports:
        print(
            f"loaded  {len(report.loaded)} parameter(s); missing {len(report.missing)}, "
            f"unexpected {len(report.unexpected)}, incomplete {len(report.incomplete)}"
        )
        for name in sorted(report.missing)[:8]:
            print(f"missing  {name}")
    return model, config.transformer, layers


def run_stack_out(stack: Path, checkpoint: Path, out: Path) -> int:
    """Run the reference's stack on the candidate's input and write its output beside it.

    The input is the whole contract: both implementations run the same bytes through the same
    layer count at the same step, so the comparison is about arithmetic rather than about who drew
    the noise or which packing produced the rows.

    Four things it checks before it runs, each refused by name: the input's shape against the
    case's `padded_tokens` and `hidden`, the checkpoint's hidden size against the case's, the
    case's `step` against the evaluations the checkpoint's schedule holds, and the case's `layers`
    against the layers it then runs (it runs the case's, so a case that says one layer and a
    checkpoint that holds fifty runs one layer of that fifty). Two things it does not
    check, because the case does not carry them: the alignment the candidate padded its row count
    to (only that the padding can hold the content rows), and a conditioned request's condition
    rows - a case whose task is not `t2va` is refused rather than packed as if it had none.
    """
    torch = reference("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        # The container lives with the layout dumper, which imports the reference's media modules;
        # on a host without the compiled extensions that import is the first thing to fail, so it
        # is reported as such rather than as a traceback from three modules deep.
        from dump_tables import load as read
    except ImportError as error:
        print(f"IMPORT FAILED: {type(error).__name__}: {error}")
        print(
            "  the reference's compiled extensions are not built in this environment"
        )
        return 1

    case_path = stack / "case.json"
    input_path = stack / "stack_in.h3t"
    for path in (case_path, input_path):
        if not path.is_file():
            raise SystemExit(
                f"{path} is absent; --stack names the candidate's `denoiser` directory, which "
                f"`examples/dit_layer.rs --stage block` writes"
            )
    case = json.loads(case_path.read_text())
    hidden = read(input_path)
    rows, width = int(case["padded_tokens"]), int(case["hidden"])
    if tuple(hidden.shape) != (rows, width):
        raise SystemExit(
            f"{input_path} holds {tuple(hidden.shape)}, and its case declares {(rows, width)}"
        )
    print(f"input   {input_path} {tuple(hidden.shape)} {hidden.dtype}")
    model, config, layers = stack_model(case, checkpoint)
    if int(config.hidden_size) != width:
        raise SystemExit(
            f"the checkpoint's hidden size is {config.hidden_size}, and the case declares {width}"
        )
    built = stack_tables(case, config)
    inputs_module = reference("uniserve.models.minimax_h3.inputs")
    attention_module = reference("uniserve.nn.attention")
    plan = built["plan"]
    lengths = attention_module.SequenceLengths(
        torch.tensor([plan.padded_tokens], dtype=torch.int32),
        torch.tensor([0, plan.padded_tokens], dtype=torch.int32),
        (plan.padded_tokens,),
    )
    visible = attention_module.VisibleInput(
        lengths,
        lengths,
        # Every query row sees the request's used keys, the padding excluded.
        torch.full((1, 1), plan.used, dtype=torch.int32),
        None,
        prefix_bounds=True,
        fully_visible=False,
    )
    empty = torch.zeros(0, dtype=torch.int64)
    inputs = inputs_module.SequenceInput(
        slice(0, plan.padded_tokens),
        model.mesh.get_group("tp"),
        visible,
        empty,
        empty,
    )
    # The case's step is the evaluation both sides run. It is read rather than assumed, so a stage
    # that ever runs another evaluation meets the same one here instead of a modulation difference
    # that would look exactly like an arithmetic one; an index the schedule does not reach is
    # refused, because the cache does not check it on the host.
    steps = int(model.adaln_cache.groups.shape[0])
    step_index = int(case.get("step", 0))
    if not 0 <= step_index < steps:
        raise SystemExit(
            f"the case's step is {step_index}, and the checkpoint's schedule evaluates "
            f"{steps} step(s): 0 to {steps - 1}"
        )
    step = torch.tensor([step_index], dtype=torch.int64)
    print(f"step    {step_index} of {steps} in the checkpoint's schedule")
    # The mask is the argument a comparison can disagree on silently: a page whose rows past the
    # message differ everywhere, with nothing arithmetic behind it, is this number on one side and
    # another on the other. The harness states the same pair through the deployment's own tables, so
    # both are on the record and a disagreement is one glance rather than an inference.
    print(
        f"mask    every query row sees {plan.used} of {plan.padded_tokens} keys"
        f" (cu_seq_lens_q = [0, {plan.padded_tokens}], cu_seq_lens_k = [0, {plan.used}])"
    )
    modulation = model.adaln_cache(step)
    chunks = ((inputs.token_slice, hidden),)
    print(f"run     {layers} layer(s) at step {step_index}, {rows} rows of {width}")
    with torch.inference_mode():
        for index, layer in enumerate(model.layers.values()):
            chunks = layer.forward_chunks(
                chunks, modulation[index], inputs, tables=built["tables"]
            )
        values = tuple(value for _, value in chunks)
        output = values[0] if len(values) == 1 else torch.cat(values)
    print(f"got     {tuple(output.shape)} {output.dtype}")
    dump(out, "stack_out", output)
    print(f"wrote   {out / 'denoiser' / 'stack_out.h3t'}")
    # The comparator recurses and keys each tensor by its path relative to the root it was given,
    # so the two outer directories are passed: every name then keeps its block prefix, and a file
    # present on one side only is listed rather than counted as a difference.
    candidate = stack.parent if stack.name == "denoiser" else stack
    print(
        f"compare cargo run -p minimax-h3 --example compare_blocks -- --reference {out} "
        f"--candidate {candidate} --out {out / 'report.md'}"
    )
    return 0


def read_tensor(path: Path):
    """Read one H3D1 tensor: the container this directory writes and the Rust comparator reads."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dump_tables import load

    try:
        return load(path)
    except ValueError as error:
        raise SystemExit(str(error)) from error


def trajectory_case(case: dict) -> dict:
    """The case fields this probe reads, refused by name when the case omits one.

    The case is the candidate harness's own statement of the request, so a field read under another
    name would be a second convention for the same number: every field this probe uses is required,
    and a missing one is reported with its name rather than defaulted here.
    """
    required = (
        "task",
        "width",
        "height",
        "frames",
        "text_tokens",
        "text_rows",
        "condition_rows",
        "layers",
        "hidden",
        "seed",
        "evaluations",
        "schedule",
    )
    missing = [name for name in required if name not in case]
    if missing:
        raise SystemExit(
            f"the case omits {', '.join(missing)}; a sampler case carries "
            f"{', '.join(required)}"
        )
    if case["task"] != "t2va":
        raise SystemExit(
            f"the case's task is {case['task']!r}, and a conditioned request's rows are not"
            f" described by this case: this probe runs a `t2va` sampler"
        )
    return case


def trajectory_denoiser(case: dict, checkpoint: Path, device: str):
    """The reference's whole denoiser at the case's layer count, on the reference's own load path.

    The layer count is the case's because a trajectory over fifty blocks is a device hour; the
    checkpoint's own transformer config is narrowed the way the stage-1 producer narrows it, so a
    case that says two layers runs two of the checkpoint's fifty and says so.
    """
    import dataclasses

    torch = reference("torch")
    h3_config = reference("uniserve.models.minimax_h3.config")
    denoiser_module = reference("uniserve.models.minimax_h3.denoiser")
    weights_module = reference("uniserve.models.minimax_h3.weights")
    from uniserve.loading import loader
    from uniserve.loading import weights as weight_options

    sidecars = h3_config.read_config(checkpoint, io=None, sources=())
    config = sidecars.denoisers["transformer"]
    layers = int(case["layers"])
    config = dataclasses.replace(
        config,
        transformer=dataclasses.replace(
            config.transformer, num_hidden_layers=layers
        ),
    )
    model = denoiser_module.Denoiser(config)
    source = stack_source(checkpoint)
    print(f"load    {source.name} from {source.root} ({len(source.files)} file(s))")
    reports = loader.load_weights(
        model,
        (source,),
        mapping=lambda _root: tuple(
            weights_module._denoiser_components("transformer", model)
        ),
        device=device,
        weights=weight_options.Config(dtype=case_dtype(case, checkpoint)),
    )
    for report in reports:
        print(
            f"loaded  {len(report.loaded)} parameter(s); missing {len(report.missing)}, "
            f"unexpected {len(report.unexpected)}, incomplete {len(report.incomplete)}"
        )
        for name in sorted(report.missing)[:8]:
            print(f"missing  {name}")
    return model, config


def allocate(configs, device: str) -> dict:
    """Allocate one dict of buffers from the model's own declarations.

    Each buffer's `host` flag is the model's own statement of where it lives: the native draws and
    the sample views are host-side and refuse anything else, and the rest are on the device the
    caller names. Reading the flag rather than deciding here is what keeps the probe from owning a
    second copy of the model's device rule.
    """
    torch = reference("torch")

    buffers = {}
    for name, config in configs.items():
        host = bool(getattr(config, "host", False))
        buffers[name] = torch.zeros(
            tuple(config.shape),
            dtype=config.dtype,
            device="cpu" if host else device,
        )
    return buffers


def run_trajectory(
    stack: Path, checkpoint: Path, out: Path, device: str
) -> int:
    """Run the reference's own trajectory on the candidate's own prefix source.

    Every number but the draw is the reference's arithmetic: `Denoiser.forward` is one whole
    evaluation with the solver step inside, so this adds the loop and the files and nothing else.
    `e0` is the draw and `eN` the final sample, so the noise is on the page as a tensor and a wrong
    draw cannot hide inside a plausible trajectory; `latents_*_final.h3t` is a copy of `eN` so a
    comparator pairs the end without knowing how many evaluations ran.

    What it does not prove: the conditioner (the prefix source is the harness's file, stated as
    synthetic in the case), the text encoder behind it, the decoder after it, and any canvas or
    frame count but the case's.
    """
    # The case and its file are read before anything of the reference is imported, so a wrong path or
    # a case from another stage refuses by name on a host where the model cannot be built at all.
    case_path, table_path = stack / "case.json", stack / "text_features.h3t"
    for path in (case_path, table_path):
        if not path.is_file():
            raise SystemExit(
                f"{path} is absent; --stack names the candidate's `denoiser` directory, and the"
                f" sampler stage writes `case.json` and `text_features.h3t` into it"
            )
    case = trajectory_case(json.loads(case_path.read_text()))
    table = read_tensor(table_path)
    print(f"input   {table_path} {tuple(table.shape)} {table.dtype}")

    torch = reference("torch")
    noise_module = reference("uniserve.diffusion.noise")
    packing = reference("uniserve.models.minimax_h3.packing")

    model, config = trajectory_denoiser(case, checkpoint, device)
    canvas = canvas_of({"height": case["height"], "width": case["width"]})
    size = model.make_size(
        int(case["frames"]), int(case["text_tokens"]), canvas=canvas
    )
    # The layout is the **case's**, not the smallest one that holds the request: the device runs the
    # layout its runner reserves, and the row count changes what every attention reads even though
    # the latent files it writes do not depend on it - so a reference at `layout_size(size)` would
    # produce a page of the same shape as the device's and a different trajectory, which reads as an
    # arithmetic difference whose cause is an arrangement.
    denoiser_module = reference("uniserve.models.minimax_h3.denoiser")
    layout = denoiser_module.DenoiserSize(
        int(case["frames"]),
        canvas,
        int(case["text_rows"]),
        int(case["condition_rows"]),
    )
    if not model.holds(layout, size):
        raise SystemExit(
            f"the case's layout {layout.num_text_tokens} text +"
            f" {layout.condition_rows} condition rows does not hold a request of"
            f" {size.num_text_tokens} text and {size.condition_rows} condition rows at"
            f" {layout.num_frames} frames and {layout.canvas}; the layout must be whole tiles"
            f" and at least the request's own counts"
        )
    steps = int(model.num_steps)
    evaluations = int(case["evaluations"])
    # The schedule is the checkpoint's and the case may only shorten it: a longer one would be a
    # different schedule, and a shorter one is how a stage's first page costs a minute.
    if evaluations > steps:
        raise SystemExit(
            f"the case asks {evaluations} evaluations and the checkpoint's schedule holds"
            f" {steps}; a case may shorten the schedule and never lengthen it"
        )
    kind = SCHEDULE_KINDS.get(type(config.schedule).__name__)
    if kind is None:
        raise SystemExit(
            f"the checkpoint's schedule {type(config.schedule).__name__} has no name in this"
            f" probe; the ones it knows are {sorted(SCHEDULE_KINDS)}"
        )
    if case["schedule"] != kind:
        raise SystemExit(
            f"the case's schedule is {case['schedule']!r} and the checkpoint's is {kind!r};"
            f" a page cannot show that one was read as the other"
        )
    packed = model._dense_packing(layout)
    rows_stated = {
        "frames": int(layout.num_frames),
        "canvas": [int(layout.canvas.height), int(layout.canvas.width)],
        "text_rows": int(layout.num_text_tokens),
        "condition_rows": int(layout.condition_rows),
        "video_rows": int(packed.video_rows),
        "audio_rows": int(packed.audio_rows),
        "packed_rows": int(packed.padded_tokens),
        # The grid, as well as the layout: two checkpoints' trajectories agree on every row count
        # above whenever they ran the same layout, so a pair from a FastH3 student and from the base
        # model would compare a block grid against a uniform one and print a table of arithmetic
        # about nothing. The word is the checkpoint's own, which this function has already refused a
        # case against, and the two step counts are the ones the print line below states.
        "schedule": kind,
        "evaluations": int(evaluations),
        "num_steps": int(steps),
    }
    print(
        f"layout  {rows_stated['packed_rows']} packed rows ="
        f" {rows_stated['video_rows']} video + {rows_stated['audio_rows']} audio +"
        f" {rows_stated['text_rows']} text + {rows_stated['condition_rows']} condition"
        f" ({layout.num_frames} frames, {evaluations} of {steps} evaluations)"
    )
    # Written beside the trajectory so the two sides can be paired by their row counts: the latent
    # files do not depend on the packed row count, so a page from two different layouts has the same
    # shape and would read as an arithmetic difference with an arrangement for a cause.
    stated_layout = out / "denoiser" / "layout.json"
    stated_layout.parent.mkdir(parents=True, exist_ok=True)
    stated_layout.write_text(json.dumps(rows_stated, indent=2) + "\n")
    print(f"stated  {stated_layout}: {json.dumps(rows_stated)}")
    width = int(config.transformer.hidden_size)
    if int(case["hidden"]) != width:
        raise SystemExit(
            f"the case's hidden size is {case['hidden']} and the checkpoint's is {width}"
        )
    # The prefix source is the text rows, the projected condition rows and one zero row the
    # generated and padding rows gather: the case's table carries the text rows, and the zero row
    # is the layout's, not the harness's.
    rows = model.text_condition_rows(layout)
    if tuple(table.shape) != (layout.num_text_tokens, width):
        raise SystemExit(
            f"the prefix source is {tuple(table.shape)} and the layout's text segment is"
            f" {(layout.num_text_tokens, width)}"
        )
    # The case states the prefix's height too, and it is a second computation of one number: the
    # harness's `text_rows + condition_rows + 1` against the layout's. They agree only at the layout
    # the device runs, so a case written at another layout is refused rather than compared.
    if "prefix_rows" in case and int(case["prefix_rows"]) != rows:
        raise SystemExit(
            f"the case states a prefix of {case['prefix_rows']} rows and the layout's is {rows}"
            f" ({layout.num_text_tokens} text + {layout.condition_rows} condition + 1)"
        )
    prefix = torch.zeros((rows, width), dtype=table.dtype, device=device)
    prefix[: layout.num_text_tokens] = table.to(device=device)
    print(f"prefix  {rows} rows: {layout.num_text_tokens} text, one zero row")

    state = allocate(model.state_buffers(layout), device)
    constants = allocate(model.constant_buffers(layout), device)
    workspace = allocate(model.workspace_buffers(layout), device)
    tables = {
        name: value
        for name, value in state.items()
        if name not in model.modalities
    }
    model.prepare_constants(layout, out=constants)
    model.prepare_state((size,), layouts=(layout,), out=tables)
    draw = {}
    for name in model.modalities:
        draw[name] = torch.empty(
            model.noise_shape(name, size), dtype=torch.float32, device="cpu"
        )
        print(f"draw    {name:<6} {tuple(draw[name].shape)} float32 on the host")
    noise_module.normal_noise((int(case["seed"]),), out=tuple(draw[name] for name in model.modalities))
    model.prepare_latents(
        (size,), noise=draw, state=state, constants=constants, workspace=workspace
    )
    print(f"seed    {case['seed']}, drawn in the order {tuple(model.modalities)}")

    latent_height, latent_width = packing.latent_raster(canvas)
    video_frames = packing.video_latent_frames(int(case["frames"]))
    channels = int(config.transformer.video_channels)
    written = []

    def write(index: str) -> None:
        """Write one evaluation's sample pair under the candidate's own names."""
        video = packing.unpatchify_video(
            state["video"],
            frames=video_frames,
            height=latent_height,
            width=latent_width,
            channels=channels,
        )[0]
        for modality, value in (("video", video), ("audio", state["audio"][0])):
            name = f"denoiser/latents_{modality}_e{index}"
            block = out / "denoiser"
            block.mkdir(parents=True, exist_ok=True)
            dump(out, name, value)
            written.append(name)

    for index in range(evaluations):
        write(str(index))
        step = torch.tensor([index], dtype=torch.int64, device=device)
        latents = {
            name: (trajectory_latent_input(torch, state[name][0], step),)
            for name in model.modalities
        }
        inputs = model.bind_inputs(
            latents=latents,
            sizes=(size,),
            step=step,
            text_features=(prefix,),
        )
        model.forward(inputs, state=state, constants=constants, workspace=workspace)
        print(f"step    {index} evaluated, sample written as latents_*_e{index}")
    write(str(evaluations))
    for modality in model.modalities:
        source = out / "denoiser" / f"latents_{modality}_e{evaluations}.h3t"
        final = out / "denoiser" / f"latents_{modality}_final.h3t"
        final.write_bytes(source.read_bytes())
        written.append(f"denoiser/latents_{modality}_final")
    print(f"wrote   {len(written)} tensors: e0 the draw, e{evaluations} the final sample and its copy")
    print(
        "compare cargo run -p minimax-h3 --example compare_blocks --"
        f" --reference {out} --candidate {stack.parent if stack.name == 'denoiser' else stack}"
    )
    return 0


def trajectory_latent_input(torch, sample, step):
    """One modality's `LatentInput`: this rank's sample rows and the evaluation's step.

    The sample is the state's own view, so the solver step writes through it and the next
    evaluation reads what the last one left; H3's modulation is gathered from `step`, which is why
    the two carry the same value.
    """
    inputs_module = reference("uniserve.models.minimax_h3.inputs")

    return inputs_module.LatentInput(tensor=sample, timestep=step)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--list", action="store_true", help="print the probe catalogue and exit"
    )
    parser.add_argument(
        "--probe", choices=sorted(CATALOGUE), help="the probe to run"
    )
    parser.add_argument(
        "--case",
        type=Path,
        help="the case description dump_tables.py also reads",
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="the MiniMax-H3 checkpoint directory"
    )
    parser.add_argument("--out", type=Path, help="where to write the dump")
    parser.add_argument(
        "--device",
        default="cpu",
        help="the device the reference's model runs on (default: cpu)",
    )
    parser.add_argument(
        "--stack",
        type=Path,
        help=(
            "the candidate's `denoiser` dump directory: `stack_in.h3t` and `case.json`"
            " (`--probe stack_out`), or `text_features.h3t` and `case.json`"
            " (`--probe trajectory`)"
        ),
    )
    options = parser.parse_args(argv)

    if options.list:
        for name, (writes, description) in CATALOGUE.items():
            print(f"{name}\t{', '.join(writes)}\t{description}")
        return 0
    if not (options.probe and options.out):
        parser.error("--probe and --out are required unless --list")
    writes = CATALOGUE[options.probe][0]
    print(f"probe   {options.probe} -> {', '.join(writes)}")

    if options.probe == TRAJECTORY:
        if options.stack is None or options.checkpoint is None:
            parser.error(
                f"--probe {options.probe} needs --stack for the candidate's tree and "
                "--checkpoint for the weights it runs"
            )
        options.out.mkdir(parents=True, exist_ok=True)
        return run_trajectory(
            options.stack, options.checkpoint, options.out, options.device
        )

    if options.probe in WEIGHED_PROBES:
        if options.stack is None or options.checkpoint is None:
            parser.error(
                f"--probe {options.probe} needs --stack for the candidate's input and "
                "--checkpoint for the weights it runs"
            )
        options.out.mkdir(parents=True, exist_ok=True)
        return run_stack_out(options.stack, options.checkpoint, options.out)

    if options.case is None:
        parser.error(f"--probe {options.probe} needs --case")
    case = json.loads(options.case.read_text())
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
            parser.error(
                "--probe noise needs --checkpoint for the latent channel counts"
            )
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
