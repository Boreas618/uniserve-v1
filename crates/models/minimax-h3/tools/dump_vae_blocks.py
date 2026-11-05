#!/usr/bin/env python3
"""Dump one probe of a MiniMax-H3 VAE block from today's implementation.

The reference half of the block-by-block numeric comparison (`specs/worker.md` 10). The Rust half is the
trace hook that writes `$UNISERVE_DIAG_DIR/<block>/<name>.npy` for the same request and seed, and
`tools/diagnosis/compare.py` is the comparator that reads both trees and writes `report.json`
(`notes/h3/diagnosis.md`). This script writes the reference tree, one tensor per probe, under the name
the hook writes, so the two trees line up by name and not by position.

It runs on the reference's own weight path - `loading.checkpoint.Source` opened through
`loading.loader.load_weights` with the `weights.ModuleMapping` that `models/minimax_h3/weights.py`
builds for the component - so a discrepancy it reports is about our mathematics and not about a
different way of reading the checkpoint. It never loads a state dict by hand.

One probe per invocation, so a failure costs one tensor and not the batch:

    .venv/bin/python crates/models/minimax-h3/tools/dump_vae_blocks.py --list
    .venv/bin/python crates/models/minimax-h3/tools/dump_vae_blocks.py \\
        --checkpoint /opt/uv/models/MiniMax-H3 --probe video_decoder:decoder.decoder.layers.0.qkv \\
        --out artifacts/diagnosis/vae-blocks/ref

The plan is printed before anything is loaded, so a wrong module path or a missing weight shard is
visible in the first screen. A **stage** probe (`--probe decode_out`) is the other half: it has no
fixed input, and runs the reference's module on the input a candidate's stage wrote, so the
comparison is about the reference's arithmetic and not about who drew the clip's latents.

    .venv/bin/python crates/models/minimax-h3/tools/dump_vae_blocks.py --probe decode_out \\
        --stack artifacts/h3-stage2/video_decoder --checkpoint /opt/uv/models/MiniMax-H3 \\
        --out artifacts/h3-stage2-ref --device cuda A probe is a module path, relative to the component's root module, and the
value written is that module's output for a fixed input the options below make reproducible.

Run it from the repository root with the repository's environment and the repository importable
(`PYTHONPATH=.` unless the package is installed editable); it needs a machine where the reference's
compiled extensions are built, which is the run host and not every sandbox. The plan is printed before
the first import of the model, so a broken environment and a wrong probe path are different failures.

This is a **diagnosis, never an acceptance gate** (`specs/worker.md` 10).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn


# One fixed input per component: the probe's value is only comparable between runs if the input is.
# The video decoder reads one clip of latents - `clip_latents + overlap` frames of the request's
# latent raster, the shape its own `latent_shape` declares - and the audio decoder reads one stereo
# latent timeline of `latent_channels` channels.
VIDEO_LATENT_FRAMES = 7
VIDEO_LATENT_SIDE = 16
AUDIO_LATENT_FRAMES = 20
# The encoder reads pixels, so its fixed input is one tile of one clip: the
# checkpoint's 256-pixel tile, and the clip length its own config states.
TILE_SIDE = 256
# One clip's frames: the encoder reads a still or a clip, never a fraction.
ENCODER_FRAMES = 17
SEED = 0

# The compute dtypes a caller may load the reference's parameters in; the checkpoint's own is the
# default, because a comparison in another dtype measures the conversion and not the mathematics.
DTYPES = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
}

# The probe catalogue: component -> (module paths relative to that component's root, the arithmetic the
# output's shape must satisfy). Each path is a place a plausible-wrong implementation diverges, and
# together they localise a divergence to one layer rather than to the whole pipeline.
VIDEO_LAYERS = (0, 3)
CATALOGUE = {
    "video_decoder": (
        # The reference's `decoder.post_quant_conv` is deliberately **not** a probe: this implementation
        # folds the latent statistics, that one-by-one projection and the transformer's input projection
        # into one affine at load (design 10 declares the composition, and the fold's own test checks the
        # three steps against it). A probe of it could only ever be "not dumped by the Rust side", which
        # a report reads as a hole rather than as the declared deviation it is.
        (
            "decoder.decoder.input",
            "the transformer's input projection: one row per latent voxel plus the register tokens",
        ),
        *(
            (
                f"decoder.decoder.layers.{index}",
                f"layer {index}'s residual stream after both adds",
            )
            for index in VIDEO_LAYERS
        ),
        *(
            (
                f"decoder.decoder.layers.{index}.{name}",
                f"layer {index}'s {description}",
            )
            for index in VIDEO_LAYERS
            for name, description in (
                ("qkv", "the merged query, key and value projection, before the rotary - one file per branch"),
                ("attention", "attention output, after the rotary and the softmax"),
                ("mlp", "gated feed-forward output, after the gate's product"),
                ("output", "output projection, before the residual add"),
            )
        ),
        ("decoder.decoder.norm", "the transformer's final norm"),
        ("decoder.decoder.output", "the transformer's output projection to pixels"),
    ),
    "video_encoder": (
        # The condition path's first half, which had no probe at all. These
        # four localise a divergence to the input convolution, the first
        # stage's blocks and downsampling, the moment convolution, or the
        # one-by-one projection that follows the stitch.
        (
            "encoder.input",
            "the input convolution, unbiased: normalized pixels to width 0",
        ),
        (
            "encoder.stages.0",
            "stage 0's output: two blocks then its downsampling",
        ),
        (
            "encoder.output",
            "the moment convolution, after the final norm and activation",
        ),
        (
            "quant_conv",
            "the projection the condition stage reads the posterior from",
        ),
    ),
    "audio_decoder": (
        ("decoder.input", "the latent to the decoder's width"),
        ("decoder.network.conv_pre", "the first convolution, after the latent projection"),
        ("decoder.network.ups.0", "the first upsampling stage"),
        ("decoder.network.resblocks.0", "the first anti-aliased residual block"),
        ("decoder.network.activation_post", "the anti-aliased activation before the last convolution"),
        ("decoder.network.conv_post", "the last convolution, before the clamp"),
    ),
}


# A stage probe runs a component the candidate has already run, on the candidate's own input. The
# value written is the component's output for that input, under the name the candidate's tree gives it,
# so the two trees line up by name and a difference is arithmetic rather than a second draw.
STAGES = {
    "decode_out": (
        "video_decoder",
        "run host: the reference's video decoder on the candidate's latents, one clip",
    ),
    "encode_rows": (
        "video_encoder",
        "run host: the reference's video encoder on the candidate's pixels, its patch rows",
    ),
}

# The precision the deployment declares per component, which is the default a stage runs in: the video
# decoder is FP16 and every transformer block is BF16 (design section 1). A stage passes this as the
# module's compute dtype, so a comparison in another dtype would measure the conversion.
STAGE_DTYPES = {"decode_out": "torch.float16", "encode_rows": "torch.float32"}

# The case's `dtype` word as `--dtype` spells it. An encode stage's case states the deployment's
# type, and the reference must load at it: a comparison at another type would measure the fold.
CASE_DTYPES = {"f32": "torch.float32", "f16": "torch.float16", "bf16": "torch.bfloat16"}


def plan(probe: str) -> tuple[str, str, str]:
    """Split a probe into its component, its module path and what the value is."""
    component, _, path = probe.partition(":")
    if component not in CATALOGUE or not path:
        raise SystemExit(
            f"unknown probe {probe!r}; --list prints the catalogue as component:module.path"
        )
    for known, description in CATALOGUE[component]:
        if known == path:
            return component, path, description
    raise SystemExit(
        f"The catalogue has no probe {path!r} for {component}; --list prints what it has."
    )


def latent_shape(component: str) -> tuple[int, ...]:
    """The fixed input's shape for a component, in that component's own layout."""
    if component == "video_decoder":
        # [batch, channels, latent frames, latent height, latent width]; the channel count is the
        # checkpoint's own and is checked against the module below.
        return (1, 0, VIDEO_LATENT_FRAMES, VIDEO_LATENT_SIDE, VIDEO_LATENT_SIDE)
    if component == "video_encoder":
        # [batch, channels, frames, height, width] of normalized pixels: one
        # 256-pixel tile of one clip. The values are a standard normal draw,
        # which is *not* what a real condition looks like - a normalized pixel
        # is bounded - so this is the reference's own reproducible value, to be
        # replayed by whichever case writes the same raster on the crate side.
        return (1, 0, ENCODER_FRAMES, TILE_SIDE, TILE_SIDE)
    if component == "audio_decoder":
        # [stereo, channels, frames]: each stereo channel is one batch row.
        return (2, 0, AUDIO_LATENT_FRAMES)
    raise SystemExit(f"no fixed input for component {component}")


def build(component: str, config):
    """Build the component's reference module from its typed config."""
    from uniserve.models.minimax_h3 import audio_vae, video_vae

    if component == "video_decoder":
        return video_vae.Model(config.video_vae)
    if component == "video_encoder":
        # The reference's own encoder: the causal network, then the projection.
        return video_vae.Encoder(config.video_vae)
    return audio_vae.Model(config.audio_vae)


def mapping(component: str, module):
    """The checkpoint mapping `weights.py` declares for this component, built the same way."""
    from uniserve.loading import weights

    if component == "video_decoder":
        from uniserve.models.minimax_h3 import video_vae

        return weights.ModuleMapping(
            module,
            "video_decoder",
            lambda reader: video_vae.assignments(module, reader),
            frozenset(name for name, _ in module.named_parameters()),
        )
    if component == "video_encoder":
        from uniserve.models.minimax_h3 import video_vae

        return weights.ModuleMapping(
            module,
            "video_encoder",
            lambda reader: video_vae.encoder_assignments(module, reader),
            frozenset(name for name, _ in module.named_parameters()),
        )
    from uniserve.models.minimax_h3 import audio_vae

    return weights.ModuleMapping(
        module,
        "audio_decoder",
        lambda reader: audio_vae.assignments(module, reader),
        frozenset(name for name, _ in module.named_parameters()),
    )


def component_source(name: str, checkpoint: Path):
    """The checkpoint source for one component: the files the loader would read its weights from."""
    from uniserve.loading import checkpoint as checkpoint_module

    directory = checkpoint / (
        "audio_vae" if name == "audio_decoder" else "vae"
    )
    files = tuple(sorted(directory.glob("*.safetensors")))
    if not files:
        index = directory / "diffusion_pytorch_model.safetensors.index.json"
        hint = (
            f" ({index.name} lists them, so the shards are not fetched)"
            if index.is_file()
            else ""
        )
        raise SystemExit(
            f"{directory} holds no weight file{hint}; this dump needs the checkpoint's own weights "
            f"beside its config, which the serving loader fetches before it constructs the module"
        )
    return checkpoint_module.Source(name, root=directory, files=files)


def capture(module: nn.Module, path: str) -> dict[str, torch.Tensor]:
    """Hook one module and return the dict a forward run fills with its output.

    A module that returns one tensor fills the key `path`. A *merged* projection returns a mapping of
    branch name to tensor, and one key per branch is filled: `f"{path}.{branch}"`. The reference's
    fused Q/K/V is three projections in one container and never a concatenated tensor
    (`uniserve/nn/linear.py:448`), so a branch per file is the form a comparison can read - each
    branch against the Rust side's own projection of the same name.

    A tuple is refused rather than indexed. Writing its first element under a name that claims all of
    them would report agreement on a third of what the name says.
    """
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        if isinstance(output, Mapping):
            if not output:
                raise SystemExit(f"{path} returned no branch to write")
            for branch, value in output.items():
                captured[f"{path}.{branch}"] = value.detach().to(torch.float32).cpu()
            return
        if isinstance(output, tuple):
            raise SystemExit(
                f"{path} returned a {len(output)}-tuple; a probe is one tensor or a merged projection's "
                f"branches, and indexing here would write the first of them under a name that claims them all"
            )
        captured[path] = output.detach().to(torch.float32).cpu()

    module.get_submodule(path).register_forward_hook(hook)
    return captured


def dump(out: Path, name: str, values) -> None:
    """Write one tensor in the container `examples/compare_blocks.rs` reads."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dump_tables import dump as write

    write(out, name, values)


def dtype_of(options, default: str):
    """The compute dtype a probe loads its parameters in, named as the option spells it."""
    name = options.dtype or default
    dtype = DTYPES.get(name)
    if dtype is None:
        raise SystemExit(f"--dtype takes one of {sorted(DTYPES)}")
    return dtype


def read_tensor(path: Path) -> torch.Tensor:
    """Read one H3D1 tensor: the container this directory writes and the Rust comparator reads."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dump_tables import load

    try:
        return load(path)
    except ValueError as error:
        raise SystemExit(str(error)) from error


def read_sidecars(checkpoint: Path):
    """The checkpoint's typed configs, through the reference's own loading boundary."""
    try:
        from uniserve.models.minimax_h3 import config as h3_config
    except ImportError as error:
        print(f"IMPORT FAILED: {type(error).__name__}: {error}")
        print("  the reference's compiled extensions are not built in this environment")
        raise SystemExit(1) from error
    try:
        return h3_config.read_config(checkpoint, io=None, sources=())
    except Exception as error:  # noqa: BLE001 - the first line of a bad checkpoint path
        print(f"CONFIG FAILED: {type(error).__name__}: {error}")
        raise SystemExit(1) from error


def run_stage(options) -> int:
    """Run a component on the candidate's own input and write its output under the candidate's name.

    The candidate's tree is the contract: this reads the tensor the candidate's stage read and the
    `case.json` beside it, so both sides compute over the same bytes. Nothing here draws an input,
    which is why a stage probe has no fixed input in the catalogue.
    """
    if options.probe == "encode_rows":
        return run_encode_rows(options)
    component, description = STAGES[options.probe]
    stack = options.stack
    if stack is None:
        raise SystemExit(
            f"--probe {options.probe} reads the candidate's tree: --stack <out>/{component}"
        )
    case_path, input_path = stack / "case.json", stack / "latents.h3t"
    for path in (case_path, input_path):
        if not path.is_file():
            raise SystemExit(
                f"{path} is missing; a decode stage writes `case.json` and `latents.h3t` under its"
                f" own directory"
            )
    case = json.loads(case_path.read_text())
    latents = read_tensor(input_path)
    print(f"probe   {options.probe} ({description})")
    print(f"stack   {stack}")
    print(
        f"case    canvas {case.get('canvas')},"
        f" {case.get('latent_frames')} latent frames,"
        f" latent dtype {case.get('latent_dtype')},"
        f" {case.get('tiles')} tile(s),"
        f" {case.get('position_rows')} position rows"
    )
    # The candidate's case says where its latents came from. It changes nothing about the
    # comparison - this reads the candidate's own bytes, so both sides read the same numbers
    # either way - but a generated input is a sawtooth and not a request, and a page that does not
    # say so invites the wrong conclusion about what the stage proves.
    print(f"latents {case.get('latents')}; the input both sides read")
    print(f"input   {tuple(latents.shape)} {latents.dtype}")
    if latents.ndim != 5:
        raise SystemExit(
            f"{input_path} holds {tuple(latents.shape)};"
            f" a decode reads [1, C, T, h, w] latents"
        )

    sidecars = read_sidecars(options.checkpoint)
    module = load_module(component, build(component, sidecars), options)
    dtype = dtype_of(options, STAGE_DTYPES[options.probe])
    print(f"dtype   {dtype} parameters on {options.device}")
    device = torch.device(options.device)
    with torch.no_grad():
        decoded = module(latents.to(device=device))
    # NCTHW in and out: one batch row, one clip, the decoded precision. The stage's tree is one
    # clip as `[frames, height, width, 3]` channels last, which is the layout the checkpoint's
    # output path and the crate's decoder both state.
    if decoded.ndim != 5 or decoded.shape[0] != 1 or decoded.shape[1] != 3:
        raise SystemExit(
            f"the module returned {tuple(decoded.shape)}; a clip decodes to [1, 3, T, H, W]"
        )
    raster = decoded[0].permute(1, 2, 3, 0).contiguous()
    print(f"got     {tuple(raster.shape)} {raster.dtype} channels last")
    name = f"{component}/raster"
    (options.out / component).mkdir(parents=True, exist_ok=True)
    dump(options.out, name, raster)
    (options.out / component / "case.json").write_text(case_path.read_text())
    print(f"wrote   {options.out / (name + '.h3t')}")
    # Same convention as the stack producer: the root the comparator is given is the one holding
    # the component directories, so both trees keep the same names and a file on one side only is
    # listed rather than counted as a difference.
    candidate = stack.parent if stack.name == component else stack
    print(
        "compare cargo run -p minimax-h3 --example compare_blocks --"
        f" --reference {options.out}"
        f" --candidate {candidate}"
        f" --out {options.out}-compare.txt"
    )
    return 0


def run_encode_rows(options) -> int:
    """Encode the candidate's pixels with the reference's video encoder.

    The candidate's stage writes `pixels.h3t`, FP32 `[channels, frames, rows, columns]` in the unit
    range - the raster the encoder reads - and a `case.json` that names it. The reference encodes
    RGB24 frames, so the raster is scaled back to bytes; the harness's pattern is an integer over
    255, so the round trip is exact. The posterior draw is the encoder's own `draw_noise`, seeded
    with the reference's `POSTERIOR_SEED` rather than with the request's seed, because the
    reference fixes it so that a condition always encodes to the same latent - so this half has no
    convention to reproduce beyond the pixels themselves.

    The rows that come back are the conditioner's input, `[rows, channels * 4]` FP32. The crate's
    stage writes the *projected* prefix instead, whose projection is the DiT's; when the crate
    writes the same rows beside the pixels this compares them, and until then it writes the
    reference side and says so.
    """
    component = "video_encoder"
    stack = options.stack
    if stack is None:
        raise SystemExit(
            "--probe encode_rows reads the candidate's tree:"
            " --stack <out>/latent_encoder"
        )
    case_path = stack / "case.json"
    if not case_path.is_file():
        raise SystemExit(
            f"{case_path} is missing; an encode stage writes `case.json`"
            f" under its own directory"
        )
    case = json.loads(case_path.read_text())
    # The case names the raster it read; `pixels` is a description of the pattern, not a path.
    stated = case.get("pixels_file")
    pixels_path = Path(stated) if isinstance(stated, str) else Path("pixels.h3t")
    if not pixels_path.is_absolute():
        pixels_path = stack / pixels_path
    if not pixels_path.is_file():
        raise SystemExit(
            f"{pixels_path} is missing; the encode stage writes the raster it"
            f" read, so that both sides compute over one input rather than two"
        )
    pixels = read_tensor(pixels_path)
    print(f"probe   encode_rows ({STAGES['encode_rows'][1]})")
    print(f"stack   {stack}")
    print(
        f"case    canvas {case.get('canvas')}, dtype {case.get('dtype')},"
        f" prefix rows {case.get('prefix_rows')}"
    )
    print(f"pixels  {pixels_path.name}: {tuple(pixels.shape)} {pixels.dtype}")
    if pixels.dtype != torch.float32:
        raise SystemExit(
            f"{pixels_path} holds {pixels.dtype}; the crate's raster is FP32"
        )
    if pixels.ndim != 4:
        raise SystemExit(
            f"{pixels_path} holds {tuple(pixels.shape)};"
            f" an encode reads [channels, frames, rows, columns]"
        )
    stated_shape = case.get("pixels_shape")
    if isinstance(stated_shape, list) and tuple(stated_shape) != tuple(
        int(extent) for extent in pixels.shape
    ):
        raise SystemExit(
            f"{pixels_path.name} holds {tuple(pixels.shape)} and the case states"
            f" {tuple(stated_shape)}: the file is not the raster the case names"
        )
    channels, frames, height, width = (int(extent) for extent in pixels.shape)
    if channels != 3:
        raise SystemExit(
            f"{pixels_path} holds {channels} channels;"
            f" the reference encodes RGB24 frames"
        )
    try:
        from uniserve.media import image, video
    except ImportError as error:
        print(f"IMPORT FAILED: {type(error).__name__}: {error}")
        print("  the reference's compiled extensions are not built in this environment")
        return 1
    if options.dtype is None:
        stated_dtype = CASE_DTYPES.get(str(case.get("dtype")))
        if stated_dtype is not None:
            # The case states the deployment's element type and the reference loads
            # at it, which is what makes this one computation rather than two types.
            options.dtype = stated_dtype
    print(f"dtype   {options.dtype or STAGE_DTYPES['encode_rows']}")
    sidecars = read_sidecars(options.checkpoint)
    module = load_module(component, build(component, sidecars), options)
    device = torch.device(options.device)
    size = video.Config(frames, image.Config(height, width))
    layout = module.output_layout(size)["video"]
    shape = tuple(int(extent) for extent in layout.shape)
    print(f"layout  {shape}, the conditioner's input rows")
    # The posterior is given noise, and the reference's is the **encoder's own** draw: `draw_noise`
    # seeds a fresh generator with the module's `POSTERIOR_SEED`, element zero every time, so a
    # condition always encodes to the same latent. The request's draw is the *anchor's* alone
    # (`denoiser.py:524`), a different stream - so the case's `noise_file` is not comparable with
    # this one and is not compared: a difference here is the design, not a defect.
    shape_noise = tuple(int(extent) for extent in module.posterior_shape(size))
    draw = torch.empty(shape_noise, dtype=torch.float32, device=device)
    module.draw_noise(size, draw)
    print(
        f"noise   the encoder's own draw at the module's POSTERIOR_SEED, {shape_noise}"
        f" - the request's draw is the anchor's alone"
    )
    stated_noise = case.get("noise_file")
    if isinstance(stated_noise, str):
        noise_path = Path(stated_noise)
        if not noise_path.is_absolute():
            noise_path = stack / noise_path
        if noise_path.is_file():
            theirs = read_tensor(noise_path)
            print(
                f"noise   {noise_path.name} is the anchor's draw,"
                f" {tuple(theirs.shape)} {theirs.dtype}: a different stream, not compared"
            )
        else:
            print(f"noise   {noise_path} is absent; the anchor's draw is not read here")
    # The pattern the harness writes is an integer over 255, so scaling back is exact.
    frames_rgb = (
        pixels.mul(255.0).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 3, 0)
    )
    rows = torch.zeros(shape, dtype=torch.float32)
    if hasattr(module, "load_kernels"):
        module.load_kernels()
    with torch.no_grad():
        for interval in module.clips(frames):
            piece = module.encode(
                (frames_rgb[interval].contiguous().to(device),),
                clips=(interval,),
                num_frames=(frames,),
                noise=(draw,),
            )[0]
            span = piece.layout.local_slice[0]
            rows[span.start : span.stop] = piece.tensor.detach().to(torch.float32).cpu()
            print(
                f"clip    frames {interval.start}..{interval.stop}"
                f" -> rows {span.start}..{span.stop}"
            )
    print(f"got     {tuple(rows.shape)} {rows.dtype}, mean |x| {rows.abs().mean():.4g}")
    (options.out / component).mkdir(parents=True, exist_ok=True)
    dump(options.out, f"{component}/rows_ref", rows)
    (options.out / component / "case.json").write_text(case_path.read_text())
    print(f"wrote   {options.out / (component + '/rows_ref.h3t')}")
    candidate = stack / "rows.h3t"
    if not candidate.is_file():
        print(
            "compare the crate writes no unprojected rows today - its stage"
            " writes the projected prefix, whose projection is the DiT's. This"
            " file is the reference side; the same rows written beside the"
            " pixels make the comparison one command."
        )
        return 0
    got = read_tensor(candidate).to(torch.float32)
    if tuple(got.shape) != tuple(rows.shape):
        print(
            f"SHAPE   {candidate.name} is {tuple(got.shape)}, the reference's"
            f" {tuple(rows.shape)}: the two rows cannot be compared"
        )
        return 1
    difference = got - rows
    relative = (difference.norm() / rows.norm().clamp_min(1e-12)).item()
    print(
        f"compare rows.h3t against rows_ref.h3t: rel L2 {relative:.3e},"
        f" max |d| {difference.abs().max():.3e}"
    )
    return 0


def load_module(component: str, module: nn.Module, options):
    """Load the checkpoint's own tensors into a built component, at the probe's dtype."""
    from uniserve.loading import loader, weights as weight_options

    source = component_source(component, options.checkpoint)
    print(f"load    {source.name} from {source.root} ({len(source.files)} file(s))")
    dtype = dtype_of(options, "torch.bfloat16")
    try:
        reports = loader.load_weights(
            module,
            (source,),
            mapping=lambda root: (mapping(component, root),),
            device=options.device,
            weights=weight_options.Config(dtype=dtype),
        )
    except Exception as error:  # noqa: BLE001 - the lead's first-line rule
        print(f"LOAD FAILED: {type(error).__name__}: {error}")
        raise SystemExit(1) from error
    for report in reports:
        print(
            f"loaded  {len(report.loaded)} parameter(s); missing {len(report.missing)}, "
            f"unexpected {len(report.unexpected)}, incomplete {len(report.incomplete)}"
        )
        for name in sorted(report.missing)[:8]:
            print(f"missing  {name}")
        for name in sorted(report.unexpected)[:8]:
            print(f"unexpected  {name}")
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print the probe catalogue and exit")
    parser.add_argument("--probe", help="component:module.path, as --list prints it")
    parser.add_argument("--checkpoint", type=Path, help="the MiniMax-H3 checkpoint directory")
    parser.add_argument("--out", type=Path, help="where to write the dump")
    parser.add_argument(
        "--stack",
        type=Path,
        help=(
            "the candidate's stage directory: `latents.h3t` and `case.json` for"
            " `--probe decode_out`, `pixels.h3t` and `case.json` for `--probe"
            " encode_rows`"
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="the device the module runs on (default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help=(
            "the compute dtype the module's parameters load in; the default is the"
            " deployment's own - bfloat16 for a transformer block, float16 for the"
            " video decoder"
        ),
    )
    options = parser.parse_args(argv)

    if options.list:
        for component, probes in CATALOGUE.items():
            for path, description in probes:
                print(f"{component}:{path}\t{description}")
        for stage, (component, description) in STAGES.items():
            print(f"{stage}\t{description} (component {component})")
        return 0
    if not (options.probe and options.checkpoint and options.out):
        parser.error("--probe, --checkpoint and --out are required unless --list")
    if options.probe in STAGES:
        return run_stage(options)

    component, path, description = plan(options.probe)
    print(f"probe   {component}:{path}")
    print(f"value   {description}")
    print(f"input   {latent_shape(component)[:2]}... {latent_shape(component)[2:]} (fixed, seed {SEED})")

    try:
        from uniserve.models.minimax_h3 import config as h3_config
    except ImportError as error:
        print(f"IMPORT FAILED: {type(error).__name__}: {error}")
        print("  the reference's compiled extensions are not built in this environment")
        return 1

    try:
        sidecars = h3_config.read_config(options.checkpoint, io=None, sources=())
    except Exception as error:  # noqa: BLE001 - the first line of a bad checkpoint path
        print(f"CONFIG FAILED: {type(error).__name__}: {error}")
        return 1
    if component == "video_encoder":
        clip = sidecars.video_vae.clip_length
        if clip != ENCODER_FRAMES:
            print(
                f"CONFIG FAILED: the checkpoint's clip_length is {clip}, and"
                f" this probe's fixed input is one {ENCODER_FRAMES}-frame clip"
            )
            return 1
    module = build(component, sidecars)
    # The channel count is the checkpoint's; the fixed input only fixes the extents.
    shape = latent_shape(component)
    channels = _latent_channels(component, sidecars, module)
    shape = (shape[0], channels, *shape[2:])
    print(f"module  {type(module).__name__} with {len(list(module.parameters()))} parameters")
    print(f"probe   resolves to {type(module.get_submodule(path)).__name__}")
    print(f"expect  {component} output of one fixed input, shape set by {shape}")

    load_module(component, module, options)

    captured = capture(module, path)
    generator = torch.Generator().manual_seed(SEED)
    latents = torch.randn(shape, generator=generator, dtype=torch.float32)
    print(f"run     input {tuple(latents.shape)} -> {path}")
    with torch.no_grad():
        module(latents)
    if not captured:
        print(f"NO VALUE: {path} did not run; the input may be too small for this probe")
        return 1
    # The tree's convention is the Rust side's: one file per probe at `<block>/<name>.npy`, which is
    # what `compare.py` matches on (`$UNISERVE_DIAG_DIR/<block>/<name>.npy`, notes/h3/diagnosis.md 3).
    # A merged projection fills one name per branch, and each is a probe of its own.
    (options.out / component).mkdir(parents=True, exist_ok=True)
    for name, value in captured.items():
        print(f"got     {name} {tuple(value.shape)} {value.dtype} on the component's own path")
        dump(options.out, f"{component}/{name}", value)
        print(f"wrote   {options.out / (component + '/' + name + '.h3t')}")
    return 0


def _latent_channels(component: str, config, module: nn.Module) -> int:
    """The checkpoint's latent channel count for a component, from its config and module.

    The video decoder's post-quantisation convolution and the audio decoder's input projection both
    take the latent width, so the module itself names it; the config is asked first so a mismatch
    between the two stops here rather than three tiles into a decode.
    """
    if component == "video_encoder":
        # The encoder reads pixels and writes the posterior's two fields; the
        # config is asked first so a mismatch stops here rather than one stage
        # into an encode.
        expected = config.video_vae.in_channels
        found = module.encoder.input.in_channels
        if expected != found:
            raise SystemExit(
                f"{component}: the config says {expected} input channels,"
                f" the module expects {found}"
            )
        return found
    expected = (
        config.video_vae.latent_channels
        if component == "video_decoder"
        else config.audio_vae.latent_channels
    )
    found = (
        module.decoder.post_quant_conv.in_channels
        if component == "video_decoder"
        else module.decoder.input.in_channels
    )
    if expected != found:
        raise SystemExit(
            f"{component}: the config says {expected} latent channels and the module expects {found}"
        )
    return found


if __name__ == "__main__":
    raise SystemExit(main())
