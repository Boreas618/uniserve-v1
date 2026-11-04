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
visible in the first screen. A probe is a module path, relative to the component's root module, and the
value written is that module's output for a fixed input the options below make reproducible.

Run it from the repository root with the repository's environment and the repository importable
(`PYTHONPATH=.` unless the package is installed editable); it needs a machine where the reference's
compiled extensions are built, which is the run host and not every sandbox. The plan is printed before
the first import of the model, so a broken environment and a wrong probe path are different failures.

This is a **diagnosis, never an acceptance gate** (`specs/worker.md` 10).
"""

from __future__ import annotations

import argparse
import sys
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
                ("qkv", "fused query, key and value projection, before the rotary"),
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
    """Hook one module and return the dict a forward run fills with its output."""
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        captured[path] = value.detach().to(torch.float32).cpu()

    module.get_submodule(path).register_forward_hook(hook)
    return captured


def dump(out: Path, name: str, values) -> None:
    """Write one tensor in the container `examples/compare_blocks.rs` reads."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dump_tables import dump as write

    write(out, name, values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print the probe catalogue and exit")
    parser.add_argument("--probe", help="component:module.path, as --list prints it")
    parser.add_argument("--checkpoint", type=Path, help="the MiniMax-H3 checkpoint directory")
    parser.add_argument("--out", type=Path, help="where to write the dump")
    parser.add_argument(
        "--dtype",
        default="torch.bfloat16",
        help="the compute dtype the module's parameters load in (default: the checkpoint's)",
    )
    options = parser.parse_args(argv)

    if options.list:
        for component, probes in CATALOGUE.items():
            for path, description in probes:
                print(f"{component}:{path}\t{description}")
        return 0
    if not (options.probe and options.checkpoint and options.out):
        parser.error("--probe, --checkpoint and --out are required unless --list")

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

    try:
        from uniserve.loading import loader, weights as weight_options
    except ImportError as error:
        print(f"IMPORT FAILED: {type(error).__name__}: {error}")
        return 1

    source = component_source(component, options.checkpoint)
    print(f"load    {source.name} from {source.root} ({len(source.files)} file(s))")
    dtype = DTYPES.get(options.dtype)
    if dtype is None:
        parser.error(f"--dtype takes one of {sorted(DTYPES)}")
    try:
        reports = loader.load_weights(
            module,
            (source,),
            mapping=lambda root: (mapping(component, root),),
            device="cpu",
            weights=weight_options.Config(dtype=dtype),
        )
    except Exception as error:  # noqa: BLE001 - the lead's first-line rule
        print(f"LOAD FAILED: {type(error).__name__}: {error}")
        return 1
    for report in reports:
        print(
            f"loaded  {len(report.loaded)} parameter(s); missing {len(report.missing)}, "
            f"unexpected {len(report.unexpected)}, incomplete {len(report.incomplete)}"
        )
        for name in sorted(report.missing)[:8]:
            print(f"missing  {name}")
        for name in sorted(report.unexpected)[:8]:
            print(f"unexpected  {name}")

    captured = capture(module, path)
    generator = torch.Generator().manual_seed(SEED)
    latents = torch.randn(shape, generator=generator, dtype=torch.float32)
    print(f"run     input {tuple(latents.shape)} -> {path}")
    with torch.no_grad():
        module(latents)
    if path not in captured:
        print(f"NO VALUE: {path} did not run; the input may be too small for this probe")
        return 1
    value = captured[path]
    print(f"got     {tuple(value.shape)} {value.dtype} on the component's own path")
    # The tree's convention is the Rust side's: one file per probe at `<block>/<name>.npy`, which is what
    # `compare.py` matches on (`$UNISERVE_DIAG_DIR/<block>/<name>.npy`, notes/h3/diagnosis.md 3).
    name = f"{component}/{path}"
    (options.out / component).mkdir(parents=True, exist_ok=True)
    dump(options.out, name, value)
    print(f"wrote   {options.out / (name + '.h3t')}")
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
