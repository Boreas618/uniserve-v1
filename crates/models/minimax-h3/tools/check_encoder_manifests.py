#!/usr/bin/env python3
"""Check the encoders' descriptor values against the package manifests that declare them.

    .venv/bin/python crates/models/minimax-h3/tools/check_encoder_manifests.py \
        [--root /opt/uv/models/MiniMax-H3] [--kernels crates/ops/kernels]

A descriptor outside a package's `dimensions` is a key with no compiled program: `_compile.py`
accepts exactly the range the manifest declares, and `Generation::kernel` answers `MissingKernel`
for anything else. So a value the model builds and no manifest lists is a refusal at the first
call that takes it - and nothing host-side finds it by running the model, because a deployment
plan sizes buffers and never names a program.

This is the encoders' half of that check, beside `check_manifests.py`'s DiT half. It reads the
text encoder's own `config.json` (whose `text_config` and `vision_config` carry every number) and
the VAE's `config.json`.

**One descriptor at a time, which is the difference from the DiT half.** A package may hold several
`[[kernel.descriptors]]` blocks - `qk_norm_rope` states one without a `rotated` field and one with
it, `rotary_factors` one per axis count - and a key matches **one** block, not the union of them.
So every row here is a whole call: a dict of the fields that call's descriptor carries, and the
call is missing unless a single block declares every one of them. `manifest_values` in
`check_manifests.py` unions the blocks, which under-reports an op with a second variant; that is
why this file reads the blocks itself and imports only the pattern that parses one.

The rows name a code site, so a failure is actionable without reading this file. Two things it
deliberately does not check, for the same reasons as the DiT half:

  * tensor names - `keys()` is that check;
  * values whose descriptor depends on a view rather than on a config number (`frame_pad`'s tile
    split is computed by the operator from the views, and `frame_moments`' epsilon and the
    `posterior` block are parameters of the key rather than extents of the shape).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from check_manifests import DIMENSIONS

# One call: the package, the fields its descriptor carries, and why it carries them.
Call = tuple[str, dict[str, str], str]


def descriptor_blocks(path: pathlib.Path) -> list[dict[str, set[str]]]:
    """Every `[[kernel.descriptors]]` block of a package manifest, as field -> values."""
    blocks: list[dict[str, set[str]]] = []
    for line in path.read_text().splitlines():
        match = DIMENSIONS.match(line.strip())
        if match is None:
            continue
        block: dict[str, set[str]] = {}
        for field in re.finditer(r"(\w+)\s*=\s*\[([^\]]*)\]", match.group(1)):
            listed = {
                item.strip().strip('"') for item in field.group(2).split(",")
            }
            block[field.group(1)] = {value for value in listed if value}
        blocks.append(block)
    return blocks


def text_calls(text: dict) -> list[Call]:
    """Every descriptor call the text encoder's load and forward make."""
    hidden = text["hidden_size"]
    heads = text["num_attention_heads"]
    kv_heads = text["num_key_value_heads"]
    head_dim = text["head_dim"]
    ffn = text["intermediate_size"]
    epsilon = text["rms_norm_eps"]
    sections = text["rope_scaling"]["mrope_section"]
    calls: list[Call] = []

    # The widths count frequencies and the operator counts rotated elements, so each is doubled
    # (`text_encoder.rs::sections_of`); the cut is interleaved, which is the checkpoint's own.
    rotated_sections = "-".join(str(width * 2) for width in sections)
    for degree in (1, 2):
        if heads % degree or kv_heads % degree:
            continue
        why = f"over {degree} tensor-parallel rank(s)"
        calls.append(
            (
                "dense_attention/fa4",
                {
                    "dtype": "bf16",
                    "kv": "bf16",
                    "out": "bf16",
                    "head_dim": str(head_dim),
                    "q_heads": str(heads // degree),
                    "kv_heads": str(kv_heads // degree),
                    "mask": "dense",
                    "window": "false",
                },
                f"attention.rs: the text stack's dense attention, {heads}/{kv_heads} heads {why}",
            )
        )
        calls.append(
            (
                "qk_norm_rope/triton",
                {
                    "head_dim": str(head_dim),
                    "q_heads": str(heads // degree),
                    "k_heads": str(kv_heads // degree),
                    "rotated": str(head_dim),
                    "rotation": "half",
                    "block": "128",
                },
                f"rope.rs: the qk-norm rotation, full width over {degree} rank(s)",
            )
        )

    calls.append(
        (
            "rotary_factors/triton",
            {
                "dtype": "f32",
                "positions": "i32",
                "columns": str(head_dim // 2),
                "axes": str(len(sections)),
                "sections": rotated_sections,
                "interleaved": "true",
            },
            "positions.rs: the prompt's three-axis factors, the strongest compiled axis count",
        )
    )
    calls.append(
        (
            "rms_norm/triton",
            {"dtype": "bf16", "columns": str(hidden)},
            "text_encoder.rs: a layer's norms",
        )
    )
    calls.append(
        (
            "act_and_mul/triton",
            {
                "dtype": "bf16",
                "width": str(ffn),
                "gate_stride": str(2 * ffn),
                "value_stride": str(2 * ffn),
                "activation": "silu",
                "wide": "false",
                "stepwise": "false",
            },
            "text_encoder.rs: the gated MLP's activation",
        )
    )
    # The norms' epsilon is a descriptor value in this package, encoded as an f32 bit pattern.
    calls.append(
        (
            "qk_norm_rope/triton",
            {"epsilon": f"0x{_f32_bits(epsilon):08x}"},
            "text_encoder.rs: the qk norm's epsilon",
        )
    )
    return calls


def tower_calls(vision: dict) -> list[Call]:
    """Every descriptor call the vision tower's load and forward make."""
    hidden = vision["hidden_size"]
    heads = vision["num_heads"]
    head_dim = hidden // heads
    ffn = vision["intermediate_size"]
    calls: list[Call] = [
        (
            "layer_norm/triton",
            {"dtype": "bf16", "columns": str(hidden), "bias": "true"},
            "vision.rs: a block's norms",
        ),
        (
            "dense_attention/fa4",
            {
                "dtype": "bf16",
                "kv": "bf16",
                "out": "bf16",
                "head_dim": str(head_dim),
                "q_heads": str(heads),
                "kv_heads": str(heads),
                "mask": "dense",
                "window": "false",
            },
            "vision.rs: one time step's patches, every head a query and a key",
        ),
        (
            "rope_apply/triton",
            {
                "dtype": "bf16",
                "output": "bf16",
                "head_dim": str(head_dim),
                "heads": str(heads),
                "rotation": "half",
                "block": "128",
            },
            "vision.rs: the tower's own rotation, host-computed factors",
        ),
        (
            "gelu/triton",
            {"dtype": "bf16", "columns": str(ffn), "approximate": "tanh"},
            "vision.rs: the MLP's activation",
        ),
    ]
    return calls


def vae_calls(vae: dict) -> list[Call]:
    """Every descriptor call the video VAE encoder's load and forward make."""
    calls: list[Call] = []
    for width in sorted(set(vae["block_out_channels"])):
        calls.append(
            (
                "layer_norm/triton",
                {"dtype": "f32", "columns": str(width), "bias": "true"},
                f"encoding.rs: a residual block's norm at {width} channels",
            )
        )
        calls.append(
            (
                "snake/triton",
                {"dtype": "f32", "channels": str(width)},
                f"encoding.rs: a level's snake at {width} channels",
            )
        )
    calls.append(
        (
            "frame_moments/nvcc",
            {"dtype": "f32"},
            "encoding.rs: the frame statistics a pre-activated convolution reads",
        )
    )
    calls.append(
        (
            "frame_norm_pad/nvcc",
            {
                "dtype": "f32",
                "channels_last": "true",
                "centered": "false",
                "single_pixel": "false",
            },
            "encoding.rs: the pre-activated convolution's own norm and pad",
        )
    )
    return calls


def _f32_bits(value: float) -> int:
    """The f32 bit pattern of `value`, which is how the manifests encode an epsilon."""
    import struct

    return struct.unpack("<I", struct.pack("<f", value))[0]


def report(name: str, calls: list[Call], kernels: pathlib.Path) -> int:
    """Print `name`'s calls that no single manifest block declares, and answer how many."""
    print(name)
    missing = 0
    for package, fields, reason in calls:
        path = kernels / package / "package.toml"
        if not path.is_file():
            print(f"  NO PACKAGE {package}", file=sys.stderr)
            return -1
        blocks = descriptor_blocks(path)
        declared = any(
            all(
                field in block and value in block[field]
                for field, value in fields.items()
            )
            for block in blocks
        )
        if not declared:
            shown = ", ".join(
                f"{field} = {value}" for field, value in fields.items()
            )
            print(f"  MISSING {package}: {shown}  [{reason}]")
            missing += 1
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/opt/uv/models/MiniMax-H3")
    parser.add_argument("--kernels", default="crates/ops/kernels")
    arguments = parser.parse_args()

    root = pathlib.Path(arguments.root)
    kernels = pathlib.Path(arguments.kernels)
    text_path = root / "text_encoder" / "config.json"
    vae_path = root / "vae" / "config.json"
    for path in (text_path, vae_path):
        if not path.is_file():
            print(f"no {path}", file=sys.stderr)
            return 2

    text_config = json.loads(text_path.read_text())
    sections = [
        ("text encoder", text_calls(text_config["text_config"])),
        ("vision tower", tower_calls(text_config["vision_config"])),
        ("video VAE encoder", vae_calls(json.loads(vae_path.read_text()))),
    ]

    total = 0
    checked = 0
    for name, calls in sections:
        checked += len(calls)
        found = report(name, calls, kernels)
        if found < 0:
            return 2
        total += found
    print()
    print(
        f"{checked} descriptor calls checked, {total} with no declared program"
    )
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
