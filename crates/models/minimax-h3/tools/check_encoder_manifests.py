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

# The complete field set each operator's key carries, in the order the operator's `Display` writes
# them. A call that carries all of them names one key exactly; `--keys` emits the fixture's lines
# from those and names the calls it cannot complete rather than printing a partial key.
KEY_FIELDS: dict[str, tuple[str, ...]] = {
    "dense_attention/fa4": (
        "dtype",
        "kv",
        "out",
        "head_dim",
        "q_heads",
        "kv_heads",
        "mask",
        "window",
    ),
    "qk_norm_rope/triton": (
        "dtype",
        "output",
        "head_dim",
        "q_heads",
        "k_heads",
        "rotation",
        "epsilon",
        "rows",
        "block",
        "pdl",
    ),
    "rotary_factors/triton": (
        "dtype",
        "positions",
        "columns",
        "block",
        "scale",
        "pdl",
        "axes",
        "sections",
        "interleaved",
    ),
    "act_and_mul/triton": (
        "dtype",
        "width",
        "block",
        "activation",
        "gate_stride",
        "value_stride",
        "wide",
        "pdl",
        "stepwise",
    ),
    "layer_norm/triton": ("dtype", "columns", "tile", "bias", "pdl"),
}

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
                    "dtype": "bf16",
                    "output": "bf16",
                    "head_dim": str(head_dim),
                    "q_heads": str(heads // degree),
                    "k_heads": str(kv_heads // degree),
                    "rotation": "half",
                    "epsilon": f"0x{_f32_bits(epsilon):08x}",
                    "rows": "1",
                    "block": _head_block(head_dim),
                    "pdl": "false",
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
                "block": "256",
                "scale": "0x3f800000",
                "pdl": "false",
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
                "block": "1024",
                # The gate and the up projection are two separate `[tokens, intermediate]`
                # buffers (`text_encoder.rs`, the layer's forward), not one interleaved one: the
                # stride is the intermediate width, and a `2 * ffn` here named a program the code
                # never launches.
                "gate_stride": str(ffn),
                "value_stride": str(ffn),
                "activation": "silu",
                "wide": "false",
                "pdl": "false",
                "stepwise": "false",
            },
            "text_encoder.rs: the gated MLP's activation",
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
            {
                "dtype": "bf16",
                "columns": str(hidden),
                "tile": _row_tile(hidden),
                "bias": "true",
                "pdl": "false",
            },
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
                {
                    "dtype": "f32",
                    "columns": str(width),
                    "tile": _row_tile(width),
                    "bias": "true",
                    "pdl": "false",
                },
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


def _head_block(head_dim: int) -> str:
    """The head tile a rotation loads at a time: the head's own power of two, capped at 1024.

    `crates/nn/src/layers/rope.rs` rounds the head width rather than taking a constant, so a
    128-wide head compiles a 128-lane tile and a 256-wide one a 256-lane tile.
    """
    return str(min(1 << (head_dim - 1).bit_length(), 1024))


def _row_tile(columns: int) -> str:
    """The lane tile a row of `columns` takes: its own power of two (`nn::layers::row_tile`)."""
    return str(1 << (columns - 1).bit_length())


def called_keys(calls: list[Call]) -> tuple[list[tuple[str, str]], list[str]]:
    """The fixture lines these calls need, and the calls this tool cannot express.

    A call that does not carry every field its operator's `Display` writes names no key, and is
    reported rather than printed short: a partial key in the store's fixture would be a line that
    compiles nothing.
    """
    lines: list[tuple[str, str]] = []
    unknown: list[str] = []
    for package, fields, reason in calls:
        order = KEY_FIELDS.get(package)
        if order is None:
            unknown.append(
                f"NOT EMITTED {package}: this tool does not model its field set [{reason}]"
            )
            continue
        missing = [field for field in order if field not in fields]
        if missing:
            unknown.append(
                f"NOT EMITTED {package}: the call omits {', '.join(missing)} [{reason}]"
            )
            continue
        operator = package.split("/")[0]
        key = "_".join(f"{field}_{fields[field]}" for field in order)
        lines.append((operator, key))
    return lines, unknown


def keys(calls: list[Call]) -> int:
    """Print the fixture's lines: each completed call's operator name and its exact key."""
    lines, unknown = called_keys(calls)
    for line in unknown:
        print(f"  {line}")
    for operator, key in lines:
        print(f"{operator}\t{key}")
    return len(lines)


def against(sections: list[tuple[str, list[Call]]], path: pathlib.Path) -> int:
    """Ask the store's list whether it holds every program the calls name.

    This is the check that found the store a stage short: two lists, each
    internally consistent and one missing a row the other had. The dangerous
    direction is a key a call names and the fixture lacks - the forward binds a
    program the store cannot launch - and it is a failure. The other direction
    is not: the fixture holds every stage's rows, so a row nothing here calls
    belongs to another stage.

    The verdict covers the calls whose key this tool derives and states both
    counts, because a partial verdict that called itself complete would be the
    same disease in the instrument that exists to find it.
    """
    if not path.is_file():
        print(f"no {path}", file=sys.stderr)
        return 2
    listed: set[tuple[str, str]] = set()
    for index, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t")
        if len(parts) != 2:
            print(f"{path}:{index}: not `operator<TAB>key`", file=sys.stderr)
            return 2
        listed.add((parts[0], parts[1]))

    called: list[tuple[str, str, str]] = []
    unexpressed = 0
    for name, calls in sections:
        lines, gaps = called_keys(calls)
        called.extend((name, operator, key) for operator, key in lines)
        for gap in gaps:
            print(f"  not expressed {name}: {gap}", file=sys.stderr)
            unexpressed += 1

    missing = [
        (name, operator, key)
        for name, operator, key in called
        if (operator, key) not in listed
    ]
    print(
        f"fixture {path}: {len(listed)} rows; expressed {len(called)} of"
        f" {len(called) + unexpressed} calls, {unexpressed} name no key this"
        " tool derives"
    )
    for name, operator, key in missing:
        print(f"  MISSING {name}: {operator} {key}")
    print(
        f"{len(called) - len(missing)} of {len(called)} called keys are in the"
        f" fixture, {len(missing)} missing"
    )
    return 1 if missing else 0


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
    parser.add_argument(
        "--against",
        type=pathlib.Path,
        default=None,
        help=(
            "after the manifests, ask the store's list (`notes/programs/minimax-h3-keys.txt`) "
            "whether it holds every program these calls name"
        ),
    )
    parser.add_argument(
        "--keys",
        action="store_true",
        help=(
            "print the operators' keys in the fixture's format "
            "(`notes/programs/minimax-h3-keys.txt`) instead of checking the manifests"
        ),
    )
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

    if arguments.keys:
        emitted = sum(keys(calls) for _, calls in sections)
        print()
        print(f"{emitted} keys emitted")
        return 0

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
    code = 1 if total else 0
    if arguments.against is not None:
        print()
        code = against(sections, arguments.against) or code
    return code


if __name__ == "__main__":
    sys.exit(main())
