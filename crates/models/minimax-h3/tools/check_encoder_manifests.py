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
    # The same operator's FP32 backend declares its own descriptors, so a call at f32 is served by
    # this package and not by `fa4` - reading it as a gap in `fa4` is reading the wrong manifest.
    "dense_attention/f32": (
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
    # Four operators whose fields the call sites state outright - no `row_tile` and no width-derived
    # tile among them - so a key computed here is the call's own key and not a second implementation
    # of `nn`. Two of the day's three store gaps were rows of these shapes: a `gelu` at the merger's
    # width and a `scaled_residual` at the text stack's.
    "embedding/triton": ("dtype", "ids", "hidden", "first", "rows", "block", "pdl"),
    "gelu/triton": ("dtype", "columns", "block", "approximate", "pdl"),
    "scaled_residual/triton": ("dtype", "columns", "block", "update_bias"),
    # `rms_norm`'s tile is `row_tile(columns)` - the width rounded up to a power of two - which is the
    # one derived field this tool computes. A wrong derivation here cannot pass silently: it produces a
    # MISSING row that names a tile the fixture does not hold, which is a report to read, not a green.
    "rms_norm/triton": ("dtype", "columns", "tile", "pdl"),
    "snake/triton": ("dtype", "channels", "block", "pdl"),
    "frame_moments/nvcc": ("dtype", "epsilon"),
    "frame_norm_pad/nvcc": ("dtype", "channels_last", "centered", "single_pixel"),
    "rope_apply/triton": (
        "dtype",
        "output",
        "head_dim",
        "heads",
        "rotation",
        "rows",
        "block",
        "pdl",
    ),
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


def attention_package(dtype: str) -> str:
    """The `dense_attention` backend that declares a call at `dtype`.

    The operator has two packages and they do not overlap: `fa4` declares bf16 and f16, `f32`
    declares the FP32 path (cuBLASLt SGEMM plus a softmax kernel). The store picks the module by the
    descriptor's key, so the package a call is served by follows its element type.
    """
    return "dense_attention/f32" if dtype == "f32" else "dense_attention/fa4"


def text_calls(text: dict, dtype: str, ulysses: int = 1) -> list[Call]:
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
    # One key per degree the *deployment* binds: a degree-1 deployment never launches the split
    # variants, and a fixture generated from a walk at degree 1 legitimately holds none of them -
    # so deriving both would report a gap that no run can reach, which is a false red on the
    # strongest instrument this half has.
    degree = ulysses
    if heads % degree or kv_heads % degree:
        raise ValueError(
            f"{heads} query and {kv_heads} key heads do not split over {degree} rank(s)"
        )
    if True:
        why = f"over {degree} tensor-parallel rank(s)"
        calls.append(
            (
                attention_package(dtype),
                {
                    "dtype": dtype,
                    "kv": dtype,
                    "out": dtype,
                    "head_dim": str(head_dim),
                    "q_heads": str(heads // degree),
                    "kv_heads": str(kv_heads // degree),
                    # The reference hands the language stack
                    # `VarlenInput(lengths, lengths, (True,) * len(counts))`
                    # (`uniserve/models/encoder.py:482`), so the key is the causal form; the tower's
                    # is `(False,)` (`qwen3_vl/vision.py:281`). One stage reads both.
                    "mask": "causal",
                    "window": "false",
                },
                f"attention.rs: the text stack's causal attention, {heads}/{kv_heads} heads {why}",
            )
        )
        calls.append(
            (
                "qk_norm_rope/triton",
                {
                    "dtype": dtype,
                    "output": dtype,
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
            {"dtype": dtype, "columns": str(hidden), "tile": _row_tile(hidden), "pdl": "false"},
            "text_encoder.rs: a layer's norms",
        )
    )
    calls.append(
        (
            "act_and_mul/triton",
            {
                "dtype": dtype,
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
    # The token table the prompt is gathered from: one row per vocabulary id, `hidden` wide, and
    # eight ids per program. Its key carries the table's shape, not the request's token count, so one
    # row covers every prompt (`embedding.rs:93`).
    calls.append(
        (
            "embedding/triton",
            {
                "dtype": dtype,
                "ids": "i32",
                "hidden": str(hidden),
                "first": "0",
                "rows": str(text["vocab_size"]),
                "block": "8",
                "pdl": "false",
            },
            "text_encoder.rs: the token embedding the prompt is gathered from",
        )
    )
    # The language stack's residual stream: each retained layer's down projection folded into
    # `[tokens, hidden]` before the next layer's norm. It is the *text* width, and the tower's own
    # residual (below) is 1152 - two rows, neither covering the other.
    calls.append(
        (
            "scaled_residual/triton",
            {
                "dtype": dtype,
                "columns": str(hidden),
                "block": "1024",
                "update_bias": "false",
            },
            "text_encoder.rs: the residual stream the layers write into",
        )
    )
    return calls


def tower_calls(vision: dict, dtype: str) -> list[Call]:
    """Every descriptor call the vision tower's load and forward make."""
    hidden = vision["hidden_size"]
    heads = vision["num_heads"]
    head_dim = hidden // heads
    ffn = vision["intermediate_size"]
    merge = vision["spatial_merge_size"]
    calls: list[Call] = [
        (
            "layer_norm/triton",
            {
                "dtype": dtype,
                "columns": str(hidden),
                "tile": _row_tile(hidden),
                "bias": "true",
                "pdl": "false",
            },
            "vision.rs: a block's norms",
        ),
        (
            attention_package(dtype),
            {
                "dtype": dtype,
                "kv": dtype,
                "out": dtype,
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
                "dtype": dtype,
                "output": dtype,
                "head_dim": str(head_dim),
                "heads": str(heads),
                "rotation": "half",
                "rows": "8",
                "block": "128",
                "pdl": "false",
            },
            "vision.rs: the tower's own rotation, host-computed factors",
        ),
        (
            "gelu/triton",
            {"dtype": dtype, "columns": str(ffn), "block": "1024", "approximate": "tanh",
             "pdl": "false"},
            "vision.rs: the MLP's activation",
        ),
        (
            # The merger's activation is a **different** GELU from the MLP's: the reference builds
            # `nn.GELU()` there and `gelu_pytorch_tanh` in the blocks (`vision.rs:49`), so the key
            # differs in `approximate` as well as in width - `merge² · hidden` before the first
            # projection, which is 4608 for this tower against the MLP's 4304.
            "gelu/triton",
            {
                "dtype": dtype,
                "columns": str(merge * merge * hidden),
                "block": "1024",
                "approximate": "exact",
                "pdl": "false",
            },
            "vision.rs: the patch merger's activation",
        ),
        (
            "scaled_residual/triton",
            {
                "dtype": dtype,
                "columns": str(hidden),
                "block": "1024",
                "update_bias": "false",
            },
            "vision.rs: the tower's residual stream at its own 1152",
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
                {"dtype": "f32", "channels": str(width), "block": "1024", "pdl": "false"},
                f"encoding.rs: a level's snake at {width} channels",
            )
        )
    calls.append(
        (
            "frame_moments/nvcc",
            {"dtype": "f32", "epsilon": f"0x{_f32_bits(vae['norm_eps']):08x}"},
            "encoding.rs: the frame statistics a pre-activated convolution reads",
        )
    )
    calls.append(
        (
            "frame_norm_pad/nvcc",
            {
                "dtype": "f32",
                # `encoding.rs:998` builds `FrameNormPad::new` and never calls
                # `with_channels_last`, so the frames are read channel-first (`frame.rs:150`).
                # `single_pixel` is the request's frame count, not a field this tool can state: a clip
                # of more than one frame is the encode path's case, and the fixture holds both rows.
                "channels_last": "false",
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


# The safetensors element-type spellings, and the words the descriptors use for them.
ELEMENTS = {"F32": "f32", "F16": "f16", "BF16": "bf16", "F64": "f64"}


def element_type(asked: str, directory: pathlib.Path) -> str:
    """The element type the text stack computes in.

    A descriptor key names it (`dtype_bf16_columns_5120`), and a build compiles one set: a list
    built at the wrong width checks rows the run never binds and misses the ones it does. The
    export's own answer is in its shards' headers; a host holding metadata only cannot read them, so
    `bf16` stands in and the assumption is printed rather than hidden.
    """
    if asked != "auto":
        return asked
    found = {}
    for path in sorted(directory.glob("*.safetensors")):
        try:
            import json
            import struct

            with path.open("rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(length))
        except OSError:
            continue
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            spelling = entry.get("dtype")
            if spelling is not None:
                found.setdefault(spelling, name)
    if not found:
        print(
            "dtype: the export's shards are absent here, so `bf16` is assumed; pass --dtype to"
            " check the set a build actually compiles",
            file=sys.stderr,
        )
        return "bf16"
    if len(found) > 1:
        print(
            "dtype: the export holds "
            + ", ".join(f"{spelling} (first {name})" for spelling, name in found.items())
            + "; pass --dtype to name the one this build compiles",
            file=sys.stderr,
        )
        return "bf16"
    spelling = next(iter(found))
    name = ELEMENTS.get(spelling)
    if name is None:
        print(f"dtype: an element type this tool does not know: {spelling}", file=sys.stderr)
        return "bf16"
    return name


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
                f"NOT EMITTED {package}: this tool does not model its field set, and the"
                f" fields it lacks are functions of the layers' own helpers - `row_tile` of the"
                f" width, `BLOCK`, `ROWS` - so a value computed here would be a second"
                f" implementation of `nn` [{reason}]"
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
    parser.add_argument(
        "--dtype",
        default="auto",
        help=(
            "the element type the text stack computes in: `auto` reads the export's "
            "`text_encoder` shard headers, and a host holding metadata only falls back to `bf16` "
            "with the assumption printed"
        ),
    )
    parser.add_argument("--kernels", default="crates/ops/kernels")
    parser.add_argument(
        "--ulysses",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "the Ulysses degree the deployment runs: the text stack's attention and its fused "
            "qk-norm rotation split over the degrees, so only the deployment's own keys are bound"
        ),
    )
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

    dtype = element_type(arguments.dtype, root / "text_encoder")
    text_config = json.loads(text_path.read_text())
    sections = [
        ("text encoder", text_calls(text_config["text_config"], dtype, arguments.ulysses)),
        ("vision tower", tower_calls(text_config["vision_config"], dtype)),
        # No video-encoder section: this tool derives the *text* stack's and the *tower*'s calls,
        # which are the two components this half owns. The video encoder normalizes with
        # `frame_moments` + `frame_norm_pad` rather than `layer_norm`, and its rows are generated by
        # `VideoEncoder::programs` - the same function the store's list is built from - so restating
        # them here produced four `layer_norm` calls the encoder never launches. A false miss is worse
        # than a covered silence: it sends a reader to add rows for an operator nothing binds.
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
