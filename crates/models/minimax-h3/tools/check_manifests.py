#!/usr/bin/env python3
"""Check the DiT's descriptor values against the package manifests that must declare them.

    .venv/bin/python crates/models/minimax-h3/tools/check_manifests.py \
        [--root <checkpoint>] [--kernels <package root>]

`--root` is a directory or a repository id, read from the local Hub cache; it defaults to
`MiniMaxAI/MiniMax-H3`, and `--kernels` to this repository's `crates/ops/kernels`.

A descriptor outside a package's `dimensions` is a key with no compiled program: `_compile.py`
accepts exactly the range the manifest declares, and `Generation::kernel` answers
`MissingKernel` for anything else. So a value the model builds and no manifest lists is a refusal
at the first call that takes it - and nothing host-side finds it by running the model, because a
deployment plan sizes buffers and never names a program.

This tool is the check that does. It reads the DiT's own `config.json` for the numbers, and for
each operator the block launches it asserts the values the crate passes into the descriptor are
declared by that operator's package. The `reason` column says which code path builds the value, so
a failure is actionable without reading the tool.

It is a diagnosis, not a gate on the branch: run it before a device run, and read what it prints.

Scope: the **dense** DiT of the export at `--root`. The sparse family (`vsa_*`) is not checked,
because the sparse deployment is a different export (the distilled student) and its constants are that
checkpoint's, not this one's; running this tool against a pipeline root that holds a sparse DiT would
make it a one-line extension of `expectations`.

Two things it deliberately does not do:
  * it does not enumerate tensor names - `Denoiser::keys` is that check, and `load` refuses on it;
  * it does not know about operators whose key carries no shape (`gemm`, `patch`,
    `unpatchify_video`, `philox_noise`, `rows`, the `vsa_*` family at one rank), because a
    manifest cannot be wrong about a value that is not in the key.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

#: This repository, found from this file's own place in it.
REPOSITORY = pathlib.Path(__file__).resolve().parents[4]


def model_dir(model: str) -> pathlib.Path:
    """A directory, or the local Hub cache's snapshot of the repository `model` names.

    A repository id may carry a revision after an `@`. Nothing is downloaded.
    """
    path = pathlib.Path(model)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    repo_id, _, revision = model.partition("@")
    return pathlib.Path(snapshot_download(repo_id, revision=revision or None, local_files_only=True))

# One `dimensions = { field = ["value", ...], ... }` line of a package manifest.
DIMENSIONS = re.compile(r"^dimensions\s*=\s*\{(.*)\}\s*$")


def manifest_values(path: pathlib.Path) -> dict[str, set[str]]:
    """Every value each descriptor field declares, over all the file's descriptors."""
    values: dict[str, set[str]] = {}
    for line in path.read_text().splitlines():
        match = DIMENSIONS.match(line.strip())
        if match is None:
            continue
        for field in re.finditer(r"(\w+)\s*=\s*\[([^\]]*)\]", match.group(1)):
            listed = {item.strip().strip('"') for item in field.group(2).split(",")}
            values.setdefault(field.group(1), set()).update(v for v in listed if v)
    return values


def expectations(config: dict, degrees: list[int]) -> list[tuple[str, str, str, str]]:
    """(package, field, value, reason) for every value this block passes into a descriptor."""
    heads = config["num_attention_heads"]
    head_dim = config["attention_head_dim"]
    hidden = config["hidden_size"]
    ffn = config["ffn_dim"]
    time_hidden = config["time_embed_hidden_dim"]
    # The timestep groups a step reads: the generated pair (§`t2va`), and the two conditioned
    # levels on top of it (`fl2va`, `ref2va`).
    groups = [2, 4]

    rows: list[tuple[str, str, str, str]] = []
    add = rows.append

    # The Ulysses degrees the deployments are meant to run. `Partition::validate` accepts any power of
    # two that divides the head count - for this DiT that is 1, 2, 4 and 8 - so a degree outside this
    # list is a legal partition that would refuse at its first attention launch; the caller is told so
    # once, below, rather than the tool failing for cells no deployment asks for.
    #
    # The attention is FP32 SGEMM only for the consumers that hold FP32 activations; the DiT's dense
    # attention is BF16 and binds `fa4`, whose key carries the rank's head counts.
    for degree in degrees:
        if heads % degree == 0:
            for field in ("q_heads", "kv_heads"):
                add(
                    (
                        "dense_attention/fa4",
                        field,
                        str(heads // degree),
                        f"attention.rs: the DiT's {heads} heads over {degree} Ulysses rank(s)",
                    )
                )
                add(
                    (
                        "qk_norm_rope/triton",
                        "q_heads" if field == "q_heads" else "k_heads",
                        str(heads // degree),
                        f"attention.rs: the rotation names this rank's {heads // degree} heads",
                    )
                )

    add(("qk_norm_rope/triton", "head_dim", str(head_dim), "attention.rs: one head's width"))
    add(("qk_norm_rope/triton", "rotated", "96", "attention.rs: the rotated channels of a head"))
    add(("rms_norm/triton", "columns", str(hidden), "conditioning.rs: the refiner's norms"))

    # The AdaLN path cuts its tables by column, so a descriptor names the table's row stride.
    for field in ("shift_stride", "scale_stride"):
        add(("modulated_rms_norm/triton", field, str(2 * hidden), "transformer.rs: the output norm"))
        add(("modulated_rms_norm/triton", field, str(6 * hidden), "transformer.rs: the six vectors"))
    add(("modulated_rms_norm/triton", "gate_stride", str(6 * hidden), "transformer.rs: the six vectors"))
    add(("modulated_rms_norm/triton", "width", str(hidden), "transformer.rs: the DiT's hidden width"))

    # The AdaLN gathers: the step→entry table at the group count, then two tables the model owns.
    for count in groups:
        add(("logits_gather/triton", "columns", str(count), "modulation.rs: the step→entry table"))
    for width, why in (
        (2 * hidden, "modulation.rs: the output table"),
        (18 * hidden, "modulation.rs: the per-layer table"),
    ):
        add(("logits_gather/triton", "columns", str(width), why))
    # The gathers of whole hidden rows: the video's and the audio's at every evaluation, and the
    # conditioner's prefix rows once per request. They are `logits_gather` launches, not a separate
    # operator, because `nn::layers::gather_rows` binds `LogitsGather` with the width in the key.
    add(
        (
            "logits_gather/triton",
            "columns",
            str(hidden),
            "denoiser.rs: the video, audio and prefix row gathers",
        )
    )
    # The packed rows' rotary factors: the DiT rotates three axes of 32 channels each (the attention's
    # `Sections::new([32, 32, 32])`), and the descriptor names the per-axis widths in its key.
    add(
        (
            "rotary_factors/triton",
            "sections",
            "32-32-32",
            "attention.rs: the packed rows' three rotary axes",
        )
    )
    add(("logits_gather/triton", "dtype", "i32", "modulation.rs: the step→entry gather is i32"))
    add(("logits_gather/triton", "columns", "1", "solver.rs: the coefficient table"))

    add(("gated_residual/triton", "width", str(hidden), "transformer.rs: the block's gate"))
    add(("gated_residual/triton", "gate_stride", str(6 * hidden), "transformer.rs: the block's gate"))
    add(("act_and_mul/triton", "width", str(ffn), "transformer.rs: the feed-forward's activation"))
    for field in ("gate_stride", "value_stride"):
        add(("act_and_mul/triton", field, str(2 * ffn), "transformer.rs: the packed gate and value"))
    add(("silu/triton", "columns", str(time_hidden), "modulation.rs: the cache's activation"))
    # `scaled_add` keys `target`, `source` and `block` - **not a width**. The row width is the
    # operand's: the kernel takes the extent as an argument and derives the width from the target's
    # own shape, so keying it would state rows for the clips a walk happened to name and leave every
    # other length with no kernel (`ops/src/residual.rs:654-659`). The three crossings that used to
    # be listed here - the timestep activation narrowed into the modulation type, a widened hidden
    # row and a widened patch row - are launch extents, and their keys are the two element types.
    for target, source, why in (
        ("bf16", "f32", "transformer.rs: a widened source into a narrowed hidden row"),
        ("f32", "bf16", "transformer.rs: a narrowed source widened into an f32 row"),
        ("bf16", "bf16", "modulation.rs: the timestep activation in the modulation type"),
    ):
        add(("scaled_add/triton", "target", target, why))
        add(("scaled_add/triton", "source", source, why))
    add(("scaled_add/triton", "block", "1024", "residual.rs: the elements one program walks"))
    add(("scaled_residual/triton", "columns", "1", "solver.rs: a scalar step coefficient"))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--kernels", default=REPOSITORY / "crates/ops/kernels")
    parser.add_argument(
        "--ulysses",
        default="1,2",
        help="the Ulysses degrees the deployments run, comma separated (default 1,2)",
    )
    arguments = parser.parse_args()
    degrees = [int(value) for value in arguments.ulysses.split(",") if value.strip()]

    root = model_dir(arguments.root)
    kernels = pathlib.Path(arguments.kernels)
    configs = sorted(root.glob("transformer*/config.json"))
    if not configs:
        print(f"no DiT config under {root}", file=sys.stderr)
        return 2

    missing = 0
    checked = 0
    for path in configs:
        config = json.loads(path.read_text())
        print(f"{path.parent.name}: {config['num_attention_heads']} heads, "
              f"{config['hidden_size']} wide, head width {config['attention_head_dim']}")
        legal = [
            degree
            for degree in (1, 2, 4, 8, 16)
            if degree & (degree - 1) == 0
            and config["num_attention_heads"] % degree == 0
        ]
        beyond = [degree for degree in legal if degree not in degrees]
        if beyond:
            print(
                f"  note: degrees {beyond} are legal partitions and are not checked; "
                f"a deployment at one would refuse at its first attention launch"
            )
        for package, field, value, reason in expectations(config, degrees):
            manifest = kernels / package / "package.toml"
            if not manifest.is_file():
                print(f"  NO MANIFEST {package}")
                missing += 1
                continue
            checked += 1
            declared = manifest_values(manifest).get(field, set())
            if value not in declared:
                missing += 1
                print(f"  MISSING {package}: {field} = {value}  [{reason}]")
    print(f"\n{checked} values checked, {missing} not declared")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
