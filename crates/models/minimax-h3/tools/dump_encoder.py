#!/usr/bin/env python3
"""Dump the text encoder's own output for one request, from today's implementation.

The reference half of the encoder stage's comparison. The candidate's half is
`crates/models/minimax-h3/examples/encoder.rs`, which writes its prompt's hidden rows for the same
case; this reads the candidate's case and writes the reference's rows for it, so the comparison is
about arithmetic rather than about which side built the prompt.

    .venv/bin/python crates/models/minimax-h3/tools/dump_encoder.py --probe prompt_hidden \
        --stack artifacts/h3-stage4/text_encoder --checkpoint /opt/uv/models/MiniMax-H3 \
        --out artifacts/h3-stage4-ref --device cuda

One file per invocation, so a failure costs one tensor:

    <out>/text_encoder/prompt_hidden.h3t        [tokens, hidden]  the checkpoint's dtype

`vision_destinations` - where the tower's rows land among these - is written once a case with images
exists: this probe refuses a case that asks for vision tokens rather than encoding the prompt as if
it had none, because a silent fallback there is a splice that never happened.

The case is the candidate harness's own statement of the request, exactly as for the other stages:

    {"task": "t2va", "text_tokens": 15, "hidden": 5120, "prompt": "...", "prompt_ids": [...],
     "vision": "none"}

`prompt_ids` is the tokenizer's answer, carried as data rather than re-derived here: the tokenizer is
one variable, and a tokenizer disagreement must be able to appear as itself instead of as a
difference in every row. The `prompt` string beside it is the cross-check: this probe tokenizes it
with the **reference's** tokenizer and refuses on a mismatch, naming the first index where the two
lists differ. The crates have no tokenizer of their own, so without that check a front end's
tokenizer would be invisible until a page disagreed in every row. `vision` names what the tower is asked to encode; `"none"` is a prompt with
no images, which is the cheap case this probe is meant to answer first.

It runs on the reference's own weight path - the `text_encoder` component's shards through
`uniserve.loading.loader.load_weights` with the mapping `weights._text_component` builds - so a
discrepancy it reports is about the mathematics and not about a different way of reading the
checkpoint.

This is a **diagnosis, never an acceptance gate** (`specs/worker.md` 10).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The probes this script produces. One so far: the whole prompt's hidden rows, which is the tensor a
# request's first step hands the denoiser's conditioner.
CATALOGUE = {
    "prompt_hidden": (
        "run host: the reference's text encoder on the candidate's own prompt ids"
    ),
}

# The first and last ids printed. An ids list of the right length in the wrong order is the
# encoder's own version of a permuted draw: no shape refuses it, so the page carries the numbers a
# reader needs to compare the two sides on the prompt as well as on the tensor.
ID_SAMPLE = 6


def reference(name: str):
    """Import one of the reference's modules, naming a missing compiled extension as the series does."""
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError as error:
        print(f"IMPORT FAILED: {type(error).__name__}: {error}")
        print(
            "  the reference's compiled extensions are not built in this environment"
        )
        raise SystemExit(1) from error


def read_tensor(path: Path):
    """Read one H3D1 tensor: the container this directory writes and the Rust comparator reads."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dump_tables import load

    try:
        return load(path)
    except ValueError as error:
        raise SystemExit(str(error)) from error


def dump(out: Path, name: str, values) -> None:
    """Write one tensor in the container `examples/compare_blocks.rs` reads."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dump_tables import dump as write

    write(out, name, values)


def encoder_case(case: dict) -> dict:
    """The case fields this probe reads, refused by name when the case omits one.

    A field read under another name would be a second convention for the same number, so every field
    the probe uses is required and its absence is reported rather than defaulted.
    """
    required = ("task", "text_tokens", "hidden", "prompt_ids")
    missing = [name for name in required if name not in case]
    if missing:
        raise SystemExit(
            f"the case omits {', '.join(missing)}; an encoder case carries "
            f"{', '.join(required)}"
        )
    if case["task"] != "t2va":
        raise SystemExit(
            f"the case's task is {case['task']!r}; this probe encodes a prompt and no"
            f" conditions, which is a `t2va` request"
        )
    tokens = int(case["text_tokens"])
    ids = list(case["prompt_ids"])
    # The prompt's length is the layout's content count: a longer list would encode tokens the
    # request cannot carry, and a shorter one would encode a different prompt. Both numbers, so the
    # reader can see which side moved.
    if len(ids) != tokens:
        raise SystemExit(
            f"the case's prompt_ids hold {len(ids)} ids and its text_tokens is {tokens};"
            f" the two are the same number and the case says they are not"
        )
    if not ids:
        raise SystemExit("the case's prompt_ids is empty; a request has a prompt")
    return case


def tokenizer_check(
    case: dict, checkpoint: Path, ids: list[int]
) -> None:
    """Tokenize the case's prompt with the reference's own tokenizer and refuse a mismatch.

    The ids are data because the crates have no tokenizer; the string is the cross-check, so a
    disagreement between a front end's tokenizer and the checkpoint's appears as itself, named at
    the first index where the two lists part, rather than as a difference in every row of the
    tensor.
    """
    prompt = case.get("prompt")
    if prompt is None:
        print("prompt  no string in the case; the ids are taken on trust")
        return
    if not isinstance(prompt, str):
        raise SystemExit(
            f"the case's prompt is {type(prompt).__name__}; a prompt is a string"
        )
    directory = checkpoint / "tokenizer"
    if not directory.is_dir():
        raise SystemExit(
            f"{directory} is absent, and the case carries a prompt to cross-check; point"
            f" --checkpoint at the export's root"
        )
    # The repository root, so `uniserve` resolves when the package is not installed editable: the
    # tokenizer is the one part of the reference this probe can exercise without a device.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    processing = reference("uniserve.processing")
    tokenizer = processing.load_tokenizer(directory)
    tokenized = [int(value) for value in tokenizer(prompt)["input_ids"]]
    if tokenized == ids:
        print(f"prompt  the tokenizer agrees on all {len(ids)} ids")
        return
    for index, (mine, theirs) in enumerate(zip(ids, tokenized, strict=False)):
        if mine != theirs:
            raise SystemExit(
                f"the case's prompt_ids and the checkpoint's tokenizer differ at index"
                f" {index}: the case has {mine} and the tokenizer {theirs}"
            )
    raise SystemExit(
        f"the case's prompt_ids hold {len(ids)} ids and the checkpoint's tokenizer"
        f" answers {len(tokenized)}"
    )


def text_source(checkpoint: Path):
    """The checkpoint source of the text encoder's own shards."""
    from uniserve.loading import checkpoint as checkpoint_module

    directory = checkpoint / "text_encoder"
    files = tuple(sorted(directory.glob("*.safetensors")))
    if not files:
        index = directory / "diffusion_pytorch_model.safetensors.index.json"
        hint = (
            f" ({index.name} lists them, so the shards are not fetched)"
            if index.is_file()
            else ""
        )
        raise SystemExit(
            f"{directory} holds no weight file{hint}; this probe runs the reference's own"
            f" arithmetic and needs the weights the serving loader fetches"
        )
    return checkpoint_module.Source("text_encoder", root=directory, files=files)


def encoder_sidecars(checkpoint: Path):
    """The checkpoint's typed configs, through the reference's own loading boundary."""
    h3_config = reference("uniserve.models.minimax_h3.config")

    try:
        return h3_config.read_config(checkpoint, io=None, sources=())
    except Exception as error:  # noqa: BLE001 - the first line of a bad checkpoint path
        print(f"CONFIG FAILED: {type(error).__name__}: {error}")
        raise SystemExit(1) from error


def load_encoder(case: dict, checkpoint: Path, device: str):
    """The reference's text encoder loaded through its own loader, at the checkpoint's dtype."""
    torch = reference("torch")
    encoder_module = reference("uniserve.models.minimax_h3.encoder")
    weights_module = reference("uniserve.models.minimax_h3.weights")
    from uniserve.loading import loader
    from uniserve.loading import weights as weight_options

    sidecars = encoder_sidecars(checkpoint)
    model = encoder_module.TextEncoder(sidecars.text_encoder)
    source = text_source(checkpoint)
    print(f"load    {source.name} from {source.root} ({len(source.files)} file(s))")
    mapping = weights_module._text_component(model)
    reports = loader.load_weights(
        model,
        (source,),
        mapping=lambda _root: (mapping,),
        device=device,
        weights=weight_options.Config(dtype=torch.bfloat16),
    )
    for report in reports:
        print(
            f"loaded  {len(report.loaded)} parameter(s); missing {len(report.missing)}, "
            f"unexpected {len(report.unexpected)}, incomplete {len(report.incomplete)}"
        )
        for name in sorted(report.missing)[:8]:
            print(f"missing  {name}")
    return model


def run_prompt_hidden(
    stack: Path, checkpoint: Path, out: Path, device: str
) -> int:
    """Encode the candidate's own prompt with the reference's text encoder.

    The page carries the prompt as well as the tensor: the ids' length, the vocabulary it fits and
    its first and last few values, because a list of the right length in the wrong order is the one
    defect neither a shape nor a dtype can refuse.

    What it does not prove: the vision tower's rows (a case with images is refused rather than
    encoded without them), the conditioner that reads this output, and anything after it.
    """
    case = encoder_case(json.loads((stack / "case.json").read_text()))
    ids = [int(value) for value in case["prompt_ids"]]
    print(f"prompt  {len(ids)} ids, first {ids[:ID_SAMPLE]}, last {ids[-ID_SAMPLE:]}")
    tokenizer_check(case, checkpoint, ids)
    vision = case.get("vision", "none")
    if vision not in (None, "none"):
        raise SystemExit(
            f"the case asks for {vision!r}; this probe encodes a prompt with no vision tokens,"
            f" and a case with images needs the tower's pixels and its own row ranges"
        )

    torch = reference("torch")
    model = load_encoder(case, checkpoint, device)
    width = int(model.network.hidden_size)
    if int(case["hidden"]) != width:
        raise SystemExit(
            f"the case's hidden size is {case['hidden']} and the checkpoint's is {width}"
        )
    if not 0 <= min(ids) and max(ids) < int(case.get("vocab_size", model.config.vocab_size)):
        raise SystemExit(
            f"the case's ids run from {min(ids)} to {max(ids)}, outside the checkpoint's"
            f" vocabulary of {model.config.vocab_size}"
        )
    tokens = torch.tensor(ids, dtype=torch.int64, device=device)
    # The M-RoPE coordinates of the prompt: the reference's own `rope_index` over these ids, with no
    # vision blocks to place. Without them every axis would take the text coordinates, which differ
    # from the reference's for the audio and video axes of a real prompt.
    positions = model.positions(ids)
    print(f"rope    {tuple(positions.shape)} coordinates, every axis the prompt's own")
    with torch.inference_mode():
        encoded = model.encode((tokens,), positions=(positions,))
    if encoded is None:
        raise SystemExit(
            "the encoder answered None, which is what it does on a pipeline stage after the"
            " first; this probe builds one stage"
        )
    hidden = encoded[0]
    if tuple(hidden.shape) != (len(ids), width):
        raise SystemExit(
            f"the encoder answered {tuple(hidden.shape)}; a prompt of {len(ids)} ids is"
            f" {(len(ids), width)}"
        )
    print(f"got     {tuple(hidden.shape)} {hidden.dtype}")
    (out / "text_encoder").mkdir(parents=True, exist_ok=True)
    dump(out, "text_encoder/prompt_hidden", hidden)
    (out / "text_encoder" / "case.json").write_text(
        (stack / "case.json").read_text()
    )
    print(f"wrote   {out / 'text_encoder' / 'prompt_hidden.h3t'}")
    print(
        "compare cargo run -p minimax-h3 --example compare_blocks --"
        f" --reference {out}"
        f" --candidate {stack.parent if stack.name == 'text_encoder' else stack}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--list", action="store_true", help="print the probe catalogue and exit"
    )
    parser.add_argument(
        "--probe", choices=sorted(CATALOGUE), help="the probe to run"
    )
    parser.add_argument(
        "--stack",
        type=Path,
        help=(
            "the candidate's stage directory, holding `case.json`"
            " (`--probe prompt_hidden`)"
        ),
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
    options = parser.parse_args(argv)

    if options.list:
        for name, description in CATALOGUE.items():
            print(f"{name}\t{description}")
        return 0
    if not (options.probe and options.out):
        parser.error("--probe and --out are required unless --list")
    print(f"probe   {options.probe} -> {CATALOGUE[options.probe]}")
    if options.stack is None or options.checkpoint is None:
        parser.error(
            f"--probe {options.probe} needs --stack for the candidate's tree and"
            f" --checkpoint for the weights it runs"
        )
    case_path = options.stack / "case.json"
    if not case_path.is_file():
        raise SystemExit(
            f"{case_path} is absent; --stack names the candidate's `text_encoder` directory,"
            f" which the encoder harness writes `case.json` into"
        )
    options.out.mkdir(parents=True, exist_ok=True)
    return run_prompt_hidden(
        options.stack, options.checkpoint, options.out, options.device
    )


if __name__ == "__main__":
    raise SystemExit(main())
