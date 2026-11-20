r"""Reference vectors of DeepSeek-V4.1-Flash's DSpark drafter.

The values are computed by the checkpoint's own inference code,
`inference/model.py` (with its `kernel.py` and `engram.py`) of
deepseek-ai/DeepSeek-V4.1-Flash-1017, imported from the checkpoint and run
on one GPU without tensor parallelism (MP=1):

- The target's hidden states of a fixed prompt: `Transformer.forward` of
  the whole prompt at `start_pos=0` (one prefill, batch of one), whose
  `main_hidden` is the mean over the four hyper-connection streams of the
  residual entering each of the layers `dspark_target_layer_ids` (37, 38
  and 39), concatenated, `[n, 3 * dim]` BF16; and the argmax of the
  target's logits at every position.
- The drafter at those hidden states: the three DSpark stages (`mtp.0..2`,
  layer ids 40..42), built as `Transformer.__init__` builds them with the
  target's token embedding and head shared, driven by
  `Transformer.forward_spec` with teacher-forced anchors.

Protocol:

- Prompt (`--prompt`): the token ids `ids` of a prompt file written by
  `generate_logprobs.py` (its `prompt_1024.safetensors`: the chat-mode
  rendering of a documentation text, 968 tokens), used as they are; the
  file's `argmax` (its composition A at MP=4) is compared with this
  script's target argmax, position by position, which states how far the
  single-GPU reduction order moves the target's greedy choice.
- Target: one prefill of the n prompt tokens. `target_argmax[t]` is the
  argmax of the FP32 logits at position t (`sample` at temperature 0).
- Drafter, per group of `GROUPS`, with start p and S steps: a seeding call
  `forward_spec(ids[p], main_hidden[:, :p], start_pos=0)`, which writes
  each stage's sliding-window ring from the target hidden states of
  positions 0..p-1, then for each step s = 0..S-1 at
  `start_pos = p + s` the call `forward_spec(ids[start_pos + 1],
  main_hidden[:, start_pos:start_pos + 1], start_pos)`: the target hidden
  state of position start_pos enters the ring, and the anchor is the
  prompt's token at start_pos + 1 (teacher forcing in place of the
  target's own sample). The block of `dspark_block_size` (5) draft
  positions start_pos + 1 .. start_pos + 5 holds the anchor followed by
  the noise token; draft i (0-based) predicts the token at
  start_pos + 2 + i. Each step records `output_ids` (the anchor and the
  five greedy drafts, `sample` at temperature 0), the FP32 `logits` the
  drafts were chosen from (the head's logits with the Markov head's bias
  added in place, which `forward_head` returns), the confidence, and the
  rows the head normalizes (`hc_pre` of the last stage's output with its
  mix, the input of the last stage's `norm`, taken by a forward pre-hook).
  Group g0 (p = 16, S = 8) has the window ring still filling: a step reads
  `min(window_size, start_pos + 1)` of its entries. Group g1 (p = n - 48,
  S = 32) reads the full ring of 128 entries. The groups run in that
  order on one drafter; each seeding call rewrites every ring entry a
  later step reads.
- Model arguments: the release's `inference/config.json` with one
  sequence (`max_batch_size=1`), `max_seq_len=4096` (it sizes the caches
  and the rotary tables, whose rows do not depend on it), greedy sampling
  (`temperature=0`, which `DSparkBlock` reads as well) and the vision
  tower off (`vision_n_layers=0`).

What differs from `inference/generate.py`, and why:

- The weights are read from the released shards and mapped in memory as
  `inference/convert.py --model-parallel 1 --expert-dtype fp4` writes its
  one file (`generate_logprobs.rank_state` of rank 0 of 1: the renaming,
  `wo_a` dequantized to BF16 by its block scales and the int8 expert
  weights viewed as packed E2M1). The vision tower and the image tokens'
  routing bias (`gate.bias_vl`, built only with the vision tower) are not
  read: a text prompt does not reach them.
- The model does not fit one GPU, so the backbone is streamed:
  `Transformer.forward` is the reference's own function, called with a
  stand-in for `self` whose `layers` builds each `Block` on the GPU and
  loads its weights (`load_state_dict(strict=True)`) when the loop
  reaches it, and lets it go when the loop moves on. A block's output
  depends only on its weights, its input and the attention state its
  source layers publish (`model.shared_attn`, which keeps the source
  layer's cache tensors alive), so the values are the whole model's.
- The Engram tables (layers 1 and 14, 98 GB each) are not held whole: the
  prompt's hash ids are computed by the reference's `NgramHashState`, as
  `Transformer.forward` computes them, and each layer's
  `ParallelEngramEmbedding` is built over a compact table holding exactly
  the rows those ids name, read row range by row range from the shards.
  The lookup maps each hash id to its compact row by a sorted search and
  runs the reference's own forward (gather, dequantization by the E8M0
  scale of each 32 values, BF16) on it, so every looked-up row has the
  value the whole table gives.
- The head forms the logits of every position (`ParallelHead.forward`
  with `full_logits=True`) instead of the last one, so the reference's
  `sample` returns the target argmax at every position.
- `generate.py` does not run the drafter: the release implements the
  DSpark forward pass but no speculative-decoding loop. Here
  `Transformer.forward_spec` (the reference's function, its `self` a
  stand-in holding the three stages and `hc_mult`) is driven as the
  protocol states.
- With MP=1 no partial sums are reduced across ranks: a row-parallel
  projection's output is its GEMM's BF16 result and the routed experts'
  outputs accumulate in one FP32 sum. The rounding therefore differs from
  the prompt file's MP=4 values, which is what the argmax agreement
  measures.

Writes into `--out`:

- `vectors.safetensors`: `ids` [n] int32, `main_hidden` [n, 15360] BF16,
  `target_argmax` [n] int32; per group `<g>.start` [1] int32 (p),
  `<g>.positions` [S] int32 (start_pos of each step), `<g>.anchors` [S]
  int32 (`ids[start_pos + 1]`), `<g>.output_ids` [S, 6] int32,
  `<g>.logits` [S, 5, vocab] FP32, `<g>.confidence` [S, 5] FP32 and
  `<g>.head_rows` [S, 5, 5120] BF16, row s of each being step s;
- `manifest.json`: the release revision, the SHA-256 of the reference
  files, the environment and GPU, the checkpoint path, the model
  arguments' overrides, the prompt's source, n, the groups, the argmax
  agreement with the prompt file, the drafts' agreement with the
  teacher-forced tokens, the run times and the peak GPU memory.

Run from the repository root on a host with one GPU of at least 40 GiB
free, with an interpreter that has the reference's requirements
(`inference/requirements.txt`: torch, tilelang 0.1.12, safetensors,
transformers, sympy, numpy, Pillow)::

    mkdir -p artifacts/deepseek_v41/dspark-reference
    CUDA_VISIBLE_DEVICES=<gpu> MAX_JOBS=16 CUDA_HOME=<CUDA toolkit> \
        nice -n 19 python \
        crates/models/tests/it/deepseek_v41/generate_dspark.py \
        --checkpoint <DeepSeek-V4.1-Flash-1017> \
        --prompt artifacts/deepseek_v41/reference/prompt_1024.safetensors \
        --out artifacts/deepseek_v41/dspark-reference \
        2>&1 | tee artifacts/deepseek_v41/dspark-reference/generate.log

TileLang compiles the kernels with the `nvcc` of `CUDA_HOME`, which must be
a complete toolkit of the CUDA release of torch's build (see
`generate_logprobs.py`). The process caps its GPU memory at `GPU_MEMORY`.

`--layers K --tokens T` proves a setup cheaply: the first T prompt tokens
through the first K backbone layers, the residuals entering the last three
of them standing in for the target layers. Its vectors are not reference
values, and the manifest records the slice.
"""

import argparse
import dataclasses
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import safetensors
import torch
import transformers
from generate_logprobs import (
    REFERENCE_FILES,
    Checkpoint,
    converted,
    import_reference,
    rank_state,
    revision,
    sha256,
)
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

#: The drafter's groups: name, start p (negative counts from the end of the
#: prompt) and the number of steps S.
GROUPS = (("g0", 16, 8), ("g1", -48, 32))

#: The cache capacity in tokens; it sizes the caches and the rotary tables.
MAX_SEQ_LEN = 4096

#: The GPU memory this process's allocator may hold, in bytes.
GPU_MEMORY = 36 * 2**30

#: Released tensors no module built here holds: the vision tower and its
#: aligner, the learned image-span embeddings, and the image tokens' routing
#: bias (built only with the vision tower).
UNUSED = ("vision.", "aligner.", "image_")

START = time.monotonic()


def log(message: str) -> None:
    """Prints one progress line, prefixed by the seconds since the start."""
    print(f"[{time.monotonic() - START:7.1f} s] {message}", flush=True)


def module_groups(checkpoint: Checkpoint) -> dict[str, list[str]]:
    """The released tensors read here, by the module that holds them.

    Returns:
        Released names by group, `embed`, `norm`, `head`, `layers.<i>` and
        `mtp.<i>`, each in shard order.
    """
    groups = defaultdict(list)
    for source in sorted(checkpoint.shards, key=checkpoint.shards.get):
        name, _ = converted(source)
        if name.startswith(UNUSED) or name.endswith(".gate.bias_vl"):
            continue
        parts = name.split(".")
        depth = 2 if parts[0] in ("layers", "mtp") else 1
        groups[".".join(parts[:depth])].append(source)
    return groups


def module_state(
    checkpoint: Checkpoint, sources: list[str], prefix: str, n_experts: int
) -> dict[str, torch.Tensor]:
    """The tensors of one module, as `convert.py` maps them for MP=1.

    Args:
        checkpoint: The released shards.
        sources: The module's released tensor names.
        prefix: The module's name with its trailing dot, removed from the
            keys.
        n_experts: The module's routed experts (unused without experts).

    Returns:
        The module's state by parameter name, on the host. The shards are
        closed afterwards, so their memory maps do not accumulate.
    """
    # Read on the host whatever the default device: a shard's tensors are
    # created on the default device.
    with torch.device("cpu"):
        state = rank_state(checkpoint, sources, 0, 1, n_experts)
    checkpoint.files.clear()
    return {name.removeprefix(prefix): t for name, t in state.items()}


def runs(rows: list[int]) -> list[tuple[int, int]]:
    """The maximal ranges `[start, stop)` of consecutive sorted `rows`."""
    ranges = []
    start = previous = rows[0]
    for row in rows[1:]:
        if row != previous + 1:
            ranges.append((start, previous + 1))
            start = row
        previous = row
    ranges.append((start, previous + 1))
    return ranges


def compact_table(
    checkpoint: Checkpoint, layer_id: int, ids: torch.Tensor, rows: int
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The rows of one Engram table that the hash ids `ids` name.

    Args:
        checkpoint: The released shards.
        layer_id: The Engram layer.
        ids: The layer's hash ids, table rows of any shape.
        rows: The table's rows in the layout the ids were hashed for.

    Returns:
        The sorted distinct table rows named, int64, and the compact table
        `weight` [m, head_dim] E4M3 and `scale` [m, head_dim / 32] E8M0 of
        those rows in that order, on the host.
    """
    named = torch.unique(ids.flatten().cpu())
    ranges = runs(named.tolist())
    table = {}
    for part in ("weight", "scale"):
        name = f"layers.{layer_id}.engram.embed.{part}"
        whole = checkpoint.file(name).get_slice(name)
        assert whole.get_shape()[0] == rows, (name, whole.get_shape(), rows)
        # On the host: a slice read on a GPU default device would first
        # place the whole table there.
        with torch.device("cpu"):
            parts = [whole[start:stop] for start, stop in ranges]
            table[part] = torch.cat(parts)
        assert table[part].size(0) == named.numel(), name
    checkpoint.files.clear()
    return named, table


class CompactTable(torch.nn.Module):
    """An Engram table restricted to the rows a prompt's hash ids name.

    Holds the reference's `ParallelEngramEmbedding` over those rows and the
    sorted table row each compact row stands for. A lookup maps the hash
    ids to their compact rows and runs the reference's forward on them.
    """

    def __init__(self, table: torch.nn.Module, rows: torch.Tensor) -> None:
        super().__init__()
        self.table = table
        self.rows: torch.Tensor
        self.register_buffer("rows", rows, persistent=False)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """The reference's lookup of the table rows `indices` (int64).

        Raises:
            AssertionError: When an index is not a row of the compact table.
        """
        last = self.rows.numel() - 1
        local = torch.searchsorted(self.rows, indices).clamp_max(last)
        assert torch.equal(self.rows[local], indices), "a row outside it"
        return self.table(local)


class StreamedLayers:
    """`Transformer.layers` for one pass of its forward loop.

    Iterating builds block i on the GPU, loads its weights and yields it.
    The loop's own variable holds the previous block while the next one
    loads, so two blocks are resident at a time; the buffers the reference
    publishes in `shared_attn` (a source layer's compressed KV and index
    keys) stay alive through that reference for the later layers that read
    them.
    """

    def __init__(self, count: int, build) -> None:
        self.count = count
        self.build = build

    def __iter__(self):
        for layer_id in range(self.count):
            yield self.build(layer_id)


def agreement(drafts: torch.Tensor, expected: torch.Tensor) -> dict:
    """How the greedy drafts of a group agree with the teacher-forced ids.

    Args:
        drafts: [S, 5] the drafts of each step.
        expected: [S, 5] the prompt's tokens they predict.

    Returns:
        The fraction of steps whose draft i is the prompt's token, per i,
        and the mean length of the agreeing prefix of the drafts.
    """
    hits = drafts == expected
    accepted = hits.int().cumprod(dim=1).sum(dim=1)
    return {
        "draft_is_next": hits.double().mean(dim=0).tolist(),
        "mean_accepted_prefix": accepted.double().mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/deepseek_v41/dspark-reference"),
    )
    parser.add_argument(
        "--layers", type=int, help="run only the first LAYERS backbone layers"
    )
    parser.add_argument(
        "--tokens", type=int, help="run only the first TOKENS prompt tokens"
    )
    arguments = parser.parse_args()
    root = arguments.checkpoint

    # `generate.py`'s process setup, with this process's share of the GPU
    # capped.
    reference, tilelang, _ = import_reference(root)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY / total)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.set_default_device("cuda")

    config = json.loads((root / "inference/config.json").read_text())
    overrides = {
        "max_batch_size": 1,
        "max_seq_len": MAX_SEQ_LEN,
        "temperature": 0.0,
        "vision_n_layers": 0,
    }
    args = reference.ModelArgs(**{**config, **overrides})

    # The module globals `Transformer.__init__` sets: one process, no
    # tensor parallelism, FP8 storage of the quantized projections.
    reference.world_size = 1
    reference.rank = 0
    reference.default_dtype = (
        torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    )

    # The prompt and the target argmax it was written with.
    with torch.device("cpu"):
        prompt = load_file(arguments.prompt)
    prompt_records = json.loads(
        (arguments.prompt.parent / "manifest.json").read_text()
    )
    prompt_record = next(
        record
        for record in prompt_records["prompts"]
        if record["file"] == arguments.prompt.name
    )
    ids = prompt["ids"].long()[: arguments.tokens].cuda()
    n = ids.numel()
    assert n <= args.max_seq_len, n
    layers = arguments.layers or args.n_layers
    sliced = layers < args.n_layers or n < prompt["ids"].numel()
    target_layer_ids = tuple(args.dspark_target_layer_ids)
    if layers < args.n_layers:
        target_layer_ids = tuple(range(layers - 3, layers))
    groups = [(name, start % n, steps) for name, start, steps in GROUPS]
    for name, start, steps in groups:
        # The last step's drafts predict up to the token at start + steps + 5.
        assert 0 < start and start + steps + 6 <= n, (name, start, steps, n)
    log(f"prompt {arguments.prompt}: {n} tokens, {layers} layers")

    checkpoint = Checkpoint(root)
    names = module_groups(checkpoint)
    expected = {"embed", "norm", "head"}
    expected |= {f"layers.{i}" for i in range(args.n_layers)}
    expected |= {f"mtp.{i}" for i in range(args.n_mtp_layers)}
    assert set(names) == expected, sorted(set(names) ^ expected)

    # The prompt's hash ids, as `Transformer.forward` computes them, and the
    # Engram tables' rows they name.
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(root)
    layout = reference.EngramLayout.from_args(args)
    engram_hash = reference.NgramHashState(args, layout, tokenizer)
    hashes = engram_hash(ids[None], 0)
    tables = {}
    for index, layer_id in enumerate(layout.layer_ids):
        if layer_id < layers:
            tables[layer_id] = compact_table(
                checkpoint,
                layer_id,
                hashes[:, :, index],
                layout.num_embeddings[index],
            )
            log(
                f"layer {layer_id}: {tables[layer_id][0].numel()} Engram "
                f"rows of {layout.num_embeddings[index]}"
            )
    compact_layout = dataclasses.replace(
        layout,
        num_embeddings=tuple(
            tables[layer_id][0].numel() if layer_id in tables else rows
            for layer_id, rows in zip(
                layout.layer_ids, layout.num_embeddings, strict=True
            )
        ),
    )
    hash_seconds = time.monotonic() - start

    def build(layer_id: int):
        """Backbone block `layer_id` on the GPU with its weights loaded."""
        began = time.monotonic()
        # The loop runs under `forward`'s inference mode; the block is built
        # and loaded as ordinary parameters, as a loaded model's are.
        with torch.inference_mode(False):
            block = reference.Block(layer_id, args, compact_layout)
            prefix = f"layers.{layer_id}."
            sources = [
                name
                for name in names[f"layers.{layer_id}"]
                if ".engram.embed." not in name
            ]
            n_experts, _ = args.get_moe_config(layer_id)
            state = module_state(checkpoint, sources, prefix, n_experts)
            if layer_id in tables:
                rows, table = tables[layer_id]
                state["engram.embed.weight"] = table["weight"]
                state["engram.embed.scale"] = table["scale"]
            size = sum(t.numel() * t.element_size() for t in state.values())
            block.load_state_dict(state, strict=True)
            del state
            if layer_id in tables:
                block.engram.embed = CompactTable(
                    block.engram.embed, rows.cuda()
                )
        log(
            f"layer {layer_id}: {size / 2**30:.2f} GiB in "
            f"{time.monotonic() - began:.1f} s, GPU "
            f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB"
        )
        return block

    def top_level(module, name: str):
        """`module` with the weights of the top-level group `name`."""
        state = module_state(checkpoint, names[name], f"{name}.", 0)
        module.load_state_dict(state, strict=True)
        return module

    # The target: `Transformer.forward` of the whole prompt, its blocks
    # streamed, its head forming the logits of every position.
    start = time.monotonic()
    embed = top_level(
        reference.ParallelEmbedding(args.vocab_size, args.dim), "embed"
    )
    norm = top_level(reference.RMSNorm(args.dim, args.norm_eps), "norm")
    head = top_level(
        reference.ParallelHead(
            args.vocab_size, args.dim, args.norm_eps, args.hc_eps
        ),
        "head",
    )
    target = SimpleNamespace(
        engram_hash=engram_hash,
        embed=embed,
        layers=StreamedLayers(layers, build),
        target_layer_ids=target_layer_ids,
        hc_mult=args.hc_mult,
        norm=norm,
        head=lambda x: head(x, full_logits=True),
        temperature=args.temperature,
    )
    target_argmax, _, main_hidden = reference.Transformer.forward(
        target, ids[None], 0
    )
    torch.cuda.synchronize()
    target_seconds = time.monotonic() - start
    target_argmax = target_argmax[0].int().cpu()
    shape = tuple(main_hidden.shape)
    log(f"target: {target_seconds:.0f} s, main_hidden {shape}")

    # The drafter: the three DSpark stages with the target's embedding and
    # head, as `Transformer.__init__` builds them.
    start = time.monotonic()
    mtp = torch.nn.ModuleList()
    for stage in range(args.n_mtp_layers):
        layer_id = args.n_layers + stage
        block = reference.DSparkBlock(layer_id, args)
        n_experts, _ = args.get_moe_config(layer_id)
        state = module_state(
            checkpoint, names[f"mtp.{stage}"], f"mtp.{stage}.", n_experts
        )
        block.load_state_dict(state, strict=True)
        del state
        block.embed = embed
        block.head = head
        mtp.append(block)
    drafter = SimpleNamespace(mtp=mtp, hc_mult=args.hc_mult)
    head_rows = []
    mtp[-1].norm.register_forward_pre_hook(
        lambda module, inputs: head_rows.append(inputs[0])
    )
    drafter_load_seconds = time.monotonic() - start
    log(
        f"drafter: loaded in {drafter_load_seconds:.0f} s, GPU "
        f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB"
    )

    tensors = {
        "ids": ids.int().cpu(),
        "main_hidden": main_hidden[0].cpu(),
        "target_argmax": target_argmax,
    }
    summaries = []
    group_seconds = {}
    for name, start_pos_0, steps in groups:
        start = time.monotonic()
        seeded = reference.Transformer.forward_spec(
            drafter,
            ids[start_pos_0 : start_pos_0 + 1],
            main_hidden[:, :start_pos_0],
            0,
        )
        assert seeded is None
        head_rows.clear()
        records = defaultdict(list)
        for start_pos in range(start_pos_0, start_pos_0 + steps):
            anchor = ids[start_pos + 1 : start_pos + 2]
            output_ids, logits, confidence = reference.Transformer.forward_spec(
                drafter,
                anchor,
                main_hidden[:, start_pos : start_pos + 1],
                start_pos,
            )
            assert output_ids[0, 0] == anchor[0]
            records["output_ids"].append(output_ids[0].int().cpu())
            records["logits"].append(logits[0].cpu())
            records["confidence"].append(confidence[0].cpu())
        assert len(head_rows) == steps, len(head_rows)
        torch.cuda.synchronize()
        group_seconds[name] = time.monotonic() - start

        # The recorded tensors are on the host (the default device is the
        # GPU's).
        positions = torch.arange(start_pos_0, start_pos_0 + steps, device="cpu")
        group = {key: torch.stack(rows) for key, rows in records.items()}
        group["head_rows"] = torch.cat(head_rows).cpu()
        group["start"] = torch.tensor(
            [start_pos_0], dtype=torch.int32, device="cpu"
        )
        group["positions"] = positions.int()
        group["anchors"] = tensors["ids"][positions + 1]
        tensors.update({f"{name}.{key}": t for key, t in group.items()})

        following = positions[:, None] + 2 + torch.arange(5, device="cpu")
        summary = {
            "name": name,
            "start": start_pos_0,
            "steps": steps,
            "positions": [start_pos_0, start_pos_0 + steps - 1],
            "window_entries_read": [
                min(args.window_size, int(p) + 1)
                for p in (positions[0], positions[-1])
            ],
            **agreement(group["output_ids"][:, 1:], tensors["ids"][following]),
        }
        summaries.append(summary)
        log(f"{name}: {group_seconds[name]:.1f} s, {json.dumps(summary)}")

    # The target argmax against the prompt file's (MP=4) and the prompt.
    target_summary = {
        "argmax_is_next": (
            (target_argmax[:-1] == tensors["ids"][1:]).double().mean().item()
        ),
    }
    if layers == args.n_layers:
        theirs = prompt["argmax"][: n - 1]
        same = target_argmax[: n - 1] == theirs
        target_summary["argmax_agreement_with_prompt_file"] = {
            "positions": n - 1,
            "agreeing": int(same.sum()),
            "fraction": same.double().mean().item(),
            "disagreeing_positions": (~same).nonzero()[:, 0].tolist(),
        }
    log(f"target: {json.dumps(target_summary)}")

    arguments.out.mkdir(parents=True, exist_ok=True)
    release = revision(root)
    metadata = {
        "release": f"deepseek-ai/DeepSeek-V4.1-Flash-1017@{release}",
        "prompt": prompt_record["source"],
    }
    tensors = {key: t.contiguous() for key, t in tensors.items()}
    save_file(tensors, arguments.out / "vectors.safetensors", metadata)

    manifest = {
        "release": {
            "repository": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
            "revision": release,
        },
        "checkpoint": str(root.resolve()),
        "reference_sha256": {
            name: sha256(root / name) for name in REFERENCE_FILES
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "tilelang": tilelang.__version__,
            "safetensors": safetensors.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(),
        },
        "model_parallel": 1,
        "model_args_overrides": overrides,
        "slice": (
            {
                "layers": layers,
                "tokens": n,
                "target_layer_ids": target_layer_ids,
            }
            if sliced
            else None
        ),
        "prompt": {
            "file": str(arguments.prompt.resolve()),
            "sha256": sha256(arguments.prompt),
            "source": prompt_record["source"],
            "target": prompt_record["target"],
            "tokens": n,
            "text": prompt_record.get("text"),
            "argmax": (
                "composition A of generate_logprobs.py at MP="
                f"{prompt_records['model_parallel']}"
            ),
        },
        "protocol": {
            "target": "Transformer.forward of the whole prompt, start_pos=0",
            "main_hidden": (
                "concatenated mean over the hyper-connection streams of the "
                f"residual entering layers {list(target_layer_ids)}"
            ),
            "drafter": (
                "Transformer.forward_spec: seeded at start_pos=0 with "
                "main_hidden[:, :start], then per step at start_pos the "
                "anchor ids[start_pos + 1] and main_hidden[:, start_pos]"
            ),
            "sampling": "greedy (temperature 0)",
            "head_rows": "hc_pre of the last stage, the input of its norm",
            "groups": summaries,
        },
        "target": target_summary,
        "seconds": {
            "hashes_and_engram_rows": hash_seconds,
            "target": target_seconds,
            "drafter_load": drafter_load_seconds,
            "groups": group_seconds,
            "total": time.monotonic() - START,
        },
        "peak_gpu_memory": {
            "allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        },
    }
    path = arguments.out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
