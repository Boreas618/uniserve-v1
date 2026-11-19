"""Reference next-token log-probabilities of DeepSeek-V4.1-Flash.

The values are computed by the checkpoint's own inference code,
`inference/model.py` (with its `kernel.py` and `engram.py`) of
deepseek-ai/DeepSeek-V4.1-Flash-1017, imported from the checkpoint and run
unmodified with its tensor parallelism over four GPUs (MP=4). Two things
differ from `inference/generate.py`:

- The weights are read from the released shards and mapped to each rank in
  memory exactly as `inference/convert.py --model-parallel 4 --expert-dtype
  fp4` writes its rank files: its renaming, its sharding along the
  dimensions of its `mapping`, the routed experts by index range, the
  Engram table rows in `ceil(rows / MP)` blocks padded with 0 (scales 1),
  `wo_a` dequantized to BF16 by its block scales after sharding and the
  int8 expert weights viewed as packed E2M1. Each rank reads only its share
  and loads it with `load_state_dict` layer by layer; the union of the
  layers' tensors is checked to be exactly the model's parameters, which is
  `strict=True` without holding a rank's 120 GB on the host at once. The
  DSpark draft stages (`mtp.*`) and the vision tower are not built
  (`dspark_block_size=0`, `vision_n_layers=0`): a text prompt does not
  reach them, so their tensors and the image routing bias are not read.
- The head's product runs over every position instead of the last one
  (`ParallelHead.forward` with `full_logits=True`), in blocks of `ROWS`
  positions: the head is handed the normalized hidden rows, and each
  block's logits are its own `F.linear` of the FP32 rows and the gather
  over the ranks, so a prompt of 128K positions never holds its
  `[n, vocab]` FP32 logits at once. The logits are the reference's FP32
  head output.
- A prefill of more than `QUERIES` tokens scores its index queries in
  blocks of `QUERIES` queries (`Indexer.forward`, the reference's code
  with its scoring, masking, candidate blocks and top-k taken per block of
  query rows, which every query computes alone), so a 128K-token prompt
  never holds its `[n, heads, n]` scores; the candidate blocks layer 20
  hands the later indexers are kept per block (the first column of each
  block of its mask, which the mask repeats) and repeated when read.
  The hyper-connections' collapse and expansion (`Block.hc_pre`,
  `Block.hc_post`), which multiply each position's streams by its own mix
  in FP32 and sum over the four streams, run the reference's expression
  per block of `POSITIONS` positions, the same arithmetic for every
  element without the `[n, 4, 4, dim]` FP32 products of a 128K-token
  prompt. Shorter prompts run the reference's indexer as it is, and
  `--check-blocked` runs a prompt's composition A a second time with the
  indexer whole and states how far the two lie apart.

Protocol:

- Prompts (`--sources`): four texts of this repository's documentation,
  one source each, `docs/expert_parallelism.md`, `docs/configuration.md`,
  `docs/architecture.md` and `docs/architecture.md` followed by
  `docs/configuration.md` (a source of several files joins them, each
  without its leading and trailing newlines, by a blank line), read from
  the working tree with each file's SHA-256 recorded. Each text is cut at
  the last blank line (paragraph boundary) whose prefix encodes to at
  most its target, 1024, 4096, 8192 and 16384 tokens in that order. The cut
  text is stored in the manifest, so a comparison sends exactly that text.
- Prompts (`--agentx`): AgentX payloads of the InferenceX corpus
  `semianalysisai/cc-traces-weka-062126` (the request bodies AIPerf sends,
  `payloads.jsonl`, and vLLM's `deepseek_v41` token ids of each,
  `vllm.jsonl`, of one directory): for each target of `--targets`, the
  payload of the most prompt tokens at or below it (`vllm_counts.jsonl`).
  Its ids are vLLM's, checked to be the reference encoder's rendering of
  its messages (thinking mode at the encoder's default reasoning effort,
  which vLLM's renderer applies), and the manifest keeps its messages, so
  a comparison sends exactly them.
- Encoding: the reference's chat mode for one user message, as
  `encoding/encoding.py` renders it (`thinking_mode="chat"`): the
  beginning-of-sentence token, `<｜User｜>`, the text, `<｜Assistant｜>`
  and `</think>`; tokenized by the checkpoint's tokenizer without added
  special tokens.
- Composition A: one prefill of the whole prompt of n tokens, alone (batch
  of one, `start_pos=0`). For every position t < n - 1, the FP32
  log-softmax of the logits at t gives the log-probability of the next
  token `ids[t + 1]`, the 64 most probable ids with their log-probabilities
  (descending), the log-probability of the rest of the vocabulary (the
  log-sum-exp over the other ids) and the argmax of the logits.
- Composition B: a prefill of the first n - 32 tokens, then the tokens
  n - 32 .. n - 2 decoded one per step with `start_pos` advancing (the
  reference's decode path: the window ring, the compressors' carried
  groups, the indexers over the cached keys). Its 32 predicting positions
  n - 33 .. n - 2 (the prefill's last position predicts the first) record
  the same quantities, B's log-probabilities at A's 64 ids and B's
  log-probability of the rest of the vocabulary beside those ids.
- Spread of the reference between the compositions, per position of B:
  KL(A || B) on the partition of the vocabulary into A's 64 ids and the
  rest, `sum_i pA_i (log pA_i - log pB_i) + rA (log rA - log rB)` over the
  64 ids, with `r` each side's mass of the rest, read from its recorded
  log-probability (not `1 - sum_i p_i`, which the rounding of the 64 FP32
  values can bring to 0 or below); exact on that partition and a lower
  bound of the vocabulary's KL, which is recorded beside it. And whether
  A's and B's argmax agree. The spread is the
  reference's own variation across batch compositions of one computation,
  and bounds what a comparison against A can expect.

Writes into `--out`:

- `prompt_<target>.safetensors` per prompt: `ids` [n] int32,
  `next_logprob` [n-1] f32, `top_ids` [n-1, 64] int32, `top_logprobs`
  [n-1, 64] f32, `top_rest_logprob` [n-1] f32 and `argmax` [n-1] int32 of
  composition A (row t is position t); `decode_positions` [32] int32 and,
  row for row, `decode_next_logprob`, `decode_top_ids`,
  `decode_top_logprobs`, `decode_top_rest_logprob`, `decode_argmax` of
  composition B (its own top ids), `decode_logprob_at_top_ids` [32, 64] and
  `decode_rest_logprob` [32] (B at A's ids `top_ids[decode_positions]` and
  at the rest beside them), `decode_kl` [32] (the partition's KL) and
  `decode_kl_vocabulary` [32] (over the whole vocabulary);
- `manifest.json` (rank 0): the release revision, the SHA-256 of the
  reference files, the environment, MP, the model arguments' overrides, the
  prompts' sources, texts and token counts, their summaries and spread
  statistics, the load and run times and each rank's peak GPU memory.

Run from the repository root on a host with four free GPUs, with an
interpreter that has the reference's requirements
(`inference/requirements.txt`: torch, tilelang 0.1.12, safetensors,
transformers, sympy, numpy, Pillow)::

    CUDA_VISIBLE_DEVICES=0,1,2,3 MAX_JOBS=16 CUDA_HOME=<CUDA toolkit> \
        python -m torch.distributed.run --standalone --nproc-per-node 4 \
        crates/models/tests/it/deepseek_v41/generate_logprobs.py \
        --checkpoint <DeepSeek-V4.1-Flash-1017> \
        --out artifacts/deepseek_v41/reference

TileLang compiles the kernels with the `nvcc` of `CUDA_HOME`, which must be
a complete toolkit whose compiler and headers are of one release (the CUDA
release of torch's build): without it TileLang takes the `nvcc` of a pip
`nvidia-cuda-nvcc`, which need not match the pip runtime's headers.

`--sources` and `--targets` (pairwise) select a subset of the prompts, for
example the first one alone to prove a setup. With `--agentx <payloads>`
the prompts are the AgentX payloads nearest below each of `--targets`, for
example `--targets 65536 131072`.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import safetensors
import torch
import torch.distributed as dist
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

#: The prompts' sources, relative to the repository root, `+` joining the
#: files of one source, and the token budgets they are cut to, pairwise.
SOURCES = (
    "docs/expert_parallelism.md",
    "docs/configuration.md",
    "docs/architecture.md",
    "docs/architecture.md+docs/configuration.md",
)
TARGETS = (1024, 4096, 8192, 16384)

#: The chat-mode rendering of one user message around its text.
PREFIX = "<｜begin▁of▁sentence｜><｜User｜>"
SUFFIX = "<｜Assistant｜></think>"

#: The most probable ids recorded per position.
TOP_K = 64

#: The predicting positions of composition B: the prefill's last one and
#: the tokens decoded one per step after it.
DECODED = 32

#: The reference's tensor parallelism the values are of.
MODEL_PARALLEL = 4

#: The least cache capacity in tokens, `generate.py`'s interactive setting,
#: raised to the longest prompt rounded up to 1024. It sizes the caches and
#: the rotary tables.
MAX_SEQ_LEN = 64 * 1024

#: The positions whose logits are formed and reduced at once, which bounds
#: the head's and the reduction's memory.
ROWS = 1024

#: The index queries a long prefill scores at once: a block of 1024 queries
#: against 128K keys holds 2 GiB of BF16 scores per rank's 8 heads.
QUERIES = 1024

#: The positions whose hyper-connection mixes are formed at once: 16384
#: positions' `[4, 4, 5120]` FP32 products are 5 GiB.
POSITIONS = 16384

#: The tensors `convert.py` shards along a dimension, by the name component
#: it keys them with (its `mapping`, whose renames are all identities).
SHARDED = {
    "embed": 0,
    "wq_b": 0,
    "wo_a": 0,
    "wo_b": 1,
    "head": 0,
    "attn_sink": 0,
    "weights_proj": 0,
}

#: The reference files the values depend on, relative to the checkpoint.
REFERENCE_FILES = (
    "inference/config.json",
    "inference/convert.py",
    "inference/engram.py",
    "inference/generate.py",
    "inference/image_processor.py",
    "inference/kernel.py",
    "inference/model.py",
    "inference/vision.py",
    "encoding/encoding.py",
)


def log(message: str) -> None:
    """Prints one progress line of this rank, prefixed by its rank."""
    print(f"[rank {dist.get_rank()}] {message}", flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Checkpoint:
    """The released tensors by name, read lazily from their shards.

    Each shard is opened once and kept open (a memory map), so a rank reads
    only the bytes of the tensors and rows it asks for.
    """

    def __init__(self, root: Path) -> None:
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.root = root
        self.shards: dict[str, str] = index["weight_map"]
        self.files: dict[str, object] = {}

    def file(self, name: str):
        shard = self.shards[name]
        if shard not in self.files:
            self.files[shard] = safe_open(self.root / shard, "pt")
        return self.files[shard]

    def tensor(self, name: str) -> torch.Tensor:
        return self.file(name).get_tensor(name)

    def rows(self, name: str, start: int, stop: int) -> torch.Tensor:
        return self.file(name).get_slice(name)[start:stop]

    def num_rows(self, name: str) -> int:
        return self.file(name).get_slice(name).get_shape()[0]


def converted(source: str) -> tuple[str, int | None]:
    """`convert.py`'s name of a released tensor and its sharded dimension.

    The renames are those of `convert.py`; the release already uses the
    converted names, so they leave every name of this checkpoint as it is.
    """
    name = source.removeprefix("model.")
    name = name.replace("self_attn", "attn")
    if not name.startswith("vision."):
        name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")

    # Tensors without a `.weight` suffix are keyed by their last component.
    bare = ("hc", "attn_sink", "tie2eid", "tid2eid", "ape", "image_")
    key = name.split(".")[-1 if any(x in name for x in bare) else -2]
    return name, SHARDED.get(key)


def in_text_model(name: str) -> bool:
    """Whether a converted tensor is one of the text model built here.

    The DSpark draft stages, the vision tower and its aligner, the learned
    image-span embeddings and the image tokens' routing bias are not.
    """
    if name.startswith(("mtp.", "vision.", "aligner.", "image_")):
        return False
    return not name.endswith(".gate.bias_vl")


def routed_experts(names: list[str]) -> int:
    """The backbone's routed experts, from the weight names.

    The count is `convert.py`'s `infer_num_experts` of the backbone.
    """
    indices = [
        int(match.group(1))
        for name in names
        if (match := re.search(r"(?:mlp|ffn)\.experts\.(\d+)\.", name))
        and not name.startswith("mtp.")
    ]
    return max(indices) + 1


def rank_state(
    checkpoint: Checkpoint,
    sources: list[str],
    rank: int,
    world_size: int,
    n_experts: int,
) -> dict[str, torch.Tensor]:
    """The part of one rank's `convert.py` file that `sources` make up.

    Args:
        checkpoint: The released shards.
        sources: Released tensor names of the text model. A `wo_a` weight
            and its scale must be among them together.
        rank: The rank whose share is read.
        world_size: The tensor parallelism (MP).
        n_experts: The backbone's routed experts.

    Returns:
        The rank's tensors by converted name, on the host, in the dtypes
        `convert.py` saves (the model's `load_state_dict` casts to the
        parameters' dtypes as it does for the converted files).
    """
    local_experts = n_experts // world_size
    state = {}
    for source in sources:
        name, dim = converted(source)
        if "experts" in name and "shared_experts" not in name:
            # Routed experts are not split: a rank holds a range of them.
            index = int(name.split(".")[-3])
            first = rank * local_experts
            if not first <= index < first + local_experts:
                continue
            tensor = checkpoint.tensor(source)
        elif ".engram.embed." in name:
            # The table's rows in blocks of ceil(rows / MP); the last block
            # is padded with zero rows whose scales are 1.
            rows = checkpoint.num_rows(source)
            block = (rows + world_size - 1) // world_size
            start = min(rank * block, rows)
            stop = min(start + block, rows)
            tensor = checkpoint.rows(source, start, stop)
            if tensor.size(0) < block:
                fill = 1 if name.endswith(".scale") else 0
                padding = tensor.new_full(
                    (block - tensor.size(0), tensor.size(1)), fill
                )
                tensor = torch.cat([tensor, padding])
        elif dim is not None:
            tensor = checkpoint.tensor(source)
            assert tensor.size(dim) % world_size == 0, (name, tensor.shape)
            size = tensor.size(dim) // world_size
            tensor = tensor.narrow(dim, rank * size, size).contiguous()
        else:
            tensor = checkpoint.tensor(source)
        state[name] = tensor

    # `convert.py`'s second pass over a rank's tensors.
    for name in list(state):
        if name.endswith("wo_a.weight"):
            # The grouped output projection is used as a BF16 einsum
            # operand: E4M3 times its E8M0 scale per block, after sharding.
            weight = state[name]
            scale = state.pop(name.replace("weight", "scale"))
            out_block = weight.size(0) // scale.size(0)
            in_block = weight.size(1) // scale.size(1)
            assert (out_block, in_block) in ((32, 32), (128, 128)), name
            weight = (
                weight.unflatten(0, (-1, out_block))
                .unflatten(-1, (-1, in_block))
                .float()
                * scale[:, None, :, None].float()
            )
            state[name] = weight.flatten(2, 3).flatten(0, 1).bfloat16()
        elif "experts" in name and state[name].dtype == torch.int8:
            # Two E2M1 values per byte, the low nibble first.
            state[name] = state[name].view(torch.float4_e2m1fn_x2)
    return state


def load(model: torch.nn.Module, root: Path) -> None:
    """Loads this rank's share of the release into `model`, layer by layer.

    Raises:
        AssertionError: When a tensor has no parameter, or a parameter no
            tensor: together `strict=True` over the whole model.
    """
    rank, world_size = dist.get_rank(), dist.get_world_size()
    checkpoint = Checkpoint(root)
    names = list(checkpoint.shards)
    n_experts = routed_experts(names)
    assert n_experts % world_size == 0, n_experts

    # One group per decoder layer and one per remaining top-level module,
    # each read shard by shard.
    groups = defaultdict(list)
    for source in sorted(names, key=lambda name: checkpoint.shards[name]):
        name, _ = converted(source)
        if in_text_model(name):
            parts = name.split(".")
            depth = 2 if parts[0] == "layers" else 1
            groups[".".join(parts[:depth])].append(source)

    expected = set(model.state_dict())
    loaded: set[str] = set()
    read = 0
    start = time.monotonic()
    for index, (group, sources) in enumerate(groups.items()):
        state = rank_state(checkpoint, sources, rank, world_size, n_experts)
        result = model.load_state_dict(state, strict=False)
        assert not result.unexpected_keys, result.unexpected_keys
        loaded.update(state)
        read += sum(t.numel() * t.element_size() for t in state.values())
        del state
        if rank == 0 and (index % 8 == 0 or index == len(groups) - 1):
            log(
                f"loaded {group} ({index + 1}/{len(groups)}): "
                f"{read / 2**30:.1f} GiB in {time.monotonic() - start:.0f} s"
            )
    checkpoint.files.clear()

    missing = expected - loaded
    assert not missing, sorted(missing)[:16]
    log(f"loaded {read / 2**30:.1f} GiB in {time.monotonic() - start:.0f} s")


def source_text(spec: str) -> tuple[str, list[dict]]:
    """The text of one prompt source and its files' paths and digests."""
    paths = [Path(path) for path in spec.split("+")]
    texts = [path.read_text() for path in paths]
    files = [{"path": str(path), "sha256": sha256(path)} for path in paths]
    return "\n\n".join(text.strip("\n") for text in texts), files


def cut(text: str, target: int, tokenizer, encode_messages) -> str:
    """`text` up to its last blank line that keeps the prompt in `target`.

    The prompt of the cut text encodes to at most `target` tokens. The
    prefixes are tried in order: a longer prefix holds the shorter's
    tokens and the paragraph after them, so the first that exceeds the
    target ends the search.
    """
    ends = [match.start() for match in re.finditer(r"\n\n", text)]
    best = None
    for end in [*ends, len(text)]:
        prompt = PREFIX + text[:end] + SUFFIX
        if len(tokenizer.encode(prompt, add_special_tokens=False)) > target:
            break
        best = text[:end]
    assert best is not None, f"no paragraph fits in {target} tokens"

    # The template is the reference encoder's rendering of the message.
    message = [{"role": "user", "content": best}]
    rendered = encode_messages(message, thinking_mode="chat")
    assert rendered == PREFIX + best + SUFFIX, rendered[:64]
    return best


def prompts(arguments, tokenizer, encode_messages) -> list[dict]:
    """The prompts of the protocol: sources, cut texts and token ids."""
    records = []
    for spec, target in zip(arguments.sources, arguments.targets, strict=True):
        text, files = source_text(spec)
        text = cut(text, target, tokenizer, encode_messages)
        ids = tokenizer.encode(PREFIX + text + SUFFIX, add_special_tokens=False)
        records.append(
            {
                "file": f"prompt_{target}.safetensors",
                "source": spec,
                "files": files,
                "target": target,
                "tokens": len(ids),
                "text": text,
                "ids": ids,
                "compare_blocked": target in arguments.check_blocked,
            }
        )
    return records


def agentx_prompts(arguments, tokenizer, encode_messages) -> list[dict]:
    """The AgentX payloads of the protocol: one per target, with vLLM's ids.

    The ids must be the reference encoder's rendering of the payload's
    messages, tokenized without added special tokens.
    """
    root = arguments.agentx
    counts = [json.loads(line) for line in (root / "vllm_counts.jsonl").open()]
    chosen = {}
    for target in arguments.targets:
        below = [row for row in counts if row["num_tokens"] <= target]
        best = max(below, key=lambda row: (row["num_tokens"], row["id"]))
        chosen[best["id"]] = target

    def lines(name: str) -> dict:
        """The lines of `name` of the chosen payloads, by id."""
        found = {}
        for line in (root / name).open():
            identity = json.loads(line[: line.index(",")] + "}")["id"]
            if identity in chosen:
                found[identity] = json.loads(line)
                if len(found) == len(chosen):
                    break
        return found

    payloads, tokens = lines("payloads.jsonl"), lines("vllm.jsonl")
    records = []
    for identity, target in chosen.items():
        messages = payloads[identity]["payload"]["messages"]
        ids = tokens[identity]["token_ids"]
        rendered = encode_messages(messages, thinking_mode="thinking")
        assert tokenizer.encode(rendered, add_special_tokens=False) == ids, (
            identity
        )
        records.append(
            {
                "file": f"agentx_{target}.safetensors",
                "source": f"agentx:{identity}",
                "target": target,
                "tokens": len(ids),
                "messages": messages,
                "ids": ids,
                "compare_blocked": target in arguments.check_blocked,
            }
        )
    return records


def head_logits(head, rows: torch.Tensor) -> torch.Tensor:
    """The reference head's FP32 logits of `rows`, `[m, dim]` normalized
    hidden rows: `ParallelHead.forward` over every row given.
    """  # noqa: D205
    logits = torch.nn.functional.linear(rows.float(), head.weight)
    if dist.get_world_size() > 1:
        parts = [torch.empty_like(logits) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, logits)
        logits = torch.cat(parts, dim=-1)
    return logits


def blocked_indexer(reference):
    """`Indexer.forward` of `reference` with a long prefill's scoring taken
    per block of `QUERIES` query rows.

    The reference's statements are kept in their order; only the scores'
    einsum and what reads them run per block. A query's scores, its
    visibility mask, its candidate blocks and its top-k depend on its own
    row alone, so the blocks are the reference's rows. Layer 20's candidate
    blocks are kept as the first column of each block of their mask and
    repeated when a later indexer reads them, which is the mask.
    """  # noqa: D205
    original = reference.Indexer.forward
    shared = reference.shared_attn

    def forward(self, x, qr, latent, start_pos, offset):
        bsz, seqlen, _ = x.size()
        if start_pos != 0 or seqlen <= QUERIES:
            return original(self, x, qr, latent, start_pos, offset)
        ratio, rd, end_pos = (
            self.compress_ratio,
            self.rope_head_dim,
            start_pos + seqlen,
        )

        if self.owns_k and latent is not None:
            freqs = self.freqs_cis[: seqlen - seqlen % ratio : ratio]
            k = self.k_norm(self.wk(latent))
            reference.apply_rotary_emb(k[..., -rd:], freqs)
            reference.fp4_act_quant(k, reference.fp4_block_size, True)
            self.k_cache[
                :bsz, start_pos // ratio : start_pos // ratio + k.size(1)
            ] = k
        if self.owns_k:
            shared.index_k = self.k_cache

        q = self.wq_b(qr).unflatten(
            -1, (self.n_local_heads, self.index_head_dim)
        )
        reference.apply_rotary_emb(
            q[..., -rd:], self.freqs_cis[start_pos:end_pos]
        )
        reference.fp4_act_quant(q, reference.fp4_block_size, True)

        index_k = shared.index_k[:bsz, : end_pos // ratio]
        weights = self.weights_proj(x) * (
            self.softmax_scale * self.n_heads**-0.5
        )
        width = end_pos // ratio
        topk = min(self.index_topk, end_pos // ratio)
        block = self.candidate_block_size
        kept_blocks, selected = [], []
        for first in range(0, seqlen, QUERIES):
            last = min(seqlen, first + QUERIES)
            index_score = torch.einsum(
                "bshd,btd->bsht", q[:, first:last], index_k
            )
            index_score = (
                index_score.relu_() * weights[:, first:last].unsqueeze(-1)
            ).sum(dim=2)
            if reference.world_size > 1:
                dist.all_reduce(index_score)

            positions = torch.arange(first + 1, last + 1, device=x.device)
            compress_lens = (positions // ratio).unsqueeze(-1)
            index_score.masked_fill_(
                torch.arange(seqlen // ratio, device=x.device) >= compress_lens,
                -torch.inf,
            )
            if self.is_candidate_source:
                mask = reference.select_candidate_blocks(
                    index_score,
                    compress_lens,
                    self.candidate_topk_blocks,
                    block,
                )
                kept_blocks.append(mask[..., ::block])
            elif self.uses_candidates:
                keep = shared.candidates[:, first:last]
                mask = keep.repeat_interleave(block, dim=-1)[..., :width]
                index_score = index_score.masked_fill(~mask, -torch.inf)

            idxs = (
                index_score.topk(topk, dim=-1, sorted=False)
                .indices.sort(dim=-1)
                .values
            )
            selected.append(
                torch.where(idxs < compress_lens, idxs + offset, -1).int()
            )
        if self.is_candidate_source:
            shared.candidates = torch.cat(kept_blocks, dim=1)
        return torch.cat(selected, dim=1)

    return forward


def blocked_hyper_connections(reference) -> None:
    """`Block.hc_pre` and `Block.hc_post` of `reference` per block of
    `POSITIONS` positions.

    Each position's output is the reference's expression of its own rows
    (its mix times its streams, summed over the streams in the reference's
    order), so the blocks compute every element as the whole does.
    """  # noqa: D205
    block = reference.Block
    pre, post = block.hc_pre, block.hc_post

    def hc_pre(self, x, pre_mix):
        if x.size(1) <= POSITIONS:
            return pre(self, x, pre_mix)
        return torch.cat(
            [
                pre(
                    self, x[:, i : i + POSITIONS], pre_mix[:, i : i + POSITIONS]
                )
                for i in range(0, x.size(1), POSITIONS)
            ],
            dim=1,
        )

    def hc_post(self, x, residual, mix, comb):
        if x.size(1) <= POSITIONS:
            return post(self, x, residual, mix, comb)
        return torch.cat(
            [
                post(
                    self,
                    x[:, i : i + POSITIONS],
                    residual[:, i : i + POSITIONS],
                    mix[:, i : i + POSITIONS],
                    comb[:, i : i + POSITIONS],
                )
                for i in range(0, x.size(1), POSITIONS)
            ],
            dim=1,
        )

    block.hc_pre, block.hc_post = hc_pre, hc_post


def rest_logprob(logprobs: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """The log-probability of the vocabulary beside `ids`, per row.

    Args:
        logprobs: [m, vocab] log-probabilities.
        ids: [m, k] int64 ids of each row.

    Returns:
        [m] the log-sum-exp of each row's log-probabilities of every id
        outside its `ids`.
    """
    return logprobs.scatter(1, ids, -torch.inf).logsumexp(dim=-1)


def reduce(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """The recorded quantities of logit rows predicting `targets`.

    Args:
        logits: [m, vocab] FP32 logits.
        targets: [m] the token each row predicts.

    Returns:
        `next_logprob` [m], `top_ids` [m, TOP_K] (int32), `top_logprobs`
        [m, TOP_K] (descending), `top_rest_logprob` [m] and `argmax` [m]
        (int32), on the host.
    """
    parts = defaultdict(list)
    for start in range(0, logits.size(0), ROWS):
        rows = logits[start : start + ROWS]
        logprobs = torch.log_softmax(rows.float(), dim=-1)
        expected = targets[start : start + ROWS, None]
        parts["next_logprob"].append(logprobs.gather(1, expected)[:, 0])
        top = logprobs.topk(TOP_K, dim=-1)
        parts["top_ids"].append(top.indices.int())
        parts["top_logprobs"].append(top.values)
        rest = rest_logprob(logprobs, top.indices)
        parts["top_rest_logprob"].append(rest)
        # The reference's greedy choice (`sample` at temperature 0).
        parts["argmax"].append(rows.argmax(dim=-1).int())
    return {name: torch.cat(part).cpu() for name, part in parts.items()}


def spread(
    top_ids: torch.Tensor,
    top_logprobs: torch.Tensor,
    top_rest: torch.Tensor,
    a_logprobs: torch.Tensor,
    b_logprobs: torch.Tensor,
) -> dict:
    """KL(A || B) per position, on A's top ids plus the rest and exactly.

    Args:
        top_ids: [m, TOP_K] A's most probable ids.
        top_logprobs: [m, TOP_K] A's log-probabilities of them, as stored.
        top_rest: [m] A's log-probability of the rest, as stored.
        a_logprobs: [m, vocab] A's log-probabilities.
        b_logprobs: [m, vocab] B's log-probabilities.

    Returns:
        `decode_logprob_at_top_ids` [m, TOP_K], `decode_rest_logprob` [m],
        `decode_kl` [m] on the partition and `decode_kl_vocabulary` [m],
        on the host.
    """
    ids = top_ids.long()
    at_top = b_logprobs.gather(1, ids)
    b_rest = rest_logprob(b_logprobs, ids)

    # The partition's KL from the values a comparison reads, in FP64: the
    # 64 ids' terms and the rest's.
    la, lb = top_logprobs.double(), at_top.double()
    ra, rb = top_rest.double(), b_rest.double()
    kl = (la.exp() * (la - lb)).sum(-1) + ra.exp() * (ra - rb)

    a, b = a_logprobs.double(), b_logprobs.double()
    vocabulary = (a.exp() * (a - b)).sum(-1)
    return {
        "decode_logprob_at_top_ids": at_top.cpu(),
        "decode_rest_logprob": b_rest.cpu(),
        "decode_kl": kl.float().cpu(),
        "decode_kl_vocabulary": vocabulary.float().cpu(),
    }


@torch.inference_mode()
def prefill(model, ids: torch.Tensor, positions: torch.Tensor):
    """Composition A of `ids`: one prefill of the whole prompt, the head's
    logits of every position but the last formed and reduced a block of
    `ROWS` positions at a time.

    Returns:
        The reduced quantities of every predicting position (on rank 0, an
        empty dict elsewhere) and the log-probabilities of `positions` (on
        every rank).
    """  # noqa: D205
    leader = dist.get_rank() == 0
    n = ids.numel()
    hidden = model.forward(ids[None], 0)[1][0]
    parts = defaultdict(list)
    for first in range(0, n - 1, ROWS):
        last = min(n - 1, first + ROWS)
        logits = head_logits(model.head, hidden[first:last])
        if leader:
            reduced = reduce(logits, ids[first + 1 : last + 1])
            for name, part in reduced.items():
                parts[name].append(part)
    tensors = {name: torch.cat(part) for name, part in parts.items()}
    logprobs = torch.log_softmax(
        head_logits(model.head, hidden[positions]), dim=-1
    )
    return tensors, logprobs


@torch.inference_mode()
def evaluate(model, record: dict) -> dict | None:
    """Runs compositions A and B of one prompt on every rank.

    Returns:
        On rank 0 the prompt's tensors and summary; None elsewhere. Every
        rank holds the gathered logits, so rank 0 alone reduces them.
    """
    leader = dist.get_rank() == 0
    ids = torch.tensor(record["ids"], dtype=torch.long, device="cuda")
    n = ids.numel()
    assert n > DECODED + 1, n
    # The predicting positions of B, n - 33 .. n - 2.
    positions = torch.arange(n - 1 - DECODED, n - 1, device="cuda")

    # A: the whole prompt in one prefill.
    start = time.monotonic()
    tensors, a_logprobs = prefill(model, ids, positions)
    torch.cuda.synchronize()
    a_seconds = time.monotonic() - start

    # The blocked indexer against the reference's own, on a prompt whose
    # whole scores fit: A again with every indexer as it is.
    blocked = None
    if record.get("compare_blocked"):
        global QUERIES
        queries, QUERIES = QUERIES, n
        whole, _ = prefill(model, ids, positions)
        QUERIES = queries
        if leader:
            delta = (whole["next_logprob"] - tensors["next_logprob"]).abs()
            blocked = {
                "queries": queries,
                "next_logprob_max_abs_difference": delta.max().item(),
                "next_logprob_mean_abs_difference": delta.double()
                .mean()
                .item(),
                "argmax_agreement": (
                    (whole["argmax"] == tensors["argmax"])
                    .double()
                    .mean()
                    .item()
                ),
            }

    # B: a prefill of the first n - 32 tokens, whose last position predicts
    # token n - 32, then one decode step per token up to n - 2.
    start = time.monotonic()
    rows = [model.forward(ids[None, : n - DECODED], 0)[1][0, -1]]
    for position in range(n - DECODED, n - 1):
        token = ids[None, position : position + 1]
        rows.append(model.forward(token, position)[1][0, -1])
    b_logits = head_logits(model.head, torch.stack(rows))
    torch.cuda.synchronize()
    b_seconds = time.monotonic() - start
    if not leader:
        return None

    b_logprobs = torch.log_softmax(b_logits.float(), dim=-1)
    decoded = reduce(b_logits, ids[positions + 1])
    tensors.update({f"decode_{name}": t for name, t in decoded.items()})
    rows_of_b = positions.cpu()
    tensors.update(
        spread(
            tensors["top_ids"][rows_of_b].cuda(),
            tensors["top_logprobs"][rows_of_b].cuda(),
            tensors["top_rest_logprob"][rows_of_b].cuda(),
            a_logprobs,
            b_logprobs,
        )
    )
    tensors["decode_positions"] = rows_of_b.int()
    tensors["ids"] = ids.int().cpu()

    hits = tensors["argmax"] == tensors["ids"][1:]
    agree = tensors["argmax"][rows_of_b] == tensors["decode_argmax"]
    difference = tensors["next_logprob"][rows_of_b]
    difference = (difference - tensors["decode_next_logprob"]).abs()
    summary = {
        **({"blocked_against_whole_indexer": blocked} if blocked else {}),
        "mean_next_logprob": tensors["next_logprob"].double().mean().item(),
        "argmax_is_next": hits.double().mean().item(),
        "decode": {
            "positions": [rows_of_b[0].item(), rows_of_b[-1].item()],
            "kl_mean": tensors["decode_kl"].double().mean().item(),
            "kl_max": tensors["decode_kl"].max().item(),
            "kl_vocabulary_mean": (
                tensors["decode_kl_vocabulary"].double().mean().item()
            ),
            "kl_vocabulary_max": tensors["decode_kl_vocabulary"].max().item(),
            "top1_agreement": agree.double().mean().item(),
            "next_logprob_max_abs_difference": difference.max().item(),
        },
        "seconds": {"a": a_seconds, "b": b_seconds},
    }
    return {"tensors": tensors, "summary": summary}


def revision(root: Path) -> str | None:
    """The release commit the Hugging Face download recorded, if any."""
    files = sorted((root / ".cache/huggingface/download").glob("*.metadata"))
    commits = {path.read_text().splitlines()[0] for path in files}
    assert len(commits) <= 1, commits
    return commits.pop() if commits else None


def import_reference(root: Path):
    """The reference's `model` module and its chat encoder.

    The modules import each other by bare name from their directories, as
    `generate.py` does; no bytecode is written beside them.
    """
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "encoding"))
    sys.path.insert(0, str(root / "inference"))
    import model
    import tilelang
    from encoding import encode_messages

    return model, tilelang, encode_messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/deepseek_v41/reference")
    )
    parser.add_argument("--sources", nargs="+", default=list(SOURCES))
    parser.add_argument("--targets", nargs="+", type=int, default=list(TARGETS))
    parser.add_argument(
        "--agentx",
        type=Path,
        help="a directory of AgentX payloads to take the prompts from",
    )
    parser.add_argument(
        "--check-blocked",
        nargs="*",
        type=int,
        default=[],
        help="targets whose prompts also run composition A with the "
        "reference's indexer whole, against the blocked one",
    )
    arguments = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert world_size == MODEL_PARALLEL, world_size
    root = arguments.checkpoint

    # `generate.py`'s process setup.
    reference, tilelang, encode_messages = import_reference(root)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)

    tokenizer = AutoTokenizer.from_pretrained(root)
    # Rank 0 cuts and encodes the prompts; every rank runs the same ids.
    shared = [None]
    if rank == 0:
        chosen = agentx_prompts if arguments.agentx else prompts
        shared = [chosen(arguments, tokenizer, encode_messages)]
    dist.broadcast_object_list(shared, 0)
    records = shared[0]
    if rank == 0:
        for record in records:
            log(f"{record['file']}: {record['tokens']} tokens")

    # The release's arguments for one sequence of the text model, sampling
    # greedily so the reference's `sample` is the argmax of the logits.
    config = json.loads((root / "inference/config.json").read_text())
    longest = max(record["tokens"] for record in records)
    overrides = {
        "max_batch_size": 1,
        "max_seq_len": max(MAX_SEQ_LEN, -(-longest // 1024) * 1024),
        "temperature": 0.0,
        "dspark_block_size": 0,
        "vision_n_layers": 0,
    }
    args = reference.ModelArgs(**{**config, **overrides})
    assert max(record["tokens"] for record in records) <= args.max_seq_len

    start = time.monotonic()
    with torch.device("cuda"):
        model = reference.Transformer(args, tokenizer)
    build_seconds = time.monotonic() - start
    start = time.monotonic()
    load(model, root)
    dist.barrier()
    load_seconds = time.monotonic() - start
    torch.set_default_device("cuda")

    # The head is handed the normalized rows of every position, whose
    # logits `evaluate` forms a block at a time (`head_logits`), and a long
    # prefill's indexers score a block of queries at a time.
    model.head.forward = lambda rows, full_logits=False: rows
    reference.Indexer.forward = blocked_indexer(reference)
    blocked_hyper_connections(reference)

    if rank == 0:
        arguments.out.mkdir(parents=True, exist_ok=True)
    release = revision(root)
    start = time.monotonic()
    done = []
    for record in records:
        result = evaluate(model, record)
        if result is not None:
            metadata = {
                "release": f"deepseek-ai/DeepSeek-V4.1-Flash-1017@{release}",
                "source": record["source"],
                "target": str(record["target"]),
            }
            tensors = {k: t.contiguous() for k, t in result["tensors"].items()}
            save_file(tensors, arguments.out / record["file"], metadata)
            record.update(result["summary"])
            log(f"{record['file']}: {json.dumps(result['summary'])}")
        done.append({k: v for k, v in record.items() if k != "ids"})

        # The manifest of the prompts done so far, after each one, so a
        # later prompt's failure leaves the earlier ones stated.
        peak = {
            "allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        peaks = [None] * world_size
        dist.all_gather_object(peaks, peak)
        if rank == 0:
            manifest = {
                "release": {
                    "repository": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
                    "revision": release,
                },
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
                "model_parallel": world_size,
                "model_args_overrides": overrides,
                "protocol": {
                    "encoding": (
                        "AgentX: vLLM's deepseek_v41 ids of the payload's "
                        "messages, the reference encoder's thinking mode"
                        if arguments.agentx
                        else PREFIX + "{text}" + SUFFIX
                    ),
                    "thinking_mode": (
                        "thinking" if arguments.agentx else "chat"
                    ),
                    "add_special_tokens": False,
                    "top_k": TOP_K,
                    "decode_positions": DECODED,
                    "index_queries_per_block": QUERIES,
                    "hyper_connection_positions_per_block": POSITIONS,
                    "logprobs": (
                        "float32 log_softmax of the float32 head logits"
                    ),
                    "decode_kl": (
                        "KL(A||B) on the partition into A's top-k ids and "
                        "the rest of the vocabulary"
                    ),
                },
                "seconds": {
                    "build": build_seconds,
                    "load": load_seconds,
                    "run": time.monotonic() - start,
                },
                "peak_gpu_memory": peaks,
                "prompts": done,
            }
            path = arguments.out / "manifest.json"
            path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
            log(f"wrote {path}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
