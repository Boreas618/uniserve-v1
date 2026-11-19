"""Reference vectors of DeepSeek-V4.1-Flash's sparse latent attention.

A direct port of the checkpoint's own formulas (`inference/model.py`:
`Attention`, `Compressor`, `Indexer`, `select_candidate_blocks`,
`precompute_freqs_cis`, `apply_rotary_emb`; `inference/kernel.py`:
`act_quant`, `fp4_act_quant`, `sparse_attn`) to PyTorch on the CPU, evaluated
on the real parameters of layers 2 (ratio 2, the owner of layers 3-7's
compressed latents) and 20 (ratio 1, the candidate source) over one fixed
prefill of a few thousand tokens.

The TileLang kernels are restated with the arithmetic they compile to: the
fake quantizations round to E4M3 and E2M1 to nearest even with the scales
read off the FP32 bits (`fast_log2_ceil`), and `sparse_attn` walks the index
list in blocks of 64 with its online softmax, its probabilities rounded to
BF16 before the value product and its sink added to the denominator. The
projections whose weights are block-scaled FP8 are evaluated as products of
the exactly dequantized weights in FP32, rounded to BF16: that is the
projection's arithmetic without the FP8 rounding of its activation, which
belongs to the GEMM and not to the attention.

Writes two files into `--out`:

- `parameters.safetensors`: the parameters the modules load, under the
  checkpoint's names, the index query projection `indexer.wq_b` dequantized
  to BF16 (exactly: an E4M3 value times a power of two is a BF16 value);
- `vectors.safetensors`: the input rows `x`, the module inputs the model's
  projections compute (`qr`, the unrotated queries `q`, the latent projection
  `kv` before its normalization) and the reference's results per layer: the
  compressed latents before their rotation, the index scores, the selection
  (sorted entry indices, -1 for none) and the attention's output after the
  inverse rotation.

Run with an environment that has PyTorch and safetensors::

    python generate_attention.py --checkpoint <DeepSeek-V4.1-Flash-1017> \
        --out <directory> [--tokens 3000]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

#: The layers whose parameters the vectors are computed with.
LAYERS = (2, 20)

#: The E4M3 and E2M1 extremes.
E4M3_MAX = 448.0
E2M1_MAX = 6.0

#: The blocks one program of `sparse_attn` walks its index list in.
ATTENTION_BLOCK = 64


class Checkpoint:
    """The checkpoint's tensors by name, read from their shards."""

    def __init__(self, root: Path) -> None:
        self.root = root
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.shards = index["weight_map"]
        self.config = json.loads((root / "config.json").read_text())["text_config"]

    def __getitem__(self, name: str) -> torch.Tensor:
        with safe_open(self.root / self.shards[name], "pt") as shard:
            return shard.get_tensor(name)

    def dequantized(self, name: str) -> torch.Tensor:
        """A block-scaled FP8 weight, E4M3 times its E8M0 scale per 32x32
        block, in FP32."""
        weight = self[f"{name}.weight"].float()
        scale = self[f"{name}.scale"].float()
        rows, columns = weight.shape
        scale = scale.repeat_interleave(32, 0)[:rows].repeat_interleave(32, 1)
        return weight * scale[:, :columns]


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """`x @ weight.T` accumulated in FP32 and rounded to BF16."""
    return (x.float() @ weight.float().T).to(torch.bfloat16)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The reference's `RMSNorm`: FP32 statistics, the gain times the
    normalized row, back in the input's type."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (weight * x).to(dtype)


def freqs_cis(
    dim: int, seqlen: int, original: int, base: float, factor: float, fast: int, slow: int
) -> torch.Tensor:
    """`precompute_freqs_cis`: the rotary factors of every position as
    complex exponentials, YaRN-interpolated when `original` > 0."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original > 0:

        def corrected(rotations: float) -> float:
            return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected(fast)), 0)
        high = min(math.ceil(corrected(slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    angles = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(angles), angles)


def rotate(x: torch.Tensor, freqs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """`apply_rotary_emb` of the trailing channels `x` (a view), in place:
    adjacent pairs as complex numbers times the factors (conjugated for the
    inverse), rounded back into `x`'s type."""
    complex_x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs = freqs.conj()
    shape = [1] * complex_x.ndim
    shape[0], shape[-1] = complex_x.shape[0], complex_x.shape[-1]
    x.copy_(torch.view_as_real(complex_x * freqs.view(shape)).flatten(-2))
    return x


def log2_ceil(x: torch.Tensor) -> torch.Tensor:
    """`fast_log2_ceil`: the exponent of each FP32 value plus one for a
    non-zero mantissa."""
    bits = x.float().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    return exponent + ((bits & 0x7FFFFF) != 0).to(torch.int32)


def e2m1(x: torch.Tensor) -> torch.Tensor:
    """`x`, within +-6, rounded to the E2M1 grid to nearest even."""
    magnitude = x.abs()
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    # The boundaries between neighbouring codes are their midpoints; a midpoint
    # belongs to the even code, so the comparison is strict below an odd code.
    code = (
        (magnitude > 0.25).int()
        + (magnitude >= 0.75).int()
        + (magnitude > 1.25).int()
        + (magnitude >= 1.75).int()
        + (magnitude > 2.5).int()
        + (magnitude >= 3.5).int()
        + (magnitude > 5.0).int()
    )
    return torch.copysign(grid[code], x)


def fp8_fake(x: torch.Tensor) -> torch.Tensor:
    """`act_quant(x, 32, "ue8m0", e8m0, inplace=True)`: E4M3 per 32 under a
    power-of-two scale."""
    blocks = x.float().unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = torch.ldexp(torch.ones_like(amax), log2_ceil(amax * torch.tensor(1 / E4M3_MAX, dtype=torch.float32)))
    codes = (blocks / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    return (codes * scale).flatten(-2).to(x.dtype)


def fp4_fake(x: torch.Tensor, block: int, e4m3_scale: bool) -> torch.Tensor:
    """`fp4_act_quant(x, block, inplace=True)`: E2M1 per `block` under an
    E4M3 scale (the compressed latents) or a power-of-two one (the index
    keys and queries)."""
    blocks = x.float().unflatten(-1, (-1, block))
    amax = blocks.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        scale = (amax.clamp_min(E2M1_MAX * 2**-9) / E2M1_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(E2M1_MAX * 2**-126)
        scale = torch.ldexp(torch.ones_like(amax), log2_ceil(amax * torch.tensor(1 / E2M1_MAX, dtype=torch.float32)))
    codes = e2m1((blocks / scale).clamp(-E2M1_MAX, E2M1_MAX))
    return (codes * scale).flatten(-2).to(x.dtype)


def window_indices(window: int, seqlen: int) -> torch.Tensor:
    """`get_window_topk_idxs` of a prefill from position 0: each query's
    causal window, -1 before the sequence starts."""
    end = torch.arange(seqlen).unsqueeze(1)
    indices = (end - window + 1).clamp(0) + torch.arange(min(seqlen, window))
    return torch.where(indices > end, -1, indices)


def sparse_attention(
    q: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor, indices: torch.Tensor, scale: float
) -> torch.Tensor:
    """`sparse_attn_kernel` over one sequence: each query's index list in
    blocks of 64, an online FP32 softmax with the maximum floored at -1e30,
    the probabilities rounded to BF16 before the value product, the sink added
    to the denominator."""
    tokens, heads, _ = q.shape
    out = torch.empty_like(q)
    chunk = 128
    for start in range(0, tokens, chunk):
        stop = min(start + chunk, tokens)
        query = q[start:stop].float()
        rows = stop - start
        acc = torch.zeros(rows, heads, kv.shape[-1])
        total = torch.zeros(rows, heads)
        peak = torch.full((rows, heads), -1e30)
        for block in range(0, indices.shape[1], ATTENTION_BLOCK):
            names = indices[start:stop, block : block + ATTENTION_BLOCK]
            valid = names >= 0
            latents = kv[names.clamp_min(0)].float() * valid.unsqueeze(-1)
            scores = torch.einsum("thd,tnd->thn", query, latents)
            scores = torch.where(valid.unsqueeze(1), scores, -torch.inf) * scale
            previous = peak
            peak = torch.maximum(peak, scores.amax(-1))
            rescale = torch.exp(previous - peak)
            weights = torch.exp(scores - peak.unsqueeze(-1))
            total = total * rescale + weights.sum(-1)
            rounded = weights.to(torch.bfloat16).float()
            acc = acc * rescale.unsqueeze(-1) + torch.einsum("thn,tnd->thd", rounded, latents)
        total = total + torch.exp(sink.unsqueeze(0) - peak)
        out[start:stop] = (acc / total.unsqueeze(-1)).to(torch.bfloat16)
    return out


def select_candidate_blocks(logits: torch.Tensor, visible: torch.Tensor, blocks: int, size: int) -> torch.Tensor:
    """`select_candidate_blocks`: the `blocks` best blocks of `size` entries,
    the newest pinned, as a mask over the entries."""
    width = logits.size(-1)
    scores = torch.nn.functional.pad(logits, (0, -width % size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, size)).amax(dim=-1)
    count = scores.size(-1)
    last = (visible - 1) // size
    scores = scores.masked_fill(torch.arange(count) == last, torch.inf)
    top = scores.topk(min(blocks, count), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(size, dim=-1)[..., :width]


def attention_layer(
    checkpoint: Checkpoint, layer: int, x: torch.Tensor, outputs: dict, parameters: dict
) -> None:
    """One layer's attention over the prefill `x` from position 0, with its
    module inputs, its results and its parameters recorded."""
    config = checkpoint.config
    prefix = f"layers.{layer}.attn"
    tokens = x.shape[0]
    heads, head_dim, rotated = config["num_attention_heads"], config["head_dim"], config["qk_rope_head_dim"]
    ratio = config["compress_ratios"][layer]
    eps = config["rms_norm_eps"]
    window = config["sliding_window"]
    index_heads, index_dim, index_top = config["index_n_heads"], config["index_head_dim"], config["index_topk"]
    scaling = config["rope_scaling"]
    freqs = freqs_cis(
        rotated,
        tokens,
        scaling["original_max_position_embeddings"],
        config["compress_rope_theta"],
        scaling["factor"],
        scaling["beta_fast"],
        scaling["beta_slow"],
    )

    def own(name: str) -> torch.Tensor:
        tensor = checkpoint[f"{prefix}.{name}"]
        parameters[f"{prefix}.{name}"] = tensor.contiguous()
        return tensor

    # The projections of the model, recorded as the modules' inputs.
    qr = rms_norm(linear(x, checkpoint.dequantized(f"{prefix}.wq_a")), checkpoint[f"{prefix}.q_norm.weight"], eps)
    q = linear(qr, checkpoint.dequantized(f"{prefix}.wq_b")).unflatten(-1, (heads, head_dim))
    kv_projected = linear(x, checkpoint.dequantized(f"{prefix}.wkv"))
    outputs[f"{prefix}.qr"] = qr
    outputs[f"{prefix}.q"] = q.clone()
    outputs[f"{prefix}.kv"] = kv_projected

    # The window: normalized, rotated, rounded to E4M3 per 32.
    kv = rms_norm(kv_projected, own("kv_norm.weight"), eps)
    rotate(kv[..., -rotated:], freqs)
    kv = fp8_fake(kv)
    q = q.clone()
    rotate(q[..., -rotated:], freqs)

    # The compressor: FP32 gated pooling of each pair at ratio 2, a plain
    # normalized projection at ratio 1.
    wkv = own("compressor.wkv.weight")
    norm = own("compressor.norm.weight")
    if ratio == 1:
        latent = rms_norm(linear(x, wkv), norm, eps)
    else:
        wgate = own("compressor.wgate.weight")
        values = x.float() @ wkv.float().T
        gates = x.float() @ wgate.float().T
        groups = tokens // ratio
        values = values[: groups * ratio].unflatten(0, (groups, ratio))
        gates = gates[: groups * ratio].unflatten(0, (groups, ratio))
        latent = rms_norm((values * gates.softmax(dim=1)).sum(dim=1).to(torch.bfloat16), norm, eps)
    outputs[f"{prefix}.latent"] = latent.clone()
    groups = latent.shape[0]
    group_freqs = freqs[: tokens - tokens % ratio : ratio]

    # The indexer: keys from the latent before its rotation, queries from qr.
    wq_b = checkpoint.dequantized(f"{prefix}.indexer.wq_b")
    parameters[f"{prefix}.indexer.wq_b.weight"] = wq_b.to(torch.bfloat16)
    weights_proj = own("indexer.weights_proj.weight")
    key = rms_norm(linear(latent, own("indexer.wk.weight")), own("indexer.k_norm.weight"), eps)
    rotate(key[..., -rotated:], group_freqs)
    key = fp4_fake(key, 32, e4m3_scale=False)
    index_query = linear(qr, wq_b).unflatten(-1, (index_heads, index_dim))
    rotate(index_query[..., -rotated:], freqs)
    index_query = fp4_fake(index_query, 32, e4m3_scale=False)
    weights = linear(x, weights_proj) * (index_dim**-0.5 * index_heads**-0.5)
    scores = torch.einsum("shd,td->sht", index_query, key)
    scores = (scores.relu_() * weights.unsqueeze(-1)).sum(dim=1)
    visible = (torch.arange(1, tokens + 1) // ratio).unsqueeze(-1)
    scores = scores.masked_fill(torch.arange(groups) >= visible, -torch.inf)
    outputs[f"{prefix}.index_scores"] = scores.float()
    if layer == config["candidate_source_layer_id"]:
        candidates = select_candidate_blocks(
            scores, visible, config["candidate_topk_blocks"], config["candidate_block_size"]
        )
        outputs[f"{prefix}.candidates"] = candidates.to(torch.uint8)
    top = min(index_top, tokens // ratio)
    selected = scores.topk(top, dim=-1, sorted=False).indices.sort(dim=-1).values
    selected = torch.where(selected < visible, selected, -1).int()
    outputs[f"{prefix}.selection"] = selected

    # The compressed latents: rotated at their groups' first positions, then
    # rounded to E2M1 per 16 under E4M3 scales.
    compressed = latent.clone()
    rotate(compressed[..., -rotated:], group_freqs)
    compressed = fp4_fake(compressed, 16, e4m3_scale=True)

    # One index list per query: its window, then its selection behind the
    # window's rows.
    indices = torch.cat([window_indices(window, tokens), torch.where(selected >= 0, selected + tokens, -1)], dim=-1)
    latents = torch.cat([kv, compressed], dim=0)
    sink = own("attn_sink")
    attended = sparse_attention(q, latents, sink, indices, head_dim**-0.5)
    rotate(attended[..., -rotated:], freqs, inverse=True)
    outputs[f"{prefix}.attention"] = attended


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 32))
    checkpoint = Checkpoint(arguments.checkpoint)
    hidden = checkpoint.config["hidden_size"]

    # The fixed input: Gaussian rows normalized by layer 2's attention norm,
    # the scale an attention layer reads.
    raw = torch.randn(arguments.tokens, hidden, generator=torch.Generator().manual_seed(arguments.seed))
    x = rms_norm(raw.to(torch.bfloat16), checkpoint["layers.2.attn_norm.weight"], checkpoint.config["rms_norm_eps"])

    outputs = {"x": x}
    parameters: dict = {}
    for layer in LAYERS:
        attention_layer(checkpoint, layer, x, outputs, parameters)

    arguments.out.mkdir(parents=True, exist_ok=True)
    save_file({name: tensor.contiguous() for name, tensor in parameters.items()}, arguments.out / "parameters.safetensors")
    save_file({name: tensor.contiguous() for name, tensor in outputs.items()}, arguments.out / "vectors.safetensors")


if __name__ == "__main__":
    main()
