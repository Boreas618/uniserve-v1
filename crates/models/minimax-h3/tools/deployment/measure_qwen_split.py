"""Qwen3-VL: how much of the kept set a tensor-parallel rank does not hold."""
import re
import sys

import torch

torch.set_default_device("meta")
from transformers import AutoConfig, AutoModel

if len(sys.argv) < 2:
    sys.exit("usage: measure_qwen_split.py <checkpoint root>")
ROOT = sys.argv[1]

cfg = AutoConfig.from_pretrained(f"{ROOT}/text_encoder")
model = AutoModel.from_config(cfg, dtype=torch.bfloat16)

LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VOCAB = ("embed_tokens", "lm_head")

sharded = shard_visual = vocab = 0
dropped = 0
kept_layers = 50
for name, tensor in model.named_parameters():
    count = tensor.numel()
    match = re.search(r"layers\.(\d+)\.", name)
    if match and int(match.group(1)) >= kept_layers:
        dropped += count
        continue
    if match is None and not name.startswith(("visual", "model.visual")):
        if any(part in name for part in VOCAB):
            vocab += count
            continue
    if "visual" in name:
        shard_visual += count
        continue
    if any(part in name for part in LINEAR):
        sharded += count
    else:
        vocab += count  # norms and the small pieces: replicated

kept = sharded + vocab + shard_visual
print(f"kept {kept/1e9:.3f} B params")
print(f"  linear      {sharded/1e9:7.3f} B  (TP2 halves)")
print(f"  vocab+norm  {vocab/1e9:7.3f} B  (vocab halves at TP2; norms replicate)")
print(f"  visual      {shard_visual/1e9:7.3f} B  (replicated)")
print(f"TP1 BF16 {kept*2/2**30:7.3f} GiB, TP2 BF16 {(sharded/2 + vocab/2 + shard_visual)*2/2**30:7.3f} GiB")
