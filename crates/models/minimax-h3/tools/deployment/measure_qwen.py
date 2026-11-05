"""Qwen3-VL's parameter count, whole and with the layers the model keeps."""
import json
import sys

import torch

torch.set_default_device("meta")
import transformers
from transformers import AutoConfig, AutoModel

if len(sys.argv) < 2:
    sys.exit("usage: measure_qwen.py <checkpoint root>")
ROOT = sys.argv[1]

print("transformers", transformers.__version__)
cfg = AutoConfig.from_pretrained(f"{ROOT}/text_encoder")
model = AutoModel.from_config(cfg, torch_dtype=torch.bfloat16)
groups = {}
for name, tensor in model.named_parameters():
    key = name.split(".")[0] if not name.startswith("model.") else ".".join(name.split(".")[:3])
    groups[key] = groups.get(key, 0) + tensor.numel()
total = sum(groups.values())
for key, value in sorted(groups.items(), key=lambda kv: -kv[1])[:10]:
    print(f"{key:44s} {value:>16,} BF16 {value*2/2**30:8.3f} GiB")
print(f"{'TOTAL':44s} {total:>16,} BF16 {total*2/2**30:8.3f} GiB")
