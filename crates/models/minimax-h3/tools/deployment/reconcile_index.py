"""The checkpoint index's tensor names against the module's declared parameters."""
import collections, json, sys, types
import torch

if len(sys.argv) < 2:
    sys.exit("usage: reconcile_index.py <checkpoint root>")
ROOT = sys.argv[1]

pkg = types.ModuleType("uniserve_kernels"); tri = types.ModuleType("uniserve_kernels.triton")
tri.require_kernel = lambda *a, **k: (lambda f: f); pkg.triton = tri
sys.modules["uniserve_kernels"] = pkg; sys.modules["uniserve_kernels.triton"] = tri
worker = types.ModuleType("uniserve.worker")
_CANVASES = (1536, 672, 1344, 768, 1024, 768, 768, 768, 768, 1024, 768, 1344)
worker.library = lambda: collections.defaultdict(lambda: (lambda *a, **k: _CANVASES),
                                                 {"video_tasks": lambda: ("t2va", "fl2va", "ref2va"),
                                                  "video_canvas": lambda *a, **k: _CANVASES})
sys.modules["uniserve.worker"] = worker
torch.set_default_device("meta")

from uniserve.models.minimax_h3 import attention as attn, config as h3config, transformer as tr

fields = dict(hidden_size=5376, num_attention_heads=56, head_dim=128, num_hidden_layers=50,
              num_refiner_layers=2, intermediate_size=14336, video_channels=24, audio_channels=32,
              text_dim=5120, frequency_dim=256, time_hidden_dim=5376, time_dim=2688,
              rope_frequency_dim=16, rope_theta=10000.0, norm_eps=1e-5, qk_norm_eps=1e-5, output_heads=1)
base = h3config.TransformerConfig(**fields)
model = tr.Transformer(base, attention=attn.Dense(base), entries=2, steps=4, groups=4)

declared = {}
for name, tensor in model.named_parameters():
    declared[name] = tensor.numel()
print(f"module: {len(declared)} tensors, {sum(declared.values()):,} parameters")

root = f"{ROOT}/transformer"
index = json.load(open(f"{root}/diffusion_pytorch_model.safetensors.index.json"))
named = index["weight_map"]
print(f"index:  {len(named)} names, total_size {index['metadata']['total_size']:,} B")
print(f"index implied at BF16: {index['metadata']['total_size'] / 2:,.0f} parameters")

only_index = sorted(set(named) - set(declared))
only_module = sorted(set(declared) - set(named))
print(f"\nonly in the index ({len(only_index)}): {only_index[:12]}")
print(f"\nonly in the module ({len(only_module)}): {only_module[:12]}")
common = set(named) & set(declared)
print(f"\ncommon {len(common)}; module parameters over common names {sum(declared[n] for n in common):,}")

# The uniform-size tell: if the index's names cannot be given shapes, its total cannot be checked
# per tensor, so compare its mean tensor size against the module's.

# Compare the two namespaces by pattern: strip layer indices, and the checkpoint's own prefixes, so
# a name present on one side only is visible as a pattern rather than as a renaming.
import re

def pattern(name):
    return re.sub(r"\.\d+\.", ".N.", name)

def strip_known(name):
    for prefix in ("transformer.", "model.", "net."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

index_patterns = collections.Counter(pattern(strip_known(n)) for n in named)
module_patterns = collections.Counter(pattern(n) for n in declared)
print("\nper-layer index names:", sum(1 for n in named if "layers." in n))
print("per-layer module names:", sum(1 for n in declared if n.startswith("layers.")))
print("\npatterns only in the index:")
for key in sorted(set(index_patterns) - set(module_patterns))[:24]:
    print(f"   {key:60s} x{index_patterns[key]}")
print("\npatterns only in the module:")
for key in sorted(set(module_patterns) - set(index_patterns))[:24]:
    print(f"   {key:60s} x{module_patterns[key]}")
