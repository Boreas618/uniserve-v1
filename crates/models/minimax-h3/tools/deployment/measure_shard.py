"""How much of each component a tensor-parallel rank does not hold."""
import collections, sys, types
import torch

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

from uniserve.models.minimax_h3 import attention as attn, config as h3config, conditioning, transformer as tr

fields = dict(hidden_size=5376, num_attention_heads=56, head_dim=128, num_hidden_layers=50,
              num_refiner_layers=2, intermediate_size=14336, video_channels=24, audio_channels=32,
              text_dim=5120, frequency_dim=256, time_hidden_dim=5376, time_dim=2688,
              rope_frequency_dim=16, rope_theta=10000.0, norm_eps=1e-5, qk_norm_eps=1e-5, output_heads=1)
base = h3config.TransformerConfig(**fields)

SHARDED = ("to_q", "to_k", "to_v", "to_out", "gate_up", "mlp.down",
           "video_input", "audio_input", "video_output", "audio_output")

def split(name, module):
    sharded = replicated = 0
    for key, tensor in module.named_parameters():
        if any(part in key for part in SHARDED):
            sharded += tensor.numel()
        else:
            replicated += tensor.numel()
    total = sharded + replicated
    print(f"{name:22s} total {total/1e9:7.3f} B  shardable {sharded/1e9:7.3f} B ({100*sharded/total:5.1f}%)"
          f"  replicated {replicated/1e6:9.1f} M  TP2 {((sharded/2+replicated)*2)/2**30:7.3f} GiB BF16")

model = tr.Transformer(base, attention=attn.Dense(base), entries=2, steps=4, groups=4)
names = collections.Counter()
for key, tensor in model.named_parameters():
    head = ".".join(key.split(".")[2:-1]) if key.startswith("layers.") else key
    names[head] += tensor.numel()
for key, value in sorted(names.items(), key=lambda kv: -kv[1])[:14]:
    print(f"   {key:44s} {value/1e6:10.2f} M")
split("dit_base", model)
split("token_refiner", conditioning.TokenRefiner(base))
