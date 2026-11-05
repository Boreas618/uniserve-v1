"""Weights per component, from the reference's own modules on the meta device.

Shapes come from the shipped sidecars and the reference's constructors; no weight file is read, so
every figure here is a count of parameters a component holds, at its own dtype, not a file size.
"""

import collections
import sys
import types
from collections import Counter

import torch

pkg = types.ModuleType("uniserve_kernels")
tri = types.ModuleType("uniserve_kernels.triton")
tri.require_kernel = lambda *a, **k: (lambda f: f)
pkg.triton = tri
sys.modules["uniserve_kernels"] = pkg
sys.modules["uniserve_kernels.triton"] = tri

# The model modules read the video task list from the worker library at import; the probe serves it
# the three tasks the request rules name, which is all that check needs.
worker = types.ModuleType("uniserve.worker")
_CANVASES = (1536, 672, 1344, 768, 1024, 768, 768, 768, 768, 1024, 768, 1344)
_rules = {
    "video_tasks": lambda: ("t2va", "fl2va", "ref2va"),
    "video_canvas": lambda *a, **k: _CANVASES,
}
worker.library = lambda: collections.defaultdict(
    lambda: (lambda *a, **k: _CANVASES), _rules
)
sys.modules["uniserve.worker"] = worker

# The components are measured at the deployment's representation: BF16 is what the checkpoint holds
# and what a rank reserves, so a count here is the note's figure and not twice it.
torch.set_default_device("meta")
torch.set_default_dtype(torch.bfloat16)

from uniserve.models.minimax_h3 import attention as attn
from uniserve.models.minimax_h3 import audio_vae as avae
from uniserve.models.minimax_h3 import conditioning, decoding
from uniserve.models.minimax_h3 import config as h3config
from uniserve.models.minimax_h3 import transformer as tr
from uniserve.models.minimax_h3 import video_vae

if len(sys.argv) < 2:
    sys.exit("usage: measure_params.py <checkpoint root>")
ROOT = sys.argv[1]


def bytes_by_dtype(module):
    counts = Counter()
    for tensor in module.parameters():
        counts[str(tensor.dtype)] += tensor.numel() * tensor.element_size()
    return counts


def show(name, module):
    counts = bytes_by_dtype(module)
    total = sum(counts.values())
    detail = ", ".join(f"{dtype} {value / 2**30:.3f}" for dtype, value in sorted(counts.items()))
    print(f"{name:34s} {total / 2**30:9.3f} GiB  ({detail})")
    return total


def transformer_config(**overrides):
    fields = dict(
        hidden_size=5376,
        num_attention_heads=56,
        head_dim=128,
        num_hidden_layers=50,
        num_refiner_layers=2,
        intermediate_size=14336,
        video_channels=24,
        audio_channels=32,
        text_dim=5120,
        frequency_dim=256,
        time_hidden_dim=5376,
        time_dim=2688,
        rope_frequency_dim=16,
        rope_theta=10000.0,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
        output_heads=1,
    )
    fields.update(overrides)
    return h3config.TransformerConfig(**fields)


def sidecar(path, keys):
    """A shipped config's own fields, as the keywords its frozen dataclass takes."""
    import json

    values = json.load(open(path))
    kept = {key: value for key, value in values.items() if key in keys}

    # The configs are frozen dataclasses whose sequences are tuples; a JSON array is a list.
    def tuples(value):
        if isinstance(value, list):
            return tuple(tuples(item) for item in value)
        return value

    return {name: tuples(value) for name, value in kept.items()}


def video_config():
    return video_vae.Config(**sidecar(f"{ROOT}/vae/config.json", video_vae.Config.__dataclass_fields__))


def audio_config():
    return avae.Config(**sidecar(f"{ROOT}/audio_vae/config.json", avae.Config.__dataclass_fields__))


def main():
    base = transformer_config()
    dense = attn.Dense(base)
    total = 0
    total += show("DiT base (50 layers, dense)", tr.Transformer(base, attention=dense, entries=2, steps=4, groups=4))

    refiner = conditioning.TokenRefiner(base)
    total += show("token refiner (2 layers)", refiner)
    conditioner = conditioning.Conditioner(base)
    total += show("conditioner", conditioner)

    student = transformer_config(output_heads=8)
    total += show("DiT student (8 interval heads)", tr.Transformer(student, attention=attn.Dense(student), entries=2, steps=8, groups=8))

    vae = video_config()
    total += show("video vae decoder", video_vae.Decoder(vae))
    total += show("video vae encoder", video_vae.Encoder(vae))

    audio = audio_config()
    total += show("audio vae model", avae.Model(audio))
    total += show("audio vae encoder", avae.Encoder(audio))
    print(f"{'every component together':34s} {total / 2**30:9.3f} GiB")


if __name__ == "__main__" and "--components" not in sys.argv:
    main()


def components():
    """Parameter counts of the modules a deployment places, element counts first."""
    counts = {}

    def count(name, module):
        counts[name] = sum(t.numel() for t in module.parameters())
        print(f"{name:34s} {counts[name]:>16,} params  BF16 {counts[name] * 2 / 2**30:8.3f} GiB")

    base = transformer_config()
    count("dit_base", tr.Transformer(base, attention=attn.Dense(base), entries=2, steps=4, groups=4))
    count("dit_refiner_2l", conditioning.TokenRefiner(base))
    count("conditioner", conditioning.Conditioner(base))
    student = transformer_config(output_heads=8)
    count("dit_student_8h", tr.Transformer(student, attention=attn.Dense(student), entries=2, steps=8, groups=8))

    vae = video_config()
    count("video_vae_decoder", video_vae.Decoder(vae))
    count("video_vae_encoder", video_vae.Encoder(vae))

    audio = audio_config()
    count("audio_vae_model", avae.Model(audio))
    count("audio_vae_encoder", avae.Encoder(audio))
    return counts


if __name__ == "__main__" and "--components" in sys.argv:
    components()
