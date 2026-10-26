"""Every resident weight in the layout its kernels read.

Loading ends by storing each encoded weight in the layout the kernel
provider serving it declares, once, before the weights are published:

- a block-scaled (MXFP8 or NVFP4) projection weight in the block-scale
  layout of the matmul provider ``auto`` resolves for it
  (``uniserve.backends.matmul.Backend.scale_layout``);
- routed expert weights in the scale layout and row orders of the expert
  kernels that serve the layer (``uniserve.backends.moe.Backend.
  weight_layout``): the provider ``auto`` resolves on the layer's device,
  or the fused or split MegaMoE kernels an expert exchange runs them with.

Preparation verifies these layouts and never rearranges a weight, so the
bytes a worker retains are exactly the bytes its kernels read, and a
read-only mapping of them serves unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import torch
from torch import nn

from uniserve.quantization import QuantizedTensor

# Which kernels read a model's routed experts: the expert provider ``auto``
# resolves on the layer's device, the fused MegaMoE kernel of a ``megamoe``
# exchange over colocated experts, or split MegaMoE's expert kernels on
# dedicated expert ranks.
ExpertKernels = Literal["local", "megamoe", "split"]

# Encodings whose weights carry block scales in a provider-declared layout.
_BLOCK_SCALED = frozenset(("mxfp8", "nvfp4"))


def place(module: nn.Module, layout) -> None:
    """Store a ``FusedMoE``'s expert weights in ``layout``.

    ``layout`` maps projection names (``up_gate``, ``down``) to the scale
    layout and row order a provider reads them in
    (``uniserve.backends.moe.Backend.weight_layout``). Each rearranged
    weight replaces its parameter, so one copy stays resident; the transient
    copy is one projection of one layer.

    Raises:
        ValueError: A named projection's weight is not encoded, or cannot be
            stored in the layout (``QuantizedTensor.repack``).
    """
    for name, (scale_layout, row_order) in layout.items():
        linear = getattr(module, name)
        weight = linear.weight
        if not isinstance(weight, QuantizedTensor):
            raise ValueError(
                f"{name} weights are dense; only encoded expert weights "
                "take a provider layout"
            )
        placed = weight.repack(scale_layout=scale_layout, row_order=row_order)
        if placed is not weight:
            linear.weight = nn.Parameter(placed, requires_grad=False)


@torch.no_grad()
def finalize(model: nn.Module, *, experts: ExpertKernels = "local") -> None:
    """Store every resident encoded weight of ``model`` in its kernels' layout.

    Meta and host weights are left as they are. A routed expert layer no
    provider serves keeps its loaded layout; its preparation reports why.
    The conversions complete before returning, so any stream may read the
    weights afterwards.

    Args:
        model: A loaded model.
        experts: Which kernels read the routed experts (``ExpertKernels``).
    """
    from uniserve.backends import matmul, moe
    from uniserve.backends.experts import split
    from uniserve.backends.moe import megamoe
    from uniserve.nn.linear import Linear
    from uniserve.nn.moe import FusedMoE

    provider: moe.Backend
    # Every (module, name) holding each parameter: a replaced parameter is
    # installed on all of them, preserving shared identity.
    owners: dict[int, list[tuple[nn.Module, str]]] = defaultdict(list)
    for _, module in model.named_modules(remove_duplicate=False):
        for name, parameter in module._parameters.items():
            if parameter is not None:
                owners[id(parameter)].append((module, name))

    devices = set()
    for module in model.modules():
        if isinstance(module, FusedMoE):
            weight = module.up_gate.weight
            if weight.device.type != "cuda":
                continue
            devices.add(weight.device)
            if experts == "split":
                layout = split.weight_layout(module)
            elif experts == "megamoe":
                provider = megamoe.Backend()
                if provider.unsupported(module) is not None:
                    continue
                layout = provider.weight_layout(module)
            else:
                try:
                    provider = moe.resolve(
                        "auto", module=module, device=weight.device
                    )
                except ValueError:
                    continue
                layout = provider.weight_layout(module)
            place(module, layout)
        elif isinstance(module, Linear):
            weight = module.weight
            if (
                not isinstance(weight, QuantizedTensor)
                or weight.quantizer.format not in _BLOCK_SCALED
                or weight.device.type != "cuda"
            ):
                continue
            devices.add(weight.device)
            target = matmul.resolve("auto", weight).scale_layout
            placed = weight.repack(scale_layout=target)
            if placed is weight:
                continue
            resident = nn.Parameter(placed, requires_grad=False)
            for owner, name in owners[id(weight)]:
                owner._parameters[name] = resident

    for device in devices:
        torch.cuda.current_stream(device).synchronize()
