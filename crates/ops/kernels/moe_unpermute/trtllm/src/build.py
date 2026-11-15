"""The moe_unpermute package's build module: one descriptor -> one module.

The module is trtllm-gen's combine kernel, compiled from the FlashInfer
distribution's sources (`uniserve_build_server.trtllm.COMBINE`) into a shared
object with this package's C entry in front of it. A descriptor is the
problem's hidden width, top-k expansion, expert count and the three element
types; the kernel reads them at launch, so every descriptor of this package
compiles the same module.

The sources come from the FlashInfer package the toolchain pins, not from this
tree: the kernel is NVIDIA's, and the package's `[toolchain] packages` names
where it comes from.

The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.
"""

from uniserve_build_server import trtllm
from uniserve_build_server.request import Request


def compile(request: Request) -> None:
    """Compile the request's descriptor into ``module.so``."""
    trtllm.build(request, trtllm.COMBINE)
