"""The gather_gemm_mxfp4 package's trtllm-gen build module.

One descriptor -> one module.

The module is trtllm-gen's routed, block-scaled first matrix over MXFP8
activations and MXFP4 weights -- the launch code FlashInfer calls
`PermuteGemm1::Runner` -- compiled out of the FlashInfer distribution's data
tree against the pinned `trtllm-gen-bmm` artifact, with this package's C entry
(`[source] unit`) in front of it and the artifact's cubins published beside
the module under `cubins/`. The steps are the routed products' shared build,
`uniserve_build_server.trtllm.build_routed_gemm`, and the kernels it selects
are the ones the manifest's `[parameters.selected]` options and the
descriptor's tile and cluster name.

A descriptor is the operator's: hidden, intermediate, experts, top, tile,
cluster.
"""

from uniserve_build_server import trtllm
from uniserve_build_server.request import Request


def compile(request: Request) -> None:
    """Compile the request's descriptor into ``module.so`` with its cubins."""
    trtllm.build_routed_gemm(request)
