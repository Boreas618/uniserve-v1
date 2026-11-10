"""The gather_gemm_nvfp4 package's trtllm-gen compile script.

One descriptor -> one module.

The module is trtllm-gen's routed, block-scaled first matrix -- the launch code
FlashInfer calls `PermuteGemm1::Runner` -- compiled out of the FlashInfer
distribution's data tree against the pinned `trtllm-gen-bmm` artifact, with this
package's C entry in front of it and the artifact's cubins published beside the
module under `cubins/`. The steps are the routed products' shared build,
`uniserve_program_server.trtllm.build_routed_gemm`; the kernels it selects are
the ones the manifest's `[parameters.selected]` options and the descriptor's
tile and cluster name.

A descriptor is the operator's: hidden, intermediate, experts, top, activation,
dtype, scale, layout, tile, cluster.
"""

from uniserve_program_server import trtllm

if __name__ == "__main__":
    trtllm.build_routed_gemm("uniserve_gather_gemm_nvfp4")
