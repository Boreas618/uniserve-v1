"""The route_gemm_nvfp4 package's trtllm-gen build module.

One descriptor -> one module.

The module is trtllm-gen's routed, block-scaled second matrix -- the launch
code FlashInfer calls `Gemm2::Runner` -- compiled out of the FlashInfer
distribution's data tree against the pinned `trtllm-gen-bmm` artifact, with
this package's C entry (`[source] unit`) in front of it and the artifact's
cubins published beside the module under `cubins/`. The Build Server compiles
a descriptor by calling :func:`compile` with the compile request; the steps are
the routed products' shared build,
`uniserve_build_server.trtllm.build_routed_gemm`, and the kernels it selects
are the ones the manifest's `[parameters.selected]` options and the
descriptor's tile and cluster name.

A descriptor is the operator's: hidden, intermediate, experts, top, dtype,
scale, layout, tile, cluster.
"""

from uniserve_build_server import trtllm
from uniserve_build_server.request import Request

#: The eltwise activation each of the operator's activations is, in the
#: artifact's own enumeration (`gemm::EltwiseActType`). This product is linear,
#: so it is `none`; the `relu2` family beside it applies a squared ReLU and is a
#: different kernel.
ELTWISE_ACTIVATIONS = {"none": 0, "gelu": 1, "relu2": 2}


def compile(request: Request) -> None:
    """Compile the request's descriptor into ``module.so`` with its cubins.

    The manifest selects `mEltwiseActType`, an option only this product's
    kernels carry, so its rendering is added to the shared ones.
    """
    trtllm.build_routed_gemm(
        request,
        renderings={
            "mEltwiseActType": lambda name: (
                f"gemm::EltwiseActType({ELTWISE_ACTIVATIONS[name]})"
            )
        },
    )
