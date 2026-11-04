"""The gather_gemm package's compile script: one descriptor -> one module.

FC1 of a mixture of experts over FlashInfer's MoE sort metadata: gather each
permuted tile's token rows, multiply by the expert's up and gate rows and store
the gated product. A descriptor is the MMA tile and cluster (the tactic), the
top-k expansion the sort metadata uses, the gated activation and the problem's
K and N, which the kernel's tile shapes are derived from.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

import _compile
import gather_gemm as kernels

#: The gated activations the kernel class implements, by the name the package
#: declares.
ACTIVATIONS = {"silu": "silu", "gelu_tanh": "gelu_tanh"}

#: (MMA tile M, N) and cluster (M, N) of each tactic the operator offers; the
#: MoE sort's routing tile equals the MMA tile's M.
TACTICS = {
    "m128n128": ((128, 128), (1, 1)),
    "m256n128": ((256, 128), (2, 1)),
    "m128n256": ((128, 256), (1, 1)),
    "m256n256": ((256, 256), (2, 1)),
}

#: Elements of a BF16 row a pointer must be aligned to: 32 bytes.
ALIGN = 16


def pointer(dtype):
    """A fake device pointer of ``dtype``, as ``cute.compile`` describes one."""
    return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=ALIGN)


def spec(fields, parameters, target):
    """The compilation of one gather_gemm descriptor."""
    tiler, cluster = TACTICS[fields["tactic"]]
    activation = ACTIVATIONS[fields["activation"]]
    k = int(fields["k"])
    n = int(fields["n"])
    experts = int(fields["experts"])
    topk = int(fields["topk"])
    # The sort metadata and the output rows are runtime sizes; the compile
    # describes one representative problem big enough for every bucket, since
    # the kernel's grids derive from the tile count at launch.
    tokens = int(parameters["tokens"])
    m = int(parameters["rows"])
    kernel = kernels.GatherGroupedGemmKernel(
        mma_tiler_mn=tiler,
        cluster_shape_mn=cluster,
        topk=topk,
        activation=activation,
    )
    return {
        "name": "gather_gemm",
        "entry": "uniserve_gather_gemm",
        "kernel": kernel.wrapper,
        "args": [
            pointer(cutlass.BFloat16),
            pointer(cutlass.BFloat16),
            pointer(cutlass.BFloat16),
            pointer(cutlass.Int32),
            pointer(cutlass.Int32),
            pointer(cutlass.Int32),
            pointer(cutlass.Int32),
            tokens,
            m,
            n,
            k,
            experts,
        ],
        "options": {
            "tile_size": tiler[0],
            "max_active_clusters": int(parameters["max_active_clusters"]),
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
