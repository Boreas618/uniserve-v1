"""The route_gemm package's compile script: one descriptor -> one module.

FC2 of a mixture of experts: multiply each permuted row by its expert's down
projection and store the unweighted result at the row's expanded index, so the
output holds every token's uncombined routes. A descriptor is the MMA tile and
cluster (the tactic), the problem's K and N and the expert count.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

import _compile
import route_gemm as kernels

#: (MMA tile M, N) and cluster (M, N) of each tactic the operator offers; the
#: MoE sort's routing tile equals the MMA tile's M.
TACTICS = {
    "m128n128": ((128, 128), (1, 1)),
    "m128n256": ((128, 256), (1, 1)),
    "m256n128": ((256, 128), (2, 1)),
    "m256n256": ((256, 256), (2, 1)),
}

#: Elements of a BF16 row a pointer must be aligned to: 32 bytes.
ALIGN = 16


def pointer(dtype):
    """A fake device pointer of ``dtype``, as ``cute.compile`` describes one."""
    return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=ALIGN)


def spec(fields, parameters, target):
    """The compilation of one route_gemm descriptor."""
    tiler, cluster = TACTICS[fields["tactic"]]
    k = int(fields["k"])
    n = int(fields["n"])
    experts = int(fields["experts"])
    m = int(parameters["rows"])
    routes = m
    kernel = kernels.RouteGroupedGemmKernel(
        mma_tiler_mn=tiler,
        cluster_shape_mn=cluster,
    )
    return {
        "name": "route_gemm",
        "entry": "uniserve_route_gemm",
        "kernel": kernel.wrapper,
        "args": [
            pointer(cutlass.BFloat16),
            pointer(cutlass.BFloat16),
            pointer(cutlass.BFloat16),
            pointer(cutlass.Int32),
            pointer(cutlass.Int32),
            pointer(cutlass.Int32),
            pointer(cutlass.Int32),
            m,
            n,
            k,
            experts,
            routes,
        ],
        "options": {
            "tile_size": tiler[0],
            "max_active_clusters": int(parameters["max_active_clusters"]),
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
