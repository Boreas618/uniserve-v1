"""The route_gemm package's compile script: one descriptor -> one module.

FC2 of a mixture of experts over the MoE sort's tile metadata: multiply each
permuted tile's rows by the expert's down projection and store each route's row
of the model's hidden width. A descriptor is the problem's hidden and
intermediate widths, its expert count, its top-k expansion, the element type,
the MMA tile and the cluster, which is what the operator's key carries.
"""

import cutlass
import route_gemm as kernels
from uniserve_build_server import cute_dsl as driver

#: The element types the kernel serves, by the operator's name.
DTYPES = {"bf16": cutlass.BFloat16}

#: The alignment, in bytes, of every operand's base: the operands move by TMA
#: and each product row by a bulk copy of whole 16-byte units.
ALIGN = 16


def spec(fields, parameters, target):
    """The compilation of one route_gemm descriptor."""
    tiler = tuple(fields["tile"])
    element = DTYPES[fields["dtype"]]
    kernel = kernels.RouteGroupedGemmKernel(
        mma_tiler_mn=tiler,
        cluster_shape_mn=tuple(fields["cluster"]),
    )
    # The row counts, both widths and the expert count are the call's runtime
    # words: the kernel's grids derive from the tile count at launch, so the
    # compilation states only their types.
    return {
        "name": "route_gemm",
        "compile": driver.kernel(
            kernel.wrapper,
            [
                driver.pointer(element, ALIGN),
                driver.pointer(element, ALIGN),
                driver.pointer(element, ALIGN),
                driver.pointer(cutlass.Int32, ALIGN),
                driver.pointer(cutlass.Int32, ALIGN),
                driver.pointer(cutlass.Int32, ALIGN),
                driver.pointer(cutlass.Int32, ALIGN),
                driver.scalar(cutlass.Int64),
                driver.scalar(cutlass.Int64),
                driver.scalar(cutlass.Int64),
                driver.scalar(cutlass.Int64),
                driver.scalar(cutlass.Int64),
            ],
            {
                "tile_size": tiler[0],
                "max_active_clusters": int(parameters["max_active_clusters"]),
            },
        ),
    }
