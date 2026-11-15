"""The gather_gemm package's compile script: one descriptor -> one module.

FC1 of a mixture of experts over the MoE sort's tile metadata: gather each
permuted tile's token rows, multiply by the expert's up and gate rows and store
the gated product. A descriptor is the problem's hidden and intermediate widths,
its expert count, its top-k expansion, the gated activation, the element type,
the MMA tile and the cluster, which is what the operator's key carries.
"""

import cutlass
import gather_gemm as kernels
from uniserve_build_server import cute_dsl as driver

#: The element types the kernel serves, by the operator's name.
DTYPES = {"bf16": cutlass.BFloat16}

#: The alignment, in bytes, of every operand's base: the activation rows are
#: gathered with 16-byte copies and the weights and the product move by TMA.
ALIGN = 16


def spec(request):
    """The compilation of one gather_gemm descriptor."""
    tiler = tuple(request.field("tile"))
    element = DTYPES[request.field("dtype")]
    # The kernel class names its gated activations as the operator does, and
    # the manifest's `activation` dimension admits only the ones it implements.
    kernel = kernels.GatherGroupedGemmKernel(
        mma_tiler_mn=tiler,
        cluster_shape_mn=tuple(request.field("cluster")),
        topk=request.field("top"),
        activation=request.field("activation"),
        smem_capacity=request.target.shared_memory,
        tmem_columns=request.target.tensor_memory_columns,
    )
    # The token count, the permuted rows, both widths and the expert count are
    # the call's runtime words: the kernel's grids derive from the tile count
    # at launch, so the compilation states only their types.
    return {
        "name": "gather_gemm",
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
                # The multiprocessors of the device the launch runs on.
                driver.scalar(cutlass.Int32),
            ],
            {"tile_size": tiler[0]},
        ),
    }
