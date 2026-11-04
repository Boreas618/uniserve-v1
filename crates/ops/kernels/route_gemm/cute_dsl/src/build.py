"""The route_gemm package's compile script: one descriptor -> one module.

FC2 of a mixture of experts over the MoE sort's tile metadata: multiply each
permuted tile's rows by the expert's down projection and store each route's row
of the model's hidden width. A descriptor is the problem's hidden and
intermediate widths, its expert count, its top-k expansion, the element type,
the MMA tile and the cluster, which is what the operator's key carries.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

import _compile
import route_gemm as kernels

#: The element types the kernel serves, by the operator's name.
DTYPES = {"bf16": cutlass.BFloat16}

#: Elements of a row a pointer must be aligned to: 32 bytes.
ALIGN = 16


def shape_of(text):
    """The (rows, columns) a `MxN` descriptor value names."""
    rows, columns = text.split("x")
    return (int(rows), int(columns))


def pointer(dtype):
    """A fake device pointer of ``dtype``, as ``cute.compile`` describes one."""
    return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=ALIGN)


def spec(fields, parameters, target):
    """The compilation of one route_gemm descriptor."""
    tiler = shape_of(fields["tile"])
    cluster = shape_of(fields["cluster"])
    element = DTYPES[fields["dtype"]]
    experts = int(fields["experts"])
    hidden = int(fields["hidden"])
    intermediate = int(fields["intermediate"])
    # The sort metadata and the row counts are runtime sizes; the compile
    # describes one representative problem big enough for every bucket.
    permuted_rows = int(parameters["permuted_rows"])
    expanded_rows = int(parameters["expanded_rows"])
    kernel = kernels.RouteGroupedGemmKernel(
        mma_tiler_mn=tiler,
        cluster_shape_mn=cluster,
    )
    return {
        "name": "route_gemm",
        "entry": "uniserve_route_gemm",
        "kernel": kernel.wrapper,
        "args": [
            pointer(element),
            pointer(element),
            pointer(element),
            pointer(cutlass.Int32),
            pointer(cutlass.Int32),
            pointer(cutlass.Int32),
            pointer(cutlass.Int32),
            permuted_rows,
            hidden,
            intermediate,
            experts,
            expanded_rows,
        ],
        "options": {
            "tile_size": tiler[0],
            "max_active_clusters": int(parameters["max_active_clusters"]),
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
