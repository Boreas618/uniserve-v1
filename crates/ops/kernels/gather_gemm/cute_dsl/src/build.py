"""The gather_gemm package's compile script: one descriptor -> one module.

FC1 of a mixture of experts over the MoE sort's tile metadata: gather each
permuted tile's token rows, multiply by the expert's up and gate rows and store
the gated product. A descriptor is the problem's hidden and intermediate widths,
its expert count, its top-k expansion, the gated activation, the element type,
the MMA tile and the cluster, which is what the operator's key carries.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

import _compile
import gather_gemm as kernels

#: The gated activations the kernel class implements, by the operator's name.
ACTIVATIONS = {"silu": "silu", "gelu_tanh": "gelu_tanh"}

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
    """The compilation of one gather_gemm descriptor."""
    tiler = shape_of(fields["tile"])
    cluster = shape_of(fields["cluster"])
    activation = ACTIVATIONS[fields["activation"]]
    element = DTYPES[fields["dtype"]]
    experts = int(fields["experts"])
    top = int(fields["top"])
    hidden = int(fields["hidden"])
    intermediate = int(fields["intermediate"])
    # The sort metadata and the output rows are runtime sizes; the compile
    # describes one representative problem big enough for every bucket, since
    # the kernel's grids derive from the tile count at launch.
    tokens = int(parameters["tokens"])
    permuted_rows = int(parameters["permuted_rows"])
    kernel = kernels.GatherGroupedGemmKernel(
        mma_tiler_mn=tiler,
        cluster_shape_mn=cluster,
        topk=top,
        activation=activation,
    )
    return {
        "name": "gather_gemm",
        "entry": "uniserve_gather_gemm",
        "kernel": kernel.wrapper,
        "args": [
            pointer(element),
            pointer(element),
            pointer(element),
            pointer(cutlass.Int32),
            pointer(cutlass.Int32),
            pointer(cutlass.Int32),
            pointer(cutlass.Int32),
            tokens,
            permuted_rows,
            2 * intermediate,
            hidden,
            experts,
        ],
        "options": {
            "tile_size": tiler[0],
            "max_active_clusters": int(parameters["max_active_clusters"]),
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
