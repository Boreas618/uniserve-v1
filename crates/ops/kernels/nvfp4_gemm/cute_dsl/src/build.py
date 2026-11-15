"""The nvfp4_gemm package's compile script: one descriptor -> one module.

A descriptor is the operands' element types, the activation scales' layout, what
follows the contraction and the problem's three extents. The product is a dense
``[m, n]`` tile product over packed E2M1 operands, so one module covers exactly
the problem size its key names; the manifest's dimensions admit only the scale
type, scale layout and epilogue the kernel reads, and only whole tiles.

The CTA's tile is the package's choice: 128 rows by 256 columns of the product
and 256 elements of the contraction, the smallest tile CUTLASS's block-scaled
FP4 instruction pair forms.
"""

import cutlass
import nvfp4_gemm as kernels
from uniserve_build_server import cute_dsl as driver

#: The product's element types, by the operator's name.
OUTPUTS = {"bf16": cutlass.BFloat16, "f16": cutlass.Float16}

#: The CTA's product tile and the instruction's contraction step.
TILE_MN = (128, 256)
MMA_INST_SHAPE_K = 64

#: The alignment, in bytes, of every operand's base: the operands and both
#: scale fields move by TMA, and the product's rows are stored in 16-byte units.
ALIGN = 16


def spec(request):
    """The compilation of one nvfp4_gemm descriptor."""
    m = request.field("m")
    n = request.field("n")
    k = request.field("k")
    out_dtype = OUTPUTS[request.field("output")]
    return {
        "name": "nvfp4_gemm",
        "compile": driver.kernel(
            kernels.Nvfp4GemmKernel(
                tile_mn=TILE_MN,
                mma_inst_shape_k=MMA_INST_SHAPE_K,
                acc_dtype=cutlass.Float32,
                out_dtype=out_dtype,
                tmem_columns=request.target.tensor_memory_columns,
            ).launch,
            # The operator packs the activation, the weight, both scale
            # matrices, the alpha and the output, then the three extents, which
            # the kernel compiles in.
            [
                driver.pointer(cutlass.Float4E2M1FN, ALIGN),
                driver.pointer(cutlass.Float4E2M1FN, ALIGN),
                driver.pointer(cutlass.Float8E4M3FN, ALIGN),
                driver.pointer(cutlass.Float8E4M3FN, ALIGN),
                driver.pointer(cutlass.Float32, ALIGN),
                driver.pointer(out_dtype, ALIGN),
                m,
                n,
                k,
            ],
            {},
        ),
        # The module states the problem it was compiled for, so a call of other
        # extents is refused instead of run against the compiled tiles.
        "defines": [
            f"UNISERVE_NVFP4_M={m}",
            f"UNISERVE_NVFP4_N={n}",
            f"UNISERVE_NVFP4_K={k}",
        ],
    }
