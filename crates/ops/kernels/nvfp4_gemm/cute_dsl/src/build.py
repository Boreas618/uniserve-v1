"""The nvfp4_gemm package's compile script: one descriptor -> one module.

A descriptor is the operands' element types, the activation scales' layout, what
follows the contraction, the problem's three extents and the dependent-launch
flag. The product is a dense ``[m, n]`` tile product over packed E2M1 operands,
so one module covers exactly the problem size its key names.

The CTA's tile is the package's choice: 128 rows by 256 columns of the product
and 256 elements of the contraction, the smallest tile CUTLASS's block-scaled FP4
instruction pair forms. A descriptor's ``m`` and ``n`` must therefore be whole
tiles.
"""

import cutlass
import cutlass.cute as cute
import nvfp4_gemm as kernels
from cutlass.cute.runtime import make_ptr

import _compile

#: The product's element types, by the operator's name.
OUTPUTS = {"bf16": cutlass.BFloat16, "f16": cutlass.Float16}

#: The CTA's product tile and the instruction's contraction step.
TILE_MN = (128, 256)
MMA_INST_SHAPE_K = 64

#: Elements of a row the operands' pointers are aligned to: the packed E2M1 rows
#: are 32-byte aligned and the product's rows 16.
ALIGN = 16


def pointer(dtype):
    """A fake device pointer of ``dtype``, as ``cute.compile`` describes one."""
    return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=ALIGN)


def spec(fields, parameters, target):
    """The compilation of one nvfp4_gemm descriptor."""
    m = int(fields["m"])
    n = int(fields["n"])
    k = int(fields["k"])
    if fields["layout"] != "linear":
        raise SystemExit(f"the {fields['layout']} activation scale layout is not served")
    if fields["epilogue"] != "none":
        raise SystemExit(f"the {fields['epilogue']} epilogue is not served")
    if m % TILE_MN[0] != 0 or n % TILE_MN[1] != 0 or k % (MMA_INST_SHAPE_K * 4) != 0:
        raise SystemExit(f"{m}x{n}x{k} is not whole {TILE_MN[0]}x{TILE_MN[1]} tiles of 256")
    out_dtype = OUTPUTS[fields["output"]]
    return {
        "name": "nvfp4_gemm",
        "entry": "uniserve_nvfp4_gemm",
        "kernel": kernels.Nvfp4GemmKernel(
            tile_mn=TILE_MN,
            mma_inst_shape_k=MMA_INST_SHAPE_K,
            acc_dtype=cutlass.Float32,
            out_dtype=out_dtype,
        ).launch,
        # The operator packs the activation, the weight, both scale matrices, the
        # alpha and the output, then the three extents.
        "args": [
            pointer(cutlass.Float4E2M1FN),
            pointer(cutlass.Float4E2M1FN),
            pointer(cutlass.Float8E4M3FN),
            pointer(cutlass.Float8E4M3FN),
            pointer(cutlass.Float32),
            pointer(out_dtype),
            m,
            n,
            k,
        ],
        "options": {},
        # The module states the problem it was compiled for, so a call of other
        # extents is refused instead of run against the compiled tiles.
        "defines": [
            f"UNISERVE_NVFP4_M={m}",
            f"UNISERVE_NVFP4_N={n}",
            f"UNISERVE_NVFP4_K={k}",
        ],
    }


if __name__ == "__main__":
    _compile.main(spec)
