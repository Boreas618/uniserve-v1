"""The nvfp4_gemm package's compile script: one descriptor -> one module.

A descriptor is the operands' element types, the activation scales' layout, what
follows the contraction and the problem's three extents. The product is a dense
``[m, n]`` tile product over packed E2M1 operands, so one module covers exactly
the problem size its key names; the manifest's dimensions admit only the scale
type, scale layout and epilogue the kernel reads, and only whole tiles.

The kernel is CUTLASS's tutorial NVFP4 GEMM, the Build Server's
``cutlass-nvfp4-gemm`` pin, with the package's ``patches/nvfp4_gemm.patch``
applied to a staged copy (:func:`kernels`). The CTA's tile is the package's
choice: 128 rows by 256 columns of the product and 256 elements of the
contraction, the smallest tile CUTLASS's block-scaled FP4 instruction pair
forms.
"""

import importlib

import cutlass
from uniserve_build_server import cute_dsl as driver
from uniserve_build_server import fetch

#: The pinned source of the kernel, and its module once the pin is loaded.
PIN = "cutlass-nvfp4-gemm"
MODULE = "tutorial_gemm.nvfp4_gemm_0"

#: The product's element types, by the operator's name.
OUTPUTS = {"bf16": cutlass.BFloat16, "f16": cutlass.Float16}

#: The CTA's product tile and the instruction's contraction step.
TILE_MN = (128, 256)
MMA_INST_SHAPE_K = 64

#: The alignment, in bytes, of every operand's base: the operands and both
#: scale fields move by TMA, and the product's rows are stored in 16-byte units.
ALIGN = 16


def kernels(request):
    """The kernel's module: the pin, patched in the build's scratch and loaded.

    The pin is fetched into the Build Server's cache from the locked
    distribution, staged as a copy (the patch changes it, the cache stays the
    distribution's bytes) under the build's output, in a dot directory the
    Build Server does not publish, and imported from there (``fetch.load``).
    The staged file stays while the build runs: the DSL reads a kernel's
    source when it traces it.
    """
    pinned = fetch.pin(PIN)
    staged = request.output / f".{PIN}"
    try:
        names = list(fetch.manifest(pinned, request.cache))
        fetch.fetch(pinned, request.cache, names, request.jobs)
        fetch.stage(pinned, request.cache, staged, names, copy=True)
        fetch.apply_patch(
            staged, request.package / "patches" / "nvfp4_gemm.patch"
        )
        fetch.load(pinned, staged)
    except fetch.FetchError as error:
        raise SystemExit(str(error)) from error
    return importlib.import_module(MODULE)


def spec(request):
    """The compilation of one nvfp4_gemm descriptor."""
    m = request.field("m")
    n = request.field("n")
    k = request.field("k")
    out_dtype = OUTPUTS[request.field("output")]
    kernel = kernels(request).Nvfp4GemmKernel(
        tile_mn=TILE_MN,
        mma_inst_shape_k=MMA_INST_SHAPE_K,
        acc_dtype=cutlass.Float32,
        out_dtype=out_dtype,
        tmem_columns=request.target.tensor_memory_columns,
    )
    return {
        "name": "nvfp4_gemm",
        "compile": driver.kernel(
            kernel.launch,
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
        "definitions": {
            "UNISERVE_NVFP4_M": m,
            "UNISERVE_NVFP4_N": n,
            "UNISERVE_NVFP4_K": k,
        },
    }
