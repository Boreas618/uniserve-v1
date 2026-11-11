"""The fp8_gemm package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
gives, and the compilation options. The Build Server's Triton source drives
``uniserve_build_server.triton`` with it.

The tile and the bias flag are baked in; the three extents, the activation's row stride and the
two scales are launch words, so one module serves every projection of one activation type.
"""

import fp8_gemm as kernels
from uniserve_build_server import triton as driver

#: The 16-bit types an activation and its output may have, as Triton spells them in a signature.
DTYPES = {"bf16": "bf16", "f16": "fp16"}


def spec(fields, parameters, target):
    """The compilation of one fp8_gemm descriptor."""
    if driver.capability(target) < 89:
        raise SystemExit(f"E4M3 products need sm_89 or newer, not {target}")
    dtype = "*" + DTYPES[fields["dtype"]]
    block_m, block_n, block_k = (int(fields[name]) for name in ("block_m", "block_n", "block_k"))
    return {
        "kernel": kernels._fp8_gemm_kernel,
        "signature": {
            "x_ptr": dtype,
            "w_ptr": "*fp8e4nv",
            "bias_ptr": dtype,
            "out_ptr": dtype,
            "m": "i32",
            "n": "i32",
            "k": "i32",
            "x_row_stride": "i32",
            "inverse_scale": "fp32",
            "alpha": "fp32",
            "BLOCK_M": "constexpr",
            "BLOCK_N": "constexpr",
            "BLOCK_K": "constexpr",
            "BIAS": "constexpr",
        },
        "constexprs": {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "BIAS": fields["bias"] == "true",
        },
        "options": {"num_warps": 4 if block_m * block_n <= 64 * 64 else 8, "num_stages": 3},
        # Every operand is an arena tensor or a weight, whose rows start on sixteen-byte bounds.
        "aligned": ["x_ptr", "w_ptr", "out_ptr"],
    }


if __name__ == "__main__":
    driver.main(spec)
