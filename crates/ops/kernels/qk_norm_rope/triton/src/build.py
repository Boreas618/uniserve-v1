"""The qk_norm_rope package's compile script: one descriptor -> one module.

A descriptor is the row dtype, the head width, the Q and K head counts, the
factor table dtype, the rounding recipe and the row layout. The kernel's domains
and rotary axes are the head's single full-head domain and single axis, as
Qwen3-MoE's rotary projection asks for one normalization weight per tensor and
one split-half factor pair. Programs own complete token/head rows, so outputs
may alias their sources.
"""

import _compile
import qk_norm_rope as kernels

#: The floating dtypes of Q, K, their weights and their outputs.
DTYPES = {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}

#: The factor table dtypes.
FACTORS = {"fp32": "fp32", "bf16": "bf16"}

#: Token/head row layouts: rows are [tokens, heads, dim] with a unit feature
#: stride, so the token stride is heads * dim and the head stride is dim.
LAYOUTS = {"tokens_heads": True}


def rows_per_program(block):
    """Rows one program covers: about 2048 features per tile, at most 16."""
    return min(16, max(1, 2048 // block))


def spec(fields, parameters, target):
    """The compilation of one qk_norm_rope descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    factors = "*" + FACTORS[fields["factors"]]
    dim = int(fields["head_dim"])
    q_heads = int(fields["q_heads"])
    k_heads = int(fields["k_heads"])
    if fields["layout"] not in LAYOUTS:
        raise SystemExit(f"layout {fields['layout']} has no kernel in this package")
    block = 1 << (dim - 1).bit_length()
    rows = rows_per_program(block)
    stepwise = fields["rounding"] == "stepwise"
    pdl = _compile.capability(target) >= 100
    # One full-head RMS domain per tensor, one split-half rotary axis over the
    # complete head: starts at feature 0, ends at the head width.
    q_strides = (q_heads * dim, dim)
    k_strides = (k_heads * dim, dim)
    # The compact factor table of one token and axis per row of `dim / 2`
    # factors, contiguous.
    factor_stride = dim // 2
    return {
        "kernel": kernels._qk_norm_rope_kernel,
        "signature": {
            "q": dtype,
            "k": dtype,
            "q_weights": (dtype,),
            "k_weights": (dtype,),
            "cosines": (factors,),
            "sines": (factors,),
            "q_out": dtype,
            "k_out": dtype,
            "q_rows": "i32",
            "k_rows": "i32",
            "Q_HEADS": "constexpr",
            "K_HEADS": "constexpr",
            "Q_STRIDES": "constexpr",
            "K_STRIDES": "constexpr",
            "Q_OUT_STRIDES": "constexpr",
            "K_OUT_STRIDES": "constexpr",
            "DIM": "constexpr",
            "DOMAIN_STARTS": "constexpr",
            "DOMAIN_ENDS": "constexpr",
            "AXIS_STARTS": "constexpr",
            "AXIS_ROTATED": "constexpr",
            "FACTOR_STRIDES": "constexpr",
            "EPS": "constexpr",
            "BLOCK": "constexpr",
            "ROWS": "constexpr",
            "PDL": "constexpr",
            "HALVES": "constexpr",
            "STEPWISE": "constexpr",
        },
        "constexprs": {
            "Q_HEADS": q_heads,
            "K_HEADS": k_heads,
            "Q_STRIDES": q_strides,
            "K_STRIDES": k_strides,
            "Q_OUT_STRIDES": q_strides,
            "K_OUT_STRIDES": k_strides,
            "DIM": dim,
            "DOMAIN_STARTS": (0,),
            "DOMAIN_ENDS": (dim,),
            "AXIS_STARTS": (0,),
            "AXIS_ROTATED": (dim,),
            "FACTOR_STRIDES": (factor_stride,),
            "EPS": float(parameters["eps"]),
            "BLOCK": block,
            "ROWS": rows,
            "PDL": pdl,
            "HALVES": False,
            "STEPWISE": stepwise,
        },
        "options": {
            "num_warps": 4,
            "launch_pdl": pdl,
            "enable_fp_fusion": not stepwise,
        },
    }


if __name__ == "__main__":
    _compile.main(spec)
