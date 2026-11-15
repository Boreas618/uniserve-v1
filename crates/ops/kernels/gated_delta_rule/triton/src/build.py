"""The gated_delta_rule package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel
function, its argument signature in the order the operator launches them, the
compile-time values the descriptor gives, and the compilation options. The
Build Server's Triton source drives ``uniserve_build_server.triton`` with it.

The head counts, the head widths, the value block and the walk (one token at a
time, or chunks of ``chunk`` tokens) are all the key's: each is a constant the
program's tiles or loops are compiled over.
"""

import gated_delta_rule as kernels
from uniserve_build_server import triton as driver

#: The floating dtypes of the queries, keys, values, gate inputs and output, as Triton spells them
#: in an argument signature.
DTYPES = {"bf16": "bf16", "f16": "fp16", "f32": "fp32"}

#: The 32-bit tensor-memory columns of one sm_100 multiprocessor, which a program's products share.
TENSOR_MEMORY = 512


def spec(fields, parameters, target):
    """The compilation of one gated_delta_rule descriptor."""
    dtype = "*" + DTYPES[fields["dtype"]]
    form = fields["form"]
    chunk = int(fields["chunk"])
    if (form == "recurrent") != (chunk == 1):
        raise SystemExit(f"a {form} rule walks chunks of {chunk} tokens")
    key_heads, value_heads = int(fields["key_heads"]), int(fields["value_heads"])
    value_dim, block = int(fields["value_dim"]), int(fields["block"])
    if value_heads % key_heads or value_dim % block:
        raise SystemExit(
            f"{value_heads} value heads over {key_heads} key heads, {value_dim} columns in "
            f"blocks of {block}"
        )
    # A chunk's products hold their accumulators in tensor memory on sm_100: two chunk-square sites
    # (the key and query scores, the inverse's doubling) and four `block`-wide ones (the state's
    # reads, the new values, their scores and the state's update), out of 512 columns.
    if chunk > 1 and driver.capability(target) >= 100 and 2 * chunk + 4 * block > TENSOR_MEMORY:
        raise SystemExit(
            f"a chunk of {chunk} over {block} value columns needs {2 * chunk + 4 * block} "
            f"tensor-memory columns of {TENSOR_MEMORY}"
        )
    return {
        "aligned": [
            "query_ptr", "key_ptr", "value_ptr", "state_ptr", "checkpoint_ptr", "trail_ptr", "out_ptr"
        ],
        "kernel": kernels._gated_delta_rule_kernel,
        "signature": {
            "query_ptr": dtype,
            "key_ptr": dtype,
            "value_ptr": dtype,
            "a_ptr": dtype,
            "b_ptr": dtype,
            "a_log_ptr": dtype,
            "dt_bias_ptr": dtype,
            "offsets_ptr": "*i32",
            "seq_lens_ptr": "*i32",
            "slots_ptr": "*i32",
            "checkpoints_ptr": "*i32",
            "state_ptr": "*fp32",
            "checkpoint_ptr": "*fp32",
            "trail_ptr": "*fp32",
            "out_ptr": dtype,
            "row_stride": "i32",
            "gate_stride": "i32",
            "scale": "fp32",
            "trailing": "i32",
            "KEY_HEADS": "constexpr",
            "VALUE_HEADS": "constexpr",
            "KEY_DIM": "constexpr",
            "VALUE_DIM": "constexpr",
            "BLOCK": "constexpr",
            "CHUNK": "constexpr",
            "LEVELS": "constexpr",
        },
        "constexprs": {
            "KEY_HEADS": key_heads,
            "VALUE_HEADS": value_heads,
            "KEY_DIM": int(fields["key_dim"]),
            "VALUE_DIM": value_dim,
            "BLOCK": block,
            "CHUNK": chunk,
            # The doubling levels of a chunk's triangular inverse: log2 of the chunk.
            "LEVELS": chunk.bit_length() - 1,
        },
        # A chunk's tiles (its state, its keys, its triangular solve) need the wider warp set.
        "options": {"num_warps": 8 if chunk > 1 else 4},
    }
