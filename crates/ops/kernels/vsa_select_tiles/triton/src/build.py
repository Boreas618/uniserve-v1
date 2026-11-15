"""The vsa_select_tiles package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel function, its
argument signature in the order the operator launches them, the compile-time values the descriptor
dimensions give, and the compilation options. The Build Server's Triton source drives
``uniserve_build_server.triton`` with it.

The key-tile lane width is the whole descriptor: it bounds the tiles one query tile ranks, and the
kernel reduces over those lanes. The tile count, the query-tile count and the window offset are
launch words.
"""

import vsa_select_tiles as kernels


def spec(fields, parameters, target):
    """The compilation of one vsa_select_tiles descriptor."""
    return {
        "kernel": kernels._select_tiles_kernel,
        "signature": {
            "scores": "*fp32",
            "tile_segments": "*i32",
            "variable_block_sizes": "*i32",
            "segment_keep": "*i32",
            "indices": "*i32",
            "counts": "*i32",
            "query_tiles": "i32",
            "tiles": "i32",
            "query_offset": "i32",
            "block": "constexpr",
        },
        "constexprs": {"block": int(fields["block"])},
        "options": {"num_warps": 4},
    }
