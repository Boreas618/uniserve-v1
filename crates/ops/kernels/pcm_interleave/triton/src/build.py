"""The pcm_interleave package's compile script: one descriptor -> one module.

``spec`` states what this package's kernel needs for one descriptor: the kernel
function, its argument signature in the order the operator launches them, the
compile-time values the descriptor dimensions give, and the compilation options.
The Program Server's Triton source drives ``_compile`` with it.

Two things are compiled in. ``BLOCK``, the samples one program converts, so the
grid a launch computes and the module agree. And ``PDL``: a module compiled for
programmatic dependent launch waits for its predecessor and releases its successor,
which only Blackwell's capability 100 has.

What is *not* compiled in is the sample count: a request's duration varies while
the arithmetic does not, so the count is a word and one module serves every clip.
The output is always i16 and is never in the key.
"""

import _compile
import pcm_interleave as kernels


def warps_of(block):
    """The warp count one block of samples needs: two loads and two stores a sample."""
    return max(1, min(8, block // 512))


def spec(fields, parameters, target):
    """The compilation of one pcm_interleave descriptor."""
    block = int(fields["block"])
    pdl = fields["pdl"] == "true" and _compile.capability(target) >= 100
    return {
        "kernel": kernels._pcm_interleave_kernel,
        "signature": {
            "left": "*fp32",
            "right": "*fp32",
            "out": "*i16",
            "samples": "i32",
            "block": "i32",
            "pdl": "i32",
            "BLOCK": "constexpr",
            "PDL": "constexpr",
        },
        "constexprs": {"BLOCK": block, "PDL": pdl},
        "options": {"num_warps": warps_of(block), "launch_pdl": pdl},
    }


if __name__ == "__main__":
    _compile.main(spec)
