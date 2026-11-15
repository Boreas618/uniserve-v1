"""The programmatic dependent launch prologue of a Triton kernel.

A module compiled for a dependent launch (its ``PDL`` parameter, which the
Triton driver binds from the descriptor's ``pdl`` field and the target) starts
before the grid it depends on has finished. Its first statement waits for that
grid's memory and then lets the grid after it start, so the next kernel's
launch overlaps this one's work while every read still sees what the previous
kernel wrote.
"""

import triton
import triton.language as tl


@triton.jit
def prologue(PDL: tl.constexpr):  # noqa: N803
    """Wait for the preceding grid, then release the next one."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
