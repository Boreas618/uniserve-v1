"""The dense_attention package's fa4 compile script: one descriptor, one module.

A descriptor is the element type, the head width, the head counts, the mask and
the window; the extents of a call (batch, tokens, offsets, the longest query
sequence) and the softmax scale are runtime words of the module, and the
package's tuning data states the forward pass's configuration
(:mod:`fa4`).
"""

import cutlass
import fa4
from uniserve_build_server import cute_dsl as driver

#: The tuning data that configures the forward pass, relative to the package.
TUNING = "tuning/forward.toml"

#: The alignment, in bytes, of the packed operands' bases (FA4's 16-byte
#: copies) and of the offset tables'.
OPERAND_ALIGN = 16
OFFSETS_ALIGN = 4


def spec(request):
    """The compilation of one dense_attention descriptor."""
    element = fa4.DTYPES[request.field("dtype")]
    # The query, key, value and output are packed `[tokens, heads, head_dim]`
    # with the head width contiguous; the offset tables are `[batch + 1]`.
    operands = [driver.tensor(element, 3, 2, OPERAND_ALIGN) for _ in range(4)]
    offsets = [
        driver.tensor(cutlass.Int32, 1, 0, OFFSETS_ALIGN) for _ in range(2)
    ]
    launcher = fa4.DenseForward(
        fa4.forward_kernel(request.fields, request.tuned(TUNING))
    )
    return {
        "name": fa4.NAME,
        "compile": driver.kernel(
            launcher,
            [
                *operands,
                driver.scalar(cutlass.Float32),
                *offsets,
                driver.scalar(cutlass.Int32),
            ],
            {},
        ),
    }
