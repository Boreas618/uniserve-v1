"""The sparse_attention flashinfer package's compile script.

One descriptor, one module: FlashInfer's SM120 CuTe DSL block-sparse forward
pass (:mod:`bsa_sm120`), compiled by ``uniserve_build_server.cute_dsl`` and
exported to C, then linked with this package's entry unit (which fills the
generated structs from the operator's call) and the CuTe DSL runtime.

The descriptor decides the element type and the head geometry the kernel is
compiled for; the extents, the strides, the selection's width and the softmax
scale are runtime words of the module.
"""

import bsa_sm120
import cutlass
from uniserve_build_server import cute_dsl as driver

#: The alignment, in bytes, of the query, key, value and output bases (the
#: kernel's 128-byte rows) and of the softmax normalizer's.
OPERAND_ALIGN = 128
NORMALIZER_ALIGN = 4


def spec(request):
    """The compilation of one sparse_attention descriptor."""
    element = bsa_sm120.DTYPES[request.field("dtype")]
    # Each tensor in the kernel's layout (`bsa_sm120`): the queries, keys and
    # output have the head width as their second, unit-stride mode, the values
    # and the selection's tables their first; the selection's tables and the
    # valid rows of each key block are aligned to one element.
    query, key, out = (
        driver.tensor(element, 4, 1, OPERAND_ALIGN) for _ in range(3)
    )
    value = driver.tensor(element, 4, 0, OPERAND_ALIGN)
    args = [
        query,
        key,
        value,
        out,
        driver.tensor(cutlass.Float32, 3, 0, NORMALIZER_ALIGN),
        driver.tensor(cutlass.Int32, 4, 0, None),
        driver.tensor(cutlass.Int32, 3, 0, None),
        driver.scalar(cutlass.Int32),
        driver.tensor(cutlass.Int32, 1, 0, None),
        driver.scalar(cutlass.Float32),
    ]
    return {
        "name": bsa_sm120.NAME,
        "compile": driver.kernel(
            bsa_sm120.sparse_kernel(request.fields), args, {}
        ),
    }
