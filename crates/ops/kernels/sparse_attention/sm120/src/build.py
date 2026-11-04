"""The sparse_attention sm120 package's compile script: one descriptor -> one module.

The kernel is FlashInfer's SM120 CuTe DSL block-sparse forward pass, so the module is produced the
way the FA4 packages' are: `_compile` compiles the kernel through ``bsa_sm120.BsaCompiler`` and
exports it to C, then links the exported launcher with this package's entry unit (which fills the
generated structs from the operator's call) and the CuTe DSL runtime.

The descriptor decides the element type and the head geometry the kernel is compiled for; the
selection's own widths and the extents are launch values, which is why they are the package's
parameters rather than its dimensions.
"""

import _compile
from bsa_sm120 import NAME, BsaCompiler

#: The entry the module exports, which the package's `[module.entries]` names.
ENTRY = "uniserve_sparse_attention"


def spec(fields, parameters, target):
    """The compilation of one sparse_attention sm120 descriptor."""
    return {
        "name": NAME,
        "entry": ENTRY,
        "compile": BsaCompiler(fields, parameters, target).compile,
    }


if __name__ == "__main__":
    _compile.main(spec)
