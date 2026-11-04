"""The dense_attention package's fa4 compile script: one descriptor -> one module.

A descriptor is the element types, the head width, the head counts, the mask, the
window and the longest key sequence; the extents of a call — batch, tokens, offsets —
are runtime words, and the package's parameters state the representative problem the
vendor's configuration is chosen for.
"""

import fa4

import _compile


def spec(fields, parameters, target):
    """The compilation of one dense_attention descriptor."""
    if fields["kv"] != fields["dtype"] or fields["out"] != fields["dtype"]:
        raise SystemExit("FA4's dense pass serves one element type for query, cache and output")
    compiler = fa4.DenseCompiler(fields, parameters, target)
    return {
        "name": fa4.NAME,
        "entry": "uniserve_dense_attention",
        "compile": compiler,
    }


if __name__ == "__main__":
    _compile.main(spec)
