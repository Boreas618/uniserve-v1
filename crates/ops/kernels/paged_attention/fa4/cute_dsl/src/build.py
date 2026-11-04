"""The paged_attention package's fa4 compile script: one descriptor -> one module.

A descriptor is the phase, the element types, the head width, the page size, the
head counts, the mask, the window and the longest key sequence; the extents of a
call — batch, tokens, page-table width — are runtime words, and the package's
parameters state the representative problem the vendor's configuration is chosen
for. Only whole-query prefill is served: the decode cells are the trtllm module's,
and FA4's decode path has not been checked against exact attention here.
"""

import fa4

import _compile


def spec(fields, parameters, target):
    """The compilation of one paged_attention descriptor."""
    if fields["phase"] != "prefill":
        raise SystemExit("FA4's paged forward pass is compiled for whole-query prefill only")
    if fields["kv"] != fields["dtype"] or fields["out"] != fields["dtype"]:
        raise SystemExit("FA4's paged pass serves one element type for query, cache and output")
    compiler = fa4.PrefillCompiler(fields, parameters, target)
    return {
        "name": fa4.NAME,
        "entry": "uniserve_paged_attention",
        "compile": compiler,
    }


if __name__ == "__main__":
    _compile.main(spec)
