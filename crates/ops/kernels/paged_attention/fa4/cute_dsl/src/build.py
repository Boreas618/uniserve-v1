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
    # The module states the geometry it was compiled for, so a call of another
    # geometry is refused instead of run against the wrong tiles.
    codes = {"bf16": "UNISERVE_DTYPE_BF16", "f16": "UNISERVE_DTYPE_F16"}
    defines = [
        f"UNISERVE_FA4_DTYPE={codes[fields['dtype']]}",
        f"UNISERVE_FA4_HEAD_DIM={int(fields['head_dim'])}",
        f"UNISERVE_FA4_PAGE={int(fields['page'])}",
        f"UNISERVE_FA4_Q_HEADS={int(fields['q_heads'])}",
        f"UNISERVE_FA4_KV_HEADS={int(fields['kv_heads'])}",
        f"UNISERVE_FA4_CAUSAL={1 if fields['mask'] == 'causal' else 0}",
        f"UNISERVE_FA4_MAX_KV_LEN={int(fields['len'])}",
    ]
    return {
        "name": fa4.NAME,
        "entry": "uniserve_paged_attention",
        "compile": compiler,
        "defines": defines,
    }


if __name__ == "__main__":
    _compile.main(spec)
