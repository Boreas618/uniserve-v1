"""The paged_attention package's trtllm build module.

One descriptor -> one module.

The module is trtllm-gen's FMHA launch code — the artifact's kernel table plus
`TllmGenFmhaRunner` — compiled against the generated header, with the cubins
the descriptor's configurations can select published beside the module in
`cubins/`. The Build Server compiles a descriptor by calling :func:`compile`
with the compile request.

Steps:
1. stage the pinned `trtllm-gen-fmha` artifact's generated
   `flashInferMetaInfo.h` (`uniserve_build_server.trtllm.stage`, which
   verifies every file against the pin);
2. read the artifact's kernel table and keep the entries the descriptor can
   select — the element types, `head_dim`, `page`, the mask and the phase, on
   the architecture families the target runs;
3. fetch those cubins and publish them beside the module, under `cubins/`;
4. compile the entry unit and the FMHA launch code against the staged header,
   and link `module.so` (`uniserve_build_server.trtllm.build`).

A descriptor is the operator's: phase, dtype, kv, out, head_dim, page, q_heads,
kv_heads, mask, window, len.
"""

import hashlib
import re
import shutil
import sys

from uniserve_build_server import cuda, trtllm
from uniserve_build_server.request import Request

#: The operator's element types, as the artifact names them in a kernel.
NAMES = {"bf16": "Bfloat16", "f16": "Fp16", "f8e4m3": "E4m3", "f4e2m1": "E2m1"}

#: The artifact's generated kernel table.
METADATA_HEADER = "include/flashInferMetaInfo.h"


def matching_kernels(header, fields, target):
    """The kernel names of the artifact the descriptor's configurations select.

    The table is the artifact's own: an entry names its element types, head
    width, page size, mask, phase and architecture family, so the filter is the
    descriptor's values, not a guess. A target none of whose families the
    artifact was built for matches no kernel, which refuses the descriptor.
    """
    families = trtllm.families(target)
    query = NAMES[fields["dtype"]]
    key_value = NAMES[fields["kv"]]
    out_type = NAMES[fields["out"]]
    prefix = (
        f"Q{query}Kv{key_value}O{out_type}"
        if query != key_value
        else f"Qkv{query}O{out_type}"
    )
    head = f"H{fields['head_dim']}"
    page = f"P{fields['page']}"
    decode = fields["phase"] == "decode"
    # The artifact's generation kernels are named causally whatever mask a call
    # asks for: with one query token per request, a dense and a causal mask
    # cover the same keys, which is why the vendor's own launcher hands its
    # generation kernels a causal mask either way. The prefill's mask does pick
    # the name, since its tiles see whole queries.
    mask = "Causal" if decode or fields["mask"] == "causal" else "Dense"
    kept = []
    for line in header.splitlines():
        if not line.startswith("{ DATA_TYPE_"):
            continue
        name = re.search(r'"([^"]+)"', line)
        if name is None:
            continue
        name = name.group(1)
        if prefix not in name or head not in name:
            continue
        if f"PagedKv{mask}{page}" not in name:
            continue
        if not any(
            name.startswith(f"fmha{family}Kernel") for family in families
        ):
            continue
        # The artifact's prefill kernels end in `Context`; the decode ones carry
        # the generation phase in their name.
        if ("Context" in name) == decode:
            continue
        # A skipped softmax is a variant this module never asks for, and the
        # sliding-window ones only serve a window the descriptor does not have.
        if "SkipsSoftmax" in name:
            continue
        # A window is its own trade: the sliding kernels read a window size the
        # operator's call states, and this package does not serve one.
        if "SlidingOrChunked" in name:
            continue
        kept.append(name)
    if not kept:
        raise SystemExit(
            f"the artifact has no {prefix}{head} {mask}{page} kernel for "
            f"{target}"
        )
    return sorted(set(kept))


def compile(request: Request) -> None:
    """Compile the request's descriptor into ``module.so`` with its cubins.

    The artifact is the build's input, not the module: it is staged into the
    build's scratch (`uniserve_build_server.cuda.scratch`), compiled
    against, and never enters the directory a runner is handed.
    """
    with cuda.scratch(request) as artifact:
        trtllm.stage(request, trtllm.FMHA.library, artifact, [METADATA_HEADER])
        header_path = artifact / METADATA_HEADER
        names = matching_kernels(
            header_path.read_text(), request.fields, request.target
        )

        cubins = [f"{name}.cubin" for name in names]
        trtllm.stage(request, trtllm.FMHA.library, artifact, cubins)
        published = request.output / "cubins"
        published.mkdir(parents=True, exist_ok=True)
        for cubin in cubins:
            shutil.copyfile(artifact / cubin, published / cubin)
        print(f"published {len(names)} cubins", file=sys.stderr)

        # The launch code's header refuses to compile without the table's hash
        # and the directory its loader prefixes to a cubin's name, both of
        # which FlashInfer's own build defines.
        metainfo_hash = hashlib.sha256(header_path.read_bytes()).hexdigest()
        trtllm.build(
            request,
            trtllm.FMHA,
            includes=[artifact / "include"],
            definitions={
                "TLLM_GEN_FMHA_CUBIN_PATH": '"."',
                "TLLM_GEN_FMHA_METAINFO_HASH": f'"{metainfo_hash}"',
            },
        )
