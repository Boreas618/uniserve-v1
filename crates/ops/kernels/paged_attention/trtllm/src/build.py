"""The paged_attention package's trtllm build module.

One descriptor -> one module.

The module is trtllm-gen's FMHA launch code — the library's kernel table plus
`TllmGenFmhaRunner` — compiled against the generated header, with the cubins
the descriptor's calls can select published beside the module in `cubins/`.
The Build Server compiles a descriptor by calling :func:`compile` with the
compile request.

Steps:
1. stage the pinned `trtllm-gen-fmha` library's generated
   `flashInferMetaInfo.h` (`uniserve_build_server.trtllm.stage`, which
   verifies every file against the pin);
2. read the library's kernel table and select the kernels the descriptor's
   calls can run (:func:`options`, `uniserve_build_server.trtllm.select`);
3. publish those cubins beside the module, under `cubins/`;
4. compile the entry unit and the FMHA launch code against the staged header,
   and link `module.so` (`uniserve_build_server.trtllm.build`).

A descriptor is the operator's: phase, dtype, kv, out, head_dim, page, q_heads,
kv_heads, mask, window, len.
"""

import hashlib
from typing import Any

from uniserve_build_server import cuda, trtllm
from uniserve_build_server.request import Request

#: The library's generated kernel table.
TABLE = "include/flashInferMetaInfo.h"

#: The headers of FlashInfer's data tree that declare the enumerations the
#: table's members take: the element types, and the FMHA runner's options.
ENUMERATIONS = (
    "include/flashinfer/trtllm/common.h",
    "include/flashinfer/trtllm/fmha/fmhaRunnerParams.h",
)

#: The operator's element types, as FlashInfer's `Data_type` names them.
DATA_TYPES = {
    "bf16": "Data_type::DATA_TYPE_BF16",
    "f16": "Data_type::DATA_TYPE_FP16",
}


def options(fields: dict[str, Any]) -> dict[str, Any]:
    """The table's options of every kernel a call of this descriptor can run.

    They are the options the entry fixes in the runner's parameters
    (`src/entry.cu`): the element types, the head width, a paged cache of the
    descriptor's page, and none of the variants it never asks for (skipped
    softmax, sparse attention, an FP16 softmax, compressed pages, the
    BF16-query FP8-cache transforms). The runner chooses among the rest --
    tiles, schedulers, the multi-CTA split -- for each call's shape, so every
    kernel that differs only in those is published.

    The decode is the generation kernels, which take a causal mask whatever
    the call's: with one query token per request, a dense and a causal mask
    cover the same keys, and the entry hands them a causal one. The
    prefill's mask is the call's.
    """
    decode = fields["phase"] == "decode"
    causal = decode or fields["mask"] == "causal"
    return {
        "mDataTypeQ": DATA_TYPES[fields["dtype"]],
        "mDataTypeKv": DATA_TYPES[fields["kv"]],
        "mDataTypeO": DATA_TYPES[fields["out"]],
        "mHeadDimQk": fields["head_dim"],
        "mHeadDimV": fields["head_dim"],
        "mQkvLayout": "QkvLayout::PagedKv",
        "mNumTokensPerPage": fields["page"],
        "mMaskType": (
            "TrtllmGenAttentionMaskType::Causal"
            if causal
            else "TrtllmGenAttentionMaskType::Dense"
        ),
        "mKernelType": (
            [
                "FmhaKernelType::SwapsMmaAbForGeneration",
                "FmhaKernelType::KeepsMmaAbForGeneration",
            ]
            if decode
            else "FmhaKernelType::Context"
        ),
        "mSkipsSoftmaxWhenPossible": False,
        "mSparseAttn": "TrtllmGenSparseMlaType::None",
        "mFp16Softmax": False,
        "mUsesSpcompress": False,
        "mEnablesBf16QFp8KvKOnlyTransform": False,
        "mSeparateTransformedKv": False,
    }


def compile(request: Request) -> None:
    """Compile the request's descriptor into ``module.so`` with its cubins.

    The library is the build's input, not the module: it is staged into the
    build's scratch (`uniserve_build_server.cuda.scratch`), compiled
    against, and never enters the directory a runner is handed.
    """
    library = trtllm.FMHA.library
    data = trtllm.data_tree()
    with cuda.scratch(request) as staging:
        trtllm.stage(request, library, staging, [TABLE])
        enums = trtllm.enumerations(
            (data / one).read_text() for one in ENUMERATIONS
        )
        kernels = trtllm.select(
            trtllm.fmha_table((staging / TABLE).read_text(), enums),
            {
                name: trtllm.resolve(value, enums)
                for name, value in options(request.fields).items()
            },
            request.target,
        )
        trtllm.publish(request, library, kernels, staging)

        # The launch code's header refuses to compile without the table's hash
        # and the directory its loader prefixes to a cubin's name, both of
        # which FlashInfer's own build defines.
        table_hash = hashlib.sha256((staging / TABLE).read_bytes()).hexdigest()
        trtllm.build(
            request,
            trtllm.FMHA,
            includes=[staging / "include"],
            definitions={
                "TLLM_GEN_FMHA_CUBIN_PATH": '"."',
                "TLLM_GEN_FMHA_METAINFO_HASH": f'"{table_hash}"',
            },
        )
