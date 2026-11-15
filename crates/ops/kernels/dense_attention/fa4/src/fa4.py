"""FlashAttention-4's dense forward pass over packed sequences.

The kernel is FlashAttention-4's Blackwell forward pass,
``flash_attn.cute.flash_fwd_sm100.FlashAttentionForwardSm100`` of the locked
``flash-attn-4`` distribution, constructed here with an explicit
configuration and compiled through this package's launcher
(:class:`DenseForward`), whose parameters are the C module's. Every part of the
configuration is stated by one owner:

- the descriptor: the head width, the grouped-query ratio and the mask, and
  with the ratio whether a tile packs the query heads of one key head (FA4
  packs them whenever a key head serves several);
- the C module's call: packed sequences named by two offset tables, no window,
  no split of the keys, no page table, no score or mask modifier, and the
  schedule FA4 gives a packed launch whose sequences may span several tiles
  (no static persistent schedule, which FA4 chooses only when every sequence
  fits one tile, and no two-CTA MMA, which it never chooses for packed
  queries);
- the package's tuning data (``tuning/forward.toml``): the tile, the query
  stages a CTA keeps in flight and the tile scheduler.

The pass is the packed-sequence one: queries, keys and values are runs of rows
named by the two offset tables, there is no cache and no page table, and the
mask is the descriptor's (causal or dense).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100

#: The module's name: the exported C wrapper and the module handle carry it.
NAME = "flash_attn_dense"

#: The element types the operator's query, key, value and output may have.
DTYPES = {"bf16": cutlass.BFloat16, "f16": cutlass.Float16}


def forward_kernel(fields, configuration):
    """FA4's forward pass for one descriptor and its tuned configuration."""
    grouped = fields["q_heads"] // fields["kv_heads"]
    tile_m, tile_n = configuration["tile"]
    return FlashAttentionForwardSm100(
        fields["head_dim"],
        fields["head_dim"],
        qhead_per_kvhead=grouped,
        is_causal=fields["mask"] == "causal",
        is_local=False,
        is_split_kv=False,
        pack_gqa=grouped > 1,
        m_block_size=tile_m,
        n_block_size=tile_n,
        q_stage=configuration["query_stages"],
        is_static_persistent=False,
        is_varlen_q=True,
        use_2cta_instrs=False,
        use_clc_scheduler=configuration["cluster_launch_control"],
    )


class DenseForward:
    """The C module's launcher: FA4's forward pass over packed sequences.

    Its parameters are what the C module takes at launch, in the export's
    order, and carry FA4's names, which name the export's tensor structs
    (``flash_attn_dense_Tensor_mQ_t``) that ``entry.cpp`` fills. The vendor's
    forward pass is called inside this trace, where the arguments a C module
    cannot take (the absent tensors and the score and mask modifiers' data)
    are its defaults.
    """

    def __init__(self, kernel):
        self.kernel = kernel

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # noqa: N803
        mK: cute.Tensor,  # noqa: N803
        mV: cute.Tensor,  # noqa: N803
        mO: cute.Tensor,  # noqa: N803
        softmax_scale: cutlass.Float32,
        mCuSeqlensQ: cute.Tensor,  # noqa: N803
        mCuSeqlensK: cute.Tensor,  # noqa: N803
        max_seqlen_q: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Launch FA4's forward pass on ``stream``.

        ``mQ`` and ``mO`` are the packed ``[q_tokens, q_heads, head_dim]``
        queries and output, ``mK`` and ``mV`` the packed ``[kv_tokens,
        kv_heads, head_dim]`` keys and values, each with its head width
        contiguous; the offset tables hold ``batch + 1`` entries.
        """
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            None,
            softmax_scale,
            mCuSeqlensQ,
            mCuSeqlensK,
            max_seqlen_q=max_seqlen_q,
            stream=stream,
        )
