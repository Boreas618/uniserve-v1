"""FlashAttention 4's paged forward pass: what this package compiles and exports.

FA4 is not a kernel source this package owns: it is the FlashAttention distribution's
Blackwell CuTe DSL kernel plus the glue that decides *how* to compile it. That glue
(``flash_attn.cute.interface._flash_attn_fwd``) is written for a torch caller and ends
in a TVM-FFI function, so this module drives it once at build time and keeps what it
produced:

1. build fake tensors of the descriptor's geometry under ``FakeTensorMode`` (no device
   is opened and no kernel runs);
2. call ``_flash_attn_fwd`` with the vendor's own selection, intercepting its
   ``cute.compile`` so the chosen kernel and its arguments are captured instead of
   being compiled into a TVM-FFI module;
3. compile that captured kernel the CuTe DSL's plain way — pointers and scalars, no
   TVM-FFI — and export it to C.

Two things the vendor's interface assumes must be true of a C module instead, and are
set here on its own objects:

* ``aux_data`` describes score and mask modifiers. With none, it is a constant of the
  compilation and not an argument a C caller could pass;
* ``max_seqlen_q`` is an extent the call carries, so it is one runtime word.

Everything else — tiles, pipes, whether the schedule is persistent, whether the
cluster is two CTAs — the vendor's selection decides, from the descriptor's geometry
and the representative problem the package's parameters state. That representative
problem is the one the compiled configuration is chosen for: a call with a very
different batch or query length would run a configuration chosen for another shape,
so the package's parameters are the caller's declaration of the shapes it serves.
"""

import os

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import cutlass
import cutlass.cute as cute

#: The element types the operator's cache and query may have, as torch spells them.
TORCH_DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16}

#: The module's name: the exported C wrapper and the module handle carry it.
NAME = "flash_attn_paged"


def _interface():
    """The vendor's forward interface and its Blackwell kernel, annotated for a C module."""
    import flash_attn.cute.flash_fwd_sm100 as kernels
    import flash_attn.cute.interface as interface

    forward = kernels.FlashAttentionForwardSm100.__call__
    forward.__annotations__["aux_data"] = cutlass.Constexpr
    forward.__annotations__["max_seqlen_q"] = cutlass.Int32
    return interface


class PrefillCompiler:
    """One compilation of FA4's paged forward pass for whole queries per request."""

    def __init__(self, fields, parameters, target):
        self.dtype = TORCH_DTYPES[fields["dtype"]]
        self.cache_dtype = TORCH_DTYPES[fields["kv"]]
        self.head_dim = int(fields["head_dim"])
        self.page = int(fields["page"])
        self.q_heads = int(fields["q_heads"])
        self.kv_heads = int(fields["kv_heads"])
        self.causal = fields["mask"] == "causal"
        if fields["window"] != "false":
            raise SystemExit("FA4's paged pass is compiled without a window")
        # The longest key sequence the cache holds; the page table's width is
        # that length in pages, and the vendor sizes its scheduler from it.
        self.max_kv_len = int(fields["len"])
        self.pages_per_seq = (self.max_kv_len + self.page - 1) // self.page
        # The representative problem the vendor's selection is asked about.
        self.batch = int(parameters.get("batch", 8))
        self.query_len = int(parameters.get("query_len", 512))
        self.target = target

    def tensors(self):
        """Fake tensors of the descriptor's geometry, as ``_flash_attn_fwd`` reads them."""
        batch, query_len, page = self.batch, self.query_len, self.page
        tokens = batch * query_len
        pages = batch * self.pages_per_seq
        device = "cuda"
        query = torch.empty(tokens, self.q_heads, self.head_dim, dtype=self.dtype, device=device)
        cache = torch.empty(
            pages, page, self.kv_heads, self.head_dim, dtype=self.cache_dtype, device=device
        )
        return {
            "q": query,
            "k": cache,
            "v": torch.empty_like(cache),
            "page_table": torch.empty(batch, self.pages_per_seq, dtype=torch.int32, device=device),
            "cu_seqlens_q": torch.empty(batch + 1, dtype=torch.int32, device=device),
            "seqused_k": torch.empty(batch, dtype=torch.int32, device=device),
        }

    def __call__(self):
        """Compile the configuration the vendor selects, exported to C."""
        interface = _interface()
        captured = []
        real_compile = cute.compile

        class Capturing:
            """The cute module, with only ``compile`` replaced."""

            @staticmethod
            def compile(function, *args, **options):
                options.pop("options", None)
                captured.append((function, args))
                return None

            def __getattr__(self, name):
                return getattr(cute, name)

        original = interface.cute
        interface.cute = Capturing()
        try:
            with FakeTensorMode():
                tensors = self.tensors()
                interface._flash_attn_fwd(
                    tensors["q"],
                    tensors["k"],
                    tensors["v"],
                    cu_seqlens_q=tensors["cu_seqlens_q"],
                    page_table=tensors["page_table"],
                    seqused_k=tensors["seqused_k"],
                    max_seqlen_q=self.query_len,
                    max_seqlen_k=self.pages_per_seq * self.page,
                    softmax_scale=self.head_dim**-0.5,
                    causal=self.causal,
                    # No window: FA4 reads a window size as a distance, so an
                    # absent window is the absence of both bounds.
                    window_size_left=None,
                    window_size_right=None,
                    _arch=cute_arch(self.target),
                )
        finally:
            interface.cute = original
        if len(captured) != 1:
            raise SystemExit(f"FA4 compiled {len(captured)} kernels for this geometry; expected one")
        function, arguments = captured[0]
        # The stream is the caller's at run time; the compile describes a fake one.
        arguments = [one for one in arguments if "Stream" not in type(one).__name__]
        # Scalars are runtime words of the module, not Python constants.
        arguments = [
            cutlass.Float32(one)
            if isinstance(one, float)
            else cutlass.Int32(one)
            if isinstance(one, int) and not isinstance(one, bool)
            else one
            for one in arguments
        ]
        return real_compile(function, *arguments, stream=cute.runtime.make_fake_stream())


def cute_arch(target):
    """The compute capability a target names, ``"sm_100a"`` -> ``100``."""
    digits = target.removeprefix("sm_")
    while digits and digits[-1].isalpha():
        digits = digits[:-1]
    if not digits.isdigit():
        raise SystemExit(f"target {target!r} is not an sm_<capability> architecture")
    return int(digits)
