"""FlashAttention 4's dense forward pass: what this package compiles and exports.

FA4 is not a kernel source this package owns: it is the FlashAttention distribution's
Blackwell CuTe DSL kernel plus the glue that decides *how* to compile it. That glue
(``flash_attn.cute.interface._flash_attn_fwd``) is written for a torch caller and ends
in a TVM-FFI function, so this module drives it once at build time and keeps what it
produced: it makes one fake call of the vendor's interface, captures the kernel that
call selects, compiles it the plain CuTe DSL way — pointers and scalars, no TVM-FFI —
and hands it back for ``export_to_c``.

The capturing and compiling half is ``compile_selection``, the same mechanism
`paged_attention`'s fa4 package carries; this package holds its own copy because a
package's sources are its own, and states only the one ``request`` it makes.

The pass is the packed-sequence one: queries, keys and values are runs of rows named by
the two offset tables, there is no cache and no page table, and the mask is the
descriptor's (causal or dense).

Two things the vendor's interface assumes must be true of a C module instead, and
``compile_selection`` sets them on the vendor's own objects:

* ``aux_data`` describes score and mask modifiers. With none, it is a constant of the
  compilation and not an argument a C caller could pass;
* ``max_seqlen_q`` is an extent the call carries, so it is one runtime word.

Everything else — tiles, pipes, whether the schedule is persistent, whether the
cluster is two CTAs — the vendor's selection decides, from the descriptor's geometry
and the representative problem the package's parameters state.
"""

import os

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import cutlass
import cutlass.cute as cute

#: The element types the operator's query, key, value and output may have.
TORCH_DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16}

#: The module's name: the exported C wrapper and the module handle carry it.
NAME = "flash_attn_dense"


def compile_selection(request, target):
    """Compile what FlashAttention's interface selects for one call, to C.

    ``request`` is called once, inside ``FakeTensorMode``, with the vendor's
    interface module and must make exactly one ``_flash_attn_fwd`` call. The
    kernel that call selects is compiled the plain CuTe DSL way — pointers and
    scalars, no TVM-FFI — and handed back for ``export_to_c``.

    This is the whole of what a caller needs from the vendor's glue: the
    selection (tiles, pipes, schedule, cluster) is the vendor's, and the two
    arguments a C module cannot take are set on the vendor's own objects first.
    """
    import flash_attn.cute.flash_fwd_sm100 as kernels
    import flash_attn.cute.interface as interface

    forward = kernels.FlashAttentionForwardSm100.__call__
    forward.__annotations__["aux_data"] = cutlass.Constexpr
    forward.__annotations__["max_seqlen_q"] = cutlass.Int32

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
            request(interface)
    finally:
        interface.cute = original
    if len(captured) != 1:
        raise SystemExit(f"the selection compiled {len(captured)} kernels; expected one")
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


class DenseCompiler:
    """One compilation of FA4's dense forward pass over packed sequences."""

    def __init__(self, fields, parameters, target):
        self.dtype = TORCH_DTYPES[fields["dtype"]]
        self.key_dtype = TORCH_DTYPES[fields["kv"]]
        self.output_dtype = TORCH_DTYPES[fields["out"]]
        self.head_dim = int(fields["head_dim"])
        self.q_heads = int(fields["q_heads"])
        self.kv_heads = int(fields["kv_heads"])
        self.causal = fields["mask"] == "causal"
        if fields["window"] != "false":
            raise SystemExit("FA4's dense pass is compiled without a window")
        # The longest key sequence the module serves, which the package's parameters declare:
        # the descriptor's bound is deliberately absent from the key, so it cannot be a field.
        self.max_kv_len = int(parameters.get("max_kv_len", 40960))
        # The representative problem the vendor's selection is asked about.
        self.batch = int(parameters.get("batch", 8))
        self.query_len = int(parameters.get("query_len", 512))
        self.target = target

    def request(self, interface):
        """Make the one ``_flash_attn_fwd`` call this descriptor compiles.

        The tensors are fake — this runs inside ``FakeTensorMode``, where a tensor is a
        shape and a stride and no device is opened. Every sequence holds ``query_len``
        queries and keys, which the offset tables name.
        """
        batch, query_len = self.batch, self.query_len
        device = "cuda"
        tokens = batch * query_len
        offsets = torch.arange(batch + 1, dtype=torch.int32, device=device) * query_len
        interface._flash_attn_fwd(
            torch.empty(tokens, self.q_heads, self.head_dim, dtype=self.dtype, device=device),
            torch.empty(tokens, self.kv_heads, self.head_dim, dtype=self.key_dtype, device=device),
            torch.empty(tokens, self.kv_heads, self.head_dim, dtype=self.key_dtype, device=device),
            cu_seqlens_q=offsets,
            cu_seqlens_k=offsets.clone(),
            max_seqlen_q=query_len,
            max_seqlen_k=query_len,
            softmax_scale=self.head_dim**-0.5,
            causal=self.causal,
            # No window: FA4 reads a window size as a distance, so an absent window is
            # the absence of both bounds.
            window_size_left=None,
            window_size_right=None,
            _arch=cute_arch(self.target),
        )

    def __call__(self):
        """Compile the configuration the vendor selects, exported to C."""
        return compile_selection(self.request, self.target)


def cute_arch(target):
    """The compute capability a target names, ``"sm_100a"`` -> ``100``."""
    digits = target.removeprefix("sm_")
    while digits and digits[-1].isalpha():
        digits = digits[:-1]
    if not digits.isdigit():
        raise SystemExit(f"target {target!r} is not an sm_<capability> architecture")
    return int(digits)
