"""The block-sparse attention modules against the operator's definition.

Each module test loads a built module of the `sparse_attention` operator
(resolved at the session's Build Server for the session's device) as a runner
loads it (``module.so``), calls its entry with the call structure of
``crates/ops/src/attention/sparse.rs`` (its mirror held to the generated
header, which needs no device) and compares the output, and the
normalizer where the module writes one, with the operator's definition
computed in torch: each query tile attends the key blocks its selection
names, each block cut to its valid key count.

The backends are packages of their own, each compiled for its block, which
is the kernel's tile: ``wgmma`` (128 rows in both directions, no normalizer),
``tcgen05`` (64) and ``flashinfer`` (FlashInfer's SM120 kernel, 64 rows in
both domains, which always writes the normalizer as the natural logarithm the
operator promises). Each package builds only for a target that meets its
requirements; on a device whose target does not, the Build Server refuses its
module and the test is skipped with the server's reason.
"""

import ctypes
import math

import pytest
import torch
from uniserve_testing.kernels.packages import (
    abi_enumerator,
    assert_abi_layout,
)

pytestmark = pytest.mark.unit

#: `ops::abi::code`'s number of BF16.
BF16 = abi_enumerator("UNISERVE_DTYPE_BF16")

#: The words of the Build Server's refusal of a package for a target that does
#: not meet the package's requirements.
UNMET_REQUIREMENT = "the package requires"

#: The backends: the block each is compiled for, and whether the check asks for
#: the normalizer.
BACKENDS = {
    "wgmma": (128, False),
    "tcgen05": (64, True),
    "flashinfer": (64, True),
}

#: The problem: key rows (a whole number of blocks), query tiles, heads, the
#: widest selection and the head width.
KEYS, TILES, HEADS, SELECTED, HEAD_DIM = 512, 4, 8, 3, 128

#: The check's allowance: 5% of the output's largest value (at least 1), and
#: 0.05 for the natural-log normalizer.
RELATIVE = 0.05
LSE_ABSOLUTE = 0.05


class Call(ctypes.Structure):
    """`ops::attention::sparse::SparseCall`, field for field."""

    _fields_ = [
        ("query", ctypes.c_void_p),
        ("key", ctypes.c_void_p),
        ("value", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("lse", ctypes.c_void_p),
        ("indices", ctypes.c_void_p),
        ("counts", ctypes.c_void_p),
        ("valid", ctypes.c_void_p),
        ("heads", ctypes.c_int64),
        ("query_rows", ctypes.c_int64),
        ("key_rows", ctypes.c_int64),
        ("q_row_stride", ctypes.c_int64),
        ("q_head_stride", ctypes.c_int64),
        ("o_row_stride", ctypes.c_int64),
        ("o_head_stride", ctypes.c_int64),
        ("k_row_stride", ctypes.c_int64),
        ("k_head_stride", ctypes.c_int64),
        ("v_row_stride", ctypes.c_int64),
        ("v_head_stride", ctypes.c_int64),
        ("selected", ctypes.c_int64),
        ("head_dim", ctypes.c_int64),
        ("dtype", ctypes.c_int64),
        ("output_dtype", ctypes.c_int64),
        ("scale", ctypes.c_float),
        ("error", ctypes.c_char * 512),
    ]


def test_the_call_mirror_is_the_headers_sparse_attention_call():
    """The generated header asserts the size and offsets `SparseCall` has."""
    assert_abi_layout(Call, "uniserve_sparse_attention_call")


def pointer(tensor):
    """A tensor's device address as the call structure carries it."""
    return ctypes.c_void_p(tensor.data_ptr())


def entry(kernel_module, backend, block, lse):
    """The C entry of `backend`'s module for `block`, or a skip when the
    session's target does not meet the package's requirements.
    """  # noqa: D205
    key = (
        f"block_{block}_dtype_bf16_output_bf16_head_dim_{HEAD_DIM}_"
        f"lse_{str(lse).lower()}"
    )
    try:
        module = kernel_module("sparse_attention", backend, key)
    except RuntimeError as refusal:
        if UNMET_REQUIREMENT in str(refusal):
            pytest.skip(
                f"the session's device does not run {backend}: {refusal}"
            )
        raise
    library = ctypes.CDLL(str(module / "module.so"))
    function = library.uniserve_sparse_attention
    function.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    function.restype = ctypes.c_int32
    return function


def reference(query, key, value, indices, counts, valid, scale, block):
    """The operator's definition in FP32: the output, and the normalizer as
    a natural logarithm.
    """  # noqa: D205
    query_rows, heads, head_dim = query.shape
    out = torch.zeros(
        query_rows, heads, head_dim, dtype=torch.float32, device=query.device
    )
    lse = torch.full(
        (heads, query_rows),
        float("-inf"),
        dtype=torch.float32,
        device=query.device,
    )
    for head in range(heads):
        rows = query[:, head].float()
        keys = key[:, head].float()
        values = value[:, head].float()
        for tile in range(query_rows // block):
            blocks = indices[head, tile, : int(counts[head, tile])].long()
            # The visible key rows of the selection: whole blocks, cut to each
            # block's valid count, which is what the kernel's masking reads.
            visible = (
                torch.cat(
                    [
                        block_id * block
                        + torch.arange(
                            int(valid[block_id]), device=query.device
                        )
                        for block_id in blocks
                    ]
                )
                if blocks.numel()
                else torch.empty(0, dtype=torch.long, device=query.device)
            )
            if not visible.numel():
                continue
            start, stop = tile * block, (tile + 1) * block
            logits = rows[start:stop] @ keys[visible].transpose(0, 1) * scale
            out[start:stop, head] = (
                torch.softmax(logits, dim=-1) @ values[visible]
            )
            top = logits.max(dim=-1).values
            lse[head, start:stop] = top + torch.log(
                torch.exp(logits - top[:, None]).sum(-1)
            )
    return out, lse


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_sparse_module_matches_the_operators_definition(
    backend, torch_device, kernel_module
):
    block, normalized = BACKENDS[backend]
    function = entry(kernel_module, backend, block, normalized)
    torch.manual_seed(0)
    query_rows = TILES * block
    key_blocks = KEYS // block

    def rows(count):
        return (
            torch.randn(
                count,
                HEADS,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device=torch_device,
            )
            * 0.1
        )

    query = rows(query_rows)
    key = rows(KEYS)
    value = rows(KEYS)
    out = torch.zeros_like(query)
    lse = torch.zeros(
        HEADS, query_rows, dtype=torch.float32, device=torch_device
    )
    # A selection that keeps every block id in range, one tile that reads a
    # single block, and a partial trailing block so the masking is exercised.
    indices = torch.randint(
        0, key_blocks, (HEADS, TILES, SELECTED), device=torch_device
    ).to(torch.int32)
    counts = torch.full(
        (HEADS, TILES), SELECTED, dtype=torch.int32, device=torch_device
    )
    counts[:, 0, 0] = 1
    valid = torch.full(
        (key_blocks,), block, dtype=torch.int32, device=torch_device
    )
    valid[-1] = block // 2

    call = Call()
    call.query = pointer(query)
    call.key = pointer(key)
    call.value = pointer(value)
    call.out = pointer(out)
    # A backend whose kernel writes no normalizer is asked for none.
    call.lse = pointer(lse) if normalized else None
    call.indices = pointer(indices)
    call.counts = pointer(counts)
    call.valid = pointer(valid)
    call.heads = HEADS
    call.query_rows = query_rows
    call.key_rows = KEYS
    call.q_row_stride = HEADS * HEAD_DIM
    call.q_head_stride = HEAD_DIM
    call.o_row_stride = HEADS * HEAD_DIM
    call.o_head_stride = HEAD_DIM
    call.k_row_stride = HEADS * HEAD_DIM
    call.k_head_stride = HEAD_DIM
    call.v_row_stride = HEADS * HEAD_DIM
    call.v_head_stride = HEAD_DIM
    call.selected = SELECTED
    call.head_dim = HEAD_DIM
    call.dtype = BF16
    call.output_dtype = BF16
    call.scale = 1.0 / math.sqrt(HEAD_DIM)

    stream = torch.cuda.current_stream(torch_device).cuda_stream
    status = function(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize(torch_device)
    assert status == 0, (
        f"the entry refused the call: {call.error.decode(errors='replace')}"
    )

    expected, expected_lse = reference(
        query, key, value, indices, counts, valid, call.scale, block
    )
    error = (out.float() - expected).abs().max().item()
    largest = expected.abs().max().item()
    allowed = RELATIVE * max(largest, 1.0)
    assert error <= allowed, (
        f"{backend}, block {block}: max |out - exact| = {error:.6g} exceeds "
        f"{allowed:.6g} (largest value {largest:.6g})"
    )
    if normalized:
        visible = expected_lse.isfinite()
        lse_error = (lse[visible] - expected_lse[visible]).abs().max().item()
        assert lse_error <= LSE_ABSOLUTE, (
            f"{backend}, block {block}: max |lse - exact| = {lse_error:.6g} "
            f"over {int(visible.sum())} rows exceeds {LSE_ABSOLUTE}"
        )
