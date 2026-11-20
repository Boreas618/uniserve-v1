"""The mHC programs against the operators' own formulas.

``mhc_boundary`` writes a pending output back into the streams, computes the
next connection's mixes and reads the next input with the carried pre-mix;
``mhc_post`` and ``mhc_pre`` are the write-back and the read alone
(``crates/ops/src/mhc.rs``)::

    h'_k  = post_k y + sum_j comb[j, k] h_j
    z     = (flat(h) @ fn^T) rsqrt(mean(flat(h)^2) + rms_eps) s_g + b
    pre   = sigmoid(z[:n]) + eps;   post = 2 sigmoid(z[n:2n])
    comb  = sinkhorn(z[2n:].view(n, n))
    x     = rms_norm(sum_j carried_j h_j) w

**What is compared.** The reference is the formula in FP64 and the streams
are FP32, so no narrowing store stands between the two (Triton's interpreter
truncates a store into BF16 where a device rounds): every output is held to
the operator's bound relative to its largest magnitude
(``MhcBoundary::tolerance``, ``MhcPost::tolerance``, ``MhcPre::tolerance``):
the mixes' ``n d``-term product of mixed signs, its three-pass TF32 excess
counted, ``2 gamma(n d + 21 + ceil(3 n d / 1024)) sqrt(n d) + u``. A split
launch must also leave its arrival counters at zero, which the next launch
reads.

The cases are small widths of the model's four streams, each named for the
path it covers: the read alone, the write-back, two token blocks, and one
block's columns split across four programs whose last finishes it.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.unit

#: Four streams, as the model's; the mixes and their tile.
STREAMS = 4
MIXES = STREAMS * (STREAMS + 2)
MIX_TILE = 32

#: The cases: tokens, whether a pending output is written back, the column
#: splits of a block, the seed.
CASES = {
    "a read of three tokens": (3, False, 1, 1),
    "a write-back, then the read": (5, True, 1, 2),
    "two token blocks": (20, True, 1, 3),
    "one block split across four programs": (2, True, 4, 4),
}

HIDDEN = 64
BLOCK_T = 16
BLOCK_D = 16
U32 = 2.0**-24


def gamma(terms):
    """`n u / (1 - n u)` for an FP32 reduction of `terms` terms."""
    return terms * U32 / (1 - terms * U32)


def bound(width):
    """The mixes' bound, the largest of the boundary's three
    (`boundary_tolerance` in `crates/ops/src/mhc.rs`).
    """  # noqa: D205
    excess = 21 + math.ceil(3 * width / 1024)
    return 2 * gamma(width + excess) * math.sqrt(width) + U32


def sinkhorn(logits, iterations, eps):
    """Row softmax plus `eps`, then columns, then rows and columns again."""
    comb = torch.softmax(logits, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return comb


def reference(h, y, post, comb, fn, scale, base, carried, norm, iterations):
    """The boundary in FP64: written streams, input and mixes."""
    h, carried, norm = h.double(), carried.double(), norm.double()
    if y is not None:
        h = post.double().unsqueeze(-1) * y.double().unsqueeze(-2) + torch.sum(
            comb.double().unsqueeze(-1) * h.unsqueeze(-2), dim=1
        )
    flat = h.flatten(1)
    inverse = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-20)
    z = flat @ fn.double().t() * inverse
    s = scale.double()
    b = base.double()
    n = STREAMS
    pre = torch.sigmoid(z[:, :n] * s[0] + b[:n]) + 1e-6
    post_mix = 2 * torch.sigmoid(z[:, n : 2 * n] * s[1] + b[n : 2 * n])
    logits = (z[:, 2 * n :] * s[2] + b[2 * n :]).unflatten(-1, (n, n))
    comb_mix = sinkhorn(logits, iterations, 1e-6)
    collapsed = torch.sum(carried.unsqueeze(-1) * h, dim=1)
    inverse = torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + 1e-20)
    return h, collapsed * inverse * norm, pre, post_mix, comb_mix


def assert_within(what, got, expected, relative):
    """`got` within `relative` of `expected`'s largest magnitude."""
    got, expected = got.double().cpu(), expected.double().cpu()
    peak = expected.abs().max().item()
    worst = (got - expected).abs().max().item()
    assert worst <= relative * peak, (
        f"{what}: largest difference {worst:.3e} beyond {relative:.3e} of "
        f"the peak {peak:.3e}"
    )


@pytest.mark.parametrize("case", CASES)
def test_a_boundary_writes_mixes_and_reads(case, torch_device, kernel_source):
    tokens, post, splits, seed = CASES[case]
    generator = torch.Generator().manual_seed(seed)

    def draw(*dims, scale=1.0, shift=0.0):
        values = torch.rand(*dims, generator=generator) * 2 - 1
        return (values * scale + shift).to(torch_device)

    h = draw(tokens, STREAMS, HIDDEN, scale=2.0)
    y = draw(tokens, HIDDEN)
    post_mix = draw(tokens, STREAMS, scale=1.0, shift=1.0)
    comb = draw(tokens, STREAMS, STREAMS, scale=0.1, shift=0.25)
    fn = draw(MIXES, STREAMS * HIDDEN, scale=4.0)
    scale = draw(3, scale=0.05, shift=0.1)
    base = draw(MIXES, scale=3.0)
    carried = draw(tokens, STREAMS, scale=0.5, shift=0.5)
    norm = draw(HIDDEN, scale=0.5, shift=1.0)

    blocks = -(-tokens // BLOCK_T)
    empty = lambda *dims: torch.full(dims, float("nan"), device=torch_device)  # noqa: E731
    written = empty(tokens, STREAMS, HIDDEN)
    inputs = empty(tokens, HIDDEN)
    pre_out, post_out = empty(tokens, STREAMS), empty(tokens, STREAMS)
    comb_out = empty(tokens, STREAMS, STREAMS)
    partials = torch.zeros(splits, tokens, MIX_TILE + 2, device=torch_device)
    counters = torch.zeros(blocks, dtype=torch.int32, device=torch_device)

    kernels = kernel_source("mhc_boundary")
    kernels._mhc_boundary_kernel[(blocks, splits)](
        h,
        y,
        post_mix,
        comb,
        written,
        fn,
        scale,
        base,
        carried,
        norm,
        inputs,
        pre_out,
        post_out,
        comb_out,
        partials,
        counters,
        tokens,
        splits,
        1e-20,
        1e-6,
        1e-20,
        STREAMS=STREAMS,
        HIDDEN=HIDDEN,
        ITERATIONS=20,
        POST=post,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        MIXES=MIXES,
        MIX_TILE=MIX_TILE,
        NORM=64,
        CHUNK=4,
    )

    expected = reference(
        h,
        y if post else None,
        post_mix,
        comb,
        fn,
        scale,
        base,
        carried,
        norm,
        20,
    )
    relative = bound(STREAMS * HIDDEN)
    if post:
        assert_within(f"{case}: written", written, expected[0], relative)
    for what, got, wanted in zip(
        ("input", "pre-mix", "post-mix", "combination"),
        (inputs, pre_out, post_out, comb_out),
        expected[1:],
        strict=True,
    ):
        assert_within(f"{case}: {what}", got, wanted, relative)
    assert int(counters.abs().sum()) == 0, (
        f"{case}: the arrival counters are {counters.tolist()}"
    )


def test_a_write_and_a_read_alone(torch_device, kernel_source):
    generator = torch.Generator().manual_seed(9)
    tokens = 3

    def draw(*dims, shift=0.0):
        values = torch.rand(*dims, generator=generator) * 2 - 1 + shift
        return values.to(torch_device)

    h, y = draw(tokens, STREAMS, HIDDEN), draw(tokens, HIDDEN)
    post, comb = (
        draw(tokens, STREAMS, shift=1.0),
        draw(tokens, STREAMS, STREAMS),
    )
    pre, norm = draw(tokens, STREAMS, shift=1.0), draw(HIDDEN, shift=1.0)

    written = torch.empty_like(h)
    kernel_source("mhc_post")._mhc_post_kernel[(tokens, 2)](
        h, y, post, comb, written, STREAMS=STREAMS, HIDDEN=HIDDEN, BLOCK=32
    )
    read = torch.empty(tokens, HIDDEN, device=torch_device)
    kernel_source("mhc_pre")._mhc_pre_kernel[(tokens,)](
        h, pre, norm, read, 1e-6, STREAMS=STREAMS, HIDDEN=HIDDEN, TILE=64
    )

    hd = h.double()
    expected = post.double().unsqueeze(-1) * y.double().unsqueeze(
        -2
    ) + torch.sum(comb.double().unsqueeze(-1) * hd.unsqueeze(-2), dim=1)
    # `n + 1` FP32 terms of mixed signs (`MhcPost::tolerance`).
    assert_within("written", written, expected, 2 * gamma(STREAMS + 1) + U32)
    collapsed = torch.sum(pre.double().unsqueeze(-1) * hd, dim=1)
    inverse = torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + 1e-6)
    # The `d`-term mean square and the roundings (`MhcPre::tolerance`).
    assert_within(
        "read",
        read,
        collapsed * inverse * norm.double(),
        2 * gamma(HIDDEN) + 3 * U32,
    )
