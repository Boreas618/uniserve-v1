"""The hashed n-gram memory's programs against the operators' own formulas.

``ngram_hash`` names the table rows of every n-gram that ends at a token,
``embedding_mxfp8`` dequantizes a shard's MXFP8 rows and ``ngram_gate`` adds
the recalled value into each stream under the gate of its key's agreement
with it (``crates/ops/src/ngram.rs``, ``crates/ops/src/embedding.rs``)::

    rolling = t_0 mu_0;  rolling ^= t_i mu_i             i tokens back
    row     = rolling mod P + O                          per head of length i+1
    value   = e4m3 * 2^(e - 127)                         zeros outside the shard
    dot     = sum(h q kappa key) rsqrt(mean(h^2) + eps)
              rsqrt(mean(key^2) + eps) / sqrt(d)
    h'      = h + sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)) value

**What is compared.** The hash and the lookup are exact, integers and one
rounding of an exact product, so they are held bit for bit
(``Tolerance::exact``). The gate is held to its operator's bound relative to
the output's largest magnitude (``NgramGate::tolerance``): the agreement's
``d``-term FP32 product, through the signed square root (Hoelder, one half)
and the sigmoid (one quarter), ``sqrt(2 gamma(d) sqrt(d) + u) / 4 + 2 u``.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.unit

U32 = 2.0**-24


def test_the_hash_names_the_rows_of_every_head(torch_device, kernel_source):
    generator = torch.Generator().manual_seed(1)
    tokens, order, heads, vocabulary = 70, 4, 8, 500
    normalize = torch.randint(
        0, 300, (vocabulary,), generator=generator, dtype=torch.int32
    )
    # The last ids are past the table, which reads its last entry.
    context = torch.randint(
        0, vocabulary + 20, (tokens, order), generator=generator
    ).to(torch.int32)
    multipliers = (
        torch.randint(0, 2**40, (order,), generator=generator) * 2 + 1
    ).to(torch.int64)
    moduli = torch.tensor(
        [16_000_057 + 22 * index for index in range((order - 1) * heads)]
    ).view(order - 1, heads)
    offsets = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=torch.int64), moduli.flatten()[:-1]]),
        0,
    ).view(order - 1, heads)
    rows = torch.empty(
        tokens, (order - 1) * heads, dtype=torch.int64, device=torch_device
    )

    kernel_source("ngram_hash")._ngram_hash_kernel[(-(-tokens // 64),)](
        context.to(torch_device),
        normalize.to(torch_device),
        multipliers.to(torch_device),
        moduli.to(torch_device),
        offsets.to(torch_device),
        rows,
        tokens,
        vocabulary,
        ORDER=order,
        HEADS=heads,
        HEAD_TILE=8,
        BLOCK=64,
    )

    ids = normalize[context.long().clamp(max=vocabulary - 1)].long().flip(-1)
    products = ids * multipliers
    rolling, expected = products[:, 0], []
    for back in range(1, order):
        rolling = rolling ^ products[:, back]
        expected.append(rolling.unsqueeze(-1) % moduli[back - 1])
    expected = torch.cat(expected, -1) + offsets.flatten()
    assert torch.equal(rows.cpu(), expected), "the rows differ"


def test_a_shard_dequantizes_its_rows_and_zeros_the_others(
    torch_device, kernel_source
):
    generator = torch.Generator().manual_seed(2)
    rows, width, first = 300, 256, 40
    codes = torch.randint(0, 256, (rows, width), generator=generator).to(
        torch.uint8
    )
    # 0x7F and 0xFF are not numbers in E4M3.
    codes[(codes & 0x7F) == 0x7F] -= 1
    values = codes.view(torch.float8_e4m3fn)
    scales = torch.randint(
        100, 150, (rows, width // 32), generator=generator
    ).to(torch.uint8)
    ids = torch.randint(0, rows + 2 * first, (50,), generator=generator)
    out = torch.full((50, width), float("nan"), device=torch_device)

    kernel_source("embedding_mxfp8")._embedding_mxfp8_kernel[(-(-50 // 16),)](
        ids.to(torch_device),
        values.to(torch_device),
        scales.to(torch_device),
        out,
        50,
        first,
        rows,
        HIDDEN=width,
        GROUP=32,
        TILE=width,
        BLOCK=16,
    )

    local = ids - first
    held = (local >= 0) & (local < rows)
    index = local.clamp(0, rows - 1)
    factors = torch.pow(2.0, scales[index].double() - 127)
    expected = (
        values[index].double().unflatten(-1, (-1, 32)) * factors.unsqueeze(-1)
    ).flatten(-2)
    expected[~held] = 0.0
    assert torch.equal(out.cpu().double(), expected), "the rows differ"


def test_the_gate_adds_the_value_by_each_streams_agreement(
    torch_device, kernel_source
):
    generator = torch.Generator().manual_seed(3)
    tokens, streams, hidden = 3, 4, 128

    def draw(*dims, scale=1.0, shift=0.0):
        values = torch.rand(*dims, generator=generator) * 2 - 1
        return values * scale + shift

    h = draw(tokens, streams, hidden)
    projected = draw(tokens, (streams + 1) * hidden)
    query = draw(streams, hidden, scale=0.3, shift=1.0)
    key = draw(streams, hidden, scale=0.3, shift=1.0)
    # One stream's key is its stream's negative, so its gate closes below a
    # half.
    projected[0, :hidden] = -h[0, 0]
    out = torch.empty(tokens, streams, hidden, device=torch_device)

    kernel_source("ngram_gate")._ngram_gate_kernel[(tokens, streams)](
        h.to(torch_device),
        projected.to(torch_device),
        query.to(torch_device),
        key.to(torch_device),
        out,
        projected.stride(0),
        1e-20,
        hidden**-0.5,
        STREAMS=streams,
        HIDDEN=hidden,
        TILE=hidden,
    )

    hd = h.double()
    keys = projected.double()[:, : streams * hidden].unflatten(
        -1, (streams, hidden)
    )
    value = projected.double()[:, streams * hidden :]
    inverse = torch.rsqrt(hd.square().mean(-1) + 1e-20) * torch.rsqrt(
        keys.square().mean(-1) + 1e-20
    )
    dot = (hd * query.double() * key.double() * keys).sum(-1) * inverse
    dot = dot * hidden**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    expected = hd + gate.unsqueeze(-1) * value.unsqueeze(-2)

    gamma = hidden * U32 / (1 - hidden * U32)
    relative = math.sqrt(2 * gamma * math.sqrt(hidden) + U32) / 4 + 2 * U32
    peak = expected.abs().max().item()
    worst = (out.cpu().double() - expected).abs().max().item()
    assert worst <= relative * peak, (
        f"largest difference {worst:.3e} beyond {relative:.3e} of the peak "
        f"{peak:.3e}"
    )
