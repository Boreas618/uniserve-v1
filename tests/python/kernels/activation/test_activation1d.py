"""The activation1d program against the five stages it fuses, unfused.

The operator is a chain (replicate pad, transposed convolution, crop,
SnakeBeta, low-pass), so the check builds each stage in torch, in the
operator's own order, and compares::

    xp  = replicate(x, kernel/ratio - 1)
    u   = crop(ratio * conv_transpose1d(xp, filter_up), 3 (kernel - ratio) / 2)
    y   = u + sin^2(exp(alpha) u) / (exp(beta) + 1e-9)
    out = conv1d(replicate(y, kernel/2 - 1, kernel/2), filter_down, ratio)

The fused form computes the upsampled values a low-pass tap needs from the
input itself, so the comparison is against the unfused chain: same
arithmetic, same order inside each sum, one rounding at the end. That is what
the operator's tolerance states.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.unit

#: The reciprocal gain's floor, as the operator writes it.
BETA_FLOOR = 1e-9

#: The geometries: channels, length, filter taps, ratio and the program's
#: block. `kernel = 12, ratio = 2` is the decoder's own pair, where every
#: derived constant is exact; a kernel and ratio whose `3 (kernel - ratio)` is
#: odd would need a half-sample crop, which the operator does not define.
GEOMETRIES = [(2, 8, 12, 2, 16), (1, 5, 12, 2, 8), (2, 8, 12, 2, 1024)]


def reference(values, alpha, beta, filter_up, filter_down, kernel, ratio):
    """The five stages, unfused, as `crates/ops/src/activation1d.rs`
    describes them.
    """  # noqa: D205
    channels, _ = values.shape
    pad = kernel // ratio - 1
    crop = (3 * (kernel - ratio)) // 2
    left = kernel // 2 - 1

    padded = F.pad(values, (pad, pad), mode="replicate")
    upsampled = ratio * F.conv_transpose1d(
        padded,
        filter_up.expand(channels, 1, kernel),
        stride=ratio,
        groups=channels,
    )
    cropped = upsampled[..., crop : upsampled.shape[-1] - crop]
    activated = cropped + torch.sin(
        torch.exp(alpha)[:, None] * cropped
    ) ** 2 / (torch.exp(beta)[:, None] + BETA_FLOOR)
    widened = F.pad(activated, (left, kernel // 2), mode="replicate")
    return F.conv1d(
        widened,
        filter_down.expand(channels, 1, kernel),
        stride=ratio,
        groups=channels,
    )


def launch(kernels, operands, out, channels, kernel, ratio, length, block):
    """One launch: a program per channel and block of input positions."""
    out_length = out.shape[-1]
    kernels._activation1d_kernel[(channels, (length + block - 1) // block, 1)](
        *operands,
        out,
        channels,
        kernel,
        ratio,
        length,
        out_length,
        block,
        KERNEL=kernel,
        RATIO=ratio,
        BLOCK=block,
    )


@pytest.mark.parametrize(
    ("channels", "length", "kernel", "ratio", "block"), GEOMETRIES
)
def test_the_fused_chain_matches_its_stages(
    channels, length, kernel, ratio, block, torch_device, kernel_source
):
    torch.manual_seed(0)
    values = torch.randn(channels, length, dtype=torch.float32) * 0.5
    alpha = torch.randn(channels, dtype=torch.float32) * 0.1
    beta = torch.randn(channels, dtype=torch.float32) * 0.1
    filter_up = torch.randn(1, 1, kernel, dtype=torch.float32) / kernel
    filter_down = torch.randn(1, 1, kernel, dtype=torch.float32) / kernel
    operands = (values, alpha, beta, filter_up, filter_down)

    out_length = (length * ratio - 1) // ratio + 1
    out = torch.full((channels, out_length), float("nan"), device=torch_device)
    launch(
        kernel_source("activation1d"),
        [operand.to(torch_device) for operand in operands],
        out,
        channels,
        kernel,
        ratio,
        length,
        block,
    )

    expected = reference(*operands, kernel, ratio)
    got = out.cpu()
    error = (got - expected).abs().max().item()
    largest = expected.abs().max().item()
    tolerance = 8.0 * torch.finfo(torch.float32).eps * max(largest, 1.0)
    assert not torch.isnan(got).any(), (
        f"{int(torch.isnan(got).sum())} outputs are NaN"
    )
    assert error <= tolerance, (
        f"channels={channels} length={length} kernel={kernel} ratio={ratio} "
        f"block={block}: max |error| = {error:.3e} (largest {largest:.3e}), "
        f"tolerance {tolerance:.3e}"
    )
