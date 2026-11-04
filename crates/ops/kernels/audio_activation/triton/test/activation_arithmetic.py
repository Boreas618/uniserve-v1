#!/usr/bin/env python3
"""Execute the audio_activation kernel on the CPU and check the five stages it fuses.

Triton's interpreter runs the kernel body with numpy-backed tensors, one program at a time. The
operator is a chain — replicate pad, transposed convolution, crop, SnakeBeta, low-pass — so the
check builds each stage in torch, in the operator's own order, and compares:

    xp    = replicate(x, kernel/ratio − 1)
    u     = crop(ratio · conv_transpose1d(xp, filter_up), 3·(kernel − ratio)/2)
    y     = u + sin²(exp(alpha)·u) / (exp(beta) + 1e-9)
    out   = conv1d(replicate(y, kernel/2 − 1, kernel/2), filter_down, stride=ratio)

The fused form computes the upsampled values a low-pass tap needs from the input itself, so the
comparison is against the unfused chain: same arithmetic, same order inside each sum, one rounding
at the end. That is what the operator's tolerance states.

    TRITON_INTERPRET=1 python activation_arithmetic.py
"""

import os
import pathlib
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

import audio_activation as kernels  # noqa: E402

#: The reciprocal gain's floor, as the operator writes it.
BETA_FLOOR = 1e-9


def reference(values, alpha, beta, filter_up, filter_down, kernel, ratio):
    """The five stages, unfused, as `crates/ops/src/audio_activation.rs` describes them."""
    channels, length = values.shape
    pad = kernel // ratio - 1
    crop = (3 * (kernel - ratio)) // 2
    left = kernel // 2 - 1

    padded = F.pad(values, (pad, pad), mode="replicate")
    upsampled = ratio * F.conv_transpose1d(
        padded, filter_up.expand(channels, 1, kernel), stride=ratio, groups=channels
    )
    cropped = upsampled[..., crop : upsampled.shape[-1] - crop]
    activated = cropped + torch.sin(torch.exp(alpha)[:, None] * cropped) ** 2 / (
        torch.exp(beta)[:, None] + BETA_FLOOR
    )
    widened = F.pad(activated, (left, kernel // 2), mode="replicate")
    out = F.conv1d(
        widened, filter_down.expand(channels, 1, kernel), stride=ratio, groups=channels
    )
    return out


def one(channels, length, kernel, ratio, block):
    """Check one geometry."""
    torch.manual_seed(0)
    values = torch.randn(channels, length, dtype=torch.float32) * 0.5
    alpha = torch.randn(channels, dtype=torch.float32) * 0.1
    beta = torch.randn(channels, dtype=torch.float32) * 0.1
    filter_up = torch.randn(1, 1, kernel, dtype=torch.float32) / kernel
    filter_down = torch.randn(1, 1, kernel, dtype=torch.float32) / kernel

    out_length = (length * ratio - 1) // ratio + 1
    out = torch.full((channels, out_length), float("nan"))
    grid = (channels, (length + block - 1) // block, 1)
    kernels._audio_activation_kernel[grid](
        values,
        alpha,
        beta,
        filter_up,
        filter_down,
        out,
        channels,
        kernel,
        ratio,
        length,
        out_length,
        block,
        0,
        KERNEL=kernel,
        RATIO=ratio,
        BLOCK=block,
    )

    expected = reference(values, alpha, beta, filter_up, filter_down, kernel, ratio)
    error = (out - expected).abs().max().item()
    largest = expected.abs().max().item()
    tolerance = 8.0 * torch.finfo(torch.float32).eps * max(largest, 1.0)
    print(
        f"channels={channels} length={length} kernel={kernel} ratio={ratio} block={block}: "
        f"max |error| = {error:.3e} (largest {largest:.3e})"
    )
    return not torch.isnan(out).any() and error <= tolerance


def main() -> int:
    # `kernel = 12, ratio = 2` is the decoder's own pair, where every derived constant is exact;
    # a kernel and ratio whose 3·(kernel − ratio) is odd would need a half-sample crop, which the
    # operator does not define.
    cases = [(2, 8, 12, 2, 16), (1, 5, 12, 2, 8), (2, 8, 12, 2, 1024)]
    return 0 if all(one(*case) for case in cases) else 1


if __name__ == "__main__":
    sys.exit(main())
