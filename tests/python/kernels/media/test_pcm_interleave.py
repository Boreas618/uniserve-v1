"""The pcm_interleave program against the PCM quantization, sample for sample.

The operator's tolerance is exact (``ops::pcm::PcmInterleave::tolerance``):
the output is an integer the formula fixes, so the comparison is an equality:

```text
quantize(x) = round_half_to_even(clamp(x, -1, 1) · 32767)     as i16
```

What it can get wrong is the rounding of ties: the reference's ``numpy.rint``
(``f32::round_ties_even``) takes them to the even neighbour, and a module that
truncated or rounded them away from zero would differ on exactly the half
samples. The samples therefore include values that land on a tie under the
scale, and both ends of the range and past them, in two channels written to
``2j`` and ``2j + 1``.

The kernel rounds with libdevice's ``rint``, which Triton's interpreter does not
evaluate, so the check runs on a device only.
"""

import numpy as np
import pytest
import torch
import triton

pytestmark = pytest.mark.unit

FULL_SCALE = np.float32(32767.0)

#: The samples per channel and the samples one program writes.
SAMPLES = 65536
BLOCK = 1024


def launch(kernels, left, right, out, samples, block):
    """One launch interleaving `samples` of both channels into `out`."""
    kernels._pcm_interleave_kernel[(triton.cdiv(samples, block),)](
        left,
        right,
        out,
        samples,
        block,
        0,
        BLOCK=block,
        PDL=False,
        num_warps=max(1, min(8, block // 512)),
    )


def quantize(x):
    """The reference: ``round_half_to_even(clamp(x, -1, 1) · 32767)`` in
    FP32.
    """  # noqa: D205
    clamped = np.clip(x, np.float32(-1.0), np.float32(1.0)).astype(np.float32)
    return np.rint(clamped * FULL_SCALE).astype(np.int16)


def samples(count, seed):
    """One channel of decoder-like floats with the samples a wrong rounding
    gets wrong.
    """  # noqa: D205
    generator = np.random.default_rng(seed)
    channel = generator.normal(0.0, 0.3, count).astype(np.float32)
    # Scaled, these are half-integers: half of them round up and half down,
    # which tells ties-to-even from ties-away-from-zero and from truncation.
    halves = np.arange(count // 8, dtype=np.float32) + np.float32(0.5)
    ties = (halves / FULL_SCALE).astype(np.float32)
    channel[: ties.size] = ties
    # Both ends and past them: the clamp, not a wrapping overflow.
    edges = np.array(
        [-2.0, -1.0, -0.9999999, 0.0, 0.9999999, 1.0, 2.0], dtype=np.float32
    )
    channel[-edges.size :] = edges
    return channel


@pytest.mark.gpu
def test_every_sample_is_the_quantized_value(torch_device, kernel_source):
    left = samples(SAMPLES, seed=0)
    right = samples(SAMPLES, seed=1)
    out = torch.empty(SAMPLES * 2, dtype=torch.int16, device=torch_device)

    launch(
        kernel_source("pcm_interleave"),
        torch.from_numpy(left).to(torch_device),
        torch.from_numpy(right).to(torch_device),
        out,
        SAMPLES,
        BLOCK,
    )

    got = out.view(-1, 2).cpu().numpy()
    expected = np.stack([quantize(left), quantize(right)], axis=1)
    wrong = np.argwhere(got != expected)
    if wrong.size:
        row, side = (int(one) for one in wrong[0])
        pytest.fail(
            f"{SAMPLES} samples, block {BLOCK}: {wrong.shape[0]} sample(s) "
            "differ, max |out - exact| = "
            f"{int(np.abs(got.astype(np.int32) - expected).max())}; the first "
            f"at ({row}, {side}): got {int(got[row, side])}, exact "
            f"{int(expected[row, side])} from "
            f"{float(left[row] if side == 0 else right[row]):.9g}"
        )
