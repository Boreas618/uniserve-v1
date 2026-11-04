#!/usr/bin/env python3
"""Run the audio resampler and check every sample against the reference module's formula.

    python resample_check.py                              # downsample, ratio 2, kernel 12
    python resample_check.py --mode upsample --ratio 6 --kernel 12
    python resample_check.py --interpreter                 # run on the CPU

Both directions are one depthwise convolution behind a Kaiser-windowed sinc whose taps sum
to one, and both reproduce the reference module's **replicate pad**, which is what keeps the
resampler from ringing at the ends of a waveform — so the ends are the first thing this check
is built to catch, and the pad is `replicate`, not zero and not a clamp of the whole window.

- **Downsample** pads `kernel/2 − (kernel even)` samples on the left and `kernel/2` on the
  right, replicates, and takes `out[j] = Σ_t padded[j·ratio + t] · tap[t]`, of length
  `floor((padded − kernel)/ratio) + 1`.
- **Upsample** pads `kernel/ratio − 1` samples on both sides, replicates, applies the
  transposed convolution of stride `ratio` scaled by `ratio`, and crops
  `pad·ratio + (kernel − ratio)/2` from the front and `pad·ratio + (kernel − ratio + 1)/2`
  from the end, which leaves exactly `width · ratio` samples.

The filter is the caller's, so it is built here as the reference builds it: a Kaiser-windowed
sinc at `cutoff = 0.5/ratio`, normalised so the taps sum to one.

The bound is the operator's own, `Tolerance::dot(kernel, F32, out)` — one output sample is one
`kernel`-term dot product in FP32 rounded once into the output type, so the bound carries the
`√kernel` a plain sum bound would not. `atol` is zero and a dot product of signed taps
cancels, so the run also prints the same comparison with the floor the taps imply
(`2·γ_k·√k·Σ|tap·sample|`), and `--bound` decides the run, defaulting to `declared`.

**The case that matters most is `ratio = 6, kernel = 12`**, where the tap count is exactly
`2 · ratio` and the sinc is sampled at its sparsest: a kernel that gets the phase wrong is off
by a whole tap there while every `ratio = 2` case still looks right. The second is the ends —
a one-sample-wide input, and an input whose length the ratio does not divide.

A `bf16` run under Triton's interpreter truncates its reference the way the interpreter
truncates narrow stores: the value's own mantissa, with no narrower type in between.

**The mutation this check was built against is the filter shifted by one tap**
(`filter[tap + 1]`), and it is caught at both phases and in both directions: 6 of 6 elements
outside the bound at `ratio 6, kernel 12` downsampling, 16 of 16 at `ratio 2, kernel 24`,
158 of 192 and 64 of 64 upsampling. That matters because a check of a convolution whose
reference shares the kernel's indexing cannot see the error at all, and the pair is what
settles it.
"""

import argparse
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

#: The output types' unit roundoff, as `DType::unit_roundoff` gives it.
UNIT_ROUNDOFF = {"f16": 2.0**-11, "bf16": 2.0**-8, "f32": 2.0**-24}

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}


def gamma(reduction):
    """`γₙ = n·u/(1 − n·u)` for `n` FP32 roundings, as the operator crate computes it."""
    terms = reduction * UNIT_ROUNDOFF["f32"]
    return terms / (1.0 - terms)


def bound(kernel, dtype):
    """The operator's bound: `2·γ_k·√k + u_out`, from `Tolerance::dot`."""
    return 2.0 * gamma(kernel) * math.sqrt(kernel) + UNIT_ROUNDOFF[dtype]


def truncate(values):
    """The reference as Triton's interpreter stores it: one ulp low, not rounded."""
    bits = values.view(torch.int64)
    return (bits & ~((1 << 45) - 1)).view(torch.float64)


def kaiser_sinc(kernel, ratio):
    """The reference resampler's filter: a Kaiser-windowed sinc, taps summing to one."""
    import numpy

    cutoff = 0.5 / ratio
    beta = 8.6  # The reference module's window shape.
    taps = torch.zeros(kernel, dtype=torch.float64)
    centre = (kernel - 1) / 2.0
    for index in range(kernel):
        offset = index - centre
        sinc = (
            1.0
            if offset == 0.0
            else math.sin(2.0 * math.pi * cutoff * offset)
            / (2.0 * math.pi * cutoff * offset)
        )
        window = float(
            numpy.i0(beta * math.sqrt(max(0.0, 1.0 - (offset / centre) ** 2)))
            / numpy.i0(beta)
        )
        taps[index] = 2.0 * cutoff * sinc * window
    return taps / taps.sum()


def replicate_pad(signal, left, right):
    """Pad every channel's samples by repeating its first and last sample."""
    channels = signal.shape[0]
    return torch.cat(
        [
            signal[:, :1].repeat(1, left),
            signal,
            signal[:, -1:].repeat(1, right),
        ],
        dim=1,
    )


def reference(x, taps, mode, ratio, kernel):
    """The reference module's own arithmetic, in FP64, for one `[channels, width]` signal."""
    channels, width = x.shape
    outputs = []
    if mode == "downsample":
        left = kernel // 2 - (1 if kernel % 2 == 0 else 0)
        right = kernel // 2
        padded = replicate_pad(x, left, right)
        out_width = (padded.shape[1] - kernel) // ratio + 1
        for channel in range(channels):
            row = padded[channel]
            samples = torch.zeros(out_width, dtype=torch.float64)
            for index in range(out_width):
                start = index * ratio
                samples[index] = (row[start : start + kernel] * taps).sum()
            outputs.append(samples)
    else:
        pad = kernel // ratio - 1
        padded = replicate_pad(x, pad, pad)
        length = padded.shape[1]
        # The transposed convolution of stride `ratio`, scaled by `ratio`.
        full = (length - 1) * ratio + kernel
        pad_left = pad * ratio + (kernel - ratio) // 2
        pad_right = pad * ratio + (kernel - ratio + 1) // 2
        out_width = width * ratio
        for channel in range(channels):
            row = padded[channel]
            transposed = torch.zeros(full, dtype=torch.float64)
            for sample in range(length):
                base = sample * ratio
                for tap in range(kernel):
                    transposed[base + tap] += row[sample] * taps[tap]
            transposed *= ratio
            end = full - pad_right
            outputs.append(transposed[pad_left:end])
        if any(samples.numel() != out_width for samples in outputs):
            raise AssertionError(
                f"the crop leaves {outputs[0].numel()} samples, not {out_width}"
            )
    return torch.stack(outputs)


def geometry(mode, width, kernel, ratio):
    """The three pad amounts the call passes, from the operator's own arithmetic."""
    if mode == "downsample":
        left = kernel // 2 - (1 if kernel % 2 == 0 else 0)
        right = kernel // 2
        out_width = (width + left + right - kernel) // ratio + 1
        return 0, left, right, out_width
    pad = kernel // ratio - 1
    left = pad * ratio + (kernel - ratio) // 2
    right = pad * ratio + (kernel - ratio + 1) // 2
    return pad, left, right, width * ratio


def check(case):
    """Resample one signal and compare every sample against the reference."""
    import audio_resample as kernels

    dtype = TORCH_TYPES[case["dtype"]]
    generator = torch.Generator().manual_seed(case["seed"])
    channels, width = case["channels"], case["width"]
    kernel, ratio = case["kernel"], case["ratio"]
    if kernel % ratio == 0 and kernel // ratio == 1:
        raise AssertionError(
            "a filter shorter than twice the ratio cannot low-pass"
        )

    x = torch.randn(1, channels, width, generator=generator) * 0.5
    taps = kaiser_sinc(kernel, ratio)
    pad, pad_left, pad_right, out_width = geometry(
        case["mode"], width, kernel, ratio
    )
    out = torch.full((1, channels, out_width), float("nan"), dtype=dtype)

    kernels._audio_resample_kernel[(math.ceil(out_width / case["block"]),)](
        x,
        taps.to(torch.float32),
        out,
        1,
        channels,
        width,
        out_width,
        pad,
        pad_left,
        pad_right,
        0,
        CHANNELS=channels,
        KERNEL=kernel,
        RATIO=ratio,
        UPSAMPLE=case["mode"] == "upsample",
        BLOCK=case["block"],
        PDL=False,
    )

    expected = reference(
        x[0].double(), taps, case["mode"], ratio, kernel
    ).unsqueeze(0)
    if case["truncate"]:
        expected = truncate(expected)
    got = out.double()
    difference = (got - expected).abs()

    relative = bound(kernel, case["dtype"])
    # The taps carry signs, so the roundings are of the terms rather than of their sum.
    padded = torch.nn.functional.pad(
        x[0].double().abs(), (kernel, kernel), mode="replicate"
    )
    floor = (
        2.0
        * gamma(kernel)
        * math.sqrt(kernel)
        * (
            padded.unfold(1, kernel, 1).amax(dim=-1) * taps.abs().sum()
        ).unsqueeze(0)
    )
    floor = floor[:, :, : got.shape[2]]
    if floor.shape != expected.shape:
        floor = torch.zeros_like(expected)
    declared_outside = (~(difference <= relative * expected.abs())).sum().item()
    scaled_outside = (
        (~(difference <= relative * expected.abs() + floor)).sum().item()
    )
    largest = (
        (difference / expected.abs().clamp(min=1e-30))
        .nan_to_num(nan=float("inf"))
        .max()
        .item()
    )
    deciding = (
        declared_outside if case["bound"] == "declared" else scaled_outside
    )
    print(
        f"{'exact' if deciding == 0 else 'FAIL'}: {case['mode']}, {case['label']} — "
        f"{channels} channels of {width} samples, ratio {ratio}, kernel {kernel} "
        f"({kernel // ratio} taps per output sample per phase), {case['dtype']}, "
        f"out {out_width}"
        f"{', interpreter cast truncated' if case['truncate'] else ''}"
    )
    print(
        f"  max absolute {difference.nan_to_num(nan=float('inf')).max().item():.3e}, "
        f"max relative {largest:.3e}, declared bound {relative:.3e} "
        f"({declared_outside} of {expected.numel()} outside), with the taps' floor "
        f"{scaled_outside} outside"
    )
    return deciding


def cases(flags, mode):
    """The cases one run checks, the sparsest phase first."""
    common = {"dtype": flags.dtype, "mode": mode, "block": flags.block}
    return [
        {
            **common,
            "label": "the sparsest phase: two taps per ratio step",
            "channels": min(flags.channels, 4),
            "width": flags.width,
            "kernel": 12,
            "ratio": 6,
            "seed": 1,
        },
        {
            **common,
            "label": "a long filter, where a one-tap shift is a small error",
            "channels": min(flags.channels, 4),
            "width": flags.width,
            "kernel": 24,
            "ratio": 2,
            "seed": 2,
        },
        {
            **common,
            "label": "the fixture's own geometry, one channel",
            "channels": 1,
            "width": flags.width,
            "kernel": flags.kernel,
            "ratio": flags.ratio,
            "seed": 3,
        },
        {
            **common,
            "label": "a single sample, which is all pad",
            "channels": 2,
            "width": 1,
            "kernel": flags.kernel,
            "ratio": flags.ratio,
            "seed": 4,
        },
        {
            **common,
            "label": "a length the ratio does not divide",
            "channels": 2,
            "width": flags.width + 3,
            "kernel": flags.kernel,
            "ratio": flags.ratio,
            "seed": 5,
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["downsample", "upsample"], default="downsample"
    )
    parser.add_argument("--dtype", choices=sorted(TORCH_TYPES), default="f32")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--kernel", type=int, default=12)
    parser.add_argument("--ratio", type=int, default=2)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--pdl", default="false")
    parser.add_argument("--expect-key", default=None)
    parser.add_argument(
        "--bound",
        choices=["declared", "scale"],
        default="declared",
        help="which bound decides the run: the operator's own (atol 0), or the same with "
        "the floor the signed taps imply",
    )
    parser.add_argument(
        "--interpreter",
        action="store_true",
        help="run through Triton's interpreter instead of a device",
    )
    arguments = parser.parse_args()
    if arguments.pdl != "false":
        parser.error(
            "the interpreter cannot run a dependent launch; use --pdl false"
        )
    if not arguments.interpreter and not torch.cuda.is_available():
        # A host without a device takes the interpreter: a check must run where it lands.
        print(
            "no GPU on this host: running through Triton's interpreter instead",
            file=sys.stderr,
        )
        arguments.interpreter = True
    if arguments.interpreter:
        import os

        os.environ["TRITON_INTERPRET"] = "1"

    key = (
        f"mode_{arguments.mode}_dtype_{arguments.dtype}"
        f"_channels_{arguments.channels}_kernel_{arguments.kernel}"
        f"_ratio_{arguments.ratio}_block_{arguments.block}_pdl_{arguments.pdl}"
    )
    print(f"descriptor: {key}")
    if arguments.expect_key is not None and arguments.expect_key != key:
        parser.error(f"the flags name {key}, not {arguments.expect_key}")

    failures = 0
    for case in cases(arguments, arguments.mode):
        # The interpreter's narrow store truncates; simulate it only where it is known.
        case["truncate"] = arguments.interpreter and arguments.dtype == "bf16"
        case["bound"] = arguments.bound
        failures += check(case)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
