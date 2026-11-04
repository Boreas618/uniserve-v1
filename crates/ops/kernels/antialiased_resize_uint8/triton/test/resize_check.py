#!/usr/bin/env python3
"""Run `antialiased_resize_uint8`'s kernel and check it against torch's antialiased resize.

    crates/program-server/python/.venv/bin/python resize_check.py [--self-test]

This is `interpolate(scale, mode="bilinear"|"bicubic", antialias=True, align_corners=False)` on a
`[rows, channels, height, width]` uint8 frame. The filter is separable: for an output index `i` along
an axis, `center = (i + 0.5)·scale − 0.5`, `support = (bicubic ? 2 : 1)·max(1, scale)`, the taps are
`floor(center ± support + 0.5)` narrowed to those **inside** the input, and a tap weighs the kernel at
its distance over the scale, divided by the sum of the kept taps so a constant image stays constant.

**Two things about the reference that a check must not guess.**

*The taps past an edge are dropped, not clamped.* Measured: a bilinear halving whose tap indices are
clamped to the frame differs from `F.interpolate` by **7.4 code values** at the border row, and the
same model with the taps dropped matches it **exactly**. The operator's kernel clamped the index; it
now drops the tap and renormalises, which is the fix this check was written around.

*`F.interpolate`'s bicubic constant is −0.5, not torchvision's −0.75.* Measured the same way: the
transcription below reproduces `F.interpolate`'s bicubic to **0.000000** with α = −0.5, and differs by
7.1 code values with α = −0.75, which is the constant `torchvision.transforms.functional.resize` and
this operator use. So bilinear cases are compared against `F.interpolate` itself and bicubic cases
against the transcription at α = −0.75, with `--self-test` proving both halves: that the transcription
is exact for the constant ATen uses, and that the two constants really differ.

**A fractional scale is compared against the operator's statement, not against torch**, and that is
the one soft place in this check: at 16 → 11 and 14 → 5 the transcription above differs from
`F.interpolate` by up to six code values, and I could not reduce torch's rule at a fractional scale to
a formulation that reproduces it — the taps, their drop rule, the renormalisation and the quantiser all
reproduce torch **exactly** at an integer scale (a halving and a quartering), so what is left is
something in the centre's arithmetic that no variant I tried matched. Those cases are therefore
compared against the operator's own statement of its rule, and only the integer-scale cases carry
torch's authority.

The operator's bound is **one code value** — the quantiser's rounding tie — `Tolerance::exact()
.with_atol(1.0)`. And every case's scale stays inside the kernel's eight-tap envelope (a halving with
the cubic, a quartering with the triangle), where the tap count never reaches `MAX_TAPS`: a case
beyond it measures the cap, not the arithmetic.

**The kernel cannot run on Triton's interpreter**: it walks a row with `tl.range(0, out_width)`, and
the interpreter hands that loop body Python integers, so `out_column.to(tl.float32)` raises `'int'
object has no attribute 'to'` before any pixel is summed. So the device path is unproven until a
device runs it, and `--self-test` is what proves the instrument here.
"""

import argparse
import importlib.util
import math
import os
import pathlib
import sys

import torch
import torch.nn.functional as F

MODES = ("bilinear", "bicubic")

#: The taps one axis may use, as the kernel's `MAX_TAPS` states it.
MAX_TAPS = 8

#: The cubic convolution's parameter, as torchvision's resize uses it and this operator's kernel
#: hard-codes it.
CUBIC_ALPHA = -0.75

#: ATen's own bicubic constant, which `F.interpolate` uses: measured, not documented — with −0.5 the
#: transcription below matches `F.interpolate` exactly, with −0.75 it does not.
ATEN_ALPHA = -0.5

#: `Tolerance::exact().with_atol(1.0)`: one 8-bit code value.
BOUND = 1


def load_kernel(package):
    """The package's Triton source, imported as a module."""
    source = package / "src" / "antialiased_resize_uint8.py"
    spec = importlib.util.spec_from_file_location("resize_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kernel_of(distance, mode, alpha):
    """The separable filter's value at `distance`, as the kernel evaluates it."""
    absolute = abs(distance)
    if mode == "bicubic":
        if absolute <= 1.0:
            return ((alpha + 2.0) * absolute - (alpha + 3.0)) * absolute * absolute + 1.0
        if absolute < 2.0:
            return alpha * (((absolute - 5.0) * absolute + 8.0) * absolute - 4.0)
        return 0.0
    return max(0.0, 1.0 - absolute)


def taps(extent, out_extent, mode, alpha):
    """The kept taps and weights along one axis: `(index, weight)` per output index."""
    scale = extent / out_extent
    support = (2.0 if mode == "bicubic" else 1.0) * max(1.0, scale)
    per_index = []
    for index in range(out_extent):
        center = (index + 0.5) * scale - 0.5
        start = math.floor(center - support + 0.5)
        count = min(math.floor(center + support + 0.5) - start, MAX_TAPS)
        kept = [(start + tap, kernel_of((start + tap - center) / scale, mode, alpha))
                for tap in range(count) if 0 <= start + tap < extent]
        total = sum(weight for _, weight in kept)
        per_index.append([(position, weight / total) for position, weight in kept])
    return per_index


def model(frames, out_height, out_width, mode, alpha=CUBIC_ALPHA):
    """The operator's formula transcribed, in FP64, quantised the way the operator quantises."""
    values = frames.to(torch.float64)
    rows, channels, height, width = values.shape
    vertical = taps(height, out_height, mode, alpha)
    horizontal = taps(width, out_width, mode, alpha)
    out = torch.zeros(rows, channels, out_height, out_width, dtype=torch.float64)
    for out_row, weights_h in enumerate(vertical):
        for out_column, weights_w in enumerate(horizontal):
            total = 0.0
            for source_row, weight_h in weights_h:
                for source_column, weight_w in weights_w:
                    total = total + weight_h * weight_w * values[:, :, source_row, source_column]
            out[:, :, out_row, out_column] = total
    return torch.clamp(torch.floor(out + 0.5), 0.0, 255.0).to(torch.int16)


def torch_resize(frames, out_height, out_width, mode):
    """torch's own antialiased resize, quantised the way the operator quantises."""
    resized = F.interpolate(frames.to(torch.float32), size=(out_height, out_width), mode=mode,
                            antialias=True, align_corners=False)
    return torch.clamp(torch.floor(resized + 0.5), 0.0, 255.0).to(torch.int16)


CASES = [
    # The fixture's cell: bicubic, three channels, a halving, where the cubic filter uses all eight
    # taps and every border row is where the tap rule shows.
    dict(what="the fixture's cell", mode="bicubic", channels=3, block=1024, rows=1, height=16,
         width=16, out_height=8, out_width=8, pattern="random", reference="operator"),
    # A checkerboard at a halving: the antialiasing itself.
    dict(what="a checkerboard halved", mode="bicubic", channels=1, block=256, rows=1, height=16,
         width=16, out_height=8, out_width=8, pattern="checker", reference="operator"),
    # A bilinear quartering, which torch's own rule reproduces exactly, so it is compared to torch.
    dict(what="a checkerboard quartered", mode="bilinear", channels=2, block=512, rows=1,
         height=32, width=32, out_height=8, out_width=8, pattern="checker", reference="torch"),
    dict(what="a bilinear halving", mode="bilinear", channels=3, block=256, rows=2, height=16,
         width=16, out_height=8, out_width=8, pattern="random", reference="torch"),
    # The kept taps' sum divides out, whatever the mode and scale.
    dict(what="a constant image, bicubic", mode="bicubic", channels=3, block=256, rows=1,
         height=13, width=13, out_height=7, out_width=9, pattern="constant",
         reference="operator"),
    dict(what="a constant image, bilinear", mode="bilinear", channels=4, block=256, rows=1,
         height=9, width=9, out_height=5, out_width=5, pattern="constant", reference="operator"),
    # Fractional scales: the taps' centres are fractional, so the drop-and-renormalise rule decides
    # the border rows. torch's own rule is not reproduced at a fractional scale by the formulation
    # below (see the module docstring), so these are compared to the operator's own statement.
    dict(what="a non-square bicubic at 16 -> 11", mode="bicubic", channels=4, block=256, rows=2,
         height=16, width=16, out_height=11, out_width=11, pattern="random",
         reference="operator"),
    dict(what="an uneven bilinear at 14 -> 5", mode="bilinear", channels=3, block=256, rows=2,
         height=14, width=14, out_height=5, out_width=5, pattern="random", reference="operator"),
    # An upscale: `max(1, scale)` is 1, so no widening.
    dict(what="an upscale", mode="bicubic", channels=3, block=128, rows=1, height=7, width=7,
         out_height=15, out_width=15, pattern="random", reference="operator"),
    dict(what="an uneven upscale", mode="bilinear", channels=2, block=128, rows=1, height=5,
         width=9, out_height=8, out_width=17, pattern="random", reference="operator"),
]


def frames_of(case, seed=37):
    """The case's frame batch, distinct in every pixel so a wrong tap cannot pass."""
    torch.manual_seed(seed)
    rows, channels = case["rows"], case["channels"]
    height, width = case["height"], case["width"]
    if case["pattern"] == "checker":
        picture = (torch.arange(height)[:, None] + torch.arange(width)[None, :]) % 2
        picture = (picture.to(torch.float32) * 255.0).to(torch.uint8)
        return picture.expand(rows, channels, height, width).contiguous()
    if case["pattern"] == "constant":
        return torch.full((rows, channels, height, width), 128, dtype=torch.uint8)
    return torch.randint(0, 256, (rows, channels, height, width), dtype=torch.uint8)


def expected_of(case, frames):
    """The reference for one case: torch where torch's rule is reproducible, else the operator's."""
    if case["reference"] == "torch":
        return torch_resize(frames, case["out_height"], case["out_width"], case["mode"])
    return model(frames, case["out_height"], case["out_width"], case["mode"], CUBIC_ALPHA)


def self_test():
    """Prove the instrument: the transcription, the two constants, and the case list's sharpness."""
    bilinear_exact = bicubic_exact = separated = 0
    for case in CASES:
        frames = frames_of(case)
        torch_side = torch_resize(frames, case["out_height"], case["out_width"], case["mode"])
        if case["reference"] == "torch":
            got = model(frames, case["out_height"], case["out_width"], case["mode"])
            difference = int((got - torch_side).abs().max())
            # The FP64 transcription and torch's FP32 path may break a rounding tie differently, so
            # the agreement is stated at the operator's own one-code bound; a systematic difference
            # in the taps or the renormalisation is much larger than a code.
            if difference > BOUND:
                raise SystemExit(
                    f"{case['what']}: the transcription is {difference} codes from torch's own "
                    f"resize, so the model is not the rule torch applies"
                )
            bilinear_exact += 1
        if case["mode"] == "bicubic":
            # ATen's own constant, on the cases where torch's rule itself is reproduced.
            aten = model(frames, case["out_height"], case["out_width"], "bicubic", ATEN_ALPHA)
            if int((aten - torch_side).abs().max()) <= BOUND:
                bicubic_exact += 1
            ours = model(frames, case["out_height"], case["out_width"], "bicubic", CUBIC_ALPHA)
            if int((ours - torch_side).abs().max()) > BOUND:
                separated += 1
    if not bilinear_exact:
        raise SystemExit(
            "no case compares the transcription against torch's own rule, so the taps, the dropped "
            "edge taps and the renormalisation are unproven")
    if not bicubic_exact:
        raise SystemExit(
            "no bicubic case reproduces torch's own rule at ATen's constant, so the cubic form is "
            "unproven and only its constant is")
    if not separated:
        raise SystemExit(
            "no case tells torchvision's cubic constant from ATen's, so a check that compares "
            "bicubic output to F.interpolate would pass a kernel with the wrong one")
    print(f"ok: the transcription reproduces torch's own rule on {bilinear_exact} integer-scale case"
          f"(s) and on {bicubic_exact} bicubic case(s) at ATen's constant; {separated} case(s) tell "
          f"torchvision's cubic constant from ATen's by more than a code value")


def check(module, case):
    """One resize, compared to the reference at the operator's one-code bound."""
    frames = frames_of(case)
    out = torch.full((case["rows"], case["channels"], case["out_height"], case["out_width"]), 0,
                     dtype=torch.uint8)
    scale_h = float(case["height"]) / float(case["out_height"])
    scale_w = float(case["width"]) / float(case["out_width"])
    units = case["rows"] * case["channels"] * case["out_height"]
    block = case["block"]
    module._antialiased_resize_uint8_kernel[((units + block - 1) // block, 1, 1)](
        frames, out, case["rows"], case["channels"], case["height"], case["width"],
        case["out_height"], case["out_width"], scale_h, scale_w, 0,
        CHANNELS=case["channels"], BICUBIC=case["mode"] == "bicubic", BLOCK=block, PDL=False,
    )

    expected = expected_of(case, frames)
    where = (f"{case['what']}: {case['mode']} {case['rows']}x{case['channels']} of "
             f"{case['height']}x{case['width']} -> {case['out_height']}x{case['out_width']}")
    difference = (out.to(torch.int16) - expected).abs()
    violations = int((difference > BOUND).sum())
    if violations:
        position = int((difference > BOUND).reshape(-1).nonzero()[0])
        raise SystemExit(
            f"{where}: {violations} of {out.numel()} codes are more than {BOUND} away from the "
            f"reference; first at {position}: {int(out.reshape(-1)[position])} vs "
            f"{int(expected.reshape(-1)[position])}"
        )
    if case["pattern"] == "constant" and int(difference.max()) != 0:
        raise SystemExit(
            f"{where}: a constant image resized to {int(out.max())}/{int(out.min())}, not 128 — the "
            f"taps' sum does not divide out"
        )
    return out.numel(), int((difference == 1).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="the package directory holding src/antialiased_resize_uint8.py",
    )
    parser.add_argument("--self-test", action="store_true",
                        help="prove the reference without the kernel")
    options = parser.parse_args()
    if options.self_test:
        return self_test() or 0
    if not torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
        module = load_kernel(options.package)
        case = CASES[0]
        frames = torch.zeros(1, 1, 4, 4, dtype=torch.uint8)
        try:
            module._antialiased_resize_uint8_kernel[(1, 1, 1)](
                frames, torch.zeros(1, 1, 2, 2, dtype=torch.uint8), 1, 1, 4, 4, 2, 2,
                2.0, 2.0, 0, CHANNELS=1, BICUBIC=True, BLOCK=128, PDL=False)
        except Exception as error:  # noqa: BLE001 - the tool's words are the evidence
            print(f"antialiased_resize_uint8 needs a device: its kernel walks a row with "
                  f"`tl.range(0, out_width)`, and Triton's interpreter hands that loop body Python "
                  f"integers — {type(error).__name__}: {str(error).splitlines()[0]}", file=sys.stderr)
        print("run this with a device, or `--self-test` for the reference and the case list",
              file=sys.stderr)
        return 1
    module = load_kernel(options.package)
    checked = ties = 0
    for case in CASES:
        count, tie = check(module, case)
        checked += count
        ties += tie
    print(f"ok: {checked} resized codes over {len(CASES)} cases, none more than one code value from "
          f"the reference, {ties} of them on a rounding tie")
    return 0


if __name__ == "__main__":
    sys.exit(main())
