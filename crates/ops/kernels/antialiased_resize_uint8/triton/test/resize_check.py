#!/usr/bin/env python3
"""Run `antialiased_resize_uint8`'s kernel and check it against the rule its contract names.

    crates/program-server/python/.venv/bin/python resize_check.py [--self-test] [--package DIR]

This is `interpolate(scale, mode="bilinear"|"bicubic", antialias=True, align_corners=False)` on a
`[rows, channels, height, width]` uint8 frame. The filter is separable: for an output index `i` along
an axis, `center = (i + 0.5)·scale − 0.5` and `support = (bicubic ? 2 : 1)·max(1, scale)`; the taps
that contribute are the integers in `[center − support, center + support]`, each weighing the kernel at
its distance divided by the scale, and the weights are divided by the sum of the ones that stay, so a
constant image stays constant.

**The contract this program implements names torchvision's antialiased resize; H3's live image and
video paths use Pillow's LANCZOS (`uniserve/media/reader.py`) and swscale's lanczos
(`uniserve/video/media.py`) respectively, so a caller comparing against either must expect them to
differ.** The check is pointed at the authority the contract names, not at whichever resampler is
easier to call: that is the mistake the −0.5/−0.75 finding below is about. H3's aspect fit is also
followed by a **crop**, so a resize that is right can still land in the wrong pixels if its edge
convention differs from the cropper's; nothing here can see that.

**Three things about the reference a check must not guess.**

*The taps past an edge are dropped, not clamped.* Measured: a bilinear halving whose tap indices are
clamped to the frame differs from the rule by **7.4 code values** at the border row, and the same model
with the taps dropped matches it **exactly**.

*The window's ends are the interval's.* The contributing taps are those in
`[center − support, center + support]`, so the window runs from its **ceiling** to its **floor**.
Rounding both ends — `floor(center ± support + 0.5)`, the natural-looking form — drops the last
contributing tap whenever `center + support` is not near a half-integer, and at an integer scale it
usually is. Measured on a random frame against `F.interpolate`, worst pixel of the frame:

| scale | rounded ends | the interval |
| --- | --- | --- |
| 16 → 11 | 23.19 code values | 0.0000 |
| 13 → 9 | 20.17 | 0.0000 |
| 9 → 5 | 8.66 | 0.0000 |
| 14 → 5 | 4.31 | 0.0000 |
| 16 → 8, 32 → 8 | 0.0000 | 0.0000 |

against a bound of one code value. The kernel and this transcription both had the rounded window, which
is why they agreed with each other and were both wrong.

*`F.interpolate`'s bicubic constant is −0.5, not torchvision's −0.75.* Measured the same way: the
transcription matches `F.interpolate` to **0.000000** with α = −0.5 and differs by 7.1 code values with
α = −0.75, which is the constant `torchvision.transforms.functional.resize` and this operator use (the
kernel hard-codes it). So bilinear cases are compared against `F.interpolate` itself, bicubic cases
against the transcription at −0.75, and `--self-test` proves both halves.

**The cases are the two scale pairs the pipeline runs** — `1280x720 → 1344x768` (1.05, an *upscale* in
both axes) and `1920x1080 → 1344x768` (0.70) — not a toy geometry, so whatever authority is named later,
the shapes under test are the shapes the canvas is. An **upscale is compared against the transcription
and not against torch**, and that is measured rather than assumed:
`F.interpolate(mode="bilinear", antialias=True)` on a 5×9 → 8×17 frame differs from the same separable
rule by **109** code values, so torch's own upscale path is a question this check does not answer. The
small pairs remain, in `--self-test`, as the lattice on which the transcription is proved against torch.

The operator's bound is **one code value** — the quantiser's rounding tie — `Tolerance::exact()
.with_atol(1.0)`. Every scale stays inside the kernel's eight-tap envelope, where the tap count never
reaches `MAX_TAPS`: a case beyond it measures the cap, not the arithmetic.

**The kernel cannot run on Triton's interpreter**: it walks a row with `tl.range(0, out_width)`, and the
interpreter hands that loop body Python integers, so `out_column.to(tl.float32)` raises `'int' object
has no attribute 'to'` before any pixel is summed. Without a device the script says so and exits 0; the
device path is written and unproven until a device runs it.
"""

import argparse
import importlib.util
import math
import os
import pathlib
import sys

import torch
import torch.nn.functional as F

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


def weights(extent, out_extent, mode, alpha=CUBIC_ALPHA):
    """The separable weights along one axis: one row of `out_extent` values per output index."""
    scale = extent / out_extent
    support = (2.0 if mode == "bicubic" else 1.0) * max(1.0, scale)
    matrix = torch.zeros(out_extent, extent, dtype=torch.float64)
    for index in range(out_extent):
        center = (index + 0.5) * scale - 0.5
        # The window is the interval the filter covers: from its ceiling to its floor.
        start = math.floor(center - support) + 1
        count = min(math.floor(center + support) - start + 1, MAX_TAPS)
        kept = [(start + tap, kernel_of((start + tap - center) / scale, mode, alpha))
                for tap in range(count) if 0 <= start + tap < extent]
        total = sum(weight for _, weight in kept)
        for position, weight in kept:
            matrix[index, position] = weight / total
    return matrix


def model(frames, out_height, out_width, mode, alpha=CUBIC_ALPHA):
    """The operator's formula, in FP64, quantised the way the operator quantises.

    Written as the separable product it is: a `[out, in]` weight matrix per axis, applied to the frame.
    A per-pixel loop is the same arithmetic and too slow to run at 1080p.
    """
    values = frames.to(torch.float64)
    vertical = weights(frames.shape[2], out_height, mode, alpha)
    horizontal = weights(frames.shape[3], out_width, mode, alpha)
    out = torch.einsum("ih,rchw,jw->rcij", vertical, values, horizontal)
    return torch.clamp(torch.floor(out + 0.5), 0.0, 255.0).to(torch.int16)


def torch_resize(frames, out_height, out_width, mode):
    """torch's own antialiased resize, quantised the way the operator quantises."""
    resized = F.interpolate(frames.to(torch.float32), size=(out_height, out_width), mode=mode,
                            antialias=True, align_corners=False)
    return torch.clamp(torch.floor(resized + 0.5), 0.0, 255.0).to(torch.int16)


#: The two scale pairs the pipeline runs. `reference` says which authority the case is compared against:
#: `torch` where `F.interpolate` reproduces the contract's rule, `operator` where it does not.
CASES = [
    # 1920x1080 -> 1344x768: 0.70 both axes, the canvas H3 fits down to.
    dict(what="1920x1080 -> 1344x768, bicubic", mode="bicubic", channels=3, block=1024, rows=1,
         height=1080, width=1920, out_height=768, out_width=1344, pattern="random",
         reference="operator", seed=21),
    dict(what="1920x1080 -> 1344x768, bilinear", mode="bilinear", channels=3, block=1024, rows=1,
         height=1080, width=1920, out_height=768, out_width=1344, pattern="random",
         reference="torch", seed=22),
    # 1280x720 -> 1344x768: 1.05 and 1.067 both axes, an upscale in each — compared against the
    # transcription, because torch's own upscale path is not this rule (see the module docstring).
    dict(what="1280x720 -> 1344x768, bicubic", mode="bicubic", channels=3, block=1024, rows=1,
         height=720, width=1280, out_height=768, out_width=1344, pattern="random",
         reference="operator", seed=23),
    dict(what="1280x720 -> 1344x768, bilinear", mode="bilinear", channels=3, block=1024, rows=1,
         height=720, width=1280, out_height=768, out_width=1344, pattern="random",
         reference="operator", seed=24),
    # The kept taps' sum divides out, whatever the mode and scale: a constant frame stays constant.
    dict(what="a constant 1280x720 frame, bicubic", mode="bicubic", channels=3, block=1024, rows=1,
         height=720, width=1280, out_height=768, out_width=1344, pattern="constant",
         reference="operator", seed=25),
]

#: The lattice on which `--self-test` proves the transcription against torch, per mode, kept inside the
#: kernel's eight-tap envelope: a bilinear window holds `2·scale + 1` taps and a cubic `4·scale + 1`, so a
#: cubic case is at most a halving. A pair beyond it would measure `MAX_TAPS` and not the arithmetic.
PROOF_PAIRS = {
    "bilinear": [(16, 11), (13, 9), (9, 5), (14, 5), (16, 8), (32, 8), (33, 17)],
    "bicubic": [(16, 11), (13, 9), (16, 9), (9, 8), (16, 8), (33, 17)],
}


def frames_of(case):
    """The case's frame batch, distinct in every pixel so a wrong tap cannot pass."""
    generator = torch.Generator().manual_seed(case["seed"])
    rows, channels = case["rows"], case["channels"]
    height, width = case["height"], case["width"]
    if case["pattern"] == "constant":
        return torch.full((rows, channels, height, width), 128, dtype=torch.uint8)
    return torch.randint(0, 256, (rows, channels, height, width), generator=generator,
                         dtype=torch.uint8)


def expected_of(case, frames):
    """The reference for one case: torch where torch's rule is the contract's, else the operator's."""
    if case["reference"] == "torch":
        return torch_resize(frames, case["out_height"], case["out_width"], case["mode"])
    return model(frames, case["out_height"], case["out_width"], case["mode"], CUBIC_ALPHA)


def self_test():
    """Prove the instrument: the rule at many scales, the two constants, and the case list's weight."""
    bilinear_exact = bicubic_exact = separated = 0
    for mode, pairs in PROOF_PAIRS.items():
        for extent, out in pairs:
            generator = torch.Generator().manual_seed(extent * 100 + out)
            picture = torch.randint(0, 256, (1, 1, extent, extent), generator=generator,
                                    dtype=torch.uint8)
            torch_side = torch_resize(picture, out, out, mode)
            if mode == "bilinear":
                difference = int((model(picture, out, out, mode) - torch_side).abs().max())
                if difference > BOUND:
                    raise SystemExit(
                        f"bilinear {extent} -> {out}: the transcription is {difference} codes from "
                        f"torch's own resize, so the model is not the rule torch applies")
                bilinear_exact += 1
                continue
            aten = model(picture, out, out, mode, ATEN_ALPHA)
            if int((aten - torch_side).abs().max()) > BOUND:
                raise SystemExit(
                    f"bicubic {extent} -> {out}: the transcription at ATen's own constant is not "
                    f"torch's answer, so the cubic form is wrong and not only its constant")
            bicubic_exact += 1
            ours = model(picture, out, out, mode, CUBIC_ALPHA)
            if int((ours - torch_side).abs().max()) > BOUND:
                separated += 1
    if not bilinear_exact:
        raise SystemExit(
            "no case compares the transcription against torch's own rule, so the window, the dropped "
            "edge taps and the renormalisation are unproven")
    if not bicubic_exact:
        raise SystemExit(
            "no case reproduces torch's rule at ATen's constant, so the cubic form is unproven and "
            "only its constant is")
    if not separated:
        raise SystemExit(
            "no case tells torchvision's cubic constant from ATen's, so a check that compared "
            "bicubic output to F.interpolate would pass a kernel with the wrong one")
    # And the reference holds at the sizes the pipeline runs, which is what the cases are.
    for case in CASES:
        if case["reference"] != "torch":
            continue
        frames = frames_of(case)
        difference = int((model(frames, case["out_height"], case["out_width"], case["mode"])
                          - torch_resize(frames, case["out_height"], case["out_width"],
                                         case["mode"])).abs().max())
        if difference > BOUND:
            raise SystemExit(
                f"{case['what']}: the transcription is {difference} codes from torch at the size the "
                f"pipeline runs")
    print(f"ok: the transcription is torch's own rule on {bilinear_exact} bilinear pairs and on "
          f"{bicubic_exact} bicubic pairs at ATen's constant, {separated} pairs tell torchvision's "
          f"cubic constant from ATen's by more than a code value, and it holds at the two scale pairs "
          f"the pipeline runs")
    return 0


def check(module, case):
    """One resize, compared to the reference at the operator's one-code bound."""
    frames = frames_of(case)
    out = torch.zeros((case["rows"], case["channels"], case["out_height"], case["out_width"]),
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
    where = (f"{case['what']}: {case['rows']}x{case['channels']} of {case['height']}x"
             f"{case['width']} -> {case['out_height']}x{case['out_width']}, against "
             f"{'torch' if case['reference'] == 'torch' else 'the transcription'}")
    difference = (out.to(torch.int16) - expected).abs()
    violations = int((difference > BOUND).sum())
    if violations:
        position = int((difference > BOUND).reshape(-1).nonzero()[0])
        raise SystemExit(
            f"{where}: {violations} of {out.numel()} codes are more than {BOUND} away from the "
            f"reference; first at {position}: {int(out.reshape(-1)[position])} vs "
            f"{int(expected.reshape(-1)[position])}")
    if case["pattern"] == "constant" and int(difference.max()) != 0:
        raise SystemExit(
            f"{where}: a constant frame resized to {int(out.max())}/{int(out.min())}, not 128 — the "
            f"taps' sum does not divide out")
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
        return self_test()
    if not torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
        module = load_kernel(options.package)
        frames = torch.zeros(1, 1, 4, 4, dtype=torch.uint8)
        try:
            module._antialiased_resize_uint8_kernel[(1, 1, 1)](
                frames, torch.zeros(1, 1, 2, 2, dtype=torch.uint8), 1, 1, 4, 4, 2, 2,
                2.0, 2.0, 0, CHANNELS=1, BICUBIC=True, BLOCK=128, PDL=False)
        except Exception as error:  # noqa: BLE001 - the tool's words are the evidence
            print(f"skipped: needs a device — its kernel walks a row with `tl.range(0, out_width)`, "
                  f"and Triton's interpreter hands that loop body Python integers: "
                  f"{type(error).__name__}: {str(error).splitlines()[0]}")
        print("skipped: needs a device — run this on a device, or `--self-test` for the reference "
              "and the case list")
        return 0
    module = load_kernel(options.package)
    checked = ties = 0
    for case in CASES:
        count, tie = check(module, case)
        checked += count
        ties += tie
        print(f"  {case['what']}: {count} codes")
    print(f"ok: {checked} resized codes over {len(CASES)} cases, none more than one code value from "
          f"the reference, {ties} of them on a rounding tie")
    return 0


if __name__ == "__main__":
    sys.exit(main())
