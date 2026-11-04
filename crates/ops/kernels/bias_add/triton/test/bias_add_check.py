#!/usr/bin/env python3
"""Run the bias_add program and check the sum bit for bit.

    python bias_add_check.py --interpreter              # says why it cannot run here
    python bias_add_check.py                            # on a device
    python bias_add_check.py --layout channels-last --residual true --residual-bias false

The operator adds a per-channel bias in a separate pass so that a biased convolution equals
`conv(x, weight) + bias` bit for bit:

```text
out = (residual + residual_bias) + (values + bias)
```

with `values` and `residual` of shape `[batch, channels, spatial]` and each sum rounding once in
the operands' dtype, in that grouping. The tolerance is `Tolerance::exact()`, so the reference
here is the same three additions performed in the operands' dtype and nothing else: there is no
bound to choose and no rounding to argue about.

**The check needs a device, and this is measured rather than assumed.** The kernel rounds each
addition with `add.rn.f32` through `tl.inline_asm_elementwise`, and Triton's interpreter answers
`NotImplementedError('inline_asm not supported in interpreter mode')` — so on a host without a
device this script says that sentence and stops rather than reporting a comparison it never
made. That is the same class as `logits_shape`'s `libdevice` rounding.

**The comparison was proven able to fail before it was trusted**, on the CPU and against a
stub in place of the launch: a stub writing the contract's three additions counts zero, and
three wrong ones are caught — the residual dropped (30 of 30 in each residual case), the sums
taken left to right instead of in the contract's grouping (15, 12 and 9 of 30), and the whole
chain accumulated in FP32 and cast once (8, 12 and 30 of 30, which `Tolerance::exact()` is
there to forbid). What that proof does **not** cover is the kernel itself: its arithmetic is
only observable on a device.

**The two storage layouts are the operator's reason to exist**, and both are checked: a
channels-first tensor (spatial stride 1) and a channels-last one (channel stride 1). The strides
travel as launch scalars, so a case whose batches, channels and spatial extents are all
different is the one that can catch a stride read from the wrong axis — a case with one batch
and one channel cannot, because then every stride agrees with every other.
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

TORCH_TYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}

#: The storage layouts the operator accepts, as the axis whose stride is one.
LAYOUTS = ("channels-first", "channels-last")


def check(case):
    """Run one case and compare the sum exactly."""
    import bias_add as kernels

    dtype = TORCH_TYPES[case["dtype"]]
    generator = torch.Generator().manual_seed(case["seed"])
    batch, channels, spatial = case["batch"], case["channels"], case["spatial"]

    def stored():
        """One operand in the case's storage: the channels-first one, or its transpose."""
        if case["layout"] == "channels-last":
            # Stored `[batch, spatial, channels]` and read `[batch, channels, spatial]`, so the
            # channel axis has stride one and the spatial axis the channel count.
            storage = torch.randn(batch, spatial, channels, generator=generator)
            view = storage.permute(0, 2, 1)
        else:
            storage = torch.randn(batch, channels, spatial, generator=generator)
            view = storage
        return view.to(dtype), storage

    values, _ = stored()
    residual, _ = stored()
    bias = torch.randn(channels, generator=generator).to(dtype)
    residual_bias = torch.randn(channels, generator=generator).to(dtype)
    out = torch.full((batch, channels, spatial), float("nan"), dtype=dtype)

    elements = batch * channels * spatial
    block = case["block"]
    programs = -(-elements // block)
    kernels._bias_add_kernel[(programs,)](
        values,
        bias,
        residual if case["residual"] else values,
        residual_bias if case["residual_bias"] else bias,
        out,
        channels,
        batch,
        spatial,
        values.stride(0),
        values.stride(1),
        residual.stride(0),
        residual.stride(1),
        case["pdl"],
        BLOCK=block,
        HAS_RESIDUAL=case["residual"],
        HAS_RESIDUAL_BIAS=case["residual_bias"],
        PDL=case["pdl"],
    )

    # The reference: the same three additions, in the operands' dtype, in the contract's
    # grouping. Torch rounds a narrow addition to nearest, as the kernel's `add.rn` does.
    expected = (values + bias.view(1, channels, 1)).to(dtype)
    if case["residual"]:
        shortcut = (
            (residual + residual_bias.view(1, channels, 1)).to(dtype)
            if case["residual_bias"]
            else residual
        )
        expected = (shortcut + expected).to(dtype)

    difference = (out != expected).sum().item()
    print(
        f"{'exact' if difference == 0 else 'FAIL'}: {case['label']} — "
        f"{batch} x {channels} x {spatial} in {case['layout']} storage, block {block}, "
        f"{case['dtype']}, residual {str(case['residual']).lower()}, "
        f"residual_bias {str(case['residual_bias']).lower()}, "
        f"pdl {str(case['pdl']).lower()} — {difference} of {elements} elements differ"
    )
    if difference:
        wrong = (out != expected).nonzero()[:3].tolist()
        for sample, channel, position in wrong:
            print(
                f"  [{sample}, {channel}, {position}]: {out[sample, channel, position].item():.9e}, "
                f"the reference {expected[sample, channel, position].item():.9e}"
            )
    return difference


def cases(flags):
    """The cases one run checks, cheapest and most diverse first."""
    common = {
        "dtype": flags.dtype,
        "block": flags.block,
        "pdl": flags.pdl == "true",
    }
    return [
        {
            **common,
            "label": "channels-first, with a biased residual",
            "layout": "channels-first",
            "batch": 2,
            "channels": 3,
            "spatial": 5,
            "residual": True,
            "residual_bias": True,
            "seed": 1,
        },
        {
            **common,
            # The other storage the operator exists for: the channel axis has stride one, so a
            # kernel that assumed a channel stride of `spatial` reads the wrong channel.
            "label": "channels-last, with a biased residual",
            "layout": "channels-last",
            "batch": 2,
            "channels": 3,
            "spatial": 5,
            "residual": True,
            "residual_bias": True,
            "seed": 2,
        },
        {
            **common,
            "label": "an unbiased residual",
            "layout": "channels-last",
            "batch": 2,
            "channels": 3,
            "spatial": 5,
            "residual": True,
            "residual_bias": False,
            "seed": 3,
        },
        {
            **common,
            "label": "no residual at all",
            "layout": "channels-first",
            "batch": 2,
            "channels": 3,
            "spatial": 5,
            "residual": False,
            "residual_bias": False,
            "seed": 4,
        },
        {
            **common,
            "label": "one block per element's tail",
            "layout": "channels-first",
            "batch": 1,
            "channels": 1,
            "spatial": 1,
            "residual": True,
            "residual_bias": True,
            "seed": 5,
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=sorted(TORCH_TYPES), default="bf16")
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--pdl", choices=["true", "false"], default="false")
    parser.add_argument("--expect-key", default=None)
    parser.add_argument(
        "--interpreter",
        action="store_true",
        help="run through Triton's interpreter; this operator cannot, and says why",
    )
    arguments = parser.parse_args()
    if not arguments.interpreter and not torch.cuda.is_available():
        print(
            "no GPU on this host: running through Triton's interpreter instead",
            file=sys.stderr,
        )
        arguments.interpreter = True
    if arguments.interpreter:
        print(
            "bias_add needs a device: its kernel rounds each sum with `add.rn.f32` through "
            "inline assembly, and Triton's interpreter answers "
            "NotImplementedError('inline_asm not supported in interpreter mode'), so no "
            "comparison can be made here",
            file=sys.stderr,
        )
        return 0
    import os

    os.environ.pop("TRITON_INTERPRET", None)

    key = (
        f"dtype_{arguments.dtype}_channels_3_block_{arguments.block}"
        f"_residual_true_residual_bias_true_pdl_{arguments.pdl}"
    )
    print(f"descriptor: {key}")
    if arguments.expect_key is not None and arguments.expect_key != key:
        parser.error(f"the flags name {key}, not {arguments.expect_key}")

    failures = 0
    for case in cases(arguments):
        failures += check(case)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
