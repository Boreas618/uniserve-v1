#!/usr/bin/env python3
"""Run vsa_compress_pooled's kernel and compare what it wrote with the operator's composition.

The operator is one softmax over the key tiles, weighted into the pooled per-head means, narrowed
once into the attention's row type. The reference is the composition the design states, written
here in torch from the same two lines the kernel computes:

```text
weights[head, i, j] = softmax_j(scores[head, i, j])     over the tiles that have live keys
out[head, i, w]     = Σ_j weights[head, i, j] · values[j, head, w]
```

The cases on Triton's interpreter, each a property the already-observed defects of this family
would break:

* **the fixture's cell** — every tile live, the comparison exact in FP32: a program that indexed the
  scores `[heads, query tiles, tiles]` and the values `[tiles, heads, width]` with the head and the
  query tile swapped would still pass a symmetric case, so the scores are an asymmetric ramp;
* **an empty tile** — `sizes[j] == 0` excludes tile `j` from the softmax and renormalizes the rest;
  a kernel that weighted it with its raw score reads a pooled row nothing wrote;
* **a partial last tile** — `sizes` smaller than the tile's capacity changes no weight (the pooling
  already averaged the live keys), so the run must equal the same case with a full size: this checks
  the kernel does *not* renormalize by key count;
* **`width < block`** — the lane mask, on a width that is not a power of two;
* **BF16 output** — the store narrows once, and the comparison is the operator's own tolerance
  (one ulp of the row type), because Triton's interpreter truncates where a device rounds.

The interpreter's `tl.dot` is an FP32 product whatever the `input_precision`, so the weighted sum's
own precision, three TF32 passes on the tensor cores, shows only on a device: `--device` runs the
module's extents (64 key tiles a step, at least sixteen lanes) over many live tiles, some empty,
with positive values, so that no output cancels and each is compared against its own magnitude at
the operator's bound, `Tolerance::split_dot(n, tf32x3(n), f32, out)`.

    python vsa_compress_pooled_check.py [--package DIR]             # Triton's interpreter
    python vsa_compress_pooled_check.py --device [--package DIR]    # a device
"""

import argparse
import importlib.util
import os
import pathlib
import sys

#: Triton reads `TRITON_INTERPRET` when it is imported, so the mode is chosen from the command line
#: before the kernel is loaded: `--device` compiles for the device, anything else interprets.
DEVICE = "--device" in sys.argv
if not DEVICE:
    os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402  (after TRITON_INTERPRET, which Triton reads at import)

#: The kernel packages' root: this check lives in `tests/python/kernels/<family>/`.
KERNELS = (
    pathlib.Path(__file__).resolve().parents[4] / "crates" / "ops" / "kernels"
)
#: The package this check covers.
PACKAGE = KERNELS / "vsa_compress_pooled" / "triton"

#: The interpreter's extents: a few tiles, a width that is not a power of two.
INTERPRETED = {
    "heads": 2,
    "width": 6,
    "query_tiles": 2,
    "tiles": 3,
    "max_tiles": 4,
    "keys": 4,
    "block": 8,
}
#: A device's extents: the module's 64 key tiles a step over a tile count that fills no step, a
#: width under its lanes, and query tiles that fill one program's sixteen and part of another's.
ON_DEVICE = {
    "heads": 2,
    "width": 96,
    "query_tiles": 20,
    "tiles": 150,
    "max_tiles": 1024,
    "keys": 64,
    "block": 128,
}


def load_kernel(package):
    """The package's kernel, imported from its source directory."""
    del (
        package
    )  # the check's own location is the package's, wherever it is run from
    path = PACKAGE / "src" / "vsa_compress_pooled.py"
    spec = importlib.util.spec_from_file_location(
        "compress_pooled_kernel", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._vsa_compress_pooled_kernel


def tensors(shape, sizes, dtype=torch.float32, seed=7, positive=False):
    """The two operands and the tile sizes, in the layouts the operator states."""
    heads, query_tiles, tiles, width = (
        shape[name] for name in ("heads", "query_tiles", "tiles", "width")
    )
    generator = torch.Generator().manual_seed(seed)
    if positive:
        # Scores that spread the weight over many tiles, and values that sum without cancelling.
        scores = (
            torch.rand((heads, query_tiles, tiles), generator=generator) * 4.0
        )
        values = torch.rand((tiles, heads, width), generator=generator) + 0.5
    else:
        # An asymmetric ramp: a head/query swap changes the answer.
        scores = torch.arange(
            heads * query_tiles * tiles, dtype=torch.float32
        ).reshape(heads, query_tiles, tiles)
        scores = (scores - 5.0) / 3.0
        scores = scores + torch.rand(scores.shape, generator=generator) * 0.25
        values = (
            torch.rand((tiles, heads, width), generator=generator) * 2.0 - 1.0
        )
    sizes = torch.tensor(sizes, dtype=torch.int32)
    out = torch.zeros((heads, query_tiles, width), dtype=dtype)
    return scores, values, sizes, out


def reference(scores, values, sizes):
    """The composition the operator states, in torch's FP32, before the store's rounding."""
    weights = scores.clone()
    weights[:, :, sizes == 0] = float("-inf")
    peak = weights.max(dim=-1, keepdim=True).values
    peak = torch.where(torch.isinf(peak), torch.zeros_like(peak), peak)
    exponent = torch.exp(weights - peak)
    total = exponent.sum(dim=-1, keepdim=True)
    total = torch.where(total == 0.0, torch.ones_like(total), total)
    weights = exponent / total
    return torch.einsum("hqt,thw->hqw", weights, values)


def launch(kernel, shape, sizes, dtype=torch.float32, device="cpu"):
    """Launch one compression and answer what it wrote beside the operands."""
    scores, values, sizes_tensor, out = tensors(
        shape, sizes, dtype, positive=device != "cpu"
    )
    out = out.to(device)
    kernel[((shape["query_tiles"] + 15) // 16, shape["heads"], 1)](
        scores.to(device),
        values.to(device),
        sizes_tensor.to(device),
        out,
        shape["query_tiles"],
        shape["tiles"],
        # The dependent-launch word is an i32 the operator packs, not a Python bool: the
        # interpreter would infer int1 from `False` and refuse the argument.
        0,
        HEADS=shape["heads"],
        WIDTH=shape["width"],
        QUERIES=16,
        KEYS=shape["keys"],
        BLOCK=shape["block"],
        PDL=False,
    )
    return scores, values, sizes_tensor, out.cpu()


#: The unit roundoff of the accumulation's type and of the row type the store narrows into.
UNIT_ROUNDOFF = 2.0**-24
# A BF16 significand is eight bits, so a value just above 0.25 steps by 2^-9, one step of 2^-7
# relative to the bottom of its binade. The store is compared with the unrounded reference, so it
# may move a value by the half step a round-to-nearest store makes on a device, and by the FULL
# step on Triton's interpreter, which truncates a BF16 store where a device rounds
# (cache_copy_region's check found the same).
BF16_ROUNDOFF = 2.0**-8 if DEVICE else 2.0**-7


def gamma(reduction, unit=UNIT_ROUNDOFF):
    """``γₙ = n·u / (1 − n·u)``."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def tf32x3(reduction):
    """``ops::tolerance::tf32x3``: the roundings the three-pass product costs beyond FP32's."""
    return 21 + -(-3 * reduction // 1024)


def declared_bound(reduction, output_roundoff):
    """``VsaCompressPooled::tolerance``: ``split_dot(n, tf32x3(n), f32, output)``'s ``rtol``."""
    return (
        2.0 * gamma(reduction + tf32x3(reduction)) * reduction**0.5
        + output_roundoff
    )


def compare(
    name,
    kernel,
    shape,
    sizes,
    dtype=torch.float32,
    output_roundoff=UNIT_ROUNDOFF,
    device="cpu",
):
    """Compare one case against the reference within the operator's declared bound."""
    scores, values, sizes_tensor, out = launch(
        kernel, shape, sizes, dtype, device
    )
    expected = reference(scores, values, sizes_tensor)
    got = out.to(torch.float32)
    # The operator's reduction length is the longer of its two: the tile softmax's bound and the
    # width of the weighted sum (and here the compiled tile block, which is what it reduces in).
    bound = declared_bound(
        max(shape["max_tiles"], shape["width"]), output_roundoff
    )
    difference = (got - expected).abs()
    close = difference <= bound * expected.abs()
    if bool(close.all()):
        return []
    differing = (~close).nonzero()
    head, query, feature = differing[0].tolist()
    return [
        f"{name}: {len(differing)} of {got.numel()} values differ; first at head {head}, query "
        f"{query}, feature {feature}: got {got[head, query, feature].item():.9g}, expected "
        f"{expected[head, query, feature].item():.9g}"
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=".", help="the package directory")
    parser.add_argument(
        "--device", action="store_true", help="run on a CUDA device"
    )
    arguments = parser.parse_args()
    kernel = load_kernel(arguments.package)

    problems = []
    if arguments.device:
        shape = ON_DEVICE
        # Every seventh tile empty, the last one partial.
        sizes = [0 if j % 7 == 3 else 4 for j in range(shape["tiles"] - 1)]
        sizes.append(2)
        problems += compare("many tiles", kernel, shape, sizes, device="cuda")
        problems += compare(
            "many tiles, bf16 rows",
            kernel,
            shape,
            sizes,
            torch.bfloat16,
            BF16_ROUNDOFF,
            device="cuda",
        )
    else:
        shape = INTERPRETED
        problems += compare("every tile live", kernel, shape, [4, 4, 4])
        problems += compare("an empty tile", kernel, shape, [4, 0, 4])
        problems += compare("the last tile empty", kernel, shape, [4, 4, 0])
        problems += compare("every tile empty", kernel, shape, [0, 0, 0])
        problems += compare("a partial last tile", kernel, shape, [4, 4, 2])
        problems += compare("width under block", kernel, shape, [4, 4, 4])
        problems += compare(
            "bf16 rows",
            kernel,
            shape,
            [4, 0, 4],
            torch.bfloat16,
            BF16_ROUNDOFF,
        )

    if problems:
        print(f"{len(problems)} problem(s):")
        for problem in problems:
            print(f"  {problem}")
        raise SystemExit(1)
    print("# every compression matches the reference")


if __name__ == "__main__":
    main()
