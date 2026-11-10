#!/usr/bin/env python3
"""Run vsa_compress_pooled's kernel under Triton's interpreter and compare what it wrote.

The operator is one softmax over the key tiles, weighted into the pooled per-head means, narrowed
once into the attention's row type. The reference is the composition the design states, written
here in torch from the same two lines the kernel computes:

```text
weights[head, i, j] = softmax_j(scores[head, i, j])     over the tiles that have live keys
out[head, i, w]     = Σ_j weights[head, i, j] · values[j, head, w]
```

The cases, each a property the already-observed defects of this family would break:

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

    TRITON_INTERPRET=1 python compress_pooled_check.py [--package DIR]
"""

import argparse
import importlib.util
import pathlib

import torch

HEADS = 2
WIDTH = 6
QUERY_TILES = 2
TILES = 3
MAX_TILES = 4
BLOCK = 8


def load_kernel(package):
    """The package's kernel, imported from its source directory."""
    del package  # the check's own location is the package's, wherever it is run from
    path = pathlib.Path(__file__).resolve().parents[1] / "src" / "vsa_compress_pooled.py"
    spec = importlib.util.spec_from_file_location("compress_pooled_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._vsa_compress_pooled_kernel


def tensors(sizes, dtype=torch.float32, seed=7):
    """The two operands and the tile sizes, in the layouts the operator states."""
    generator = torch.Generator().manual_seed(seed)
    # An asymmetric ramp: a head/query swap changes the answer.
    scores = torch.arange(HEADS * QUERY_TILES * TILES, dtype=torch.float32).reshape(
        HEADS, QUERY_TILES, TILES
    )
    scores = (scores - 5.0) / 3.0
    scores = scores + torch.rand(scores.shape, generator=generator) * 0.25
    values = torch.rand((TILES, HEADS, WIDTH), generator=generator) * 2.0 - 1.0
    sizes = torch.tensor(sizes, dtype=torch.int32)
    out = torch.zeros((HEADS, QUERY_TILES, WIDTH), dtype=dtype)
    return scores, values, sizes, out


def reference(scores, values, sizes, dtype):
    """The composition the operator states, in torch."""
    weights = scores.clone()
    weights[:, :, sizes == 0] = float("-inf")
    peak = weights.max(dim=-1, keepdim=True).values
    peak = torch.where(torch.isinf(peak), torch.zeros_like(peak), peak)
    exponent = torch.exp(weights - peak)
    total = exponent.sum(dim=-1, keepdim=True)
    total = torch.where(total == 0.0, torch.ones_like(total), total)
    weights = exponent / total
    out = torch.einsum("hqt,thw->hqw", weights, values)
    return out.to(dtype)


def launch(kernel, sizes, dtype=torch.float32):
    """Launch one compression and answer what it wrote beside the operands."""
    scores, values, sizes_tensor, out = tensors(sizes, dtype)
    kernel[(1, HEADS, 1)](
        scores,
        values,
        sizes_tensor,
        out,
        QUERY_TILES,
        TILES,
        # The dependent-launch word is an i32 the operator packs, not a Python bool: the
        # interpreter would infer int1 from `False` and refuse the argument.
        0,
        HEADS=HEADS,
        WIDTH=WIDTH,
        QUERIES=16,
        KEYS=4,
        BLOCK=BLOCK,
    )
    return scores, values, sizes_tensor, out


#: The unit roundoff of the accumulation's type and of the row type the store narrows into.
UNIT_ROUNDOFF = 2.0**-24
# A BF16 significand is eight bits, so a value just above 0.25 steps by 2^-9, one step of 2^-7
# relative to the bottom of its binade. The case allows the FULL step rather than the half a
# round-to-nearest store needs: Triton's interpreter truncates a BF16 store where a device
# rounds (cache_copy_region's check found the same), so the device answer lies inside this
# bound and the case still fails if the arithmetic is wrong by more than the row type carries.
BF16_ROUNDOFF = 2.0**-7


def gamma(reduction, unit=UNIT_ROUNDOFF):
    """``γₙ = n·u / (1 − n·u)``."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def declared_bound(reduction, output_roundoff):
    """``Tolerance::dot(reduction, f32, output)``'s ``rtol``, as ``segments.rs`` declares it."""
    return 2.0 * gamma(reduction) * reduction**0.5 + output_roundoff


def compare(name, kernel, sizes, dtype=torch.float32, output_roundoff=UNIT_ROUNDOFF):
    """Compare one case against the reference within the operator's declared bound."""
    scores, values, sizes_tensor, out = launch(kernel, sizes, dtype)
    expected = reference(scores, values, sizes_tensor, dtype).to(torch.float32)
    got = out.to(torch.float32)
    # The operator's reduction length is the longer of its two: the tile softmax's bound and the
    # width of the weighted sum (and here the compiled tile block, which is what it reduces in).
    bound = declared_bound(max(MAX_TILES, WIDTH), output_roundoff)
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
    arguments = parser.parse_args()
    kernel = load_kernel(arguments.package)

    problems = []
    problems += compare("every tile live", kernel, [4, 4, 4])
    problems += compare("an empty tile", kernel, [4, 0, 4])
    problems += compare("the last tile empty", kernel, [4, 4, 0])
    problems += compare("every tile empty", kernel, [0, 0, 0])
    problems += compare("a partial last tile", kernel, [4, 4, 2])
    problems += compare("width under block", kernel, [4, 4, 4])
    problems += compare("bf16 rows", kernel, [4, 0, 4], torch.bfloat16, BF16_ROUNDOFF)

    if problems:
        print(f"{len(problems)} problem(s):")
        for problem in problems:
            print(f"  {problem}")
        raise SystemExit(1)
    print("# every compression matches the reference")


if __name__ == "__main__":
    main()
