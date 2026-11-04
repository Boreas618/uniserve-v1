#!/usr/bin/env python3
"""Run the `moe_unpermute` module on a device and check the combine it writes.

    crates/program-server/python/.venv/bin/python combine_check.py <module_dir> [--tokens N]

`<module_dir>` is what `resolve` hands a runner: the directory holding `module.so`. The module is
compiled for one descriptor of the fixture —
`hidden_2048_top_8_experts_128_input_bf16_scales_f32_output_bf16` — so every case runs those shapes
and varies only the token count.

The call is the operator's `moe::combine::Call` (600 bytes, `error` at 88), and the operator's
arithmetic is one line:

```text
out[t] = Σ_{k visible} scales[t, k] · routes[expanded_to_permuted[t, k]]
```

in FP32 **in route order**, rounded once to the output dtype; a route whose map entry is `-1`
contributes nothing, which is how an expert-parallel rank states that a route belongs to another
rank. The check therefore covers the two things a combine can get wrong while every product is
right:

- **a `-1` route beside visible ones**, where a kernel that read the route anyway would add another
  rank's product, and a kernel that treated `-1` as row zero would add the first row's;
- **a token whose every route is `-1`**, whose output row must be **zeros** — the sum of nothing, not
  an untouched NaN and not the scales' product with a stray row;
- **`-1` in the middle and at the end of the route list**, since the skip must not shift the order of
  the routes after it — the operator fixes the order so repeated calls agree bit for bit.

The reference accumulates in the operator's order and rounds once; rows are compared with
`Tolerance::dot(top, f32, output)`, the operator's own bound for a `top`-term reduction.
"""

import argparse
import ctypes
import math
import pathlib
import sys

import torch

DTYPES = {"bf16": (4, torch.bfloat16), "f16": (3, torch.float16), "f32": (5, torch.float32)}


class Call(ctypes.Structure):
    """`moe::combine::Call`, field for field."""

    _fields_ = [
        ("routes", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("expanded_to_permuted", ctypes.c_void_p),
        ("scales", ctypes.c_void_p),
        ("tokens", ctypes.c_int64),
        ("top", ctypes.c_int64),
        ("hidden", ctypes.c_int64),
        ("routes_rows", ctypes.c_int64),
        ("input_dtype", ctypes.c_int64),
        ("scales_dtype", ctypes.c_int64),
        ("output_dtype", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


#: `crates/ops/src/moe/combine.rs`, `UNPERMUTE_CALL_LAYOUT`'s size, which `entry.h` also pins.
#: Measured rather than remembered: a field added to the operator's struct shifts every one after it.
UNPERMUTE_CALL_SIZE = 600
assert ctypes.sizeof(Call) == UNPERMUTE_CALL_SIZE, (
    f"the check's Call mirror is {ctypes.sizeof(Call)} bytes, the operator's layout says "
    f"{UNPERMUTE_CALL_SIZE} (crates/ops/src/moe/combine.rs)"
)


def gamma(reduction, unit):
    """`Tolerance::gamma`, the operator's own expression."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def bound(top, output_roundoff):
    """`Tolerance::dot(top, f32, output)`: a `top`-term reduction and the output's rounding."""
    return 2.0 * gamma(top, 2.0**-24) * math.sqrt(top) + output_roundoff


def run(entry, stream, *, tokens, top, hidden, input_dtype, scales_dtype, output_dtype, dropped,
        seed=0):
    """One launch and the operator's reference; `dropped` is a set of `(token, route)` to skip."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    in_code, in_element = DTYPES[input_dtype]
    scales_code, scales_element = DTYPES[scales_dtype]
    out_code, out_element = DTYPES[output_dtype]

    routes_rows = tokens * top
    routes = (torch.randn(routes_rows, hidden, dtype=torch.float32, device=device) * 0.5).to(
        in_element
    )
    scales = (torch.randn(tokens, top, dtype=torch.float32, device=device) * 0.5).to(
        scales_element
    )
    # A permutation of the expanded rows, so a kernel that read `t · top + k` instead of the map
    # would read the wrong route on almost every entry.
    permuted = torch.randperm(routes_rows, device=device).to(torch.int32)
    mapping = permuted.reshape(tokens, top).clone()
    for token, route in dropped:
        mapping[token, route] = -1
    out = torch.full((tokens, hidden), float("nan"), dtype=out_element, device=device)
    routes_before, scales_before = routes.clone(), scales.clone()

    call = Call(
        routes=ctypes.c_void_p(routes.data_ptr()),
        out=ctypes.c_void_p(out.data_ptr()),
        expanded_to_permuted=ctypes.c_void_p(mapping.data_ptr()),
        scales=ctypes.c_void_p(scales.data_ptr()),
        tokens=tokens,
        top=top,
        hidden=hidden,
        routes_rows=routes_rows,
        input_dtype=in_code,
        scales_dtype=scales_code,
        output_dtype=out_code,
        error=b"",
    )
    status = entry(ctypes.byref(call), stream)
    if status != 0:
        message = call.error.decode("utf-8", "replace").rstrip("\x00")
        raise SystemExit(f"the module refused the call with status {status}: {message}")

    if not torch.equal(routes, routes_before) or not torch.equal(scales, scales_before):
        raise SystemExit("the module wrote into its inputs; the combine writes only `out`")

    # The operator's sum, in route order, in FP32, rounded once.
    total = torch.zeros(tokens, hidden, dtype=torch.float32, device=device)
    for route in range(top):
        row = mapping[:, route].to(torch.int64)
        visible = row >= 0
        safe = row.clamp_min(0)
        product = routes[safe].to(torch.float32) * scales[:, route].to(torch.float32)[:, None]
        total = total + torch.where(visible[:, None], product, torch.zeros_like(product))
    expected = total.to(out_element)

    got = out.to(torch.float32)
    want = expected.to(torch.float32)
    rounding = 2.0 ** {torch.bfloat16: -8, torch.float16: -11, torch.float32: -24}[out_element]
    relative = bound(top, rounding)
    allowed = relative * want.abs()
    error = (got - want).abs()
    # A token whose every route is dropped must be exactly zero, not merely within the bound.
    empty = (mapping < 0).all(dim=1)
    if bool(empty.any()):
        if bool(got[empty].any()):
            raise SystemExit(
                f"a token whose every route is -1 wrote non-zero output: "
                f"{float(got[empty].abs().max()):.3e}"
            )
    # A difference that is not a number is never inside a bound: `>` would say it is.
    violations = int((~(error[~empty] <= allowed[~empty])).sum())
    print(
        f"{input_dtype}/{scales_dtype}->{output_dtype} tokens {tokens} top {top} hidden {hidden}: "
        f"{len(dropped)} dropped routes, {int(empty.sum())} tokens with no route, "
        f"max |out - reference| = {float(error[~empty].max()) if bool((~empty).any()) else 0.0:.6g}, "
        f"bound rtol {relative:.3e}"
    )
    if violations:
        position = int((~(error[~empty] <= allowed[~empty])).reshape(-1).nonzero()[0])
        raise SystemExit(
            f"{violations} of {int((~empty).sum()) * hidden} combined values are outside the bound; "
            f"first at {position}"
        )
    return int((~empty).sum()) * hidden


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", type=pathlib.Path, help="the module directory resolve hands a runner")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--input", default="bf16", choices=tuple(DTYPES))
    parser.add_argument("--scales", default="f32", choices=tuple(DTYPES))
    parser.add_argument("--output", default="bf16", choices=tuple(DTYPES))
    options = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "this module is a compiled native library: it needs the device it was built for"
        )
    library = ctypes.CDLL(str(options.module / "module.so"))
    entry = library.uniserve_moe_unpermute
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    tokens, top, hidden = options.tokens, options.top, options.hidden
    shapes = dict(
        tokens=tokens, top=top, hidden=hidden, input_dtype=options.input,
        scales_dtype=options.scales, output_dtype=options.output,
    )
    checked = 0
    # A rank that holds every route but one per token, and one token that holds none.
    dropped = {(token, top - 1) for token in range(tokens)}
    dropped |= {(token, 0) for token in range(0, tokens, 3)}
    if tokens:
        dropped |= {(0, route) for route in range(top)}
    checked += run(entry, stream, dropped=dropped, **shapes)
    # Nothing dropped: the plain combine, which a map-reading kernel and an index-arithmetic kernel
    # both pass, so it is the case that says the bound itself is sane.
    checked += run(entry, stream, dropped=set(), **shapes)
    # Only the middle routes dropped, so the skip cannot fall at the ends of the loop.
    checked += run(
        entry, stream, dropped={(token, route) for token in range(tokens)
                                for route in range(1, top - 1, 2)}, **shapes
    )
    print(f"ok: {checked} combined values over three route patterns, within dot(top)'s bound")
    return 0


if __name__ == "__main__":
    sys.exit(main())
