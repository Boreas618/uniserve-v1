"""The `moe_unpermute` module's combine against the operator's sum.

The module of the descriptor below (resolved at the session's Build Server for
the session's device) is called with the operator's `moe::combine::Call`
(``crates/ops/src/moe/combine.rs``; the check runs the permuted layout and
states it), and the operator's arithmetic is one line::

    out[t] = Σ_{k visible} scales[t, k] · routes[expanded_to_permuted[t, k]]

in FP32 in route order, rounded once to the output type; a route whose map
entry is `-1` contributes nothing, which is how an expert-parallel rank states
that a route belongs to another rank. The cases cover what a combine can get
wrong while every product is right:

- **a `-1` route beside visible ones**, where a kernel that read the route
  anyway would add another rank's product, and one that treated `-1` as row
  zero would add the first row's;
- **a token whose every route is `-1`**, whose output row must be zeros, the
  sum of nothing, not an untouched NaN;
- **`-1` in the middle of the route list**, since the skip must not shift the
  order of the routes after it;
- **nothing dropped**, the plain combine, which says the bound itself is sane;
- **the rows no map entry names hold NaN**: they are the padding a sort leaves
  behind, and a kernel that indexed `t · top + k` in place of the map, or fell
  back to row zero for a `-1` route, carries the NaN into the output.

The reference accumulates in the operator's order and rounds once; rows are
compared with `Tolerance::dot(top, f32, output)`, the operator's own bound for
a `top`-term reduction, and the module must write nothing into its inputs.
"""

import ctypes
import math

import pytest
import torch

pytestmark = pytest.mark.unit

#: `ops::abi::code`'s number and torch's type of each element type.
DTYPES = {
    "bf16": (4, torch.bfloat16),
    "f16": (3, torch.float16),
    "f32": (5, torch.float32),
}

#: The unit roundoff of each output type.
ROUNDOFF = {
    torch.bfloat16: 2.0**-8,
    torch.float16: 2.0**-11,
    torch.float32: 2.0**-24,
}

#: The layouts the operator distinguishes: the routes read in permuted order,
#: or in the order the expanded map gives.
PERMUTED, EXPANDED = 0, 1

#: The `static_assert` of the package's entry header that pins the call's size.
SIZE_ASSERTION = "sizeof(uniserve_unpermute_call) =="

#: The module's descriptor and the tokens the cases combine.
HIDDEN, TOP, EXPERTS, INPUT, SCALES, OUTPUT = (
    2048,
    8,
    128,
    "bf16",
    "f32",
    "bf16",
)
TOKENS = 256
KEY = (
    f"layout_permuted_hidden_{HIDDEN}_top_{TOP}_experts_{EXPERTS}_input_{INPUT}"
    f"_scales_{SCALES}_output_{OUTPUT}"
)


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
        ("layout", ctypes.c_int64),
        ("error", ctypes.c_char * 512),
    ]


def test_the_call_mirror_is_the_headers_unpermute_call(kernels):
    """The package's ``entry.h`` pins the size, as
    `crates/ops/src/moe/combine.rs` does on the operator's side; it is read
    rather than restated.
    """  # noqa: D205
    header = kernels / "moe_unpermute" / "trtllm" / "src" / "entry.h"
    text = header.read_text()
    assert SIZE_ASSERTION in text, f"{header}: no `{SIZE_ASSERTION}` assertion"
    declared = int(
        text.split(SIZE_ASSERTION, 1)[1].lstrip(" =").split()[0].rstrip(",")
    )
    assert ctypes.sizeof(Call) == declared, (
        f"the check's Call mirror is {ctypes.sizeof(Call)} bytes, the "
        f"package's header says {declared}"
    )


def gamma(reduction, unit):
    """`Tolerance::gamma`, the operator's own expression."""
    terms = reduction * unit
    return 1.0 if terms >= 1.0 else terms / (1.0 - terms)


def bound(top, output_roundoff):
    """`Tolerance::dot(top, f32, output)`: a `top`-term reduction and the
    output's rounding.
    """  # noqa: D205
    return 2.0 * gamma(top, 2.0**-24) * math.sqrt(top) + output_roundoff


#: The route patterns, each a set of `(token, route)` the map drops, and
#: whether the rows no map entry names hold NaN.
CASES = {
    # A rank that holds every route but one per token, and one token that holds
    # none.
    "dropped routes and a token with none": (
        {(token, TOP - 1) for token in range(TOKENS)}
        | {(token, 0) for token in range(0, TOKENS, 3)}
        | {(0, route) for route in range(TOP)},
        False,
    ),
    "nothing dropped": (set(), False),
    # Only the middle routes dropped, so the skip cannot fall at the ends of
    # the loop.
    "middle routes dropped": (
        {
            (token, route)
            for token in range(TOKENS)
            for route in range(1, TOP - 1, 2)
        },
        False,
    ),
    "unnamed rows poisoned": (
        {(token, TOP - 1) for token in range(TOKENS)},
        True,
    ),
}


@pytest.mark.gpu
@pytest.mark.parametrize("case", CASES)
def test_the_combine_is_the_operators_sum_in_route_order(
    case, torch_device, kernel_module
):
    dropped, poison_unused = CASES[case]
    library = ctypes.CDLL(
        str(kernel_module("moe_unpermute", "trtllm", KEY) / "module.so")
    )
    entry = library.uniserve_moe_unpermute
    entry.argtypes = [ctypes.POINTER(Call), ctypes.c_void_p]
    entry.restype = ctypes.c_int32

    torch.manual_seed(0)
    in_code, in_element = DTYPES[INPUT]
    scales_code, scales_element = DTYPES[SCALES]
    out_code, out_element = DTYPES[OUTPUT]
    routes_rows = TOKENS * TOP
    routes = (
        torch.randn(
            routes_rows, HIDDEN, dtype=torch.float32, device=torch_device
        )
        * 0.5
    ).to(in_element)
    scales = (
        torch.randn(TOKENS, TOP, dtype=torch.float32, device=torch_device) * 0.5
    ).to(scales_element)
    # A permutation of the expanded rows, so a kernel that read `t · top + k`
    # instead of the map would read the wrong route on almost every entry.
    permuted = torch.randperm(routes_rows, device=torch_device).to(torch.int32)
    mapping = permuted.reshape(TOKENS, TOP).clone()
    for token, route in dropped:
        mapping[token, route] = -1
    if poison_unused:
        named = torch.zeros(routes_rows, dtype=torch.bool, device=torch_device)
        named[mapping[mapping >= 0].to(torch.int64)] = True
        assert bool((~named).any()), (
            "the padding case needs a row no map entry names: drop a route so "
            "its permuted row is left behind"
        )
        routes[~named] = float("nan")
    out = torch.full(
        (TOKENS, HIDDEN), float("nan"), dtype=out_element, device=torch_device
    )
    routes_before, scales_before = routes.clone(), scales.clone()

    call = Call(
        routes=routes.data_ptr(),
        out=out.data_ptr(),
        expanded_to_permuted=mapping.data_ptr(),
        scales=scales.data_ptr(),
        tokens=TOKENS,
        top=TOP,
        hidden=HIDDEN,
        routes_rows=routes_rows,
        input_dtype=in_code,
        scales_dtype=scales_code,
        output_dtype=out_code,
        layout=PERMUTED,
        error=b"",
    )
    stream = torch.cuda.current_stream(torch_device).cuda_stream
    status = entry(ctypes.byref(call), ctypes.c_void_p(stream))
    torch.cuda.synchronize(torch_device)
    message = call.error.decode("utf-8", "replace").rstrip("\x00")
    assert status == 0, f"the module refused the call ({status}): {message}"
    # Compared as bytes: the poisoned rows are NaN, which no value equals.
    assert torch.equal(
        routes.view(torch.uint8), routes_before.view(torch.uint8)
    ) and torch.equal(scales, scales_before), (
        "the module wrote into its inputs; the combine writes only `out`"
    )

    # The operator's sum, in route order, in FP32, rounded once.
    total = torch.zeros(
        TOKENS, HIDDEN, dtype=torch.float32, device=torch_device
    )
    for route in range(TOP):
        row = mapping[:, route].to(torch.int64)
        visible = row >= 0
        product = (
            routes[row.clamp_min(0)].float() * scales[:, route].float()[:, None]
        )
        total = total + torch.where(
            visible[:, None], product, torch.zeros_like(product)
        )
    want = total.to(out_element).float()
    got = out.float()

    # A token whose every route is dropped must be exactly zero, not merely
    # within the bound.
    empty = (mapping < 0).all(dim=1)
    assert not bool(got[empty].any()), (
        f"a token whose every route is -1 wrote non-zero output: "
        f"{float(got[empty].abs().max()):.3e}"
    )
    if poison_unused:
        # A NaN anywhere in the output came from a row the map does not name.
        carried = int(torch.isnan(got).sum())
        assert carried == 0, (
            f"{carried} combined values are not a number: a NaN from the "
            f"{int((~named).sum())} padding rows reached the output"
        )
    relative = bound(TOP, ROUNDOFF[out_element])
    error = (got - want).abs()[~empty]
    # A difference that is not a number is never inside a bound.
    outside = ~(error <= relative * want.abs()[~empty])
    position = int(outside.reshape(-1).nonzero()[0]) if outside.any() else 0
    assert not bool(outside.any()), (
        f"{int(outside.sum())} of {error.numel()} combined values are outside "
        f"rtol {relative:.3e}; max |out - reference| = "
        f"{float(torch.nan_to_num(error).max()):.6g}; first at {position}"
    )
