"""The latent_attention program against the operator's formula.

The operator (``ops::attention::LatentAttention``) attends one shared latent,
both key and value, over each query's window entries and the compressed
entries its selection names, in one softmax with a per-head sink::

    entries(t) = window positions max(0, p - W + 1) ..= p, then the
                 compressed entries selection[t, :] names (>= 0)
    s          = scale * q . c
    m          = max(-1e30, max s)
    o          = sum e^(s - m) c / (sum e^(s - m) + e^(sink - m))

The operands, the reference and the bound are the ones every check of this
operator shares (:mod:`uniserve_testing.kernels.attention.latent`): the
formula in FP64 over the latents the records decode to, read from each
request's logical entries rather than through the pages, and the operator's
terms with each element's own magnitude in place of the declared latent
envelope.

The cases, each a property a defect of this family breaks:

- **every record format** of each side: a BF16 and an MXFP8 window; BF16,
  MXFP8, NVFP4 and MXFP4 compressed entries;
- **a window-only layer**, compiled without the compressed side;
- **windows that start at zero**: contexts shorter than the window;
- **windows held from a row's start**, as a request resumed from a prefix hit
  holds them: entries before the start are never read;
- **a ring of window pages** narrower than the request, read modulo its width;
- **negative selection slots** among the named ones, and rows that see no
  compressed entry yet;
- **a padding token** (position -1, an empty selection), which reads no entry
  and attends its sink alone: zeros;
- **64 heads in blocks of 16**, and **8 heads**, a block below the sixteen rows
  a tensor-core product needs;
- **decode-like batches** of a few tokens per request, and **a prefill-like
  row** of many tokens of one request, each attending its own causal window.

and, at the model's shapes (512-wide latents, 64 heads, 128 window and 512
selected entries), a decode of 48 requests of 1 to 6 tokens and a prefill
chunk of 512 tokens.

Both of the program's products run on the tensor cores over BF16 operands,
which Triton's interpreter multiplies as the integers it stores them in; so
every case needs a device.
"""

import pytest
import torch
from uniserve_testing.kernels.attention.latent import (
    LATENT_ENVELOPE,
    Case,
    batch,
    bound,
    key,
    reference,
)

pytestmark = pytest.mark.unit


#: Small shapes, where every edge a row meets fits a few tiles: a latent of
#: 64, a window of 16 in pages of 8, compressed pages of 4 and selections of
#: 8.
SMALL = {"head_dim": 64, "window": 16, "page": 8, "entries": 4, "columns": 8}

CASES = {
    "a window-only layer over BF16 windows that start at zero": Case(
        heads=16,
        block=16,
        window_format="bf16",
        compressed=None,
        requests=((5, 1), (12, 2), (30, 3)),
        padding=True,
        seed=1,
        **SMALL,
    ),
    "an MXFP8 window over a ring of pages narrower than the requests": Case(
        heads=16,
        block=16,
        window_format="mxfp8",
        compressed=None,
        requests=((41, 1), (57, 2), (33, 1), (64, 3)),
        ring=True,
        seed=2,
        **SMALL,
    ),
    "an MXFP8 window and BF16 entries with negative selection slots": Case(
        heads=16,
        block=16,
        window_format="mxfp8",
        compressed="bf16",
        requests=((3, 1), (26, 2), (40, 1)),
        padding=True,
        seed=3,
        **SMALL,
    ),
    "windows held from a replay's start, read from it on": Case(
        heads=16,
        block=16,
        window_format="mxfp8",
        compressed="nvfp4",
        requests=((40, 4), (20, 2), (12, 1)),
        starts=(34, 0, 12),
        seed=8,
        **SMALL,
    ),
    "MXFP8 entries over 64 heads in blocks of 16": Case(
        heads=64,
        block=16,
        window_format="mxfp8",
        compressed="mxfp8",
        requests=((22, 2), (35, 1)),
        seed=4,
        **SMALL,
    ),
    "NVFP4 entries and a prefill row of many tokens of one request": Case(
        heads=16,
        block=16,
        window_format="mxfp8",
        compressed="nvfp4",
        requests=((48, 40),),
        seed=5,
        **SMALL,
    ),
    "NVFP4 entries over 8 heads, a block below a product's 16 rows": Case(
        heads=8,
        block=8,
        window_format="bf16",
        compressed="nvfp4",
        requests=((19, 3), (44, 2)),
        seed=6,
        **SMALL,
    ),
    "MXFP4 entries under a BF16 window": Case(
        heads=16,
        block=16,
        window_format="bf16",
        compressed="mxfp4",
        requests=((31, 2), (52, 1)),
        seed=7,
        **SMALL,
    ),
}

#: The model's shapes: 64 heads of 512 in blocks of 16, a window of 128 in
#: pages of 64, and 512 selected entries.
MODEL = {"heads": 64, "block": 16, "head_dim": 512, "window": 128, "page": 64}

MODEL_CASES = {
    "the model's decode of 1 to 6 tokens per request over NVFP4 entries": Case(
        window_format="mxfp8",
        compressed="nvfp4",
        entries=32,
        columns=512,
        requests=tuple(
            (600 + 47 * index, 1 + index % 6) for index in range(48)
        ),
        seed=11,
        **MODEL,
    ),
    "the model's prefill chunk of 512 tokens over MXFP8 entries": Case(
        window_format="mxfp8",
        compressed="mxfp8",
        entries=64,
        columns=512,
        requests=((2048, 512),),
        seed=12,
        **MODEL,
    ),
    "the model's window-only layer, decode over a BF16 window": Case(
        window_format="bf16",
        compressed=None,
        entries=1,
        columns=1,
        requests=tuple(
            (130 + 61 * index, 1 + index % 4) for index in range(32)
        ),
        seed=13,
        **MODEL,
    ),
}


def launch(stated, case, operands, device):
    """One launch over every head block and token, as the package states it;
    a window-only layer passes placeholder compressed views, which its
    program never reads, and a table width of zero, as the operator does.
    """  # noqa: D205
    tokens = operands.query.shape[0]
    out = torch.full_like(operands.query, float("nan"), device=device)
    if operands.compressed is None:
        compressed = torch.zeros(1, dtype=torch.uint8)
        table, width = torch.zeros((1, 1), dtype=torch.int32), 0
    else:
        compressed = operands.compressed.pages
        table = operands.compressed.table
        width = table.shape[1]
    stated.kernel[(case.heads // case.block, tokens, 1)](
        operands.query.to(device),
        operands.window.pages.to(device),
        operands.window.table.to(device),
        operands.starts.to(device),
        compressed.to(device),
        table.to(device),
        operands.selection.to(device),
        operands.rows.to(device),
        operands.positions.to(device),
        operands.sink.to(device),
        out,
        tokens,
        operands.window.table.shape[1],
        width,
        operands.scale,
        **stated.constexprs,
        **stated.options,
    )
    return out.cpu()


def check(case, stated, device, interpreted):
    """Launch ``case`` and hold the output to the reference within the
    operator's bound.
    """  # noqa: D205
    operands = batch(case)
    out = launch(stated, case, operands, device).double()
    expected, magnitude = reference(case, operands)
    # The drawn latents are within the operator's declared envelope, so the
    # bound below is within its data-free one.
    assert float(magnitude.max()) <= LATENT_ENVELOPE

    allowed = bound(case, magnitude, expected, interpreted)
    error = (out - expected).abs()
    worst = int(torch.argmax(error - allowed))
    assert torch.all(error <= allowed), (
        f"{key(case)}: |out - reference| = {error.flatten()[worst]:.3e} at "
        f"element {worst} exceeds its bound {allowed.flatten()[worst]:.3e} "
        f"(reference {expected.flatten()[worst]:.3e}, magnitude "
        f"{magnitude.flatten()[worst]:.3e})"
    )
    if case.padding:
        assert torch.equal(out[-1], torch.zeros_like(out[-1])), (
            "a token that reads no entry attends its sink alone: zeros"
        )


@pytest.mark.gpu
@pytest.mark.parametrize("name", CASES)
def test_every_query_attends_its_window_and_its_selection_with_its_sink(
    name, torch_device, interpreted, stated_launch, launch_target
):
    case = CASES[name]
    stated = stated_launch("latent_attention", key(case), launch_target)
    check(case, stated, torch_device, interpreted)


@pytest.mark.gpu
@pytest.mark.parametrize("name", MODEL_CASES)
def test_the_model_shapes_attend_within_the_bound(
    name, torch_device, interpreted, stated_launch, launch_target
):
    case = MODEL_CASES[name]
    stated = stated_launch("latent_attention", key(case), launch_target)
    check(case, stated, torch_device, interpreted)
