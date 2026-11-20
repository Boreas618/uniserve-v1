"""The FlashMLA latent attention module against the operator's formula.

The module (``module.so``, resolved at the session's Build Server for the
session's device) is called with the operator's call structure
(``ops::attention::latent``, :class:`uniserve_testing.kernels.attention.latent.
LatentAttentionCall`) on a stream and while a stream captures, as the engine's
worker calls it, and its output is compared with the operator's formula in
FP64 over the latents the records decode to, inside the operator's own bound
(:mod:`uniserve_testing.kernels.attention.latent`).

The module reads the entries of a query by their slot in the planes and
writes those slots itself, from the page tables the call passes, so the cases
are the addressing a defect of that step breaks and the shapes the deployment
serves:

- **the model's decode** of 48 requests of 1 to 6 tokens over NVFP4 compressed
  entries, the rows of a decode step and of a speculative verification;
- **a prefill chunk** of 512 tokens of one request, each attending its own
  window and its own selection;
- **a window-only layer**, compiled without a compressed side (layers 0 and 1);
- **windows held from a replay's start**, as a request resumed from a prefix
  hit holds them: entries before the start are never read;
- **a ring of window pages** narrower than the requests, read modulo its
  width;
- **negative selection slots** among the named ones and **a padding token**
  (position -1, an empty selection), which reads no entry and attends its sink
  alone: zeros;
- **a context of 1,048,576 compressed entries**, the release's length: each
  selection names entries across the whole table, the oldest and the newest
  among them, and every record no entry names holds a trap value.

A call the module cannot serve is refused by one of the guards its entry
states before it reaches the device, and the refusal must name the condition
it fails: a head count past the 64 rows of the kernel's product, a record
width that is not the format's, and a workspace too short for the call's
tokens. Each refused call is built exactly as the served one is, so the
refusal is the guard's and not a side effect of a malformed call.

The checkpoint's own layers 2 (ratio 2) and 20 (ratio 1) attend as the
checkpoint's reference formulas do: over the reference vectors
``generate_attention.py`` writes from the checkpoint
(``crates/models/tests/it/deepseek_v41/``, into
``artifacts/deepseek_v41/attention-reference``), the module reads the
layer's rotated queries, its window and compressed latents in their records
and the reference's selection, and its output, rotated back, is held to the
reference's attention within the bound of the layer's CPU check
(``crates/models/tests/it/deepseek_v41/attention.rs``): a relative L2 of
``4u`` over the output and five times it on a row, ``u = 2^-8``. Without
the vectors those cases are skipped.

Every test but the call structure's layout needs a device.
"""

import ctypes
import importlib.util
import math

import pytest
import torch
from safetensors import safe_open
from uniserve_testing.kernels import capture, records
from uniserve_testing.kernels.attention.latent import (
    LATENT_ENVELOPE,
    Batch,
    Case,
    LatentAttentionCall,
    batch,
    bound,
    key,
    paged,
    reference,
    workspace_bytes,
)
from uniserve_testing.kernels.packages import assert_abi_layout

pytestmark = pytest.mark.unit

#: The package's backend.
BACKEND = "flashmla"

#: The model's shapes: 64 heads of 512 in blocks of 16, a window of 128 in
#: pages of 64, and 512 selected entries of pages of 64 (layer 20's ratio of
#: one) or 32 (the ratio-two sources).
MODEL = {"heads": 64, "block": 16, "head_dim": 512, "window": 128, "page": 64}

#: The release's context, which the longest case addresses: 1,048,576 tokens,
#: one compressed entry apiece.
CONTEXT = 1 << 20

CASES = {
    "the model's decode of 1 to 6 tokens per request": Case(
        window_format="mxfp8",
        compressed="nvfp4",
        entries=32,
        columns=512,
        requests=tuple(
            (600 + 47 * index, 1 + index % 6) for index in range(48)
        ),
        padding=True,
        seed=21,
        **MODEL,
    ),
    "the model's prefill chunk of 512 tokens": Case(
        window_format="mxfp8",
        compressed="nvfp4",
        entries=64,
        columns=512,
        requests=((2048, 512),),
        seed=22,
        **MODEL,
    ),
    "a window-only layer's decode": Case(
        window_format="mxfp8",
        compressed=None,
        entries=1,
        columns=1,
        requests=tuple(
            (130 + 61 * index, 1 + index % 4) for index in range(32)
        ),
        padding=True,
        seed=23,
        **MODEL,
    ),
    "windows held from a replay's start": Case(
        window_format="mxfp8",
        compressed="nvfp4",
        entries=32,
        columns=512,
        requests=((700, 6), (300, 2), (129, 1)),
        starts=(576, 0, 128),
        seed=24,
        **MODEL,
    ),
    "a ring of window pages narrower than the requests": Case(
        window_format="mxfp8",
        compressed="nvfp4",
        entries=32,
        columns=512,
        requests=((4000, 1), (3001, 3), (1500, 1)),
        ring=True,
        seed=25,
        **MODEL,
    ),
    "a context of 1,048,576 compressed entries": Case(
        window_format="mxfp8",
        compressed="nvfp4",
        entries=64,
        columns=512,
        requests=((CONTEXT, 2),),
        ring=True,
        ratio=1,
        extremes=True,
        seed=26,
        **MODEL,
    ),
}

#: The guards: the field a call perturbs, the value it sets and the words the
#: entry answers with, from `src/attention.cu`.
REFUSALS = [
    pytest.param(
        "heads",
        65,
        "attends 1 to 64 query heads",
        id="65 query heads",
    ),
    pytest.param(
        "window_record",
        264,
        "MXFP8 window records of 528 bytes",
        id="a 264-byte window record",
    ),
    pytest.param(
        "compressed_record",
        528,
        "NVFP4 compressed records of 288 bytes",
        id="a 528-byte compressed record",
    ),
    pytest.param(
        "workspace_bytes",
        1024,
        "and the workspace holds 1024",
        id="a workspace of 1024 bytes",
    ),
]


def entry_of(module):
    """The module's entry, declared as the operator calls it."""
    library = ctypes.CDLL(str(module / "module.so"))
    entry = library.uniserve_latent_attention
    entry.argtypes = [ctypes.POINTER(LatentAttentionCall), ctypes.c_void_p]
    entry.restype = ctypes.c_int32
    return entry


def built(case, operands, device):
    """The call of ``case`` over ``operands`` on ``device``, and the output and
    workspace it writes into.

    Every operand is moved to the device and held by the returned tuple: a
    tensor the call names must outlive it.
    """  # noqa: D205
    tokens = operands.query.shape[0]
    held = {
        "query": operands.query.to(device),
        "out": torch.full_like(operands.query, float("nan"), device=device),
        "window": operands.window.pages.to(device),
        "window_table": operands.window.table.to(device),
        "window_starts": operands.starts.to(device),
        "rows": operands.rows.to(device),
        "positions": operands.positions.to(device),
        "sink": operands.sink.to(device),
        "workspace": torch.zeros(
            workspace_bytes(case, tokens), dtype=torch.uint8, device=device
        ),
    }
    if operands.compressed is not None:
        held["compressed"] = operands.compressed.pages.to(device)
        held["compressed_table"] = operands.compressed.table.to(device)
        held["selection"] = operands.selection.to(device)

    def address(name):
        return held[name].data_ptr() if name in held else None

    width = case.head_dim
    compressed = operands.compressed
    call = LatentAttentionCall(
        query=address("query"),
        out=address("out"),
        window=address("window"),
        window_table=address("window_table"),
        window_starts=address("window_starts"),
        compressed=address("compressed"),
        compressed_table=address("compressed_table"),
        selection=address("selection"),
        rows=address("rows"),
        positions=address("positions"),
        sink=address("sink"),
        workspace=address("workspace"),
        workspace_bytes=held["workspace"].numel(),
        tokens=tokens,
        heads=case.heads,
        head_dim=width,
        window_size=case.window,
        page=case.page,
        window_pages=operands.window.table.shape[1],
        window_record=records.record_bytes(case.window_format, width),
        window_plane_pages=operands.window.pages.shape[0],
        entries=case.entries if compressed is not None else 0,
        columns=case.columns if compressed is not None else 0,
        compressed_pages=(
            compressed.table.shape[1] if compressed is not None else 0
        ),
        compressed_record=(
            records.record_bytes(case.compressed, width)
            if compressed is not None
            else 0
        ),
        compressed_plane_pages=(
            compressed.pages.shape[0] if compressed is not None else 0
        ),
        multiprocessors=torch.cuda.get_device_properties(
            device
        ).multi_processor_count,
        scale=operands.scale,
        error=b"",
    )
    return call, held


def test_the_call_mirror_is_the_headers_call_structure():
    """The generated header asserts the size and every field offset it holds."""
    assert_abi_layout(LatentAttentionCall, "uniserve_latent_attention_call")


@pytest.mark.gpu
@pytest.mark.parametrize("captured", capture.MODES)
@pytest.mark.parametrize("name", CASES)
def test_every_query_attends_its_window_and_its_selection_with_its_sink(
    name, captured, kernel_module, torch_device
):
    case = CASES[name]
    entry = entry_of(kernel_module("latent_attention", BACKEND, key(case)))
    operands = batch(case)
    call, held = built(case, operands, torch_device)

    capture.call(entry, call, torch_device, captured)
    out = held["out"].cpu().double()

    expected, magnitude = reference(case, operands)
    # The drawn latents are within the operator's declared envelope, so the
    # bound below is within its data-free one.
    assert float(magnitude.max()) <= LATENT_ENVELOPE

    allowed = bound(case, magnitude, expected, interpreted=False)
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
@pytest.mark.parametrize(("field", "value", "words"), REFUSALS)
def test_a_call_the_module_cannot_serve_is_refused_by_its_guard(
    field, value, words, kernel_module, torch_device
):
    case = CASES["the model's decode of 1 to 6 tokens per request"]
    entry = entry_of(kernel_module("latent_attention", BACKEND, key(case)))
    operands = batch(case)
    call, _held = built(case, operands, torch_device)

    setattr(call, field, value)
    status = entry(
        ctypes.byref(call),
        ctypes.c_void_p(torch.cuda.current_stream(torch_device).cuda_stream),
    )
    message = call.error.decode("utf-8", "replace").rstrip("\x00")
    assert status != 0, f"the module served {field} = {value}"
    assert words in message, (
        f"{field} = {value} was refused with {message!r}, which does not "
        f"name {words!r}"
    )


#: Where the checkpoint's reference vectors are generated
#: (``generate_attention.py``), and the generator, whose rotary factors and
#: rotation give the layer's operands the reference's own arithmetic.
REFERENCE = "artifacts/deepseek_v41/attention-reference"
GENERATOR = "crates/models/tests/it/deepseek_v41/generate_attention.py"

#: The layers the vectors hold, with each one's compressor ratio
#: (``compress_ratios`` of the checkpoint's ``config.json``).
CHECKPOINT_LAYERS = {2: 2, 20: 1}

#: The rotary factors of layers 2 to 39 (``config.json``): the trailing 64
#: channels, YaRN over 65,536 original positions at θ 160,000 with factor 16
#: and β 32 and 1; and the RMSNorm epsilon.
ROTARY = {"original": 65536, "base": 160000.0, "factor": 16.0, "fast": 32}
ROTATED = 64
EPS = 1e-20

#: BF16's unit roundoff, the layer check's bound unit.
BF16_UNIT = 2.0**-8


def checkpoint_operands(directory, generator_path, layer, ratio):
    """Layer ``layer``'s attention operands as the reference computes them:
    the queries rotated, the window latents normalized and rotated, the
    compressed latents rotated at their groups' first positions, each latent
    encoded into its record, and the reference's selection; with the
    reference's attention and the rotary factors that rotate the output
    back.
    """  # noqa: D205
    spec = importlib.util.spec_from_file_location(
        "generate_attention", generator_path
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    prefix = f"layers.{layer}.attn"
    with safe_open(directory / "vectors.safetensors", "pt") as vectors:
        query = vectors.get_tensor(f"{prefix}.q").clone()
        projected = vectors.get_tensor(f"{prefix}.kv")
        latent = vectors.get_tensor(f"{prefix}.latent").clone()
        selection = vectors.get_tensor(f"{prefix}.selection")
        expected = vectors.get_tensor(f"{prefix}.attention")
    with safe_open(directory / "parameters.safetensors", "pt") as parameters:
        norm = parameters.get_tensor(f"{prefix}.kv_norm.weight")
        sink = parameters.get_tensor(f"{prefix}.attn_sink")

    tokens = query.shape[0]
    freqs = generator.freqs_cis(
        ROTATED,
        tokens,
        ROTARY["original"],
        ROTARY["base"],
        ROTARY["factor"],
        ROTARY["fast"],
        1,
    )
    generator.rotate(query[..., -ROTATED:], freqs)
    window = generator.rms_norm(projected, norm, EPS)
    generator.rotate(window[..., -ROTATED:], freqs)
    groups = latent.shape[0]
    generator.rotate(latent[..., -ROTATED:], freqs[: groups * ratio : ratio])

    case = Case(
        window_format="mxfp8",
        compressed="nvfp4",
        entries=MODEL["page"] // ratio,
        columns=selection.shape[1],
        requests=((tokens, tokens),),
        ratio=ratio,
        **MODEL,
    )
    shuffled = torch.Generator().manual_seed(layer)
    operands = Batch(
        query=query,
        window=paged(
            "mxfp8",
            [window.float()],
            case.page,
            math.ceil(tokens / case.page),
            False,
            shuffled,
        ),
        starts=torch.zeros(1, dtype=torch.int32),
        compressed=paged(
            "nvfp4",
            [latent.float()],
            case.entries,
            math.ceil(groups / case.entries),
            False,
            shuffled,
        ),
        selection=selection.to(torch.int32).contiguous(),
        rows=torch.zeros(tokens, dtype=torch.int32),
        positions=torch.arange(tokens, dtype=torch.int32),
        sink=sink.float(),
        scale=case.head_dim**-0.5,
    )
    return case, operands, expected, generator, freqs


@pytest.mark.gpu
@pytest.mark.parametrize("layer", CHECKPOINT_LAYERS)
def test_a_checkpoint_layer_attends_as_the_reference(
    layer, repository, kernel_module, torch_device
):
    directory = repository / REFERENCE
    if not (directory / "vectors.safetensors").exists():
        pytest.skip(f"no reference vectors in {REFERENCE}: generate them")
    case, operands, expected, generator, freqs = checkpoint_operands(
        directory, repository / GENERATOR, layer, CHECKPOINT_LAYERS[layer]
    )
    entry = entry_of(kernel_module("latent_attention", BACKEND, key(case)))
    call, held = built(case, operands, torch_device)

    capture.call(entry, call, torch_device, captured=True)
    out = held["out"].cpu()
    # The output rotated back as the layer rotates it, rounded into BF16.
    generator.rotate(out[..., -ROTATED:], freqs, inverse=True)

    got, wanted = out.double().flatten(1), expected.double().flatten(1)
    difference = (got - wanted).norm(dim=1)
    norms = wanted.norm(dim=1)
    whole = float(difference.norm() / norms.norm())
    # A row's norm floored at a tenth of the mean row norm, as the layer
    # check floors it.
    row = float((difference / norms.clamp_min(0.1 * norms.mean())).max())
    limit = 4.0 * BF16_UNIT
    assert whole <= limit and row <= 5.0 * limit, (
        f"layer {layer}: the attention differs from the reference's by a "
        f"relative L2 of {whole:.3e} (worst row {row:.3e}), bound {limit:.3e}"
    )
