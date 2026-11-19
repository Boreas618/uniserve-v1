"""The `topk_softmax` router against its build and the routing it must pin.

The module of the descriptor below (``module.cubin`` and its
``metadata.json``, resolved at the session's Build Server for the session's
device) is held to its own build: the compile constants the metadata records
(``definitions``: the expert count and the top-k the kernel derives its
iterations and its butterfly from) are the descriptor's, and the fifteen words
of its signature are the operator's, in order and by kind; the second pointer
is the scale, which a ``scale_none`` module still declares as one, the third
the selection bias, and the last three the expert map, the live count and the
expert load, which the operator passes null without them. A module
built for another descriptor fails here, the observable form of the kernel's
`__trap()`.

**The rule it pins.** The scores are a softmax over the vocabulary, and the
picks are its top `top`, descending, **equal probabilities in ascending expert
id**: the kernel compares `(value, expert)` pairs, so a tie must resolve by id
and not by whichever lane spoke last. Renormalized, the weights are the picks'
share of the picks' own sum, floored at FP32's epsilon. The fixed input states
both cases: one row whose second and third picks are exactly equal in BF16,
and padding past `experts` filled with a value that would win if it were ever
read (a row stride above the vocabulary).

**The launch.** The cubin is loaded through the CUDA driver and launched on
the fixed input the way the operator launches it (one token per warp of the
module's block, the parameters in the signature's order), twice: plain,
comparing the ids exactly and the weights within the operator's tolerance,
and with the logical-to-physical map reversed and the last row turned into a
padding row of NaN scores, as a captured bucket's rows past the batch hold.
The mapped ids must be the map's entries of the same picks, and the padding
row must keep an id that names no expert: it picks none, and a read of the map
at that id is an illegal address.

**The expert load.** Launched twice into the same counters with the map
reversed and a live count below the rows, the counters must hold, per logical
expert, twice the picks of the live rows: the count is by the expert's own id,
whatever slot the map names, and a row past the live count adds nothing.

**The sqrt-softplus router** (DeepSeek-V4.1-Flash's: 384 experts, top 6, a
selection bias and a routed scaling factor of 1.5 as the per-expert scale) is
held to the reference router: each score ``sqrt(softplus(logit))``, the picks
the top of ``score + bias`` with ties in ascending id, the weights the picks'
unbiased scores over their sum plus ``1e-20``, times the scale. Its bound is
the operator's, over a reduction of the picks.

The module half and the launch need a module built for a device; the pinned
reference's own tie rule is checked on the host.
"""

import ctypes
import json

import pytest
import torch

pytestmark = pytest.mark.unit

#: The router's descriptor, by field in the package's dimension order.
DESCRIPTOR = {
    "scores": "bf16",
    "experts": 128,
    "top": 8,
    "scoring": "softmax",
    "renormalize": True,
    "bias": False,
    "scale": "none",
}


def key_of(descriptor):
    """A descriptor's key, its fields in the package's dimension order."""
    return "_".join(
        f"{field}_{str(value).lower()}" for field, value in descriptor.items()
    )


KEY = key_of(DESCRIPTOR)
EXPERTS, TOP = DESCRIPTOR["experts"], DESCRIPTOR["top"]

#: DeepSeek-V4.1-Flash's router.
SQRT_SOFTPLUS = {
    "scores": "bf16",
    "experts": 384,
    "top": 6,
    "scoring": "sqrtsoftplus",
    "renormalize": True,
    "bias": True,
    "scale": "f32",
}
ROUTED_SCALE = 1.5

#: The fixed input's rows, and the elements between them: the vocabulary.
ROWS, ROW_STRIDE = 4, DESCRIPTOR["experts"]

#: The operator's parameters, in declaration order, by kind.
SIGNATURE = (
    ("scores", "pointer"),
    ("score_kind", "i32"),
    ("row_stride", "i64"),
    ("rows", "i32"),
    ("experts", "i32"),
    ("top", "i32"),
    ("renormalize", "i32"),
    ("scale", "pointer"),
    ("scale_kind", "i32"),
    ("bias", "pointer"),
    ("ids", "pointer"),
    ("weights", "pointer"),
    ("map", "pointer"),
    ("live", "pointer"),
    ("load", "pointer"),
)

TORCH_TYPES = {
    "f32": torch.float32,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
}

#: The scores' type as a module's signature spells it.
SIGNATURE_TYPES = {"f32": "*fp32", "bf16": "*bf16", "f16": "*fp16"}

#: FP32's epsilon, the floor the renormalization divides by, and its unit
#: roundoff.
FLT_EPSILON = 1.1920928955078125e-07
UNIT_ROUNDOFF = 2.0**-24

#: The storage kinds the kernel reads, as its `score_kind` and `scale_kind`
#: take them.
KINDS = {"f32": 0, "bf16": 1, "f16": 2, "none": -1}


def fixed_case(experts, top, rows, row_stride, renormalize):
    """The input every test agrees on, and the routing it must produce.

    The scores are built in BF16, so a tie made in the values is a tie the
    kernel sees. Row 0 carries its second and third picks exactly equal, which
    only the ascending-id rule orders; every row's lanes past `experts` hold a
    value large enough to win if the padding were ever read.
    """
    generator = torch.Generator().manual_seed(20240704)
    scores = torch.randn(rows, row_stride, generator=generator) * 0.5
    if row_stride > experts:
        # Nothing may read these: if a lane did, its pick would be the pad.
        scores[:, experts:] = 1000.0
    scores[0, 0] = 10.0
    scores[0, 1] = 5.0
    scores[0, 2] = 5.0
    scores = scores.to(torch.bfloat16)
    exact = scores.double()[:, :experts]

    # The softmax the kernel computes, in FP64: `exp(x - max)` over the
    # vocabulary, normalized.
    maximum = exact.max(dim=-1, keepdim=True).values
    probabilities = torch.exp(exact - maximum)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

    # The picks: descending, ties ascending by id. `torch.topk` does not
    # promise that order, so the reference sorts explicitly.
    ids = torch.empty(rows, top, dtype=torch.int64)
    picked = torch.empty(rows, top, dtype=torch.float64)
    for row in range(rows):
        order = sorted(
            range(experts),
            key=lambda expert: (-probabilities[row, expert].item(), expert),
        )[:top]
        ids[row] = torch.tensor(order, dtype=torch.int64)
        picked[row] = probabilities[row][order]

    # Renormalized over the picks, as the kernel's butterfly does; its floor is
    # FP32's epsilon. A module built without renormalization reports the
    # vocabulary's own probabilities.
    weights = picked
    if renormalize:
        total = picked.sum(dim=-1, keepdim=True)
        total = torch.where(
            total < FLT_EPSILON, torch.full_like(total, FLT_EPSILON), total
        )
        weights = picked / total
    return scores, ids, weights


def pinned():
    """The fixed case of the router's descriptor."""
    return fixed_case(EXPERTS, TOP, ROWS, ROW_STRIDE, DESCRIPTOR["renormalize"])


def test_the_pinned_routing_resolves_a_tie_by_ascending_id():
    _, ids, _ = pinned()
    assert int(ids[0, 1]) < int(ids[0, 2]), (
        f"the tie in row 0 resolved to {ids[0, 1].item()} before "
        f"{ids[0, 2].item()}"
    )


@pytest.fixture
def module(kernel_module):
    """The router's built module directory."""
    return kernel_module("topk_softmax", "nvcc", KEY)


@pytest.fixture
def metadata(module):
    """The module's `metadata.json`."""
    return json.loads((module / "metadata.json").read_text())


@pytest.mark.gpu
def test_the_module_was_built_with_the_descriptors_constants(metadata):
    definitions = metadata.get("definitions")
    assert definitions, (
        "the module records no definitions, so its build constants are "
        "unobservable"
    )
    for name, value in (("UNISERVE_EXPERTS", EXPERTS), ("UNISERVE_TOP", TOP)):
        assert definitions.get(name) == value, (
            f"the module was built with {name} = {definitions.get(name)}, "
            f"its descriptor {KEY} needs {value}"
        )


@pytest.mark.gpu
def test_the_modules_signature_is_the_operators(metadata):
    signature = metadata.get("signature")
    assert signature, (
        "the module publishes no signature, so its words cannot be checked"
    )
    operator = [name for name, _ in SIGNATURE]
    assert list(signature) == operator, (
        f"the module's parameters are {list(signature)}, the operator's are "
        f"{operator}"
    )
    for (name, kind), published in zip(SIGNATURE, signature.values()):
        if kind == "pointer":
            assert published.startswith("*"), (
                f"parameter {name} is {published}, the operator passes a "
                "pointer"
            )
        else:
            assert published == kind, (
                f"parameter {name} is {published}, the operator passes an "
                f"{kind}"
            )
    wanted = SIGNATURE_TYPES[DESCRIPTOR["scores"]]
    assert signature["scores"] == wanted, (
        f"the module reads {signature['scores']} scores, the descriptor "
        f"states {wanted}"
    )


def launch(
    module,
    metadata,
    device,
    scores,
    mapping=None,
    live=None,
    load=None,
    descriptor=DESCRIPTOR,
    scale=None,
    bias=None,
):
    """Launch the module's cubin on `scores`; answers the ids and the weights
    it wrote, on the host.

    The cubin is loaded into the context torch made current and launched on
    torch's current stream, one token per warp of the block the metadata
    states. `mapping`, when given, is the device's logical-to-physical expert
    map; `live` the count of leading live rows; `load` the device's int64
    counters the live routes add into; `scale` and `bias` the per-expert FP32
    factors and selection bias of a descriptor that reads them. The operator
    passes a null pointer for each it is not given.
    """  # noqa: D205
    from cuda.bindings import driver

    def checked(result):
        error, *values = result
        assert error == driver.CUresult.CUDA_SUCCESS, (
            f"a CUDA driver call failed: {error}"
        )
        return values[0] if len(values) == 1 else values

    rows, row_stride = scores.shape
    dtype, top = descriptor["scores"], descriptor["top"]
    device_scores = scores.to(TORCH_TYPES[dtype]).contiguous().to(device)
    ids = torch.full((rows, top), -1, dtype=torch.int32, device=device)
    weights = torch.zeros((rows, top), dtype=torch.float32, device=device)
    scale = (
        torch.zeros(16, dtype=torch.float32, device=device)
        if scale is None
        else scale.to(device)
    )
    device_bias = None if bias is None else bias.to(device)
    device_map = None if mapping is None else mapping.to(device)
    device_live = (
        None
        if live is None
        else torch.tensor([live], dtype=torch.int32, device=device)
    )
    torch.cuda.synchronize(device)

    loaded = checked(
        driver.cuModuleLoadData((module / "module.cubin").read_bytes())
    )
    try:
        function = checked(
            driver.cuModuleGetFunction(loaded, metadata["name"].encode())
        )
        values = (
            device_scores.data_ptr(),
            KINDS[dtype],
            row_stride,
            rows,
            descriptor["experts"],
            top,
            int(descriptor["renormalize"]),
            scale.data_ptr(),
            KINDS[descriptor["scale"]],
            0 if device_bias is None else device_bias.data_ptr(),
            ids.data_ptr(),
            weights.data_ptr(),
            0 if device_map is None else device_map.data_ptr(),
            0 if device_live is None else device_live.data_ptr(),
            0 if load is None else load.data_ptr(),
        )
        kinds = (
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        warps = metadata["num_warps"]
        checked(
            driver.cuLaunchKernel(
                function,
                -(-rows // warps),
                1,
                1,
                warps * 32,
                1,
                1,
                metadata.get("shared", 0),
                torch.cuda.current_stream(device).cuda_stream,
                (values, kinds),
                0,
            )
        )
        torch.cuda.synchronize(device)
    finally:
        checked(driver.cuModuleUnload(loaded))
    return ids.cpu(), weights.cpu()


def reference_bound(experts):
    """The operator's own tolerance: `γₙ` over the FP32 softmax plus four
    roundings.
    """  # noqa: D205
    terms = experts * UNIT_ROUNDOFF
    return terms / (1.0 - terms) + 4.0 * UNIT_ROUNDOFF


@pytest.mark.gpu
def test_the_router_routes_the_pinned_input(module, metadata, torch_device):
    scores, ids, weights = pinned()

    got_ids, got_weights = launch(module, metadata, torch_device, scores)

    wrong = got_ids.long() != ids
    first = wrong.nonzero()[:3].tolist()
    assert not bool(wrong.any()), (
        f"{int(wrong.sum())} of {ids.numel()} picks are not the ones the "
        "scores rank; first "
        + ", ".join(
            f"[{row}, {slot}]: {got_ids[row, slot].item()} for "
            f"{ids[row, slot].item()}"
            for row, slot in first
        )
    )
    bound = reference_bound(EXPERTS)
    outside = ~((got_weights.double() - weights).abs() <= bound * weights.abs())
    first = outside.nonzero()[:3].tolist()
    assert not bool(outside.any()), (
        f"{int(outside.sum())} of {weights.numel()} weights are outside "
        f"{bound:.3e} relative; first "
        + ", ".join(
            f"[{row}, {slot}]: {got_weights[row, slot].item():.9e} for "
            f"{weights[row, slot].item():.9e}"
            for row, slot in first
        )
    )


@pytest.mark.gpu
def test_a_mapped_launch_names_slots_and_a_padding_row_none(
    module, metadata, torch_device
):
    scores, ids, _ = pinned()
    padded = scores.clone()
    padded[-1, :] = float("nan")
    # Logical expert `e` lies in slot `experts - 1 - e`.
    reversed_map = torch.arange(EXPERTS - 1, -1, -1, dtype=torch.int32)

    got_ids, _ = launch(module, metadata, torch_device, padded, reversed_map)

    expected = reversed_map.long()[ids[:-1]]
    wrong = int((got_ids[:-1].long() != expected).sum())
    assert wrong == 0, (
        f"{wrong} of {expected.numel()} routes do not hold the slot the map "
        "names"
    )
    padding = got_ids[-1].long()
    assert not bool(((padding >= 0) & (padding < EXPERTS)).any()), (
        f"the padding row names a slot: {padding.tolist()}"
    )


@pytest.mark.gpu
def test_the_load_counts_each_live_route_by_its_logical_expert(
    module, metadata, torch_device
):
    scores, ids, _ = pinned()
    reversed_map = torch.arange(EXPERTS - 1, -1, -1, dtype=torch.int32)
    live = ROWS - 1
    load = torch.zeros(EXPERTS, dtype=torch.int64, device=torch_device)

    # Two steps add into the same counters, as two replays of one graph do.
    for _ in range(2):
        launch(
            module,
            metadata,
            torch_device,
            scores,
            reversed_map,
            live=live,
            load=load,
        )

    expected = 2 * torch.bincount(ids[:live].flatten(), minlength=EXPERTS)
    counted = load.cpu()
    wrong = (counted != expected).nonzero().flatten()[:3].tolist()
    assert not wrong, (
        f"{len(wrong)} or more experts hold another count than twice the live "
        "rows' picks; first "
        + ", ".join(
            f"expert {expert}: {counted[expert].item()} for "
            f"{expected[expert].item()}"
            for expert in wrong
        )
    )


def sqrt_softplus_case(rows):
    """Logits of a router's spread, a selection bias that moves picks, and the
    reference's routing of them, in FP64: ids by `score + bias` with ties in
    ascending id, weights the unbiased scores renormalized and scaled.
    """  # noqa: D205
    experts, top = SQRT_SOFTPLUS["experts"], SQRT_SOFTPLUS["top"]
    generator = torch.Generator().manual_seed(4101)
    logits = (torch.randn(rows, experts, generator=generator) * 2.0).to(
        torch.bfloat16
    )
    bias = (torch.randn(experts, generator=generator) * 0.05).float()
    exact = logits.double()
    # PyTorch's softplus: the logit itself above its threshold of 20.
    softplus = torch.where(exact > 20.0, exact, torch.log1p(torch.exp(exact)))
    scores = softplus.sqrt()
    keys = scores + bias.double()
    ids = torch.empty(rows, top, dtype=torch.int64)
    weights = torch.empty(rows, top, dtype=torch.float64)
    for row in range(rows):
        order = sorted(range(experts), key=lambda e: (-keys[row, e].item(), e))[
            :top
        ]
        ids[row] = torch.tensor(order)
        picked = scores[row][order]
        weights[row] = picked / (picked.sum() + 1e-20) * ROUTED_SCALE
    return logits, bias, ids, weights


@pytest.mark.gpu
def test_a_sqrt_softplus_router_selects_by_the_bias_and_weighs_without_it(
    kernel_module, torch_device
):
    directory = kernel_module("topk_softmax", "nvcc", key_of(SQRT_SOFTPLUS))
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["definitions"]["UNISERVE_SQRTSOFTPLUS"] == 1
    rows = 64
    logits, bias, ids, weights = sqrt_softplus_case(rows)
    scale = torch.full((SQRT_SOFTPLUS["experts"],), ROUTED_SCALE)

    got_ids, got_weights = launch(
        directory,
        metadata,
        torch_device,
        logits,
        descriptor=SQRT_SOFTPLUS,
        scale=scale,
        bias=bias,
    )

    wrong = got_ids.long() != ids
    assert not bool(wrong.any()), (
        f"{int(wrong.sum())} of {ids.numel()} picks are not the reference's; "
        f"first at {wrong.nonzero()[0].tolist()}"
    )
    # A score is its own expert's, so the reduction is the picks' sum; the
    # softplus, the root and the scale cost one rounding each.
    terms = SQRT_SOFTPLUS["top"] * UNIT_ROUNDOFF
    bound = terms / (1.0 - terms) + 4.0 * UNIT_ROUNDOFF
    outside = ~((got_weights.double() - weights).abs() <= bound * weights.abs())
    assert not bool(outside.any()), (
        f"{int(outside.sum())} of {weights.numel()} weights outside "
        f"{bound:.3e} relative; first at {outside.nonzero()[0].tolist()}"
    )
    # The bias moved picks: without it some token takes another expert.
    unbiased = (
        torch.where(
            logits.double() > 20.0,
            logits.double(),
            torch.log1p(torch.exp(logits.double())),
        )
        .topk(SQRT_SOFTPLUS["top"])
        .indices.sort()
        .values
    )
    assert not torch.equal(unbiased, ids.sort().values), (
        "the case's bias changes no pick, so it does not test the selection"
    )
