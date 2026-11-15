"""The causal_conv1d program over a batch of rows against FP64.

The batch is the operator's: packed rows of different lengths, rows starting
at position zero and rows continuing from the inputs their slot keeps
(including rows shorter than the state, whose new state is part old state and
part new input), a padding row of slot -1, rows that keep a checkpoint inside
their tokens, at their end and within the state's reach of their start, and
decode rows of one token. The inputs are a column range of a wider projection,
so the program reads its row stride rather than the view's width. The
interpreter runs 48 channels; a device runs the model's 8192.

The reference is ``torch.nn.functional.conv1d`` in FP64 over each row's inputs
preceded by its slot's state (or by zeros), followed by the SiLU, as
transformers' ``causal_conv1d_fn`` and ``causal_conv1d_update`` compute it.
The bound follows the arithmetic: ``K`` products of rounded inputs and taps
summed in FP32, the SiLU's few FP32 ulps, and one rounding into the output
type, measured against the sum of the products' magnitudes (the sum may
cancel). The states, the checkpoints and the trail move inputs unchanged, so
they must agree exactly, and a padding row's output is exactly zero with its
slot untouched.
"""

import pytest
import torch

pytestmark = pytest.mark.unit

TORCH_TYPES = {"bf16": torch.bfloat16, "f32": torch.float32}
UNIT_ROUNDOFF = {"bf16": 2.0**-8, "f32": 2.0**-24}

#: The seed every case draws its values from.
SEED = 3

#: The batches, as the operator's rows: their lengths, the positions they start
#: at, their slots (-1 pads) and the position each keeps a checkpoint at.
CASES = {
    "prompt rows": {
        "lengths": [1, 2, 7, 40, 130, 33],
        "starts": [0, 5, 0, 37, 64, 2],
        "slots": [3, 0, -1, 1, 2, 4],
        "checkpoints": [0, 6, 0, 64, 192, 3],
    },
    "decode rows": {
        "lengths": [1] * 9,
        "starts": [3 * r + 1 for r in range(9)],
        "slots": [r if r % 4 else -1 for r in range(9)],
        "checkpoints": [3 * r + 2 if r % 2 else 0 for r in range(9)],
    },
    # A verification's rows: a few tokens each, continuing from their slots,
    # with the window after every token kept in the trail.
    "verification rows": {
        "lengths": [4, 1, 3, 4],
        "starts": [9, 30, 61, 2],
        "slots": [1, 3, -1, 0],
        "checkpoints": [0, 0, 0, 4],
        "trailing": True,
    },
}


def widths(interpreted):
    """The channels, taps and channel block of a run: small on the
    interpreter, the model's on a device.
    """  # noqa: D205
    if interpreted:
        return {"channels": 48, "width": 4, "block": 32}
    return {"channels": 8192, "width": 4, "block": 256}


def launch(
    kernels, x, weight, out, rows, columns, state, checkpoint, trail, **shape
):
    """One launch of the program over `rows` rows; `columns` are the int32
    offsets, sequence lengths, slots and checkpoints.
    """  # noqa: D205
    channels, width, block = shape["channels"], shape["width"], shape["block"]
    offsets, seq_lens, slots, checkpoints = columns
    kernels._causal_conv1d_kernel[((channels + block - 1) // block, rows)](
        x,
        weight,
        out,
        offsets,
        seq_lens,
        slots,
        checkpoints,
        state,
        checkpoint,
        trail,
        shape["stride"],
        int(shape["trailing"]),
        CHANNELS=channels,
        WIDTH=width,
        KEPT=1 << (width - 2).bit_length(),
        BLOCK=block,
        TOKENS=16,
        num_warps=4,
    )


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize("case", CASES)
def test_a_batch_matches_the_fp64_convolution(
    case, dtype, torch_device, interpreted, kernel_source
):
    torch.manual_seed(SEED)
    element = TORCH_TYPES[dtype]
    shape = widths(interpreted)
    channels, width = shape["channels"], shape["width"]
    lengths, starts, slots, checkpoints = (
        CASES[case][name]
        for name in ("lengths", "starts", "slots", "checkpoints")
    )
    trailing = CASES[case].get("trailing", False)
    rows, tokens = len(lengths), sum(lengths)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    stride = channels + 64
    wide = (torch.randn(tokens, stride) * 2.0).to(element)
    x = wide[:, 16 : 16 + channels]
    weight = (torch.randn(channels, width) * 0.5).to(element)
    slot_count = max(slots) + 2
    initial = torch.randn(slot_count, width - 1, channels).to(element)

    expected = torch.zeros(tokens, channels, dtype=torch.float64)
    terms = torch.zeros(tokens, channels, dtype=torch.float64)
    finals, kept = {}, {}
    # The window after each token of a row: the slot's state had the row ended
    # there.
    trailed = torch.zeros(tokens, width - 1, channels, dtype=torch.float64)
    for r in range(rows):
        if slots[r] < 0:
            continue
        lo, hi = offsets[r], offsets[r + 1]
        history = (
            initial[slots[r]]
            if starts[r] > 0
            else torch.zeros(width - 1, channels)
        )
        extended = torch.cat([history.double(), x[lo:hi].double()], dim=0)
        signal = extended.t()[None]  # [1, C, K-1+n]
        taps = weight.double()[:, None, :]
        summed = torch.nn.functional.conv1d(signal, taps, groups=channels)
        magnitude = torch.nn.functional.conv1d(
            signal.abs(), taps.abs(), groups=channels
        )
        summed, magnitude = summed[0].t(), magnitude[0].t()
        expected[lo:hi] = summed * torch.sigmoid(summed)
        terms[lo:hi] = magnitude
        finals[slots[r]] = extended[-(width - 1) :]
        for index in range(hi - lo):
            trailed[lo + index] = extended[index + 1 : index + width]
        # The checkpoint's place among the row's new tokens.
        point = checkpoints[r] - starts[r]
        if 0 < point <= lengths[r]:
            kept[slots[r]] = extended[point : point + width - 1]

    def column(values):
        return torch.tensor(values, dtype=torch.int32, device=torch_device)

    seq_lens = [start + length for start, length in zip(starts, lengths)]
    state = initial.clone().to(torch_device)
    checkpoint = torch.full_like(state, float("nan"))
    out = torch.full(
        (tokens, channels), float("nan"), dtype=element, device=torch_device
    )
    trail = torch.full(
        (tokens, width - 1, channels),
        float("nan"),
        dtype=element,
        device=torch_device,
    )
    on_device = wide.to(torch_device)
    launch(
        kernel_source("causal_conv1d"),
        on_device[:, 16 : 16 + channels],
        weight.to(torch_device),
        out,
        rows,
        tuple(map(column, (offsets, seq_lens, slots, checkpoints))),
        state,
        checkpoint,
        trail,
        stride=stride,
        trailing=trailing,
        **shape,
    )

    live = torch.tensor(
        [slots[r] >= 0 for r in range(rows) for _ in range(lengths[r])]
    )
    got = out.cpu().double()
    # The FP32 sum of K products and the SiLU, then one rounding: of the
    # products' magnitudes.
    allowed = (UNIT_ROUNDOFF[dtype] + (width + 4) * 2.0**-24) * (
        expected.abs() + terms
    )
    error = (got - expected).abs()
    outside = int((~(error[live] <= allowed[live])).sum())
    relative = float((error[live] / terms[live].clamp(min=1e-30)).max())
    assert outside == 0, (
        f"{case}, {dtype}: {outside} of {int(live.sum()) * channels} outputs "
        f"outside their bound; max abs {float(error[live].max()):.3e}, max "
        f"relative to its terms {relative:.3e}"
    )
    zeros = int((got[~live] != 0).sum())
    assert zeros == 0, f"padding rows: {zeros} nonzero outputs"
    for slot, final in finals.items():
        differ = int((state[slot].cpu().double() != final).sum())
        assert differ == 0, f"state of slot {slot}: {differ} elements differ"
    for slot, expected_kept in kept.items():
        differ = int((checkpoint[slot].cpu().double() != expected_kept).sum())
        assert differ == 0, (
            f"checkpoint of slot {slot}: {differ} elements differ"
        )
    if trailing:
        differ = int((trail.cpu().double()[live] != trailed[live]).sum())
        assert differ == 0, f"trail of every live token: {differ} differ"
    for slot in range(slot_count):
        if slot not in finals:
            changed = int((state[slot].cpu() != initial[slot]).sum())
            assert changed == 0, (
                f"slot {slot}, no row's, changed in {changed} elements"
            )
