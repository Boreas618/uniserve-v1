"""The posterior program against the reference's chain and recorded bits.

The operator draws a latent from the encoder's moments::

    log_variance = clamp(log_variance, -30, 20)
    sample       = mean + exp(0.5 * log_variance) * noise
    out          = round_to_f16(sample)                    stored as f32

Two claims are defended:

1. **The chain.** The output equals the reference's expression evaluated in
   torch, in its order, bit for bit: the tolerance is exact, the arithmetic
   *is* the specification, and one ulp is a failure.
2. **The fixture.** h3 recorded ground truth for four elements of two
   channels after the affine a conditioner applies (the *normalised* latents,
   as IEEE-754 bits, big-endian as h3 recorded them). The affine applied in
   torch to the program's output must give the recorded bits, which pins the
   two steps a fused program drops first: ``0.25000014901161194`` becomes
   exactly ``0.25`` through the round trip, and ``-44054.4296875`` becomes
   ``-44064.0``.

Every operand is read through its own row and channel strides, so both
physical layouts are one launch and must give the same bits: a channels-last
caller passes ``(channels, 1)``, with the moments one ``[positions, 2 *
channels]`` tensor cut on the channel axis (h3's ``VideoEncoder::posterior``)
so the row stride is twice the sample's; a channel-major caller (the VAE's
rasters) passes ``(1, positions)``, the moments' halves an offset apart.

A module compiled for a dependent launch reads only what the grid before it
wrote (:mod:`uniserve_testing.kernels.chain`); that launch needs a device.
"""

import struct

import pytest
import torch
from uniserve_testing.kernels import chain

pytestmark = pytest.mark.unit

#: The reference's two per-channel tables (`latents_mean`, `latents_std`).
LATENTS_MEAN = [0.858090341091156, -0.9606591463088989]
LATENTS_STD = [1.2223774194717407, 1.2767263650894165]

#: h3's fixture: two positions of two channels, and the normalised sample as
#: recorded bits.
MEAN = [[0.25, -1.5], [3.0, 0.125]]
LOG_VARIANCE = [[-40.0, 25.0], [-0.5, 0.0]]
NOISE = [[0.5, -2.0], [1.0, 0.75]]
GROUND_TRUTH = ["befeb3c5", "c706d084", "4018f211", "3fb8095e"]

#: The elements one program covers.
BLOCK = 1024


def launch(kernels, operands, out, channels, total, block=BLOCK, pdl=False):
    """One launch over every element, dependent when `pdl` is.

    ``operands`` are the mean, the log-variance and the noise, each with its
    row and channel strides; ``out`` is written at its own strides.
    """
    tensors = [tensor for tensor, _, _ in operands]
    strides = [stride for _, *pair in operands for stride in pair]
    kernels._posterior_kernel[(-(-total // block),)](
        *tensors,
        out,
        channels,
        *strides,
        out.stride(0),
        out.stride(1),
        total,
        BLOCK=block,
        PDL=pdl,
        launch_pdl=pdl,
    )


def bits(value):
    """The IEEE-754 single bits of a float, eight hex digits, most
    significant first, as h3 recorded them.
    """  # noqa: D205
    return struct.pack(">f", float(value)).hex()


def operands_of(layout, device):
    """The fixture's operands in one physical layout, with their strides."""
    mean = torch.tensor(MEAN, dtype=torch.float32, device=device)
    log_variance = torch.tensor(
        LOG_VARIANCE, dtype=torch.float32, device=device
    )
    noise = torch.tensor(NOISE, dtype=torch.float32, device=device)
    positions = mean.shape[0]
    if layout == "channels-last":
        # One row per position, the channel innermost: the moments joined on
        # that axis, so each half is a view with a row stride of twice the
        # channels.
        halves = torch.cat([mean, log_variance], dim=1).chunk(2, dim=1)
        return [(half, half.stride(0), half.stride(1)) for half in halves] + [
            (noise, noise.stride(0), noise.stride(1))
        ]
    # One column per position, the position innermost: the encoder's raster,
    # the moments one buffer whose halves are an offset apart.
    halves = torch.cat([mean, log_variance], dim=0).chunk(2, dim=0)
    return [(half, half.stride(0), half.stride(1)) for half in halves] + [
        (noise.t().contiguous().t(), 1, positions)
    ]


@pytest.mark.parametrize("layout", ["channels-last", "channel-major"])
def test_the_sample_is_the_chain_and_the_recorded_bits(
    layout, torch_device, kernel_source
):
    positions, channels = len(MEAN), len(MEAN[0])
    out = torch.zeros(positions, channels, device=torch_device)

    launch(
        kernel_source("posterior"),
        operands_of(layout, torch_device),
        out,
        channels,
        positions * channels,
    )
    out = out.cpu()

    # The reference's chain in torch, in its order, with the FP16 round trip.
    log_variance = torch.tensor(LOG_VARIANCE).clamp(-30.0, 20.0)
    sample = torch.tensor(MEAN) + torch.exp(0.5 * log_variance) * torch.tensor(
        NOISE
    )
    expected = sample.to(torch.float16).to(torch.float32)
    chain_faults = [
        f"element {index}: {got!r} ({bits(got)}), the reference {want!r} "
        f"({bits(want)})"
        for index, (got, want) in enumerate(
            zip(out.reshape(-1).tolist(), expected.reshape(-1).tolist())
        )
        if bits(got) != bits(want)
    ]
    assert not chain_faults, f"{layout} chain: " + "; ".join(chain_faults)

    # The affine a conditioner applies, in the reference's order.
    normalised = (out - torch.tensor(LATENTS_MEAN)) / torch.tensor(LATENTS_STD)
    recorded = [bits(value) for value in normalised.reshape(-1).tolist()]
    assert recorded == GROUND_TRUTH, (
        f"{layout} fixture: the normalised sample is {recorded}, h3 recorded "
        f"{GROUND_TRUTH}"
    )


@pytest.mark.gpu
def test_a_dependent_launch_reads_what_its_producer_wrote(kernel_source):
    torch.manual_seed(0)
    kernels = kernel_source("posterior")
    positions, channels, block = 64, 16, 256
    # Every operand is a `[positions, channels]` view with a channel stride
    # of 2: a stride of 1 would be specialized into the launch here, where
    # the module takes it as a word.
    row, step = 2 * channels, 2

    def launched(inputs, outputs, pdl):
        operands = [
            (inputs[name], row, step)
            for name in ("mean", "log_variance", "noise")
        ]
        out = outputs["out"].as_strided((positions, channels), (row, step))
        launch(
            kernels, operands, out, channels, positions * channels, block, pdl
        )
        return [outputs["out"]]

    case = chain.Case(
        "posterior",
        {
            "mean": torch.randn(positions, row),
            "log_variance": torch.randn(positions, row),
            "noise": torch.randn(positions, row),
        },
        {"out": torch.zeros(positions, row)},
        launched,
    )
    failures = chain.check(case)
    assert failures == 0, (
        f"posterior launched as a dependent: {failures} of {chain.ROUNDS} "
        "rounds differ from its launch after a synchronization"
    )
