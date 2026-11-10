"""The image, video and audio layout packages' kernels against their references.

Several of these packages carry two checks: one that runs on the interpreter
(``*_interp_check.py``) and one that compiles the kernel for a device.
"""

import pathlib

import pytest

from tests.python.kernels.checks import interpreted, on_device

pytestmark = pytest.mark.unit

FAMILY = pathlib.Path(__file__).resolve().parent

#: The checks that run on Triton's interpreter.
INTERPRETED = (
    "adaptive_pool_interp_check.py",
    "cross_fade_check.py",
    "frame_pad_check.py",
    "patch_check.py",
    "rgb_bytes_interp_check.py",
    "transpose_interp_check.py",
    "unpatchify_video_check.py",
    "scaled_add_check.py",
)

#: The checks that compile their kernel for a device.
DEVICE = (
    "adaptive_pool_check.py",
    "pcm_interleave_check.py",
    "rgb_bytes_check.py",
    "transpose_check.py",
)


@pytest.mark.parametrize("check", INTERPRETED)
def test_a_layout_kernel_matches_its_reference_on_the_interpreter(check):
    interpreted(FAMILY / check)


@pytest.mark.gpu
@pytest.mark.parametrize("check", DEVICE)
def test_a_layout_kernel_matches_its_reference_on_the_device(check):
    on_device(FAMILY / check)


#: The operators whose dependent launch `pdl_chain_check.py` checks.
DEPENDENT = (
    "adaptive_pool",
    "cross_fade",
    "rgb_bytes",
)


@pytest.mark.gpu
@pytest.mark.parametrize("op", DEPENDENT)
def test_a_dependent_launch_reads_what_its_producer_wrote(op):
    on_device(FAMILY / "pdl_chain_check.py", "--op", op)
