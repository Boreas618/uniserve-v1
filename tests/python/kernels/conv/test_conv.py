"""The cuDNN convolution packages against their reference.

The three convolution operators share one source and one check, which names
its operator with ``--op``; each operator's built module
(``UNISERVE_KERNEL_MODULES``) is checked on a device.
"""

import pathlib

import pytest

from tests.python.kernels.checks import module, on_device

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

FAMILY = pathlib.Path(__file__).resolve().parent

#: The operators the convolution check covers.
CONVOLUTIONS = ("conv1d", "conv3d", "conv_transpose1d")


@pytest.mark.parametrize("op", CONVOLUTIONS)
def test_a_convolution_matches_its_reference(op):
    on_device(FAMILY / "conv_check.py", "--op", op, module(op, "cudnn"))


@pytest.mark.parametrize("op", CONVOLUTIONS)
def test_a_convolution_refuses_a_call_it_cannot_run(op):
    on_device(FAMILY / "conv_refusal_check.py", "--op", op, module(op, "cudnn"))
