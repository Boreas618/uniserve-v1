"""The tensor primitives' kernels against numpy and torch.

Each primitive's check launches its package's kernel the way its operator
launches it (``operands.py``) over strided, broadcast and multi-program
inputs of rank 1 to 6. A check runs whole on Triton's interpreter, and with
``--device`` it runs the same cases compiled for a device, where a BF16 store
rounds to nearest and libdevice evaluates its own functions.
"""

import pathlib

import pytest

from tests.python.kernels.checks import interpreted, on_device

pytestmark = pytest.mark.unit

FAMILY = pathlib.Path(__file__).resolve().parent

#: One check per primitive.
CHECKS = (
    "elementwise_unary_check.py",
    "elementwise_binary_check.py",
    "elementwise_where_check.py",
    "cast_check.py",
    "copy_strided_check.py",
    "fill_check.py",
)


@pytest.mark.parametrize("check", CHECKS)
def test_a_primitive_matches_its_reference_on_the_interpreter(check):
    interpreted(FAMILY / check)


@pytest.mark.gpu
@pytest.mark.parametrize("check", CHECKS)
def test_a_primitive_matches_its_reference_on_the_device(check):
    on_device(FAMILY / check, "--device")
