"""The rotary embedding packages' kernels against their references.

``rotary_factors`` reduces its phase through libdevice's cosine and sine,
which the interpreter does not have, so it is checked on a device.
"""

import pathlib

import pytest

from tests.python.kernels.checks import interpreted, on_device

pytestmark = pytest.mark.unit

FAMILY = pathlib.Path(__file__).resolve().parent

#: The checks that run whole on Triton's interpreter.
INTERPRETED = (
    "rope_apply_check.py",
    "qk_norm_rope_check.py",
    "position_interpolate_check.py",
)


@pytest.mark.parametrize("check", INTERPRETED)
def test_a_rotary_kernel_matches_its_reference_on_the_interpreter(check):
    interpreted(FAMILY / check)


@pytest.mark.gpu
def test_the_rotary_factors_match_their_reference_on_the_device():
    on_device(FAMILY / "rotary_factors_check.py")
