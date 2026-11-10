"""The video sparse attention helpers' kernels against their references.

Three of them pin their rounding with inline PTX, which the interpreter cannot
execute, so they are checked on a device only.
"""

import pathlib

import pytest

from tests.python.kernels.checks import interpreted, on_device

pytestmark = pytest.mark.unit

FAMILY = pathlib.Path(__file__).resolve().parent

#: The checks that run whole on Triton's interpreter.
INTERPRETED = (
    "vsa_compress_pooled_check.py",
    "vsa_scores_check.py",
    "vsa_select_tiles_check.py",
    "vsa_zero_padding_check.py",
)

#: The checks whose kernels round through inline PTX.
DEVICE = (
    "vsa_add_gated_tiles_check.py",
    "vsa_compose_gated_rows_check.py",
    "vsa_prepare_tiles_check.py",
)


@pytest.mark.parametrize("check", INTERPRETED)
def test_a_vsa_kernel_matches_its_reference_on_the_interpreter(check):
    interpreted(FAMILY / check)


@pytest.mark.gpu
@pytest.mark.parametrize("check", DEVICE)
def test_a_vsa_kernel_rounds_as_its_reference_on_the_device(check):
    on_device(FAMILY / check)
