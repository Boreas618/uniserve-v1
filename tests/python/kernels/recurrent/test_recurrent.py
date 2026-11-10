"""The recurrent-state packages' kernels against their references.

``causal_conv1d_check.py`` and ``gated_delta_rule_check.py`` run a batch of
rows that start fresh, continue from their slots' states, keep checkpoints and
pad, at small widths on the interpreter and at the Qwen3.6 widths on a device.
"""

import pathlib

import pytest

from tests.python.kernels.checks import interpreted, on_device

pytestmark = pytest.mark.unit

FAMILY = pathlib.Path(__file__).resolve().parent

#: The checks, each run once per element type.
CHECKS = ("causal_conv1d_check.py", "gated_delta_rule_check.py")


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize("check", CHECKS)
def test_a_recurrent_operator_matches_its_reference_on_the_interpreter(
    check, dtype
):
    interpreted(FAMILY / check, "--dtype", dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize("check", CHECKS)
def test_a_recurrent_operator_matches_its_reference_on_the_device(
    check, dtype
):
    """The model's widths, and the device's own stores and dot products."""
    on_device(FAMILY / check, "--dtype", dtype)
