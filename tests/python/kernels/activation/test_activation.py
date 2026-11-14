"""The activation packages' kernels against their references.

``swiglu_check.py`` is one check for the two gated forms, which differ only in
which half of the packed row is the value; it names its operator with
``--op``.
"""

import pathlib

import pytest

from tests.python.kernels.checks import interpreted, on_device

pytestmark = pytest.mark.unit

FAMILY = pathlib.Path(__file__).resolve().parent

#: The checks that run whole on Triton's interpreter.
INTERPRETED = (
    "act_and_mul_check.py",
    "activation1d_check.py",
    "gelu_check.py",
    "silu_check.py",
    "snake_check.py",
)


@pytest.mark.parametrize("check", INTERPRETED)
def test_an_activation_matches_its_reference_on_the_interpreter(check):
    interpreted(FAMILY / check)


def test_the_sigmoid_gate_matches_its_reference_on_the_interpreter():
    """An attention's output gate: `act_and_mul` with the plain sigmoid."""
    interpreted(FAMILY / "act_and_mul_check.py", "--activation", "sigmoid")


@pytest.mark.parametrize("op", ["swiglu", "value_first_swiglu"])
def test_a_gated_activation_matches_its_reference_on_the_interpreter(op):
    interpreted(FAMILY / "swiglu_check.py", "--op", op)


#: The operators whose dependent launch `pdl_chain_check.py` checks.
DEPENDENT = ("silu",)


@pytest.mark.gpu
@pytest.mark.parametrize("op", DEPENDENT)
def test_a_dependent_launch_reads_what_its_producer_wrote(op):
    on_device(FAMILY / "pdl_chain_check.py", "--op", op)
