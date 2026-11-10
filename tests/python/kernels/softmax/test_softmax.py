"""The softmax packages' kernels against their reference.

``softmax_check.py`` covers the softmax and its logarithm, named by ``--op``.
"""

import pathlib

import pytest

from tests.python.kernels.checks import interpreted

pytestmark = pytest.mark.unit

FAMILY = pathlib.Path(__file__).resolve().parent


@pytest.mark.parametrize("op", ["softmax", "log_softmax"])
def test_a_softmax_matches_its_reference_on_the_interpreter(op):
    interpreted(FAMILY / "softmax_check.py", "--op", op)
