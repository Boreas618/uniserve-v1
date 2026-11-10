"""The sampling packages' kernels against their references.

``sampling_check.py`` is one check for the nine operators of the sampling
path, named by ``--op``. Two of them round with operations the interpreter
does not reproduce (``filter_candidates``' mixed comparison and
``logits_shape``'s libdevice rounding); on the interpreter their check says so
and compares the rest, and the device run compares them whole.
"""

import pathlib

import pytest

from tests.python.kernels.checks import interpreted, on_device

pytestmark = pytest.mark.unit

FAMILY = pathlib.Path(__file__).resolve().parent

#: The operators `sampling_check.py` covers.
SAMPLING = (
    "logits_gather",
    "vocab_unpad",
    "logits_shape",
    "filter_candidates",
    "inverse_cdf",
    "argmax",
    "logprob_gather",
    "top_k",
    "top_logprobs",
)

#: The checks of one operator that run whole on Triton's interpreter.
INTERPRETED = (
    "candidate_order_check.py",
    "slot_commit_check.py",
    "posterior_check.py",
    "philox_noise_check.py",
)


@pytest.mark.parametrize("op", SAMPLING)
def test_a_sampling_step_matches_its_reference_on_the_interpreter(op):
    interpreted(FAMILY / "sampling_check.py", "--op", op)


@pytest.mark.parametrize("check", INTERPRETED)
def test_a_sampling_kernel_matches_its_reference_on_the_interpreter(check):
    interpreted(FAMILY / check)


@pytest.mark.gpu
@pytest.mark.parametrize("op", SAMPLING)
def test_a_sampling_step_matches_its_reference_on_the_device(op):
    on_device(FAMILY / "sampling_check.py", "--op", op)
