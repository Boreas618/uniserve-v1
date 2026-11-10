"""The row and frame normalization packages' kernels against their references.

A check shared by the operators of one computation names its operator with
``--op``: ``row_norm_check.py`` covers the plain and residual RMS norms,
``fused_row_norm_check.py`` the scaled-residual RMS and layer norms, and
``modulation_check.py`` the modulated norm and the gated residual.
"""

import pathlib

import pytest

from tests.python.kernels.checks import interpreted, module, on_device

pytestmark = pytest.mark.unit

FAMILY = pathlib.Path(__file__).resolve().parent

#: Each shared check and the operators it covers.
SHARED = (
    ("row_norm_check.py", "rms_norm"),
    ("row_norm_check.py", "add_rms_norm"),
    ("fused_row_norm_check.py", "scaled_residual_rms_norm"),
    ("fused_row_norm_check.py", "scaled_residual_layer_norm"),
    ("modulation_check.py", "modulated_rms_norm"),
    ("modulation_check.py", "gated_residual"),
)

#: The checks of one operator that run whole on Triton's interpreter.
INTERPRETED = (
    "layer_norm_check.py",
    "weighted_rms_norm_check.py",
    "scaled_residual_check.py",
)

#: The native frame norms: their host half needs no module and no device.
FRAME = (
    ("frame_moments_check.py", "frame_moments"),
    ("frame_norm_pad_check.py", "frame_norm_pad"),
)


@pytest.mark.parametrize(("check", "op"), SHARED)
def test_a_row_norm_matches_its_reference_on_the_interpreter(check, op):
    interpreted(FAMILY / check, "--op", op)


@pytest.mark.parametrize("check", INTERPRETED)
def test_a_norm_matches_its_reference_on_the_interpreter(check):
    interpreted(FAMILY / check)


@pytest.mark.parametrize(("check", "op"), FRAME)
def test_a_frame_norm_states_a_consistent_call_structure(check, op):
    interpreted(FAMILY / check, "--cpu-only")


@pytest.mark.gpu
@pytest.mark.parametrize("op", ["modulated_rms_norm", "gated_residual"])
def test_a_modulation_stores_its_narrow_output_on_the_device(op):
    """The interpreter stores E4M3 wrong, so the E4M3 output is checked here."""
    on_device(FAMILY / "modulation_check.py", "--op", op)


@pytest.mark.gpu
@pytest.mark.parametrize(("check", "op"), FRAME)
def test_a_frame_norm_module_matches_its_reference(check, op):
    on_device(FAMILY / check, module(op, "nvcc"))
