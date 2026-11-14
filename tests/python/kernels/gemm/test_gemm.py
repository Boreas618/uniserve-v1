"""The GEMM and NVFP4 quantization packages against their references.

The NVFP4 checks have a host half that needs no module and no device. The FP8
encode and product and the measured factors (``dynamic_scale``) are Triton
programs, launched from their source on a device; the factors also on the
interpreter.
Every other GEMM is a native module, checked with a built one
(``UNISERVE_KERNEL_MODULES``) on a device.
"""

import pathlib

import pytest

from tests.python.kernels.checks import interpreted, module, on_device

pytestmark = pytest.mark.unit

FAMILY = pathlib.Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "check", ["nvfp4_gemm_check.py", "nvfp4_quantize_check.py"]
)
def test_an_nvfp4_kernel_states_a_consistent_host_side(check):
    interpreted(FAMILY / check, "--cpu-only")


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("check", "op", "backend"),
    [
        ("cublas_lt_gemm_check.py", "gemm", "cublas_lt"),
        ("nvfp4_gemm_check.py", "nvfp4_gemm", "cute_dsl"),
        ("gather_gemm_check.py", "gather_gemm", "cute_dsl"),
        ("route_gemm_check.py", "route_gemm", "cute_dsl"),
    ],
)
def test_a_gemm_module_matches_its_reference(check, op, backend):
    on_device(FAMILY / check, module(op, backend))


@pytest.mark.gpu
def test_the_nvfp4_quantization_feeds_the_nvfp4_gemm():
    on_device(
        FAMILY / "nvfp4_quantize_check.py",
        module("nvfp4_quantize", "triton"),
        "--gemm",
        module("nvfp4_gemm", "cute_dsl"),
    )


def test_the_measured_factors_state_their_formula_on_the_interpreter():
    interpreted(FAMILY / "dynamic_scale_check.py", "--cpu-only")


@pytest.mark.gpu
def test_a_measured_fp8_projection_is_within_its_formats_bound():
    on_device(FAMILY / "dynamic_scale_check.py")


@pytest.mark.gpu
def test_the_fp8_gemm_matches_its_reference_on_the_device():
    """The interpreter does not store E4M3 as a device does.

    The product is therefore checked here.
    """
    on_device(FAMILY / "fp8_gemm_check.py")


#: The operators whose dependent launch `pdl_chain_check.py` checks.
DEPENDENT = ("nvfp4_quantize",)


@pytest.mark.gpu
@pytest.mark.parametrize("op", DEPENDENT)
def test_a_dependent_launch_reads_what_its_producer_wrote(op):
    on_device(FAMILY / "pdl_chain_check.py", "--op", op)
