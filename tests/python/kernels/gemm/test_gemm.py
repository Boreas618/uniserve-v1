"""The GEMM and NVFP4 quantization packages against their references.

The NVFP4 checks have a host half that needs no module and no device. The FP8
product is a Triton program, launched from its source on a device. Every other
GEMM is a native module, checked with a built one
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


@pytest.mark.gpu
def test_the_fp8_gemm_matches_its_reference_on_the_device():
    """The interpreter does not store E4M3 as a device does, so the product is checked here."""
    on_device(FAMILY / "fp8_gemm_check.py")
