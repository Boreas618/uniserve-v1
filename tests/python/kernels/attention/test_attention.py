"""The attention and KV-write packages' kernels against their references.

The Triton paged attention and the KV writes run on the interpreter, and again
compiled for a device. The dense, sparse and trtllm attention backends are
native modules: their checks load a built module (``UNISERVE_KERNEL_MODULES``)
and launch it on a device.
"""

import pathlib

import pytest

from tests.python.kernels.checks import interpreted, module, on_device

pytestmark = pytest.mark.unit

FAMILY = pathlib.Path(__file__).resolve().parent

#: The checks that run whole on Triton's interpreter.
INTERPRETED = (
    "paged_triton_decode_check.py",
    "paged_triton_prefill_check.py",
    "kv_write_check.py",
    "kv_write_scaled_check.py",
)


@pytest.mark.parametrize("check", INTERPRETED)
def test_an_attention_kernel_matches_its_reference_on_the_interpreter(check):
    interpreted(FAMILY / check)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "check",
    [
        "paged_triton_decode_check.py",
        "paged_triton_prefill_check.py",
        "kv_write_scaled_check.py",
    ],
)
def test_the_triton_paged_attention_matches_its_reference_on_the_device(check):
    """The same cases, with the device's rounding and its concurrent programs.

    The interpreter runs one program after another, so two programs that race
    on a page of the scaled write show only here.
    """
    on_device(FAMILY / check, "--device")


@pytest.mark.gpu
def test_the_kv_write_rounds_into_every_cache_type_on_the_device():
    """The roundings into a BF16 cache need a device.

    The interpreter truncates a cast into BF16.
    """
    on_device(FAMILY / "kv_write_check.py", "--device")


@pytest.mark.gpu
@pytest.mark.parametrize("mask", ["causal", "dense"])
def test_the_f32_dense_attention_matches_its_reference(mask):
    on_device(
        FAMILY / "dense_f32_check.py",
        "--mask",
        mask,
        module("dense_attention", "f32", mask),
    )


@pytest.mark.gpu
def test_the_f32_dense_attention_refuses_a_call_its_module_was_not_built_for():
    on_device(
        FAMILY / "dense_f32_refusal_check.py",
        module("dense_attention", "f32", "causal"),
        module("dense_attention", "f32", "dense"),
    )


@pytest.mark.gpu
def test_the_fa4_dense_attention_matches_its_reference():
    on_device(FAMILY / "dense_fa4_check.py", module("dense_attention", "fa4"))


@pytest.mark.gpu
def test_the_trtllm_paged_decode_matches_its_reference():
    on_device(
        FAMILY / "paged_trtllm_decode_check.py",
        module("paged_attention", "trtllm"),
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("check", "backend", "block"),
    [
        ("sparse_sm90_check.py", "wgmma", "128"),
        ("sparse_sm100_check.py", "tcgen05", "64"),
        ("sparse_sm120_check.py", "flashinfer", "64"),
    ],
)
def test_a_sparse_attention_matches_its_reference(check, backend, block):
    on_device(
        FAMILY / check,
        module("sparse_attention", backend),
        "--block",
        block,
    )
