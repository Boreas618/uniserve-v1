"""The mixture-of-experts routing and exchange packages against references.

The router (``topk_softmax``), the sort, the combine and the exchange are
compiled modules: their checks read a
built module (``UNISERVE_KERNEL_MODULES``). The router's module half checks
its build constants and published signature without a device; its launch half
loads the cubin through the CUDA driver and launches it on the fixed input,
plain and through a reversed expert map.
"""

import pathlib

import pytest

from tests.python.kernels.checks import (
    interpreted,
    module,
    on_device,
    require_devices,
    run,
)

pytestmark = pytest.mark.unit

FAMILY = pathlib.Path(__file__).resolve().parent

#: The router descriptor the module directory holds: the Qwen3-MoE routing,
#: stated because the directory's name is not the module's descriptor key.
ROUTER = ("--experts", "128", "--top", "8", "--dtype", "bf16")
ROUTER += ("--renormalize", "true")


def test_the_router_module_states_its_build_and_signature():
    interpreted(
        FAMILY / "topk_softmax_check.py",
        "--module",
        module("topk_softmax", "nvcc", device=False),
        *ROUTER,
    )


@pytest.mark.gpu
def test_the_router_routes_the_pinned_input_on_the_device():
    on_device(
        FAMILY / "topk_softmax_check.py",
        "--module",
        module("topk_softmax", "nvcc"),
        *ROUTER,
        "--launch",
    )


@pytest.mark.gpu
def test_the_moe_sort_matches_its_reference():
    on_device(FAMILY / "moe_sort_check.py", module("moe_sort", "trtllm"))


@pytest.mark.gpu
def test_the_moe_combine_matches_its_reference():
    on_device(
        FAMILY / "moe_unpermute_check.py", module("moe_unpermute", "trtllm")
    )


@pytest.mark.gpu
def test_the_moe_combine_check_reads_red_for_each_plausible_wrong_combine():
    on_device(FAMILY / "moe_unpermute_mutations.py")


@pytest.mark.gpu
def test_the_alltoall_exchange_matches_its_reference_across_two_devices():
    require_devices(2)
    run(
        FAMILY / "expert_exchange_check.py",
        (module("expert_exchange", "alltoall"), "0", "1"),
        device=True,
    )
