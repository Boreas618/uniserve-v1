"""The cache, page and row movement kernels against their references."""

import pathlib

import pytest

from tests.python.kernels.checks import interpreted, on_device

pytestmark = pytest.mark.unit

FAMILY = pathlib.Path(__file__).resolve().parent

#: The checks that run on Triton's interpreter.
INTERPRETED = (
    "cache_block_scatter_check.py",
    "cache_block_snapshot_check.py",
    "cache_copy_region_check.py",
    "copy_block_check.py",
    "page_conversion_check.py",
    "embedding_check.py",
    "rows_check.py",
)


@pytest.mark.parametrize("check", INTERPRETED)
def test_a_movement_kernel_matches_its_reference_on_the_interpreter(check):
    interpreted(FAMILY / check)


@pytest.mark.gpu
def test_the_page_conversion_rounds_every_narrow_case_on_the_device():
    """The cases that decode E4M3 or round into a narrow type need a device."""
    on_device(FAMILY / "page_conversion_check.py", "--device")


#: The operators whose dependent launch `pdl_chain_check.py` checks.
DEPENDENT = (
    "rows",
    "cache_copy_region",
)


@pytest.mark.gpu
@pytest.mark.parametrize("op", DEPENDENT)
def test_a_dependent_launch_reads_what_its_producer_wrote(op):
    on_device(FAMILY / "pdl_chain_check.py", "--op", op)
