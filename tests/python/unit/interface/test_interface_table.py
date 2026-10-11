"""The numerical interface table the worker library receives."""

import subprocess
import sys

import pytest

from uniserve_models import interface

pytestmark = pytest.mark.unit

# Entries the worker resolves at startup; a missing one fails the rank.
STARTUP_ENTRIES = {
    "model.cache_plan",
    "model.reads_pages",
    "cache.bind_units",
    "cache.bind_conversion",
    "cache.pool",
    "graphs.pool",
    "component.prepare",
    "component.bind",
    "distributed.initialize",
    "distributed.create_groups",
    "distributed.mesh",
    "distributed.destroy",
    "kernels.choices",
    "profiling.start",
    "profiling.stop",
    "runtime.freeze_gc",
    "runtime.import",
}


# Entries an expert-parallel rank resolves; a missing one fails the rank.
EXPERT_ENTRIES = {
    "graphs.begin_capture",
    "graphs.end_capture",
    "graphs.release",
    "distributed.bind_collectives",
}


def test_table_provides_every_startup_entry_as_a_function():
    table = interface.table()
    assert STARTUP_ENTRIES | EXPERT_ENTRIES <= table.keys()
    assert all(callable(function) for function in table.values())
    assert all(name.count(".") == 1 for name in table)


def test_building_the_table_does_not_initialize_cuda():
    # A fresh interpreter: an earlier test could have initialized CUDA.
    code = (
        "import torch\n"
        "from uniserve_models import interface\n"
        "interface.table()\n"
        "print(torch.cuda.is_initialized())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_kernel_choices_count_selections():
    table = interface.table()
    assert isinstance(table["kernels.choices"](), int)
