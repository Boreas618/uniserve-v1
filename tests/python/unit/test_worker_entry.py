"""The function through which the rank runtime reaches the library."""

import subprocess
import sys

import pytest
import torch
import tvm_ffi

from uniserve import errors
from uniserve.worker import import_path

pytestmark = pytest.mark.unit


class Doubler:
    """A class the runtime constructs and whose method it calls."""

    def __init__(self, scale: int) -> None:
        self.scale = scale

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale


def test_a_path_resolves_to_a_function_a_class_or_a_method():
    module = f"{__name__}"
    construct = import_path(f"{module}.Doubler")
    apply = import_path(f"{module}.Doubler.apply")
    doubler = construct(2)
    assert isinstance(doubler, Doubler)
    values = torch.arange(4, dtype=torch.float32)
    torch.testing.assert_close(apply(doubler, values), values * 2)


def test_ffi_tensors_arrive_as_torch_tensors_aliasing_their_memory():
    storage = torch.zeros(3, dtype=torch.float32)
    fill = import_path("torch.Tensor.fill_")
    fill(tvm_ffi.from_dlpack(storage), 5.0)
    assert storage.tolist() == [5.0, 5.0, 5.0]


def test_a_failure_crosses_with_its_error_kind():
    construct = import_path(f"{__name__}.Doubler")
    with pytest.raises(errors.InputError, match="TypeError"):
        construct()


def test_resolving_startup_paths_does_not_initialize_cuda():
    # A fresh interpreter: an earlier test could have initialized CUDA.
    code = (
        "import torch\n"
        "from uniserve.worker import import_path\n"
        "for path in (\n"
        "    'uniserve.distributed.world.initialize',\n"
        "    'uniserve.cuda_graph.pool',\n"
        "    'uniserve.backends.kernel_choices',\n"
        "    'uniserve.profiling.start_profiler',\n"
        "    'uniserve.loading.rank.read_config',\n"
        "    'gc.freeze',\n"
        "):\n"
        "    import_path(path)\n"
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
