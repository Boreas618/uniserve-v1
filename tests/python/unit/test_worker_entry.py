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


def passed_on(value):
    """What a library call receives when the runtime passes ``value`` to it.

    The runtime holds the results of its calls and passes them to later
    calls through TVM-FFI; a TVM-FFI function made from a Python callable
    stands for such a later call.
    """
    received = []
    tvm_ffi.convert(received.append)(value)
    return received[0]


def test_a_path_resolves_to_a_function_a_class_or_a_method():
    module = f"{__name__}"
    construct = import_path(f"{module}.Doubler")
    apply = import_path(f"{module}.Doubler.apply")
    doubler = construct(2)
    assert isinstance(doubler, Doubler)
    values = torch.arange(4, dtype=torch.float32)
    torch.testing.assert_close(passed_on(apply(doubler, values)), values * 2)


def test_wrapped_storage_reaches_every_call_as_one_tensor_aliasing_it():
    storage = torch.zeros(3, dtype=torch.float32)
    # The runtime wraps a view of its storage once, through the import path.
    wrap = import_path("torch.from_dlpack")
    handle = wrap(tvm_ffi.from_dlpack(storage))

    first, second = passed_on(handle), passed_on(handle)
    assert isinstance(first, torch.Tensor)
    assert first is second
    first.fill_(5.0)
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
