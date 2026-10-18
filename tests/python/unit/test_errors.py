"""Error kinds that cross the TVM-FFI interface."""

import pytest
import torch
import tvm_ffi

from uniserve import errors

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (errors.SetupError("placement"), "UnsupportedSetup"),
        (torch.OutOfMemoryError("CUDA out of memory"), "OutOfMemory"),
        (NotImplementedError("call"), "UnsupportedCall"),
        (KeyError("slot"), "InputError"),
        (ValueError("shape"), "InputError"),
        (AssertionError("invariant"), "InvariantViolation"),
        (FileNotFoundError("checkpoint"), "FileNotFound"),
        (RuntimeError("kernel"), "ComputeError"),
    ],
)
def test_failures_cross_the_interface_with_their_kind(error, kind):
    def raises():
        raise error

    # Calling through a TVM-FFI function sends the error through the C ABI,
    # where the worker reads its kind.
    function = tvm_ffi.convert(errors.translate(raises))
    with pytest.raises(Exception) as raised:
        function()
    assert type(raised.value).__name__ == kind
    assert str(raised.value).startswith(f"{type(error).__name__}: ")


def test_translation_keeps_results_and_the_original_cause():
    assert errors.translate(lambda value: value + 1)(1) == 2

    def raises():
        raise ValueError("bad shape")

    with pytest.raises(errors.InputError) as raised:
        errors.translate(raises)()
    assert isinstance(raised.value.__cause__, ValueError)
    assert raised.value.__traceback__ is not None
