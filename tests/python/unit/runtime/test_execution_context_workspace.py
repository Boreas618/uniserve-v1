"""Execution-context workspace: borrowed and persistent buffers."""

import pytest
import torch

from uniserve.bindings.context import ExecutionContext, TensorBuffers
from uniserve.tensors import BufferConfig

pytestmark = pytest.mark.unit


class Scaled(torch.nn.Module):
    """A module declaring per-size constants and workspace."""

    def constant_buffers(self, size):
        return {"scale": BufferConfig((size,), torch.float32)}

    def workspace_buffers(self, size):
        return {"rows": BufferConfig((2, size), torch.int64)}

    def prepare_constants(self, size, *, out):
        if size == 0:
            raise ValueError("scale requires at least one value")
        out["scale"].fill_(size)

    def forward(self, values, *, scale):
        return values * scale


def test_buffers_lend_compact_leading_elements():
    storage = TensorBuffers.allocate(
        {"rows": BufferConfig((2, 3), torch.float32, capacity_shape=(4, 6))},
        device="cpu",
    )
    full = BufferConfig((4, 6), torch.float32)
    storage.view({"rows": full})["rows"].copy_(torch.arange(24).reshape(4, 6))
    views = storage.view({"rows": BufferConfig((2, 3), torch.float32)})
    torch.testing.assert_close(
        views["rows"], torch.arange(6).float().reshape(2, 3)
    )
    views["rows"].add_(10)
    expected = torch.arange(24).float()
    expected[:6] += 10
    torch.testing.assert_close(
        storage.view({"rows": full})["rows"], expected.reshape(4, 6)
    )
    with pytest.raises(ValueError, match="capacity"):
        storage.view({"rows": BufferConfig((5, 6), torch.float32)})
    incompatible = TensorBuffers.from_tensors(
        {"rows": torch.empty((4, 6), dtype=torch.int32)}
    )
    with pytest.raises(ValueError, match="dtype"):
        incompatible.view({"rows": BufferConfig((2, 3), torch.float32)})
    storage.close()
    with pytest.raises(RuntimeError, match="closed"):
        storage.view({"rows": full})


def test_failed_preparation_keeps_borrowed_buffers():
    model = Scaled()
    fields = model.constant_buffers(4)
    with TensorBuffers.allocate(fields, device="cpu") as buffers:
        with ExecutionContext(model) as context:
            context.prepare(4, constants=buffers)
            torch.testing.assert_close(
                model(torch.ones(4), **context.constants),
                torch.full((4,), 4.0),
            )
            with pytest.raises(ValueError, match="at least one value"):
                context.prepare(0, constants=buffers)
            assert not context.constants

            context.prepare(2, constants=buffers)
            torch.testing.assert_close(
                model(torch.ones(2), **context.constants),
                torch.full((2,), 2.0),
            )
        torch.testing.assert_close(
            buffers.view(fields)["scale"], torch.tensor([2.0, 2.0, 4.0, 4.0])
        )
