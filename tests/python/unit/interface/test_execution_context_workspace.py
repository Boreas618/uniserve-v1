"""Execution-context workspace: standalone, declared and bound backing."""

import pytest
import torch

from uniserve_models.bindings.context import (
    BoundWorkspace,
    DeclaredWorkspace,
    ExecutionContext,
    TensorBuffers,
)
from uniserve_models.tensors import BufferConfig

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


def test_declared_requirements_are_served_by_the_bound_tensors():
    model = Scaled()
    declared = DeclaredWorkspace()
    with ExecutionContext(model, workspace=declared) as context:
        context.prepare(3)
        requirements = {item.name: item for item in declared.requirements}
    assert {
        name: (item.role, item.dtype, item.shape)
        for name, item in requirements.items()
    } == {
        "0000/scale": (None, torch.float32, (3,)),
        "0001/rows": (None, torch.int64, (2, 3)),
    }

    # The bound context computes with the caller's storage.
    tensors = {
        "0000/scale": torch.zeros(3),
        "0001/rows": torch.zeros((2, 3), dtype=torch.int64),
    }
    with ExecutionContext(model, workspace=BoundWorkspace(tensors)) as context:
        context.prepare(3)
        assert context.constants["scale"].data_ptr() == (
            tensors["0000/scale"].data_ptr()
        )
        assert context.workspace_views["rows"].data_ptr() == (
            tensors["0001/rows"].data_ptr()
        )
        torch.testing.assert_close(
            model(torch.ones(3), **context.constants), torch.full((3,), 3.0)
        )
    # Bound storage belongs to the caller and keeps its values.
    torch.testing.assert_close(tensors["0000/scale"], torch.full((3,), 3.0))


def test_shared_scratch_is_declared_once_per_role_at_its_largest_extent():
    declared = DeclaredWorkspace()
    for width in (2, 5, 3):
        declared.scratch(
            "matmul", {"work": BufferConfig((4, width), torch.uint8)}, "cpu"
        )
    (requirement,) = declared.requirements
    assert (requirement.name, requirement.role, requirement.shape) == (
        "matmul/work",
        "matmul",
        (4, 5),
    )
    bound = BoundWorkspace(
        {"matmul/work": torch.zeros((4, 5), dtype=torch.uint8)}
    )
    view = bound.scratch(
        "matmul", {"work": BufferConfig((4, 3), torch.uint8)}, "cpu"
    )["work"]
    assert view.shape == (4, 3)


def test_attention_storage_declares_symmetric_fields_with_their_members():
    from uniserve_models.distributed import Communicator
    from uniserve_models.nn.attention.storage import StorageRequirement

    requirement = StorageRequirement(
        Communicator((3, 5), 0, "ulysses"),
        {
            "output": BufferConfig((2, 4), torch.float32),
            "sync_input": BufferConfig((1,), torch.int32),
        },
        ("output",),
        (),
    )
    declared = DeclaredWorkspace()
    tensors, peers = declared.storage(requirement)
    assert set(tensors) == {"output", "sync_input"}
    assert len(peers["output"]) == 2
    assert {item.name: item.symmetric for item in declared.requirements} == {
        "0000/output": (3, 5),
        "0000/sync_input": None,
    }

    # The bound walk serves the local tensors and every member's peer view.
    supplied = {
        "0000/output": torch.zeros((2, 4)),
        "0000/output.peer0": torch.zeros((2, 4)),
        "0000/output.peer1": torch.ones((2, 4)),
        "0000/sync_input": torch.zeros(1, dtype=torch.int32),
    }
    tensors, peers = BoundWorkspace(supplied).storage(requirement)
    assert tensors["output"].data_ptr() == supplied["0000/output"].data_ptr()
    assert [peer.data_ptr() for peer in peers["output"]] == [
        supplied["0000/output.peer0"].data_ptr(),
        supplied["0000/output.peer1"].data_ptr(),
    ]
