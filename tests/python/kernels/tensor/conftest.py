"""The fixture the tensor primitives' checks launch through."""

import pytest
import torch

from tests.python.kernels.packages import Launches
from tests.python.kernels.tensor.operands import Primitives


@pytest.fixture(scope="session")
def primitives(
    torch_device: torch.device, interpreted: bool, stated_launch: Launches
) -> Primitives:
    """The session's device and the launches the packages state for keys."""
    return Primitives(torch_device, interpreted, stated_launch)
