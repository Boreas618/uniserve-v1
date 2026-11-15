"""The fixture the tensor primitives' checks launch through."""

import pytest
import torch
from uniserve_testing.kernels.packages import Launches
from uniserve_testing.kernels.tensor.operands import Primitives


@pytest.fixture(scope="session")
def primitives(
    torch_device: torch.device,
    interpreted: bool,
    stated_launch: Launches,
    launch_target: str,
) -> Primitives:
    """The session's device and the launches the packages state for keys on
    the session's launch target.
    """  # noqa: D205
    return Primitives(torch_device, interpreted, stated_launch, launch_target)
