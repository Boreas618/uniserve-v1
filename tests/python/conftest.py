"""Collection rules every Python test of the repository shares."""

from __future__ import annotations

import os

import pytest

#: The variable that opts a run into the GPU end-to-end tests.
GPU_E2E_GATE = "UNISERVE_RUN_GPU_E2E"


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip the GPU end-to-end tests unless ``UNISERVE_RUN_GPU_E2E=1``.

    A GPU end-to-end case starts real servers on the host's GPUs and reads
    checkpoints the host provides, so a run asks for them explicitly: the GPU
    recipes (``just test-python-gpu``, ``just test-python-sanitizer``) and the
    nightly workflows set the variable, and a plain ``pytest`` reports the
    cases skipped, naming it.
    """
    if os.environ.get(GPU_E2E_GATE) == "1":
        return
    skip = pytest.mark.skip(
        reason=f"{GPU_E2E_GATE}=1 runs the GPU end-to-end tests"
    )
    for item in items:
        if "e2e" in item.keywords and "gpu" in item.keywords:
            item.add_marker(skip)
