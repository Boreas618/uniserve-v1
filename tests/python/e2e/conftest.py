"""Options of the end-to-end tests."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Name the inputs a test cannot find for itself."""
    parser.addoption(
        "--other-weights",
        default=None,
        help=(
            "a copy of the Qwen3-MoE checkpoint's files at another path: the "
            "other weights an update switches to"
        ),
    )


@pytest.fixture
def other_weights(pytestconfig: pytest.Config) -> str:
    """The copy of the checkpoint ``--other-weights`` names.

    The tests that switch to other weights skip without it: a copy at another
    path is what makes the engine read the same files as other weights.
    """
    named = pytestconfig.getoption("--other-weights")
    if not named:
        pytest.skip("--other-weights names no copy of the checkpoint")
    return str(named)
