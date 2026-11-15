"""The target the video sparse attention checks state their launches for."""

import pytest

#: The target an interpreted launch is stated for. The interpreter compiles
#: for no target and ignores what a target selects; the packages these checks
#: state launches of have no dependent launch, so every target states the same
#: kernel, compile-time values and options, and this one, which has no
#: programmatic dependent launch either, states them.
INTERPRETED_TARGET = "sm_90a"


@pytest.fixture(scope="session")
def launch_target(interpreted: bool, request: pytest.FixtureRequest) -> str:
    """The target a check's launch is stated for: the session device's, or on
    the interpreter :data:`INTERPRETED_TARGET`.
    """  # noqa: D205
    if interpreted:
        return INTERPRETED_TARGET
    return request.getfixturevalue("module_target")
