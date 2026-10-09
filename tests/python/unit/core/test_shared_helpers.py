"""Behavior tests for shared worker helpers."""

from __future__ import annotations

import pytest

from uniserve_models.env import flag_from_value

pytestmark = pytest.mark.unit


def test_flag_from_value_allowlist_semantics():
    for token in ("1", "true", "TRUE", "Yes", "on", "  on  "):
        assert flag_from_value(token) is True, token
    for token in ("0", "false", "no", "off"):
        assert flag_from_value(token) is False, token
    # Unset / empty -> default; unrecognized -> default (conservative).
    assert flag_from_value(None) is False
    assert flag_from_value("") is False
    assert flag_from_value("garbage") is False
    assert flag_from_value(None, default=True) is True
    assert flag_from_value("garbage", default=True) is True
    # Explicit false tokens always win over a True default.
    assert flag_from_value("0", default=True) is False
