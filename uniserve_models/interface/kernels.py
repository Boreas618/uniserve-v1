"""Kernel provider choices made by the numerical library."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from uniserve_models.backends import kernel_choices


def functions() -> dict[str, Callable[..., Any]]:
    return {"kernels.choices": choices}


def choices() -> int:
    """How many kernel choices this process has made so far.

    The worker compares the counter before and after a batch to detect
    kernels selected while serving.
    """
    return kernel_choices()
