"""Interpreter state of a serving rank."""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any


def functions() -> dict[str, Callable[..., Any]]:
    return {"runtime.freeze_gc": freeze_gc}


def freeze_gc() -> None:
    """Collect, freeze and disable the cyclic garbage collector.

    Called once after warmup: resident models and graphs are permanent, and
    a collection inside a serving step would traverse them.
    """
    gc.collect()
    gc.freeze()
    gc.disable()
