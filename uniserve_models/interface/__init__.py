"""The numerical interface table the Rust worker calls through TVM-FFI.

``table()`` maps ``<area>.<verb>`` names to plain functions. Each area module
exposes ``functions()`` with its entries; a capability runtime adds its
module to ``_AREAS``. Every entry is wrapped by ``errors.translate`` so a
failure crosses the interface with a fixed error kind.

Building the table imports the numerical library but never initializes
CUDA: the worker selects devices and contexts before any CUDA call.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from .errors import translate

# Interface modules in this package, each exposing ``functions()``.
_AREAS = (
    "loading",
    "cache",
    "binding",
    "distributed",
    "memory",
    "kernels",
    "profiling",
    "runtime",
)


def table() -> dict[str, Callable[..., Any]]:
    """Every interface entry by name.

    Raises:
        ValueError: Two areas declare the same entry.
    """
    entries: dict[str, Callable[..., Any]] = {}
    for area in _AREAS:
        for name, function in (
            import_module(f"{__name__}.{area}").functions().items()
        ):
            if name in entries:
                raise ValueError(f"interface entry {name!r} is declared twice")
            entries[name] = translate(function)
    return entries
