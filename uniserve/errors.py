"""Failure kinds reported to the Rust worker through TVM-FFI.

The worker classifies a numerical failure by the TVM-FFI error kind, the
name of the exception class that crossed into it. Python's exception
hierarchy is not visible from Rust, so :func:`translate` maps every
exception a library function the worker calls raises onto a fixed
vocabulary of kinds, keeping the original type name in the message and the
original traceback.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import torch
import tvm_ffi

F = TypeVar("F", bound=Callable[..., Any])


class SetupError(Exception):
    """A deployment the model cannot serve: placement, devices or kernels."""


class KindError(Exception):
    """Base of the error kinds that cross the interface."""


def _kind(name: str) -> type[KindError]:
    # TVM-FFI reports the registered name of the exact raised type as the
    # error kind, and "RuntimeError" for an unregistered one.
    kind = type(name, (KindError,), {"__module__": __name__})
    return tvm_ffi.error.register_error(name, kind)


UnsupportedSetup = _kind("UnsupportedSetup")
OutOfMemory = _kind("OutOfMemory")
GraphError = _kind("GraphError")
UnsupportedCall = _kind("UnsupportedCall")
InputError = _kind("InputError")
InvariantViolation = _kind("InvariantViolation")
FileNotFound = _kind("FileNotFound")
ComputeError = _kind("ComputeError")


def kind(error: BaseException) -> type[KindError]:
    """The interface kind of a Python exception."""
    names = [cls.__name__ for cls in type(error).__mro__]
    if isinstance(error, KindError):
        return type(error)
    if isinstance(error, SetupError):
        return UnsupportedSetup
    if isinstance(error, torch.OutOfMemoryError) or any(
        "outofmemory" in name.lower() for name in names
    ):
        return OutOfMemory
    if "CUDAGraphError" in names:
        return GraphError
    if isinstance(error, NotImplementedError):
        return UnsupportedCall
    if isinstance(error, (KeyError, IndexError, TypeError, ValueError)):
        return InputError
    if isinstance(error, AssertionError):
        return InvariantViolation
    if isinstance(error, FileNotFoundError):
        return FileNotFound
    return ComputeError


def translate(function: F) -> F:
    """Wrap an interface function so its failures carry an interface kind.

    The raised kind's message is ``"{original type}: {message}"``; the
    original exception is its cause and its traceback is kept.
    """

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except KindError:
            raise
        except Exception as error:
            translated = kind(error)(f"{type(error).__name__}: {error}")
            raise translated.with_traceback(error.__traceback__) from error

    return wrapper  # type: ignore[return-value]
