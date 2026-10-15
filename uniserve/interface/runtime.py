"""Interpreter state of a serving rank."""

from __future__ import annotations

import functools
import gc
import pkgutil
from collections.abc import Callable
from typing import Any

import torch
import tvm_ffi

from .errors import translate


def functions() -> dict[str, Callable[..., Any]]:
    return {"runtime.freeze_gc": freeze_gc, "runtime.import": import_path}


def freeze_gc() -> None:
    """Collect, freeze and disable the cyclic garbage collector.

    Called once after warmup: resident models and graphs are permanent, and
    a collection inside a serving step would traverse them.
    """
    gc.collect()
    gc.freeze()
    gc.disable()


def import_path(path: str) -> Callable[..., Any]:
    """The callable at a dotted import path.

    The rank runtime reaches the library by import path, for example
    ``uniserve.runner.causal_lm.CausalLMRunner.decode``: a function,
    a class (called to construct it) or a method (called with the object
    first). The runtime's storage arrives as DLPack tensors, which the
    callable receives as torch tensors aliasing the same memory, as a
    Python caller would pass them. A failure of the callable crosses with
    the interface's error kinds.
    """
    function = pkgutil.resolve_name(path)

    @functools.wraps(function)
    def call(*arguments: Any) -> Any:
        # FIXME: every call converts its tensor arguments. TVM-FFI turns a
        # torch tensor returned to the runtime back into an FFI tensor, so
        # the runtime cannot hold the converted tensor as a handle and pass
        # it again.
        return function(
            *(
                torch.from_dlpack(argument)
                if isinstance(argument, tvm_ffi.Tensor)
                else argument
                for argument in arguments
            )
        )

    return translate(call)
