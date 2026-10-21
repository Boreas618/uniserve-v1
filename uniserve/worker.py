"""The Python side of a worker: what the worker library calls first.

A worker is the ``uniserve-worker`` executable, which hosts the interpreter
on a Python thread of its own and calls :func:`attach` once it is up.
``attach`` configures logging and diagnostic dumps and registers
:func:`import_path`, through which the library reaches every function of the
numerical library by its import path. Everything else is the library's:
descriptor parsing, registration, model loading, serving and shutdown.

The same library exports the video request rules of the ``models`` crate,
which the numerical library plans MiniMax-H3 requests with
(:func:`library`).
"""

from __future__ import annotations

import faulthandler
import functools
import importlib.util
import logging
import pkgutil
import signal
import sys
from collections.abc import Callable
from typing import Any

import torch
import tvm_ffi

from uniserve.errors import translate


def import_path(path: str) -> Callable[..., Any]:
    """The callable at a dotted import path.

    The worker runtime reaches the library by import path, for example
    ``uniserve.runner.causal_lm.CausalLMRunner.decode``: a function, a class
    (called to construct it) or a method (called with the object first).
    A torch tensor the callable returns crosses as an opaque Python object,
    which the runtime holds as a handle and passes back as that same tensor.
    The runtime wraps each view of its storage this way once, by calling
    ``torch.from_dlpack`` through this function, so every call that takes
    the view receives one torch tensor aliasing the runtime's memory, as a
    Python caller would pass it. A failure of the callable crosses with the
    error kinds of :mod:`uniserve.errors`.

    Resolving a path imports its module and never initializes CUDA: the
    runtime selects devices and contexts before any CUDA call.
    """
    function = pkgutil.resolve_name(path)

    @functools.wraps(function)
    def call(*arguments: Any) -> Any:
        result = function(*arguments)
        if isinstance(result, torch.Tensor):
            # Returned as is, the tensor would cross as a TVM-FFI tensor, a
            # DLPack import that every later call would convert back into a
            # new torch tensor. An opaque Python object
            # (kTVMFFIOpaquePyObject) crosses unconverted: a callable it is
            # passed to receives this tensor itself. tvm_ffi.convert makes
            # the same object for values TVM-FFI has no type of its own for.
            return tvm_ffi.core._convert_to_opaque_object(result)
        return result

    return translate(call)


@functools.cache
def library() -> tvm_ffi.Module:
    """The worker library ``uniserve._worker``, loaded once.

    Besides the worker entry it exports the video canvas and frame-count
    rules (``video_canvas`` and the functions beside it), which any process
    may call: loading the library starts nothing. In a worker process this
    is the library the ``uniserve-worker`` executable already loaded.

    Raises:
        ImportError: The library is not installed.
    """
    spec = importlib.util.find_spec("uniserve._worker")
    if spec is None or spec.origin is None:
        raise ImportError(
            "the worker library uniserve._worker is not installed"
        )
    return tvm_ffi.load_module(spec.origin)


def attach() -> None:
    """Prepare the interpreter a worker hosts, before the library calls it.

    Configures logging, dumps every thread's Python stack on SIGQUIT and
    SIGUSR1 for a wedged worker (a C-level handler that runs while the
    library holds no GIL), and registers :func:`import_path` as the TVM-FFI
    global function ``uniserve.worker.import_path``, which the library looks
    up. The interpreter installs no other signal handler: the library owns
    SIGINT and SIGTERM.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for name in ("SIGQUIT", "SIGUSR1"):
        number = getattr(signal, name, None)
        if number is not None:
            faulthandler.register(number, file=sys.stderr, all_threads=True)
    tvm_ffi.register_global_func("uniserve.worker.import_path", import_path)
