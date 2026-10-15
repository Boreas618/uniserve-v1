"""Worker rank entry: load the Rust worker library and run it.

The engine starts a rank as ``python -m uniserve.worker``. This entry
configures logging and diagnostic dumps, loads the worker library through
TVM-FFI and hands it :func:`import_path`, through which the library reaches
every function of the numerical library by its import path. Everything
after that is the library's: descriptor parsing, registration, model
loading, serving and shutdown. The process exits with the status it
returns.
"""

from __future__ import annotations

import faulthandler
import functools
import importlib.util
import logging
import os
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

    The rank runtime reaches the library by import path, for example
    ``uniserve.runner.causal_lm.CausalLMRunner.decode``: a function, a class
    (called to construct it) or a method (called with the object first).
    The runtime's storage arrives as DLPack tensors, which the callable
    receives as torch tensors aliasing the same memory, as a Python caller
    would pass them. A failure of the callable crosses with the error kinds
    of :mod:`uniserve.errors`.

    Resolving a path imports its module and never initializes CUDA: the
    runtime selects devices and contexts before any CUDA call.
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Dump every thread's Python stack for a wedged rank. The handler is
    # C-level and runs while the library holds no GIL.
    for name in ("SIGQUIT", "SIGUSR1"):
        number = getattr(signal, name, None)
        if number is not None:
            faulthandler.register(number, file=sys.stderr, all_threads=True)
    # The library owns SIGINT and SIGTERM. Python's KeyboardInterrupt handler
    # would otherwise raise inside a library call on this thread.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    spec = importlib.util.find_spec("uniserve._worker")
    if spec is None or spec.origin is None:
        sys.stderr.write(
            "uniserve-worker: error: the worker library "
            "uniserve._worker is not installed\n"
        )
        sys.stderr.flush()
        os._exit(1)
    library = tvm_ffi.load_module(spec.origin)
    code = int(library["main"](sys.argv[1:], import_path))
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip interpreter finalization: an aborted rank retains device
    # resources, and finalizers could wait on a failed peer.
    os._exit(code)


if __name__ == "__main__":
    main()
