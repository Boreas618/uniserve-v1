"""Worker rank entry: load the Rust worker library and run it.

The engine starts a rank as ``python -m uniserve_models.worker``. This entry
configures logging and diagnostic dumps, loads the worker library through
TVM-FFI and hands it the numerical interface table. Everything after that
is the library's: descriptor parsing, registration, model loading,
serving and shutdown. The process exits with the status it returns.
"""

from __future__ import annotations

import faulthandler
import importlib.util
import logging
import os
import signal
import sys


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
    # would otherwise raise inside an interface call on this thread.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    import tvm_ffi

    from uniserve_models import interface

    spec = importlib.util.find_spec("uniserve_models._worker")
    if spec is None or spec.origin is None:
        sys.stderr.write(
            "uniserve-worker: error: the worker library "
            "uniserve_models._worker is not installed\n"
        )
        sys.stderr.flush()
        os._exit(1)
    library = tvm_ffi.load_module(spec.origin)
    # The table must not initialize CUDA; the library selects devices.
    table = interface.table()
    code = int(library["main"](sys.argv[1:], table))
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip interpreter finalization: an aborted rank retains device
    # resources, and finalizers could wait on a failed peer.
    os._exit(code)


if __name__ == "__main__":
    main()
