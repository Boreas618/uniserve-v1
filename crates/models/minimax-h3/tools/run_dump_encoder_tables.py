#!/usr/bin/env python3
"""Run the reference half of the encoder's host-table diagnosis on any host.

**This tool needs the retired Python tree.** It reads `uniserve`, the Python package that left this
repository with the Python worker, as the reference half of design 10's diagnosis. Run it with a
checkout of the last tree that held the package and that tree's interpreter, with the tree on the
path; `/workspace/uniserve-v1` is the reference tree of this project's hosts:

    PYTHONPATH=/workspace/uniserve-v1:/workspace/uniserve-v1/uniserve_kernels/src \\
        /workspace/uniserve-v1/.venv/bin/python <this file> ...

``dump_encoder_tables.py`` imports the reference's own modules, and two of those import
``uniserve_kernels`` unconditionally - ``uniserve/nn/rope.py`` imports ``rope`` before it ever
tests ``positions.is_cuda``, and every accelerated module imports ``require_kernel`` from
``uniserve_kernels.triton``. On a host where the extension is not built, or on the CPU, the import
fails long before the host arithmetic the dumper needs is reached.

This wrapper installs a stub for exactly those imports that no real module answers, then runs the
dumper unchanged. The stub **refuses to be called**: every table this diagnosis builds is plain host
arithmetic, so a call that reaches a stub means the host path was not taken and should be reported
rather than silently computed by something else. Where the extension is present, nothing is stubbed
and the real modules are used.

Usage::

    python crates/models/minimax-h3/tools/run_dump_encoder_tables.py \\
        --case case.json --out artifacts/diagnosis/<run>/ref \\
        [--text-encoder /opt/uv/models/MiniMax-H3/text_encoder]

Everything after this script's own name is passed through to ``dump_encoder_tables.py``.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import pathlib
import runpy
import sys
import types

DUMPER = pathlib.Path(__file__).with_name("dump_encoder_tables.py")
STUBBED_ROOT = "uniserve_kernels"


class KernelStub(types.ModuleType):
    """A stand-in for a kernel module, whose every member refuses to run on the host."""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)

        def refuse(*args, **kwargs):
            raise RuntimeError(
                f"the kernel {self.__name__}.{name} was called on the host: the CPU path of the "
                f"reference was not taken, so the import stub cannot stand in for it"
            )

        return refuse


class KernelFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Answers an import under `uniserve_kernels` that no real module could satisfy."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != STUBBED_ROOT and not fullname.startswith(f"{STUBBED_ROOT}."):
            return None
        return importlib.util.spec_from_loader(fullname, self, is_package=True)

    def create_module(self, spec):
        module = KernelStub(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        return None


def stub_missing_kernels() -> None:
    """Let imports of absent kernel modules resolve, without displacing any that exist.

    The finder goes last, so every path the interpreter already searches - the installed extension,
    the source tree, the site packages - is asked first, and only an import nothing else answers
    lands on the stub.
    """
    sys.meta_path.append(KernelFinder())


def main(argv: list[str]) -> int:
    stub_missing_kernels()
    # The dumper imports the reference by package name, and it is run from the repository root.
    sys.path.insert(0, os.getcwd())
    sys.argv = [str(DUMPER), *argv]
    runpy.run_path(str(DUMPER), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
