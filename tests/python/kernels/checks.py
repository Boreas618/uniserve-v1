"""Running the kernel packages' numerical checks from pytest.

Each family directory under ``tests/python/kernels/`` holds the checks of its
operators' packages: scripts that launch a package's kernel and compare it
with a reference, and that print what they compared. A family's
``test_<family>.py`` runs every one of them as a subprocess, so a check keeps
its own interpreter state (Triton reads ``TRITON_INTERPRET`` when it is
imported) and a failure fails the test with the check's output.

A check runs in one of three ways:

- :func:`interpreted`: on Triton's interpreter with every device hidden, the
  ``unit`` layer. A check that offers ``--cpu-only`` is called with it here,
  which runs the half of it that needs no device.
- :func:`on_device`: compiled for and launched on a CUDA device, the ``gpu``
  layer; skipped on a host without one.
- with a built module: a ``gpu`` check of a native or cubin module takes the
  module's directory from ``UNISERVE_KERNEL_MODULES`` through :func:`module`,
  and is skipped, naming the variable, when it is unset.
"""

from __future__ import annotations

import functools
import os
import pathlib
import subprocess
import sys

import pytest

#: The root the family directories live under.
ROOT = pathlib.Path(__file__).resolve().parent

#: The variable naming the directory of built modules the module checks read:
#: ``<op>/<backend>/`` holds the module of the descriptor the check's defaults
#: describe (``module.cubin`` or ``module.so`` with its metadata), and a check
#: that compares two builds reads named subdirectories of it.
MODULES = "UNISERVE_KERNEL_MODULES"

#: Seconds one check may take: the interpreter walks every program instance on
#: the host, and the widest sampling cases take minutes there.
TIMEOUT = 1800


def run(script: pathlib.Path, arguments: tuple[str, ...], device: bool) -> None:
    """Run one check and fail with its output unless it exits with status 0."""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    if device:
        environment.pop("TRITON_INTERPRET", None)
    else:
        # The interpreter run is the host's: no device is visible, so a check
        # that would pick one runs the interpreter path it states instead.
        environment["TRITON_INTERPRET"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [sys.executable, script.name, *arguments],
        cwd=script.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        pytest.fail(
            f"{script.name} {' '.join(arguments)} exited with "
            f"{completed.returncode}:\n{output}",
            pytrace=False,
        )


def interpreted(script: pathlib.Path, *arguments: str) -> None:
    """Run a check on Triton's interpreter, with no device visible."""
    run(script, arguments, device=False)


@functools.cache
def device_count() -> int:
    """The CUDA devices this host shows, asked of a child interpreter."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.cuda.device_count())",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(completed.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return 0


def require_devices(count: int = 1) -> None:
    """Skip the calling test unless the host has `count` CUDA devices."""
    if device_count() < count:
        pytest.skip(
            f"needs {count} CUDA device(s), this host has {device_count()}"
        )


def on_device(script: pathlib.Path, *arguments: str) -> None:
    """Run a check compiled for and launched on a device."""
    require_devices()
    run(script, arguments, device=True)


def module(op: str, backend: str, *name: str, device: bool = True) -> str:
    """The built module a check reads, from ``UNISERVE_KERNEL_MODULES``.

    Skips the test when the variable is unset; a variable naming a module
    directory that does not exist fails it, because a run that asked for the
    check and did not get it must not read as a pass. A module launched on a
    device (``device``) is asked for only on a host that has one.
    """
    if device:
        require_devices()
    root = os.environ.get(MODULES)
    if not root:
        pytest.skip(f"{MODULES} names the built modules the module checks read")
    path = pathlib.Path(root, op, backend, *name)
    if not path.is_dir():
        pytest.fail(
            f"{MODULES}={root} holds no module at {path}", pytrace=False
        )
    return str(path)
