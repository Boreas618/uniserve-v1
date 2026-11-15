"""A server process that reports to its caller once it has started.

It takes ``--load-s`` seconds to start, as a server loading a model does;
with ``--worker`` it then starts a worker that stays in its process group (an
:mod:`~uniserve_testing.standins.idle` stand-in), as a server's workers do.
Once started it reports so with an HTTP ``POST`` to ``--report``, whose body
is the worker's process id in decimal (empty without a worker), and then runs
until a signal ends it.

The caller owns the listener ``--report`` names: it learns from the report
when the process has started and which worker it started, so the process
itself listens on nothing.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
import urllib.request


def main() -> None:
    """Start, report the start to the caller, then wait for a signal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        required=True,
        help="the HTTP URL the start is reported to",
    )
    parser.add_argument(
        "--load-s",
        type=float,
        default=0.0,
        help="the seconds the process takes to start",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="start a worker in this process's group before reporting",
    )
    args = parser.parse_args()

    # The start's work: the time a server spends loading before it can
    # serve.
    time.sleep(args.load_s)

    # A worker started without a session of its own stays in this process's
    # group, where only a signal to the group reaches it once this process
    # is gone.
    body = b""
    if args.worker:
        worker = subprocess.Popen(
            [sys.executable, "-m", "uniserve_testing.standins.idle"]
        )
        body = str(worker.pid).encode()

    report = urllib.request.Request(args.report, data=body, method="POST")
    with urllib.request.urlopen(report):
        pass

    while True:
        signal.pause()


if __name__ == "__main__":
    main()
