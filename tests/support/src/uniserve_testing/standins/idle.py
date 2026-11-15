"""A process that runs until a signal ends it.

The stand-in for a process a test needs only to exist, to be found and to be
stopped: a host service such as the Storage Server, or a worker a server
starts. It prints its process id when it starts, so the log its starter gives
it names it.

Its arguments are not read: they are the command line by which a caller finds
the process, as the evaluator finds a host service by its ``serve`` argument
and its socket path
(``python -m uniserve_testing.standins.idle storage-server serve --socket
PATH``).
"""

from __future__ import annotations

import os
import signal


def main() -> None:
    """Announce the process, then wait for the signal that ends it."""
    print(os.getpid(), flush=True)

    # Nothing happens until a signal arrives. A terminating signal (SIGTERM,
    # SIGKILL) ends the process by its default action; a stop and a
    # continuation (SIGSTOP, SIGCONT) only pause it, so it waits again.
    while True:
        signal.pause()


if __name__ == "__main__":
    main()
