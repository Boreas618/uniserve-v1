"""An engine as the daemon starts one: it ends when its parent does.

The daemon starts each engine as the leader of a process group of its own, so
signalling the daemon's group does not reach it, and the engine ends with the
daemon through its lifeline. The stand-in's lifeline is its standard input: a
pipe whose write end its parent holds and never writes to. The kernel closes
that end when the parent exits, however it exits, and the stand-in, reading to
the end of its input, then ends too.

Its arguments are not read: they are the command line by which a caller finds
the process (``python -m uniserve_testing.standins.engine uniserve-engine``).
"""

from __future__ import annotations

import sys


def main() -> None:
    """Wait for the end of standard input, which the parent's exit closes."""
    sys.stdin.buffer.read()


if __name__ == "__main__":
    main()
