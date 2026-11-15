"""The second host of a two-host case.

A two-host case runs a deployment across this host and another one, which
``--peer-host`` names as an ssh destination reachable without a password
prompt; ``--peer-address`` is the address this host reaches it at and
``--leader-address`` the address it reaches this host at. The other host
runs the ``uniserve`` its login installs, with its own environment: a
command is an argument vector of installed programs, run over ssh with no
variable of this host's environment and no shell text.
"""

from __future__ import annotations

import dataclasses
import shlex
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

#: ssh's options: no password prompt, and its own messages limited to
#: errors, so a command's output is the command's alone.
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "LogLevel=ERROR")

#: How long a short command on the other host may take, in seconds.
COMMAND_S = 120.0


@dataclasses.dataclass(frozen=True)
class RemoteHost:
    """The other host of a two-host case, as the run's options state it."""

    destination: str
    address: str
    leader_address: str

    def _ssh(self, argv: Sequence[str]) -> list[str]:
        """The ssh command that runs `argv` there.

        ssh hands the remote side one command line, so each argument is
        quoted as one word of it.
        """
        return ["ssh", *SSH_OPTIONS, self.destination, "--", shlex.join(argv)]

    def run(self, *argv: str, timeout_s: float = COMMAND_S) -> str:
        """Run the program `argv` there and return its output.

        Raises:
            subprocess.CalledProcessError: When it fails, with its output.
        """
        return subprocess.run(
            self._ssh(argv),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        ).stdout

    def write(self, path: str, text: str) -> None:
        """Write `text` to the file `path` there.

        ``tee`` reads it from the connection's standard input.

        Raises:
            subprocess.CalledProcessError: When it cannot be written.
        """
        subprocess.run(
            self._ssh(["tee", path]),
            input=text,
            stdout=subprocess.DEVNULL,
            text=True,
            timeout=COMMAND_S,
            check=True,
        )

    @contextmanager
    def runtime_directory(self) -> Iterator[str]:
        """A runtime directory there (mode 0700, which ``mktemp`` makes),
        removed when the block ends.
        """  # noqa: D205
        directory = self.run("mktemp", "-d").strip()
        try:
            yield directory
        finally:
            self.run("rm", "-rf", directory)

    def start(
        self,
        argv: Sequence[str],
        log: Path,
        *,
        environment: Mapping[str, str],
    ) -> subprocess.Popen[bytes]:
        """Start the program `argv` there with the variables `environment`
        states for it, its output to `log` on this host; the returned process
        is the ssh connection, which ends when the program does.
        """  # noqa: D205
        declared = [f"{name}={value}" for name, value in environment.items()]
        with log.open("wb") as output:
            return subprocess.Popen(
                self._ssh(["env", *declared, *argv]),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
