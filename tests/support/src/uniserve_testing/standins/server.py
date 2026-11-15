"""A stand-in for ``uniserve serve``, launched from an evaluator profile.

Run as ``python -m uniserve_testing.standins.server --port PORT``. Like
``uniserve serve``, it first starts the host service it needs in its runtime
directory, ``$XDG_RUNTIME_DIR/uniserve`` (the directory the evaluator creates
for each deployment): a Storage Server, as a detached process of its own
session that outlives the server until the evaluator stops it. It then starts
an engine, the leader of a process group of its own that ends with the server
(:mod:`~uniserve_testing.standins.engine`), and serves:

- on ``127.0.0.1:PORT``, an OpenAI-compatible chat API whose
  ``POST /v1/chat/completions`` streams ``max_completion_tokens`` tokens per
  request, each with its id and top-5 logprobs, then the usage;
  ``GET /metrics``, a Prometheus exposition of ``stand_in_scrapes_total``
  (the scrapes answered, this one included) and
  ``stand_in_check_failures_total`` (always 0); and ``GET /version``;
- on ``admin-PORT.sock`` in the runtime directory, the admin listener: an
  update (``PUT /admin/config``) moves through ``switching`` for 0.2 s and
  then ``trial``, and a commit (``POST /admin/config/commit``) or a rollback
  (``POST /admin/config/rollback``) ends it as ``committed`` or
  ``rolled_back``; ``GET /admin/events?after=N&wait=S`` long-polls the
  update's event log.

The tokens: a request's prompt is its first message's content, and its token
at a position is :func:`token` of the prompt, the position and the
configuration it is served in. The server serves configuration ``A`` until an
update sets ``configuration=B``, and ``A`` again after a rollback; a request
keeps the configuration it arrived in, or names its own (``configuration`` in
its body). Its usage counts 64 prompt tokens, all but the last cached when it
served the prompt before. Tokens are 20 ms apart.

Faults: a request's body may carry a ``fault``. The third request with
``lost``, ``repeat``, ``tie_inside``, ``tie_outside``, ``fail`` or ``close``
is struck with it: its token at position 4 is left out, sent twice, swapped
with its second candidate, or replaced by one outside its top 5; it fails
with HTTP 500; or its connection closes after three tokens. ``late`` holds
the first token of every request that arrives during an update's switch for
1 s, ``slow`` spaces every request's tokens by 0.1 s, and ``uncached`` reports
nothing cached once an update switched.

What it reports, for the test to read: it prints ``completion`` for each chat
request it answers, and writes in its working directory ``storage-server.pid``
(a line ``PID SERVICES`` for the Storage Server it started, ``SERVICES`` being
its directory of host services), ``engine.pid`` (its engine's pid) and
``disconnected.txt`` (the prompt of each stream its client closed, one per
line).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any

from aiohttp import web

#: Seconds between two tokens of a stream, and between those of a ``slow``
#: stream.
TOKEN_INTERVAL_S = 0.02
SLOW_TOKEN_INTERVAL_S = 0.1

#: Seconds an update stays in ``switching`` before its ``trial``.
SWITCH_S = 0.2

#: Seconds a ``late`` request's first token is held when it arrived during a
#: switch.
HELD_S = 1.0

#: The request a fault strikes: the third that carries it.
STRUCK_REQUEST = 3

#: The position a token fault strikes, and the token after which a ``close``
#: fault closes the connection.
STRUCK_POSITION = 4
CLOSED_AFTER = 2

#: The prompt tokens of every request's usage.
PROMPT_TOKENS = 64


def token(prompt: str, position: int, configuration: str = "A") -> int:
    """The token the stand-in serves at ``position`` of ``prompt``'s output.

    Configuration ``A``'s tokens hash the prompt and the position; every
    other configuration's hash its name too, so they differ from ``A``'s.
    """
    key = f"{prompt}/{position}"
    if configuration != "A":
        key = f"{configuration}/{key}"
    return 1000 + zlib.crc32(key.encode()) % 50000


def _event_chunk(event: dict[str, Any]) -> bytes:
    """One server-sent event of the chat stream."""
    return b"data: " + json.dumps(event).encode() + b"\n\n"


class StandIn:
    """The state the stand-in's two listeners share."""

    def __init__(self) -> None:
        """Start serving configuration ``A`` with no update applied."""
        self.configuration = "A"
        # Whether an update is in its switch now, and whether one ever
        # switched (which ends the prompts' caching for ``uncached``).
        self.switching = False
        self.switched = False
        # The prompts served so far, and how many requests carried each
        # fault.
        self.seen: set[str] = set()
        self.faults: dict[str, int] = {}
        self.scrapes = 0
        # The update's event log, and the condition its long-polls wait on.
        self.events: list[dict[str, Any]] = []
        self.appended = asyncio.Condition()

    async def record(self, phase: str) -> None:
        """Append an update event of ``phase`` and wake the long-polls."""
        self.events.append(
            {
                "seq": len(self.events) + 1,
                "time": time.time(),
                "kind": "update",
                "message": f"update 1: {phase}",
                "data": {"phase": phase},
            }
        )
        async with self.appended:
            self.appended.notify_all()

    async def chat(self, request: web.Request) -> web.StreamResponse:
        """Stream a request's tokens, with the fault it carries."""
        body = await request.json()
        prompt = body["messages"][0]["content"]
        count = body["max_completion_tokens"]
        print("completion", flush=True)

        fault = body.get("fault")
        struck = False
        if fault is not None:
            self.faults[fault] = self.faults.get(fault, 0) + 1
            struck = self.faults[fault] == STRUCK_REQUEST
        if struck and fault == "fail":
            return web.json_response(
                {"error": {"message": "failed"}}, status=500
            )

        held = fault == "late" and self.switching
        served = body.get("configuration", self.configuration)
        cached = prompt in self.seen and not (
            fault == "uncached" and self.switched
        )
        self.seen.add(prompt)

        response = web.StreamResponse(
            headers={"content-type": "text/event-stream"}
        )
        await response.prepare(request)
        await asyncio.sleep(HELD_S if held else TOKEN_INTERVAL_S)

        positions = list(range(count))
        if struck and fault == "lost":
            positions.remove(STRUCK_POSITION)
        if struck and fault == "repeat":
            positions.insert(STRUCK_POSITION, STRUCK_POSITION)
        interval = (
            SLOW_TOKEN_INTERVAL_S if fault == "slow" else TOKEN_INTERVAL_S
        )

        try:
            for index, position in enumerate(positions):
                chosen = token(prompt, position, served)
                others = [chosen + 100000 * rank for rank in range(1, 5)]
                if struck and position == STRUCK_POSITION:
                    if fault == "tie_inside":
                        chosen, others[0] = others[0], chosen
                    if fault == "tie_outside":
                        chosen, others = 7, [8, 9, 10, 11]
                await response.write(_event_chunk(_token_event(chosen, others)))

                if struck and fault == "close" and index == CLOSED_AFTER:
                    assert request.transport is not None
                    request.transport.close()
                    return response
                await asyncio.sleep(interval)
        except (ConnectionResetError, asyncio.CancelledError):
            # The client closed the stream: the test reads which prompts'
            # streams were closed.
            with Path("disconnected.txt").open("a") as handle:
                handle.write(prompt + "\n")
            raise

        usage = {
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": count,
            # An admission always computes the prompt's last token.
            "prompt_tokens_details": {
                "cached_tokens": PROMPT_TOKENS - 1 if cached else 0
            },
        }
        final = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            "usage": usage,
        }
        await response.write(_event_chunk(final) + b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def update(self, request: web.Request) -> web.Response:
        """Apply an update's settings through its switch to its trial."""
        body = await request.json()
        self.switching = self.switched = True
        for setting in body.get("set", []):
            name, _, value = setting.partition("=")
            if name == "configuration":
                self.configuration = value
        await self.record("switching")

        await asyncio.sleep(SWITCH_S)
        self.switching = False
        await self.record("trial")
        return web.json_response({"kind": "config", "state": "succeeded"})

    async def commit(self, request: web.Request) -> web.Response:
        """Commit the update in trial."""
        await request.read()
        await self.record("committed")
        return web.json_response({"kind": "commit", "state": "succeeded"})

    async def rollback(self, request: web.Request) -> web.Response:
        """Roll the update in trial back to configuration ``A``."""
        await request.read()
        self.configuration = "A"
        await self.record("rolled_back")
        return web.json_response({"kind": "rollback", "state": "succeeded"})

    async def read_events(self, request: web.Request) -> web.Response:
        """Answer the events after ``after``, waiting up to ``wait`` seconds.

        The answer is a page as the daemon's event log writes it: its events,
        the sequence number to read after next, and no gap.
        """
        after = int(request.query.get("after", "0"))
        wait_s = float(request.query.get("wait", "0"))
        async with self.appended:
            try:
                await asyncio.wait_for(
                    self.appended.wait_for(lambda: len(self.events) > after),
                    wait_s,
                )
            except TimeoutError:
                pass
        return web.json_response(
            {
                "events": self.events[after:],
                "next": max(after, len(self.events)),
                "gap": False,
            }
        )

    async def metrics(self, request: web.Request) -> web.Response:
        """Answer the exposition, counting this scrape."""
        self.scrapes += 1
        return web.Response(
            text='stand_in_check_failures_total{pool="0"} 0\n'
            f"stand_in_scrapes_total {self.scrapes}\n"
        )

    async def version(self, request: web.Request) -> web.Response:
        """Answer the server's version."""
        return web.json_response({"version": "stand-in"})


def _token_event(chosen: int, others: list[int]) -> dict[str, Any]:
    """A chat stream event carrying one token with its top-5 candidates."""
    top = [
        {"token": f"token_id:{item}", "logprob": -0.1 - rank}
        for rank, item in enumerate([chosen, *others])
    ]
    entry = {
        "token": f"token_id:{chosen}",
        "logprob": -0.1,
        "top_logprobs": top,
    }
    choice = {
        "index": 0,
        "delta": {"content": f" {chosen}"},
        "token_ids": [chosen],
        "logprobs": {"content": [entry]},
    }
    return {"choices": [choice]}


def _start_processes(services: Path) -> subprocess.Popen[bytes]:
    """Start the Storage Server and the engine; record their pids.

    Returns:
        The engine, whose standard input pipe is its lifeline: the caller
        keeps the object, and with it the pipe's write end, for as long as
        the server runs.
    """
    # The Storage Server is found by its command line, as a host service
    # serving a socket in the runtime directory, and appends its output to
    # the log beside its socket.
    with (services / "storage-server.log").open("a") as log:
        storage = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uniserve_testing.standins.idle",
                "storage-server",
                "serve",
                "--socket",
                str(services / "storage-server.sock"),
            ],
            stdout=log,
            start_new_session=True,
        )
    with Path("storage-server.pid").open("a") as handle:
        handle.write(f"{storage.pid} {services}\n")

    engine = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uniserve_testing.standins.engine",
            "uniserve-engine",
        ],
        stdin=subprocess.PIPE,
        process_group=0,
    )
    Path("engine.pid").write_text(str(engine.pid))
    return engine


async def serve(port: int) -> None:
    """Start the processes, then serve both listeners until killed."""
    services = Path(os.environ["XDG_RUNTIME_DIR"]) / "uniserve"
    services.mkdir(parents=True, exist_ok=True)
    engine = _start_processes(services)
    stand_in = StandIn()

    admin = web.Application()
    admin.router.add_put("/admin/config", stand_in.update)
    admin.router.add_post("/admin/config/commit", stand_in.commit)
    admin.router.add_post("/admin/config/rollback", stand_in.rollback)
    admin.router.add_get("/admin/events", stand_in.read_events)
    api = web.Application()
    api.router.add_post("/v1/chat/completions", stand_in.chat)
    api.router.add_get("/metrics", stand_in.metrics)
    api.router.add_get("/version", stand_in.version)

    admin_runner, api_runner = web.AppRunner(admin), web.AppRunner(api)
    await admin_runner.setup()
    await api_runner.setup()
    await web.UnixSite(
        admin_runner, str(services / f"admin-{port}.sock")
    ).start()
    # The client listener last: the evaluator's readiness probe connects to
    # it, so the server is ready only once everything else serves.
    await web.TCPSite(api_runner, "127.0.0.1", port).start()

    try:
        await asyncio.Event().wait()
    finally:
        # An interrupt cancels the wait and the server ends its engine by
        # closing the lifeline; a signal that kills the process closes the
        # lifeline with the process.
        assert engine.stdin is not None
        engine.stdin.close()


def main() -> None:
    """Serve on the port the command line names."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="the TCP port of the chat API on 127.0.0.1",
    )
    args = parser.parse_args()
    asyncio.run(serve(args.port))


if __name__ == "__main__":
    main()
