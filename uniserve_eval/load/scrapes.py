"""Scrapes every server of a deployment once a second while a load runs.

A timeline load's evidence includes each host's ``/metrics`` exposition every
second (``artifacts/buffer-pool/heavy/protocol.md``, Records): the counters'
course over the load, such as when a scheduler counter rose relative to a
change, is read from them afterwards. ``Scraper`` takes them on a fixed
cadence from ``start`` until ``stop``, every origin at every tick, and keeps
each scrape as the server answered it.

A tick does not wait for the scrapes of earlier ticks: a server that answers
slowly delays none of the others' scrapes nor the cadence, and each scrape
is bounded by ``scrape_metrics``'s timeout. Each entry of ``entries`` is

- ``origin``: the server's base URL;
- ``tick``: the tick's index, from 0 at ``start``;
- ``sent`` and ``received``: ``time.perf_counter()`` readings when the
  request was sent and when its answer (or its failure) arrived, the clock
  of the load's request records;
- ``text``: the exposition, or ``None`` when the server did not answer it.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import time
from collections.abc import Sequence
from typing import Any

import aiohttp

from ..server_metrics import scrape_metrics

#: Seconds between two ticks: the protocol's one scrape per second.
SCRAPE_PERIOD_S = 1.0


class Scraper:
    """Scrapes a fixed set of origins on a fixed cadence until stopped."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        origins: Sequence[str],
        *,
        period_s: float = SCRAPE_PERIOD_S,
    ) -> None:
        """Configure the session, the origins and the cadence.

        Raises:
            ValueError: If ``period_s`` is not positive.
        """
        if not period_s > 0:
            raise ValueError("period_s must be positive")
        self.session = session
        self.origins = tuple(origin.rstrip("/") for origin in origins)
        self.period_s = period_s
        self.entries: list[dict[str, Any]] = []
        self._ticker: asyncio.Task[None] | None = None
        self._scrapes: set[asyncio.Task[None]] = set()

    def start(self) -> None:
        """Take the first tick now and one every period after it.

        Raises:
            RuntimeError: If the scraper was started already.
        """
        if self._ticker is not None:
            raise RuntimeError("the scraper was started already")
        self._ticker = asyncio.create_task(self._tick())

    async def stop(self) -> None:
        """Stop ticking and wait for the scrapes still in flight.

        A scrape in flight ends within its own timeout, so ``entries`` holds
        an entry for every tick taken once this returns. Stopping a scraper
        that was never started does nothing.
        """
        if self._ticker is None:
            return
        self._ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._ticker
        await asyncio.gather(*self._scrapes, return_exceptions=True)

    async def _tick(self) -> None:
        """Launch every origin's scrape at each tick, on the start's grid."""
        origin_time = time.perf_counter()
        for tick in itertools.count():
            for origin in self.origins:
                task = asyncio.create_task(self._scrape(origin, tick))
                self._scrapes.add(task)
                task.add_done_callback(self._scrapes.discard)

            # The next tick lies on the grid from the first one, so a late
            # wake-up does not shift the ticks after it.
            next_tick = origin_time + (tick + 1) * self.period_s
            await asyncio.sleep(max(0.0, next_tick - time.perf_counter()))

    async def _scrape(self, origin: str, tick: int) -> None:
        """Scrape one origin and keep its answer with its times."""
        sent = time.perf_counter()
        text = await scrape_metrics(self.session, origin)
        self.entries.append(
            {
                "origin": origin,
                "tick": tick,
                "sent": sent,
                "received": time.perf_counter(),
                "text": text,
            }
        )
