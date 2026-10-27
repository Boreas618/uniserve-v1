"""Read a profiling session's per-worker SQLite files.

A worker writes one file per session (``POST /admin/profile``,
``docs/profiling.md``) in UniServe's own schema: a table per record kind
beside ``metadata``, ``strings``, ``clocks``, ``steps`` and ``collectors``.
The schema states its version (``schema_version``) and changes only by
additions, so this reader reads every file of its version or a later one,
and refuses an earlier one it does not know.

Every record's time is in one of two clocks, which ``metadata``'s
``doc.clocks`` names per table: CUPTI's clock (about ``CLOCK_REALTIME``) and
a device's ``%globaltimer`` (the fence probes). The file's ``clocks`` also
sample the host's ``CLOCK_MONOTONIC``, for a client that lines up host-side
data of its own. :meth:`ProfileFile.to_cupti` maps a time of either other
clock onto CUPTI's through those samples, taking the sample of least
uncertainty.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The schema version this reader was written for; files of later versions
# only add tables and columns.
SCHEMA_VERSION = 1

# The probe ids of a fence step's markers.
STEP_BEGIN = 0xF_FFFE
STEP_END = 0xF_FFFF


class ProfileError(ValueError):
    """A file that is not a profiling session's file this reader reads."""


@dataclass(frozen=True)
class Offset:
    """What to add to a time of one clock to place it on CUPTI's clock,
    and the alignment's uncertainty, nanoseconds.
    """  # noqa: D205

    nanoseconds: int
    uncertainty: int


class ProfileFile:
    """One worker's file of one session, opened read-only."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise ProfileError(f"{self.path} is not a file")
        self._db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self._db.row_factory = sqlite3.Row
        try:
            row = self._db.execute(
                "SELECT version FROM schema_version"
            ).fetchone()
        except sqlite3.DatabaseError as error:
            raise ProfileError(
                f"{self.path} is not a profiling session's file: {error}"
            ) from error
        if row is None or int(row[0]) < SCHEMA_VERSION:
            raise ProfileError(
                f"{self.path} has schema version {row and row[0]}; this "
                f"reader reads version {SCHEMA_VERSION} and later"
            )
        self.version = int(row[0])
        self.metadata = dict(
            self._db.execute("SELECT key, value FROM metadata").fetchall()
        )
        self.strings = {
            int(key): text
            for key, text in self._db.execute("SELECT id, text FROM strings")
        }

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> ProfileFile:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def rank(self) -> int:
        return int(self.metadata.get("rank", 0))

    @property
    def session(self) -> int:
        return int(self.metadata.get("session", 0))

    def tables(self) -> list[str]:
        """The tables the file holds."""
        return [
            row[0]
            for row in self._db.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table' "
                "ORDER BY name"
            )
        ]

    def rows(self, table: str, order: str | None = None) -> Iterator[dict]:
        """Every row of ``table`` as a dictionary, in ``order`` when
        given (an ``ORDER BY`` expression of its columns).
        """  # noqa: D205
        if table not in self.tables():
            return
        query = f'SELECT * FROM "{table}"'
        if order:
            query += f" ORDER BY {order}"
        for row in self._db.execute(query):
            yield dict(row)

    def count(self, table: str) -> int:
        if table not in self.tables():
            return 0
        return int(
            self._db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        )

    def name(self, string_id: int | None) -> str:
        """The text of a string id; empty for none."""
        if not string_id:
            return ""
        return self.strings.get(int(string_id), f"<string {string_id}>")

    def offset(self, clock: str, device: int | None = None) -> Offset | None:
        """What maps ``clock`` (``globaltimer`` of ``device``, or
        ``monotonic``) onto CUPTI's clock, from the sample of least
        uncertainty; ``None`` without one.
        """  # noqa: D205
        if clock == "globaltimer":
            query = (
                "SELECT cupti - gpu, uncertainty FROM clocks WHERE source = "
                "'globaltimer' AND device = ? AND cupti IS NOT NULL "
                "ORDER BY uncertainty LIMIT 1"
            )
            row = self._db.execute(query, (device,)).fetchone()
        elif clock == "monotonic":
            row = self._db.execute(
                "SELECT cupti - monotonic, uncertainty FROM clocks WHERE "
                "monotonic IS NOT NULL AND cupti IS NOT NULL "
                "ORDER BY uncertainty LIMIT 1"
            ).fetchone()
        else:
            raise ProfileError(f"no clock is named {clock}")
        if row is None:
            return None
        return Offset(int(row[0]), int(row[1]))

    def to_cupti(
        self, time: int, clock: str, device: int | None = None
    ) -> int | None:
        """``time`` of ``clock`` on CUPTI's clock; ``None`` without an
        alignment.
        """  # noqa: D205
        offset = self.offset(clock, device)
        return None if offset is None else time + offset.nanoseconds

    def fence_intervals(self) -> Iterator[dict[str, Any]]:
        """Every probed component instance's call, from the fence records:
        its device, step, instance path and component, and its start and
        end in ``%globaltimer`` nanoseconds. A step's entries and exits
        pair per instance in time order; an unpaired one is left out.
        """  # noqa: D205
        probes = {int(row["probe"]): row for row in self.rows("fence_probe")}
        entered: dict[tuple[int, int, int], list[int]] = {}
        for row in self.rows("fence", order="device, time"):
            probe = int(row["probe"])
            if probe >= STEP_BEGIN:
                continue
            key = (int(row["device"]), int(row["step"]), probe // 2)
            if probe % 2 == 0:
                entered.setdefault(key, []).append(int(row["time"]))
                continue
            starts = entered.get(key)
            if not starts:
                continue
            start = starts.pop()
            described = probes.get(probe - 1, {})
            yield {
                "device": key[0],
                "step": key[1],
                "instance": key[2],
                "path": described.get("path", ""),
                "component": described.get("component", ""),
                "start": start,
                "end": int(row["time"]),
            }

    def summary(self) -> dict[str, Any]:
        """What the file holds: its session, worker and rows by table."""
        return {
            "path": str(self.path),
            "schema_version": self.version,
            "session": self.session,
            "world": self.metadata.get("world"),
            "rank": self.rank,
            "device": self.metadata.get("device"),
            "chip": self.metadata.get("chip"),
            "end_reason": self.metadata.get("end_reason"),
            "steps_recorded": self.metadata.get("steps_recorded"),
            "collectors": [
                {
                    "name": row["name"],
                    "state": row["state"],
                    "detail": row["detail"],
                }
                for row in self.rows("collectors", order="time")
            ],
            "rows": {table: self.count(table) for table in self.tables()},
        }
