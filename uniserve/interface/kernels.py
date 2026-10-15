"""Kernel provider choices made by the numerical library, and their table.

Every runner reports one record per prepared operator: its layer ``path``,
its ``op``, the ``provider`` serving it, for some operators the provider of
each input class (``inputs``), and fields particular to the operator. The
rank collects the records as JSON object texts it does not read and hands
them back to :func:`table`, which renders the kernel table it logs.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from uniserve.backends import kernel_choices


def functions() -> dict[str, Callable[..., Any]]:
    return {"kernels.choices": choices}


def choices() -> int:
    """How many kernel choices this process has made so far.

    The worker compares the counter before and after a batch to detect
    kernels selected while serving.
    """
    return kernel_choices()


def layer_paths(paths: Sequence[str]) -> list[str]:
    """Compress numbered layer paths into ranges: ``layers.{0-3,5}.mlp``.

    Paths that differ in one numbered part share a template; paths that
    differ in several stay as they are.
    """
    templates: dict[tuple[str, ...], list[list[str]]] = {}
    for path in paths:
        parts = path.split(".")
        key = tuple("*" if part.isdecimal() else part for part in parts)
        templates.setdefault(key, []).append(parts)

    result: list[str] = []
    for key, members in templates.items():
        varying = [
            position
            for position, part in enumerate(key)
            if part == "*"
            and any(
                member[position] != members[0][position] for member in members
            )
        ]
        if len(varying) > 1:
            result.extend(".".join(parts) for parts in members)
            continue
        parts = list(members[0])
        if varying:
            (position,) = varying
            numbers = sorted({int(member[position]) for member in members})
            runs: list[list[int]] = []
            for number in numbers:
                if runs and runs[-1][1] + 1 == number:
                    runs[-1][1] = number
                else:
                    runs.append([number, number])
            ranges = ",".join(
                str(start) if start == stop else f"{start}-{stop}"
                for start, stop in runs
            )
            parts[position] = "{" + ranges + "}"
        result.append(".".join(parts))
    return result


def _call_sites(
    records: Sequence[str],
) -> list[tuple[dict[str, Any], list[str]]]:
    """Group a runner's records into call sites with their layer paths."""
    # Each layer's distinct records, in the order they were reported.
    layers: dict[str, list[dict[str, Any]]] = {}
    for text in records:
        record = json.loads(text)
        values = layers.setdefault(str(record.pop("path")), [])
        if record not in values:
            values.append(record)

    sites: list[tuple[dict[str, Any], list[str]]] = []
    for path, values in layers.items():
        # An attention layer with a prepared provider reports its
        # unprepared default too; the prepared one is the call site.
        prepared = any(
            record.get("op") == "attention"
            and record.get("provider") is not None
            for record in values
        )
        for record in values:
            if (
                prepared
                and record.get("op") == "attention"
                and record.get("provider") is None
            ):
                continue
            for site, paths in sites:
                if site == record:
                    paths.append(path)
                    break
            else:
                sites.append((record, [path]))
    return sites


def table(
    stage: str,
    rank: int,
    device: str,
    labels: Sequence[str],
    devices: Sequence[str],
    counts: Sequence[int],
    records: Sequence[str],
) -> str:
    """Render the kernel table of a rank's runners as one JSON object.

    Args:
        stage: ``"startup"`` or ``"serving"``.
        rank: The reporting rank.
        device: The rank's device name.
        labels: Each runner's label.
        devices: Each runner's device name.
        counts: How many of ``records`` each runner reported.
        records: Every runner's kernel records, back to back, as the JSON
            object texts the runners returned.

    Returns:
        ``{"stage", "rank", "device", "runners", "portable"}``: each runner
        with its call sites (equal records of several layers share one,
        with compressed ``layers``), and the call sites of CUDA runners
        that portable torch code serves, with the inputs it serves.
    """
    runners = []
    portable = []
    first = 0
    for label, placed, count in zip(labels, devices, counts, strict=True):
        reported = records[first : first + int(count)]
        first += int(count)
        calls = []
        for record, paths in _call_sites(reported):
            compressed = layer_paths(paths)
            calls.append({**record, "layers": compressed})
            provider = record.get("provider")
            if provider == "torch":
                served = ["every input"]
            elif provider is None:
                served = []
            else:
                served = [
                    name
                    for name, serving in (record.get("inputs") or {}).items()
                    if serving == "torch"
                ]
            if served and (placed == "cuda" or placed.startswith("cuda:")):
                portable.append(
                    {
                        "runner": label,
                        "op": record.get("op"),
                        "layers": compressed,
                        "inputs": served,
                    }
                )
        runners.append(
            {"runner": str(label), "device": str(placed), "call_sites": calls}
        )
    return json.dumps(
        {
            "stage": str(stage),
            "rank": int(rank),
            "device": str(device),
            "runners": runners,
            "portable": portable,
        },
        separators=(",", ":"),
    )
