"""MiniMax-H3's request rules, answered by the `uniserve rules` command.

The serving path plans every MiniMax-H3 request in Rust
(``crates/models/rules/src/video``). The evaluator holds a run to what the
server planned, so it asks the same rules rather than restating them: the
canvas a target names and the frame count of a duration, by running
``uniserve rules`` with one JSON request on standard input and one JSON answer
on standard output. No rule is computed here; a function of this module states
the arguments and returns what the rule answers.

The constants below are the model's released facts, the arguments the rules
take: its frame rate, its soundtrack, the 768p class a request selects with
its short edge, the aspect ratios it names and its frame rule.

Each call is one invocation, and its answer is held for the life of the
process, because the rules are pure: the same call and arguments answer the
same bytes. The executable must be built from a tree that carries the
``rules`` command. A binary that predates it answers an unknown command, and a
shell reports that as the same exit code a refusal uses, so the failure reads
as an unreadable answer rather than as a missing command: :func:`answers`
names the executable, its exit code and its own message when that happens.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "AUDIO_CHANNELS",
    "AUDIO_SAMPLE_RATE",
    "CANVAS_SHORT_EDGE",
    "FPS",
    "NAMED_ASPECT_RATIOS",
    "REFUSED",
    "PlanError",
    "RulesError",
    "RulesRuntimeError",
    "answers",
    "canvas",
    "executable",
    "frame_count",
    "named_canvas",
]

#: The exit status of a request in which a rule refused an argument.
REFUSED = 2

#: The model's output frame rate.
FPS = 24

#: The channels and the sample rate of the soundtrack every output carries.
AUDIO_CHANNELS = 2
AUDIO_SAMPLE_RATE = 32_000

#: The 768p class: the short edge a request selects the rule's canvases with.
CANVAS_SHORT_EDGE = 768

#: The aspect ratios a request names, as ``(width, height)``.
NAMED_ASPECT_RATIOS = ((21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16))

#: The frame rule as the rules take it: H3 generates ``17 * n + 5`` frames with
#: ``n`` positive, so the fewest are 22, and each further 17-frame clip adds
#: five latent frames to the two the trailing overlap keeps.
_FRAME_RULE = (22, 17, 5, 2)


class RulesError(ValueError):
    """A rule refused an argument.

    Attributes:
        kind: ``"value"`` for an argument a rule refused, ``"runtime"`` for
            a failure while answering a valid one.
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class RulesRuntimeError(RuntimeError):
    """The rules could not be asked: no executable, or no readable answer."""


class PlanError(ValueError):
    """A target MiniMax-H3 cannot serve.

    Attributes:
        field: The request field at fault, such as ``"target.aspect_ratio"``.
    """

    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field


def executable() -> Path:
    """The executable whose ``rules`` command answers.

    ``UNISERVE_EXECUTABLE`` names it when set; otherwise it is the
    ``uniserve`` beside this interpreter, and failing that the one on
    ``PATH``.

    Raises:
        RulesRuntimeError: No executable is found.
    """
    override = os.environ.get("UNISERVE_EXECUTABLE")
    if override:
        return Path(override)
    beside = Path(sys.executable).with_name("uniserve")
    if beside.is_file():
        return beside
    found = shutil.which("uniserve")
    if found:
        return Path(found)
    raise RulesRuntimeError(
        "no uniserve executable: not at UNISERVE_EXECUTABLE, not beside "
        f"{sys.executable}, not on PATH; set UNISERVE_EXECUTABLE"
    )


def answers(requests: Sequence[tuple[str, dict[str, Any]]]) -> list[Any]:
    """Answer calls, in one invocation.

    Args:
        requests: One ``(name, arguments)`` pair per call, in answer order.

    Returns:
        One answer per request, in the same order.

    Raises:
        RulesError: A rule refused an argument; the first refusal in answer
            order, carrying that rule's own message.
        RulesRuntimeError: The request or the answer could not be read.
    """
    payload = {
        "calls": [
            {"call": name, "args": arguments} for name, arguments in requests
        ]
    }
    program = executable()
    completed = subprocess.run(
        [str(program), "rules"],
        input=json.dumps(payload).encode(),
        capture_output=True,
        check=False,
    )
    if completed.returncode not in (0, REFUSED):
        raise RulesRuntimeError(
            f"{program} rules exited {completed.returncode}: "
            f"{completed.stderr.decode(errors='replace').strip()}"
        )
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        # A binary built before the command existed exits with a usage error
        # and no JSON, which reaches here as a refusal with nothing to read.
        raise RulesRuntimeError(
            f"{program} rules answered no JSON (exit {completed.returncode}): "
            f"{completed.stderr.decode(errors='replace').strip()}; build the "
            f"binary from the tree carrying the rules command"
        ) from error
    answered = document.get("answers")
    if not isinstance(answered, list) or len(answered) != len(requests):
        raise RulesRuntimeError(
            f"uniserve rules answered {answered!r} for {len(requests)} calls"
        )
    values: list[Any] = []
    for entry in answered:
        if "error" in entry:
            refusal = entry["error"]
            kind = str(refusal.get("kind", "runtime"))
            message = str(refusal.get("message", ""))
            if kind == "runtime":
                raise RulesRuntimeError(message)
            raise RulesError(kind, message)
        values.append(entry["answer"])
    return values


@functools.cache
def _answer(name: str, arguments_json: str) -> Any:
    """Answer one call, once per process."""
    return answers([(name, json.loads(arguments_json))])[0]


def _call(name: str, arguments: dict[str, Any]) -> Any:
    """Answer one call of a rule, from this process's held answers."""
    return _answer(name, json.dumps(arguments, sort_keys=True))


def canvas(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    """Resolve a display aspect ratio into the released canvas.

    Only the ratio of the arguments matters: an aspect ratio or a keyframe's
    displayed dimensions.

    Returns:
        The canvas as ``(width, height)`` in pixels.

    Raises:
        RulesError: The ratio is not positive or lies outside 1:4 to 4:1.
    """
    width, height = _call(
        "video_canvas",
        {
            "aspect_width": float(aspect_width),
            "aspect_height": float(aspect_height),
        },
    )
    return int(width), int(height)


def frame_count(seconds: float) -> int:
    """Return the generated frame count of a duration.

    Raises:
        RulesError: The duration lies outside 4 to 15 seconds.
    """
    return int(
        _call(
            "video_frame_count",
            {
                "seconds": float(seconds),
                "max_seconds": None,
                "frame_rate": FPS,
                "frame_rule": list(_FRAME_RULE),
            },
        )
    )


def named_canvas(task: str, aspect_ratio: str) -> tuple[int, int] | None:
    """Return the canvas a target names at the 768p class without media.

    ``auto`` is 16:9 for ``t2va`` and ``ref2va``; ``None`` is ``fl2va`` with
    ``auto``, whose canvas follows the first keyframe. ``fl2va`` also takes
    any ratio from 1:4 to 4:1, which the canvas rule resolves at its 768-pixel
    short edge.

    Returns:
        The canvas as ``(width, height)`` in pixels, or ``None``.

    Raises:
        PlanError: No served canvas the ratio names at the short edge, and
            the canvas rule does not apply to it.
        RulesError: The task is not one the rules name.
    """
    served = [
        [CANVAS_SHORT_EDGE, f"{width}:{height}", *canvas(width, height)]
        for width, height in NAMED_ASPECT_RATIOS
    ]
    rejection, values = _call(
        "video_named_canvas",
        {
            "task": str(task),
            "short_edge": CANVAS_SHORT_EDGE,
            "aspect_ratio": str(aspect_ratio),
            "canvases": served,
        },
    )
    if rejection is not None:
        field_path, message = rejection
        raise PlanError(str(field_path), str(message))
    return None if values is None else (int(values[0]), int(values[1]))
