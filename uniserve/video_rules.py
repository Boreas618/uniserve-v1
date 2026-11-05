"""The Rust video request rules, answered by the `uniserve rules` command.

The serving path plans every MiniMax-H3 request in Rust
(``crates/models/rules/src/video``). This module answers the parts of those
rules the model package's computation and its tools read — the tasks, the
canvas rule, the largest references a request may bring, the canvas a target
names among the canvases a deployment serves, the frame count of a duration,
and the patch grid a still image is read on — by running ``uniserve rules``:
one JSON request on standard input, one JSON answer on standard output. No
rule is computed here; a function of this module states the arguments and
returns what the rule answers.

Each call is one invocation, and its answer is held for the life of the
process, because the rules are pure: the same call and arguments answer the
same bytes. A caller that knows its whole batch should use :func:`answers`
and pay one launch for all of it.
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
    "REFUSED",
    "RulesError",
    "RulesRuntimeError",
    "answers",
    "call",
    "video_canvas",
    "video_frame_count",
    "video_image_grid",
    "video_named_canvas",
    "video_reference_bounds",
    "video_tasks",
]

#: The exit status of a request in which a rule refused an argument.
REFUSED = 2


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


@functools.cache
def _answer(name: str, arguments_json: str) -> Any:
    """Answer one call, once per process."""
    return answers([(name, json.loads(arguments_json))])[0]


def call(name: str, arguments: dict[str, Any]) -> Any:
    """Answer one call of a rule.

    Args:
        name: The rule's name, such as ``"video_canvas"``.
        arguments: The rule's arguments, as the rules state them.

    Raises:
        RulesError: A rule refused an argument.
        RulesRuntimeError: The rules could not be asked.
    """
    return _answer(name, json.dumps(arguments, sort_keys=True))


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
    completed = subprocess.run(
        [str(executable()), "rules"],
        input=json.dumps(payload).encode(),
        capture_output=True,
        check=False,
    )
    if completed.returncode not in (0, REFUSED):
        raise RulesRuntimeError(
            f"uniserve rules exited {completed.returncode}: "
            f"{completed.stderr.decode(errors='replace').strip()}"
        )
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RulesRuntimeError(
            f"uniserve rules answered no JSON: {error}"
        ) from error
    answered = document.get("answers")
    if not isinstance(answered, list) or len(answered) != len(requests):
        raise RulesRuntimeError(
            f"uniserve rules answered {answered!r} for {len(requests)} calls"
        )
    values: list[Any] = []
    for entry in answered:
        if "error" in entry:
            error = entry["error"]
            kind = str(error.get("kind", "runtime"))
            message = str(error.get("message", ""))
            if kind == "runtime":
                raise RulesRuntimeError(message)
            raise RulesError(kind, message)
        values.append(entry["answer"])
    return values


def video_tasks() -> list[str]:
    """The request names of the tasks, in canonical order."""
    return call("video_tasks", {})


def video_canvas(aspect_width: float, aspect_height: float) -> list[int]:
    """The released canvas of a display aspect, as ``[width, height]``."""
    return call(
        "video_canvas",
        {
            "aspect_width": float(aspect_width),
            "aspect_height": float(aspect_height),
        },
    )


def video_frame_count(
    seconds: float,
    max_seconds: float | None,
    frame_rate: int,
    frame_rule: Sequence[int],
) -> int:
    """The generated frame count of a duration."""
    return call(
        "video_frame_count",
        {
            "seconds": float(seconds),
            "max_seconds": None if max_seconds is None else float(max_seconds),
            "frame_rate": int(frame_rate),
            "frame_rule": [int(count) for count in frame_rule],
        },
    )


def video_reference_bounds(
    frames: int,
    frame_rate: int,
    frame_rule: Sequence[int],
    image_short_edge: int,
) -> list[Any]:
    """The largest references a request generating ``frames`` may bring."""
    return call(
        "video_reference_bounds",
        {
            "frames": int(frames),
            "frame_rate": int(frame_rate),
            "frame_rule": [int(count) for count in frame_rule],
            "image_short_edge": int(image_short_edge),
        },
    )


def video_named_canvas(
    task: str,
    short_edge: int,
    aspect_ratio: str,
    canvases: Sequence[Sequence[Any]],
) -> list[Any]:
    """The canvas a target names, or ``[field, message]`` its refusal."""
    return call(
        "video_named_canvas",
        {
            "task": str(task),
            "short_edge": int(short_edge),
            "aspect_ratio": str(aspect_ratio),
            "canvases": [
                [int(edge), str(ratio), int(width), int(height)]
                for edge, ratio, width, height in canvases
            ],
        },
    )


def video_image_grid(
    image_config: str, video_config: str, height: int, width: int
) -> list[int]:
    """The patch grid a still image is read on, as ``[t, h, w]``."""
    return call(
        "video_image_grid",
        {
            "image_config": image_config,
            "video_config": video_config,
            "height": int(height),
            "width": int(width),
        },
    )
