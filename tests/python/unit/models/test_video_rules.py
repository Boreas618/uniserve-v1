"""The video request rules are answered by the `uniserve rules` command.

The rules themselves are `crates/models/rules/src/video`'s; these tests hold
the transport to its contract — the answer shapes its callers destructure, one
invocation per batch, a refusal that carries the rule's own message, and the
same bytes for the same request.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from uniserve import video_rules

pytestmark = pytest.mark.unit

#: An aspect the canvas rule resolves, and the canvas it resolves to: the
#: 768-pixel short edge at 16:9.
ASPECT = (16.0, 9.0)
CANVAS = [1344, 768]

#: The H3 frame rule, `[min_frames, clip_frames, clip_latent_frames,
#: tail_latent_frames]`, as `processing._frame_clips` reads it.
FRAME_RULE = [107, 17, 5, 2]


def test_a_rule_answers_what_its_callers_destructure():
    assert video_rules.video_canvas(*ASPECT) == CANVAS
    assert video_rules.video_tasks() == ["t2va", "fl2va", "ref2va"]
    # A five-second request at 24 frames a second aligns up to 124 frames.
    assert video_rules.video_frame_count(5.0, None, 24, FRAME_RULE) == 124


def test_one_invocation_answers_a_whole_batch_in_request_order():
    answers = video_rules.answers(
        [
            (
                "video_canvas",
                {"aspect_width": ASPECT[0], "aspect_height": ASPECT[1]},
            ),
            (
                "video_frame_count",
                {
                    "seconds": 8.0,
                    "max_seconds": None,
                    "frame_rate": 24,
                    "frame_rule": FRAME_RULE,
                },
            ),
        ]
    )
    assert answers == [CANVAS, 192]


def test_a_refusal_carries_the_rules_own_message():
    with pytest.raises(video_rules.RulesError) as raised:
        video_rules.video_canvas(16.0, 1.0)
    assert "outside 1:4 to 4:1" in str(raised.value)
    assert raised.value.kind == "value"


def test_an_unknown_rule_is_named():
    with pytest.raises(video_rules.RulesError) as raised:
        video_rules.call("no_such_rule", {})
    assert "unknown rule" in str(raised.value)


def test_the_same_request_answers_the_same_bytes():
    request = json.dumps(
        {
            "calls": [
                {
                    "call": "video_canvas",
                    "args": {"aspect_width": 21, "aspect_height": 9},
                }
            ]
        }
    )
    command = [str(video_rules.executable()), "rules"]
    first = subprocess.run(
        command, input=request.encode(), capture_output=True, check=True
    )
    second = subprocess.run(
        command, input=request.encode(), capture_output=True, check=True
    )
    assert first.stdout == second.stdout


def test_a_refused_request_exits_non_zero():
    request = json.dumps(
        {
            "calls": [
                {
                    "call": "video_canvas",
                    "args": {"aspect_width": 16, "aspect_height": 1},
                }
            ]
        }
    )
    completed = subprocess.run(
        [str(video_rules.executable()), "rules"],
        input=request.encode(),
        capture_output=True,
        check=False,
    )
    assert completed.returncode == video_rules.REFUSED
    answer = json.loads(completed.stdout)["answers"][0]
    assert "answer" not in answer
    assert answer["error"]["kind"] == "value"


def test_a_binary_without_the_command_names_itself(monkeypatch, tmp_path):
    stale = tmp_path / "uniserve"
    stale.write_text(
        "#!/bin/sh\necho 'Error: unknown command rules' >&2\nexit 2\n"
    )
    stale.chmod(0o755)
    monkeypatch.setenv("UNISERVE_EXECUTABLE", str(stale))
    with pytest.raises(video_rules.RulesRuntimeError) as raised:
        video_rules.answers([("video_tasks", {})])
    message = str(raised.value)
    assert str(stale) in message
    assert "unknown command" in message
