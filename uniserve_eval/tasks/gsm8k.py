"""Defines GSM8K's scored text-completion benchmark behavior.

The design confirms Qwen3-MoE quality with GSM8K's strict extracted-answer
accuracy (``specs/worker.md`` §10). A response is correct only when the value
after its last ``#### `` marker equals the value after the last marker of the
row's reference answer. The marker includes its trailing space, so
``####20`` carries no answer; a marker whose text does not parse is an
incorrect answer, not a request failure, and every measured row stays in the
denominator.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, ClassVar

from ..types import (
    Example,
    RequestRecord,
    TaskName,
    TaskRequest,
    ValidationResult,
)
from .base import BenchmarkTask, ImageCountRule

#: The system message every measured GSM8K request carries.
SYSTEM_PROMPT = (
    "Solve the problem step by step. End your answer with the final answer "
    "on its own line in the form `#### <number>`."
)

#: The marker that introduces a final answer. The space belongs to it: text
#: such as ``####20`` holds no marker.
ANSWER_MARKER = "#### "

# A value may sit between whitespace and punctuation: the optional ``$`` of
# a money answer, parentheses, or a full stop the model wrote after the
# number. A letter or digit next to the value means the text around the
# marker is not the answer alone, so ``#### The answer is 18`` and
# ``#### 18 apples`` hold no value. The value itself keeps its sign,
# thousands commas and decimal fraction.
_ANSWER_VALUE = re.compile(
    r"[^\w+\-]*"
    r"(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"[^\w]*"
)


def extract_answer(text: str) -> Decimal | None:
    """Return the number introduced by the text's last answer marker.

    The marker is ``#### `` including its space, so text whose only marker is
    ``####20`` holds no answer. Whitespace and punctuation may surround the
    value; text that does not reduce to one optionally signed integer or
    decimal, with optional ``$`` and thousands commas, yields ``None``. A
    letter or digit next to the value is not punctuation, so
    ``#### The answer is 18`` and ``#### 18 apples`` hold no answer either.

    Args:
        text: A generated response or a dataset row's reference answer.

    Returns:
        The parsed value, or ``None`` when the text holds no marker or no
        value after its last marker.
    """
    index = text.rfind(ANSWER_MARKER)
    if index < 0:
        return None
    value = _ANSWER_VALUE.fullmatch(text[index + len(ANSWER_MARKER) :])
    if value is None:
        return None
    return Decimal(value.group("number").replace(",", ""))


def reference_answer(record: RequestRecord) -> Decimal | None:
    """Return the gold value of the row a request was built from.

    The dataset adapter stores the reference answer as the row's
    ``metadata['answer']``; a record without one, or whose answer does not
    parse, yields ``None``.
    """
    example = record.example if isinstance(record.example, dict) else {}
    metadata = example.get("metadata")
    answer = metadata.get("answer") if isinstance(metadata, dict) else None
    return extract_answer(answer) if isinstance(answer, str) else None


@dataclass(frozen=True)
class Score:
    """Outcome counts of one GSM8K run over its measured rows.

    Attributes:
        correct: Successful requests whose extracted value equals the gold
            value.
        total: Every measured row, including failed requests and answers
            that do not parse.
        unparseable: Successful requests whose generated text carries no
            parseable value; failed requests are not counted here.
    """

    correct: int
    total: int
    unparseable: int

    @property
    def accuracy(self) -> float:
        """Return correct over total, zero when no row was measured."""
        return self.correct / self.total if self.total else 0.0


def score_records(records: Sequence[RequestRecord]) -> Score:
    """Score every measured record against its gold answer.

    A failed request is incorrect even when its partial text holds the right
    value, since the request produced no canonical response.
    """
    correct = 0
    unparseable = 0
    for record in records:
        if not record.success:
            continue
        prediction = extract_answer(record.generated_text)
        if prediction is None:
            unparseable += 1
            continue
        gold = reference_answer(record)
        if gold is not None and prediction == gold:
            correct += 1
    return Score(correct=correct, total=len(records), unparseable=unparseable)


class GSM8KTask(BenchmarkTask):
    """Builds and scores deterministic, streamed GSM8K completions.

    Every request is one system message holding ``SYSTEM_PROMPT`` and one
    user message holding the row's question verbatim, streamed with usage so
    the run records authoritative token counts. Sampling controls come from
    the point's ``sampling`` table and ``extra_body``.
    """

    name: ClassVar[TaskName] = TaskName.GSM8K
    default_stream: ClassVar[bool] = True
    accepts_image: ClassVar[bool] = False
    image_count: ClassVar[ImageCountRule] = ImageCountRule.FORBIDDEN

    def build_request(self, example: Example) -> TaskRequest:
        """Build a streamed chat-completions request for one question.

        The output limit is the row's ``max_tokens``, else the point's
        ``sampling.max_tokens``; with neither, ``max_completion_tokens`` is
        omitted. ``sampling.extra_body`` is merged last and so overrides any
        field set here.
        """
        limit = (
            example.max_tokens
            if example.max_tokens is not None
            else self.point.sampling.max_tokens
        )
        payload: dict[str, Any] = {
            "model": self.point.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example.prompt},
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if limit is not None:
            payload["max_completion_tokens"] = int(limit)
        self.apply_text_sampling(payload)
        return TaskRequest(self.point.endpoint, payload, stream=True)

    def validate_output(
        self, records: Sequence[RequestRecord]
    ) -> ValidationResult:
        """Require a gold answer per row and report the scored outcomes.

        A row whose reference answer does not parse can never be scored, so
        it fails ``gold_answers`` and the run is not a valid confirmation.
        """
        score = score_records(records)
        return ValidationResult(
            checks={
                "gold_answers": bool(records)
                and all(
                    reference_answer(record) is not None for record in records
                ),
            },
            statistics={
                "correct": score.correct,
                "total": score.total,
                "unparseable_predictions": score.unparseable,
            },
        )

    def summary_metrics(
        self, records: Sequence[RequestRecord]
    ) -> dict[str, float | int]:
        """Return accuracy, correct and total for the run's summary."""
        score = score_records(records)
        return {
            "accuracy": score.accuracy,
            "correct": score.correct,
            "total": score.total,
        }
