r"""Defines GSM8K's scored text-completion benchmark behavior.

The design confirms Qwen3-MoE quality with GSM8K's strict extracted-answer
accuracy (``specs/worker.md`` §10). A response is correct only when its final
answer value equals the value after the last marker of the row's reference
answer. A model states its final answer either after a ``#### `` marker or
inside LaTeX's ``\\boxed{...}``; the form that starts last decides, because
that is the last final answer the model wrote. A marker includes its trailing
space, so ``####20`` carries no answer, and a deciding form whose text does
not reduce to one number is an incorrect answer, never a request failure, so
every measured row stays in the denominator.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, ClassVar, Literal

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

#: The LaTeX form of a final answer. Its payload excludes braces, so a value
#: wrapped in display math is read and a nested expression such as
#: ``\\boxed{\\frac{1}{2}}`` yields a payload that is not a number.
BOXED_ANSWER = re.compile(r"\\boxed\{(?P<payload>[^{}]*)\}")

#: The form that decided one response's final answer.
AnswerForm = Literal["marker", "boxed", "none"]

MARKER_FORM: AnswerForm = "marker"
BOXED_FORM: AnswerForm = "boxed"
NO_FORM: AnswerForm = "none"

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


@dataclass(frozen=True)
class Answer:
    """One response's final answer and the form that decided it.

    Attributes:
        value: The parsed value, or ``None`` when the deciding form's text
            does not reduce to one number, and when the text holds no form
            at all.
        form: The deciding form, one of ``MARKER_FORM``, ``BOXED_FORM`` or
            ``NO_FORM``.
    """

    value: Decimal | None
    form: AnswerForm


def _parse_value(text: str) -> Decimal | None:
    """Return the one number `text` reduces to, or ``None``.

    `text` is a marker's or a boxed payload's text, which may carry
    surrounding whitespace and punctuation but must otherwise be a single
    optionally signed integer or decimal, with optional thousands commas.
    """
    value = _ANSWER_VALUE.fullmatch(text)
    if value is None:
        return None
    return Decimal(value.group("number").replace(",", ""))


def extract_answer_form(text: str) -> Answer:
    r"""Return the final answer of `text` and the form that decided it.

    Both forms are read and the later one decides: a ``#### `` marker and a
    ``\\boxed{...}`` payload are candidate final answers, and the one that
    starts last in the text is the answer the model wrote last. Only that
    one form's text is parsed, so a malformed last form yields no answer
    rather than falling back to an earlier candidate, which would guess at
    which value the model meant.

    Args:
        text: A generated response.

    Returns:
        The deciding form and its value, which is ``None`` when that form's
        text is not one number and when the text carries neither form.
    """
    marker = text.rfind(ANSWER_MARKER)
    boxed: re.Match[str] | None = None
    for match in BOXED_ANSWER.finditer(text):
        boxed = match
    if boxed is not None and boxed.start() > marker:
        return Answer(_parse_value(boxed.group("payload")), BOXED_FORM)
    if marker >= 0:
        rest = text[marker + len(ANSWER_MARKER) :]
        return Answer(_parse_value(rest), MARKER_FORM)
    return Answer(None, NO_FORM)


def extract_answer(text: str) -> Decimal | None:
    r"""Return the number the text's last final-answer form introduces.

    The forms are ``#### `` including its space, so text whose only marker
    is ``####20`` holds no answer, and ``\\boxed{...}``. The later of the
    two decides. Whitespace and punctuation may surround the value; text
    that does not reduce to one optionally signed integer or decimal, with
    optional ``$`` and thousands commas, yields ``None``. A letter or digit
    next to the value is not punctuation, so ``#### The answer is 18``,
    ``#### 18 apples`` and ``\\boxed{18 apples}`` hold no answer either.

    Args:
        text: A generated response or a dataset row's reference answer.

    Returns:
        The parsed value, or ``None`` when the text holds no form or the
        deciding form's text is not one number.
    """
    return extract_answer_form(text).value


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
    r"""Outcome counts of one GSM8K run over its measured rows.

    Every count of a form is over successful requests only; a failed request
    produced no canonical response and so is counted in `total` alone.

    Attributes:
        correct: Successful requests whose extracted value equals the gold
            value.
        total: Every measured row, including failed requests and answers
            that do not parse.
        unparseable: Successful requests whose deciding form holds no
            parseable value.
        by_marker: Successful requests decided by a ``#### `` marker.
        by_boxed: Successful requests decided by ``\\boxed{...}``.
        without_form: Successful requests carrying neither form.
    """

    correct: int
    total: int
    unparseable: int
    by_marker: int
    by_boxed: int
    without_form: int

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
    by_marker = 0
    by_boxed = 0
    without_form = 0
    for record in records:
        if not record.success:
            continue
        answer = extract_answer_form(record.generated_text)
        if answer.form == MARKER_FORM:
            by_marker += 1
        elif answer.form == BOXED_FORM:
            by_boxed += 1
        else:
            without_form += 1
        if answer.value is None:
            unparseable += 1
            continue
        gold = reference_answer(record)
        if gold is not None and answer.value == gold:
            correct += 1
    return Score(
        correct=correct,
        total=len(records),
        unparseable=unparseable,
        by_marker=by_marker,
        by_boxed=by_boxed,
        without_form=without_form,
    )


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
        The statistics name how many rows each answer form decided, so a
        change in how the model states its answers is visible in the run
        summary.
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
                "decided_by_marker": score.by_marker,
                "decided_by_boxed": score.by_boxed,
                "without_answer_form": score.without_form,
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
