"""The sampling inputs of a text forward's sampled calls.

A sampled call is a *task*. :class:`SamplingMetadata` lays out the inputs
the runner's ``select`` (or, for a verification, ``verify``) reads besides
the logits, in task order: per task its sampler (``GREEDY``,
``TOP_K_PATH`` or ``CATEGORICAL`` of :mod:`uniserve.sampling.select`),
slot, gate, forced finish, finish and transition token sets and whether its
request's logit biases apply; per candidate row its logits row, draw,
parameters and token set. A task's output position is its index.

Token sets cross as bitmasks of :func:`bitmask_words` int32 words, bit ``b``
of word ``w`` holding token ``32 * w + b``, tokens outside the vocabulary
dropped. Candidate rows' sets (allowed tokens less the suppressed ones) are
deduplicated into a table the rows index, so the rows of one request share
one set.

The tasks reporting logprobs pack their words into one compact column
(:class:`LogprobLayout`) scored at :func:`logprob_shape`, the padded shape a
captured scoring graph holds.

A decode forward whose rows all take the top-k sampler or the unshaped
greedy path selects in its own graph instead (:func:`own_selection`):
:class:`RowSampling` holds its rows' draws and parameters, and no selection
follows.

Two builders produce the same metadata. :meth:`SamplingMetadata.build`
lays out per-call :class:`SampleTask` controls, as a call with a sampling
state or a verification needs. :meth:`SamplingMetadata.gather` builds the
metadata of single-row tasks without a sampling state from the slot-indexed
request constants (:class:`~uniserve.executor.text.state.RequestTable`) as
array expressions, the decode hot path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from uniserve.errors import OutOfMemory
from uniserve.sampling.select import (
    CATEGORICAL,
    GREEDY,
    TOKEN_VALUE_MASK,
    TOP_K_PATH,
)
from uniserve.sampling.top_k import PARAMETER_COLUMNS

from .philox import DrawWindow, sampling_uniform

if TYPE_CHECKING:
    from .state import RequestTable

__all__ = [
    "FINISH_IDS",
    "LOGPROB_WORDS_PER_TOKEN",
    "NEUTRAL",
    "PROMPT_BLOCK_BYTES",
    "TOP_K_CANDIDATES",
    "LogprobLayout",
    "OwnSelection",
    "PaddedSelection",
    "RowSampling",
    "SampleTask",
    "SamplingControls",
    "SamplingMetadata",
    "SamplingParams",
    "SamplingState",
    "bitmask_words",
    "candidates",
    "controls_for_call",
    "finish_token_ids",
    "least_top",
    "logprob_capacity",
    "logprob_count",
    "logprob_shape",
    "logprob_words",
    "own_selection",
    "pad_selection",
    "padding_selection",
    "prompt_block",
    "sampling_paths",
    "task_bucket",
    "task_draws",
    "top_bucket",
    "top_k_width",
]

# Finish tokens a decode row passes to the decode forward. A row with more
# is selected by the runner's ``select``.
FINISH_IDS = 8

# The most candidates the top-k sampler keeps per row: a row samples from
# its top-k logits when its top-k width is at most this. The captured
# decode selection keeps this many for every row, so its shapes do not
# depend on the rows' widths.
TOP_K_CANDIDATES = 128

# The parameter row of a neutral row, a padding row of a forward that
# selects its rows: zero temperature, no cutoff, no penalty. Columns are
# those of ``uniserve.sampling.top_k`` (temperature, top-p, min-p,
# typical-p, repetition, frequency, presence, top-k width).
NEUTRAL = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
NEUTRAL.flags.writeable = False

# Logprob words per token of a binding's capacity: room for 20 top tokens
# per scored position.
LOGPROB_WORDS_PER_TOKEN = 3 + 3 * 20

# The bytes of transient logits, scores and words one ``score_prompt`` call
# may hold.
PROMPT_BLOCK_BYTES = 512 << 20


def _next_power_of_two(count: int) -> int:
    # The least power of two holding ``count``; one for zero.
    return 1 if count <= 1 else 1 << (count - 1).bit_length()


def candidates(vocab: int) -> int:
    """Return the candidates the top-k sampler keeps per row of ``vocab``."""
    return min(TOP_K_CANDIDATES, vocab)


def top_k_width(top_k: int, vocab: int) -> int:
    """Return the tokens a row's top-k cutoff keeps: its width, or ``vocab``.

    A width of zero is no cutoff; a width beyond the vocabulary keeps it
    whole.
    """
    return vocab if top_k == 0 else min(top_k, vocab)


def bitmask_words(vocab: int) -> int:
    """Return the int32 words of one token set over ``vocab`` tokens."""
    return -(-vocab // 32)


def task_bucket(count: int, capacity: int) -> int:
    """Return the task bucket of ``count`` tasks on ``capacity`` rows.

    The bucket is the next power of two, at most the capacity
    (:func:`~uniserve.executor.text.shapes.task_buckets`).
    """
    return min(_next_power_of_two(count), capacity)


def top_bucket(top: int, vocab: int) -> int:
    """Return the top bucket of ``top`` reported tokens over ``vocab``.

    Zero when none is reported, else the next power of two, at most the
    vocabulary.
    """
    if top == 0:
        return 0
    return min(_next_power_of_two(top), vocab)


def least_top(top: int) -> int:
    """Return the fewest top tokens top bucket ``top`` holds.

    One past the power of two below the bucket, or the bucket itself for
    zero and one.
    """
    if top < 2:
        return top
    return (1 << ((top - 1).bit_length() - 1)) + 1


def logprob_shape(
    rows: int, top: int, vocab: int, capacity: int
) -> tuple[int, int]:
    """Return the packed ``(rows, top)`` shape of a logprob column.

    The shape of ``rows`` tasks reporting at most ``top`` top tokens each,
    on a binding of ``capacity`` rows over ``vocab`` tokens: the rows' task
    bucket and the top bucket. A shape at most doubles each count, so its
    words are at most four times the unpadded ones.
    """
    return task_bucket(rows, capacity), top_bucket(top, vocab)


def logprob_count(count: int, vocab: int) -> int:
    """Return the top count a call reports per position, within ``vocab``.

    ``count`` is the generated tokens' ``n_logprobs`` or the prompt's
    ``n_prompt_logprobs``.
    """
    return min(int(count), vocab)


def logprob_words(rows: int, count: int) -> int:
    """Return the words of ``rows`` logprob rows of ``count`` top tokens.

    Each row holds its selected token, score and rank, then its top tokens,
    scores and ranks.
    """
    return rows * (3 + 3 * count)


def logprob_capacity(tokens: int) -> int:
    """Return the logprob words a forward of a binding may report.

    ``tokens`` is the binding's token capacity; a forward whose logprobs
    exceed the words is refused (:class:`~uniserve.errors.OutOfMemory`).
    """
    return tokens * LOGPROB_WORDS_PER_TOKEN


def prompt_block(top: int, vocab: int, tokens: int) -> int:
    """Return the prompt rows one ``score_prompt`` call scores at ``top``.

    As many rows as keep the call's transient logits, scores and words
    within :data:`PROMPT_BLOCK_BYTES` (24 bytes per vocabulary entry: the
    projected, gathered and retained logits and their FP32 scores; 36 per
    top token: its values, indices and words), no more than report the
    bucket's least top tokens within the binding's logprob words, and no
    more than the binding's ``tokens``: a call projects its rows' hidden
    states through operators prepared for that many; at least one.
    """
    fitting = PROMPT_BLOCK_BYTES // max(24 * vocab + 36 * top, 1)
    reporting = logprob_capacity(tokens) // logprob_words(1, least_top(top))
    return max(min(fitting, reporting, tokens), 1)


def _f32(value: Any) -> float:
    # Sampling parameters are float32 on the wire; every comparison happens
    # on the float32 value.
    return float(np.float32(value))


def _tokens(values: Any) -> tuple[int, ...]:
    return tuple(int(token) for token in values)


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """A request's admitted text sampling parameters (``ipc``'s).

    Floating fields hold float32 values. The defaults are the direct
    construction defaults: greedy, no cutoff, penalty or constraint.
    """

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    ignore_eos: bool = False
    seed: int | None = None
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: tuple[tuple[int, float], ...] = ()
    min_tokens: int = 0
    return_logprobs: bool = False
    n_logprobs: int = 0
    return_prompt_logprobs: bool = False
    n_prompt_logprobs: int = 0
    allowed_token_ids: tuple[int, ...] | None = None
    typical_p: float = 1.0
    forced_token_ids: tuple[int, ...] = ()

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> SamplingParams:
        """Decode the JSON of ``ipc::SamplingParams``.

        Absent fields take their defaults; host-only fields the worker does
        not read (``bad_words_ids``) are ignored.
        """
        default = cls()
        allowed = value.get("allowed_token_ids")
        seed = value.get("seed")
        return cls(
            temperature=_f32(value.get("temperature", default.temperature)),
            top_k=int(value.get("top_k", default.top_k)),
            top_p=_f32(value.get("top_p", default.top_p)),
            ignore_eos=bool(value.get("ignore_eos", default.ignore_eos)),
            seed=None if seed is None else int(seed),
            min_p=_f32(value.get("min_p", default.min_p)),
            repetition_penalty=_f32(
                value.get("repetition_penalty", default.repetition_penalty)
            ),
            frequency_penalty=_f32(
                value.get("frequency_penalty", default.frequency_penalty)
            ),
            presence_penalty=_f32(
                value.get("presence_penalty", default.presence_penalty)
            ),
            logit_bias=tuple(
                (int(token), _f32(bias))
                for token, bias in value.get("logit_bias", default.logit_bias)
            ),
            min_tokens=int(value.get("min_tokens", default.min_tokens)),
            return_logprobs=bool(
                value.get("return_logprobs", default.return_logprobs)
            ),
            n_logprobs=int(value.get("n_logprobs", default.n_logprobs)),
            return_prompt_logprobs=bool(
                value.get(
                    "return_prompt_logprobs", default.return_prompt_logprobs
                )
            ),
            n_prompt_logprobs=int(
                value.get("n_prompt_logprobs", default.n_prompt_logprobs)
            ),
            allowed_token_ids=None if allowed is None else _tokens(allowed),
            typical_p=_f32(value.get("typical_p", default.typical_p)),
            forced_token_ids=_tokens(
                value.get("forced_token_ids", default.forced_token_ids)
            ),
        )

    def uses_penalties(self) -> bool:
        """Return whether token-count penalties affect selection."""
        return (
            self.repetition_penalty != 1.0
            or self.frequency_penalty != 0.0
            or self.presence_penalty != 0.0
        )

    def generated_logprobs_requested(self) -> bool:
        """Return whether generated-token logprobs are reported."""
        return self.return_logprobs or self.n_logprobs > 0

    def prompt_logprobs_requested(self) -> bool:
        """Return whether prompt-token logprobs are reported."""
        return self.return_prompt_logprobs or self.n_prompt_logprobs > 0

    def device_greedy(self) -> bool:
        """Return whether the parameters permit an unshaped device argmax.

        Per-call token sets and draft chains are checked by the controls.
        """
        return (
            self.temperature <= 0.0
            and self.top_k == 0
            and self.top_p == 1.0
            and self.typical_p == 1.0
            and not self.generated_logprobs_requested()
            and not self.logit_bias
            and not self.uses_penalties()
        )

    def top_k_eligible(self, vocab: int) -> bool:
        """Return the parameter half of the top-k sampler's eligibility.

        The top-k sampler reads only per-row parameters and draws: no bias,
        no typical-p cutoff, no reported logprob (whose packed layout the
        host plans), and at most its candidates when sampling. A
        zero-temperature row selects its largest shaped logit, which the
        candidates always hold.
        """
        return (
            not self.logit_bias
            and self.typical_p == 1.0
            and not self.generated_logprobs_requested()
            and (
                self.temperature <= 0.0
                or top_k_width(self.top_k, vocab) <= candidates(vocab)
            )
        )

    def parameters(self, vocab: int) -> np.ndarray:
        """Return the ``[PARAMETER_COLUMNS]`` float32 parameter row.

        The last column is the top-k width inside the vocabulary, or zero
        without one: an integer far below 2**24, which float32 holds
        exactly.
        """
        width = top_k_width(self.top_k, vocab)
        return np.array(
            [
                self.temperature,
                self.top_p,
                self.min_p,
                self.typical_p,
                self.repetition_penalty,
                self.frequency_penalty,
                self.presence_penalty,
                width if width < vocab else 0.0,
            ],
            dtype=np.float32,
        )

    def sorts(self, vocab: int) -> bool:
        """Return whether a general-path row sorts its whole vocabulary row.

        It does for a top-k width beyond the top-k sampler's candidates, a
        nucleus cutoff without a top-k width, or a typical-p cutoff.
        """
        width = top_k_width(self.top_k, vocab)
        width = width if width < vocab else 0
        return (
            width > candidates(vocab)
            or (width == 0 and 0.0 < self.top_p < 1.0)
            or self.typical_p < 1.0
        )


@dataclass(frozen=True, slots=True)
class SamplingState:
    """A call's branch-local sampling state (``ipc::SamplingState``)."""

    allowed_token_ids: tuple[int, ...] | None = None
    suppressed_token_ids: tuple[int, ...] = ()
    finish_token_ids: tuple[int, ...] = ()
    transition_token_ids: tuple[int, ...] = ()
    force_finish: bool = False

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> SamplingState:
        """Decode the JSON of ``ipc::SamplingState``."""
        allowed = value.get("allowed_token_ids")
        return cls(
            allowed_token_ids=None if allowed is None else _tokens(allowed),
            suppressed_token_ids=_tokens(value.get("suppressed_token_ids", ())),
            finish_token_ids=_tokens(value.get("finish_token_ids", ())),
            transition_token_ids=_tokens(value.get("transition_token_ids", ())),
            force_finish=bool(value.get("force_finish", False)),
        )


def finish_token_ids(
    admitted: Sequence[int], state: SamplingState | None
) -> tuple[int, ...]:
    """Merge a request's admitted stop tokens with its branch's tokens.

    Either list alone is kept as it is; both are merged into their sorted,
    deduplicated union.
    """
    local = state.finish_token_ids if state is not None else ()
    if not local:
        return tuple(admitted)
    if not admitted:
        return tuple(local)
    return tuple(sorted(set(admitted) | set(local)))


@dataclass(slots=True)
class SamplingControls:
    """Per-call constraints aligned with the call's candidate rows.

    Attributes:
        allowed: Per candidate row, the tokens it may select, or ``None``
            for every token.
        suppress: Tokens no row may select.
        finish_token_ids: Tokens that finish the call's request.
        transition_token_ids: Tokens that advance a structured-generation
            transition.
        force_finish: The call finishes whatever it selects.
        draft_token_ids: A verification's draft tokens.
        terminal_draft_prefix: One-based length of the draft prefix ending
            at the first finish token, or ``None``.
        return_transition: The call has a transition output.
    """

    allowed: list[tuple[int, ...] | None] = field(default_factory=list)
    suppress: tuple[int, ...] = ()
    finish_token_ids: tuple[int, ...] = ()
    transition_token_ids: tuple[int, ...] = ()
    force_finish: bool = False
    draft_token_ids: tuple[int, ...] = ()
    terminal_draft_prefix: int | None = None
    return_transition: bool = False

    def device_greedy(self, params: SamplingParams) -> bool:
        """Return whether the call selects by an unshaped device argmax."""
        return (
            not self.draft_token_ids
            and params.device_greedy()
            and all(allowed is None for allowed in self.allowed)
        )

    def path(self, params: SamplingParams, vocab: int) -> int:
        """Return the call's sampler without inspecting device logits.

        A greedy call with suppressed tokens still takes ``GREEDY``; the
        own selection of a decode forward refuses it (:func:`own_selection`).
        """
        if self.device_greedy(params):
            return GREEDY
        if (
            not self.draft_token_ids
            and all(allowed is None for allowed in self.allowed)
            and not self.suppress
            and params.top_k_eligible(vocab)
        ):
            return TOP_K_PATH
        return CATEGORICAL


def controls_for_call(
    params: SamplingParams,
    state: SamplingState | None,
    admitted_finish: Sequence[int],
    positions: Sequence[int] | np.ndarray,
    drafts: Sequence[int],
    key: int,
    return_transition: bool = False,
) -> tuple[SamplingControls, np.ndarray | None]:
    """Resolve a sampled call's controls and draws.

    Draw coordinates describe the candidate rows, independent of scheduling
    order or draft acceptance. ``forced_token_ids[i]`` forces the call's
    ``i``-th candidate row (its position within the call); other rows take
    the state's allowed tokens, else the request's.

    Args:
        params: The request's admitted parameters.
        state: The call's sampling state, if any.
        admitted_finish: The request's admitted finish tokens.
        positions: The candidate rows' semantic sampling positions.
        drafts: A verification's draft tokens.
        key: The request's stream key
            (:func:`~uniserve.executor.text.philox.request_sampling_key`).
        return_transition: The call has a transition output.

    Returns:
        The controls, and one float32 draw per candidate row: Philox draws
        for a positive temperature, none for the unshaped greedy path, and
        zeros otherwise.
    """
    allowed_default = params.allowed_token_ids
    if state is not None and state.allowed_token_ids is not None:
        allowed_default = state.allowed_token_ids
    forced = params.forced_token_ids
    controls = SamplingControls(
        allowed=[
            (forced[index],) if index < len(forced) else allowed_default
            for index in range(len(positions))
        ],
        suppress=state.suppressed_token_ids if state is not None else (),
        finish_token_ids=finish_token_ids(admitted_finish, state),
        transition_token_ids=(
            state.transition_token_ids if state is not None else ()
        ),
        force_finish=state is not None and state.force_finish,
        draft_token_ids=tuple(drafts),
        return_transition=return_transition,
    )
    finish = set(controls.finish_token_ids)
    controls.terminal_draft_prefix = next(
        (
            index + 1
            for index, token in enumerate(controls.draft_token_ids)
            if token in finish
        ),
        None,
    )

    draws: np.ndarray | None
    if params.temperature > 0.0:
        draws = sampling_uniform(key, np.asarray(positions, dtype=np.uint64))
    elif controls.device_greedy(params):
        draws = None
    else:
        draws = np.zeros(len(positions), dtype=np.float32)
    return controls, draws


@dataclass(slots=True)
class SampleTask:
    """One sampled call of an invocation.

    Attributes:
        params: The request's admitted parameters.
        controls: The call's controls.
        draws: One draw per candidate row; ``None`` for the unshaped greedy
            path.
        slot: The request slot.
        row: The call's first candidate row in the invocation's logits.
        gated: The call is gated by its request's continuation flag.
    """

    params: SamplingParams
    controls: SamplingControls
    draws: np.ndarray | None
    slot: int
    row: int
    gated: bool

    @property
    def rows(self) -> int:
        """Return the task's candidate rows."""
        return len(self.controls.allowed)


@dataclass(frozen=True, slots=True)
class LogprobLayout:
    """Row selection of one compact logprob column.

    Field by field over its rows: each row's selected token, its score and
    its rank, then each row's own top tokens, scores and ranks,
    ``counts[i]`` of them for packed row ``i``, which reports for
    ``rows[i]``.
    """

    rows: np.ndarray
    counts: np.ndarray

    @property
    def words(self) -> int:
        """Return the column's words: three per row and per top token."""
        return 3 * (len(self.rows) + int(np.sum(self.counts)))


def _token_bits(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Word index and uint32 bit of each token of a bitmask.
    return tokens >> 5, np.left_shift(
        np.uint32(1), (tokens & 31).astype(np.uint32)
    )


def _insert(words: np.ndarray, tokens: Sequence[int], vocab: int) -> None:
    # Set `tokens` in a uint32 bitmask; tokens outside the vocabulary are
    # dropped.
    values = np.asarray(tokens, dtype=np.int64)
    values = values[(values >= 0) & (values < vocab)]
    index, bits = _token_bits(values)
    np.bitwise_or.at(words, index, bits)


def _permitted(
    allowed: Sequence[int] | None, suppressed: Sequence[int], vocab: int
) -> np.ndarray:
    # The set a row may select: its allowed tokens (every token without a
    # list), never the suppressed ones; bits past the vocabulary stay clear.
    words = np.zeros(bitmask_words(vocab), dtype=np.uint32)
    if allowed is None:
        words[:] = 0xFFFF_FFFF
        if vocab % 32:
            words[-1] = (1 << (vocab % 32)) - 1
    else:
        _insert(words, allowed, vocab)
    values = np.asarray(suppressed, dtype=np.int64)
    values = values[(values >= 0) & (values < vocab)]
    index, bits = _token_bits(values)
    np.bitwise_and.at(words, index, ~bits)
    return words.view(np.int32)


class _TokenSets:
    """Distinct token sets of a forward's candidate rows, in first use."""

    def __init__(self, vocab: int, capacity: int) -> None:
        self.vocab = vocab
        self.capacity = capacity
        self.index: dict[tuple[Any, ...], int] = {}
        self.masks: list[np.ndarray] = []

    def row(
        self, allowed: Sequence[int] | None, suppress: Sequence[int]
    ) -> int:
        """Return the set of a row, -1 when every token is permitted."""
        if allowed is None and not suppress:
            return -1
        key = (None if allowed is None else tuple(allowed), tuple(suppress))
        found = self.index.get(key)
        if found is not None:
            return found
        index = len(self.index)
        self.index[key] = index
        if len(self.index) > self.capacity:
            raise OutOfMemory("token sets exceed the binding's capacity")
        self.masks.append(_permitted(allowed, suppress, self.vocab))
        return index

    def table(self) -> np.ndarray:
        """Return the ``[sets, words]`` int32 set table."""
        if not self.masks:
            return np.zeros((0, bitmask_words(self.vocab)), dtype=np.int32)
        return np.stack(self.masks)


def _check_vocabulary(vocab: int) -> None:
    if vocab > TOKEN_VALUE_MASK:
        raise NotImplementedError(
            "vocabulary exceeds the device token decision range"
        )


def _validate(task: SampleTask, vocab: int) -> None:
    # Host-only checks; no device value is read.
    rows = task.rows
    if rows == 0 or rows != len(task.controls.draft_token_ids) + 1:
        raise ValueError(
            "sampling rows must cover the draft chain and bonus token"
        )
    if task.controls.device_greedy(task.params):
        if task.draws is not None or rows != 1:
            raise ValueError("greedy sampling task has shaped metadata")
    elif task.draws is None or len(task.draws) != rows:
        raise ValueError("sampling task draws must align with its rows")
    if any(token >= vocab for token in task.controls.draft_token_ids):
        raise ValueError(
            "speculative draft token is outside the model vocabulary"
        )


def _logprobs(
    counts: np.ndarray, vocab: int, capacity: int, logprob_capacity: int
) -> tuple[LogprobLayout | None, tuple[int, int]]:
    # The compact layout of the reporting tasks' words and their scoring
    # bucket; none without a reporting task.
    if len(counts) == 0:
        return None, (0, 0)
    layout = LogprobLayout(
        rows=np.arange(len(counts), dtype=np.int64),
        counts=counts.astype(np.int64),
    )
    if layout.words > logprob_capacity:
        raise OutOfMemory("logprob words exceed the binding's capacity")
    bucket = logprob_shape(len(counts), int(counts.max()), vocab, capacity)
    return layout, bucket


@dataclass(slots=True)
class SamplingMetadata:
    """The selection inputs of one forward's tasks, in task order.

    ``W`` is the forward's candidate rows per task (one, or the most rows of
    a verification's tasks); a task's rows past its own repeat its first
    row's logits row, draw, parameters and set.

    Attributes:
        tasks: Tasks ``T``.
        width: Candidate rows per task ``W``.
        rows: ``[T * W]`` int64 each candidate row's logits row.
        paths: ``[T]`` int64 each task's sampler.
        draws: ``[T * W]`` float32 draws.
        params: ``[T * W, PARAMETER_COLUMNS]`` float32 parameters.
        slots: ``[T]`` int64 request slots.
        gates: ``[T]`` int64, 1 for a task gated by its continuation flag.
        masks: ``[sets, words]`` int32 distinct token sets.
        mask_rows: ``[T * W]`` int64 each candidate row's set, or -1.
        biased: ``[T]`` uint8 tasks whose request's logit biases apply.
        force: ``[T]`` uint8 tasks that finish whatever they select.
        finish: ``[T, words]`` int32 finish token sets.
        transitions: ``[T, words]`` int32 transition token sets; empty for
            a task without a transition output.
        drafts: ``[T, W - 1]`` int64 draft tokens, -1 past a task's own.
        terminal: ``[T]`` int64 one-based terminal draft prefix, or zero.
        top_k: Some task takes the top-k sampler.
        categorical: Some task takes the general sampler.
        sorts: The general sampler sorts a whole vocabulary row for some
            task.
        logprob_tasks: The reporting tasks, in task order.
        logprobs: The compact layout of their words, which start the
            binding's logprob column; ``None`` without a reporting task.
        logprob_bucket: ``(rows, top)`` of the ``score`` graph scoring them
            (:func:`logprob_shape`); ``(0, 0)`` without logprobs.
    """

    tasks: int
    width: int
    rows: np.ndarray
    paths: np.ndarray
    draws: np.ndarray
    params: np.ndarray
    slots: np.ndarray
    gates: np.ndarray
    masks: np.ndarray
    mask_rows: np.ndarray
    biased: np.ndarray
    force: np.ndarray
    finish: np.ndarray
    transitions: np.ndarray
    drafts: np.ndarray
    terminal: np.ndarray
    top_k: bool
    categorical: bool
    sorts: bool
    logprob_tasks: np.ndarray
    logprobs: LogprobLayout | None
    logprob_bucket: tuple[int, int]

    @classmethod
    def build(
        cls,
        tasks: Sequence[SampleTask],
        vocab: int,
        capacity: int,
        logprob_capacity: int,
    ) -> SamplingMetadata:
        """Validate ``tasks`` and lay out their selection inputs.

        Args:
            tasks: The forward's sampled calls, in task order.
            vocab: The global vocabulary.
            capacity: The binding's rows: the tasks and token sets it holds.
            logprob_capacity: The logprob words a forward may report.

        Raises:
            NotImplementedError: The vocabulary exceeds the token range.
            ValueError: A task's controls, draws or drafts are malformed.
            AssertionError: More tasks than the binding's rows.
            OutOfMemory: Token sets or logprob words exceed the binding.
        """
        _check_vocabulary(vocab)
        for task in tasks:
            _validate(task, vocab)
        if len(tasks) > capacity:
            raise AssertionError("a forward samples more tasks than its rows")

        words = bitmask_words(vocab)
        width = max((task.rows for task in tasks), default=1)
        count = len(tasks)
        rows = np.zeros((count, width), dtype=np.int64)
        draws = np.zeros((count, width), dtype=np.float32)
        params = np.zeros((count, width, PARAMETER_COLUMNS), dtype=np.float32)
        mask_rows = np.full((count, width), -1, dtype=np.int64)
        finish = np.zeros((count, words), dtype=np.uint32)
        transitions = np.zeros((count, words), dtype=np.uint32)
        drafts = np.full((count, width - 1), -1, dtype=np.int64)
        paths = np.zeros(count, dtype=np.int64)
        terminal = np.zeros(count, dtype=np.int64)
        sorts = False
        sets = _TokenSets(vocab, capacity)
        reporting: list[int] = []
        counts: list[int] = []

        for index, task in enumerate(tasks):
            controls = task.controls
            path = controls.path(task.params, vocab)
            paths[index] = path
            if path == CATEGORICAL:
                sorts |= task.params.sorts(vocab)
            _insert(finish[index], controls.finish_token_ids, vocab)
            if controls.return_transition:
                _insert(
                    transitions[index], controls.transition_token_ids, vocab
                )
            own_drafts = controls.draft_token_ids
            drafts[index, : len(own_drafts)] = own_drafts
            terminal[index] = controls.terminal_draft_prefix or 0
            if task.params.generated_logprobs_requested():
                reporting.append(index)
                counts.append(logprob_count(task.params.n_logprobs, vocab))

            # The task's candidate rows, then copies of its first.
            own = np.arange(width)
            own[own >= task.rows] = 0
            rows[index] = task.row + own
            if task.draws is not None:
                draws[index] = np.asarray(task.draws, dtype=np.float32)[own]
            params[index] = task.params.parameters(vocab)
            for offset in range(width):
                mask_rows[index, offset] = sets.row(
                    controls.allowed[own[offset]], controls.suppress
                )

        layout, bucket = _logprobs(
            np.asarray(counts, dtype=np.int64),
            vocab,
            capacity,
            logprob_capacity,
        )
        return cls(
            tasks=count,
            width=width,
            rows=rows.reshape(-1),
            paths=paths,
            draws=draws.reshape(-1),
            params=params.reshape(-1, PARAMETER_COLUMNS),
            slots=np.array([task.slot for task in tasks], dtype=np.int64),
            gates=np.array([task.gated for task in tasks], dtype=np.int64),
            masks=sets.table(),
            mask_rows=mask_rows.reshape(-1),
            biased=np.array(
                [bool(task.params.logit_bias) for task in tasks],
                dtype=np.uint8,
            ),
            force=np.array(
                [task.controls.force_finish for task in tasks], dtype=np.uint8
            ),
            finish=finish.view(np.int32),
            transitions=transitions.view(np.int32),
            drafts=drafts,
            terminal=terminal,
            top_k=bool(np.any(paths == TOP_K_PATH)),
            categorical=bool(np.any(paths == CATEGORICAL)),
            sorts=sorts,
            logprob_tasks=np.asarray(reporting, dtype=np.int64),
            logprobs=layout,
            logprob_bucket=bucket,
        )

    @classmethod
    def gather(
        cls,
        table: RequestTable,
        slots: np.ndarray,
        rows: np.ndarray,
        gates: np.ndarray,
        draws: np.ndarray,
        capacity: int,
        logprob_capacity: int,
    ) -> SamplingMetadata:
        """Lay out single-row tasks without a sampling state from slot data.

        The result equals :meth:`build` of the same calls' tasks
        (:func:`controls_for_call` with no state and no drafts): every
        column is an array expression over ``slots`` against the request
        constants, and only requests with allowed or forced tokens, or with
        more than :data:`FINISH_IDS` finish tokens, are visited one by one.

        Args:
            table: The slot-indexed request constants.
            slots: ``[T]`` each task's request slot.
            rows: ``[T]`` each task's logits row.
            gates: ``[T]`` whether each task is gated by its continuation
                flag.
            draws: ``[T]`` float32 draws (:func:`task_draws`).
            capacity: The binding's rows.
            logprob_capacity: The logprob words a forward may report.

        Raises:
            NotImplementedError: The vocabulary exceeds the token range.
            AssertionError: More tasks than the binding's rows.
            OutOfMemory: Token sets or logprob words exceed the binding.
        """
        vocab = table.vocab
        _check_vocabulary(vocab)
        slots = np.asarray(slots, dtype=np.int64)
        count = len(slots)
        if count > capacity:
            raise AssertionError("a forward samples more tasks than its rows")
        paths = sampling_paths(table, slots)

        # Finish sets from each slot's first finish tokens, then the full
        # list of the requests that have more.
        words = bitmask_words(vocab)
        finish = np.zeros((count, words), dtype=np.uint32)
        ids = table.finish_ids[slots]
        listed = (ids >= 0) & (ids < vocab)
        task_index = np.broadcast_to(np.arange(count)[:, None], ids.shape)
        index, bits = _token_bits(ids[listed])
        np.bitwise_or.at(finish, (task_index[listed], index), bits)
        for task in np.flatnonzero(table.finish_count[slots] > FINISH_IDS):
            _insert(finish[task], table.finish_tokens[int(slots[task])], vocab)

        # Token sets of the requests with allowed or forced tokens; forced
        # tokens name the call's first (here only) candidate row.
        sets = _TokenSets(vocab, capacity)
        mask_rows = np.full(count, -1, dtype=np.int64)
        forced = table.first_forced[slots]
        constrained = np.flatnonzero((forced >= 0) | table.has_allowed[slots])
        for task in constrained:
            allowed = (
                (int(forced[task]),)
                if forced[task] >= 0
                else table.allowed[int(slots[task])]
            )
            mask_rows[task] = sets.row(allowed, ())

        reporting = np.flatnonzero(table.reports[slots])
        layout, bucket = _logprobs(
            table.n_logprobs[slots][reporting],
            vocab,
            capacity,
            logprob_capacity,
        )
        categorical = paths == CATEGORICAL
        return cls(
            tasks=count,
            width=1,
            rows=np.asarray(rows, dtype=np.int64),
            paths=paths,
            draws=np.asarray(draws, dtype=np.float32),
            params=table.params[slots],
            slots=slots,
            gates=np.asarray(gates, dtype=np.int64),
            masks=sets.table(),
            mask_rows=mask_rows,
            biased=table.biased[slots].astype(np.uint8),
            force=np.zeros(count, dtype=np.uint8),
            finish=finish.view(np.int32),
            transitions=np.zeros((count, words), dtype=np.int32),
            drafts=np.zeros((count, 0), dtype=np.int64),
            terminal=np.zeros(count, dtype=np.int64),
            top_k=bool(np.any(paths == TOP_K_PATH)),
            categorical=bool(np.any(categorical)),
            sorts=bool(np.any(table.sorts[slots] & categorical)),
            logprob_tasks=reporting.astype(np.int64),
            logprobs=layout,
            logprob_bucket=bucket,
        )

    def logprob_end(self) -> int:
        """Return the words of the logprob column the tasks' logprobs pack."""
        return self.logprobs.words if self.logprobs is not None else 0


def sampling_paths(table: RequestTable, slots: np.ndarray) -> np.ndarray:
    """Return the sampler of single-row calls without a sampling state.

    The vectorized :meth:`SamplingControls.path`: a call takes the unshaped
    greedy path when its parameters permit it and no allowed or forced token
    constrains it, the top-k sampler when the parameters permit it and no
    token constrains it, and the general sampler otherwise.

    Args:
        table: The slot-indexed request constants.
        slots: ``[T]`` each call's request slot.

    Returns:
        ``[T]`` int64 samplers.
    """
    unconstrained = (table.first_forced[slots] < 0) & ~table.has_allowed[slots]
    greedy = table.device_greedy[slots] & unconstrained
    top_k = ~greedy & unconstrained & table.top_k_eligible[slots]
    return np.where(
        greedy, GREEDY, np.where(top_k, TOP_K_PATH, CATEGORICAL)
    ).astype(np.int64)


def task_draws(
    table: RequestTable,
    window: DrawWindow,
    slots: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Return single-row tasks' draws at their sampling positions.

    A task of positive temperature draws from its request's stream; any
    other task's draw is zero (the unshaped greedy path has none, and a
    zero-temperature sampler reads none).

    Args:
        table: The slot-indexed request constants.
        window: The rank's cached draws.
        slots: ``[T]`` each task's request slot.
        positions: ``[T]`` each task's semantic sampling position.

    Returns:
        ``[T]`` float32 draws.
    """
    slots = np.asarray(slots, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    draws = np.zeros(len(slots), dtype=np.float32)
    sampled = table.positive_temperature[slots]
    if sampled.any():
        draws[sampled] = window.draws(
            slots[sampled], table.key[slots[sampled]], positions[sampled]
        )
    return draws


class OwnSelection(IntEnum):
    """How a decode forward selects its rows' tokens itself."""

    # The argmax of each row's logits.
    GREEDY = 0
    # Each row sampled from its top-k candidates under its own parameters
    # and draw; greedy rows are its zero-temperature rows.
    TOP_K = 1


def own_selection(
    *,
    sampled: np.ndarray,
    decode: np.ndarray,
    tagged: np.ndarray,
    token_output: np.ndarray,
    transition_output: np.ndarray,
    transitions: np.ndarray,
    finish_count: np.ndarray,
    paths: np.ndarray,
    suppressed: np.ndarray,
) -> OwnSelection | None:
    """Return the selection a decode forward makes for all its calls.

    The forward selects for every call only when each call is a sampled
    decode relayed by its slot's token, with a token relay output, no
    transition output and no transition tokens, at most
    :data:`FINISH_IDS` merged finish tokens and the unsuppressed greedy or
    the top-k sampler. It selects greedily when every call is greedy, else
    each row from its top-k candidates.

    Every argument is a ``[n]`` array over the invocation's input calls.

    Args:
        sampled: The call samples.
        decode: The call is a decode.
        tagged: The call's predicate relays a token.
        token_output: The call has a token relay output.
        transition_output: The call has a transition output.
        transitions: The call's sampling state has transition tokens.
        finish_count: The call's merged finish tokens.
        paths: The call's sampler (any value for an unsampled call).
        suppressed: The call's sampling state suppresses tokens.

    Returns:
        The forward's own selection, or ``None`` when a sampling method
        must follow the forward (or there is no call).
    """
    if len(sampled) == 0:
        return None
    qualifies = (
        sampled
        & decode
        & tagged
        & token_output
        & ~transition_output
        & ~transitions
        & (finish_count <= FINISH_IDS)
        & (((paths == GREEDY) & ~suppressed) | (paths == TOP_K_PATH))
    )
    if not np.all(qualifies):
        return None
    if np.any(paths == TOP_K_PATH):
        return OwnSelection.TOP_K
    return OwnSelection.GREEDY


@dataclass(slots=True)
class RowSampling:
    """Draws and parameters of a decode forward that samples its own rows.

    One row per task, in task order, which is the forward's row order.

    Attributes:
        draws: ``[n]`` float32 draws; zero for a row selecting by argmax.
        params: ``[n, PARAMETER_COLUMNS]`` float32 parameters.
    """

    draws: np.ndarray
    params: np.ndarray

    @classmethod
    def build(cls, tasks: Sequence[SampleTask], vocab: int) -> RowSampling:
        """Return the rows of single-row top-k or unsuppressed greedy tasks.

        Raises:
            ValueError: A task's controls or draws are malformed.
            AssertionError: A task takes another sampler or has more rows.
        """
        draws = np.zeros(len(tasks), dtype=np.float32)
        params = np.zeros((len(tasks), PARAMETER_COLUMNS), dtype=np.float32)
        for index, task in enumerate(tasks):
            _validate(task, vocab)
            path = task.controls.path(task.params, vocab)
            own = path == TOP_K_PATH or (
                path == GREEDY and not task.controls.suppress
            )
            if not own or task.rows != 1:
                raise AssertionError(
                    "a decode forward samples only top-k and greedy rows"
                )
            if task.draws is not None:
                draws[index] = task.draws[0]
            params[index] = task.params.parameters(vocab)
        return cls(draws=draws, params=params)

    @classmethod
    def neutral(cls, rows: int) -> RowSampling:
        """Return ``rows`` neutral rows: zero draws, argmax parameters."""
        return cls(
            draws=np.zeros(rows, dtype=np.float32),
            params=np.tile(NEUTRAL, (rows, 1)),
        )

    def padded(self, rows: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the draws and parameters of ``rows`` forward rows.

        These rows, then neutral padding rows.
        """
        extra = max(rows - len(self.draws), 0)
        draws = np.concatenate([self.draws, np.zeros(extra, np.float32)])
        params = np.concatenate([self.params, np.tile(NEUTRAL, (extra, 1))])
        return draws, params


@dataclass(slots=True)
class PaddedSelection:
    """A forward's selection inputs padded to its task bucket.

    Padding tasks select greedily over logits row 0 in slot 0 with neutral
    parameters, every token allowed, and their results are not read. A
    verification, which selects eagerly, is not padded. The reporting
    tasks pad to their bucket with rows of task 0 and count -1, which
    report nothing. Field shapes follow :class:`SamplingMetadata`, flat.
    """

    tasks: int
    rows: np.ndarray
    paths: np.ndarray
    draws: np.ndarray
    params: np.ndarray
    slots: np.ndarray
    gates: np.ndarray
    mask_rows: np.ndarray
    biased: np.ndarray
    force: np.ndarray
    drafts: np.ndarray
    terminal: np.ndarray
    logprob_tasks: np.ndarray
    logprob_counts: np.ndarray


def _pad(values: np.ndarray, fill: Any, count: int) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    return np.concatenate([flat, np.full(count, fill, dtype=flat.dtype)])


def pad_selection(metadata: SamplingMetadata, capacity: int) -> PaddedSelection:
    """Pad ``metadata``'s inputs on a binding of ``capacity`` rows.

    Single-row tasks pad to their task bucket (:func:`task_bucket`); a
    verification's tasks are not padded.
    """
    width = metadata.width
    tasks = (
        task_bucket(metadata.tasks, capacity) if width == 1 else metadata.tasks
    )
    extra = tasks - metadata.tasks
    bucket_rows, _ = metadata.logprob_bucket
    reporting = len(metadata.logprob_tasks)
    counts = (
        metadata.logprobs.counts
        if metadata.logprobs is not None
        else np.zeros(0, dtype=np.int64)
    )
    return PaddedSelection(
        tasks=tasks,
        rows=_pad(metadata.rows, 0, extra * width),
        paths=_pad(metadata.paths, GREEDY, extra),
        draws=_pad(metadata.draws, 0.0, extra * width),
        params=np.concatenate(
            [metadata.params, np.tile(NEUTRAL, (extra * width, 1))]
        ),
        slots=_pad(metadata.slots, 0, extra),
        gates=_pad(metadata.gates, 0, extra),
        mask_rows=_pad(metadata.mask_rows, -1, extra * width),
        biased=_pad(metadata.biased, 0, extra),
        force=_pad(metadata.force, 0, extra),
        drafts=_pad(metadata.drafts, -1, extra * (width - 1)),
        terminal=_pad(metadata.terminal, 0, extra),
        logprob_tasks=_pad(metadata.logprob_tasks, 0, bucket_rows - reporting),
        logprob_counts=_pad(
            counts.astype(np.int64), -1, bucket_rows - reporting
        ),
    )


def padding_selection(tasks: int) -> PaddedSelection:
    """Return the inputs of ``tasks`` padding tasks, none reporting.

    Startup captures the selection and scoring graphs over them: the values
    do not choose their kernels.
    """
    return PaddedSelection(
        tasks=tasks,
        rows=np.zeros(tasks, dtype=np.int64),
        paths=np.full(tasks, GREEDY, dtype=np.int64),
        draws=np.zeros(tasks, dtype=np.float32),
        params=np.tile(NEUTRAL, (tasks, 1)),
        slots=np.zeros(tasks, dtype=np.int64),
        gates=np.zeros(tasks, dtype=np.int64),
        mask_rows=np.full(tasks, -1, dtype=np.int64),
        biased=np.zeros(tasks, dtype=np.uint8),
        force=np.zeros(tasks, dtype=np.uint8),
        drafts=np.zeros(0, dtype=np.int64),
        terminal=np.zeros(tasks, dtype=np.int64),
        logprob_tasks=np.zeros(tasks, dtype=np.int64),
        logprob_counts=np.full(tasks, -1, dtype=np.int64),
    )
