"""Per-slot request constants and continuation state of a text computation.

A request slot ``s`` in ``1..=slots`` holds one admitted request; slot 0 is
padding everywhere, so every slot-indexed array has ``S = slots + 1`` rows.

:class:`RequestTable` keeps what a request's sampling parameters decide once
at its start, in slot-indexed numpy arrays, so that a step resolves its
calls' samplers, parameters, finish tokens and logprob reports by indexing
with its slots instead of reading each request.

:class:`SlotState` is the device state a request's next token call reads
without a host round trip:

| Field | Element | Shape | Meaning |
| --- | --- | --- | --- |
| ``future_input_tokens`` | int64 | ``[S, 1]`` | next input token |
| ``predicates`` | uint8 | ``[S]`` | the last selection continues |
| ``logical_lengths`` | int64 | ``[S]`` | next token's logical position |
| ``sampling_positions`` | int64 | ``[S]`` | semantic draw position |
| ``penalty_counts`` | int32 | ``[S, V]`` | generated-token counts |
| ``prompt_logits`` | model | ``[S, V]`` | a scored chunk's last logits |
| ``logit_bias`` | float32 | ``[S, V]`` | the request's logit biases |
| ``adapters`` | int64 | ``[S]`` | adapter pool slot, 0 for none |
| ``verified_lengths`` | int32 | ``[S]`` | verified cache length |

``verified_lengths`` belongs to the worker's block tables, which attention
gathers read too; the state borrows it. Every method enqueues its writes on
the current stream of the tensors' device: the caller orders them on the
primary stream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from uniserve.executor.arena import ALLOCATION_ROUNDING
from uniserve.tensors import staged

from .philox import request_sampling_key
from .sampling import FINISH_IDS, NEUTRAL, SamplingParams, logprob_count

__all__ = ["NewRequest", "RequestTable", "SlotState"]

# The fields :class:`SlotState` allocates, the borrowed ``verified_lengths``
# excluded.
_FIELDS = 8


@dataclass(frozen=True, slots=True)
class NewRequest:
    """What a text computation keeps of a request's start (``ipc::NewRequest``).

    Attributes:
        slot: The request's slot (``request_pool_idx``), positive.
        engine_id: The request key's engine.
        request_id: The request key's identifier.
        epoch: The request key's admission epoch.
        sampling: The admitted sampling parameters; ``None`` for a request
            that generates no tokens, which no text call may compute for.
        finish_token_ids: Tokens that finish the request.
        generated_token_ids: Tokens the request generated on its previous
            engines, which its penalties count.
        adapter: The adapter pool slot it computes with, 0 for none.
        input_images: The input images the request carries.
    """

    slot: int
    engine_id: int
    request_id: int
    epoch: int
    sampling: SamplingParams | None
    finish_token_ids: tuple[int, ...] = ()
    generated_token_ids: tuple[int, ...] = ()
    adapter: int = 0
    input_images: int = 0

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> NewRequest:
        """Decode the JSON of ``ipc::NewRequest``."""
        key = value["request_key"]
        generation = value.get("generation")
        sampling = None
        finish: tuple[int, ...] = ()
        generated: tuple[int, ...] = ()
        adapter = 0
        if generation is not None:
            sampling = SamplingParams.from_json(generation["sampling"])
            finish = tuple(int(t) for t in generation["finish_token_ids"])
            generated = tuple(
                int(t) for t in generation.get("generated_token_ids", ())
            )
            adapter = int(generation.get("adapter", 0))
        return cls(
            slot=int(value["request_pool_idx"]),
            engine_id=int(key["engine_id"]),
            request_id=int(key["request_id"]),
            epoch=int(key["request_epoch"]),
            sampling=sampling,
            finish_token_ids=finish,
            generated_token_ids=generated,
            adapter=adapter,
            input_images=int(value.get("input_images", 0)),
        )


class RequestTable:
    """Slot-indexed constants of the requests a text computation serves.

    Each array has one row per slot, row 0 the padding slot. A slot without
    a request holds neutral values: greedy, no constraint, no report.

    Attributes:
        vocab: The global vocabulary.
        admitted: ``bool[S]`` the slot holds a request generating tokens.
        engine_id, request_id, epoch: ``int64[S]`` the request key.
        params: ``float32[S, PARAMETER_COLUMNS]`` the parameter row.
        key: ``uint64[S]`` the request's sampling stream key.
        positive_temperature: ``bool[S]`` a positive temperature (draws needed).
        device_greedy: ``bool[S]`` the parameters permit an unshaped
            argmax.
        top_k_eligible: ``bool[S]`` the parameters permit the top-k sampler.
        has_allowed: ``bool[S]`` the parameters restrict allowed tokens.
        first_forced: ``int64[S]`` the first forced token, or -1.
        biased: ``bool[S]`` the request has logit biases.
        penalties: ``bool[S]`` token-count penalties apply.
        sorts: ``bool[S]`` a general-path row sorts its vocabulary row.
        reports: ``bool[S]`` generated-token logprobs are reported.
        n_logprobs: ``int64[S]`` reported top tokens per generated token.
        scores_prompt: ``bool[S]`` prompt logprobs are reported.
        n_prompt_logprobs: ``int64[S]`` reported top tokens per prompt
            token.
        finish_ids: ``int64[S, FINISH_IDS]`` the first finish tokens,
            padded with -1.
        finish_count: ``int64[S]`` the admitted finish tokens.
        input_images: ``int64[S]`` the input images of the request.
        sampling: Per slot, the admitted parameters, for calls resolved one
            by one.
        finish_tokens: Per slot, the full admitted finish list.
        allowed: Per slot, the allowed tokens of a restricted request.
    """

    def __init__(self, slots: int, vocab: int) -> None:
        if slots <= 0 or vocab <= 0:
            raise ValueError("a request table holds slots over a vocabulary")
        rows = slots + 1
        self.vocab = vocab
        self.admitted = np.zeros(rows, dtype=bool)
        self.engine_id = np.zeros(rows, dtype=np.int64)
        self.request_id = np.zeros(rows, dtype=np.int64)
        self.epoch = np.zeros(rows, dtype=np.int64)
        self.params = np.zeros((rows, len(NEUTRAL)), dtype=np.float32)
        self.key = np.zeros(rows, dtype=np.uint64)
        self.positive_temperature = np.zeros(rows, dtype=bool)
        self.device_greedy = np.zeros(rows, dtype=bool)
        self.top_k_eligible = np.zeros(rows, dtype=bool)
        self.has_allowed = np.zeros(rows, dtype=bool)
        self.first_forced = np.zeros(rows, dtype=np.int64)
        self.biased = np.zeros(rows, dtype=bool)
        self.penalties = np.zeros(rows, dtype=bool)
        self.sorts = np.zeros(rows, dtype=bool)
        self.reports = np.zeros(rows, dtype=bool)
        self.n_logprobs = np.zeros(rows, dtype=np.int64)
        self.scores_prompt = np.zeros(rows, dtype=bool)
        self.n_prompt_logprobs = np.zeros(rows, dtype=np.int64)
        self.finish_ids = np.zeros((rows, FINISH_IDS), dtype=np.int64)
        self.finish_count = np.zeros(rows, dtype=np.int64)
        self.input_images = np.zeros(rows, dtype=np.int64)
        self.sampling: dict[int, SamplingParams] = {}
        self.finish_tokens: dict[int, tuple[int, ...]] = {}
        self.allowed: dict[int, tuple[int, ...]] = {}
        self.release(range(rows))

    @property
    def slots(self) -> int:
        """Return the request slots, the padding slot excluded."""
        return len(self.admitted) - 1

    def _require(self, slots: np.ndarray) -> None:
        if np.any((slots <= 0) | (slots > self.slots)):
            raise ValueError("request slot is outside the request table")

    def start(self, request: NewRequest) -> None:
        """Install a started request's constants in its slot.

        Raises:
            ValueError: The slot is the padding slot or outside the table.
        """
        slot = request.slot
        self._require(np.asarray([slot]))
        self.release([slot])
        self.engine_id[slot] = request.engine_id
        self.request_id[slot] = request.request_id
        self.epoch[slot] = request.epoch
        self.input_images[slot] = request.input_images
        params = request.sampling
        if params is None:
            return

        vocab = self.vocab
        self.admitted[slot] = True
        self.sampling[slot] = params
        self.params[slot] = params.parameters(vocab)
        self.key[slot] = request_sampling_key(
            params.seed, request.engine_id, request.request_id, request.epoch
        )
        self.positive_temperature[slot] = params.temperature > 0.0
        self.device_greedy[slot] = params.device_greedy()
        self.top_k_eligible[slot] = params.top_k_eligible(vocab)
        self.has_allowed[slot] = params.allowed_token_ids is not None
        if params.allowed_token_ids is not None:
            self.allowed[slot] = params.allowed_token_ids
        self.first_forced[slot] = (
            params.forced_token_ids[0] if params.forced_token_ids else -1
        )
        self.biased[slot] = bool(params.logit_bias)
        self.penalties[slot] = params.uses_penalties()
        self.sorts[slot] = params.sorts(vocab)
        self.reports[slot] = params.generated_logprobs_requested()
        self.n_logprobs[slot] = logprob_count(params.n_logprobs, vocab)
        self.scores_prompt[slot] = params.prompt_logprobs_requested()
        self.n_prompt_logprobs[slot] = logprob_count(
            params.n_prompt_logprobs, vocab
        )
        finish = request.finish_token_ids
        self.finish_tokens[slot] = finish
        self.finish_count[slot] = len(finish)
        first = finish[:FINISH_IDS]
        self.finish_ids[slot, : len(first)] = first

    def release(self, slots: Sequence[int] | range | np.ndarray) -> None:
        """Return ``slots`` to the neutral values of a slot without request."""
        index = np.asarray(list(slots), dtype=np.int64)
        self.admitted[index] = False
        self.engine_id[index] = 0
        self.request_id[index] = 0
        self.epoch[index] = 0
        self.params[index] = NEUTRAL
        self.key[index] = 0
        self.positive_temperature[index] = False
        self.device_greedy[index] = True
        self.top_k_eligible[index] = True
        self.has_allowed[index] = False
        self.first_forced[index] = -1
        self.biased[index] = False
        self.penalties[index] = False
        self.sorts[index] = False
        self.reports[index] = False
        self.n_logprobs[index] = 0
        self.scores_prompt[index] = False
        self.n_prompt_logprobs[index] = 0
        self.finish_ids[index] = -1
        self.finish_count[index] = 0
        self.input_images[index] = 0
        for slot in index.tolist():
            self.sampling.pop(slot, None)
            self.finish_tokens.pop(slot, None)
            self.allowed.pop(slot, None)

    def check(
        self,
        slots: np.ndarray,
        request_ids: np.ndarray,
        epochs: np.ndarray,
    ) -> None:
        """Check that calls name the requests their slots hold.

        Args:
            slots: ``[n]`` each call's slot.
            request_ids: ``[n]`` each call's request identifier.
            epochs: ``[n]`` each call's admission epoch.

        Raises:
            ValueError: A slot is outside the table, holds no request
                generating tokens, or holds another request.
        """
        slots = np.asarray(slots, dtype=np.int64)
        self._require(slots)
        if not np.all(self.admitted[slots]):
            raise ValueError(
                "a text call's slot holds no request generating tokens"
            )
        if np.any(self.request_id[slots] != request_ids) or np.any(
            self.epoch[slots] != epochs
        ):
            raise ValueError("a call names a request its slot does not hold")


class SlotState:
    """The continuation state of every request slot of a text computation.

    Args:
        future_input_tokens: ``int64 [S, 1]``.
        predicates: ``uint8 [S]``.
        logical_lengths: ``int64 [S]``.
        sampling_positions: ``int64 [S]``.
        penalty_counts: ``int32 [S, V]``.
        prompt_logits: ``[S, V]`` in the model dtype.
        logit_bias: ``float32 [S, V]``.
        adapters: ``int64 [S]``.
        verified_lengths: ``int32 [S]``, the block tables' column.
    """

    def __init__(
        self,
        *,
        future_input_tokens: torch.Tensor,
        predicates: torch.Tensor,
        logical_lengths: torch.Tensor,
        sampling_positions: torch.Tensor,
        penalty_counts: torch.Tensor,
        prompt_logits: torch.Tensor,
        logit_bias: torch.Tensor,
        adapters: torch.Tensor,
        verified_lengths: torch.Tensor,
    ) -> None:
        rows, vocab = penalty_counts.shape
        expected = {
            "future_input_tokens": (future_input_tokens, torch.int64),
            "predicates": (predicates, torch.uint8),
            "logical_lengths": (logical_lengths, torch.int64),
            "sampling_positions": (sampling_positions, torch.int64),
            "penalty_counts": (penalty_counts, torch.int32),
            "logit_bias": (logit_bias, torch.float32),
            "adapters": (adapters, torch.int64),
            "verified_lengths": (verified_lengths, torch.int32),
        }
        for name, (tensor, dtype) in expected.items():
            if tensor.dtype != dtype or tensor.shape[0] != rows:
                raise ValueError(f"slot state {name} has the wrong layout")
        if (
            vocab == 0
            or tuple(future_input_tokens.shape) != (rows, 1)
            or tuple(prompt_logits.shape) != (rows, vocab)
            or tuple(logit_bias.shape) != (rows, vocab)
        ):
            raise ValueError(
                "slot state requires a vocabulary and per-slot rows"
            )
        self.rows = rows
        self.vocab = vocab
        self.future_input_tokens = future_input_tokens
        self.predicates = predicates
        self.logical_lengths = logical_lengths
        self.sampling_positions = sampling_positions
        self.penalty_counts = penalty_counts
        self.prompt_logits = prompt_logits
        self.logit_bias = logit_bias
        self.adapters = adapters
        self.verified_lengths = verified_lengths

    # The most bytes the allocator's rounding adds to the fields, each a
    # tensor of its own.
    ALIGNMENT_SLACK = _FIELDS * (ALLOCATION_ROUNDING - 1)

    @staticmethod
    def slot_bytes(vocab: int, logits: torch.dtype) -> int:
        """Return the device bytes of one slot's row of every field.

        The state of ``S`` rows takes at most ``S`` times this plus
        :attr:`ALIGNMENT_SLACK`; the borrowed ``verified_lengths`` is not
        counted.
        """
        itemsize = torch.empty((), dtype=logits).element_size()
        # Next token, logical and sampling positions and adapter (int64
        # each) and the continuation flag (uint8), then the penalty counts
        # (int32), retained logits and logit biases (float32) over the
        # vocabulary.
        return 4 * 8 + 1 + vocab * (4 + itemsize + 4)

    @classmethod
    def allocate(
        cls,
        slots: int,
        vocab: int,
        logits: torch.dtype,
        verified_lengths: torch.Tensor,
    ) -> SlotState:
        """Allocate zeroed state beside ``verified_lengths`` and reset it.

        The tensors live on ``verified_lengths``' device; the writes are
        enqueued on its current stream.
        """
        rows = slots + 1
        device = verified_lengths.device

        def zeros(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, device=device)

        state = cls(
            future_input_tokens=zeros((rows, 1), torch.int64),
            predicates=zeros((rows,), torch.uint8),
            logical_lengths=zeros((rows,), torch.int64),
            sampling_positions=zeros((rows,), torch.int64),
            penalty_counts=zeros((rows, vocab), torch.int32),
            prompt_logits=zeros((rows, vocab), logits),
            logit_bias=zeros((rows, vocab), torch.float32),
            adapters=zeros((rows,), torch.int64),
            verified_lengths=verified_lengths,
        )
        state.reset(range(rows))
        return state

    def _slots(self, slots: Sequence[int] | range | np.ndarray) -> torch.Tensor:
        index = np.asarray(list(slots), dtype=np.int64)
        if np.any((index < 0) | (index >= self.rows)):
            raise ValueError("request slot is outside the slot state")
        device = self.penalty_counts.device
        return staged(torch.from_numpy(index), device).to(
            device, non_blocking=True
        )

    def _slot(self, slot: int) -> int:
        if not 0 <= slot < self.rows:
            raise ValueError("request slot is outside the slot state")
        return slot

    def reset(self, slots: Sequence[int] | range | np.ndarray) -> None:
        """Reset newly admitted slots.

        A reset slot's next token is 1; it has no continuation, zero
        logical and sampling positions, verified length and penalty counts,
        and no adapter. Its retained prompt logits are read only after a
        scored chunk wrote them, so they are not reset; its logit biases
        are installed only for a request that has some
        (:meth:`install_bias`), and read only by such a request's tasks.

        Raises:
            ValueError: A slot lies outside the state.
        """
        index = self._slots(slots)
        self.future_input_tokens.index_fill_(0, index, 1)
        self.predicates.index_fill_(0, index, 0)
        self.logical_lengths.index_fill_(0, index, 0)
        self.sampling_positions.index_fill_(0, index, 0)
        self.adapters.index_fill_(0, index, 0)
        self.verified_lengths.index_fill_(0, index, 0)
        self.penalty_counts.index_fill_(0, index, 0)

    def select_adapter(self, slot: int, adapter: int) -> None:
        """Have slot ``slot``'s request compute with adapter ``adapter``."""
        slot = self._slot(slot)
        # A fill passes the index to its kernel; an indexed assignment would
        # copy it from the host and wait for the stream.
        self.adapters[slot : slot + 1].fill_(int(adapter))

    def count_generated(self, slot: int, tokens: Sequence[int]) -> None:
        """Write slot ``slot``'s penalty counts of previously generated tokens.

        The tokens a request generated on a predecessor engine count as
        generated for its penalties. The slot's whole row is written, after
        :meth:`reset` zeroed it.

        Raises:
            ValueError: The slot lies outside the state, or a token outside
                the vocabulary.
        """
        slot = self._slot(slot)
        if len(tokens) == 0:
            return
        values = np.asarray(tokens, dtype=np.int64)
        if np.any((values < 0) | (values >= self.vocab)):
            raise ValueError("a generated token is outside the vocabulary")
        counts = np.bincount(values, minlength=self.vocab).astype(np.int32)
        self.penalty_counts[slot].copy_(
            staged(torch.from_numpy(counts), self.penalty_counts.device),
            non_blocking=True,
        )

    def install_bias(
        self, slot: int, biases: Sequence[tuple[int, float]]
    ) -> None:
        """Install ``biases`` as slot ``slot``'s logit bias row.

        The row's tokens outside the list are zero; a later bias of a token
        replaces an earlier one, and a bias of a token outside the
        vocabulary has no logit to shift.

        Raises:
            ValueError: The slot lies outside the state.
        """
        slot = self._slot(slot)
        row = np.zeros(self.vocab, dtype=np.float32)
        for token, bias in biases:
            if 0 <= token < self.vocab:
                row[token] = bias
        self.logit_bias[slot].copy_(
            staged(torch.from_numpy(row), self.logit_bias.device),
            non_blocking=True,
        )

    def set_predicates(
        self, slots: Sequence[int] | range | np.ndarray, value: bool
    ) -> None:
        """Set ``slots``' continuation flags to ``value``.

        Warmup runs its captured decode rows as active continuations.

        Raises:
            ValueError: A slot lies outside the state.
        """
        index = self._slots(slots)
        self.predicates.index_fill_(0, index, int(value))
