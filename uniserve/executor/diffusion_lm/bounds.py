"""What a diffusion LM computation sizes its columns, banks and graphs from.

:class:`Facts` are the model's: its block length, vocabulary, widths and
dtypes, pipeline stage, whether the rank denoises blocks and whether
``finish`` exchanges tokens with expert peers. :class:`Bounds` derives every
bound of the computation from the worker's batch limits, the request slots,
the prefix cache's groups and pool, the facts and the served block sampling
(:class:`ServedSampling`): once for capacity planning, before the pool is
fitted, and again at bind over the fitted pool, so that both agree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from uniserve.executor.text.buffers import PageGroup, pool_rows
from uniserve.models import DiffusionLM
from uniserve.runner.diffusion_lm import tail_exchanges

from .shapes import Shapes, chunk_rows

__all__ = ["Bounds", "Denoising", "Facts", "ServedSampling"]


@dataclass(frozen=True, slots=True)
class Facts:
    """What a diffusion LM's rank computes.

    Attributes:
        sequence_length: Tokens of one generated block (``C``).
        vocab_size: Real vocabulary entries (``V``), on every pipeline
            stage.
        hidden_size: Width of the hidden states and token embeddings
            (``H``).
        embedding_dtype: The self-conditioning embeddings' element type.
        attend_widths: Per-token widths of the attend state ``finish``
            completes on the final stage; empty elsewhere.
        compute_dtype: The element type the layers compute in, which the
            attend state has.
        generates: The rank denoises resident blocks (one pipeline stage
            holding the unsharded embedding table); otherwise it serves
            readouts only.
        local_tail: ``finish`` exchanges no tokens across ranks, so it
            follows its readout's expert step and has no expert capacity.
        pipeline_rank: This rank's pipeline stage.
        pipeline_size: The model's pipeline stages.
    """

    sequence_length: int
    vocab_size: int
    hidden_size: int
    embedding_dtype: torch.dtype
    attend_widths: tuple[int, ...]
    compute_dtype: torch.dtype
    generates: bool
    local_tail: bool
    pipeline_rank: int = 0
    pipeline_size: int = 1

    @classmethod
    def read(
        cls, model: DiffusionLM, *, exchange: bool, attention_ranks: int
    ) -> Facts:
        """Read the facts of a placed diffusion LM.

        Args:
            model: The diffusion LM of the computation's component.
            exchange: Routed experts communicate through an expert exchange
                (the deployment's exchange is set and is not DWDP).
            attention_ranks: The attention ranks of a disaggregated expert
                union; zero when every rank holds its experts.
        """
        rank, size = model.stage()
        return cls(
            sequence_length=int(model.sequence_length()),
            vocab_size=int(model.vocab_size()),
            hidden_size=int(model.hidden_size()),
            embedding_dtype=model.embedding_dtype(),
            attend_widths=tuple(int(width) for width in model.attend_widths()),
            compute_dtype=model.compute_dtype(),
            generates=bool(model.generates()),
            local_tail=not tail_exchanges(
                model, exchange=exchange, attention_ranks=attention_ranks
            ),
            pipeline_rank=int(rank),
            pipeline_size=int(size),
        )

    @property
    def last_stage(self) -> bool:
        """Whether the rank holds the head and completes readouts."""
        return self.pipeline_rank + 1 == self.pipeline_size


@dataclass(frozen=True, slots=True)
class ServedSampling:
    """The block sampling a deployment serves (``CanvasSampling``).

    Every generated block of the deployment has ``canvas_length`` tokens and
    takes at most ``max_steps`` denoising steps; a request carries no
    sampling of its own besides its seed.

    Attributes:
        canvas_length: Tokens of one block.
        max_steps: Denoising steps a block runs at most.
        entropy_bound: Entropy, in nats, the accepted samples of one step
            may carry beyond the largest one.
        t_min: Sampling temperature of a block's last step.
        t_max: Sampling temperature of a block's first step.
        confidence_threshold: Mean canvas entropy, in nats, below which a
            stable block stops; zero never stops a block early.
        stability_threshold: Steps the argmax canvas must hold unchanged
            before a block stops; zero stops on confidence alone.
    """

    canvas_length: int
    max_steps: int
    entropy_bound: float
    t_min: float
    t_max: float
    confidence_threshold: float
    stability_threshold: int

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> ServedSampling:
        """Decode the worker settings' JSON of the served sampling."""
        return cls(
            canvas_length=int(value["canvas_length"]),
            max_steps=int(value["max_steps"]),
            entropy_bound=float(value["entropy_bound"]),
            t_min=float(value["t_min"]),
            t_max=float(value["t_max"]),
            confidence_threshold=float(value["confidence_threshold"]),
            stability_threshold=int(value["stability_threshold"]),
        )


class Denoising(NamedTuple):
    """The denoising bounds of a generating rank.

    Attributes:
        rows: Blocks the denoising columns hold: a step's padded rows.
        chunk_rows: Blocks a sampler chunk projects and steps at once.
    """

    rows: int
    chunk_rows: int


@dataclass(frozen=True, slots=True)
class Bounds:
    """Every bound of a diffusion LM computation.

    Attributes:
        sequence_length: Tokens of one block.
        slots: The request slots, the padding slot excluded.
        capacity: Blocks a denoising step may carry: the ``denoise``
            buckets' limit.
        readout_rows: Blocks a readout may carry: the rows of a pass.
        rows: Packed sequences of one forward; an ``attend`` bucket's
            padding doubles its blocks.
        tokens: Tokens of one forward.
        block_size: Tokens per page, per numerical block table.
        windowed: Whether pages retire behind a window, per table.
        widths: Page columns every table's rows hold, per table.
        graphs: Startup captures graphs.
        denoise: The denoising bounds of a generating rank; ``None`` on a
            rank that serves readouts only.
    """

    sequence_length: int
    slots: int
    capacity: int
    readout_rows: int
    rows: int
    tokens: int
    block_size: tuple[int, ...]
    windowed: tuple[bool, ...]
    widths: tuple[int, ...]
    graphs: bool
    denoise: Denoising | None

    @classmethod
    def derive(
        cls,
        *,
        max_batch_calls: int,
        max_batch_tokens: int,
        max_model_len: int,
        slots: int,
        groups: Sequence[PageGroup],
        units: int | None,
        facts: Facts,
        cuda: bool,
        graph_policy: str,
        sampling: ServedSampling | None,
    ) -> Bounds:
        """Derive the bounds of a computation.

        Args:
            max_batch_calls: The calls of one batch.
            max_batch_tokens: The tokens of one batch.
            max_model_len: The tokens of one sequence.
            slots: The request slots, the padding slot excluded.
            groups: The prefix cache's page groups as the worker plans
                them, in table order.
            units: The prefix cache pool's units, sentinel included, once
                fitted; ``None`` before fitting, when the pool bounds
                nothing.
            facts: The model's facts.
            cuda: The computation runs on a CUDA device.
            graph_policy: The worker's graph policy; ``"off"`` captures
                none.
            sampling: The served block sampling; ``None`` when the
                deployment generates no block.

        Raises:
            ValueError: No block fits a batch or the pool, or the served
                canvas length is not the model's block length on a
                generating rank.
        """
        max_rows = min(max_batch_calls, max(slots, 1))
        tokens = max_batch_tokens
        length = facts.sequence_length

        # A generating block holds a page of every group, so the pool bounds
        # the blocks a denoising step may hold. Readout blocks only read
        # their requests' prefixes, which they may share, so a pass carries
        # as many blocks as the engine's rows allow.
        pooled = None if units is None else pool_rows(groups, units)
        graphs = cuda and graph_policy != "off"
        capacity = min(max_rows, tokens // length)
        if pooled is not None:
            capacity = min(capacity, pooled)
        if capacity == 0:
            pool = "an unbounded pool" if pooled is None else pooled
            raise ValueError(
                f"no block of {length} tokens fits a batch of {tokens} "
                f"tokens and a pool of {pool} blocks"
            )
        readout_rows = min(max_rows, tokens)
        rows = 2 * readout_rows if graphs else max_rows

        # Every table of a group selects the group's width.
        block_size: list[int] = []
        windowed: list[bool] = []
        widths: list[int] = []
        for group in groups:
            width = group.table_width(max_model_len, tokens, units)
            block_size += [group.block_size] * group.units_per_page
            windowed += [group.window is not None] * group.units_per_page
            widths += [width] * group.units_per_page

        denoise = None
        if facts.generates and sampling is not None:
            if sampling.canvas_length != length:
                raise ValueError(
                    f"the served canvas length {sampling.canvas_length} is "
                    f"not the model's canvas length {length}"
                )
            # Graphs pad a step to a bucket of at most the capacity; an
            # eager step holds the rows a batch may carry.
            steps = capacity if graphs else min(max_rows, tokens // length)
            steps = min(steps, slots)
            denoise = Denoising(
                steps, chunk_rows(length, facts.vocab_size, steps)
            )

        return cls(
            sequence_length=length,
            slots=slots,
            capacity=capacity,
            readout_rows=readout_rows,
            rows=rows,
            tokens=tokens,
            block_size=tuple(block_size),
            windowed=tuple(windowed),
            widths=tuple(widths),
            graphs=graphs,
            denoise=denoise,
        )

    def shapes(self) -> Shapes:
        """Return the computation's graph buckets.

        A generating computation never steps more blocks than its slots.
        """
        return Shapes(
            self.sequence_length,
            self.capacity,
            self.readout_rows,
            self.tokens,
            self.slots if self.denoise is not None else None,
        )
