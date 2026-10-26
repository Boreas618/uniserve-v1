"""Padding and empty rows of a bucketed denoiser step change no live result.

A captured step evaluates its rows in a bucket of more rows and tokens than
they hold: empty rows of no tokens, and a padding row holding the remaining
tokens, which attends to no prefix, reads no sample and predicts into the
dump row. These tests run the image denoiser's packed step forward
(``RowFamily``) on CPU over tiny BAGEL and SenseNova checkpoints whose
generation experts carry seeded weights, with rows attending to a prompt
prefilled into the prefix cache, and compare each bucketed forward with the
exact forward of each image alone.

Tolerance: a bucket changes only the row count of each projection and
which rows surround each sequence, so a token's result can differ only by
the FP32 accumulation order of its reductions. Each BF16 rounding point then
moves by at most one rounding step (2**-8 relative); over the networks'
few rounding points that stays within two BF16 epsilons (2 * 2**-7) of the
predictions' scale.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.python.fixtures.checkpoints import (
    bagel_checkpoint,
    load_bagel,
    sensenova_checkpoint,
)
from uniserve.bindings.context import ExecutionContext
from uniserve.bindings.workspace import Scratch
from uniserve.executor.image.denoiser import (
    CacheTables,
    Denoiser,
    LatentBanks,
    RowPages,
)
from uniserve.executor.image.layout import ImageLayout
from uniserve.executor.image.rows import (
    ImageRow,
    RowColumns,
    RowField,
)
from uniserve.executor.image.shapes import RowLimits
from uniserve.models import TextInput, TextSize
from uniserve.nn.attention import AttentionBatch, PagedInput
from uniserve.prefix_cache.units import UnitPool

pytestmark = pytest.mark.unit

ROWS, TOKENS, UNITS = 12, 96, 32
# Slot 1 holds a six-token prompt in two four-token pages.
PROMPT, BLOCK, SLOTS = 6, 4, 2
EPSILON = 2**-7


def _seed_generation_experts(model) -> None:
    """Give the flow experts seeded weights; the fixtures zero some."""
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".flow." not in name and not name.endswith("norm.flow.weight"):
                continue
            noise = torch.randn(parameter.shape, generator=generator)
            # Norm gains near one, projections at the fixtures' scale.
            value = 1 + 0.1 * noise if parameter.ndim == 1 else 0.05 * noise
            parameter.copy_(value.to(parameter.dtype))


class _Session:
    """A loaded tiny image model, its prefix cache and its denoiser."""

    def __init__(self, model) -> None:
        _seed_generation_experts(model)
        self.model = model
        self.layout = ImageLayout.of(model.denoiser)
        self.pool = UnitPool.allocate(
            model.text.cache_config, num_units=8, block_size=BLOCK, device="cpu"
        )
        self.tables = len(self.pool.tables)
        assert self.tables == 1
        groups = [self.pool.groups[table.group] for table in self.pool.tables]
        self.windowed = sum(group.window is not None for group in groups)
        # Each table gives slot 1 its own two units.
        unit_tables = torch.zeros((self.tables, SLOTS, 2), dtype=torch.int32)
        for table in range(self.tables):
            unit_tables[table, 1] = torch.tensor([1 + 2 * table, 2 + 2 * table])
        self._prefill(unit_tables)

        dtype = next(model.denoiser.parameters()).dtype
        generator = torch.Generator().manual_seed(29)
        samples = torch.randn(
            (UNITS, self.layout.width), generator=generator
        ).to(dtype)
        tables = CacheTables.of(
            self.pool,
            {
                "unit_tables": unit_tables,
                "start_pages": torch.zeros(
                    (len(self.pool.groups), SLOTS), dtype=torch.int32
                ),
                "table_shapes": torch.tensor(
                    [
                        [table.group, groups[index].block_size, -1]
                        for index, table in enumerate(self.pool.tables)
                    ],
                    dtype=torch.int32,
                ),
            },
            max_sequence=2 * BLOCK,
            max_query=self.layout.max_row_tokens,
        )
        self._tables = tables
        self._samples = samples
        self.denoiser = self.make(multiple=1)

    def make(self, multiple: int) -> Denoiser:
        """A denoiser of exact forwards over the session's cache.

        ``multiple`` divides its forwards' tokens.
        """
        dtype = self._samples.dtype
        bank = torch.zeros((2, BLOCK, self.layout.width), dtype=dtype)
        denoiser = Denoiser(
            network=self.model.denoiser,
            owner="model.denoiser",
            layout=self.layout,
            stream=None,
            device=-1,
            scratch=Scratch(),
            attention="torch",
            tables=self._tables,
            banks=LatentBanks((bank, bank.clone())),
            slots=SLOTS,
            step_limits=RowLimits(
                min_row_tokens=1 + self.layout.framing_tokens,
                max_row_tokens=self.layout.max_row_tokens,
                max_rows=ROWS,
                max_tokens=TOKENS,
            ),
            condition_limits=None,
            step_shapes=None,
            condition_shapes=None,
            pool=None,
            peers=(),
            collectives=None,
            frame=lambda source: (),
            multiple=multiple,
        )
        with torch.inference_mode():
            denoiser.workspace[:UNITS].copy_(self._samples)
        return denoiser

    def _prefill(self, unit_tables: torch.Tensor) -> None:
        """Write slot 1's prompt into the prefix cache as text."""
        tokens = torch.tensor([1, 4, 8, 3, 9, 2])
        with ExecutionContext(
            self.model.text, cache=self.pool, attention="torch"
        ) as context:
            context.prepare(TextSize(PROMPT, 1))
            # The tiny checkpoints' layers share one cache table.
            batch = AttentionBatch.single(
                PagedInput.from_blocks(
                    blocks=(tuple(unit_tables[0, 1].tolist()),),
                    query_lengths=(PROMPT,),
                    prefix_lengths=(0,),
                    block_size=BLOCK,
                    causal=True,
                    device="cpu",
                )
            )
            context.bind_attention(batch)
            with torch.no_grad():
                self.model.text(TextInput(tokens, torch.arange(PROMPT), batch))

    def step(
        self, columns: RowColumns, denoiser: Denoiser | None = None
    ) -> torch.Tensor:
        """Run one packed denoising forward; return the predictions."""
        denoiser = denoiser or self.denoiser
        live = columns.column(RowField.PREFIX)[: columns.live]
        pages = np.where(live > 0, -(-PROMPT // BLOCK), 0)
        counts = np.stack([pages] * self.tables).astype(np.int64)
        predictions = denoiser.predictions
        with torch.inference_mode():
            predictions.fill_(float("nan"))
        denoiser.steps.run(columns, RowPages(counts, np.zeros_like(counts)))
        return predictions.clone()


@pytest.fixture(scope="module", params=["bagel", "sensenova_u1"])
def session(request, tmp_path_factory) -> _Session:
    root = tmp_path_factory.mktemp(request.param)
    if request.param == "bagel":
        _, _, config = bagel_checkpoint(root)
        return _Session(load_bagel(root, config))
    model, _ = sensenova_checkpoint(root, torch.bfloat16)
    return _Session(model)


def _image(session, image, sample, size, branches, first):
    """One image's rows, one per guidance branch.

    The conditioned branch reads slot 1's prompt and continues after it;
    the unconditional ones read no prefix.
    """
    height, width = size
    units = session.layout.latent_rows(height, width)
    scale = session.model.denoiser.noise_scale.scale(units)
    return [
        ImageRow(
            image=image,
            sample=sample,
            height=height,
            width=width,
            positions=(PROMPT if branch == 0 else 0,) * 3,
            timestep=0.25 + 0.5 * image,
            noise_scale=scale,
            prediction=first + branch * units,
            slot=1 if branch == 0 else 0,
            prefix=PROMPT if branch == 0 else 0,
        )
        for branch in range(branches)
    ]


def _assert_rows_equal(actual, expected, rows):
    scale = expected[rows].abs().max().item()
    torch.testing.assert_close(
        actual[rows],
        expected[rows],
        rtol=2 * EPSILON,
        atol=2 * EPSILON * scale,
    )


def _two_images(session):
    # An 8x8 image (four latent rows) on two branches, then an 8x16 image
    # (eight latent rows) on three; predictions are consecutive per image.
    first = _image(session, 0, 0, (8, 8), 2, 0)
    second = _image(session, 1, 4, (8, 16), 3, 8)
    return first, second, 8 + 24


def test_a_bucketed_step_equals_each_image_alone(session):
    first, second, predicted = _two_images(session)
    layout = session.layout
    alone = session.step(RowColumns.of(layout, first)).clone()
    alone[8:predicted] = session.step(RowColumns.of(layout, second))[
        8:predicted
    ]

    live = RowColumns.of(layout, first + second)
    together = session.step(live)
    bucketed = session.step(
        live.padded(live.tokens + 8, ROWS, layout.max_row_tokens)
    )

    rows = slice(0, predicted)
    assert torch.isfinite(alone[rows]).all()
    # Mixed sizes in one forward equal separate forwards, and so does a
    # bucket with empty rows and a padding row.
    _assert_rows_equal(together, alone, rows)
    _assert_rows_equal(bucketed, alone, rows)


@pytest.mark.parametrize("spare_rows", [1, ROWS - 5])
def test_buckets_at_their_boundaries_leave_live_rows_unchanged(
    session, spare_rows
):
    first, second, predicted = _two_images(session)
    live = RowColumns.of(session.layout, first + second)
    exact = session.step(live)
    # One padding token, and either no empty row (the live rows fill all
    # but the padding row) or many.
    bucketed = session.step(
        live.padded(
            live.tokens + 1,
            live.rows + spare_rows,
            session.layout.max_row_tokens,
        )
    )
    _assert_rows_equal(bucketed, exact, slice(0, predicted))


def test_an_all_padding_bucket_is_finite_and_predicts_nothing(session):
    first, second, predicted = _two_images(session)
    before = session.step(RowColumns.of(session.layout, first + second))
    columns = RowColumns.of(session.layout, []).padded(
        session.layout.max_row_tokens, ROWS, session.layout.max_row_tokens
    )
    predictions = session.denoiser.predictions
    with torch.inference_mode():
        predictions.copy_(before)
    empty = np.zeros((session.tables, 0), np.int64)
    session.denoiser.steps.run(columns, RowPages(empty, empty.copy()))
    # Every padding token predicts into the dump row, finitely, and the live
    # predictions stay as they were.
    assert torch.isfinite(predictions[-1]).all()
    torch.testing.assert_close(
        predictions[:predicted], before[:predicted], rtol=0, atol=0
    )


def test_sequence_parallel_forwards_pad_to_the_degree_alone(session):
    # Without buckets a forward under sequence parallelism of two still
    # pads to an even token count with one padding row, which changes no
    # live prediction.
    first, second, predicted = _two_images(session)
    live = RowColumns.of(session.layout, first + second)
    exact = session.step(live)
    parallel = session.make(multiple=2)
    padded = parallel.steps.fit(live)
    assert padded.tokens % 2 == 0 and padded.tokens > live.tokens
    assert padded.rows == live.rows + 1
    _assert_rows_equal(
        session.step(padded, parallel), exact, slice(0, predicted)
    )
