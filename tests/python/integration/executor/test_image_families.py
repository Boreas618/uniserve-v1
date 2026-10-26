"""An image model's packed forwards replay their buckets' graphs exactly.

The image executor captures one graph per bucket of each family of packed
forwards: a denoising step's guidance branch rows and a condition write's
latent images as ``(T, R)`` row buckets, a vision encode's images as
``(P, S)`` patch-row buckets, and each image size of the latent codec. The
rows of a forward enter a graph only as device values, so a graph captured
over one set of rows replays any other set that fits its bucket. These
tests capture over synthetic rows, as warmup does, then replay other rows
and compare each live result with the same rows evaluated eagerly at their
exact shape, on one GPU, over tiny BAGEL and SenseNova checkpoints and the
stub model's codec:

- a step's predictions, mixed image sizes in one forward, a bucket whose
  live rows fill all but its padding row and token, and an all-padding
  bucket that leaves the live predictions untouched;
- a condition write's keys and values;
- a vision bucket's features of each packed image;
- a latent encode's rows and a decode's pixels and bytes at a captured size.

Tolerance: a bucket changes the row and token counts of each projection, so
a token's result may differ from the exact forward only by the accumulation
order of its reductions; as in the CPU padding tests, BF16 outputs stay
within two BF16 epsilons (2 * 2**-7) of their scale. The FP32 vision tower
and the codec's graphs replay the eager kernels on the same shapes and are
compared exactly or at FP32 tolerance.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from tests.python.fixtures.checkpoints import (
    bagel_checkpoint,
    load_bagel,
    sensenova_checkpoint,
)
from tests.python.fixtures.stub import LOCAL, load_model, read_config
from uniserve.bindings.context import ExecutionContext
from uniserve.bindings.workspace import Scratch
from uniserve.executor.image.codec import LatentEncoder, PixelDecoder
from uniserve.executor.image.denoiser import (
    CacheTables,
    Denoiser,
    LatentBanks,
    RowPages,
    split_rows,
)
from uniserve.executor.image.layout import ImageLayout
from uniserve.executor.image.rows import ImageRow, RowColumns, RowField
from uniserve.executor.image.shapes import (
    RowLimits,
    condition_rows,
    denoise_rows,
    row_shapes,
    token_sizes,
)
from uniserve.executor.image.vision import VisionEncoder, VisionImage
from uniserve.executor.text.context import ConditionRow
from uniserve.executor.text.protocol import GraphMode
from uniserve.models import PatchEncoder, TextInput, TextSize
from uniserve.models.sensenova_u1 import vision
from uniserve.nn.attention import AttentionBatch, PagedInput
from uniserve.prefix_cache.units import UnitPool

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="the image computations run on a GPU",
    ),
]

DEVICE = torch.device("cuda", 0)
EPSILON = 2**-7
# Slot 1 holds a six-token prompt; slots 2 to 4 receive condition writes.
# Each slot holds up to eight pages of every table.
PROMPT, SLOTS, WIDTH = 6, 5, 8
UNITS = 1 + (SLOTS - 1) * WIDTH
# Images of the steps: at most two images of three branches.
IMAGES = 2


def _seed_generation_experts(model) -> None:
    """Give the flow experts seeded weights; the fixtures zero some."""
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".flow." not in name and not name.endswith("norm.flow.weight"):
                continue
            noise = torch.randn(parameter.shape, generator=generator)
            value = 1 + 0.1 * noise if parameter.ndim == 1 else 0.05 * noise
            parameter.copy_(value.to(parameter.dtype))


class _Session:
    """A tiny image model on the GPU, its prefix cache and two denoisers.

    One denoiser evaluates exact shapes eagerly, the other replays row
    buckets.
    """

    def __init__(self, model, attention: str, block: int) -> None:
        _seed_generation_experts(model)
        self.model = model
        self.attention = attention
        self.block = block
        self.layout = ImageLayout.of(model.denoiser)
        self.pool = UnitPool.allocate(
            model.text.cache_config,
            num_units=UNITS,
            block_size=block,
            device=DEVICE,
        )
        self.tables = len(self.pool.tables)
        groups = [self.pool.groups[table.group] for table in self.pool.tables]
        # Every slot owns its own run of units.
        self.unit_tables = torch.zeros(
            (self.tables, SLOTS, WIDTH), dtype=torch.int32
        )
        for table in range(self.tables):
            for slot in range(1, SLOTS):
                first = 1 + (slot - 1) * WIDTH
                self.unit_tables[table, slot] = torch.arange(
                    first, first + WIDTH
                )
        self.unit_tables = self.unit_tables.to(DEVICE)
        self._prefill()
        self.stream = torch.cuda.Stream(DEVICE)
        cache = CacheTables.of(
            self.pool,
            {
                "unit_tables": self.unit_tables,
                "start_pages": torch.zeros(
                    (len(self.pool.groups), SLOTS),
                    dtype=torch.int32,
                    device=DEVICE,
                ),
                "table_shapes": torch.tensor(
                    [
                        [table.group, groups[index].block_size, -1]
                        for index, table in enumerate(self.pool.tables)
                    ],
                    dtype=torch.int32,
                    device=DEVICE,
                ),
            },
            max_sequence=WIDTH * block,
            max_query=self.layout.max_row_tokens,
        )
        layout = self.layout
        branches = layout.guidance_branches
        self.step_limits = RowLimits(
            min_row_tokens=1 + layout.framing_tokens,
            max_row_tokens=layout.max_row_tokens,
            max_rows=IMAGES * branches,
            max_tokens=layout.max_row_tokens * branches,
        )
        self.condition_limits = None
        if layout.framing_tokens == 2:
            self.condition_limits = RowLimits(
                min_row_tokens=3,
                max_row_tokens=layout.max_row_tokens,
                max_rows=2,
                max_tokens=2 * layout.max_row_tokens,
            )
        dtype = next(model.denoiser.parameters()).dtype
        bank = torch.zeros((2, block, layout.width), dtype=dtype, device=DEVICE)
        self.banks = LatentBanks((bank, bank.clone()))
        self.eager = self._denoiser(cache, graphs=False)
        self.graphs = self._denoiser(cache, graphs=True)
        generator = torch.Generator().manual_seed(29)
        self.samples = torch.randn(
            (2 * layout.max_sequence_tokens, layout.width), generator=generator
        ).to(device=DEVICE, dtype=dtype)
        with torch.inference_mode():
            for denoiser in (self.eager, self.graphs):
                denoiser.workspace[: self.samples.shape[0]].copy_(self.samples)

    def _denoiser(self, cache: CacheTables, *, graphs: bool) -> Denoiser:
        step_shapes = condition_shapes = None
        pool = None
        if graphs:
            pool = torch.cuda.MemPool()
            step_shapes = row_shapes(
                token_sizes(
                    self.step_limits.max_tokens,
                    row_tokens=self.step_limits.max_row_tokens,
                    step=8,
                    linear=32,
                ),
                denoise_rows(
                    guidance_branches=self.layout.guidance_branches,
                    max_images=IMAGES,
                ),
                self.step_limits,
            )
            if self.condition_limits is not None:
                condition_shapes = row_shapes(
                    token_sizes(
                        self.condition_limits.max_tokens,
                        row_tokens=self.condition_limits.max_row_tokens,
                        step=8,
                        linear=32,
                    ),
                    condition_rows(max_calls=self.condition_limits.max_rows),
                    self.condition_limits,
                )
        return Denoiser(
            network=self.model.denoiser,
            owner="model.denoiser",
            layout=self.layout,
            stream=self.stream,
            device=DEVICE.index,
            scratch=Scratch(),
            attention=self.attention,
            tables=cache,
            banks=self.banks,
            slots=SLOTS - 1,
            step_limits=self.step_limits,
            condition_limits=self.condition_limits,
            step_shapes=step_shapes,
            condition_shapes=condition_shapes,
            pool=pool,
            peers=(),
            collectives=None,
            frame=lambda source: (),
        )

    def _prefill(self) -> None:
        """Write slot 1's prompt into the prefix cache as text."""
        tokens = torch.tensor([1, 4, 8, 3, 9, 2], device=DEVICE)
        pages = -(-PROMPT // self.block)
        with ExecutionContext(
            self.model.text, cache=self.pool, attention=self.attention
        ) as context:
            context.prepare(TextSize(PROMPT, 1))
            batch = AttentionBatch.single(
                PagedInput.from_blocks(
                    blocks=(tuple(self.unit_tables[0, 1, :pages].tolist()),),
                    query_lengths=(PROMPT,),
                    prefix_lengths=(0,),
                    block_size=self.block,
                    causal=True,
                    device=DEVICE,
                )
            )
            context.bind_attention(batch)
            with torch.inference_mode():
                self.model.text(
                    TextInput(
                        tokens, torch.arange(PROMPT, device=DEVICE), batch
                    )
                )
        torch.cuda.synchronize(DEVICE)

    def pages(self, columns: RowColumns) -> RowPages:
        """Each live row's pages: its slot's prompt pages, or none."""
        live = columns.column(RowField.PREFIX)[: columns.live]
        counts = np.where(live > 0, -(-PROMPT // self.block), 0)
        counts = np.stack([counts] * self.tables).astype(np.int64)
        return RowPages(counts, np.zeros_like(counts))

    def step(self, denoiser: Denoiser, columns: RowColumns):
        """Run one step forward; return its mode and predictions."""
        with torch.inference_mode():
            denoiser.predictions.fill_(float("nan"))
        mode = denoiser.steps.run(columns, self.pages(columns))
        self.stream.synchronize()
        return mode, denoiser.predictions.clone()


@pytest.fixture(
    scope="module",
    params=[
        # The portable provider evaluates a capture over device lengths.
        ("bagel", "torch", 8, 4),
        ("sensenova_u1", "torch", 8, 4),
        # The native kernels the serving path selects on this device.
        ("bagel", "auto", 128, 16),
    ],
    ids=lambda param: f"{param[0]}-{param[1]}",
)
def session(request, tmp_path_factory) -> _Session:
    name, attention, head_dim, block = request.param
    root = tmp_path_factory.mktemp(name)
    if name == "bagel":
        _, _, config = bagel_checkpoint(root, head_dim=head_dim)
        model = load_bagel(root, config, device=DEVICE)
    else:
        model, _ = sensenova_checkpoint(root, torch.bfloat16, device=DEVICE)
    return _Session(model, attention, block)


def _image(session, image, sample, size, branches, first, time):
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
            timestep=time,
            noise_scale=scale,
            prediction=first + branch * units,
            slot=1 if branch == 0 else 0,
            prefix=PROMPT if branch == 0 else 0,
        )
        for branch in range(branches)
    ]


def _assert_close(actual, expected, rows):
    scale = expected[rows].abs().max().item()
    torch.testing.assert_close(
        actual[rows],
        expected[rows],
        rtol=2 * EPSILON,
        atol=2 * EPSILON * scale,
    )


def _capture_every_bucket(session) -> None:
    """Warm and capture every step bucket over synthetic rows.

    As warmup does: rows of the shortest images reading slot 1's prompt.
    """
    family = session.graphs.steps
    assert family.shapes is not None
    layout = session.layout
    for shape in family.shapes.capture_order():
        lengths = split_rows(shape, session.step_limits, family.shapes)
        assert lengths is not None
        rows, sample = [], 0
        for index, tokens in enumerate(lengths):
            units = tokens - layout.framing_tokens
            rows.append(
                ImageRow(
                    image=index,
                    sample=sample,
                    height=layout.downsample,
                    width=layout.downsample * units,
                    positions=(PROMPT,) * 3,
                    timestep=0.5,
                    prediction=sample,
                    slot=1,
                    prefix=PROMPT,
                )
            )
            sample += units
        columns = RowColumns.of(layout, rows).padded(
            shape.tokens, shape.rows, family.row_tokens
        )
        mode, _ = session.step(session.graphs, columns)
        assert mode == GraphMode.CAPTURE
    family.seal()


def test_warmup_lists_the_steps_then_the_condition_writes(session):
    """Warmup's forwards are the step family's, then the condition writes'.

    A denoiser that writes no latent conditioning (SenseNova frames no
    images) lists no condition writes.
    """
    steps, conditions = session.graphs.warm_forwards(slots=2)
    assert steps and all(forward.family == "step" for forward in steps)
    if session.condition_limits is None:
        assert conditions == []
    else:
        assert conditions
        assert all(forward.family == "condition" for forward in conditions)


@pytest.fixture(scope="module")
def captured(session) -> _Session:
    _capture_every_bucket(session)
    return session


def _sizes(session):
    """Two image sizes of the tiny denoiser: a square and a wider one."""
    side = session.layout.downsample
    return (2 * side, 2 * side), (2 * side, 4 * side)


def test_replayed_steps_equal_each_image_evaluated_alone(captured):
    session = captured
    layout = session.layout
    small, wide = _sizes(session)
    first = _image(session, 0, 0, small, 2, 0, 0.3)
    units = layout.latent_rows(*small)
    second = _image(session, 1, units, wide, 3, 2 * units, 0.7)
    predicted = 2 * units + 3 * layout.latent_rows(*wide)

    # Each image alone, at its exact shape.
    _, alone = session.step(session.eager, RowColumns.of(layout, first))
    _, other = session.step(session.eager, RowColumns.of(layout, second))
    alone[2 * units : predicted] = other[2 * units : predicted]

    # Both images together replay the graph of their bucket, captured over
    # other rows.
    live = RowColumns.of(layout, first + second)
    mode, together = session.step(
        session.graphs, session.graphs.steps.fit(live)
    )
    assert mode == GraphMode.REPLAY
    rows = slice(0, predicted)
    assert torch.isfinite(alone[rows]).all()
    _assert_close(together, alone, rows)


def test_a_bucket_filled_to_its_padding_row_replays_exactly(captured):
    session = captured
    family = session.graphs.steps
    layout = session.layout
    # The largest bucket's live rows all but fill it: one padding row of
    # one token.
    shape = max(family.shapes.shapes, key=lambda s: (s.rows, s.tokens))
    live_rows = shape.rows - 1
    tokens = shape.tokens - 1
    lengths = [tokens // live_rows] * live_rows
    lengths[-1] += tokens - sum(lengths)
    if any(
        length > layout.max_row_tokens or length < 1 + layout.framing_tokens
        for length in lengths
    ):
        pytest.skip("the largest bucket's boundary needs longer rows")
    rows, sample = [], 0
    for index, length in enumerate(lengths):
        units = length - layout.framing_tokens
        rows.append(
            ImageRow(
                image=index,
                sample=sample,
                height=layout.downsample,
                width=layout.downsample * units,
                positions=(PROMPT,) * 3,
                timestep=0.4,
                prediction=sample,
                slot=1,
                prefix=PROMPT,
            )
        )
        sample += units
    live = RowColumns.of(layout, rows)
    assert family.select(live.live, live.live_tokens) == shape
    _, exact = session.step(session.eager, live)
    mode, replayed = session.step(session.graphs, family.fit(live))
    assert mode == GraphMode.REPLAY
    _assert_close(replayed, exact, slice(0, sample))


def test_an_all_padding_bucket_predicts_nothing(captured):
    session = captured
    layout = session.layout
    small, _ = _sizes(session)
    rows = _image(session, 0, 0, small, 2, 0, 0.3)
    _, before = session.step(
        session.graphs, session.graphs.steps.fit(RowColumns.of(layout, rows))
    )
    predicted = 2 * layout.latent_rows(*small)
    family = session.graphs.steps
    smallest = min(family.shapes.shapes, key=lambda s: (s.rows, s.tokens))
    columns = RowColumns.of(layout, []).padded(
        smallest.tokens, smallest.rows, family.row_tokens
    )
    predictions = session.graphs.predictions
    with torch.inference_mode():
        predictions.copy_(before)
    empty = np.zeros((session.tables, 0), np.int64)
    mode = family.run(columns, RowPages(empty, empty.copy()))
    session.stream.synchronize()
    assert mode == GraphMode.REPLAY
    # Every padding token predicts into the dump row, finitely, and the
    # live predictions stay as they were.
    assert torch.isfinite(predictions[-1]).all()
    torch.testing.assert_close(
        predictions[:predicted], before[:predicted], rtol=0, atol=0
    )


def test_condition_writes_replay_the_eager_keys_and_values(session):
    if session.condition_limits is None:
        pytest.skip("the denoiser does not frame images")
    layout = session.layout
    graphs, eager = session.graphs, session.eager
    family = graphs.conditioning
    assert family is not None and family.shapes is not None
    small, wide = _sizes(session)
    side = layout.downsample

    def written(slot, size):
        units = layout.latent_rows(*size)
        return ConditionRow(
            features=session.samples[:units],
            height=size[0],
            width=size[1],
            slot=slot,
            prefix=0,
            position=PROMPT,
        )

    def pages(*sizes):
        counts = [
            -(-layout.sequence_length(*size) // session.block) for size in sizes
        ]
        return np.array(
            [[0] * len(sizes), counts] * len(session.pool.groups),
            dtype=np.int64,
        )

    # Capture every condition bucket over writes into slot 3, as warmup
    # does.
    for shape in family.shapes.capture_order():
        lengths = split_rows(shape, session.condition_limits, family.shapes)
        assert lengths is not None
        sizes = [
            (side, side * (length - layout.framing_tokens))
            for length in lengths
        ]
        graphs.condition_latents(
            [written(3, size) for size in sizes], pages(*sizes), session.stream
        )
    graphs.seal()

    # The eager write of the square image into slot 2, then one replay
    # writing it into slot 3 beside the wider image into slot 4.
    eager.condition_latents([written(2, small)], pages(small), session.stream)
    graphs.condition_latents(
        [written(3, small), written(4, wide)],
        pages(small, wide),
        session.stream,
    )
    session.stream.synchronize()

    # A step reading either slot's conditioning predicts the same.
    visible = layout.sequence_length(*small)
    units = layout.latent_rows(*small)
    results = []
    for slot in (2, 3):
        rows = [
            ImageRow(
                image=0,
                sample=0,
                height=small[0],
                width=small[1],
                positions=(PROMPT + layout.rope_advance,) * 3,
                timestep=0.6,
                slot=slot,
                prefix=visible,
            )
        ]
        columns = RowColumns.of(layout, rows)
        counts = np.full(
            (session.tables, 1), -(-visible // session.block), np.int64
        )
        with torch.inference_mode():
            eager.predictions.fill_(float("nan"))
        eager.steps.run(columns, RowPages(counts, np.zeros_like(counts)))
        session.stream.synchronize()
        results.append(eager.predictions[:units].clone())
    assert torch.isfinite(results[0]).all()
    _assert_close(results[1], results[0], slice(0, units))


def test_vision_buckets_replay_each_image_as_alone():
    torch.manual_seed(13)
    module = PatchEncoder(
        vision.Encoder(vision.Config(16, 24, 0.5, 4, 3, 10000.0)),
        nn.Identity(),
        patch_size=4,
        downsample=2,
        output_size=24,
        output_dtype=torch.float32,
        max_patches=32,
        max_slots=4,
    ).to(DEVICE)
    stream = torch.cuda.Stream(DEVICE)
    encoder = VisionEncoder(
        module=module,
        owner="model.vision_encoder",
        stream=stream,
        device=DEVICE.index,
        scratch=Scratch(),
        attention="torch",
        graphs=True,
        pool=torch.cuda.MemPool(),
        peers=(),
        pixel_dtype=torch.float32,
        row_budget=128,
        max_images=4,
    )
    encoder.warm()
    encoder.seal()
    shapes = [(4, 6), (2, 4), (4, 4)]
    images = [
        torch.randn(rows * columns, 48, device=DEVICE)
        for rows, columns in shapes
    ]
    outputs = [
        torch.full((rows * columns // 4, 24), float("nan"), device=DEVICE)
        for rows, columns in shapes
    ]
    mode = encoder.encode(
        [
            VisionImage(pixels, shape, out)
            for pixels, shape, out in zip(images, shapes, outputs, strict=True)
        ]
    )
    stream.synchronize()
    assert mode == GraphMode.REPLAY
    with torch.inference_mode():
        for pixels, shape, out in zip(images, shapes, outputs, strict=True):
            alone = module.network(
                pixels, torch.tensor([shape], device=DEVICE), (shape,)
            )
            torch.testing.assert_close(out, alone)


def test_codec_graphs_replay_the_eager_encode_and_decode():
    stub = load_model(
        read_config(
            components=json.dumps({"model": LOCAL, "vision_encoder": LOCAL}),
            device="cuda:0",
        ),
        device="cuda:0",
    )
    stream = torch.cuda.Stream(DEVICE)

    def codecs(graphs: bool):
        pool = torch.cuda.MemPool() if graphs else None
        encoder = LatentEncoder(
            module=stub.module.latent_encoder,
            owner="model.latent_encoder",
            stream=stream,
            device=DEVICE.index,
            scratch=Scratch(),
            attention="torch",
            sizes=[(32, 32)],
            pixel_dtype=torch.bfloat16,
            pool=pool,
            peers=(),
        )
        decoder = PixelDecoder(
            module=stub.module.image_decoder,
            owner="model.image_decoder",
            stream=stream,
            device=DEVICE.index,
            scratch=Scratch(),
            attention="torch",
            sizes=[(32, 32, 4)],
            latent_width=3 * 16 * 16,
            latent_dtype=torch.bfloat16,
            pool=pool,
            peers=(),
        )
        encoder.warm(32)
        decoder.warm(32, 4)
        encoder.seal()
        decoder.seal()
        return encoder, decoder

    pixels = torch.rand(3, 32, 32, device=DEVICE).to(torch.bfloat16)
    results = []
    for graphs in (False, True):
        encoder, decoder = codecs(graphs)
        rows = torch.zeros((4, 768), dtype=torch.bfloat16, device=DEVICE)
        encoder.encode(pixels, rows, seed=7, draw=0)
        product = torch.zeros((3, 32, 32), dtype=torch.bfloat16, device=DEVICE)
        staged = torch.zeros(3 * 32 * 32, dtype=torch.uint8).pin_memory()
        mode = decoder.decode(rows, 32, 32, product, staged)
        quantized = torch.zeros_like(staged)
        decoder.quantize(product, (-1.0, 1.0), quantized)
        stream.synchronize()
        assert mode == (GraphMode.REPLAY if graphs else GraphMode.EAGER)
        results.append((rows, product, staged, quantized))
    for eager, replayed in zip(*results, strict=True):
        assert torch.equal(eager.cpu(), replayed.cpu())
