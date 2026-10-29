"""A stored trajectory laid out in the pages of another parallel
configuration (``uniserve.executor.video``).

A video denoising rank declares how its request slot pages hold its share
of a request's samples (``SampleLayout.storage``, the copy manifest
library's ``StorageLayout``); a rank that restores a trajectory stored by
the ranks of another parallel configuration plans its copies from their
layouts to its own with the worker library (``latent_restore_copies``, the
planner the worker runs at a restore) and executes them as a latent
preparation does (``copy_runs``). The tests store the samples of a
sequence-parallel denoiser of one and of two ranks, whose ranks hold
contiguous runs of the packed rows as FastH3's denoiser divides them, and
check that every restoring rank's pages hold exactly the values its layout
declares, bit for bit: a change of distribution moves values and computes
none. The planner and the copies run on the host.
"""  # noqa: D205

from __future__ import annotations

import json

import pytest
import torch

from uniserve import worker
from uniserve.executor.trajectories import copy_runs
from uniserve.executor.video.plan import (
    REQUEST_PAGES,
    SAMPLE_ALIGNMENT,
    LayoutFields,
    SampleLayout,
    SampleRows,
    StateField,
)

pytestmark = pytest.mark.integration

# Rows and channels of each modality's complete sample, in modality order:
# shapes whose shares and offsets fall inside pages.
SAMPLES = {"video": (1000, 96), "audio": (130, 64)}

# Packed rows of one tile: every rank's share of the sequence is a whole
# number of tiles (``token_multiple`` of the FastH3 denoiser).
TILE = 128

# The trajectory's content (``ipc::LatentRestore::content``).
CONTENT = list(range(32))


def _aligned(elements: int) -> int:
    return -(-elements // SAMPLE_ALIGNMENT) * SAMPLE_ALIGNMENT


def _page_units() -> int:
    """Values of a page, as ``SamplePages.plan`` sizes them from the
    complete samples, which bound every rank's share.
    """  # noqa: D205
    capacity = sum(_aligned(rows * width) for rows, width in SAMPLES.values())
    return _aligned(-(-capacity // REQUEST_PAGES))


def _fields(rank: int, ranks: int) -> LayoutFields:
    """Rank ``rank`` of a sequence-parallel denoiser of ``ranks`` ranks: the
    packed rows, video then audio, padded to a whole number of tiles a rank,
    and each rank's contiguous run of them.
    """  # noqa: D205
    total = sum(rows for rows, _ in SAMPLES.values())
    padded = -(-total // (TILE * ranks)) * TILE * ranks
    width = padded // ranks
    low, high = rank * width, (rank + 1) * width
    samples = []
    state = []
    first = 0
    for name, (rows, columns) in SAMPLES.items():
        start = min(max(low - first, 0), rows)
        stop = min(max(high - first, 0), rows)
        samples.append((name, SampleRows(rows, columns, start, stop)))
        state.append(
            StateField(name, torch.float32, (stop - start) * columns, True)
        )
        first += rows
    return LayoutFields(
        state=tuple(state),
        samples=tuple(samples),
        noise=(),
        condition_noise=0,
        text_condition=(0, 0),
        workspace=0,
    )


def _value(modality: int, row: int, channel: int) -> float:
    """The value every configuration holds at ``row``, ``channel`` of
    modality ``modality``'s sample: distinct, and exact in float32.
    """  # noqa: D205
    return float((modality + 1) * 1_000_000 + row * 100 + channel)


def _pages(fields: LayoutFields, page_units: int) -> tuple[torch.Tensor, dict]:
    """A rank's slot pages holding its share where its layout says, NaN
    elsewhere, and the layout it declares.
    """  # noqa: D205
    layout = SampleLayout.of(fields, page_units)
    storage = layout.storage(fields)
    pages = torch.full((REQUEST_PAGES * page_units,), float("nan"))
    for modality, ((_, rows), offset) in enumerate(
        zip(fields.samples, layout.offsets, strict=True)
    ):
        count = rows.stop - rows.start
        row = torch.arange(rows.start, rows.stop, dtype=torch.float64)
        channel = torch.arange(rows.width, dtype=torch.float64)
        values = (modality + 1) * 1_000_000 + row[:, None] * 100 + channel
        pages[offset : offset + count * rows.width] = values.flatten().float()
    return pages, storage


def test_a_rank_declares_its_rows_of_each_sample_in_its_pages():
    page_units = _page_units()
    fields = _fields(1, 2)
    layout = SampleLayout.of(fields, page_units)
    storage = layout.storage(fields)
    # Rank 1 of two holds the video sample's last rows and every audio row.
    video, audio = (rows for _, rows in fields.samples)
    assert (video.start, video.stop) == (640, 1000)
    assert (audio.start, audio.stop) == (0, 130)
    declared = {field["tensor"]: field for field in storage["fields"]}
    assert set(declared) == {"video_latents", "audio_latents"}
    for (name, rows), offset in zip(fields.samples, layout.offsets):
        field = declared[f"{name}_latents"]
        count = rows.stop - rows.start
        assert field["logical"]["shape"] == [rows.rows, rows.width]
        assert field["logical"]["segments"] == [
            [
                {
                    "span": {"start": rows.start, "count": count, "step": 1},
                    "local": 0,
                }
            ],
            [
                {
                    "span": {"start": 0, "count": rows.width, "step": 1},
                    "local": 0,
                }
            ],
        ]
        assert field["shape"] == [count, rows.width]
        assert field["strides"] == [rows.width, 1]
        assert field["offset"] == offset * 4
        assert field["dtype"] == "float32" and field["element"] == 4
    # The storage ends with the last value the rank holds.
    audio_offset = layout.offsets[1]
    assert storage["bytes"] == (audio_offset + 130 * 64) * 4
    # A rank that holds no row of a modality declares no field of it.
    first = _fields(0, 2)
    names = [
        field["tensor"]
        for field in SampleLayout.of(first, page_units).storage(first)["fields"]
    ]
    assert names == ["video_latents"]


@pytest.mark.parametrize(
    ("stored", "restoring"),
    [(2, 1), (1, 2), (2, 2)],
    ids=["two-to-one", "one-to-two", "same"],
)
def test_stored_pages_are_laid_out_in_another_configurations_pages(
    stored: int, restoring: int
):
    page_units = _page_units()
    page_bytes = page_units * 4
    sources = [
        _pages(_fields(rank, stored), page_units) for rank in range(stored)
    ]
    for rank in range(restoring):
        fields = _fields(rank, restoring)
        expected, target_layout = _pages(fields, page_units)
        request = {
            "sources": [
                {"layout": layout, "page_bytes": page_bytes}
                for _, layout in sources
            ],
            "target": {"layout": target_layout, "page_bytes": page_bytes},
            "content": CONTENT,
        }
        copies = json.loads(
            str(worker.library()["latent_restore_copies"](json.dumps(request)))
        )
        # Every stored page as a flat byte tensor, as the worker binds the
        # pool's pages.
        stored_pages = [
            page.view(torch.uint8)
            for pages, _ in sources
            for page in pages.view(REQUEST_PAGES, page_units)
        ]
        target = torch.full((REQUEST_PAGES * page_units,), float("nan"))
        copy_runs(
            target.view(torch.uint8),
            stored_pages,
            [
                [source * REQUEST_PAGES + page, offset, at, length]
                for source, page, offset, at, length in copies
            ],
        )
        # The rank's pages hold exactly its declared values; nothing else
        # of them is read.
        layout = SampleLayout.of(fields, page_units)
        for (_, rows), offset in zip(fields.samples, layout.offsets):
            count = (rows.stop - rows.start) * rows.width
            held = target[offset : offset + count]
            assert torch.equal(
                held.view(torch.int32),
                expected[offset : offset + count].view(torch.int32),
            )
        if stored == restoring:
            # The stored layout: every page is copied whole from its own.
            assert all(
                (offset, length) == (0, page_bytes)
                for _, _, offset, _, length in copies
            )
