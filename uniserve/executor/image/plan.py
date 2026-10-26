"""What each call of an image batch reads and writes, and how it is grouped.

The worker asks the executor, before it leases anything, what every call of
a batch reads and writes (``Executor.prepare``); the answers depend on call
metadata and model facts alone, so they are known at preparation:

- a vision encoding writes ``[features, output_size]`` features of the
  image's patch grid;
- a latent encoding writes the image's ``[latent rows, width]`` latent;
- a latent preparation opens the trajectory's initial latent and a
  denoising step advances it, both in the worker's latent ledger;
- an image decoding reads a trajectory's committed latent (or a resident
  image) and stages the image's RGB bytes, writing its pixels as a product
  when the call publishes one.

:class:`PlanEntry` is one call's answer in the preparation's JSON form: the
generic ``reads``, ``writes`` and ``publication`` every prepared
computation states (empty for image calls, whose products are fields of
their own) and the ``image`` entry the worker binds them by
(``crates/worker/src/executor/image.rs``).

Vision encodes pack every image of a forward into one patch-row bucket: the
images' patch rows back to back, then one padding segment
(:func:`segment_grids`). Images beyond a bucket's slots or patch rows split
into consecutive chunks (:func:`chunk_rows`); an encoder without packing
encodes images of one pixel shape per call (:func:`shape_groups`).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "NOTHING",
    "Field",
    "PlanEntry",
    "Write",
    "chunk_rows",
    "decoding_entry",
    "latent_entry",
    "range_interval",
    "range_name",
    "segment_grids",
    "shape_groups",
    "vision_entry",
]

# A call that reads and writes nothing: a refused or skipped call, or one of
# a computation that binds nothing for it.
NOTHING: Mapping[str, Any] = {"reads": [], "writes": [], "publication": None}


class Field(StrEnum):
    """The call field naming a product an image call writes."""

    # A vision or latent encoding's features.
    ENCODER_OUTPUT = "encoder_output"
    # An image decoding's pixels.
    IMAGE_OUTPUT = "image_output"


# The pixel interval of each product value range.
_RANGES: Mapping[str, tuple[float, float]] = {
    "unit": (0.0, 1.0),
    "signed_unit": (-1.0, 1.0),
}


def range_name(value_range: tuple[float, float]) -> str | None:
    """The product value range of a pixel interval, ``None`` for another.

    ``(0, 1)`` is ``unit`` and ``(-1, 1)`` is ``signed_unit``.
    """
    low, high = (float(value) for value in value_range)
    return next(
        (name for name, interval in _RANGES.items() if interval == (low, high)),
        None,
    )


def range_interval(name: str | None) -> tuple[float, float]:
    """The pixel interval of a product value range (:func:`range_name`).

    Raises:
        ValueError: ``name`` is no product value range: an image product
            without one, whose values no pixel interval states.
    """
    interval = None if name is None else _RANGES.get(name)
    if interval is None:
        raise ValueError(
            f"an image of value range {name!r} has no pixel interval"
        )
    return interval


@dataclass(frozen=True, slots=True)
class Write:
    """One product a call writes: its field, shape and metadata.

    Attributes:
        field: The call field naming the product.
        shape: The product's shape.
        metadata: ``{"feature": {height, width}}`` for features,
            ``{"image": {height, width, value_range}}`` for an image.
    """

    field: Field
    shape: tuple[int, ...]
    metadata: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "field": self.field.value,
            "shape": [int(extent) for extent in self.shape],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PlanEntry:
    """One image call's reads, writes and ledger binding.

    Attributes:
        read: The call reads its resident image product (``image_input``),
            leased on the computation's device.
        pixels: The call reads its inline image's host-prepared pixels.
        writes: The products it writes, in tensor order.
        staging: Host bytes the call stages: a decoded image's RGB bytes,
            lent from the batch's output lease.
        latent: The call binds its trajectory in the latent ledger.
    """

    read: bool = False
    pixels: bool = False
    writes: tuple[Write, ...] = ()
    staging: int = 0
    latent: bool = False

    @property
    def tensors(self) -> int:
        """The tensors the worker passes the call's execution.

        In order: the resident read, the pixels, each write, the staged
        bytes.
        """
        return (
            int(self.read)
            + int(self.pixels)
            + len(self.writes)
            + int(self.staging > 0)
        )

    def to_json(self) -> dict[str, Any]:
        return {
            **NOTHING,
            "image": {
                "read": self.read,
                "pixels": self.pixels,
                "writes": [write.to_json() for write in self.writes],
                "bytes": int(self.staging),
                "latent": self.latent,
            },
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> PlanEntry | None:
        """The entry a preparation stated; ``None`` without image work."""
        image = value.get("image")
        if image is None:
            return None
        return cls(
            read=bool(image["read"]),
            pixels=bool(image["pixels"]),
            writes=tuple(
                Write(
                    Field(write["field"]),
                    tuple(int(extent) for extent in write["shape"]),
                    write["metadata"],
                )
                for write in image["writes"]
            ),
            staging=int(image["bytes"]),
            latent=bool(image["latent"]),
        )


def vision_entry(
    *,
    grid: tuple[int, int],
    downsample: int,
    output_size: int,
    height: int,
    width: int,
    resident: bool,
) -> PlanEntry:
    """A vision encoding of an image whose patch grid is ``grid``.

    The encoder merges each ``downsample x downsample`` square of patches
    into one feature, so the image writes ``rows * columns / downsample**2``
    features of ``output_size`` values; their metadata is the image's
    prepared canvas ``height`` and ``width``. A ``resident`` image is read
    from its product, any other's pixels were prepared on the host.

    Raises:
        ValueError: The grid is not a whole number of features.
    """
    rows, columns = grid
    if min(rows, columns) < 1 or rows % downsample or columns % downsample:
        raise ValueError(
            "a vision image's patch grid must align with its downsampling"
        )
    features = rows * columns // downsample**2
    return PlanEntry(
        read=resident,
        pixels=not resident,
        writes=(
            Write(
                Field.ENCODER_OUTPUT,
                (features, output_size),
                {"feature": {"height": height, "width": width}},
            ),
        ),
    )


def latent_entry(
    *,
    rows: int,
    width: int,
    height: int,
    canvas_width: int,
    resident: bool,
) -> PlanEntry:
    """A latent encoding writing ``rows`` latent rows of ``width`` values.

    The features' metadata is the image's prepared canvas (``height`` x
    ``canvas_width``).
    """
    return PlanEntry(
        read=resident,
        pixels=not resident,
        writes=(
            Write(
                Field.ENCODER_OUTPUT,
                (rows, width),
                {"feature": {"height": height, "width": canvas_width}},
            ),
        ),
    )


def decoding_entry(
    *,
    height: int,
    width: int,
    value_range: str | None,
    publishes: bool,
    trajectory: bool,
) -> PlanEntry:
    """An image decoding: its RGB bytes staged, its pixels a product.

    A ``trajectory`` decoding reads its trajectory's committed latent from
    the ledger and, when it ``publishes``, writes the ``[3, height, width]``
    pixels of ``value_range`` as its image product; any other decoding
    quantizes the resident image it reads.
    """
    writes: tuple[Write, ...] = ()
    if trajectory and publishes:
        writes = (
            Write(
                Field.IMAGE_OUTPUT,
                (3, height, width),
                {
                    "image": {
                        "height": height,
                        "width": width,
                        "value_range": value_range,
                    }
                },
            ),
        )
    return PlanEntry(
        read=not trajectory,
        writes=writes,
        staging=3 * height * width,
        latent=trajectory,
    )


def segment_grids(
    shapes: Sequence[tuple[int, int]],
    slots: int,
    patches: int,
    downsample: int,
) -> list[tuple[int, int]]:
    """Lay out images in a vision bucket of ``slots`` slots and ``patches``.

    Returns ``slots + 1`` ``(rows, columns)`` segment grids that cover the
    bucket's patch rows in order: each image's patch grid in the leading
    slots, ``(0, 0)`` for each empty slot, then one ``downsample`` rows
    high grid covering the remaining patch rows, ``(0, 0)`` when the images
    fill the bucket. The encoder reads the grids as device values
    (``PatchEncoder.encode_segments``), so every layout of a bucket replays
    one graph.

    Raises:
        ValueError: More images than slots, an image grid that does not
            align with the downsampling, or images exceeding the patch rows.
    """
    unit = downsample**2
    if len(shapes) > slots or patches % unit:
        raise ValueError(
            "vision images exceed their slots, or the bucket holds a "
            "partial feature"
        )
    if any(
        min(shape) < 1 or any(side % downsample for side in shape)
        for shape in shapes
    ):
        raise ValueError(
            "vision image grids must align with spatial downsampling"
        )
    used = sum(rows * columns for rows, columns in shapes)
    if used > patches:
        raise ValueError(
            f"vision images of {used} patch rows exceed their bucket's "
            f"{patches}"
        )
    padding = patches - used
    return [
        *((rows, columns) for rows, columns in shapes),
        *((0, 0) for _ in range(slots - len(shapes))),
        (downsample, padding // downsample) if padding else (0, 0),
    ]


def chunk_rows(
    group: Iterable[int],
    budget: int,
    rows: Callable[[int], int],
    *,
    members: int | None = None,
) -> list[list[int]]:
    """Split a group into consecutive chunks of at most ``budget`` rows.

    ``rows`` gives each member's rows; a member larger than the budget takes
    a chunk of its own. ``members`` also bounds each chunk's member count.
    """
    chunks: list[list[int]] = []
    used = 0
    for index in group:
        count = rows(index)
        if (
            chunks
            and used + count <= budget
            and (members is None or len(chunks[-1]) < members)
        ):
            chunks[-1].append(index)
            used += count
        else:
            chunks.append([index])
            used = count
    return chunks


def shape_groups(
    shapes: Iterable[tuple[int, Hashable]],
) -> list[list[int]]:
    """Group indexed images by exact pixel shape, in first-appearance order.

    ``shapes`` pairs each image's index with its pixel shape; members keep
    their own indices, in order.
    """
    groups: dict[Hashable, list[int]] = {}
    for index, shape in shapes:
        groups.setdefault(shape, []).append(index)
    return list(groups.values())
