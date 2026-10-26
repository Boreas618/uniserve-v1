"""The buckets an image model's forwards are captured at.

A captured forward holds static shapes, and its rows' sizes enter only as
device metadata, so a graph's key is a bucket of its forward's extents:

- a denoising step and a latent condition write pack *rows* (framed image
  sequences, ``uniserve.executor.image.rows``): their key is ``(T, R)``,
  tokens and rows (:class:`RowShape`);
- a vision encode packs images' patch rows into *segments*: its key is
  ``(P, S)``, patch rows and image slots (:class:`VisionShape`).

A row bucket serves forwards of fewer rows and fewer tokens than it holds:
its last row is a padding row holding the remaining tokens, at least one
(``RowColumns.padded``). The padding row is a row like any other, so it
holds at most the longest framed sequence; consecutive token buckets are
therefore at most that far apart (:func:`token_sizes`). A vision bucket
serves forwards of at most its slots and patch rows; one padding segment
covers its remaining patch rows.

:class:`RowShapes` and :class:`VisionShapes` hold a family's buckets: every
configured pair some admissible forward selects, the others pruned
(:func:`row_shapes`, :func:`vision_shapes`); ``select`` returns the
smallest bucket holding a forward, and ``capture_order`` the order startup
captures them in, largest first, a function of configuration and model
facts alone, so every rank of a component captures the same keys in the
same order; :func:`digest` names that order for comparing ranks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "RowLimits",
    "RowShape",
    "RowShapes",
    "VisionLimits",
    "VisionShape",
    "VisionShapes",
    "condition_rows",
    "denoise_rows",
    "digest",
    "patch_sizes",
    "row_shapes",
    "slot_sizes",
    "token_sizes",
    "vision_shapes",
]


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def token_sizes(
    cap: int,
    *,
    multiple: int = 1,
    row_tokens: int,
    step: int = 256,
    linear: int = 4096,
    growth: float = 1.25,
    coarse: int = 256,
) -> list[int]:
    """The default token buckets of a row family.

    Buckets step by ``step`` up to ``linear``, then grow by ``growth``,
    rounded up to a multiple of ``coarse * multiple``, while they hold at
    most ``cap`` tokens, the most a forward carries; the last bucket is the
    least multiple above ``cap``, so every forward has a padding token to
    spare. Every bucket is a multiple of ``multiple`` (the
    sequence-parallel degree, which divides a bucket's tokens evenly among
    the ranks). A bucket's padding row holds at most ``row_tokens`` tokens,
    so the first bucket exceeds one token by at most that many and
    consecutive buckets differ by at most that many; buckets are added
    between the stepped ones to keep it so.

    Raises:
        ValueError: ``row_tokens`` is below ``multiple``, so no bucket list
            keeps the padding within one row.
    """
    if row_tokens < multiple:
        raise ValueError(
            f"a row of {row_tokens} tokens cannot pad buckets of "
            f"multiples of {multiple} tokens"
        )
    stride = _round_up(step, multiple)
    sizes = list(range(stride, min(linear, cap) + 1, stride))
    size = sizes[-1] if sizes else 0
    while size:
        size = _round_up(int(size * growth), coarse * multiple)
        if size > cap:
            break
        sizes.append(size)
    sizes.append(_round_up(cap + 1, multiple))
    return _bounded_gaps(sizes, row_tokens, multiple)


def _bounded_gaps(
    sizes: Iterable[int], row_tokens: int, multiple: int
) -> list[int]:
    """Insert buckets so each exceeds its predecessor by at most a row.

    The first bucket serves forwards of one token, so it may exceed zero
    by one more token than a row holds.
    """
    widest = row_tokens // multiple * multiple
    result: list[int] = []
    previous = 0
    for size in sorted(set(sizes)):
        # Forwards selecting ``size`` hold at least ``previous`` tokens, and
        # at least one below the first bucket.
        while size - max(previous, 1) > row_tokens:
            previous += widest
            result.append(previous)
        result.append(size)
        previous = size
    return result


@dataclass(frozen=True, slots=True, order=True)
class RowShape:
    """A row bucket: ``tokens`` packed tokens in ``rows`` rows."""

    tokens: int
    rows: int

    @property
    def key(self) -> tuple[int, int]:
        """The graph key ``(T, R)``."""
        return self.tokens, self.rows


@dataclass(frozen=True, slots=True)
class RowLimits:
    """The forwards a row family admits.

    Attributes:
        min_row_tokens: Tokens of the shortest live row: one latent row and
            its framing.
        max_row_tokens: Tokens of the longest live row, which also bounds
            the padding row.
        max_rows: The most live rows of one forward.
        max_tokens: The most live tokens of one forward.
    """

    min_row_tokens: int
    max_row_tokens: int
    max_rows: int
    max_tokens: int

    def __post_init__(self) -> None:
        if not (
            1 <= self.min_row_tokens <= self.max_row_tokens
            and self.max_rows >= 1
            and self.max_tokens >= self.min_row_tokens
        ):
            raise ValueError(
                "row limits require a nonempty range of row lengths, rows "
                "and tokens"
            )


class RowShapes:
    """The captured row buckets of one family (denoise or condition).

    Args:
        shapes: The buckets (:func:`row_shapes`).
    """

    def __init__(self, shapes: Iterable[RowShape]) -> None:
        self.shapes = tuple(
            sorted(set(shapes), key=lambda s: (s.rows, s.tokens))
        )

    def select(self, rows: int, tokens: int) -> RowShape | None:
        """The least ``(R, T)`` bucket with ``R > rows`` and ``T > tokens``.

        The strict bounds leave the bucket a padding row with at least one
        token. ``None`` when no bucket holds the forward.
        """
        held = [
            shape
            for shape in self.shapes
            if shape.rows > rows and shape.tokens > tokens
        ]
        return min(held, key=lambda s: (s.rows, s.tokens), default=None)

    def capture_order(self) -> list[RowShape]:
        """Buckets in capture order: by ``T * R``, then ``T``, descending."""
        return sorted(
            self.shapes,
            key=lambda s: (s.tokens * s.rows, s.tokens),
            reverse=True,
        )

    @property
    def max_rows(self) -> int:
        """The most live rows a bucket serves."""
        return max((shape.rows for shape in self.shapes), default=1) - 1

    @property
    def max_tokens(self) -> int:
        """The most live tokens a bucket serves."""
        return max((shape.tokens for shape in self.shapes), default=1) - 1


def row_shapes(
    tokens: Sequence[int], rows: Sequence[int], limits: RowLimits
) -> RowShapes:
    """The configured row buckets some admissible forward selects.

    A forward of ``r`` live rows and ``t`` live tokens is admissible when
    ``r`` lies in ``[1, max_rows]`` and ``t`` in ``[r * min_row_tokens, r *
    max_row_tokens]``, at most ``max_tokens``. It selects the least row
    bucket above ``r``, then within it the least token bucket above ``t``
    (:meth:`RowShapes.select`); a bucket no admissible forward selects is
    not captured, nor is one whose padding row could exceed a row's longest
    sequence.

    Raises:
        ValueError: Some admissible forward selects no bucket.
    """
    token_buckets = sorted(set(tokens))
    row_buckets = sorted({size for size in rows if size > 1})
    captured = []
    for index, row_bucket in enumerate(row_buckets):
        lowest = row_buckets[index - 1] if index else 1
        highest = min(row_bucket - 1, limits.max_rows)
        for position, token_bucket in enumerate(token_buckets):
            previous = token_buckets[position - 1] if position else 0
            # Forwards of r rows selecting this token bucket hold t tokens
            # in [previous, token_bucket - 1], and pad by at most a row.
            reachable = any(
                max(
                    previous,
                    count * limits.min_row_tokens,
                    token_bucket - limits.max_row_tokens,
                )
                <= min(
                    token_bucket - 1,
                    count * limits.max_row_tokens,
                    limits.max_tokens,
                )
                for count in range(lowest, highest + 1)
            )
            if reachable:
                captured.append(RowShape(token_bucket, row_bucket))
    shapes = RowShapes(captured)
    if shapes.max_rows < limits.max_rows or shapes.max_tokens < min(
        limits.max_tokens, limits.max_rows * limits.max_row_tokens
    ):
        raise ValueError(
            f"row buckets serve {shapes.max_rows} rows of {shapes.max_tokens} "
            f"tokens; the family admits {limits.max_rows} rows of "
            f"{limits.max_tokens} tokens"
        )
    return shapes


def denoise_rows(*, guidance_branches: int, max_images: int) -> list[int]:
    """The default row buckets of denoising steps.

    One image's branches, and ``max_images`` images' branches, each with
    the padding row.
    """
    return sorted({guidance_branches + 1, max_images * guidance_branches + 1})


def condition_rows(*, max_calls: int) -> list[int]:
    """The default row buckets of condition writes.

    One latent image, and one per call of a text batch, each with the
    padding row.
    """
    return sorted({2, max_calls + 1})


@dataclass(frozen=True, slots=True, order=True)
class VisionShape:
    """A vision bucket: ``patches`` patch rows in ``slots`` image slots."""

    patches: int
    slots: int

    @property
    def key(self) -> tuple[int, int]:
        """The graph key ``(P, S)``."""
        return self.patches, self.slots


@dataclass(frozen=True, slots=True)
class VisionLimits:
    """The vision encodes a patch tower admits.

    Attributes:
        downsample: Patch rows per output feature on each axis; an image's
            grid sides are multiples of it.
        max_patches: Patch rows of the largest image.
        max_slots: The most images of one encode.
    """

    downsample: int
    max_patches: int
    max_slots: int

    def __post_init__(self) -> None:
        if (
            self.downsample < 1
            or self.max_slots < 1
            or self.max_patches < self.downsample**2
        ):
            raise ValueError(
                "vision limits require an image of at least one feature and "
                "at least one slot"
            )

    @property
    def context(self) -> tuple[int, int]:
        """``(tokens, rows)`` the encoder's context is prepared for.

        The largest bucket: every slot of the largest image's patch rows,
        and the slots' segments with the padding segment.
        """
        return self.max_slots * self.max_patches, self.max_slots + 1


class VisionShapes:
    """The captured vision buckets of one patch tower.

    Args:
        shapes: The buckets (:func:`vision_shapes`).
    """

    def __init__(self, shapes: Iterable[VisionShape]) -> None:
        self.shapes = tuple(
            sorted(set(shapes), key=lambda s: (s.slots, s.patches))
        )

    def select(self, images: int, patches: int) -> VisionShape | None:
        """The least ``(S, P)`` bucket with ``S >= images``, ``P >= patches``.

        ``None`` when no bucket holds the encode.
        """
        held = [
            shape
            for shape in self.shapes
            if shape.slots >= images and shape.patches >= patches
        ]
        return min(held, key=lambda s: (s.slots, s.patches), default=None)

    def capture_order(self) -> list[VisionShape]:
        """Buckets in capture order: by ``P * S``, then ``P``, descending."""
        return sorted(
            self.shapes,
            key=lambda s: (s.patches * s.slots, s.patches),
            reverse=True,
        )


def patch_sizes(limits: VisionLimits, *, first: int = 1024) -> list[int]:
    """The default patch-row buckets of a patch tower.

    Doubling from ``first`` up to every slot of the largest image, each a
    multiple of whole features (``downsample**2`` patch rows).
    """
    unit = limits.downsample**2
    cap = limits.max_slots * limits.max_patches
    sizes = []
    size = _round_up(first, unit)
    while size < cap:
        sizes.append(size)
        size = _round_up(2 * size, unit)
    sizes.append(_round_up(cap, unit))
    return sizes


def slot_sizes(limits: VisionLimits) -> list[int]:
    """The default slot buckets: one, two and four images, and the most."""
    return sorted(
        {size for size in (1, 2, 4) if size < limits.max_slots}
        | {limits.max_slots}
    )


def vision_shapes(
    patches: Sequence[int], slots: Sequence[int], limits: VisionLimits
) -> VisionShapes:
    """The configured vision buckets some admissible encode selects.

    An encode of ``n`` images and ``p`` patch rows is admissible when ``n``
    lies in ``[1, max_slots]`` and ``p`` in ``[n * downsample**2, n *
    max_patches]``. It selects the least slot bucket holding ``n``, then
    within it the least patch bucket holding ``p``; a bucket no admissible
    encode selects is not captured.

    Raises:
        ValueError: A patch bucket is not a whole number of features, or
            some admissible encode selects no bucket.
    """
    unit = limits.downsample**2
    if any(size % unit for size in patches):
        raise ValueError(
            f"vision patch buckets must be multiples of {unit} patch rows"
        )
    patch_buckets = sorted(set(patches))
    slot_buckets = sorted({size for size in slots if size >= 1})
    captured = []
    for index, slot_bucket in enumerate(slot_buckets):
        lowest = slot_buckets[index - 1] + 1 if index else 1
        highest = min(slot_bucket, limits.max_slots)
        for position, patch_bucket in enumerate(patch_buckets):
            previous = patch_buckets[position - 1] if position else 0
            reachable = any(
                max(previous + 1, count * unit)
                <= min(patch_bucket, count * limits.max_patches)
                for count in range(lowest, highest + 1)
            )
            if reachable:
                captured.append(VisionShape(patch_bucket, slot_bucket))
    shapes = VisionShapes(captured)
    largest = shapes.select(limits.max_slots, limits.context[0])
    if largest is None:
        raise ValueError(
            f"vision buckets do not hold {limits.max_slots} images of "
            f"{limits.max_patches} patch rows"
        )
    return shapes


def digest(keys: Iterable[Sequence[int]]) -> str:
    """The SHA-256 of an ordered key list, which ranks compare.

    Ranks of one component capture the same keys in the same order exactly
    when their digests agree.
    """
    ordered = [list(map(int, key)) for key in keys]
    return hashlib.sha256(json.dumps(ordered).encode()).hexdigest()
