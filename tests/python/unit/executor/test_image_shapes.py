"""Row and vision buckets hold every admissible forward, smallest first."""

import itertools

import pytest

from uniserve.executor.image.denoiser import split_rows
from uniserve.executor.image.shapes import (
    RowLimits,
    RowShape,
    RowShapes,
    VisionLimits,
    VisionShape,
    condition_rows,
    denoise_rows,
    digest,
    patch_sizes,
    row_shapes,
    slot_sizes,
    token_sizes,
    vision_shapes,
)

pytestmark = pytest.mark.unit

# Framed images of one to twelve latent rows, three guidance branches and up
# to two images a step.
SHORTEST, LONGEST, BRANCHES, IMAGES = 3, 14, 3, 2
LIMITS = RowLimits(
    min_row_tokens=SHORTEST,
    max_row_tokens=LONGEST,
    max_rows=IMAGES * BRANCHES,
    max_tokens=IMAGES * BRANCHES * LONGEST,
)


def _denoise_shapes(multiple: int = 1) -> RowShapes:
    return row_shapes(
        token_sizes(LIMITS.max_tokens, multiple=multiple, row_tokens=LONGEST),
        denoise_rows(guidance_branches=BRANCHES, max_images=IMAGES),
        LIMITS,
    )


def _batches():
    """Every step of up to two images: each image's branches and length."""
    images = [
        (branches, length)
        for branches in range(1, BRANCHES + 1)
        for length in range(SHORTEST, LONGEST + 1)
    ]
    for count in range(1, IMAGES + 1):
        for batch in itertools.product(images, repeat=count):
            yield (
                sum(branches for branches, _ in batch),
                sum(branches * length for branches, length in batch),
            )


def _admissible():
    """Every forward ``RowLimits`` admits: rows and tokens."""
    for rows in range(1, LIMITS.max_rows + 1):
        for tokens in range(
            rows * LIMITS.min_row_tokens,
            min(rows * LIMITS.max_row_tokens, LIMITS.max_tokens) + 1,
        ):
            yield rows, tokens


def test_token_buckets_step_then_grow_past_the_largest_forward():
    sizes = token_sizes(3 * 4098, row_tokens=4098)
    assert sizes[:16] == list(range(256, 4097, 256))
    assert sizes[16:] == [5120, 6400, 8192, 10240, 12295]

    # Under sequence parallelism of two, every bucket divides evenly and
    # the grown ones are multiples of 512.
    parallel = token_sizes(3 * 4098, multiple=2, row_tokens=4098)
    assert all(size % 2 == 0 for size in parallel)
    assert parallel[16:-1] == [5120, 6656, 8704, 11264]
    assert parallel[-1] == 12296


def test_token_buckets_pad_by_at_most_one_row():
    for multiple in (1, 2, 4):
        sizes = token_sizes(84, multiple=multiple, row_tokens=LONGEST)
        assert sizes[0] <= LONGEST + 1
        assert all(b - a <= LONGEST for a, b in itertools.pairwise(sizes))
        assert sizes[-1] > 84
        assert all(size % multiple == 0 for size in sizes)
    with pytest.raises(ValueError, match="cannot pad"):
        token_sizes(84, multiple=16, row_tokens=8)


def test_every_admissible_step_selects_a_bucket_with_a_row_to_spare():
    shapes = _denoise_shapes()
    for rows, tokens in _batches():
        shape = shapes.select(rows, tokens)
        assert shape is not None, (rows, tokens)
        assert shape.rows > rows and shape.tokens > tokens
        # The padding row holds the remaining tokens, at most one row's.
        assert shape.tokens - tokens <= LONGEST


def test_buckets_select_strictly_above_their_live_rows_and_tokens():
    shapes = RowShapes(
        RowShape(tokens, rows) for tokens in (16, 32, 48) for rows in (4, 7)
    )
    # A bucket serves one row and one token fewer than it holds.
    assert shapes.select(3, 15) == RowShape(16, 4)
    # A forward filling a bucket's tokens takes the next token bucket.
    assert shapes.select(3, 16) == RowShape(32, 4)
    # A forward filling a bucket's rows takes the next row bucket.
    assert shapes.select(4, 15) == RowShape(16, 7)
    assert shapes.select(7, 15) is None
    assert shapes.select(3, 48) is None


def test_captures_exclude_buckets_no_admitted_forward_can_reach():
    tokens = token_sizes(LIMITS.max_tokens, row_tokens=LONGEST)
    rows = denoise_rows(guidance_branches=BRANCHES, max_images=IMAGES)
    every = RowShapes(RowShape(t, r) for t in tokens for r in rows)
    reached = {every.select(*forward) for forward in _admissible()}
    captured = set(_denoise_shapes().shapes)
    assert captured == reached
    # One image's forwards never reach the largest token buckets of the
    # one-image row bucket.
    assert RowShape(tokens[-1], BRANCHES + 1) not in captured
    assert len(captured) < len(every.shapes)


def test_buckets_capture_largest_first_from_configuration_alone():
    shapes = _denoise_shapes()
    order = shapes.capture_order()
    footprints = [(shape.tokens * shape.rows, shape.tokens) for shape in order]
    assert footprints == sorted(footprints, reverse=True)
    # The order is a function of configuration and model facts: building
    # the buckets again gives the same keys in the same order.
    again = _denoise_shapes().capture_order()
    assert digest(shape.key for shape in order) == digest(
        shape.key for shape in again
    )
    assert digest(shape.key for shape in order) != digest(
        shape.key for shape in reversed(order)
    )


def test_buckets_that_cannot_serve_the_admitted_forwards_are_refused():
    with pytest.raises(ValueError, match="admits"):
        row_shapes([16, 32], [4, 7], LIMITS)


def test_default_row_buckets_hold_one_and_the_most_images_or_calls():
    assert denoise_rows(guidance_branches=3, max_images=1) == [4]
    assert denoise_rows(guidance_branches=3, max_images=4) == [4, 13]
    assert condition_rows(max_calls=1) == [2]
    assert condition_rows(max_calls=8) == [2, 9]


VISION = VisionLimits(downsample=2, max_patches=64, max_slots=6)


def _encodes():
    """Every encode of up to six images of 4 to 64 patch rows."""
    sides = [
        (2 * rows, 2 * columns) for rows in (1, 2, 4) for columns in (1, 2, 4)
    ]
    patches = sorted({rows * columns for rows, columns in sides})
    for count in range(1, VISION.max_slots + 1):
        for batch in itertools.combinations_with_replacement(patches, count):
            yield count, sum(batch)


def test_every_admissible_encode_selects_a_vision_bucket():
    shapes = vision_shapes(
        patch_sizes(VISION, first=16), slot_sizes(VISION), VISION
    )
    assert slot_sizes(VISION) == [1, 2, 4, 6]
    for images, patches in _encodes():
        shape = shapes.select(images, patches)
        assert shape is not None, (images, patches)
        assert shape.slots >= images and shape.patches >= patches
        assert shape.patches % VISION.downsample**2 == 0


def test_vision_captures_exclude_buckets_no_encode_can_reach():
    patches = patch_sizes(VISION, first=16)
    slots = slot_sizes(VISION)
    every = [VisionShape(p, s) for p in patches for s in slots]
    captured = set(vision_shapes(patches, slots, VISION).shapes)
    # One image never needs more than its slot's patch rows.
    assert VisionShape(patches[-1], 1) not in captured
    assert all(shape in every for shape in captured)
    order = vision_shapes(patches, slots, VISION).capture_order()
    footprints = [
        (shape.patches * shape.slots, shape.patches) for shape in order
    ]
    assert footprints == sorted(footprints, reverse=True)


def test_a_vision_context_covers_every_slot_of_its_largest_bucket():
    limits = VisionLimits(downsample=3, max_patches=2520, max_slots=16)
    # Every slot's patch rows, and one segment per slot plus the padding
    # segment.
    assert limits.context == (16 * 2520, 17)
    shapes = vision_shapes(patch_sizes(limits), slot_sizes(limits), limits)
    assert shapes.select(16, 16 * 2520) == VisionShape(16 * 2520, 16)
    with pytest.raises(ValueError, match="multiples"):
        vision_shapes([1000], [1], limits)


@pytest.mark.parametrize("multiple", [1, 2])
def test_each_buckets_warm_forward_selects_that_bucket(multiple):
    # Warmup captures every bucket by a forward of rows the family admits;
    # the forward selects exactly the bucket it captures.
    shapes = _denoise_shapes(multiple)
    for shape in shapes.capture_order():
        lengths = split_rows(shape, LIMITS, shapes)
        assert lengths is not None, shape
        assert all(
            LIMITS.min_row_tokens <= length <= LIMITS.max_row_tokens
            for length in lengths
        )
        assert shapes.select(len(lengths), sum(lengths)) == shape
    # Without buckets one row of the shortest image warms the family.
    assert split_rows(RowShape(SHORTEST + 1, 2), LIMITS, None) == [SHORTEST]
