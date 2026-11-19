"""The `record_write` program against the record formats, record for record.

The operator writes one record per token into a paged cache plane
(`ops::record::RecordWrite`)::

    y      = norm ? x · rsqrt(mean(x²) + eps) · gain : x    in the row type
    y'     = y, its trailing `rotated` channels turned by position e · ratio
    record = encode(format, y')                              (`ops::record`)

at slot `e mod entries` of page `table[row, (e / entries) mod max_pages]`, and
nothing for a negative entry. The encoding is exact: one block has one code
per value and one scale (`uniserve_testing.kernels.records`, the operator
crate's rules in torch), so a record of an unnormalized, unrotated row is held
to the reference byte for byte. A normalized or rotated row carries the FP32
evaluation of its statistics and its cosines, which two implementations do not
share bit for bit; such a row is held to an FP64 reference at the rounding
bound of its type, and its packed record to the encoding of the very row the
same program stores in a BF16 plane, so the encoding stays exact.

What a wrong kernel gets wrong, and the case that catches each:

- **the address**: two table rows whose entries are out of token order, one
  entry past the table's width (read modulo it, a ring of pages) and one
  negative entry; the plane is pre-filled with a sentinel, and every record no
  token names must keep it, two pages of it whole;
- **the layout**: each packed format in both layouts, a record's scales after
  its own values or after the page's values;
- **the scale**: blocks of zeros (the peak's floor), of tiny values (E4M3's
  subnormal steps) and of values past NVFP4's largest E4M3 scale (saturated),
  beside blocks spread over five decades;
- **the rounding points**: a normalization that rounds into the row type and
  a rotation that rounds each turned value.

Triton's interpreter truncates a cast into BF16, which the normalized cases
model there, and does not round a conversion into E4M3 to nearest even, so the
MXFP8 and NVFP4 records are checked on a device. The rotation reads
libdevice's cosines and sines, which only a device runs, so its cases are
marked `gpu`.
"""

import math

import pytest
import torch
from uniserve_testing.kernels import records
from uniserve_testing.kernels.rope.rounding import UNIT_ROUNDOFF, gamma

pytestmark = pytest.mark.unit

#: The row types the operator writes from.
DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32}

#: Records per page, the table's width and the plane's pages.
ENTRIES = 4
MAX_PAGES = 2
PAGES = 6

#: Each token's table row and entry. Row 0's entries are out of token order
#: and one is negative; row 1's entry 9 lies past the table's two columns of
#: four records and lands in column `(9 / 4) mod 2 = 0`, beside the row's own
#: entries 3 and 10 of that page.
TABLE_ROWS = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
TOKEN_ENTRIES = [5, 0, 9, -1, 3, 7, 4, 2, 10, 1]

#: Each table row's pages; pages 0 and 5 are named by no row.
TABLE = [[3, 1], [2, 4]]

#: The byte a fresh plane holds, which no written record keeps everywhere.
SENTINEL = 0x5A

#: The epsilon of the normalized cases: DeepSeek-V4.1's.
EPS = 1e-20

#: The formats that round values into E4M3: MXFP8's values and NVFP4's
#: scales. Triton's interpreter converts into E4M3 rounding a tie away from
#: zero and a subnormal toward it, where a device rounds to nearest even, so
#: their records are checked on a device.
E4M3_FORMATS = ("mxfp8", "nvfp4")


def key(dtype, width, norm, rotated, ratio, fmt, layout):
    """`RecordWriteDescriptor`'s key."""
    return (
        f"dtype_{dtype}_width_{width}_norm_{str(norm).lower()}_rotated_"
        f"{rotated}_ratio_{ratio}_format_{fmt}_layout_{layout}_entries_"
        f"{ENTRIES}"
    )


def drawn_rows(width, dtype, seed):
    """`[tokens, width]` rows in `dtype` whose blocks of 16 span the cases of
    the module documentation: five decades, zeros, tiny values and values
    past NVFP4's largest scale.
    """  # noqa: D205
    generator = torch.Generator().manual_seed(seed)
    tokens = len(TOKEN_ENTRIES)
    blocks = width // 16
    decades = 10.0 ** torch.randint(
        -3, 2, (tokens, blocks, 1), generator=generator
    )
    rows = torch.randn(tokens, blocks, 16, generator=generator) * decades
    rows[0, 0] = 0.0
    rows[1, 1] = torch.randn(16, generator=generator) * 1e-6
    rows[2, 2] = torch.randn(16, generator=generator) * 3000.0
    return rows.reshape(tokens, width).to(DTYPES[dtype])


def fresh_plane(fmt, width, device):
    """A plane of `PAGES` pages filled with the sentinel."""
    if fmt == "bf16":
        filled = torch.full(
            (PAGES, ENTRIES, 2 * width), SENTINEL, dtype=torch.uint8
        )
        return filled.view(torch.bfloat16).to(device)
    record = records.record_bytes(fmt, width)
    return torch.full(
        (PAGES, ENTRIES, record), SENTINEL, dtype=torch.uint8, device=device
    )


def launch(stated, rows, plane, *, gain=None, inv_freq=None):
    """One launch over every token, as the package states it."""
    device = rows.device
    unread = torch.zeros(1, dtype=rows.dtype, device=device)
    stated.kernel[(rows.shape[0],)](
        rows,
        unread if gain is None else gain,
        torch.zeros(1, device=device) if inv_freq is None else inv_freq,
        torch.tensor(TOKEN_ENTRIES, dtype=torch.int32, device=device),
        torch.tensor(TABLE_ROWS, dtype=torch.int32, device=device),
        torch.tensor(TABLE, dtype=torch.int32, device=device),
        plane,
        rows.shape[0],
        rows.stride(0),
        MAX_PAGES,
        EPS,
        **stated.constexprs,
        **stated.options,
    )


def written_plane(fmt, layout, width, payload):
    """The plane the reference writes: the sentinel plane with each token's
    record of `payload` (FP32 rows, encoded) at its entry's slot.
    """  # noqa: D205
    values, scales = records.encode(fmt, payload)
    plane = fresh_plane(fmt, width, "cpu")
    stored_values, stored_scales = records.unplane(fmt, layout, plane, width)
    for token, entry in enumerate(TOKEN_ENTRIES):
        if entry < 0:
            continue
        page = TABLE[TABLE_ROWS[token]][(entry // ENTRIES) % MAX_PAGES]
        record = page * ENTRIES + entry % ENTRIES
        stored_values[record] = values[token]
        if stored_scales is not None:
            stored_scales[record] = scales[token]
    if fmt == "bf16":
        return stored_values.view(torch.bfloat16).reshape(plane.shape)
    return records.plane(fmt, layout, stored_values, stored_scales, ENTRIES)


def assert_same_bytes(where, got, expected):
    """Assert two planes are bytewise equal, naming the first record that is
    not.
    """  # noqa: D205
    got = got.cpu().contiguous().view(torch.uint8).reshape(PAGES * ENTRIES, -1)
    expected = (
        expected.contiguous().view(torch.uint8).reshape(PAGES * ENTRIES, -1)
    )
    differing = (got != expected).any(dim=1)
    if differing.any():
        record = int(differing.nonzero()[0])
        column = int((got[record] != expected[record]).nonzero()[0])
        raise AssertionError(
            f"{where}: {int(differing.sum())} of {PAGES * ENTRIES} records "
            f"differ; first record {record} (page {record // ENTRIES}, slot "
            f"{record % ENTRIES}) at byte {column}: "
            f"{int(got[record, column])} vs {int(expected[record, column])}"
        )


def stored_rows(plane, width):
    """Each token's row as a BF16 plane stores it, `[tokens, width]` FP32."""
    flat = plane.cpu().reshape(PAGES * ENTRIES, width).to(torch.float32)
    picked = [
        TABLE[TABLE_ROWS[token]][(entry // ENTRIES) % MAX_PAGES] * ENTRIES
        + entry % ENTRIES
        for token, entry in enumerate(TOKEN_ENTRIES)
    ]
    return flat[torch.tensor(picked)]


def normalized(rows, gain):
    """The rows' RMS normalization by `gain`, in FP64."""
    wide = rows.to(torch.float64)
    scale = torch.rsqrt(wide.square().mean(dim=-1, keepdim=True) + EPS)
    return wide * scale * gain.to(torch.float64)


def rotated(rows, inv_freq, ratio):
    """The rows with their trailing channels turned in adjacent pairs by the
    angles of each token's entry times `ratio`, in FP64 over the FP32 angles,
    and the terms `|even · cos| + |odd · sin|` of each turned value.
    """  # noqa: D205
    pairs = inv_freq.numel()
    positions = torch.tensor([max(entry, 0) * ratio for entry in TOKEN_ENTRIES])
    angle = (positions.to(torch.float32)[:, None] * inv_freq[None, :]).to(
        torch.float64
    )
    cos, sin = torch.cos(angle), torch.sin(angle)
    wide = rows.to(torch.float64).clone()
    tail = wide[:, -2 * pairs :].reshape(-1, pairs, 2)
    even, odd = tail[..., 0], tail[..., 1]
    turned = torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin), dim=-1
    )
    terms = torch.stack(((even * cos).abs() + (odd * sin).abs(),) * 2, dim=-1)
    wide[:, -2 * pairs :] = turned.reshape(-1, 2 * pairs)
    spread = torch.zeros_like(wide)
    spread[:, -2 * pairs :] = terms.reshape(-1, 2 * pairs)
    return wide, spread


def frequencies(rotated_channels, theta=10000.0):
    """The inverse frequencies of a rotation of `rotated_channels`."""
    pairs = rotated_channels // 2
    exponents = (
        torch.arange(pairs, dtype=torch.float32) * 2.0 / rotated_channels
    )
    return (1.0 / theta**exponents).to(torch.float32)


def assert_within(where, got, expected, allowed):
    """Assert `|got − expected| ≤ allowed` elementwise, naming the first
    element that is not.
    """  # noqa: D205
    difference = (got.to(torch.float64) - expected).abs()
    outside = ~(difference <= allowed)
    if outside.any():
        token, channel = (int(index) for index in outside.nonzero()[0])
        raise AssertionError(
            f"{where}: {int(outside.sum())} of {got.numel()} values outside "
            f"the bound; first token {token} channel {channel}: "
            f"{float(got[token, channel])} vs "
            f"{float(expected[token, channel])}, allowed "
            f"{float(allowed[token, channel]):.3e}"
        )


#: The encodings, each format in each layout it has, at the latent's width
#: and the index key's.
ENCODINGS = [
    pytest.param(
        fmt,
        layout,
        width,
        id=f"{fmt}-{layout}-{width}",
        marks=[pytest.mark.gpu] if fmt in E4M3_FORMATS else [],
    )
    for fmt, layouts in (
        ("bf16", ["entry"]),
        ("mxfp8", ["entry", "page"]),
        ("nvfp4", ["entry", "page"]),
        ("mxfp4", ["entry", "page"]),
    )
    for layout in layouts
    for width in (512, 128)
]


@pytest.mark.parametrize("fmt, layout, width", ENCODINGS)
def test_every_record_is_its_rows_encoding_at_its_entry(
    fmt, layout, width, torch_device, stated_launch, launch_target
):
    rows = drawn_rows(width, "bf16", seed=width + len(fmt))
    plane = fresh_plane(fmt, width, torch_device)

    stated = stated_launch(
        "record_write",
        key("bf16", width, False, 0, 1, fmt, layout),
        launch_target,
    )
    launch(stated, rows.to(torch_device), plane)

    expected = written_plane(fmt, layout, width, rows.to(torch.float32))
    assert_same_bytes(f"{fmt} {layout} records of {width}", plane, expected)


@pytest.mark.parametrize(
    "fmt",
    [
        pytest.param("mxfp4", id="mxfp4"),
        pytest.param("mxfp8", id="mxfp8", marks=pytest.mark.gpu),
        pytest.param("nvfp4", id="nvfp4", marks=pytest.mark.gpu),
        # A BF16 record of an FP32 row narrows, which the interpreter truncates.
        pytest.param("bf16", id="bf16", marks=pytest.mark.gpu),
    ],
)
def test_an_f32_row_is_encoded_from_its_own_values(
    fmt, torch_device, stated_launch, launch_target
):
    width = 512
    rows = drawn_rows(width, "f32", seed=7)
    plane = fresh_plane(fmt, width, torch_device)

    stated = stated_launch(
        "record_write",
        key("f32", width, False, 0, 1, fmt, "entry"),
        launch_target,
    )
    launch(stated, rows.to(torch_device), plane)

    expected = written_plane(fmt, "entry", width, rows)
    assert_same_bytes(f"{fmt} records of f32 rows", plane, expected)


#: The normalized and rotated cases: the width, the rotated channels, the
#: entry's ratio and whether the row is normalized. A rotation reads
#: libdevice, so its cases need a device.
TRANSFORMS = [
    pytest.param(512, 0, 1, True, id="normalized latent"),
    pytest.param(128, 0, 2, True, id="normalized index key"),
    pytest.param(512, 64, 1, False, id="rotated latent", marks=pytest.mark.gpu),
    pytest.param(
        512,
        64,
        1,
        True,
        id="normalized rotated window latent",
        marks=pytest.mark.gpu,
    ),
    pytest.param(
        512, 64, 2, False, id="rotated compressed latent", marks=pytest.mark.gpu
    ),
    pytest.param(
        128,
        64,
        2,
        True,
        id="normalized rotated index key",
        marks=pytest.mark.gpu,
    ),
]


@pytest.mark.parametrize("width, rotated_channels, ratio, norm", TRANSFORMS)
def test_a_transformed_row_is_stored_at_its_bound_and_encoded_exactly(
    width,
    rotated_channels,
    ratio,
    norm,
    torch_device,
    interpreted,
    stated_launch,
    launch_target,
):
    rows = drawn_rows(width, "bf16", seed=width + rotated_channels + ratio)
    generator = torch.Generator().manual_seed(11)
    gain = (1.0 + 0.25 * torch.randn(width, generator=generator)).to(
        torch.bfloat16
    )
    inv_freq = frequencies(rotated_channels) if rotated_channels else None

    def write(fmt, layout):
        plane = fresh_plane(fmt, width, torch_device)
        stated = stated_launch(
            "record_write",
            key("bf16", width, norm, rotated_channels, ratio, fmt, layout),
            launch_target,
        )
        launch(
            stated,
            rows.to(torch_device),
            plane,
            gain=gain.to(torch_device) if norm else None,
            inv_freq=None if inv_freq is None else inv_freq.to(torch_device),
        )
        return plane

    # The stored row against the FP64 reference: a normalized value carries
    # its FP32 statistics' error, 2γ_width, and its rounding into BF16, one
    # unit roundoff to nearest or one step truncated on the interpreter; a
    # turned value's error is relative to its two terms, and its inputs are
    # BF16 values within one rounding of their reference.
    bf16 = UNIT_ROUNDOFF[torch.bfloat16]
    rounding = (2.0 if interpreted else 1.0) * bf16
    expected = normalized(rows, gain) if norm else rows.to(torch.float64)
    relative = rounding + 2.0 * gamma(width) if norm else 0.0
    allowed = relative * expected.abs()
    if rotated_channels:
        expected, terms = rotated(expected, inv_freq, ratio)
        allowed = (
            rounding * expected.abs()
            + (2.0 * relative + 8.0 * UNIT_ROUNDOFF[torch.float32]) * terms
        )
        allowed = torch.where(terms > 0, allowed, relative * expected.abs())
    allowed = allowed + math.ldexp(1.0, -133)
    stored = stored_rows(write("bf16", "entry"), width)
    live = torch.tensor(TOKEN_ENTRIES) >= 0
    assert_within(
        f"stored rows of {width}, norm {norm}, {rotated_channels} rotated",
        stored[live],
        expected[live],
        allowed[live],
    )

    # Each packed record is the encoding of the row the BF16 plane stored;
    # the interpreter checks the format that rounds no value into E4M3.
    packed = [("mxfp8", "entry"), ("nvfp4", "entry"), ("mxfp4", "page")]
    for fmt, layout in packed:
        if interpreted and fmt in E4M3_FORMATS:
            continue
        plane = write(fmt, layout)
        reference = written_plane(fmt, layout, width, stored)
        assert_same_bytes(
            f"{fmt} {layout} records of {width} rows, norm {norm}, "
            f"{rotated_channels} rotated",
            plane,
            reference,
        )
