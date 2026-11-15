"""The `kv_write_scaled` quantizing page write against the rule it derives.

The operator writes full-precision rows into an E4M3 cache whose pages carry
one f32 scale each::

    source       [tokens, heads · head_dim]      bf16 or f32
    slots        [tokens] i32   the flat `page · page_size + offset`, or -1
    scales       [pages, 1, 1, 1] f32            read and written: the page's
                                                 scale
    cache        [pages, page, heads, head_dim]  E4M3, read and written
    initialized  [pages, 1, 1, 1] u8             read and written: the page's
                                                 flag, one byte

and it derives the scale rather than reading it, so the rule is the reference.
What a wrong kernel gets wrong, and the case that catches each:

- **a scale that shrinks**: the proposal is smaller than the stored scale and
  the stored bytes must not move; a kernel that takes the proposal re-encodes
  the page and silently changes values a reader already sees;
- **a grown scale without a re-encode**: the proposal is larger and every
  stored row must be decoded at the old scale and re-encoded at the new one;
- **an all-zero write**: the proposal floors at `1e-12`, so the page's scale
  is `1e-12 / 448` and not zero, which nothing could decode;
- **a page the launch does not touch**: its bytes, its scale and its flag stay
  exactly as they were;
- **two programs on one page**: one program per token, and the page's first
  token's program writes it; the cases give a page tokens that are
  consecutive, interleaved with another page's, past a block of the slots'
  scan (narrowed to make that cheap), and a slot written twice, which holds the
  later row. The interpreter runs one program after another, so only a device
  shows two programs racing on a page.

The scale, the comparisons and both divisions are f32, so on the interpreter
the reference computes the same maximum and the same scale bit for bit, and
both sides convert into E4M3 through the interpreter's own conversion: the
bytes compare exactly. On a device the kernel divides with the hardware's f32
division, within two units in the last place, so the scales are held to a
relative `2^-21` and the decoded bytes to the operator's tolerance, `rtol 2^-3`
and `atol 2^-6` in the page's scaled units; the flags are exact either way.
"""

import pytest
import torch
import triton
import triton.language as tl

pytestmark = pytest.mark.unit

DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32}

#: The largest finite E4M3 magnitude and the floor under the page's maximum:
#: the page's scale is `max(|v|) floored at 1e-12, over 448`.
E4M3_MAX = 448.0
SCALE_FLOOR = 1e-12

#: The operator's tolerance, in the page's scaled units, for two
#: implementations that round into E4M3 with different hardware.
RTOL = 2.0**-3
ATOL = 2.0**-6
#: The scales' relative bound on a device: two units in the last place of an
#: f32 division.
SCALE_RTOL = 2.0**-21

#: The bytes the flag buffer holds past the last page, where a four-byte flag
#: store at a page count not divisible by four would write.
FLAG_TAIL = 4

#: The shapes of the four cases the rule turns on: a single-page cache, so the
#: page under test is the whole cache, a row split in two blocks and a skipped
#: slot.
RULE = {
    "tokens": 4,
    "pages": 1,
    "page": 4,
    "heads": 2,
    "head_dim": 8,
    "block": 16,
    "slots": [0, 1, 2, -1],
}


@triton.jit
def _cast_kernel(source_ptr, cache_ptr, count, BLOCK: tl.constexpr):  # noqa: N803
    """Store f32 values into an E4M3 tensor, one block per program."""
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < count
    tl.store(
        cache_ptr + offset,
        tl.load(source_ptr + offset, mask=mask, other=0.0),
        mask=mask,
    )


def e4m3(values, interpreted):
    """`values` converted into E4M3 the way the session's kernels convert.

    Triton's interpreter does not round into E4M3 as torch's cast does
    (`126.7` becomes `64.0` where the format holds `128.0`), and that is the
    interpreter's conversion, not the kernel's arithmetic: on the interpreter
    the reference converts through the interpreter itself, so the comparison
    measures the rule and not the format's rounding. On a device the reference
    rounds to nearest, as the hardware's conversion does.
    """
    if not interpreted:
        return values.to(torch.float32).to(torch.float8_e4m3fn)
    flat = values.reshape(-1).to(torch.float32).contiguous()
    out = torch.zeros(flat.numel(), dtype=torch.float8_e4m3fn)
    block = max(16, triton.next_power_of_2(flat.numel()))
    _cast_kernel[(triton.cdiv(flat.numel(), block),)](
        flat, out, flat.numel(), BLOCK=block
    )
    return out.reshape(values.shape)


def reference(source, slots, scales, cache, initialized, page, interpreted):
    """The rule, in the order the operator states it, in f32."""
    values = source.to(torch.float32)
    scale = scales.clone()
    flag = initialized.reshape(-1).clone()
    page_bytes = e4m3(cache.to(torch.float32), interpreted)
    heads, head_dim = page_bytes.shape[2], page_bytes.shape[3]
    for current in range(int(scales.shape[0])):
        incoming = [
            token
            for token, slot in enumerate(slots.tolist())
            if slot >= 0 and slot // page == current
        ]
        if not incoming:
            continue
        # 1. The proposal: the maximum |v| over the rows this launch brings to
        # the page.
        largest = max(float(values[token].abs().max()) for token in incoming)
        proposed = max(largest, SCALE_FLOOR) / E4M3_MAX
        # 2. A scale never shrinks, and a fresh page takes the proposal
        # whatever it held.
        stored = float(scales[current])
        updated = max(stored, proposed) if flag[current] != 0 else proposed
        # 3. A grown scale re-encodes every row already stored.
        if flag[current] != 0 and updated > stored:
            decoded = page_bytes[current].to(torch.float32) * stored
            page_bytes[current] = e4m3(
                (decoded / updated).clamp(-E4M3_MAX, E4M3_MAX), interpreted
            )
        # 4. The rows this launch brings.
        for token in incoming:
            offset = slots[token].item() % page
            page_bytes[current, offset] = e4m3(
                (values[token].reshape(heads, head_dim) / updated).clamp(
                    -E4M3_MAX, E4M3_MAX
                ),
                interpreted,
            )
        scale[current] = updated
        flag[current] = 1
    return scale, page_bytes, flag.reshape_as(initialized)


def launch(kernels, source, slots, scales, cache, flags, page, block, scan):
    """One write, with the grid the operator computes: one program per token.

    `scan` is the slots' scan block, the package's own when `None`.
    """
    tokens = source.shape[0]
    heads, head_dim = cache.shape[2], cache.shape[3]
    kernels._kv_write_scaled_kernel[(max(tokens, 1), 1, 1)](
        source,
        slots,
        scales,
        cache,
        flags,
        tokens,
        page,
        heads,
        head_dim,
        BLOCK=block,
        PDL=False,
        SCAN=kernels.SCAN if scan is None else scan,
    )


def prefilled(page, heads, head_dim, *, scale, flag=1, seed=3):
    """A cache whose every page holds the same bytes, written at `scale` from
    values that scale is derived from, and whose first page states them: its
    scale and its flag. The other pages are fresh and hold the bytes as stale
    contents.
    """  # noqa: D205

    def fill(scales, cache, initialized):
        generator = torch.Generator().manual_seed(seed)
        rows = torch.randn(
            page, heads, head_dim, dtype=torch.float32, generator=generator
        )
        cache.copy_(
            (rows / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        )
        scales[0] = scale
        initialized[0] = flag

    return fill


def write(
    kernels,
    device,
    interpreted,
    *,
    dtype,
    tokens,
    pages,
    page,
    heads,
    head_dim,
    block,
    slots,
    seeds=None,
    prefill=None,
    untouched=(),
    scan=None,
):
    """One write held to the rule: the page's scale and flag, the bytes, the
    untouched pages and the bytes past the last page. Answers the scales.

    `prefill(scales, cache, initialized)` may state a page as already written,
    which is how the grow, the shrink and the untouched cases are set up.
    """  # noqa: D205
    element = DTYPES[dtype]
    generator = torch.Generator().manual_seed(11)
    source = torch.randn(
        tokens, heads * head_dim, dtype=torch.float32, generator=generator
    ).to(element)
    if seeds is not None:
        source = seeds.to(element)
    scales = torch.zeros(pages, 1, 1, 1, dtype=torch.float32)
    cache = torch.zeros(pages, page, heads, head_dim, dtype=torch.float8_e4m3fn)
    # One byte per page, and the buffer is longer than the view so the bytes
    # after the last page are visible.
    flags = torch.zeros(pages + FLAG_TAIL, dtype=torch.uint8)
    initialized = flags[:pages].view(pages, 1, 1, 1)
    if prefill is not None:
        prefill(scales, cache, initialized)
    slot_tensor = torch.tensor(slots, dtype=torch.int32)
    before = (scales.clone(), cache.clone(), initialized.clone())

    placed = [
        tensor.to(device)
        for tensor in (source, slot_tensor, scales, cache, flags)
    ]
    launch(kernels, *placed, page=page, block=block, scan=scan)
    scales, cache, flags = (tensor.cpu() for tensor in placed[2:])
    initialized = flags[:pages].view(pages, 1, 1, 1)

    where = (
        f"{dtype} tokens {tokens} pages {pages} page {page} heads {heads} "
        f"head_dim {head_dim} block {block} scan {scan} slots {slots}"
    )
    want_scale, want_bytes, want_flag = reference(
        source, slot_tensor, *before, page, interpreted
    )

    if interpreted:
        scales_close = torch.equal(scales, want_scale)
    else:
        scales_close = torch.allclose(
            scales, want_scale, rtol=SCALE_RTOL, atol=0.0
        )
    assert scales_close, (
        f"{where}: the scales differ beyond "
        f"{'none' if interpreted else f'a relative {SCALE_RTOL:.3g}'}: "
        f"{scales.reshape(-1).tolist()} vs {want_scale.reshape(-1).tolist()}"
    )
    assert torch.equal(initialized, want_flag), (
        f"{where}: the flags differ: {initialized.reshape(-1).tolist()} vs "
        f"{want_flag.reshape(-1).tolist()}"
    )
    # The bytes past the last page: a flag stored as four bytes at a page count
    # not divisible by four clobbers them.
    tail = flags[pages:]
    assert not bool((tail != 0).any()), (
        f"{where}: the flags wrote past the last page: {tail.tolist()} after "
        f"the {pages}-page plane"
    )

    got = cache.view(torch.uint8)
    expect = want_bytes.view(torch.uint8)
    for current in range(pages):
        if current in untouched:
            assert torch.equal(
                got[current], before[1][current].view(torch.uint8)
            ), f"{where}: page {current} is untouched and its bytes moved"
            assert torch.equal(scales[current], before[0][current]), (
                f"{where}: page {current} is untouched and its scale moved"
            )
            assert torch.equal(initialized[current], before[2][current]), (
                f"{where}: page {current} is untouched and its flag moved"
            )
            continue
        if interpreted:
            # Both sides convert through the interpreter, so a difference is
            # the rule's and not the format's rounding.
            close = torch.equal(got[current], expect[current])
        else:
            # The decoded values within the operator's tolerance, in the page's
            # scaled units.
            close = torch.allclose(
                cache[current].float(),
                want_bytes[current].float(),
                rtol=RTOL,
                atol=ATOL,
            )
        differing = (got[current] != expect[current]).reshape(-1).nonzero()
        first = int(differing[0]) if len(differing) else 0
        assert close, (
            f"{where}: page {current} differs at {len(differing)} positions "
            f"beyond {'none' if interpreted else f'rtol {RTOL} atol {ATOL}'}; "
            f"first at {first}: "
            f"{float(cache[current].float().reshape(-1)[first])} vs "
            f"{float(want_bytes[current].float().reshape(-1)[first])} at "
            f"scale {float(scales[current])}"
        )
    return scales.reshape(-1).tolist()


def f32_quotient(numerator, denominator):
    """`numerator / denominator` divided in f32, as the kernel divides."""
    return float(
        torch.tensor(numerator, dtype=torch.float32)
        / torch.tensor(denominator, dtype=torch.float32)
    )


def test_a_fresh_page_takes_its_proposal(
    torch_device, interpreted, kernel_source
):
    seeds = torch.tensor(
        [[0.0] * 16, [3.0] * 16, [-7.0] * 16, [0.0] * 16], dtype=torch.float32
    )
    scales = write(
        kernel_source("kv_write_scaled"),
        torch_device,
        interpreted,
        dtype="f32",
        seeds=seeds,
        **RULE,
    )
    proposal = f32_quotient(7.0, E4M3_MAX)
    assert scales[0] == proposal, (
        f"a fresh page: the scale is {scales[0]}, the proposal {proposal}"
    )


def test_an_all_zero_write_floors_the_scale(
    torch_device, interpreted, kernel_source
):
    scales = write(
        kernel_source("kv_write_scaled"),
        torch_device,
        interpreted,
        dtype="f32",
        seeds=torch.zeros(4, 16, dtype=torch.float32),
        **RULE,
    )
    # The floor is an f32 constant and the division is f32, so the expectation
    # is the same two constants divided in the same width.
    floored = f32_quotient(SCALE_FLOOR, E4M3_MAX)
    assert scales[0] == floored, (
        f"an all-zero write: the scale is {scales[0]}, the floor {floored}"
    )


def test_a_smaller_proposal_does_not_shrink_the_page(
    torch_device, interpreted, kernel_source
):
    stored = 1.0
    scales = write(
        kernel_source("kv_write_scaled"),
        torch_device,
        interpreted,
        dtype="f32",
        seeds=torch.full((4, 16), 0.25, dtype=torch.float32),
        prefill=prefilled(4, 2, 8, scale=stored),
        **RULE,
    )
    assert scales[0] == stored, (
        f"a smaller proposal: the scale became {scales[0]}, stored {stored}"
    )


def test_a_larger_proposal_re_encodes_the_stored_rows(
    torch_device, interpreted, kernel_source
):
    """The stored page is written at a small scale, so the incoming values
    propose a larger one and every stored row has to move; the reference holds
    the decoded values of the page's earlier rows.
    """  # noqa: D205
    stored = 0.1
    scales = write(
        kernel_source("kv_write_scaled"),
        torch_device,
        interpreted,
        dtype="f32",
        seeds=torch.full((4, 16), 400.0, dtype=torch.float32),
        prefill=prefilled(4, 2, 8, scale=stored),
        **RULE,
    )
    assert scales[0] > stored, (
        f"a larger proposal: the scale stayed {scales[0]} of {stored}, so the "
        "re-encode never ran"
    )


#: The ownership of a page by its first token: a page's tokens consecutive,
#: interleaved with another page's, and on both sides of a block of the slots'
#: scan, which a scan of four slots puts at every fourth token.
INTERLEAVED = [9, 0, 10, 3, 1, 11, -1, 4, 2, 5, 6, 12, 7, 8, 13]

#: The writes held to the rule alone, by the case they state.
WRITES = {
    # Two pages, one of them touched: the untouched page keeps its bytes, its
    # scale and its flag.
    "a page the launch does not touch": {
        "dtype": "bf16",
        "tokens": 2,
        "pages": 2,
        "page": 4,
        "heads": 2,
        "head_dim": 8,
        "block": 8,
        "slots": [7, 5],
        "prefill": prefilled(4, 2, 8, scale=0.5, seed=5),
        "untouched": (0,),
    },
    # Every slot skipped: nothing is written and nothing is flagged.
    "no token persisted": {
        "dtype": "bf16",
        "tokens": 3,
        "pages": 2,
        "page": 2,
        "heads": 1,
        "head_dim": 8,
        "block": 8,
        "slots": [-1, -1, -1],
        "untouched": (0, 1),
    },
    # The row walked in passes, the block narrower than a row and wider.
    **{
        f"a row walked in passes of {block}, five pages": {
            "dtype": "f32",
            "tokens": 5,
            "pages": 5,
            "page": 3,
            "heads": 3,
            "head_dim": 5,
            "block": block,
            "slots": [3, 0, 4, -1, 2],
            "untouched": (2, 3, 4),
        }
        for block in (4, 8, 16, 32)
    },
    # A scan of four slots, then the package's own scan.
    **{
        (
            "pages written by their first token, a scan of "
            f"{scan or 'the package'}"
        ): {
            "dtype": "f32",
            "tokens": len(INTERLEAVED),
            "pages": 6,
            "page": 3,
            "heads": 2,
            "head_dim": 8,
            "block": 8,
            "slots": INTERLEAVED,
            "scan": scan,
            "untouched": (5,),
        }
        for scan in (4, None)
    },
    # A slot written twice in one launch holds the later row, as the launch's
    # order states.
    "a slot written twice holds the later row": {
        "dtype": "f32",
        "tokens": 4,
        "pages": 2,
        "page": 4,
        "heads": 1,
        "head_dim": 16,
        "block": 16,
        "slots": [1, 2, 1, 6],
    },
    # A prefill's long run: 300 tokens over five pages of 64 behind a prefix, a
    # grown page among them, and padding, with a scan of sixteen slots.
    "a long run of consecutive slots": {
        "dtype": "bf16",
        "tokens": 320,
        "pages": 7,
        "page": 64,
        "heads": 2,
        "head_dim": 8,
        "block": 16,
        "slots": list(range(20, 320)) + [-1] * 20,
        "prefill": prefilled(64, 2, 8, scale=0.001),
        "scan": 16,
        "untouched": (5, 6),
    },
}


@pytest.mark.parametrize("case", WRITES)
def test_a_write_follows_the_rule(
    case, torch_device, interpreted, kernel_source
):
    write(
        kernel_source("kv_write_scaled"),
        torch_device,
        interpreted,
        **WRITES[case],
    )
