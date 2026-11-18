"""The sampling_from_logits cubin draws the token the operator's rule names.

The rule (``ops::sampler::SamplingFromLogits``): in each row's descending
order, equal logits in ascending token id, the top-k width keeps the first
``limit`` positions, the nucleus keeps a position while the mass before it is
at most ``floor(top_p * the top-k set's mass)``, and min-p keeps the logits at
or above ``max + ln(min_p)``; a mass is ``exp(x - max)`` in FP32 held as an
integer in units of 2^-40. The draw takes the first kept token by id whose
cumulative mass exceeds ``floor(draw * total)``. The arithmetic is integer, so
the module's token is compared with the rule's exactly.

The cases reach every way the kernel finds a boundary: a top-k width or a
nucleus whose candidates it lists in shared memory, a width or a nucleus too
wide to list (selected digit by digit over the row), a row drawn whole, min-p,
a width that cuts a group of equal logits, rows that are not 16-byte aligned,
a vocabulary narrower than one chunk, and the rows that keep nothing usable.

The distribution is held to an independent reference: the top-k, top-p and
min-p filtering of a sorted FP64 softmax as vLLM's sampler applies it, whose
probabilities the module's tokens over evenly spaced draws must reproduce to
the spacing.

The module is the one the session's Build Server builds for the session's
device, launched as the operator launches it: one block per row.
"""

import ctypes
import json
import math

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.gpu]

KEY = "logits_f32_tokens_i32"

#: The operator's parameters, in declaration order.
SIGNATURE = {
    "logits": "*fp32",
    "row_stride": "i64",
    "rows": "i32",
    "vocabulary": "i32",
    "parameters": "*fp32",
    "draws": "*fp32",
    "tokens": "*i32",
    "valid": "*i32",
}

#: The columns of a row of request parameters, and the three it reads.
PARAMETER_COLUMNS = 8
TOP_K, TOP_P, MIN_P = 1, 2, 3

#: A mass's unit, and the vocabulary of the checkpoint the sampler serves.
MASS_UNIT = 2**40
VOCABULARY = 151936


@pytest.fixture(scope="module")
def module(kernel_module):
    """The built module directory."""
    return kernel_module("sampling_from_logits", "nvcc", KEY)


@pytest.fixture(scope="module")
def metadata(module):
    """The module's `metadata.json`."""
    return json.loads((module / "metadata.json").read_text())


def launch(module, metadata, logits, parameters, draws):
    """One launch over `logits` (rows of a view whose row stride may exceed
    the vocabulary): answers the tokens and the validity flags, on the host.
    """  # noqa: D205
    from cuda.bindings import driver

    def checked(result):
        error, *values = result
        assert error == driver.CUresult.CUDA_SUCCESS, (
            f"a CUDA driver call failed: {error}"
        )
        return values[0] if len(values) == 1 else values

    rows, vocabulary = logits.shape
    tokens = torch.full((rows,), -7, dtype=torch.int32, device="cuda")
    valid = torch.full((rows,), -7, dtype=torch.int32, device="cuda")
    parameters = parameters.to("cuda").contiguous()
    draws = draws.to("cuda").contiguous()
    torch.cuda.synchronize()

    loaded = checked(
        driver.cuModuleLoadData((module / "module.cubin").read_bytes())
    )
    try:
        function = checked(
            driver.cuModuleGetFunction(loaded, metadata["name"].encode())
        )
        values = (
            logits.data_ptr(),
            logits.stride(0),
            rows,
            vocabulary,
            parameters.data_ptr(),
            draws.data_ptr(),
            tokens.data_ptr(),
            valid.data_ptr(),
        )
        kinds = (
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        checked(
            driver.cuLaunchKernel(
                function,
                rows,
                1,
                1,
                metadata["num_warps"] * 32,
                1,
                1,
                metadata.get("shared", 0),
                torch.cuda.current_stream().cuda_stream,
                (values, kinds),
                0,
            )
        )
        torch.cuda.synchronize()
    finally:
        checked(driver.cuModuleUnload(loaded))
    return tokens.cpu().tolist(), valid.cpu().tolist()


def keys_of(row):
    """The order-preserving keys of a row of FP32 logits."""
    bits = row.view(np.uint32)
    return np.where(
        bits >> 31 == 1, ~bits, bits | np.uint32(0x80000000)
    ).astype(np.uint32)


def share(total, fraction):
    """`floor(fraction * total)`, exactly."""
    return (total * int(float(np.float32(fraction)) * 2.0**64)) >> 64


def rule(row, top_k, top_p, min_p, draw):
    """The operator's token and validity flag for one row, exactly."""
    vocabulary = row.size
    keys = keys_of(row)
    order = np.lexsort((np.arange(vocabulary), ~keys))
    largest = int(keys[order[0]])
    negative_nans = int((keys < 0x007FFFFF).sum())
    limit = (
        int(math.floor(min(np.float32(top_k), np.float32(vocabulary))))
        if top_k > 0
        else vocabulary
    )
    maximum = row[order[0]]
    applies = not min_p <= 0
    threshold = (
        np.float32(maximum + np.float32(np.log(np.float32(min_p))))
        if applies
        else None
    )
    usable = (
        0x007FFFFF < largest < 0xFF800000
        and 0 < limit <= vocabulary - negative_nans
        and not top_p < 0
        and (not applies or maximum >= threshold)
    )
    if not usable:
        return 0, 0
    masses = np.rint(
        np.exp((row - maximum).astype(np.float32)).astype(np.float32)
        * np.float32(MASS_UNIT)
    ).astype(np.uint64)

    kept = limit
    if top_p < 1:
        top_k_mass = sum(int(masses[token]) for token in order[:limit])
        nucleus = share(top_k_mass, top_p)
        before, kept = 0, 0
        for token in order[:limit]:
            if before > nucleus:
                break
            kept += 1
            before += int(masses[token])
    positions = order[:kept]
    if applies:
        positions = positions[row[positions] >= threshold]

    chosen = np.sort(positions)
    total = sum(int(masses[token]) for token in chosen)
    target = share(total, draw)
    running = 0
    for token in chosen:
        running += int(masses[token])
        if masses[token] > 0 and running > target:
            return int(token), 1
    raise AssertionError("the rule found no crossing")


def language_logits(rows, vocabulary, generator, temperature):
    """Rows shaped like a language model's at `temperature`: a broad body
    and a few dozen tokens well above it.
    """  # noqa: D205
    logits = torch.randn(rows, vocabulary, generator=generator) * 2.5
    for row in range(rows):
        top = torch.randint(0, vocabulary, (64,), generator=generator)
        logits[row, top] += torch.rand(64, generator=generator) * 12 + 6
    return logits / temperature


def tied_row(vocabulary):
    """A row whose largest logits are 200 equal ones, then 3000 equal ones."""
    row = torch.zeros(vocabulary)
    row[100:300] = 5.0
    row[4000:7000] = 4.0
    return row


#: The cases: the rows' logits, their (top-k, top-p, min-p), and the path
#: through the kernel they take.
CASES = {
    # The checkpoint's generation defaults: a width of 20 the first digit
    # bounds, listed.
    "the defaults, top-k 20 and top-p 0.95": ("language", 0.6, (20, 0.95, 0.0)),
    "a nucleus alone, listed": ("language", 0.6, (0, 0.95, 0.0)),
    "a top-k alone, listed": ("language", 1.0, (50, 1.0, 0.0)),
    "a width too wide to list": ("language", 0.6, (5000, 0.95, 0.0)),
    "a flat nucleus, too wide to list": ("language", 20.0, (0, 0.99, 0.0)),
    "the whole row": ("language", 1.0, (0, 1.0, 0.0)),
    "min-p over the whole row": ("language", 1.0, (0, 1.0, 0.05)),
    "min-p binding under a nucleus": ("language", 0.6, (0, 0.9, 0.3)),
    "every filter at once": ("language", 1.5, (40, 0.8, 0.05)),
    "a width that cuts equal logits, listed": ("tied", None, (250, 1.0, 0.0)),
    "a width that cuts equal logits, too wide to list": (
        "tied",
        None,
        (2500, 0.97, 0.0),
    ),
    "a nucleus that ends among equal logits": ("tied", None, (0, 0.97, 0.0)),
}

#: The draws of every case: the bottom and the top of the unit interval and
#: a spread between.
DRAWS = (0.0, 0.123, 0.5, 0.876, (2**24 - 1) / 2**24)


def parameter_rows(filters, rows):
    """`rows` rows of request parameters stating `filters`."""
    top_k, top_p, min_p = filters
    parameters = torch.zeros(rows, PARAMETER_COLUMNS)
    parameters[:, 0] = 1.0
    parameters[:, TOP_K] = top_k
    parameters[:, TOP_P] = top_p
    parameters[:, MIN_P] = min_p
    return parameters


@pytest.mark.parametrize("case", CASES)
def test_every_row_draws_the_rules_token(case, module, metadata):
    kind, temperature, filters = CASES[case]
    generator = torch.Generator().manual_seed(len(case))
    rows = len(DRAWS)
    if kind == "language":
        logits = language_logits(rows, VOCABULARY, generator, temperature)
    else:
        logits = tied_row(9000).repeat(rows, 1)
    draws = torch.tensor(DRAWS, dtype=torch.float32)

    got = launch(
        module,
        metadata,
        logits.cuda(),
        parameter_rows(filters, rows),
        draws,
    )

    wanted = [
        rule(logits[row].numpy(), *filters, float(draws[row]))
        for row in range(rows)
    ]
    assert list(zip(*got)) == wanted, (
        f"{case}: drew {list(zip(*got))}, the rule draws {wanted}"
    )


def test_rows_off_alignment_and_narrower_than_a_chunk(module, metadata):
    # A vocabulary of 1001 puts every odd row off a 16-byte boundary; one of
    # 7 is narrower than one chunk of 128 tokens.
    generator = torch.Generator().manual_seed(11)
    for vocabulary in (1001, 7):
        logits = language_logits(4, vocabulary, generator, 1.0)
        filters = [(20, 0.95, 0.0), (0, 0.9, 0.0), (3, 1.0, 0.0), (0, 1.0, 0.1)]
        parameters = torch.cat([parameter_rows(f, 1) for f in filters])
        draws = torch.tensor([0.3, 0.6, 0.9, 0.0])

        got = launch(module, metadata, logits.cuda(), parameters, draws)

        wanted = [
            rule(logits[row].numpy(), *filters[row], float(draws[row]))
            for row in range(4)
        ]
        assert list(zip(*got)) == wanted, f"a vocabulary of {vocabulary}"


def test_a_row_that_keeps_nothing_usable_reports_it(module, metadata):
    vocabulary = 300
    rows = torch.linspace(-3.0, 3.0, vocabulary).repeat(7, 1)
    rows[0] = float("-inf")
    rows[1, 17] = float("nan")
    rows[2, 5] = float("inf")
    filters = [
        (0, 1.0, 0.0),
        (0, 1.0, 0.0),
        (20, 1.0, 0.0),
        (0.5, 1.0, 0.0),
        (0, -0.1, 0.0),
        (0, 1.0, 2.0),
        (0, 1.0, 0.0),
    ]
    parameters = torch.cat([parameter_rows(f, 1) for f in filters])

    tokens, valid = launch(
        module, metadata, rows.cuda(), parameters, torch.full((7,), 0.5)
    )

    # A row of no finite logit, a NaN or +inf among the candidates, a width
    # below one, a negative nucleus and a min-p above one keep nothing.
    assert valid == [0, 0, 0, 0, 0, 0, 1]
    assert tokens[:6] == [0] * 6


def reference_probabilities(row, top_k, top_p, min_p):
    """The distribution vLLM's sampler draws from, in FP64: the top-k mask of
    the sorted logits, the top-p mask of their ascending cumulative softmax
    (a token is dropped when the mass at or below it is at most `1 - top_p`),
    then min-p against the largest probability.
    """  # noqa: D205
    logits = row.double()
    if top_k > 0:
        kth = torch.topk(logits, int(top_k)).values[-1]
        logits = torch.where(logits < kth, float("-inf"), logits)
    ordered, order = torch.sort(logits, descending=False, stable=True)
    probabilities = torch.softmax(ordered, dim=-1)
    dropped = torch.cumsum(probabilities, dim=-1) <= 1 - top_p
    dropped[-1] = False
    ordered = torch.where(dropped, float("-inf"), ordered)
    logits = torch.full_like(logits, float("-inf")).scatter(0, order, ordered)
    probabilities = torch.softmax(logits, dim=-1)
    if min_p > 0:
        floor = probabilities.max() * min_p
        probabilities = torch.where(probabilities < floor, 0.0, probabilities)
        probabilities = probabilities / probabilities.sum()
    return probabilities


@pytest.mark.parametrize(
    "filters",
    [(20, 0.95, 0.0), (0, 0.9, 0.0), (50, 1.0, 0.0), (0, 1.0, 0.1)],
    ids=["defaults", "nucleus", "top-k", "min-p"],
)
def test_evenly_spaced_draws_reproduce_the_reference_distribution(
    filters, module, metadata
):
    # One language row at the defaults' temperature, drawn at 2^14 evenly
    # spaced uniforms: each token takes its probability's share of the draws,
    # to the spacing, plus the reference's FP64 rounding.
    draws = 1 << 14
    generator = torch.Generator().manual_seed(5)
    row = language_logits(1, VOCABULARY, generator, 0.6)[0]
    if filters == (0, 1.0, 0.1):
        row = row * 0.6 / 1.5
    uniforms = (torch.arange(draws, dtype=torch.float64) + 0.5) / draws

    # Every draw reads the one row: a row stride of zero.
    tokens, valid = launch(
        module,
        metadata,
        row.cuda().expand(draws, -1),
        parameter_rows(filters, draws),
        uniforms.float(),
    )

    assert all(flag == 1 for flag in valid)
    shares = (
        torch.bincount(torch.tensor(tokens), minlength=VOCABULARY).double()
        / draws
    )
    wanted = reference_probabilities(row, *filters)
    assert bool(((shares > 0) <= (wanted > 0)).all()), (
        f"{filters}: a token drawn outside the reference's support"
    )
    # A token of probability p takes floor(p * draws) or one more of the
    # evenly spaced draws, so each share is within 1 / draws of p.
    support = int((wanted > 0).sum())
    distance = 0.5 * float((shares - wanted).abs().sum())
    assert distance <= 0.5 * support / draws + 1e-6, (
        f"{filters}: total variation {distance:.3g} from the reference over "
        f"{support} candidates and {draws} draws"
    )


def test_the_modules_signature_is_the_operators(metadata):
    assert metadata.get("signature") == SIGNATURE, (
        f"the module's parameters are {metadata.get('signature')}, the "
        f"operator's are {SIGNATURE}"
    )
