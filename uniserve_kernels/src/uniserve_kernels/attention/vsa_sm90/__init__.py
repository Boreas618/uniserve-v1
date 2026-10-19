"""SM90a CUDA block-sparse attention over 128-row tiles.

The kernel sources live in ``csrc/`` and build through
:mod:`uniserve_kernels.jit` on the CUTLASS headers vendored with DeepGEMM:
the first :func:`load` with a given source and toolchain compiles them,
later processes import the finished module, and support queries never
compile. ``uniserve.backends.attention.vsa.sm90`` calls :func:`load` while
preparing its operator and routes complete 128-row calls here.
"""

import re
from functools import cache
from pathlib import Path

import torch

# Rows of one query tile and of one key block.
TILE = 128

# The CuTe and CUTLASS headers the kernel is written against.
_CUTLASS = (
    Path(__file__).parents[2]
    / "deepgemm"
    / "csrc"
    / "third-party"
    / "cutlass"
    / "include"
)


def supported(device: torch.device | None = None) -> bool:
    """Report whether ``device`` can run the extension, without compiling it.

    The extension contains sm_90a code, whose architecture-conditional
    instructions (GMMA, register reconfiguration) run on compute capability
    9.0 alone.
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability(
        device
    ) == (9, 0)


def load() -> None:
    """Compile or load the extension before serving or CUDA capture."""
    _extension()


def _cutlass_version() -> str:
    """The vendored CUTLASS release, which enters the build identity."""
    text = (_CUTLASS / "cutlass" / "version.h").read_text()
    parts = []
    for part in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"#define CUTLASS_{part} (\d+)", text)
        if match is None:
            raise RuntimeError(f"{_CUTLASS} states no CUTLASS release")
        parts.append(match.group(1))
    return ".".join(parts)


@cache
def _extension():
    from uniserve_kernels import jit

    directory = Path(__file__).parent / "csrc"
    return jit.load(
        "uniserve_sparse_attention_sm90",
        [directory / "attention.cu"],
        headers=sorted(directory.glob("*.cuh")),
        # The headers are read in place; their release stands for them in
        # the build identity, as an installed dependency's version does.
        include_dirs=[_CUTLASS],
        dependencies=[f"cutlass=={_cutlass_version()}"],
        cxx_flags=["-O3", "-std=c++20"],
        # GMMA and register reconfiguration are architecture-conditional,
        # so the target is exactly sm_90a.
        cuda_flags=[
            "-O3",
            "-std=c++20",
            "--use_fast_math",
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            "-Xcompiler=-fno-strict-aliasing",
            "-gencode=arch=compute_90a,code=sm_90a",
        ],
    )


def block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    *,
    scale: float,
    query_valid_rows: torch.Tensor | None = None,
) -> None:
    """Attend from query tiles to their selected complete key blocks.

    Q, K, V and ``out`` are BF16 ``[rows, heads, 128]`` tensors with
    contiguous channels and 16-byte-aligned row and head strides, such as
    row-major views of merged projections; each may use its own strides. Q and
    K row counts are independent multiples of 128, the rows of one query tile
    and of one key block. Metadata uses global key-block indices ``[heads,
    query tiles, selected]``, counts ``[heads, query tiles]`` and per-key-block
    valid sizes in ``[0, 128]``; these device values are consumed without a
    host synchronization. Call :func:`load` before CUDA capture.

    Only the first ``counts[h, t]`` indices of each query tile are read. The
    kernel does not range-check metadata values: counts must not exceed the
    selected width and indices must address existing key blocks. Keys past a
    block's valid size never contribute, at any positive scale, and a query
    tile with a zero count or no valid selected key produces zero output.

    ``query_valid_rows``, int32 ``[query tiles]`` in ``[0, 128]``, states the
    valid rows that lead each query tile; ``None`` states every row valid.
    Valid query rows are always computed. A tile with at most 64 valid rows
    computes its first 64 rows and writes zeros to the rest, which spares half
    its work; padded rows of other tiles are computed. The call launches on
    the current CUDA stream and writes ``out`` in place.
    """
    _extension().forward(
        query,
        key,
        value,
        out,
        indices,
        counts,
        variable_block_sizes,
        query_valid_rows,
        float(scale),
    )
