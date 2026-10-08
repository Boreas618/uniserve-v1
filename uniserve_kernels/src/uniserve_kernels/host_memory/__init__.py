"""Pinned host storage read by native asynchronous copies.

PyTorch's pinned allocator reuses a block once the streams it knows about
retire their copies. A native library that enqueues its own DMA from pinned
storage (FlashInfer's plan upload) records its stream here, so the block
outlives that copy.
"""

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _extension():
    # One build or cache lookup per process, content-addressed by
    # uniserve_kernels.jit.load.
    from uniserve_kernels import jit

    return jit.load(
        "uniserve_host_memory",
        [Path(__file__).parent / "csrc" / "host_memory.cpp"],
        cxx_flags=["-O2", "-std=c++20"],
        with_cuda=True,
    )


def record_host_usage(tensor: torch.Tensor, stream: torch.cuda.Stream) -> None:
    """Retain pinned storage through a native asynchronous DMA submission.

    Call after native code enqueues a host/device copy outside PyTorch's
    copy operator. The pinned allocator delays storage reuse until this
    stream retires the copy; the owner must leave the submitted contents
    immutable meanwhile.
    """
    _extension().record_host_usage(
        tensor, stream.device_index, int(stream.cuda_stream)
    )
