"""A pytest plugin for compute-sanitizer's initcheck: it marks initialized
the outputs of the kernels whose stores initcheck does not record.

compute-sanitizer 2025.3's initcheck does not record the stores of the
cuBLAS GEMM kernels on GB200, nor those of the SM100 block-sparse
attention kernels: every later read of their outputs is reported as a read
of uninitialized storage (a single ``torch.addmm`` followed by an RMS norm
gives one report per 16-byte load of the product, and none when the same
tensor is written by ``normal_`` or marked initialized), and those reports
hide the ones that matter. While the plugin is active, a dispatch mode
marks the bytes of each matrix-product operator's outputs initialized, and
a wrapper of the block-sparse attention entries marks their outputs, through
the NVTX API (``mark_initialized.c``, compiled on first use), on the current
stream after the kernel is enqueued; every other store stays as initcheck
records it. Outside a sanitizer the NVTX calls do nothing.

    compute-sanitizer --tool initcheck ... python -m pytest
    -p tests.python.sanitizer.initcheck_marks ...

A marked output is not checked, and initcheck probably does not check the
reads those kernels make either (inference).
"""  # noqa: D205

import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

_aten = torch.ops.aten

# The matrix-product operators whose outputs cuBLAS writes. ``linear`` and
# ``matmul`` reach a dispatch mode whole (their cuBLAS calls run beneath
# it), so they are marked as products themselves.
PRODUCTS = {
    _aten.linear,
    _aten.matmul,
    _aten.mm,
    _aten.addmm,
    _aten.bmm,
    _aten.baddbmm,
    _aten.mv,
    _aten.addmv,
    _aten._addmm_activation,
    _aten._scaled_mm,
    _aten._int_mm,
}


def _library() -> ctypes.CDLL:
    """The NVTX shim, compiled from ``mark_initialized.c`` with the CUDA
    toolkit's headers (``$CUDA_HOME``, ``/usr/local/cuda`` by default).
    """  # noqa: D205
    source = Path(__file__).with_name("mark_initialized.c")
    target = Path(tempfile.mkdtemp(prefix="uniserve-initcheck-")) / "libmark.so"
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    subprocess.run(
        [
            os.environ.get("CC", "cc"),
            "-shared",
            "-fPIC",
            "-O2",
            f"-I{cuda}/include",
            str(source),
            "-o",
            str(target),
        ],
        check=True,
    )
    library = ctypes.CDLL(str(target))
    library.mark_initialized.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    return library


def _runs(tensor: torch.Tensor) -> list[tuple[int, int]]:
    """The contiguous byte runs ``(pointer, bytes)`` a tensor's elements
    occupy: one for a dense tensor, one per innermost row otherwise.
    """  # noqa: D205
    size = tensor.element_size()
    if tensor.numel() == 0:
        return []
    if tensor.is_contiguous():
        return [(tensor.data_ptr(), tensor.numel() * size)]
    if tensor.stride(-1) != 1:
        return [(element.data_ptr(), size) for element in tensor.reshape(-1)]
    width = tensor.shape[-1] * size
    base = tensor.data_ptr()
    # Rows of a strided view: every leading index's first element.
    offsets = torch.zeros(1, dtype=torch.int64)
    for extent, stride in zip(tensor.shape[:-1], tensor.stride()[:-1]):
        offsets = (offsets[:, None] + torch.arange(extent) * stride).reshape(-1)
    return [(base + int(offset) * size, width) for offset in offsets.tolist()]


class _Marks:
    """Marks device tensors initialized on the current stream."""

    def __init__(self) -> None:
        self.library = _library()

    def __call__(self, tensors) -> None:
        runs = [
            run
            for tensor in tensors
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda
            for run in _runs(tensor)
        ]
        if not runs:
            return
        pointers = (ctypes.c_uint64 * len(runs))(
            *(pointer for pointer, _ in runs)
        )
        sizes = (ctypes.c_size_t * len(runs))(*(size for _, size in runs))
        stream = torch.cuda.current_stream().cuda_stream
        self.library.mark_initialized(
            ctypes.c_void_p(stream), len(runs), pointers, sizes
        )


class _MarkProducts(TorchDispatchMode):
    """Mark the outputs of every matrix-product operator initialized."""

    def __init__(self, marks: _Marks) -> None:
        super().__init__()
        self.marks = marks

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func.overloadpacket in PRODUCTS:
            self.marks(tree_leaves(result))
        return result


def _marking_output(function, marks: _Marks):
    """Wrap a block-sparse attention entry ``(q, k, v, out, ...)`` so that
    its output is marked initialized after its kernel is enqueued.
    """  # noqa: D205

    def wrapped(q, k, v, out, *args, **kwargs):
        result = function(q, k, v, out, *args, **kwargs)
        marks([out])
        return result

    return wrapped


_mode: _MarkProducts | None = None


def pytest_configure(config):
    global _mode
    torch.cuda.init()
    marks = _Marks()
    from uniserve_kernels.attention import vsa_cute, vsa_native

    for module in (vsa_cute, vsa_native):
        module.block_sparse_attention = _marking_output(
            module.block_sparse_attention, marks
        )
    _mode = _MarkProducts(marks)
    _mode.__enter__()


def pytest_unconfigure(config):
    if _mode is not None:
        _mode.__exit__(None, None, None)
