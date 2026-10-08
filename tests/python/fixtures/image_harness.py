"""The worker's image harness library.

``crates/worker/examples/image_harness.rs`` (built with ``cargo build -p
uniserve-worker --example image_harness``) binds image runners over the
real numerical interface and returns their Rust-owned results.
``UNISERVE_IMAGE_HARNESS`` overrides its path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import tvm_ffi


def image_harness():
    """Load the harness library, failing the test when it is not built."""
    root = Path(__file__).resolve().parents[3]
    path = Path(
        os.environ.get(
            "UNISERVE_IMAGE_HARNESS",
            root / "target" / "debug" / "examples" / "libimage_harness.so",
        )
    )
    if not path.exists():
        pytest.fail(
            f"{path} is missing: build it with "
            "`cargo build -p uniserve-worker --example image_harness`"
        )
    return tvm_ffi.load_module(str(path))
