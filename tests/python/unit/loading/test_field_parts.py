"""Each retained field states the part of its logical tensor it holds."""

import json

import pytest
import torch
from torch.distributed.tensor import Shard

from uniserve.distributed import DeviceMesh, Distribution
from uniserve.distributed.logical import Logical
from uniserve.loading.retained import (
    field_parts,
    from_canonical,
    tensor_entry,
    to_canonical,
)
from uniserve.quantization import Quantizer, RowOrder, ScaleLayout

pytestmark = pytest.mark.unit


def test_an_affine_quantized_tensor_holds_its_part_in_every_field():
    # A per-tensor FP8 matrix, the second of two row shards: values hold
    # the tensor's part element by element, and its one scale is whole.
    mesh = DeviceMesh(ranks=(0, 1), shape=(2,), axes=("tp",), rank=1)
    weight = Logical.distributed(
        Distribution(mesh, (Shard(0),)), (8, 16), ("out", "in"), (1, 0)
    )
    encoded = Quantizer("fp8").quantize(torch.randn(4, 16))
    parts = field_parts(tensor_entry("p:w", encoded), encoded, weight)
    assert parts["values"]["encoding"] == {"kind": "affine"}
    assert parts["values"]["logical"] == weight.describe()
    assert parts["scale"]["encoding"] == {"kind": "affine"}
    assert parts["scale"]["logical"]["segments"] == []
    json.dumps(parts)


def test_an_encoded_tensor_converts_through_its_canonical_form():
    # Block scales in swizzled tiles and rows in a shuffled order are no
    # strides of the tensor's part: the fields convert through their
    # canonical form, in which each holds its part affinely.
    experts, rows, columns = 2, 128, 64
    linear = Quantizer("mxfp8").from_tensors(
        {
            "values": torch.randint(
                0, 120, (experts, rows, columns), dtype=torch.uint8
            ).view(torch.float8_e4m3fn),
            "scale": torch.randint(
                100, 130, (experts * rows, columns // 32), dtype=torch.uint8
            ),
        },
        shape=(experts, rows, columns),
        dtype=torch.bfloat16,
    )
    stored = linear.repack(
        scale_layout=ScaleLayout.SWIZZLED_128X4,
        row_order=RowOrder.SHUFFLED_128,
    )
    whole = Logical.whole((experts, rows, columns))
    parts = field_parts(tensor_entry("p:w", stored), stored, whole)
    values = parts["values"]["encoding"]
    assert values["kind"] == "encoded" and values["group"] == "p:w"
    assert values["canonical"]["shape"] == [experts, rows, columns]
    # Linear block scales flatten the matrices' rows; viewed in the tensor's
    # dimensions they hold one scale per 32 elements of a row, after the
    # values in the group's staging bytes.
    scale = parts["scale"]["encoding"]["canonical"]
    assert scale["shape"] == [experts, rows, columns // 32]
    assert scale["offset"] % 256 == 0
    assert scale["offset"] >= experts * rows * columns
    json.dumps(parts)

    # The library's conversions are inverse to each other.
    canonical = to_canonical(stored)
    for name, value in canonical.buffers().items():
        assert torch.equal(
            value.view(torch.uint8), linear.buffers()[name].view(torch.uint8)
        )
    again = from_canonical(
        canonical,
        {
            "scale_layout": stored.scale_layout.value,
            "row_order": stored.row_order.value,
        },
    )
    for name, value in again.buffers().items():
        assert torch.equal(
            value.view(torch.uint8), stored.buffers()[name].view(torch.uint8)
        )
