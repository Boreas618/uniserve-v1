"""The image executor serves an image model's lanes on host storage.

The stub model's components hold every kind of image work: a vision tower
(its own component), a latent encoder, a zero-predicting image denoiser over
the real schedule and solver, and an RGB image decoder. These tests drive
the executor as the worker does (``crates/worker/src/executor/image.rs``):
the lanes it states, each batch's preparation, then its execution on the
tensors the worker binds in the preparation's order, and compare what it
writes with the model's own computation.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from tests.python.fixtures.stub import load_stub
from uniserve.bindings.workspace import Scratch
from uniserve.diffusion import normal_noise
from uniserve.executor.image import (
    ImageBinding,
    ImageExecutor,
    ImageLaneBinding,
)
from uniserve.media import image
from uniserve.media.pixels import quantize_image_hwc
from uniserve.models import VisionInput
from uniserve.nn.rng import flow_noise_seed
from uniserve.prefix_cache.units import UnitPool
from uniserve.runner.encoder import EncoderRunner

pytestmark = pytest.mark.unit

SLOTS, BLOCK, UNITS = 3, 4, 16
# Latent pages: the padding page and four pages of four rows.
PAGES, PAGE_UNITS = 5, 4
KEY = {"engine_id": 1, "request_id": 9, "request_epoch": 1}


def _components():
    """The stub's placed components, as the worker states them."""
    return [
        {
            "name": "model",
            "host": False,
            "calls": [
                {"capability": "causal_lm", "path": ""},
                {"capability": "denoiser", "path": "denoiser"},
                {"capability": "image_denoiser", "path": "denoiser"},
                {"capability": "patch_autoencoder", "path": "latent_encoder"},
                {"capability": "image_decoder", "path": "image_decoder"},
            ],
            "kinds": [
                "prefill",
                "decode",
                "latent_encoding",
                "latent_preparation",
                "denoising",
                "image_decoding",
            ],
        },
        {
            "name": "vision_encoder",
            "host": False,
            "calls": [
                {"capability": "patch_encoder", "path": "vision_encoder"}
            ],
            "kinds": ["vision_encoding"],
        },
    ]


def _settings():
    return {
        "graphs": False,
        "model_dtype": "bfloat16",
        "components": _components(),
        "image": {
            "max_batch_calls": 2,
            "max_batch_tokens": 64,
            "max_request_slots": SLOTS,
            "max_model_len": 32,
            "flow_shapes": [[32, 32]],
        },
    }


@pytest.fixture(scope="module")
def stub():
    return load_stub()


class _Rank:
    """An image executor bound to host storage, as a worker binds it."""

    def __init__(self, stub) -> None:
        self.stub = stub
        self.executor = ImageExecutor(stub, _settings())
        self.lanes = {
            tuple(entry["kinds"]): index
            for index, entry in enumerate(self.executor.lanes())
        }
        self.pool = UnitPool.allocate(
            stub.module.cache_config,
            num_units=UNITS,
            block_size=BLOCK,
            device="cpu",
        )
        groups = self.pool.groups
        tables = len(self.pool.tables)
        # Slot 1 holds a four-token context in its first unit.
        unit_tables = torch.zeros((tables, SLOTS + 1, 4), dtype=torch.int32)
        unit_tables[:, 1, 0] = 1
        width = 3 * 16 * 16
        self.banks = (
            torch.zeros((PAGES, PAGE_UNITS, width), dtype=torch.bfloat16),
            torch.zeros((PAGES, PAGE_UNITS, width), dtype=torch.bfloat16),
        )
        self.report = self.executor.bind(
            ImageBinding(
                slots=SLOTS,
                lanes=tuple(
                    ImageLaneBinding(stream=None, scratch=Scratch())
                    for _ in self.executor.lanes()
                ),
                unit_pool=self.pool,
                tables={
                    "unit_tables": unit_tables,
                    "start_pages": torch.zeros(
                        (len(groups), SLOTS + 1), dtype=torch.int32
                    ),
                    "table_shapes": torch.tensor(
                        [
                            [table.group, groups[table.group].block_size, -1]
                            for table in self.pool.tables
                        ],
                        dtype=torch.int32,
                    ),
                },
                banks=self.banks,
            )
        )

    def lane(self, kind: str) -> int:
        return next(
            index for kinds, index in self.lanes.items() if kind in kinds
        )

    def run(self, kind, calls, tensors, forward=None):
        """Prepare then execute one batch of ``calls`` of ``kind``.

        ``tensors`` maps a call to its tensors given its preparation;
        returns the preparation and each call's graph mode.
        """
        lane = self.lane(kind)
        step = {"batch_id": 1, "lane": lane, "calls": calls}
        prepared = self.executor.prepare(lane, step)
        bound = [
            tensor
            for entry, planned in zip(calls, prepared, strict=True)
            for tensor in tensors(entry, planned)
        ]
        if forward is not None:
            bound.append(forward)
        modes = self.executor.execute(
            lane, {**step, "prepared": prepared}, bound
        )
        return prepared, modes


@pytest.fixture
def rank(stub) -> _Rank:
    return _Rank(stub)


def _payload(height: int, width: int, seed: int) -> str:
    raw = np.random.default_rng(seed).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )
    encoded = io.BytesIO()
    Image.fromarray(raw).save(encoded, format="PNG")
    return base64.b64encode(encoded.getvalue()).decode()


def _prepared(model, tower: str, payload: str):
    """An inline image's facts and pixels, as the worker's host prepares."""
    height, width = EncoderRunner.image_size(payload)
    facts = EncoderRunner.image_facts(model, tower, height, width, 1)
    dtype = getattr(torch, EncoderRunner.pixel_dtype(model))
    pixels = torch.empty(facts[4:], dtype=dtype)
    EncoderRunner.prepare_image(model, tower, payload, 1, pixels)
    return facts, pixels


def _call(kind: str, index: int = 0, slot: int = 1, **fields):
    return {
        "index": index,
        "slot": slot,
        "active": True,
        "refused": False,
        "call": {"kind": kind, "request_key": KEY, **fields},
        "output_names": [],
        "latent": None,
    }


def _start(rank: _Rank, steps: int = 2) -> None:
    rank.executor.start(
        [
            {
                "request_key": KEY,
                "request_pool_idx": 1,
                "generation": {"negative_token_ids": []},
                "image": {
                    "steps": steps,
                    "cfg_text_scale": 1.0,
                    "cfg_img_scale": 1.0,
                    "cfg_renorm_type": "global",
                    "cfg_renorm_min": 0.0,
                    "cfg_interval": [0.0, 1.0],
                    "timestep_shift": 1.0,
                    "height": 32,
                    "width": 32,
                    "seed": 7,
                },
                "input_images": 1,
            }
        ]
    )


def test_lanes_name_each_kind_of_work_and_the_resources_it_uses(rank):
    lanes = rank.executor.lanes()
    by_kind = {tuple(entry["kinds"]): entry for entry in lanes}
    assert by_kind[("vision_encoding",)]["component"] == "vision_encoder"
    assert by_kind[("vision_encoding",)]["resources"] == ["images"]
    assert by_kind[("latent_encoding",)]["resources"] == ["images"]
    denoiser = by_kind[("latent_preparation", "denoising", "image_decoding")]
    assert denoiser["resources"] == ["pages", "latents", "images"]
    assert all(entry["prepares"] and not entry["commits"] for entry in lanes)
    # The longest trajectory sizes the ledger.
    assert rank.executor.manifest()["latent"] == {
        "rows": 1024,
        "width": 3 * 16 * 16,
        "dtype": "bfloat16",
    }
    # Without graphs, the engine warms each branch count at a square image
    # the ledger holds, and admits steps within the denoiser's rows.
    report = rank.report
    assert report["captured_images"] is None
    assert [shape["branches"] for shape in report["warmups"]] == [1, 2, 3]
    assert report["capacity"]["max_step_rows"] == 2 * 3


def test_inline_vision_images_encode_as_the_tower_encodes_them(rank, stub):
    facts, pixels = _prepared(stub, "vit", _payload(32, 48, 3))
    calls = [
        {
            **_call("vision_encoding"),
            "image": {
                "input": {
                    "inline": {
                        "shape": facts[4:],
                        "grid": facts[2:4],
                        "height": facts[0],
                        "width": facts[1],
                    }
                },
                "input_images": 1,
            },
        }
    ]
    outputs = []

    def tensors(entry, planned):
        (write,) = planned["image"]["writes"]
        out = torch.full(write["shape"], float("nan"), dtype=torch.bfloat16)
        outputs.append(out)
        return [pixels, out]

    prepared, modes = rank.run("vision_encoding", calls, tensors)
    assert prepared[0]["image"]["writes"][0]["metadata"] == {
        "feature": {"height": 32, "width": 48}
    }
    assert modes == [0]
    grid = tuple(facts[2:4])
    (expected,) = stub.module.vision_encoder.encode(
        VisionInput((pixels,), (torch.tensor([grid]),), (grid,))
    )
    torch.testing.assert_close(outputs[0], expected, rtol=0, atol=0)


def test_inline_latent_images_encode_into_their_rows(rank, stub):
    facts, pixels = _prepared(stub, "vae", _payload(32, 32, 5))
    calls = [
        {
            **_call("latent_encoding"),
            "image": {
                "input": {
                    "inline": {
                        "shape": facts[4:],
                        "grid": None,
                        "height": facts[0],
                        "width": facts[1],
                    }
                },
                "input_images": 1,
            },
        }
    ]
    _start(rank)
    outputs = []

    def tensors(entry, planned):
        (write,) = planned["image"]["writes"]
        out = torch.zeros(write["shape"], dtype=torch.bfloat16)
        outputs.append(out)
        return [pixels, out]

    prepared, _ = rank.run("latent_encoding", calls, tensors)
    assert prepared[0]["image"]["writes"][0]["shape"] == [4, 768]
    expected = stub.module.latent_encoder.encode(pixels[None])[0]
    torch.testing.assert_close(outputs[0], expected, rtol=0, atol=0)


def _forward(slot: int, visible: int, query: int) -> torch.Tensor:
    """The worker's attention rows of one single-branch denoising call.

    Its branch's row on its own context, then its conditioning row; one
    cache group, each row selecting its slot's one page.
    """
    return torch.tensor(
        [
            [0, 0],
            [slot, slot],
            [visible + query, visible],
            [query, 0],
            [0, 0],
            [0, 0],
            [1, 1],
        ],
        dtype=torch.int64,
    )


def test_a_trajectory_opens_steps_and_decodes_through_the_ledger(rank, stub):
    _start(rank)
    pages = [1]
    latent = {
        "pages": pages,
        "units": 4,
        "height": 32,
        "width": 32,
    }

    # The latent preparation writes the image's initial latent into the
    # bank the ledger bound: its seeded noise, patched.
    prepare = {
        **_call("latent_preparation", image_index=0),
        "latent": {"read": 1, "write": 1, "step": 0, **latent},
    }
    prepared, modes = rank.run(
        "latent_preparation", [prepare], lambda entry, planned: []
    )
    assert prepared[0]["image"]["latent"] is True
    assert modes == [0]
    network = stub.module.denoiser
    size = image.Config(32, 32)
    noise = torch.empty(
        (1, *network.noise_shape("image", size)), dtype=torch.bfloat16
    )
    normal_noise((flow_noise_seed(7, 0),), out=(noise,))
    initial = torch.zeros((1, 4, 768), dtype=torch.bfloat16)
    network.prepare_latents(
        (size,),
        noise={"image": noise},
        state={"image": initial},
        constants={},
        workspace={},
    )
    first = rank.banks[1][1].clone()
    torch.testing.assert_close(first, initial[0], rtol=0, atol=0)

    # A step of the zero-predicting denoiser leaves the latent unchanged in
    # the other bank.
    step = {
        **_call("denoising"),
        "latent": {"read": 1, "write": 0, "step": 0, **latent},
        "image": {
            "input": None,
            "input_images": 1,
            "progress": {"logical": 4, "visible": 4},
            "allocated": [[1, 16]],
        },
    }
    _, modes = rank.run(
        "denoising",
        [step],
        lambda entry, planned: [],
        forward=_forward(1, 4, 4 + 2),
    )
    assert modes == [0]
    torch.testing.assert_close(rank.banks[0][1], first, rtol=0, atol=0)

    # The decode reads the committed latent, publishes its pixels and lends
    # their quantized bytes.
    decode = {
        **_call(
            "image_decoding",
            latent_input={"generation": 3},
            image_output={"generation": 1},
        ),
        "latent": {"read": 0, "write": 0, "step": 2, **latent},
    }
    written = {}

    def tensors(entry, planned):
        (write,) = planned["image"]["writes"]
        written["product"] = torch.zeros(write["shape"], dtype=torch.bfloat16)
        written["bytes"] = torch.zeros(
            planned["image"]["bytes"], dtype=torch.uint8
        )
        return [written["product"], written["bytes"]]

    prepared, _ = rank.run("image_decoding", [decode], tensors)
    assert prepared[0]["image"]["bytes"] == 3 * 32 * 32
    (decoded,) = stub.module.image_decoder.decode((first,), sizes=(size,))
    torch.testing.assert_close(
        written["product"], decoded.to(torch.bfloat16), rtol=0, atol=0
    )
    expected = quantize_image_hwc(decoded, value_range=(-1.0, 1.0))
    assert torch.equal(written["bytes"], expected.reshape(-1))


def test_resident_images_decode_by_quantizing(rank):
    pixels = torch.linspace(-1, 1, 3 * 8 * 16).reshape(3, 8, 16)
    resident = pixels.to(torch.bfloat16)
    decode = {
        **_call("image_decoding"),
        "image": {
            "input": {
                "resident": {
                    "shape": [3, 8, 16],
                    "height": 8,
                    "width": 16,
                    "value_range": "signed_unit",
                }
            },
            "input_images": 0,
        },
    }
    staged = {}

    def tensors(entry, planned):
        staged["bytes"] = torch.zeros(
            planned["image"]["bytes"], dtype=torch.uint8
        )
        return [resident, staged["bytes"]]

    prepared, _ = rank.run("image_decoding", [decode], tensors)
    assert prepared[0]["image"]["read"] is True
    expected = quantize_image_hwc(resident, value_range=(-1.0, 1.0))
    assert torch.equal(staged["bytes"], expected.reshape(-1))


def test_skipped_calls_bind_nothing(rank):
    skipped = {**_call("vision_encoding"), "active": False}
    prepared, modes = rank.run(
        "vision_encoding", [skipped], lambda entry, planned: []
    )
    assert prepared == [{"reads": [], "writes": [], "publication": None}]
    assert modes == [0]
