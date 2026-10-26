"""An image executor warms, seals and replays its lanes on one GPU.

The tests play the worker's half of the stub model's image lanes under CUDA
graphs: they bind the lanes to a prefix cache, latent banks and a graph
pool, run every warmup phase the executor states over scratch tables
installed as the worker installs them (each step's page columns selected
from the scratch slots' pages), seal, and then serve a latent encode, a
trajectory's preparation, guided denoising steps and its decode. Every
served forward replays a graph warmup captured, and every result equals the
same batches served by an executor without graphs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tests.python.fixtures.stub import LOCAL, load_model, read_config
from uniserve.bindings.workspace import Scratch
from uniserve.executor.image import (
    ImageBinding,
    ImageExecutor,
    ImageLaneBinding,
)
from uniserve.executor.text.protocol import GraphMode
from uniserve.prefix_cache.units import UnitPool

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="the image lanes run on a GPU"
    ),
]

DEVICE = torch.device("cuda", 0)
SLOTS, BLOCK, UNITS, WIDTH = 4, 4, 64, 8
PAGES, PAGE_UNITS = 9, 4
KEY = {"engine_id": 1, "request_id": 9, "request_epoch": 1}


def _settings(graphs: bool) -> dict:
    return {
        "graphs": graphs,
        "model_dtype": "bfloat16",
        "components": [
            {
                "name": "model",
                "host": False,
                "calls": [
                    {"capability": "causal_lm", "path": ""},
                    {"capability": "image_denoiser", "path": "denoiser"},
                    {
                        "capability": "patch_autoencoder",
                        "path": "latent_encoder",
                    },
                    {"capability": "image_decoder", "path": "image_decoder"},
                ],
                "kinds": [
                    "prefill",
                    "latent_encoding",
                    "latent_preparation",
                    "denoising",
                    "image_decoding",
                ],
            }
        ],
        "image": {
            "max_batch_calls": 1,
            "max_batch_tokens": 32,
            "max_request_slots": SLOTS,
            "max_model_len": WIDTH * BLOCK,
            "flow_shapes": [[32, 32]],
        },
    }


@pytest.fixture(scope="module")
def stub():
    return load_model(
        read_config(
            components=json.dumps({"model": LOCAL, "vision_encoder": LOCAL}),
            device="cuda:0",
        ),
        device="cuda:0",
    )


class _Worker:
    """The worker's half of an image executor's lanes on one stream."""

    def __init__(self, stub, graphs: bool) -> None:
        self.executor = ImageExecutor(stub, _settings(graphs))
        self.pool = UnitPool.allocate(
            stub.module.cache_config,
            num_units=UNITS,
            block_size=BLOCK,
            device=DEVICE,
        )
        groups = self.pool.groups
        tables = len(self.pool.tables)
        self.unit_tables = torch.zeros(
            (tables, SLOTS + 1, WIDTH), dtype=torch.int32, device=DEVICE
        )
        self.stream = torch.cuda.Stream(DEVICE)
        width = 3 * 16 * 16
        self.banks = tuple(
            torch.zeros(
                (PAGES, PAGE_UNITS, width), dtype=torch.bfloat16, device=DEVICE
            )
            for _ in range(2)
        )
        lanes = self.executor.lanes()
        self.lanes = {
            kind: index
            for index, entry in enumerate(lanes)
            for kind in entry["kinds"]
        }
        self.report = self.executor.bind(
            ImageBinding(
                slots=SLOTS,
                lanes=tuple(
                    ImageLaneBinding(
                        stream=self.stream,
                        scratch=Scratch(),
                        pool=torch.cuda.MemPool() if graphs else None,
                    )
                    for _ in lanes
                ),
                unit_pool=self.pool,
                tables={
                    "unit_tables": self.unit_tables,
                    "start_pages": torch.zeros(
                        (len(groups), SLOTS + 1),
                        dtype=torch.int32,
                        device=DEVICE,
                    ),
                    "table_shapes": torch.tensor(
                        [
                            [table.group, groups[table.group].block_size, -1]
                            for table in self.pool.tables
                        ],
                        dtype=torch.int32,
                        device=DEVICE,
                    ),
                },
                banks=self.banks,
            )
        )

    def install(self, lengths) -> None:
        """Give slots ``1..`` tables of ``lengths`` tokens.

        As the worker installs scratch tables and a batch's tables.
        """
        self.unit_tables.zero_()
        unit = 1
        for slot, length in enumerate(lengths, start=1):
            pages = -(-max(length, 1) // BLOCK)
            self.unit_tables[:, slot, :pages] = torch.arange(
                unit, unit + pages, dtype=torch.int32, device=DEVICE
            )
            unit += pages
        self.lengths = list(lengths)

    def columns(self, rows) -> torch.Tensor:
        """Each row's selected pages per cache group.

        A full-attention table's installed pages.
        """
        counts = [
            -(-max(self.lengths[slot - 1], 1) // BLOCK) if slot else 0
            for slot, _, _ in rows
        ]
        groups = len(self.pool.groups)
        return torch.tensor(
            [[0] * len(rows), counts] * groups, dtype=torch.int64
        )

    def warm(self) -> list[dict]:
        phases = self.executor.warmup_phases()
        for index, phase in enumerate(phases):
            if phase["lengths"] is None:
                self.executor.warmup_phase(index)
                continue
            self.install(phase["lengths"])
            self.executor.warmup_phase(
                index, *(self.columns(rows) for rows in phase["steps"])
            )
        self.executor.warmup_finish()
        self.executor.seal()
        return phases

    def run(self, kind, calls, tensors, forward=None):
        lane = self.lanes[kind]
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
        self.stream.synchronize()
        return modes


def _call(kind, **fields):
    return {
        "index": 0,
        "slot": 1,
        "active": True,
        "refused": False,
        "call": {"kind": kind, "request_key": KEY, **fields},
        "output_names": [],
        "latent": None,
    }


def _serve(worker: _Worker) -> dict:
    """Serve one guided image request's batches; return what they wrote."""
    worker.executor.start(
        [
            {
                "request_key": KEY,
                "request_pool_idx": 1,
                "generation": {"negative_token_ids": []},
                "image": {
                    "steps": 2,
                    "cfg_text_scale": 4.0,
                    "cfg_img_scale": 2.0,
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
    results: dict = {"modes": []}

    # A generated image's latent encode.
    pixels = torch.linspace(0, 1, 3 * 32 * 32, device=DEVICE)
    pixels = pixels.reshape(3, 32, 32).to(torch.bfloat16)
    encode = {
        **_call("latent_encoding"),
        "image": {
            "input": {
                "resident": {
                    "shape": [3, 32, 32],
                    "height": 32,
                    "width": 32,
                    "value_range": "unit",
                }
            },
            "input_images": 1,
        },
    }
    rows = torch.zeros((4, 768), dtype=torch.bfloat16, device=DEVICE)
    results["modes"] += worker.run(
        "latent_encoding", [encode], lambda entry, planned: [pixels, rows]
    )
    results["rows"] = rows.cpu()

    # The trajectory, guided by all three branches: slot 1's context, then
    # the empty prefixes of slots 2 and 3.
    worker.install([4, 0, 0])
    latent = {"pages": [3, 4], "units": 4, "height": 32, "width": 32}
    prepare = {
        **_call("latent_preparation", image_index=0),
        "latent": {"read": 1, "write": 1, "step": 0, **latent},
    }
    results["modes"] += worker.run(
        "latent_preparation", [prepare], lambda entry, planned: []
    )
    query = 4 + 2
    for step, (read, write) in enumerate(((1, 0), (0, 1))):
        call = {
            **_call("denoising"),
            "latent": {"read": read, "write": write, "step": step, **latent},
            "image": {
                "input": None,
                "input_images": 1,
                "progress": {"logical": 4, "visible": 4},
                "allocated": [[1, 16], [2, 0], [3, 0]],
            },
        }
        attention = [(1, 4, query), (2, 0, query), (3, 0, query), (1, 4, 0)]
        forward = torch.tensor(
            [
                [0] * 4,
                [slot for slot, _, _ in attention],
                [prefix + count for _, prefix, count in attention],
                [count for _, _, count in attention],
                [0] * 4,
                *worker.columns(attention).tolist(),
            ],
            dtype=torch.int64,
        )
        results["modes"] += worker.run(
            "denoising", [call], lambda entry, planned: [], forward=forward
        )
    results["latent"] = worker.banks[1][3:5].cpu()

    decode = {
        **_call(
            "image_decoding",
            latent_input={"generation": 3},
            image_output={"generation": 1},
        ),
        "latent": {"read": 1, "write": 1, "step": 2, **latent},
    }
    product = torch.zeros((3, 32, 32), dtype=torch.bfloat16, device=DEVICE)
    staged = torch.zeros(3 * 32 * 32, dtype=torch.uint8).pin_memory()
    results["modes"] += worker.run(
        "image_decoding", [decode], lambda entry, planned: [product, staged]
    )
    results["product"] = product.cpu()
    results["bytes"] = staged.clone()
    return results


def test_served_batches_replay_warmup_captures_and_equal_eager(stub):
    graphs = _Worker(stub, graphs=True)
    phases = graphs.warm()
    # The latent encoder's sizes, the denoiser's steps and condition
    # writes over scratch tables, then the decoder's sizes.
    assert [phase["lengths"] is None for phase in phases] == [
        True,
        False,
        False,
        True,
    ]
    assert [phase.get("writes") for phase in phases[1:3]] == [False, True]
    assert graphs.report["captured_latents"] == [[32, 32]]
    assert {
        (shape["height"], shape["width"])
        for shape in graphs.report["captured_images"]
    } == {(32, 32)}
    replayed = _serve(graphs)
    assert replayed["modes"] == [
        GraphMode.REPLAY,
        GraphMode.EAGER,
        GraphMode.REPLAY,
        GraphMode.REPLAY,
        GraphMode.REPLAY,
    ]

    eager = _Worker(stub, graphs=False)
    eager.warm()
    served = _serve(eager)
    assert served["modes"] == [GraphMode.EAGER] * 5
    for name in ("rows", "latent", "product", "bytes"):
        assert torch.equal(replayed[name], served[name]), name
    assert np.isfinite(replayed["latent"].float().numpy()).all()
