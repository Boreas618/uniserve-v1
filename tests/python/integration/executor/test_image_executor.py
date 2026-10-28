"""An image executor warms, seals and replays its computations on one GPU.

The tests play the worker's half of the stub model's image computations
under CUDA graphs: they bind the computations to a prefix cache, latent
banks and a graph pool, run every warmup phase the executor states over
scratch tables installed as the worker installs them (each step's page
columns selected from the scratch slots' pages), seal, and then serve a
latent encode, a trajectory's preparation, guided denoising steps and its
decode. Every served forward replays a graph warmup captured, and every
result equals the same batches served by an executor without graphs.
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
    StreamBinding,
)
from uniserve.executor.text.protocol import GraphMode
from uniserve.prefix_cache.units import UnitPool

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="the image computations run on a GPU",
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
    """The worker's half of an image executor's computations on one stream."""

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
        computations = self.executor.computations()
        self.computations = {
            kind: index
            for index, entry in enumerate(computations)
            for kind in entry["kinds"]
        }
        self.report = self.executor.bind(
            ImageBinding(
                slots=SLOTS,
                computations=tuple(
                    StreamBinding(
                        stream=self.stream,
                        scratch=Scratch(),
                        pool=torch.cuda.MemPool() if graphs else None,
                    )
                    for _ in computations
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

    def run(self, kind, calls, tensors, forward=None, predicate=None):
        """Prepare and execute one batch.

        ``predicate`` gates the first call on a leased ``uint8 [1]`` device
        predicate, which follows every other tensor as the worker binds it.
        """
        computation = self.computations[kind]
        step = {"batch_id": 1, "computation": computation, "calls": calls}
        prepared = self.executor.prepare(computation, step)
        bound = [
            tensor
            for entry, planned in zip(calls, prepared, strict=True)
            for tensor in tensors(entry, planned)
        ]
        if forward is not None:
            bound.append(forward)
        if predicate is not None:
            calls[0]["predicate"] = len(bound)
            bound.append(predicate)
        modes = self.executor.execute(
            computation, {**step, "prepared": prepared}, bound
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


# The served trajectory's pages and its rows of the latent banks.
LATENT = {"pages": [3, 4], "units": 4, "height": 32, "width": 32}


def _start(worker: _Worker) -> None:
    """Start one guided image request in slot 1."""
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


def _preparation() -> dict:
    """The trajectory's latent preparation, into bank 1."""
    return {
        **_call("latent_preparation", image_index=0),
        "latent": {"read": 1, "write": 1, "step": 0, **LATENT},
    }


def _denoising(worker: _Worker, read: int, write: int, step: int):
    """A guided denoising step and its attention rows.

    Its branches read slot 1's context, then the empty prefixes of slots 2
    and 3.
    """
    call = {
        **_call("denoising"),
        "latent": {"read": read, "write": write, "step": step, **LATENT},
        "image": {
            "input": None,
            "input_images": 1,
            "progress": {"logical": 4, "visible": 4},
            "allocated": [[1, 16], [2, 0], [3, 0]],
        },
    }
    query = 4 + 2
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
    return call, forward


def _serve(worker: _Worker) -> dict:
    """Serve one guided image request's batches; return what they wrote."""
    _start(worker)
    results: dict = {"modes": []}

    # A generated image's latent encode.
    pixels = torch.linspace(0, 1, 3 * 32 * 32, device=DEVICE)
    pixels = pixels.reshape(3, 32, 32).to(torch.bfloat16)
    encode = {
        **_call("latent_encoding", latent_draw=0),
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

    # The trajectory, guided by all three branches.
    worker.install([4, 0, 0])
    results["modes"] += worker.run(
        "latent_preparation", [_preparation()], lambda entry, planned: []
    )
    for step, (read, write) in enumerate(((1, 0), (0, 1))):
        call, forward = _denoising(worker, read, write, step)
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
        "latent": {"read": 1, "write": 1, "step": 2, **LATENT},
    }
    product = torch.zeros((3, 32, 32), dtype=torch.bfloat16, device=DEVICE)
    # The RGB bytes, then the product's BF16 values.
    staged = torch.zeros(9 * 32 * 32, dtype=torch.uint8).pin_memory()
    results["modes"] += worker.run(
        "image_decoding", [decode], lambda entry, planned: [product, staged]
    )
    results["product"] = product.cpu()
    results["bytes"] = staged.clone()
    # The staged values are the product's: the image's exact feedback
    # input, which a migrated request carries.
    values = staged[3 * 32 * 32 :].view(torch.bfloat16).reshape(3, 32, 32)
    assert torch.equal(values, results["product"])
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


def test_a_carried_image_encodes_as_its_resident_product(stub):
    """A carried image encodes as the product it was decoded as.

    A migrated request's carried image, the decoded product's values the
    host staged, encodes to the latent rows the product itself encodes to,
    at the same posterior draw.
    """
    worker = _Worker(stub, graphs=False)
    worker.warm()
    _start(worker)
    pixels = torch.linspace(-1, 1, 3 * 32 * 32, device=DEVICE)
    pixels = pixels.reshape(3, 32, 32).to(torch.bfloat16)
    image = {
        "shape": [3, 32, 32],
        "height": 32,
        "width": 32,
        "value_range": "signed_unit",
    }
    encoded = {}
    for kind, source in (
        ("resident", pixels),
        ("carried", pixels.cpu().pin_memory()),
    ):
        encode = {
            **_call("latent_encoding", latent_draw=3),
            "image": {"input": {kind: image}, "input_images": 0},
        }
        rows = torch.zeros((4, 768), dtype=torch.bfloat16, device=DEVICE)
        worker.run(
            "latent_encoding", [encode], lambda entry, planned: [source, rows]
        )
        encoded[kind] = rows.cpu()
    assert torch.equal(encoded["carried"], encoded["resident"])
    assert encoded["resident"].abs().sum() > 0


def test_a_false_predicate_leaves_the_trajectory_as_it_was(stub):
    worker = _Worker(stub, graphs=False)
    _start(worker)
    worker.install([4, 0, 0])
    true = torch.ones(1, dtype=torch.uint8, device=DEVICE)
    false = torch.zeros(1, dtype=torch.uint8, device=DEVICE)
    pages = slice(3, 5)
    nothing = lambda entry, planned: []  # noqa: E731

    # A skipped preparation leaves the bank it would write as it was; a
    # run one writes the trajectory's initial latent.
    worker.banks[1].fill_(0.5)
    before = worker.banks[1][pages].clone()
    worker.run("latent_preparation", [_preparation()], nothing, predicate=false)
    assert torch.equal(worker.banks[1][pages], before)
    worker.run("latent_preparation", [_preparation()], nothing, predicate=true)
    initial = worker.banks[1][pages].clone()
    assert not torch.equal(initial, before)

    # A skipped step writes the committed latent; a run one writes what the
    # ungated step writes. The stub predicts zero, so no step advances the
    # latent here (`test_image_families` steps a real denoiser).
    written = {}
    for name, predicate in (("false", false), ("true", true), ("none", None)):
        worker.banks[0].zero_()
        call, forward = _denoising(worker, 1, 0, 0)
        worker.run(
            "denoising", [call], nothing, forward=forward, predicate=predicate
        )
        written[name] = worker.banks[0][pages].clone()
    assert torch.equal(written["false"], initial)
    assert torch.equal(written["true"], written["none"])
