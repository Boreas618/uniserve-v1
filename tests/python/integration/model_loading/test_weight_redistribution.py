"""A copy manifest turns one parallel configuration's weights into another's.

Each rank of a source and a target configuration loads Qwen3-30B-A3B on
the host as a worker loads it on its GPU (``uniserve.models.registry``),
describes its storages (``uniserve.loading.retained.Layout``) and has the
worker library chunk and key them as a worker retains them
(``weight_storages``). The library then plans the manifest from the source
world to the target world (``copy_manifest``) over four GPUs, and the
manifest is executed on the source ranks' bytes: its read-only mappings as
the chunks they name, its copies row by row, its fills. Every field of
every target rank then holds exactly the bytes the target rank loaded
directly.

The scenarios are the four-GPU ones of a world change of Qwen3-30B-A3B:
two data-parallel replicas of tensor parallelism 2 becoming four
expert-parallel ranks with data-parallel attention, and two such ranks
becoming four ranks of tensor-parallel attention over four-way expert
parallelism. The latter partition is assembled from two loads of one
rank, its attention and embeddings under tensor parallelism and its
experts under expert parallelism, as the library partitions each.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

from uniserve import loading, worker
from uniserve.distributed import Communicator, DeviceMesh
from uniserve.loading import precision
from uniserve.loading.retained import Layout, fingerprints
from uniserve.models import registry
from uniserve.nn.moe import FusedMoE

pytestmark = [pytest.mark.integration, pytest.mark.slow]

MODEL = Path(
    os.environ.get("UNISERVE_QWEN3_MOE", "/workspace/models/Qwen3-30B-A3B")
)

# The allocation granularity of the GPUs weights are retained on: CUDA's
# POSIX-descriptor VMM granularity of a GB200.
GRANULARITY = 2 << 20

# The tensor- and expert-parallel degrees the four GPUs' configurations
# use, which retained weights are chunked for.
DEGREES = [1, 2, 4]

# Four GPUs by UUID, as the Storage Server and its Transfer Server name them.
GPUS = [[index + 1] * 16 for index in range(4)]


def _name(gpu) -> str:
    return f"gpu{GPUS.index(list(gpu))}"


def _links():
    """Links between the GPUs as the Transfer Server's listing reports its
    measurement (``transfer.links``); these bandwidths stand in for it, a
    copy within a GPU faster than one between two.
    """  # noqa: D205
    return [
        {
            "link": {
                "source": {"device": {"gpu": source}},
                "destination": {"device": {"gpu": target}},
            },
            "bytes_per_second": 3 * 10**12 if source == target else 6 * 10**11,
        }
        for source in GPUS
        for target in GPUS
    ]


@dataclass(frozen=True)
class Partition:
    """One rank's partition: tensor-parallel rank ``tp_rank`` of ``tp``
    (attention, embeddings and, without expert parallelism, the experts'
    channels) and expert-parallel rank ``ep_rank`` of ``ep``.
    """  # noqa: D205

    tp: int = 1
    tp_rank: int = 0
    ep: int = 1
    ep_rank: int = 0


@dataclass
class Rank:
    """A loaded rank: its torch storages, layout columns and storages."""

    models: list
    columns: dict
    planned: dict


def _mesh(partition: Partition) -> DeviceMesh:
    members = tuple(range(partition.tp))
    mesh = DeviceMesh(
        ranks=members,
        shape=(partition.tp,),
        axes=("tp",),
        rank=partition.tp_rank,
    )
    group = Communicator(members, partition.tp_rank, "tp")
    object.__setattr__(mesh, "_groups", {("tp",): group})
    return mesh


def _load(config, partition: Partition, **selection):
    """The model one load of ``partition`` materializes on the host."""
    started = time.perf_counter()
    experts = (
        Communicator(tuple(range(partition.ep)), partition.ep_rank, "ep")
        if partition.ep > 1
        else None
    )
    meshes = {"": _mesh(partition)} if partition.tp > 1 else None
    result = registry.load_model(
        config, device="cpu", meshes=meshes, experts=experts, **selection
    )
    print(
        f"loaded {partition} {sorted(selection)} in "
        f"{time.perf_counter() - started:.0f} s",
        flush=True,
    )
    return result


def _merge(columns: list[dict]) -> dict:
    """The columns of several layouts of one rank, storages renumbered."""
    merged: dict[str, list] = {name: [] for name in columns[0]}
    storages = 0
    for part in columns:
        for name, values in part.items():
            if name == "field_storages":
                values = [storages + index for index in values]
            merged[name].extend(values)
        storages += len(part["storage_bytes"])
    return merged


def _rank(partition: Partition) -> Rank:
    """Load ``partition`` and plan its retained storages."""
    io = loading.Config()
    config = registry.read_config(MODEL, io=io)
    # The weight choices a launch without precision selectors makes.
    config = replace(
        config, weights=precision.weight_config(config, {}, "bfloat16")
    )
    if partition.tp > 1 and partition.ep > 1:
        # Experts divide over the expert group alone; attention and the
        # embeddings over the tensor-parallel group.
        with torch.device("meta"):
            skeleton = config.model_class(config.model)
        experts = frozenset(
            path
            for path, module in skeleton.named_modules()
            if isinstance(module, FusedMoE)
        )
        loads = [
            _load(
                config,
                Partition(partition.tp, partition.tp_rank),
                exclude_modules=experts,
            ),
            _load(
                config,
                Partition(ep=partition.ep, ep_rank=partition.ep_rank),
                modules=experts,
            ),
        ]
    else:
        loads = [_load(config, partition)]
    columns = _merge(
        [
            Layout.describe(
                result.model,
                origins=result.origins,
                devices=frozenset(("cpu",)),
            ).columns()
            for result in loads
        ]
    )
    description = {
        **columns,
        "granularity": [GRANULARITY],
        "degrees": DEGREES,
        "fingerprints": fingerprints(config, io, "local"),
    }
    planned = json.loads(
        str(worker.library()["weight_storages"](json.dumps(description)))
    )
    return Rank([result.model for result in loads], columns, planned)


def _storages(rank: Rank) -> list[torch.UntypedStorage]:
    """Each torch storage of the rank, in layout order."""
    layouts = []
    for model in rank.models:
        layout = Layout.describe(model, devices=frozenset(("cpu",)))
        layouts.extend(layout._storages)
    return layouts


def _images(rank: Rank) -> list[torch.Tensor]:
    """Each retained storage's bytes as loading leaves them: a storage of
    its own viewed in place, a pack of small ones copied together.
    """  # noqa: D205
    storages = _storages(rank)
    members: dict[int, list[tuple[int, int]]] = {}
    for index, (storage, offset) in enumerate(rank.planned["members"]):
        members.setdefault(storage, []).append((index, offset))
    images = []
    for index, planned in enumerate(rank.planned["storages"]):
        held = members[index]
        padded = -(-planned["layout"]["bytes"] // GRANULARITY) * GRANULARITY
        if len(held) == 1 and held[0][1] == 0:
            storage = storages[held[0][0]]
            view = torch.empty(0, dtype=torch.uint8)
            view.set_(storage, 0, (storage.nbytes(),), (1,))
            images.append(view)
            continue
        image = torch.zeros(padded, dtype=torch.uint8)
        for member, offset in held:
            storage = storages[member]
            view = torch.empty(0, dtype=torch.uint8)
            view.set_(storage, 0, (storage.nbytes(),), (1,))
            image[offset : offset + storage.nbytes()] = view
        images.append(image)
    return images


def _world(ranks: list[Rank], gpus: list[str | None]) -> dict:
    return {
        "ranks": [
            {"gpu": gpu, "storages": rank.planned["storages"]}
            for rank, gpu in zip(ranks, gpus, strict=True)
        ]
    }


def _strided(image: torch.Tensor, offset: int, length: int, rows: int, pitch):
    return image.as_strided(
        (rows, length), (pitch if rows > 1 else 1, 1), offset
    )


def _apply(
    manifest, rank: int, sources, source_storages, storages
) -> list[torch.Tensor]:
    """Execute what ``manifest`` directs for target rank ``rank``, whose
    planned storages are ``storages``, from the source ranks' images (of
    their planned storages ``source_storages``) into fresh storage: its
    mappings, copies and fills.
    """  # noqa: D205
    assert not manifest["stagings"], "dense weights convert nothing"
    applied = [
        torch.full(
            (-(-storage["layout"]["bytes"] // GRANULARITY) * GRANULARITY,),
            0xCD,
            dtype=torch.uint8,
        )
        for storage in storages
    ]
    for mapping in manifest["mappings"]:
        if mapping["rank"] != rank:
            continue
        source = sources[mapping["source_rank"]][mapping["source_storage"]]
        mapped = source_storages[mapping["source_rank"]][
            mapping["source_storage"]
        ]["chunks"][mapping["source_chunk"]]
        chunk = storages[mapping["storage"]]["chunks"][mapping["chunk"]]
        assert mapped["key"] == chunk["key"]
        # The target maps the source's chunk: its bytes where the target's
        # chunk lies. A source image of its own ends at its torch storage's
        # bytes; the chunk's tail past them is padding no field holds.
        start = mapped["offset"]
        held = min(start + mapped["length"], source.numel()) - start
        into = chunk["offset"]
        applied[mapping["storage"]][into : into + held] = source[
            start : start + held
        ]
    for copy in manifest["copies"]:
        if copy["target"]["rank"] != rank:
            continue
        assert copy["source"]["space"]["kind"] == "storage"
        source = sources[copy["source"]["rank"]][
            copy["source"]["space"]["storage"]
        ]
        target = applied[copy["target"]["space"]["storage"]]
        _strided(
            target,
            copy["target"]["offset"],
            copy["length"],
            copy["rows"],
            copy["target_pitch"],
        ).copy_(
            _strided(
                source,
                copy["source"]["offset"],
                copy["length"],
                copy["rows"],
                copy["source_pitch"],
            )
        )
    for fill in manifest["fills"]:
        if fill["rank"] != rank:
            continue
        _strided(
            applied[fill["storage"]],
            fill["offset"],
            fill["length"],
            fill["rows"],
            fill["pitch"],
        ).zero_()
    return applied


def _differing_fields(planned, applied, direct) -> list[str]:
    """The fields of a target rank whose applied bytes differ from the
    bytes it loaded directly.
    """  # noqa: D205
    differing = []
    for storage, image, loaded in zip(
        planned["storages"], applied, direct, strict=True
    ):
        for field in storage["layout"]["fields"]:
            # The field's elements as bytes: a view of a wider matrix
            # leaves gaps between its rows that are no field's.
            element = field["element"]
            shape = (*field["shape"], element)
            strides = (*(stride * element for stride in field["strides"]), 1)

            def elements(values, shape=shape, strides=strides, field=field):
                return values.as_strided(shape, strides, field["offset"])

            if not torch.equal(elements(image), elements(loaded)):
                differing.append(field["tensor"])
    return differing


SCENARIOS = {
    # Two replicas of tensor parallelism 2 on GPUs 0, 1 and 2, 3 become
    # four expert-parallel ranks with data-parallel attention.
    "dp2-tp2-to-dep4": (
        [Partition(tp=2, tp_rank=gpu % 2) for gpu in range(4)],
        GPUS,
        [Partition(ep=4, ep_rank=rank) for rank in range(4)],
    ),
    # Two expert-parallel ranks on GPUs 0 and 1 become four ranks of
    # tensor-parallel attention over four-way expert parallelism.
    "dep2-to-tep4": (
        [Partition(ep=2, ep_rank=rank) for rank in range(2)],
        GPUS[:2],
        [
            Partition(tp=4, tp_rank=rank, ep=4, ep_rank=rank)
            for rank in range(4)
        ],
    ),
}


@pytest.mark.skipif(
    not MODEL.is_dir(), reason="needs the Qwen3-30B-A3B checkpoint"
)
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_an_update_holds_the_weights_a_direct_start_loads(scenario):
    sources, source_gpus, targets = SCENARIOS[scenario]
    # Ranks of equal partitions load the same bytes: each loads once.
    loaded: dict[Partition, Rank] = {}
    for partition in sources:
        if partition not in loaded:
            loaded[partition] = _rank(partition)
    source_ranks = [loaded[partition] for partition in sources]
    source_images = {
        partition: _images(rank) for partition, rank in loaded.items()
    }

    # Each target rank loads once: its storages plan the manifest, its
    # bytes check what the manifest makes of the source's.
    target_ranks: list[Rank | None] = [
        _rank(partition) for partition in targets
    ]
    plans = [rank.planned for rank in target_ranks if rank is not None]
    request = {
        "source": _world(source_ranks, source_gpus),
        "target": {
            "ranks": [
                {"gpu": None, "storages": plan["storages"]} for plan in plans
            ]
        },
        "gpus": GPUS,
        "links": _links(),
    }
    started = time.perf_counter()
    manifest = json.loads(
        str(worker.library()["copy_manifest"](json.dumps(request)))
    )
    print(
        f"{scenario}: planned in {time.perf_counter() - started:.1f} s; "
        f"GPUs {[_name(gpu) for gpu in manifest['gpus']]}; "
        f"{len(manifest['mappings'])} mapped chunks, "
        f"{len(manifest['copies'])} copies, {len(manifest['fills'])} fills",
        flush=True,
    )
    assert not manifest["missing"]

    sources_by_rank = [source_images[partition] for partition in sources]
    for index, plan in enumerate(plans):
        applied = _apply(
            manifest,
            index,
            sources_by_rank,
            [rank.planned["storages"] for rank in source_ranks],
            plan["storages"],
        )
        direct = target_ranks[index]
        assert direct is not None
        differing = _differing_fields(plan, applied, _images(direct))
        mapped = sum(
            plan["storages"][mapping["storage"]]["chunks"][mapping["chunk"]][
                "length"
            ]
            for mapping in manifest["mappings"]
            if mapping["rank"] == index
        )
        copied: dict[str, int] = {}
        for copy in manifest["copies"]:
            if copy["target"]["rank"] == index:
                gpu = _name(source_gpus[copy["source"]["rank"]])
                copied[gpu] = copied.get(gpu, 0) + copy["length"] * copy["rows"]
        fields = sum(len(s["layout"]["fields"]) for s in plan["storages"])
        print(
            f"{scenario}: target rank {index} on "
            f"{_name(manifest['gpus'][index])}: "
            f"{mapped / 2**20:.1f} MiB mapped; MiB copied from "
            f"{ {gpu: round(size / 2**20) for gpu, size in copied.items()} }; "
            f"{len(differing)} of {fields} fields differ",
            flush=True,
        )
        assert not differing, differing[:10]
        # The rank's storage returns before the next one is checked.
        target_ranks[index] = None
        del applied, direct
