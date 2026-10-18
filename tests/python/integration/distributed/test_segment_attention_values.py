"""Segment-wise VSA keeps its values through every sequence partition.

Each rank projects its context interval of the gathered sequence for its head
shard, attends every row of it against the keys of the whole sequence and
receives its own token shard back with every head; the rows must equal the
dense equations of the unpartitioned layer, also when a captured call replays
over changed values.
"""

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from tests.python.fixtures.distributed import (
    bind_mesh,
    process_world,
    rendezvous,
)
from tests.python.fixtures.vsa import (
    SEGMENT_TILE,
    SEGMENT_VALID,
    segment_reference,
    segment_tables,
)
from uniserve import cuda_graph
from uniserve.bindings.context import ExecutionContext
from uniserve.distributed import DeviceMesh, parallelize_
from uniserve.nn import MergedColumnParallelLinear
from uniserve.nn.attention import (
    AttentionParallelConfig,
    ContextParallelConfig,
    Ulysses,
    vsa,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

HEADS, WIDTH, HIDDEN = 4, 128, 64
ROWS = len(SEGMENT_VALID) * SEGMENT_TILE

# Mesh shape, axes and the attention partition they realize.
CASES = {
    "ulysses": (
        (2,),
        ("heads",),
        AttentionParallelConfig(heads=Ulysses("heads")),
    ),
    "context": (
        (2,),
        ("context",),
        AttentionParallelConfig(
            context=ContextParallelConfig(gather_axis="context")
        ),
    ),
    "hybrid": (
        (2, 2),
        ("context", "heads"),
        AttentionParallelConfig(
            heads=Ulysses("heads"),
            context=ContextParallelConfig(gather_axis="context"),
        ),
    ),
}


def _run(rank, listener, case):
    shape, axes, config = CASES[case]
    members = torch.Size(shape).numel()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    with process_world(rank, members, listener, device) as world:
        generator = torch.Generator(device="cpu").manual_seed(91)
        hidden = torch.randn(
            ROWS, HIDDEN, dtype=torch.bfloat16, generator=generator
        ).to(device)
        matrices = {
            name: (
                torch.randn(HEADS * WIDTH, HIDDEN, generator=generator) / 8
            ).to(device, torch.bfloat16)
            for name in ("q", "k", "v", "gate")
        }
        # The projection gathers the token shards of every axis together.
        mesh = bind_mesh(
            world,
            DeviceMesh(
                ranks=tuple(range(members)),
                shape=shape,
                axes=axes,
                rank=rank,
            ),
            axes=(*((axis,) for axis in axes), axes),
        )
        layer = torch.nn.Module()
        layer.projection = MergedColumnParallelLinear(
            HIDDEN,
            dict.fromkeys(matrices, HEADS * WIDTH),
            branch_width=WIDTH,
            bias=False,
            dtype=torch.bfloat16,
            device=device,
        )
        for name, value in matrices.items():
            layer.projection.projections[name].weight.data.copy_(value)
        layer.attention = vsa.SegmentAttention(vsa.BlockAttention(WIDTH**-0.5))
        parallelize_(layer, mesh, attention=config)

        heads = mesh.get_group(
            () if config.heads is None else config.heads.axis
        )
        context = mesh.get_group(
            () if config.context is None else config.context.gather_axis
        )
        local_heads = HEADS // heads.size
        interval = layer.attention.query_rows(ROWS)
        # The token shard this rank owns: its share of the context interval.
        shard = (interval.stop - interval.start) // heads.size
        owned = slice(
            interval.start + heads.rank * shard,
            interval.start + (heads.rank + 1) * shard,
        )
        valid, tile_segments, starts, keep = segment_tables(device)
        # The leading tiles are dense or empty.
        segments = vsa.Segments(
            SEGMENT_TILE,
            ROWS,
            valid,
            tile_segments,
            starts,
            keep,
            dense_prefix_tiles=3,
        )

        def attend(execution):
            # The layer's call sites borrow the context's bindings, which
            # an activation installs.
            with execution.activate():
                chunks = sorted(
                    layer.projection.forward_chunks(
                        hidden[owned], token_slice=owned, num_tokens=ROWS
                    ),
                    key=lambda chunk: chunk[0].start,
                )
                q, k, v, gate = (
                    torch.cat([values[name] for _, values in chunks]).view(
                        interval.stop - interval.start, local_heads, WIDTH
                    )
                    for name in matrices
                )
                return layer.attention(q, k, v, gate, segments)

        def check(actual):
            projections = [
                F.linear(hidden, value).view(ROWS, HEADS, WIDTH)
                for value in matrices.values()
            ]
            expected, attending = segment_reference(*projections)
            assert actual.shape == (shard, HEADS, WIDTH)
            # The established BF16 attention tolerance of the VSA tests.
            torch.testing.assert_close(
                actual[attending[owned]],
                expected[owned][attending[owned]],
                rtol=2e-2,
                atol=2e-2,
            )

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with (
            torch.inference_mode(),
            ExecutionContext(layer, stream=stream) as execution,
        ):
            execution.prepare(None)
            check(attend(execution))

            captured = torch.empty(
                (shard, HEADS, WIDTH), dtype=torch.bfloat16, device=device
            )
            graph = cuda_graph.capture(
                lambda: captured.copy_(attend(execution)), stream=stream
            )
            hidden.mul_(0.75)
            graph.replay()
            torch.cuda.synchronize(device)
            check(captured)
            graph.release()
        assert context.size * heads.size == members


@pytest.mark.parametrize("case", sorted(CASES))
def test_segment_attention_keeps_values_through_partitions(case):
    shape = CASES[case][0]
    members = torch.Size(shape).numel()
    if torch.cuda.device_count() < members:
        pytest.skip(f"the {case} partition needs {members} devices")
    with rendezvous() as listener:
        mp.spawn(
            _run,
            args=(listener, case),
            nprocs=members,
            join=True,
        )
