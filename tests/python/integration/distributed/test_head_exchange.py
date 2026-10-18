"""A chunked head exchange returns the whole exchange's rows by interval.

Each member receives, interval by interval and in token order, exactly the
rows ``HeadExchange.tokens`` returns for the same head-shard rows: with the
logical members opposite to backend order, with a shard spanning several
exchange intervals, and when a captured exchange replays over new values. A
consumer that stops after the first interval leaves no transfer writing the
exchange's storage.
"""

import pytest
import torch
import torch.multiprocessing as mp

from tests.python.fixtures.distributed import (
    bind_mesh,
    process_world,
    rendezvous,
)
from uniserve import cuda_graph
from uniserve.distributed import DeviceMesh
from uniserve.distributed.chunks import exchange_rows
from uniserve.distributed.tokens import HeadExchange

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

HEADS, FEATURES = 2, 128


@torch.inference_mode()
def _run(rank, listener):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    with process_world(rank, 2, listener, device) as world:
        # Logical members run opposite to backend order.
        mesh = bind_mesh(
            world,
            DeviceMesh(ranks=(1, 0), shape=(2,), axes=("heads",), rank=rank),
        )
        exchange = HeadExchange(mesh.get_group("heads"))
        # Two whole exchange intervals and a partial third one per member.
        step = exchange_rows(HEADS * FEATURES * 2, 1)
        rows = 2 * step + 384
        generator = torch.Generator(device=device).manual_seed(rank)
        values = torch.randn(
            2 * rows, HEADS, FEATURES, device=device, generator=generator
        ).bfloat16()

        def compose(interval, payload, slots):
            # Member m's tokens of this member's heads are its rows onward.
            for member, slot in enumerate(slots):
                start = member * rows + interval.start
                payload[slot].copy_(
                    values[start : start + interval.stop - interval.start]
                )

        def chunks():
            return exchange.token_chunks(
                rows,
                HEADS,
                FEATURES,
                dtype=values.dtype,
                device=device,
                compose=compose,
            )

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            expected = exchange.tokens(values)
            received = list(chunks())
            assert [interval for interval, _ in received] == [
                slice(start, min(start + step, rows))
                for start in range(0, rows, step)
            ]
            assert torch.equal(
                torch.cat([value for _, value in received]), expected
            )

            partial = chunks()
            first, value = next(partial)
            assert torch.equal(value, expected[first])
            partial.close()

        captured = torch.empty_like(expected)

        def replayed():
            captured.copy_(torch.cat([value for _, value in chunks()]))
            return captured

        graph = cuda_graph.capture(replayed, stream=stream)
        with torch.cuda.stream(stream):
            values.neg_()
            actual = graph.replay()
            assert torch.equal(actual, exchange.tokens(values))
        torch.cuda.synchronize(device)
        graph.release()


def test_chunked_head_exchange_returns_the_whole_exchange():
    if torch.cuda.device_count() < 2:
        pytest.skip("the head exchange needs two devices")
    with rendezvous() as listener:
        mp.spawn(_run, args=(listener,), nprocs=2, join=True)
