"""A suspended world's collectives replay in the graphs captured over them.

A parked standby suspends its world's NCCL communicators
(``uniserve.distributed.world.suspend``), which releases their device
storage, and resumes them at its promotion before any graph runs. Two ranks
capture an all-reduce over the world's instance communicator into a CUDA
graph, suspend and resume the world, and replay the graph: it sums the new
inputs of both ranks, at a small and at a large message size.
"""

import pytest
import torch
import torch.multiprocessing as mp

from tests.python.fixtures.distributed import process_world, rendezvous
from uniserve import cuda_graph
from uniserve.distributed import world as distributed

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

# Elements of a message NCCL sends in one chunk, and of one it pipelines.
SIZES = (1024, 1 << 24)


def _replay_after_resume(rank, listener):
    with process_world(rank, 2, listener, device="cuda") as world:
        communicator = world.instance
        stream = torch.cuda.Stream()
        for elements in SIZES:
            value = torch.zeros(elements, device=world.device)
            out = torch.empty_like(value)

            def step(value=value, out=out):
                return communicator.all_reduce(value, op="sum", out=out)

            # Warm the communicator up eagerly, then capture the step.
            with torch.cuda.stream(stream):
                step()
            torch.cuda.synchronize()
            graph = cuda_graph.capture(step, stream=stream)

            def replay(scale, value=value, out=out, graph=graph):
                value.fill_(scale * (communicator.rank + 1))
                with torch.cuda.stream(stream):
                    graph.replay()
                torch.cuda.synchronize()
                # Ranks contribute 1 and 2 times `scale`.
                return torch.equal(out, torch.full_like(out, 3 * scale))

            assert replay(1.0)
            held = torch.cuda.mem_get_info(world.device)[0]
            distributed.suspend(world)
            released = torch.cuda.mem_get_info(world.device)[0] - held
            assert released > 0, "suspending released no device storage"
            distributed.resume(world)
            assert replay(2.0)
            graph.release()


def test_a_captured_collective_replays_after_the_world_resumes():
    if torch.cuda.device_count() < 2:
        pytest.fail("the suspension test needs two GPUs")
    with rendezvous() as listener:
        mp.spawn(_replay_after_resume, args=(listener,), nprocs=2, join=True)
