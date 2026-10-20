"""Quantized split experts preserve encodings and weight completed routes.

Source ranks route rows to dedicated expert ranks through split MegaMoE on
peer-mapped symmetric storage; eager steps and graph replays return the
exact routed equation, also when a source has no rows and only joins.
"""

import argparse
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from uniserve import cuda_graph
from uniserve.backends.experts import TorchBootstrap
from uniserve.backends.experts import create as create_transport
from uniserve.backends.experts.split import symmetric_bytes
from uniserve.bindings.context import ExecutionContext
from uniserve.distributed import Communicator, partition_experts
from uniserve.models import TextSize
from uniserve.nn.moe import FusedMoE
from uniserve.quantization import Quantizer

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def _peer_storage(nbytes: int, device: torch.device):
    """Map ``nbytes`` on every rank of the world and every rank's address.

    The split transport takes its peer-mapped storage from its caller; in
    serving the worker runtime maps it (``PeerStorage``). Here torch's NCCL
    symmetric memory maps it over the world group, whose ranks are the
    union's members in order. Collective: every rank calls it at the same
    point. Returns this rank's allocation, every rank's address of its
    allocation, and the rendezvous handle that keeps the mappings alive.
    """
    import torch.distributed._symmetric_memory as symmetric_memory

    if symmetric_memory.get_backend(device) != "NCCL":
        symmetric_memory.set_backend("NCCL")
    # The NCCL rendezvous uses the group's communicator, which torch
    # creates at its first collective.
    dist.all_reduce(torch.zeros(1, device=device))
    local = symmetric_memory.empty((nbytes,), dtype=torch.int8, device=device)
    handle = symmetric_memory.rendezvous(local, dist.group.WORLD)
    pointers = [
        handle.get_buffer(member, (nbytes,), torch.int8).data_ptr()
        for member in range(dist.get_world_size())
    ]
    return local, pointers, handle


@torch.inference_mode()
def _exercise(
    rank,
    local_rank,
    world_size,
    attention_ranks,
    init_method,
    format,
    width,
    intermediate,
):
    device = torch.device("cuda", local_rank)
    capacity, top_k = 32, 2
    expert_ranks = tuple(range(attention_ranks, world_size))
    experts = 2 * len(expert_ranks)
    source = rank < attention_ranks
    torch.cuda.set_device(device)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        union = Communicator(
            tuple(range(world_size)), rank, "experts", device, dist.group.WORLD
        )
        activation = (
            Quantizer(format, calibrated_scale=1.0)
            if format == "nvfp4"
            else Quantizer(format)
        )
        model = torch.nn.ModuleList(
            FusedMoE(
                experts,
                width,
                intermediate,
                top_k=top_k,
                activation="silu",
                device="meta",
                dtype=torch.bfloat16,
            )
            for _ in range(2)
        )
        if not source:
            partition_experts(
                model,
                Communicator(
                    expert_ranks, rank - attention_ranks, "experts", device
                ),
            )
        for index, layer in enumerate(model):
            layer.up_gate.input_quantizer = activation
            layer.down.input_quantizer = activation
            if source:
                continue
            resident = layer.up_gate.num_experts
            up_gate = torch.zeros(
                (resident, 2 * intermediate, width),
                device=device,
                dtype=torch.bfloat16,
            )
            down = torch.zeros(
                (resident, width, intermediate),
                device=device,
                dtype=torch.bfloat16,
            )
            rows = torch.arange(width, device=device)
            up_gate[:, rows, rows] = 1
            # x[0]=6 and x[1]=2 make every gate 32 (64 at layer 1),
            # where SiLU equals the identity in FP32. All intermediate
            # blocks and both weight encodings are exact powers of two.
            up_gate[:, intermediate:, :2] = 4 * 2**index
            for expert in range(resident):
                global_expert = layer.expert_slice.start + expert
                down[expert, rows, rows] = (global_expert % 2 + 1) / 8
            for linear, value in ((layer.up_gate, up_gate), (layer.down, down)):
                linear.weight = torch.nn.Parameter(
                    Quantizer(format).quantize(value), requires_grad=False
                )

        storage, pointers, _mapping = _peer_storage(
            symmetric_bytes(
                size=world_size,
                attention_ranks=attention_ranks,
                max_tokens=capacity,
                num_experts=experts,
                top_k=top_k,
                hidden=width,
                intermediate=intermediate,
                format=format,
            ),
            device,
        )
        exchange = create_transport(
            "megamoe",
            union,
            TorchBootstrap(dist.group.WORLD),
            max_tokens=capacity,
            top_k=top_k,
            num_experts=experts,
            hidden_size=width,
            intermediate_size=intermediate,
            activation="silu",
            attention_ranks=attention_ranks,
            device=device,
            encoding=format,
            symmetric={
                "storage": storage,
                "pointers": pointers,
            },
        )
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with ExecutionContext(
            model, stream=stream, experts=exchange
        ) as context:
            context.prepare(TextSize(capacity, 1))
            exchange.bind_microbatches([exchange])
            for empty in (False, True):
                count = 0 if not source or empty and rank == 0 else 5 + rank % 3
                magnitudes = torch.tensor(
                    (0.5, 1, 1.5, 2, 3, 4, 6), device=device
                )
                hidden = (
                    magnitudes[torch.arange(count * width, device=device) % 7]
                    .reshape(count, width)
                    .bfloat16()
                )
                hidden[:, ::16] = 6
                hidden[:, 1] = 2
                ids = torch.arange(
                    count * top_k, device=device, dtype=torch.int32
                ).reshape(count, top_k)
                ids.add_(2 * rank).remainder_(experts)
                weights = (
                    torch.tensor((0.3, 0.7), device=device)
                    .expand(count, top_k)
                    .contiguous()
                )

                def call():
                    with context.activate(), context.expert_step(capacity):
                        return (
                            torch.stack(
                                [layer(hidden, ids, weights) for layer in model]
                            )
                            if source
                            else hidden.expand(len(model), -1, -1)
                        )

                def compare(output):
                    # The exact routed equation; six BF16 roundoffs cover
                    # projections, remote weighted rows and combine, matching
                    # the distributed grouped-expert numerical contract.
                    expected = torch.stack(
                        [
                            hidden.double()
                            * (32 * 2**layer)
                            * (
                                weights[:, 0, None].double() / 8
                                + weights[:, 1, None].double() / 4
                            )
                            for layer in range(len(model))
                        ]
                    )
                    torch.testing.assert_close(
                        output.double(), expected, rtol=6 / 250, atol=0
                    )

                compare(call())
                torch.cuda.synchronize(device)
                graph = cuda_graph.capture(call, stream=stream)
                for _ in range(4):
                    with torch.cuda.stream(stream):
                        output = graph.replay()
                    stream.synchronize()
                    compare(output)
                graph.release()
        exchange.close()
    finally:
        dist.destroy_process_group()


def _local(rank, port, format, width, intermediate):
    _exercise(
        rank, rank, 2, 1, f"tcp://127.0.0.1:{port}", format, width, intermediate
    )


@pytest.mark.parametrize("format", ("mxfp8", "nvfp4"))
@pytest.mark.parametrize("width,intermediate", ((512, 512), (384, 768)))
def test_split_experts_preserve_quantization_and_routes(
    format, width, intermediate
):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    if any(
        torch.cuda.get_device_capability(i) not in {(10, 0), (10, 3)}
        for i in range(2)
    ):
        pytest.skip("requires two Blackwell CUDA devices")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(
        _local, args=(port, format, width, intermediate), nprocs=2, join=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-ranks", type=int, required=True)
    parser.add_argument("--format", choices=("mxfp8", "nvfp4"), required=True)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--intermediate", type=int, default=768)
    arguments = parser.parse_args()
    _exercise(
        int(os.environ["RANK"]),
        int(os.environ["LOCAL_RANK"]),
        int(os.environ["WORLD_SIZE"]),
        arguments.attention_ranks,
        "env://",
        arguments.format,
        arguments.width,
        arguments.intermediate,
    )
