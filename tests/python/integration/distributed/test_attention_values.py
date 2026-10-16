"""Public attention preserves token order, GQA replication and empty shards.

Replicated attention keeps its complete heads and tokens on every rank, and
an encoder its complete samples.
"""

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from tests.python.fixtures.distributed import (
    bind_mesh,
    process_world,
    rendezvous,
)
from uniserve import cuda_graph
from uniserve.bindings.context import ExecutionContext
from uniserve.distributed import DeviceMesh, communication_axes, parallelize_
from uniserve.models import Encoder, TextSize
from uniserve.nn.attention import (
    Attention,
    AttentionBatch,
    AttentionParallelConfig,
    DenseInput,
    PagedInput,
    Ulysses,
)
from uniserve.nn.vae import AttentionBlock
from uniserve.prefix_cache import Config, mha
from uniserve.prefix_cache.units import UnitPool

pytestmark = pytest.mark.integration


def _run(rank, listener):
    with process_world(rank, 4, listener) as world:
        mesh = bind_mesh(
            world,
            DeviceMesh(
                ranks=(3, 1, 0, 2), shape=(4,), axes=("tokens",), rank=rank
            ),
        )
        group = mesh.get_group("tokens")
        generator = torch.Generator().manual_seed(71)
        for count in (0, 2, 7):
            q = torch.randn(
                count, 8, 4, dtype=torch.float32, generator=generator
            )
            k, v = (
                torch.randn(
                    count, 2, 4, dtype=torch.float32, generator=generator
                )
                for _ in range(2)
            )
            layer = Attention(8, 2, 4, cache_name="attention")
            parallelize_(
                layer,
                mesh,
                attention=AttentionParallelConfig(heads=Ulysses("tokens")),
            )
            width = (count + 3) // 4
            start = min(count, group.rank * width)
            stop = min(count, start + width)
            config = Config(
                {
                    "attention": mha.Config(
                        2, 4, (group.rank // 2,), torch.float32
                    )
                }
            )
            cache = UnitPool.allocate(
                config, num_units=1, block_size=16, device="cpu"
            )
            with ExecutionContext(
                layer, cache=cache, attention="torch"
            ) as context:
                context.prepare(TextSize(count, 1))
                batch = AttentionBatch.single(
                    PagedInput.from_blocks(
                        blocks=((0,),),
                        query_lengths=(count,),
                        prefix_lengths=(0,),
                        block_size=16,
                        causal=True,
                        device="cpu",
                    )
                )
                context.bind_attention(batch)
                out = torch.empty_like(q[start:stop])
                actual = layer(
                    q[start:stop],
                    k[start:stop],
                    v[start:stop],
                    batch,
                    out=out,
                )
                expected = (
                    F.scaled_dot_product_attention(
                        q.transpose(0, 1).unsqueeze(0),
                        k.transpose(0, 1).unsqueeze(0),
                        v.transpose(0, 1).unsqueeze(0),
                        is_causal=True,
                        enable_gqa=True,
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                )
                assert actual is out
                torch.testing.assert_close(actual, expected[start:stop])
                key, value = cache.state("attention").read(
                    (0,), start=0, length=count
                )
                head = group.rank // 2
                torch.testing.assert_close(key, k[:, head : head + 1])
                torch.testing.assert_close(value, v[:, head : head + 1])


def test_ulysses_paged_attention_values():
    with rendezvous() as listener:
        mp.spawn(_run, args=(listener,), nprocs=4, join=True)


def _replicated(rank, listener):
    with process_world(rank, 2, listener) as world:
        # A component partitioned over two ranks along its channels, and one
        # partitioned along its tokens.
        layouts = (
            (("tp",), AttentionParallelConfig()),
            (("ulysses",), AttentionParallelConfig(heads=Ulysses())),
        )
        generator = torch.Generator().manual_seed(29)
        # [batch, channels, height, width] complete feature maps.
        pixels = torch.randn(2, 64, 5, 3, generator=generator)
        for axes, attention in layouts:
            mesh = bind_mesh(
                world,
                DeviceMesh(ranks=(1, 0), shape=(2,), axes=axes, rank=rank),
            )
            block = AttentionBlock(64)
            with torch.no_grad():
                for parameter in block.parameters():
                    parameter.copy_(
                        torch.randn(parameter.shape, generator=generator) / 8
                    )
            expected = block(pixels)

            # The single-head spatial attention communicates over no axis and
            # computes on each rank what one device computes.
            assert communication_axes(block, mesh, attention=attention) == ()
            parallelize_(block, mesh, attention=attention)
            torch.testing.assert_close(block(pixels), expected)


def test_replicated_attention_keeps_heads_and_tokens_on_every_rank():
    with rendezvous() as listener:
        mp.spawn(_replicated, args=(listener,), nprocs=2, join=True)


class _SelfAttention(nn.Module):
    """Two-head dense self-attention over ``[samples, tokens, 8]`` inputs."""

    def __init__(self):
        super().__init__()
        self.attention = Attention(2, 2, 4)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        count, tokens, _ = samples.shape
        # [samples, heads, tokens, head_dim], used as query, key and value.
        heads = samples.view(count, tokens, 2, 4).transpose(1, 2)
        attended = self.attention(
            heads,
            heads,
            heads,
            AttentionBatch.single(DenseInput(False, None)),
        )
        return attended.transpose(1, 2).reshape(count, tokens, 8)


def _encoder(rank, listener):
    with process_world(rank, 2, listener) as world:
        mesh = bind_mesh(
            world,
            DeviceMesh(ranks=(1, 0), shape=(2,), axes=("ulysses",), rank=rank),
        )
        attention = AttentionParallelConfig(heads=Ulysses())
        samples = torch.randn(
            3, 5, 8, generator=torch.Generator().manual_seed(31)
        )
        encoder = Encoder(_SelfAttention())
        expected = encoder.encode(samples)

        # An encoder bound as a component's own module, under the
        # component's token partition, still encodes complete samples.
        assert communication_axes(encoder, mesh, attention=attention) == ()
        parallelize_(encoder, mesh, attention=attention)
        for actual, wanted in zip(
            encoder.encode(samples), expected, strict=True
        ):
            torch.testing.assert_close(actual, wanted)


def test_encoder_keeps_complete_samples_under_token_partition():
    with rendezvous() as listener:
        mp.spawn(_encoder, args=(listener,), nprocs=2, join=True)


def _context(rank, listener, gpu):
    from uniserve.nn.attention import (
        ContextParallelConfig,
        SegmentedInput,
        SequenceLengths,
        VarlenInput,
    )

    device = torch.device("cuda", rank) if gpu else torch.device("cpu")
    if gpu:
        torch.cuda.set_device(device)
    dtype = torch.bfloat16 if gpu else torch.float32
    with process_world(rank, 4, listener, device) as world:
        cases = (
            (
                (4,),
                ("context",),
                AttentionParallelConfig(
                    context=ContextParallelConfig(gather_axis="context")
                ),
            ),
            (
                (2, 2),
                ("heads", "context"),
                AttentionParallelConfig(
                    heads=Ulysses("heads"),
                    context=ContextParallelConfig(gather_axis="context"),
                ),
            ),
        )
        for shape, axes, parallel in cases:
            mesh = bind_mesh(
                world,
                DeviceMesh(
                    ranks=(3, 1, 0, 2), shape=shape, axes=axes, rank=rank
                ),
                axes=(*((axis,) for axis in axes), axes),
            )
            tokens = mesh.get_group(axes)
            layer = Attention(8, 2, 64, cache_name="attention")
            parallelize_(layer, mesh, attention=parallel)
            heads = mesh.get_group(
                () if parallel.heads is None else parallel.heads.axis
            )
            first_head = heads.rank * (2 // heads.size)
            local_heads = tuple(range(first_head, first_head + 2 // heads.size))
            config = Config(
                {"attention": mha.Config(2, 64, local_heads, dtype)}
            )
            stream = torch.cuda.Stream(device=device) if gpu else None
            if stream is not None:
                stream.wait_stream(torch.cuda.current_stream(device))
            cache = UnitPool.allocate(
                config, num_units=3, block_size=16, device=device
            )
            with ExecutionContext(
                layer,
                cache=cache,
                stream=stream,
                attention="auto" if gpu else "torch",
            ) as context:
                context.prepare(TextSize(7, 3))
                for count in (0, 2, 7):
                    generator = torch.Generator().manual_seed(41 + count)
                    q = torch.randn(count, 8, 64, generator=generator).to(
                        device, dtype
                    )
                    k, v = (
                        torch.randn(count, 2, 64, generator=generator).to(
                            device, dtype
                        )
                        for _ in range(2)
                    )
                    width = (count + 3) // 4
                    interval = slice(
                        min(count, tokens.rank * width),
                        min(count, (tokens.rank + 1) * width),
                    )
                    counts = (count // 2, 0, count - count // 2)
                    lengths = SequenceLengths.from_lengths(
                        counts, device=device
                    )
                    batch = AttentionBatch.single(
                        VarlenInput(lengths, lengths, (True, False, False))
                    )
                    expected = []
                    for index, (query, key, value) in enumerate(
                        zip(
                            q.split(counts),
                            k.split(counts),
                            v.split(counts),
                            strict=True,
                        )
                    ):
                        expected.append(
                            F.scaled_dot_product_attention(
                                query.transpose(0, 1).unsqueeze(0),
                                key.transpose(0, 1).unsqueeze(0),
                                value.transpose(0, 1).unsqueeze(0),
                                is_causal=index == 0,
                                enable_gqa=True,
                            )
                            .squeeze(0)
                            .transpose(0, 1)
                        )
                    expected = torch.cat(expected)[interval]
                    actual = layer(q[interval], k[interval], v[interval], batch)
                    torch.testing.assert_close(actual, expected)
                    cached = AttentionBatch.single(
                        PagedInput.from_blocks(
                            blocks=((0,), (1,), (2,)),
                            query_lengths=counts,
                            prefix_lengths=(0, 0, 0),
                            block_size=16,
                            causal=(True, False, False),
                            device=device,
                        )
                    )
                    actual = layer(
                        q[interval], k[interval], v[interval], cached
                    )
                    torch.testing.assert_close(actual, expected)
                    offset = 0
                    for block, length in enumerate(counts):
                        key, value = cache.state("attention").read(
                            (block,), start=0, length=length
                        )
                        torch.testing.assert_close(
                            key, k[offset : offset + length, local_heads]
                        )
                        torch.testing.assert_close(
                            value, v[offset : offset + length, local_heads]
                        )
                        offset += length

                    if count != 7:
                        continue
                    prefix_counts = (2, 0, 1)
                    source_prefix_counts = prefix_counts
                    prefix_k, prefix_v = (
                        torch.randn(3, 2, 64, generator=generator).to(
                            device, dtype
                        )
                        for _ in range(2)
                    )
                    prefix_interval = slice(
                        min(3, tokens.rank), min(3, tokens.rank + 1)
                    )
                    # Two prefix tokens land in block 0 and one in block
                    # 2, at physical addresses 0, 1 and 32.
                    layer.update_cache(
                        prefix_k[prefix_interval],
                        prefix_v[prefix_interval],
                        AttentionBatch.single(
                            PagedInput.from_blocks(
                                blocks=((0,), (1,), (2,)),
                                query_lengths=(2, 0, 1),
                                prefix_lengths=(0, 0, 0),
                                block_size=16,
                                causal=True,
                                device=device,
                            )
                        ),
                    )
                    ends = torch.zeros(
                        (3, max(counts)), dtype=torch.int32, device=device
                    )
                    for row, length in enumerate(counts):
                        ends[row, :length] = (
                            torch.arange(length, device=device) + 1
                        )
                    segmented = SegmentedInput(
                        lengths,
                        SequenceLengths.from_lengths(
                            prefix_counts, device=device
                        ),
                        cached.entry(None).block_table,
                        None,
                        ends,
                        False,
                    )

                    def reference():
                        outputs = []
                        start = prefix_start = 0
                        for row, (length, prefix_count) in enumerate(
                            zip(counts, prefix_counts, strict=True)
                        ):
                            key = torch.cat(
                                (
                                    prefix_k[
                                        prefix_start : prefix_start
                                        + prefix_count
                                    ],
                                    k[start : start + length],
                                )
                            )
                            value = torch.cat(
                                (
                                    prefix_v[
                                        prefix_start : prefix_start
                                        + prefix_count
                                    ],
                                    v[start : start + length],
                                )
                            )
                            allowed = torch.arange(key.shape[0], device=device)[
                                None
                            ] < (prefix_count + ends[row, :length, None])
                            outputs.append(
                                F.scaled_dot_product_attention(
                                    q[start : start + length]
                                    .transpose(0, 1)
                                    .unsqueeze(0),
                                    key.transpose(0, 1).unsqueeze(0),
                                    value.transpose(0, 1).unsqueeze(0),
                                    attn_mask=allowed,
                                    enable_gqa=True,
                                )
                                .squeeze(0)
                                .transpose(0, 1)
                            )
                            start += length
                            prefix_start += source_prefix_counts[row]
                        return torch.cat(outputs)[interval]

                    def invoke():
                        return layer(
                            q[interval],
                            k[interval],
                            v[interval],
                            AttentionBatch.single(segmented),
                        )

                    torch.testing.assert_close(
                        invoke(),
                        reference(),
                        rtol=2e-2 if gpu else 1e-5,
                        atol=2e-2 if gpu else 1e-6,
                    )
                    if gpu:
                        graph = cuda_graph.capture(invoke, stream=stream)
                        q.mul_(0.7)
                        k.add_(0.25)
                        ends.sub_(1).clamp_min_(0)
                        prefix_counts = (1, 0, 1)
                        segmented.prefixes.values.copy_(
                            torch.tensor(
                                prefix_counts,
                                dtype=torch.int32,
                                device=device,
                            )
                        )
                        segmented.prefixes.offsets.copy_(
                            torch.tensor(
                                [0, 1, 1, 2],
                                dtype=torch.int32,
                                device=device,
                            )
                        )
                        from dataclasses import replace

                        context.bind_attention(
                            AttentionBatch.single(
                                replace(
                                    segmented,
                                    prefixes=SequenceLengths(
                                        host=prefix_counts,
                                        values=segmented.prefixes.values,
                                        offsets=segmented.prefixes.offsets,
                                    ),
                                )
                            )
                        )
                        torch.testing.assert_close(
                            graph.replay(),
                            reference(),
                            rtol=2e-2,
                            atol=2e-2,
                        )
                        torch.cuda.synchronize(device)
                        graph.release()
                    start = 0
                    for block, length in enumerate(source_prefix_counts):
                        key, value = cache.state("attention").read(
                            (block,), start=0, length=length
                        )
                        torch.testing.assert_close(
                            key,
                            prefix_k[start : start + length, local_heads],
                        )
                        torch.testing.assert_close(
                            value,
                            prefix_v[start : start + length, local_heads],
                        )
                        start += length


def test_context_gather_preserves_sequences_and_cache_values():
    with rendezvous() as listener:
        mp.spawn(
            _context,
            args=(listener, False),
            nprocs=4,
            join=True,
        )


@pytest.mark.gpu
def test_context_peer_and_column_gather_preserve_sequences_and_cache_values():
    with rendezvous() as listener:
        mp.spawn(
            _context,
            args=(listener, True),
            nprocs=4,
            join=True,
        )
