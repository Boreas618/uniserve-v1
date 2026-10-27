"""What distributions and partitioned parameters hold of logical tensors."""

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard

from uniserve.distributed import (
    Communicator,
    DeviceMesh,
    Distribution,
    parallelize_,
    partition_experts,
)
from uniserve.distributed.distribution import BlockShard, Cyclic, Strided
from uniserve.distributed.logical import Logical, logical_tensors
from uniserve.loading.retained import tensors
from uniserve.models import qwen3
from uniserve.nn.moe import FusedMoE

pytestmark = pytest.mark.unit


def _mesh(*shape, axes=None, rank=0):
    axes = axes or tuple(f"axis{index}" for index in range(len(shape)))
    members = 1
    for size in shape:
        members *= size
    return DeviceMesh(
        ranks=tuple(range(members)), shape=tuple(shape), axes=axes, rank=rank
    )


def _held(distribution, extent, rank):
    """``{local position: logical index}`` of one dimension at ``rank``."""
    (segments,) = distribution.segments(
        (extent,), distribution.mesh.coordinate(rank)
    )
    held = {}
    for segment in segments:
        for k, index in enumerate(segment.span.indices()):
            held[segment.local + k] = index
    return held


@pytest.mark.parametrize(
    "divisions,shape,expected",
    (
        # Even contiguous shares.
        ((Shard(0),), (4,), lambda r: list(range(3 * r, 3 * r + 3))),
        # Position mod N, in increasing order.
        ((Cyclic(0),), (4,), lambda r: list(range(r, 12, 4))),
        # Two named sub-blocks of 4 and 8, each split on its own.
        (
            (BlockShard(0, (("a", 4), ("b", 8))),),
            (4,),
            lambda r: [r, 4 + 2 * r, 5 + 2 * r],
        ),
        # Shards nested over two axes: the second divides the first's share.
        (
            (Shard(0), Shard(0)),
            (2, 2),
            lambda r: list(range(3 * r, 3 * r + 3)),
        ),
        # A replicated axis keeps everything.
        ((Replicate(),), (4,), lambda r: list(range(12))),
    ),
)
def test_divisions_resolve_to_the_indices_each_member_holds(
    divisions, shape, expected
):
    distribution = Distribution(_mesh(*shape), divisions)
    for rank in range(len(distribution.mesh.ranks)):
        held = _held(distribution, 12, rank)
        # Local positions run from zero without gaps, in the order given.
        assert sorted(held) == list(range(len(held)))
        assert [held[position] for position in sorted(held)] == expected(rank)


@pytest.mark.parametrize("members", (2, 4))
def test_a_compressed_index_lies_with_the_positions_it_stands_for(members):
    # Entry j stands for position 4j + 1, which lies on member
    # (4j + 1) mod `members` under position-mod-N.
    distribution = Distribution(_mesh(members), (Strided(0, 4, 1),))
    for rank in range(members):
        held = _held(distribution, 10, rank)
        expected = [j for j in range(10) if (4 * j + 1) % members == rank]
        assert [held[position] for position in sorted(held)] == expected


def _config(kv_heads):
    return qwen3.Config(
        vocab_size=50,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        head_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=128,
        attention_bias=False,
        tie_word_embeddings=False,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=(),
    )


def _covered(local, part):
    """The local positions ``part`` holds, and every block of it."""
    covered = torch.zeros(local.shape, dtype=torch.bool)
    blocks = []
    for spans, start in part.blocks():
        window = tuple(
            slice(first, first + span.count)
            for first, span in zip(start, spans, strict=True)
        )
        covered[window] = True
        blocks.append((window, spans))
    return covered, blocks


def _whole(kv_heads):
    """An unpartitioned model whose every element is distinct, its padding
    (a vocabulary's rows past its tokens) zero as loading leaves it.
    """  # noqa: D205
    torch.manual_seed(0)
    model = qwen3.Model(_config(kv_heads))
    declared = logical_tensors(model)
    with torch.no_grad():
        for _, tensor in tensors(model):
            tensor.copy_(torch.randn_like(tensor))
            part = declared.get(id(tensor)) or Logical.whole(
                tuple(tensor.shape)
            )
            covered, _ = _covered(tensor, part)
            tensor[~covered] = 0
    return model


def _tensor_parallel(model, size, rank, *, whole_experts=False):
    mesh = DeviceMesh(
        ranks=tuple(range(size)), shape=(size,), axes=("tp",), rank=rank
    )
    object.__setattr__(
        mesh, "_groups", {("tp",): Communicator(tuple(range(size)), rank, "tp")}
    )
    parallelize_(model, mesh, whole_experts=whole_experts)
    return model


def _tensor_parallel_experts(model, size, rank):
    return _tensor_parallel(model, size, rank, whole_experts=True)


def _expert_parallel(model, size, rank):
    partition_experts(model, Communicator(tuple(range(size)), rank, "ep"))
    return model


@pytest.mark.parametrize(
    "kv_heads,partition,size",
    (
        (2, _tensor_parallel, 2),
        # Fewer KV heads than ranks: each rank holds one head's copy.
        (1, _tensor_parallel, 2),
        (2, _tensor_parallel, 4),
        (2, _expert_parallel, 2),
        # Tensor-parallel attention with the group dividing whole experts.
        (2, _tensor_parallel_experts, 2),
        (2, _tensor_parallel_experts, 4),
    ),
)
def test_partitioned_parameters_hold_their_declared_parts(
    kv_heads, partition, size
):
    whole = _whole(kv_heads)
    logical = dict(tensors(whole))
    for rank in range(size):
        model = partition(_whole(kv_heads), size, rank)
        declared = logical_tensors(model)
        for identity, local in tensors(model):
            part = declared.get(id(local)) or Logical.whole(tuple(local.shape))
            reference = logical[identity]
            covered, blocks = _covered(local, part)
            for window, spans in blocks:
                indices = tuple(
                    slice(
                        span.start,
                        span.start + span.step * span.count,
                        span.step,
                    )
                    for span in spans
                )
                assert torch.equal(local[window], reference[indices]), identity
            # Positions no segment holds are padding: zeros.
            assert not local[~covered].any(), identity


def test_expert_parts_name_the_up_and_gate_rows_of_their_channels():
    model = _tensor_parallel(_whole(2), 2, 1)
    experts = next(m for m in model.modules() if isinstance(m, FusedMoE))
    part = logical_tensors(model)[id(experts.up_gate.weight)]
    # Rank 1 of 2 holds channels 4..8 of the up rows and of the gate rows.
    assert part.shape == (4, 16, 16)
    assert [
        (segment.span.start, segment.span.count, segment.local)
        for segment in part.segments[1]
    ] == [(4, 4, 0), (12, 4, 4)]


def test_a_tensor_parallel_group_dividing_experts_keeps_their_channels():
    model = _tensor_parallel_experts(_whole(2), 4, 2)
    experts = next(m for m in model.modules() if isinstance(m, FusedMoE))
    part = logical_tensors(model)[id(experts.up_gate.weight)]
    # Rank 2 of 4 holds experts 2 of the 4, every up and gate row whole.
    assert experts.expert_slice == slice(2, 3)
    assert experts.group is experts.expert_group
    assert part.shape == (4, 16, 16)
    assert [
        (segment.span.start, segment.span.count) for segment in part.segments[0]
    ] == [(2, 1)]
    assert [
        (segment.span.start, segment.span.count) for segment in part.segments[1]
    ] == [(0, 16)]
