"""Neural layers, parallel primitives and model operators.

Numerical formulas live in :mod:`uniserve_models.nn.functional`; the modules
exported here compose them with parameters, partitioning and bound execution
resources.
"""

from uniserve_models.nn.activation import GELUAndMul, SiLUAndMul, get_act_fn
from uniserve_models.nn.attention import Attention
from uniserve_models.nn.linear import (
    ColumnParallelLinear,
    Linear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelHead,
)
from uniserve_models.nn.mlp import GatedMLP
from uniserve_models.nn.modulation import Modulation
from uniserve_models.nn.moe import FusedMoE, TopK
from uniserve_models.nn.norm import RMSNorm
from uniserve_models.nn.rope import RotaryEmbedding
from uniserve_models.nn.routing import RoutedTensor, RouteSpan
from uniserve_models.nn.timestep import TimestepEmbedding
from uniserve_models.nn.vae.layers import (
    AttentionBlock,
    DiagonalGaussian,
    Downsample,
    ResidualBlock,
    Upsample,
)
from uniserve_models.nn.vae.patch import PatchAutoencoder, RGBDecoder
from uniserve_models.nn.vision import (
    MLPConnector,
    PatchEmbed,
    PositionEmbedding,
    TubeletEmbed,
)

__all__ = [
    "Attention",
    "AttentionBlock",
    "ColumnParallelLinear",
    "DiagonalGaussian",
    "Downsample",
    "FusedMoE",
    "GELUAndMul",
    "GatedMLP",
    "Linear",
    "MLPConnector",
    "MergedColumnParallelLinear",
    "Modulation",
    "PatchAutoencoder",
    "PatchEmbed",
    "PositionEmbedding",
    "QKVParallelLinear",
    "RGBDecoder",
    "RMSNorm",
    "ResidualBlock",
    "RotaryEmbedding",
    "RouteSpan",
    "RoutedTensor",
    "RowParallelLinear",
    "SiLUAndMul",
    "TimestepEmbedding",
    "TopK",
    "TubeletEmbed",
    "Upsample",
    "VocabParallelEmbedding",
    "VocabParallelHead",
    "get_act_fn",
]
