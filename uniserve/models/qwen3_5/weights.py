"""Qwen3.5 checkpoint assignments and numerical precision presets.

Checkpoint names follow the Transformers ``Qwen3_5ForConditionalGeneration``
layout in one checkpoint source, ``primary``, at the checkpoint root: the
language model under ``model.language_model.``, the head as
``lm_head.weight`` and the multi-token prediction layers under ``mtp.``.
The vision tower under ``model.visual.`` is not part of the model and stays
unloaded. Parameters of linear layers map to complete logical tensors, which
the loader narrows to the rank's partition; the gated delta parameters map
to the rank's channels and heads explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from uniserve.loading import checkpoint, weights
from uniserve.nn.gated_delta import GatedDeltaNet

from .config import Config

if TYPE_CHECKING:
    from .model import Model


checkpoint_sources = (checkpoint.Config(name="primary"),)

# The checkpoint name prefix of the language model and of the vision tower.
LANGUAGE = "model.language_model."
VISION = "model.visual."

# Named presets that load_model accepts. read_config starts from the first.
_PRECISIONS = MappingProxyType(
    {"bf16": weights.Config(), "fp16": weights.Config(dtype=torch.float16)}
)


def precisions(config: Config) -> Mapping[str, weights.Config]:
    """Named presets ``load_model`` accepts for ``config``'s checkpoint."""
    return _PRECISIONS


def checkpoint_precision(config: Config) -> weights.Config:
    """Base precision of a calibrated ModelOpt checkpoint."""
    return _PRECISIONS["bf16"]


def _layer(
    names: dict[str, str], config: Config, target: str, source: str, *, linear
) -> None:
    """Map one decoder layer's parameters, except its gated delta mixer's."""
    names[f"{target}.input_norm.weight"] = f"{source}.input_layernorm.weight"
    names[f"{target}.post_attention_norm.weight"] = (
        f"{source}.post_attention_layernorm.weight"
    )
    for branch in ("gate", "up"):
        names[f"{target}.mlp.gate_up.projections.{branch}.weight"] = (
            f"{source}.mlp.{branch}_proj.weight"
        )
    names[f"{target}.mlp.down.weight"] = f"{source}.mlp.down_proj.weight"
    if linear:
        mixer = f"{target}.linear_attention"
        for branch in ("z", "b", "a"):
            names[f"{mixer}.projection.projections.{branch}.weight"] = (
                f"{source}.linear_attn.in_proj_{branch}.weight"
            )
        names[f"{mixer}.norm.weight"] = f"{source}.linear_attn.norm.weight"
        names[f"{mixer}.output.weight"] = (
            f"{source}.linear_attn.out_proj.weight"
        )
        return
    fields = ("weight", "bias") if config.attention_bias else ("weight",)
    for field in fields:
        for branch in ("q", "k", "v"):
            names[f"{target}.attention.qkv.projections.{branch}.{field}"] = (
                f"{source}.self_attn.{branch}_proj.{field}"
            )
        names[f"{target}.attention.output.{field}"] = (
            f"{source}.self_attn.o_proj.{field}"
        )
    names[f"{target}.attention.query_norm.weight"] = (
        f"{source}.self_attn.q_norm.weight"
    )
    names[f"{target}.attention.key_norm.weight"] = (
        f"{source}.self_attn.k_norm.weight"
    )


def parameter_sources(config: Config) -> Mapping[str, str]:
    """Map the parameters of a complete ``Model`` to checkpoint names.

    Covers every parameter except the gated delta layers' fused input
    projection and mixer parameters, which ``mixer_sources`` maps.
    """
    names = {
        "backbone.embedding.weight": f"{LANGUAGE}embed_tokens.weight",
        "backbone.norm.weight": f"{LANGUAGE}norm.weight",
        "lm_head.weight": f"{LANGUAGE}embed_tokens.weight"
        if config.tie_word_embeddings
        else "lm_head.weight",
    }
    for index in range(config.num_hidden_layers):
        _layer(
            names,
            config,
            f"backbone.layers.{index}",
            f"{LANGUAGE}layers.{index}",
            linear=config.linear(index),
        )
    if config.mtp_num_hidden_layers:
        names["mtp.embedding_norm.weight"] = "mtp.pre_fc_norm_embedding.weight"
        names["mtp.hidden_norm.weight"] = "mtp.pre_fc_norm_hidden.weight"
        names["mtp.projection.weight"] = "mtp.fc.weight"
        names["mtp.norm.weight"] = "mtp.norm.weight"
        for index in range(config.mtp_num_hidden_layers):
            _layer(
                names,
                config,
                f"mtp.layers.{index}",
                f"mtp.layers.{index}",
                linear=False,
            )
    return names


def mixer_sources(config: Config) -> Mapping[str, str]:
    """Each gated delta layer's module path and its checkpoint prefix."""
    return {
        f"backbone.layers.{index}.linear_attention": (
            f"{LANGUAGE}layers.{index}.linear_attn"
        )
        for index in range(config.num_hidden_layers)
        if config.linear(index)
    }


def _mixer_assignments(module, prefix: str, resolve) -> list:
    """Assign one gated delta layer's fused projection and mixer.

    The checkpoint's ``in_proj_qkv`` rows hold the query, key and value
    channels in that order; each is a complete logical branch of the fused
    projection, which the loader narrows to the rank's heads. The
    convolution's channels and the per-head decay parameters are narrowed
    here to the rank's (``GatedDeltaNet.channel_slices``).
    """
    mixer: GatedDeltaNet = module.mixer
    keys = mixer.num_key_heads * mixer.key_dim
    values = mixer.num_value_heads * mixer.value_dim
    result = []
    fused = resolve(f"{prefix}.in_proj_qkv.weight")
    if fused is not None:
        branches = module.projection.projections
        for branch, rows in (
            ("q", slice(0, keys)),
            ("k", slice(keys, 2 * keys)),
            ("v", slice(2 * keys, 2 * keys + values)),
        ):
            result.append(
                weights.Assignment(
                    branches[branch].weight,
                    fused,
                    source_slice=(rows, slice(0, fused.shape[1])),
                )
            )
    conv = resolve(f"{prefix}.conv1d.weight")
    if conv is not None:
        start = 0
        for part in mixer.channel_slices():
            width = part.stop - part.start
            result.append(
                weights.Assignment(
                    mixer.conv_weight,
                    conv,
                    source_slice=(part, slice(0, 1), slice(0, mixer.kernel)),
                    target_slice=(
                        slice(start, start + width),
                        slice(0, 1),
                        slice(0, mixer.kernel),
                    ),
                )
            )
            start += width
    heads = slice(mixer.value_heads.start, mixer.value_heads.stop)
    for name, source in (("a_log", "A_log"), ("dt_bias", "dt_bias")):
        value = resolve(f"{prefix}.{source}")
        if value is not None:
            result.append(
                weights.Assignment(
                    getattr(mixer, name), value, source_slice=(heads,)
                )
            )
    return result


def checkpoint_mappings(model: Model) -> tuple[weights.ModuleMapping, ...]:
    """Declare how the resident parameters of ``model`` load from ``primary``.

    ``uniserve.models.registry.read_config`` calls this on the complete meta
    skeleton, and ``uniserve.loading.load_model`` calls it again after
    ``parallelize_`` has bound the pipeline stage, so ``model`` may lack
    layers, the embedding or the head; checkpoint tensors of those absent
    parameters are declared nonresident. The vision tower's tensors are
    nonresident on every rank.
    """
    config = model.config
    names = parameter_sources(config)
    mixers = mixer_sources(config)
    modules = dict(model.named_modules(remove_duplicate=False))
    parameters = dict(model.named_parameters(remove_duplicate=False))

    # Every checkpoint tensor of a gated delta layer whose module this rank
    # does not hold.
    mixer_tensors = (
        "in_proj_qkv.weight",
        "conv1d.weight",
        "A_log",
        "dt_bias",
    )
    off_stage = frozenset(
        source for target, source in names.items() if target not in parameters
    ) | frozenset(
        f"{prefix}.{tensor}"
        for path, prefix in mixers.items()
        if path not in modules
        for tensor in mixer_tensors
    )
    if config.tie_word_embeddings:
        off_stage |= {"lm_head.weight"}
    used = {names[target] for target in parameters if target in names}
    off_stage -= used

    def assign(reader):
        available = frozenset(reader.names())

        def resolve(source):
            return reader.get(source) if source in available else None

        result: list[weights.Assignment] = []
        for path, prefix in mixers.items():
            module = modules.get(path)
            if module is not None:
                result.extend(_mixer_assignments(module, prefix, resolve))
        for target, parameter in parameters.items():
            weight = resolve(names[target]) if target in names else None
            if weight is not None:
                result.append(weights.Assignment(parameter, weight))
        return tuple(result)

    return (
        weights.ModuleMapping(
            model,
            "primary",
            assign,
            frozenset(parameters),
            nonresident=off_stage,
            nonresident_prefixes=(VISION,),
        ),
    )
