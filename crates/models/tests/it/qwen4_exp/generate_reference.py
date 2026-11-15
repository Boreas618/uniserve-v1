"""Generates the fixtures of the qwen4-exp crate's CPU tests.

Two tiny checkpoints, each `model/config.json` and `model/model.safetensors`:

- `reference`: a two-layer model (a gated delta network layer, then an indexed
  attention layer that holds the PLE layer) in BF16 with NVFP4 routed experts,
  whose weights are the ones the float32 references of the kernel challenge
  generate, and `reference.safetensors`: the inputs those references generate
  and the outputs they compute, on the CPU.
- `consistency`: a four-layer model in FP32 with random weights and dense
  stacked experts, for the checks that compare the model with itself (a prefill
  against decode and verify steps); it has no reference vectors.

## Which reference code runs

The references are `refs/qwen-kernel-challenge/packages/<package>/reference.py`
(`gdn_moe_layer`, `qsa_moe_layer`, `embed_ple`, `lm_head`, `mtp_draft`) with
`_common/{layer_common,nvfp4}.py`, and four functions of vLLM's own test module
`refs/vllm/tests/models/qwen4_exp/test_qsa_reference.py`, taken by name from
its source because the module imports vLLM (`_qsa_mqa_paged_reference`,
`_qsa_relative_topk_reference`, `_qsa_sparse_paged_attention_reference`).

What is changed to run them on the CPU, and nothing else:

- every device string `"cuda:0"` and `f"cuda:{rank}"` is `"cpu"`;
- `driver.host_alloc` (page-locked memory through libcuda) is a stand-in that
  returns ordinary memory, and `torch.cuda.set_device` does nothing;
- the size constants at the top of each file are reduced (`CONSTANTS` below),
  and the literal `160` of `qsa_moe_layer`'s shared expert, which is the
  package's `EXPERT_DIM`, with them.

The references compute one tensor-parallel rank's shard at a time and sum the
ranks. A checkpoint holds whole tensors, so the script joins the four shards
along the axis the ranks split (heads, intermediate columns, vocabulary rows).
Three of the references' per-rank draws have no whole-model form and are made
equal across the ranks before the reference runs, as inputs of the reference
and not as changes to it:

- the routed experts' NVFP4 tensor scale, which a checkpoint states once per
  projection: every rank's experts are quantized again by the reference's own
  `nvfp4.quantize` with the largest rank's scale;
- the gated delta network's output-norm weight, one vector for every head;
- `A_log`, which a checkpoint stores in BF16: the reference reads the rounded
  values.

The n-gram table of `embed_ple` is BF16 in the reference and E4M3 with one
scale in a checkpoint: the fixture draws an E4M3 table and gives the reference
its decoded rows.

`qsa_moe_layer` keeps its caches, its index keys and its index queries in E4M3,
the challenge's deployment; vLLM keeps them in BF16 when a deployment sets no
cache type, which is what this crate computes. Where every visible group is
selected the two differ by the E4M3 rounding of the values, inside the
challenge's bounds, and the reference runs as it is. Where the indexer chooses
among more groups than a row keeps, a selection made on E4M3-rounded scores is
another selection, so that one case runs the reference with BF16 caches: its
input caches are cast to BF16 and `torch.float8_e4m3fn` reads as
`torch.bfloat16` for the call (`bf16_caches` below).

Run from the repository root with the repository's interpreter:

    .venv/bin/python crates/models/tests/it/qwen4_exp/generate_reference.py

The script is deterministic: every weight and input is drawn from a seeded CPU
generator, so a rerun on the same PyTorch build writes the same bytes. A CPU
generator's stream is not a CUDA generator's, so these are not the challenge
judge's vectors.
"""

import ast
import ctypes
import json
import math
import os
import pathlib
import re
import sys
import types

import torch
from safetensors.torch import save_file

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"

#: The widths every reduced reference shares.
HIDDEN = 128
STREAMS = 4
RANK = 16
EXPERTS = 8
TOP = 3
EXPERT_DIM = 16
RANKS = 4
ROWS = 4

COMMON = {
    "HIDDEN": HIDDEN,
    "HC_RANK": RANK,
    "HC_DOWN_ROWS": 32,
    "EXPERTS": EXPERTS,
    "TOP_K": TOP,
    "EXPERT_DIM": EXPERT_DIM,
}

#: The constants each reference file is reduced to.
CONSTANTS = {
    "layer_common": COMMON,
    "gdn_moe_layer": {
        **COMMON,
        "HEAD_DIM": 16,
        "KEY_HEADS": 1,
        "VALUE_HEADS": 3,
    },
    "qsa_moe_layer": {
        "Q_HEADS": 2,
        "HEAD_DIM": 32,
        "ROTARY": 8,
        "INDEX_HEADS": 2,
        "INDEX_DIM": 16,
        "BLOCK_TOPK": 8,
        "PAGE_TOKENS": 64,
        "RING_WIDTH": 16 + 12,
        "MAX_POSITION": 1024,
    },
    "embed_ple": {
        "HIDDEN": HIDDEN,
        "HC_RANK": RANK,
        "HC_DOWN_ROWS": 32,
        "VOCAB": 256,
        "EOS": 250,
        "MAX_REQS": 8,
        "TABLE_ROWS": 348,
        "HEAD_ROWS": 50,
    },
    "lm_head": {
        "HIDDEN": HIDDEN,
        "HC_RANK": RANK,
        "HC_DOWN_ROWS": 32,
        "VOCAB": 256,
    },
    "mtp_draft": {"HIDDEN": HIDDEN},
}

#: The literal of `qsa_moe_layer`'s shared expert that is its `EXPERT_DIM`.
LITERALS = {
    "qsa_moe_layer": [
        ("2 * 160, HIDDEN", f"2 * {EXPERT_DIM}, HIDDEN"),
        (
            "HIDDEN, 160, std=640**-0.5",
            f"HIDDEN, {EXPERT_DIM}, std={EXPERT_DIM * RANKS}**-0.5",
        ),
    ],
}

#: The static activation scales of the references' NVFP4 experts.
A13 = 8.0 / (6.0 * 448.0)
A2 = 16.0 / (6.0 * 448.0)


def references():
    """The directory that holds the reference trees."""
    candidates = [
        os.environ.get("UNISERVE_REFS"),
        HERE.parents[4] / "refs",
        "/workspace/dev/uniserve-v1/refs",
    ]
    for candidate in candidates:
        if (
            candidate
            and pathlib.Path(candidate, "qwen-kernel-challenge").is_dir()
        ):
            return pathlib.Path(candidate)
    raise SystemExit(
        "no refs/qwen-kernel-challenge: set UNISERVE_REFS to the references"
    )


REFS = references()
PACKAGES = REFS / "qwen-kernel-challenge" / "packages"

# The stand-ins of what needs a device: page-locked memory, the current device.
_held = []


def _host_alloc(size):
    buffer = (ctypes.c_ubyte * size)()
    _held.append(buffer)
    return ctypes.addressof(buffer)


sys.modules["driver"] = types.SimpleNamespace(host_alloc=_host_alloc)
torch.cuda.set_device = lambda device: None
sys.path.insert(0, str(PACKAGES / "_common"))


def reference(package, path=None):
    """One reference file as a module, its device strings and sizes replaced."""
    path = path or PACKAGES / package / "reference.py"
    source = re.sub(r'f?"cuda:[^"]*"', '"cpu"', path.read_text())
    for name, value in CONSTANTS[package].items():
        source, count = re.subn(
            rf"^{name} = .*$", f"{name} = {value!r}", source, flags=re.M
        )
        assert count == 1, (package, name)
    for old, new in LITERALS.get(package, []):
        assert old in source, (package, old)
        source = source.replace(old, new)

    module = types.ModuleType(package)
    sys.modules[package] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


layer_common = reference(
    "layer_common", PACKAGES / "_common" / "layer_common.py"
)
import nvfp4  # noqa: E402

gdn = reference("gdn_moe_layer")
qsa = reference("qsa_moe_layer")
ple = reference("embed_ple")
head = reference("lm_head")
mtp = reference("mtp_draft")


def vllm_functions(names):
    """The named functions of vLLM's QSA test module, from its source text."""
    path = (
        REFS
        / "vllm"
        / "tests"
        / "models"
        / "qwen4_exp"
        / "test_qsa_reference.py"
    )
    tree = ast.parse(path.read_text())
    kept = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(kept) == len(names), [node.name for node in kept]
    namespace = {"torch": torch, "math": math}
    exec(
        compile(ast.Module(body=kept, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return [namespace[name] for name in names]


def bf16(x):
    return x.to(torch.bfloat16)


# ---- checkpoint tensors from the references' rank shards --------------------


def joined(weights, name, rows=slice(None), dim=0):
    """The ranks' shards of `name`, rows `rows` of each, joined along `dim`."""
    return torch.cat([w[name][rows] for w in weights], dim=dim)


def share_tensor_scales(weights):
    """Quantize every rank's experts by one tensor scale per projection."""
    for values, scales, tensor_scale in (
        ("w13", "w13_sf", "w13_gs"),
        ("w2", "w2_sf", "w2_gs"),
    ):
        common = max(float(w[tensor_scale][0]) for w in weights)
        for w in weights:
            dense = nvfp4.dequantize(
                w[values], w[scales], float(w[tensor_scale][0])
            )
            w[values], w[scales] = nvfp4.quantize(dense, common)
            w[tensor_scale] = torch.full_like(w[tensor_scale], common)


def hyper_connection(prefix, hc):
    return {
        f"{prefix}.hc_norm.weight": hc["norm"],
        f"{prefix}.input_mix_weight_down.weight": hc["down"][:RANK],
        f"{prefix}.block_inject_weight.weight": hc["down"][
            RANK : RANK + STREAMS
        ],
        f"{prefix}.input_mix_weight_up.weight": hc["up"],
    }


def experts(prefix, weights):
    """A layer's router, NVFP4 experts and shared expert, as a checkpoint's."""
    tensors = {
        f"{prefix}.gate.weight": weights[0]["gate"],
        f"{prefix}.shared_expert_gate.weight": weights[0]["shared_expert_gate"],
        f"{prefix}.shared_expert.gate_proj.weight": joined(
            weights, "shared_gate_up", slice(0, EXPERT_DIM)
        ),
        f"{prefix}.shared_expert.up_proj.weight": joined(
            weights, "shared_gate_up", slice(EXPERT_DIM, None)
        ),
        f"{prefix}.shared_expert.down_proj.weight": joined(
            weights, "shared_down", dim=1
        ),
    }
    scale13, scale2 = (
        float(weights[0]["w13_gs"][0]),
        float(weights[0]["w2_gs"][0]),
    )
    for expert in range(EXPERTS):
        gate, up = slice(0, EXPERT_DIM), slice(EXPERT_DIM, None)
        fields = {
            "gate_proj": (
                torch.cat([w["w13"][expert, gate] for w in weights]),
                torch.cat([w["w13_sf"][expert, gate] for w in weights]),
                scale13,
                A13,
            ),
            "up_proj": (
                torch.cat([w["w13"][expert, up] for w in weights]),
                torch.cat([w["w13_sf"][expert, up] for w in weights]),
                scale13,
                A13,
            ),
            "down_proj": (
                torch.cat([w["w2"][expert] for w in weights], dim=1),
                torch.cat([w["w2_sf"][expert] for w in weights], dim=1),
                scale2,
                A2,
            ),
        }
        for name, (values, scales, tensor_scale, input_scale) in fields.items():
            module = f"{prefix}.experts.{expert}.{name}"
            tensors[f"{module}.weight"] = values
            tensors[f"{module}.weight_scale"] = scales
            tensors[f"{module}.weight_scale_2"] = torch.tensor(
                tensor_scale, dtype=torch.float32
            )
            tensors[f"{module}.input_scale"] = torch.tensor(
                input_scale, dtype=torch.float32
            )
    return tensors


def linear_layer(prefix, weights):
    """A gated delta network layer: the ranks' heads in `[q | k | v]` order."""
    key, value = gdn.KEY_HEADS * gdn.HEAD_DIM, gdn.VALUE_HEADS * gdn.HEAD_DIM
    q, k, v, z = (
        slice(0, key),
        slice(key, 2 * key),
        slice(2 * key, 2 * key + value),
        slice(2 * key + value, None),
    )
    b, a = slice(0, gdn.VALUE_HEADS), slice(gdn.VALUE_HEADS, None)
    net = f"{prefix}.linear_attn"
    tensors = {
        f"{net}.in_proj_qkv.weight": torch.cat(
            [joined(weights, "in_proj_qkvz", part) for part in (q, k, v)]
        ),
        f"{net}.in_proj_z.weight": joined(weights, "in_proj_qkvz", z),
        f"{net}.in_proj_b.weight": joined(weights, "in_proj_ba", b),
        f"{net}.in_proj_a.weight": joined(weights, "in_proj_ba", a),
        f"{net}.conv1d.weight": torch.cat(
            [joined(weights, "conv1d", part) for part in (q, k, v)]
        ).unsqueeze(1),
        f"{net}.A_log": bf16(joined(weights, "A_log")),
        f"{net}.dt_bias": joined(weights, "dt_bias"),
        f"{net}.norm.weight": weights[0]["norm"],
        f"{net}.out_proj.weight": joined(weights, "out_proj", dim=1),
    }
    tensors |= hyper_connection(
        f"{prefix}.attn_hyper_connection", weights[0]["attn_hc"]
    )
    tensors |= hyper_connection(
        f"{prefix}.mlp_hyper_connection", weights[0]["mlp_hc"]
    )
    return tensors | experts(f"{prefix}.mlp", weights)


def linear_state(conv, ssm):
    """The ranks' windows and states as the whole model's, in `[q | k | v]`."""
    key = gdn.KEY_HEADS * gdn.HEAD_DIM
    parts = (slice(0, key), slice(key, 2 * key), slice(2 * key, None))
    window = torch.cat(
        [
            torch.cat([rank[..., part] for rank in conv], dim=-1)
            for part in parts
        ],
        dim=-1,
    )
    return window, torch.cat(ssm, dim=1)


def attention_layer(prefix, weights):
    """An indexed attention layer: the query heads, the two key/value heads."""
    head_dim = qsa.HEAD_DIM
    attn = f"{prefix}.self_attn"
    key_value = [weights[0]["kv_proj"], weights[2]["kv_proj"]]
    tensors = {
        f"{attn}.q_proj.weight": joined(weights, "q_proj"),
        f"{attn}.k_proj.weight": torch.cat(
            [head[:head_dim] for head in key_value]
        ),
        f"{attn}.v_proj.weight": torch.cat(
            [head[head_dim:] for head in key_value]
        ),
        f"{attn}.q_norm.weight": weights[0]["q_norm"],
        f"{attn}.k_norm.weight": weights[0]["k_norm"],
        f"{attn}.o_proj.weight": joined(weights, "o_proj", dim=1),
        f"{attn}.indexer.index_qk_proj.weight": weights[0]["index_proj"],
        f"{attn}.indexer.q_layernorm.weight": weights[0]["index_q_norm"],
        f"{attn}.indexer.k_layernorm.weight": weights[0]["index_k_norm"],
    }
    tensors |= hyper_connection(
        f"{prefix}.attn_hyper_connection", weights[0]["attn_hc"]
    )
    tensors |= hyper_connection(
        f"{prefix}.mlp_hyper_connection", weights[0]["mlp_hc"]
    )
    return tensors | experts(f"{prefix}.mlp", weights)


def attention_inputs(data):
    """A QSA step's caches and ring as the model's planes and state fields."""
    ring = data["ring"]
    positions = ring[:, :, qsa.INDEX_DIM :].contiguous().view(torch.int64)
    return {
        "hidden": data["hidden"],
        "block_output": data["block_output"],
        "injection": data["injection"][:, RANK : RANK + STREAMS].contiguous(),
        "start": data["start"].to(torch.int32),
        "page_table": data["page_table"],
        "key": bf16(
            torch.stack(
                [cache[:, :, 0].float() for cache in data["kv_cache"]], dim=2
            )
        ),
        "value": bf16(
            torch.stack(
                [cache[:, :, 1].float() for cache in data["kv_cache"]], dim=2
            )
        ),
        "index_keys": bf16(data["compressed"].float()),
        "ring_keys": ring[:, :, : qsa.INDEX_DIM].contiguous(),
        "ring_positions": positions.to(torch.int32),
    }


def attention_outputs(out):
    ring = out["ring"]
    return {
        "out.hidden": out["hidden"],
        "out.block_output": out["block_output"],
        "out.injection": out["injection"],
        "out.key": bf16(
            torch.stack(
                [cache[:, :, 0].float() for cache in out["kv_cache"]], dim=2
            )
        ),
        "out.value": bf16(
            torch.stack(
                [cache[:, :, 1].float() for cache in out["kv_cache"]], dim=2
            )
        ),
        "out.index_keys": bf16(out["compressed"].float()),
        "out.ring_keys": ring[:, :, : qsa.INDEX_DIM].contiguous(),
    }


def named(prefix, tensors):
    return {f"{prefix}.{name}": tensor for name, tensor in tensors.items()}


def bf16_caches(data, weights):
    """`qsa_moe_layer.ref_kernel` with its caches and index keys in BF16."""
    data["kv_cache"] = [bf16(cache.float()) for cache in data["kv_cache"]]
    data["compressed"] = bf16(data["compressed"].float())
    stored, torch.float8_e4m3fn = torch.float8_e4m3fn, torch.bfloat16
    try:
        return qsa.ref_kernel(data, weights)
    finally:
        torch.float8_e4m3fn = stored


# ---- the reference fixture ---------------------------------------------------


def reference_fixture():
    model, vectors = {}, {}
    language = "model.language_model"

    # One gated delta network layer with its experts (`gdn_moe_layer`).
    weights = gdn.generate_weights()
    share_tensor_scales(weights)
    for w in weights:
        w["norm"] = weights[0]["norm"]
        w["A_log"] = bf16(w["A_log"]).float()
    model |= linear_layer(f"{language}.layers.0", weights)

    for index, case in enumerate(
        [
            {"batch": 1, "accepted": 4},
            {"batch": 2, "accepted": 2},
            {"batch": 3, "accepted": 1},
        ]
    ):
        data = gdn.generate_input(case, 1)
        out = gdn.ref_kernel(data, weights)
        conv, ssm = linear_state(data["conv_state"], data["ssm_state"])
        conv_out, ssm_out = linear_state(out["conv_state"], out["ssm_state"])
        vectors |= named(
            f"gdn.{index}",
            {
                "hidden": data["hidden"],
                "block_output": data["block_output"],
                "injection": data["injection"][
                    :, RANK : RANK + STREAMS
                ].contiguous(),
                "state_indices": data["state_indices"],
                "num_accepted": data["num_accepted"],
                "conv_state": conv,
                "ssm_state": ssm,
                "out.hidden": out["hidden"],
                "out.block_output": out["block_output"],
                "out.injection": out["injection"],
                "out.conv_state": conv_out,
                "out.ssm_state": ssm_out,
            },
        )

    # One indexed attention layer with its experts (`qsa_moe_layer`): every
    # visible group selected, and more groups than the indexer keeps.
    weights = qsa.generate_weights()
    share_tensor_scales(weights)
    model |= attention_layer(f"{language}.layers.1", weights)

    for index, case in enumerate(
        [
            {"batch": 1, "ctx": 20},
            {"batch": 3, "ctx": 24},
            {"batch": 2, "ctx": 100},
        ]
    ):
        data = qsa.generate_input(case, 1)
        # Past the budget of 32 tokens the indexer chooses: BF16 caches there.
        out = (
            bf16_caches(data, weights)
            if case["ctx"] > qsa.BLOCK_TOPK * qsa.COMPRESS
            else qsa.ref_kernel(data, weights)
        )
        vectors |= named(
            f"qsa.{index}", attention_inputs(data) | attention_outputs(out)
        )

    # The embedding, the n-gram ids and rows, and the PLE layer (`embed_ple`).
    weights = ple.generate_weights()
    generator = torch.Generator().manual_seed(4242)
    table_rows = ple.TABLE_ROWS * RANKS
    codes = torch.randint(
        0,
        256,
        (table_rows, ple.HEAD_DIM),
        generator=generator,
        dtype=torch.uint8,
    )
    # No code of the table is the format's NaN.
    codes = torch.where(codes & 0x7F == 0x7F, codes & 0xFE, codes)
    table = codes.view(torch.float8_e4m3fn)
    scale = bf16(torch.tensor([0.0625]))
    decoded = bf16(bf16(table.float()).float() * scale.float())
    for rank, w in enumerate(weights):
        w["ple_table"] = decoded[
            rank * ple.TABLE_ROWS : (rank + 1) * ple.TABLE_ROWS
        ]

    shared = weights[0]
    layer = f"{language}.layers.1.ple"
    model |= {
        f"{language}.embed_tokens.weight": joined(weights, "embed"),
        f"{layer}.key_proj.weight": shared["key_proj"],
        f"{layer}.value_proj.weight": shared["value_proj"],
        f"{layer}.norm_key.weight": shared["norm_key"],
        f"{layer}.norm_query.weight": shared["norm_query"],
        f"{layer}.norm_conv.weight": shared["norm_conv"],
        f"{layer}.conv1d.weight": shared["conv1d"],
        f"{layer}.ple_embedding.layer_multipliers": shared["multipliers"],
        f"{layer}.ple_embedding.ngram_heads_vocab_sizes": shared["head_sizes"],
        f"{layer}.ple_embedding.ngram_heads_offsets": shared["head_offsets"],
        f"{layer}.ple_embedding.ngram_embedding.weight_scale": scale,
    }
    for shard in range(RANKS):
        rows = table[shard * ple.TABLE_ROWS : (shard + 1) * ple.TABLE_ROWS]
        model[f"{layer}.ple_embedding.ngram_embedding.shard_{shard}.weight"] = (
            rows.contiguous()
        )

    reference_weights = {
        k: v for k, v in shared.items() if k not in ("ple_table", "embed")
    }
    cases = [
        {"batch": 1, "accepted": 4},
        {"batch": 2, "accepted": 2, "eos": 0.25},
        {"batch": 3, "accepted": 1, "eos": 0.05},
    ]
    for index, case in enumerate(cases):
        data = ple.generate_input(case, 1)
        out = ple.ref_kernel(data, weights)
        batch = data["batch"]
        vectors |= named(
            f"ple.{index}",
            {
                "input_ids": data["input_ids"],
                "ngram_context": data["ngram_context"][:batch].contiguous(),
                "block_output": data["block_output"],
                "injection": data["injection"][
                    :, RANK : RANK + STREAMS
                ].contiguous(),
                "state_indices": data["state_indices"],
                "num_accepted": data["num_accepted"],
                "conv_state": data["conv_state"],
                "ids": ple.ngram_ids(data, reference_weights),
                "out.streams": out["streams"],
                "out.hidden": out["hidden"],
                "out.conv_state": out["conv_state"],
            },
        )

    # The read-out of the streams and the vocabulary head (`lm_head`).
    weights = head.generate_weights()
    mixer = f"{language}.hyper_connection_mixer"
    model |= {
        f"{mixer}.hc_norm.weight": weights[0]["hc_norm"],
        f"{mixer}.input_mix_weight_down.weight": weights[0]["mix_down"],
        f"{mixer}.input_mix_weight_up.weight": weights[0]["mix_up"],
        "lm_head.weight": joined(weights, "lm_head"),
    }
    lm_weights = weights
    data = head.generate_input({"batch": 3, "shuffle": True}, 1)
    out = head.ref_kernel(data, weights)
    vectors |= named(
        "head",
        {
            "hidden": data["hidden"],
            "block_output": data["block_output"],
            "injection": data["injection"][
                :, RANK : RANK + STREAMS
            ].contiguous(),
            "logits_indices": data["logits_indices"].to(torch.int32),
            "out.hidden": out["hidden"],
            "out.logits": out["logits"],
        },
    )

    # The drafter: the input fusion (`mtp_draft.fuse`), an indexed attention
    # layer and the read-out, composed of the references above with weights
    # of another seed.
    generator = torch.Generator().manual_seed(4343)
    width = STREAMS * HIDDEN

    def draw(*shape, std):
        return bf16(torch.randn(shape, generator=generator) * std)

    fusion = {
        "embed": model[f"{language}.embed_tokens.weight"],
        "norm_embedding": draw(HIDDEN, std=0.1),
        "fc_embedding": draw(HIDDEN, HIDDEN, std=HIDDEN**-0.5),
        "norm_hidden": draw(width, std=0.1),
        "fc_hidden": draw(HIDDEN, HIDDEN, std=HIDDEN**-0.5),
    }
    weights = qsa.generate_weights(3840)
    share_tensor_scales(weights)
    mixer_weights = head.generate_weights(3840)[0]
    draft_mixer = "mtp.hyper_connection_mixer"
    model |= attention_layer("mtp.layers.0", weights)
    model |= {
        "mtp.pre_fc_norm_embedding.weight": fusion["norm_embedding"],
        "mtp.pre_fc_norm_hidden.weight": fusion["norm_hidden"],
        "mtp.fc_embedding.weight": fusion["fc_embedding"],
        "mtp.fc_hidden.weight": fusion["fc_hidden"],
        f"{draft_mixer}.hc_norm.weight": mixer_weights["hc_norm"],
        f"{draft_mixer}.input_mix_weight_down.weight": mixer_weights[
            "mix_down"
        ],
        f"{draft_mixer}.input_mix_weight_up.weight": mixer_weights["mix_up"],
    }

    data = qsa.generate_input({"batch": 2, "ctx": 20}, 2)
    tokens = torch.randint(
        0, 250, (data["hidden"].shape[0],), generator=generator
    )
    streams = draw(tokens.shape[0], width, std=1.0)
    fused = mtp.fuse(tokens, streams.float(), fusion)
    data["hidden"] = bf16(fused)
    data["block_output"] = torch.zeros_like(data["block_output"])
    out = qsa.ref_kernel(data, weights)
    logits = data["injection"][:, RANK : RANK + STREAMS]
    drafted, normed = head._combine_norm(
        out["hidden"],
        out["block_output"],
        out["injection"],
        mixer_weights["hc_norm"],
    )
    sample = head._mix(normed, mixer_weights)
    vectors |= named("mtp", attention_inputs(data) | attention_outputs(out))
    vectors |= named(
        "mtp",
        {
            "tokens": tokens.to(torch.int32),
            "streams": streams,
            "out.fused": bf16(fused),
            "out.streams": bf16(drafted.flatten(1)),
            "out.logits": torch.cat(
                [bf16(sample @ w["lm_head"].float().T) for w in lm_weights],
                dim=1,
            ),
        },
    )
    del logits

    # vLLM's own references of the indexer's scores, its top-k and the attention
    # over a selection, in FP32, over inputs of their own generator.
    generator = torch.Generator().manual_seed(4545)
    scores_of, top_of, attend = vllm_functions(
        [
            "_qsa_mqa_paged_reference",
            "_qsa_relative_topk_reference",
            "_qsa_sparse_paged_attention_reference",
        ]
    )
    rows, heads, width, groups, pages = 6, 2, 16, 16, 5
    query = torch.randn(rows, heads, width, generator=generator)
    keys = torch.randn(pages, groups, 1, width, generator=generator)
    page_table = torch.tensor([[3, 1], [0, 4]], dtype=torch.int32)
    token_to_req = torch.tensor([0, 0, 1, 1, 1, 0], dtype=torch.int32)
    visible = torch.tensor([0, 5, 16, 23, 32, 9], dtype=torch.int32)
    scores = scores_of(
        query, keys, page_table, token_to_req, visible
    ) * math.sqrt(width)
    best = top_of(scores, torch.zeros_like(visible), visible, 8)

    q_heads, kv_heads, head_dim, page = 4, 2, 8, 16
    queries = torch.randn(rows, q_heads, head_dim, generator=generator)
    k_cache = torch.randn(pages, page, kv_heads, head_dim, generator=generator)
    v_cache = torch.randn(pages, page, kv_heads, head_dim, generator=generator)
    selection = torch.full((rows, 11), -1, dtype=torch.int32)
    for row, count in enumerate([0, 3, 11, 7, 1, 9]):
        selection[row, :count] = torch.randperm(2 * page, generator=generator)[
            :count
        ].to(torch.int32)
    scale = head_dim**-0.5
    attended = attend(
        queries, k_cache, v_cache, selection, page_table, token_to_req, scale
    )
    vectors |= named(
        "vllm",
        {
            "index_query": query,
            "index_keys": keys[:, :, 0].contiguous(),
            "page_table": page_table,
            "rows": token_to_req,
            "visible": visible,
            "out.scores": scores,
            "out.best": best,
            "query": queries,
            "key": k_cache,
            "value": v_cache,
            "selection": selection,
            "out.attended": attended,
        },
    )

    config = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "image_token_id": 252,
        "video_token_id": 253,
        "tie_word_embeddings": False,
        "text_config": {
            "attention_bias": False,
            "eos_token_id": ple.EOS,
            "hc_count": STREAMS,
            "hc_lowrank": RANK,
            "head_dim": qsa.HEAD_DIM,
            "heads_per_ngram": ple.HEADS_PER_NGRAM,
            "hidden_act": "silu",
            "hidden_size": HIDDEN,
            "indexer_budget": qsa.BLOCK_TOPK * qsa.COMPRESS,
            "indexer_compress_ratio": qsa.COMPRESS,
            "indexer_head_dim": qsa.INDEX_DIM,
            "indexer_kv_heads": 1,
            "indexer_n_heads": qsa.INDEX_HEADS,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_conv_kernel_dim": gdn.CONV_WIDTH,
            "linear_key_head_dim": gdn.HEAD_DIM,
            "linear_num_key_heads": gdn.KEY_HEADS * RANKS,
            "linear_num_value_heads": gdn.VALUE_HEADS * RANKS,
            "linear_value_head_dim": gdn.HEAD_DIM,
            "make_ngram_vocab_size_divisible_by": 16,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": qsa.MAX_POSITION,
            "moe_intermediate_size": EXPERT_DIM * RANKS,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "ngram_size": ple.NGRAM,
            "ngram_vocab_size_base": ple.HEAD_ROWS,
            "num_attention_heads": qsa.Q_HEADS * RANKS,
            "num_experts": EXPERTS,
            "num_experts_per_tok": TOP,
            "num_hidden_layers": 2,
            "num_key_value_heads": qsa.KV_HEADS,
            "output_gate_type": "sigmoid",
            "ple_conv_kernel_size": ple.CONV_TAPS,
            "ple_embed_dim": HIDDEN,
            "ple_layer_ids": [2],
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [2, 1, 1],
                "partial_rotary_factor": qsa.ROTARY / qsa.HEAD_DIM,
                "rope_theta": qsa.ROPE_THETA,
                "rope_type": "default",
            },
            "seed": ple.HASH_SEED,
            "shared_expert_intermediate_size": EXPERT_DIM * RANKS,
            "split_ngram_parts": RANKS,
            "vocab_size": ple.VOCAB,
        },
        "vision_config": VISION,
        # The routed experts' NVFP4 weights, declared as a ModelOpt export
        # declares them.
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
        },
    }
    write("reference", config, model, vectors)


#: A vision tower's configuration; the fixtures hold no tower and serve text.
VISION = {
    "deepstack_visual_indexes": [],
    "depth": 1,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 8,
    "in_channels": 3,
    "intermediate_size": 16,
    "num_heads": 2,
    "num_position_embeddings": 16,
    "out_hidden_size": HIDDEN,
    "patch_size": 2,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}


# ---- the consistency fixture -------------------------------------------------


def consistency_fixture():
    generator = torch.Generator().manual_seed(4444)
    hidden, streams, rank = 32, 4, 8
    width = streams * hidden
    heads, kv_heads, head_dim = 4, 2, 32
    index_heads, index_dim = 2, 16
    key_heads, value_heads, linear_dim, taps = 2, 4, 8, 4
    experts_count, top, intermediate = 4, 2, 16
    vocab, eos = 64, 60
    ngram_heads, base, parts = 16, 10, 2
    layers = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]

    def draw(*shape, std):
        return torch.randn(shape, generator=generator) * std

    def connection(prefix, injects=True):
        tensors = {
            f"{prefix}.hc_norm.weight": draw(width, std=0.1),
            f"{prefix}.input_mix_weight_down.weight": draw(
                rank, width, std=2 * width**-0.5
            ),
            f"{prefix}.input_mix_weight_up.weight": draw(
                width, rank, std=rank**-0.5
            ),
        }
        if injects:
            tensors[f"{prefix}.block_inject_weight.weight"] = draw(
                streams, width, std=2 * width**-0.5
            )
        return tensors

    def feed_forward(prefix):
        return {
            f"{prefix}.gate.weight": draw(
                experts_count, hidden, std=2 * hidden**-0.5
            ),
            f"{prefix}.experts.gate_up_proj": draw(
                experts_count, 2 * intermediate, hidden, std=hidden**-0.5
            ),
            f"{prefix}.experts.down_proj": draw(
                experts_count, hidden, intermediate, std=intermediate**-0.5
            ),
            f"{prefix}.shared_expert.gate_proj.weight": draw(
                intermediate, hidden, std=hidden**-0.5
            ),
            f"{prefix}.shared_expert.up_proj.weight": draw(
                intermediate, hidden, std=hidden**-0.5
            ),
            f"{prefix}.shared_expert.down_proj.weight": draw(
                hidden, intermediate, std=intermediate**-0.5
            ),
            f"{prefix}.shared_expert_gate.weight": draw(
                1, hidden, std=hidden**-0.5
            ),
        }

    def attention(prefix):
        return {
            f"{prefix}.q_proj.weight": draw(
                2 * heads * head_dim, hidden, std=hidden**-0.5
            ),
            f"{prefix}.k_proj.weight": draw(
                kv_heads * head_dim, hidden, std=hidden**-0.5
            ),
            f"{prefix}.v_proj.weight": draw(
                kv_heads * head_dim, hidden, std=hidden**-0.5
            ),
            f"{prefix}.q_norm.weight": draw(head_dim, std=0.05),
            f"{prefix}.k_norm.weight": draw(head_dim, std=0.05),
            f"{prefix}.o_proj.weight": draw(
                hidden, heads * head_dim, std=(heads * head_dim) ** -0.5
            ),
            f"{prefix}.indexer.index_qk_proj.weight": draw(
                (index_heads + 1) * index_dim, hidden, std=hidden**-0.5
            ),
            f"{prefix}.indexer.q_layernorm.weight": draw(index_dim, std=0.05),
            f"{prefix}.indexer.k_layernorm.weight": draw(index_dim, std=0.05),
        }

    def network(prefix):
        channels = (2 * key_heads + value_heads) * linear_dim
        values = value_heads * linear_dim
        return {
            f"{prefix}.in_proj_qkv.weight": draw(
                channels, hidden, std=hidden**-0.5
            ),
            f"{prefix}.in_proj_z.weight": draw(
                values, hidden, std=hidden**-0.5
            ),
            f"{prefix}.in_proj_b.weight": draw(
                value_heads, hidden, std=hidden**-0.5
            ),
            f"{prefix}.in_proj_a.weight": draw(
                value_heads, hidden, std=hidden**-0.5
            ),
            f"{prefix}.conv1d.weight": draw(channels, 1, taps, std=0.5),
            f"{prefix}.A_log": torch.log(
                torch.rand(value_heads, generator=generator) * 15 + 1
            ),
            f"{prefix}.dt_bias": draw(value_heads, std=1.0),
            f"{prefix}.norm.weight": 1 + draw(linear_dim, std=0.05),
            f"{prefix}.out_proj.weight": draw(hidden, values, std=values**-0.5),
        }

    language = "model.language_model"
    model = {
        f"{language}.embed_tokens.weight": draw(vocab, hidden, std=1.0),
        "lm_head.weight": draw(vocab, hidden, std=hidden**-0.5),
    }
    model |= connection(f"{language}.hyper_connection_mixer", injects=False)
    for index, kind in enumerate(layers):
        prefix = f"{language}.layers.{index}"
        model |= connection(f"{prefix}.attn_hyper_connection")
        model |= connection(f"{prefix}.mlp_hyper_connection")
        model |= feed_forward(f"{prefix}.mlp")
        model |= (
            network(f"{prefix}.linear_attn")
            if kind == "linear_attention"
            else attention(f"{prefix}.self_attn")
        )

    # The PLE layer of the stack's second layer, with the hash constants its
    # configuration
    # states and an E4M3 table.
    sizes, offsets, size, offset = [], [], base, 0
    for _ in range(ngram_heads):
        size += 1
        while any(
            size % divisor == 0 for divisor in range(2, int(size**0.5) + 1)
        ):
            size += 1
        sizes.append(size)
        offsets.append(offset)
        offset += size
    table_rows = offset + (-offset) % parts
    mask = (1 << 64) - 1

    def splitmix64(x):
        x = (x + 0x9E3779B97F4A7C15) & mask
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
        return (x ^ (x >> 31)) & mask

    bound = ((1 << 63) - 1) // vocab // 2
    multipliers = [
        2 * (splitmix64(1234 + 0x9E3779B97F4A7C15 * (i + 1)) % bound) + 1
        for i in range(3)
    ]
    codes = torch.randint(
        0,
        256,
        (table_rows, hidden // ngram_heads),
        generator=generator,
        dtype=torch.uint8,
    )
    codes = torch.where(codes & 0x7F == 0x7F, codes & 0xFE, codes)
    layer = f"{language}.layers.1.ple"
    model |= {
        f"{layer}.key_proj.weight": draw(width, hidden, std=hidden**-0.5),
        f"{layer}.value_proj.weight": draw(hidden, hidden, std=hidden**-0.5),
        f"{layer}.norm_key.weight": draw(width, std=0.1),
        f"{layer}.norm_query.weight": draw(width, std=0.1),
        f"{layer}.norm_conv.weight": draw(width, std=0.1),
        f"{layer}.conv1d.weight": draw(width, 1, taps, std=0.5),
        f"{layer}.ple_embedding.layer_multipliers": torch.tensor(
            multipliers, dtype=torch.int64
        ),
        f"{layer}.ple_embedding.ngram_heads_vocab_sizes": torch.tensor(
            sizes, dtype=torch.int64
        ),
        f"{layer}.ple_embedding.ngram_heads_offsets": torch.tensor(
            offsets, dtype=torch.int64
        ),
        f"{layer}.ple_embedding.ngram_embedding.weight_scale": torch.tensor(
            [0.03125]
        ),
    }
    shard_rows = table_rows // parts
    for shard in range(parts):
        rows = codes[shard * shard_rows : (shard + 1) * shard_rows].contiguous()
        model[f"{layer}.ple_embedding.ngram_embedding.shard_{shard}.weight"] = (
            rows.view(torch.float8_e4m3fn)
        )

    config = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "image_token_id": 62,
        "video_token_id": 63,
        "tie_word_embeddings": False,
        "text_config": {
            "attention_bias": False,
            "eos_token_id": eos,
            "hc_count": streams,
            "hc_lowrank": rank,
            "head_dim": head_dim,
            "heads_per_ngram": ngram_heads // 2,
            "hidden_act": "silu",
            "hidden_size": hidden,
            "indexer_budget": 16,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": index_dim,
            "indexer_kv_heads": 1,
            "indexer_n_heads": index_heads,
            "layer_types": layers,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": taps,
            "linear_key_head_dim": linear_dim,
            "linear_num_key_heads": key_heads,
            "linear_num_value_heads": value_heads,
            "linear_value_head_dim": linear_dim,
            "make_ngram_vocab_size_divisible_by": parts,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 256,
            "moe_intermediate_size": intermediate,
            "mtp_num_hidden_layers": 0,
            "ngram_size": 3,
            "ngram_vocab_size_base": base,
            "num_attention_heads": heads,
            "num_experts": experts_count,
            "num_experts_per_tok": top,
            "num_hidden_layers": len(layers),
            "num_key_value_heads": kv_heads,
            "output_gate_type": "sigmoid",
            "ple_conv_kernel_size": taps,
            "ple_embed_dim": hidden,
            "ple_layer_ids": [2],
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [2, 1, 1],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            "seed": 1234,
            "shared_expert_intermediate_size": intermediate,
            "split_ngram_parts": parts,
            "vocab_size": vocab,
        },
        "vision_config": {**VISION, "out_hidden_size": hidden},
    }
    write("consistency", config, model, None)


def write(name, config, model, vectors):
    directory = DATA / name / "model"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(
        {name: tensor.contiguous() for name, tensor in model.items()},
        directory / "model.safetensors",
    )
    if vectors is not None:
        save_file(
            {name: tensor.contiguous() for name, tensor in vectors.items()},
            DATA / name / "reference.safetensors",
        )
    size = sum(
        path.stat().st_size
        for path in (DATA / name).rglob("*")
        if path.is_file()
    )
    print(f"{name}: {len(model)} tensors, {size} bytes")


if __name__ == "__main__":
    torch.manual_seed(0)
    reference_fixture()
    consistency_fixture()
