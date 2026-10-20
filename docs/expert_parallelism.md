# Expert parallelism

Mixture-of-experts models can shard their routed experts across GPUs. UniServe supports three deployments: data-parallel replicas that shard the experts among themselves and exchange tokens at every expert layer, dedicated expert workers that hold the experts while attention replicas route tokens to them, and distributed weight data parallelism (DWDP), in which replicas read missing expert weights from their peers instead of exchanging tokens.

## Replicas sharing their experts

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/uniserve serve /models/Qwen3-30B-A3B \
  --served-model-name qwen3-moe --data-parallel-size 2 \
  --set moe_config.expert_parallel=true --set moe_config.exchange=alltoall
```

Every replica keeps its own scheduler, KV cache and attention, and holds `1 / N` of each layer's routed experts. The replicas run every expert layer together: before each forward, the replicas agree over a host connection on which replica's batch runs and on the step's per-replica token capacity, and every other replica joins the step without tokens of its own, so its experts serve the tokens routed to them. A replica that is idle keeps joining. Expert parallelism requires one-rank replicas and one execution lane.

`moe_config.exchange` selects the token transport:

| Exchange | Transport | Requirements |
| --- | --- | --- |
| `alltoall` (default) | FlashInfer's NVLink all-to-all around each rank's grouped experts | NVLink peers (MNNVL fabric or peer memory) |
| `megamoe` | The fused MegaMoE kernel: one launch exchanges, runs the experts and combines | NVFP4 expert weights, Blackwell GPUs |
| `deepep` | DeepEP elastic dispatch and combine | Dedicated expert workers |
| `dwdp` | Weight prefetch instead of token exchange (see below) | CuTeDSL expert kernels, peer-accessible GPUs on one host |

Each step's capacity is the smallest power of two (or the batch token bound) that holds the largest participating batch, so startup captures every graph shape at every capacity that can hold it; this lengthens startup in proportion to the number of capacities. Different peer batches change the expert kernels' batching and can change floating-point rounding.

## Dedicated expert workers

A deployment file places attention and experts on different workers:

```toml
[[workers]]
id = "attention"
ranks = ["cuda:0"]
queue_depth = 2

[workers.components.model]

[[workers]]
id = "experts"
role = "experts"
ranks = ["cuda:1"]
queue_depth = 2
```

```bash
.venv/bin/uniserve serve /models/Qwen3-30B-A3B \
  --workers attention_experts.toml --set moe_config.expert_parallel=true \
  --set moe_config.exchange=deepep --set moe_config.microbatches=2
```

Attention ranks hold no routed-expert weights; expert ranks hold every routed expert, partitioned among themselves, and own no request state. Dedicated expert workers require `deepep` or `megamoe`. `configs/afd/qwen3_moe.toml` places a tensor-parallel attention replica and two expert ranks on two hosts.

`moe_config.microbatches` splits each attention batch into microbatches that run on separate streams and host threads and take turns at every expert dispatch, so one microbatch's attention overlaps another's expert computation on the expert ranks. Expert ranks follow the same turns. The split MegaMoE kernel serves at most four microbatches.

## Distributed weight data parallelism

```bash
.venv/bin/uniserve serve /models/Qwen3-30B-A3B \
  --data-parallel-size 2 --set moe_config.expert_parallel=true \
  --set moe_config.exchange=dwdp
```

Each replica keeps its expert shard resident and, before each expert layer, copies the shards of the other replicas from peer memory into two reusable buffers per weight, prefetching the next layer while the current one computes. Replicas advance independently and never wait for each other during inference; only startup is collective. Replicas need peer-accessible GPUs on one host and one execution lane, and the experts must run on the CuTeDSL kernels (BF16 with widths that are multiples of 64, or NVFP4, on SM100). The startup log reports the resident, buffer and per-invocation copy bytes.
