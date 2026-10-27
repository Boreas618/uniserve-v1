# Expert parallelism

Mixture-of-experts models can shard their routed experts across GPUs. UniServe supports four deployments: schedulers that shard the experts among themselves and exchange tokens at every expert layer, the tensor-parallel ranks of one scheduler sharding the experts among themselves without exchanging tokens, dedicated expert workers that hold the experts while attention workers route tokens to them, and distributed weight data parallelism (DWDP), in which schedulers read missing expert weights from their peers instead of exchanging tokens.

## Schedulers sharing their experts

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/uniserve serve /models/Qwen3-30B-A3B \
  --served-model-name qwen3-moe --data-parallel-size 2 \
  --set moe_config.expert_parallel=true --set moe_config.exchange=alltoall
```

Every scheduler keeps its own KV cache and attention, and its worker holds `1 / N` of each layer's routed experts. The schedulers run every expert layer together: before each forward, their workers agree over a host connection on which scheduler's batch runs and on the step's per-scheduler token capacity, and every other worker joins the step without tokens of its own, so its experts serve the tokens routed to them. A worker that is idle keeps joining. Schedulers sharing their experts each run one one-rank worker group.

`moe_config.exchange` selects the token transport:

| Exchange | Transport | Requirements |
| --- | --- | --- |
| `alltoall` (default) | FlashInfer's NVLink all-to-all around each rank's grouped experts | NVLink peers (MNNVL fabric or peer memory) |
| `megamoe` | The fused MegaMoE kernel: one launch exchanges, runs the experts and combines | NVFP4 expert weights, Blackwell GPUs |
| `deepep` | DeepEP elastic dispatch and combine | Dedicated expert workers |
| `dwdp` | Weight prefetch instead of token exchange (see below) | CuTeDSL expert kernels, peer-accessible GPUs on one host |

Each step's capacity is the smallest power of two (or the batch token bound) that holds the largest participating batch, so startup captures every graph shape at every capacity that can hold it; this lengthens startup in proportion to the number of capacities. Different peer batches change the expert kernels' batching and can change floating-point rounding.

## Tensor-parallel ranks sharing their experts

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/uniserve serve /models/Qwen3-30B-A3B \
  --served-model-name qwen3-moe --tensor-parallel-size 4 \
  --set moe_config.expert_parallel=true
```

With one scheduler, expert parallelism shards the experts over the ranks of its single tensor-parallel worker group: attention and the other layers stay tensor-parallel, and rank `r` of `P` keeps the whole experts `[r * E / P, (r + 1) * E / P)` of every layer instead of `1 / P` of every expert's channels. The ranks hold every token of the batch, so no exchange moves tokens: each runs its experts over all of them, routes to the other ranks' experts contributing nothing, and the group sums the ranks' outputs with the all-reduce that sums tensor-parallel expert shards. `moe_config.exchange` keeps its default; the other exchanges apply across schedulers or to dedicated expert workers.

## Dedicated expert workers

A deployment file places attention and experts on different workers:

```toml
[[worker_groups]]
id = "attention"
ranks = ["cuda:0"]
queue_depth = 2

[worker_groups.components.model]

[[worker_groups]]
id = "experts"
role = "experts"
ranks = ["cuda:1"]
queue_depth = 2
```

```bash
.venv/bin/uniserve serve /models/Qwen3-30B-A3B \
  --worker-groups attention_experts.toml --set moe_config.expert_parallel=true \
  --set moe_config.exchange=deepep --set moe_config.microbatches=2
```

Attention ranks hold no routed-expert weights; expert ranks hold every routed expert, partitioned among themselves, and own no request state. Dedicated expert workers require `deepep` or `megamoe`. `configs/afd/qwen3_moe.toml` places a tensor-parallel attention group and two expert workers on two hosts.

`moe_config.microbatches` splits each attention batch into microbatches that run on separate streams and host threads and take turns at every expert dispatch, so one microbatch's attention overlaps another's expert computation on the expert ranks. Expert ranks follow the same turns. The split MegaMoE kernel serves at most four microbatches.

## Distributed weight data parallelism

```bash
.venv/bin/uniserve serve /models/Qwen3-30B-A3B \
  --data-parallel-size 2 --set moe_config.expert_parallel=true \
  --set moe_config.exchange=dwdp
```

Each scheduler's worker keeps its expert shard resident and, before each expert layer, copies the shards of the other workers from peer memory into two reusable buffers per weight, prefetching the next layer while the current one computes. Schedulers advance independently and never wait for each other during inference; only startup is collective. The workers need peer-accessible GPUs on one host, and the experts must run on the CuTeDSL kernels (BF16 with widths that are multiples of 64, or NVFP4, on SM100). The startup log reports the resident, buffer and per-invocation copy bytes.
