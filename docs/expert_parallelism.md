# Expert parallelism

Mixture-of-experts models can shard their routed experts across GPUs. UniServe supports four deployments: schedulers that shard the experts among themselves and exchange tokens at every expert layer, the tensor-parallel ranks of each scheduler sharding its experts among themselves without exchanging tokens, dedicated expert workers that hold the experts while attention workers route tokens to them, and distributed weight data parallelism (DWDP), in which schedulers read missing expert weights from their peers instead of exchanging tokens.

## Schedulers sharing their experts

```toml
# qwen3-moe/config.toml
model = "Qwen/Qwen3-30B-A3B"
served_model_name = "qwen3-moe"
data_parallel_size = 2

[moe_config]
expert_parallel = true
```

```bash
CUDA_VISIBLE_DEVICES=0,1 uniserve serve qwen3-moe
```

Every scheduler keeps its own KV cache and attention, and its worker holds `1 / N` of each layer's routed experts. The schedulers run every expert layer together: before each forward, their workers agree over a host connection on which scheduler's batch runs and on the step's per-scheduler token capacity, and every other worker joins the step without tokens of its own, so its experts serve the tokens routed to them. A worker that is idle keeps joining. Schedulers sharing their experts each run one one-rank worker group.

The token transport is the package that fills the `expert_exchange` operator's descriptor key: the installed package that covers the key with the highest manifest priority, `alltoall` among the tree's packages of equal priority, unless the deployment's `[kernels]` table pins the key to another (`docs/configuration.md`, the `[kernels]` table):

| Exchange | Transport | Requirements |
| --- | --- | --- |
| `alltoall` | FlashInfer's NVLink all-to-all around each rank's grouped experts | NVLink peers (MNNVL fabric or peer storage); no attention ranks |
| `megamoe` | The fused MegaMoE kernel: one launch exchanges, runs the experts and combines | NVFP4 expert weights, Blackwell GPUs |
| `deepep` | DeepEP elastic dispatch and combine | Dedicated expert workers |

`alltoall` and `deepep` dispatch the activation rows and combine their sums in two launches, and each rank runs its own experts' grouped products between them: BF16 experts, or the MXFP4 experts of a block-scaled export (DeepSeek-V4.1), whose first product encodes the received rows into MXFP8 as it reads them. `megamoe` runs calibrated NVFP4 experts inside its one launch.

Each step's capacity is the smallest power of two (or the batch token bound) that holds the largest participating batch, so startup captures every graph shape at every capacity that can hold it; this lengthens startup in proportion to the number of capacities. Different peer batches change the expert kernels' batching and can change floating-point rounding.

## Tensor-parallel ranks sharing their experts

```toml
model = "Qwen/Qwen3-30B-A3B"
served_model_name = "qwen3-moe"
tensor_parallel_size = 4

[moe_config]
expert_parallel = true
```

Without dedicated expert workers and without schedulers of one rank each, expert parallelism shards the experts over the ranks of each scheduler's tensor-parallel worker group: attention and the other layers stay tensor-parallel, and rank `r` of `P` keeps the whole experts `[r * E / P, (r + 1) * E / P)` of every layer, at their full intermediate width, instead of `1 / P` of every expert's channels. The ranks hold every token of the batch, so no exchange moves tokens: each runs its experts over all of them, routes to the other ranks' experts contributing nothing, and the group sums the ranks' outputs with the all-reduce that sums tensor-parallel expert shards. The key/value cache is laid out as the same group's without the flag, and the shards of a calibrated NVFP4 checkpoint's experts are read whole, unpadded.

```toml
model = "Qwen/Qwen3-30B-A3B"
served_model_name = "qwen3-moe"
tensor_parallel_size = 2
data_parallel_size = 2

[moe_config]
expert_parallel = true
```

Several such schedulers exchange nothing with each other, so they are independent replicas, each an engine of its own as without the flag, and each shards its experts over its own tensor-parallel ranks: `tensor_parallel_size = 2` with `data_parallel_size = 2` runs two replicas of two ranks, each rank holding half the experts. A deployment layout's tensor-parallel groups shard their experts the same way. `moe_config.exchange` keeps its default, and any other exchange is refused, since the exchanges apply across schedulers of one rank each or to dedicated expert workers; a group split over pipeline stages or sequence members has no tensor-parallel ranks to shard the experts over and is refused with the flag.

## Dedicated expert workers

A deployment layout places attention and experts on different workers:

```toml
model = "Qwen/Qwen3-30B-A3B"

[moe_config]
expert_parallel = true
microbatches = 2

[[worker_groups]]
id = "attention"
ranks = ["cuda:0"]
queue_depth = 2

[worker_groups.blocks.model]

[[worker_groups]]
id = "experts"
role = "experts"
ranks = ["cuda:1"]
queue_depth = 2
```

Attention ranks hold no routed-expert weights; expert ranks hold every routed expert, partitioned among themselves, and own no request state. Dedicated expert workers require `deepep` or `megamoe`. The package `configs/afd/qwen3_moe` places a tensor-parallel attention group on its first host and two expert workers on its second; its `hosts` name the hosts it serves on.

`moe_config.microbatches` splits each attention batch into microbatches that run on separate streams and host threads and take turns at every expert dispatch, so one microbatch's attention overlaps another's expert computation on the expert ranks. Expert ranks follow the same turns. The split MegaMoE kernel serves at most four microbatches.
