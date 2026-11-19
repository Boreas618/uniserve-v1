# DeepSeek-V4.1-Flash

UniServe serves `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`DeepseekV41ForCausalLM`) at its context of 1M tokens (`max_position_embeddings = 1048576`) with the checkpoint's own numerics: FP8 dense projections under 32 × 32 UE8M0 block scales, read as MXFP8; MXFP4 routed experts; the window, compressed-latent and index-key caches in the reference's MXFP8, NVFP4 and MXFP4 records; the hyper-connected residual of four streams; and the hashed n-gram memories (Engram) of layers 1 and 14. The DSpark drafter and the vision tower are not served: a deployment does not speculate, and a request's image parts are refused.

## Serving it on four GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uniserve serve configs/deepseek_v41/dp4-ep4
```

The package (`configs/deepseek_v41/dp4-ep4/config.toml`) runs four data-parallel schedulers whose ranks share their routed experts (`data_parallel_size = 4` over four `[[worker_groups]]` of one rank each, `moe_config.expert_parallel`, [Expert parallelism](expert_parallelism.md)). Each rank holds:

| Part | Placement | Per GPU |
| --- | --- | --- |
| Attention, shared experts, router, embedding, head (FP32) | whole on every rank | about 11 GiB |
| Routed experts | 96 of every layer's 384, behind the `alltoall` exchange | about 68 GiB |
| n-gram tables (two of about 384M rows of 256 E4M3 values) | six of each table's 24 hash heads, a contiguous range of its rows; each lookup gathers every rank's row ids and returns each rank its own tokens' rows | about 47 GiB |

The weights take about 125 GiB of each GPU's 185, so the package raises the deployment's share of a GPU (`mem_fraction_static = 0.95`) to hold them beside the graph pool and the KV cache. It serves the release's context (`max_model_len = 1048576`), which AgentX prompts reach (990K tokens at most, 35K at the median), to 16 running requests a scheduler, 64 over the four, and prefills a prompt in chunks of 4096 tokens. On GB200 each GPU's share of 188.0 GB holds:

| Part | Bytes |
| --- | --- |
| Weights | 134.7 GB |
| Process storage, the graph pool among it | 5.6 GB |
| Compressed latents and index keys (360,448 units of 64 tokens, 23.1M tokens of full history) | 20.5 GB |
| The compressor rings' checkpoints (`kv_cache_config.state_ratio = 1.0`) | 23.1 GB |

The checkpoint publishes no chat template; its prompt format is its reference encoder (`encoding/encoding.py`), which the family's request side ports (`processing::deepseek_v41`): thinking mode by default, chat mode with `reasoning_effort` `none` or `chat_template_kwargs` `{"enable_thinking": false}`, reasoning between `<think>` and `</think>`, and DSML tool calls.

## Caches

Every layer keeps a sliding window of its last 128 latents; four layers (2, 8 and 14 compressing every two tokens, 20 every token) keep the compressed latents and the index keys the layers after them read; layers 2, 8 and 14 keep each request's compressor remainder. A request's full history takes about 891 bytes a token (the compressed latents' 720 and the index keys' 170), so a prompt of 1M tokens holds under 1 GB of a rank's KV cache beside its window of a few megabytes. The window's pages are never cached: a prefix hit of the compressed fields rewinds the window in whole pages and recomputes it (bounded replay, [Architecture](architecture.md#storage-and-device-storage)).

## Long contexts

Eight layers index their compressed entries (2, 8, 14 and 20 over every visible entry, up to 512K and 1M of them; 24, 28, 32 and 36 inside the 2048 blocks of 8 entries layer 20 ranks best) and keep 512 entries a query. An indexer scores and selects its queries a block at a time, each block's FP32 scores within 1 GiB, so a prefill chunk of 4096 tokens over a table of 1M entries holds 256 queries' scores at once rather than 16 GiB; a decode batch is one block. The scoring and the selection of a query cost the entries its request holds, not the table its page table could address: a short request in a deployment of 1M tokens scores and ranks as it would in a short one.

## Numerics

Every quantized value is the reference's (`inference/kernel.py`): the projections' activations are encoded per 32 by the covering UE8M0 rule, the cache records are the reference's fake quantizations exactly, the router scores in FP32 and the head computes FP32 logits. Three steps differ from the reference in their order: the routing weight multiplies an expert's output after its down projection, in the combine, where the reference scales the gated row before encoding it; the first MXFP4 product encodes the gated row by the OCP Microscaling rule in its epilogue; and the routed sum is rounded to BF16 before the shared expert's output is added. Every reduction runs in its kernels' own order, and the indexer's selection may break near ties otherwise than the reference's BF16 ranking.

## Limits

- The attention, the indexer and the routed experts run correct-first Triton and trtllm-gen kernels.
- A tensor-parallel split of the attention, pipeline and sequence splits, and speculative decoding are refused.
- Prefill/decode handoffs and the host KV tier do not hold the window.
