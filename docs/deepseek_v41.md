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
| Process storage, the graph pool (2.4 GB) among it | 5.6 GB |
| Compressed latents and index keys (770,048 units of 64 tokens, 49.3M tokens of full history) | 43.9 GB |
| The window pool, the request slots and the pooled page tables | the rest |

The compressor rings keep no checkpoint, so the package keeps no state region ([Caches](#caches)). The cache is numbered for 802,816 units, 49 segments of 16,384; the warm process grows by about 3.3 GB past what the start measured, 1.5 to 1.7 GB past the share, and the rank gives the last two segments back.

An epoch update that brings the package onto GPUs another epoch of the deployment serves sizes it beside that epoch within one share of each GPU, the deployment's quota, rather than a share of its own ([Admin](admin.md)): each rank numbers its cache for the whole share, creates the segments the share holds beside the serving epoch's reservations and the GPU's free memory, and grows them as the serving epoch drains. Beside an epoch that holds 25 GB of a GPU, a rank starts with about 26 GB of latent cache, 27 segments or 28M tokens, enough for a prompt of 1M tokens (0.93 GB). What the share does not cover, the warm-up's growth past it until the give-back and the serving epoch's growth past its reservations, must fit in the GPU's memory the share leaves, 9.9 GB at 0.95.

The checkpoint publishes no chat template; its prompt format is its reference encoder (`encoding/encoding.py`), which the family's request side ports (`processing::deepseek_v41`): thinking mode by default, chat mode with `reasoning_effort` `none` or `chat_template_kwargs` `{"enable_thinking": false}`, reasoning between `<think>` and `</think>`, and DSML tool calls.

## Caches

Every layer keeps a sliding window of its last 128 latents; four layers (2, 8 and 14 compressing every two tokens, 20 every token) keep the compressed latents and the index keys the layers after them read; layers 2, 8 and 14 keep each request's compressor remainder, a ring of its last eight positions' projections in its request slot, which a forward from a page boundary reads nothing of since every pair is complete there, so the prefix cache keeps no checkpoint of it. A request's full history takes about 891 bytes a token (the compressed latents' 720 and the index keys' 170), so a prompt of 1M tokens holds under 1 GB of a rank's KV cache beside its window of a few megabytes. The window's pages are never cached: a prefix hit of the compressed fields rewinds the window in whole pages and recomputes it (bounded replay, [Architecture](architecture.md#storage-and-device-storage)). The window's pages lie in a pool of their own on each rank, 1.35 MB a page of 64 tokens over the 40 layers that keep it, sized for the request slots and three steps' tokens: every request's window and partial pages (two, two and one ahead, five a slot) and three times a step's 4096 tokens (64 pages), since the scheduler asks for a prompt's next chunk in the step that forms the current one, while the pages the previous chunk left behind the window are still on their way back. At 16 slots that is 1 + 16 × 5 + 3 × 64 = 273 pages, 369 MB.

## Long contexts

Eight layers index their compressed entries (2, 8, 14 and 20 over every visible entry, up to 512K and 1M of them; 24, 28, 32 and 36 inside the 2048 blocks of 8 entries layer 20 ranks best) and keep 512 entries a query. An indexer scores and selects its queries a block at a time, each block's FP32 scores within 1 GiB, so a prefill chunk of 4096 tokens over a table of 1M entries holds 256 queries' scores at once rather than 16 GiB; a decode batch is one block. The scoring and the selection of a query cost the entries its request holds, not the table its page table could address: a short request in a deployment of 1M tokens scores and ranks as it would in a short one.

## Numerics

Every quantized value is the reference's (`inference/kernel.py`): the projections' activations are encoded per 32 by the covering UE8M0 rule, the cache records are the reference's fake quantizations exactly, the router scores in FP32 and the head computes FP32 logits. Three steps differ from the reference in their order: the routing weight multiplies an expert's output after its down projection, in the combine, where the reference scales the gated row before encoding it; the first MXFP4 product encodes the gated row by the OCP Microscaling rule in its epilogue; and the routed sum is rounded to BF16 before the shared expert's output is added. Every reduction runs in its kernels' own order, and the indexer's selection may break near ties otherwise than the reference's BF16 ranking.

Open question: the served distributions move by about 0.02 KL with where a request's pages lie and with what the deployment served before it, while one deployment answers a request the same way every time. The mean KL between the answers to a 32,908-token AgentX prompt and to its prefix hit was 0.048 in one deployment and 0.065 in another of the same build, which served no other prompt first, while that of a 65,678-token prompt was bitwise the same in both; builds whose caches lay pages at other unit indices differ from each other by the same amount, cold and hit alike. The cause is not established; candidates are the indexer's top-k breaking ties by entry or unit position, and an accumulation order over pages. Every such spread lies within the comparison bound (0.124).

## Limits

- The attention, the indexer and the routed experts run correct-first Triton and trtllm-gen kernels: on GB200 a cold prefill of a 575K-token prompt takes about 9 minutes, and AgentX's longest prompt (990K tokens) about 23 minutes beside three shorter ones. A prefix hit computes only the prompt past it and the window's pages before its end.
- A tensor-parallel split of the attention, pipeline and sequence splits, and speculative decoding are refused.
- Prefill/decode handoffs and the host KV tier do not hold the window.
