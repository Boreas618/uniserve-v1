# Qwen3-MoE GSM8K quality confirmation

This page specifies how UniServe confirms Qwen3-MoE quality on GSM8K. Design section 10 confirms the model family by accuracy on a public benchmark: `uniserve-eval` runs the same checkpoint on the same sample set with the same sampling settings twice, once against the implementation in the checkout and once against a candidate build named with `--executable`, and accepts the candidate when the accuracy difference lies within the two-sided 95% confidence interval of the difference of two binomial proportions at that sample size. The confirmation covers three cells: the BF16 checkpoint, its ModelOpt NVFP4 export, and the BF16 checkpoint with FP8 KV pages.

The protocol lives in `uniserve_eval/qwen3_moe.toml` (suite `qwen3-moe-gsm8k`); the workload, sample set, request shape, extraction rule and comparison are fixed there and in the evaluator, not chosen per run. This page states those fixed choices and the commands that exercise them. The [results](#results) are pending.

## Checkpoint and cells

| Cell | Checkpoint | KV cache dtype | `provenance` |
| --- | --- | --- | --- |
| `qwen3-moe-gsm8k-bf16` | `/workspace/models/Qwen3-30B-A3B` | `bfloat16` | `precision="bfloat16"`, `kv_cache_dtype="bfloat16"` |
| `qwen3-moe-gsm8k-nvfp4` | local `nvidia/Qwen3-30B-A3B-NVFP4` export | `bfloat16` | `precision="nvfp4"`, `kv_cache_dtype="bfloat16"` |
| `qwen3-moe-gsm8k-bf16-fp8-kv` | `/workspace/models/Qwen3-30B-A3B` | `float8_e4m3fn` | `precision="bfloat16"`, `kv_cache_dtype="float8_e4m3fn"` |

The BF16 and FP8 KV cells serve the Qwen3-MoE checkpoint at `/workspace/models/Qwen3-30B-A3B`, named by `UNISERVE_QWEN3_MOE_MODEL`; the profile defaults the path. The NVFP4 cell serves a local copy of `nvidia/Qwen3-30B-A3B-NVFP4`, a ModelOpt unified export whose own `config.json` declares its quantization; `UNISERVE_QWEN3_MOE_NVFP4_MODEL` names that directory and has no default, so a host without the export cannot run the NVFP4 cell. The other two cells' server commands do not name the variable. The cells differ in the checkpoint path and the KV cache dtype, and the FP8 KV cell also names an attention provider (see [FP8 KV pages](#fp8-kv-pages)). Each cell's `provenance` is its identity in the comparison.

No cell states `dtype` or `quantization`: the checkpoint's own representation is used, and `quantization_config` and its precision presets are retired in favour of the checkpoint representation and the `quantization` field. The serving capacity is build configuration: `--mem-fraction-static` defaults to 0.80 and `kv_cache_config.max_tokens` to 65536, and a host may lower either through `UNISERVE_QWEN3_MOE_MEM_FRACTION` or `UNISERVE_QWEN3_MOE_KV_MAX_TOKENS` without making a cell's two runs incomparable.

## Dataset and sample set

The dataset is `openai/gsm8k`'s test split at revision `740312add88f781978c0658806c59bc2815b9866`, file `main/test-00000-of-00001.parquet`: 1319 rows read in file order, without shuffling or filtering. Each row's question is the user message verbatim and its reference answer supplies the gold value. `dataset_path` may name a local Parquet or JSON Lines file, or a directory holding the split; only an unset path downloads the pinned file from the Hub. The profile states `num_prompts = 1319`, and every run records the selected rows' identity as `{"count": 1319, "sha256": "6403e64f54dff14d84e6f6839f9dc8d7178c6500e4d791a826d05e3ff00cbdb8"}`.

The complete split is the fixed sample set. A comparison requires both runs to record the same identity and a count of 1319, so a run over another sample size is not this confirmation.

## Request and sampling

Each measured row becomes one streamed chat-completions request to `POST /v1/chat/completions` with `stream_options.include_usage`. The request carries one system message and one user message:

```text
Solve the problem step by step. End your answer with the final answer on its own line in the form `#### <number>`.
```

The user message is the row's question verbatim. The sampling fields are fixed: `temperature = 0.0`, `top_p = 1.0`, `seed = 42`, `ignore_eos = false` (no early stop), `max_completion_tokens = 1024`, and `chat_template_kwargs.enable_thinking = false`. Each cell measures all 1319 rows after one unmeasured warmup request built from the first row; requests are submitted without an artificial delay, at most 64 in flight.

## Scoring

Accuracy is strict extracted-answer accuracy. The prediction is the text after the last `#### ` marker in the generated text, with reasoning and content both counted; the gold value is the text after the last `#### ` marker in the row's reference answer. The marker includes its trailing space. The remainder after the marker must reduce to one optionally signed integer or decimal, optionally with a `$`, thousands commas, and surrounding whitespace or punctuation, and only non-word characters may follow it: `#### 1,250`, `#### $18`, `#### 18.`, `#### (18)` and `#### -18.5` parse, while `####20`, `#### 18 because 2+3`, `#### The answer is 18` and `#### 1,25` do not. Values compare numerically, so `18` and `18.0` are equal.

An answer that does not parse is incorrect, not a request failure. A failed request is also incorrect. Every measured row stays in the denominator, so `total` is 1319 for a complete run and `accuracy = correct / total`; a failed request additionally fails `all_requests_succeeded`, so a run with any failed request is not a valid comparison input. The summary carries `accuracy`, `correct` and `total`.

## Acceptance

Each cell is measured twice with the same profile: a baseline run on the implementation in the checkout, and a candidate run with `--executable <new server binary>`. Both runs must be completed, valid `gsm8k` runs over the same `selected_rows` identity whose count is 1319, with `load.num_prompts` equal to that count and the same `workload.sampling`, `workload.provenance`, `workload.model` and resolved daemon model argument. A pair that fails a premise is reported as not comparable instead of a verdict.

For baseline accuracy `p1 = correct1 / n1` and candidate accuracy `p2 = correct2 / n2`, with `n1 = n2 = 1319`:

```text
d  = p2 - p1
SE = sqrt(p1 (1 - p1) / n1 + p2 (1 - p2) / n2)
interval = d ± 1.959963985 * SE
```

The factor `1.959963985` is the two-sided 95% normal quantile. The candidate is accepted exactly when `|d| <= 1.959963985 * SE`, equivalently when the interval contains zero. `uniserve-eval compare` prints the report as JSON with both accuracies and counts, `d`, `SE`, the interval, the verdict and both provenances, and exits 0 when accepted, 2 when rejected, and 1 when a bundle is missing or the runs are not comparable.

## FP8 KV pages

The FP8 KV cell is the BF16 checkpoint with `kv_cache_config.dtype = float8_e4m3fn`. It is the only cell that names an attention provider: `--set attention_backend=${UNISERVE_QWEN3_FP8_ATTENTION_BACKEND:-torch}`, because today only the torch provider reads per-block FP8 pages at head dimension 128 and the automatic selection raises for them. The baseline leaves the variable unset and so uses `torch`; the candidate sets it to `auto` or to its own provider, and changes nothing else. The BF16 and NVFP4 cells name no attention backend, so each implementation selects the provider it has.

## Running the confirmation

The profile reads these environment variables:

```bash
export UNISERVE_QWEN3_MOE_MODEL=/workspace/models/Qwen3-30B-A3B
export UNISERVE_QWEN3_MOE_NVFP4_MODEL=<local directory of the NVFP4 export>
export UNISERVE_QWEN3_MOE_CUDA_VISIBLE_DEVICES=0
# Optional: UNISERVE_GSM8K_DATASET names a local copy of the test split;
# UNISERVE_QWEN3_MOE_MEM_FRACTION, UNISERVE_QWEN3_MOE_KV_MAX_TOKENS and
# UNISERVE_QWEN3_MOE_TENSOR_PARALLEL_SIZE set capacity and parallelism;
# UNISERVE_QWEN3_FP8_ATTENTION_BACKEND selects the FP8 cell's provider.
```

The commands run from the checkout with the environment of the installation instructions, including the `bench` extra for the Parquet split. Run the evaluator's GSM8K tests, the fixed-sample-set pre-flight, the plan, and the two runs, then compare each cell:

```bash
.venv/bin/python -m pytest -q tests/python/unit/eval_driver/test_gsm8k_benchmark.py tests/python/unit/eval_driver/test_accuracy_compare.py
.venv/bin/python -c "from uniserve_eval.config import load_config; from uniserve_eval.datasets import load_examples; c = load_config('uniserve_eval/qwen3_moe.toml'); rows, _ = load_examples(c.selected_points('qwen3-moe-gsm8k-bf16')[0]); print(len(rows), rows[0].id, rows[-1].id)"
.venv/bin/uniserve-eval --config uniserve_eval/qwen3_moe.toml plan qwen3-moe-gsm8k
.venv/bin/uniserve-eval --config uniserve_eval/qwen3_moe.toml run qwen3-moe-gsm8k --output-root artifacts/qwen3_moe/gsm8k-baseline
.venv/bin/uniserve-eval --config uniserve_eval/qwen3_moe.toml run qwen3-moe-gsm8k --output-root artifacts/qwen3_moe/gsm8k-candidate --executable <new server binary>
for point in qwen3-moe-gsm8k-bf16 qwen3-moe-gsm8k-nvfp4 qwen3-moe-gsm8k-bf16-fp8-kv; do
  .venv/bin/uniserve-eval compare \
    artifacts/qwen3_moe/gsm8k-baseline/$point \
    artifacts/qwen3_moe/gsm8k-candidate/$point
done
```

The pre-flight prints `1319 gsm8k-00000 gsm8k-01318`. The suite runs its three cells serially, one server at a time; `run` refuses a non-empty result directory, so each of the two runs needs its own root and a rerun needs a fresh one. The candidate run may differ from the baseline only in `--executable`, `--output-root` and, for the FP8 KV cell, `UNISERVE_QWEN3_FP8_ATTENTION_BACKEND`.

## Results

The baseline run and the candidate comparison are pending. When both have run on the target deployment, this section records, per cell, both accuracies and counts, the difference, its standard error, the 95% interval and the verdict. No accuracy, interval or verdict is claimed here.
