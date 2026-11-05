# Qwen3-MoE GSM8K quality confirmation

This page specifies how UniServe confirms Qwen3-MoE quality on GSM8K: `uniserve-eval` runs **each cell** on the same checkpoint, sample set and sampling settings twice — once on the existing implementation's tree (`f1de185a2`), once on a candidate build named with `--executable` — and accepts the candidate when the accuracy difference lies within the two-sided 95% confidence interval of the difference of two binomial proportions at that sample size. The confirmation covers three cells: the BF16 checkpoint, its ModelOpt NVFP4 export, and the BF16 checkpoint with FP8 KV pages. Each cell is compared with itself; two different cells are not a comparable pair, because they differ in the checkpoint or the KV cache dtype, and the comparison requires both runs to share their recorded provenance.

The protocol lives in `uniserve_eval/qwen3_moe.toml` (suite `qwen3-moe-gsm8k`); the workload, sample set, request shape, extraction rule and comparison are fixed there and in the evaluator, not chosen per run. This page states those fixed choices and the commands that exercise them. The [results](#results) are pending.

## Checkpoint and cells

| Cell | Checkpoint | KV cache dtype | `provenance` |
| --- | --- | --- | --- |
| `qwen3-moe-gsm8k-bf16` | `/workspace/models/Qwen3-30B-A3B` | `bfloat16` | `precision="bfloat16"`, `kv_cache_dtype="bfloat16"` |
| `qwen3-moe-gsm8k-nvfp4` | local `nvidia/Qwen3-30B-A3B-NVFP4` export | `bfloat16` | `precision="nvfp4"`, `kv_cache_dtype="bfloat16"` |
| `qwen3-moe-gsm8k-bf16-fp8-kv` | `/workspace/models/Qwen3-30B-A3B` | `float8_e4m3fn` | `precision="bfloat16"`, `kv_cache_dtype="float8_e4m3fn"` |

All three cells also record `checkpoint = "Qwen3-30B-A3B"`; the column shows the fields that differ.

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

Accuracy is strict extracted-answer accuracy. A model states its final answer either after a `#### ` marker or inside LaTeX's `\boxed{...}`; both are read, with reasoning and content both counted, and the form that starts last decides, because that is the final answer the model wrote last. Only that form's text is parsed, so a malformed last form is an unparseable answer rather than a fallback to an earlier number. The marker includes its trailing space, so `####20` carries no answer. The payload must reduce to one optionally signed integer or decimal, optionally with a `$`, thousands commas, and surrounding whitespace or punctuation, and only non-word characters may follow it: `#### 1,250`, `#### $18`, `#### 18.`, `#### (18)`, `#### -18.5`, `\boxed{18}` and `$$\boxed{ 1,250 }$$` parse, while `####20`, `#### 18 because 2+3`, `#### The answer is 18`, `#### 1,25`, `\boxed{\frac{1}{2}}`, `\boxed{18 apples}` and `\boxed{2:00 pm}` do not. Values compare numerically, so `18` and `18.0` are equal. The gold value is the value after the last `#### ` marker in the row's reference answer.

An answer that does not parse is incorrect, not a request failure. A failed request is also incorrect. Every measured row stays in the denominator, so `total` is 1319 for a complete run and `accuracy = correct / total`; a failed request additionally fails `all_requests_succeeded`, so a run with any failed request is not a valid comparison input. The summary carries `accuracy`, `correct` and `total`, and the validation statistics carry `correct`, `total`, `unparseable_predictions` and the form counts `decided_by_marker`, `decided_by_boxed` and `without_answer_form`, so a shift in how a checkpoint states its answers is visible in the run summary.

## Acceptance

Each cell is measured twice with the same profile: a baseline run on the current implementation, and a candidate run with `--executable <new server binary>`. Both runs must be completed, valid `gsm8k` runs over the same `selected_rows` identity whose count is 1319, with `load.num_prompts` equal to that count and the same `workload.sampling`, `workload.provenance`, `workload.model` and resolved daemon model argument. A pair that fails a premise is reported as not comparable instead of a verdict.

For baseline accuracy `p1 = correct1 / n1` and candidate accuracy `p2 = correct2 / n2`, with `n1 = n2 = 1319`:

```text
d  = p2 - p1
SE = sqrt(p1 (1 - p1) / n1 + p2 (1 - p2) / n2)
interval = d ± 1.959963985 * SE
```

The factor is the two-sided 95% normal quantile the evaluator computes, `statistics.NormalDist().inv_cdf(0.975) = 1.959963984540054` (`1.959963985` rounded here); the acceptance is `|d| <=` that factor times `SE`, equivalently the interval contains zero. `uniserve-eval compare` prints the report as JSON with both accuracies and counts, `d`, `SE`, the interval, the verdict and both provenances, and exits 0 when accepted, 2 when rejected, and 1 when a bundle is missing or the runs are not comparable.

## FP8 KV pages

The FP8 KV cell is the BF16 checkpoint with `kv_cache_config.dtype = float8_e4m3fn`. It is the only cell that names an attention provider, and **the two implementations name different ones**, because a provider is a name in each implementation's own registry:

- **The baseline** (today's implementation) uses `torch`, the only provider that reads per-block FP8 pages at head dimension 128; the automatic selection raises for them, and every native paged provider refuses a per-block FP8 cache. The cell states it as `--set attention_backend=${UNISERVE_QWEN3_FP8_ATTENTION_BACKEND:-torch}`.
- **The candidate** reads the planes in its own attention layer, not through a provider: `nn::layers::Attention` accepts a cache whose two planes are both `F8E4M3` over a BF16 compute type (`crates/nn/src/layers/attention.rs:535-556`) and requires each FP8 page's scale (`page_scale`, `:733-745`), and the model's attention is always that paged backend. `--set attention_backend` does not choose it: the string is carried to the Python loading path (`crates/worker/src/worker/loading.rs:244`), so the cell states a name that path accepts (`auto`) rather than a provider name — `torch`, the baseline's, does not exist in the rewrite.

The BF16 and NVFP4 cells name no attention backend, so each implementation selects the provider it has.

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

The commands run from the checkout with the environment of the installation instructions, including the `bench` extra for the Parquet split. The reference side is a **second checkout of the baseline tree** — `git worktree add /tmp/gsm8k-reference f1de185a2`, with its own `uv sync` — so the two implementations never share an environment; the block below is that side's, then the candidate's. The candidate needs the root package's release binary and an environment installed from this checkout, because the worker imports the `uniserve` package of the environment it runs in. Run the evaluator's GSM8K tests, the fixed-sample-set pre-flight and the reference plan and run, then build and run the candidate's three cells by name, then compare each cell with itself:

```bash
cd /tmp/gsm8k-reference   # the baseline tree, installed on its own
.venv/bin/python -m pytest -q tests/python/unit/eval_driver/test_gsm8k_benchmark.py tests/python/unit/eval_driver/test_accuracy_compare.py
.venv/bin/python -c "from uniserve_eval.config import load_config; from uniserve_eval.datasets import load_examples; c = load_config('uniserve_eval/qwen3_moe.toml'); rows, _ = load_examples(c.selected_points('qwen3-moe-gsm8k-bf16')[0]); print(len(rows), rows[0].id, rows[-1].id)"
.venv/bin/python -m uniserve_eval --config uniserve_eval/qwen3_moe.toml plan qwen3-moe-gsm8k
.venv/bin/python -m uniserve_eval --config uniserve_eval/qwen3_moe.toml run qwen3-moe-gsm8k --output-root artifacts/qwen3_moe/gsm8k-baseline
# Candidate: build the root package's binary, install this checkout's environment, then run the cells by name.
cargo build --release
export VIRTUAL_ENV=$PWD/.venv
.venv/bin/python -m uniserve_eval --config uniserve_eval/qwen3_moe.toml plan qwen3-moe-gsm8k --executable "$PWD/target/release/uniserve"
# The FP8 KV cell's provider name differs by implementation: the baseline's
# `torch` provider does not exist in the rewrite, whose FP8 planes its own
# attention layer reads. The profile reads this variable for that cell only.
export UNISERVE_QWEN3_FP8_ATTENTION_BACKEND=auto
for cell in qwen3-moe-gsm8k-bf16 qwen3-moe-gsm8k-nvfp4 qwen3-moe-gsm8k-bf16-fp8-kv; do
  .venv/bin/python -m uniserve_eval --config uniserve_eval/qwen3_moe.toml run "$cell" --executable "$PWD/target/release/uniserve" --output-root artifacts/qwen3_moe/gsm8k-candidate || break
done
for cell in qwen3-moe-gsm8k-bf16 qwen3-moe-gsm8k-nvfp4 qwen3-moe-gsm8k-bf16-fp8-kv; do
  .venv/bin/python -m uniserve_eval compare "artifacts/qwen3_moe/gsm8k-baseline/$cell" "artifacts/qwen3_moe/gsm8k-candidate/$cell"
done
```

The pre-flight prints `1319 gsm8k-00000 gsm8k-01318`. The baseline suite runs its three cells serially, one server at a time. The candidate builds the root package's `target/release/uniserve`, installs the environment from the same checkout, names that binary with `--executable`, and runs the same three cells by name, also serially, each writing only its own `<point>` directory. `run` refuses a non-empty result directory, so the baseline and candidate roots are separate and a rerun needs a fresh one. The candidate run uses the same profile and may differ from the baseline only in `--executable`, `--output-root`, and — for the FP8 KV cell alone — the name the loading path is given for its attention: `torch` on the baseline, `auto` on the candidate. A provider is a name in one implementation's registry, so the two runs cannot name the same one: `torch` is the baseline's only reader of per-block FP8 pages, and the rewrite reads those planes in its own attention layer and takes `attention_backend` as a loading-path name.

## Results

The baselines of all three cells are measured on the baseline tree; every candidate cell is pending. `kv_write_scaled` now has a kernel package (`crates/ops/kernels/kv_write_scaled/triton/package.toml`), so the FP8 KV cell is pending like the other two rather than held. A baseline is a measurement of one side, not a comparison: no interval and no verdict is claimed until the same cell has run at the candidate commit on the same checkpoint, sample set and settings.

| Cell | Commit | Requests | Correct | Accuracy | `decided_by_marker` / `decided_by_boxed` / `unparseable` |
| --- | --- | --- | --- | --- | --- |
| `qwen3-moe-gsm8k-bf16` | `0b47282ee` | 1319/1319 | 1248 | 0.946171 | not recorded at this commit; the 1249-row run at `5c6c06a0e` reported 1182 / 136 / 2 |
| `qwen3-moe-gsm8k-nvfp4` | `becb9e9a5` | 1319/1319 | 1233 | 0.934799 | 1080 / 239 / 4 |
| `qwen3-moe-gsm8k-bf16-fp8-kv` | `f1de185a2` | 1319/1319 | 1250 | 0.947688 | 1171 / 148 / 2 |

The three form counts are what each cell's `summary.json` reports, not a partition of the requests into four buckets: a prediction the extractor cannot parse is still counted under the form that was tried.

The reference of each confirmation is measured again, in the same campaign as the candidate, from the tree that produced these numbers (`f1de185a2`, a descendant of every cell's first-run commit); a historical number is not a denominator, because a run at another commit differs by whatever else changed between the commits. For the record, the three baselines were measured at three commits of the baseline branch, because each cell was first made to run at the commit that fixed its own path: `0b47282ee` (BF16, 1248), `becb9e9a5` (NVFP4, 1233) and `f1de185a2` (FP8 KV, 1250, together with a second BF16 measurement of 1251). The spread across the three BF16 measurements — 1248, 1249, 1251 — is three rows at n = 1319, where the standard error of a proportion near 0.95 is about 0.0061 (about sixteen rows at 95 %), so they are one implementation measured three times, not three implementations. Those three commits do not differ in serving code: `git diff f1de185a2 98596c50a` over `uniserve/` and `uniserve_eval/` is empty — the commits between them change Rust test code only — so all three baselines run the same Python attention path and the same evaluation code, and the commit spread is a record of when each cell was first made to run rather than a difference between the runs. A baseline bundle records its own commit in its `summary.json`; a bundle from another campaign is not a reference unless the cell is measured again there.

The acceptance test is `uniserve_eval compare`, which computes `|d| ≤ 1.959963985 · sqrt(p1(1−p1)/n1 + p2(1−p2)/n2)` from **both** measured proportions at n = 1319 and confidence 0.95. What is fixed before any candidate result is the formula, that confidence and that n; the width is not a single number, because it depends on the candidate's accuracy as well. Near the baselines it is ±0.0172 against BF16 (about 23 rows) and ±0.0188 against NVFP4 (about 25 rows), and it widens as the candidate's accuracy falls: ±0.0184 at p2 = 0.93 and ±0.0203 at 0.90. A candidate inside the measured interval is accepted; one outside it is a measured regression, not noise, and a rejection argued against the near-baseline figures rather than the measured interval would be wrong.
