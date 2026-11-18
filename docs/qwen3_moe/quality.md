# Qwen3-MoE GSM8K quality confirmation

This page specifies how UniServe confirms Qwen3-MoE quality on GSM8K: `uniserve-eval` runs **each cell** on the same checkpoint, sample set and sampling settings twice — once on the existing implementation's tree (`f1de185a2`), once on a candidate build — and accepts the candidate when the accuracy difference lies within the two-sided 95% confidence interval of the difference of two binomial proportions at that sample size. The confirmation covers three cells: the BF16 checkpoint, its ModelOpt NVFP4 export, and the BF16 checkpoint with FP8 KV pages. Each cell is compared with itself; two different cells are not a comparable pair, because they differ in the checkpoint or the KV cache dtype, and the comparison requires both runs to share their recorded provenance.

This page fixes the workload, sample set, request shape, extraction rule and comparison; the evaluator applies the extraction rule and the comparison, and each run states the rest in the profile it is given, which [Running the confirmation](#running-the-confirmation) writes for each cell. Nothing is chosen per run. The [results](#results) are pending.

## Checkpoint and cells

| Cell | Checkpoint | KV cache dtype | `provenance` |
| --- | --- | --- | --- |
| `qwen3-moe-gsm8k-bf16` | `/workspace/models/Qwen3-30B-A3B` | `bfloat16` | `precision="bfloat16"`, `kv_cache_dtype="bfloat16"` |
| `qwen3-moe-gsm8k-nvfp4` | local `nvidia/Qwen3-30B-A3B-NVFP4` export | `bfloat16` | `precision="nvfp4"`, `kv_cache_dtype="bfloat16"` |
| `qwen3-moe-gsm8k-bf16-fp8-kv` | `/workspace/models/Qwen3-30B-A3B` | `float8_e4m3fn` | `precision="bfloat16"`, `kv_cache_dtype="float8_e4m3fn"` |

All three cells also record `checkpoint = "Qwen3-30B-A3B"`; the column shows the fields that differ.

The BF16 and FP8 KV cells serve the Qwen3-MoE checkpoint at `/workspace/models/Qwen3-30B-A3B`. The NVFP4 cell serves a local copy of `nvidia/Qwen3-30B-A3B-NVFP4`, a ModelOpt unified export whose own `config.json` declares its quantization, so a host without the export cannot run the NVFP4 cell. The cells differ in the checkpoint path and the KV cache dtype, and the FP8 KV cell also names an attention provider (see [FP8 KV pages](#fp8-kv-pages)). Each cell's `provenance` is its identity in the comparison.

No cell states `dtype` or `quantization`: the checkpoint's own representation is used, and `quantization_config` and its precision presets are retired in favour of the checkpoint representation and the `quantization` field. The serving capacity is build configuration: `mem_fraction_static = 0.80` and `kv_cache_config.max_tokens = 65536`, and a host may lower either without making a cell's two runs incomparable, as long as both runs of the cell state the same values.

## Dataset and sample set

The dataset is `openai/gsm8k`'s test split at revision `740312add88f781978c0658806c59bc2815b9866`, file `main/test-00000-of-00001.parquet`: 1319 rows read in file order, without shuffling or filtering. Each row's question is the user message verbatim and its reference answer supplies the gold value. `dataset_path` may name a local Parquet or JSON Lines file, or a directory holding the split; only an unset path downloads the pinned file from the Hub. The point states `num_prompts = 1319`, and every run records the selected rows' identity as `{"count": 1319, "sha256": "6403e64f54dff14d84e6f6839f9dc8d7178c6500e4d791a826d05e3ff00cbdb8"}`.

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

Each cell is measured twice with the same point: a reference run that serves the existing implementation (`f1de185a2`), and a candidate run that serves the new one. Each cell is compared with itself — the same checkpoint, the same KV cache dtype — because two different cells differ in the fields below and a cross-cell pair is reported as not comparable. Both runs must be completed, valid `gsm8k` runs over the same `selected_rows` identity whose count is 1319, with `load.num_prompts` equal to that count and the same `workload.sampling`, `workload.provenance`, `workload.model` and resolved daemon model argument. A pair that fails a premise is reported as not comparable instead of a verdict.

For baseline accuracy `p1 = correct1 / n1` and candidate accuracy `p2 = correct2 / n2`, with `n1 = n2 = 1319`:

```text
d  = p2 - p1
SE = sqrt(p1 (1 - p1) / n1 + p2 (1 - p2) / n2)
interval = d ± 1.959963985 * SE
```

The factor is the two-sided 95% normal quantile the evaluator computes (Wichura's AS241, `1.959963984540054` at 0.975; `1.959963985` rounded here); the acceptance is `|d| <=` that factor times `SE`, equivalently the interval contains zero. `uniserve-eval compare` prints the report as JSON with both accuracies and counts, `d`, `SE`, the interval, the verdict and both provenances, and exits 0 when accepted, 2 when rejected, and 1 when a bundle is missing or the runs are not comparable.

## FP8 KV pages

The FP8 KV cell is the BF16 checkpoint with `kv_cache_config.dtype = float8_e4m3fn`. It is the only cell that names an attention provider, and **the two implementations name different ones**, because a provider is a name in each implementation's own registry:

- **The baseline** (the Python worker's tree) uses `torch`, the only provider that reads per-block FP8 pages at head dimension 128; the automatic selection raises for them, and every native paged provider refuses a per-block FP8 cache. Its reference run states `--set 'attention_backend="torch"'`.
- **The candidate** reads the planes in its own attention layer, not through a provider: `nn::modules::Attention::forward_paged` writes and reads the layer's pages through `nn::functional::kv_write_` and `nn::functional::paged_attention`, which accept a cache whose two planes are both `F8E4M3` over a BF16 compute type (`crates/nn/src/functional/attention.rs:808-826`) and require each FP8 page's scale (`page_scale`, `:461-501`), and the model's attention is always that paged operator. `attention_backend` does not choose a provider on this side: the option (unset by default, `crates/daemon/src/config/mod.rs`) names the package of the `paged_attention` operator in the epoch's module set (`attention_packages`, `crates/daemon/src/epoch.rs:656-690`), and an unset option names none, so the Build Server selects for each kernel key the package of highest priority that declares it. Of the operator's two packages, `trtllm` reads a cache of the queries' own type only, and `triton` (`crates/ops/kernels/paged_attention/triton/package.toml`) is the one that reads E4M3 planes with their page scales, so `auto` selects `triton` for this cell. The cell therefore states `auto`, a name the path accepts, rather than a provider name; `torch`, the baseline's, is not one of the candidate's attention packages.

The BF16 and NVFP4 cells name no attention backend, so each implementation selects the provider it has.

## Running the confirmation

Both sides run the evaluator of this checkout, `tools/eval`; only the server each run starts differs. The reference side is a **second checkout of the baseline tree**, `git worktree add "$HOME/gsm8k-reference" f1de185a2` with its own `uv sync`, so the two implementations never share an environment: the daemon resolves its other processes through `$VIRTUAL_ENV/bin/uniserve`, so each server runs with its own tree's environment. The candidate side installs its root package's release binary into this checkout's environment, and `cmp` proves the installed copy is the build; `--help` does not.

A cell's point is written at use from the cell's values in [Checkpoint and cells](#checkpoint-and-cells); `cell` below writes it for one side, the server command naming that side's executable and environment. Both sides serve the same options, one policy package per cell and side: the candidate serves the package's directory, and the reference tree, whose `serve` reads its options from a configuration file, the package's `config.toml` (`--config`):

```bash
cargo build --release -p eval        # target/release/uniserve-eval
EVAL=$PWD/target/release/uniserve-eval
GSM8K=                               # a local copy of the test split, or empty to download it

# cell NAME CHECKPOINT PRECISION KV_DTYPE SIDE [ATTENTION]: the profile of one cell's run on SIDE
# (reference or candidate), on the server of that side's tree, with the attention provider ATTENTION
# when it is given. Its policy package is artifacts/qwen3_moe/packages/SIDE/NAME.
cell() {
  local package=$PWD/artifacts/qwen3_moe/packages/$5/$1 tree=$PWD attention= command
  [ -n "${6:-}" ] && attention="attention_backend = \"$6\""
  mkdir -p "$package"
  cat > "$package/config.toml" <<EOF
model_dir = "$2"
served_model_name = "Qwen3-30B-A3B"
tensor_parallel_size = 1
max_running_requests = 64
max_num_batched_tokens = 16384
mem_fraction_static = 0.80
$attention
admin_config = { host = "127.0.0.1", port = 18081 }
scheduler_config = { chunked_prefill_size = 16384 }
kv_cache_config = { max_tokens = 65536, block_size = 64, dtype = "$4" }
EOF
  command="\"$tree/.venv/bin/uniserve\", \"serve\", \"$package\""
  if [ "$5" = reference ]; then
    tree=$REFERENCE
    command="\"$tree/.venv/bin/uniserve\", \"serve\", \"--config\", \"$package/config.toml\""
  fi
  cat <<EOF
[servers.$1]
host = "127.0.0.1"
port = 18081
command = [$command]
[servers.$1.environment]
CUDA_VISIBLE_DEVICES = "0"
VIRTUAL_ENV = "$tree/.venv"

[benchmarks.$1]
server = "$1"
task = "gsm8k"
model = "Qwen3-30B-A3B"
dataset = "gsm8k"
dataset_revision = "740312add88f781978c0658806c59bc2815b9866"
dataset_path = "$GSM8K"
load = { num_prompts = 1319, request_rate = inf, max_concurrency = 64, warmup_requests = 1, seed = 42 }
sampling = { temperature = 0.0, top_p = 1.0, sampling_seed = 42, ignore_eos = false, max_tokens = 1024, stream = true, extra_body = { chat_template_kwargs = { enable_thinking = false } } }
metrics = { accuracy = "higher" }
provenance = { checkpoint = "Qwen3-30B-A3B", precision = "$3", kv_cache_dtype = "$4" }
EOF
}

REFERENCE=$HOME/gsm8k-reference
cargo build --release
install -D -m 755 target/release/uniserve .venv/bin/uniserve
cmp target/release/uniserve .venv/bin/uniserve   # silent: the installed copy is the build

BF16=/workspace/models/Qwen3-30B-A3B
NVFP4=<local directory of the NVFP4 export>
for side in reference candidate; do
  fp8=torch
  # The candidate reads FP8 pages in its own attention layer and names no provider.
  [ $side = candidate ] && fp8=
  out=$PWD/artifacts/qwen3_moe/gsm8k-$side
  mkdir -p "$out"
  cell qwen3-moe-gsm8k-bf16 $BF16 bfloat16 bfloat16 $side > "$out/bf16.toml"
  cell qwen3-moe-gsm8k-nvfp4 $NVFP4 nvfp4 bfloat16 $side > "$out/nvfp4.toml"
  cell qwen3-moe-gsm8k-bf16-fp8-kv $BF16 bfloat16 float8_e4m3fn $side $fp8 > "$out/fp8-kv.toml"
  for profile in bf16:qwen3-moe-gsm8k-bf16 nvfp4:qwen3-moe-gsm8k-nvfp4 fp8-kv:qwen3-moe-gsm8k-bf16-fp8-kv; do
    $EVAL plan --config "$out/${profile%%:*}.toml" "${profile#*:}"
    $EVAL run --config "$out/${profile%%:*}.toml" "${profile#*:}" --output-root "$out" || break 2
  done
done

# One verdict per cell: the same cell on both sides.
for cell in qwen3-moe-gsm8k-bf16 qwen3-moe-gsm8k-nvfp4 qwen3-moe-gsm8k-bf16-fp8-kv; do
  $EVAL compare artifacts/qwen3_moe/gsm8k-reference/$cell artifacts/qwen3_moe/gsm8k-candidate/$cell
done
```

Before a cell, the server's one-request probe must return a token: a server that cannot serve is reported as that, never as a quality difference, and no cell starts on a failed probe. Each side runs its three cells serially, one server at a time. `run` refuses a non-empty bundle directory, so a cell measured again is written under a new root rather than over the one it replaces.

## Results

The baselines of all three cells are measured on the baseline tree; every candidate cell is pending. `kv_write_scaled` now has a kernel package (`crates/ops/kernels/kv_write_scaled/triton/package.toml`), so the FP8 KV cell is pending like the other two rather than held. A baseline is a measurement of one side, not a comparison: no interval and no verdict is claimed until the same cell has run at the candidate commit on the same checkpoint, sample set and settings.

| Cell | Commit | Requests | Correct | Accuracy | `decided_by_marker` / `decided_by_boxed` / `unparseable` |
| --- | --- | --- | --- | --- | --- |
| `qwen3-moe-gsm8k-bf16` | `0b47282ee` | 1319/1319 | 1248 | 0.946171 | not recorded at this commit; the 1249-row run at `5c6c06a0e` reported 1182 / 136 / 2 |
| `qwen3-moe-gsm8k-nvfp4` | `becb9e9a5` | 1319/1319 | 1233 | 0.934799 | 1080 / 239 / 4 |
| `qwen3-moe-gsm8k-bf16-fp8-kv` | `f1de185a2` | 1319/1319 | 1250 | 0.947688 | 1171 / 148 / 2 |

The three form counts are what each cell's `summary.json` reports, not a partition of the requests into four buckets: a prediction the extractor cannot parse is still counted under the form that was tried.

The reference of each confirmation is measured again, in the same campaign as the candidate, from the tree that produced these numbers (`f1de185a2`, a descendant of every cell's first-run commit); a historical number is not a denominator, because a run at another commit differs by whatever else changed between the commits. For the record, the three baselines were measured at three commits of the baseline branch, because each cell was first made to run at the commit that fixed its own path: `0b47282ee` (BF16, 1248), `becb9e9a5` (NVFP4, 1233) and `f1de185a2` (FP8 KV, 1250, together with a second BF16 measurement of 1251). The spread across the three BF16 measurements — 1248, 1249, 1251 — is three rows at n = 1319, where the standard error of a proportion near 0.95 is about 0.0061 (about sixteen rows at 95 %), so they are one implementation measured three times, not three implementations. Those three commits do not differ in serving code: `git diff f1de185a2 98596c50a` over `uniserve/` and `uniserve_eval/` is empty — the commits between them change Rust test code only — so all three baselines run the same Python attention path and the same evaluation code, and the commit spread is a record of when each cell was first made to run rather than a difference between the runs. A baseline bundle records its own commit in its `summary.json`; a bundle from another campaign is not a reference unless the cell is measured again there.

The acceptance test is `uniserve-eval compare`, which computes `|d| ≤ 1.959963985 · sqrt(p1(1−p1)/n1 + p2(1−p2)/n2)` from **both** measured proportions at n = 1319 and confidence 0.95. What is fixed before any candidate result is the formula, that confidence and that n; the width is not a single number, because it depends on the candidate's accuracy as well. Near the baselines it is ±0.0172 against BF16 (about 23 rows) and ±0.0188 against NVFP4 (about 25 rows), and it widens as the candidate's accuracy falls: ±0.0184 at p2 = 0.93 and ±0.0203 at 0.90. A candidate inside the measured interval is accepted; one outside it is a measured regression, not noise, and a rejection argued against the near-baseline figures rather than the measured interval would be wrong.
