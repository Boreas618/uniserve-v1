# Testing

The repository's checks run from the `justfile`: `just lint`, `just test-rust`, `just test-python-fast` (no GPU) and the GPU suites `just test-python-gpu` and `just test-python-sanitizer`. This page states what the GPU suites need from the host.

## Kernel package checks

The numerical checks of the kernel packages under `crates/ops/kernels` live in `tests/python/kernels/<family>/`, one copy per family of operators (a check shared by several operators takes the operator with `--op`), and each family's `test_<family>.py` runs every one of them as a subprocess. The ones marked `unit and not gpu` run on Triton's interpreter with no device visible and are part of `just test-python-fast`; the ones marked `gpu` compile and launch their kernel on a device and are skipped on a host without one.

| Variable | What it names | Tests |
| --- | --- | --- |
| `UNISERVE_KERNEL_MODULES` | A directory of built modules, `<op>/<backend>/` holding the module of the descriptor the check's defaults describe (a check comparing two builds reads named subdirectories, such as `dense_attention/f32/causal`) | the module checks of the native and cubin packages, among them the `topk_softmax` router's launch |
| `UNISERVE_KERNEL_KEYS`, `UNISERVE_KERNEL_STORE` | A key fixture and the store built from it | `test_kernel_arguments.py`, which compares every built cell's arguments with its operator's |

## GPU end-to-end tests

The HTTP tests under `tests/python/e2e` start real servers on the host's GPUs. `just test-python-gpu` runs every test marked `e2e and gpu`; `just test-python-sanitizer` runs those marked `missing_state` with every main process under compute-sanitizer's memcheck. Both call `tools/gpu_e2e.sh`, which the nightly workflows call as well:

- it builds the release `uniserve` executable, the scheduling policy precompiler and `uniserve-manager` once, and every process of every test deployment runs that one build;
- it puts the NCCL the project pins on the loader's path (`tools/check_nccl.sh`) and states the Python environment (`VIRTUAL_ENV`) a main process takes CUPTI from;
- it runs pytest with one worker per GPU (pytest-xdist), and each case leases the GPUs it serves on, so independent cases run beside one another on different GPUs while a case that observes host-wide state runs alone; the cases of a module that serves one deployment for all of them stay on one worker;
- it writes pytest's temporary directory, which holds every server's log, and a JUnit report to `artifacts/e2e/<UTC time>/`.

Every module starts its own Program Server with every kernel package of the tree installed, and each deployment of the module resolves its programs there; the Program Server compiles a kernel the first time a deployment asks for it, so a module's first deployment takes longer.

A plain `pytest` skips the GPU end-to-end tests: they run only when `UNISERVE_RUN_GPU_E2E=1`, which the recipes set. A test whose host asset is unset is skipped with a reason naming the variable, so a run on a host without an asset reports what it did not test; a variable that names a path that does not exist fails the test.

### Host assets

These name the checkpoints and inputs of the host. Set the ones the host has; the tests that need the others are skipped.

| Variable | What it names | Tests |
| --- | --- | --- |
| `UNISERVE_FFMPEG_DIR` | The shared FFmpeg 8.1.2 + libx264 prefix every build links (`crates/media/native/build.sh` builds one, see `crates/media/README.md`); required to build | all |
| `UNISERVE_QWEN3_MODEL` | A Qwen3-MoE BF16 checkpoint directory (`Qwen/Qwen3-30B-A3B`) | every Qwen3 module |
| `UNISERVE_QWEN3_UPDATED_MODEL` | Another directory with the same checkpoint files and other weight values, such as a copy whose shards are renormalized, for weight updates | `test_epoch_changes_http.py`, `test_live_updates_http.py` |
| `UNISERVE_H3_MODEL` | A FastH3 checkpoint directory (`FastVideo/FastVideo-FastH3-8-Step-V2`; the [FastH3 guide](minimax_h3/minimax_h3.md) lists the supported ones) | the FastH3 modules |
| `UNISERVE_MINIMAX_H3_MODEL` | The MiniMax-H3 diffusers root | `test_minimax_h3_http.py`, `test_minimax_h3_conditions_http.py` |
| `UNISERVE_FASTH3_ADAPTER` | A FastH3 Preview v1 adapter directory (`vsa-datafree` or `dense-datafree` of `FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA`), served over `UNISERVE_MINIMAX_H3_MODEL` | `test_minimax_h3_http.py` |
| `UNISERVE_MINIMAX_H3_OMNIREF` | A FastH3 OmniRef export (`FastVideo/FastVideo-FastH3-Omni-8-Step-V1`), whose omitted components are read from the Hugging Face cache (`HF_HOME`) at the base revision it pins | `test_minimax_h3_conditions_http.py` |
| `UNISERVE_MINIMAX_H3_INPUTS` | The protocol inputs directory holding `media/`, the reference files of `uniserve_eval/workloads/minimax_h3` ([MiniMax-H3 workloads](minimax_h3/evaluation.md)) | `test_minimax_h3_conditions_http.py` |
| `UNISERVE_MOONCAKE_MASTER`, `MOONCAKE_STORE_LIBRARY` | A running Mooncake master (`host:port`) and the `libmooncake_store.so` its clients load | the Mooncake cases of `test_epoch_changes_http.py` |
| `UNISERVE_AFD_URL`, `UNISERVE_AFD_MODEL` | A running attention-expert disaggregated deployment and its served model name | `test_disaggregated_http.py` |
| `UNISERVE_H3_FOLLOWER`, `UNISERVE_H3_LEADER_ADDRESS` | A second host and this host's address that the second reaches, for the layouts of two hosts ([Two hosts](#two-hosts)) | `test_h3_parallel_http.py`, `test_cross_host_http.py`, `test_peer_kv_cache_http.py`, `test_prefill_decode_http.py` |
| `UNISERVE_FOLLOWER_DEVICE` | The second host's GPU ordinal the two-host cases serve on (`0` unless stated) | `test_peer_kv_cache_http.py`, `test_prefill_decode_http.py` |
| `UNISERVE_NIXL_LIBRARY` | NIXL's `libnixl_capi.so` (`docs/nixl.md`), which the test-owned Storage Servers of the peer reads load, and whose directory both hosts' Storage Servers of a two-host prefill/decode deployment find it in | `test_peer_kv_cache_http.py`, `test_prefill_decode_http.py` |

### Run settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `UNISERVE_E2E_DEVICES` | every GPU `nvidia-smi` lists | The GPU ordinals the run may lease, comma-separated; a case that needs more is skipped |
| `UNISERVE_E2E_WORKERS` | the number of those GPUs | pytest-xdist workers |
| `UNISERVE_E2E_ARTIFACTS` | `artifacts/e2e/<UTC time>` | Where the run's temporary directory and JUnit report go |
| `MAX_JOBS` | `16` | Kernel compilations a Program Server runs at once |
| `UNISERVE_E2E_SET` | none | `PATH=VALUE` assignments every test deployment adds to its options |
| `UNISERVE_E2E_TIME_SCALE` | `1` | The factor every wait of a test is stretched by |
| `UNISERVE_PROGRAM_SERVER_PYTHON` | `crates/program-server/python/.venv/bin/python` | The Program Server's interpreter, which `uv sync --locked --project crates/program-server/python` creates |
| `UNISERVE_PROGRAM_TARGET` | the served GPU's `sm_{major}{minor}a` | The target the Program Server builds kernels for |
| `UNISERVE_PROGRAM_REBIND_PACKAGE`, `UNISERVE_PROGRAM_RECAPTURE_PACKAGE`, `UNISERVE_PROGRAM_POLICY_PACKAGE`, `UNISERVE_PROGRAM_WRONG_PACKAGE` | the packages under `crates/ops/program-fixtures` | The packages `test_program_update_http.py` installs |
| `SANITIZER_LOGS`, `COMPUTE_SANITIZER` | the run's `memcheck/`, `compute-sanitizer` | Where memcheck writes and which sanitizer runs (`just test-python-sanitizer`) |

`tools/gpu_e2e.sh` sets `UNISERVE_RUN_GPU_E2E`, `UNISERVE_BINARY`, `UNISERVE_WORKER_EXECUTABLE`, `UNISERVE_SCHEDULING_PRECOMPILE`, `UNISERVE_MEMCHECK_TARGET` and `VIRTUAL_ENV` itself. A case's lease sets `UNISERVE_WORLD_DEVICES` to the GPUs it holds, comma-separated, and every test takes its devices from it (`http_helpers.granted_devices`); `cuda:N` ranks in a test's deployment index those GPUs.

### Two hosts

The eight-rank layouts of `test_h3_parallel_http.py` run on this host and a second one when `UNISERVE_H3_FOLLOWER` names the second as an ssh destination (reachable without a password prompt) and `UNISERVE_H3_LEADER_ADDRESS` names this host's address as the second reaches it. The second host must hold the same checkout at the same path with the same build, which a filesystem both hosts mount provides, and four free GPUs: the test leases this host's GPUs, not the second's. The second host's daemon runs in the test's environment, and `UNISERVE_FOLLOWER_LAUNCHER` names a command it runs under, such as one that sets up the second host's environment. Without the two variables the eight-rank layouts are skipped, naming the device count they need. `test_cross_host_http.py` serves a tensor group of two ranks, one on each host, with a standby epoch over both: it kills the second host's serving main process while a request streams and checks that the standby epoch is promoted and the request continues; it needs one free GPU on each host (the second host's first). `test_peer_kv_cache_http.py` serves Qwen3 on both hosts: one deployment per host whose second reads the first's host KV cache pages through the Storage Servers it starts with a TCP endpoint and NIXL (`UNISERVE_NIXL_LIBRARY`, and `UCX_NET_DEVICES` naming the hosts' usable RDMA ports); it needs one free GPU on each host (`UNISERVE_FOLLOWER_DEVICE` on the second). Its converted case serves the second host's deployment with an FP8 KV cache, which loads the first host's BF16 pages by converting them, and compares its answer with its answer to the same prompt over pages it wrote itself by vLLM's `check_logprobs_close` rule. The two-host case of `test_prefill_decode_http.py` serves Qwen3's prefill group on this host and its decode group on the second, whose Transfer Server reads the prefill rank's cache through NIXL (`UNISERVE_NIXL_LIBRARY`, `UCX_NET_DEVICES`); it needs one free GPU on each host.

### A build shared by several hosts

Hosts that mount the same checkout can run different modules at once from one build: build on one host (`tools/gpu_e2e.sh` builds when it runs there, or `cargo build --release -p uniserve --bin uniserve -p scheduling --bin scheduling-precompile -p manager --bin uniserve-manager`), and on the others name the executables with `UNISERVE_BINARY` and `UNISERVE_SCHEDULING_PRECOMPILE`, which makes `tools/gpu_e2e.sh` build nothing. The executable records the FFmpeg prefix it was linked against as its run path, so that prefix must lie on the shared filesystem too. Each host still needs the Rust toolchain with its dependencies fetched (`cargo fetch --locked`), because a module's Program Server builds the scheduling policy for `wasm32-unknown-unknown` offline, and the Program Server's environment.

### Example

```bash
export UNISERVE_FFMPEG_DIR=/opt/ffmpeg-8.1.2
export UNISERVE_QWEN3_MODEL=/models/Qwen3-30B-A3B
export UNISERVE_H3_MODEL=/models/FastVideo-FastH3-8-Step-V2
just test-python-gpu
```

Arguments after the recipe's own go to pytest through `tools/gpu_e2e.sh`, so one module runs with `tools/gpu_e2e.sh -m "e2e and gpu" tests/python/e2e/test_live_updates_http.py`.
