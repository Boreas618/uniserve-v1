# Testing

The repository's checks run from the `justfile`: `just lint`, `just test-rust`, `just test-python-fast` (no GPU) and the GPU suites `just test-python-gpu` and `just test-python-sanitizer`. This page states what the GPU suites need from the host.

## Kernel package checks

The numerical checks of the kernel packages under `crates/ops/kernels` live in `tests/python/kernels/<family>/`, one copy per family of operators (a check shared by several operators takes the operator with `--op`), and each family's `test_<family>.py` runs every one of them as a subprocess. The ones marked `unit and not gpu` run on Triton's interpreter with no device visible and are part of `just test-python-fast`; the ones marked `gpu` compile and launch their kernel on a device and are skipped on a host without one.

| Variable | What it names | Tests |
| --- | --- | --- |
| `UNISERVE_KERNEL_MODULES` | A directory of built modules, `<op>/<backend>/` holding the module of the descriptor the check's defaults describe (a check comparing two builds reads named subdirectories, such as `dense_attention/f32/causal`) | the module checks of the native and cubin packages, among them the `topk_softmax` router's launch |
| `UNISERVE_BINARY` | The base executable, whose `build-server fields` reads a descriptor key against its package's manifest; `uniserve` on the search path when unset. `just test-python-fast` builds it and sets the variable | the tensor primitives' checks (`tests/python/kernels/tensor`), which launch the module a key names |
| `UNISERVE_KERNEL_KEYS`, `UNISERVE_KERNEL_STORE` | A key fixture and the store built from it | `test_kernel_arguments.py`, which compares every built cell's arguments with its operator's |

## GPU end-to-end tests

The HTTP tests under `tests/python/e2e` start real servers on the host's GPUs. `just test-python-gpu` runs every test marked `e2e and gpu`; `just test-python-sanitizer` runs those marked `missing_state` with every main process under compute-sanitizer's memcheck. Both call `tools/gpu_e2e.sh`, which the nightly workflows call as well:

- it builds the release `uniserve` executable and `uniserve-manager` once, and every process of every test deployment runs that one build; each test module's Build Server compiles kernels with its toolchain environment, whose program `uniserve-build-server` must be on the search path (`uv sync --locked --project crates/build-server/python` creates the environment);
- its serving processes take the NVIDIA driver, NCCL and CUPTI from the host's default loader path, and FFmpeg as the build found it through pkg-config: the host provides the releases the [README](../README.md#requirements) names, and a process refuses an older driver, NCCL or CUPTI by name;
- it runs pytest with one worker per GPU (pytest-xdist), and each case leases the GPUs it serves on, so independent cases run beside one another on different GPUs while a case that observes host-wide state runs alone; the cases of a module that serves one deployment for all of them stay on one worker;
- it writes pytest's temporary directory, which holds every server's log, and a JUnit report to `artifacts/e2e/<UTC time>/`.

Every module starts its own Build Server with every kernel package of the tree installed, and each deployment of the module resolves its modules there; the Build Server compiles a kernel the first time a deployment asks for it, so a module's first deployment takes longer.

Every server a test starts binds its listeners on ports it picks and reports them in its readiness record (`docs/admin.md`, Listeners), which the helpers read from the descriptor they pass it (`--ready-fd`): a test's client URL, admin socket and join endpoint come from the record (`http_helpers.server_process`, `Server`), never from a port the test chooses, and the test waits for the record that states `serving` rather than probing `/health`. Each deployment runs under a runtime directory of its own (`XDG_RUNTIME_DIR`, which every daemon requires), whose Storage Server is stopped with `uniserve storage-server stop` when the test ends.

A plain `pytest` skips the GPU end-to-end tests: they run only when `UNISERVE_RUN_GPU_E2E=1`, which the recipes set. A test whose host asset is unset is skipped with a reason naming the variable, so a run on a host without an asset reports what it did not test; a variable that names a path that does not exist fails the test.

### Host assets

These name the checkpoints and inputs of the host. Set the ones the host has; the tests that need the others are skipped.

| Variable | What it names | Tests |
| --- | --- | --- |
| `UNISERVE_QWEN3_MODEL` | A Qwen3-MoE BF16 checkpoint directory (`Qwen/Qwen3-30B-A3B`) | every Qwen3 module |
| `UNISERVE_QWEN3_UPDATED_MODEL` | Another directory with the same checkpoint files and other weight values, such as a copy whose shards are renormalized, for weight updates | `test_epoch_changes_http.py`, `test_live_updates_http.py` |
| `UNISERVE_H3_MODEL` | A FastH3 checkpoint directory (`FastVideo/FastVideo-FastH3-8-Step-V2`; the [FastH3 guide](minimax_h3/minimax_h3.md) lists the supported ones) | the FastH3 modules |
| `UNISERVE_MINIMAX_H3_MODEL` | The MiniMax-H3 diffusers root | `test_minimax_h3_http.py`, `test_minimax_h3_conditions_http.py` |
| `UNISERVE_FASTH3_ADAPTER` | A FastH3 Preview v1 adapter directory (`vsa-datafree` or `dense-datafree` of `FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA`), served over `UNISERVE_MINIMAX_H3_MODEL` | `test_minimax_h3_http.py` |
| `UNISERVE_MINIMAX_H3_OMNIREF` | A FastH3 OmniRef export (`FastVideo/FastVideo-FastH3-Omni-8-Step-V1`), whose omitted parts (text encoder, tokenizer, processor and VAEs) are read from the Hugging Face cache (`HF_HOME`) at the base revision it pins | `test_minimax_h3_conditions_http.py` |
| `UNISERVE_MINIMAX_H3_INPUTS` | The protocol inputs directory holding `media/`, the reference files of `uniserve_eval/workloads/minimax_h3` ([MiniMax-H3 workloads](minimax_h3/evaluation.md)) | `test_minimax_h3_conditions_http.py` |
| `UNISERVE_MOONCAKE_MASTER` | A running Mooncake master (`host:port`); the Storage Server's connector opens `libmooncake_store.so` from the loader's default search path, where the Mooncake package installs it | the Mooncake cases of `test_epoch_changes_http.py` |
| `UNISERVE_AFD_URL`, `UNISERVE_AFD_MODEL` | A running attention-expert disaggregated deployment and its served model name | `test_disaggregated_http.py` |
| `UNISERVE_H3_FOLLOWER`, `UNISERVE_H3_LEADER_ADDRESS` | A second host and this host's address that the second reaches, for the cases that serve on two hosts ([Two hosts](#two-hosts)) | `test_h3_parallel_http.py`, `test_cross_host_http.py`, `test_peer_kv_cache_http.py`, `test_prefill_decode_http.py` |
| `UNISERVE_FOLLOWER_DEVICE` | The second host's GPU ordinal the two-host cases serve on (`0` unless stated) | `test_peer_kv_cache_http.py`, `test_prefill_decode_http.py` |

### Run settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `UNISERVE_E2E_DEVICES` | every GPU `nvidia-smi` lists | The GPU ordinals the run may lease, comma-separated; a case that needs more is skipped |
| `UNISERVE_E2E_WORKERS` | the number of those GPUs | pytest-xdist workers |
| `UNISERVE_E2E_ARTIFACTS` | `artifacts/e2e/<UTC time>` | Where the run's temporary directory and JUnit report go |
| `MAX_JOBS` | `16` | Kernel compilations a Build Server runs at once |
| `UNISERVE_E2E_SET` | none | `PATH=VALUE` assignments every test deployment adds to its options |
| `UNISERVE_E2E_TIME_SCALE` | `1` | The factor every wait of a test is stretched by |
| `UNISERVE_MODULE_TARGET` | the served GPU's `sm_{major}{minor}a` | The target the Build Server builds kernels for |
| `UNISERVE_REBIND_PACKAGE`, `UNISERVE_RECAPTURE_PACKAGE`, `UNISERVE_POLICY_PACKAGE`, `UNISERVE_WRONG_PACKAGE` | the kernel packages under `crates/ops/package-fixtures` and the policy package `crates/scheduling/policies/fixture` | The packages `test_generation_update_http.py` installs |
| `SANITIZER_LOGS`, `COMPUTE_SANITIZER` | the run's `memcheck/`, `compute-sanitizer` | Where memcheck writes and which sanitizer runs (`just test-python-sanitizer`) |

`tools/gpu_e2e.sh` sets `UNISERVE_RUN_GPU_E2E`, `UNISERVE_BINARY` and `UNISERVE_MEMCHECK_TARGET` itself. A case's lease sets `UNISERVE_WORLD_DEVICES` to the GPUs it holds, comma-separated, and every test takes its devices from it (`http_helpers.granted_devices`); `cuda:N` ranks in a test's deployment index those GPUs.

### Two hosts

A two-host case runs on this host and a second one when `UNISERVE_H3_FOLLOWER` names the second as an ssh destination (reachable without a password prompt) and `UNISERVE_H3_LEADER_ADDRESS` names this host's address as the second reaches it; without the two variables it is skipped. The second host must hold the same checkout at the same path with the same build, which a filesystem both hosts mount provides. A deployment that spans both hosts starts its leader here as node `rank-0` with `--nnodes 2` and `--dist-init-addr` at that address and port 0, so the leader binds its join endpoint on a port it picks; once the leader's readiness record states that endpoint, the second host's daemon joins it there as node `rank-1` and runs the ranks placed on that host's GPUs, and the leader serves once it joined. That daemon runs in the test's environment under a runtime directory of its own, which `mktemp` creates there, so the Storage Server it starts serves that deployment alone; its readiness records cross the ssh connection, and `UNISERVE_FOLLOWER_LAUNCHER` names a command it runs under, such as one that sets up the second host's environment. `test_h3_parallel_http.py` leases four GPUs of this host and uses the second host's first four, which must be free: `test_replicas_on_two_hosts_serve_complete_media` serves `configs/minimax_h3/fasth3-ulysses4x2-two-host.toml`, one Ulysses4 replica on each host, with `--max-running-requests 4`, and requires complete media for eight concurrent text-to-video requests; the eight-rank layouts of `test_block_bindings_release_cancelled_requests` place the ranks past this host's four GPUs on the second host's (`rank-1/cuda:0` to `rank-1/cuda:3`), so their worker groups span both hosts, and require the leader to refuse the deployment and exit as it starts, naming at least one product that would pass between the two hosts' main processes with the worker groups that write and read it; without the two variables those layouts are skipped, naming the device count they need. `test_cross_host_http.py` serves a tensor group of two ranks, one on each host, with a standby epoch over both: it kills the second host's serving main process while a request streams and checks that the standby epoch is promoted and the request continues; it needs one free GPU on each host (the second host's first). `test_peer_kv_cache_http.py` serves Qwen3 on both hosts: one deployment per host whose second reads the first's host KV cache pages through the hosts' Storage Servers, which the deployments' daemons start with a TCP endpoint on a port each picks (`storage_config.tcp`; the first deployment's readiness record states its server's as the peer endpoint the second names) and the NIXL package (`docs/nixl.md`, with `UCX_NET_DEVICES` naming the hosts' usable RDMA ports); it needs one free GPU on each host (`UNISERVE_FOLLOWER_DEVICE` on the second). Its converted case serves the second host's deployment with an FP8 KV cache, which loads the first host's BF16 pages by converting them, and compares its answer with its answer to the same prompt over pages it wrote itself by vLLM's `check_logprobs_close` rule. The two-host case of `test_prefill_decode_http.py` serves Qwen3's prefill group on this host and its decode group on the second, whose Transfer Server reads the prefill rank's cache through the NIXL package (`docs/nixl.md`, `UCX_NET_DEVICES`); it needs one free GPU on each host.

### A build shared by several hosts

Hosts that mount the same checkout can run different modules at once from one build: build on one host (`tools/gpu_e2e.sh` builds when it runs there, or `cargo build --release -p uniserve --bin uniserve -p manager --bin uniserve-manager`), and on the others name the executable with `UNISERVE_BINARY`, which makes `tools/gpu_e2e.sh` build nothing. An executable built with the media crate's `vendored` feature records the FFmpeg installation in its target directory as its run path, so that target directory must lie on the shared filesystem too; one built against a host's FFmpeg needs that FFmpeg on every host. Each host still needs the Rust toolchain with its dependencies fetched (`cargo fetch --locked`), because a module's Build Server builds the scheduling policy for `wasm32-unknown-unknown` offline, and the Build Server's environment.

### Example

```bash
export UNISERVE_QWEN3_MODEL=/models/Qwen3-30B-A3B
export UNISERVE_H3_MODEL=/models/FastVideo-FastH3-8-Step-V2
just test-python-gpu
```

Arguments after the recipe's own go to pytest through `tools/gpu_e2e.sh`, so one module runs with `tools/gpu_e2e.sh -m "e2e and gpu" tests/python/e2e/test_live_updates_http.py`.
