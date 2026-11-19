# Testing

The repository's checks run from the `justfile`: `just lint`, `just test-rust`, `just test-python-fast` (no GPU) and the GPU suites `just test-python-gpu` and `just test-python-sanitizer`. This page states how the Python tests are organized and what the GPU suites need from the host.

## The Python tests

The Python tests run in the repository's environment (`uv sync --locked --extra dev --extra test`), which installs the root project's executables (`uniserve` and `uniserve-manager`, a release build) and the test-support package `uniserve-testing` (`tests/support`). The package is a pytest plugin, which pytest loads through its `pytest11` entry point, and the helpers the test directories share; a test module imports them by name (`uniserve_testing.servers`, `uniserve_testing.chat`, ...), never by its place in the tree, and pytest imports each test module on its own (`--import-mode=importlib`). The tests run the `uniserve` the environment installed, so a change to the Rust sources is tested once `uv sync` has installed it again.

A plain `pytest` deselects the GPU end-to-end cases (`addopts` in `pyproject.toml` holds `-m "not (e2e and gpu)"`); a run that states its own marker expression replaces it: `-m "unit and not gpu"` is the fast suite, `-m "e2e and gpu"` the GPU suite.

The plugin's options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--kernel-device {interpreter,cuda}` | `interpreter` | Where the kernel checks run |
| `--e2e-devices` | every GPU NVML reports | The GPUs the end-to-end cases may lease, as comma-separated device indices; a case that needs more is skipped |
| `--engine` | the tree's engine package | The engine package every test deployment runs, by its backend name |
| `--serve-config FILE` | none | A TOML file of options every test deployment's `config.toml` states over its own |
| `--time-scale` | `1` | The factor every wait of an end-to-end case is stretched by |
| `--peer-host` | none | The second host of a two-host case, as an ssh destination ([Two hosts](#two-hosts)) |
| `--mooncake-master HOST:PORT` | none | A running Mooncake master, the remote store of the cases that keep host blocks in one; they are skipped without it |

## Kernel package checks

The numerical checks of the kernel packages under `crates/ops/kernels` live in `tests/python/kernels/<family>/`. A check is a pytest module, `test_<op>.py`, whose test functions launch a package's kernel in the test process the way its operator launches it and hold the result to a reference within the operator's bound, one test per case; `tests/python/kernels/conftest.py` states the fixtures and conventions every family shares, and the families' shared reference arithmetic and call mirrors are modules of `uniserve_testing.kernels`.

Where the kernels run is one choice per session, `--kernel-device`:

```text
pytest tests/python/kernels -m "unit and not gpu"   # Triton's interpreter, no device: part of just test-python-fast
pytest tests/python/kernels --kernel-device cuda    # compiled for and launched on the host's CUDA device
```

The interpreter is the default. A test marked `gpu` needs a device (a libdevice call, a dependent launch, a built native module, a device's own rounding) and is skipped with that reason on the interpreter; a test without the mark runs in both modes, on host tensors for the interpreter and on the device otherwise. In the `cuda` mode a host without a CUDA device skips every check.

A check that launches what a Triton package's launch table states for a descriptor key asks the environment's `uniserve build-server launch --source` with the repository, which creates the repository's Build Server environment with uv and evaluates the table there; a `gpu` check of a native or cubin module loads the module a Build Server of the session builds.

## GPU end-to-end tests

The HTTP tests under `tests/python/e2e` start real servers on the host's GPUs. `just test-python-gpu` runs every test marked `e2e and gpu`; `just test-python-sanitizer` runs those marked `missing_state` under compute-sanitizer's memcheck. The nightly workflow (`.github/workflows/gpu-nightly.yml`) runs both recipes:

- every process of every test deployment runs the `uniserve` executable the environment installed; the run's Build Server serves from the repository (`--source`): it compiles kernels with the toolchain environment uv creates from the repository, and installs the repository's packages before it serves;
- its serving processes take the NVIDIA driver, NCCL and CUPTI from the host's default loader path, and FFmpeg as the build found it through pkg-config: the host provides the releases the [README](../README.md#requirements) names, and a process refuses an older driver, NCCL or CUPTI by name;
- the recipe runs pytest on one pytest-xdist worker per GPU the run may lease (`-n auto`, which the plugin answers with the count of `--e2e-devices`, else of the GPUs NVML reports), and each case leases the GPUs it serves on (`tests/python/e2e/conftest.py`), so independent cases run beside one another on different GPUs while a case that observes host-wide state runs alone; the cases of a module that serves one deployment for all of them stay on one worker (`--dist loadgroup`). Leases are advisory locks a case takes under one allocator lock, blocking on a GPU until its holder releases it;
- the recipe writes pytest's temporary directory, which holds every server's log, and a JUnit report to `artifacts/e2e/<UTC time>/`, or to the directory `just e2e=DIR test-python-gpu` names.

Each test session (each pytest-xdist worker) starts one Build Server of its own (`uniserve_testing.builds`), whose store, under the session's work directory (`store/packages` and `store/modules`, the layout `build-server build --store` writes), holds every package of the tree installed (the kernel packages, the scheduling policy, the processing program and the engine). Every deployment of the session runs its own Build Server, which its daemon starts, and its `build_config` (`builds.deployment_config`) names the session's store as a read-only store (`read_only_stores`) beside the session server's source tree, targets and limits: the deployment's Build Server finds the tree's packages installed there and installs none of them again. What it compiles it keeps in its own store root, the deployment's directory of the session's data directory for deployments (`XDG_DATA_HOME` and `XDG_CACHE_HOME` of every daemon of the session, under `deployments/` of the session's work directory), so the cases of a session that serve a deployment of one name compile its kernels and its engine once, the first case taking longer, and deployments of other names never read them. A deployment's package root is its case's own (`<name>.packages` in the case's directory), so a revision a case installs at its deployment's Build Server (`server.build_server`, `builds.BuildServer.of`), as the update cases do, is selected by that deployment's next epoch and by no other case's deployment. A case's runtime directory ends with the deployment's Storage Server and Build Server stopped (`deployments.stop_deployment_servers`), so no server of one case holds a lock the next case's server takes.

A case starts its deployments through the `deployments` fixture (`uniserve_testing.deployments`): a case states each deployment's options as the tree of its `config.toml` (`uniserve_testing.packages`), the fixture writes its policy package into the case's directory, named for the deployment, with the run's Build Server and options added, and starts `uniserve serve` on it. Each daemon serves on the GPUs the case leases (which it sees as `cuda:0` onwards), with the deployments' persistent state in the case's own directory (`XDG_STATE_HOME`), so no case starts from a revision another left, and under a runtime directory of its own (`XDG_RUNTIME_DIR`, which the caller of a UniServe server creates and names), where it starts the deployment's Storage Server and Build Server (`uniserve/deployments/<name>/` there), which serve that deployment alone and which the case stops when the deployment ends. A case changes a serving deployment as an operator does, with the next package (`PUT /admin/policy`, `uniserve_testing.admin`), which states every write (pausing, sleeping, flushing the prefix cache, profiling sessions, retiring the deployment's storage, a hot restart by a change of the daemon's own options), and reads what it executes from the deployment's observation store with SQL (`/admin/query`): the gauges the leader states each second of its replicas and GPU quotas, the counters the engines report, and the events of every request, revision, process and log line; an update commits by itself at the end of the trial its package states (`update_config.trial`), so a case that observes or faults a trial states one long enough for it, and one that does not states a short one. A launched process runs with a declared environment: the search path, the user's home and the standard Hugging Face variables, and nothing else of the test's environment. Every server a test starts binds its listeners on ports it picks and reports them in its readiness record (`docs/admin.md`, Listeners), which the harness reads from the descriptor it passes (`--ready-fd`): a case's client URL, admin socket, join endpoint and Storage Server endpoints come from the record, never from a port the case chooses, and a case waits for the record that states the server serves rather than probing it. A case waits for what an event announces (an update's phase, an engine's lifecycle change, a profiling session's state, the node serving) by reading the daemon's clock (`SELECT CAST(now() AS BIGINT)`) before it causes it and then the events after that moment, and reads a gauge in a whole second after the moment it waits for. A deployment that transfers between worker groups or hosts names the transfer calibration of its GPUs, which the case measures once with `uniserve storage-server calibrate` (`Deployments.calibration`) into a directory of its own in the system's temporary directory, the host's own storage; without one its Storage Server runs no Transfer Server.

### Assets

The checkpoints and adapters the cases serve are released Hugging Face repositories at pinned commits, listed once in `uniserve_testing.assets`; a deployment names its model as its package's `model` states it (`<owner>/<name>@<revision>[/<folder>]`) and the daemon resolves it into the standard Hugging Face cache, as a case does for a file it reads itself. A host without network access populates the cache beforehand and sets the standard `HF_HUB_OFFLINE`; a mirror is named by `HF_ENDPOINT`, a private repository's token by `HF_TOKEN`, and the cache's location by `HF_HOME`.

| Asset | Repository | Cases |
| --- | --- | --- |
| `qwen3` | `Qwen/Qwen3-30B-A3B` | every Qwen3 module |
| `qwen3.6` | `Qwen/Qwen3.6-35B-A3B` | `test_processing_program_http.py` |
| `fasth3` | `FastVideo/FastVideo-FastH3-8-Step-V2` | the FastH3 modules |
| `minimax-h3` | `MiniMaxAI/MiniMax-H3` | `test_minimax_h3_http.py`, `test_minimax_h3_conditions_http.py` |
| `fasth3-omniref` | `FastVideo/FastVideo-FastH3-Omni-8-Step-V1` | `test_minimax_h3_conditions_http.py` |
| `fasth3-adapter`, `fasth3-adapter-student` | the `vsa-datafree` folder of `FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA`, and the inference contract of `FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` beside it | the adapter case of `test_minimax_h3_http.py` |

Two assets are published under no repository yet, and the cases that need them are skipped with that reason: a second Qwen3-30B-A3B checkpoint with other weight values (the weight-update cases' target) and the MiniMax-H3 protocol's reference media.

### Two hosts

A two-host case runs on this host and a second one when `--peer-host` names the second as an ssh destination (reachable without a password prompt); without it the case is skipped. The second host runs the `uniserve` its login installs, with its own environment: the harness runs installed programs there over ssh, as argument vectors with no variable of this host's environment and no shell text. A deployment that spans both hosts lists them in its package's `hosts` by their host names (`uname -n`, which the harness reads on each host), this host first, and its leader binds the join port the case picks (`admin_config.join`) on every interface; once the leader's readiness record states its listeners, the harness writes the same package, with a `build_config` of the second host's own (the repository's path as its source tree and a package root in its runtime directory), under a runtime directory of its own there and starts `uniserve serve` on it, the second host's daemon joins the leader at this host's name, and the leader serves once it joined. Each host therefore resolves the other's host name: on hosts whose names no DNS resolves, such as the pods of one cluster whose search domains do not hold each other's names, each host's `/etc/hosts` lists the other's name and address. The second host's daemon resolves its modules at the deployment's Build Server there, which it starts and which installs the tree's packages from this repository's path with the toolchains it finds on that host's login search path, so the search path an ssh command gets there holds `cargo` and `rustc` (the repository's Rust toolchain, through rustup) as this host's does; a cargo package (the engine, a scheduling policy) whose toolchain it lacks is not installed there, and the deployment's start fails naming it. Until a deployment's package revisions reach another host's Build Server through the Build Server's protocol, the second host must mount this repository at the same path. Its Hugging Face cache holds the case's checkpoints at their pinned revisions (or it reaches the Hub), and a deployment that names a transfer calibration has the second host's own measured at the same path there: each host's Storage Server reads its own GPUs' calibration from the one path the package names, so the path lies on each host's own storage rather than in the case's directory, which both hosts may mount.

### Under compute-sanitizer

`just test-python-sanitizer` runs the deployments with the engine package `memcheck` (`tests/python/sanitizer`, `--engine memcheck`), more device storage left free and a longer `worker_config.watchdog_timeout` (`--serve-config tests/python/sanitizer/config.toml`) and every wait stretched (`--time-scale 20`). The package runs the tree's engine under compute-sanitizer's memcheck through a wrapper that reads its target and log directory from its environment, which the declared launch environment no longer carries: the suite runs again once a declared engine launcher replaces the package.

### Example

```bash
uv sync --locked --extra dev --extra test
just test-python-gpu
```

The recipes run pytest in the repository's environment through `uv run --locked`, which first brings the environment up to date with the lockfile. Arguments after a recipe's name go to pytest, so one module runs with `just test-python-gpu tests/python/e2e/test_live_updates_http.py`, and a run that may lease four GPUs only with `just test-python-gpu --e2e-devices 0,1,2,3`.
