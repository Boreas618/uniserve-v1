# The Program Server's Python environment

Python appears in UniServe in one place: here, as the compiler of kernel programs (design section 6.2). The Program Server runs every program package's compile script with this environment's interpreter; it never imports it, and the environment never runs in the main process or on the serving path.

It is **not** the repository's `.venv`. The serving environment holds the full Python runtime for the models; this one holds only what compiles a kernel, so a package's compile script cannot depend by accident on a package the serving environment happens to have.

## Creating it

```text
uv sync --project crates/program-server/python --python /usr/bin/python3.12
```

This writes `python/.venv` (git-ignored) from `pyproject.toml` and `uv.lock`. The Program Server's `--python` option names the interpreter and defaults to `python/.venv/bin/python` under the repository root; `Toolchains::discover` resolves that relative path from the workspace root, so both `uniserve program-server serve` started at the root and one started inside `crates/program-server` find it.

A host without the environment is not an error of the Program Server: the Triton and CuTe DSL toolchains are recorded as missing with that reason, and only a build that needs them fails.

## What it locks

| distribution | version | used by |
|---|---|---|
| `triton` | 3.7.1 | `TritonSource`: `triton.compile` writes the cubin and its metadata JSON (design 5.3) |
| `nvidia-cutlass-dsl[cu13]` | 4.6.2 | `CuteDslSource`: `cute.compile` and `export_to_c` write the host launcher that is linked into a native module |
| `flashinfer-python` | 0.6.18 | the CuTe DSL kernels a package imports (`moe_sort`, `moe_unpermute`, the grouped GEMM) |
| `flash-attn-4[cu13]` | 4.0.0b30 | the FlashAttention-4 package |
| `torch` | 2.13.0 | required by the two above; it is never imported by the Program Server itself |

A package names the distributions it imports and their versions in its `[toolchain] packages`; changing any of them is a new installation and a new revision (design 5.2).

## How a package's compile script is run

The Program Server runs `<interpreter> <package>/<source.script>` with the package root as the working directory and this environment:

| variable | value |
|---|---|
| `UNISERVE_OP`, `UNISERVE_BACKEND` | the operator and backend of the module's store path |
| `UNISERVE_DESCRIPTOR` | the descriptor key, such as `dtype_bf16_width_2048_layout_contiguous` |
| `UNISERVE_TARGET` | the target of the module's store path, such as `sm_100a` |
| `UNISERVE_PACKAGE` | the package root |
| `UNISERVE_OUT` | the build's output directory; the script writes `module.cubin` and its metadata JSON there, or `module.so` for a native module |
| `MAX_JOBS` | the compile queue's limit on parallel compilers |

The script reads the descriptor's parameter values out of its own manifest, so the keys a package accepts are exactly the descriptor range its `[kernel.descriptors]` declares.

## The trtllm-gen artifact

The attention package that launches TensorRT-LLM's FMHA kernels (design section 9, the native decode attention) compiles against a generated header and loads cubins that no Python distribution carries: FlashInfer pins them as a directory in NVIDIA's kernel artifact repository (`flashinfer.artifacts.ArtifactPath.TRTLLM_GEN_FMHA`, `CheckSumHash.TRTLLM_GEN_FMHA`). `uniserve_program_server.trtllm_gen` fetches that artifact, verifies it and caches it:

```text
.venv/bin/python -m uniserve_program_server.trtllm_gen --out DIR [--all | --cubin NAME] [--offline] [--verify]
```

- The pin's identity is the sha256 of its `checksums.txt` (`d79b5c51…`); the manifest is verified against that hash, and every file against the manifest.
- The cache is content-addressed: `~/.cache/uniserve/program-server/artifacts/<sha256>/`, or `$UNISERVE_ARTIFACT_CACHE`. It works offline for what it holds, and a file that does not verify is fetched again.
- `--out DIR` stages the generated header (`include/flashInferMetaInfo.h`) and the checksums into `DIR` as symbolic links; `--all` stages all 35 324 cubins, `--cubin NAME` one of them. The command prints a JSON description: the cache, the include directory, every cubin's name, what it staged.
- A package's build script runs this against `$UNISERVE_OUT` to get the harness it compiles against, and copies the cubins its launch code needs into `$UNISERVE_OUT` so they are published with the module.

Compiling the launch code against the artifact was proved on this host with nvcc 13.0.48, no GPU:

```text
nvcc -c -std=c++20 -arch=sm_100a \
  -DTLLM_GEN_FMHA_CUBIN_PATH='"2d6a5a029eefcc388ec0ceb87efb55d8bcce5c3c/fmha/trtllm-gen/"' \
  -DTLLM_GEN_FMHA_METAINFO_HASH='"d79b5c51fc8597fac57dae0da4afa114fb2014575e4ec3df099ad856d97cabc3"' \
  -I <flashinfer>/data/include -I <staged>/include \
  -I <flashinfer>/data/cutlass/include \
  -I <flashinfer>/data/cccl/libcudacxx/include \
  -I <flashinfer>/data/cccl/cub -I <flashinfer>/data/cccl/thrust \
  -o probe.o probe.cu
```

The two `-D` macros are required (`fmhaKernels.cuh` static-asserts on them). The cubins are loaded at run time through `flashinfer::trtllm_cubin_loader::getCubin`, which the module implements: FlashInfer's `cubin_loader.h` provides the C callback mechanism (`FlashInferSetCubinCallback`/`FlashInferSetCurrentCubin`), and the module decides where the cubins are.

