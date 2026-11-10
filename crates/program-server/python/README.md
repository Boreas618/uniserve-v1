# The Program Server's Python environment

No serving process carries a Python interpreter: in a deployment, Python appears here alone, as the compiler of kernel programs. The Program Server runs every program package's compile script with this environment's interpreter; it never imports it, and the environment never runs in the main process or on the serving path.

It is **not** the repository's `.venv`, which holds the `uniserve` executable with the development, test and evaluation tools; this one holds only what compiles a kernel, so a package's compile script cannot depend by accident on a package another environment happens to have.

## Creating it

```text
uv sync --project crates/program-server/python --python /usr/bin/python3.12
```

This writes `python/.venv` (git-ignored) from `pyproject.toml` and `uv.lock`. The Program Server's `--python` option (`program_config.python` for a server the daemon starts) names the interpreter, `crates/program-server/python/.venv/bin/python`; a relative path is made absolute against the directory the command was started in.

A server without the interpreter is not an error: the Triton and CuTe DSL toolchains are recorded as missing with that reason, and only a build that needs them fails.

## What it locks

| distribution | version | used by |
|---|---|---|
| `triton` | 3.7.1 | `TritonSource`: `triton.compile` writes the cubin and its metadata JSON |
| `nvidia-cutlass-dsl[cu13]` | 4.6.2 | `CuteDslSource`: `cute.compile` and `export_to_c` write the host launcher that is linked into a native module |
| `flashinfer-python` | 0.6.18 | the trtllm-gen launch code (`paged_attention/trtllm`, the routed NVFP4 products), the `moe_sort` and `moe_unpermute` sources, and the MegaMoE kernel (the `megamoe` pin) |
| `flash-attn-4[cu13]` | 4.0.0b30 | the FlashAttention-4 package |
| `torch` | 2.13.0 | required by the two above; it is never imported by the Program Server itself |

A package names the distributions it imports and their versions in its `[toolchain] packages`, by distribution name (`flash-attn-4`, `flashinfer-python`) and at the version this lock fixes; changing any of them is a new installation and a new revision. The Program Server refuses to build a package whose declared distribution this environment does not have at that version, naming each disagreement, as it refuses a toolchain of another version.

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

The script reads the descriptor's parameter values out of its own manifest through `uniserve_program_server.package.fields`, so the keys a package accepts are exactly the descriptor range its `[kernel.descriptors]` declares: a dimension's enumerated values, or, for a dimension declared by a constraint table (`{ multiple = 16 }`, `{ power_of_two = true }`, `{ separator = "x" }`, `{ bits = 32 }`; the grammar is `Dimension` in `crates/program-server/src/manifest.rs`), the text up to the next `_` when it meets the constraint. A manifest with several `[[kernel.descriptors]]` shapes is read against each in turn.

A Triton or CuTe DSL package's script states only what one descriptor of its kernel needs, `spec(fields, parameters, target)`, and hands it to the shared driver:

```python
from uniserve_program_server import triton as driver   # or cute_dsl

def spec(fields, parameters, target):
    return {"kernel": ..., "signature": {...}, "constexprs": {...}, "options": {...}}

if __name__ == "__main__":
    driver.main(spec)
```

`uniserve_program_server.triton` compiles the spec with `triton.compile` on an explicit target and writes `module.cubin` and `metadata.json` with the launch signature; `uniserve_program_server.cute_dsl` compiles it with `cute.compile`, exports the host launcher and links it with the package's `src/entry.cpp` into `module.so`. Both need no GPU. The drivers belong to this environment, not to a package's revision: like the compilers this lock fixes, a change to them is a change of the environment.

## Pinned third-party sources

A package that builds against code it does not own names that code by a pin of `uniserve_program_server.fetch` instead of carrying a copy. Every pin is addressed the same way: its identity is the sha256 of a manifest listing each file with its own sha256; the manifest is published beside the files or kept in `src/uniserve_program_server/pins/<name>.sha256`; every file is verified against it before use and cached content-addressed under `~/.cache/uniserve/program-server/artifacts/<identity>/` (or `$UNISERVE_ARTIFACT_CACHE`), so a cached file is reused offline and one that does not verify is fetched again. Pins differ only in where a file comes from.

| pin | what | origin | identity |
|---|---|---|---|
| `trtllm-gen-fmha` | TensorRT-LLM's FMHA cubins and generated table, as FlashInfer pins them (`ArtifactPath.TRTLLM_GEN_FMHA`) | NVIDIA's kernel artifact repository (`$UNISERVE_ARTIFACT_REPOSITORY`), its published `checksums.txt` | `d79b5c51…` |
| `trtllm-gen-bmm` | trtllm-gen's batched-GEMM cubins, table and export headers (the routed NVFP4 MoE products) | the same repository | `011635d3…` |
| `deepep` | DeepEP's elastic dispatch and combine kernels, deepseek-ai/DeepEP commit `b306af06afd4` | each file from the commit's raw tree | `ee22833b…` |
| `nccl` | NCCL 2.30.4's headers, the device API DeepEP is built against | the `nvidia-nccl-cu13==2.30.4` wheel the serving environment locks, its sha256 pinned per architecture; read out of the archive, which is not kept | `67a3f54b…` |
| `megamoe` | FlashInfer's fused MegaMoE CuTe DSL kernel | the `flashinfer-python==0.6.18` distribution this environment locks | `dae28b34…` |

```text
.venv/bin/python -m uniserve_program_server.fetch PIN [--file NAME ...] [--all] [--out DIR] [--offline]
```

fetches the named files (`--all`: all of them; the FMHA library has 35 324 cubins) into the cache, stages them into `DIR` as symbolic links, and prints a JSON description of the pin. A build calls the same functions in process: `fetch.fetch(pin)` returns the verified cache directory a compile includes from, `fetch.stage` links (or, for a build that patches them, copies) files into the build's staging directory, and `fetch.apply_patch` applies a package's own patch exactly, refusing a hunk that does not match where it says. The megamoe package carries `patches/megamoe.patch` (the tanh-approximated GELU gate and the token exchange's peer-metadata stride) and applies it to a staged copy.

### The trtllm-gen packages

`uniserve_program_server.trtllm` holds what the packages launching trtllm-gen kernels share: FlashInfer's data tree (the launch code), staging a pinned library, and the whole build of the routed NVFP4 products, which `gather_gemm_nvfp4` and `route_gemm_nvfp4` call with their entry symbol. The launch code compiles against the staged artifact with nvcc alone, without a GPU:

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
