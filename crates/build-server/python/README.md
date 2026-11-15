# The Build Server's Python environment

No serving process carries a Python interpreter: in a deployment, Python appears here alone, as the compiler of kernel packages. The Build Server compiles every Triton, CuTe DSL and CUDA package through this environment; it never imports it, and the environment never runs in the main process or on the serving path.

It is **not** the repository's `.venv`, which holds the `uniserve` executable with the development, test and evaluation tools; this one holds only what compiles a kernel, so a package's build module cannot depend by accident on a package another environment happens to have.

## Creating it

```text
uv sync --locked --project crates/build-server/python
```

This writes `python/.venv` (git-ignored) from `pyproject.toml` and `uv.lock` with an interpreter that meets `requires-python`, and installs the environment's program, `uniserve-build-server`, in its `bin` directory. The Build Server runs that program from its search path, as any installed program is run, so a host puts the environment's `bin` directory on the search path of the processes that start Build Servers (a daemon, or `uniserve build-server serve`).

A server without the environment is not an error: the Triton, CuTe DSL and nvcc toolchains are recorded as missing with the reason, and only a build that needs them fails.

## The toolchain description

```text
uniserve-build-server toolchain --json
```

prints what this environment compiles with, which the Build Server reads once when it starts (`crates/build-server/src/toolchain.rs`):

| entry | how it is found |
|---|---|
| `interpreter` | the environment's own interpreter, which runs every compilation |
| `driver` | the SHA-256 of this package's files, read through `importlib.resources`: a module is stored under the driver it was built through, so an edited driver, pin or target description is another identity |
| `kernels` | the SHA-256 of the kernel packages' shared sources (`uniserve_kernels`, `crates/ops/python`, which this environment installs and the packages import), read the same way, so an edit of what a package imports is another identity |
| `distributions` | every installed distribution and its version, through `importlib.metadata`; a module's identity carries their digest, so a relock is another identity |
| `triton`, `cute_dsl` | the versions of the locked `triton` and `nvidia-cutlass-dsl` distributions |
| `cuda` | the CUDA toolkit of the release the lock fixes (the release of the locked `nvidia-cuda-runtime`, such as `13.0`), through the `cudart-<release>` and `cuda-<release>` pkg-config modules NVIDIA's distribution packages register: its root, `nvcc`, include and library directories and the driver stub's directory; nvcc's version is read from the macros it predefines (`__CUDACC_VER_MAJOR__` and its siblings) |
| `cudnn` | cuDNN of the release the lock fixes (the major release of the locked `nvidia-cudnn-cu13`, such as `9`), as NVIDIA's distribution packages (`libcudnn9-dev-cuda-13`) install it on the host's default search paths: the host C++ compiler preprocesses an include of `cudnn_version.h`, whose line markers name its directory and whose macros its version (`9.12.0`), and resolves `libcudnn.so` on the linker's search path. A module links the library by its soname, and the serving process's loader finds the same file by the same default search, so no path is recorded in a module and none is patched at run time |
| `host_compiler` | the host C++ compiler the program `c++` names on the search path, resolved to its file |

A toolchain the host lacks is `{"missing": "<reason>"}` instead of `{"available": ...}`. A package states no toolchain version: the one this lock fixes builds it, and every revision records it. A package's `[toolchain] packages` are checked against the distributions, and one the environment lacks refuses the build by name.

## What it locks

| distribution | version | used by |
|---|---|---|
| `triton` | 3.7.1 | the Triton driver: `triton.compile` writes the cubin and its metadata JSON |
| `nvidia-cutlass-dsl[cu13]` | 4.6.2 | the CuTe DSL driver: `cute.compile` and `export_to_c` write the host launcher that is linked into a native module |
| `flashinfer-python` | 0.6.18 | the trtllm-gen launch code (`paged_attention/trtllm`, the routed NVFP4 products), the `moe_sort` and `moe_unpermute` sources, and the MegaMoE kernel (the `megamoe` pin) |
| `flash-attn-4[cu13]` | 4.0.0b30 | the FlashAttention-4 package |
| `urllib3`, `patch-ng` | 2.8.0, 1.19.1 | the pinned sources' fetcher and its patch of a staged copy |
| `torch` | 2.13.0 | required by FlashInfer and FlashAttention-4; never imported by the Build Server itself |

A package names the distributions it imports in its `[toolchain] packages`, by distribution name (`flash-attn-4`, `flashinfer-python`); their versions are this lock's, and a relock is another build identity for every module. The kernel packages' shared sources, `uniserve-kernels` (`crates/ops/python`), are a distribution of this environment too, installed from the repository.

## How a package is compiled

The Build Server compiles one descriptor by running

```text
<interpreter> -m uniserve_build_server compile
```

in the package's source directory `src/`, with the compile request as a JSON object on standard input (`uniserve_build_server.request.Request`); `-m` puts that directory first on the import path, so the package's modules import as any program's own. The request is the whole interface, and a compilation reads no environment variable:

| field | value |
|---|---|
| `op`, `backend`, `package` | the package and its installed root |
| `descriptor`, `fields` | the descriptor key and its fields as the Build Server read them against the manifest's `[[kernel.descriptors]]` (`Descriptors::typed_fields` in `crates/build-server/src/manifest.rs`, the one reader of the key grammar), in the key's order, each typed as its dimension declares it: a number, a list of numbers for an extent tuple (a `separator` constraint's value, or one of an enumeration of shapes such as a GEMM's `tile = ["128x128", "256x128"]`), the float of a bit pattern, a boolean for a flag, or a name |
| `parameters` | the manifest's `[parameters]` |
| `source`, `module`, `unit` | the source kind, the module the compilation imports (a Triton package's kernel module `<op>`, or the build module `build`; none for a CUDA package's declared compilation), and the entry unit (`[source] unit`, `src/entry.cpp` for a CuTe DSL package unless it names another) a driver compiles |
| `launch`, `compile`, `entries` | a Triton package's launch table (`[launch]`, `crates/build-server/src/launch.rs`), a CUDA package's declared compilation (`[compile]`, `crates/build-server/src/compile.rs`), and a native module's C entries as the manifest resolves them (`launch` is the operator's `uniserve_<op>` unless the manifest names another) |
| `target` | the target's name and its architecture's hardware facts (`targets.toml`: capability, warp size, families, shared and tensor memory, the warpgroup matrix instructions, dependent launch, FP8), so a package states requirements instead of hardware facts; `str(target)` is its name |
| `output` | the directory the module's files are written into; a dot directory in it is the compilation's own scratch and is not published |
| `jobs` | the compiler processes the compilation may run, the compile queue's limit |
| `cache` | the Build Server's cache of pinned sources |
| `toolchain` | the toolchain description the Build Server checked, as above |

The source kind selects how the package states its compilation:

- **Triton**: no build module. The manifest's launch table names the kernel, a function of `src/<op>.py`, and binds each of its parameters from literals and the descriptor's fields: each launch argument's type (a pointer to the element type a field names, a type a field's value selects, or a Triton type), each compile-time parameter's value, the warp count, the aligned arguments and further `triton.compile` options; a `derive` function of the source module computes fields the table reads where a value follows from several. `uniserve_build_server.triton` takes the signature's order from the kernel's own parameters, spells element types by its one table, binds the kernel's `PDL` parameter from the descriptor's `pdl` field and the target's dependent launch (a kernel without one never launches dependently), compiles with `triton.compile` on the target and writes `module.cubin` and `metadata.json` with the launch signature;
- **CuTe DSL**: the module states `spec(request)`, handed the compile request, returning the exported launcher's `name`, a `compile` callable (`cute_dsl.kernel(launcher, args, options)`: the package's launcher, its own kernel or a vendor's kernel class it constructs with an explicit configuration) and optional `definitions` of the entry unit by macro name; a compilation describes the launcher's runtime arguments with `cute_dsl.pointer(dtype, align)` (a fake device pointer of the alignment the kernel relies on), `cute_dsl.tensor(dtype, rank, leading, align)` (a fake device tensor whose extents and strides but the unit one are runtime words) and `cute_dsl.scalar(dtype)` (a runtime word, whose value is never compiled in), and reads its tuning data (the manifest's `tuning`) with `request.tuned(file)`, the first `[[configuration]]` whose `descriptors` the descriptor's fields equal; `uniserve_build_server.cute_dsl` states the target to the DSL, exports the launcher, and compiles the manifest's entry unit (with the shared entry header of `uniserve_kernels/include` on its include path) and links it with the launcher and the CuTe DSL's dialect runtime into `module.so` through `uniserve_build_server.cuda`'s `objects` and `link`;
- **CUDA**: the manifest declares the compilation in `[compile]`, which the manifest reader types and validates and `uniserve_build_server.cuda` runs. It states the translation units (`units`, relative to the package root), the C++ standard (`standard`), each preprocessor definition (`[compile.definitions]`: a literal, a field's value, the value a field selects or a parameter, as a launch table states a constant; a boolean is defined as 1 or 0), the libraries a native module links beside the CUDA runtime (`libraries`: `cuda`, the driver API through the toolkit's stub; `cublasLt`; `cudnn`, the distribution the toolchain description names), the package's own options (`host_contraction = false` for host code that must round every operation), and, for a cubin module, its kernel (`[compile.kernel]`: the symbol, the warps, the dynamic shared memory and the signature in parameter order, each type as a launch table types an argument), from which the driver writes `metadata.json` in the fields `ops::CubinMetadata` reads, with the definitions the module was compiled with. The driver compiles every unit with the toolchain description's nvcc at `-O3` for the target, with the package's `include/` on the include path, at most `jobs` compiler processes at once, into objects in a scratch directory of the output, links them into `module.so` (or compiles the one unit into `module.cubin`), and prints each command and the compiler's whole output into the build's log. A package whose compilation is computed from what it compiles against (a vendor kernel table it selects from, a launcher the CuTe DSL exports, a pinned source tree) keeps a build module, `src/build.py`, whose `compile(request)` decides its units, definitions and include directories and calls the driver's routines (`cuda.Unit`, `cuda.native`, `cuda.cubin`, `cuda.objects`, `cuda.link`, `cuda.scratch`, `cuda.includes`, `cuda.library`) instead of writing a command line.

The Build Server then checks the module its manifest's form requires: a cubin whose metadata names its kernel, or a native library that defines every C entry it is launched through.

`python -m uniserve_build_server` and `uniserve-build-server` are one program.

## A package's launches outside a build

```text
uniserve build-server launch <package> --target <target> [<key>...]
```

reads each descriptor key against a Triton package's manifest with the Build Server's reader and prints one JSON line per key: `{"key": ..., "launch": {...}}` with the kernel's name, its signature in parameter order, the compile-time values, the aligned arguments' attributes by name and `triton.compile`'s options, as `uniserve_build_server.triton.stated` states them for a build of that key, or `{"key": ..., "refused": "..."}` with the reason the package states none (a key its dimensions do not declare, or one its table or derivation refuses). Given no key, it reads keys from standard input, one per line, and answers each before it reads the next, so a caller keeps one command per package running and asks it key by key. The command runs this environment's `uniserve-build-server launch`, which imports the package's kernel module once, from the package's source directory by its location; a kernel check launches the named kernel from its own import of the same source (`tests/python/kernels/conftest.py`).

## Pinned third-party sources

A package that builds against code it does not own names that code by a pin of `uniserve_build_server.fetch` instead of carrying a copy. Every pin is one data file, `src/uniserve_build_server/pins/<name>.toml`: its identity, the sha256 of a manifest listing each file with its own sha256, and its origin. The manifest is published beside the files or kept beside the pin's data file as `<name>.sha256`; every file is verified against it before use and cached content-addressed in the Build Server's cache (`<cache>/<identity>/<path>`), so a cached file is reused offline and one that does not verify is fetched again. A build stages a cached file into its own workspace as a hard link, both being directories of the Build Server's one cache root.

| pin | what | origin |
|---|---|---|
| `trtllm-gen-fmha` | TensorRT-LLM's FMHA cubins and generated table, as FlashInfer pins them | NVIDIA's kernel artifact repository, with its published `checksums.txt` |
| `trtllm-gen-bmm` | trtllm-gen's batched-GEMM cubins, table and export headers (the routed NVFP4 MoE products) | the same repository |
| `deepep` | DeepEP's elastic dispatch and combine kernels, deepseek-ai/DeepEP commit `b306af06afd4` | each file from the commit's raw tree |
| `nccl` | NCCL 2.30.4's headers, the device API DeepEP is built against | the `nvidia-nccl-cu13==2.30.4` wheel the serving environment locks, whose headers are the same for every architecture; read out of the archive, which is not kept |
| `megamoe` | FlashInfer's fused MegaMoE CuTe DSL kernel | the `flashinfer-python` distribution this environment locks |
| `cutlass-nvfp4-gemm` | CUTLASS's CuTe DSL tutorial NVFP4 GEMM (`nvfp4_gemm_0.py`), the kernel of `nvfp4_gemm` | the CUTLASS examples among the data files of the `flashinfer-python` distribution this environment locks |

```text
uniserve-build-server provision PIN --cache DIR [--file NAME ...] [--all] [--jobs N] [--offline]
```

fetches the named files (`--all`: all of them; the FMHA library has 35 324 cubins) into a cache and prints a JSON description of the pin. A build calls the same functions with the request's cache: `fetch.fetch(pin, cache, names, jobs)`, `fetch.stage` (or, for a build that patches the files, a copy) and `fetch.apply_patch`, which applies a package's own unified diff to a staged copy and refuses one that does not apply. A pin of Python sources states in `packages` the directories of its files that are packages, and `fetch.load(pin, tree)` imports each from a build's staged copy under its directory's name, without adding to the interpreter's search path, so sources that name their siblings as top-level modules import from that copy. The megamoe package carries `patches/megamoe.patch`, applies it to a staged copy and loads the kernel's packages from it; the nvfp4_gemm package does the same with `patches/nvfp4_gemm.patch` and the tutorial's directory, a namespace package of the one module the pin holds.

### The trtllm-gen packages

`uniserve_build_server.trtllm` holds what the packages launching trtllm-gen kernels share: FlashInfer's data tree (the launch code, found through the locked distribution's metadata), the architecture families of a target as trtllm-gen spells them, staging a pinned library, and the whole build of the routed NVFP4 products, which `gather_gemm_nvfp4` and `route_gemm_nvfp4` call with their entry symbol. The launch code compiles against the staged artifact with nvcc alone, without a GPU; the cubins are loaded at run time through `flashinfer::trtllm_cubin_loader::getCubin`, which the module implements over the cubins it publishes beside itself.
