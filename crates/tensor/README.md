# tensor

The tensor library of [UniServe](../../README.md): n-dimensional arrays of one element type on a device, their views, the operations on them, a CPU backend, and an open interface through which every other device is implemented outside this repository.

The crate serves inference only. A tensor is a value: it records no computation graph, and there are no gradients, variables or random initializers.

```rust
use tensor::{Device, Result, Tensor};

fn main() -> Result<()> {
    let a = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    let b = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((3, 4))?;
    let c = a.matmul(&b)?;
    assert_eq!(c.dims(), [2, 4]);
    Ok(())
}
```

## What the crate provides

- `Tensor`: a reference-counted view (`Layout`: shape, strides, start offset) over a reference-counted `Storage`. Views (`narrow`, `reshape`, `transpose`, `permute`, `squeeze`, `unsqueeze`, `broadcast_as`, `chunk`, `t`, indexing with `i(...)`) share the storage of the tensor they come from.
- Tensor operations built from backend primitives: element-wise arithmetic and comparisons with broadcasting, the integer operations `remainder`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_left_shift` and `bitwise_right_shift`, `where_cond`, reductions, `to_dtype`, `view` (an element type of the same size read over the same elements), `contiguous`, `cat`, `stack`, `pad_with_zeros`, `gather`, `scatter`, `scatter_add`, `index_select`, `index_add`, `slice_set`, `matmul`, `conv1d`, `conv2d`, their transposed forms, pooling and upsampling.
- `DType`: `U8`, `U32`, `I16`, `I32`, `I64`, `BF16`, `F16`, `F32`, `F64`, `F8E4M3`, and `F4`, the FP4 (E2M1) element packed two to a byte. `DType::size_in_bytes` is the size of one element, `None` for `F4`, and `DType::storage_bytes` the bytes of a count of contiguous elements for every element type. An `F4` tensor keeps the packed bytes it was built from with `Tensor::from_raw_buffer`, moves them between devices whole and writes them out with `Tensor::write_bytes`; no host element type holds one of its elements, so the CPU backend computes nothing on it, copies no single element and refuses each such operation with an error.
- `Shape`, `Dim`, `D`, `Layout`, `NdArray` and `WithDType` for building tensors from host values and reading them back (`to_vec0` to `to_vec3`, `to_scalar`).
- `Module`: a `forward` over one tensor.
- `Tensor::from_raw_buffer` and `Tensor::write_bytes`: a tensor built from the little-endian bytes of its elements, and those bytes written out.
- `Display` for tensors and the crate's `Error`.

Index tensors of `index_select`, `gather`, `scatter`, `scatter_add` and `index_add` may be `U8`, `U32` or `I64`.

Arithmetic follows PyTorch where the two have the same expression:

- An integer tensor computes in its element type: `+`, `-` and `*` wrap, `/` truncates toward zero (`torch.div(.., rounding_mode="trunc")`), and `remainder` is `torch.remainder`, whose result is zero or has the sign of the divisor. A zero integer divisor is an error on the CPU backend.
- A unary operation on an integer tensor is computed when its result is an integer of the element type (`abs`, which wraps for the smallest signed value, `relu`, `sign`, `floor`, `ceil`, `round`) and is an error otherwise (`exp`, `neg`, `sqr`, `sqrt`, `gelu`, ...).
- A scalar operand of an integer tensor (`&x + 1.0`, `7.0 / &x`) is an integer of the element type and the operation is the one of two tensors, so it is exact for every value. No operation changes the element type: a scalar with a fractional part is an error, where PyTorch would return a float tensor, and so is one outside the element type's range, which PyTorch refuses as well; a caller that wants the float result converts the tensor with `to_dtype` first.
- Sums and matrix products of `BF16` and `F16` tensors accumulate in `F32` on the CPU backend and round once into the element type. Other element-wise operations on those types round each step in the element type.
- `Tensor::view(dtype)` is PyTorch's `Tensor.view(dtype)` between element types of one size. On a backend device the result shares the tensor's memory; the CPU backend's storage is a vector of one element type, so there the result holds a copy of the bytes.

## Devices and the backend interface

A `Device` is `Cpu` or `Backend(Arc<dyn BackendDevice>)`, and a `Storage` is `Cpu(CpuStorage)` or `Backend(Box<dyn BackendStorage>)`. The CPU backend is built in. Any other device is a pair of types implementing the two object-safe traits of `tensor::backend` in another crate:

- `BackendDevice` states the device's `DeviceLocation` (a kind such as `"cuda"` and an ordinal), decides with `same_device` which devices may meet in one operation, allocates storages (`zeros`, `alloc_uninit`), copies host values to the device (`storage_from_cpu_storage`) and waits for issued work (`synchronize`).
- `BackendStorage` holds the elements of a tensor and implements the primitives tensor operations are built from: `affine`, `powf`, `elu`, `unary`, `binary`, `cmp`, `where_cond`, `reduce`, `to_dtype`, `view`, `matmul`, `index_select`, `gather`, `index_add`, `scatter_set`, `scatter_add_set`, `copy_strided_src`, `copy2d`, `const_set`, the convolutions, pooling and upsampling, together with `try_clone` and the device-to-host copy `to_cpu_storage`. Operations are named by the enums `UnaryOp`, `BinaryOp`, `CmpOp` and `ReduceOp`.

Tensor operations are the same on every device. Each one validates shapes and element types, checks that its operands are on the same device, and calls one primitive of the storage's backend with the layouts of its operands. Another operand reaches the primitive as `&dyn BackendStorage`, which the backend downcasts through `as_any` to its own storage type.

The rules a backend can rely on:

- The CPU and a backend device are never the same device, so the operands of an operation are all CPU storages or all backend storages; mixing them is a `DeviceMismatchBinaryOp` error. For two backend devices the answer is that of `BackendDevice::same_device`, asked of the first operand's device.
- `Tensor::to_device` returns the tensor itself when the target is the same device, copies through `storage_from_cpu_storage` from the CPU to a backend device, copies through `to_cpu_storage` from a backend device to the CPU, and is an error between two backend devices that are not the same device. Both copies keep every element at its position, so a view keeps its layout.
- Tensors built from host values on a backend device (`Tensor::new`, `from_vec`, `from_slice`, `from_raw_buffer`) are built on the host and copied with `storage_from_cpu_storage`.
- Reading values (`to_vec*`, `to_scalar`, `Display`) calls `to_cpu_storage`.
- Sorting (`arg_sort_last_dim`, `sort_last_dim`) is computed by the CPU backend only and is an error on a backend tensor.

What a backend uses to reach storage:

- `Tensor::from_storage(storage, shape)` builds a tensor over a storage the caller made.
- `Tensor::storage_and_layout()` and `Tensor::storage_mut_and_layout()` lock the storage under a tensor for reading or writing and return it with the tensor's layout.

`tests/common/mod.rs` holds a complete backend written against the public interface only, which keeps its elements in host memory and computes with `CpuStorage`. The integration tests run on the CPU device and on that backend device.

## Building

The crate builds for Linux on x86_64 and aarch64 with the default features of its target; it needs no `rustflags` and has no Cargo features, so a workspace that depends on it sets nothing. Its vector kernels are written for AVX2 on x86_64, selected at run time, and for Advanced SIMD (NEON) on aarch64, part of every aarch64 CPU; any other architecture runs their scalar loops.

The CPU backend multiplies `F32` and `F64` matrices with [`gemm`](https://crates.io/crates/gemm). `F16` and `BF16` products are computed in `F32` on every target: the operands are widened, which is exact, the product accumulates in `F32` and is rounded to the 16-bit type once. `gemm`'s `f16` feature is not enabled because its half-precision kernels do not keep that contract: they round the running sums into `F16` after each block of the inner dimension, and on aarch64 CPUs with FP16 arithmetic they accumulate in `F16` (they also do not compile in unoptimized aarch64 builds).

The crate is the workspace member `crates/tensor` of UniServe and takes the workspace's edition, lints and the versions of the element-type crates `half` and `float8`; it depends on no other member.

```sh
cargo build -p tensor
cargo clippy -p tensor --all-targets -- -D warnings
cargo fmt -p tensor -- --check
cargo test -p tensor
cargo bench -p tensor
```

`cargo bench -p tensor` measures one table of cases (`benches/benchmarks/operations.rs`): each operation for each element type and each shape or layout it is measured at, together with the vector dot-product kernels. A case's rate counts the bytes of its operands as it reads them and of its output, or the floating-point operations of a matrix product.

## CPU executor

The CPU backend computes its parallel operations on one executor per process (`cpu_backend::CpuExecutor`). It has a rayon pool, on which matrix products, convolutions and sorting split their work, and a pool of element-wise workers, on which element-wise maps above a size split their elements into chunks that the calling thread and the workers take in turn. Both pools run on one set of CPUs, to which every thread is pinned.

The process's owner configures the executor with an `ExecutorConfig` and installs it before the first parallel CPU operation; a process that installs none gets the default configuration on that first operation. `ExecutorConfig::for_process` states the defaults, and a second install is an error:

```rust
use tensor::cpu_backend::{CpuExecutor, ExecutorConfig};

fn configure() -> tensor::Result<()> {
    let config = ExecutorConfig {
        threads: 8,
        ..ExecutorConfig::for_process()?
    };
    CpuExecutor::install(config)
}
```

| Field | Meaning | Default |
| --- | --- | --- |
| `cpus` | The CPU numbers every thread of the executor is pinned to. | The CPUs the starting thread may run on (`sched_getaffinity`). |
| `threads` | The threads computing one operation, the calling thread included for an element-wise map. | One per CPU of `cpus`. |
| `idle_spins` | The busy-wait iterations an idle element-wise worker watches for its next map before it parks. | 25,000 |
| `parallel_elements` | The fewest elements an element-wise map splits across threads. | 65,536 |
| `chunk_elements` | The elements of one chunk a thread takes of a split map. | 16,384 |

## Origin and license

This crate is derived from `candle-core` of [huggingface/candle](https://github.com/huggingface/candle), imported at upstream commit `cfcbd7186d13271843ce7eb41d25659bd0835850`. Candle is licensed under either of the MIT license or the Apache License, Version 2.0, at your option; this crate is distributed under the same terms, unlike the rest of the workspace, and both license texts, with the upstream copyright notices, are kept beside this README (`LICENSE-MIT`, `LICENSE-APACHE`).

The first commit of the crate's history is that import with only the crate renamed and its manifest made standalone. Every later change is UniServe's own and is not proposed upstream. Relative to the import, the crate keeps the tensor, its views and operations and the CPU backend; the device backends other than the CPU are replaced by the open backend interface; and training support (backpropagation, variables, random initialization), custom operators, `safetensors` support, the quantized formats, the `pickle` and `npy` readers, the streaming module and the vendor BLAS features are not part of it.

"Candle" names the upstream project only; this crate is not candle and is not endorsed by its authors.
