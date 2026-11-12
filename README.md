# uniserve-tensor

The tensor library of [UniServe](https://github.com/Boreas618): n-dimensional arrays of one element type on a device, their views, the operations on them, a CPU backend, and an open interface through which every other device is implemented outside this repository.

The crate serves inference only. A tensor is a value: it records no computation graph, and there are no gradients, variables or random initializers.

```rust
use uniserve_tensor::{Device, Result, Tensor};

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
- Tensor operations built from backend primitives: element-wise arithmetic and comparisons with broadcasting, the integer operations `remainder`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_left_shift` and `bitwise_right_shift`, `where_cond`, reductions, `to_dtype`, `contiguous`, `cat`, `stack`, `pad_with_zeros`, `gather`, `scatter`, `scatter_add`, `index_select`, `index_add`, `slice_set`, `matmul`, `conv1d`, `conv2d`, their transposed forms, pooling and upsampling.
- `DType`: `U8`, `U32`, `I16`, `I32`, `I64`, `BF16`, `F16`, `F32`, `F64`, `F8E4M3`, and the packed types `F6E2M3`, `F6E3M2`, `F4`, `F8E8M0`, whose tensors keep their bytes as loaded and support no arithmetic.
- `Shape`, `Dim`, `D`, `Layout`, `NdArray` and `WithDType` for building tensors from host values and reading them back (`to_vec0` to `to_vec3`, `to_scalar`).
- `Module`: a `forward` over one tensor.
- `safetensors`: reading headers and tensors from files, memory maps and buffers, and saving tensors.
- `CustomOp1/2/3` and `InplaceOp1/2/3`: operators defined outside the crate.
- `Display` for tensors and the crate's `Error`.

Index tensors of `index_select`, `gather`, `scatter`, `scatter_add` and `index_add` may be `U8`, `U32` or `I64`.

Arithmetic follows PyTorch where the two have the same expression:

- An integer tensor computes in its element type: `+`, `-` and `*` wrap, `/` truncates toward zero (`torch.div(.., rounding_mode="trunc")`), and `remainder` is `torch.remainder`, whose result is zero or has the sign of the divisor. A zero integer divisor is an error on the CPU backend.
- A scalar operand of an integer tensor (`&x + 1.0`, `7.0 / &x`) is an integer of the element type and the operation is the one of two tensors, so it is exact for every value. No operation changes the element type: a scalar with a fractional part is an error, where PyTorch would return a float tensor, and so is one outside the element type's range, which PyTorch refuses as well; a caller that wants the float result converts the tensor with `to_dtype` first.

## Devices and the backend interface

A `Device` is `Cpu` or `Backend(Arc<dyn BackendDevice>)`, and a `Storage` is `Cpu(CpuStorage)` or `Backend(Box<dyn BackendStorage>)`. The CPU backend is built in. Any other device is a pair of types implementing the two object-safe traits of `uniserve_tensor::backend` in another crate:

- `BackendDevice` states the device's `DeviceLocation` (a kind such as `"cuda"` and an ordinal), decides with `same_device` which devices may meet in one operation, allocates storages (`zeros`, `alloc_uninit`), copies host values to the device (`storage_from_cpu_storage`) and waits for issued work (`synchronize`).
- `BackendStorage` holds the elements of a tensor and implements the primitives tensor operations are built from: `affine`, `powf`, `elu`, `unary`, `binary`, `cmp`, `where_cond`, `reduce`, `to_dtype`, `matmul`, `index_select`, `gather`, `index_add`, `scatter_set`, `scatter_add_set`, `copy_strided_src`, `copy2d`, `const_set`, the convolutions, pooling and upsampling, together with `try_clone` and the device-to-host copy `to_cpu_storage`. Operations are named by the enums `UnaryOp`, `BinaryOp`, `CmpOp` and `ReduceOp`.

Tensor operations are the same on every device. Each one validates shapes and element types, checks that its operands are on the same device, and calls one primitive of the storage's backend with the layouts of its operands. Another operand reaches the primitive as `&dyn BackendStorage`, which the backend downcasts through `as_any` to its own storage type.

The rules a backend can rely on:

- The CPU and a backend device are never the same device, so the operands of an operation are all CPU storages or all backend storages; mixing them is a `DeviceMismatchBinaryOp` error. For two backend devices the answer is that of `BackendDevice::same_device`, asked of the first operand's device.
- `Tensor::to_device` returns the tensor itself when the target is the same device, copies through `storage_from_cpu_storage` from the CPU to a backend device, copies through `to_cpu_storage` from a backend device to the CPU, and is an error between two backend devices that are not the same device. Both copies keep every element at its position, so a view keeps its layout.
- Tensors built from host values on a backend device (`Tensor::new`, `from_vec`, `from_slice`, the `safetensors` loaders) are built on the host and copied with `storage_from_cpu_storage`.
- Reading values (`to_vec*`, `to_scalar`, `Display`) calls `to_cpu_storage`.

What a backend and operators built on it use to reach storage:

- `Tensor::from_storage(storage, shape)` builds a tensor over a storage the caller made.
- `Tensor::storage_and_layout()` and `Tensor::storage_mut_and_layout()` lock the storage under a tensor for reading or writing and return it with the tensor's layout.
- `CustomOp1/2/3` compute a new tensor from one, two or three tensors with `cpu_fwd` on CPU storages and `backend_fwd` on backend storages, and are applied with `Tensor::apply_op1/2/3`. `InplaceOp1/2/3` write into their first operand and are applied with `Tensor::inplace_op1/2/3`; `InplaceOpN` also admits sources that alias the destination, according to its `AccessPattern`. An operator without a `backend_fwd` is an error on a backend tensor, as sorting (`arg_sort_last_dim`, `sort_last_dim`) is.

`tests/common/mod.rs` holds a complete backend written against the public interface only, which keeps its elements in host memory and computes with `CpuStorage`. The integration tests run on the CPU device and on that backend device.

## Building

The crate builds with the default features of its target; it needs no `rustflags` and has no Cargo features, so a workspace that depends on it sets nothing.

The CPU backend multiplies `F32` and `F64` matrices with [`gemm`](https://crates.io/crates/gemm). `F16` and `BF16` products are computed in `F32`: the operands are widened, which is exact, and the result is rounded to the 16-bit type once. `gemm`'s `f16` feature is disabled: its half-precision kernels do not compile in unoptimized builds for `aarch64` targets without FP16 in their baseline, and they accumulate in half precision.

```sh
cargo build
cargo clippy --all-targets -- -D warnings
cargo fmt --check
cargo test
```

## CPU threads

The CPU backend runs large operations on its own thread pools. These environment variables are read once, when a pool is first used:

| Variable | Effect |
| --- | --- |
| `RAYON_NUM_THREADS` | Threads of the rayon pool and of `gemm`; defaults to the physical cores, or to the preferred cores on Linux. |
| `UNISERVE_TENSOR_NUM_THREADS` | Threads of the barrier pool used by element-wise operations; defaults to the rayon value. |
| `UNISERVE_TENSOR_BARRIER_POOL_SPIN_LIMIT` | Spins of an idle barrier-pool worker before it parks. |
| `UNISERVE_TENSOR_CPU_MASK` | Linux: the CPU list the pools count and pin to, such as `0-7,16`. |
| `UNISERVE_TENSOR_CPU_AFFINITY` | Linux: any value other than `0` or `false` pins pool threads to the preferred CPUs. |

## Origin and license

This crate is derived from `candle-core` of [huggingface/candle](https://github.com/huggingface/candle), imported at upstream commit `cfcbd7186d13271843ce7eb41d25659bd0835850`. Candle is licensed under either of the MIT license or the Apache License, Version 2.0, at your option; this crate is distributed under the same terms, and both license texts, with the upstream copyright notices, are kept in this repository (`LICENSE-MIT`, `LICENSE-APACHE`).

The first commit of this repository is that import with only the crate renamed and its manifest made standalone. Every later change is UniServe's own and is not proposed upstream. Relative to the import, the crate keeps the tensor, its views and operations, the CPU backend, custom operators and `safetensors` support; the device backends other than the CPU are replaced by the open backend interface; and training support (backpropagation, variables, random initialization), the quantized formats, the `pickle` and `npy` readers, the streaming module and the vendor BLAS features are not part of it.

"Candle" names the upstream project only; this crate is not candle and is not endorsed by its authors.
