pub(crate) mod affine;
pub(crate) mod binary;
pub(crate) mod broadcast;
pub(crate) mod cat;
pub(crate) mod contiguous;
pub(crate) mod conv_transpose2d;
pub(crate) mod copy;
pub(crate) mod matmul;
pub(crate) mod reduce;
pub(crate) mod unary;
pub(crate) mod vec_dot;
pub(crate) mod where_cond;

use uniserve_tensor::{DType, Device, DeviceLocation, Shape, Tensor};

/// The name of a benchmark group: the device kind, then the benchmark.
pub(crate) fn bench_name<S: Into<String>>(device: &Device, name: S) -> String {
    match device.location() {
        DeviceLocation::Cpu => format!("cpu_{}", name.into()),
        DeviceLocation::Backend { kind, .. } => format!("{kind}_{}", name.into()),
    }
}

/// A tensor of `dtype` values spread over `[lo, up)`, the same on every run.
pub(crate) fn uniform<S: Into<Shape>>(
    lo: f64,
    up: f64,
    shape: S,
    dtype: DType,
    device: &Device,
) -> Tensor {
    let shape = shape.into();

    // A 64-bit linear congruential generator; its high bits are the well-distributed ones.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let values = (0..shape.elem_count())
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            lo + (up - lo) * unit
        })
        .collect::<Vec<_>>();

    Tensor::from_vec(values, shape, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}
