//! The measurement every benchmark shares: a [`Case`] names an operation on operands made
//! beforehand, and [`measure`] times its runs and states what each one counts.

pub(crate) mod operations;

use std::hint::black_box;
use std::time::Instant;

use criterion::{Criterion, Throughput};
use tensor::{DType, Device, DeviceLocation, Shape, Tensor};

/// A run of a case, its operands already made.
pub(crate) type Run = Box<dyn FnOnce() -> Tensor>;

/// One benchmark: an operation on fixed operands, its runs timed without making the operands.
pub(crate) struct Case {
    /// The group name, after the device kind.
    pub(crate) name: String,
    /// What one run counts toward the group's rate.
    pub(crate) work: Throughput,
    /// A run ready to execute; making it is not timed.
    pub(crate) prepare: Box<dyn Fn() -> Run>,
}

impl Case {
    /// `run` of tensors built once, counted as the bytes of its operands as it reads them and of
    /// the output it writes.
    pub(crate) fn tensors(
        name: impl Into<String>,
        inputs: Vec<Tensor>,
        run: fn(&[Tensor]) -> Tensor,
    ) -> Self {
        let output = run(&inputs);
        let bytes = inputs.iter().chain([&output]).map(bytes_of).sum();

        Self {
            name: name.into(),
            work: Throughput::Bytes(bytes),
            prepare: Box::new(move || {
                // A clone of a tensor shares its storage.
                let inputs = inputs.clone();
                Box::new(move || run(&inputs))
            }),
        }
    }
}

/// The bytes of the elements `tensor`'s layout addresses, a broadcast element once per position.
pub(crate) fn bytes_of(tensor: &Tensor) -> u64 {
    let size = tensor.dtype().size_in_bytes().unwrap();
    (tensor.elem_count() * size) as u64
}

/// Times `case` on `device`: the runs of an iteration count back to back, then a wait for the
/// work they issued.
pub(crate) fn measure(c: &mut Criterion, device: &Device, case: Case) {
    let mut group = c.benchmark_group(bench_name(device, &case.name));
    group.throughput(case.work);
    group.bench_function("iter", |b| {
        b.iter_custom(|iters| {
            let runs: Vec<Run> = (0..iters).map(|_| (case.prepare)()).collect();

            let start = Instant::now();
            for run in runs {
                black_box(run());
            }
            device.synchronize().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

/// The name of a benchmark group: the device kind, then the benchmark.
pub(crate) fn bench_name(device: &Device, name: &str) -> String {
    match device.location() {
        DeviceLocation::Cpu => format!("cpu_{name}"),
        DeviceLocation::Backend { kind, .. } => format!("{kind}_{name}"),
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
