use crate::benchmarks::{bench_name, uniform};
use criterion::{criterion_group, Criterion, Throughput};
use half::{bf16, f16};
use std::hint::black_box;
use std::time::Instant;
use uniserve_tensor::{DType, Device, Tensor};

fn run_sum(a: &Tensor) {
    a.sum_keepdim(2).unwrap();
}
fn run_arg_min(a: &Tensor) {
    a.argmin_keepdim(2).unwrap();
}

fn criterion_benchmark(c: &mut Criterion) {
    let (lo, up) = (-1000.0f32, 1000.0f32);
    let device = Device::Cpu;
    run_reduce(c, &device, (lo, up), false);
    run_reduce(c, &device, (f16::from_f32(lo), f16::from_f32(up)), false);
    run_reduce(c, &device, (bf16::from_f32(lo), bf16::from_f32(up)), false);

    run_arg_reduce(c, &device, (lo, up), false);
    run_arg_reduce(c, &device, (f16::from_f32(lo), f16::from_f32(up)), false);
    run_arg_reduce(c, &device, (bf16::from_f32(lo), bf16::from_f32(up)), false);

    run_reduce(c, &device, (lo, up), true);
    run_reduce(c, &device, (f16::from_f32(lo), f16::from_f32(up)), true);
    run_reduce(c, &device, (bf16::from_f32(lo), bf16::from_f32(up)), true);

    run_arg_reduce(c, &device, (lo, up), true);
    run_arg_reduce(c, &device, (f16::from_f32(lo), f16::from_f32(up)), true);
    run_arg_reduce(c, &device, (bf16::from_f32(lo), bf16::from_f32(up)), true);
}

fn run_reduce<T: uniserve_tensor::FloatDType>(
    c: &mut Criterion,
    device: &Device,
    (lo, up): (T, T),
    strided: bool,
) {
    let b = 1;
    let m = 1024;
    let k = 1024;

    let a = uniform(lo.to_f64(), up.to_f64(), (b, m, k), T::DTYPE, device);
    let a = if strided {
        a.transpose(0, 2).unwrap()
    } else {
        a
    };

    let flops = b * m * k * T::DTYPE.size_in_bytes();

    let name = match T::DTYPE {
        DType::F32 => {
            if strided {
                "reduce_f32_strided"
            } else {
                "reduce_f32"
            }
        }
        DType::F16 => {
            if strided {
                "reduce_f16_strided"
            } else {
                "reduce_f16"
            }
        }
        DType::BF16 => {
            if strided {
                "reduce_bf16_strided"
            } else {
                "reduce_bf16"
            }
        }
        _ => "unknown",
    };

    let mut group = c.benchmark_group(bench_name(device, name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_sum(black_box(&a));
            }
            device.synchronize().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn run_arg_reduce<T: uniserve_tensor::FloatDType>(
    c: &mut Criterion,
    device: &Device,
    (lo, up): (T, T),
    strided: bool,
) {
    let b = 1;
    let m = 1024;
    let k = 1024;

    let a = uniform(lo.to_f64(), up.to_f64(), (b, m, k), T::DTYPE, device);
    let a = if strided {
        a.transpose(0, 2).unwrap()
    } else {
        a
    };

    let flops = b * m * k * T::DTYPE.size_in_bytes();

    let name = match T::DTYPE {
        DType::F32 => {
            if strided {
                "arg_reduce_f32_strided"
            } else {
                "arg_reduce_f32"
            }
        }
        DType::F16 => {
            if strided {
                "arg_reduce_f16_strided"
            } else {
                "arg_reduce_f16"
            }
        }
        DType::BF16 => {
            if strided {
                "arg_reduce_bf16_strided"
            } else {
                "arg_reduce_bf16"
            }
        }
        _ => "unknown",
    };

    let mut group = c.benchmark_group(bench_name(device, name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_arg_min(black_box(&a));
            }
            device.synchronize().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
