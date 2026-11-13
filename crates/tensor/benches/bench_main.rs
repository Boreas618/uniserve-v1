//! The CPU backend's benchmarks, one group per operation and element type.

#![expect(
    clippy::unwrap_used,
    reason = "a tensor operation that fails on the benchmark's own inputs invalidates the benchmark"
)]

mod benchmarks;

use criterion::criterion_main;

criterion_main!(
    benchmarks::vec_dot::benches,
    benchmarks::affine::benches,
    benchmarks::cat::benches,
    benchmarks::contiguous::benches,
    benchmarks::binary::benches,
    benchmarks::broadcast::benches,
    benchmarks::copy::benches,
    benchmarks::conv_transpose2d::benches,
    benchmarks::matmul::benches,
    benchmarks::reduce::benches,
    benchmarks::unary::benches,
    benchmarks::where_cond::benches,
);
