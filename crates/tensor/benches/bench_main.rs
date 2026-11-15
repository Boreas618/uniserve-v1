//! The CPU backend's benchmarks: one group per operation, element type and shape.

#![expect(
    clippy::unwrap_used,
    reason = "a tensor operation that fails on the benchmark's own inputs invalidates the benchmark"
)]

mod benchmarks;

use criterion::criterion_main;

criterion_main!(benchmarks::operations::benches);
