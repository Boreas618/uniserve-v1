//! Prints what cuBLASLt's heuristic answers for a small product, layout by
//! layout and workspace size by workspace size, which is the evidence for a
//! refused `Matmul::candidates`.
//!
//! Run it by hand on the commit under test:
//!
//! ```text
//! cargo run -p platform --example matmul_probe
//! ```
//!
//! The GPU test `a_matrix_product_runs_on_an_algorithm_the_heuristic_offers`
//! asks the same question of the row-major 4x6x3 descriptor with no
//! workspace; this program asks it of both layouts and two workspace sizes so
//! that a refusal names its input rather than the call.

use platform::{
    ComputeType, DataType, Device, DeviceId, Epilogue, Matmul, MatmulDescriptor, MatrixLayout,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("the probe could not reach cuBLASLt: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), platform::Error> {
    let device = Device::new(DeviceId(0))?;
    let matmul = Matmul::new(&device)?;
    println!("cuBLASLt version {}", matmul.version());

    let (m, n, k) = (4u64, 6, 3);
    let column = MatmulDescriptor {
        compute: ComputeType::F32,
        order: platform::Order::ColumnMajor,
        a: MatrixLayout::column_major(m, k, DataType::F32),
        b: MatrixLayout::column_major(k, n, DataType::F32),
        c: MatrixLayout::column_major(m, n, DataType::F32),
        d: MatrixLayout::column_major(m, n, DataType::F32),
        transpose_a: false,
        transpose_b: false,
        epilogue: Epilogue::Default,
    };
    let row = MatmulDescriptor::row_major(m, n, k, DataType::F32, ComputeType::F32);
    for (name, descriptor) in [("column-major", &column), ("row-major", &row)] {
        for workspace in [0usize, 1 << 20] {
            match matmul.candidates(descriptor, workspace, 4) {
                Ok(algorithms) => println!(
                    "{name}, {workspace} bytes of workspace: {} algorithm(s), first needs {}",
                    algorithms.len(),
                    algorithms
                        .first()
                        .map_or(0, |algorithm| algorithm.workspace_bytes())
                ),
                Err(error) => println!("{name}, {workspace} bytes of workspace: {error}"),
            }
        }
    }
    Ok(())
}
