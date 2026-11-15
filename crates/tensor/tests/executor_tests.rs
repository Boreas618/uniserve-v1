//! The process's CPU executor as its owner configures it.
//!
//! The executor is process-wide, so these tests have a test binary of their own: no other test
//! starts the executor before `an_installed_executor_computes_and_stays` installs it.

use tensor::cpu_backend::{CpuExecutor, ExecutorConfig};
use tensor::{Device, Result, Tensor};

#[test]
fn a_configuration_without_cpus_threads_or_chunk_elements_is_refused() -> Result<()> {
    let config = ExecutorConfig::for_process()?;
    assert!(!config.cpus.is_empty());
    assert_eq!(config.threads, config.cpus.len());

    for refused in [
        ExecutorConfig {
            cpus: Vec::new(),
            ..config.clone()
        },
        ExecutorConfig {
            threads: 0,
            ..config.clone()
        },
        ExecutorConfig {
            chunk_elements: 0,
            ..config.clone()
        },
    ] {
        let message = CpuExecutor::install(refused).unwrap_err().to_string();
        assert!(message.contains("the configuration"), "{message}");
    }
    Ok(())
}

#[test]
fn an_installed_executor_computes_and_stays() -> Result<()> {
    // Every map of more than one element splits into chunks of three, on two threads.
    let process = ExecutorConfig::for_process()?;
    let config = ExecutorConfig {
        cpus: process.cpus.iter().rev().copied().collect(),
        threads: 2,
        parallel_elements: 2,
        chunk_elements: 3,
        ..process.clone()
    };
    CpuExecutor::install(config.clone())?;

    // The executor runs the configuration, its CPUs in ascending order.
    let running = CpuExecutor::current()?.config();
    assert_eq!(running.cpus, process.cpus);
    assert_eq!(
        (
            running.threads,
            running.parallel_elements,
            running.chunk_elements
        ),
        (2, 2, 3)
    );

    // An element-wise map, a broadcast and a matrix product on it.
    let values: Vec<f32> = (0..10).map(|v| v as f32).collect();
    let x = Tensor::new(values.as_slice(), &Device::Cpu)?;
    assert_eq!(
        x.sqr()?.to_vec1::<f32>()?,
        values.iter().map(|v| v * v).collect::<Vec<_>>()
    );

    let rows = x.reshape((2, 5))?;
    let shifted = rows.broadcast_add(&Tensor::new(&[[100f32], [200.]], &Device::Cpu)?)?;
    assert_eq!(
        shifted.to_vec2::<f32>()?,
        [
            [100., 101., 102., 103., 104.],
            [205., 206., 207., 208., 209.]
        ]
    );
    assert_eq!(
        rows.matmul(&rows.t()?)?.to_vec2::<f32>()?,
        [[30., 80.], [80., 255.]]
    );

    // The process has its executor: a second one is refused.
    let message = CpuExecutor::install(config).unwrap_err().to_string();
    assert!(message.contains("already runs"), "{message}");
    Ok(())
}
