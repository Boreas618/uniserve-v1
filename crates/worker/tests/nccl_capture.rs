//! NCCL with graph capture in one process.
//!
//! The worker gives each device one thread and one stream, and captures every bucket of that
//! stream into a graph. A tensor-parallel or expert-parallel collective inside a captured bucket is
//! then an NCCL call issued from the capturing thread, in a process that drives more than one GPU.
//! This test holds that contract:
//!
//! 1. One process holds both ranks of a two-rank communicator, each on its own device and thread,
//!    and completes a collective on their streams.
//! 2. That collective is captured into a graph on each rank's stream, instantiated and replayed,
//!    and the replay still communicates.
//!
//! A refusal fails the test naming the call that refused and its message.
//!
//! Each collective starts from a fill of the rank's own value, because the two passes read the same
//! buffer: the sum a pass leaves is evidence only if it began at the pass's own input.
//!
//! It needs two CUDA devices on one host and NCCL, and is ignored by default; requested without
//! them it fails. Run it (a few seconds) with:
//!
//! ```text
//! cargo test -p worker --test nccl_capture -- --ignored
//! ```

// Always true in an integration test crate; it makes clippy treat the helpers below as test code,
// where a failed prerequisite is the test's failure report.
#![cfg(test)]

use std::sync::{Arc, Barrier};

use platform::nccl::ReduceOp;
use platform::{CaptureMode, Communicator, DataType, Device, DeviceId, DeviceRange, PoolKind};

/// The test's device storage is 64 bytes and is not booked against a device allowance.
struct Unbounded;

impl platform::Budget for Unbounded {
    fn book(&self, _device: DeviceId, _pool: PoolKind, _bytes: u64) -> platform::Result<()> {
        Ok(())
    }
}

/// What one rank read back.
#[derive(Debug)]
struct RankReport {
    rank: i32,
    device: DeviceId,
    /// The four f32 the buffer held after one uncaptured collective.
    eager: [f32; 4],
    /// The same after a replay of the captured collective.
    replayed: [f32; 4],
}

/// What one rank does on its own thread.
///
/// Both ranks write `16 * (rank + 1)` into each of four f32 and sum in place, so a rank that
/// received both reads 48 in every element; a rank that only saw itself reads 16 or 32, which the
/// assertion rejects.
fn run_rank(
    device: DeviceId,
    rank: i32,
    size: i32,
    id: platform::nccl::UniqueId,
    barrier: &Barrier,
) -> Result<RankReport, String> {
    let handle = Device::new(device).map_err(|error| format!("Device::new: {error}"))?;
    let mut allocator = platform::Allocator::new(&handle, Arc::new(Unbounded))
        .map_err(|error| format!("Allocator::new: {error}"))?;
    allocator
        .reserve(PoolKind::Scratch, 1 << 16)
        .map_err(|error| format!("Allocator::reserve: {error}"))?;
    let buffer = allocator
        .allocate(PoolKind::Scratch, 16)
        .map_err(|error| format!("Allocator::allocate: {error}"))?;
    let stream = platform::Stream::new(device).map_err(|error| format!("Stream::new: {error}"))?;
    let mut host = platform::PinnedBuffer::new(device, 16)
        .map_err(|error| format!("PinnedBuffer::new: {error}"))?;

    let value = 16.0f32 * (rank as f32 + 1.0);
    fill(value, &mut host, &buffer, &stream)?;

    let communicator = Communicator::init(device, size, rank, &id)
        .map_err(|error| format!("Communicator::init: {error}"))?;

    // An uncaptured collective first. A refusal here is about one process holding both
    // ranks, not about capture.
    let range = DeviceRange::from(&buffer);
    communicator
        .all_reduce(&stream, range, range, 4, DataType::F32, ReduceOp::Sum)
        .map_err(|error| format!("all_reduce before capture: {error}"))?;
    stream
        .synchronize()
        .map_err(|error| format!("synchronize after the collective: {error}"))?;
    let eager = read_back(&host, &buffer, &stream)?;

    // Both ranks reach capture with a completed collective behind them.
    barrier.wait();

    // Then the same collective inside a capture, instantiated and replayed.
    let capture = platform::Capture::begin(&stream, CaptureMode::Relaxed)
        .map_err(|error| format!("Capture::begin: {error}"))?;
    let captured = communicator.all_reduce(
        &stream,
        DeviceRange::from(&buffer),
        DeviceRange::from(&buffer),
        4,
        DataType::F32,
        ReduceOp::Sum,
    );
    let graph = match captured {
        Ok(()) => capture
            .end()
            .map_err(|error| format!("Capture::end: {error}"))?,
        Err(error) => {
            // A refused collective must not leave the stream capturing, so end the capture even
            // on failure and report which call said no.
            let ended = capture.end();
            return Err(format!(
                "capture is refused: ncclAllReduce {error}; Capture::end then {ended:?}"
            ));
        }
    };
    let exec = graph
        .instantiate()
        .map_err(|error| format!("Graph::instantiate: {error}"))?;
    // The buffer is returned to this rank's own value before the replay, so each reading is that
    // pass's own result: the eager collective left both ranks at the sum, and a replay that ran on
    // top of it would read twice the sum whether or not it communicated. From this rank's value,
    // the sum proves the replay communicated and this rank's value alone proves it did not.
    fill(value, &mut host, &buffer, &stream)?;
    exec.launch(&stream)
        .map_err(|error| format!("GraphExec::launch: {error}"))?;
    stream
        .synchronize()
        .map_err(|error| format!("synchronize after the replay: {error}"))?;
    let replayed = read_back(&host, &buffer, &stream)?;

    // The graphs must be destroyed before the communicator they hold: NCCL's own teardown of a
    // communicator whose captured graphs still exist hangs. The worker's pools and generations follow the same rule: a generation's graph execs are
    // destroyed before the communicator their collective node names.
    drop(exec);
    drop(graph);
    communicator
        .destroy()
        .map_err(|error| format!("Communicator::destroy: {error}"))?;
    Ok(RankReport {
        rank,
        device,
        eager,
        replayed,
    })
}

/// Puts this rank's own value in all four f32 of `buffer`, through the pinned staging buffer.
///
/// Each collective in the test starts from this fill: what it reads is a sum over both
/// ranks only if this rank's contribution is its own value and not a previous pass's result.
fn fill(
    value: f32,
    host: &mut platform::PinnedBuffer,
    buffer: &platform::Buffer,
    stream: &platform::Stream,
) -> Result<(), String> {
    // SAFETY: `host` is exclusive to this thread and no copy that writes it is pending.
    let target = unsafe { host.as_mut_slice() };
    for slot in target.as_chunks_mut::<4>().0 {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    // SAFETY: the copy is 16 bytes from a 16-byte pinned buffer into a 16-byte allocation, and the
    // buffer outlives the wait below.
    unsafe { platform::memcpy_async(buffer.address(), host.address(), 16, stream) }
        .map_err(|error| format!("memcpy_async to the device: {error}"))?;
    stream
        .synchronize()
        .map_err(|error| format!("synchronize after the fill: {error}"))
}

/// Copies the buffer's four f32 back to the host and reads them.
fn read_back(
    host: &platform::PinnedBuffer,
    buffer: &platform::Buffer,
    stream: &platform::Stream,
) -> Result<[f32; 4], String> {
    // SAFETY: the copy writes 16 bytes into a 16-byte pinned buffer, on a stream synchronized
    // before the host side is read.
    unsafe { platform::memcpy_async(host.address(), buffer.address(), 16, stream) }
        .map_err(|error| format!("memcpy_async to the host: {error}"))?;
    stream
        .synchronize()
        .map_err(|error| format!("synchronize after the read-back: {error}"))?;
    let mut values = [0.0f32; 4];
    // SAFETY: the copy above wrote the buffer's 16 bytes and nothing else is pending.
    unsafe {
        let bytes = host.as_slice();
        for (slot, chunk) in values.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *slot = f32::from_le_bytes(*chunk);
        }
    }
    Ok(values)
}

#[test]
#[ignore = "requires two CUDA devices on one host and NCCL"]
fn one_process_captures_a_collective_across_two_devices() {
    let devices = DeviceId::count().expect("the CUDA driver reports its devices");
    assert!(
        devices >= 2,
        "the collective needs two CUDA devices on one host, and {devices} are visible"
    );
    let ids = [DeviceId(0), DeviceId(1)];
    let size = 2i32;
    let unique = platform::nccl::UniqueId::new().expect("NCCL creates a unique id");
    let barrier = Barrier::new(size as usize);

    let mut reports: Vec<RankReport> = Vec::new();
    let mut failure: Option<String> = None;
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (rank, device) in ids.iter().copied().enumerate() {
            let id = unique;
            let barrier = &barrier;
            handles.push(scope.spawn(move || run_rank(device, rank as i32, size, id, barrier)));
        }
        for handle in handles {
            match handle.join() {
                Ok(Ok(report)) => reports.push(report),
                Ok(Err(error)) => failure = Some(error),
                Err(_) => failure = Some("a rank thread panicked".to_string()),
            }
        }
    });

    if let Some(error) = failure {
        panic!("NCCL with graph capture in one process: {error}");
    }
    assert_eq!(reports.len(), 2, "both ranks reported: {reports:?}");

    // Each rank's four elements hold its own value (16 for rank 0, 32 for rank 1), so a sum over
    // both ranks leaves every element at 48. A rank that saw only itself reads 16 or 32.
    let expected = 16.0 + 32.0;
    for report in &reports {
        for (what, values) in [("eager", report.eager), ("replayed", report.replayed)] {
            assert!(
                values.iter().all(|value| (*value - expected).abs() < 1e-4),
                "the {what} collective leaves rank {} on {:?} with {values:?}; a rank that saw \
                 both ranks reads {expected}",
                report.rank,
                report.device
            );
        }
    }
}
