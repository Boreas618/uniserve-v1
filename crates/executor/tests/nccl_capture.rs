//! Probe for design 12's open question: **NCCL with graph capture in one process**.
//!
//! The design (specs/worker.md 2.4) gives each device one executor thread and one stream, and 4.5
//! captures every bucket of that stream into a graph. A tensor-parallel or expert-parallel
//! collective inside a captured bucket is then an NCCL call issued from the capturing thread, in a
//! process that drives more than one GPU - which is exactly the case the design leaves unverified.
//!
//! The probe answers:
//!
//! 1. Can one process hold both ranks of a two-rank communicator, each on its own device and
//!    thread, and complete a collective on their streams?
//! 2. Can that collective be captured into a graph on each rank's stream, instantiated and
//!    replayed - and does the replay still communicate?
//! 3. If either is refused, at which call and with what message?
//!
//! It prints one answer line, `probe(nccl-capture): ANSWER 12 ...`, followed by both ranks'
//! readings.
//!
//! Run (caller, one host with two free GPUs; a few seconds):
//!
//! ```text
//! cargo test -p executor --test nccl_capture -- --nocapture
//! ```
//!
//! With fewer than two devices it prints why and returns, so it is safe on a CPU-only host.

use std::sync::{Arc, Barrier};

use platform::nccl::ReduceOp;
use platform::{CaptureMode, Communicator, DataType, Device, DeviceId, DeviceRange, PoolKind};

/// The probe's device storage is 64 bytes and is not booked against a device allowance.
struct Unbounded;

impl platform::Budget for Unbounded {
    fn book(&self, _device: DeviceId, _pool: PoolKind, _bytes: u64) -> platform::Result<()> {
        Ok(())
    }
}

/// One rank's answer.
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
/// Both ranks write `16 * (rank + 1)` into four f32 and sum in place, so a rank that received both
/// reads `4 * (16 + 32) = 192`; a rank that only saw itself reads \(4 \times 16\) or
/// \(4 \times 32\), which the caller's assertion rejects.
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
    {
        let values = [value; 4];
        // SAFETY: `host` is exclusive to this thread and no copy that writes it is pending.
        let target = unsafe { host.as_mut_slice() };
        for (slot, value) in target.chunks_exact_mut(4).zip(values) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
    }
    // SAFETY: the copy is 16 bytes from a 16-byte pinned buffer into a 16-byte allocation, and the
    // buffer outlives the wait below.
    unsafe { platform::memcpy_async(buffer.address(), host.address(), 16, &stream) }
        .map_err(|error| format!("memcpy_async to the device: {error}"))?;
    stream
        .synchronize()
        .map_err(|error| format!("synchronize after the fill: {error}"))?;

    let mut communicator = Communicator::init(device, size, rank, &id)
        .map_err(|error| format!("Communicator::init: {error}"))?;

    // Question 1: an uncaptured collective. A refusal here is about one process holding both
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

    // Question 2: the same collective inside a capture, instantiated and replayed.
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
    exec.launch(&stream)
        .map_err(|error| format!("GraphExec::launch: {error}"))?;
    stream
        .synchronize()
        .map_err(|error| format!("synchronize after the replay: {error}"))?;
    let replayed = read_back(&host, &buffer, &stream)?;

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
        for (slot, chunk) in values.iter_mut().zip(bytes.chunks_exact(4)) {
            *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
    }
    Ok(values)
}

#[test]
#[ignore = "requires a CUDA device: two GPUs on one host"]
fn one_process_captures_a_collective_across_two_devices() {
    let devices = match DeviceId::count() {
        Ok(count) => count,
        Err(error) => {
            println!("probe(nccl-capture): no driver, the probe needs two GPUs: {error}");
            return;
        }
    };
    if devices < 2 {
        println!("probe(nccl-capture): {devices} device(s); the probe needs two on one host");
        return;
    }
    let ids = [DeviceId(0), DeviceId(1)];
    let size = 2i32;
    let unique = match platform::nccl::UniqueId::new() {
        Ok(id) => id,
        Err(error) => {
            println!("probe(nccl-capture): no UniqueId: {error}");
            return;
        }
    };
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

    let Some(error) = failure else {
        assert_probe_answer(&reports);
        return;
    };
    println!("probe(nccl-capture): ANSWER 12 the collective does not work in one process: {error}");
    panic!("NCCL with graph capture in one process: {error}");
}

/// Asserts what the executor's design needs and prints the answer.
fn assert_probe_answer(reports: &[RankReport]) {
    let expected = 4.0 * (16.0 + 32.0);
    let rank0 = reports
        .iter()
        .find(|report| report.rank == 0)
        .expect("rank 0 reported");
    for (what, values) in [("eager", rank0.eager), ("replayed", rank0.replayed)] {
        assert!(
            values.iter().all(|value| (*value - expected).abs() < 1e-4),
            "the {what} collective leaves rank 0 with {values:?}; a rank that saw both ranks reads {expected}"
        );
    }
    println!(
        "probe(nccl-capture): ANSWER 12 one process captures and replays a collective across two devices"
    );
    println!("probe(nccl-capture): {reports:?}");
}
