//! Design section 12's open question: can a new and an old generation of
//! programs share one graph pool?
//!
//! The design leaves open whether the graphs of two generations - the serving
//! one and the one a precopy filled - can live in one device's graph pool at
//! the same time, or whether each generation needs a pool of its own. This
//! probe answers it by measurement on one GPU: every activation the graphs
//! address is mapped before the first capture and never reallocated, two
//! generations' graphs are captured over those same buffers, a third is
//! captured while both are alive, and the three are replayed alternately with
//! each generation's own arithmetic visible in the result.
//!
//! What it measures, and why each part is the question:
//!
//! - **capture allocates nothing**: `Allocator::mapped_bytes` is unchanged
//!   across all three captures, which is design 4.5's "no allocation during a
//!   step" applied to capture;
//! - **both generations coexist**: a second capture succeeds while the first
//!   generation's executable is instantiated and its buffers are mapped;
//! - **a replay is the generation it was captured from**: generation 0 adds
//!   `value` once, generation 1 adds it twice, and replays interleave, so a
//!   re-pointed or shared node would show the wrong arithmetic;
//! - **a third capture while two are live**: the pool holds three executables'
//!   worth of graphs without a second reservation.
//!
//! A second probe answers the question a purge asks of the same structures: which generation's
//! graphs a purge may release, and whether the generation that still serves keeps its own. See
//! `a_purge_keeps_the_graph_a_live_generation_still_serves`.
//!
//! Run them on a GPU host, alone on the device:
//!
//! ```text
//! cargo test -p executor --test graph_pools -- --ignored --test-threads=1
//! ```
//!
//! It needs `nvcc` on `PATH`, and `UNISERVE_PROBE_ARCH` names the architecture
//! it compiles for (default `sm_100a`).

#![allow(
    clippy::expect_used,
    reason = "a probe's prerequisite failure must fail the test, not be handled"
)]
use std::ffi::c_void;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use platform::{
    Allocator, Budget, Buffer, Capture, CaptureMode, Device, DeviceId, Function, Graph, GraphExec,
    LaunchConfig, Library, PinnedBuffer, PoolKind, Result, Stream, memcpy_async, memset_d32,
};

const DEVICE: DeviceId = DeviceId(0);
const WORDS: usize = 32;
const BYTES: usize = WORDS * 4;
const RESERVATION: usize = 1 << 20;

/// A budget that records the pool's cumulative total and never refuses: the
/// probe is about the pool's addresses, not about running out of device memory.
#[derive(Default)]
struct Allowance(Mutex<u64>);

impl Budget for Allowance {
    fn book(&self, _device: DeviceId, _pool: PoolKind, bytes: u64) -> Result<()> {
        *self.0.lock().unwrap_or_else(|poison| poison.into_inner()) = bytes;
        Ok(())
    }
}

/// Compiles the probe's two generations of one kernel into a cubin.
fn cubin() -> PathBuf {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/generations.cu");
    let output = std::env::temp_dir().join(format!("generations-{}.cubin", std::process::id()));
    let arch = std::env::var("UNISERVE_PROBE_ARCH").unwrap_or_else(|_| "sm_100a".to_owned());
    let status = Command::new("nvcc")
        .arg("-cubin")
        .arg(format!("-arch={arch}"))
        .arg("-o")
        .arg(&output)
        .arg(&source)
        .status()
        .expect("this probe compiles its kernel with nvcc, which must be on PATH");
    assert!(status.success(), "nvcc failed on {}", source.display());
    output
}

/// Launches `kernel` inside one capture and instantiates the graph it records.
///
/// `data` and `value` must outlive the capture: the driver records their
/// addresses as the node's arguments, and the graph is replayed from them.
fn capture(stream: &Stream, kernel: &Function, data: &mut u64, value: &mut u32) -> GraphExec {
    let mut count = WORDS as i32;
    let mut parameters: [*mut c_void; 3] = [
        (data as *mut u64).cast(),
        (&raw mut count).cast(),
        (value as *mut u32).cast(),
    ];
    let mut recording = Capture::begin(stream, CaptureMode::Relaxed).expect("begin a capture");
    kernel
        .launch_ex(
            stream,
            &LaunchConfig::new([1, 1, 1], [WORDS as u32, 1, 1]),
            &mut parameters,
        )
        .expect("launch inside the capture");
    recording.record().expect("record the launch's node");
    recording
        .end()
        .expect("end the capture")
        .instantiate()
        .expect("instantiate the graph - failing here means two generations cannot share the pool")
}

/// Replays `graph` and answers the buffer's words.
fn replay(stream: &Stream, graph: &GraphExec, data: u64) -> Vec<u32> {
    graph.launch(stream).expect("replay");
    let host = PinnedBuffer::new(DEVICE, BYTES).expect("pinned staging");
    // SAFETY: the copy reads `BYTES` from the allocation into a pinned buffer
    // of the same size, and the stream that wrote the allocation is
    // synchronized before the buffer is read.
    unsafe { memcpy_async(host.address(), data, BYTES, stream) }.expect("copy back");
    stream.synchronize().expect("synchronize");
    // SAFETY: the copy completed and the buffer holds `BYTES` readable bytes.
    unsafe { host.as_slice() }
        .chunks_exact(4)
        .map(|word| u32::from_ne_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

#[test]
#[ignore = "requires a CUDA device"]
fn two_generations_share_one_graph_pool() {
    let device = Device::new(DEVICE).expect("a CUDA device");
    let allowance = Arc::new(Allowance::default());
    let mut allocator = Allocator::new(&device, allowance).expect("an allocator");
    allocator
        .reserve(PoolKind::Graph, RESERVATION)
        .expect("reserve the graph pool");
    let buffer: Buffer = allocator
        .allocate(PoolKind::Graph, BYTES)
        .expect("one activation buffer");
    let data = buffer.address();
    let stream = Stream::new(DEVICE).expect("a stream");
    let image = std::fs::read(cubin()).expect("the compiled probe kernel");
    let library = Library::load(&device, &image).expect("load the probe kernel");
    let first = library.kernel("generation_0").expect("generation_0");
    let second = library.kernel("generation_1").expect("generation_1");

    // SAFETY: the fill lies inside the allocation.
    unsafe { memset_d32(data, 0, WORDS, &stream) }.expect("zero the buffer");

    let mapped_before = allocator.mapped_bytes(PoolKind::Graph);
    let mut address = data;
    let mut value = 1u32;
    let serving = capture(&stream, &first, &mut address, &mut value);
    let precopied = capture(&stream, &second, &mut address, &mut value);
    let mapped_after_two = allocator.mapped_bytes(PoolKind::Graph);

    assert_eq!(
        mapped_before, mapped_after_two,
        "a capture mapped storage: the pool grew from {mapped_before} to {mapped_after_two} bytes"
    );
    assert_eq!(buffer.address(), address, "the activation buffer moved");

    assert_eq!(
        replay(&stream, &serving, data),
        vec![1; WORDS],
        "the first generation's graph does not add value once"
    );
    assert_eq!(
        replay(&stream, &precopied, data),
        vec![3; WORDS],
        "the second generation's graph does not add value twice"
    );

    let mut value = 10u32;
    let late = capture(&stream, &first, &mut address, &mut value);
    let mapped_after_three = allocator.mapped_bytes(PoolKind::Graph);
    assert_eq!(
        mapped_before, mapped_after_three,
        "the third capture mapped storage: {mapped_before} -> {mapped_after_three} bytes"
    );

    assert_eq!(
        replay(&stream, &serving, data),
        vec![4; WORDS],
        "the first generation's graph no longer replays after another capture"
    );
    assert_eq!(
        replay(&stream, &late, data),
        vec![14; WORDS],
        "the third graph does not add ten"
    );
    assert_eq!(
        replay(&stream, &precopied, data),
        vec![16; WORDS],
        "the second generation's graph no longer replays after two more captures"
    );

    // Dropping every executable hands the block back without unmapping it; the
    // pool's figure is what the budget was charged, so it stays until `free`.
    drop(serving);
    drop(precopied);
    drop(late);
    let mapped_before_free = allocator.mapped_bytes(PoolKind::Graph);
    allocator.free(buffer);
    // A pool keeps the storage it mapped for the pool's life and hands a freed chunk to the next
    // allocation of its size (`VirtualPool::free` marks the chunk free; only `release_chunk` - the
    // restore-failure path - unmaps it). So the figure a budget reads is a bound the pool holds,
    // not a live count of what allocations use, and freeing the last buffer does not lower it.
    assert_eq!(
        allocator.mapped_bytes(PoolKind::Graph),
        mapped_before_free,
        "freeing a buffer changed the storage the pool maps"
    );
    assert!(
        mapped_before_free <= RESERVATION,
        "the pool mapped {mapped_before_free} bytes of a {RESERVATION}-byte reservation"
    );
}

/// Launches `kernel` inside one capture and answers the recorded graph.
///
/// [`Captured`] instantiates its own executable from the template, so the probe hands it the
/// graph rather than an executable: what a purge drops and what a live generation replays must be
/// the same object's two halves.
fn capture_graph(stream: &Stream, kernel: &Function, data: &mut u64, value: &mut u32) -> Graph {
    let mut count = WORDS as i32;
    let mut parameters: [*mut c_void; 3] = [
        (data as *mut u64).cast(),
        (&raw mut count).cast(),
        (value as *mut u32).cast(),
    ];
    let mut recording = Capture::begin(stream, CaptureMode::Relaxed).expect("begin a capture");
    kernel
        .launch_ex(
            stream,
            &LaunchConfig::new([1, 1, 1], [WORDS as u32, 1, 1]),
            &mut parameters,
        )
        .expect("launch inside the capture");
    recording.record().expect("record the launch's node");
    recording.end().expect("end the capture")
}

/// A purge releases a generation's own graphs and keeps one a live generation still serves.
///
/// This is the predicate `Purge` decides on: `CausalLmRunner::purge` releases a bucket's arena plan
/// only where `Graphs::holds` is false, because one plan per bucket is shared by every generation's
/// graph of it. A purge that answered `false` while another generation held the bucket would hand
/// that generation's mapped pages to the next capture - the defect `ee66bb268` fixed - and a purge
/// that dropped the live generation's executable would leave it serving a graph that no longer
/// exists. Both halves are visible here: the surviving generation replays *its own* arithmetic
/// after the purge, and the bucket is free only once the last generation's graph is discarded.
#[test]
#[ignore = "requires a CUDA device"]
fn a_purge_keeps_the_graph_a_live_generation_still_serves() {
    use executor::captured::Captured;
    use executor::graph::Graphs;
    use executor::plan::{Bucket, Variant};

    let device = Device::new(DEVICE).expect("a CUDA device");
    let allowance = Arc::new(Allowance::default());
    let mut allocator = Allocator::new(&device, allowance).expect("an allocator");
    allocator
        .reserve(PoolKind::Graph, RESERVATION)
        .expect("reserve the graph pool");
    let buffer: Buffer = allocator
        .allocate(PoolKind::Graph, BYTES)
        .expect("one activation buffer");
    let data = buffer.address();
    let stream = Stream::new(DEVICE).expect("a stream");
    let image = std::fs::read(cubin()).expect("the compiled probe kernel");
    let library = Library::load(&device, &image).expect("load the probe kernel");
    let serving_kernel = library.kernel("generation_0").expect("generation_0");
    let precopied_kernel = library.kernel("generation_1").expect("generation_1");

    // SAFETY: the fill lies inside the allocation.
    unsafe { memset_d32(data, 0, WORDS, &stream) }.expect("zero the buffer");

    // Two generations' graphs of one bucket, each with its own arithmetic: `generation_0` adds its
    // value once and `generation_1` twice, so a replay says which generation it belongs to.
    let bucket = Bucket::decode(1, Variant::DecodeGreedy);
    let mut address = data;
    let mut serving_value = 1u32;
    let serving = Captured::new(
        capture_graph(&stream, &serving_kernel, &mut address, &mut serving_value),
        Vec::new(),
        Vec::new(),
    )
    .expect("the serving generation's graph");
    let mut precopied_value = 10u32;
    let precopied = Captured::new(
        capture_graph(
            &stream,
            &precopied_kernel,
            &mut address,
            &mut precopied_value,
        ),
        Vec::new(),
        Vec::new(),
    )
    .expect("the precopied generation's graph");

    let mut graphs = Graphs::new();
    graphs.insert(bucket, 1, serving);
    graphs.insert(bucket, 2, precopied);
    assert!(graphs.holds(bucket), "the bucket was captured twice");

    // The serving generation's batch finished last, so its purge runs while generation 2 is still
    // waiting to switch over.
    assert_eq!(
        // The serving generation is 2: generation 1 is what the commit retires.
        graphs.purge(1, 2),
        vec![bucket],
        "the purge did not name the generation it released"
    );
    assert!(
        graphs.holds(bucket),
        "a purge reported the bucket free while generation 2 still serves its graph"
    );
    let live = graphs
        .captured(bucket, 2)
        .expect("the live generation's graph went with the purge");
    // The graph's node holds the *address* of the host parameter it was captured with, and this
    // generation's is `precopied_value = 10`: `generation_1` adds it twice, so a replay of the live
    // generation over the zeroed buffer is 20. A shared or re-pointed node would show the serving
    // generation's 1 (2) or nothing at all.
    assert_eq!(
        replay(&stream, &live.exec(), data),
        vec![20; WORDS],
        "the live generation's graph no longer adds its own value (10) twice"
    );
    assert!(
        graphs.captured(bucket, 1).is_none(),
        "the purged generation's graph is still held"
    );

    // The generation that never switched over is discarded; only now is the bucket free.
    assert_eq!(
        graphs.discard(2),
        vec![bucket],
        "the discard named no bucket"
    );
    assert!(
        !graphs.holds(bucket),
        "the bucket is still held after every generation's graph was released"
    );

    allocator.free(buffer);
}
